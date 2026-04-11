"""BBS/DP3/WSClean sky model format reader and writer.

Supports both header syntaxes:
  BBS:     ``# (Name, Type, Ra, Dec, I, ...) = format``
  WSClean: ``Format = Name, Type, Ra, Dec, I, ...``

Reference:
  - LOFAR wiki: makesourcedb format string
  - OSKAR sky model format documentation
  - WSClean component list documentation
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Any

import numpy as np

from ._registry import register_loader

if TYPE_CHECKING:
    from rrivis.core.precision import PrecisionConfig

    from .region import SkyRegion

logger = logging.getLogger(__name__)

# ============================================================================
# Coordinate parsing helpers
# ============================================================================


def _parse_bbs_ra(ra_str: str) -> float:
    """Parse BBS RA string to degrees.

    Accepts:
      - Sexagesimal with colons: ``08:13:36.06`` (hours)
      - Decimal with suffix: ``123.4deg``, ``2.15rad``
    """
    ra_str = ra_str.strip()
    if ra_str.endswith("deg"):
        return float(ra_str[:-3])
    if ra_str.endswith("rad"):
        return np.degrees(float(ra_str[:-3]))
    if ":" in ra_str:
        parts = ra_str.split(":")
        h = float(parts[0])
        m = float(parts[1]) if len(parts) > 1 else 0.0
        s = float(parts[2]) if len(parts) > 2 else 0.0
        sign = -1 if h < 0 else 1
        return sign * (abs(h) + m / 60.0 + s / 3600.0) * 15.0
    if "h" in ra_str.lower():
        parts = re.split(r"[hHmMsS]", ra_str)
        parts = [p for p in parts if p.strip()]
        h = float(parts[0])
        m = float(parts[1]) if len(parts) > 1 else 0.0
        s = float(parts[2]) if len(parts) > 2 else 0.0
        return (h + m / 60.0 + s / 3600.0) * 15.0
    return float(ra_str)


def _parse_bbs_dec(dec_str: str) -> float:
    """Parse BBS Dec string to degrees.

    Accepts:
      - Sexagesimal with dots: ``+48.13.02.25`` (degrees)
      - Sexagesimal with d/m/s: ``+48d13m02.25``
      - Decimal with suffix: ``-30.5deg``, ``-0.53rad``
    """
    dec_str = dec_str.strip()
    if dec_str.endswith("deg"):
        return float(dec_str[:-3])
    if dec_str.endswith("rad"):
        return np.degrees(float(dec_str[:-3]))
    if (
        "d" in dec_str.lower()
        and not dec_str.replace("-", "").replace("+", "").replace(".", "").isdigit()
    ):
        parts = re.split(r"[dDmMsS]", dec_str)
        parts = [p for p in parts if p.strip()]
        d = float(parts[0])
        m = float(parts[1]) if len(parts) > 1 else 0.0
        s = float(parts[2]) if len(parts) > 2 else 0.0
        sign = -1 if d < 0 or dec_str.strip().startswith("-") else 1
        return sign * (abs(d) + m / 60.0 + s / 3600.0)
    if "." in dec_str:
        # Count dots to distinguish decimal from sexagesimal
        ndots = dec_str.count(".")
        if ndots >= 2:
            # Sexagesimal: +48.13.02.25 or -48.13.02
            sign = -1 if dec_str.startswith("-") else 1
            clean = dec_str.lstrip("+-")
            parts = clean.split(".")
            d = float(parts[0])
            m = float(parts[1]) if len(parts) > 1 else 0.0
            # parts[2:] are seconds (possibly with fractional part)
            if len(parts) > 2:
                sec_str = ".".join(parts[2:])
                s = float(sec_str) if sec_str else 0.0
            else:
                s = 0.0
            return sign * (d + m / 60.0 + s / 3600.0)
    return float(dec_str)


def _format_ra_bbs(ra_deg: float) -> str:
    """Format RA degrees to BBS sexagesimal ``hh:mm:ss.ssssss``."""
    ra_h = ra_deg / 15.0
    sign = -1 if ra_h < 0 else 1
    total_s = abs(ra_h) * 3600.0
    h = int(total_s // 3600)
    total_s -= h * 3600
    m = int(total_s // 60)
    s = total_s - m * 60
    if s >= 59.9999995:
        s = 0.0
        m += 1
    if m >= 60:
        m = 0
        h += 1
    prefix = "-" if sign < 0 else ""
    return f"{prefix}{h:02d}:{m:02d}:{s:09.6f}"


def _format_dec_bbs(dec_deg: float) -> str:
    """Format Dec degrees to BBS sexagesimal ``+dd.mm.ss.sssss``."""
    sign = -1 if dec_deg < 0 else 1
    total_s = abs(dec_deg) * 3600.0
    d = int(total_s // 3600)
    total_s -= d * 3600
    m = int(total_s // 60)
    s = total_s - m * 60
    if s >= 59.999995:
        s = 0.0
        m += 1
    if m >= 60:
        m = 0
        d += 1
    prefix = "-" if sign < 0 else "+"
    return f"{prefix}{d:02d}.{m:02d}.{s:08.5f}"


# ============================================================================
# Format header parsing
# ============================================================================

_DEFAULT_RE = re.compile(r"(\w+)\s*=\s*'([^']*)'")
_DEFAULT_FIXED_RE = re.compile(r"(\w+)\s*=\s*fixed'([^']*)'")


def _parse_format_header(line: str) -> tuple[list[str], dict[str, str]]:
    """Parse BBS or WSClean format header.

    Returns (column_names, defaults_dict).
    """
    line = line.strip()

    # BBS style: # (col1, col2, ...) = format
    m = re.match(r"#?\s*\((.+)\)\s*=\s*format", line, re.IGNORECASE)
    if not m:
        # WSClean style: Format = col1, col2, ...
        m = re.match(r"format\s*=\s*(.+)", line, re.IGNORECASE)
    if not m:
        raise ValueError(f"Cannot parse format header: {line!r}")

    body = m.group(1).strip()

    # Extract defaults like ReferenceFrequency='150e6'
    defaults: dict[str, str] = {}
    for dm in _DEFAULT_RE.finditer(body):
        defaults[dm.group(1).lower()] = dm.group(2)
    for dm in _DEFAULT_FIXED_RE.finditer(body):
        defaults[dm.group(1).lower()] = dm.group(2)

    # Strip default values from column names
    clean = _DEFAULT_RE.sub(lambda x: x.group(1), body)
    clean = _DEFAULT_FIXED_RE.sub(lambda x: x.group(1), clean)

    # Split by comma, strip whitespace
    cols = [c.strip() for c in clean.split(",") if c.strip()]
    return cols, defaults


# ============================================================================
# BBS Loader and Writer (standalone functions)
# ============================================================================


@register_loader(
    "bbs",
    config_section="bbs",
    use_flag="use_bbs",
    requires_file=True,
    network_service=None,
)
def load_bbs(
    filename: str,
    *,
    flux_limit: float = 0.0,
    region: SkyRegion | None = None,
    precision: PrecisionConfig | None = None,
    brightness_conversion: str = "planck",
) -> Any:
    """Load a sky model from BBS/DP3/WSClean format.

    Supports both BBS ``# (...) = format`` and WSClean ``Format = ...``
    header syntax.  POINT and GAUSSIAN source types are supported;
    SHAPELET sources are skipped with a warning.

    Parameters
    ----------
    filename : str
        Path to the sky model file.
    flux_limit : float, default 0.0
        Minimum Stokes I flux in Jy.
    region : SkyRegion, optional
        Spatial filter.
    precision : PrecisionConfig, optional
        Precision configuration.
    brightness_conversion : str, default ``"planck"``
        Brightness conversion method: ``"planck"`` or ``"rayleigh-jeans"``.
    """
    from .model import SkyModel

    columns, defaults = None, {}
    ref_freq_from_header: float = 0.0

    # Accumulate per-source arrays directly (avoids SkyCoord round-trip)
    ra_deg_list: list[float] = []
    dec_deg_list: list[float] = []
    flux_list: list[float] = []
    alpha_list: list[float] = []
    sq_list: list[float] = []
    su_list: list[float] = []
    sv_list: list[float] = []
    rm_list: list[float] = []
    ref_freq_list: list[float] = []
    major_list: list[float] = []
    minor_list: list[float] = []
    pa_list: list[float] = []
    sp_coeffs_list: list[list[float]] = []
    has_gaussian = False
    has_spectral_coeffs = False

    with open(filename) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # Detect format header
            if columns is None:
                low = line.lower()
                if "= format" in low or low.startswith("format"):
                    columns, defaults = _parse_format_header(line)
                    ref_freq_from_header = float(
                        defaults.get("referencefrequency", "0")
                    )
                    continue
                if line.startswith("#"):
                    continue
                # Fixed-format (no header) -- legacy OSKAR 12-column
                columns = [
                    "Ra",
                    "Dec",
                    "I",
                    "Q",
                    "U",
                    "V",
                    "ReferenceFrequency",
                    "SpectralIndex",
                    "RotationMeasure",
                    "MajorAxis",
                    "MinorAxis",
                    "Orientation",
                ]

            if line.startswith("#"):
                continue

            # Build column index (case-insensitive)
            col_lower = [c.lower() for c in columns]

            # Split data line by comma (BBS) or whitespace (fixed-format)
            # Must handle bracket-delimited arrays like [-0.7,-0.05]
            if "," in line:
                fields = []
                current = ""
                depth = 0
                for ch in line:
                    if ch == "[":
                        depth += 1
                    elif ch == "]":
                        depth -= 1
                    if ch == "," and depth == 0:
                        fields.append(current.strip())
                        current = ""
                    else:
                        current += ch
                fields.append(current.strip())
            else:
                fields = line.split()

            # Check for patch definition (empty Name and Type)
            name_idx = col_lower.index("name") if "name" in col_lower else -1
            type_idx = col_lower.index("type") if "type" in col_lower else -1
            if name_idx >= 0 and type_idx >= 0:
                name_val = fields[name_idx] if name_idx < len(fields) else ""
                type_val = fields[type_idx] if type_idx < len(fields) else ""
                if not name_val.strip() and not type_val.strip():
                    continue  # patch definition line -- skip

            # Helper to get field value with default
            def _get(
                col_name: str,
                default: str = "0",
                _cl: list[str] = col_lower,
                _fl: list[str] = fields,
                _df: dict[str, str] = defaults,
            ) -> str:
                cn = col_name.lower()
                if cn in _cl:
                    idx = _cl.index(cn)
                    if idx < len(_fl) and _fl[idx].strip():
                        return _fl[idx].strip()
                return _df.get(cn, default)

            # Source type
            src_type = _get("type", "POINT").upper()
            if src_type == "SHAPELET":
                logger.warning("Skipping SHAPELET source (not supported)")
                continue

            # Coordinates
            ra_deg = _parse_bbs_ra(_get("ra"))
            dec_deg = _parse_bbs_dec(_get("dec"))

            # Stokes I
            stokes_i = float(_get("i", "0"))
            if stokes_i <= 0:
                continue

            # Stokes Q, U, V
            stokes_q = float(_get("q", "0"))
            stokes_u = float(_get("u", "0"))
            stokes_v = float(_get("v", "0"))

            # Spectral index (bracket array or single value)
            si_str = _get("spectralindex", "[]")
            si_str = si_str.strip("[]")
            if si_str:
                si_coeffs = [float(x) for x in si_str.split(",") if x.strip()]
            else:
                si_coeffs = [-0.7]
            alpha = si_coeffs[0] if si_coeffs else -0.7

            # Rotation measure
            rm = float(_get("rotationmeasure", "0"))

            # Polarization from angle/fraction (if Q/U not set)
            pol_angle = float(_get("polarizationangle", "0"))
            pol_frac = float(_get("polarizedfraction", "0"))
            if stokes_q == 0 and stokes_u == 0 and pol_frac > 0:
                chi0 = np.deg2rad(pol_angle)
                stokes_q = pol_frac * stokes_i * np.cos(2 * chi0)
                stokes_u = pol_frac * stokes_i * np.sin(2 * chi0)

            # Gaussian morphology
            major = float(_get("majoraxis", "0"))
            minor = float(_get("minoraxis", "0"))
            orientation = float(_get("orientation", "0"))

            # Also accept OSKAR-style aliases
            if major == 0:
                major = float(_get("major_ax", "0"))
            if minor == 0:
                minor = float(_get("minor_ax", "0"))
            if orientation == 0:
                orientation = float(_get("positionangle", "0"))

            src_ref_freq = float(
                _get(
                    "referencefrequency",
                    str(ref_freq_from_header) if ref_freq_from_header > 0 else "0",
                )
            )

            # Accumulate into parallel lists
            ra_deg_list.append(ra_deg)
            dec_deg_list.append(dec_deg)
            flux_list.append(stokes_i)
            alpha_list.append(alpha)
            sq_list.append(stokes_q)
            su_list.append(stokes_u)
            sv_list.append(stokes_v)
            rm_list.append(rm)
            ref_freq_list.append(src_ref_freq)
            major_list.append(major)
            minor_list.append(minor)
            pa_list.append(orientation)

            if major > 0:
                has_gaussian = True

            if len(si_coeffs) > 1:
                has_spectral_coeffs = True
            sp_coeffs_list.append(si_coeffs)

    n_parsed = len(flux_list)
    if n_parsed == 0:
        logger.warning(f"No sources found in {filename}")

    # Convert all accumulation lists to numpy arrays
    ra_deg_arr = np.array(ra_deg_list, dtype=np.float64)
    dec_deg_arr = np.array(dec_deg_list, dtype=np.float64)
    flux_arr = np.array(flux_list, dtype=np.float64)
    alpha_arr = np.array(alpha_list, dtype=np.float64)
    sq_arr = np.array(sq_list, dtype=np.float64)
    su_arr = np.array(su_list, dtype=np.float64)
    sv_arr = np.array(sv_list, dtype=np.float64)
    rm_arr = np.array(rm_list, dtype=np.float64)
    ref_freq_arr = np.array(ref_freq_list, dtype=np.float64)
    major_arr = np.array(major_list, dtype=np.float64)
    minor_arr = np.array(minor_list, dtype=np.float64)
    pa_arr = np.array(pa_list, dtype=np.float64)

    # Apply flux limit as a vectorized mask
    if flux_limit > 0 and n_parsed > 0:
        mask = flux_arr >= flux_limit
        ra_deg_arr = ra_deg_arr[mask]
        dec_deg_arr = dec_deg_arr[mask]
        flux_arr = flux_arr[mask]
        alpha_arr = alpha_arr[mask]
        sq_arr = sq_arr[mask]
        su_arr = su_arr[mask]
        sv_arr = sv_arr[mask]
        rm_arr = rm_arr[mask]
        ref_freq_arr = ref_freq_arr[mask]
        major_arr = major_arr[mask]
        minor_arr = minor_arr[mask]
        pa_arr = pa_arr[mask]
        sp_coeffs_list = [sp_coeffs_list[i] for i in np.flatnonzero(mask)]

    # Build optional array fields (set to None if all zeros / not needed)
    rm_arr = rm_arr if np.any(rm_arr != 0) else None
    ref_freq_arr = ref_freq_arr if np.any(ref_freq_arr > 0) else None
    major_arr = major_arr if has_gaussian else None
    minor_arr = minor_arr if has_gaussian else None
    pa_arr = pa_arr if has_gaussian else None

    sp_coeffs_arr = None
    if has_spectral_coeffs and sp_coeffs_list:
        max_terms = max(len(c) for c in sp_coeffs_list)
        if max_terms > 0:
            n = len(sp_coeffs_list)
            sp_coeffs_arr = np.zeros((n, max_terms), dtype=np.float64)
            for i, coeffs in enumerate(sp_coeffs_list):
                sp_coeffs_arr[i, : len(coeffs)] = coeffs

    sky = SkyModel.from_arrays(
        ra_rad=np.deg2rad(ra_deg_arr),
        dec_rad=np.deg2rad(dec_deg_arr),
        flux=flux_arr,
        spectral_index=alpha_arr,
        stokes_q=sq_arr,
        stokes_u=su_arr,
        stokes_v=sv_arr,
        rotation_measure=rm_arr,
        major_arcsec=major_arr,
        minor_arcsec=minor_arr,
        pa_deg=pa_arr,
        spectral_coeffs=sp_coeffs_arr,
        ref_freq=ref_freq_arr,
        model_name=f"bbs:{filename.split('/')[-1]}",
        brightness_conversion=brightness_conversion,
        precision=precision,
    )

    # Apply reference frequency from file header
    if ref_freq_from_header > 0:
        sky = sky.with_reference_frequency(ref_freq_from_header)

    if region is not None:
        sky = sky.filter_region(region)

    logger.info(f"Loaded {sky.n_point_sources} sources from BBS file {filename}")
    return sky


def write_bbs(
    sky_model: SkyModel,  # noqa: F821
    filename: str,
    *,
    reference_frequency_hz: float | None = None,
) -> None:
    """Write a SkyModel to BBS/DP3 format.

    Parameters
    ----------
    sky_model : SkyModel
        The sky model to write.
    filename : str
        Output file path (typically ``*.skymodel``).
    reference_frequency_hz : float, optional
        Override reference frequency. Defaults to ``sky_model.reference_frequency``.
    """
    if sky_model.ra_rad is None or len(sky_model.ra_rad) == 0:
        raise ValueError("Cannot write an empty SkyModel to BBS format.")

    ref_freq = reference_frequency_hz or sky_model.reference_frequency or 1e8
    n = len(sky_model.ra_rad)
    prefix = sky_model.model_name or "src"

    has_rm = sky_model.rotation_measure is not None and np.any(
        sky_model.rotation_measure != 0
    )
    has_gauss = sky_model.major_arcsec is not None and np.any(
        sky_model.major_arcsec > 0
    )
    has_pol = (sky_model.stokes_q is not None and np.any(sky_model.stokes_q != 0)) or (
        sky_model.stokes_u is not None and np.any(sky_model.stokes_u != 0)
    )

    # Build format header
    cols = ["Name", "Type", "Ra", "Dec", "I"]
    if has_pol:
        cols.extend(["Q", "U", "V"])
    cols.extend(
        [
            f"ReferenceFrequency='{ref_freq}'",
            "SpectralIndex='[]'",
            "LogarithmicSI='true'",
        ]
    )
    if has_gauss:
        cols.extend(["MajorAxis", "MinorAxis", "Orientation"])
    if has_rm:
        cols.append("RotationMeasure")

    header = f"# ({', '.join(cols)}) = format\n"

    with open(filename, "w") as f:
        f.write(header)

        ra_deg = np.rad2deg(sky_model.ra_rad)
        dec_deg = np.rad2deg(sky_model.dec_rad)

        for i in range(n):
            name = f"{prefix}_{i}"
            is_gauss = has_gauss and sky_model.major_arcsec[i] > 0
            src_type = "GAUSSIAN" if is_gauss else "POINT"
            ra_str = _format_ra_bbs(ra_deg[i])
            dec_str = _format_dec_bbs(dec_deg[i])
            flux_i = sky_model.flux[i]

            # Spectral index
            if sky_model.spectral_coeffs is not None:
                si_list = sky_model.spectral_coeffs[i].tolist()
                # Trim trailing zeros
                while len(si_list) > 1 and si_list[-1] == 0:
                    si_list.pop()
                si_str = "[" + ",".join(f"{c}" for c in si_list) + "]"
            else:
                si_str = f"[{sky_model.spectral_index[i]}]"

            parts = [name, src_type, ra_str, dec_str, f"{flux_i}"]

            if has_pol:
                q = sky_model.stokes_q[i] if sky_model.stokes_q is not None else 0.0
                u = sky_model.stokes_u[i] if sky_model.stokes_u is not None else 0.0
                v = sky_model.stokes_v[i] if sky_model.stokes_v is not None else 0.0
                parts.extend([f"{q}", f"{u}", f"{v}"])

            parts.extend(["", si_str, "true"])

            if has_gauss:
                maj = sky_model.major_arcsec[i] if is_gauss else 0.0
                mi = sky_model.minor_arcsec[i] if is_gauss else 0.0
                pa = sky_model.pa_deg[i] if is_gauss else 0.0
                parts.extend([f"{maj}", f"{mi}", f"{pa}"])

            if has_rm:
                rm_val = (
                    sky_model.rotation_measure[i]
                    if sky_model.rotation_measure is not None
                    else 0.0
                )
                parts.append(f"{rm_val}")

            f.write(", ".join(parts) + "\n")

    logger.info(f"SkyModel written to BBS format: {filename}")
