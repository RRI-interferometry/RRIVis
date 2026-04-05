"""BBS/DP3/WSClean sky model format reader and writer.

Supports both header syntaxes:
  BBS:     ``# (Name, Type, Ra, Dec, I, ...) = format``
  WSClean: ``Format = Name, Type, Ra, Dec, I, ...``

Reference:
  - LOFAR wiki: makesourcedb format string
  - OSKAR sky model format documentation
  - WSClean component list documentation
"""

import logging
import re
from typing import TYPE_CHECKING, Any

import numpy as np
from astropy.coordinates import SkyCoord

if TYPE_CHECKING:
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
# BBS Loader Mixin
# ============================================================================


class _BBSLoadersMixin:
    """Mixin providing BBS/DP3/WSClean sky model I/O."""

    @classmethod
    def from_bbs(
        cls,
        filename: str,
        *,
        flux_limit: float = 0.0,
        region: "SkyRegion | None" = None,
        precision: Any = None,
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
        """
        columns, defaults = None, {}
        sources: list[dict[str, Any]] = []

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
                        continue
                    if line.startswith("#"):
                        continue
                    # Fixed-format (no header) — legacy OSKAR 12-column
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
                        continue  # patch definition line — skip

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

                src: dict[str, Any] = {
                    "coords": SkyCoord(
                        ra=ra_deg, dec=dec_deg, unit="deg", frame="icrs"
                    ),
                    "flux": stokes_i,
                    "spectral_index": alpha,
                    "stokes_q": stokes_q,
                    "stokes_u": stokes_u,
                    "stokes_v": stokes_v,
                    "rotation_measure": rm,
                }

                if major > 0:
                    src["major_arcsec"] = major
                    src["minor_arcsec"] = minor
                    src["pa_deg"] = orientation

                if len(si_coeffs) > 1:
                    src["spectral_coeffs"] = si_coeffs

                sources.append(src)

        if not sources:
            logger.warning(f"No sources found in {filename}")

        # Apply flux limit
        if flux_limit > 0:
            sources = [s for s in sources if s["flux"] >= flux_limit]

        sky = cls.from_point_sources(
            sources,
            model_name=f"bbs:{filename.split('/')[-1]}",
            precision=precision,
        )

        # Set reference frequency from file (use first source's ref_freq)
        if sources:
            _rf = float(_get("referencefrequency", "0")) if columns else 0.0
            if _rf > 0:
                sky.frequency = _rf

        if region is not None:
            sky = sky.filter_region(region)

        logger.info(f"Loaded {sky.n_sources} sources from BBS file {filename}")
        return sky

    def to_bbs(
        self,
        filename: str,
        *,
        reference_frequency_hz: float | None = None,
    ) -> None:
        """Write this SkyModel to BBS/DP3 format.

        Parameters
        ----------
        filename : str
            Output file path (typically ``*.skymodel``).
        reference_frequency_hz : float, optional
            Override reference frequency. Defaults to ``self.frequency``.
        """
        if self._ra_rad is None or len(self._ra_rad) == 0:
            raise ValueError("Cannot write an empty SkyModel to BBS format.")

        ref_freq = reference_frequency_hz or self.frequency or 1e8
        n = len(self._ra_rad)
        prefix = self.model_name or "src"

        has_rm = self._rotation_measure is not None and np.any(
            self._rotation_measure != 0
        )
        has_gauss = self._major_arcsec is not None and np.any(self._major_arcsec > 0)
        has_pol = (self._stokes_q is not None and np.any(self._stokes_q != 0)) or (
            self._stokes_u is not None and np.any(self._stokes_u != 0)
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

            ra_deg = np.rad2deg(self._ra_rad)
            dec_deg = np.rad2deg(self._dec_rad)

            for i in range(n):
                name = f"{prefix}_{i}"
                is_gauss = has_gauss and self._major_arcsec[i] > 0
                src_type = "GAUSSIAN" if is_gauss else "POINT"
                ra_str = _format_ra_bbs(ra_deg[i])
                dec_str = _format_dec_bbs(dec_deg[i])
                flux_i = self._flux_ref[i]

                # Spectral index
                if self._spectral_coeffs is not None:
                    si_list = self._spectral_coeffs[i].tolist()
                    # Trim trailing zeros
                    while len(si_list) > 1 and si_list[-1] == 0:
                        si_list.pop()
                    si_str = "[" + ",".join(f"{c}" for c in si_list) + "]"
                else:
                    si_str = f"[{self._alpha[i]}]"

                parts = [name, src_type, ra_str, dec_str, f"{flux_i}"]

                if has_pol:
                    q = self._stokes_q[i] if self._stokes_q is not None else 0.0
                    u = self._stokes_u[i] if self._stokes_u is not None else 0.0
                    v = self._stokes_v[i] if self._stokes_v is not None else 0.0
                    parts.extend([f"{q}", f"{u}", f"{v}"])

                parts.extend(["", si_str, "true"])

                if has_gauss:
                    maj = self._major_arcsec[i] if is_gauss else 0.0
                    mi = self._minor_arcsec[i] if is_gauss else 0.0
                    pa = self._pa_deg[i] if is_gauss else 0.0
                    parts.extend([f"{maj}", f"{mi}", f"{pa}"])

                if has_rm:
                    rm_val = (
                        self._rotation_measure[i]
                        if self._rotation_measure is not None
                        else 0.0
                    )
                    parts.append(f"{rm_val}")

                f.write(", ".join(parts) + "\n")

        logger.info(f"SkyModel written to BBS format: {filename}")
