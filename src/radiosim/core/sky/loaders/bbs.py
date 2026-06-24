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
from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from ..registry import loader_registry

if TYPE_CHECKING:
    from radiosim.core.precision import PrecisionConfig

    from ..containers import SkyProvenance
    from ..containers.model import SkyModel
    from ..operations.region import SkyRegion

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


def _classify_dec_format(dec_str: str) -> str:
    """Explicitly classify a BBS Dec string's coordinate format.

    Returns one of ``"deg"``, ``"rad"``, ``"dms"`` (``d``/``m``/``s``
    letter-delimited sexagesimal), ``"dotted"`` (``dd.mm.ss[.sss]``
    dot-delimited sexagesimal), or ``"decimal"`` (plain decimal degrees).

    This replaces the previous dot-counting heuristic with an explicit
    rule set so the decimal-vs-sexagesimal decision is unambiguous:

    * a ``deg``/``rad`` suffix wins;
    * a non-numeric ``d`` letter (e.g. ``+48d13m02``) is ``dms``;
    * a value whose numeric body contains two or more ``.`` separators
      (``dd.mm.ss`` or ``dd.mm.ss.sss``) is dotted sexagesimal;
    * anything else (at most one ``.``) is plain decimal.
    """
    if dec_str.endswith("deg"):
        return "deg"
    if dec_str.endswith("rad"):
        return "rad"
    body = dec_str.lstrip("+-")
    has_letter_d = "d" in dec_str.lower() and not body.replace(".", "").isdigit()
    if has_letter_d:
        return "dms"
    if body.count(".") >= 2:
        return "dotted"
    return "decimal"


def _parse_bbs_dec(dec_str: str) -> float:
    """Parse BBS Dec string to degrees.

    Accepts:
      - Sexagesimal with dots: ``+48.13.02.25`` (degrees)
      - Sexagesimal with d/m/s: ``+48d13m02.25``
      - Decimal with suffix: ``-30.5deg``, ``-0.53rad``
      - Plain decimal degrees: ``-30.25``

    The decimal-vs-sexagesimal decision is made by explicit format
    detection (:func:`_classify_dec_format`), not by counting dots.
    """
    dec_str = dec_str.strip()
    fmt = _classify_dec_format(dec_str)

    if fmt == "deg":
        return float(dec_str[:-3])
    if fmt == "rad":
        return np.degrees(float(dec_str[:-3]))
    if fmt == "dms":
        parts = re.split(r"[dDmMsS]", dec_str)
        parts = [p for p in parts if p.strip()]
        d = float(parts[0])
        m = float(parts[1]) if len(parts) > 1 else 0.0
        s = float(parts[2]) if len(parts) > 2 else 0.0
        sign = -1 if d < 0 or dec_str.startswith("-") else 1
        return sign * (abs(d) + m / 60.0 + s / 3600.0)
    if fmt == "dotted":
        # Sexagesimal: +48.13.02.25 or -48.13.02
        sign = -1 if dec_str.startswith("-") else 1
        parts = dec_str.lstrip("+-").split(".")
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


_FIXED_FORMAT_COLUMNS = [
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


def _tokenize_data_line(line: str) -> list[str]:
    """Split a BBS data line into its column fields.

    Comma-separated (BBS/WSClean) lines are split on top-level commas only,
    so bracket-delimited arrays such as ``[-0.7,-0.05]`` are kept intact as a
    single field.  Whitespace-separated (legacy fixed-format) lines are split
    on whitespace.
    """
    if "," not in line:
        return line.split()

    fields: list[str] = []
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
    return fields


def _make_field_getter(
    columns: list[str],
    fields: list[str],
    defaults: dict[str, str],
):
    """Build a case-insensitive ``get(col_name, default)`` accessor for a row.

    Resolves a value from the row's ``fields`` by column name, falling back to
    the header ``defaults`` and then the supplied literal default.
    """
    col_lower = [c.lower() for c in columns]

    def get(col_name: str, default: str = "0") -> str:
        cn = col_name.lower()
        if cn in col_lower:
            idx = col_lower.index(cn)
            if idx < len(fields) and fields[idx].strip():
                return fields[idx].strip()
        return defaults.get(cn, default)

    return get


@dataclass(frozen=True)
class _BbsRow:
    """A single parsed BBS component row."""

    ra_deg: float
    dec_deg: float
    flux: float
    spectral_coeffs: list[float]
    stokes_q: float
    stokes_u: float
    stokes_v: float
    rotation_measure: float
    ref_freq: float
    major_arcsec: float
    minor_arcsec: float
    pa_deg: float
    source_name: str


# Skip-reason sentinels returned by ``_parse_bbs_row`` alongside ``None``.
_SKIP_PATCH = "patch"
_SKIP_SHAPELET = "shapelet"
_SKIP_NONPOSITIVE = "nonpositive"


def _parse_bbs_row(
    columns: list[str],
    fields: list[str],
    defaults: dict[str, str],
    *,
    ref_freq_from_header: float,
) -> tuple[_BbsRow | None, str | None]:
    """Parse one tokenized BBS data row.

    Returns ``(row, None)`` for a kept component, or ``(None, reason)`` for a
    skipped one, where *reason* is one of ``_SKIP_PATCH`` (patch definition
    with empty Name and Type), ``_SKIP_SHAPELET`` (unsupported SHAPELET), or
    ``_SKIP_NONPOSITIVE`` (non-positive / non-finite Stokes I).
    """
    get = _make_field_getter(columns, fields, defaults)
    col_lower = [c.lower() for c in columns]

    # Patch definition (empty Name and Type) -> skip.
    if "name" in col_lower and "type" in col_lower:
        name_idx = col_lower.index("name")
        type_idx = col_lower.index("type")
        name_val = fields[name_idx] if name_idx < len(fields) else ""
        type_val = fields[type_idx] if type_idx < len(fields) else ""
        if not name_val.strip() and not type_val.strip():
            return None, _SKIP_PATCH

    src_type = get("type", "POINT").upper()
    if src_type == "SHAPELET":
        logger.warning("Skipping SHAPELET source (not supported)")
        return None, _SKIP_SHAPELET

    stokes_i = float(get("i", "0"))
    if not np.isfinite(stokes_i) or stokes_i <= 0:
        # Non-positive / non-finite Stokes I is dropped (negative CLEAN
        # components are not representable as catalog flux); the caller
        # counts and reports these.
        return None, _SKIP_NONPOSITIVE

    ra_deg = _parse_bbs_ra(get("ra"))
    dec_deg = _parse_bbs_dec(get("dec"))

    stokes_q = float(get("q", "0"))
    stokes_u = float(get("u", "0"))
    stokes_v = float(get("v", "0"))

    # Spectral index (bracket array or single value).
    si_str = get("spectralindex", "[]").strip("[]")
    if si_str:
        si_coeffs = [float(x) for x in si_str.split(",") if x.strip()]
    else:
        si_coeffs = [-0.7]
    if not si_coeffs:
        si_coeffs = [-0.7]

    rm = float(get("rotationmeasure", "0"))

    # Polarization from angle/fraction (only when Q/U not already set).
    pol_angle = float(get("polarizationangle", "0"))
    pol_frac = float(get("polarizedfraction", "0"))
    if stokes_q == 0 and stokes_u == 0 and pol_frac > 0:
        chi0 = np.deg2rad(pol_angle)
        stokes_q = pol_frac * stokes_i * np.cos(2 * chi0)
        stokes_u = pol_frac * stokes_i * np.sin(2 * chi0)

    # Gaussian morphology (with OSKAR-style aliases).
    major = float(get("majoraxis", "0"))
    minor = float(get("minoraxis", "0"))
    orientation = float(get("orientation", "0"))
    if major == 0:
        major = float(get("major_ax", "0"))
    if minor == 0:
        minor = float(get("minor_ax", "0"))
    if orientation == 0:
        orientation = float(get("positionangle", "0"))

    src_ref_freq = float(
        get(
            "referencefrequency",
            str(ref_freq_from_header) if ref_freq_from_header > 0 else "0",
        )
    )

    row = _BbsRow(
        ra_deg=ra_deg,
        dec_deg=dec_deg,
        flux=stokes_i,
        spectral_coeffs=si_coeffs,
        stokes_q=stokes_q,
        stokes_u=stokes_u,
        stokes_v=stokes_v,
        rotation_measure=rm,
        ref_freq=src_ref_freq,
        major_arcsec=major,
        minor_arcsec=minor,
        pa_deg=orientation,
        source_name=get("name", "").strip(),
    )
    return row, None


@dataclass(frozen=True)
class _BbsParsedSources:
    ref_freq_from_header: float
    ra_deg: np.ndarray
    dec_deg: np.ndarray
    flux: np.ndarray
    spectral_index: np.ndarray
    stokes_q: np.ndarray
    stokes_u: np.ndarray
    stokes_v: np.ndarray
    rotation_measure: np.ndarray
    ref_freq: np.ndarray
    major_arcsec: np.ndarray
    minor_arcsec: np.ndarray
    pa_deg: np.ndarray
    spectral_coeffs: list[list[float]]
    source_name: np.ndarray | None
    has_gaussian: bool
    has_spectral_coeffs: bool


# ============================================================================
# BBS Loader and Writer (standalone functions)
# ============================================================================


def _parse_bbs_lines(lines: Iterable[str], *, filename: str) -> _BbsParsedSources:
    columns, defaults = None, {}
    ref_freq_from_header: float = 0.0

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
    name_list: list[str] = []
    has_gaussian = False
    has_spectral_coeffs = False
    n_dropped_nonpositive = 0

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Detect format header
        if columns is None:
            low = line.lower()
            if "= format" in low or low.startswith("format"):
                columns, defaults = _parse_format_header(line)
                ref_freq_from_header = float(defaults.get("referencefrequency", "0"))
                continue
            if line.startswith("#"):
                continue
            # Fixed-format (no header) -- legacy OSKAR 12-column
            columns = list(_FIXED_FORMAT_COLUMNS)

        if line.startswith("#"):
            continue

        fields = _tokenize_data_line(line)
        row, skip_reason = _parse_bbs_row(
            columns,
            fields,
            defaults,
            ref_freq_from_header=ref_freq_from_header,
        )
        if row is None:
            if skip_reason == _SKIP_NONPOSITIVE:
                n_dropped_nonpositive += 1
            continue

        ra_deg_list.append(row.ra_deg)
        dec_deg_list.append(row.dec_deg)
        flux_list.append(row.flux)
        alpha_list.append(row.spectral_coeffs[0])
        sq_list.append(row.stokes_q)
        su_list.append(row.stokes_u)
        sv_list.append(row.stokes_v)
        rm_list.append(row.rotation_measure)
        ref_freq_list.append(row.ref_freq)
        major_list.append(row.major_arcsec)
        minor_list.append(row.minor_arcsec)
        pa_list.append(row.pa_deg)
        name_list.append(row.source_name)

        if row.major_arcsec > 0:
            has_gaussian = True
        if len(row.spectral_coeffs) > 1:
            has_spectral_coeffs = True
        sp_coeffs_list.append(row.spectral_coeffs)

    if n_dropped_nonpositive:
        logger.warning(
            "%s: dropped %d component(s) with non-positive or non-finite "
            "Stokes I (e.g. negative CLEAN components, which are not "
            "representable as catalog flux).",
            filename,
            n_dropped_nonpositive,
        )
    if not flux_list:
        logger.warning(f"No sources found in {filename}")

    return _BbsParsedSources(
        ref_freq_from_header=ref_freq_from_header,
        ra_deg=np.array(ra_deg_list, dtype=np.float64),
        dec_deg=np.array(dec_deg_list, dtype=np.float64),
        flux=np.array(flux_list, dtype=np.float64),
        spectral_index=np.array(alpha_list, dtype=np.float64),
        stokes_q=np.array(sq_list, dtype=np.float64),
        stokes_u=np.array(su_list, dtype=np.float64),
        stokes_v=np.array(sv_list, dtype=np.float64),
        rotation_measure=np.array(rm_list, dtype=np.float64),
        ref_freq=np.array(ref_freq_list, dtype=np.float64),
        major_arcsec=np.array(major_list, dtype=np.float64),
        minor_arcsec=np.array(minor_list, dtype=np.float64),
        pa_deg=np.array(pa_list, dtype=np.float64),
        spectral_coeffs=sp_coeffs_list,
        source_name=np.array(name_list, dtype=str) if name_list else None,
        has_gaussian=has_gaussian,
        has_spectral_coeffs=has_spectral_coeffs,
    )


def _build_bbs_sky(
    *,
    parsed: _BbsParsedSources,
    filename: str,
    flux_limit: float,
    region: SkyRegion | None,
    precision: PrecisionConfig,
    brightness_conversion: str,
    provenance: SkyProvenance | None,
) -> SkyModel:
    n_parsed = len(parsed.flux)
    ra_deg_arr = parsed.ra_deg
    dec_deg_arr = parsed.dec_deg
    flux_arr = parsed.flux
    alpha_arr = parsed.spectral_index
    sq_arr = parsed.stokes_q
    su_arr = parsed.stokes_u
    sv_arr = parsed.stokes_v
    rm_arr = parsed.rotation_measure
    ref_freq_arr = parsed.ref_freq
    major_arr = parsed.major_arcsec
    minor_arr = parsed.minor_arcsec
    pa_arr = parsed.pa_deg
    sp_coeffs_list = parsed.spectral_coeffs
    source_name_arr = parsed.source_name
    has_gaussian = parsed.has_gaussian
    has_spectral_coeffs = parsed.has_spectral_coeffs

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
        if source_name_arr is not None:
            source_name_arr = source_name_arr[mask]

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

    from ..operations.factories import create_from_arrays

    sky = create_from_arrays(
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
        source_name=(
            source_name_arr
            if source_name_arr is not None and np.any(source_name_arr != "")
            else None
        ),
        source_id=(
            source_name_arr
            if source_name_arr is not None and np.any(source_name_arr != "")
            else None
        ),
        model_name=f"bbs:{filename.split('/')[-1]}",
        brightness_conversion=brightness_conversion,
        precision=precision,
    )

    if parsed.ref_freq_from_header > 0:
        # The BBS fluxes are already defined at the header reference frequency;
        # record it as metadata rather than re-anchoring the flux.
        sky = sky.replace(reference_frequency=parsed.ref_freq_from_header)

    if region is not None:
        sky = sky.filter_region(region)

    logger.info(f"Loaded {sky.n_point_sources} sources from BBS file {filename}")
    if provenance is not None:
        sky = sky.replace(provenance=provenance)
    return sky


@loader_registry.register_loader(
    "bbs",
    config_section="bbs",
    use_flag="use_bbs",
    category="file",
    requires_file=True,
    network_service=None,
    config_fields={"filename": "filename", "flux_limit": "flux_limit"},
)
def load_bbs(
    filename: str,
    *,
    flux_limit: float = 0.0,
    region: SkyRegion | None = None,
    precision: PrecisionConfig,
    brightness_conversion: str = "planck",
    provenance: SkyProvenance | None = None,
) -> SkyModel:
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
    precision : PrecisionConfig
        Precision configuration.
    brightness_conversion : str, default ``"planck"``
        Brightness conversion method: ``"planck"`` or ``"rayleigh-jeans"``.
    """
    try:
        source_file = open(filename)
    except OSError as e:
        raise OSError(f"Could not open BBS sky model file {filename!r}: {e}") from e

    with source_file as f:
        parsed = _parse_bbs_lines(f, filename=filename)

    return _build_bbs_sky(
        parsed=parsed,
        filename=filename,
        flux_limit=flux_limit,
        region=region,
        brightness_conversion=brightness_conversion,
        precision=precision,
        provenance=provenance,
    )


def write_bbs(
    sky_model: SkyModel,
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
    point = sky_model.point
    if point is None or point.is_empty:
        raise ValueError("Cannot write an empty SkyModel to BBS format.")

    ref_freq = reference_frequency_hz or sky_model.reference_frequency or 1e8
    n = point.n_sources
    prefix = sky_model.model_name or "src"

    polarization = point.polarization
    rotation_measure = (
        polarization.rotation_measure if polarization is not None else None
    )
    morphology = point.morphology
    metadata = point.metadata

    has_rm = rotation_measure is not None and np.any(rotation_measure != 0)
    has_gauss = morphology is not None and np.any(morphology.major_arcsec > 0)
    has_pol = np.any(point.stokes_q != 0) or np.any(point.stokes_u != 0)

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

        ra_deg = np.rad2deg(point.ra_rad)
        dec_deg = np.rad2deg(point.dec_rad)
        source_name = metadata.source_name if metadata is not None else None
        source_id = metadata.source_id if metadata is not None else None

        for i in range(n):
            if source_name is not None and source_name[i]:
                name = str(source_name[i])
            elif source_id is not None and source_id[i] is not None:
                name = str(source_id[i])
            else:
                name = f"{prefix}_{i}"
            is_gauss = (
                morphology is not None and has_gauss and morphology.major_arcsec[i] > 0
            )
            src_type = "GAUSSIAN" if is_gauss else "POINT"
            ra_str = _format_ra_bbs(ra_deg[i])
            dec_str = _format_dec_bbs(dec_deg[i])
            flux_i = point.flux[i]

            # Spectral index
            if point.spectral_coeffs is not None:
                si_list = point.spectral_coeffs[i].tolist()
                # Trim trailing zeros
                while len(si_list) > 1 and si_list[-1] == 0:
                    si_list.pop()
                si_str = "[" + ",".join(f"{c}" for c in si_list) + "]"
            else:
                si_str = f"[{point.spectral_index[i]}]"

            parts = [name, src_type, ra_str, dec_str, f"{flux_i}"]

            if has_pol:
                q = point.stokes_q[i]
                u = point.stokes_u[i]
                v = point.stokes_v[i]
                parts.extend([f"{q}", f"{u}", f"{v}"])

            parts.extend(["", si_str, "true"])

            if has_gauss:
                assert morphology is not None
                maj = morphology.major_arcsec[i] if is_gauss else 0.0
                mi = morphology.minor_arcsec[i] if is_gauss else 0.0
                pa = morphology.pa_deg[i] if is_gauss else 0.0
                parts.extend([f"{maj}", f"{mi}", f"{pa}"])

            if has_rm:
                rm_val = rotation_measure[i] if rotation_measure is not None else 0.0
                parts.append(f"{rm_val}")

            f.write(", ".join(parts) + "\n")

    logger.info(f"SkyModel written to BBS format: {filename}")
