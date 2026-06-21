"""RACS catalog loader (CASDA TAP-driven, separate from VizieR)."""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Any

import numpy as np
from astroquery.utils.tap.core import TapPlus

from radiosim.utils.network import require_service

from ...operations.factories import create_empty, create_from_arrays
from ...operations.region import SkyRegion
from ...registry.catalogs import (
    CASDA_TAP_URL,
    RACS_CATALOGS,
)
from ...registry.facade import loader_registry
from .core import (
    _extract_masked_column,
    _extract_text_column,
    _find_id_column,
    _find_name_column,
)
from .provenance import _build_point_catalog_provenance

if TYPE_CHECKING:
    from radiosim.core.precision import PrecisionConfig

logger = logging.getLogger(__name__)

# ADQL bare table/column identifiers: dotted alphanumerics with underscores.
# TAP does not support bound parameters for identifiers, so the only safe way
# to interpolate a table/column name is to validate it against a strict
# allowlist pattern before it can reach the query string.
_ADQL_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)*$")


def _validate_adql_identifier(identifier: str, *, kind: str) -> str:
    """Return *identifier* if it is a safe ADQL table/column name, else raise.

    Guards against ADQL injection through catalog-metadata identifiers: only
    dotted alphanumeric-with-underscore names are accepted (no quotes, spaces,
    parentheses, or statement separators).
    """
    if not isinstance(identifier, str) or not _ADQL_IDENTIFIER_RE.match(identifier):
        raise ValueError(
            f"Unsafe ADQL {kind} identifier {identifier!r}; expected a dotted "
            "alphanumeric/underscore name."
        )
    return identifier


def _adql_number(value: float) -> str:
    """Return a finite numeric *value* formatted as a safe ADQL literal."""
    num = float(value)
    if not np.isfinite(num):
        raise ValueError(f"Non-finite ADQL numeric literal: {value!r}.")
    return repr(num)


def _build_racs_adql(
    *,
    info: Any,
    max_rows: int,
    flux_limit_mjy: float,
    region: SkyRegion | None,
) -> str:
    """Assemble the RACS CASDA TAP query from validated identifiers/literals.

    All table and column names are validated against
    :data:`_ADQL_IDENTIFIER_RE`, and every numeric value is rendered through
    :func:`_adql_number`, so no raw caller- or metadata-supplied string is
    interpolated unchecked.
    """
    ra_col = _validate_adql_identifier(info.ra_col, kind="column")
    dec_col = _validate_adql_identifier(info.dec_col, kind="column")
    flux_col = _validate_adql_identifier(info.flux_col, kind="column")
    tap_table = _validate_adql_identifier(info.tap_table, kind="table")

    top_n = int(max_rows)
    if top_n <= 0:
        raise ValueError(f"max_rows must be a positive integer, got {max_rows!r}.")

    adql = (
        f"SELECT TOP {top_n} "
        f"{ra_col}, {dec_col}, {flux_col} "
        f"FROM {tap_table} "
        f"WHERE {flux_col} >= {_adql_number(flux_limit_mjy)}"
    )
    if region is not None:
        from ...operations.region import ConeRegion

        pt = f"POINT('ICRS', {ra_col}, {dec_col})"
        spatial_parts = []
        for sub in region._iter_atomic():
            if isinstance(sub, ConeRegion):
                spatial_parts.append(
                    f"CONTAINS({pt}, CIRCLE('ICRS', "
                    f"{_adql_number(sub.center.ra.deg)}, "
                    f"{_adql_number(sub.center.dec.deg)}, "
                    f"{_adql_number(sub.radius.deg)})) = 1"
                )
            else:  # box
                spatial_parts.append(
                    f"CONTAINS({pt}, BOX('ICRS', "
                    f"{_adql_number(sub.center.ra.deg)}, "
                    f"{_adql_number(sub.center.dec.deg)}, "
                    f"{_adql_number(sub.width.deg)}, "
                    f"{_adql_number(sub.height.deg)})) = 1"
                )
        adql += " AND (" + " OR ".join(spatial_parts) + ")"
    return adql


def _parse_racs_results_with_fallback(
    result: Any,
    info: Any,
    flux_limit: float,
    *,
    band: str,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray | None,
    np.ndarray | None,
]:
    """Extract ``(ra, dec, flux_jy, source_name, source_id)`` from a RACS TAP result.

    Tries vectorized extraction first.  On failure (e.g. unexpected null
    patterns or a column the masked-array path can't handle), falls back to
    a row-by-row parser that skips bad rows individually.  Both paths use
    the same ``info.flux_unit_conversion_factor`` to convert native flux
    units (mJy) to Jy.
    """
    factor = info.flux_unit_conversion_factor
    try:
        flux_raw = _extract_masked_column(result, info.flux_col) * factor
        ra_raw = _extract_masked_column(result, info.ra_col)
        dec_raw = _extract_masked_column(result, info.dec_col)
        valid = (
            np.isfinite(flux_raw)
            & np.isfinite(ra_raw)
            & np.isfinite(dec_raw)
            & (flux_raw >= flux_limit)
        )
        ra_arr = ra_raw[valid]
        dec_arr = dec_raw[valid]
        flux_arr = flux_raw[valid]
        id_col = _find_id_column(result)
        name_col = _find_name_column(result)
        source_id = (
            _extract_text_column(result, id_col)[valid] if id_col is not None else None
        )
        source_name = (
            _extract_text_column(result, name_col)[valid]
            if name_col is not None
            else None
        )
        if source_id is None and source_name is not None:
            source_id = source_name.copy()
        return ra_arr, dec_arr, flux_arr, source_name, source_id
    except (TypeError, AttributeError, NameError):
        # Programming errors in the fast path must not be silently swallowed by
        # the row-by-row fallback — they indicate a logic bug, not malformed
        # data, so let them surface.
        raise
    except Exception as e:
        logger.warning(
            f"Vectorized extraction failed for RACS-{band}, "
            f"falling back to row-by-row: {e}"
        )

    ra_list, dec_list, flux_list = [], [], []
    source_name_list: list[str] = []
    source_id_list: list[str] = []
    id_col = _find_id_column(result)
    name_col = _find_name_column(result)
    n_rows_skipped = 0
    for row in result:
        try:
            flux_native = row[info.flux_col]
            if np.ma.is_masked(flux_native) or not np.isfinite(float(flux_native)):
                continue
            flux_jy = float(flux_native) * factor
            if flux_jy < flux_limit:
                continue
            ra_val = row[info.ra_col]
            dec_val = row[info.dec_col]
            if np.ma.is_masked(ra_val) or np.ma.is_masked(dec_val):
                continue
            ra_list.append(float(ra_val))
            dec_list.append(float(dec_val))
            flux_list.append(flux_jy)
            if name_col is not None:
                source_name_list.append(
                    "" if np.ma.is_masked(row[name_col]) else str(row[name_col])
                )
            if id_col is not None:
                source_id_list.append(
                    "" if np.ma.is_masked(row[id_col]) else str(row[id_col])
                )
        except Exception as row_err:
            n_rows_skipped += 1
            logger.debug(f"Skipping RACS row: {row_err}")
            continue
    if n_rows_skipped:
        total = len(result)
        level = (
            logging.WARNING if n_rows_skipped > 0.1 * max(total, 1) else logging.INFO
        )
        logger.log(
            level,
            "RACS-%s row-by-row parse skipped %d/%d rows. A large fraction "
            "usually means a systematic schema/column issue, not isolated "
            "bad rows.",
            band,
            n_rows_skipped,
            total,
        )
    ra_arr = np.array(ra_list, dtype=np.float64)
    dec_arr = np.array(dec_list, dtype=np.float64)
    flux_arr = np.array(flux_list, dtype=np.float64)
    source_name = np.array(source_name_list, dtype=str) if source_name_list else None
    source_id = np.array(source_id_list, dtype=str) if source_id_list else None
    if source_id is None and source_name is not None:
        source_id = source_name.copy()
    return ra_arr, dec_arr, flux_arr, source_name, source_id


@loader_registry.register_loader(
    "racs",
    config_section="racs",
    use_flag="use_racs",
    network_service="casda",
    aliases={
        "racs_low": {"band": "low"},
        "racs_mid": {"band": "mid"},
        "racs_high": {"band": "high"},
    },
    config_fields={
        "flux_limit": "flux_limit",
        "band": "band",
        "max_rows": "max_rows",
    },
)
def load_racs(
    band: str = "low",
    flux_limit: float = 1.0,
    max_rows: int = 1_000_000,
    brightness_conversion: str = "planck",
    *,
    precision: PrecisionConfig,
    region: SkyRegion | None = None,
) -> SkyModel:  # noqa: F821
    """
    Load a RACS catalog via CASDA TAP (887.5 / 1367.5 / 1655.5 MHz).

    RACS (McConnell et al. 2020) is the Rapid ASKAP Continuum Survey.
    Data are retrieved via CASDA TAP (astroquery). The column names used
    here are best-effort -- verify against the live CASDA schema if errors
    occur.

    Parameters
    ----------
    band : str, default="low"
        Survey band: "low" (887.5 MHz), "mid" (1367.5 MHz), or
        "high" (1655.5 MHz).
    flux_limit : float, default=1.0
        Minimum flux density in Jy. Converted to mJy internally for
        the TAP query.
    max_rows : int, default=1_000_000
        Maximum rows to retrieve (TOP N in ADQL).
    brightness_conversion : str, default="planck"
        Conversion method: "planck" or "rayleigh-jeans".

    Returns
    -------
    SkyModel

    Raises
    ------
    ValueError
        If ``band`` is not "low", "mid", or "high".
    """
    from ...containers.model import SkyModel

    band = band.lower()
    if band not in RACS_CATALOGS:
        raise ValueError(
            f"Unknown RACS band '{band}'. Available: {sorted(RACS_CATALOGS.keys())}"
        )

    info = RACS_CATALOGS[band]
    # Convert Jy threshold to catalog units via the table-driven factor.
    flux_limit_native = flux_limit / info.flux_unit_conversion_factor
    flux_limit_mjy = flux_limit_native  # named for clarity (RACS catalogs are mJy)
    model_name = f"racs_{band}"

    logger.info(f"Fetching {info.description} via CASDA TAP")

    require_service("casda", f"download RACS-{band} catalog from CASDA")

    adql = _build_racs_adql(
        info=info,
        max_rows=max_rows,
        flux_limit_mjy=flux_limit_mjy,
        region=region,
    )

    try:
        tap = TapPlus(url=CASDA_TAP_URL)
        job = tap.launch_job(adql)
        result = job.get_results()

    except (ConnectionError, TypeError, AttributeError, NameError):
        # Network errors propagate as-is; programming errors are not relabeled
        # as a CASDA schema change.
        raise
    except Exception as e:
        raise RuntimeError(
            f"Failed to fetch RACS-{band} from CASDA TAP: {e}\n"
            f"If this is a network issue, check your connection. "
            f"If the error persists, the CASDA schema may have changed -- "
            f"please report at https://github.com/RRI-interferometry/RadioSim/issues"
        ) from e

    freq_hz = info.freq_mhz * 1e6

    ra_arr, dec_arr, flux_arr, source_name, source_id = (
        _parse_racs_results_with_fallback(result, info, flux_limit, band=band)
    )

    # Client-side region trim
    if region is not None and len(flux_arr) > 0:
        in_region = region.contains(np.deg2rad(ra_arr), np.deg2rad(dec_arr))
        ra_arr = ra_arr[in_region]
        dec_arr = dec_arr[in_region]
        flux_arr = flux_arr[in_region]
        if source_name is not None:
            source_name = source_name[in_region]
        if source_id is not None:
            source_id = source_id[in_region]

    n = len(flux_arr)
    logger.info(
        f"RACS-{band.upper()} loaded: {n:,} sources "
        f"(flux >= {flux_limit} Jy, freq={info.freq_mhz} MHz)"
    )

    if n == 0:
        provenance = _build_point_catalog_provenance(
            info=info,
            flux_limit_jy=flux_limit,
            flux_jy=None,
            catalog_key=f"racs_{band}",
            region=region,
        )
        return create_empty(
            model_name,
            brightness_conversion,
            precision=precision,
            reference_frequency=freq_hz,
            provenance=provenance,
        )

    sky = create_from_arrays(
        ra_rad=SkyModel.deg_to_rad_at_precision(ra_arr, precision),
        dec_rad=SkyModel.deg_to_rad_at_precision(dec_arr, precision),
        flux=flux_arr,
        ref_freq=np.full(n, freq_hz, dtype=np.float64),
        source_name=source_name,
        source_id=source_id,
        model_name=model_name,
        reference_frequency=freq_hz,
        brightness_conversion=brightness_conversion,
        precision=precision,
    )
    provenance = _build_point_catalog_provenance(
        info=info,
        flux_limit_jy=flux_limit,
        flux_jy=flux_arr,
        catalog_key=f"racs_{band}",
        region=region,
    )
    return sky.replace(provenance=provenance)


# =========================================================================
# Listing helpers (module-level functions)
# =========================================================================
