"""Pure TAP point-catalog result parsing (no network I/O)."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from .core import (
    _extract_masked_column,
    _extract_text_column,
    _find_id_column,
    _find_name_column,
)

logger = logging.getLogger(__name__)


def parse_tap_point_catalog_results(
    result: Any,
    info: Any,
    flux_limit: float,
    *,
    catalog_label: str,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray | None,
    np.ndarray | None,
]:
    """Extract ``(ra, dec, flux_jy, source_name, source_id)`` from a TAP table.

    Tries vectorized extraction first. On failure (unexpected null patterns or
    a column the masked-array path cannot handle), falls back to a row-by-row
    parser that skips bad rows individually. Both paths apply
    ``info.flux_unit_conversion_factor`` to convert catalog-native flux to Jy.
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
        raise
    except Exception as e:
        logger.warning(
            f"Vectorized extraction failed for {catalog_label}, "
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
            logger.debug(f"Skipping {catalog_label} row: {row_err}")
            continue
    if n_rows_skipped:
        total = len(result)
        level = (
            logging.WARNING if n_rows_skipped > 0.1 * max(total, 1) else logging.INFO
        )
        logger.log(
            level,
            "%s row-by-row parse skipped %d/%d rows. A large fraction "
            "usually means a systematic schema/column issue, not isolated "
            "bad rows.",
            catalog_label,
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
