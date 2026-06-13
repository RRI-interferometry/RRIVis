"""Inspection helpers: list catalogs and their metadata/column maps."""

from __future__ import annotations

import functools
import logging
from typing import Any

from astroquery.utils.tap.core import TapPlus
from astroquery.vizier import Vizier

from radiosim.utils.network import require_service

from ...registry.catalogs import CASDA_TAP_URL, RACS_CATALOGS, VIZIER_POINT_CATALOGS

logger = logging.getLogger(__name__)


def list_point_catalogs() -> dict[str, str]:
    """List available VizieR point-source catalogs with their descriptions.

    Returns
    -------
    dict[str, str]
        Mapping of catalog key to description string.

    Examples
    --------
    >>> for name, desc in list_point_catalogs().items():
    ...     print(f"{name}: {desc[:80]}...")
    """
    return {name: info.description for name, info in VIZIER_POINT_CATALOGS.items()}


def list_racs_catalogs() -> dict[str, str]:
    """List available RACS (Rapid ASKAP Continuum Survey) bands with their descriptions.

    Returns
    -------
    dict[str, str]
        Mapping of band name to description string.

    Examples
    --------
    >>> for name, desc in list_racs_catalogs().items():
    ...     print(f"{name}: {desc}")
    """
    return {name: info.description for name, info in RACS_CATALOGS.items()}


def get_point_catalog_metadata(catalog_key: str) -> dict[str, Any]:
    """Get static metadata for a VizieR point-source catalog (no network).

    Returns all locally-stored metadata for the catalog without
    making any network calls. For live column queries from VizieR,
    use ``get_catalog_columns()`` instead.

    Parameters
    ----------
    catalog_key : str
        Key in ``VIZIER_POINT_CATALOGS`` (e.g. ``"gleam_egc"``, ``"nvss"``).

    Returns
    -------
    dict
        Keys:

        - ``"vizier_id"`` : str -- VizieR catalog identifier
        - ``"description"`` : str -- catalog description
        - ``"freq_mhz"`` : float -- survey reference frequency
        - ``"flux_col"`` : str -- flux column name
        - ``"flux_unit"`` : str -- unit of the flux column
        - ``"ra_col"`` : str -- RA column name
        - ``"dec_col"`` : str -- Dec column name
        - ``"spindex_col"`` : str or None -- spectral index column
        - ``"default_spindex"`` : float -- default spectral index
        - ``"coord_frame"`` : str -- coordinate frame (e.g. ``"icrs"``, ``"fk4"``)

    Raises
    ------
    ValueError
        If ``catalog_key`` is not recognized.

    Examples
    --------
    >>> info = get_point_catalog_metadata("nvss")
    >>> print(info["freq_mhz"])
    1400.0
    >>> print(info["flux_col"], info["flux_unit"])
    S1.4 mJy
    """
    if catalog_key not in VIZIER_POINT_CATALOGS:
        raise ValueError(
            f"Unknown catalog key '{catalog_key}'. "
            f"Available: {sorted(VIZIER_POINT_CATALOGS.keys())}"
        )
    info = VIZIER_POINT_CATALOGS[catalog_key]
    return {
        "vizier_id": info.vizier_id,
        "description": info.description,
        "freq_mhz": info.freq_mhz,
        "flux_col": info.flux_col,
        "flux_unit": info.flux_unit,
        "ra_col": info.ra_col,
        "dec_col": info.dec_col,
        "spindex_col": info.spindex_col,
        "default_spindex": info.default_spindex,
        "coord_frame": info.coord_frame,
    }


def get_racs_metadata(band: str) -> dict[str, Any]:
    """Get static metadata for a RACS catalog band (no network).

    Returns all locally-stored metadata for the RACS band without
    making any network calls. For live column queries from CASDA TAP,
    use ``get_racs_columns()`` instead.

    Parameters
    ----------
    band : str
        RACS band: ``"low"``, ``"mid"``, or ``"high"``.

    Returns
    -------
    dict
        Keys:

        - ``"description"`` : str -- band description
        - ``"freq_mhz"`` : float -- survey frequency
        - ``"tap_table"`` : str -- CASDA TAP table name
        - ``"ra_col"`` : str -- RA column name
        - ``"dec_col"`` : str -- Dec column name
        - ``"flux_col"`` : str -- flux column name
        - ``"flux_unit"`` : str -- unit of the flux column

    Raises
    ------
    ValueError
        If ``band`` is not recognized.

    Examples
    --------
    >>> info = get_racs_metadata("low")
    >>> print(info["freq_mhz"])
    887.5
    """
    band = band.lower()
    if band not in RACS_CATALOGS:
        raise ValueError(
            f"Unknown RACS band '{band}'. Available: {sorted(RACS_CATALOGS.keys())}"
        )
    info = RACS_CATALOGS[band]
    return {
        "description": info.description,
        "freq_mhz": info.freq_mhz,
        "tap_table": info.tap_table,
        "ra_col": info.ra_col,
        "dec_col": info.dec_col,
        "flux_col": info.flux_col,
        "flux_unit": info.flux_unit,
    }


@functools.lru_cache(maxsize=32)
def get_catalog_columns(catalog_key: str) -> dict[str, Any]:
    """Query VizieR for all available columns in a point-source catalog.

    Fetches one row from VizieR and returns the full list of column names
    along with metadata about which columns RadioSim uses. Column
    descriptions and units are extracted from VizieR's own metadata.

    Parameters
    ----------
    catalog_key : str
        Key in ``VIZIER_POINT_CATALOGS`` (e.g. ``"gleam_egc"``, ``"nvss"``).

    Returns
    -------
    dict
        Keys:

        - ``"columns"`` : list[str] -- all column names in the catalog
        - ``"column_details"`` : dict[str, dict] -- per-column metadata
          from VizieR with ``"description"`` and ``"unit"`` keys
        - ``"used_by_radiosim"`` : dict[str, str | None] -- columns used by
          RadioSim (``"ra"``, ``"dec"``, ``"flux"``, ``"spectral_index"``)
        - ``"vizier_id"`` : str -- VizieR catalog identifier
        - ``"freq_mhz"`` : float -- survey reference frequency
        - ``"flux_unit"`` : str -- unit of the flux column
        - ``"description"`` : str -- catalog description

    Raises
    ------
    RuntimeError
        If the VizieR query fails (network error or schema drift). The
        failure is not cached, so a later call retries cleanly.

    Examples
    --------
    >>> info = get_catalog_columns("nvss")
    >>> print(info["columns"][:5])
    ['recno', 'Field', 'Xpos', 'Ypos', 'NVSS']
    >>> print(info["used_by_radiosim"])
    {'ra': 'RAJ2000', 'dec': 'DEJ2000', 'flux': 'S1.4', 'spectral_index': None}
    >>> print(info["column_details"]["S1.4"])
    {'description': '...', 'unit': 'mJy'}
    """
    if catalog_key not in VIZIER_POINT_CATALOGS:
        raise ValueError(
            f"Unknown catalog key '{catalog_key}'. "
            f"Available: {sorted(VIZIER_POINT_CATALOGS.keys())}"
        )

    info = VIZIER_POINT_CATALOGS[catalog_key]

    require_service("vizier", f"query live columns for '{catalog_key}' from VizieR")

    try:
        v = Vizier(columns=["**"], row_limit=1)
        tables = v.get_catalogs(info.vizier_id)
        if not tables:
            raise RuntimeError(f"No tables returned from VizieR for '{catalog_key}'.")

        catalog = None
        if info.table is not None:
            for t in tables:
                if info.table in t.meta.get("name", ""):
                    catalog = t
                    break
        if catalog is None:
            catalog = tables[0]

        columns = list(catalog.colnames)

        column_details = {}
        for col_name in columns:
            col = catalog.columns[col_name]
            column_details[col_name] = {
                "description": getattr(col, "description", None) or "",
                "unit": str(col.unit) if col.unit else None,
            }
    except Exception as e:
        # Raise (do not return an error dict): this function is @lru_cache'd,
        # and a cached error dict would permanently poison the cache after a
        # single transient network failure. A raised exception is not cached,
        # so the next call retries cleanly.
        raise RuntimeError(
            f"Failed to query VizieR columns for '{catalog_key}': {e}"
        ) from e

    return {
        "columns": columns,
        "column_details": column_details,
        "used_by_radiosim": {
            "ra": info.ra_col,
            "dec": info.dec_col,
            "flux": info.flux_col,
            "spectral_index": info.spindex_col,
        },
        "vizier_id": info.vizier_id,
        "freq_mhz": info.freq_mhz,
        "flux_unit": info.flux_unit,
        "description": info.description,
    }


@functools.lru_cache(maxsize=8)
def get_racs_columns(band: str) -> dict[str, Any]:
    """Query CASDA TAP for available columns in a RACS catalog.

    Parameters
    ----------
    band : str
        RACS band: ``"low"``, ``"mid"``, or ``"high"``.

    Returns
    -------
    dict
        Keys:

        - ``"columns"`` : list[str] -- all column names in the TAP table
        - ``"used_by_radiosim"`` : dict[str, str] -- columns used by RadioSim
        - ``"tap_table"`` : str -- CASDA TAP table name
        - ``"freq_mhz"`` : float -- survey frequency
        - ``"description"`` : str -- band description

    Raises
    ------
    RuntimeError
        If the CASDA TAP query fails. The failure is not cached.

    Examples
    --------
    >>> info = get_racs_columns("low")
    >>> print(info["freq_mhz"])
    887.5
    """
    band = band.lower()
    if band not in RACS_CATALOGS:
        raise ValueError(
            f"Unknown RACS band '{band}'. Available: {sorted(RACS_CATALOGS.keys())}"
        )

    info = RACS_CATALOGS[band]

    require_service("casda", f"query live columns for RACS-{band} from CASDA")

    try:
        tap = TapPlus(url=CASDA_TAP_URL)
        job = tap.launch_job(
            f"SELECT column_name, description, unit "
            f"FROM tap_schema.columns "
            f"WHERE table_name='{info.tap_table}'"
        )
        result = job.get_results()
        columns = list(result["column_name"])

        column_details = {}
        for row in result:
            col_name = row["column_name"]
            column_details[col_name] = {
                "description": str(row["description"]) if row["description"] else "",
                "unit": str(row["unit"]) if row["unit"] else None,
            }
    except Exception as e:
        # Raise rather than return a cached error dict (see get_catalog_columns).
        raise RuntimeError(
            f"Failed to query CASDA TAP columns for RACS-{band}: {e}"
        ) from e

    return {
        "columns": columns,
        "column_details": column_details,
        "used_by_radiosim": {
            "ra": info.ra_col,
            "dec": info.dec_col,
            "flux": info.flux_col,
        },
        "tap_table": info.tap_table,
        "freq_mhz": info.freq_mhz,
        "description": info.description,
    }
