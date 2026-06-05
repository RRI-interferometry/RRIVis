"""Core VizieR loader: column extractors + the parametric loader function.

``_load_from_vizier_catalog`` is the heavy-lifting routine that every
public ``load_<catalog>`` wrapper delegates to. Catalog-specific
behaviour lives in this function's branches keyed off
``info = VIZIER_POINT_CATALOGS[catalog_key]``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord
from astroquery.vizier import Vizier

from radiosim.utils.network import require_service

from ...operations.factories import create_empty, create_from_arrays
from ...operations.region import SkyRegion
from ...registry.catalogs import VIZIER_POINT_CATALOGS, VizierCatalogEntry
from .provenance import _build_point_catalog_provenance

if TYPE_CHECKING:
    from radiosim.core.precision import PrecisionConfig

logger = logging.getLogger(__name__)


# =========================================================================
# Column extractors
# =========================================================================


def _extract_masked_column(catalog, col_name: str, dtype=np.float64) -> np.ndarray:
    """Extract an astropy Table column as a plain ndarray, masked entries -> NaN.

    Uses ``np.ma.filled`` instead of ``np.array`` to avoid astropy's
    default fill_value (often 1e+20) silently replacing masked entries.

    Parameters
    ----------
    catalog : astropy.table.Table
        Source table (from VizieR or TAP query).
    col_name : str
        Column name to extract.
    dtype : numpy dtype
        Output dtype (default float64).

    Returns
    -------
    np.ndarray
        Plain (non-masked) array with NaN where values were masked or
        non-finite.
    """
    col = catalog[col_name]
    arr = np.ma.filled(np.ma.array(col), fill_value=np.nan).astype(dtype)
    # np.asarray strips astropy Column/Quantity wrappers (which carry units
    # like deg or mJy) down to a plain ndarray, preventing unit conflicts
    # when the caller multiplies by astropy units (e.g. * u.deg -> deg^2).
    arr = np.asarray(arr)
    arr[~np.isfinite(arr)] = np.nan
    return arr


def _select_table(tables, info: VizierCatalogEntry) -> Any:
    """Select the correct table from a VizieR TableList.

    If the catalog metadata specifies a ``table`` name, search for it;
    otherwise return the first table.  Returns ``None`` if *tables* is
    empty.
    """
    if not tables:
        return None
    if info.table is not None:
        for t in tables:
            if info.table in t.meta.get("name", ""):
                return t
    return tables[0]


def _find_name_column(catalog) -> str | None:
    """Best-effort lookup for a stable source-name column."""
    lowered = {name.lower(): name for name in catalog.colnames}
    exact = (
        "source_name",
        "sourcename",
        "component_name",
        "componentname",
        "name",
    )
    for candidate in exact:
        if candidate in lowered:
            return lowered[candidate]
    for name in catalog.colnames:
        if "name" in name.lower():
            return name
    return None


def _find_id_column(catalog) -> str | None:
    """Best-effort lookup for a stable source-identifier column."""
    lowered = {name.lower(): name for name in catalog.colnames}
    exact = (
        "source_id",
        "sourceid",
        "component_id",
        "componentid",
        "objid",
        "id",
    )
    for candidate in exact:
        if candidate in lowered:
            return lowered[candidate]
    for name in catalog.colnames:
        low = name.lower()
        if low.endswith("_id") or low == "id":
            return name
    return None


def _extract_text_column(catalog, col_name: str) -> np.ndarray:
    """Extract a text-like astropy column as a plain string ndarray."""
    return np.asarray(
        np.ma.filled(np.ma.array(catalog[col_name]), fill_value=""),
        dtype=str,
    )


@dataclass(frozen=True)
class _VizierSources:
    ra_rad: np.ndarray
    dec_rad: np.ndarray
    flux_jy: np.ndarray
    spectral_index: np.ndarray
    source_name: np.ndarray | None
    source_id: np.ndarray | None
    major_arcsec: np.ndarray | None
    minor_arcsec: np.ndarray | None
    pa_deg: np.ndarray | None

    @property
    def n_sources(self) -> int:
        return len(self.flux_jy)


def _empty_vizier_sources() -> _VizierSources:
    empty = np.array([], dtype=np.float64)
    return _VizierSources(
        ra_rad=empty,
        dec_rad=empty,
        flux_jy=empty,
        spectral_index=empty,
        source_name=None,
        source_id=None,
        major_arcsec=None,
        minor_arcsec=None,
        pa_deg=None,
    )


def _empty_vizier_sky(
    *,
    info: VizierCatalogEntry,
    catalog_key: str,
    flux_limit: float,
    brightness_conversion: str,
    precision: PrecisionConfig | None,
    region: SkyRegion | None,
):
    provenance = _build_point_catalog_provenance(
        info=info,
        flux_limit_jy=flux_limit,
        flux_jy=None,
        catalog_key=catalog_key,
        region=region,
    )
    return create_empty(
        catalog_key,
        brightness_conversion,
        precision=precision,
        reference_frequency=info.freq_mhz * 1e6,
        provenance=provenance,
    )


def _fetch_vizier_catalog(
    *,
    catalog_key: str,
    info: VizierCatalogEntry,
    flux_limit: float,
    region: SkyRegion | None,
    max_rows: int | None,
    allow_full_catalog: bool,
):
    if region is None and max_rows is None and not allow_full_catalog:
        raise ValueError(
            f"Catalog '{catalog_key}' requires region=..., max_rows=..., or "
            "allow_full_catalog=True before downloading from VizieR."
        )

    logger.info(f"Fetching {info.description}")
    require_service("vizier", f"download catalog '{catalog_key}' from VizieR")
    logger.info("Downloading from VizieR...")

    try:
        needed_cols = [info.ra_col, info.dec_col, info.flux_col]
        if info.spindex_col:
            needed_cols.append(info.spindex_col)
        if info.major_col:
            needed_cols.extend([info.major_col, info.minor_col, info.pa_col])
        v = Vizier(columns=needed_cols, row_limit=max_rows or -1)

        limit_in_catalog_units = flux_limit / info.flux_unit_conversion_factor
        v.column_filters = {info.flux_col: f">={limit_in_catalog_units}"}

        if region is not None:
            from astropy.table import vstack

            all_tables = []
            from ...operations.region import ConeRegion

            for sub in region._iter_atomic():
                if isinstance(sub, ConeRegion):
                    tables = v.query_region(
                        sub.center,
                        radius=sub.radius,
                        catalog=[info.vizier_id],
                    )
                else:
                    tables = v.query_region(
                        sub.center,
                        width=sub.width,
                        height=sub.height,
                        catalog=[info.vizier_id],
                    )
                if tables:
                    table = _select_table(tables, info)
                    if table is not None:
                        all_tables.append(table)
            if not all_tables:
                raise ValueError("No tables returned from VizieR")
            return vstack(all_tables) if len(all_tables) > 1 else all_tables[0]

        tables = v.get_catalogs(info.vizier_id)
        if not tables:
            raise ValueError("No tables returned from VizieR")
        catalog = _select_table(tables, info)
        return catalog if catalog is not None else tables[0]
    except ConnectionError:
        raise
    except Exception as e:
        raise RuntimeError(
            f"Failed to fetch catalog '{catalog_key}' from VizieR: {e}\n"
            f"If this is a network issue, check your connection. "
            f"If the error persists, the VizieR schema may have changed -- "
            f"please report at https://github.com/RRI-interferometry/RadioSim/issues"
        ) from e


def _extract_vizier_sources(
    *,
    catalog_key: str,
    info: VizierCatalogEntry,
    catalog: Any,
    flux_limit: float,
) -> _VizierSources:
    n_rows = len(catalog)
    logger.info(f"Downloaded {n_rows:,} rows, processing...")

    if n_rows > 1_000_000:
        logger.warning(
            f"Catalog '{catalog_key}' has {n_rows:,} rows. "
            "This may require significant memory. "
            "Consider increasing flux_limit to reduce the source count."
        )

    is_sexagesimal = info.coords_sexagesimal
    coord_frame = info.coord_frame

    if not is_sexagesimal and len(catalog) > 0:
        sample_ra = catalog[0][info.ra_col]
        if isinstance(sample_ra, (str, np.str_)):
            try:
                float(sample_ra)
            except ValueError:
                is_sexagesimal = True
                logger.debug(f"{catalog_key}: auto-detected sexagesimal coordinates")

    flux_raw = _extract_masked_column(catalog, info.flux_col)
    flux_jy_raw = flux_raw * info.flux_unit_conversion_factor
    flux_valid = np.isfinite(flux_jy_raw) & (flux_jy_raw >= flux_limit)

    if not np.any(flux_valid):
        logger.info(
            f"{catalog_key.upper()}: no sources above flux limit {flux_limit} Jy"
        )
        return _empty_vizier_sources()

    if is_sexagesimal:
        ra_strs = [str(value) for value in catalog[info.ra_col][flux_valid]]
        dec_strs = [str(value) for value in catalog[info.dec_col][flux_valid]]
        sc = SkyCoord(ra_strs, dec_strs, unit=(u.hourangle, u.deg), frame=coord_frame)
    else:
        ra_raw = _extract_masked_column(catalog, info.ra_col)
        dec_raw = _extract_masked_column(catalog, info.dec_col)
        coord_ok = flux_valid & np.isfinite(ra_raw) & np.isfinite(dec_raw)
        sc = SkyCoord(
            ra=ra_raw[coord_ok] * u.deg,
            dec=dec_raw[coord_ok] * u.deg,
            frame=coord_frame,
        )
        flux_valid = coord_ok

    if coord_frame != "icrs":
        sc = sc.icrs

    flux_jy = flux_jy_raw[flux_valid]
    n = len(flux_jy)
    if n == 0:
        return _empty_vizier_sources()

    valid_indices = np.where(flux_valid)[0]
    alpha_arr = np.full(n, info.default_spindex, dtype=np.float64)
    name_col = _find_name_column(catalog)
    source_name = (
        _extract_text_column(catalog, name_col)[valid_indices]
        if name_col is not None
        else None
    )
    id_col = _find_id_column(catalog)
    source_id = (
        _extract_text_column(catalog, id_col)[valid_indices]
        if id_col is not None
        else None
    )
    if source_id is None and source_name is not None:
        source_id = source_name.copy()

    if info.spindex_col and info.spindex_col in catalog.colnames:
        spindex_raw = _extract_masked_column(catalog, info.spindex_col)
        spindex_valid = spindex_raw[valid_indices]
        finite_mask = np.isfinite(spindex_valid)
        alpha_arr[finite_mask] = spindex_valid[finite_mask]

    gauss_major = None
    gauss_minor = None
    gauss_pa = None
    if (
        info.major_col
        and info.minor_col
        and info.pa_col
        and info.major_col in catalog.colnames
    ):
        raw_maj = _extract_masked_column(catalog, info.major_col)[valid_indices]
        raw_min = _extract_masked_column(catalog, info.minor_col)[valid_indices]
        raw_pa = _extract_masked_column(catalog, info.pa_col)[valid_indices]
        gauss_major = np.where(np.isfinite(raw_maj), raw_maj, 0.0)
        gauss_minor = np.where(np.isfinite(raw_min), raw_min, 0.0)
        gauss_pa = np.where(np.isfinite(raw_pa), raw_pa, 0.0)

    return _VizierSources(
        ra_rad=sc.ra.rad,
        dec_rad=sc.dec.rad,
        flux_jy=flux_jy,
        spectral_index=alpha_arr,
        source_name=source_name,
        source_id=source_id,
        major_arcsec=gauss_major,
        minor_arcsec=gauss_minor,
        pa_deg=gauss_pa,
    )


def _apply_region_and_dedup(
    *,
    sources: _VizierSources,
    region: SkyRegion | None,
) -> _VizierSources:
    if region is None or sources.n_sources == 0:
        return sources

    in_region = region.contains(sources.ra_rad, sources.dec_rad)
    ra_rad = sources.ra_rad[in_region]
    dec_rad = sources.dec_rad[in_region]
    flux_jy = sources.flux_jy[in_region]
    alpha_arr = sources.spectral_index[in_region]
    source_name = (
        sources.source_name[in_region] if sources.source_name is not None else None
    )
    source_id = sources.source_id[in_region] if sources.source_id is not None else None
    gauss_major = (
        sources.major_arcsec[in_region] if sources.major_arcsec is not None else None
    )
    gauss_minor = (
        sources.minor_arcsec[in_region] if sources.minor_arcsec is not None else None
    )
    gauss_pa = sources.pa_deg[in_region] if sources.pa_deg is not None else None

    n = len(flux_jy)
    if n > 0 and len(region._iter_atomic()) > 1:
        unique_idx = None
        if source_id is not None and np.all(source_id != ""):
            _, unique_idx = np.unique(source_id, return_index=True)
        elif source_name is not None and np.all(source_name != ""):
            _, unique_idx = np.unique(source_name, return_index=True)
        else:
            coords_key = np.round(np.column_stack([ra_rad, dec_rad]), decimals=8)
            _, unique_idx = np.unique(coords_key, axis=0, return_index=True)
        unique_idx = np.sort(unique_idx)
        ra_rad = ra_rad[unique_idx]
        dec_rad = dec_rad[unique_idx]
        flux_jy = flux_jy[unique_idx]
        alpha_arr = alpha_arr[unique_idx]
        if source_name is not None:
            source_name = source_name[unique_idx]
        if source_id is not None:
            source_id = source_id[unique_idx]
        if gauss_major is not None:
            gauss_major = gauss_major[unique_idx]
            gauss_minor = gauss_minor[unique_idx]
            gauss_pa = gauss_pa[unique_idx]

    return _VizierSources(
        ra_rad=ra_rad,
        dec_rad=dec_rad,
        flux_jy=flux_jy,
        spectral_index=alpha_arr,
        source_name=source_name,
        source_id=source_id,
        major_arcsec=gauss_major,
        minor_arcsec=gauss_minor,
        pa_deg=gauss_pa,
    )


# =========================================================================
# Core VizieR loader (module-level function)
# =========================================================================


def _load_from_vizier_catalog(
    catalog_key: str,
    flux_limit: float = 1.0,
    brightness_conversion: str = "planck",
    precision: PrecisionConfig | None = None,
    region: SkyRegion | None = None,
    max_rows: int | None = None,
    allow_full_catalog: bool = False,
) -> SkyModel:  # noqa: F821
    """
    Load a point-source catalog from VizieR using unified metadata.

    This private helper is called by all public load_*() VizieR wrappers.
    It handles flux unit conversion (mJy->Jy), coordinate parsing
    (decimal/sexagesimal, ICRS/FK4), and spectral index extraction.

    Parameters
    ----------
    catalog_key : str
        Key into ``VIZIER_POINT_CATALOGS`` (e.g. "vlssr", "nvss").
    flux_limit : float, default=1.0
        Minimum flux density in Jy; rows below this are skipped.
    brightness_conversion : str, default="planck"
        Conversion method: "planck" or "rayleigh-jeans".
    region : SkyRegion, optional
        If given, only sources inside this sky region are loaded.
        Uses VizieR ``query_region()`` for server-side spatial
        filtering, then applies a client-side trim.

    Returns
    -------
    SkyModel
        Sky model with loaded point sources.

    Raises
    ------
    ValueError
        If ``catalog_key`` is not in ``VIZIER_POINT_CATALOGS``.
    """
    if catalog_key not in VIZIER_POINT_CATALOGS:
        raise ValueError(
            f"Unknown VizieR catalog key '{catalog_key}'. "
            f"Available: {sorted(VIZIER_POINT_CATALOGS.keys())}"
        )

    info = VIZIER_POINT_CATALOGS[catalog_key]
    catalog = _fetch_vizier_catalog(
        catalog_key=catalog_key,
        info=info,
        flux_limit=flux_limit,
        region=region,
        max_rows=max_rows,
        allow_full_catalog=allow_full_catalog,
    )
    sources = _extract_vizier_sources(
        catalog_key=catalog_key,
        info=info,
        catalog=catalog,
        flux_limit=flux_limit,
    )
    sources = _apply_region_and_dedup(sources=sources, region=region)

    if sources.n_sources == 0:
        return _empty_vizier_sky(
            info=info,
            catalog_key=catalog_key,
            flux_limit=flux_limit,
            brightness_conversion=brightness_conversion,
            precision=precision,
            region=region,
        )

    logger.info(
        f"{catalog_key.upper()} loaded: {sources.n_sources:,} sources "
        f"(flux >= {flux_limit} Jy)"
    )

    sky = create_from_arrays(
        ra_rad=sources.ra_rad,
        dec_rad=sources.dec_rad,
        flux=sources.flux_jy,
        spectral_index=sources.spectral_index,
        ref_freq=np.full(sources.n_sources, info.freq_mhz * 1e6, dtype=np.float64),
        major_arcsec=sources.major_arcsec,
        minor_arcsec=sources.minor_arcsec,
        pa_deg=sources.pa_deg,
        source_name=sources.source_name,
        source_id=sources.source_id,
        model_name=catalog_key,
        reference_frequency=info.freq_mhz * 1e6,
        brightness_conversion=brightness_conversion,
        precision=precision,
    )
    provenance = _build_point_catalog_provenance(
        info=info,
        flux_limit_jy=flux_limit,
        flux_jy=sources.flux_jy,
        catalog_key=catalog_key,
        region=region,
    )
    return sky.replace(provenance=provenance)


# =========================================================================
# Public loader functions (registered)
# =========================================================================
