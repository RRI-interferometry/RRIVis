# rrivis/core/sky/_loaders_vizier.py
"""VizieR and CASDA TAP loader functions for SkyModel.

TODO: Data available on VizieR but not yet extracted
------------------------------------------------------
- NVSS polarization: columns ``Pol`` (polarised flux, mJy) and ``PA``
  (polarisation E-vector angle, deg).  Could derive Stokes Q/U via
  ``Q = Pol * cos(2*PA)``, ``U = Pol * sin(2*PA)``.
- Taylor+2009 RM catalog (VizieR ``J/ApJ/702/1230``): 37,543 NVSS sources
  with rotation measure.  Would need a dedicated ``from_taylor_rm()`` loader
  or cross-match with NVSS by angular separation.
- GLEAM Gaussian columns (``awide``, ``bwide``, ``pawide``) are *fitted*
  sizes (not deconvolved).  Proper extraction requires subtracting the PSF
  in quadrature using the ``psfawide``/``psfbwide`` columns.
- Multi-term spectral indices are not available from VizieR catalogs; they
  come from imaging pipelines (BBS/WSClean output files).
"""

from __future__ import annotations

import functools
import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from rrivis.core.precision import PrecisionConfig

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord
from astroquery.utils.tap.core import TapPlus
from astroquery.vizier import Vizier

from rrivis.utils.network import require_service

from ._factories import create_empty, create_from_arrays
from ._registry import get_loader, register_loader
from .catalogs import (
    CASDA_TAP_URL,
    RACS_CATALOGS,
    VIZIER_POINT_CATALOGS,
    VizierCatalogEntry,
)
from .region import SkyRegion

logger = logging.getLogger(__name__)


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

    def _empty():
        return create_empty(
            catalog_key,
            brightness_conversion,
            precision=precision,
        )

    info = VIZIER_POINT_CATALOGS[catalog_key]
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
        # Optional Gaussian morphology columns
        _major_col = info.major_col
        _minor_col = info.minor_col
        _pa_col = info.pa_col
        if _major_col:
            needed_cols.extend([_major_col, _minor_col, _pa_col])
        v = Vizier(columns=needed_cols, row_limit=max_rows or -1)

        # Push flux_limit filter to VizieR server to reduce download size
        flux_unit = info.flux_unit
        limit_in_catalog_units = flux_limit * (1000.0 if flux_unit == "mJy" else 1.0)
        v.column_filters = {info.flux_col: f">={limit_in_catalog_units}"}

        if region is not None:
            # Server-side spatial query -- one per atomic sub-region
            from astropy.table import vstack

            all_tables = []
            from .region import ConeRegion

            for sub in region._iter_atomic():
                if isinstance(sub, ConeRegion):
                    t = v.query_region(
                        sub.center,
                        radius=sub.radius,
                        catalog=[info.vizier_id],
                    )
                else:  # box
                    t = v.query_region(
                        sub.center,
                        width=sub.width,
                        height=sub.height,
                        catalog=[info.vizier_id],
                    )
                if t:
                    tbl = _select_table(t, info)
                    if tbl is not None:
                        all_tables.append(tbl)
            if not all_tables:
                raise ValueError("No tables returned from VizieR")
            catalog = vstack(all_tables) if len(all_tables) > 1 else all_tables[0]
        else:
            tables = v.get_catalogs(info.vizier_id)
            if not tables:
                raise ValueError("No tables returned from VizieR")
            catalog = _select_table(tables, info)
            if catalog is None:
                catalog = tables[0]
    except ConnectionError:
        raise  # Network issues already have good messages from require_service
    except Exception as e:
        raise RuntimeError(
            f"Failed to fetch catalog '{catalog_key}' from VizieR: {e}\n"
            f"If this is a network issue, check your connection. "
            f"If the error persists, the VizieR schema may have changed -- "
            f"please report at https://github.com/RRI-interferometry/RRIVis/issues"
        ) from e

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

    # Auto-detect sexagesimal coordinates: if the first valid RA value is a
    # string that can't be parsed as a float, treat coords as sexagesimal.
    if not is_sexagesimal and len(catalog) > 0:
        sample_ra = catalog[0][info.ra_col]
        if isinstance(sample_ra, (str, np.str_)):
            try:
                float(sample_ra)
            except ValueError:
                is_sexagesimal = True
                logger.debug(f"{catalog_key}: auto-detected sexagesimal coordinates")

    flux_col = info.flux_col
    flux_raw = _extract_masked_column(catalog, flux_col)
    flux_jy_raw = flux_raw * (1e-3 if flux_unit == "mJy" else 1.0)
    flux_valid = np.isfinite(flux_jy_raw) & (flux_jy_raw >= flux_limit)

    if not np.any(flux_valid):
        logger.info(
            f"{catalog_key.upper()}: no sources above flux limit {flux_limit} Jy"
        )
        return _empty()

    if is_sexagesimal:
        ra_strs = [str(v) for v in catalog[info.ra_col][flux_valid]]
        dec_strs = [str(v) for v in catalog[info.dec_col][flux_valid]]
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

    ra_rad = sc.ra.rad
    dec_rad = sc.dec.rad
    flux_jy = flux_jy_raw[flux_valid]
    n = len(flux_jy)

    valid_indices = np.where(flux_valid)[0]
    default_spindex = info.default_spindex
    alpha_arr = np.full(n, default_spindex, dtype=np.float64)
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

    # Extract Gaussian morphology columns if available
    _gauss_major = None
    _gauss_minor = None
    _gauss_pa = None
    if _major_col and _major_col in catalog.colnames:
        _raw_maj = _extract_masked_column(catalog, _major_col)[valid_indices]
        _raw_min = _extract_masked_column(catalog, _minor_col)[valid_indices]
        _raw_pa = _extract_masked_column(catalog, _pa_col)[valid_indices]
        # Replace NaN with 0 (unresolved -> point source)
        _gauss_major = np.where(np.isfinite(_raw_maj), _raw_maj, 0.0)
        _gauss_minor = np.where(np.isfinite(_raw_min), _raw_min, 0.0)
        _gauss_pa = np.where(np.isfinite(_raw_pa), _raw_pa, 0.0)

    # Client-side region trim (catches VizieR edge cases) + dedup
    if region is not None:
        in_region = region.contains(ra_rad, dec_rad)
        ra_rad = ra_rad[in_region]
        dec_rad = dec_rad[in_region]
        flux_jy = flux_jy[in_region]
        alpha_arr = alpha_arr[in_region]
        if source_name is not None:
            source_name = source_name[in_region]
        if source_id is not None:
            source_id = source_id[in_region]
        if _gauss_major is not None:
            _gauss_major = _gauss_major[in_region]
            _gauss_minor = _gauss_minor[in_region]
            _gauss_pa = _gauss_pa[in_region]
        n = len(flux_jy)

        # Dedup overlapping sub-region results
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
            if _gauss_major is not None:
                _gauss_major = _gauss_major[unique_idx]
                _gauss_minor = _gauss_minor[unique_idx]
                _gauss_pa = _gauss_pa[unique_idx]
            n = len(flux_jy)

    if n == 0:
        return _empty()

    logger.info(
        f"{catalog_key.upper()} loaded: {n:,} sources (flux >= {flux_limit} Jy)"
    )

    sky = create_from_arrays(
        ra_rad=ra_rad,
        dec_rad=dec_rad,
        flux=flux_jy,
        spectral_index=alpha_arr,
        ref_freq=np.full(n, info.freq_mhz * 1e6, dtype=np.float64),
        major_arcsec=_gauss_major,
        minor_arcsec=_gauss_minor,
        pa_deg=_gauss_pa,
        source_name=source_name,
        source_id=source_id,
        model_name=catalog_key,
        reference_frequency=info.freq_mhz * 1e6,
        brightness_conversion=brightness_conversion,
        precision=precision,
    )
    return sky


# =========================================================================
# Public loader functions (registered)
# =========================================================================


@register_loader(
    "gleam",
    config_section="gleam",
    use_flag="use_gleam",
    network_service="vizier",
    config_fields={
        "flux_limit": "flux_limit",
        "catalog": "catalog",
        "max_rows": "max_rows",
        "allow_full_catalog": "allow_full_catalog",
    },
)
def load_gleam(
    flux_limit: float = 1.0,
    catalog: str = "gleam_egc",
    brightness_conversion: str = "planck",
    *,
    precision: PrecisionConfig,
    region: SkyRegion | None = None,
    max_rows: int | None = None,
    allow_full_catalog: bool = False,
) -> SkyModel:  # noqa: F821
    """
    Load GLEAM catalog from VizieR.

    Parameters
    ----------
    flux_limit : float, default=1.0
        Minimum flux density in Jy.
    catalog : str, default="gleam_egc"
        Catalog key in ``VIZIER_POINT_CATALOGS``. Available GLEAM
        catalogs: ``"gleam_egc"``, ``"gleam_x_dr1"``, ``"gleam_x_dr2"``,
        ``"gleam_gal"``.
    precision : PrecisionConfig
        Precision configuration for array dtypes.

    Returns
    -------
    SkyModel
        Sky model with GLEAM sources.
    """
    _gleam_keys = sorted(k for k in VIZIER_POINT_CATALOGS if k.startswith("gleam"))
    if catalog not in VIZIER_POINT_CATALOGS:
        raise ValueError(f"Unknown GLEAM catalog '{catalog}'. Available: {_gleam_keys}")
    return _load_from_vizier_catalog(
        catalog,
        flux_limit,
        brightness_conversion,
        precision,
        region=region,
        max_rows=max_rows,
        allow_full_catalog=allow_full_catalog,
    )


@register_loader(
    "mals",
    config_section="mals",
    use_flag="use_mals",
    network_service="vizier",
    config_fields={
        "flux_limit": "flux_limit",
        "release": "release",
        "max_rows": "max_rows",
        "allow_full_catalog": "allow_full_catalog",
    },
)
def load_mals(
    flux_limit: float = 1.0,
    release: str = "dr2",
    brightness_conversion: str = "planck",
    *,
    precision: PrecisionConfig,
    region: SkyRegion | None = None,
    max_rows: int | None = None,
    allow_full_catalog: bool = False,
) -> SkyModel:  # noqa: F821
    """
    Load MALS catalog from VizieR.

    Parameters
    ----------
    flux_limit : float, default=1.0
        Minimum flux density in Jy. The catalog stores flux in mJy;
        unit conversion is handled internally by the generic loader.
    release : str, default="dr2"
        Data release: "dr1" or "dr2".
    precision : PrecisionConfig
        Precision configuration for array dtypes.

    Returns
    -------
    SkyModel
        Sky model with MALS sources.
    """
    key = f"mals_{release.lower()}"
    if key not in VIZIER_POINT_CATALOGS:
        raise ValueError(f"Unknown MALS release '{release}'. Available: 'dr1', 'dr2'.")
    return _load_from_vizier_catalog(
        key,
        flux_limit,
        brightness_conversion,
        precision,
        region=region,
        max_rows=max_rows,
        allow_full_catalog=allow_full_catalog,
    )


# =========================================================================
# Data-driven registration of simple VizieR loaders
# =========================================================================

# Catalog keys with only flux_limit as their config-driven parameter.
# Complex loaders (gleam, mals, lotss, racs) are kept as explicit functions.
_SIMPLE_VIZIER_CATALOGS = {
    "vlssr": "vlssr",
    "tgss": "tgss",
    "wenss": "wenss",
    "sumss": "sumss",
    "nvss": "nvss",
    "3c": "three_c",  # config_section differs from loader name
    "vlass": "vlass",
}


def _make_simple_vizier_loader(catalog_key: str):
    """Create a simple VizieR loader function for *catalog_key*."""
    info = VIZIER_POINT_CATALOGS[catalog_key]

    def loader(
        flux_limit: float = info.default_flux_limit,
        brightness_conversion: str = "planck",
        *,
        precision: PrecisionConfig,
        region: SkyRegion | None = None,
        max_rows: int | None = None,
        allow_full_catalog: bool = False,
    ):
        return _load_from_vizier_catalog(
            catalog_key,
            flux_limit,
            brightness_conversion,
            precision,
            region=region,
            max_rows=max_rows,
            allow_full_catalog=allow_full_catalog,
        )

    loader.__name__ = f"load_{catalog_key}"
    loader.__qualname__ = f"load_{catalog_key}"
    loader.__doc__ = (
        f"Load the {catalog_key.upper()} catalog from VizieR.\n\n"
        f"{info.description}\n\n"
        f"Parameters\n----------\n"
        f"flux_limit : float, default={info.default_flux_limit}\n"
        f"    Minimum flux density in Jy.\n"
        f"brightness_conversion : str, default='planck'\n"
        f"    Conversion method: 'planck' or 'rayleigh-jeans'.\n"
        f"precision : PrecisionConfig\n"
        f"    Precision configuration for array dtypes.\n"
        f"region : SkyRegion, optional\n"
        f"    Spatial filter.\n\n"
        f"Returns\n-------\nSkyModel\n"
    )
    return loader


for _key, _config_section in _SIMPLE_VIZIER_CATALOGS.items():
    _fn = _make_simple_vizier_loader(_key)
    register_loader(
        _key,
        config_section=_config_section,
        use_flag=f"use_{_key}",
        network_service="vizier",
        config_fields={
            "flux_limit": "flux_limit",
            "max_rows": "max_rows",
            "allow_full_catalog": "allow_full_catalog",
        },
    )(_fn)

# Expose as module-level names for direct import
load_vlssr = get_loader("vlssr")
load_tgss = get_loader("tgss")
load_wenss = get_loader("wenss")
load_sumss = get_loader("sumss")
load_nvss = get_loader("nvss")
load_3c = get_loader("3c")
load_vlass = get_loader("vlass")


@register_loader(
    "lotss",
    config_section="lotss",
    use_flag="use_lotss",
    network_service="vizier",
    config_fields={
        "flux_limit": "flux_limit",
        "release": "release",
        "max_rows": "max_rows",
        "allow_full_catalog": "allow_full_catalog",
    },
)
def load_lotss(
    release: str = "dr2",
    flux_limit: float = 0.001,
    brightness_conversion: str = "planck",
    *,
    precision: PrecisionConfig,
    region: SkyRegion | None = None,
    max_rows: int | None = None,
    allow_full_catalog: bool = False,
) -> SkyModel:  # noqa: F821
    """
    Load the LoTSS catalog from VizieR (144 MHz, DR1: ~325k, DR2: ~4.4M sources).

    LoTSS (Shimwell et al.) is the LOFAR Two-metre Sky Survey at 144 MHz.
    DR1: J/A+A/622/A1, DR2: J/A+A/659/A1 on VizieR.

    Parameters
    ----------
    release : str, default="dr2"
        Data release: "dr1" or "dr2".
    flux_limit : float, default=0.001
        Minimum flux density in Jy.
    brightness_conversion : str, default="planck"
        Conversion method: "planck" or "rayleigh-jeans".
    precision : PrecisionConfig
        Precision configuration for array dtypes.

    Returns
    -------
    SkyModel

    Raises
    ------
    ValueError
        If ``release`` is not "dr1" or "dr2".
    """
    key = f"lotss_{release.lower()}"
    if key not in VIZIER_POINT_CATALOGS:
        raise ValueError(f"Unknown LoTSS release '{release}'. Available: 'dr1', 'dr2'.")
    return _load_from_vizier_catalog(
        key,
        flux_limit,
        brightness_conversion,
        precision,
        region=region,
        max_rows=max_rows,
        allow_full_catalog=allow_full_catalog,
    )


@register_loader(
    "racs",
    config_section="racs",
    use_flag="use_racs",
    network_service="casda",
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
    from .model import SkyModel

    band = band.lower()
    if band not in RACS_CATALOGS:
        raise ValueError(
            f"Unknown RACS band '{band}'. Available: {sorted(RACS_CATALOGS.keys())}"
        )

    info = RACS_CATALOGS[band]
    flux_limit_mjy = flux_limit * 1000.0
    model_name = f"racs_{band}"

    logger.info(f"Fetching {info.description} via CASDA TAP")

    require_service("casda", f"download RACS-{band} catalog from CASDA")

    try:
        tap = TapPlus(url=CASDA_TAP_URL)
        adql = (
            f"SELECT TOP {max_rows} "
            f"{info.ra_col}, {info.dec_col}, {info.flux_col} "
            f"FROM {info.tap_table} "
            f"WHERE {info.flux_col} >= {flux_limit_mjy}"
        )
        if region is not None:
            spatial_parts = []
            pt = f"POINT('ICRS', {info.ra_col}, {info.dec_col})"
            from .region import ConeRegion

            for sub in region._iter_atomic():
                if isinstance(sub, ConeRegion):
                    spatial_parts.append(
                        f"CONTAINS({pt}, CIRCLE('ICRS', "
                        f"{sub.center.ra.deg}, {sub.center.dec.deg}, "
                        f"{sub.radius.deg})) = 1"
                    )
                else:  # box
                    spatial_parts.append(
                        f"CONTAINS({pt}, BOX('ICRS', "
                        f"{sub.center.ra.deg}, {sub.center.dec.deg}, "
                        f"{sub.width.deg}, {sub.height.deg})) = 1"
                    )
            adql += " AND (" + " OR ".join(spatial_parts) + ")"
        job = tap.launch_job(adql)
        result = job.get_results()

    except ConnectionError:
        raise
    except Exception as e:
        raise RuntimeError(
            f"Failed to fetch RACS-{band} from CASDA TAP: {e}\n"
            f"If this is a network issue, check your connection. "
            f"If the error persists, the CASDA schema may have changed -- "
            f"please report at https://github.com/RRI-interferometry/RRIVis/issues"
        ) from e

    freq_hz = info.freq_mhz * 1e6

    try:
        flux_raw = _extract_masked_column(result, info.flux_col) * 1e-3
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
        for row in result:
            try:
                flux_mjy = row[info.flux_col]
                if np.ma.is_masked(flux_mjy) or not np.isfinite(float(flux_mjy)):
                    continue
                flux_jy = float(flux_mjy) * 1e-3
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
                logger.debug(f"Skipping RACS row: {row_err}")
                continue
        ra_arr = np.array(ra_list, dtype=np.float64)
        dec_arr = np.array(dec_list, dtype=np.float64)
        flux_arr = np.array(flux_list, dtype=np.float64)
        source_name = (
            np.array(source_name_list, dtype=str) if source_name_list else None
        )
        source_id = np.array(source_id_list, dtype=str) if source_id_list else None
        if source_id is None and source_name is not None:
            source_id = source_name.copy()

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
        return create_empty(
            model_name,
            brightness_conversion,
            precision=precision,
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
    return sky


# =========================================================================
# Listing helpers (module-level functions)
# =========================================================================


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
    along with metadata about which columns RRIVis uses. Column
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
        - ``"used_by_rrivis"`` : dict[str, str | None] -- columns used by
          RRIVis (``"ra"``, ``"dec"``, ``"flux"``, ``"spectral_index"``)
        - ``"vizier_id"`` : str -- VizieR catalog identifier
        - ``"freq_mhz"`` : float -- survey reference frequency
        - ``"flux_unit"`` : str -- unit of the flux column
        - ``"description"`` : str -- catalog description

        If the query fails, the dict contains an ``"error"`` key instead.

    Examples
    --------
    >>> info = get_catalog_columns("nvss")
    >>> print(info["columns"][:5])
    ['recno', 'Field', 'Xpos', 'Ypos', 'NVSS']
    >>> print(info["used_by_rrivis"])
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
            return {"error": f"No tables returned from VizieR for '{catalog_key}'"}

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
        logger.error(f"Failed to query VizieR columns for {catalog_key}: {e}")
        return {"error": str(e)}

    return {
        "columns": columns,
        "column_details": column_details,
        "used_by_rrivis": {
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
        - ``"used_by_rrivis"`` : dict[str, str] -- columns used by RRIVis
        - ``"tap_table"`` : str -- CASDA TAP table name
        - ``"freq_mhz"`` : float -- survey frequency
        - ``"description"`` : str -- band description

        If the query fails, the dict contains an ``"error"`` key instead.

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
        logger.error(f"Failed to query CASDA TAP columns for RACS-{band}: {e}")
        return {"error": str(e)}

    return {
        "columns": columns,
        "column_details": column_details,
        "used_by_rrivis": {
            "ra": info.ra_col,
            "dec": info.dec_col,
            "flux": info.flux_col,
        },
        "tap_table": info.tap_table,
        "freq_mhz": info.freq_mhz,
        "description": info.description,
    }
