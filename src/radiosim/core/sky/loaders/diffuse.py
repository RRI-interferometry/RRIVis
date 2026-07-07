"""Diffuse sky model loader functions (pygdsm, PySM3) for SkyModel.

Spatial ``region`` arguments use the client-side convention: maps are built
(or reprojected) first, then cropped to in-region HEALPix pixels via mask
selection in :mod:`radiosim.core.sky.loaders._healpix_builder`.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np

from radiosim.utils.network import require_service

from ..containers import (
    MonopoleConvention,
    SkyProvenance,
    SourceSubtractionStatus,
)
from ..registry import DIFFUSE_MODELS, diffuse_sky_loader_registration, loader_registry
from ..support.frequencies import resolve_frequency_config
from ..support.healpy import healpy_rotator
from ..support.healpy import lazy_healpy as hp
from ..support.provenance_coverage import coverage_provenance
from ._healpix_builder import build_healpix_from_stokes_cube

if TYPE_CHECKING:
    from radiosim.core.precision import PrecisionConfig

    from ..operations.region import SkyRegion
    from ..registry import DiffuseModelEntry

logger = logging.getLogger(__name__)


def _pysm3() -> Any:
    """Import ``pysm3`` (and its units) on demand with a friendly error.

    PySM3 is an optional dependency; this lazy helper keeps the sky package
    importable without it and mirrors ``_pyradiosky_cls()``/``_h5py()``.

    Returns
    -------
    tuple
        ``(pysm3, pysm3.units)``.
    """
    try:
        import pysm3
        import pysm3.units as pysm3_units
    except ImportError as exc:
        raise ImportError(
            "Loading PySM3 sky models requires the optional 'pysm3' package. "
            "Install it with `pip install pysm3`."
        ) from exc
    return pysm3, pysm3_units


def _resolve_model_class(class_path: str) -> type:
    """Resolve a dotted class path to the actual class object.

    Parameters
    ----------
    class_path : str
        Fully qualified class name (e.g. ``"pygdsm.GlobalSkyModel"``).

    Returns
    -------
    type
        The resolved class.
    """
    import importlib

    module_name, class_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


# =========================================================================
# Listing helpers (module-level functions)
# =========================================================================


def list_diffuse_models() -> dict[str, str]:
    """List available diffuse sky models with their descriptions.

    Returns
    -------
    dict[str, str]
        Mapping of model name to description string.

    Examples
    --------
    >>> for name, desc in list_diffuse_models().items():
    ...     print(f"{name}: {desc[:80]}...")
    """
    return {name: info.description for name, info in DIFFUSE_MODELS.items()}


def get_diffuse_model_info(model_name: str) -> dict[str, Any]:
    """Get configuration parameters and metadata for a diffuse sky model.

    Parameters
    ----------
    model_name : str
        Model name: ``"gsm2008"``, ``"gsm2016"``, ``"lfsm"``, ``"haslam"``.

    Returns
    -------
    dict
        Keys:

        - ``"parameters"`` : dict -- constructor keyword arguments and defaults
        - ``"freq_range_hz"`` : tuple[float, float] -- valid frequency range
        - ``"description"`` : str -- model description
        - ``"class_name"`` : str -- pygdsm class name

    Raises
    ------
    ValueError
        If ``model_name`` is not recognized.

    Examples
    --------
    >>> info = get_diffuse_model_info("gsm2008")
    >>> print(info["parameters"])
    {'freq_unit': 'Hz', 'basemap': 'haslam', 'interpolation': 'pchip', 'include_cmb': False}
    """
    model_name = model_name.lower()
    if model_name not in DIFFUSE_MODELS:
        raise ValueError(
            f"Unknown diffuse model '{model_name}'. "
            f"Available: {sorted(DIFFUSE_MODELS.keys())}"
        )

    info = DIFFUSE_MODELS[model_name]
    return {
        "parameters": dict(info.init_kwargs),
        "freq_range_hz": info.freq_range,
        "description": info.description,
        "class_name": info.class_path.rsplit(".", 1)[-1],
    }


# =========================================================================
# Registered loader functions
# =========================================================================


@loader_registry.register_loader(
    "diffuse_sky",
    **diffuse_sky_loader_registration(
        config_fields=DIFFUSE_MODELS["gsm2008"].config_fields,
    ),
)
def load_diffuse_sky(
    model: str = "gsm2008",
    nside: int = 32,
    frequencies: np.ndarray | None = None,
    obs_frequency_config: dict[str, Any] | None = None,
    include_cmb: bool | None = None,
    basemap: str | None = None,
    interpolation: str | None = None,
    *,
    precision: PrecisionConfig,
    region: SkyRegion | None = None,
    memmap_path: str | None = None,
    brightness_conversion: str = "planck",
    provenance: SkyProvenance | None = None,
) -> SkyModel:  # noqa: F821
    """
    Load a diffuse sky model (GSM, LFSM, Haslam) as multi-frequency HEALPix maps.

    Calls ``pygdsm.generate(freq)`` for each observation frequency and stores
    the results as a ``{freq: T_b_map}`` dictionary. This preserves the native
    PCA spectral model of pygdsm without any two-point power-law approximation.

    Parameters
    ----------
    model : str, default="gsm2008"
        Model name: "gsm2008", "gsm2016", "lfsm", "haslam".
    nside : int, default=32
        HEALPix NSIDE resolution.
    frequencies : np.ndarray, optional
        Array of observation frequencies in Hz. Takes precedence over
        ``obs_frequency_config`` when both are provided.
    obs_frequency_config : dict, optional
        Frequency configuration dict (keys: starting_frequency,
        frequency_interval, frequency_bandwidth, frequency_unit).
        Used when ``frequencies`` is None.
    include_cmb : bool or None, default=None
        Include CMB contribution in the sky model. If None, uses the
        default from the model's ``init_kwargs`` (False for all models).
    basemap : str or None, default=None
        GSM2008-only: resolution basemap to use for PCA reconstruction.
        ``"haslam"`` (1 deg, best <1 GHz), ``"wmap"`` (2 deg, best for
        CMB frequencies), or ``"5deg"`` (native 5.1 deg PCA resolution).
        Raises ``ValueError`` if set for non-GSM2008 models.
        When None, uses the default from ``DIFFUSE_MODELS`` (``"haslam"``).
    interpolation : str or None, default=None
        GSM2008-only: frequency interpolation method.
        ``"pchip"`` (monotone, no overshoot) or ``"cubic"`` (cubic spline,
        closer to the original paper but can overshoot).
        Raises ``ValueError`` if set for non-GSM2008 models.
        When None, uses the default from ``DIFFUSE_MODELS`` (``"pchip"``).
    brightness_conversion : str, default="planck"
        Conversion method for T_b -> Jy: "planck" (exact) or "rayleigh-jeans".
    provenance : SkyProvenance, optional
        Explicit provenance override. When omitted, provenance is generated
        from diffuse-catalog metadata and the selected region.

    Returns
    -------
    SkyModel
        Sky model in healpix_map mode with one T_b map per frequency.

    Raises
    ------
    ValueError
        If neither ``frequencies`` nor ``obs_frequency_config`` is provided,
        if the model name is unknown, or if ``basemap``/``interpolation``
        are set for a non-GSM2008 model.

    Examples
    --------
    >>> freqs = np.linspace(100e6, 120e6, 20)
    >>> from radiosim.core.precision import PrecisionConfig
    >>> precision = PrecisionConfig.standard()
    >>> sky = load_diffuse_sky(
    ...     model="gsm2008",
    ...     nside=32,
    ...     frequencies=freqs,
    ...     precision=precision,
    ... )
    >>> sky.healpix is not None
    True

    >>> sky = load_diffuse_sky(
    ...     model="gsm2008",
    ...     nside=32,
    ...     frequencies=freqs,
    ...     basemap="wmap",
    ...     interpolation="cubic",
    ...     precision=precision,
    ... )

    >>> config = {
    ...     "starting_frequency": 100.0,
    ...     "frequency_interval": 1.0,
    ...     "frequency_bandwidth": 20.0,
    ...     "frequency_unit": "MHz",
    ... }
    >>> sky = load_diffuse_sky(
    ...     model="lfsm",
    ...     nside=64,
    ...     obs_frequency_config=config,
    ...     precision=precision,
    ... )
    """
    from ..containers.model import SkyModel

    model = model.lower()
    info = _validate_diffuse_model_args(model, basemap, interpolation)
    frequencies = resolve_frequency_config(frequencies, obs_frequency_config)

    model_class = _resolve_model_class(info.class_path)
    n_freq = len(frequencies)

    logger.info(
        f"Loading {model.upper()}: {n_freq} frequencies "
        f"({frequencies[0] / 1e6:.1f}\u2013{frequencies[-1] / 1e6:.1f} MHz), nside={nside}"
    )
    logger.info(f"Model info: {info.description}")

    init_kwargs = dict(info.init_kwargs)
    if include_cmb is not None:
        init_kwargs["include_cmb"] = include_cmb
    if basemap is not None:
        init_kwargs["basemap"] = basemap
    if interpolation is not None:
        init_kwargs["interpolation"] = interpolation

    require_service("pygdsm_data", f"load {model.upper()}", strict=False)
    pygdsm_instance = _instantiate_pygdsm(model, model_class, init_kwargs)

    npix = hp.nside2npix(nside)
    rot = healpy_rotator(coord=["G", "C"])
    first_full_sky_mean: float | None = None

    def _iter_stokes_rows():
        nonlocal first_full_sky_mean
        for fi, freq in enumerate(frequencies):
            temp_map = pygdsm_instance.generate(freq)
            if hp.get_nside(temp_map) != nside:
                temp_map = hp.ud_grade(
                    temp_map,
                    nside_out=nside,
                    order_in="RING",
                    order_out="RING",
                )
            temp_map = rot.rotate_map_pixel(temp_map)
            if fi == 0:
                first_full_sky_mean = float(np.mean(temp_map))
            yield (temp_map,)

    healpix = build_healpix_from_stokes_cube(
        stokes_rows=_iter_stokes_rows(),
        nside=nside,
        frequencies=frequencies,
        coordinate_frame="icrs",
        region=region,
        precision=precision,
        memmap_path=memmap_path,
        ordering="ring",
    )

    logger.info(
        f"{model.upper()} loaded: {healpix.n_pixels}/{npix} pixels "
        f"\u00d7 {n_freq} frequencies"
    )

    if provenance is None:
        provenance = _build_diffuse_provenance(
            model=model,
            info=info,
            init_kwargs=init_kwargs,
            region=region,
            first_full_sky_mean=first_full_sky_mean,
        )

    return SkyModel(
        healpix=healpix,
        model_name=model,
        brightness_conversion=brightness_conversion,
        provenance=provenance,
        precision=precision,
    )


def _validate_diffuse_model_args(
    model: str,
    basemap: str | None,
    interpolation: str | None,
) -> DiffuseModelEntry:
    """Validate the model name and GSM2008-only options; return its entry."""
    if model not in DIFFUSE_MODELS:
        raise ValueError(
            f"Unknown model '{model}'. Available: {list(DIFFUSE_MODELS.keys())}"
        )
    if basemap is not None and model != "gsm2008":
        raise ValueError(
            f"'basemap' is only supported for gsm2008, not '{model}'. "
            f"Remove the basemap parameter or use model='gsm2008'."
        )
    if interpolation is not None and model != "gsm2008":
        raise ValueError(
            f"'interpolation' is only supported for gsm2008, not '{model}'. "
            f"Remove the interpolation parameter or use model='gsm2008'."
        )
    return DIFFUSE_MODELS[model]


def _instantiate_pygdsm(model: str, model_class: type, init_kwargs: dict) -> Any:
    """Construct a pygdsm model instance, mapping network failures cleanly."""
    try:
        return model_class(**init_kwargs)
    except (TypeError, ValueError, KeyError, ImportError):
        # Bad basemap/version/argument or a missing optional dependency is a
        # configuration error, not a network failure: surface it as-is rather
        # than mislabeling it as a ConnectionError.
        raise
    except Exception as e:
        raise ConnectionError(
            f"Failed to initialize {model.upper()}: {e}\n"
            "This model requires internet access to download data files "
            "on first use. Check your network connection, or verify that "
            "Zenodo (zenodo.org) is reachable."
        ) from e


def _build_diffuse_provenance(
    *,
    model: str,
    info: DiffuseModelEntry,
    init_kwargs: dict,
    region: SkyRegion | None,
    first_full_sky_mean: float | None,
) -> SkyProvenance:
    """Assemble pygdsm provenance from catalog metadata + runtime choices."""
    if init_kwargs.get("include_cmb", False):
        monopole_convention = MonopoleConvention.ABSOLUTE_WITH_CMB
    else:
        monopole_convention = info.default_monopole_convention

    native_res_rad = (
        float(info.native_resolution_arcmin) * (np.pi / 180.0) / 60.0
        if info.native_resolution_arcmin is not None
        else None
    )
    angular_resolution_rad = (
        (native_res_rad, float(np.pi)) if native_res_rad is not None else None
    )

    if info.source_subtracted_above_jy is not None:
        src_sub = SourceSubtractionStatus.ABOVE_THRESHOLD
        src_sub_threshold = float(info.source_subtracted_above_jy)
        src_sub_freq = (
            float(info.source_subtraction_freq_hz)
            if info.source_subtraction_freq_hz is not None
            else None
        )
        src_sub_method = "gaussian_fit_inpaint"
    else:
        src_sub = SourceSubtractionStatus.NONE
        src_sub_threshold = None
        src_sub_freq = None
        src_sub_method = None

    coverage = coverage_provenance(is_full_sky=True, region=region)
    monopole_k = first_full_sky_mean if region is None else None

    return SkyProvenance(
        flux_completeness_jy=None,
        flux_completeness_freq_hz=None,
        angular_resolution_rad=angular_resolution_rad,
        sky_coverage=coverage.sky_coverage,
        coverage_fraction=coverage.coverage_fraction,
        coverage_footprint=coverage.coverage_footprint,
        monopole_convention=monopole_convention,
        monopole_k=monopole_k,
        source_subtraction=src_sub,
        source_subtraction_threshold_jy=src_sub_threshold,
        source_subtraction_freq_hz=src_sub_freq,
        source_subtraction_method=src_sub_method,
        notes=f"pygdsm/{model}",
    )


def create_pygdsm_model(
    model: str = "gsm2008",
    include_cmb: bool | None = None,
    basemap: str | None = None,
    interpolation: str | None = None,
) -> Any:
    """Create a pygdsm model instance for standalone use.

    Use this when you need direct access to pygdsm functionality
    (e.g. ``generate()``, ``view()``, ``write_fits()``) without
    going through SkyModel.

    Parameters
    ----------
    model : str, default="gsm2008"
        Model name: "gsm2008", "gsm2016", "lfsm", "haslam".
    include_cmb : bool or None
        Include CMB. None uses model default.
    basemap : str or None
        GSM2008-only basemap parameter.
    interpolation : str or None
        GSM2008-only interpolation parameter.

    Returns
    -------
    object
        A pygdsm model instance (e.g. GlobalSkyModel).
    """
    model = model.lower()
    if model not in DIFFUSE_MODELS:
        raise ValueError(
            f"Unknown model '{model}'. Available: {list(DIFFUSE_MODELS.keys())}"
        )
    info = DIFFUSE_MODELS[model]
    model_class = _resolve_model_class(info.class_path)
    init_kwargs = dict(info.init_kwargs)
    if include_cmb is not None:
        init_kwargs["include_cmb"] = include_cmb
    if basemap is not None:
        init_kwargs["basemap"] = basemap
    if interpolation is not None:
        init_kwargs["interpolation"] = interpolation
    return model_class(**init_kwargs)


def create_gsm_observer(
    basemap: str = "haslam",
    interpolation: str = "pchip",
    include_cmb: bool = False,
) -> Any:
    """Create a ``GSMObserver08`` with configurable GSM2008 parameters.

    The returned observer can be used to generate simulated sky views for
    a specific location, time, and frequency using the ``pygdsm``
    observation framework.

    Parameters
    ----------
    basemap : str, default="haslam"
        Resolution basemap: ``"haslam"`` (1 deg), ``"wmap"`` (2 deg),
        or ``"5deg"`` (native 5.1 deg PCA resolution).
    interpolation : str, default="pchip"
        Frequency interpolation: ``"pchip"`` (monotone, no overshoot)
        or ``"cubic"`` (cubic spline).
    include_cmb : bool, default=False
        Include CMB contribution (2.725 K).

    Returns
    -------
    GSMObserver08
        A pygdsm observer ready for ``.generate()`` after setting
        location and time via ``.lat``, ``.lon``, ``.date``.

    Examples
    --------
    >>> obs = create_gsm_observer(basemap="wmap")
    >>> obs.lat = "-30.72"
    >>> obs.lon = "21.43"
    >>> obs.date = "2025-01-15T00:00:00"
    >>> obs.generate(150e6)
    """
    from pygdsm import GlobalSkyModel, GSMObserver08

    gsm = GlobalSkyModel(
        freq_unit="Hz",
        basemap=basemap,
        interpolation=interpolation,
        include_cmb=include_cmb,
    )
    observer = GSMObserver08()
    observer.gsm = gsm
    return observer


@loader_registry.register_loader(
    "pysm3",
    config_section="pysm3",
    use_flag="use_pysm3",
    representations=("healpix_map",),
    category="diffuse",
    network_service="pysm3_data",
    config_fields=["components", "nside", "include_polarization"],
)
def load_pysm3(
    components: str | list[str] = "s1",
    nside: int = 64,
    frequencies: np.ndarray | None = None,
    obs_frequency_config: dict[str, Any] | None = None,
    include_polarization: bool = False,
    *,
    precision: PrecisionConfig,
    region: SkyRegion | None = None,
    memmap_path: str | None = None,
    brightness_conversion: str = "planck",
    provenance: SkyProvenance | None = None,
) -> SkyModel:  # noqa: F821
    """
    Load a PySM3 diffuse sky model as multi-frequency HEALPix maps.

    Generates one brightness temperature map per observation frequency
    using PySM3's native per-channel computation. Maps are rotated from
    Galactic to Equatorial (ICRS) coordinates and stored using the
    configured HEALPix precision.

    Parameters
    ----------
    components : str or list of str, default="s1"
        PySM3 preset string(s) (e.g. "s1", "d1", ["s1", "d1", "f1"]).
        See PySM3 documentation for available presets.
    nside : int, default=64
        HEALPix NSIDE resolution.
    frequencies : np.ndarray, optional
        Array of observation frequencies in Hz. Takes precedence over
        ``obs_frequency_config`` when both are provided.
    obs_frequency_config : dict, optional
        Frequency configuration dict (keys: starting_frequency,
        frequency_interval, frequency_bandwidth, frequency_unit).
    include_polarization : bool, default=False
        If True, extract Stokes Q and U maps from PySM3 in addition to
        Stokes I. The data is in K_RJ units; ``brightness_conversion``
        is forced to ``"rayleigh-jeans"`` when polarization is included.
        Coordinate rotation uses ``rotate_map_alms()`` for correct
        spin-2 handling of Q/U.
    brightness_conversion : str, default="planck"
        Conversion method for T_b -> Jy: ``"planck"`` or
        ``"rayleigh-jeans"``. Overridden to ``"rayleigh-jeans"`` when
        ``include_polarization=True``.
    provenance : SkyProvenance, optional
        Explicit provenance override. When omitted, provenance is generated
        from PySM component metadata and the selected region.

    Returns
    -------
    SkyModel
        Sky model in healpix_map mode.

    Raises
    ------
    ValueError
        If neither ``frequencies`` nor ``obs_frequency_config`` is provided.
    """
    pysm3, pysm3_u = _pysm3()

    from ..containers.model import SkyModel

    frequencies = resolve_frequency_config(frequencies, obs_frequency_config)

    if include_polarization:
        if brightness_conversion != "rayleigh-jeans":
            logger.info(
                "Using Rayleigh-Jeans conversion (required: polarized K_RJ data)"
            )
        brightness_conversion = "rayleigh-jeans"
    else:
        if brightness_conversion == "planck":
            logger.debug("Using Planck conversion (Stokes I only, default)")
        else:
            logger.debug("Using Rayleigh-Jeans conversion (user override)")

    components_list = [components] if isinstance(components, str) else list(components)
    n_freq = len(frequencies)

    logger.info(
        f"Loading PySM3 components {components_list}: {n_freq} frequencies "
        f"({frequencies[0] / 1e6:.1f}\u2013{frequencies[-1] / 1e6:.1f} MHz), "
        f"nside={nside}, polarization={'IQUV' if include_polarization else 'I'}"
    )

    require_service("pysm3_data", f"load PySM3 {components_list}", strict=False)

    try:
        pysm_sky = pysm3.Sky(nside=nside, preset_strings=components_list)
    except Exception as e:
        raise ConnectionError(
            f"Failed to initialize PySM3 with components {components_list}: {e}\n"
            "PySM3 requires internet access to download data files on "
            "first use. Check your network connection, or verify that "
            "NERSC portal (portal.nersc.gov) is reachable."
        ) from e
    npix = hp.nside2npix(nside)
    rot = healpy_rotator(coord=["G", "C"])
    first_full_sky_mean: float | None = None

    def _iter_stokes_rows():
        nonlocal first_full_sky_mean
        for fi, freq in enumerate(frequencies):
            emission = pysm_sky.get_emission(freq * pysm3_u.Hz)
            emission_krj = emission.to(
                pysm3_u.K_RJ,
                equivalencies=pysm3_u.cmb_equivalencies(freq * pysm3_u.Hz),
            )

            if include_polarization and emission_krj.shape[0] >= 3:
                i_map = np.array(emission_krj[0])
                q_map = np.array(emission_krj[1])
                u_map = np.array(emission_krj[2])
                current_nside = hp.get_nside(i_map)
                if current_nside != nside:
                    i_map = hp.ud_grade(
                        i_map,
                        nside_out=nside,
                        order_in="RING",
                        order_out="RING",
                    )
                    q_map = hp.ud_grade(
                        q_map,
                        nside_out=nside,
                        order_in="RING",
                        order_out="RING",
                    )
                    u_map = hp.ud_grade(
                        u_map,
                        nside_out=nside,
                        order_in="RING",
                        order_out="RING",
                    )
                iqu_rot = rot.rotate_map_alms(np.array([i_map, q_map, u_map]))
                i_row = iqu_rot[0]
                q_row = iqu_rot[1]
                u_row = iqu_rot[2]
            else:
                i_row = np.array(emission_krj[0])
                if hp.get_nside(i_row) != nside:
                    i_row = hp.ud_grade(
                        i_row,
                        nside_out=nside,
                        order_in="RING",
                        order_out="RING",
                    )
                i_row = rot.rotate_map_pixel(i_row)
                if include_polarization:
                    q_row = np.zeros_like(i_row)
                    u_row = np.zeros_like(i_row)
                else:
                    q_row = None
                    u_row = None

            if fi == 0:
                first_full_sky_mean = float(np.mean(i_row))
            yield (i_row, q_row, u_row, None)

    healpix = build_healpix_from_stokes_cube(
        stokes_rows=_iter_stokes_rows(),
        nside=nside,
        frequencies=frequencies,
        coordinate_frame="icrs",
        region=region,
        precision=precision,
        memmap_path=memmap_path,
        ordering="ring",
    )

    model_name = f"pysm3:{'+'.join(components_list)}"
    logger.info(
        f"PySM3 {components_list} loaded: {healpix.n_pixels}/{npix} pixels "
        f"\u00d7 {n_freq} frequencies"
        f"{', stokes=IQU' if include_polarization else ''}"
    )

    # PySM3 components are absolute, CMB-free brightness templates (unless "c1"
    # is requested, which adds CMB).  Detect the CMB component to set the
    # monopole convention; otherwise fall back to ABSOLUTE_NO_CMB.
    includes_cmb = any(c.lower().startswith("c") for c in components_list)
    monopole_convention = (
        MonopoleConvention.ABSOLUTE_WITH_CMB
        if includes_cmb
        else MonopoleConvention.ABSOLUTE_NO_CMB
    )

    # PySM3's native resolution depends on the chosen preset; as a conservative
    # lower bound take ~1.5 * pixel-scale at the chosen nside.
    pixel_res_rad = float(hp.nside2resol(nside))
    angular_resolution_rad = (pixel_res_rad, float(np.pi))

    coverage = coverage_provenance(is_full_sky=True, region=region)
    monopole_k = first_full_sky_mean if region is None else None

    generated_provenance = SkyProvenance(
        angular_resolution_rad=angular_resolution_rad,
        sky_coverage=coverage.sky_coverage,
        coverage_fraction=coverage.coverage_fraction,
        coverage_footprint=coverage.coverage_footprint,
        monopole_convention=monopole_convention,
        monopole_k=monopole_k,
        source_subtraction=SourceSubtractionStatus.NONE,
        notes=f"pysm3:{'+'.join(components_list)}",
    )
    model_provenance = generated_provenance if provenance is None else provenance

    return SkyModel(
        healpix=healpix,
        model_name=model_name,
        brightness_conversion=brightness_conversion,
        provenance=model_provenance,
        precision=precision,
    )
