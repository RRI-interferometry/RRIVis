# rrivis/core/sky/_loaders_diffuse.py
"""Diffuse sky model loader functions (pygdsm, PySM3) for SkyModel."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import healpy as hp
import numpy as np
from healpy.rotator import Rotator

from rrivis.utils.frequency import parse_frequency_config
from rrivis.utils.network import require_service

from ._data import HealpixData
from ._precision import get_sky_storage_dtype
from ._registry import register_loader
from .catalogs import DIFFUSE_MODELS
from .model import SkyFormat

if TYPE_CHECKING:
    from rrivis.core.precision import PrecisionConfig

    from .region import SkyRegion

logger = logging.getLogger(__name__)


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


@register_loader(
    "diffuse_sky",
    config_section="gsm_healpix",
    use_flag="use_gsm",
    is_healpix=True,
    network_service="pygdsm_data",
    aliases=["gsm", "gsm2008", "gsm2016", "lfsm", "haslam"],
    config_fields={
        "gsm_catalogue": "model",
        "nside": "nside",
        "include_cmb": "include_cmb",
        "basemap": "basemap",
        "interpolation": "interpolation",
    },
)
def load_diffuse_sky(
    model: str = "gsm2008",
    nside: int = 32,
    frequencies: np.ndarray | None = None,
    obs_frequency_config: dict[str, Any] | None = None,
    include_cmb: bool | None = None,
    basemap: str | None = None,
    interpolation: str | None = None,
    brightness_conversion: str = "planck",
    precision: PrecisionConfig | None = None,
    region: SkyRegion | None = None,
    memmap_path: str | None = None,
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
    >>> sky = load_diffuse_sky(model="gsm2008", nside=32, frequencies=freqs)
    >>> sky.mode
    'healpix_map'

    >>> sky = load_diffuse_sky(
    ...     model="gsm2008",
    ...     nside=32,
    ...     frequencies=freqs,
    ...     basemap="wmap",
    ...     interpolation="cubic",
    ... )

    >>> config = {
    ...     "starting_frequency": 100.0,
    ...     "frequency_interval": 1.0,
    ...     "frequency_bandwidth": 20.0,
    ...     "frequency_unit": "MHz",
    ... }
    >>> sky = load_diffuse_sky(model="lfsm", nside=64, obs_frequency_config=config)
    """
    from .model import SkyModel

    model = model.lower()
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

    if frequencies is None and obs_frequency_config is None:
        raise ValueError(
            "Either 'frequencies' or 'obs_frequency_config' must be provided. "
            "Example: load_diffuse_sky(model='gsm2008', nside=32, "
            "frequencies=np.linspace(100e6, 120e6, 20))"
        )

    if frequencies is None:
        frequencies = parse_frequency_config(obs_frequency_config)
    frequencies = np.asarray(frequencies, dtype=np.float64)

    info = DIFFUSE_MODELS[model]
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

    try:
        pygdsm_instance = model_class(**init_kwargs)
    except Exception as e:
        raise ConnectionError(
            f"Failed to initialize {model.upper()}: {e}\n"
            "This model requires internet access to download data files "
            "on first use. Check your network connection, or verify that "
            "Zenodo (zenodo.org) is reachable."
        ) from e

    from ._allocation import allocate_cube, ensure_scratch_dir, finalize_cube

    npix = hp.nside2npix(nside)
    scratch = ensure_scratch_dir(memmap_path) if memmap_path is not None else None
    hp_dtype = get_sky_storage_dtype(precision, "healpix_maps")
    i_arr = allocate_cube((n_freq, npix), hp_dtype, scratch, "i_maps")
    rot = Rotator(coord=["G", "C"])
    for fi, freq in enumerate(frequencies):
        temp_map = pygdsm_instance.generate(freq)
        if hp.get_nside(temp_map) != nside:
            temp_map = hp.ud_grade(temp_map, nside_out=nside)
        temp_map = rot.rotate_map_pixel(temp_map)
        i_arr[fi] = temp_map.astype(hp_dtype)

    # Apply region mask (zero out-of-region pixels)
    if region is not None:
        mask = region.healpix_mask(nside)
        n_retained = int(mask.sum())
        i_arr[:, ~mask] = 0.0
        logger.info(f"Region mask applied: {n_retained}/{npix} pixels retained")

    logger.info(f"{model.upper()} loaded: {npix} pixels \u00d7 {n_freq} frequencies")

    # Flush and re-open read-only if memmap-backed.
    i_arr = finalize_cube(i_arr, scratch, "i_maps")

    return SkyModel(
        healpix=HealpixData(maps=i_arr, nside=nside, frequencies=frequencies),
        native_representation=SkyFormat.HEALPIX,
        active_representation=SkyFormat.HEALPIX,
        model_name=model,
        brightness_conversion=brightness_conversion,
        _precision=precision,
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


@register_loader(
    "pysm3",
    config_section="pysm3",
    use_flag="use_pysm3",
    is_healpix=True,
    network_service="pysm3_data",
    config_fields={"components": "components", "nside": "nside"},
)
def load_pysm3(
    components: str | list[str] = "s1",
    nside: int = 64,
    frequencies: np.ndarray | None = None,
    obs_frequency_config: dict[str, Any] | None = None,
    include_polarization: bool = False,
    brightness_conversion: str = "planck",
    precision: PrecisionConfig | None = None,
    region: SkyRegion | None = None,
    memmap_path: str | None = None,
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

    Returns
    -------
    SkyModel
        Sky model in healpix_map mode.

    Raises
    ------
    ValueError
        If neither ``frequencies`` nor ``obs_frequency_config`` is provided.
    """
    import pysm3
    import pysm3.units as pysm3_u

    from .model import SkyModel

    if frequencies is None and obs_frequency_config is None:
        raise ValueError(
            "Either 'frequencies' or 'obs_frequency_config' must be provided. "
            "Example: load_pysm3(components='s1', nside=64, "
            "frequencies=np.linspace(100e6, 120e6, 20))"
        )

    if frequencies is None:
        frequencies = parse_frequency_config(obs_frequency_config)
    frequencies = np.asarray(frequencies, dtype=np.float64)

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
    from ._allocation import allocate_cube, ensure_scratch_dir, finalize_cube

    npix = hp.nside2npix(nside)
    scratch = ensure_scratch_dir(memmap_path) if memmap_path is not None else None
    hp_dtype = get_sky_storage_dtype(precision, "healpix_maps")
    i_arr = allocate_cube((n_freq, npix), hp_dtype, scratch, "i_maps")
    q_arr = (
        allocate_cube((n_freq, npix), hp_dtype, scratch, "q_maps")
        if include_polarization
        else None
    )
    u_arr = (
        allocate_cube((n_freq, npix), hp_dtype, scratch, "u_maps")
        if include_polarization
        else None
    )

    rot = Rotator(coord=["G", "C"])
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
                i_map = hp.ud_grade(i_map, nside_out=nside)
                q_map = hp.ud_grade(q_map, nside_out=nside)
                u_map = hp.ud_grade(u_map, nside_out=nside)
            iqu_rot = rot.rotate_map_alms(np.array([i_map, q_map, u_map]))
            i_arr[fi] = iqu_rot[0].astype(hp_dtype)
            q_arr[fi] = iqu_rot[1].astype(hp_dtype)
            u_arr[fi] = iqu_rot[2].astype(hp_dtype)
        else:
            temp_map = np.array(emission_krj[0])
            if hp.get_nside(temp_map) != nside:
                temp_map = hp.ud_grade(temp_map, nside_out=nside)
            temp_map = rot.rotate_map_pixel(temp_map)
            i_arr[fi] = temp_map.astype(hp_dtype)

    # Apply region mask (zero out-of-region pixels)
    if region is not None:
        mask = region.healpix_mask(nside)
        n_retained = int(mask.sum())
        i_arr[:, ~mask] = 0.0
        if q_arr is not None:
            q_arr[:, ~mask] = 0.0
        if u_arr is not None:
            u_arr[:, ~mask] = 0.0
        logger.info(f"Region mask applied: {n_retained}/{npix} pixels retained")

    model_name = f"pysm3:{'+'.join(components_list)}"
    logger.info(
        f"PySM3 {components_list} loaded: {npix} pixels "
        f"\u00d7 {n_freq} frequencies"
        f"{', stokes=IQU' if include_polarization else ''}"
    )

    # Flush and re-open read-only if memmap-backed.
    i_arr = finalize_cube(i_arr, scratch, "i_maps")
    if q_arr is not None:
        q_arr = finalize_cube(q_arr, scratch, "q_maps")
    if u_arr is not None:
        u_arr = finalize_cube(u_arr, scratch, "u_maps")

    return SkyModel(
        healpix=HealpixData(
            maps=i_arr,
            nside=nside,
            frequencies=frequencies,
            q_maps=q_arr,
            u_maps=u_arr,
        ),
        native_representation=SkyFormat.HEALPIX,
        active_representation=SkyFormat.HEALPIX,
        model_name=model_name,
        brightness_conversion=brightness_conversion,
        _precision=precision,
    )
