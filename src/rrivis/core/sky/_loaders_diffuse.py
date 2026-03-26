# rrivis/core/sky/_loaders_diffuse.py
"""Diffuse sky model loader mixin (pygdsm, PySM3, ULSA) for SkyModel."""

import logging
from typing import Any

import healpy as hp
import numpy as np
from healpy.rotator import Rotator
from pygdsm import GlobalSkyModel, GSMObserver08

from .catalogs import DIFFUSE_MODELS

logger = logging.getLogger(__name__)


class _DiffuseLoadersMixin:
    """Mixin providing diffuse-sky factory classmethods for SkyModel."""

    @staticmethod
    def list_diffuse_models() -> dict[str, str]:
        """List available diffuse sky models with their descriptions.

        Returns
        -------
        dict[str, str]
            Mapping of model name to description string.

        Examples
        --------
        >>> for name, desc in SkyModel.list_diffuse_models().items():
        ...     print(f"{name}: {desc[:80]}...")
        """
        return {name: info["description"] for name, info in DIFFUSE_MODELS.items()}

    @classmethod
    def from_diffuse_sky(
        cls,
        model: str = "gsm2008",
        nside: int = 32,
        frequencies: np.ndarray | None = None,
        obs_frequency_config: dict[str, Any] | None = None,
        include_cmb: bool | None = None,
        basemap: str | None = None,
        interpolation: str | None = None,
        retain_pygdsm_instance: bool = False,
        brightness_conversion: str = "planck",
        precision: Any = None,
    ) -> "SkyModel":  # noqa: F821
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
        retain_pygdsm_instance : bool, default=False
            If True, keep the pygdsm model object on the returned SkyModel
            (accessible via the ``pygdsm_model`` property). This adds ~63 MB
            of memory overhead for GSM2008. When False (default), the pygdsm
            instance is discarded after map generation.
        brightness_conversion : str, default="planck"
            Conversion method for T_b -> Jy: "planck" (exact) or "rayleigh-jeans".

        Returns
        -------
        SkyModel
            Sky model in healpix_multifreq mode with one T_b map per frequency.

        Raises
        ------
        ValueError
            If neither ``frequencies`` nor ``obs_frequency_config`` is provided,
            if the model name is unknown, or if ``basemap``/``interpolation``
            are set for a non-GSM2008 model.

        Examples
        --------
        >>> freqs = np.linspace(100e6, 120e6, 20)
        >>> sky = SkyModel.from_diffuse_sky(
        ...     model="gsm2008", nside=32, frequencies=freqs
        ... )
        >>> sky.mode
        'healpix_multifreq'

        >>> sky = SkyModel.from_diffuse_sky(
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
        >>> sky = SkyModel.from_diffuse_sky(
        ...     model="lfsm", nside=64, obs_frequency_config=config
        ... )
        """
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
                "Example: from_diffuse_sky(model='gsm2008', nside=32, "
                "frequencies=np.linspace(100e6, 120e6, 20))"
            )

        if frequencies is None:
            frequencies = cls._parse_frequency_config(obs_frequency_config)
        frequencies = np.asarray(frequencies, dtype=np.float64)

        info = DIFFUSE_MODELS[model]
        model_class = info["class"]
        n_freq = len(frequencies)

        logger.info(
            f"Loading {model.upper()}: {n_freq} frequencies "
            f"({frequencies[0] / 1e6:.1f}\u2013{frequencies[-1] / 1e6:.1f} MHz), nside={nside}"
        )
        logger.info(f"Model info: {info['description']}")

        init_kwargs = info["init_kwargs"].copy()
        if include_cmb is not None:
            init_kwargs["include_cmb"] = include_cmb
        if basemap is not None:
            init_kwargs["basemap"] = basemap
        if interpolation is not None:
            init_kwargs["interpolation"] = interpolation
        pygdsm_instance = model_class(**init_kwargs)

        rot = Rotator(coord=["G", "C"])

        healpix_maps: dict[float, np.ndarray] = {}
        for freq in frequencies:
            temp_map = pygdsm_instance.generate(freq)
            current_nside = hp.get_nside(temp_map)
            if current_nside != nside:
                temp_map = hp.ud_grade(temp_map, nside_out=nside)
            temp_map = rot.rotate_map_pixel(temp_map)
            healpix_maps[float(freq)] = temp_map.astype(np.float32)

        logger.info(
            f"{model.upper()} loaded: {hp.nside2npix(nside)} pixels \u00d7 {n_freq} frequencies"
        )

        result = cls(
            _healpix_maps=healpix_maps,
            _healpix_nside=nside,
            _observation_frequencies=frequencies,
            _native_format="healpix",
            frequency=float(frequencies[0]),
            model_name=model,
            brightness_conversion=brightness_conversion,
            _precision=precision,
            _pygdsm_instance=pygdsm_instance if retain_pygdsm_instance else None,
        )
        result._ensure_dtypes()
        return result

    @staticmethod
    def create_gsm_observer(
        basemap: str = "haslam",
        interpolation: str = "pchip",
        include_cmb: bool = False,
    ) -> "GSMObserver08":
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
        >>> obs = SkyModel.create_gsm_observer(basemap="wmap")
        >>> obs.lat = "-30.72"
        >>> obs.lon = "21.43"
        >>> obs.date = "2025-01-15T00:00:00"
        >>> obs.generate(150e6)
        """
        gsm = GlobalSkyModel(
            freq_unit="Hz",
            basemap=basemap,
            interpolation=interpolation,
            include_cmb=include_cmb,
        )
        observer = GSMObserver08()
        observer.gsm = gsm
        return observer

    @classmethod
    def from_pysm3(
        cls,
        components: str | list[str] = "s1",
        nside: int = 64,
        frequencies: np.ndarray | None = None,
        obs_frequency_config: dict[str, Any] | None = None,
        brightness_conversion: str = "planck",
        precision: Any = None,
    ) -> "SkyModel":  # noqa: F821
        """
        Load a PySM3 diffuse sky model as multi-frequency HEALPix maps.

        Generates one brightness temperature map per observation frequency
        using PySM3's native per-channel computation. Maps are rotated from
        Galactic to Equatorial (ICRS) coordinates and stored as float32.

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
        brightness_conversion : str, default="planck"
            Conversion method for T_b -> Jy: "planck" or "rayleigh-jeans".

        Returns
        -------
        SkyModel
            Sky model in healpix_multifreq mode.

        Raises
        ------
        ValueError
            If neither ``frequencies`` nor ``obs_frequency_config`` is provided.
        """
        import pysm3
        import pysm3.units as pysm3_u

        if frequencies is None and obs_frequency_config is None:
            raise ValueError(
                "Either 'frequencies' or 'obs_frequency_config' must be provided. "
                "Example: from_pysm3(components='s1', nside=64, "
                "frequencies=np.linspace(100e6, 120e6, 20))"
            )

        if frequencies is None:
            frequencies = cls._parse_frequency_config(obs_frequency_config)
        frequencies = np.asarray(frequencies, dtype=np.float64)

        components_list = (
            [components] if isinstance(components, str) else list(components)
        )
        n_freq = len(frequencies)

        logger.info(
            f"Loading PySM3 components {components_list}: {n_freq} frequencies "
            f"({frequencies[0] / 1e6:.1f}\u2013{frequencies[-1] / 1e6:.1f} MHz), nside={nside}"
        )

        sky = pysm3.Sky(nside=nside, preset_strings=components_list)
        rot = Rotator(coord=["G", "C"])

        healpix_maps: dict[float, np.ndarray] = {}
        for freq in frequencies:
            emission = sky.get_emission(freq * pysm3_u.Hz)
            emission_krj = emission.to(
                pysm3_u.K_RJ,
                equivalencies=pysm3_u.cmb_equivalencies(freq * pysm3_u.Hz),
            )
            temp_map = np.array(emission_krj[0])  # Stokes I

            current_nside = hp.get_nside(temp_map)
            if current_nside != nside:
                temp_map = hp.ud_grade(temp_map, nside_out=nside)

            temp_map = rot.rotate_map_pixel(temp_map)
            healpix_maps[float(freq)] = temp_map.astype(np.float32)

        model_name = f"pysm3:{'+'.join(components_list)}"
        logger.info(
            f"PySM3 {components_list} loaded: {hp.nside2npix(nside)} pixels \u00d7 {n_freq} frequencies"
        )

        sky = cls(
            _healpix_maps=healpix_maps,
            _healpix_nside=nside,
            _observation_frequencies=frequencies,
            _native_format="healpix",
            frequency=float(frequencies[0]),
            model_name=model_name,
            brightness_conversion=brightness_conversion,
            _precision=precision,
        )
        sky._ensure_dtypes()
        return sky

    @classmethod
    def from_ulsa(
        cls,
        nside: int = 64,
        frequencies: np.ndarray | None = None,
        obs_frequency_config: dict[str, Any] | None = None,
        brightness_conversion: str = "planck",
        precision: Any = None,
    ) -> "SkyModel":  # noqa: F821
        """
        Load the ULSA ultra-low-frequency sky model as multi-frequency HEALPix maps.

        ULSA (Cong et al.) provides global sky brightness temperature maps
        below ~100 MHz. Maps are rotated from Galactic to Equatorial coordinates.

        Parameters
        ----------
        nside : int, default=64
            HEALPix NSIDE resolution.
        frequencies : np.ndarray, optional
            Array of observation frequencies in Hz. Takes precedence over
            ``obs_frequency_config`` when both are provided.
        obs_frequency_config : dict, optional
            Frequency configuration dict (keys: starting_frequency,
            frequency_interval, frequency_bandwidth, frequency_unit).
        brightness_conversion : str, default="planck"
            Conversion method for T_b -> Jy: "planck" or "rayleigh-jeans".

        Returns
        -------
        SkyModel
            Sky model in healpix_multifreq mode.

        Raises
        ------
        ImportError
            If ``ULSA`` is not installed.
        ValueError
            If neither ``frequencies`` nor ``obs_frequency_config`` is provided.
        """
        try:
            import ULSA as ulsa
        except ImportError as err:
            raise ImportError(
                "ULSA is required for from_ulsa(). "
                "Install it with: "
                "pip install git+https://github.com/Yanping-Cong/ULSA.git"
            ) from err

        if frequencies is None and obs_frequency_config is None:
            raise ValueError(
                "Either 'frequencies' or 'obs_frequency_config' must be provided. "
                "Example: from_ulsa(nside=64, frequencies=np.linspace(1e6, 100e6, 20))"
            )

        if frequencies is None:
            frequencies = cls._parse_frequency_config(obs_frequency_config)
        frequencies = np.asarray(frequencies, dtype=np.float64)

        n_freq = len(frequencies)
        logger.info(
            f"Loading ULSA: {n_freq} frequencies "
            f"({frequencies[0] / 1e6:.3f}\u2013{frequencies[-1] / 1e6:.3f} MHz), nside={nside}"
        )

        rot = Rotator(coord=["G", "C"])
        healpix_maps: dict[float, np.ndarray] = {}

        for freq in frequencies:
            freq_mhz = freq / 1e6
            try:
                # Try modern API: ulsa.generate(freq_mhz, nside=nside)
                temp_map = ulsa.generate(freq_mhz, nside=nside)
            except (AttributeError, TypeError):
                try:
                    # Fallback to older API: ulsa.Sky(nside).generate(freq_mhz)
                    temp_map = ulsa.Sky(nside).generate(freq_mhz)
                except Exception as e:
                    logger.error(f"ULSA generation failed at {freq_mhz:.3f} MHz: {e}")
                    npix = hp.nside2npix(nside)
                    temp_map = np.zeros(npix, dtype=np.float32)

            if len(temp_map) != hp.nside2npix(nside):
                temp_map = hp.ud_grade(temp_map, nside_out=nside)

            temp_map = rot.rotate_map_pixel(temp_map)
            healpix_maps[float(freq)] = temp_map.astype(np.float32)

        logger.info(
            f"ULSA loaded: {hp.nside2npix(nside)} pixels \u00d7 {n_freq} frequencies"
        )

        sky = cls(
            _healpix_maps=healpix_maps,
            _healpix_nside=nside,
            _observation_frequencies=frequencies,
            _native_format="healpix",
            frequency=float(frequencies[0]),
            model_name="ulsa",
            brightness_conversion=brightness_conversion,
            _precision=precision,
        )
        sky._ensure_dtypes()
        return sky
