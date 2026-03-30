"""High-level Simulator API for RRIvis.

Provides a clean, user-friendly interface for running visibility simulations
in Python scripts and Jupyter notebooks. This is the recommended entry point
for most users.

Examples
--------
>>> # From configuration file
>>> from rrivis.api import Simulator
>>> sim = Simulator.from_config("config.yaml")
>>> results = sim.run()
>>> sim.plot()
>>> sim.save("output/")

>>> # Programmatic usage
>>> sim = Simulator(
...     antenna_layout="antennas.txt",
...     frequencies=[100, 150, 200],  # MHz
...     sky_model="gleam",
...     backend="auto",  # Auto-detect GPU
... )
>>> results = sim.run()

>>> # With specific simulator algorithm
>>> sim = Simulator(..., simulator="rime")  # Direct RIME (default)

>>> # With precision control
>>> from rrivis.core.precision import PrecisionConfig
>>> sim = Simulator(
...     antenna_layout="antennas.txt",
...     backend="jax",
...     precision="fast",  # Use float32 where safe
... )
>>> # Or use a custom precision config
>>> sim = Simulator(
...     antenna_layout="antennas.txt",
...     precision=PrecisionConfig.precise(),  # float128 for critical paths
... )
"""

import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, Union

import astropy.units as u
import numpy as np
from astropy.constants import c as speed_of_light

from rrivis.__about__ import __version__
from rrivis.utils.logging import (
    console,
    print_header,
    print_info,
    print_success,
    print_table,
    print_warning,
)

if TYPE_CHECKING:
    from rrivis.core.precision import PrecisionConfig


logger = logging.getLogger(__name__)


class Simulator:
    """
    High-level API for radio interferometry visibility simulation.

    This class provides a simple interface for configuring and running
    visibility simulations using the RIME (Radio Interferometer
    Measurement Equation) with full polarization support.

    The Simulator handles all the complexity of:
    - Loading antenna positions from various formats
    - Generating baselines
    - Loading sky models (test sources, GLEAM, GSM)
    - Computing beam patterns (analytic or FITS)
    - Running the visibility calculation with GPU acceleration
    - Saving results and generating plots

    Parameters
    ----------
    antenna_layout : str or Path, optional
        Path to antenna positions file.
    frequencies : list of float, optional
        List of frequencies in MHz.
    sky_model : str, optional
        Sky model name ("test", "gleam", "gsm").
    location : dict, optional
        Observatory location with lat, lon, height keys.
    start_time : str, optional
        Observation start time (ISO format).
    backend : str, optional
        Computation backend ("auto", "numpy", "jax", "numba").
        "auto" selects the best available backend.
    precision : str or PrecisionConfig, optional
        Numerical precision configuration. Can be:
        - None: Use standard float64 precision (default)
        - str: Preset name ("standard", "fast", "precise", "ultra")
        - PrecisionConfig: Full configuration object for granular control
        See rrivis.core.precision for details.
    simulator : str, optional
        Simulator algorithm ("rime"). Default is "rime".
    config : dict, optional
        Full configuration dictionary (overrides other args).

    Attributes
    ----------
    config : dict
        Configuration dictionary.
    results : dict or None
        Simulation results after run().
    version : str
        RRIvis version string.

    Examples
    --------
    >>> # Basic usage with config file
    >>> sim = Simulator.from_config("config.yaml")
    >>> results = sim.run()
    >>> sim.plot()
    >>> sim.save("output/")

    >>> # Programmatic usage with GPU
    >>> sim = Simulator(
    ...     antenna_layout="HERA65.csv",
    ...     frequencies=[100, 150, 200],
    ...     sky_model="gleam",
    ...     backend="jax",  # Use JAX GPU backend
    ... )
    >>> results = sim.run()

    See Also
    --------
    rrivis.simulator : Simulator algorithms
    rrivis.backends : Computation backends
    """

    def __init__(
        self,
        antenna_layout: str | Path | None = None,
        frequencies: list[float] | None = None,
        sky_model: str | None = None,
        location: dict[str, float] | None = None,
        start_time: str | None = None,
        backend: str = "auto",
        precision: Union["PrecisionConfig", str] | None = None,
        simulator: str = "rime",
        config: dict[str, Any] | None = None,
    ):
        """Initialize the Simulator."""
        self.version = __version__
        self._results: dict[str, Any] | None = None
        self._backend_name = backend
        self._precision = precision
        self._simulator_name = simulator
        self._backend = None
        self._simulator = None

        # Internal state (populated by setup())
        self._antennas: dict | None = None
        self._baselines: dict | None = None
        self._sources: list | None = None
        self._sky_model = None  # SkyModel for healpix_map representation
        self._sky_representation: str = "point_sources"  # Default sky representation
        self._location = None
        self._obstime = None
        self._frequencies_hz: np.ndarray | None = None
        self._wavelengths = None
        self._beam_config: dict = {}
        self._beam_manager = None
        self._is_setup = False

        # Build configuration from arguments
        if config is not None:
            self.config = config
        else:
            self.config = self._build_config(
                antenna_layout=antenna_layout,
                frequencies=frequencies,
                sky_model=sky_model,
                location=location,
                start_time=start_time,
            )

    def _build_config(
        self,
        antenna_layout: str | Path | None = None,
        frequencies: list[float] | None = None,
        sky_model: str | None = None,
        location: dict[str, float] | None = None,
        start_time: str | None = None,
    ) -> dict[str, Any]:
        """Build configuration dictionary from arguments."""
        config: dict[str, Any] = {}

        if antenna_layout:
            config["antenna_layout"] = {
                "antenna_positions_file": str(antenna_layout),
                "antenna_file_format": "rrivis",
                "all_antenna_diameter": 14.0,
            }

        if frequencies:
            freq_array = np.array(frequencies)
            config["obs_frequency"] = {
                "starting_frequency": float(np.min(freq_array)),
                "frequency_bandwidth": float(np.max(freq_array) - np.min(freq_array)),
                "frequency_interval": float(np.mean(np.diff(freq_array)))
                if len(freq_array) > 1
                else 1.0,
                "frequency_unit": "MHz",
            }

        if sky_model:
            test_sources_config = {
                "use_test_sources": sky_model in ["test", "test_sources"]
            }
            if test_sources_config["use_test_sources"]:
                test_sources_config.update(
                    {
                        "flux_min": 2.0,
                        "flux_max": 8.0,
                        "dec_deg": -30.72,
                        "spectral_index": -0.8,
                    }
                )
            config["sky_model"] = {
                "test_sources": test_sources_config,
                "gleam": {"use_gleam": sky_model == "gleam"},
                "gsm_healpix": {"use_gsm": sky_model == "gsm"},
            }

        if location:
            config["location"] = location

        if start_time:
            config["obs_time"] = {"start_time": start_time}

        return config

    @classmethod
    def from_config(cls, config_path: str | Path) -> "Simulator":
        """
        Create a Simulator from a YAML configuration file.

        Parameters
        ----------
        config_path : str or Path
            Path to YAML configuration file.

        Returns
        -------
        Simulator
            Configured Simulator instance.

        Examples
        --------
        >>> sim = Simulator.from_config("configs/config.yaml")
        >>> results = sim.run()

        With precision in config file:

        >>> # config.yaml:
        >>> # precision:
        >>> #   preset: "fast"  # or "standard", "precise", "ultra"
        >>> sim = Simulator.from_config("config.yaml")
        """
        from rrivis.io.config import RRIvisConfig

        config_path = Path(config_path)
        rrivis_config = RRIvisConfig.from_yaml(config_path)

        # Extract precision config if present
        precision = None
        if rrivis_config.precision is not None:
            precision = rrivis_config.precision.to_precision_config()

        # Convert Pydantic model to dict
        config_dict = rrivis_config.model_dump()

        return cls(config=config_dict, precision=precision)

    @property
    def results(self) -> dict[str, Any] | None:
        """Get simulation results (None if not yet run)."""
        return self._results

    @property
    def antennas(self) -> dict | None:
        """Get loaded antenna dictionary."""
        return self._antennas

    @property
    def baselines(self) -> dict | None:
        """Get generated baselines dictionary."""
        return self._baselines

    @property
    def sources(self) -> list | None:
        """Get loaded sources list."""
        return self._sources

    @property
    def precision(self) -> Optional["PrecisionConfig"]:
        """Get the precision configuration.

        Returns None if setup() hasn't been called yet or if no precision
        was specified (uses backend default).
        """
        if self._backend is not None:
            return self._backend.precision
        return None

    def setup(self) -> "Simulator":
        """
        Set up simulation components (antennas, baselines, sources).

        This method is called automatically by run(), but can be called
        separately to inspect the setup before running.

        Returns
        -------
        Simulator
            self (for method chaining)

        Examples
        --------
        >>> sim = Simulator.from_config("config.yaml")
        >>> sim.setup()
        >>> print(f"Antennas: {len(sim.antennas)}")
        >>> print(f"Baselines: {len(sim.baselines)}")
        """
        if self._is_setup:
            return self

        # Import core modules
        from rrivis.backends import get_backend
        from rrivis.core.antenna import read_antenna_positions
        from rrivis.core.baseline import generate_baselines
        from rrivis.core.observation import get_location_and_time
        from rrivis.simulator import get_simulator

        print_info("Setting up simulation...")

        # Initialize backend with precision
        self._backend = get_backend(self._backend_name, precision=self._precision)
        logger.debug(f"Using backend: {self._backend.name}")
        if self._backend.precision:
            logger.debug(f"Precision: {self._backend.precision.default}")

        # Initialize simulator
        self._simulator = get_simulator(self._simulator_name)
        logger.debug(
            f"Using simulator: {self._simulator.name} ({self._simulator.complexity})"
        )

        # Load antenna positions
        antenna_config = self.config.get("antenna_layout", {})
        antenna_file = antenna_config.get("antenna_positions_file")

        if antenna_file:
            # Only show verbose output if logger is at DEBUG level
            verbose = logger.isEnabledFor(logging.DEBUG)
            self._antennas = read_antenna_positions(
                antenna_file,
                format_type=antenna_config.get("antenna_file_format", "rrivis"),
                verbose=verbose,
            )
            logger.debug(f"Loaded {len(self._antennas)} antennas from {antenna_file}")
        else:
            raise ValueError("No antenna_positions_file specified in config")

        # Set antenna diameter for each antenna
        antenna_diameter = antenna_config.get("all_antenna_diameter", 14.0)

        for ant_id in self._antennas:
            self._antennas[ant_id]["diameter"] = antenna_diameter

        # Generate baselines
        # Build simple beam metadata maps for baseline generation
        ant_nums = [ant["Number"] for ant in self._antennas.values()]
        beams_per_antenna = dict.fromkeys(ant_nums, "gaussian")
        self._baselines = generate_baselines(
            self._antennas,
            beams_per_antenna,
            beams_per_antenna,
            verbose=verbose,
        )
        logger.debug(f"Generated {len(self._baselines)} baselines")

        # Get location and observation time
        loc_config = self.config.get("location", {})
        time_config = self.config.get("obs_time", {})

        lat = loc_config.get("lat") if loc_config.get("lat") != "" else None
        lon = loc_config.get("lon") if loc_config.get("lon") != "" else None
        height = loc_config.get("height") if loc_config.get("height") != "" else None

        self._location, self._obstime = get_location_and_time(
            lat=float(lat) if lat else None,
            lon=float(lon) if lon else None,
            height=float(height) if height else None,
            starttime=time_config.get("start_time"),
        )

        # Set up frequencies
        freq_config = self.config.get("obs_frequency", {})
        start_freq = freq_config.get("starting_frequency", 100.0)
        bandwidth = freq_config.get("frequency_bandwidth", 50.0)
        interval = freq_config.get("frequency_interval", 1.0)
        freq_unit = freq_config.get("frequency_unit", "MHz")

        # Convert to Hz
        unit_multipliers = {"Hz": 1, "kHz": 1e3, "MHz": 1e6, "GHz": 1e9}
        multiplier = unit_multipliers.get(freq_unit, 1e6)

        n_channels = max(1, int(bandwidth / interval)) + 1
        self._frequencies_hz = np.linspace(
            start_freq * multiplier, (start_freq + bandwidth) * multiplier, n_channels
        )

        # Calculate wavelengths
        self._wavelengths = (speed_of_light / (self._frequencies_hz * u.Hz)).to(u.m)

        logger.debug(
            f"Frequencies: {len(self._frequencies_hz)} channels, "
            f"{self._frequencies_hz[0] / 1e6:.1f} - {self._frequencies_hz[-1] / 1e6:.1f} MHz"
        )

        # Extract beam configuration for aperture-based model
        beam_config = self.config.get("beams", {})
        self._beam_config = {
            "aperture_shape": beam_config.get("aperture_shape", "circular"),
            "taper": beam_config.get("taper", "gaussian"),
            "edge_taper_dB": beam_config.get("edge_taper_dB", 10.0),
            "feed_model": beam_config.get("feed_model", "none"),
            "feed_computation": beam_config.get("feed_computation", "analytical"),
            "feed_params": beam_config.get("feed_params", {}),
            "reflector_type": beam_config.get("reflector_type", "prime_focus"),
            "magnification": beam_config.get("magnification", 1.0),
            "aperture_params": beam_config.get("aperture_params", {}),
        }

        # Get visibility configuration
        visibility_config = self.config.get("visibility", {})
        calculation_type = visibility_config.get("calculation_type", "direct_sum")
        sky_representation = visibility_config.get("sky_representation")

        # Validate calculation type
        if calculation_type == "spherical_harmonic":
            raise NotImplementedError(
                "Spherical harmonic (m-mode) visibility calculation is not yet implemented. "
                "Use calculation_type='direct_sum' instead."
            )

        # Load sky model using unified SkyModel class
        # Extract precision config to pass to SkyModel factory methods
        from rrivis.core.precision import PrecisionConfig
        from rrivis.core.sky import SkyModel

        _precision = (
            self._backend.precision if self._backend else PrecisionConfig.standard()
        )
        if _precision is None:
            _precision = PrecisionConfig.standard()

        sky_config = self.config.get("sky_model", {})

        # Flux unit conversion: convert user-specified flux values to canonical Jy
        flux_unit = sky_config.get("flux_unit")
        if flux_unit is None:
            raise ValueError(
                "sky_model.flux_unit is required. Set to 'Jy', 'mJy', or 'uJy'."
            )
        _flux_multipliers = {"Jy": 1.0, "mJy": 1e-3, "uJy": 1e-6}
        _flux_mul = _flux_multipliers[flux_unit]
        self._flux_unit = flux_unit

        test_config = sky_config.get("test_sources", {})
        test_healpix_config = sky_config.get("test_sources_healpix", {})
        gleam_config = sky_config.get("gleam", {})
        gsm_config = sky_config.get("gsm_healpix", {})
        mals_config = sky_config.get("mals", {})

        frequency = float(self._frequencies_hz[0])
        nside = gsm_config.get("nside", 64)

        # Collect all requested sky models
        sky_models = []

        if test_config.get("use_test_sources", False):
            num_sources = test_config.get("num_sources", 100)
            flux_min = test_config.get("flux_min")
            flux_max = test_config.get("flux_max")
            flux_range = (
                (flux_min * _flux_mul, flux_max * _flux_mul)
                if flux_min is not None and flux_max is not None
                else None
            )
            sky_models.append(
                SkyModel.from_test_sources(
                    num_sources=num_sources,
                    flux_range=flux_range,
                    dec_deg=test_config.get("dec_deg"),
                    spectral_index=test_config.get("spectral_index"),
                    distribution=test_config.get("distribution", "uniform"),
                    seed=test_config.get("seed"),
                    dec_range_deg=test_config.get("dec_range_deg"),
                    precision=_precision,
                )
            )
            logger.debug(f"Loaded test sources: {num_sources} sources")

        if test_healpix_config.get("use_test_sources", False):
            num_sources = test_healpix_config.get("num_sources", 100)
            flux_min = test_healpix_config.get("flux_min")
            flux_max = test_healpix_config.get("flux_max")
            flux_range = (
                (flux_min * _flux_mul, flux_max * _flux_mul)
                if flux_min is not None and flux_max is not None
                else None
            )
            hp_nside = test_healpix_config.get("nside", 64)
            sky = SkyModel.from_test_sources(
                num_sources=num_sources,
                flux_range=flux_range,
                dec_deg=test_healpix_config.get("dec_deg"),
                spectral_index=test_healpix_config.get("spectral_index"),
                distribution=test_healpix_config.get("distribution", "uniform"),
                seed=test_healpix_config.get("seed"),
                dec_range_deg=test_healpix_config.get("dec_range_deg"),
                precision=_precision,
            )
            obs_freq_config = self.config.get("obs_frequency", {})
            sky.to_healpix_for_observation(
                nside=hp_nside,
                obs_frequency_config=obs_freq_config,
            )
            sky_models.append(sky)
            logger.debug(
                f"Loaded test sources (HEALPix): {num_sources} sources, nside={hp_nside}"
            )

        if gleam_config.get("use_gleam", False):
            flux_limit = gleam_config.get("flux_limit", 1.0) * _flux_mul
            sky_models.append(
                SkyModel.from_gleam(flux_limit=flux_limit, precision=_precision)
            )

        if mals_config.get("use_mals", False):
            flux_limit = mals_config.get("flux_limit", 1.0) * _flux_mul
            release = mals_config.get("mals_release", "dr2")
            sky_models.append(
                SkyModel.from_mals(
                    flux_limit=flux_limit, release=release, precision=_precision
                )
            )

        if gsm_config.get("use_gsm", False):
            model_name = gsm_config.get("gsm_catalogue", "gsm2008")
            obs_freq_config = self.config.get("obs_frequency", {})
            sky_models.append(
                SkyModel.from_diffuse_sky(
                    model=model_name,
                    nside=nside,
                    obs_frequency_config=obs_freq_config,
                    include_cmb=gsm_config.get("include_cmb", False),
                    basemap=gsm_config.get("basemap"),
                    interpolation=gsm_config.get("interpolation"),
                    precision=_precision,
                )
            )

        # --- New point-source catalogs ---
        vlssr_config = sky_config.get("vlssr", {})
        if vlssr_config.get("use_vlssr", False):
            sky_models.append(
                SkyModel.from_vlssr(
                    flux_limit=vlssr_config.get("flux_limit", 1.0) * _flux_mul,
                    precision=_precision,
                )
            )

        tgss_config = sky_config.get("tgss", {})
        if tgss_config.get("use_tgss", False):
            sky_models.append(
                SkyModel.from_tgss(
                    flux_limit=tgss_config.get("flux_limit", 0.1) * _flux_mul,
                    precision=_precision,
                )
            )

        wenss_config = sky_config.get("wenss", {})
        if wenss_config.get("use_wenss", False):
            sky_models.append(
                SkyModel.from_wenss(
                    flux_limit=wenss_config.get("flux_limit", 0.05) * _flux_mul,
                    precision=_precision,
                )
            )

        sumss_config = sky_config.get("sumss", {})
        if sumss_config.get("use_sumss", False):
            sky_models.append(
                SkyModel.from_sumss(
                    flux_limit=sumss_config.get("flux_limit", 0.008) * _flux_mul,
                    precision=_precision,
                )
            )

        nvss_config = sky_config.get("nvss", {})
        if nvss_config.get("use_nvss", False):
            sky_models.append(
                SkyModel.from_nvss(
                    flux_limit=nvss_config.get("flux_limit", 0.0025) * _flux_mul,
                    precision=_precision,
                )
            )

        lotss_config = sky_config.get("lotss", {})
        if lotss_config.get("use_lotss", False):
            sky_models.append(
                SkyModel.from_lotss(
                    release=lotss_config.get("lotss_release", "dr2"),
                    flux_limit=lotss_config.get("flux_limit", 0.001) * _flux_mul,
                    precision=_precision,
                )
            )

        three_c_config = sky_config.get("three_c", {})
        if three_c_config.get("use_3c", False):
            sky_models.append(
                SkyModel.from_3c(
                    flux_limit=three_c_config.get("flux_limit", 1.0) * _flux_mul,
                    precision=_precision,
                )
            )

        vlass_config = sky_config.get("vlass", {})
        if vlass_config.get("use_vlass", False):
            sky_models.append(
                SkyModel.from_vlass(
                    flux_limit=vlass_config.get("flux_limit", 0.001) * _flux_mul,
                    precision=_precision,
                )
            )

        racs_config = sky_config.get("racs", {})
        if racs_config.get("use_racs", False):
            sky_models.append(
                SkyModel.from_racs(
                    band=racs_config.get("racs_band", "low"),
                    flux_limit=racs_config.get("flux_limit", 1.0) * _flux_mul,
                    max_rows=racs_config.get("max_rows", 1_000_000),
                    precision=_precision,
                )
            )

        # --- New diffuse models ---
        pysm3_config = sky_config.get("pysm3", {})
        if pysm3_config.get("use_pysm3", False):
            obs_freq_config = self.config.get("obs_frequency", {})
            sky_models.append(
                SkyModel.from_pysm3(
                    components=pysm3_config.get("components", "s1"),
                    nside=pysm3_config.get("nside", 64),
                    obs_frequency_config=obs_freq_config,
                    precision=_precision,
                )
            )

        # --- Local file loader via pyradiosky ---
        pyradiosky_config = sky_config.get("pyradiosky", {})
        if pyradiosky_config.get("use_pyradiosky", False):
            filename = pyradiosky_config.get("filename", "")
            if filename:
                obs_freq_config = self.config.get("obs_frequency", {})
                sky_models.append(
                    SkyModel.from_pyradiosky_file(
                        filename=filename,
                        filetype=pyradiosky_config.get("filetype"),
                        flux_limit=pyradiosky_config.get("flux_limit", 0.0) * _flux_mul,
                        reference_frequency_hz=pyradiosky_config.get(
                            "reference_frequency_hz"
                        ),
                        precision=_precision,
                        obs_frequency_config=obs_freq_config,
                    )
                )
            else:
                logger.warning(
                    "pyradiosky enabled but no filename specified; skipping."
                )

        # If no models selected, raise an error
        if not sky_models:
            raise ValueError(
                "No sky model enabled in configuration. "
                "Enable at least one sky model (e.g., test_sources, gleam, gsm_healpix, etc.) "
                "in the sky_model section of your config."
            )

        # Combine all models into one
        if len(sky_models) == 1:
            self._sky_model = sky_models[0]
        else:
            self._sky_model = SkyModel.combine(
                sky_models,
                representation=sky_representation,
                nside=nside,
                frequency=frequency,
                obs_frequency_config=self.config.get("obs_frequency", {}),
                precision=_precision,
            )

        # Ensure sky model is in requested representation
        self._sky_model.frequency = frequency
        self._sky_model.get_for_visibility(
            representation=sky_representation, nside=nside, frequency=frequency
        )

        # Get point sources for RIME calculator (needed even in healpix mode for some calculations)
        self._sources = self._sky_model.to_point_sources(frequency=frequency)

        # Store the sky representation for use in run()
        self._sky_representation = sky_representation

        self._is_setup = True
        n_sky = self._sky_model.n_sources
        sky_type = "pixels" if sky_representation == "healpix_map" else "sources"
        print_success(
            f"Setup complete: {len(self._antennas)} antennas, {len(self._baselines)} baselines, {n_sky} {sky_type}"
        )
        return self

    def run(
        self,
        progress: bool = True,
        n_workers: int | None = None,
    ) -> dict[str, Any]:
        """
        Run the visibility simulation.

        Parameters
        ----------
        progress : bool, optional
            Show progress information (default: True).
        n_workers : int, optional
            Number of parallel workers (default: auto).

        Returns
        -------
        dict
            Dictionary containing simulation results:
            - visibilities: Complex visibility data per baseline
            - baselines: Baseline information
            - frequencies: Frequency array in Hz
            - antennas: Antenna information
            - sources: Source list
            - metadata: Additional metadata

        Examples
        --------
        >>> sim = Simulator.from_config("config.yaml")
        >>> results = sim.run()
        >>> print(results.keys())
        dict_keys(['visibilities', 'baselines', 'frequencies', ...])
        """
        t_start = time.perf_counter()

        if progress:
            # Print beautiful header panel FIRST, before setup
            print_header(
                f"RRIvis Simulator v{self.version}",
                "Radio Interferometer Visibility Simulator",
            )

        # Set up if not already done
        if not self._is_setup:
            self.setup()

        t_setup = time.perf_counter() - t_start

        if progress:
            # Print configuration table (after setup, needs backend/sky_model info)
            n_sky = self._sky_model.n_sources if self._sky_model else len(self._sources)
            sky_label = (
                f"{n_sky} pixels (HEALPix)"
                if self._sky_representation == "healpix_map"
                else f"{n_sky} sources"
            )
            config_data = {
                "Backend": self._backend.name,
                "Precision": self._backend.precision.default
                if self._backend.precision
                else "standard",
                "Simulator": f"{self._simulator.name} ({self._simulator.complexity})",
                "Sky Mode": self._sky_representation,
                "Antennas": len(self._antennas),
                "Baselines": len(self._baselines),
                "Sky Model": sky_label,
                "Frequencies": f"{len(self._frequencies_hz)} channels",
            }
            if hasattr(self, "_flux_unit") and self._flux_unit != "Jy":
                config_data["Flux Unit"] = self._flux_unit
            print_table("Simulation Configuration", config_data)
            console.print()  # Add spacing

        print_info(
            f"Running visibility simulation ({self._sky_representation} mode)..."
        )

        # Calculate visibilities based on sky representation
        duration_seconds = self.config.get("obs_time", {}).get("duration_seconds", 1.0)
        time_step_seconds = self.config.get("obs_time", {}).get(
            "time_step_seconds", 1.0
        )

        if self._sky_representation == "healpix_map" and self._sky_model is not None:
            # Use direct HEALPix visibility calculation
            from rrivis.core.visibility_healpix import calculate_visibility_healpix

            use_pol = (
                self._sky_model is not None
                and self._sky_model.has_polarized_healpix_maps
            )

            healpix_result = calculate_visibility_healpix(
                sky_model=self._sky_model,
                antennas=self._antennas,
                baselines=self._baselines,
                location=self._location,
                obstime=self._obstime,
                wavelengths=self._wavelengths,
                freqs=self._frequencies_hz,
                duration_seconds=duration_seconds,
                time_step_seconds=time_step_seconds,
                beam_manager=self._beam_manager,
                backend=self._backend,
                output_units="Jy",
                beam_config=self._beam_config,
                include_polarization=use_pol,
            )

            # Convert healpix result format to match RIME format for compatibility
            # healpix returns: {"visibilities": (n_bl, n_t, n_f[, 2, 2]), ...}
            # RIME returns: {(ant1, ant2): {"I": ..., "XX": ..., ...}, ...}
            visibilities = {}
            baseline_keys = healpix_result["baseline_keys"]
            vis_array = healpix_result["visibilities"]

            if healpix_result.get("polarized", False):
                # Polarized: vis_array is (n_bl, n_t, n_f, 2, 2)
                for bl_idx, bl_key in enumerate(baseline_keys):
                    V = vis_array[bl_idx]  # (n_t, n_f, 2, 2)
                    visibilities[bl_key] = {
                        "XX": V[..., 0, 0],
                        "XY": V[..., 0, 1],
                        "YX": V[..., 1, 0],
                        "YY": V[..., 1, 1],
                        "I": V[..., 0, 0] + V[..., 1, 1],  # XX + YY = I
                    }
            else:
                # Scalar: vis_array is (n_bl, n_t, n_f)
                for bl_idx, bl_key in enumerate(baseline_keys):
                    visibilities[bl_key] = {
                        "I": vis_array[bl_idx],
                        "XX": vis_array[bl_idx],
                        "YY": vis_array[bl_idx],
                        "XY": np.zeros_like(vis_array[bl_idx]),
                        "YX": np.zeros_like(vis_array[bl_idx]),
                    }

        else:
            # Use point source RIME calculation (original behavior)
            # Build jones_config with beam settings
            jones_config = {"beam": self._beam_config}
            visibilities = self._simulator.calculate_visibilities(
                antennas=self._antennas,
                baselines=self._baselines,
                sources=self._sources,
                frequencies=self._frequencies_hz,
                backend=self._backend,
                # Required kwargs for RIME
                location=self._location,
                obstime=self._obstime,
                wavelengths=self._wavelengths,
                # Time-stepping parameters
                duration_seconds=duration_seconds,
                time_step_seconds=time_step_seconds,
                # Optional kwargs
                beam_manager=self._beam_manager,
                return_correlations=True,
                jones_config=jones_config,
            )

        t_total = time.perf_counter() - t_start

        # Compile results
        n_sky = self._sky_model.n_sources if self._sky_model else len(self._sources)
        self._results = {
            "visibilities": visibilities,
            "frequencies": self._frequencies_hz,
            "baselines": self._baselines,
            "antennas": self._antennas,
            "sources": self._sources,
            "sky_model": self._sky_model,
            "location": self._location,
            "obstime": self._obstime,
            "wavelengths": self._wavelengths,
            "timing": {"total": t_total, "setup": t_setup},
            "metadata": {
                "version": self.version,
                "backend": self._backend.name,
                "precision": self._backend.precision.model_dump()
                if self._backend.precision
                else None,
                "simulator": self._simulator.name,
                "sky_representation": self._sky_representation,
                "n_antennas": len(self._antennas),
                "n_baselines": len(self._baselines),
                "n_sky_elements": n_sky,
                "n_frequencies": len(self._frequencies_hz),
                "config": self.config,
            },
        }

        if progress:
            print_success(
                f"Simulation complete! ({t_total:.3f}s total, setup {t_setup:.3f}s)"
            )

        return self._results

    def plot(
        self,
        plot_type: str = "all",
        output_dir: str | Path | None = None,
        backend: str = "bokeh",
        show: bool = True,
        overwrite: bool = False,
    ) -> list[Path]:
        """
        Generate visualization plots.

        Parameters
        ----------
        plot_type : str, optional
            Type of plot to generate:
            - "all": All available plots (default)
            - "antenna": Antenna layout (2D and 3D)
            - "visibility": Visibility amplitude/phase vs time
            - "heatmap": Visibility frequency-time heatmaps
            - "frequency": Visibility modulus/phase vs frequency
        output_dir : str or Path, optional
            Directory to save plots. If None, displays interactively.
        backend : str, optional
            Plotting backend ("bokeh", "matplotlib").
        show : bool, optional
            Whether to display plots (default: True).

        Raises
        ------
        RuntimeError
            If no results are available (run simulation first).

        Notes
        -----
        Generated plots (when plot_type="all"):
        - antenna_layout.html: 2D antenna positions (E vs N)
        - antenna_layout_3d.html: 3D antenna positions (Plotly)
        - visibility-phase-lsts.html: Visibility amp/phase vs time
        - heatmaps-freq-time.html: Frequency-time heatmaps
        - modulus-phase-freq.html: Visibility vs frequency

        Examples
        --------
        >>> sim.run()
        >>> sim.plot(plot_type="antenna")
        >>> sim.plot(plot_type="all", output_dir="plots/")
        """
        if self._results is None:
            raise RuntimeError(
                "No results to plot. Run simulation first with sim.run()"
            )

        print_info(f"Generating {plot_type} plots with {backend}...")

        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

        # Snapshot of HTML files already in output_dir before plotting (to detect new ones).
        # If overwrite=True, skip the snapshot so all post-plot HTML files are returned.
        html_before: set[Path] = set()
        if output_dir and not overwrite:
            html_before = set(output_dir.glob("*.html"))

        # Antenna layout plot
        if plot_type in ["antenna", "all"]:
            from rrivis.visualization.bokeh_plots import (
                plot_antenna_layout,
                plot_antenna_layout_3d_plotly,
                plot_heatmaps,
                plot_modulus_vs_frequency,
                plot_visibility,
            )

            # 2D antenna layout
            plot_antenna_layout(
                self._antennas,
                plotting=backend,
                save_simulation_data=output_dir is not None,
                folder_path=str(output_dir) if output_dir else None,
                open_in_browser=show,
            )

            # 3D antenna layout (Plotly)
            plot_antenna_layout_3d_plotly(
                self._antennas,
                save_simulation_data=output_dir is not None,
                folder_path=str(output_dir) if output_dir else None,
                open_in_browser=show,
            )

        # Visibility time-series plots require multi-time data
        if plot_type in ["visibility", "heatmap", "all"]:
            # Check if we have multi-time data
            from rrivis.core.visibility import calculate_modulus_phase

            moduli, phases = calculate_modulus_phase(self._results["visibilities"])

            # Check if data has time dimension
            first_baseline = list(moduli.keys())[0]
            has_time_data = moduli[first_baseline].ndim == 2  # Shape (n_times, n_freq)

            if has_time_data and moduli[first_baseline].shape[0] > 1:
                # Multi-time data - generate time-series plots
                logger.debug(f"Generating {plot_type} plots with multi-time data")

                # Generate time points array
                time_config = self.config.get("obs_time", {})
                duration_sec = time_config.get("duration_seconds", 1.0)
                time_step_sec = time_config.get("time_step_seconds", 1.0)
                n_times = max(1, int(duration_sec / time_step_sec))

                if self._obstime:
                    start_mjd = self._obstime.mjd
                    time_points_mjd = np.array(
                        [
                            start_mjd + (i * time_step_sec / 86400.0)
                            for i in range(n_times)
                        ]
                    )
                else:
                    time_points_mjd = np.linspace(0, duration_sec / 86400.0, n_times)

                if plot_type in ["visibility", "all"]:
                    plot_visibility(
                        moduli_over_time=moduli,
                        phases_over_time=phases,
                        baselines=self._baselines,
                        mjd_time_points=time_points_mjd,
                        freqs=self._frequencies_hz,
                        total_seconds=duration_sec,
                        plotting=backend,
                        save_simulation_data=True,
                        folder_path=str(output_dir) if output_dir else None,
                        open_in_browser=show,
                    )

                if plot_type in ["heatmap", "all"]:
                    plot_heatmaps(
                        moduli_over_time=moduli,
                        phases_over_time=phases,
                        baselines=self._baselines,
                        freqs=self._frequencies_hz,
                        total_seconds=duration_sec,
                        mjd_time_points=time_points_mjd,
                        plotting=backend,
                        save_simulation_data=True,
                        folder_path=str(output_dir) if output_dir else None,
                        open_in_browser=show,
                    )

                if plot_type in ["frequency", "all"]:
                    plot_modulus_vs_frequency(
                        moduli_over_time=moduli,
                        phases_over_time=phases,
                        baselines=self._baselines,
                        freqs=self._frequencies_hz,
                        mjd_time_points=time_points_mjd,
                        plotting=backend,
                        save_simulation_data=True,
                        folder_path=str(output_dir) if output_dir else None,
                        open_in_browser=show,
                    )
            else:
                # Single-time data
                print_warning(
                    f"Plot type '{plot_type}' requires multi-time data. "
                    f"Current: {moduli[first_baseline].shape[0] if has_time_data else 1} time step(s). "
                    "Use plot_type='antenna' for single-time results."
                )

        # Collect newly written HTML files
        saved_paths: list[Path] = []
        if output_dir:
            html_after = set(output_dir.glob("*.html"))
            saved_paths = sorted(html_after - html_before)

        return saved_paths

    def save(
        self,
        output_dir: str | Path,
        format: str = "hdf5",
        overwrite: bool = False,
        telescope_name: str | None = None,
        filename: str | None = None,
    ) -> Path:
        """
        Save simulation results to disk.

        Parameters
        ----------
        output_dir : str or Path
            Output directory path.
        format : str, optional
            Output format: "hdf5" (default), "json", "ms" (Measurement Set).
        overwrite : bool, optional
            Overwrite existing files (default: False).
        telescope_name : str, optional
            Telescope name for MS metadata (default: from config or "RRIvis").
        filename : str, optional
            Output filename stem (without extension). Defaults to
            output_file_name from config, or "visibilities" if not set.

        Returns
        -------
        Path
            Path to saved output file.

        Raises
        ------
        RuntimeError
            If no results are available.
        ImportError
            If MS format requested but python-casacore not installed.

        Examples
        --------
        >>> sim.run()
        >>> output_path = sim.save("output/", format="hdf5")
        >>> print(f"Saved to {output_path}")

        Save as Measurement Set for use with CASA/QuartiCal:

        >>> output_path = sim.save("output/", format="ms")
        >>> # Can now run: quartical output/simulation.ms

        Notes
        -----
        The MS format is compatible with:
        - CASA: ``casabrowser output/simulation.ms``
        - QuartiCal: ``goquartical output/simulation.ms``
        - WSClean: ``wsclean -name image output/simulation.ms``
        """
        if self._results is None:
            raise RuntimeError(
                "No results to save. Run simulation first with sim.run()"
            )

        from rrivis.io.writers import save_visibilities_hdf5

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.debug(f"Saving results to {output_dir}...")

        if format.lower() == "hdf5":
            stem = (
                filename
                or self.config.get("output", {}).get("output_file_name")
                or "visibilities"
            )
            output_path = output_dir / f"{stem}.h5"

            if output_path.exists() and not overwrite:
                raise FileExistsError(
                    f"{output_path} already exists. Use overwrite=True to overwrite."
                )

            # Calculate time points for observation
            time_config = self.config.get("obs_time", {})
            duration_sec = time_config.get("duration_seconds", 1.0)
            time_step_sec = time_config.get("time_step_seconds", 1.0)
            n_times = max(1, int(duration_sec / time_step_sec))

            # Generate time points in MJD
            if self._obstime:
                start_mjd = self._obstime.mjd
                time_mjd = np.array(
                    [
                        start_mjd
                        + (i * time_step_sec / 86400.0)  # Convert seconds to days
                        for i in range(n_times)
                    ]
                )
            else:
                time_mjd = np.linspace(0, duration_sec / 86400.0, n_times)

            # Convert visibility format for HDF5 writer
            # Our format: {(ant1, ant2): {"I": (n_times, n_freq), ...}}
            # Writer expects: {(ant1, ant2): [(n_freq,), (n_freq,), ...]}
            # We need to split the time dimension into a list of arrays
            visibilities_for_writer = {}
            for bl_key, vis_dict in self._results["visibilities"].items():
                if isinstance(vis_dict, dict):
                    vis_array = vis_dict.get("I", vis_dict.get("XX", np.array([])))
                    # Split (n_times, n_freq) into list of (n_freq,) arrays
                    if vis_array.ndim == 2:
                        visibilities_for_writer[bl_key] = [
                            vis_array[i] for i in range(vis_array.shape[0])
                        ]
                    else:
                        visibilities_for_writer[bl_key] = [vis_array]
                else:
                    vis_array = vis_dict
                    if vis_array.ndim == 2:
                        visibilities_for_writer[bl_key] = [
                            vis_array[i] for i in range(vis_array.shape[0])
                        ]
                    else:
                        visibilities_for_writer[bl_key] = [vis_array]

            save_visibilities_hdf5(
                output_path=output_path,
                visibilities=visibilities_for_writer,
                frequencies=self._results["frequencies"],
                time_points_mjd=time_mjd,
                metadata=self._results["metadata"],
            )

            logger.debug(f"Saved HDF5 to {output_path}")
            return output_path

        elif format.lower() == "json":
            import json

            stem = (
                filename
                or self.config.get("output", {}).get("output_file_name")
                or "visibilities"
            )
            output_path = output_dir / f"{stem}.json"

            if output_path.exists() and not overwrite:
                raise FileExistsError(
                    f"{output_path} already exists. Use overwrite=True to overwrite."
                )

            # Convert complex arrays to serializable format
            json_data = {
                "metadata": self._results["metadata"],
                "frequencies": self._results["frequencies"].tolist(),
                "n_baselines": len(self._results["baselines"]),
            }

            with open(output_path, "w") as f:
                json.dump(json_data, f, indent=2, default=str)

            logger.debug(f"Saved JSON to {output_path}")
            return output_path

        elif format.lower() == "ms":
            from rrivis.io import MS_AVAILABLE, write_ms

            if not MS_AVAILABLE:
                raise ImportError(
                    "Measurement Set support not available.\n"
                    "Install with: pip install rrivis[ms]\n"
                    "Or: pip install python-casacore"
                )

            stem = (
                filename
                or self.config.get("output", {}).get("output_file_name")
                or "simulation"
            )
            output_path = output_dir / f"{stem}.ms"

            # Get telescope name from config or parameter
            if telescope_name is None:
                telescope_name = self.config.get("telescope", {}).get(
                    "telescope_name", "RRIvis"
                )

            # Get phase center from config if available
            location_config = self.config.get("location", {})
            phase_center_ra = float(location_config.get("ra", 0.0) or 0.0)
            phase_center_dec = float(location_config.get("dec", -30.0) or -30.0)

            write_ms(
                output_path=output_path,
                visibilities=self._results["visibilities"],
                frequencies=self._results["frequencies"],
                antennas=self._results["antennas"],
                baselines=self._results["baselines"],
                location=self._results["location"],
                obstime=self._results["obstime"],
                telescope_name=telescope_name,
                phase_center_ra=phase_center_ra,
                phase_center_dec=phase_center_dec,
                overwrite=overwrite,
            )

            logger.debug(f"Saved MS to {output_path}")
            return output_path

        else:
            raise ValueError(
                f"Unknown format: {format}. Supported: 'hdf5', 'json', 'ms'"
            )

    def get_memory_estimate(self) -> dict[str, Any]:
        """
        Estimate memory requirements for the simulation.

        Takes into account the precision configuration if set. Using float32
        precision reduces memory by ~50%, while float128 increases it by ~100%.

        Returns
        -------
        dict
            Memory estimates with human-readable sizes, including:
            - total_bytes: Total estimated bytes
            - total_human: Human-readable size
            - precision_factor: Memory multiplier from precision config

        Examples
        --------
        >>> sim = Simulator.from_config("config.yaml")
        >>> sim.setup()
        >>> mem = sim.get_memory_estimate()
        >>> print(f"Estimated memory: {mem['total_human']}")
        """
        if not self._is_setup:
            self.setup()

        # Get base memory estimate
        estimate = self._simulator.get_memory_estimate(
            n_antennas=len(self._antennas),
            n_baselines=len(self._baselines),
            n_sources=len(self._sources),
            n_frequencies=len(self._frequencies_hz),
        )

        # Adjust for precision if configured
        if self._backend.precision:
            precision_factor = self._backend.precision.estimate_memory_factor()
            estimate["precision_factor"] = precision_factor

            # Adjust byte estimates
            if "total_bytes" in estimate:
                estimate["total_bytes"] = int(
                    estimate["total_bytes"] * precision_factor
                )
                # Update human-readable string
                total_bytes = estimate["total_bytes"]
                if total_bytes > 1e9:
                    estimate["total_human"] = f"{total_bytes / 1e9:.1f} GB"
                elif total_bytes > 1e6:
                    estimate["total_human"] = f"{total_bytes / 1e6:.1f} MB"
                else:
                    estimate["total_human"] = f"{total_bytes / 1e3:.1f} KB"
        else:
            estimate["precision_factor"] = 1.0

        return estimate

    def __repr__(self) -> str:
        """String representation of Simulator."""
        status = "configured" if self._results is None else "completed"
        backend = self._backend.name if self._backend else self._backend_name
        return f"<Simulator v{self.version} [{status}] backend={backend}>"
