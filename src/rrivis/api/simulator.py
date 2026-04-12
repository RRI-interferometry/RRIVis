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
        self._source_arrays: dict | None = None
        self._sky_model = None  # SkyModel for healpix_map representation
        self._location = None
        self._obstime = None
        self._frequencies_hz: np.ndarray | None = None
        self._wavelengths = None
        self._beam_config: dict = {}
        self._beam_manager = None
        self._is_setup = False
        self._network_status = None
        self._device_resources = None
        self._offline = (
            config.get("compute", {}).get("offline", False) if config else False
        )

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
                # Store the raw frequency array so parse_frequency_config()
                # can use it directly instead of the lossy linspace
                # reconstruction from start/bandwidth/interval.
                "frequencies_hz": (freq_array * 1e6).tolist(),
            }

        if sky_model:
            from rrivis.core.sky.registry import (
                build_alias_map,
                get_loader_definition,
                list_loaders,
            )

            aliases = build_alias_map()
            synthetic_names = {"test", "test_sources", "test_healpix"}

            if sky_model in ("test", "test_sources"):
                source = {
                    "kind": "test_sources",
                    "flux_min": 2.0,
                    "flux_max": 8.0,
                    "dec_deg": -30.72,
                    "spectral_index": -0.8,
                }
                is_healpix = False
            elif sky_model == "test_healpix":
                source = {
                    "kind": "test_sources",
                    "representation": "healpix_map",
                    "nside": 64,
                }
                is_healpix = True
            else:
                canonical = aliases.get(sky_model, sky_model)
                try:
                    definition = get_loader_definition(canonical)
                except ValueError:
                    available = sorted(
                        set(list_loaders()) | set(aliases) | synthetic_names
                    )
                    raise ValueError(
                        f"Unknown sky_model '{sky_model}'. Available: {available}."
                    ) from None
                source = {"kind": canonical}
                if canonical == "diffuse_sky" and sky_model in aliases:
                    source["model"] = sky_model
                is_healpix = definition.is_healpix

            config["sky_model"] = {"flux_unit": "Jy", "sources": [source]}
            config["visibility"] = {
                "sky_representation": "healpix_map" if is_healpix else "point_sources",
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
    def source_arrays(self) -> dict | None:
        """Get loaded source arrays dict."""
        return self._source_arrays

    @property
    def precision(self) -> Optional["PrecisionConfig"]:
        """Get the precision configuration.

        Returns None if setup() hasn't been called yet or if no precision
        was specified (uses backend default).
        """
        if self._backend is not None:
            return self._backend.precision
        return None

    @property
    def network_status(self):
        """Get the network status detected during setup.

        Returns None if setup() hasn't been called yet.
        """
        return self._network_status

    @property
    def device_resources(self):
        """Get the device resources detected during setup.

        Returns None if setup() hasn't been called yet.
        """
        return self._device_resources

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

        # Device resource detection
        from rrivis.utils.device import get_device_resources

        self._device_resources = get_device_resources()
        print_info(f"Device: {self._device_resources.summary()}")

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

        # Set up frequencies (single source of truth: parse_frequency_config)
        from rrivis.utils.frequency import parse_frequency_config

        self._frequencies_hz = parse_frequency_config(
            self.config.get("obs_frequency", {})
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
        sky_representation = visibility_config.get(
            "sky_representation", "point_sources"
        )
        allow_lossy_point_materialization = visibility_config.get(
            "allow_lossy_point_materialization", False
        )

        # Validate calculation type
        if calculation_type == "spherical_harmonic":
            raise NotImplementedError(
                "Spherical harmonic (m-mode) visibility calculation is not yet implemented. "
                "Use calculation_type='direct_sum' instead."
            )

        # Network connectivity check (before sky model loading)
        from rrivis.utils.network import (
            SERVICE_DISPLAY_NAMES,
            get_network_status,
            get_required_services,
        )

        self._network_status = get_network_status(offline=self._offline)

        sky_config = self.config.get("sky_model", {})
        required_services = get_required_services(sky_config)

        if self._network_status.forced_offline:
            status_label = "offline (forced)"
        elif self._network_status.is_online:
            status_label = "online"
        else:
            status_label = "offline"

        if required_services:
            service_names = [SERVICE_DISPLAY_NAMES.get(s, s) for s in required_services]
            print_info(
                f"Network: {status_label} (required: {', '.join(service_names)})"
            )
            if not self._network_status.is_online:
                for service, models in required_services.items():
                    display = SERVICE_DISPLAY_NAMES.get(service, service)
                    model_names = ", ".join(models)
                    print_warning(
                        f"Sky model(s) [{model_names}] require {display} "
                        f"but network is unavailable"
                    )
        else:
            print_info(f"Network: {status_label} (no network-dependent models)")

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
        flux_unit = sky_config.get("flux_unit", "Jy")
        _flux_multipliers = {"Jy": 1.0, "mJy": 1e-3, "uJy": 1e-6}
        _flux_mul = _flux_multipliers[flux_unit]
        self._flux_unit = flux_unit

        # Brightness conversion method (applies to all loaders)
        _brightness_conv = sky_config.get("brightness_conversion", "planck")
        mixed_model_policy = sky_config.get("mixed_model_policy", "error")

        # Build sky region from config (if specified)
        from rrivis.core.sky.region import SkyRegion

        def _build_sky_region(entry: dict) -> SkyRegion:
            if entry.get("shape", "cone") == "cone":
                return SkyRegion.cone(
                    entry["center_ra_deg"],
                    entry["center_dec_deg"],
                    entry["radius_deg"],
                )
            return SkyRegion.box(
                entry["center_ra_deg"],
                entry["center_dec_deg"],
                entry["width_deg"],
                entry["height_deg"],
            )

        region_config = sky_config.get("region")
        region = None
        if region_config:
            if isinstance(region_config, list):
                region = SkyRegion.union([_build_sky_region(e) for e in region_config])
            else:
                region = _build_sky_region(region_config)

        from rrivis.io.config import SkySourceConfig

        frequency = float(self._frequencies_hz[0])
        source_specs = [
            spec
            if isinstance(spec, SkySourceConfig)
            else SkySourceConfig.model_validate(spec)
            for spec in sky_config.get("sources", [])
        ]
        nside = next(
            (int(spec.nside) for spec in source_specs if spec.nside is not None),
            64,
        )

        # Collect all requested sky models
        sky_models = []
        from rrivis.core.sky.registry import get_loader_definition

        obs_freq_config = self.config.get("obs_frequency", {})
        loader_requests: list[tuple[str, dict[str, Any]]] = []

        for spec in source_specs:
            get_loader_definition(spec.kind)
            loader_requests.append(
                spec.to_loader_request(
                    flux_multiplier=_flux_mul,
                    region=region,
                    brightness_conversion=_brightness_conv,
                    frequencies=self._frequencies_hz,
                    obs_frequency_config=obs_freq_config,
                )
            )

        if loader_requests:
            sky_models.extend(
                SkyModel.load_parallel(
                    loader_requests,
                    max_workers=8,
                    precision=_precision,
                    strict=True,
                )
            )

        # If no models selected, raise an error
        if not sky_models:
            raise ValueError(
                "No sky model enabled in configuration. "
                "Enable at least one sky model source spec "
                "(for example kind='test_sources', 'gleam', or 'diffuse_sky') "
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
                frequencies=self._frequencies_hz,
                obs_frequency_config=self.config.get("obs_frequency", {}),
                allow_lossy_point_materialization=allow_lossy_point_materialization,
                mixed_model_policy=mixed_model_policy,
                precision=_precision,
            )

        # Ensure sky model is in requested representation (immutable chain)
        if sky_representation == "healpix_map":
            if self._sky_model.healpix is not None:
                self._sky_model = self._sky_model.activate("healpix_map")
            else:
                self._sky_model = self._sky_model.materialize_healpix(
                    nside=nside,
                    frequencies=self._frequencies_hz,
                    ref_frequency=frequency,
                )
        else:
            if self._sky_model.point is not None:
                self._sky_model = self._sky_model.activate("point_sources")
            else:
                if not allow_lossy_point_materialization:
                    raise ValueError(
                        "Requested visibility.sky_representation='point_sources' "
                        "for a HEALPix-only sky model. Set "
                        "visibility.allow_lossy_point_materialization=true to "
                        "opt in to lossy HEALPix-to-point conversion."
                    )
                self._sky_model = self._sky_model.materialize_point_sources(
                    frequency=frequency,
                    lossy=True,
                )

        # Get point source arrays for RIME calculator (only in point_sources mode)
        from rrivis.core.sky.model import SkyFormat

        if self._sky_model.mode == SkyFormat.HEALPIX:
            self._source_arrays = None
        else:
            self._source_arrays = self._sky_model.as_point_source_arrays()

        self._is_setup = True
        n_sky = self._sky_model.n_sky_elements
        sky_type = "pixels" if self._sky_model.mode == SkyFormat.HEALPIX else "sources"
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
            from rrivis.core.sky.model import SkyFormat as _SF

            # Print configuration table (after setup, needs backend/sky_model info)
            n_sky = self._sky_model.n_sky_elements if self._sky_model else 0
            _sky_mode = self._sky_model.mode if self._sky_model else _SF.POINT_SOURCES
            sky_label = (
                f"{n_sky} pixels (HEALPix)"
                if _sky_mode == _SF.HEALPIX
                else f"{n_sky} sources"
            )
            config_data = {
                "Backend": self._backend.name,
                "Precision": self._backend.precision.default
                if self._backend.precision
                else "standard",
                "Simulator": f"{self._simulator.name} ({self._simulator.complexity})",
                "Sky Mode": _sky_mode.value,
                "Antennas": len(self._antennas),
                "Baselines": len(self._baselines),
                "Sky Model": sky_label,
                "Frequencies": f"{len(self._frequencies_hz)} channels",
            }
            if hasattr(self, "_flux_unit") and self._flux_unit != "Jy":
                config_data["Flux Unit"] = self._flux_unit
            print_table("Simulation Configuration", config_data)
            console.print()  # Add spacing

        from rrivis.core.sky.model import SkyFormat

        _sky_mode = self._sky_model.mode if self._sky_model else SkyFormat.POINT_SOURCES
        print_info(f"Running visibility simulation ({_sky_mode.value} mode)...")

        # Calculate visibilities based on sky representation
        duration_seconds = self.config.get("obs_time", {}).get("duration_seconds", 1.0)
        time_step_seconds = self.config.get("obs_time", {}).get(
            "time_step_seconds", 1.0
        )

        if _sky_mode == SkyFormat.HEALPIX and self._sky_model is not None:
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
                source_arrays=self._source_arrays,
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
        n_sky = self._sky_model.n_sky_elements if self._sky_model else 0
        self._results = {
            "visibilities": visibilities,
            "frequencies": self._frequencies_hz,
            "baselines": self._baselines,
            "antennas": self._antennas,
            "source_arrays": self._source_arrays,
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
                "sky_representation": _sky_mode,
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

    def show_strip(
        self,
        *,
        lst_start_hours: float | None = None,
        lst_end_hours: float | None = None,
        background_mode: str = "gsm",
        use_lst_axis: bool = False,
        beam_lst_hours: float | None = None,
        max_point_sources: int = 1000,
        open_in_browser: bool = True,
        save_path: str | None = None,
        **kwargs,
    ):
        """Show the observable sky strip for this simulation's config.

        Produces an interactive Bokeh visualisation of the sky region
        visible to the telescope, overlaid with the configured sky model
        and beam pattern.  Can be called before or after ``run()``.

        Parameters
        ----------
        lst_start_hours, lst_end_hours : float, optional
            LST range (hours).  If not given, derived from config obs_time.
        background_mode : str
            ``"gsm"`` | ``"reference"`` | ``"none"``.
        use_lst_axis : bool
            Display x-axis as LST hours.
        beam_lst_hours : float, optional
            LST where beam is centred (defaults to strip centre).
        max_point_sources : int
            Brightest sources to show (default 1000).
        open_in_browser : bool
            Open the HTML plot in a browser.
        save_path : str, optional
            Directory to save the HTML file.
        **kwargs
            Forwarded to :class:`~rrivis.utils.diagnostics.DiagnosticsPlanner`.

        Returns
        -------
        Bokeh layout
        """
        from rrivis.utils.diagnostics import DiagnosticsPlanner, StripPlotter

        cfg = self.config

        # Extract location
        loc = cfg.get("location", {})
        lat = float(loc.get("lat", -30.72))
        lon = float(loc.get("lon", 21.43))
        height = float(loc.get("height", 1073.0))

        # Frequency (MHz)
        obs_f = cfg.get("obs_frequency", {})
        unit = obs_f.get("frequency_unit", "MHz")
        _mult = {"Hz": 1e-6, "kHz": 1e-3, "MHz": 1.0, "GHz": 1e3}
        freq_mhz = float(obs_f.get("starting_frequency", 150)) * _mult.get(unit, 1.0)

        # Time — fallback to config if LST not provided
        start_iso = None
        duration = None
        if lst_start_hours is None or lst_end_hours is None:
            obs_t = cfg.get("obs_time", {})
            start_iso = obs_t.get("start_time")
            duration = obs_t.get("duration_seconds")

        # Beam
        beams = cfg.get("beams", {})
        ant_cfg = cfg.get("antenna_layout", {})
        diameter = ant_cfg.get("all_antenna_diameter")

        planner = DiagnosticsPlanner(
            latitude_deg=lat,
            longitude_deg=lon,
            height_m=height,
            lst_start_hours=lst_start_hours,
            lst_end_hours=lst_end_hours,
            start_time_iso=start_iso,
            duration_seconds=duration,
            frequency_mhz=freq_mhz,
            beam_diameter_m=float(diameter) if diameter else None,
            beam_config=beams if beams else None,
            beam_fits_path=beams.get("beam_file"),
            beam_lst_hours=beam_lst_hours,
            max_point_sources=max_point_sources,
            background_mode=background_mode,
            **kwargs,
        )

        strip = planner.compute()
        plotter = StripPlotter(strip, use_lst_axis=use_lst_axis)
        layout = plotter.create_plot()

        if save_path or open_in_browser:
            plotter.save(
                layout,
                folder_path=save_path,
                open_in_browser=open_in_browser,
            )

        return layout

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
            n_sources=len(self._source_arrays["ra_rad"]) if self._source_arrays else 0,
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
