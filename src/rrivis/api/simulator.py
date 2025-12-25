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

from pathlib import Path
from typing import Any, Dict, List, Optional, Union, TYPE_CHECKING
import logging

import numpy as np
import astropy.units as u
from astropy.constants import c as speed_of_light

from rrivis.__about__ import __version__

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
        antenna_layout: Optional[Union[str, Path]] = None,
        frequencies: Optional[List[float]] = None,
        sky_model: Optional[str] = None,
        location: Optional[Dict[str, float]] = None,
        start_time: Optional[str] = None,
        backend: str = "auto",
        precision: Optional[Union["PrecisionConfig", str]] = None,
        simulator: str = "rime",
        config: Optional[Dict[str, Any]] = None,
    ):
        """Initialize the Simulator."""
        self.version = __version__
        self._results: Optional[Dict[str, Any]] = None
        self._backend_name = backend
        self._precision = precision
        self._simulator_name = simulator
        self._backend = None
        self._simulator = None

        # Internal state (populated by setup())
        self._antennas: Optional[Dict] = None
        self._baselines: Optional[Dict] = None
        self._sources: Optional[List] = None
        self._location = None
        self._obstime = None
        self._frequencies_hz: Optional[np.ndarray] = None
        self._wavelengths = None
        self._hpbw_per_antenna: Optional[Dict] = None
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
        antenna_layout: Optional[Union[str, Path]] = None,
        frequencies: Optional[List[float]] = None,
        sky_model: Optional[str] = None,
        location: Optional[Dict[str, float]] = None,
        start_time: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Build configuration dictionary from arguments."""
        config: Dict[str, Any] = {}

        if antenna_layout:
            config["antenna_layout"] = {
                "antenna_positions_file": str(antenna_layout),
                "antenna_file_format": "rrivis",
                "all_antenna_type": "parabolic",
                "all_antenna_diameter": 14.0,
            }

        if frequencies:
            freq_array = np.array(frequencies)
            config["obs_frequency"] = {
                "starting_frequency": float(np.min(freq_array)),
                "frequency_bandwidth": float(np.max(freq_array) - np.min(freq_array)),
                "frequency_interval": float(np.mean(np.diff(freq_array))) if len(freq_array) > 1 else 1.0,
                "frequency_unit": "MHz",
            }

        if sky_model:
            config["sky_model"] = {
                "test_sources": {"use_test_sources": sky_model in ["test", "test_sources"]},
                "gleam": {"use_gleam": sky_model == "gleam"},
                "gsm_healpix": {"use_gsm": sky_model == "gsm"},
            }

        if location:
            config["location"] = location

        if start_time:
            config["obs_time"] = {"start_time": start_time}

        return config

    @classmethod
    def from_config(cls, config_path: Union[str, Path]) -> "Simulator":
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
        >>> sim = Simulator.from_config("configs/01_parabolic.yaml")
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
    def results(self) -> Optional[Dict[str, Any]]:
        """Get simulation results (None if not yet run)."""
        return self._results

    @property
    def antennas(self) -> Optional[Dict]:
        """Get loaded antenna dictionary."""
        return self._antennas

    @property
    def baselines(self) -> Optional[Dict]:
        """Get generated baselines dictionary."""
        return self._baselines

    @property
    def sources(self) -> Optional[List]:
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
        from rrivis.core.antenna import read_antenna_positions
        from rrivis.core.baseline import generate_baselines
        from rrivis.core.source import get_sources
        from rrivis.core.observation import get_location_and_time
        from rrivis.core.beams import calculate_hpbw_radians
        from rrivis.backends import get_backend
        from rrivis.simulator import get_simulator

        logger.info("Setting up simulation...")

        # Initialize backend with precision
        self._backend = get_backend(self._backend_name, precision=self._precision)
        logger.info(f"Using backend: {self._backend.name}")
        if self._backend.precision:
            logger.info(f"Precision: {self._backend.precision.default}")

        # Initialize simulator
        self._simulator = get_simulator(self._simulator_name)
        logger.info(f"Using simulator: {self._simulator.name} ({self._simulator.complexity})")

        # Load antenna positions
        antenna_config = self.config.get("antenna_layout", {})
        antenna_file = antenna_config.get("antenna_positions_file")

        if antenna_file:
            self._antennas = read_antenna_positions(
                antenna_file,
                format_type=antenna_config.get("antenna_file_format", "rrivis"),
            )
            logger.info(f"Loaded {len(self._antennas)} antennas from {antenna_file}")
        else:
            raise ValueError("No antenna_positions_file specified in config")

        # Set up beam types and responses for each antenna
        beam_config = self.config.get("beams", {})
        all_beam_response = beam_config.get("all_beam_response", "gaussian")
        antenna_type = antenna_config.get("all_antenna_type", "parabolic")
        antenna_diameter = antenna_config.get("all_antenna_diameter", 14.0)

        # Set diameter and beam info for each antenna
        beams_per_antenna = {}
        beam_response_per_antenna = {}
        for ant_id in self._antennas:
            self._antennas[ant_id]["diameter"] = antenna_diameter
            beams_per_antenna[self._antennas[ant_id]["Number"]] = antenna_type
            beam_response_per_antenna[self._antennas[ant_id]["Number"]] = all_beam_response

        # Generate baselines
        self._baselines = generate_baselines(
            self._antennas,
            beams_per_antenna,
            beam_response_per_antenna,
        )
        logger.info(f"Generated {len(self._baselines)} baselines")

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
            start_freq * multiplier,
            (start_freq + bandwidth) * multiplier,
            n_channels
        )

        # Calculate wavelengths
        self._wavelengths = (speed_of_light / (self._frequencies_hz * u.Hz)).to(u.m)

        logger.info(f"Frequencies: {len(self._frequencies_hz)} channels, "
                    f"{self._frequencies_hz[0]/1e6:.1f} - {self._frequencies_hz[-1]/1e6:.1f} MHz")

        # Calculate HPBW per antenna
        # calculate_hpbw_radians(frequencies_hz, dish_diameter) returns array of HPBW in radians
        self._hpbw_per_antenna = {}
        for ant_id, ant_data in self._antennas.items():
            ant_num = ant_data["Number"]
            diameter = ant_data.get("diameter", antenna_diameter)

            # Calculate HPBW for all frequencies at once
            hpbw_array = calculate_hpbw_radians(self._frequencies_hz, diameter)
            self._hpbw_per_antenna[ant_num] = hpbw_array

        # Load sources
        sky_config = self.config.get("sky_model", {})
        test_config = sky_config.get("test_sources", {})
        gleam_config = sky_config.get("gleam", {})
        gsm_config = sky_config.get("gsm_healpix", {})

        self._sources, _ = get_sources(
            use_test_sources=test_config.get("use_test_sources", False),
            use_gleam=gleam_config.get("use_gleam", False),
            use_gsm=gsm_config.get("use_gsm", False),
            frequency=float(self._frequencies_hz[0]),
            flux_limit=test_config.get("flux_limit", 50.0),
            nside=gsm_config.get("nside", 32),
            num_sources=test_config.get("num_sources", 100),
        )
        logger.info(f"Loaded {len(self._sources)} sources")

        self._is_setup = True
        return self

    def run(
        self,
        progress: bool = True,
        n_workers: Optional[int] = None,
    ) -> Dict[str, Any]:
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
        # Set up if not already done
        if not self._is_setup:
            self.setup()

        if progress:
            print(f"RRIvis Simulator v{self.version}")
            print(f"  Backend: {self._backend.name}")
            if self._backend.precision:
                print(f"  Precision: {self._backend.precision.default}")
            print(f"  Simulator: {self._simulator.name}")
            print(f"  Complexity: {self._simulator.complexity}")
            print(f"  Antennas: {len(self._antennas)}")
            print(f"  Baselines: {len(self._baselines)}")
            print(f"  Sources: {len(self._sources)}")
            print(f"  Frequencies: {len(self._frequencies_hz)}")

        logger.info("Running visibility simulation...")

        # Get beam pattern configuration
        beam_config = self.config.get("beams", {})
        beam_pattern_per_antenna = {}
        for ant_id, ant_data in self._antennas.items():
            beam_pattern_per_antenna[ant_data["Number"]] = beam_config.get(
                "all_beam_response", "gaussian"
            )

        beam_pattern_params = {
            "cosine_taper_exponent": beam_config.get("cosine_taper_exponent", 1.0),
            "exponential_taper_dB": beam_config.get("exponential_taper_dB", 10.0),
        }

        # Calculate visibilities using the simulator
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
            hpbw_per_antenna=self._hpbw_per_antenna,
            # Optional kwargs
            beam_manager=self._beam_manager,
            beam_pattern_per_antenna=beam_pattern_per_antenna,
            beam_pattern_params=beam_pattern_params,
            return_correlations=True,
        )

        # Compile results
        self._results = {
            "visibilities": visibilities,
            "frequencies": self._frequencies_hz,
            "baselines": self._baselines,
            "antennas": self._antennas,
            "sources": self._sources,
            "location": self._location,
            "obstime": self._obstime,
            "wavelengths": self._wavelengths,
            "metadata": {
                "version": self.version,
                "backend": self._backend.name,
                "precision": self._backend.precision.model_dump() if self._backend.precision else None,
                "simulator": self._simulator.name,
                "n_antennas": len(self._antennas),
                "n_baselines": len(self._baselines),
                "n_sources": len(self._sources),
                "n_frequencies": len(self._frequencies_hz),
                "config": self.config,
            },
        }

        if progress:
            print("Simulation complete!")

        logger.info("Simulation complete")
        return self._results

    def plot(
        self,
        plot_type: str = "all",
        output_dir: Optional[Union[str, Path]] = None,
        backend: str = "bokeh",
        show: bool = True,
    ) -> None:
        """
        Generate visualization plots.

        Parameters
        ----------
        plot_type : str, optional
            Type of plot to generate:
            - "all": All available plots
            - "antenna": Antenna layout only
            - "visibility": Visibility amplitude/phase (requires multi-time data)
            - "heatmap": Visibility heatmaps (requires multi-time data)
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
        Currently only antenna layout plotting is fully supported for single-time
        simulations. For visibility vs time plots, run multiple time steps.

        Examples
        --------
        >>> sim.run()
        >>> sim.plot(plot_type="antenna")
        >>> sim.plot(plot_type="antenna", output_dir="plots/")
        """
        if self._results is None:
            raise RuntimeError("No results to plot. Run simulation first with sim.run()")

        logger.info(f"Generating {plot_type} plots with {backend}")

        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

        # Antenna layout plot
        if plot_type in ["antenna", "all"]:
            from rrivis.visualization.bokeh_plots import plot_antenna_layout

            plot_antenna_layout(
                self._antennas,
                plotting=backend,
                save_simulation_data=output_dir is not None,
                folder_path=str(output_dir) if output_dir else None,
                open_in_browser=show,
            )

        # Visibility time-series plots require multi-time data
        if plot_type in ["visibility", "heatmap"]:
            # Current single-time simulation doesn't have time-series data
            # The visualization functions expect moduli_over_time and phases_over_time
            # which require multiple time steps
            logger.warning(
                f"Plot type '{plot_type}' requires multi-time simulation data. "
                "Current implementation runs single time step. "
                "Use plot_type='antenna' for single-time results, or run "
                "multiple time steps for time-series visualization."
            )
            print(
                f"Warning: '{plot_type}' plots require multi-time simulation data.\n"
                "Currently only 'antenna' layout plots are available for single-time runs."
            )

        if output_dir:
            logger.info(f"Plots saved to {output_dir}")

    def save(
        self,
        output_dir: Union[str, Path],
        format: str = "hdf5",
        overwrite: bool = False,
        telescope_name: Optional[str] = None,
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
            raise RuntimeError("No results to save. Run simulation first with sim.run()")

        from rrivis.io.writers import save_visibilities_hdf5

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Saving results to {output_dir}")

        if format.lower() == "hdf5":
            output_path = output_dir / "visibilities.h5"

            if output_path.exists() and not overwrite:
                raise FileExistsError(
                    f"{output_path} already exists. Use overwrite=True to overwrite."
                )

            # Prepare time array (single time point for now)
            time_mjd = np.array([self._obstime.mjd]) if self._obstime else np.array([0.0])

            # Convert visibility format from Dict[tuple, Dict] to Dict[tuple, list]
            # The writer expects list of arrays for time stacking
            # Our format is {(ant1, ant2): {"XX": array, "I": array, ...}}
            # Convert to {(ant1, ant2): [array]} using Stokes I
            visibilities_for_writer = {}
            for bl_key, vis_dict in self._results["visibilities"].items():
                # Use Stokes I visibility (or XX if I not available)
                if isinstance(vis_dict, dict):
                    vis_array = vis_dict.get("I", vis_dict.get("XX", np.array([])))
                    visibilities_for_writer[bl_key] = [vis_array]
                else:
                    visibilities_for_writer[bl_key] = [vis_dict]

            save_visibilities_hdf5(
                output_path=output_path,
                visibilities=visibilities_for_writer,
                frequencies=self._results["frequencies"],
                time_points_mjd=time_mjd,
                metadata=self._results["metadata"],
            )

            logger.info(f"Saved HDF5 to {output_path}")
            return output_path

        elif format.lower() == "json":
            import json

            output_path = output_dir / "visibilities.json"

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

            logger.info(f"Saved JSON to {output_path}")
            return output_path

        elif format.lower() == "ms":
            from rrivis.io import write_ms, MS_AVAILABLE

            if not MS_AVAILABLE:
                raise ImportError(
                    "Measurement Set support not available.\n"
                    "Install with: pip install rrivis[ms]\n"
                    "Or: pip install python-casacore"
                )

            output_path = output_dir / "simulation.ms"

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

            logger.info(f"Saved MS to {output_path}")
            return output_path

        else:
            raise ValueError(
                f"Unknown format: {format}. Supported: 'hdf5', 'json', 'ms'"
            )

    def get_memory_estimate(self) -> Dict[str, Any]:
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
                estimate["total_bytes"] = int(estimate["total_bytes"] * precision_factor)
                # Update human-readable string
                total_bytes = estimate["total_bytes"]
                if total_bytes > 1e9:
                    estimate["total_human"] = f"{total_bytes/1e9:.1f} GB"
                elif total_bytes > 1e6:
                    estimate["total_human"] = f"{total_bytes/1e6:.1f} MB"
                else:
                    estimate["total_human"] = f"{total_bytes/1e3:.1f} KB"
        else:
            estimate["precision_factor"] = 1.0

        return estimate

    def __repr__(self) -> str:
        """String representation of Simulator."""
        status = "configured" if self._results is None else "completed"
        backend = self._backend.name if self._backend else self._backend_name
        return f"<Simulator v{self.version} [{status}] backend={backend}>"
