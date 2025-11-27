"""High-level Simulator API for RRIvis.

Provides a clean, user-friendly interface for running visibility simulations
in Python scripts and Jupyter notebooks.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import numpy as np

from rrivis.__about__ import __version__


class Simulator:
    """
    High-level API for radio interferometry visibility simulation.

    This class provides a simple interface for configuring and running
    visibility simulations using the RIME (Radio Interferometer
    Measurement Equation).

    Examples:
        >>> # Basic usage with config file
        >>> sim = Simulator.from_config("config.yaml")
        >>> results = sim.run()
        >>> sim.plot()
        >>> sim.save("output/")

        >>> # Programmatic usage
        >>> sim = Simulator(
        ...     antenna_layout="HERA65.csv",
        ...     frequencies=[100, 150, 200],
        ...     sky_model="gleam",
        ... )
        >>> results = sim.run()

    Attributes:
        config: Configuration dictionary
        results: Simulation results (after run())
        version: RRIvis version string
    """

    def __init__(
        self,
        antenna_layout: Optional[Union[str, Path]] = None,
        frequencies: Optional[List[float]] = None,
        sky_model: Optional[str] = None,
        location: Optional[Dict[str, float]] = None,
        start_time: Optional[str] = None,
        backend: str = "auto",
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the Simulator.

        Args:
            antenna_layout: Path to antenna positions file
            frequencies: List of frequencies in MHz
            sky_model: Sky model name ("test", "gleam", "gsm")
            location: Observatory location dict with lat, lon, height
            start_time: Observation start time (ISO format)
            backend: Computation backend ("auto", "numpy", "jax", "numba")
            config: Full configuration dictionary (overrides other args)
        """
        self.version = __version__
        self._results = None
        self._backend = backend

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
            }

        if frequencies:
            config["obs_frequency"] = {
                "starting_frequency": min(frequencies),
                "frequency_bandwidth": max(frequencies) - min(frequencies),
                "n_channels": len(frequencies),
            }

        if sky_model:
            config["sky_model"] = {
                "test_sources": {"use_test_sources": sky_model == "test"},
                "gleam": {"use_gleam": sky_model == "gleam"},
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

        Args:
            config_path: Path to YAML configuration file

        Returns:
            Configured Simulator instance
        """
        import yaml

        config_path = Path(config_path)
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        return cls(config=config)

    @property
    def results(self) -> Optional[Dict[str, Any]]:
        """Get simulation results (None if not yet run)."""
        return self._results

    def run(
        self,
        progress: bool = True,
        n_workers: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Run the visibility simulation.

        Args:
            progress: Show progress bar (default: True)
            n_workers: Number of parallel workers (default: auto)

        Returns:
            Dictionary containing simulation results:
            - visibilities: Complex visibility data
            - baselines: Baseline information
            - frequencies: Frequency array
            - times: Time array
            - metadata: Additional metadata
        """
        # Import here to avoid circular imports
        # This will be connected to the actual simulation in Phase 2
        from rrivis.utils.logging import get_logger

        logger = get_logger("rrivis.api")
        logger.info(f"Starting simulation with backend: {self._backend}")

        # Placeholder for actual simulation
        # Full implementation will call core.visibility.calculate_visibility
        self._results = {
            "visibilities": None,
            "baselines": None,
            "frequencies": None,
            "times": None,
            "metadata": {
                "version": self.version,
                "backend": self._backend,
                "config": self.config,
            },
        }

        logger.info("Simulation complete")
        return self._results

    def plot(
        self,
        plot_type: str = "all",
        output_dir: Optional[Union[str, Path]] = None,
        backend: str = "bokeh",
    ) -> None:
        """
        Generate visualization plots.

        Args:
            plot_type: Type of plot ("all", "antenna", "visibility", "uv")
            output_dir: Directory to save plots (None for display only)
            backend: Plotting backend ("bokeh", "matplotlib")
        """
        if self._results is None:
            raise RuntimeError("No results to plot. Run simulation first.")

        from rrivis.utils.logging import get_logger

        logger = get_logger("rrivis.api")
        logger.info(f"Generating {plot_type} plots with {backend} backend")

        # Placeholder - will connect to visualization module
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Plots saved to {output_dir}")

    def save(
        self,
        output_dir: Union[str, Path],
        format: str = "hdf5",
        overwrite: bool = False,
    ) -> Path:
        """
        Save simulation results to disk.

        Args:
            output_dir: Output directory path
            format: Output format ("hdf5", "json", "ms")
            overwrite: Overwrite existing files (default: False)

        Returns:
            Path to saved output file
        """
        if self._results is None:
            raise RuntimeError("No results to save. Run simulation first.")

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        from rrivis.utils.logging import get_logger

        logger = get_logger("rrivis.api")

        if format == "hdf5":
            output_path = output_dir / "visibilities.h5"
            logger.info(f"Saving results to {output_path}")
            # Placeholder - will connect to io.writers
        elif format == "json":
            output_path = output_dir / "visibilities.json"
            logger.info(f"Saving results to {output_path}")
        elif format == "ms":
            output_path = output_dir / "simulation.ms"
            logger.info(f"Saving results to {output_path}")
        else:
            raise ValueError(f"Unknown format: {format}")

        return output_path

    def __repr__(self) -> str:
        """String representation of Simulator."""
        status = "configured" if self._results is None else "completed"
        return f"<Simulator v{self.version} [{status}]>"
