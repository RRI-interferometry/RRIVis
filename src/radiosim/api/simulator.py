"""High-level resolved-configuration Simulator API for RadioSim.

Examples
--------
>>> from radiosim.api import Simulator
>>> sim = Simulator.from_yaml("config.yaml")
>>> results = sim.run()
>>> sim.save("output/")

>>> sim = Simulator.from_mapping(config_data, base_dir=project_dir)
>>> results = sim.run()

>>> sim = Simulator.from_config(config_model, base_dir=project_dir)
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import astropy.units as u
import numpy as np
from astropy.constants import c as speed_of_light

from radiosim.__about__ import __version__
from radiosim.utils.logging import (
    console,
    print_header,
    print_info,
    print_success,
    print_table,
    print_warning,
)

if TYPE_CHECKING:
    from radiosim.core.beam import BeamSystem, LoadedBeamState
    from radiosim.core.instrument import (
        ResolvedAntenna,
        ResolvedBaseline,
        ResolvedInstrument,
    )
    from radiosim.core.precision import PrecisionConfig
    from radiosim.core.runtime_config import (
        ConfigurationProvenance,
        ResolvedConfiguration,
        ResolvedSimulationConfig,
    )
    from radiosim.io.beam_config import BeamsConfig
    from radiosim.io.config import (
        ExecutionConfig,
        RadioSimConfig,
        SkyModelConfig,
        VisibilityConfig,
    )
    from radiosim.io.config_resolution import SimulationOverrides
    from radiosim.io.instrument_config import (
        BaselineSelectionConfig,
        InstrumentConfig,
    )


logger = logging.getLogger(__name__)


def _runtime_loader_value(value: Any) -> Any:
    """Copy immutable resolved loader values into ordinary call arguments."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _runtime_loader_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_runtime_loader_value(item) for item in value)
    return value


_TIER3G_OBSERVABILITY_MESSAGE = (
    "Tier 3G observability migration is required for this beam mode"
)


def _guard_legacy_observability_beam(
    beams: object,
    antennas: Sequence[ResolvedAntenna],
) -> None:
    """Allow only the exact legacy-equivalent observability beam mode."""
    from radiosim.core.beam import (
        ResolvedAnalyticBeamsInput,
        ResolvedCircularApertureBeamModel,
    )

    if type(beams) is not ResolvedAnalyticBeamsInput:
        raise NotImplementedError(_TIER3G_OBSERVABILITY_MESSAGE)
    if type(beams.model.model) is not ResolvedCircularApertureBeamModel:
        raise NotImplementedError(_TIER3G_OBSERVABILITY_MESSAGE)
    diameters = {float(antenna.diameter_m) for antenna in antennas}
    if len(diameters) != 1:
        raise NotImplementedError(_TIER3G_OBSERVABILITY_MESSAGE)


def _project_direct_circular_beam(beams: object) -> dict[str, object]:
    """Project the one Tier 3B runtime-supported model into the solver shape."""
    from radiosim.core.beam import (
        ResolvedGaussianTaper,
        ResolvedParabolicSquaredTaper,
        ResolvedParabolicTaper,
    )

    model = beams.model.model
    taper = model.taper
    projected: dict[str, object] = {
        "aperture_shape": "circular",
        "taper": taper.kind,
        "feed_model": "none",
        "feed_computation": "analytical",
        "feed_params": {},
        "reflector_type": "prime_focus",
        "magnification": 1.0,
        "aperture_params": {},
    }
    if type(taper) in (
        ResolvedGaussianTaper,
        ResolvedParabolicTaper,
        ResolvedParabolicSquaredTaper,
    ):
        projected["edge_taper_dB"] = taper.edge_taper_db
    return projected


class Simulator:
    """
    High-level API for radio interferometry visibility simulation.

    This class provides a source-resolved interface for configuring and running
    direct-sum visibility simulations using the RIME (Radio Interferometer
    Measurement Equation).

    The Simulator handles all the complexity of:
    - loading antenna positions from supported layout formats;
    - generating and selecting canonical baselines;
    - loading strict sky-model requests;
    - computing the supported analytic beam;
    - selecting a resolved backend and precision policy; and
    - returning results for explicit saving or plotting.

    Analytic, FITS, and mixed per-antenna beams share one canonical BeamSystem.
    Backend selection does not promise end-to-end GPU execution.

    Parameters
    ----------
    resolved : ResolvedSimulationConfig
        Already validated, deeply immutable scientific/runtime configuration.

    Examples
    --------
    >>> # Basic usage with a YAML document
    >>> sim = Simulator.from_yaml("config.yaml")
    >>> results = sim.run()
    >>> sim.plot()
    >>> sim.save("output/")

    >>> # Programmatic mapping usage
    >>> sim = Simulator.from_mapping(config_data, base_dir=project_dir)
    >>> results = sim.run()

    See Also
    --------
    radiosim.simulator : Simulator algorithms
    radiosim.backends : Computation backends
    """

    def __init__(self, resolved: ResolvedSimulationConfig, /) -> None:
        """Construct from an already validated immutable runtime config."""
        from radiosim.core.runtime_config import ResolvedSimulationConfig

        if type(resolved) is not ResolvedSimulationConfig:
            raise TypeError("Simulator accepts only ResolvedSimulationConfig")

        self.version = __version__
        self._resolved = resolved
        self._provenance: ConfigurationProvenance | None = None
        self._results: dict[str, Any] | None = None
        self._backend_name = resolved.execution.backend_strategy
        self._precision = resolved.execution.precision
        self._simulator_name = resolved.execution.simulator
        self._backend = None
        self._simulator = None
        self._solver_instrument_view = None

        # Canonical state is assigned atomically before later setup work.
        self._instrument_state = None
        self._beam_system: BeamSystem | None = None
        self._beam_system_lock = threading.RLock()
        self._source_arrays: dict | None = None
        self._sky_model = None  # SkyModel for healpix_map representation
        self._location = None
        self._obstime = None
        self._frequencies_hz: np.ndarray | None = None
        self._wavelengths = None
        self._is_setup = False
        self._network_status = None
        self._device_resources = None
        self._offline = resolved.execution.offline

    @classmethod
    def _from_bundle(cls, bundle: ResolvedConfiguration) -> Simulator:
        simulator = cls(bundle.runtime)
        simulator._provenance = bundle.provenance.runtime_only()
        return simulator

    @staticmethod
    def _input_section(value: object) -> object:
        if hasattr(value, "model_dump"):
            return value.model_dump(mode="python")
        if isinstance(value, Mapping):
            return dict(value)
        return value

    @classmethod
    def from_yaml(
        cls,
        path: str | Path,
        *,
        overrides: SimulationOverrides | None = None,
    ) -> Simulator:
        """Load and resolve YAML using the document parent as its path base."""
        from radiosim.io.config import load_config

        return cls._from_bundle(load_config(path, overrides=overrides))

    @classmethod
    def from_config(
        cls,
        config: RadioSimConfig,
        *,
        base_dir: str | Path | None = None,
        overrides: SimulationOverrides | None = None,
    ) -> Simulator:
        """Resolve a typed input model with explicit model-source semantics."""
        from radiosim.io.config import RadioSimConfig
        from radiosim.io.config_resolution import ConfigurationSource, resolve_config

        if type(config) is not RadioSimConfig:
            raise TypeError("from_config accepts only RadioSimConfig")
        invocation_dir = Path.cwd().resolve(strict=False)
        bundle = resolve_config(
            config,
            source=ConfigurationSource.for_model(
                base_dir=base_dir,
                invocation_dir=invocation_dir,
            ),
            overrides=overrides,
        )
        return cls._from_bundle(bundle)

    @classmethod
    def from_mapping(
        cls,
        data: Mapping[str, object],
        *,
        base_dir: str | Path | None = None,
        overrides: SimulationOverrides | None = None,
    ) -> Simulator:
        """Resolve a Python mapping through the complete shared pipeline."""
        from radiosim.io.config import RadioSimConfig
        from radiosim.io.config_resolution import ConfigurationSource, resolve_config

        if isinstance(data, RadioSimConfig) or not isinstance(data, Mapping):
            raise TypeError("from_mapping accepts only a Mapping, not RadioSimConfig")
        invocation_dir = Path.cwd().resolve(strict=False)
        bundle = resolve_config(
            data,
            source=ConfigurationSource.for_mapping(
                base_dir=base_dir,
                invocation_dir=invocation_dir,
            ),
            overrides=overrides,
        )
        return cls._from_bundle(bundle)

    @classmethod
    def from_parameters(
        cls,
        *,
        instrument: InstrumentConfig,
        baseline_selection: BaselineSelectionConfig | None = None,
        channel_frequencies_hz: Sequence[float],
        start_time: str,
        duration_seconds: float = 1.0,
        time_step_seconds: float = 1.0,
        sky_model: SkyModelConfig | Mapping[str, object],
        beams: BeamsConfig | Mapping[str, object] | None = None,
        visibility: VisibilityConfig | Mapping[str, object] | None = None,
        execution: ExecutionConfig | Mapping[str, object] | None = None,
        base_dir: str | Path | None = None,
        overrides: SimulationOverrides | None = None,
    ) -> Simulator:
        """Build an explicit-Hz document and resolve it as parameter input."""
        from radiosim.io.config_resolution import ConfigurationSource, resolve_config
        from radiosim.io.instrument_config import (
            BaselineSelectionConfig,
            InstrumentConfig,
        )

        if type(instrument) is not InstrumentConfig:
            raise TypeError("instrument must be an InstrumentConfig")
        if baseline_selection is not None and (
            type(baseline_selection) is not BaselineSelectionConfig
        ):
            raise TypeError("baseline_selection must be a BaselineSelectionConfig")

        data: dict[str, object] = {
            "instrument": instrument.model_dump(mode="python"),
            "baseline_selection": (
                BaselineSelectionConfig()
                if baseline_selection is None
                else baseline_selection
            ).model_dump(mode="python"),
            "obs_time": {
                "start_time": start_time,
                "duration_seconds": duration_seconds,
                "time_step_seconds": time_step_seconds,
            },
            "obs_frequency": {
                "mode": "explicit",
                "channel_frequencies_hz": channel_frequencies_hz,
            },
            "sky_model": cls._input_section(sky_model),
        }
        for name, value in (
            ("beams", beams),
            ("visibility", visibility),
            ("execution", execution),
        ):
            if value is not None:
                data[name] = cls._input_section(value)
        invocation_dir = Path.cwd().resolve(strict=False)
        bundle = resolve_config(
            data,
            source=ConfigurationSource.for_parameters(
                base_dir=base_dir,
                invocation_dir=invocation_dir,
            ),
            overrides=overrides,
        )
        return cls._from_bundle(bundle)

    @property
    def config(self) -> ResolvedSimulationConfig:
        """Return immutable resolved scientific/runtime configuration."""
        return self._resolved

    @property
    def provenance(self) -> ConfigurationProvenance | None:
        """Return immutable source provenance for classmethod construction."""
        return self._provenance

    @property
    def results(self) -> dict[str, Any] | None:
        """Get simulation results (None if not yet run)."""
        return self._results

    @property
    def instrument(self) -> ResolvedInstrument:
        """Return the exact canonical resolved instrument."""
        if self._instrument_state is None:
            raise RuntimeError("Instrument resolution has not completed")
        return self._instrument_state.instrument

    @property
    def antennas(self) -> tuple[ResolvedAntenna, ...]:
        """Return the exact immutable canonical antenna tuple."""
        if self._instrument_state is None:
            raise RuntimeError("Instrument resolution has not completed")
        return self._instrument_state.instrument.antennas

    @property
    def baselines(self) -> tuple[ResolvedBaseline, ...]:
        """Return the exact immutable selected baseline tuple."""
        if self._instrument_state is None:
            raise RuntimeError("Instrument resolution has not completed")
        return self._instrument_state.selection.baselines

    @property
    def beam_system(self) -> BeamSystem:
        """Return the exact successfully loaded canonical BeamSystem."""
        if self._beam_system is None:
            raise RuntimeError("Beam resolution has not completed")
        return self._beam_system

    @property
    def beam_state(self) -> LoadedBeamState:
        """Return the immutable loaded beam-state snapshot."""
        if self._beam_system is None:
            raise RuntimeError("Beam resolution has not completed")
        return self._beam_system.state

    @property
    def source_arrays(self) -> dict | None:
        """Get loaded source arrays dict."""
        return self._source_arrays

    @property
    def precision(self) -> PrecisionConfig | None:
        """Get requested precision, or the backend's actual precision after setup."""
        if self._backend is not None:
            return self._backend.precision
        return self._precision

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

    def _ensure_instrument_state(self) -> None:
        """Resolve and atomically retain the canonical instrument-only state."""
        if self._instrument_state is not None:
            return

        from radiosim.core.baseline_resolution import (
            generate_resolved_baselines,
            select_resolved_baselines,
        )
        from radiosim.core.instrument_adapters import ResolvedInstrumentState
        from radiosim.core.instrument_resolution import resolve_instrument

        instrument = resolve_instrument(self._resolved.instrument)
        all_baselines = generate_resolved_baselines(instrument)
        selection = select_resolved_baselines(
            all_baselines,
            instrument=instrument,
            config=self._resolved.baseline_selection,
        )
        state = ResolvedInstrumentState(
            instrument=instrument,
            all_baselines=all_baselines,
            selection=selection,
        )
        self._instrument_state = state

    def _ensure_beam_system(self) -> None:
        """Resolve and atomically retain one complete canonical BeamSystem."""
        if self._beam_system is not None:
            return
        with self._beam_system_lock:
            if self._beam_system is not None:
                return
            if self._instrument_state is None:
                raise RuntimeError("Instrument resolution has not completed")

            from radiosim.core.beam import (
                load_beam_system,
                resolve_beam_assignments,
            )

            resolved_beams = resolve_beam_assignments(
                self._resolved.beams,
                self.instrument,
            )
            loaded = load_beam_system(
                resolved_beams,
                observation_frequencies_hz=(
                    self._resolved.frequency.channel_frequencies_hz
                ),
                precision=self._precision,
            )
            self._beam_system = loaded

    def _clear_later_runtime_state(self) -> None:
        """Clear setup state that is safe to recreate after instrument resolution."""
        self._backend = None
        self._simulator = None
        self._solver_instrument_view = None
        self._source_arrays = None
        self._sky_model = None
        self._location = None
        self._obstime = None
        self._frequencies_hz = None
        self._wavelengths = None
        self._network_status = None
        self._device_resources = None
        self._is_setup = False

    def setup(self) -> Simulator:
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
        >>> sim = Simulator.from_yaml("config.yaml")
        >>> sim.setup()
        >>> print(f"Antennas: {len(sim.antennas)}")
        >>> print(f"Baselines: {len(sim.baselines)}")
        """
        if self._is_setup:
            return self
        self._ensure_instrument_state()
        try:
            self._ensure_beam_system()
            self._clear_later_runtime_state()
            return self._setup_after_instrument_state()
        except Exception:
            self._clear_later_runtime_state()
            raise

    def _setup_after_instrument_state(self) -> Simulator:
        """Create backend, observation, and sky state after canonical resolution."""
        if self._instrument_state is None:
            raise RuntimeError("Instrument resolution has not completed")
        if self._beam_system is None:
            raise RuntimeError("Beam resolution has not completed")

        # Import core modules
        from radiosim.backends import get_backend
        from radiosim.core.observation import get_location_and_time
        from radiosim.simulator import get_simulator

        print_info("Setting up simulation...")

        # Device resource detection
        from radiosim.utils.device import get_device_resources

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

        from radiosim.core.instrument_adapters import SolverInstrumentView

        self._solver_instrument_view = SolverInstrumentView.from_state(
            self._instrument_state
        )

        logger.debug(
            "Resolved %d antennas and selected %d of %d baselines",
            len(self.antennas),
            len(self.baselines),
            len(self._instrument_state.all_baselines),
        )

        # Get location and observation time
        loc_config = self.instrument.location
        time_config = self._resolved.observation
        self._location, self._obstime = get_location_and_time(
            lat=loc_config.latitude_deg,
            lon=loc_config.longitude_deg,
            height=loc_config.height_m,
            starttime=time_config.start_time_iso,
        )

        # Create one new Simulator-owned runtime array from the immutable tuple.
        self._frequencies_hz = self._resolved.frequency.as_numpy()

        # Calculate wavelengths
        self._wavelengths = (speed_of_light / (self._frequencies_hz * u.Hz)).to(u.m)

        logger.debug(
            f"Frequencies: {len(self._frequencies_hz)} channels, "
            f"{self._frequencies_hz[0] / 1e6:.1f} - {self._frequencies_hz[-1] / 1e6:.1f} MHz"
        )

        # Get visibility configuration
        visibility_config = self._resolved.visibility
        sky_representation = visibility_config["sky_representation"]
        allow_lossy_point_materialization = visibility_config.get(
            "allow_lossy_point_materialization", False
        )

        # Network connectivity check (before sky model loading)
        from radiosim.utils.network import (
            SERVICE_DISPLAY_NAMES,
            get_network_status,
            get_required_services,
        )

        self._network_status = get_network_status(offline=self._offline)

        sky_config = self._resolved.sky_model
        required_services = get_required_services(
            {"sources": [{"kind": request.kind} for request in sky_config.sources]}
        )

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
        from radiosim.core.precision import PrecisionConfig

        _precision = (
            self._backend.precision if self._backend else PrecisionConfig.standard()
        )
        if _precision is None:
            _precision = PrecisionConfig.standard()

        # Flux unit conversion: convert user-specified flux values to canonical Jy
        flux_unit = sky_config.flux_unit
        _flux_multipliers = {"Jy": 1.0, "mJy": 1e-3, "uJy": 1e-6}
        _flux_mul = _flux_multipliers[flux_unit]
        self._flux_unit = flux_unit

        # Brightness conversion method (applies to all loaders)
        _brightness_conv = sky_config.brightness_conversion
        mixed_model_policy = sky_config.mixed_model_policy
        assume_disjoint = sky_config.assume_disjoint

        from radiosim.io.config import build_sky_region

        region = build_sky_region(sky_config.region)

        frequency = float(self._frequencies_hz[0])
        nside = next(
            (
                int(request.options["nside"])
                for request in sky_config.sources
                if request.options.get("nside") is not None
            ),
            64,
        )

        # Collect all requested sky models
        sky_models = []
        from radiosim.core.sky.registry import loader_registry

        loader_requests: list[tuple[str, dict[str, Any]]] = []

        for request in sky_config.sources:
            definition = loader_registry.definition(request.kind)
            kwargs = {
                name: _runtime_loader_value(value)
                for name, value in request.options.items()
                if value is not None
            }
            if (
                definition.path_options.get("file_glob") == "glob"
                and isinstance(kwargs.get("file_glob"), tuple)
                and "filenames" in definition.config_fields
            ):
                # Resolution has already expanded and validated the glob. Pass
                # those deterministic matches to the loader without re-globbing.
                kwargs["filenames"] = list(kwargs.pop("file_glob"))
            for flux_field in ("flux_limit", "flux_min", "flux_max"):
                if flux_field in kwargs:
                    kwargs[flux_field] *= _flux_mul
            brightness_conversion = request.brightness_conversion or _brightness_conv
            kwargs["brightness_conversion"] = brightness_conversion
            source_region = (
                build_sky_region(request.region)
                if request.region is not None
                else region
            )
            if source_region is not None:
                kwargs["region"] = source_region
            if definition.supports_healpix_map:
                kwargs["frequencies"] = self._frequencies_hz
            if request.provenance_override is not None:
                from radiosim.core.sky.containers import SkyFootprint, SkyProvenance

                provenance = _runtime_loader_value(request.provenance_override)
                footprint = provenance.get("coverage_footprint")
                if footprint is not None:
                    provenance["coverage_footprint"] = SkyFootprint(
                        nside=footprint["nside"],
                        hpx_inds=np.asarray(footprint["hpx_inds"], dtype=np.int64),
                        coordinate_frame=footprint["coordinate_frame"],
                    )
                kwargs["provenance"] = SkyProvenance(**provenance)
            loader_requests.append((request.kind, kwargs))

        if loader_requests:
            from radiosim.core.sky.operations.parallel import (
                load_models_parallel,
                recommend_executor_for_loaders,
            )

            sky_models.extend(
                load_models_parallel(
                    loader_requests,
                    max_workers=8,
                    precision=_precision,
                    strict=True,
                    executor=recommend_executor_for_loaders(loader_requests),
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

        from radiosim.core.sky.combine.pipeline import prepare_sky_model

        # Compute an approximate primary-beam FWHM for the nside advisor.
        # Uses the standard uniform-aperture rule `λ/D · 1.22` at the lowest
        # (widest-beam) frequency.  Falls back to None when no antenna-diameter
        # info is available (advisor disabled).
        beam_fwhm_rad: float | None = None
        try:
            diameters = [float(ant.diameter_m) for ant in self.antennas]
            if diameters and len(self._frequencies_hz):
                d_min = min(diameters)
                lam_max = float(speed_of_light.value) / float(self._frequencies_hz[0])
                beam_fwhm_rad = 1.22 * lam_max / d_min
        except Exception:  # noqa: BLE001  # advisor is best-effort
            beam_fwhm_rad = None

        self._sky_model = prepare_sky_model(
            sky_models,
            representation=sky_representation,
            nside=nside,
            frequency=frequency,
            frequencies=self._frequencies_hz,
            allow_lossy=allow_lossy_point_materialization,
            mixed_model_policy=mixed_model_policy,
            assume_disjoint=assume_disjoint,
            brightness_conversion=_brightness_conv,
            precision=_precision,
            backend=self._backend,
            beam_fwhm_rad=beam_fwhm_rad,
        )

        # Get point source arrays for RIME calculator (only in point_sources mode)
        from radiosim.core.sky.containers.model import SkyFormat

        sky_mode = SkyFormat(sky_representation)
        if sky_mode == SkyFormat.HEALPIX:
            self._source_arrays = None
        else:
            self._source_arrays = self._sky_model.as_point_source_arrays()

        self._is_setup = True
        n_sky = self._sky_model.n_sky_elements_for(sky_mode)
        sky_type = "pixels" if sky_mode == SkyFormat.HEALPIX else "sources"
        print_success(
            f"Setup complete: {len(self.antennas)} antennas, {len(self.baselines)} baselines, {n_sky} {sky_type}"
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
        >>> sim = Simulator.from_yaml("config.yaml")
        >>> results = sim.run()
        >>> print(results.keys())
        dict_keys(['visibilities', 'baselines', 'frequencies', ...])
        """
        if n_workers is not None:
            raise NotImplementedError(
                "run(n_workers=...): visibility worker control is not implemented and "
                "would be ignored. Omit n_workers for current serial orchestration. "
                "Target remediation: Tier 6."
            )

        t_start = time.perf_counter()

        # Set up if not already done
        if not self._is_setup:
            self.setup()

        if progress:
            print_header(
                f"RadioSim Simulator v{self.version}",
                "Radio Interferometer Visibility Simulator",
            )

        t_setup = time.perf_counter() - t_start

        if progress:
            from radiosim.core.sky.containers.model import SkyFormat as _SF

            # Print configuration table (after setup, needs backend/sky_model info)
            _sky_mode = _SF(self._resolved.visibility["sky_representation"])
            n_sky = (
                self._sky_model.n_sky_elements_for(_sky_mode) if self._sky_model else 0
            )
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
                "Antennas": len(self.antennas),
                "Baselines": len(self.baselines),
                "Sky Model": sky_label,
                "Frequencies": f"{len(self._frequencies_hz)} channels",
            }
            if hasattr(self, "_flux_unit") and self._flux_unit != "Jy":
                config_data["Flux Unit"] = self._flux_unit
            print_table("Simulation Configuration", config_data)
            console.print()  # Add spacing

        from radiosim.core.sky.containers.model import SkyFormat

        _sky_mode = SkyFormat(self._resolved.visibility["sky_representation"])
        print_info(f"Running visibility simulation ({_sky_mode.value} mode)...")

        # Calculate visibilities based on sky representation
        duration_seconds = self._resolved.observation.duration_seconds
        time_step_seconds = self._resolved.observation.time_step_seconds

        if _sky_mode == SkyFormat.HEALPIX and self._sky_model is not None:
            # Use direct HEALPix visibility calculation
            from radiosim.core.visibility_healpix import calculate_visibility_healpix

            use_pol = (
                self._sky_model is not None
                and self._sky_model.has_polarized_healpix_maps
            )

            healpix_result = calculate_visibility_healpix(
                sky_model=self._sky_model,
                instrument=self._solver_instrument_view,
                beam_system=self.beam_system,
                location=self._location,
                obstime=self._obstime,
                wavelengths=self._wavelengths,
                freqs=self._frequencies_hz,
                duration_seconds=duration_seconds,
                time_step_seconds=time_step_seconds,
                output_units="Jy",
                include_polarization=use_pol,
                backend=self._backend,
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
                    parallel_hand = vis_array[bl_idx] / 2.0
                    visibilities[bl_key] = {
                        "I": vis_array[bl_idx],
                        "XX": parallel_hand,
                        "YY": parallel_hand,
                        "XY": np.zeros_like(vis_array[bl_idx]),
                        "YX": np.zeros_like(vis_array[bl_idx]),
                    }

        else:
            # Use point source RIME calculation (original behavior)
            visibilities = self._simulator.calculate_visibilities(
                instrument=self._solver_instrument_view,
                beam_system=self.beam_system,
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
                return_correlations=True,
                jones_config=None,
            )

        t_total = time.perf_counter() - t_start

        # Compile results
        n_sky = self._sky_model.n_sky_elements_for(_sky_mode) if self._sky_model else 0
        if self._instrument_state is None:
            raise RuntimeError("Instrument resolution has not completed")
        instrument_resolution = self.instrument.to_snapshot()
        instrument_resolution["baseline_selection"] = (
            self._instrument_state.selection.to_snapshot()
        )

        self._results = {
            "visibilities": visibilities,
            "frequencies": self._frequencies_hz,
            "baselines": self.baselines,
            "antennas": self.antennas,
            "source_arrays": self._source_arrays,
            "sky_model": self._sky_model,
            "location": self._location,
            "obstime": self._obstime,
            "wavelengths": self._wavelengths,
            "timing": {"total": t_total, "setup": t_setup},
            "metadata": {
                "version": self.version,
                "backend": self._backend.name,
                "requested_backend": self._resolved.execution.backend_strategy,
                "precision": self._backend.precision.model_dump(mode="json")
                if self._backend.precision
                else None,
                "requested_precision": self._resolved.execution.precision.model_dump(
                    mode="json"
                ),
                "simulator": self._simulator.name,
                "sky_representation": _sky_mode.value,
                "n_antennas": len(self.antennas),
                "n_baselines": len(self.baselines),
                "n_sky_elements": n_sky,
                "n_frequencies": len(self._frequencies_hz),
                "config": self._resolved.to_json_safe(),
                "instrument_resolution": instrument_resolution,
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
            from radiosim.visualization.bokeh_plots import (
                plot_antenna_layout,
                plot_antenna_layout_3d_plotly,
                plot_heatmaps,
                plot_modulus_vs_frequency,
                plot_visibility,
            )

            # 2D antenna layout
            plot_antenna_layout(
                self.antennas,
                plotting=backend,
                save_simulation_data=output_dir is not None,
                folder_path=str(output_dir) if output_dir else None,
                open_in_browser=show,
            )

            # 3D antenna layout (Plotly)
            plot_antenna_layout_3d_plotly(
                self.antennas,
                save_simulation_data=output_dir is not None,
                folder_path=str(output_dir) if output_dir else None,
                open_in_browser=show,
            )

        # Visibility time-series plots require multi-time data
        if plot_type in ["visibility", "heatmap", "all"]:
            # Check if we have multi-time data
            from radiosim.core.visibility import calculate_modulus_phase

            moduli, phases = calculate_modulus_phase(self._results["visibilities"])

            # Check if data has time dimension
            first_baseline = list(moduli.keys())[0]
            has_time_data = moduli[first_baseline].ndim == 2  # Shape (n_times, n_freq)

            if has_time_data and moduli[first_baseline].shape[0] > 1:
                # Multi-time data - generate time-series plots
                logger.debug(f"Generating {plot_type} plots with multi-time data")

                # Generate time points array
                duration_sec = self._resolved.observation.duration_seconds
                time_step_sec = self._resolved.observation.time_step_seconds
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
                        baselines=self.baselines,
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
                        baselines=self.baselines,
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
                        baselines=self.baselines,
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

    def plot_observability(
        self,
        *,
        lst_start_hours: float | None = None,
        lst_end_hours: float | None = None,
        x_axis: str = "ra",
        background_layer: str = "diffuse",
        footprint_model: str = "swept_beam",
        mode: str = "summary",
        snapshot_step_seconds: float = 3600.0,
        footprint_step_seconds: float = 60.0,
        beam_reference: str | float = "midpoint",
        max_point_sources: int = 1000,
        top_n_sources: int = 5,
        show_source_colorbar: bool = False,
        color_scale: str = "log",
        open_in_browser: bool = True,
        save_path: str | None = None,
        **kwargs,
    ):
        """Render a sky-visibility view for this simulation's config.

        Produces a geometry-aware sky-visibility plot with optional diffuse
        background, point-source metrics, and beam overlay. Can be called
        before or after :meth:`run`.

        Parameters
        ----------
        lst_start_hours, lst_end_hours : float, optional
            LST range (hours).  If not given, derived from config obs_time.
        x_axis : {"ra", "lst"}
            X-axis convention: wrapped RA degrees or sidereal hours.
        background_layer : {"diffuse", "none"}
            Whether to draw the diffuse HEALPix background.
        footprint_model : {"swept_beam", "rectangular_approx"}
            Visibility-footprint model for the summary overlay.
        mode : {"summary", "snapshots"}
            Plot a time-swept summary or a snapshot grid.
        snapshot_step_seconds : float
            Snapshot cadence used when ``mode="snapshots"``.
        footprint_step_seconds : float
            Sampling cadence used for the summary footprint and source metrics.
        beam_reference : {"midpoint", "start", "end"} or float
            Beam-reference choice for the summary overlay. A float is treated
            as an LST hour.
        max_point_sources : int
            Brightest point sources to include in the plot.
        top_n_sources : int
            Number of visible sources to rank and label.
        show_source_colorbar : bool
            Draw a flux colorbar for point sources.
        color_scale : {"log", "linear"}
            Flux color scaling for point sources.
        open_in_browser : bool
            Open the HTML plot in a browser.
        save_path : str, optional
            Directory to save the HTML file.
        **kwargs
            Forwarded to :class:`~radiosim.core.observability.ObservabilityPlanner`.

        Returns
        -------
        Bokeh layout
        """
        self._ensure_instrument_state()
        _guard_legacy_observability_beam(self._resolved.beams, self.antennas)
        self._ensure_beam_system()

        from radiosim.core.observability import ObservabilityPlanner

        distinct_diameters = tuple(
            sorted({float(antenna.diameter_m) for antenna in self.antennas})
        )
        assert len(distinct_diameters) == 1
        diameter = distinct_diameters[0]

        from radiosim.visualization.observability import ObservabilityBokehRenderer

        location = self.instrument.location
        lat = location.latitude_deg
        lon = location.longitude_deg
        height = location.height_m

        # Frequency (MHz), taken directly from the exact resolved channel tuple.
        freq_mhz = self._resolved.frequency.channel_frequencies_hz[0] / 1e6

        # Time — fallback to config if LST not provided
        start_iso = None
        duration = None
        if lst_start_hours is None or lst_end_hours is None:
            start_iso = self._resolved.observation.start_time_iso
            duration = self._resolved.observation.duration_seconds

        # Beam
        beam_config = _project_direct_circular_beam(self._resolved.beams)

        planner = ObservabilityPlanner(
            latitude_deg=lat,
            longitude_deg=lon,
            height_m=height,
            lst_start_hours=lst_start_hours,
            lst_end_hours=lst_end_hours,
            start_time_iso=start_iso,
            duration_seconds=duration,
            frequency_mhz=freq_mhz,
            field_radius_deg=kwargs.pop("field_radius_deg", None),
            beam_diameter_m=diameter,
            beam_config=beam_config,
            beam_fits_path=None,
            beam_reference=beam_reference,
            sky_model=getattr(self, "_sky_model", None),
            x_axis=x_axis,
            background_layer=background_layer,
            footprint_model=footprint_model,
            mode=mode,
            snapshot_step_seconds=snapshot_step_seconds,
            footprint_step_seconds=footprint_step_seconds,
            max_point_sources=max_point_sources,
            top_n_sources=top_n_sources,
            **kwargs,
        )

        plan = planner.build()
        renderer = ObservabilityBokehRenderer(
            plan,
            show_source_colorbar=show_source_colorbar,
            color_scale=color_scale,
        )
        layout = renderer.create_plot()

        if save_path or open_in_browser:
            renderer.save(
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
        filename : str, optional
            Output filename stem (without extension). Defaults to
            "visibilities" for HDF5/JSON and "simulation" for MS.

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

        from radiosim.io.writers import save_visibilities_hdf5

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.debug(f"Saving results to {output_dir}...")

        if format.lower() == "hdf5":
            stem = filename or "visibilities"
            output_path = output_dir / f"{stem}.h5"

            if output_path.exists() and not overwrite:
                raise FileExistsError(
                    f"{output_path} already exists. Use overwrite=True to overwrite."
                )

            # Calculate time points for observation
            duration_sec = self._resolved.observation.duration_seconds
            time_step_sec = self._resolved.observation.time_step_seconds
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

            stem = filename or "visibilities"
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
                json.dump(json_data, f, indent=2, allow_nan=False)

            logger.debug(f"Saved JSON to {output_path}")
            return output_path

        elif format.lower() == "ms":
            from radiosim.io.measurement_set import (
                CASACORE_AVAILABLE,
                PYUVDATA_AVAILABLE,
                write_ms,
            )

            if not (CASACORE_AVAILABLE and PYUVDATA_AVAILABLE):
                raise ImportError(
                    "Measurement Set support not available.\n"
                    "Install with: pip install radiosim[ms]\n"
                    "Or: pip install python-casacore"
                )

            stem = filename or "simulation"
            output_path = output_dir / f"{stem}.ms"

            if self._instrument_state is None:
                raise RuntimeError("Instrument resolution has not completed")

            write_ms(
                output_path=output_path,
                visibilities=self._results["visibilities"],
                frequencies=self._results["frequencies"],
                instrument=self.instrument,
                selection=self._instrument_state.selection,
                obstime=self._results["obstime"],
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
        >>> sim = Simulator.from_yaml("config.yaml")
        >>> sim.setup()
        >>> mem = sim.get_memory_estimate()
        >>> print(f"Estimated memory: {mem['total_human']}")
        """
        if not self._is_setup:
            self.setup()

        # Get base memory estimate
        estimate = self._simulator.get_memory_estimate(
            n_antennas=len(self.antennas),
            n_baselines=len(self.baselines),
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
