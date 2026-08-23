"""High-level resolved-configuration Simulator API for RadioSim.

Examples
--------
Every entry point needs a configuration document or an already resolved
configuration model, so these are illustrative rather than executed:

.. code-block:: python

    from radiosim.api import Simulator

    sim = Simulator.from_yaml("config.yaml")
    result = sim.run()
    assert result is sim.result

    sim = Simulator.from_mapping(config_data, base_dir=project_dir)
    result = sim.run()

    sim = Simulator.from_config(config_model, base_dir=project_dir)

``examples/scripts/simple_simulation.py`` is the executed counterpart: it
builds a simulator with :meth:`Simulator.from_parameters` and runs offline.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

import astropy.units as u
import numpy as np
from astropy.constants import c as speed_of_light

from radiosim.__about__ import __version__
from radiosim.io.result_format import ResultFormat
from radiosim.utils.logging import (
    console,
    print_header,
    print_info,
    print_success,
    print_table,
    print_warning,
)

if TYPE_CHECKING:
    from bokeh.models import UIElement

    from radiosim.core.beam import BeamSystem, LoadedBeamState
    from radiosim.core.instrument import (
        ResolvedAntenna,
        ResolvedBaseline,
        ResolvedInstrument,
    )
    from radiosim.core.jones_terms import ResolvedJonesTerms
    from radiosim.core.observability import ObservabilityPlan
    from radiosim.core.precision import PrecisionConfig
    from radiosim.core.receptor import ResolvedReceptorSet
    from radiosim.core.result import SimulationResult
    from radiosim.core.runtime_config import (
        ConfigurationProvenance,
        ResolvedConfiguration,
        ResolvedSimulationConfig,
    )
    from radiosim.core.sky.containers.point import SourceArrays
    from radiosim.core.sky.operations.parallel import LoaderExecutionRecord
    from radiosim.io.beam_config import BeamsConfig
    from radiosim.io.config import (
        ExecutionConfig,
        RadioSimConfig,
        SkyModelConfig,
        VisibilityConfig,
    )
    from radiosim.io.config_resolution import SimulationOverrides
    from radiosim.io.instrument_config import (
        AntennaReference,
        BaselineSelectionConfig,
        InstrumentConfig,
    )
    from radiosim.simulator.base import SkySolveRequest
    from radiosim.utils.healpix import BeamSamplingRequirement


logger = logging.getLogger(__name__)

#: Declared canonical plot files, kept here so importing the API never imports
#: Bokeh, Plotly, or any renderer module.
_PLOT_FILENAMES: dict[str, str] = {
    "antenna": "antenna_layout.html",
    "visibility": "visibility-phase-lsts.html",
    "heatmap": "heatmaps-freq-time.html",
    "frequency": "modulus-phase-freq.html",
}

_PLOT_FAMILIES: dict[str, tuple[str, ...]] = {
    "all": ("antenna", "visibility", "heatmap", "frequency"),
    "antenna": ("antenna",),
    "visibility": ("visibility",),
    "heatmap": ("heatmap",),
    "frequency": ("frequency",),
}


def _runtime_loader_value(value: Any) -> Any:
    """Copy immutable resolved loader values into ordinary call arguments."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _runtime_loader_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_runtime_loader_value(item) for item in value)
    return value


def _format_memory_bytes(value: int) -> str:
    """Format one nonnegative byte count using binary unit boundaries."""
    amount = float(value)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if amount < 1024:
            return f"{amount:.1f} {unit}"
        amount /= 1024
    return f"{amount:.1f} PB"


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
    Illustrative rather than executed; each form needs a configuration
    document or mapping supplied by the caller:

    .. code-block:: python

        # Basic usage with a YAML document
        sim = Simulator.from_yaml("config.yaml")
        result = sim.run()
        assert result is sim.result

        # Programmatic mapping usage
        sim = Simulator.from_mapping(config_data, base_dir=project_dir)
        result = sim.run()

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
        self._result: SimulationResult | None = None
        self._backend_name = resolved.execution.backend_strategy
        self._precision = resolved.execution.precision
        self._simulator_name = resolved.execution.simulator
        self._backend = None
        self._simulator = None
        self._solver_instrument_view = None
        #: The Section 3.1 exact-turn ERA grid of a full-sidereal m-mode run,
        #: built once and passed by identity to the selected strategy.
        self._era_grid: Any = None

        # Canonical state is assigned atomically before later setup work.
        self._instrument_state = None
        self._receptor_set: ResolvedReceptorSet | None = None
        self._jones_terms: ResolvedJonesTerms | None = None
        self._beam_system: BeamSystem | None = None
        self._beam_system_lock = threading.RLock()
        self._source_arrays: SourceArrays | None = None
        self._sky_model = None  # SkyModel for healpix_map representation
        self._location: Any | None = None
        self._obstime = None
        self._frequencies_hz: np.ndarray | None = None
        self._wavelengths = None
        self._is_setup = False
        self._network_status = None
        self._device_resources = None
        self._offline = resolved.execution.offline
        self._loader_execution: LoaderExecutionRecord | None = None

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
        channel_widths_hz: Sequence[float],
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
                "channel_widths_hz": channel_widths_hz,
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
    def result(self) -> SimulationResult | None:
        """Return the last successfully published canonical result."""
        return self._result

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
    def receptors(self) -> ResolvedReceptorSet:
        """Return the exact canonical resolved receptor set."""
        if self._receptor_set is None:
            raise RuntimeError("Receptor resolution has not completed")
        return self._receptor_set

    @property
    def jones_terms(self) -> ResolvedJonesTerms:
        """Return the exact canonical resolved Jones-term inventory.

        Empty when the configuration has no ``jones:`` section.  This means no
        optional Jones term is enabled; always-present beam, receptor,
        reporting-basis, and geometric factors still apply.
        """
        if self._jones_terms is None:
            raise RuntimeError("Jones resolution has not completed")
        return self._jones_terms

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

    @property
    def loader_execution(self) -> LoaderExecutionRecord | None:
        """Get the loader worker policy that actually ran during setup.

        Returns None before ``setup()``, and also after a setup that needed no
        loader (a sky model supplied by another route).
        """
        return self._loader_execution

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

    def _ensure_receptor_set(self) -> None:
        """Resolve and atomically retain the canonical receptor set.

        Receptor resolution is pure and runs after instrument resolution and
        before any beam load, backend selection, device transfer, filesystem
        access, or network access (``Tier5ReceptorFeedPlan.md`` Section 25.2).
        """
        if self._receptor_set is not None:
            return
        if self._instrument_state is None:
            raise RuntimeError("Instrument resolution has not completed")

        from radiosim.core.receptor import resolve_receptors

        self._receptor_set = resolve_receptors(
            self._resolved.receptors,
            self._instrument_state.instrument,
        )

    def _ensure_jones_terms(self) -> None:
        """Resolve and atomically retain the canonical Jones-term inventory.

        Runs after instrument and receptor resolution and **before** any beam
        load, sky load, network access, or solver work
        (``Tier7JonesSciencePlan.md`` Section 26.1): every ``jones:`` rejection
        is therefore raised before the first side effect, which is the Tier 1
        "reject before side effects" property extended to the new section.

        The resolved baseline selection and the resolved channel widths are
        handed down with the instrument because two terms are functions of the
        *run* rather than of the document: ``M`` is keyed by baseline, so R14
        needs the selection, and ``Q`` smears over the declared width of each
        channel and the declared integration time of each sample rather than
        over anything ``jones.Q`` could have said (Section 20.11).
        """
        if self._jones_terms is not None:
            return
        if self._instrument_state is None:
            raise RuntimeError("Instrument resolution has not completed")

        from radiosim.core.jones_terms import resolve_jones_terms

        self._jones_terms = resolve_jones_terms(
            self._resolved.jones,
            self._instrument_state.instrument,
            frequencies_hz=self._resolved.frequency.channel_frequencies_hz,
            channel_widths_hz=self._resolved.frequency.channel_widths_hz,
            time_grid=self._resolved.observation.time_grid,
            baseline_selection=self._instrument_state.selection,
            precision=self._precision,
        )

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
                receptors=self._receptor_set,
            )
            self._beam_system = loaded

    def _clear_later_runtime_state(self) -> None:
        """Clear setup state that is safe to recreate after instrument resolution."""
        self._backend = None
        self._simulator = None
        self._solver_instrument_view = None
        self._era_grid = None
        self._source_arrays = None
        self._sky_model = None
        self._location = None
        self._obstime = None
        self._frequencies_hz = None
        self._wavelengths = None
        self._network_status = None
        self._device_resources = None
        self._loader_execution = None
        self._is_setup = False

    def _result_history(self) -> tuple[str, ...]:
        """Return the canonical result history for one completed run.

        The executed loader worker policy travels as one encoded line, so it
        survives HDF5, the summary JSON, and the standard visibility formats
        without a result-schema field (plan Section 19).
        """
        history = (f"RadioSim {self.version} canonical visibility simulation",)
        if self._loader_execution is not None:
            history += (self._loader_execution.to_history_line(),)
        return history

    def _requested_healpix_nside(self) -> int:
        """Return the exact configured grid target or the materialization default."""
        return next(
            (
                int(request.options["nside"])
                for request in self._resolved.sky_model.sources
                if request.options.get("nside") is not None
            ),
            64,
        )

    @staticmethod
    def _warn_for_coarse_beam_sampling(
        requirement: BeamSamplingRequirement,
    ) -> bool:
        """Log the deterministic advisory when the actual grid is too coarse."""
        if requirement.actual_pixel_scale_rad <= requirement.pixel_limit_rad:
            return False
        p = f"{requirement.baseline_ant1.number}:{requirement.baseline_ant1.name}"
        q = f"{requirement.baseline_ant2.number}:{requirement.baseline_ant2.name}"
        logger.warning(
            f"HEALPix nside={requirement.actual_nside} has pixel scale "
            f"{requirement.actual_pixel_scale_rad:.6g} rad, above the Tier 3 "
            f"beam-product limit {requirement.pixel_limit_rad:.6g} rad "
            f"(smallest feature "
            f"{requirement.product_feature_scale_rad:.6g} rad, safety factor 5, "
            f"baseline {p}-{q}, frequency {requirement.frequency_hz:.6g} Hz). "
            f"Use at least nside={requirement.recommended_nside}; the requested "
            "NSIDE is unchanged."
        )
        return True

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
        .. code-block:: python

            sim = Simulator.from_yaml("config.yaml")
            sim.setup()
            print(f"Antennas: {len(sim.antennas)}")
            print(f"Baselines: {len(sim.baselines)}")
        """
        if self._is_setup:
            return self
        self._ensure_instrument_state()
        from radiosim.core.beam import BeamSamplingDerivationError
        from radiosim.utils import network as network_module

        previous_offline_policy = network_module.offline_policy()
        try:
            self._ensure_receptor_set()
            self._ensure_jones_terms()
            self._ensure_beam_system()
            self._clear_later_runtime_state()
            return self._setup_after_instrument_state()
        except BeamSamplingDerivationError:
            # Sampling characterization is part of accepting the loaded beam
            # state. A retry must rebuild it from the retained instrument state.
            self._beam_system = None
            self._clear_later_runtime_state()
            raise
        except Exception:
            self._clear_later_runtime_state()
            raise
        finally:
            network_module.set_offline_policy(previous_offline_policy)

    def _setup_after_instrument_state(self) -> Simulator:
        """Create backend, observation, and sky state after canonical resolution."""
        if self._instrument_state is None:
            raise RuntimeError("Instrument resolution has not completed")
        if self._beam_system is None:
            raise RuntimeError("Beam resolution has not completed")

        from radiosim.utils.healpix import derive_beam_sampling_requirement

        visibility_config = self._resolved.visibility
        sky_representation = visibility_config["sky_representation"]
        allow_lossy_point_materialization = visibility_config.get(
            "allow_lossy_point_materialization", False
        )
        allow_lossy_point_rasterization = visibility_config.get(
            "allow_lossy_point_rasterization", False
        )
        requested_nside = self._requested_healpix_nside()
        pre_sky_sampling = derive_beam_sampling_requirement(
            selected_baselines=self.baselines,
            beam_state=self.beam_state,
            observation_frequencies_hz=(
                self._resolved.frequency.channel_frequencies_hz
            ),
            actual_nside=requested_nside,
        )
        pre_sky_warning_emitted = False
        if sky_representation in {"healpix_map", "hybrid"}:
            pre_sky_warning_emitted = self._warn_for_coarse_beam_sampling(
                pre_sky_sampling
            )

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

        # Network connectivity check (before sky model loading)
        from radiosim.utils import network as network_module
        from radiosim.utils.network import (
            SERVICE_DISPLAY_NAMES,
            get_network_status,
            get_required_services,
        )

        # Section 20.1 step 6: the resolved offline policy is installed once,
        # before network status or any loader runs, and is propagated into every
        # loader worker. ``setup()`` restores the caller's previous policy after
        # this setup attempt succeeds or fails.
        network_module.set_offline_policy(self._offline)
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
            from radiosim.core.sky.operations import parallel as parallel_module

            sky_loading = self._resolved.execution.sky_loading
            loaded_models, loader_record = parallel_module.load_models_parallel(
                loader_requests,
                max_workers=sky_loading.max_workers,
                precision=_precision,
                strict=True,
                executor=sky_loading.executor,
            )
            sky_models.extend(loaded_models)
            self._loader_execution = loader_record

        # If no models selected, raise an error
        if not sky_models:
            raise ValueError(
                "No sky model enabled in configuration. "
                "Enable at least one sky model source spec "
                "(for example kind='test_sources', 'gleam', or 'diffuse_sky') "
                "in the sky_model section of your config."
            )

        from radiosim.core.hybrid import check_representation_compatibility
        from radiosim.core.sky.combine.pipeline import prepare_sky_model

        # ``hybrid`` is a solve mode, not a payload representation: it resolves
        # to the existing hybrid-preserving combine path, which reduces the
        # point pile and the HEALPix pile independently and converts neither
        # (``Tier6HybridRuntimePlan.md`` Section 8.1).
        combine_representation = (
            None if sky_representation == "hybrid" else sky_representation
        )
        self._sky_model = prepare_sky_model(
            sky_models,
            representation=combine_representation,
            nside=requested_nside,
            frequency=frequency,
            frequencies=self._frequencies_hz,
            allow_lossy=allow_lossy_point_materialization,
            mixed_model_policy=mixed_model_policy,
            assume_disjoint=assume_disjoint,
            brightness_conversion=_brightness_conv,
            precision=_precision,
            backend=self._backend,
        )

        # Section 20.1 step 9: reject a representation that would silently lose
        # or degrade a payload.  This runs after combination because the hybrid
        # and point-source decisions need the *combined* model's payload set,
        # and before any solver work or output path exists.
        check_representation_compatibility(
            sky_representation=sky_representation,
            contributed_models=sky_models,
            resolved_model=self._sky_model,
            allow_lossy_point_rasterization=allow_lossy_point_rasterization,
        )

        # Get point source arrays for RIME calculator (not in healpix_map mode)
        if self._sky_model.healpix is not None:
            actual_nside = int(self._sky_model.healpix.nside)
            post_sky_sampling = derive_beam_sampling_requirement(
                selected_baselines=self.baselines,
                beam_state=self.beam_state,
                observation_frequencies_hz=(
                    self._resolved.frequency.channel_frequencies_hz
                ),
                actual_nside=actual_nside,
            )
            if not (
                pre_sky_warning_emitted
                and actual_nside == pre_sky_sampling.actual_nside
            ):
                _ = self._warn_for_coarse_beam_sampling(post_sky_sampling)

        if sky_representation == "healpix_map":
            self._source_arrays = None
        else:
            self._source_arrays = self._sky_model.as_point_source_arrays()

        self._is_setup = True
        print_success(
            f"Setup complete: {len(self.antennas)} antennas, "
            f"{len(self.baselines)} baselines, "
            f"{self._sky_label(sky_representation)}"
        )
        return self

    def _solved_sky_element_count(self) -> int:
        """Return the total sky elements every solved component consumes.

        Before Tier 6F this counted only the point payload, so a ``healpix_map``
        run reported zero and a hybrid run would have under-reported (defect
        ``D19``).  The estimate now sums the components the requested mode
        actually solves (``Tier6HybridRuntimePlan.md`` Section 17).
        """
        from radiosim.core.hybrid import component_names_for_representation

        representation = str(self._resolved.visibility["sky_representation"])
        total = 0
        for name in component_names_for_representation(representation):
            if name == "point":
                total += (
                    len(self._source_arrays["ra_rad"]) if self._source_arrays else 0
                )
            elif self._sky_model is not None:
                total += self._sky_model.n_healpix_pixels
        return total

    def _sky_label(self, sky_representation: str) -> str:
        """Describe the resolved sky the way the requested mode consumes it.

        A hybrid run prints *both* component counts, because reporting only one
        of them would understate the sky the solvers actually see
        (``Tier6HybridRuntimePlan.md`` Section 17).
        """
        from radiosim.core.hybrid import component_names_for_representation

        if self._sky_model is None:
            return "0 sources"
        labels = []
        for name in component_names_for_representation(sky_representation):
            if name == "point":
                labels.append(f"{self._sky_model.n_point_sources} sources")
            else:
                labels.append(f"{self._sky_model.n_healpix_pixels} pixels")
        return " + ".join(labels)

    def _resolve_era_grid(self) -> Any:
        """Return the retained ``CanonicalEraGrid`` of a full-sidereal run.

        ``docs/development/sci004_mmode_design.md`` Section 3.1 requires one
        immutable exact-turn object to be passed *by identity* to every
        consumer, so it is built once here and handed to the strategy.  A
        UTC-uniform ``rime`` run has no such object and receives ``None``.
        """
        from radiosim.io.config import FullSiderealObsTimeConfig
        from radiosim.io.config_resolution import resolve_canonical_era_grid

        obs_time = getattr(self._resolved, "obs_time", None)
        if not isinstance(obs_time, FullSiderealObsTimeConfig):
            return None
        if self._era_grid is None:
            self._era_grid = resolve_canonical_era_grid(obs_time)
        return self._era_grid

    def build_solve_request(self) -> SkySolveRequest:
        """Return the immutable request ``run()`` hands the selected strategy.

        ``docs/development/sci004_mmode_design.md`` Section 2 makes this object
        the whole strategy boundary, so it is public: an evidence generator or
        an out-of-band re-derivation needs exactly the inputs a run consumed,
        and rebuilding them by hand would be a second, divergent definition of
        the boundary.  Setup runs first when it has not already.
        """
        from radiosim.simulator.base import SkySolveRequest

        if not self._is_setup:
            self.setup()
        backend = self._backend
        instrument_view = self._solver_instrument_view
        frequencies = self._frequencies_hz
        location = self._location
        if (
            backend is None
            or instrument_view is None
            or frequencies is None
            or location is None
            or self._sky_model is None
        ):
            raise RuntimeError("Simulation setup did not publish complete solver state")
        return SkySolveRequest(
            sky_representation=str(self._resolved.visibility["sky_representation"]),
            sky_model=self._sky_model,
            source_arrays=self._source_arrays,
            instrument=instrument_view,
            beam_system=self.beam_system,
            location=location,
            time_grid=self._resolved.observation.time_grid,
            frequencies=frequencies,
            receptors=self.receptors,
            jones=self.jones_terms,
            backend=backend,
            worker_policy=self._resolved.execution.solver,
            mmode=self._resolved.execution.mmode,
            era_grid=self._resolve_era_grid(),
        )

    def run(self, *, progress: bool = True) -> SimulationResult:
        """
        Run the visibility simulation.

        Solver concurrency is **not** a ``run()`` argument: it is declared once,
        in ``execution.solver.workers``, resolved centrally, and recorded in the
        result's provenance (``Tier6HybridRuntimePlan.md`` Sections 11.3, 12.1).

        Parameters
        ----------
        progress : bool, optional
            Show progress information (default: True). Keyword-only.

        Returns
        -------
        SimulationResult
            Immutable canonical result with dimensions ``(T, B, F, 4)`` and
            row-major correlation order for the resolved output basis:
            ``XX, XY, YX, YY`` for ``linear_xy`` or ``RR, RL, LR, LL`` for
            ``circular_rl``.

        Examples
        --------
        .. code-block:: python

            sim = Simulator.from_yaml("config.yaml")
            result = sim.run()
            assert result is sim.result
        """
        t_start = time.perf_counter()

        # Set up if not already done
        if not self._is_setup:
            self.setup()

        backend = self._backend
        solver = self._simulator
        instrument_view = self._solver_instrument_view
        frequencies = self._frequencies_hz
        location = self._location
        if (
            backend is None
            or solver is None
            or instrument_view is None
            or frequencies is None
            or location is None
        ):
            raise RuntimeError("Simulation setup did not publish complete solver state")

        if progress:
            print_header(
                f"RadioSim Simulator v{self.version}",
                "Radio Interferometer Visibility Simulator",
            )

        t_setup = time.perf_counter() - t_start

        sky_representation = str(self._resolved.visibility["sky_representation"])

        if progress:
            # Print configuration table (after setup, needs backend/sky_model info)
            sky_label = self._sky_label(sky_representation)
            config_data = {
                "Backend": self._backend.name,
                "Precision": self._backend.precision.default
                if self._backend.precision
                else "standard",
                "Simulator": f"{self._simulator.name} ({self._simulator.complexity})",
                "Sky Mode": sky_representation,
                "Antennas": len(self.antennas),
                "Baselines": len(self.baselines),
                "Sky Model": sky_label,
                "Frequencies": f"{len(self._frequencies_hz)} channels",
            }
            if hasattr(self, "_flux_unit") and self._flux_unit != "Jy":
                config_data["Flux Unit"] = self._flux_unit
            print_table("Simulation Configuration", config_data)
            console.print()  # Add spacing

        print_info(f"Running visibility simulation ({sky_representation} mode)...")

        if self._sky_model is None:
            raise RuntimeError("Simulation setup did not publish a sky model")

        solver_started = time.perf_counter()

        # ``docs/development/sci004_mmode_design.md`` Section 2: the high-level
        # API calls only the *selected registered strategy*, through one
        # immutable whole-``SkyModel`` request.  ``RIMESimulator.solve`` is a
        # thin wrapper around the maintained ``core.hybrid.solve_sky`` path, so
        # every component still receives the identical shared objects and a
        # hybrid run's two cubes are still summed in the backend array domain
        # before the single host transfer below (plan Sections 8.4, 9.1).
        outcome = solver.solve(self.build_solve_request())
        receptor_visibilities = outcome.receptor_visibilities

        solver_seconds = time.perf_counter() - solver_started
        if self._instrument_state is None:
            raise RuntimeError("Instrument resolution has not completed")
        from radiosim.core.phase_center import PhaseCenter
        from radiosim.core.precision import PrecisionConfig
        from radiosim.core.result import (
            BackendResultProvenance,
            MModeSolverResultProvenance,
            ResultPerformance,
            SolverResultProvenance,
            build_simulation_result,
        )

        actual_precision = backend.precision or PrecisionConfig.standard()
        result_dtype = np.dtype(backend.get_complex_dtype("output")).name
        elapsed_before_result = time.perf_counter() - t_start
        performance = ResultPerformance(
            setup_seconds=t_setup,
            solver_seconds=solver_seconds,
            solver_point_seconds=outcome.seconds_for("point"),
            solver_healpix_seconds=outcome.seconds_for("healpix"),
            result_construction_seconds=0.0,
            host_transfer_seconds=0.0,
            total_seconds=elapsed_before_result,
        )
        result = build_simulation_result(
            receptor_visibilities=receptor_visibilities,
            backend=backend,
            time_grid=self._resolved.observation.time_grid,
            frequencies_hz=self._resolved.frequency.channel_frequencies_hz,
            channel_widths_hz=self._resolved.frequency.channel_widths_hz,
            instrument=self.instrument,
            selection=self._instrument_state.selection,
            beam_state=self.beam_state,
            receptors=self.receptors,
            jones_terms=self.jones_terms,
            phase_center=PhaseCenter(),
            backend_provenance=BackendResultProvenance(
                requested_backend=self._resolved.execution.backend_strategy,
                actual_backend=backend.name,
                requested_precision=self._resolved.execution.precision.model_dump(
                    mode="json"
                ),
                actual_precision=actual_precision.model_dump(mode="json"),
                result_dtype=result_dtype,
                device_kind=backend.device_kind,
                compilation_used=backend.supports_compilation,
            ),
            solver_provenance=(
                # ``docs/development/sci004_mmode_design.md`` Section 10: the
                # solver record is a strict tagged union.  A direct run keeps
                # the unchanged ``rime`` arm byte for byte; an m-mode run
                # publishes the snapshot its own strategy built.
                MModeSolverResultProvenance(snapshot=outcome.solver_record)
                if outcome.solver_record is not None
                else SolverResultProvenance(
                    solver="rime",
                    sky_representation=cast(
                        Literal["point_sources", "healpix_map", "hybrid"],
                        sky_representation,
                    ),
                    convention="radiosim.rime-zenith-drift.v1",
                    execution_path=outcome.execution_path,
                    components=outcome.components,
                    component_element_counts=outcome.component_element_counts,
                )
            ),
            resolved_config=self._resolved.to_json_safe(),
            configuration_provenance=(
                None if self._provenance is None else self._provenance.to_json_safe()
            ),
            performance=performance,
            history=self._result_history(),
        )

        if progress:
            print_success(
                "Simulation complete! "
                f"({result.performance.total_seconds:.3f}s total, "
                f"setup {result.performance.setup_seconds:.3f}s)"
            )

        self._result = result
        return result

    def plot(
        self,
        *,
        plot_type: Literal[
            "all",
            "antenna",
            "visibility",
            "heatmap",
            "frequency",
        ] = "all",
        output_dir: str | Path | None = None,
        backend: Literal["bokeh", "matplotlib"] = "bokeh",
        show: bool = True,
        overwrite: bool = False,
        visibility_phase_unit: Literal["radians", "degrees"] = "radians",
    ) -> tuple[Path, ...]:
        """Render the published canonical result into one explicit directory.

        Every requested renderer consumes the published `SimulationResult`
        coordinate arrays directly.  Contract validation and collision checks
        precede any filesystem work; browser presentation follows publication
        of every declared file.

        Parameters
        ----------
        plot_type
            Which canonical plot family to render.
        output_dir
            Required explicit directory receiving the declared HTML files.
        backend
            Rendering backend; only ``bokeh`` is implemented.
        show
            Open each published file in a browser after all files are written.
        overwrite
            Replace declared files that already exist.
        visibility_phase_unit
            Display unit for visibility phase; canonical values stay radians.

        Returns
        -------
        tuple of Path
            The published plot files in deterministic declaration order.
        """
        from radiosim.core.result import ResultUnavailableError
        from radiosim.io.result_errors import OutputCollisionError, OutputPathError
        from radiosim.visualization.errors import (
            ResultBrowserError,
            ResultPlotContractError,
        )

        if plot_type not in _PLOT_FAMILIES:
            raise ResultPlotContractError(
                f"plot_type must be one of {sorted(_PLOT_FAMILIES)}; "
                f"received {plot_type!r}"
            )
        if backend != "bokeh":
            raise ResultPlotContractError(
                "only the bokeh result renderer is implemented; "
                f"received backend {backend!r}"
            )
        if type(show) is not bool:
            raise ResultPlotContractError("show must be a boolean")
        if type(overwrite) is not bool:
            raise ResultPlotContractError("overwrite must be a boolean")
        if visibility_phase_unit not in ("radians", "degrees"):
            raise ResultPlotContractError(
                "visibility_phase_unit must be 'radians' or 'degrees'; "
                f"received {visibility_phase_unit!r}"
            )

        result = self._result
        if result is None:
            raise ResultUnavailableError(
                "plotting requires a successfully published SimulationResult"
            )
        if output_dir is None:
            raise OutputPathError("plotting requires an explicit output_dir")
        if not isinstance(output_dir, (str, Path)):
            raise OutputPathError("output_dir must be a string or Path")

        directory = Path(output_dir)
        requested = _PLOT_FAMILIES[plot_type]
        targets = tuple(directory / _PLOT_FILENAMES[name] for name in requested)
        for target in targets:
            if target.is_symlink() or (target.exists() and not target.is_file()):
                raise OutputCollisionError(
                    f"plot target is not a safe regular file: {target}"
                )
            if target.exists() and not overwrite:
                raise OutputCollisionError(f"plot target already exists: {target}")

        from radiosim.visualization.bokeh_plots import (
            plot_antenna_layout,
            plot_heatmaps,
            plot_modulus_vs_frequency,
            plot_visibility,
        )

        directory.mkdir(parents=True, exist_ok=True)
        for name, target in zip(requested, targets, strict=True):
            if name == "antenna":
                plot_antenna_layout(
                    result.instrument.antennas,
                    save_simulation_data=True,
                    folder_path=str(directory),
                    open_in_browser=False,
                )
                continue
            renderer = {
                "visibility": plot_visibility,
                "heatmap": plot_heatmaps,
                "frequency": plot_modulus_vs_frequency,
            }[name]
            renderer(
                result,
                output_path=target,
                visibility_phase_unit=visibility_phase_unit,
            )
        for target in targets:
            if not target.is_file() or target.is_symlink():
                raise ResultPlotContractError(
                    f"declared plot file was not published: {target}"
                )

        if show:
            import webbrowser

            for target in targets:
                try:
                    webbrowser.open(target.as_uri())
                except Exception as exc:
                    raise ResultBrowserError(
                        f"published plot could not be opened in a browser: {target}"
                    ) from exc
        return targets

    def plan_observability(
        self,
        *,
        reference_antenna: AntennaReference | None = None,
        channel_index: int | None = None,
        lst_start_hours: float | None = None,
        lst_end_hours: float | None = None,
        x_axis: Literal["ra", "lst"] = "ra",
        background_layer: Literal["none", "diffuse"] = "none",
        footprint_model: Literal[
            "beam_threshold",
            "manual_circular",
        ] = "beam_threshold",
        field_radius_deg: float | None = None,
        mode: Literal["summary", "snapshots"] = "summary",
        snapshot_step_seconds: float = 3600.0,
        footprint_step_seconds: float = 60.0,
        beam_time_reference: Literal["start", "midpoint", "end"] = "midpoint",
        beam_contour_min_db: float = -40.0,
        beam_contour_max_db: float = 0.0,
        grid_resolution_deg: float = 1.0,
        max_point_sources: int = 1000,
        top_n_sources: int = 5,
        nearby_source_count: int = 3,
        nearby_buffer_deg: float = 10.0,
        include_source_metrics: bool = False,
    ) -> ObservabilityPlan:
        """Build a canonical observability plan without full simulation setup."""
        from radiosim.core.instrument import AntennaId
        from radiosim.core.observability import (
            InvalidObservabilityContextError,
            InvalidObservabilityReferenceError,
            LSTObservabilityWindow,
            ObservabilityOptions,
            ObservabilityPlanner,
            UTCObservabilityWindow,
        )
        from radiosim.io.instrument_config import (
            AntennaNameReference,
            AntennaNumberReference,
        )

        self._ensure_instrument_state()
        self._ensure_receptor_set()
        self._ensure_beam_system()

        frequencies = self._resolved.frequency.channel_frequencies_hz
        if channel_index is None:
            if len(frequencies) != 1:
                raise InvalidObservabilityContextError(
                    "channel_index is required for a multi-channel observation."
                )
            selected_channel = 0
        elif type(channel_index) is not int or not (
            0 <= channel_index < len(frequencies)
        ):
            raise InvalidObservabilityContextError(
                "channel_index must be a strict integer in range."
            )
        else:
            selected_channel = channel_index
        frequency_hz = frequencies[selected_channel]

        state = self.beam_state
        handler_by_id = {handler.handler_id: handler for handler in state.handlers}
        assigned = tuple(state.assignment_handler_ids)
        assigned_fingerprints = {
            handler_by_id[handler_id].scientific_fingerprint
            for _antenna_id, handler_id in assigned
        }
        if reference_antenna is None:
            if len(assigned_fingerprints) != 1:
                raise InvalidObservabilityReferenceError(
                    "Heterogeneous beam assignments require an explicit exact "
                    "Tier 2 reference antenna."
                )
            selected_reference = min(
                (antenna_id for antenna_id, _handler_id in assigned),
                key=lambda antenna_id: antenna_id.number,
            )
            selection_reason: Literal[
                "explicit",
                "homogeneous_default_minimum_number",
            ] = "homogeneous_default_minimum_number"
        else:
            if type(reference_antenna) is AntennaNumberReference:
                matched = tuple(
                    antenna.id
                    for antenna in self.antennas
                    if antenna.id.number == reference_antenna.number
                )
            elif type(reference_antenna) is AntennaNameReference:
                matched = tuple(
                    antenna.id
                    for antenna in self.antennas
                    if antenna.id.name == reference_antenna.name
                )
            else:
                raise InvalidObservabilityReferenceError(
                    "reference_antenna must be an exact tagged Tier 2 AntennaReference."
                )
            if len(matched) != 1:
                raise InvalidObservabilityReferenceError(
                    "reference_antenna does not match one exact canonical antenna."
                )
            selected_reference = matched[0]
            selection_reason = "explicit"

        canonical_reference = AntennaId(
            selected_reference.number,
            selected_reference.name,
        )
        if (lst_start_hours is None) != (lst_end_hours is None):
            raise InvalidObservabilityContextError(
                "lst_start_hours and lst_end_hours must be supplied together."
            )
        if lst_start_hours is not None:
            from importlib import import_module

            time_module: Any = import_module("astropy.time")
            start_mjd = float(
                time_module.Time(
                    self._resolved.observation.start_time_iso,
                    format="isot",
                    scale="utc",
                ).mjd
            )
            window = LSTObservabilityWindow(
                kind="lst",
                start_hours=lst_start_hours,
                end_hours=cast(float, lst_end_hours),
                wraps_midnight=cast(float, lst_end_hours) < lst_start_hours,
                source="explicit_lst",
                beam_evaluation_time_mjd=start_mjd,
            )
        else:
            window = UTCObservabilityWindow(
                kind="utc",
                start_time_iso=self._resolved.observation.start_time_iso,
                duration_seconds=self._resolved.observation.duration_seconds,
                source="resolved_utc",
            )
        options = ObservabilityOptions(
            x_axis=x_axis,
            background_layer=background_layer,
            footprint_model=footprint_model,
            field_radius_deg=field_radius_deg,
            mode=mode,
            snapshot_step_seconds=snapshot_step_seconds,
            footprint_step_seconds=footprint_step_seconds,
            beam_time_reference=beam_time_reference,
            beam_contour_min_db=beam_contour_min_db,
            beam_contour_max_db=beam_contour_max_db,
            grid_resolution_deg=grid_resolution_deg,
            max_point_sources=max_point_sources,
            top_n_sources=top_n_sources,
            nearby_source_count=nearby_source_count,
            nearby_buffer_deg=nearby_buffer_deg,
            include_source_metrics=include_source_metrics,
        )
        return ObservabilityPlanner(
            instrument=self.instrument,
            beam_system=self.beam_system,
            reference_antenna=canonical_reference,
            reference_selection_reason=selection_reason,
            location=self.instrument.location,
            frequency_hz=frequency_hz,
            channel_index=selected_channel,
            window=window,
            sky_model=self._sky_model,
            options=options,
        ).build()

    def plot_observability(
        self,
        *,
        reference_antenna: AntennaReference | None = None,
        channel_index: int | None = None,
        lst_start_hours: float | None = None,
        lst_end_hours: float | None = None,
        x_axis: Literal["ra", "lst"] = "ra",
        background_layer: Literal["none", "diffuse"] = "none",
        footprint_model: Literal[
            "beam_threshold",
            "manual_circular",
        ] = "beam_threshold",
        field_radius_deg: float | None = None,
        mode: Literal["summary", "snapshots"] = "summary",
        snapshot_step_seconds: float = 3600.0,
        footprint_step_seconds: float = 60.0,
        beam_time_reference: Literal["start", "midpoint", "end"] = "midpoint",
        beam_contour_min_db: float = -40.0,
        beam_contour_max_db: float = 0.0,
        grid_resolution_deg: float = 1.0,
        max_point_sources: int = 1000,
        top_n_sources: int = 5,
        nearby_source_count: int = 3,
        nearby_buffer_deg: float = 10.0,
        include_source_metrics: bool = False,
        show_source_colorbar: bool = False,
        color_scale: Literal["log", "linear"] = "log",
        output_dir: Path | None = None,
        filename: str | None = None,
        overwrite: bool = False,
        open_in_browser: bool = False,
    ) -> UIElement:
        """Plan first, then render and optionally persist an observability view."""
        plan = self.plan_observability(
            reference_antenna=reference_antenna,
            channel_index=channel_index,
            lst_start_hours=lst_start_hours,
            lst_end_hours=lst_end_hours,
            x_axis=x_axis,
            background_layer=background_layer,
            footprint_model=footprint_model,
            field_radius_deg=field_radius_deg,
            mode=mode,
            snapshot_step_seconds=snapshot_step_seconds,
            footprint_step_seconds=footprint_step_seconds,
            beam_time_reference=beam_time_reference,
            beam_contour_min_db=beam_contour_min_db,
            beam_contour_max_db=beam_contour_max_db,
            grid_resolution_deg=grid_resolution_deg,
            max_point_sources=max_point_sources,
            top_n_sources=top_n_sources,
            nearby_source_count=nearby_source_count,
            nearby_buffer_deg=nearby_buffer_deg,
            include_source_metrics=include_source_metrics,
        )
        from radiosim.core.observability import ObservabilityOutputError

        if (output_dir is None) != (filename is None):
            raise ObservabilityOutputError(
                "output_dir and filename must be supplied together."
            )
        if open_in_browser and output_dir is None:
            raise ObservabilityOutputError(
                "open_in_browser=True requires an explicit output target."
            )
        from radiosim.visualization.observability import ObservabilityBokehRenderer

        renderer = ObservabilityBokehRenderer(
            plan,
            show_source_colorbar=show_source_colorbar,
            color_scale=color_scale,
        )
        layout = renderer.create_plot()
        if output_dir is not None and filename is not None:
            renderer.save(
                layout,
                output_dir=output_dir,
                filename=filename,
                overwrite=overwrite,
                open_in_browser=open_in_browser,
            )
        return layout

    def save(
        self,
        path: str | Path,
        /,
        *,
        format: ResultFormat = ResultFormat.HDF5,
        overwrite: bool = False,
    ) -> Path:
        """Save the last successful result to one exact final artifact path."""
        from radiosim.core.result import ResultUnavailableError
        from radiosim.io.result_format import (
            normalize_result_path,
            require_result_format,
        )

        if self._result is None:
            raise ResultUnavailableError(
                "no successfully published SimulationResult is available to save"
            )
        typed_format = require_result_format(format)
        final = normalize_result_path(path, typed_format)
        if typed_format is ResultFormat.HDF5:
            from radiosim.io.hdf5 import write_result_hdf5

            return write_result_hdf5(self._result, final, overwrite=overwrite)
        if typed_format is ResultFormat.SUMMARY_JSON:
            from radiosim.io.summary_json import write_result_summary_json

            return write_result_summary_json(self._result, final, overwrite=overwrite)
        if typed_format is ResultFormat.MS:
            from radiosim.io.measurement_set import write_measurement_set

            return write_measurement_set(self._result, final, overwrite=overwrite)
        if typed_format is ResultFormat.UVFITS:
            from radiosim.io.uvfits import write_uvfits

            return write_uvfits(self._result, final, overwrite=overwrite)
        raise AssertionError("unreachable ResultFormat dispatch")

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
        .. code-block:: python

            sim = Simulator.from_yaml("config.yaml")
            sim.setup()
            mem = sim.get_memory_estimate()
            print(f"Estimated memory: {mem['total_human']}")
        """
        if not self._is_setup:
            self.setup()

        # Get base memory estimate
        logical_n_sources = self._solved_sky_element_count()
        kernel_n_sources = logical_n_sources
        if logical_n_sources > 0 and bool(
            getattr(self._backend, "supports_compilation", False)
        ):
            kernel_n_sources = 1 << (logical_n_sources - 1).bit_length()

        estimate = self._simulator.get_memory_estimate(
            n_antennas=len(self.antennas),
            n_baselines=len(self.baselines),
            n_sources=logical_n_sources,
            n_frequencies=len(self._frequencies_hz),
            n_times=len(self._resolved.observation.time_grid),
            kernel_n_sources=kernel_n_sources,
        )

        # Apply the existing approximate precision factor to every byte leaf,
        # then derive aggregates from those scaled leaves. Scaling ``total`` in
        # isolation made the returned breakdown internally contradictory.
        precision = getattr(self._backend, "precision", None)
        precision_factor = (
            precision.estimate_memory_factor() if precision is not None else 1.0
        )
        estimate["precision_factor"] = precision_factor

        breakdown_bytes = estimate.get("breakdown_bytes")
        if isinstance(breakdown_bytes, Mapping):
            scaled_breakdown = {
                str(name): int(int(value) * precision_factor)
                for name, value in breakdown_bytes.items()
            }
            estimate["breakdown_bytes"] = scaled_breakdown
            estimate["breakdown"] = {
                name: _format_memory_bytes(value)
                for name, value in scaled_breakdown.items()
            }
            estimate["working_bytes"] = sum(scaled_breakdown.values())
        elif "working_bytes" in estimate:
            estimate["working_bytes"] = int(
                int(estimate["working_bytes"]) * precision_factor
            )

        if "output_bytes" in estimate:
            estimate["output_bytes"] = int(
                int(estimate["output_bytes"]) * precision_factor
            )

        if "output_bytes" in estimate and "working_bytes" in estimate:
            estimate["total_bytes"] = int(estimate["output_bytes"]) + int(
                estimate["working_bytes"]
            )
        elif "total_bytes" in estimate:
            estimate["total_bytes"] = int(
                int(estimate["total_bytes"]) * precision_factor
            )

        for byte_key, human_key in (
            ("output_bytes", "output_human"),
            ("working_bytes", "working_human"),
            ("total_bytes", "total_human"),
        ):
            if byte_key in estimate:
                estimate[human_key] = _format_memory_bytes(int(estimate[byte_key]))

        return estimate

    def __repr__(self) -> str:
        """String representation of Simulator."""
        status = "configured" if self._result is None else "completed"
        backend = self._backend.name if self._backend else self._backend_name
        return f"<Simulator v{self.version} [{status}] backend={backend}>"
