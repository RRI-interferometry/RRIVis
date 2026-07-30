"""Hybrid sky-representation compatibility rules and component summation.

Tier 6F makes ``visibility.sky_representation: hybrid`` a first-class solve mode
(``Tier6HybridRuntimePlan.md`` Sections 8, 9, 18.3, 20.1).  Two things live here
and nowhere else:

1. **The representation compatibility gate.**  Before Tier 6F a requested
   representation silently discarded whatever payload it did not consume
   (defect ``D3``/``D4``) or silently rasterized point sources into a HEALPix
   grid (the silent half of ``D5``).  :func:`check_representation_compatibility`
   ends both: a request that would lose a payload is now rejected with the
   verbatim Section 18.3 message, and the rasterization capability survives only
   behind the explicit ``visibility.allow_lossy_point_rasterization`` opt-in.

2. **The component orchestration and summation.**  :func:`solve_sky` is the one
   place any solver is called for a run.  A hybrid run has exactly two
   components, in the fixed order ``("point", "healpix")`` (Section 8.3), both
   receiving the *identical* instrument view, beam system, receptor set, and
   time grid objects (Section 8.4).  Their cubes are summed with
   :meth:`ArrayBackend.add` **in the backend array domain**, before any host
   transfer, so exactly one ``SimulationResult`` is built per run and the entire
   Tier 4 hardening path runs once (Section 9.1).

Because both component cubes are produced in ``backend.get_complex_dtype
("output")`` and floating-point addition of two given values is deterministic,
``V_hybrid`` is **bit-identical** to ``V_point + V_healpix`` on the NumPy
backend, not merely equal within tolerance (Section 9.2, invariant ``S1``).
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final, Literal

if TYPE_CHECKING:  # pragma: no cover - typing only
    from radiosim.backends.base import ArrayBackend
    from radiosim.core.receptor import ResolvedReceptorSet
    from radiosim.core.runtime_config import ResolvedSolverExecutionConfig
    from radiosim.core.sky.containers.model import SkyModel, SourceArrays
    from radiosim.core.time_grid import ObservationTimeGrid
    from radiosim.simulator.base import VisibilitySimulator

__all__ = [
    "HYBRID_COMPONENT_NAMES",
    "HybridSkyError",
    "HybridSolveOutcome",
    "SolvedComponent",
    "check_representation_compatibility",
    "component_names_for_representation",
    "solve_sky",
]

POINT_COMPONENT: Final = "point"
HEALPIX_COMPONENT: Final = "healpix"

#: The fixed component order of a hybrid run (Section 8.3).  Not configurable:
#: the order fixes the summation and every provenance record, run to run.
HYBRID_COMPONENT_NAMES: Final[tuple[str, ...]] = (POINT_COMPONENT, HEALPIX_COMPONENT)

_COMPONENTS_BY_REPRESENTATION: Final[dict[str, tuple[str, ...]]] = {
    "point_sources": (POINT_COMPONENT,),
    "healpix_map": (HEALPIX_COMPONENT,),
    "hybrid": HYBRID_COMPONENT_NAMES,
}


class HybridSkyError(ValueError):
    """A requested sky representation is incompatible with the resolved model.

    Raised for the three Section 18.3 runtime rejections: a ``hybrid`` request
    whose resolved model carries only one payload, a ``point_sources`` request
    that would discard a HEALPix payload, and a ``healpix_map`` request that
    would rasterize point sources without the explicit opt-in.
    """


def component_names_for_representation(sky_representation: str) -> tuple[str, ...]:
    """Return the canonical component names a representation solves.

    Args:
        sky_representation: ``point_sources``, ``healpix_map``, or ``hybrid``.

    Returns:
        The component names, in the fixed Section 8.3 order.

    Raises:
        ValueError: If the representation is not one of the three literals.
    """
    try:
        return _COMPONENTS_BY_REPRESENTATION[sky_representation]
    except KeyError:
        raise ValueError(
            f"unsupported sky representation {sky_representation!r}"
        ) from None


def _format_set_text(model: SkyModel) -> str:
    """Render a model's populated representations for a rejection message."""
    return "{" + ", ".join(sorted(fmt.value for fmt in model.formats)) + "}"


def check_representation_compatibility(
    *,
    sky_representation: str,
    contributed_models: Sequence[SkyModel],
    resolved_model: SkyModel,
    allow_lossy_point_rasterization: bool,
) -> None:
    """Reject a representation that would silently lose or degrade a payload.

    This is step 9 of the Section 20.1 mandatory failure ordering: it runs after
    sky loading and combination, because the ``hybrid`` and ``point_sources``
    decisions need the *combined* model's payload set, and it runs before any
    solver work or output path exists.

    Args:
        sky_representation: The requested ``visibility.sky_representation``.
        contributed_models: The loaded models handed to ``prepare_sky_model``.
            Needed because a combination targeting one representation can drop
            or fold a contributor's payload before the resolved model is built.
        resolved_model: The combined, materialized model the solvers would see.
        allow_lossy_point_rasterization: The
            ``visibility.allow_lossy_point_rasterization`` opt-in.

    Raises:
        HybridSkyError: With the verbatim Section 18.3 message for the rule that
            fired.
    """
    if sky_representation == "hybrid":
        if resolved_model.point is None or resolved_model.healpix is None:
            raise HybridSkyError(
                "visibility.sky_representation=hybrid requires a sky model with "
                "both a point-source payload and a HEALPix payload; the resolved "
                f"model carries only {_format_set_text(resolved_model)}. Request "
                "point_sources or healpix_map, or add a source of the missing "
                "kind."
            )
        return

    if sky_representation == "point_sources":
        # Two shapes lose maps under a point request: the resolved model still
        # carries them (single hybrid model, defect D3), or the combination
        # dropped a contributor's maps on the way to a point-only result
        # (defect D4).  Neither is affected by
        # ``allow_lossy_point_materialization``: that flag converts a
        # HEALPix-*only* contributor into point sources and is a no-op once a
        # point payload already exists
        # (``core/sky/operations/operations.py::materialize_point_sources_model``),
        # so honoring it here would silently re-open D3.
        drops_maps = resolved_model.healpix is not None or any(
            model.healpix is not None and model.point is not None
            for model in contributed_models
        )
        if drops_maps:
            raise HybridSkyError(
                "visibility.sky_representation=point_sources would discard the "
                "HEALPix payload carried by the resolved sky model. Request "
                "hybrid to sum both components, or set "
                "visibility.allow_lossy_point_materialization=true to convert "
                "the HEALPix payload to point sources."
            )
        return

    if sky_representation == "healpix_map":
        if allow_lossy_point_rasterization:
            return
        rasterized = sum(
            model.n_point_sources
            for model in contributed_models
            if model.point is not None
        )
        if rasterized:
            raise HybridSkyError(
                f"visibility.sky_representation=healpix_map would rasterize "
                f"{rasterized} point source(s) into the HEALPix grid, which "
                "quantizes positions to pixel centers. Request hybrid to sum "
                "both components, or set "
                "visibility.allow_lossy_point_rasterization=true to opt in."
            )
        return

    raise ValueError(f"unsupported sky representation {sky_representation!r}")


@dataclass(frozen=True, slots=True)
class SolvedComponent:
    """One solved component of a run: its identity, size, and wall time."""

    name: str
    element_count: int
    seconds: float


@dataclass(frozen=True, slots=True)
class HybridSolveOutcome:
    """The single receptor-visibility cube of a run and its component record."""

    receptor_visibilities: Any
    components: tuple[SolvedComponent, ...]
    execution_path: Literal["scalar", "polarized"]

    @property
    def component_names(self) -> tuple[str, ...]:
        """Return the solved component names in Section 8.3 order."""
        return tuple(component.name for component in self.components)

    @property
    def component_element_counts(self) -> tuple[int, ...]:
        """Return each component's true element count, in the same order."""
        return tuple(component.element_count for component in self.components)

    def seconds_for(self, name: str) -> float:
        """Return a component's wall time, or ``0.0`` when it did not run."""
        for component in self.components:
            if component.name == name:
                return component.seconds
        return 0.0


def solve_sky(
    *,
    sky_representation: str,
    sky_model: SkyModel,
    source_arrays: SourceArrays | None,
    point_solver: VisibilitySimulator,
    backend: ArrayBackend,
    instrument: Any,
    beam_system: Any,
    location: Any,
    time_grid: ObservationTimeGrid,
    frequencies: Any,
    receptors: ResolvedReceptorSet,
    solver_execution: ResolvedSolverExecutionConfig,
) -> HybridSolveOutcome:
    """Solve every component of a run and return one summed cube.

    Every component receives the *same objects* — instrument view, beam system,
    receptor set, time grid, frequency array, location, and backend — so a
    hybrid run cannot drift between its components (Section 8.4, invariant
    ``S4``).  For ``hybrid`` the two cubes are added with
    :meth:`ArrayBackend.add` while they are still backend arrays, so exactly one
    host transfer, one dtype cast, one finiteness check, and one fingerprint
    follow (Section 9.1).

    Args:
        sky_representation: ``point_sources``, ``healpix_map``, or ``hybrid``.
        sky_model: The resolved combined model.
        source_arrays: The point payload as solver arrays; required whenever the
            representation solves the ``point`` component.
        point_solver: The strategy object owning the point-source solver.
        backend: The one array backend for the run.
        instrument: The one ``SolverInstrumentView``.
        beam_system: The one ``BeamSystem``.
        location: The one observing ``EarthLocation``.
        time_grid: The one ``ObservationTimeGrid``.
        frequencies: The one channel-frequency array.
        receptors: The one ``ResolvedReceptorSet``.
        solver_execution: The centrally resolved solver worker policy.

    Returns:
        The summed cube plus the per-component identity, element count, and
        wall time.

    Raises:
        RuntimeError: If the point component is requested without source arrays.
        ValueError: If the representation is not one of the three literals.
    """
    from radiosim.core.visibility_healpix import calculate_visibility_healpix

    component_names = component_names_for_representation(sky_representation)

    cubes: list[Any] = []
    components: list[SolvedComponent] = []
    polarized = False

    for name in component_names:
        started = time.perf_counter()
        if name == POINT_COMPONENT:
            if source_arrays is None:
                raise RuntimeError("Point-source setup did not publish source arrays")
            cube = point_solver.calculate_visibilities(
                instrument=instrument,
                beam_system=beam_system,
                source_arrays=source_arrays,
                frequencies=frequencies,
                backend=backend,
                location=location,
                time_grid=time_grid,
                receptors=receptors,
                jones_config=None,
                solver_execution=solver_execution,
            )
            element_count = len(source_arrays["ra_rad"])
            # The point solver always runs the polarized path.
            polarized = True
        else:
            include_polarization = sky_model.has_polarized_healpix_maps
            cube = calculate_visibility_healpix(
                sky_model=sky_model,
                instrument=instrument,
                beam_system=beam_system,
                location=location,
                time_grid=time_grid,
                frequencies=frequencies,
                output_units="Jy",
                include_polarization=include_polarization,
                backend=backend,
                receptors=receptors,
                solver_execution=solver_execution,
            )
            element_count = sky_model.n_healpix_pixels
            polarized = polarized or include_polarization
        components.append(
            SolvedComponent(
                name=name,
                element_count=int(element_count),
                seconds=time.perf_counter() - started,
            )
        )
        cubes.append(cube)

    total = cubes[0]
    for cube in cubes[1:]:
        total = backend.add(total, cube)

    return HybridSolveOutcome(
        receptor_visibilities=total,
        components=tuple(components),
        execution_path="polarized" if polarized else "scalar",
    )
