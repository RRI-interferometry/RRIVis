"""The SCI-004 m-mode forward simulator (phase M1, scalar).

``docs/development/sci004_mmode_design.md`` Section 1 adopts one bounded science
driver: a HERA-like, fixed-zenith drift-scan survey requiring repeated
full-sidereal visibility evaluation, with direct-RIME agreement on small
polarized skies and controlled spherical-harmonic truncation error.  The
production name is ``execution.simulator: mmode``, and it is a **second complete
forward model**, not a Jones term, a point-source optimization, a map maker, or a
new name for the existing direct sum.

Capability truth is phase-local (Section 9).  In M1
``supports_polarization`` is explicitly overridden to ``False`` -- "a new
simulator registry entry may not inherit the base class's permissive default" --
and the request validator rejects any sky with non-zero ``Q``, ``U`` or ``V``
using ``mmode_m1_scalar_only``.  ``supports_gpu`` is ``False`` because no
independently accepted end-to-end accelerator record exists for this solver;
NumPy is the scientific reference and the recorded execution policy is
``host_harmonics_backend_native_dense_v1``, which is explicitly *not* an
accelerator claim (register row ``PERF-001`` governs every performance
statement).  Only accepted M2, after point, HEALPix and hybrid full-Stokes
direct oracles pass, may deliberately flip the property and its named Tier 7
characterization assertion.

Both flags are plain class attributes rather than properties.  Section 9 makes
the Tier 7 characterization file the authoritative record of two facts stated
together -- ``MModeSimulator.supports_polarization is False`` beside the
unchanged ``RIMESimulator.supports_polarization is True`` -- and that assertion
reads them from the classes themselves.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Final

import numpy as np

from radiosim.core.jones_terms import EMPTY_JONES_TERMS
from radiosim.core.mmode.types import MMODE_EXECUTION_POLICY
from radiosim.simulator.base import (
    SkySolveOutcome,
    SkySolveRequest,
    VisibilitySimulator,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from radiosim.backends.base import ArrayBackend
    from radiosim.core.beam import BeamSystem
    from radiosim.core.instrument_adapters import SolverInstrumentView
    from radiosim.core.jones_terms import ResolvedJonesTerms
    from radiosim.core.receptor import ResolvedReceptorSet
    from radiosim.core.sky.containers.model import SourceArrays
    from radiosim.core.time_grid import ObservationTimeGrid

__all__ = [
    "MMODE_M1_SCALAR_ONLY_CODE",
    "MMODE_M1_SCALAR_ONLY_MESSAGE",
    "MMODE_POINT_MORPHOLOGY_CODE",
    "MMODE_POINT_MORPHOLOGY_MESSAGE",
    "MModeSimulator",
]

#: Section 8's exact ``mmode_m1_scalar_only`` code and message.
MMODE_M1_SCALAR_ONLY_CODE: Final = "mmode_m1_scalar_only"
MMODE_M1_SCALAR_ONLY_MESSAGE: Final = (
    "MModeSimulator phase M1 accepts Stokes I only; non-zero Q, U, or V "
    "requires accepted phase M2."
)

#: Section 8's exact ``mmode_point_morphology`` code and message.
MMODE_POINT_MORPHOLOGY_CODE: Final = "mmode_point_morphology"
MMODE_POINT_MORPHOLOGY_MESSAGE: Final = (
    "execution.simulator='mmode' does not yet support Gaussian point-source "
    "morphology; use rime or remove the morphology."
)

_POLARIZED_STOKES: Final[tuple[str, ...]] = ("Q", "U", "V")


class MModeSimulator(VisibilitySimulator):
    """The m-mode forward simulator, restricted to the accepted M1 scope.

    Examples
    --------
    >>> from radiosim.simulator import get_simulator
    >>> sim = get_simulator("mmode")
    >>> print(sim.name, sim.supports_polarization, sim.supports_gpu)
    mmode False False
    """

    #: Section 9: explicitly overridden, never inherited from the permissive
    #: base default.  M1 adds no polarized capability claim.
    supports_polarization = False

    #: Section 9: no independently accepted end-to-end accelerator record names
    #: this solver, so the flag stays ``False`` (register row ``PERF-001``).
    supports_gpu = False

    @property
    def name(self) -> str:
        """Simulator identifier, and the accepted ``execution.simulator`` value."""
        return "mmode"

    @property
    def description(self) -> str:
        """Human-readable description."""
        return "m-mode full-sidereal harmonic forward model (phase M1, scalar)"

    @property
    def complexity(self) -> str:
        """Algorithm complexity of the per-``m`` forward products."""
        return "O(N_m × N_bl × N_freq × N_lm)"

    @property
    def transform_execution_policy(self) -> str:
        """Return Section 9's recorded transform execution policy.

        Astropy frame work, IERS mapping, beam sampling, HEALPix geometry and
        the harmonic transforms are host-side NumPy work for every backend;
        only the dense per-``m`` contractions and time synthesis may execute on
        a backend.  This literal records that split and is **not** an
        end-to-end accelerator claim.
        """
        return MMODE_EXECUTION_POLICY

    # -- Section 8/9 typed payload rejections --------------------------------

    def validate_scalar_sky_payload(self, stokes: Mapping[str, float] | Any) -> None:
        """Reject any sky carrying non-zero ``Q``, ``U`` or ``V``.

        Parameters
        ----------
        stokes : mapping
            Resolved Stokes values keyed by ``"I"``, ``"Q"``, ``"U"`` and
            ``"V"``.  Each value may be a scalar or an array; a payload is
            polarized when any finite element compares unequal to zero, and
            both signed zeros are inactive.

        Raises
        ------
        radiosim.io.config_resolution.UnsupportedConfigError
            With issue ``mmode_m1_scalar_only`` and the exact Section 8 message.
        """
        from radiosim.io.config import ConfigIssue
        from radiosim.io.config_resolution import UnsupportedConfigError

        offenders = [
            field
            for field in _POLARIZED_STOKES
            if _has_nonzero(stokes.get(field) if isinstance(stokes, Mapping) else None)
        ]
        if not offenders:
            return
        raise UnsupportedConfigError(
            [
                ConfigIssue(
                    "sky_model",
                    MMODE_M1_SCALAR_ONLY_CODE,
                    MMODE_M1_SCALAR_ONLY_MESSAGE,
                    stage="unsupported",
                )
            ]
        )

    def validate_point_morphology(
        self,
        *,
        major_arcsec: float | None = None,
        minor_arcsec: float | None = None,
        pa_deg: float | None = None,
    ) -> None:
        """Reject Gaussian point-source morphology.

        Section 7.1 rejects it because a baseline-dependent envelope is not one
        common sky field; adding analytic extended-source harmonics requires a
        design successor rather than a runtime approximation.

        Raises
        ------
        radiosim.io.config_resolution.UnsupportedConfigError
            With issue ``mmode_point_morphology`` and the exact Section 8
            message.
        """
        from radiosim.io.config import ConfigIssue
        from radiosim.io.config_resolution import UnsupportedConfigError

        declared = [
            value
            for value in (major_arcsec, minor_arcsec, pa_deg)
            if value is not None and _has_nonzero(value)
        ]
        if not declared:
            return
        raise UnsupportedConfigError(
            [
                ConfigIssue(
                    "sky_model",
                    MMODE_POINT_MORPHOLOGY_CODE,
                    MMODE_POINT_MORPHOLOGY_MESSAGE,
                    stage="unsupported",
                )
            ]
        )

    # -- Section 2 strategy boundary -----------------------------------------

    def calculate_visibilities(
        self,
        instrument: SolverInstrumentView,
        beam_system: BeamSystem,
        source_arrays: SourceArrays,
        frequencies: np.ndarray,
        backend: ArrayBackend,
        *,
        location: Any,
        time_grid: ObservationTimeGrid,
        receptors: ResolvedReceptorSet,
        jones_terms: ResolvedJonesTerms = EMPTY_JONES_TERMS,
    ) -> Any:
        """Reject the point-only interface.

        Section 2 records that adding
        ``MModeSimulator.calculate_visibilities(SourceArrays, ...)`` to the
        registry would be false architecture, because HEALPix and hybrid runs
        would still bypass it.  The m-mode strategy consumes the whole
        ``SkySolveRequest`` instead, so this inherited entry point is closed
        rather than quietly delegating to a direct kernel.
        """
        raise NotImplementedError(
            "MModeSimulator consumes the whole-SkyModel SkySolveRequest through "
            "solve(request); the point-only calculate_visibilities interface is "
            "not an m-mode entry point"
        )

    def solve(self, request: SkySolveRequest) -> SkySolveOutcome:
        """Solve one full-sidereal m-mode run from the whole-sky request.

        The request's resolved ``SkyModel`` drives the harmonic sky
        coefficients, the frozen frame drives the transfer, and the exact-turn
        grid drives the synthesis; the direct point and HEALPix production
        kernels are never called.

        Section 4.2's ``FrameApplicabilityCertificate`` is computed in memory
        first, before any harmonic work, with both censuses over the complete
        direction ledger; Section 7.3's authoritative complete frozen-direct
        gate then runs before any result exists.

        Raises
        ------
        radiosim.core.mmode.frame.MModeHorizonUnresolved
            If either horizon census cannot be certified.
        radiosim.core.mmode.solver.MModeTruncationGateFailed
            If the every-run complete frozen-direct comparison exceeds its fixed
            Section 7.3 limits.  Those limits are fixed SCI-004 bounds and are
            never widened to admit a run.
        """
        from radiosim.core.mmode.solver import solve_mmode

        return solve_mmode(request)


def _has_nonzero(value: Any) -> bool:
    """Return whether a resolved Stokes payload has any non-zero element."""
    if value is None:
        return False
    array = np.atleast_1d(np.asarray(value, dtype=np.float64))
    if array.size == 0:
        return False
    finite = array[np.isfinite(array)]
    return bool(np.any(finite != 0.0))
