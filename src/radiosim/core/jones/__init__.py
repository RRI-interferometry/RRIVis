"""
Jones Matrix Framework for Radio Interferometer Measurement Equation (RIME).

This module provides a complete implementation of the Jones matrix formalism
for modeling radio interferometric visibilities. The RIME expresses the
measured visibility as:

    V_pq = J_p @ C @ J_q^H

where:
- V_pq is the measured 2x2 visibility matrix for baseline p-q
- J_p, J_q are the Jones matrices for antennas p and q
- C is the 2x2 coherency matrix of the source
- ^H denotes conjugate transpose

The full Jones matrix for an antenna is the product of individual
Jones terms representing different propagation effects:

    J = H @ G @ B @ Rc @ Kd @ X @ D @ C @ E @ P @ T @ Z   (K applied separately)

where (from sky to correlator):
- K: Geometric phase (direction-dependent fringe).  Not a chain term: it is
     per-baseline, so the solvers apply ``geometric_phase()`` separately.
- Z: Ionospheric effects (TEC phase, ionospheric Faraday rotation)
- T: Tropospheric effects (delay, opacity)
- P: Parallactic angle / field rotation (direction-dependent)
- E: Primary beam (direction-dependent gain)
- C: Receptor configuration (basis and static feed rotation)
- D: Polarization leakage (instrumental polarization)
- X: Cross-hand phase and delay
- Kd: Instrumental delay offset
- Rc: RF cable reflection ripple
- B: Bandpass (frequency-dependent gains)
- G: Electronic gains (complex gains)
- H: Reporting-basis transform

Term status
-----------
Every exported term declares ``term_status``, which is exactly
``"implemented"`` or ``"planned"``.  ``C`` and ``H`` (Tier 5), ``G`` and ``B``
(Tier 7D), ``D``, ``X``, ``Kd`` and ``Rc`` (Tier 7E), ``P`` (Tier 7F), and ``Z``
and ``T`` (Tier 7G) are implemented -- every per-antenna term in the chain.  The
remaining two -- ``M`` and ``Q``, both :class:`JonesBaselineTerm` and neither in
the matrix chain -- are planned, which means they have a name, a chain position
and a documented physical effect, and that ``compute_baseline_factor``
**raises** rather than returning an identity.  No exported term multiplies by
the identity in silence (``Tier7JonesSciencePlan.md`` invariant I20, ``Fix.md``
Section 16).

Twenty-six classes that were identity scaffolds for effects RadioSim does not
plan to model were deleted before v1.0 -- turbulent and GPS ionospheres,
w-projection, element beams and array factors, fringe fitting, Mueller and IXR
leakage variants, and the rest.  ``docs/migration_guide.md`` names the
replacement for each.

Classes
-------
JonesTerm : Abstract base class for Jones matrix terms
JonesChain : Manager for combining multiple Jones terms
DirectionBatch : One immutable batch of sky directions per (time, frequency)

Functions
---------
geometric_phase : the K term; per-baseline, so a function and not a class
evaluate_antenna_jones : the one chain-evaluation entry point both solvers use

Examples
--------
>>> from radiosim.core.jones import DirectionBatch, JonesChain, evaluate_antenna_jones
>>> from radiosim.backends import get_backend
>>>
>>> backend = get_backend("numpy")
>>> chain = JonesChain(backend)
>>> chain.add_term(basis_transform_term)  # H, correlator side first
>>> chain.add_term(receptor_config_term)  # C
>>> chain.add_term(beam_term)  # E, sky side last
>>>
>>> # One evaluation per antenna over the whole direction batch
>>> jones_by_row = evaluate_antenna_jones(
...     chain=chain,
...     antenna_rows=(0, 1),
...     directions=directions,
...     frequency_hz=1.5e8,
...     freq_idx=0,
...     time_mjd=60000.0,
...     time_idx=0,
...     backend=backend,
...     dtype=backend.get_complex_dtype("accumulation"),
... )
"""

from importlib import import_module

from .base import JonesTerm

__all__ = [
    # Base classes
    "JonesTerm",
    "JonesChain",
    "JonesBaselineTerm",
    # Direction batch and the shared evaluator
    "DirectionBatch",
    "evaluate_antenna_jones",
    # K term (per-baseline, hence a function)
    "geometric_phase",
    # G term
    "GainJones",
    # B term
    "BandpassJones",
    # D term
    "PolarizationLeakageJones",
    # P term
    "ParallacticAngleJones",
    # Z term
    "IonosphereJones",
    # T term
    "TroposphereJones",
    # C + H terms
    "ReceptorConfigJones",
    "BasisTransformJones",
    # Kd / Rc terms
    "DelayJones",
    "CableReflectionJones",
    # X term
    "CrosshandJones",
    # M / Q terms (baseline-dependent)
    "BaselineMultiplicativeJones",
    "SmearingFactorJones",
]


_LAZY_EXPORTS = {
    "JonesChain": (".chain", "JonesChain"),
    "JonesBaselineTerm": (".baseline_errors", "JonesBaselineTerm"),
    "DirectionBatch": (".directions", "DirectionBatch"),
    "evaluate_antenna_jones": (".evaluate", "evaluate_antenna_jones"),
    "geometric_phase": (".geometric", "geometric_phase"),
    "GainJones": (".gain", "GainJones"),
    "BandpassJones": (".bandpass", "BandpassJones"),
    "PolarizationLeakageJones": (
        ".polarization_leakage",
        "PolarizationLeakageJones",
    ),
    "ParallacticAngleJones": (".parallactic", "ParallacticAngleJones"),
    "IonosphereJones": (".ionosphere", "IonosphereJones"),
    "TroposphereJones": (".troposphere", "TroposphereJones"),
    "ReceptorConfigJones": (".receptor", "ReceptorConfigJones"),
    "BasisTransformJones": (".receptor", "BasisTransformJones"),
    "DelayJones": (".delay", "DelayJones"),
    "CableReflectionJones": (".delay", "CableReflectionJones"),
    "CrosshandJones": (".crosshand", "CrosshandJones"),
    "BaselineMultiplicativeJones": (
        ".baseline_errors",
        "BaselineMultiplicativeJones",
    ),
    "SmearingFactorJones": (".baseline_errors", "SmearingFactorJones"),
}


def __getattr__(name: str) -> object:
    try:
        module_name, attribute_name = _LAZY_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(import_module(module_name, __name__), attribute_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Include lazy public exports in interactive discovery."""
    return sorted(set(globals()) | set(_LAZY_EXPORTS))
