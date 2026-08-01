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

    J = H @ G @ B @ D @ P @ C @ E @ T @ Z      (K applied separately)

where (from sky to correlator):
- K: Geometric phase (direction-dependent fringe).  Not a chain term: it is
     per-baseline, so the solvers apply ``geometric_phase()`` separately.
- Z: Ionospheric effects (Faraday rotation, TEC)
- T: Tropospheric effects (delay, attenuation)
- E: Primary beam (direction-dependent gain)
- C: Receptor configuration (basis and static feed rotation)
- P: Parallactic angle rotation
- D: Polarization leakage (instrumental polarization)
- B: Bandpass (frequency-dependent gains)
- G: Electronic gains (complex gains)
- H: Reporting-basis transform

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
    "TimeVariableGainJones",
    "ElevationGainJones",
    # B term
    "BandpassJones",
    "PolynomialBandpassJones",
    "SplineBandpassJones",
    "RFIFlaggedBandpassJones",
    # D term
    "PolarizationLeakageJones",
    "IXRLeakageJones",
    "MuellerLeakageJones",
    "BeamSquintLeakageJones",
    # P term
    "ParallacticAngleJones",
    "FieldRotationJones",
    "VLBIFeedRotationJones",
    # Z term
    "IonosphereJones",
    "TurbulentIonosphereJones",
    "GPSIonosphereJones",
    # T term
    "TroposphereJones",
    "SaastamoinenTroposphereJones",
    "TurbulentTroposphereJones",
    "TroposphericOpacityJones",
    # F term
    "FaradayRotationJones",
    "DifferentialFaradayJones",
    # W term
    "WPhaseJones",
    "WProjectionJones",
    "WidefieldPolarimetricJones",
    # C + H terms
    "ReceptorConfigJones",
    "BasisTransformJones",
    # Ee / a / dE terms
    "ElementBeamJones",
    "ArrayFactorJones",
    "DifferentialBeamJones",
    # Kd / Rc / ff terms
    "DelayJones",
    "CableReflectionJones",
    "FringeFitJones",
    # X / Kx / DF terms
    "CrosshandPhaseJones",
    "CrosshandDelayJones",
    "FrequencyDependentLeakageJones",
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
    "TimeVariableGainJones": (".gain", "TimeVariableGainJones"),
    "ElevationGainJones": (".gain", "ElevationGainJones"),
    "BandpassJones": (".bandpass", "BandpassJones"),
    "PolynomialBandpassJones": (".bandpass", "PolynomialBandpassJones"),
    "SplineBandpassJones": (".bandpass", "SplineBandpassJones"),
    "RFIFlaggedBandpassJones": (".bandpass", "RFIFlaggedBandpassJones"),
    "PolarizationLeakageJones": (
        ".polarization_leakage",
        "PolarizationLeakageJones",
    ),
    "IXRLeakageJones": (".polarization_leakage", "IXRLeakageJones"),
    "MuellerLeakageJones": (".polarization_leakage", "MuellerLeakageJones"),
    "BeamSquintLeakageJones": (
        ".polarization_leakage",
        "BeamSquintLeakageJones",
    ),
    "ParallacticAngleJones": (".parallactic", "ParallacticAngleJones"),
    "FieldRotationJones": (".parallactic", "FieldRotationJones"),
    "VLBIFeedRotationJones": (".parallactic", "VLBIFeedRotationJones"),
    "IonosphereJones": (".ionosphere", "IonosphereJones"),
    "TurbulentIonosphereJones": (".ionosphere", "TurbulentIonosphereJones"),
    "GPSIonosphereJones": (".ionosphere", "GPSIonosphereJones"),
    "TroposphereJones": (".troposphere", "TroposphereJones"),
    "SaastamoinenTroposphereJones": (
        ".troposphere",
        "SaastamoinenTroposphereJones",
    ),
    "TurbulentTroposphereJones": (
        ".troposphere",
        "TurbulentTroposphereJones",
    ),
    "TroposphericOpacityJones": (
        ".troposphere",
        "TroposphericOpacityJones",
    ),
    "FaradayRotationJones": (".faraday", "FaradayRotationJones"),
    "DifferentialFaradayJones": (".faraday", "DifferentialFaradayJones"),
    "WPhaseJones": (".wterm", "WPhaseJones"),
    "WProjectionJones": (".wterm", "WProjectionJones"),
    "WidefieldPolarimetricJones": (".wterm", "WidefieldPolarimetricJones"),
    "ReceptorConfigJones": (".receptor", "ReceptorConfigJones"),
    "BasisTransformJones": (".receptor", "BasisTransformJones"),
    "ElementBeamJones": (".element_beam", "ElementBeamJones"),
    "ArrayFactorJones": (".element_beam", "ArrayFactorJones"),
    "DifferentialBeamJones": (".element_beam", "DifferentialBeamJones"),
    "DelayJones": (".delay", "DelayJones"),
    "CableReflectionJones": (".delay", "CableReflectionJones"),
    "FringeFitJones": (".delay", "FringeFitJones"),
    "CrosshandPhaseJones": (".crosshand", "CrosshandPhaseJones"),
    "CrosshandDelayJones": (".crosshand", "CrosshandDelayJones"),
    "FrequencyDependentLeakageJones": (
        ".crosshand",
        "FrequencyDependentLeakageJones",
    ),
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
