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

    J = B @ G @ D @ P @ E @ T @ Z @ K

where (from sky to correlator):
- K: Geometric phase (direction-dependent fringe)
- Z: Ionospheric effects (Faraday rotation, TEC)
- T: Tropospheric effects (delay, attenuation)
- E: Primary beam (direction-dependent gain)
- P: Parallactic angle rotation
- D: Polarization leakage (instrumental polarization)
- G: Electronic gains (complex gains)
- B: Bandpass (frequency-dependent gains)

Classes
-------
JonesTerm : Abstract base class for Jones matrix terms
JonesChain : Manager for combining multiple Jones terms

GeometricPhaseJones : K term (geometric/fringe delay)
BeamJones, AnalyticBeamJones : E term (primary beam)
GainJones, TimeVariableGainJones : G term (electronic gains)
BandpassJones, PolynomialBandpassJones : B term (bandpass)
PolarizationLeakageJones, IXRLeakageJones : D term (pol leakage)
ParallacticAngleJones, FieldRotationJones : P term (feed rotation)
IonosphereJones, TurbulentIonosphereJones : Z term (ionosphere)
TroposphereJones, SaastamoinenTroposphereJones : T term (troposphere)

Examples
--------
>>> from radiosim.core.jones import JonesChain, GeometricPhaseJones, GainJones
>>> from radiosim.backends import get_backend
>>>
>>> # Create backend
>>> backend = get_backend("numpy")
>>>
>>> # Create Jones terms
>>> k_jones = GeometricPhaseJones(source_lmn, wavelengths)
>>> g_jones = GainJones(n_antennas, n_times)
>>>
>>> # Create chain
>>> chain = JonesChain([k_jones, g_jones], backend)
>>>
>>> # Compute visibility for a baseline
>>> coherency = np.array([[1, 0], [0, 1]], dtype=complex)  # Unpolarized
>>> vis = chain.compute_baseline_visibility(
...     antenna_p=0,
...     antenna_q=1,
...     source_idx=0,
...     freq_idx=0,
...     time_idx=0,
...     coherency_matrix=coherency,
...     baseline_uvw=uvw,
... )
"""

from importlib import import_module

from .base import JonesTerm

__all__ = [
    # Base classes
    "JonesTerm",
    "JonesChain",
    "JonesBaselineTerm",
    # K term
    "GeometricPhaseJones",
    # E term
    "BeamJones",
    "AnalyticBeamJones",
    "FITSBeamJones",
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
    "GeometricPhaseJones": (".geometric", "GeometricPhaseJones"),
    "BeamJones": (".beam", "BeamJones"),
    "AnalyticBeamJones": (".beam", "AnalyticBeamJones"),
    "FITSBeamJones": (".beam", "FITSBeamJones"),
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
