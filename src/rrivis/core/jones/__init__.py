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
>>> from rrivis.core.jones import JonesChain, GeometricPhaseJones, GainJones
>>> from rrivis.backends import get_backend
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
...     antenna_p=0, antenna_q=1,
...     source_idx=0, freq_idx=0, time_idx=0,
...     coherency_matrix=coherency,
...     baseline_uvw=uvw
... )
"""

# Base classes
from .base import JonesTerm
from .chain import JonesChain

# K term: Geometric phase
from .geometric import GeometricPhaseJones

# E term: Primary beam
from .beam import (
    BeamJones,
    AnalyticBeamJones,
    FITSBeamJones,
)

# G term: Electronic gains
from .gain import (
    GainJones,
    TimeVariableGainJones,
)

# B term: Bandpass
from .bandpass import (
    BandpassJones,
    PolynomialBandpassJones,
    SplineBandpassJones,
    RFIFlaggedBandpassJones,
)

# D term: Polarization leakage
from .polarization_leakage import (
    PolarizationLeakageJones,
    IXRLeakageJones,
    MuellerLeakageJones,
    BeamSquintLeakageJones,
)

# P term: Parallactic angle
from .parallactic import (
    ParallacticAngleJones,
    FieldRotationJones,
    VLBIFeedRotationJones,
)

# Z term: Ionosphere
from .ionosphere import (
    IonosphereJones,
    TurbulentIonosphereJones,
    GPSIonosphereJones,
)

# T term: Troposphere
from .troposphere import (
    TroposphereJones,
    SaastamoinenTroposphereJones,
    TurbulentTroposphereJones,
    TroposphericOpacityJones,
)

# F term: Faraday rotation
from .faraday import (
    FaradayRotationJones,
    DifferentialFaradayJones,
)

# W term: Non-coplanar phase correction
from .wterm import (
    WPhaseJones,
    WProjectionJones,
    WidefieldPolarimetricJones,
)

# C + H terms: Receptor configuration and basis transforms
from .receptor import (
    ReceptorConfigJones,
    BasisTransformJones,
)

# Ee / a / dE terms: Element beam, array factor, differential beam
from .element_beam import (
    ElementBeamJones,
    ArrayFactorJones,
    DifferentialBeamJones,
)

# K_delay / cable / fringefit: Delay, reflections, VLBI calibration
from .delay import (
    DelayJones,
    CableReflectionJones,
    FringeFitJones,
)

# X / KCROSS / DF terms: Cross-hand effects and frequency-dependent leakage
from .crosshand import (
    CrosshandPhaseJones,
    CrosshandDelayJones,
    FrequencyDependentLeakageJones,
)

# M_pq / Q_spq: Baseline-based terms (NOT per-antenna, use JonesBaselineTerm)
from .baseline_errors import (
    JonesBaselineTerm,
    BaselineMultiplicativeJones,
    SmearingFactorJones,
)

# G term extensions: Elevation gain
from .gain import (
    ElevationGainJones,
)

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
