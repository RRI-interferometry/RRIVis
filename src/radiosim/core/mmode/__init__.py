"""SCI-004 m-mode forward-simulator core.

``docs/development/sci004_mmode_design.md`` describes a second **complete
forward model**, not a Jones term, a point-source optimization, a map maker, or
a new name for the existing direct sum.  It consumes one resolved ``SkyModel``,
forms sky and instrument harmonic representations, evaluates independent forward
matrix-vector products for each ``m``, and synthesizes the existing time-domain
visibility result.

The observing regime is deliberately narrow (Section 1): the array, beams,
receptors and accepted instrumental terms are fixed in the terrestrial frame;
the phase centre and boresight are the existing fixed zenith; the sky is
sidereal and fixed over one Earth rotation; sample centres are a complete,
unflagged, uniformly spaced Earth Rotation Angle cycle with no duplicated
endpoint; and the output remains a simulated visibility cube.

The slice this package implements is **full Stokes**.
``MModeSimulator.supports_polarization`` is explicitly ``True``, a payload with
non-zero ``Q``, ``U`` or ``V`` takes the polarized execution path, and no
fingerprint pin, speed claim or accelerator claim is made anywhere: a polarized
capability is a statement about which sky the solver integrates, never a
performance one (``PERF-001`` governs every performance statement).

Modules
-------
``types``
    Section 14.0's canonical digest vocabulary plus the frozen packed-harmonic
    containers Section 5.3 makes inseparable from their value buffers.
``time``
    Section 3.1's exact-turn ``CanonicalEraGrid``, the bundled-IERS context, the
    Section 6 normalized transform pair and exposure ``sinc``, and the certified
    rational-interval trigonometric kernel Section 12.1 requires.
``frame``
    Section 4.1's frozen-CIRS rigid-ERA frame, the public-Astropy Richardson
    tangent-transport oracle, and the Section 12.1 frozen analytic horizon
    oracle.
``harmonics``
    Section 5.3's orthonormal Condon-Shortley scalar harmonics and packed block
    table.
``sky``
    Section 7.1's analytic point-delta, HEALPix solid-angle and hybrid
    coefficient constructions.
``transfer``
    Section 6's scalar baseline transfer ``B_lm`` built from the reference RIME
    kernel, its fringe, and the rigid-rotation law.
``solver``
    Section 2's whole-``SkyModel`` forward solve and Section 7.3's every-run
    complete frozen-direct gate.
"""

from radiosim.core.mmode.types import (
    FIELD_ORDER,
    MMODE_CONVENTION,
    MMODE_EXECUTION_POLICY,
    MMODE_FRAME_MODEL,
    MMODE_HARMONIC_CONVENTION,
    MMODE_QUADRATURE_POLICY,
    MMODE_STOKES_BRIDGE,
    MMODE_TANGENT_FRAME_M1,
    MMODE_TIME_GRID_CONVENTION,
    MMODE_TRUNCATION_POLICY,
    SPIN_ORDER,
    MModeDimensions,
    ScalarHarmonicCoefficients,
    ScalarPackedCube,
    ScalarPackedTable,
    derive_mmode_dimensions,
)

__all__ = [
    "FIELD_ORDER",
    "MMODE_CONVENTION",
    "MMODE_EXECUTION_POLICY",
    "MMODE_FRAME_MODEL",
    "MMODE_HARMONIC_CONVENTION",
    "MMODE_QUADRATURE_POLICY",
    "MMODE_STOKES_BRIDGE",
    "MMODE_TANGENT_FRAME_M1",
    "MMODE_TIME_GRID_CONVENTION",
    "MMODE_TRUNCATION_POLICY",
    "SPIN_ORDER",
    "MModeDimensions",
    "ScalarHarmonicCoefficients",
    "ScalarPackedCube",
    "ScalarPackedTable",
    "derive_mmode_dimensions",
]
