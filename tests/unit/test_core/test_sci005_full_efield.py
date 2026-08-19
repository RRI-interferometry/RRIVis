"""SCI-005 Stage-3 analytic invariants for the full efield Jones response.

``docs/development/sci005_beam_physics_plan.md`` Sections 5.1.1, 5.2.1, 5.3 and
5.5 freeze a *second accepted subset* of the same pyuvdata
``beam_type == "efield"`` file, selected by exactly one authored literal on the
already accepted FITS source block::

    beams.beam.normalization: uvbeam_peak_common_v1

Its files carry a generally full complex ``data_array`` and a full-stored-grid
unit peak; the accepted ``peak`` subset keeps its identity basis, equal
diagonals, cross-hand-free science, and visible-row peak. The stage's physics
is one fixed real orthogonal conversion per direction and one receptor
factorization:

.. math::

    T(\\varphi)=
    \\begin{pmatrix}\\sin\\varphi & -\\cos\\varphi\\\\
    \\cos\\varphi & \\sin\\varphi\\end{pmatrix},
    \\qquad
    J_{\\rm native}[f,c]=\\sum_a \\mathrm{data}[a,f]\\,T(\\varphi)[a,c],

.. math::

    E=C^{\\dagger}J_{\\rm native},\\qquad C\\,E=J_{\\rm native}.

**Independent oracles.** Every expectation below is built in the test body from
the frozen design text and a published closed form, never by importing the
production helper under test:

* the crossed-ideal-dipole file's stored components are bit-identical to
  pyuvdata 3.2.1's own ``ShortDipoleBeam._efield_eval`` -- an independent
  published implementation of the same model -- while the expected
  ``J_native`` is assembled here from the ``(East, North, Up)`` triad, the
  spherical pair :math:`(\\hat\\theta,\\hat\\varphi)`, and Ludwig's third
  definition :math:`\\hat e_{\\rm co}=\\hat\\theta\\cos\\varphi-
  \\hat\\varphi\\sin\\varphi`, :math:`\\hat e_{\\rm cross}=
  \\hat\\theta\\sin\\varphi+\\hat\\varphi\\cos\\varphi`. The two halves are
  written in different coordinates, so a sign, transpose, or row-order mistake
  in :math:`T(\\varphi)` cannot cancel between them;
* the quadrupolar oracle is the expression
  ``docs/development/beam_physics_scope.md`` states -- ``epsilon(theta) =
  epsilon_0 (theta / theta_ref)**2``, ``cross = epsilon(theta) sin(2 phi)``,
  assembled as ``[[co, cross], [-cross, co]]`` -- and never becomes a public
  production model (Section 1);
* the receptor matrix is rebuilt from Section 5.2.1's frozen formulas
  ``C = M(basis) @ R(chi)``, ``R(chi) = [[cos chi, sin chi], [-sin chi,
  cos chi]]``, ``M(linear) = [[0, 1], [1, 0]]``, ``M(circular) =
  (1/sqrt(2)) * [[1, i], [1, -i]]`` -- deliberately not imported from
  :mod:`radiosim.core.jones.receptor`, which is the production side; and
* IXR is computed here from the singular values by Section 5.3's own total,
  deterministic, fixed-relative-tolerance rule. Section 5.3 rules that Stage 3
  adds "no public production method, no public dataclass, no
  ``core/beam/__init__.py`` export, and no configuration field" for it, so the
  diagnostic's only homes are these tests and the retained evidence.

**Tolerances.** Section 5.2.1 introduces none: every predicate reuses a
constant the accepted code already fixes -- ``_BASIS_TOLERANCE`` and
``_FEED_ANGLE_TOLERANCE_RAD`` of ``1e-12``, the dtype-derived
``normalization_absolute_tolerance``, and the dtype-derived
``scalar_absolute_tolerance`` / ``scalar_relative_tolerance`` pair in the
accepted combined form ``atol + rtol * scale``. Section 5.3 fixes the IXR
classification tolerance at the same ``1e-12``, applied relatively. Section
8.1's frozen separation bound ``max(1e-3, 1024 * atol)`` is the only negative
control threshold used here. Nothing below is authorable in YAML.

**Why the frozen literal is authored rather than a production symbol.**
Section 5.1.1 selects the stage on ``FITSBeamSourceConfig.normalization``
alone and freezes no new production name for the widened subset except the
accepted-subset version literal ``sci005-stage3-full-efield-v1``. This module
therefore binds the stage exclusively through the authored document and
through ``load_beam_system``'s already-accepted ``receptors`` keyword, so no
assertion depends on a private signature the design does not freeze.

**The stored basis is validated, never composed.** The accepted bounded
basis-vector and provenance correction settles what the first cut of this
module reported as unbuildable. ``UVBeam._prepare_basis_vector_array`` raises a
bare untyped ``NotImplementedError`` for any stored basis with a strictly
positive off-diagonal entry and otherwise *discards* the stored array,
rebuilding the exact native identity per interpolation point, so a stored
non-identity basis either crashes evaluation outside every typed rejection or
is silently replaced. Corrected Section 5.1.1 item 10 therefore requires the
stored array to be **exactly** the native identity -- ``1.0`` diagonals and
``0.0`` off-diagonals, at a real floating stored dtype judged by kind and
width with both widths accepted -- under the frozen precedence *non-finite,
then dtype kind, then identity*; corrected Section 5.2.1 drops the
``B = basis * T`` composition entirely and applies
``J_native[f,c] = sum_a data[a,f] T(phi)[a,c]`` to the native components; and
the evaluator keeps ``return_basis_vector=True`` in order to **verify** that
the returned array is the identity, a violated pinned-dependency contract
being an internal failure raised as ``UnsupportedBeamBasisError``. The
transpose- and conjugation-observability the retired stored-non-identity
fixture was meant to provide is supplied here by ``T(phi)`` itself -- real,
non-identity, non-symmetric, and direction-dependent, so it cannot be absorbed
into any constant relabelling -- evaluated against complex efield samples. The
pinned dependency behaviour behind all of this is measured in
``tests/unit/test_core/test_beam_pyuvdata_contract.py``.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from radiosim.core.instrument import AntennaId
from tests.fixtures.beamfits import (
    NON_IDENTITY_STORED_BASES,
    EfieldScienceVariant,
    UnsupportedEfieldVariant,
    build_efield_uvbeam,
    build_efield_variant,
    canonical_azimuth_grid,
    canonical_zenith_angle_grid,
    constant_basis_vector_array,
    crossed_ideal_dipole_components,
    forge_interpolation_basis,
    native_identity_basis_vector_array,
    quadrupolar_native_jones,
    scalar_voltage_reference,
    write_efield_beamfits,
    write_scalar_efield_beamfits,
)
from tests.fixtures.configs import valid_config_mapping

# --- frozen literals ----------------------------------------------------------

#: Section 5.1.1's one new authored ``normalization`` literal.
FULL_EFIELD_NORMALIZATION = "uvbeam_peak_common_v1"

#: Section 5.1.1's accepted-subset version literals, one per accepted literal.
SCALAR_SUBSET_VERSION = "tier3-scalar-v1"
FULL_EFIELD_SUBSET_VERSION = "sci005-stage3-full-efield-v1"

#: Section 8.1's four Stage-3 ``scientific_conventions`` literals.
EFIELD_NORMALIZATION_CONVENTION = "uvbeam_peak_common_v1"
BASIS_CONVERSION_CONVENTION = "ludwig3_az_za_to_north_east_v1"
ZENITH_LIMIT_CONVENTION = "north_east_tangent_limit_v1"
FACTORIZATION_CONVENTION = "receptor_conjugated_native_efield_v1"

#: Section 5.2.1's retained structural tolerances, both exactly ``1e-12``.
BASIS_TOLERANCE = 1e-12
FEED_ANGLE_TOLERANCE_RAD = 1e-12

#: Section 5.3's fixed relative classification tolerance, the same constant.
IXR_CLASSIFICATION_TOLERANCE = 1e-12

#: Section 5.2.1's dtype-derived converted-matrix pair, at ``complex128``.
_EPS = float(np.finfo(np.float64).eps)
ATOL = max(1e-12, 32.0 * _EPS)
RTOL = max(1e-10, 32.0 * _EPS)
NORMALIZATION_ATOL = max(1e-12, 32.0 * _EPS)

#: Section 8.1's frozen Stage-3 separation bound for a negative control.
SEPARATION_BOUND = max(1e-3, 1024.0 * ATOL)

#: The one byte-frozen load-stage message Stage 3 retains (Section 5.1.1).
FLOAT128_MESSAGE_TEMPLATE = (
    "BeamFITS {path}: beam precision 'float128' would require complex256, but "
    "accepted files and pyuvdata interpolation provide at most complex128; "
    "select beam float32 or float64."
)

#: Section 8.1's frozen ``exception_type`` per rejected ``probe_kind``. The
#: values are imported per case rather than at module scope, exactly as Stage 1
#: imported ``InvalidBeamGeometryError``, so that a missing Stage-3 surface can
#: never take collection down and hide this module's green controls.
REJECTION_EXCEPTION_BY_PROBE_KIND: dict[str, str] = {
    "power_beam": "UnsupportedBeamTypeError",
    "phased_array_antenna": "UnsupportedBeamTypeError",
    "healpix_pixels": "UnsupportedBeamCoordinateError",
    "zenith_single_valued": "UnsupportedBeamCoordinateError",
    "wrap_continuity": "UnsupportedBeamCoordinateError",
    "grid_coverage": "BeamAngularDomainError",
    "vector_dimension": "UnsupportedBeamBasisError",
    "basis_vector_not_identity": "UnsupportedBeamBasisError",
    "basis_vector_complex": "UnsupportedBeamBasisError",
    "feed_pair": "UnsupportedBeamFeedError",
    "feed_pair_receptor_mismatch": "UnsupportedBeamFeedError",
    "feed_angle": "UnsupportedBeamFeedError",
    "derived_orientation": "UnsupportedBeamFeedError",
    "mount": "UnsupportedBeamFeedError",
    "data_dtype": "UnsupportedBeamPrecisionError",
    "extended_precision": "UnsupportedBeamPrecisionError",
    "basis_vector_non_finite": "NonFiniteBeamResponseError",
    "data_non_finite": "NonFiniteBeamResponseError",
    "data_normalization": "BeamNormalizationError",
    "bandpass": "BeamNormalizationError",
    "visible_only_peak": "BeamNormalizationError",
}

# --- the shipped fixture ------------------------------------------------------

ANT0 = AntennaId(0, "ANT0")
ANT1 = AntennaId(1, "ANT1")

#: A rotated linear receptor: Section 8.1 requires at least one ``linear`` row
#: with non-zero ``feed_rotation_deg``.
ROTATED_FEED_ROTATION_DEG = 31.0

#: Probe directions inside the visible hemisphere. The first entry is the
#: zenith itself, where the North/East tangent limit lives.
_PROBE_ZENITH_ANGLE_RAD = np.array([0.0, 0.12, 0.35, 0.60, 0.95], dtype=np.float64)
_PROBE_AZIMUTH_RAD = np.array([0.0, 0.4, 2.1, 3.6, 5.2], dtype=np.float64)
PROBE_ALTITUDE_RAD = np.pi / 2.0 - _PROBE_ZENITH_ANGLE_RAD
PROBE_AZIMUTH_RAD = _PROBE_AZIMUTH_RAD


# --- memo-derived oracles, written here rather than imported -------------------


def tangent_conversion(phi_rad: Any) -> np.ndarray:
    """Return Section 5.2.1's frozen ``T(phi)``, shape ``(..., 2, 2)``."""
    phi = np.asarray(phi_rad, dtype=np.float64)
    matrix = np.empty(phi.shape + (2, 2), dtype=np.float64)
    matrix[..., 0, 0] = np.sin(phi)
    matrix[..., 0, 1] = -np.cos(phi)
    matrix[..., 1, 0] = np.cos(phi)
    matrix[..., 1, 1] = np.sin(phi)
    return matrix


def radiosim_azimuth_rad(azimuth_uv_rad: Any) -> np.ndarray:
    """Invert the accepted ``az_uv = (pi/2 - az_radiosim) mod 2*pi`` mapping."""
    return np.asarray((np.pi / 2.0 - np.asarray(azimuth_uv_rad)) % (2.0 * np.pi))


def convert_native_jones(data: np.ndarray, phi_rad: float) -> np.ndarray:
    """Return corrected Section 5.2.1's ``J_native`` for one direction.

    ``J_native[f,c] = sum_a data[a,f] T(phi)[a,c]``, "with no conjugation and
    no implicit transpose anywhere, and with no intermediate composed basis:
    the stored array contributes no factor, because it is required to be the
    identity and is discarded by the interpolator in any case". ``data`` is
    indexed ``[vector_axis, feed]``, vector axis ``0`` being the azimuth
    component and ``1`` the zenith-angle component.
    """
    return np.einsum(
        "af,ac->fc",
        np.asarray(data, dtype=np.complex128),
        tangent_conversion(phi_rad),
    )


def receptor_matrix(basis: str, chi_rad: float) -> np.ndarray:
    """Return Section 5.2.1's ``C = M(basis) @ R(chi)`` at ``complex128``."""
    rotation = np.array(
        [
            [math.cos(chi_rad), math.sin(chi_rad)],
            [-math.sin(chi_rad), math.cos(chi_rad)],
        ],
        dtype=np.complex128,
    )
    if basis == "linear":
        leading = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
    elif basis == "circular":
        leading = (1.0 / math.sqrt(2.0)) * np.array(
            [[1.0, 1.0j], [1.0, -1.0j]],
            dtype=np.complex128,
        )
    else:  # pragma: no cover - the vocabulary has exactly two members
        raise AssertionError(f"unknown receptor basis {basis!r}")
    return leading @ rotation


def ludwig3_tangent_pair(
    azimuth_uv_rad: float,
    zenith_angle_rad: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(e_co, e_cross)`` as ``(East, North, Up)`` unit vectors.

    Written from the geometry rather than from ``T``: the spherical pair about
    the zenith with ``phi`` measured North through East is
    ``theta_hat = (cos(theta) sin(phi), cos(theta) cos(phi), -sin(theta))`` and
    ``phi_hat = (cos(phi), -sin(phi), 0)``, and Ludwig's third definition
    combines them with ``cos(phi)`` and ``sin(phi)``.
    """
    phi = float(radiosim_azimuth_rad(azimuth_uv_rad))
    theta = float(zenith_angle_rad)
    theta_hat = np.array(
        [
            math.cos(theta) * math.sin(phi),
            math.cos(theta) * math.cos(phi),
            -math.sin(theta),
        ],
        dtype=np.float64,
    )
    phi_hat = np.array([math.cos(phi), -math.sin(phi), 0.0], dtype=np.float64)
    e_co = theta_hat * math.cos(phi) - phi_hat * math.sin(phi)
    e_cross = theta_hat * math.sin(phi) + phi_hat * math.cos(phi)
    return e_co, e_cross


def crossed_dipole_expected_jones(
    azimuth_uv_rad: float,
    zenith_angle_rad: float,
) -> np.ndarray:
    """Return the independent Ludwig-3 projection of two crossed dipoles.

    An infinitesimal dipole along ``p`` has a far field proportional to the
    transverse projection of ``p``, so the voltage it delivers along a tangent
    unit vector is exactly the Euclidean inner product. The ``x`` feed is the
    East-aligned dipole and the ``y`` feed the North-aligned one, which is what
    the file's ``feed_angle`` pair ``(pi/2, 0)`` -- pyuvdata's own "measured
    from north" convention -- declares.
    """
    e_co, e_cross = ludwig3_tangent_pair(azimuth_uv_rad, zenith_angle_rad)
    east = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    north = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    expected = np.zeros((2, 2), dtype=np.complex128)
    expected[0, 0] = east @ e_co
    expected[0, 1] = east @ e_cross
    expected[1, 0] = north @ e_co
    expected[1, 1] = north @ e_cross
    return expected


def ixr_state(sigma_max: float, sigma_min: float) -> str:
    """Return Section 5.3's state by its exact precedence, first match winning."""
    if sigma_max == 0.0 or sigma_min <= IXR_CLASSIFICATION_TOLERANCE * sigma_max:
        return "singular"
    if sigma_max - sigma_min <= IXR_CLASSIFICATION_TOLERANCE * sigma_max:
        return "unitary_scaled"
    return "nonsingular"


def ixr_row(matrix: np.ndarray) -> dict[str, Any]:
    """Return one Section 5.3 diagnostic row for a 2x2 complex matrix.

    The singular values are computed in binary64 from the converted matrix
    promoted once to ``complex128``, exactly as Section 5.3 requires.
    """
    singular = np.linalg.svd(
        np.asarray(matrix, dtype=np.complex128),
        compute_uv=False,
    )
    sigma_max = float(singular[0])
    sigma_min = float(singular[-1])
    state = ixr_state(sigma_max, sigma_min)
    row: dict[str, Any] = {
        "state": state,
        "sigma_max": sigma_max,
        "sigma_min": sigma_min,
        "condition_number": None,
        "ixr_linear": None,
        "ixr_db": None,
        "leakage_magnitude": None,
    }
    if state == "singular":
        return row
    condition_number = sigma_max / sigma_min
    row["condition_number"] = condition_number
    if state == "unitary_scaled":
        return row
    ixr_linear = ((condition_number + 1.0) / (condition_number - 1.0)) ** 2
    row["ixr_linear"] = ixr_linear
    row["ixr_db"] = 10.0 * math.log10(ixr_linear)
    row["leakage_magnitude"] = 1.0 / math.sqrt(ixr_linear)
    return row


def combined_bound(*matrices: np.ndarray) -> float:
    """Return Section 5.2.1's ``atol + rtol * scale`` for compared matrices."""
    scale = max(float(np.max(np.abs(np.asarray(item)))) for item in matrices)
    return ATOL + RTOL * scale


# --- document builders and the load seam --------------------------------------


def fits_beams(
    path: Path,
    *,
    normalization: str = FULL_EFIELD_NORMALIZATION,
    frequency_interpolation: str = "cubic",
) -> dict[str, Any]:
    """One ``shared_fits`` beams block naming this transport and literal."""
    return {
        "mode": "shared_fits",
        "beam": {
            "kind": "fits",
            "path": str(path),
            "normalization": normalization,
            "frequency_interpolation": frequency_interpolation,
        },
    }


def receptors_block(
    *,
    basis: str = "linear",
    feed_rotation_deg: float = 0.0,
    output_basis: str = "auto",
    overrides: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """One strict ``receptors:`` section in the accepted Tier-5 spelling."""
    block: dict[str, Any] = {
        "default": {"basis": basis, "feed_rotation_deg": feed_rotation_deg},
        "output_basis": output_basis,
    }
    if overrides is not None:
        block["overrides"] = overrides
    return block


class _MemoryLoader:
    """Return one already-built dependency object at the injectable seam."""

    def __init__(self, beam: Any) -> None:
        self.beam = beam
        self.paths: list[Path] = []

    def read(self, path: Path) -> Any:
        self.paths.append(Path(path))
        return self.beam


def resolve_document(
    tmp_path: Path,
    beams: dict[str, Any],
    *,
    receptors: dict[str, Any] | None = None,
    beam_precision: str | None = None,
) -> Any:
    """Resolve one document against the shipped two-antenna fixture."""
    from radiosim.io.config_resolution import ConfigurationSource, resolve_config

    tmp_path.mkdir(parents=True, exist_ok=True)
    overrides: dict[str, Any] = {"beams": beams}
    if receptors is not None:
        overrides["receptors"] = receptors
    data = valid_config_mapping(tmp_path, **overrides)
    if beam_precision is not None:
        data["execution"]["precision"] = {"jones": {"beam": beam_precision}}
    return resolve_config(
        data,
        source=ConfigurationSource.for_mapping(
            base_dir=tmp_path,
            invocation_dir=tmp_path,
        ),
    )


def load_efield_system(
    tmp_path: Path,
    *,
    beam: Any = None,
    path: Path | None = None,
    normalization: str = FULL_EFIELD_NORMALIZATION,
    receptors: dict[str, Any] | None = None,
    beam_precision: str | None = None,
    pass_receptors: bool = True,
) -> tuple[Any, Any, Any]:
    """Return ``(system, receptor_set, resolved_beam_state)`` for one document.

    ``beam`` injects an already-built dependency object through the private
    loader seam ``core/beam/runtime._load_beam_system`` already exposes, which
    is how the accepted suite reaches states pyuvdata refuses to write. When it
    is ``None`` the real transport at ``path`` is read by the production
    loader.

    Section 5.1.1 rules the receptor half of items 5 through 7 onto beam-system
    load and reuses ``load_beam_system``'s already-accepted ``receptors``
    keyword, so that keyword is the only widened surface this module binds.
    """
    from radiosim.core.beam.resolution import resolve_beam_assignments
    from radiosim.core.beam.runtime import (
        _load_beam_system,  # pyright: ignore[reportPrivateUsage]
        _ProductionUVBeamLoader,  # pyright: ignore[reportPrivateUsage]
    )
    from radiosim.core.instrument_resolution import resolve_instrument
    from radiosim.core.receptor import resolve_receptors

    tmp_path.mkdir(parents=True, exist_ok=True)
    if path is None:
        path = tmp_path / "efield.beamfits"
        if not path.exists():
            path.write_bytes(b"injected dependency transport for a Stage-3 probe")
    bundle = resolve_document(
        tmp_path,
        fits_beams(path, normalization=normalization),
        receptors=receptors,
        beam_precision=beam_precision,
    )
    runtime = bundle.runtime
    instrument = resolve_instrument(runtime.instrument)
    receptor_set = resolve_receptors(runtime.receptors, instrument)
    state = resolve_beam_assignments(runtime.beams, instrument)
    loader = _ProductionUVBeamLoader() if beam is None else _MemoryLoader(beam)
    system = _load_beam_system(
        state,
        observation_frequencies_hz=runtime.frequency.channel_frequencies_hz,
        precision=runtime.execution.precision,
        loader=loader,
        receptors=receptor_set if pass_receptors else None,
    )
    return system, receptor_set, state


def evaluate(
    system: Any,
    antenna_id: AntennaId = ANT0,
    *,
    frequency_hz: float = 100e6,
    altitude_rad: np.ndarray = PROBE_ALTITUDE_RAD,
    azimuth_rad: np.ndarray = PROBE_AZIMUTH_RAD,
    time_mjd: float = 60000.0,
) -> np.ndarray:
    """Evaluate one antenna's composed ``E`` batch.

    Section 5.2.1: "``evaluate_jones`` gains no parameter: an efield antenna
    carries no squint, so both Stage-2 boresight keywords remain ``None`` for
    it and the accepted two-sided rule is unchanged."
    """
    return np.asarray(
        system.evaluate_jones(
            antenna_id,
            altitude_rad=altitude_rad,
            azimuth_rad=azimuth_rad,
            frequency_hz=frequency_hz,
            time_mjd=time_mjd,
        )
    )


# ==============================================================================
# Green controls: the fixtures and the oracles, with no Stage-3 surface involved
# ==============================================================================


def test_the_crossed_dipole_fixture_reproduces_pyuvdatas_short_dipole_beam() -> None:
    """The stored components are an independent published model, not our own.

    ``pyuvdata.analytic_beam.ShortDipoleBeam._efield_eval`` documents "The
    first dimension is for [azimuth, zenith angle] in that order, the second
    dimension is for feed [e, n] in that order", which is the vector-axis
    convention Section 5.2.1 assumes. Agreement here is what makes every sign
    assertion in this module a statement about RadioSim rather than about the
    fixture.
    """
    from pyuvdata.analytic_beam import ShortDipoleBeam

    azimuth = canonical_azimuth_grid()
    zenith_angle = np.full(azimuth.size, 0.37)
    reference = ShortDipoleBeam()
    evaluated = reference.efield_eval(
        az_array=azimuth,
        za_array=zenith_angle,
        freq_array=np.array([100e6], dtype=np.float64),
    )

    assert tuple(reference.feed_array) == ("x", "y")
    fixture = crossed_ideal_dipole_components(
        azimuth_uv_rad=azimuth,
        zenith_angle_rad=zenith_angle,
    )
    np.testing.assert_array_equal(evaluated[:, :, 0, :], fixture)


def test_the_frozen_conversion_matrix_is_real_orthogonal_and_power_preserving() -> None:
    """Section 5.2.1: ``det T = 1`` and ``T^T T = I_2`` by construction.

    Both residuals are judged at the fixed ``_BASIS_TOLERANCE`` of ``1e-12``,
    which Section 8.1 also requires of every ``basis_conversions`` row's
    ``power_preservation_max_abs_residual`` and
    ``orthogonality_max_abs_residual``, "those two being predicates on the real
    ``float64`` conversion matrix".
    """
    phi = np.linspace(-3.0 * np.pi, 3.0 * np.pi, 97, dtype=np.float64)
    matrices = tangent_conversion(phi)

    assert matrices.dtype == np.dtype(np.float64)
    determinants = np.linalg.det(matrices)
    assert float(np.max(np.abs(determinants - 1.0))) <= BASIS_TOLERANCE
    products = np.einsum("nac,nab->ncb", matrices, matrices)
    residual = float(np.max(np.abs(products - np.eye(2))))
    assert residual <= BASIS_TOLERANCE
    # Power preservation is then a corollary rather than a second measurement.
    vectors = np.array([[0.3, -0.8], [1.4, 0.2], [-0.5, -0.5]], dtype=np.float64)
    for matrix in matrices:
        rotated = vectors @ matrix
        assert (
            float(
                np.max(
                    np.abs(np.sum(rotated**2, axis=-1) - np.sum(vectors**2, axis=-1))
                )
            )
            <= BASIS_TOLERANCE
        )


def test_the_frozen_conversion_reproduces_the_ludwig3_dipole_oracle() -> None:
    """Section 5.2: the crossed-ideal-dipole case "fixes signs, row/column
    order, and zenith limits".

    The two sides are written in different coordinates: the stored components
    live in ``(azimuth, zenith angle)`` and the expectation in the
    ``(East, North, Up)`` triad through ``e_co`` and ``e_cross``. A transposed
    or sign-flipped ``T`` fails here even though both sides describe the same
    two dipoles.
    """
    azimuth = canonical_azimuth_grid()
    zenith_angle = canonical_zenith_angle_grid()

    worst = 0.0
    for az_value in azimuth:
        for za_value in zenith_angle:
            data = crossed_ideal_dipole_components(
                azimuth_uv_rad=az_value,
                zenith_angle_rad=za_value,
            )
            observed = convert_native_jones(
                data,
                float(radiosim_azimuth_rad(az_value)),
            )
            expected = crossed_dipole_expected_jones(float(az_value), float(za_value))
            worst = max(worst, float(np.max(np.abs(observed - expected))))
    assert worst <= ATOL + RTOL * 1.0


def test_the_crossed_dipole_zenith_row_is_single_valued_at_the_north_east_pair() -> (
    None
):
    """Section 5.2.1: the zenith row is one physical direction sampled at
    ``Naxes1`` arbitrary azimuths, and the ``theta -> 0`` limit at ``phi = 0``
    is exactly ``(North, East)``.

    An East-aligned dipole at the zenith responds only to the East tangent
    component and a North-aligned one only to the North component, so the
    converted zenith matrix is exactly the exchange matrix at every azimuth.
    """
    azimuth = canonical_azimuth_grid()
    matrices = np.stack(
        [
            convert_native_jones(
                crossed_ideal_dipole_components(
                    azimuth_uv_rad=value,
                    zenith_angle_rad=0.0,
                ),
                float(radiosim_azimuth_rad(value)),
            )
            for value in azimuth
        ]
    )

    spread = float(np.max(np.abs(matrices - matrices[0])))
    assert spread <= combined_bound(matrices)
    exchange = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
    assert float(np.max(np.abs(matrices[0] - exchange))) <= combined_bound(matrices)


def test_the_quadrupolar_oracle_has_principal_plane_zeros_and_row_parity() -> None:
    """``beam_physics_scope.md``: the cross-polar term vanishes on the
    principal planes ``phi = 0, pi/2``, peaks at ``phi = pi/4``, and the two
    feed rows carry opposite parity, so the matrix is ``[[co, cross],
    [-cross, co]]`` rather than ``diag(co, co)``.
    """
    zenith_angle = 0.8
    principal = [
        np.pi / 2.0,
        0.0,
        -np.pi / 2.0,
        np.pi,
    ]  # az_uv where phi = 0, pi/2, ...
    for az_value in principal:
        jones = quadrupolar_native_jones(
            azimuth_uv_rad=az_value,
            zenith_angle_rad=zenith_angle,
        )
        assert abs(complex(jones[0, 1])) <= ATOL
        assert abs(complex(jones[1, 0])) <= ATOL

    peak = quadrupolar_native_jones(
        azimuth_uv_rad=np.pi / 4.0,
        zenith_angle_rad=zenith_angle,
    )
    assert abs(complex(peak[0, 1])) > SEPARATION_BOUND
    # Opposite parity: the two cross-hands are exact negatives of each other
    # up to the per-row phase the fixture applies.
    from tests.fixtures.beamfits import QUADRUPOLAR_ROW_PHASE_SLOPE

    phases = [
        np.exp(1j * slope * zenith_angle) for slope in QUADRUPOLAR_ROW_PHASE_SLOPE
    ]
    assert (
        abs(complex(peak[0, 1]) / phases[1] + complex(peak[1, 0]) / phases[0]) <= ATOL
    )
    # Genuinely complex, so a conjugation mistake in the conversion shows up.
    assert abs(complex(peak[0, 0]).imag) > SEPARATION_BOUND


def test_the_quadrupolar_zenith_row_is_single_valued_under_the_row_phase() -> None:
    """The deterministic row phase is a function of zenith angle alone, so the
    fixture satisfies Section 5.2.1's zenith rule while still being complex
    everywhere off the zenith."""
    azimuth = canonical_azimuth_grid()
    matrices = np.stack(
        [
            quadrupolar_native_jones(azimuth_uv_rad=value, zenith_angle_rad=0.0)
            for value in azimuth
        ]
    )
    assert float(np.max(np.abs(matrices - np.eye(2)))) <= ATOL


def test_the_frozen_conversion_is_continuous_across_the_azimuth_wrap() -> None:
    """Section 5.2.1's wrap witness: the converted difference across the seam
    is compared against the difference across the adjacent interior pair."""
    azimuth = canonical_azimuth_grid()
    zenith_angle = 0.42

    def converted(index: int) -> np.ndarray:
        value = float(azimuth[index])
        return convert_native_jones(
            crossed_ideal_dipole_components(
                azimuth_uv_rad=value,
                zenith_angle_rad=zenith_angle,
            ),
            float(radiosim_azimuth_rad(value)),
        )

    matrices = [converted(index) for index in range(azimuth.size)]
    wrap_delta = float(np.max(np.abs(matrices[-1] - matrices[0])))
    interior_delta = float(np.max(np.abs(matrices[-2] - matrices[-1])))
    assert wrap_delta <= interior_delta + combined_bound(*matrices)


def test_the_scalar_subset_file_is_not_zenith_single_valued_under_the_conversion() -> (
    None
):
    """Section 5.1.1: "An identity-basis scalar file generally fails exactly
    this predicate, which is the honest boundary between the two accepted
    subsets."

    The accepted scalar subset reads its diagonal directly and calls the result
    one direction-independent scalar; the full-efield subset reads the same
    bytes as azimuth and zenith-angle field components, which turns the same
    file into ``s * T(phi)``. The zenith row is then a family of different
    physical matrices, which is exactly what the rule rejects.
    """
    azimuth = canonical_azimuth_grid()
    matrices = []
    for value in azimuth:
        scalar = complex(
            scalar_voltage_reference(
                azimuth_uv_rad=value,
                zenith_angle_rad=0.0,
                frequency_index=0,
            )
        )
        data = np.zeros((2, 2), dtype=np.complex128)
        data[0, 0] = scalar
        data[1, 1] = scalar
        matrices.append(convert_native_jones(data, float(radiosim_azimuth_rad(value))))

    spread = float(np.max(np.abs(np.stack(matrices) - matrices[0])))
    assert spread >= max(1e-3, 1024.0 * ATOL)


def test_the_scalar_reading_and_the_full_efield_conversion_of_one_file_diverge() -> (
    None
):
    """Section 8.1's ``scalar_subset_control`` row: the two accepted literals
    are two *interpretations* of the same bytes, not a repair of one another,
    and the retained witness is the measured divergence.
    """
    azimuth_uv = np.pi / 4.0
    zenith_angle = 0.31
    scalar = complex(
        scalar_voltage_reference(
            azimuth_uv_rad=azimuth_uv,
            zenith_angle_rad=zenith_angle,
            frequency_index=0,
        )
    )
    data = np.zeros((2, 2), dtype=np.complex128)
    data[0, 0] = scalar
    data[1, 1] = scalar

    scalar_reading = scalar * np.eye(2, dtype=np.complex128)
    full_efield_reading = convert_native_jones(
        data,
        float(radiosim_azimuth_rad(azimuth_uv)),
    )
    residual = float(np.max(np.abs(scalar_reading - full_efield_reading)))
    assert residual >= max(1e-3, 1024.0 * ATOL)


@pytest.mark.parametrize(
    ("label", "matrix", "state"),
    [
        pytest.param(
            "rank_one",
            np.array([[1.0, 2.0], [0.5, 1.0]], dtype=np.complex128),
            "singular",
            id="singular_rank_one",
        ),
        pytest.param(
            "all_zero",
            np.zeros((2, 2), dtype=np.complex128),
            "singular",
            id="singular_all_zero",
        ),
        pytest.param(
            "scaled_identity",
            0.37 * np.eye(2, dtype=np.complex128),
            "unitary_scaled",
            id="unitary_scaled_identity",
        ),
        pytest.param(
            "generic",
            np.array([[1.0, 0.25j], [-0.1, 0.6]], dtype=np.complex128),
            "nonsingular",
            id="nonsingular_generic",
        ),
    ],
)
def test_the_ixr_classification_rule_assigns_each_state_by_precedence(
    label: str,
    matrix: np.ndarray,
    state: str,
) -> None:
    """Section 5.3's total, deterministic, fixed-relative-tolerance rule.

    The rank-one matrix is mathematically singular and returns a
    ``sigma_min`` of order ``1e-16`` rather than zero, which is exactly why
    Section 5.3 forbids an exact-tie rule; the all-zero matrix is the
    degenerate overlap that rule 1's first clause settles.
    """
    row = ixr_row(matrix)

    assert row["state"] == state
    assert row["sigma_max"] >= row["sigma_min"] >= 0.0
    if label == "rank_one":
        assert 0.0 < row["sigma_min"] <= IXR_CLASSIFICATION_TOLERANCE * row["sigma_max"]


def test_a_receptor_conjugated_scaled_identity_is_unitary_scaled() -> None:
    """Section 5.3: "a matrix that is exactly ``b C^dagger C`` for unitary
    ``C`` returns a split of the same order", so the state must survive the
    realized floating split rather than demand an exact tie."""
    matrices = [
        receptor_matrix(basis, chi).conj().T
        @ (0.83 * np.eye(2, dtype=np.complex128))
        @ receptor_matrix(basis, chi)
        for basis, chi in (("linear", 0.0), ("circular", 0.54), ("linear", -1.1))
    ]
    # The skew leakage form ``[[1, d], [-conj(d), 1]]`` is the same state for
    # the same reason: ``D^H D = (1 + |d|**2) I`` exactly.
    matrices.append(np.array([[1.0, 0.2], [-0.2, 1.0]], dtype=np.complex128))
    matrices.append(np.array([[1.0, 0.3j], [0.3j, 1.0]], dtype=np.complex128))

    for matrix in matrices:
        row = ixr_row(matrix)
        assert row["state"] == "unitary_scaled"
        assert row["condition_number"] is not None
        assert 1.0 <= float(row["condition_number"]) <= 1.0 + 2e-12
        assert row["ixr_linear"] is None
        assert row["ixr_db"] is None
        assert row["leakage_magnitude"] is None


def test_the_ixr_derived_fields_are_null_or_finite_exactly_per_state() -> None:
    """Section 5.3's per-state field contract, and Section 8.1's cross-field
    rule that no infinite or ``NaN`` value is ever written."""
    singular = ixr_row(np.array([[1.0, 2.0], [0.5, 1.0]], dtype=np.complex128))
    assert singular["condition_number"] is None
    assert singular["ixr_linear"] is None
    assert singular["ixr_db"] is None
    assert singular["leakage_magnitude"] is None

    nonsingular = ixr_row(np.array([[1.0, 0.25j], [-0.1, 0.6]], dtype=np.complex128))
    for field in ("condition_number", "ixr_linear", "ixr_db", "leakage_magnitude"):
        value = nonsingular[field]
        assert value is not None
        assert math.isfinite(float(value))
    condition_number = float(nonsingular["condition_number"])
    assert condition_number - 1.0 > IXR_CLASSIFICATION_TOLERANCE
    assert condition_number < 1e12


def test_the_leakage_cross_check_inverts_the_scope_documents_relation() -> None:
    """Section 5.3 requires ``|d| = 1 / sqrt(IXR_J)`` on every ``nonsingular``
    row "so that an inverted-formula regression of the kind the scope document
    already corrected cannot pass unnoticed".

    ``beam_physics_scope.md`` states the two limits directly: ``|d| -> 0`` as
    ``IXR_dB -> infinity`` and ``|d| = 1`` at ``IXR_dB = 0``, with
    ``IXR_dB = -20 log10 |d|``. The ``1 +/- |d|`` singular-value pair belongs
    to the **Hermitian** leakage form ``[[1, d], [conj(d), 1]]``; the
    skew form ``[[1, d], [-conj(d), 1]]`` satisfies
    ``D^H D = (1 + |d|**2) I`` exactly and is therefore ``unitary_scaled``
    with no finite IXR at all, which is why it is exercised as its own state
    witness in :func:`test_a_receptor_conjugated_scaled_identity_is_unitary_scaled`
    rather than here.
    """
    for leakage in (0.001, 0.01, 0.05, 0.2, 0.5):
        matrix = np.array([[1.0, leakage], [leakage, 1.0]], dtype=np.complex128)
        row = ixr_row(matrix)
        assert row["state"] == "nonsingular"
        assert abs(row["sigma_max"] - (1.0 + leakage)) <= 1e-12
        assert abs(row["sigma_min"] - (1.0 - leakage)) <= 1e-12
        assert abs(float(row["leakage_magnitude"]) - leakage) <= 1e-12
        assert abs(float(row["ixr_db"]) - (-20.0 * math.log10(leakage))) <= 1e-9
        # The two relations must agree in both directions.
        condition_number = float(row["condition_number"])
        assert abs(condition_number - (1.0 + leakage) / (1.0 - leakage)) <= 1e-12
        assert abs(
            float(row["ixr_linear"]) - 10.0 ** (float(row["ixr_db"]) / 10.0)
        ) <= 1e-9 * float(row["ixr_linear"])


def test_a_peak_document_keeps_todays_beam_provenance_snapshot_keys(
    tmp_path: Path,
) -> None:
    """Section 5.2.1: every new ``BeamFileProvenance`` field is declared with
    an exact ``None`` default and left ``None`` on the scalar path, and
    ``models._optional_block_fields`` then omits it "from both ``to_snapshot``
    and the canonical fingerprint payload", so "a ``peak`` document's beam
    snapshot, scientific digest, HDF5 ``provenance/beam_json``, and result
    bytes are byte-identical to today".

    This is the disabled-control half of the Stage-3 evidence, pinned as a key
    set so that a widening which forgot the ``None`` default fails here rather
    than in a fingerprint diff.
    """
    written = write_scalar_efield_beamfits(tmp_path)
    system, _receptors, _state = load_efield_system(
        tmp_path,
        path=written.path,
        normalization="peak",
        receptors=receptors_block(),
    )

    snapshot = system.state.to_snapshot()
    handler = snapshot["handlers"][0]
    assert sorted(handler) == [
        "definition_fingerprint",
        "file",
        "handler_id",
        "kind",
        "scientific_fingerprint",
        "voltage_feature_scale_by_frequency",
    ]
    assert sorted(handler["file"]) == [
        "antenna_type",
        "azimuth_step_rad",
        "basis_tolerance",
        "beam_type",
        "data_normalization",
        "data_shape",
        "feed_array",
        "frequency_count",
        "frequency_max_hz",
        "frequency_min_hz",
        "mount_type",
        "native_dtype",
        "normalization_absolute_tolerance",
        "pixel_coordinate_system",
        "pyuvdata_version",
        "resolved_path",
        "scalar_absolute_tolerance",
        "scalar_relative_tolerance",
        "sha256",
        "size_bytes",
        "x_orientation",
        "zenith_angle_max_rad",
        "zenith_angle_step_rad",
    ]
    assert handler["file"]["x_orientation"] == "east"
    assert handler["file"]["basis_tolerance"] == BASIS_TOLERANCE


def test_a_peak_document_still_evaluates_a_scalar_response_with_no_cross_hands(
    tmp_path: Path,
) -> None:
    """The accepted scalar subset is unchanged by this stage: its composed
    response stays ``e I2`` with exactly zero cross-hands and exactly equal
    diagonals (Section 5.1.1: "this gate changes no byte of the accepted
    ``peak`` path")."""
    written = write_scalar_efield_beamfits(tmp_path)
    system, _receptors, _state = load_efield_system(
        tmp_path,
        path=written.path,
        normalization="peak",
        receptors=receptors_block(),
    )

    response = evaluate(system)
    assert response.shape == (PROBE_ALTITUDE_RAD.size, 2, 2)
    np.testing.assert_array_equal(response[:, 0, 1], np.zeros(response.shape[0]))
    np.testing.assert_array_equal(response[:, 1, 0], np.zeros(response.shape[0]))
    np.testing.assert_array_equal(response[:, 0, 0], response[:, 1, 1])


def test_a_peak_document_response_key_is_still_exactly_the_handler_id(
    tmp_path: Path,
) -> None:
    """Section 5.2.1: "An antenna with no efield definition contributes no key
    at all and produces a byte-identical response key to today."""
    written = write_scalar_efield_beamfits(tmp_path)
    system, _receptors, _state = load_efield_system(
        tmp_path,
        path=written.path,
        normalization="peak",
        receptors=receptors_block(feed_rotation_deg=ROTATED_FEED_ROTATION_DEG),
    )

    handler_id = system.state.handlers[0].handler_id
    assert system.response_key(ANT0) == handler_id
    assert system.response_key(ANT1) == handler_id


# ==============================================================================
# Red: the Stage-3 accepted subset
# ==============================================================================


def test_the_full_efield_literal_resolves_into_the_beam_definition(
    tmp_path: Path,
) -> None:
    """Section 5.1.1: ``FITSBeamSourceConfig.normalization`` widens from
    ``Literal["peak"]`` to ``Literal["peak", "uvbeam_peak_common_v1"]``,
    "keeping its ``"peak"`` default, and the resolved leaf
    ``ResolvedFITSBeamDefinition.normalization`` widens identically"."""
    written = write_efield_beamfits(tmp_path)
    bundle = resolve_document(
        tmp_path,
        fits_beams(written.path),
        receptors=receptors_block(),
    )

    beams = bundle.runtime.beams
    assert beams.mode == "shared_fits"
    definition = beams.assignments[0].definition
    assert definition.normalization == FULL_EFIELD_NORMALIZATION
    assert definition.kind == "fits"


def test_the_two_normalization_literals_never_share_a_definition_fingerprint(
    tmp_path: Path,
) -> None:
    """Section 5.1.1: "Because the resolved leaf's ``definition_fingerprint``
    payload already binds ``normalization``, and because the pre-load handler
    key already includes it, the two literals never share a loaded handler and
    a ``peak`` document's every fingerprint stays byte-identical."""
    written = write_efield_beamfits(tmp_path)
    scalar = resolve_document(
        tmp_path / "scalar",
        fits_beams(written.path, normalization="peak"),
        receptors=receptors_block(),
    )
    full = resolve_document(
        tmp_path / "full",
        fits_beams(written.path),
        receptors=receptors_block(),
    )

    scalar_definition = scalar.runtime.beams.assignments[0].definition
    full_definition = full.runtime.beams.assignments[0].definition
    assert scalar_definition.normalization == "peak"
    assert full_definition.normalization == FULL_EFIELD_NORMALIZATION
    assert (
        scalar_definition.definition_fingerprint
        != full_definition.definition_fingerprint
    )


def test_a_full_efield_file_loads_and_records_the_accepted_subset_version(
    tmp_path: Path,
) -> None:
    """Section 5.1.1: "The internal accepted-subset version literal is exactly
    ``tier3-scalar-v1`` for ``peak`` and exactly
    ``sci005-stage3-full-efield-v1`` for ``uvbeam_peak_common_v1``; it enters
    the handler pre-load key and the handler scientific fingerprint exactly
    where the scalar literal does today."

    The version literal is not a public field, so what is asserted here is the
    consequence Section 5.1.1 states: the same bytes under the two literals are
    two different loaded handlers with two different scientific fingerprints.
    """
    written = write_efield_beamfits(tmp_path)
    full_system, _receptors, _state = load_efield_system(
        tmp_path / "full",
        path=written.path,
        receptors=receptors_block(),
    )

    handler = full_system.state.handlers[0]
    assert handler.kind == "fits"
    assert handler.file is not None
    assert handler.file.sha256 == written.sha256
    assert handler.file.data_normalization == "peak"
    assert tuple(handler.file.feed_array) == ("x", "y")

    scalar_written = write_scalar_efield_beamfits(tmp_path / "scalar")
    scalar_system, _r, _s = load_efield_system(
        tmp_path / "scalar",
        path=scalar_written.path,
        normalization="peak",
        receptors=receptors_block(),
    )
    assert (
        handler.scientific_fingerprint
        != scalar_system.state.handlers[0].scientific_fingerprint
    )


def test_the_loaded_full_efield_response_is_a_generally_full_two_by_two(
    tmp_path: Path,
) -> None:
    """Section 5.1.1: the Stage-3 subset accepts "a generally full matrix",
    which is the whole point of the stage -- the accepted scalar subset would
    have rejected these same bytes for carrying cross-polar terms."""
    written = write_efield_beamfits(tmp_path, science=EfieldScienceVariant.QUADRUPOLAR)
    system, _receptors, _state = load_efield_system(
        tmp_path,
        path=written.path,
        receptors=receptors_block(),
    )

    response = evaluate(system)
    assert response.shape == (PROBE_ALTITUDE_RAD.size, 2, 2)
    assert np.all(np.isfinite(response))
    assert float(np.max(np.abs(response[:, 0, 1]))) > SEPARATION_BOUND
    assert float(np.max(np.abs(response[:, 1, 0]))) > SEPARATION_BOUND
    assert float(np.max(np.abs(response[:, 0, 0] - response[:, 1, 1]))) > 0.0


def test_the_composed_e_equals_the_independent_receptor_conjugated_oracle(
    tmp_path: Path,
) -> None:
    """Section 5.2.1: ``E = C^dagger J_native`` and ``C E = J_native``.

    ``J_native`` here is the independent Ludwig-3 projection of the crossed
    dipole pair and ``C`` is rebuilt from the frozen formulas, so both factors
    of the production composition are checked at once. Section 8.1's
    ``factorization_max_abs_residual`` and ``chain_order_max_abs_residual`` are
    exactly these two measurements.
    """
    written = write_efield_beamfits(tmp_path)
    system, receptor_set, _state = load_efield_system(
        tmp_path,
        path=written.path,
        receptors=receptors_block(feed_rotation_deg=0.0),
    )

    response = evaluate(system)
    receptor = receptor_set.receptor_by_antenna[ANT0]
    composed_receptor = receptor_matrix(receptor.basis, receptor.feed_rotation_rad)

    for index in range(PROBE_ALTITUDE_RAD.size):
        zenith_angle = float(np.pi / 2.0 - PROBE_ALTITUDE_RAD[index])
        azimuth_uv = float((np.pi / 2.0 - PROBE_AZIMUTH_RAD[index]) % (2.0 * np.pi))
        native = crossed_dipole_expected_jones(azimuth_uv, zenith_angle)
        expected = composed_receptor.conj().T @ native
        bound = combined_bound(expected, response[index])
        assert float(np.max(np.abs(response[index] - expected))) <= bound
        # ``C E = J_native`` is the same statement read the other way.
        assert (
            float(np.max(np.abs(composed_receptor @ response[index] - native))) <= bound
        )


def test_the_factorization_holds_for_a_rotated_linear_and_a_circular_receptor(
    tmp_path: Path,
) -> None:
    """Section 8.1 requires at least one ``linear`` row with non-zero
    ``feed_rotation_deg`` and rows for both receptor bases; Section 5.2.1
    requires the ``C`` inside ``E`` to be the same authority as the chain's own
    ``C``, so a rotated or circular receptor may never disagree.
    """
    cases = (
        ("linear", ROTATED_FEED_ROTATION_DEG, ("x", "y")),
        ("circular", 0.0, ("r", "l")),
        ("circular", ROTATED_FEED_ROTATION_DEG, ("r", "l")),
    )
    for index, (basis, rotation_deg, feeds) in enumerate(cases):
        root = tmp_path / f"case-{index}"
        written = write_efield_beamfits(
            root,
            science=EfieldScienceVariant.QUADRUPOLAR,
            feed_array=feeds,
            feed_rotation_rad=math.radians(rotation_deg),
        )
        system, receptor_set, _state = load_efield_system(
            root,
            path=written.path,
            receptors=receptors_block(basis=basis, feed_rotation_deg=rotation_deg),
        )

        response = evaluate(system)
        receptor = receptor_set.receptor_by_antenna[ANT0]
        assert receptor.basis == basis
        composed_receptor = receptor_matrix(basis, receptor.feed_rotation_rad)
        assert np.all(np.isfinite(response))

        for probe in range(PROBE_ALTITUDE_RAD.size):
            zenith_angle = float(np.pi / 2.0 - PROBE_ALTITUDE_RAD[probe])
            azimuth_uv = float((np.pi / 2.0 - PROBE_AZIMUTH_RAD[probe]) % (2.0 * np.pi))
            # ``E = C^dagger J_native`` makes ``C E`` the file's own native
            # matrix, which is the physical model itself and is therefore the
            # same for every receptor built on the one file.
            native = np.asarray(
                quadrupolar_native_jones(
                    azimuth_uv_rad=azimuth_uv,
                    zenith_angle_rad=zenith_angle,
                ),
                dtype=np.complex128,
            )
            expected = composed_receptor.conj().T @ native
            bound = combined_bound(expected, response[probe])
            assert float(np.max(np.abs(response[probe] - expected))) <= bound
            assert (
                float(np.max(np.abs(composed_receptor @ response[probe] - native)))
                <= bound
            )


def test_the_zenith_limit_is_the_phi_zero_north_east_member(tmp_path: Path) -> None:
    """Section 5.2.1: "RadioSim then uses the ``phi = 0`` member as the zenith
    value", whose tangent pair is exactly ``(North, East)``.

    The crossed-dipole file's converted zenith matrix is the exchange matrix,
    so the composed ``E`` at the zenith for an unrotated linear receptor --
    whose ``C`` is that same exchange matrix -- is exactly the identity.
    """
    written = write_efield_beamfits(tmp_path)
    system, _receptors, _state = load_efield_system(
        tmp_path,
        path=written.path,
        receptors=receptors_block(),
    )

    zenith = evaluate(
        system,
        altitude_rad=np.full(3, np.pi / 2.0),
        azimuth_rad=np.array([0.0, 1.3, 4.9], dtype=np.float64),
    )
    bound = combined_bound(zenith)
    assert float(np.max(np.abs(zenith - zenith[0]))) <= bound
    assert float(np.max(np.abs(zenith[0] - np.eye(2)))) <= bound


def test_the_full_efield_subset_requires_the_resolved_receptor_set(
    tmp_path: Path,
) -> None:
    """Section 5.1.1: ``load_beam_system`` "requires its already-accepted
    ``receptors`` keyword whenever any resolved antenna is assigned a
    ``uvbeam_peak_common_v1`` definition, exactly as it already does for
    squint, and raises the same ``TypeError`` when it is absent"."""
    written = write_efield_beamfits(tmp_path)
    with pytest.raises(TypeError, match="receptor"):
        load_efield_system(
            tmp_path,
            path=written.path,
            receptors=receptors_block(),
            pass_receptors=False,
        )


def test_the_response_key_separates_two_receptors_sharing_one_efield_file(
    tmp_path: Path,
) -> None:
    """Section 5.2.1: the efield sub-object's "receptor half is part of the
    identity because ``E = C^dagger J_native`` differs between two antennas
    that share one handler and one file but not one receptor", and "the
    HEALPix cache is that same identity".
    """
    written = write_efield_beamfits(tmp_path)
    system, _receptors, _state = load_efield_system(
        tmp_path,
        path=written.path,
        receptors=receptors_block(
            overrides=[
                {
                    "antenna": {"kind": "number", "number": 1},
                    "feed_rotation_deg": ROTATED_FEED_ROTATION_DEG,
                }
            ]
        ),
    )

    handler_id = system.state.handlers[0].handler_id
    assert len(system.state.handlers) == 1
    first = system.response_key(ANT0)
    second = system.response_key(ANT1)
    assert first != second
    assert first != handler_id
    assert second != handler_id
    assert not np.array_equal(evaluate(system, ANT0), evaluate(system, ANT1))


def test_ixr_from_the_production_accepted_matrix_is_nonsingular_and_consistent(
    tmp_path: Path,
) -> None:
    """Section 5.3: IXR "is a pure function of the accepted ``J_native``, which
    the evidence already retains as an authenticated projection", computed "in
    the red tests and the retained evidence from the accepted matrix".

    The quadrupolar fixture is deliberately non-degenerate off the principal
    planes, so the retained state there is ``nonsingular`` and every derived
    field is finite and mutually consistent.
    """
    written = write_efield_beamfits(tmp_path, science=EfieldScienceVariant.QUADRUPOLAR)
    system, receptor_set, _state = load_efield_system(
        tmp_path,
        path=written.path,
        receptors=receptors_block(),
    )

    receptor = receptor_set.receptor_by_antenna[ANT0]
    composed_receptor = receptor_matrix(receptor.basis, receptor.feed_rotation_rad)
    response = evaluate(
        system,
        altitude_rad=np.array([np.pi / 2.0 - 0.8], dtype=np.float64),
        azimuth_rad=np.array([np.pi / 4.0], dtype=np.float64),
    )
    native = composed_receptor @ response[0]
    row = ixr_row(native)

    assert row["state"] == "nonsingular"
    condition_number = float(row["condition_number"])
    assert condition_number > 1.0 + IXR_CLASSIFICATION_TOLERANCE
    assert math.isfinite(float(row["ixr_db"]))
    assert (
        abs(float(row["leakage_magnitude"]) - 1.0 / math.sqrt(float(row["ixr_linear"])))
        <= 1e-12
    )


# ==============================================================================
# Red: the Section 5.1.1 ordered load contract, one typed rejection per item
# ==============================================================================


def _beam_error(name: str) -> type[BaseException]:
    """Return one frozen error class from the public beam error hierarchy.

    Imported per call rather than at module scope: Section 5.1.1 adds no error
    class, so every name below already exists, but keeping the lookup local
    matches the accepted Stage-1 and Stage-2 practice and keeps this module's
    green controls collectable whatever the Stage-3 surface does.
    """
    from radiosim.core.beam import errors as beam_errors

    return getattr(beam_errors, name)


@pytest.mark.parametrize(
    "probe_kind",
    [variant.value for variant in UnsupportedEfieldVariant],
)
def test_the_ordered_load_contract_rejects_each_probe_with_its_frozen_type(
    tmp_path: Path,
    probe_kind: str,
) -> None:
    """Section 5.1.1's thirteen ordered items, and Section 8.1's frozen
    ``exception_type`` per rejected ``probe_kind``.

    Section 2 requires the concrete type and the stable identity, not a message
    substring; Section 5.1.1 adds that every load-stage message is
    "named-fields, not byte-frozen", so the resolved file path is asserted and
    the rendered bytes are not.
    """
    variant = UnsupportedEfieldVariant(probe_kind)
    expected = _beam_error(REJECTION_EXCEPTION_BY_PROBE_KIND[probe_kind])
    fixture = build_efield_variant(variant)

    with pytest.raises(expected) as error:
        load_efield_system(
            tmp_path,
            beam=fixture.beam,
            receptors=receptors_block(),
        )

    assert "efield.beamfits" in str(error.value)


def test_a_file_whose_feed_pair_matches_one_antenna_and_not_another_is_rejected(
    tmp_path: Path,
) -> None:
    """Section 5.1.1 item 5: "a shared file whose pair matches one assigned
    antenna and not another is ``UnsupportedBeamFeedError``"."""
    written = write_efield_beamfits(tmp_path, feed_array=("x", "y"))
    expected = _beam_error("UnsupportedBeamFeedError")

    with pytest.raises(expected) as error:
        load_efield_system(
            tmp_path,
            path=written.path,
            receptors=receptors_block(
                overrides=[
                    {
                        "antenna": {"kind": "number", "number": 1},
                        "basis": "circular",
                    }
                ]
            ),
        )

    message = str(error.value)
    assert "ANT1" in message or "number=1" in message


def test_a_file_whose_feed_angles_match_one_antenna_and_not_another_is_rejected(
    tmp_path: Path,
) -> None:
    """Section 5.1.1 item 6 is per assigned antenna, and the message must name
    "the canonical antenna number and name and both compared values, because
    those interpolate per-antenna state that no fixture can pin across
    arrays"."""
    written = write_efield_beamfits(tmp_path, feed_rotation_rad=0.0)
    expected = _beam_error("UnsupportedBeamFeedError")

    with pytest.raises(expected) as error:
        load_efield_system(
            tmp_path,
            path=written.path,
            receptors=receptors_block(
                overrides=[
                    {
                        "antenna": {"kind": "number", "number": 1},
                        "feed_rotation_deg": ROTATED_FEED_ROTATION_DEG,
                    }
                ]
            ),
        )

    message = str(error.value)
    assert "ANT1" in message or "number=1" in message


def test_the_scalar_subset_file_is_rejected_under_the_full_efield_literal(
    tmp_path: Path,
) -> None:
    """Section 8.1: "a ``zenith_single_valued`` row's file is the
    identity-basis scalar file that Section 5.1.1 names as the honest boundary
    between the two accepted subsets", rejected as
    ``UnsupportedBeamCoordinateError`` -- "the coordinate class rather than the
    basis class, because the failure is a degenerate row of the native
    ``az_za`` grid and not a defect of the Jones structure"."""
    written = write_scalar_efield_beamfits(tmp_path)
    expected = _beam_error("UnsupportedBeamCoordinateError")

    with pytest.raises(expected):
        load_efield_system(
            tmp_path,
            path=written.path,
            receptors=receptors_block(),
        )


def test_extended_precision_keeps_its_existing_byte_frozen_rejection(
    tmp_path: Path,
) -> None:
    """Section 5.1.1 item 13 and Section 8.1's ``extended_precision`` row:
    "``float128`` keeps its existing typed ``UnsupportedBeamPrecisionError``
    and its existing exact message", which is "the one byte-frozen load-stage
    message Stage 3 retains"."""
    written = write_efield_beamfits(tmp_path)
    expected = _beam_error("UnsupportedBeamPrecisionError")

    with pytest.raises(expected) as error:
        load_efield_system(
            tmp_path,
            path=written.path,
            receptors=receptors_block(),
            beam_precision="float128",
        )

    assert str(error.value) == FLOAT128_MESSAGE_TEMPLATE.format(
        path=written.path.resolve(strict=False)
    )


# ==============================================================================
# Red: the corrected stored-basis contract and the T(phi) observability control
# ==============================================================================
#
# The accepted bounded basis-vector and provenance correction replaced the
# gate's "general real basis-vector array" with an exactness predicate, retired
# the ``basis_vector_dtype`` and ``basis_vector_degenerate`` probe kinds in
# favour of ``basis_vector_not_identity``, froze the seven new
# ``BeamFileProvenance`` fields by name, order, and annotation, and moved the
# transpose/conjugation control onto ``T(phi)`` itself.

#: Corrected Section 5.2.1's exact seven-field extension, in its frozen order
#: after the twenty-three fields ``BeamFileProvenance`` already declares.
FROZEN_PROVENANCE_FIELD_ORDER: tuple[str, ...] = (
    "resolved_path",
    "size_bytes",
    "sha256",
    "pyuvdata_version",
    "beam_type",
    "antenna_type",
    "pixel_coordinate_system",
    "mount_type",
    "data_normalization",
    "feed_array",
    "x_orientation",
    "data_shape",
    "native_dtype",
    "frequency_min_hz",
    "frequency_max_hz",
    "frequency_count",
    "azimuth_step_rad",
    "zenith_angle_step_rad",
    "zenith_angle_max_rad",
    "basis_tolerance",
    "scalar_absolute_tolerance",
    "scalar_relative_tolerance",
    "normalization_absolute_tolerance",
    "accepted_subset_version",
    "radiosim_normalization",
    "resolved_feed_array",
    "derived_x_orientation_verdict",
    "basis_vector_convention",
    "factorization_convention",
    "stored_grid_peak_by_frequency",
)

#: The seven appended names alone.
STAGE3_PROVENANCE_FIELDS: tuple[str, ...] = FROZEN_PROVENANCE_FIELD_ORDER[23:]

#: An azimuth at which ``T(phi)`` is both non-identity and non-symmetric.
#: ``T`` is symmetric only where ``cos(phi) == 0`` and is the identity only at
#: ``phi = pi/2``, so any sample with a non-zero cosine serves; the canonical
#: grid's ``az_uv = pi/4`` gives ``phi = pi/4``.
OBSERVABLE_AZIMUTH_UV_RAD = np.pi / 4.0
OBSERVABLE_ZENITH_ANGLE_RAD = 0.62


def test_t_of_phi_is_non_identity_and_non_symmetric_where_it_is_exercised() -> None:
    """Corrected Section 5.2.1: the retired stored-non-identity fixture is
    replaced by ``T(phi)`` itself, "real, non-identity, non-symmetric, and
    direction-dependent", so a transpose mistake cannot be absorbed into a
    constant relabelling.

    This green control proves the probe azimuth actually has that property
    before any test relies on it.
    """
    matrix = tangent_conversion(float(radiosim_azimuth_rad(OBSERVABLE_AZIMUTH_UV_RAD)))

    assert float(np.max(np.abs(matrix - np.eye(2)))) >= SEPARATION_BOUND
    assert float(np.max(np.abs(matrix - matrix.T))) >= SEPARATION_BOUND
    # Direction dependence: a second azimuth gives a genuinely different matrix.
    other = tangent_conversion(
        float(radiosim_azimuth_rad(OBSERVABLE_AZIMUTH_UV_RAD + 1.0))
    )
    assert float(np.max(np.abs(matrix - other))) >= SEPARATION_BOUND


def test_the_conversion_is_observable_under_transpose_and_conjugation() -> None:
    """Corrected Section 5.2 and 5.2.1's replacement observability control.

    The frozen sum uses ``T(phi)`` with "no conjugation and no implicit
    transpose". Substituting ``T(phi).T`` or conjugating the complex efield
    samples must both change ``J_native`` by more than the frozen separation
    bound, which is what makes the production comparison below a real test of
    the mapping rather than of a symmetric coincidence.
    """
    phi = float(radiosim_azimuth_rad(OBSERVABLE_AZIMUTH_UV_RAD))
    data = np.asarray(
        quadrupolar_components_at(
            OBSERVABLE_AZIMUTH_UV_RAD, OBSERVABLE_ZENITH_ANGLE_RAD
        ),
        dtype=np.complex128,
    )
    assert float(np.max(np.abs(data.imag))) >= SEPARATION_BOUND

    frozen = convert_native_jones(data, phi)
    transposed = np.einsum("af,ca->fc", data, tangent_conversion(phi))
    conjugated = convert_native_jones(np.conj(data), phi)
    swapped_rows = np.einsum("fa,ac->fc", data, tangent_conversion(phi))

    assert float(np.max(np.abs(frozen - transposed))) >= SEPARATION_BOUND
    assert float(np.max(np.abs(frozen - conjugated))) >= SEPARATION_BOUND
    assert float(np.max(np.abs(frozen - swapped_rows))) >= SEPARATION_BOUND


def quadrupolar_components_at(
    azimuth_uv_rad: float,
    zenith_angle_rad: float,
) -> np.ndarray:
    """Return the quadrupolar stored components at one direction."""
    from tests.fixtures.beamfits import quadrupolar_components

    return np.asarray(
        quadrupolar_components(
            azimuth_uv_rad=azimuth_uv_rad,
            zenith_angle_rad=zenith_angle_rad,
        ),
        dtype=np.complex128,
    )


def test_the_production_conversion_uses_t_of_phi_untransposed_and_unconjugated(
    tmp_path: Path,
) -> None:
    """The production side of the same control.

    ``C @ E`` recovers ``J_native``, which must equal the frozen ``T(phi)``
    mapping of the file's own stored components and must differ from both the
    transposed and the conjugated variants by more than the separation bound.
    """
    written = write_efield_beamfits(tmp_path, science=EfieldScienceVariant.QUADRUPOLAR)
    system, receptor_set, _state = load_efield_system(
        tmp_path,
        path=written.path,
        receptors=receptors_block(),
    )

    zenith_angle = OBSERVABLE_ZENITH_ANGLE_RAD
    azimuth_uv = OBSERVABLE_AZIMUTH_UV_RAD
    phi = float(radiosim_azimuth_rad(azimuth_uv))
    response = evaluate(
        system,
        altitude_rad=np.array([np.pi / 2.0 - zenith_angle], dtype=np.float64),
        azimuth_rad=np.array([phi], dtype=np.float64),
    )
    receptor = receptor_set.receptor_by_antenna[ANT0]
    native = receptor_matrix(receptor.basis, receptor.feed_rotation_rad) @ response[0]

    data = quadrupolar_components_at(azimuth_uv, zenith_angle)
    expected = convert_native_jones(data, phi)
    transposed = np.einsum("af,ca->fc", data, tangent_conversion(phi))
    conjugated = convert_native_jones(np.conj(data), phi)

    assert float(np.max(np.abs(native - expected))) <= combined_bound(expected, native)
    assert float(np.max(np.abs(native - transposed))) >= SEPARATION_BOUND
    assert float(np.max(np.abs(native - conjugated))) >= SEPARATION_BOUND


@pytest.mark.parametrize("label", sorted(NON_IDENTITY_STORED_BASES))
def test_a_stored_basis_that_is_not_exactly_the_native_identity_is_rejected(
    tmp_path: Path,
    label: str,
) -> None:
    """Corrected Section 5.1.1 item 10: "Any other stored basis -- including
    one pyuvdata itself would tolerate, such as ``0.5*I`` or a negative
    off-diagonal -- is ``UnsupportedBeamBasisError`` with the frozen probe
    kind ``basis_vector_not_identity``."

    All four stored bases below pass ``UVBeam.check``; only RadioSim rejects
    them, which is exactly the point -- pyuvdata would otherwise either crash
    evaluation with an untyped ``NotImplementedError`` or silently substitute
    a different basis than the file declares.
    """
    expected = _beam_error("UnsupportedBeamBasisError")
    beam = build_efield_uvbeam(
        basis_vector_array=constant_basis_vector_array(NON_IDENTITY_STORED_BASES[label])
    )
    assert beam.check(check_extra=True, run_check_acceptability=True) is True

    with pytest.raises(expected):
        load_efield_system(tmp_path, beam=beam, receptors=receptors_block())


@pytest.mark.parametrize("stored_dtype", [np.float32, np.float64])
def test_both_stored_identity_widths_are_accepted(
    tmp_path: Path,
    stored_dtype: Any,
) -> None:
    """Corrected Section 5.1.1 item 10 judges the stored dtype "by **kind and
    width**, never by a byte-order-qualified comparison", and accepts both
    real floating widths "because the identity values ``1.0`` and ``0.0`` are
    exactly representable and round-trip bit-exactly in each".

    This case is what the retired ``basis_vector_dtype`` rejection becomes: the
    ``float32`` exact-identity file the first cut asserted must be rejected is
    accepted under the corrected law.
    """
    root = tmp_path / np.dtype(stored_dtype).name
    written = write_efield_beamfits(
        root,
        basis_vector_array=native_identity_basis_vector_array(dtype=stored_dtype),
    )
    system, _receptors, _state = load_efield_system(
        root,
        path=written.path,
        receptors=receptors_block(),
    )

    handler = system.state.handlers[0]
    assert handler.file is not None
    assert handler.file.sha256 == written.sha256
    response = evaluate(system)
    assert response.shape == (PROBE_ALTITUDE_RAD.size, 2, 2)
    assert np.all(np.isfinite(response))


def test_the_stored_basis_precedence_is_non_finite_then_dtype_then_identity(
    tmp_path: Path,
) -> None:
    """Corrected Section 5.1.1 item 10's frozen precedence.

    Each fixture below violates the identity predicate *as well as* its own
    one, so the reported type is decided by the precedence and by nothing
    else: "a ``NaN`` basis reports non-finiteness rather than a dtype or
    identity failure, and a complex basis reports its dtype rather than
    non-identity".
    """
    matrix = NON_IDENTITY_STORED_BASES["half_identity"]

    non_finite = build_efield_uvbeam()
    basis = constant_basis_vector_array(matrix)
    basis[0, 0, 0, 0] = np.nan
    non_finite.basis_vector_array = basis
    with pytest.raises(_beam_error("NonFiniteBeamResponseError")):
        load_efield_system(
            tmp_path / "non-finite", beam=non_finite, receptors=receptors_block()
        )

    complex_basis = build_efield_uvbeam()
    complex_basis.basis_vector_array = np.array(
        constant_basis_vector_array(matrix), dtype=np.complex128
    )
    with pytest.raises(_beam_error("UnsupportedBeamBasisError")):
        load_efield_system(
            tmp_path / "complex", beam=complex_basis, receptors=receptors_block()
        )

    real_non_identity = build_efield_uvbeam(
        basis_vector_array=constant_basis_vector_array(matrix)
    )
    with pytest.raises(_beam_error("UnsupportedBeamBasisError")):
        load_efield_system(
            tmp_path / "identity", beam=real_non_identity, receptors=receptors_block()
        )


def test_a_returned_interpolation_basis_that_is_not_the_identity_is_an_internal_failure(
    tmp_path: Path,
) -> None:
    """Corrected Section 5.2.1: the returned basis "is requested in order to be
    **verified**, not composed", and an array that is not the identity "means
    the pinned dependency contract has changed beneath RadioSim. It is
    therefore an **internal failure**, not a file rejection, and raises
    ``UnsupportedBeamBasisError`` naming the pinned ``pyuvdata 3.2.1``
    contract."

    No committed file can express that state, because pyuvdata builds the
    returned array from ``numpy.ones`` and ``numpy.zeros`` itself, so the probe
    is a dependency object that violates the pinned return contract on purpose
    while remaining an ordinary accepted file at load.
    """
    beam = forge_interpolation_basis(
        build_efield_uvbeam(science=EfieldScienceVariant.QUADRUPOLAR)
    )
    system, _receptors, _state = load_efield_system(
        tmp_path,
        beam=beam,
        receptors=receptors_block(),
    )

    with pytest.raises(_beam_error("UnsupportedBeamBasisError")) as error:
        evaluate(system)

    assert "3.2.1" in str(error.value)


def test_the_widened_provenance_field_order_is_exact(tmp_path: Path) -> None:
    """Corrected Section 5.2.1 freezes the seven appended
    ``BeamFileProvenance`` fields "by name, order, and annotation, because
    Section 7.4 requires ``S3`` to extend the exact field-order pin ... and an
    unnamed surface cannot be extended twice the same way".

    Section 7.4 assigns the *extension of the accepted pin* in
    ``tests/unit/test_core/test_beam_fits.py`` to ``S3``; this is the
    Stage-3-owned red statement of the same tuple, so the two agree by
    construction once the implementation lands.
    """
    import dataclasses

    from radiosim.core.beam import BeamFileProvenance

    observed = tuple(field.name for field in dataclasses.fields(BeamFileProvenance))
    assert observed == FROZEN_PROVENANCE_FIELD_ORDER
    annotations = {
        field.name: field for field in dataclasses.fields(BeamFileProvenance)
    }
    for name in STAGE3_PROVENANCE_FIELDS:
        assert annotations[name].default is None


def test_a_peak_document_leaves_every_new_provenance_field_none(
    tmp_path: Path,
) -> None:
    """Corrected Section 5.2.1: "Every one is annotated ``<type> | None = None``
    and is left ``None`` on the accepted ``peak`` path, which is what keeps the
    ``None``-omission fingerprint mechanism above intact."

    The companion green control
    :func:`test_a_peak_document_keeps_todays_beam_provenance_snapshot_keys`
    asserts the observable consequence -- that the scalar snapshot's key set
    does not move -- and stays green at every stage.
    """
    written = write_scalar_efield_beamfits(tmp_path)
    system, _receptors, _state = load_efield_system(
        tmp_path,
        path=written.path,
        normalization="peak",
        receptors=receptors_block(),
    )

    provenance = system.state.handlers[0].file
    assert provenance is not None
    for name in STAGE3_PROVENANCE_FIELDS:
        assert getattr(provenance, name) is None


def test_the_full_efield_provenance_records_the_frozen_stage_three_facts(
    tmp_path: Path,
) -> None:
    """Corrected Section 5.2.1 fixes each field's full-efield value exactly.

    ``derived_x_orientation_verdict`` renders item 7's agreed ``None`` as the
    lower-case string ``none`` "so the field stays a ``str | None`` whose
    ``None`` means 'scalar path' rather than 'circular receptor'"; the
    unrotated linear fixture here agrees at ``east``.
    """
    written = write_efield_beamfits(tmp_path)
    system, _receptors, _state = load_efield_system(
        tmp_path,
        path=written.path,
        receptors=receptors_block(),
    )

    provenance = system.state.handlers[0].file
    assert provenance is not None
    assert provenance.accepted_subset_version == FULL_EFIELD_SUBSET_VERSION
    assert provenance.radiosim_normalization == FULL_EFIELD_NORMALIZATION
    assert provenance.resolved_feed_array == ("x", "y")
    assert provenance.derived_x_orientation_verdict == "east"
    assert provenance.basis_vector_convention == BASIS_CONVERSION_CONVENTION
    assert provenance.factorization_convention == FACTORIZATION_CONVENTION

    peaks = provenance.stored_grid_peak_by_frequency
    assert type(peaks) is tuple
    frequencies = [pair[0] for pair in peaks]
    assert frequencies == sorted(frequencies)
    assert len(set(frequencies)) == len(frequencies)
    for _frequency_hz, observed_peak in peaks:
        assert abs(float(observed_peak) - 1.0) <= NORMALIZATION_ATOL


def test_a_rotated_linear_receptor_records_the_none_orientation_verdict(
    tmp_path: Path,
) -> None:
    """Item 7: ``None`` "is a legal, common result -- it is what that function
    returns for a rotated linear receptor" -- and corrected Section 5.2.1
    renders it as the lower-case string ``none``."""
    written = write_efield_beamfits(
        tmp_path,
        feed_rotation_rad=math.radians(ROTATED_FEED_ROTATION_DEG),
    )
    system, _receptors, _state = load_efield_system(
        tmp_path,
        path=written.path,
        receptors=receptors_block(feed_rotation_deg=ROTATED_FEED_ROTATION_DEG),
    )

    provenance = system.state.handlers[0].file
    assert provenance is not None
    assert provenance.derived_x_orientation_verdict == "none"
