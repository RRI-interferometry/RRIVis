"""SCI-005 Stage-1 Ruze coherent loss and scattered-power diagnostic.

``docs/development/sci005_beam_physics_plan.md`` Section 3.4 keeps the accepted
coherent-voltage meaning of ``beams.surface_error`` -- with
:math:`s = 4\\pi\\sigma_h/\\lambda`, :math:`\\langle e\\rangle =
e^{-s^2/2}e_{\\rm det}` and :math:`B_{\\rm coherent}=e^{-s^2}|e_{\\rm det}|^2`
(Ruze 1952, DOI 10.1007/BF02903409; Ruze 1966, DOI 10.1109/PROC.1966.4784) --
and adds one optional *ensemble-power* diagnostic on top of it.

The literal ``gaussian_covariance_power`` names a complete field law, not just a
radial function: a real, zero-mean, jointly Gaussian, second-order stationary
aperture-equivalent surface-error field with
:math:`\\rho_h(\\Delta)=\\exp[-(|\\Delta|/L)^2]`. It is the *characteristic
function* of that jointly Gaussian law, not the covariance alone, that licenses
the mutual-coherence kernel :math:`\\exp\\{-s^2[1-\\rho_h]\\}`; a
covariance-matched non-Gaussian field gives a different kernel, and this module
proves it with an exact discrete counterexample.

Section 3.4.1 forbids the displayed :math:`O(SQ^2)` double integral in
production and fixes ``poisson_gauss_hermite_aperture_v1`` as the only Stage-1
method. The independent :math:`O(Q^2)` pair oracle therefore lives *here*, in
tests only, and is never imported by production.

Section 3.4 also fixes what the diagnostic is *not*: ``sigma`` and ``L`` do not
determine a deterministic complex voltage, ``sqrt(B_main + B_error)`` "would
invent a phase and perfectly correlated structure, so that operation is
forbidden", and "A test requiring that diagnostic power to change a
cross-baseline visibility is itself a design violation."

This module binds ``radiosim.core.beam.aperture`` and the public
``BeamSystem.evaluate_ruze_power_diagnostic``; neither exists yet, so the whole
file is red at collection.
"""

from __future__ import annotations

import dataclasses
import inspect
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from radiosim.core.beam.aperture import (
    STAGE1_SCIENTIFIC_CONVENTIONS,
    RuzePowerConvergence,
    RuzePowerDiagnostic,
)
from radiosim.core.beam.errors import (
    BeamAngularDomainError,
    BeamEvaluationError,
    BeamSamplingDerivationError,
)
from tests.unit.test_core.test_sci005_aperture_physics import (
    _SPEED_OF_LIGHT_M_PER_S,
    ATOL,
    FIXTURE_DIAMETER_M,
    RTOL,
    _analytic_beams,
    _aperture_block,
    _assert_within_frozen_tolerance,
    _beam_system,
    _noll_zernike,
)

#: Section 3.4.1's allowed one-axis Gauss-Hermite orders.
ALLOWED_HERMITE_ORDERS: tuple[int, ...] = (8, 16, 32, 64, 128, 256, 512)

#: Section 3.4.2's exact public result fields, in the memo's declared order.
DIAGNOSTIC_FIELDS: tuple[str, ...] = (
    "schema_version",
    "method",
    "antenna_id",
    "covariance_convention",
    "normalization_convention",
    "frequency_hz",
    "time_mjd",
    "rms_surface_error_m",
    "correlation_length_m",
    "altitude_rad",
    "azimuth_rad",
    "coherent_main_power",
    "total_ensemble_power",
    "scattered_power",
    "convergence",
)

#: Section 3.4.2's exact convergence fields, in the memo's declared order.
CONVERGENCE_FIELDS: tuple[str, ...] = (
    "real_dtype",
    "complex_dtype",
    "poisson_mu",
    "poisson_first_order",
    "poisson_last_order",
    "poisson_term_count",
    "poisson_lower_omitted_mass",
    "poisson_upper_omitted_mass",
    "poisson_total_omitted_mass",
    "poisson_retained_weight_sum",
    "hermite_order",
    "hermite_evaluation_count",
    "hermite_penultimate_max_abs_delta",
    "hermite_final_max_abs_delta",
    "aperture_method",
    "aperture_partition_count",
    "aperture_topology_breakpoint_count",
    "aperture_topology_sha256",
    "aperture_refinement_count",
    "aperture_max_node_count",
    "aperture_penultimate_max_abs_delta",
    "aperture_final_max_abs_delta",
    "aperture_q_max",
    "surface_phase_kappa",
    "surface_radial_derivative_bound",
    "surface_angular_derivative_bound",
    "fhat_evaluation_count",
    "phase_product_count",
    "batch_size",
    "atol",
    "rtol",
    "estimated_peak_bytes",
    "maximum_abs_e_deterministic",
    "minimum_scattered_power",
    "maximum_total_power",
    "returned_balance_max_abs_residual",
)

#: Section 8.1 classifies exactly these convergence fields as exact integers.
CONVERGENCE_INTEGER_FIELDS: frozenset[str] = frozenset(
    {
        "poisson_first_order",
        "poisson_last_order",
        "poisson_term_count",
        "hermite_order",
        "hermite_evaluation_count",
        "aperture_partition_count",
        "aperture_topology_breakpoint_count",
        "aperture_refinement_count",
        "aperture_max_node_count",
        "fhat_evaluation_count",
        "phase_product_count",
        "batch_size",
        "estimated_peak_bytes",
    }
)
CONVERGENCE_STRING_FIELDS: frozenset[str] = frozenset(
    {"real_dtype", "complex_dtype", "aperture_method", "aperture_topology_sha256"}
)

DIAGNOSTIC_SCHEMA = "radiosim.ruze_power_diagnostic.v1"
DIAGNOSTIC_METHOD = "poisson_gauss_hermite_aperture_v1"
COVARIANCE_CONVENTION = "gaussian_one_over_e_surface_covariance_v1"
NORMALIZATION_CONVENTION = "unmodified_ideal_aperture_v1"
APERTURE_METHOD = "boundary_fitted_polar_gauss_legendre_v1"

EMPTY_DIRECTION_MESSAGE = "Ruze power diagnostic requires at least one direction."
UNCONFIGURED_MESSAGE = "A Ruze power diagnostic is not configured for this antenna."

#: The oracle case: a supported uniform pupil, one deterministic defocus mode,
#: a small surface RMS, and a correlation length comparable with the aperture.
ORACLE_MODES: tuple[tuple[int, int, float], ...] = ((2, 0, 0.02),)
ORACLE_RMS_M = 0.02
ORACLE_CORRELATION_M = 2.0

_ALTITUDE_RAD = np.array([1.2, 0.8, 0.4], dtype=np.float64)
_AZIMUTH_RAD = np.array([0.4, 2.0, 4.0], dtype=np.float64)


# --- test-only helpers --------------------------------------------------------


def _diagnostic_block(
    correlation_length_m: float = ORACLE_CORRELATION_M,
) -> dict[str, Any]:
    return {
        "kind": "gaussian_covariance_power",
        "correlation_length_m": correlation_length_m,
    }


def _surface_error(
    rms_surface_error_m: float = ORACLE_RMS_M,
    *,
    correlation_length_m: float = ORACLE_CORRELATION_M,
    diagnostic: bool = True,
) -> dict[str, Any]:
    default: dict[str, Any] = {"rms_surface_error_m": rms_surface_error_m}
    if diagnostic:
        default["error_beam_diagnostic"] = _diagnostic_block(correlation_length_m)
    return {"default": default}


def _oracle_beams(
    *,
    rms_surface_error_m: float = ORACLE_RMS_M,
    correlation_length_m: float = ORACLE_CORRELATION_M,
    modes: tuple[tuple[int, int, float], ...] = ORACLE_MODES,
    diagnostic: bool = True,
) -> dict[str, Any]:
    aperture = (
        _aperture_block(
            zernike_modes=[
                {"n": n, "m": m, "surface_height_coefficient_m": c} for n, m, c in modes
            ]
        )
        if modes
        else None
    )
    return _analytic_beams(
        aperture_physics=aperture,
        surface_error=_surface_error(
            rms_surface_error_m,
            correlation_length_m=correlation_length_m,
            diagnostic=diagnostic,
        ),
    )


def _evaluate(
    system: Any,
    frequency_hz: float,
    *,
    altitude_rad: np.ndarray = _ALTITUDE_RAD,
    azimuth_rad: np.ndarray = _AZIMUTH_RAD,
) -> Any:
    from radiosim.core.instrument import AntennaId

    return system.evaluate_ruze_power_diagnostic(
        AntennaId(0, "ANT0"),
        altitude_rad=altitude_rad,
        azimuth_rad=azimuth_rad,
        frequency_hz=frequency_hz,
        time_mjd=60000.0,
    )


def _aperture_nodes(
    modes: tuple[tuple[int, int, float], ...],
    *,
    radial_order: int = 32,
    angular_order: int = 64,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(north_m, east_m, weight, height_m)`` for the unmasked disk.

    The weights already carry the ``1/pi`` of Section 3.3's normalized integral,
    so a plain weighted sum of the phase factors *is* ``e_det``. Thirty-two
    radial by sixty-four angular nodes reproduce the pair integral to better
    than 1e-14 for these directions, which is well inside the frozen tolerance.
    """
    abscissa, weights = np.polynomial.legendre.leggauss(radial_order)
    radial = 0.5 * (abscissa + 1.0)
    radial_weight = 0.5 * weights * radial
    angular = 2.0 * np.pi * np.arange(angular_order) / angular_order
    angular_weight = np.full(angular_order, 2.0 * np.pi / angular_order)
    grid_weight = np.outer(radial_weight, angular_weight) / np.pi
    rho = np.broadcast_to(radial[:, None], grid_weight.shape).ravel().copy()
    phi = np.broadcast_to(angular[None, :], grid_weight.shape).ravel().copy()
    height = np.zeros_like(rho)
    for n, m, coefficient in modes:
        height = height + coefficient * _noll_zernike(n, m, rho, phi)
    radius = 0.5 * FIXTURE_DIAMETER_M
    return (
        radius * rho * np.cos(phi),
        radius * rho * np.sin(phi),
        grid_weight.ravel().copy(),
        height,
    )


def _pair_oracle(
    *,
    altitude_rad: np.ndarray,
    azimuth_rad: np.ndarray,
    wavelength_m: float,
    rms_surface_error_m: float,
    correlation_length_m: float,
    modes: tuple[tuple[int, int, float], ...],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Section 3.4's displayed double integral, evaluated pair by pair.

    This is the forbidden-in-production ``O(SQ^2)`` form, written independently
    here so that a factor of two, a missing ``1/pi``, a dropped deterministic
    phase difference, a flipped forward-transform sign, or a renormalized
    ``N_0`` all show up as a residual.
    """
    north, east, weight, height = _aperture_nodes(modes)
    kappa = 4.0 * np.pi / wavelength_m
    wavenumber = 2.0 * np.pi / wavelength_m
    s_squared = (kappa * rms_surface_error_m) ** 2
    coherent = np.empty(altitude_rad.shape, dtype=np.float64)
    total = np.empty(altitude_rad.shape, dtype=np.float64)
    detector_phase = np.exp(-1j * kappa * height)
    block = 256
    for index in range(altitude_rad.size):
        q_north = wavenumber * np.cos(altitude_rad[index]) * np.cos(azimuth_rad[index])
        q_east = wavenumber * np.cos(altitude_rad[index]) * np.sin(azimuth_rad[index])
        factor = (
            weight * detector_phase * np.exp(-1j * (q_north * north + q_east * east))
        )
        coherent[index] = math.exp(-s_squared) * abs(complex(np.sum(factor))) ** 2
        accumulated = 0.0
        for start in range(0, north.size, block):
            stop = min(start + block, north.size)
            delta_north = north[start:stop, None] - north[None, :]
            delta_east = east[start:stop, None] - east[None, :]
            correlation = np.exp(
                -s_squared
                * (
                    1.0
                    - np.exp(
                        -(delta_north**2 + delta_east**2) / correlation_length_m**2
                    )
                )
            )
            accumulated += float(
                np.real(
                    np.sum(
                        factor[start:stop, None]
                        * np.conj(factor)[None, :]
                        * correlation
                    )
                )
            )
        total[index] = accumulated
    return coherent, total, total - coherent


def _first_allowed_hermite_order(floor: int) -> int:
    for order in ALLOWED_HERMITE_ORDERS:
        if order >= floor:
            return order
    raise AssertionError(f"no allowed Hermite order reaches {floor}")


def _hermite_floor(*, last_order: int, correlation_length_m: float) -> int:
    """Section 3.4.1: ``H_floor = 8 + ceil(4*sqrt(m_last)*D/L)``."""
    return 8 + math.ceil(
        4.0 * math.sqrt(last_order) * FIXTURE_DIAMETER_M / correlation_length_m
    )


@pytest.fixture(scope="module")
def oracle_case(tmp_path_factory: pytest.TempPathFactory) -> tuple[Any, float, Any]:
    system, frequency_hz = _beam_system(
        tmp_path_factory.mktemp("ruze-oracle"), _oracle_beams()
    )
    return system, frequency_hz, _evaluate(system, frequency_hz)


# --- Section 3.4.2: the frozen public result ----------------------------------


def test_conventions_record_names_the_covariance_and_method_literals() -> None:
    """Section 8.1's convention record carries both Ruze literals."""
    assert STAGE1_SCIENTIFIC_CONVENTIONS["ruze_covariance"] == COVARIANCE_CONVENTION
    assert STAGE1_SCIENTIFIC_CONVENTIONS["ruze_method"] == DIAGNOSTIC_METHOD


def test_frozen_diagnostic_fields_types_and_literals(
    oracle_case: tuple[Any, float, Any],
) -> None:
    """Section 3.4.2's exact field list, literals, and scalar types."""
    _system, frequency_hz, diagnostic = oracle_case

    assert type(diagnostic) is RuzePowerDiagnostic
    assert tuple(f.name for f in dataclasses.fields(diagnostic)) == DIAGNOSTIC_FIELDS
    assert diagnostic.schema_version == DIAGNOSTIC_SCHEMA
    assert diagnostic.method == DIAGNOSTIC_METHOD
    assert diagnostic.covariance_convention == COVARIANCE_CONVENTION
    assert diagnostic.normalization_convention == NORMALIZATION_CONVENTION
    for name in (
        "frequency_hz",
        "time_mjd",
        "rms_surface_error_m",
        "correlation_length_m",
    ):
        value = getattr(diagnostic, name)
        assert type(value) is float
        assert math.isfinite(value)
    assert diagnostic.frequency_hz == frequency_hz
    assert diagnostic.rms_surface_error_m == ORACLE_RMS_M
    assert diagnostic.correlation_length_m == ORACLE_CORRELATION_M
    assert diagnostic.frequency_hz > 0.0
    assert diagnostic.rms_surface_error_m > 0.0
    assert diagnostic.correlation_length_m > 0.0


def test_frozen_result_is_immutable_and_rejects_unknown_fields(
    oracle_case: tuple[Any, float, Any],
) -> None:
    """Section 3.4.2: "The dataclasses are frozen, final, and slotted"."""
    _system, _frequency_hz, diagnostic = oracle_case

    for record in (diagnostic, diagnostic.convergence):
        assert dataclasses.is_dataclass(record)
        assert type(record).__dataclass_params__.frozen is True
        assert hasattr(type(record), "__slots__")
        assert not hasattr(record, "__dict__")
        with pytest.raises(dataclasses.FrozenInstanceError):
            record.schema_version = "mutated"  # pyright: ignore[reportAttributeAccessIssue]
    with pytest.raises(TypeError):
        RuzePowerDiagnostic(unknown_field=1)  # pyright: ignore[reportCallIssue]


def test_diagnostic_arrays_are_owned_read_only_and_c_contiguous(
    oracle_case: tuple[Any, float, Any],
) -> None:
    """Section 3.4.2: "owned, C-contiguous, read-only" arrays of shape ``(S,)``."""
    _system, _frequency_hz, diagnostic = oracle_case
    shape = (_ALTITUDE_RAD.size,)

    for name in ("altitude_rad", "azimuth_rad"):
        array = getattr(diagnostic, name)
        assert type(array) is np.ndarray
        assert array.dtype == np.dtype(np.float64)
        assert array.shape == shape
        assert array.flags.c_contiguous
        assert array.flags.owndata
        assert not array.flags.writeable
    real_dtype = np.dtype(diagnostic.convergence.real_dtype)
    for name in ("coherent_main_power", "total_ensemble_power", "scattered_power"):
        array = getattr(diagnostic, name)
        assert type(array) is np.ndarray
        assert array.dtype == real_dtype
        assert array.shape == shape
        assert array.flags.c_contiguous
        assert array.flags.owndata
        assert not array.flags.writeable
    np.testing.assert_array_equal(diagnostic.altitude_rad, _ALTITUDE_RAD)
    np.testing.assert_array_equal(diagnostic.azimuth_rad, _AZIMUTH_RAD)


def test_frozen_convergence_field_order_and_scalar_classification(
    oracle_case: tuple[Any, float, Any],
) -> None:
    """Section 3.4.2's declared order and Section 8.1's type classification."""
    _system, _frequency_hz, diagnostic = oracle_case
    convergence = diagnostic.convergence

    assert type(convergence) is RuzePowerConvergence
    assert tuple(f.name for f in dataclasses.fields(convergence)) == CONVERGENCE_FIELDS
    assert (convergence.real_dtype, convergence.complex_dtype) in (
        ("float32", "complex64"),
        ("float64", "complex128"),
    )
    assert convergence.aperture_method == APERTURE_METHOD
    digest = convergence.aperture_topology_sha256
    assert type(digest) is str
    assert len(digest) == 64
    assert all(character in "0123456789abcdef" for character in digest)
    for name in CONVERGENCE_INTEGER_FIELDS:
        value = getattr(convergence, name)
        assert type(value) is int
        assert value >= 0
    for field in dataclasses.fields(convergence):
        if field.name in CONVERGENCE_INTEGER_FIELDS | CONVERGENCE_STRING_FIELDS:
            continue
        value = getattr(convergence, field.name)
        assert type(value) is float
        assert math.isfinite(value)
        assert value >= 0.0
    # Section 3.4.2: there is no ``converged`` field and no false state.
    assert not hasattr(convergence, "converged")


def test_the_frozen_tolerances_are_the_section_three_values(
    oracle_case: tuple[Any, float, Any],
) -> None:
    """Section 3.3 freezes ``atol``/``rtol``; they cannot be authored in YAML."""
    _system, _frequency_hz, diagnostic = oracle_case
    eps = float(np.finfo(np.dtype(diagnostic.convergence.real_dtype)).eps)

    assert diagnostic.convergence.atol == max(1e-12, 32.0 * eps)
    assert diagnostic.convergence.rtol == max(1e-10, 32.0 * eps)


def test_public_method_is_host_side_and_takes_no_backend_argument() -> None:
    """Section 3.4.2: "never accepts a backend argument: the complete algorithm
    is host-side"."""
    from radiosim.core.beam.runtime import BeamSystem

    signature = inspect.signature(BeamSystem.evaluate_ruze_power_diagnostic)

    assert "backend" not in signature.parameters
    assert list(signature.parameters) == [
        "self",
        "antenna_id",
        "altitude_rad",
        "azimuth_rad",
        "frequency_hz",
        "time_mjd",
    ]
    for name in ("altitude_rad", "azimuth_rad", "frequency_hz", "time_mjd"):
        assert signature.parameters[name].kind is inspect.Parameter.KEYWORD_ONLY


def test_no_error_beam_voltage_is_created_anywhere(
    oracle_case: tuple[Any, float, Any],
) -> None:
    """Section 3.4: "Taking ``sqrt(B_main+B_error)`` would invent a phase and
    perfectly correlated structure, so that operation is forbidden."."""
    import radiosim.core.beam.aperture as aperture_module

    _system, _frequency_hz, diagnostic = oracle_case

    forbidden = [
        name
        for name in dir(aperture_module)
        if not name.startswith("_")
        and ("voltage" in name.lower() or "error_beam" in name.lower())
    ]
    assert forbidden == []
    for field in dataclasses.fields(diagnostic):
        value = getattr(diagnostic, field.name)
        if isinstance(value, np.ndarray):
            assert value.dtype.kind == "f", (
                f"{field.name} is complex; the diagnostic reports ensemble power, "
                "never a voltage"
            )


# --- Section 3.4.2: input contract --------------------------------------------


def test_empty_direction_batch_raises_the_exact_message(tmp_path: Path) -> None:
    """Section 3.4.2: the diagnostic requires ``S >= 1``."""
    system, frequency_hz = _beam_system(tmp_path, _oracle_beams())

    with pytest.raises(BeamAngularDomainError) as error:
        _evaluate(
            system,
            frequency_hz,
            altitude_rad=np.array([], dtype=np.float64),
            azimuth_rad=np.array([], dtype=np.float64),
        )

    assert str(error.value) == EMPTY_DIRECTION_MESSAGE


def test_an_unconfigured_antenna_raises_the_exact_message(tmp_path: Path) -> None:
    """Section 3.4.2: available only when the antenna carries the nested block."""
    system, frequency_hz = _beam_system(tmp_path, _oracle_beams(diagnostic=False))

    with pytest.raises(BeamEvaluationError) as error:
        _evaluate(system, frequency_hz)

    assert str(error.value) == UNCONFIGURED_MESSAGE


# --- Section 3.4.1: the required positive covariance mixture -------------------


def test_small_node_pair_oracle_agrees_with_the_poisson_gauss_hermite_result(
    oracle_case: tuple[Any, float, Any],
) -> None:
    """Section 3.5: the independent ``O(Q^2)`` pair oracle, in tests only."""
    _system, frequency_hz, diagnostic = oracle_case
    wavelength_m = _SPEED_OF_LIGHT_M_PER_S / frequency_hz

    coherent, total, scattered = _pair_oracle(
        altitude_rad=_ALTITUDE_RAD,
        azimuth_rad=_AZIMUTH_RAD,
        wavelength_m=wavelength_m,
        rms_surface_error_m=ORACLE_RMS_M,
        correlation_length_m=ORACLE_CORRELATION_M,
        modes=ORACLE_MODES,
    )

    _assert_within_frozen_tolerance(
        np.asarray(diagnostic.coherent_main_power, dtype=np.float64), coherent
    )
    _assert_within_frozen_tolerance(
        np.asarray(diagnostic.total_ensemble_power, dtype=np.float64), total
    )
    _assert_within_frozen_tolerance(
        np.asarray(diagnostic.scattered_power, dtype=np.float64), scattered
    )
    assert float(np.max(scattered)) > 0.0


def test_the_deterministic_phase_enters_the_ensemble_average(
    tmp_path: Path,
    oracle_case: tuple[Any, float, Any],
) -> None:
    """Section 3.4: "the ensemble-average power therefore includes the
    deterministic phase difference"."""
    _system, frequency_hz, with_surface = oracle_case
    flat_system, flat_frequency = _beam_system(tmp_path, _oracle_beams(modes=()))

    flat = _evaluate(flat_system, flat_frequency)

    assert flat_frequency == frequency_hz
    difference = float(
        np.max(
            np.abs(
                np.asarray(flat.total_ensemble_power, dtype=np.float64)
                - np.asarray(with_surface.total_ensemble_power, dtype=np.float64)
            )
        )
    )
    assert difference > 1e-6
    assert flat.convergence.surface_radial_derivative_bound == 0.0
    assert flat.convergence.surface_angular_derivative_bound == 0.0
    assert with_surface.convergence.surface_radial_derivative_bound > 0.0


def test_infinite_correlation_length_gives_the_closed_scattered_identity(
    tmp_path: Path,
) -> None:
    """Section 3.5: the ``L -> infinity`` identity
    :math:`B_{\\rm sc}=(1-e^{-\\mu})|e_{\\rm det}|^2`."""
    system, frequency_hz = _beam_system(
        tmp_path, _oracle_beams(correlation_length_m=1.0e9)
    )

    diagnostic = _evaluate(system, frequency_hz)

    mu = diagnostic.convergence.poisson_mu
    coherent = np.asarray(diagnostic.coherent_main_power, dtype=np.float64)
    # B_coherent = exp(-mu)*|e_det|^2, so |e_det|^2 = B_coherent*exp(mu).
    expected = -math.expm1(-mu) * coherent * math.exp(mu)
    _assert_within_frozen_tolerance(
        np.asarray(diagnostic.scattered_power, dtype=np.float64), expected
    )


def test_a_vanishing_mu_resolves_the_frozen_zero_term_poisson_case(
    tmp_path: Path,
) -> None:
    """Section 3.4.1's exact zero-term state, including its positive zero."""
    system, frequency_hz = _beam_system(
        tmp_path, _oracle_beams(rms_surface_error_m=1.0e-8)
    )

    diagnostic = _evaluate(system, frequency_hz)

    convergence = diagnostic.convergence
    assert -math.expm1(-convergence.poisson_mu) <= convergence.atol / 8.0
    assert (convergence.poisson_first_order, convergence.poisson_last_order) == (0, 0)
    assert convergence.poisson_term_count == 0
    assert convergence.poisson_retained_weight_sum == 0.0
    assert convergence.hermite_order == 0
    assert convergence.hermite_evaluation_count == 0
    assert convergence.hermite_penultimate_max_abs_delta == 0.0
    assert convergence.hermite_final_max_abs_delta == 0.0
    assert convergence.poisson_lower_omitted_mass == 0.0
    assert convergence.poisson_upper_omitted_mass == -math.expm1(
        -convergence.poisson_mu
    )
    assert convergence.poisson_total_omitted_mass == -math.expm1(
        -convergence.poisson_mu
    )
    scattered = np.asarray(diagnostic.scattered_power)
    np.testing.assert_array_equal(scattered, np.zeros_like(scattered))
    assert not np.any(np.signbit(scattered))


def test_two_sided_poisson_tail_is_contiguous_and_internally_consistent(
    oracle_case: tuple[Any, float, Any],
) -> None:
    """Section 3.4.1: a contiguous retained interval and a bounded two-sided
    tail, with no renormalization of the retained weights."""
    _system, _frequency_hz, diagnostic = oracle_case
    convergence = diagnostic.convergence

    assert convergence.poisson_first_order >= 1
    assert convergence.poisson_last_order >= convergence.poisson_first_order
    assert convergence.poisson_term_count == (
        convergence.poisson_last_order - convergence.poisson_first_order + 1
    )
    assert convergence.poisson_term_count <= 256
    assert (
        convergence.poisson_lower_omitted_mass + convergence.poisson_upper_omitted_mass
        == convergence.poisson_total_omitted_mass
    )
    assert convergence.poisson_total_omitted_mass <= convergence.atol / 8.0
    # Retained weights are never renormalized, so they stay below the whole mass.
    assert convergence.poisson_retained_weight_sum <= -math.expm1(
        -convergence.poisson_mu
    )
    assert convergence.poisson_mu == pytest.approx(
        (convergence.surface_phase_kappa * diagnostic.rms_surface_error_m) ** 2,
        rel=RTOL,
        abs=ATOL,
    )
    wavelength_m = _SPEED_OF_LIGHT_M_PER_S / diagnostic.frequency_hz
    assert convergence.surface_phase_kappa == pytest.approx(
        4.0 * math.pi / wavelength_m, rel=RTOL, abs=ATOL
    )


def test_hermite_order_respects_the_frozen_floor_and_allowed_set(
    oracle_case: tuple[Any, float, Any],
) -> None:
    """Section 3.4.1: "no convergence comparison may count below the first
    allowed order not less than ``H_floor``"."""
    _system, _frequency_hz, diagnostic = oracle_case
    convergence = diagnostic.convergence

    floor = _hermite_floor(
        last_order=convergence.poisson_last_order,
        correlation_length_m=diagnostic.correlation_length_m,
    )
    assert convergence.hermite_order in ALLOWED_HERMITE_ORDERS
    assert convergence.hermite_order >= _first_allowed_hermite_order(floor)
    # Two consecutive successful comparisons are retained separately.
    assert convergence.hermite_final_max_abs_delta <= (
        0.25
        * (
            convergence.atol
            + convergence.rtol * float(np.max(np.abs(diagnostic.total_ensemble_power)))
        )
    )
    assert convergence.hermite_evaluation_count > 0
    assert convergence.aperture_refinement_count >= 0
    assert convergence.aperture_final_max_abs_delta <= (
        convergence.atol + convergence.rtol * convergence.maximum_abs_e_deterministic
    )


def test_more_than_256_retained_poisson_terms_fails_before_any_evaluation(
    tmp_path: Path,
) -> None:
    """Section 3.4.1: "More than 256 retained terms raises
    ``BeamSamplingDerivationError`` before any aperture evaluation"."""
    system, frequency_hz = _beam_system(
        tmp_path, _oracle_beams(rms_surface_error_m=5.0)
    )

    with pytest.raises(BeamSamplingDerivationError):
        _evaluate(system, frequency_hz)


def test_no_room_above_the_hermite_floor_fails_before_it_starts(
    tmp_path: Path,
) -> None:
    """Section 3.4.1: "There must be room for two higher allowed orders or
    evaluation fails before it starts"."""
    system, frequency_hz = _beam_system(
        tmp_path, _oracle_beams(correlation_length_m=0.05)
    )

    assert _hermite_floor(last_order=1, correlation_length_m=0.05) > max(
        ALLOWED_HERMITE_ORDERS
    )
    with pytest.raises(BeamSamplingDerivationError):
        _evaluate(system, frequency_hz)


def test_entire_plane_shifted_wavevectors_avoid_the_sky_domain_gate(
    oracle_case: tuple[Any, float, Any],
) -> None:
    """Section 3.4.1: the internal helper "accepts every finite real two-vector"
    and "never applies a sky angular-domain or horizon check"."""
    _system, frequency_hz, diagnostic = oracle_case
    wavelength_m = _SPEED_OF_LIGHT_M_PER_S / frequency_hz
    radius = 0.5 * FIXTURE_DIAMETER_M

    base_q_max = float(
        np.max(
            radius
            * (2.0 * np.pi / wavelength_m)
            * np.cos(np.asarray(diagnostic.altitude_rad, dtype=np.float64))
        )
    )

    # A Hermite-shifted wavevector leaves the visible sky disk entirely.
    assert diagnostic.convergence.aperture_q_max > base_q_max
    assert diagnostic.convergence.fhat_evaluation_count > _ALTITUDE_RAD.size
    assert diagnostic.convergence.phase_product_count > 0
    assert diagnostic.convergence.batch_size >= 1
    assert diagnostic.convergence.batch_size <= 256


# --- Section 3.4: exact balance, bounds, and non-negativity --------------------


def test_power_balance_is_exact_in_the_result_dtype_without_clipping(
    oracle_case: tuple[Any, float, Any],
) -> None:
    """Section 3.4.1: "The returned balance is exact in the result dtype" and
    "There is no clipping"."""
    _system, _frequency_hz, diagnostic = oracle_case
    real_dtype = np.dtype(diagnostic.convergence.real_dtype)

    total = np.asarray(diagnostic.total_ensemble_power)
    rebuilt = np.asarray(
        diagnostic.coherent_main_power + diagnostic.scattered_power, dtype=real_dtype
    )

    np.testing.assert_array_equal(total, rebuilt)
    assert diagnostic.convergence.returned_balance_max_abs_residual == 0.0
    assert np.all(np.asarray(diagnostic.scattered_power) >= 0.0)
    assert np.all(np.isfinite(total))
    assert diagnostic.convergence.minimum_scattered_power >= 0.0


def test_amplitude_and_total_power_bounds_hold_within_the_frozen_tolerance(
    oracle_case: tuple[Any, float, Any],
) -> None:
    """Section 3.4.1: :math:`|e_{\\rm det}| \\le 1` proves non-negative
    scattered power and bounds the omitted Poisson mass."""
    _system, _frequency_hz, diagnostic = oracle_case
    convergence = diagnostic.convergence
    limit = convergence.atol + convergence.rtol

    assert convergence.maximum_abs_e_deterministic <= 1.0 + limit
    assert convergence.maximum_total_power <= 1.0 + limit
    assert float(np.max(np.asarray(diagnostic.total_ensemble_power))) <= 1.0 + limit


# --- Section 3.5: the diagnostic is not a Jones voltage -----------------------


def test_requesting_the_diagnostic_never_calls_or_changes_evaluate_jones(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Section 3.5: "proof by spy and data-flow test that requesting the
    diagnostic neither calls nor changes ``evaluate_jones``"."""
    from radiosim.core.beam.runtime import BeamSystem
    from radiosim.core.instrument import AntennaId

    system, frequency_hz = _beam_system(tmp_path, _oracle_beams())
    arguments = {
        "altitude_rad": _ALTITUDE_RAD,
        "azimuth_rad": _AZIMUTH_RAD,
        "frequency_hz": frequency_hz,
        "time_mjd": 60000.0,
    }
    before = np.array(system.evaluate_jones(AntennaId(0, "ANT0"), **arguments))

    calls: list[object] = []
    original = BeamSystem.evaluate_jones

    def _spy(self: Any, antenna_id: Any, **kwargs: Any) -> Any:
        calls.append(antenna_id)
        return original(self, antenna_id, **kwargs)

    monkeypatch.setattr(BeamSystem, "evaluate_jones", _spy)
    diagnostic = _evaluate(system, frequency_hz)
    monkeypatch.undo()

    assert calls == []
    assert diagnostic.schema_version == DIAGNOSTIC_SCHEMA
    after = np.array(system.evaluate_jones(AntennaId(0, "ANT0"), **arguments))
    np.testing.assert_array_equal(before, after)


def test_repeated_evaluation_does_not_mutate_the_retained_record(
    oracle_case: tuple[Any, float, Any],
) -> None:
    """Section 3.5: "repeated evaluation does not mutate either"."""
    system, frequency_hz, first = oracle_case

    second = _evaluate(system, frequency_hz)

    for name in ("coherent_main_power", "total_ensemble_power", "scattered_power"):
        np.testing.assert_array_equal(getattr(first, name), getattr(second, name))
    assert first.convergence == second.convergence
    assert first.altitude_rad is not second.altitude_rad


# --- Section 3.5: the covariance kernel is licensed by the field law ----------


def _gaussian_characteristic_function(
    *, s_squared: float, correlation: float, order: int = 200
) -> float:
    """``E[exp(-i(X - Y))]`` for a centred bivariate normal, by quadrature.

    ``X`` and ``Y`` each have variance ``s_squared`` and correlation
    ``correlation``; the expectation is real by symmetry.
    """
    nodes, weights = np.polynomial.hermite.hermgauss(order)
    weights = weights / math.sqrt(math.pi)
    sigma = math.sqrt(s_squared)
    # X = sigma*a, Y = sigma*(corr*a + sqrt(1-corr^2)*b) with a, b standard normal.
    first = math.sqrt(2.0) * nodes
    second = math.sqrt(2.0) * nodes
    difference = sigma * (
        first[:, None] * (1.0 - correlation)
        - math.sqrt(max(1.0 - correlation**2, 0.0)) * second[None, :]
    )
    grid = np.outer(weights, weights)
    return float(np.sum(grid * np.cos(difference)))


@pytest.mark.parametrize("correlation", [0.0, 0.25, 0.6, 0.9])
def test_the_jointly_gaussian_characteristic_function_licenses_the_kernel(
    correlation: float,
) -> None:
    """Section 3.4: "the Gaussian characteristic function, rather than
    covariance alone, licenses the mutual-coherence kernel"."""
    s_squared = 0.8

    observed = _gaussian_characteristic_function(
        s_squared=s_squared, correlation=correlation
    )

    _assert_within_frozen_tolerance(
        observed, math.exp(-s_squared * (1.0 - correlation))
    )


@pytest.mark.parametrize("correlation", [0.0, 0.25, 0.6])
def test_a_covariance_matched_non_gaussian_field_breaks_the_kernel(
    correlation: float,
) -> None:
    """Section 3.5's required counterexample: covariance alone is not enough.

    Let the phase take the two values ``+/-s`` with ``P(equal) = (1+rho)/2``.
    Its covariance is exactly ``s^2 rho``, matching the Gaussian field, but its
    characteristic function is a cosine mixture rather than ``exp[-s^2(1-rho)]``.
    """
    s_squared = 0.8
    s = math.sqrt(s_squared)

    matched_covariance = s_squared * (
        (1.0 + correlation) / 2.0 - (1.0 - correlation) / 2.0
    )
    characteristic = (1.0 + correlation) / 2.0 + (1.0 - correlation) / 2.0 * math.cos(
        2.0 * s
    )

    _assert_within_frozen_tolerance(matched_covariance, s_squared * correlation)
    assert abs(characteristic - math.exp(-s_squared * (1.0 - correlation))) > 1e-3
