"""SCI-005 Stage-1 analytic invariants for the one scalar aperture transform.

``docs/development/sci005_beam_physics_plan.md`` Sections 3.1--3.3 define a
single normalized aperture integral

.. math::

    e(\\mathbf q,\\lambda)=\\frac{1}{N_0}\\int_{\\mathcal P_0}
    A(\\mathbf u)M(\\mathbf u)
    \\exp[-i(4\\pi/\\lambda)h(\\mathbf u)]\\exp(-i\\mathbf q\\cdot\\mathbf u)
    \\,d^2u,
    \\qquad N_0=\\int_{\\mathcal P_0}A(\\mathbf u)\\,d^2u,

in which the central blockage, the support shadows, and the deterministic
Zernike surface height are a mask and a phase *inside one integral*. ``N_0`` is
always the unmodified ideal-aperture integral, the modified beam is never
re-peak-normalized, and separately evaluated far-field patterns may never be
multiplied as if the Fourier transform distributed over aperture
multiplication.

Every oracle below is a published closed form evaluated independently in the
test body: the blocked uniform aperture
:math:`e_\\epsilon(x)=2[J_1(x)-\\epsilon J_1(\\epsilon x)]/x` with
:math:`e_\\epsilon(0)=1-\\epsilon^2` and :math:`\\eta_b=(1-\\epsilon^2)^2`
(Section 3.2; NASA TM X-63186, ITU-R SA.2401-0), the real unit-RMS disk Zernike
basis of R. J. Noll, *Zernike polynomials and atmospheric turbulence*, JOSA
**66**, 207 (1976), DOI 10.1364/JOSA.66.000207, and the three normalized Hankel
transforms :math:`2J_1(x)/x`, :math:`8J_2(x)/x^2`, :math:`48J_3(x)/x^3` of the
Section 3.1 pupil-profile table. Mahajan, JOSA **71**, 75 (1981) is why the
authored coefficients stay in the *unobscured* disk basis rather than an
annular one, and why the quadrature sum of coefficients is not the RMS over the
transmitting annulus.

The comparison tolerances are the ones Section 3.3 freezes,
``atol = max(1e-12, 32*eps)`` and ``rtol = max(1e-10, 32*eps)``; nothing here
invents a tolerance.

This module binds the Stage-1 surface of ``radiosim.core.beam.aperture``, the
Section 7.2 production owner of the pupil profiles, support mask, Zernike
phase, and frozen diagnostic records. That module does not exist yet, so the
whole file is red at collection.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from scipy.special import jv

from radiosim.core.beam.aperture import (
    STAGE1_SCIENTIFIC_CONVENTIONS,
    ZERNIKE_MAX_RADIAL_ORDER,
    aperture_transmission_mask,
    gauss_legendre_order_seed,
    real_unit_rms_zernike,
    support_leg_half_angle,
    surface_height_phase,
    zernike_surface_height,
)
from radiosim.core.precision import COMPLEX256_AVAILABLE
from tests.fixtures.configs import valid_config_mapping

_SPEED_OF_LIGHT_M_PER_S = 299792458.0

#: Section 3.3's frozen convergence tolerances at float64 width.
_EPS = float(np.finfo(np.float64).eps)
ATOL = max(1e-12, 32.0 * _EPS)
RTOL = max(1e-10, 32.0 * _EPS)

#: Section 8.1 freezes this exact mapping as the Stage-1 convention record.
EXPECTED_CONVENTIONS: dict[str, str] = {
    "pupil_profile_set": "radiosim.circular_stage1_pupil_profiles.v1",
    "aperture_normalization": "unmodified_ideal_aperture_v1",
    "aperture_axes": "north_east_azimuth_north_through_east_v1",
    "support_mask": "radiosim.central_disk_outward_half_strip_ne.v1",
    "zernike_surface": "radiosim.real_unit_rms_disk_surface_height.v1",
    "aperture_method": "boundary_fitted_polar_gauss_legendre_v1",
    "ruze_covariance": "gaussian_one_over_e_surface_covariance_v1",
    "ruze_method": "poisson_paired_pupil_separation_v1",
}

ZERNIKE_CONVENTION = EXPECTED_CONVENTIONS["zernike_surface"]
APERTURE_NORMALIZATION = EXPECTED_CONVENTIONS["aperture_normalization"]

#: The fixture layout gives both antennas this diameter.
FIXTURE_DIAMETER_M = 14.0

_ALTITUDE_RAD = np.array([np.pi / 2.0, 1.3, 1.0, 0.7, 0.4], dtype=np.float64)
_AZIMUTH_RAD = np.array([0.0, 0.3, 1.2, 2.5, 4.0], dtype=np.float64)


def _assert_within_frozen_tolerance(
    observed: np.ndarray | complex | float,
    expected: np.ndarray | complex | float,
) -> None:
    """Apply Section 3.3's exact comparison, ``atol + rtol*max(abs(refined))``.

    The frozen predicate uses one array-wide limit rather than an elementwise
    relative tolerance, so a direction whose response is near a null is not held
    to a tighter standard than the quadrature ever promised.
    """
    observed_array = np.asarray(observed)
    expected_array = np.asarray(expected)
    assert observed_array.shape == expected_array.shape
    limit = ATOL + RTOL * float(np.max(np.abs(expected_array)))
    residual = float(np.max(np.abs(observed_array - expected_array)))
    assert residual <= limit, f"residual {residual!r} exceeds frozen limit {limit!r}"


# --- independent oracles ------------------------------------------------------


def _noll_radial(n: int, m: int, rho: np.ndarray) -> np.ndarray:
    """Section 3.3's radial polynomial, summed term by term in the test body."""
    order = abs(m)
    total = np.zeros_like(np.asarray(rho, dtype=np.float64))
    for s in range((n - order) // 2 + 1):
        coefficient = (
            (-1.0) ** s
            * math.factorial(n - s)
            / (
                math.factorial(s)
                * math.factorial((n + order) // 2 - s)
                * math.factorial((n - order) // 2 - s)
            )
        )
        total = total + coefficient * np.asarray(rho, dtype=np.float64) ** (n - 2 * s)
    return total


def _noll_zernike(n: int, m: int, rho: np.ndarray, phi: np.ndarray) -> np.ndarray:
    """Section 3.3's real unit-RMS basis (Noll 1976), evaluated independently."""
    radial = _noll_radial(n, m, rho)
    if m == 0:
        return math.sqrt(n + 1.0) * radial
    normalization = math.sqrt(2.0 * (n + 1.0))
    if m > 0:
        return normalization * radial * np.cos(m * np.asarray(phi, dtype=np.float64))
    return normalization * radial * np.sin(abs(m) * np.asarray(phi, dtype=np.float64))


def _disk_quadrature(
    radial_order: int = 64,
    angular_order: int = 256,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(rho, phi, weight)`` for ``(1/pi) * integral over the unit disk``.

    Gauss-Legendre in the radius and the periodic trapezoid rule in the angle,
    both built here rather than imported from production.
    """
    nodes, weights = np.polynomial.legendre.leggauss(radial_order)
    rho = 0.5 * (nodes + 1.0)
    radial_weight = 0.5 * weights * rho
    phi = 2.0 * np.pi * np.arange(angular_order) / angular_order
    angular_weight = np.full(angular_order, 2.0 * np.pi / angular_order)
    weight = np.outer(radial_weight, angular_weight) / np.pi
    return (
        np.broadcast_to(rho[:, None], weight.shape).copy(),
        np.broadcast_to(phi[None, :], weight.shape).copy(),
        weight,
    )


def _aperture_oracle(
    *,
    altitude_rad: np.ndarray,
    azimuth_rad: np.ndarray,
    diameter_m: float,
    wavelength_m: float,
    modes: tuple[tuple[int, int, float], ...] = (),
) -> np.ndarray:
    """Evaluate Section 3.3's normalized integral for an unmasked disk.

    Only the smooth (unmasked) case is used as an oracle: a hard mask needs the
    boundary-fitted panels the production rule owns, and comparing against a
    smooth product rule there would measure the oracle, not the implementation.
    """
    rho, phi, weight = _disk_quadrature()
    kappa = 4.0 * np.pi / wavelength_m
    height = np.zeros_like(rho)
    for n, m, coefficient in modes:
        height = height + coefficient * _noll_zernike(n, m, rho, phi)
    radius = 0.5 * diameter_m
    wavenumber = 2.0 * np.pi / wavelength_m
    values = np.empty(altitude_rad.shape, dtype=np.complex128)
    for index in range(altitude_rad.size):
        q_north = wavenumber * np.cos(altitude_rad[index]) * np.cos(azimuth_rad[index])
        q_east = wavenumber * np.cos(altitude_rad[index]) * np.sin(azimuth_rad[index])
        phase = kappa * height + rho * radius * (
            q_north * np.cos(phi) + q_east * np.sin(phi)
        )
        values[index] = np.sum(weight * np.exp(-1j * phase))
    return values


def _hankel_mixture(
    x: np.ndarray, *, edge_taper_db: float, squared: bool
) -> np.ndarray:
    """Section 3.1's normalized transform of ``p*U + (1-p)*P`` (or ``P2``)."""
    pedestal = 10.0 ** (-edge_taper_db / 20.0)
    uniform = np.ones_like(x)
    tapered = np.ones_like(x)
    nonzero = x != 0.0
    uniform[nonzero] = 2.0 * jv(1, x[nonzero]) / x[nonzero]
    if squared:
        tapered[nonzero] = 48.0 * jv(3, x[nonzero]) / x[nonzero] ** 3
    else:
        tapered[nonzero] = 8.0 * jv(2, x[nonzero]) / x[nonzero] ** 2
    return pedestal * uniform + (1.0 - pedestal) * tapered


def _blocked_uniform_voltage(x: np.ndarray, epsilon: float) -> np.ndarray:
    """Section 3.2's exact blocked uniform-aperture oracle."""
    values = np.full(x.shape, 1.0 - epsilon**2, dtype=np.float64)
    nonzero = x != 0.0
    values[nonzero] = (
        2.0 * (jv(1, x[nonzero]) - epsilon * jv(1, epsilon * x[nonzero])) / x[nonzero]
    )
    return values


def _beam_argument(altitude_rad: np.ndarray, wavelength_m: float) -> np.ndarray:
    """``x = pi * D * sin(theta) / lambda`` for the fixture diameter."""
    theta = 0.5 * np.pi - altitude_rad
    return np.pi * FIXTURE_DIAMETER_M * np.sin(theta) / wavelength_m


# --- runtime helpers ----------------------------------------------------------


def _analytic_beams(
    model: dict[str, Any] | None = None,
    *,
    aperture_physics: dict[str, Any] | None = None,
    surface_error: dict[str, Any] | None = None,
) -> dict[str, Any]:
    beams: dict[str, Any] = {
        "mode": "analytic",
        "model": {"kind": "circular_aperture", "taper": {"kind": "uniform"}}
        if model is None
        else model,
    }
    if aperture_physics is not None:
        beams["aperture_physics"] = aperture_physics
    if surface_error is not None:
        beams["surface_error"] = surface_error
    return beams


def _aperture_block(
    *,
    blockage: dict[str, Any] | None = None,
    zernike_modes: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    block: dict[str, Any] = {"normalization": APERTURE_NORMALIZATION}
    if blockage is not None:
        block["blockage"] = blockage
    if zernike_modes is not None:
        block["zernike_surface"] = {
            "convention": ZERNIKE_CONVENTION,
            "modes": zernike_modes,
        }
    return block


def _beam_system(
    tmp_path: Path,
    beams: dict[str, Any],
    *,
    beam_precision: str | None = None,
) -> tuple[Any, float]:
    """Load one canonical ``BeamSystem`` and return it with its first frequency.

    ``tmp_path`` is created if absent: ``valid_config_mapping`` writes the
    antenna layout straight into it without creating parents, so every caller
    that wants one system per subdirectory would otherwise have to remember.
    """
    tmp_path.mkdir(parents=True, exist_ok=True)

    from radiosim.core.beam.resolution import resolve_beam_assignments
    from radiosim.core.beam.runtime import load_beam_system
    from radiosim.core.instrument_resolution import resolve_instrument
    from radiosim.io.config_resolution import ConfigurationSource, resolve_config

    data = valid_config_mapping(tmp_path, beams=beams)
    if beam_precision is not None:
        data["execution"]["precision"] = {"jones": {"beam": beam_precision}}
    bundle = resolve_config(
        data,
        source=ConfigurationSource.for_mapping(
            base_dir=tmp_path,
            invocation_dir=tmp_path,
        ),
    )
    runtime = bundle.runtime
    instrument = resolve_instrument(runtime.instrument)
    state = resolve_beam_assignments(runtime.beams, instrument)
    frequencies = runtime.frequency.channel_frequencies_hz
    system = load_beam_system(
        state,
        observation_frequencies_hz=frequencies,
        precision=runtime.execution.precision,
    )
    return system, float(frequencies[0])


def _voltage(
    system: Any,
    frequency_hz: float,
    *,
    altitude_rad: np.ndarray = _ALTITUDE_RAD,
    azimuth_rad: np.ndarray = _AZIMUTH_RAD,
) -> np.ndarray:
    """Return the scalar ``E`` diagonal, after asserting ``E = e * I2``."""
    from radiosim.core.instrument import AntennaId

    jones = system.evaluate_jones(
        AntennaId(0, "ANT0"),
        altitude_rad=altitude_rad,
        azimuth_rad=azimuth_rad,
        frequency_hz=frequency_hz,
        time_mjd=60000.0,
    )
    assert jones.shape == (altitude_rad.size, 2, 2)
    np.testing.assert_array_equal(jones[:, 0, 1], 0.0)
    np.testing.assert_array_equal(jones[:, 1, 0], 0.0)
    np.testing.assert_array_equal(jones[:, 0, 0], jones[:, 1, 1])
    return np.asarray(jones[:, 0, 0])


# --- Section 3.3: the declared Zernike basis ----------------------------------


def test_stage1_scientific_conventions_are_the_frozen_literals() -> None:
    """Section 8.1 freezes the exact convention record the evidence retains."""
    assert dict(STAGE1_SCIENTIFIC_CONVENTIONS) == EXPECTED_CONVENTIONS


def test_zernike_upper_radial_order_bound_is_thirty_two() -> None:
    """Section 3.3: "a v1 computation bound, not a statement that higher
    physical modes do not exist"."""
    assert ZERNIKE_MAX_RADIAL_ORDER == 32


@pytest.mark.parametrize(
    ("n", "m"),
    [(2, 0), (2, 2), (2, -2), (3, 1), (3, -3), (4, 0), (4, 2), (6, -2), (8, 4)],
)
def test_real_unit_rms_zernike_matches_the_declared_closed_form(
    n: int,
    m: int,
) -> None:
    """Section 3.3's radial sum and its ``sqrt(n+1)``/``sqrt(2(n+1))`` scaling."""
    rho = np.linspace(0.0, 1.0, 37)
    phi = np.linspace(0.0, 2.0 * np.pi, 37, endpoint=False)
    grid_rho, grid_phi = np.meshgrid(rho, phi, indexing="ij")

    observed = real_unit_rms_zernike(n, m, grid_rho, grid_phi)

    _assert_within_frozen_tolerance(observed, _noll_zernike(n, m, grid_rho, grid_phi))


@pytest.mark.parametrize(
    ("left", "right"),
    [
        ((2, 0), (2, 0)),
        ((2, 2), (2, 2)),
        ((3, -1), (3, -1)),
        ((4, 0), (4, 0)),
        ((2, 0), (4, 0)),
        ((2, 2), (2, -2)),
        ((3, 1), (3, -1)),
        ((2, 0), (2, 2)),
    ],
)
def test_declared_zernike_basis_is_orthonormal_on_the_unobscured_disk(
    left: tuple[int, int],
    right: tuple[int, int],
) -> None:
    """Section 3.3: ``(1/pi) * int Z Z' rho drho dphi = delta``, so each
    coefficient is a unit-RMS surface height on the *unobscured* disk."""
    rho, phi, weight = _disk_quadrature()

    product = real_unit_rms_zernike(*left, rho, phi) * real_unit_rms_zernike(
        *right, rho, phi
    )
    observed = float(np.sum(weight * product))

    _assert_within_frozen_tolerance(observed, 1.0 if left == right else 0.0)


def test_zernike_surface_height_is_the_signed_metre_sum_of_authored_modes() -> None:
    """Section 3.3: "each coefficient is signed aperture-equivalent reflector
    surface-height error in metres"."""
    modes = ((2, 0, 5.0e-4), (3, 1, -2.0e-4), (4, -2, 1.5e-4))
    rho = np.linspace(0.0, 1.0, 23)
    phi = np.linspace(0.0, 2.0 * np.pi, 23, endpoint=False)
    grid_rho, grid_phi = np.meshgrid(rho, phi, indexing="ij")

    observed = zernike_surface_height(modes, grid_rho, grid_phi)

    expected = np.zeros_like(grid_rho)
    for n, m, coefficient in modes:
        expected = expected + coefficient * _noll_zernike(n, m, grid_rho, grid_phi)
    _assert_within_frozen_tolerance(observed, expected)
    # Section 3.3: sorting modes for a stable fingerprint does not change the sum.
    _assert_within_frozen_tolerance(
        zernike_surface_height(tuple(sorted(modes)), grid_rho, grid_phi), observed
    )


def test_surface_height_phase_uses_the_negative_four_pi_over_lambda_convention() -> (
    None
):
    """Section 3.1: the signed excess path is exactly ``2*h``, and RadioSim's
    positive-delay convention produces ``exp(-i * 4*pi*h/lambda)``."""
    wavelength_m = 2.0
    heights = np.array([0.0, 0.125, 0.25, -0.25, 0.5], dtype=np.float64)

    observed = surface_height_phase(heights, wavelength_m)

    _assert_within_frozen_tolerance(
        observed, np.exp(-1j * 4.0 * np.pi * heights / wavelength_m)
    )
    # An eighth-wave of surface height is a quarter-wave of path: with
    # ``wavelength_m = 2.0`` that is ``heights[2] = 0.25``, whose phase is
    # ``exp(-i*pi/2) = -1j``.  ``heights[1] = 0.125`` gives ``exp(-i*pi/4)``.
    _assert_within_frozen_tolerance(observed[2], -1j)
    _assert_within_frozen_tolerance(np.abs(observed), np.ones_like(heights))


def test_reversing_every_coefficient_conjugates_the_surface_phase() -> None:
    """Section 3.3's red list: "the exact defocus/conjugation relation under
    sign reversal"."""
    modes = ((2, 0, 5.0e-3), (4, 0, -1.0e-3))
    reversed_modes = tuple((n, m, -c) for n, m, c in modes)
    rho = np.linspace(0.0, 1.0, 17)
    phi = np.linspace(0.0, 2.0 * np.pi, 17, endpoint=False)
    grid_rho, grid_phi = np.meshgrid(rho, phi, indexing="ij")
    wavelength_m = 3.0

    forward = surface_height_phase(
        zernike_surface_height(modes, grid_rho, grid_phi), wavelength_m
    )
    reversed_phase = surface_height_phase(
        zernike_surface_height(reversed_modes, grid_rho, grid_phi), wavelength_m
    )

    _assert_within_frozen_tolerance(reversed_phase, np.conj(forward))


@pytest.mark.parametrize("incidence_deg", [0.0, 15.0, 30.0, 45.0, 60.0])
def test_authored_height_is_half_reflected_opd_at_any_incidence(
    incidence_deg: float,
) -> None:
    """Section 3.1: ``h = delta_n * cos(i)``, and ``h = delta_n`` only at normal
    incidence.  Stage 1 never invents ``i`` from the beam model."""
    wavelength_m = 3.0
    normal_displacement_m = 0.01
    incidence_rad = math.radians(incidence_deg)
    height_m = normal_displacement_m * math.cos(incidence_rad)

    phase = surface_height_phase(np.array([height_m]), wavelength_m)

    # The signed excess path is exactly 2*h, so the phase is that path times
    # -2*pi/lambda.
    excess_path_m = 2.0 * height_m
    _assert_within_frozen_tolerance(
        phase[0], np.exp(-1j * 2.0 * np.pi * excess_path_m / wavelength_m)
    )
    if incidence_deg == 0.0:
        assert height_m == normal_displacement_m
    else:
        assert height_m < normal_displacement_m


# --- Section 3.2: blockage geometry -------------------------------------------


def _independent_mask(
    north_m: np.ndarray,
    east_m: np.ndarray,
    *,
    diameter_m: float,
    central_diameter_ratio: float,
    support_legs: tuple[tuple[float, float], ...],
) -> np.ndarray:
    """Section 3.2's set definitions, coded independently in the test body."""
    radius = 0.5 * diameter_m
    distance = np.hypot(north_m, east_m)
    inside = distance <= radius
    central = distance <= central_diameter_ratio * radius
    blocked = central.copy()
    for angle_deg, width_m in support_legs:
        beta = math.radians(angle_deg)
        along = north_m * math.cos(beta) + east_m * math.sin(beta)
        across = -north_m * math.sin(beta) + east_m * math.cos(beta)
        leg = (
            inside
            & (distance >= central_diameter_ratio * radius)
            & (along >= 0.0)
            & (np.abs(across) <= 0.5 * width_m)
        )
        blocked = blocked | leg
    return inside & ~blocked


@pytest.mark.parametrize("angle_deg", [0.0, 90.0, 180.0, -90.0])
def test_a_support_leg_is_one_outward_half_strip_not_a_chord(
    angle_deg: float,
) -> None:
    """Section 3.2: "a leg is one outward half-strip, not an infinite chord and
    not a pair of diametrically opposed legs"."""
    diameter_m = 14.0
    ratio = 0.2
    width_m = 0.6
    beta = math.radians(angle_deg)
    outward = np.array([math.cos(beta), math.sin(beta)]) * 4.0
    inward = -outward

    probes_north = np.array([outward[0], inward[0]])
    probes_east = np.array([outward[1], inward[1]])
    observed = aperture_transmission_mask(
        probes_north,
        probes_east,
        diameter_m=diameter_m,
        central_diameter_ratio=ratio,
        support_legs=((angle_deg, width_m),),
    )

    assert observed.tolist() == [False, True]
    np.testing.assert_array_equal(
        observed,
        _independent_mask(
            probes_north,
            probes_east,
            diameter_m=diameter_m,
            central_diameter_ratio=ratio,
            support_legs=((angle_deg, width_m),),
        ),
    )


def test_a_structure_on_both_sides_is_authored_as_two_records_180_apart() -> None:
    """Section 3.2: "A physical structure on both sides is authored as two
    records separated by 180 degrees"."""
    north = np.array([4.0, -4.0])
    east = np.array([0.0, 0.0])
    common = {
        "diameter_m": 14.0,
        "central_diameter_ratio": 0.2,
    }

    single = aperture_transmission_mask(
        north, east, support_legs=((0.0, 0.6),), **common
    )
    both = aperture_transmission_mask(
        north, east, support_legs=((0.0, 0.6), (180.0, 0.6)), **common
    )

    assert single.tolist() == [False, True]
    assert both.tolist() == [False, False]


def test_the_closed_boundary_rules_are_exactly_as_written() -> None:
    """Section 3.2's closed sets: ``r <= epsilon R`` and ``|u.p| <= w/2`` are
    blocked, and ``r > R`` is outside the ideal pupil."""
    diameter_m = 14.0
    radius = 0.5 * diameter_m
    ratio = 0.2
    width_m = 0.6
    north = np.array(
        [
            ratio * radius,  # exactly on the central-shadow boundary: blocked
            radius,  # exactly on the pupil edge: transmitting
            radius * (1.0 + 1e-9),  # outside the ideal pupil
            4.0,  # inside the leg, exactly on its edge
            4.0,
        ]
    )
    east = np.array([0.0, 0.0, 0.0, 0.5 * width_m, 0.5 * width_m * (1.0 + 1e-9)])
    legs = ((0.0, width_m),)

    observed = aperture_transmission_mask(
        north,
        east,
        diameter_m=diameter_m,
        central_diameter_ratio=ratio,
        support_legs=legs,
    )

    assert observed.tolist() == [False, False, False, False, True]


def test_overlapping_support_shadows_are_removed_once_by_set_union() -> None:
    """Section 3.2: "Masks combine by set union, so overlapping shadows are
    removed once", and the union is taken before the mask is evaluated."""
    diameter_m = 14.0
    ratio = 0.15
    legs = ((0.0, 1.2), (6.0, 1.2), (90.0, 0.8))
    axis = np.linspace(-8.0, 8.0, 129)
    north, east = np.meshgrid(axis, axis, indexing="ij")

    observed = aperture_transmission_mask(
        north,
        east,
        diameter_m=diameter_m,
        central_diameter_ratio=ratio,
        support_legs=legs,
    )

    np.testing.assert_array_equal(
        observed,
        _independent_mask(
            north,
            east,
            diameter_m=diameter_m,
            central_diameter_ratio=ratio,
            support_legs=legs,
        ),
    )
    # A union never removes more than the sum of its parts.
    singles = [
        aperture_transmission_mask(
            north,
            east,
            diameter_m=diameter_m,
            central_diameter_ratio=ratio,
            support_legs=(leg,),
        )
        for leg in legs
    ]
    blocked_together = int(np.count_nonzero(~observed))
    blocked_separately = sum(int(np.count_nonzero(~single)) for single in singles)
    assert blocked_together < blocked_separately


@pytest.mark.parametrize("width_ratio", [0.02, 0.1, 0.25])
def test_support_leg_half_angle_saturates_below_the_width_ratio(
    width_ratio: float,
) -> None:
    """Section 3.3's ``alpha_j(rho)``: exactly ``pi/2`` at and below ``a_j``,
    ``atan2(a, sqrt(rho^2 - a^2))`` above it, and continuous across it."""
    below = np.linspace(0.0, width_ratio, 11)
    above = np.linspace(width_ratio, 1.0, 41)

    saturated = support_leg_half_angle(below, width_ratio)
    resolved = support_leg_half_angle(above, width_ratio)

    _assert_within_frozen_tolerance(saturated, np.full_like(below, 0.5 * np.pi))
    expected = np.arctan2(
        width_ratio, np.sqrt(np.maximum(above**2 - width_ratio**2, 0.0))
    )
    _assert_within_frozen_tolerance(resolved, expected)
    assert np.all(np.diff(resolved) <= ATOL)


@pytest.mark.parametrize(
    ("bandwidth", "panel_length", "degree"),
    [
        (0.0, 0.0, 0),
        (1.0, 0.5, 0),
        (1.0, 0.5, 12),
        (1.0, 0.5, 40),
        (40.0, 1.0, 2),
        (400.0, 1.0, 2),
        (4000.0, 0.25, 2),
        (12.3, 0.37, 5),
    ],
)
def test_gauss_legendre_order_seed_matches_the_frozen_expression(
    bandwidth: float,
    panel_length: float,
    degree: int,
) -> None:
    """Section 3.3: ``max(16, 2(d+1), 8 + ceil(8*B*L/pi))``, so an authored
    frequency, diameter, or coefficient cannot create an unseeded oscillation."""
    expected = max(
        16,
        2 * (degree + 1),
        8 + math.ceil(8.0 * bandwidth * panel_length / math.pi),
    )

    observed = gauss_legendre_order_seed(bandwidth, panel_length, degree)

    assert type(observed) is int
    assert observed == expected


def test_the_quadrature_seed_grows_with_far_field_and_surface_phase_bandwidth() -> None:
    """Section 3.3: ``B = Q_max + kappa*H``, so *both* terms move the seed.

    The floor case is chosen so that its bandwidth term genuinely does not
    dominate: ``8 + ceil(8*B*L_p/pi) <= 16`` needs ``B*L_p <= pi``, which
    ``B = 3.0`` at ``L_p = 1.0`` satisfies and ``B = 4.0`` does not.
    """
    floored = gauss_legendre_order_seed(3.0, 1.0, 2)
    low = gauss_legendre_order_seed(4.0, 1.0, 2)
    far_field = gauss_legendre_order_seed(400.0, 1.0, 2)
    surface = gauss_legendre_order_seed(4.0 + 400.0, 1.0, 2)

    assert floored == 16
    assert low == 19
    assert far_field == 1027
    # Adding surface-phase bandwidth to the same far field must grow the seed:
    # a rule that ignored ``kappa*H`` would leave these two equal.
    assert surface == 1037
    assert surface > far_field > low > 8


# --- Section 3.1: normalization and composition through the runtime -----------


@pytest.mark.parametrize(
    ("taper", "squared"),
    [
        ({"kind": "parabolic", "edge_taper_db": 12.0}, False),
        ({"kind": "parabolic_squared", "edge_taper_db": 12.0}, True),
    ],
)
def test_supported_profiles_reproduce_their_normalized_hankel_transform(
    tmp_path: Path,
    taper: dict[str, Any],
    squared: bool,
) -> None:
    """Section 3.1: ``p`` keeps its existing mixture-weight meaning, so the
    table's profile is the current unmodified scalar response."""
    system, frequency_hz = _beam_system(
        tmp_path,
        _analytic_beams({"kind": "circular_aperture", "taper": taper}),
    )
    wavelength_m = _SPEED_OF_LIGHT_M_PER_S / frequency_hz

    observed = _voltage(system, frequency_hz)

    expected = _hankel_mixture(
        _beam_argument(_ALTITUDE_RAD, wavelength_m),
        edge_taper_db=float(taper["edge_taper_db"]),
        squared=squared,
    )
    _assert_within_frozen_tolerance(observed.real, expected)
    _assert_within_frozen_tolerance(observed.imag, np.zeros_like(expected))


def test_uniform_profile_reproduces_two_j1_over_x(tmp_path: Path) -> None:
    """The ``U(rho)`` row of Section 3.1's table."""
    system, frequency_hz = _beam_system(tmp_path, _analytic_beams())
    wavelength_m = _SPEED_OF_LIGHT_M_PER_S / frequency_hz

    observed = _voltage(system, frequency_hz)

    expected = _hankel_mixture(
        _beam_argument(_ALTITUDE_RAD, wavelength_m),
        edge_taper_db=0.0,
        squared=False,
    )
    _assert_within_frozen_tolerance(observed.real, expected)


@pytest.mark.parametrize("epsilon", [0.1, 0.15, 0.3])
def test_blocked_uniform_aperture_matches_the_closed_form(
    tmp_path: Path,
    epsilon: float,
) -> None:
    """Section 3.2's exact oracle, including the boresight loss ``1 - eps^2``
    that a re-peak-normalized beam would have thrown away."""
    system, frequency_hz = _beam_system(
        tmp_path,
        _analytic_beams(
            aperture_physics=_aperture_block(
                blockage={
                    "central_diameter_ratio": epsilon,
                    "support_legs": [],
                }
            )
        ),
    )
    wavelength_m = _SPEED_OF_LIGHT_M_PER_S / frequency_hz

    observed = _voltage(system, frequency_hz)

    argument = _beam_argument(_ALTITUDE_RAD, wavelength_m)
    _assert_within_frozen_tolerance(
        observed.real, _blocked_uniform_voltage(argument, epsilon)
    )
    # Section 3.2: e(0) = 1 - eps^2 and the blockage efficiency is its square.
    assert argument[0] == 0.0
    _assert_within_frozen_tolerance(observed[0].real, 1.0 - epsilon**2)
    _assert_within_frozen_tolerance(abs(observed[0]) ** 2, (1.0 - epsilon**2) ** 2)


def test_tapered_blockage_boresight_is_illumination_weighted(
    tmp_path: Path,
) -> None:
    """Section 3.2: "It is not generally ``(1-epsilon**2)`` for tapered
    illumination"."""
    epsilon = 0.25
    edge_taper_db = 12.0
    system, frequency_hz = _beam_system(
        tmp_path,
        _analytic_beams(
            {
                "kind": "circular_aperture",
                "taper": {"kind": "parabolic", "edge_taper_db": edge_taper_db},
            },
            aperture_physics=_aperture_block(
                blockage={
                    "central_diameter_ratio": epsilon,
                    "support_legs": [],
                }
            ),
        ),
    )

    observed = float(
        _voltage(
            system,
            frequency_hz,
            altitude_rad=np.array([np.pi / 2.0]),
            azimuth_rad=np.array([0.0]),
        )[0].real
    )

    pedestal = 10.0 ** (-edge_taper_db / 20.0)
    nodes, weights = np.polynomial.legendre.leggauss(200)

    def _weighted(lower: float) -> float:
        rho = 0.5 * (nodes + 1.0) * (1.0 - lower) + lower
        scale = 0.5 * (1.0 - lower)
        profile = pedestal + (1.0 - pedestal) * 2.0 * (1.0 - rho**2)
        return float(np.sum(scale * weights * profile * rho))

    expected = _weighted(epsilon) / _weighted(0.0)
    _assert_within_frozen_tolerance(observed, expected)
    assert abs(observed - (1.0 - epsilon**2)) > 1e-3


def test_zernike_surface_matches_an_independent_aperture_integral(
    tmp_path: Path,
) -> None:
    """Section 3.1's one integral, evaluated independently on the unmasked disk.

    The surface phase makes ``e`` complex, so this also proves the sign of the
    negative-forward transform and of ``exp(-i*4*pi*h/lambda)`` together.
    """
    modes = ((2, 0, 0.05), (3, 1, -0.02))
    system, frequency_hz = _beam_system(
        tmp_path,
        _analytic_beams(
            aperture_physics=_aperture_block(
                zernike_modes=[
                    {"n": n, "m": m, "surface_height_coefficient_m": c}
                    for n, m, c in modes
                ]
            )
        ),
    )
    wavelength_m = _SPEED_OF_LIGHT_M_PER_S / frequency_hz

    observed = _voltage(system, frequency_hz)

    expected = _aperture_oracle(
        altitude_rad=_ALTITUDE_RAD,
        azimuth_rad=_AZIMUTH_RAD,
        diameter_m=FIXTURE_DIAMETER_M,
        wavelength_m=wavelength_m,
        modes=modes,
    )
    _assert_within_frozen_tolerance(observed, expected)
    assert float(np.max(np.abs(observed.imag))) > 1e-3


def test_composed_mask_and_phase_differ_from_a_product_of_far_field_factors(
    tmp_path: Path,
) -> None:
    """Section 3.1: separately evaluated far-field patterns "must never be
    multiplied as if Fourier transformation distributed over aperture
    multiplication"."""
    blockage = {"central_diameter_ratio": 0.2, "support_legs": []}
    modes = [{"n": 2, "m": 0, "surface_height_coefficient_m": 0.05}]

    blocked_system, frequency_hz = _beam_system(
        tmp_path / "blocked",
        _analytic_beams(aperture_physics=_aperture_block(blockage=blockage)),
    )
    surface_system, _ = _beam_system(
        tmp_path / "surface",
        _analytic_beams(aperture_physics=_aperture_block(zernike_modes=modes)),
    )
    composed_system, _ = _beam_system(
        tmp_path / "composed",
        _analytic_beams(
            aperture_physics=_aperture_block(blockage=blockage, zernike_modes=modes)
        ),
    )

    blocked = _voltage(blocked_system, frequency_hz)
    surface = _voltage(surface_system, frequency_hz)
    composed = _voltage(composed_system, frequency_hz)

    wrong_product = blocked * surface
    assert float(np.max(np.abs(composed - wrong_product))) > 1e-3
    # The composition is not the trivial one either.
    assert float(np.max(np.abs(composed - blocked))) > 1e-3
    assert float(np.max(np.abs(composed - surface))) > 1e-3


def test_a_single_support_leg_breaks_azimuthal_symmetry_north_through_east(
    tmp_path: Path,
) -> None:
    """Section 3.1: aperture azimuth is measured North through East, matching
    RadioSim's topocentric azimuth, so rotating the leg rotates the pattern."""
    altitude = np.full(4, 0.9)
    azimuth = np.array([0.0, 0.5 * np.pi, np.pi, 1.5 * np.pi])
    blockage_north = {
        "central_diameter_ratio": 0.15,
        "support_legs": [{"position_angle_deg": 0.0, "width_m": 1.2}],
    }
    blockage_east = {
        "central_diameter_ratio": 0.15,
        "support_legs": [{"position_angle_deg": 90.0, "width_m": 1.2}],
    }

    north_system, frequency_hz = _beam_system(
        tmp_path / "north",
        _analytic_beams(aperture_physics=_aperture_block(blockage=blockage_north)),
    )
    east_system, _ = _beam_system(
        tmp_path / "east",
        _analytic_beams(aperture_physics=_aperture_block(blockage=blockage_east)),
    )

    north = _voltage(
        north_system, frequency_hz, altitude_rad=altitude, azimuth_rad=azimuth
    )
    east = _voltage(
        east_system, frequency_hz, altitude_rad=altitude, azimuth_rad=azimuth
    )

    # A single leg is not axisymmetric.
    assert abs(float(north[0].real) - float(north[1].real)) > 1e-6
    # Rotating the leg by 90 degrees rotates the response by the same 90
    # degrees, North through East.
    _assert_within_frozen_tolerance(east[[1, 2, 3, 0]], north[[0, 1, 2, 3]])


@pytest.mark.skipif(
    not COMPLEX256_AVAILABLE,
    reason=(
        "Section 8.1's extended-width predicate: this NumPy runtime exposes no "
        "distinct 32-byte clongdouble"
    ),
)
def test_extended_precision_composition_never_narrows_to_complex128(
    tmp_path: Path,
) -> None:
    """Section 3.1: "Its target-width quadrature nodes, weights, and
    accumulation may not pass through float64 when the resolved dtype is
    wider"."""
    system, frequency_hz = _beam_system(
        tmp_path,
        _analytic_beams(
            aperture_physics=_aperture_block(
                blockage={"central_diameter_ratio": 0.2, "support_legs": []},
                zernike_modes=[{"n": 2, "m": 0, "surface_height_coefficient_m": 0.05}],
            )
        ),
        beam_precision="float128",
    )

    observed = _voltage(system, frequency_hz)

    assert observed.dtype == np.dtype(np.complex256)
    # A result that had passed through complex128 would be exactly
    # representable there; the composed transform must not be.
    narrowed = observed.astype(np.complex128).astype(np.complex256)
    assert float(np.max(np.abs(observed - narrowed))) > 0.0
