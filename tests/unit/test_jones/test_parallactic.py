"""Tier 7F: the ``P`` term's mathematics, mounts, flags, and observable effect.

``Tier7JonesSciencePlan.md`` Section 20.7.  The parallactic angle is

.. code-block:: text

    psi(H, dec, lat) = atan2( sin H cos lat,
                              sin lat cos dec - cos lat sin dec cos H )

and the Jones factor is the real rotation ``P_p = R(eta_p psi + nasmyth_p el)``
with ``R(a) = [[cos a, sin a], [-sin a, cos a]]`` -- the same ``R`` the accepted
receptor mathematics uses, so that ``C_p P_p`` composes into
``M(basis) R(chi + psi)`` exactly (Section 12.1).

Invariants asserted here: **I2**, **I3**, **I6**, **I7**, **I9**, the Section
29.1 astropy cross-check, and Section 20.7's own statements -- ``P P^T = I2``,
antisymmetry in hour angle, ``psi = 0`` for ``dec = lat`` at ``H = 0``, and the
per-mount factors.

Three independent oracles are used for ``psi``, deliberately:

1. ``SkyCoord.position_angle`` on a bare sphere.  astropy's implementation of
   the position angle of the zenith at the source -- the definition of the
   parallactic angle -- rather than RadioSim's.
2. Vector algebra written out in the test body: project the pole and the zenith
   into the source's tangent plane and take the signed angle between them.  This
   shares no line of algebra with the two-argument arctangent above, so it is
   the check that would survive a sign error in the closed form.
3. astropy's **full** frame machinery (``AltAz`` -> ``CIRS`` at a real
   ``EarthLocation`` and ``Time``).  This one agrees to ~1e-5 rad rather than to
   1e-10, and the test says why: the residual is the difference between
   RadioSim's idealized spherical site model and astropy's rigorous transform
   (polar motion and diurnal aberration), which lives in Tier 7B's
   ``DirectionBatch`` and not in ``P``.  Feeding both the *same* ``(H, dec,
   lat)`` closes the gap to 1e-15, and that is asserted too, so the residual is
   attributed rather than tolerated.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest
from astropy import units as u
from astropy.coordinates import CIRS, ICRS, AltAz, SkyCoord
from astropy.time import Time

from radiosim.backends import get_backend
from radiosim.core.jones.directions import DirectionBatch, equatorial_from_horizontal
from radiosim.core.jones.parallactic import (
    MOUNT_FACTORS,
    ROTATING_MOUNT_TYPES,
    SUPPORTED_MOUNT_TYPES,
    ParallacticAngleJones,
    parallactic_angle,
)
from radiosim.core.jones.receptor import basis_rotation_matrix
from radiosim.core.polarization_basis import (
    SKY_NORTH_EAST_TO_CIRCULAR_RL,
    SKY_NORTH_EAST_TO_LINEAR_XY,
)
from radiosim.core.visibility import calculate_visibility
from tests.characterization.test_tier6_current_behavior import (
    WORKLOAD_LOCATION,
    WORKLOAD_TIME_GRID,
    _workload_point_sources,
)
from tests.unit.test_core.test_jones_resolution import (
    resolve_for,
    solver_components_with_jones,
)

_BACKEND = get_backend("numpy")

#: The HERA site the shipped fixture uses, so ``psi`` is exercised at a real
#: latitude rather than at a convenient one.
_SITE_LATITUDE_RAD = math.radians(-30.72152)


# ---------------------------------------------------------------------------
# Oracles, written out here rather than imported
# ---------------------------------------------------------------------------


def plan_psi(hour_angle_rad, dec_rad, latitude_rad: float) -> np.ndarray:
    """Section 20.7's closed form, transcribed from the plan."""
    hour_angle = np.asarray(hour_angle_rad, dtype=np.float64)
    dec = np.asarray(dec_rad, dtype=np.float64)
    return np.arctan2(
        np.sin(hour_angle) * math.cos(latitude_rad),
        math.sin(latitude_rad) * np.cos(dec)
        - math.cos(latitude_rad) * np.sin(dec) * np.cos(hour_angle),
    )


def astropy_sphere_psi(hour_angle_rad, dec_rad, latitude_rad: float) -> np.ndarray:
    """Oracle 1: astropy's own position angle of the zenith at the source.

    The parallactic angle *is* the position angle of the zenith seen from the
    source, measured North through East.  Placing the source at
    ``ra = -H, dec`` and the zenith at ``ra = 0, dec = lat`` on a bare sphere
    reproduces the (pole, zenith, source) triangle exactly, and lets astropy
    rather than RadioSim compute the angle.
    """
    hour_angle = np.atleast_1d(np.asarray(hour_angle_rad, dtype=np.float64))
    dec = np.atleast_1d(np.asarray(dec_rad, dtype=np.float64))
    source = SkyCoord(ra=-hour_angle * u.rad, dec=dec * u.rad, frame=ICRS())
    zenith = SkyCoord(
        ra=np.zeros_like(hour_angle) * u.rad,
        dec=np.full_like(hour_angle, latitude_rad) * u.rad,
        frame=ICRS(),
    )
    angle = source.position_angle(zenith).to_value(u.rad)
    return np.mod(angle + np.pi, 2.0 * np.pi) - np.pi


def vector_psi(alt_rad, az_rad, latitude_rad: float) -> np.ndarray:
    """Oracle 2: signed tangent-plane angle from the pole to the zenith.

    No arctangent identity is reused: the pole and the zenith are projected into
    the plane tangent to the sphere at the source, and the signed angle between
    the two projections is read off a dot product and a scalar triple product.
    """
    alt = np.atleast_1d(np.asarray(alt_rad, dtype=np.float64))
    az = np.atleast_1d(np.asarray(az_rad, dtype=np.float64))
    # Horizontal frame: x North, y East, z Up (azimuth North through East).
    source = np.stack(
        [np.cos(alt) * np.cos(az), np.cos(alt) * np.sin(az), np.sin(alt)], axis=-1
    )
    zenith = np.broadcast_to(np.array([0.0, 0.0, 1.0]), source.shape)
    pole = np.broadcast_to(
        np.array([math.cos(latitude_rad), 0.0, math.sin(latitude_rad)]), source.shape
    )

    def tangent(target: np.ndarray) -> np.ndarray:
        projected = target - source * np.sum(source * target, axis=-1, keepdims=True)
        return projected / np.linalg.norm(projected, axis=-1, keepdims=True)

    to_pole = tangent(pole)
    to_zenith = tangent(zenith)
    return np.arctan2(
        np.sum(np.cross(to_pole, to_zenith) * source, axis=-1),
        np.sum(to_pole * to_zenith, axis=-1),
    )


def plan_rotation(angle_rad: float) -> np.ndarray:
    """Section 20.7's ``R(a)``, transcribed from the plan."""
    return np.array(
        [
            [math.cos(angle_rad), math.sin(angle_rad)],
            [-math.sin(angle_rad), math.cos(angle_rad)],
        ],
        dtype=np.complex128,
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _directions(
    *,
    alt_rad: np.ndarray,
    az_rad: np.ndarray,
    latitude_rad: float = _SITE_LATITUDE_RAD,
    local_sidereal_time_rad: float = 0.0,
) -> DirectionBatch:
    return DirectionBatch.from_horizontal(
        alt_rad=alt_rad,
        az_rad=az_rad,
        dir_l=np.cos(alt_rad) * np.sin(az_rad),
        dir_m=np.cos(alt_rad) * np.cos(az_rad),
        dir_n=np.sin(alt_rad),
        latitude_rad=latitude_rad,
        local_sidereal_time_rad=local_sidereal_time_rad,
    )


def _sample_directions(n_dir: int = 24, *, spread_deg: float = 30.0) -> DirectionBatch:
    rng = np.random.default_rng(20260801)
    alt = np.radians(rng.uniform(90.0 - spread_deg, 85.0, n_dir))
    az = np.radians(rng.uniform(0.0, 360.0, n_dir))
    return _directions(alt_rad=alt, az_rad=az)


def _term(
    mount_types: tuple[str | None, ...] = ("alt-az", "alt-az"),
    *,
    latitude_rad: float = _SITE_LATITUDE_RAD,
) -> ParallacticAngleJones:
    return ParallacticAngleJones(latitude_rad=latitude_rad, mount_types=mount_types)


def _evaluate(
    term: ParallacticAngleJones,
    directions: DirectionBatch,
    *,
    antenna_idx: int = 0,
) -> np.ndarray:
    return np.asarray(
        term.compute_jones_batch(
            antenna_idx=antenna_idx,
            directions=directions,
            frequency_hz=1.0e8,
            freq_idx=0,
            time_mjd=60000.0,
            time_idx=0,
            backend=_BACKEND,
            dtype=np.complex128,
        )
    )


# ---------------------------------------------------------------------------
# psi against three independent oracles (Section 29.1)
# ---------------------------------------------------------------------------


def test_psi_matches_astropys_position_angle_over_a_dense_grid() -> None:
    """Oracle 1, over a dense ``(H, dec, lat)`` grid, to better than 1e-10 rad.

    Section 20.7 states the tolerance; the grid is dense in all three arguments
    and spans both hemispheres, both sides of the meridian, and the polar caps,
    because the whole reason for the two-argument arctangent is that the
    quadrant must survive over the whole sky.
    """
    hour_angles = np.radians(np.arange(-175.0, 180.0, 5.0))
    decs = np.radians(np.arange(-85.0, 86.0, 5.0))
    latitudes = np.radians(np.array([-80.0, -30.72152, -5.0, 0.0, 12.0, 52.0, 78.0]))

    grid_h, grid_d = np.meshgrid(hour_angles, decs, indexing="ij")
    flat_h = grid_h.reshape(-1)
    flat_d = grid_d.reshape(-1)
    assert flat_h.size > 2000

    for latitude in latitudes:
        computed = parallactic_angle(
            hour_angle_rad=flat_h, dec_rad=flat_d, latitude_rad=float(latitude)
        )
        reference = astropy_sphere_psi(flat_h, flat_d, float(latitude))
        difference = np.mod(computed - reference + np.pi, 2.0 * np.pi) - np.pi
        assert float(np.max(np.abs(difference))) < 1e-10


def test_psi_matches_an_independent_vector_construction() -> None:
    """Oracle 2: different algebra entirely, agreeing to machine precision."""
    directions = _sample_directions(64, spread_deg=70.0)

    computed = parallactic_angle(
        hour_angle_rad=directions.hour_angle_rad,
        dec_rad=directions.dec_rad,
        latitude_rad=_SITE_LATITUDE_RAD,
    )
    reference = vector_psi(directions.alt_rad, directions.az_rad, _SITE_LATITUDE_RAD)

    np.testing.assert_allclose(computed, reference, rtol=0.0, atol=1e-12)


def test_psi_matches_astropys_full_frame_machinery_within_the_site_model() -> None:
    """Oracle 3, and the attribution of its residual.

    Given the same ``(H, dec, lat)``, RadioSim's closed form and astropy's own
    position angle agree to ~1e-15.  Given the same *sky directions*, they
    differ by ~1e-5 rad, because ``DirectionBatch`` inverts the horizontal
    transform with an idealized spherical rotation while astropy carries polar
    motion and diurnal aberration.  That residual is a Tier 7B property of the
    direction batch, not a property of ``P``, and this test says which is which
    rather than widening one tolerance to cover both.
    """
    obstime = Time(60676.0, format="mjd")
    frame = AltAz(obstime=obstime, location=WORKLOAD_LOCATION)
    latitude = WORKLOAD_LOCATION.lat.rad

    rng = np.random.default_rng(7)
    alt = np.radians(rng.uniform(20.0, 80.0, 96))
    az = np.radians(rng.uniform(0.0, 360.0, 96))

    cirs = CIRS(obstime=obstime, location=WORKLOAD_LOCATION)
    source = SkyCoord(alt=alt * u.rad, az=az * u.rad, frame=frame).transform_to(cirs)
    zenith = SkyCoord(
        alt=np.full_like(alt, 0.5 * np.pi) * u.rad, az=az * u.rad, frame=frame
    ).transform_to(cirs)
    astropy_angle = source.position_angle(zenith).to_value(u.rad)
    astropy_angle = np.mod(astropy_angle + np.pi, 2.0 * np.pi) - np.pi

    # Same (H, dec, lat) as astropy resolved them: agreement is exact.
    astropy_hour_angle = (
        np.mod(
            zenith.ra.to_value(u.rad) - source.ra.to_value(u.rad) + np.pi, 2.0 * np.pi
        )
        - np.pi
    )
    exact = parallactic_angle(
        hour_angle_rad=astropy_hour_angle,
        dec_rad=source.dec.to_value(u.rad),
        latitude_rad=float(np.mean(zenith.dec.to_value(u.rad))),
    )
    np.testing.assert_allclose(exact, astropy_angle, rtol=0.0, atol=1e-12)

    # Same directions, RadioSim's own spherical inverse: agreement is bounded
    # by the site model, and the bound is small enough that no field rotation
    # a simulation reports is in question.
    hour_angle, dec = equatorial_from_horizontal(
        alt_rad=alt, az_rad=az, latitude_rad=latitude
    )
    modelled = parallactic_angle(
        hour_angle_rad=hour_angle, dec_rad=dec, latitude_rad=latitude
    )
    residual = np.max(np.abs(modelled - astropy_angle))
    assert residual < 1e-4
    assert residual > 1e-8  # it is a real difference, not a rounding artefact


# ---------------------------------------------------------------------------
# Section 20.7's own statements about psi
# ---------------------------------------------------------------------------


def test_psi_vanishes_for_a_source_at_the_zenith_meridian() -> None:
    """``psi = 0`` identically for ``dec = lat`` at ``H = 0`` (Section 20.7)."""
    for latitude in (-1.2, -0.5361, 0.0, 0.9):
        angle = parallactic_angle(
            hour_angle_rad=np.array([0.0]),
            dec_rad=np.array([latitude]),
            latitude_rad=latitude,
        )
        assert angle[0] == 0.0


def test_psi_is_antisymmetric_about_transit() -> None:
    """``psi(-H) = -psi(H)`` for every declination (Section 20.7)."""
    hour_angles = np.radians(np.arange(1.0, 180.0, 3.0))
    for dec_deg in (-70.0, -30.0, 0.0, 25.0, 60.0):
        dec = np.full_like(hour_angles, math.radians(dec_deg))
        forward = parallactic_angle(
            hour_angle_rad=hour_angles, dec_rad=dec, latitude_rad=_SITE_LATITUDE_RAD
        )
        backward = parallactic_angle(
            hour_angle_rad=-hour_angles, dec_rad=dec, latitude_rad=_SITE_LATITUDE_RAD
        )
        np.testing.assert_allclose(backward, -forward, rtol=0.0, atol=1e-15)


def test_psi_uses_a_two_argument_arctangent_over_the_whole_sky() -> None:
    """The quadrant survives: ``psi`` reaches every quadrant of ``[-pi, pi)``.

    An implementation built on ``arcsin`` or a one-argument ``arctan`` would
    fold the second and third quadrants onto the first and fourth, and would
    pass every narrow-field test while being wrong for a source below the pole.
    """
    hour_angles = np.radians(np.arange(-179.0, 180.0, 1.0))
    dec = np.full_like(hour_angles, math.radians(-80.0))
    angles = parallactic_angle(
        hour_angle_rad=hour_angles, dec_rad=dec, latitude_rad=_SITE_LATITUDE_RAD
    )

    quadrants = {int(np.sign(np.sin(a))) * (1 if np.cos(a) >= 0 else 2) for a in angles}
    assert quadrants >= {1, 2, -1, -2}


# ---------------------------------------------------------------------------
# The mount table (Section 20.7)
# ---------------------------------------------------------------------------


def test_the_mount_table_is_exactly_the_five_designed_rows() -> None:
    """No sixth mount is modelled, and the rotating subset is named once."""
    assert set(SUPPORTED_MOUNT_TYPES) == {
        "alt-az",
        "equatorial",
        "fixed",
        "alt-az+nasmyth-l",
        "alt-az+nasmyth-r",
    }
    assert MOUNT_FACTORS["alt-az"] == (1.0, 0.0)
    assert MOUNT_FACTORS["equatorial"] == (0.0, 0.0)
    assert MOUNT_FACTORS["fixed"] == (0.0, 0.0)
    assert MOUNT_FACTORS["alt-az+nasmyth-r"] == (1.0, 1.0)
    assert MOUNT_FACTORS["alt-az+nasmyth-l"] == (1.0, -1.0)
    assert ROTATING_MOUNT_TYPES == {
        "alt-az",
        "alt-az+nasmyth-l",
        "alt-az+nasmyth-r",
    }


@pytest.mark.parametrize("mount_type", ["fixed", "equatorial", None])
def test_a_non_rotating_mount_contributes_exactly_the_identity(
    mount_type: str | None,
) -> None:
    """``eta = 0``: the feeds do not rotate relative to the sky.

    ``None`` -- which is what every layout-file source resolves to, because no
    layout format carries a mount column -- is the ``fixed`` case.  That is the
    choice invariant I1 rests on: an instrument with no mount metadata behaves
    exactly as it did before this term existed.
    """
    directions = _sample_directions()
    block = _evaluate(_term((mount_type, mount_type)), directions)

    np.testing.assert_array_equal(
        block, np.broadcast_to(np.eye(2, dtype=np.complex128), block.shape)
    )


def test_an_alt_az_mount_applies_exactly_the_parallactic_rotation() -> None:
    """``P = R(psi)``, element by element, against the transcribed forms."""
    directions = _sample_directions()
    angles = plan_psi(directions.hour_angle_rad, directions.dec_rad, _SITE_LATITUDE_RAD)

    block = _evaluate(_term(("alt-az", "alt-az")), directions)

    expected = np.stack([plan_rotation(float(angle)) for angle in angles])
    np.testing.assert_allclose(block, expected, rtol=0.0, atol=1e-15)


@pytest.mark.parametrize(
    ("mount_type", "sign"), [("alt-az+nasmyth-r", 1.0), ("alt-az+nasmyth-l", -1.0)]
)
def test_a_nasmyth_mount_adds_the_signed_elevation(
    mount_type: str, sign: float
) -> None:
    """``psi + el`` for Nasmyth right and ``psi - el`` for Nasmyth left.

    The two differ by ``2 el``, which is what makes a mixed-Nasmyth array a
    genuinely heterogeneous instrument rather than a relabelling.
    """
    directions = _sample_directions()
    angles = plan_psi(
        directions.hour_angle_rad, directions.dec_rad, _SITE_LATITUDE_RAD
    ) + sign * np.asarray(directions.alt_rad)

    block = _evaluate(_term((mount_type, mount_type)), directions)

    expected = np.stack([plan_rotation(float(angle)) for angle in angles])
    np.testing.assert_allclose(block, expected, rtol=0.0, atol=1e-15)


def test_each_antenna_row_carries_its_own_mount() -> None:
    """A heterogeneous array is per-antenna, not per-array (Section 20.7)."""
    directions = _sample_directions()
    term = _term(("alt-az", "fixed"))

    rotating = _evaluate(term, directions, antenna_idx=0)
    static = _evaluate(term, directions, antenna_idx=1)

    np.testing.assert_array_equal(
        static, np.broadcast_to(np.eye(2, dtype=np.complex128), static.shape)
    )
    assert float(np.max(np.abs(rotating - static))) > 0.1


def test_an_antenna_row_outside_the_instrument_is_rejected() -> None:
    """A row/mount mismatch is a defect, not a silently reused last entry."""
    with pytest.raises(IndexError):
        _evaluate(_term(("alt-az", "alt-az")), _sample_directions(), antenna_idx=2)


def test_an_unmodelled_mount_cannot_reach_the_term() -> None:
    """The constructor refuses what Section 24's R12 rejects one level up."""
    with pytest.raises(ValueError, match="mount_type"):
        _term(("alt-az", "phased"))


# ---------------------------------------------------------------------------
# I2 and I3 -- declared flags are true, and the batch has the mandated shape
# ---------------------------------------------------------------------------


def test_p_is_direction_and_time_dependent_and_achromatic() -> None:
    """Section 20.12's row for ``P``, as declared and as computed."""
    term = _term()
    assert term.name == "P"
    assert term.term_status == "implemented"
    assert term.is_direction_dependent is True
    assert term.is_time_dependent is True
    assert term.is_frequency_dependent is False

    directions = _sample_directions()
    low = _evaluate(term, directions)
    high = np.asarray(
        term.compute_jones_batch(
            antenna_idx=0,
            directions=directions,
            frequency_hz=8.0e8,
            freq_idx=3,
            time_mjd=60000.0,
            time_idx=0,
            backend=_BACKEND,
            dtype=np.complex128,
        )
    )
    np.testing.assert_array_equal(low, high)


def test_a_dde_term_returns_one_matrix_per_direction() -> None:
    """I3: ``(n_dir, 2, 2)``, never a single broadcast matrix."""
    for n_dir in (1, 5, 37):
        block = _evaluate(_term(), _sample_directions(n_dir))
        assert block.shape == (n_dir, 2, 2)


def test_p_is_unitary_everywhere_and_never_diagonal_when_it_rotates() -> None:
    """I2: ``P P^H = I2`` exactly, and the declared ``False`` has a witness."""
    term = _term()
    assert term.is_unitary() is True
    assert term.is_diagonal() is False
    assert term.is_scalar() is False
    assert term.is_identity() is False

    block = _evaluate(term, _sample_directions(48, spread_deg=80.0))
    products = np.einsum("nij,nkj->nik", block, block.conjugate())
    np.testing.assert_allclose(
        products,
        np.broadcast_to(np.eye(2, dtype=np.complex128), products.shape),
        rtol=0.0,
        atol=1e-15,
    )
    # The witness for is_diagonal() == False: an off-diagonal that is not zero.
    assert float(np.max(np.abs(block[:, 0, 1]))) > 1e-3

    # P is real: a rotation carries no phase, which is what distinguishes it
    # from the circular-basis composite S P S^H.
    np.testing.assert_array_equal(block.imag, np.zeros_like(block.imag))


def test_a_non_rotating_array_declares_the_identity_it_is() -> None:
    """The flags are computed from the resolved mounts, never hard-coded.

    R7 makes this configuration unreachable from a document, and the flags are
    still computed rather than asserted, because a hard-coded ``False`` is a
    claim nothing checks -- the vacuous-flag failure mode I2 exists to prevent.
    """
    term = _term(("fixed", "equatorial"))
    assert term.is_identity() is True
    assert term.is_diagonal() is True
    assert term.is_scalar() is True
    assert term.is_unitary() is True


# ---------------------------------------------------------------------------
# I9 -- P is wide-field
# ---------------------------------------------------------------------------


def test_psi_varies_across_a_wide_field_and_is_constant_across_a_narrow_one() -> None:
    """I9, both halves, with the narrow-field limit stated as a limit.

    Over a 20-degree batch ``psi`` varies by a measurable amount; as the batch
    shrinks it converges on the single-direction value.  This is the property
    that makes ``P`` direction-dependent rather than a per-antenna scalar
    rotation, and it is why the two deleted wide-field rotation classes are
    subsumed exactly rather than approximately.

    DEVIATION FROM SECTION 27's LITERAL I9.  The invariant reads "over a
    0.01-degree batch it is constant to ``1e-12``".  That is not achievable and
    is not the physics: ``dpsi/dtheta`` is of order unity away from the poles,
    so a 0.01-degree batch spans of order ``1e-4`` radians of direction and
    therefore of order ``1e-5`` radians of ``psi``.  A test asserting ``1e-12``
    there would be asserting that ``P`` is *not* wide-field, which contradicts
    the same invariant's first half.  What is asserted instead is strictly
    stronger than a single tolerance: the spread is first order in the field
    width (halving the width halves it, to one part in a thousand), and it does
    reach ``1e-12`` once the batch is small enough for that scaling to take it
    there.
    """
    centre_alt = math.radians(50.0)
    centre_az = math.radians(70.0)

    def psi_over(half_width_deg: float, n_dir: int = 41) -> np.ndarray:
        offsets = np.radians(np.linspace(-half_width_deg, half_width_deg, n_dir))
        directions = _directions(
            alt_rad=centre_alt + offsets,
            az_rad=np.full(n_dir, centre_az),
        )
        return parallactic_angle(
            hour_angle_rad=directions.hour_angle_rad,
            dec_rad=directions.dec_rad,
            latitude_rad=_SITE_LATITUDE_RAD,
        )

    centre = psi_over(0.0, 1)[0]

    # Wide: 20 degrees across, and psi moves by more than a degree.
    assert float(np.ptp(psi_over(10.0))) > math.radians(1.0)

    # First order in the field width: halving the batch halves the spread.
    spreads = [float(np.ptp(psi_over(width))) for width in (0.02, 0.01, 0.005)]
    assert spreads[0] / spreads[1] == pytest.approx(2.0, rel=1e-3)
    assert spreads[1] / spreads[2] == pytest.approx(2.0, rel=1e-3)

    # And the limit itself: a batch small enough that the linear term is below
    # 1e-12 has psi constant to 1e-12 and equal to the single-direction value.
    tiny = psi_over(1.0e-11)
    assert float(np.ptp(tiny)) < 1e-12
    np.testing.assert_allclose(tiny, centre, rtol=0.0, atol=1e-12)


def test_the_matrix_itself_varies_across_a_wide_batch() -> None:
    """The same statement one level up: a wide batch is not one matrix."""
    n_dir = 32
    directions = _directions(
        alt_rad=np.radians(np.linspace(40.0, 60.0, n_dir)),
        az_rad=np.full(n_dir, math.radians(70.0)),
    )
    block = _evaluate(_term(), directions)

    assert float(np.max(np.abs(block - block[0]))) > 1e-2


# ---------------------------------------------------------------------------
# I6 -- the chain-order correction, as algebra
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("chi_deg", [17.0, -63.0])
@pytest.mark.parametrize("psi_rad", [0.31, -1.24])
def test_c_times_p_is_the_receptor_at_the_combined_angle(
    chi_deg: float, psi_rad: float
) -> None:
    """I6: ``C P = M(basis) R(chi + psi)``, and ``P C`` is not.

    This is the whole reason Tier 7F moves ``P``.  For a *linear* receptor
    ``M = I2`` and the two orders agree, which is exactly the case Tier 5
    tested; for a circular receptor they are different matrices, and the
    corrected order is the one that composes into a single rotation of the
    receptor pair.
    """
    chi = math.radians(chi_deg)
    receptor = np.asarray(SKY_NORTH_EAST_TO_CIRCULAR_RL) @ basis_rotation_matrix(chi)
    rotation = plan_rotation(psi_rad)
    combined = np.asarray(SKY_NORTH_EAST_TO_CIRCULAR_RL) @ basis_rotation_matrix(
        chi + psi_rad
    )

    np.testing.assert_allclose(receptor @ rotation, combined, rtol=0.0, atol=1e-15)
    assert not np.allclose(rotation @ receptor, combined, atol=1e-6)

    # SCI-006 makes the linear receptor ``P R(chi)``.  Its corrected sky-side
    # placement still combines angles, while the reversed placement does not.
    linear = np.asarray(SKY_NORTH_EAST_TO_LINEAR_XY) @ basis_rotation_matrix(chi)
    linear_combined = np.asarray(SKY_NORTH_EAST_TO_LINEAR_XY) @ basis_rotation_matrix(
        chi + psi_rad
    )
    np.testing.assert_allclose(linear @ rotation, linear_combined, rtol=0.0, atol=1e-15)
    assert not np.allclose(rotation @ linear, linear_combined, atol=1e-6)


@pytest.mark.parametrize("psi_rad", [0.4, -0.9, 2.7])
def test_a_field_rotation_is_a_pair_of_phases_in_the_circular_basis(
    psi_rad: float,
) -> None:
    """Section 12.1's identity: ``S R(psi) S^H = diag(e^{-i psi}, e^{+i psi})``.

    Under the Tier 5 order the circular ``(R, L)`` pair would be *mixed* by a
    real 2x2 rotation, which is not what a field rotation does to circular
    polarizations; under the corrected order it is a pair of opposite phases.
    """
    s_matrix = np.asarray(SKY_NORTH_EAST_TO_CIRCULAR_RL)
    composed = s_matrix @ plan_rotation(psi_rad) @ s_matrix.conj().T

    expected = np.diag(
        np.array(
            [
                complex(math.cos(psi_rad), -math.sin(psi_rad)),
                complex(math.cos(psi_rad), math.sin(psi_rad)),
            ],
            dtype=np.complex128,
        )
    )
    np.testing.assert_allclose(composed, expected, rtol=0.0, atol=1e-15)
    assert abs(composed[0, 1]) < 1e-15


@pytest.mark.parametrize("psi_rad", [0.37, -1.1])
def test_a_field_rotation_turns_q_and_u_by_twice_the_angle(psi_rad: float) -> None:
    """The linear-basis statement, on the coherency matrix itself.

    ``R(psi) B R(psi)^T`` leaves Stokes ``I`` and ``V`` alone and rotates
    ``(Q, U)`` by ``2 psi``.  Written out here from the Stokes decomposition
    rather than taken from the production converter, because a test that asks
    the code what the answer is has asked nothing.
    """
    stokes_i, stokes_q, stokes_u, stokes_v = 3.0, 0.8, -0.5, 0.2
    coherency = 0.5 * np.array(
        [
            [stokes_i + stokes_q, stokes_u + 1j * stokes_v],
            [stokes_u - 1j * stokes_v, stokes_i - stokes_q],
        ],
        dtype=np.complex128,
    )

    rotation = plan_rotation(psi_rad)
    rotated = rotation @ coherency @ rotation.T

    cos2, sin2 = math.cos(2.0 * psi_rad), math.sin(2.0 * psi_rad)
    expected = 0.5 * np.array(
        [
            [
                stokes_i + stokes_q * cos2 + stokes_u * sin2,
                -stokes_q * sin2 + stokes_u * cos2 + 1j * stokes_v,
            ],
            [
                -stokes_q * sin2 + stokes_u * cos2 - 1j * stokes_v,
                stokes_i - stokes_q * cos2 - stokes_u * sin2,
            ],
        ],
        dtype=np.complex128,
    )
    np.testing.assert_allclose(rotated, expected, rtol=0.0, atol=1e-15)


def test_an_unpolarized_coherency_is_invariant_under_a_field_rotation() -> None:
    """The I-row invariant, as algebra: ``R I2 R^T = I2`` for any angle."""
    coherency = 0.5 * np.eye(2, dtype=np.complex128)
    for psi in (0.2, -1.7, 3.0):
        rotation = plan_rotation(psi)
        np.testing.assert_allclose(
            rotation @ coherency @ rotation.T, coherency, rtol=0.0, atol=1e-16
        )


# ---------------------------------------------------------------------------
# Through the solver
# ---------------------------------------------------------------------------


def _cube(
    tmp_path,
    jones: dict[str, Any] | None,
    *,
    mount_types: Any = "alt-az",
    polarized: bool = True,
    **section_overrides: Any,
) -> np.ndarray:
    instrument, beam_system, receptors, jones_terms, frequencies = (
        solver_components_with_jones(
            tmp_path, jones, mount_types=mount_types, **section_overrides
        )
    )
    return np.asarray(
        calculate_visibility(
            instrument=instrument,
            beam_system=beam_system,
            source_arrays=_workload_point_sources(polarized=polarized, gaussian=False),
            location=WORKLOAD_LOCATION,
            time_grid=WORKLOAD_TIME_GRID,
            frequencies=frequencies,
            backend=_BACKEND,
            receptors=receptors,
            jones_terms=jones_terms,
        )
    )


def test_a_configured_parallactic_angle_changes_the_visibilities(tmp_path) -> None:
    """I7, for ``P``: a polarized sky on an alt-az array is not the same run."""
    clean = _cube(tmp_path, None, mount_types="fixed")
    rotated = _cube(tmp_path, {"P": {"enabled": True}})

    difference = np.max(np.abs(rotated - clean)) / np.max(np.abs(clean))
    assert float(difference) > 1e-10


def test_an_unpolarized_sky_on_a_homogeneous_array_is_untouched(tmp_path) -> None:
    """Section 20.7's I-row invariant, end to end.

    Every antenna shares one mount and one latitude, so every antenna's ``P`` is
    the same rotation; an unpolarized coherency commutes with it, and the
    reported cube is the one the run would have produced with no field rotation
    at all.  This is the statement that makes the *polarized* difference above
    a polarization effect rather than a bug.
    """
    clean = _cube(tmp_path, None, mount_types="fixed", polarized=False)
    rotated = _cube(tmp_path, {"P": {"enabled": True}}, polarized=False)

    # The cross hands of an unpolarized sky are exactly zero, so the comparison
    # is absolute against the cube's own scale rather than relative: R R^T is
    # the identity to rounding, not bit for bit, and 1e-14 of the peak is three
    # orders of magnitude below anything a field rotation would produce.
    scale = float(np.max(np.abs(clean)))
    np.testing.assert_allclose(rotated, clean, rtol=1e-12, atol=1e-14 * scale)


def test_a_heterogeneous_array_breaks_that_invariance(tmp_path) -> None:
    """Two different mounts on one baseline: even Stokes ``I`` is affected.

    ``P_p R(0)^T`` is not a similarity transform of the coherency when the two
    antennas rotate by different angles, which is precisely the effect a
    per-antenna ``mount_type`` exists to model.
    """
    clean = _cube(tmp_path, None, mount_types="fixed", polarized=False)
    mixed = _cube(
        tmp_path,
        {"P": {"enabled": True}},
        mount_types=("alt-az", "fixed"),
        polarized=False,
    )

    assert float(np.max(np.abs(mixed - clean))) / float(np.max(np.abs(clean))) > 1e-6


def test_a_rotated_receptor_composes_with_the_field_rotation(tmp_path) -> None:
    """Tier 5's static ``chi`` and Tier 7F's ``psi(t)`` add, and are not rejected.

    ``Tier5ReceptorFeedPlan.md`` Section 12.3 refused the combination outright
    and said it would become legal "when Tier 7 implements ``P``".  This is that
    discharge: the run is accepted, and the composite is the receptor at
    ``chi + psi`` rather than a double rotation or a dropped one.
    """
    receptors = {"receptors": {"default": {"feed_rotation_deg": 31.0}}}

    rotated_feeds_only = _cube(tmp_path, None, mount_types="fixed", **receptors)
    both = _cube(tmp_path, {"P": {"enabled": True}}, **receptors)

    assert np.all(np.isfinite(both))
    assert float(np.max(np.abs(both - rotated_feeds_only))) > 0.0


def test_the_term_is_stateless_across_direction_batches() -> None:
    """No memo between calls, because time blocks may run concurrently.

    ``execute_time_blocks`` is allowed to evaluate two time steps at once, so a
    term that cached ``psi`` against the last batch it saw would be a data race
    with a wrong answer rather than a slow one.  Evaluating two batches in
    alternation and getting each batch's own answer is the observable form of
    that.
    """
    term = _term()
    first = _sample_directions(8)
    second = _directions(
        alt_rad=np.radians(np.linspace(20.0, 40.0, 8)),
        az_rad=np.radians(np.linspace(200.0, 260.0, 8)),
    )

    a1 = _evaluate(term, first)
    b1 = _evaluate(term, second)
    a2 = _evaluate(term, first)
    b2 = _evaluate(term, second)

    np.testing.assert_array_equal(a1, a2)
    np.testing.assert_array_equal(b1, b2)
    assert not np.allclose(a1, b1)
    assert not hasattr(term, "__dict__") or all(
        not key.startswith("_cache") for key in vars(term)
    )


def test_the_resolved_term_uses_the_instruments_own_latitude(tmp_path) -> None:
    """``psi`` is evaluated at the site, not at a default (Section 20.7)."""
    resolved = resolve_for(tmp_path, {"P": {"enabled": True}}, mount_types="alt-az")
    term = resolved.term("P")

    assert term is not None
    assert term.latitude_rad == pytest.approx(_SITE_LATITUDE_RAD, abs=1e-12)
