"""Tier 7B: the direction batch every Jones term is evaluated over.

``Tier7JonesSciencePlan.md`` Section 30 gives this module three jobs:
construction, immutability, and frame consistency -- that the horizontal
``(alt, az)`` description and the equatorial ``(ra, dec, H)`` description in one
batch really do describe the same directions, including quadrant, which is the
whole reason the equatorial half exists (Section 13.2).
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
from astropy import units as u
from astropy.coordinates import TETE, AltAz, EarthLocation, SkyCoord
from astropy.time import Time

from radiosim.core.jones import DirectionBatch
from radiosim.core.jones.directions import (
    DirectionBatchError,
    equatorial_from_horizontal,
    hour_angle_from_lst,
)

LOCATION = EarthLocation.from_geodetic(
    21.4283 * u.deg,
    -30.72152 * u.deg,
    1073.0 * u.m,
)
OBSTIME = Time("2025-01-01T00:00:00", format="isot", scale="utc")


def _batch(n_dir: int = 3, **overrides) -> DirectionBatch:
    values = np.linspace(0.1, 1.0, n_dir)
    fields = {
        "alt_rad": values,
        "az_rad": values,
        "dir_l": values,
        "dir_m": values,
        "dir_n": values,
        "ra_rad": values,
        "dec_rad": values,
        "hour_angle_rad": values,
        "n_dir": n_dir,
    }
    fields.update(overrides)
    return DirectionBatch(**fields)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_every_array_is_owned_promoted_and_read_only() -> None:
    source = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    batch = _batch(3, alt_rad=source)

    assert batch.alt_rad.dtype == np.float64
    assert not batch.alt_rad.flags.writeable
    assert batch.alt_rad is not source

    # Mutating the caller's array cannot reach the batch.
    source[0] = 99.0
    assert batch.alt_rad[0] == pytest.approx(0.1, abs=1e-7)

    with pytest.raises(ValueError):
        batch.alt_rad[0] = 1.0


def test_the_batch_itself_is_frozen() -> None:
    batch = _batch()
    with pytest.raises(dataclasses.FrozenInstanceError):
        batch.n_dir = 7  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        batch.alt_rad = np.zeros(3)  # type: ignore[misc]


def test_an_array_of_the_wrong_length_is_rejected() -> None:
    with pytest.raises(DirectionBatchError) as excinfo:
        _batch(3, dec_rad=np.zeros(2))
    assert "dec_rad" in str(excinfo.value)
    assert "n_dir" in str(excinfo.value)


def test_a_two_dimensional_array_is_rejected() -> None:
    with pytest.raises(DirectionBatchError) as excinfo:
        _batch(3, dir_l=np.zeros((3, 1)))
    assert "one-dimensional" in str(excinfo.value)


def test_a_non_finite_entry_is_rejected() -> None:
    with pytest.raises(DirectionBatchError) as excinfo:
        _batch(3, dir_n=np.array([0.0, np.nan, 1.0]))
    assert "finite" in str(excinfo.value)


def test_a_non_integer_direction_count_is_rejected() -> None:
    values = np.linspace(0.1, 1.0, 3)
    with pytest.raises(DirectionBatchError):
        DirectionBatch(
            alt_rad=values,
            az_rad=values,
            dir_l=values,
            dir_m=values,
            dir_n=values,
            ra_rad=values,
            dec_rad=values,
            hour_angle_rad=values,
            n_dir=3.0,  # type: ignore[arg-type]
        )


def test_an_empty_batch_is_legal() -> None:
    """A time step with nothing above the horizon is a batch, not an error."""
    empty = DirectionBatch(
        alt_rad=np.array([]),
        az_rad=np.array([]),
        dir_l=np.array([]),
        dir_m=np.array([]),
        dir_n=np.array([]),
        ra_rad=np.array([]),
        dec_rad=np.array([]),
        hour_angle_rad=np.array([]),
        n_dir=0,
    )
    assert len(empty) == 0


def test_the_declared_field_set_is_the_designed_one() -> None:
    """Section 13.2's nine fields, in order."""
    assert _batch().field_names == (
        "alt_rad",
        "az_rad",
        "dir_l",
        "dir_m",
        "dir_n",
        "ra_rad",
        "dec_rad",
        "hour_angle_rad",
        "n_dir",
    )


# ---------------------------------------------------------------------------
# Hour angle
# ---------------------------------------------------------------------------


def test_hour_angle_is_lst_minus_ra_wrapped_to_pi() -> None:
    ra = np.array([0.0, 1.0, 3.0, 6.0])
    lst = 0.5
    hour_angle = hour_angle_from_lst(lst, ra)

    assert np.all(hour_angle >= -np.pi)
    assert np.all(hour_angle < np.pi)
    # Equal modulo 2 pi to the unwrapped difference.
    unwrapped = lst - ra
    np.testing.assert_allclose(
        np.mod(hour_angle - unwrapped, 2.0 * np.pi),
        0.0,
        atol=1e-12,
    )


def test_from_horizontal_fills_the_equatorial_half_and_the_count() -> None:
    batch = DirectionBatch.from_horizontal(
        alt_rad=np.array([1.0, 1.1]),
        az_rad=np.array([0.3, 0.4]),
        dir_l=np.array([0.1, 0.2]),
        dir_m=np.array([0.2, 0.3]),
        dir_n=np.array([0.9, 0.8]),
        latitude_rad=-0.536,
        local_sidereal_time_rad=1.25,
    )

    assert batch.n_dir == 2
    hour_angle, declination = equatorial_from_horizontal(
        alt_rad=np.array([1.0, 1.1]),
        az_rad=np.array([0.3, 0.4]),
        latitude_rad=-0.536,
    )
    np.testing.assert_array_equal(batch.hour_angle_rad, hour_angle)
    np.testing.assert_array_equal(batch.dec_rad, declination)
    # RA is LST - H, in [0, 2 pi).
    np.testing.assert_allclose(
        np.mod(batch.ra_rad + batch.hour_angle_rad - 1.25, 2.0 * np.pi),
        0.0,
        atol=1e-15,
    )
    assert np.all(batch.ra_rad >= 0.0)
    assert np.all(batch.ra_rad < 2.0 * np.pi)


def test_the_equatorial_half_round_trips_back_to_the_horizontal_half() -> None:
    """Exact by construction: the two halves are one set of directions.

    The forward transform is written out here from the standard relations, so
    this is a round trip through an independent expression rather than a
    restatement of the production one.
    """
    latitude = -0.536
    alt = np.array([0.05, 0.4, 0.9, 1.3, 1.5])
    az = np.array([0.0, 1.1, 2.9, 4.2, 6.1])
    batch = DirectionBatch.from_horizontal(
        alt_rad=alt,
        az_rad=az,
        dir_l=np.cos(alt) * np.sin(az),
        dir_m=np.cos(alt) * np.cos(az),
        dir_n=np.sin(alt),
        latitude_rad=latitude,
        local_sidereal_time_rad=2.0,
    )

    declination = batch.dec_rad
    hour_angle = batch.hour_angle_rad
    sin_alt = np.sin(declination) * np.sin(latitude) + np.cos(declination) * np.cos(
        latitude
    ) * np.cos(hour_angle)
    cos_alt_sin_az = -np.cos(declination) * np.sin(hour_angle)
    cos_alt_cos_az = np.cos(latitude) * np.sin(declination) - np.sin(latitude) * np.cos(
        declination
    ) * np.cos(hour_angle)

    np.testing.assert_allclose(sin_alt, np.sin(alt), atol=1e-14)
    np.testing.assert_allclose(cos_alt_sin_az, np.cos(alt) * np.sin(az), atol=1e-14)
    np.testing.assert_allclose(cos_alt_cos_az, np.cos(alt) * np.cos(az), atol=1e-14)


def test_the_hour_angle_keeps_its_quadrant_east_and_west_of_the_meridian() -> None:
    """East of the meridian is ``H < 0``, west is ``H > 0``, in both hemispheres.

    This is the property an arcsine would destroy, and the reason the batch
    carries the hour angle rather than leaving each term to recover it.
    """
    for latitude in (-0.536, +0.7):
        east = DirectionBatch.from_horizontal(
            alt_rad=np.array([0.8]),
            az_rad=np.array([np.pi / 2]),  # due East
            dir_l=np.array([0.0]),
            dir_m=np.array([0.0]),
            dir_n=np.array([1.0]),
            latitude_rad=latitude,
            local_sidereal_time_rad=0.0,
        )
        west = DirectionBatch.from_horizontal(
            alt_rad=np.array([0.8]),
            az_rad=np.array([3.0 * np.pi / 2]),  # due West
            dir_l=np.array([0.0]),
            dir_m=np.array([0.0]),
            dir_n=np.array([1.0]),
            latitude_rad=latitude,
            local_sidereal_time_rad=0.0,
        )
        assert east.hour_angle_rad[0] < 0.0
        assert west.hour_angle_rad[0] > 0.0


# ---------------------------------------------------------------------------
# Frame consistency
# ---------------------------------------------------------------------------


def test_the_equatorial_half_matches_an_independent_astropy_derivation() -> None:
    """The derived apparent equatorial angles agree with astropy's own.

    The batch derives ``(H, dec)`` from ``(alt, az)`` with nothing but the site
    latitude and the local apparent sidereal time.  Here the same directions are
    carried the other way -- ICRS to the true-equinox-of-date frame -- by astropy,
    and the two must agree.  This is what pins the latitude and sidereal-time
    conventions; the round-trip test above pins the algebra.

    The residual is the topocentric part astropy models and the spherical
    relation does not (diurnal aberration and parallax at a station on a
    rotating Earth), which is of order an arcsecond.
    """
    coords = SkyCoord(
        ra=np.array([10.0, 80.0, 200.0, 330.0]) * u.deg,
        dec=np.array([-60.0, -20.0, 5.0, 40.0]) * u.deg,
        frame="icrs",
    )
    altaz = coords.transform_to(AltAz(obstime=OBSTIME, location=LOCATION))
    lst = float(OBSTIME.sidereal_time("apparent", longitude=LOCATION.lon).rad)

    alt = altaz.alt.rad
    az = altaz.az.rad
    batch = DirectionBatch.from_horizontal(
        alt_rad=alt,
        az_rad=az,
        dir_l=np.cos(alt) * np.sin(az),
        dir_m=np.cos(alt) * np.cos(az),
        dir_n=np.sin(alt),
        latitude_rad=float(LOCATION.lat.rad),
        local_sidereal_time_rad=lst,
    )

    apparent = coords.transform_to(TETE(obstime=OBSTIME, location=LOCATION))
    np.testing.assert_allclose(batch.dec_rad, apparent.dec.rad, atol=1e-5)
    np.testing.assert_allclose(
        np.mod(batch.ra_rad - apparent.ra.rad + np.pi, 2.0 * np.pi) - np.pi,
        0.0,
        atol=1e-5,
    )
    np.testing.assert_allclose(
        batch.hour_angle_rad,
        hour_angle_from_lst(lst, apparent.ra.rad),
        atol=1e-5,
    )


def test_the_direction_cosines_are_a_unit_vector_in_the_enu_frame() -> None:
    alt = np.array([0.2, 0.9, 1.4])
    az = np.array([0.0, 2.0, 4.5])
    batch = DirectionBatch.from_horizontal(
        alt_rad=alt,
        az_rad=az,
        dir_l=np.cos(alt) * np.sin(az),
        dir_m=np.cos(alt) * np.cos(az),
        dir_n=np.sin(alt),
        latitude_rad=-0.536,
        local_sidereal_time_rad=0.0,
    )

    norm = batch.dir_l**2 + batch.dir_m**2 + batch.dir_n**2
    np.testing.assert_allclose(norm, 1.0, atol=1e-15)
    np.testing.assert_allclose(batch.dir_n, np.sin(alt), atol=0.0, rtol=0.0)
