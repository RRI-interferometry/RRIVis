"""SCI-007's fixture-scoped public polarization-frame accuracy bound."""

from __future__ import annotations

from pathlib import Path

import astropy
import numpy as np
from astropy import units as u
from astropy.coordinates import AltAz, EarthLocation, SkyCoord
from astropy.time import Time
from astropy.utils import iers

from radiosim.core.jones.directions import DirectionBatch
from radiosim.core.jones.parallactic import parallactic_angle

_SITE = EarthLocation.from_geodetic(
    lon=21.42830 * u.deg,
    lat=-30.72152 * u.deg,
    height=1073.0 * u.m,
)
_START_TIME = Time("2025-01-01T00:00:00", format="isot", scale="utc")
_TIMES = _START_TIME + np.arange(3, dtype=np.float64) * 120.0 * u.s
_SOURCES_ICRS = SkyCoord(
    ra=np.array([20.0, 25.0, 15.0]) * u.deg,
    dec=np.array([-30.72, -26.0, -35.0]) * u.deg,
    frame="icrs",
)


def _wrap_to_pi(values: np.ndarray) -> np.ndarray:
    return np.mod(values + np.pi, 2.0 * np.pi) - np.pi


def test_public_astropy_oracle_bounds_sci007_fixture_frame_rotation() -> None:
    """Bound the retained source-time grid without optional dependencies."""
    bundled_path = Path(iers.IERS_A_FILE).resolve()
    table = iers.IERS_A.open(iers.IERS_A_FILE)
    table_path = Path(str(table.meta.get("data_path", ""))).resolve()
    iers_context = (
        f"astropy={astropy.__version__}, table_class={type(table).__name__}, "
        f"bundled_path={bundled_path}, table_path={table_path}"
    )
    assert type(table) is iers.IERS_A, iers_context
    assert table_path == bundled_path, iers_context

    delta_public = np.empty((_TIMES.size, _SOURCES_ICRS.size), dtype=np.float64)
    with (
        iers.conf.set_temp("auto_download", False),
        iers.earth_orientation_table.set(table),
    ):
        assert iers.conf.auto_download is False, iers_context
        assert iers.earth_orientation_table.get() is table, iers_context

        for time_index, obstime in enumerate(_TIMES):
            altaz_frame = AltAz(
                obstime=obstime,
                location=_SITE,
                pressure=0.0 * u.hPa,
            )
            source_altaz = _SOURCES_ICRS.transform_to(altaz_frame)
            altitude = source_altaz.alt.to_value(u.rad)
            azimuth = source_altaz.az.to_value(u.rad)
            cos_altitude = np.cos(altitude)
            directions = DirectionBatch.from_horizontal(
                alt_rad=altitude,
                az_rad=azimuth,
                dir_l=cos_altitude * np.sin(azimuth),
                dir_m=cos_altitude * np.cos(azimuth),
                dir_n=np.sin(altitude),
                latitude_rad=float(_SITE.lat.to_value(u.rad)),
                local_sidereal_time_rad=float(
                    obstime.sidereal_time("apparent", longitude=_SITE.lon).to_value(
                        u.rad
                    )
                ),
            )
            psi_radiosim = parallactic_angle(
                hour_angle_rad=directions.hour_angle_rad,
                dec_rad=directions.dec_rad,
                latitude_rad=float(_SITE.lat.to_value(u.rad)),
            )
            zenith_icrs = SkyCoord(
                az=0.0 * u.deg,
                alt=90.0 * u.deg,
                frame=altaz_frame,
            ).transform_to("icrs")
            source_to_zenith = _SOURCES_ICRS.position_angle(zenith_icrs).to_value(u.rad)
            delta_public[time_index] = _wrap_to_pi(psi_radiosim - source_to_zenith)

    absolute_delta = np.abs(delta_public)
    minimum_delta = float(np.min(absolute_delta))
    maximum_delta = float(np.max(absolute_delta))
    maximum_spin_two_effect = float(np.max(np.abs(np.exp(2j * delta_public) - 1.0)))
    bound_context = (
        f"{iers_context}, delta_public_rad={delta_public.tolist()}, "
        f"min_abs={minimum_delta:.16e}, max_abs={maximum_delta:.16e}, "
        f"max_spin_two={maximum_spin_two_effect:.16e}"
    )
    assert 6.0e-4 < minimum_delta, bound_context
    assert maximum_delta < 1.2e-3, bound_context
    assert maximum_spin_two_effect < 2.4e-3, bound_context
