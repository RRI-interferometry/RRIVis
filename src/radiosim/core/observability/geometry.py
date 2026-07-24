"""Observability-specific geometry helpers tied to ``BeamSkyProjection``.

General coordinate-system helpers (``radec_to_za_az``, ``wrap_ra_deg``,
``angular_separation_deg``, ``split_wrapped_path``, etc.) live in
:mod:`radiosim.utils.coordinates`.
"""

from __future__ import annotations

from collections.abc import Callable
from importlib import import_module
from typing import TYPE_CHECKING, Any

import numpy as np

from radiosim.core.beam import (
    BeamDisplayNormalizationError,
    BeamSystem,
    NonFiniteBeamResponseError,
)
from radiosim.core.instrument import AntennaId
from radiosim.core.jones.beam.projection import BeamSkyProjection
from radiosim.utils.coordinates import radec_to_za_az

if TYPE_CHECKING:
    from radiosim.backends.base import ArrayBackend


def _evaluate_reference_power(
    *,
    beam_system: BeamSystem,
    reference_antenna: AntennaId,
    zenith_angle_rad: np.ndarray,
    azimuth_rad: np.ndarray,
    frequency_hz: float,
    time_mjd: float,
    backend: ArrayBackend | None = None,
) -> np.ndarray:
    """Evaluate canonical unpolarized power as ``0.5 trace(J J^H)``."""
    if type(beam_system) is not BeamSystem:
        raise TypeError("beam_system must be an exact BeamSystem")
    if type(reference_antenna) is not AntennaId:
        raise TypeError("reference_antenna must be an exact AntennaId")
    if type(frequency_hz) is not float or not np.isfinite(frequency_hz):
        raise TypeError("frequency_hz must be an exact finite float")
    if type(time_mjd) is not float or not np.isfinite(time_mjd):
        raise TypeError("time_mjd must be an exact finite float")

    za = np.asarray(zenith_angle_rad, dtype=np.float64)
    az = np.asarray(azimuth_rad, dtype=np.float64)
    if za.shape != az.shape:
        raise ValueError("zenith_angle_rad and azimuth_rad must have matching shapes")
    if not np.all(np.isfinite(za)) or not np.all(np.isfinite(az)):
        raise NonFiniteBeamResponseError(
            "Observability directions must contain only finite angles."
        )

    altitude = np.pi / 2.0 - za
    jones = beam_system.evaluate_jones(
        reference_antenna,
        altitude_rad=altitude.reshape(-1),
        azimuth_rad=az.reshape(-1),
        frequency_hz=frequency_hz,
        time_mjd=time_mjd,
        backend=backend,
    )
    host_jones = np.asarray(jones)
    expected_shape = (za.size, 2, 2)
    if host_jones.shape != expected_shape:
        raise NonFiniteBeamResponseError(
            "BeamSystem returned Jones shape "
            f"{host_jones.shape}; expected {expected_shape}."
        )
    if not np.all(np.isfinite(host_jones)):
        raise NonFiniteBeamResponseError(
            "BeamSystem returned a non-finite observability Jones response."
        )
    power = 0.5 * np.sum(np.abs(host_jones) ** 2, axis=(-2, -1))
    power = np.asarray(power, dtype=np.float64)
    power[altitude.reshape(-1) <= 0.0] = 0.0
    if not np.all(np.isfinite(power)):
        raise NonFiniteBeamResponseError(
            "Canonical observability beam power is non-finite."
        )
    return power.reshape(za.shape)


def compute_beam_power_on_full_sky_grid(
    beam_power_func: Callable[[np.ndarray, np.ndarray], np.ndarray],
    zenith_ra_deg: float,
    zenith_dec_deg: float,
    ra_grid_deg: np.ndarray,
    dec_grid_deg: np.ndarray,
    max_za_deg: float = 90.0,
) -> BeamSkyProjection:
    """Evaluate a beam model on a full-sky RA/Dec grid."""
    ra_mesh, dec_mesh = np.meshgrid(ra_grid_deg, dec_grid_deg)
    za_rad, az_rad = radec_to_za_az(
        ra_mesh,
        dec_mesh,
        zenith_ra_deg=zenith_ra_deg,
        zenith_dec_deg=zenith_dec_deg,
    )
    power = np.array(beam_power_func(za_rad, az_rad), dtype=float, copy=True)
    if power.shape != za_rad.shape:
        raise ValueError("beam_power_func must return the input direction shape")
    power[za_rad > np.deg2rad(max_za_deg)] = np.nan
    with np.errstate(divide="ignore", invalid="ignore"):
        power_db = 10.0 * np.log10(np.where(np.isnan(power), np.nan, power + 1e-30))
    power_db[np.isnan(power)] = np.nan
    return BeamSkyProjection(
        ra_grid_deg=ra_grid_deg,
        dec_grid_deg=dec_grid_deg,
        power_db=power_db,
        zenith_ra_deg=zenith_ra_deg,
        zenith_dec_deg=zenith_dec_deg,
        max_za_deg=max_za_deg,
    )


def compute_beam_map_on_healpix(
    *,
    beam_system: BeamSystem,
    reference_antenna: AntennaId,
    nside: int,
    zenith_ra_deg: float,
    zenith_dec_deg: float,
    frequency_hz: float,
    time_mjd: float,
) -> np.ndarray:
    """Evaluate the canonical selected beam on every HEALPix pixel.

    Parameters
    ----------
    beam_system
        Canonical loaded beam service.
    reference_antenna
        Exact selected canonical antenna identity.
    nside
        HEALPix nside of the output map.  The function returns a dense
        ``ndarray`` of shape ``(12 * nside ** 2,)`` in RING ordering.
    zenith_ra_deg, zenith_dec_deg
        Pointing centre (instantaneous zenith) in degrees.
    frequency_hz
        Exact BeamSystem observation frequency.
    time_mjd
        Exact deterministic beam-evaluation time seam.

    Returns
    -------
    np.ndarray
        Owned, finite, non-writeable float64 HEALPix display map.
    """
    hp: Any = import_module("healpy")

    if type(nside) is not int or not hp.isnsideok(nside):
        raise ValueError("nside must be a strict positive HEALPix NSIDE")
    for name, value in (
        ("zenith_ra_deg", zenith_ra_deg),
        ("zenith_dec_deg", zenith_dec_deg),
    ):
        if type(value) is not float or not np.isfinite(value):
            raise TypeError(f"{name} must be an exact finite float")

    npix = hp.nside2npix(nside)
    theta, phi = hp.pix2ang(nside, np.arange(npix))
    pix_ra_deg = np.degrees(phi)
    pix_dec_deg = 90.0 - np.degrees(theta)

    za_rad, az_rad = radec_to_za_az(
        pix_ra_deg,
        pix_dec_deg,
        zenith_ra_deg=zenith_ra_deg,
        zenith_dec_deg=zenith_dec_deg,
    )
    beam_map = _evaluate_reference_power(
        beam_system=beam_system,
        reference_antenna=reference_antenna,
        zenith_angle_rad=za_rad,
        azimuth_rad=az_rad,
        frequency_hz=frequency_hz,
        time_mjd=time_mjd,
    )
    beam_map = np.array(beam_map, dtype=np.float64, copy=True, order="C")
    beam_map[za_rad > np.pi / 2.0] = 0.0
    if not np.all(np.isfinite(beam_map)):
        raise NonFiniteBeamResponseError(
            "HEALPix beam display response contains non-finite values."
        )
    peak = float(np.max(beam_map))
    if not np.isfinite(peak):
        raise NonFiniteBeamResponseError(
            "HEALPix beam display normalization peak is non-finite."
        )
    if peak <= 0.0:
        raise BeamDisplayNormalizationError(
            "HEALPix beam display has no finite positive normalization domain."
        )
    beam_map /= peak
    beam_map.setflags(write=False)
    return beam_map
