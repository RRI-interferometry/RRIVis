"""Drift-scan lightcurves: integrate sky × beam over LST.

Encapsulates the §10/§11 pattern from the EoR notebooks (eor_multipole_*,
sidelobe_*, zenith_*) where a HEALPix sky cube is multiplied with a
FITS-beam power pattern at each LST, summed, and plotted vs LST.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from .geometry import compute_beam_map_on_healpix

if TYPE_CHECKING:
    from radiosim.core.sky.model import SkyModel


@dataclass(frozen=True)
class DriftScanLightcurve:
    """Drift-scan integration result for one frequency channel."""

    lst_hours: np.ndarray
    """LST samples in hours, shape ``(n_lst,)``."""

    integrated_flux: np.ndarray
    """``Σ(sky × beam)`` per LST, shape ``(n_lst,)``.  Same units as the
    input HEALPix map summed over pixels."""

    mean_brightness: np.ndarray | None
    """``⟨I⟩_beam = Σ(I · B) / Σ(B)`` per LST when ``area_normalize=True``;
    otherwise ``None``."""

    mask_horizon: bool
    """Whether pixels with ``za > 90°`` were zeroed before integration."""

    frequency_hz: float
    """Frequency channel selected from the input ``SkyModel``."""

    nside: int
    """HEALPix nside used for the integration grid."""


def compute_drift_scan_lightcurve(
    sky: SkyModel,
    *,
    latitude_deg: float,
    longitude_deg: float,
    height_m: float,
    beam_fits_path: str,
    beam_diameter_m: float,
    frequency_hz: float,
    lst_hours: np.ndarray,
    mask_horizon: bool = True,
    area_normalize: bool = False,
) -> DriftScanLightcurve:
    """Integrate a HEALPix sky map against a FITS beam at each LST.

    Builds a single :class:`ObservabilityPlanner` to extract the beam's
    ``beam_power_func`` from the FITS file, then for every requested LST
    projects the beam onto a HEALPix grid centred on that LST's zenith and
    sums ``sky × beam``.  The beam pattern itself does not depend on LST —
    only the pointing centre changes — so the FITS file is opened once.

    Parameters
    ----------
    sky
        Input model with a populated HEALPix payload (``sky.healpix``).
    latitude_deg, longitude_deg, height_m
        Site coordinates.  Latitude is also the dec of the zenith.
    beam_fits_path
        Path to a ``pyuvdata``-readable beam FITS file.
    beam_diameter_m
        Dish/aperture diameter in metres (forwarded to the planner for
        consistency; not used when a FITS beam is supplied).
    frequency_hz
        Frequency at which to evaluate the beam and select the sky channel.
        The closest channel in ``sky.healpix.frequencies`` is used.
    lst_hours
        LST samples (hours) at which to evaluate the lightcurve.
    mask_horizon
        Zero pixels with ``za > 90°`` before integration (default
        ``True``).  Set to ``False`` to keep the full sphere — useful
        for diagnosing how much below-horizon flux a beam picks up.
    area_normalize
        Also report ``Σ(I · B) / Σ(B)`` per LST in
        :attr:`DriftScanLightcurve.mean_brightness`.

    Returns
    -------
    DriftScanLightcurve
        Frozen dataclass with ``lst_hours``, ``integrated_flux``, and
        optional ``mean_brightness``.

    Raises
    ------
    ValueError
        If ``sky`` lacks a HEALPix payload, or the beam FITS path cannot
        be loaded.
    """
    if sky.healpix is None:
        raise ValueError(
            "compute_drift_scan_lightcurve requires a SkyModel with a "
            "HEALPix payload; the input has only point sources. Use "
            "radiosim.core.sky.materialize_healpix_model(...) first."
        )

    from .planner import ObservabilityPlanner

    nside = int(sky.healpix.nside)
    healpix = sky.healpix.require_dense("compute_drift_scan_lightcurve")
    freq_idx = sky.healpix.resolve_frequency_index(float(frequency_hz))
    selected_freq_hz = float(healpix.frequencies[freq_idx])
    sky_map = np.asarray(healpix.maps[freq_idx], dtype=float)

    planner = ObservabilityPlanner(
        latitude_deg=latitude_deg,
        longitude_deg=longitude_deg,
        height_m=height_m,
        lst_start_hours=0.0,
        lst_end_hours=0.0,
        frequency_mhz=selected_freq_hz / 1e6,
        beam_diameter_m=beam_diameter_m,
        beam_fits_path=beam_fits_path,
        beam_reference="start",
        footprint_model="swept_beam",
        background_layer="none",
        mode="summary",
    )
    beam_power_func = planner._fits_beam_power_func()

    lsts = np.asarray(lst_hours, dtype=float)
    integrated = np.empty(lsts.shape, dtype=float)
    mean_brightness = np.empty(lsts.shape, dtype=float) if area_normalize else None
    max_za = 90.0 if mask_horizon else 180.0

    for i, lst in enumerate(lsts):
        zenith_ra = float(((lst * 15.0) + 180.0) % 360.0 - 180.0)
        beam_map = compute_beam_map_on_healpix(
            beam_power_func,
            nside=nside,
            zenith_ra_deg=zenith_ra,
            zenith_dec_deg=float(latitude_deg),
            max_za_deg=max_za,
            peak_normalize=True,
        )
        product = sky_map * beam_map
        integrated[i] = float(np.sum(product))
        if mean_brightness is not None:
            denom = float(np.sum(beam_map))
            mean_brightness[i] = (
                float(np.sum(product) / denom) if denom > 0.0 else float("nan")
            )

    return DriftScanLightcurve(
        lst_hours=lsts,
        integrated_flux=integrated,
        mean_brightness=mean_brightness,
        mask_horizon=mask_horizon,
        frequency_hz=selected_freq_hz,
        nside=nside,
    )


def fractional_horizon_excess(
    masked: DriftScanLightcurve,
    unmasked: DriftScanLightcurve,
) -> np.ndarray:
    """Per-LST relative excess from keeping below-horizon pixels.

    Returns ``(unmasked.integrated_flux - masked.integrated_flux) /
    masked.integrated_flux``.  Both lightcurves must be sampled at the
    same LST grid and the same frequency.

    Raises
    ------
    ValueError
        If the two lightcurves disagree on LST grid or frequency.
    """
    if masked.lst_hours.shape != unmasked.lst_hours.shape or not np.allclose(
        masked.lst_hours, unmasked.lst_hours
    ):
        raise ValueError(
            "fractional_horizon_excess requires both lightcurves to share "
            "the same LST grid."
        )
    if masked.frequency_hz != unmasked.frequency_hz:
        raise ValueError(
            "fractional_horizon_excess requires both lightcurves at the "
            f"same frequency; got {masked.frequency_hz} Hz vs "
            f"{unmasked.frequency_hz} Hz."
        )

    masked_flux = masked.integrated_flux
    with np.errstate(divide="ignore", invalid="ignore"):
        excess = (unmasked.integrated_flux - masked_flux) / masked_flux
    excess[masked_flux == 0.0] = np.nan
    return excess


__all__ = [
    "DriftScanLightcurve",
    "compute_drift_scan_lightcurve",
    "fractional_horizon_excess",
]
