"""Beam-pattern radial analysis: azimuth-averaged profile + null/sidelobe
detection.

Given a HEALPix beam map (e.g. produced by
:func:`rrivis.core.observability.geometry.compute_beam_map_on_healpix`),
``azimuthal_radial_profile`` collapses the 2-D pattern to a 1-D
profile vs zenith angle, and ``detect_beam_features`` extracts the HPBW,
``-10 dB`` radius, and the position/depth of every null and sidelobe
identified by :func:`scipy.signal.find_peaks` on the dB profile.

These routines are pure-array — they do not import the planner or any
visualisation code — so they live in the beam package alongside
``BeamFITSHandler`` and the analytic beam models.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BeamRadialProfile:
    """Azimuth-averaged radial profile of a beam power map (in dB)."""

    za_deg: np.ndarray
    """Zenith-angle bin centres in degrees, shape ``(n_bins,)``."""

    power_db: np.ndarray
    """Mean beam power in each bin, in dB, shape ``(n_bins,)``."""


@dataclass(frozen=True)
class BeamFeatures:
    """Null and sidelobe positions extracted from a radial profile."""

    nulls_za_deg: np.ndarray
    """Zenith angles (degrees) of nulls (local minima of the dB profile)."""

    nulls_power_db: np.ndarray
    """Beam power (dB) at each null."""

    sidelobes_za_deg: np.ndarray
    """Zenith angles (degrees) of sidelobes (local maxima of the dB profile)."""

    sidelobes_power_db: np.ndarray
    """Peak beam power (dB) at each sidelobe."""

    hpbw_deg: float
    """Empirical HPBW: zenith angle where the profile crosses ``-3 dB``."""

    radius_minus10_deg: float
    """Zenith angle where the profile crosses ``-10 dB``."""


def azimuthal_radial_profile(
    beam_map: np.ndarray,
    *,
    nside: int,
    zenith_ra_deg: float,
    zenith_dec_deg: float,
    bin_step_deg: float = 0.25,
    max_za_deg: float = 90.0,
) -> BeamRadialProfile:
    """Bin a HEALPix beam map by zenith angle and return the dB profile.

    Parameters
    ----------
    beam_map
        Dense HEALPix beam power map, shape ``(12 * nside ** 2,)``.  Below-
        horizon pixels should already be zero (the binning still works if
        they are not, but the dB result is dominated by the floor).
    nside
        HEALPix nside of ``beam_map``.
    zenith_ra_deg, zenith_dec_deg
        Pointing centre used to define zenith angle.
    bin_step_deg
        Width of each radial bin in degrees.  ``0.25°`` matches the
        empirical sidelobe-finding cadence used in the EoR notebooks.
    max_za_deg
        Upper edge of the binning range.

    Returns
    -------
    BeamRadialProfile
        ``za_deg`` bin centres and the corresponding mean power in dB.
        Empty bins (no pixels) report ``-inf`` dB.
    """
    import healpy as hp

    from rrivis.utils.coordinates import radec_to_za_az

    npix = hp.nside2npix(nside)
    if beam_map.shape != (npix,):
        raise ValueError(
            f"beam_map has shape {beam_map.shape}; expected ({npix},) for nside={nside}."
        )

    theta, phi = hp.pix2ang(nside, np.arange(npix))
    pix_ra_deg = np.degrees(phi)
    pix_dec_deg = 90.0 - np.degrees(theta)
    za_rad, _ = radec_to_za_az(
        pix_ra_deg,
        pix_dec_deg,
        zenith_ra_deg=zenith_ra_deg,
        zenith_dec_deg=zenith_dec_deg,
    )
    za_deg_all = np.degrees(za_rad)

    mask_up = za_deg_all <= max_za_deg
    za_flat = za_deg_all[mask_up]
    b_flat = np.asarray(beam_map[mask_up], dtype=float)

    bin_edges = np.arange(0.0, max_za_deg + bin_step_deg / 2.0, bin_step_deg)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bin_idx = np.digitize(za_flat, bin_edges) - 1
    keep = (bin_idx >= 0) & (bin_idx < len(bin_centers))

    counts = np.bincount(bin_idx[keep], minlength=len(bin_centers))
    sums = np.bincount(bin_idx[keep], weights=b_flat[keep], minlength=len(bin_centers))
    profile = np.where(counts > 0, sums / np.maximum(counts, 1), 0.0)

    with np.errstate(divide="ignore", invalid="ignore"):
        profile_db = 10.0 * np.log10(np.maximum(profile, 1e-30))
    profile_db[counts == 0] = -np.inf

    return BeamRadialProfile(za_deg=bin_centers, power_db=profile_db)


def detect_beam_features(
    profile: BeamRadialProfile,
    *,
    prominence_db: float = 3.0,
    exclude_inner_deg: float = 5.0,
) -> BeamFeatures:
    """Extract nulls, sidelobes, HPBW, and ``-10 dB`` radius from a profile.

    Nulls are local minima of ``profile.power_db``; sidelobes are local
    maxima.  Both are filtered by a minimum prominence threshold (in dB)
    to avoid noise picks.  Sidelobes within ``exclude_inner_deg`` of the
    pointing centre are dropped — they would otherwise pick up the main
    lobe's plateau if the bin step is small.

    Parameters
    ----------
    profile
        Output of :func:`azimuthal_radial_profile`.
    prominence_db
        Minimum prominence (dB) passed to :func:`scipy.signal.find_peaks`
        for both nulls and sidelobes.
    exclude_inner_deg
        Sidelobes at ``za < exclude_inner_deg`` are discarded.

    Returns
    -------
    BeamFeatures
        Detected feature locations and the empirical HPBW / ``-10 dB``
        radius.
    """
    from scipy.signal import find_peaks

    za = profile.za_deg
    db = profile.power_db

    finite = np.isfinite(db)
    za_finite = za[finite]
    db_finite = db[finite]

    peak_idx, _ = find_peaks(db_finite, prominence=prominence_db)
    null_idx, _ = find_peaks(-db_finite, prominence=prominence_db)

    sidelobe_za = za_finite[peak_idx]
    sidelobe_db = db_finite[peak_idx]
    keep = sidelobe_za > exclude_inner_deg
    sidelobe_za = sidelobe_za[keep]
    sidelobe_db = sidelobe_db[keep]

    null_za = za_finite[null_idx]
    null_db = db_finite[null_idx]

    hpbw = _crossing_za(za_finite, db_finite, level_db=-3.0)
    radius_m10 = _crossing_za(za_finite, db_finite, level_db=-10.0)

    return BeamFeatures(
        nulls_za_deg=null_za,
        nulls_power_db=null_db,
        sidelobes_za_deg=sidelobe_za,
        sidelobes_power_db=sidelobe_db,
        hpbw_deg=hpbw,
        radius_minus10_deg=radius_m10,
    )


def _crossing_za(
    za_deg: np.ndarray,
    profile_db: np.ndarray,
    *,
    level_db: float,
) -> float:
    """Zenith angle of the first crossing of ``level_db`` away from boresight.

    Uses linear interpolation between the two adjacent bins.  Returns
    ``nan`` if the profile never reaches the level within the binned
    range.
    """
    if za_deg.size == 0 or profile_db.size == 0:
        return float("nan")

    below = profile_db <= level_db
    if not np.any(below):
        return float("nan")
    first = int(np.argmax(below))
    if first == 0:
        return float(za_deg[0])

    z0, z1 = za_deg[first - 1], za_deg[first]
    p0, p1 = profile_db[first - 1], profile_db[first]
    if p1 == p0:
        return float(z1)
    frac = (level_db - p0) / (p1 - p0)
    return float(z0 + frac * (z1 - z0))


__all__ = [
    "BeamRadialProfile",
    "BeamFeatures",
    "azimuthal_radial_profile",
    "detect_beam_features",
]
