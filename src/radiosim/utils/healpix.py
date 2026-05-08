"""HEALPix resolution advisors.

Small utilities that help users pick a HEALPix ``nside`` whose pixel size
is adequate for a given beam FWHM.  Rule of thumb: pixel resolution should
be at most ``beam_fwhm / safety_factor`` (default ``safety_factor=5``) so
that individual sources are not blurred across multiple pixels and the
visibility integral is well sampled.
"""

from __future__ import annotations

import healpy as hp
import numpy as np

__all__ = [
    "recommend_nside_for_beam",
    "pixel_too_coarse",
]


def recommend_nside_for_beam(
    beam_fwhm_rad: float,
    safety_factor: float = 5.0,
) -> int:
    """Return the smallest power-of-two ``nside`` whose pixel scale is at most
    ``beam_fwhm_rad / safety_factor`` radians.

    Parameters
    ----------
    beam_fwhm_rad
        Primary-beam FWHM (radians).
    safety_factor
        Target ratio ``beam_fwhm / pixel_resolution``.  Default 5 mirrors
        the widely-used "five pixels across the beam" rule.

    Returns
    -------
    int
        The smallest power-of-two ``nside`` satisfying
        ``hp.nside2resol(nside) ≤ beam_fwhm_rad / safety_factor``.

    Raises
    ------
    ValueError
        If ``beam_fwhm_rad`` or ``safety_factor`` is non-positive.

    Examples
    --------
    >>> recommend_nside_for_beam(np.deg2rad(1.0))  # 1 deg beam
    128
    >>> recommend_nside_for_beam(np.deg2rad(1.0 / 60.0))  # 1 arcmin beam
    8192
    """
    if not np.isfinite(beam_fwhm_rad) or beam_fwhm_rad <= 0.0:
        raise ValueError(
            f"beam_fwhm_rad must be a positive finite number, got {beam_fwhm_rad!r}."
        )
    if not np.isfinite(safety_factor) or safety_factor <= 0.0:
        raise ValueError(
            f"safety_factor must be a positive finite number, got {safety_factor!r}."
        )

    target_resol = float(beam_fwhm_rad) / float(safety_factor)
    nside = 1
    # hp.nside2resol(1) ≈ 0.841 rad; bump until pixel ≤ target_resol.
    max_nside = 1 << 16  # 65536 — well beyond any realistic simulation.
    while nside < max_nside and hp.nside2resol(nside) > target_resol:
        nside <<= 1
    return int(nside)


def pixel_too_coarse(
    nside: int,
    beam_fwhm_rad: float,
    safety_factor: float = 5.0,
) -> bool:
    """Return True if the pixel size at ``nside`` exceeds
    ``beam_fwhm_rad / safety_factor``.

    Parameters
    ----------
    nside
        HEALPix nside to test.  Must be a positive power of two.
    beam_fwhm_rad
        Primary-beam FWHM (radians).  Non-positive values short-circuit to
        ``False`` (advisor disabled).
    safety_factor
        Target ratio of beam FWHM to pixel scale.  Default 5.

    Returns
    -------
    bool
    """
    if beam_fwhm_rad is None or not np.isfinite(beam_fwhm_rad) or beam_fwhm_rad <= 0.0:
        return False
    if safety_factor <= 0.0:
        return False
    return bool(
        hp.nside2resol(int(nside)) > float(beam_fwhm_rad) / float(safety_factor)
    )
