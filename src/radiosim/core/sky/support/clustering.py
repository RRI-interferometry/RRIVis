"""Angular clustering support for synthetic source populations.

Implements the 2-point angular correlation function (2PACF) machinery of
Mittal et al. 2024 (MNRAS 534, 1317, their eqs. 7-9): a power-law 2PACF

    C(chi) = A * (chi / 1 degree)^-gamma

(default parameters from Rana & Bagla 2019, MNRAS 485, 5891, fitted to
TGSS ADR1 at 150 MHz) is converted to an angular power spectrum ``C_ell``
by Gauss-Legendre quadrature of

    C_ell = 2 pi * integral_{-1}^{1} C(arccos x) P_ell(x) dx,

realized as a Gaussian overdensity map ``delta`` from explicitly seeded
spherical-harmonic coefficients, and turned into per-pixel Poisson rates
``lambda_p = nbar * (1 + delta_p)``.

Deviations from the ``epspy`` reference implementation, by design:

* ``C_ell`` is computed by direct quadrature up to a caller-chosen
  ``lmax`` (typically ``3 * nside - 1``) instead of ``transformcl`` with a
  fixed 200-point grid, so the clustering realization is not silently
  band-limited to ``ell <= 199`` at high resolution.
* The monopole ``C_0`` is zeroed by default: the mean source density is
  fixed by the source-count normalization, and the power-law 2PACF is a
  fitted form whose extrapolation to all angles would otherwise inject a
  random total-count offset on top of the Poisson fluctuation.
* Spherical-harmonic coefficients are drawn from the caller's
  :class:`numpy.random.Generator` (``epspy`` uses healpy's global RNG).
* Pixels with ``1 + delta < 0`` are clipped to zero rate with a logged
  warning (``epspy`` terminates the process); an error is raised only when
  the clipped fraction is large enough to invalidate the Gaussian model.
"""

from __future__ import annotations

import logging

import numpy as np

from .healpy import lazy_healpy as hp

logger = logging.getLogger(__name__)

#: Fraction of sky allowed to fall below zero rate (and be clipped) before
#: the Gaussian overdensity model is declared invalid for the requested
#: 2PACF parameters.
DEFAULT_CLIP_ERROR_FRACTION = 0.25

#: Relative tolerance (vs. the spectrum maximum) for negative quadrature
#: residuals in the ACF -> C_ell transform. Larger negatives indicate the
#: requested 2PACF is not realizable as a nonnegative spectrum at lmax.
_NEGATIVE_CL_RTOL = 1e-4


def power_law_acf_to_cl(
    amplitude: float,
    gamma: float,
    lmax: int,
    *,
    n_quad: int | None = None,
    zero_monopole: bool = True,
) -> np.ndarray:
    """Transform the power-law 2PACF into an angular power spectrum.

    Parameters
    ----------
    amplitude
        2PACF amplitude ``A`` at 1 degree separation. Must be positive.
    gamma
        Power-law exponent; requires ``0 < gamma < 2`` so the quadrature
        integrand stays integrable at zero separation.
    lmax
        Highest multipole to compute (inclusive).
    n_quad
        Gauss-Legendre node count. Defaults to ``max(4096, 2 * (lmax + 1))``
        so the highest requested multipole stays resolved.
    zero_monopole
        Zero out ``C_0`` (default). See the module docstring.

    Returns
    -------
    numpy.ndarray
        ``C_ell`` for ``ell = 0..lmax`` (float64, nonnegative).
    """
    if not np.isfinite(amplitude) or amplitude <= 0.0:
        raise ValueError(f"2PACF amplitude must be positive, got {amplitude!r}.")
    if not np.isfinite(gamma) or not 0.0 < gamma < 2.0:
        raise ValueError(
            f"2PACF exponent gamma must satisfy 0 < gamma < 2, got {gamma!r}."
        )
    if lmax < 1:
        raise ValueError(f"lmax must be at least 1, got {lmax!r}.")

    nodes = n_quad if n_quad is not None else max(4096, 2 * (lmax + 1))
    x, w = np.polynomial.legendre.leggauss(int(nodes))
    chi_deg = np.degrees(np.arccos(np.clip(x, -1.0, 1.0)))
    weighted_acf = w * amplitude * chi_deg**-gamma

    cl = np.empty(lmax + 1, dtype=np.float64)
    p_prev = np.ones_like(x)
    cl[0] = 2.0 * np.pi * np.sum(weighted_acf * p_prev)
    p_cur = x
    if lmax >= 1:
        cl[1] = 2.0 * np.pi * np.sum(weighted_acf * p_cur)
    for ell in range(1, lmax):
        p_next = ((2 * ell + 1) * x * p_cur - ell * p_prev) / (ell + 1)
        cl[ell + 1] = 2.0 * np.pi * np.sum(weighted_acf * p_next)
        p_prev, p_cur = p_cur, p_next

    negative = cl < 0.0
    if np.any(negative):
        worst = float(-cl[negative].max())
        ceiling = float(cl.max())
        if ceiling <= 0.0 or worst > _NEGATIVE_CL_RTOL * ceiling:
            raise ValueError(
                "The requested power-law 2PACF does not transform into a "
                f"nonnegative angular power spectrum at lmax={lmax} "
                f"(most negative C_ell = {-worst:.3e}). Reduce lmax or "
                "revisit the 2PACF parameters."
            )
        cl[negative] = 0.0

    if zero_monopole:
        cl[0] = 0.0
    return cl


def gaussian_overdensity_map(
    cl: np.ndarray,
    nside: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw a Gaussian overdensity map from an angular power spectrum.

    Equivalent to ``healpy.synfast`` but with every random draw taken from
    the supplied :class:`numpy.random.Generator`, so realizations are fully
    reproducible from a recorded seed. The draw order (m = 0, then
    ascending m with ascending ell) is part of the reproducibility
    contract.

    Returns the RING-ordered ``delta`` map (float64, zero mean in
    expectation).
    """
    cl = np.asarray(cl, dtype=np.float64)
    if cl.ndim != 1 or cl.size < 2:
        raise ValueError(f"cl must be a 1-D spectrum with lmax >= 1, got {cl.shape}.")
    if np.any(cl < 0.0) or not np.all(np.isfinite(cl)):
        raise ValueError("cl must be finite and nonnegative.")
    lmax = cl.size - 1

    alm = np.zeros(hp.Alm.getsize(lmax), dtype=np.complex128)
    ells = np.arange(lmax + 1)
    alm[hp.Alm.getidx(lmax, ells, 0)] = rng.normal(size=lmax + 1) * np.sqrt(cl)
    for m in range(1, lmax + 1):
        ells_m = np.arange(m, lmax + 1)
        idx = hp.Alm.getidx(lmax, ells_m, m)
        scale = np.sqrt(cl[ells_m] / 2.0)
        alm[idx] = (
            rng.normal(size=ells_m.size) + 1j * rng.normal(size=ells_m.size)
        ) * scale

    delta = hp.alm2map(alm, nside, lmax=lmax)
    return np.asarray(delta, dtype=np.float64)


def clustered_pixel_rates(
    nbar_per_pixel: float,
    delta: np.ndarray,
    *,
    clip_error_fraction: float = DEFAULT_CLIP_ERROR_FRACTION,
) -> np.ndarray:
    """Per-pixel Poisson rates ``nbar * (1 + delta)`` with clip handling.

    Pixels where ``1 + delta < 0`` are clipped to zero rate with a logged
    warning. When the clipped fraction exceeds ``clip_error_fraction`` the
    Gaussian overdensity model itself is invalid for the requested 2PACF
    parameters and a ``ValueError`` is raised instead.
    """
    if not np.isfinite(nbar_per_pixel) or nbar_per_pixel < 0.0:
        raise ValueError(
            f"nbar_per_pixel must be finite and nonnegative, got {nbar_per_pixel!r}."
        )
    delta = np.asarray(delta, dtype=np.float64)
    rates = nbar_per_pixel * (1.0 + delta)
    negative = rates < 0.0
    n_negative = int(np.count_nonzero(negative))
    if n_negative:
        fraction = n_negative / rates.size
        if fraction > clip_error_fraction:
            raise ValueError(
                f"{fraction:.1%} of pixels have negative source rates for the "
                "requested 2PACF parameters; the Gaussian overdensity model "
                "is not valid this deep into the nonlinear regime. Reduce "
                "the 2PACF amplitude or increase the pixel scale."
            )
        logger.warning(
            "Clustered source rates clipped to zero on %d/%d pixels (%.2f%%).",
            n_negative,
            rates.size,
            100.0 * fraction,
        )
        rates[negative] = 0.0
    return rates


def dither_positions_in_pixels(
    pixels_ring: np.ndarray,
    nside: int,
    rng: np.random.Generator,
    *,
    subdivision_levels: int = 8,
) -> tuple[np.ndarray, np.ndarray]:
    """Draw near-uniform positions inside RING-ordered HEALPix pixels.

    Each source assigned to a pixel is placed at the center of a random
    NESTED child pixel ``subdivision_levels`` deeper (a factor
    ``2**subdivision_levels`` finer per side, i.e. 1/256 of the pixel scale
    by default), which samples the parent pixel area uniformly at child
    granularity. Returns ``(ra_rad, dec_rad)``.
    """
    pixels_ring = np.asarray(pixels_ring)
    if pixels_ring.size == 0:
        empty = np.zeros(0, dtype=np.float64)
        return empty, empty
    if subdivision_levels < 0:
        raise ValueError(
            f"subdivision_levels must be nonnegative, got {subdivision_levels!r}."
        )

    children_per_pixel = 4**subdivision_levels
    nest = hp.ring2nest(nside, pixels_ring).astype(np.int64)
    child = nest * children_per_pixel + rng.integers(
        0, children_per_pixel, size=nest.size, dtype=np.int64
    )
    theta, phi = hp.pix2ang(nside * 2**subdivision_levels, child, nest=True)
    ra_rad = np.asarray(phi, dtype=np.float64)
    dec_rad = np.pi / 2.0 - np.asarray(theta, dtype=np.float64)
    return ra_rad, dec_rad


__all__ = [
    "DEFAULT_CLIP_ERROR_FRACTION",
    "clustered_pixel_rates",
    "dither_positions_in_pixels",
    "gaussian_overdensity_map",
    "power_law_acf_to_cl",
]
