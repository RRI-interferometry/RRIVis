"""Pure compute functions for sky-model diagnostics.

These functions operate directly on NumPy arrays and HEALPix maps, with no
dependence on :class:`~rrivis.core.sky.SkyModel` or matplotlib. They are used
by :class:`~rrivis.core.sky.plotter.SkyPlotter` analysis methods and are
independently testable.

Conventions
-----------
- Angular power spectrum: computed with ``healpy.anafast`` after optional
  monopole removal via ``healpy.remove_monopole``. Returns ``C_ell``; the
  CMB-style ``D_ell = ell(ell+1) C_ell / (2 pi)`` is a derived quantity.
- Delay spectrum: FFT along the frequency axis per pixel, with a configurable
  window (Blackman-Harris by default, HERA/Parsons 2012 convention). Window
  is RMS-normalised to preserve total power.
- k_parallel conversion: ``k|| = 2 pi tau H(z) f_21 / (c (1+z)^2)``
  (Parsons et al. 2012 eq. 7; Morales & Hewitt 2004 eq. 2).
- Gaussianity: D'Agostino K^2 test (``scipy.stats.normaltest``) — large-N
  safe; Shapiro-Wilk is only reliable for N < 5000 and is skipped above that.
"""

from __future__ import annotations

import logging
from typing import Any

import healpy as hp
import numpy as np

logger = logging.getLogger(__name__)

#: 21 cm rest frequency (Hz)
F_21_HZ = 1_420_405_751.768

#: Speed of light (m/s)
_C_LIGHT = 299_792_458.0

#: Valid window names for :func:`compute_delay_spectrum`
_WINDOW_NAMES = ("blackmanharris", "blackman", "hann", "hamming", "none")


# ---------------------------------------------------------------------------
# Angular power spectrum
# ---------------------------------------------------------------------------


def compute_angular_power_spectrum(
    healpix_map: np.ndarray,
    lmax: int | None = None,
    remove_monopole: bool = True,
) -> np.ndarray:
    """Angular power spectrum ``C_ell`` of a HEALPix map.

    Parameters
    ----------
    healpix_map : np.ndarray
        HEALPix map in RING ordering, shape ``(npix,)``.
    lmax : int, optional
        Maximum multipole.  Defaults to ``3 * nside - 1``.
    remove_monopole : bool
        If ``True`` (default), subtract the monopole before the transform to
        suppress ringing from the unphysical sky mean.

    Returns
    -------
    np.ndarray
        ``C_ell`` of shape ``(lmax + 1,)``.
    """
    m = healpix_map
    if remove_monopole:
        m = hp.remove_monopole(m)
    nside = hp.npix2nside(len(m))
    if lmax is None:
        lmax = 3 * nside - 1
    return hp.anafast(m, lmax=lmax)


def compute_cross_cell(
    map1: np.ndarray,
    map2: np.ndarray,
    lmax: int | None = None,
    remove_monopole: bool = True,
) -> np.ndarray:
    """Cross-power spectrum ``C_ell(nu_1, nu_2)`` of two HEALPix maps.

    Parameters
    ----------
    map1, map2 : np.ndarray
        HEALPix maps (RING), shape ``(npix,)`` each; must be the same nside.
    lmax : int, optional
        Maximum multipole.  Defaults to ``3 * nside - 1``.
    remove_monopole : bool
        Remove monopole from each map before the transform.

    Returns
    -------
    np.ndarray
        Cross-``C_ell`` of shape ``(lmax + 1,)``.
    """
    if len(map1) != len(map2):
        raise ValueError(f"Maps must be same size, got {len(map1)} and {len(map2)}")
    m1, m2 = map1, map2
    if remove_monopole:
        m1 = hp.remove_monopole(m1)
        m2 = hp.remove_monopole(m2)
    nside = hp.npix2nside(len(m1))
    if lmax is None:
        lmax = 3 * nside - 1
    return hp.anafast(m1, m2, lmax=lmax)


def filter_ell_band(
    healpix_map: np.ndarray,
    ell_lo: int,
    ell_hi: int,
    lmax: int | None = None,
) -> np.ndarray:
    """Isolate a range of multipoles in a HEALPix map.

    Decomposes the map into spherical harmonics, zeros all coefficients
    outside ``[ell_lo, ell_hi]``, and reconstructs.

    Parameters
    ----------
    healpix_map : np.ndarray
        HEALPix map (RING), shape ``(npix,)``.
    ell_lo, ell_hi : int
        Inclusive lower and upper multipoles to retain.
    lmax : int, optional
        Max multipole for the transform.  Defaults to ``3 * nside - 1``.

    Returns
    -------
    np.ndarray
        Filtered HEALPix map, same shape as input.
    """
    if ell_lo < 0 or ell_hi < ell_lo:
        raise ValueError(f"Invalid ell band ({ell_lo}, {ell_hi})")
    nside = hp.npix2nside(len(healpix_map))
    if lmax is None:
        lmax = 3 * nside - 1

    alm = hp.map2alm(healpix_map, lmax=lmax)
    # Vectorised mask over ell values for the alm array
    ells, _ = hp.Alm.getlm(lmax)
    mask = (ells >= ell_lo) & (ells <= ell_hi)
    alm_filt = np.where(mask, alm, 0.0 + 0.0j)
    return hp.alm2map(alm_filt, nside, lmax=lmax)


# ---------------------------------------------------------------------------
# Frequency-domain analysis
# ---------------------------------------------------------------------------


def compute_frequency_correlation(
    cube: np.ndarray,
    center: bool = True,
) -> np.ndarray:
    """Pearson correlation matrix across frequency channels.

    Parameters
    ----------
    cube : np.ndarray
        Data cube, shape ``(n_freq, n_pix)``.
    center : bool
        Subtract the per-channel mean before correlating so that the
        monopole does not dominate.  Default ``True``.

    Returns
    -------
    np.ndarray
        Correlation matrix, shape ``(n_freq, n_freq)``.  Diagonal == 1.
    """
    if cube.ndim != 2:
        raise ValueError(f"cube must be 2-D (n_freq, n_pix), got {cube.shape}")
    data = cube
    if center:
        data = cube - np.mean(cube, axis=1, keepdims=True)
    corr = np.corrcoef(data)
    np.fill_diagonal(corr, 1.0)
    return corr


def _get_window(name: str, n: int) -> np.ndarray:
    """Return an RMS-normalised window of length ``n``."""
    from scipy.signal import windows as _sigw

    name = name.lower()
    if name == "none":
        w = np.ones(n)
    elif name == "blackmanharris":
        w = _sigw.blackmanharris(n)
    elif name == "blackman":
        w = _sigw.blackman(n)
    elif name == "hann":
        w = _sigw.hann(n)
    elif name == "hamming":
        w = _sigw.hamming(n)
    else:
        raise ValueError(f"Unknown window '{name}'. Choose from: {_WINDOW_NAMES}")
    # RMS-normalise so mean power is preserved
    w = w / np.sqrt(np.mean(w**2))
    return w


def compute_delay_spectrum(
    cube: np.ndarray,
    frequencies_hz: np.ndarray,
    window: str = "blackmanharris",
    remove_mean: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Delay power spectrum along the frequency axis.

    Parameters
    ----------
    cube : np.ndarray
        Data cube, shape ``(n_freq, n_pix)``.
    frequencies_hz : np.ndarray
        Frequency samples in Hz, shape ``(n_freq,)``.  Uniform spacing
        assumed.
    window : str
        Window function: ``"blackmanharris"`` (default, HERA convention),
        ``"blackman"``, ``"hann"``, ``"hamming"``, or ``"none"``.
    remove_mean : bool
        Subtract the per-pixel mean before FFT to suppress the zero-delay
        monopole.

    Returns
    -------
    tau_s : np.ndarray
        Delay axis in seconds, shape ``(n_freq,)``, ``fftshift``-ordered.
    delay_ps : np.ndarray
        Delay power spectrum averaged over pixels, shape ``(n_freq,)``,
        ``fftshift``-ordered.  Units: ``cube-units^2``.
    """
    if cube.ndim != 2:
        raise ValueError(f"cube must be 2-D (n_freq, n_pix), got {cube.shape}")
    n_freq, n_pix = cube.shape

    if frequencies_hz.shape != (n_freq,):
        raise ValueError(
            f"frequencies_hz shape {frequencies_hz.shape} "
            f"does not match cube first axis {n_freq}"
        )

    dfreq = float(np.diff(frequencies_hz).mean())
    w = _get_window(window, n_freq)

    data = cube
    if remove_mean:
        data = cube - np.mean(cube, axis=0, keepdims=True)

    # Per-pixel FFT then power
    windowed = data * w[:, None]
    ft = np.fft.fft(windowed, axis=0)
    per_pixel_power = np.abs(ft) ** 2 / n_freq

    # Average over pixels, then fftshift
    delay_ps = np.fft.fftshift(np.mean(per_pixel_power, axis=1))
    tau_s = np.fft.fftshift(np.fft.fftfreq(n_freq, dfreq))

    return tau_s, delay_ps


def compute_kparallel(
    tau_s: np.ndarray,
    z: float,
    cosmology: Any | None = None,
) -> np.ndarray:
    """Convert delay (seconds) to line-of-sight wavenumber ``k_parallel``.

    Uses the Parsons et al. (2012) convention::

        k_parallel = 2 * pi * tau * H(z) * f_21 / (c * (1 + z) ^ 2)

    Parameters
    ----------
    tau_s : np.ndarray
        Delay in seconds.
    z : float
        Redshift.  Typically ``z = f_21 / <f_obs> - 1``.
    cosmology : astropy.cosmology.FLRW, optional
        Cosmology for ``H(z)``.  Defaults to Planck15.

    Returns
    -------
    np.ndarray
        ``k_parallel`` in Mpc^-1, same shape as ``tau_s``.
    """
    if cosmology is None:
        from astropy.cosmology import Planck15 as cosmology

    h_kms_mpc = cosmology.H(z).to("km/s/Mpc").value  # km/s/Mpc
    c_kms = _C_LIGHT / 1e3  # km/s
    return (
        2.0 * np.pi * np.asarray(tau_s) * h_kms_mpc * F_21_HZ / (c_kms * (1.0 + z) ** 2)
    )


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def gaussianity_stats(
    values: np.ndarray,
    run_shapiro: bool = False,
    shapiro_n_max: int = 5000,
    seed: int = 42,
) -> dict[str, float]:
    """Summary of how Gaussian a sample is.

    Always reports skewness, excess kurtosis, and the D'Agostino K^2 normality
    test p-value (``scipy.stats.normaltest``), which is reliable for large
    samples.  Optionally runs Shapiro-Wilk on a subsample (the SciPy
    implementation is unreliable for N > 5000).

    Parameters
    ----------
    values : np.ndarray
        Sample values (any shape; flattened internally).
    run_shapiro : bool
        If ``True``, also run Shapiro-Wilk on a subsample of at most
        ``shapiro_n_max`` values.  Default ``False``.
    shapiro_n_max : int
        Maximum N for the Shapiro-Wilk subsample.  Default ``5000``.
    seed : int
        RNG seed for the Shapiro-Wilk subsample.

    Returns
    -------
    dict
        Keys: ``mean``, ``std``, ``skewness``, ``excess_kurtosis``,
        ``normaltest_stat``, ``normaltest_p`` (D'Agostino K^2), and
        optionally ``shapiro_stat``, ``shapiro_p``.
    """
    from scipy import stats as _stats

    x = np.asarray(values).ravel()
    x = x[np.isfinite(x)]
    if x.size < 20:
        raise ValueError(
            f"Need at least 20 finite values for Gaussianity test, got {x.size}"
        )

    out: dict[str, float] = {
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
        "skewness": float(_stats.skew(x)),
        "excess_kurtosis": float(_stats.kurtosis(x)),
    }
    stat, p = _stats.normaltest(x)
    out["normaltest_stat"] = float(stat)
    out["normaltest_p"] = float(p)

    if run_shapiro:
        rng = np.random.default_rng(seed)
        sample = (
            x
            if x.size <= shapiro_n_max
            else rng.choice(x, size=shapiro_n_max, replace=False)
        )
        s_stat, s_p = _stats.shapiro(sample)
        out["shapiro_stat"] = float(s_stat)
        out["shapiro_p"] = float(s_p)

    return out


__all__ = [
    "F_21_HZ",
    "compute_angular_power_spectrum",
    "compute_cross_cell",
    "compute_frequency_correlation",
    "compute_delay_spectrum",
    "compute_kparallel",
    "filter_ell_band",
    "gaussianity_stats",
]
