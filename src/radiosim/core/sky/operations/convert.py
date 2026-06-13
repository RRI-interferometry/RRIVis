# radiosim/core/sky/convert.py
"""Bidirectional conversion between point-source and HEALPix representations.

Pure functions that accept and return raw numpy arrays. No SkyModel dependency.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import healpy as hp
import numpy as np

from ..containers import empty_source_arrays as _empty_source_arrays
from ..containers.constants import (
    brightness_temp_to_flux_density,
    flux_density_to_brightness_temp,
    rayleigh_jeans_factor,
)
from ..containers.footprint import _normalize_coordinate_frame
from ..containers.spectral import (
    apply_faraday_rotation,
    compute_spectral_scale,
    nearest_channel_index_with_warning,
)
from ..diagnostics.discovery import estimate_healpix_memory
from ..support.allocation import allocate_cube, ensure_scratch_dir, finalize_cube

if TYPE_CHECKING:
    from radiosim.backends import ArrayBackend

logger = logging.getLogger(__name__)


def _fit_pixel_spectral_indices(
    healpix_maps: np.ndarray,
    observation_frequencies: np.ndarray,
    valid_idx: np.ndarray,
    ref_frequency: float,
    omega: float,
    brightness_conversion: str,
) -> np.ndarray:
    """Fit per-pixel spectral indices from multi-frequency HEALPix maps.

    Uses vectorized log-log linear regression:
    ``log10(S) = log10(S_ref) + alpha * log10(f / f_ref)``

    Parameters
    ----------
    healpix_maps : np.ndarray
        Stokes I brightness temperature cube, shape ``(n_freq, npix)``.
    observation_frequencies : np.ndarray
        Frequency array in Hz, shape ``(n_freq,)``.
    valid_idx : np.ndarray
        Indices of pixels with positive flux at the reference frequency.
    ref_frequency : float
        Reference frequency in Hz.
    omega : float
        Pixel solid angle in steradians.
    brightness_conversion : str
        ``"planck"`` or ``"rayleigh-jeans"``.

    Returns
    -------
    np.ndarray
        Fitted spectral indices, shape ``(n_valid,)``.
    """
    n_freq = len(observation_frequencies)
    n_valid = len(valid_idx)

    # Convert T_b -> flux density (Jy) at each frequency for valid pixels
    # flux_matrix shape: (n_freq, n_valid)
    flux_matrix = np.zeros((n_freq, n_valid), dtype=np.float64)
    for fi, freq in enumerate(observation_frequencies):
        t_vals = healpix_maps[fi][valid_idx].astype(np.float64)
        pos = t_vals > 0
        if np.any(pos):
            flux_matrix[fi, pos] = brightness_temp_to_flux_density(
                t_vals[pos], float(freq), omega, method=brightness_conversion
            )

    # Build mask: pixel must have positive flux at >=2 frequencies to fit
    positive_mask = flux_matrix > 0  # (n_freq, n_valid)
    n_positive = positive_mask.sum(axis=0)  # (n_valid,)
    fittable = n_positive >= 2

    alpha = np.zeros(n_valid, dtype=np.float64)

    if not np.any(fittable):
        return alpha

    # Vectorized log-log linear regression for fittable pixels
    log_ratio = np.log10(observation_frequencies / ref_frequency)  # (n_freq,)

    # For pixels where all frequencies are valid, use the fast path
    all_valid = n_positive == n_freq
    if np.any(all_valid):
        log_S = np.log10(flux_matrix[:, all_valid])  # (n_freq, n_all_valid)
        x = log_ratio
        N = n_freq
        sum_x = np.sum(x)
        sum_x2 = np.sum(x**2)
        sum_y = np.sum(log_S, axis=0)  # (n_all_valid,)
        sum_xy = np.sum(x[:, None] * log_S, axis=0)  # (n_all_valid,)
        denom = N * sum_x2 - sum_x**2
        if abs(denom) > 1e-30:
            alpha[all_valid] = (N * sum_xy - sum_x * sum_y) / denom

    # For pixels with partial frequency coverage, use masked vectorized sums.
    partial = fittable & ~all_valid
    if np.any(partial):
        masked_flux = np.where(positive_mask[:, partial], flux_matrix[:, partial], 1.0)
        log_s = np.where(positive_mask[:, partial], np.log10(masked_flux), 0.0)
        valid = positive_mask[:, partial].astype(np.float64)
        x = log_ratio[:, None]
        n_fit = valid.sum(axis=0)
        sum_x = (x * valid).sum(axis=0)
        sum_x2 = ((x**2) * valid).sum(axis=0)
        sum_y = log_s.sum(axis=0)
        sum_xy = (x * log_s).sum(axis=0)
        denom = n_fit * sum_x2 - sum_x**2
        good = np.abs(denom) > 1e-30
        partial_alpha = np.zeros(int(partial.sum()), dtype=np.float64)
        partial_alpha[good] = (
            n_fit[good] * sum_xy[good] - sum_x[good] * sum_y[good]
        ) / denom[good]
        alpha[partial] = partial_alpha

    # Clamp non-finite values
    bad = ~np.isfinite(alpha)
    if np.any(bad):
        alpha[bad] = 0.0
        logger.warning(
            "Clamped %d non-finite fitted spectral indices to 0.",
            int(bad.sum()),
        )

    return alpha


def healpix_map_to_point_arrays(
    temp_map: np.ndarray,
    frequency: float,
    brightness_conversion: str,
    healpix_q_maps: np.ndarray | None = None,
    healpix_u_maps: np.ndarray | None = None,
    healpix_v_maps: np.ndarray | None = None,
    observation_frequencies: np.ndarray | None = None,
    freq_index: int | None = None,
    healpix_maps: np.ndarray | None = None,
    coordinate_frame: str = "icrs",
    ref_freq_out: float | None = None,
    *,
    nest: bool = False,
    polarization_brightness_conversion: str = "rayleigh-jeans",
    warn: bool = True,
    backend: ArrayBackend | None = None,
) -> dict[str, np.ndarray]:
    """Convert a HEALPix brightness temperature map to columnar point-source arrays.

    Only positive-temperature pixels are stored (no flux_limit filtering
    here -- apply that in the caller).

    When ``healpix_maps`` (full multi-frequency Stokes I cube) and
    ``observation_frequencies`` (with ≥2 entries) are provided, a per-pixel
    spectral index is fitted via log-log linear regression of flux density
    vs. frequency.  Otherwise all pixels receive ``alpha=0``.

    .. note::

       **Brightness-conversion asymmetry between I and Q/U/V.**  Stokes I
       is converted with the user-selected ``brightness_conversion`` method
       (``"planck"`` is the natural choice for non-negative intensities at
       all frequencies).  Stokes Q/U/V are converted with
       ``polarization_brightness_conversion`` which defaults to
       ``"rayleigh-jeans"`` because polarized brightness can legitimately
       be negative and the Planck inverse is undefined for non-positive
       arguments.  At ``ν ≲ 100 MHz`` the Planck and Rayleigh-Jeans
       conversions disagree at the 5–15% level; a polarized HEALPix→point
       round-trip with the default polarization setting therefore inflates
       fractional polarization by roughly that fraction.  Pass
       ``polarization_brightness_conversion="planck"`` to use the same
       method for I and Q/U/V; this requires every Q/U/V pixel to be
       strictly positive and will ``ValueError`` otherwise.

    Parameters
    ----------
    temp_map : np.ndarray
        Brightness temperature map in Kelvin (reference frequency slice).
    frequency : float
        Frequency in Hz for T_b -> Jy conversion (reference frequency).
    brightness_conversion : str
        Conversion method for Stokes I: ``"planck"`` or ``"rayleigh-jeans"``.
    healpix_q_maps : np.ndarray or None
        Stokes Q maps, shape ``(n_freq, npix)`` or None.
    healpix_u_maps : np.ndarray or None
        Stokes U maps, shape ``(n_freq, npix)`` or None.
    healpix_v_maps : np.ndarray or None
        Stokes V maps, shape ``(n_freq, npix)`` or None.
    observation_frequencies : np.ndarray or None
        Frequency array in Hz, used to find freq_index if not given.
    freq_index : int or None
        Index into the (n_freq, npix) polarization arrays. If None and
        ``observation_frequencies`` is provided, the nearest index is found.
    healpix_maps : np.ndarray or None
        Full Stokes I brightness temperature cube, shape ``(n_freq, npix)``.
        When provided together with ``observation_frequencies`` (≥2 channels),
        enables per-pixel spectral index fitting.
    coordinate_frame : {"icrs", "galactic"}, default "icrs"
        Coordinate frame of the input HEALPix pixel indexing. Returned point
        coordinates are always ICRS.
    ref_freq_out : float or None
        Reference frequency stored in the output ``ref_freq`` array.
        Defaults to ``frequency`` if not given.
    polarization_brightness_conversion : {"rayleigh-jeans", "planck"}, default "rayleigh-jeans"
        Brightness-conversion method for Stokes Q/U/V.  See the note above.

    Returns
    -------
    dict
        TypedDict with keys: ``"ra_rad"``, ``"dec_rad"``, ``"flux"``,
        ``"spectral_index"``, ``"ref_freq"``, ``"stokes_q"``, ``"stokes_u"``,
        ``"stokes_v"``.
        All arrays have shape ``(N,)`` where N is the number of valid pixels.
    """
    pol_method = str(polarization_brightness_conversion).lower()
    if pol_method not in {"rayleigh-jeans", "planck"}:
        raise ValueError(
            "polarization_brightness_conversion must be 'rayleigh-jeans' or "
            f"'planck', got {polarization_brightness_conversion!r}."
        )
    npix = len(temp_map)
    nside = hp.npix2nside(npix)
    omega = 4 * np.pi / npix
    ref_freq_val = ref_freq_out if ref_freq_out is not None else frequency
    frame = _normalize_coordinate_frame(coordinate_frame)

    flux_jy = np.zeros(npix, dtype=np.float64)
    pos = temp_map > 0
    if np.any(pos):
        flux_jy[pos] = brightness_temp_to_flux_density(
            temp_map[pos].astype(np.float64),
            frequency,
            omega,
            method=brightness_conversion,
        )

    valid_idx = np.where(flux_jy > 0)[0]
    if len(valid_idx) == 0:
        logger.warning("No pixels with positive flux in HEALPix map")
        return _empty_source_arrays()

    theta, phi = hp.pix2ang(nside, valid_idx, nest=nest)
    lat_rad = np.pi / 2 - theta
    if frame == "galactic":
        from astropy.coordinates import SkyCoord

        icrs = SkyCoord(l=phi, b=lat_rad, unit="rad", frame="galactic").icrs
        ra_rad = icrs.ra.rad
        dec_rad = icrs.dec.rad
    else:
        ra_rad = phi
        dec_rad = lat_rad
    flux_ref = flux_jy[valid_idx]
    n = len(valid_idx)

    # --- Per-pixel spectral index fitting ---
    can_fit = (
        healpix_maps is not None
        and observation_frequencies is not None
        and len(observation_frequencies) >= 2
    )

    if can_fit:
        alpha = _fit_pixel_spectral_indices(
            healpix_maps,
            observation_frequencies,
            valid_idx,
            frequency,
            omega,
            brightness_conversion,
        )
        logger.info(
            "Fitted per-pixel spectral indices from %d frequency channels "
            "(%.1f\u2013%.1f MHz). Median alpha=%.3f.",
            len(observation_frequencies),
            observation_frequencies[0] / 1e6,
            observation_frequencies[-1] / 1e6,
            float(np.median(alpha)),
        )
    else:
        alpha = np.zeros(n, dtype=np.float64)
        n_freq = (
            len(observation_frequencies) if observation_frequencies is not None else 0
        )
        if warn and n_freq <= 1:
            logger.warning(
                "Only %d frequency channel available \u2014 cannot fit spectral "
                "index. All pixels assigned alpha=0 (flat spectrum). For "
                "accurate multi-frequency results, use 'healpix_map' "
                "representation directly.",
                max(n_freq, 1),
            )
        elif warn:
            logger.warning(
                "HEALPix-to-point-source conversion assigns alpha=0 (flat "
                "spectrum) to all pixels. For accurate multi-frequency "
                "results, use 'healpix_map' representation directly.",
            )

    # Position quantization warning
    resol_arcmin = np.degrees(hp.nside2resol(nside)) * 60
    if warn:
        logger.warning(
            "HEALPix-to-point-source conversion: source positions are quantized "
            "to pixel centers (nside=%d, angular resolution ~%.2f arcmin). "
            "Sub-pixel positions from the original catalog are lost.",
            nside,
            resol_arcmin,
        )

    # Stokes Q/U/V conversion: by default Rayleigh-Jeans (linear,
    # sign-preserving) because polarized brightness can legitimately be
    # negative and the inverse Planck law is undefined for non-positive
    # arguments.  Override with polarization_brightness_conversion="planck"
    # to use the same non-linear conversion as Stokes I — only valid when
    # every Q/U/V pixel is strictly positive (e.g. Stokes-I-only maps in
    # which Q/U/V are not actually polarized observables).
    fi = freq_index
    if fi is None and observation_frequencies is not None:
        fi = int(np.argmin(np.abs(observation_frequencies - frequency)))

    def _convert_pol_slice(temp_slice: np.ndarray, name: str) -> np.ndarray:
        if pol_method == "planck":
            if np.any(temp_slice <= 0):
                raise ValueError(
                    "polarization_brightness_conversion='planck' requires "
                    f"strictly positive {name} values; got values <= 0 (Planck "
                    "is undefined for non-positive arguments).  Use "
                    "'rayleigh-jeans' for sign-preserving linear conversion."
                )
            return brightness_temp_to_flux_density(
                temp_slice, frequency, omega, method="planck"
            )
        return temp_slice * rayleigh_jeans_factor(frequency, omega)

    if healpix_q_maps is not None and fi is not None:
        stokes_q = _convert_pol_slice(
            healpix_q_maps[fi][valid_idx].astype(np.float64), "Stokes Q"
        )
    else:
        stokes_q = np.zeros(n, dtype=np.float64)

    if healpix_u_maps is not None and fi is not None:
        stokes_u = _convert_pol_slice(
            healpix_u_maps[fi][valid_idx].astype(np.float64), "Stokes U"
        )
    else:
        stokes_u = np.zeros(n, dtype=np.float64)

    if healpix_v_maps is not None and fi is not None:
        stokes_v = _convert_pol_slice(
            healpix_v_maps[fi][valid_idx].astype(np.float64), "Stokes V"
        )
    else:
        stokes_v = np.zeros(n, dtype=np.float64)

    return {
        "ra_rad": ra_rad,
        "dec_rad": dec_rad,
        "flux": flux_ref,
        "spectral_index": alpha,
        "ref_freq": np.full(n, ref_freq_val, dtype=np.float64),
        "stokes_q": stokes_q,
        "stokes_u": stokes_u,
        "stokes_v": stokes_v,
        "rotation_measure": None,
        "major_arcsec": None,
        "minor_arcsec": None,
        "pa_deg": None,
        "spectral_coeffs": None,
        "per_channel_flux": None,
        "per_channel_stokes_q": None,
        "per_channel_stokes_u": None,
        "per_channel_stokes_v": None,
        "channel_frequencies": None,
    }


def bin_sources_to_flux(
    ipix: np.ndarray,
    flux: np.ndarray,
    spectral_index: np.ndarray,
    spectral_coeffs: np.ndarray | None,
    freq: float,
    ref_frequency: float | np.ndarray,
    npix: int,
    *,
    scale: np.ndarray | None = None,
    per_channel_flux: np.ndarray | None = None,
    channel_frequencies: np.ndarray | None = None,
    backend: ArrayBackend | None = None,
) -> np.ndarray:
    """Bin point sources into a HEALPix flux density map at a given frequency.

    Computes the spectral scaling factor for each source and accumulates
    the scaled flux into HEALPix pixels via ``np.bincount``.

    Parameters
    ----------
    ipix : np.ndarray
        HEALPix pixel index for each source, shape ``(N_sources,)``.
    flux : np.ndarray
        Reference flux density in Jy, shape ``(N_sources,)``.
    spectral_index : np.ndarray
        Spectral index, shape ``(N_sources,)``.
    spectral_coeffs : np.ndarray or None
        Log-polynomial coefficients, shape ``(N_sources, N_terms)``.
    freq : float
        Observation frequency in Hz.
    ref_frequency : float or np.ndarray
        Reference frequency in Hz (scalar or per-source).
    npix : int
        Total number of HEALPix pixels.
    scale : np.ndarray or None, optional
        Pre-computed spectral scale factor, shape ``(N_sources,)``.
        When provided, skips redundant ``compute_spectral_scale`` call.

    Returns
    -------
    np.ndarray
        Flux density map in Jy, shape ``(npix,)``.
    """
    xp = np if backend is None else backend.xp
    if per_channel_flux is not None and channel_frequencies is not None:
        idx = nearest_channel_index_with_warning(
            channel_frequencies, freq, label="per-channel flux binning"
        )
        flux_f = xp.asarray(per_channel_flux[idx], dtype=np.float64)
    else:
        if scale is None:
            scale = compute_spectral_scale(
                spectral_index, spectral_coeffs, freq, ref_frequency, xp=xp
            )
        flux_f = xp.asarray(flux, dtype=np.float64) * scale
    if backend is None:
        return np.bincount(ipix, weights=flux_f, minlength=npix)
    return backend.bincount(ipix, weights=flux_f, minlength=npix)


def point_sources_to_healpix_maps(
    ra_rad: np.ndarray,
    dec_rad: np.ndarray,
    flux: np.ndarray,
    spectral_index: np.ndarray,
    spectral_coeffs: np.ndarray | None,
    stokes_q: np.ndarray | None,
    stokes_u: np.ndarray | None,
    stokes_v: np.ndarray | None,
    rotation_measure: np.ndarray | None,
    nside: int,
    frequencies: np.ndarray,
    ref_frequency: float | np.ndarray,
    brightness_conversion: str,
    coordinate_frame: str = "icrs",
    output_dtype: np.dtype = np.float32,
    memmap_path: str | None = None,
    per_channel_flux: np.ndarray | None = None,
    per_channel_stokes_q: np.ndarray | None = None,
    per_channel_stokes_u: np.ndarray | None = None,
    per_channel_stokes_v: np.ndarray | None = None,
    channel_frequencies: np.ndarray | None = None,
    *,
    polarization_brightness_conversion: str = "rayleigh-jeans",
    backend: ArrayBackend | None = None,
) -> tuple[
    np.ndarray,
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray | None,
    dict[str, int],
]:
    """Convert point sources to multi-frequency HEALPix brightness temperature maps.

    Vectorized implementation: uses ``np.bincount`` for O(N_sources) memory per
    frequency channel instead of a Python loop over sources.

    Parameters
    ----------
    ra_rad, dec_rad : np.ndarray
        Source coordinates in radians, shape ``(N_sources,)``.
    flux : np.ndarray
        Reference flux density in Jy, shape ``(N_sources,)``.
    spectral_index : np.ndarray
        Spectral index, shape ``(N_sources,)``.
    spectral_coeffs : np.ndarray or None
        Log-polynomial coefficients, shape ``(N_sources, N_terms)``.
    stokes_q, stokes_u, stokes_v : np.ndarray or None
        Stokes polarization, shape ``(N_sources,)``.
    rotation_measure : np.ndarray or None
        Rotation measure in rad/m^2, shape ``(N_sources,)``.
    nside : int
        HEALPix NSIDE parameter.
    frequencies : np.ndarray
        Array of frequencies in Hz.
    ref_frequency : float or np.ndarray
        Reference frequency in Hz. Can be a scalar (shared by all sources)
        or a per-source array of shape ``(N_sources,)`` for correct spectral
        scaling of sources from different catalogs.
    brightness_conversion : str
        ``"planck"`` or ``"rayleigh-jeans"``.
    coordinate_frame : {"icrs", "galactic"}, default "icrs"
        Coordinate frame of the target HEALPix pixel indexing. Input point
        coordinates are always interpreted as ICRS.
    output_dtype : np.dtype, default np.float32
        Dtype for output HEALPix arrays. Use ``precision.sky_model.get_dtype("healpix_maps")``
        to respect the user's precision configuration.

    .. note::

       **Brightness-conversion asymmetry between I and Q/U/V.**  Stokes I
       uses ``brightness_conversion`` (Planck-by-default for non-negative
       intensities).  Stokes Q/U/V use
       ``polarization_brightness_conversion`` which defaults to
       ``"rayleigh-jeans"`` because Q/U/V can be negative and the inverse
       Planck law is undefined for non-positive arguments.  At
       ``ν ≲ 100 MHz`` the two conventions differ by 5–15%; a polarized
       point→HEALPix→point round-trip with the default polarization setting
       therefore changes fractional polarization by roughly that amount.
       Pass ``polarization_brightness_conversion="planck"`` to use the same
       conversion for I and Q/U/V; this requires every binned Q/U/V
       per-pixel sum to be strictly positive and will ``ValueError``
       otherwise.

    Returns
    -------
    i_maps : np.ndarray
        Stokes I brightness temperature maps, shape ``(n_freq, npix)``.
    q_maps : np.ndarray or None
        Stokes Q maps (K), shape ``(n_freq, npix)``, or None.
    u_maps : np.ndarray or None
        Stokes U maps (K), shape ``(n_freq, npix)``, or None.
    v_maps : np.ndarray or None
        Stokes V maps (K), shape ``(n_freq, npix)``, or None.
    collision_stats : dict
        ``{"n_sources": int, "n_collisions": int, "n_merged": int}`` —
        number of pixels that received multiple sources (``n_collisions``)
        and the total source count whose identities were merged
        (``n_merged``).  Both are ``0`` when every source landed in its
        own pixel.  Useful for downstream provenance tagging.
    """
    pol_method = str(polarization_brightness_conversion).lower()
    if pol_method not in {"rayleigh-jeans", "planck"}:
        raise ValueError(
            "polarization_brightness_conversion must be 'rayleigh-jeans' or "
            f"'planck', got {polarization_brightness_conversion!r}."
        )
    xp = np if backend is None else backend.xp
    to_numpy = np.asarray if backend is None else backend.to_numpy

    npix = hp.nside2npix(nside)
    n_freq = len(frequencies)
    n_sources = len(ra_rad)

    if n_sources == 0:
        empty = np.zeros((n_freq, npix), dtype=output_dtype)
        return (
            empty,
            None,
            None,
            None,
            {
                "n_sources": 0,
                "n_collisions": 0,
                "n_merged": 0,
            },
        )

    omega_pixel = 4 * np.pi / npix
    frame = _normalize_coordinate_frame(coordinate_frame)

    # Check if any source has non-zero polarization
    has_pol = (
        stokes_q is not None
        and stokes_u is not None
        and stokes_v is not None
        and (np.any(stokes_q != 0) or np.any(stokes_u != 0) or np.any(stokes_v != 0))
    )

    n_stokes = 4 if has_pol else 1
    mem_info = estimate_healpix_memory(nside, n_freq, output_dtype, n_stokes)
    logger.info(
        f"Creating {n_freq} HEALPix maps (nside={nside}, "
        f"stokes={'IQUV' if has_pol else 'I'}): "
        f"~{mem_info['total_mb']:.1f} MB"
    )

    if frame == "galactic":
        from astropy.coordinates import SkyCoord

        galactic = SkyCoord(ra=ra_rad, dec=dec_rad, unit="rad", frame="icrs").galactic
        lon_rad = galactic.l.rad
        lat_rad = galactic.b.rad
    else:
        lon_rad = ra_rad
        lat_rad = dec_rad

    ipix = hp.ang2pix(nside, np.pi / 2 - lat_rad, lon_rad)

    # Detect pixel collisions (multiple sources in one pixel) and capture
    # counts so the caller can record them in SkyProvenance.notes.
    _unique_pixels, _counts = np.unique(ipix, return_counts=True)
    _multi = _counts > 1
    n_collisions = int(np.sum(_multi))
    n_merged = int(np.sum(_counts[_multi])) if n_collisions else 0
    if n_collisions:
        logger.warning(
            "HEALPix pixelization: %d sources were merged into %d pixels "
            "(out of %d total sources). Individual source identities and "
            "per-source spectral indices are irreversibly combined. "
            "Increase nside (currently %d) to reduce merging.",
            n_merged,
            n_collisions,
            n_sources,
            nside,
        )
    collision_stats: dict[str, int] = {
        "n_sources": int(n_sources),
        "n_collisions": n_collisions,
        "n_merged": n_merged,
    }

    scratch = ensure_scratch_dir(memmap_path) if memmap_path is not None else None
    i_arr = allocate_cube((n_freq, npix), output_dtype, scratch, "i_maps")
    q_arr = (
        allocate_cube((n_freq, npix), output_dtype, scratch, "q_maps")
        if has_pol
        else None
    )
    u_arr = (
        allocate_cube((n_freq, npix), output_dtype, scratch, "u_maps")
        if has_pol
        else None
    )
    v_arr = (
        allocate_cube((n_freq, npix), output_dtype, scratch, "v_maps")
        if has_pol
        else None
    )

    use_per_channel = per_channel_flux is not None and channel_frequencies is not None
    flux_backend = xp.asarray(flux, dtype=np.float64)
    spectral_index_backend = xp.asarray(spectral_index, dtype=np.float64)
    spectral_coeffs_backend = (
        None
        if spectral_coeffs is None
        else xp.asarray(spectral_coeffs, dtype=np.float64)
    )
    ref_frequency_backend = xp.asarray(ref_frequency, dtype=np.float64)
    stokes_q_backend = (
        None if stokes_q is None else xp.asarray(stokes_q, dtype=np.float64)
    )
    stokes_u_backend = (
        None if stokes_u is None else xp.asarray(stokes_u, dtype=np.float64)
    )
    stokes_v_backend = (
        None if stokes_v is None else xp.asarray(stokes_v, dtype=np.float64)
    )
    rotation_measure_backend = (
        None
        if rotation_measure is None
        else xp.asarray(rotation_measure, dtype=np.float64)
    )
    per_channel_flux_backend = (
        None
        if per_channel_flux is None
        else xp.asarray(per_channel_flux, dtype=np.float64)
    )
    per_channel_stokes_q_backend = (
        None
        if per_channel_stokes_q is None
        else xp.asarray(per_channel_stokes_q, dtype=np.float64)
    )
    per_channel_stokes_u_backend = (
        None
        if per_channel_stokes_u is None
        else xp.asarray(per_channel_stokes_u, dtype=np.float64)
    )
    per_channel_stokes_v_backend = (
        None
        if per_channel_stokes_v is None
        else xp.asarray(per_channel_stokes_v, dtype=np.float64)
    )

    # Defined once (was redefined every loop iteration): convert a per-pixel
    # Stokes Q/U/V flux map (Jy) to brightness temperature (K) at one channel.
    def _pol_flux_to_K(
        flux_map: np.ndarray, name: str, freq_hz: float, rj_inv: float
    ) -> np.ndarray:
        flux_map_np = to_numpy(flux_map)
        if pol_method == "planck":
            if np.any(flux_map_np <= 0):
                raise ValueError(
                    "polarization_brightness_conversion='planck' requires "
                    f"strictly positive {name} flux per pixel; got values <= 0 "
                    f"after binning at {freq_hz / 1e6:.3f} MHz. Use "
                    "'rayleigh-jeans' for sign-preserving linear conversion."
                )
            return flux_density_to_brightness_temp(
                flux_map_np, freq_hz, omega_pixel, method="planck"
            )
        return flux_map_np * rj_inv

    for fi, freq in enumerate(frequencies):
        if use_per_channel:
            scale = None  # unused on per-channel path
            flux_map = bin_sources_to_flux(
                ipix,
                flux_backend,
                spectral_index_backend,
                spectral_coeffs_backend,
                float(freq),
                ref_frequency_backend,
                npix,
                per_channel_flux=per_channel_flux_backend,
                channel_frequencies=channel_frequencies,
                backend=backend,
            )
        else:
            scale = compute_spectral_scale(
                spectral_index_backend,
                spectral_coeffs_backend,
                float(freq),
                ref_frequency_backend,
                xp=xp,
            )
            flux_map = bin_sources_to_flux(
                ipix,
                flux_backend,
                spectral_index_backend,
                spectral_coeffs_backend,
                float(freq),
                ref_frequency_backend,
                npix,
                scale=scale,
                backend=backend,
            )

        temp_out = np.zeros(npix, dtype=output_dtype)
        flux_map_np = to_numpy(flux_map)
        occupied = flux_map_np > 0
        if np.any(occupied):
            temp_out[occupied] = flux_density_to_brightness_temp(
                flux_map_np[occupied],
                float(freq),
                omega_pixel,
                method=brightness_conversion,
            ).astype(output_dtype)
        i_arr[fi] = temp_out

        if has_pol:
            # Jy -> K conversion for Q/U/V.  Default Rayleigh-Jeans is
            # linear and sign-preserving; "planck" matches Stokes I but
            # requires the per-pixel binned flux to be strictly positive.
            freq_hz = float(freq)
            rj_inv = 1.0 / rayleigh_jeans_factor(freq_hz, omega_pixel)

            if use_per_channel:
                ch_idx = nearest_channel_index_with_warning(
                    channel_frequencies, float(freq), label="per-channel polarization"
                )
                if (
                    per_channel_stokes_q_backend is not None
                    and per_channel_stokes_u_backend is not None
                ):
                    q_flux = xp.asarray(
                        per_channel_stokes_q_backend[ch_idx], dtype=np.float64
                    )
                    u_flux = xp.asarray(
                        per_channel_stokes_u_backend[ch_idx], dtype=np.float64
                    )
                else:
                    q_flux = (
                        stokes_q_backend
                        if stokes_q_backend is not None
                        else xp.zeros(n_sources)
                    )
                    u_flux = (
                        stokes_u_backend
                        if stokes_u_backend is not None
                        else xp.zeros(n_sources)
                    )
                if per_channel_stokes_v_backend is not None:
                    v_flux = xp.asarray(
                        per_channel_stokes_v_backend[ch_idx], dtype=np.float64
                    )
                else:
                    v_flux = (
                        stokes_v_backend
                        if stokes_v_backend is not None
                        else xp.zeros(n_sources)
                    )
            else:
                q_flux, u_flux = apply_faraday_rotation(
                    stokes_q_backend,
                    stokes_u_backend,
                    rotation_measure_backend,
                    float(freq),
                    ref_frequency_backend,
                    scale,
                    xp=xp,
                )
                v_flux = stokes_v_backend * scale
            q_map = (
                np.bincount(ipix, weights=q_flux, minlength=npix)
                if backend is None
                else backend.bincount(ipix, weights=q_flux, minlength=npix)
            )
            q_arr[fi] = _pol_flux_to_K(q_map, "Stokes Q", freq_hz, rj_inv).astype(
                output_dtype
            )

            u_map = (
                np.bincount(ipix, weights=u_flux, minlength=npix)
                if backend is None
                else backend.bincount(ipix, weights=u_flux, minlength=npix)
            )
            u_arr[fi] = _pol_flux_to_K(u_map, "Stokes U", freq_hz, rj_inv).astype(
                output_dtype
            )

            v_map = (
                np.bincount(ipix, weights=v_flux, minlength=npix)
                if backend is None
                else backend.bincount(ipix, weights=v_flux, minlength=npix)
            )
            v_arr[fi] = _pol_flux_to_K(v_map, "Stokes V", freq_hz, rj_inv).astype(
                output_dtype
            )

    logger.info(
        f"Converted {n_sources} point sources to {n_freq} HEALPix maps "
        f"({frequencies[0] / 1e6:.1f}-{frequencies[-1] / 1e6:.1f} MHz)"
    )

    # Flush and re-open read-only if memmap-backed.
    i_arr = finalize_cube(i_arr, scratch, "i_maps")
    if q_arr is not None:
        q_arr = finalize_cube(q_arr, scratch, "q_maps")
    if u_arr is not None:
        u_arr = finalize_cube(u_arr, scratch, "u_maps")
    if v_arr is not None:
        v_arr = finalize_cube(v_arr, scratch, "v_maps")

    return i_arr, q_arr, u_arr, v_arr, collision_stats
