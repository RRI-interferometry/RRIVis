"""Bidirectional conversion between point-source and HEALPix representations.

Pure functions that accept and return raw numpy arrays. No SkyModel dependency.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import healpy as hp
import numpy as np
import numpy.typing as npt

from ..containers import empty_source_arrays as _empty_source_arrays
from ..containers.constants import (
    brightness_temp_to_flux_density,
    flux_density_to_brightness_temp,
    rayleigh_jeans_factor,
)
from ..containers.footprint import _normalize_coordinate_frame, _normalize_ordering
from ..containers.spectral import (
    apply_faraday_rotation,
    compute_spectral_scale,
    nearest_channel_index_with_warning,
)
from ..diagnostics.discovery import estimate_healpix_memory
from ..support.allocation import allocate_cube, ensure_scratch_dir, finalize_cube
from ..support.backend_helpers import maybe_asarray
from ..support.healpix_geometry import pixel_solid_angle
from ..support.precision import get_sky_storage_dtype

if TYPE_CHECKING:
    from radiosim.backends import ArrayBackend
    from radiosim.core.precision import PrecisionConfig

logger = logging.getLogger(__name__)

#: Polarization values whose magnitude is below this (Jy) are treated as
#: unpolarized when deciding whether to allocate Q/U/V maps. Replaces the
#: brittle ``!= 0.0`` float comparison (spec item F5).
_POLARIZATION_PRESENCE_ATOL: float = 1e-20


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
    precision: PrecisionConfig | None = None,
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
    precision : PrecisionConfig or None
        Storage precision for the returned point arrays. When ``None`` the
        arrays are float64 (positions, flux, Stokes, ref_freq). When given,
        positions use ``precision.sky_model.source_positions`` and
        flux/Stokes/ref_freq use ``precision.sky_model.flux`` so a
        point→HEALPix→point round-trip preserves the caller's dtype policy.

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
    pos_dtype = get_sky_storage_dtype(precision, "source_positions", np.float64)
    flux_dtype = get_sky_storage_dtype(precision, "flux", np.float64)
    si_dtype = get_sky_storage_dtype(precision, "spectral_index", np.float64)

    npix = len(temp_map)
    nside = hp.npix2nside(npix)
    omega = pixel_solid_angle(nside)
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
            "(%.1f–%.1f MHz). Median alpha=%.3f.",
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
                "Only %d frequency channel available — cannot fit spectral "
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
        "ra_rad": ra_rad.astype(pos_dtype),
        "dec_rad": dec_rad.astype(pos_dtype),
        "flux": flux_ref.astype(flux_dtype),
        "spectral_index": alpha.astype(si_dtype),
        "ref_freq": np.full(n, ref_freq_val, dtype=flux_dtype),
        "stokes_q": stokes_q.astype(flux_dtype),
        "stokes_u": stokes_u.astype(flux_dtype),
        "stokes_v": stokes_v.astype(flux_dtype),
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


def bin_per_channel_flux(
    ipix: np.ndarray,
    per_channel_flux: np.ndarray,
    channel_frequencies: np.ndarray | None,
    freq: float,
    npix: int,
    *,
    backend: ArrayBackend | None = None,
) -> np.ndarray:
    """Bin per-channel flux tables into a HEALPix flux density map.

    Selects the nearest tabulated channel for ``freq`` and accumulates the
    resulting per-source flux into HEALPix pixels via ``bincount``.

    Parameters
    ----------
    ipix : np.ndarray
        HEALPix pixel index for each source, shape ``(N_sources,)``.
    per_channel_flux : np.ndarray
        Flux density in Jy at each tabulated channel, shape
        ``(N_channels, N_sources)``.
    channel_frequencies : np.ndarray or None
        Tabulated channel frequencies in Hz, shape ``(N_channels,)``.
    freq : float
        Observation frequency in Hz.
    npix : int
        Total number of HEALPix pixels.

    Returns
    -------
    np.ndarray
        Flux density map in Jy, shape ``(npix,)``.
    """
    if channel_frequencies is None:
        raise ValueError("per-channel flux binning requires channel_frequencies.")
    xp = np if backend is None else backend.xp
    idx = nearest_channel_index_with_warning(
        channel_frequencies, freq, label="per-channel flux binning"
    )
    flux_f = xp.asarray(per_channel_flux[idx], dtype=np.float64)
    if backend is None:
        return np.bincount(ipix, weights=flux_f, minlength=npix)
    return backend.bincount(ipix, weights=flux_f, minlength=npix)


def bin_scaled_flux(
    ipix: np.ndarray,
    flux: np.ndarray,
    spectral_index: np.ndarray,
    spectral_coeffs: np.ndarray | None,
    freq: float,
    ref_frequency: float | np.ndarray,
    npix: int,
    *,
    scale: np.ndarray | None = None,
    backend: ArrayBackend | None = None,
) -> np.ndarray:
    """Bin spectrally scaled point sources into a HEALPix flux density map.

    Computes the spectral scaling factor for each source (or uses a
    pre-computed ``scale``) and accumulates the scaled flux into HEALPix
    pixels via ``bincount``.

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
    if scale is None:
        scale = compute_spectral_scale(
            spectral_index, spectral_coeffs, freq, ref_frequency, xp=xp
        )
    flux_f = xp.asarray(flux, dtype=np.float64) * scale
    if backend is None:
        return np.bincount(ipix, weights=flux_f, minlength=npix)
    return backend.bincount(ipix, weights=flux_f, minlength=npix)


# =============================================================================
# point_sources_to_healpix_maps — grouped inputs + per-channel helpers
# =============================================================================


@dataclass(frozen=True)
class PointSourceHealpixInputs:
    """Point-source arrays and spectral data for point-to-HEALPix conversion."""

    ra_rad: np.ndarray
    dec_rad: np.ndarray
    flux: np.ndarray
    spectral_index: np.ndarray
    spectral_coeffs: np.ndarray | None
    stokes_q: np.ndarray | None
    stokes_u: np.ndarray | None
    stokes_v: np.ndarray | None
    rotation_measure: np.ndarray | None
    ref_frequency: float | np.ndarray
    per_channel_flux: np.ndarray | None = None
    per_channel_stokes_q: np.ndarray | None = None
    per_channel_stokes_u: np.ndarray | None = None
    per_channel_stokes_v: np.ndarray | None = None
    channel_frequencies: np.ndarray | None = None


@dataclass(frozen=True)
class HealpixConversionConfig:
    """HEALPix geometry and output policy for point-source conversion."""

    nside: int
    frequencies: np.ndarray
    brightness_conversion: str
    coordinate_frame: str = "icrs"
    ordering: str = "ring"
    output_dtype: npt.DTypeLike = np.float32
    memmap_path: str | None = None
    polarization_brightness_conversion: str = "rayleigh-jeans"


@dataclass(frozen=True)
class _SourceInputs:
    """Reference-frequency per-source arrays cast to the active backend.

    All arrays are backend arrays (float64); the optional ones stay ``None``
    when the caller supplied nothing. ``ipix`` is the HEALPix pixel index of
    each source. ``ref_frequency`` is scalar-or-per-source.
    """

    ipix: np.ndarray
    flux: np.ndarray
    spectral_index: np.ndarray
    spectral_coeffs: np.ndarray | None
    ref_frequency: np.ndarray
    stokes_q: np.ndarray | None
    stokes_u: np.ndarray | None
    stokes_v: np.ndarray | None
    rotation_measure: np.ndarray | None
    n_sources: int


@dataclass(frozen=True)
class _PerChannelInputs:
    """Per-channel (spectrum table) per-source arrays cast to the backend.

    All entries are ``None`` unless the caller supplied a per-channel
    spectrum. ``channel_frequencies`` stays numpy (used only for nearest-
    channel lookups, not arithmetic on the backend).
    """

    flux: np.ndarray | None
    stokes_q: np.ndarray | None
    stokes_u: np.ndarray | None
    stokes_v: np.ndarray | None
    channel_frequencies: np.ndarray | None

    @property
    def active(self) -> bool:
        return self.flux is not None and self.channel_frequencies is not None


@dataclass(frozen=True)
class _ChannelConfig:
    """Shared geometry / method configuration for one HEALPix channel build."""

    nside: int
    npix: int
    omega_pixel: float
    output_dtype: npt.DTypeLike
    brightness_conversion: str
    pol_method: str


def _cast_inputs_to_backend(
    *,
    ipix: np.ndarray,
    flux: np.ndarray,
    spectral_index: np.ndarray,
    spectral_coeffs: np.ndarray | None,
    ref_frequency: float | np.ndarray,
    stokes_q: np.ndarray | None,
    stokes_u: np.ndarray | None,
    stokes_v: np.ndarray | None,
    rotation_measure: np.ndarray | None,
    per_channel_flux: np.ndarray | None,
    per_channel_stokes_q: np.ndarray | None,
    per_channel_stokes_u: np.ndarray | None,
    per_channel_stokes_v: np.ndarray | None,
    channel_frequencies: np.ndarray | None,
    n_sources: int,
    backend: ArrayBackend | None,
) -> tuple[_SourceInputs, _PerChannelInputs]:
    """Cast every per-source array to the active backend once.

    Routes the repeated ``None if x is None else backend.asarray(x)`` ternary
    through :func:`maybe_asarray` (spec item B1) and packs the result into the
    two grouped-input dataclasses consumed by the channel builders.
    """
    sources = _SourceInputs(
        ipix=ipix,
        flux=maybe_asarray(backend, flux, dtype=np.float64),
        spectral_index=maybe_asarray(backend, spectral_index, dtype=np.float64),
        spectral_coeffs=maybe_asarray(backend, spectral_coeffs, dtype=np.float64),
        ref_frequency=maybe_asarray(backend, ref_frequency, dtype=np.float64),
        stokes_q=maybe_asarray(backend, stokes_q, dtype=np.float64),
        stokes_u=maybe_asarray(backend, stokes_u, dtype=np.float64),
        stokes_v=maybe_asarray(backend, stokes_v, dtype=np.float64),
        rotation_measure=maybe_asarray(backend, rotation_measure, dtype=np.float64),
        n_sources=n_sources,
    )
    per_channel = _PerChannelInputs(
        flux=maybe_asarray(backend, per_channel_flux, dtype=np.float64),
        stokes_q=maybe_asarray(backend, per_channel_stokes_q, dtype=np.float64),
        stokes_u=maybe_asarray(backend, per_channel_stokes_u, dtype=np.float64),
        stokes_v=maybe_asarray(backend, per_channel_stokes_v, dtype=np.float64),
        channel_frequencies=channel_frequencies,
    )
    return sources, per_channel


def _compute_stokes_i_channel(
    freq: float,
    sources: _SourceInputs,
    per_channel: _PerChannelInputs,
    cfg: _ChannelConfig,
    backend: ArrayBackend | None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Build the Stokes-I brightness-temperature row for one channel.

    Returns ``(temp_out, scale)`` where ``temp_out`` is the length-``npix``
    K row (``cfg.output_dtype``) and ``scale`` is the per-source spectral
    scale used by the polarization builder (``None`` on the per-channel path,
    which does not share a scale).
    """
    xp = np if backend is None else backend.xp
    to_numpy = np.asarray if backend is None else backend.to_numpy

    if per_channel.active:
        scale = None
        flux_map = bin_per_channel_flux(
            sources.ipix,
            per_channel.flux,
            per_channel.channel_frequencies,
            float(freq),
            cfg.npix,
            backend=backend,
        )
    else:
        scale = compute_spectral_scale(
            sources.spectral_index,
            sources.spectral_coeffs,
            float(freq),
            sources.ref_frequency,
            xp=xp,
        )
        flux_map = bin_scaled_flux(
            sources.ipix,
            sources.flux,
            sources.spectral_index,
            sources.spectral_coeffs,
            float(freq),
            sources.ref_frequency,
            cfg.npix,
            scale=scale,
            backend=backend,
        )

    temp_out = np.zeros(cfg.npix, dtype=cfg.output_dtype)
    flux_map_np = to_numpy(flux_map)
    occupied = flux_map_np > 0
    if np.any(occupied):
        temp_out[occupied] = flux_density_to_brightness_temp(
            flux_map_np[occupied],
            float(freq),
            cfg.omega_pixel,
            method=cfg.brightness_conversion,
        ).astype(cfg.output_dtype)
    return temp_out, scale


def _pol_flux_map_to_K(
    flux_map: np.ndarray,
    name: str,
    freq_hz: float,
    rj_inv: float,
    cfg: _ChannelConfig,
    backend: ArrayBackend | None,
) -> np.ndarray:
    """Convert a per-pixel Stokes Q/U/V flux map (Jy) to brightness T (K).

    Default Rayleigh-Jeans is linear and sign-preserving; ``"planck"``
    matches Stokes I but requires the per-pixel binned flux to be strictly
    positive.
    """
    to_numpy = np.asarray if backend is None else backend.to_numpy
    flux_map_np = to_numpy(flux_map)
    if cfg.pol_method == "planck":
        if np.any(flux_map_np <= 0):
            raise ValueError(
                "polarization_brightness_conversion='planck' requires "
                f"strictly positive {name} flux per pixel; got values <= 0 "
                f"after binning at {freq_hz / 1e6:.3f} MHz. Use "
                "'rayleigh-jeans' for sign-preserving linear conversion."
            )
        return flux_density_to_brightness_temp(
            flux_map_np, freq_hz, cfg.omega_pixel, method="planck"
        )
    return flux_map_np * rj_inv


def _bincount(
    ipix: np.ndarray,
    weights: np.ndarray,
    npix: int,
    backend: ArrayBackend | None,
) -> np.ndarray:
    if backend is None:
        return np.bincount(ipix, weights=weights, minlength=npix)
    return backend.bincount(ipix, weights=weights, minlength=npix)


def _compute_stokes_pol_channel(
    freq: float,
    scale: np.ndarray | None,
    sources: _SourceInputs,
    per_channel: _PerChannelInputs,
    cfg: _ChannelConfig,
    backend: ArrayBackend | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build the Stokes Q/U/V brightness-temperature rows for one channel.

    Returns ``(q_row, u_row, v_row)``, each length-``npix`` in
    ``cfg.output_dtype``. ``scale`` is the per-source Stokes-I spectral scale
    (used by the Faraday-rotation path); it is ``None`` on the per-channel
    path which reads flux directly from the spectrum table.
    """
    xp = np if backend is None else backend.xp
    freq_hz = float(freq)
    rj_inv = 1.0 / rayleigh_jeans_factor(freq_hz, cfg.omega_pixel)

    if per_channel.active:
        ch_idx = nearest_channel_index_with_warning(
            per_channel.channel_frequencies,
            freq_hz,
            label="per-channel polarization",
        )
        if per_channel.stokes_q is not None and per_channel.stokes_u is not None:
            q_flux = xp.asarray(per_channel.stokes_q[ch_idx], dtype=np.float64)
            u_flux = xp.asarray(per_channel.stokes_u[ch_idx], dtype=np.float64)
        else:
            q_flux = (
                sources.stokes_q
                if sources.stokes_q is not None
                else xp.zeros(sources.n_sources)
            )
            u_flux = (
                sources.stokes_u
                if sources.stokes_u is not None
                else xp.zeros(sources.n_sources)
            )
        if per_channel.stokes_v is not None:
            v_flux = xp.asarray(per_channel.stokes_v[ch_idx], dtype=np.float64)
        else:
            v_flux = (
                sources.stokes_v
                if sources.stokes_v is not None
                else xp.zeros(sources.n_sources)
            )
    else:
        q_flux, u_flux = apply_faraday_rotation(
            sources.stokes_q,
            sources.stokes_u,
            sources.rotation_measure,
            freq_hz,
            sources.ref_frequency,
            scale,
            xp=xp,
        )
        v_flux = sources.stokes_v * scale

    q_map = _bincount(sources.ipix, q_flux, cfg.npix, backend)
    u_map = _bincount(sources.ipix, u_flux, cfg.npix, backend)
    v_map = _bincount(sources.ipix, v_flux, cfg.npix, backend)

    q_row = _pol_flux_map_to_K(q_map, "Stokes Q", freq_hz, rj_inv, cfg, backend).astype(
        cfg.output_dtype
    )
    u_row = _pol_flux_map_to_K(u_map, "Stokes U", freq_hz, rj_inv, cfg, backend).astype(
        cfg.output_dtype
    )
    v_row = _pol_flux_map_to_K(v_map, "Stokes V", freq_hz, rj_inv, cfg, backend).astype(
        cfg.output_dtype
    )
    return q_row, u_row, v_row


def _has_polarization(
    stokes_q: np.ndarray | None,
    stokes_u: np.ndarray | None,
    stokes_v: np.ndarray | None,
) -> bool:
    """Decide whether any source carries meaningful polarization.

    Uses a magnitude tolerance (:data:`_POLARIZATION_PRESENCE_ATOL`) instead
    of a brittle ``!= 0.0`` exact comparison (spec item F5), so floating-point
    dust (e.g. a ``1e-30`` Stokes Q) does not force a full Q/U/V allocation.
    """
    if stokes_q is None or stokes_u is None or stokes_v is None:
        return False
    atol = _POLARIZATION_PRESENCE_ATOL
    return bool(
        np.any(np.abs(stokes_q) > atol)
        or np.any(np.abs(stokes_u) > atol)
        or np.any(np.abs(stokes_v) > atol)
    )


def _point_sources_to_healpix_maps_impl(
    sources: PointSourceHealpixInputs,
    config: HealpixConversionConfig,
    *,
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
    ordering : {"ring", "nest"}, default "ring"
        HEALPix pixel ordering for ``ang2pix`` and the output map layout.
    output_dtype : DTypeLike, default np.float32
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
    ra_rad = sources.ra_rad
    dec_rad = sources.dec_rad
    flux = sources.flux
    spectral_index = sources.spectral_index
    spectral_coeffs = sources.spectral_coeffs
    stokes_q = sources.stokes_q
    stokes_u = sources.stokes_u
    stokes_v = sources.stokes_v
    rotation_measure = sources.rotation_measure
    ref_frequency = sources.ref_frequency
    per_channel_flux = sources.per_channel_flux
    per_channel_stokes_q = sources.per_channel_stokes_q
    per_channel_stokes_u = sources.per_channel_stokes_u
    per_channel_stokes_v = sources.per_channel_stokes_v
    channel_frequencies = sources.channel_frequencies

    nside = config.nside
    frequencies = config.frequencies
    brightness_conversion = config.brightness_conversion
    coordinate_frame = config.coordinate_frame
    output_dtype = config.output_dtype
    memmap_path = config.memmap_path

    ra_rad = sources.ra_rad
    dec_rad = sources.dec_rad
    flux = sources.flux
    spectral_index = sources.spectral_index
    spectral_coeffs = sources.spectral_coeffs
    stokes_q = sources.stokes_q
    stokes_u = sources.stokes_u
    stokes_v = sources.stokes_v
    rotation_measure = sources.rotation_measure
    ref_frequency = sources.ref_frequency
    per_channel_flux = sources.per_channel_flux
    per_channel_stokes_q = sources.per_channel_stokes_q
    per_channel_stokes_u = sources.per_channel_stokes_u
    per_channel_stokes_v = sources.per_channel_stokes_v
    channel_frequencies = sources.channel_frequencies

    nside = config.nside
    frequencies = config.frequencies
    brightness_conversion = config.brightness_conversion
    coordinate_frame = config.coordinate_frame
    output_dtype = config.output_dtype
    memmap_path = config.memmap_path

    pol_method = str(config.polarization_brightness_conversion).lower()
    if pol_method not in {"rayleigh-jeans", "planck"}:
        raise ValueError(
            "polarization_brightness_conversion must be 'rayleigh-jeans' or "
            f"'planck', got {config.polarization_brightness_conversion!r}."
        )

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
            {"n_sources": 0, "n_collisions": 0, "n_merged": 0},
        )

    omega_pixel = pixel_solid_angle(nside)
    frame = _normalize_coordinate_frame(coordinate_frame)
    ordering = _normalize_ordering(config.ordering)
    nest = ordering == "nest"

    has_pol = _has_polarization(stokes_q, stokes_u, stokes_v)
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

    ipix = hp.ang2pix(nside, np.pi / 2 - lat_rad, lon_rad, nest=nest)
    collision_stats = _collision_stats(ipix, n_sources, nside)

    cfg = _ChannelConfig(
        nside=nside,
        npix=npix,
        omega_pixel=omega_pixel,
        output_dtype=output_dtype,
        brightness_conversion=brightness_conversion,
        pol_method=pol_method,
    )
    sources, per_channel = _cast_inputs_to_backend(
        ipix=ipix,
        flux=flux,
        spectral_index=spectral_index,
        spectral_coeffs=spectral_coeffs,
        ref_frequency=ref_frequency,
        stokes_q=stokes_q,
        stokes_u=stokes_u,
        stokes_v=stokes_v,
        rotation_measure=rotation_measure,
        per_channel_flux=per_channel_flux,
        per_channel_stokes_q=per_channel_stokes_q,
        per_channel_stokes_u=per_channel_stokes_u,
        per_channel_stokes_v=per_channel_stokes_v,
        channel_frequencies=channel_frequencies,
        n_sources=n_sources,
        backend=backend,
    )

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

    for fi, freq in enumerate(frequencies):
        temp_out, scale = _compute_stokes_i_channel(
            freq, sources, per_channel, cfg, backend
        )
        i_arr[fi] = temp_out
        if has_pol:
            q_row, u_row, v_row = _compute_stokes_pol_channel(
                freq, scale, sources, per_channel, cfg, backend
            )
            q_arr[fi] = q_row
            u_arr[fi] = u_row
            v_arr[fi] = v_row

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


def point_sources_to_healpix_maps(
    sources: PointSourceHealpixInputs,
    config: HealpixConversionConfig,
    *,
    backend: ArrayBackend | None = None,
) -> tuple[
    np.ndarray,
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray | None,
    dict[str, int],
]:
    """Convert point sources to HEALPix maps using grouped inputs."""
    return _point_sources_to_healpix_maps_impl(sources, config, backend=backend)


def _collision_stats(ipix: np.ndarray, n_sources: int, nside: int) -> dict[str, int]:
    """Count pixel collisions and warn when source identities are merged."""
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
    return {
        "n_sources": int(n_sources),
        "n_collisions": n_collisions,
        "n_merged": n_merged,
    }
