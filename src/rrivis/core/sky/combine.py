# rrivis/core/sky/combine.py
"""Sky model combination helpers.

Provides functions for concatenating point-source arrays and combining
HEALPix multi-frequency maps from multiple :class:`SkyModel` instances.

The two internal helpers (``concat_point_sources``, ``combine_healpix``)
return raw data dicts; only the public ``combine_models()`` constructs a
:class:`SkyModel` via a late import to avoid circular dependencies.
"""

from __future__ import annotations

import logging
import warnings
from typing import TYPE_CHECKING, Any, TypedDict

import healpy as hp
import numpy as np

from .constants import (
    brightness_temp_to_flux_density,
    flux_density_to_brightness_temp,
    rayleigh_jeans_factor,
)
from .convert import bin_sources_to_flux
from .model import SkyFormat
from .spectral import apply_faraday_rotation, compute_spectral_scale

if TYPE_CHECKING:
    from rrivis.core.precision import PrecisionConfig

    from .model import SkyModel

logger = logging.getLogger(__name__)


class CombineHealpixData(TypedDict):
    """Return type for combine_healpix."""

    healpix_maps: np.ndarray
    healpix_q_maps: np.ndarray | None
    healpix_u_maps: np.ndarray | None
    healpix_v_maps: np.ndarray | None
    healpix_nside: int
    observation_frequencies: np.ndarray
    reference_frequency: float | None


# =============================================================================
# Internal helper: concat point sources
# =============================================================================


def concat_point_sources(
    models: list[SkyModel],
    reference_frequency: float | None = None,
    brightness_conversion: str = "planck",
    precision: PrecisionConfig | None = None,
) -> dict[str, Any]:
    """Concatenate columnar arrays from multiple point-source SkyModels.

    Each model that lacks point-source arrays is first converted via
    ``with_representation("point_sources", ...)``.  Empty models are
    silently skipped.

    Parameters
    ----------
    models : list of SkyModel
        Models to concatenate.
    reference_frequency : float, optional
        Reference frequency for healpix-to-point-source conversion.
    brightness_conversion : str, default ``"planck"``
        Brightness conversion method (carried through to the result).
    precision : PrecisionConfig, optional
        Precision configuration (not applied here -- the caller handles
        dtype casting via the SkyModel constructor).

    Returns
    -------
    dict
        Raw data dict with keys matching SkyModel property names:
        ``ra_rad``, ``dec_rad``, ``flux``, ``spectral_index``,
        ``stokes_q``, ``stokes_u``, ``stokes_v``,
        ``rotation_measure``, ``major_arcsec``, ``minor_arcsec``,
        ``pa_deg``, ``spectral_coeffs``, ``reference_frequency``.
        Array values are ``np.ndarray``; optional fields are ``None``
        when no model contributes data.  An empty-model result has
        zero-length arrays.
    """
    # Ensure each model has point-source arrays populated; skip empties
    populated: list[SkyModel] = []
    for m in models:
        if m.ra_rad is None:
            m = m.with_representation("point_sources", frequency=reference_frequency)
        if m.ra_rad is not None and len(m.ra_rad) > 0:
            populated.append(m)

    if not populated:
        from .model import _empty_source_arrays

        return {**_empty_source_arrays(), "reference_frequency": None}

    # --- Required arrays ---
    ra = np.concatenate([m.ra_rad for m in populated])
    dec = np.concatenate([m.dec_rad for m in populated])
    flux = np.concatenate([m.flux for m in populated])
    si = np.concatenate([m.spectral_index for m in populated])
    sq = np.concatenate([m.stokes_q for m in populated])
    su = np.concatenate([m.stokes_u for m in populated])
    sv = np.concatenate([m.stokes_v for m in populated])

    ref_freq_arr = np.concatenate(
        [
            m.ref_freq
            if m.ref_freq is not None
            else np.full(
                len(m.ra_rad),
                m.reference_frequency or reference_frequency or 0.0,
                dtype=np.float64,
            )
            for m in populated
        ]
    )

    n = len(ra)

    # --- Optional: rotation measure ---
    rm: np.ndarray | None = None
    if any(m.rotation_measure is not None for m in populated):
        rm = np.concatenate(
            [
                m.rotation_measure
                if m.rotation_measure is not None
                else np.zeros(len(m.ra_rad), dtype=np.float64)
                for m in populated
            ]
        )

    # --- Optional: Gaussian morphology ---
    major: np.ndarray | None = None
    minor: np.ndarray | None = None
    pa: np.ndarray | None = None
    if any(m.major_arcsec is not None for m in populated):
        major = np.concatenate(
            [
                m.major_arcsec
                if m.major_arcsec is not None
                else np.zeros(len(m.ra_rad), dtype=np.float64)
                for m in populated
            ]
        )
        minor = np.concatenate(
            [
                m.minor_arcsec
                if m.minor_arcsec is not None
                else np.zeros(len(m.ra_rad), dtype=np.float64)
                for m in populated
            ]
        )
        pa = np.concatenate(
            [
                m.pa_deg
                if m.pa_deg is not None
                else np.zeros(len(m.ra_rad), dtype=np.float64)
                for m in populated
            ]
        )

    # --- Optional: spectral coefficients (may differ in N_terms) ---
    sp_coeffs: np.ndarray | None = None
    if any(m.spectral_coeffs is not None for m in populated):
        max_terms = max(
            m.spectral_coeffs.shape[1]
            for m in populated
            if m.spectral_coeffs is not None
        )
        parts: list[np.ndarray] = []
        for m in populated:
            n_m = len(m.ra_rad)
            if m.spectral_coeffs is not None:
                arr = m.spectral_coeffs
                if arr.shape[1] < max_terms:
                    pad = np.zeros((n_m, max_terms - arr.shape[1]), dtype=arr.dtype)
                    arr = np.concatenate([arr, pad], axis=1)
                parts.append(arr)
            else:
                # Default: column 0 = alpha, rest zero
                fallback = np.zeros((n_m, max_terms), dtype=np.float64)
                fallback[:, 0] = m.spectral_index
                parts.append(fallback)
        sp_coeffs = np.concatenate(parts, axis=0)

    # Infer reference frequency from models if not provided
    ref_freq_val = reference_frequency
    if ref_freq_val is None:
        for m in populated:
            if m.reference_frequency is not None:
                ref_freq_val = m.reference_frequency
                break

    logger.info(f"Concatenated {len(populated)} models: {n} total sources")

    return {
        "ra_rad": ra,
        "dec_rad": dec,
        "flux": flux,
        "spectral_index": si,
        "stokes_q": sq,
        "stokes_u": su,
        "stokes_v": sv,
        "ref_freq": ref_freq_arr,
        "rotation_measure": rm,
        "major_arcsec": major,
        "minor_arcsec": minor,
        "pa_deg": pa,
        "spectral_coeffs": sp_coeffs,
        "reference_frequency": ref_freq_val,
    }


# =============================================================================
# Internal helper: combine HEALPix maps
# =============================================================================


def combine_healpix(
    models: list[SkyModel],
    ref_nside: int,
    ref_freqs: np.ndarray,
    ref_frequency: float | None,
    brightness_conversion: str = "planck",
    precision: PrecisionConfig | None = None,
) -> CombineHealpixData:
    """Combine models by element-wise addition in Jy space per frequency channel.

    All ``healpix_map`` models must share the same nside and frequency
    grid.  Point-source models are binned into the same grid via
    ``np.bincount``.  Stokes I is converted T_b -> Jy -> T_b so that
    addition is physically correct under both Planck and Rayleigh-Jeans
    conversions.

    Parameters
    ----------
    models : list of SkyModel
        Models to combine.
    ref_nside : int
        Common HEALPix nside (from first ``healpix_map`` model).
    ref_freqs : np.ndarray
        Common frequency grid in Hz, shape ``(n_freq,)``.
    ref_frequency : float or None
        Reference frequency for spectral extrapolation of point sources.
    brightness_conversion : str, default ``"planck"``
        Brightness conversion method.
    precision : PrecisionConfig, optional
        Precision configuration.  Controls the output dtype of HEALPix
        arrays via ``precision.sky_model.get_dtype("healpix_maps")``.

    Returns
    -------
    dict
        Raw data dict with keys: ``_healpix_maps`` (``np.ndarray`` of
        shape ``(n_freq, npix)``), ``_healpix_q_maps``,
        ``_healpix_u_maps``, ``_healpix_v_maps`` (same shape or
        ``None``), ``_healpix_nside`` (int),
        ``_observation_frequencies`` (``np.ndarray``), ``frequency``
        (``float | None``).

    Raises
    ------
    ValueError
        If HEALPix models have mismatched nside or frequency grids.
    """
    # Validate all healpix_map models share the same nside and
    # frequency grid before doing element-wise arithmetic.
    for m in models:
        if m.mode != SkyFormat.HEALPIX:
            continue
        m_nside = m.healpix_nside
        m_freqs = m.observation_frequencies
        if m_nside != ref_nside:
            raise ValueError(
                f"Cannot combine HEALPix models with different nside values: "
                f"reference has nside={ref_nside}, model '{m.model_name}' has "
                f"nside={m_nside}. Resample to a common nside with hp.ud_grade() "
                f"before combining."
            )
        if not np.array_equal(m_freqs, ref_freqs):
            raise ValueError(
                f"Cannot combine HEALPix models with different frequency grids: "
                f"reference has {len(ref_freqs)} channels "
                f"({ref_freqs[0] / 1e6:.3f}\u2013{ref_freqs[-1] / 1e6:.3f} MHz), "
                f"model '{m.model_name}' has {len(m_freqs)} channels "
                f"({m_freqs[0] / 1e6:.3f}\u2013{m_freqs[-1] / 1e6:.3f} MHz). "
                f"All models must share the same observation frequency grid."
            )

    npix = hp.nside2npix(ref_nside)
    n_freq = len(ref_freqs)
    omega_pixel = 4 * np.pi / npix

    # Collect point-source data for pixel-binning
    ps_models_data = []
    for m in models:
        if m.mode == SkyFormat.POINT_SOURCES and m.has_point_sources:
            ipix_m = hp.ang2pix(ref_nside, np.pi / 2 - m.dec_rad, m.ra_rad)
            ps_models_data.append((ipix_m, m.flux, m.spectral_index, m))

    # Check if any model has polarized maps
    any_pol = any(
        m.has_polarized_healpix_maps for m in models if m.mode == SkyFormat.HEALPIX
    ) or any(
        m.stokes_q is not None
        and (
            np.any(m.stokes_q != 0)
            or np.any(m.stokes_u != 0)
            or np.any(m.stokes_v != 0)
        )
        for m in models
        if m.mode == SkyFormat.POINT_SOURCES and m.has_point_sources
    )

    # Resolve output dtype from precision config
    hp_dtype = (
        precision.sky_model.get_dtype("healpix_maps")
        if precision is not None
        else np.float32
    )

    # Pre-allocate output arrays: shape (n_freq, npix)
    combined_I = np.zeros((n_freq, npix), dtype=hp_dtype)
    combined_Q: np.ndarray | None = (
        np.zeros((n_freq, npix), dtype=hp_dtype) if any_pol else None
    )
    combined_U: np.ndarray | None = (
        np.zeros((n_freq, npix), dtype=hp_dtype) if any_pol else None
    )
    combined_V: np.ndarray | None = (
        np.zeros((n_freq, npix), dtype=hp_dtype) if any_pol else None
    )

    for freq_idx, freq_hz in enumerate(ref_freqs):
        combined_flux = np.zeros(npix, dtype=np.float64)
        combined_q_flux = np.zeros(npix, dtype=np.float64) if any_pol else None
        combined_u_flux = np.zeros(npix, dtype=np.float64) if any_pol else None
        combined_v_flux = np.zeros(npix, dtype=np.float64) if any_pol else None

        rj_factor = rayleigh_jeans_factor(freq_hz, omega_pixel)

        # --- Add healpix_map models ---
        for m in models:
            if m.mode == SkyFormat.HEALPIX:
                t_map = m.healpix_maps[freq_idx].astype(np.float64)
                pos = t_map > 0
                if np.any(pos):
                    combined_flux[pos] += brightness_temp_to_flux_density(
                        t_map[pos],
                        freq_hz,
                        omega_pixel,
                        method=brightness_conversion,
                    )

                if any_pol and m.has_polarized_healpix_maps:
                    if m.healpix_q_maps is not None:
                        q_t = m.healpix_q_maps[freq_idx]
                        if q_t is not None:
                            combined_q_flux += q_t.astype(np.float64) * rj_factor
                    if m.healpix_u_maps is not None:
                        u_t = m.healpix_u_maps[freq_idx]
                        if u_t is not None:
                            combined_u_flux += u_t.astype(np.float64) * rj_factor
                    if m.healpix_v_maps is not None:
                        v_t = m.healpix_v_maps[freq_idx]
                        if v_t is not None:
                            combined_v_flux += v_t.astype(np.float64) * rj_factor

        # --- Add point-source models via bincount ---
        for ipix_m, flux_ref_m, alpha_m, m_obj in ps_models_data:
            ps_ref_freq = (
                m_obj.ref_freq
                if m_obj.ref_freq is not None
                else np.full(
                    len(flux_ref_m),
                    m_obj.reference_frequency or ref_frequency,
                    dtype=np.float64,
                )
            )
            combined_flux += bin_sources_to_flux(
                ipix_m,
                flux_ref_m,
                alpha_m,
                m_obj.spectral_coeffs,
                float(freq_hz),
                ps_ref_freq,
                npix,
            )

            if any_pol and m_obj.stokes_q is not None:
                scale = compute_spectral_scale(
                    alpha_m, m_obj.spectral_coeffs, float(freq_hz), ps_ref_freq
                )
                q_f, u_f = apply_faraday_rotation(
                    m_obj.stokes_q,
                    m_obj.stokes_u,
                    m_obj.rotation_measure,
                    float(freq_hz),
                    ps_ref_freq,
                    scale,
                )
                combined_q_flux += np.bincount(ipix_m, weights=q_f, minlength=npix)
                combined_u_flux += np.bincount(ipix_m, weights=u_f, minlength=npix)
                combined_v_flux += np.bincount(
                    ipix_m, weights=m_obj.stokes_v * scale, minlength=npix
                )

        # --- Convert combined flux back to brightness temperature ---
        combined_T_b = np.zeros(npix, dtype=np.float64)
        pos_flux = combined_flux > 0
        if np.any(pos_flux):
            combined_T_b[pos_flux] = flux_density_to_brightness_temp(
                combined_flux[pos_flux],
                freq_hz,
                omega_pixel,
                method=brightness_conversion,
            )
        combined_I[freq_idx] = combined_T_b.astype(hp_dtype)

        if any_pol:
            rj_inv = 1.0 / rj_factor if rj_factor != 0 else 0.0
            combined_Q[freq_idx] = (combined_q_flux * rj_inv).astype(hp_dtype)
            combined_U[freq_idx] = (combined_u_flux * rj_inv).astype(hp_dtype)
            combined_V[freq_idx] = (combined_v_flux * rj_inv).astype(hp_dtype)

    logger.info(
        f"Combined {len(models)} models into healpix_map "
        f"({n_freq} channels, nside={ref_nside}"
        f"{', stokes=IQUV' if any_pol else ''})"
    )

    return {
        "healpix_maps": combined_I,
        "healpix_q_maps": combined_Q,
        "healpix_u_maps": combined_U,
        "healpix_v_maps": combined_V,
        "healpix_nside": ref_nside,
        "observation_frequencies": ref_freqs,
        "reference_frequency": None,
    }


# =============================================================================
# Combination helpers (private)
# =============================================================================


def _resolve_combination_params(
    models: list[SkyModel],
    representation: str | None,
    frequency: float | None,
    ref_frequency: float | None,
) -> tuple[str, float | None, float | None]:
    """Auto-detect representation and resolve frequency defaults.

    Returns (representation, frequency, ref_frequency).
    """
    # Auto-detect representation
    if representation is None:
        has_healpix_native = any(m.native_format == SkyFormat.HEALPIX for m in models)
        representation = (
            SkyFormat.HEALPIX if has_healpix_native else SkyFormat.POINT_SOURCES
        )

    # Resolve frequency from model metadata
    freq = frequency
    if freq is None:
        for m in models:
            if m.reference_frequency is not None:
                freq = m.reference_frequency
                break

    # Resolve ref_frequency
    if ref_frequency is None:
        for m in models:
            if m.reference_frequency is not None:
                ref_frequency = m.reference_frequency
                break
        if ref_frequency is None:
            ref_frequency = freq

    return representation, freq, ref_frequency


def _warn_combination_issues(
    models: list[SkyModel], brightness_conversion: str
) -> None:
    """Emit warnings for inconsistent brightness_conversion and double-counting."""
    bc_values = {m.brightness_conversion for m in models}
    if len(bc_values) > 1:
        warnings.warn(
            f"Combining models with different brightness_conversion settings: "
            f"{bc_values}. The combined model will use '{brightness_conversion}'. "
            f"For consistent results, ensure all models use the same conversion.",
            UserWarning,
            stacklevel=3,
        )

    has_catalog = any(
        m.mode == SkyFormat.POINT_SOURCES and m.has_point_sources for m in models
    )
    has_diffuse = any(m.mode == SkyFormat.HEALPIX for m in models)
    if has_catalog and has_diffuse:
        warnings.warn(
            "Combining catalog sources (GLEAM/MALS) with diffuse models (GSM/LFSM/Haslam) "
            "may result in double-counting of bright sources. Diffuse models already include "
            "integrated emission from bright sources. Consider using only one model type "
            "or implementing source subtraction for accurate results.",
            UserWarning,
            stacklevel=3,
        )


def _combine_as_healpix_merge(
    models: list[SkyModel],
    ref_frequency: float | None,
    brightness_conversion: str,
    precision: PrecisionConfig | None,
) -> SkyModel:
    """Combine models with existing HEALPix maps via Jy-space addition."""
    from .model import SkyModel

    ref_model = next(m for m in models if m.mode == SkyFormat.HEALPIX)
    ref_nside = ref_model.healpix_nside
    ref_freqs = ref_model.observation_frequencies

    data = combine_healpix(
        models,
        ref_nside=ref_nside,
        ref_freqs=ref_freqs,
        ref_frequency=ref_frequency,
        brightness_conversion=brightness_conversion,
        precision=precision,
    )

    return SkyModel(
        _healpix_maps=data["healpix_maps"],
        _healpix_q_maps=data["healpix_q_maps"],
        _healpix_u_maps=data["healpix_u_maps"],
        _healpix_v_maps=data["healpix_v_maps"],
        _healpix_nside=data["healpix_nside"],
        _observation_frequencies=data["observation_frequencies"],
        _native_format=SkyFormat.HEALPIX,
        reference_frequency=data["reference_frequency"],
        model_name="combined",
        brightness_conversion=brightness_conversion,
        _precision=precision,
    )


def _combine_as_point_sources(
    models: list[SkyModel],
    frequency: float | None,
    brightness_conversion: str,
    precision: PrecisionConfig | None,
) -> SkyModel:
    """Combine models by concatenating point-source arrays."""
    from .model import SkyModel

    data = concat_point_sources(
        models,
        reference_frequency=frequency,
        brightness_conversion=brightness_conversion,
        precision=precision,
    )

    return SkyModel(
        _ra_rad=data["ra_rad"],
        _dec_rad=data["dec_rad"],
        _flux=data["flux"],
        _spectral_index=data["spectral_index"],
        _stokes_q=data["stokes_q"],
        _stokes_u=data["stokes_u"],
        _stokes_v=data["stokes_v"],
        _ref_freq=data["ref_freq"],
        _rotation_measure=data["rotation_measure"],
        _major_arcsec=data["major_arcsec"],
        _minor_arcsec=data["minor_arcsec"],
        _pa_deg=data["pa_deg"],
        _spectral_coeffs=data["spectral_coeffs"],
        _native_format=SkyFormat.POINT_SOURCES,
        model_name="combined",
        reference_frequency=data["reference_frequency"],
        brightness_conversion=brightness_conversion,
        _precision=precision,
    )


# =============================================================================
# Public API: combine_models
# =============================================================================


def combine_models(
    models: list[SkyModel],
    representation: str | None = None,
    nside: int = 64,
    frequency: float | None = None,
    obs_frequency_config: dict[str, Any] | None = None,
    ref_frequency: float | None = None,
    brightness_conversion: str = "planck",
    precision: PrecisionConfig | None = None,
) -> SkyModel:
    """Combine multiple sky models into one.

    Dispatches to the appropriate combination strategy based on the input
    models and requested representation.  See module docstring for details.

    Parameters
    ----------
    models : list of SkyModel
        Sky models to combine.
    representation : str or None, default None
        Output representation: ``"point_sources"`` or ``"healpix_map"``.
        When ``None``, auto-detects from input models.
    nside : int, default 64
        HEALPix NSIDE for ``healpix_map`` output.
    frequency : float, optional
        Frequency for HEALPix-to-point-source conversions.
    obs_frequency_config : dict, optional
        Required for ``healpix_map`` when no model already has maps.
    ref_frequency : float, optional
        Reference frequency for spectral extrapolation (Hz).
    brightness_conversion : str, default ``"planck"``
        Brightness conversion method.
    precision : PrecisionConfig, optional
        Precision configuration for the combined model.

    Returns
    -------
    SkyModel
        Combined sky model.
    """
    from .model import SkyModel

    if not models:
        return SkyModel.empty_sky(
            model_name="combined_empty",
            brightness_conversion=brightness_conversion,
            precision=precision,
        )

    _warn_combination_issues(models, brightness_conversion)
    representation, freq, ref_freq = _resolve_combination_params(
        models, representation, frequency, ref_frequency
    )

    has_healpix_map = any(m.mode == SkyFormat.HEALPIX for m in models)

    # Path 1: HEALPix merge (at least one model already has maps)
    if representation == SkyFormat.HEALPIX and has_healpix_map:
        return _combine_as_healpix_merge(
            models, ref_freq, brightness_conversion, precision
        )

    # Path 2: Concatenate as point sources
    combined = _combine_as_point_sources(models, freq, brightness_conversion, precision)

    # Path 3: Optionally convert concatenated PS to HEALPix
    if representation == SkyFormat.HEALPIX:
        if obs_frequency_config is None:
            raise ValueError(
                "obs_frequency_config is required for healpix_map representation. "
                "Provide a dict with: starting_frequency, frequency_interval, "
                "frequency_bandwidth, frequency_unit."
            )
        combined = combined.with_healpix_maps(
            nside=nside,
            obs_frequency_config=obs_frequency_config,
            ref_frequency=ref_freq,
        )

    return combined
