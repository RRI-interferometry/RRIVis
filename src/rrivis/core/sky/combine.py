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
from typing import TYPE_CHECKING, Any

import healpy as hp
import numpy as np

from .constants import (
    C_LIGHT,
    K_BOLTZMANN,
    brightness_temp_to_flux_density,
    flux_density_to_brightness_temp,
)
from .spectral import apply_faraday_rotation, compute_spectral_scale

if TYPE_CHECKING:
    from rrivis.core.precision import PrecisionConfig

    from .model import SkyModel

logger = logging.getLogger(__name__)


# =============================================================================
# Internal helper: concat point sources
# =============================================================================


def concat_point_sources(
    models: list[SkyModel],
    frequency: float | None = None,
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
    frequency : float, optional
        Frequency for healpix-to-point-source conversion.
    brightness_conversion : str, default ``"planck"``
        Brightness conversion method (carried through to the result).
    precision : PrecisionConfig, optional
        Precision configuration (not applied here -- the caller handles
        dtype casting via the SkyModel constructor).

    Returns
    -------
    dict
        Raw data dict with keys matching SkyModel constructor fields:
        ``_ra_rad``, ``_dec_rad``, ``_flux_ref``, ``_alpha``,
        ``_stokes_q``, ``_stokes_u``, ``_stokes_v``,
        ``_rotation_measure``, ``_major_arcsec``, ``_minor_arcsec``,
        ``_pa_deg``, ``_spectral_coeffs``, ``frequency``.
        Array values are ``np.ndarray``; optional fields are ``None``
        when no model contributes data.  An empty-model result has
        zero-length arrays.
    """
    # Ensure each model has point-source arrays populated; skip empties
    populated: list[SkyModel] = []
    for m in models:
        if m._ra_rad is None:
            m = m.with_representation("point_sources", frequency=frequency)
        if m._ra_rad is not None and len(m._ra_rad) > 0:
            populated.append(m)

    if not populated:
        return {
            "_ra_rad": np.zeros(0, dtype=np.float64),
            "_dec_rad": np.zeros(0, dtype=np.float64),
            "_flux_ref": np.zeros(0, dtype=np.float64),
            "_alpha": np.zeros(0, dtype=np.float64),
            "_stokes_q": np.zeros(0, dtype=np.float64),
            "_stokes_u": np.zeros(0, dtype=np.float64),
            "_stokes_v": np.zeros(0, dtype=np.float64),
            "_rotation_measure": None,
            "_major_arcsec": None,
            "_minor_arcsec": None,
            "_pa_deg": None,
            "_spectral_coeffs": None,
            "frequency": None,
        }

    # --- Required arrays ---
    ra = np.concatenate([m._ra_rad for m in populated])
    dec = np.concatenate([m._dec_rad for m in populated])
    flux = np.concatenate([m._flux_ref for m in populated])
    alpha = np.concatenate([m._alpha for m in populated])
    sq = np.concatenate([m._stokes_q for m in populated])
    su = np.concatenate([m._stokes_u for m in populated])
    sv = np.concatenate([m._stokes_v for m in populated])

    n = len(ra)

    # --- Optional: rotation measure ---
    rm: np.ndarray | None = None
    if any(m._rotation_measure is not None for m in populated):
        rm = np.concatenate(
            [
                m._rotation_measure
                if m._rotation_measure is not None
                else np.zeros(len(m._ra_rad), dtype=np.float64)
                for m in populated
            ]
        )

    # --- Optional: Gaussian morphology ---
    major: np.ndarray | None = None
    minor: np.ndarray | None = None
    pa: np.ndarray | None = None
    if any(m._major_arcsec is not None for m in populated):
        major = np.concatenate(
            [
                m._major_arcsec
                if m._major_arcsec is not None
                else np.zeros(len(m._ra_rad), dtype=np.float64)
                for m in populated
            ]
        )
        minor = np.concatenate(
            [
                m._minor_arcsec
                if m._minor_arcsec is not None
                else np.zeros(len(m._ra_rad), dtype=np.float64)
                for m in populated
            ]
        )
        pa = np.concatenate(
            [
                m._pa_deg
                if m._pa_deg is not None
                else np.zeros(len(m._ra_rad), dtype=np.float64)
                for m in populated
            ]
        )

    # --- Optional: spectral coefficients (may differ in N_terms) ---
    sp_coeffs: np.ndarray | None = None
    if any(m._spectral_coeffs is not None for m in populated):
        max_terms = max(
            m._spectral_coeffs.shape[1]
            for m in populated
            if m._spectral_coeffs is not None
        )
        parts: list[np.ndarray] = []
        for m in populated:
            n_m = len(m._ra_rad)
            if m._spectral_coeffs is not None:
                arr = m._spectral_coeffs
                if arr.shape[1] < max_terms:
                    pad = np.zeros((n_m, max_terms - arr.shape[1]), dtype=arr.dtype)
                    arr = np.concatenate([arr, pad], axis=1)
                parts.append(arr)
            else:
                # Default: column 0 = alpha, rest zero
                fallback = np.zeros((n_m, max_terms), dtype=np.float64)
                fallback[:, 0] = m._alpha
                parts.append(fallback)
        sp_coeffs = np.concatenate(parts, axis=0)

    # Infer frequency from models if not provided
    freq = frequency
    if freq is None:
        for m in populated:
            if m.frequency is not None:
                freq = m.frequency
                break

    logger.info(f"Concatenated {len(populated)} models: {n} total sources")

    return {
        "_ra_rad": ra,
        "_dec_rad": dec,
        "_flux_ref": flux,
        "_alpha": alpha,
        "_stokes_q": sq,
        "_stokes_u": su,
        "_stokes_v": sv,
        "_rotation_measure": rm,
        "_major_arcsec": major,
        "_minor_arcsec": minor,
        "_pa_deg": pa,
        "_spectral_coeffs": sp_coeffs,
        "frequency": freq,
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
) -> dict[str, Any]:
    """Combine models by element-wise addition in Jy space per frequency channel.

    All ``healpix_multifreq`` models must share the same nside and frequency
    grid.  Point-source models are binned into the same grid via
    ``np.bincount``.  Stokes I is converted T_b -> Jy -> T_b so that
    addition is physically correct under both Planck and Rayleigh-Jeans
    conversions.

    Parameters
    ----------
    models : list of SkyModel
        Models to combine.
    ref_nside : int
        Common HEALPix nside (from first ``healpix_multifreq`` model).
    ref_freqs : np.ndarray
        Common frequency grid in Hz, shape ``(n_freq,)``.
    ref_frequency : float or None
        Reference frequency for spectral extrapolation of point sources.
    brightness_conversion : str, default ``"planck"``
        Brightness conversion method.
    precision : PrecisionConfig, optional
        Precision configuration (not applied here -- the caller handles
        dtype casting).

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
    # Validate all healpix_multifreq models share the same nside and
    # frequency grid before doing element-wise arithmetic.
    for m in models:
        if m.mode != "healpix_multifreq":
            continue
        m_nside = m._healpix_nside
        m_freqs = m._observation_frequencies
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
        if m.mode == "point_sources" and m._has_point_sources():
            ipix_m = hp.ang2pix(ref_nside, np.pi / 2 - m._dec_rad, m._ra_rad)
            ps_models_data.append((ipix_m, m._flux_ref, m._alpha, m))

    # Check if any model has polarized maps
    any_pol = any(
        m.has_polarized_healpix_maps for m in models if m.mode == "healpix_multifreq"
    ) or any(
        m._stokes_q is not None
        and (
            np.any(m._stokes_q != 0)
            or np.any(m._stokes_u != 0)
            or np.any(m._stokes_v != 0)
        )
        for m in models
        if m.mode == "point_sources" and m._has_point_sources()
    )

    # Pre-allocate output arrays: shape (n_freq, npix)
    combined_I = np.zeros((n_freq, npix), dtype=np.float32)
    combined_Q: np.ndarray | None = (
        np.zeros((n_freq, npix), dtype=np.float32) if any_pol else None
    )
    combined_U: np.ndarray | None = (
        np.zeros((n_freq, npix), dtype=np.float32) if any_pol else None
    )
    combined_V: np.ndarray | None = (
        np.zeros((n_freq, npix), dtype=np.float32) if any_pol else None
    )

    for freq_idx, freq_hz in enumerate(ref_freqs):
        combined_flux = np.zeros(npix, dtype=np.float64)
        combined_q_flux = np.zeros(npix, dtype=np.float64) if any_pol else None
        combined_u_flux = np.zeros(npix, dtype=np.float64) if any_pol else None
        combined_v_flux = np.zeros(npix, dtype=np.float64) if any_pol else None

        rj_factor = (2 * K_BOLTZMANN * freq_hz**2 / C_LIGHT**2) * omega_pixel / 1e-26

        # --- Add healpix_multifreq models ---
        for m in models:
            if m.mode == "healpix_multifreq":
                t_map = m._healpix_maps[freq_idx].astype(np.float64)
                pos = t_map > 0
                if np.any(pos):
                    combined_flux[pos] += brightness_temp_to_flux_density(
                        t_map[pos],
                        freq_hz,
                        omega_pixel,
                        method=brightness_conversion,
                    )

                if any_pol and m.has_polarized_healpix_maps:
                    if m._healpix_q_maps is not None:
                        q_t = m._healpix_q_maps[freq_idx]
                        if q_t is not None:
                            combined_q_flux += q_t.astype(np.float64) * rj_factor
                    if m._healpix_u_maps is not None:
                        u_t = m._healpix_u_maps[freq_idx]
                        if u_t is not None:
                            combined_u_flux += u_t.astype(np.float64) * rj_factor
                    if m._healpix_v_maps is not None:
                        v_t = m._healpix_v_maps[freq_idx]
                        if v_t is not None:
                            combined_v_flux += v_t.astype(np.float64) * rj_factor

        # --- Add point-source models via bincount ---
        for ipix_m, flux_ref_m, alpha_m, m_obj in ps_models_data:
            scale = compute_spectral_scale(
                alpha_m, m_obj._spectral_coeffs, float(freq_hz), ref_frequency
            )
            flux_at_f = flux_ref_m * scale
            combined_flux += np.bincount(ipix_m, weights=flux_at_f, minlength=npix)

            if any_pol and m_obj._stokes_q is not None:
                q_f, u_f = apply_faraday_rotation(
                    m_obj._stokes_q,
                    m_obj._stokes_u,
                    m_obj._rotation_measure,
                    float(freq_hz),
                    ref_frequency,
                    scale,
                )
                combined_q_flux += np.bincount(ipix_m, weights=q_f, minlength=npix)
                combined_u_flux += np.bincount(ipix_m, weights=u_f, minlength=npix)
                combined_v_flux += np.bincount(
                    ipix_m, weights=m_obj._stokes_v * scale, minlength=npix
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
        combined_I[freq_idx] = combined_T_b.astype(np.float32)

        if any_pol:
            rj_inv = 1.0 / rj_factor if rj_factor != 0 else 0.0
            combined_Q[freq_idx] = (combined_q_flux * rj_inv).astype(np.float32)
            combined_U[freq_idx] = (combined_u_flux * rj_inv).astype(np.float32)
            combined_V[freq_idx] = (combined_v_flux * rj_inv).astype(np.float32)

    freq = float(ref_freqs[0]) if len(ref_freqs) > 0 else None

    logger.info(
        f"Combined {len(models)} models into healpix_multifreq "
        f"({n_freq} channels, nside={ref_nside}"
        f"{', stokes=IQUV' if any_pol else ''})"
    )

    return {
        "_healpix_maps": combined_I,
        "_healpix_q_maps": combined_Q,
        "_healpix_u_maps": combined_U,
        "_healpix_v_maps": combined_V,
        "_healpix_nside": ref_nside,
        "_observation_frequencies": ref_freqs,
        "frequency": freq,
    }


# =============================================================================
# Public API: combine_models
# =============================================================================


def combine_models(
    models: list[SkyModel],
    representation: str = "point_sources",
    nside: int = 64,
    frequency: float | None = None,
    obs_frequency_config: dict[str, Any] | None = None,
    ref_frequency: float | None = None,
    brightness_conversion: str = "planck",
    precision: PrecisionConfig | None = None,
) -> SkyModel:
    """Combine multiple sky models into one.

    The combination always works by first concatenating all models as point
    sources, preserving each source's individual properties (flux, spectral
    index, coordinates). If ``healpix_map`` representation is requested, the
    concatenated sources are then converted to multi-frequency HEALPix maps.

    Parameters
    ----------
    models : list of SkyModel
        Sky models to combine.
    representation : str, default ``"point_sources"``
        Output representation: ``"point_sources"`` or ``"healpix_map"``.
    nside : int, default 64
        HEALPix NSIDE for ``healpix_map`` output.
    frequency : float, optional
        Frequency for HEALPix-to-point-source conversions.
    obs_frequency_config : dict, optional
        Required for ``healpix_map`` representation when no model already
        has multi-frequency maps. Observation frequency configuration with
        keys: ``starting_frequency``, ``frequency_interval``,
        ``frequency_bandwidth``, ``frequency_unit``.
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

    Raises
    ------
    ValueError
        If ``healpix_map`` representation is requested without
        ``obs_frequency_config`` and no model already has multi-frequency
        maps.

    Warns
    -----
    UserWarning
        When combining catalog sources (GLEAM, MALS) with diffuse models
        (GSM), as this can result in double-counting of bright sources.

    Notes
    -----
    **How concatenation works**:

    Each model is converted to point sources, then all sources are collected
    into a single list.  Each source retains its individual properties:

    - RA, Dec position (radians)
    - flux (Jy at reference frequency)
    - spectral index (power-law exponent)
    - Stokes Q, U, V (polarization)

    No averaging occurs during concatenation -- if two sources happen to be
    at similar positions, they remain as separate sources.  This preserves
    the correct spectral behaviour when later converting to HEALPix maps.

    **Double-counting warning**: Diffuse sky models (GSM, LFSM, Haslam)
    include integrated emission from all sources, including bright ones that
    also appear in catalogs like GLEAM.  Naive combination will
    double-count these sources.

    Examples
    --------
    >>> # Combine as point sources (default)
    >>> combined = combine_models([gleam, test])

    >>> # Combine and convert to multi-frequency HEALPix
    >>> obs_config = {
    ...     "starting_frequency": 100.0,
    ...     "frequency_interval": 1.0,
    ...     "frequency_bandwidth": 20.0,
    ...     "frequency_unit": "MHz",
    ... }
    >>> combined = combine_models(
    ...     [gleam, test],
    ...     representation="healpix_map",
    ...     nside=64,
    ...     obs_frequency_config=obs_config,
    ... )
    """
    # Late import to break circular dependency
    from .model import SkyModel

    if not models:
        return SkyModel(
            _ra_rad=np.zeros(0, dtype=np.float64),
            _dec_rad=np.zeros(0, dtype=np.float64),
            _flux_ref=np.zeros(0, dtype=np.float64),
            _alpha=np.zeros(0, dtype=np.float64),
            _stokes_q=np.zeros(0, dtype=np.float64),
            _stokes_u=np.zeros(0, dtype=np.float64),
            _stokes_v=np.zeros(0, dtype=np.float64),
            model_name="combined_empty",
            brightness_conversion=brightness_conversion,
            _precision=precision,
        )

    # Warn about potential double-counting
    has_catalog = any(
        m.mode == "point_sources" and m._has_point_sources() for m in models
    )
    has_diffuse = any(m.mode == "healpix_multifreq" for m in models)

    if has_catalog and has_diffuse:
        warnings.warn(
            "Combining catalog sources (GLEAM/MALS) with diffuse models (GSM/LFSM/Haslam) "
            "may result in double-counting of bright sources. Diffuse models already include "
            "integrated emission from bright sources. Consider using only one model type "
            "or implementing source subtraction for accurate results.",
            UserWarning,
            stacklevel=2,
        )

    # Resolve frequency from model metadata if not provided
    freq = frequency
    if freq is None:
        for m in models:
            if m.frequency is not None:
                freq = m.frequency
                break

    # Resolve ref_frequency from model metadata if not provided
    if ref_frequency is None:
        for m in models:
            if m.frequency is not None:
                ref_frequency = m.frequency
                break
        if ref_frequency is None:
            ref_frequency = freq  # last resort

    has_healpix_multifreq = any(m.mode == "healpix_multifreq" for m in models)

    # ==================================================================
    # HEALPIX_MAP REPRESENTATION WITH MULTIFREQ MODELS
    # ==================================================================
    if representation == "healpix_map" and has_healpix_multifreq:
        ref_model = next(m for m in models if m.mode == "healpix_multifreq")
        ref_nside = ref_model._healpix_nside
        ref_freqs = ref_model._observation_frequencies

        data = combine_healpix(
            models,
            ref_nside=ref_nside,
            ref_freqs=ref_freqs,
            ref_frequency=ref_frequency,
            brightness_conversion=brightness_conversion,
            precision=precision,
        )

        return SkyModel(
            _healpix_maps=data["_healpix_maps"],
            _healpix_q_maps=data["_healpix_q_maps"],
            _healpix_u_maps=data["_healpix_u_maps"],
            _healpix_v_maps=data["_healpix_v_maps"],
            _healpix_nside=data["_healpix_nside"],
            _observation_frequencies=data["_observation_frequencies"],
            _native_format="healpix",
            frequency=data["frequency"],
            model_name="combined",
            brightness_conversion=brightness_conversion,
            _precision=precision,
        )

    # ==================================================================
    # ALL OTHER CASES: Concatenate as point sources first
    # ==================================================================
    data = concat_point_sources(
        models,
        frequency=freq,
        brightness_conversion=brightness_conversion,
        precision=precision,
    )

    combined = SkyModel(
        _ra_rad=data["_ra_rad"],
        _dec_rad=data["_dec_rad"],
        _flux_ref=data["_flux_ref"],
        _alpha=data["_alpha"],
        _stokes_q=data["_stokes_q"],
        _stokes_u=data["_stokes_u"],
        _stokes_v=data["_stokes_v"],
        _rotation_measure=data["_rotation_measure"],
        _major_arcsec=data["_major_arcsec"],
        _minor_arcsec=data["_minor_arcsec"],
        _pa_deg=data["_pa_deg"],
        _spectral_coeffs=data["_spectral_coeffs"],
        _native_format="point_sources",
        model_name="combined",
        frequency=data["frequency"],
        brightness_conversion=brightness_conversion,
        _precision=precision,
    )

    if representation == "healpix_map":
        if obs_frequency_config is None:
            raise ValueError(
                "obs_frequency_config is required for healpix_map representation. "
                "Provide a dict with: starting_frequency, frequency_interval, "
                "frequency_bandwidth, frequency_unit."
            )

        combined = combined.with_healpix_maps(
            nside=nside,
            obs_frequency_config=obs_frequency_config,
            ref_frequency=ref_frequency,
        )

    return combined
