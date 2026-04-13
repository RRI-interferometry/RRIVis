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
from typing import TYPE_CHECKING, Any, Literal, TypedDict

import healpy as hp
import numpy as np

from rrivis.utils.frequency import parse_frequency_config

from ._data import HealpixData, PointSourceData
from .constants import (
    BrightnessConversion,
    brightness_temp_to_flux_density,
    flux_density_to_brightness_temp,
    rayleigh_jeans_factor,
)
from .convert import bin_sources_to_flux
from .model import SkyFormat
from .operations import materialize_healpix_model, materialize_point_sources_model
from .spectral import apply_faraday_rotation, compute_spectral_scale

if TYPE_CHECKING:
    from rrivis.core.precision import PrecisionConfig

    from .model import SkyModel

logger = logging.getLogger(__name__)

MixedModelPolicy = Literal["error", "warn", "allow"]


class CombineHealpixData(TypedDict):
    """Return type for combine_healpix."""

    healpix_maps: np.ndarray
    healpix_q_maps: np.ndarray | None
    healpix_u_maps: np.ndarray | None
    healpix_v_maps: np.ndarray | None
    healpix_nside: int
    observation_frequencies: np.ndarray
    coordinate_frame: str
    reference_frequency: float | None


def _concat_string_metadata(
    models: list[SkyModel],
    field_name: str,
) -> np.ndarray | None:
    """Concatenate per-source string metadata, filling missing values with blanks."""
    if not any(
        m.point is not None and getattr(m.point, field_name) is not None for m in models
    ):
        return None
    parts: list[np.ndarray] = []
    for model in models:
        if model.point is None:
            continue
        values = getattr(model.point, field_name)
        if values is None:
            parts.append(np.full(model.point.n_sources, "", dtype=str))
            continue
        parts.append(np.asarray(values, dtype=str))
    return np.concatenate(parts) if parts else None


def _concat_object_metadata(
    models: list[SkyModel],
    field_name: str,
) -> np.ndarray | None:
    """Concatenate per-source metadata with a permissive object dtype."""
    if not any(
        m.point is not None and getattr(m.point, field_name) is not None for m in models
    ):
        return None
    parts: list[np.ndarray] = []
    for model in models:
        if model.point is None:
            continue
        values = getattr(model.point, field_name)
        if values is None:
            parts.append(np.full(model.point.n_sources, None, dtype=object))
            continue
        parts.append(np.asarray(values, dtype=object))
    return np.concatenate(parts) if parts else None


def _concat_extra_columns(models: list[SkyModel]) -> dict[str, np.ndarray]:
    """Concatenate arbitrary metadata columns, filling missing values with None."""
    keys = sorted(
        {
            key
            for model in models
            if model.point is not None
            for key in model.point.extra_columns
        }
    )
    if not keys:
        return {}

    extra_columns: dict[str, np.ndarray] = {}
    for key in keys:
        parts: list[np.ndarray] = []
        for model in models:
            if model.point is None:
                continue
            values = model.point.extra_columns.get(key)
            if values is None:
                parts.append(np.full(model.point.n_sources, None, dtype=object))
                continue
            parts.append(np.asarray(values, dtype=object))
        extra_columns[key] = np.concatenate(parts) if parts else np.zeros(0)
    return extra_columns


def _format_healpix_freq_grid(frequencies: np.ndarray) -> str:
    """Return a compact human-readable summary of a frequency grid."""
    freqs = np.asarray(frequencies, dtype=np.float64)
    if freqs.size == 0:
        return "0 channels"
    if freqs.size == 1:
        return f"1 channel ({freqs[0] / 1e6:.3f} MHz)"
    return f"{len(freqs)} channels ({freqs[0] / 1e6:.3f}–{freqs[-1] / 1e6:.3f} MHz)"


def _resolve_requested_healpix_frequencies(
    frequencies: np.ndarray | None,
    obs_frequency_config: dict[str, Any] | None,
) -> np.ndarray | None:
    """Resolve an explicit frequency request to a concrete array."""
    if frequencies is not None and obs_frequency_config is not None:
        raise ValueError(
            "Provide either 'frequencies' or 'obs_frequency_config', not both."
        )
    if frequencies is not None:
        return np.asarray(frequencies, dtype=np.float64)
    if obs_frequency_config is not None:
        return parse_frequency_config(obs_frequency_config)
    return None


def _resolve_common_healpix_frame(models: list[SkyModel]) -> str:
    """Return the shared HEALPix frame or raise on mismatches."""
    frames = {m.healpix.coordinate_frame for m in models if m.healpix is not None}
    if not frames:
        return "icrs"
    if len(frames) != 1:
        raise ValueError(
            "Cannot combine HEALPix models with different coordinate_frame "
            f"values: {sorted(frames)}."
        )
    return next(iter(frames))


def _point_source_healpix_indices(
    point: PointSourceData,
    nside: int,
    *,
    coordinate_frame: str,
) -> np.ndarray:
    if coordinate_frame == "galactic":
        from astropy.coordinates import SkyCoord

        galactic = SkyCoord(
            ra=point.ra_rad,
            dec=point.dec_rad,
            unit="rad",
            frame="icrs",
        ).galactic
        lon_rad = galactic.l.rad
        lat_rad = galactic.b.rad
    else:
        lon_rad = point.ra_rad
        lat_rad = point.dec_rad
    return hp.ang2pix(nside, np.pi / 2 - lat_rad, lon_rad)


def _validate_requested_healpix_grid(
    models: list[SkyModel],
    nside: int | None,
    frequencies: np.ndarray | None,
) -> None:
    """Reject requests that would silently ignore an existing HEALPix grid."""
    healpix_models = [m for m in models if m.healpix is not None]
    if not healpix_models:
        return

    ref_model = healpix_models[0]
    assert ref_model.healpix is not None
    ref_nside = ref_model.healpix.nside
    ref_freqs = np.asarray(ref_model.healpix.frequencies)

    if nside is not None and nside != ref_nside:
        raise ValueError(
            "Requested HEALPix nside does not match the existing HEALPix payload: "
            f"requested nside={nside}, but model '{ref_model.model_name or 'unnamed'}' "
            f"already carries nside={ref_nside}. "
            "Regrid that model first with "
            "`regrid_healpix_model(model, nside=...)` or omit nside to keep the "
            "existing grid."
        )

    if frequencies is not None and not np.array_equal(frequencies, ref_freqs):
        raise ValueError(
            "Requested HEALPix frequency grid does not match the existing "
            f"payload in model '{ref_model.model_name or 'unnamed'}': "
            f"existing grid = {_format_healpix_freq_grid(ref_freqs)}, "
            f"requested grid = {_format_healpix_freq_grid(frequencies)}. "
            "Exact frequency regridding is not implemented yet; regrid or "
            "regenerate the HEALPix payload first."
        )


def regrid_healpix_model(
    model: SkyModel,
    *,
    nside: int | None = None,
    frequencies: np.ndarray | None = None,
    obs_frequency_config: dict[str, Any] | None = None,
) -> SkyModel:
    """Explicitly regrid a HEALPix SkyModel.

    First pass policy:
    - ``nside`` changes are supported via ``healpy.ud_grade``.
    - frequency changes are exact-only; requested frequencies must match the
      existing HEALPix axis exactly.
    """
    if model.healpix is None:
        raise ValueError("regrid_healpix_model requires a SkyModel with HEALPix data.")

    requested_freqs = _resolve_requested_healpix_frequencies(
        frequencies,
        obs_frequency_config,
    )
    source_healpix = (
        model.healpix.to_dense() if model.healpix.is_sparse else model.healpix
    )
    current_freqs = np.asarray(source_healpix.frequencies, dtype=np.float64)
    if requested_freqs is not None and not np.array_equal(
        requested_freqs, current_freqs
    ):
        raise ValueError(
            "Exact frequency regridding is not implemented yet. "
            f"Existing grid = {_format_healpix_freq_grid(current_freqs)}, "
            f"requested grid = {_format_healpix_freq_grid(requested_freqs)}."
        )

    target_nside = source_healpix.nside if nside is None else nside
    if target_nside == source_healpix.nside:
        if requested_freqs is None or np.array_equal(requested_freqs, current_freqs):
            return model
        return model._replace(
            healpix=HealpixData(
                maps=source_healpix.maps,
                nside=source_healpix.nside,
                frequencies=requested_freqs,
                coordinate_frame=source_healpix.coordinate_frame,
                hpx_inds=source_healpix.hpx_inds,
                q_maps=source_healpix.q_maps,
                u_maps=source_healpix.u_maps,
                v_maps=source_healpix.v_maps,
                i_unit=source_healpix.i_unit,
                q_unit=source_healpix.q_unit,
                u_unit=source_healpix.u_unit,
                v_unit=source_healpix.v_unit,
                i_brightness_conversion=source_healpix.i_brightness_conversion,
                q_brightness_conversion=source_healpix.q_brightness_conversion,
                u_brightness_conversion=source_healpix.u_brightness_conversion,
                v_brightness_conversion=source_healpix.v_brightness_conversion,
            )
        )

    def _regrid_rows(arr: np.ndarray) -> np.ndarray:
        rows = [
            hp.ud_grade(row, nside_out=target_nside, power=0) for row in np.asarray(arr)
        ]
        return np.stack(rows, axis=0)

    q_maps = (
        None if source_healpix.q_maps is None else _regrid_rows(source_healpix.q_maps)
    )
    u_maps = (
        None if source_healpix.u_maps is None else _regrid_rows(source_healpix.u_maps)
    )
    v_maps = (
        None if source_healpix.v_maps is None else _regrid_rows(source_healpix.v_maps)
    )

    return model._replace(
        healpix=HealpixData(
            maps=_regrid_rows(source_healpix.maps),
            nside=target_nside,
            frequencies=current_freqs if requested_freqs is None else requested_freqs,
            coordinate_frame=source_healpix.coordinate_frame,
            q_maps=q_maps,
            u_maps=u_maps,
            v_maps=v_maps,
            i_unit=source_healpix.i_unit,
            q_unit=source_healpix.q_unit,
            u_unit=source_healpix.u_unit,
            v_unit=source_healpix.v_unit,
            i_brightness_conversion=source_healpix.i_brightness_conversion,
            q_brightness_conversion=source_healpix.q_brightness_conversion,
            u_brightness_conversion=source_healpix.u_brightness_conversion,
            v_brightness_conversion=source_healpix.v_brightness_conversion,
        )
    )


# =============================================================================
# Internal helper: concat point sources
# =============================================================================


def concat_point_sources(
    models: list[SkyModel],
    reference_frequency: float | None = None,
    brightness_conversion: BrightnessConversion | str | None = None,
    precision: PrecisionConfig | None = None,
    allow_lossy_point_materialization: bool = False,
) -> dict[str, Any]:
    """Concatenate columnar arrays from multiple point-source SkyModels.

    Each model that lacks point-source arrays must either opt in to
    lossy HEALPix-to-point conversion or be excluded before calling this
    function. Empty models are silently skipped.

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
        if m.point is None and m.healpix is not None:
            if not allow_lossy_point_materialization:
                raise ValueError(
                    "Point-source combination requires converting a HEALPix-only "
                    "model to point sources, which is lossy. Re-run with "
                    "allow_lossy_point_materialization=True to opt in."
                )
            m = materialize_point_sources_model(
                m,
                frequency=reference_frequency,
                lossy=True,
            )
        if m.point is not None and not m.point.is_empty:
            populated.append(m)

    if not populated:
        from ._data import empty_source_arrays

        return {
            **empty_source_arrays(),
            "source_name": None,
            "source_id": None,
            "extra_columns": {},
            "reference_frequency": None,
        }

    # --- Required arrays ---
    ra = np.concatenate([m.point.ra_rad for m in populated if m.point is not None])
    dec = np.concatenate([m.point.dec_rad for m in populated if m.point is not None])
    flux = np.concatenate([m.point.flux for m in populated if m.point is not None])
    si = np.concatenate(
        [m.point.spectral_index for m in populated if m.point is not None]
    )
    sq = np.concatenate([m.point.stokes_q for m in populated if m.point is not None])
    su = np.concatenate([m.point.stokes_u for m in populated if m.point is not None])
    sv = np.concatenate([m.point.stokes_v for m in populated if m.point is not None])

    ref_freq_arr = np.concatenate(
        [
            m.point.ref_freq
            if m.point is not None and m.point.ref_freq is not None
            else np.full(
                m.point.n_sources if m.point is not None else 0,
                m.reference_frequency or reference_frequency or 0.0,
                dtype=np.float64,
            )
            for m in populated
        ]
    )

    n = len(ra)

    # --- Optional: rotation measure ---
    rm: np.ndarray | None = None
    if any(
        m.point is not None and m.point.rotation_measure is not None for m in populated
    ):
        rm = np.concatenate(
            [
                m.point.rotation_measure
                if m.point is not None and m.point.rotation_measure is not None
                else np.zeros(
                    m.point.n_sources if m.point is not None else 0, dtype=np.float64
                )
                for m in populated
            ]
        )

    # --- Optional: Gaussian morphology ---
    major: np.ndarray | None = None
    minor: np.ndarray | None = None
    pa: np.ndarray | None = None
    if any(m.point is not None and m.point.major_arcsec is not None for m in populated):
        major = np.concatenate(
            [
                m.point.major_arcsec
                if m.point is not None and m.point.major_arcsec is not None
                else np.zeros(
                    m.point.n_sources if m.point is not None else 0, dtype=np.float64
                )
                for m in populated
            ]
        )
        minor = np.concatenate(
            [
                m.point.minor_arcsec
                if m.point is not None and m.point.minor_arcsec is not None
                else np.zeros(
                    m.point.n_sources if m.point is not None else 0, dtype=np.float64
                )
                for m in populated
            ]
        )
        pa = np.concatenate(
            [
                m.point.pa_deg
                if m.point is not None and m.point.pa_deg is not None
                else np.zeros(
                    m.point.n_sources if m.point is not None else 0, dtype=np.float64
                )
                for m in populated
            ]
        )

    # --- Optional: spectral coefficients (may differ in N_terms) ---
    sp_coeffs: np.ndarray | None = None
    if any(
        m.point is not None and m.point.spectral_coeffs is not None for m in populated
    ):
        max_terms = max(
            m.point.spectral_coeffs.shape[1]
            for m in populated
            if m.point is not None and m.point.spectral_coeffs is not None
        )
        parts: list[np.ndarray] = []
        for m in populated:
            if m.point is None:
                continue
            n_m = m.point.n_sources
            if m.point.spectral_coeffs is not None:
                arr = m.point.spectral_coeffs
                if arr.shape[1] < max_terms:
                    pad = np.zeros((n_m, max_terms - arr.shape[1]), dtype=arr.dtype)
                    arr = np.concatenate([arr, pad], axis=1)
                parts.append(arr)
            else:
                # Default: column 0 = alpha, rest zero
                fallback = np.zeros((n_m, max_terms), dtype=np.float64)
                fallback[:, 0] = m.point.spectral_index
                parts.append(fallback)
        sp_coeffs = np.concatenate(parts, axis=0)

    ref_freq_val = reference_frequency
    if ref_freq_val is None:
        positive = ref_freq_arr[ref_freq_arr > 0]
        if positive.size > 0 and np.allclose(positive, positive[0]):
            ref_freq_val = float(positive[0])

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
        "source_name": _concat_string_metadata(populated, "source_name"),
        "source_id": _concat_object_metadata(populated, "source_id"),
        "extra_columns": _concat_extra_columns(populated),
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
    memmap_path: str | None = None,
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
    healpix_models = [m for m in models if m.healpix is not None]
    coordinate_frame = _resolve_common_healpix_frame(healpix_models)
    point_only_models = [
        m
        for m in models
        if m.healpix is None and m.point is not None and not m.point.is_empty
    ]

    for m in healpix_models:
        if m.healpix is None:
            continue
        m_nside = m.healpix.nside
        m_freqs = m.healpix.frequencies
        if m_nside != ref_nside:
            raise ValueError(
                f"Cannot combine HEALPix models with different nside values: "
                f"reference has nside={ref_nside}, model '{m.model_name}' has "
                f"nside={m_nside}. Regrid one of the models first with "
                f"`regrid_healpix_model(model, nside=...)` before combining."
            )
        if not np.array_equal(m_freqs, ref_freqs):
            raise ValueError(
                f"Cannot combine HEALPix models with different frequency grids: "
                f"reference has {_format_healpix_freq_grid(ref_freqs)}, "
                f"model '{m.model_name}' has {_format_healpix_freq_grid(m_freqs)}. "
                "Frequency interpolation is not implemented; align the grids "
                "exactly before combining."
            )

    npix = hp.nside2npix(ref_nside)
    n_freq = len(ref_freqs)
    omega_pixel = 4 * np.pi / npix

    # Collect point-source data for pixel-binning
    ps_models_data = []
    for m in point_only_models:
        if m.has_point_sources and m.point is not None:
            ipix_m = _point_source_healpix_indices(
                m.point,
                ref_nside,
                coordinate_frame=coordinate_frame,
            )
            ps_models_data.append((ipix_m, m.point.flux, m.point.spectral_index, m))

    # Check if any model has polarized maps
    any_pol = any(m.has_polarized_healpix_maps for m in healpix_models) or any(
        m.point is not None
        and (
            np.any(m.point.stokes_q != 0)
            or np.any(m.point.stokes_u != 0)
            or np.any(m.point.stokes_v != 0)
        )
        for m in point_only_models
    )

    # Resolve output dtype from precision config
    hp_dtype = (
        precision.sky_model.get_dtype("healpix_maps")
        if precision is not None
        else np.float32
    )

    # Pre-allocate output arrays: shape (n_freq, npix)
    from ._allocation import allocate_cube, ensure_scratch_dir, finalize_cube

    scratch = ensure_scratch_dir(memmap_path) if memmap_path is not None else None
    combined_I = allocate_cube((n_freq, npix), hp_dtype, scratch, "i_maps")
    combined_Q: np.ndarray | None = (
        allocate_cube((n_freq, npix), hp_dtype, scratch, "q_maps") if any_pol else None
    )
    combined_U: np.ndarray | None = (
        allocate_cube((n_freq, npix), hp_dtype, scratch, "u_maps") if any_pol else None
    )
    combined_V: np.ndarray | None = (
        allocate_cube((n_freq, npix), hp_dtype, scratch, "v_maps") if any_pol else None
    )

    # Determine if we can use the RJ fast path (T_b linearly additive)
    is_rj = brightness_conversion == BrightnessConversion.RAYLEIGH_JEANS or (
        isinstance(brightness_conversion, str)
        and brightness_conversion == "rayleigh-jeans"
    )

    for freq_idx, freq_hz in enumerate(ref_freqs):
        rj_factor = rayleigh_jeans_factor(freq_hz, omega_pixel)

        if is_rj:
            # --- RJ fast path: T_b is linearly additive ---
            combined_T_b = np.zeros(npix, dtype=np.float64)

            # Add healpix T_b maps directly
            for m in healpix_models:
                if m.healpix is not None:
                    pixel_indices = m.healpix.pixel_indices
                    combined_T_b[pixel_indices] += m.healpix.maps[freq_idx].astype(
                        np.float64
                    )

            # Add point-source contributions (flux → T_b via RJ factor)
            for ipix_m, flux_ref_m, alpha_m, m_obj in ps_models_data:
                ps_ref_freq = (
                    m_obj.point.ref_freq
                    if m_obj.point is not None and m_obj.point.ref_freq is not None
                    else np.full(
                        len(flux_ref_m),
                        m_obj.reference_frequency or ref_frequency,
                        dtype=np.float64,
                    )
                )
                flux_map = bin_sources_to_flux(
                    ipix_m,
                    flux_ref_m,
                    alpha_m,
                    m_obj.point.spectral_coeffs if m_obj.point is not None else None,
                    float(freq_hz),
                    ps_ref_freq,
                    npix,
                )
                # Jy → K_RJ: divide by RJ factor
                if rj_factor != 0:
                    combined_T_b += flux_map / rj_factor

            combined_I[freq_idx] = combined_T_b.astype(hp_dtype)

            # Polarization (always RJ for Q/U/V)
            if any_pol:
                combined_q_T = np.zeros(npix, dtype=np.float64)
                combined_u_T = np.zeros(npix, dtype=np.float64)
                combined_v_T = np.zeros(npix, dtype=np.float64)

                for m in healpix_models:
                    if m.has_polarized_healpix_maps:
                        pixel_indices = m.healpix.pixel_indices
                        if m.healpix.q_maps is not None:
                            combined_q_T[pixel_indices] += m.healpix.q_maps[
                                freq_idx
                            ].astype(np.float64)
                        if m.healpix.u_maps is not None:
                            combined_u_T[pixel_indices] += m.healpix.u_maps[
                                freq_idx
                            ].astype(np.float64)
                        if m.healpix.v_maps is not None:
                            combined_v_T[pixel_indices] += m.healpix.v_maps[
                                freq_idx
                            ].astype(np.float64)

                for ipix_m, flux_ref_m, alpha_m, m_obj in ps_models_data:
                    if m_obj.point is not None:
                        ps_ref_freq = (
                            m_obj.point.ref_freq
                            if m_obj.point.ref_freq is not None
                            else np.full(
                                len(flux_ref_m),
                                m_obj.reference_frequency or ref_frequency,
                                dtype=np.float64,
                            )
                        )
                        scale = compute_spectral_scale(
                            alpha_m,
                            m_obj.point.spectral_coeffs,
                            float(freq_hz),
                            ps_ref_freq,
                        )
                        q_f, u_f = apply_faraday_rotation(
                            m_obj.point.stokes_q,
                            m_obj.point.stokes_u,
                            m_obj.point.rotation_measure,
                            float(freq_hz),
                            ps_ref_freq,
                            scale,
                        )
                        rj_inv = 1.0 / rj_factor if rj_factor != 0 else 0.0
                        combined_q_T += (
                            np.bincount(ipix_m, weights=q_f, minlength=npix) * rj_inv
                        )
                        combined_u_T += (
                            np.bincount(ipix_m, weights=u_f, minlength=npix) * rj_inv
                        )
                        combined_v_T += (
                            np.bincount(
                                ipix_m,
                                weights=m_obj.point.stokes_v * scale,
                                minlength=npix,
                            )
                            * rj_inv
                        )

                combined_Q[freq_idx] = combined_q_T.astype(hp_dtype)
                combined_U[freq_idx] = combined_u_T.astype(hp_dtype)
                combined_V[freq_idx] = combined_v_T.astype(hp_dtype)

        else:
            # --- Planck path: must round-trip through Jy (non-linear) ---
            combined_flux = np.zeros(npix, dtype=np.float64)
            combined_q_flux = np.zeros(npix, dtype=np.float64) if any_pol else None
            combined_u_flux = np.zeros(npix, dtype=np.float64) if any_pol else None
            combined_v_flux = np.zeros(npix, dtype=np.float64) if any_pol else None

            # Add healpix_map models
            for m in healpix_models:
                if m.healpix is not None:
                    t_map = m.healpix.maps[freq_idx].astype(np.float64)
                    pixel_indices = m.healpix.pixel_indices
                    pos = t_map > 0
                    if np.any(pos):
                        combined_flux[pixel_indices[pos]] += (
                            brightness_temp_to_flux_density(
                                t_map[pos],
                                freq_hz,
                                omega_pixel,
                                method=brightness_conversion,
                            )
                        )

                    if any_pol and m.has_polarized_healpix_maps:
                        if m.healpix.q_maps is not None:
                            q_t = m.healpix.q_maps[freq_idx]
                            if q_t is not None:
                                combined_q_flux[pixel_indices] += (
                                    q_t.astype(np.float64) * rj_factor
                                )
                        if m.healpix.u_maps is not None:
                            u_t = m.healpix.u_maps[freq_idx]
                            if u_t is not None:
                                combined_u_flux[pixel_indices] += (
                                    u_t.astype(np.float64) * rj_factor
                                )
                        if m.healpix.v_maps is not None:
                            v_t = m.healpix.v_maps[freq_idx]
                            if v_t is not None:
                                combined_v_flux[pixel_indices] += (
                                    v_t.astype(np.float64) * rj_factor
                                )

            # Add point-source models via bincount
            for ipix_m, flux_ref_m, alpha_m, m_obj in ps_models_data:
                ps_ref_freq = (
                    m_obj.point.ref_freq
                    if m_obj.point is not None and m_obj.point.ref_freq is not None
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
                    m_obj.point.spectral_coeffs if m_obj.point is not None else None,
                    float(freq_hz),
                    ps_ref_freq,
                    npix,
                )

                if any_pol and m_obj.point is not None:
                    scale = compute_spectral_scale(
                        alpha_m,
                        m_obj.point.spectral_coeffs,
                        float(freq_hz),
                        ps_ref_freq,
                    )
                    q_f, u_f = apply_faraday_rotation(
                        m_obj.point.stokes_q,
                        m_obj.point.stokes_u,
                        m_obj.point.rotation_measure,
                        float(freq_hz),
                        ps_ref_freq,
                        scale,
                    )
                    combined_q_flux += np.bincount(ipix_m, weights=q_f, minlength=npix)
                    combined_u_flux += np.bincount(ipix_m, weights=u_f, minlength=npix)
                    combined_v_flux += np.bincount(
                        ipix_m,
                        weights=m_obj.point.stokes_v * scale,
                        minlength=npix,
                    )

            # Convert combined flux back to brightness temperature
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

    # Flush and re-open read-only if memmap-backed.
    combined_I = finalize_cube(combined_I, scratch, "i_maps")
    if combined_Q is not None:
        combined_Q = finalize_cube(combined_Q, scratch, "q_maps")
    if combined_U is not None:
        combined_U = finalize_cube(combined_U, scratch, "u_maps")
    if combined_V is not None:
        combined_V = finalize_cube(combined_V, scratch, "v_maps")

    return {
        "healpix_maps": combined_I,
        "healpix_q_maps": combined_Q,
        "healpix_u_maps": combined_U,
        "healpix_v_maps": combined_V,
        "healpix_nside": ref_nside,
        "observation_frequencies": ref_freqs,
        "coordinate_frame": coordinate_frame,
        "reference_frequency": None,
    }


# =============================================================================
# Combination helpers (private)
# =============================================================================


def _resolve_combination_params(
    models: list[SkyModel],
    representation: SkyFormat | str | None,
    frequency: float | None,
    ref_frequency: float | None,
) -> tuple[SkyFormat, float | None, float | None]:
    """Auto-detect representation and resolve frequency defaults.

    Returns (representation, frequency, ref_frequency).
    """
    # Coerce string to SkyFormat
    if isinstance(representation, str) and not isinstance(representation, SkyFormat):
        representation = SkyFormat(representation)

    # Auto-detect representation
    if representation is None:
        representation = (
            SkyFormat.HEALPIX
            if any(m.healpix is not None for m in models)
            else SkyFormat.POINT_SOURCES
        )

    freq = frequency
    if ref_frequency is None:
        ref_frequency = freq

    return representation, freq, ref_frequency


def _check_combination_issues(
    models: list[SkyModel],
    mixed_model_policy: MixedModelPolicy,
) -> None:
    """Validate or warn on combination policies."""
    has_catalog = any(
        m.point is not None
        and not m.point.is_empty
        and m.source_format == SkyFormat.POINT_SOURCES
        for m in models
    )
    has_diffuse = any(
        m.healpix is not None and m.source_format == SkyFormat.HEALPIX for m in models
    )
    if has_catalog and has_diffuse:
        message = (
            "Combining catalog sources (GLEAM/MALS) with diffuse models (GSM/LFSM/Haslam) "
            "may result in double-counting of bright sources. Diffuse models already include "
            "integrated emission from bright sources. Consider using only one model type "
            "or set sky_model.mixed_model_policy='warn' or 'allow' to override this guard."
        )
        if mixed_model_policy == "error":
            raise ValueError(message)
        if mixed_model_policy == "warn":
            warnings.warn(message, UserWarning, stacklevel=3)


def _resolve_brightness_conversion(
    models: list[SkyModel],
    requested: BrightnessConversion | str | None,
) -> BrightnessConversion:
    """Resolve output brightness conversion without silent clobbering."""
    values = {m.brightness_conversion for m in models}
    if requested is None:
        if not values:
            return BrightnessConversion.PLANCK
        if len(values) == 1:
            return next(iter(values))
        raise ValueError(
            "Cannot combine sky models with different brightness_conversion "
            f"settings without an explicit brightness_conversion target: {values}."
        )
    return BrightnessConversion(requested)


def _combine_as_healpix_merge(
    models: list[SkyModel],
    ref_frequency: float | None,
    brightness_conversion: BrightnessConversion,
    precision: PrecisionConfig | None,
    memmap_path: str | None = None,
) -> SkyModel:
    """Combine models with existing HEALPix maps via Jy-space addition."""
    from .model import SkyModel

    ref_model = next(m for m in models if m.healpix is not None)
    ref_nside = ref_model.healpix.nside
    ref_freqs = ref_model.healpix.frequencies

    data = combine_healpix(
        models,
        ref_nside=ref_nside,
        ref_freqs=ref_freqs,
        ref_frequency=ref_frequency,
        brightness_conversion=brightness_conversion,
        precision=precision,
        memmap_path=memmap_path,
    )

    return SkyModel(
        healpix=HealpixData(
            maps=data["healpix_maps"],
            nside=data["healpix_nside"],
            frequencies=data["observation_frequencies"],
            coordinate_frame=data["coordinate_frame"],
            q_maps=data["healpix_q_maps"],
            u_maps=data["healpix_u_maps"],
            v_maps=data["healpix_v_maps"],
            i_brightness_conversion=brightness_conversion.value,
        ),
        source_format=SkyFormat.HEALPIX,
        reference_frequency=data["reference_frequency"],
        model_name="combined",
        brightness_conversion=brightness_conversion,
        _precision=precision,
    )


def _combine_as_point_sources(
    models: list[SkyModel],
    frequency: float | None,
    brightness_conversion: BrightnessConversion,
    precision: PrecisionConfig | None,
    allow_lossy_point_materialization: bool,
) -> SkyModel:
    """Combine models by concatenating point-source arrays."""
    from .model import SkyModel

    data = concat_point_sources(
        models,
        reference_frequency=frequency,
        brightness_conversion=brightness_conversion,
        precision=precision,
        allow_lossy_point_materialization=allow_lossy_point_materialization,
    )

    return SkyModel(
        point=PointSourceData(
            ra_rad=data["ra_rad"],
            dec_rad=data["dec_rad"],
            flux=data["flux"],
            spectral_index=data["spectral_index"],
            stokes_q=data["stokes_q"],
            stokes_u=data["stokes_u"],
            stokes_v=data["stokes_v"],
            ref_freq=data["ref_freq"],
            rotation_measure=data["rotation_measure"],
            major_arcsec=data["major_arcsec"],
            minor_arcsec=data["minor_arcsec"],
            pa_deg=data["pa_deg"],
            spectral_coeffs=data["spectral_coeffs"],
            source_name=data["source_name"],
            source_id=data["source_id"],
            extra_columns=data["extra_columns"],
        ),
        source_format=SkyFormat.POINT_SOURCES,
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
    representation: SkyFormat | str | None = None,
    nside: int | None = None,
    frequency: float | None = None,
    frequencies: np.ndarray | None = None,
    obs_frequency_config: dict[str, Any] | None = None,
    ref_frequency: float | None = None,
    brightness_conversion: BrightnessConversion | str | None = None,
    allow_lossy_point_materialization: bool = False,
    mixed_model_policy: MixedModelPolicy = "error",
    precision: PrecisionConfig | None = None,
    memmap_path: str | None = None,
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
    nside : int, optional
        HEALPix NSIDE for ``healpix_map`` output.
    frequency : float, optional
        Frequency for HEALPix-to-point-source conversions.
    frequencies : np.ndarray, optional
        Frequency array for point-to-HEALPix conversion when no input
        model already carries HEALPix maps.
    obs_frequency_config : dict, optional
        Frequency config fallback for point-to-HEALPix conversion.
    ref_frequency : float, optional
        Reference frequency for spectral extrapolation (Hz).
    brightness_conversion : str or BrightnessConversion, optional
        Output brightness conversion method. When omitted, all inputs must
        already agree; otherwise an explicit target is required.
    allow_lossy_point_materialization : bool, default False
        Allow lossy HEALPix-to-point conversion when point-source output
        is requested.
    mixed_model_policy : {"error", "warn", "allow"}, default "error"
        Policy for combining point catalogs with diffuse HEALPix models.
    precision : PrecisionConfig, optional
        Precision configuration for the combined model.
    memmap_path : str or None, optional
        If given, stream the combined HEALPix cube to memory-mapped files
        at this directory (created if needed) rather than allocating it
        in RAM.  Only affects HEALPix output paths.

    Returns
    -------
    SkyModel
        Combined sky model.
    """
    if not models:
        from ._factories import create_empty

        return create_empty(
            model_name="combined_empty",
            brightness_conversion=(
                BrightnessConversion.PLANCK
                if brightness_conversion is None
                else BrightnessConversion(brightness_conversion)
            ),
            precision=precision,
        )

    brightness_conversion = _resolve_brightness_conversion(
        models, brightness_conversion
    )
    _check_combination_issues(models, mixed_model_policy)
    requested_freqs = _resolve_requested_healpix_frequencies(
        frequencies, obs_frequency_config
    )
    representation, freq, ref_freq = _resolve_combination_params(
        models, representation, frequency, ref_frequency
    )

    has_healpix_map = any(m.healpix is not None for m in models)

    if representation == SkyFormat.HEALPIX and has_healpix_map:
        _validate_requested_healpix_grid(models, nside, requested_freqs)

    # Path 1: HEALPix merge (at least one model already has maps)
    if representation == SkyFormat.HEALPIX and has_healpix_map:
        return _combine_as_healpix_merge(
            models,
            ref_freq,
            brightness_conversion,
            precision,
            memmap_path=memmap_path,
        )

    # Path 2: Concatenate as point sources
    combined = _combine_as_point_sources(
        models,
        freq,
        brightness_conversion,
        precision,
        allow_lossy_point_materialization,
    )

    # Path 3: Optionally convert concatenated PS to HEALPix
    if representation == SkyFormat.HEALPIX:
        if requested_freqs is None:
            raise ValueError(
                "healpix_map output requires 'frequencies' or "
                "'obs_frequency_config' when no input model already carries "
                "HEALPix maps."
            )
        combined = materialize_healpix_model(
            combined,
            nside=64 if nside is None else nside,
            frequencies=requested_freqs,
            ref_frequency=ref_freq,
            memmap_path=memmap_path,
        )

    return combined
