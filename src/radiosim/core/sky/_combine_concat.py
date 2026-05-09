# radiosim/core/sky/_combine_concat.py
"""Point-source concatenation and extra-column metadata helpers."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np

from ._data import empty_source_arrays
from .constants import BrightnessConversion
from .operations import materialize_point_sources_model
from .spectral import per_source_reference_frequencies

if TYPE_CHECKING:
    from radiosim.core.precision import PrecisionConfig

    from .model import SkyModel

logger = logging.getLogger(__name__)


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
    """Concatenate arbitrary metadata columns across models.

    Per-column dtype is preserved when possible: if every populated part has a
    compatible numeric dtype (via ``np.result_type``) and missing values can be
    represented in that dtype (i.e. floating: filled with NaN), the resulting
    column keeps its native dtype. When a column is absent from any contributing
    model and the resolved dtype cannot represent a sentinel (e.g. integer,
    bool), the column falls back to ``dtype=object`` filled with ``None``.
    """
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
        present_arrays: list[np.ndarray] = []
        contributing_models: list[SkyModel] = []
        any_missing = False
        for model in models:
            if model.point is None:
                continue
            contributing_models.append(model)
            values = model.point.extra_columns.get(key)
            if values is None:
                any_missing = True
                continue
            present_arrays.append(np.asarray(values))

        if not present_arrays:
            extra_columns[key] = np.zeros(0)
            continue

        try:
            common_dtype = np.result_type(*[a.dtype for a in present_arrays])
        except TypeError:
            common_dtype = np.dtype(object)

        use_native = True
        if any_missing:
            if np.issubdtype(common_dtype, np.floating) or np.issubdtype(
                common_dtype, np.complexfloating
            ):
                fill_value: Any = np.nan
            else:
                use_native = False
                fill_value = None
        else:
            fill_value = None  # unused

        if not use_native:
            common_dtype = np.dtype(object)

        parts: list[np.ndarray] = []
        present_iter = iter(present_arrays)
        for model in contributing_models:
            values = model.point.extra_columns.get(key)
            if values is None:
                parts.append(
                    np.full(model.point.n_sources, fill_value, dtype=common_dtype)
                )
                continue
            parts.append(next(present_iter).astype(common_dtype, copy=False))

        extra_columns[key] = np.concatenate(parts) if parts else np.zeros(0)
    return extra_columns


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
            per_source_reference_frequencies(
                m.point,
                model_reference_frequency=m.reference_frequency,
                fallback=reference_frequency,
            )
            for m in populated
            if m.point is not None
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
