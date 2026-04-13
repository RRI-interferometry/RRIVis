# rrivis/core/sky/_serialization.py
"""Serialization helpers for SkyModel (SkyH5 via pyradiosky).

Extracted from model.py to keep SkyModel focused on data access.
"""

from __future__ import annotations

import logging
import warnings
from typing import TYPE_CHECKING, Any

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord

if TYPE_CHECKING:
    from rrivis.core.precision import PrecisionConfig

    from .model import SkyModel

logger = logging.getLogger(__name__)


def _resolve_serialization_format(
    sky: SkyModel,
    representation: Any,
) -> Any:
    from .model import SkyFormat

    if representation is not None:
        target = (
            SkyFormat(representation)
            if isinstance(representation, str)
            else representation
        )
        if target not in sky.available_formats:
            raise ValueError(
                f"Cannot serialize as {target.value!r}; payload is not available. "
                f"Available: {[fmt.value for fmt in sky.available_formats]}"
            )
        return target
    if len(sky.available_formats) == 1:
        return next(iter(sky.available_formats))
    raise ValueError(
        "SkyModel contains both point and HEALPix payloads. "
        "Pass representation='point_sources' or representation='healpix_map' "
        "to to_pyradiosky() or save_skyh5()."
    )


def _sanitize_extra_column(values: np.ndarray) -> np.ndarray:
    """Convert metadata columns to pyradiosky-friendly array dtypes."""
    arr = np.asarray(values)
    if arr.dtype.kind != "O":
        return arr

    flat = arr.tolist()
    present = [value for value in flat if value is not None]
    if not present:
        return np.full(len(arr), "", dtype=str)
    if all(isinstance(value, (str, bytes, np.str_)) for value in present):
        return np.asarray(
            ["" if value is None else str(value) for value in flat],
            dtype=str,
        )
    if all(
        isinstance(
            value,
            (
                bool,
                int,
                float,
                np.bool_,
                np.integer,
                np.floating,
            ),
        )
        for value in present
    ):
        return np.asarray(
            [np.nan if value is None else float(value) for value in flat],
            dtype=np.float64,
        )
    return np.asarray(
        ["" if value is None else str(value) for value in flat],
        dtype=str,
    )


def to_pyradiosky(sky: SkyModel, representation: Any = None) -> Any:
    """Convert a SkyModel to a pyradiosky.SkyModel for serialization.

    ``representation`` is required when both point and HEALPix payloads
    are populated.
    """
    from pyradiosky import SkyModel as PyRadioSkyModel

    from .model import SkyFormat

    target = _resolve_serialization_format(sky, representation)

    if target == SkyFormat.HEALPIX:
        if sky.healpix is None:
            raise ValueError("Cannot serialize missing HEALPix SkyModel payload.")
        healpix = sky.healpix
        healpix_maps = healpix.maps
        nside = healpix.nside
        observation_frequencies = healpix.frequencies
        npix = healpix.n_pixels
        sorted_indices = np.argsort(observation_frequencies)
        sorted_freqs = observation_frequencies[sorted_indices]
        n_freq = len(sorted_freqs)

        stokes_arr = np.zeros((4, n_freq, npix), dtype=np.float32)
        q_maps = healpix.q_maps
        u_maps = healpix.u_maps
        v_maps = healpix.v_maps
        for out_i, src_i in enumerate(sorted_indices):
            stokes_arr[0, out_i, :] = healpix_maps[src_i]
            if q_maps is not None:
                stokes_arr[1, out_i, :] = q_maps[src_i]
            if u_maps is not None:
                stokes_arr[2, out_i, :] = u_maps[src_i]
            if v_maps is not None:
                stokes_arr[3, out_i, :] = v_maps[src_i]

        from astropy.coordinates import ICRS, Galactic

        frame = Galactic() if healpix.coordinate_frame == "galactic" else ICRS()

        return PyRadioSkyModel(
            nside=nside,
            hpx_order="ring",
            hpx_inds=(
                healpix.hpx_inds if healpix.hpx_inds is not None else np.arange(npix)
            ),
            stokes=stokes_arr * u.K,
            spectral_type="full",
            freq_array=sorted_freqs * u.Hz,
            component_type="healpix",
            frame=frame,
            history=f"RRIVis SkyModel: {sky.model_name or 'unknown'}, "
            f"brightness_conversion={sky.brightness_conversion}",
        )

    if sky.point is not None and not sky.point.is_empty:
        point = sky.point
        n = point.n_sources
        if point.source_name is not None:
            names = np.asarray(
                ["" if value is None else str(value) for value in point.source_name],
                dtype=str,
            )
        else:
            names = np.full(n, "", dtype=str)

        skycoord = SkyCoord(
            ra=point.ra_rad, dec=point.dec_rad, unit="rad", frame="icrs"
        )

        stokes_arr = np.zeros((4, 1, n), dtype=np.float64)
        stokes_arr[0, 0, :] = point.flux
        stokes_arr[1, 0, :] = point.stokes_q
        stokes_arr[2, 0, :] = point.stokes_u
        stokes_arr[3, 0, :] = point.stokes_v

        ref_freq = point.ref_freq
        if ref_freq is not None and len(ref_freq) == n:
            ref_freq_arr = ref_freq * u.Hz
        else:
            scalar_ref_freq = sky.reference_frequency or 1e8
            ref_freq_arr = np.full(n, scalar_ref_freq) * u.Hz

        extra_column_dict = {
            name: _sanitize_extra_column(values)
            for name, values in point.extra_columns.items()
        }
        if point.source_id is not None and "source_id" not in extra_column_dict:
            extra_column_dict["source_id"] = _sanitize_extra_column(point.source_id)

        return PyRadioSkyModel(
            name=names,
            skycoord=skycoord,
            stokes=stokes_arr * u.Jy,
            spectral_type="spectral_index",
            spectral_index=point.spectral_index.copy(),
            reference_frequency=ref_freq_arr,
            component_type="point",
            extra_column_dict=extra_column_dict or None,
            history=f"RRIVis SkyModel: {sky.model_name or 'unknown'}, "
            f"brightness_conversion={sky.brightness_conversion}",
        )

    raise ValueError("Cannot save an empty SkyModel (no sources or maps).")


def save_skyh5(
    sky: SkyModel,
    filename: str,
    *,
    representation: Any = None,
    clobber: bool = False,
    compression: str | None = "gzip",
) -> None:
    """Save a SkyModel to SkyH5 format (HDF5 via pyradiosky).

    Parameters
    ----------
    sky : SkyModel
        The model to save.
    filename : str
        Output file path (typically *.skyh5).
    representation : SkyFormat or str, optional
        Required when both point-source and HEALPix payloads are present.
    clobber : bool, default False
        Overwrite an existing file.
    compression : str or None, default "gzip"
        HDF5 compression for data arrays.
    """
    lost = []
    point = sky.point
    rotation_measure = point.rotation_measure if point is not None else None
    major_arcsec = point.major_arcsec if point is not None else None
    spectral_coeffs = point.spectral_coeffs if point is not None else None
    if rotation_measure is not None and np.any(rotation_measure != 0):
        lost.append("rotation measure")
    if major_arcsec is not None and np.any(major_arcsec > 0):
        lost.append("Gaussian morphology")
    if spectral_coeffs is not None and spectral_coeffs.shape[1] > 1:
        lost.append("multi-term spectral coefficients")
    if lost:
        warnings.warn(
            f"SkyH5 format does not preserve: {', '.join(lost)}. "
            "Use rrivis.core.sky.write_bbs() for lossless export.",
            UserWarning,
            stacklevel=3,
        )
    psky = to_pyradiosky(sky, representation=representation)
    psky.write_skyh5(filename, clobber=clobber, data_compression=compression)
    logger.info(f"SkyModel saved to {filename}")


def load_skyh5(
    filename: str,
    *,
    precision: PrecisionConfig,
    **kwargs: Any,
) -> SkyModel:
    """Load a SkyModel from a SkyH5 file.

    Convenience wrapper around load_pyradiosky_file().
    """
    from ._loaders_pyradiosky import load_pyradiosky_file

    return load_pyradiosky_file(
        filename, filetype="skyh5", precision=precision, **kwargs
    )
