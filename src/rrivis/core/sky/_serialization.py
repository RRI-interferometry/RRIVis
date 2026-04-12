# rrivis/core/sky/_serialization.py
"""Serialization helpers for SkyModel (SkyH5 via pyradiosky).

Extracted from model.py to keep SkyModel focused on data access.
"""

from __future__ import annotations

import logging
import warnings
from typing import TYPE_CHECKING, Any

import astropy.units as u
import healpy as hp
import numpy as np
from astropy.coordinates import SkyCoord

if TYPE_CHECKING:
    from rrivis.core.precision import PrecisionConfig

    from .model import SkyModel

logger = logging.getLogger(__name__)


def to_pyradiosky(sky: SkyModel) -> Any:
    """Convert a SkyModel to a pyradiosky.SkyModel for serialization.

    Uses the current mode (not _native_format) to decide which
    representation to serialize.
    """
    from pyradiosky import SkyModel as PyRadioSkyModel

    # Prefer HEALPix if maps are populated
    if sky._healpix_maps is not None:
        nside = sky._healpix_nside
        npix = hp.nside2npix(nside)
        sorted_indices = np.argsort(sky._observation_frequencies)
        sorted_freqs = sky._observation_frequencies[sorted_indices]
        n_freq = len(sorted_freqs)

        stokes_arr = np.zeros((4, n_freq, npix), dtype=np.float32)
        for out_i, src_i in enumerate(sorted_indices):
            stokes_arr[0, out_i, :] = sky._healpix_maps[src_i]
            if sky._healpix_q_maps is not None:
                stokes_arr[1, out_i, :] = sky._healpix_q_maps[src_i]
            if sky._healpix_u_maps is not None:
                stokes_arr[2, out_i, :] = sky._healpix_u_maps[src_i]
            if sky._healpix_v_maps is not None:
                stokes_arr[3, out_i, :] = sky._healpix_v_maps[src_i]

        from astropy.coordinates import ICRS

        return PyRadioSkyModel(
            nside=nside,
            hpx_order="ring",
            hpx_inds=np.arange(npix),
            stokes=stokes_arr * u.K,
            spectral_type="full",
            freq_array=sorted_freqs * u.Hz,
            component_type="healpix",
            frame=ICRS(),
            history=f"RRIVis SkyModel: {sky.model_name or 'unknown'}, "
            f"brightness_conversion={sky.brightness_conversion}",
        )

    # Fall back to point-source serialization
    if sky._ra_rad is not None and len(sky._ra_rad) > 0:
        n = len(sky._ra_rad)
        prefix = sky.model_name or "src"
        names = np.array([f"{prefix}_{i}" for i in range(n)])

        skycoord = SkyCoord(ra=sky._ra_rad, dec=sky._dec_rad, unit="rad", frame="icrs")

        stokes_arr = np.zeros((4, 1, n), dtype=np.float64)
        stokes_arr[0, 0, :] = sky._flux
        if sky._stokes_q is not None:
            stokes_arr[1, 0, :] = sky._stokes_q
        if sky._stokes_u is not None:
            stokes_arr[2, 0, :] = sky._stokes_u
        if sky._stokes_v is not None:
            stokes_arr[3, 0, :] = sky._stokes_v

        if sky._ref_freq is not None and len(sky._ref_freq) == n:
            ref_freq_arr = sky._ref_freq * u.Hz
        else:
            ref_freq = sky.reference_frequency or 1e8
            ref_freq_arr = np.full(n, ref_freq) * u.Hz

        return PyRadioSkyModel(
            name=names,
            skycoord=skycoord,
            stokes=stokes_arr * u.Jy,
            spectral_type="spectral_index",
            spectral_index=sky._spectral_index.copy(),
            reference_frequency=ref_freq_arr,
            component_type="point",
            history=f"RRIVis SkyModel: {sky.model_name or 'unknown'}, "
            f"brightness_conversion={sky.brightness_conversion}",
        )

    raise ValueError("Cannot save an empty SkyModel (no sources or maps).")


def save_skyh5(
    sky: SkyModel,
    filename: str,
    *,
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
    clobber : bool, default False
        Overwrite an existing file.
    compression : str or None, default "gzip"
        HDF5 compression for data arrays.
    """
    lost = []
    if sky._rotation_measure is not None and np.any(sky._rotation_measure != 0):
        lost.append("rotation measure")
    if sky._major_arcsec is not None and np.any(sky._major_arcsec > 0):
        lost.append("Gaussian morphology")
    if sky._spectral_coeffs is not None and sky._spectral_coeffs.shape[1] > 1:
        lost.append("multi-term spectral coefficients")
    if lost:
        warnings.warn(
            f"SkyH5 format does not preserve: {', '.join(lost)}. "
            "Use write_bbs() from _loaders_bbs for lossless export.",
            UserWarning,
            stacklevel=3,
        )
    psky = to_pyradiosky(sky)
    psky.write_skyh5(filename, clobber=clobber, data_compression=compression)
    logger.info(f"SkyModel saved to {filename}")


def load_skyh5(
    filename: str,
    *,
    precision: PrecisionConfig | None = None,
    **kwargs: Any,
) -> SkyModel:
    """Load a SkyModel from a SkyH5 file.

    Convenience wrapper around load_pyradiosky_file().
    """
    from ._loaders_pyradiosky import load_pyradiosky_file

    return load_pyradiosky_file(
        filename, filetype="skyh5", precision=precision, **kwargs
    )
