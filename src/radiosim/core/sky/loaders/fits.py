"""FITS image sky model reader.

Reads WCS-projected FITS images (2D, 3D frequency cubes, or 4D Stokes+freq)
and reprojects them onto HEALPix maps using the ``reproject`` package.
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING, Any

import healpy as hp
import numpy as np

from ..containers.constants import (
    flux_density_to_brightness_temp,
    rayleigh_jeans_factor,
)
from ..registry.facade import loader_registry
from ._healpix_builder import build_healpix_from_stokes_cube

if TYPE_CHECKING:
    from radiosim.core.precision import PrecisionConfig

    from ..containers.data import SkyProvenance
    from ..containers.model import SkyModel
    from ..operations.region import SkyRegion

logger = logging.getLogger(__name__)


def _find_axis(header: Any, prefix: str) -> int | None:
    """Find FITS axis number (1-based) whose CTYPE starts with *prefix*."""
    for i in range(1, header.get("NAXIS", 0) + 1):
        ctype = header.get(f"CTYPE{i}", "").upper()
        if ctype.startswith(prefix):
            return i
    return None


def _axis_values(header: Any, axis: int, n: int) -> np.ndarray:
    """Compute world-coordinate values along a FITS axis."""
    crval = header.get(f"CRVAL{axis}", 0.0)
    cdelt = header.get(f"CDELT{axis}", 1.0)
    crpix = header.get(f"CRPIX{axis}", 1.0)
    return crval + (np.arange(n) + 1 - crpix) * cdelt


@loader_registry.register(
    "fits_image",
    config_section="fits_image",
    use_flag="use_fits_image",
    representations=("healpix_map",),
    category="file",
    requires_file=True,
    network_service=None,
    config_fields={"filename": "filename", "nside": "nside"},
)
def load_fits_image(
    filename: str,
    *,
    nside: int = 128,
    frequencies: np.ndarray | None = None,
    region: SkyRegion | None = None,
    brightness_conversion: str = "planck",
    precision: PrecisionConfig,
    memmap_path: str | None = None,
    provenance: SkyProvenance | None = None,
) -> SkyModel:
    """Load a FITS image and reproject to HEALPix multi-frequency maps.

    Supports 2D images (single frequency, Stokes I), 3D cubes
    (frequency axis), and 4D cubes (Stokes + frequency).

    Parameters
    ----------
    filename : str
        Path to the FITS file.
    nside : int, default 128
        HEALPix NSIDE parameter.
    frequencies : np.ndarray, optional
        Observation frequencies in Hz. Required if the FITS file has no
        frequency axis.
    region : SkyRegion, optional
        Spatial filter applied after reprojection.
    brightness_conversion : str, default "planck"
        Conversion method: "planck" or "rayleigh-jeans".
    precision : PrecisionConfig
        Precision configuration.

    Returns
    -------
    SkyModel
        In ``healpix_map`` mode.
    """
    from ..containers.model import SkyModel

    try:
        from reproject import reproject_to_healpix
    except ImportError as e:
        raise ImportError(
            "The 'reproject' package is required for FITS image loading. "
            "Install it with: pixi add reproject"
        ) from e

    from astropy.io import fits
    from astropy.wcs import WCS

    try:
        hdul = fits.open(filename)
    except OSError as e:
        raise OSError(f"Could not open FITS image file {filename!r}: {e}") from e

    with hdul:
        # Find the image HDU
        hdu = hdul[0]
        if hdu.data is None:
            for h in hdul[1:]:
                if h.data is not None:
                    hdu = h
                    break
        if hdu.data is None:
            raise ValueError(f"No image data found in {filename}")

        data = np.array(hdu.data, dtype=np.float64)
        header = hdu.header

    full_wcs = WCS(header)
    ndim = data.ndim

    # Identify axes
    freq_ax = _find_axis(header, "FREQ")
    stokes_ax = _find_axis(header, "STOKES")

    # Map to 0-based Python axis indices (FITS is 1-based, reversed)
    def _py_ax(fits_ax: int) -> int:
        return ndim - fits_ax

    # Determine frequency array
    if freq_ax is not None:
        n_freq = header[f"NAXIS{freq_ax}"]
        freq_vals = _axis_values(header, freq_ax, n_freq)
        # Convert to Hz if needed
        cunit = header.get(f"CUNIT{freq_ax}", "Hz").strip().upper()
        unit_scale = {"HZ": 1.0, "KHZ": 1e3, "MHZ": 1e6, "GHZ": 1e9}
        freq_vals *= unit_scale.get(cunit, 1.0)
    elif frequencies is not None:
        freq_vals = np.asarray(frequencies, dtype=np.float64)
    else:
        # Try RESTFRQ or CRVAL3
        restfrq = header.get("RESTFRQ") or header.get("RESTFREQ")
        if restfrq:
            freq_vals = np.array([float(restfrq)])
        else:
            raise ValueError(
                "Cannot determine frequency from FITS header. "
                "Provide the 'frequencies' parameter."
            )

    # Determine Stokes indices
    if stokes_ax is not None:
        n_stokes = header[f"NAXIS{stokes_ax}"]
        stokes_vals = _axis_values(header, stokes_ax, n_stokes)
        stokes_vals = np.round(stokes_vals).astype(int)
    else:
        stokes_vals = np.array([1])  # Stokes I only
        n_stokes = 1

    # Handle BUNIT
    bunit = header.get("BUNIT", "").strip().upper()
    is_jy_beam = "JY/BEAM" in bunit
    is_jy_pixel = "JY/PIX" in bunit or "JY/PIXEL" in bunit
    is_jy_sr = "JY/SR" in bunit

    beam_area_sr = None
    pixel_area_sr = None
    if is_jy_beam:
        bmaj = header.get("BMAJ")
        bmin = header.get("BMIN")
        if bmaj is None or bmin is None:
            raise ValueError(
                f"BUNIT='{bunit}' but BMAJ/BMIN not found in header. "
                "Cannot convert Jy/beam to Jy/pixel."
            )
        # Beam area in steradians
        bmaj_rad = np.deg2rad(bmaj)
        bmin_rad = np.deg2rad(bmin)
        beam_area_sr = math.pi * bmaj_rad * bmin_rad / (4 * math.log(2))
        # Pixel area in steradians
        cdelt1 = abs(header.get("CDELT1", 1.0))
        cdelt2 = abs(header.get("CDELT2", 1.0))
        pixel_area_sr = np.deg2rad(cdelt1) * np.deg2rad(cdelt2)

    # Build 2D spatial WCS (drop non-spatial axes)
    wcs_2d = full_wcs.celestial

    # Reproject each frequency channel and Stokes to HEALPix
    npix = hp.nside2npix(nside)
    omega_pixel = 4 * np.pi / npix

    # Determine the final output frequency grid up front so we can allocate
    # the final-shape cube directly and write rows into their sorted
    # position — no post-hoc sort reindex, no broadcast-and-copy.
    freq_vals_raw = np.asarray(freq_vals, dtype=np.float64)
    single_freq_replicate = (
        freq_ax is None and frequencies is not None and len(freq_vals_raw) == 1
    )
    if single_freq_replicate:
        # Replicate the single FITS slice across the caller-supplied
        # frequency grid.  The sort below is a no-op because the grid
        # is written in its final order.
        final_freqs = np.asarray(frequencies, dtype=np.float64)
        sort_idx = np.argsort(final_freqs)
        final_freqs = final_freqs[sort_idx]
        # The source FITS slice index for each output row is always 0
        # because there is only one slice to replicate.
        src_row_for_out = [0] * len(final_freqs)
    else:
        sort_idx = np.argsort(freq_vals_raw)
        final_freqs = freq_vals_raw[sort_idx]
        # Map each output row to its source row (inverse of sort).
        src_row_for_out = sort_idx.tolist()

    n_freq_out = len(final_freqs)

    def _get_slice(stokes_idx: int | None, freq_idx: int | None) -> np.ndarray:
        """Extract a 2D spatial slice from the data cube."""
        if ndim == 2:
            return data
        if ndim == 3:
            if freq_ax is not None:
                py = _py_ax(freq_ax)
                return np.take(data, freq_idx or 0, axis=py)
            if stokes_ax is not None:
                py = _py_ax(stokes_ax)
                return np.take(data, stokes_idx or 0, axis=py)
            return data[freq_idx or 0]
        if ndim == 4:
            # Standard: (Stokes, Freq, Dec, RA) or (Freq, Stokes, Dec, RA)
            si = stokes_idx or 0
            fi = freq_idx or 0
            if stokes_ax and freq_ax:
                s_py = _py_ax(stokes_ax)
                f_py = _py_ax(freq_ax)
                slc = np.take(data, si, axis=s_py)
                # After taking one axis, adjust the other
                f_py_adj = f_py if f_py < s_py else f_py - 1
                return np.take(slc, fi, axis=f_py_adj)
            return data[si, fi]
        # Fallback for higher dimensions: take first slices
        slc = data
        while slc.ndim > 2:
            slc = slc[0]
        return slc

    def _reproject_slice(image_2d: np.ndarray) -> np.ndarray:
        """Reproject a 2D image to HEALPix."""
        hp_array, _footprint = reproject_to_healpix(
            (image_2d.astype(np.float64), wcs_2d),
            "icrs",
            nside=nside,
            order="bilinear",
            nested=False,
        )
        hp_array = np.asarray(hp_array, dtype=np.float64)
        hp_array[~np.isfinite(hp_array)] = 0.0
        return hp_array

    # Cache reprojected + unit-converted slices for single-freq replication
    # so we don't redo the reprojection work for each replicated row.
    cached_stokes_row: (
        tuple[
            np.ndarray,
            np.ndarray | None,
            np.ndarray | None,
            np.ndarray | None,
        ]
        | None
    ) = None

    def _iter_stokes_rows():
        nonlocal cached_stokes_row
        for out_fi in range(n_freq_out):
            src_fi = src_row_for_out[out_fi]
            freq_hz = float(final_freqs[out_fi])

            if single_freq_replicate and cached_stokes_row is not None:
                yield cached_stokes_row
                continue

            i_row: np.ndarray | None = None
            q_row: np.ndarray | None = None
            u_row: np.ndarray | None = None
            v_row: np.ndarray | None = None

            for si, stokes_code in enumerate(stokes_vals):
                if si >= n_stokes:
                    break

                fits_fi = src_fi if freq_ax is not None else None
                fits_si = si if stokes_ax is not None else None

                image_2d = _get_slice(fits_si, fits_fi)
                hp_map = _reproject_slice(image_2d)

                # Unit conversion. Stokes Q/U/V are signed linear components, so
                # convert them with the signed Rayleigh-Jeans relation even when
                # Stokes I uses the positive-only Planck inversion.
                if is_jy_beam:
                    assert pixel_area_sr is not None
                    assert beam_area_sr is not None
                    hp_map *= pixel_area_sr / beam_area_sr

                flux_map: np.ndarray | None = None
                if is_jy_beam or is_jy_pixel:
                    flux_map = hp_map
                elif is_jy_sr:
                    flux_map = hp_map * omega_pixel

                is_stokes_i = stokes_code == 1 or n_stokes == 1
                if flux_map is not None:
                    if is_stokes_i:
                        pos = flux_map > 0
                        temp_map = np.zeros_like(hp_map)
                        if np.any(pos):
                            temp_map[pos] = flux_density_to_brightness_temp(
                                flux_map[pos],
                                freq_hz,
                                omega_pixel,
                                method=brightness_conversion,
                            )
                        hp_map = temp_map
                    else:
                        hp_map = flux_map / rayleigh_jeans_factor(freq_hz, omega_pixel)

                # Stokes mapping: I=1, Q=2, U=3, V=4
                if is_stokes_i:
                    i_row = hp_map
                elif stokes_code == 2:
                    q_row = hp_map
                elif stokes_code == 3:
                    u_row = hp_map
                elif stokes_code == 4:
                    v_row = hp_map

            if i_row is None:
                raise ValueError(f"No Stokes I data found in {filename}")
            stokes_row = (i_row, q_row, u_row, v_row)
            if single_freq_replicate:
                cached_stokes_row = stokes_row
            yield stokes_row

    obs_freqs = final_freqs

    healpix = build_healpix_from_stokes_cube(
        stokes_rows=_iter_stokes_rows(),
        nside=nside,
        frequencies=obs_freqs,
        coordinate_frame="icrs",
        region=None,
        precision=precision,
        memmap_dir=memmap_path,
    )

    sky = SkyModel(
        healpix=healpix,
        model_name=f"fits:{filename.split('/')[-1]}",
        brightness_conversion=brightness_conversion,
        precision=precision,
    )

    if region is not None:
        # FITS reprojection always builds the full ICRS HEALPix grid first;
        # keep the single existing model-level crop here to avoid double-cropping.
        sky = sky.filter_region(region)

    logger.info(
        f"Loaded FITS image {filename} -> HEALPix nside={nside}, "
        f"{len(obs_freqs)} freq channels"
    )
    if provenance is not None:
        sky = sky.replace(provenance=provenance)
    return sky
