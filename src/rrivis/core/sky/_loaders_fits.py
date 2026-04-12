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

from ._data import HealpixData
from ._precision import get_sky_storage_dtype
from ._registry import register_loader
from .constants import flux_density_to_brightness_temp
from .model import SkyFormat

if TYPE_CHECKING:
    from rrivis.core.precision import PrecisionConfig

    from .region import SkyRegion

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


@register_loader(
    "fits_image",
    config_section="fits_image",
    use_flag="use_fits_image",
    is_healpix=True,
    requires_file=True,
    network_service=None,
)
def load_fits_image(
    filename: str,
    *,
    nside: int = 128,
    frequencies: np.ndarray | None = None,
    region: SkyRegion | None = None,
    brightness_conversion: str = "planck",
    precision: PrecisionConfig | None = None,
    memmap_path: str | None = None,
) -> Any:
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
    precision : PrecisionConfig, optional
        Precision configuration.

    Returns
    -------
    SkyModel
        In ``healpix_map`` mode.
    """
    from .model import SkyModel

    try:
        from reproject import reproject_to_healpix
    except ImportError as e:
        raise ImportError(
            "The 'reproject' package is required for FITS image loading. "
            "Install it with: pixi add reproject"
        ) from e

    from astropy.io import fits
    from astropy.wcs import WCS

    with fits.open(filename) as hdul:
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

    from ._allocation import allocate_cube, ensure_scratch_dir, finalize_cube

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
    scratch = ensure_scratch_dir(memmap_path) if memmap_path is not None else None
    hp_dtype = get_sky_storage_dtype(precision, "healpix_maps")
    i_arr = allocate_cube((n_freq_out, npix), hp_dtype, scratch, "i_maps")
    q_arr = allocate_cube((n_freq_out, npix), hp_dtype, scratch, "q_maps")
    u_arr = allocate_cube((n_freq_out, npix), hp_dtype, scratch, "u_maps")
    v_arr = allocate_cube((n_freq_out, npix), hp_dtype, scratch, "v_maps")
    has_q = False
    has_u = False
    has_v = False

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

    i_has_data = False
    # Cache reprojected + unit-converted slices for single-freq replication
    # so we don't redo the reprojection work for each replicated row.
    cache_i: np.ndarray | None = None
    cache_q: np.ndarray | None = None
    cache_u: np.ndarray | None = None
    cache_v: np.ndarray | None = None

    for out_fi in range(n_freq_out):
        src_fi = src_row_for_out[out_fi]
        freq_hz = float(final_freqs[out_fi])

        if single_freq_replicate and cache_i is not None:
            # Replicate cached FITS slice.  Memmap-safe: row-by-row copy,
            # no broadcast-then-copy allocation of the full cube.
            i_arr[out_fi] = cache_i
            if cache_q is not None:
                q_arr[out_fi] = cache_q
            if cache_u is not None:
                u_arr[out_fi] = cache_u
            if cache_v is not None:
                v_arr[out_fi] = cache_v
            continue

        for si, stokes_code in enumerate(stokes_vals):
            if si >= n_stokes:
                break

            fits_fi = src_fi if freq_ax is not None else None
            fits_si = si if stokes_ax is not None else None

            image_2d = _get_slice(fits_si, fits_fi)
            hp_map = _reproject_slice(image_2d)

            # Unit conversion
            if is_jy_beam:
                hp_map *= pixel_area_sr / beam_area_sr

            if is_jy_beam or is_jy_pixel:
                pos = hp_map > 0
                temp_map = np.zeros_like(hp_map)
                if np.any(pos):
                    temp_map[pos] = flux_density_to_brightness_temp(
                        hp_map[pos],
                        freq_hz,
                        omega_pixel,
                        method=brightness_conversion,
                    )
                hp_map = temp_map
            elif is_jy_sr:
                hp_jy = hp_map * omega_pixel
                pos = hp_jy > 0
                temp_map = np.zeros_like(hp_map)
                if np.any(pos):
                    temp_map[pos] = flux_density_to_brightness_temp(
                        hp_jy[pos],
                        freq_hz,
                        omega_pixel,
                        method=brightness_conversion,
                    )
                hp_map = temp_map

            hp_map_cast = hp_map.astype(hp_dtype)

            # Stokes mapping: I=1, Q=2, U=3, V=4
            if stokes_code == 1 or n_stokes == 1:
                i_arr[out_fi] = hp_map_cast
                i_has_data = True
                if single_freq_replicate:
                    cache_i = hp_map_cast
            elif stokes_code == 2:
                q_arr[out_fi] = hp_map_cast
                has_q = True
                if single_freq_replicate:
                    cache_q = hp_map_cast
            elif stokes_code == 3:
                u_arr[out_fi] = hp_map_cast
                has_u = True
                if single_freq_replicate:
                    cache_u = hp_map_cast
            elif stokes_code == 4:
                v_arr[out_fi] = hp_map_cast
                has_v = True
                if single_freq_replicate:
                    cache_v = hp_map_cast

    if not i_has_data:
        raise ValueError(f"No Stokes I data found in {filename}")

    obs_freqs = final_freqs

    # Flush and re-open read-only if memmap-backed.
    i_arr = finalize_cube(i_arr, scratch, "i_maps")
    if has_q:
        q_arr = finalize_cube(q_arr, scratch, "q_maps")
    if has_u:
        u_arr = finalize_cube(u_arr, scratch, "u_maps")
    if has_v:
        v_arr = finalize_cube(v_arr, scratch, "v_maps")

    sky = SkyModel(
        healpix=HealpixData(
            maps=i_arr,
            nside=nside,
            frequencies=obs_freqs,
            q_maps=q_arr if has_q else None,
            u_maps=u_arr if has_u else None,
            v_maps=v_arr if has_v else None,
        ),
        native_representation=SkyFormat.HEALPIX,
        active_representation=SkyFormat.HEALPIX,
        model_name=f"fits:{filename.split('/')[-1]}",
        brightness_conversion=brightness_conversion,
        _precision=precision,
    )

    if region is not None:
        sky = sky.filter_region(region)

    logger.info(
        f"Loaded FITS image {filename} -> HEALPix nside={nside}, "
        f"{len(obs_freqs)} freq channels"
    )
    return sky
