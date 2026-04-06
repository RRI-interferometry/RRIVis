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

from ._registry import register_loader
from .constants import flux_density_to_brightness_temp

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


@register_loader("fits_image")
def load_fits_image(
    filename: str,
    *,
    nside: int = 128,
    frequencies: np.ndarray | None = None,
    region: SkyRegion | None = None,
    brightness_conversion: str = "planck",
    precision: PrecisionConfig | None = None,
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
        In ``healpix_multifreq`` mode.
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

    i_maps: dict[float, np.ndarray] = {}
    q_maps: dict[float, np.ndarray] = {}
    u_maps: dict[float, np.ndarray] = {}
    v_maps: dict[float, np.ndarray] = {}

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

    n_freq_out = len(freq_vals)
    for fi in range(n_freq_out):
        freq_hz = float(freq_vals[fi]) if fi < len(freq_vals) else float(freq_vals[0])

        for si, stokes_code in enumerate(stokes_vals):
            if si >= n_stokes:
                break

            # Determine freq index into FITS data
            fits_fi = fi if freq_ax is not None else None
            fits_si = si if stokes_ax is not None else None

            image_2d = _get_slice(fits_si, fits_fi)
            hp_map = _reproject_slice(image_2d)

            # Unit conversion
            if is_jy_beam:
                hp_map *= pixel_area_sr / beam_area_sr
                # Now in Jy/pixel -- convert to Jy
                # (already per pixel after rescaling)

            if is_jy_beam or is_jy_pixel:
                # Convert Jy -> brightness temperature K
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
                # Jy/sr -> Jy per pixel
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
            # else: already in K or unknown unit -- use as-is

            hp_map_f32 = hp_map.astype(np.float32)

            # Stokes mapping: I=1, Q=2, U=3, V=4
            if stokes_code == 1 or n_stokes == 1:
                i_maps[freq_hz] = hp_map_f32
            elif stokes_code == 2:
                q_maps[freq_hz] = hp_map_f32
            elif stokes_code == 3:
                u_maps[freq_hz] = hp_map_f32
            elif stokes_code == 4:
                v_maps[freq_hz] = hp_map_f32

    # If single frequency and no freq axis, replicate for each requested freq
    if freq_ax is None and frequencies is not None and len(i_maps) == 1:
        single_freq = next(iter(i_maps))
        single_map = i_maps[single_freq]
        i_maps = {float(f): single_map.copy() for f in frequencies}

    if not i_maps:
        raise ValueError(f"No Stokes I data found in {filename}")

    obs_freqs = np.sort(np.array(list(i_maps.keys())))

    sky = SkyModel._from_freq_dict_maps(
        i_maps,
        q_maps if q_maps else None,
        u_maps if u_maps else None,
        v_maps if v_maps else None,
        nside,
        _native_format="healpix",
        frequency=float(obs_freqs[0]),
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
