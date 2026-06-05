"""FITS image sky model reader.

Reads WCS-projected FITS images (2D, 3D frequency cubes, or 4D Stokes+freq)
and reprojects them onto HEALPix maps using the ``reproject`` package.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
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

    from ..containers import SkyProvenance
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


@dataclass(frozen=True)
class _FitsAxesAndUnits:
    ndim: int
    freq_ax: int | None
    stokes_ax: int | None
    freq_vals: np.ndarray
    stokes_vals: np.ndarray
    n_stokes: int
    wcs_2d: Any
    is_jy_beam: bool
    is_jy_pixel: bool
    is_jy_sr: bool
    beam_area_sr: float | None
    pixel_area_sr: float | None
    omega_pixel: float


def _fits_to_python_axis(ndim: int, fits_ax: int) -> int:
    """Map a 1-based FITS axis number onto NumPy's 0-based array axis."""
    return ndim - fits_ax


def _parse_fits_axes_and_units(
    *,
    header: Any,
    data: np.ndarray,
    frequencies: np.ndarray | None,
    nside: int,
) -> _FitsAxesAndUnits:
    from astropy.wcs import WCS

    full_wcs = WCS(header)
    ndim = data.ndim
    freq_ax = _find_axis(header, "FREQ")
    stokes_ax = _find_axis(header, "STOKES")

    if freq_ax is not None:
        n_freq = header[f"NAXIS{freq_ax}"]
        freq_vals = _axis_values(header, freq_ax, n_freq)
        cunit = header.get(f"CUNIT{freq_ax}", "Hz").strip().upper()
        unit_scale = {"HZ": 1.0, "KHZ": 1e3, "MHZ": 1e6, "GHZ": 1e9}
        freq_vals *= unit_scale.get(cunit, 1.0)
    elif frequencies is not None:
        freq_vals = np.asarray(frequencies, dtype=np.float64)
    else:
        restfrq = header.get("RESTFRQ") or header.get("RESTFREQ")
        if restfrq:
            freq_vals = np.array([float(restfrq)])
        else:
            raise ValueError(
                "Cannot determine frequency from FITS header. "
                "Provide the 'frequencies' parameter."
            )

    if stokes_ax is not None:
        n_stokes = header[f"NAXIS{stokes_ax}"]
        stokes_vals = _axis_values(header, stokes_ax, n_stokes)
        stokes_vals = np.round(stokes_vals).astype(int)
    else:
        stokes_vals = np.array([1])
        n_stokes = 1

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
        bmaj_rad = np.deg2rad(bmaj)
        bmin_rad = np.deg2rad(bmin)
        beam_area_sr = math.pi * bmaj_rad * bmin_rad / (4 * math.log(2))
        cdelt1 = abs(header.get("CDELT1", 1.0))
        cdelt2 = abs(header.get("CDELT2", 1.0))
        pixel_area_sr = np.deg2rad(cdelt1) * np.deg2rad(cdelt2)

    npix = hp.nside2npix(nside)
    return _FitsAxesAndUnits(
        ndim=ndim,
        freq_ax=freq_ax,
        stokes_ax=stokes_ax,
        freq_vals=np.asarray(freq_vals, dtype=np.float64),
        stokes_vals=stokes_vals,
        n_stokes=n_stokes,
        wcs_2d=full_wcs.celestial,
        is_jy_beam=is_jy_beam,
        is_jy_pixel=is_jy_pixel,
        is_jy_sr=is_jy_sr,
        beam_area_sr=beam_area_sr,
        pixel_area_sr=pixel_area_sr,
        omega_pixel=4 * np.pi / npix,
    )


def _extract_fits_spatial_slice(
    data: np.ndarray,
    spec: _FitsAxesAndUnits,
    *,
    stokes_idx: int | None,
    freq_idx: int | None,
) -> np.ndarray:
    """Extract a 2D spatial image from a FITS image or cube."""
    if spec.ndim == 2:
        return data
    if spec.ndim == 3:
        if spec.freq_ax is not None:
            py_axis = _fits_to_python_axis(spec.ndim, spec.freq_ax)
            return np.take(data, freq_idx or 0, axis=py_axis)
        if spec.stokes_ax is not None:
            py_axis = _fits_to_python_axis(spec.ndim, spec.stokes_ax)
            return np.take(data, stokes_idx or 0, axis=py_axis)
        return data[freq_idx or 0]
    if spec.ndim == 4:
        si = stokes_idx or 0
        fi = freq_idx or 0
        if spec.stokes_ax and spec.freq_ax:
            stokes_py = _fits_to_python_axis(spec.ndim, spec.stokes_ax)
            freq_py = _fits_to_python_axis(spec.ndim, spec.freq_ax)
            stokes_slice = np.take(data, si, axis=stokes_py)
            freq_py_adjusted = freq_py if freq_py < stokes_py else freq_py - 1
            return np.take(stokes_slice, fi, axis=freq_py_adjusted)
        return data[si, fi]

    spatial_slice = data
    while spatial_slice.ndim > 2:
        spatial_slice = spatial_slice[0]
    return spatial_slice


def _resolve_fits_output_frequencies(
    spec: _FitsAxesAndUnits,
    frequencies: np.ndarray | None,
) -> tuple[np.ndarray, list[int], bool]:
    freq_vals_raw = np.asarray(spec.freq_vals, dtype=np.float64)
    single_freq_replicate = (
        spec.freq_ax is None and frequencies is not None and len(freq_vals_raw) == 1
    )
    if single_freq_replicate:
        final_freqs = np.asarray(frequencies, dtype=np.float64)
        sort_idx = np.argsort(final_freqs)
        final_freqs = final_freqs[sort_idx]
        src_row_for_out = [0] * len(final_freqs)
    else:
        sort_idx = np.argsort(freq_vals_raw)
        final_freqs = freq_vals_raw[sort_idx]
        src_row_for_out = sort_idx.tolist()
    return final_freqs, src_row_for_out, single_freq_replicate


def _reproject_fits_stokes(
    *,
    data: np.ndarray,
    spec: _FitsAxesAndUnits,
    nside: int,
    frequencies: np.ndarray | None,
    brightness_conversion: str,
    filename: str,
    reproject_to_healpix: Any,
):
    final_freqs, src_row_for_out, single_freq_replicate = (
        _resolve_fits_output_frequencies(spec, frequencies)
    )

    def _reproject_slice(image_2d: np.ndarray) -> np.ndarray:
        hp_array, _footprint = reproject_to_healpix(
            (image_2d.astype(np.float64), spec.wcs_2d),
            "icrs",
            nside=nside,
            order="bilinear",
            nested=False,
        )
        hp_array = np.asarray(hp_array, dtype=np.float64)
        hp_array[~np.isfinite(hp_array)] = 0.0
        return hp_array

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
        for out_fi, src_fi in enumerate(src_row_for_out):
            freq_hz = float(final_freqs[out_fi])

            if single_freq_replicate and cached_stokes_row is not None:
                yield cached_stokes_row
                continue

            i_row: np.ndarray | None = None
            q_row: np.ndarray | None = None
            u_row: np.ndarray | None = None
            v_row: np.ndarray | None = None

            for si, stokes_code in enumerate(spec.stokes_vals):
                if si >= spec.n_stokes:
                    break

                fits_fi = src_fi if spec.freq_ax is not None else None
                fits_si = si if spec.stokes_ax is not None else None

                image_2d = _extract_fits_spatial_slice(
                    data,
                    spec,
                    stokes_idx=fits_si,
                    freq_idx=fits_fi,
                )
                hp_map = _reproject_slice(image_2d)

                if spec.is_jy_beam:
                    assert spec.pixel_area_sr is not None
                    assert spec.beam_area_sr is not None
                    hp_map *= spec.pixel_area_sr / spec.beam_area_sr

                flux_map: np.ndarray | None = None
                if spec.is_jy_beam or spec.is_jy_pixel:
                    flux_map = hp_map
                elif spec.is_jy_sr:
                    flux_map = hp_map * spec.omega_pixel

                is_stokes_i = stokes_code == 1 or spec.n_stokes == 1
                if flux_map is not None:
                    if is_stokes_i:
                        pos = flux_map > 0
                        temp_map = np.zeros_like(hp_map)
                        if np.any(pos):
                            temp_map[pos] = flux_density_to_brightness_temp(
                                flux_map[pos],
                                freq_hz,
                                spec.omega_pixel,
                                method=brightness_conversion,
                            )
                        hp_map = temp_map
                    else:
                        hp_map = flux_map / rayleigh_jeans_factor(
                            freq_hz,
                            spec.omega_pixel,
                        )

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

    return final_freqs, _iter_stokes_rows()


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

    spec = _parse_fits_axes_and_units(
        header=header,
        data=data,
        frequencies=frequencies,
        nside=nside,
    )
    obs_freqs, stokes_rows = _reproject_fits_stokes(
        data=data,
        spec=spec,
        nside=nside,
        frequencies=frequencies,
        brightness_conversion=brightness_conversion,
        filename=filename,
        reproject_to_healpix=reproject_to_healpix,
    )

    healpix = build_healpix_from_stokes_cube(
        stokes_rows=stokes_rows,
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
