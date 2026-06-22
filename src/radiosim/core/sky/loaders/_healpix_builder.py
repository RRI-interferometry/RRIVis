"""Shared HEALPix cube assembly helpers for sky loaders."""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING, Literal

import healpy as hp
import numpy as np

from ..containers import HealpixData
from ..containers._shared import validate_frequency_axis
from ..support import allocation as _allocation
from ..support.healpix_geometry import close_memmap
from ..support.precision import get_sky_storage_dtype

if TYPE_CHECKING:
    from radiosim.core.precision import PrecisionConfig

    from ..operations.region import SkyRegion

logger = logging.getLogger(__name__)

StokesComponent = Literal["I", "Q", "U", "V", 1, 2, 3, 4]
StokesRow = Sequence[np.ndarray | None]

_COMPONENT_TO_INDEX = {
    "I": 0,
    "Q": 1,
    "U": 2,
    "V": 3,
    1: 0,
    2: 1,
    3: 2,
    4: 3,
}


def extract_stokes_component(
    stokes: np.ndarray,
    component: StokesComponent,
    n_available: int | None = None,
) -> np.ndarray | None:
    """Return one Stokes component row, or ``None`` when it is unavailable."""
    if component not in _COMPONENT_TO_INDEX:
        raise ValueError(f"Unknown Stokes component {component!r}.")
    index = _COMPONENT_TO_INDEX[component]
    if n_available is None:
        n_available = int(stokes.shape[0])
    if index >= n_available:
        return None
    return np.asarray(stokes[index])


def _prepare_pixel_selection(
    *,
    nside: int,
    hpx_inds: np.ndarray | None,
    region: SkyRegion | None,
    coordinate_frame: str,
) -> tuple[np.ndarray | None, np.ndarray | slice, int]:
    full_npix = hp.nside2npix(nside)
    if hpx_inds is not None:
        input_hpx_inds = np.asarray(hpx_inds, dtype=np.int64)
        if input_hpx_inds.ndim != 1:
            raise ValueError(
                f"hpx_inds must be a 1-D array, got shape {input_hpx_inds.shape}."
            )
        if (
            input_hpx_inds.size
            and np.unique(input_hpx_inds).size != input_hpx_inds.size
        ):
            raise ValueError("hpx_inds must be unique; duplicate HEALPix pixels found.")
        if input_hpx_inds.size and (
            input_hpx_inds.min() < 0 or input_hpx_inds.max() >= full_npix
        ):
            raise ValueError(
                f"hpx_inds must lie within [0, {full_npix}); got range "
                f"[{input_hpx_inds.min()}, {input_hpx_inds.max()}]."
            )
        input_npix = len(input_hpx_inds)
    else:
        input_hpx_inds = None
        input_npix = full_npix

    if region is None:
        output_hpx_inds = input_hpx_inds
        if output_hpx_inds is not None and len(output_hpx_inds) == full_npix:
            if np.array_equal(output_hpx_inds, np.arange(full_npix, dtype=np.int64)):
                output_hpx_inds = None
        return output_hpx_inds, slice(None), input_npix

    mask = region.healpix_mask(nside, coordinate_frame=coordinate_frame)
    mask = np.asarray(mask, dtype=bool)
    if len(mask) != full_npix:
        raise ValueError(
            "Region HEALPix mask length must match the full HEALPix grid "
            f"({len(mask)} != {full_npix})."
        )

    if input_hpx_inds is None:
        keep: np.ndarray | slice = mask
        output_hpx_inds = np.flatnonzero(mask).astype(np.int64, copy=False)
    else:
        keep = mask[input_hpx_inds]
        output_hpx_inds = input_hpx_inds[keep]

    logger.info(
        "HEALPix region retained %d/%d input pixels",
        int(len(output_hpx_inds)),
        input_npix,
    )
    return output_hpx_inds, keep, int(len(output_hpx_inds))


def _selected_row(
    row: np.ndarray,
    *,
    keep: np.ndarray | slice,
    expected_input_npix: int,
    expected_output_npix: int,
) -> np.ndarray:
    row = np.asarray(row)
    if row.ndim != 1:
        raise ValueError(f"HEALPix Stokes rows must be 1-D, got shape {row.shape}.")
    if len(row) != expected_input_npix:
        raise ValueError(
            "HEALPix Stokes row length does not match the input pixel axis "
            f"({len(row)} != {expected_input_npix})."
        )
    selected = row[keep]
    if len(selected) != expected_output_npix:
        raise ValueError(
            "HEALPix selected row length does not match the output pixel axis "
            f"({len(selected)} != {expected_output_npix})."
        )
    return selected


def build_healpix_from_stokes_cube(
    *,
    stokes_rows: Iterable[StokesRow],
    nside: int,
    frequencies: np.ndarray,
    coordinate_frame: str,
    hpx_inds: np.ndarray | None = None,
    region: SkyRegion | None = None,
    precision: PrecisionConfig | None = None,
    memmap_dir: str | None = None,
    ordering: str = "ring",
) -> HealpixData:
    """Build a :class:`HealpixData` from per-frequency Stokes rows.

    ``stokes_rows`` must yield one sequence per frequency in I/Q/U/V order.
    Stokes I is required for every row; Q/U/V arrays are allocated lazily
    only when the corresponding component appears.
    """
    frequencies = validate_frequency_axis(
        frequencies, label="HEALPix builder frequencies", ascending=False
    )
    n_freq = len(frequencies)
    output_hpx_inds, keep, n_output_pix = _prepare_pixel_selection(
        nside=nside,
        hpx_inds=hpx_inds,
        region=region,
        coordinate_frame=coordinate_frame,
    )
    input_npix = hp.nside2npix(nside) if hpx_inds is None else len(hpx_inds)

    scratch = (
        _allocation.ensure_scratch_dir(memmap_dir) if memmap_dir is not None else None
    )
    hp_dtype = get_sky_storage_dtype(precision, "healpix_maps")
    shape = (n_freq, n_output_pix)

    i_arr = _allocation.allocate_cube(shape, hp_dtype, scratch, "i_maps")
    q_arr: np.ndarray | None = None
    u_arr: np.ndarray | None = None
    v_arr: np.ndarray | None = None

    rows_seen = 0
    try:
        for fi, row in enumerate(stokes_rows):
            if fi >= n_freq:
                raise ValueError(
                    "HEALPix stokes_rows yielded more rows than frequencies "
                    f"({fi + 1} > {n_freq})."
                )
            if len(row) == 0 or row[0] is None:
                raise ValueError(f"Missing Stokes I row for frequency index {fi}.")

            i_arr[fi] = _selected_row(
                row[0],
                keep=keep,
                expected_input_npix=input_npix,
                expected_output_npix=n_output_pix,
            ).astype(hp_dtype, copy=False)

            if len(row) > 1 and row[1] is not None:
                if q_arr is None:
                    q_arr = _allocation.allocate_cube(
                        shape, hp_dtype, scratch, "q_maps"
                    )
                q_arr[fi] = _selected_row(
                    row[1],
                    keep=keep,
                    expected_input_npix=input_npix,
                    expected_output_npix=n_output_pix,
                ).astype(hp_dtype, copy=False)

            if len(row) > 2 and row[2] is not None:
                if u_arr is None:
                    u_arr = _allocation.allocate_cube(
                        shape, hp_dtype, scratch, "u_maps"
                    )
                u_arr[fi] = _selected_row(
                    row[2],
                    keep=keep,
                    expected_input_npix=input_npix,
                    expected_output_npix=n_output_pix,
                ).astype(hp_dtype, copy=False)

            if len(row) > 3 and row[3] is not None:
                if v_arr is None:
                    v_arr = _allocation.allocate_cube(
                        shape, hp_dtype, scratch, "v_maps"
                    )
                v_arr[fi] = _selected_row(
                    row[3],
                    keep=keep,
                    expected_input_npix=input_npix,
                    expected_output_npix=n_output_pix,
                ).astype(hp_dtype, copy=False)
            rows_seen = fi + 1

        if rows_seen != n_freq:
            raise ValueError(
                "HEALPix stokes_rows yielded fewer rows than frequencies "
                f"({rows_seen} != {n_freq})."
            )

        i_arr = _allocation.finalize_cube(i_arr, scratch, "i_maps")
        if q_arr is not None:
            q_arr = _allocation.finalize_cube(q_arr, scratch, "q_maps")
        if u_arr is not None:
            u_arr = _allocation.finalize_cube(u_arr, scratch, "u_maps")
        if v_arr is not None:
            v_arr = _allocation.finalize_cube(v_arr, scratch, "v_maps")
    except Exception:
        for arr in (i_arr, q_arr, u_arr, v_arr):
            close_memmap(arr)
        raise

    return HealpixData(
        maps=i_arr,
        nside=nside,
        frequencies=frequencies,
        coordinate_frame=coordinate_frame,
        ordering=ordering,
        hpx_inds=output_hpx_inds,
        q_maps=q_arr,
        u_maps=u_arr,
        v_maps=v_arr,
    )
