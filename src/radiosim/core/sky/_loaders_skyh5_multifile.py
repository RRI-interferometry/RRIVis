# radiosim/core/sky/_loaders_skyh5_multifile.py
"""Multi-file skyh5 loader.

Reads a set of single-frequency skyh5 files (each representing one channel
of the same simulation cube) and stacks them along the frequency axis into
a single :class:`SkyModel`.  Supports both ``component_type='healpix'`` and
``component_type='point'`` branches.

Point-source branch: populates ``PointSourceData.per_channel_flux`` (and
Q/U/V tables) for lossless per-channel storage; downstream consumers use
nearest-channel lookup instead of spectral-index extrapolation.

HEALPix branch: mirrors the behaviour of ``load_pyradiosky_file`` but
stacks multiple single-frequency files into one ``(n_files, npix)`` cube.
"""

from __future__ import annotations

import glob as _glob
import logging
import os
from typing import TYPE_CHECKING, Any

import astropy.units as u
import h5py
import healpy as hp
import numpy as np
from pyradiosky import SkyModel as PyRadioSkyModel

from ._allocation import allocate_cube, ensure_scratch_dir, finalize_cube
from ._data import HealpixData, PointSourceData
from ._precision import get_sky_storage_dtype
from ._registry import register_loader
from .model import SkyModel

if TYPE_CHECKING:
    from radiosim.core.precision import PrecisionConfig

    from ._data import SkyProvenance
    from .region import SkyRegion

logger = logging.getLogger(__name__)

_FREQ_DUPLICATE_TOL_HZ = 1.0


def _decode_bytes(value: Any) -> Any:
    """Decode numpy/HDF5 bytes scalars to ``str`` when applicable."""
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.ndarray) and value.dtype.kind == "S":
        return value.astype(str)
    return value


def _read_header(filename: str) -> dict[str, Any]:
    """Peek at a skyh5 file's Header group without a full pyradiosky read."""
    info: dict[str, Any] = {"filename": filename}
    with h5py.File(filename, "r") as f:
        header = f["Header"]
        info["component_type"] = _decode_bytes(header["component_type"][()])
        info["spectral_type"] = _decode_bytes(header["spectral_type"][()])
        info["Nfreqs"] = int(header["Nfreqs"][()])
        info["Ncomponents"] = int(header["Ncomponents"][()])
        freq_arr = np.asarray(header["freq_array"][()], dtype=np.float64)
        info["freq_array"] = freq_arr
        if info["component_type"] == "healpix":
            info["nside"] = int(header["nside"][()])
            info["hpx_order"] = _decode_bytes(header["hpx_order"][()]).lower()
            info["hpx_inds"] = np.asarray(header["hpx_inds"][()], dtype=np.int64)
            if "hpx_frame" in header:
                frame = _decode_bytes(header["hpx_frame/frame"][()])
                info["coordinate_frame"] = str(frame).lower()
            else:
                info["coordinate_frame"] = "icrs"
        stokes = f["Data/stokes"]
        info["stokes_shape"] = tuple(stokes.shape)
        info["stokes_unit"] = _decode_bytes(stokes.attrs.get("unit", ""))
    return info


def _resolve_file_list(
    file_glob: str | None,
    filenames: list[str] | None,
) -> list[str]:
    """Expand ``file_glob`` or validate ``filenames``; return a concrete list."""
    if (file_glob is None) == (filenames is None):
        raise ValueError(
            "load_skyh5_multifile: specify exactly one of `file_glob` or `filenames`."
        )
    if file_glob is not None:
        matches = sorted(_glob.glob(file_glob))
        if not matches:
            raise ValueError(
                f"load_skyh5_multifile: file_glob {file_glob!r} matched no files."
            )
        return matches
    assert filenames is not None
    if len(filenames) == 0:
        raise ValueError("load_skyh5_multifile: filenames list is empty.")
    for fn in filenames:
        if not os.path.exists(fn):
            raise FileNotFoundError(f"Sky model file not found: {fn}")
    return list(filenames)


def _validate_shared_metadata(headers: list[dict[str, Any]]) -> dict[str, Any]:
    """Cross-file invariants; returns the shared metadata for the whole set."""
    ref = headers[0]
    for h in headers[1:]:
        if h["component_type"] != ref["component_type"]:
            raise ValueError(
                "load_skyh5_multifile: mixed component_type across files "
                f"({ref['filename']}={ref['component_type']!r}, "
                f"{h['filename']}={h['component_type']!r})."
            )
        if h["spectral_type"] != ref["spectral_type"]:
            raise ValueError(
                "load_skyh5_multifile: mixed spectral_type across files "
                f"({ref['filename']}={ref['spectral_type']!r}, "
                f"{h['filename']}={h['spectral_type']!r})."
            )
        if h["Nfreqs"] != 1:
            raise ValueError(
                "load_skyh5_multifile: each file must hold exactly one "
                f"frequency (Nfreqs=1); {h['filename']} has Nfreqs="
                f"{h['Nfreqs']}. For multi-channel files, use the "
                "`pyradiosky_file` loader."
            )
        if h["stokes_shape"][0] != ref["stokes_shape"][0]:
            raise ValueError(
                "load_skyh5_multifile: inconsistent Stokes axis length "
                f"({ref['filename']} has {ref['stokes_shape'][0]}, "
                f"{h['filename']} has {h['stokes_shape'][0]})."
            )
    if ref["Nfreqs"] != 1:
        raise ValueError(
            "load_skyh5_multifile: each file must hold exactly one "
            f"frequency; {ref['filename']} has Nfreqs={ref['Nfreqs']}."
        )
    if ref["spectral_type"] not in ("full", "subband"):
        raise ValueError(
            "load_skyh5_multifile: spectral_type must be 'full' or 'subband'; "
            f"got {ref['spectral_type']!r}. 'flat'/'spectral_index' files "
            "describe a model not a sampled channel."
        )
    return ref


def _sort_and_validate_frequencies(headers: list[dict[str, Any]]) -> list[int]:
    """Return indices into ``headers`` that produce strictly ascending freqs."""
    freqs = np.array([h["freq_array"][0] for h in headers], dtype=np.float64)
    order = np.argsort(freqs)
    sorted_freqs = freqs[order]
    if sorted_freqs.size > 1:
        gaps = np.diff(sorted_freqs)
        if np.any(gaps <= _FREQ_DUPLICATE_TOL_HZ):
            dup_pairs = [
                (headers[order[i]]["filename"], headers[order[i + 1]]["filename"])
                for i, g in enumerate(gaps)
                if g <= _FREQ_DUPLICATE_TOL_HZ
            ]
            raise ValueError(
                "load_skyh5_multifile: duplicate or near-duplicate frequencies "
                f"(<{_FREQ_DUPLICATE_TOL_HZ} Hz apart): {dup_pairs}."
            )
    return [int(i) for i in order]


def _maybe_cross_check_frequencies(
    sorted_freqs: np.ndarray,
    frequencies: np.ndarray | None,
) -> None:
    if frequencies is None:
        return
    requested = np.asarray(frequencies, dtype=np.float64)
    if requested.shape != sorted_freqs.shape or not np.allclose(
        requested, sorted_freqs, atol=_FREQ_DUPLICATE_TOL_HZ
    ):
        raise ValueError(
            "load_skyh5_multifile: requested `frequencies` do not match the "
            "file-derived channel grid within 1 Hz. Omit the argument to use "
            "the file grid, or align it exactly."
        )


@register_loader(
    "skyh5_multifile",
    representations=("point_sources", "healpix_map"),
    config_section="skyh5_multifile",
    use_flag="use_skyh5_multifile",
    category="file",
    requires_file=True,
    network_service=None,
    aliases=["pyradiosky_multifile"],
    config_fields={
        "file_glob": "file_glob",
        "filenames": "filenames",
        "brightness_conversion": "brightness_conversion",
        "reference_frequency_hz": "reference_frequency_hz",
    },
)
def load_skyh5_multifile(
    file_glob: str | None = None,
    filenames: list[str] | None = None,
    brightness_conversion: str = "rayleigh-jeans",
    reference_frequency_hz: float | None = None,
    *,
    precision: PrecisionConfig,
    frequencies: np.ndarray | None = None,
    obs_frequency_config: dict[str, Any] | None = None,  # noqa: ARG001
    region: SkyRegion | None = None,
    memmap_path: str | None = None,
    provenance: SkyProvenance | None = None,
) -> SkyModel:
    """Load and stack a set of single-frequency skyh5 files along frequency.

    Each input file must hold exactly one frequency channel
    (``Nfreqs==1``) of the same underlying sky cube.  The loader branches
    on ``component_type``: HEALPix inputs become one ``HealpixData``
    cube of shape ``(n_files, n_stored_pix)``; point-source inputs
    become a ``PointSourceData`` carrying a lossless per-channel flux
    table.

    Parameters
    ----------
    file_glob : str, optional
        Glob pattern matching the input files (mutually exclusive with
        ``filenames``).
    filenames : list[str], optional
        Explicit list of file paths (mutually exclusive with
        ``file_glob``).
    brightness_conversion : str, default ``"rayleigh-jeans"``
        Brightness conversion for HEALPix data. Forced to
        ``"rayleigh-jeans"`` when polarization is present.
    reference_frequency_hz : float, optional
        Observation frequency to use as the reference channel for
        ``PointSourceData.flux`` / ``ref_freq``.  Defaults to the lowest
        channel frequency.

    Returns
    -------
    SkyModel

    Raises
    ------
    ValueError
        If the cross-file invariants are violated (mixed component_type,
        mixed nside/frame/hpx_order/hpx_inds, mismatched source list,
        duplicate frequencies, etc.).
    FileNotFoundError
        If any explicit filename does not exist.
    """
    paths = _resolve_file_list(file_glob, filenames)
    headers = [_read_header(p) for p in paths]
    shared = _validate_shared_metadata(headers)
    order = _sort_and_validate_frequencies(headers)
    sorted_headers = [headers[i] for i in order]
    sorted_paths = [paths[i] for i in order]
    sorted_freqs = np.array(
        [h["freq_array"][0] for h in sorted_headers], dtype=np.float64
    )
    _maybe_cross_check_frequencies(sorted_freqs, frequencies)

    component_type = shared["component_type"]
    n_stokes_avail = int(shared["stokes_shape"][0])
    if component_type == "healpix":
        return _load_healpix_branch(
            sorted_paths=sorted_paths,
            sorted_headers=sorted_headers,
            sorted_freqs=sorted_freqs,
            n_stokes_avail=n_stokes_avail,
            brightness_conversion=brightness_conversion,
            precision=precision,
            region=region,
            memmap_path=memmap_path,
            provenance=provenance,
        )
    if component_type == "point":
        return _load_point_branch(
            sorted_paths=sorted_paths,
            sorted_freqs=sorted_freqs,
            n_stokes_avail=n_stokes_avail,
            reference_frequency_hz=reference_frequency_hz,
            precision=precision,
            region=region,
            memmap_path=memmap_path,
            provenance=provenance,
        )
    raise ValueError(
        f"load_skyh5_multifile: unsupported component_type={component_type!r}. "
        "Only 'healpix' and 'point' are supported."
    )


def _load_healpix_branch(
    *,
    sorted_paths: list[str],
    sorted_headers: list[dict[str, Any]],
    sorted_freqs: np.ndarray,
    n_stokes_avail: int,
    brightness_conversion: str,
    precision: PrecisionConfig,
    region: SkyRegion | None,
    memmap_path: str | None,
    provenance: SkyProvenance | None,
) -> SkyModel:
    ref = sorted_headers[0]
    nside = int(ref["nside"])
    npix = hp.nside2npix(nside)
    hpx_order = ref["hpx_order"]
    coordinate_frame = ref["coordinate_frame"]
    hpx_inds_ref = np.asarray(ref["hpx_inds"], dtype=np.int64)

    for h in sorted_headers[1:]:
        if int(h["nside"]) != nside:
            raise ValueError(
                "load_skyh5_multifile: mismatched nside "
                f"({ref['filename']}={nside}, {h['filename']}={h['nside']})."
            )
        if h["hpx_order"] != hpx_order:
            raise ValueError(
                "load_skyh5_multifile: mismatched hpx_order "
                f"({ref['filename']}={hpx_order!r}, "
                f"{h['filename']}={h['hpx_order']!r})."
            )
        if h["coordinate_frame"] != coordinate_frame:
            raise ValueError(
                "load_skyh5_multifile: mismatched coordinate_frame "
                f"({ref['filename']}={coordinate_frame!r}, "
                f"{h['filename']}={h['coordinate_frame']!r})."
            )
        if not np.array_equal(h["hpx_inds"], hpx_inds_ref):
            raise ValueError(
                "load_skyh5_multifile: hpx_inds differ between "
                f"{ref['filename']} and {h['filename']}. All HEALPix files "
                "must share the same pixel index array."
            )

    is_sparse = hpx_inds_ref.size < npix
    is_nested = hpx_order == "nested"
    has_pol = n_stokes_avail >= 3
    has_v = n_stokes_avail >= 4

    if has_pol and brightness_conversion != "rayleigh-jeans":
        logger.info(
            "load_skyh5_multifile: forcing rayleigh-jeans conversion for "
            "polarized HEALPix data."
        )
        brightness_conversion = "rayleigh-jeans"

    # Ring-ordered pixel indices. For sparse maps, ``pix`` is 1-D of length
    # n_stored; stored Stokes rows are written in file order, with pix[j]
    # giving the ring-ordered pixel index of row j.
    pix: np.ndarray | None
    if is_sparse:
        pix = hp.nest2ring(nside, hpx_inds_ref) if is_nested else hpx_inds_ref.copy()
    else:
        pix = None

    # Region masking: apply once at the stored-axis level (identical across
    # files because all hpx_inds are validated identical).
    region_mask: np.ndarray | None = None
    sparse_keep_mask: np.ndarray | None = None
    if region is not None:
        region_mask = region.healpix_mask(nside, coordinate_frame=coordinate_frame)
        if is_sparse and pix is not None:
            sparse_keep_mask = region_mask[pix]
            pix = pix[sparse_keep_mask]
            logger.info(
                "load_skyh5_multifile: region retained %d/%d stored pixels",
                int(sparse_keep_mask.sum()),
                len(sparse_keep_mask),
            )

    if is_sparse:
        n_stored = 0 if pix is None else int(len(pix))
    else:
        n_stored = npix

    n_freq = len(sorted_paths)
    hp_dtype = get_sky_storage_dtype(precision, "healpix_maps")
    scratch = ensure_scratch_dir(memmap_path) if memmap_path is not None else None

    i_arr = allocate_cube((n_freq, n_stored), hp_dtype, scratch, "i_maps")
    q_arr = (
        allocate_cube((n_freq, n_stored), hp_dtype, scratch, "q_maps")
        if has_pol
        else None
    )
    u_arr = (
        allocate_cube((n_freq, n_stored), hp_dtype, scratch, "u_maps")
        if has_pol
        else None
    )
    v_arr = (
        allocate_cube((n_freq, n_stored), hp_dtype, scratch, "v_maps")
        if has_v
        else None
    )

    for fi, path in enumerate(sorted_paths):
        with h5py.File(path, "r") as f:
            stokes = f["Data/stokes"]
            unit = _decode_bytes(stokes.attrs.get("unit", ""))
            stokes_slice = np.asarray(stokes[:, 0, :], dtype=np.float64)
        # Convert Jy/sr -> K via Rayleigh-Jeans.  I[Jy/sr] * 1e-26 gives SI
        # (W/m^2/Hz/sr); then T_RJ = I_SI * c^2 / (2 k_B nu^2).
        if unit == "Jy / sr":
            from .constants import C_LIGHT, K_BOLTZMANN

            freq_hz = float(sorted_freqs[fi])
            conv = 1e-26 * (C_LIGHT**2) / (2.0 * K_BOLTZMANN * freq_hz**2)
            stokes_slice = stokes_slice * conv

        i_row = stokes_slice[0]
        q_row = stokes_slice[1] if has_pol else None
        u_row = stokes_slice[2] if has_pol else None
        v_row = stokes_slice[3] if has_v else None

        if is_sparse:
            # File rows are in file-stored order; if region masked, drop the
            # same positions from every file using the precomputed mask.
            if sparse_keep_mask is not None:
                i_row = i_row[sparse_keep_mask]
                if q_row is not None:
                    q_row = q_row[sparse_keep_mask]
                if u_row is not None:
                    u_row = u_row[sparse_keep_mask]
                if v_row is not None:
                    v_row = v_row[sparse_keep_mask]
        else:
            if is_nested:
                i_row = hp.reorder(i_row, n2r=True)
                if q_row is not None:
                    q_row = hp.reorder(q_row, n2r=True)
                if u_row is not None:
                    u_row = hp.reorder(u_row, n2r=True)
                if v_row is not None:
                    v_row = hp.reorder(v_row, n2r=True)
            if region_mask is not None:
                i_row = np.where(region_mask, i_row, 0.0)
                if q_row is not None:
                    q_row = np.where(region_mask, q_row, 0.0)
                if u_row is not None:
                    u_row = np.where(region_mask, u_row, 0.0)
                if v_row is not None:
                    v_row = np.where(region_mask, v_row, 0.0)

        i_arr[fi] = i_row.astype(hp_dtype)
        if q_arr is not None and q_row is not None:
            q_arr[fi] = q_row.astype(hp_dtype)
        if u_arr is not None and u_row is not None:
            u_arr[fi] = u_row.astype(hp_dtype)
        if v_arr is not None and v_row is not None:
            v_arr[fi] = v_row.astype(hp_dtype)

    i_arr = finalize_cube(i_arr, scratch, "i_maps")
    if q_arr is not None:
        q_arr = finalize_cube(q_arr, scratch, "q_maps")
    if u_arr is not None:
        u_arr = finalize_cube(u_arr, scratch, "u_maps")
    if v_arr is not None:
        v_arr = finalize_cube(v_arr, scratch, "v_maps")

    stokes_label = "I" + ("QU" if has_pol else "") + ("V" if has_v else "")
    model_name = "skyh5_multifile:" + os.path.basename(
        os.path.dirname(sorted_paths[0]) or sorted_paths[0]
    )
    logger.info(
        "load_skyh5_multifile: HEALPix cube %d freq x %d pix, stokes=%s",
        n_freq,
        n_stored,
        stokes_label,
    )

    sky = SkyModel(
        healpix=HealpixData(
            maps=i_arr,
            nside=nside,
            frequencies=sorted_freqs,
            coordinate_frame=coordinate_frame,
            hpx_inds=pix if is_sparse else None,
            q_maps=q_arr,
            u_maps=u_arr,
            v_maps=v_arr,
        ),
        model_name=model_name,
        brightness_conversion=brightness_conversion,
        _precision=precision,
    )
    if provenance is not None:
        sky = sky.replace(provenance=provenance)
    return sky


def _load_point_branch(
    *,
    sorted_paths: list[str],
    sorted_freqs: np.ndarray,
    n_stokes_avail: int,
    reference_frequency_hz: float | None,
    precision: PrecisionConfig,
    region: SkyRegion | None,
    memmap_path: str | None,
    provenance: SkyProvenance | None,
) -> SkyModel:
    ref_psky = PyRadioSkyModel()
    ref_psky.read(sorted_paths[0])
    n_components = int(ref_psky.Ncomponents)
    ra_rad = np.array(
        ref_psky.ra.rad if hasattr(ref_psky.ra, "rad") else ref_psky.ra,
        dtype=np.float64,
    )
    dec_rad = np.array(
        ref_psky.dec.rad if hasattr(ref_psky.dec, "rad") else ref_psky.dec,
        dtype=np.float64,
    )
    source_name = (
        np.asarray(ref_psky.name)
        if getattr(ref_psky, "name", None) is not None
        else None
    )

    has_pol = n_stokes_avail >= 3
    has_v = n_stokes_avail >= 4

    flux_dt = get_sky_storage_dtype(precision, "flux")
    src_dt = get_sky_storage_dtype(precision, "source_positions")
    si_dt = get_sky_storage_dtype(precision, "spectral_index")
    n_freq = len(sorted_paths)

    scratch = ensure_scratch_dir(memmap_path) if memmap_path is not None else None
    pc_flux = allocate_cube((n_freq, n_components), flux_dt, scratch, "pc_flux")
    pc_q = (
        allocate_cube((n_freq, n_components), flux_dt, scratch, "pc_q")
        if has_pol
        else None
    )
    pc_u = (
        allocate_cube((n_freq, n_components), flux_dt, scratch, "pc_u")
        if has_pol
        else None
    )
    pc_v = (
        allocate_cube((n_freq, n_components), flux_dt, scratch, "pc_v")
        if has_v
        else None
    )

    # First file already loaded as ref_psky; iterate for consistency.
    for fi, path in enumerate(sorted_paths):
        psky = ref_psky if fi == 0 else PyRadioSkyModel()
        if fi != 0:
            psky.read(path)
        if int(psky.Ncomponents) != n_components:
            raise ValueError(
                "load_skyh5_multifile: Ncomponents differs between "
                f"{sorted_paths[0]} ({n_components}) and {path} "
                f"({int(psky.Ncomponents)})."
            )
        ra_f = np.array(
            psky.ra.rad if hasattr(psky.ra, "rad") else psky.ra, dtype=np.float64
        )
        dec_f = np.array(
            psky.dec.rad if hasattr(psky.dec, "rad") else psky.dec, dtype=np.float64
        )
        if not np.allclose(ra_f, ra_rad, atol=1e-9) or not np.allclose(
            dec_f, dec_rad, atol=1e-9
        ):
            raise ValueError(
                "load_skyh5_multifile: source RA/Dec differ between "
                f"{sorted_paths[0]} and {path}. All point-source files must "
                "share the same source list."
            )
        stokes = psky.stokes
        if hasattr(stokes, "to_value"):
            stokes = stokes.to_value(u.Jy)
        stokes_arr = np.asarray(stokes, dtype=np.float64)  # (n_stokes, 1, N)
        pc_flux[fi] = stokes_arr[0, 0, :].astype(flux_dt)
        if pc_q is not None:
            pc_q[fi] = stokes_arr[1, 0, :].astype(flux_dt)
        if pc_u is not None:
            pc_u[fi] = stokes_arr[2, 0, :].astype(flux_dt)
        if pc_v is not None:
            pc_v[fi] = stokes_arr[3, 0, :].astype(flux_dt)

    pc_flux = finalize_cube(pc_flux, scratch, "pc_flux")
    if pc_q is not None:
        pc_q = finalize_cube(pc_q, scratch, "pc_q")
    if pc_u is not None:
        pc_u = finalize_cube(pc_u, scratch, "pc_u")
    if pc_v is not None:
        pc_v = finalize_cube(pc_v, scratch, "pc_v")

    # Reference channel (for PointSourceData.flux / ref_freq).
    if reference_frequency_hz is None:
        ref_idx = 0
    else:
        ref_idx = int(np.argmin(np.abs(sorted_freqs - float(reference_frequency_hz))))
    ref_freq_hz = float(sorted_freqs[ref_idx])

    flux_ref = np.asarray(pc_flux[ref_idx], dtype=flux_dt)
    q_ref = (
        np.asarray(pc_q[ref_idx], dtype=flux_dt)
        if pc_q is not None
        else np.zeros(n_components, dtype=flux_dt)
    )
    u_ref = (
        np.asarray(pc_u[ref_idx], dtype=flux_dt)
        if pc_u is not None
        else np.zeros(n_components, dtype=flux_dt)
    )
    v_ref = (
        np.asarray(pc_v[ref_idx], dtype=flux_dt)
        if pc_v is not None
        else np.zeros(n_components, dtype=flux_dt)
    )

    if region is not None:
        mask = region.contains(ra_rad, dec_rad)
    else:
        mask = np.ones(n_components, dtype=bool)

    ra_rad = ra_rad[mask].astype(src_dt, copy=False)
    dec_rad = dec_rad[mask].astype(src_dt, copy=False)
    flux_ref = flux_ref[mask]
    q_ref = q_ref[mask]
    u_ref = u_ref[mask]
    v_ref = v_ref[mask]
    pc_flux_out = np.ascontiguousarray(np.asarray(pc_flux)[:, mask])
    pc_q_out = (
        np.ascontiguousarray(np.asarray(pc_q)[:, mask]) if pc_q is not None else None
    )
    pc_u_out = (
        np.ascontiguousarray(np.asarray(pc_u)[:, mask]) if pc_u is not None else None
    )
    pc_v_out = (
        np.ascontiguousarray(np.asarray(pc_v)[:, mask]) if pc_v is not None else None
    )
    source_name = source_name[mask] if source_name is not None else None
    n_kept = int(mask.sum())

    ref_freq_arr = np.full(n_kept, ref_freq_hz, dtype=flux_dt)
    spectral_index = np.zeros(n_kept, dtype=si_dt)

    model_name = "skyh5_multifile:" + os.path.basename(
        os.path.dirname(sorted_paths[0]) or sorted_paths[0]
    )
    logger.info(
        "load_skyh5_multifile: point cube %d freq x %d sources",
        n_freq,
        n_kept,
    )

    from ._data import PointSpectrum

    sky = SkyModel(
        point=PointSourceData(
            ra_rad=ra_rad,
            dec_rad=dec_rad,
            flux=flux_ref,
            spectral_index=spectral_index,
            stokes_q=q_ref,
            stokes_u=u_ref,
            stokes_v=v_ref,
            ref_freq=ref_freq_arr,
            source_name=source_name,
            spectrum=PointSpectrum(
                flux=pc_flux_out,
                frequencies=sorted_freqs.astype(np.float64),
                stokes_q=pc_q_out,
                stokes_u=pc_u_out,
                stokes_v=pc_v_out,
            ),
        ),
        model_name=model_name,
        reference_frequency=ref_freq_hz,
        _precision=precision,
    )
    if provenance is not None:
        sky = sky.replace(provenance=provenance)
    return sky
