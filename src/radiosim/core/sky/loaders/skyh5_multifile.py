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

Spatial ``region`` filtering follows the file-loader client-side convention
(see :mod:`radiosim.core.sky.support.region_filter`).
"""

from __future__ import annotations

import glob as _glob
import logging
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import astropy.units as u
import numpy as np

from radiosim.utils.frequency import parse_frequency_config

from ..containers.model import SkyModel
from ..registry import loader_registry
from ..support.allocation import allocate_cube, ensure_scratch_dir, finalize_cube
from ..support.brightness import skyh5_stokes_slice_to_kelvin
from ..support.healpix_geometry import ordered_row
from ..support.healpy import lazy_healpy as hp
from ..support.point_builder import point_source_data_from_mapping
from ..support.precision import get_sky_storage_dtype
from ..support.quantities import to_value
from ._healpix_builder import build_healpix_from_stokes_cube, extract_stokes_component

if TYPE_CHECKING:
    from radiosim.core.precision import PrecisionConfig

    from ..containers import SkyProvenance
    from ..operations.region import SkyRegion

logger = logging.getLogger(__name__)

_FREQ_DUPLICATE_TOL_HZ = 1.0


def _h5py() -> Any:
    """Import ``h5py`` on demand with a friendly error (optional dependency)."""
    try:
        import h5py
    except ImportError as exc:
        raise ImportError(
            "Reading skyh5 files requires the optional 'h5py' package. "
            "Install it with `pip install h5py`."
        ) from exc
    return h5py


def _pyradiosky_cls() -> Any:
    """Import ``pyradiosky.SkyModel`` on demand with a friendly error."""
    try:
        from pyradiosky import SkyModel as _cls
    except ImportError as exc:
        raise ImportError(
            "Reading skyh5 multi-file sets requires the optional 'pyradiosky' "
            "package. Install it with `pip install pyradiosky`."
        ) from exc
    return _cls


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
    with _h5py().File(filename, "r") as f:
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


def _resolve_expected_frequency_grid(
    frequencies: np.ndarray | None,
    obs_frequency_config: dict[str, Any] | None,
) -> np.ndarray | None:
    """Resolve the caller-supplied cross-check frequency grid.

    The multi-file loader never resamples: an explicit grid (either an
    array via ``frequencies`` or a config dict via ``obs_frequency_config``)
    is only used to assert that the file-derived channel grid matches. When
    neither is supplied the file grid is authoritative (returns ``None``).

    Raises
    ------
    ValueError
        If both ``frequencies`` and ``obs_frequency_config`` are supplied.
    """
    if frequencies is not None and obs_frequency_config is not None:
        raise ValueError(
            "load_skyh5_multifile: provide at most one of 'frequencies' or "
            "'obs_frequency_config' (got both)."
        )
    if frequencies is not None:
        return np.asarray(frequencies, dtype=np.float64)
    if obs_frequency_config is not None:
        return np.asarray(
            parse_frequency_config(obs_frequency_config), dtype=np.float64
        )
    return None


def _maybe_cross_check_frequencies(
    sorted_freqs: np.ndarray,
    frequencies: np.ndarray | None,
) -> None:
    if frequencies is None:
        return
    requested = np.sort(np.asarray(frequencies, dtype=np.float64))
    if requested.shape != sorted_freqs.shape or not np.allclose(
        requested, sorted_freqs, atol=_FREQ_DUPLICATE_TOL_HZ
    ):
        raise ValueError(
            "load_skyh5_multifile: requested `frequencies` do not match the "
            "file-derived channel grid within 1 Hz. Omit the argument to use "
            "the file grid, or align it exactly."
        )


@loader_registry.register_loader(
    "skyh5_multifile",
    representations=("point_sources", "healpix_map"),
    config_section="skyh5_multifile",
    use_flag="use_skyh5_multifile",
    category="file",
    requires_file=True,
    network_service=None,
    aliases=["pyradiosky_multifile"],
    config_fields=[
        "file_glob",
        "filenames",
        "brightness_conversion",
        "reference_frequency_hz",
    ],
)
def load_skyh5_multifile(
    file_glob: str | None = None,
    filenames: list[str] | None = None,
    brightness_conversion: str = "planck",
    reference_frequency_hz: float | None = None,
    *,
    precision: PrecisionConfig,
    frequencies: np.ndarray | None = None,
    obs_frequency_config: dict[str, Any] | None = None,
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
    brightness_conversion : str, default ``"planck"``
        Brightness conversion for HEALPix data (matching the sibling
        loaders). Forced to ``"rayleigh-jeans"`` when polarization is
        present.
    reference_frequency_hz : float, optional
        Observation frequency to use as the reference channel for
        ``PointSourceData.flux`` / ``ref_freq``.  Defaults to the lowest
        channel frequency.
    frequencies : np.ndarray, optional
        Explicit observation-frequency grid (Hz). When given, it is
        cross-checked against the file-derived channel grid (must match
        within 1 Hz); it does not resample the data.
    obs_frequency_config : dict, optional
        Frequency-configuration dict (e.g. ``starting_frequency`` /
        ``frequency_interval`` / ``frequency_bandwidth`` / ``frequency_unit``
        or a raw ``frequencies_hz`` array). Used as the cross-check grid when
        ``frequencies`` is not supplied. Mutually exclusive with
        ``frequencies``.

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
    expected_grid = _resolve_expected_frequency_grid(frequencies, obs_frequency_config)
    _maybe_cross_check_frequencies(sorted_freqs, expected_grid)

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
        if region is not None:
            pix = (
                hp.nest2ring(nside, hpx_inds_ref) if is_nested else hpx_inds_ref.copy()
            )

    builder_hpx_inds = (
        pix if pix is not None and (is_sparse or region is not None) else None
    )

    def _ring_ordered_row(row: np.ndarray) -> np.ndarray:
        # skyh5 never reaches the dense-scatter branch: sparse maps always set
        # ``builder_hpx_inds`` (the cube builder owns the scatter), so ``pix`` is
        # left ``None`` here and reordering is the only non-builder path.
        return ordered_row(
            row,
            builder_handles_scatter=builder_hpx_inds is not None,
            pix=None,
            npix=npix,
            is_nested=is_nested,
        )

    def _iter_stokes_rows():
        for fi, path in enumerate(sorted_paths):
            with _h5py().File(path, "r") as f:
                stokes = f["Data/stokes"]
                unit = _decode_bytes(stokes.attrs.get("unit", ""))
                stokes_slice = np.asarray(stokes[:, 0, :], dtype=np.float64)
            stokes_slice = skyh5_stokes_slice_to_kelvin(
                stokes_slice,
                unit=unit,
                freq_hz=float(sorted_freqs[fi]),
            )

            i_row = extract_stokes_component(stokes_slice, "I", n_stokes_avail)
            if i_row is None:
                raise ValueError(
                    "load_skyh5_multifile: HEALPix data is missing Stokes I."
                )
            q_row = (
                extract_stokes_component(stokes_slice, "Q", n_stokes_avail)
                if has_pol
                else None
            )
            u_row = (
                extract_stokes_component(stokes_slice, "U", n_stokes_avail)
                if has_pol
                else None
            )
            v_row = (
                extract_stokes_component(stokes_slice, "V", n_stokes_avail)
                if has_v
                else None
            )
            yield (
                _ring_ordered_row(i_row),
                _ring_ordered_row(q_row) if q_row is not None else None,
                _ring_ordered_row(u_row) if u_row is not None else None,
                _ring_ordered_row(v_row) if v_row is not None else None,
            )

    healpix = build_healpix_from_stokes_cube(
        stokes_rows=_iter_stokes_rows(),
        nside=nside,
        frequencies=sorted_freqs,
        coordinate_frame=coordinate_frame,
        hpx_inds=builder_hpx_inds,
        region=region,
        precision=precision,
        memmap_path=memmap_path,
    )

    stokes_label = "I" + ("QU" if has_pol else "") + ("V" if has_v else "")
    model_name = "skyh5_multifile:" + os.path.basename(
        os.path.dirname(sorted_paths[0]) or sorted_paths[0]
    )
    logger.info(
        "load_skyh5_multifile: HEALPix cube %d freq x %d pix, stokes=%s",
        len(sorted_paths),
        healpix.n_pixels,
        stokes_label,
    )

    sky = SkyModel(
        healpix=healpix,
        model_name=model_name,
        brightness_conversion=brightness_conversion,
        precision=precision,
    )
    if provenance is not None:
        sky = sky.replace(provenance=provenance)
    return sky


@dataclass
class _PointChannelCubes:
    """Per-channel Stokes cubes plus the shared source geometry.

    ``flux``/``q``/``u``/``v`` are ``(n_freq, n_components)`` arrays (Q/U/V
    are ``None`` when the inputs carry no polarization). ``ra_rad`` /
    ``dec_rad`` / ``source_name`` describe the (shared) source list.
    """

    ra_rad: np.ndarray
    dec_rad: np.ndarray
    source_name: np.ndarray | None
    flux: np.ndarray
    q: np.ndarray | None
    u: np.ndarray | None
    v: np.ndarray | None


def _read_point_geometry(
    ref_psky: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Extract the shared (ra, dec, name) source list from the first file."""
    ra_rad = np.array(to_value(ref_psky.ra, u.rad), dtype=np.float64)
    dec_rad = np.array(to_value(ref_psky.dec, u.rad), dtype=np.float64)
    source_name = (
        np.asarray(ref_psky.name)
        if getattr(ref_psky, "name", None) is not None
        else None
    )
    return ra_rad, dec_rad, source_name


def _read_point_channel_cubes(
    *,
    sorted_paths: list[str],
    n_stokes_avail: int,
    precision: PrecisionConfig,
    memmap_path: str | None,
) -> _PointChannelCubes:
    """Read every channel file into per-frequency Stokes cubes.

    Validates that all files share the same source list (Ncomponents +
    RA/Dec) and stacks the per-channel Stokes tables along the freq axis.
    """
    ref_psky = _pyradiosky_cls()()
    ref_psky.read(sorted_paths[0])
    n_components = int(ref_psky.Ncomponents)
    ra_rad, dec_rad, source_name = _read_point_geometry(ref_psky)

    has_pol = n_stokes_avail >= 3
    has_v = n_stokes_avail >= 4

    flux_dt = get_sky_storage_dtype(precision, "flux")
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
        psky = ref_psky if fi == 0 else _pyradiosky_cls()()
        if fi != 0:
            psky.read(path)
        if int(psky.Ncomponents) != n_components:
            raise ValueError(
                "load_skyh5_multifile: Ncomponents differs between "
                f"{sorted_paths[0]} ({n_components}) and {path} "
                f"({int(psky.Ncomponents)})."
            )
        ra_f = np.array(to_value(psky.ra, u.rad), dtype=np.float64)
        dec_f = np.array(to_value(psky.dec, u.rad), dtype=np.float64)
        if not np.allclose(ra_f, ra_rad, atol=1e-9) or not np.allclose(
            dec_f, dec_rad, atol=1e-9
        ):
            raise ValueError(
                "load_skyh5_multifile: source RA/Dec differ between "
                f"{sorted_paths[0]} and {path}. All point-source files must "
                "share the same source list."
            )
        stokes_arr = np.asarray(
            to_value(psky.stokes, u.Jy), dtype=np.float64
        )  # (n_stokes, 1, N)
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

    return _PointChannelCubes(
        ra_rad=ra_rad,
        dec_rad=dec_rad,
        source_name=source_name,
        flux=pc_flux,
        q=pc_q,
        u=pc_u,
        v=pc_v,
    )


def _select_reference_channel(
    sorted_freqs: np.ndarray,
    reference_frequency_hz: float | None,
) -> tuple[int, float]:
    """Return the (index, frequency) of the reference channel."""
    if reference_frequency_hz is None:
        ref_idx = 0
    else:
        ref_idx = int(np.argmin(np.abs(sorted_freqs - float(reference_frequency_hz))))
    return ref_idx, float(sorted_freqs[ref_idx])


def _point_region_mask(
    cubes: _PointChannelCubes,
    region: SkyRegion | None,
) -> np.ndarray:
    """Boolean keep-mask over sources for an optional spatial region."""
    if region is not None:
        return np.asarray(region.contains(cubes.ra_rad, cubes.dec_rad), dtype=bool)
    return np.ones(cubes.ra_rad.shape[0], dtype=bool)


def _assemble_point_sky(
    *,
    cubes: _PointChannelCubes,
    sorted_freqs: np.ndarray,
    ref_idx: int,
    ref_freq_hz: float,
    mask: np.ndarray,
    sorted_paths: list[str],
    precision: PrecisionConfig,
    provenance: SkyProvenance | None,
) -> SkyModel:
    """Mask the cubes and build the final point-source :class:`SkyModel`."""
    from ..containers import PointSpectrum

    flux_dt = get_sky_storage_dtype(precision, "flux")
    src_dt = get_sky_storage_dtype(precision, "source_positions")
    si_dt = get_sky_storage_dtype(precision, "spectral_index")
    n_components = cubes.ra_rad.shape[0]

    def _ref_row(cube: np.ndarray | None) -> np.ndarray:
        if cube is not None:
            return np.asarray(cube[ref_idx], dtype=flux_dt)
        return np.zeros(n_components, dtype=flux_dt)

    flux_ref = _ref_row(cubes.flux)
    q_ref = _ref_row(cubes.q)
    u_ref = _ref_row(cubes.u)
    v_ref = _ref_row(cubes.v)

    def _masked_cube(cube: np.ndarray | None) -> np.ndarray | None:
        if cube is None:
            return None
        return np.ascontiguousarray(np.asarray(cube)[:, mask])

    ra_rad = cubes.ra_rad[mask].astype(src_dt, copy=False)
    dec_rad = cubes.dec_rad[mask].astype(src_dt, copy=False)
    source_name = cubes.source_name[mask] if cubes.source_name is not None else None
    n_kept = int(mask.sum())

    model_name = "skyh5_multifile:" + os.path.basename(
        os.path.dirname(sorted_paths[0]) or sorted_paths[0]
    )
    logger.info(
        "load_skyh5_multifile: point cube %d freq x %d sources",
        len(sorted_paths),
        n_kept,
    )

    sky = SkyModel(
        point=point_source_data_from_mapping(
            {
                "ra_rad": ra_rad,
                "dec_rad": dec_rad,
                "flux": flux_ref[mask],
                "spectral_index": np.zeros(n_kept, dtype=si_dt),
                "stokes_q": q_ref[mask],
                "stokes_u": u_ref[mask],
                "stokes_v": v_ref[mask],
                "ref_freq": np.full(n_kept, ref_freq_hz, dtype=flux_dt),
                "source_name": source_name,
                "spectrum": PointSpectrum(
                    flux=_masked_cube(cubes.flux),
                    frequencies=sorted_freqs.astype(np.float64),
                    stokes_q=_masked_cube(cubes.q),
                    stokes_u=_masked_cube(cubes.u),
                    stokes_v=_masked_cube(cubes.v),
                ),
            },
            precision=precision,
        ),
        model_name=model_name,
        reference_frequency=ref_freq_hz,
        precision=precision,
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
    cubes = _read_point_channel_cubes(
        sorted_paths=sorted_paths,
        n_stokes_avail=n_stokes_avail,
        precision=precision,
        memmap_path=memmap_path,
    )
    ref_idx, ref_freq_hz = _select_reference_channel(
        sorted_freqs, reference_frequency_hz
    )
    mask = _point_region_mask(cubes, region)
    return _assemble_point_sky(
        cubes=cubes,
        sorted_freqs=sorted_freqs,
        ref_idx=ref_idx,
        ref_freq_hz=ref_freq_hz,
        mask=mask,
        sorted_paths=sorted_paths,
        precision=precision,
        provenance=provenance,
    )
