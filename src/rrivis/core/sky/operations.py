"""Functional sky-model operations.

These helpers keep mutation-free transformations and memory-management
operations outside ``SkyModel`` itself.
"""

from __future__ import annotations

import tempfile
import warnings
from typing import TYPE_CHECKING, Any

import healpy as hp
import numpy as np

from ._data import HealpixData, PointSourceData

if TYPE_CHECKING:
    from .model import SkyModel


def materialize_healpix_model(
    sky: SkyModel,
    *,
    nside: int,
    frequencies: np.ndarray | None = None,
    obs_frequency_config: dict[str, Any] | None = None,
    ref_frequency: float | None = None,
    memmap_path: str | None = None,
) -> SkyModel:
    """Materialize a HEALPix payload from a point-source payload."""
    if sky.point is None:
        raise ValueError(
            "No point sources available for conversion. "
            "Load a point-source model first, for example with "
            "rrivis.core.sky.loaders.load_gleam()."
        )

    if frequencies is not None and obs_frequency_config is not None:
        raise ValueError(
            "Provide either 'frequencies' or 'obs_frequency_config', not both."
        )
    if frequencies is None and obs_frequency_config is not None:
        from rrivis.utils.frequency import parse_frequency_config

        frequencies = parse_frequency_config(obs_frequency_config)
    if frequencies is None:
        raise ValueError(
            "Either 'frequencies' (np.ndarray) or 'obs_frequency_config' "
            "(dict) is required."
        )

    from .convert import point_sources_to_healpix_maps

    effective_ref_freq: float | np.ndarray | None = None
    if sky.point.ref_freq is not None and np.any(sky.point.ref_freq > 0):
        effective_ref_freq = sky.point.ref_freq
    else:
        effective_ref_freq = ref_frequency or sky.reference_frequency
        if effective_ref_freq is None:
            raise ValueError(
                "ref_frequency must be provided when this SkyModel has no "
                "per-source ref_freq values and no reference_frequency. "
                "Set it via with_reference_frequency() or pass ref_frequency "
                "explicitly."
            )

    i_maps, q_maps, u_maps, v_maps = point_sources_to_healpix_maps(
        ra_rad=sky.point.ra_rad,
        dec_rad=sky.point.dec_rad,
        flux=sky.point.flux,
        spectral_index=sky.point.spectral_index,
        spectral_coeffs=sky.point.spectral_coeffs,
        stokes_q=sky.point.stokes_q,
        stokes_u=sky.point.stokes_u,
        stokes_v=sky.point.stokes_v,
        rotation_measure=sky.point.rotation_measure,
        nside=nside,
        frequencies=frequencies,
        ref_frequency=effective_ref_freq,
        brightness_conversion=sky.brightness_conversion,
        coordinate_frame="icrs",
        output_dtype=sky._healpix_dtype(),
        memmap_path=memmap_path,
    )

    return sky._replace(
        healpix=HealpixData(
            maps=i_maps,
            nside=nside,
            frequencies=frequencies,
            coordinate_frame="icrs",
            q_maps=q_maps,
            u_maps=u_maps,
            v_maps=v_maps,
            i_brightness_conversion=sky.brightness_conversion.value,
        ),
    )


def materialize_point_sources_model(
    sky: SkyModel,
    frequency: float | None = None,
    flux_limit: float = 0.0,
    *,
    lossy: bool = False,
) -> SkyModel:
    """Materialize a point-source payload from a HEALPix payload."""
    if sky.point is not None:
        return sky

    if sky.healpix is None:
        raise ValueError("No HEALPix payload available for conversion.")
    if not lossy:
        raise ValueError(
            "HEALPix-to-point-source conversion is lossy. "
            "Call materialize_point_sources_model(..., lossy=True) to opt in."
        )

    freq = frequency or sky.reference_frequency
    healpix = sky.healpix.to_dense() if sky.healpix.is_sparse else sky.healpix
    n_freq = len(healpix.frequencies)
    resol_arcmin = float(hp.nside2resol(healpix.nside, arcmin=True))
    warnings.warn(
        f"HEALPix-to-point-source conversion is lossy: positions are "
        f"quantized to pixel centers (nside={healpix.nside}, "
        f"~{resol_arcmin:.1f}' resolution) and spectral indices are "
        f"fit from {n_freq} channels. Use 'healpix_map' mode for "
        f"full-fidelity diffuse emission.",
        stacklevel=2,
    )

    from .convert import healpix_map_to_point_arrays

    resolve_freq = freq or float(healpix.frequencies[0])
    if resolve_freq is None:
        raise ValueError(
            "frequency is required for HEALPix-to-point-source conversion."
        )
    fi = sky.resolve_frequency_index(resolve_freq)
    temp_map = healpix.maps[fi]
    arrays = healpix_map_to_point_arrays(
        temp_map,
        resolve_freq,
        sky.brightness_conversion,
        healpix_q_maps=healpix.q_maps,
        healpix_u_maps=healpix.u_maps,
        healpix_v_maps=healpix.v_maps,
        observation_frequencies=healpix.frequencies,
        freq_index=fi,
        healpix_maps=healpix.maps,
        coordinate_frame=healpix.coordinate_frame,
        ref_freq_out=resolve_freq,
        warn=False,
    )
    if flux_limit > 0:
        mask = arrays["flux"] >= flux_limit
        arrays = {
            key: (value[mask] if isinstance(value, np.ndarray) else value)
            for key, value in arrays.items()
        }

    return sky._replace(
        point=PointSourceData(
            ra_rad=arrays["ra_rad"],
            dec_rad=arrays["dec_rad"],
            flux=arrays["flux"],
            spectral_index=arrays["spectral_index"],
            stokes_q=arrays["stokes_q"],
            stokes_u=arrays["stokes_u"],
            stokes_v=arrays["stokes_v"],
            ref_freq=arrays["ref_freq"],
            rotation_measure=arrays["rotation_measure"],
            major_arcsec=arrays["major_arcsec"],
            minor_arcsec=arrays["minor_arcsec"],
            pa_deg=arrays["pa_deg"],
            spectral_coeffs=arrays["spectral_coeffs"],
        ),
    )


def with_memmap_backing(
    sky: SkyModel,
    path: str | None = None,
) -> SkyModel:
    """Return a copy with HEALPix maps backed by memory-mapped files."""
    if sky.healpix is None:
        raise ValueError(
            "No HEALPix maps to back with memmap. Materialize a HEALPix payload first."
        )

    if path is None:
        path = tempfile.mkdtemp(prefix="rrivis_memmap_")

    import os

    os.makedirs(path, exist_ok=True)

    def _to_memmap(arr: np.ndarray, name: str) -> np.memmap:
        fpath = os.path.join(path, f"{name}.dat")
        mm = np.memmap(fpath, dtype=arr.dtype, mode="w+", shape=arr.shape)
        mm[:] = arr
        mm.flush()
        return np.memmap(fpath, dtype=arr.dtype, mode="r", shape=arr.shape)

    healpix = HealpixData(
        maps=_to_memmap(sky.healpix.maps, "i_maps"),
        nside=sky.healpix.nside,
        frequencies=sky.healpix.frequencies,
        coordinate_frame=sky.healpix.coordinate_frame,
        hpx_inds=sky.healpix.hpx_inds,
        q_maps=(
            _to_memmap(sky.healpix.q_maps, "q_maps")
            if sky.healpix.q_maps is not None
            else None
        ),
        u_maps=(
            _to_memmap(sky.healpix.u_maps, "u_maps")
            if sky.healpix.u_maps is not None
            else None
        ),
        v_maps=(
            _to_memmap(sky.healpix.v_maps, "v_maps")
            if sky.healpix.v_maps is not None
            else None
        ),
        i_unit=sky.healpix.i_unit,
        q_unit=sky.healpix.q_unit,
        u_unit=sky.healpix.u_unit,
        v_unit=sky.healpix.v_unit,
        i_brightness_conversion=sky.healpix.i_brightness_conversion,
        q_brightness_conversion=sky.healpix.q_brightness_conversion,
        u_brightness_conversion=sky.healpix.u_brightness_conversion,
        v_brightness_conversion=sky.healpix.v_brightness_conversion,
    )

    return sky._replace(healpix=healpix)
