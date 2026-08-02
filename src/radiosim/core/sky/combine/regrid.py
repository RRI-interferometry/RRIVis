"""HEALPix regridding and shared frequency-grid helpers."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np

from ..containers import PointSourceData
from ..support.frequencies import validate_observation_frequencies
from ..support.healpy import lazy_healpy as hp

if TYPE_CHECKING:
    from ..containers.model import SkyModel


def _format_healpix_freq_grid(frequencies: np.ndarray) -> str:
    """Return a compact human-readable summary of a frequency grid."""
    freqs = np.asarray(frequencies, dtype=np.float64)
    if freqs.size == 0:
        return "0 channels"
    if freqs.size == 1:
        return f"1 channel ({freqs[0] / 1e6:.3f} MHz)"
    return f"{len(freqs)} channels ({freqs[0] / 1e6:.3f}–{freqs[-1] / 1e6:.3f} MHz)"


def _resolve_requested_healpix_frequencies(
    frequencies: Sequence[float] | np.ndarray | None,
) -> np.ndarray | None:
    """Resolve an explicit frequency request to a concrete array."""
    if frequencies is None:
        return None
    return validate_observation_frequencies(
        frequencies,
        label="requested HEALPix frequencies",
    )


def _resolve_common_healpix_frame(models: list[SkyModel]) -> str:
    """Return the shared HEALPix frame or raise on mismatches."""
    frames = {m.healpix.coordinate_frame for m in models if m.healpix is not None}
    if not frames:
        return "icrs"
    if len(frames) != 1:
        raise ValueError(
            "Cannot combine HEALPix models with different coordinate_frame "
            f"values: {sorted(frames)}."
        )
    return next(iter(frames))


def _resolve_common_healpix_ordering(models: list[SkyModel]) -> str:
    """Return the shared HEALPix ordering or raise on mismatches."""
    orderings = {m.healpix.ordering for m in models if m.healpix is not None}
    if not orderings:
        return "ring"
    if len(orderings) != 1:
        raise ValueError(
            "Cannot combine HEALPix models with different ordering "
            f"values: {sorted(orderings)}. Reorder one model with "
            "``HealpixData.reordered(...)`` before combining."
        )
    return next(iter(orderings))


def _point_source_healpix_indices(
    point: PointSourceData,
    nside: int,
    *,
    coordinate_frame: str,
    nest: bool = False,
) -> np.ndarray:
    if coordinate_frame == "galactic":
        from astropy.coordinates import SkyCoord

        galactic = SkyCoord(
            ra=point.ra_rad,
            dec=point.dec_rad,
            unit="rad",
            frame="icrs",
        ).galactic
        lon_rad = galactic.l.rad
        lat_rad = galactic.b.rad
    else:
        lon_rad = point.ra_rad
        lat_rad = point.dec_rad
    return hp.ang2pix(nside, np.pi / 2 - lat_rad, lon_rad, nest=nest)


def _validate_requested_healpix_grid(
    models: list[SkyModel],
    nside: int | None,
    frequencies: np.ndarray | None,
) -> None:
    """Reject requests that would silently ignore an existing HEALPix grid."""
    healpix_models = [m for m in models if m.healpix is not None]
    if not healpix_models:
        return

    ref_model = healpix_models[0]
    assert ref_model.healpix is not None
    ref_nside = ref_model.healpix.nside
    ref_freqs = np.asarray(ref_model.healpix.frequencies)

    if nside is not None and nside != ref_nside:
        raise ValueError(
            "Requested HEALPix nside does not match the existing HEALPix payload: "
            f"requested nside={nside}, but model '{ref_model.model_name or 'unnamed'}' "
            f"already carries nside={ref_nside}. "
            "Regrid that model first with "
            "`regrid_healpix_model(model, nside=...)` or omit nside to keep the "
            "existing grid."
        )

    if frequencies is not None and not np.array_equal(frequencies, ref_freqs):
        raise ValueError(
            "Requested HEALPix frequency grid does not match the existing "
            f"payload in model '{ref_model.model_name or 'unnamed'}': "
            f"existing grid = {_format_healpix_freq_grid(ref_freqs)}, "
            f"requested grid = {_format_healpix_freq_grid(frequencies)}. "
            "Exact frequency regridding is not implemented yet; regrid or "
            "regenerate the HEALPix payload first."
        )


def regrid_healpix_model(
    model: SkyModel,
    *,
    nside: int | None = None,
    frequencies: Sequence[float] | np.ndarray | None = None,
) -> SkyModel:
    """Explicitly regrid a HEALPix SkyModel.

    First pass policy:

    - ``nside`` changes are supported via ``healpy.ud_grade``.
    - frequency changes are exact-only; requested frequencies must match the
      existing HEALPix axis exactly.
    """
    if model.healpix is None:
        raise ValueError("regrid_healpix_model requires a SkyModel with HEALPix data.")

    requested_freqs = _resolve_requested_healpix_frequencies(frequencies)
    source_healpix = model.healpix.require_dense("regrid_healpix_model")
    current_freqs = np.asarray(source_healpix.frequencies, dtype=np.float64)
    if requested_freqs is not None and not np.array_equal(
        requested_freqs, current_freqs
    ):
        raise ValueError(
            "Exact frequency regridding is not implemented yet. "
            f"Existing grid = {_format_healpix_freq_grid(current_freqs)}, "
            f"requested grid = {_format_healpix_freq_grid(requested_freqs)}."
        )

    target_nside = source_healpix.nside if nside is None else nside
    if target_nside == source_healpix.nside:
        if requested_freqs is None or np.array_equal(requested_freqs, current_freqs):
            return model
        return model.replace(
            healpix=source_healpix.replace(frequencies=requested_freqs)
        )

    hp_order = "NESTED" if source_healpix.is_nested else "RING"

    def _regrid_rows(arr: np.ndarray) -> np.ndarray:
        rows = [
            hp.ud_grade(
                row,
                nside_out=target_nside,
                power=0,
                order_in=hp_order,
                order_out=hp_order,
            )
            for row in np.asarray(arr)
        ]
        return np.stack(rows, axis=0)

    q_maps = (
        None if source_healpix.q_maps is None else _regrid_rows(source_healpix.q_maps)
    )
    u_maps = (
        None if source_healpix.u_maps is None else _regrid_rows(source_healpix.u_maps)
    )
    v_maps = (
        None if source_healpix.v_maps is None else _regrid_rows(source_healpix.v_maps)
    )

    return model.replace(
        healpix=source_healpix.replace(
            maps=_regrid_rows(source_healpix.maps),
            nside=target_nside,
            frequencies=current_freqs if requested_freqs is None else requested_freqs,
            q_maps=q_maps,
            u_maps=u_maps,
            v_maps=v_maps,
        )
    )
