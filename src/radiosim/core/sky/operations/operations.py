"""Functional sky-model operations.

These helpers keep mutation-free transformations and memory-management
operations outside ``SkyModel`` itself.
"""

from __future__ import annotations

import os
import tempfile
import warnings
from typing import TYPE_CHECKING, Any

import healpy as hp
import numpy as np

from radiosim.utils.frequency import parse_frequency_config

from ..containers import (
    HealpixData,
    MonopoleConvention,
    PointSourceData,
)
from ..containers.constants import BrightnessConversion
from ..containers.spectral import per_source_reference_frequencies
from .convert import healpix_map_to_point_arrays, point_sources_to_healpix_maps

if TYPE_CHECKING:
    from ..containers.model import SkyModel


def materialize_healpix_model(
    sky: SkyModel,
    *,
    nside: int,
    frequencies: np.ndarray | None = None,
    obs_frequency_config: dict[str, Any] | None = None,
    ref_frequency: float | None = None,
    memmap_path: str | None = None,
    clear_other: bool = True,
) -> SkyModel:
    """Materialize a HEALPix payload from a point-source payload.

    By default the result is a HEALPix-only model — the original point
    payload is dropped so the simulator and downstream code see exactly
    one representation. Pass ``clear_other=False`` to keep the point
    payload alongside the new HEALPix payload (a hybrid model).

    Stokes I uses ``sky.brightness_conversion``.  Stokes Q/U/V use
    ``sky.polarization_brightness_conversion`` (defaults to Rayleigh-Jeans
    because polarized brightness can be negative; the Planck inverse is
    undefined for non-positive arguments).  Set the policy on the model
    itself via ``sky.replace(polarization_brightness_conversion="planck")``
    when you need Stokes-I-style non-linear conversion for polarised maps.
    """
    if sky.point is None:
        raise ValueError(
            "No point sources available for conversion. "
            "Load a point-source model first, for example with "
            "radiosim.core.sky.loaders.load_gleam()."
        )

    if frequencies is not None and obs_frequency_config is not None:
        raise ValueError(
            "Provide either 'frequencies' or 'obs_frequency_config', not both."
        )
    if frequencies is None and obs_frequency_config is not None:
        frequencies = parse_frequency_config(obs_frequency_config)
    if frequencies is None:
        raise ValueError(
            "Either 'frequencies' (np.ndarray) or 'obs_frequency_config' "
            "(dict) is required."
        )

    effective_ref_freq = per_source_reference_frequencies(
        sky.point,
        model_reference_frequency=sky.reference_frequency,
        fallback=ref_frequency,
    )
    if not np.any(effective_ref_freq > 0):
        raise ValueError(
            "ref_frequency must be provided when this SkyModel has no "
            "per-source ref_freq values and no reference_frequency. "
            "Set it via with_reference_frequency() or pass ref_frequency "
            "explicitly."
        )

    spectrum = sky.point.spectrum
    if sky.coherent_brightness_conversion:
        # Force Rayleigh-Jeans for both I and Q/U/V — the only convention
        # that handles the negative values polarised brightness can take
        # and the only one that gives a bit-exact HEALPix↔point round
        # trip for I.
        i_method = "rayleigh-jeans"
        pol_method = "rayleigh-jeans"
    else:
        i_method = sky.brightness_conversion.value
        pol_method = sky.polarization_brightness_conversion.value
    i_maps, q_maps, u_maps, v_maps, collision_stats = point_sources_to_healpix_maps(
        ra_rad=sky.point.ra_rad,
        dec_rad=sky.point.dec_rad,
        flux=sky.point.flux,
        spectral_index=sky.point.spectral_index,
        spectral_coeffs=sky.point.spectral_coeffs,
        stokes_q=sky.point.stokes_q,
        stokes_u=sky.point.stokes_u,
        stokes_v=sky.point.stokes_v,
        rotation_measure=(
            sky.point.polarization.rotation_measure
            if sky.point.polarization is not None
            else None
        ),
        nside=nside,
        frequencies=frequencies,
        ref_frequency=effective_ref_freq,
        brightness_conversion=BrightnessConversion(i_method),
        coordinate_frame="icrs",
        output_dtype=sky._healpix_dtype(),
        memmap_path=memmap_path,
        per_channel_flux=spectrum.flux if spectrum is not None else None,
        per_channel_stokes_q=spectrum.stokes_q if spectrum is not None else None,
        per_channel_stokes_u=spectrum.stokes_u if spectrum is not None else None,
        per_channel_stokes_v=spectrum.stokes_v if spectrum is not None else None,
        channel_frequencies=spectrum.frequencies if spectrum is not None else None,
        polarization_brightness_conversion=pol_method,
    )

    new_healpix = HealpixData(
        maps=i_maps,
        nside=nside,
        frequencies=frequencies,
        coordinate_frame="icrs",
        q_maps=q_maps,
        u_maps=u_maps,
        v_maps=v_maps,
        i_brightness_conversion=i_method,
        q_brightness_conversion=pol_method,
        u_brightness_conversion=pol_method,
        v_brightness_conversion=pol_method,
    )

    # Record pixel collisions in provenance.notes so a downstream consumer
    # can detect — programmatically — that per-source identities were
    # merged into individual pixels (and therefore that per-source spectral
    # indices, source IDs, etc. cannot be recovered).
    new_prov = sky.provenance
    n_merged = collision_stats.get("n_merged", 0)
    n_sources = collision_stats.get("n_sources", 0)
    if n_merged > 0 and n_sources > 0:
        pct = n_merged / n_sources
        note = f"pixel_collisions={n_merged}/{n_sources} ({pct:.2%}) at nside={nside}"
        merged_notes = (
            f"{sky.provenance.notes}; {note}" if sky.provenance.notes else note
        )
        new_prov = sky.provenance.replace(notes=merged_notes)

    if clear_other:
        return sky.replace(point=None, healpix=new_healpix, provenance=new_prov)
    return sky.replace(healpix=new_healpix, provenance=new_prov)


def materialize_point_sources_model(
    sky: SkyModel,
    frequency: float | None = None,
    flux_limit: float = 0.0,
    *,
    lossy: bool = False,
    clear_other: bool = True,
) -> SkyModel:
    """Materialize a point-source payload from a HEALPix payload.

    By default the result is a point-only model — the original HEALPix
    payload is dropped so the simulator and downstream code see exactly
    one representation. Pass ``clear_other=False`` to keep the HEALPix
    payload alongside the new point payload (a hybrid model).

    Stokes I uses ``sky.brightness_conversion``.  Stokes Q/U/V use
    ``sky.polarization_brightness_conversion`` (defaults to Rayleigh-Jeans
    because polarized brightness can be negative).  Set the policy on the
    model itself via
    ``sky.replace(polarization_brightness_conversion="planck")`` when you
    need Stokes-I-style non-linear conversion for polarised maps.
    """
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

    resolve_freq = freq or float(healpix.frequencies[0])
    if resolve_freq is None:
        raise ValueError(
            "frequency is required for HEALPix-to-point-source conversion."
        )
    fi = healpix.resolve_frequency_index(resolve_freq)
    temp_map = healpix.maps[fi]
    if sky.coherent_brightness_conversion:
        i_conversion: BrightnessConversion | str = "rayleigh-jeans"
        pol_conversion = "rayleigh-jeans"
    else:
        i_conversion = sky.brightness_conversion
        pol_conversion = sky.polarization_brightness_conversion.value
    arrays = healpix_map_to_point_arrays(
        temp_map,
        resolve_freq,
        i_conversion,
        healpix_q_maps=healpix.q_maps,
        healpix_u_maps=healpix.u_maps,
        healpix_v_maps=healpix.v_maps,
        observation_frequencies=healpix.frequencies,
        freq_index=fi,
        healpix_maps=healpix.maps,
        coordinate_frame=healpix.coordinate_frame,
        ref_freq_out=resolve_freq,
        polarization_brightness_conversion=pol_conversion,
        warn=False,
    )
    if flux_limit > 0:
        mask = arrays["flux"] >= flux_limit
        arrays = {
            key: (value[mask] if isinstance(value, np.ndarray) else value)
            for key, value in arrays.items()
        }

    new_point = PointSourceData(
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
    )

    # Update provenance: the new point catalog has *no* sub-pixel positional
    # information.  Its effective angular resolution is the HEALPix pixel
    # size at the nside used during conversion.  Without this update,
    # downstream code that trusts ``angular_resolution_rad`` would still see
    # the original diffuse template's resolution band, which no longer
    # describes the quantized point catalog we just produced.
    pixel_resolution_rad = float(hp.nside2resol(healpix.nside))
    new_prov = sky.provenance.replace(
        angular_resolution_rad=(pixel_resolution_rad, float(np.pi)),
    )

    if clear_other:
        return sky.replace(point=new_point, healpix=None, provenance=new_prov)
    return sky.replace(point=new_point, provenance=new_prov)


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
        path = tempfile.mkdtemp(prefix="radiosim_memmap_")

    os.makedirs(path, exist_ok=True)

    def _to_memmap(arr: np.ndarray, name: str) -> np.memmap:
        fpath = os.path.join(path, f"{name}.dat")
        mm = np.memmap(fpath, dtype=arr.dtype, mode="w+", shape=arr.shape)
        mm[:] = arr
        mm.flush()
        return np.memmap(fpath, dtype=arr.dtype, mode="r", shape=arr.shape)

    healpix = sky.healpix.replace(
        maps=_to_memmap(sky.healpix.maps, "i_maps"),
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
    )

    return sky.replace(healpix=healpix)


# =============================================================================
# Linear-polarisation diagnostics
# =============================================================================


def compute_linear_polarization(
    sky: SkyModel,
    *,
    frequency: float | None = None,
) -> dict[str, np.ndarray]:
    """Derive ``(P, χ, P/|I|)`` from a SkyModel's Stokes Q/U.

    For a HEALPix payload, returns dense maps shaped ``(npix,)`` when
    ``frequency`` is given (the closest channel is selected) or
    ``(n_freq, npix)`` when ``frequency=None``.  For a point-source
    payload, returns ``(n_sources,)`` arrays — Q/U here are intrinsic
    Stokes parameters, no per-frequency scaling is applied.

    Parameters
    ----------
    sky
        Sky model carrying Stokes Q and U.  ``ValueError`` is raised if
        either is absent.
    frequency
        Optional frequency (Hz) at which to slice a HEALPix payload.
        Ignored for point-source payloads.

    Returns
    -------
    dict
        Keys:

        - ``"P"`` : ``sqrt(Q² + U²)`` (linear polarisation amplitude).
        - ``"chi_deg"`` : ``0.5 · atan2(U, Q)`` in degrees, range
          ``(-90°, 90°]``.
        - ``"frac_pol"`` : ``P / |I|`` (fractional linear polarisation).
          ``nan`` where ``I = 0``.

    Raises
    ------
    ValueError
        If neither payload carries Q and U.
    """
    if sky.healpix is not None:
        if sky.healpix.q_maps is None or sky.healpix.u_maps is None:
            raise ValueError(
                "compute_linear_polarization requires Stokes Q and U HEALPix "
                "maps; the input has none.  Load a polarised template (e.g. "
                "PySM3 with synchrotron) or supply Q/U arrays explicitly."
            )
        if frequency is None:
            i_maps = sky.healpix.maps
            q_maps = sky.healpix.q_maps
            u_maps = sky.healpix.u_maps
        else:
            idx = sky.healpix.resolve_frequency_index(float(frequency))
            i_maps = sky.healpix.maps[idx]
            q_maps = sky.healpix.q_maps[idx]
            u_maps = sky.healpix.u_maps[idx]
        return _linear_pol_arrays(i_maps, q_maps, u_maps)

    if sky.point is not None:
        if sky.point.stokes_q is None or sky.point.stokes_u is None:
            raise ValueError(
                "compute_linear_polarization requires Stokes Q and U "
                "components on the point payload; got neither."
            )
        return _linear_pol_arrays(
            sky.point.flux,
            sky.point.stokes_q,
            sky.point.stokes_u,
        )

    raise ValueError("SkyModel carries no payload; cannot derive polarisation.")


def _linear_pol_arrays(
    i: np.ndarray,
    q: np.ndarray,
    u: np.ndarray,
) -> dict[str, np.ndarray]:
    q_arr = np.asarray(q, dtype=float)
    u_arr = np.asarray(u, dtype=float)
    i_arr = np.asarray(i, dtype=float)
    p = np.hypot(q_arr, u_arr)
    chi_rad = 0.5 * np.arctan2(u_arr, q_arr)
    chi_deg = np.degrees(chi_rad)
    with np.errstate(divide="ignore", invalid="ignore"):
        frac_pol = p / np.abs(i_arr)
    frac_pol = np.where(i_arr == 0.0, np.nan, frac_pol)
    return {"P": p, "chi_deg": chi_deg, "frac_pol": frac_pol}


# =============================================================================
# Monopole bookkeeping operations
# =============================================================================


def _coerce_monopole_convention(
    convention: MonopoleConvention | str,
) -> MonopoleConvention:
    if isinstance(convention, MonopoleConvention):
        return convention
    return MonopoleConvention(convention)


def with_monopole(
    sky: SkyModel,
    value_k: float,
    convention: MonopoleConvention | str = MonopoleConvention.ABSOLUTE_NO_CMB,
) -> SkyModel:
    """Return a new :class:`SkyModel` with ``value_k`` added to the sky monopole.

    For HEALPix payloads, ``value_k`` is added uniformly to every pixel of
    the Stokes-I cube (Q/U/V are zero-mean by construction and are not
    touched).  For pure point-source payloads the map arrays are unchanged —
    only the provenance is updated to advertise the new monopole.

    Parameters
    ----------
    sky
        Input sky model.
    value_k
        Brightness-temperature monopole to add, in Kelvin.
    convention
        Monopole convention to declare on the returned model.  Use
        :class:`MonopoleConvention.ABSOLUTE_WITH_CMB` when re-adding the CMB,
        :class:`MonopoleConvention.ABSOLUTE_NO_CMB` otherwise.

    Returns
    -------
    SkyModel
        A new model with the DC level shifted and provenance updated.
    """
    convention = _coerce_monopole_convention(convention)
    if np.ndim(value_k) != 0:
        raise TypeError(
            "with_monopole(value_k=...) must be a scalar (a uniform DC shift "
            "applied to every pixel of the Stokes-I cube); received an "
            f"array-like with shape {np.asarray(value_k).shape}.  Did you "
            "intend to pass a per-channel mean?  with_monopole only supports "
            "a single full-sky scalar."
        )
    value_k = float(value_k)
    if sky.provenance.is_partial_sky:
        raise ValueError(
            "with_monopole requires a full-sky model; partial-sky products do "
            "not have a well-defined global monopole."
        )

    old_prov = sky.provenance
    old_monopole = old_prov.monopole_k
    new_monopole = old_monopole + value_k if old_monopole is not None else value_k
    new_prov = old_prov.replace(
        monopole_convention=convention,
        monopole_k=new_monopole,
    )

    if sky.healpix is None:
        return sky.replace(provenance=new_prov)

    new_maps = sky.healpix.maps + np.asarray(value_k, dtype=sky.healpix.maps.dtype)
    new_healpix = sky.healpix.replace(maps=new_maps)
    return sky.replace(healpix=new_healpix, provenance=new_prov)


def with_monopole_subtracted(sky: SkyModel) -> SkyModel:
    """Return a new :class:`SkyModel` with the per-frequency Stokes-I mean removed.

    For HEALPix payloads, the pixel-weighted mean of each frequency channel is
    subtracted from the Stokes-I cube (Q/U/V channels are left untouched —
    they are already mean-zero by construction).  For pure point-source
    payloads only the provenance is updated (no array modification).  In
    both cases the returned model's ``provenance.monopole_convention`` becomes
    :class:`MonopoleConvention.MEAN_SUBTRACTED` and ``monopole_k`` is set to 0.

    Raises
    ------
    ValueError
        If the input model already has ``monopole_convention = MEAN_SUBTRACTED``
        (idempotent subtraction on an already-zero-mean sky is a user error).
    """
    if sky.provenance.monopole_convention is MonopoleConvention.MEAN_SUBTRACTED:
        raise ValueError(
            "SkyModel is already mean-subtracted "
            "(provenance.monopole_convention=MEAN_SUBTRACTED); "
            "with_monopole_subtracted would subtract the mean twice."
        )
    if sky.provenance.is_partial_sky:
        raise ValueError(
            "with_monopole_subtracted requires a full-sky model; partial-sky "
            "products do not have a well-defined global monopole."
        )

    new_prov = sky.provenance.replace(
        monopole_convention=MonopoleConvention.MEAN_SUBTRACTED,
        monopole_k=0.0,
    )

    if sky.healpix is None:
        return sky.replace(provenance=new_prov)

    maps = sky.healpix.maps
    # Per-channel pixel-area-weighted mean: pixels are equal-area on the HEALPix
    # grid so a plain mean over stored pixels is the correct solid-angle average.
    means = maps.mean(axis=1, keepdims=True)
    new_maps = maps - means.astype(maps.dtype)

    new_healpix = sky.healpix.replace(maps=new_maps)
    return sky.replace(healpix=new_healpix, provenance=new_prov)
