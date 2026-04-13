# rrivis/core/sky/_data.py
"""Inner data containers for SkyModel.

This module is a LEAF dependency — it imports only numpy and healpy.
No imports from model.py, convert.py, combine.py, or loaders.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TypedDict

import healpy as hp
import numpy as np

# =============================================================================
# SourceArrays TypedDict
# =============================================================================


class SourceArrays(TypedDict):
    """Return type for point-source array extraction.

    Keys match the interface consumed by ``visibility.py`` and align with
    ``PointSourceData`` field names.
    """

    ra_rad: np.ndarray
    dec_rad: np.ndarray
    flux: np.ndarray
    spectral_index: np.ndarray
    stokes_q: np.ndarray
    stokes_u: np.ndarray
    stokes_v: np.ndarray
    ref_freq: np.ndarray
    rotation_measure: np.ndarray | None
    major_arcsec: np.ndarray | None
    minor_arcsec: np.ndarray | None
    pa_deg: np.ndarray | None
    spectral_coeffs: np.ndarray | None


def empty_source_arrays() -> SourceArrays:
    """Return an empty ``SourceArrays`` dict (zero-length float64 arrays)."""
    z = np.zeros(0, dtype=np.float64)
    return {
        "ra_rad": z.copy(),
        "dec_rad": z.copy(),
        "flux": z.copy(),
        "spectral_index": z.copy(),
        "stokes_q": z.copy(),
        "stokes_u": z.copy(),
        "stokes_v": z.copy(),
        "ref_freq": z.copy(),
        "rotation_measure": None,
        "major_arcsec": None,
        "minor_arcsec": None,
        "pa_deg": None,
        "spectral_coeffs": None,
    }


# =============================================================================
# PointSourceData
# =============================================================================


@dataclass(frozen=True)
class PointSourceData:
    """Columnar arrays for point-source sky model.

    All core arrays have shape ``(N,)``.  This is always fully populated
    (even if zero-length for an empty sky).  No field is ever None for
    the core arrays — an empty model uses zero-length arrays.

    Optional extension arrays (rotation_measure, morphology, spectral_coeffs)
    are None when that feature is absent for ALL sources.
    """

    ra_rad: np.ndarray
    dec_rad: np.ndarray
    flux: np.ndarray
    spectral_index: np.ndarray
    stokes_q: np.ndarray
    stokes_u: np.ndarray
    stokes_v: np.ndarray
    ref_freq: np.ndarray

    # Optional per-source extensions
    rotation_measure: np.ndarray | None = None
    major_arcsec: np.ndarray | None = None
    minor_arcsec: np.ndarray | None = None
    pa_deg: np.ndarray | None = None
    spectral_coeffs: np.ndarray | None = None  # shape (N, N_terms) or None
    source_name: np.ndarray | None = None
    source_id: np.ndarray | None = None
    extra_columns: dict[str, np.ndarray] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate array consistency."""
        n = len(self.ra_rad)
        core_fields = {
            "ra_rad": self.ra_rad,
            "dec_rad": self.dec_rad,
            "flux": self.flux,
            "spectral_index": self.spectral_index,
            "stokes_q": self.stokes_q,
            "stokes_u": self.stokes_u,
            "stokes_v": self.stokes_v,
            "ref_freq": self.ref_freq,
        }
        for name, arr in core_fields.items():
            if len(arr) != n:
                raise ValueError(
                    f"PointSourceData: {name} has length {len(arr)}, "
                    f"expected {n} (must match ra_rad)."
                )

        # Morphology: all-or-none
        morph = (self.major_arcsec, self.minor_arcsec, self.pa_deg)
        morph_present = sum(1 for m in morph if m is not None)
        if morph_present not in (0, 3):
            raise ValueError(
                "PointSourceData: major_arcsec, minor_arcsec, pa_deg must be "
                "all set or all None."
            )

        # Optional arrays must match length
        for name, arr in [
            ("rotation_measure", self.rotation_measure),
            ("major_arcsec", self.major_arcsec),
            ("minor_arcsec", self.minor_arcsec),
            ("pa_deg", self.pa_deg),
        ]:
            if arr is not None and len(arr) != n:
                raise ValueError(
                    f"PointSourceData: {name} has length {len(arr)}, expected {n}."
                )
        for name in ("source_name", "source_id"):
            arr = getattr(self, name)
            if arr is None:
                continue
            arr_np = np.asarray(arr)
            if len(arr_np) != n:
                raise ValueError(
                    f"PointSourceData: {name} has length {len(arr_np)}, expected {n}."
                )
            object.__setattr__(self, name, arr_np)
        if self.spectral_coeffs is not None and self.spectral_coeffs.shape[0] != n:
            raise ValueError(
                f"PointSourceData: spectral_coeffs has {self.spectral_coeffs.shape[0]} "
                f"rows, expected {n}."
            )
        normalized_extra: dict[str, np.ndarray] = {}
        for name, arr in self.extra_columns.items():
            arr = np.asarray(arr)
            if arr.ndim != 1:
                raise ValueError(
                    f"PointSourceData: extra column {name!r} must be 1-D, "
                    f"got shape {arr.shape}."
                )
            if len(arr) != n:
                raise ValueError(
                    f"PointSourceData: extra column {name!r} has length {len(arr)}, "
                    f"expected {n}."
                )
            normalized_extra[name] = arr
        object.__setattr__(self, "extra_columns", normalized_extra)

    @property
    def n_sources(self) -> int:
        """Number of point sources."""
        return len(self.ra_rad)

    @property
    def is_empty(self) -> bool:
        """True if no sources are present."""
        return len(self.ra_rad) == 0

    @classmethod
    def empty(cls) -> PointSourceData:
        """Create an empty PointSourceData (zero-length arrays)."""
        z = np.zeros(0, dtype=np.float64)
        return cls(
            ra_rad=z.copy(),
            dec_rad=z.copy(),
            flux=z.copy(),
            spectral_index=z.copy(),
            stokes_q=z.copy(),
            stokes_u=z.copy(),
            stokes_v=z.copy(),
            ref_freq=z.copy(),
        )

    def masked(self, mask: np.ndarray) -> PointSourceData:
        """Return new instance with boolean mask applied to all arrays.

        Parameters
        ----------
        mask : np.ndarray
            Boolean mask of shape ``(n_sources,)``.

        Returns
        -------
        PointSourceData
        """
        return PointSourceData(
            ra_rad=self.ra_rad[mask],
            dec_rad=self.dec_rad[mask],
            flux=self.flux[mask],
            spectral_index=self.spectral_index[mask],
            stokes_q=self.stokes_q[mask],
            stokes_u=self.stokes_u[mask],
            stokes_v=self.stokes_v[mask],
            ref_freq=self.ref_freq[mask],
            rotation_measure=(
                self.rotation_measure[mask]
                if self.rotation_measure is not None
                else None
            ),
            major_arcsec=(
                self.major_arcsec[mask] if self.major_arcsec is not None else None
            ),
            minor_arcsec=(
                self.minor_arcsec[mask] if self.minor_arcsec is not None else None
            ),
            pa_deg=self.pa_deg[mask] if self.pa_deg is not None else None,
            spectral_coeffs=(
                self.spectral_coeffs[mask] if self.spectral_coeffs is not None else None
            ),
            source_name=(
                self.source_name[mask] if self.source_name is not None else None
            ),
            source_id=self.source_id[mask] if self.source_id is not None else None,
            extra_columns={name: arr[mask] for name, arr in self.extra_columns.items()},
        )

    def as_source_arrays(
        self, flux_limit: float = 0.0, reference_frequency: float = 0.0
    ) -> SourceArrays:
        """Convert to SourceArrays dict for visibility calculation.

        Parameters
        ----------
        flux_limit : float, default 0.0
            Minimum flux in Jy.
        reference_frequency : float, default 0.0
            Fallback reference frequency (used only if ref_freq is all-zero).

        Returns
        -------
        SourceArrays
        """
        if self.is_empty:
            return empty_source_arrays()

        if flux_limit > 0:
            mask = self.flux >= flux_limit
            n = int(mask.sum())
            if n == 0:
                return empty_source_arrays()
        else:
            mask = np.ones(self.n_sources, dtype=bool)
            n = self.n_sources

        ref_freq = self.ref_freq[mask]
        if reference_frequency and np.all(ref_freq == 0):
            ref_freq = np.full(n, reference_frequency, dtype=ref_freq.dtype)

        return {
            "ra_rad": self.ra_rad[mask],
            "dec_rad": self.dec_rad[mask],
            "flux": self.flux[mask],
            "spectral_index": self.spectral_index[mask],
            "stokes_q": self.stokes_q[mask],
            "stokes_u": self.stokes_u[mask],
            "stokes_v": self.stokes_v[mask],
            "ref_freq": ref_freq,
            "rotation_measure": (
                self.rotation_measure[mask]
                if self.rotation_measure is not None
                else None
            ),
            "major_arcsec": (
                self.major_arcsec[mask] if self.major_arcsec is not None else None
            ),
            "minor_arcsec": (
                self.minor_arcsec[mask] if self.minor_arcsec is not None else None
            ),
            "pa_deg": self.pa_deg[mask] if self.pa_deg is not None else None,
            "spectral_coeffs": (
                self.spectral_coeffs[mask] if self.spectral_coeffs is not None else None
            ),
        }

    # Tuple of all per-source 1-D array field names (for iteration).
    _CORE_FIELDS: tuple[str, ...] = (
        "ra_rad",
        "dec_rad",
        "flux",
        "spectral_index",
        "stokes_q",
        "stokes_u",
        "stokes_v",
        "ref_freq",
    )

    _OPTIONAL_FIELDS: tuple[str, ...] = (
        "rotation_measure",
        "major_arcsec",
        "minor_arcsec",
        "pa_deg",
    )

    _METADATA_FIELDS: tuple[str, ...] = ("source_name", "source_id")


# =============================================================================
# HealpixData
# =============================================================================


@dataclass(frozen=True)
class HealpixData:
    """Multi-frequency HEALPix brightness temperature maps.

    Dense maps have shape ``(n_freq, npix)`` where
    ``npix = hp.nside2npix(nside)``.  Sparse maps have shape
    ``(n_freq, n_stored_pix)`` with ``hpx_inds`` giving the full-sky
    HEALPix indices for each stored pixel.  The ``frequencies`` array
    provides the frequency axis in Hz.
    """

    maps: np.ndarray  # Stokes I, shape (n_freq, npix), in Kelvin
    nside: int
    frequencies: np.ndarray  # shape (n_freq,), in Hz
    coordinate_frame: str = "icrs"
    hpx_inds: np.ndarray | None = None

    q_maps: np.ndarray | None = None
    u_maps: np.ndarray | None = None
    v_maps: np.ndarray | None = None

    i_unit: str = "K"
    q_unit: str = "K"
    u_unit: str = "K"
    v_unit: str = "K"
    i_brightness_conversion: str | None = None
    q_brightness_conversion: str = "rayleigh-jeans"
    u_brightness_conversion: str = "rayleigh-jeans"
    v_brightness_conversion: str = "rayleigh-jeans"

    def __post_init__(self) -> None:
        """Validate array shapes."""
        frame = str(self.coordinate_frame).lower()
        if frame not in {"icrs", "galactic"}:
            raise ValueError(
                "HealpixData.coordinate_frame must be 'icrs' or 'galactic', "
                f"got {self.coordinate_frame!r}."
            )
        object.__setattr__(self, "coordinate_frame", frame)

        expected_npix = hp.nside2npix(self.nside)
        if self.maps.ndim != 2:
            raise ValueError(
                f"HealpixData: maps must be 2-D (n_freq, npix), "
                f"got shape {self.maps.shape}"
            )
        n_freq = self.maps.shape[0]
        if len(self.frequencies) != n_freq:
            raise ValueError(
                f"HealpixData: frequencies has {len(self.frequencies)} entries "
                f"but maps has {n_freq} frequency channels."
            )

        if self.hpx_inds is not None:
            hpx_inds = np.asarray(self.hpx_inds)
            if hpx_inds.ndim != 1:
                raise ValueError(
                    f"HealpixData: hpx_inds must be 1-D, got shape {hpx_inds.shape}"
                )
            if len(hpx_inds) != self.maps.shape[1]:
                raise ValueError(
                    "HealpixData: hpx_inds length must match the number of "
                    f"stored pixels ({len(hpx_inds)} != {self.maps.shape[1]})."
                )
            if np.any(hpx_inds < 0) or np.any(hpx_inds >= expected_npix):
                raise ValueError(
                    f"HealpixData: hpx_inds must be in [0, {expected_npix}); "
                    f"got min={int(np.min(hpx_inds)) if len(hpx_inds) else 'n/a'}, "
                    f"max={int(np.max(hpx_inds)) if len(hpx_inds) else 'n/a'}."
                )
            if self.maps.shape[1] != len(hpx_inds):
                raise ValueError(
                    "HealpixData: maps has "
                    f"{self.maps.shape[1]} pixels per map, but hpx_inds has "
                    f"length {len(hpx_inds)}."
                )
            object.__setattr__(self, "hpx_inds", hpx_inds.astype(np.int64, copy=False))
        elif self.maps.shape[1] != expected_npix:
            raise ValueError(
                f"HealpixData: maps has {self.maps.shape[1]} pixels per map, "
                f"expected {expected_npix} for nside={self.nside}"
            )

        for name, arr in [
            ("q_maps", self.q_maps),
            ("u_maps", self.u_maps),
            ("v_maps", self.v_maps),
        ]:
            if arr is not None and arr.shape != self.maps.shape:
                raise ValueError(
                    f"HealpixData: {name} shape {arr.shape} does not match "
                    f"maps shape {self.maps.shape}"
                )

        for name, unit in [
            ("i_unit", self.i_unit),
            ("q_unit", self.q_unit),
            ("u_unit", self.u_unit),
            ("v_unit", self.v_unit),
        ]:
            if not unit:
                raise ValueError(f"HealpixData: {name} must be a non-empty string.")

    @property
    def n_frequencies(self) -> int:
        """Number of frequency channels."""
        return len(self.frequencies)

    @property
    def n_pixels(self) -> int:
        """Number of stored HEALPix pixels per map."""
        return self.maps.shape[1]

    @property
    def full_n_pixels(self) -> int:
        """Number of pixels in the full HEALPix grid for ``nside``."""
        return hp.nside2npix(self.nside)

    @property
    def pixel_solid_angle(self) -> float:
        """Solid angle per pixel in steradians."""
        return 4 * np.pi / self.full_n_pixels

    @property
    def is_sparse(self) -> bool:
        """True when the maps only store a subset of HEALPix pixels."""
        return self.hpx_inds is not None and len(self.hpx_inds) < self.full_n_pixels

    @property
    def pixel_indices(self) -> np.ndarray:
        """Return the HEALPix indices corresponding to stored pixels."""
        if self.hpx_inds is None:
            return np.arange(self.full_n_pixels, dtype=np.int64)
        return self.hpx_inds

    @property
    def has_polarization(self) -> bool:
        """True if any Stokes Q/U/V maps are populated."""
        return any(m is not None for m in (self.q_maps, self.u_maps, self.v_maps))

    def to_dense(self) -> HealpixData:
        """Return a dense copy with full-sky arrays."""
        if not self.is_sparse:
            return self

        dense_shape = (self.n_frequencies, self.full_n_pixels)
        dense_maps = np.zeros(dense_shape, dtype=self.maps.dtype)
        dense_maps[:, self.hpx_inds] = self.maps

        def _dense_copy(arr: np.ndarray | None) -> np.ndarray | None:
            if arr is None:
                return None
            dense_arr = np.zeros(dense_shape, dtype=arr.dtype)
            dense_arr[:, self.hpx_inds] = arr
            return dense_arr

        return HealpixData(
            maps=dense_maps,
            nside=self.nside,
            frequencies=self.frequencies,
            coordinate_frame=self.coordinate_frame,
            q_maps=_dense_copy(self.q_maps),
            u_maps=_dense_copy(self.u_maps),
            v_maps=_dense_copy(self.v_maps),
            i_unit=self.i_unit,
            q_unit=self.q_unit,
            u_unit=self.u_unit,
            v_unit=self.v_unit,
            i_brightness_conversion=self.i_brightness_conversion,
            q_brightness_conversion=self.q_brightness_conversion,
            u_brightness_conversion=self.u_brightness_conversion,
            v_brightness_conversion=self.v_brightness_conversion,
        )

    def masked_region(self, healpix_mask: np.ndarray) -> HealpixData:
        """Return new HealpixData masked to a sky region.

        Parameters
        ----------
        healpix_mask : np.ndarray
            Boolean mask of shape ``(npix,)`` — True for pixels to keep.

        Returns
        -------
        HealpixData
        """
        healpix_mask = np.asarray(healpix_mask, dtype=bool)
        if len(healpix_mask) != self.full_n_pixels:
            raise ValueError(
                "HealpixData.masked_region: mask length must match the full "
                f"HEALPix grid ({len(healpix_mask)} != {self.full_n_pixels})."
            )

        if self.is_sparse:
            keep = healpix_mask[self.hpx_inds]
            if np.all(keep):
                return self

            new_maps = self.maps[:, keep]
            new_q = self.q_maps[:, keep] if self.q_maps is not None else None
            new_u = self.u_maps[:, keep] if self.u_maps is not None else None
            new_v = self.v_maps[:, keep] if self.v_maps is not None else None
            new_inds = self.hpx_inds[keep]
            return HealpixData(
                maps=new_maps,
                nside=self.nside,
                frequencies=self.frequencies,
                coordinate_frame=self.coordinate_frame,
                hpx_inds=new_inds,
                q_maps=new_q,
                u_maps=new_u,
                v_maps=new_v,
                i_unit=self.i_unit,
                q_unit=self.q_unit,
                u_unit=self.u_unit,
                v_unit=self.v_unit,
                i_brightness_conversion=self.i_brightness_conversion,
                q_brightness_conversion=self.q_brightness_conversion,
                u_brightness_conversion=self.u_brightness_conversion,
                v_brightness_conversion=self.v_brightness_conversion,
            )

        if np.all(healpix_mask):
            return self

        inv_mask = ~healpix_mask
        new_maps = self.maps.copy()
        new_maps[:, inv_mask] = 0.0

        new_q = None
        new_u = None
        new_v = None
        if self.q_maps is not None:
            new_q = self.q_maps.copy()
            new_q[:, inv_mask] = 0.0
        if self.u_maps is not None:
            new_u = self.u_maps.copy()
            new_u[:, inv_mask] = 0.0
        if self.v_maps is not None:
            new_v = self.v_maps.copy()
            new_v[:, inv_mask] = 0.0

        return HealpixData(
            maps=new_maps,
            nside=self.nside,
            frequencies=self.frequencies,
            coordinate_frame=self.coordinate_frame,
            q_maps=new_q,
            u_maps=new_u,
            v_maps=new_v,
            i_unit=self.i_unit,
            q_unit=self.q_unit,
            u_unit=self.u_unit,
            v_unit=self.v_unit,
            i_brightness_conversion=self.i_brightness_conversion,
            q_brightness_conversion=self.q_brightness_conversion,
            u_brightness_conversion=self.u_brightness_conversion,
            v_brightness_conversion=self.v_brightness_conversion,
        )
