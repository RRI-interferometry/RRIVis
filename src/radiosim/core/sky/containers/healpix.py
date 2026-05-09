"""HealpixData — multi-frequency HEALPix brightness-temperature cube.

Supports dense and sparse storage (sparse drives partial-sky inputs
through the pipeline without densifying), Stokes Q/U/V maps, per-channel
unit and brightness-conversion metadata, and a ``replace()`` helper that
re-runs every pydantic validator (use it instead of
``dataclasses.replace`` to preserve invariants).
"""

from __future__ import annotations

from typing import Any

import healpy as hp
import numpy as np
from pydantic import field_validator, model_validator
from pydantic.dataclasses import dataclass

from ._shared import _FROZEN_NDARRAY_CONFIG, _arrays_equal


@dataclass(frozen=True, eq=False, config=_FROZEN_NDARRAY_CONFIG)
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
    channel_widths_hz: np.ndarray | None = None
    """Per-channel bandwidth in Hz, shape ``(n_freq,)``, strictly positive.

    Encodes the bandwidth that each ``frequencies`` sample integrates over.
    ``None`` means the source data does not carry channel-width information
    (do **not** synthesise this from frequency spacing — adjacent samples
    can be far apart while individual channels remain narrow).  Downstream
    visibility code may use this to integrate steep-spectrum sources over
    the channel rather than evaluating at the centre frequency.
    """
    coordinate_frame: str = "icrs"
    ordering: str = "ring"  # "ring" or "nest" — HEALPix pixel ordering scheme
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

    @field_validator("coordinate_frame", mode="before")
    @classmethod
    def _validate_coordinate_frame(cls, value: object) -> str:
        frame = str(value).lower()
        if frame not in {"icrs", "galactic"}:
            raise ValueError(
                "HealpixData.coordinate_frame must be 'icrs' or 'galactic', "
                f"got {value!r}."
            )
        return frame

    @field_validator("ordering", mode="before")
    @classmethod
    def _validate_ordering(cls, value: object) -> str:
        ordering = str(value).lower()
        if ordering not in {"ring", "nest"}:
            raise ValueError(
                f"HealpixData.ordering must be 'ring' or 'nest', got {value!r}."
            )
        return ordering

    @field_validator("channel_widths_hz", mode="before")
    @classmethod
    def _validate_channel_widths(cls, value: object) -> np.ndarray | None:
        if value is None:
            return None
        widths = np.asarray(value, dtype=np.float64)
        if widths.ndim != 1:
            raise ValueError(
                f"HealpixData: channel_widths_hz must be 1-D, got shape {widths.shape}."
            )
        if not np.all(np.isfinite(widths)) or np.any(widths <= 0):
            raise ValueError(
                "HealpixData: channel_widths_hz must be finite and strictly positive."
            )
        return widths

    @field_validator("hpx_inds", mode="before")
    @classmethod
    def _coerce_hpx_inds(cls, value: object) -> np.ndarray | None:
        if value is None:
            return None
        hpx_inds = np.asarray(value)
        if hpx_inds.ndim != 1:
            raise ValueError(
                f"HealpixData: hpx_inds must be 1-D, got shape {hpx_inds.shape}"
            )
        return hpx_inds.astype(np.int64, copy=False)

    @field_validator("i_unit", "q_unit", "u_unit", "v_unit", mode="before")
    @classmethod
    def _validate_unit(cls, value: object) -> str:
        unit = str(value) if value is not None else ""
        if not unit:
            raise ValueError("HealpixData: unit must be a non-empty string.")
        return unit

    @model_validator(mode="after")
    def _validate_shapes(self) -> HealpixData:
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

        if (
            self.channel_widths_hz is not None
            and self.channel_widths_hz.shape[0] != n_freq
        ):
            raise ValueError(
                "HealpixData: channel_widths_hz must have the same length as "
                f"frequencies ({n_freq}), got shape {self.channel_widths_hz.shape}."
            )

        expected_npix = hp.nside2npix(self.nside)
        if self.hpx_inds is not None:
            if len(self.hpx_inds) != self.maps.shape[1]:
                raise ValueError(
                    "HealpixData: hpx_inds length must match the number of "
                    f"stored pixels ({len(self.hpx_inds)} != "
                    f"{self.maps.shape[1]})."
                )
            if self.hpx_inds.size and (
                np.any(self.hpx_inds < 0) or np.any(self.hpx_inds >= expected_npix)
            ):
                raise ValueError(
                    f"HealpixData: hpx_inds must be in [0, {expected_npix}); "
                    f"got min={int(np.min(self.hpx_inds))}, "
                    f"max={int(np.max(self.hpx_inds))}."
                )
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
        return self

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

    @property
    def pixel_coords(self) -> Any:
        """:class:`astropy.coordinates.SkyCoord` of the stored pixel centres.

        Returned in the stored ``coordinate_frame`` (ICRS or Galactic).
        For dense maps this is the full HEALPix grid; for sparse maps it
        is only the retained pixels.
        """
        from astropy.coordinates import SkyCoord

        theta, phi = hp.pix2ang(self.nside, self.pixel_indices)
        lat_rad = np.pi / 2 - theta
        if self.coordinate_frame == "galactic":
            return SkyCoord(l=phi, b=lat_rad, unit="rad", frame="galactic")
        return SkyCoord(ra=phi, dec=lat_rad, unit="rad", frame="icrs")

    def resolve_frequency_index(self, frequency: float) -> int:
        """Return the index of the channel nearest to ``frequency`` in Hz.

        Logs a warning when the request is far enough off-grid that the
        nearest channel is unlikely to be what the caller intended (uses
        the same threshold as user-facing entry points).
        """
        # Lazy import: ``_data.py`` is a leaf module; ``spectral`` is loaded
        # on demand to keep the dependency graph tight.
        from .spectral import nearest_channel_index_with_warning

        return nearest_channel_index_with_warning(
            self.frequencies, frequency, label="resolve_frequency_index"
        )

    def get_map_at_frequency(self, frequency: float) -> np.ndarray:
        """Return the Stokes-I map at ``frequency`` (nearest channel)."""
        return self.maps[self.resolve_frequency_index(frequency)]

    def get_multifreq_maps(self) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(maps, frequencies)`` for the Stokes-I cube.

        Read :attr:`nside` directly when needed.
        """
        return self.maps, self.frequencies

    def get_stokes_maps_at_frequency(
        self, frequency: float
    ) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None, np.ndarray | None]:
        """Return ``(I, Q, U, V)`` maps at the channel nearest ``frequency``."""
        idx = self.resolve_frequency_index(frequency)
        i_map = self.maps[idx]
        q_map = self.q_maps[idx] if self.q_maps is not None else None
        u_map = self.u_maps[idx] if self.u_maps is not None else None
        v_map = self.v_maps[idx] if self.v_maps is not None else None
        return i_map, q_map, u_map, v_map

    def get_multifreq_stokes_maps(
        self,
    ) -> tuple[
        np.ndarray,
        np.ndarray | None,
        np.ndarray | None,
        np.ndarray | None,
        np.ndarray,
    ]:
        """Return ``(I_maps, Q_maps, U_maps, V_maps, frequencies)``.

        Read :attr:`nside` directly when needed.
        """
        return self.maps, self.q_maps, self.u_maps, self.v_maps, self.frequencies

    def iter_frequency_maps(self):
        """Yield ``(freq_hz, I_map, Q_map, U_map, V_map)`` per channel.

        Useful for memory-efficient processing when the full
        ``(n_freq, npix)`` cube is not needed at once.
        """
        for i, freq in enumerate(self.frequencies):
            s_q = self.q_maps[i] if self.q_maps is not None else None
            s_u = self.u_maps[i] if self.u_maps is not None else None
            s_v = self.v_maps[i] if self.v_maps is not None else None
            yield float(freq), self.maps[i], s_q, s_u, s_v

    def require_dense(self, operation: str) -> HealpixData:
        """Return ``self`` if dense, otherwise raise ``ValueError``.

        Sparse ``HealpixData`` is the canonical form for partial-sky inputs
        and propagates losslessly through load → combine → simulate.
        Operations that genuinely need a full-sky array (plotting, harmonic
        regridding, lightcurves, observability projections, bright-source
        subtraction) call this helper to surface a single, predictable
        error message rather than silently densifying — densification can
        balloon memory by orders of magnitude and should be the user's
        explicit choice.
        """
        if not self.is_sparse:
            return self
        raise ValueError(
            f"{operation} requires a dense HEALPix cube; the input has "
            f"{self.n_pixels}/{self.full_n_pixels} pixels stored. "
            "Call sky.replace(healpix=sky.healpix.to_dense()) first to opt "
            "in to densification."
        )

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

        return self.replace(
            maps=dense_maps,
            hpx_inds=None,
            q_maps=_dense_copy(self.q_maps),
            u_maps=_dense_copy(self.u_maps),
            v_maps=_dense_copy(self.v_maps),
        )

    def _validate_full_grid_mask(self, healpix_mask: np.ndarray) -> np.ndarray:
        """Coerce and shape-check a boolean mask defined on the full HEALPix grid."""
        healpix_mask = np.asarray(healpix_mask, dtype=bool)
        if len(healpix_mask) != self.full_n_pixels:
            raise ValueError(
                "HealpixData mask length must match the full HEALPix grid "
                f"({len(healpix_mask)} != {self.full_n_pixels})."
            )
        return healpix_mask

    def cropped_to_mask(self, healpix_mask: np.ndarray) -> HealpixData:
        """Return a new sparse :class:`HealpixData` keeping only mask=True pixels.

        Works for both sparse and dense input. The result is always sparse:
        ``hpx_inds`` is set to the indices of mask=True pixels intersected
        with currently-stored pixels, and the ``maps`` arrays are sliced
        accordingly.

        For a fully-dense input where every mask=True pixel is stored, the
        result drops to ``maps.shape == (n_freq, mask.sum())`` with
        matching ``hpx_inds``.
        """
        healpix_mask = self._validate_full_grid_mask(healpix_mask)

        if self.is_sparse:
            keep = healpix_mask[self.hpx_inds]
            if np.all(keep):
                return self
            new_inds = self.hpx_inds[keep]
        else:
            keep = healpix_mask
            new_inds = np.flatnonzero(keep).astype(np.int64, copy=False)

        new_maps = self.maps[:, keep]
        new_q = self.q_maps[:, keep] if self.q_maps is not None else None
        new_u = self.u_maps[:, keep] if self.u_maps is not None else None
        new_v = self.v_maps[:, keep] if self.v_maps is not None else None

        return self.replace(
            maps=new_maps,
            hpx_inds=new_inds,
            q_maps=new_q,
            u_maps=new_u,
            v_maps=new_v,
        )

    def zero_outside_mask(self, healpix_mask: np.ndarray) -> HealpixData:
        """Return a new dense :class:`HealpixData` with mask=False pixels zeroed.

        Requires a dense input (call :meth:`to_dense` first if sparse —
        zero-filling a sparse cube would force materialization of every
        un-stored pixel and balloon memory).  Shape is preserved.
        """
        self.require_dense("zero_outside_mask")
        healpix_mask = self._validate_full_grid_mask(healpix_mask)

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

        return self.replace(
            maps=new_maps,
            q_maps=new_q,
            u_maps=new_u,
            v_maps=new_v,
        )

    _REPLACE_FIELDS: tuple[str, ...] = (
        "maps",
        "nside",
        "frequencies",
        "channel_widths_hz",
        "coordinate_frame",
        "ordering",
        "hpx_inds",
        "q_maps",
        "u_maps",
        "v_maps",
        "i_unit",
        "q_unit",
        "u_unit",
        "v_unit",
        "i_brightness_conversion",
        "q_brightness_conversion",
        "u_brightness_conversion",
        "v_brightness_conversion",
    )

    def replace(self, **changes: Any) -> HealpixData:
        """Return a new ``HealpixData`` with the given fields replaced.

        Always use this instead of ``dataclasses.replace`` — direct calls
        bypass the pydantic field validators that enforce shape, ordering,
        coordinate-frame, and ``hpx_inds`` invariants.

        Unknown field names raise :class:`TypeError`.
        """
        unknown = set(changes) - set(self._REPLACE_FIELDS)
        if unknown:
            raise TypeError(
                f"HealpixData.replace() received unsupported fields: {sorted(unknown)}"
            )
        data = {name: getattr(self, name) for name in self._REPLACE_FIELDS}
        data.update(changes)
        return HealpixData(**data)

    __hash__ = None  # type: ignore[assignment]

    _ARRAY_FIELDS: tuple[str, ...] = (
        "maps",
        "frequencies",
        "channel_widths_hz",
        "hpx_inds",
        "q_maps",
        "u_maps",
        "v_maps",
    )
    _SCALAR_FIELDS: tuple[str, ...] = (
        "nside",
        "coordinate_frame",
        "ordering",
        "i_unit",
        "q_unit",
        "u_unit",
        "v_unit",
        "i_brightness_conversion",
        "q_brightness_conversion",
        "u_brightness_conversion",
        "v_brightness_conversion",
    )

    def _compare(
        self, other: HealpixData, *, close: bool, rtol: float, atol: float
    ) -> bool:
        for name in self._SCALAR_FIELDS:
            if getattr(self, name) != getattr(other, name):
                return False
        for name in self._ARRAY_FIELDS:
            if not _arrays_equal(
                getattr(self, name),
                getattr(other, name),
                close=close,
                rtol=rtol,
                atol=atol,
            ):
                return False
        return True

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, HealpixData):
            return NotImplemented
        return self._compare(other, close=False, rtol=0.0, atol=0.0)

    def is_close(
        self, other: HealpixData, *, rtol: float = 1e-7, atol: float = 0.0
    ) -> bool:
        if not isinstance(other, HealpixData):
            return False
        return self._compare(other, close=True, rtol=rtol, atol=atol)
