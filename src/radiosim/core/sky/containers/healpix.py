"""HealpixData — multi-frequency HEALPix brightness-temperature cube.

Supports dense and sparse storage (sparse drives partial-sky inputs
through the pipeline without densifying), Stokes Q/U/V maps, per-channel
unit and brightness-conversion metadata, and a ``replace()`` helper that
re-runs every pydantic validator (use it instead of
``dataclasses.replace`` to preserve invariants).
"""

from __future__ import annotations

from typing import Any, cast

import numpy as np
from pydantic import field_validator, model_validator
from pydantic.dataclasses import dataclass

from ..support.healpy import lazy_healpy as hp
from ._shared import (
    _FROZEN_NDARRAY_CONFIG,
    _arrays_equal,
    _freeze,
    _require_floating_array,
    validate_frequency_axis,
)
from .constants import BrightnessConversion
from .point import TangentPolarizationFrame
from .polarization_materialization import PolarizationMaterializationEvidence


@dataclass(frozen=True, eq=False, config=_FROZEN_NDARRAY_CONFIG)
class HealpixData:
    """Multi-frequency HEALPix brightness temperature maps.

    Dense maps have shape ``(n_freq, npix)`` where
    ``npix = hp.nside2npix(nside)``.  Sparse maps have shape
    ``(n_freq, n_stored_pix)`` with ``hpx_inds`` giving the full-sky
    HEALPix indices for each stored pixel.  The ``frequencies`` array
    provides the frequency axis in Hz.

    Pixel ordering (RING / NEST)
    ----------------------------
    ``ordering`` may be ``'ring'`` (default) or ``'nest'``.  NEST support is
    complete across the sky stack: ang2pix/pix2ang, ud_grade (regrid),
    query_disc, get_all_neighbours, region filtering, point↔HEALPix
    conversion, combine point-binning, and subtract_bright_sources all thread
    ``nest=`` / ``order_in`` / ``order_out`` from this field.  Diffuse loaders
    (GSM, PySM, etc.) remain ring-native inputs and always emit
    ``ordering='ring'``; FITS and programmatic builders accept ``ordering=``.

    Frequency-axis dtype policy
    ---------------------------
    ``frequencies`` is **always stored as float64**, independent of the
    flux/storage precision (which governs the brightness ``maps``). This
    matches :class:`~.point.PointSpectrum.frequencies` so a HEALPix↔point
    round-trip never changes the frequency-axis dtype. The policy is enforced
    by :func:`._shared.validate_frequency_axis` (a ``mode="before"`` field
    validator), so even a flux-dtype cast applied upstream is normalized back
    to float64 here.

    Brightness-map dtype policy
    ---------------------------
    ``maps`` and optional ``q_maps`` / ``u_maps`` / ``v_maps`` must use a
    floating dtype (``float32`` or ``float64``). Integer, complex, and object
    arrays are rejected at construction via
    :func:`._shared._require_floating_array`, matching the core point-source
    column contract in :class:`~.point.PointSourceData`.

    Attached evidence requires the caller to exclude mutation or rebinding
    through every payload, frequency-axis and pixel-ID alias for the entire
    construction, replacement or validating operation: old-record validation,
    normalization, all scans/hashes and publication. Frozen fields, read-only
    flags and ``model.replace()`` may share backing storage; they establish
    neither exclusive ownership nor a coherent concurrent snapshot. Detection
    of later sequential staleness is a separate guarantee.
    """

    maps: np.ndarray  # Stokes I, shape (n_freq, npix), in Kelvin
    nside: int
    frequencies: np.ndarray  # shape (n_freq,), in Hz, always float64
    coordinate_frame: str = "icrs"
    # HEALPix pixel ordering. NEST is fully supported: all sky ops thread
    # nest= / order_in/out through ang2pix, regrid, combine, subtract, and
    # point↔HEALPix conversion. Diffuse loaders remain ring-native inputs.
    ordering: str = "ring"
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

    #: Canonical frame joined to the attached evidence; no implicit declaration.
    tangent_polarization_frame: TangentPolarizationFrame | None = None
    #: Actual stored-value evidence, validated after constructor normalization.
    polarization_materialization: PolarizationMaterializationEvidence | None = None

    @field_validator("tangent_polarization_frame", mode="plain")
    @classmethod
    def _preserve_tangent_frame(cls, value: object) -> TangentPolarizationFrame | None:
        if value is not None and type(value) is not TangentPolarizationFrame:
            raise ValueError("native tangent frame requires the typed frame")
        return value

    @field_validator("polarization_materialization", mode="plain")
    @classmethod
    def _preserve_materialization(
        cls, value: object
    ) -> PolarizationMaterializationEvidence | None:
        if value is not None and type(value) is not PolarizationMaterializationEvidence:
            raise ValueError("native materialization requires typed evidence")
        return value

    def validate_polarization_materialization(
        self, *, brightness_conversion: BrightnessConversion | None = None
    ) -> None:
        """Validate attached canonical identity against actual stored values.

        The caller must exclude mutation/rebinding through every payload,
        frequency-axis and pixel-ID alias for the full validation interval,
        including all scans and hashes. The same obligation covers implicit
        constructor/replacement validation through old-record checks,
        normalization and publication. Read-only flags and model replacement
        may share storage; no exclusive or coherent concurrent snapshot is
        established. Later sequential stale checks do not supply this exclusion.

        Parameters
        ----------
        brightness_conversion : BrightnessConversion, optional
            Enclosing model context, when known; otherwise the retained context.

        Raises
        ------
        ValueError
            If evidence is stale, its context differs, or a frame lacks evidence.
            Unbound owners without a frame remain unbound.
        """
        evidence = self.polarization_materialization
        if evidence is None:
            if self.tangent_polarization_frame is not None:
                raise ValueError(
                    "native tangent frame requires materialization evidence"
                )
            return
        if type(cast(object, evidence)) is not PolarizationMaterializationEvidence:
            raise ValueError("native materialization requires typed evidence")
        from ._polarization_materialization import require_native_identity

        require_native_identity(
            self,
            brightness_conversion=(
                evidence.brightness_conversion
                if brightness_conversion is None
                else brightness_conversion
            ),
            source_profile="radiosim_ne_iau_v1",
            tangent_frame=self.tangent_polarization_frame,
            expected=evidence,
        )

    @field_validator("nside", mode="before")
    @classmethod
    def _validate_nside(cls, value: object) -> int:
        n = int(value)  # type: ignore[arg-type]
        # nest=True enforces power-of-two, the standard HEALPix constraint and a
        # hard requirement now that NESTED ordering is supported.
        if not hp.isnsideok(n, nest=True):
            raise ValueError(
                f"HealpixData.nside must be a valid HEALPix NSIDE (power of 2), "
                f"got {value!r}."
            )
        return n

    @field_validator("frequencies", mode="before")
    @classmethod
    def _validate_frequencies(cls, value: object) -> np.ndarray:
        # Finite + positive, but not strictly-ascending: HEALPix cubes can be
        # assembled from out-of-order multi-file inputs and the channel order is
        # carried verbatim alongside the maps.
        return validate_frequency_axis(
            value, label="HealpixData.frequencies", ascending=False
        )

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
        hpx_inds = hpx_inds.astype(np.int64, copy=False)
        # Reject duplicates: indices align positionally with the columns of
        # ``maps``, so a duplicate would make ``to_dense()`` last-write-wins and
        # silently drop a stored pixel's flux. (Do NOT np.unique() — that would
        # desync indices from maps; an error is the correct contract.)
        if hpx_inds.size and np.unique(hpx_inds).size != hpx_inds.size:
            raise ValueError(
                "HealpixData: hpx_inds must be unique (each stored pixel maps to "
                "a distinct full-sky HEALPix index); got duplicates."
            )
        return hpx_inds

    #: Brightness units accepted on input. ``"Jy/sr"`` maps are converted to
    #: Kelvin at construction (see :meth:`_normalize_units_to_kelvin`) so that
    #: stored maps are always brightness temperature — the unit field is
    #: load-bearing: it drives that conversion rather than being ignored.
    ALLOWED_UNITS: tuple[str, ...] = ("K", "Jy/sr")

    @model_validator(mode="before")
    @classmethod
    def _normalize_units_to_kelvin(cls, values: object) -> object:
        """Convert any ``Jy/sr`` Stokes map to Kelvin before construction.

        Runs before the field validators, so by the time ``_validate_unit``
        and the shape checks see the data, every map is in Kelvin and every
        unit is ``"K"``.  A map declared ``Jy/sr`` is therefore *handled*
        (converted via the Rayleigh-Jeans law at each channel frequency)
        rather than silently mis-read as brightness temperature.
        """
        if isinstance(values, dict):
            kwargs = values
        elif hasattr(values, "kwargs") and isinstance(values.kwargs, dict):
            kwargs = values.kwargs
        else:
            return values

        freqs = kwargs.get("frequencies")
        if freqs is None:
            return values
        freqs = np.asarray(freqs, dtype=np.float64)

        from .constants import flux_density_to_brightness_temp

        def _convert(map_key: str, unit_key: str) -> None:
            arr = kwargs.get(map_key)
            unit = kwargs.get(unit_key, "K")
            if arr is None or unit is None or str(unit) != "Jy/sr":
                return
            arr = np.asarray(arr)
            if arr.ndim != 2 or arr.shape[0] != freqs.size:
                return  # shape validator will report the mismatch
            out = np.empty(arr.shape, dtype=arr.dtype)
            for i in range(arr.shape[0]):
                # solid_angle=1 sr → surface-brightness (Jy/sr) to T_b.
                out[i] = flux_density_to_brightness_temp(
                    arr[i].astype(np.float64),
                    float(freqs[i]),
                    1.0,
                    method="rayleigh-jeans",
                )
            kwargs[map_key] = out
            kwargs[unit_key] = "K"

        _convert("maps", "i_unit")
        _convert("q_maps", "q_unit")
        _convert("u_maps", "u_unit")
        _convert("v_maps", "v_unit")
        return values

    @field_validator("i_unit", "q_unit", "u_unit", "v_unit", mode="before")
    @classmethod
    def _validate_unit(cls, value: object) -> str:
        unit = str(value) if value is not None else ""
        if not unit:
            raise ValueError("HealpixData: unit must be a non-empty string.")
        if unit not in cls.ALLOWED_UNITS:
            raise ValueError(
                f"HealpixData: unit must be one of {cls.ALLOWED_UNITS}, got "
                f"{unit!r}. (Jy/sr maps are converted to Kelvin at construction.)"
            )
        return unit

    @model_validator(mode="after")
    def _validate_shapes(self) -> HealpixData:
        _require_floating_array(self.maps, label="HealpixData.maps")
        for name in ("q_maps", "u_maps", "v_maps"):
            _require_floating_array(getattr(self, name), label=f"HealpixData.{name}")
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

        # Lock every stored buffer read-only — enforces the copy-on-write
        # contract (see _shared._freeze).
        for arr in (
            self.maps,
            self.frequencies,
            self.hpx_inds,
            self.q_maps,
            self.u_maps,
            self.v_maps,
        ):
            _freeze(arr)
        self.validate_polarization_materialization()
        return self

    @property
    def is_nested(self) -> bool:
        """True when pixels are stored in NESTED ordering (else RING)."""
        return self.ordering == "nest"

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
        from ..support.healpix_geometry import pixel_solid_angle

        return pixel_solid_angle(self.nside)

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

        theta, phi = hp.pix2ang(self.nside, self.pixel_indices, nest=self.is_nested)
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
        self.validate_polarization_materialization()
        if not self.is_sparse:
            return self
        raise ValueError(
            f"{operation} requires a dense HEALPix cube; the input has "
            f"{self.n_pixels}/{self.full_n_pixels} pixels stored. "
            "Call sky.replace(healpix=sky.healpix.to_dense()) first to opt "
            "in to densification."
        )

    def to_dense(self, fill: float = 0.0) -> HealpixData:
        """Return a dense copy with full-sky arrays.

        Pixels not present in the sparse ``hpx_inds`` are set to ``fill``
        (cast to each Stokes map's dtype). The same ``fill`` is applied to
        every Stokes map (I and any Q/U/V).

        Partial-sky fill semantics
        --------------------------
        The default ``fill=0.0`` is **indistinguishable from a measured
        zero**, so monopole / power-spectrum statistics computed over a
        densified partial-sky map include the un-observed region as zero —
        mask or weight by the original footprint when that matters.

        Pass ``fill=np.nan`` to make the un-observed region explicit: NaN
        pixels are then distinguishable from measured zeros and propagate
        through ``np.nanmean`` / ``np.nansum`` style reductions instead of
        silently biasing them toward zero. (NaN requires a floating-point map
        dtype, which brightness-temperature maps always are.)

        A dense input has no un-observed pixels, so ``fill`` is irrelevant and
        ``self`` is returned unchanged.
        """
        self.validate_polarization_materialization()
        if not self.is_sparse:
            return self

        dense_shape = (self.n_frequencies, self.full_n_pixels)

        def _dense_copy(arr: np.ndarray | None) -> np.ndarray | None:
            if arr is None:
                return None
            dense_arr = np.full(dense_shape, fill, dtype=arr.dtype)
            dense_arr[:, self.hpx_inds] = arr
            return dense_arr

        return self.replace(
            maps=_dense_copy(self.maps),
            hpx_inds=None,
            q_maps=_dense_copy(self.q_maps),
            u_maps=_dense_copy(self.u_maps),
            v_maps=_dense_copy(self.v_maps),
        )

    def reordered(self, target_ordering: str) -> HealpixData:
        """Return a copy with maps converted to ``target_ordering``.

        Uses ``healpy.reorder`` to convert between RING and NESTED pixel
        ordering, updating every Stokes map and the ``ordering`` field.
        Requires a dense cube (reordering only makes sense full-sky); call
        :meth:`to_dense` first for sparse inputs.  Returns ``self`` unchanged
        when already in the requested ordering.
        """
        self.validate_polarization_materialization()
        target = str(target_ordering).lower()
        if target not in {"ring", "nest"}:
            raise ValueError(
                f"target_ordering must be 'ring' or 'nest', got {target_ordering!r}."
            )
        if target == self.ordering:
            return self
        self.require_dense("reordered")
        r2n = target == "nest"

        def _re(arr: np.ndarray | None) -> np.ndarray | None:
            if arr is None:
                return None
            return np.stack(
                [hp.reorder(row, r2n=r2n, n2r=not r2n) for row in arr], axis=0
            )

        return self.replace(
            maps=_re(self.maps),
            q_maps=_re(self.q_maps),
            u_maps=_re(self.u_maps),
            v_maps=_re(self.v_maps),
            ordering=target,
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
        self.validate_polarization_materialization()
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
        self.validate_polarization_materialization()
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
        "tangent_polarization_frame",
        "polarization_materialization",
    )

    def replace(self, **changes: Any) -> HealpixData:
        """Return a new ``HealpixData`` with the given fields replaced.

        Always use this instead of ``dataclasses.replace`` — direct calls
        bypass the pydantic field validators that enforce shape, ordering,
        coordinate-frame, and ``hpx_inds`` invariants.

        Unknown field names raise :class:`TypeError`.

        With attached evidence, the caller excludes mutation/rebinding through
        all payload/axis/ID aliases from old validation through reconstruction,
        scans/hashes and publication. Read-only flags do not make shared storage
        exclusive or provide a coherent concurrent snapshot.
        """
        unknown = set(changes) - set(self._REPLACE_FIELDS)
        if unknown:
            raise TypeError(
                f"HealpixData.replace() received unsupported fields: {sorted(unknown)}"
            )
        self.validate_polarization_materialization()
        if (
            self.polarization_materialization is not None
            and changes.get(
                "polarization_materialization", self.polarization_materialization
            )
            is not self.polarization_materialization
        ):
            raise ValueError("replace cannot drop or reissue native materialization")
        data = {name: getattr(self, name) for name in self._REPLACE_FIELDS}
        data.update(changes)
        return HealpixData(**data)

    __hash__ = None  # type: ignore[assignment]

    _ARRAY_FIELDS: tuple[str, ...] = (
        "maps",
        "frequencies",
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
        "tangent_polarization_frame",
        "polarization_materialization",
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
