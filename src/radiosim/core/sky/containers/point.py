"""Point-source columnar payload + per-channel spectrum table.

Provides :class:`PointSourceData` (Nx columns of RA/Dec/flux/Stokes/etc.)
and :class:`PointSpectrum` (lossless per-channel flux samples, used by
catalogues that supply tabulated spectra rather than spectral indices).

``SourceArrays`` is the flat-dict view consumed by the visibility code.

The optional per-source extension fields (Gaussian morphology, rotation
measure, source name/id, extra columns) live in dedicated frozen
sub-dataclasses (:class:`PointMorphology`, :class:`PointPolarization`,
:class:`PointMetadata`) so each block carries its own all-or-nothing
shape rules and so adding a future field — e.g. an RM uncertainty — is
local.
"""

from __future__ import annotations

from dataclasses import field
from typing import Any, TypedDict

import numpy as np
from pydantic import field_validator, model_validator
from pydantic.dataclasses import dataclass

from ._shared import (
    _FROZEN_NDARRAY_CONFIG,
    _arrays_equal,
    _freeze,
    _validate_mask,
    validate_frequency_axis,
)
from .constants import SpectralType

# =============================================================================
# SourceArrays TypedDict
# =============================================================================


class SourceArrays(TypedDict):
    """Return type for point-source array extraction.

    Keys match the interface consumed by ``visibility.py``. The flat
    layout is preserved on purpose so the visibility code does not need
    to traverse the nested PointSourceData sub-objects.
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
    per_channel_flux: np.ndarray | None
    per_channel_stokes_q: np.ndarray | None
    per_channel_stokes_u: np.ndarray | None
    per_channel_stokes_v: np.ndarray | None
    channel_frequencies: np.ndarray | None


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
        "per_channel_flux": None,
        "per_channel_stokes_q": None,
        "per_channel_stokes_u": None,
        "per_channel_stokes_v": None,
        "channel_frequencies": None,
    }


# =============================================================================
# PointSpectrum (per-channel flux table)
# =============================================================================


@dataclass(frozen=True, eq=False, config=_FROZEN_NDARRAY_CONFIG)
class PointSpectrum:
    """Lossless multi-frequency Stokes-flux samples for point sources.

    All map arrays have shape ``(n_freq, N)``. Q and U are paired (both or
    neither); V is independently optional. ``frequencies`` is 1-D, strictly
    ascending, finite, and positive. When attached to a
    :class:`PointSourceData`, ``N`` matches the source count and consumers
    use nearest-channel lookup at observation frequencies.

    Frequency-axis dtype policy
    ---------------------------
    ``frequencies`` is **always stored as float64**, independent of the flux
    precision (which governs the ``flux`` / ``stokes_*`` arrays). This is the
    same policy as :class:`~.healpix.HealpixData.frequencies`, so a
    HEALPix↔point round-trip leaves the frequency-axis dtype unchanged. The
    policy is enforced by :func:`._shared.validate_frequency_axis`.
    """

    flux: np.ndarray  # shape (n_freq, N), Stokes I in Jy
    frequencies: np.ndarray  # shape (n_freq,), Hz, ascending, always float64
    stokes_q: np.ndarray | None = None  # shape (n_freq, N)
    stokes_u: np.ndarray | None = None  # shape (n_freq, N)
    stokes_v: np.ndarray | None = None  # shape (n_freq, N)

    @field_validator("frequencies", mode="before")
    @classmethod
    def _validate_frequencies(cls, value: object) -> np.ndarray:
        return validate_frequency_axis(
            value, label="PointSpectrum.frequencies", ascending=True
        )

    @model_validator(mode="after")
    def _validate_shapes(self) -> PointSpectrum:
        if self.flux.ndim != 2 or self.flux.shape[0] != self.frequencies.size:
            raise ValueError(
                f"PointSpectrum.flux shape {self.flux.shape} does not match "
                f"frequencies (expected first axis = {self.frequencies.size})."
            )
        if (self.stokes_q is None) != (self.stokes_u is None):
            raise ValueError(
                "PointSpectrum.stokes_q and stokes_u must be set together."
            )
        for name in ("stokes_q", "stokes_u", "stokes_v"):
            arr = getattr(self, name)
            if arr is not None and arr.shape != self.flux.shape:
                raise ValueError(
                    f"PointSpectrum.{name} shape {arr.shape} does not match "
                    f"flux shape {self.flux.shape}."
                )
        for arr in (
            self.flux,
            self.frequencies,
            self.stokes_q,
            self.stokes_u,
            self.stokes_v,
        ):
            _freeze(arr)
        return self

    @property
    def n_frequencies(self) -> int:
        return int(self.frequencies.size)

    @property
    def n_sources(self) -> int:
        return int(self.flux.shape[1])

    def masked_sources(self, mask: np.ndarray) -> PointSpectrum:
        """Return a new PointSpectrum with a boolean mask applied along sources."""
        mask = _validate_mask(mask, self.n_sources, label="PointSpectrum mask")
        return PointSpectrum(
            flux=self.flux[:, mask],
            frequencies=self.frequencies,
            stokes_q=self.stokes_q[:, mask] if self.stokes_q is not None else None,
            stokes_u=self.stokes_u[:, mask] if self.stokes_u is not None else None,
            stokes_v=self.stokes_v[:, mask] if self.stokes_v is not None else None,
        )

    __hash__ = None  # type: ignore[assignment]

    def _compare(
        self, other: PointSpectrum, *, close: bool, rtol: float, atol: float
    ) -> bool:
        for name in ("flux", "frequencies", "stokes_q", "stokes_u", "stokes_v"):
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
        if not isinstance(other, PointSpectrum):
            return NotImplemented
        return self._compare(other, close=False, rtol=0.0, atol=0.0)

    def is_close(
        self, other: PointSpectrum, *, rtol: float = 1e-7, atol: float = 0.0
    ) -> bool:
        if not isinstance(other, PointSpectrum):
            return False
        return self._compare(other, close=True, rtol=rtol, atol=atol)


# =============================================================================
# Per-source extension sub-dataclasses
# =============================================================================


@dataclass(frozen=True, eq=False, config=_FROZEN_NDARRAY_CONFIG)
class PointMorphology:
    """Gaussian morphology per source (all three axes are mandatory).

    ``major_arcsec``, ``minor_arcsec`` are FWHMs along the major and
    minor axes; ``pa_deg`` is the position angle of the major axis east
    of north in degrees. The all-or-nothing rule (a source either has
    Gaussian morphology fully specified or is treated as a delta function)
    is enforced here so callers cannot construct a half-specified state.
    """

    major_arcsec: np.ndarray  # shape (N,), FWHM in arcsec
    minor_arcsec: np.ndarray  # shape (N,), FWHM in arcsec
    pa_deg: np.ndarray  # shape (N,), position angle in degrees

    @model_validator(mode="after")
    def _validate_lengths(self) -> PointMorphology:
        n = len(self.major_arcsec)
        for name, arr in (
            ("minor_arcsec", self.minor_arcsec),
            ("pa_deg", self.pa_deg),
        ):
            if len(arr) != n:
                raise ValueError(
                    f"PointMorphology: {name} has length {len(arr)}, "
                    f"expected {n} (must match major_arcsec)."
                )
        for arr in (self.major_arcsec, self.minor_arcsec, self.pa_deg):
            _freeze(arr)
        return self

    @property
    def n_sources(self) -> int:
        return int(len(self.major_arcsec))

    def masked(self, mask: np.ndarray) -> PointMorphology:
        mask = _validate_mask(mask, self.n_sources, label="PointMorphology mask")
        return PointMorphology(
            major_arcsec=self.major_arcsec[mask],
            minor_arcsec=self.minor_arcsec[mask],
            pa_deg=self.pa_deg[mask],
        )

    __hash__ = None  # type: ignore[assignment]

    def _compare(
        self, other: PointMorphology, *, close: bool, rtol: float, atol: float
    ) -> bool:
        for name in ("major_arcsec", "minor_arcsec", "pa_deg"):
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
        if not isinstance(other, PointMorphology):
            return NotImplemented
        return self._compare(other, close=False, rtol=0.0, atol=0.0)


@dataclass(frozen=True, eq=False, config=_FROZEN_NDARRAY_CONFIG)
class PointPolarization:
    """Polarisation extras per source.

    Currently only ``rotation_measure`` lives here. The block exists so
    future polarisation-specific fields (RM uncertainties, depolarisation
    parameters) can be added without disturbing the rest of
    :class:`PointSourceData`.
    """

    rotation_measure: np.ndarray  # shape (N,), rad/m^2

    @model_validator(mode="after")
    def _freeze_arrays(self) -> PointPolarization:
        _freeze(self.rotation_measure)
        return self

    @property
    def n_sources(self) -> int:
        return int(len(self.rotation_measure))

    def masked(self, mask: np.ndarray) -> PointPolarization:
        mask = _validate_mask(mask, self.n_sources, label="PointPolarization mask")
        return PointPolarization(rotation_measure=self.rotation_measure[mask])

    __hash__ = None  # type: ignore[assignment]

    def _compare(
        self, other: PointPolarization, *, close: bool, rtol: float, atol: float
    ) -> bool:
        return _arrays_equal(
            self.rotation_measure,
            other.rotation_measure,
            close=close,
            rtol=rtol,
            atol=atol,
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, PointPolarization):
            return NotImplemented
        return self._compare(other, close=False, rtol=0.0, atol=0.0)


@dataclass(frozen=True, eq=False, config=_FROZEN_NDARRAY_CONFIG)
class PointMetadata:
    """Free-form per-source metadata: source_name, source_id, extra_columns.

    Each field is independently optional (no all-or-nothing rule); if
    nothing is set, callers should pass ``metadata=None`` rather than an
    empty PointMetadata.
    """

    source_name: np.ndarray | None = None
    source_id: np.ndarray | None = None
    extra_columns: dict[str, np.ndarray] = field(default_factory=dict)

    @field_validator("source_name", "source_id", mode="before")
    @classmethod
    def _normalize_metadata_array(cls, value: object) -> np.ndarray | None:
        if value is None:
            return None
        return np.asarray(value)

    @field_validator("extra_columns", mode="before")
    @classmethod
    def _normalize_extra_columns(cls, value: object) -> dict[str, np.ndarray]:
        if value is None:
            return {}
        if not isinstance(value, dict):
            raise TypeError(
                "PointMetadata.extra_columns must be a dict, got "
                f"{type(value).__name__}."
            )
        normalized: dict[str, np.ndarray] = {}
        for name, arr in value.items():
            arr_np = np.asarray(arr)
            if arr_np.ndim != 1:
                raise ValueError(
                    f"PointMetadata: extra column {name!r} must be 1-D, "
                    f"got shape {arr_np.shape}."
                )
            normalized[name] = arr_np
        return normalized

    @model_validator(mode="after")
    def _validate_lengths(self) -> PointMetadata:
        """Every populated field must have the same per-source length."""
        observed_lengths: dict[str, int] = {}
        if self.source_name is not None:
            observed_lengths["source_name"] = len(self.source_name)
        if self.source_id is not None:
            observed_lengths["source_id"] = len(self.source_id)
        for name, arr in self.extra_columns.items():
            observed_lengths[f"extra_columns[{name!r}]"] = len(arr)
        if not observed_lengths:
            return self
        unique_lengths = set(observed_lengths.values())
        if len(unique_lengths) > 1:
            details = ", ".join(
                f"{name}={length}" for name, length in sorted(observed_lengths.items())
            )
            raise ValueError(
                "PointMetadata: populated fields disagree on per-source length: "
                f"{details}."
            )
        for arr in (self.source_name, self.source_id):
            _freeze(arr)
        for arr in self.extra_columns.values():
            _freeze(arr)
        return self

    def n_sources(self) -> int | None:
        """Return the per-source length implied by populated fields, or None.

        Returns ``None`` when nothing is populated — used by
        :class:`PointSourceData` to decide whether to skip the
        length-check.
        """
        for arr in (self.source_name, self.source_id):
            if arr is not None:
                return int(len(arr))
        for arr in self.extra_columns.values():
            return int(len(arr))
        return None

    def masked(self, mask: np.ndarray) -> PointMetadata:
        n = self.n_sources()
        if n is not None:
            mask = _validate_mask(mask, n, label="PointMetadata mask")
        return PointMetadata(
            source_name=self.source_name[mask]
            if self.source_name is not None
            else None,
            source_id=self.source_id[mask] if self.source_id is not None else None,
            extra_columns={name: arr[mask] for name, arr in self.extra_columns.items()},
        )

    @property
    def is_empty(self) -> bool:
        return (
            self.source_name is None
            and self.source_id is None
            and not self.extra_columns
        )

    __hash__ = None  # type: ignore[assignment]

    def _compare(
        self, other: PointMetadata, *, close: bool, rtol: float, atol: float
    ) -> bool:
        for name in ("source_name", "source_id"):
            if not _arrays_equal(
                getattr(self, name),
                getattr(other, name),
                close=close,
                rtol=rtol,
                atol=atol,
            ):
                return False
        if self.extra_columns.keys() != other.extra_columns.keys():
            return False
        for name in sorted(self.extra_columns):
            if not _arrays_equal(
                self.extra_columns[name],
                other.extra_columns[name],
                close=close,
                rtol=rtol,
                atol=atol,
            ):
                return False
        return True

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, PointMetadata):
            return NotImplemented
        return self._compare(other, close=False, rtol=0.0, atol=0.0)


# =============================================================================
# PointSourceData
# =============================================================================


_FLAT_MORPHOLOGY_FIELDS = ("major_arcsec", "minor_arcsec", "pa_deg")
_FLAT_POLARIZATION_FIELDS = ("rotation_measure",)
_FLAT_METADATA_FIELDS = ("source_name", "source_id", "extra_columns")


@dataclass(frozen=True, eq=False, config=_FROZEN_NDARRAY_CONFIG)
class PointSourceData:
    """Columnar arrays for point-source sky model.

    Core arrays (always populated, possibly zero-length) carry RA/Dec,
    Stokes I/Q/U/V at the reference frequency, the spectral index and
    reference-frequency value. Extension blocks
    (:class:`PointMorphology`, :class:`PointPolarization`,
    :class:`PointMetadata`) are independently optional and grouped so
    each block validates its own shape rules in isolation.

    The constructor accepts both nested objects and flat per-source kwargs
    (e.g. ``major_arcsec=…``); a pre-validator (:meth:`_pack_flat_kwargs`)
    packs flat kwargs into the matching sub-dataclass before the dataclass
    is built.  The flat path is the live column-oriented construction route
    used by :func:`create_from_arrays`,
    :func:`support.point_builder.point_source_data_from_mapping`, and
    ``combine/engine.py``; it is **not** a deprecated shim.
    """

    ra_rad: np.ndarray
    dec_rad: np.ndarray
    flux: np.ndarray
    spectral_index: np.ndarray
    stokes_q: np.ndarray
    stokes_u: np.ndarray
    stokes_v: np.ndarray
    ref_freq: np.ndarray

    spectral_coeffs: np.ndarray | None = None  # shape (N, N_terms) or None

    morphology: PointMorphology | None = None
    polarization: PointPolarization | None = None
    metadata: PointMetadata | None = None

    # Optional lossless multi-frequency Stokes-flux table. When populated,
    # consumers evaluate flux at an observation frequency via nearest-channel
    # lookup rather than spectral-index extrapolation.
    spectrum: PointSpectrum | None = None

    @model_validator(mode="before")
    @classmethod
    def _pack_flat_kwargs(cls, values: object) -> object:
        """Pack flat per-source kwargs into the nested sub-dataclasses.

        ``major_arcsec``/``minor_arcsec``/``pa_deg`` collapse into a
        :class:`PointMorphology`; ``rotation_measure`` into
        :class:`PointPolarization`; ``source_name``/``source_id``/
        ``extra_columns`` into :class:`PointMetadata`. Already-nested
        kwargs win over flat ones (passing both is a TypeError).

        This is a **live** construction path, not a deprecated shim: it is
        how :func:`create_from_arrays`,
        :func:`support.point_builder.point_source_data_from_mapping`, and
        ``combine/engine.py`` build a ``PointSourceData`` from their column
        dicts.  It runs on a copy of the incoming kwargs (pydantic passes a
        fresh ``ArgsKwargs``/dict per construction), so popping a flat key
        does not leak back to the caller's mapping.
        """
        if isinstance(values, dict):
            kwargs = values
        elif hasattr(values, "kwargs") and isinstance(values.kwargs, dict):
            kwargs = values.kwargs
        else:
            return values

        def _pop_flat(names: tuple[str, ...]) -> dict[str, Any]:
            popped: dict[str, Any] = {}
            for name in names:
                if name in kwargs:
                    popped[name] = kwargs.pop(name)
            return popped

        morph_flat = _pop_flat(_FLAT_MORPHOLOGY_FIELDS)
        if morph_flat:
            if kwargs.get("morphology") is not None:
                raise TypeError(
                    "PointSourceData: pass either 'morphology' or the flat "
                    "morphology kwargs (major_arcsec/minor_arcsec/pa_deg), "
                    "not both."
                )
            present = {k for k, v in morph_flat.items() if v is not None}
            if present and len(present) != 3:
                raise ValueError(
                    "PointSourceData: major_arcsec, minor_arcsec, pa_deg must "
                    "be all set or all None."
                )
            if present:
                kwargs["morphology"] = PointMorphology(
                    major_arcsec=morph_flat["major_arcsec"],
                    minor_arcsec=morph_flat["minor_arcsec"],
                    pa_deg=morph_flat["pa_deg"],
                )

        pol_flat = _pop_flat(_FLAT_POLARIZATION_FIELDS)
        if pol_flat:
            if kwargs.get("polarization") is not None:
                raise TypeError(
                    "PointSourceData: pass either 'polarization' or the flat "
                    "rotation_measure kwarg, not both."
                )
            rm = pol_flat["rotation_measure"]
            if rm is not None:
                kwargs["polarization"] = PointPolarization(rotation_measure=rm)

        meta_flat = _pop_flat(_FLAT_METADATA_FIELDS)
        if meta_flat:
            if kwargs.get("metadata") is not None:
                raise TypeError(
                    "PointSourceData: pass either 'metadata' or the flat "
                    "metadata kwargs (source_name/source_id/extra_columns), "
                    "not both."
                )
            non_empty = (
                meta_flat.get("source_name") is not None
                or meta_flat.get("source_id") is not None
                or meta_flat.get("extra_columns")
            )
            if non_empty:
                kwargs["metadata"] = PointMetadata(
                    source_name=meta_flat.get("source_name"),
                    source_id=meta_flat.get("source_id"),
                    extra_columns=meta_flat.get("extra_columns") or {},
                )

        return values

    @model_validator(mode="after")
    def _validate_lengths(self) -> PointSourceData:
        if self.ra_rad.ndim != 1:
            raise ValueError(
                f"PointSourceData: ra_rad must be 1-D, got shape {self.ra_rad.shape}."
            )
        n = len(self.ra_rad)
        for name, arr in (
            ("dec_rad", self.dec_rad),
            ("flux", self.flux),
            ("spectral_index", self.spectral_index),
            ("stokes_q", self.stokes_q),
            ("stokes_u", self.stokes_u),
            ("stokes_v", self.stokes_v),
            ("ref_freq", self.ref_freq),
        ):
            if arr.ndim != 1:
                raise ValueError(
                    f"PointSourceData: {name} must be 1-D, got shape {arr.shape}."
                )
            if len(arr) != n:
                raise ValueError(
                    f"PointSourceData: {name} has length {len(arr)}, "
                    f"expected {n} (must match ra_rad)."
                )

        if self.spectral_coeffs is not None and self.spectral_coeffs.shape[0] != n:
            raise ValueError(
                f"PointSourceData: spectral_coeffs has {self.spectral_coeffs.shape[0]} "
                f"rows, expected {n}."
            )

        if self.morphology is not None and self.morphology.n_sources != n:
            raise ValueError(
                f"PointSourceData: morphology has {self.morphology.n_sources} "
                f"sources, expected {n}."
            )
        if self.polarization is not None and self.polarization.n_sources != n:
            raise ValueError(
                f"PointSourceData: polarization has {self.polarization.n_sources} "
                f"sources, expected {n}."
            )
        if self.metadata is not None:
            meta_n = self.metadata.n_sources()
            if meta_n is not None and meta_n != n:
                raise ValueError(
                    f"PointSourceData: metadata has {meta_n} sources, expected {n}."
                )

        if self.spectrum is not None and self.spectrum.n_sources != n:
            raise ValueError(
                f"PointSourceData: spectrum has {self.spectrum.n_sources} "
                f"sources, expected {n}."
            )

        for name in (*self._CORE_FIELDS, "spectral_coeffs"):
            _freeze(getattr(self, name))
        return self

    @property
    def n_sources(self) -> int:
        """Number of point sources."""
        return len(self.ra_rad)

    @property
    def spectral_type(self) -> SpectralType:
        """Which spectral model drives flux evaluation (see :class:`SpectralType`).

        Precedence: per-channel ``spectrum`` > log-polynomial
        ``spectral_coeffs`` (>1 term) > power-law ``spectral_index``.
        """
        if self.spectrum is not None:
            return SpectralType.PER_CHANNEL
        if self.spectral_coeffs is not None and self.spectral_coeffs.shape[1] > 1:
            return SpectralType.LOG_POLYNOMIAL
        return SpectralType.POWER_LAW

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
        """Return new instance with boolean mask applied to all arrays."""
        mask = _validate_mask(mask, self.n_sources, label="PointSourceData mask")
        return PointSourceData(
            ra_rad=self.ra_rad[mask],
            dec_rad=self.dec_rad[mask],
            flux=self.flux[mask],
            spectral_index=self.spectral_index[mask],
            stokes_q=self.stokes_q[mask],
            stokes_u=self.stokes_u[mask],
            stokes_v=self.stokes_v[mask],
            ref_freq=self.ref_freq[mask],
            spectral_coeffs=(
                self.spectral_coeffs[mask] if self.spectral_coeffs is not None else None
            ),
            morphology=(
                self.morphology.masked(mask) if self.morphology is not None else None
            ),
            polarization=(
                self.polarization.masked(mask)
                if self.polarization is not None
                else None
            ),
            metadata=(
                self.metadata.masked(mask) if self.metadata is not None else None
            ),
            spectrum=(
                self.spectrum.masked_sources(mask)
                if self.spectrum is not None
                else None
            ),
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

        rotation_measure = (
            self.polarization.rotation_measure
            if self.polarization is not None
            else None
        )
        major_arcsec = (
            self.morphology.major_arcsec if self.morphology is not None else None
        )
        minor_arcsec = (
            self.morphology.minor_arcsec if self.morphology is not None else None
        )
        pa_deg = self.morphology.pa_deg if self.morphology is not None else None

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
                rotation_measure[mask] if rotation_measure is not None else None
            ),
            "major_arcsec": major_arcsec[mask] if major_arcsec is not None else None,
            "minor_arcsec": minor_arcsec[mask] if minor_arcsec is not None else None,
            "pa_deg": pa_deg[mask] if pa_deg is not None else None,
            "spectral_coeffs": (
                self.spectral_coeffs[mask] if self.spectral_coeffs is not None else None
            ),
            "per_channel_flux": (
                self.spectrum.flux[:, mask] if self.spectrum is not None else None
            ),
            "per_channel_stokes_q": (
                self.spectrum.stokes_q[:, mask]
                if self.spectrum is not None and self.spectrum.stokes_q is not None
                else None
            ),
            "per_channel_stokes_u": (
                self.spectrum.stokes_u[:, mask]
                if self.spectrum is not None and self.spectrum.stokes_u is not None
                else None
            ),
            "per_channel_stokes_v": (
                self.spectrum.stokes_v[:, mask]
                if self.spectrum is not None and self.spectrum.stokes_v is not None
                else None
            ),
            "channel_frequencies": (
                self.spectrum.frequencies if self.spectrum is not None else None
            ),
        }

    # Tuple of all per-source 1-D core array field names (for iteration).
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

    __hash__ = None  # type: ignore[assignment]

    def _compare(
        self, other: PointSourceData, *, close: bool, rtol: float, atol: float
    ) -> bool:
        for name in (*self._CORE_FIELDS, "spectral_coeffs"):
            if not _arrays_equal(
                getattr(self, name),
                getattr(other, name),
                close=close,
                rtol=rtol,
                atol=atol,
            ):
                return False
        for name in ("morphology", "polarization", "metadata", "spectrum"):
            mine = getattr(self, name)
            theirs = getattr(other, name)
            if (mine is None) != (theirs is None):
                return False
            if mine is not None and not mine._compare(
                theirs, close=close, rtol=rtol, atol=atol
            ):
                return False
        return True

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, PointSourceData):
            return NotImplemented
        return self._compare(other, close=False, rtol=0.0, atol=0.0)

    def is_close(
        self, other: PointSourceData, *, rtol: float = 1e-7, atol: float = 0.0
    ) -> bool:
        if not isinstance(other, PointSourceData):
            return False
        return self._compare(other, close=True, rtol=rtol, atol=atol)
