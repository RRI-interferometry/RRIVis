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

from collections.abc import Mapping
from dataclasses import field
from typing import TypedDict

import numpy as np
from pydantic import ConfigDict, field_validator, model_validator
from pydantic.dataclasses import dataclass

from ._shared import (
    _FROZEN_NDARRAY_CONFIG,
    _arrays_equal,
    _freeze,
    _require_floating_array,
    _validate_mask,
    validate_frequency_axis,
)
from .constants import SpectralType

#: PointSourceData is nested-only: flat per-source column dicts are packed into
#: the nested sub-blocks by support.point_builder.point_source_data_from_mapping
#: *before* construction. Forbidding extras turns a stray flat kwarg passed to
#: the raw constructor into a loud error instead of a silently-dropped column.
_POINT_SOURCE_DATA_CONFIG = ConfigDict(arbitrary_types_allowed=True, extra="forbid")


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

    Flux-table dtype policy
    -----------------------
    ``flux`` and optional ``stokes_*`` tables must use a floating dtype
    (``float32`` or ``float64``). Integer, complex, and object arrays are
    rejected at construction via :func:`._shared._require_floating_array`.
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
        _require_floating_array(self.flux, label="PointSpectrum.flux")
        for name in ("stokes_q", "stokes_u", "stokes_v"):
            _require_floating_array(getattr(self, name), label=f"PointSpectrum.{name}")
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
        for name, arr in (
            ("major_arcsec", self.major_arcsec),
            ("minor_arcsec", self.minor_arcsec),
            ("pa_deg", self.pa_deg),
        ):
            _require_floating_array(arr, label=f"PointMorphology.{name}")
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
        _require_floating_array(
            self.rotation_measure, label="PointPolarization.rotation_measure"
        )
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


@dataclass(frozen=True, eq=False, config=_POINT_SOURCE_DATA_CONFIG)
class PointSourceData:
    """Columnar arrays for point-source sky model.

    Core arrays (always populated, possibly zero-length) carry RA/Dec,
    Stokes I/Q/U/V at the reference frequency, the spectral index and
    reference-frequency value. Extension blocks
    (:class:`PointMorphology`, :class:`PointPolarization`,
    :class:`PointMetadata`) are independently optional and grouped so
    each block validates its own shape rules in isolation.

    The constructor is **nested-only**: it accepts the core/spectral arrays
    plus pre-built :class:`PointMorphology` / :class:`PointPolarization` /
    :class:`PointMetadata` / :class:`PointSpectrum` blocks. Flat per-source
    column dicts (e.g. ``major_arcsec=…``, ``rotation_measure=…``,
    ``source_name=…``) are packed into the nested blocks by
    :func:`support.point_builder.point_source_data_from_mapping`, which is the
    single column-oriented construction route used by
    :func:`create_from_arrays` and ``combine/engine.py``.
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

    @model_validator(mode="after")
    def _validate_lengths(self) -> PointSourceData:
        for name in self._CORE_FIELDS:
            _require_floating_array(
                getattr(self, name), label=f"PointSourceData.{name}"
            )
        _require_floating_array(
            self.spectral_coeffs, label="PointSourceData.spectral_coeffs"
        )

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
    def populated_spectral_fields(self) -> frozenset[SpectralType]:
        """Every spectral representation actually carried by this payload.

        Unlike :attr:`spectral_type` (which names only the *driving*
        representation by precedence), this lists *all* populated ones. The
        power law (``spectral_index``) is always term 0 of any spectral model,
        so :attr:`SpectralType.POWER_LAW` is always present.
        :attr:`SpectralType.LOG_POLYNOMIAL` is added when ``spectral_coeffs``
        carries more than one term, and :attr:`SpectralType.PER_CHANNEL` when a
        :class:`PointSpectrum` is attached.

        This is the introspection surface for the opt-in exclusivity gate
        :meth:`assert_single_spectral_representation`.
        """
        present = {SpectralType.POWER_LAW}
        if self.spectral_coeffs is not None and self.spectral_coeffs.shape[1] > 1:
            present.add(SpectralType.LOG_POLYNOMIAL)
        if self.spectrum is not None:
            present.add(SpectralType.PER_CHANNEL)
        return frozenset(present)

    def assert_single_spectral_representation(self) -> None:
        """Assert at most one higher-order spectral representation is populated.

        Opt-in exclusivity gate for callers that require single-representation
        mode. The bare power law (``spectral_index``) is always allowed; this
        only forbids co-populating more than one of the *higher-order*
        representations (log-polynomial ``spectral_coeffs`` and per-channel
        ``spectrum``). Construction itself never enforces this — the layered
        design permits co-population — so call this explicitly where a single
        representation is contractually required.

        Raises
        ------
        ValueError
            If more than one higher-order representation is populated, naming
            the co-populated representations.
        """
        extra = self.populated_spectral_fields - {SpectralType.POWER_LAW}
        if len(extra) > 1:
            names = ", ".join(sorted(member.value for member in extra))
            raise ValueError(
                "PointSourceData carries multiple higher-order spectral "
                f"representations ({names}); only one higher-order "
                "representation may be populated in single-representation mode. "
                "Use populated_spectral_fields to inspect which are present."
            )

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


# ---------------------------------------------------------------------------
# SCI-004 Section 5.1: canonical tangent-polarization metadata
# ---------------------------------------------------------------------------

#: Section 5.1's exact schema literal.
TANGENT_POLARIZATION_SCHEMA = "radiosim.sky-tangent-polarization.v1"

#: Section 5.1's exact six-key surface, in the order the design prints it.
TANGENT_POLARIZATION_KEYS: tuple[str, ...] = (
    "schema_version",
    "coordinate_frame",
    "axes",
    "position_angle",
    "linear_complex",
    "stokes_v",
)

#: The two coordinate frames Section 5.1 admits.
TANGENT_COORDINATE_FRAMES: frozenset[str] = frozenset({"icrs", "galactic"})

#: The canonical position-angle convention, and the HEALPix/CMB one that must be
#: converted into it before canonical storage.
POSITION_ANGLE_NORTH_THROUGH_EAST = "north_through_east"
POSITION_ANGLE_NORTH_THROUGH_WEST = "north_through_west"


@dataclass(frozen=True, config=ConfigDict(arbitrary_types_allowed=True))
class TangentPolarizationFrame:
    """SCI-004 Section 5.1's strict frozen tangent-polarization record.

    ``docs/development/sci004_mmode_design.md`` Section 5.1 requires exactly six
    fields, and requires every point or HEALPix payload with non-zero ``Q`` or
    ``U`` to carry them: "Today point and HEALPix containers carry numerical
    ``Q``/``U`` arrays but no complete tangent-basis record. That is insufficient
    for spin harmonics."

    Point ``Q``/``U`` are defined in the local tangent plane of each catalogue
    direction and HEALPix ``Q``/``U`` in the local tangent plane of each pixel.
    A HEALPix/CMB ``U`` convention -- position angle measured North *through
    West* -- is converted explicitly to RadioSim IAU North-through-East before
    canonical storage; relabelling the payload is forbidden.  An ``I``/``V``-only
    payload may omit the block entirely.

    Examples
    --------
    >>> frame = TangentPolarizationFrame.canonical("icrs")
    >>> tuple(frame.as_mapping())
    ('schema_version', 'coordinate_frame', 'axes', 'position_angle', 'linear_complex', 'stokes_v')
    >>> frame.position_angle
    'north_through_east'
    """

    schema_version: str = TANGENT_POLARIZATION_SCHEMA
    coordinate_frame: str = "icrs"
    axes: str = "north_east"
    position_angle: str = POSITION_ANGLE_NORTH_THROUGH_EAST
    linear_complex: str = "q_plus_i_u"
    stokes_v: str = "iau_incoming_r_minus_l"

    @model_validator(mode="after")
    def _validate_literals(self) -> TangentPolarizationFrame:
        if self.schema_version != TANGENT_POLARIZATION_SCHEMA:
            raise ValueError(
                f"tangent frame schema_version must be {TANGENT_POLARIZATION_SCHEMA!r}"
            )
        if self.coordinate_frame not in TANGENT_COORDINATE_FRAMES:
            raise ValueError(
                "tangent frame coordinate_frame must be 'icrs' or 'galactic'"
            )
        if self.axes != "north_east":
            raise ValueError("tangent frame axes must be 'north_east'")
        if self.position_angle != POSITION_ANGLE_NORTH_THROUGH_EAST:
            raise ValueError(
                "a stored tangent frame is canonical IAU north_through_east; a "
                "HEALPix/CMB payload is converted with to_canonical() first"
            )
        if self.linear_complex != "q_plus_i_u":
            raise ValueError("tangent frame linear_complex must be 'q_plus_i_u'")
        if self.stokes_v != "iau_incoming_r_minus_l":
            raise ValueError("tangent frame stokes_v must be 'iau_incoming_r_minus_l'")
        return self

    @classmethod
    def canonical(cls, coordinate_frame: str = "icrs") -> TangentPolarizationFrame:
        """Return the canonical frame for one coordinate system."""
        return cls(coordinate_frame=str(coordinate_frame))

    def as_mapping(self) -> dict[str, str]:
        """Return the exact six-key object, in Section 5.1's order."""
        return {
            "schema_version": self.schema_version,
            "coordinate_frame": self.coordinate_frame,
            "axes": self.axes,
            "position_angle": self.position_angle,
            "linear_complex": self.linear_complex,
            "stokes_v": self.stokes_v,
        }

    @classmethod
    def from_mapping(cls, payload: object) -> TangentPolarizationFrame:
        """Build a frame from a declared mapping, rejecting unknown keys."""
        if isinstance(payload, TangentPolarizationFrame):
            return payload
        if not isinstance(payload, Mapping):
            raise ValueError("a declared tangent frame must be a mapping")
        unknown = set(payload) - set(TANGENT_POLARIZATION_KEYS)
        if unknown:
            raise ValueError(f"unknown tangent frame keys {sorted(unknown)}")
        missing = set(TANGENT_POLARIZATION_KEYS) - set(payload)
        if missing:
            raise ValueError(f"missing tangent frame keys {sorted(missing)}")
        return cls(**{key: str(payload[key]) for key in TANGENT_POLARIZATION_KEYS})

    @staticmethod
    def to_canonical(
        *,
        stokes_q: object,
        stokes_u: object,
        position_angle: str,
    ) -> tuple[object, object]:
        r"""Convert a declared source convention to IAU North-through-East.

        Section 5.1: "a HEALPix/CMB ``U`` convention is converted explicitly to
        RadioSim IAU North-through-East before canonical storage; tests pin the
        sign with a rotated pure-Q map."  A position angle ``chi`` measured
        North through East gives ``Q = p cos(2 chi)``, ``U = p sin(2 chi)``;
        measuring it North through West reverses the sense of ``chi`` and
        therefore the sign of ``U`` alone, leaving ``Q`` untouched.

        Copying ``U`` through unchanged, or rotating only pixel indices, is a
        different sky object and is forbidden.
        """
        convention = str(position_angle)
        if convention == POSITION_ANGLE_NORTH_THROUGH_EAST:
            return (stokes_q, stokes_u)
        if convention == POSITION_ANGLE_NORTH_THROUGH_WEST:
            return (
                stokes_q,
                -np.asarray(stokes_u)
                if isinstance(stokes_u, np.ndarray)
                else -stokes_u,
            )
        raise ValueError(
            "position_angle must be 'north_through_east' or 'north_through_west'"
        )

    @staticmethod
    def require_for(
        *,
        stokes_q: object,
        stokes_u: object,
        stokes_v: object = 0.0,
        frame: object = None,
    ) -> TangentPolarizationFrame | None:
        """Resolve the frame a payload must carry, or reject an undeclared one.

        Section 5.1: "Every point or HEALPix payload with non-zero ``Q`` or ``U``
        must carry it. ... A programmatic polarized input without a declared
        source convention is rejected. An I/V-only payload may omit the tangent
        block."  ``V`` is deliberately not part of the trigger: it is a scalar
        (spin-0) field with no tangent-basis dependence.
        """
        del stokes_v
        linear = _has_nonzero_component(stokes_q) or _has_nonzero_component(stokes_u)
        if not linear:
            return (
                TangentPolarizationFrame.from_mapping(frame)
                if frame is not None
                else None
            )
        if frame is None:
            raise ValueError(
                "a polarized sky payload with non-zero Q or U requires an explicit "
                "canonical tangent-polarization frame (SCI-004 Section 5.1)"
            )
        return TangentPolarizationFrame.from_mapping(frame)


def _has_nonzero_component(value: object) -> bool:
    """Return whether a Stokes payload has any finite non-zero element."""
    if value is None:
        return False
    array = np.atleast_1d(np.asarray(value, dtype=np.float64))
    if array.size == 0:
        return False
    finite = array[np.isfinite(array)]
    return bool(np.any(finite != 0.0))
