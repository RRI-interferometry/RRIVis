"""Point-source columnar payload + per-channel spectrum table.

Provides :class:`PointSourceData` (Nx columns of RA/Dec/flux/Stokes/etc.)
and :class:`PointSpectrum` (lossless per-channel flux samples, used by
catalogues that supply tabulated spectra rather than spectral indices).

``SourceArrays`` is the flat-dict view consumed by the visibility code.
"""

from __future__ import annotations

from dataclasses import field
from typing import TypedDict

import numpy as np
from pydantic import field_validator, model_validator
from pydantic.dataclasses import dataclass

from ._shared import _FROZEN_NDARRAY_CONFIG, _arrays_equal

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
    """

    flux: np.ndarray  # shape (n_freq, N), Stokes I in Jy
    frequencies: np.ndarray  # shape (n_freq,), Hz, ascending
    stokes_q: np.ndarray | None = None  # shape (n_freq, N)
    stokes_u: np.ndarray | None = None  # shape (n_freq, N)
    stokes_v: np.ndarray | None = None  # shape (n_freq, N)

    @field_validator("frequencies", mode="before")
    @classmethod
    def _validate_frequencies(cls, value: object) -> np.ndarray:
        freqs = np.asarray(value)
        if freqs.ndim != 1 or freqs.size == 0:
            raise ValueError("PointSpectrum.frequencies must be a non-empty 1-D array.")
        if not np.all(np.isfinite(freqs)) or np.any(freqs <= 0):
            raise ValueError("PointSpectrum.frequencies must be finite and positive.")
        if freqs.size > 1 and not np.all(np.diff(freqs) > 0):
            raise ValueError("PointSpectrum.frequencies must be strictly ascending.")
        return freqs

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
        return self

    @property
    def n_frequencies(self) -> int:
        return int(self.frequencies.size)

    @property
    def n_sources(self) -> int:
        return int(self.flux.shape[1])

    def masked_sources(self, mask: np.ndarray) -> PointSpectrum:
        """Return a new PointSpectrum with a boolean mask applied along sources."""
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
# PointSourceData
# =============================================================================


@dataclass(frozen=True, eq=False, config=_FROZEN_NDARRAY_CONFIG)
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

    # Optional lossless multi-frequency Stokes-flux table. When populated,
    # consumers evaluate flux at an observation frequency via nearest-channel
    # lookup rather than spectral-index extrapolation.
    spectrum: PointSpectrum | None = None

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
                "PointSourceData.extra_columns must be a dict, got "
                f"{type(value).__name__}."
            )
        normalized: dict[str, np.ndarray] = {}
        for name, arr in value.items():
            arr_np = np.asarray(arr)
            if arr_np.ndim != 1:
                raise ValueError(
                    f"PointSourceData: extra column {name!r} must be 1-D, "
                    f"got shape {arr_np.shape}."
                )
            normalized[name] = arr_np
        return normalized

    @model_validator(mode="after")
    def _validate_lengths(self) -> PointSourceData:
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

        morph = (self.major_arcsec, self.minor_arcsec, self.pa_deg)
        morph_present = sum(1 for m in morph if m is not None)
        if morph_present not in (0, 3):
            raise ValueError(
                "PointSourceData: major_arcsec, minor_arcsec, pa_deg must be "
                "all set or all None."
            )

        for name, arr in [
            ("rotation_measure", self.rotation_measure),
            ("major_arcsec", self.major_arcsec),
            ("minor_arcsec", self.minor_arcsec),
            ("pa_deg", self.pa_deg),
            ("source_name", self.source_name),
            ("source_id", self.source_id),
        ]:
            if arr is not None and len(arr) != n:
                raise ValueError(
                    f"PointSourceData: {name} has length {len(arr)}, expected {n}."
                )
        if self.spectral_coeffs is not None and self.spectral_coeffs.shape[0] != n:
            raise ValueError(
                f"PointSourceData: spectral_coeffs has {self.spectral_coeffs.shape[0]} "
                f"rows, expected {n}."
            )

        if self.spectrum is not None and self.spectrum.n_sources != n:
            raise ValueError(
                f"PointSourceData: spectrum has {self.spectrum.n_sources} "
                f"sources, expected {n}."
            )

        for name, arr in self.extra_columns.items():
            if len(arr) != n:
                raise ValueError(
                    f"PointSourceData: extra column {name!r} has length {len(arr)}, "
                    f"expected {n}."
                )
        return self

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
            spectrum=self.spectrum.masked_sources(mask)
            if self.spectrum is not None
            else None,
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

    __hash__ = None  # type: ignore[assignment]

    def _compare(
        self, other: PointSourceData, *, close: bool, rtol: float, atol: float
    ) -> bool:
        array_fields = (
            *self._CORE_FIELDS,
            *self._OPTIONAL_FIELDS,
            *self._METADATA_FIELDS,
            "spectral_coeffs",
        )
        for name in array_fields:
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
        if (self.spectrum is None) != (other.spectrum is None):
            return False
        if self.spectrum is not None and other.spectrum is not None:
            if not self.spectrum._compare(
                other.spectrum, close=close, rtol=rtol, atol=atol
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


# =============================================================================
# HealpixData
# =============================================================================
