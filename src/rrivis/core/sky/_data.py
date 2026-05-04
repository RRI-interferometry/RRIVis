# rrivis/core/sky/_data.py
"""Inner data containers for SkyModel.

This module is a LEAF dependency — it imports only numpy and healpy.
No imports from model.py, convert.py, combine.py, or loaders.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TypedDict

import healpy as hp
import numpy as np

# =============================================================================
# Provenance metadata
# =============================================================================


class MonopoleConvention(str, Enum):
    """How a sky model represents the sky-average (DC) brightness temperature.

    The enum values are the canonical strings used for serialization and for
    cross-model compatibility checks during combination.
    """

    ABSOLUTE_WITH_CMB = "absolute_with_cmb"
    ABSOLUTE_NO_CMB = "absolute_no_cmb"
    MEAN_SUBTRACTED = "mean_subtracted"
    UNKNOWN = "unknown"


class SkyCoverage(str, Enum):
    """Whether a sky model represents the full sky or a subset of it."""

    FULL_SKY = "full_sky"
    PARTIAL_SKY = "partial_sky"
    UNKNOWN = "unknown"


class SourceSubtractionStatus(str, Enum):
    """Whether discrete sources have been removed from a sky model's payload."""

    NONE = "none"
    ABOVE_THRESHOLD = "above_threshold"
    ALL = "all"
    UNKNOWN = "unknown"


DEFAULT_COVERAGE_FOOTPRINT_NSIDE = 256
DEFAULT_COVERAGE_FOOTPRINT_COORDINATE_FRAME = "icrs"


def _normalize_coverage_coordinate_frame(coordinate_frame: str) -> str:
    frame = str(coordinate_frame).lower()
    if frame not in {"icrs", "galactic"}:
        raise ValueError(
            "SkyFootprint.coordinate_frame must be 'icrs' or 'galactic', got "
            f"{coordinate_frame!r}."
        )
    return frame


@dataclass(frozen=True, eq=False)
class SkyFootprint:
    """Sparse HEALPix support mask for a sky product's angular footprint."""

    nside: int
    hpx_inds: np.ndarray
    coordinate_frame: str = DEFAULT_COVERAGE_FOOTPRINT_COORDINATE_FRAME

    def __post_init__(self) -> None:
        nside = int(self.nside)
        if not hp.isnsideok(nside):
            raise ValueError(f"SkyFootprint.nside must be a valid NSIDE, got {nside}.")
        object.__setattr__(self, "nside", nside)
        object.__setattr__(
            self,
            "coordinate_frame",
            _normalize_coverage_coordinate_frame(self.coordinate_frame),
        )

        full_n_pixels = hp.nside2npix(nside)
        hpx_inds = np.asarray(self.hpx_inds, dtype=np.int64)
        if hpx_inds.ndim != 1:
            raise ValueError(
                "SkyFootprint.hpx_inds must be a 1-D integer array of pixel indices."
            )
        if hpx_inds.size:
            if np.any(hpx_inds < 0) or np.any(hpx_inds >= full_n_pixels):
                raise ValueError(
                    "SkyFootprint.hpx_inds contains indices outside the valid "
                    f"range [0, {full_n_pixels})."
                )
            hpx_inds = np.unique(hpx_inds)
        object.__setattr__(self, "hpx_inds", hpx_inds)

    @classmethod
    def from_mask(
        cls,
        mask: np.ndarray,
        *,
        nside: int,
        coordinate_frame: str = DEFAULT_COVERAGE_FOOTPRINT_COORDINATE_FRAME,
    ) -> SkyFootprint:
        """Build a sparse footprint from a dense boolean HEALPix mask."""
        mask = np.asarray(mask, dtype=bool)
        if mask.ndim != 1:
            raise ValueError("SkyFootprint.from_mask requires a 1-D boolean mask.")
        expected = hp.nside2npix(int(nside))
        if mask.size != expected:
            raise ValueError(
                "SkyFootprint.from_mask got a mask of length "
                f"{mask.size}, expected {expected} for nside={int(nside)}."
            )
        return cls(
            nside=int(nside),
            coordinate_frame=coordinate_frame,
            hpx_inds=np.flatnonzero(mask),
        )

    @property
    def full_n_pixels(self) -> int:
        """Total number of HEALPix pixels on the footprint grid."""
        return int(hp.nside2npix(self.nside))

    @property
    def coverage_fraction(self) -> float:
        """Fraction of the full sky covered by this footprint."""
        return float(self.hpx_inds.size / self.full_n_pixels)

    @property
    def is_full_sky(self) -> bool:
        """True when the footprint covers every pixel on its grid."""
        return self.hpx_inds.size == self.full_n_pixels

    def to_mask(self) -> np.ndarray:
        """Materialize the sparse footprint to a dense boolean HEALPix mask."""
        mask = np.zeros(self.full_n_pixels, dtype=bool)
        mask[self.hpx_inds] = True
        return mask

    def _require_compatible(self, other: SkyFootprint) -> None:
        if not isinstance(other, SkyFootprint):
            raise TypeError(
                "SkyFootprint operations require another SkyFootprint, got "
                f"{type(other).__name__}."
            )
        if self.nside != other.nside:
            raise ValueError(
                "SkyFootprint operations require matching nside values, got "
                f"{self.nside} and {other.nside}."
            )
        if self.coordinate_frame != other.coordinate_frame:
            raise ValueError(
                "SkyFootprint operations require matching coordinate frames, got "
                f"{self.coordinate_frame!r} and {other.coordinate_frame!r}."
            )

    def union(self, *others: SkyFootprint) -> SkyFootprint:
        """Return the geometric union with one or more compatible footprints."""
        if not others:
            return self
        parts = [self.hpx_inds]
        for other in others:
            self._require_compatible(other)
            parts.append(other.hpx_inds)
        return SkyFootprint(
            nside=self.nside,
            coordinate_frame=self.coordinate_frame,
            hpx_inds=np.unique(np.concatenate(parts)),
        )

    def intersect(self, other: SkyFootprint) -> SkyFootprint:
        """Return the geometric intersection with another compatible footprint."""
        self._require_compatible(other)
        return SkyFootprint(
            nside=self.nside,
            coordinate_frame=self.coordinate_frame,
            hpx_inds=np.intersect1d(
                self.hpx_inds,
                other.hpx_inds,
                assume_unique=True,
            ),
        )

    def intersect_mask(self, mask: np.ndarray) -> SkyFootprint:
        """Intersect the footprint with a dense boolean mask on the same grid."""
        mask = np.asarray(mask, dtype=bool)
        if mask.ndim != 1 or mask.size != self.full_n_pixels:
            raise ValueError(
                "SkyFootprint.intersect_mask requires a 1-D boolean mask of length "
                f"{self.full_n_pixels}."
            )
        return SkyFootprint(
            nside=self.nside,
            coordinate_frame=self.coordinate_frame,
            hpx_inds=self.hpx_inds[mask[self.hpx_inds]],
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SkyFootprint):
            return NotImplemented
        return (
            self.nside == other.nside
            and self.coordinate_frame == other.coordinate_frame
            and np.array_equal(self.hpx_inds, other.hpx_inds)
        )

    def __hash__(self) -> int:
        return hash(
            (
                self.nside,
                self.coordinate_frame,
                self.hpx_inds.tobytes(),
            )
        )


@dataclass(frozen=True)
class SkyProvenance:
    """Physical-correctness metadata attached to a :class:`SkyModel`.

    Used by ``combine_models`` to verify that models being summed are
    physically disjoint (no double-counting).  Every field is optional and
    defaults to an ``UNKNOWN`` sentinel so that user code that does not
    declare provenance keeps working until it crosses the disjointness
    validator.

    Attributes
    ----------
    flux_completeness_jy
        ``(S_min, S_max)`` flux-density range in Jy over which the model is
        complete, at ``flux_completeness_freq_hz``.  ``None`` if not declared.
    flux_completeness_freq_hz
        Reference frequency (Hz) at which ``flux_completeness_jy`` is evaluated.
    angular_resolution_rad
        ``(theta_min, theta_max)`` angular-scale range (radians) that the
        model faithfully represents.  For a point catalog, ``theta_min`` is
        the catalog's synthesized-beam FWHM and ``theta_max`` is the largest
        extended structure recoverable.  For a diffuse map, ``theta_min`` is
        the native pixel/beam resolution and ``theta_max`` is ``pi`` (full sky).
    sky_coverage
        Whether the payload represents the full sky or only a subset.  This
        governs whether ``monopole_k`` may be interpreted as a true full-sky
        average brightness temperature.
    coverage_fraction
        Fraction of ``4π`` steradians covered by the payload, when known.
    coverage_footprint
        Exact HEALPix footprint of the payload support when known. When
        provided, ``sky_coverage`` and ``coverage_fraction`` are derived from it
        and must remain consistent with that geometry.
    monopole_convention
        How the DC (sky-average) level is represented.  Mixing incompatible
        conventions is unambiguously wrong and raises under every policy.
    monopole_k
        The full-sky Stokes-I sky-average brightness temperature in Kelvin.
        ``None`` when the payload is partial-sky or when the monopole is not
        known.
    source_subtraction
        Whether discrete sources have been removed from the model's payload.
    source_subtraction_threshold_jy
        If ``source_subtraction == ABOVE_THRESHOLD``, the flux-density cut
        above which sources were removed (Jy).
    source_subtraction_freq_hz
        Reference frequency at which ``source_subtraction_threshold_jy`` was
        evaluated (Hz).
    source_subtraction_method
        Free-form tag describing the subtraction method
        (e.g. ``"gaussian_fit_inpaint"``, ``"catalog_masking"``).
    notes
        Free-form provenance notes (e.g. paper reference or pipeline tag).
    """

    flux_completeness_jy: tuple[float, float] | None = None
    flux_completeness_freq_hz: float | None = None
    angular_resolution_rad: tuple[float, float] | None = None
    sky_coverage: SkyCoverage = SkyCoverage.UNKNOWN
    coverage_fraction: float | None = None
    coverage_footprint: SkyFootprint | None = None
    monopole_convention: MonopoleConvention = MonopoleConvention.UNKNOWN
    monopole_k: float | None = None
    source_subtraction: SourceSubtractionStatus = SourceSubtractionStatus.UNKNOWN
    source_subtraction_threshold_jy: float | None = None
    source_subtraction_freq_hz: float | None = None
    source_subtraction_method: str | None = None
    notes: str | None = None

    def __post_init__(self) -> None:
        """Coerce string inputs to the declared Enum types."""
        if isinstance(self.monopole_convention, str) and not isinstance(
            self.monopole_convention, MonopoleConvention
        ):
            object.__setattr__(
                self,
                "monopole_convention",
                MonopoleConvention(self.monopole_convention),
            )
        if isinstance(self.sky_coverage, str) and not isinstance(
            self.sky_coverage, SkyCoverage
        ):
            object.__setattr__(
                self,
                "sky_coverage",
                SkyCoverage(self.sky_coverage),
            )
        if isinstance(self.source_subtraction, str) and not isinstance(
            self.source_subtraction, SourceSubtractionStatus
        ):
            object.__setattr__(
                self,
                "source_subtraction",
                SourceSubtractionStatus(self.source_subtraction),
            )
        if isinstance(self.coverage_footprint, dict):
            object.__setattr__(
                self,
                "coverage_footprint",
                SkyFootprint(**self.coverage_footprint),
            )
        elif self.coverage_footprint is not None and not isinstance(
            self.coverage_footprint, SkyFootprint
        ):
            raise TypeError(
                "SkyProvenance.coverage_footprint must be a SkyFootprint, a dict "
                f"of its fields, or None; got "
                f"{type(self.coverage_footprint).__name__}."
            )

        if self.coverage_footprint is not None:
            footprint_fraction = self.coverage_footprint.coverage_fraction
            footprint_coverage = (
                SkyCoverage.FULL_SKY
                if self.coverage_footprint.is_full_sky
                else SkyCoverage.PARTIAL_SKY
            )
            if self.sky_coverage == SkyCoverage.UNKNOWN:
                object.__setattr__(self, "sky_coverage", footprint_coverage)
            elif self.sky_coverage != footprint_coverage:
                raise ValueError(
                    "SkyProvenance: coverage_footprint implies "
                    f"sky_coverage={footprint_coverage.value!r}, got "
                    f"{self.sky_coverage.value!r}."
                )
            if self.coverage_fraction is None:
                object.__setattr__(self, "coverage_fraction", footprint_fraction)
            elif not np.isclose(float(self.coverage_fraction), footprint_fraction):
                raise ValueError(
                    "SkyProvenance: coverage_fraction is inconsistent with "
                    "coverage_footprint."
                )

        # Sanity checks — when a threshold is given, the status must match.
        if (
            self.source_subtraction_threshold_jy is not None
            and self.source_subtraction == SourceSubtractionStatus.NONE
        ):
            raise ValueError(
                "SkyProvenance: source_subtraction_threshold_jy was set but "
                "source_subtraction is NONE."
            )
        if (
            self.source_subtraction == SourceSubtractionStatus.ABOVE_THRESHOLD
            and self.source_subtraction_threshold_jy is None
        ):
            raise ValueError(
                "SkyProvenance: source_subtraction=ABOVE_THRESHOLD requires a "
                "source_subtraction_threshold_jy."
            )
        if self.coverage_fraction is not None:
            coverage_fraction = float(self.coverage_fraction)
            if not np.isfinite(coverage_fraction) or not (
                0.0 <= coverage_fraction <= 1.0
            ):
                raise ValueError(
                    "SkyProvenance: coverage_fraction must lie in [0, 1], got "
                    f"{self.coverage_fraction!r}."
                )
            object.__setattr__(self, "coverage_fraction", coverage_fraction)

        if self.sky_coverage == SkyCoverage.FULL_SKY:
            if self.coverage_fraction is None:
                object.__setattr__(self, "coverage_fraction", 1.0)
            elif not np.isclose(self.coverage_fraction, 1.0):
                raise ValueError(
                    "SkyProvenance: sky_coverage=FULL_SKY requires "
                    "coverage_fraction=1.0."
                )
        elif self.sky_coverage == SkyCoverage.PARTIAL_SKY:
            if self.coverage_fraction is not None and np.isclose(
                self.coverage_fraction, 1.0
            ):
                raise ValueError(
                    "SkyProvenance: sky_coverage=PARTIAL_SKY requires "
                    "coverage_fraction < 1.0."
                )
            if self.monopole_k is not None:
                raise ValueError(
                    "SkyProvenance: monopole_k is a full-sky quantity and must "
                    "be None when sky_coverage=PARTIAL_SKY."
                )

    @property
    def has_flux_completeness(self) -> bool:
        """True if the flux-completeness range is declared."""
        return (
            self.flux_completeness_jy is not None
            and self.flux_completeness_freq_hz is not None
        )

    @property
    def has_angular_resolution(self) -> bool:
        """True if the angular-resolution range is declared."""
        return self.angular_resolution_rad is not None

    @property
    def is_source_subtracted(self) -> bool:
        """True if the model has sources removed at a known threshold."""
        return self.source_subtraction in (
            SourceSubtractionStatus.ABOVE_THRESHOLD,
            SourceSubtractionStatus.ALL,
        )

    @property
    def is_full_sky(self) -> bool:
        """True if the payload is declared to cover the full sky."""
        return self.sky_coverage == SkyCoverage.FULL_SKY

    @property
    def is_partial_sky(self) -> bool:
        """True if the payload is declared to cover a sky subset."""
        return self.sky_coverage == SkyCoverage.PARTIAL_SKY


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

    # Optional per-channel flux tables (lossless multi-frequency spectrum).
    # When populated, consumers evaluate flux at an observation frequency
    # via nearest-channel lookup rather than spectral-index extrapolation.
    per_channel_flux: np.ndarray | None = None  # shape (n_freq, N) — Stokes I Jy
    per_channel_stokes_q: np.ndarray | None = None  # shape (n_freq, N)
    per_channel_stokes_u: np.ndarray | None = None  # shape (n_freq, N)
    per_channel_stokes_v: np.ndarray | None = None  # shape (n_freq, N)
    channel_frequencies: np.ndarray | None = None  # shape (n_freq,), Hz, ascending

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

        # Per-channel flux tables (all-or-none between per_channel_flux and
        # channel_frequencies; Q/U paired; V independently optional).
        if (self.per_channel_flux is None) != (self.channel_frequencies is None):
            raise ValueError(
                "PointSourceData: per_channel_flux and channel_frequencies must "
                "be set together (or both None)."
            )
        if self.channel_frequencies is not None:
            freqs = np.asarray(self.channel_frequencies)
            if freqs.ndim != 1 or freqs.size == 0:
                raise ValueError(
                    "PointSourceData: channel_frequencies must be a non-empty 1-D array."
                )
            if not np.all(np.isfinite(freqs)) or np.any(freqs <= 0):
                raise ValueError(
                    "PointSourceData: channel_frequencies must be finite and positive."
                )
            if freqs.size > 1 and not np.all(np.diff(freqs) > 0):
                raise ValueError(
                    "PointSourceData: channel_frequencies must be strictly ascending."
                )
            n_freq = freqs.size
            assert self.per_channel_flux is not None  # for type checkers
            if self.per_channel_flux.shape != (n_freq, n):
                raise ValueError(
                    "PointSourceData: per_channel_flux shape "
                    f"{self.per_channel_flux.shape} does not match "
                    f"(n_channel_frequencies={n_freq}, n_sources={n})."
                )
            if (self.per_channel_stokes_q is None) != (
                self.per_channel_stokes_u is None
            ):
                raise ValueError(
                    "PointSourceData: per_channel_stokes_q and per_channel_stokes_u "
                    "must be set together."
                )
            for name in (
                "per_channel_stokes_q",
                "per_channel_stokes_u",
                "per_channel_stokes_v",
            ):
                arr = getattr(self, name)
                if arr is not None and arr.shape != (n_freq, n):
                    raise ValueError(
                        f"PointSourceData: {name} shape {arr.shape} does not match "
                        f"per_channel_flux shape {(n_freq, n)}."
                    )
        else:
            # Guard against user setting per-channel Stokes without the primary table.
            for name in (
                "per_channel_stokes_q",
                "per_channel_stokes_u",
                "per_channel_stokes_v",
            ):
                if getattr(self, name) is not None:
                    raise ValueError(
                        f"PointSourceData: {name} requires per_channel_flux to be set."
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
            per_channel_flux=(
                self.per_channel_flux[:, mask]
                if self.per_channel_flux is not None
                else None
            ),
            per_channel_stokes_q=(
                self.per_channel_stokes_q[:, mask]
                if self.per_channel_stokes_q is not None
                else None
            ),
            per_channel_stokes_u=(
                self.per_channel_stokes_u[:, mask]
                if self.per_channel_stokes_u is not None
                else None
            ),
            per_channel_stokes_v=(
                self.per_channel_stokes_v[:, mask]
                if self.per_channel_stokes_v is not None
                else None
            ),
            channel_frequencies=self.channel_frequencies,
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
                self.per_channel_flux[:, mask]
                if self.per_channel_flux is not None
                else None
            ),
            "per_channel_stokes_q": (
                self.per_channel_stokes_q[:, mask]
                if self.per_channel_stokes_q is not None
                else None
            ),
            "per_channel_stokes_u": (
                self.per_channel_stokes_u[:, mask]
                if self.per_channel_stokes_u is not None
                else None
            ),
            "per_channel_stokes_v": (
                self.per_channel_stokes_v[:, mask]
                if self.per_channel_stokes_v is not None
                else None
            ),
            "channel_frequencies": self.channel_frequencies,
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
