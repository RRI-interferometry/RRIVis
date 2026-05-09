# radiosim/core/sky/model.py
"""
Unified SkyModel dataclass.

Holds sky data in either point-source (columnar arrays) or multi-frequency
HEALPix format, with bidirectional conversion, combination, and precision
management.  Methods that change payloads return *new* instances via
``dataclasses.replace``.

HEALPix maps are stored as 2-D numpy arrays of shape ``(n_freq, npix)``
rather than ``dict[float, ndarray]``. ``HealpixData.frequencies`` provides
the frequency axis.
"""

import logging
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import TYPE_CHECKING, Any

import numpy as np

from ._data import (
    HealpixData,
    PointSourceData,
    PointSpectrum,
    SkyCoverage,
    SkyProvenance,
    SourceArrays,
)
from .constants import BrightnessConversion

if TYPE_CHECKING:
    from radiosim.core.precision import PrecisionConfig

    from .region import SkyRegion

logger = logging.getLogger(__name__)


# =============================================================================
# Sky Format Enum
# =============================================================================


class SkyFormat(Enum):
    """Sky model representation format."""

    POINT_SOURCES = "point_sources"
    HEALPIX = "healpix_map"


def _coerce_format(representation: "SkyFormat | str") -> SkyFormat:
    """Coerce a string or SkyFormat to a SkyFormat enum value."""
    if isinstance(representation, SkyFormat):
        return representation
    try:
        return SkyFormat(representation)
    except ValueError:
        raise ValueError(
            f"Unknown representation '{representation}'. "
            f"Supported: SkyFormat.POINT_SOURCES, SkyFormat.HEALPIX."
        ) from None


# =============================================================================
# SkyModel Class
# =============================================================================


@dataclass(frozen=True, eq=False)
class SkyModel:
    """Unified sky model built from typed payloads.

    Frozen — use :meth:`replace` to derive new instances. The set of populated
    payloads (:attr:`formats`) is the canonical "what's loaded" signal; both
    point and healpix may be populated simultaneously (a hybrid model).

    Equality semantics
    ------------------
    ``==`` is bit-exact: it requires every payload array to match element-by-
    element via :func:`numpy.array_equal`.  Cross-precision comparisons
    (e.g. an f32 model versus the same logical sky in f64) will return
    ``False`` even when the only differences are representation error.

    For tolerant comparisons, call :meth:`is_close`, which accepts ``rtol`` /
    ``atol`` and is the appropriate tool for round-trip and precision tests.
    """

    point: PointSourceData | None = field(default=None, repr=False)
    healpix: HealpixData | None = field(default=None, repr=False)
    reference_frequency: float | None = None
    model_name: str | None = None
    brightness_conversion: BrightnessConversion = BrightnessConversion.PLANCK
    polarization_brightness_conversion: BrightnessConversion = (
        BrightnessConversion.RAYLEIGH_JEANS
    )
    provenance: SkyProvenance = field(default_factory=SkyProvenance)
    _precision: "PrecisionConfig | None" = field(default=None, repr=False)

    def __post_init__(self) -> None:
        """Validate payload presence and precision."""
        if self.point is None and self.healpix is None:
            raise ValueError(
                "SkyModel requires at least one payload. "
                "Use create_empty() for an empty point-source model."
            )

        if not isinstance(self.brightness_conversion, BrightnessConversion):
            try:
                object.__setattr__(
                    self,
                    "brightness_conversion",
                    BrightnessConversion(self.brightness_conversion),
                )
            except (ValueError, KeyError):
                raise ValueError(
                    f"brightness_conversion must be 'planck' or 'rayleigh-jeans', "
                    f"got '{self.brightness_conversion}'"
                ) from None

        if not isinstance(
            self.polarization_brightness_conversion, BrightnessConversion
        ):
            try:
                object.__setattr__(
                    self,
                    "polarization_brightness_conversion",
                    BrightnessConversion(self.polarization_brightness_conversion),
                )
            except (ValueError, KeyError):
                raise ValueError(
                    "polarization_brightness_conversion must be 'planck' or "
                    f"'rayleigh-jeans', got "
                    f"'{self.polarization_brightness_conversion}'"
                ) from None

        if isinstance(self.provenance, dict):
            object.__setattr__(self, "provenance", SkyProvenance(**self.provenance))
        elif not isinstance(self.provenance, SkyProvenance):
            raise TypeError(
                "SkyModel.provenance must be a SkyProvenance or a dict of its "
                f"fields, got {type(self.provenance).__name__}."
            )

        if self._precision is None:
            raise ValueError(
                "SkyModel requires an explicit PrecisionConfig. "
                "Pass precision=... to a loader or constructor."
            )

        if self.point is not None:
            object.__setattr__(
                self, "point", self._cast_point_data(self.point, self._precision)
            )
        if self.healpix is not None:
            object.__setattr__(
                self,
                "healpix",
                self._cast_healpix_data(
                    self.healpix,
                    self._precision,
                    self.brightness_conversion,
                ),
            )

    @classmethod
    def _cast_point_data(
        cls,
        point: PointSourceData | None,
        precision: "PrecisionConfig | None",
    ) -> PointSourceData | None:
        if point is None or precision is None:
            return point
        src_dt = precision.sky_model.get_dtype("source_positions")
        flux_dt = precision.sky_model.get_dtype("flux")
        si_dt = precision.sky_model.get_dtype("spectral_index")
        return PointSourceData(
            ra_rad=np.asarray(point.ra_rad, dtype=src_dt),
            dec_rad=np.asarray(point.dec_rad, dtype=src_dt),
            flux=np.asarray(point.flux, dtype=flux_dt),
            spectral_index=np.asarray(point.spectral_index, dtype=si_dt),
            stokes_q=np.asarray(point.stokes_q, dtype=flux_dt),
            stokes_u=np.asarray(point.stokes_u, dtype=flux_dt),
            stokes_v=np.asarray(point.stokes_v, dtype=flux_dt),
            ref_freq=np.asarray(point.ref_freq, dtype=flux_dt),
            rotation_measure=(
                None
                if point.rotation_measure is None
                else np.asarray(point.rotation_measure, dtype=flux_dt)
            ),
            major_arcsec=(
                None
                if point.major_arcsec is None
                else np.asarray(point.major_arcsec, dtype=src_dt)
            ),
            minor_arcsec=(
                None
                if point.minor_arcsec is None
                else np.asarray(point.minor_arcsec, dtype=src_dt)
            ),
            pa_deg=None
            if point.pa_deg is None
            else np.asarray(point.pa_deg, dtype=src_dt),
            spectral_coeffs=(
                None
                if point.spectral_coeffs is None
                else np.asarray(point.spectral_coeffs, dtype=si_dt)
            ),
            source_name=(
                None if point.source_name is None else np.asarray(point.source_name)
            ),
            source_id=None if point.source_id is None else np.asarray(point.source_id),
            extra_columns={
                name: np.asarray(values) for name, values in point.extra_columns.items()
            },
            spectrum=(
                None
                if point.spectrum is None
                else PointSpectrum(
                    flux=np.asarray(point.spectrum.flux, dtype=flux_dt),
                    frequencies=np.asarray(
                        point.spectrum.frequencies, dtype=np.float64
                    ),
                    stokes_q=(
                        None
                        if point.spectrum.stokes_q is None
                        else np.asarray(point.spectrum.stokes_q, dtype=flux_dt)
                    ),
                    stokes_u=(
                        None
                        if point.spectrum.stokes_u is None
                        else np.asarray(point.spectrum.stokes_u, dtype=flux_dt)
                    ),
                    stokes_v=(
                        None
                        if point.spectrum.stokes_v is None
                        else np.asarray(point.spectrum.stokes_v, dtype=flux_dt)
                    ),
                )
            ),
        )

    @classmethod
    def _cast_healpix_data(
        cls,
        healpix_data: HealpixData | None,
        precision: "PrecisionConfig | None",
        brightness_conversion: BrightnessConversion,
    ) -> HealpixData | None:
        if healpix_data is None or precision is None:
            return healpix_data
        hp_dt = precision.sky_model.get_dtype("healpix_maps")
        flux_dt = precision.sky_model.get_dtype("flux")

        def _cast_map(arr: np.ndarray | None) -> np.ndarray | None:
            if arr is None:
                return None
            return arr if arr.dtype == hp_dt else arr.astype(hp_dt, copy=False)

        return HealpixData(
            maps=_cast_map(healpix_data.maps),
            nside=healpix_data.nside,
            frequencies=np.asarray(healpix_data.frequencies, dtype=flux_dt),
            channel_widths_hz=healpix_data.channel_widths_hz,
            coordinate_frame=healpix_data.coordinate_frame,
            ordering=healpix_data.ordering,
            hpx_inds=healpix_data.hpx_inds,
            q_maps=_cast_map(healpix_data.q_maps),
            u_maps=_cast_map(healpix_data.u_maps),
            v_maps=_cast_map(healpix_data.v_maps),
            i_unit=healpix_data.i_unit,
            q_unit=healpix_data.q_unit,
            u_unit=healpix_data.u_unit,
            v_unit=healpix_data.v_unit,
            i_brightness_conversion=(
                healpix_data.i_brightness_conversion or str(brightness_conversion.value)
            ),
            q_brightness_conversion=healpix_data.q_brightness_conversion,
            u_brightness_conversion=healpix_data.u_brightness_conversion,
            v_brightness_conversion=healpix_data.v_brightness_conversion,
        )

    # =========================================================================
    # Precision Helpers
    # =========================================================================

    def _source_dtype(self) -> np.dtype:
        """Get dtype for source position arrays (RA/Dec)."""
        return self._precision.sky_model.get_dtype("source_positions")

    def _flux_dtype(self) -> np.dtype:
        """Get dtype for flux and Stokes arrays."""
        return self._precision.sky_model.get_dtype("flux")

    def _spectral_index_dtype(self) -> np.dtype:
        """Get dtype for spectral index arrays."""
        return self._precision.sky_model.get_dtype("spectral_index")

    def _healpix_dtype(self) -> np.dtype:
        """Get dtype for HEALPix brightness temperature maps."""
        return self._precision.sky_model.get_dtype("healpix_maps")

    @staticmethod
    def deg_to_rad_at_precision(
        arr: np.ndarray, precision: "PrecisionConfig | None"
    ) -> np.ndarray:
        """Convert degrees to radians at the precision config's dtype.

        Parameters
        ----------
        arr : np.ndarray
            Array of angles in degrees.
        precision : PrecisionConfig, optional
            Precision configuration object. If None, uses numpy default (float64).

        Returns
        -------
        np.ndarray
            Array of angles in radians, at the dtype set by precision config.
        """
        if precision is None:
            return np.deg2rad(arr)
        src_dt = precision.sky_model.get_dtype("source_positions")
        return np.deg2rad(arr.astype(src_dt, copy=False))

    @staticmethod
    def rad_to_deg_at_precision(
        arr: np.ndarray, precision: "PrecisionConfig | None"
    ) -> np.ndarray:
        """Convert radians to degrees at the precision config's dtype.

        Parameters
        ----------
        arr : np.ndarray
            Array of angles in radians.
        precision : PrecisionConfig, optional
            Precision configuration object. If None, uses numpy default.

        Returns
        -------
        np.ndarray
            Array of angles in degrees, at the dtype set by precision config.
        """
        if precision is None:
            return np.rad2deg(arr)
        src_dt = precision.sky_model.get_dtype("source_positions")
        return np.rad2deg(arr.astype(src_dt, copy=False))

    # =========================================================================
    # Immutable Replace Helper
    # =========================================================================

    def replace(self, **changes: Any) -> "SkyModel":
        """Return a new ``SkyModel`` with the given fields replaced.

        Wraps ``dataclasses.replace`` with precision-aware payload casting and
        a field whitelist. Always use this instead of ``dataclasses.replace``
        on a SkyModel — direct calls bypass dtype coercion.
        """
        import dataclasses

        precision = changes.pop("_precision", self._precision)
        if precision is None:
            raise ValueError(
                "SkyModel.replace(): _precision must not be None. "
                "This is a bug -- all factory methods should set precision."
            )

        field_changes: dict[str, Any] = {"_precision": precision}
        brightness_conversion = changes.get(
            "brightness_conversion",
            self.brightness_conversion,
        )
        if not isinstance(brightness_conversion, BrightnessConversion):
            brightness_conversion = BrightnessConversion(brightness_conversion)

        if "point" in changes:
            field_changes["point"] = self._cast_point_data(
                changes.pop("point"), precision
            )

        if "healpix" in changes:
            field_changes["healpix"] = self._cast_healpix_data(
                changes.pop("healpix"),
                precision,
                brightness_conversion,
            )

        for key in (
            "reference_frequency",
            "model_name",
            "brightness_conversion",
            "polarization_brightness_conversion",
            "provenance",
        ):
            if key in changes:
                field_changes[key] = changes.pop(key)

        if changes:
            unknown = ", ".join(sorted(changes))
            raise TypeError(
                f"SkyModel.replace() received unsupported fields: {unknown}"
            )

        return dataclasses.replace(self, **field_changes)

    # =========================================================================
    # Per-Source Field Helpers
    # =========================================================================

    def _masked_point_source_data(self, mask: np.ndarray) -> PointSourceData | None:
        """Return point-source payload with a boolean mask applied."""
        if self.point is None:
            return None
        return self.point.masked(mask)

    # =========================================================================
    # Properties
    # =========================================================================

    @property
    def has_multifreq_maps(self) -> bool:
        """Return True if multi-frequency HEALPix maps are available."""
        return self.healpix is not None

    @property
    def formats(self) -> set[SkyFormat]:
        """Return the set of representations populated on this model.

        Hybrid models (both point and healpix populated) return both members.
        """
        result: set[SkyFormat] = set()
        if self.point is not None:
            result.add(SkyFormat.POINT_SOURCES)
        if self.healpix is not None:
            result.add(SkyFormat.HEALPIX)
        return result

    @property
    def n_frequencies(self) -> int:
        """Return the number of frequency channels (0 if no multi-freq maps)."""
        if self.healpix is not None:
            return len(self.healpix.frequencies)
        return 0

    def n_sky_elements_for(self, representation: "SkyFormat | str") -> int:
        """Return the element count for an explicit representation (0 if absent)."""
        target = _coerce_format(representation)
        if target == SkyFormat.HEALPIX:
            return self.healpix.n_pixels if self.healpix is not None else 0
        return self.point.n_sources if self.point is not None else 0

    @property
    def n_point_sources(self) -> int:
        """Return the number of point-source catalog entries (0 if none)."""
        return self.point.n_sources if self.point is not None else 0

    @property
    def n_healpix_pixels(self) -> int:
        """Return the number of stored HEALPix pixels (0 if no healpix payload)."""
        return self.healpix.n_pixels if self.healpix is not None else 0

    @property
    def n_pixels(self) -> int:
        """Return the number of HEALPix pixels (0 if no maps)."""
        return self.healpix.n_pixels if self.healpix is not None else 0

    @property
    def has_polarized_healpix_maps(self) -> bool:
        """Return True if any polarization (Q/U/V) HEALPix maps are populated."""
        return self.healpix is not None and self.healpix.has_polarization

    @property
    def has_point_sources(self) -> bool:
        """Return True if columnar point-source arrays are populated and non-empty."""
        return self.point is not None and not self.point.is_empty

    @property
    def precision(self) -> "PrecisionConfig | None":
        """Precision configuration for this model."""
        return self._precision

    # =========================================================================
    # Region Filtering
    # =========================================================================

    def _masked_healpix_data(self, region: "SkyRegion") -> HealpixData | None:
        """Return region-cropped HEALPix payload (sparse, mask-only pixels stored)."""
        if self.healpix is None:
            return None
        hp_mask = region.healpix_mask(
            self.healpix.nside,
            coordinate_frame=self.healpix.coordinate_frame,
        )
        return self.healpix.cropped_to_mask(hp_mask)

    def filter_region(self, region: "SkyRegion") -> "SkyModel":
        """Return a new SkyModel containing only sources/pixels within *region*.

        For point-source data, applies a boolean mask to all columnar
        arrays.  For HEALPix data, crops to a sparse representation that
        stores only the in-region pixels (use ``healpix.to_dense()``
        afterwards if a full-grid array is required).
        When both representations are present, both are filtered.

        Does **not** mutate ``self`` -- always returns a new instance.

        Parameters
        ----------
        region : SkyRegion
            Sky region to filter to (cone, box, or union).

        Returns
        -------
        SkyModel
            Filtered copy.
        """
        point = self.point
        healpix = self.healpix
        coverage_footprint = self.provenance.coverage_footprint

        if self.healpix is not None:
            hp_mask = region.healpix_mask(
                self.healpix.nside,
                coordinate_frame=self.healpix.coordinate_frame,
            )
            healpix = self.healpix.cropped_to_mask(hp_mask)

        if self.point is not None:
            mask = region.contains(self.point.ra_rad, self.point.dec_rad)
            if not np.all(mask):
                point = self._masked_point_source_data(mask)

        if coverage_footprint is not None:
            coverage_footprint = coverage_footprint.intersect_mask(
                region.healpix_mask(
                    coverage_footprint.nside,
                    coordinate_frame=coverage_footprint.coordinate_frame,
                )
            )
        elif self.provenance.is_full_sky:
            coverage_footprint = region.footprint()

        coverage_fraction = (
            coverage_footprint.coverage_fraction
            if coverage_footprint is not None
            else None
        )

        provenance = replace(
            self.provenance,
            sky_coverage=SkyCoverage.PARTIAL_SKY,
            coverage_fraction=coverage_fraction,
            coverage_footprint=coverage_footprint,
            monopole_k=None,
        )

        return self.replace(
            point=point,
            healpix=healpix,
            model_name=self.model_name,
            reference_frequency=self.reference_frequency,
            brightness_conversion=self.brightness_conversion,
            provenance=provenance,
            _precision=self._precision,
        )

    # =========================================================================
    # Immutable Conversion Methods
    # =========================================================================

    def with_reference_frequency(self, reference_frequency: float) -> "SkyModel":
        """Return a new SkyModel with the reference frequency changed.

        Parameters
        ----------
        reference_frequency : float
            New reference frequency in Hz.

        Returns
        -------
        SkyModel
            Copy with updated ``reference_frequency``.
        """
        return self.replace(reference_frequency=reference_frequency)

    def as_point_source_arrays(
        self,
        flux_limit: float = 0.0,
    ) -> SourceArrays:
        """Get point-source arrays without performing implicit conversion."""
        if self.point is None:
            hint = (
                " Use radiosim.core.sky.materialize_point_sources_model("
                "sky, frequency=..., lossy=True) first."
                if self.healpix is not None
                else ""
            )
            raise ValueError(f"No point-source payload available.{hint}")

        return self.point.as_source_arrays(
            flux_limit=flux_limit,
            reference_frequency=float(self.reference_frequency or 0.0),
        )

    # =========================================================================
    # String Representation
    # =========================================================================

    def __repr__(self) -> str:
        """Return a human-readable summary of the sky model.

        Returns
        -------
        str
            Summary string including model name and populated formats.
        """
        parts: list[str] = [
            f"model='{self.model_name}'",
            f"formats={[fmt.value for fmt in sorted(self.formats, key=lambda f: f.value)]}",
        ]

        # Point-source info
        if self.point is not None and not self.point.is_empty:
            extras = []
            if self.point.rotation_measure is not None and np.any(
                self.point.rotation_measure != 0
            ):
                extras.append("RM")
            if self.point.major_arcsec is not None and np.any(
                self.point.major_arcsec > 0
            ):
                n_gauss = int(np.sum(self.point.major_arcsec > 0))
                extras.append(f"gaussian={n_gauss}")
            if (
                self.point.spectral_coeffs is not None
                and self.point.spectral_coeffs.shape[1] > 1
            ):
                extras.append(f"spectral_terms={self.point.spectral_coeffs.shape[1]}")
            extra_str = f", {', '.join(extras)}" if extras else ""
            parts.append(f"n_sources={self.n_point_sources}{extra_str}")

        # HEALPix info
        if self.healpix is not None:
            freqs = self.healpix.frequencies
            freq_range = (
                f"{freqs[0] / 1e6:.1f}-{freqs[-1] / 1e6:.1f}"
                if len(freqs) > 1
                else f"{freqs[0] / 1e6:.1f}"
            )
            stokes_components = "I"
            n_stokes = 1
            for arr, letter in [
                (self.healpix.q_maps, "Q"),
                (self.healpix.u_maps, "U"),
                (self.healpix.v_maps, "V"),
            ]:
                if arr is not None:
                    stokes_components += letter
                    n_stokes += 1
            stored_arrays = [
                arr
                for arr in (
                    self.healpix.maps,
                    self.healpix.q_maps,
                    self.healpix.u_maps,
                    self.healpix.v_maps,
                )
                if arr is not None
            ]
            total_bytes = sum(arr.nbytes for arr in stored_arrays)
            if self.healpix.hpx_inds is not None:
                total_bytes += self.healpix.hpx_inds.nbytes
            memory_mb = total_bytes / 1e6
            sparse_note = ", sparse=True" if self.healpix.is_sparse else ""
            parts.append(
                f"nside={self.healpix.nside}, n_freq={len(freqs)}, "
                f"pixels={self.healpix.n_pixels}{sparse_note}, "
                f"freq_range={freq_range}MHz, stokes='{stokes_components}', "
                f"frame='{self.healpix.coordinate_frame}', "
                f"memory={memory_mb:.1f}MB"
            )

        return f"SkyModel({', '.join(parts)})"

    # =========================================================================
    # Equality
    # =========================================================================

    # Disable auto-generated __hash__ since payloads carry numpy arrays.
    __hash__ = None  # type: ignore[assignment]

    def _scalar_fields_equal(self, other: "SkyModel") -> bool:
        return (
            self.model_name == other.model_name
            and self.reference_frequency == other.reference_frequency
            and self.brightness_conversion == other.brightness_conversion
            and self.polarization_brightness_conversion
            == other.polarization_brightness_conversion
        )

    def _payloads_compare(
        self, other: "SkyModel", *, close: bool, rtol: float, atol: float
    ) -> bool:
        if (self.point is None) != (other.point is None):
            return False
        if self.point is not None and other.point is not None:
            if not self.point._compare(other.point, close=close, rtol=rtol, atol=atol):
                return False
        if (self.healpix is None) != (other.healpix is None):
            return False
        if self.healpix is not None and other.healpix is not None:
            if not self.healpix._compare(
                other.healpix, close=close, rtol=rtol, atol=atol
            ):
                return False
        return True

    def __eq__(self, other: object) -> bool:
        """Bit-exact value equality.

        Compares every payload array via :func:`numpy.array_equal` (no
        tolerance) and every scalar field via ``==``.  Cross-precision
        comparisons (e.g. f32 vs f64) return ``False`` even when the only
        differences are representation error — use :meth:`is_close` for
        that case.
        """
        if not isinstance(other, SkyModel):
            return NotImplemented
        if not self._scalar_fields_equal(other):
            return False
        if self.provenance != other.provenance:
            return False
        return self._payloads_compare(other, close=False, rtol=0.0, atol=0.0)

    def is_close(
        self, other: "SkyModel", rtol: float = 1e-7, atol: float = 0.0
    ) -> bool:
        """Approximate equality (useful for round-trip and precision testing).

        Parameters
        ----------
        other : SkyModel
            Model to compare against.
        rtol : float, default 1e-7
            Relative tolerance for ``np.allclose``.
        atol : float, default 0.0
            Absolute tolerance for ``np.allclose``.

        Returns
        -------
        bool
        """
        if not isinstance(other, SkyModel):
            return False
        if not self._scalar_fields_equal(other):
            return False
        return self._payloads_compare(other, close=True, rtol=rtol, atol=atol)
