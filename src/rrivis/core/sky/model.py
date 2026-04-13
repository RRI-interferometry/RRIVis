# rrivis/core/sky/model.py
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
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, ClassVar

import healpy as hp
import numpy as np
from astropy.coordinates import SkyCoord

from ._data import HealpixData, PointSourceData, SourceArrays
from .constants import BrightnessConversion

if TYPE_CHECKING:
    from rrivis.core.precision import PrecisionConfig

    from .region import SkyRegion

logger = logging.getLogger(__name__)


# =============================================================================
# Sky Format Enum
# =============================================================================


class SkyFormat(Enum):
    """Sky model representation format."""

    POINT_SOURCES = "point_sources"
    HEALPIX = "healpix_map"


# =============================================================================
# SkyModel Class
# =============================================================================


@dataclass
class SkyModel:
    """Unified sky model built from typed payloads.

    ``source_format`` records the format the model was loaded from. It is
    provenance only; public operations choose representations explicitly.
    """

    point: PointSourceData | None = field(default=None, repr=False)
    healpix: HealpixData | None = field(default=None, repr=False)
    source_format: SkyFormat = SkyFormat.POINT_SOURCES
    reference_frequency: float | None = None
    model_name: str | None = None
    brightness_conversion: BrightnessConversion = BrightnessConversion.PLANCK
    _precision: "PrecisionConfig | None" = field(default=None, repr=False)

    def __post_init__(self) -> None:
        """Validate payload consistency and precision."""
        if self.point is None and self.healpix is None:
            raise ValueError(
                "SkyModel requires at least one payload. "
                "Use create_empty() for an empty point-source model."
            )

        if isinstance(self.source_format, str):
            self.source_format = self._coerce_representation(self.source_format)
        if not isinstance(self.source_format, SkyFormat):
            raise TypeError(
                f"source_format must be a SkyFormat enum, got "
                f"{type(self.source_format).__name__}: {self.source_format!r}."
            )

        if not isinstance(self.brightness_conversion, BrightnessConversion):
            try:
                self.brightness_conversion = BrightnessConversion(
                    self.brightness_conversion
                )
            except (ValueError, KeyError):
                raise ValueError(
                    f"brightness_conversion must be 'planck' or 'rayleigh-jeans', "
                    f"got '{self.brightness_conversion}'"
                ) from None

        if self.source_format == SkyFormat.POINT_SOURCES and self.point is None:
            raise ValueError("source_format='point_sources' requires a point payload.")
        if self.source_format == SkyFormat.HEALPIX and self.healpix is None:
            raise ValueError("source_format='healpix_map' requires a HEALPix payload.")

        if self._precision is None:
            raise ValueError(
                "SkyModel requires an explicit PrecisionConfig. "
                "Pass precision=... to a loader or constructor."
            )

        if self.point is not None:
            self.point = self._cast_point_data(self.point, self._precision)
        if self.healpix is not None:
            self.healpix = self._cast_healpix_data(
                self.healpix,
                self._precision,
                self.brightness_conversion,
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
            coordinate_frame=healpix_data.coordinate_frame,
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

    # Maps precision category names to the field names they govern.
    _PRECISION_CATEGORIES: ClassVar[dict[str, tuple[str, ...]]] = {
        "source_positions": (
            "ra_rad",
            "dec_rad",
            "major_arcsec",
            "minor_arcsec",
            "pa_deg",
        ),
        "flux": (
            "flux",
            "stokes_q",
            "stokes_u",
            "stokes_v",
            "rotation_measure",
            "ref_freq",
        ),
        "spectral_index": ("spectral_index", "spectral_coeffs"),
        "healpix_maps": (
            "maps",
            "q_maps",
            "u_maps",
            "v_maps",
        ),
    }

    # =========================================================================
    # Immutable Replace Helper
    # =========================================================================

    def _replace(self, **changes: Any) -> "SkyModel":
        """Return a new ``SkyModel`` with the given fields replaced."""
        import dataclasses

        precision = changes.pop("_precision", self._precision)
        if precision is None:
            raise ValueError(
                "SkyModel._replace(): _precision must not be None. "
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
            "source_format",
            "reference_frequency",
            "model_name",
            "brightness_conversion",
        ):
            if key in changes:
                field_changes[key] = changes.pop(key)

        if changes:
            unknown = ", ".join(sorted(changes))
            raise TypeError(
                f"SkyModel._replace() received unsupported fields: {unknown}"
            )

        return dataclasses.replace(self, **field_changes)

    # =========================================================================
    # Per-Source Field Helpers
    # =========================================================================

    # Tuple of all per-source 1-D array field names.
    _PER_SOURCE_FIELDS: tuple[str, ...] = (
        "ra_rad",
        "dec_rad",
        "flux",
        "spectral_index",
        "stokes_q",
        "stokes_u",
        "stokes_v",
        "ref_freq",
        "rotation_measure",
        "major_arcsec",
        "minor_arcsec",
        "pa_deg",
        "source_name",
        "source_id",
    )

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
    def available_formats(self) -> set[SkyFormat]:
        """Return the set of representations populated on this model."""
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

    @property
    def n_sky_elements(self) -> int:
        """Return count for single-format models.

        Models carrying both point and HEALPix payloads are ambiguous; use
        ``n_sky_elements_for(...)`` to state the requested representation.
        """
        if len(self.available_formats) > 1:
            raise ValueError(
                "n_sky_elements is ambiguous because both point and HEALPix "
                "payloads are present. Use n_sky_elements_for(representation)."
            )
        if self.healpix is not None:
            return self.healpix.n_pixels
        return self.point.n_sources if self.point is not None else 0

    def n_sky_elements_for(self, representation: "SkyFormat | str") -> int:
        """Return the count for an explicit representation."""
        target = self._coerce_representation(representation)
        if target == SkyFormat.HEALPIX:
            return self.healpix.n_pixels if self.healpix is not None else 0
        return self.point.n_sources if self.point is not None else 0

    @property
    def n_point_sources(self) -> int:
        """Return the number of point-source catalog entries (0 if none)."""
        return self.point.n_sources if self.point is not None else 0

    @property
    def n_pixels(self) -> int:
        """Return the number of HEALPix pixels (0 if no maps)."""
        return self.healpix.n_pixels if self.healpix is not None else 0

    @property
    def has_polarized_healpix_maps(self) -> bool:
        """Return True if any polarization (Q/U/V) HEALPix maps are populated."""
        return self.healpix is not None and self.healpix.has_polarization

    @property
    def pixel_solid_angle(self) -> float:
        """Solid angle per HEALPix pixel in steradians.

        Raises
        ------
        ValueError
            If no HEALPix maps are available.
        """
        if self.healpix is None:
            raise ValueError(
                "No HEALPix maps available. "
                "Load a HEALPix model or materialize one first."
            )
        return self.healpix.pixel_solid_angle

    @property
    def pixel_coords(self) -> SkyCoord:
        """SkyCoord of the stored HEALPix pixel centers in the stored frame.

        Returns
        -------
        SkyCoord
            Coordinates of the stored pixels.  For dense maps this is the
            full HEALPix grid; for sparse maps it is only the retained pixels.

        Raises
        ------
        ValueError
            If no HEALPix maps are available.
        """
        if self.healpix is None:
            raise ValueError(
                "No HEALPix maps available. "
                "Load a HEALPix model or materialize one first."
            )
        nside = self.healpix.nside
        pixel_indices = self.healpix.pixel_indices
        theta, phi = hp.pix2ang(nside, pixel_indices)
        lat_rad = np.pi / 2 - theta
        if self.healpix.coordinate_frame == "galactic":
            return SkyCoord(l=phi, b=lat_rad, unit="rad", frame="galactic")
        return SkyCoord(ra=phi, dec=lat_rad, unit="rad", frame="icrs")

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
        """Return region-masked HEALPix payload."""
        if self.healpix is None:
            return None
        hp_mask = region.healpix_mask(
            self.healpix.nside,
            coordinate_frame=self.healpix.coordinate_frame,
        )
        return self.healpix.masked_region(hp_mask)

    def filter_region(self, region: "SkyRegion") -> "SkyModel":
        """Return a new SkyModel containing only sources/pixels within *region*.

        For point-source data, applies a boolean mask to all columnar
        arrays.  For HEALPix data, sets out-of-region pixels to ``0.0``.
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

        if self.healpix is not None:
            healpix = self._masked_healpix_data(region)

        if self.point is not None:
            mask = region.contains(self.point.ra_rad, self.point.dec_rad)
            if not np.all(mask):
                point = self._masked_point_source_data(mask)

        if point is self.point and healpix is self.healpix:
            return self  # empty model — nothing to filter

        return self._replace(
            point=point,
            healpix=healpix,
            source_format=self.source_format,
            model_name=self.model_name,
            reference_frequency=self.reference_frequency,
            brightness_conversion=self.brightness_conversion,
            _precision=self._precision,
        )

    # =========================================================================
    # Helper Methods
    # =========================================================================

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
        return self._replace(reference_frequency=reference_frequency)

    @staticmethod
    def _coerce_representation(representation: "SkyFormat | str") -> SkyFormat:
        if isinstance(representation, str) and not isinstance(
            representation, SkyFormat
        ):
            try:
                return SkyFormat(representation)
            except ValueError:
                raise ValueError(
                    f"Unknown representation '{representation}'. "
                    f"Supported: SkyFormat.POINT_SOURCES, SkyFormat.HEALPIX."
                ) from None
        return representation

    def as_point_source_arrays(
        self,
        flux_limit: float = 0.0,
    ) -> SourceArrays:
        """Get point-source arrays without performing implicit conversion."""
        if self.point is None:
            hint = (
                " Use rrivis.core.sky.materialize_point_sources_model("
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
    # HEALPix Map Accessors (array-indexed)
    # =========================================================================

    def resolve_frequency_index(self, frequency: float) -> int:
        """Resolve a frequency to the nearest HEALPix frequency index.

        Returns the exact index if present, or the nearest one (with a warning
        if the difference exceeds a channel-width-scaled threshold).

        Parameters
        ----------
        frequency : float
            Frequency in Hz.

        Returns
        -------
        int
            Index into ``healpix.frequencies``.
        """
        if self.healpix is None:
            raise ValueError("No observation frequencies available.")
        freqs = self.healpix.frequencies
        idx = int(np.argmin(np.abs(freqs - frequency)))
        nearest_freq = float(freqs[idx])
        diff_hz = abs(frequency - nearest_freq)
        if len(freqs) > 1:
            spacing_hz = float(np.median(np.diff(np.sort(freqs))))
            warn_threshold_hz = max(1_000.0, 0.1 * spacing_hz)
        else:
            warn_threshold_hz = 1_000.0
        if diff_hz > warn_threshold_hz:
            logger.warning(
                f"Exact frequency {frequency / 1e6:.3f} MHz not found. "
                f"Using nearest: {nearest_freq / 1e6:.3f} MHz "
                f"(diff: {diff_hz / 1e6:.3f} MHz)"
            )
        return idx

    def get_map_at_frequency(self, frequency: float) -> np.ndarray:
        """
        Get HEALPix brightness temperature map at a specific frequency.

        Parameters
        ----------
        frequency : float
            Frequency in Hz.

        Returns
        -------
        temp_map : np.ndarray
            Brightness temperature map in Kelvin.

        Raises
        ------
        ValueError
            If no multi-frequency maps are available or frequency not found.

        Notes
        -----
        If the exact frequency is not available, returns the map at the
        nearest available frequency with a warning.

        Examples
        --------
        >>> from rrivis.core.sky import materialize_healpix_model
        >>> sky = materialize_healpix_model(sky, nside=64, frequencies=freqs)
        >>> map_100mhz = sky.get_map_at_frequency(100e6)
        """
        if self.healpix is None:
            raise ValueError(
                "No multi-frequency HEALPix maps available. "
                "Materialize a HEALPix payload first."
            )
        idx = self.resolve_frequency_index(frequency)
        return self.healpix.maps[idx]

    def get_multifreq_maps(self) -> tuple[np.ndarray, int, np.ndarray]:
        """
        Get all multi-frequency HEALPix maps.

        Returns
        -------
        temp_maps : np.ndarray
            Stokes I brightness temperature maps, shape ``(n_freq, npix)``.
        nside : int
            HEALPix NSIDE parameter.
        frequencies : np.ndarray
            Array of frequencies in Hz.

        Raises
        ------
        ValueError
            If no multi-frequency maps are available.

        Examples
        --------
        >>> maps, nside, freqs = sky.get_multifreq_maps()
        >>> for i, freq in enumerate(freqs):
        ...     print(f"{freq / 1e6:.1f} MHz: max T_b = {maps[i].max():.2f} K")
        """
        if self.healpix is None:
            raise ValueError(
                "No multi-frequency HEALPix maps available. "
                "Materialize a HEALPix payload first."
            )

        return (self.healpix.maps, self.healpix.nside, self.healpix.frequencies)

    def get_stokes_maps_at_frequency(
        self, frequency: float
    ) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None, np.ndarray | None]:
        """Get Stokes I/Q/U/V HEALPix maps at a specific frequency.

        Parameters
        ----------
        frequency : float
            Frequency in Hz.

        Returns
        -------
        I_map : np.ndarray
            Stokes I brightness temperature map (K_RJ).
        Q_map : np.ndarray or None
            Stokes Q map (K_RJ), or None if not available.
        U_map : np.ndarray or None
            Stokes U map (K_RJ), or None if not available.
        V_map : np.ndarray or None
            Stokes V map (K_RJ), or None if not available.

        Raises
        ------
        ValueError
            If no multi-frequency HEALPix maps are available.
        """
        if self.healpix is None:
            raise ValueError(
                "No multi-frequency HEALPix maps available. "
                "Materialize a HEALPix payload first."
            )
        idx = self.resolve_frequency_index(frequency)
        I_map = self.healpix.maps[idx]
        Q_map = self.healpix.q_maps[idx] if self.healpix.q_maps is not None else None
        U_map = self.healpix.u_maps[idx] if self.healpix.u_maps is not None else None
        V_map = self.healpix.v_maps[idx] if self.healpix.v_maps is not None else None
        return I_map, Q_map, U_map, V_map

    def get_multifreq_stokes_maps(
        self,
    ) -> tuple[
        np.ndarray,
        np.ndarray | None,
        np.ndarray | None,
        np.ndarray | None,
        int,
        np.ndarray,
    ]:
        """Get all multi-frequency Stokes I/Q/U/V HEALPix maps.

        Returns
        -------
        I_maps : np.ndarray
            Stokes I maps, shape ``(n_freq, npix)``.
        Q_maps : np.ndarray or None
            Stokes Q maps, shape ``(n_freq, npix)``, or None.
        U_maps : np.ndarray or None
            Stokes U maps, shape ``(n_freq, npix)``, or None.
        V_maps : np.ndarray or None
            Stokes V maps, shape ``(n_freq, npix)``, or None.
        nside : int
            HEALPix NSIDE parameter.
        frequencies : np.ndarray
            Array of frequencies in Hz.

        Raises
        ------
        ValueError
            If no multi-frequency maps are available.
        """
        if self.healpix is None:
            raise ValueError(
                "No multi-frequency HEALPix maps available. "
                "Materialize a HEALPix payload first."
            )
        return (
            self.healpix.maps,
            self.healpix.q_maps,
            self.healpix.u_maps,
            self.healpix.v_maps,
            self.healpix.nside,
            self.healpix.frequencies,
        )

    # =========================================================================
    # Frequency Iteration & Memory Management
    # =========================================================================

    def iter_frequency_maps(
        self,
    ):
        """Yield ``(freq_hz, I_map, Q_map, U_map, V_map)`` one channel at a time.

        Useful for memory-efficient processing when the full
        ``(n_freq, npix)`` cube is not needed simultaneously.  The
        visibility engine processes one frequency at a time, so this is
        the natural access pattern.

        Yields
        ------
        freq_hz : float
            Frequency in Hz.
        I_map : np.ndarray
            Stokes I brightness temperature map, shape ``(npix,)``.
        Q_map : np.ndarray or None
            Stokes Q map (K_RJ), or None.
        U_map : np.ndarray or None
            Stokes U map (K_RJ), or None.
        V_map : np.ndarray or None
            Stokes V map (K_RJ), or None.

        Raises
        ------
        ValueError
            If no HEALPix maps are available.
        """
        if self.healpix is None:
            raise ValueError(
                "No multi-frequency HEALPix maps available. "
                "Materialize a HEALPix payload first."
            )
        for i, freq in enumerate(self.healpix.frequencies):
            s_i = self.healpix.maps[i]
            s_q = self.healpix.q_maps[i] if self.healpix.q_maps is not None else None
            s_u = self.healpix.u_maps[i] if self.healpix.u_maps is not None else None
            s_v = self.healpix.v_maps[i] if self.healpix.v_maps is not None else None
            yield float(freq), s_i, s_q, s_u, s_v

    # =========================================================================
    # Serialization (delegates to _serialization.py)
    # =========================================================================
    # String Representation
    # =========================================================================

    def __repr__(self) -> str:
        """Return a human-readable summary of the sky model.

        Returns
        -------
        str
            Summary string including source format, model name, and available formats.
        """
        parts: list[str] = [
            f"source_format='{self.source_format.value}'",
            f"model='{self.model_name}'",
            f"available={[fmt.value for fmt in sorted(self.available_formats, key=lambda f: f.value)]}",
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

    # Disable auto-generated __hash__ since we define __eq__ with numpy arrays.
    __hash__ = None  # type: ignore[assignment]

    def __eq__(self, other: object) -> bool:
        """Value equality: compare all scalar and array fields.

        Uses ``np.array_equal`` for array comparisons (exact, handles
        None and shape/dtype mismatches).
        """
        if not isinstance(other, SkyModel):
            return NotImplemented

        # Scalar fields
        if (
            self.model_name != other.model_name
            or self.reference_frequency != other.reference_frequency
            or self.brightness_conversion != other.brightness_conversion
            or self.source_format != other.source_format
        ):
            return False

        return self._payloads_equal(other, close=False, rtol=0.0, atol=0.0)

    def _payloads_equal(
        self,
        other: "SkyModel",
        *,
        close: bool,
        rtol: float,
        atol: float,
    ) -> bool:
        """Compare point and HEALPix payload arrays."""

        def _arrays_equal(a: np.ndarray | None, b: np.ndarray | None) -> bool:
            if a is None and b is None:
                return True
            if a is None or b is None:
                return False
            if close:
                return bool(np.allclose(a, b, rtol=rtol, atol=atol, equal_nan=True))
            return bool(np.array_equal(a, b))

        if (self.point is None) != (other.point is None):
            return False
        if self.point is not None and other.point is not None:
            point_fields = (
                *PointSourceData._CORE_FIELDS,
                *PointSourceData._OPTIONAL_FIELDS,
                *PointSourceData._METADATA_FIELDS,
                "spectral_coeffs",
            )
            for name in point_fields:
                a = getattr(self.point, name)
                b = getattr(other.point, name)
                if not _arrays_equal(a, b):
                    return False
            if self.point.extra_columns.keys() != other.point.extra_columns.keys():
                return False
            for name in sorted(self.point.extra_columns):
                if not _arrays_equal(
                    self.point.extra_columns[name],
                    other.point.extra_columns[name],
                ):
                    return False

        if (self.healpix is None) != (other.healpix is None):
            return False
        if self.healpix is not None and other.healpix is not None:
            if self.healpix.nside != other.healpix.nside:
                return False
            if self.healpix.coordinate_frame != other.healpix.coordinate_frame:
                return False
            for name in (
                "i_unit",
                "q_unit",
                "u_unit",
                "v_unit",
                "i_brightness_conversion",
                "q_brightness_conversion",
                "u_brightness_conversion",
                "v_brightness_conversion",
            ):
                if getattr(self.healpix, name) != getattr(other.healpix, name):
                    return False
            for name in (
                "maps",
                "frequencies",
                "hpx_inds",
                "q_maps",
                "u_maps",
                "v_maps",
            ):
                if not _arrays_equal(
                    getattr(self.healpix, name), getattr(other.healpix, name)
                ):
                    return False

        return True

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

        if (
            self.model_name != other.model_name
            or self.reference_frequency != other.reference_frequency
            or self.brightness_conversion != other.brightness_conversion
            or self.source_format != other.source_format
        ):
            return False

        return self._payloads_equal(other, close=True, rtol=rtol, atol=atol)
