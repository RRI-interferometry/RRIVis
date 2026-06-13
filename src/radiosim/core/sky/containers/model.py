# radiosim/core/sky/containers/model.py
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
from dataclasses import field
from enum import Enum
from typing import TYPE_CHECKING, Any

import numpy as np
from pydantic import model_validator
from pydantic.dataclasses import dataclass

from radiosim.core.precision import PrecisionConfig

from ._shared import _FROZEN_NDARRAY_CONFIG
from .constants import BrightnessConversion
from .footprint import SkyCoverage
from .healpix import HealpixData
from .point import (
    PointMetadata,
    PointMorphology,
    PointPolarization,
    PointSourceData,
    PointSpectrum,
    SourceArrays,
)
from .provenance import SkyProvenance

if TYPE_CHECKING:
    from ..operations.region import SkyRegion

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


@dataclass(frozen=True, eq=False, config=_FROZEN_NDARRAY_CONFIG)
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
    # When True, materialize_healpix_model / materialize_point_sources_model
    # force Rayleigh-Jeans for both Stokes I and Q/U/V regardless of the
    # default per-component conventions. The default per-component setup
    # (Planck for I, RJ for Q/U/V because Q/U/V can be negative) inflates
    # fractional polarisation by ~5–15 % at ν ≲ 100 MHz on a HEALPix↔point
    # round-trip; coherent mode keeps that round-trip bit-exact at the
    # cost of using the linear-temperature approximation for I.
    coherent_brightness_conversion: bool = False
    provenance: SkyProvenance = field(default_factory=SkyProvenance)
    precision: PrecisionConfig | None = field(default=None, repr=False)

    @model_validator(mode="before")
    @classmethod
    def _coerce_inputs(cls, values: object) -> object:
        """Coerce input kwargs (string enums, dict provenance, dtype-cast payloads).

        Mirrors the :meth:`SkyProvenance._derive_inputs` pattern: we
        peek at the input dict / ``ArgsKwargs``, mutate it in place, and
        return the original handle.  Pydantic constructs the frozen
        instance from the transformed inputs — no ``object.__setattr__``
        gymnastics.
        """
        if isinstance(values, dict):
            kwargs = values
        elif hasattr(values, "kwargs") and isinstance(values.kwargs, dict):
            kwargs = values.kwargs
        else:
            return values

        bc = kwargs.get("brightness_conversion")
        if bc is not None and not isinstance(bc, BrightnessConversion):
            try:
                kwargs["brightness_conversion"] = BrightnessConversion(bc)
            except (ValueError, KeyError):
                raise ValueError(
                    f"brightness_conversion must be 'planck' or 'rayleigh-jeans', "
                    f"got '{bc}'"
                ) from None

        pbc = kwargs.get("polarization_brightness_conversion")
        if pbc is not None and not isinstance(pbc, BrightnessConversion):
            try:
                kwargs["polarization_brightness_conversion"] = BrightnessConversion(pbc)
            except (ValueError, KeyError):
                raise ValueError(
                    "polarization_brightness_conversion must be 'planck' or "
                    f"'rayleigh-jeans', got '{pbc}'"
                ) from None

        prov = kwargs.get("provenance")
        if isinstance(prov, dict):
            kwargs["provenance"] = SkyProvenance(**prov)
        elif prov is not None and not isinstance(prov, SkyProvenance):
            raise TypeError(
                "SkyModel.provenance must be a SkyProvenance or a dict of its "
                f"fields, got {type(prov).__name__}."
            )

        precision = kwargs.get("precision")
        if precision is not None:
            bc_for_cast = kwargs.get(
                "brightness_conversion", BrightnessConversion.PLANCK
            )
            point = kwargs.get("point")
            if point is not None:
                kwargs["point"] = cls._cast_point_data(point, precision)
            healpix = kwargs.get("healpix")
            if healpix is not None:
                kwargs["healpix"] = cls._cast_healpix_data(
                    healpix, precision, bc_for_cast
                )

        return values

    @model_validator(mode="after")
    def _validate_state(self) -> "SkyModel":
        """Cross-field invariants applied after construction."""
        if self.point is None and self.healpix is None:
            raise ValueError(
                "SkyModel requires at least one payload. "
                "Use create_empty() for an empty point-source model."
            )
        if self.precision is None:
            raise ValueError(
                "SkyModel requires an explicit PrecisionConfig. "
                "Pass precision=... to a loader or constructor."
            )
        return self

    def check(self, *, run_check_acceptability: bool = True) -> None:
        """Re-validate the model, aggregating every problem into one error.

        Construction already enforces structural invariants, but ``check()``
        gives a user a single explicit entry point — mirroring pyradiosky's
        ``check()`` and the project's pre-flight collector — that gathers all
        issues at once instead of failing on the first.  Structural problems
        (missing payload, shape/length mismatches) are always checked; value
        *acceptability* problems (non-finite or negative flux, non-positive
        reference frequencies in use) are checked when ``run_check_acceptability``
        is True.

        Raises
        ------
        ValueError
            If any problem is found; the message lists every problem.
        """
        problems: list[str] = []

        # Structural: re-run the constructor validators by rebuilding. With
        # frozen, read-only arrays this normally cannot fail, but it is the
        # authoritative structural gate.
        try:
            self.replace()
        except (ValueError, TypeError) as exc:
            problems.append(f"structural: {exc}")

        if run_check_acceptability:
            point = self.point
            if point is not None and not point.is_empty:
                if not np.all(np.isfinite(point.flux)):
                    problems.append("point.flux contains non-finite values")
                if np.any(point.flux < 0):
                    problems.append("point.flux contains negative values")
                for name in ("ra_rad", "dec_rad", "spectral_index"):
                    if not np.all(np.isfinite(getattr(point, name))):
                        problems.append(f"point.{name} contains non-finite values")
                if np.any(point.ref_freq < 0):
                    problems.append("point.ref_freq contains negative values")
            healpix = self.healpix
            if healpix is not None:
                if not np.all(np.isfinite(healpix.frequencies)):
                    problems.append("healpix.frequencies contains non-finite values")
                if np.any(healpix.frequencies <= 0):
                    problems.append("healpix.frequencies contains non-positive values")

        if problems:
            raise ValueError(
                "SkyModel.check() found "
                f"{len(problems)} problem(s):\n  - " + "\n  - ".join(problems)
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
        morphology = point.morphology
        polarization = point.polarization
        metadata = point.metadata
        return PointSourceData(
            ra_rad=np.asarray(point.ra_rad, dtype=src_dt),
            dec_rad=np.asarray(point.dec_rad, dtype=src_dt),
            flux=np.asarray(point.flux, dtype=flux_dt),
            spectral_index=np.asarray(point.spectral_index, dtype=si_dt),
            stokes_q=np.asarray(point.stokes_q, dtype=flux_dt),
            stokes_u=np.asarray(point.stokes_u, dtype=flux_dt),
            stokes_v=np.asarray(point.stokes_v, dtype=flux_dt),
            ref_freq=np.asarray(point.ref_freq, dtype=flux_dt),
            polarization=(
                None
                if polarization is None
                else PointPolarization(
                    rotation_measure=np.asarray(
                        polarization.rotation_measure,
                        dtype=flux_dt,
                    )
                )
            ),
            morphology=(
                None
                if morphology is None
                else PointMorphology(
                    major_arcsec=np.asarray(morphology.major_arcsec, dtype=src_dt),
                    minor_arcsec=np.asarray(morphology.minor_arcsec, dtype=src_dt),
                    pa_deg=np.asarray(morphology.pa_deg, dtype=src_dt),
                )
            ),
            spectral_coeffs=(
                None
                if point.spectral_coeffs is None
                else np.asarray(point.spectral_coeffs, dtype=si_dt)
            ),
            metadata=(
                None
                if metadata is None
                else PointMetadata(
                    source_name=(
                        None
                        if metadata.source_name is None
                        else np.asarray(metadata.source_name)
                    ),
                    source_id=(
                        None
                        if metadata.source_id is None
                        else np.asarray(metadata.source_id)
                    ),
                    extra_columns={
                        name: np.asarray(values)
                        for name, values in metadata.extra_columns.items()
                    },
                )
            ),
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

        return healpix_data.replace(
            maps=_cast_map(healpix_data.maps),
            frequencies=np.asarray(healpix_data.frequencies, dtype=flux_dt),
            q_maps=_cast_map(healpix_data.q_maps),
            u_maps=_cast_map(healpix_data.u_maps),
            v_maps=_cast_map(healpix_data.v_maps),
            i_brightness_conversion=(
                healpix_data.i_brightness_conversion or str(brightness_conversion.value)
            ),
        )

    # =========================================================================
    # Precision Helpers
    # =========================================================================

    def _source_dtype(self) -> np.dtype:
        """Get dtype for source position arrays (RA/Dec)."""
        return self.precision.sky_model.get_dtype("source_positions")

    def _flux_dtype(self) -> np.dtype:
        """Get dtype for flux and Stokes arrays."""
        return self.precision.sky_model.get_dtype("flux")

    def _spectral_index_dtype(self) -> np.dtype:
        """Get dtype for spectral index arrays."""
        return self.precision.sky_model.get_dtype("spectral_index")

    def _healpix_dtype(self) -> np.dtype:
        """Get dtype for HEALPix brightness temperature maps."""
        return self.precision.sky_model.get_dtype("healpix_maps")

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

    _REPLACE_FIELDS: tuple[str, ...] = (
        "point",
        "healpix",
        "reference_frequency",
        "model_name",
        "brightness_conversion",
        "polarization_brightness_conversion",
        "coherent_brightness_conversion",
        "provenance",
        "precision",
    )

    def replace(self, **changes: Any) -> "SkyModel":
        """Return a new ``SkyModel`` with the given fields replaced.

        Constructs a fresh instance via ``SkyModel(**data)`` so every
        pydantic validator (precision-aware dtype casting, payload-shape
        consistency, monopole/footprint invariants on provenance) re-runs.
        Always use this instead of ``dataclasses.replace`` — direct calls
        bypass the validators.
        """
        unknown = set(changes) - set(self._REPLACE_FIELDS)
        if unknown:
            raise TypeError(
                f"SkyModel.replace() received unsupported fields: {sorted(unknown)}"
            )

        precision = changes.get("precision", self.precision)
        if precision is None:
            raise ValueError(
                "SkyModel.replace(): precision must not be None. "
                "This is a bug -- all factory methods should set precision."
            )

        data: dict[str, Any] = {
            name: getattr(self, name) for name in self._REPLACE_FIELDS
        }
        data["precision"] = precision
        data.update(changes)

        # The constructor's _coerce_inputs validator casts point/healpix
        # payloads to ``precision`` exactly once. Pre-casting here as well would
        # rebuild the nested payloads twice on every .replace() — so we let the
        # constructor own the single cast.
        return SkyModel(**data)

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
    def has_polarized_healpix_maps(self) -> bool:
        """Return True if any polarization (Q/U/V) HEALPix maps are populated."""
        return self.healpix is not None and self.healpix.has_polarization

    @property
    def has_point_sources(self) -> bool:
        """Return True if columnar point-source arrays are populated and non-empty."""
        return self.point is not None and not self.point.is_empty

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
            nest=self.healpix.is_nested,
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
                nest=self.healpix.is_nested,
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

        provenance = self.provenance.replace(
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
            precision=self.precision,
        )

    # =========================================================================
    # Immutable Conversion Methods
    # =========================================================================

    def with_reference_frequency(self, reference_frequency: float) -> "SkyModel":
        """Return a new SkyModel re-anchored to a new reference frequency.

        For a point-source payload this **re-anchors the flux**: every Stokes
        array (I/Q/U/V) is rescaled by the power-law factor
        ``(new_freq / old_ref) ** spectral_index`` evaluated from each source's
        own reference frequency, and ``ref_freq`` is overwritten with
        ``new_freq``.  The result describes the same physical sky with its
        reference point moved — not merely a metadata relabel.

        Sources carrying log-polynomial ``spectral_coeffs`` cannot be
        re-anchored unambiguously (the coefficients are defined relative to the
        original reference frequency); a :class:`NotImplementedError` is raised
        for that case.

        For a HEALPix-only payload the maps carry their own frequency axis, so
        only the model-level ``reference_frequency`` metadata is updated.

        Parameters
        ----------
        reference_frequency : float
            New reference frequency in Hz.

        Returns
        -------
        SkyModel
            Re-anchored copy.
        """
        new_freq = float(reference_frequency)
        if new_freq <= 0:
            raise ValueError(
                f"reference_frequency must be positive, got {reference_frequency!r}."
            )

        point = self.point
        if point is None or point.is_empty:
            return self.replace(reference_frequency=new_freq)

        if point.spectral_coeffs is not None:
            raise NotImplementedError(
                "with_reference_frequency cannot re-anchor sources carrying "
                "log-polynomial spectral_coeffs: the coefficients are defined "
                "relative to the original reference frequency. Re-anchor "
                "power-law catalogs, or refit the polynomial to the new "
                "reference frequency first."
            )

        from .spectral import (
            compute_spectral_scale,
            per_source_reference_frequencies,
        )

        old_ref = per_source_reference_frequencies(
            point, model_reference_frequency=self.reference_frequency
        )
        if np.any(old_ref <= 0):
            raise ValueError(
                "with_reference_frequency requires a positive reference "
                "frequency for every source (set per-source ref_freq or the "
                "model reference_frequency before re-anchoring)."
            )

        scale = compute_spectral_scale(point.spectral_index, None, new_freq, old_ref)
        new_ref = np.full(point.n_sources, new_freq, dtype=point.ref_freq.dtype)
        new_point = PointSourceData(
            ra_rad=point.ra_rad,
            dec_rad=point.dec_rad,
            flux=point.flux * scale,
            spectral_index=point.spectral_index,
            stokes_q=point.stokes_q * scale,
            stokes_u=point.stokes_u * scale,
            stokes_v=point.stokes_v * scale,
            ref_freq=new_ref,
            spectral_coeffs=None,
            morphology=point.morphology,
            polarization=point.polarization,
            metadata=point.metadata,
            spectrum=point.spectrum,
        )
        return self.replace(point=new_point, reference_frequency=new_freq)

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
            polarization = self.point.polarization
            if polarization is not None and np.any(polarization.rotation_measure != 0):
                extras.append("RM")
            morphology = self.point.morphology
            if morphology is not None and np.any(morphology.major_arcsec > 0):
                n_gauss = int(np.sum(morphology.major_arcsec > 0))
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
