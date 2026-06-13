"""SkyProvenance — physical-correctness metadata for a SkyModel.

Carries flux completeness, angular resolution, sky coverage (full/partial,
sparse footprint), monopole convention, source-subtraction status — all
the information :func:`prepare_sky_model` needs to verify that two inputs
are physically disjoint before summing them.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from pydantic import field_validator, model_validator
from pydantic.dataclasses import dataclass

from ._shared import _FROZEN_NDARRAY_CONFIG
from .footprint import (
    DEFAULT_COVERAGE_FOOTPRINT_COORDINATE_FRAME,
    MonopoleConvention,
    SkyCoverage,
    SkyFootprint,
    SourceSubtractionStatus,
)


@dataclass(frozen=True, config=_FROZEN_NDARRAY_CONFIG)
class SkyProvenance:
    """Physical-correctness metadata attached to a :class:`SkyModel`.

    Used by ``prepare_sky_model`` to verify that models being summed are
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
    rng_seed: int | None = None
    """Seed of the random generator used to draw a stochastic payload
    (e.g. Poisson confusion). Recorded so a realization is reproducible from
    its own metadata. ``None`` for deterministic models."""

    @field_validator("monopole_convention", mode="before")
    @classmethod
    def _coerce_monopole_convention(cls, v: object) -> MonopoleConvention:
        return v if isinstance(v, MonopoleConvention) else MonopoleConvention(v)

    @field_validator("sky_coverage", mode="before")
    @classmethod
    def _coerce_sky_coverage(cls, v: object) -> SkyCoverage:
        return v if isinstance(v, SkyCoverage) else SkyCoverage(v)

    @field_validator("source_subtraction", mode="before")
    @classmethod
    def _coerce_source_subtraction(cls, v: object) -> SourceSubtractionStatus:
        return (
            v if isinstance(v, SourceSubtractionStatus) else SourceSubtractionStatus(v)
        )

    @field_validator("coverage_footprint", mode="before")
    @classmethod
    def _coerce_coverage_footprint(cls, v: object) -> SkyFootprint | None:
        if v is None or isinstance(v, SkyFootprint):
            return v
        if isinstance(v, dict):
            return SkyFootprint(**v)
        raise TypeError(
            "SkyProvenance.coverage_footprint must be a SkyFootprint, a dict "
            f"of its fields, or None; got {type(v).__name__}."
        )

    @field_validator("coverage_fraction", mode="before")
    @classmethod
    def _coerce_coverage_fraction(cls, v: object) -> float | None:
        if v is None:
            return None
        coverage_fraction = float(v)  # type: ignore[arg-type]
        if not np.isfinite(coverage_fraction) or not (0.0 <= coverage_fraction <= 1.0):
            raise ValueError(
                f"SkyProvenance: coverage_fraction must lie in [0, 1], got {v!r}."
            )
        return coverage_fraction

    @model_validator(mode="before")
    @classmethod
    def _derive_inputs(cls, values: object) -> object:
        """Cross-field input derivation.

        Pydantic dataclasses pass an ``ArgsKwargs`` object here; BaseModel
        would pass a dict.  We handle both so this validator is portable.
        Three derivations happen, all on the input kwargs (no
        ``object.__setattr__`` needed because we transform values before
        construction):

        1. ``coverage_footprint`` accepts a dict and is coerced to
           :class:`SkyFootprint`.
        2. When a footprint is supplied, ``sky_coverage`` is derived from
           ``footprint.is_full_sky`` (or validated for consistency) and
           ``coverage_fraction`` is filled from ``footprint.coverage_fraction``.
        3. When ``sky_coverage=FULL_SKY`` is supplied without an explicit
           ``coverage_fraction``, the fraction is filled to 1.0.
        """
        kwargs: dict[str, Any]
        if isinstance(values, dict):
            kwargs = values
        elif hasattr(values, "kwargs") and isinstance(values.kwargs, dict):
            kwargs = values.kwargs
        else:
            return values
        footprint = kwargs.get("coverage_footprint")
        if isinstance(footprint, dict):
            footprint = SkyFootprint(**footprint)
            kwargs["coverage_footprint"] = footprint
        if isinstance(footprint, SkyFootprint):
            footprint_coverage = (
                SkyCoverage.FULL_SKY
                if footprint.is_full_sky
                else SkyCoverage.PARTIAL_SKY
            )
            sky_coverage_raw = kwargs.get("sky_coverage", SkyCoverage.UNKNOWN)
            sky_coverage = (
                sky_coverage_raw
                if isinstance(sky_coverage_raw, SkyCoverage)
                else SkyCoverage(sky_coverage_raw)
            )
            if sky_coverage == SkyCoverage.UNKNOWN:
                kwargs["sky_coverage"] = footprint_coverage
            elif sky_coverage != footprint_coverage:
                raise ValueError(
                    "SkyProvenance: coverage_footprint implies "
                    f"sky_coverage={footprint_coverage.value!r}, got "
                    f"{sky_coverage.value!r}."
                )
            cf_raw = kwargs.get("coverage_fraction")
            if cf_raw is None:
                kwargs["coverage_fraction"] = footprint.coverage_fraction
            elif not np.isclose(float(cf_raw), footprint.coverage_fraction):
                raise ValueError(
                    "SkyProvenance: coverage_fraction is inconsistent with "
                    "coverage_footprint."
                )

        # FULL_SKY without an explicit coverage_fraction => derive 1.0.
        sky_coverage_raw = kwargs.get("sky_coverage")
        if sky_coverage_raw is not None and kwargs.get("coverage_fraction") is None:
            sky_coverage = (
                sky_coverage_raw
                if isinstance(sky_coverage_raw, SkyCoverage)
                else SkyCoverage(sky_coverage_raw)
            )
            if sky_coverage == SkyCoverage.FULL_SKY:
                kwargs["coverage_fraction"] = 1.0

        return values

    @model_validator(mode="after")
    def _validate_consistency(self) -> SkyProvenance:
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

        if self.sky_coverage == SkyCoverage.FULL_SKY:
            if self.coverage_fraction is None:
                raise ValueError(
                    "SkyProvenance: sky_coverage=FULL_SKY requires "
                    "coverage_fraction=1.0 (it should have been auto-derived)."
                )
            if not np.isclose(self.coverage_fraction, 1.0):
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
        return self

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

    # -------------------------------------------------------------------------
    # Validated replacement helper.
    # -------------------------------------------------------------------------
    #
    # Always derive new SkyProvenance instances via this method (or the
    # constructor).  Never use ``dataclasses.replace`` on a SkyProvenance --
    # that bypasses ``_derive_inputs`` and ``_validate_consistency`` and can
    # produce states the validators were written to forbid (e.g. partial-sky
    # with monopole_k set, or footprint inconsistent with sky_coverage).

    def replace(self, **changes: Any) -> SkyProvenance:
        """Return a new ``SkyProvenance`` with the given fields replaced.

        Re-runs every Pydantic validator so the result satisfies the same
        invariants as a freshly constructed instance.

        Special-case: when ``coverage_footprint`` is changed to a non-None
        footprint and the caller does not also pass ``sky_coverage`` /
        ``coverage_fraction``, the old derived values are dropped so the
        constructor can re-derive them from the new footprint.  Pass either
        explicitly to override that behaviour.
        """
        import dataclasses as _dc

        field_names = {field.name for field in _dc.fields(self)}
        unknown = set(changes) - field_names
        if unknown:
            raise TypeError(
                "SkyProvenance.replace() received unsupported fields: "
                f"{sorted(unknown)}"
            )

        data: dict[str, Any] = {name: getattr(self, name) for name in field_names}

        new_footprint = changes.get("coverage_footprint", data["coverage_footprint"])
        footprint_swapped = (
            "coverage_footprint" in changes and new_footprint is not None
        )
        if footprint_swapped:
            if "sky_coverage" not in changes:
                data["sky_coverage"] = SkyCoverage.UNKNOWN
            if "coverage_fraction" not in changes:
                data["coverage_fraction"] = None

        data.update(changes)
        return SkyProvenance(**data)

    # -------------------------------------------------------------------------
    # JSON-friendly serialization (used by skyh5 round-trip).
    # -------------------------------------------------------------------------

    def to_dict(self) -> dict:
        """Return a JSON-serializable dict representation.

        Enums are encoded as their string ``.value``.  ``coverage_footprint``
        is encoded as ``{"nside", "coordinate_frame", "hpx_inds"}`` with
        ``hpx_inds`` as a list of ints.  ``inf`` and ``-inf`` are represented
        as the JSON-compatible strings ``"inf"`` / ``"-inf"`` so that
        round-trips through strict JSON parsers do not silently coerce them
        to a finite number.
        """

        def _encode_float(x: float | None) -> float | str | None:
            if x is None:
                return None
            if np.isposinf(x):
                return "inf"
            if np.isneginf(x):
                return "-inf"
            return float(x)

        def _encode_pair(
            pair: tuple[float, float] | None,
        ) -> list[float | str] | None:
            if pair is None:
                return None
            return [_encode_float(pair[0]), _encode_float(pair[1])]

        footprint = None
        if self.coverage_footprint is not None:
            footprint = {
                "nside": int(self.coverage_footprint.nside),
                "coordinate_frame": self.coverage_footprint.coordinate_frame,
                "hpx_inds": self.coverage_footprint.hpx_inds.astype(int).tolist(),
            }

        return {
            "flux_completeness_jy": _encode_pair(self.flux_completeness_jy),
            "flux_completeness_freq_hz": _encode_float(self.flux_completeness_freq_hz),
            "angular_resolution_rad": _encode_pair(self.angular_resolution_rad),
            "sky_coverage": self.sky_coverage.value,
            "coverage_fraction": _encode_float(self.coverage_fraction),
            "coverage_footprint": footprint,
            "monopole_convention": self.monopole_convention.value,
            "monopole_k": _encode_float(self.monopole_k),
            "source_subtraction": self.source_subtraction.value,
            "source_subtraction_threshold_jy": _encode_float(
                self.source_subtraction_threshold_jy
            ),
            "source_subtraction_freq_hz": _encode_float(
                self.source_subtraction_freq_hz
            ),
            "source_subtraction_method": self.source_subtraction_method,
            "notes": self.notes,
            "rng_seed": self.rng_seed,
        }

    @classmethod
    def from_dict(cls, data: dict) -> SkyProvenance:
        """Build a :class:`SkyProvenance` from the dict produced by ``to_dict``.

        Tolerates missing keys (defaults are used) for forward-compatibility.
        """

        def _decode_float(x: object) -> float | None:
            if x is None:
                return None
            if isinstance(x, str):
                if x == "inf":
                    return float("inf")
                if x == "-inf":
                    return float("-inf")
                return float(x)
            return float(x)

        def _decode_pair(x: object) -> tuple[float, float] | None:
            if x is None:
                return None
            if not isinstance(x, list | tuple) or len(x) != 2:
                raise ValueError(
                    f"SkyProvenance.from_dict: expected a 2-tuple/list, got {x!r}."
                )
            a = _decode_float(x[0])
            b = _decode_float(x[1])
            if a is None or b is None:
                raise ValueError(
                    "SkyProvenance.from_dict: pair entries must not be None."
                )
            return (a, b)

        footprint_raw = data.get("coverage_footprint")
        footprint: SkyFootprint | None = None
        if footprint_raw is not None:
            footprint = SkyFootprint(
                nside=int(footprint_raw["nside"]),
                hpx_inds=np.asarray(footprint_raw["hpx_inds"], dtype=np.int64),
                coordinate_frame=footprint_raw.get(
                    "coordinate_frame", DEFAULT_COVERAGE_FOOTPRINT_COORDINATE_FRAME
                ),
            )

        return cls(
            flux_completeness_jy=_decode_pair(data.get("flux_completeness_jy")),
            flux_completeness_freq_hz=_decode_float(
                data.get("flux_completeness_freq_hz")
            ),
            angular_resolution_rad=_decode_pair(data.get("angular_resolution_rad")),
            sky_coverage=SkyCoverage(
                data.get("sky_coverage", SkyCoverage.UNKNOWN.value)
            ),
            coverage_fraction=_decode_float(data.get("coverage_fraction")),
            coverage_footprint=footprint,
            monopole_convention=MonopoleConvention(
                data.get("monopole_convention", MonopoleConvention.UNKNOWN.value)
            ),
            monopole_k=_decode_float(data.get("monopole_k")),
            source_subtraction=SourceSubtractionStatus(
                data.get("source_subtraction", SourceSubtractionStatus.UNKNOWN.value)
            ),
            source_subtraction_threshold_jy=_decode_float(
                data.get("source_subtraction_threshold_jy")
            ),
            source_subtraction_freq_hz=_decode_float(
                data.get("source_subtraction_freq_hz")
            ),
            source_subtraction_method=data.get("source_subtraction_method"),
            notes=data.get("notes"),
            rng_seed=data.get("rng_seed"),
        )
