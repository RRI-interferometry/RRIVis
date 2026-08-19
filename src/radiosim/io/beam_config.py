"""Strict, frozen input models for Tier 3 beam configuration.

This module owns only user-authored beam shape.  It performs no path
resolution, file access, assignment resolution, dependency import, or runtime
activation.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import BeforeValidator, Field, field_validator, model_validator
from pydantic_core import PydanticCustomError

from radiosim.io.instrument_config import (
    AntennaNameReference,
    AntennaNumberReference,
    AntennaReference,
)
from radiosim.io.model_base import StrictFrozenModel

_ENVIRONMENT_PATH = re.compile(r"\$(?:\{[^}]+\}|[A-Za-z_][A-Za-z0-9_]*)")

_StrictPositiveFiniteFloat = Annotated[
    float,
    Field(strict=True, gt=0.0, allow_inf_nan=False),
]
_StrictNonNegativeFiniteFloat = Annotated[
    float,
    Field(strict=True, ge=0.0, allow_inf_nan=False),
]


def _reject_bool_and_int(value: Any) -> Any:
    """Reject ``bool`` and ``int`` where an exact finite float is required.

    ``docs/development/sci005_beam_physics_plan.md`` Section 3.5 records the
    mechanic: Section 2's rule that "integers are not silently accepted as
    strict floats" is *not* delivered by strict Pydantic floats, which accept a
    Python ``int`` as a lossless widening.  Bools are already rejected by strict
    float validation; rejecting both here, uniformly, keeps every Stage-1 float
    field reporting Pydantic's own ``float_type`` issue code.
    """
    if isinstance(value, (bool, int)):
        raise PydanticCustomError("float_type", "Input should be a valid number")
    return value


_Stage1Float = Annotated[
    float,
    BeforeValidator(_reject_bool_and_int),
    Field(strict=True, allow_inf_nan=False),
]
_StrictExactInt = Annotated[int, Field(strict=True)]


def _validate_beam_path(value: Any) -> Any:
    """Validate authored path syntax without touching the filesystem."""
    if not isinstance(value, (str, Path)):
        raise ValueError("path must be a string or Path")
    raw = str(value)
    if not raw.strip() or raw == ".":
        raise ValueError("path must be a nonempty path")
    if _ENVIRONMENT_PATH.search(raw):
        raise ValueError("environment-variable syntax is not allowed in path")
    return value


class _BeamInputModel(StrictFrozenModel):
    """Private final-class boundary for canonical beam input values."""

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if any(
            base is not _BeamInputModel and issubclass(base, _BeamInputModel)
            for base in cls.__bases__
        ):
            raise TypeError("beam input models do not support subclassing")


class UniformTaperConfig(_BeamInputModel):
    """Uniform direct circular-aperture illumination."""

    kind: Literal["uniform"] = "uniform"


class GaussianTaperConfig(_BeamInputModel):
    """Gaussian direct circular-aperture illumination."""

    kind: Literal["gaussian"] = "gaussian"
    edge_taper_db: _StrictNonNegativeFiniteFloat = 10.0


class ParabolicTaperConfig(_BeamInputModel):
    """Parabolic direct circular-aperture illumination."""

    kind: Literal["parabolic"] = "parabolic"
    edge_taper_db: _StrictNonNegativeFiniteFloat = 10.0


class ParabolicSquaredTaperConfig(_BeamInputModel):
    """Parabolic-squared direct circular-aperture illumination."""

    kind: Literal["parabolic_squared"] = "parabolic_squared"
    edge_taper_db: _StrictNonNegativeFiniteFloat = 10.0


class CosineTaperConfig(_BeamInputModel):
    """Cosine direct circular-aperture illumination."""

    kind: Literal["cosine"] = "cosine"


DirectTaperConfig = Annotated[
    UniformTaperConfig
    | GaussianTaperConfig
    | ParabolicTaperConfig
    | ParabolicSquaredTaperConfig
    | CosineTaperConfig,
    Field(discriminator="kind"),
]


class DerivedGaussianTaperConfig(_BeamInputModel):
    """Gaussian profile whose edge taper is derived from illumination."""

    kind: Literal["gaussian"] = "gaussian"


class DerivedParabolicTaperConfig(_BeamInputModel):
    """Parabolic profile whose edge taper is derived from illumination."""

    kind: Literal["parabolic"] = "parabolic"


class DerivedParabolicSquaredTaperConfig(_BeamInputModel):
    """Parabolic-squared profile with a derived edge taper."""

    kind: Literal["parabolic_squared"] = "parabolic_squared"


FeedDerivedTaperConfig = Annotated[
    DerivedGaussianTaperConfig
    | DerivedParabolicTaperConfig
    | DerivedParabolicSquaredTaperConfig,
    Field(discriminator="kind"),
]


class CorrugatedHornIlluminationConfig(_BeamInputModel):
    """Corrugated-horn illumination parameters."""

    kind: Literal["corrugated_horn"] = "corrugated_horn"
    focal_ratio: _StrictPositiveFiniteFloat = 0.4
    q: _StrictPositiveFiniteFloat = 1.15


class OpenWaveguideIlluminationConfig(_BeamInputModel):
    """Open-waveguide illumination parameters."""

    kind: Literal["open_waveguide"] = "open_waveguide"
    focal_ratio: _StrictPositiveFiniteFloat = 0.4
    b_over_lambda: _StrictPositiveFiniteFloat = 0.7


class DipoleGroundPlaneIlluminationConfig(_BeamInputModel):
    """Dipole-over-ground-plane illumination parameters."""

    kind: Literal["dipole_ground_plane"] = "dipole_ground_plane"
    focal_ratio: _StrictPositiveFiniteFloat = 0.4
    height_wavelengths: _StrictPositiveFiniteFloat = 0.25


IlluminationConfig = Annotated[
    CorrugatedHornIlluminationConfig
    | OpenWaveguideIlluminationConfig
    | DipoleGroundPlaneIlluminationConfig,
    Field(discriminator="kind"),
]


class PrimeFocusReflectorConfig(_BeamInputModel):
    """Prime-focus reflector selection."""

    kind: Literal["prime_focus"] = "prime_focus"


class CassegrainReflectorConfig(_BeamInputModel):
    """Cassegrain reflector parameters."""

    kind: Literal["cassegrain"] = "cassegrain"
    magnification: Annotated[
        float,
        Field(strict=True, gt=1.0, allow_inf_nan=False),
    ]


ReflectorConfig = Annotated[
    PrimeFocusReflectorConfig | CassegrainReflectorConfig,
    Field(discriminator="kind"),
]


class CircularApertureBeamModelConfig(_BeamInputModel):
    """Circular aperture using each resolved antenna diameter."""

    kind: Literal["circular_aperture"] = "circular_aperture"
    taper: DirectTaperConfig = Field(default_factory=GaussianTaperConfig)


class RectangularApertureBeamModelConfig(_BeamInputModel):
    """Rectangular aperture with explicit north/east dimensions."""

    kind: Literal["rectangular_aperture"] = "rectangular_aperture"
    north_length_m: _StrictPositiveFiniteFloat
    east_length_m: _StrictPositiveFiniteFloat


class EllipticalApertureBeamModelConfig(_BeamInputModel):
    """Elliptical aperture with explicit north/east diameters."""

    kind: Literal["elliptical_aperture"] = "elliptical_aperture"
    north_diameter_m: _StrictPositiveFiniteFloat
    east_diameter_m: _StrictPositiveFiniteFloat


class AnalyticalIlluminationBeamModelConfig(_BeamInputModel):
    """Analytically derived circular-aperture illumination."""

    kind: Literal["analytical_illumination"] = "analytical_illumination"
    illumination: IlluminationConfig
    taper_profile: FeedDerivedTaperConfig = Field(
        default_factory=DerivedGaussianTaperConfig
    )
    reflector: ReflectorConfig = Field(default_factory=PrimeFocusReflectorConfig)


class NumericalIlluminationBeamModelConfig(_BeamInputModel):
    """Numerically integrated circular-aperture illumination."""

    kind: Literal["numerical_illumination"] = "numerical_illumination"
    illumination: IlluminationConfig
    reflector: ReflectorConfig = Field(default_factory=PrimeFocusReflectorConfig)


AnalyticBeamModelConfig = Annotated[
    CircularApertureBeamModelConfig
    | RectangularApertureBeamModelConfig
    | EllipticalApertureBeamModelConfig
    | AnalyticalIlluminationBeamModelConfig
    | NumericalIlluminationBeamModelConfig,
    Field(discriminator="kind"),
]


class FITSBeamSourceConfig(_BeamInputModel):
    """One unresolved local BeamFITS source and fixed load options.

    ``docs/development/sci005_beam_physics_plan.md`` Section 5.1.1 selects the
    Stage-3 full-efield accepted subset with exactly one authored literal on
    this block.  ``uvbeam_peak_common_v1`` names an accepted *subset* of the
    committed bytes, not a normalizing operation: RadioSim renormalizes nothing
    under either literal, and the two are different accepted interpretations of
    the same ``beam_type == "efield"`` file rather than a strict widening of one
    another.
    """

    kind: Literal["fits"] = "fits"
    path: Path
    normalization: Literal["peak", "uvbeam_peak_common_v1"] = "peak"
    angular_interpolation: Literal["bilinear"] = "bilinear"
    frequency_interpolation: Literal["cubic", "linear"] = "cubic"

    @field_validator("path", mode="before")
    @classmethod
    def validate_path(cls, value: Any) -> Any:
        return _validate_beam_path(value)


class FITSBeamAssignmentConfig(_BeamInputModel):
    """One authored tagged antenna-to-FITS assignment."""

    antenna: AntennaReference
    beam: FITSBeamSourceConfig

    @field_validator("antenna")
    @classmethod
    def require_exact_antenna_reference(
        cls, value: AntennaReference
    ) -> AntennaReference:
        if type(value) not in (AntennaNumberReference, AntennaNameReference):
            raise ValueError("antenna must be an exact AntennaReference model")
        return value


class AnalyticBeamChoiceConfig(_BeamInputModel):
    """An analytic choice inside a mixed assignment list."""

    kind: Literal["analytic"] = "analytic"


MixedBeamChoiceConfig = Annotated[
    AnalyticBeamChoiceConfig | FITSBeamSourceConfig,
    Field(discriminator="kind"),
]


class MixedBeamAssignmentConfig(_BeamInputModel):
    """One authored tagged antenna choice in mixed mode."""

    antenna: AntennaReference
    beam: MixedBeamChoiceConfig

    @field_validator("antenna")
    @classmethod
    def require_exact_antenna_reference(
        cls, value: AntennaReference
    ) -> AntennaReference:
        if type(value) not in (AntennaNumberReference, AntennaNameReference):
            raise ValueError("antenna must be an exact AntennaReference model")
        return value


_StrictFiniteAzimuthDeg = Annotated[
    float,
    Field(strict=True, ge=-180.0, le=180.0, allow_inf_nan=False),
]
_StrictFiniteElevationDeg = Annotated[
    float,
    Field(strict=True, ge=-90.0, le=90.0, allow_inf_nan=False),
]


class PointingOffsetConfig(_BeamInputModel):
    """The array-wide default deterministic pointing offset.

    ``azimuth_offset_deg`` rotates the beam frame about the local vertical,
    North through East; ``elevation_offset_deg`` then tilts the boresight away
    from the zenith. For RadioSim's zenith-pointed beams the boresight lands at
    topocentric azimuth ``azimuth_offset_deg`` and zenith angle
    ``elevation_offset_deg``, so the peak moves by exactly that great-circle
    angle and a pure azimuth offset moves it not at all.
    """

    azimuth_offset_deg: _StrictFiniteAzimuthDeg = 0.0
    elevation_offset_deg: _StrictFiniteElevationDeg = 0.0


class AntennaPointingOffsetConfig(_BeamInputModel):
    """One authored per-antenna pointing override.

    An entry whose two angles are both zero is the explicit way to say that this
    antenna is perfectly pointed while the array-wide default is not.
    """

    antenna: AntennaReference
    azimuth_offset_deg: _StrictFiniteAzimuthDeg = 0.0
    elevation_offset_deg: _StrictFiniteElevationDeg = 0.0

    @field_validator("antenna")
    @classmethod
    def require_exact_antenna_reference(
        cls, value: AntennaReference
    ) -> AntennaReference:
        if type(value) not in (AntennaNumberReference, AntennaNameReference):
            raise ValueError("antenna must be an exact AntennaReference model")
        return value


class BeamPointingConfig(_BeamInputModel):
    """The optional ``beams.pointing`` block."""

    default: PointingOffsetConfig | None = None
    per_antenna: tuple[AntennaPointingOffsetConfig, ...] = ()

    @model_validator(mode="after")
    def require_a_non_zero_offset(self) -> BeamPointingConfig:
        authored: list[float] = []
        if self.default is not None:
            authored.extend(
                (self.default.azimuth_offset_deg, self.default.elevation_offset_deg)
            )
        for entry in self.per_antenna:
            authored.extend((entry.azimuth_offset_deg, entry.elevation_offset_deg))
        if not any(value != 0.0 for value in authored):
            raise ValueError(
                "beams.pointing: every authored offset is zero, so the block has "
                "no effect; remove it, or give at least one antenna a non-zero "
                "azimuth_offset_deg or elevation_offset_deg."
            )
        return self


class RuzeErrorBeamDiagnosticConfig(_BeamInputModel):
    """One authored ``error_beam_diagnostic`` declaration.

    ``docs/development/sci005_beam_physics_plan.md`` Section 3.4: the literal
    ``gaussian_covariance_power`` names a real, zero-mean, jointly Gaussian,
    second-order stationary aperture-equivalent surface-error field with
    ``rho_h(Delta) = exp[-(|Delta|/L)^2]``, so ``correlation_length_m`` is that
    field's one-over-e correlation length ``L``.  It declares an
    *ensemble-power* diagnostic and never a deterministic error-beam voltage.
    """

    kind: Literal["gaussian_covariance_power"]
    correlation_length_m: _Stage1Float


class SurfaceErrorConfig(_BeamInputModel):
    """The array-wide default Ruze random-surface RMS error."""

    rms_surface_error_m: _StrictNonNegativeFiniteFloat = 0.0
    error_beam_diagnostic: RuzeErrorBeamDiagnosticConfig | None = None


class AntennaSurfaceErrorConfig(_BeamInputModel):
    """One authored per-antenna surface-error override."""

    antenna: AntennaReference
    rms_surface_error_m: _StrictNonNegativeFiniteFloat = 0.0
    error_beam_diagnostic: RuzeErrorBeamDiagnosticConfig | None = None

    @field_validator("antenna")
    @classmethod
    def require_exact_antenna_reference(
        cls, value: AntennaReference
    ) -> AntennaReference:
        if type(value) not in (AntennaNumberReference, AntennaNameReference):
            raise ValueError("antenna must be an exact AntennaReference model")
        return value


class BeamSurfaceErrorConfig(_BeamInputModel):
    """The optional ``beams.surface_error`` block."""

    default: SurfaceErrorConfig | None = None
    per_antenna: tuple[AntennaSurfaceErrorConfig, ...] = ()

    @model_validator(mode="after")
    def require_a_non_zero_surface_error(self) -> BeamSurfaceErrorConfig:
        authored: list[float] = []
        if self.default is not None:
            authored.append(self.default.rms_surface_error_m)
        authored.extend(entry.rms_surface_error_m for entry in self.per_antenna)
        if not any(value != 0.0 for value in authored):
            raise ValueError(
                "beams.surface_error: every authored surface error is zero, so "
                "the block has no effect; remove it, or give at least one "
                "antenna a positive rms_surface_error_m."
            )
        return self


class SquintRecordConfig(_BeamInputModel):
    """One complete authored native-feed squint record (Section 4.1).

    ``docs/development/sci005_beam_physics_plan.md`` Section 4.1: the nominal
    pointing is the midpoint of the two native feeds, so
    ``per_feed_offset_deg_at_reference`` is the displacement of **one** hand and
    the total feed-to-feed separation is twice it.  The mechanical position
    angle describes the physical off-axis feed location, measured North through
    East in the antenna beam frame; it is not the electrical
    ``receptors.*.feed_rotation_deg`` used to build ``C``.

    The Cotton/Uson frequency law is exact
    (``delta(nu) = asin[(nu_ref / nu) sin delta_ref]``); the small-angle
    ``1/nu`` limit is documentation, not the production law.
    """

    convention: Literal["cotton_uson_exact_v1"]
    reference_frequency_hz: _Stage1Float
    per_feed_offset_deg_at_reference: _Stage1Float
    mechanical_feed_position_angle_deg: _Stage1Float
    positive_native_feed: Literal["x", "y", "r", "l"]


class AntennaSquintConfig(_BeamInputModel):
    """One authored per-antenna squint record.

    Section 4.1.1: a per-antenna record carries exactly ``antenna`` plus one
    complete squint record's five fields.  There is no suppression form in v1 --
    an array in which some antennas must not squint is authored with no
    ``default`` and one record per squinting antenna.
    """

    antenna: AntennaReference
    convention: Literal["cotton_uson_exact_v1"]
    reference_frequency_hz: _Stage1Float
    per_feed_offset_deg_at_reference: _Stage1Float
    mechanical_feed_position_angle_deg: _Stage1Float
    positive_native_feed: Literal["x", "y", "r", "l"]

    @field_validator("antenna")
    @classmethod
    def require_exact_antenna_reference(
        cls, value: AntennaReference
    ) -> AntennaReference:
        if type(value) not in (AntennaNumberReference, AntennaNameReference):
            raise ValueError("antenna must be an exact AntennaReference model")
        return value


class BeamSquintConfig(_BeamInputModel):
    """The optional ``beams.squint`` block (Section 4.1.1).

    Exactly two fields.  A block carrying neither a ``default`` nor any
    ``per_antenna`` record is an exact identity and is rejected as a
    ``ConfigSemanticError`` with the frozen ``beam.squint.identity_block`` code,
    not here: the identity, value-domain, and unsupported-family rules are
    document-level checks with frozen paths and messages.
    """

    default: SquintRecordConfig | None = None
    per_antenna: tuple[AntennaSquintConfig, ...] = ()


class SupportLegConfig(_BeamInputModel):
    """One authored support leg.

    Section 3.2: a leg is the closed radial strip of physical width ``width_m``
    running from the edge of the central shadow to the ideal pupil edge, centred
    on its mechanical position angle measured North through East.  It is one
    *outward half-strip*, so a structure on both sides of the dish is authored
    as two records separated by 180 degrees.
    """

    position_angle_deg: _Stage1Float
    width_m: _Stage1Float


class ApertureBlockageConfig(_BeamInputModel):
    """The authored ``beams.aperture_physics.blockage`` child (Section 3.2)."""

    central_diameter_ratio: _Stage1Float
    support_legs: tuple[SupportLegConfig, ...] = ()


class ZernikeModeConfig(_BeamInputModel):
    """One authored real unit-RMS disk Zernike mode (Section 3.3).

    Exactly three keys.  ``n`` and ``m`` are exact Python integers, never
    booleans and never a Noll or OSA single index, and the coefficient is signed
    aperture-equivalent reflector surface-height error in metres -- one half of
    the reflected optical-path difference (R. J. Noll, JOSA 66, 207 (1976), DOI
    10.1364/JOSA.66.000207).
    """

    n: _StrictExactInt
    m: _StrictExactInt
    surface_height_coefficient_m: _Stage1Float


class ZernikeSurfaceConfig(_BeamInputModel):
    """The authored ``beams.aperture_physics.zernike_surface`` child."""

    convention: Literal["radiosim.real_unit_rms_disk_surface_height.v1"]
    modes: tuple[ZernikeModeConfig, ...] = Field(min_length=1)


class AperturePhysicsConfig(_BeamInputModel):
    """The authored array-wide ``beams.aperture_physics`` block.

    Section 3.1 fixes the normalization literal: ``N_0`` is always the
    unmodified ideal-aperture integral, it is not recomputed after masking, and
    the modified beam is never re-peak-normalized, so blockage and aberration
    loss occur exactly once in ``E``.
    """

    normalization: Literal["unmodified_ideal_aperture_v1"]
    blockage: ApertureBlockageConfig | None = None
    zernike_surface: ZernikeSurfaceConfig | None = None


class AnalyticBeamsConfig(_BeamInputModel):
    """One shared analytic model."""

    mode: Literal["analytic"] = "analytic"
    model: AnalyticBeamModelConfig = Field(
        default_factory=CircularApertureBeamModelConfig
    )
    pointing: BeamPointingConfig | None = None
    surface_error: BeamSurfaceErrorConfig | None = None
    aperture_physics: AperturePhysicsConfig | None = None
    squint: BeamSquintConfig | None = None


class SharedFITSBeamsConfig(_BeamInputModel):
    """One shared FITS source."""

    mode: Literal["shared_fits"] = "shared_fits"
    beam: FITSBeamSourceConfig
    pointing: BeamPointingConfig | None = None
    surface_error: BeamSurfaceErrorConfig | None = None
    aperture_physics: AperturePhysicsConfig | None = None
    squint: BeamSquintConfig | None = None


class PerAntennaFITSBeamsConfig(_BeamInputModel):
    """An ordered nonempty list of authored FITS assignments."""

    mode: Literal["per_antenna_fits"] = "per_antenna_fits"
    assignments: tuple[FITSBeamAssignmentConfig, ...] = Field(min_length=1)
    pointing: BeamPointingConfig | None = None
    surface_error: BeamSurfaceErrorConfig | None = None
    aperture_physics: AperturePhysicsConfig | None = None
    squint: BeamSquintConfig | None = None


class MixedBeamsConfig(_BeamInputModel):
    """One analytic model plus ordered per-antenna analytic/FITS choices."""

    mode: Literal["mixed"] = "mixed"
    analytic_model: AnalyticBeamModelConfig = Field(
        default_factory=CircularApertureBeamModelConfig
    )
    assignments: tuple[MixedBeamAssignmentConfig, ...] = Field(min_length=1)
    pointing: BeamPointingConfig | None = None
    surface_error: BeamSurfaceErrorConfig | None = None
    aperture_physics: AperturePhysicsConfig | None = None
    squint: BeamSquintConfig | None = None


BeamsConfig = Annotated[
    AnalyticBeamsConfig
    | SharedFITSBeamsConfig
    | PerAntennaFITSBeamsConfig
    | MixedBeamsConfig,
    Field(discriminator="mode"),
]


__all__ = [
    "AnalyticBeamChoiceConfig",
    "AntennaPointingOffsetConfig",
    "AntennaSquintConfig",
    "AntennaSurfaceErrorConfig",
    "ApertureBlockageConfig",
    "AperturePhysicsConfig",
    "RuzeErrorBeamDiagnosticConfig",
    "SupportLegConfig",
    "ZernikeModeConfig",
    "ZernikeSurfaceConfig",
    "BeamPointingConfig",
    "BeamSquintConfig",
    "BeamSurfaceErrorConfig",
    "PointingOffsetConfig",
    "SquintRecordConfig",
    "SurfaceErrorConfig",
    "AnalyticBeamModelConfig",
    "AnalyticBeamsConfig",
    "AnalyticalIlluminationBeamModelConfig",
    "BeamsConfig",
    "CassegrainReflectorConfig",
    "CircularApertureBeamModelConfig",
    "CorrugatedHornIlluminationConfig",
    "CosineTaperConfig",
    "DerivedGaussianTaperConfig",
    "DerivedParabolicSquaredTaperConfig",
    "DerivedParabolicTaperConfig",
    "DipoleGroundPlaneIlluminationConfig",
    "DirectTaperConfig",
    "EllipticalApertureBeamModelConfig",
    "FITSBeamAssignmentConfig",
    "FITSBeamSourceConfig",
    "FeedDerivedTaperConfig",
    "GaussianTaperConfig",
    "IlluminationConfig",
    "MixedBeamAssignmentConfig",
    "MixedBeamChoiceConfig",
    "MixedBeamsConfig",
    "NumericalIlluminationBeamModelConfig",
    "OpenWaveguideIlluminationConfig",
    "ParabolicSquaredTaperConfig",
    "ParabolicTaperConfig",
    "PerAntennaFITSBeamsConfig",
    "PrimeFocusReflectorConfig",
    "RectangularApertureBeamModelConfig",
    "ReflectorConfig",
    "SharedFITSBeamsConfig",
    "UniformTaperConfig",
]
