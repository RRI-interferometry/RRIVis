"""Strict, frozen input models for Tier 3 beam configuration.

This module owns only user-authored beam shape.  It performs no path
resolution, file access, assignment resolution, dependency import, or runtime
activation.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import Field, field_validator

from radiosim.io.config import StrictFrozenModel
from radiosim.io.instrument_config import AntennaReference

_ENVIRONMENT_PATH = re.compile(r"\$(?:\{[^}]+\}|[A-Za-z_][A-Za-z0-9_]*)")

_StrictPositiveFiniteFloat = Annotated[
    float,
    Field(strict=True, gt=0.0, allow_inf_nan=False),
]
_StrictNonNegativeFiniteFloat = Annotated[
    float,
    Field(strict=True, ge=0.0, allow_inf_nan=False),
]


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


class UniformTaperConfig(StrictFrozenModel):
    """Uniform direct circular-aperture illumination."""

    kind: Literal["uniform"] = "uniform"


class GaussianTaperConfig(StrictFrozenModel):
    """Gaussian direct circular-aperture illumination."""

    kind: Literal["gaussian"] = "gaussian"
    edge_taper_db: _StrictNonNegativeFiniteFloat = 10.0


class ParabolicTaperConfig(StrictFrozenModel):
    """Parabolic direct circular-aperture illumination."""

    kind: Literal["parabolic"] = "parabolic"
    edge_taper_db: _StrictNonNegativeFiniteFloat = 10.0


class ParabolicSquaredTaperConfig(StrictFrozenModel):
    """Parabolic-squared direct circular-aperture illumination."""

    kind: Literal["parabolic_squared"] = "parabolic_squared"
    edge_taper_db: _StrictNonNegativeFiniteFloat = 10.0


class CosineTaperConfig(StrictFrozenModel):
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


class DerivedGaussianTaperConfig(StrictFrozenModel):
    """Gaussian profile whose edge taper is derived from illumination."""

    kind: Literal["gaussian"] = "gaussian"


class DerivedParabolicTaperConfig(StrictFrozenModel):
    """Parabolic profile whose edge taper is derived from illumination."""

    kind: Literal["parabolic"] = "parabolic"


class DerivedParabolicSquaredTaperConfig(StrictFrozenModel):
    """Parabolic-squared profile with a derived edge taper."""

    kind: Literal["parabolic_squared"] = "parabolic_squared"


FeedDerivedTaperConfig = Annotated[
    DerivedGaussianTaperConfig
    | DerivedParabolicTaperConfig
    | DerivedParabolicSquaredTaperConfig,
    Field(discriminator="kind"),
]


class CorrugatedHornIlluminationConfig(StrictFrozenModel):
    """Corrugated-horn illumination parameters."""

    kind: Literal["corrugated_horn"] = "corrugated_horn"
    focal_ratio: _StrictPositiveFiniteFloat = 0.4
    q: _StrictPositiveFiniteFloat = 1.15


class OpenWaveguideIlluminationConfig(StrictFrozenModel):
    """Open-waveguide illumination parameters."""

    kind: Literal["open_waveguide"] = "open_waveguide"
    focal_ratio: _StrictPositiveFiniteFloat = 0.4
    b_over_lambda: _StrictPositiveFiniteFloat = 0.7


class DipoleGroundPlaneIlluminationConfig(StrictFrozenModel):
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


class PrimeFocusReflectorConfig(StrictFrozenModel):
    """Prime-focus reflector selection."""

    kind: Literal["prime_focus"] = "prime_focus"


class CassegrainReflectorConfig(StrictFrozenModel):
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


class CircularApertureBeamModelConfig(StrictFrozenModel):
    """Circular aperture using each resolved antenna diameter."""

    kind: Literal["circular_aperture"] = "circular_aperture"
    taper: DirectTaperConfig = Field(default_factory=GaussianTaperConfig)


class RectangularApertureBeamModelConfig(StrictFrozenModel):
    """Rectangular aperture with explicit north/east dimensions."""

    kind: Literal["rectangular_aperture"] = "rectangular_aperture"
    north_length_m: _StrictPositiveFiniteFloat
    east_length_m: _StrictPositiveFiniteFloat


class EllipticalApertureBeamModelConfig(StrictFrozenModel):
    """Elliptical aperture with explicit north/east diameters."""

    kind: Literal["elliptical_aperture"] = "elliptical_aperture"
    north_diameter_m: _StrictPositiveFiniteFloat
    east_diameter_m: _StrictPositiveFiniteFloat


class AnalyticalIlluminationBeamModelConfig(StrictFrozenModel):
    """Analytically derived circular-aperture illumination."""

    kind: Literal["analytical_illumination"] = "analytical_illumination"
    illumination: IlluminationConfig
    taper_profile: FeedDerivedTaperConfig = Field(
        default_factory=DerivedGaussianTaperConfig
    )
    reflector: ReflectorConfig = Field(default_factory=PrimeFocusReflectorConfig)


class NumericalIlluminationBeamModelConfig(StrictFrozenModel):
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


class FITSBeamSourceConfig(StrictFrozenModel):
    """One unresolved local BeamFITS source and fixed load options."""

    kind: Literal["fits"] = "fits"
    path: Path
    normalization: Literal["peak"] = "peak"
    angular_interpolation: Literal["bilinear"] = "bilinear"
    frequency_interpolation: Literal["cubic", "linear"] = "cubic"

    @field_validator("path", mode="before")
    @classmethod
    def validate_path(cls, value: Any) -> Any:
        return _validate_beam_path(value)


class FITSBeamAssignmentConfig(StrictFrozenModel):
    """One authored tagged antenna-to-FITS assignment."""

    antenna: AntennaReference
    beam: FITSBeamSourceConfig


class AnalyticBeamChoiceConfig(StrictFrozenModel):
    """An analytic choice inside a mixed assignment list."""

    kind: Literal["analytic"] = "analytic"


MixedBeamChoiceConfig = Annotated[
    AnalyticBeamChoiceConfig | FITSBeamSourceConfig,
    Field(discriminator="kind"),
]


class MixedBeamAssignmentConfig(StrictFrozenModel):
    """One authored tagged antenna choice in mixed mode."""

    antenna: AntennaReference
    beam: MixedBeamChoiceConfig


class AnalyticBeamsConfig(StrictFrozenModel):
    """One shared analytic model."""

    mode: Literal["analytic"] = "analytic"
    model: AnalyticBeamModelConfig = Field(
        default_factory=CircularApertureBeamModelConfig
    )


class SharedFITSBeamsConfig(StrictFrozenModel):
    """One shared FITS source."""

    mode: Literal["shared_fits"] = "shared_fits"
    beam: FITSBeamSourceConfig


class PerAntennaFITSBeamsConfig(StrictFrozenModel):
    """An ordered nonempty list of authored FITS assignments."""

    mode: Literal["per_antenna_fits"] = "per_antenna_fits"
    assignments: tuple[FITSBeamAssignmentConfig, ...] = Field(min_length=1)


class MixedBeamsConfig(StrictFrozenModel):
    """One analytic model plus ordered per-antenna analytic/FITS choices."""

    mode: Literal["mixed"] = "mixed"
    analytic_model: AnalyticBeamModelConfig = Field(
        default_factory=CircularApertureBeamModelConfig
    )
    assignments: tuple[MixedBeamAssignmentConfig, ...] = Field(min_length=1)


BeamsConfig = Annotated[
    AnalyticBeamsConfig
    | SharedFITSBeamsConfig
    | PerAntennaFITSBeamsConfig
    | MixedBeamsConfig,
    Field(discriminator="mode"),
]


__all__ = [
    "AnalyticBeamChoiceConfig",
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
