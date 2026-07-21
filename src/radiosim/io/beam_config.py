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


class AnalyticBeamsConfig(_BeamInputModel):
    """One shared analytic model."""

    mode: Literal["analytic"] = "analytic"
    model: AnalyticBeamModelConfig = Field(
        default_factory=CircularApertureBeamModelConfig
    )


class SharedFITSBeamsConfig(_BeamInputModel):
    """One shared FITS source."""

    mode: Literal["shared_fits"] = "shared_fits"
    beam: FITSBeamSourceConfig


class PerAntennaFITSBeamsConfig(_BeamInputModel):
    """An ordered nonempty list of authored FITS assignments."""

    mode: Literal["per_antenna_fits"] = "per_antenna_fits"
    assignments: tuple[FITSBeamAssignmentConfig, ...] = Field(min_length=1)


class MixedBeamsConfig(_BeamInputModel):
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
