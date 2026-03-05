# rrivis/core/precision.py
"""Precision configuration for RRIvis simulations.

This module provides granular control over numerical precision at different
stages of the visibility simulation pipeline. Different components have
different precision sensitivity:

- Geometric phase (K): CRITICAL - small errors accumulate at high frequencies
- Direction cosines: HIGH - feed directly into phase calculations
- Beam interpolation: LOW - already approximate from grid sampling
- Accumulation: MEDIUM - sum of many terms can accumulate rounding errors

Supported precision levels:
- float32/complex64: Single precision (~7 decimal digits)
- float64/complex128: Double precision (~15 decimal digits) - DEFAULT
- float128/complex256: Extended precision (~34 digits) - NumPy only, slow

Examples
--------
Using presets:

>>> from rrivis.core.precision import PrecisionConfig
>>> precision = PrecisionConfig.standard()  # All float64
>>> precision = PrecisionConfig.fast()  # float32 where safe
>>> precision = PrecisionConfig.precise()  # float128 for critical paths

Granular control:

>>> precision = PrecisionConfig(
...     default="float64",
...     jones=JonesPrecision(
...         geometric_phase="float128",  # Critical for phase
...         beam="float32",  # Less sensitive
...     ),
... )

In Simulator:

>>> from rrivis import Simulator
>>> sim = Simulator(backend="jax", precision="fast")
>>> sim = Simulator(backend="numpy", precision=PrecisionConfig.precise())
"""

from __future__ import annotations

import warnings
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, Field, field_validator, model_validator

# =============================================================================
# Precision Level Type
# =============================================================================

PrecisionLevel = Literal["float32", "float64", "float128"]

# Valid precision strings for validation
VALID_PRECISIONS = {"float32", "float64", "float128"}


# =============================================================================
# Platform-specific dtype detection
# =============================================================================


def _check_float128_support() -> bool:
    """Check if float128 is available on this platform."""
    try:
        # Try to create a float128 array
        arr = np.array([1.0], dtype=np.float128)
        # Verify it's actually extended precision, not just an alias
        return arr.dtype.itemsize > 8
    except (TypeError, AttributeError):
        return False


def _check_complex256_support() -> bool:
    """Check if complex256 is available on this platform."""
    try:
        arr = np.array([1.0 + 1.0j], dtype=np.complex256)
        return arr.dtype.itemsize > 16
    except (TypeError, AttributeError):
        return False


# Platform capabilities (evaluated once at import)
FLOAT128_AVAILABLE = _check_float128_support()
COMPLEX256_AVAILABLE = _check_complex256_support()


# =============================================================================
# Dtype Mapping
# =============================================================================


def get_real_dtype(precision: PrecisionLevel, backend: str = "numpy") -> Any:
    """Get the real (non-complex) dtype for a precision level.

    Parameters
    ----------
    precision : str
        Precision level: "float32", "float64", or "float128"
    backend : str
        Backend name for compatibility checking

    Returns
    -------
    dtype
        NumPy or backend-specific dtype

    Raises
    ------
    ValueError
        If precision level is not supported on this backend
    """
    if precision == "float32":
        return np.float32
    elif precision == "float64":
        return np.float64
    elif precision == "float128":
        if backend in ("jax", "numba"):
            warnings.warn(
                f"float128 not supported on {backend} backend. "
                "Falling back to float64.",
                UserWarning,
                stacklevel=2,
            )
            return np.float64
        if not FLOAT128_AVAILABLE:
            warnings.warn(
                "float128 not available on this platform. Falling back to float64.",
                UserWarning,
                stacklevel=2,
            )
            return np.float64
        return np.float128
    else:
        raise ValueError(f"Unknown precision level: {precision}")


def get_complex_dtype(precision: PrecisionLevel, backend: str = "numpy") -> Any:
    """Get the complex dtype for a precision level.

    Parameters
    ----------
    precision : str
        Precision level: "float32", "float64", or "float128"
    backend : str
        Backend name for compatibility checking

    Returns
    -------
    dtype
        NumPy or backend-specific complex dtype
    """
    if precision == "float32":
        return np.complex64
    elif precision == "float64":
        return np.complex128
    elif precision == "float128":
        if backend in ("jax", "numba"):
            warnings.warn(
                f"complex256 not supported on {backend} backend. "
                "Falling back to complex128.",
                UserWarning,
                stacklevel=2,
            )
            return np.complex128
        if not COMPLEX256_AVAILABLE:
            warnings.warn(
                "complex256 not available on this platform. "
                "Falling back to complex128.",
                UserWarning,
                stacklevel=2,
            )
            return np.complex128
        return np.complex256
    else:
        raise ValueError(f"Unknown precision level: {precision}")


def get_dtype_size(precision: PrecisionLevel, complex_type: bool = False) -> int:
    """Get the size in bytes for a precision level.

    Parameters
    ----------
    precision : str
        Precision level
    complex_type : bool
        If True, return size for complex type

    Returns
    -------
    int
        Size in bytes
    """
    sizes = {
        "float32": (4, 8),  # real, complex
        "float64": (8, 16),
        "float128": (16, 32),
    }
    return sizes[precision][1 if complex_type else 0]


# =============================================================================
# Coordinate Precision Configuration
# =============================================================================


class CoordinatePrecision(BaseModel):
    """Precision settings for coordinate calculations.

    Attributes
    ----------
    antenna_positions : str
        Precision for antenna ENU/ECEF positions
    source_positions : str
        Precision for source RA/Dec coordinates
    direction_cosines : str
        Precision for l, m, n direction cosines (HIGH sensitivity)
    uvw : str
        Precision for baseline UVW coordinates
    """

    antenna_positions: PrecisionLevel = Field(
        default="float64", description="Antenna position precision"
    )
    source_positions: PrecisionLevel = Field(
        default="float64", description="Source coordinate precision"
    )
    direction_cosines: PrecisionLevel = Field(
        default="float64",
        description="Direction cosine (l,m,n) precision - feeds into phase",
    )
    uvw: PrecisionLevel = Field(
        default="float64", description="Baseline UVW coordinate precision"
    )

    @field_validator("*", mode="before")
    @classmethod
    def validate_precision(cls, v: Any) -> str:
        if isinstance(v, str) and v in VALID_PRECISIONS:
            return v
        raise ValueError(f"Invalid precision: {v}. Must be one of {VALID_PRECISIONS}")

    def get_dtype(self, component: str, backend: str = "numpy") -> Any:
        """Get dtype for a specific coordinate component."""
        precision = getattr(self, component)
        return get_real_dtype(precision, backend)


# =============================================================================
# Jones Matrix Precision Configuration
# =============================================================================


class JonesPrecision(BaseModel):
    """Precision settings for Jones matrix calculations.

    Different Jones terms have different precision requirements:
    - K (geometric_phase): CRITICAL - phase errors accumulate
    - E (beam): LOW - already interpolated from grid
    - Z (ionosphere): MEDIUM - TEC to phase conversion
    - G (gain): LOW - typically ~1% accuracy anyway

    Attributes
    ----------
    geometric_phase : str
        K term - geometric delay/phase (CRITICAL)
    beam : str
        E term - primary beam response
    ionosphere : str
        Z term - ionospheric effects
    troposphere : str
        T term - tropospheric effects
    parallactic : str
        P term - parallactic angle rotation
    gain : str
        G term - antenna gains
    bandpass : str
        B term - bandpass response
    polarization_leakage : str
        D term - polarization leakage
    """

    geometric_phase: PrecisionLevel = Field(
        default="float64",
        description="K term (geometric delay) - CRITICAL for phase accuracy",
    )
    beam: PrecisionLevel = Field(default="float64", description="E term (primary beam)")
    ionosphere: PrecisionLevel = Field(
        default="float64", description="Z term (ionosphere)"
    )
    troposphere: PrecisionLevel = Field(
        default="float64", description="T term (troposphere)"
    )
    parallactic: PrecisionLevel = Field(
        default="float64", description="P term (parallactic angle)"
    )
    gain: PrecisionLevel = Field(
        default="float64", description="G term (antenna gains)"
    )
    bandpass: PrecisionLevel = Field(default="float64", description="B term (bandpass)")
    polarization_leakage: PrecisionLevel = Field(
        default="float64", description="D term (polarization leakage)"
    )

    @field_validator("*", mode="before")
    @classmethod
    def validate_precision(cls, v: Any) -> str:
        if isinstance(v, str) and v in VALID_PRECISIONS:
            return v
        raise ValueError(f"Invalid precision: {v}. Must be one of {VALID_PRECISIONS}")

    def get_dtype(self, jones_term: str, backend: str = "numpy") -> Any:
        """Get complex dtype for a specific Jones term."""
        precision = getattr(self, jones_term)
        return get_complex_dtype(precision, backend)

    def get_real_dtype(self, jones_term: str, backend: str = "numpy") -> Any:
        """Get real dtype for a specific Jones term."""
        precision = getattr(self, jones_term)
        return get_real_dtype(precision, backend)


# =============================================================================
# Sky Model Precision Configuration
# =============================================================================


class SkyModelPrecision(BaseModel):
    """Precision settings for sky model data storage.

    Controls the numerical precision of arrays stored inside ``SkyModel``.
    Source positions (RA/Dec) are phase-critical and default to float64.
    HEALPix brightness temperature maps default to float32 (adequate for
    ~4-digit sky temperatures).

    Attributes
    ----------
    source_positions : str
        Precision for RA/Dec arrays — feeds into phase calculations.
    flux : str
        Precision for flux density and Stokes parameters.
    spectral_index : str
        Precision for power-law spectral index arrays.
    healpix_maps : str
        Precision for HEALPix brightness temperature maps.
    """

    source_positions: PrecisionLevel = Field(
        default="float64", description="RA/Dec precision — phase-critical"
    )
    flux: PrecisionLevel = Field(
        default="float64", description="Flux density and Stokes parameter precision"
    )
    spectral_index: PrecisionLevel = Field(
        default="float64", description="Power-law spectral index precision"
    )
    healpix_maps: PrecisionLevel = Field(
        default="float32", description="HEALPix brightness temperature map precision"
    )

    @field_validator("*", mode="before")
    @classmethod
    def validate_precision(cls, v: Any) -> str:
        if isinstance(v, str) and v in VALID_PRECISIONS:
            return v
        raise ValueError(f"Invalid precision: {v}. Must be one of {VALID_PRECISIONS}")

    def get_dtype(self, component: str, backend: str = "numpy") -> Any:
        """Get real dtype for a specific sky model component.

        Parameters
        ----------
        component : str
            Component name: "source_positions", "flux", "spectral_index",
            or "healpix_maps".
        backend : str
            Backend name for compatibility checking.

        Returns
        -------
        dtype
            NumPy dtype.
        """
        precision = getattr(self, component)
        return get_real_dtype(precision, backend)


# =============================================================================
# Main Precision Configuration
# =============================================================================


class PrecisionConfig(BaseModel):
    """Main precision configuration for RRIvis simulations.

    Provides hierarchical control over numerical precision at different
    stages of the visibility simulation.

    Attributes
    ----------
    default : str
        Default precision level (fallback for unspecified components)
    coordinates : CoordinatePrecision
        Precision for coordinate calculations
    jones : JonesPrecision
        Precision for Jones matrix calculations
    accumulation : str
        Precision for visibility accumulation (summing sources)
    output : str
        Precision for output visibility data

    Examples
    --------
    >>> config = PrecisionConfig.standard()  # All float64
    >>> config = PrecisionConfig.fast()  # Optimized for speed
    >>> config = PrecisionConfig(
    ...     default="float64",
    ...     jones=JonesPrecision(geometric_phase="float128"),
    ... )
    """

    default: PrecisionLevel = Field(
        default="float64", description="Default precision (fallback)"
    )
    coordinates: CoordinatePrecision = Field(
        default_factory=CoordinatePrecision, description="Coordinate precision settings"
    )
    jones: JonesPrecision = Field(
        default_factory=JonesPrecision, description="Jones matrix precision settings"
    )
    accumulation: PrecisionLevel = Field(
        default="float64", description="Visibility accumulation precision"
    )
    output: PrecisionLevel = Field(
        default="float64", description="Output visibility precision"
    )
    sky_model: SkyModelPrecision = Field(
        default_factory=SkyModelPrecision,
        description="Sky model data precision settings",
    )

    model_config = {
        "extra": "forbid",  # Reject unknown fields
        "validate_assignment": True,
    }

    @field_validator("default", "accumulation", "output", mode="before")
    @classmethod
    def validate_precision(cls, v: Any) -> str:
        if isinstance(v, str) and v in VALID_PRECISIONS:
            return v
        raise ValueError(f"Invalid precision: {v}. Must be one of {VALID_PRECISIONS}")

    @model_validator(mode="after")
    def apply_defaults(self) -> PrecisionConfig:
        """Apply default precision to sub-configs where not explicitly set."""
        # This validator runs after initial construction
        # We could use it to propagate defaults, but for now we keep
        # sub-configs with their own defaults
        return self

    # =========================================================================
    # Preset Factory Methods
    # =========================================================================

    @classmethod
    def standard(cls) -> PrecisionConfig:
        """Standard precision: float64 everywhere (current default behavior).

        Use for: General simulations, 21cm cosmology, precision-critical work.

        Returns
        -------
        PrecisionConfig
            Configuration with all components set to float64
        """
        return cls(default="float64")

    @classmethod
    def fast(cls) -> PrecisionConfig:
        """Fast precision: float32 where safe, float64 for critical paths.

        Use for: Large SKA simulations, quick previews, GPU optimization.

        Keeps float64 for:
        - Geometric phase (K term) - phase errors are critical
        - Direction cosines - feed into phase
        - Accumulation - prevent rounding in sums

        Returns
        -------
        PrecisionConfig
            Speed-optimized configuration
        """
        return cls(
            default="float32",
            coordinates=CoordinatePrecision(
                antenna_positions="float32",
                source_positions="float32",
                direction_cosines="float64",  # Keep precise
                uvw="float32",
            ),
            jones=JonesPrecision(
                geometric_phase="float64",  # CRITICAL - keep precise
                beam="float32",
                ionosphere="float32",
                troposphere="float32",
                parallactic="float32",
                gain="float32",
                bandpass="float32",
                polarization_leakage="float32",
            ),
            sky_model=SkyModelPrecision(
                source_positions="float32",
                flux="float32",
                spectral_index="float32",
                healpix_maps="float32",
            ),
            accumulation="float64",  # Prevent rounding errors in sums
            output="float32",
        )

    @classmethod
    def precise(cls) -> PrecisionConfig:
        """High precision: float128 for critical paths, float64 elsewhere.

        Use for: Validation, debugging numerical issues, cross-validation
                 with other simulators.

        Note: float128 only works with NumPy backend and may be slow.

        Returns
        -------
        PrecisionConfig
            Precision-optimized configuration
        """
        return cls(
            default="float64",
            coordinates=CoordinatePrecision(
                antenna_positions="float64",
                source_positions="float64",
                direction_cosines="float128",  # Maximum precision
                uvw="float64",
            ),
            jones=JonesPrecision(
                geometric_phase="float128",  # Maximum precision
                beam="float64",
                ionosphere="float64",
                troposphere="float64",
                parallactic="float64",
                gain="float64",
                bandpass="float64",
                polarization_leakage="float64",
            ),
            sky_model=SkyModelPrecision(
                source_positions="float64",
                flux="float64",
                spectral_index="float64",
                healpix_maps="float64",
            ),
            accumulation="float128",  # Maximum precision for sums
            output="float64",
        )

    @classmethod
    def ultra(cls) -> PrecisionConfig:
        """Ultra precision: float128 everywhere.

        Use for: Debugging only - very slow, NumPy only.

        Warning: This is 10-20x slower than standard and only works
                 with the NumPy backend on supported platforms.

        Returns
        -------
        PrecisionConfig
            Maximum precision configuration
        """
        return cls(
            default="float128",
            coordinates=CoordinatePrecision(
                antenna_positions="float128",
                source_positions="float128",
                direction_cosines="float128",
                uvw="float128",
            ),
            jones=JonesPrecision(
                geometric_phase="float128",
                beam="float128",
                ionosphere="float128",
                troposphere="float128",
                parallactic="float128",
                gain="float128",
                bandpass="float128",
                polarization_leakage="float128",
            ),
            sky_model=SkyModelPrecision(
                source_positions="float128",
                flux="float128",
                spectral_index="float128",
                healpix_maps="float64",
            ),
            accumulation="float128",
            output="float128",
        )

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def with_overrides(self, **kwargs) -> PrecisionConfig:
        """Create a new config with specified overrides.

        Parameters
        ----------
        **kwargs
            Fields to override. Can include nested dicts for
            coordinates and jones.

        Returns
        -------
        PrecisionConfig
            New configuration with overrides applied

        Examples
        --------
        >>> config = PrecisionConfig.fast().with_overrides(
        ...     output="float64",
        ...     jones={"beam": "float64"},
        ... )
        """
        data = self.model_dump()

        for key, value in kwargs.items():
            if key in ("coordinates", "jones", "sky_model") and isinstance(value, dict):
                # Merge nested dicts
                data[key].update(value)
            else:
                data[key] = value

        return PrecisionConfig(**data)

    def get_real_dtype(
        self, component: str, sub_component: str | None = None, backend: str = "numpy"
    ) -> Any:
        """Get real dtype for a component.

        Parameters
        ----------
        component : str
            Top-level component: "coordinates", "jones", "accumulation", "output"
        sub_component : str, optional
            Sub-component for coordinates or jones
        backend : str
            Backend name for compatibility

        Returns
        -------
        dtype
            NumPy dtype
        """
        if component == "coordinates" and sub_component:
            return self.coordinates.get_dtype(sub_component, backend)
        elif component == "jones" and sub_component:
            return self.jones.get_real_dtype(sub_component, backend)
        elif component == "sky_model" and sub_component:
            return self.sky_model.get_dtype(sub_component, backend)
        elif component in ("accumulation", "output", "default"):
            precision = getattr(self, component)
            return get_real_dtype(precision, backend)
        else:
            return get_real_dtype(self.default, backend)

    def get_complex_dtype(
        self, component: str, sub_component: str | None = None, backend: str = "numpy"
    ) -> Any:
        """Get complex dtype for a component.

        Parameters
        ----------
        component : str
            Top-level component: "coordinates", "jones", "accumulation", "output"
        sub_component : str, optional
            Sub-component for jones
        backend : str
            Backend name for compatibility

        Returns
        -------
        dtype
            NumPy complex dtype
        """
        if component == "jones" and sub_component:
            return self.jones.get_dtype(sub_component, backend)
        elif component in ("accumulation", "output", "default"):
            precision = getattr(self, component)
            return get_complex_dtype(precision, backend)
        else:
            return get_complex_dtype(self.default, backend)

    def estimate_memory_factor(self) -> float:
        """Estimate memory usage factor relative to float64 baseline.

        Returns
        -------
        float
            Approximate memory multiplier (1.0 = float64 baseline)
        """
        # Weight different components by typical memory usage
        weights = {
            "jones": 0.3,  # Jones matrices
            "accumulation": 0.5,  # Visibility arrays
            "output": 0.2,  # Output storage
        }

        size_factors = {
            "float32": 0.5,
            "float64": 1.0,
            "float128": 2.0,
        }

        factor = 0.0

        # Jones average
        jones_precisions = [
            self.jones.geometric_phase,
            self.jones.beam,
            self.jones.ionosphere,
            self.jones.gain,
        ]
        jones_avg = sum(size_factors[p] for p in jones_precisions) / len(
            jones_precisions
        )
        factor += weights["jones"] * jones_avg

        # Accumulation and output
        factor += weights["accumulation"] * size_factors[self.accumulation]
        factor += weights["output"] * size_factors[self.output]

        return factor

    def validate_for_backend(self, backend_name: str) -> list[str]:
        """Check compatibility with a backend and return warnings.

        Parameters
        ----------
        backend_name : str
            Backend name ("numpy", "jax", "numba")

        Returns
        -------
        list of str
            Warning messages for incompatible settings
        """
        warnings_list = []

        if backend_name in ("jax", "numba"):
            # Check for float128 usage
            float128_fields = []

            if self.default == "float128":
                float128_fields.append("default")
            if self.accumulation == "float128":
                float128_fields.append("accumulation")
            if self.output == "float128":
                float128_fields.append("output")

            for field in self.coordinates.model_fields:
                if getattr(self.coordinates, field) == "float128":
                    float128_fields.append(f"coordinates.{field}")

            for field in self.jones.model_fields:
                if getattr(self.jones, field) == "float128":
                    float128_fields.append(f"jones.{field}")

            for field in self.sky_model.model_fields:
                if getattr(self.sky_model, field) == "float128":
                    float128_fields.append(f"sky_model.{field}")

            if float128_fields:
                warnings_list.append(
                    f"float128 not supported on {backend_name} backend. "
                    f"The following will fall back to float64: {float128_fields}"
                )

        if backend_name == "numpy" and not FLOAT128_AVAILABLE:
            # Check for float128 usage on unsupported platform
            if any(
                getattr(self, f, None) == "float128"
                or any(
                    getattr(self.coordinates, f2, None) == "float128"
                    for f2 in self.coordinates.model_fields
                )
                or any(
                    getattr(self.jones, f3, None) == "float128"
                    for f3 in self.jones.model_fields
                )
                for f in ["default", "accumulation", "output"]
            ):
                warnings_list.append(
                    "float128 not available on this platform. "
                    "Will fall back to float64."
                )

        return warnings_list


# =============================================================================
# Helper Functions
# =============================================================================


def resolve_precision(precision: str | PrecisionConfig | None) -> PrecisionConfig:
    """Resolve precision argument to a PrecisionConfig.

    Parameters
    ----------
    precision : str, PrecisionConfig, or None
        Precision specification:
        - None: Use standard (float64)
        - str: Preset name ("standard", "fast", "precise", "ultra")
               or precision level ("float32", "float64", "float128")
        - PrecisionConfig: Use directly

    Returns
    -------
    PrecisionConfig
        Resolved configuration
    """
    if precision is None:
        return PrecisionConfig.standard()

    if isinstance(precision, PrecisionConfig):
        return precision

    if isinstance(precision, str):
        # Check if it's a preset name
        presets = {
            "standard": PrecisionConfig.standard,
            "fast": PrecisionConfig.fast,
            "precise": PrecisionConfig.precise,
            "ultra": PrecisionConfig.ultra,
        }
        if precision in presets:
            return presets[precision]()

        # Check if it's a precision level (use as default for all)
        if precision in VALID_PRECISIONS:
            return PrecisionConfig(default=precision)

        raise ValueError(
            f"Unknown precision: {precision}. "
            f"Use preset name ({list(presets.keys())}) or "
            f"precision level ({list(VALID_PRECISIONS)})"
        )

    raise TypeError(
        f"precision must be str, PrecisionConfig, or None, got {type(precision)}"
    )


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    # Types
    "PrecisionLevel",
    "VALID_PRECISIONS",
    # Platform info
    "FLOAT128_AVAILABLE",
    "COMPLEX256_AVAILABLE",
    # Dtype helpers
    "get_real_dtype",
    "get_complex_dtype",
    "get_dtype_size",
    # Config classes
    "CoordinatePrecision",
    "JonesPrecision",
    "SkyModelPrecision",
    "PrecisionConfig",
    # Helper functions
    "resolve_precision",
]
