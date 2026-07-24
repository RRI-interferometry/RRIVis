"""Generic beam analysis and sky-projection helpers.

Runtime beam loading, assignment, and evaluation are owned by
``radiosim.core.beam.BeamSystem``. This package retains only generic analysis
and projection helpers that do not define a second beam runtime.

Pure analytic numeric primitives remain available from
``radiosim.core.jones.beam.analytic``.
"""

from importlib import import_module

__all__ = [
    # Beam sky projection
    "BeamSkyProjection",
    "compute_beam_power_on_radec_grid",
    "create_rgba_overlay",
    "extract_contours",
    # Beam analysis
    "BeamRadialProfile",
    "BeamFeatures",
    "azimuthal_radial_profile",
    "detect_beam_features",
]


_LAZY_EXPORTS = {
    "BeamFeatures": ("radiosim.core.jones.beam.analysis", "BeamFeatures"),
    "BeamRadialProfile": (
        "radiosim.core.jones.beam.analysis",
        "BeamRadialProfile",
    ),
    "azimuthal_radial_profile": (
        "radiosim.core.jones.beam.analysis",
        "azimuthal_radial_profile",
    ),
    "detect_beam_features": (
        "radiosim.core.jones.beam.analysis",
        "detect_beam_features",
    ),
    "BeamSkyProjection": (
        "radiosim.core.jones.beam.projection",
        "BeamSkyProjection",
    ),
    "compute_beam_power_on_radec_grid": (
        "radiosim.core.jones.beam.projection",
        "compute_beam_power_on_radec_grid",
    ),
    "create_rgba_overlay": (
        "radiosim.core.jones.beam.projection",
        "create_rgba_overlay",
    ),
    "extract_contours": (
        "radiosim.core.jones.beam.projection",
        "extract_contours",
    ),
}


def __getattr__(name: str) -> object:
    try:
        module_name, attribute_name = _LAZY_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(import_module(module_name), attribute_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Include lazy public exports in interactive discovery."""
    return sorted(set(globals()) | set(_LAZY_EXPORTS))
