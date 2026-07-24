"""Pure numeric analytic beam primitives.

Canonical typed beam composition and runtime evaluation are privately owned by
``radiosim.core.beam.BeamSystem``. This package exposes only independent
numeric aperture, taper, feed-pattern, reflector-geometry, and diagnostic HPBW
functions.

Modules
-------
aperture
    Aperture shape far-field patterns (Airy, sinc, elliptical Airy).
taper
    Illumination taper functions (uniform, Gaussian, parabolic, cosine).
feed
    Feed pattern models and reflector geometry (prime-focus, Cassegrain).
numerical_hpbw
    Diagnostic HPBW finder for beam pattern visualization.
"""

from collections.abc import Callable as _Callable
from typing import cast as _cast

import numpy as _np
from numpy.typing import NDArray as _NDArray

from radiosim.core.jones.beam.analytic import numerical_hpbw as _numerical_hpbw
from radiosim.core.jones.beam.analytic.aperture import (
    airy_voltage_pattern,
    compute_u_beam,
    elliptical_airy_voltage_pattern,
    sinc_voltage_pattern,
)
from radiosim.core.jones.beam.analytic.feed import (
    cassegrain_angle,
    compute_edge_angle,
    corrugated_horn_pattern,
    dipole_ground_plane_pattern,
    open_waveguide_pattern,
    prime_focus_angle,
)
from radiosim.core.jones.beam.analytic.taper import (
    cosine_taper,
    gaussian_taper_pattern,
    parabolic_squared_taper,
    parabolic_taper,
    uniform_taper,
)

compute_hpbw_numerical = _cast(
    _Callable[..., _NDArray[_np.float64]],
    vars(_numerical_hpbw)["compute_hpbw_numerical"],
)

__all__ = [
    "compute_u_beam",
    "airy_voltage_pattern",
    "sinc_voltage_pattern",
    "elliptical_airy_voltage_pattern",
    "uniform_taper",
    "gaussian_taper_pattern",
    "parabolic_taper",
    "parabolic_squared_taper",
    "cosine_taper",
    "corrugated_horn_pattern",
    "open_waveguide_pattern",
    "dipole_ground_plane_pattern",
    "prime_focus_angle",
    "cassegrain_angle",
    "compute_edge_angle",
    "compute_hpbw_numerical",
]
