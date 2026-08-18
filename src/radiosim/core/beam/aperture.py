"""SCI-005 Stage-1 scalar aperture physics.

``docs/development/sci005_beam_physics_plan.md`` Sections 3.1--3.3 define one
normalized aperture transform

.. math::

    e(\\mathbf q,\\lambda)=\\frac{1}{N_0}\\int_{\\mathcal P_0}
    A(\\mathbf u)M(\\mathbf u)
    \\exp\\!\\left[-i\\frac{4\\pi}{\\lambda}h(\\mathbf u)\\right]
    \\exp(-i\\mathbf q\\cdot\\mathbf u)\\,d^2u,
    \\qquad
    N_0=\\int_{\\mathcal P_0}A(\\mathbf u)\\,d^2u,

in which the central blockage, the support shadows, and the deterministic
Zernike surface height are a mask and a phase *inside one integral*.  ``N_0`` is
always the unmodified ideal-aperture integral: it is not recomputed after
masking and the modified beam is never re-peak-normalized, so blockage and
aberration loss occur exactly once in ``E``.  Separately evaluated far-field
patterns are never multiplied, because the Fourier transform does not
distribute over aperture multiplication.

This module is the sole production owner of the Stage-1 pupil profiles, the
support mask, the Zernike phase, and the boundary-fitted polar Gauss-Legendre
quadrature that evaluates them together.

Primary sources
---------------
* Aperture blockage and reflector analysis: NASA TM X-63186
  (https://ntrs.nasa.gov/citations/19680013447) and ITU-R SA.2401-0
  (https://www.itu.int/pub/R-REP-SA.2401-2017).
* Real unit-RMS disk Zernikes: R. J. Noll, *Zernike polynomials and atmospheric
  turbulence*, JOSA **66**, 207 (1976), DOI 10.1364/JOSA.66.000207.  The
  annular-basis distinction -- why authored coefficients stay in the
  *unobscured* disk basis and why their quadrature sum is not the RMS over a
  transmitting annulus -- is V. N. Mahajan, *Zernike annular polynomials for
  imaging systems with annular pupils*, JOSA **71**, 75 (1981),
  DOI 10.1364/JOSA.71.000075.
* Random reflector errors: J. Ruze, *The effect of aperture errors on the
  antenna radiation pattern* (1952), DOI 10.1007/BF02903409, and *Antenna
  tolerance theory -- a review* (1966), DOI 10.1109/PROC.1966.4784.
"""

from __future__ import annotations

import dataclasses
import hashlib
import math
from collections.abc import Mapping
from dataclasses import dataclass
from fractions import Fraction
from types import MappingProxyType
from typing import Any, Final, Literal

import numpy as np

from radiosim.core.beam.errors import (
    BeamAngularDomainError,
    BeamSamplingDerivationError,
)
from radiosim.core.beam.models import (
    ZERNIKE_MAX_RADIAL_ORDER,
    ResolvedAperturePhysics,
)

__all__ = [
    "STAGE1_SCIENTIFIC_CONVENTIONS",
    "RuzePowerConvergence",
    "RuzePowerDiagnostic",
    "evaluate_ruze_power_diagnostic",
    "ZERNIKE_MAX_RADIAL_ORDER",
    "aperture_transmission_mask",
    "gauss_legendre_order_seed",
    "real_unit_rms_zernike",
    "support_leg_half_angle",
    "surface_height_phase",
    "zernike_surface_height",
]


#: Section 8.1's exact Stage-1 convention record.
#:
#: Every literal here is a *version* of a scientific convention, not a tuning
#: knob: none of them can be authored in YAML, and each enters the scientific
#: fingerprint whenever a Stage-1 feature is explicit.
STAGE1_SCIENTIFIC_CONVENTIONS: Final[Mapping[str, str]] = MappingProxyType(
    {
        "pupil_profile_set": "radiosim.circular_stage1_pupil_profiles.v1",
        "aperture_normalization": "unmodified_ideal_aperture_v1",
        "aperture_axes": "north_east_azimuth_north_through_east_v1",
        "support_mask": "radiosim.central_disk_outward_half_strip_ne.v1",
        "zernike_surface": "radiosim.real_unit_rms_disk_surface_height.v1",
        "aperture_method": "boundary_fitted_polar_gauss_legendre_v1",
        "ruze_covariance": "gaussian_one_over_e_surface_covariance_v1",
        "ruze_method": "poisson_paired_pupil_separation_v1",
    }
)

#: Section 3.3's frozen convergence predicate, evaluated at the resolved width.
_ATOL_FLOOR: Final[float] = 1e-12
_RTOL_FLOOR: Final[float] = 1e-10

#: Section 3.3's fixed pre-allocation caps for the deterministic transform.
_MAX_PANEL_ORDER: Final[int] = 4096
_MAX_NODES_PER_DIRECTION: Final[int] = 2**24
_MAX_PHASE_EVALUATIONS: Final[int] = 2**28
_MAX_WORKSPACE_BYTES: Final[int] = 2**31
_MAX_DOUBLINGS: Final[int] = 4
_MAX_BATCH: Final[int] = 256
_SPEED_OF_LIGHT_M_PER_S: Final[float] = 299_792_458.0

ProfileKind = Literal["uniform", "parabolic", "parabolic_squared"]


# --- target-width primitives --------------------------------------------------


def _default_real_dtype(value: Any) -> np.dtype[Any]:
    dtype = np.asarray(value).dtype
    if dtype.kind == "f" and dtype.itemsize >= np.dtype(np.float64).itemsize:
        return dtype
    return np.dtype(np.float64)


def _complex_for(real_dtype: np.dtype[Any]) -> np.dtype[Any]:
    return np.dtype(np.result_type(real_dtype, np.complex64))


def _pi(real_dtype: np.dtype[Any]) -> Any:
    """Return pi at the resolved width, never narrowed through float64."""
    if real_dtype.itemsize <= np.dtype(np.float64).itemsize:
        return real_dtype.type(np.pi)
    # 50 significant digits comfortably exceeds every NumPy longdouble.
    return real_dtype.type("3.14159265358979323846264338327950288419716939937511")


def _tolerances(real_dtype: np.dtype[Any]) -> tuple[float, float]:
    """Return Section 3.3's frozen ``(atol, rtol)`` at the resolved width."""
    eps = float(np.finfo(real_dtype).eps)
    return max(_ATOL_FLOOR, 32.0 * eps), max(_RTOL_FLOOR, 32.0 * eps)


_GAUSS_LEGENDRE_CACHE: dict[tuple[int, str], tuple[np.ndarray, np.ndarray]] = {}


def _gauss_legendre(order: int, real_dtype: np.dtype[Any]) -> tuple[Any, Any]:
    """Return validated Gauss-Legendre nodes and weights at the target width.

    Section 3.3: nodes and weights are generated above the resolved width,
    rounded once to the resolved dtype, and rejected unless they are finite,
    symmetric, strictly ordered, positive in weight, and sum to two within
    ``32*eps``.  A wider-than-float64 beam never reuses float64 nodes: the
    float64 roots are only a starting guess, and the Newton refinement, the
    derivative, and the weights are all formed in the resolved dtype.
    """
    key = (order, real_dtype.str)
    cached = _GAUSS_LEGENDRE_CACHE.get(key)
    if cached is not None:
        return cached[0], cached[1]
    if order < 1:
        raise BeamSamplingDerivationError(
            f"Gauss-Legendre order {order} is not positive."
        )
    seed, _seed_weights = np.polynomial.legendre.leggauss(order)
    nodes = np.asarray(seed, dtype=real_dtype)
    one = real_dtype.type(1.0)
    for _ in range(4):
        value, derivative = _legendre_pair(order, nodes, real_dtype)
        nodes = nodes - value / derivative
    # Enforce the exact symmetry the predicate requires rather than hoping the
    # refinement preserved it: the Legendre roots are symmetric about zero, so
    # averaging each mirror pair is an exactness statement, not a fudge.
    nodes = real_dtype.type(0.5) * (nodes - nodes[::-1])
    _value, derivative = _legendre_pair(order, nodes, real_dtype)
    weights = real_dtype.type(2.0) / ((one - nodes * nodes) * derivative * derivative)
    weights = real_dtype.type(0.5) * (weights + weights[::-1])

    eps = real_dtype.type(np.finfo(real_dtype).eps)
    if not (np.all(np.isfinite(nodes)) and np.all(np.isfinite(weights))):
        raise BeamSamplingDerivationError(
            f"Gauss-Legendre order {order} produced non-finite nodes or weights."
        )
    if order > 1 and not bool(np.all(np.diff(nodes) > 0)):
        raise BeamSamplingDerivationError(
            f"Gauss-Legendre order {order} nodes are not strictly ordered."
        )
    if not bool(np.all(weights > 0)):
        raise BeamSamplingDerivationError(
            f"Gauss-Legendre order {order} produced a non-positive weight."
        )
    if float(np.max(np.abs(nodes + nodes[::-1]))) > float(32 * eps):
        raise BeamSamplingDerivationError(
            f"Gauss-Legendre order {order} nodes are not symmetric."
        )
    if float(abs(np.sum(weights) - real_dtype.type(2.0))) > float(32 * eps):
        raise BeamSamplingDerivationError(
            f"Gauss-Legendre order {order} weights do not sum to two."
        )
    nodes.setflags(write=False)
    weights.setflags(write=False)
    _GAUSS_LEGENDRE_CACHE[key] = (nodes, weights)
    return nodes, weights


def _legendre_pair(
    order: int,
    x: Any,
    real_dtype: np.dtype[Any],
) -> tuple[Any, Any]:
    """Return ``(P_n(x), P_n'(x))`` from the stable three-term recurrence."""
    previous = np.ones_like(x)
    current = np.array(x, dtype=real_dtype, copy=True)
    if order == 0:
        return previous, np.zeros_like(x)
    for degree in range(2, order + 1):
        factor = real_dtype.type(degree)
        previous, current = (
            current,
            (
                (real_dtype.type(2.0) * factor - real_dtype.type(1.0)) * x * current
                - (factor - real_dtype.type(1.0)) * previous
            )
            / factor,
        )
    derivative = (
        real_dtype.type(order)
        * (x * current - previous)
        / (x * x - real_dtype.type(1.0))
    )
    return current, derivative


# --- Section 3.3: the declared real unit-RMS disk Zernike basis ---------------


def _radial_coefficients(n: int, order: int) -> tuple[tuple[int, int], ...]:
    """Return the exact integer ``(power, coefficient)`` radial polynomial terms.

    The Zernike radial coefficients are integers, so they are formed as exact
    rationals and asserted integral rather than accumulated in floating point.
    That keeps the basis reproducible at every supported width, including the
    extended one where a float64 factorial would already have lost bits.
    """
    terms: list[tuple[int, int]] = []
    for s in range((n - order) // 2 + 1):
        value = Fraction(
            (-1) ** s * math.factorial(n - s),
            math.factorial(s)
            * math.factorial((n + order) // 2 - s)
            * math.factorial((n - order) // 2 - s),
        )
        if value.denominator != 1:  # pragma: no cover - defensive
            raise ValueError("Zernike radial coefficients must be integral")
        terms.append((n - 2 * s, int(value)))
    return tuple(terms)


def real_unit_rms_zernike(
    n: int,
    m: int,
    rho: Any,
    phi: Any,
    *,
    real_dtype: np.dtype[Any] | None = None,
) -> np.ndarray:
    """Evaluate Noll's real unit-RMS disk Zernike ``Z_n^m(rho, phi)``.

    Section 3.3 fixes the basis exactly:

    .. math::

        Z_n^m=\\begin{cases}
        \\sqrt{n+1}\\,R_n^0(\\rho), & m=0,\\\\
        \\sqrt{2(n+1)}\\,R_n^m(\\rho)\\cos(m\\varphi), & m>0,\\\\
        \\sqrt{2(n+1)}\\,R_n^{|m|}(\\rho)\\sin(|m|\\varphi), & m<0,
        \\end{cases}

    normalized so that ``(1/pi) * int Z Z' rho drho dphi = delta``.  Each
    authored coefficient is therefore a unit-RMS surface height on the
    *unobscured* disk (Noll 1976).  After a blockage mask is applied these
    ordinary disk functions cease to be orthogonal over the transmitting
    annulus, which is why the quadrature sum of coefficients is not the RMS over
    that annulus (Mahajan 1981).

    Args:
        n: Radial order, ``0 <= n <= 32``.
        m: Azimuthal order with ``|m| <= n`` and ``n - |m|`` even.
        rho: Normalized radius array.
        phi: Aperture azimuth array in radians, North through East.
        real_dtype: Target real width; defaults to the width of ``rho``.

    Returns:
        The basis function evaluated at ``(rho, phi)``.
    """
    if type(n) is not int or type(m) is not int:
        raise TypeError("n and m must be exact Python integers")
    if not 0 <= n <= ZERNIKE_MAX_RADIAL_ORDER:
        raise ValueError(f"n must satisfy 0 <= n <= {ZERNIKE_MAX_RADIAL_ORDER}")
    if abs(m) > n or (n - abs(m)) % 2 != 0:
        raise ValueError("m must satisfy |m| <= n with n - |m| even")
    dtype = real_dtype if real_dtype is not None else _default_real_dtype(rho)
    radius = np.asarray(rho, dtype=dtype)
    order = abs(m)
    radial = np.zeros(radius.shape, dtype=dtype)
    for power, coefficient in _radial_coefficients(n, order):
        radial = radial + dtype.type(coefficient) * radius**power
    if m == 0:
        return np.asarray(np.sqrt(dtype.type(n + 1)) * radial, dtype=dtype)
    angle = np.asarray(phi, dtype=dtype)
    normalization = np.sqrt(dtype.type(2.0) * dtype.type(n + 1))
    if m > 0:
        return np.asarray(
            normalization * radial * np.cos(dtype.type(m) * angle), dtype=dtype
        )
    return np.asarray(
        normalization * radial * np.sin(dtype.type(order) * angle), dtype=dtype
    )


def zernike_surface_height(
    modes: Any,
    rho: Any,
    phi: Any,
    *,
    real_dtype: np.dtype[Any] | None = None,
) -> np.ndarray:
    """Return the deterministic surface height in metres.

    Section 3.3: each coefficient is signed aperture-equivalent reflector
    surface-height error in metres -- one half of the reflected optical-path
    difference -- so the map is the plain signed sum of the authored modes.
    Sorting the modes for a stable fingerprint cannot change that sum.

    Args:
        modes: Iterable of ``(n, m, surface_height_coefficient_m)`` triples.
        rho: Normalized radius array.
        phi: Aperture azimuth array in radians, North through East.
        real_dtype: Target real width; defaults to the width of ``rho``.
    """
    dtype = real_dtype if real_dtype is not None else _default_real_dtype(rho)
    total = np.zeros(np.asarray(rho, dtype=dtype).shape, dtype=dtype)
    for n, m, coefficient in modes:
        total = total + dtype.type(coefficient) * real_unit_rms_zernike(
            int(n), int(m), rho, phi, real_dtype=dtype
        )
    return total


def surface_height_phase(
    height_m: Any,
    wavelength_m: float,
    *,
    real_dtype: np.dtype[Any] | None = None,
) -> np.ndarray:
    """Return ``exp(-i * 4 * pi * h / lambda)`` for aperture-equivalent height.

    Section 3.1: ``h`` is one half of the signed reflected optical-path
    difference, so the signed excess path is exactly ``2*h`` and RadioSim's
    positive-delay convention produces the negative exponent above.  A physical
    normal displacement maps as ``h = delta_n * cos(i)`` at incidence angle
    ``i``; Stage 1 never invents ``i`` from the beam model, and ``h = delta_n``
    only at normal incidence.
    """
    dtype = real_dtype if real_dtype is not None else _default_real_dtype(height_m)
    if not math.isfinite(wavelength_m) or wavelength_m <= 0.0:
        raise ValueError("wavelength_m must be finite and positive")
    height = np.asarray(height_m, dtype=dtype)
    kappa = dtype.type(4.0) * _pi(dtype) / dtype.type(wavelength_m)
    complex_dtype = _complex_for(dtype)
    return np.asarray(np.exp(-1j * (kappa * height).astype(dtype)), dtype=complex_dtype)


# --- Section 3.2: the support mask -------------------------------------------


def aperture_transmission_mask(
    north_m: Any,
    east_m: Any,
    *,
    diameter_m: float,
    central_diameter_ratio: float | None,
    support_legs: Any = (),
) -> np.ndarray:
    """Return ``True`` where the ideal pupil transmits (Section 3.2).

    For aperture coordinates ``u = (north, east)`` in metres, ``R = D/2``,
    ``r = |u|``, normalized blockage diameter ``epsilon``, and leg angle
    ``beta`` measured North through East:

    .. math::

        \\mathcal L(\\beta,w)=\\{\\mathbf u\\in\\mathcal P_0:
        \\epsilon R\\le r\\le R,\\ \\mathbf u\\cdot\\mathbf d_\\beta\\ge0,\\
        |\\mathbf u\\cdot\\mathbf p_\\beta|\\le w/2\\},

    with ``d = (cos beta, sin beta)`` and ``p = (-sin beta, cos beta)``.  A leg
    is therefore one *outward half-strip*, not an infinite chord and not a pair
    of diametrically opposed legs: a structure on both sides of the dish is
    authored as two records 180 degrees apart.  Masks combine by set union, so
    an overlap is removed once; the central disk is unioned with every leg
    before the single mask is evaluated.  Every boundary is a closed set exactly
    as written and has zero continuum measure.

    Args:
        north_m: North aperture coordinate array in metres.
        east_m: East aperture coordinate array in metres.
        diameter_m: Resolved aperture diameter in metres.
        central_diameter_ratio: Normalized central-shadow diameter, or ``None``.
        support_legs: Iterable of ``(position_angle_deg, width_m)`` pairs.
    """
    north = np.asarray(north_m, dtype=np.float64)
    east = np.asarray(east_m, dtype=np.float64)
    radius = 0.5 * float(diameter_m)
    distance = np.hypot(north, east)
    inside = distance <= radius
    ratio = 0.0 if central_diameter_ratio is None else float(central_diameter_ratio)
    shadow_radius = ratio * radius
    blocked = distance <= shadow_radius if ratio > 0.0 else np.zeros_like(inside)
    for angle_deg, width_m in support_legs:
        beta = math.radians(float(angle_deg))
        along = north * math.cos(beta) + east * math.sin(beta)
        across = -north * math.sin(beta) + east * math.cos(beta)
        blocked = blocked | (
            inside
            & (distance >= shadow_radius)
            & (along >= 0.0)
            & (np.abs(across) <= 0.5 * float(width_m))
        )
    return np.asarray(inside & ~blocked)


def support_leg_half_angle(
    rho: Any,
    width_ratio: float,
    *,
    real_dtype: np.dtype[Any] | None = None,
) -> np.ndarray:
    """Return Section 3.3's ``alpha_j(rho)`` for one leg.

    .. math::

        \\alpha_j(\\rho)=\\begin{cases}
        \\pi/2, & \\rho\\le a_j,\\\\
        \\operatorname{atan2}(a_j,\\sqrt{\\rho^2-a_j^2}), & \\rho>a_j,
        \\end{cases}

    with ``a_j = w_j / D``.  The two branches agree exactly at ``rho = a_j``,
    where the square root is zero, so the half-angle is continuous across
    saturation and the panel that begins there is the one that needs the
    square-root endpoint transformation.
    """
    dtype = real_dtype if real_dtype is not None else _default_real_dtype(rho)
    radius = np.asarray(rho, dtype=dtype)
    ratio = dtype.type(width_ratio)
    inner = np.maximum(radius * radius - ratio * ratio, dtype.type(0.0))
    resolved = np.arctan2(ratio, np.sqrt(inner))
    half_pi = _pi(dtype) / dtype.type(2.0)
    return np.asarray(np.where(radius <= ratio, half_pi, resolved), dtype=dtype)


def gauss_legendre_order_seed(
    bandwidth: float,
    panel_length: float,
    degree: int,
) -> int:
    """Return Section 3.3's exact per-panel quadrature seed.

    .. math::

        \\operatorname{order}(B,L_p,d)=
        \\max\\!\\left(16,\\;2(d+1),\\;
        8+\\left\\lceil\\frac{8BL_p}{\\pi}\\right\\rceil\\right)

    ``B`` carries both the far-field bandwidth ``Q_max`` and the surface-phase
    bandwidth ``kappa*H``, so an authored frequency, diameter, or surface
    coefficient cannot create an unseeded far-field or phase oscillation.
    """
    if not math.isfinite(bandwidth) or bandwidth < 0.0:
        raise BeamSamplingDerivationError(
            "quadrature bandwidth must be finite and >= 0"
        )
    if not math.isfinite(panel_length) or panel_length < 0.0:
        raise BeamSamplingDerivationError("panel length must be finite and >= 0")
    return max(
        16,
        2 * (int(degree) + 1),
        8 + math.ceil(8.0 * bandwidth * panel_length / math.pi),
    )


# --- the resolved aperture specification --------------------------------------


@dataclass(frozen=True, slots=True)
class _RadialPanel:
    """One topology-fitted radial panel and its endpoint transformation."""

    lower: float
    upper: float
    saturation_anchor: float | None


@dataclass(frozen=True, slots=True)
class ApertureSpecification:
    """Everything one antenna's Stage-1 aperture transform needs.

    Built once per beam handler.  The profile keeps ``p`` its existing
    mixture-weight meaning, so the unmasked, unaberrated transform reproduces
    the accepted scalar Hankel response exactly.
    """

    diameter_m: float
    profile_kind: ProfileKind
    pedestal: float
    epsilon: float | None
    legs: tuple[tuple[float, float], ...]
    modes: tuple[tuple[int, int, float], ...]

    @property
    def radius_m(self) -> float:
        return 0.5 * self.diameter_m

    @property
    def max_radial_order(self) -> int:
        return max((n for n, _m, _c in self.modes), default=0)

    @property
    def max_azimuthal_order(self) -> int:
        return max((abs(m) for _n, m, _c in self.modes), default=0)


def build_aperture_specification(
    aperture_physics: ResolvedAperturePhysics | None,
    *,
    diameter_m: float,
    profile_kind: ProfileKind,
    edge_taper_db: float,
) -> ApertureSpecification:
    """Resolve one antenna's aperture specification from its configuration.

    ``aperture_physics`` may be ``None``: the Ruze power diagnostic needs the
    same resolved pupil profile for an antenna that authored no aperture block
    at all, and that antenna's pupil is simply the unobstructed, unaberrated
    one.
    """
    pedestal = 10.0 ** (-float(edge_taper_db) / 20.0)
    if aperture_physics is None:
        return ApertureSpecification(
            diameter_m=float(diameter_m),
            profile_kind=profile_kind,
            pedestal=pedestal,
            epsilon=None,
            legs=(),
            modes=(),
        )
    blockage = aperture_physics.blockage
    epsilon = None if blockage is None else blockage.central_diameter_ratio
    legs: tuple[tuple[float, float], ...] = ()
    if blockage is not None:
        legs = tuple(
            (math.radians(leg.position_angle_deg), leg.width_m / float(diameter_m))
            for leg in blockage.support_legs
        )
    modes: tuple[tuple[int, int, float], ...] = ()
    if aperture_physics.zernike_surface is not None:
        modes = tuple(
            (mode.n, mode.m, mode.surface_height_coefficient_m)
            for mode in aperture_physics.zernike_surface.modes
        )
    return ApertureSpecification(
        diameter_m=float(diameter_m),
        profile_kind=profile_kind,
        pedestal=pedestal,
        epsilon=epsilon,
        legs=legs,
        modes=modes,
    )


def _profile(
    spec: ApertureSpecification,
    rho: Any,
    real_dtype: np.dtype[Any],
) -> Any:
    """Return Section 3.1's radial pupil profile ``A(rho)``.

    All three profiles satisfy ``int_0^1 A(rho) rho drho = 1/2``, so the
    unmodified ideal-aperture integral is exactly ``N_0 = pi R^2`` analytically
    and is never re-estimated from numerical nodes.
    """
    radius = np.asarray(rho, dtype=real_dtype)
    one = real_dtype.type(1.0)
    if spec.profile_kind == "uniform":
        return np.ones_like(radius)
    pedestal = real_dtype.type(spec.pedestal)
    complement = one - radius * radius
    if spec.profile_kind == "parabolic":
        tapered = real_dtype.type(2.0) * complement
    else:
        tapered = real_dtype.type(3.0) * complement * complement
    return pedestal * np.ones_like(radius) + (one - pedestal) * tapered


# --- Section 3.3: boundary-fitted radial topology -----------------------------


def _wrap_two_pi(angle: float) -> float:
    wrapped = math.fmod(angle, 2.0 * math.pi)
    return wrapped + 2.0 * math.pi if wrapped < 0.0 else wrapped


def _wrap_pi(angle: float) -> float:
    wrapped = _wrap_two_pi(angle)
    return wrapped - 2.0 * math.pi if wrapped > math.pi else wrapped


def _half_angle(ratio: float, radius: float) -> float:
    if radius <= ratio:
        return 0.5 * math.pi
    return math.atan2(ratio, math.sqrt(radius * radius - ratio * ratio))


def _bisect_to_representable(
    function: Any,
    low: float,
    high: float,
    real_dtype: np.dtype[Any],
) -> float | None:
    """Isolate one sign change down to adjacent representable target values.

    Section 3.3: when a root is not exactly representable the bisection
    continues to adjacent bracketing values and the canonical breakpoint is
    always the *upper* one, so a left-closed/right-open panel boundary puts the
    rounded root in the post-root panel.
    """
    value_low = function(low)
    value_high = function(high)
    if value_low == 0.0:
        return float(np.asarray(low, dtype=real_dtype))
    if value_high == 0.0:
        return float(np.asarray(high, dtype=real_dtype))
    if (value_low > 0.0) == (value_high > 0.0):
        return None
    lower = np.asarray(low, dtype=real_dtype).item()
    upper = np.asarray(high, dtype=real_dtype).item()
    for _ in range(4096):
        if not lower < upper:
            break
        middle = np.asarray(0.5 * (lower + upper), dtype=real_dtype).item()
        if middle <= lower or middle >= upper:
            break
        if (function(middle) > 0.0) == (value_low > 0.0):
            lower = middle
        else:
            upper = middle
    return float(upper)


def _topology_breakpoints(
    spec: ApertureSpecification,
    real_dtype: np.dtype[Any],
) -> tuple[float, ...]:
    """Return the sorted unique canonical radial-panel boundaries.

    The panel set fits every hard boundary: the active lower bound, every
    saturation radius ``a_j``, every support-topology radius where two legs
    become tangent or one contains the other, every radius where an interval
    endpoint crosses the fixed periodic cut, and the outer boundary one.  Two
    unequal mathematical roots that would collapse onto the same canonical
    breakpoint are a representability failure, not a silently merged panel.
    """
    lower = 0.0 if spec.epsilon is None else float(spec.epsilon)
    candidates: list[tuple[float, str]] = []

    for beta, ratio in spec.legs:
        if not math.isfinite(ratio) or ratio <= 0.0:
            raise BeamSamplingDerivationError(
                "a support leg width ratio underflowed to a non-positive value"
            )
        if lower < ratio < 1.0:
            candidates.append((ratio, f"saturation:{ratio!r}"))
        # Periodic-cut roots: alpha_j(rho) equals the wrapped leg angle, and the
        # closed form rho = a / sin(alpha) inverts the half-angle exactly.
        for target in (_wrap_two_pi(beta), _wrap_two_pi(-beta)):
            if 0.0 < target <= 0.5 * math.pi:
                root = ratio / math.sin(target)
                if lower < root < 1.0:
                    candidates.append((root, f"cut:{beta!r}:{target!r}"))

    legs = spec.legs
    for first in range(len(legs)):
        for second in range(first + 1, len(legs)):
            beta_a, ratio_a = legs[first]
            beta_b, ratio_b = legs[second]
            separation = abs(_wrap_pi(beta_a - beta_b))
            segment_bounds = sorted(
                {lower, min(ratio_a, ratio_b), max(ratio_a, ratio_b), 1.0}
            )
            segments = [
                (left, right)
                for left, right in zip(segment_bounds, segment_bounds[1:], strict=False)
                if lower <= left < right <= 1.0
            ]
            for left, right in segments:
                for sign in (1.0, -1.0):

                    def event(
                        radius: float,
                        _a: float = ratio_a,
                        _b: float = ratio_b,
                        _s: float = sign,
                        _d: float = separation,
                    ) -> float:
                        return (
                            _half_angle(_a, radius) + _s * _half_angle(_b, radius) - _d
                            if _s > 0.0
                            else abs(_half_angle(_a, radius) - _half_angle(_b, radius))
                            - _d
                        )

                    root = _bisect_to_representable(event, left, right, real_dtype)
                    if root is not None and lower < root < 1.0:
                        candidates.append((root, f"pair:{first}:{second}:{sign!r}"))

    resolved: dict[float, list[tuple[float, str]]] = {}
    for value, origin in candidates:
        canonical = float(np.asarray(value, dtype=real_dtype))
        resolved.setdefault(canonical, []).append((value, origin))
    eps = float(np.finfo(real_dtype).eps)
    for canonical, entries in resolved.items():
        smallest = min(item[0] for item in entries)
        largest = max(item[0] for item in entries)
        if largest - smallest > 8.0 * eps * max(1.0, abs(canonical)):
            raise BeamSamplingDerivationError(
                "two unequal Stage-1 aperture topology boundaries resolve to the "
                f"same canonical breakpoint {canonical!r}; the geometry is not "
                "representable at the resolved width."
            )
    boundaries = sorted({lower, *resolved, 1.0})
    if not all(math.isfinite(value) for value in boundaries):
        raise BeamSamplingDerivationError(
            "a Stage-1 aperture panel boundary is not finite"
        )
    return tuple(boundaries)


def _radial_panels(
    spec: ApertureSpecification,
    real_dtype: np.dtype[Any],
) -> tuple[_RadialPanel, ...]:
    boundaries = _topology_breakpoints(spec, real_dtype)
    saturation = {ratio for _beta, ratio in spec.legs}
    panels: list[_RadialPanel] = []
    for lower, upper in zip(boundaries, boundaries[1:], strict=False):
        if not upper > lower:
            continue
        anchor = lower if any(abs(lower - a) == 0.0 for a in saturation) else None
        panels.append(_RadialPanel(lower, upper, anchor))
    if not panels:
        raise BeamSamplingDerivationError(
            "the resolved Stage-1 aperture has no positive-measure radial panel"
        )
    return tuple(panels)


def _transmitting_intervals(
    spec: ApertureSpecification,
    radius: float,
) -> tuple[tuple[float, float], ...]:
    """Return the disjoint transmitting angular intervals at one radius.

    The blocked intervals are split at zero, mapped into ``[0, 2*pi)``, sorted
    by their exact endpoints, and unioned before their complement is
    integrated.  There is no merge tolerance.
    """
    if not spec.legs:
        return ((0.0, 2.0 * math.pi),)
    blocked: list[tuple[float, float]] = []
    for beta, ratio in spec.legs:
        half = _half_angle(ratio, radius)
        start = beta - half
        end = beta + half
        if end - start >= 2.0 * math.pi:
            return ()
        start = _wrap_two_pi(start)
        end = start + 2.0 * half
        if end <= 2.0 * math.pi:
            blocked.append((start, end))
        else:
            blocked.append((start, 2.0 * math.pi))
            blocked.append((0.0, end - 2.0 * math.pi))
    blocked.sort()
    merged: list[tuple[float, float]] = []
    for start, end in blocked:
        if merged and start <= merged[-1][1]:
            if end > merged[-1][1]:
                merged[-1] = (merged[-1][0], end)
        else:
            merged.append((start, end))
    intervals: list[tuple[float, float]] = []
    cursor = 0.0
    for start, end in merged:
        if start > cursor:
            intervals.append((cursor, start))
        cursor = max(cursor, end)
    if cursor < 2.0 * math.pi:
        intervals.append((cursor, 2.0 * math.pi))
    return tuple(item for item in intervals if item[1] > item[0])


# --- Section 3.3: the one production transform --------------------------------


@dataclass(frozen=True, slots=True)
class _Bandwidths:
    radial: float
    angular: float
    kappa: float
    surface_radial: float
    surface_angular: float
    q_max: float


def _bandwidths(
    spec: ApertureSpecification,
    wavevectors: np.ndarray,
    wavelength_m: float,
) -> _Bandwidths:
    """Return Section 3.3's phase-bandwidth bounds for one direction batch."""
    kappa = 4.0 * math.pi / wavelength_m
    surface_radial = 0.0
    surface_angular = 0.0
    for n, m, coefficient in spec.modes:
        normalization = math.sqrt(n + 1.0) if m == 0 else math.sqrt(2.0 * (n + 1.0))
        surface_radial += abs(coefficient) * normalization * float(n) ** 2
        surface_angular += abs(coefficient) * normalization * float(abs(m))
    if wavevectors.size:
        q_max = float(np.max(spec.radius_m * np.sqrt(np.sum(wavevectors**2, axis=1))))
    else:  # pragma: no cover - guarded by the caller
        q_max = 0.0
    return _Bandwidths(
        radial=q_max + kappa * surface_radial,
        angular=q_max + kappa * surface_angular,
        kappa=kappa,
        surface_radial=surface_radial,
        surface_angular=surface_angular,
        q_max=q_max,
    )


def _panel_radial_seed(
    panel: _RadialPanel, bandwidths: _Bandwidths, degree: int
) -> int:
    if panel.saturation_anchor is None:
        length = panel.upper - panel.lower
    else:
        # rho = a + (rho_hi - a) t^2 has max |d rho / d t| = 2 (rho_hi - a).
        length = 2.0 * (panel.upper - panel.saturation_anchor)
    return gauss_legendre_order_seed(bandwidths.radial, length, degree)


def _check_panel_order(order: int) -> int:
    if order > _MAX_PANEL_ORDER:
        raise BeamSamplingDerivationError(
            f"Stage-1 aperture quadrature requires per-panel order {order}, above "
            f"the fixed cap {_MAX_PANEL_ORDER}."
        )
    return order


def _evaluate_once(
    spec: ApertureSpecification,
    panels: tuple[_RadialPanel, ...],
    wavevectors: np.ndarray,
    bandwidths: _Bandwidths,
    radial_factor: int,
    angular_factor: int,
    real_dtype: np.dtype[Any],
    complex_dtype: np.dtype[Any],
    budget: dict[str, int],
) -> tuple[np.ndarray, int]:
    """Evaluate the normalized transform once at one refinement level."""
    count = wavevectors.shape[0]
    total = np.zeros(count, dtype=complex_dtype)
    pi_value = _pi(real_dtype)
    kappa = real_dtype.type(bandwidths.kappa)
    radius = real_dtype.type(spec.radius_m)
    q_north = np.asarray(wavevectors[:, 0], dtype=real_dtype)
    q_east = np.asarray(wavevectors[:, 1], dtype=real_dtype)
    node_total = 0
    budget["fhat"] = budget.get("fhat", 0) + count
    degree_radial = spec.max_radial_order
    degree_angular = spec.max_azimuthal_order

    for panel in panels:
        midpoint = 0.5 * (panel.lower + panel.upper)
        if not _transmitting_intervals(spec, midpoint):
            # A fully blocked panel is proven zero from its exact interval union
            # and receives no radial or angular quadrature nodes.
            continue
        order = _check_panel_order(
            _panel_radial_seed(panel, bandwidths, degree_radial) * radial_factor
        )
        nodes, weights = _gauss_legendre(order, real_dtype)
        if panel.saturation_anchor is None:
            half = real_dtype.type(0.5 * (panel.upper - panel.lower))
            centre = real_dtype.type(0.5 * (panel.upper + panel.lower))
            radii = centre + half * nodes
            jacobian = half * weights
        else:
            anchor = real_dtype.type(panel.saturation_anchor)
            span = real_dtype.type(panel.upper - panel.saturation_anchor)
            parameter = real_dtype.type(0.5) * (nodes + real_dtype.type(1.0))
            radii = anchor + span * parameter * parameter
            jacobian = (
                real_dtype.type(0.5) * weights * real_dtype.type(2.0) * span * parameter
            )
        profile = _profile(spec, radii, real_dtype)

        for index in range(order):
            radius_value = float(radii[index])
            intervals = _transmitting_intervals(spec, radius_value)
            if not intervals:
                continue
            angles: list[Any] = []
            angle_weights: list[Any] = []
            for start, end in intervals:
                angular_order = _check_panel_order(
                    gauss_legendre_order_seed(
                        bandwidths.angular, end - start, degree_angular
                    )
                    * angular_factor
                )
                angular_nodes, angular_raw = _gauss_legendre(angular_order, real_dtype)
                half_width = real_dtype.type(0.5 * (end - start))
                centre = real_dtype.type(0.5 * (end + start))
                angles.append(centre + half_width * angular_nodes)
                angle_weights.append(half_width * angular_raw)
            phi = np.concatenate(angles)
            weight = np.concatenate(angle_weights)
            node_total += int(phi.size)
            budget["nodes"] = max(budget["nodes"], node_total)
            budget["phases"] += int(phi.size) * count
            if node_total > _MAX_NODES_PER_DIRECTION:
                raise BeamSamplingDerivationError(
                    "Stage-1 aperture quadrature exceeded the fixed "
                    f"{_MAX_NODES_PER_DIRECTION} node-per-direction cap."
                )
            if budget["phases"] > _MAX_PHASE_EVALUATIONS:
                raise BeamSamplingDerivationError(
                    "Stage-1 aperture quadrature exceeded the fixed "
                    f"{_MAX_PHASE_EVALUATIONS} direction-node phase cap."
                )
            radius_node = radii[index]
            if spec.modes:
                height = zernike_surface_height(
                    spec.modes,
                    np.full(phi.shape, radius_node, dtype=real_dtype),
                    phi,
                    real_dtype=real_dtype,
                )
                surface = kappa * height
            else:
                surface = np.zeros(phi.shape, dtype=real_dtype)
            geometry = (
                radius
                * radius_node
                * (
                    q_north[:, None] * np.cos(phi)[None, :]
                    + q_east[:, None] * np.sin(phi)[None, :]
                )
            )
            phase = surface[None, :] + geometry
            contribution = np.exp(-1j * phase.astype(real_dtype)) @ weight.astype(
                real_dtype
            )
            total = total + (
                jacobian[index] * profile[index] * radius_node
            ) * contribution.astype(complex_dtype)

    return np.asarray(total / pi_value, dtype=complex_dtype), node_total


@dataclass(frozen=True, slots=True)
class _ApertureSolveResult:
    """The accepted transform plus the Section 3.4.2 fields it must report."""

    values: np.ndarray
    radial_factor: int
    angular_factor: int
    refinement_count: int
    max_node_count: int
    penultimate_max_abs_delta: float
    final_max_abs_delta: float
    q_max: float
    phase_products: int
    presentations: int


def _converged(
    coarse: np.ndarray,
    fine: np.ndarray,
    atol: float,
    rtol: float,
) -> tuple[bool, float]:
    delta = float(np.max(np.abs(fine - coarse))) if fine.size else 0.0
    limit = atol + rtol * (float(np.max(np.abs(fine))) if fine.size else 0.0)
    return delta <= limit, delta


def solve_aperture_transform(
    spec: ApertureSpecification,
    wavevectors: np.ndarray,
    *,
    wavelength_m: float,
    real_dtype: np.dtype[Any],
    complex_dtype: np.dtype[Any],
    budget: dict[str, int] | None = None,
) -> _ApertureSolveResult:
    """Solve the one normalized Stage-1 aperture transform and report on it.

    Section 3.3 fixes the whole numerical method: boundary-fitted polar
    Gauss-Legendre panels, phase-bandwidth seeds, angular convergence at fixed
    radial order followed by radial convergence of the two angularly converged
    arrays, two consecutive successful comparisons per dimension under
    ``atol + rtol*max(abs(refined))``, at most four doublings per dimension, and
    every fixed resource cap checked before allocation.  No one-pair or
    unconverged non-empty result is ever returned.

    Args:
        spec: The resolved aperture specification.
        wavevectors: ``(S, 2)`` array of ``(q_north, q_east)`` in rad/m.
        wavelength_m: Observing wavelength in metres.
        real_dtype: Resolved real width.
        complex_dtype: Resolved complex width.
        budget: Optional shared counter mapping, so a caller that runs many
            solves in one public call accumulates one cumulative total.

    Returns:
        The accepted values together with the orders, refinement counts, and
        residuals Section 3.4.2 retains separately.
    """
    directions = np.asarray(wavevectors, dtype=real_dtype)
    if directions.ndim != 2 or directions.shape[1] != 2:
        raise ValueError("wavevectors must have shape (S, 2)")
    counters = budget if budget is not None else {}
    counters.setdefault("nodes", 0)
    counters.setdefault("phases", 0)
    counters.setdefault("presentations", 0)
    if directions.shape[0] == 0:
        return _ApertureSolveResult(
            values=np.zeros(0, dtype=complex_dtype),
            radial_factor=1,
            angular_factor=1,
            refinement_count=0,
            max_node_count=0,
            penultimate_max_abs_delta=0.0,
            final_max_abs_delta=0.0,
            q_max=0.0,
            phase_products=counters["phases"],
            presentations=counters["presentations"],
        )
    panels = _radial_panels(spec, real_dtype)
    bandwidths = _bandwidths(spec, directions, wavelength_m)
    if not any(
        _transmitting_intervals(spec, 0.5 * (panel.lower + panel.upper))
        for panel in panels
    ):
        # Section 3.3: an entirely blocked transmitting pupil is exactly zero,
        # with positive-zero components and no refinement residual.
        return _ApertureSolveResult(
            values=np.zeros(directions.shape[0], dtype=complex_dtype),
            radial_factor=1,
            angular_factor=1,
            refinement_count=0,
            max_node_count=0,
            penultimate_max_abs_delta=0.0,
            final_max_abs_delta=0.0,
            q_max=bandwidths.q_max,
            phase_products=counters["phases"],
            presentations=counters["presentations"],
        )

    atol, rtol = _tolerances(real_dtype)
    refinements = 0
    accepted: dict[str, Any] = {}

    def angular_converged(radial_factor: int) -> tuple[np.ndarray, int, float, float]:
        nonlocal refinements
        angular_factor = 1
        previous, _ = _evaluate_once(
            spec,
            panels,
            directions,
            bandwidths,
            radial_factor,
            angular_factor,
            real_dtype,
            complex_dtype,
            counters,
        )
        successes = 0
        deltas: list[float] = []
        for _ in range(_MAX_DOUBLINGS):
            angular_factor *= 2
            refinements += 1
            refined, _ = _evaluate_once(
                spec,
                panels,
                directions,
                bandwidths,
                radial_factor,
                angular_factor,
                real_dtype,
                complex_dtype,
                counters,
            )
            ok, delta = _converged(previous, refined, atol, rtol)
            if ok:
                deltas.append(delta)
                successes += 1
            else:
                deltas.clear()
                successes = 0
            previous = refined
            if successes >= 2:
                return previous, angular_factor, deltas[-2], deltas[-1]
        raise BeamSamplingDerivationError(
            "Stage-1 aperture angular quadrature did not reach two consecutive "
            "successful comparisons within the permitted doublings."
        )

    radial_factor = 1
    previous, angular_factor, angular_penultimate, angular_final = angular_converged(
        radial_factor
    )
    successes = 0
    radial_deltas: list[float] = []
    for _ in range(_MAX_DOUBLINGS):
        radial_factor *= 2
        refinements += 1
        refined, angular_factor, angular_penultimate, angular_final = angular_converged(
            radial_factor
        )
        ok, delta = _converged(previous, refined, atol, rtol)
        if ok:
            radial_deltas.append(delta)
            successes += 1
        else:
            radial_deltas.clear()
            successes = 0
        previous = refined
        if successes >= 2:
            accepted["values"] = previous
            break
    else:
        raise BeamSamplingDerivationError(
            "Stage-1 aperture radial quadrature did not reach two consecutive "
            "successful comparisons within the permitted doublings."
        )
    return _ApertureSolveResult(
        values=accepted["values"],
        radial_factor=radial_factor,
        angular_factor=angular_factor,
        refinement_count=refinements,
        max_node_count=counters["nodes"],
        # Section 3.4.2: the licensing deltas are maximized across both
        # dimensions, not merely the final radial pair.
        penultimate_max_abs_delta=max(angular_penultimate, radial_deltas[-2]),
        final_max_abs_delta=max(angular_final, radial_deltas[-1]),
        q_max=bandwidths.q_max,
        phase_products=counters["phases"],
        presentations=counters["presentations"],
    )


def evaluate_aperture_transform(
    spec: ApertureSpecification,
    wavevectors: np.ndarray,
    *,
    wavelength_m: float,
    real_dtype: np.dtype[Any],
    complex_dtype: np.dtype[Any],
) -> np.ndarray:
    """Return only the accepted values of :func:`solve_aperture_transform`."""
    return solve_aperture_transform(
        spec,
        wavevectors,
        wavelength_m=wavelength_m,
        real_dtype=real_dtype,
        complex_dtype=complex_dtype,
    ).values


def aperture_batch_size(direction_count: int) -> int:
    """Return Section 3.3's internal direction batch ``B <= min(S, 256)``."""
    limit = min(direction_count, _MAX_BATCH)
    batch = 1
    while batch * 2 <= limit:
        batch *= 2
    return max(batch, 1)


def evaluate_aperture_response(
    spec: ApertureSpecification,
    altitude_rad: np.ndarray,
    azimuth_rad: np.ndarray,
    *,
    wavelength_m: float,
    real_dtype: np.dtype[Any],
    complex_dtype: np.dtype[Any],
) -> np.ndarray:
    """Return the scalar voltage response for one beam-frame direction batch.

    Section 3.3: for an outer resolved beam-frame direction,
    ``q_N = (2 pi / lambda) cos(alt) cos(az)`` and
    ``q_E = (2 pi / lambda) cos(alt) sin(az)``.  The existing pointing transform
    and true-horizon gate stay exactly where they are in the canonical beam
    runtime; this function sees the directions they produced.
    """
    altitude = np.asarray(altitude_rad, dtype=real_dtype)
    azimuth = np.asarray(azimuth_rad, dtype=real_dtype)
    if altitude.shape != azimuth.shape:
        raise ValueError("altitude_rad and azimuth_rad must have identical shapes")
    if altitude.size == 0:
        return np.zeros(0, dtype=complex_dtype)
    wavenumber = real_dtype.type(2.0) * _pi(real_dtype) / real_dtype.type(wavelength_m)
    cosine = np.cos(altitude)
    wavevectors = np.stack(
        (
            wavenumber * cosine * np.cos(azimuth),
            wavenumber * cosine * np.sin(azimuth),
        ),
        axis=1,
    )
    batch = aperture_batch_size(int(altitude.size))
    pieces: list[np.ndarray] = []
    for start in range(0, wavevectors.shape[0], batch):
        pieces.append(
            evaluate_aperture_transform(
                spec,
                wavevectors[start : start + batch],
                wavelength_m=wavelength_m,
                real_dtype=real_dtype,
                complex_dtype=complex_dtype,
            )
        )
    return np.concatenate(pieces)


# --- Section 3.4: Ruze coherent loss and scattered-power diagnostic ------------

#: Section 3.4.1's fixed Ruze resource caps.
_RUZE_MAX_POISSON_TERMS: Final[int] = 256
_RUZE_MAX_SOLVE_NODES: Final[int] = 2**18
_RUZE_MAX_PRESENTATIONS: Final[int] = 2**20
_RUZE_MAX_PHASE_PRODUCTS: Final[int] = 2**34
_RUZE_MAX_WORKSPACE_BYTES: Final[int] = 8 * 2**30

RUZE_UNOBSTRUCTED_MESSAGE: Final[str] = (
    "Stage-1 Ruze power diagnostics v1 require an unobstructed pupil; the "
    "resolved aperture physics declares a blockage."
)
RUZE_EMPTY_DIRECTION_MESSAGE: Final[str] = (
    "Ruze power diagnostic requires at least one direction."
)
RUZE_UNCONFIGURED_MESSAGE: Final[str] = (
    "A Ruze power diagnostic is not configured for this antenna."
)


def _encode_count(value: int) -> bytes:
    return int(value).to_bytes(8, "little", signed=False)


def _encode_bytes(value: bytes) -> bytes:
    return _encode_count(len(value)) + value


def _encode_float(value: float, real_dtype: np.dtype[Any]) -> bytes:
    """Encode one finite float at the declared little-endian real width."""
    normalized = 0.0 if value == 0.0 else float(value)
    scalar = np.asarray(normalized, dtype=real_dtype.newbyteorder("<"))
    return scalar.tobytes()


@dataclass(frozen=True, slots=True)
class _PoissonSupport:
    """The resolved contiguous Poisson interval and its exact tail masses."""

    first_order: int
    last_order: int
    term_count: int
    lower_omitted_mass: float
    upper_omitted_mass: float
    total_omitted_mass: float
    retained_weight_sum: float
    weights: tuple[float, ...]


def _poisson_terms(mu: float) -> list[float]:
    """Return ``p_m`` for ``m >= 1`` without forming ``mu**m / m!``."""
    if mu <= 0.0:
        return []
    log_mu = math.log(mu)
    terms: list[float] = []
    order = 1
    while True:
        value = math.exp(-mu + order * log_mu - math.lgamma(order + 1))
        terms.append(value)
        if order > mu and value < 1e-300:
            break
        order += 1
        if order > 100_000:  # pragma: no cover - defensive
            break
    return terms


def _resolve_poisson_support(mu: float, tau: float) -> _PoissonSupport:
    """Choose the fewest retained Poisson terms for Section 3.4.1's budget.

    Ties break toward the smaller first order.  The retained weights are never
    renormalized, and the total scattered mass is ``-expm1(-mu)`` rather than a
    sum, so the zero-term case is exact rather than a cancellation artefact.
    """
    total = -math.expm1(-mu)
    if total <= tau:
        return _PoissonSupport(0, 0, 0, 0.0, total, total, 0.0, ())
    terms = _poisson_terms(mu)
    count = len(terms)
    lower = [0.0] * (count + 2)
    for index in range(1, count + 1):
        lower[index] = math.fsum(terms[: index - 1])
    upper = [0.0] * (count + 2)
    for index in range(count, 0, -1):
        upper[index] = math.fsum(terms[index:])
    for retained in range(1, min(count, _RUZE_MAX_POISSON_TERMS) + 1):
        for first in range(1, count - retained + 2):
            last = first + retained - 1
            if lower[first] + upper[last] <= tau:
                mode = max(1, min(last, int(mu) if mu >= 1.0 else 1))
                mode = min(max(mode, first), last)
                ordered = sorted(
                    range(first, last + 1), key=lambda m: (abs(m - mode), m)
                )
                weight_sum = math.fsum(terms[m - 1] for m in ordered)
                return _PoissonSupport(
                    first,
                    last,
                    retained,
                    lower[first],
                    upper[last],
                    lower[first] + upper[last],
                    weight_sum,
                    tuple(terms[m - 1] for m in range(first, last + 1)),
                )
    raise BeamSamplingDerivationError(
        "Stage-1 Ruze power diagnostic requires more than "
        f"{_RUZE_MAX_POISSON_TERMS} retained Poisson terms for mu={mu!r}."
    )


@dataclass(frozen=True, slots=True)
class _SeparationPanel:
    """One separation radial panel and its endpoint transformation."""

    lower: float
    upper: float
    transform: Literal["linear", "outer_sqrt"]


def _separation_panels(
    diameter_m: float,
    separation_cut_m: float,
) -> tuple[_SeparationPanel, ...]:
    """Return Section 3.4.1's separation radial panels.

    When the cut reaches the pupil diameter, ``C`` reaches its outer zero like
    ``(D - delta)**1.5``, so the outer panel is presented through
    ``delta = D - (D/2)(1-t)**2`` rather than to Gauss-Legendre as a smooth
    function.
    """
    if separation_cut_m < diameter_m:
        return (_SeparationPanel(0.0, separation_cut_m, "linear"),)
    return (
        _SeparationPanel(0.0, 0.5 * diameter_m, "linear"),
        _SeparationPanel(0.5 * diameter_m, diameter_m, "outer_sqrt"),
    )


def _separation_angular_order(
    delta_m: float,
    *,
    q_max: float,
    kappa: float,
    surface_bound: float,
    radius_m: float,
    azimuthal_degree: int,
) -> int:
    """Return the equispaced trapezoid order for one separation radius.

    The rule is the smallest power of two not below Section 3.4.1's bound.  A
    periodic analytic integrand makes the trapezoid exponentially convergent,
    and the direction factor's aliasing error is exactly
    ``2 * sum_j |J_{jN}(q delta)|``, superexponentially small once
    ``N >= 3 q delta``.
    """
    bound = max(
        16,
        8
        + math.ceil(
            4.0
            * (
                q_max * delta_m
                + 2.0 * kappa * surface_bound * delta_m / radius_m
                + azimuthal_degree
            )
        ),
    )
    order = 16
    while order < bound:
        order *= 2
    return order


@dataclass(frozen=True, slots=True)
class _PairedNodes:
    """The paired-pupil quadrature nodes for one separation radius."""

    weight: Any
    rho: Any
    rho_shifted: Any
    phi_relative: Any
    phi_shifted_relative: Any
    boundary_radius: float
    panel_transforms: tuple[str, ...]
    node_count: int


def _paired_pupil_nodes(
    spec: ApertureSpecification,
    delta_m: float,
    *,
    real_dtype: np.dtype[Any],
    radial_factor: int,
    angular_factor: int,
    surface_bound: float,
    kappa: float,
) -> _PairedNodes | None:
    """Build the paired transmitting region's nodes at one separation radius.

    Section 3.4.1 fixes the region exactly: in the unshifted pupil's polar
    coordinates the paired set is
    ``rho in [max(0, delta/R - 1), 1]`` with ``|phi - psi| <= Phi(rho)``, where
    ``Phi = arccos((R^2 rho^2 + delta^2 - R^2) / (2 R rho delta))``.  There is
    exactly one transmitting angular interval at every interior radius, so this
    partition has no topology roots and no merge tolerance: the boundary
    ``b = |1 - delta/R|`` and the domain ends are its only breakpoints.
    """
    radius_m = spec.radius_m
    if delta_m >= spec.diameter_m:
        return None
    boundary = abs(1.0 - delta_m / radius_m)
    panels: list[tuple[float, float, str]] = []
    if delta_m < radius_m and boundary > 0.0:
        panels.append((0.0, boundary, "linear"))
    if boundary < 1.0:
        panels.append((boundary, 1.0, "boundary_sqrt"))
    if not panels:
        return None

    radial_degree = 2 * (spec.max_radial_order + 4)
    angular_degree = 2 * (spec.max_azimuthal_order + 1)
    bandwidth = 2.0 * kappa * surface_bound
    weights: list[Any] = []
    rho_all: list[Any] = []
    phi_all: list[Any] = []
    transforms: list[str] = []
    total_nodes = 0

    for lower, upper, transform in panels:
        transforms.append(transform)
        order = _check_panel_order(
            gauss_legendre_order_seed(bandwidth, upper - lower, radial_degree)
            * radial_factor
        )
        nodes, raw = _gauss_legendre(order, real_dtype)
        if transform == "linear":
            half = real_dtype.type(0.5 * (upper - lower))
            centre = real_dtype.type(0.5 * (upper + lower))
            radii = centre + half * nodes
            jacobian = half * raw
        else:
            anchor = real_dtype.type(lower)
            span = real_dtype.type(upper - lower)
            parameter = real_dtype.type(0.5) * (nodes + real_dtype.type(1.0))
            radii = anchor + span * parameter * parameter
            jacobian = (
                real_dtype.type(0.5) * raw * real_dtype.type(2.0) * span * parameter
            )
        argument = np.where(
            radii > real_dtype.type(0.0),
            (
                real_dtype.type(radius_m) ** 2 * radii * radii
                + real_dtype.type(delta_m) ** 2
                - real_dtype.type(radius_m) ** 2
            )
            / (
                real_dtype.type(2.0 * radius_m * delta_m)
                * np.maximum(radii, real_dtype.type(np.finfo(real_dtype).tiny))
            ),
            real_dtype.type(-1.0),
        )
        half_angle = np.arccos(
            np.clip(argument, real_dtype.type(-1.0), real_dtype.type(1.0))
        )
        for index in range(order):
            extent = float(half_angle[index])
            if extent <= 0.0:
                continue
            angular_order = _check_panel_order(
                gauss_legendre_order_seed(bandwidth, 2.0 * extent, angular_degree)
                * angular_factor
            )
            angular_nodes, angular_raw = _gauss_legendre(angular_order, real_dtype)
            scale = real_dtype.type(extent)
            phi_all.append(scale * angular_nodes)
            weights.append(jacobian[index] * radii[index] * scale * angular_raw)
            rho_all.append(np.full(angular_order, radii[index], dtype=real_dtype))
            total_nodes += angular_order

    if total_nodes == 0:
        return None
    if total_nodes > _RUZE_MAX_SOLVE_NODES:
        raise BeamSamplingDerivationError(
            f"Stage-1 Ruze paired-pupil solve requires {total_nodes} aperture "
            f"nodes, above the fixed cap {_RUZE_MAX_SOLVE_NODES}."
        )
    weight = np.concatenate(weights)
    rho = np.concatenate(rho_all)
    phi = np.concatenate(phi_all)
    # The shifted point in the frame rotated by psi: r - Delta with Delta along
    # the local x axis.  Both its radius and its relative angle are independent
    # of psi, which is what lets one node set serve every angular node.
    east = real_dtype.type(radius_m) * rho * np.cos(phi) - real_dtype.type(delta_m)
    north = real_dtype.type(radius_m) * rho * np.sin(phi)
    rho_shifted = np.hypot(east, north) / real_dtype.type(radius_m)
    phi_shifted = np.arctan2(north, east)
    return _PairedNodes(
        weight=weight,
        rho=rho,
        rho_shifted=rho_shifted,
        phi_relative=phi,
        phi_shifted_relative=phi_shifted,
        boundary_radius=boundary,
        panel_transforms=tuple(transforms),
        node_count=total_nodes,
    )


def _radial_polynomial(
    n: int,
    order: int,
    rho: Any,
    real_dtype: np.dtype[Any],
) -> Any:
    """Return ``R_n^{|m|}(rho)`` from its exact integer coefficients."""
    radius = np.asarray(rho, dtype=real_dtype)
    total = np.zeros(radius.shape, dtype=real_dtype)
    for power, coefficient in _radial_coefficients(n, order):
        total = total + real_dtype.type(coefficient) * radius**power
    return total


def _surface_components(
    spec: ApertureSpecification,
    rho: Any,
    phi_relative: Any,
    real_dtype: np.dtype[Any],
) -> tuple[Any, list[tuple[int, Any, Any]]]:
    """Split the surface height into psi-independent Fourier components.

    ``h(rho, psi + phi_rel)`` expands into ``cos(m psi)`` and ``sin(m psi)``
    parts whose coefficients depend only on the node, so the polynomial work is
    done once per separation radius and is reused for every angular node.  The
    expansion is exact, not an approximation: it is the angle-addition identity
    applied to Section 3.3's real basis.
    """
    constant = np.zeros(np.asarray(rho).shape, dtype=real_dtype)
    components: list[tuple[int, Any, Any]] = []
    for n, m, coefficient in spec.modes:
        order = abs(m)
        radial = _radial_polynomial(n, order, rho, real_dtype)
        normalization = (
            np.sqrt(real_dtype.type(n + 1))
            if m == 0
            else np.sqrt(real_dtype.type(2.0) * real_dtype.type(n + 1))
        )
        scale = real_dtype.type(coefficient) * normalization
        if m == 0:
            constant = constant + scale * radial
            continue
        angle = real_dtype.type(order) * np.asarray(phi_relative, dtype=real_dtype)
        components.append(
            (m, scale * radial * np.cos(angle), scale * radial * np.sin(angle))
        )
    return constant, components


def _surface_at(
    constant: Any,
    components: list[tuple[int, Any, Any]],
    psi: float,
    real_dtype: np.dtype[Any],
) -> Any:
    total = constant
    for m, cosine, sine in components:
        order = abs(m)
        angle = real_dtype.type(order) * real_dtype.type(psi)
        if m > 0:
            total = total + np.cos(angle) * cosine - np.sin(angle) * sine
        else:
            total = total + np.sin(angle) * cosine + np.cos(angle) * sine
    return total


def _autocorrelation(
    spec: ApertureSpecification,
    delta_m: float,
    psi_values: Any,
    *,
    real_dtype: np.dtype[Any],
    complex_dtype: np.dtype[Any],
    radial_factor: int,
    angular_factor: int,
    kappa: float,
    surface_bound: float,
    counters: dict[str, int],
) -> tuple[Any, _PairedNodes | None]:
    """Evaluate the paired-pupil autocorrelation ``C`` at one separation radius.

    ``C(Delta) = int f(r) f*(r - Delta) d^2r`` with
    ``f = A M exp(-i phi_det)``.  It carries no far-field oscillation, which is
    the whole point of the separation domain: one ``C`` array serves every
    retained Poisson order and every requested direction.  At ``delta = D`` the
    paired region is empty and ``C`` is exactly ``0+0j`` with positive-zero
    components.
    """
    psi = np.asarray(psi_values, dtype=real_dtype)
    nodes = _paired_pupil_nodes(
        spec,
        delta_m,
        real_dtype=real_dtype,
        radial_factor=radial_factor,
        angular_factor=angular_factor,
        surface_bound=surface_bound,
        kappa=kappa,
    )
    if nodes is None:
        return np.zeros(psi.shape, dtype=complex_dtype), None

    counters["presentations"] = counters.get("presentations", 0) + int(psi.size)
    counters["phases"] = counters.get("phases", 0) + int(psi.size) * nodes.node_count
    counters["nodes"] = max(counters.get("nodes", 0), nodes.node_count)
    if counters["presentations"] > _RUZE_MAX_PRESENTATIONS:
        raise BeamSamplingDerivationError(
            "Stage-1 Ruze power diagnostic exceeded the fixed "
            f"{_RUZE_MAX_PRESENTATIONS} separation-node presentation cap."
        )
    if counters["phases"] > _RUZE_MAX_PHASE_PRODUCTS:
        raise BeamSamplingDerivationError(
            "Stage-1 Ruze power diagnostic exceeded the fixed "
            f"{_RUZE_MAX_PHASE_PRODUCTS} phase-product cap."
        )

    profile = _profile(spec, nodes.rho, real_dtype) * _profile(
        spec, nodes.rho_shifted, real_dtype
    )
    amplitude = nodes.weight * profile * real_dtype.type(spec.radius_m) ** 2
    if not spec.modes:
        value = complex_dtype.type(np.sum(amplitude))
        return np.full(psi.shape, value, dtype=complex_dtype), nodes

    constant, components = _surface_components(
        spec, nodes.rho, nodes.phi_relative, real_dtype
    )
    shifted_constant, shifted_components = _surface_components(
        spec, nodes.rho_shifted, nodes.phi_shifted_relative, real_dtype
    )
    if spec.max_azimuthal_order == 0:
        # An axisymmetric surface makes C independent of psi, so one evaluation
        # is the whole answer.  Section 3.4.2's counts are definitional, so the
        # presentations above are still charged in full.
        difference = constant - shifted_constant
        value = complex_dtype.type(
            np.sum(amplitude * np.exp(-1j * (real_dtype.type(kappa) * difference)))
        )
        return np.full(psi.shape, value, dtype=complex_dtype), nodes

    values = np.empty(psi.shape, dtype=complex_dtype)
    for index in range(psi.size):
        angle = float(psi[index])
        difference = _surface_at(constant, components, angle, real_dtype) - _surface_at(
            shifted_constant, shifted_components, angle, real_dtype
        )
        values[index] = np.sum(
            amplitude * np.exp(-1j * (real_dtype.type(kappa) * difference))
        )
    return values, nodes


@dataclass(frozen=True, slots=True)
class _SeparationGrid:
    """The resolved separation partition and its per-node angular trapezoids."""

    panels: tuple[_SeparationPanel, ...]
    radial_order: int
    deltas: tuple[float, ...]
    weights: tuple[float, ...]
    angular_orders: tuple[int, ...]
    node_count: int

    @property
    def angular_order_max(self) -> int:
        return max(self.angular_orders, default=0)


def _separation_grid(
    spec: ApertureSpecification,
    panels: tuple[_SeparationPanel, ...],
    *,
    radial_factor: int,
    angular_factor: int,
    bandwidth_delta: float,
    q_max: float,
    kappa: float,
    surface_bound: float,
    real_dtype: np.dtype[Any],
) -> _SeparationGrid:
    """Build the separation partition at one refinement level."""
    radial_degree = 2 * (spec.max_radial_order + 4)
    deltas: list[float] = []
    weights: list[float] = []
    orders: list[int] = []
    resolved_order = 0
    for panel in panels:
        order = _check_panel_order(
            gauss_legendre_order_seed(
                bandwidth_delta, panel.upper - panel.lower, radial_degree
            )
            * radial_factor
        )
        resolved_order = max(resolved_order, order)
        nodes, raw = _gauss_legendre(order, real_dtype)
        if panel.transform == "linear":
            half = 0.5 * (panel.upper - panel.lower)
            centre = 0.5 * (panel.upper + panel.lower)
            for index in range(order):
                deltas.append(centre + half * float(nodes[index]))
                weights.append(half * float(raw[index]))
        else:
            diameter = spec.diameter_m
            for index in range(order):
                parameter = 0.5 * (float(nodes[index]) + 1.0)
                deltas.append(diameter - 0.5 * diameter * (1.0 - parameter) ** 2)
                weights.append(0.5 * float(raw[index]) * diameter * (1.0 - parameter))
    for delta in deltas:
        orders.append(
            _separation_angular_order(
                delta,
                q_max=q_max,
                kappa=kappa,
                surface_bound=surface_bound,
                radius_m=spec.radius_m,
                azimuthal_degree=spec.max_azimuthal_order,
            )
            * angular_factor
        )
    return _SeparationGrid(
        panels=panels,
        radial_order=resolved_order,
        deltas=tuple(deltas),
        weights=tuple(weights),
        angular_orders=tuple(orders),
        node_count=sum(orders),
    )


def _nested_psi_keys(order: int) -> list[tuple[int, int]]:
    """Return the reduced ``(numerator, denominator)`` key of each psi node.

    The equispaced trapezoid nests: at order ``2N`` every even index reproduces
    an order-``N`` node exactly.  Reducing ``j/N`` to lowest terms makes that
    identity exact in the cache key rather than dependent on float rounding, so
    a doubled angular level re-uses every separation node already evaluated and
    Section 3.4.2's count increments only for the nodes it actually adds.
    """
    keys: list[tuple[int, int]] = []
    for index in range(order):
        divisor = math.gcd(index, order)
        keys.append((index // divisor, order // divisor))
    return keys


def _autocorrelation_batch(
    spec: ApertureSpecification,
    delta_index: int,
    delta_m: float,
    order: int,
    *,
    cache: dict[tuple[int, int, int], complex],
    real_dtype: np.dtype[Any],
    complex_dtype: np.dtype[Any],
    paired_radial_factor: int,
    paired_angular_factor: int,
    kappa: float,
    surface_bound: float,
    counters: dict[str, int],
    batch_size: int,
    boundaries: list[tuple[float, tuple[str, ...]]] | None,
) -> np.ndarray:
    """Return ``C`` at every psi node of one separation radius, using the cache."""
    keys = _nested_psi_keys(order)
    missing = [
        index
        for index, key in enumerate(keys)
        if (delta_index, key[0], key[1]) not in cache
    ]
    recorded = False
    for start in range(0, len(missing), max(batch_size, 1)):
        chunk = missing[start : start + max(batch_size, 1)]
        psi = np.asarray(
            [2.0 * math.pi * index / order for index in chunk], dtype=np.float64
        )
        values, nodes = _autocorrelation(
            spec,
            delta_m,
            psi,
            real_dtype=real_dtype,
            complex_dtype=complex_dtype,
            radial_factor=paired_radial_factor,
            angular_factor=paired_angular_factor,
            kappa=kappa,
            surface_bound=surface_bound,
            counters=counters,
        )
        if boundaries is not None and not recorded:
            boundaries.append(
                (0.0, ())
                if nodes is None
                else (nodes.boundary_radius, nodes.panel_transforms)
            )
            recorded = True
        for offset, index in enumerate(chunk):
            key = keys[index]
            cache[(delta_index, key[0], key[1])] = complex(values[offset])
    if boundaries is not None and not recorded:
        nodes = _paired_pupil_nodes(
            spec,
            delta_m,
            real_dtype=real_dtype,
            radial_factor=paired_radial_factor,
            angular_factor=paired_angular_factor,
            surface_bound=surface_bound,
            kappa=kappa,
        )
        boundaries.append(
            (0.0, ())
            if nodes is None
            else (nodes.boundary_radius, nodes.panel_transforms)
        )
    return np.asarray(
        [cache[(delta_index, key[0], key[1])] for key in keys], dtype=np.complex128
    )


def _assemble_mixture(
    spec: ApertureSpecification,
    grid: _SeparationGrid,
    *,
    wavevectors: np.ndarray,
    lengths: tuple[float, ...],
    paired_radial_factor: int,
    paired_angular_factor: int,
    kappa: float,
    surface_bound: float,
    real_dtype: np.dtype[Any],
    complex_dtype: np.dtype[Any],
    counters: dict[str, int],
    cache: dict[tuple[int, int, int], complex],
    batch_size: int,
    boundaries: list[tuple[float, tuple[str, ...]]] | None = None,
) -> tuple[np.ndarray, float]:
    """Return ``P_m(q)`` for every retained order and direction, plus |Im| max.

    Each assembled separation integral is real because
    ``C(-Delta) = C*(Delta)``; the largest absolute imaginary part actually
    formed is returned so the caller can hold it to the frozen predicate rather
    than silently discard it.
    """
    normalization = (math.pi * spec.radius_m**2) ** 2
    accumulator = np.zeros((len(lengths), wavevectors.shape[0]), dtype=np.complex128)
    q_north = np.asarray(wavevectors[:, 0], dtype=np.float64)
    q_east = np.asarray(wavevectors[:, 1], dtype=np.float64)
    for index, delta in enumerate(grid.deltas):
        order = grid.angular_orders[index]
        values = _autocorrelation_batch(
            spec,
            index,
            delta,
            order,
            cache=cache,
            real_dtype=real_dtype,
            complex_dtype=complex_dtype,
            paired_radial_factor=paired_radial_factor,
            paired_angular_factor=paired_angular_factor,
            kappa=kappa,
            surface_bound=surface_bound,
            counters=counters,
            batch_size=batch_size,
            boundaries=boundaries,
        )
        psi = 2.0 * math.pi * np.arange(order, dtype=np.float64) / order
        trapezoid = 2.0 * math.pi / order
        radial_weight = grid.weights[index] * delta * trapezoid
        phase = np.exp(
            -1j
            * (
                q_north[:, None] * (delta * np.cos(psi))[None, :]
                + q_east[:, None] * (delta * np.sin(psi))[None, :]
            )
        )
        angular = phase @ values
        counters["phases"] = counters.get("phases", 0) + int(phase.size)
        for order_index, length in enumerate(lengths):
            accumulator[order_index] += (
                radial_weight * math.exp(-((delta / length) ** 2))
            ) * angular
    accumulator /= normalization
    imaginary = float(np.max(np.abs(accumulator.imag))) if accumulator.size else 0.0
    return np.asarray(accumulator.real, dtype=np.float64), imaginary


def _aperture_topology_manifest(
    spec: ApertureSpecification,
    panels: tuple[_RadialPanel, ...],
    bandwidths: _Bandwidths,
    radial_factor: int,
    angular_factor: int,
    real_dtype: np.dtype[Any],
) -> tuple[bytes, int, int]:
    """Build Section 3.4.2's base coherent aperture-topology byte stream."""
    stream = bytearray(b"radiosim.aperture_topology.v1\x00")
    stream += _encode_bytes(real_dtype.name.encode("ascii"))
    if spec.epsilon is None:
        stream += _encode_bytes(b"none")
    else:
        stream += _encode_bytes(b"ratio") + _encode_float(spec.epsilon, real_dtype)
    stream += _encode_count(len(spec.legs))
    for beta, ratio in spec.legs:
        stream += _encode_float(beta, real_dtype) + _encode_float(ratio, real_dtype)
    live = [
        panel
        for panel in panels
        if _transmitting_intervals(spec, 0.5 * (panel.lower + panel.upper))
    ]
    stream += _encode_count(len(live))
    partition_count = 0
    for panel in live:
        transform = b"linear" if panel.saturation_anchor is None else b"saturation_sqrt"
        stream += _encode_bytes(transform)
        stream += _encode_float(panel.lower, real_dtype)
        stream += _encode_float(panel.upper, real_dtype)
        partition_count += len(
            _transmitting_intervals(spec, 0.5 * (panel.lower + panel.upper))
        )
    degree = spec.max_radial_order
    for panel in live:
        order = _panel_radial_seed(panel, bandwidths, degree) * radial_factor
        nodes, _raw = _gauss_legendre(order, real_dtype)
        if panel.saturation_anchor is None:
            half = 0.5 * (panel.upper - panel.lower)
            centre = 0.5 * (panel.upper + panel.lower)
            radii = [centre + half * float(node) for node in nodes]
        else:
            anchor = panel.saturation_anchor
            span = panel.upper - panel.saturation_anchor
            radii = [anchor + span * (0.5 * (float(node) + 1.0)) ** 2 for node in nodes]
        stream += _encode_count(len(radii))
        for radius in radii:
            intervals = _transmitting_intervals(spec, radius)
            stream += _encode_count(len(intervals))
            for start, end in intervals:
                stream += _encode_float(start, real_dtype)
                stream += _encode_float(end, real_dtype)
    del angular_factor
    boundaries = {panel.lower for panel in panels} | {panels[-1].upper}
    return bytes(stream), partition_count, len(boundaries)


def _separation_topology_manifest(
    grid: _SeparationGrid | None,
    separation_cut_m: float,
    boundaries: list[tuple[float, tuple[str, ...]]],
    real_dtype: np.dtype[Any],
) -> bytes:
    """Build Section 3.4.2's separation-partition byte stream."""
    stream = bytearray(b"radiosim.ruze_separation_partition.v1\x00")
    stream += _encode_bytes(real_dtype.name.encode("ascii"))
    if grid is None:
        # The resolved zero-term Poisson case digests the domain prefix, the
        # real-dtype literal, and zero counts.
        stream += _encode_count(0) + _encode_count(0) + _encode_count(0)
        return bytes(stream)
    stream += _encode_float(separation_cut_m, real_dtype)
    stream += _encode_count(len(grid.panels))
    for panel in grid.panels:
        stream += _encode_bytes(panel.transform.encode("ascii"))
        stream += _encode_float(panel.lower, real_dtype)
        stream += _encode_float(panel.upper, real_dtype)
    stream += _encode_count(len(grid.angular_orders))
    for order in grid.angular_orders:
        stream += _encode_count(order)
    stream += _encode_count(len(boundaries))
    for boundary, transforms in boundaries:
        stream += _encode_float(boundary, real_dtype)
        stream += _encode_count(len(transforms))
        for transform in transforms:
            stream += _encode_bytes(transform.encode("ascii"))
    return bytes(stream)


@dataclass(frozen=True, slots=True)
class RuzePowerConvergence:
    """Section 3.4.2's frozen convergence record, in its declared field order.

    Every retained residual is separate: the Poisson tail, the separation
    truncation bound, the two successive separation residuals, the paired-pupil
    and base coherent aperture residuals, and the imaginary residual of the
    assembled real integral.  There is deliberately no ``converged`` boolean and
    no false state -- a failure returns no record at all.
    """

    real_dtype: str
    complex_dtype: str
    poisson_mu: float
    poisson_first_order: int
    poisson_last_order: int
    poisson_term_count: int
    poisson_lower_omitted_mass: float
    poisson_upper_omitted_mass: float
    poisson_total_omitted_mass: float
    poisson_retained_weight_sum: float
    separation_cut_m: float
    separation_omitted_bound: float
    separation_radial_order: int
    separation_angular_order_max: int
    separation_node_count: int
    separation_evaluation_count: int
    separation_penultimate_max_abs_delta: float
    separation_final_max_abs_delta: float
    separation_imaginary_max_abs_residual: float
    separation_topology_sha256: str
    aperture_method: str
    aperture_partition_count: int
    aperture_topology_breakpoint_count: int
    aperture_topology_sha256: str
    aperture_refinement_count: int
    aperture_max_node_count: int
    aperture_penultimate_max_abs_delta: float
    aperture_final_max_abs_delta: float
    aperture_q_max: float
    surface_phase_kappa: float
    surface_radial_derivative_bound: float
    surface_angular_derivative_bound: float
    fhat_evaluation_count: int
    phase_product_count: int
    batch_size: int
    atol: float
    rtol: float
    estimated_peak_bytes: int
    maximum_abs_e_deterministic: float
    minimum_scattered_power: float
    maximum_total_power: float
    returned_balance_max_abs_residual: float


@dataclass(frozen=True, slots=True)
class RuzePowerDiagnostic:
    """Section 3.4.2's frozen public ensemble-power record.

    It reports coherent-main power, total ensemble power, and their
    non-negative scattered difference at every requested direction.  It is not a
    Jones voltage: ``sqrt(B_main + B_error)`` would invent a phase and perfectly
    correlated structure, so no complex field is derived from it and no
    cross-baseline visibility ever changes because of it.
    """

    schema_version: str
    method: str
    antenna_id: Any
    covariance_convention: str
    normalization_convention: str
    frequency_hz: float
    time_mjd: float
    rms_surface_error_m: float
    correlation_length_m: float
    altitude_rad: np.ndarray
    azimuth_rad: np.ndarray
    coherent_main_power: np.ndarray
    total_ensemble_power: np.ndarray
    scattered_power: np.ndarray
    convergence: RuzePowerConvergence


def _reject_mutation(self: Any, name: str, value: Any = None) -> None:
    """Refuse every assignment on a frozen Stage-1 record.

    ``@dataclass(frozen=True, slots=True)`` only raises ``FrozenInstanceError``
    for a *declared field*; an unknown name falls through to a stale
    ``super()`` binding left behind when the slots decorator recreated the
    class.  Section 3.4.2 says the records are frozen, so every name is
    refused, and refused with the same typed error.
    """
    raise dataclasses.FrozenInstanceError(
        f"cannot assign to {name!r}: this Stage-1 record is frozen"
    )


for _frozen_record in (RuzePowerConvergence, RuzePowerDiagnostic):
    _frozen_record.__setattr__ = _reject_mutation  # type: ignore[method-assign]
    _frozen_record.__delattr__ = _reject_mutation  # type: ignore[method-assign]


def _owned_read_only(values: Any, dtype: np.dtype[Any]) -> np.ndarray:
    array = np.array(values, dtype=dtype, copy=True, order="C")
    array.setflags(write=False)
    return array


def _paired_pupil_converged_orders(
    spec: ApertureSpecification,
    grid: _SeparationGrid,
    *,
    kappa: float,
    surface_bound: float,
    real_dtype: np.dtype[Any],
    complex_dtype: np.dtype[Any],
    atol: float,
    rtol: float,
    counters: dict[str, int],
    batch_size: int,
) -> tuple[int, int, float, float]:
    """Converge the paired-pupil orders once against the seed partition.

    Section 3.4.1: the accepted order is then held while the separation
    partition itself is refined, and the two residual sequences are retained
    separately.  The complete array compared here is ``C`` over every separation
    node of this partition -- never a scalar and never the weighted sum.
    """

    def sample(radial_factor: int, angular_factor: int) -> np.ndarray:
        cache: dict[tuple[int, int, int], complex] = {}
        values: list[complex] = []
        for index, delta in enumerate(grid.deltas):
            evaluated = _autocorrelation_batch(
                spec,
                index,
                delta,
                grid.angular_orders[index],
                cache=cache,
                real_dtype=real_dtype,
                complex_dtype=complex_dtype,
                paired_radial_factor=radial_factor,
                paired_angular_factor=angular_factor,
                kappa=kappa,
                surface_bound=surface_bound,
                counters=counters,
                batch_size=batch_size,
                boundaries=None,
            )
            values.extend(evaluated.tolist())
        return np.asarray(values, dtype=np.complex128)

    def angular_converged(radial_factor: int) -> tuple[np.ndarray, int, float, float]:
        angular_factor = 1
        previous = sample(radial_factor, angular_factor)
        successes = 0
        deltas: list[float] = []
        for _ in range(_MAX_DOUBLINGS):
            angular_factor *= 2
            refined = sample(radial_factor, angular_factor)
            ok, delta = _converged(previous, refined, atol, rtol)
            if ok:
                deltas.append(delta)
                successes += 1
            else:
                deltas.clear()
                successes = 0
            previous = refined
            if successes >= 2:
                return previous, angular_factor, deltas[-2], deltas[-1]
        raise BeamSamplingDerivationError(
            "Stage-1 Ruze paired-pupil angular quadrature did not reach two "
            "consecutive successful comparisons within the permitted doublings."
        )

    radial_factor = 1
    previous, angular_factor, penultimate, final = angular_converged(radial_factor)
    successes = 0
    radial_deltas: list[float] = []
    for _ in range(_MAX_DOUBLINGS):
        radial_factor *= 2
        refined, angular_factor, penultimate, final = angular_converged(radial_factor)
        ok, delta = _converged(previous, refined, atol, rtol)
        if ok:
            radial_deltas.append(delta)
            successes += 1
        else:
            radial_deltas.clear()
            successes = 0
        previous = refined
        if successes >= 2:
            return (
                radial_factor,
                angular_factor,
                max(penultimate, radial_deltas[-2]),
                max(final, radial_deltas[-1]),
            )
    raise BeamSamplingDerivationError(
        "Stage-1 Ruze paired-pupil radial quadrature did not reach two "
        "consecutive successful comparisons within the permitted doublings."
    )


def evaluate_ruze_power_diagnostic(
    spec: ApertureSpecification,
    *,
    antenna_id: Any,
    altitude_rad: np.ndarray,
    azimuth_rad: np.ndarray,
    frequency_hz: float,
    time_mjd: float,
    rms_surface_error_m: float,
    correlation_length_m: float,
    real_dtype: np.dtype[Any],
    complex_dtype: np.dtype[Any],
) -> RuzePowerDiagnostic:
    """Evaluate Section 3.4's ensemble-power diagnostic in the separation domain.

    The production method is ``poisson_paired_pupil_separation_v1``: the
    positive Poisson mixture is evaluated as

    .. math::

        P_m(\\mathbf q)=\\frac{1}{|N_0|^2}\\int_{\\mathbb R^2}
        C(\\boldsymbol\\Delta)\\,e^{-i\\mathbf q\\cdot\\boldsymbol\\Delta}\\,
        e^{-|\\boldsymbol\\Delta|^2/\\ell_m^2}\\,d^2\\Delta,

    the exact separation-variable form of the shifted-wavevector integral, with
    ``C`` the paired-pupil autocorrelation.  No rule here scales with ``D/L``:
    ``C`` carries no far-field oscillation, the Gaussian confines the separation
    to a few ``ell_m``, and one ``C`` array serves every retained order and
    every requested direction.

    Cost note.  When every resolved Zernike mode has ``m == 0`` the surface is
    axisymmetric, so ``C(delta, psi)`` does not depend on ``psi`` and one
    evaluation per separation radius answers the whole angular trapezoid.  A
    mode with ``m != 0`` removes that identity and the work grows by the angular
    order.  Section 3.4.2's counts are *definitional* rather than a record of
    cache hits, so the caps are charged in full either way, and a sufficiently
    aberrated asymmetric surface reaches the ``2**34`` phase-product cap and
    fails closed with :class:`BeamSamplingDerivationError` rather than returning
    a partial result.  That is the designed behaviour, not a limitation to work
    around.
    """
    if spec.epsilon is not None or spec.legs:
        raise BeamSamplingDerivationError(RUZE_UNOBSTRUCTED_MESSAGE)
    altitude = np.asarray(altitude_rad, dtype=np.float64)
    azimuth = np.asarray(azimuth_rad, dtype=np.float64)
    if altitude.size == 0:
        raise BeamAngularDomainError(RUZE_EMPTY_DIRECTION_MESSAGE)

    atol, rtol = _tolerances(real_dtype)
    wavelength_m = _SPEED_OF_LIGHT_M_PER_S / float(frequency_hz)
    kappa = 4.0 * math.pi / wavelength_m
    mu = (kappa * float(rms_surface_error_m)) ** 2
    support = _resolve_poisson_support(mu, atol / 8.0)

    counters: dict[str, int] = {
        "nodes": 0,
        "phases": 0,
        "presentations": 0,
        "fhat": 0,
    }
    wavenumber = 2.0 * math.pi / wavelength_m
    cosine = np.cos(altitude)
    wavevectors = np.stack(
        (wavenumber * cosine * np.cos(azimuth), wavenumber * cosine * np.sin(azimuth)),
        axis=1,
    )
    base = solve_aperture_transform(
        spec,
        wavevectors,
        wavelength_m=wavelength_m,
        real_dtype=real_dtype,
        complex_dtype=complex_dtype,
        budget=counters,
    )
    if counters["nodes"] > _RUZE_MAX_SOLVE_NODES:
        raise BeamSamplingDerivationError(
            f"Stage-1 Ruze base coherent solve used {counters['nodes']} aperture "
            f"nodes, above the fixed cap {_RUZE_MAX_SOLVE_NODES}."
        )
    e_det = np.asarray(base.values, dtype=np.complex128)
    coherent = math.exp(-mu) * np.abs(e_det) ** 2
    bandwidths = _bandwidths(
        spec, np.asarray(wavevectors, dtype=real_dtype), wavelength_m
    )
    surface_bound = bandwidths.surface_radial + bandwidths.surface_angular
    panels = _radial_panels(spec, real_dtype)
    aperture_stream, partition_count, breakpoint_count = _aperture_topology_manifest(
        spec,
        panels,
        bandwidths,
        base.radial_factor,
        base.angular_factor,
        real_dtype,
    )

    grid: _SeparationGrid | None = None
    boundaries: list[tuple[float, tuple[str, ...]]] = []
    separation_cut_m = 0.0
    separation_omitted = 0.0
    separation_penultimate = 0.0
    separation_final = 0.0
    imaginary_residual = 0.0
    paired_penultimate = 0.0
    paired_final = 0.0
    batch_size = 0
    scattered = np.zeros(altitude.shape, dtype=np.float64)

    if support.term_count > 0:
        wide = float(correlation_length_m) / math.sqrt(support.first_order)
        narrow = float(correlation_length_m) / math.sqrt(support.last_order)
        tau_s = atol / 8.0
        cut = wide * math.sqrt(math.log(1.0 / tau_s))
        # The identity exp(-(cut/wide)**2) == tau_S is exact in real arithmetic
        # but the square root and the squaring each round, so the *realized*
        # bound can land one ulp above tau_S.  Nudging the cut outward discards
        # strictly less, which is what makes the retained bound a bound rather
        # than an approximation of one.
        for _ in range(64):
            if math.exp(-((cut / wide) ** 2)) <= tau_s:
                break
            cut = math.nextafter(cut, math.inf)
        separation_cut_m = min(spec.diameter_m, cut)
        separation_omitted = (
            0.0
            if separation_cut_m >= spec.diameter_m
            else math.exp(-((separation_cut_m / wide) ** 2))
        )
        separation_panels = _separation_panels(spec.diameter_m, separation_cut_m)
        q_max = float(np.max(np.sqrt(np.sum(np.asarray(wavevectors) ** 2, axis=1))))
        bandwidth_delta = (
            q_max + 2.0 / narrow + (2.0 * kappa * surface_bound + 2.0) / spec.radius_m
        )
        lengths = tuple(
            float(correlation_length_m) / math.sqrt(order)
            for order in range(support.first_order, support.last_order + 1)
        )
        seed_grid = _separation_grid(
            spec,
            separation_panels,
            radial_factor=1,
            angular_factor=1,
            bandwidth_delta=bandwidth_delta,
            q_max=q_max,
            kappa=kappa,
            surface_bound=surface_bound,
            real_dtype=real_dtype,
        )
        batch_size = 1
        while batch_size * 2 <= min(_MAX_BATCH, max(seed_grid.angular_orders)):
            batch_size *= 2
        paired_radial, paired_angular, paired_penultimate, paired_final = (
            _paired_pupil_converged_orders(
                spec,
                seed_grid,
                kappa=kappa,
                surface_bound=surface_bound,
                real_dtype=real_dtype,
                complex_dtype=complex_dtype,
                atol=atol,
                rtol=rtol,
                counters=counters,
                batch_size=batch_size,
            )
        )

        caches: dict[int, dict[tuple[int, int, int], complex]] = {}

        def mixture(
            radial_factor: int, angular_factor: int
        ) -> tuple[
            np.ndarray, float, _SeparationGrid, list[tuple[float, tuple[str, ...]]]
        ]:
            level = _separation_grid(
                spec,
                separation_panels,
                radial_factor=radial_factor,
                angular_factor=angular_factor,
                bandwidth_delta=bandwidth_delta,
                q_max=q_max,
                kappa=kappa,
                surface_bound=surface_bound,
                real_dtype=real_dtype,
            )
            collected: list[tuple[float, tuple[str, ...]]] = []
            values, imaginary = _assemble_mixture(
                spec,
                level,
                wavevectors=np.asarray(wavevectors),
                lengths=lengths,
                paired_radial_factor=paired_radial,
                paired_angular_factor=paired_angular,
                kappa=kappa,
                surface_bound=surface_bound,
                real_dtype=real_dtype,
                complex_dtype=complex_dtype,
                counters=counters,
                cache=caches.setdefault(radial_factor, {}),
                batch_size=batch_size,
                boundaries=collected,
            )
            return values, imaginary, level, collected

        quarter_atol = atol / 4.0
        quarter_rtol = rtol / 4.0

        def angular_converged(
            radial_factor: int,
        ) -> tuple[np.ndarray, float, _SeparationGrid, list[Any], float, float]:
            angular_factor = 1
            previous, imaginary, level, collected = mixture(
                radial_factor, angular_factor
            )
            successes = 0
            deltas: list[float] = []
            for _ in range(_MAX_DOUBLINGS):
                angular_factor *= 2
                refined, imaginary, level, collected = mixture(
                    radial_factor, angular_factor
                )
                ok, delta = _converged(previous, refined, quarter_atol, quarter_rtol)
                if ok:
                    deltas.append(delta)
                    successes += 1
                else:
                    deltas.clear()
                    successes = 0
                previous = refined
                if successes >= 2:
                    return previous, imaginary, level, collected, deltas[-2], deltas[-1]
            raise BeamSamplingDerivationError(
                "Stage-1 Ruze separation angular quadrature did not reach two "
                "consecutive successful comparisons within the permitted doublings."
            )

        radial_factor = 1
        previous, imaginary_residual, grid, boundaries, sep_pen, sep_fin = (
            angular_converged(radial_factor)
        )
        successes = 0
        radial_deltas: list[float] = []
        for _ in range(_MAX_DOUBLINGS):
            radial_factor *= 2
            refined, imaginary_residual, grid, boundaries, sep_pen, sep_fin = (
                angular_converged(radial_factor)
            )
            ok, delta = _converged(previous, refined, quarter_atol, quarter_rtol)
            if ok:
                radial_deltas.append(delta)
                successes += 1
            else:
                radial_deltas.clear()
                successes = 0
            previous = refined
            if successes >= 2:
                break
        else:
            raise BeamSamplingDerivationError(
                "Stage-1 Ruze separation radial quadrature did not reach two "
                "consecutive successful comparisons within the permitted doublings."
            )
        separation_penultimate = max(sep_pen, radial_deltas[-2])
        separation_final = max(sep_fin, radial_deltas[-1])
        weights = np.asarray(support.weights, dtype=np.float64)
        scattered = np.asarray(weights[:, None] * previous, dtype=np.float64).sum(
            axis=0
        )

    if not np.all(np.isfinite(scattered)) or bool(np.any(scattered < 0.0)):
        raise BeamSamplingDerivationError(
            "Stage-1 Ruze scattered power is negative or non-finite; there is no "
            "clipping, so this is an internal numerical failure."
        )
    limit = atol + rtol * (float(np.max(np.abs(scattered))) if scattered.size else 0.0)
    if imaginary_residual > limit:
        raise BeamSamplingDerivationError(
            "Stage-1 Ruze assembled separation integral carried an imaginary "
            f"residual {imaginary_residual!r} above {limit!r}."
        )
    if support.term_count == 0:
        scattered = np.zeros(altitude.shape, dtype=np.float64)

    real_array = np.dtype(real_dtype)
    coherent_out = _owned_read_only(coherent, real_array)
    scattered_out = _owned_read_only(scattered, real_array)
    total_out = _owned_read_only(coherent_out + scattered_out, real_array)
    balance = float(
        np.max(np.abs(total_out - (coherent_out + scattered_out)))
        if total_out.size
        else 0.0
    )
    maximum_abs = float(np.max(np.abs(e_det))) if e_det.size else 0.0
    tolerance = atol + rtol
    if maximum_abs > 1.0 + tolerance:
        raise BeamSamplingDerivationError(
            f"Stage-1 Ruze deterministic amplitude {maximum_abs!r} exceeds one."
        )
    maximum_total = float(np.max(total_out)) if total_out.size else 0.0
    if maximum_total > 1.0 + tolerance:
        raise BeamSamplingDerivationError(
            f"Stage-1 Ruze total ensemble power {maximum_total!r} exceeds one."
        )

    real_bytes = max(8, real_array.itemsize)
    complex_bytes = max(16, np.dtype(complex_dtype).itemsize)
    node_count = counters["nodes"]
    directions = int(altitude.size)
    separation_nodes = 0 if grid is None else grid.node_count
    estimated = real_bytes * (
        16 * node_count
        + 8 * max(batch_size, 1) * node_count
        + 16 * max(batch_size, 1)
        + 12 * directions
        + 4 * max(support.term_count, 1)
        + 6 * max(separation_nodes, 1)
    ) + complex_bytes * (
        4 * max(batch_size, 1) * node_count
        + 8 * max(batch_size, 1)
        + 6 * directions
        + 4 * max(separation_nodes, 1)
    )
    if estimated > _RUZE_MAX_WORKSPACE_BYTES:
        raise BeamSamplingDerivationError(
            f"Stage-1 Ruze workspace estimate {estimated} exceeds the fixed cap "
            f"{_RUZE_MAX_WORKSPACE_BYTES}."
        )

    convergence = RuzePowerConvergence(
        real_dtype=real_array.name,
        complex_dtype=np.dtype(complex_dtype).name,
        poisson_mu=mu,
        poisson_first_order=support.first_order,
        poisson_last_order=support.last_order,
        poisson_term_count=support.term_count,
        poisson_lower_omitted_mass=support.lower_omitted_mass,
        poisson_upper_omitted_mass=support.upper_omitted_mass,
        poisson_total_omitted_mass=support.total_omitted_mass,
        poisson_retained_weight_sum=support.retained_weight_sum,
        separation_cut_m=separation_cut_m,
        separation_omitted_bound=separation_omitted,
        separation_radial_order=0 if grid is None else grid.radial_order,
        separation_angular_order_max=0 if grid is None else grid.angular_order_max,
        separation_node_count=separation_nodes,
        separation_evaluation_count=counters["presentations"],
        separation_penultimate_max_abs_delta=separation_penultimate,
        separation_final_max_abs_delta=separation_final,
        separation_imaginary_max_abs_residual=imaginary_residual,
        separation_topology_sha256=hashlib.sha256(
            _separation_topology_manifest(
                grid, separation_cut_m, boundaries, real_array
            )
        ).hexdigest(),
        aperture_method="boundary_fitted_polar_gauss_legendre_v1",
        aperture_partition_count=partition_count,
        aperture_topology_breakpoint_count=breakpoint_count,
        aperture_topology_sha256=hashlib.sha256(aperture_stream).hexdigest(),
        aperture_refinement_count=base.refinement_count,
        aperture_max_node_count=counters["nodes"],
        aperture_penultimate_max_abs_delta=max(
            base.penultimate_max_abs_delta, paired_penultimate
        ),
        aperture_final_max_abs_delta=max(base.final_max_abs_delta, paired_final),
        aperture_q_max=base.q_max,
        surface_phase_kappa=kappa,
        surface_radial_derivative_bound=bandwidths.surface_radial,
        surface_angular_derivative_bound=bandwidths.surface_angular,
        fhat_evaluation_count=counters["fhat"],
        phase_product_count=counters["phases"],
        batch_size=batch_size,
        atol=atol,
        rtol=rtol,
        estimated_peak_bytes=int(estimated),
        maximum_abs_e_deterministic=maximum_abs,
        minimum_scattered_power=float(np.min(scattered_out)),
        maximum_total_power=maximum_total,
        returned_balance_max_abs_residual=balance,
    )
    return RuzePowerDiagnostic(
        schema_version="radiosim.ruze_power_diagnostic.v1",
        method="poisson_paired_pupil_separation_v1",
        antenna_id=antenna_id,
        covariance_convention="gaussian_one_over_e_surface_covariance_v1",
        normalization_convention="unmodified_ideal_aperture_v1",
        frequency_hz=float(frequency_hz),
        time_mjd=float(time_mjd),
        rms_surface_error_m=float(rms_surface_error_m),
        correlation_length_m=float(correlation_length_m),
        altitude_rad=_owned_read_only(altitude, np.dtype(np.float64)),
        azimuth_rad=_owned_read_only(azimuth, np.dtype(np.float64)),
        coherent_main_power=coherent_out,
        total_ensemble_power=total_out,
        scattered_power=scattered_out,
        convergence=convergence,
    )
