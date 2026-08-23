r"""The canonical exact-turn ERA grid, its UTC/UT1 mapping, and the m transform.

``docs/development/sci004_mmode_design.md`` Section 3.1 makes the m-mode time
coordinate the continuously unwrapped number of Earth-rotation turns

.. math::

    u(t)=\frac{\operatorname{ERA}(t)-\operatorname{ERA}(t_0)}{2\pi},
    \qquad u(t_0)=0,

and builds the entire grid in **exact rational arithmetic**.  UTC is an output
and provenance coordinate; it is not the group coordinate that diagonalizes a
transit telescope, so it never owns endpoint ownership, wrapping, or cycle
closure.  Radians are a derived numerical view: every one of them is a single
round-to-nearest-ties-to-even of ``exact(tau) * <exact rational>`` with no
intermediate binary64 arithmetic.

Two consequences are load-bearing.  Exact-turn equality, not rounded-radian
subtraction, is the closure authority -- Section 3.1 states that there is
*deliberately no* assertion ``horizon_hi_rad - horizon_lo_rad == tau``, and
Section 14.2 makes a predicate requiring it a validation failure.  And there is
no sample at ``u = 1``: that value exists only as the virtual closure point.

This module also owns two things Section 12.1 names it for.  The first is the
correctly rounded unit-circle kernel the DFT, synthesis and exposure ``sinc``
consume: each reduces its exact rational turn argument modulo one turn *before*
any floating-point work, so none of them regenerates topology from ``k``, ``N``,
radians or ``tau``.  The second is the certified rational-interval trigonometric
kernel :func:`certified_two_pi_trig`, which the Section 12.1 frozen analytic
horizon oracle uses to prove root topology.  That kernel takes ``pi`` by an
exact rational bracket, evaluates alternating Taylor series with a proved
remainder over exact rational arguments, and rounds outward to binary64 only
after the proof; applying ``numpy.nextafter`` to an arbitrary platform ``libm``
result is explicitly insufficient and is not done anywhere here.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from fractions import Fraction
from types import MappingProxyType
from typing import Any, Final

import numpy as np

from radiosim.core.mmode.types import (
    MMODE_RADIAN_GRID_CONVENTION,
    MMODE_TIME_GRID_CONVENTION,
    TAU,
    array_digest,
    canonical_json,
    canonical_rational,
    domain_digest,
    f64be,
    object_digest,
    round_to_nearest_turn_radians,
)

__all__ = [
    "ERA_CENTER_LIMIT_RAD",
    "ERA_RATE_TURNS_PER_UT1_DAY",
    "ERA_STEP_LIMIT_RAD",
    "IERS_PACKAGE",
    "IERS_RESOURCE",
    "UT1_UTC_ROUNDTRIP_LIMIT_SECONDS",
    "CanonicalEraGrid",
    "InstalledIers",
    "build_canonical_era_grid",
    "certified_two_pi_trig",
    "exact_turn_grid",
    "exposure_edges_resolve",
    "exposure_sinc_weights",
    "forward_m_transform",
    "installed_iers",
    "installed_iers_context",
    "synthesize_time_series",
    "unit_circle_turn",
]

# ---------------------------------------------------------------------------
# Section 3.1 frozen constants.  Constants, never YAML fields; a tolerance is
# never widened because a platform misses it.
# ---------------------------------------------------------------------------

#: Maximum unwrapped ERA centre residual.
ERA_CENTER_LIMIT_RAD: Final[float] = 2e-11
#: Maximum ERA step residual from the retained ``tau / N``.
ERA_STEP_LIMIT_RAD: Final[float] = 2e-11
#: Maximum UT1 -> UTC -> UT1 round-trip residual.
UT1_UTC_ROUNDTRIP_LIMIT_SECONDS: Final[float] = 1e-6

#: Section 3.1's locked offline Earth-orientation resource.  No network lookup
#: and no implicit Astropy table selection is permitted.
IERS_PACKAGE: Final = "astropy_iers_data"
IERS_RESOURCE: Final = "data/finals2000A.all"

#: Section 3.1's ERA-rate literal, treated as its exact decimal rational inside
#: the two-part-JD implementation before final rounding.  ``tau`` therefore
#: cancels analytically and is not an input to time ownership or cadence.
ERA_RATE_TURNS_PER_UT1_DAY: Final[Fraction] = Fraction("1.00273781191135448")

# ---------------------------------------------------------------------------
# Certified rational-interval trigonometry (Section 12.1)
# ---------------------------------------------------------------------------

#: ``pi`` bracketed by two exact rationals.  The 50-digit decimal expansion of
#: ``pi`` is a published mathematical constant, so the bracket below is a proof
#: obligation discharged by literature rather than by a floating-point library.
#: Every certified bound in this module inherits its rigour from these two
#: rationals; nothing here trusts a platform ``libm`` result.
_PI_DIGITS: Final = "3.14159265358979323846264338327950288419716939937510"
_PI_LO: Final[Fraction] = Fraction(_PI_DIGITS)
_PI_HI: Final[Fraction] = _PI_LO + Fraction(1, 10**50)

#: Taylor truncation order.  The alternating bracket's width *is* the first
#: omitted term, ``t**(2n+1)/(2n+1)!``; on the reduced argument
#: ``t <= pi/4 + eps`` and ``n = 10`` that is under ``2e-22`` -- six orders
#: below one binary64 ulp near unity, and far tighter than the ``1e-13 rad``
#: root width and ``2e-13`` residual Section 12.1 requires.
_TAYLOR_TERMS: Final[int] = 10


def _alternating_bracket(terms: Sequence[Fraction]) -> tuple[Fraction, Fraction]:
    """Bracket an alternating series from two consecutive partial sums.

    For a series whose terms alternate in sign and strictly decrease in
    magnitude, consecutive partial sums bracket the limit.  Returning the last
    two partial sums *is* the proof: nothing beyond the alternating-series
    theorem is asserted, and no floating-point remainder estimate is trusted.
    """
    partial = Fraction(0)
    previous = Fraction(0)
    for term in terms:
        previous = partial
        partial += term
    return (min(previous, partial), max(previous, partial))


#: The dyadic working grid of the certified series evaluation.  Every rational
#: entering or leaving a Taylor step is snapped to a multiple of ``2**-120`` so
#: numerators stay 120-bit integers instead of growing with the power index;
#: the accumulated snap error is bounded far below ``_SERIES_GUARD``, which is
#: added outward to the final bracket, so soundness is preserved exactly.
_SERIES_GRID: Final[int] = 1 << 120
_SERIES_GUARD: Final[Fraction] = Fraction(1, 1 << 110)


def _snap(value: Fraction) -> Fraction:
    """Return ``value`` rounded to the nearest multiple of ``2**-200``."""
    return Fraction(round(value * _SERIES_GRID), _SERIES_GRID)


def _sin_point(t: Fraction) -> tuple[Fraction, Fraction]:
    """Bracket ``sin(t)`` for an exact rational ``t`` in ``[0, pi/2]``."""
    argument = _snap(t)
    square = _snap(argument * argument)
    terms: list[Fraction] = []
    power = argument
    factorial = 1
    for index in range(_TAYLOR_TERMS):
        terms.append(_snap((-1 if index % 2 else 1) * power / factorial))
        power = _snap(power * square)
        factorial *= (2 * index + 2) * (2 * index + 3)
    low, high = _alternating_bracket(terms)
    return (low - _SERIES_GUARD, high + _SERIES_GUARD)


def _cos_point(t: Fraction) -> tuple[Fraction, Fraction]:
    """Bracket ``cos(t)`` for an exact rational ``t`` in ``[0, pi/2]``."""
    argument = _snap(t)
    square = _snap(argument * argument)
    terms: list[Fraction] = []
    power = Fraction(1)
    factorial = 1
    for index in range(_TAYLOR_TERMS):
        terms.append(_snap((-1 if index % 2 else 1) * power / factorial))
        power = _snap(power * square)
        factorial *= (2 * index + 1) * (2 * index + 2)
    low, high = _alternating_bracket(terms)
    return (low - _SERIES_GUARD, high + _SERIES_GUARD)


def _first_eighth_brackets(
    eighth_turn: Fraction,
) -> tuple[tuple[Fraction, Fraction], tuple[Fraction, Fraction]]:
    """Bracket ``(cos, sin)`` of ``2*pi*s`` for an exact ``s`` in ``[0, 1/8]``.

    ``pi`` enters only through its exact rational bracket, so the argument
    interval ``[2*PI_LO*s, 2*PI_HI*s]`` is itself certified.  Both functions are
    monotone there -- ``sin`` increasing, ``cos`` decreasing -- so the endpoint
    brackets are the interval bounds.
    """
    t_lo = _snap(2 * _PI_LO * eighth_turn)
    t_hi = _snap(2 * _PI_HI * eighth_turn) + _SERIES_GUARD
    sin_lo = _sin_point(t_lo)[0]
    sin_hi = _sin_point(t_hi)[1]
    cos_lo = _cos_point(t_hi)[0]
    cos_hi = _cos_point(t_lo)[1]
    return ((cos_lo, cos_hi), (sin_lo, sin_hi))


def _point_brackets(
    turn: Fraction,
) -> tuple[tuple[Fraction, Fraction], tuple[Fraction, Fraction]]:
    """Bracket ``(cos(2*pi*u), sin(2*pi*u))`` at one exact rational turn.

    The turn is reduced modulo one turn, then into a quadrant and finally into
    the first eighth, all in exact rational arithmetic.  Only the residual
    eighth-turn argument reaches a series evaluation.
    """
    value = Fraction(turn) % 1
    quadrant = int(value * 4)
    residue = value - Fraction(quadrant, 4)
    if residue <= Fraction(1, 8):
        (cos_lo, cos_hi), (sin_lo, sin_hi) = _first_eighth_brackets(residue)
    else:
        # ``cos(2*pi*s) == sin(2*pi*(1/4 - s))`` and vice versa.
        (sin_lo, sin_hi), (cos_lo, cos_hi) = _first_eighth_brackets(
            Fraction(1, 4) - residue
        )
    if quadrant == 0:
        return ((cos_lo, cos_hi), (sin_lo, sin_hi))
    if quadrant == 1:
        return ((-sin_hi, -sin_lo), (cos_lo, cos_hi))
    if quadrant == 2:
        return ((-cos_hi, -cos_lo), (-sin_hi, -sin_lo))
    return ((sin_lo, sin_hi), (-cos_hi, -cos_lo))


def _arc_meets(lower: Fraction, upper: Fraction, node: Fraction) -> bool:
    """Return whether the closed arc ``[lower, upper]`` meets ``node + Z``.

    Exact rational floors and ceilings answer this; a floating epsilon never
    classifies the topology.
    """
    return math.floor(upper - node) >= math.ceil(lower - node)


def certified_two_pi_trig(
    lower_turn: Fraction, upper_turn: Fraction
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Return outward binary64 enclosures of ``cos(2*pi*u)`` and ``sin(2*pi*u)``.

    Parameters
    ----------
    lower_turn, upper_turn : Fraction
        Exact rational endpoints of a closed turn interval, ``lower <= upper``.

    Returns
    -------
    tuple
        ``((cos_lo, cos_hi), (sin_lo, sin_hi))``, rounded outward to binary64
        only after the rational proof completes.

    Notes
    -----
    Section 12.1 requires the enclosure to be sound, not merely tight.  The
    extrema of ``cos(2*pi*u)`` occur exactly at integer turns and integers plus
    a half turn, and those of ``sin(2*pi*u)`` at quarter and three-quarter
    turns; whether the closed arc contains one is an exact rational question,
    answered here with no floating-point comparison anywhere in the decision.
    """
    low = Fraction(lower_turn)
    high = Fraction(upper_turn)
    if high < low:
        raise ValueError("certified_two_pi_trig requires lower <= upper")
    if high - low >= 1:
        return ((-1.0, 1.0), (-1.0, 1.0))
    (cos_a_lo, cos_a_hi), (sin_a_lo, sin_a_hi) = _point_brackets(low)
    if high == low:
        # A singleton needs one series evaluation, not two identical ones.
        (cos_b_lo, cos_b_hi), (sin_b_lo, sin_b_hi) = (
            (cos_a_lo, cos_a_hi),
            (sin_a_lo, sin_a_hi),
        )
    else:
        (cos_b_lo, cos_b_hi), (sin_b_lo, sin_b_hi) = _point_brackets(high)
    cos_low = min(cos_a_lo, cos_b_lo)
    cos_high = max(cos_a_hi, cos_b_hi)
    sin_low = min(sin_a_lo, sin_b_lo)
    sin_high = max(sin_a_hi, sin_b_hi)
    if _arc_meets(low, high, Fraction(0)):
        cos_high = Fraction(1)
    if _arc_meets(low, high, Fraction(1, 2)):
        cos_low = Fraction(-1)
    if _arc_meets(low, high, Fraction(1, 4)):
        sin_high = Fraction(1)
    if _arc_meets(low, high, Fraction(3, 4)):
        sin_low = Fraction(-1)
    return (
        (round_down(cos_low), round_up(cos_high)),
        (round_down(sin_low), round_up(sin_high)),
    )


def round_down(value: Fraction) -> float:
    """Return the greatest binary64 not larger than an exact rational."""
    nearest = float(value)
    if Fraction(nearest) > value:
        return math.nextafter(nearest, -math.inf)
    return nearest


def round_up(value: Fraction) -> float:
    """Return the least binary64 not smaller than an exact rational."""
    nearest = float(value)
    if Fraction(nearest) < value:
        return math.nextafter(nearest, math.inf)
    return nearest


def certified_two_pi_bracket() -> tuple[Fraction, Fraction]:
    """Return the exact rational bracket of ``2 * pi`` used by the certifier."""
    return (2 * _PI_LO, 2 * _PI_HI)


def unit_circle_turn(turn: Fraction) -> complex:
    """Return ``exp(+i * 2 * pi * u)`` from the exact rational turn ``u``.

    The argument is reduced modulo one turn, into a quadrant, and finally into
    the first eighth, all in exact rational arithmetic *before* any
    floating-point work.  The value handed to the platform kernel therefore
    never exceeds ``pi/4``, and the DFT, synthesis and exposure ``sinc`` never
    regenerate topology from ``k``, ``N``, radians or ``tau``.
    """
    value = Fraction(turn) % 1
    quadrant = int(value * 4)
    residue = value - Fraction(quadrant, 4)
    if residue <= Fraction(1, 8):
        angle = 2.0 * math.pi * float(residue)
        cosine = math.cos(angle)
        sine = math.sin(angle)
    else:
        angle = 2.0 * math.pi * float(Fraction(1, 4) - residue)
        cosine = math.sin(angle)
        sine = math.cos(angle)
    if quadrant == 0:
        return complex(cosine, sine)
    if quadrant == 1:
        return complex(-sine, cosine)
    if quadrant == 2:
        return complex(-cosine, -sine)
    return complex(sine, -cosine)


# ---------------------------------------------------------------------------
# Section 3.1 exact turn construction
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ExactTurnGrid:
    """The exact rational turn coordinates of one full-sidereal cycle."""

    sidereal_samples: int
    integration_fraction: float
    fraction_ratio: Fraction
    centers: tuple[Fraction, ...]
    lower_edges: tuple[Fraction, ...]
    upper_edges: tuple[Fraction, ...]
    exposure_width: Fraction
    horizon_lo: Fraction
    horizon_hi: Fraction


def exact_turn_grid(
    sidereal_samples: int, integration_fraction: float
) -> ExactTurnGrid:
    """Build Section 3.1's exact rational turn grid.

    The already validated finite binary64 ``integration_fraction`` is decoded by
    its exact IEEE-754 integer ratio ``p_f / q_f``; the source decimal spelling
    is not an arithmetic input.  The edge formulas ``(2k -/+ f)/(2N)`` are
    evaluated with ``f`` as that exact ratio and are never formed by binary64
    subtraction or addition.
    """
    samples = _require_positive_int(sidereal_samples, "sidereal_samples")
    fraction = _require_integration_fraction(integration_fraction)
    numerator, denominator = fraction.as_integer_ratio()
    centers = tuple(Fraction(2 * k, 2 * samples) for k in range(samples))
    lower = tuple(
        Fraction(2 * k * denominator - numerator, 2 * samples * denominator)
        for k in range(samples)
    )
    upper = tuple(
        Fraction(2 * k * denominator + numerator, 2 * samples * denominator)
        for k in range(samples)
    )
    return ExactTurnGrid(
        sidereal_samples=samples,
        integration_fraction=fraction,
        fraction_ratio=Fraction(numerator, denominator),
        centers=centers,
        lower_edges=lower,
        upper_edges=upper,
        exposure_width=Fraction(numerator, samples * denominator),
        horizon_lo=Fraction(-1, 2 * samples),
        horizon_hi=Fraction(2 * samples - 1, 2 * samples),
    )


def _require_positive_int(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{field_name} must be a strict positive integer")
    number = int(value)
    if number < 1:
        raise ValueError(f"{field_name} must be a strict positive integer")
    return number


def _require_integration_fraction(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (float, int, np.floating)):
        raise TypeError("integration_fraction must be a strict finite float")
    number = float(value)
    if not math.isfinite(number) or not 0.0 < number <= 1.0:
        raise ValueError("integration_fraction must be a finite float in (0, 1]")
    if math.copysign(1.0, number) < 0.0:  # pragma: no cover - defensive
        raise ValueError("integration_fraction must be a finite float in (0, 1]")
    return number


def exposure_edges_resolve(sidereal_samples: int, integration_fraction: float) -> bool:
    """Return whether the derived binary64 exposure edges stay distinct.

    Section 3.1 requires strictly increasing derived centres and
    ``lower_rad[k] < alpha_rad[k] < upper_rad[k]``.  A fraction that is finite
    and inside ``(0, 1]`` can still collapse those edges once they are rounded,
    which is a domain failure of the *derived* grid rather than a schema failure
    of the authored value; Section 8 gives it the ``mmode_exposure_resolution``
    code for exactly that reason.
    """
    grid = exact_turn_grid(sidereal_samples, integration_fraction)
    previous: float | None = None
    for index in range(grid.sidereal_samples):
        center = round_to_nearest_turn_radians(grid.centers[index])
        lower = round_to_nearest_turn_radians(grid.lower_edges[index])
        upper = round_to_nearest_turn_radians(grid.upper_edges[index])
        if not lower < center < upper:
            return False
        if previous is not None and not center > previous:
            return False
        previous = center
    return True


# ---------------------------------------------------------------------------
# Section 3.1 bundled offline IERS resolution
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class InstalledIers:
    """The one locked Earth-orientation table a run is allowed to consume."""

    resource_path: str
    table_sha256: str
    package_version: str
    table: Any

    @property
    def auto_download(self) -> bool:
        """Return ``False``: no network lookup is ever permitted."""
        return False


_INSTALLED: InstalledIers | None = None


def installed_iers() -> InstalledIers:
    """Resolve, hash and open exactly the Section 3.1 locked resource."""
    global _INSTALLED
    if _INSTALLED is not None:
        return _INSTALLED
    import importlib.metadata as metadata
    import importlib.resources as resources

    from astropy.utils.iers import IERS_A

    resource = resources.files(IERS_PACKAGE) / IERS_RESOURCE
    payload = resource.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    table = IERS_A.open(str(resource))
    try:
        version = metadata.version(IERS_PACKAGE)
    except metadata.PackageNotFoundError:  # pragma: no cover - defensive
        version = "unknown"
    _INSTALLED = InstalledIers(
        resource_path=str(resource),
        table_sha256=digest,
        package_version=str(version),
        table=table,
    )
    return _INSTALLED


@contextmanager
def installed_iers_context() -> Iterator[InstalledIers]:
    """Install the locked table around every time and coordinate operation."""
    from astropy.utils.iers import conf as iers_conf
    from astropy.utils.iers import earth_orientation_table

    resolved = installed_iers()
    previous_download = iers_conf.auto_download
    previous_degraded = iers_conf.iers_degraded_accuracy
    iers_conf.auto_download = False
    iers_conf.iers_degraded_accuracy = "error"
    try:
        with earth_orientation_table.set(resolved.table):
            yield resolved
    finally:
        iers_conf.auto_download = previous_download
        iers_conf.iers_degraded_accuracy = previous_degraded


def full_sidereal_iers_covered(start_time: str) -> bool:
    """Return whether one full sidereal cycle stays inside the installed table.

    Section 3.1 requires sample centres, exposure boundaries, horizon-split
    oracle nodes and the anchor attitude all to lie inside the installed table's
    accepted range.  The cell-centred cycle ``H_N`` spans strictly less than one
    turn either side of the anchor, so probing the anchor and both one-day
    envelopes covers every retained node conservatively without mapping the
    whole grid.
    """
    from astropy.time import Time, TimeDelta

    with installed_iers_context():
        try:
            anchor = Time(str(start_time).strip(), scale="utc")
            envelope = anchor + TimeDelta(
                np.asarray([-1.0, 0.0, 1.0], dtype=np.float64), format="jd"
            )
        except Exception:  # pragma: no cover - malformed times reject earlier
            return False
        return bool(utc_times_are_covered(envelope))


def utc_times_are_covered(times: Any) -> bool:
    """Return whether every UTC coordinate lies inside the installed table."""
    from astropy.utils import iers as iers_module

    resolved = installed_iers()
    _, status = resolved.table.ut1_utc(times, return_status=True)
    codes = np.atleast_1d(np.asarray(status))
    outside = {
        int(iers_module.TIME_BEFORE_IERS_RANGE),
        int(iers_module.TIME_BEYOND_IERS_RANGE),
    }
    return not any(int(code) in outside for code in codes.ravel().tolist())


# ---------------------------------------------------------------------------
# Section 3.1 CanonicalEraGrid
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CanonicalEraGrid:
    """The exact turn object, its derived binary64 view, and their identities.

    Section 3.1 requires this same immutable object to be passed *by identity*
    to the time mapper, the operational isolator, the phase ledger, the harmonic
    window and both direct oracles.  Reconstructing a turn coordinate from
    ``k``, radians, a width or an adjacent edge inside any consumer is a
    validation failure, so every consumer reads the retained rationals here.
    """

    start_time_iso: str
    sidereal_samples: int
    integration_fraction: float
    exact: ExactTurnGrid
    center_turns: tuple[str, ...]
    lower_edge_turns: tuple[str, ...]
    upper_edge_turns: tuple[str, ...]
    exposure_width_turn: str
    horizon_lo_turn: str
    horizon_hi_turn: str
    alpha_rad: tuple[float, ...]
    lower_rad: tuple[float, ...]
    upper_rad: tuple[float, ...]
    delta_alpha_rad: float
    horizon_lo_rad: float
    horizon_hi_rad: float
    canonical_era_turn_grid: Mapping[str, Any]
    canonical_era_grid: Mapping[str, Any]
    era_center_turn_sha256: str
    era_lower_edge_turn_sha256: str
    era_upper_edge_turn_sha256: str
    canonical_era_turn_grid_sha256: str
    era_center_rad_sha256: str
    era_lower_edge_rad_sha256: str
    era_upper_edge_rad_sha256: str
    canonical_era_grid_sha256: str
    utc_two_part: tuple[np.ndarray, np.ndarray]
    ut1_two_part: tuple[np.ndarray, np.ndarray]
    lower_utc_two_part: tuple[np.ndarray, np.ndarray]
    upper_utc_two_part: tuple[np.ndarray, np.ndarray]
    integration_time_seconds: np.ndarray
    integration_time_seconds_sha256: str
    era_center_max_residual_rad: float
    era_step_max_residual_rad: float
    ut1_utc_roundtrip_seconds: float
    iers_table_sha256: str

    def __len__(self) -> int:
        return self.sidereal_samples

    @property
    def era_center_limit_rad(self) -> float:
        """Return Section 3.1's fixed centre-residual limit."""
        return ERA_CENTER_LIMIT_RAD

    @property
    def era_step_limit_rad(self) -> float:
        """Return Section 3.1's fixed step-residual limit."""
        return ERA_STEP_LIMIT_RAD

    @property
    def ut1_utc_roundtrip_limit_seconds(self) -> float:
        """Return Section 3.1's fixed UT1/UTC round-trip limit."""
        return UT1_UTC_ROUNDTRIP_LIMIT_SECONDS

    def center_turn(self, index: int) -> Fraction:
        """Return one retained exact centre turn."""
        return self.exact.centers[index]

    def exposure_turns(self, index: int) -> tuple[Fraction, Fraction]:
        """Return one retained exact exposure interval."""
        return (self.exact.lower_edges[index], self.exact.upper_edges[index])

    @property
    def exposure_width(self) -> Fraction:
        """Return the retained exact exposure width ``Delta_u``."""
        return self.exact.exposure_width

    @property
    def horizon_domain(self) -> tuple[Fraction, Fraction]:
        """Return the exact cell-centred cycle ``H_N = [h_N^-, h_N^+)``."""
        return (self.exact.horizon_lo, self.exact.horizon_hi)


def _turn_array_digest(domain: str, values: Sequence[str]) -> str:
    return domain_digest(domain, _json_array(values))


def _json_array(values: Sequence[str]) -> bytes:
    return canonical_json(list(values))


def build_canonical_era_grid(
    *,
    sidereal_samples: int,
    integration_fraction: float,
    start_time: str,
) -> CanonicalEraGrid:
    """Build Section 3.1's canonical exact-turn ERA grid and its UTC mapping.

    Parameters
    ----------
    sidereal_samples : int
        ``N``, a strict positive integer.
    integration_fraction : float
        ``f``, a strict finite float in ``(0, 1]``.  It scales the top-hat width
        without removing a sample.
    start_time : str
        The anchor, an Astropy-parseable UTC ISO string.

    Returns
    -------
    CanonicalEraGrid
        The retained exact turn object, its one-final-round radian view, the
        UTC/UT1 mapping, and all eight Section 3.1 component digests.

    Raises
    ------
    radiosim.io.config_resolution.ConfigSemanticError
        With issue code ``mmode_exposure_resolution`` when the derived binary64
        exposure edges collapse, or ``mmode_iers_range`` when the mapped cycle
        leaves the installed offline table.
    """
    from radiosim.io.config import ConfigIssue
    from radiosim.io.config_resolution import ConfigSemanticError

    exact = exact_turn_grid(sidereal_samples, integration_fraction)
    samples = exact.sidereal_samples

    alpha_rad = tuple(round_to_nearest_turn_radians(turn) for turn in exact.centers)
    lower_rad = tuple(round_to_nearest_turn_radians(turn) for turn in exact.lower_edges)
    upper_rad = tuple(round_to_nearest_turn_radians(turn) for turn in exact.upper_edges)
    delta_alpha_rad = round_to_nearest_turn_radians(exact.exposure_width)
    horizon_lo_rad = round_to_nearest_turn_radians(exact.horizon_lo)
    horizon_hi_rad = round_to_nearest_turn_radians(exact.horizon_hi)

    for index in range(samples):
        if not lower_rad[index] < alpha_rad[index] < upper_rad[index]:
            raise ConfigSemanticError(
                [
                    ConfigIssue(
                        "obs_time.integration_fraction",
                        "mmode_exposure_resolution",
                        MMODE_EXPOSURE_RESOLUTION_MESSAGE,
                    )
                ]
            )
        if index and not alpha_rad[index] > alpha_rad[index - 1]:
            raise ConfigSemanticError(
                [
                    ConfigIssue(
                        "obs_time.integration_fraction",
                        "mmode_exposure_resolution",
                        MMODE_EXPOSURE_RESOLUTION_MESSAGE,
                    )
                ]
            )

    center_turns = tuple(canonical_rational(turn) for turn in exact.centers)
    lower_turns = tuple(canonical_rational(turn) for turn in exact.lower_edges)
    upper_turns = tuple(canonical_rational(turn) for turn in exact.upper_edges)

    turn_grid: dict[str, Any] = {
        "schema_version": MMODE_TIME_GRID_CONVENTION,
        "sidereal_samples": samples,
        "integration_fraction_f64be": f64be(exact.integration_fraction),
        "integration_fraction_ratio": canonical_rational(exact.fraction_ratio),
        "exposure_width_turn": canonical_rational(exact.exposure_width),
        "horizon_lo_turn": canonical_rational(exact.horizon_lo),
        "horizon_hi_turn": canonical_rational(exact.horizon_hi),
        "center_turns": list(center_turns),
        "lower_edge_turns": list(lower_turns),
        "upper_edge_turns": list(upper_turns),
    }

    center_turn_sha = _turn_array_digest(
        "radiosim.mmode-era-center-turns.v1", center_turns
    )
    lower_turn_sha = _turn_array_digest(
        "radiosim.mmode-era-lower-edge-turns.v1", lower_turns
    )
    upper_turn_sha = _turn_array_digest(
        "radiosim.mmode-era-upper-edge-turns.v1", upper_turns
    )
    turn_grid_sha = object_digest(MMODE_TIME_GRID_CONVENTION, turn_grid)

    radian_domain = "radiosim.mmode-era-radian-array.v1"
    center_rad_sha = array_digest(
        radian_domain,
        "center",
        ["sample"],
        "rad",
        np.asarray(alpha_rad, dtype=np.float64),
        dtype="float64-be",
    )
    lower_rad_sha = array_digest(
        radian_domain,
        "lower_edge",
        ["sample"],
        "rad",
        np.asarray(lower_rad, dtype=np.float64),
        dtype="float64-be",
    )
    upper_rad_sha = array_digest(
        radian_domain,
        "upper_edge",
        ["sample"],
        "rad",
        np.asarray(upper_rad, dtype=np.float64),
        dtype="float64-be",
    )

    radian_grid: dict[str, Any] = {
        "schema_version": MMODE_RADIAN_GRID_CONVENTION,
        "canonical_era_turn_grid_sha256": turn_grid_sha,
        "era_center_turn_sha256": center_turn_sha,
        "era_lower_edge_turn_sha256": lower_turn_sha,
        "era_upper_edge_turn_sha256": upper_turn_sha,
        "tau_f64be": f64be(TAU),
        "delta_alpha_rad_f64be": f64be(delta_alpha_rad),
        "horizon_lo_rad_f64be": f64be(horizon_lo_rad),
        "horizon_hi_rad_f64be": f64be(horizon_hi_rad),
        "era_center_rad_sha256": center_rad_sha,
        "era_lower_edge_rad_sha256": lower_rad_sha,
        "era_upper_edge_rad_sha256": upper_rad_sha,
    }
    radian_grid_sha = object_digest(MMODE_RADIAN_GRID_CONVENTION, radian_grid)

    mapping = _map_turns_to_time(exact, start_time)

    return CanonicalEraGrid(
        start_time_iso=mapping["start_time_iso"],
        sidereal_samples=samples,
        integration_fraction=exact.integration_fraction,
        exact=exact,
        center_turns=center_turns,
        lower_edge_turns=lower_turns,
        upper_edge_turns=upper_turns,
        exposure_width_turn=canonical_rational(exact.exposure_width),
        horizon_lo_turn=canonical_rational(exact.horizon_lo),
        horizon_hi_turn=canonical_rational(exact.horizon_hi),
        alpha_rad=alpha_rad,
        lower_rad=lower_rad,
        upper_rad=upper_rad,
        delta_alpha_rad=delta_alpha_rad,
        horizon_lo_rad=horizon_lo_rad,
        horizon_hi_rad=horizon_hi_rad,
        canonical_era_turn_grid=MappingProxyType(turn_grid),
        canonical_era_grid=MappingProxyType(radian_grid),
        era_center_turn_sha256=center_turn_sha,
        era_lower_edge_turn_sha256=lower_turn_sha,
        era_upper_edge_turn_sha256=upper_turn_sha,
        canonical_era_turn_grid_sha256=turn_grid_sha,
        era_center_rad_sha256=center_rad_sha,
        era_lower_edge_rad_sha256=lower_rad_sha,
        era_upper_edge_rad_sha256=upper_rad_sha,
        canonical_era_grid_sha256=radian_grid_sha,
        utc_two_part=mapping["utc"],
        ut1_two_part=mapping["ut1"],
        lower_utc_two_part=mapping["lower_utc"],
        upper_utc_two_part=mapping["upper_utc"],
        integration_time_seconds=mapping["integration_time_seconds"],
        integration_time_seconds_sha256=mapping["integration_time_seconds_sha256"],
        era_center_max_residual_rad=mapping["era_center_max_residual_rad"],
        era_step_max_residual_rad=mapping["era_step_max_residual_rad"],
        ut1_utc_roundtrip_seconds=mapping["ut1_utc_roundtrip_seconds"],
        iers_table_sha256=mapping["iers_table_sha256"],
    )


MMODE_EXPOSURE_RESOLUTION_MESSAGE: Final = (
    "obs_time.integration_fraction is too small for distinct canonical "
    "binary64 exposure edges at this sidereal_samples."
)
MMODE_IERS_RANGE_MESSAGE: Final = (
    "the full-sidereal UTC mapping is outside the available offline IERS table."
)


def _two_part_from_turns(
    jd1: float, jd2: float, turns: Sequence[Fraction]
) -> tuple[np.ndarray, np.ndarray]:
    """Map exact turns onto two-part UT1 Julian dates with one final rounding.

    ``JD_UT1(u) = JD_UT1_0 + u / 1.00273781191135448`` with the ERA-rate literal
    treated as its exact decimal rational, so ``tau`` cancels analytically and
    is not an input to time ownership or cadence.
    """
    exact_jd2 = Fraction(jd2)
    second = np.empty(len(turns), dtype=np.float64)
    for index, turn in enumerate(turns):
        offset_days = Fraction(turn) / ERA_RATE_TURNS_PER_UT1_DAY
        second[index] = float(exact_jd2 + offset_days)
    first = np.full(len(turns), float(jd1), dtype=np.float64)
    return first, second


def _map_turns_to_time(exact: ExactTurnGrid, start_time: str) -> dict[str, Any]:
    """Map every retained exact turn to UT1/UTC inside the installed context."""
    import erfa
    from astropy.time import Time

    from radiosim.io.config import ConfigIssue
    from radiosim.io.config_resolution import ConfigSemanticError

    if not isinstance(start_time, str) or not start_time.strip():
        raise ValueError("start_time must be a nonblank string")

    with installed_iers_context() as resolved:
        anchor = Time(start_time.strip(), scale="utc")
        if not utc_times_are_covered(anchor):
            raise ConfigSemanticError(
                [
                    ConfigIssue(
                        "obs_time.start_time",
                        "mmode_iers_range",
                        MMODE_IERS_RANGE_MESSAGE,
                    )
                ]
            )
        anchor_ut1 = anchor.ut1
        jd1 = float(np.asarray(anchor_ut1.jd1))
        jd2 = float(np.asarray(anchor_ut1.jd2))

        center_ut1 = _two_part_from_turns(jd1, jd2, exact.centers)
        lower_ut1 = _two_part_from_turns(jd1, jd2, exact.lower_edges)
        upper_ut1 = _two_part_from_turns(jd1, jd2, exact.upper_edges)
        horizon_ut1 = _two_part_from_turns(
            jd1, jd2, (exact.horizon_lo, exact.horizon_hi)
        )

        try:
            center_time = Time(center_ut1[0], center_ut1[1], format="jd", scale="ut1")
            lower_time = Time(lower_ut1[0], lower_ut1[1], format="jd", scale="ut1")
            upper_time = Time(upper_ut1[0], upper_ut1[1], format="jd", scale="ut1")
            horizon_time = Time(
                horizon_ut1[0], horizon_ut1[1], format="jd", scale="ut1"
            )
            center_utc = center_time.utc
            lower_utc = lower_time.utc
            upper_utc = upper_time.utc
            horizon_utc = horizon_time.utc
        except Exception as exc:  # pragma: no cover - astropy raises on range
            raise ConfigSemanticError(
                [
                    ConfigIssue(
                        "obs_time.start_time",
                        "mmode_iers_range",
                        MMODE_IERS_RANGE_MESSAGE,
                    )
                ]
            ) from exc

        for candidate in (center_utc, lower_utc, upper_utc, horizon_utc):
            if not utc_times_are_covered(candidate):
                raise ConfigSemanticError(
                    [
                        ConfigIssue(
                            "obs_time.start_time",
                            "mmode_iers_range",
                            MMODE_IERS_RANGE_MESSAGE,
                        )
                    ]
                )

        roundtrip = center_utc.ut1
        roundtrip_seconds = float(
            np.max(np.abs(np.asarray((roundtrip - center_time).to_value("s"))))
        )

        era = np.asarray(erfa.era00(center_ut1[0], center_ut1[1]), dtype=np.float64)
        era0 = float(erfa.era00(jd1, jd2))
        unwrapped = np.unwrap(np.concatenate(([era0], era)))[1:] - era0
        residual = float(
            np.max(np.abs(unwrapped - np.asarray(_alpha_view(exact), dtype=np.float64)))
        )
        expected_step = round_to_nearest_turn_radians(
            Fraction(1, exact.sidereal_samples)
        )
        if exact.sidereal_samples > 1:
            steps = np.diff(unwrapped)
            step_residual = float(np.max(np.abs(steps - expected_step)))
        else:
            step_residual = 0.0

        widths = np.asarray((upper_utc - lower_utc).to_value("s"), dtype=np.float64)
        widths = np.atleast_1d(widths).astype(np.float64)
        width_digest = array_digest(
            "radiosim.mmode-integration-time.v1",
            "integration_time",
            ["sample"],
            "s",
            widths,
            dtype="float64-be",
        )

        return {
            "start_time_iso": str(anchor.utc.isot),
            "utc": (
                np.asarray(center_utc.jd1, dtype=np.float64),
                np.asarray(center_utc.jd2, dtype=np.float64),
            ),
            "ut1": center_ut1,
            "lower_utc": (
                np.asarray(lower_utc.jd1, dtype=np.float64),
                np.asarray(lower_utc.jd2, dtype=np.float64),
            ),
            "upper_utc": (
                np.asarray(upper_utc.jd1, dtype=np.float64),
                np.asarray(upper_utc.jd2, dtype=np.float64),
            ),
            "integration_time_seconds": widths,
            "integration_time_seconds_sha256": width_digest,
            "era_center_max_residual_rad": residual,
            "era_step_max_residual_rad": step_residual,
            "ut1_utc_roundtrip_seconds": roundtrip_seconds,
            "iers_table_sha256": resolved.table_sha256,
        }


def _alpha_view(exact: ExactTurnGrid) -> tuple[float, ...]:
    return tuple(round_to_nearest_turn_radians(turn) for turn in exact.centers)


# ---------------------------------------------------------------------------
# Section 6 normalized transform, synthesis and exposure window
# ---------------------------------------------------------------------------


def forward_m_transform(
    grid: CanonicalEraGrid,
    series: Sequence[complex] | np.ndarray,
    *,
    mmax: int,
) -> dict[int, complex]:
    """Return ``bar_v_m = (1/N) sum_k bar_V_k exp(-i 2 pi m u_k)``.

    The retained exact turns drive the kernel directly; the transform never
    regenerates topology from ``k``, ``N``, radians or ``tau``.  The returned
    mapping is keyed by the **signed** mode index, so ``coefficients[-3]`` is the
    ``m = -3`` coefficient rather than a positional lookup.
    """
    values = np.asarray(series, dtype=np.complex128)
    if values.shape != (grid.sidereal_samples,):
        raise ValueError("series must have one value per retained sample centre")
    if isinstance(mmax, bool) or not isinstance(mmax, (int, np.integer)) or mmax < 0:
        raise ValueError("mmax must be a non-negative integer")
    modes = int(mmax)
    result: dict[int, complex] = {}
    for mode in range(-modes, modes + 1):
        total = 0j
        for index in range(grid.sidereal_samples):
            phase = unit_circle_turn(-mode * grid.exact.centers[index])
            total += complex(values[index]) * phase
        result[mode] = total / grid.sidereal_samples
    return result


def synthesize_time_series(
    grid: CanonicalEraGrid,
    coefficients: Mapping[int, complex],
) -> tuple[complex, ...]:
    """Return ``bar_V_k = sum_m bar_v_m exp(+i 2 pi m u_k)``."""
    output: list[complex] = []
    for index in range(grid.sidereal_samples):
        turn = grid.exact.centers[index]
        total = 0j
        for mode, value in coefficients.items():
            total += complex(value) * unit_circle_turn(mode * turn)
        output.append(total)
    return tuple(output)


def exposure_sinc_weights(grid: CanonicalEraGrid, *, mmax: int) -> dict[int, float]:
    """Return Section 6's ``w_m = sinc(pi m Delta_u)`` from the exact width.

    ``w_0`` is exactly one.  The exposure top hat is a diagonal ``w_m`` factor,
    not a spectral taper: the centre-sample window stays the full rectangular
    periodic window with weight one.
    """
    width = grid.exact.exposure_width
    weights: dict[int, float] = {}
    for mode in range(-mmax, mmax + 1):
        if mode == 0:
            weights[mode] = 1.0
            continue
        argument_turn = Fraction(mode) * width / 2
        # ``sin(pi * m * Delta_u) == Im(exp(i * 2 * pi * (m * Delta_u / 2)))``.
        numerator = unit_circle_turn(argument_turn).imag
        denominator = math.pi * float(Fraction(mode) * width)
        weights[mode] = numerator / denominator
    return weights


def integration_time_identity(widths: np.ndarray) -> str:
    """Return Section 3.1's exact integration-width array identity."""
    return array_digest(
        "radiosim.mmode-integration-time.v1",
        "integration_time",
        ["sample"],
        "s",
        np.asarray(widths, dtype=np.float64),
        dtype="float64-be",
    )


def utc_manifest(grid: CanonicalEraGrid) -> tuple[Mapping[str, Any], str]:
    """Return Section 14.0's ``utc_manifest`` and its digest."""
    return _time_manifest(grid, scale="utc")


def ut1_manifest(grid: CanonicalEraGrid) -> tuple[Mapping[str, Any], str]:
    """Return Section 14.0's ``ut1_manifest`` and its digest."""
    return _time_manifest(grid, scale="ut1")


def _time_manifest(
    grid: CanonicalEraGrid, *, scale: str
) -> tuple[Mapping[str, Any], str]:
    domain = f"radiosim.mmode-{scale}-grid.v1"
    if scale == "utc":
        center = grid.utc_two_part
        lower = grid.lower_utc_two_part
        upper = grid.upper_utc_two_part
    else:
        center = grid.ut1_two_part
        lower = grid.lower_utc_two_part
        upper = grid.upper_utc_two_part
    manifest: dict[str, Any] = {
        "schema_version": domain,
        "scale": scale,
        "axis_order": ["sample"],
        "shape": [grid.sidereal_samples],
        "center_jd1_f64be": [f64be(value) for value in center[0].tolist()],
        "center_jd2_f64be": [f64be(value) for value in center[1].tolist()],
        "lower_jd1_f64be": [f64be(value) for value in lower[0].tolist()],
        "lower_jd2_f64be": [f64be(value) for value in lower[1].tolist()],
        "upper_jd1_f64be": [f64be(value) for value in upper[0].tolist()],
        "upper_jd2_f64be": [f64be(value) for value in upper[1].tolist()],
    }
    return MappingProxyType(manifest), object_digest(domain, manifest)
