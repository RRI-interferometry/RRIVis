"""The tropospheric term (T).

``T_p(s, nu, t)`` carries the neutral atmosphere's two effects in one
antenna-side factor -- excess path delay and opacity attenuation -- and both are
scalars times the identity::

    T_p(s, nu) = a_opacity(s) * exp( -2 pi i nu tau_trop(s) ) * I2

    tau_trop(s) = ( ZHD m_h(el) + ZWD m_w(el) ) / c
    a_opacity(s) = exp( -tau_0 / (2 sin el) )

so ``T`` commutes with every other term.  That is stated *and* tested rather
than assumed (invariant I2).

The factor of two in the opacity exponent
-----------------------------------------
It is deliberate, and it is the single easiest sign-class error in this term.
``T`` is a **voltage** Jones matrix while the opacity ``tau_0`` is defined on
**power**, so the voltage attenuation is ``exp(-tau/2)`` per antenna and the
visibility of a baseline of two identical antennas is scaled by ``exp(-tau)``.
Invariant I10 pins exactly that product.

The delay is non-dispersive
---------------------------
``tau_trop`` does not depend on frequency, so the phase ``-2 pi nu tau`` is
exactly **linear in frequency** -- which is what distinguishes ``T`` from ``Z``,
whose phase goes as ``1/nu``.  What distinguishes it from ``Kd``, whose phase is
also linear in frequency, is that ``tau_trop`` depends on the *direction*
through the elevation while an instrumental delay is one number per feed.  A
three-way discrimination test asserts all of that in
``tests/unit/test_jones/test_troposphere.py``.

Zenith delays
-------------
``ZHD``, the zenith hydrostatic delay, is either configured directly or computed
by the Saastamoinen formula::

    ZHD = 0.0022768 P_0 / (1 - 0.00266 cos(2 lat) - 0.00028 h_km)

with the surface pressure ``P_0`` in hPa, the site's geodetic latitude, and the
antenna's height above sea level in kilometres -- both of which the resolved
instrument already carries.  ``ZWD``, the zenith wet delay, is configured
directly and has **no** model: every credible wet model needs humidity and
temperature profiles RadioSim does not have, and inventing one would be a
configuration surface with nothing behind it
(``Tier7JonesSciencePlan.md`` Section 4).

Mapping functions
-----------------
``simple`` is the flat-atmosphere ``1 / sin(el)``.  ``niell`` is the Niell (1996)
three-term continued fraction with its published coefficients, latitude- and
season-dependent for the hydrostatic component and latitude-dependent for the
wet one, including the published height correction.  Both diverge at the
horizon, which is why a direction below ``minimum_elevation_deg`` is **rejected**
rather than allowed to produce an unbounded delay (R13).

RadioSim models the zenith opacity as one configured number per run: a
frequency-dependent ``tau_0(nu)`` would need an atmospheric absorption model,
which is the same data ingestion Section 4 excludes.  The number is applied at
every channel, and the docstring says so rather than the field pretending
otherwise.

Delay and opacity are one term with two sub-blocks, not two classes: the model
is a field, not a subclass (Section 9.1).  Stochastic turbulent screens are out
of scope (Section 4).

References
----------
Saastamoinen (1972), in *The Use of Artificial Satellites for Geodesy*,
Geophys. Monogr. Ser. 15, 247 -- the zenith hydrostatic delay.
Niell (1996), J. Geophys. Res. 101, 3227, Tables 3 and 4 -- the mapping
functions and their coefficients.
Thompson, Moran & Swenson (2017), *Interferometry and Synthesis in Radio
Astronomy*, 3rd ed., Sections 13.1-13.2.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, Final, Literal

import numpy as np
import numpy.typing as npt

from radiosim.core.jones.base import JonesTerm
from radiosim.core.jones_errors import (
    InvalidJonesConfigError,
    require_finite_jones_block,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from radiosim.core.jones.directions import DirectionBatch

__all__ = [
    "MAPPING_FUNCTIONS",
    "NIELL_HEIGHT_CORRECTION",
    "NIELL_LATITUDES_DEG",
    "NIELL_PHASE_DAY_OF_YEAR",
    "SPEED_OF_LIGHT_M_PER_S",
    "TroposphereJones",
    "day_of_year_from_mjd",
    "niell_mapping_function",
    "saastamoinen_zenith_hydrostatic_delay_m",
    "simple_mapping_function",
]

#: The speed of light in vacuum, in metres per second.
#:
#: Written out rather than imported from
#: ``radiosim.core.sky.containers.constants``, which holds the same exact value:
#: importing anything from ``radiosim.core.sky`` pulls that package's whole
#: ``__init__`` chain -- loaders, and through them a network client library --
#: into every import of ``radiosim.core.jones``, and
#: ``tests/unit/test_core/test_tier3_beam_cleanup.py`` pins the Jones package's
#: freedom from exactly that.  Importing it from a *sibling term* instead would
#: make the troposphere depend on the ionosphere for no physical reason, so the
#: value is written here as well.  That is safe because ``c`` is *defined* to be
#: this integer by the SI rather than measured, and both copies are asserted
#: equal to the canonical one in this module's tests.
SPEED_OF_LIGHT_M_PER_S: Final[float] = 299_792_458.0

#: The two mapping-function names the schema accepts, in the order Section 20.9
#: names them.
MAPPING_FUNCTIONS: Final[tuple[str, ...]] = ("simple", "niell")

#: The tabular latitudes of Niell (1996) Tables 3 and 4, in degrees.
NIELL_LATITUDES_DEG: Final[tuple[float, ...]] = (15.0, 30.0, 45.0, 60.0, 75.0)

#: Niell (1996) Table 3, hydrostatic: the annual **average** of ``a``, ``b``,
#: ``c`` at each tabular latitude.
NIELL_HYDROSTATIC_AVERAGE: Final[tuple[tuple[float, ...], ...]] = (
    (1.2769934e-3, 1.2683230e-3, 1.2465397e-3, 1.2196049e-3, 1.2045996e-3),
    (2.9153695e-3, 2.9152299e-3, 2.9288445e-3, 2.9022565e-3, 2.9024912e-3),
    (62.610505e-3, 62.837393e-3, 63.721774e-3, 63.824265e-3, 64.258455e-3),
)

#: Niell (1996) Table 3, hydrostatic: the seasonal **amplitude** of ``a``, ``b``,
#: ``c``.  Zero at 15 degrees, where the seasons do not swing the profile.
NIELL_HYDROSTATIC_AMPLITUDE: Final[tuple[tuple[float, ...], ...]] = (
    (0.0, 1.2709626e-5, 2.6523662e-5, 3.4000452e-5, 4.1202191e-5),
    (0.0, 2.1414979e-5, 3.0160779e-5, 7.2562722e-5, 11.723375e-5),
    (0.0, 9.0128400e-5, 4.3497037e-5, 84.795348e-5, 170.37206e-5),
)

#: Niell (1996) Table 3, height correction: ``(a_ht, b_ht, c_ht)``.
NIELL_HEIGHT_CORRECTION: Final[tuple[float, float, float]] = (2.53e-5, 5.49e-3, 1.14e-3)

#: Niell (1996) Table 4, wet: ``a``, ``b``, ``c`` at each tabular latitude.  The
#: wet mapping function has no seasonal term, because water vapour is not in
#: hydrostatic equilibrium and its distribution is not predictable from the
#: season the way the dry profile is.
NIELL_WET: Final[tuple[tuple[float, ...], ...]] = (
    (5.8021897e-4, 5.6794847e-4, 5.8118019e-4, 5.9727542e-4, 6.1641693e-4),
    (1.4275268e-3, 1.5138625e-3, 1.4572752e-3, 1.5007428e-3, 1.7599082e-3),
    (4.3472961e-2, 4.6729510e-2, 4.3908931e-2, 4.4626982e-2, 5.4736038e-2),
)

#: The adopted phase of the seasonal sinusoid: day of year 28, the value Niell
#: obtained when all the radiosonde data were fitted together.
NIELL_PHASE_DAY_OF_YEAR: Final[float] = 28.0

#: The length of the sinusoid's year, in days, as published.
NIELL_YEAR_DAYS: Final[float] = 365.25

#: Saastamoinen (1972): the zenith hydrostatic delay per hPa of surface
#: pressure, in metres, before the latitude and height correction.
SAASTAMOINEN_COEFFICIENT_M_PER_HPA: Final[float] = 0.0022768


def day_of_year_from_mjd(mjd: float) -> float:
    """Return the fractional day of year for a Modified Julian Date.

    Written out with the Fliegel-Van Flandern calendar algorithm rather than
    built on astropy, because it is called once per ``(time, frequency)`` step
    per antenna inside the solver's loops and an astropy ``Time`` construction
    there would cost more than the whole mapping function.  The result is
    compared against astropy in ``tests/unit/test_jones/test_troposphere.py``,
    which is what makes the shortcut checkable rather than merely fast.

    Parameters
    ----------
    mjd : float
        Modified Julian Date (UTC), whole part and fraction.

    Returns
    -------
    float
        ``1.0`` at midnight on 1 January, carrying the fraction of the day, so
        the seasonal sinusoid is continuous across a day boundary.
    """
    whole = math.floor(float(mjd))
    fraction = float(mjd) - whole
    julian_day_number = int(whole) + 2400001
    year, _month, _day = _civil_from_julian_day_number(julian_day_number)
    january_first = _julian_day_number(year, 1, 1)
    return float(julian_day_number - january_first + 1) + fraction


def _civil_from_julian_day_number(julian_day_number: int) -> tuple[int, int, int]:
    """Return ``(year, month, day)`` for a Julian day number (Gregorian)."""
    remainder = julian_day_number + 68569
    century = (4 * remainder) // 146097
    remainder -= (146097 * century + 3) // 4
    year_in_century = (4000 * (remainder + 1)) // 1461001
    remainder += 31 - (1461 * year_in_century) // 4
    month_index = (80 * remainder) // 2447
    day = remainder - (2447 * month_index) // 80
    remainder = month_index // 11
    month = month_index + 2 - 12 * remainder
    year = 100 * (century - 49) + year_in_century + remainder
    return int(year), int(month), int(day)


def _julian_day_number(year: int, month: int, day: int) -> int:
    """Return the Julian day number of a Gregorian calendar date.

    The published form writes the January/February shift as the integer division
    ``(month - 14) / 12``, which is ``-1`` for those two months and ``0``
    otherwise only under C's truncation.  Python's ``//`` floors, so the shift is
    written out rather than divided: a silent off-by-one here would move the day
    of year by one and the seasonal term with it.
    """
    shift = -1 if month <= 2 else 0
    return (
        (1461 * (year + 4800 + shift)) // 4
        + (367 * (month - 2 - 12 * shift)) // 12
        - (3 * ((year + 4900 + shift) // 100)) // 4
        + day
        - 32075
    )


def simple_mapping_function(alt_rad: Any) -> npt.NDArray[np.float64]:
    """Return the flat-atmosphere mapping ``1 / sin(el)``.

    The elementary model, offered because it is the one whose behaviour a reader
    can verify by hand, and because it is the limit the Niell function is
    compared against at high elevation.
    """
    return 1.0 / np.sin(np.asarray(alt_rad, dtype=np.float64))


def _continued_fraction(
    sine_elevation: npt.NDArray[np.float64],
    a: Any,
    b: Any,
    c: Any,
) -> npt.NDArray[np.float64]:
    """Return Niell's normalized three-term continued fraction (his eq. 4)."""
    numerator = 1.0 + a / (1.0 + b / (1.0 + c))
    denominator = sine_elevation + a / (sine_elevation + b / (sine_elevation + c))
    return numerator / denominator


def _interpolate_in_latitude(table: tuple[float, ...], latitude_deg: float) -> float:
    """Linearly interpolate one tabular coefficient at a site latitude.

    Niell tabulates by *absolute* latitude and does not extrapolate: outside
    ``[15, 75]`` the nearest tabular value is used, which is his own prescription
    and is why a polar or equatorial site is defined at all.
    """
    return float(
        np.interp(
            abs(float(latitude_deg)),
            NIELL_LATITUDES_DEG,
            table,
        )
    )


def _seasonal_coefficient(
    average: tuple[float, ...],
    amplitude: tuple[float, ...],
    *,
    latitude_deg: float,
    day_of_year: float,
) -> float:
    """Return one hydrostatic coefficient at a latitude and a day of year.

    ``xi(phi, t) = xi_avg(phi) + xi_amp(phi) cos(2 pi (t - 28) / 365.25)``, with
    half a year added to the phase in the southern hemisphere -- Niell's own
    treatment of the inverted seasons, adopted because no southern radiosonde
    data entered the fit.
    """
    phase_day = NIELL_PHASE_DAY_OF_YEAR
    if latitude_deg < 0.0:
        phase_day += 0.5 * NIELL_YEAR_DAYS
    seasonal = math.cos(
        2.0 * math.pi * (float(day_of_year) - phase_day) / NIELL_YEAR_DAYS
    )
    return _interpolate_in_latitude(
        average, latitude_deg
    ) + seasonal * _interpolate_in_latitude(amplitude, latitude_deg)


def niell_mapping_function(
    alt_rad: Any,
    *,
    component: Literal["hydrostatic", "wet"],
    latitude_deg: float,
    height_m: float = 0.0,
    day_of_year: float = NIELL_PHASE_DAY_OF_YEAR,
) -> npt.NDArray[np.float64]:
    """Return the Niell (1996) mapping function for one component.

    Parameters
    ----------
    alt_rad : array_like
        Direction elevations in radians.
    component : {"hydrostatic", "wet"}
        Which of the two published tables to use.  They are different functions
        of different arguments, not two scalings of one function: the wet one has
        no seasonal term and no height correction, because water vapour is not in
        hydrostatic equilibrium.
    latitude_deg : float
        The site's geodetic latitude.  Its sign selects the seasonal phase; its
        magnitude interpolates the tables.
    height_m : float
        The site's height above sea level, in metres.  Used only by the
        hydrostatic component's published height correction.
    day_of_year : float
        Fractional day of year, from :func:`day_of_year_from_mjd`.  Ignored by
        the wet component.

    Returns
    -------
    ndarray
        ``float64`` mapping factors, ``1`` at zenith for both components.
    """
    sine_elevation = np.sin(np.asarray(alt_rad, dtype=np.float64))
    if component == "wet":
        coefficients = tuple(
            _interpolate_in_latitude(table, latitude_deg) for table in NIELL_WET
        )
        return _continued_fraction(sine_elevation, *coefficients)
    if component != "hydrostatic":
        raise ValueError(f"component must be 'hydrostatic' or 'wet', got {component!r}")
    coefficients = tuple(
        _seasonal_coefficient(
            average,
            amplitude,
            latitude_deg=latitude_deg,
            day_of_year=day_of_year,
        )
        for average, amplitude in zip(
            NIELL_HYDROSTATIC_AVERAGE, NIELL_HYDROSTATIC_AMPLITUDE, strict=True
        )
    )
    mapping = _continued_fraction(sine_elevation, *coefficients)
    height_km = float(height_m) / 1000.0
    if height_km == 0.0:
        return mapping
    correction = (
        1.0 / sine_elevation
        - _continued_fraction(sine_elevation, *NIELL_HEIGHT_CORRECTION)
    ) * height_km
    return mapping + correction


def saastamoinen_zenith_hydrostatic_delay_m(
    *,
    surface_pressure_hpa: float,
    latitude_deg: float,
    height_m: float,
) -> float:
    """Return the Saastamoinen (1972) zenith hydrostatic delay, in metres.

    ``ZHD = 0.0022768 P_0 / (1 - 0.00266 cos(2 lat) - 0.00028 h_km)``.  About
    2.3070 m for standard sea-level pressure at mid latitude (45 degrees, where
    the ``cos(2 lat)`` correction vanishes exactly), which is the number to
    compare any implementation of this formula against --
    ``tests/unit/test_jones/test_troposphere.py`` asserts exactly this value.
    """
    gravity_correction = (
        1.0
        - 0.00266 * math.cos(2.0 * math.radians(float(latitude_deg)))
        - 0.00028 * (float(height_m) / 1000.0)
    )
    return (
        SAASTAMOINEN_COEFFICIENT_M_PER_HPA
        * float(surface_pressure_hpa)
        / gravity_correction
    )


def _per_row_table(field: str, values: Any, rows: int) -> npt.NDArray[np.float64]:
    """Return a validated, read-only ``(rows,)`` float array."""
    table = np.array(values, dtype=np.float64, copy=True, order="C")
    if table.shape != (rows,):
        raise ValueError(
            f"TroposphereJones {field} must have one entry per antenna row, got "
            f"{table.shape} for {rows} rows"
        )
    if not bool(np.isfinite(table).all()):
        raise ValueError(f"TroposphereJones {field} must be finite")
    table.setflags(write=False)
    return table


class TroposphereJones(JonesTerm):
    """Tropospheric delay and opacity ``T`` (Section 20.9).

    Constructed only by
    :func:`~radiosim.core.jones_terms.resolve_jones_terms`, from a validated
    ``jones.T`` block plus the resolved instrument's latitude and antenna
    heights.

    Parameters
    ----------
    zenith_hydrostatic_delay_m, zenith_wet_delay_m : ndarray
        ``(n_antenna_rows,)`` zenith delays in metres, in solver antenna-row
        order.  They are per antenna and not per array because the Saastamoinen
        formula reads each antenna's own height.
    mapping_function : {"simple", "niell"}
        Which mapping function both components use.
    latitude_deg : float
        The array's geodetic latitude, for the Niell tables.
    heights_m : ndarray
        ``(n_antenna_rows,)`` heights above sea level in metres, for the Niell
        hydrostatic height correction.
    zenith_opacity : float or None
        The dimensionless zenith opacity on **power**.  ``None`` is no opacity
        at all, which is the case in which ``T`` is unitary.
    minimum_elevation_deg : float
        The elevation below which the mapping function is refused (R13).  ``0``
        accepts every direction the horizon mask passes.

    Raises
    ------
    ValueError
        A shape mismatch, an unknown mapping function, a negative opacity, or a
        minimum elevation outside ``[0, 90)``.
    """

    def __init__(
        self,
        *,
        zenith_hydrostatic_delay_m: npt.NDArray[np.float64],
        zenith_wet_delay_m: npt.NDArray[np.float64],
        mapping_function: str,
        latitude_deg: float,
        heights_m: npt.NDArray[np.float64],
        zenith_opacity: float | None,
        minimum_elevation_deg: float,
    ) -> None:
        hydrostatic = np.array(
            zenith_hydrostatic_delay_m, dtype=np.float64, copy=True, order="C"
        )
        if hydrostatic.ndim != 1 or hydrostatic.size < 1:
            raise ValueError(
                "TroposphereJones zenith_hydrostatic_delay_m must have one entry "
                f"per antenna row, got shape {hydrostatic.shape}"
            )
        rows = int(hydrostatic.size)
        if not bool(np.isfinite(hydrostatic).all()):
            raise ValueError(
                "TroposphereJones zenith_hydrostatic_delay_m must be finite"
            )
        hydrostatic.setflags(write=False)
        if mapping_function not in MAPPING_FUNCTIONS:
            raise ValueError(
                f"TroposphereJones mapping_function must be one of "
                f"{MAPPING_FUNCTIONS}, got {mapping_function!r}"
            )
        latitude = float(latitude_deg)
        if not math.isfinite(latitude) or not -90.0 <= latitude <= 90.0:
            raise ValueError("TroposphereJones latitude_deg must be in [-90, 90]")
        opacity = None if zenith_opacity is None else float(zenith_opacity)
        if opacity is not None and (not math.isfinite(opacity) or opacity < 0.0):
            raise ValueError(
                "TroposphereJones zenith_opacity must be non-negative; a negative "
                "opacity would amplify"
            )
        minimum_elevation = float(minimum_elevation_deg)
        if not math.isfinite(minimum_elevation) or not 0.0 <= minimum_elevation < 90.0:
            raise ValueError(
                "TroposphereJones minimum_elevation_deg must be in [0, 90)"
            )
        self._zenith_hydrostatic_delay_m = hydrostatic
        self._zenith_wet_delay_m = _per_row_table(
            "zenith_wet_delay_m", zenith_wet_delay_m, rows
        )
        self._heights_m = _per_row_table("heights_m", heights_m, rows)
        self._mapping_function = str(mapping_function)
        self._latitude_deg = latitude
        self._zenith_opacity = opacity
        self._minimum_elevation_deg = minimum_elevation

    # ------------------------------------------------------------------ shape

    @property
    def name(self) -> str:
        return "T"

    @property
    def term_status(self) -> str:
        """``"implemented"``: ``T`` carries the exact Section 20.9 mathematics."""
        return "implemented"

    @property
    def mapping_function(self) -> str:
        """The resolved mapping function name."""
        return self._mapping_function

    @property
    def zenith_opacity(self) -> float | None:
        """The resolved zenith opacity on power, or ``None`` for no opacity."""
        return self._zenith_opacity

    @property
    def minimum_elevation_deg(self) -> float:
        """The elevation floor below which R13 refuses to evaluate."""
        return self._minimum_elevation_deg

    @property
    def zenith_hydrostatic_delay_m(self) -> npt.NDArray[np.float64]:
        """One zenith hydrostatic delay per antenna row, read-only."""
        return self._zenith_hydrostatic_delay_m

    @property
    def zenith_wet_delay_m(self) -> npt.NDArray[np.float64]:
        """One zenith wet delay per antenna row, read-only."""
        return self._zenith_wet_delay_m

    @property
    def is_direction_dependent(self) -> bool:
        """``True``: both the mapping function and the opacity read the elevation."""
        return True

    @property
    def is_time_dependent(self) -> bool:
        """``True``: a direction's elevation changes with time, and so does its delay."""
        return True

    @property
    def is_frequency_dependent(self) -> bool:
        """``True`` for any non-zero zenith delay, and ``False`` for opacity alone.

        The delay is a phase slope in frequency; the opacity, as RadioSim models
        it, is one number for the whole band.  A ``T`` configured with opacity
        only really is achromatic, so claiming chromaticity for it would be a
        false ``True`` -- the vacuous-flag failure mode invariant I2 exists to
        prevent.
        """
        return self._has_delay

    @property
    def _has_delay(self) -> bool:
        return bool(
            np.any(self._zenith_hydrostatic_delay_m != 0.0)
            or np.any(self._zenith_wet_delay_m != 0.0)
        )

    @property
    def _has_opacity(self) -> bool:
        return self._zenith_opacity is not None and self._zenith_opacity != 0.0

    def is_diagonal(self) -> bool:
        """``True`` always: ``T`` is a scalar times ``I2`` by construction."""
        return True

    def is_scalar(self) -> bool:
        """``True`` always: both factors multiply the identity (Section 20.9)."""
        return True

    def is_unitary(self) -> bool:
        """``True`` only with the opacity disabled.

        This is the flag that separates ``T`` from every other propagation term
        in Tier 7: an absorbing atmosphere removes power, so ``T T^H = |a|^2 I2``
        with ``|a| < 1``, and a term that claimed unitarity here would be
        claiming the sky is brighter than the correlator sees.
        """
        return not self._has_opacity

    def is_identity(self) -> bool:
        """``True`` when both zenith delays and the opacity are exactly zero.

        R7's condition verbatim: a transparent atmosphere of zero excess path
        cannot change the visibilities at any frequency, direction or time.
        """
        return not self._has_delay and not self._has_opacity

    # --------------------------------------------------------------- physics

    def reject_low_elevation(self, directions: DirectionBatch) -> None:
        """Raise R13 when a direction below the configured floor reaches ``T``.

        Raised at evaluation and not at resolution, because the condition R13
        states -- "a direction survives the horizon mask below
        ``minimum_elevation_deg``" -- is a statement about directions, and the
        first direction exists only when the solver resolves one for a
        ``(time, frequency)`` step.  Both mapping functions grow without bound as
        ``el -> 0``, and RadioSim refuses rather than writing an unbounded delay
        into a visibility.
        """
        if self._minimum_elevation_deg <= 0.0 or directions.n_dir == 0:
            return
        lowest_deg = math.degrees(float(np.min(directions.alt_rad)))
        if lowest_deg >= self._minimum_elevation_deg:
            return
        raise InvalidJonesConfigError(
            f"jones.T.minimum_elevation_deg={self._minimum_elevation_deg} excludes "
            f"no direction, but the mapping function diverges below "
            f"{self._minimum_elevation_deg} deg; raise the minimum elevation or the "
            "horizon mask."
        )

    def mapping_factors(
        self,
        alt_rad: Any,
        *,
        antenna_idx: int,
        time_mjd: float,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Return ``(m_h, m_w)`` for one antenna over a set of elevations.

        Public because the two factors are what the invariant tests compare
        against the published Niell values and the ``1/sin(el)`` limit.
        """
        row = self._require_row(antenna_idx)
        if self._mapping_function == "simple":
            simple = simple_mapping_function(alt_rad)
            return simple, simple
        day_of_year = day_of_year_from_mjd(time_mjd)
        hydrostatic = niell_mapping_function(
            alt_rad,
            component="hydrostatic",
            latitude_deg=self._latitude_deg,
            height_m=float(self._heights_m[row]),
            day_of_year=day_of_year,
        )
        wet = niell_mapping_function(
            alt_rad,
            component="wet",
            latitude_deg=self._latitude_deg,
            day_of_year=day_of_year,
        )
        return hydrostatic, wet

    def delay_s(
        self,
        antenna_idx: int,
        directions: DirectionBatch,
        time_mjd: float,
    ) -> npt.NDArray[np.float64]:
        """Return this antenna's excess tropospheric delay per direction, in seconds.

        ``(ZHD m_h + ZWD m_w) / c``.  Frequency-free by construction: the neutral
        atmosphere's excess path is non-dispersive at radio wavelengths, and that
        is the property the three-way delay discrimination test rests on.
        """
        row = self._require_row(antenna_idx)
        hydrostatic, wet = self.mapping_factors(
            directions.alt_rad, antenna_idx=row, time_mjd=time_mjd
        )
        excess_path_m = (
            float(self._zenith_hydrostatic_delay_m[row]) * hydrostatic
            + float(self._zenith_wet_delay_m[row]) * wet
        )
        return excess_path_m / SPEED_OF_LIGHT_M_PER_S

    def opacity_attenuation(
        self,
        directions: DirectionBatch,
    ) -> npt.NDArray[np.float64]:
        """Return the **voltage** attenuation ``exp(-tau_0 / (2 sin el))``.

        One factor of two, stated once: ``tau_0`` is a power opacity and this is
        a voltage Jones matrix, so a baseline of two identical antennas is scaled
        by ``exp(-tau_0 / sin el)`` in visibility amplitude (invariant I10).
        """
        if not self._has_opacity:
            return np.ones(directions.n_dir, dtype=np.float64)
        assert self._zenith_opacity is not None
        airmass = 1.0 / np.sin(np.asarray(directions.alt_rad, dtype=np.float64))
        return np.exp(-0.5 * self._zenith_opacity * airmass)

    def _require_row(self, antenna_idx: int) -> int:
        row = int(antenna_idx)
        if row < 0 or row >= self._zenith_hydrostatic_delay_m.size:
            raise IndexError(
                f"TroposphereJones has {self._zenith_hydrostatic_delay_m.size} "
                f"antenna rows; row {row} is out of range."
            )
        return row

    def compute_jones_batch(
        self,
        *,
        antenna_idx: int,
        directions: DirectionBatch,
        frequency_hz: float,
        freq_idx: int,
        time_mjd: float,
        time_idx: int,
        backend: Any,
        dtype: Any,
    ) -> Any:
        """Return this antenna's ``(n_dir, 2, 2)`` tropospheric matrices.

        Direction-dependent, so the return is one matrix per direction
        (invariant I3).  ``time_mjd`` is read here and not only through the
        direction batch, because the Niell hydrostatic coefficients carry a
        seasonal term and the day of year is not recoverable from a set of
        elevations.

        All arithmetic runs on the host over ``float64`` arrays; no array value
        is branched on and no device value is read back (Section 17.2).
        """
        self.reject_low_elevation(directions)
        delay = self.delay_s(antenna_idx, directions, time_mjd)
        scalar = self.opacity_attenuation(directions) * np.exp(
            -2j * np.pi * float(frequency_hz) * delay
        )
        block = np.zeros((scalar.size, 2, 2), dtype=np.complex128)
        block[:, 0, 0] = scalar
        block[:, 1, 1] = scalar
        require_finite_jones_block(self.name, block)
        return backend.xp.array(block, dtype=dtype)

    def get_config(self) -> dict[str, Any]:
        """Include the resolved atmosphere in the term's record."""
        config = super().get_config()
        config["zenith_hydrostatic_delay_m"] = [
            float(value) for value in self._zenith_hydrostatic_delay_m
        ]
        config["zenith_wet_delay_m"] = [
            float(value) for value in self._zenith_wet_delay_m
        ]
        config["mapping_function"] = self._mapping_function
        config["latitude_deg"] = self._latitude_deg
        config["zenith_opacity"] = self._zenith_opacity
        config["minimum_elevation_deg"] = self._minimum_elevation_deg
        return config
