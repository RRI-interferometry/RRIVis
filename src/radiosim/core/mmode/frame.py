r"""The frozen-CIRS rigid-ERA frame and the frozen analytic horizon oracle.

``docs/development/sci004_mmode_design.md`` Section 4.1 fixes the accepted
m-mode frame literal ``radiosim.frozen-cirs-rigid-era.v1`` as a *site-specific
and executable* construction, not a shorthand for an unspecified Astropy
transform.  The present direct path transforms ICRS directions to ``AltAz`` at
every UTC instant and reconstructs an operational apparent direction; that is
not a one-parameter rigid rotation about a fixed celestial ``z`` axis, and
calling it m-mode compatible would recreate the frame ambiguity ``SCI-007``
closed as a bounded limitation.

Four things are therefore normative here.

1. **One polar-motion unit conversion.**  ``pm_xy`` returns arcseconds,
   ``float(erfa.DAS2R)`` converts once, and ``erfa.pom00`` receives radians.
   Passing the unitless arcsecond values, applying ``DAS2R`` twice, or taking a
   different unit path is forbidden, and both unit literals are serialized.
2. **A geocentric anchor.**  The CIRS frame is explicitly ``CIRS(obstime=t0)``;
   passing ``location=site`` is forbidden because it would add a different
   topocentric/diurnal-aberration model.
3. **The SOFA passive attitude.**  ``[ITRS] = RPOM0 R3(ERA) [CIRS]``, matching
   ``c2tcio``.  Because ``R3(a) R3(b) = R3(a+b)``, ``T(alpha) = T(0) R3(alpha)``
   and Section 6's ``exp(+i m alpha)`` transfer law follows.  No transpose,
   longitude sign, or fitted phase offset is allowed.
4. **A public-Astropy tangent oracle.**  The one-time ICRS-to-CIRS tangent
   transport is Richardson-extrapolated from public ``SkyCoord`` transforms at
   ``h = 2**-12 rad``.  A private frame-graph helper or a position-angle
   shortcut is explicitly not an equivalent authority.

Section 12.1's frozen half also lives here.  For each frozen CIRS direction the
attitude makes

.. math::

    \sin(\operatorname{alt}(u)) = A\cos(2\pi u) + B\sin(2\pi u) + C,

whose coefficients follow in closed form from the retained matrices.  The
topology decision is taken from the **exact integer ratios** of ``A``, ``B`` and
``C`` -- a floating epsilon never classifies it -- and the two transverse roots
are placed in certified sign-changing brackets refined with the exact-rational
interval kernel in :mod:`radiosim.core.mmode.time`.

The Section 12.1 *operational* half is a separate, and currently blocked,
construction; :class:`OperationalEnclosureUnavailable` documents precisely what
it requires and why this slice cannot supply it.
"""

from __future__ import annotations

import bisect
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Final, Literal

import numpy as np

from radiosim.core.mmode.time import (
    ERA_CENTER_LIMIT_RAD,
    ERA_STEP_LIMIT_RAD,
    UT1_UTC_ROUNDTRIP_LIMIT_SECONDS,
    certified_two_pi_bracket,
    certified_two_pi_trig,
    installed_iers_context,
    round_down,
    round_up,
)
from radiosim.core.mmode.types import (
    MMODE_FRAME_MODEL,
    TAU,
    canonical_json,
    canonical_rational,
    decode_f64be,
    f64be,
    object_digest,
)

__all__ = [
    "HORIZON_ROOT_WIDTH_RAD",
    "SCAN_DERIVATIVE_CEILING_PER_TURN",
    "SCAN_SCHEMA",
    "PM_SOURCE_UNIT",
    "POM00_ARGUMENT_UNIT",
    "TANGENT_HALVING_LIMIT_RAD",
    "TANGENT_STEP_RAD",
    "FrozenFrame",
    "FrozenHorizonTrajectory",
    "GeocentricCirsAnchor",
    "HorizonRootEnclosure",
    "MModeHorizonUnresolved",
    "OperationalHorizonScan",
    "OperationalScanRejected",
    "scan_operational_horizon",
    "TransportedTangentFrame",
    "strict_horizon_indicator",
    "strict_horizon_visible",
    "build_frozen_frame",
    "frozen_horizon_trajectory",
    "passive_r3",
]

#: Section 4.1's two exact unit literals.
PM_SOURCE_UNIT: Final = "arcsec"
POM00_ARGUMENT_UNIT: Final = "rad"

#: Section 4.1's fixed Richardson step and halving bound.
TANGENT_STEP_RAD: Final[float] = 2.0**-12
TANGENT_HALVING_LIMIT_RAD: Final[float] = 2e-10

#: Section 12.1's certified frozen root width and numerator residual.  The
#: frozen analytic census keeps its exact-rational construction unchanged.
HORIZON_ROOT_WIDTH_RAD: Final[float] = 1e-13
HORIZON_ROOT_RESIDUAL: Final[float] = 2e-13

# --- Section 12.1 certified-ceiling scan constants ---------------------------
#
# These are the operational census's own new fixed constants, not a widening of
# any frozen-model constant.  Each one is a ``constant_rows`` entry of the scan
# manifest.

#: The design-frozen derivative ceiling, in ``turn**-1``.  The operational
#: direction is transported by the rigid Earth rotation at exactly one cycle per
#: turn composed with the operational corrections -- polar-motion drift, UT1
#: interpolation, precession-nutation evolution, annual and diurnal aberration,
#: and light deflection -- whose combined angular rates over one cycle the cited
#: IERS Conventions (2010) magnitudes bound at well below ``1e-3`` of the rigid
#: rate.  ``L_op`` exceeds ``2*pi*(1+1e-3)``.  It is never fitted or
#: configurable; a trajectory violating it would violate the frozen attitude
#: model itself and is outside the certified regime.
SCAN_DERIVATIVE_CEILING_PER_TURN: Final[float] = 6.2895

#: The uniform initial cell width of the scan partition, in turns.
SCAN_INITIAL_SPACING_TURN: Final[Fraction] = Fraction(1, 2**12)

#: The certified operational root-enclosure width, in radians.
SCAN_ROOT_WIDTH_RAD: Final[float] = 1e-11

#: The census orientation probe offset, in turns, and its magnitude floor.
SCAN_PROBE_OFFSET_TURN: Final[Fraction] = Fraction(1, 10**8)
SCAN_PROBE_MAGNITUDE_FLOOR: Final[float] = 1e-10

#: The deep-tangency signature: a cell reaching this exact width with neither
#: classification rejects the entire certificate.
SCAN_UNRESOLVED_WIDTH_TURN: Final[Fraction] = Fraction(1, 2**44)

#: The retained operational root's numerator residual bound, consistent with
#: ``L_op * (1e-11/2/(2*pi))``.
SCAN_ROOT_RESIDUAL: Final[float] = 5e-12

#: The scan's schema and algorithm literal.
SCAN_SCHEMA: Final = "radiosim.mmode-operational-horizon-scan.v1"

#: Section 12.1's three exact terminal-cell classifications.
SCAN_CLASSIFICATIONS: Final[tuple[str, ...]] = (
    "ceiling_excludes_root",
    "scan_crossing",
    "excluded_upper_endpoint",
)


class MModeHorizonUnresolved(RuntimeError):
    """Section 12.1's typed ``mmode_horizon_unresolved`` rejection.

    Raised for a trajectory that lies identically on the horizon, for the single
    stationary/tangent equality, and for any cell that proves neither root-free
    nor uniquely transverse.  No root is silently merged or discarded.
    """

    issue_code: Final = "mmode_horizon_unresolved"
    exact_message: Final = (
        "execution.simulator='mmode' could not certify complete horizon-root "
        "isolation; tangent, identically-zero, and unresolved intervals are "
        "rejected."
    )

    def __init__(self, detail: str = "") -> None:
        message = self.exact_message if not detail else f"{self.exact_message} {detail}"
        super().__init__(message)


# ---------------------------------------------------------------------------
# Section 6 shared horizon predicate
# ---------------------------------------------------------------------------

#: Section 14.0's ``horizon_predicate`` convention literal.
HORIZON_PREDICATE_ID: Final = "sin_altitude_strictly_greater_than_zero"


def strict_horizon_visible(local_up: Any) -> np.ndarray:
    r"""Return Section 6's strict horizon predicate ``1[alt(n) > 0]``.

    .. math::

        H(\hat n)=\mathbf 1[\operatorname{alt}(\hat n)>0]
        =\begin{cases}1,&\hat n\cdot e_U>0\\0,&\hat n\cdot e_U\le 0\end{cases}

    Equality is **excluded**, matching both maintained direct solvers: no
    epsilon, beam cutoff, or half-weight at the horizon is allowed.

    Section 6 (as corrected) makes this one shared code object normative:
    "the private direct oracles and the harmonic-transfer kernel construction
    invoke the identical tested implementation of this horizon predicate -- one
    shared code object, never a re-derivation of the same formula -- so a
    horizon-application defect cannot differ between the compared models and
    thereby escape the Section 7.3 tier-1a horizon-free shell, the tier-2
    budgets, and the Section 12 direct machinery simultaneously."

    Parameters
    ----------
    local_up : array-like
        The direction's component along the fixed local Up column of
        Section 4.1's ITRS basis -- equivalently ``sin(alt)``.

    Returns
    -------
    ndarray of bool
        ``True`` exactly where the direction is strictly above the horizon.
    """
    return np.asarray(local_up, dtype=np.float64) > 0.0


def strict_horizon_indicator(local_up: Any) -> np.ndarray:
    """Return :func:`strict_horizon_visible` as a ``0.0``/``1.0`` factor.

    This is the multiplicative form the Section 6 kernel applies, and it is a
    thin cast of the one shared predicate rather than a second comparison.
    """
    return strict_horizon_visible(local_up).astype(np.float64)


# ---------------------------------------------------------------------------
# Section 4.1 attitude algebra
# ---------------------------------------------------------------------------


def passive_r3(theta: float) -> np.ndarray:
    """Return the SOFA passive rotation about the third axis.

    ``R3(theta)`` rotates *coordinates*; the equivalent active sky rotation in
    fixed ITRS is ``A_z(-theta) = R3(+theta)``.
    """
    cosine = math.cos(theta)
    sine = math.sin(theta)
    return np.array(
        [[cosine, sine, 0.0], [-sine, cosine, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def _enu_basis(longitude_rad: float, latitude_rad: float) -> dict[str, np.ndarray]:
    """Return Section 4.1's exact ``(East, North, Up)`` ITRS triad."""
    sin_lon = math.sin(longitude_rad)
    cos_lon = math.cos(longitude_rad)
    sin_lat = math.sin(latitude_rad)
    cos_lat = math.cos(latitude_rad)
    return {
        "east": np.array([-sin_lon, cos_lon, 0.0], dtype=np.float64),
        "north": np.array(
            [-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat], dtype=np.float64
        ),
        "up": np.array(
            [cos_lat * cos_lon, cos_lat * sin_lon, sin_lat], dtype=np.float64
        ),
    }


@dataclass(frozen=True, slots=True)
class GeocentricCirsAnchor:
    """The retained record of Section 4.1's geocentric ``CIRS(obstime=t0)``.

    ``location`` is the observing location that was **supplied** to the anchor
    constructor, and it is always ``None``: Section 4.1 forbids passing
    ``location=site`` here because that would add a different
    topocentric/diurnal-aberration model.  Astropy's own ``CIRS`` frame carries
    a geocentre default rather than ``None`` for that attribute, so the record
    states the construction rather than re-reading the default; ``astropy_frame``
    is the exact frame object every public transform is taken against.
    """

    obstime: Any
    astropy_frame: Any
    location: None = None


@dataclass(frozen=True, slots=True)
class TransportedTangentFrame:
    """One ICRS direction and its North/East tangent columns, in CIRS."""

    direction_cirs: np.ndarray
    north_cirs: np.ndarray
    east_cirs: np.ndarray
    step_rad: float
    method: str


@dataclass(frozen=True, slots=True)
class FrozenFrame:
    """The one frozen-CIRS rigid-ERA frame object of a run.

    Section 4.1 requires the zenith phase centre, field rotation, beam, fringe,
    ground-stationary Jones terms and the strict horizon factor to be derived
    from *this* object.  The harmonic solver and the private frozen-frame direct
    oracle must receive that identical object; reconstructing the attitude
    independently is a validation failure.
    """

    frame_model: str
    start_time_iso: str
    longitude_deg: float
    latitude_deg: float
    height_m: float
    pm_source_unit: str
    pom00_argument_unit: str
    xp0_arcsec: float
    yp0_arcsec: float
    das2r_rad_per_arcsec: float
    xp0_rad: float
    yp0_rad: float
    sp0_rad: float
    rpom0: np.ndarray
    era0_rad: float
    cirs_to_itrs_anchor: np.ndarray
    local_east_itrs: np.ndarray
    local_north_itrs: np.ndarray
    local_up_itrs: np.ndarray
    itrs_xyz_m: np.ndarray
    tt_two_part: tuple[float, float]
    ut1_two_part: tuple[float, float]
    utc_two_part: tuple[float, float]
    cirs_frame: Any
    iers_resource_path: str
    iers_table_sha256: str
    iers_package_version: str
    site_manifest: dict[str, Any]
    site_sha256: str
    frame_matrix_manifest: dict[str, Any]
    frame_matrix_sha256: str

    @property
    def cirs_frame_is_geocentric(self) -> bool:
        """Return ``True``: the anchor carries no observing ``location``."""
        return self.cirs_frame.location is None

    @property
    def iers_auto_download(self) -> bool:
        """Return ``False``: no network lookup is ever permitted."""
        return False

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

    @property
    def tangent_halving_limit_rad(self) -> float:
        """Return Section 4.1's fixed tangent-halving limit."""
        return TANGENT_HALVING_LIMIT_RAD

    def attitude_at(self, relative_phase_rad: float) -> np.ndarray:
        """Return ``T(alpha) = T(0) R3(alpha)``, the rigid group composition."""
        return np.asarray(self.cirs_to_itrs_anchor) @ passive_r3(
            float(relative_phase_rad)
        )

    def transport_tangent_frame(
        self,
        *,
        ra_deg: float,
        dec_deg: float,
        step_rad: float = TANGENT_STEP_RAD,
    ) -> TransportedTangentFrame:
        """Transport one ICRS direction and its tangent columns into CIRS.

        The Jacobian is the Richardson extrapolation Section 4.1 prints,

        .. math::

            d_q(h)=\\frac{c_q(+h)-c_q(-h)}{2\\sin h},\\qquad
            d_q=\\frac{4d_q(h/2)-d_q(h)}{3},

        built from **public** Astropy coordinate objects and transforms only.
        The result is projected with ``I - c c^T``, North is normalized, East is
        Gram-Schmidt'ed against North, and the declared North/East handedness
        ``dot(cross(N, E), c) < 0`` is required.
        """
        return _transport_tangent_frame(
            self, ra_deg=ra_deg, dec_deg=dec_deg, step_rad=float(step_rad)
        )

    def cirs_directions(self, ra_rad: Any, dec_rad: Any) -> np.ndarray:
        """Transform ICRS unit directions into the frozen CIRS anchor frame."""
        right_ascension = np.atleast_1d(np.asarray(ra_rad, dtype=np.float64))
        declination = np.atleast_1d(np.asarray(dec_rad, dtype=np.float64))
        vectors = np.stack(
            (
                np.cos(declination) * np.cos(right_ascension),
                np.cos(declination) * np.sin(right_ascension),
                np.sin(declination),
            ),
            axis=-1,
        )
        return _transform_icrs_cartesian_to_cirs(self, vectors)


# ---------------------------------------------------------------------------
# Section 4.1 construction
# ---------------------------------------------------------------------------


def build_frozen_frame(
    *,
    start_time: str,
    longitude_deg: float,
    latitude_deg: float,
    height_m: float,
) -> FrozenFrame:
    """Build the one Section 4.1 frozen-CIRS rigid-ERA frame of a run.

    Every operation runs inside the installed bundled-IERS context of
    Section 3.1, so no network lookup or implicit Astropy table selection can
    occur.
    """
    import erfa
    from astropy import units as u
    from astropy.coordinates import CIRS, EarthLocation
    from astropy.time import Time

    from radiosim.core.mmode.time import installed_iers

    with installed_iers_context() as resolved:
        anchor = Time(str(start_time).strip(), scale="utc")
        anchor_ut1 = anchor.ut1
        anchor_tt = anchor.tt
        ut1_two_part = (float(anchor_ut1.jd1), float(anchor_ut1.jd2))
        tt_two_part = (float(anchor_tt.jd1), float(anchor_tt.jd2))
        utc_two_part = (float(anchor.jd1), float(anchor.jd2))

        # Section 4.1 step 1: polar motion has exactly one unit conversion.
        xp_quantity, yp_quantity = resolved.table.pm_xy(anchor)
        xp0_arcsec = float(xp_quantity.to_value(u.arcsec))
        yp0_arcsec = float(yp_quantity.to_value(u.arcsec))
        das2r_rad_per_arcsec = float(erfa.DAS2R)
        xp0_rad = xp0_arcsec * das2r_rad_per_arcsec
        yp0_rad = yp0_arcsec * das2r_rad_per_arcsec
        sp0_rad = float(erfa.sp00(*tt_two_part))
        rpom0 = np.asarray(
            erfa.pom00(xp0_rad, yp0_rad, sp0_rad), dtype=np.float64
        ).reshape(3, 3)

        # Section 4.1 step 3: the SOFA passive attitude at the anchor.
        era0_rad = float(erfa.era00(*ut1_two_part))
        anchor_matrix = rpom0 @ passive_r3(era0_rad)

        # Section 4.1 step 1: the geocentric CIRS frame, deliberately without a
        # site ``location``.
        cirs_frame = GeocentricCirsAnchor(
            obstime=anchor, astropy_frame=CIRS(obstime=anchor)
        )

        # Section 4.1 step 4: the fixed local ITRS basis.
        longitude = float(longitude_deg)
        latitude = float(latitude_deg)
        height = float(height_m)
        basis = _enu_basis(math.radians(longitude), math.radians(latitude))

        site = EarthLocation.from_geodetic(
            lon=longitude * u.deg, lat=latitude * u.deg, height=height * u.m
        )
        itrs_xyz = np.asarray(
            [
                float(site.x.to_value(u.m)),
                float(site.y.to_value(u.m)),
                float(site.z.to_value(u.m)),
            ],
            dtype=np.float64,
        )

    site_manifest = {
        "schema_version": "radiosim.mmode-site.v1",
        "longitude_deg_f64be": f64be(longitude),
        "latitude_deg_f64be": f64be(latitude),
        "height_m_f64be": f64be(height),
        "itrs_xyz_m_f64be": [f64be(value) for value in itrs_xyz.tolist()],
    }
    site_sha256 = object_digest("radiosim.mmode-site.v1", site_manifest)

    frame_matrix_manifest = {
        "schema_version": "radiosim.mmode-frame-matrices.v1",
        "era0_rad_f64be": f64be(era0_rad),
        "rpom0_f64be": [f64be(value) for value in rpom0.reshape(-1).tolist()],
        "cirs_to_itrs_anchor_f64be": [
            f64be(value) for value in anchor_matrix.reshape(-1).tolist()
        ],
        "local_east_itrs_f64be": [f64be(value) for value in basis["east"].tolist()],
        "local_north_itrs_f64be": [f64be(value) for value in basis["north"].tolist()],
        "local_up_itrs_f64be": [f64be(value) for value in basis["up"].tolist()],
    }
    frame_matrix_sha256 = object_digest(
        "radiosim.mmode-frame-matrices.v1", frame_matrix_manifest
    )

    resolved = installed_iers()
    return FrozenFrame(
        frame_model=MMODE_FRAME_MODEL,
        start_time_iso=str(anchor.utc.isot),
        longitude_deg=longitude,
        latitude_deg=latitude,
        height_m=height,
        pm_source_unit=PM_SOURCE_UNIT,
        pom00_argument_unit=POM00_ARGUMENT_UNIT,
        xp0_arcsec=xp0_arcsec,
        yp0_arcsec=yp0_arcsec,
        das2r_rad_per_arcsec=das2r_rad_per_arcsec,
        xp0_rad=xp0_rad,
        yp0_rad=yp0_rad,
        sp0_rad=sp0_rad,
        rpom0=rpom0,
        era0_rad=era0_rad,
        cirs_to_itrs_anchor=anchor_matrix,
        local_east_itrs=basis["east"],
        local_north_itrs=basis["north"],
        local_up_itrs=basis["up"],
        itrs_xyz_m=itrs_xyz,
        tt_two_part=tt_two_part,
        ut1_two_part=ut1_two_part,
        utc_two_part=utc_two_part,
        cirs_frame=cirs_frame,
        iers_resource_path=resolved.resource_path,
        iers_table_sha256=resolved.table_sha256,
        iers_package_version=resolved.package_version,
        site_manifest=site_manifest,
        site_sha256=site_sha256,
        frame_matrix_manifest=frame_matrix_manifest,
        frame_matrix_sha256=frame_matrix_sha256,
    )


def _transform_icrs_cartesian_to_cirs(
    frame: FrozenFrame, vectors: np.ndarray
) -> np.ndarray:
    """Transform unit ICRS Cartesian vectors through the public Astropy API."""
    from astropy.coordinates import CartesianRepresentation, SkyCoord

    values = np.atleast_2d(np.asarray(vectors, dtype=np.float64))
    with installed_iers_context():
        coordinates = SkyCoord(
            CartesianRepresentation(
                values[:, 0], values[:, 1], values[:, 2], copy=True
            ),
            frame="icrs",
        )
        transformed = coordinates.transform_to(frame.cirs_frame.astropy_frame)
        cartesian = transformed.cartesian
        result = np.stack(
            (
                np.asarray(cartesian.x.value, dtype=np.float64),
                np.asarray(cartesian.y.value, dtype=np.float64),
                np.asarray(cartesian.z.value, dtype=np.float64),
            ),
            axis=-1,
        )
    norms = np.linalg.norm(result, axis=-1, keepdims=True)
    return result / norms


def _transport_tangent_frame(
    frame: FrozenFrame, *, ra_deg: float, dec_deg: float, step_rad: float
) -> TransportedTangentFrame:
    """Evaluate Section 4.1's public-Astropy Richardson tangent oracle."""
    right_ascension = math.radians(float(ra_deg))
    declination = math.radians(float(dec_deg))
    direction = np.array(
        [
            math.cos(declination) * math.cos(right_ascension),
            math.cos(declination) * math.sin(right_ascension),
            math.sin(declination),
        ],
        dtype=np.float64,
    )
    # The canonical North/East tangent columns, with catalogue longitude
    # retained as the gauge at an exact coordinate pole.
    north = np.array(
        [
            -math.sin(declination) * math.cos(right_ascension),
            -math.sin(declination) * math.sin(right_ascension),
            math.cos(declination),
        ],
        dtype=np.float64,
    )
    east = np.array(
        [-math.sin(right_ascension), math.cos(right_ascension), 0.0],
        dtype=np.float64,
    )

    steps = (float(step_rad), float(step_rad) / 2.0)
    probes: list[np.ndarray] = [direction]
    for column in (north, east):
        for step in steps:
            probes.append(direction * math.cos(step) + column * math.sin(step))
            probes.append(direction * math.cos(step) - column * math.sin(step))
    transformed = _transform_icrs_cartesian_to_cirs(frame, np.stack(probes))

    center = transformed[0]
    derivatives: list[np.ndarray] = []
    cursor = 1
    for _ in (north, east):
        coarse = (transformed[cursor] - transformed[cursor + 1]) / (
            2.0 * math.sin(steps[0])
        )
        fine = (transformed[cursor + 2] - transformed[cursor + 3]) / (
            2.0 * math.sin(steps[1])
        )
        derivatives.append((4.0 * fine - coarse) / 3.0)
        cursor += 4

    projector = np.eye(3, dtype=np.float64) - np.outer(center, center)
    north_cirs = projector @ derivatives[0]
    north_cirs = north_cirs / float(np.linalg.norm(north_cirs))
    east_cirs = projector @ derivatives[1]
    east_cirs = east_cirs - float(np.dot(east_cirs, north_cirs)) * north_cirs
    east_cirs = east_cirs / float(np.linalg.norm(east_cirs))

    if float(np.dot(np.cross(north_cirs, east_cirs), center)) >= 0.0:
        raise ValueError(
            "transported tangent frame lost the declared North/East handedness"
        )
    return TransportedTangentFrame(
        direction_cirs=center,
        north_cirs=north_cirs,
        east_cirs=east_cirs,
        step_rad=float(step_rad),
        method="public_astropy_richardson_v1",
    )


# ---------------------------------------------------------------------------
# Section 12.1 certified-ceiling scan (the operational census)
# ---------------------------------------------------------------------------


class OperationalScanRejected(MModeHorizonUnresolved):
    """A scan cell hit the deep-tangency signature or a probe floor.

    Section 12.1 rejects the entire certificate rather than merging or
    discarding a root: a cell whose exact width reaches ``2**-44`` turn with
    neither classification, a retained root whose numerator residual exceeds
    ``5e-12``, and a probe magnitude at or below ``1e-10`` all reject as
    ``mmode_horizon_unresolved``, so a near-tangent transit is rejected rather
    than misclassified.
    """


@dataclass(frozen=True, slots=True)
class OperationalHorizonScan:
    """The certified-ceiling census, retained in Section 12.1's economy form.

    The full terminal-cell array is of order ten million rows for the bounded
    driver, so Section 12.1 retains a **bounded projection** of it rather than
    the array itself: every ``scan_crossing`` and ``excluded_upper_endpoint``
    row verbatim, plus one per-direction summary row.  The full array's digest
    is computed *streamingly* during the census -- the array is serialized row
    by row into the hash and never materialized -- and the array itself is
    reconstructible by deterministic replay, which the ``A1`` re-derivation
    performs.
    """

    crossing_rows: tuple[Mapping[str, Any], ...]
    summary_rows: tuple[Mapping[str, Any], ...]
    ledger_sha256: str
    roots: tuple[tuple[HorizonRootEnclosure, ...], ...]
    #: ``f_o`` at every retained sample centre, shaped ``(sample, direction)``.
    #: Section 4.2 requires the certificate's *operational* membership to come
    #: from the same public-API evaluations this scan consumes, and every
    #: centre is already one of its shared cell boundaries, so the values are
    #: carried out rather than recomputed through a second path.
    centre_values: np.ndarray
    #: The live public-transform evaluator, so the outside-slab sign census
    #: takes its operational values from the same source.
    evaluator: Any
    guard_count: int
    evaluation_count: int
    #: The total number of terminal rows across all directions.  It equals the
    #: summary rows' ``terminal_cell_count`` sum by construction.
    isolation_interval_count: int
    astropy_version: str
    erfa_version: str
    iers_table_sha256: str

    def manifest_sha256(self) -> str:
        """Return SHA-256 of the scan manifest's exact canonical bytes."""
        return hashlib.sha256(canonical_json(self.manifest())).hexdigest()

    def manifest(self) -> dict[str, Any]:
        """Return Section 12.1's ``horizon_scan_manifest`` object, in key order."""
        implementation = [
            {
                "path": path,
                "sha256": hashlib.sha256(
                    (Path(__file__).resolve().parents[4] / path).read_bytes()
                ).hexdigest(),
            }
            for path in sorted(
                (
                    "src/radiosim/core/mmode/frame.py",
                    "src/radiosim/core/mmode/time.py",
                )
            )
        ]
        return {
            "schema_version": SCAN_SCHEMA,
            "algorithm_id": SCAN_SCHEMA,
            "implementation_files": implementation,
            "constant_rows": scan_constant_rows(),
            "astropy_version": self.astropy_version,
            "erfa_version": self.erfa_version,
            "iers_table_sha256": self.iers_table_sha256,
        }


#: Section 12.1's exact terminal-row field order.  This is the sole
#: discriminated-format exception: the scan row is serialized in *this* order,
#: not with Section 14's lexicographic object-key sorting.
SCAN_ROW_FIELDS: tuple[str, ...] = (
    "direction_id",
    "cell_index",
    "turn_lo",
    "turn_hi",
    "classification",
    "f_lo_f64be",
    "f_hi_f64be",
    "ceiling_margin_f64be",
    "left_sign",
    "right_sign",
    "root_turn_lo",
    "root_turn_hi",
    "root_orientation",
    "root_residual_f64be",
)

#: Section 12.1's exact per-direction summary-row field order.
SCAN_SUMMARY_FIELDS: tuple[str, ...] = (
    "direction_id",
    "terminal_cell_count",
    "boundary_evaluation_count",
    "crossing_count",
    "min_ceiling_margin_f64be",
)


def scan_row_bytes(row: Mapping[str, Any]) -> bytes:
    """Serialize one terminal scan row in Section 12.1's exact field order.

    ``ensure_ascii=true``, separators ``(',',':')``, no whitespace, no trailing
    newline.  The three root fields are JSON null exactly for
    ``ceiling_excludes_root``; every ``*_f64be`` value is an exact string
    rather than a JSON number.
    """
    parts = []
    for field in SCAN_ROW_FIELDS:
        value = row[field]
        if value is None:
            rendered = "null"
        elif isinstance(value, bool):  # pragma: no cover - defensive
            rendered = "true" if value else "false"
        elif isinstance(value, int):
            rendered = str(int(value))
        else:
            rendered = json.dumps(str(value), ensure_ascii=True)
        parts.append(json.dumps(field, ensure_ascii=True) + ":" + rendered)
    return ("{" + ",".join(parts) + "}").encode("utf-8")


def scan_constant_rows() -> list[dict[str, str]]:
    """Return every scan constant as a name-sorted Section 12.1 row.

    ``constant_rows`` contains every ceiling, spacing, refinement, root-width,
    probe-offset, unresolved-width and residual constant the implementation
    consumes; each row has exactly ``name``, ``type`` and ``value``.
    """
    rows = [
        ("L_op", "binary64", f64be(SCAN_DERIVATIVE_CEILING_PER_TURN)),
        ("h_0", "rational", canonical_rational(SCAN_INITIAL_SPACING_TURN)),
        ("probe_magnitude_floor", "binary64", f64be(SCAN_PROBE_MAGNITUDE_FLOOR)),
        ("probe_offset_turn", "rational", canonical_rational(SCAN_PROBE_OFFSET_TURN)),
        ("root_enclosure_width_rad", "binary64", f64be(SCAN_ROOT_WIDTH_RAD)),
        ("root_residual_bound", "binary64", f64be(SCAN_ROOT_RESIDUAL)),
        ("scan_algorithm", "literal", SCAN_SCHEMA),
        (
            "unresolved_width_turn",
            "rational",
            canonical_rational(SCAN_UNRESOLVED_WIDTH_TURN),
        ),
    ]
    return [
        {"name": name, "type": kind, "value": value}
        for name, kind, value in sorted(rows)
    ]


class _OperationalTrajectory:
    """Batched public-Astropy evaluation of ``f_o(u) = sin(alt_operational(u))``.

    Section 12.1 consumes only public
    ``SkyCoord.transform_to(AltAz(obstime=..., location=..., pressure=0))``
    values; a private frame surrogate is never an authority.  Every distinct
    boundary value is computed exactly once and reused, and the evaluation set
    is deterministic.
    """

    def __init__(self, frame: FrozenFrame, grid: Any, ra_rad: Any, dec_rad: Any):
        from astropy import units as u
        from astropy.coordinates import EarthLocation, SkyCoord

        self._frame = frame
        self._grid = grid
        self._units = u
        self._site = EarthLocation.from_geodetic(
            lon=frame.longitude_deg * u.deg,
            lat=frame.latitude_deg * u.deg,
            height=frame.height_m * u.m,
        )
        self._coords = SkyCoord(
            ra=np.asarray(ra_rad, dtype=np.float64) * u.rad,
            dec=np.asarray(dec_rad, dtype=np.float64) * u.rad,
            frame="icrs",
        )
        self.direction_count = int(self._coords.size)
        self.evaluations = 0
        #: Section 12.1's per-direction boundary tally.  Every distinct turn at
        #: which ``f_o`` is evaluated for a direction is counted exactly once,
        #: which is what the summary row's ``boundary_evaluation_count``
        #: reports.
        self.per_direction = [0] * self.direction_count

    def _times(self, turns: Any) -> Any:
        from astropy.time import Time

        from radiosim.core.mmode.time import ERA_RATE_TURNS_PER_UT1_DAY

        jd1, jd2 = self._frame.ut1_two_part
        exact = Fraction(jd2)
        second = np.asarray(
            [
                float(exact + Fraction(turn) / ERA_RATE_TURNS_PER_UT1_DAY)
                for turn in turns
            ],
            dtype=np.float64,
        )
        first = np.full(second.shape, float(jd1), dtype=np.float64)
        return Time(first, second, format="jd", scale="ut1")

    def at_common_turn(self, turn: Fraction) -> np.ndarray:
        """Return ``f_o`` for every direction at one shared turn."""
        from astropy.coordinates import AltAz

        times = self._times([turn])
        frame = AltAz(obstime=times[0], location=self._site, pressure=0)
        altitude = self._coords.transform_to(frame).alt.to_value(self._units.rad)
        self.evaluations += self.direction_count
        for index in range(self.direction_count):
            self.per_direction[index] += 1
        return np.sin(np.atleast_1d(np.asarray(altitude, dtype=np.float64)))

    def at_pairs(self, indices: Any, turns: Any) -> np.ndarray:
        """Return ``f_o`` for an element-wise ``(direction, turn)`` batch."""
        from astropy.coordinates import AltAz

        index_array = np.asarray(indices, dtype=np.int64)
        if index_array.size == 0:
            return np.zeros(0, dtype=np.float64)
        times = self._times(turns)
        frame = AltAz(obstime=times, location=self._site, pressure=0)
        altitude = (
            self._coords[index_array].transform_to(frame).alt.to_value(self._units.rad)
        )
        self.evaluations += int(index_array.size)
        for index in index_array.tolist():
            self.per_direction[int(index)] += 1
        return np.sin(np.atleast_1d(np.asarray(altitude, dtype=np.float64)))


def _sign(value: float) -> int:
    if value > 0.0:
        return 1
    if value < 0.0:
        return -1
    return 0


def scan_operational_horizon(
    *,
    frame: FrozenFrame,
    grid: Any,
    ra_rad: Any,
    dec_rad: Any,
    frozen_root_bounds: Sequence[Sequence[Fraction]],
    direction_ids: Sequence[str],
) -> OperationalHorizonScan:
    """Run Section 12.1's deterministic certified-ceiling scan.

    The initial partition of ``[h_N^-, h_N^+]`` is the uniform exact-turn grid
    of spacing ``h_0 = 2**-12`` turn, refined so that every retained centre and
    edge turn from the same Section 3.1 grid object and every frozen root bound
    is also a cell boundary.  ``f_o`` is evaluated exactly once at every
    distinct cell boundary; evaluations are batched, each boundary value is
    computed once and reused, and the evaluation set is deterministic.

    A cell ``[a, b]`` of exact width ``h`` is proven root-free when
    ``min(|f_o(a)|, |f_o(b)|) > L_op * h``: any zero inside would force both
    endpoint magnitudes to at most ``L_op * h`` by the ceiling.  A cell with a
    sign change contains a crossing and is bisected at its exactly
    representable midpoint to the fixed ``1e-11 rad`` enclosure width.  Any
    other cell is bisected and both children re-enter the queue; a cell whose
    exact width reaches ``2**-44`` turn with neither classification rejects the
    entire certificate.

    Section 12.1's economy projection is produced here rather than downstream:
    the complete terminal array is serialized one direction at a time straight
    into a streaming SHA-256 and never materialized, while the crossing rows and
    the per-direction summaries -- the bounded projection the evidence embeds --
    are retained.

    Parameters
    ----------
    direction_ids : sequence of str
        The canonical direction-ledger identifiers, in ledger order.  They label
        the emitted rows, so the scan's own output joins the ledger directly.

    Returns
    -------
    OperationalHorizonScan
        The retained economy projection, the streamed full-array digest, the
        certified root enclosures per direction, the deterministic evaluation
        count, and the environment identities the scan manifest binds.
    """
    import erfa
    from astropy import __version__ as astropy_version

    from radiosim.core.mmode.time import installed_iers, installed_iers_context

    if not isinstance(frozen_root_bounds, (list, tuple)):
        raise ValueError("frozen root bounds must be a list or tuple")
    for bounds in frozen_root_bounds:
        if not isinstance(bounds, (list, tuple)):
            raise ValueError("each frozen root-bound entry must be a list or tuple")
        for bound in bounds:
            if type(bound) is not Fraction:
                raise ValueError("frozen root-bound endpoints must be exact Fractions")
    horizon_lo, horizon_hi = grid.horizon_domain
    trajectory = _OperationalTrajectory(frame, grid, ra_rad, dec_rad)
    direction_count = trajectory.direction_count
    if len(frozen_root_bounds) != direction_count:
        raise ValueError("one frozen root-bound list is required per direction")
    if len(direction_ids) != direction_count:
        raise ValueError("one direction identifier is required per direction")

    steps = int((horizon_hi - horizon_lo) / SCAN_INITIAL_SPACING_TURN)
    base: set[Fraction] = {
        horizon_lo + index * SCAN_INITIAL_SPACING_TURN for index in range(steps + 1)
    }
    base.add(horizon_hi)
    for index in range(grid.sidereal_samples):
        base.add(grid.center_turn(index))
        lower, upper = grid.exposure_turns(index)
        base.add(lower)
        base.add(upper)
    for bounds in frozen_root_bounds:
        base.update(bounds)
    shared = sorted(bound for bound in base if horizon_lo <= bound <= horizon_hi)

    with installed_iers_context():
        matrix = np.empty((len(shared), direction_count), dtype=np.float64)
        for position, bound in enumerate(shared):
            matrix[position] = trajectory.at_common_turn(bound)

        widths = np.asarray(
            [float(shared[i + 1] - shared[i]) for i in range(len(shared) - 1)],
            dtype=np.float64,
        )
        low_values = matrix[:-1]
        high_values = matrix[1:]
        margins = np.minimum(np.abs(low_values), np.abs(high_values)) - (
            SCAN_DERIVATIVE_CEILING_PER_TURN * widths[:, None]
        )
        root_free = margins > 0.0
        crossing = (low_values * high_values) < 0.0
        ambiguous = ~(root_free | crossing)

        # A shared-grid cell the vectorized pass did not prove root-free is
        # replaced in the terminal partition by whatever the queue produces
        # below, so its own row is suppressed.
        subdivided = ~root_free
        bisected: list[tuple[int, Fraction, Fraction, float, float, float]] = []

        pending: list[tuple[int, Fraction, Fraction, float, float]] = []
        crossings: list[tuple[int, Fraction, Fraction, float, float]] = []
        for cell_index, direction in zip(*np.nonzero(crossing), strict=True):
            crossings.append(
                (
                    int(direction),
                    shared[int(cell_index)],
                    shared[int(cell_index) + 1],
                    float(low_values[cell_index, direction]),
                    float(high_values[cell_index, direction]),
                )
            )
        for cell_index, direction in zip(*np.nonzero(ambiguous), strict=True):
            pending.append(
                (
                    int(direction),
                    shared[int(cell_index)],
                    shared[int(cell_index) + 1],
                    float(low_values[cell_index, direction]),
                    float(high_values[cell_index, direction]),
                )
            )

        while pending:
            classified: list[tuple[int, Fraction, Fraction, float, float]] = []
            for item in pending:
                direction, low, high, f_low, f_high = item
                width = high - low
                margin = min(abs(f_low), abs(f_high)) - (
                    SCAN_DERIVATIVE_CEILING_PER_TURN * float(width)
                )
                if margin > 0.0:
                    bisected.append((direction, low, high, f_low, f_high, margin))
                    continue
                if f_low * f_high < 0.0:
                    crossings.append(item)
                    continue
                if width <= SCAN_UNRESOLVED_WIDTH_TURN:
                    raise OperationalScanRejected(
                        "a scan cell reached the deep-tangency width with neither "
                        "the ceiling exclusion nor a certified crossing"
                    )
                classified.append(item)
            if not classified:
                break
            midpoints = [(low + high) / 2 for _, low, high, _, _ in classified]
            observed = trajectory.at_pairs([item[0] for item in classified], midpoints)
            pending = []
            for position, item in enumerate(classified):
                direction, low, high, f_low, f_high = item
                middle = midpoints[position]
                f_middle = float(observed[position])
                pending.append((direction, low, middle, f_low, f_middle))
                pending.append((direction, middle, high, f_middle, f_high))

        target = Fraction(SCAN_ROOT_WIDTH_RAD) / Fraction(*TAU.as_integer_ratio())
        # Section 12.1: the same-sign sibling produced beside a crossing at each
        # bisection step is *not* discarded.  It re-enters the bullet-4 queue
        # keyed to the crossing it flanks, so the terminal rows tile the
        # crossing's parent cell together with the retained enclosure, and the
        # residual the ceiling rule cannot classify becomes a guard rather than
        # a hole.
        enclosure: dict[int, tuple[int, Fraction, Fraction, float, float]] = {}
        siblings: list[tuple[int, int, Fraction, Fraction, float, float]] = []
        active = list(enumerate(crossings))
        while active:
            wide: list[tuple[int, tuple[int, Fraction, Fraction, float, float]]] = []
            for owner, item in active:
                if (item[2] - item[1]) <= target:
                    enclosure[owner] = item
                else:
                    wide.append((owner, item))
            if not wide:
                break
            midpoints = [(item[1] + item[2]) / 2 for _owner, item in wide]
            observed = trajectory.at_pairs(
                [item[0] for _owner, item in wide], midpoints
            )
            active = []
            for position, (owner, item) in enumerate(wide):
                direction, low, high, f_low, f_high = item
                middle = midpoints[position]
                f_middle = float(observed[position])
                if f_low * f_middle <= 0.0:
                    active.append((owner, (direction, low, middle, f_low, f_middle)))
                    siblings.append((owner, direction, middle, high, f_middle, f_high))
                else:
                    active.append((owner, (direction, middle, high, f_middle, f_high)))
                    siblings.append((owner, direction, low, middle, f_low, f_middle))
        refined = [enclosure[owner] for owner in sorted(enclosure)]

        # Refine every retained sibling under the same bullet-4 rule.  A cell
        # classifies root-free once its width falls below ``beta`` times its
        # distance to the root, which for an ordinary crossing happens after
        # roughly ``log2(1/beta)`` extra refinements; the contiguous residue
        # that reaches the ``2**-44`` turn floor still touching its crossing's
        # enclosure becomes that flank's single guard interval.
        classified: list[tuple[int, Fraction, Fraction, float, float, float]] = []
        unresolved: dict[
            tuple[int, str], list[tuple[Fraction, Fraction, float, float]]
        ] = {}
        pending_guard: list[tuple[int, int, Fraction, Fraction, float, float]] = [
            entry for entry in siblings if entry[3] > entry[2]
        ]
        while pending_guard:
            split: list[tuple[int, int, Fraction, Fraction, float, float]] = []
            for owner, direction, low, high, f_low, f_high in pending_guard:
                width = high - low
                margin = min(abs(f_low), abs(f_high)) - (
                    SCAN_DERIVATIVE_CEILING_PER_TURN * float(width)
                )
                if margin > 0.0:
                    classified.append((direction, low, high, f_low, f_high, margin))
                    continue
                if width <= SCAN_UNRESOLVED_WIDTH_TURN:
                    side = "left" if high <= enclosure[owner][1] else "right"
                    unresolved.setdefault((owner, side), []).append(
                        (low, high, f_low, f_high)
                    )
                    continue
                split.append((owner, direction, low, high, f_low, f_high))
            if not split:
                break
            midpoints = [(low + high) / 2 for _o, _d, low, high, _fl, _fh in split]
            observed = trajectory.at_pairs([entry[1] for entry in split], midpoints)
            pending_guard = []
            for position, entry in enumerate(split):
                owner, direction, low, high, f_low, f_high = entry
                middle = midpoints[position]
                f_middle = float(observed[position])
                pending_guard.append((owner, direction, low, middle, f_low, f_middle))
                pending_guard.append((owner, direction, middle, high, f_middle, f_high))

        # Each flank's unresolved residue must be one contiguous block touching
        # the enclosure; an orphan block is a genuine deep tangency and rejects.
        guards: dict[tuple[int, str], tuple[Fraction, Fraction, float, float]] = {}
        for (owner, side), pieces in unresolved.items():
            pieces.sort(key=lambda piece: piece[0])
            lower, upper = pieces[0][0], pieces[-1][1]
            merged = lower
            for piece_lo, piece_hi, _f_lo, _f_hi in pieces:
                if piece_lo != merged:
                    raise OperationalScanRejected(
                        "an unresolved scan residue beside a certified crossing is "
                        "not contiguous; the deep-tangency signature is not a guard"
                    )
                merged = piece_hi
            root_lo, root_hi = enclosure[owner][1], enclosure[owner][2]
            touches = upper == root_lo if side == "left" else lower == root_hi
            if not touches:
                raise OperationalScanRejected(
                    "an unresolved scan cell reached the deep-tangency width "
                    "without touching a certified root enclosure"
                )
            if (upper - lower) > SCAN_PROBE_OFFSET_TURN:
                raise OperationalScanRejected(
                    "a guard interval exceeded the probe offset that bounds it"
                )
            guards[(owner, side)] = (lower, upper, pieces[0][2], pieces[-1][3])

        # Probe signs fix each retained root's census orientation.
        probes: list[tuple[int, Fraction]] = []
        for direction, low, high, _, _ in refined:
            probes.append((direction, low - SCAN_PROBE_OFFSET_TURN))
            probes.append((direction, high + SCAN_PROBE_OFFSET_TURN))
        probe_values = (
            trajectory.at_pairs(
                [item[0] for item in probes], [item[1] for item in probes]
            )
            if probes
            else np.zeros(0, dtype=np.float64)
        )
        midpoints = [(low + high) / 2 for _, low, high, _, _ in refined]
        residuals = (
            np.abs(trajectory.at_pairs([item[0] for item in refined], midpoints))
            if refined
            else np.zeros(0, dtype=np.float64)
        )

        # Every retained root, plus the terminal rows the refinement produced.
        roots: list[list[HorizonRootEnclosure]] = [[] for _ in range(direction_count)]
        extra: list[list[dict[str, Any]]] = [[] for _ in range(direction_count)]
        owners = sorted(enclosure)
        guard_total = 0
        for position, (direction, low, high, f_low, f_high) in enumerate(refined):
            owner = owners[position]
            before = float(probe_values[2 * position])
            after = float(probe_values[2 * position + 1])
            if (
                abs(before) <= SCAN_PROBE_MAGNITUDE_FLOOR
                or abs(after) <= SCAN_PROBE_MAGNITUDE_FLOOR
                or _sign(before) == _sign(after)
            ):
                raise OperationalScanRejected(
                    "a retained operational root failed its probe-sign certificate; "
                    "a near-tangent transit is rejected rather than misclassified"
                )
            residual = float(residuals[position])
            if residual > SCAN_ROOT_RESIDUAL:
                raise OperationalScanRejected(
                    "a retained operational root exceeded its numerator residual bound"
                )
            orientation: Literal["rising", "setting"] = (
                "rising" if before < 0.0 else "setting"
            )
            classification = (
                "excluded_upper_endpoint" if high == horizon_hi else "scan_crossing"
            )
            extra[direction].append(
                {
                    "turn_lo": low,
                    "turn_hi": high,
                    "classification": classification,
                    "f_lo_f64be": f64be(f_low),
                    "f_hi_f64be": f64be(f_high),
                    "ceiling_margin_f64be": f64be(0.0),
                    "left_sign": _sign(before),
                    "right_sign": _sign(after),
                    "root_turn_lo": canonical_rational(low),
                    "root_turn_hi": canonical_rational(high),
                    "root_orientation": orientation,
                    "root_residual_f64be": f64be(residual),
                }
            )
            left = guards.get((owner, "left"))
            right = guards.get((owner, "right"))
            for side, guard in (("left", left), ("right", right)):
                if guard is None:
                    continue
                guard_lo, guard_hi, guard_f_lo, guard_f_hi = guard
                guard_total += 1
                extra[direction].append(
                    {
                        "turn_lo": guard_lo,
                        "turn_hi": guard_hi,
                        "classification": "guard_interval",
                        "f_lo_f64be": f64be(guard_f_lo),
                        "f_hi_f64be": f64be(guard_f_hi),
                        "ceiling_margin_f64be": f64be(0.0),
                        "left_sign": _sign(guard_f_lo),
                        "right_sign": _sign(guard_f_hi),
                        "root_turn_lo": None,
                        "root_turn_hi": None,
                        "root_orientation": None,
                        "root_residual_f64be": None,
                    }
                )
                del side
            if classification == "scan_crossing":
                roots[direction].append(
                    HorizonRootEnclosure(
                        turn_lo=low,
                        turn_hi=high,
                        orientation=orientation,
                        residual=residual,
                        guard_turn_lo=None if left is None else left[0],
                        guard_turn_hi=None if right is None else right[1],
                    )
                )
        for direction in range(direction_count):
            roots[direction].sort(key=lambda entry: entry.turn_lo)

        # The refined siblings are ordinary root-free terminal rows.
        for direction, low, high, f_low, f_high, margin in classified:
            extra[direction].append(
                {
                    "turn_lo": low,
                    "turn_hi": high,
                    "classification": "ceiling_excludes_root",
                    "f_lo_f64be": f64be(f_low),
                    "f_hi_f64be": f64be(f_high),
                    "ceiling_margin_f64be": f64be(margin),
                    "left_sign": _sign(f_low),
                    "right_sign": _sign(f_high),
                    "root_turn_lo": None,
                    "root_turn_hi": None,
                    "root_orientation": None,
                    "root_residual_f64be": None,
                }
            )

        # The subdivided cells are exactly those the vectorized pass did not
        # prove root-free; their shared-grid rows are replaced by the terminal
        # rows the queue produced.
        for item in bisected:
            direction, low, high, f_low, f_high, margin = item
            extra[direction].append(
                {
                    "turn_lo": low,
                    "turn_hi": high,
                    "classification": "ceiling_excludes_root",
                    "f_lo_f64be": f64be(f_low),
                    "f_hi_f64be": f64be(f_high),
                    "ceiling_margin_f64be": f64be(margin),
                    "left_sign": _sign(f_low),
                    "right_sign": _sign(f_high),
                    "root_turn_lo": None,
                    "root_turn_hi": None,
                    "root_orientation": None,
                    "root_residual_f64be": None,
                }
            )

        # Section 12.1's economy projection.  The complete array is serialized
        # one direction at a time straight into the hash and never retained;
        # only the crossing rows and the per-direction summaries survive.
        #
        # The complete array is of order sixteen million rows for the bounded
        # driver, so the serialization is written for that scale: every value
        # that repeats across directions -- the shared grid's exact rational
        # bounds, the field-name prefixes, the contiguous cell indices -- is
        # rendered once, and each direction's three binary64 columns are
        # converted in one big-endian buffer rather than value by value.  The
        # bytes are identical to :func:`scan_row_bytes` row for row.
        digest = hashlib.sha256()
        digest.update(b"[")
        crossing_rows: list[Mapping[str, Any]] = []
        summary_rows: list[Mapping[str, Any]] = []
        emitted = 0
        shared_turn_json = [
            json.dumps(canonical_rational(bound), ensure_ascii=True) for bound in shared
        ]
        widest = len(shared) + max((len(rows) for rows in extra), default=0) + 1
        index_json = [str(index) for index in range(widest)]
        names = [json.dumps(field, ensure_ascii=True) for field in SCAN_ROW_FIELDS]
        excluded_tail = (
            f',{names[4]}:"ceiling_excludes_root"'
            f',{names[5]}:"%s",{names[6]}:"%s",{names[7]}:"%s"'
            f",{names[8]}:%s,{names[9]}:%s"
            f",{names[10]}:null,{names[11]}:null"
            f",{names[12]}:null,{names[13]}:null}}"
        )
        # Every extra row belongs to the shared cell its lower bound falls in,
        # which is what restores the ascending ``turn_lo`` order without a
        # global sort over exact rationals.
        parents: list[list[dict[str, Any]]] = [[] for _ in range(direction_count)]
        for direction in range(direction_count):
            owned = sorted(extra[direction], key=lambda entry: entry["turn_lo"])
            parents[direction] = owned
        boundary_evaluations = list(trajectory.per_direction)

        for direction in range(direction_count):
            identifier = json.dumps(direction_ids[direction], ensure_ascii=True)
            prefix = "{" + names[0] + ":" + identifier + "," + names[1] + ":"
            keep = ~subdivided[:, direction]
            low_hex = (
                np.ascontiguousarray(low_values[:, direction], dtype=">f8")
                .tobytes()
                .hex()
            )
            high_hex = (
                np.ascontiguousarray(high_values[:, direction], dtype=">f8")
                .tobytes()
                .hex()
            )
            margin_hex = (
                np.ascontiguousarray(margins[:, direction], dtype=">f8").tobytes().hex()
            )
            low_signs = np.sign(low_values[:, direction]).astype(np.int64)
            high_signs = np.sign(high_values[:, direction]).astype(np.int64)

            owned = parents[direction]
            by_parent: dict[int, list[dict[str, Any]]] = {}
            for row in owned:
                position = bisect.bisect_right(shared, row["turn_lo"]) - 1
                by_parent.setdefault(max(position, 0), []).append(row)

            chunks: list[bytes] = []
            crossing_count = 0
            cell_index = 0
            smallest = math.inf
            for cell in range(len(shared) - 1):
                if keep[cell]:
                    margin = float(margins[cell, direction])
                    if margin < smallest:
                        smallest = margin
                    chunks.append(
                        (
                            prefix
                            + index_json[cell_index]
                            + ","
                            + names[2]
                            + ":"
                            + shared_turn_json[cell]
                            + ","
                            + names[3]
                            + ":"
                            + shared_turn_json[cell + 1]
                            + excluded_tail
                            % (
                                low_hex[cell * 16 : cell * 16 + 16],
                                high_hex[cell * 16 : cell * 16 + 16],
                                margin_hex[cell * 16 : cell * 16 + 16],
                                low_signs[cell],
                                high_signs[cell],
                            )
                        ).encode("utf-8")
                    )
                    cell_index += 1
                    continue
                for row in by_parent.get(cell, ()):
                    canonical = {
                        "direction_id": direction_ids[direction],
                        "cell_index": cell_index,
                        "turn_lo": canonical_rational(row["turn_lo"]),
                        "turn_hi": canonical_rational(row["turn_hi"]),
                        **{
                            field: row[field]
                            for field in SCAN_ROW_FIELDS
                            if field
                            not in ("direction_id", "cell_index", "turn_lo", "turn_hi")
                        },
                    }
                    chunks.append(scan_row_bytes(canonical))
                    cell_index += 1
                    kind = canonical["classification"]
                    if kind == "ceiling_excludes_root":
                        margin = decode_f64be(canonical["ceiling_margin_f64be"])
                        if margin < smallest:
                            smallest = margin
                    elif kind == "guard_interval":
                        # A guard is retained verbatim beside the crossing it
                        # flanks -- the strict validator's adjacency and
                        # partition rules need the row itself -- but it is not
                        # a crossing and never enters the root census.
                        crossing_rows.append(canonical)
                    else:
                        crossing_count += 1
                        crossing_rows.append(canonical)
            if chunks:
                if emitted:
                    digest.update(b",")
                digest.update(b",".join(chunks))
                emitted += len(chunks)
            summary_rows.append(
                {
                    "direction_id": direction_ids[direction],
                    "terminal_cell_count": cell_index,
                    "boundary_evaluation_count": boundary_evaluations[direction],
                    "crossing_count": crossing_count,
                    "min_ceiling_margin_f64be": f64be(
                        smallest if math.isfinite(smallest) else 0.0
                    ),
                }
            )
        digest.update(b"]")

        # Section 4.2's independent membership: every retained sample centre is
        # one of the shared cell boundaries above, so the operational values the
        # certificate compares are exactly the public-API values this scan
        # already consumed, taken by index rather than recomputed.
        centre_positions = [
            bisect.bisect_left(shared, grid.center_turn(index))
            for index in range(grid.sidereal_samples)
        ]
        for index, position in enumerate(centre_positions):
            if position >= len(shared) or shared[position] != grid.center_turn(index):
                raise OperationalScanRejected(
                    "a retained sample centre is not a scanned cell boundary"
                )
        centre_values = matrix[centre_positions, :]

        installed = installed_iers()
        return OperationalHorizonScan(
            crossing_rows=tuple(crossing_rows),
            summary_rows=tuple(summary_rows),
            ledger_sha256=digest.hexdigest(),
            roots=tuple(tuple(entries) for entries in roots),
            centre_values=centre_values,
            evaluator=trajectory,
            guard_count=guard_total,
            evaluation_count=trajectory.evaluations,
            isolation_interval_count=emitted,
            astropy_version=str(astropy_version),
            erfa_version=str(erfa.__version__),
            iers_table_sha256=installed.table_sha256,
        )


# ---------------------------------------------------------------------------
# Section 12.1 frozen analytic horizon oracle
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class HorizonRootEnclosure:
    """One certified closed exact-turn root enclosure and its orientation.

    ``guard_turn_lo`` and ``guard_turn_hi`` are the outer bounds of the Section
    12.1 flanking guard intervals, which default to the enclosure itself when
    the refinement classified every neighbour.  Section 12's exposure machinery
    cuts at and error-disks over the *guarded* span, so any structure the
    ceiling rule could not exclude beside a shallow crossing is certified
    bounded rather than assumed absent.  The frozen analytic census owns exact
    roots and never carries guards.
    """

    turn_lo: Fraction
    turn_hi: Fraction
    orientation: Literal["rising", "setting"]
    residual: float
    guard_turn_lo: Fraction | None = None
    guard_turn_hi: Fraction | None = None

    @property
    def ambiguous_span(self) -> tuple[Fraction, Fraction]:
        """Return the closed enclosure-plus-guards union of this crossing."""
        lower = self.turn_lo if self.guard_turn_lo is None else self.guard_turn_lo
        upper = self.turn_hi if self.guard_turn_hi is None else self.guard_turn_hi
        return (lower, upper)

    def as_row(self) -> dict[str, Any]:
        """Return the canonical serialization of this enclosure."""
        return {
            "root_turn_lo": canonical_rational(self.turn_lo),
            "root_turn_hi": canonical_rational(self.turn_hi),
            "root_orientation": self.orientation,
            "astropy_residual_f64be": f64be(self.residual),
        }


@dataclass(frozen=True, slots=True)
class FrozenHorizonTrajectory:
    """The exact ``A cos + B sin + C`` trajectory of one frozen direction."""

    a_coefficient: float
    b_coefficient: float
    c_coefficient: float
    topology: str
    roots: tuple[HorizonRootEnclosure, ...]

    def value(self, turn: float) -> float:
        """Return ``sin(alt(u))`` at one binary64 turn."""
        angle = 2.0 * math.pi * float(turn)
        return (
            self.a_coefficient * math.cos(angle)
            + self.b_coefficient * math.sin(angle)
            + self.c_coefficient
        )

    def derivative(self, turn: float) -> float:
        """Return ``d sin(alt(u)) / du`` at one binary64 turn."""
        angle = 2.0 * math.pi * float(turn)
        return (
            2.0
            * math.pi
            * (
                -self.a_coefficient * math.sin(angle)
                + self.b_coefficient * math.cos(angle)
            )
        )

    def value_interval(self, lower: Fraction, upper: Fraction) -> tuple[float, float]:
        """Return a certified outward enclosure of ``sin(alt)`` on a turn cell."""
        (cos_lo, cos_hi), (sin_lo, sin_hi) = certified_two_pi_trig(lower, upper)
        return _linear_interval(
            self.a_coefficient,
            (cos_lo, cos_hi),
            self.b_coefficient,
            (sin_lo, sin_hi),
            self.c_coefficient,
        )

    def derivative_interval(
        self, lower: Fraction, upper: Fraction
    ) -> tuple[float, float]:
        """Return a certified outward enclosure of the derivative on a cell."""
        (cos_lo, cos_hi), (sin_lo, sin_hi) = certified_two_pi_trig(lower, upper)
        inner = _linear_interval(
            -self.a_coefficient,
            (sin_lo, sin_hi),
            self.b_coefficient,
            (cos_lo, cos_hi),
            0.0,
        )
        two_pi_lo, two_pi_hi = certified_two_pi_bracket()
        products = [
            Fraction(inner[0]) * two_pi_lo,
            Fraction(inner[0]) * two_pi_hi,
            Fraction(inner[1]) * two_pi_lo,
            Fraction(inner[1]) * two_pi_hi,
        ]
        return (round_down(min(products)), round_up(max(products)))


def _linear_interval(
    first: float,
    first_range: tuple[float, float],
    second: float,
    second_range: tuple[float, float],
    offset: float,
) -> tuple[float, float]:
    """Return the outward enclosure of ``a*X + b*Y + c`` over two intervals."""
    exact_first = Fraction(first)
    exact_second = Fraction(second)
    candidates_first = [
        exact_first * Fraction(first_range[0]),
        exact_first * Fraction(first_range[1]),
    ]
    candidates_second = [
        exact_second * Fraction(second_range[0]),
        exact_second * Fraction(second_range[1]),
    ]
    low = min(candidates_first) + min(candidates_second) + Fraction(offset)
    high = max(candidates_first) + max(candidates_second) + Fraction(offset)
    return (round_down(low), round_up(high))


def frozen_horizon_coefficients(
    frame: FrozenFrame, cirs_direction: np.ndarray
) -> tuple[float, float, float]:
    r"""Return ``(A, B, C)`` of ``sin(alt(u)) = A cos(2 pi u) + B sin(2 pi u) + C``.

    With ``t(alpha) = T(0) R3(alpha) c`` and ``g = T(0)^T e_U``, expanding
    ``R3(alpha) c`` gives ``A = g_1 c_1 + g_2 c_2``, ``B = g_1 c_2 - g_2 c_1``
    and ``C = g_3 c_3`` exactly.  Nothing is fitted and no phase offset is
    introduced.
    """
    gravity = np.asarray(frame.cirs_to_itrs_anchor, dtype=np.float64).T @ np.asarray(
        frame.local_up_itrs, dtype=np.float64
    )
    direction = np.asarray(cirs_direction, dtype=np.float64)
    a_coefficient = float(gravity[0] * direction[0] + gravity[1] * direction[1])
    b_coefficient = float(gravity[0] * direction[1] - gravity[1] * direction[0])
    c_coefficient = float(gravity[2] * direction[2])
    return (a_coefficient, b_coefficient, c_coefficient)


def frozen_horizon_trajectory(
    frame: FrozenFrame,
    cirs_direction: np.ndarray,
    *,
    horizon_lo: Fraction,
    horizon_hi: Fraction,
) -> FrozenHorizonTrajectory:
    """Enumerate every frozen analytic horizon root over the exact cycle.

    Section 12.1's exhaustive case analysis is taken from the **exact integer
    ratios** of ``A``, ``B`` and ``C``:

    1. ``A == B == 0`` and ``C != 0``: constant sign, no root;
    2. ``A == B == C == 0``: identically on the horizon -- a typed rejection;
    3. ``rho2 > 0`` and ``C**2 > rho2``: circumpolar, no root;
    4. ``rho2 > 0`` and ``C**2 == rho2``: the stationary/tangent equality -- a
       typed rejection; and
    5. ``rho2 > 0`` and ``C**2 < rho2``: the two and only two transverse roots.

    The two-root formulas are used only *after* that exact decision.
    """
    a_coefficient, b_coefficient, c_coefficient = frozen_horizon_coefficients(
        frame, cirs_direction
    )
    exact_a = Fraction(a_coefficient)
    exact_b = Fraction(b_coefficient)
    exact_c = Fraction(c_coefficient)
    rho2 = exact_a * exact_a + exact_b * exact_b
    c_squared = exact_c * exact_c

    if rho2 == 0:
        if c_squared == 0:
            raise MModeHorizonUnresolved(
                "the frozen trajectory lies identically on the horizon"
            )
        return FrozenHorizonTrajectory(
            a_coefficient,
            b_coefficient,
            c_coefficient,
            "constant",
            (),
        )
    if c_squared > rho2:
        return FrozenHorizonTrajectory(
            a_coefficient,
            b_coefficient,
            c_coefficient,
            "circumpolar",
            (),
        )
    if c_squared == rho2:
        raise MModeHorizonUnresolved("the frozen trajectory is tangent to the horizon")

    radius = math.hypot(a_coefficient, b_coefficient)
    phase = math.atan2(b_coefficient, a_coefficient)
    offset = math.acos(max(-1.0, min(1.0, -c_coefficient / radius)))
    trajectory = FrozenHorizonTrajectory(
        a_coefficient, b_coefficient, c_coefficient, "transverse", ()
    )
    approximate = [
        (phase + offset) / (2.0 * math.pi),
        (phase - offset) / (2.0 * math.pi),
    ]
    enclosures: list[HorizonRootEnclosure] = []
    for guess in approximate:
        lifted = _lift_into_domain(guess, horizon_lo, horizon_hi)
        enclosures.append(_refine_root(trajectory, lifted))
    enclosures.sort(key=lambda entry: entry.turn_lo)
    if enclosures[0].turn_hi >= enclosures[1].turn_lo:
        raise MModeHorizonUnresolved("the two frozen root enclosures are not disjoint")
    if enclosures[0].orientation == enclosures[1].orientation:
        raise MModeHorizonUnresolved("the two frozen roots share one orientation label")
    return FrozenHorizonTrajectory(
        a_coefficient,
        b_coefficient,
        c_coefficient,
        "transverse",
        tuple(enclosures),
    )


def _lift_into_domain(
    guess: float, horizon_lo: Fraction, horizon_hi: Fraction
) -> Fraction:
    """Lift an approximate turn by the unique integer turn landing in ``H_N``.

    The seed is snapped to a **dyadic** rational so that every bisection
    midpoint below stays dyadic with a bounded numerator; a decimal seed would
    make each halving grow the denominator without bound.
    """
    value = Fraction(round(float(guess) * (1 << 44)), 1 << 44)
    lift = math.floor(horizon_lo - value) + 1
    while value + lift > horizon_hi:
        lift -= 1
    while value + lift < horizon_lo:
        lift += 1
    return value + lift


def _refine_root(
    trajectory: FrozenHorizonTrajectory, seed: Fraction
) -> HorizonRootEnclosure:
    """Refine one analytic root into a certified sign-changing turn bracket.

    The *search* is an ordinary binary64 bisection on dyadic turns: the frozen
    trajectory is the closed form ``A cos + B sin + C``, whose binary64
    evaluation resolves a root far below the ``1e-13 rad`` width Section 12.1
    asks for.  What the design requires to be **certified** is the retained
    bracket itself, and that is what the exact-rational interval kernel proves
    here: opposite non-zero endpoint value intervals, and a derivative interval
    over the closed bracket that excludes zero.  No certified evaluation runs
    inside the search loop, so the proof obligation is unchanged while the cost
    is not quadratic in the direction count.
    """
    width = Fraction(1, 1 << 20)
    lower = seed - width
    upper = seed + width
    attempts = 0
    while trajectory.value(float(lower)) * trajectory.value(float(upper)) > 0.0:
        lower -= width
        upper += width
        width *= 2
        attempts += 1
        if attempts > 64:  # pragma: no cover - defensive
            raise MModeHorizonUnresolved("no sign-changing frozen bracket")

    two_pi = float(TAU)
    target = Fraction(HORIZON_ROOT_WIDTH_RAD) / Fraction(*TAU.as_integer_ratio())
    f_low = trajectory.value(float(lower))
    while upper - lower > target:
        middle = (lower + upper) / 2
        f_middle = trajectory.value(float(middle))
        if f_middle == 0.0:
            lower = middle
            upper = middle
            break
        if f_low * f_middle < 0.0:
            upper = middle
        else:
            lower = middle
            f_low = f_middle
    del two_pi

    low_range = trajectory.value_interval(lower, lower)
    high_range = trajectory.value_interval(upper, upper)
    if lower != upper and not (
        (low_range[1] < 0.0 < high_range[0]) or (high_range[1] < 0.0 < low_range[0])
    ):
        # The certified endpoint intervals no longer separate zero, which means
        # the bracket has been narrowed into the certifier's own resolution.
        # Widening it back to the last provably sign-changing pair keeps the
        # retained enclosure certified rather than merely narrow.
        span = upper - lower
        for _ in range(64):
            lower -= span
            upper += span
            span *= 2
            low_range = trajectory.value_interval(lower, lower)
            high_range = trajectory.value_interval(upper, upper)
            if (low_range[1] < 0.0 < high_range[0]) or (
                high_range[1] < 0.0 < low_range[0]
            ):
                break
        else:  # pragma: no cover - defensive
            raise MModeHorizonUnresolved(
                "no certified sign-changing frozen bracket could be retained"
            )

    derivative_range = trajectory.derivative_interval(lower, upper)
    if derivative_range[0] <= 0.0 <= derivative_range[1]:
        raise MModeHorizonUnresolved(
            "the certified derivative interval of a frozen root contains zero"
        )
    orientation: Literal["rising", "setting"] = (
        "rising" if derivative_range[0] > 0.0 else "setting"
    )
    residual = abs(trajectory.value(float((lower + upper) / 2)))
    if not math.isfinite(residual) or residual > HORIZON_ROOT_RESIDUAL:
        raise MModeHorizonUnresolved(
            "a retained frozen root exceeds its certified numerator residual"
        )
    return HorizonRootEnclosure(
        turn_lo=lower, turn_hi=upper, orientation=orientation, residual=residual
    )
