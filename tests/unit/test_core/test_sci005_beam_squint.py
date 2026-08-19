"""SCI-005 Stage-2 analytic invariants for native-feed beam squint.

``docs/development/sci005_beam_physics_plan.md`` Sections 4.1, 4.1.1 and 4.2.1
define one strict ``beams.squint`` block whose two native feeds sample the
antenna's existing scalar pattern at oppositely displaced directions, and one
receptor-conjugated composition that carries the result into the *existing*
sky-side ``E`` slot without changing Jones-chain order:

.. math::

    \\delta(\\nu)=\\sin^{-1}\\!\\left[\\frac{\\nu_{\\rm ref}}{\\nu}
    \\sin\\delta_{\\rm ref}\\right],
    \\qquad
    D_b=\\operatorname{diag}(b_0,b_1),
    \\qquad
    E=C^{\\dagger}D_bC,
    \\qquad CE=D_bC.

Every oracle below is built in the test body from the frozen design text and a
published closed form, never by importing the production helper that the same
production code uses:

* the displaced sampling directions are Rodrigues rotations
  :math:`\\hat{\\mathbf n}_f=R(-s_f\\delta;\\hat{\\mathbf a}_p)\\hat{\\mathbf n}`
  about the horizontal axis
  :math:`\\hat{\\mathbf a}_p=\\sin\\beta_{\\rm squint}\\hat{\\mathbf N}
  -\\cos\\beta_{\\rm squint}\\hat{\\mathbf E}` of Section 4.2.1, written out
  here in the right-handed ``(East, North, Up)`` triad;
* the scalar response of the fixture's uniformly illuminated circular aperture
  is the closed-form Airy voltage :math:`2J_1(x)/x` with
  :math:`x=(\\pi D/\\lambda)\\sin\\theta`, evaluated from ``scipy.special.jv``
  in the test body; and
* the receptor matrix is rebuilt from Section 4.2.1's frozen formulas
  ``C = M(basis) @ R(chi)``, ``R(chi) = [[cos chi, sin chi], [-sin chi,
  cos chi]]``, ``M(linear) = [[0, 1], [1, 0]]`` and
  ``M(circular) = (1/sqrt(2)) * [[1, i], [1, -i]]`` -- deliberately *not*
  imported from :mod:`radiosim.core.jones.receptor`, whose
  ``receptor_matrix`` is the production side of the comparison.

**Tolerances.** Section 4.2.1 fixes no numeric tolerance of its own, and the
Section 8.1 Stage-2 envelope constrains ``atol`` only to be positive with two
relations on top of it (``factorization_max_abs_residual <= atol`` and
``order_control_max_abs_difference >= max(1e-3, 1024*atol)``). Rather than
invent a number this module reuses the memo's own frozen float64 comparison
tolerance from Section 3.3, ``atol = max(1e-12, 32*eps)``, and the two frozen
relations above it; the small-angle control uses Section 8.1's own
``>= 8 * tolerance`` factor.

**Two frozen binding names.** The original gate left ``load_beam_system``'s
new receptor keyword and the resolved squint record's class and attribute
unnamed; the accepted heading-and-binding correction (the operative ``D2``)
froze them as the keyword-only ``receptors`` parameter, the frozen dataclass
``ResolvedSquint``, and the ``ResolvedBeamAssignment.squint`` attribute --
each by exact parallel with the ``pointing`` / ``surface_error`` /
``aperture_physics`` attributes the frozen config field name ``beams.squint``
mirrors. This module binds those frozen names through
:func:`_load_beam_system_with_receptors` and
:func:`_resolved_squint_record` alone, so every assertion reads the frozen
surface and nothing implementation-private.

**Why the new names are imported inside tests.** Section 4.1.1 rules two new
``BeamLoadError`` subclasses that do not exist yet. Importing them at module
scope would take collection down and hide the green controls this slice is
required to keep green (the no-squint call surface and the pre-Stage-2 response
key), so they are imported per test exactly as Stage 1 imported
``InvalidBeamGeometryError``.
"""

from __future__ import annotations

import inspect
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from scipy.special import jv

from radiosim.core.instrument import AntennaId
from radiosim.core.precision import COMPLEX256_AVAILABLE
from tests.fixtures.configs import valid_config_mapping

_SPEED_OF_LIGHT_M_PER_S = 299792458.0

# --- frozen literals ----------------------------------------------------------

#: Section 4.1's one accepted ``convention`` literal.
SQUINT_CONVENTION = "cotton_uson_exact_v1"

#: Section 4.2.1's three convention literals, which enter the squint payload
#: beside the six resolved field values.
DIRECTION_CONVENTION = "feed_ray_plus_half_pi_north_through_east_v1"
FRAME_CONVENTION = "pointing_then_squint_great_circle_v1"
FACTORIZATION_CONVENTION = "receptor_conjugated_native_diagonal_v1"

#: Section 4.1's five accepted mount literals and their ``(eta_p, nu_p)``
#: field-rotation factors.  ``None`` retains its accepted ``fixed`` reading.
MOUNT_FACTORS: dict[str, tuple[float, float]] = {
    "alt-az": (1.0, 0.0),
    "equatorial": (0.0, 0.0),
    "fixed": (0.0, 0.0),
    "alt-az+nasmyth-r": (1.0, 1.0),
    "alt-az+nasmyth-l": (1.0, -1.0),
}

#: The two accepted native feed pairs.  Section 4.1.1: "The other native feed
#: of a squint record is fixed by the label pair of its basis".
NATIVE_FEED_ORDER: dict[str, tuple[str, str]] = {
    "linear": ("x", "y"),
    "circular": ("r", "l"),
}

#: Section 3.3's frozen float64 comparison tolerances, reused here (see the
#: module docstring: Stage 2 freezes no numeric tolerance of its own).
_EPS = float(np.finfo(np.float64).eps)
ATOL = max(1e-12, 32.0 * _EPS)
RTOL = max(1e-10, 32.0 * _EPS)

#: Section 8.1's frozen Stage-2 separation bound for a negative control,
#: ``max(1e-3, 1024 * atol)``.
SEPARATION_BOUND = max(1e-3, 1024.0 * ATOL)

#: Section 8.1's frozen small-angle separation factor, ``8 * tolerance``.
SMALL_ANGLE_FACTOR = 8.0

# --- the shipped fixture --------------------------------------------------

#: ``tests/fixtures/configs.py`` gives both antennas this diameter.
FIXTURE_DIAMETER_M = 14.0

ANT0 = AntennaId(0, "ANT0")
ANT1 = AntennaId(1, "ANT1")

#: Three strictly increasing observation channels.  Section 8.1's
#: ``squint_frequency_laws`` row requires at least three samples.
CHANNEL_FREQUENCIES_HZ: tuple[float, ...] = (1.0e8, 1.5e8, 2.0e8)

EXPLICIT_BAND: dict[str, Any] = {
    "mode": "explicit",
    "channel_frequencies_hz": list(CHANNEL_FREQUENCIES_HZ),
    "channel_widths_hz": [1.0e6, 1.0e6, 1.0e6],
}

#: A uniformly illuminated circular aperture: its far-field voltage is exactly
#: the Airy form ``2 J1(x)/x``, so the oracle below is a published closed form
#: rather than a second numerical model.
UNIFORM_CIRCULAR: dict[str, Any] = {
    "kind": "circular_aperture",
    "taper": {"kind": "uniform"},
}

#: The reference squint record.  ``2 deg`` at 150 MHz keeps every resolved
#: offset well inside the main lobe at the highest channel: the first Airy null
#: of a 14 m dish at 200 MHz is ``1.22 lambda / D = 0.1307 rad``, while the
#: largest resolved offset in the band is ``asin(1.5 sin 2deg) = 0.0524 rad``.
REFERENCE_FREQUENCY_HZ = 1.5e8
PER_FEED_OFFSET_DEG = 2.0
MECHANICAL_ANGLE_DEG = 35.0

#: A boresight pair for a non-rotating mount: Section 4.2.1 says the adapter
#: supplies exactly ``0.0`` for ``eta_p == 0``, and the unpointed boresight is
#: the topocentric zenith.
FIXED_BORESIGHT: dict[str, float] = {
    "boresight_parallactic_rad": 0.0,
    "boresight_altitude_rad": math.pi / 2.0,
}

#: Probe directions inside the main lobe at every channel.  The first entry is
#: the resolved boresight itself, which is where the midpoint invariant lives.
_PROBE_ZENITH_ANGLE_RAD = np.array([0.0, 0.02, 0.035, 0.05, 0.05], dtype=np.float64)
_PROBE_AZIMUTH_RAD = np.array([0.0, 0.4, 2.1, 3.6, 5.2], dtype=np.float64)
PROBE_ALTITUDE_RAD = np.pi / 2.0 - _PROBE_ZENITH_ANGLE_RAD
PROBE_AZIMUTH_RAD = _PROBE_AZIMUTH_RAD


# --- document builders --------------------------------------------------------


def _squint_record(
    *,
    reference_frequency_hz: float = REFERENCE_FREQUENCY_HZ,
    per_feed_offset_deg_at_reference: float = PER_FEED_OFFSET_DEG,
    mechanical_feed_position_angle_deg: float = MECHANICAL_ANGLE_DEG,
    positive_native_feed: str = "x",
) -> dict[str, Any]:
    """One complete Section 4.1 squint record, all five fields authored."""
    return {
        "convention": SQUINT_CONVENTION,
        "reference_frequency_hz": reference_frequency_hz,
        "per_feed_offset_deg_at_reference": per_feed_offset_deg_at_reference,
        "mechanical_feed_position_angle_deg": mechanical_feed_position_angle_deg,
        "positive_native_feed": positive_native_feed,
    }


def _analytic_beams(
    *,
    squint: dict[str, Any] | None = None,
    model: dict[str, Any] | None = None,
    surface_error: dict[str, Any] | None = None,
) -> dict[str, Any]:
    beams: dict[str, Any] = {
        "mode": "analytic",
        "model": dict(UNIFORM_CIRCULAR if model is None else model),
    }
    if squint is not None:
        beams["squint"] = squint
    if surface_error is not None:
        beams["surface_error"] = surface_error
    return beams


def _default_squint_beams(**record: Any) -> dict[str, Any]:
    return _analytic_beams(squint={"default": _squint_record(**record)})


# --- resolution and load ------------------------------------------------------


def _resolve(
    tmp_path: Path,
    beams: dict[str, Any],
    *,
    receptors: dict[str, Any] | None = None,
    beam_precision: str | None = None,
) -> Any:
    """Resolve one document against the shipped two-antenna fixture."""
    from radiosim.io.config_resolution import ConfigurationSource, resolve_config

    tmp_path.mkdir(parents=True, exist_ok=True)
    overrides: dict[str, Any] = {"beams": beams, "frequency": dict(EXPLICIT_BAND)}
    if receptors is not None:
        overrides["receptors"] = receptors
    data = valid_config_mapping(tmp_path, **overrides)
    if beam_precision is not None:
        data["execution"]["precision"] = {"jones": {"beam": beam_precision}}
    return resolve_config(
        data,
        source=ConfigurationSource.for_mapping(
            base_dir=tmp_path,
            invocation_dir=tmp_path,
        ),
    )


def _with_mount_types(instrument: Any, mounts: Any) -> Any:
    """Return the resolved instrument with these mount types stamped on.

    No instrument source RadioSim reads from a layout file carries a mount
    type, so a mount-dependent rule is invisible on every shipped fixture.
    This is the same restamp
    :func:`tests.unit.test_core.test_jones_resolution.restamp_mount_types`
    performs, reduced to the :class:`ResolvedInstrument` that
    ``resolve_beam_assignments`` consumes, and it recomputes
    ``instrument_sha256`` from the canonical content through the production
    function so the result is a genuine resolved instrument.
    """
    from dataclasses import replace

    from radiosim.core.instrument import _compute_instrument_sha256

    if isinstance(mounts, str) or mounts is None:
        wanted: tuple[str | None, ...] = (mounts,) * len(instrument.antennas)
    else:
        wanted = tuple(mounts)
    antennas = tuple(
        replace(antenna, mount_type=mount)
        for antenna, mount in zip(instrument.antennas, wanted, strict=True)
    )
    return replace(
        instrument,
        antennas=antennas,
        provenance=replace(
            instrument.provenance,
            instrument_sha256=_compute_instrument_sha256(
                instrument.name,
                instrument.location,
                antennas,
                telescope_name_source=instrument.provenance.telescope_name_source,
                location_source=instrument.provenance.location_source,
            ),
        ),
    )


def _load_beam_system_with_receptors(
    state: Any,
    *,
    observation_frequencies_hz: Any,
    precision: Any,
    receptors: Any,
) -> Any:
    """Section 4.2.1's widened load, through the one keyword this slice picked.

    ``D2`` rules that ``load_beam_system`` "gains the resolved receptor set ...
    through a new keyword" and that it "requires it whenever any resolved
    antenna carries squint", but freezes no spelling for the keyword. This
    helper is the single place the chosen spelling appears.
    """
    from radiosim.core.beam.runtime import load_beam_system

    return load_beam_system(
        state,
        observation_frequencies_hz=observation_frequencies_hz,
        precision=precision,
        receptors=receptors,
    )


def _beam_system(
    tmp_path: Path,
    beams: dict[str, Any],
    *,
    receptors: dict[str, Any] | None = None,
    mount_types: Any = None,
    beam_precision: str | None = None,
) -> tuple[Any, Any, Any, Any]:
    """Return ``(system, instrument, receptor_set, resolved_beam_state)``.

    A document with no ``squint`` block is loaded through the *unwidened*
    call, so the green controls in this module exercise exactly today's call
    surface rather than a Stage-2 one.
    """
    from radiosim.core.beam.resolution import resolve_beam_assignments
    from radiosim.core.beam.runtime import load_beam_system
    from radiosim.core.instrument_resolution import resolve_instrument
    from radiosim.core.receptor import resolve_receptors

    bundle = _resolve(
        tmp_path, beams, receptors=receptors, beam_precision=beam_precision
    )
    runtime = bundle.runtime
    instrument = resolve_instrument(runtime.instrument)
    if mount_types is not None:
        instrument = _with_mount_types(instrument, mount_types)
    receptor_set = resolve_receptors(runtime.receptors, instrument)
    state = resolve_beam_assignments(runtime.beams, instrument)
    frequencies = runtime.frequency.channel_frequencies_hz
    if "squint" in beams:
        system = _load_beam_system_with_receptors(
            state,
            observation_frequencies_hz=frequencies,
            precision=runtime.execution.precision,
            receptors=receptor_set,
        )
    else:
        system = load_beam_system(
            state,
            observation_frequencies_hz=frequencies,
            precision=runtime.execution.precision,
        )
    return system, instrument, receptor_set, state


def _resolved_squint_record(state: Any, antenna_id: AntennaId) -> Any:
    """Return one antenna's resolved squint record from the beam state."""
    for assignment in state.assignments:
        if assignment.antenna_id == antenna_id:
            return assignment.squint
    raise AssertionError(f"no resolved assignment for {antenna_id!r}")


def _evaluate(
    system: Any,
    antenna_id: AntennaId,
    *,
    frequency_hz: float,
    altitude_rad: np.ndarray = PROBE_ALTITUDE_RAD,
    azimuth_rad: np.ndarray = PROBE_AZIMUTH_RAD,
    boresight: dict[str, float] | None = None,
    time_mjd: float = 60000.0,
) -> np.ndarray:
    """Evaluate the composed ``E`` batch, with the Section 4.2.1 boresight pair."""
    kwargs = dict(FIXED_BORESIGHT if boresight is None else boresight)
    return np.asarray(
        system.evaluate_jones(
            antenna_id,
            altitude_rad=altitude_rad,
            azimuth_rad=azimuth_rad,
            frequency_hz=frequency_hz,
            time_mjd=time_mjd,
            **kwargs,
        )
    )


# --- independent geometry oracle ----------------------------------------------
#
# Every vector below is written in the right-handed ``(East, North, Up)``
# triad.  Section 4.2.1 states the squint direction and rotation axis in the
# ``(North, East)`` tangent pair; ``N x E = -U`` there, so evaluating the
# Rodrigues cross product with the components in that written order would flip
# the sign of the rotation.  Writing them as ``(E, N, U)`` keeps the frame
# right-handed and reproduces the memo's own statement that "rotating the
# beam-frame zenith by ``+delta`` about ``a_p`` moves it along ``+u``".

_BEAM_FRAME_ZENITH = np.array([0.0, 0.0, 1.0], dtype=np.float64)


def _unit_vector(altitude_rad: np.ndarray, azimuth_rad: np.ndarray) -> np.ndarray:
    """``(E, N, U)`` components of a direction, North through East."""
    altitude = np.asarray(altitude_rad, dtype=np.float64)
    azimuth = np.asarray(azimuth_rad, dtype=np.float64)
    cos_altitude = np.cos(altitude)
    return np.stack(
        [
            cos_altitude * np.sin(azimuth),
            cos_altitude * np.cos(azimuth),
            np.sin(altitude),
        ],
        axis=-1,
    )


def _squint_unit(beta_rad: float) -> np.ndarray:
    """``u_squint = cos(beta) N + sin(beta) E`` as ``(E, N, U)``."""
    return np.array([math.sin(beta_rad), math.cos(beta_rad), 0.0], dtype=np.float64)


def _squint_axis(beta_rad: float) -> np.ndarray:
    """``a_p = sin(beta) N - cos(beta) E`` as ``(E, N, U)``."""
    return np.array([-math.cos(beta_rad), math.sin(beta_rad), 0.0], dtype=np.float64)


def _rodrigues(vectors: np.ndarray, axis: np.ndarray, angle_rad: float) -> np.ndarray:
    """Rotate ``vectors`` by ``angle_rad`` about the unit ``axis``."""
    cos_angle = math.cos(angle_rad)
    sin_angle = math.sin(angle_rad)
    vectors = np.asarray(vectors, dtype=np.float64)
    cross = np.cross(np.broadcast_to(axis, vectors.shape), vectors)
    dot = vectors @ axis
    return (
        vectors * cos_angle
        + cross * sin_angle
        + axis * dot[..., None] * (1.0 - cos_angle)
    )


def _wrap_to_pi(angle_rad: float) -> float:
    """Section 4.1's ``wrap`` onto the canonical ``(-pi, pi]`` interval."""
    wrapped = math.remainder(angle_rad, 2.0 * math.pi)
    return math.pi if wrapped == -math.pi else wrapped


def _feed_position_angle(
    *,
    mechanical_deg: float,
    mount_type: str | None,
    parallactic_rad: float,
    altitude_rad: float,
) -> float:
    """Section 4.1/4.2.1: ``wrap(beta_mech + eta psi + nu alt)``."""
    eta, nu = MOUNT_FACTORS["fixed" if mount_type is None else mount_type]
    return _wrap_to_pi(
        math.radians(mechanical_deg) + eta * parallactic_rad + nu * altitude_rad
    )


def _exact_offset_rad(
    frequency_hz: float,
    *,
    reference_frequency_hz: float = REFERENCE_FREQUENCY_HZ,
    per_feed_offset_deg: float = PER_FEED_OFFSET_DEG,
) -> float:
    """Section 4.1.1's exact binary64 Cotton/Uson arcsine law."""
    return math.asin(
        (reference_frequency_hz / frequency_hz)
        * math.sin(math.radians(per_feed_offset_deg))
    )


def _small_angle_offset_rad(
    frequency_hz: float,
    *,
    reference_frequency_hz: float = REFERENCE_FREQUENCY_HZ,
    per_feed_offset_deg: float = PER_FEED_OFFSET_DEG,
) -> float:
    """Section 8.1's small-angle limit, which is *not* the production law."""
    return math.radians(per_feed_offset_deg) * reference_frequency_hz / frequency_hz


def _airy_voltage(
    zenith_angle_rad: np.ndarray,
    *,
    diameter_m: float = FIXTURE_DIAMETER_M,
    frequency_hz: float,
    dtype: Any = np.complex128,
) -> np.ndarray:
    """The uniformly illuminated circular aperture's closed-form voltage.

    ``2 J1(x)/x`` with ``x = (pi D / lambda) sin(theta)`` and the removable
    singularity ``e(0) = 1`` written out rather than divided by zero.
    """
    wavelength_m = _SPEED_OF_LIGHT_M_PER_S / float(frequency_hz)
    argument = (math.pi * diameter_m / wavelength_m) * np.sin(
        np.asarray(zenith_angle_rad, dtype=np.float64)
    )
    safe = np.where(argument == 0.0, 1.0, argument)
    voltage = np.where(argument == 0.0, 1.0, 2.0 * jv(1, safe) / safe)
    return np.asarray(voltage, dtype=dtype)


def _sampled_zenith_angles(
    *,
    altitude_rad: np.ndarray,
    azimuth_rad: np.ndarray,
    beta_squint_rad: float,
    signed_offset_rad: float,
) -> np.ndarray:
    """Zenith angles of ``R(-s_f delta; a_p) n`` for every probe direction."""
    axis = _squint_axis(beta_squint_rad)
    rotated = _rodrigues(
        _unit_vector(altitude_rad, azimuth_rad), axis, -signed_offset_rad
    )
    return np.arccos(np.clip(rotated[..., 2], -1.0, 1.0))


def _plan_receptor_matrix(basis: str, chi_rad: float, dtype: Any) -> np.ndarray:
    """Section 4.2.1's ``C = M(basis) @ R(chi)``, written from the memo."""
    real_dtype = np.empty(0, dtype=dtype).real.dtype
    chi = real_dtype.type(chi_rad)
    rotation = np.array(
        [
            [np.cos(chi), np.sin(chi)],
            [-np.sin(chi), np.cos(chi)],
        ],
        dtype=dtype,
    )
    if basis == "linear":
        leading = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=dtype)
    elif basis == "circular":
        one = real_dtype.type(1.0)
        inverse_root_two = one / np.sqrt(real_dtype.type(2.0))
        leading = inverse_root_two * np.array(
            [[1.0, 1.0j], [1.0, -1.0j]],
            dtype=dtype,
        )
    else:  # pragma: no cover - the accepted vocabulary has exactly two members
        raise AssertionError(f"unknown receptor basis {basis!r}")
    return leading @ rotation


def _expected_native_diagonal(
    *,
    basis: str,
    positive_native_feed: str,
    beta_squint_rad: float,
    offset_rad: float,
    altitude_rad: np.ndarray,
    azimuth_rad: np.ndarray,
    frequency_hz: float,
    dtype: Any = np.complex128,
) -> np.ndarray:
    """``D_b = diag(b_0, b_1)`` in the antenna's resolved native feed order."""
    samples = []
    for label in NATIVE_FEED_ORDER[basis]:
        sign = 1.0 if label == positive_native_feed else -1.0
        zenith_angles = _sampled_zenith_angles(
            altitude_rad=altitude_rad,
            azimuth_rad=azimuth_rad,
            beta_squint_rad=beta_squint_rad,
            signed_offset_rad=sign * offset_rad,
        )
        samples.append(
            _airy_voltage(zenith_angles, frequency_hz=frequency_hz, dtype=dtype)
        )
    diagonal = np.zeros((np.asarray(altitude_rad).size, 2, 2), dtype=dtype)
    diagonal[:, 0, 0] = samples[0]
    diagonal[:, 1, 1] = samples[1]
    return diagonal


def _expected_composed_e(
    *,
    basis: str = "linear",
    chi_rad: float = 0.0,
    positive_native_feed: str = "x",
    mechanical_deg: float = MECHANICAL_ANGLE_DEG,
    mount_type: str | None = None,
    parallactic_rad: float = 0.0,
    boresight_altitude_rad: float = math.pi / 2.0,
    offset_rad: float | None = None,
    altitude_rad: np.ndarray = PROBE_ALTITUDE_RAD,
    azimuth_rad: np.ndarray = PROBE_AZIMUTH_RAD,
    frequency_hz: float,
    dtype: Any = np.complex128,
) -> np.ndarray:
    """The independently composed ``E = C^dagger D_b C`` of Section 4.2.1."""
    beta_feed = _feed_position_angle(
        mechanical_deg=mechanical_deg,
        mount_type=mount_type,
        parallactic_rad=parallactic_rad,
        altitude_rad=boresight_altitude_rad,
    )
    beta_squint = _wrap_to_pi(beta_feed + math.pi / 2.0)
    resolved_offset = (
        _exact_offset_rad(frequency_hz) if offset_rad is None else offset_rad
    )
    diagonal = _expected_native_diagonal(
        basis=basis,
        positive_native_feed=positive_native_feed,
        beta_squint_rad=beta_squint,
        offset_rad=resolved_offset,
        altitude_rad=altitude_rad,
        azimuth_rad=azimuth_rad,
        frequency_hz=frequency_hz,
        dtype=dtype,
    )
    receptor = _plan_receptor_matrix(basis, chi_rad, dtype)
    return receptor.conj().T @ diagonal @ receptor


def _max_abs_difference(left: np.ndarray, right: np.ndarray) -> float:
    left_array = np.asarray(left)
    right_array = np.asarray(right)
    assert left_array.shape == right_array.shape
    return float(np.max(np.abs(left_array - right_array)))


# ==============================================================================
# Section 4.1.1: the two new typed load rejections
# ==============================================================================


def test_the_two_squint_load_errors_are_append_only_beam_load_errors() -> None:
    """Section 4.1.1 and Section 7.3's bounded append-only grant.

    Section 7.3 grants ``core/beam/errors.py`` "exactly two new classes
    ``SquintFrequencyDomainError(BeamLoadError)`` and
    ``SquintReceptorBasisError(BeamLoadError)``, each with docstring and its
    ``__all__`` entry; no existing byte changes", and grants
    ``core/beam/__init__.py`` "the two new error exports alone".
    """
    import radiosim.core.beam as beam_package
    from radiosim.core.beam import errors as errors_module
    from radiosim.core.beam.errors import (
        SquintFrequencyDomainError,
        SquintReceptorBasisError,
    )

    for new_error in (SquintFrequencyDomainError, SquintReceptorBasisError):
        assert issubclass(new_error, errors_module.BeamLoadError)
        assert issubclass(new_error, errors_module.BeamError)
        assert new_error.__name__ in errors_module.__all__
        assert getattr(beam_package, new_error.__name__) is new_error
        assert new_error.__name__ in beam_package.__all__
    assert SquintFrequencyDomainError is not SquintReceptorBasisError

    # Append-only: every pre-existing load error is still its own class.
    for existing in (
        "BeamDependencyError",
        "BeamFileReadError",
        "BeamFileChangedError",
        "UnsupportedBeamMetadataError",
        "BeamNormalizationError",
        "UnsupportedBeamPrecisionError",
        "BeamSamplingDerivationError",
    ):
        assert existing in errors_module.__all__
        assert getattr(errors_module, existing) not in (
            SquintFrequencyDomainError,
            SquintReceptorBasisError,
        )


def test_an_out_of_domain_arcsine_argument_is_rejected_at_beam_system_load(
    tmp_path: Path,
) -> None:
    """Section 4.1.1: the preflight "rejects, never clips".

    ``sin(30 deg) = 0.5`` scaled by ``150/100`` is ``0.75`` and scaled by
    ``150/50`` would be ``1.5``; the 50 MHz channel below is therefore the one
    offending frequency, and the exact binary64 argument for it leaves
    ``[-1, 1]``.
    """
    from radiosim.core.beam.errors import SquintFrequencyDomainError

    offending_frequency_hz = 5.0e7
    beams = _default_squint_beams(per_feed_offset_deg_at_reference=30.0)
    argument = (REFERENCE_FREQUENCY_HZ / offending_frequency_hz) * math.sin(
        math.radians(30.0)
    )
    assert argument > 1.0

    from radiosim.core.beam.resolution import resolve_beam_assignments
    from radiosim.core.instrument_resolution import resolve_instrument
    from radiosim.core.receptor import resolve_receptors

    bundle = _resolve(tmp_path, beams)
    runtime = bundle.runtime
    instrument = resolve_instrument(runtime.instrument)
    receptor_set = resolve_receptors(runtime.receptors, instrument)
    state = resolve_beam_assignments(runtime.beams, instrument)

    with pytest.raises(SquintFrequencyDomainError) as error:
        _load_beam_system_with_receptors(
            state,
            observation_frequencies_hz=(offending_frequency_hz,),
            precision=runtime.execution.precision,
            receptors=receptor_set,
        )

    message = str(error.value)
    # Section 4.1.1: the message "names the antenna, the offending observation
    # frequency, the reference frequency, and the reference offset".
    assert "ANT0" in message or "0" in message
    assert repr(offending_frequency_hz) in message or "5" in message
    assert repr(REFERENCE_FREQUENCY_HZ) in message or "150" in message
    assert "30" in message


def test_an_in_domain_band_is_not_rejected_by_the_arcsine_preflight(
    tmp_path: Path,
) -> None:
    """The preflight is a domain check, not a blanket rejection.

    The shipped band's smallest channel is 100 MHz, where the argument is
    ``1.5 sin(2 deg) = 0.05235`` -- comfortably inside ``[-1, 1]`` -- so the
    same document that is rejected above must load here.
    """
    system, _instrument, _receptors, state = _beam_system(
        tmp_path, _default_squint_beams()
    )

    assert system.state is not None
    assert _resolved_squint_record(state, ANT0) is not None


@pytest.mark.parametrize(
    ("basis", "positive_native_feed"),
    [("linear", "r"), ("linear", "l"), ("circular", "x"), ("circular", "y")],
)
def test_a_feed_label_from_the_wrong_basis_is_rejected_at_beam_system_load(
    tmp_path: Path,
    basis: str,
    positive_native_feed: str,
) -> None:
    """Section 4.1.1: ``x``/``y`` require ``linear`` and ``r``/``l`` require
    ``circular``, and the check is owned by load rather than the document
    "because per-antenna receptor bases exist only after receptor resolution"."""
    from radiosim.core.beam.errors import SquintReceptorBasisError

    with pytest.raises(SquintReceptorBasisError) as error:
        _beam_system(
            tmp_path,
            _default_squint_beams(positive_native_feed=positive_native_feed),
            receptors={"default": {"basis": basis}, "output_basis": basis},
        )

    message = str(error.value)
    assert positive_native_feed in message
    assert basis in message
    assert "ANT0" in message or "0" in message


def test_an_unknown_squint_antenna_reference_is_rejected_at_assignment_resolution(
    tmp_path: Path,
) -> None:
    """Section 4.1.1: ``resolve_beam_assignments`` resolves the per-antenna
    squint map "exactly as ``beams.pointing`` does today"."""
    from radiosim.core.beam.errors import UnknownBeamAntennaError
    from radiosim.core.beam.resolution import resolve_beam_assignments
    from radiosim.core.instrument_resolution import resolve_instrument

    beams = _analytic_beams(
        squint={
            "per_antenna": [
                {
                    "antenna": {"kind": "number", "number": 7},
                    **_squint_record(),
                }
            ]
        }
    )
    bundle = _resolve(tmp_path, beams)
    instrument = resolve_instrument(bundle.runtime.instrument)

    with pytest.raises(UnknownBeamAntennaError) as error:
        resolve_beam_assignments(bundle.runtime.beams, instrument)

    assert "beams.squint.per_antenna" in str(error.value)


def test_a_repeated_squint_antenna_reference_is_rejected_at_assignment_resolution(
    tmp_path: Path,
) -> None:
    """Section 4.1.1: "a repeated canonical antenna raises the existing typed
    ``DuplicateBeamAssignmentError``"."""
    from radiosim.core.beam.errors import DuplicateBeamAssignmentError
    from radiosim.core.beam.resolution import resolve_beam_assignments
    from radiosim.core.instrument_resolution import resolve_instrument

    beams = _analytic_beams(
        squint={
            "per_antenna": [
                {"antenna": {"kind": "number", "number": 0}, **_squint_record()},
                {
                    "antenna": {"kind": "name", "name": "ANT0"},
                    **_squint_record(positive_native_feed="y"),
                },
            ]
        }
    )
    bundle = _resolve(tmp_path, beams)
    instrument = resolve_instrument(bundle.runtime.instrument)

    with pytest.raises(DuplicateBeamAssignmentError) as error:
        resolve_beam_assignments(bundle.runtime.beams, instrument)

    assert "beams.squint.per_antenna" in str(error.value)


#: Section 4.2.1's exact frozen degeneracy message.
ZENITH_BORESIGHT_MESSAGE = (
    "Beam squint on a rotating mount is undefined at an exactly zenith boresight."
)


def test_a_rotating_mount_at_an_exactly_zenith_boresight_is_rejected(
    tmp_path: Path,
) -> None:
    """Section 4.2.1's one named limitation, with its exact message.

    "An alt-az antenna with no pointing offset is exactly this case": with no
    ``beams.pointing`` the resolved boresight *is* the topocentric zenith, its
    altitude is exactly ``pi/2`` in binary64, and the parallactic angle there
    is undefined. The adapter must raise rather than adopt ``arctan2(0, 0)``.

    ``jones.P`` is enabled because R15 already requires it for a rotating
    mount, and Section 4.1 keeps that rule ("A rotating mount still requires
    ``jones.P``, as it does today"), so a document without it would be
    rejected for the wrong reason.
    """
    from radiosim.core.beam.errors import BeamAngularDomainError
    from tests.unit.test_core.test_jones_resolution import simulator_for

    tmp_path.mkdir(parents=True, exist_ok=True)
    simulator = simulator_for(
        tmp_path,
        {"P": {"enabled": True}},
        mount_types="alt-az",
        beams=_default_squint_beams(),
        frequency=dict(EXPLICIT_BAND),
    )

    with pytest.raises(BeamAngularDomainError) as error:
        simulator.run(progress=False)

    assert str(error.value) == ZENITH_BORESIGHT_MESSAGE


# ==============================================================================
# Section 4.2.1: the widened ``evaluate_jones`` call surface
# ==============================================================================


def test_evaluate_jones_gains_exactly_two_keyword_only_boresight_parameters() -> None:
    """Section 4.2.1 freezes both names, both kinds, and both defaults."""
    from radiosim.core.beam import BeamSystem

    parameters = inspect.signature(BeamSystem.evaluate_jones).parameters
    for name in ("boresight_parallactic_rad", "boresight_altitude_rad"):
        assert name in parameters, name
        parameter = parameters[name]
        assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
        assert parameter.default is None

    # Nothing else about the surface moved: the accepted parameters keep their
    # names, their kinds, and ``backend``'s default.
    assert list(parameters)[:2] == ["self", "antenna_id"]
    for accepted in ("altitude_rad", "azimuth_rad", "frequency_hz", "time_mjd"):
        assert parameters[accepted].kind is inspect.Parameter.KEYWORD_ONLY
        assert parameters[accepted].default is inspect.Parameter.empty
    assert parameters["backend"].default is None


@pytest.mark.parametrize(
    "boresight",
    [
        pytest.param({}, id="both_absent"),
        pytest.param({"boresight_parallactic_rad": 0.0}, id="altitude_absent"),
        pytest.param({"boresight_altitude_rad": 1.1}, id="parallactic_absent"),
        pytest.param(
            {"boresight_parallactic_rad": None, "boresight_altitude_rad": 1.1},
            id="parallactic_none",
        ),
        pytest.param(
            {
                "boresight_parallactic_rad": float("nan"),
                "boresight_altitude_rad": 1.1,
            },
            id="parallactic_nan",
        ),
        pytest.param(
            {
                "boresight_parallactic_rad": 0.0,
                "boresight_altitude_rad": float("inf"),
            },
            id="altitude_infinite",
        ),
        pytest.param(
            {"boresight_parallactic_rad": 0, "boresight_altitude_rad": 1.1},
            id="parallactic_int",
        ),
        pytest.param(
            {"boresight_parallactic_rad": 0.0, "boresight_altitude_rad": True},
            id="altitude_bool",
        ),
    ],
)
def test_a_squint_antenna_requires_both_finite_boresight_floats(
    tmp_path: Path,
    boresight: dict[str, Any],
) -> None:
    """Section 4.2.1: "When the resolved antenna carries squint, both must be
    exact finite Python floats ... and a violation of either rule raises
    ``BeamEvaluationError``"."""
    from radiosim.core.beam.errors import BeamEvaluationError

    system, _instrument, _receptors, _state = _beam_system(
        tmp_path, _default_squint_beams()
    )

    with pytest.raises(BeamEvaluationError):
        system.evaluate_jones(
            ANT0,
            altitude_rad=PROBE_ALTITUDE_RAD,
            azimuth_rad=PROBE_AZIMUTH_RAD,
            frequency_hz=REFERENCE_FREQUENCY_HZ,
            time_mjd=60000.0,
            **boresight,
        )


@pytest.mark.parametrize(
    "boresight",
    [
        pytest.param(
            {"boresight_parallactic_rad": 0.0, "boresight_altitude_rad": 1.1},
            id="both_supplied",
        ),
        pytest.param({"boresight_parallactic_rad": 0.0}, id="parallactic_only"),
        pytest.param({"boresight_altitude_rad": 1.1}, id="altitude_only"),
    ],
)
def test_a_squint_free_antenna_requires_both_boresight_values_to_be_none(
    tmp_path: Path,
    boresight: dict[str, Any],
) -> None:
    """Section 4.2.1: "when it does not, both must be ``None``"."""
    from radiosim.core.beam.errors import BeamEvaluationError

    system, _instrument, _receptors, _state = _beam_system(tmp_path, _analytic_beams())

    with pytest.raises(BeamEvaluationError):
        system.evaluate_jones(
            ANT0,
            altitude_rad=PROBE_ALTITUDE_RAD,
            azimuth_rad=PROBE_AZIMUTH_RAD,
            frequency_hz=REFERENCE_FREQUENCY_HZ,
            time_mjd=60000.0,
            **boresight,
        )


def test_the_no_squint_call_surface_and_result_are_unchanged(
    tmp_path: Path,
) -> None:
    """GREEN CONTROL. Section 4.2.1: "The no-squint call surface, behavior, and
    results are byte-identical to today."

    Byte-identity is pinned as an equality *form* rather than a literal: the
    same document loaded twice must give the same bytes, and the response must
    still be the scalar ``e * I2`` with exactly zero off-diagonals that the
    accepted subset promises. Both statements are true before Stage 2 lands and
    must remain true after it.
    """
    first, _instrument, _receptors, _state = _beam_system(
        tmp_path / "first", _analytic_beams()
    )
    second, _instrument2, _receptors2, _state2 = _beam_system(
        tmp_path / "second", _analytic_beams()
    )

    baseline = np.asarray(
        first.evaluate_jones(
            ANT0,
            altitude_rad=PROBE_ALTITUDE_RAD,
            azimuth_rad=PROBE_AZIMUTH_RAD,
            frequency_hz=REFERENCE_FREQUENCY_HZ,
            time_mjd=60000.0,
        )
    )
    repeat = np.asarray(
        second.evaluate_jones(
            ANT0,
            altitude_rad=PROBE_ALTITUDE_RAD,
            azimuth_rad=PROBE_AZIMUTH_RAD,
            frequency_hz=REFERENCE_FREQUENCY_HZ,
            time_mjd=60000.0,
        )
    )

    assert baseline.shape == (PROBE_ALTITUDE_RAD.size, 2, 2)
    assert baseline.dtype == np.dtype(np.complex128)
    np.testing.assert_array_equal(baseline, repeat)
    np.testing.assert_array_equal(baseline[:, 0, 1], 0.0)
    np.testing.assert_array_equal(baseline[:, 1, 0], 0.0)
    np.testing.assert_array_equal(baseline[:, 0, 0], baseline[:, 1, 1])
    # The scalar subset is exactly the closed form, so a no-squint run is still
    # the accepted ``e * I2`` and not a displaced sample of it.
    zenith_angles = np.pi / 2.0 - PROBE_ALTITUDE_RAD
    expected = _airy_voltage(zenith_angles, frequency_hz=REFERENCE_FREQUENCY_HZ)
    assert _max_abs_difference(baseline[:, 0, 0], expected) <= ATOL


def test_load_beam_system_requires_the_resolved_receptor_set_under_squint(
    tmp_path: Path,
) -> None:
    """Section 4.2.1: ``load_beam_system`` "requires it whenever any resolved
    antenna carries squint".

    The unwidened call is the one every no-squint path still makes, so the
    requirement has to be conditional rather than a new mandatory argument.
    """
    from radiosim.core.beam.resolution import resolve_beam_assignments
    from radiosim.core.beam.runtime import load_beam_system
    from radiosim.core.instrument_resolution import resolve_instrument

    bundle = _resolve(tmp_path, _default_squint_beams())
    runtime = bundle.runtime
    instrument = resolve_instrument(runtime.instrument)
    state = resolve_beam_assignments(runtime.beams, instrument)

    with pytest.raises((TypeError, ValueError, RuntimeError)):
        load_beam_system(
            state,
            observation_frequencies_hz=runtime.frequency.channel_frequencies_hz,
            precision=runtime.execution.precision,
        )


# ==============================================================================
# Section 4.1: the exact Cotton/Uson frequency law
# ==============================================================================


def test_the_resolved_offset_follows_the_exact_arcsine_law_at_three_frequencies(
    tmp_path: Path,
) -> None:
    """Section 4.1's exact law, at every channel of the shipped band.

    The three resolved offsets are ``asin(1.5 sin 2deg) = 0.05237...``,
    ``asin(sin 2deg) = 0.03490...`` and ``asin(0.75 sin 2deg) = 0.02618...``
    radians, each recomputed in the test body rather than pinned as a literal.
    A production law of the wrong shape moves the two displaced sample
    directions and therefore every entry of the composed ``E``.
    """
    system, _instrument, _receptors, _state = _beam_system(
        tmp_path, _default_squint_beams()
    )

    assert len(CHANNEL_FREQUENCIES_HZ) >= 3
    assert list(CHANNEL_FREQUENCIES_HZ) == sorted(CHANNEL_FREQUENCIES_HZ)
    for frequency_hz in CHANNEL_FREQUENCIES_HZ:
        offset_rad = _exact_offset_rad(frequency_hz)
        assert 0.0 < offset_rad < math.pi / 2.0
        observed = _evaluate(system, ANT0, frequency_hz=frequency_hz)
        expected = _expected_composed_e(
            frequency_hz=frequency_hz, offset_rad=offset_rad
        )
        assert _max_abs_difference(observed, expected) <= ATOL


def test_the_exact_law_is_distinguishable_from_the_small_angle_approximation(
    tmp_path: Path,
) -> None:
    """Section 4.1: "The approximation ``delta proportional to 1/nu`` may be
    documented as a small-angle limit but is not the production law."

    At the 100 MHz control channel the two laws differ by
    ``|0.0523731 - 0.0523599| = 1.32e-5`` radians, which is Section 8.1's
    ``small_angle_abs_separation`` and is far more than its required
    ``8 * tolerance``. The response-space control then shows the production
    output really does follow the exact law and not the approximation.
    """
    control_frequency_hz = 1.0e8
    assert control_frequency_hz in CHANNEL_FREQUENCIES_HZ
    assert control_frequency_hz != REFERENCE_FREQUENCY_HZ

    exact_rad = _exact_offset_rad(control_frequency_hz)
    small_angle_rad = _small_angle_offset_rad(control_frequency_hz)
    separation_rad = abs(small_angle_rad - exact_rad)
    assert separation_rad >= SMALL_ANGLE_FACTOR * ATOL

    system, _instrument, _receptors, _state = _beam_system(
        tmp_path, _default_squint_beams()
    )
    observed = _evaluate(system, ANT0, frequency_hz=control_frequency_hz)
    exact = _expected_composed_e(
        frequency_hz=control_frequency_hz, offset_rad=exact_rad
    )
    approximate = _expected_composed_e(
        frequency_hz=control_frequency_hz, offset_rad=small_angle_rad
    )

    assert _max_abs_difference(observed, exact) <= ATOL
    assert _max_abs_difference(observed, approximate) >= SMALL_ANGLE_FACTOR * ATOL


# ==============================================================================
# Section 4.2.1: displacement geometry
# ==============================================================================


def test_the_two_displaced_samples_coincide_at_the_resolved_boresight(
    tmp_path: Path,
) -> None:
    """Section 4.2.1: "the midpoint of the two displaced centres is the resolved
    boresight".

    Both feeds are then the same angle ``delta`` from their own centre, so a
    symmetric beam gives ``b_+ == b_-`` there and the composed
    ``E = C^dagger (b I2) C = b I2`` is scalar again -- at that one direction
    and nowhere else.
    """
    system, _instrument, _receptors, _state = _beam_system(
        tmp_path, _default_squint_beams()
    )
    boresight_altitude = np.array([np.pi / 2.0], dtype=np.float64)
    boresight_azimuth = np.array([0.0], dtype=np.float64)

    observed = _evaluate(
        system,
        ANT0,
        frequency_hz=REFERENCE_FREQUENCY_HZ,
        altitude_rad=boresight_altitude,
        azimuth_rad=boresight_azimuth,
    )[0]

    offset_rad = _exact_offset_rad(REFERENCE_FREQUENCY_HZ)
    scalar = complex(
        _airy_voltage(np.array([offset_rad]), frequency_hz=REFERENCE_FREQUENCY_HZ)[0]
    )
    assert abs(observed[0, 0] - observed[1, 1]) <= ATOL
    assert abs(observed[0, 1]) <= ATOL
    assert abs(observed[1, 0]) <= ATOL
    assert abs(observed[0, 0] - scalar) <= ATOL
    # And the midpoint invariant is not vacuous: away from the boresight the
    # same antenna's response is genuinely non-scalar.
    off_axis = _evaluate(system, ANT0, frequency_hz=REFERENCE_FREQUENCY_HZ)
    assert float(np.max(np.abs(off_axis[:, 0, 0] - off_axis[:, 1, 1]))) > 0.0


def test_the_total_feed_to_feed_separation_is_twice_the_resolved_offset(
    tmp_path: Path,
) -> None:
    """Section 4.1: "their total feed-to-feed separation is ``2*delta``".

    The observable is a pair of response maxima. With a linear receptor at zero
    feed rotation ``C = M(linear) = P`` and ``E = P diag(b_x, b_y) P
    = diag(b_y, b_x)``, so the composed matrix reports the two native samples
    directly. Each feed's sample reaches the on-axis peak exactly at its own
    displaced centre, and each reads ``e(2 delta)`` at the *other* centre --
    which is the separation, measured through the beam rather than asserted of
    the geometry alone.
    """
    system, _instrument, _receptors, _state = _beam_system(
        tmp_path, _default_squint_beams()
    )

    frequency_hz = REFERENCE_FREQUENCY_HZ
    offset_rad = _exact_offset_rad(frequency_hz)
    beta_feed = _feed_position_angle(
        mechanical_deg=MECHANICAL_ANGLE_DEG,
        mount_type=None,
        parallactic_rad=0.0,
        altitude_rad=math.pi / 2.0,
    )
    beta_squint = _wrap_to_pi(beta_feed + math.pi / 2.0)
    axis = _squint_axis(beta_squint)

    centres = np.stack(
        [
            _rodrigues(_BEAM_FRAME_ZENITH, axis, +offset_rad),
            _rodrigues(_BEAM_FRAME_ZENITH, axis, -offset_rad),
        ]
    )
    # Pure geometry first: the two centres straddle the boresight at ``2 delta``.
    separation_rad = float(
        np.arccos(np.clip(float(centres[0] @ centres[1]), -1.0, 1.0))
    )
    assert abs(separation_rad - 2.0 * offset_rad) <= ATOL
    midpoint = centres[0] + centres[1]
    midpoint = midpoint / np.linalg.norm(midpoint)
    assert float(np.max(np.abs(midpoint - _BEAM_FRAME_ZENITH))) <= ATOL

    altitude = np.arctan2(centres[:, 2], np.hypot(centres[:, 0], centres[:, 1]))
    azimuth = np.arctan2(centres[:, 0], centres[:, 1])
    observed = _evaluate(
        system,
        ANT0,
        frequency_hz=frequency_hz,
        altitude_rad=altitude,
        azimuth_rad=azimuth,
    )
    # ``positive_native_feed`` is ``x``, so ``b_x`` sits at ``E[1, 1]`` and
    # ``b_y`` at ``E[0, 0]``.
    b_x = observed[:, 1, 1]
    b_y = observed[:, 0, 0]
    peak = complex(_airy_voltage(np.array([0.0]), frequency_hz=frequency_hz)[0])
    far = complex(
        _airy_voltage(np.array([2.0 * offset_rad]), frequency_hz=frequency_hz)[0]
    )
    assert abs(peak - far) > SEPARATION_BOUND
    assert abs(b_x[0] - peak) <= ATOL
    assert abs(b_y[1] - peak) <= ATOL
    assert abs(b_x[1] - far) <= ATOL
    assert abs(b_y[0] - far) <= ATOL


def test_the_squint_direction_is_orthogonal_to_the_feed_ray_with_plus_half_pi(
    tmp_path: Path,
) -> None:
    """Section 4.1: "The squint direction is orthogonal to the
    optical-axis/feed plane, not along the feed-location ray", with the v1
    handedness ``u_squint,+ = u_feed(beta + pi/2)``.

    Three displaced centres are built: the declared ``+pi/2`` one, the
    feed-ray one (``beta_feed`` itself), and the reversed ``-pi/2`` one. Only
    the first is where the positive feed peaks; the ``-pi/2`` centre is where
    the *negative* feed peaks, which is precisely what fixes the handedness.
    """
    system, _instrument, _receptors, _state = _beam_system(
        tmp_path, _default_squint_beams()
    )
    frequency_hz = REFERENCE_FREQUENCY_HZ
    offset_rad = _exact_offset_rad(frequency_hz)
    beta_feed = _feed_position_angle(
        mechanical_deg=MECHANICAL_ANGLE_DEG,
        mount_type=None,
        parallactic_rad=0.0,
        altitude_rad=math.pi / 2.0,
    )
    beta_squint = _wrap_to_pi(beta_feed + math.pi / 2.0)
    beta_reversed = _wrap_to_pi(beta_feed - math.pi / 2.0)

    # Orthogonality is exact by construction and is asserted as such.
    assert abs(float(_squint_unit(beta_squint) @ _squint_unit(beta_feed))) <= ATOL
    assert abs(float(_squint_unit(beta_reversed) @ _squint_unit(beta_feed))) <= ATOL

    centres = np.stack(
        [
            _rodrigues(_BEAM_FRAME_ZENITH, _squint_axis(beta), +offset_rad)
            for beta in (beta_squint, beta_feed, beta_reversed)
        ]
    )
    altitude = np.arctan2(centres[:, 2], np.hypot(centres[:, 0], centres[:, 1]))
    azimuth = np.arctan2(centres[:, 0], centres[:, 1])
    observed = _evaluate(
        system,
        ANT0,
        frequency_hz=frequency_hz,
        altitude_rad=altitude,
        azimuth_rad=azimuth,
    )
    b_x = observed[:, 1, 1]
    b_y = observed[:, 0, 0]
    peak = complex(_airy_voltage(np.array([0.0]), frequency_hz=frequency_hz)[0])

    # The declared ``+pi/2`` centre is the positive feed's peak.
    assert abs(b_x[0] - peak) <= ATOL
    # The feed-ray direction is not: displacing along the feed ray would be a
    # different beam entirely.
    assert abs(b_x[1] - peak) >= SEPARATION_BOUND
    # And the reversed handedness is where the *negative* feed peaks.
    assert abs(b_x[2] - peak) >= SEPARATION_BOUND
    assert abs(b_y[2] - peak) <= ATOL


@pytest.mark.parametrize("mount_type", sorted(MOUNT_FACTORS))
def test_the_mount_field_rotation_law_holds_for_every_accepted_mount_literal(
    tmp_path: Path,
    mount_type: str,
) -> None:
    """Section 4.1/4.2.1: ``beta_feed = wrap(beta_mech + eta psi + nu alt)``.

    The boresight parallactic angle and altitude are supplied through the
    Section 4.2.1 keywords, so this pins the runtime's own use of the frozen
    formula; the adapter's derivation of those two numbers is covered by the
    exactly-zenith rejection above and by the integration cases. A non-zenith
    boresight altitude is used throughout because Section 4.2.1 rules the
    exact-``pi/2`` case undefined for a rotating mount.
    """
    parallactic_rad = 0.7
    boresight_altitude_rad = 1.1
    system, _instrument, _receptors, state = _beam_system(
        tmp_path,
        _default_squint_beams(),
        mount_types=mount_type,
    )

    record = _resolved_squint_record(state, ANT0)
    assert record.mount_type == mount_type

    observed = _evaluate(
        system,
        ANT0,
        frequency_hz=REFERENCE_FREQUENCY_HZ,
        boresight={
            "boresight_parallactic_rad": parallactic_rad,
            "boresight_altitude_rad": boresight_altitude_rad,
        },
    )
    expected = _expected_composed_e(
        mount_type=mount_type,
        parallactic_rad=parallactic_rad,
        boresight_altitude_rad=boresight_altitude_rad,
        frequency_hz=REFERENCE_FREQUENCY_HZ,
    )
    assert _max_abs_difference(observed, expected) <= ATOL

    # ``fixed`` and ``equatorial`` have ``(eta, nu) == (0, 0)``, so their
    # feed ray is exactly the mechanical angle and nothing the boresight does
    # can move it.
    static = _expected_composed_e(
        mount_type="fixed",
        parallactic_rad=0.0,
        boresight_altitude_rad=math.pi / 2.0,
        frequency_hz=REFERENCE_FREQUENCY_HZ,
    )
    if MOUNT_FACTORS[mount_type] == (0.0, 0.0):
        assert _max_abs_difference(observed, static) <= ATOL
    else:
        assert _max_abs_difference(observed, static) >= SEPARATION_BOUND


@pytest.mark.parametrize(
    "mount_type", ["alt-az", "alt-az+nasmyth-r", "alt-az+nasmyth-l"]
)
def test_the_opposite_mount_field_rotation_sign_is_a_measurably_different_beam(
    tmp_path: Path,
    mount_type: str,
) -> None:
    """Section 4.1: "The sign is the same accepted field-rotation sign used by
    ``P``; red tests include the opposite-sign control."

    Flipping ``(eta_p, nu_p)`` moves the feed ray by ``2 (eta psi + nu alt)``,
    which rotates both displaced centres about the boresight and changes the
    composed ``E`` by far more than the comparison tolerance. A production
    implementation that had inherited the opposite convention would match the
    flipped oracle instead.
    """
    parallactic_rad = 0.7
    boresight_altitude_rad = 1.1
    eta, nu = MOUNT_FACTORS[mount_type]
    system, _instrument, _receptors, _state = _beam_system(
        tmp_path,
        _default_squint_beams(),
        mount_types=mount_type,
    )

    observed = _evaluate(
        system,
        ANT0,
        frequency_hz=REFERENCE_FREQUENCY_HZ,
        boresight={
            "boresight_parallactic_rad": parallactic_rad,
            "boresight_altitude_rad": boresight_altitude_rad,
        },
    )
    accepted_beta = _wrap_to_pi(
        math.radians(MECHANICAL_ANGLE_DEG)
        + eta * parallactic_rad
        + nu * boresight_altitude_rad
    )
    flipped_beta = _wrap_to_pi(
        math.radians(MECHANICAL_ANGLE_DEG)
        - eta * parallactic_rad
        - nu * boresight_altitude_rad
    )
    assert abs(_wrap_to_pi(accepted_beta - flipped_beta)) > 0.0

    def _oracle(beta_feed_rad: float) -> np.ndarray:
        diagonal = _expected_native_diagonal(
            basis="linear",
            positive_native_feed="x",
            beta_squint_rad=_wrap_to_pi(beta_feed_rad + math.pi / 2.0),
            offset_rad=_exact_offset_rad(REFERENCE_FREQUENCY_HZ),
            altitude_rad=PROBE_ALTITUDE_RAD,
            azimuth_rad=PROBE_AZIMUTH_RAD,
            frequency_hz=REFERENCE_FREQUENCY_HZ,
        )
        receptor = _plan_receptor_matrix("linear", 0.0, np.complex128)
        return receptor.conj().T @ diagonal @ receptor

    assert _max_abs_difference(observed, _oracle(accepted_beta)) <= ATOL
    assert _max_abs_difference(observed, _oracle(flipped_beta)) >= SEPARATION_BOUND


def test_shifting_the_mechanical_angle_rotates_the_pattern_by_exactly_that_angle(
    tmp_path: Path,
) -> None:
    """Section 4.1: the mechanical position angle "describes the physical
    off-axis feed location", measured North through East.

    Shifting it by ``Delta`` must rotate the whole squinted pattern about the
    boresight by exactly ``Delta``, so a probe rotated by ``Delta`` in azimuth
    reads what the unshifted antenna reads at the unrotated probe. This is a
    production-against-production statement: no oracle profile enters it.
    """
    shift_deg = 23.0
    shift_rad = math.radians(shift_deg)
    base, _i0, _r0, _s0 = _beam_system(tmp_path / "base", _default_squint_beams())
    shifted, _i1, _r1, _s1 = _beam_system(
        tmp_path / "shifted",
        _default_squint_beams(
            mechanical_feed_position_angle_deg=MECHANICAL_ANGLE_DEG + shift_deg
        ),
    )

    observed_base = _evaluate(base, ANT0, frequency_hz=REFERENCE_FREQUENCY_HZ)
    observed_shifted = _evaluate(
        shifted,
        ANT0,
        frequency_hz=REFERENCE_FREQUENCY_HZ,
        altitude_rad=PROBE_ALTITUDE_RAD,
        azimuth_rad=PROBE_AZIMUTH_RAD + shift_rad,
    )
    assert _max_abs_difference(observed_shifted, observed_base) <= ATOL

    # Not vacuous: the same shifted antenna probed at the *unrotated*
    # directions is a different beam.
    unrotated = _evaluate(shifted, ANT0, frequency_hz=REFERENCE_FREQUENCY_HZ)
    assert _max_abs_difference(unrotated, observed_base) >= SEPARATION_BOUND


def test_swapping_the_positive_native_feed_swaps_the_displaced_centres(
    tmp_path: Path,
) -> None:
    """Section 4.1: "Swapping the positive feed reverses the Stokes-V leakage
    oracle", because it swaps which native feed carries ``+delta``.

    With ``C = M(linear)`` the composed matrix is ``diag(b_y, b_x)``, so a
    label swap must exchange the two diagonal entries exactly and leave the
    off-diagonals at zero.
    """
    positive_x, _i0, _r0, _s0 = _beam_system(
        tmp_path / "x", _default_squint_beams(positive_native_feed="x")
    )
    positive_y, _i1, _r1, _s1 = _beam_system(
        tmp_path / "y", _default_squint_beams(positive_native_feed="y")
    )

    on_x = _evaluate(positive_x, ANT0, frequency_hz=REFERENCE_FREQUENCY_HZ)
    on_y = _evaluate(positive_y, ANT0, frequency_hz=REFERENCE_FREQUENCY_HZ)

    assert _max_abs_difference(on_x[:, 0, 0], on_y[:, 1, 1]) <= ATOL
    assert _max_abs_difference(on_x[:, 1, 1], on_y[:, 0, 0]) <= ATOL
    assert float(np.max(np.abs(on_x[:, 0, 1]))) <= ATOL
    assert float(np.max(np.abs(on_y[:, 0, 1]))) <= ATOL
    # Not vacuous: the two diagonals really are different numbers.
    assert float(np.max(np.abs(on_x[:, 0, 0] - on_x[:, 1, 1]))) >= SEPARATION_BOUND


# ==============================================================================
# Section 4.2: factorization into the canonical chain
# ==============================================================================

#: Section 4.2 requires "a nontrivial unitary ``C``" and "a nontrivial ``P``";
#: these are the two fixtures used for it.
FACTORIZATION_CASES = [
    pytest.param("circular", 0.0, "r", id="circular"),
    pytest.param("linear", 31.0, "x", id="rotated_linear"),
]


@pytest.mark.parametrize(("basis", "feed_rotation_deg", "feed"), FACTORIZATION_CASES)
def test_the_composed_e_is_the_receptor_conjugated_native_diagonal(
    tmp_path: Path,
    basis: str,
    feed_rotation_deg: float,
    feed: str,
) -> None:
    """Section 4.2: ``E = C^dagger D_b C``, against an independent composition.

    The oracle rebuilds ``C`` from Section 4.2.1's frozen ``M(basis)`` and
    ``R(chi)`` in the test body. Section 4.2 also retires the old claim that
    squint merely makes ``E`` diagonal: "``E`` is generally full in RadioSim's
    sky-side space, including for a rotated linear receptor and for a circular
    receptor", which is asserted here rather than assumed.
    """
    system, _instrument, receptor_set, _state = _beam_system(
        tmp_path,
        _default_squint_beams(positive_native_feed=feed),
        receptors={
            "default": {"basis": basis, "feed_rotation_deg": feed_rotation_deg},
            "output_basis": basis,
        },
    )
    resolved_receptor = receptor_set.receptor_by_antenna[ANT0]
    assert resolved_receptor.basis == basis
    assert resolved_receptor.feed_array == NATIVE_FEED_ORDER[basis]

    observed = _evaluate(system, ANT0, frequency_hz=REFERENCE_FREQUENCY_HZ)
    expected = _expected_composed_e(
        basis=basis,
        chi_rad=math.radians(feed_rotation_deg),
        positive_native_feed=feed,
        frequency_hz=REFERENCE_FREQUENCY_HZ,
    )
    assert _max_abs_difference(observed, expected) <= ATOL

    # Section 4.2: the sky-side ``E`` is genuinely full here.
    assert float(np.max(np.abs(observed[:, 0, 1]))) >= SEPARATION_BOUND
    assert float(np.max(np.abs(observed[:, 1, 0]))) >= SEPARATION_BOUND


@pytest.mark.parametrize(("basis", "feed_rotation_deg", "feed"), FACTORIZATION_CASES)
def test_c_times_e_times_p_equals_the_physical_d_b_c_p_and_order_matters(
    tmp_path: Path,
    basis: str,
    feed_rotation_deg: float,
    feed: str,
) -> None:
    """Section 4.2: "prove ``C E P`` equals ``D_b C P`` and differs from
    ``C P E``".

    A single far probe is used so that ``b_0`` and ``b_1`` are strongly
    unequal: at ``0.09 rad`` from the boresight at 150 MHz the two samples are
    about ``0.82`` and ``0.31``, which is what makes the order control a
    statement about ordering rather than about a near-degenerate ``D_b``.
    """
    system, _instrument, _receptors, _state = _beam_system(
        tmp_path,
        _default_squint_beams(positive_native_feed=feed),
        receptors={
            "default": {"basis": basis, "feed_rotation_deg": feed_rotation_deg},
            "output_basis": basis,
        },
    )
    frequency_hz = REFERENCE_FREQUENCY_HZ
    offset_rad = _exact_offset_rad(frequency_hz)
    beta_feed = _feed_position_angle(
        mechanical_deg=MECHANICAL_ANGLE_DEG,
        mount_type=None,
        parallactic_rad=0.0,
        altitude_rad=math.pi / 2.0,
    )
    beta_squint = _wrap_to_pi(beta_feed + math.pi / 2.0)

    # A probe displaced along ``+u_squint`` by 0.09 rad from the boresight.
    probe = _rodrigues(_BEAM_FRAME_ZENITH, _squint_axis(beta_squint), 0.09)
    altitude = np.array(
        [math.atan2(probe[2], math.hypot(probe[0], probe[1]))], dtype=np.float64
    )
    azimuth = np.array([math.atan2(probe[0], probe[1])], dtype=np.float64)

    observed = _evaluate(
        system,
        ANT0,
        frequency_hz=frequency_hz,
        altitude_rad=altitude,
        azimuth_rad=azimuth,
    )
    diagonal = _expected_native_diagonal(
        basis=basis,
        positive_native_feed=feed,
        beta_squint_rad=beta_squint,
        offset_rad=offset_rad,
        altitude_rad=altitude,
        azimuth_rad=azimuth,
        frequency_hz=frequency_hz,
    )
    assert abs(diagonal[0, 0, 0] - diagonal[0, 1, 1]) >= SEPARATION_BOUND

    receptor = _plan_receptor_matrix(
        basis, math.radians(feed_rotation_deg), np.complex128
    )
    # A nontrivial field rotation, written from the same plan formula ``P``
    # uses, so the chain-order statement is about a real rotation.
    field_rotation = np.array(
        [
            [math.cos(0.6), math.sin(0.6)],
            [-math.sin(0.6), math.cos(0.6)],
        ],
        dtype=np.complex128,
    )

    physical = diagonal @ receptor @ field_rotation
    chain = receptor @ observed @ field_rotation
    swapped = receptor @ field_rotation @ observed

    assert _max_abs_difference(chain, physical) <= ATOL
    assert _max_abs_difference(chain, swapped) >= SEPARATION_BOUND


@pytest.mark.skipif(
    not COMPLEX256_AVAILABLE,
    reason=(
        "Section 8.1's extended-width predicate: this NumPy runtime exposes no "
        "distinct 32-byte clongdouble"
    ),
)
def test_extended_precision_factorization_never_narrows_to_complex128(
    tmp_path: Path,
) -> None:
    """Section 4.2.1: "``b_+``, ``b_-``, ``C``, and the composition are
    evaluated at the resolved beam dtype and never pass through a narrower
    width when the resolved dtype is wider than ``complex128``".

    Section 8.1's Stage-2 envelope requires exactly one ``complex256``
    factorization row, whose "production composition and independent oracle
    never pass through ``complex128``".
    """
    system, _instrument, _receptors, _state = _beam_system(
        tmp_path,
        _default_squint_beams(positive_native_feed="r"),
        receptors={"default": {"basis": "circular"}, "output_basis": "circular"},
        beam_precision="float128",
    )

    observed = _evaluate(system, ANT0, frequency_hz=REFERENCE_FREQUENCY_HZ)
    assert observed.dtype == np.dtype(np.complex256)

    expected = _expected_composed_e(
        basis="circular",
        positive_native_feed="r",
        frequency_hz=REFERENCE_FREQUENCY_HZ,
        dtype=np.complex256,
    )
    assert expected.dtype == np.dtype(np.complex256)
    assert _max_abs_difference(observed, expected) <= ATOL

    # A composition that had passed through ``complex128`` would be exactly
    # representable there; the extended-width one must not be.
    narrowed = observed.astype(np.complex128).astype(np.complex256)
    assert float(np.max(np.abs(observed - narrowed))) > 0.0


# ==============================================================================
# Section 4.1/4.3: the first-order Stokes-V leakage sign
# ==============================================================================


@pytest.mark.parametrize(
    ("positive_native_feed", "expected_sign"), [("r", 1), ("l", -1)]
)
def test_the_first_order_stokes_v_leakage_sign_follows_the_positive_native_feed(
    tmp_path: Path,
    positive_native_feed: str,
    expected_sign: int,
) -> None:
    """Section 4.1: "Swapping the positive feed reverses the Stokes-V leakage
    oracle."

    Sign derivation, written out because the whole test is the sign. RadioSim's
    coherency convention is ``B = (1/2) [[I+Q, U+iV], [U-iV, I-Q]]`` in the
    sky basis, so in the circular receptor's own reported basis the correlation
    matrix is ``[[RR, RL], [LR, LL]]`` with ``RR = (I+V)/2`` and
    ``LL = (I-V)/2``. The physical local response is ``D_b C``, and for an
    unpolarized source ``B = (I/2) I2`` with unitary ``C`` this gives
    ``(D_b C) B (D_b C)^dagger = (I/2) diag(|b_r|^2, |b_l|^2)``. Hence

        V / I = (|b_r|^2 - |b_l|^2) / (|b_r|^2 + |b_l|^2),

    which is positive at a probe on the ``+u_squint`` side when ``r`` carries
    the positive displacement, and negative when ``l`` does. ``output_basis``
    is ``circular``, so the reporting transform ``H`` is the identity and the
    matrix below is the reported visibility of a zero-length baseline between
    two identical antennas.
    """
    system, _instrument, _receptors, _state = _beam_system(
        tmp_path,
        _default_squint_beams(positive_native_feed=positive_native_feed),
        receptors={"default": {"basis": "circular"}, "output_basis": "circular"},
    )
    frequency_hz = REFERENCE_FREQUENCY_HZ
    beta_feed = _feed_position_angle(
        mechanical_deg=MECHANICAL_ANGLE_DEG,
        mount_type=None,
        parallactic_rad=0.0,
        altitude_rad=math.pi / 2.0,
    )
    beta_squint = _wrap_to_pi(beta_feed + math.pi / 2.0)

    # A probe on the positive-squint side of the boresight, inside the main lobe.
    probe = _rodrigues(_BEAM_FRAME_ZENITH, _squint_axis(beta_squint), 0.05)
    altitude = np.array(
        [math.atan2(probe[2], math.hypot(probe[0], probe[1]))], dtype=np.float64
    )
    azimuth = np.array([math.atan2(probe[0], probe[1])], dtype=np.float64)

    composed_e = _evaluate(
        system,
        ANT0,
        frequency_hz=frequency_hz,
        altitude_rad=altitude,
        azimuth_rad=azimuth,
    )[0]
    receptor = _plan_receptor_matrix("circular", 0.0, np.complex128)
    coherency = 0.5 * np.eye(2, dtype=np.complex128)
    local = receptor @ composed_e
    reported = local @ coherency @ local.conj().T

    stokes_i = float((reported[0, 0] + reported[1, 1]).real)
    stokes_v = float((reported[0, 0] - reported[1, 1]).real)
    assert stokes_i > 0.0
    v_over_i = stokes_v / stokes_i
    assert abs(v_over_i) >= SEPARATION_BOUND
    assert (1 if v_over_i > 0.0 else -1) == expected_sign


# ==============================================================================
# Section 4.2.1: the widened per-antenna response identity
# ==============================================================================


def test_an_antenna_without_squint_keeps_todays_response_key(
    tmp_path: Path,
) -> None:
    """GREEN CONTROL. Section 4.2.1: "An antenna without squint produces a
    byte-identical key to today."

    Pinned as the equality form the accepted docstring already states -- with
    no pointing offset, no surface error and no squint the response key *is*
    the handler id -- rather than as a digest literal, which would only pin
    today's fixture.
    """
    system, _instrument, _receptors, _state = _beam_system(tmp_path, _analytic_beams())

    handler_ids = dict(system.state.assignment_handler_ids)
    for antenna in (ANT0, ANT1):
        assert system.response_key(antenna) == handler_ids[antenna]
    # Both fixture antennas are the same 14 m dish, so they share the handler.
    assert handler_ids[ANT0] == handler_ids[ANT1]
    assert system.response_key(ANT0) == system.response_key(ANT1)


def test_a_squint_antenna_has_a_different_response_key_than_without_squint(
    tmp_path: Path,
) -> None:
    """Section 4.2.1: "The per-antenna response identity widens exactly when
    squint is present"."""
    plain, _i0, _r0, _s0 = _beam_system(tmp_path / "plain", _analytic_beams())
    squinted, _i1, _r1, _s1 = _beam_system(tmp_path / "squint", _default_squint_beams())

    assert squinted.response_key(ANT0) != plain.response_key(ANT0)
    # The handler is unchanged: squint is per-antenna state like pointing and
    # surface error, and "it never enters the handler preload key".
    assert (
        dict(squinted.state.assignment_handler_ids)[ANT0]
        == dict(plain.state.assignment_handler_ids)[ANT0]
    )


def test_two_antennas_sharing_a_handler_never_share_a_squinted_response(
    tmp_path: Path,
) -> None:
    """Section 4.2.1: the widened identity carries ``receptor_basis`` and
    ``feed_rotation_rad`` as well as the squint record, so "two antennas
    sharing one handler with different squint or receptor state never share a
    composed ``E``"."""
    # Same handler, same squint record, different static feed rotation.
    receptors = {
        "default": {"basis": "linear", "feed_rotation_deg": 0.0},
        "overrides": [
            {
                "antenna": {"kind": "number", "number": 1},
                "feed_rotation_deg": 31.0,
            }
        ],
        "output_basis": "linear",
    }
    system, _instrument, _receptors, _state = _beam_system(
        tmp_path / "receptor", _default_squint_beams(), receptors=receptors
    )
    handler_ids = dict(system.state.assignment_handler_ids)
    assert handler_ids[ANT0] == handler_ids[ANT1]
    assert system.response_key(ANT0) != system.response_key(ANT1)
    first = _evaluate(system, ANT0, frequency_hz=REFERENCE_FREQUENCY_HZ)
    second = _evaluate(system, ANT1, frequency_hz=REFERENCE_FREQUENCY_HZ)
    assert _max_abs_difference(first, second) >= SEPARATION_BOUND

    # Same handler, same receptor state, different squint record.
    per_antenna_squint = _analytic_beams(
        squint={
            "default": _squint_record(),
            "per_antenna": [
                {
                    "antenna": {"kind": "number", "number": 1},
                    **_squint_record(mechanical_feed_position_angle_deg=-70.0),
                }
            ],
        }
    )
    mixed, _i, _r, _s = _beam_system(tmp_path / "squint", per_antenna_squint)
    mixed_handlers = dict(mixed.state.assignment_handler_ids)
    assert mixed_handlers[ANT0] == mixed_handlers[ANT1]
    assert mixed.response_key(ANT0) != mixed.response_key(ANT1)
    assert (
        _max_abs_difference(
            _evaluate(mixed, ANT0, frequency_hz=REFERENCE_FREQUENCY_HZ),
            _evaluate(mixed, ANT1, frequency_hz=REFERENCE_FREQUENCY_HZ),
        )
        >= SEPARATION_BOUND
    )


def test_the_resolved_squint_record_carries_the_frozen_six_fields(
    tmp_path: Path,
) -> None:
    """Section 4.2.1's resolved dataclass: "exactly the fields ... plus
    ``mount_type`` holding one of the five accepted mount literals".

    The assignment fingerprint must move with it -- Section 4.2.1 puts squint
    into ``_assignment_fingerprint`` "only when present" -- and must be
    untouched when the block is absent.
    """
    _system, _instrument, _receptors, state = _beam_system(
        tmp_path / "squint",
        _default_squint_beams(),
        mount_types="equatorial",
    )
    record = _resolved_squint_record(state, ANT0)

    assert record.convention == SQUINT_CONVENTION
    assert record.reference_frequency_hz == REFERENCE_FREQUENCY_HZ
    assert record.per_feed_offset_deg_at_reference == PER_FEED_OFFSET_DEG
    assert record.mechanical_feed_position_angle_deg == MECHANICAL_ANGLE_DEG
    assert record.positive_native_feed == "x"
    assert record.mount_type == "equatorial"
    for value in (
        record.reference_frequency_hz,
        record.per_feed_offset_deg_at_reference,
        record.mechanical_feed_position_angle_deg,
    ):
        assert type(value) is float

    _plain_system, _pi, _pr, plain_state = _beam_system(
        tmp_path / "plain", _analytic_beams()
    )
    plain_assignment = plain_state.assignments[0]
    squint_assignment = state.assignments[0]
    assert plain_assignment.squint is None
    assert (
        squint_assignment.assignment_fingerprint
        != plain_assignment.assignment_fingerprint
    )

    # A different mechanical angle is a different resolved record and therefore
    # a different fingerprint: the six field values all enter the payload.
    _other, _oi, _or_, other_state = _beam_system(
        tmp_path / "other",
        _default_squint_beams(mechanical_feed_position_angle_deg=-70.0),
        mount_types="equatorial",
    )
    assert (
        other_state.assignments[0].assignment_fingerprint
        != squint_assignment.assignment_fingerprint
    )
