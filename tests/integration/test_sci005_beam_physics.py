"""SCI-005 Stage-1 effect-through-``Simulator`` integration.

``docs/development/sci005_beam_physics_plan.md`` Section 2 makes two symmetric
demands of every stage, and Section 3.5 narrows them for Stage 1:

* an explicitly enabled effect must reach the visibilities -- "blockage and
  Zernike each change a visibility when enabled" -- through the whole path:
  strict parse, resolution, the beam runtime, the ``E`` factor, the
  contraction, and the retained fingerprint; and
* "No absent or disabled beam block changes the resolved configuration,
  assignment/state/scientific fingerprints, result bytes, logs, or output", and
  fingerprints change "only for a workload that explicitly enables the landed
  effect".

The Ruze diagnostic is deliberately on the other side of that line. Section
3.4.1: "the existing coherent Ruze term must still change cross-baseline
visibility exactly, while the new error-beam diagnostic must change the
retained ensemble-power record and satisfy power balance. A test requiring that
diagnostic power to change a cross-baseline visibility is itself a design
violation." Both halves are asserted below.

Point and HEALPix routes are both covered (Section 3.5). Workloads are the
shipped two-antenna fixtures; the point of an integration case here is the
path, not the size.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from radiosim.api.simulator import Simulator
from radiosim.io.config_resolution import UnsupportedConfigError
from radiosim.io.result_format import ResultFormat
from tests.fixtures.configs import hybrid_config_mapping, valid_config_mapping

_SPEED_OF_LIGHT_M_PER_S = 299792458.0

#: Section 3.3's frozen comparison tolerances at float64 width.
_EPS = float(np.finfo(np.float64).eps)
ATOL = max(1e-12, 32.0 * _EPS)
RTOL = max(1e-10, 32.0 * _EPS)

APERTURE_NORMALIZATION = "unmodified_ideal_aperture_v1"
ZERNIKE_CONVENTION = "radiosim.real_unit_rms_disk_surface_height.v1"

#: A Stage-1 supported pupil: Section 3.1's ``U(rho)`` row.
UNIFORM_CIRCULAR: dict[str, Any] = {
    "kind": "circular_aperture",
    "taper": {"kind": "uniform"},
}

#: One channel, so a per-frequency scaling law is a scalar in the comparison.
SINGLE_CHANNEL: dict[str, Any] = {
    "mode": "explicit",
    "channel_frequencies_hz": [1.0e8],
    "channel_widths_hz": [1.0e6],
}

BLOCKAGE: dict[str, Any] = {
    "normalization": APERTURE_NORMALIZATION,
    "blockage": {
        "central_diameter_ratio": 0.2,
        "support_legs": [{"position_angle_deg": 0.0, "width_m": 0.8}],
    },
}
ZERNIKE: dict[str, Any] = {
    "normalization": APERTURE_NORMALIZATION,
    "zernike_surface": {
        "convention": ZERNIKE_CONVENTION,
        "modes": [{"n": 2, "m": 0, "surface_height_coefficient_m": 0.05}],
    },
}


def _beams(
    *,
    aperture_physics: dict[str, Any] | None = None,
    surface_error: dict[str, Any] | None = None,
) -> dict[str, Any]:
    beams: dict[str, Any] = {"mode": "analytic", "model": dict(UNIFORM_CIRCULAR)}
    if aperture_physics is not None:
        beams["aperture_physics"] = aperture_physics
    if surface_error is not None:
        beams["surface_error"] = surface_error
    return beams


def _run(
    tmp_path: Path,
    beams: dict[str, Any],
    *,
    route: str = "point",
    single_channel: bool = False,
) -> tuple[Simulator, Any]:
    """Run one tiny workload through the public entry point."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    overrides: dict[str, Any] = {"beams": beams}
    if single_channel:
        overrides["frequency"] = dict(SINGLE_CHANNEL)
    if route == "point":
        data = valid_config_mapping(tmp_path, **overrides)
    else:
        data = hybrid_config_mapping(tmp_path, component="healpix", **overrides)
    simulator = Simulator.from_mapping(data, base_dir=tmp_path)
    return simulator, simulator.run(progress=False)


@pytest.mark.integration
@pytest.mark.parametrize("route", ["point", "healpix"])
@pytest.mark.parametrize(
    ("label", "aperture_physics"),
    [("blockage", BLOCKAGE), ("zernike", ZERNIKE)],
)
def test_an_enabled_aperture_block_changes_the_visibilities(
    tmp_path: Path,
    route: str,
    label: str,
    aperture_physics: dict[str, Any],
) -> None:
    """Section 3.5: "blockage and Zernike each change a visibility when
    enabled", on both the point and HEALPix routes."""
    _, control = _run(tmp_path / f"{route}-{label}-off", _beams(), route=route)
    _, enabled = _run(
        tmp_path / f"{route}-{label}-on",
        _beams(aperture_physics=aperture_physics),
        route=route,
    )

    assert np.all(np.isfinite(enabled.visibilities))
    assert float(np.max(np.abs(enabled.visibilities))) > 0.0
    assert enabled.visibilities.shape == control.visibilities.shape
    changed = int(np.count_nonzero(enabled.visibilities != control.visibilities))
    assert changed > 0
    assert enabled.scientific_sha256 != control.scientific_sha256


@pytest.mark.integration
def test_an_enabled_block_survives_the_whole_save_and_reload_path(
    tmp_path: Path,
) -> None:
    """The effect must reach the retained artifact, not only the in-memory cube."""
    from radiosim.io.hdf5 import load_result_hdf5

    simulator, result = _run(tmp_path / "run", _beams(aperture_physics=BLOCKAGE))

    written = simulator.save(tmp_path / "blockage.h5", format=ResultFormat.HDF5)
    loaded = load_result_hdf5(written)

    assert loaded.scientific_sha256 == result.scientific_sha256
    np.testing.assert_array_equal(loaded.visibilities, result.visibilities)


@pytest.mark.integration
@pytest.mark.parametrize("route", ["point", "healpix"])
def test_an_absent_block_leaves_configuration_fingerprints_and_bytes_untouched(
    tmp_path: Path,
    route: str,
) -> None:
    """Section 2: no absent block changes the resolved configuration, the
    fingerprints, or the result bytes."""
    omitted_simulator, omitted = _run(
        tmp_path / f"{route}-omitted", _beams(), route=route
    )
    repeat_simulator, repeat = _run(tmp_path / f"{route}-repeat", _beams(), route=route)

    assert omitted_simulator.config.beams.aperture_physics is None
    assert omitted_simulator.config.beams.surface_error is None
    assert repeat_simulator.config.beams.aperture_physics is None
    assert repeat.scientific_sha256 == omitted.scientific_sha256
    np.testing.assert_array_equal(repeat.visibilities, omitted.visibilities)
    assert repeat.visibilities.dtype == omitted.visibilities.dtype


@pytest.mark.integration
def test_the_coherent_ruze_term_still_scales_cross_baseline_visibilities(
    tmp_path: Path,
) -> None:
    """Section 3.4: the accepted coherent-voltage meaning is unchanged, so a
    shared surface RMS scales every visibility by exactly ``eta_s``."""
    rms_surface_error_m = 0.02
    control_simulator, control = _run(
        tmp_path / "ruze-off", _beams(), single_channel=True
    )
    _, scaled = _run(
        tmp_path / "ruze-on",
        _beams(surface_error={"default": {"rms_surface_error_m": rms_surface_error_m}}),
        single_channel=True,
    )

    frequencies = control_simulator.config.frequency.channel_frequencies_hz
    assert len(frequencies) == 1
    wavelength_m = _SPEED_OF_LIGHT_M_PER_S / float(frequencies[0])
    efficiency = float(
        np.exp(-((4.0 * np.pi * rms_surface_error_m / wavelength_m) ** 2))
    )
    assert 0.0 < efficiency < 1.0

    expected = np.asarray(control.visibilities) * efficiency
    residual = float(np.max(np.abs(np.asarray(scaled.visibilities) - expected)))
    assert residual <= ATOL + RTOL * float(np.max(np.abs(expected)))
    assert float(np.max(np.abs(np.asarray(control.visibilities)))) > 0.0


@pytest.mark.integration
def test_the_power_diagnostic_changes_the_fingerprint_but_no_visibility(
    tmp_path: Path,
) -> None:
    """Section 3.4.1: the diagnostic is an ensemble-power record, not a Jones
    voltage, so requiring it to move a cross-baseline visibility would itself be
    a design violation -- while configuring it does change the fingerprint."""
    surface_error = {"default": {"rms_surface_error_m": 0.02}}
    diagnostic_surface_error = {
        "default": {
            "rms_surface_error_m": 0.02,
            "error_beam_diagnostic": {
                "kind": "gaussian_covariance_power",
                "correlation_length_m": 2.0,
            },
        }
    }

    _, plain = _run(tmp_path / "plain", _beams(surface_error=surface_error))
    _, with_diagnostic = _run(
        tmp_path / "diagnostic", _beams(surface_error=diagnostic_surface_error)
    )

    np.testing.assert_array_equal(with_diagnostic.visibilities, plain.visibilities)
    assert with_diagnostic.scientific_sha256 != plain.scientific_sha256


@pytest.mark.integration
def test_a_rejected_aperture_block_stops_the_run_before_publishing(
    tmp_path: Path,
) -> None:
    """Section 3.1's exclusions reach the public entry point, and a rejected
    document publishes nothing."""
    output_dir = tmp_path / "output"
    data = valid_config_mapping(
        tmp_path,
        beams={
            "mode": "analytic",
            "model": {"kind": "circular_aperture", "taper": {"kind": "gaussian"}},
            "aperture_physics": BLOCKAGE,
        },
    )

    with pytest.raises(UnsupportedConfigError) as error:
        Simulator.from_mapping(data, base_dir=tmp_path)

    assert [issue.path for issue in error.value.issues] == ["beams.aperture_physics"]
    assert [issue.code for issue in error.value.issues] == [
        "beam.aperture_physics.unsupported_pupil_profile"
    ]
    assert not list(output_dir.glob("**/*.h5"))


# ==============================================================================
# SCI-005 Stage 2: beam squint through ``Simulator``
# ==============================================================================
#
# Sections 4.1 and 4.2.1 make squint an ordinary member of the same two
# symmetric demands Stage 1 answered above: an enabled block must reach the
# visibilities on both the point and the HEALPix route and move the retained
# scientific hash, and an absent block must change nothing at all. Section 4.3
# adds one composition case that only an integration run can see -- "the Ruze
# voltage factor applied once to the composed ``E``, with squint composing
# correctly with the Stage-1 aperture-physics branch".
#
# Every squint fixture is a ``fixed``-mount array: the shipped layout carries
# no mount type, ``None`` retains its accepted ``fixed`` reading (Section 4.1),
# and Section 4.2.1 rules the rotating-mount boresight undefined at an exactly
# zenith pointing, which is what an unpointed fixture has. The rotating-mount
# rejection itself lives in ``tests/unit/test_core/test_sci005_beam_squint.py``.

SQUINT: dict[str, Any] = {
    "default": {
        "convention": "cotton_uson_exact_v1",
        "reference_frequency_hz": 1.5e8,
        "per_feed_offset_deg_at_reference": 2.0,
        "mechanical_feed_position_angle_deg": 35.0,
        "positive_native_feed": "x",
    }
}


def _squint_beams(
    *,
    squint: dict[str, Any] | None = None,
    surface_error: dict[str, Any] | None = None,
    aperture_physics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    beams: dict[str, Any] = {"mode": "analytic", "model": dict(UNIFORM_CIRCULAR)}
    if squint is not None:
        beams["squint"] = squint
    if surface_error is not None:
        beams["surface_error"] = surface_error
    if aperture_physics is not None:
        beams["aperture_physics"] = aperture_physics
    return beams


@pytest.mark.integration
@pytest.mark.parametrize("route", ["point", "healpix"])
def test_an_enabled_squint_block_changes_the_visibilities(
    tmp_path: Path,
    route: str,
) -> None:
    """Section 4.3: "point and HEALPix paths", and Section 2's rule that
    "Fingerprints change only for a workload that explicitly enables the
    landed effect".

    Squint is the first beam effect that makes ``E`` non-diagonal in RadioSim's
    sky-side space (Section 4.2), so the change here is not a rescaling: the
    two native feeds sample the pattern at oppositely displaced directions and
    the composed ``C^dagger D_b C`` mixes them.
    """
    _, control = _run(tmp_path / f"{route}-squint-off", _squint_beams(), route=route)
    _, enabled = _run(
        tmp_path / f"{route}-squint-on",
        _squint_beams(squint=SQUINT),
        route=route,
    )

    assert np.all(np.isfinite(enabled.visibilities))
    assert float(np.max(np.abs(enabled.visibilities))) > 0.0
    assert enabled.visibilities.shape == control.visibilities.shape
    changed = int(np.count_nonzero(enabled.visibilities != control.visibilities))
    assert changed > 0
    assert enabled.scientific_sha256 != control.scientific_sha256


@pytest.mark.integration
@pytest.mark.parametrize("route", ["point", "healpix"])
def test_an_absent_squint_block_leaves_fingerprints_and_bytes_untouched(
    tmp_path: Path,
    route: str,
) -> None:
    """GREEN CONTROL. Section 2: "No absent or disabled beam block changes the
    resolved configuration, assignment/state/scientific fingerprints, result
    bytes, logs, or output", and Section 4.3's "scalar disabled/default byte
    identity".

    The equality form is pinned rather than a digest literal, so this control
    is meaningful both before and after Stage 2 lands.
    """
    omitted_simulator, omitted = _run(
        tmp_path / f"{route}-no-squint", _squint_beams(), route=route
    )
    repeat_simulator, repeat = _run(
        tmp_path / f"{route}-no-squint-again", _squint_beams(), route=route
    )

    assert getattr(omitted_simulator.config.beams, "squint", None) is None
    assert getattr(repeat_simulator.config.beams, "squint", None) is None
    assert repeat.scientific_sha256 == omitted.scientific_sha256
    np.testing.assert_array_equal(repeat.visibilities, omitted.visibilities)
    assert repeat.visibilities.dtype == omitted.visibilities.dtype


@pytest.mark.integration
def test_an_enabled_squint_block_survives_the_whole_save_and_reload_path(
    tmp_path: Path,
) -> None:
    """The effect must reach the retained artifact, not only the in-memory cube.

    Section 8.1's Stage-2 envelope requires "at least one ``in_memory`` row and
    one ``hdf5`` row for a squint-enabled workload".
    """
    from radiosim.io.hdf5 import load_result_hdf5

    simulator, result = _run(tmp_path / "run", _squint_beams(squint=SQUINT))

    written = simulator.save(tmp_path / "squint.h5", format=ResultFormat.HDF5)
    loaded = load_result_hdf5(written)

    assert loaded.scientific_sha256 == result.scientific_sha256
    np.testing.assert_array_equal(loaded.visibilities, result.visibilities)


@pytest.mark.integration
def test_the_ruze_voltage_factor_is_applied_once_to_the_composed_squint_e(
    tmp_path: Path,
) -> None:
    """Section 4.2.1: the Ruze factor is applied "to the composed ``E`` exactly
    where the scalar path applies it today (the factor is scalar and
    commutes)".

    Applied once per composed ``E``, a shared surface RMS scales every
    visibility by exactly ``eta_v^2 == eta_s``, the same power efficiency the
    Stage-1 coherent-Ruze case above asserts. An implementation that applied
    the voltage factor to each of the two native samples ``b_+`` and ``b_-``
    before composing would scale by ``eta_s^2`` instead, which is why the
    scaling and not merely a change is asserted.
    """
    rms_surface_error_m = 0.02
    control_simulator, control = _run(
        tmp_path / "squint-ruze-off",
        _squint_beams(squint=SQUINT),
        single_channel=True,
    )
    _, scaled = _run(
        tmp_path / "squint-ruze-on",
        _squint_beams(
            squint=SQUINT,
            surface_error={"default": {"rms_surface_error_m": rms_surface_error_m}},
        ),
        single_channel=True,
    )

    frequencies = control_simulator.config.frequency.channel_frequencies_hz
    assert len(frequencies) == 1
    wavelength_m = _SPEED_OF_LIGHT_M_PER_S / float(frequencies[0])
    efficiency = float(
        np.exp(-((4.0 * np.pi * rms_surface_error_m / wavelength_m) ** 2))
    )
    assert 0.0 < efficiency < 1.0

    expected = np.asarray(control.visibilities) * efficiency
    residual = float(np.max(np.abs(np.asarray(scaled.visibilities) - expected)))
    assert residual <= ATOL + RTOL * float(np.max(np.abs(expected)))
    assert float(np.max(np.abs(np.asarray(control.visibilities)))) > 0.0
    # Not vacuous: a doubly applied factor would be a different cube.
    doubly_applied = np.asarray(control.visibilities) * efficiency * efficiency
    assert float(np.max(np.abs(doubly_applied - expected))) > residual


@pytest.mark.integration
def test_squint_composes_with_the_stage1_aperture_physics_branch(
    tmp_path: Path,
) -> None:
    """Section 4.2.1: the two displaced evaluations call "the analytic path,
    including any Stage-1 aperture-physics branch".

    Section 4.1.1 admits "every analytic model, including the Stage-1
    aperture-physics branch ... because squint only re-evaluates the existing
    scalar response at displaced directions", so the two effects must compose
    rather than exclude one another.
    """
    _, aperture_only = _run(
        tmp_path / "aperture", _squint_beams(aperture_physics=BLOCKAGE)
    )
    _, squint_only = _run(tmp_path / "squint", _squint_beams(squint=SQUINT))
    _, both = _run(
        tmp_path / "both",
        _squint_beams(squint=SQUINT, aperture_physics=BLOCKAGE),
    )

    assert np.all(np.isfinite(both.visibilities))
    assert float(np.max(np.abs(both.visibilities))) > 0.0
    assert int(np.count_nonzero(both.visibilities != aperture_only.visibilities)) > 0
    assert int(np.count_nonzero(both.visibilities != squint_only.visibilities)) > 0
    assert both.scientific_sha256 not in {
        aperture_only.scientific_sha256,
        squint_only.scientific_sha256,
    }


@pytest.mark.integration
def test_a_rejected_squint_block_stops_the_run_before_publishing(
    tmp_path: Path,
) -> None:
    """Section 4.1.1's exclusions reach the public entry point, and a rejected
    document publishes nothing."""
    output_dir = tmp_path / "output"
    data = valid_config_mapping(
        tmp_path,
        beams={
            "mode": "shared_fits",
            "beam": {"kind": "fits", "path": str(tmp_path / "beam.fits")},
            "squint": SQUINT,
        },
    )

    with pytest.raises(UnsupportedConfigError) as error:
        Simulator.from_mapping(data, base_dir=tmp_path)

    assert [issue.path for issue in error.value.issues] == ["beams.squint"]
    assert [issue.code for issue in error.value.issues] == [
        "beam.squint.unsupported_beam_family"
    ]
    assert error.value.issues[0].message == (
        "Stage-2 beam squint supports only the analytic beams mode; resolved "
        "beams mode is 'shared_fits'."
    )
    assert not list(output_dir.glob("**/*.h5"))


# ==============================================================================
# SCI-005 Stage 3: the full efield response through the public entry point
# ==============================================================================
#
# ``docs/development/sci005_beam_physics_plan.md`` Section 5.5: "Point and
# HEALPix solvers consume the same complete ``_ResolvedBeamJones`` batch ...
# NumPy and Dask must be byte-identical, and JAX must agree at the existing
# float64 tolerance after the host matrix is transferred." Section 8.1 fixes
# ``solver_cases.effect`` as exactly ``efield_point`` or ``efield_healpix``,
# each appearing at least once, with ``visibility_change_expected`` true and
# ``visibility_changed_element_count`` positive.
#
# The disabled control is the other half of the same evidence: Section 8.1's
# ``fingerprint_diff`` "disabled control must include a scalar ``peak`` FITS
# workload whose old and new digests and cube bytes are equal".

EFIELD_NORMALIZATION = "uvbeam_peak_common_v1"


def _stage3_transport(tmp_path: Path, *, feed_array: tuple[str, str] = ("x", "y")):
    """Write one full-efield BeamFITS transport shared by a comparison pair."""
    from tests.fixtures.beamfits import EfieldScienceVariant, write_efield_beamfits

    tmp_path.mkdir(parents=True, exist_ok=True)
    return write_efield_beamfits(
        tmp_path,
        science=EfieldScienceVariant.QUADRUPOLAR,
        feed_array=feed_array,
    )


def _scalar_transport(tmp_path: Path):
    """Write the accepted scalar BeamFITS transport, unchanged by this stage."""
    from tests.fixtures.beamfits import write_scalar_efield_beamfits

    tmp_path.mkdir(parents=True, exist_ok=True)
    return write_scalar_efield_beamfits(tmp_path)


def _fits_beams_block(path: Path, normalization: str) -> dict[str, Any]:
    return {
        "mode": "shared_fits",
        "beam": {
            "kind": "fits",
            "path": str(path),
            "normalization": normalization,
        },
    }


def _run_beamfits(
    tmp_path: Path,
    beams: dict[str, Any],
    *,
    route: str = "point",
    receptors: dict[str, Any] | None = None,
    backend: str | None = None,
) -> tuple[Simulator, Any]:
    """Run one tiny BeamFITS workload through the public entry point."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    overrides: dict[str, Any] = {"beams": beams}
    if receptors is not None:
        overrides["receptors"] = receptors
    if route == "point":
        data = valid_config_mapping(tmp_path, **overrides)
    else:
        data = hybrid_config_mapping(tmp_path, component="healpix", **overrides)
    if backend is not None:
        data["execution"]["backend"] = backend
    simulator = Simulator.from_mapping(data, base_dir=tmp_path)
    return simulator, simulator.run(progress=False)


@pytest.mark.integration
@pytest.mark.parametrize("route", ["point", "healpix"])
def test_a_scalar_beamfits_workload_stays_byte_identical(
    tmp_path: Path,
    route: str,
) -> None:
    """Section 5.1.1: "this gate changes no byte of the accepted ``peak``
    path", and Section 8.1's disabled control requires the scalar ``peak``
    FITS workload's digests and cube bytes to be equal.

    The scalar ``E`` is ``e I2``, so both cross-hand products are exactly zero
    -- which is precisely the property the full-efield subset must break.
    """
    written = _scalar_transport(tmp_path)
    beams = _fits_beams_block(written.path, "peak")

    first_simulator, first = _run_beamfits(
        tmp_path / f"{route}-first", beams, route=route
    )
    _second_simulator, second = _run_beamfits(
        tmp_path / f"{route}-second", beams, route=route
    )

    assert first.scientific_sha256 == second.scientific_sha256
    np.testing.assert_array_equal(first.visibilities, second.visibilities)
    assert first_simulator.config.beams.beam.normalization == "peak"
    assert first.visibilities.shape[-1] == 4
    np.testing.assert_array_equal(
        first.visibilities[..., 1], np.zeros_like(first.visibilities[..., 1])
    )
    np.testing.assert_array_equal(
        first.visibilities[..., 2], np.zeros_like(first.visibilities[..., 2])
    )


@pytest.mark.integration
@pytest.mark.parametrize("route", ["point", "healpix"])
def test_a_full_efield_beam_changes_the_visibilities_and_fills_the_cross_hands(
    tmp_path: Path,
    route: str,
) -> None:
    """Section 8.1's ``efield_point`` and ``efield_healpix`` solver cases.

    ``in_memory``'s frozen predicate (Section 5.4) is asserted alongside: the
    result carries all four correlation products in the single declared output
    basis "and both cross-hand products are non-zero somewhere in the cube --
    the last being the whole point of a non-scalar ``E``".
    """
    scalar = _scalar_transport(tmp_path / "scalar")
    efield = _stage3_transport(tmp_path / "efield")

    _control_simulator, control = _run_beamfits(
        tmp_path / f"{route}-scalar",
        _fits_beams_block(scalar.path, "peak"),
        route=route,
    )
    _enabled_simulator, enabled = _run_beamfits(
        tmp_path / f"{route}-efield",
        _fits_beams_block(efield.path, EFIELD_NORMALIZATION),
        route=route,
    )

    assert np.all(np.isfinite(enabled.visibilities))
    assert enabled.visibilities.shape == control.visibilities.shape
    changed = int(np.count_nonzero(enabled.visibilities != control.visibilities))
    assert changed > 0
    assert enabled.scientific_sha256 != control.scientific_sha256
    assert float(np.max(np.abs(enabled.visibilities[..., 1]))) > 0.0
    assert float(np.max(np.abs(enabled.visibilities[..., 2]))) > 0.0


@pytest.mark.integration
@pytest.mark.parametrize("output_basis", ["linear", "circular"])
def test_a_full_efield_result_reports_the_declared_output_basis(
    tmp_path: Path,
    output_basis: str,
) -> None:
    """Section 5.4's ``in_memory`` predicate, on the two output bases Section
    8.1 requires: "Both bases are exercised on the same underlying efield
    fixture so that the pair isolates ``H`` and nothing else"."""
    from radiosim.core.polarization_basis import PolarizationBasis

    efield = _stage3_transport(tmp_path / "efield")
    simulator, result = _run_beamfits(
        tmp_path / output_basis,
        _fits_beams_block(efield.path, EFIELD_NORMALIZATION),
        receptors={
            "default": {"basis": "linear", "feed_rotation_deg": 0.0},
            "output_basis": output_basis,
        },
    )

    expected_basis: PolarizationBasis = (
        "linear_xy" if output_basis == "linear" else "circular_rl"
    )
    assert simulator.receptors.output_basis == expected_basis
    assert result.visibilities.shape[-1] == 4
    assert float(np.max(np.abs(result.visibilities[..., 1]))) > 0.0
    assert float(np.max(np.abs(result.visibilities[..., 2]))) > 0.0


@pytest.mark.integration
def test_a_full_efield_workload_survives_the_whole_save_and_reload_path(
    tmp_path: Path,
) -> None:
    """Section 5.4's ``hdf5`` predicate: ``provenance/beam_json`` round-trips
    the complete beam snapshot, "the reconstructed ``scientific_sha256`` equals
    the written one, and the reconstructed visibility cube differs from the
    in-memory cube by exactly zero"."""
    from radiosim.io.hdf5 import load_result_hdf5

    efield = _stage3_transport(tmp_path / "efield")
    simulator, result = _run_beamfits(
        tmp_path / "run",
        _fits_beams_block(efield.path, EFIELD_NORMALIZATION),
    )

    written = simulator.save(tmp_path / "efield.h5", format=ResultFormat.HDF5)
    loaded = load_result_hdf5(written)

    assert loaded.scientific_sha256 == result.scientific_sha256
    np.testing.assert_array_equal(loaded.visibilities, result.visibilities)


@pytest.mark.integration
def test_a_full_efield_workload_agrees_across_the_three_backends(
    tmp_path: Path,
) -> None:
    """Section 5.5: "NumPy and Dask must be byte-identical, and JAX must agree
    at the existing float64 tolerance after the host matrix is transferred."

    The existing tolerance is the accepted backend-parity one recorded in
    ``CLAUDE.md`` and the backend documentation, ``rtol=1e-12``; it is not
    widened here.
    """
    pytest.importorskip("jax")
    efield = _stage3_transport(tmp_path / "efield")
    beams = _fits_beams_block(efield.path, EFIELD_NORMALIZATION)

    _numpy_simulator, numpy_result = _run_beamfits(
        tmp_path / "numpy", beams, backend="numpy"
    )
    _dask_simulator, dask_result = _run_beamfits(
        tmp_path / "dask", beams, backend="dask"
    )
    _jax_simulator, jax_result = _run_beamfits(tmp_path / "jax", beams, backend="jax")

    np.testing.assert_array_equal(dask_result.visibilities, numpy_result.visibilities)
    np.testing.assert_allclose(
        np.asarray(jax_result.visibilities),
        numpy_result.visibilities,
        rtol=1e-12,
        atol=0.0,
    )
