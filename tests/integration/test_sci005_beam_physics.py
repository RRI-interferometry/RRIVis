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
