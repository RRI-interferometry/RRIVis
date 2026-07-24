"""Tests for compute_drift_scan_lightcurve / fractional_horizon_excess.

These tests use a synthetic SkyModel and a hand-built planner-free
lightcurve to exercise the bookkeeping (LST sweep, area normalisation,
horizon mask) without requiring a real beam FITS file on disk.
"""

import inspect
from pathlib import Path

import healpy as hp
import numpy as np
import pytest

from radiosim.api.simulator import Simulator
from radiosim.core.beam import BeamSystem
from radiosim.core.observability.lightcurves import (
    DriftScanLightcurve,
    compute_drift_scan_lightcurve,
    fractional_horizon_excess,
)
from radiosim.core.precision import PrecisionConfig
from radiosim.core.sky import HealpixData, SkyModel


@pytest.fixture
def precision() -> PrecisionConfig:
    return PrecisionConfig.standard()


def _uniform_sky(
    precision: PrecisionConfig, *, nside: int = 8, value_k: float = 5.0
) -> SkyModel:
    npix = hp.nside2npix(nside)
    return SkyModel(
        healpix=HealpixData(
            maps=np.full((1, npix), value_k, dtype=np.float64),
            nside=nside,
            frequencies=np.array([150e6]),
        ),
        model_name="uniform_sky",
        precision=precision,
    )


def _lightcurve(lst_hours: np.ndarray, integrated_flux: np.ndarray):
    from radiosim.core.instrument import AntennaId

    return DriftScanLightcurve(
        lst_hours=lst_hours,
        integrated_flux=integrated_flux,
        mean_brightness=None,
        horizon_masked=True,
        frequency_hz=150_000_000.0,
        nside=4,
        beam_evaluation_time_mjd=60_676.0,
        reference_antenna=AntennaId(0, "ANT0"),
        reference_handler_id="beam-0000-" + ("a" * 12),
        reference_scientific_fingerprint="a" * 64,
        power_convention="half_trace_unpolarized",
    )


class TestFractionalHorizonExcess:
    def test_relative_difference_is_owned_read_only(self):
        a = _lightcurve(np.array([0.0, 1.0]), np.array([1.0, 2.0]))
        b = _lightcurve(np.array([0.0, 1.0]), np.array([2.0, 4.0]))
        excess = fractional_horizon_excess(a, b)
        np.testing.assert_array_equal(excess, np.ones(2))
        assert excess.flags.owndata
        assert not excess.flags.writeable

    def test_mismatched_grids_raise(self):
        a = _lightcurve(np.array([0.0, 1.0]), np.ones(2))
        b = _lightcurve(np.array([0.0, 1.0, 2.0]), np.ones(3))
        with pytest.raises(ValueError, match="LST grid"):
            fractional_horizon_excess(a, b)


class TestComputeDriftScanLightcurveValidation:
    @pytest.mark.parametrize(
        ("changes", "match"),
        [
            ({"integrated_flux": np.array([3.0, np.nan])}, "finite"),
            ({"reference_scientific_fingerprint": "g" * 64}, "SHA-256"),
            ({"reference_handler_id": " handler "}, "stripped"),
            ({"frequency_hz": 0.0}, "positive"),
            ({"nside": 3}, "HEALPix NSIDE"),
        ],
    )
    def test_public_model_rejects_hostile_state(self, changes, match):
        values = {
            "lst_hours": np.array([1.0, 2.0]),
            "integrated_flux": np.array([3.0, 4.0]),
            "mean_brightness": np.array([1.5, 2.0]),
            "horizon_masked": True,
            "frequency_hz": 150_000_000.0,
            "nside": 1,
            "beam_evaluation_time_mjd": 60_676.0,
            "reference_antenna": _lightcurve(
                np.array([0.0]),
                np.array([1.0]),
            ).reference_antenna,
            "reference_handler_id": "handler",
            "reference_scientific_fingerprint": "a" * 64,
            "power_convention": "half_trace_unpolarized",
        }
        values.update(changes)

        with pytest.raises((TypeError, ValueError), match=match):
            DriftScanLightcurve(**values)

    def test_no_healpix_payload_raises(self, tmp_path, precision):
        import radiosim.core.observability as observability
        from radiosim.core.sky import create_from_arrays

        sky = create_from_arrays(
            ra_rad=np.array([0.0]),
            dec_rad=np.array([0.0]),
            flux=np.array([1.0]),
            spectral_index=np.array([-0.7]),
            reference_frequency=150e6,
            precision=precision,
        )
        simulator = _canonical_beam_system(tmp_path)
        with pytest.raises(
            observability.ObservabilitySkyUnavailableError,
            match="HEALPix payload",
        ):
            compute_drift_scan_lightcurve(
                sky,
                beam_system=simulator.beam_system,
                reference_antenna=simulator.instrument.antennas[0].id,
                location=simulator.instrument.location,
                frequency_hz=150_000_000.0,
                lst_hours=np.array([0.0]),
                beam_evaluation_time_mjd=60_676.0,
            )


def _canonical_beam_system(tmp_path: Path) -> Simulator:
    antenna_path = tmp_path / "lightcurve-antennas.txt"
    antenna_path.write_text(
        "Name Number BeamID E N U Diameter\n"
        "ANT0 0 0 0.0 0.0 0.0 14.0\n"
        "ANT1 1 0 14.0 0.0 0.0 14.0\n",
        encoding="utf-8",
    )
    mapping = {
        "instrument": {
            "source": {
                "kind": "layout_file",
                "path": str(antenna_path),
                "format": "radiosim",
                "telescope_name": "Lightcurve Array",
            },
            "location": {
                "longitude_deg": 21.0,
                "latitude_deg": -30.0,
                "height_m": 1000.0,
            },
        },
        "baseline_selection": {"correlations": "cross"},
        "beams": {
            "mode": "analytic",
            "model": {"kind": "circular_aperture", "taper": {"kind": "uniform"}},
        },
        "obs_time": {
            "start_time": "2025-01-01T00:00:00",
            "duration_seconds": 1.0,
            "time_step_seconds": 1.0,
        },
        "obs_frequency": {
            "mode": "explicit",
            "channel_frequencies_hz": [150_000_000.0],
        },
        "sky_model": {
            "sources": [{"kind": "test_sources", "num_sources": 1, "seed": 1}]
        },
        "execution": {"backend": "numpy", "offline": True},
    }
    simulator = Simulator.from_mapping(mapping, base_dir=tmp_path)
    simulator._ensure_instrument_state()
    simulator._ensure_beam_system()
    return simulator


class TestTier3GDriftScanContract:
    def test_public_signature_has_only_canonical_beam_inputs(self):
        parameters = inspect.signature(compute_drift_scan_lightcurve).parameters

        assert tuple(parameters) == (
            "sky",
            "beam_system",
            "reference_antenna",
            "location",
            "frequency_hz",
            "lst_hours",
            "beam_evaluation_time_mjd",
            "area_normalize",
        )
        for removed in (
            "beam_fits_path",
            "beam_diameter_m",
            "latitude_deg",
            "longitude_deg",
            "height_m",
            "mask_horizon",
        ):
            assert removed not in parameters

    def test_drift_scan_uses_canonical_system_without_reopening_beam(
        self,
        tmp_path,
        precision,
        monkeypatch,
    ):
        simulator = _canonical_beam_system(tmp_path)
        sky = _uniform_sky(precision, nside=4, value_k=7.0)
        calls = 0

        def half_trace_probe(
            self,
            antenna_id,
            *,
            altitude_rad,
            azimuth_rad,
            frequency_hz,
            time_mjd,
            backend=None,
        ):
            nonlocal calls
            del self, antenna_id, azimuth_rad, frequency_hz, time_mjd, backend
            calls += 1
            visible = np.asarray(altitude_rad) > 0.0
            result = np.zeros(visible.shape + (2, 2), dtype=np.complex128)
            result[..., 0, 1] = np.where(visible, np.sqrt(2.0) * 1j, 0.0)
            result.setflags(write=False)
            return result

        def forbidden_read(*_args, **_kwargs):
            pytest.fail("drift scan reopened or independently loaded a beam")

        monkeypatch.setattr(BeamSystem, "evaluate_jones", half_trace_probe)
        monkeypatch.setattr("pyuvdata.UVBeam.read_beamfits", forbidden_read)
        reference = simulator.instrument.antennas[0].id

        result = compute_drift_scan_lightcurve(
            sky,
            beam_system=simulator.beam_system,
            reference_antenna=reference,
            location=simulator.instrument.location,
            frequency_hz=150_000_000.0,
            lst_hours=np.array([0.0, 6.0, 12.0]),
            beam_evaluation_time_mjd=60_676.0,
            area_normalize=True,
        )

        assert calls == 3
        assert result.horizon_masked is True
        assert result.reference_antenna == reference
        assert result.reference_handler_id
        assert result.reference_scientific_fingerprint
        assert result.power_convention == "half_trace_unpolarized"
        np.testing.assert_allclose(result.mean_brightness, 7.0)
        for value in (result.lst_hours, result.integrated_flux):
            assert value.flags.owndata
            assert not value.flags.writeable

    def test_exact_sky_frequency_is_required(self, tmp_path, precision):
        import radiosim.core.observability as observability

        simulator = _canonical_beam_system(tmp_path)
        sky = _uniform_sky(precision, nside=4)

        with pytest.raises(observability.InvalidObservabilityContextError):
            compute_drift_scan_lightcurve(
                sky,
                beam_system=simulator.beam_system,
                reference_antenna=simulator.instrument.antennas[0].id,
                location=simulator.instrument.location,
                frequency_hz=149_000_000.0,
                lst_hours=np.array([0.0]),
                beam_evaluation_time_mjd=60_676.0,
            )
