"""Tests for BBS sky loader edge cases."""

from __future__ import annotations

import numpy as np
import pytest

from radiosim.core.precision import PrecisionConfig
from radiosim.core.sky.loaders.bbs import load_bbs


def test_bbs_loader_parses_wsclean_format_with_metadata(tmp_path):
    model_file = tmp_path / "sources.skymodel"
    model_file.write_text(
        "\n".join(
            [
                "Format = Name, Type, Ra, Dec, I, Q, U, V, "
                "ReferenceFrequency='150000000', SpectralIndex, RotationMeasure, "
                "MajorAxis, MinorAxis, Orientation",
                "bright, GAUSSIAN, 180deg, -30deg, 2.5, 0.2, 0.1, 0.0, , "
                "[-0.7,-0.05], 1.5, 30.0, 20.0, 45.0",
                "dim, POINT, 181deg, -31deg, 0.5, 0, 0, 0, , [-0.8], 0, 0, 0, 0",
            ]
        )
    )

    sky = load_bbs(
        str(model_file),
        flux_limit=1.0,
        precision=PrecisionConfig.standard(),
    )

    assert sky.n_point_sources == 1
    assert sky.reference_frequency == 150_000_000.0
    assert sky.point is not None
    np.testing.assert_allclose(np.rad2deg(sky.point.ra_rad), [180.0])
    np.testing.assert_allclose(np.rad2deg(sky.point.dec_rad), [-30.0])
    np.testing.assert_allclose(sky.point.flux, [2.5])
    np.testing.assert_allclose(sky.point.stokes_q, [0.2])
    np.testing.assert_allclose(sky.point.stokes_u, [0.1])
    assert sky.point.polarization is not None
    np.testing.assert_allclose(sky.point.polarization.rotation_measure, [1.5])
    np.testing.assert_allclose(sky.point.ref_freq, [150_000_000.0])
    assert sky.point.morphology is not None
    np.testing.assert_allclose(sky.point.morphology.major_arcsec, [30.0])
    np.testing.assert_allclose(sky.point.morphology.minor_arcsec, [20.0])
    np.testing.assert_allclose(sky.point.morphology.pa_deg, [45.0])
    np.testing.assert_allclose(sky.point.spectral_coeffs, [[-0.7, -0.05]])
    assert sky.point.metadata is not None
    np.testing.assert_array_equal(sky.point.metadata.source_name, np.array(["bright"]))


def test_bbs_loader_missing_file_has_actionable_error(tmp_path):
    missing = tmp_path / "missing.skymodel"

    with pytest.raises(OSError, match="Could not open BBS sky model file"):
        load_bbs(str(missing), precision=PrecisionConfig.standard())
