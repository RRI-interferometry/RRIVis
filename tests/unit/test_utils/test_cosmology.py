"""Tests for rrivis.utils.cosmology — 21 cm helpers."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from rrivis.utils.cosmology import (
    F_21CM_HZ,
    add_redshift_secondary_axis,
    frequency_to_redshift_21cm,
    redshift_to_frequency_21cm,
)


class TestRedshiftRoundtrip:
    def test_scalar_roundtrip(self):
        f = 142e6
        z = frequency_to_redshift_21cm(f)
        f2 = redshift_to_frequency_21cm(z)
        assert f2 == pytest.approx(f, rel=1e-12)

    def test_array_roundtrip(self):
        f = np.linspace(50e6, 250e6, 17)
        z = frequency_to_redshift_21cm(f)
        f2 = redshift_to_frequency_21cm(z)
        np.testing.assert_allclose(f2, f, rtol=1e-12)

    def test_142mhz_is_z9(self):
        z = frequency_to_redshift_21cm(142e6)
        assert z == pytest.approx(9.0, abs=0.01)

    def test_rest_frame_is_z0(self):
        assert frequency_to_redshift_21cm(F_21CM_HZ) == pytest.approx(0.0, abs=1e-12)


class TestRedshiftValidation:
    def test_negative_frequency_raises(self):
        with pytest.raises(ValueError, match="positive"):
            frequency_to_redshift_21cm(-1e6)

    def test_zero_frequency_raises(self):
        with pytest.raises(ValueError, match="positive"):
            frequency_to_redshift_21cm(0.0)

    def test_z_le_minus_one_raises(self):
        with pytest.raises(ValueError, match="z > -1"):
            redshift_to_frequency_21cm(-1.0)


class TestSecondaryAxis:
    def test_attaches_secondary_x_axis(self):
        fig, ax = plt.subplots()
        ax.plot([100e6, 200e6], [1.0, 2.0])
        sec = add_redshift_secondary_axis(ax)
        assert sec is not None
        plt.close(fig)

    def test_attaches_secondary_y_axis(self):
        fig, ax = plt.subplots()
        ax.plot([1.0, 2.0], [100e6, 200e6])
        sec = add_redshift_secondary_axis(ax, axis="y")
        assert sec is not None
        plt.close(fig)

    def test_invalid_axis_raises(self):
        fig, ax = plt.subplots()
        with pytest.raises(ValueError, match="axis"):
            add_redshift_secondary_axis(ax, axis="z")
        plt.close(fig)
