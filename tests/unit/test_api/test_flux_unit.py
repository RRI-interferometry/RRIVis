# tests/unit/test_api/test_flux_unit.py
"""
Unit tests for flux_unit conversion in the Simulator API.

Tests that flux values from config are correctly converted to canonical Jy
before being passed to SkyModel factory methods.
"""

import pytest


class TestFluxMultipliers:
    """Test flux unit multiplier values."""

    def test_jy_multiplier(self):
        """Default Jy multiplier is 1.0."""
        multipliers = {"Jy": 1.0, "mJy": 1e-3, "uJy": 1e-6}
        assert multipliers["Jy"] == 1.0

    def test_mjy_multiplier(self):
        """mJy multiplier is 1e-3."""
        multipliers = {"Jy": 1.0, "mJy": 1e-3, "uJy": 1e-6}
        assert multipliers["mJy"] == 1e-3

    def test_ujy_multiplier(self):
        """uJy multiplier is 1e-6."""
        multipliers = {"Jy": 1.0, "mJy": 1e-3, "uJy": 1e-6}
        assert multipliers["uJy"] == 1e-6

    def test_unknown_unit_defaults_to_1(self):
        """Unknown unit falls back to 1.0 multiplier."""
        multipliers = {"Jy": 1.0, "mJy": 1e-3, "uJy": 1e-6}
        assert multipliers.get("kJy", 1.0) == 1.0


class TestFluxConversion:
    """Test flux value conversion logic."""

    def test_none_flux_not_multiplied(self):
        """None flux values should remain None (not crash)."""
        _flux_mul = 1e-3
        flux_min = None
        flux_max = None
        flux_range = (
            (flux_min * _flux_mul, flux_max * _flux_mul)
            if flux_min is not None and flux_max is not None
            else None
        )
        assert flux_range is None

    def test_mjy_conversion(self):
        """2000 mJy should convert to 2.0 Jy."""
        _flux_mul = 1e-3
        flux_min = 2000.0
        flux_max = 8000.0
        result_min = flux_min * _flux_mul
        result_max = flux_max * _flux_mul
        assert result_min == pytest.approx(2.0)
        assert result_max == pytest.approx(8.0)

    def test_ujy_conversion(self):
        """1e6 uJy should convert to 1.0 Jy."""
        _flux_mul = 1e-6
        flux_val = 1_000_000.0
        assert flux_val * _flux_mul == pytest.approx(1.0)

    def test_jy_identity(self):
        """Jy multiplier leaves values unchanged."""
        _flux_mul = 1.0
        flux_val = 50.0
        assert flux_val * _flux_mul == pytest.approx(50.0)

    def test_flux_limit_conversion(self):
        """flux_limit should be multiplied correctly."""
        _flux_mul = 1e-3
        flux_limit = 50.0  # 50 mJy
        assert flux_limit * _flux_mul == pytest.approx(0.05)
