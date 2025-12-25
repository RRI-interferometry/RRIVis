"""Tests for precision configuration module.

Tests for rrivis.core.precision including:
- PrecisionConfig class and presets
- CoordinatePrecision and JonesPrecision sub-models
- Dtype resolution functions
- Backend compatibility validation
- Precision-aware backends
"""

import pytest
import numpy as np
import warnings

from rrivis.core.precision import (
    PrecisionConfig,
    PrecisionLevel,
    CoordinatePrecision,
    JonesPrecision,
    resolve_precision,
    get_real_dtype,
    get_complex_dtype,
    get_dtype_size,
    FLOAT128_AVAILABLE,
    COMPLEX256_AVAILABLE,
    VALID_PRECISIONS,
)
from rrivis.backends import get_backend, NumPyBackend


class TestPrecisionLevel:
    """Tests for precision level type and constants."""

    def test_valid_precisions(self):
        """Test that valid precisions are defined correctly."""
        assert "float32" in VALID_PRECISIONS
        assert "float64" in VALID_PRECISIONS
        assert "float128" in VALID_PRECISIONS
        assert len(VALID_PRECISIONS) == 3


class TestGetRealDtype:
    """Tests for get_real_dtype function."""

    def test_float32(self):
        """Test float32 dtype resolution."""
        assert get_real_dtype("float32") == np.float32

    def test_float64(self):
        """Test float64 dtype resolution."""
        assert get_real_dtype("float64") == np.float64

    def test_float128_numpy_available(self):
        """Test float128 on NumPy when available."""
        if FLOAT128_AVAILABLE:
            dtype = get_real_dtype("float128", backend="numpy")
            assert dtype == np.float128
        else:
            with pytest.warns(UserWarning, match="float128 not available"):
                dtype = get_real_dtype("float128", backend="numpy")
            assert dtype == np.float64

    def test_float128_jax_fallback(self):
        """Test float128 falls back on JAX backend."""
        with pytest.warns(UserWarning, match="float128 not supported on jax"):
            dtype = get_real_dtype("float128", backend="jax")
        assert dtype == np.float64

    def test_float128_numba_fallback(self):
        """Test float128 falls back on Numba backend."""
        with pytest.warns(UserWarning, match="float128 not supported on numba"):
            dtype = get_real_dtype("float128", backend="numba")
        assert dtype == np.float64

    def test_invalid_precision_raises(self):
        """Test that invalid precision raises ValueError."""
        with pytest.raises(ValueError, match="Unknown precision level"):
            get_real_dtype("float16")


class TestGetComplexDtype:
    """Tests for get_complex_dtype function."""

    def test_float32_complex64(self):
        """Test float32 gives complex64."""
        assert get_complex_dtype("float32") == np.complex64

    def test_float64_complex128(self):
        """Test float64 gives complex128."""
        assert get_complex_dtype("float64") == np.complex128

    def test_float128_complex256_available(self):
        """Test float128 on NumPy when available."""
        if COMPLEX256_AVAILABLE:
            dtype = get_complex_dtype("float128", backend="numpy")
            assert dtype == np.complex256
        else:
            with pytest.warns(UserWarning, match="complex256 not available"):
                dtype = get_complex_dtype("float128", backend="numpy")
            assert dtype == np.complex128

    def test_float128_jax_fallback(self):
        """Test float128 complex falls back on JAX backend."""
        with pytest.warns(UserWarning, match="complex256 not supported on jax"):
            dtype = get_complex_dtype("float128", backend="jax")
        assert dtype == np.complex128


class TestGetDtypeSize:
    """Tests for get_dtype_size function."""

    def test_float32_real(self):
        """Test float32 real size."""
        assert get_dtype_size("float32", complex_type=False) == 4

    def test_float32_complex(self):
        """Test float32 complex size."""
        assert get_dtype_size("float32", complex_type=True) == 8

    def test_float64_real(self):
        """Test float64 real size."""
        assert get_dtype_size("float64", complex_type=False) == 8

    def test_float64_complex(self):
        """Test float64 complex size."""
        assert get_dtype_size("float64", complex_type=True) == 16

    def test_float128_real(self):
        """Test float128 real size."""
        assert get_dtype_size("float128", complex_type=False) == 16

    def test_float128_complex(self):
        """Test float128 complex size."""
        assert get_dtype_size("float128", complex_type=True) == 32


class TestCoordinatePrecision:
    """Tests for CoordinatePrecision sub-model."""

    def test_default_values(self):
        """Test default precision values."""
        config = CoordinatePrecision()
        assert config.antenna_positions == "float64"
        assert config.source_positions == "float64"
        assert config.direction_cosines == "float64"
        assert config.uvw == "float64"

    def test_custom_values(self):
        """Test custom precision values."""
        config = CoordinatePrecision(
            antenna_positions="float32",
            direction_cosines="float128",
        )
        assert config.antenna_positions == "float32"
        assert config.direction_cosines == "float128"

    def test_invalid_precision_raises(self):
        """Test that invalid precision raises error."""
        with pytest.raises(ValueError, match="Invalid precision"):
            CoordinatePrecision(antenna_positions="float16")

    def test_get_dtype(self):
        """Test dtype retrieval."""
        config = CoordinatePrecision(antenna_positions="float32")
        assert config.get_dtype("antenna_positions") == np.float32


class TestJonesPrecision:
    """Tests for JonesPrecision sub-model."""

    def test_default_values(self):
        """Test default Jones precision values."""
        config = JonesPrecision()
        assert config.geometric_phase == "float64"
        assert config.beam == "float64"
        assert config.ionosphere == "float64"
        assert config.gain == "float64"

    def test_all_jones_terms(self):
        """Test all 8 Jones terms are present."""
        config = JonesPrecision()
        terms = [
            "geometric_phase",  # K
            "beam",             # E
            "ionosphere",       # Z
            "troposphere",      # T
            "parallactic",      # P
            "gain",             # G
            "bandpass",         # B
            "polarization_leakage",  # D
        ]
        for term in terms:
            assert hasattr(config, term)
            assert getattr(config, term) == "float64"

    def test_custom_values(self):
        """Test custom Jones precision values."""
        config = JonesPrecision(
            geometric_phase="float128",
            beam="float32",
        )
        assert config.geometric_phase == "float128"
        assert config.beam == "float32"
        assert config.gain == "float64"  # default

    def test_get_dtype_complex(self):
        """Test complex dtype retrieval."""
        config = JonesPrecision(geometric_phase="float64")
        assert config.get_dtype("geometric_phase") == np.complex128

    def test_get_dtype_real(self):
        """Test real dtype retrieval."""
        config = JonesPrecision(beam="float32")
        assert config.get_real_dtype("beam") == np.float32


class TestPrecisionConfig:
    """Tests for main PrecisionConfig class."""

    def test_default_values(self):
        """Test default configuration."""
        config = PrecisionConfig()
        assert config.default == "float64"
        assert config.accumulation == "float64"
        assert config.output == "float64"
        assert isinstance(config.coordinates, CoordinatePrecision)
        assert isinstance(config.jones, JonesPrecision)

    def test_custom_default(self):
        """Test custom default precision."""
        config = PrecisionConfig(default="float32")
        assert config.default == "float32"

    def test_nested_config(self):
        """Test nested precision configuration."""
        config = PrecisionConfig(
            jones=JonesPrecision(geometric_phase="float128"),
            accumulation="float128",
        )
        assert config.jones.geometric_phase == "float128"
        assert config.accumulation == "float128"

    def test_invalid_precision_raises(self):
        """Test that invalid precision raises error."""
        with pytest.raises(ValueError, match="Invalid precision"):
            PrecisionConfig(default="float16")


class TestPrecisionPresets:
    """Tests for precision preset methods."""

    def test_standard_preset(self):
        """Test standard (float64) preset."""
        config = PrecisionConfig.standard()
        assert config.default == "float64"
        assert config.accumulation == "float64"
        assert config.output == "float64"

    def test_fast_preset(self):
        """Test fast (mixed) preset."""
        config = PrecisionConfig.fast()
        assert config.default == "float32"
        # Critical paths remain float64
        assert config.jones.geometric_phase == "float64"
        assert config.coordinates.direction_cosines == "float64"
        assert config.accumulation == "float64"
        # Non-critical use float32
        assert config.jones.beam == "float32"
        assert config.output == "float32"

    def test_precise_preset(self):
        """Test precise preset."""
        config = PrecisionConfig.precise()
        assert config.default == "float64"
        # Critical paths use float128
        assert config.jones.geometric_phase == "float128"
        assert config.coordinates.direction_cosines == "float128"
        assert config.accumulation == "float128"
        # Output stays float64
        assert config.output == "float64"

    def test_ultra_preset(self):
        """Test ultra (all float128) preset."""
        config = PrecisionConfig.ultra()
        assert config.default == "float128"
        assert config.accumulation == "float128"
        assert config.output == "float128"
        assert config.jones.geometric_phase == "float128"
        assert config.jones.beam == "float128"


class TestPrecisionConfigHelpers:
    """Tests for PrecisionConfig helper methods."""

    def test_with_overrides(self):
        """Test with_overrides method."""
        config = PrecisionConfig.fast()
        modified = config.with_overrides(output="float64")
        assert modified.output == "float64"
        assert modified.default == "float32"  # unchanged

    def test_with_overrides_nested(self):
        """Test with_overrides with nested dict."""
        config = PrecisionConfig.standard()
        modified = config.with_overrides(
            jones={"beam": "float32"},
        )
        assert modified.jones.beam == "float32"
        assert modified.jones.geometric_phase == "float64"  # unchanged

    def test_get_real_dtype(self):
        """Test get_real_dtype method."""
        config = PrecisionConfig(
            coordinates=CoordinatePrecision(antenna_positions="float32"),
        )
        assert config.get_real_dtype("coordinates", "antenna_positions") == np.float32
        assert config.get_real_dtype("accumulation") == np.float64

    def test_get_complex_dtype(self):
        """Test get_complex_dtype method."""
        config = PrecisionConfig(
            jones=JonesPrecision(geometric_phase="float64"),
        )
        assert config.get_complex_dtype("jones", "geometric_phase") == np.complex128

    def test_estimate_memory_factor_standard(self):
        """Test memory factor for standard config."""
        config = PrecisionConfig.standard()
        factor = config.estimate_memory_factor()
        assert 0.9 < factor < 1.1  # Should be close to 1.0

    def test_estimate_memory_factor_fast(self):
        """Test memory factor for fast config."""
        config = PrecisionConfig.fast()
        factor = config.estimate_memory_factor()
        assert factor < 1.0  # Should be less than 1.0

    def test_validate_for_backend_numpy(self):
        """Test backend validation for NumPy."""
        config = PrecisionConfig.standard()
        warnings_list = config.validate_for_backend("numpy")
        assert len(warnings_list) == 0

    def test_validate_for_backend_jax_float128(self):
        """Test backend validation for JAX with float128."""
        config = PrecisionConfig.precise()
        warnings_list = config.validate_for_backend("jax")
        assert len(warnings_list) > 0
        assert "float128 not supported" in warnings_list[0]


class TestResolvePrecision:
    """Tests for resolve_precision helper function."""

    def test_none_returns_standard(self):
        """Test None resolves to standard."""
        config = resolve_precision(None)
        assert config.default == "float64"

    def test_config_passthrough(self):
        """Test PrecisionConfig is passed through."""
        original = PrecisionConfig.fast()
        resolved = resolve_precision(original)
        assert resolved is original

    def test_preset_string(self):
        """Test preset name strings."""
        assert resolve_precision("standard").default == "float64"
        assert resolve_precision("fast").default == "float32"
        assert resolve_precision("precise").default == "float64"
        assert resolve_precision("ultra").default == "float128"

    def test_precision_level_string(self):
        """Test precision level strings."""
        config = resolve_precision("float32")
        assert config.default == "float32"

    def test_invalid_string_raises(self):
        """Test invalid string raises error."""
        with pytest.raises(ValueError, match="Unknown precision"):
            resolve_precision("invalid")

    def test_invalid_type_raises(self):
        """Test invalid type raises error."""
        with pytest.raises(TypeError, match="precision must be"):
            resolve_precision(123)


class TestBackendPrecision:
    """Tests for precision in backends."""

    def test_numpy_default_precision(self):
        """Test NumPy backend with default precision."""
        backend = NumPyBackend()
        assert backend.precision is None
        assert backend.default_real_dtype == np.float64
        assert backend.default_complex_dtype == np.complex128

    def test_numpy_with_precision_string(self):
        """Test NumPy backend with precision string."""
        backend = NumPyBackend(precision="fast")
        assert backend.precision is not None
        assert backend.precision.default == "float32"

    def test_numpy_with_precision_config(self):
        """Test NumPy backend with PrecisionConfig."""
        config = PrecisionConfig.precise()
        backend = NumPyBackend(precision=config)
        assert backend.precision is config

    def test_get_backend_with_precision(self):
        """Test get_backend with precision parameter."""
        backend = get_backend("numpy", precision="fast")
        assert backend.precision is not None
        assert backend.precision.default == "float32"

    def test_backend_zeros_uses_precision(self):
        """Test zeros method uses precision config."""
        backend = NumPyBackend(precision="fast")
        arr = backend.zeros((10,))
        # With precision=fast, default is float32
        assert arr.dtype == np.float32

    def test_backend_zeros_complex_uses_precision(self):
        """Test zeros_complex uses precision config."""
        backend = NumPyBackend(precision="fast")
        arr = backend.zeros_complex((10,))
        # With precision=fast, default is float32 -> complex64
        assert arr.dtype == np.complex64

    def test_backend_eye_uses_precision(self):
        """Test eye method uses precision config."""
        backend = NumPyBackend(precision="fast")
        arr = backend.eye(3)
        assert arr.dtype == np.float32

    def test_backend_get_config_includes_precision(self):
        """Test get_config includes precision info."""
        config = PrecisionConfig.fast()
        backend = NumPyBackend(precision=config)
        backend_config = backend.get_config()
        assert "precision" in backend_config
        assert backend_config["precision"]["default"] == "float32"


class TestFloat128Availability:
    """Tests for float128 platform detection."""

    def test_float128_flag_is_bool(self):
        """Test FLOAT128_AVAILABLE is boolean."""
        assert isinstance(FLOAT128_AVAILABLE, bool)

    def test_complex256_flag_is_bool(self):
        """Test COMPLEX256_AVAILABLE is boolean."""
        assert isinstance(COMPLEX256_AVAILABLE, bool)

    @pytest.mark.skipif(not FLOAT128_AVAILABLE, reason="float128 not available")
    def test_float128_creates_correct_size(self):
        """Test float128 array has correct size when available."""
        arr = np.array([1.0], dtype=np.float128)
        assert arr.dtype.itemsize > 8  # Should be 16 bytes

    @pytest.mark.skipif(not COMPLEX256_AVAILABLE, reason="complex256 not available")
    def test_complex256_creates_correct_size(self):
        """Test complex256 array has correct size when available."""
        arr = np.array([1.0 + 1.0j], dtype=np.complex256)
        assert arr.dtype.itemsize > 16  # Should be 32 bytes
