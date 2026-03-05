# tests/conftest.py
"""
Pytest configuration and shared fixtures for RRIvis test suite.

This module provides:
- Path configuration for imports
- Shared fixtures for antennas, baselines, sources, locations, times
- Backend fixtures for testing across NumPy/JAX/Numba
- Pytest markers for categorizing tests
"""

import os
import sys

import numpy as np
import pytest

# Add src/ directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))


# =============================================================================
# PYTEST MARKERS
# =============================================================================


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "gpu: marks tests that require GPU (deselect with '-m \"not gpu\"')"
    )
    config.addinivalue_line("markers", "integration: marks integration tests")
    config.addinivalue_line("markers", "performance: marks performance/benchmark tests")


def pytest_runtest_logreport(report):
    """Custom test result logging."""
    if report.when == "call":
        if report.passed:
            print(f"✅ {report.nodeid} passed.")
        elif report.failed:
            print(f"❌ {report.nodeid} failed.")
        elif report.skipped:
            print(f"⚠️ {report.nodeid} skipped.")


# =============================================================================
# ANTENNA FIXTURES
# =============================================================================


@pytest.fixture
def sample_antennas_simple():
    """Simple 2-antenna layout for basic tests."""
    return {
        0: {
            "Name": "Ant0",
            "Number": 0,
            "BeamID": 0,
            "Position": (0.0, 0.0, 0.0),
            "diameter": 14.0,
        },
        1: {
            "Name": "Ant1",
            "Number": 1,
            "BeamID": 0,
            "Position": (100.0, 0.0, 0.0),
            "diameter": 14.0,
        },
    }


@pytest.fixture
def sample_antennas_three():
    """Three-antenna layout (like HERA subset)."""
    return {
        136: {
            "Name": "HH136",
            "Number": 136,
            "BeamID": 0,
            "Position": (-156.5976, 2.9439, -0.1819),
            "diameter": 14.0,
        },
        140: {
            "Name": "HH140",
            "Number": 140,
            "BeamID": 0,
            "Position": (-98.1662, 3.1671, -0.3008),
            "diameter": 14.0,
        },
        121: {
            "Name": "HH121",
            "Number": 121,
            "BeamID": 0,
            "Position": (-90.8139, -9.4618, -0.1707),
            "diameter": 14.0,
        },
    }


@pytest.fixture
def sample_antennas_large():
    """Larger antenna layout (10 antennas) for performance tests."""
    antennas = {}
    np.random.seed(42)
    for i in range(10):
        antennas[i] = {
            "Name": f"Ant{i}",
            "Number": i,
            "BeamID": 0,
            "Position": tuple(np.random.randn(3) * 100),
            "diameter": 14.0,
        }
    return antennas


# =============================================================================
# BASELINE FIXTURES
# =============================================================================


@pytest.fixture
def sample_baselines_simple(sample_antennas_simple):
    """Baselines for simple 2-antenna layout."""
    return {
        (0, 1): {"BaselineVector": np.array([100.0, 0.0, 0.0])},
    }


@pytest.fixture
def sample_baselines_three(sample_antennas_three):
    """Baselines for three-antenna layout."""
    return {
        (136, 136): {"BaselineVector": np.array([0.0, 0.0, 0.0])},
        (136, 140): {"BaselineVector": np.array([58.4314, 0.2232, -0.1189])},
        (136, 121): {"BaselineVector": np.array([65.7837, 12.4057, -0.0112])},
        (140, 121): {"BaselineVector": np.array([7.3523, 12.6289, 0.1301])},
    }


# =============================================================================
# SOURCE FIXTURES
# =============================================================================


@pytest.fixture
def sample_sources_single():
    """Single bright source for basic tests."""
    import astropy.units as u
    from astropy.coordinates import SkyCoord

    return [
        {
            "coords": SkyCoord(ra=0.0 * u.deg, dec=45.0 * u.deg, frame="icrs"),
            "flux": 1.0,
            "spectral_index": -0.7,
        },
    ]


@pytest.fixture
def sample_sources_multiple():
    """Multiple sources for comprehensive tests."""
    import astropy.units as u
    from astropy.coordinates import SkyCoord

    return [
        {
            "coords": SkyCoord(ra=45.0 * u.deg, dec=45.0 * u.deg, frame="icrs"),
            "flux": 1.0,
            "spectral_index": -0.7,
        },
        {
            "coords": SkyCoord(ra=60.0 * u.deg, dec=30.0 * u.deg, frame="icrs"),
            "flux": 2.0,
            "spectral_index": -0.5,
        },
        {
            "coords": SkyCoord(ra=120.0 * u.deg, dec=-10.0 * u.deg, frame="icrs"),
            "flux": 0.5,
            "spectral_index": -0.8,
        },
    ]


@pytest.fixture
def sample_sources_polarized():
    """Polarized sources with Stokes Q, U, V."""
    import astropy.units as u
    from astropy.coordinates import SkyCoord

    return [
        {
            "coords": SkyCoord(ra=0.0 * u.deg, dec=45.0 * u.deg, frame="icrs"),
            "flux": 10.0,
            "spectral_index": -0.7,
            "stokes_q": 1.0,  # 10% polarized
            "stokes_u": 0.5,
            "stokes_v": 0.2,
        },
    ]


# =============================================================================
# LOCATION AND TIME FIXTURES
# =============================================================================


@pytest.fixture
def sample_location():
    """Sample observatory location (MWA-like)."""
    import astropy.units as u
    from astropy.coordinates import EarthLocation

    return EarthLocation(lat=-26.7033 * u.deg, lon=116.6708 * u.deg, height=377.0 * u.m)


@pytest.fixture
def sample_location_northern():
    """Northern hemisphere location for testing."""
    import astropy.units as u
    from astropy.coordinates import EarthLocation

    return EarthLocation(lat=45.0 * u.deg, lon=-93.0 * u.deg, height=300.0 * u.m)


@pytest.fixture
def sample_obstime():
    """Sample observation time."""
    from astropy.time import Time

    return Time("2023-06-21T12:00:00", scale="utc")


@pytest.fixture
def sample_obstime_midnight():
    """Midnight observation time."""
    from astropy.time import Time

    return Time("2023-01-01T00:00:00", scale="utc")


# =============================================================================
# FREQUENCY FIXTURES
# =============================================================================


@pytest.fixture
def sample_frequencies_single():
    """Single frequency channel."""
    return np.array([150e6])  # 150 MHz


@pytest.fixture
def sample_frequencies_multiple():
    """Multiple frequency channels (3 channels)."""
    return np.array([100e6, 110e6, 120e6])  # 100-120 MHz


@pytest.fixture
def sample_frequencies_wideband():
    """Wideband frequency array (10 channels)."""
    return np.linspace(100e6, 200e6, 10)  # 100-200 MHz


@pytest.fixture
def sample_wavelengths(sample_frequencies_multiple):
    """Wavelengths corresponding to sample frequencies."""
    import astropy.units as u

    return (3e8 / sample_frequencies_multiple) * u.m


# =============================================================================
# HPBW FIXTURES
# =============================================================================


@pytest.fixture
def sample_hpbw_simple(sample_antennas_simple, sample_frequencies_multiple):
    """HPBW per antenna for simple layout."""
    theta_HPBW = np.radians(10.0)
    return {
        0: np.full(len(sample_frequencies_multiple), theta_HPBW),
        1: np.full(len(sample_frequencies_multiple), theta_HPBW),
    }


@pytest.fixture
def sample_hpbw_three(sample_antennas_three, sample_frequencies_multiple):
    """HPBW per antenna for three-antenna layout."""
    theta_HPBW = np.radians(5.0)
    return {
        136: np.full(len(sample_frequencies_multiple), theta_HPBW),
        140: np.full(len(sample_frequencies_multiple), theta_HPBW),
        121: np.full(len(sample_frequencies_multiple), theta_HPBW),
    }


# =============================================================================
# BACKEND FIXTURES
# =============================================================================


@pytest.fixture
def numpy_backend():
    """NumPy backend (always available)."""
    from rrivis.backends import get_backend

    return get_backend("numpy")


@pytest.fixture
def auto_backend():
    """Auto-detected best backend."""
    from rrivis.backends import get_backend

    return get_backend("auto")


@pytest.fixture(params=["numpy"])
def all_available_backends(request):
    """Parametrized fixture for testing across all available backends."""
    from rrivis.backends import get_backend, list_backends

    backend_name = request.param
    backends = list_backends()

    if backend_name == "numba" and not backends.get("numba", False):
        pytest.skip("Numba not available")
    if backend_name == "jax" and not backends.get("jax", False):
        pytest.skip("JAX not available")

    return get_backend(backend_name)


# =============================================================================
# SIMULATOR FIXTURES
# =============================================================================


@pytest.fixture
def rime_simulator():
    """RIME simulator instance."""
    from rrivis.simulator import get_simulator

    return get_simulator("rime")


# =============================================================================
# COMPLETE SIMULATION SETUP FIXTURES
# =============================================================================


@pytest.fixture
def complete_simulation_setup(
    sample_antennas_simple,
    sample_baselines_simple,
    sample_sources_single,
    sample_location_northern,
    sample_obstime,
    sample_frequencies_multiple,
    sample_wavelengths,
    sample_hpbw_simple,
    numpy_backend,
):
    """Complete setup for running a visibility calculation."""
    return {
        "antennas": sample_antennas_simple,
        "baselines": sample_baselines_simple,
        "sources": sample_sources_single,
        "location": sample_location_northern,
        "obstime": sample_obstime,
        "frequencies": sample_frequencies_multiple,
        "wavelengths": sample_wavelengths,
        "hpbw_per_antenna": sample_hpbw_simple,
        "backend": numpy_backend,
    }


# =============================================================================
# TEMPORARY DIRECTORY FIXTURES
# =============================================================================


@pytest.fixture
def temp_output_dir(tmp_path):
    """Temporary directory for test outputs."""
    output_dir = tmp_path / "test_output"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


@pytest.fixture
def temp_config_file(tmp_path):
    """Temporary config file for testing."""
    config_content = """
antenna_layout:
  antenna_positions_file: "test_antennas.txt"
  antenna_file_format: "rrivis"
  all_antenna_type: "parabolic"
  all_antenna_diameter: 14.0

obs_frequency:
  starting_frequency: 100
  frequency_bandwidth: 50
  frequency_interval: 10
  frequency_unit: "MHz"

location:
  lat: -26.7033
  lon: 116.6708
  height: 377.0

obs_time:
  start_time: "2023-06-21T12:00:00"
  duration_seconds: 3600
  time_step_seconds: 60

sky_model:
  test_sources:
    use_test_sources: true
    num_sources: 10
  gleam:
    use_gleam: false
  gsm_healpix:
    use_gsm: false
"""
    config_path = tmp_path / "test_config.yaml"
    config_path.write_text(config_content)
    return config_path
