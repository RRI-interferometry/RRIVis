# tests/test_simulator.py
"""Tests for the simulator abstraction layer (Phase 3).

Tests cover:
- VisibilitySimulator ABC interface
- RIMESimulator implementation
- get_simulator() factory function
- list_simulators() helper
- High-level Simulator API
"""

from unittest.mock import Mock

import numpy as np
import pytest


class TestVisibilitySimulatorABC:
    """Test the VisibilitySimulator abstract base class."""

    def test_cannot_instantiate_abc(self):
        """VisibilitySimulator cannot be instantiated directly."""
        from rrivis.simulator.base import VisibilitySimulator

        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            VisibilitySimulator()

    def test_abc_requires_name_property(self):
        """Subclasses must implement name property."""
        from rrivis.simulator.base import VisibilitySimulator

        class IncompleteSimulator(VisibilitySimulator):
            @property
            def description(self) -> str:
                return "test"

            def calculate_visibilities(self, *args, **kwargs):
                pass

        with pytest.raises(TypeError, match="abstract"):
            IncompleteSimulator()

    def test_abc_requires_description_property(self):
        """Subclasses must implement description property."""
        from rrivis.simulator.base import VisibilitySimulator

        class IncompleteSimulator(VisibilitySimulator):
            @property
            def name(self) -> str:
                return "test"

            def calculate_visibilities(self, *args, **kwargs):
                pass

        with pytest.raises(TypeError, match="abstract"):
            IncompleteSimulator()

    def test_abc_requires_calculate_visibilities(self):
        """Subclasses must implement calculate_visibilities method."""
        from rrivis.simulator.base import VisibilitySimulator

        class IncompleteSimulator(VisibilitySimulator):
            @property
            def name(self) -> str:
                return "test"

            @property
            def description(self) -> str:
                return "test"

        with pytest.raises(TypeError, match="abstract"):
            IncompleteSimulator()

    def test_valid_subclass_can_be_instantiated(self):
        """A complete subclass can be instantiated."""
        from rrivis.simulator.base import VisibilitySimulator

        class TestSimulator(VisibilitySimulator):
            @property
            def name(self) -> str:
                return "test"

            @property
            def description(self) -> str:
                return "Test simulator"

            def calculate_visibilities(self, *args, **kwargs) -> dict:
                return {}

        sim = TestSimulator()
        assert sim.name == "test"
        assert sim.description == "Test simulator"

    def test_default_complexity(self):
        """Default complexity is 'Unknown'."""
        from rrivis.simulator.base import VisibilitySimulator

        class TestSimulator(VisibilitySimulator):
            @property
            def name(self) -> str:
                return "test"

            @property
            def description(self) -> str:
                return "test"

            def calculate_visibilities(self, *args, **kwargs) -> dict:
                return {}

        sim = TestSimulator()
        assert sim.complexity == "Unknown"

    def test_default_supports_polarization(self):
        """Default supports_polarization is True."""
        from rrivis.simulator.base import VisibilitySimulator

        class TestSimulator(VisibilitySimulator):
            @property
            def name(self) -> str:
                return "test"

            @property
            def description(self) -> str:
                return "test"

            def calculate_visibilities(self, *args, **kwargs) -> dict:
                return {}

        sim = TestSimulator()
        assert sim.supports_polarization is True

    def test_default_supports_gpu(self):
        """Default supports_gpu is True."""
        from rrivis.simulator.base import VisibilitySimulator

        class TestSimulator(VisibilitySimulator):
            @property
            def name(self) -> str:
                return "test"

            @property
            def description(self) -> str:
                return "test"

            def calculate_visibilities(self, *args, **kwargs) -> dict:
                return {}

        sim = TestSimulator()
        assert sim.supports_gpu is True


class TestVisibilitySimulatorValidation:
    """Test the validate_inputs method."""

    @pytest.fixture
    def simulator(self):
        """Create a test simulator."""
        from rrivis.simulator.base import VisibilitySimulator

        class TestSimulator(VisibilitySimulator):
            @property
            def name(self) -> str:
                return "test"

            @property
            def description(self) -> str:
                return "test"

            def calculate_visibilities(self, *args, **kwargs) -> dict:
                return {}

        return TestSimulator()

    def test_validate_empty_antennas(self, simulator):
        """Empty antennas should fail validation."""
        valid, errors = simulator.validate_inputs(
            antennas={},
            baselines={(0, 1): {"BaselineVector": [1, 0, 0]}},
            sources=[],
            frequencies=np.array([100e6]),
        )
        assert not valid
        assert any("empty" in e.lower() for e in errors)

    def test_validate_empty_baselines(self, simulator):
        """Empty baselines should fail validation."""
        valid, errors = simulator.validate_inputs(
            antennas={0: {"Position": [0, 0, 0]}},
            baselines={},
            sources=[],
            frequencies=np.array([100e6]),
        )
        assert not valid
        assert any("empty" in e.lower() for e in errors)

    def test_validate_empty_frequencies(self, simulator):
        """Empty frequencies should fail validation."""
        valid, errors = simulator.validate_inputs(
            antennas={0: {"Position": [0, 0, 0]}},
            baselines={(0, 1): {"BaselineVector": [1, 0, 0]}},
            sources=[],
            frequencies=np.array([]),
        )
        assert not valid
        assert any("empty" in e.lower() for e in errors)

    def test_validate_missing_position(self, simulator):
        """Antennas missing Position key should fail."""
        valid, errors = simulator.validate_inputs(
            antennas={0: {"Name": "ant0"}},  # Missing Position
            baselines={(0, 1): {"BaselineVector": [1, 0, 0]}},
            sources=[],
            frequencies=np.array([100e6]),
        )
        assert not valid
        assert any("Position" in e for e in errors)

    def test_validate_missing_baseline_vector(self, simulator):
        """Baselines missing BaselineVector should fail."""
        valid, errors = simulator.validate_inputs(
            antennas={0: {"Position": [0, 0, 0]}},
            baselines={(0, 1): {"Length": 10}},  # Missing BaselineVector
            sources=[],
            frequencies=np.array([100e6]),
        )
        assert not valid
        assert any("BaselineVector" in e for e in errors)

    def test_validate_valid_inputs(self, simulator):
        """Valid inputs should pass validation."""
        valid, errors = simulator.validate_inputs(
            antennas={0: {"Position": [0, 0, 0]}, 1: {"Position": [10, 0, 0]}},
            baselines={(0, 1): {"BaselineVector": [10, 0, 0]}},
            sources=[],  # Empty sources is allowed
            frequencies=np.array([100e6, 150e6]),
        )
        assert valid
        assert len(errors) == 0


class TestVisibilitySimulatorMemoryEstimate:
    """Test the get_memory_estimate method."""

    @pytest.fixture
    def simulator(self):
        """Create a test simulator."""
        from rrivis.simulator.base import VisibilitySimulator

        class TestSimulator(VisibilitySimulator):
            @property
            def name(self) -> str:
                return "test"

            @property
            def description(self) -> str:
                return "test"

            def calculate_visibilities(self, *args, **kwargs) -> dict:
                return {}

        return TestSimulator()

    def test_memory_estimate_returns_dict(self, simulator):
        """get_memory_estimate should return a dict with expected keys."""
        mem = simulator.get_memory_estimate(
            n_antennas=10,
            n_baselines=55,
            n_sources=100,
            n_frequencies=100,
        )
        assert isinstance(mem, dict)
        assert "output_bytes" in mem
        assert "working_bytes" in mem
        assert "total_bytes" in mem
        assert "output_human" in mem
        assert "total_human" in mem

    def test_memory_estimate_scales_with_baselines(self, simulator):
        """Memory should scale with number of baselines."""
        mem_small = simulator.get_memory_estimate(
            n_antennas=10, n_baselines=55, n_sources=100, n_frequencies=100
        )
        mem_large = simulator.get_memory_estimate(
            n_antennas=100, n_baselines=5050, n_sources=100, n_frequencies=100
        )
        assert mem_large["total_bytes"] > mem_small["total_bytes"]

    def test_memory_estimate_warning_for_high_memory(self, simulator):
        """Should warn for high memory usage."""
        mem = simulator.get_memory_estimate(
            n_antennas=350,
            n_baselines=61425,
            n_sources=100000,
            n_frequencies=2048,
        )
        assert mem["warning"] is not None


class TestRIMESimulator:
    """Test the RIMESimulator implementation."""

    def test_rime_simulator_instantiation(self):
        """RIMESimulator can be instantiated."""
        from rrivis.simulator.rime import RIMESimulator

        sim = RIMESimulator()
        assert sim is not None

    def test_rime_simulator_name(self):
        """RIMESimulator has correct name."""
        from rrivis.simulator.rime import RIMESimulator

        sim = RIMESimulator()
        assert sim.name == "rime"

    def test_rime_simulator_description(self):
        """RIMESimulator has a description."""
        from rrivis.simulator.rime import RIMESimulator

        sim = RIMESimulator()
        assert "RIME" in sim.description or "rime" in sim.description.lower()

    def test_rime_simulator_complexity(self):
        """RIMESimulator has correct complexity."""
        from rrivis.simulator.rime import RIMESimulator

        sim = RIMESimulator()
        assert "N_src" in sim.complexity or "O(" in sim.complexity

    def test_rime_simulator_supports_polarization(self):
        """RIMESimulator supports polarization."""
        from rrivis.simulator.rime import RIMESimulator

        sim = RIMESimulator()
        assert sim.supports_polarization is True

    def test_rime_simulator_supports_gpu(self):
        """RIMESimulator supports GPU."""
        from rrivis.simulator.rime import RIMESimulator

        sim = RIMESimulator()
        assert sim.supports_gpu is True

    def test_rime_simulator_repr(self):
        """RIMESimulator has string representation."""
        from rrivis.simulator.rime import RIMESimulator

        sim = RIMESimulator()
        repr_str = repr(sim)
        assert "RIMESimulator" in repr_str
        assert "rime" in repr_str

    def test_rime_simulator_str(self):
        """RIMESimulator has human-readable string."""
        from rrivis.simulator.rime import RIMESimulator

        sim = RIMESimulator()
        str_repr = str(sim)
        assert "rime" in str_repr.lower()

    def test_rime_simulator_missing_required_kwargs(self):
        """RIMESimulator raises error for missing required kwargs."""
        from rrivis.simulator.rime import RIMESimulator

        sim = RIMESimulator()
        mock_backend = Mock()

        with pytest.raises(ValueError, match="requires"):
            sim.calculate_visibilities(
                antennas={},
                baselines={},
                sources=[],
                frequencies=np.array([100e6]),
                backend=mock_backend,
                # Missing: location, obstime, wavelengths
            )


class TestGetSimulator:
    """Test the get_simulator factory function."""

    def test_get_simulator_default(self):
        """get_simulator() returns RIME simulator by default."""
        from rrivis.simulator import get_simulator
        from rrivis.simulator.rime import RIMESimulator

        sim = get_simulator()
        assert isinstance(sim, RIMESimulator)
        assert sim.name == "rime"

    def test_get_simulator_rime(self):
        """get_simulator('rime') returns RIMESimulator."""
        from rrivis.simulator import get_simulator
        from rrivis.simulator.rime import RIMESimulator

        sim = get_simulator("rime")
        assert isinstance(sim, RIMESimulator)

    def test_get_simulator_invalid_name(self):
        """get_simulator with invalid name raises ValueError."""
        from rrivis.simulator import get_simulator

        with pytest.raises(ValueError, match="Unknown simulator"):
            get_simulator("nonexistent")

    def test_get_simulator_returns_new_instance(self):
        """get_simulator returns new instance each time."""
        from rrivis.simulator import get_simulator

        sim1 = get_simulator("rime")
        sim2 = get_simulator("rime")
        assert sim1 is not sim2


class TestListSimulators:
    """Test the list_simulators function."""

    def test_list_simulators_returns_dict(self):
        """list_simulators returns a dictionary."""
        from rrivis.simulator import list_simulators

        sims = list_simulators()
        assert isinstance(sims, dict)

    def test_list_simulators_contains_rime(self):
        """list_simulators includes 'rime'."""
        from rrivis.simulator import list_simulators

        sims = list_simulators()
        assert "rime" in sims

    def test_list_simulators_values_are_strings(self):
        """list_simulators values are description strings."""
        from rrivis.simulator import list_simulators

        sims = list_simulators()
        for name, desc in sims.items():
            assert isinstance(name, str)
            assert isinstance(desc, str)
            assert len(desc) > 0


class TestGetSimulatorNames:
    """Test the get_simulator_names function."""

    def test_get_simulator_names_returns_list(self):
        """get_simulator_names returns a list."""
        from rrivis.simulator import get_simulator_names

        names = get_simulator_names()
        assert isinstance(names, list)

    def test_get_simulator_names_contains_rime(self):
        """get_simulator_names includes 'rime'."""
        from rrivis.simulator import get_simulator_names

        names = get_simulator_names()
        assert "rime" in names


class TestGetDefaultSimulator:
    """Test the get_default_simulator function."""

    def test_get_default_simulator(self):
        """get_default_simulator returns 'rime'."""
        from rrivis.simulator import get_default_simulator

        default = get_default_simulator()
        assert default == "rime"


class TestSimulatorModuleExports:
    """Test that simulator module exports are correct."""

    def test_exports_visibility_simulator(self):
        """VisibilitySimulator is exported."""
        from rrivis.simulator import VisibilitySimulator

        assert VisibilitySimulator is not None

    def test_exports_rime_simulator(self):
        """RIMESimulator is exported."""
        from rrivis.simulator import RIMESimulator

        assert RIMESimulator is not None

    def test_exports_get_simulator(self):
        """get_simulator is exported."""
        from rrivis.simulator import get_simulator

        assert callable(get_simulator)

    def test_exports_list_simulators(self):
        """list_simulators is exported."""
        from rrivis.simulator import list_simulators

        assert callable(list_simulators)


class TestHighLevelSimulatorAPI:
    """Test the high-level Simulator API."""

    def test_simulator_import(self):
        """Simulator can be imported from api module."""
        from rrivis.api.simulator import Simulator

        assert Simulator is not None

    def test_simulator_instantiation(self):
        """Simulator can be instantiated with minimal args."""
        from rrivis.api.simulator import Simulator

        # Should not raise even without arguments
        sim = Simulator()
        assert sim is not None

    def test_simulator_version(self):
        """Simulator has version attribute."""
        from rrivis.api.simulator import Simulator

        sim = Simulator()
        assert hasattr(sim, "version")
        assert isinstance(sim.version, str)

    def test_simulator_results_none_before_run(self):
        """Simulator.results is None before run()."""
        from rrivis.api.simulator import Simulator

        sim = Simulator()
        assert sim.results is None

    def test_simulator_repr(self):
        """Simulator has string representation."""
        from rrivis.api.simulator import Simulator

        sim = Simulator()
        repr_str = repr(sim)
        assert "Simulator" in repr_str
        assert "configured" in repr_str

    def test_simulator_plot_without_results_raises(self):
        """Simulator.plot() raises if no results."""
        from rrivis.api.simulator import Simulator

        sim = Simulator()
        with pytest.raises(RuntimeError, match="No results"):
            sim.plot()

    def test_simulator_save_without_results_raises(self):
        """Simulator.save() raises if no results."""
        from rrivis.api.simulator import Simulator

        sim = Simulator()
        with pytest.raises(RuntimeError, match="No results"):
            sim.save("/tmp/test")

    def test_simulator_build_config_from_frequencies(self):
        """Simulator builds config from frequencies."""
        from rrivis.api.simulator import Simulator

        sim = Simulator(frequencies=[100, 150, 200])
        assert "obs_frequency" in sim.config

    def test_simulator_build_config_from_sky_model(self):
        """Simulator builds config from sky_model."""
        from rrivis.api.simulator import Simulator

        sim = Simulator(sky_model="test")
        assert "sky_model" in sim.config


class TestSimulatorFromConfig:
    """Test Simulator.from_config() method."""

    def test_from_config_nonexistent_file(self):
        """from_config raises for nonexistent file."""
        from rrivis.api.simulator import Simulator

        with pytest.raises(FileNotFoundError):
            Simulator.from_config("/nonexistent/path/config.yaml")


# Integration tests (require more setup)
class TestSimulatorIntegration:
    """Integration tests for the Simulator (may be slow)."""

    @pytest.fixture
    def sample_config(self, tmp_path):
        """Create a sample config file."""
        config_content = """
telescope:
  telescope_name: "TEST"

antenna_layout:
  antenna_positions_file: null
  all_antenna_diameter: 14.0

location:
  lat: -30.72
  lon: 21.43
  height: 1073.0

obs_frequency:
  starting_frequency: 100.0
  frequency_bandwidth: 10.0
  frequency_interval: 5.0
  frequency_unit: "MHz"

obs_time:
  start_time: "2025-01-01T00:00:00"

sky_model:
  test_sources:
    use_test_sources: true
    num_sources: 3

output:
  save_simulation_data: false
"""
        config_path = tmp_path / "test_config.yaml"
        config_path.write_text(config_content)
        return config_path

    @pytest.mark.skip(reason="Integration test - requires full setup")
    def test_simulator_full_run(self, sample_config):
        """Test full simulation run with sample config."""

        # This would require actual antenna data, so we skip it
        # In a real test, we'd need to set up the full environment
        pass
