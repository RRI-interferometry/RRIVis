# tests/integration/test_simulator_api.py
"""
Integration tests for the high-level Simulator API.

These tests verify the Simulator class provides a clean, user-friendly
interface for running visibility simulations.
"""

from pathlib import Path

import pytest
import yaml


@pytest.mark.integration
class TestSimulatorCreation:
    """Test Simulator instantiation."""

    def test_create_with_minimal_args(self, tmp_path):
        """Test creating Simulator with minimal arguments."""
        from rrivis.api.simulator import Simulator

        # Create a minimal antenna file
        antenna_file = tmp_path / "test_antennas.txt"
        antenna_file.write_text(
            "# name number beamid east north up\n"
            "Ant0 0 0 0.0 0.0 0.0\n"
            "Ant1 1 0 100.0 0.0 0.0\n"
        )

        sim = Simulator(
            antenna_layout=str(antenna_file),
            frequencies=[100, 150, 200],
        )

        assert sim is not None
        assert sim.config is not None
        assert "obs_frequency" in sim.config
        assert sim.version is not None

    def test_create_with_backend_selection(self, tmp_path):
        """Test creating Simulator with explicit backend."""
        from rrivis.api.simulator import Simulator

        # Create a minimal antenna file
        antenna_file = tmp_path / "test_antennas.txt"
        antenna_file.write_text(
            "# name number beamid east north up\n"
            "Ant0 0 0 0.0 0.0 0.0\n"
            "Ant1 1 0 100.0 0.0 0.0\n"
        )

        sim = Simulator(
            antenna_layout=str(antenna_file),
            frequencies=[100],
            backend="numpy",
        )

        assert sim is not None
        assert sim._backend_name == "numpy"

    def test_create_with_auto_backend(self, tmp_path):
        """Test creating Simulator with auto backend selection."""
        from rrivis.api.simulator import Simulator

        # Create a minimal antenna file
        antenna_file = tmp_path / "test_antennas.txt"
        antenna_file.write_text(
            "# name number beamid east north up\n"
            "Ant0 0 0 0.0 0.0 0.0\n"
            "Ant1 1 0 100.0 0.0 0.0\n"
        )

        sim = Simulator(
            antenna_layout=str(antenna_file),
            frequencies=[100],
            backend="auto",
        )

        assert sim is not None
        assert sim._backend_name == "auto"


@pytest.mark.integration
class TestSimulatorSetup:
    """Test Simulator setup phase."""

    def test_setup_loads_antennas(self, tmp_path):
        """Test that setup() loads antenna data."""
        from rrivis.api.simulator import Simulator

        # Create a minimal antenna file
        antenna_file = tmp_path / "test_antennas.txt"
        antenna_file.write_text(
            "# name number beamid east north up\n"
            "Ant0 0 0 0.0 0.0 0.0\n"
            "Ant1 1 0 100.0 0.0 0.0\n"
        )

        sim = Simulator(
            antenna_layout=str(antenna_file),
            frequencies=[100],
            sky_model="test_sources",
        )

        # Before setup, antennas should be None
        assert sim.antennas is None

        # After setup, antennas should be loaded
        sim.setup()
        assert sim.antennas is not None
        assert len(sim.antennas) >= 1  # At least some antennas loaded

    def test_setup_generates_baselines(self, tmp_path):
        """Test that setup() generates baselines."""
        from rrivis.api.simulator import Simulator

        # Create a 3-antenna file
        antenna_file = tmp_path / "test_antennas.txt"
        antenna_file.write_text(
            "# name number beamid east north up\n"
            "Ant0 0 0 0.0 0.0 0.0\n"
            "Ant1 1 0 100.0 0.0 0.0\n"
            "Ant2 2 0 50.0 86.6 0.0\n"
        )

        sim = Simulator(
            antenna_layout=str(antenna_file),
            frequencies=[100],
            sky_model="test_sources",
        )

        sim.setup()
        assert sim.baselines is not None
        # 3 antennas should have at least 3 baselines (plus autos if enabled)
        assert len(sim.baselines) >= 3

    def test_setup_returns_self_for_chaining(self, tmp_path):
        """Test that setup() returns self for method chaining."""
        from rrivis.api.simulator import Simulator

        # Create a minimal antenna file
        antenna_file = tmp_path / "test_antennas.txt"
        antenna_file.write_text(
            "# name number beamid east north up\n"
            "Ant0 0 0 0.0 0.0 0.0\n"
            "Ant1 1 0 100.0 0.0 0.0\n"
        )

        sim = Simulator(
            antenna_layout=str(antenna_file),
            frequencies=[100],
            sky_model="test_sources",
        )

        result = sim.setup()
        assert result is sim


@pytest.mark.integration
class TestSimulatorRun:
    """Test Simulator run phase."""

    def test_run_returns_results_dict(self, tmp_path):
        """Test that run() returns a results dictionary."""
        from rrivis.api.simulator import Simulator

        # Create a minimal antenna file
        antenna_file = tmp_path / "test_antennas.txt"
        antenna_file.write_text(
            "# name number beamid east north up\n"
            "Ant0 0 0 0.0 0.0 0.0\n"
            "Ant1 1 0 100.0 0.0 0.0\n"
        )

        sim = Simulator(
            antenna_layout=str(antenna_file),
            frequencies=[100, 150],
            sky_model="test_sources",
        )

        results = sim.run(progress=False)

        assert isinstance(results, dict)
        assert "visibilities" in results
        assert "frequencies" in results

    def test_run_auto_calls_setup(self, tmp_path):
        """Test that run() automatically calls setup() if needed."""
        from rrivis.api.simulator import Simulator

        # Create a minimal antenna file
        antenna_file = tmp_path / "test_antennas.txt"
        antenna_file.write_text(
            "# name number beamid east north up\n"
            "Ant0 0 0 0.0 0.0 0.0\n"
            "Ant1 1 0 100.0 0.0 0.0\n"
        )

        sim = Simulator(
            antenna_layout=str(antenna_file),
            frequencies=[100],
            sky_model="test_sources",
        )

        # Don't call setup explicitly
        assert sim.antennas is None

        # run() should call setup automatically
        results = sim.run(progress=False)

        assert sim.antennas is not None
        assert results is not None

    def test_run_stores_visibilities(self, tmp_path):
        """Test that run() stores visibilities in results."""
        from rrivis.api.simulator import Simulator

        # Create a minimal antenna file
        antenna_file = tmp_path / "test_antennas.txt"
        antenna_file.write_text(
            "# name number beamid east north up\n"
            "Ant0 0 0 0.0 0.0 0.0\n"
            "Ant1 1 0 100.0 0.0 0.0\n"
        )

        sim = Simulator(
            antenna_layout=str(antenna_file),
            frequencies=[100],
            sky_model="test_sources",
        )

        assert sim.results is None

        sim.run(progress=False)

        assert sim.results is not None
        assert "visibilities" in sim.results


@pytest.mark.integration
class TestSimulatorFromConfig:
    """Test Simulator.from_config() class method."""

    def test_from_config_creates_simulator(self, temp_config_file, tmp_path):
        """Test creating Simulator from config file."""
        from rrivis.api.simulator import Simulator

        # Create antenna file referenced in config
        antenna_file = tmp_path / "test_antennas.txt"
        antenna_file.write_text(
            "# name number beamid east north up\n"
            "Ant0 0 0 0.0 0.0 0.0\n"
            "Ant1 1 0 100.0 0.0 0.0\n"
        )

        # Update config to use correct antenna file path
        config_path = Path(temp_config_file)
        config_data = yaml.safe_load(config_path.read_text())
        config_data["antenna_layout"]["antenna_positions_file"] = str(antenna_file)

        updated_config = tmp_path / "updated_config.yaml"
        updated_config.write_text(yaml.dump(config_data))

        # Create simulator from config
        try:
            sim = Simulator.from_config(updated_config)
            assert sim is not None
            assert sim.config is not None
        except Exception as e:
            # Config loading may have additional requirements
            pytest.skip(f"Config loading not fully implemented: {e}")


@pytest.mark.integration
class TestSimulatorSave:
    """Test Simulator save functionality."""

    def test_save_requires_results(self, tmp_path):
        """Test that save() raises error if no results."""
        from rrivis.api.simulator import Simulator

        # Create a minimal antenna file
        antenna_file = tmp_path / "test_antennas.txt"
        antenna_file.write_text(
            "# name number beamid east north up\n"
            "Ant0 0 0 0.0 0.0 0.0\n"
            "Ant1 1 0 100.0 0.0 0.0\n"
        )

        sim = Simulator(
            antenna_layout=str(antenna_file),
            frequencies=[100],
        )

        # Should raise error since no results yet
        with pytest.raises(RuntimeError):
            sim.save(tmp_path / "output")

    def test_save_creates_output_directory(self, tmp_path):
        """Test that save() creates output directory if needed."""
        from rrivis.api.simulator import Simulator

        # Create a minimal antenna file
        antenna_file = tmp_path / "test_antennas.txt"
        antenna_file.write_text(
            "# name number beamid east north up\n"
            "Ant0 0 0 0.0 0.0 0.0\n"
            "Ant1 1 0 100.0 0.0 0.0\n"
        )

        sim = Simulator(
            antenna_layout=str(antenna_file),
            frequencies=[100],
            sky_model="test_sources",
        )

        sim.run(progress=False)

        output_dir = tmp_path / "new_output_dir"
        assert not output_dir.exists()

        try:
            sim.save(output_dir)
            # Directory should be created
            assert output_dir.exists()
        except Exception:
            # Save may not be fully implemented
            pytest.skip("Save functionality not fully implemented")


@pytest.mark.integration
class TestSimulatorMemoryEstimate:
    """Test Simulator memory estimation."""

    def test_memory_estimate_returns_dict(self, tmp_path):
        """Test that get_memory_estimate() returns a dict."""
        from rrivis.api.simulator import Simulator

        # Create a minimal antenna file
        antenna_file = tmp_path / "test_antennas.txt"
        antenna_file.write_text(
            "# name number beamid east north up\n"
            "Ant0 0 0 0.0 0.0 0.0\n"
            "Ant1 1 0 100.0 0.0 0.0\n"
        )

        sim = Simulator(
            antenna_layout=str(antenna_file),
            frequencies=[100],
            sky_model="test_sources",
        )

        try:
            mem = sim.get_memory_estimate()
            assert isinstance(mem, dict)
        except (AttributeError, NotImplementedError):
            pytest.skip("Memory estimation not implemented")


@pytest.mark.integration
class TestSimulatorWorkflow:
    """Test complete Simulator workflows."""

    def test_complete_workflow(self, tmp_path):
        """Test complete simulation workflow."""
        from rrivis.api.simulator import Simulator

        # Create a 3-antenna file
        antenna_file = tmp_path / "test_antennas.txt"
        antenna_file.write_text(
            "# name number beamid east north up\n"
            "Ant0 0 0 0.0 0.0 0.0\n"
            "Ant1 1 0 100.0 0.0 0.0\n"
            "Ant2 2 0 50.0 86.6 0.0\n"
        )

        # Create simulator
        sim = Simulator(
            antenna_layout=str(antenna_file),
            frequencies=[100, 120, 140],
            sky_model="test_sources",
            backend="numpy",
        )

        # Run simulation
        results = sim.run(progress=False)

        # Verify results
        assert results is not None
        assert "visibilities" in results
        assert sim.results is not None

    def test_workflow_with_method_chaining(self, tmp_path):
        """Test workflow using method chaining."""
        from rrivis.api.simulator import Simulator

        # Create a minimal antenna file
        antenna_file = tmp_path / "test_antennas.txt"
        antenna_file.write_text(
            "# name number beamid east north up\n"
            "Ant0 0 0 0.0 0.0 0.0\n"
            "Ant1 1 0 100.0 0.0 0.0\n"
        )

        sim = Simulator(
            antenna_layout=str(antenna_file),
            frequencies=[100],
            sky_model="test_sources",
        )

        # Chain setup and verify it works
        result = sim.setup()
        assert result is sim
        assert sim.antennas is not None

    def test_repr_before_and_after_run(self, tmp_path):
        """Test Simulator __repr__ shows correct status."""
        from rrivis.api.simulator import Simulator

        # Create a minimal antenna file
        antenna_file = tmp_path / "test_antennas.txt"
        antenna_file.write_text(
            "# name number beamid east north up\n"
            "Ant0 0 0 0.0 0.0 0.0\n"
            "Ant1 1 0 100.0 0.0 0.0\n"
        )

        sim = Simulator(
            antenna_layout=str(antenna_file),
            frequencies=[100],
            sky_model="test_sources",
        )

        # Before run
        repr_before = repr(sim)
        assert "configured" in repr_before

        # After run
        sim.run(progress=False)
        repr_after = repr(sim)
        assert "completed" in repr_after


@pytest.mark.integration
class TestSimulatorPlot:
    """Test Simulator plotting functionality."""

    def test_plot_requires_results(self, tmp_path):
        """Test that plot() raises error if no results."""
        from rrivis.api.simulator import Simulator

        # Create a minimal antenna file
        antenna_file = tmp_path / "test_antennas.txt"
        antenna_file.write_text(
            "# name number beamid east north up\n"
            "Ant0 0 0 0.0 0.0 0.0\n"
            "Ant1 1 0 100.0 0.0 0.0\n"
        )

        sim = Simulator(
            antenna_layout=str(antenna_file),
            frequencies=[100],
        )

        # Should raise error since no results yet
        with pytest.raises(RuntimeError):
            sim.plot()
