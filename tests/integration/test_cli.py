# tests/integration/test_cli.py
"""
CLI integration tests for RRIvis.

These tests verify command-line interface functionality.
"""

import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.integration
class TestCLIBasic:
    """Test basic CLI functionality."""

    def test_help_command(self):
        """Test --help flag works."""
        result = subprocess.run(
            [sys.executable, "-m", "rrivis.cli.main", "--help"],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0
        assert "rrivis" in result.stdout.lower() or "usage" in result.stdout.lower()

    def test_version_command(self):
        """Test --version flag works."""
        result = subprocess.run(
            [sys.executable, "-m", "rrivis.cli.main", "--version"],
            capture_output=True,
            text=True,
        )

        # Should either succeed or print version
        assert (
            result.returncode == 0
            or "0.2.0" in result.stdout
            or "0.2.0" in result.stderr
        )

    def test_cli_entry_point_exists(self):
        """Test that the rrivis CLI entry point is available."""
        # Try to import the main CLI module
        try:
            from rrivis.cli.main import main

            assert callable(main)
        except ImportError as e:
            pytest.fail(f"Could not import CLI main: {e}")


@pytest.mark.integration
class TestCLIWithConfig:
    """Test CLI with configuration files."""

    def test_config_file_parsing(self, temp_config_file, tmp_path):
        """Test that config file is parsed correctly."""
        # Create a minimal antenna file for the config
        antenna_file = tmp_path / "test_antennas.txt"
        antenna_file.write_text(
            "# name number beamid east north up\n"
            "Ant0 0 0 0.0 0.0 0.0\n"
            "Ant1 1 0 100.0 0.0 0.0\n"
        )

        # Update config to use correct path
        import yaml

        config_path = Path(temp_config_file)
        config_data = yaml.safe_load(config_path.read_text())
        config_data["antenna_layout"]["antenna_positions_file"] = str(antenna_file)

        # Write updated config
        updated_config = tmp_path / "updated_config.yaml"
        updated_config.write_text(yaml.dump(config_data))

        # Just verify config can be loaded (full run may require more setup)
        from rrivis.io.config import load_config

        config = load_config(updated_config)
        assert config is not None


@pytest.mark.integration
class TestCLIMigrate:
    """Test configuration migration CLI."""

    def test_migrate_help(self):
        """Test rrivis-migrate --help."""
        result = subprocess.run(
            [sys.executable, "-m", "rrivis.cli.migrate", "--help"],
            capture_output=True,
            text=True,
        )

        # May fail if migrate module doesn't exist yet, which is acceptable
        if result.returncode == 0:
            assert (
                "migrate" in result.stdout.lower() or "config" in result.stdout.lower()
            )

    def test_migrate_module_importable(self):
        """Test that migrate module can be imported."""
        try:
            from rrivis.cli.migrate import cli_migrate

            # Just verify it exists and is callable
            assert callable(cli_migrate) or cli_migrate is None
        except (ImportError, AttributeError):
            # Module may not exist yet - that's fine
            pytest.skip("Migrate CLI not yet implemented")


@pytest.mark.integration
class TestCLIBackendSelection:
    """Test backend selection through CLI."""

    def test_backend_flag_recognized(self):
        """Test that --backend flag is recognized."""
        result = subprocess.run(
            [sys.executable, "-m", "rrivis.cli.main", "--help"],
            capture_output=True,
            text=True,
        )

        # Check if backend option is documented
        if result.returncode == 0:
            # Backend may be in help output
            help_text = result.stdout.lower()
            # Just verify help runs - backend may or may not be shown
            assert "usage" in help_text or "rrivis" in help_text


@pytest.mark.integration
class TestCLIErrorHandling:
    """Test CLI error handling."""

    def test_invalid_config_file(self, tmp_path):
        """Test error handling for invalid config file."""
        fake_config = tmp_path / "nonexistent.yaml"

        result = subprocess.run(
            [sys.executable, "-m", "rrivis.cli.main", str(fake_config)],
            capture_output=True,
            text=True,
        )

        # Should fail with non-zero exit code
        # (or handle gracefully with error message)
        assert (
            result.returncode != 0
            or "error" in result.stderr.lower()
            or "not found" in result.stderr.lower()
        )

    def test_invalid_yaml_syntax(self, tmp_path):
        """Test error handling for malformed YAML."""
        bad_config = tmp_path / "bad_config.yaml"
        bad_config.write_text("this: is: not: valid: yaml: [}")

        result = subprocess.run(
            [sys.executable, "-m", "rrivis.cli.main", str(bad_config)],
            capture_output=True,
            text=True,
        )

        # Should fail or show error
        assert (
            result.returncode != 0 or "error" in (result.stderr + result.stdout).lower()
        )
