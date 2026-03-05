# tests/unit/test_io/test_config.py
"""
Unit tests for Pydantic configuration validation.

Tests the RRIvisConfig model and its submodels for proper
validation, serialization, and deserialization.
"""

import pytest
import yaml
from pydantic import ValidationError

from rrivis.io.config import (
    AntennaLayoutConfig,
    BeamsConfig,
    LocationConfig,
    ObsFrequencyConfig,
    ObsTimeConfig,
    OutputConfig,
    RRIvisConfig,
    SkyModelConfig,
    TelescopeConfig,
    TestSourcesConfig,
    VisibilityConfig,
    create_default_config,
    load_config,
)

# =============================================================================
# TelescopeConfig Tests
# =============================================================================


class TestTelescopeConfig:
    """Test TelescopeConfig validation."""

    def test_default_values(self):
        """Test default telescope configuration."""
        config = TelescopeConfig()
        assert config.telescope_name == "Unknown"
        assert config.use_pyuvdata_telescope is False
        assert config.use_pyuvdata_location is False
        assert config.use_pyuvdata_antennas is False
        assert config.use_pyuvdata_diameters is False

    def test_custom_values(self):
        """Test custom telescope configuration."""
        config = TelescopeConfig(
            telescope_name="HERA",
            use_pyuvdata_telescope=True,
        )
        assert config.telescope_name == "HERA"
        assert config.use_pyuvdata_telescope is True


# =============================================================================
# AntennaLayoutConfig Tests
# =============================================================================


class TestAntennaLayoutConfig:
    """Test AntennaLayoutConfig validation."""

    def test_default_values(self):
        """Test default antenna layout configuration (all optional fields default to None)."""
        config = AntennaLayoutConfig()
        assert config.antenna_positions_file is None
        assert config.antenna_file_format is None
        assert config.all_antenna_diameter is None

    def test_valid_file_formats(self):
        """Test valid antenna file formats."""
        for fmt in ["rrivis", "casa", "measurement_set", "uvfits", "mwa", "pyuvdata"]:
            config = AntennaLayoutConfig(antenna_file_format=fmt)
            assert config.antenna_file_format == fmt

    def test_invalid_file_format(self):
        """Test invalid antenna file format raises error."""
        with pytest.raises(ValidationError):
            AntennaLayoutConfig(antenna_file_format="invalid_format")

    def test_diameter_validation(self):
        """Test antenna diameter validation via validate()."""
        config = RRIvisConfig(
            antenna_layout=AntennaLayoutConfig(all_antenna_diameter=-1.0)
        )
        errors = config.validate()
        assert any("all_antenna_diameter" in e and "must be > 0" in e for e in errors)

        config2 = RRIvisConfig(
            antenna_layout=AntennaLayoutConfig(all_antenna_diameter=0)
        )
        errors2 = config2.validate()
        assert any("all_antenna_diameter" in e and "must be > 0" in e for e in errors2)


# =============================================================================
# BeamsConfig Tests
# =============================================================================


class TestBeamsConfig:
    """Test BeamsConfig validation."""

    def test_default_values(self):
        """Test default beam configuration (optional fields default to None)."""
        config = BeamsConfig()
        assert config.beam_mode is None
        assert config.all_beam_response is None
        assert config.beam_za_max_deg is None

    def test_valid_beam_modes(self):
        """Test valid beam modes."""
        for mode in ["analytic", "shared", "per_antenna"]:
            config = BeamsConfig(beam_mode=mode)
            assert config.beam_mode == mode

    def test_valid_beam_responses(self):
        """Test valid beam response patterns."""
        for response in ["gaussian", "airy", "cosine", "exponential"]:
            config = BeamsConfig(all_beam_response=response)
            assert config.all_beam_response == response

    def test_zenith_angle_set(self):
        """Test zenith angle can be set."""
        config = BeamsConfig(beam_za_max_deg=45.0)
        assert config.beam_za_max_deg == 45.0


# =============================================================================
# LocationConfig Tests
# =============================================================================


class TestLocationConfig:
    """Test LocationConfig validation."""

    def test_default_values(self):
        """Test default location configuration."""
        config = LocationConfig()
        assert config.lat == ""
        assert config.lon == ""
        assert config.height == ""

    def test_numeric_values(self):
        """Test numeric location values."""
        config = LocationConfig(lat=-30.7, lon=21.4, height=1073.0)
        assert config.lat == -30.7
        assert config.lon == 21.4
        assert config.height == 1073.0

    def test_string_values(self):
        """Test string location values."""
        config = LocationConfig(lat="-30.7", lon="21.4", height="1073")
        assert config.lat == "-30.7"


# =============================================================================
# ObsFrequencyConfig Tests
# =============================================================================


class TestObsFrequencyConfig:
    """Test ObsFrequencyConfig validation."""

    def test_default_values(self):
        """Test default frequency configuration (numeric fields default to None)."""
        config = ObsFrequencyConfig()
        assert config.starting_frequency is None
        assert config.frequency_interval is None
        assert config.frequency_bandwidth is None
        assert config.frequency_unit == "MHz"

    def test_n_channels_property(self):
        """Test frequency channel count calculation.

        n_channels = int(bandwidth / interval) + 1, matching np.linspace
        which includes both endpoints. bw=10, interval=1 → 11 channels.
        """
        config = ObsFrequencyConfig(
            starting_frequency=100.0,
            frequency_interval=1.0,
            frequency_bandwidth=10.0,
        )
        assert config.n_channels == 11

    def test_valid_frequency_units(self):
        """Test valid frequency units."""
        for unit in ["Hz", "kHz", "MHz", "GHz"]:
            config = ObsFrequencyConfig(frequency_unit=unit)
            assert config.frequency_unit == unit

    def test_invalid_frequency_unit(self):
        """Test invalid frequency unit raises error."""
        with pytest.raises(ValidationError):
            ObsFrequencyConfig(frequency_unit="THz")

    def test_positive_frequencies(self):
        """Test frequency validation via validate()."""
        config = RRIvisConfig(
            obs_frequency=ObsFrequencyConfig(
                starting_frequency=-100.0,
                frequency_interval=1.0,
                frequency_bandwidth=10.0,
            )
        )
        errors = config.validate()
        assert any("starting_frequency" in e and "must be > 0" in e for e in errors)

        config2 = RRIvisConfig(
            obs_frequency=ObsFrequencyConfig(
                starting_frequency=100.0,
                frequency_interval=0,
                frequency_bandwidth=10.0,
            )
        )
        errors2 = config2.validate()
        assert any("frequency_interval" in e and "must be > 0" in e for e in errors2)


# =============================================================================
# ObsTimeConfig Tests
# =============================================================================


class TestObsTimeConfig:
    """Test ObsTimeConfig validation."""

    def test_default_values(self):
        """Test default time configuration (fields default to None)."""
        config = ObsTimeConfig()
        assert config.start_time is None
        assert config.duration_seconds is None
        assert config.time_step_seconds is None

    def test_custom_values(self):
        """Test custom time configuration."""
        config = ObsTimeConfig(
            start_time="2023-06-21T12:00:00",
            duration_seconds=7200.0,
            time_step_seconds=30.0,
        )
        assert config.start_time == "2023-06-21T12:00:00"
        assert config.duration_seconds == 7200.0
        assert config.time_step_seconds == 30.0

    def test_positive_duration_required(self):
        """Test that negative duration is caught by validate()."""
        config = RRIvisConfig(
            obs_time=ObsTimeConfig(
                start_time="2023-06-21T12:00:00",
                duration_seconds=-1.0,
                time_step_seconds=60.0,
            )
        )
        errors = config.validate()
        assert any("duration_seconds" in e and "must be > 0" in e for e in errors)

    def test_positive_time_step_required(self):
        """Test that negative time step is caught by validate()."""
        config = RRIvisConfig(
            obs_time=ObsTimeConfig(
                start_time="2023-06-21T12:00:00",
                duration_seconds=3600.0,
                time_step_seconds=-1.0,
            )
        )
        errors = config.validate()
        assert any("time_step_seconds" in e and "must be > 0" in e for e in errors)


# =============================================================================
# SkyModelConfig Tests
# =============================================================================


class TestSkyModelConfig:
    """Test SkyModelConfig validation."""

    def test_default_values(self):
        """Test default sky model configuration."""
        config = SkyModelConfig()
        assert config.test_sources.use_test_sources is False
        assert config.gleam.use_gleam is False
        assert config.gsm_healpix.use_gsm is False

    def test_nested_test_sources(self):
        """Test nested test sources configuration."""
        config = SkyModelConfig(
            test_sources=TestSourcesConfig(
                use_test_sources=True,
                num_sources=500,
                flux_min=2.0,
                flux_max=100.0,
                dec_deg=-30.72,
                spectral_index=-0.8,
            )
        )
        assert config.test_sources.use_test_sources is True
        assert config.test_sources.num_sources == 500
        assert config.test_sources.flux_max == 100.0


# =============================================================================
# OutputConfig Tests
# =============================================================================


class TestOutputConfig:
    """Test OutputConfig validation."""

    def test_default_values(self):
        """Test default output configuration."""
        config = OutputConfig()
        assert config.output_file_format is None
        assert config.save_simulation_data is False

    def test_valid_output_formats(self):
        """Test valid output formats."""
        for fmt in ["HDF5", "JSON", "MS", "UVFITS"]:
            config = OutputConfig(output_file_format=fmt)
            assert config.output_file_format == fmt


# =============================================================================
# RRIvisConfig Tests
# =============================================================================


class TestRRIvisConfig:
    """Test main RRIvisConfig model."""

    def test_default_config(self):
        """Test creating default configuration (optional fields are None)."""
        config = RRIvisConfig()
        assert config.telescope.telescope_name == "Unknown"
        assert config.antenna_layout.all_antenna_diameter is None
        assert config.beams.beam_mode is None
        assert config.obs_frequency.frequency_unit == "MHz"

    def test_to_dict(self):
        """Test converting config to dictionary."""
        config = RRIvisConfig()
        data = config.to_dict()
        assert isinstance(data, dict)
        assert "telescope" in data
        assert "antenna_layout" in data
        assert "beams" in data

    def test_to_yaml(self, tmp_path):
        """Test exporting config to YAML."""
        config = RRIvisConfig()
        output_path = tmp_path / "test_config.yaml"
        config.to_yaml(output_path)

        assert output_path.exists()

        # Verify content
        with open(output_path) as f:
            data = yaml.safe_load(f)
        assert "telescope" in data
        assert "antenna_layout" in data

    def test_from_yaml(self, tmp_path):
        """Test loading config from YAML."""
        # Create a test YAML file
        config_content = """
telescope:
  telescope_name: HERA
antenna_layout:
  all_antenna_diameter: 14.0
  antenna_file_format: rrivis
obs_frequency:
  starting_frequency: 100.0
  frequency_bandwidth: 50.0
  frequency_unit: MHz
"""
        config_path = tmp_path / "test_config.yaml"
        config_path.write_text(config_content)

        config = RRIvisConfig.from_yaml(config_path)
        assert config.telescope.telescope_name == "HERA"
        assert config.antenna_layout.all_antenna_diameter == 14.0
        assert config.obs_frequency.starting_frequency == 100.0

    def test_from_yaml_missing_file(self):
        """Test loading from non-existent file raises error."""
        with pytest.raises(FileNotFoundError):
            RRIvisConfig.from_yaml("/nonexistent/path/config.yaml")

    def test_from_yaml_invalid_content(self, tmp_path):
        """Test loading invalid YAML raises error."""
        config_path = tmp_path / "invalid_config.yaml"
        config_path.write_text("""
antenna_layout:
  antenna_file_format: invalid_format_type
""")

        with pytest.raises(ValueError):
            RRIvisConfig.from_yaml(config_path)

    def test_extra_fields_allowed(self, tmp_path):
        """Test that extra fields are allowed for forward compatibility."""
        config_content = """
telescope:
  telescope_name: HERA
future_feature:
  some_option: true
"""
        config_path = tmp_path / "test_config.yaml"
        config_path.write_text(config_content)

        # Should not raise error
        config = RRIvisConfig.from_yaml(config_path)
        assert config.telescope.telescope_name == "HERA"


# =============================================================================
# Utility Function Tests
# =============================================================================


class TestConfigUtilities:
    """Test configuration utility functions."""

    def test_load_config(self, tmp_path):
        """Test load_config utility function."""
        config_content = """
telescope:
  telescope_name: MWA
"""
        config_path = tmp_path / "test_config.yaml"
        config_path.write_text(config_content)

        config = load_config(config_path)
        assert config.telescope.telescope_name == "MWA"

    def test_create_default_config(self, tmp_path):
        """Test create_default_config utility function."""
        output_path = tmp_path / "default_config.yaml"
        create_default_config(output_path)

        assert output_path.exists()

        # Verify it's valid
        config = load_config(output_path)
        assert config.telescope.telescope_name == "Unknown"


# =============================================================================
# Validation Edge Cases
# =============================================================================


class TestValidationEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_yaml(self, tmp_path):
        """Test loading empty YAML creates default config."""
        config_path = tmp_path / "empty.yaml"
        config_path.write_text("")

        config = RRIvisConfig.from_yaml(config_path)
        assert config.telescope.telescope_name == "Unknown"

    def test_partial_yaml(self, tmp_path):
        """Test loading partial YAML fills defaults."""
        config_content = """
telescope:
  telescope_name: SKA
"""
        config_path = tmp_path / "partial.yaml"
        config_path.write_text(config_content)

        config = RRIvisConfig.from_yaml(config_path)
        assert config.telescope.telescope_name == "SKA"
        # Other sections should have defaults (None for optional fields)
        assert config.antenna_layout.all_antenna_diameter is None

    def test_validate_blank_config_returns_all_required_errors(self):
        """Blank config should report all required-field errors."""
        config = RRIvisConfig()
        errors = config.validate()
        assert len(errors) == 11
        assert any("antenna_positions_file" in e for e in errors)
        assert any("antenna_file_format" in e for e in errors)
        assert any("all_antenna_type" in e for e in errors)
        assert any("all_antenna_diameter" in e for e in errors)
        assert any("obs_time.start_time" in e for e in errors)
        assert any("duration_seconds" in e for e in errors)
        assert any("time_step_seconds" in e for e in errors)
        assert any("starting_frequency" in e for e in errors)
        assert any("frequency_interval" in e for e in errors)
        assert any("frequency_bandwidth" in e for e in errors)
        assert any("sky_representation" in e for e in errors)

    def test_validate_valid_config_returns_empty(self, tmp_path):
        """Fully populated config should return no errors."""
        antenna_file = tmp_path / "antennas.txt"
        antenna_file.write_text("0 0 0\n")
        config = RRIvisConfig(
            antenna_layout=AntennaLayoutConfig(
                antenna_positions_file=str(antenna_file),
                antenna_file_format="rrivis",
                all_antenna_type="parabolic",
                all_antenna_diameter=14.0,
            ),
            obs_time=ObsTimeConfig(
                start_time="2025-01-01T00:00:00",
                duration_seconds=3600.0,
                time_step_seconds=60.0,
            ),
            obs_frequency=ObsFrequencyConfig(
                starting_frequency=100.0,
                frequency_interval=1.0,
                frequency_bandwidth=50.0,
            ),
            visibility=VisibilityConfig(sky_representation="point_sources"),
        )
        assert config.validate() == []

    def test_validate_missing_file(self, tmp_path):
        """Non-existent antenna file should be reported."""
        config = RRIvisConfig(
            antenna_layout=AntennaLayoutConfig(
                antenna_positions_file="/no/such/file.txt",
                antenna_file_format="rrivis",
                all_antenna_type="parabolic",
                all_antenna_diameter=14.0,
            ),
            obs_time=ObsTimeConfig(
                start_time="2025-01-01T00:00:00",
                duration_seconds=3600.0,
                time_step_seconds=60.0,
            ),
            obs_frequency=ObsFrequencyConfig(
                starting_frequency=100.0,
                frequency_interval=1.0,
                frequency_bandwidth=50.0,
            ),
            visibility=VisibilityConfig(sky_representation="point_sources"),
        )
        errors = config.validate()
        assert len(errors) == 1
        assert "file not found" in errors[0]

    def test_validate_cross_field_freq_interval(self, tmp_path):
        """frequency_interval >= frequency_bandwidth should be caught."""
        antenna_file = tmp_path / "antennas.txt"
        antenna_file.write_text("0 0 0\n")
        config = RRIvisConfig(
            antenna_layout=AntennaLayoutConfig(
                antenna_positions_file=str(antenna_file),
                antenna_file_format="rrivis",
                all_antenna_diameter=14.0,
            ),
            obs_time=ObsTimeConfig(
                start_time="2025-01-01T00:00:00",
                duration_seconds=3600.0,
                time_step_seconds=60.0,
            ),
            obs_frequency=ObsFrequencyConfig(
                starting_frequency=100.0,
                frequency_interval=50.0,
                frequency_bandwidth=10.0,
            ),
        )
        errors = config.validate()
        assert any("frequency_interval must be <" in e for e in errors)

    def test_validate_location_out_of_range(self, tmp_path):
        """Out-of-range lat/lon/height should be reported."""
        antenna_file = tmp_path / "antennas.txt"
        antenna_file.write_text("0 0 0\n")
        config = RRIvisConfig(
            antenna_layout=AntennaLayoutConfig(
                antenna_positions_file=str(antenna_file),
                antenna_file_format="rrivis",
                all_antenna_diameter=14.0,
            ),
            obs_time=ObsTimeConfig(
                start_time="2025-01-01T00:00:00",
                duration_seconds=3600.0,
                time_step_seconds=60.0,
            ),
            obs_frequency=ObsFrequencyConfig(
                starting_frequency=100.0,
                frequency_interval=1.0,
                frequency_bandwidth=50.0,
            ),
            location=LocationConfig(lat=200.0, lon=-200.0, height=-5.0),
        )
        errors = config.validate()
        assert any("location.lat" in e for e in errors)
        assert any("location.lon" in e for e in errors)
        assert any("location.height" in e for e in errors)

    def test_validate_analytic_beam_missing_response(self, tmp_path):
        """beam_mode=analytic with no all_beam_response should be an error."""
        antenna_file = tmp_path / "antennas.txt"
        antenna_file.write_text("0 0 0\n")
        config = RRIvisConfig(
            antenna_layout=AntennaLayoutConfig(
                antenna_positions_file=str(antenna_file),
                antenna_file_format="rrivis",
                all_antenna_diameter=14.0,
            ),
            obs_time=ObsTimeConfig(
                start_time="2025-01-01T00:00:00",
                duration_seconds=3600.0,
                time_step_seconds=60.0,
            ),
            obs_frequency=ObsFrequencyConfig(
                starting_frequency=100.0,
                frequency_interval=1.0,
                frequency_bandwidth=50.0,
            ),
            beams=BeamsConfig(beam_mode="analytic", all_beam_response=None),
        )
        errors = config.validate()
        assert any("all_beam_response" in e for e in errors)

    def test_validate_invalid_start_time(self, tmp_path):
        """Garbage start_time string should be reported."""
        antenna_file = tmp_path / "antennas.txt"
        antenna_file.write_text("0 0 0\n")
        config = RRIvisConfig(
            antenna_layout=AntennaLayoutConfig(
                antenna_positions_file=str(antenna_file),
                antenna_file_format="rrivis",
                all_antenna_diameter=14.0,
            ),
            obs_time=ObsTimeConfig(
                start_time="not-a-date",
                duration_seconds=3600.0,
                time_step_seconds=60.0,
            ),
            obs_frequency=ObsFrequencyConfig(
                starting_frequency=100.0,
                frequency_interval=1.0,
                frequency_bandwidth=50.0,
            ),
        )
        errors = config.validate()
        assert any("invalid ISO format" in e for e in errors)

    def test_validate_time_step_gt_duration(self, tmp_path):
        """time_step > duration cross-field check."""
        antenna_file = tmp_path / "antennas.txt"
        antenna_file.write_text("0 0 0\n")
        config = RRIvisConfig(
            antenna_layout=AntennaLayoutConfig(
                antenna_positions_file=str(antenna_file),
                antenna_file_format="rrivis",
                all_antenna_diameter=14.0,
            ),
            obs_time=ObsTimeConfig(
                start_time="2025-01-01T00:00:00",
                duration_seconds=60.0,
                time_step_seconds=3600.0,
            ),
            obs_frequency=ObsFrequencyConfig(
                starting_frequency=100.0,
                frequency_interval=1.0,
                frequency_bandwidth=50.0,
            ),
        )
        errors = config.validate()
        assert any("time_step_seconds must be <=" in e for e in errors)

    def test_roundtrip_serialization(self, tmp_path):
        """Test config survives YAML roundtrip."""
        # Create config with custom values
        original = RRIvisConfig(
            telescope=TelescopeConfig(telescope_name="LOFAR"),
            obs_frequency=ObsFrequencyConfig(
                starting_frequency=120.0,
                frequency_bandwidth=80.0,
            ),
        )

        # Save to YAML
        yaml_path = tmp_path / "roundtrip.yaml"
        original.to_yaml(yaml_path)

        # Load back
        loaded = RRIvisConfig.from_yaml(yaml_path)

        # Verify values match
        assert loaded.telescope.telescope_name == "LOFAR"
        assert loaded.obs_frequency.starting_frequency == 120.0
        assert loaded.obs_frequency.frequency_bandwidth == 80.0
