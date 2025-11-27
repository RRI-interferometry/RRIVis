"""Pydantic-based configuration management for RRIvis.

Provides type-safe configuration loading, validation, and serialization.
"""

from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union
from pydantic import BaseModel, Field, field_validator, model_validator
import yaml


class TelescopeConfig(BaseModel):
    """Telescope configuration."""

    telescope_name: str = Field("Unknown", description="Telescope name (e.g., HERA, MWA)")
    use_pyuvdata_telescope: bool = Field(False, description="Load telescope from pyuvdata")
    use_pyuvdata_location: bool = Field(False, description="Use pyuvdata location")
    use_pyuvdata_antennas: bool = Field(False, description="Use pyuvdata antenna positions")
    use_pyuvdata_diameters: bool = Field(False, description="Use pyuvdata antenna diameters")


class AntennaLayoutConfig(BaseModel):
    """Antenna layout configuration."""

    antenna_positions_file: Optional[str] = Field(
        None, description="Path to antenna positions file"
    )
    antenna_file_format: Literal[
        "rrivis", "casa", "measurement_set", "uvfits", "mwa", "pyuvdata"
    ] = Field("rrivis", description="Antenna file format")
    all_antenna_type: str = Field(
        "parabolic_10db_taper",
        description="Default antenna type for all antennas",
    )
    use_different_antenna_types: bool = Field(
        False, description="Use per-antenna types"
    )
    antenna_types: Dict[str, str] = Field(
        default_factory=dict, description="Per-antenna type mapping"
    )
    all_antenna_diameter: float = Field(
        14.0, gt=0, description="Default antenna diameter (meters)"
    )
    use_different_diameters: bool = Field(
        False, description="Use per-antenna diameters"
    )
    diameters: Dict[str, float] = Field(
        default_factory=dict, description="Per-antenna diameter mapping"
    )
    fixed_HPBW: Optional[float] = Field(
        None, description="Fixed HPBW override (radians)"
    )


class FeedsConfig(BaseModel):
    """Feed configuration."""

    use_polarized_feeds: bool = Field(False, description="Enable polarized feeds")
    polarization_type: str = Field("", description="Polarization type")
    use_different_polarization_type: bool = Field(
        False, description="Per-antenna polarization"
    )
    polarization_per_antenna: Dict[str, str] = Field(default_factory=dict)
    use_different_feed_types: bool = Field(False, description="Per-antenna feed types")
    all_feed_type: str = Field("", description="Default feed type")
    feed_types_per_antenna: Dict[str, str] = Field(default_factory=dict)


class BeamsConfig(BaseModel):
    """Beam configuration."""

    beam_mode: Literal["analytic", "shared", "per_antenna"] = Field(
        "analytic", description="Beam mode"
    )
    beam_file: Optional[str] = Field(None, description="Beam FITS file path")
    beam_assignment: Literal["from_layout", "from_config"] = Field(
        "from_layout", description="Beam assignment method"
    )
    antenna_beam_map: Dict[str, str] = Field(
        default_factory=dict, description="Antenna to beam file mapping"
    )
    beam_za_max_deg: float = Field(
        90.0, ge=0, le=180, description="Max zenith angle (degrees)"
    )
    beam_za_buffer_deg: float = Field(5.0, ge=0, description="ZA buffer (degrees)")
    beam_freq_buffer_hz: float = Field(1e6, ge=0, description="Frequency buffer (Hz)")
    use_different_beam_responses: bool = Field(
        False, description="Per-antenna beam responses"
    )
    all_beam_response: Literal["gaussian", "airy", "cosine", "exponential"] = Field(
        "gaussian", description="Default beam response pattern"
    )
    beam_response_per_antenna: Dict[str, str] = Field(default_factory=dict)
    cosine_taper_exponent: float = Field(1.0, ge=0, description="Cosine taper exponent")
    exponential_taper_dB: float = Field(10.0, ge=0, description="Exponential taper (dB)")


class BaselineSelectionConfig(BaseModel):
    """Baseline selection configuration."""

    use_autocorrelations: bool = Field(True, description="Include autocorrelations")
    use_crosscorrelations: bool = Field(True, description="Include crosscorrelations")
    only_selective_baseline_length: bool = Field(
        False, description="Filter by baseline length"
    )
    selective_baseline_lengths: List[float] = Field(
        default_factory=list, description="Selected baseline lengths"
    )
    selective_baseline_tolerance_meters: float = Field(
        2.0, ge=0, description="Baseline length tolerance (m)"
    )
    trim_by_angle_ranges: bool = Field(False, description="Filter by angle")
    selective_angle_ranges_deg: List[List[float]] = Field(
        default_factory=list, description="Angle ranges [min, max] in degrees"
    )


class LocationConfig(BaseModel):
    """Observatory location configuration."""

    lat: Union[float, str] = Field("", description="Latitude (degrees)")
    lon: Union[float, str] = Field("", description="Longitude (degrees)")
    height: Union[float, str] = Field("", description="Height (meters)")


class TestSourcesConfig(BaseModel):
    """Test sources configuration."""

    use_test_sources: bool = Field(False, description="Use test sources")
    num_sources: int = Field(100, ge=1, description="Number of test sources")
    flux_limit: float = Field(50.0, ge=0, description="Flux limit (Jy)")
    nside: int = Field(32, ge=1, description="HEALPix nside")


class GSMConfig(BaseModel):
    """GSM configuration."""

    use_gsm: bool = Field(False, description="Use GSM")
    gsm_catalogue: str = Field("gsm2008", description="GSM catalogue")
    flux_limit: float = Field(50.0, ge=0, description="Flux limit (Jy)")
    nside: int = Field(32, ge=1, description="HEALPix nside")


class GLEAMConfig(BaseModel):
    """GLEAM configuration."""

    use_gleam: bool = Field(False, description="Use GLEAM")
    gleam_catalogue: str = Field("VIII/105/catalog", description="GLEAM catalogue")
    flux_limit: float = Field(50.0, ge=0, description="Flux limit (Jy)")
    nside: int = Field(32, ge=1, description="HEALPix nside")


class SkyModelConfig(BaseModel):
    """Sky model configuration."""

    test_sources: TestSourcesConfig = Field(default_factory=TestSourcesConfig)
    test_sources_healpix: TestSourcesConfig = Field(default_factory=TestSourcesConfig)
    gsm_healpix: GSMConfig = Field(default_factory=GSMConfig)
    gleam: GLEAMConfig = Field(default_factory=GLEAMConfig)
    gleam_healpix: GLEAMConfig = Field(default_factory=GLEAMConfig)


class ObsTimeConfig(BaseModel):
    """Observation time configuration."""

    time_interval: float = Field(1.0, gt=0, description="Time interval")
    time_interval_unit: Literal["seconds", "minutes", "hours", "days"] = Field(
        "hours", description="Time interval unit"
    )
    total_duration: float = Field(1.0, gt=0, description="Total duration")
    total_duration_unit: Literal["seconds", "minutes", "hours", "days"] = Field(
        "days", description="Duration unit"
    )
    start_time: str = Field(
        "2025-01-01T00:00:00", description="Start time (ISO format)"
    )


class ObsFrequencyConfig(BaseModel):
    """Observation frequency configuration."""

    starting_frequency: float = Field(
        50.0, gt=0, description="Starting frequency"
    )
    frequency_interval: float = Field(
        1.0, gt=0, description="Frequency interval"
    )
    frequency_bandwidth: float = Field(
        100.0, gt=0, description="Frequency bandwidth"
    )
    frequency_unit: Literal["Hz", "kHz", "MHz", "GHz"] = Field(
        "MHz", description="Frequency unit"
    )

    @property
    def n_channels(self) -> int:
        """Calculate number of frequency channels."""
        return max(1, int(self.frequency_bandwidth / self.frequency_interval))


class OutputConfig(BaseModel):
    """Output configuration."""

    simulation_data_dir: str = Field("", description="Output directory")
    output_file_name: str = Field("complex_visibility", description="Output filename")
    output_file_format: Literal["HDF5", "JSON", "MS", "UVFITS"] = Field(
        "HDF5", description="Output format"
    )
    save_simulation_data: bool = Field(False, description="Save simulation data")
    plot_results_in_bokeh: bool = Field(True, description="Plot with Bokeh")
    plot_skymodel_every_hour: bool = Field(True, description="Plot sky model")
    save_log_data: bool = Field(False, description="Save log data")
    angle_unit: Literal["degrees", "radians", ""] = Field(
        "", description="Angle display unit"
    )
    skymodel_frequency: float = Field(150.0, gt=0, description="Sky model plot frequency")


class SimulatorsConfig(BaseModel):
    """Simulator configuration."""

    use_different_simulator_for_cross_check: bool = Field(
        False, description="Use alternative simulator"
    )
    name: str = Field("", description="Simulator name")


class RRIvisConfig(BaseModel):
    """Main RRIvis configuration with validation.

    This is the top-level configuration model that contains all
    configuration sections for running a visibility simulation.

    Examples:
        >>> config = RRIvisConfig.from_yaml("config.yaml")
        >>> print(config.antenna_layout.all_antenna_diameter)
        14.0
    """

    telescope: TelescopeConfig = Field(default_factory=TelescopeConfig)
    antenna_layout: AntennaLayoutConfig = Field(default_factory=AntennaLayoutConfig)
    feeds: FeedsConfig = Field(default_factory=FeedsConfig)
    beams: BeamsConfig = Field(default_factory=BeamsConfig)
    baseline_selection: BaselineSelectionConfig = Field(
        default_factory=BaselineSelectionConfig
    )
    location: LocationConfig = Field(default_factory=LocationConfig)
    sky_model: SkyModelConfig = Field(default_factory=SkyModelConfig)
    obs_time: ObsTimeConfig = Field(default_factory=ObsTimeConfig)
    obs_frequency: ObsFrequencyConfig = Field(default_factory=ObsFrequencyConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    simulators: SimulatorsConfig = Field(default_factory=SimulatorsConfig)

    model_config = {
        "extra": "allow",  # Allow extra fields for forward compatibility
        "validate_assignment": True,  # Validate on attribute assignment
    }

    @classmethod
    def from_yaml(cls, yaml_path: Union[str, Path]) -> "RRIvisConfig":
        """
        Load configuration from YAML file with validation.

        Args:
            yaml_path: Path to YAML configuration file

        Returns:
            Validated RRIvisConfig instance

        Raises:
            ValueError: If configuration is invalid
            FileNotFoundError: If file doesn't exist
        """
        yaml_path = Path(yaml_path)
        if not yaml_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {yaml_path}")

        with open(yaml_path, "r") as f:
            data = yaml.safe_load(f) or {}

        try:
            return cls(**data)
        except Exception as e:
            raise ValueError(
                f"Invalid configuration in {yaml_path}:\n{e}\n\n"
                f"See documentation for valid config format."
            )

    def to_yaml(self, output_path: Union[str, Path]) -> None:
        """
        Export configuration to YAML file.

        Args:
            output_path: Path to write YAML file
        """
        output_path = Path(output_path)
        with open(output_path, "w") as f:
            yaml.dump(
                self.model_dump(exclude_none=True),
                f,
                default_flow_style=False,
                sort_keys=False,
            )

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return self.model_dump()


def load_config(config_path: Union[str, Path]) -> RRIvisConfig:
    """
    Load and validate configuration from file.

    Args:
        config_path: Path to YAML configuration file

    Returns:
        Validated RRIvisConfig instance
    """
    return RRIvisConfig.from_yaml(config_path)


def create_default_config(output_path: Union[str, Path]) -> None:
    """
    Create a default configuration file.

    Args:
        output_path: Path to write default config
    """
    config = RRIvisConfig()
    config.to_yaml(output_path)
