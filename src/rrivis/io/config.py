"""
Pydantic-based configuration management for RRIvis.

This module provides type-safe configuration loading, validation, and
serialization using Pydantic models. Configuration can be loaded from
YAML files or created programmatically.

Classes
-------
RRIvisConfig
    Main configuration container with all simulation settings.
TelescopeConfig
    Telescope identification and pyuvdata integration settings.
AntennaLayoutConfig
    Antenna positions, diameters, and types.
BeamsConfig
    Beam pattern configuration (analytic or FITS).
ObsFrequencyConfig
    Observation frequency band settings.
ObsTimeConfig
    Observation timing settings.
SkyModelConfig
    Sky model selection (GLEAM, GSM, test sources).
OutputConfig
    Output file settings.

Functions
---------
load_config
    Load and validate configuration from YAML file.
create_default_config
    Create a default configuration template file.

Examples
--------
Load configuration from file:

>>> from rrivis.io.config import load_config
>>> config = load_config("simulation.yaml")
>>> print(config.telescope.telescope_name)
'HERA'

Create configuration programmatically:

>>> from rrivis.io.config import RRIvisConfig, TelescopeConfig
>>> config = RRIvisConfig(
...     telescope=TelescopeConfig(telescope_name="MWA"),
...     obs_frequency={"starting_frequency": 150.0},
... )
>>> config.to_yaml("output_config.yaml")
"""

from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field


class TelescopeConfig(BaseModel):
    """Telescope configuration."""

    telescope_name: str = Field("Unknown", description="Telescope name (e.g., HERA, MWA)")
    use_pyuvdata_telescope: bool = Field(False, description="Load telescope from pyuvdata")
    use_pyuvdata_location: bool = Field(False, description="Use pyuvdata location")
    use_pyuvdata_antennas: bool = Field(False, description="Use pyuvdata antenna positions")
    use_pyuvdata_diameters: bool = Field(False, description="Use pyuvdata antenna diameters")


class AntennaLayoutConfig(BaseModel):
    """Antenna layout configuration."""

    antenna_positions_file: str | None = Field(
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
    antenna_types: dict[str, str] = Field(
        default_factory=dict, description="Per-antenna type mapping"
    )
    all_antenna_diameter: float = Field(
        14.0, gt=0, description="Default antenna diameter (meters)"
    )
    use_different_diameters: bool = Field(
        False, description="Use per-antenna diameters"
    )
    diameters: dict[str, float] = Field(
        default_factory=dict, description="Per-antenna diameter mapping"
    )
    fixed_HPBW: float | None = Field(
        None, description="Fixed HPBW override (radians)"
    )


class FeedsConfig(BaseModel):
    """Feed configuration."""

    use_polarized_feeds: bool = Field(False, description="Enable polarized feeds")
    polarization_type: str = Field("", description="Polarization type")
    use_different_polarization_type: bool = Field(
        False, description="Per-antenna polarization"
    )
    polarization_per_antenna: dict[str, str] = Field(default_factory=dict)
    use_different_feed_types: bool = Field(False, description="Per-antenna feed types")
    all_feed_type: str = Field("", description="Default feed type")
    feed_types_per_antenna: dict[str, str] = Field(default_factory=dict)


class BeamsConfig(BaseModel):
    """Beam configuration."""

    beam_mode: Literal["analytic", "shared", "per_antenna"] = Field(
        "analytic", description="Beam mode"
    )
    beam_file: str | None = Field(None, description="Beam FITS file path")
    beam_assignment: Literal["from_layout", "from_config"] = Field(
        "from_layout", description="Beam assignment method"
    )
    antenna_beam_map: dict[str, str] = Field(
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
    beam_response_per_antenna: dict[str, str] = Field(default_factory=dict)
    cosine_taper_exponent: float = Field(1.0, ge=0, description="Cosine taper exponent")
    exponential_taper_dB: float = Field(10.0, ge=0, description="Exponential taper (dB)")


class BaselineSelectionConfig(BaseModel):
    """Baseline selection configuration."""

    use_autocorrelations: bool = Field(True, description="Include autocorrelations")
    use_crosscorrelations: bool = Field(True, description="Include crosscorrelations")
    only_selective_baseline_length: bool = Field(
        False, description="Filter by baseline length"
    )
    selective_baseline_lengths: list[float] = Field(
        default_factory=list, description="Selected baseline lengths"
    )
    selective_baseline_tolerance_meters: float = Field(
        2.0, ge=0, description="Baseline length tolerance (m)"
    )
    trim_by_angle_ranges: bool = Field(False, description="Filter by angle")
    selective_angle_ranges_deg: list[list[float]] = Field(
        default_factory=list, description="Angle ranges [min, max] in degrees"
    )


class LocationConfig(BaseModel):
    """Observatory location configuration."""

    lat: float | str = Field("", description="Latitude (degrees)")
    lon: float | str = Field("", description="Longitude (degrees)")
    height: float | str = Field("", description="Height (meters)")


class SyntheticSourcesConfig(BaseModel):
    """Synthetic/test sources configuration."""

    use_test_sources: bool = Field(False, description="Use test sources")
    num_sources: int = Field(100, ge=1, description="Number of test sources")
    flux_limit: float = Field(50.0, ge=0, description="Flux limit (Jy)")
    nside: int = Field(32, ge=1, description="HEALPix nside")


# Alias for backward compatibility
TestSourcesConfig = SyntheticSourcesConfig


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


class MALSConfig(BaseModel):
    """MALS (MeerKAT Absorption Line Survey) configuration.

    MALS provides L-band (900-1670 MHz) radio continuum catalogs:
    - DR1: 495,325 sources at 1-1.4 GHz (J/ApJS/270/33)
    - DR2: 971,980 sources wideband (J/A+A/690/A163) - largest MeerKAT catalog
    - DR3: 3,640 HI 21-cm absorption features (J/A+A/698/A120)
    """

    use_mals: bool = Field(False, description="Use MALS catalog")
    mals_release: Literal["dr1", "dr2", "dr3"] = Field(
        "dr2", description="MALS data release (dr1, dr2, dr3)"
    )
    flux_limit: float = Field(1.0, ge=0, description="Flux limit in mJy")


class VLSSrConfig(BaseModel):
    """VLSSr (73.8 MHz) catalog configuration."""

    use_vlssr: bool = Field(False, description="Use VLSSr catalog")
    flux_limit: float = Field(1.0, ge=0, description="Flux limit in Jy")


class TGSSConfig(BaseModel):
    """TGSS ADR1 (150 MHz) catalog configuration."""

    use_tgss: bool = Field(False, description="Use TGSS ADR1 catalog")
    flux_limit: float = Field(0.1, ge=0, description="Flux limit in Jy")


class WENSSConfig(BaseModel):
    """WENSS (325 MHz) catalog configuration."""

    use_wenss: bool = Field(False, description="Use WENSS catalog")
    flux_limit: float = Field(0.05, ge=0, description="Flux limit in Jy")


class SUMSSConfig(BaseModel):
    """SUMSS (843 MHz) catalog configuration."""

    use_sumss: bool = Field(False, description="Use SUMSS catalog")
    flux_limit: float = Field(0.008, ge=0, description="Flux limit in Jy")


class NVSSConfig(BaseModel):
    """NVSS (1400 MHz) catalog configuration."""

    use_nvss: bool = Field(False, description="Use NVSS catalog")
    flux_limit: float = Field(0.0025, ge=0, description="Flux limit in Jy")


class FIRSTConfig(BaseModel):
    """FIRST (1400 MHz) catalog configuration."""

    use_first: bool = Field(False, description="Use FIRST catalog")
    flux_limit: float = Field(0.001, ge=0, description="Flux limit in Jy")


class LoTSSConfig(BaseModel):
    """LoTSS (144 MHz) catalog configuration."""

    use_lotss: bool = Field(False, description="Use LoTSS catalog")
    lotss_release: Literal["dr1", "dr2"] = Field("dr2", description="LoTSS data release")
    flux_limit: float = Field(0.001, ge=0, description="Flux limit in Jy")


class AT20GConfig(BaseModel):
    """AT20G (20 GHz) catalog configuration."""

    use_at20g: bool = Field(False, description="Use AT20G catalog")
    flux_limit: float = Field(0.04, ge=0, description="Flux limit in Jy")


class ThreeCConfig(BaseModel):
    """3CR (178 MHz) catalog configuration."""

    use_3c: bool = Field(False, description="Use 3CR catalog")
    flux_limit: float = Field(1.0, ge=0, description="Flux limit in Jy")


class GB6Config(BaseModel):
    """GB6 (4850 MHz) catalog configuration."""

    use_gb6: bool = Field(False, description="Use GB6 catalog")
    flux_limit: float = Field(0.018, ge=0, description="Flux limit in Jy")


class RACSConfig(BaseModel):
    """RACS (887.5 / 1367.5 / 1655.5 MHz) catalog configuration via CASDA TAP."""

    use_racs: bool = Field(False, description="Use RACS catalog")
    racs_band: Literal["low", "mid", "high"] = Field(
        "low", description="RACS band: low (887.5 MHz), mid (1367.5 MHz), high (1655.5 MHz)"
    )
    flux_limit: float = Field(1.0, ge=0, description="Flux limit in Jy")
    max_rows: int = Field(1_000_000, ge=1, description="Maximum rows to retrieve via TAP")


class PySM3Config(BaseModel):
    """PySM3 diffuse sky model configuration."""

    use_pysm3: bool = Field(False, description="Use PySM3 diffuse model")
    components: str | list[str] = Field(
        "s1",
        description="PySM3 preset string(s), e.g. 's1' or ['s1', 'd1', 'f1']"
    )
    nside: int = Field(64, ge=1, description="HEALPix NSIDE resolution")
    # Frequencies are taken from the obs_frequency section of the config


class ULSAConfig(BaseModel):
    """ULSA ultra-low-frequency diffuse model configuration."""

    use_ulsa: bool = Field(False, description="Use ULSA diffuse model")
    nside: int = Field(64, ge=1, description="HEALPix NSIDE resolution")
    # Frequencies are taken from the obs_frequency section of the config


class PyRadioSkyConfig(BaseModel):
    """Local sky model file loader via pyradiosky."""

    use_pyradiosky: bool = Field(False, description="Load sky model from local file via pyradiosky")
    filename: str = Field("", description="Path to sky model file (SkyH5, VOTable, text, FHD)")
    filetype: str | None = Field(
        None, description="File format (skyh5, votable, text, fhd). Inferred if None."
    )
    flux_limit: float = Field(0.0, ge=0, description="Minimum Stokes I flux in Jy")
    reference_frequency_hz: float | None = Field(
        None, description="Reference frequency for Stokes I extraction (Hz). Uses first channel if None."
    )


class SkyModelConfig(BaseModel):
    """Sky model configuration."""

    # --- Existing models (unchanged) ---
    test_sources: SyntheticSourcesConfig = Field(default_factory=SyntheticSourcesConfig)
    test_sources_healpix: SyntheticSourcesConfig = Field(default_factory=SyntheticSourcesConfig)
    gsm_healpix: GSMConfig = Field(default_factory=GSMConfig)
    gleam: GLEAMConfig = Field(default_factory=GLEAMConfig)
    gleam_healpix: GLEAMConfig = Field(default_factory=GLEAMConfig)
    mals: MALSConfig = Field(default_factory=MALSConfig)
    # --- New point-source catalogs ---
    vlssr: VLSSrConfig = Field(default_factory=VLSSrConfig)
    tgss: TGSSConfig = Field(default_factory=TGSSConfig)
    wenss: WENSSConfig = Field(default_factory=WENSSConfig)
    sumss: SUMSSConfig = Field(default_factory=SUMSSConfig)
    nvss: NVSSConfig = Field(default_factory=NVSSConfig)
    first: FIRSTConfig = Field(default_factory=FIRSTConfig)
    lotss: LoTSSConfig = Field(default_factory=LoTSSConfig)
    at20g: AT20GConfig = Field(default_factory=AT20GConfig)
    three_c: ThreeCConfig = Field(default_factory=ThreeCConfig)
    gb6: GB6Config = Field(default_factory=GB6Config)
    racs: RACSConfig = Field(default_factory=RACSConfig)
    # --- New diffuse models ---
    pysm3: PySM3Config = Field(default_factory=PySM3Config)
    ulsa: ULSAConfig = Field(default_factory=ULSAConfig)
    # --- Local file loader ---
    pyradiosky: PyRadioSkyConfig = Field(default_factory=PyRadioSkyConfig)


class ObsTimeConfig(BaseModel):
    """Observation time configuration."""

    start_time: str = Field(
        "2023-01-01T00:00:00", description="Start time (ISO format)"
    )
    duration_seconds: float = Field(
        3600.0, gt=0, description="Total observation duration in seconds"
    )
    time_step_seconds: float = Field(
        60.0, gt=0, description="Time step between samples in seconds"
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
    simulation_subdir: str = Field("", description="Output subdirectory name")
    output_file_name: str = Field("complex_visibility", description="Output filename")
    output_file_format: Literal["HDF5", "JSON", "MS", "UVFITS"] = Field(
        "HDF5", description="Output format"
    )
    save_simulation_data: bool = Field(False, description="Save simulation data")
    plot_results: bool = Field(True, description="Generate visualization plots")
    open_plots_in_browser: bool = Field(True, description="Open plots in browser (set False to save only)")
    plotting_backend: str = Field("bokeh", description="Plotting backend (bokeh/matplotlib)")
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


class VisibilityConfig(BaseModel):
    """Visibility calculation configuration.

    Controls how visibilities are computed from the sky model.

    Attributes
    ----------
    calculation_type : str
        The algorithm used for visibility calculation:
        - "direct_sum": Direct summation over sources/pixels (RIME-based)
        - "spherical_harmonic": m-mode formalism (NOT YET IMPLEMENTED)

    sky_representation : str
        How the sky model is represented during calculation:
        - "point_sources": Discrete sources with (RA, Dec, flux)
          Best for: catalogs (GLEAM, MALS), sparse bright sources
        - "healpix_map": HEALPix brightness temperature map
          Best for: diffuse emission (GSM, LFSM, Haslam)
          More efficient for large-scale structure, works in T_b units

    Notes
    -----
    Both "point_sources" and "healpix_map" use direct summation:
        V = Σ S_i × exp(-2πi b·ŝ/λ)  (point sources)
        V = Σ T_p × Ω_p × exp(-2πi b·ŝ/λ)  (healpix)

    The difference is in sky representation, not the algorithm.
    True spherical harmonic visibility (m-mode) would use:
        V_m = Σ_lm B_lm × a_lm
    This is planned for future implementation.
    """

    calculation_type: Literal["direct_sum", "spherical_harmonic"] = Field(
        "direct_sum",
        description="Visibility calculation algorithm: 'direct_sum' (implemented) or 'spherical_harmonic' (future)"
    )
    sky_representation: Literal["point_sources", "healpix_map"] = Field(
        "point_sources",
        description="Sky model representation: 'point_sources' or 'healpix_map'"
    )


class CoordinatePrecisionConfig(BaseModel):
    """Precision settings for coordinate calculations."""

    antenna_positions: Literal["float32", "float64", "float128"] = Field(
        "float64", description="Antenna position precision"
    )
    source_positions: Literal["float32", "float64", "float128"] = Field(
        "float64", description="Source coordinate precision"
    )
    direction_cosines: Literal["float32", "float64", "float128"] = Field(
        "float64", description="Direction cosine (l,m,n) precision"
    )
    uvw: Literal["float32", "float64", "float128"] = Field(
        "float64", description="Baseline UVW coordinate precision"
    )


class JonesPrecisionConfig(BaseModel):
    """Precision settings for Jones matrix calculations."""

    geometric_phase: Literal["float32", "float64", "float128"] = Field(
        "float64", description="K term (geometric delay) - CRITICAL"
    )
    beam: Literal["float32", "float64", "float128"] = Field(
        "float64", description="E term (primary beam)"
    )
    ionosphere: Literal["float32", "float64", "float128"] = Field(
        "float64", description="Z term (ionosphere)"
    )
    troposphere: Literal["float32", "float64", "float128"] = Field(
        "float64", description="T term (troposphere)"
    )
    parallactic: Literal["float32", "float64", "float128"] = Field(
        "float64", description="P term (parallactic angle)"
    )
    gain: Literal["float32", "float64", "float128"] = Field(
        "float64", description="G term (antenna gains)"
    )
    bandpass: Literal["float32", "float64", "float128"] = Field(
        "float64", description="B term (bandpass)"
    )
    polarization_leakage: Literal["float32", "float64", "float128"] = Field(
        "float64", description="D term (polarization leakage)"
    )


class PrecisionConfigSchema(BaseModel):
    """Precision configuration for numerical computations.

    Controls the precision of different computation stages. Using lower
    precision (float32) can improve performance and reduce memory, while
    higher precision (float128) improves accuracy for critical paths.

    Presets can be specified using the `preset` field:
    - "standard": float64 everywhere (default)
    - "fast": float32 where safe, float64 for critical paths
    - "precise": float128 for critical paths, float64 elsewhere
    - "ultra": float128 everywhere (slow, NumPy only)

    Or configure each component individually for granular control.
    """

    preset: Literal["standard", "fast", "precise", "ultra"] | None = Field(
        None, description="Use a precision preset (overrides other settings)"
    )
    default: Literal["float32", "float64", "float128"] = Field(
        "float64", description="Default precision level"
    )
    coordinates: CoordinatePrecisionConfig = Field(
        default_factory=CoordinatePrecisionConfig,
        description="Coordinate precision settings"
    )
    jones: JonesPrecisionConfig = Field(
        default_factory=JonesPrecisionConfig,
        description="Jones matrix precision settings"
    )
    accumulation: Literal["float32", "float64", "float128"] = Field(
        "float64", description="Visibility accumulation precision"
    )
    output: Literal["float32", "float64", "float128"] = Field(
        "float64", description="Output visibility precision"
    )

    def to_precision_config(self):
        """Convert to rrivis.core.precision.PrecisionConfig.

        Returns
        -------
        PrecisionConfig
            The precision configuration object.
        """
        from rrivis.core.precision import (
            CoordinatePrecision,
            JonesPrecision,
            PrecisionConfig,
        )

        # If preset is specified, use it
        if self.preset:
            presets = {
                "standard": PrecisionConfig.standard,
                "fast": PrecisionConfig.fast,
                "precise": PrecisionConfig.precise,
                "ultra": PrecisionConfig.ultra,
            }
            return presets[self.preset]()

        # Otherwise build from individual settings
        return PrecisionConfig(
            default=self.default,
            coordinates=CoordinatePrecision(
                antenna_positions=self.coordinates.antenna_positions,
                source_positions=self.coordinates.source_positions,
                direction_cosines=self.coordinates.direction_cosines,
                uvw=self.coordinates.uvw,
            ),
            jones=JonesPrecision(
                geometric_phase=self.jones.geometric_phase,
                beam=self.jones.beam,
                ionosphere=self.jones.ionosphere,
                troposphere=self.jones.troposphere,
                parallactic=self.jones.parallactic,
                gain=self.jones.gain,
                bandpass=self.jones.bandpass,
                polarization_leakage=self.jones.polarization_leakage,
            ),
            accumulation=self.accumulation,
            output=self.output,
        )


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
    visibility: VisibilityConfig = Field(
        default_factory=VisibilityConfig,
        description="Visibility calculation settings"
    )
    precision: PrecisionConfigSchema | None = Field(
        None,
        description="Precision configuration for numerical computations"
    )

    model_config = {
        "extra": "allow",  # Allow extra fields for forward compatibility
        "validate_assignment": True,  # Validate on attribute assignment
    }

    @classmethod
    def from_yaml(cls, yaml_path: str | Path) -> "RRIvisConfig":
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

        with open(yaml_path) as f:
            data = yaml.safe_load(f) or {}

        try:
            return cls(**data)
        except Exception as e:
            raise ValueError(
                f"Invalid configuration in {yaml_path}:\n{e}\n\n"
                f"See documentation for valid config format."
            )

    def to_yaml(self, output_path: str | Path) -> None:
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

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary."""
        return self.model_dump()

    def generate_output_subdir(self) -> str:
        """Generate output subdirectory name from config parameters.

        Creates a descriptive, deterministic directory name based on
        simulation parameters plus the current runtime date.

        Format:
            {telescope}_{freq_start}-{freq_end}{unit}_{n_channels}ch_{duration}s_SimTime{DDMMYYYY}

        Returns
        -------
        str
            Generated subdirectory name.

        Examples
        --------
        >>> config = load_config("config.yaml")
        >>> config.generate_output_subdir()
        'HERA_100-120MHz_21ch_600s_SimTime15012026'
        """
        # Get config parameters
        telescope = self.telescope.telescope_name.replace(" ", "_")
        freq_start = int(self.obs_frequency.starting_frequency)
        freq_end = int(
            self.obs_frequency.starting_frequency +
            self.obs_frequency.frequency_bandwidth
        )
        freq_unit = self.obs_frequency.frequency_unit
        n_channels = self.obs_frequency.n_channels
        duration = int(self.obs_time.duration_seconds)

        # Get current UTC date and time in DD-MM-YYYY_HH-MM-SS format
        runtime_utc = datetime.now(UTC).strftime("%d-%m-%Y_%H-%M-%S")

        return (
            f"{telescope}_{freq_start}-{freq_end}{freq_unit}_"
            f"{n_channels}ch_{duration}s_SimTimeUTC{runtime_utc}"
        )


def load_config(config_path: str | Path) -> RRIvisConfig:
    """
    Load and validate configuration from YAML file.

    Parameters
    ----------
    config_path : str or Path
        Path to YAML configuration file.

    Returns
    -------
    RRIvisConfig
        Validated configuration instance with all defaults filled in.

    Raises
    ------
    FileNotFoundError
        If configuration file does not exist.
    ValueError
        If configuration file contains invalid values.

    Examples
    --------
    >>> config = load_config("simulation_config.yaml")
    >>> print(config.telescope.telescope_name)
    'HERA'
    >>> print(config.obs_frequency.n_channels)
    50

    See Also
    --------
    create_default_config : Create a default configuration file.
    RRIvisConfig : Main configuration class.
    """
    return RRIvisConfig.from_yaml(config_path)


def create_default_config(output_path: str | Path) -> None:
    """
    Create a default configuration file with all options documented.

    Parameters
    ----------
    output_path : str or Path
        Path where the default configuration file will be written.

    Examples
    --------
    >>> create_default_config("my_config.yaml")
    >>> # Edit the file and load it
    >>> config = load_config("my_config.yaml")

    Notes
    -----
    The created file contains all configuration options with their
    default values, making it a useful template for customization.

    See Also
    --------
    load_config : Load configuration from file.
    """
    config = RRIvisConfig()
    config.to_yaml(output_path)
