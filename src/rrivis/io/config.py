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

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any, Literal

import yaml
from pydantic import BaseModel, Field, TypeAdapter, model_validator


class TelescopeConfig(BaseModel):
    """Telescope configuration."""

    telescope_name: str = Field(
        "Unknown", description="Telescope name (e.g., HERA, MWA)"
    )
    use_pyuvdata_telescope: bool = Field(
        False, description="Load telescope from pyuvdata"
    )
    use_pyuvdata_location: bool = Field(False, description="Use pyuvdata location")
    use_pyuvdata_antennas: bool = Field(
        False, description="Use pyuvdata antenna positions"
    )
    use_pyuvdata_diameters: bool = Field(
        False, description="Use pyuvdata antenna diameters"
    )


class AntennaLayoutConfig(BaseModel):
    """Antenna layout configuration."""

    antenna_positions_file: str | None = Field(
        None, description="Path to antenna positions file"
    )
    antenna_file_format: (
        Literal["rrivis", "casa", "measurement_set", "uvfits", "mwa", "pyuvdata"] | None
    ) = Field(None, description="Antenna file format")
    all_antenna_diameter: float | None = Field(
        None, description="Default antenna diameter (meters)"
    )
    use_different_diameters: bool = Field(
        False, description="Use per-antenna diameters"
    )
    diameters: dict[str, float] = Field(
        default_factory=dict, description="Per-antenna diameter mapping"
    )
    fixed_HPBW: float | None = Field(None, description="Fixed HPBW override (radians)")


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

    beam_mode: Literal["analytic", "fits", "mixed"] | None = Field(
        None,
        description=(
            "Beam model type. 'analytic': parametric beam (e.g. Gaussian); "
            "'fits': beam loaded from FITS file(s); "
            "'mixed': per-antenna mix of analytic and FITS beams."
        ),
    )
    per_antenna: bool = Field(
        False,
        description=(
            "If True, each antenna uses its own beam (via antenna_beam_map). "
            "If False, all antennas share a common beam."
        ),
    )
    beam_file: str | None = Field(
        None,
        description="FITS beam file path shared by all antennas (fits mode, per_antenna=False).",
    )
    antenna_beam_map: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Mapping of antenna number (str) to beam specification. "
            "For fits/mixed modes with per_antenna=True: value is a FITS file path or 'analytic'."
        ),
    )
    beam_za_max_deg: float | None = Field(
        None, description="Max zenith angle (degrees)"
    )
    beam_za_buffer_deg: float | None = Field(None, description="ZA buffer (degrees)")
    beam_freq_buffer_hz: float | None = Field(None, description="Frequency buffer (Hz)")
    beam_peak_normalize: bool = Field(
        True,
        description="Peak-normalize FITS beams after loading (pyuvsim convention)",
    )
    beam_interp_function: str | None = Field(
        None,
        description="Interpolation function for FITS beams (e.g. 'az_za_simple', 'az_za_map_coordinates'). None uses pyuvdata default.",
    )

    # Aperture shape and taper
    aperture_shape: Literal["circular", "rectangular", "elliptical"] = Field(
        "circular", description="Aperture geometry."
    )
    taper: Literal[
        "uniform", "gaussian", "parabolic", "parabolic_squared", "cosine"
    ] = Field("gaussian", description="Illumination taper function.")
    edge_taper_dB: float = Field(10.0, description="Edge taper in dB (pedestal level).")

    # Feed model
    feed_model: Literal[
        "none", "corrugated_horn", "open_waveguide", "dipole_ground_plane"
    ] = Field(
        "none", description="Feed pattern model. Overrides taper when not 'none'."
    )
    feed_computation: Literal["analytical", "numerical"] = Field(
        "analytical",
        description="Feed-to-beam computation: 'analytical' (derive edge taper) or 'numerical' (Hankel transform).",
    )
    feed_params: dict[str, float] = Field(
        default_factory=dict,
        description="Feed params: q (horn), b_over_lambda (waveguide), height_wavelengths (dipole), focal_ratio (f/D).",
    )

    # Reflector geometry
    reflector_type: Literal["prime_focus", "cassegrain"] = Field(
        "prime_focus", description="Reflector geometry type."
    )
    magnification: float = Field(
        1.0,
        description="Cassegrain magnification M = (e+1)/(e-1). Only used when reflector_type='cassegrain'.",
    )

    # Aperture-specific parameters
    aperture_params: dict[str, float] = Field(
        default_factory=dict,
        description="Aperture params: length_x/length_y (rectangular), diameter_x/diameter_y (elliptical).",
    )


class BaselineSelectionConfig(BaseModel):
    """Baseline selection configuration."""

    use_autocorrelations: bool = Field(False, description="Include autocorrelations")
    use_crosscorrelations: bool = Field(True, description="Include crosscorrelations")
    only_selective_baseline_length: bool = Field(
        False, description="Filter by baseline length"
    )
    selective_baseline_lengths: list[float] = Field(
        default_factory=list, description="Selected baseline lengths"
    )
    selective_baseline_tolerance_meters: float = Field(
        0.5, ge=0, description="Baseline length tolerance (m)"
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


class SkyRegionEntryConfig(BaseModel):
    """A single sky region filter (cone or box).

    When ``shape="cone"``, ``radius_deg`` is required.
    When ``shape="box"``, ``width_deg`` and ``height_deg`` are required.
    """

    shape: Literal["cone", "box"] = Field(
        "cone", description="Region shape: 'cone' or 'box'"
    )
    center_ra_deg: float = Field(
        ..., ge=0.0, lt=360.0, description="RA centre (ICRS degrees)"
    )
    center_dec_deg: float = Field(
        ..., ge=-90.0, le=90.0, description="Dec centre (ICRS degrees)"
    )
    radius_deg: float | None = Field(
        None, gt=0.0, le=180.0, description="Cone radius (degrees)"
    )
    width_deg: float | None = Field(
        None, gt=0.0, le=360.0, description="Box RA width (degrees)"
    )
    height_deg: float | None = Field(
        None, gt=0.0, le=180.0, description="Box Dec height (degrees)"
    )

    @model_validator(mode="after")
    def validate_shape_fields(self) -> "SkyRegionEntryConfig":
        if self.shape == "cone" and self.radius_deg is None:
            raise ValueError("radius_deg is required when region shape='cone'.")
        if self.shape == "box" and (self.width_deg is None or self.height_deg is None):
            raise ValueError(
                "width_deg and height_deg are required when region shape='box'."
            )
        return self


def build_sky_region(
    config: SkyRegionEntryConfig
    | list[SkyRegionEntryConfig]
    | dict[str, Any]
    | list[dict[str, Any]]
    | None,
) -> Any:
    """Build a runtime ``SkyRegion`` from config-shaped input."""
    if config is None:
        return None

    from rrivis.core.sky.region import SkyRegion

    def _build_one(entry: SkyRegionEntryConfig | dict[str, Any]) -> Any:
        if not isinstance(entry, SkyRegionEntryConfig):
            entry = SkyRegionEntryConfig.model_validate(entry)
        if entry.shape == "cone":
            return SkyRegion.cone(
                entry.center_ra_deg,
                entry.center_dec_deg,
                entry.radius_deg,
            )
        return SkyRegion.box(
            entry.center_ra_deg,
            entry.center_dec_deg,
            entry.width_deg,
            entry.height_deg,
        )

    if isinstance(config, list):
        return SkyRegion.union([_build_one(entry) for entry in config])
    return _build_one(config)


@dataclass(frozen=True)
class SkyLoaderRequestContext:
    """Resolved global context used to build one loader request."""

    flux_multiplier: float = 1.0
    region: Any = None
    brightness_conversion: Literal["planck", "rayleigh-jeans"] | None = None
    frequencies: Any = None
    obs_frequency_config: dict[str, Any] | None = None
    memmap_path: str | None = None


class SkySourceConfig(BaseModel):
    """Base type for one entry in ``sky_model.sources``."""

    model_config = {"extra": "forbid"}

    kind: str = Field(..., description="Canonical loader name")
    region: SkyRegionEntryConfig | list[SkyRegionEntryConfig] | None = Field(
        None,
        description="Optional source-specific sky region override.",
    )
    brightness_conversion: Literal["planck", "rayleigh-jeans"] | None = Field(
        None,
        description="Optional source-specific brightness conversion override.",
    )

    def to_loader_request(
        self,
        *,
        flux_multiplier: float = 1.0,
        region: Any = None,
        brightness_conversion: str | None = None,
        frequencies: Any = None,
        obs_frequency_config: dict[str, Any] | None = None,
        memmap_path: str | None = None,
    ) -> tuple[str, dict[str, Any]]:
        """Build an explicit loader request for this source spec."""
        context = SkyLoaderRequestContext(
            flux_multiplier=flux_multiplier,
            region=region,
            brightness_conversion=brightness_conversion,
            frequencies=frequencies,
            obs_frequency_config=obs_frequency_config,
            memmap_path=memmap_path,
        )
        return self._build_loader_request(context)

    def _scaled_flux(
        self,
        value: float | None,
        context: SkyLoaderRequestContext,
    ) -> float | None:
        if value is None:
            return None
        return value * context.flux_multiplier

    def _common_kwargs(
        self,
        context: SkyLoaderRequestContext,
        *,
        include_frequency_context: bool = False,
        include_memmap: bool = False,
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {}
        brightness_conversion = self.brightness_conversion
        if brightness_conversion is None:
            brightness_conversion = context.brightness_conversion
        if brightness_conversion is not None:
            kwargs["brightness_conversion"] = brightness_conversion

        region = (
            build_sky_region(self.region) if self.region is not None else context.region
        )
        if region is not None:
            kwargs["region"] = region

        if include_frequency_context:
            if context.frequencies is not None:
                kwargs["frequencies"] = context.frequencies
            elif context.obs_frequency_config is not None:
                kwargs["obs_frequency_config"] = context.obs_frequency_config

        if include_memmap and context.memmap_path is not None:
            kwargs["memmap_path"] = context.memmap_path
        return kwargs

    def _build_loader_request(
        self,
        context: SkyLoaderRequestContext,
    ) -> tuple[str, dict[str, Any]]:
        raise NotImplementedError


class DiffuseSkySourceConfig(SkySourceConfig):
    kind: Literal["diffuse_sky"] = "diffuse_sky"
    model: str = Field("gsm2008", description="Diffuse-model selector")
    nside: int = Field(64, ge=1, description="HEALPix NSIDE")
    include_cmb: bool | None = Field(None, description="Include CMB in diffuse sky")
    basemap: str | None = Field(None, description="GSM2008 basemap")
    interpolation: str | None = Field(None, description="GSM2008 interpolation")

    def _build_loader_request(
        self,
        context: SkyLoaderRequestContext,
    ) -> tuple[str, dict[str, Any]]:
        kwargs = self._common_kwargs(
            context,
            include_frequency_context=True,
            include_memmap=True,
        )
        kwargs.update({"model": self.model, "nside": self.nside})
        if self.include_cmb is not None:
            kwargs["include_cmb"] = self.include_cmb
        if self.basemap is not None:
            kwargs["basemap"] = self.basemap
        if self.interpolation is not None:
            kwargs["interpolation"] = self.interpolation
        return self.kind, kwargs


class Pysm3SourceConfig(SkySourceConfig):
    kind: Literal["pysm3"] = "pysm3"
    components: str | list[str] = Field("s1", description="PySM3 preset string(s)")
    nside: int = Field(64, ge=1, description="HEALPix NSIDE")
    include_polarization: bool = Field(
        False, description="Include polarized diffuse output when supported"
    )

    def _build_loader_request(
        self,
        context: SkyLoaderRequestContext,
    ) -> tuple[str, dict[str, Any]]:
        kwargs = self._common_kwargs(
            context,
            include_frequency_context=True,
            include_memmap=True,
        )
        kwargs.update(
            {
                "components": self.components,
                "nside": self.nside,
                "include_polarization": self.include_polarization,
            }
        )
        return self.kind, kwargs


class PointCatalogSourceConfig(SkySourceConfig):
    flux_limit: float | None = Field(None, ge=0.0, description="Minimum flux limit")
    max_rows: int | None = Field(None, ge=1, description="Maximum rows for TAP query")
    allow_full_catalog: bool = Field(
        False,
        description="Explicit opt-in for uncapped full-catalog network downloads.",
    )

    def _build_catalog_kwargs(
        self,
        context: SkyLoaderRequestContext,
    ) -> dict[str, Any]:
        kwargs = self._common_kwargs(context)
        if self.flux_limit is not None:
            kwargs["flux_limit"] = self._scaled_flux(self.flux_limit, context)
        if self.max_rows is not None:
            kwargs["max_rows"] = self.max_rows
        if self.allow_full_catalog is not None:
            kwargs["allow_full_catalog"] = self.allow_full_catalog
        return kwargs


class PyradioskyFileSourceConfig(SkySourceConfig):
    kind: Literal["pyradiosky_file"] = "pyradiosky_file"
    filename: str = Field(..., description="Input filename")
    filetype: str | None = Field(None, description="File format override")
    flux_limit: float | None = Field(None, ge=0.0, description="Minimum flux limit")
    reference_frequency_hz: float | None = Field(
        None, description="Reference frequency for file-based point sources"
    )
    spectral_loss_policy: Literal["warn", "error"] = Field(
        "warn",
        description="How to handle lossy point-spectrum collapse for full/subband files.",
    )

    def _build_loader_request(
        self,
        context: SkyLoaderRequestContext,
    ) -> tuple[str, dict[str, Any]]:
        kwargs = self._common_kwargs(
            context,
            include_frequency_context=True,
            include_memmap=True,
        )
        kwargs["filename"] = self.filename
        if self.filetype is not None:
            kwargs["filetype"] = self.filetype
        if self.flux_limit is not None:
            kwargs["flux_limit"] = self._scaled_flux(self.flux_limit, context)
        if self.reference_frequency_hz is not None:
            kwargs["reference_frequency_hz"] = self.reference_frequency_hz
        kwargs["spectral_loss_policy"] = self.spectral_loss_policy
        return self.kind, kwargs


class BbsSourceConfig(SkySourceConfig):
    kind: Literal["bbs"] = "bbs"
    filename: str = Field(..., description="Input filename")
    flux_limit: float | None = Field(None, ge=0.0, description="Minimum flux limit")

    def _build_loader_request(
        self,
        context: SkyLoaderRequestContext,
    ) -> tuple[str, dict[str, Any]]:
        kwargs = self._common_kwargs(context)
        kwargs["filename"] = self.filename
        if self.flux_limit is not None:
            kwargs["flux_limit"] = self._scaled_flux(self.flux_limit, context)
        return self.kind, kwargs


class FitsImageSourceConfig(SkySourceConfig):
    kind: Literal["fits_image"] = "fits_image"
    filename: str = Field(..., description="Input filename")
    nside: int = Field(128, ge=1, description="HEALPix NSIDE")

    def _build_loader_request(
        self,
        context: SkyLoaderRequestContext,
    ) -> tuple[str, dict[str, Any]]:
        kwargs = self._common_kwargs(
            context,
            include_frequency_context=True,
            include_memmap=True,
        )
        kwargs.update({"filename": self.filename, "nside": self.nside})
        return self.kind, kwargs


class TestSourcesConfig(SkySourceConfig):
    kind: Literal["test_sources"] = "test_sources"
    representation: Literal["point_sources", "healpix_map"] = Field(
        "point_sources", description="Synthetic-source output representation"
    )
    num_sources: int = Field(100, ge=1, description="Number of synthetic sources")
    distribution: Literal["uniform", "random"] = Field(
        "uniform", description="Synthetic source placement"
    )
    seed: int | None = Field(None, description="Random seed")
    flux_min: float | None = Field(None, ge=0.0, description="Minimum source flux")
    flux_max: float | None = Field(None, ge=0.0, description="Maximum source flux")
    dec_deg: float | None = Field(None, description="Source declination")
    dec_range_deg: float | None = Field(
        None, ge=0.0, description="Half-width of random declination band"
    )
    spectral_index: float | None = Field(None, description="Spectral index")
    polarization_fraction: float = Field(
        0.0, ge=0.0, le=1.0, description="Linear polarization fraction"
    )
    polarization_angle_deg: float = Field(0.0, description="Linear polarization angle")
    stokes_v_fraction: float = Field(
        0.0, ge=0.0, le=1.0, description="Circular polarization fraction"
    )
    nside: int | None = Field(None, ge=1, description="HEALPix NSIDE")

    @model_validator(mode="after")
    def validate_source_ranges(self) -> "TestSourcesConfig":
        if (
            self.flux_min is not None
            and self.flux_max is not None
            and self.flux_min > self.flux_max
        ):
            raise ValueError("flux_min must be <= flux_max.")
        if self.representation == "healpix_map" and self.nside is None:
            self.nside = 64
        return self

    def _build_loader_request(
        self,
        context: SkyLoaderRequestContext,
    ) -> tuple[str, dict[str, Any]]:
        kwargs = self._common_kwargs(
            context,
            include_frequency_context=True,
            include_memmap=True,
        )
        kwargs.update(
            {
                "representation": self.representation,
                "num_sources": self.num_sources,
                "distribution": self.distribution,
                "polarization_fraction": self.polarization_fraction,
                "polarization_angle_deg": self.polarization_angle_deg,
                "stokes_v_fraction": self.stokes_v_fraction,
            }
        )
        if self.seed is not None:
            kwargs["seed"] = self.seed
        if self.flux_min is not None:
            kwargs["flux_min"] = self._scaled_flux(self.flux_min, context)
        if self.flux_max is not None:
            kwargs["flux_max"] = self._scaled_flux(self.flux_max, context)
        if self.dec_deg is not None:
            kwargs["dec_deg"] = self.dec_deg
        if self.dec_range_deg is not None:
            kwargs["dec_range_deg"] = self.dec_range_deg
        if self.spectral_index is not None:
            kwargs["spectral_index"] = self.spectral_index
        if self.nside is not None:
            kwargs["nside"] = self.nside
        return self.kind, kwargs


class GleamSourceConfig(PointCatalogSourceConfig):
    kind: Literal["gleam"] = "gleam"
    flux_limit: float = Field(1.0, ge=0.0, description="Minimum flux limit")
    catalog: str = Field("gleam_egc", description="Catalog identifier")

    def _build_loader_request(
        self,
        context: SkyLoaderRequestContext,
    ) -> tuple[str, dict[str, Any]]:
        kwargs = self._build_catalog_kwargs(context)
        kwargs["catalog"] = self.catalog
        return self.kind, kwargs


class MalsSourceConfig(PointCatalogSourceConfig):
    kind: Literal["mals"] = "mals"
    flux_limit: float = Field(1.0, ge=0.0, description="Minimum flux limit")
    release: str = Field("dr2", description="Release identifier")

    def _build_loader_request(
        self,
        context: SkyLoaderRequestContext,
    ) -> tuple[str, dict[str, Any]]:
        kwargs = self._build_catalog_kwargs(context)
        kwargs["release"] = self.release
        return self.kind, kwargs


class LotssSourceConfig(PointCatalogSourceConfig):
    kind: Literal["lotss"] = "lotss"
    flux_limit: float = Field(0.001, ge=0.0, description="Minimum flux limit")
    release: str = Field("dr2", description="Release identifier")

    def _build_loader_request(
        self,
        context: SkyLoaderRequestContext,
    ) -> tuple[str, dict[str, Any]]:
        kwargs = self._build_catalog_kwargs(context)
        kwargs["release"] = self.release
        return self.kind, kwargs


class RacsSourceConfig(PointCatalogSourceConfig):
    kind: Literal["racs"] = "racs"
    band: str = Field("low", description="Catalog band selector")
    flux_limit: float = Field(1.0, ge=0.0, description="Minimum flux limit")
    max_rows: int = Field(1_000_000, ge=1, description="Maximum rows for TAP query")

    def _build_loader_request(
        self,
        context: SkyLoaderRequestContext,
    ) -> tuple[str, dict[str, Any]]:
        kwargs = self._common_kwargs(context)
        if self.flux_limit is not None:
            kwargs["flux_limit"] = self._scaled_flux(self.flux_limit, context)
        kwargs["band"] = self.band
        kwargs["max_rows"] = self.max_rows
        return self.kind, kwargs


class ThreeCSourceConfig(PointCatalogSourceConfig):
    kind: Literal["3c"] = "3c"
    flux_limit: float = Field(1.0, ge=0.0, description="Minimum flux limit")

    def _build_loader_request(
        self,
        context: SkyLoaderRequestContext,
    ) -> tuple[str, dict[str, Any]]:
        return self.kind, self._build_catalog_kwargs(context)


class NvssSourceConfig(PointCatalogSourceConfig):
    kind: Literal["nvss"] = "nvss"
    flux_limit: float = Field(0.0025, ge=0.0, description="Minimum flux limit")

    def _build_loader_request(
        self,
        context: SkyLoaderRequestContext,
    ) -> tuple[str, dict[str, Any]]:
        return self.kind, self._build_catalog_kwargs(context)


class SumssSourceConfig(PointCatalogSourceConfig):
    kind: Literal["sumss"] = "sumss"
    flux_limit: float = Field(0.008, ge=0.0, description="Minimum flux limit")

    def _build_loader_request(
        self,
        context: SkyLoaderRequestContext,
    ) -> tuple[str, dict[str, Any]]:
        return self.kind, self._build_catalog_kwargs(context)


class TgssSourceConfig(PointCatalogSourceConfig):
    kind: Literal["tgss"] = "tgss"
    flux_limit: float = Field(0.1, ge=0.0, description="Minimum flux limit")

    def _build_loader_request(
        self,
        context: SkyLoaderRequestContext,
    ) -> tuple[str, dict[str, Any]]:
        return self.kind, self._build_catalog_kwargs(context)


class VlassSourceConfig(PointCatalogSourceConfig):
    kind: Literal["vlass"] = "vlass"
    flux_limit: float = Field(0.001, ge=0.0, description="Minimum flux limit")

    def _build_loader_request(
        self,
        context: SkyLoaderRequestContext,
    ) -> tuple[str, dict[str, Any]]:
        return self.kind, self._build_catalog_kwargs(context)


class VlssrSourceConfig(PointCatalogSourceConfig):
    kind: Literal["vlssr"] = "vlssr"
    flux_limit: float = Field(1.0, ge=0.0, description="Minimum flux limit")

    def _build_loader_request(
        self,
        context: SkyLoaderRequestContext,
    ) -> tuple[str, dict[str, Any]]:
        return self.kind, self._build_catalog_kwargs(context)


class WenssSourceConfig(PointCatalogSourceConfig):
    kind: Literal["wenss"] = "wenss"
    flux_limit: float = Field(0.05, ge=0.0, description="Minimum flux limit")

    def _build_loader_request(
        self,
        context: SkyLoaderRequestContext,
    ) -> tuple[str, dict[str, Any]]:
        return self.kind, self._build_catalog_kwargs(context)


class CustomRegisteredSourceConfig(SkySourceConfig):
    """Fallback for ad-hoc registered loaders used outside the built-in union."""

    model_config = {"extra": "allow"}

    def _build_loader_request(
        self,
        context: SkyLoaderRequestContext,
    ) -> tuple[str, dict[str, Any]]:
        from rrivis.core.sky.registry import loader_registry

        kind = loader_registry.resolve_name(self.kind)
        definition = loader_registry.definition(kind)
        kwargs = self._common_kwargs(
            context,
            include_frequency_context=definition.supports_healpix_map,
        )

        extra_values = dict(getattr(self, "__pydantic_extra__", {}) or {})
        flux_fields = {"flux_limit", "flux_min", "flux_max"}
        for loader_arg, source_field in definition.config_fields.items():
            if source_field in extra_values:
                value = extra_values[source_field]
            elif hasattr(self, source_field):
                value = getattr(self, source_field)
            else:
                continue
            if value is None:
                continue
            if source_field in flux_fields:
                value = self._scaled_flux(value, context)
            kwargs[loader_arg] = value

        return kind, kwargs


_SKY_SOURCE_CONFIG_UNION = Annotated[
    (
        GleamSourceConfig
        | MalsSourceConfig
        | LotssSourceConfig
        | RacsSourceConfig
        | ThreeCSourceConfig
        | NvssSourceConfig
        | SumssSourceConfig
        | TgssSourceConfig
        | VlassSourceConfig
        | VlssrSourceConfig
        | WenssSourceConfig
        | DiffuseSkySourceConfig
        | Pysm3SourceConfig
        | PyradioskyFileSourceConfig
        | BbsSourceConfig
        | FitsImageSourceConfig
        | TestSourcesConfig
    ),
    Field(discriminator="kind"),
]

_SKY_SOURCE_CONFIG_ADAPTER = TypeAdapter(_SKY_SOURCE_CONFIG_UNION)
_BUILTIN_SKY_SOURCE_KINDS = {
    "gleam",
    "mals",
    "lotss",
    "racs",
    "3c",
    "nvss",
    "sumss",
    "tgss",
    "vlass",
    "vlssr",
    "wenss",
    "diffuse_sky",
    "pysm3",
    "pyradiosky_file",
    "bbs",
    "fits_image",
    "test_sources",
}


def parse_sky_source_config(data: Any) -> SkySourceConfig:
    """Parse one tagged source spec in the new explicit ``kind=...`` form."""
    if not isinstance(data, dict):
        raise TypeError(
            "sky_model.sources entries must be objects with a 'kind' field."
        )
    if data.get("kind") in _BUILTIN_SKY_SOURCE_KINDS:
        return _SKY_SOURCE_CONFIG_ADAPTER.validate_python(data)
    return CustomRegisteredSourceConfig.model_validate(data)


_LEGACY_SKY_MODEL_SECTIONS = frozenset(
    {
        "bbs",
        "fits_image",
        "gleam",
        "gleam_healpix",
        "gsm_healpix",
        "lotss",
        "mals",
        "nvss",
        "pyradiosky",
        "pysm3",
        "racs",
        "sumss",
        "test_sources",
        "test_sources_healpix",
        "tgss",
        "three_c",
        "vlass",
        "vlssr",
        "wenss",
    }
)


class SkyModelConfig(BaseModel):
    """Sky model configuration."""

    model_config = {"extra": "forbid"}

    sources: list[_SKY_SOURCE_CONFIG_UNION] = Field(
        default_factory=list,
        description="List of sky-model source specs to load and combine",
    )
    flux_unit: Literal["Jy", "mJy", "uJy"] = Field(
        "Jy",
        description="Unit for all source-spec flux values (flux_min, flux_max, flux_limit)",
    )
    brightness_conversion: Literal["planck", "rayleigh-jeans"] = Field(
        "planck",
        description=(
            "Brightness temperature conversion method for all loaders. "
            "'planck' (exact Planck law) or 'rayleigh-jeans' (RJ approximation)."
        ),
    )
    mixed_model_policy: Literal["error", "warn", "allow"] = Field(
        "error",
        description=(
            "How to handle combinations that mix point catalogs with diffuse "
            "HEALPix models: error, warn, or allow."
        ),
    )
    region: SkyRegionEntryConfig | list[SkyRegionEntryConfig] | None = Field(
        None,
        description="Sky region filter(s). Single region or list for union of regions.",
    )

    @model_validator(mode="before")
    @classmethod
    def reject_legacy_sections(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        legacy_sections = sorted(set(data) & _LEGACY_SKY_MODEL_SECTIONS)
        if legacy_sections:
            sections = ", ".join(legacy_sections)
            raise ValueError(
                "sky_model now uses only a 'sources' list. "
                f"Legacy nested section(s) are no longer accepted: {sections}. "
                "Rewrite each enabled section as an entry under sky_model.sources."
            )
        return data


class ObsTimeConfig(BaseModel):
    """Observation time configuration."""

    start_time: str | None = Field(None, description="Start time (ISO format)")
    duration_seconds: float | None = Field(
        None, description="Total observation duration in seconds"
    )
    time_step_seconds: float | None = Field(
        None, description="Time step between samples in seconds"
    )


class ObsFrequencyConfig(BaseModel):
    """Observation frequency configuration."""

    starting_frequency: float | None = Field(None, description="Starting frequency")
    frequency_interval: float | None = Field(None, description="Frequency interval")
    frequency_bandwidth: float | None = Field(None, description="Frequency bandwidth")
    frequency_unit: Literal["Hz", "kHz", "MHz", "GHz"] = Field(
        "MHz", description="Frequency unit"
    )

    @property
    def n_channels(self) -> int:
        """Calculate number of frequency channels."""
        if self.frequency_bandwidth is None or self.frequency_interval is None:
            return 0
        return max(1, int(self.frequency_bandwidth / self.frequency_interval) + 1)


class OutputConfig(BaseModel):
    """Output configuration."""

    simulation_data_dir: str = Field("", description="Output directory")
    simulation_subdir: str = Field("", description="Output subdirectory name")
    output_file_name: str = Field("visibilities", description="Output filename")
    output_file_format: Literal["HDF5", "JSON", "MS", "UVFITS"] | None = Field(
        None, description="Output format (default: HDF5)"
    )
    save_simulation_data: bool = Field(False, description="Save simulation data")
    overwrite_output: bool = Field(False, description="Overwrite existing output files")
    skip_overwrite_confirmation: bool = Field(
        False,
        description="Skip the interactive confirmation prompt when overwrite_output is true",
    )
    prompt_for_output_suffix: bool = Field(
        False,
        description="When output folder already exists, ask user for a suffix to append and create a fresh folder instead of overwriting",
    )
    plot_results: bool = Field(False, description="Generate visualization plots")
    open_plots_in_browser: bool = Field(
        False, description="Open plots in browser (set False to save only)"
    )
    plotting_backend: str = Field(
        "bokeh", description="Plotting backend (bokeh/matplotlib)"
    )
    plot_skymodel_every_hour: bool = Field(False, description="Plot sky model")
    save_log_data: bool = Field(False, description="Save log data")
    angle_unit: Literal["degrees", "radians", ""] = Field(
        "", description="Angle display unit"
    )
    skymodel_frequency: float | None = Field(
        None, description="Sky model plot frequency"
    )


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
        description="Visibility calculation algorithm: 'direct_sum' (implemented) or 'spherical_harmonic' (future)",
    )
    sky_representation: Literal["point_sources", "healpix_map"] | None = Field(
        None, description="Sky model representation: 'point_sources' or 'healpix_map'"
    )
    allow_lossy_point_materialization: bool = Field(
        False,
        description=(
            "Allow lossy HEALPix-to-point conversion when point_sources mode "
            "is requested."
        ),
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


class SkyModelPrecisionConfig(BaseModel):
    """Precision settings for sky model data storage."""

    source_positions: Literal["float32", "float64", "float128"] = Field(
        "float64", description="RA/Dec precision — phase-critical"
    )
    flux: Literal["float32", "float64", "float128"] = Field(
        "float64", description="Flux density and Stokes parameter precision"
    )
    spectral_index: Literal["float32", "float64", "float128"] = Field(
        "float64", description="Power-law spectral index precision"
    )
    healpix_maps: Literal["float32", "float64", "float128"] = Field(
        "float32", description="HEALPix brightness temperature map precision"
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
        description="Coordinate precision settings",
    )
    jones: JonesPrecisionConfig = Field(
        default_factory=JonesPrecisionConfig,
        description="Jones matrix precision settings",
    )
    sky_model: SkyModelPrecisionConfig = Field(
        default_factory=SkyModelPrecisionConfig,
        description="Sky model data precision settings",
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
            SkyModelPrecision,
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
            sky_model=SkyModelPrecision(
                source_positions=self.sky_model.source_positions,
                flux=self.sky_model.flux,
                spectral_index=self.sky_model.spectral_index,
                healpix_maps=self.sky_model.healpix_maps,
            ),
            accumulation=self.accumulation,
            output=self.output,
        )


class ComputeConfig(BaseModel):
    """Compute backend configuration."""

    backend: str = Field(
        "numpy",
        description="Computation backend: 'auto', 'numpy', 'numba', 'jax'",
    )
    offline: bool = Field(
        False,
        description=(
            "Force offline mode. When True, all network connectivity checks "
            "are skipped and sky models requiring internet will raise errors "
            "or be skipped."
        ),
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
        default_factory=VisibilityConfig, description="Visibility calculation settings"
    )
    compute: ComputeConfig = Field(default_factory=ComputeConfig)
    precision: PrecisionConfigSchema | None = Field(
        None, description="Precision configuration for numerical computations"
    )

    model_config = {
        "extra": "allow",  # Allow extra fields for forward compatibility
        "validate_assignment": True,  # Validate on attribute assignment
    }

    @staticmethod
    def _preprocess_yaml_data(data: dict, yaml_dir: Path) -> dict:
        """Resolve relative paths and coerce YAML-parsed types to expected Python types.

        Parameters
        ----------
        data : dict
            Raw dictionary from yaml.safe_load.
        yaml_dir : Path
            Resolved directory of the YAML file, used to resolve relative paths.

        Returns
        -------
        dict
            Mutated data dict ready for Pydantic construction.
        """
        from datetime import datetime as _datetime

        # YAML parses unquoted ISO timestamps (e.g. 2025-01-01T00:00:00) as datetime
        # objects. Convert them back to ISO strings so Pydantic's str field accepts them.
        obs_time = data.get("obs_time")
        if isinstance(obs_time, dict):
            start = obs_time.get("start_time")
            if isinstance(start, _datetime):
                obs_time["start_time"] = start.isoformat()

        # Resolve antenna_positions_file relative to the YAML file's directory
        antenna_file = (data.get("antenna_layout") or {}).get("antenna_positions_file")
        if antenna_file and not Path(antenna_file).is_absolute():
            data.setdefault("antenna_layout", {})["antenna_positions_file"] = str(
                yaml_dir / antenna_file
            )

        return data

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

        yaml_dir = yaml_path.parent.resolve()

        with open(yaml_path) as f:
            data = yaml.safe_load(f) or {}

        data = cls._preprocess_yaml_data(data, yaml_dir)

        from pydantic import ValidationError

        try:
            return cls(**data)
        except ValidationError:
            raise  # Let CLI handle structured field errors
        except Exception as e:
            raise ValueError(f"Invalid configuration in {yaml_path}:\n{e}") from e

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

    def validate(self) -> list[str]:
        """Collect all configuration errors upfront.

        Returns a list of human-readable error messages. An empty list means
        the configuration is valid and simulation can proceed.

        Returns
        -------
        list[str]
            Error messages, one per problem found.
        """
        errors: list[str] = []
        al = self.antenna_layout
        ot = self.obs_time
        of = self.obs_frequency
        loc = self.location

        # --- Antenna layout ---
        if not al.antenna_positions_file:
            errors.append(
                "antenna_layout.antenna_positions_file: required but not set. "
                "Set to your antenna positions file path."
            )
        elif not Path(al.antenna_positions_file).exists():
            errors.append(
                f"antenna_layout.antenna_positions_file: file not found: "
                f"'{al.antenna_positions_file}'"
            )
        if not al.antenna_file_format:
            errors.append(
                "antenna_layout.antenna_file_format: required but not set. "
                "E.g. 'rrivis', 'casa', 'uvfits'."
            )
        if al.all_antenna_diameter is None:
            errors.append(
                "antenna_layout.all_antenna_diameter: required but not set. "
                "E.g. 14.0 (meters)."
            )
        elif al.all_antenna_diameter <= 0:
            errors.append(
                f"antenna_layout.all_antenna_diameter: must be > 0, "
                f"got {al.all_antenna_diameter}."
            )

        # --- Observation time ---
        if ot.start_time is None:
            errors.append(
                "obs_time.start_time: required but not set. E.g. '2025-01-01T00:00:00'."
            )
        else:
            try:
                from astropy.time import Time as _ATime

                _ATime(ot.start_time)
            except Exception:
                errors.append(
                    f"obs_time.start_time: invalid ISO format '{ot.start_time}'. "
                    "E.g. '2025-01-01T00:00:00'."
                )
        if ot.duration_seconds is None:
            errors.append(
                "obs_time.duration_seconds: required but not set. "
                "E.g. 3600.0 (seconds)."
            )
        elif ot.duration_seconds <= 0:
            errors.append(
                f"obs_time.duration_seconds: must be > 0, got {ot.duration_seconds}."
            )
        if ot.time_step_seconds is None:
            errors.append(
                "obs_time.time_step_seconds: required but not set. E.g. 60.0 (seconds)."
            )
        elif ot.time_step_seconds <= 0:
            errors.append(
                f"obs_time.time_step_seconds: must be > 0, got {ot.time_step_seconds}."
            )
        if (
            ot.duration_seconds is not None
            and ot.time_step_seconds is not None
            and ot.duration_seconds > 0
            and ot.time_step_seconds > 0
            and ot.time_step_seconds > ot.duration_seconds
        ):
            errors.append("obs_time.time_step_seconds must be <= duration_seconds.")

        # --- Observation frequency ---
        if of.starting_frequency is None:
            errors.append(
                "obs_frequency.starting_frequency: required but not set. "
                "E.g. 50.0 (MHz)."
            )
        elif of.starting_frequency <= 0:
            errors.append(
                f"obs_frequency.starting_frequency: must be > 0, "
                f"got {of.starting_frequency}."
            )
        if of.frequency_interval is None:
            errors.append(
                "obs_frequency.frequency_interval: required but not set. "
                "E.g. 1.0 (MHz)."
            )
        elif of.frequency_interval <= 0:
            errors.append(
                f"obs_frequency.frequency_interval: must be > 0, "
                f"got {of.frequency_interval}."
            )
        if of.frequency_bandwidth is None:
            errors.append(
                "obs_frequency.frequency_bandwidth: required but not set. "
                "E.g. 100.0 (MHz)."
            )
        elif of.frequency_bandwidth <= 0:
            errors.append(
                f"obs_frequency.frequency_bandwidth: must be > 0, "
                f"got {of.frequency_bandwidth}."
            )
        if (
            of.frequency_interval is not None
            and of.frequency_bandwidth is not None
            and of.frequency_interval > 0
            and of.frequency_bandwidth > 0
            and of.frequency_interval >= of.frequency_bandwidth
        ):
            errors.append(
                "obs_frequency.frequency_interval must be < frequency_bandwidth."
            )

        # --- Location range checks (only if value is provided) ---
        def _as_float(v: float | str) -> float | None:
            if v == "":
                return None
            try:
                return float(v)
            except (TypeError, ValueError):
                return None

        lat = _as_float(loc.lat)
        lon = _as_float(loc.lon)
        height = _as_float(loc.height)
        if lat is not None and not (-90 <= lat <= 90):
            errors.append(f"location.lat: must be in [-90, 90], got {lat}.")
        if lon is not None and not (-180 <= lon <= 180):
            errors.append(f"location.lon: must be in [-180, 180], got {lon}.")
        if height is not None and height < 0:
            errors.append(f"location.height: must be >= 0, got {height}.")

        # --- Beam cross-field ---
        beams = self.beams
        if beams.beam_mode is None:
            errors.append(
                "beams.beam_mode: required. "
                "Set to 'analytic' (parametric beam), 'fits' (FITS beam file), "
                "or 'mixed' (per-antenna mix of analytic and FITS)."
            )
        elif beams.beam_mode == "analytic":
            # Validate aperture-specific parameters
            if beams.aperture_shape == "rectangular":
                if (
                    "length_x" not in beams.aperture_params
                    or "length_y" not in beams.aperture_params
                ):
                    errors.append(
                        "beams.aperture_params: 'length_x' and 'length_y' required "
                        "when aperture_shape='rectangular'."
                    )
            elif beams.aperture_shape == "elliptical":
                if (
                    "diameter_x" not in beams.aperture_params
                    or "diameter_y" not in beams.aperture_params
                ):
                    errors.append(
                        "beams.aperture_params: 'diameter_x' and 'diameter_y' required "
                        "when aperture_shape='elliptical'."
                    )
            if beams.feed_model != "none":
                if "focal_ratio" not in beams.feed_params:
                    errors.append(
                        "beams.feed_params.focal_ratio: required when feed_model is not 'none'. "
                        "Set to the f/D ratio (e.g. 0.4)."
                    )
            if beams.reflector_type == "cassegrain" and beams.magnification <= 1.0:
                errors.append(
                    "beams.magnification: must be > 1.0 when reflector_type='cassegrain'."
                )
        elif beams.beam_mode == "fits":
            if beams.per_antenna:
                if not beams.antenna_beam_map:
                    errors.append(
                        "beams.antenna_beam_map: required when beam_mode='fits' and per_antenna=true. "
                        "Provide a mapping of antenna number to FITS beam file path."
                    )
            else:
                if not beams.beam_file:
                    errors.append(
                        "beams.beam_file: required when beam_mode='fits' and per_antenna=false. "
                        "Provide the path to a FITS beam file."
                    )
        elif beams.beam_mode == "mixed":
            if not beams.antenna_beam_map:
                errors.append(
                    "beams.antenna_beam_map: required when beam_mode='mixed'. "
                    "Map each antenna number to a FITS file path or 'analytic'."
                )

        # --- Visibility sky representation ---
        if self.visibility.sky_representation is None:
            errors.append(
                "visibility.sky_representation: required but not set. "
                "Choose 'point_sources' (catalogs) or 'healpix_map' (diffuse emission)."
            )

        # --- Sky model: at least one source spec must be present ---
        sm = self.sky_model
        if not sm.sources:
            errors.append(
                "sky_model.sources: add at least one source entry "
                "(for example {'test_sources': {}}, {'gleam': {}}, "
                "or {'diffuse_sky': {}})."
            )
        else:
            from rrivis.core.sky.registry import loader_registry

            for idx, source in enumerate(sm.sources):
                try:
                    definition = loader_registry.definition(source.kind)
                except ValueError as exc:
                    errors.append(f"sky_model.sources[{idx}]: {exc}")
                    continue
                if definition.requires_file:
                    fname = source.filename or ""
                    if not fname:
                        errors.append(
                            f"sky_model.sources[{idx}].filename: required for "
                            f"loader '{source.kind}'."
                        )
                    elif not Path(fname).exists():
                        errors.append(
                            f"sky_model.sources[{idx}].filename: file not found: '{fname}'."
                        )

        return errors

    def generate_output_subdir(self) -> str:
        """Generate output subdirectory name from config parameters.

        Creates a descriptive, deterministic directory name based on
        simulation parameters.

        Format:
            {telescope}_{freq_start}-{freq_end}{unit}_{n_channels}channels_{obs_start}_to_{obs_end}_{n_times}chunks

        Returns
        -------
        str
            Generated subdirectory name.

        Examples
        --------
        >>> config = load_config("config.yaml")
        >>> config.generate_output_subdir()
        'HERA_100-120MHz_21channels_2025-01-15T00-00-00_to_2025-01-15T00-10-00_60chunks'
        """
        from datetime import timedelta

        from astropy.time import Time as AstropyTime

        # Frequency parameters
        telescope = self.telescope.telescope_name.replace(" ", "_")
        freq_start = int(self.obs_frequency.starting_frequency)
        freq_end = int(
            self.obs_frequency.starting_frequency
            + self.obs_frequency.frequency_bandwidth
        )
        freq_unit = self.obs_frequency.frequency_unit
        n_channels = self.obs_frequency.n_channels

        # Time parameters
        duration = int(self.obs_time.duration_seconds)
        obs_start_dt = AstropyTime(self.obs_time.start_time).to_datetime()
        obs_end_dt = obs_start_dt + timedelta(seconds=duration)

        def _fmt(dt: datetime) -> str:
            return dt.strftime("%Y-%m-%dT%H-%M-%S")

        obs_start = _fmt(obs_start_dt)
        obs_end = _fmt(obs_end_dt)
        n_times = max(1, int(duration / self.obs_time.time_step_seconds))

        return (
            f"{telescope}_{freq_start}-{freq_end}{freq_unit}_"
            f"{n_channels}channels_{obs_start}_to_{obs_end}_{n_times}chunks"
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
    output_path = Path(output_path)
    with open(output_path, "w") as f:
        yaml.dump(
            config.model_dump(),  # include all fields, including nulls
            f,
            default_flow_style=False,
            sort_keys=True,  # alphabetical order
        )
