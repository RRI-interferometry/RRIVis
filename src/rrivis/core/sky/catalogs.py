# rrivis/core/sky/catalogs.py
"""Catalog metadata for VizieR, CASDA TAP, and diffuse sky models.

Each catalog entry is a frozen Pydantic model that validates required
fields at import time rather than raising runtime ``KeyError``s.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class _CatalogBase(BaseModel):
    """Base class for all catalog metadata entries."""

    model_config = {"frozen": True}

    description: str = Field(
        ..., description="Short catalog description (1-2 sentences)"
    )
    reference_url: str = Field("", description="ADS or documentation URL")


class DiffuseModelEntry(_CatalogBase):
    """Metadata for a diffuse sky model (pygdsm)."""

    class_path: str = Field(..., description="Fully qualified Python class path")
    freq_range: tuple[float, float] = Field(
        ..., description="Valid frequency range in Hz (min, max)"
    )
    init_kwargs: dict[str, object] = Field(
        default_factory=dict, description="Constructor keyword arguments"
    )


class VizierCatalogEntry(_CatalogBase):
    """Metadata for a VizieR point-source catalog."""

    vizier_id: str = Field(
        ..., description="VizieR catalog identifier (e.g. 'VIII/97')"
    )
    table: str | None = Field(
        default=None, description="VizieR table name (None = first table)"
    )
    ra_col: str = Field(..., description="RA column name in the catalog")
    dec_col: str = Field(..., description="Declination column name")
    flux_col: str = Field(..., description="Flux density column name")
    flux_unit: str = Field(
        "Jy", description="Unit of flux column ('Jy', 'mJy', 'Jy/beam')"
    )
    spindex_col: str | None = Field(
        default=None, description="Spectral index column (None if unavailable)"
    )
    default_spindex: float = Field(
        -0.7, description="Default spectral index when column is missing or NaN"
    )
    freq_mhz: float = Field(..., description="Survey reference frequency in MHz")
    coords_sexagesimal: bool = Field(
        False, description="True if coordinates are sexagesimal strings"
    )
    coord_frame: str = Field("icrs", description="Coordinate frame ('icrs' or 'fk4')")
    major_col: str | None = Field(
        default=None, description="Major axis column for Gaussian sources"
    )
    minor_col: str | None = Field(
        default=None, description="Minor axis column for Gaussian sources"
    )
    pa_col: str | None = Field(
        default=None, description="Position angle column for Gaussian sources"
    )
    default_flux_limit: float = Field(
        1.0, description="Default minimum flux density in Jy for the loader"
    )


class RacsCatalogEntry(_CatalogBase):
    """Metadata for a RACS catalog accessed via CASDA TAP."""

    freq_mhz: float = Field(..., description="Survey frequency in MHz")
    tap_table: str = Field(..., description="CASDA TAP table name")
    ra_col: str = Field(..., description="RA column name")
    dec_col: str = Field(..., description="Declination column name")
    flux_col: str = Field(..., description="Flux density column name")
    flux_unit: str = Field("mJy", description="Unit of flux column")


# =============================================================================
# Diffuse sky model metadata (pygdsm)
# =============================================================================

DIFFUSE_MODELS: dict[str, DiffuseModelEntry] = {
    "gsm2008": DiffuseModelEntry(
        class_path="pygdsm.GlobalSkyModel",
        description=(
            "Global Sky Model 2008 (de Oliveira-Costa et al. 2008). "
            "PCA-based all-sky diffuse Galactic emission, 10 MHz - 94 GHz, "
            "nside=512. Basemaps: haslam (1 deg), wmap (2 deg), 5deg."
        ),
        reference_url="https://ui.adsabs.harvard.edu/abs/2008MNRAS.388..247D/abstract",
        freq_range=(10e6, 94e9),
        init_kwargs={
            "freq_unit": "Hz",
            "basemap": "haslam",
            "interpolation": "pchip",
            "include_cmb": False,
        },
    ),
    "gsm2016": DiffuseModelEntry(
        class_path="pygdsm.GlobalSkyModel16",
        description=(
            "Global Sky Model 2016 (Zheng et al. 2017). "
            "Improved PCA model from 29 maps, 10 MHz - 5 THz, "
            "including Planck data. Supersedes GSM2008."
        ),
        reference_url="https://ui.adsabs.harvard.edu/abs/2017MNRAS.464.3486Z/abstract",
        freq_range=(10e6, 5e12),
        init_kwargs={"freq_unit": "Hz", "data_unit": "TRJ", "include_cmb": False},
    ),
    "lfsm": DiffuseModelEntry(
        class_path="pygdsm.LowFrequencySkyModel",
        description=(
            "Low Frequency Sky Model (Dowell et al. 2017). "
            "PCA model optimized for 10-408 MHz using LWA1 data. "
            "Best for EoR / 21-cm / low-frequency interferometry."
        ),
        reference_url="https://ui.adsabs.harvard.edu/abs/2017MNRAS.469.4537D/abstract",
        freq_range=(10e6, 408e6),
        init_kwargs={"freq_unit": "Hz", "include_cmb": False},
    ),
    "haslam": DiffuseModelEntry(
        class_path="pygdsm.HaslamSkyModel",
        description=(
            "Haslam 408 MHz all-sky survey (Haslam et al. 1982, "
            "reprocessed Remazeilles et al. 2015). "
            "Single power-law extrapolation, nside=512."
        ),
        reference_url="https://ui.adsabs.harvard.edu/abs/1982A%26AS...47....1H/abstract",
        freq_range=(10e6, 100e9),
        init_kwargs={
            "freq_unit": "Hz",
            "spectral_index": -2.6,
            "include_cmb": False,
        },
    ),
}

# CASDA TAP endpoint for RACS catalogs
CASDA_TAP_URL = "https://casda.csiro.au/casda_vo_tools/tap"

# =============================================================================
# VizieR point-source catalogs
# =============================================================================

VIZIER_POINT_CATALOGS: dict[str, VizierCatalogEntry] = {
    "vlssr": VizierCatalogEntry(
        vizier_id="VIII/97",
        description=(
            "VLSSr: VLA Low-Frequency Sky Survey Redux (Lane et al. 2014). "
            "73.8 MHz, ~93k sources, 75 arcsec resolution."
        ),
        reference_url="https://ui.adsabs.harvard.edu/abs/2014MNRAS.440..327L/abstract",
        ra_col="RAJ2000",
        dec_col="DEJ2000",
        flux_col="Sp",
        flux_unit="Jy/beam",
        default_spindex=-0.7,
        freq_mhz=73.8,
        coords_sexagesimal=True,
    ),
    "tgss": VizierCatalogEntry(
        default_flux_limit=0.1,
        vizier_id="J/A+A/598/A78",
        description=(
            "TGSS ADR1: GMRT 150 MHz Sky Survey (Intema et al. 2017). "
            "~624k sources, 25 arcsec resolution."
        ),
        reference_url="https://ui.adsabs.harvard.edu/abs/2017A&A...598A..78I/abstract",
        ra_col="RAJ2000",
        dec_col="DEJ2000",
        flux_col="Stotal",
        flux_unit="mJy",
        default_spindex=-0.7,
        freq_mhz=150.0,
        major_col="Maj",
        minor_col="Min",
        pa_col="PA",
    ),
    "wenss": VizierCatalogEntry(
        default_flux_limit=0.05,
        vizier_id="VIII/62",
        description=(
            "WENSS: Westerbork Northern Sky Survey (Rengelink et al. 1997). "
            "325 MHz, ~229k sources, dec > +28 deg."
        ),
        reference_url="https://ui.adsabs.harvard.edu/abs/1997A&AS..124..259R/abstract",
        ra_col="RAJ2000",
        dec_col="DEJ2000",
        flux_col="Sint",
        flux_unit="mJy",
        default_spindex=-0.7,
        freq_mhz=325.0,
        coords_sexagesimal=True,
    ),
    "sumss": VizierCatalogEntry(
        default_flux_limit=0.008,
        vizier_id="VIII/81B",
        description=(
            "SUMSS: Sydney University Molonglo Sky Survey (Mauch et al. 2003). "
            "843 MHz, ~211k sources, dec < -30 deg."
        ),
        reference_url="https://ui.adsabs.harvard.edu/abs/2003MNRAS.342.1117M/abstract",
        ra_col="RAJ2000",
        dec_col="DEJ2000",
        flux_col="St",
        flux_unit="mJy",
        default_spindex=-0.7,
        freq_mhz=843.0,
        major_col="dMajAxis",
        minor_col="dMinAxis",
        pa_col="dPA",
    ),
    "nvss": VizierCatalogEntry(
        default_flux_limit=0.0025,
        vizier_id="VIII/65",
        description=(
            "NVSS: NRAO VLA Sky Survey (Condon et al. 1998). "
            "1.4 GHz, ~1.8M sources, 45 arcsec resolution."
        ),
        reference_url="https://ui.adsabs.harvard.edu/abs/1998AJ....115.1693C/abstract",
        ra_col="RAJ2000",
        dec_col="DEJ2000",
        flux_col="S1.4",
        flux_unit="mJy",
        default_spindex=-0.7,
        freq_mhz=1400.0,
        major_col="MajAxis",
        minor_col="MinAxis",
        pa_col="PA",
    ),
    "vlass": VizierCatalogEntry(
        default_flux_limit=0.001,
        vizier_id="J/ApJS/255/30",
        table="comp",
        description=(
            "VLASS QL Ep.1: VLA Sky Survey Quick Look (Gordon et al. 2021). "
            "3 GHz, ~1.9M sources, 2.5 arcsec resolution."
        ),
        reference_url="https://ui.adsabs.harvard.edu/abs/2021ApJS..255...30G/abstract",
        ra_col="RAJ2000",
        dec_col="DEJ2000",
        flux_col="Ftot",
        flux_unit="mJy",
        default_spindex=-0.7,
        freq_mhz=3000.0,
        major_col="DCMaj",
        minor_col="DCMin",
        pa_col="DCPA",
    ),
    "lotss_dr1": VizierCatalogEntry(
        vizier_id="J/A+A/622/A1",
        description=(
            "LoTSS DR1: LOFAR Two-metre Sky Survey (Shimwell et al. 2019). "
            "120-168 MHz, ~326k sources, 6 arcsec resolution."
        ),
        reference_url="https://ui.adsabs.harvard.edu/abs/2019A&A...622A...1S/abstract",
        ra_col="RAJ2000",
        dec_col="DEJ2000",
        flux_col="Sint",
        flux_unit="mJy",
        default_spindex=-0.7,
        freq_mhz=144.0,
    ),
    "lotss_dr2": VizierCatalogEntry(
        vizier_id="J/A+A/659/A1",
        description=(
            "LoTSS DR2: LOFAR Two-metre Sky Survey (Shimwell et al. 2022). "
            "120-168 MHz, ~4.4M sources, 27% of northern sky."
        ),
        reference_url="https://ui.adsabs.harvard.edu/abs/2022A&A...659A...1S/abstract",
        ra_col="RAJ2000",
        dec_col="DEJ2000",
        flux_col="SpeakTot",
        flux_unit="mJy",
        default_spindex=-0.7,
        freq_mhz=144.0,
    ),
    "3c": VizierCatalogEntry(
        vizier_id="VIII/1",
        table="3cr",
        description=(
            "3CR: Revised Third Cambridge Catalogue (Bennett 1962). "
            "178 MHz, 328 sources. B1950 FK4 coords, converted to ICRS."
        ),
        reference_url="https://ui.adsabs.harvard.edu/abs/1962MmRAS..68..163B/abstract",
        ra_col="RA1950",
        dec_col="DE1950",
        flux_col="S178MHz",
        flux_unit="Jy",
        default_spindex=-0.7,
        freq_mhz=178.0,
        coords_sexagesimal=True,
        coord_frame="fk4",
    ),
    # --- GLEAM family ---
    "gleam_egc": VizierCatalogEntry(
        vizier_id="VIII/100/gleamegc",
        description=(
            "GLEAM EGC: MWA Extragalactic Catalogue (Hurley-Walker et al. 2017). "
            "72-231 MHz, ~307k sources. Per-source spectral indices."
        ),
        reference_url="https://ui.adsabs.harvard.edu/abs/2017MNRAS.464.1146H/abstract",
        ra_col="RAJ2000",
        dec_col="DEJ2000",
        flux_col="Fpwide",
        flux_unit="Jy",
        spindex_col="alpha",
        default_spindex=-0.8,
        freq_mhz=200.0,
    ),
    "gleam_x_dr1": VizierCatalogEntry(
        vizier_id="VIII/110/catalog",
        description=(
            "GLEAM-X DR1: MWA Extended DR1 (Hurley-Walker et al. 2022). "
            "72-231 MHz, ~79k sources, higher resolution than GLEAM."
        ),
        reference_url="https://ui.adsabs.harvard.edu/abs/2022PASA...39...35H/abstract",
        ra_col="RAJ2000",
        dec_col="DEJ2000",
        flux_col="Fpwide",
        flux_unit="Jy",
        spindex_col="alpha-SP",
        default_spindex=-0.8,
        freq_mhz=200.0,
    ),
    "gleam_x_dr2": VizierCatalogEntry(
        vizier_id="VIII/113/catalog2",
        description=(
            "GLEAM-X DR2: MWA Extended DR2 (Ross et al. 2024). "
            "72-231 MHz, ~625k sources, South Galactic Pole."
        ),
        reference_url="https://ui.adsabs.harvard.edu/abs/2024PASA...41...54R/abstract",
        ra_col="RAJ2000",
        dec_col="DEJ2000",
        flux_col="Fpwide",
        flux_unit="Jy",
        spindex_col="alpha-SP",
        default_spindex=-0.8,
        freq_mhz=200.0,
    ),
    "gleam_gal": VizierCatalogEntry(
        vizier_id="VIII/102/gleamgal",
        description=(
            "GLEAM Galactic Plane: MWA Galactic component (Hurley-Walker et al. 2019). "
            "72-231 MHz, ~22k sources, |b| < 10 deg."
        ),
        reference_url="https://ui.adsabs.harvard.edu/abs/2019PASA...36...47H/abstract",
        ra_col="RAJ2000",
        dec_col="DEJ2000",
        flux_col="Fpwide",
        flux_unit="Jy",
        spindex_col="alpha",
        default_spindex=-0.8,
        freq_mhz=200.0,
    ),
    # --- MALS family ---
    "mals_dr1": VizierCatalogEntry(
        vizier_id="J/ApJS/270/33",
        table="catalog",
        description=(
            "MALS DR1: MeerKAT Absorption Line Survey (Deka et al. 2024). "
            "L-band, ~495k sources from 391 pointings."
        ),
        reference_url="https://ui.adsabs.harvard.edu/abs/2024ApJS..270...33D/abstract",
        ra_col="RAJ2000",
        dec_col="DEJ2000",
        flux_col="Flux",
        flux_unit="mJy",
        spindex_col="SpMALS",
        default_spindex=0.0,
        freq_mhz=1200.0,
    ),
    "mals_dr2": VizierCatalogEntry(
        vizier_id="J/A+A/690/A163",
        table="all",
        description=(
            "MALS DR2: MeerKAT Wideband Continuum (Wagenveld et al. 2024). "
            "L-band, ~972k sources, depth ~10 uJy/beam."
        ),
        reference_url="https://ui.adsabs.harvard.edu/abs/2024A&A...690A.163W/abstract",
        ra_col="RAJ2000",
        dec_col="DEJ2000",
        flux_col="FluxTot",
        flux_unit="mJy",
        spindex_col="SpIndex",
        default_spindex=0.0,
        freq_mhz=1284.0,
    ),
}

# =============================================================================
# RACS catalogs (CASDA TAP)
# =============================================================================

RACS_CATALOGS: dict[str, RacsCatalogEntry] = {
    "low": RacsCatalogEntry(
        freq_mhz=887.5,
        tap_table="casda.racs_dr1_sources_v2021_08_v01",
        description=(
            "RACS-Low DR1: ASKAP 887.5 MHz first all-sky survey "
            "(McConnell et al. 2020). ~2.1M sources, dec -80 to +30 deg."
        ),
        reference_url="https://ui.adsabs.harvard.edu/abs/2020PASA...37...48M/abstract",
        ra_col="ra_deg_cont",
        dec_col="dec_deg_cont",
        flux_col="flux_peak",
    ),
    "mid": RacsCatalogEntry(
        freq_mhz=1367.5,
        tap_table="casda.racs_mid_dr1_components_v01",
        description=(
            "RACS-Mid DR1: ASKAP 1367.5 MHz survey "
            "(Duchesne et al. 2024). Dec south of +49 deg."
        ),
        reference_url="https://ui.adsabs.harvard.edu/abs/2024PASA...41....3D/abstract",
        ra_col="ra_deg_cont",
        dec_col="dec_deg_cont",
        flux_col="flux_peak",
    ),
    "high": RacsCatalogEntry(
        freq_mhz=1655.5,
        tap_table="casda.racs_high_dr1_components_v01",
        description=(
            "RACS-High DR1: ASKAP 1655.5 MHz survey "
            "(Duchesne et al. 2025). ~2.7M sources, 99.2% reliability."
        ),
        reference_url="https://ui.adsabs.harvard.edu/abs/2025PASA...42...38D/abstract",
        ra_col="ra_deg_cont",
        dec_col="dec_deg_cont",
        flux_col="flux_peak",
    ),
}
