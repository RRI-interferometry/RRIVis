# rrivis/core/sky/catalogs.py
"""Catalog metadata dictionaries for VizieR, CASDA TAP, and diffuse sky models."""

from typing import Any

from pygdsm import (
    GlobalSkyModel,
    GlobalSkyModel16,
    HaslamSkyModel,
    LowFrequencySkyModel,
)


# =============================================================================
# Diffuse sky model metadata (pygdsm)
# =============================================================================

DIFFUSE_MODELS = {
    "gsm2008": {
        "class": GlobalSkyModel,
        "description": (
            "Global Sky Model 2008 (de Oliveira-Costa et al., MNRAS 388, 247, 2008). "
            "https://ui.adsabs.harvard.edu/abs/2008MNRAS.388..247D/abstract "
            "PCA-based all-sky model of diffuse Galactic radio emission from 10 MHz to "
            "94 GHz. Derived from 11 total-power radio surveys (10, 22, 45, 408 MHz; "
            "1.42, 2.326 GHz; WMAP 23, 33, 41, 61, 94 GHz). Uses 3 principal components "
            "fitted to the correlation matrix (not covariance), capturing 99.7% of "
            "variance (80% + 19% + 0.6%). Frequency interpolation via cubic spline of "
            "log(sigma) and component spectra vs log(nu). Native resolution nside=512 "
            "(HEALPix RING), output in antenna/brightness temperature (K). "
            "Accuracy ~1-10% depending on frequency and sky region. "
            "Via pygdsm (telegraphic/pygdsm). "
            "pygdsm constructor params: "
            "freq_unit ('Hz'/'MHz'/'GHz'), "
            "basemap ('haslam'=1deg locked to 408MHz [recommended <1GHz], "
            "'wmap'=2deg locked to 23GHz [recommended for CMB freqs], "
            "'5deg'=native 5.1deg PCA resolution), "
            "interpolation ('pchip'=monotone no-overshoot [pygdsm default], "
            "'cubic'=cubic spline [closer to paper, can overshoot]), "
            "include_cmb (bool, adds 2.725K). "
            "Note: pygdsm pchip/cubic interpolation differs from original Fortran "
            "gsm.f (Numerical Recipes licensed), outputs may vary by a few percent."
        ),
        "freq_range": (10e6, 94e9),
        "init_kwargs": {
            "freq_unit": "Hz",
            "basemap": "haslam",
            "interpolation": "pchip",
            "include_cmb": False,
        },
    },
    "gsm2016": {
        "class": GlobalSkyModel16,
        "description": "Global Sky Model 2016 (10 MHz - 5 THz)",
        "freq_range": (10e6, 5e12),
        "init_kwargs": {"freq_unit": "Hz", "data_unit": "TRJ", "include_cmb": False},
    },
    "lfsm": {
        "class": LowFrequencySkyModel,
        "description": "Low Frequency Sky Model (10 - 408 MHz)",
        "freq_range": (10e6, 408e6),
        "init_kwargs": {"freq_unit": "Hz", "include_cmb": False},
    },
    "haslam": {
        "class": HaslamSkyModel,
        "description": "Haslam 408 MHz map with spectral scaling",
        "freq_range": (10e6, 100e9),
        "init_kwargs": {"freq_unit": "Hz", "spectral_index": -2.6, "include_cmb": False},
    },
}

# CASDA TAP endpoint for RACS catalogs
CASDA_TAP_URL = "https://casda.csiro.au/casda_vo_tools/tap"

# Unified metadata for all VizieR-based point-source catalogs.
# Column names should be verified with:
#   Vizier.ROW_LIMIT = 1
#   print(Vizier.get_catalogs("VIII/97")[0].colnames)
VIZIER_POINT_CATALOGS: dict[str, Any] = {
    "vlssr": {
        "vizier_id": "VIII/97",
        "table": None,
        "description": "VLSSr: VLA Low-Frequency Sky Survey redux (73.8 MHz, ~92,964 sources)",
        "ra_col": "RAJ2000",
        "dec_col": "DEJ2000",
        "flux_col": "Sp",
        "flux_unit": "Jy",
        "spindex_col": None,
        "default_spindex": -0.7,
        "freq_mhz": 73.8,
        "coords_sexagesimal": True,
        "coord_frame": "icrs",
    },
    "tgss": {
        "vizier_id": "J/A+A/598/A78",
        "table": None,
        "description": "TGSS ADR1: GMRT 150 MHz All-Sky Radio Survey (150 MHz, ~623,604 sources)",
        "ra_col": "RAJ2000",
        "dec_col": "DEJ2000",
        "flux_col": "Stotal",
        "flux_unit": "mJy",
        "spindex_col": None,
        "default_spindex": -0.7,
        "freq_mhz": 150.0,
        "coords_sexagesimal": False,
        "coord_frame": "icrs",
    },
    "wenss": {
        "vizier_id": "VIII/62",
        "table": None,
        "description": "WENSS: Westerbork Northern Sky Survey (325 MHz, ~229,000 sources)",
        "ra_col": "RAJ2000",
        "dec_col": "DEJ2000",
        "flux_col": "Sint",
        "flux_unit": "mJy",
        "spindex_col": None,
        "default_spindex": -0.7,
        "freq_mhz": 325.0,
        "coords_sexagesimal": False,
        "coord_frame": "icrs",
    },
    "sumss": {
        "vizier_id": "VIII/81B",
        "table": None,
        "description": "SUMSS: Sydney University Molonglo Sky Survey (843 MHz, ~210,412 sources)",
        "ra_col": "RAJ2000",
        "dec_col": "DEJ2000",
        "flux_col": "St",
        "flux_unit": "mJy",
        "spindex_col": None,
        "default_spindex": -0.7,
        "freq_mhz": 843.0,
        "coords_sexagesimal": False,
        "coord_frame": "icrs",
    },
    "nvss": {
        "vizier_id": "VIII/65",
        "table": None,
        "description": "NVSS: NRAO VLA Sky Survey (1400 MHz, ~1.8M sources)",
        "ra_col": "RAJ2000",
        "dec_col": "DEJ2000",
        "flux_col": "S1.4",
        "flux_unit": "mJy",
        "spindex_col": None,
        "default_spindex": -0.7,
        "freq_mhz": 1400.0,
        "coords_sexagesimal": False,
        "coord_frame": "icrs",
    },
    "first": {
        "vizier_id": "VIII/92",
        "table": None,
        "description": "FIRST: Faint Images of the Radio Sky at Twenty Centimeters (1400 MHz, ~946,432 sources)",
        "ra_col": "RAJ2000",
        "dec_col": "DEJ2000",
        "flux_col": "Fint",
        "flux_unit": "mJy",
        "spindex_col": None,
        "default_spindex": -0.7,
        "freq_mhz": 1400.0,
        "coords_sexagesimal": False,
        "coord_frame": "icrs",
    },
    "lotss_dr1": {
        "vizier_id": "J/A+A/622/A1",
        "table": None,
        "description": "LoTSS DR1: LOFAR Two-metre Sky Survey DR1 (144 MHz, ~325,000 sources)",
        "ra_col": "RAJ2000",
        "dec_col": "DEJ2000",
        "flux_col": "Sint",
        "flux_unit": "mJy",
        "spindex_col": None,
        "default_spindex": -0.7,
        "freq_mhz": 144.0,
        "coords_sexagesimal": False,
        "coord_frame": "icrs",
    },
    "lotss_dr2": {
        "vizier_id": "J/A+A/659/A1",
        "table": None,
        "description": "LoTSS DR2: LOFAR Two-metre Sky Survey DR2 (144 MHz, ~4.4M sources)",
        "ra_col": "RAJ2000",
        "dec_col": "DEJ2000",
        "flux_col": "SpeakTot",
        "flux_unit": "mJy",
        "spindex_col": None,
        "default_spindex": -0.7,
        "freq_mhz": 144.0,
        "coords_sexagesimal": False,
        "coord_frame": "icrs",
    },
    "at20g": {
        "vizier_id": "J/MNRAS/402/2403",
        "table": None,
        "description": "AT20G: Australia Telescope 20 GHz Survey (20,000 MHz, ~5,890 sources)",
        "ra_col": "RAJ2000",
        "dec_col": "DEJ2000",
        "flux_col": "S20",
        "flux_unit": "mJy",
        "spindex_col": None,
        "default_spindex": -0.5,
        "freq_mhz": 20000.0,
        "coords_sexagesimal": False,
        "coord_frame": "icrs",
        # Two-frequency spectral index: α = log(S5/S8) / log(4.8e9/8.6e9)
        "spindex_from_cols": {
            "s_low": "S5",
            "s_high": "S8",
            "freq_low_hz": 4.8e9,
            "freq_high_hz": 8.6e9,
        },
    },
    "3c": {
        "vizier_id": "VIII/1",
        "table": "3cr",
        "description": "3CR: Third Cambridge Catalogue (178 MHz, ~471 sources, B1950 coords)",
        "ra_col": "RA1950",
        "dec_col": "DE1950",
        "flux_col": "S178MHz",
        "flux_unit": "Jy",
        "spindex_col": None,
        "default_spindex": -0.7,
        "freq_mhz": 178.0,
        "coords_sexagesimal": True,
        "coord_frame": "fk4",  # B1950 → converted to ICRS via .icrs
    },
    "gb6": {
        "vizier_id": "VIII/40",
        "table": None,
        "description": "GB6: Green Bank 6 cm Radio Source Catalog (4850 MHz, ~75,162 sources)",
        "ra_col": "RAJ2000",
        "dec_col": "DEJ2000",
        "flux_col": "Flux",
        "flux_unit": "mJy",
        "spindex_col": None,
        "default_spindex": -0.7,
        "freq_mhz": 4850.0,
        "coords_sexagesimal": False,
        "coord_frame": "icrs",
    },
    # --- GLEAM family ---
    "gleam_egc": {
        "vizier_id": "VIII/100/gleamegc",
        "table": None,
        "description": "GLEAM EGC catalog, version 2 (307,455 sources, 76 MHz)",
        "ra_col": "RAJ2000",
        "dec_col": "DEJ2000",
        "flux_col": "Fp076",
        "flux_unit": "Jy",
        "spindex_col": "alpha",
        "default_spindex": 0.0,
        "freq_mhz": 76.0,
        "coords_sexagesimal": False,
        "coord_frame": "icrs",
    },
    "gleam_x_dr1": {
        "vizier_id": "VIII/110/catalog",
        "table": None,
        "description": "GLEAM-X DR1 (78,967 sources, 72-231 MHz)",
        "ra_col": "RAJ2000",
        "dec_col": "DEJ2000",
        "flux_col": "Fp076",
        "flux_unit": "Jy",
        "spindex_col": "alpha-SP",
        "default_spindex": 0.0,
        "freq_mhz": 76.0,
        "coords_sexagesimal": False,
        "coord_frame": "icrs",
    },
    "gleam_x_dr2": {
        "vizier_id": "VIII/113/catalog2",
        "table": None,
        "description": "GLEAM-X DR2 (624,866 sources, 72-231 MHz)",
        "ra_col": "RAJ2000",
        "dec_col": "DEJ2000",
        "flux_col": "Fp076",
        "flux_unit": "Jy",
        "spindex_col": "alpha-SP",
        "default_spindex": 0.0,
        "freq_mhz": 76.0,
        "coords_sexagesimal": False,
        "coord_frame": "icrs",
    },
    "gleam_gal": {
        "vizier_id": "VIII/102/gleamgal",
        "table": None,
        "description": "GLEAM Galactic plane (22,037 sources, 72-231 MHz)",
        "ra_col": "RAJ2000",
        "dec_col": "DEJ2000",
        "flux_col": "Fp076",
        "flux_unit": "Jy",
        "spindex_col": "alpha",
        "default_spindex": 0.0,
        "freq_mhz": 76.0,
        "coords_sexagesimal": False,
        "coord_frame": "icrs",
    },
    "gleam_sgp": {
        "vizier_id": "VIII/109/gleamsgp",
        "table": None,
        "description": "GLEAM SGP region (108,851 sources, 200 MHz fitted)",
        "ra_col": "RAJ2000",
        "dec_col": "DEJ2000",
        "flux_col": "Fintfin200",
        "flux_unit": "Jy",
        "spindex_col": "alpha",
        "default_spindex": 0.0,
        "freq_mhz": 200.0,
        "coords_sexagesimal": False,
        "coord_frame": "icrs",
    },
    "g4jy": {
        "vizier_id": "VIII/105/catalog",
        "table": None,
        "description": "G4Jy catalog: GLEAM bright sources >4 Jy (1,960 sources)",
        "ra_col": "RAJ2000",
        "dec_col": "DEJ2000",
        "flux_col": "Fintwide",
        "flux_unit": "Jy",
        "spindex_col": "alphaG4Jy",
        "default_spindex": 0.0,
        "freq_mhz": 200.0,
        "coords_sexagesimal": False,
        "coord_frame": "icrs",
    },
    # --- MALS family ---
    "mals_dr1": {
        "vizier_id": "J/ApJS/270/33",
        "table": "catalog",
        "description": "MALS DR1: Stokes I at 1-1.4 GHz (495,325 sources)",
        "ra_col": "RAJ2000",
        "dec_col": "DEJ2000",
        "flux_col": "Flux",
        "flux_unit": "mJy",
        "spindex_col": "SpMALS",
        "default_spindex": 0.0,
        "freq_mhz": 1200.0,
        "coords_sexagesimal": False,
        "coord_frame": "icrs",
    },
    "mals_dr2": {
        "vizier_id": "J/A+A/690/A163",
        "table": "all",
        "description": "MALS DR2: Wideband continuum (971,980 sources)",
        "ra_col": "RAJ2000",
        "dec_col": "DEJ2000",
        "flux_col": "FluxTot",
        "flux_unit": "mJy",
        "spindex_col": "SpIndex",
        "default_spindex": 0.0,
        "freq_mhz": 1284.0,
        "coords_sexagesimal": False,
        "coord_frame": "icrs",
    },
    "mals_dr3": {
        "vizier_id": "J/A+A/698/A120",
        "table": "catalog",
        "description": "MALS DR3: HI 21-cm absorption (3,640 features)",
        "ra_col": "RAJ2000",
        "dec_col": "DEJ2000",
        "flux_col": "PeakFlux",
        "flux_unit": "mJy",
        "spindex_col": None,
        "default_spindex": 0.0,
        "freq_mhz": 1420.0,
        "coords_sexagesimal": True,
        "coord_frame": "icrs",
    },
}

# RACS catalogs accessed via CASDA TAP (column names are best-effort;
# verify with: tap.search("SELECT column_name FROM tap_schema.columns WHERE table_name='...'"))
RACS_CATALOGS: dict[str, Any] = {
    "low": {
        "freq_mhz": 887.5,
        "tap_table": "casda.racs_dr1_sources_v2021_08_v01",
        "ra_col": "ra_deg_cont",
        "dec_col": "dec_deg_cont",
        "flux_col": "flux_peak",
        "flux_unit": "mJy",
        "description": "RACS-Low DR1: Rapid ASKAP Continuum Survey at 887.5 MHz",
    },
    "mid": {
        "freq_mhz": 1367.5,
        "tap_table": "casda.racs_mid_dr1_components_v01",
        "ra_col": "ra_deg_cont",
        "dec_col": "dec_deg_cont",
        "flux_col": "flux_peak",
        "flux_unit": "mJy",
        "description": "RACS-Mid DR1: Rapid ASKAP Continuum Survey at 1367.5 MHz",
    },
    "high": {
        "freq_mhz": 1655.5,
        "tap_table": "casda.racs_high_dr1_components_v01",
        "ra_col": "ra_deg_cont",
        "dec_col": "dec_deg_cont",
        "flux_col": "flux_peak",
        "flux_unit": "mJy",
        "description": "RACS-High DR1: Rapid ASKAP Continuum Survey at 1655.5 MHz",
    },
}
