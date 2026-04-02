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
        "description": (
            "Global Sky Model 2016 (Zheng, Tegmark, Dillon, Kim, de Oliveira-Costa, "
            "MNRAS 464, 3486, 2017; arXiv:1605.04920). "
            "https://ui.adsabs.harvard.edu/abs/2017MNRAS.464.3486Z/abstract "
            "Improved PCA-based all-sky model of diffuse Galactic radio emission from "
            "10 MHz to 5 THz, superseding GSM2008 with a substantially expanded dataset "
            "and frequency range. Derived from 29 sky maps spanning radio through "
            "far-infrared, including low-frequency surveys (Guzman 45 MHz, Haslam "
            "408 MHz), mid-frequency surveys (Reich 1.42 GHz, WMAP 9-year 23-94 GHz), "
            "and high-frequency data (Planck 30-857 GHz, COBE-FIRAS up to 5 THz). "
            "Uses principal component analysis on the expanded dataset to interpolate "
            "and extrapolate across the full frequency range. Key improvements over "
            "GSM2008: ~3x more input maps, extension to 5 THz (capturing thermal dust "
            "emission in the far-infrared), and inclusion of Planck satellite data for "
            "improved accuracy at CMB frequencies. Native resolution depends on frequency "
            "(up to nside=1024 where input data permits). Output in antenna/brightness "
            "temperature (K) by default. "
            "Note: the model name 'GSM2016' refers to the arXiv preprint year "
            "(1605.04920); the journal publication date is 2017. "
            "Via pygdsm (telegraphic/pygdsm). "
            "pygdsm constructor params: "
            "freq_unit ('Hz'/'MHz'/'GHz'), "
            "data_unit ('TRJ'=Rayleigh-Jeans temperature [default], "
            "'TCMB'=CMB thermodynamic temperature, 'MJysr'=MJy/sr), "
            "include_cmb (bool, adds 2.725K monopole)."
        ),
        "freq_range": (10e6, 5e12),
        "init_kwargs": {"freq_unit": "Hz", "data_unit": "TRJ", "include_cmb": False},
    },
    "lfsm": {
        "class": LowFrequencySkyModel,
        "description": (
            "Low Frequency Sky Model (Dowell, Taylor, Schinzel, Kassim, Stovall, "
            "MNRAS 469, 4537, 2017). "
            "https://ui.adsabs.harvard.edu/abs/2017MNRAS.469.4537D/abstract "
            "PCA-based diffuse sky model optimized for 10-408 MHz, built primarily from "
            "the LWA1 Low Frequency Sky Survey (LFSS). Uses sky maps from the Long "
            "Wavelength Array station 1 (LWA1, a 256-dipole array in New Mexico) at "
            "multiple frequencies between 35-80 MHz, combined with the reprocessed "
            "Haslam 408 MHz map. The dense spectral sampling below 100 MHz -- where "
            "GSM2008 and GSM2016 have few or no direct survey anchor points -- makes "
            "LFSM substantially more reliable in the 30-200 MHz range critical for "
            "Epoch of Reionization (EoR), 21-cm cosmology, and low-frequency radio "
            "interferometry (LOFAR, MWA, HERA, NenuFAR). PCA decomposition similar "
            "to GSM2008 but fitted to the low-frequency-optimized input dataset. "
            "Native resolution varies with frequency (~2 deg at the lowest frequencies, "
            "improving at higher frequencies). Output in brightness temperature (K). "
            "Via pygdsm (telegraphic/pygdsm). "
            "pygdsm constructor params: "
            "freq_unit ('Hz'/'MHz'/'GHz'), "
            "include_cmb (bool, adds 2.725K monopole)."
        ),
        "freq_range": (10e6, 408e6),
        "init_kwargs": {"freq_unit": "Hz", "include_cmb": False},
    },
    "haslam": {
        "class": HaslamSkyModel,
        "description": (
            "Haslam 408 MHz All-Sky Continuum Survey (Haslam, Salter, Stoffel, Wilson, "
            "A&AS 47, 1, 1982; reprocessed by Remazeilles, Dickinson, Banday, "
            "Bigot-Sazy, Ghosh, MNRAS 451, 4311, 2015). "
            "https://ui.adsabs.harvard.edu/abs/1982A%26AS...47....1H/abstract "
            "The most widely used all-sky radio continuum template. The original survey "
            "compiled observations from the Jodrell Bank MkI (76 m, UK), Bonn Stockert "
            "(25 m, Germany), and Parkes (64 m, Australia) radio telescopes to achieve "
            "complete sky coverage at 408 MHz with ~56 arcmin angular resolution. "
            "Emission is dominated by Galactic synchrotron radiation (>90% of the signal "
            "at 408 MHz). The 2015 reprocessing by Remazeilles et al. reduced large-scale "
            "striping artifacts from the original scanning strategy, subtracted "
            "extragalactic point sources, and improved zero-level calibration. "
            "pygdsm's HaslamSkyModel extrapolates to other frequencies using a single "
            "power-law: T(v) = T_408 * (v/408 MHz)^beta, where beta is the spectral "
            "index (default -2.6, typical of optically thin synchrotron emission). This "
            "is the simplest spectral model available -- it assumes a spatially uniform "
            "spectral index, unlike GSM2008/GSM2016 which use PCA-derived spatial and "
            "spectral structure. Best suited for quick estimates or as a baseline "
            "comparison. Stored at nside=512 (HEALPix RING ordering). "
            "Via pygdsm (telegraphic/pygdsm). "
            "pygdsm constructor params: "
            "freq_unit ('Hz'/'MHz'/'GHz'), "
            "spectral_index (float, default=-2.6; power-law exponent for frequency "
            "scaling; typical synchrotron range: -2.4 to -2.8), "
            "include_cmb (bool, adds 2.725K monopole)."
        ),
        "freq_range": (10e6, 100e9),
        "init_kwargs": {
            "freq_unit": "Hz",
            "spectral_index": -2.6,
            "include_cmb": False,
        },
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
    "vlass": {
        "vizier_id": "J/ApJS/255/30",
        "table": "comp",
        "description": "VLASS QL Ep.1: VLA Sky Survey Quick Look (3000 MHz, ~1.9M sources)",
        "ra_col": "RAJ2000",
        "dec_col": "DEJ2000",
        "flux_col": "Ftot",
        "flux_unit": "mJy",
        "spindex_col": None,
        "default_spindex": -0.7,
        "freq_mhz": 3000.0,
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
    # --- GLEAM family ---
    "gleam_egc": {
        "vizier_id": "VIII/100/gleamegc",
        "table": None,
        "description": "GLEAM EGC catalog, version 2 (307,455 sources, 200 MHz wideband)",
        "ra_col": "RAJ2000",
        "dec_col": "DEJ2000",
        "flux_col": "Fpwide",
        "flux_unit": "Jy",
        "spindex_col": "alpha",
        "default_spindex": -0.8,
        "freq_mhz": 200.0,
        "coords_sexagesimal": False,
        "coord_frame": "icrs",
    },
    "gleam_x_dr1": {
        "vizier_id": "VIII/110/catalog",
        "table": None,
        "description": "GLEAM-X DR1 (78,967 sources, 200 MHz wideband)",
        "ra_col": "RAJ2000",
        "dec_col": "DEJ2000",
        "flux_col": "Fpwide",
        "flux_unit": "Jy",
        "spindex_col": "alpha-SP",
        "default_spindex": -0.8,
        "freq_mhz": 200.0,
        "coords_sexagesimal": False,
        "coord_frame": "icrs",
    },
    "gleam_x_dr2": {
        "vizier_id": "VIII/113/catalog2",
        "table": None,
        "description": "GLEAM-X DR2 (624,866 sources, 200 MHz wideband)",
        "ra_col": "RAJ2000",
        "dec_col": "DEJ2000",
        "flux_col": "Fpwide",
        "flux_unit": "Jy",
        "spindex_col": "alpha-SP",
        "default_spindex": -0.8,
        "freq_mhz": 200.0,
        "coords_sexagesimal": False,
        "coord_frame": "icrs",
    },
    "gleam_gal": {
        "vizier_id": "VIII/102/gleamgal",
        "table": None,
        "description": "GLEAM Galactic plane (22,037 sources, 200 MHz wideband)",
        "ra_col": "RAJ2000",
        "dec_col": "DEJ2000",
        "flux_col": "Fpwide",
        "flux_unit": "Jy",
        "spindex_col": "alpha",
        "default_spindex": -0.8,
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
