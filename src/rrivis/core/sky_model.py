# rrivis/core/sky_model.py
"""
Unified Sky Model for RRIVis.

This module provides a unified ``SkyModel`` class that can represent sky
brightness in two formats:

1. **Point Sources**: Discrete sources with (RA, Dec, flux, spectral_index).
   Stored internally as columnar NumPy arrays for vectorised operations.
2. **Multi-frequency HEALPix Maps**: One brightness-temperature map per
   observation channel, stored as ``{freq_hz: T_b_array}``.

Supported point-source catalogs (loaded from VizieR / CASDA TAP):

- GLEAM (76 MHz), GLEAM-X, G4Jy — ``from_gleam()``
- MALS DR1 / DR2 / DR3 (1–1.4 GHz) — ``from_mals()``
- VLSSr (73.8 MHz) — ``from_vlssr()``
- TGSS ADR1 (150 MHz) — ``from_tgss()``
- WENSS (325 MHz) — ``from_wenss()``
- SUMSS (843 MHz) — ``from_sumss()``
- NVSS (1.4 GHz) — ``from_nvss()``
- FIRST (1.4 GHz) — ``from_first()``
- LoTSS DR1 / DR2 (144 MHz) — ``from_lotss()``
- AT20G (20 GHz) — ``from_at20g()``
- 3CR (178 MHz, B1950→ICRS) — ``from_3c()``
- GB6 (4.85 GHz) — ``from_gb6()``
- RACS Low / Mid / High (887–1656 MHz, via CASDA TAP) — ``from_racs()``
- pyradiosky files (SkyH5, VOTable, text, FHD) — ``from_pyradiosky_file()``
- Synthetic test sources — ``from_test_sources()``
- User-supplied list of dicts — ``from_point_sources()``

Supported diffuse / HEALPix sky models:

- GSM2008 (10 MHz – 100 GHz, pygdsm) — ``from_diffuse_sky("gsm2008")``
- GSM2016 (10 MHz – 5 THz, pygdsm) — ``from_diffuse_sky("gsm2016")``
- LFSM (10 – 408 MHz, pygdsm) — ``from_diffuse_sky("lfsm")``
- Haslam 408 MHz (pygdsm) — ``from_diffuse_sky("haslam")``
- PySM3 presets (synchrotron, dust, free-free, …) — ``from_pysm3()``
- ULSA ultra-low-frequency (<100 MHz) — ``from_ulsa()``

Diffuse models are generated natively per-frequency using each library's full
spectral model — no single-reference-frequency power-law approximation.

Point-source catalogs can be converted to multi-frequency HEALPix maps via
``to_healpix_for_observation()``, and HEALPix maps can be decomposed back
to point sources via ``to_point_sources()``.

Examples
--------
Load a point-source catalog:

>>> sky = SkyModel.from_gleam(flux_limit=1.0)
>>> sources = sky.to_point_sources()

>>> sky = SkyModel.from_nvss(flux_limit=0.01)

Convert point sources to multi-frequency HEALPix for visibility simulation:

>>> config = {"starting_frequency": 100.0, "frequency_interval": 1.0,
...           "frequency_bandwidth": 20.0, "frequency_unit": "MHz"}
>>> sky = SkyModel.from_gleam(flux_limit=1.0)
>>> sky.to_healpix_for_observation(nside=64, obs_frequency_config=config)

Load diffuse models directly as multi-frequency HEALPix:

>>> freqs = np.linspace(100e6, 120e6, 20)
>>> sky = SkyModel.from_diffuse_sky(model="gsm2008", nside=32, frequencies=freqs)

>>> sky = SkyModel.from_pysm3(components=["s1", "d1"], nside=64, frequencies=freqs)

Load from a local pyradiosky file:

>>> sky = SkyModel.from_pyradiosky_file("catalog.skyh5", flux_limit=0.5)

Load RACS via CASDA TAP:

>>> sky = SkyModel.from_racs(band="low", flux_limit=0.01)

Combine multiple sky models:

>>> combined = SkyModel.combine(
...     [gleam_sky, gsm_sky],
...     representation="healpix_map",
...     obs_frequency_config=config,
... )

References
----------
Point-source catalogs:

- GLEAM: Hurley-Walker et al. (2017), MNRAS, 464, 1146
- MALS DR1: Deka et al. (2024), ApJS, 270, 33
- MALS DR2: Wagenveld et al. (2024), A&A, 690, A163
- VLSS: Cohen et al. (2007), AJ, 134, 1245
- VLSSr: Lane et al. (2014), MNRAS, 440, 327
- TGSS ADR1: Intema et al. (2017), A&A, 598, A78
- WENSS: Rengelink et al. (1997), A&AS, 124, 259
- SUMSS: Mauch et al. (2003), MNRAS, 342, 1117
- NVSS: Condon et al. (1998), AJ, 115, 1693
- FIRST: White et al. (1997), ApJ, 475, 479
- LoTSS DR1: Shimwell et al. (2019), A&A, 622, A1
- LoTSS DR2: Shimwell et al. (2022), A&A, 659, A1
- AT20G: Murphy et al. (2010), MNRAS, 402, 2403
- 3CR: Edge et al. (1959), MmRAS, 68, 37
- GB6: Gregory et al. (1996), ApJS, 103, 427
- RACS: McConnell et al. (2020), PASA, 37, e048

Diffuse sky models:

- GSM2008: de Oliveira-Costa et al. (2008), MNRAS, 388, 247
- GSM2016: Zheng et al. (2016), MNRAS, 464, 3486
- LFSM: Dowell et al. (2017), MNRAS, 469, 4537
- Haslam: Remazeilles et al. (2015), MNRAS, 451, 4311
- PySM3: Thorne et al. (2017), MNRAS, 469, 2821;
  Zonca et al. (2021), JOSS, 6, 3783
- ULSA: Cong et al. (2021), ApJ, 914, 128
"""

import logging
import os
import warnings
from dataclasses import dataclass, field
from typing import Any

import astropy.units as u
import healpy as hp
import numpy as np
import pysm3
import pysm3.units as pysm3_u
from astropy.coordinates import SkyCoord
from astroquery.utils.tap.core import TapPlus
from astroquery.vizier import Vizier
from healpy.rotator import Rotator
from pyradiosky import SkyModel as PyRadioSkyModel
from pygdsm import (
    GlobalSkyModel,
    GlobalSkyModel16,
    GSMObserver08,
    HaslamSkyModel,
    LowFrequencySkyModel,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Physical Constants
# =============================================================================

K_BOLTZMANN = 1.380649e-23  # Boltzmann constant (J/K)
C_LIGHT = 299792458  # Speed of light (m/s)
H_PLANCK = 6.62607015e-34  # Planck constant (J·s)


def brightness_temp_to_flux_density(
    temperature: np.ndarray,
    frequency: float,
    solid_angle: float,
    method: str = "planck",
) -> np.ndarray:
    """
    Convert brightness temperature to flux density in Jy.

    Parameters
    ----------
    temperature : np.ndarray
        Brightness temperature in Kelvin.
    frequency : float
        Frequency in Hz.
    solid_angle : float
        Solid angle in steradians.
    method : str, default="planck"
        Conversion method: "planck" (exact) or "rayleigh-jeans" (approximation).

    Returns
    -------
    np.ndarray
        Flux density in Jy.
    """
    temperature = np.asarray(temperature, dtype=np.float64)
    if method == "rayleigh-jeans":
        # S = (2 k_B T ν²/c²) × Ω / 1e-26
        return (2 * K_BOLTZMANN * temperature * frequency**2 / C_LIGHT**2) * solid_angle / 1e-26

    # Planck-exact: S = (2hν³/c²) / expm1(hν/kT) × Ω / 1e-26
    if np.any(temperature <= 0):
        bad = np.where(temperature <= 0)[0]
        raise ValueError(
            f"brightness_temp_to_flux_density: temperature must be strictly positive, "
            f"but found {len(bad)} pixel(s) with T ≤ 0 "
            f"(min value: {temperature[bad].min():.6g} K). "
            f"Filter zero/negative pixels before calling this function."
        )
    x = H_PLANCK * frequency / (K_BOLTZMANN * temperature)
    intensity = (2 * H_PLANCK * frequency**3 / C_LIGHT**2) / np.expm1(x)
    return intensity * solid_angle / 1e-26


def flux_density_to_brightness_temp(
    flux_jy: np.ndarray,
    frequency: float,
    solid_angle: float,
    method: str = "planck",
) -> np.ndarray:
    """
    Convert flux density in Jy to brightness temperature in Kelvin.

    Parameters
    ----------
    flux_jy : np.ndarray
        Flux density in Jy.
    frequency : float
        Frequency in Hz.
    solid_angle : float
        Solid angle in steradians.
    method : str, default="planck"
        Conversion method: "planck" (exact) or "rayleigh-jeans" (approximation).

    Returns
    -------
    np.ndarray
        Brightness temperature in Kelvin.
    """
    flux_jy = np.asarray(flux_jy, dtype=np.float64)
    if method == "rayleigh-jeans":
        # T = S c² / (2 k_B ν² Ω) × 1e-26
        return flux_jy * C_LIGHT**2 / (2 * K_BOLTZMANN * frequency**2 * solid_angle) * 1e-26

    # Planck-exact: T = hν / (k ln(1 + 2hν³/(c² I_ν)))
    # where I_ν = S × 1e-26 / Ω
    if np.any(flux_jy <= 0):
        bad = np.where(flux_jy <= 0)[0]
        raise ValueError(
            f"flux_density_to_brightness_temp: flux density must be strictly positive, "
            f"but found {len(bad)} pixel(s) with S ≤ 0 "
            f"(min value: {flux_jy[bad].min():.6g} Jy). "
            f"Filter zero/negative pixels before calling this function."
        )
    I_nu = flux_jy * 1e-26 / solid_angle
    ratio = 2 * H_PLANCK * frequency**3 / (C_LIGHT**2 * I_nu)
    return H_PLANCK * frequency / (K_BOLTZMANN * np.log1p(ratio))


# =============================================================================
# Catalog Metadata (VizieR, CASDA, etc.)
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


# =============================================================================
# SkyModel Class
# =============================================================================

@dataclass
class SkyModel:
    """
    Unified sky model with bidirectional conversion.

    Can hold sky data in either point source or HEALPix format, and convert
    between them on demand. This allows uniform treatment of all sky models
    (catalogs, diffuse emission, test sources) regardless of their native format.

    Columnar point-source arrays are ``(N_sources,)`` float64. ``None`` means
    not populated; ``len == 0`` means loaded but no sources passed the filter.

    Attributes
    ----------
    _ra_rad, _dec_rad : np.ndarray, optional
        Right ascension and declination in radians.
    _flux_ref : np.ndarray, optional
        Reference flux density in Jy.
    _alpha : np.ndarray, optional
        Spectral index for power-law extrapolation.
    _stokes_q, _stokes_u, _stokes_v : np.ndarray, optional
        Stokes polarization parameters.
    _healpix_maps : dict[float, np.ndarray], optional
        Multi-frequency HEALPix brightness temperature maps (freq_hz -> T_b).
        One map per observation channel; generated natively by pygdsm for
        diffuse models, or via spectral extrapolation for point source catalogs.
    _healpix_nside : int, optional
        HEALPix NSIDE parameter.
    _native_format : str
        Original format: ``"point_sources"`` or ``"healpix"``.
    frequency : float, optional
        Reference frequency in Hz (first channel for multi-freq maps).
    model_name : str, optional
        Name of the sky model.
    brightness_conversion : str
        Conversion method for T_b / Jy: ``"planck"`` or ``"rayleigh-jeans"``.
    _point_sources_frequency : float, optional
        Frequency used when columnar arrays were populated from a HEALPix map.
        Only meaningful when ``_native_format == "healpix"``; used to
        invalidate the cache when a different frequency is requested.
    """

    _ra_rad:   np.ndarray | None = field(default=None, repr=False)
    _dec_rad:  np.ndarray | None = field(default=None, repr=False)
    _flux_ref: np.ndarray | None = field(default=None, repr=False)
    _alpha:    np.ndarray | None = field(default=None, repr=False)
    _stokes_q: np.ndarray | None = field(default=None, repr=False)
    _stokes_u: np.ndarray | None = field(default=None, repr=False)
    _stokes_v: np.ndarray | None = field(default=None, repr=False)

    _healpix_maps: dict[float, np.ndarray] | None = None
    _observation_frequencies: np.ndarray | None = None
    _healpix_nside: int | None = None

    _native_format: str = "point_sources"

    frequency: float | None = None
    model_name: str | None = None
    brightness_conversion: str = "planck"

    _point_sources_frequency: float | None = field(default=None, repr=False)

    _precision: Any = field(default=None, repr=False)

    _pygdsm_instance: Any = field(default=None, repr=False)

    # =========================================================================
    # Precision Helpers
    # =========================================================================

    def _validate_precision(self) -> None:
        """Raise ValueError if _precision is not set."""
        if self._precision is None:
            raise ValueError(
                "SkyModel requires a PrecisionConfig. Pass precision=PrecisionConfig.standard() "
                "(or .fast(), .precise(), .ultra()) to the factory method. "
                "Example: SkyModel.from_test_sources(num_sources=100, "
                "precision=PrecisionConfig.standard())"
            )

    def _source_dtype(self) -> np.dtype:
        """Get dtype for source position arrays (RA/Dec)."""
        self._validate_precision()
        return self._precision.sky_model.get_dtype("source_positions")

    def _flux_dtype(self) -> np.dtype:
        """Get dtype for flux and Stokes arrays."""
        self._validate_precision()
        return self._precision.sky_model.get_dtype("flux")

    def _alpha_dtype(self) -> np.dtype:
        """Get dtype for spectral index arrays."""
        self._validate_precision()
        return self._precision.sky_model.get_dtype("spectral_index")

    def _healpix_dtype(self) -> np.dtype:
        """Get dtype for HEALPix brightness temperature maps."""
        self._validate_precision()
        return self._precision.sky_model.get_dtype("healpix_maps")

    def _ensure_dtypes(self) -> None:
        """Cast internal arrays to precision-appropriate dtypes.

        Intermediate calculations (e.g. flux accumulation via ``np.bincount``)
        remain at float64 for numerical correctness; only the final stored
        arrays are cast here.

        Raises
        ------
        ValueError
            If ``_precision`` is None.
        """
        self._validate_precision()

        src_dt = self._source_dtype()
        flux_dt = self._flux_dtype()
        alpha_dt = self._alpha_dtype()
        hp_dt = self._healpix_dtype()

        if self._ra_rad is not None:
            self._ra_rad = self._ra_rad.astype(src_dt, copy=False)
            self._dec_rad = self._dec_rad.astype(src_dt, copy=False)
        if self._flux_ref is not None:
            self._flux_ref = self._flux_ref.astype(flux_dt, copy=False)
        if self._alpha is not None:
            self._alpha = self._alpha.astype(alpha_dt, copy=False)
        if self._stokes_q is not None:
            self._stokes_q = self._stokes_q.astype(flux_dt, copy=False)
            self._stokes_u = self._stokes_u.astype(flux_dt, copy=False)
            self._stokes_v = self._stokes_v.astype(flux_dt, copy=False)
        if self._healpix_maps is not None:
            self._healpix_maps = {
                f: m.astype(hp_dt, copy=False) for f, m in self._healpix_maps.items()
            }

    @classmethod
    def _empty_sky(
        cls,
        model_name: str,
        brightness_conversion: str = "planck",
        precision: Any = None,
    ) -> "SkyModel":
        """Return an empty point-source SkyModel (zero-length arrays)."""
        return cls(
            _ra_rad=np.zeros(0, dtype=np.float64),
            _dec_rad=np.zeros(0, dtype=np.float64),
            _flux_ref=np.zeros(0, dtype=np.float64),
            _alpha=np.zeros(0, dtype=np.float64),
            _stokes_q=np.zeros(0, dtype=np.float64),
            _stokes_u=np.zeros(0, dtype=np.float64),
            _stokes_v=np.zeros(0, dtype=np.float64),
            model_name=model_name,
            brightness_conversion=brightness_conversion,
            _precision=precision,
        )

    # =========================================================================
    # Properties
    # =========================================================================

    @property
    def mode(self) -> str:
        """Return the current representation mode."""
        if self._healpix_maps is not None:
            return "healpix_multifreq"
        return "point_sources"

    @property
    def has_multifreq_maps(self) -> bool:
        """Return True if multi-frequency HEALPix maps are available."""
        return self._healpix_maps is not None

    @property
    def n_frequencies(self) -> int:
        """Return the number of frequency channels (0 if no multi-freq maps)."""
        if self._observation_frequencies is not None:
            return len(self._observation_frequencies)
        return 0

    @property
    def native_format(self) -> str:
        """Return the native/original format of this sky model."""
        return self._native_format

    @property
    def pygdsm_model(self) -> Any:
        """Return the retained pygdsm model instance, or None.

        Only populated when ``retain_pygdsm_instance=True`` was passed to
        ``from_diffuse_sky()``. The returned object is the raw pygdsm model
        (e.g. ``GlobalSkyModel``, ``GlobalSkyModel16``, etc.) and exposes
        methods like ``generate()``, ``view()``, ``write_fits()``, and
        attributes like ``generated_map_data``, ``basemap``, ``nside``.

        .. warning::

            Mutating the returned instance (e.g. calling ``generate()`` at a
            new frequency) does **not** update the SkyModel's stored HEALPix
            maps. The instance is provided for read-only / standalone use.

        Returns
        -------
        object or None
            The pygdsm model instance, or None if not retained.
        """
        return self._pygdsm_instance

    @property
    def n_sources(self) -> int:
        """Return the number of sources/pixels."""
        if self._healpix_maps is not None:
            first_map = next(iter(self._healpix_maps.values()))
            return len(first_map)
        return len(self._ra_rad) if self._ra_rad is not None else 0

    def _has_point_sources(self) -> bool:
        """Return True if columnar point-source arrays are populated and non-empty."""
        return self._ra_rad is not None and len(self._ra_rad) > 0

    # =========================================================================
    # Helper Methods
    # =========================================================================

    @staticmethod
    def _parse_frequency_config(obs_frequency_config: dict[str, Any]) -> np.ndarray:
        """
        Parse observation frequency configuration to array of frequencies in Hz.

        Parameters
        ----------
        obs_frequency_config : dict
            Observation frequency configuration with keys:
            - starting_frequency: float
            - frequency_interval: float (channel width)
            - frequency_bandwidth: float (total bandwidth)
            - frequency_unit: str ("Hz", "kHz", "MHz", "GHz")

        Returns
        -------
        frequencies : np.ndarray
            Array of frequency channel centers in Hz.

        Examples
        --------
        >>> config = {
        ...     "starting_frequency": 100.0,
        ...     "frequency_interval": 1.0,
        ...     "frequency_bandwidth": 20.0,
        ...     "frequency_unit": "MHz"
        ... }
        >>> freqs = SkyModel._parse_frequency_config(config)
        >>> len(freqs)  # 20 channels
        20
        >>> freqs[0] / 1e6  # First channel at 100 MHz
        100.0
        """
        start_freq = obs_frequency_config["starting_frequency"]
        interval = obs_frequency_config["frequency_interval"]
        bandwidth = obs_frequency_config["frequency_bandwidth"]
        unit = obs_frequency_config.get("frequency_unit", "MHz")

        unit_factors = {"Hz": 1.0, "kHz": 1e3, "MHz": 1e6, "GHz": 1e9}
        if unit not in unit_factors:
            raise ValueError(
                f"Unknown frequency unit '{unit}'. "
                f"Supported: {list(unit_factors.keys())}"
            )

        unit_factor = unit_factors[unit]
        start_hz = start_freq * unit_factor
        interval_hz = interval * unit_factor
        bandwidth_hz = bandwidth * unit_factor

        n_channels = int(bandwidth_hz / interval_hz)
        if n_channels <= 0:
            raise ValueError(
                f"Invalid frequency config: bandwidth ({bandwidth} {unit}) must be "
                f"greater than interval ({interval} {unit})"
            )

        frequencies = np.array(
            [start_hz + i * interval_hz for i in range(n_channels)],
            dtype=np.float64
        )

        logger.debug(
            f"Parsed frequency config: {n_channels} channels from "
            f"{start_freq} to {start_freq + bandwidth - interval} {unit}"
        )

        return frequencies

    @staticmethod
    def estimate_healpix_memory(
        nside: int,
        n_frequencies: int,
        dtype: type = np.float32
    ) -> dict[str, Any]:
        """
        Estimate memory usage for multi-frequency HEALPix maps.

        Parameters
        ----------
        nside : int
            HEALPix NSIDE parameter.
        n_frequencies : int
            Number of frequency channels.
        dtype : type, default=np.float32
            Data type for maps.

        Returns
        -------
        dict
            Memory estimation with keys:
            - npix: number of pixels
            - n_freq: number of frequencies
            - bytes_per_map: bytes for one map
            - total_bytes: total memory in bytes
            - total_mb: total memory in MB
            - total_gb: total memory in GB
            - resolution_arcmin: approximate pixel resolution

        Examples
        --------
        >>> info = SkyModel.estimate_healpix_memory(nside=1024, n_frequencies=20)
        >>> print(f"Memory: {info['total_mb']:.1f} MB")
        Memory: 960.0 MB
        """
        npix = hp.nside2npix(nside)
        bytes_per_value = np.dtype(dtype).itemsize
        bytes_per_map = npix * bytes_per_value
        total_bytes = bytes_per_map * n_frequencies

        # Approximate resolution in arcminutes
        resolution_arcmin = np.sqrt(4 * np.pi / npix) * (180 / np.pi) * 60

        return {
            "npix": npix,
            "n_freq": n_frequencies,
            "bytes_per_map": bytes_per_map,
            "total_bytes": total_bytes,
            "total_mb": total_bytes / 1e6,
            "total_gb": total_bytes / 1e9,
            "resolution_arcmin": resolution_arcmin,
            "dtype": dtype.__name__,
        }

    # =========================================================================
    # Conversion Methods
    # =========================================================================

    def to_point_sources(
        self,
        flux_limit: float = 0.0,
        frequency: float | None = None
    ) -> list[dict[str, Any]]:
        """
        Get sky model as point sources.

        If native format is point sources, returns them directly.
        If in healpix_multifreq mode, converts pixels from the requested
        frequency channel (or first available) to point sources.

        Parameters
        ----------
        flux_limit : float, default=0.0
            Minimum flux in Jy to include.
        frequency : float, optional
            Frequency for T_b to Jy conversion (used to select the channel
            when in healpix_multifreq mode).

        Returns
        -------
        list of dict
            Point sources with coords, flux, spectral_index, stokes.
        """
        # If columnar arrays already populated, build list-of-dicts from them
        if self._ra_rad is not None:
            if flux_limit > 0:
                mask = self._flux_ref >= flux_limit
            else:
                mask = np.ones(len(self._ra_rad), dtype=bool)
            n = int(mask.sum())
            if n == 0:
                return []
            ra_deg  = np.rad2deg(self._ra_rad[mask])
            dec_deg = np.rad2deg(self._dec_rad[mask])
            # Bulk SkyCoord construction — much faster than per-row
            coords = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg, frame="icrs")
            flux_m  = self._flux_ref[mask]
            alpha_m = self._alpha[mask]
            sq_m    = self._stokes_q[mask]
            su_m    = self._stokes_u[mask]
            sv_m    = self._stokes_v[mask]
            return [
                {
                    "coords": coords[i],
                    "flux": float(flux_m[i]),
                    "spectral_index": float(alpha_m[i]),
                    "stokes_q": float(sq_m[i]),
                    "stokes_u": float(su_m[i]),
                    "stokes_v": float(sv_m[i]),
                }
                for i in range(n)
            ]

        # Convert from multi-frequency HEALPix (lazy: populate arrays on first call)
        if self._healpix_maps is not None:
            freq = frequency or self.frequency or next(iter(self._healpix_maps))
            temp_map = self.get_map_at_frequency(freq)
            self._healpix_map_to_arrays(temp_map, freq)
            # Now _ra_rad is populated; recurse to build list-of-dicts
            return self.to_point_sources(flux_limit=flux_limit, frequency=frequency)

        return []

    def _healpix_map_to_arrays(
        self,
        temp_map: np.ndarray,
        frequency: float,
    ) -> None:
        """
        Convert a HEALPix brightness temperature map to columnar arrays.

        Populates _ra_rad, _dec_rad, _flux_ref, _alpha, _stokes_q, _stokes_u,
        _stokes_v in-place. Only positive-temperature pixels are stored
        (no flux_limit filtering here — apply that in to_point_sources()).

        Parameters
        ----------
        temp_map : np.ndarray
            Brightness temperature map in Kelvin.
        frequency : float
            Frequency in Hz for T_b → Jy conversion.
        """
        npix = len(temp_map)
        nside = hp.npix2nside(npix)
        omega = 4 * np.pi / npix

        flux_jy = np.zeros(npix, dtype=np.float64)
        pos = temp_map > 0
        if np.any(pos):
            flux_jy[pos] = brightness_temp_to_flux_density(
                temp_map[pos].astype(np.float64),
                frequency,
                omega,
                method=self.brightness_conversion,
            )

        valid_idx = np.where(flux_jy > 0)[0]
        if len(valid_idx) == 0:
            logger.warning("No pixels with positive flux in HEALPix map")
            self._ra_rad   = np.zeros(0, dtype=np.float64)
            self._dec_rad  = np.zeros(0, dtype=np.float64)
            self._flux_ref = np.zeros(0, dtype=np.float64)
            self._alpha    = np.zeros(0, dtype=np.float64)
            self._stokes_q = np.zeros(0, dtype=np.float64)
            self._stokes_u = np.zeros(0, dtype=np.float64)
            self._stokes_v = np.zeros(0, dtype=np.float64)
            return

        theta, phi = hp.pix2ang(nside, valid_idx)
        self._ra_rad   = phi                       # phi = RA in radians
        self._dec_rad  = np.pi / 2 - theta         # colatitude → declination
        self._flux_ref = flux_jy[valid_idx]
        n = len(valid_idx)
        self._alpha    = np.zeros(n, dtype=np.float64)  # no per-pixel α in HEALPix
        self._stokes_q = np.zeros(n, dtype=np.float64)
        self._stokes_u = np.zeros(n, dtype=np.float64)
        self._stokes_v = np.zeros(n, dtype=np.float64)

    def to_healpix(
        self,
        nside: int = 64,
        frequency: float | None = None
    ) -> tuple[np.ndarray, int, np.ndarray | None]:
        """
        .. deprecated::
            Single-frequency HEALPix maps are no longer supported.
            Use ``to_healpix_for_observation()`` for correct multi-frequency
            conversion that preserves the native pygdsm spectral model.

        Raises
        ------
        ValueError
            Always. Redirect callers to the multi-frequency API.
        """
        raise ValueError(
            "to_healpix() is no longer supported. "
            "Use to_healpix_for_observation(nside, obs_frequency_config) for "
            "correct multi-frequency conversion with proper spectral handling. "
            "For diffuse models, use from_diffuse_sky(frequencies=...) directly."
        )

    def _point_sources_to_healpix_multifreq(
        self,
        nside: int,
        frequencies: np.ndarray,
        ref_frequency: float = 76e6
    ) -> dict[float, np.ndarray]:
        """
        Convert point sources to multi-frequency HEALPix brightness temperature maps.

        Vectorized implementation: uses np.bincount for O(N_sources) memory per
        frequency channel instead of a Python loop over sources.

        Parameters
        ----------
        nside : int
            HEALPix NSIDE parameter.
        frequencies : np.ndarray
            Array of frequencies in Hz.
        ref_frequency : float, default=76e6
            Reference frequency for flux values (GLEAM standard: 76 MHz).

        Returns
        -------
        temp_maps : Dict[float, np.ndarray]
            Dictionary mapping frequency (Hz) to brightness temperature map (K).
        """
        if not self._has_point_sources():
            npix = hp.nside2npix(nside)
            return {float(freq): np.zeros(npix, dtype=np.float32) for freq in frequencies}

        npix = hp.nside2npix(nside)
        omega_pixel = 4 * np.pi / npix
        n_sources = len(self._ra_rad)
        n_freq = len(frequencies)

        mem_info = self.estimate_healpix_memory(nside, n_freq, np.float32)
        logger.info(
            f"Creating {n_freq} HEALPix maps (nside={nside}): "
            f"~{mem_info['total_mb']:.1f} MB"
        )

        ipix = hp.ang2pix(nside, np.pi / 2 - self._dec_rad, self._ra_rad)

        temp_maps: dict[float, np.ndarray] = {}
        for freq in frequencies:
            scale = (float(freq) / ref_frequency) ** self._alpha
            flux_f = self._flux_ref * scale

            flux_map = np.bincount(ipix, weights=flux_f, minlength=npix)

            temp_out = np.zeros(npix, dtype=np.float32)
            occupied = flux_map > 0
            if np.any(occupied):
                temp_out[occupied] = flux_density_to_brightness_temp(
                    flux_map[occupied], float(freq), omega_pixel,
                    method=self.brightness_conversion,
                ).astype(np.float32)
            temp_maps[float(freq)] = temp_out

        logger.info(
            f"Converted {n_sources} point sources to {n_freq} HEALPix maps "
            f"({frequencies[0]/1e6:.1f}-{frequencies[-1]/1e6:.1f} MHz)"
        )

        return temp_maps

    def to_healpix_for_observation(
        self,
        nside: int,
        obs_frequency_config: dict[str, Any],
        ref_frequency: float = 76e6
    ) -> "SkyModel":
        """
        Convert point sources to multi-frequency HEALPix maps for observation.

        Creates one HEALPix map per frequency channel defined in the observation
        configuration. This is the recommended method for preparing sky models
        for visibility simulation, as it correctly handles spectral extrapolation
        for each source individually.

        Parameters
        ----------
        nside : int
            HEALPix NSIDE parameter.
        obs_frequency_config : dict
            Observation frequency configuration with keys:
            - starting_frequency: float
            - frequency_interval: float (channel width)
            - frequency_bandwidth: float (total bandwidth)
            - frequency_unit: str ("Hz", "kHz", "MHz", "GHz")
        ref_frequency : float, default=76e6
            Reference frequency for flux values in Hz (GLEAM: 76 MHz).

        Returns
        -------
        SkyModel
            Self, with _healpix_maps populated.

        Raises
        ------
        ValueError
            If no point sources are available for conversion.

        Notes
        -----
        Memory usage depends on nside and number of frequency channels:
        - nside=64, 20 channels: ~4 MB
        - nside=256, 100 channels: ~600 MB
        - nside=1024, 20 channels: ~1 GB

        Use `estimate_healpix_memory()` to check before conversion.

        Examples
        --------
        >>> sky = SkyModel.from_gleam(flux_limit=1.0)
        >>> config = {
        ...     "starting_frequency": 100.0,
        ...     "frequency_interval": 1.0,
        ...     "frequency_bandwidth": 20.0,
        ...     "frequency_unit": "MHz"
        ... }
        >>> sky.to_healpix_for_observation(nside=64, obs_frequency_config=config)
        >>> sky.get_map_at_frequency(100e6)  # Get map at 100 MHz
        """
        if not self._has_point_sources():
            raise ValueError(
                "No point sources available for conversion. "
                "Load sources first using from_gleam(), from_mals(), etc."
            )

        frequencies = self._parse_frequency_config(obs_frequency_config)

        self._healpix_maps = self._point_sources_to_healpix_multifreq(
            nside, frequencies, ref_frequency
        )
        self._healpix_nside = nside
        self._observation_frequencies = frequencies

        return self

    def get_map_at_frequency(self, frequency: float) -> np.ndarray:
        """
        Get HEALPix brightness temperature map at a specific frequency.

        Parameters
        ----------
        frequency : float
            Frequency in Hz.

        Returns
        -------
        temp_map : np.ndarray
            Brightness temperature map in Kelvin.

        Raises
        ------
        ValueError
            If no multi-frequency maps are available or frequency not found.

        Notes
        -----
        If the exact frequency is not available, returns the map at the
        nearest available frequency with a warning.

        Examples
        --------
        >>> sky.to_healpix_for_observation(nside=64, obs_frequency_config=config)
        >>> map_100mhz = sky.get_map_at_frequency(100e6)
        """
        if self._healpix_maps is None:
            raise ValueError(
                "No multi-frequency HEALPix maps available. "
                "Use to_healpix_for_observation() first."
            )

        if frequency in self._healpix_maps:
            return self._healpix_maps[frequency]

        available_freqs = np.array(list(self._healpix_maps.keys()))
        idx = np.argmin(np.abs(available_freqs - frequency))
        nearest_freq = available_freqs[idx]

        freq_diff_mhz = abs(frequency - nearest_freq) / 1e6
        if freq_diff_mhz > 0.001:  # More than 1 kHz difference
            logger.warning(
                f"Exact frequency {frequency/1e6:.3f} MHz not found. "
                f"Using nearest: {nearest_freq/1e6:.3f} MHz "
                f"(diff: {freq_diff_mhz:.3f} MHz)"
            )

        return self._healpix_maps[nearest_freq]

    def get_multifreq_maps(self) -> tuple[dict[float, np.ndarray], int, np.ndarray]:
        """
        Get all multi-frequency HEALPix maps.

        Returns
        -------
        temp_maps : Dict[float, np.ndarray]
            Dictionary mapping frequency (Hz) to brightness temperature map (K).
        nside : int
            HEALPix NSIDE parameter.
        frequencies : np.ndarray
            Array of frequencies in Hz.

        Raises
        ------
        ValueError
            If no multi-frequency maps are available.

        Examples
        --------
        >>> maps, nside, freqs = sky.get_multifreq_maps()
        >>> for freq in freqs:
        ...     print(f"{freq/1e6:.1f} MHz: max T_b = {maps[freq].max():.2f} K")
        """
        if self._healpix_maps is None:
            raise ValueError(
                "No multi-frequency HEALPix maps available. "
                "Use to_healpix_for_observation() first."
            )

        return (
            self._healpix_maps,
            self._healpix_nside,
            self._observation_frequencies
        )

    def get_for_visibility(
        self,
        representation: str,
        nside: int = 64,
        flux_limit: float = 0.0,
        frequency: float | None = None
    ) -> "SkyModel":
        """
        Ensure sky model is in the requested representation.

        This is the main method to call before visibility calculation.
        It converts the sky model to the requested format if needed.

        Parameters
        ----------
        representation : str
            "point_sources" or "healpix_multifreq"
        nside : int, default=64
            HEALPix NSIDE for healpix_multifreq mode.
        flux_limit : float, default=0.0
            Minimum flux for point_sources mode.
        frequency : float, optional
            Frequency for conversion.

        Returns
        -------
        SkyModel
            Self with the requested representation populated.

        Raises
        ------
        ValueError
            If healpix_multifreq is requested but no observation frequency
            config has been set. Use to_healpix_for_observation() first.
        """
        freq = frequency or self.frequency

        if representation == "point_sources":
            # For HEALPix-native models the cached arrays have flux values
            # baked in at a specific frequency.  If a different frequency is
            # requested, invalidate the cache so conversion reruns.
            if (
                self._native_format == "healpix"
                and self._ra_rad is not None
                and freq is not None
                and self._point_sources_frequency != freq
            ):
                self._ra_rad   = None
                self._dec_rad  = None
                self._flux_ref = None
                self._alpha    = None
                self._stokes_q = None
                self._stokes_u = None
                self._stokes_v = None
                self._point_sources_frequency = None

            if self._ra_rad is None:
                # to_point_sources() populates _ra_rad etc. as a side effect
                self.to_point_sources(flux_limit=flux_limit, frequency=freq)
                if self._native_format == "healpix":
                    self._point_sources_frequency = freq
        elif representation in ("healpix_map", "healpix_multifreq"):
            if self._healpix_maps is None:
                raise ValueError(
                    "Cannot convert point sources to HEALPix on-the-fly. "
                    "Use to_healpix_for_observation(nside, obs_frequency_config) first "
                    "to create multi-frequency HEALPix maps with correct spectral handling."
                )

        return self

    # =========================================================================
    # Factory Methods
    # =========================================================================

    @classmethod
    def from_test_sources(
        cls,
        num_sources: int = 100,
        flux_range: tuple[float, float] = (2.0, 8.0),
        dec_deg: float = -30.72,
        spectral_index: float = -0.8,
        brightness_conversion: str = "planck",
        precision: Any = None,
    ) -> "SkyModel":
        """
        Generate synthetic test sources.

        Creates point sources evenly distributed in Right Ascension.

        Parameters
        ----------
        num_sources : int, default=100
            Number of sources to generate.
        flux_range : tuple, default=(2.0, 8.0)
            (min_flux, max_flux) in Jy.
        dec_deg : float, default=-30.72
            Declination for all sources (degrees).
        spectral_index : float, default=-0.8
            Spectral index for all sources.
        precision : PrecisionConfig, optional
            Precision configuration for array dtypes. If None, uses float64.

        Returns
        -------
        SkyModel
            Sky model with test sources.
        """
        n = num_sources
        if n == 1:
            ra_deg_arr = np.array([0.0])
            flux_arr   = np.array([(flux_range[0] + flux_range[1]) / 2])
        else:
            ra_deg_arr = np.array([(360.0 / n) * i for i in range(n)])
            flux_arr   = np.linspace(flux_range[0], flux_range[1], n)

        dec_deg_arr = np.full(n, dec_deg)

        logger.info(f"Generated {n} test sources")

        sky = cls(
            _ra_rad=np.deg2rad(ra_deg_arr),
            _dec_rad=np.deg2rad(dec_deg_arr),
            _flux_ref=flux_arr.astype(np.float64),
            _alpha=np.full(n, float(spectral_index)),
            _stokes_q=np.zeros(n, dtype=np.float64),
            _stokes_u=np.zeros(n, dtype=np.float64),
            _stokes_v=np.zeros(n, dtype=np.float64),
            _native_format="point_sources",
            model_name="test_sources",
            brightness_conversion=brightness_conversion,
            _precision=precision,
        )
        sky._ensure_dtypes()
        return sky

    @classmethod
    def from_gleam(
        cls,
        flux_limit: float = 1.0,
        catalog: str = "gleam_egc",
        brightness_conversion: str = "planck",
        precision: Any = None,
    ) -> "SkyModel":
        """
        Load GLEAM catalog from VizieR.

        Parameters
        ----------
        flux_limit : float, default=1.0
            Minimum flux density in Jy.
        catalog : str, default="gleam_egc"
            Catalog key in ``VIZIER_POINT_CATALOGS``. Available GLEAM
            catalogs: ``"gleam_egc"``, ``"gleam_x_dr1"``, ``"gleam_x_dr2"``,
            ``"gleam_gal"``, ``"gleam_sgp"``, ``"g4jy"``.

            Legacy VizieR IDs (e.g. ``"VIII/100/gleamegc"``) are also
            accepted for backward compatibility.
        precision : PrecisionConfig, optional
            Precision configuration for array dtypes. If None, uses float64.

        Returns
        -------
        SkyModel
            Sky model with GLEAM sources.
        """
        # Backward compat: accept old VizieR IDs
        _LEGACY_GLEAM_IDS = {
            "VIII/100/gleamegc": "gleam_egc",
            "VIII/113/catalog2": "gleam_x_dr2",
            "VIII/102/gleamgal": "gleam_gal",
            "VIII/109/gleamsgp": "gleam_sgp",
            "VIII/110/catalog": "gleam_x_dr1",
            "VIII/105/catalog": "g4jy",
        }
        key = _LEGACY_GLEAM_IDS.get(catalog, catalog)
        _gleam_keys = sorted(
            k for k in VIZIER_POINT_CATALOGS if k.startswith("gleam") or k == "g4jy"
        )
        if key not in VIZIER_POINT_CATALOGS:
            raise ValueError(
                f"Unknown GLEAM catalog '{catalog}'. Available: {_gleam_keys}"
            )
        return cls._load_from_vizier_catalog(key, flux_limit, brightness_conversion, precision)

    @classmethod
    def from_mals(
        cls,
        flux_limit: float = 1.0,
        release: str = "dr2",
        brightness_conversion: str = "planck",
        precision: Any = None,
    ) -> "SkyModel":
        """
        Load MALS catalog from VizieR.

        Parameters
        ----------
        flux_limit : float, default=1.0
            Minimum flux density in Jy. The catalog stores flux in mJy;
            unit conversion is handled internally by the generic loader.
        release : str, default="dr2"
            Data release: "dr1", "dr2", or "dr3".
        precision : PrecisionConfig, optional
            Precision configuration for array dtypes. If None, uses float64.

        Returns
        -------
        SkyModel
            Sky model with MALS sources.
        """
        key = f"mals_{release.lower()}"
        if key not in VIZIER_POINT_CATALOGS:
            raise ValueError(
                f"Unknown MALS release '{release}'. Available: 'dr1', 'dr2', 'dr3'."
            )
        return cls._load_from_vizier_catalog(key, flux_limit, brightness_conversion, precision)

    @staticmethod
    def list_diffuse_models() -> dict[str, str]:
        """List available diffuse sky models with their descriptions.

        Returns
        -------
        dict[str, str]
            Mapping of model name to description string.

        Examples
        --------
        >>> for name, desc in SkyModel.list_diffuse_models().items():
        ...     print(f"{name}: {desc[:80]}...")
        """
        return {name: info["description"] for name, info in DIFFUSE_MODELS.items()}

    @staticmethod
    def list_point_catalogs() -> dict[str, str]:
        """List available VizieR point-source catalogs with their descriptions.

        Returns
        -------
        dict[str, str]
            Mapping of catalog key to description string.

        Examples
        --------
        >>> for name, desc in SkyModel.list_point_catalogs().items():
        ...     print(f"{name}: {desc[:80]}...")
        """
        return {name: info["description"] for name, info in VIZIER_POINT_CATALOGS.items()}

    @staticmethod
    def list_racs_catalogs() -> dict[str, str]:
        """List available RACS (Rapid ASKAP Continuum Survey) bands with their descriptions.

        Returns
        -------
        dict[str, str]
            Mapping of band name to description string.

        Examples
        --------
        >>> for name, desc in SkyModel.list_racs_catalogs().items():
        ...     print(f"{name}: {desc}")
        """
        return {name: info["description"] for name, info in RACS_CATALOGS.items()}

    @staticmethod
    def list_all_models() -> dict[str, dict[str, str]]:
        """List all available sky models and catalogs with their descriptions.

        Returns
        -------
        dict[str, dict[str, str]]
            Nested mapping: category → {name: description}.
            Categories: "diffuse", "point_catalogs", "racs".

        Examples
        --------
        >>> all_models = SkyModel.list_all_models()
        >>> for category, models in all_models.items():
        ...     print(f"\\n=== {category} ===")
        ...     for name, desc in models.items():
        ...         print(f"  {name}: {desc[:80]}...")
        """
        return {
            "diffuse": SkyModel.list_diffuse_models(),
            "point_catalogs": SkyModel.list_point_catalogs(),
            "racs": SkyModel.list_racs_catalogs(),
        }

    @classmethod
    def from_diffuse_sky(
        cls,
        model: str = "gsm2008",
        nside: int = 32,
        frequencies: np.ndarray | None = None,
        obs_frequency_config: dict[str, Any] | None = None,
        include_cmb: bool | None = None,
        basemap: str | None = None,
        interpolation: str | None = None,
        retain_pygdsm_instance: bool = False,
        brightness_conversion: str = "planck",
        precision: Any = None,
    ) -> "SkyModel":
        """
        Load a diffuse sky model (GSM, LFSM, Haslam) as multi-frequency HEALPix maps.

        Calls ``pygdsm.generate(freq)`` for each observation frequency and stores
        the results as a ``{freq: T_b_map}`` dictionary. This preserves the native
        PCA spectral model of pygdsm without any two-point power-law approximation.

        Parameters
        ----------
        model : str, default="gsm2008"
            Model name: "gsm2008", "gsm2016", "lfsm", "haslam".
        nside : int, default=32
            HEALPix NSIDE resolution.
        frequencies : np.ndarray, optional
            Array of observation frequencies in Hz. Takes precedence over
            ``obs_frequency_config`` when both are provided.
        obs_frequency_config : dict, optional
            Frequency configuration dict (keys: starting_frequency,
            frequency_interval, frequency_bandwidth, frequency_unit).
            Used when ``frequencies`` is None.
        include_cmb : bool or None, default=None
            Include CMB contribution in the sky model. If None, uses the
            default from the model's ``init_kwargs`` (False for all models).
        basemap : str or None, default=None
            GSM2008-only: resolution basemap to use for PCA reconstruction.
            ``"haslam"`` (1 deg, best <1 GHz), ``"wmap"`` (2 deg, best for
            CMB frequencies), or ``"5deg"`` (native 5.1 deg PCA resolution).
            Raises ``ValueError`` if set for non-GSM2008 models.
            When None, uses the default from ``DIFFUSE_MODELS`` (``"haslam"``).
        interpolation : str or None, default=None
            GSM2008-only: frequency interpolation method.
            ``"pchip"`` (monotone, no overshoot) or ``"cubic"`` (cubic spline,
            closer to the original paper but can overshoot).
            Raises ``ValueError`` if set for non-GSM2008 models.
            When None, uses the default from ``DIFFUSE_MODELS`` (``"pchip"``).
        retain_pygdsm_instance : bool, default=False
            If True, keep the pygdsm model object on the returned SkyModel
            (accessible via the ``pygdsm_model`` property). This adds ~63 MB
            of memory overhead for GSM2008. When False (default), the pygdsm
            instance is discarded after map generation.
        brightness_conversion : str, default="planck"
            Conversion method for T_b → Jy: "planck" (exact) or "rayleigh-jeans".

        Returns
        -------
        SkyModel
            Sky model in healpix_multifreq mode with one T_b map per frequency.

        Raises
        ------
        ValueError
            If neither ``frequencies`` nor ``obs_frequency_config`` is provided,
            if the model name is unknown, or if ``basemap``/``interpolation``
            are set for a non-GSM2008 model.

        Examples
        --------
        >>> freqs = np.linspace(100e6, 120e6, 20)
        >>> sky = SkyModel.from_diffuse_sky(model="gsm2008", nside=32, frequencies=freqs)
        >>> sky.mode
        'healpix_multifreq'

        >>> sky = SkyModel.from_diffuse_sky(
        ...     model="gsm2008", nside=32, frequencies=freqs,
        ...     basemap="wmap", interpolation="cubic",
        ... )

        >>> config = {"starting_frequency": 100.0, "frequency_interval": 1.0,
        ...           "frequency_bandwidth": 20.0, "frequency_unit": "MHz"}
        >>> sky = SkyModel.from_diffuse_sky(model="lfsm", nside=64,
        ...                                obs_frequency_config=config)
        """
        model = model.lower()
        if model not in DIFFUSE_MODELS:
            raise ValueError(f"Unknown model '{model}'. Available: {list(DIFFUSE_MODELS.keys())}")

        if basemap is not None and model != "gsm2008":
            raise ValueError(
                f"'basemap' is only supported for gsm2008, not '{model}'. "
                f"Remove the basemap parameter or use model='gsm2008'."
            )
        if interpolation is not None and model != "gsm2008":
            raise ValueError(
                f"'interpolation' is only supported for gsm2008, not '{model}'. "
                f"Remove the interpolation parameter or use model='gsm2008'."
            )

        if frequencies is None and obs_frequency_config is None:
            raise ValueError(
                "Either 'frequencies' or 'obs_frequency_config' must be provided. "
                "Example: from_diffuse_sky(model='gsm2008', nside=32, "
                "frequencies=np.linspace(100e6, 120e6, 20))"
            )

        if frequencies is None:
            frequencies = cls._parse_frequency_config(obs_frequency_config)
        frequencies = np.asarray(frequencies, dtype=np.float64)

        info = DIFFUSE_MODELS[model]
        model_class = info["class"]
        n_freq = len(frequencies)

        logger.info(
            f"Loading {model.upper()}: {n_freq} frequencies "
            f"({frequencies[0]/1e6:.1f}–{frequencies[-1]/1e6:.1f} MHz), nside={nside}"
        )
        logger.info(f"Model info: {info['description']}")

        init_kwargs = info["init_kwargs"].copy()
        if include_cmb is not None:
            init_kwargs["include_cmb"] = include_cmb
        if basemap is not None:
            init_kwargs["basemap"] = basemap
        if interpolation is not None:
            init_kwargs["interpolation"] = interpolation
        pygdsm_instance = model_class(**init_kwargs)

        rot = Rotator(coord=["G", "C"])

        healpix_maps: dict[float, np.ndarray] = {}
        for freq in frequencies:
            temp_map = pygdsm_instance.generate(freq)
            current_nside = hp.get_nside(temp_map)
            if current_nside != nside:
                temp_map = hp.ud_grade(temp_map, nside_out=nside)
            temp_map = rot.rotate_map_pixel(temp_map)
            healpix_maps[float(freq)] = temp_map.astype(np.float32)

        logger.info(
            f"{model.upper()} loaded: {hp.nside2npix(nside)} pixels × {n_freq} frequencies"
        )

        result = cls(
            _healpix_maps=healpix_maps,
            _healpix_nside=nside,
            _observation_frequencies=frequencies,
            _native_format="healpix",
            frequency=float(frequencies[0]),
            model_name=model,
            brightness_conversion=brightness_conversion,
            _precision=precision,
            _pygdsm_instance=pygdsm_instance if retain_pygdsm_instance else None,
        )
        result._ensure_dtypes()
        return result

    @staticmethod
    def create_gsm_observer(
        basemap: str = "haslam",
        interpolation: str = "pchip",
        include_cmb: bool = False,
    ) -> "GSMObserver08":
        """Create a ``GSMObserver08`` with configurable GSM2008 parameters.

        The returned observer can be used to generate simulated sky views for
        a specific location, time, and frequency using the ``pygdsm``
        observation framework.

        Parameters
        ----------
        basemap : str, default="haslam"
            Resolution basemap: ``"haslam"`` (1 deg), ``"wmap"`` (2 deg),
            or ``"5deg"`` (native 5.1 deg PCA resolution).
        interpolation : str, default="pchip"
            Frequency interpolation: ``"pchip"`` (monotone, no overshoot)
            or ``"cubic"`` (cubic spline).
        include_cmb : bool, default=False
            Include CMB contribution (2.725 K).

        Returns
        -------
        GSMObserver08
            A pygdsm observer ready for ``.generate()`` after setting
            location and time via ``.lat``, ``.lon``, ``.date``.

        Examples
        --------
        >>> obs = SkyModel.create_gsm_observer(basemap="wmap")
        >>> obs.lat = "-30.72"
        >>> obs.lon = "21.43"
        >>> obs.date = "2025-01-15T00:00:00"
        >>> obs.generate(150e6)
        """
        gsm = GlobalSkyModel(
            freq_unit="Hz",
            basemap=basemap,
            interpolation=interpolation,
            include_cmb=include_cmb,
        )
        observer = GSMObserver08()
        observer.gsm = gsm
        return observer

    @classmethod
    def _load_from_vizier_catalog(
        cls,
        catalog_key: str,
        flux_limit: float = 1.0,
        brightness_conversion: str = "planck",
        precision: Any = None,
    ) -> "SkyModel":
        """
        Load a point-source catalog from VizieR using unified metadata.

        This private helper is called by all public from_*() VizieR wrappers.
        It handles flux unit conversion (mJy→Jy), coordinate parsing
        (decimal/sexagesimal, ICRS/FK4), and spectral index extraction.

        Parameters
        ----------
        catalog_key : str
            Key into ``VIZIER_POINT_CATALOGS`` (e.g. "vlssr", "nvss").
        flux_limit : float, default=1.0
            Minimum flux density in Jy; rows below this are skipped.
        brightness_conversion : str, default="planck"
            Conversion method: "planck" or "rayleigh-jeans".

        Returns
        -------
        SkyModel
            Sky model with loaded point sources.

        Raises
        ------
        ValueError
            If ``catalog_key`` is not in ``VIZIER_POINT_CATALOGS``.
        """
        if catalog_key not in VIZIER_POINT_CATALOGS:
            raise ValueError(
                f"Unknown VizieR catalog key '{catalog_key}'. "
                f"Available: {sorted(VIZIER_POINT_CATALOGS.keys())}"
            )

        def _empty():
            return cls._empty_sky(catalog_key, brightness_conversion, precision)

        info = VIZIER_POINT_CATALOGS[catalog_key]
        logger.info(f"Fetching {info['description']}")
        logger.info("Downloading from VizieR...")

        try:
            v = Vizier(columns=["**"], row_limit=-1)
            tables = v.get_catalogs(info["vizier_id"])
            if not tables:
                raise ValueError("No tables returned from VizieR")
            catalog = None
            if info["table"] is not None:
                for t in tables:
                    if info["table"] in t.meta.get("name", ""):
                        catalog = t
                        break
            if catalog is None:
                catalog = tables[0]
        except Exception as e:
            logger.error(f"Failed to fetch {catalog_key}: {e}")
            return _empty()

        n_rows = len(catalog)
        logger.info(f"Downloaded {n_rows:,} rows, processing...")

        if n_rows > 1_000_000:
            logger.warning(
                f"Catalog '{catalog_key}' has {n_rows:,} rows. "
                "This may require significant memory. "
                "Consider increasing flux_limit to reduce the source count."
            )

        is_sexagesimal = info.get("coords_sexagesimal", False)
        coord_frame    = info.get("coord_frame", "icrs")
        flux_unit      = info.get("flux_unit", "Jy")
        spindex_from_cols = info.get("spindex_from_cols")

        # Auto-detect sexagesimal coordinates: if the first valid RA value is a
        # string that can't be parsed as a float, treat coords as sexagesimal.
        # This handles VizieR returning sexagesimal strings when columns=["**"].
        if not is_sexagesimal and len(catalog) > 0:
            sample_ra = catalog[0][info["ra_col"]]
            if isinstance(sample_ra, (str, np.str_)):
                try:
                    float(sample_ra)
                except ValueError:
                    is_sexagesimal = True
                    logger.debug(
                        f"{catalog_key}: auto-detected sexagesimal coordinates"
                    )

        flux_col = info["flux_col"]
        flux_raw_list = []
        for row in catalog:
            val = row[flux_col]
            if np.ma.is_masked(val) or not np.isfinite(float(val)):
                flux_raw_list.append(np.nan)
            else:
                flux_raw_list.append(float(val))
        flux_raw = np.array(flux_raw_list, dtype=np.float64)
        flux_jy_raw = flux_raw * (1e-3 if flux_unit == "mJy" else 1.0)
        flux_valid = np.isfinite(flux_jy_raw) & (flux_jy_raw >= flux_limit)

        if not np.any(flux_valid):
            logger.info(f"{catalog_key.upper()}: no sources above flux limit {flux_limit} Jy")
            return _empty()

        if is_sexagesimal:
            ra_strs  = [str(row[info["ra_col"]])  for i, row in enumerate(catalog) if flux_valid[i]]
            dec_strs = [str(row[info["dec_col"]]) for i, row in enumerate(catalog) if flux_valid[i]]
            sc = SkyCoord(ra_strs, dec_strs, unit=(u.hourangle, u.deg), frame=coord_frame)
        else:
            ra_list, dec_list = [], []
            coord_ok = np.ones(n_rows, dtype=bool)
            for i, row in enumerate(catalog):
                if not flux_valid[i]:
                    coord_ok[i] = False
                    ra_list.append(0.0)
                    dec_list.append(0.0)
                    continue
                ra_val  = row[info["ra_col"]]
                dec_val = row[info["dec_col"]]
                if np.ma.is_masked(ra_val) or np.ma.is_masked(dec_val):
                    coord_ok[i] = False
                    ra_list.append(0.0)
                    dec_list.append(0.0)
                    continue
                if not (np.isfinite(float(ra_val)) and np.isfinite(float(dec_val))):
                    coord_ok[i] = False
                    ra_list.append(0.0)
                    dec_list.append(0.0)
                    continue
                ra_list.append(float(ra_val))
                dec_list.append(float(dec_val))
            combined_valid = flux_valid & coord_ok
            sc = SkyCoord(
                ra=np.array(ra_list, dtype=np.float64)[combined_valid] * u.deg,
                dec=np.array(dec_list, dtype=np.float64)[combined_valid] * u.deg,
                frame=coord_frame,
            )
            flux_valid = combined_valid  # update mask

        if coord_frame != "icrs":
            sc = sc.icrs

        ra_rad  = sc.ra.rad
        dec_rad = sc.dec.rad
        flux_jy = flux_jy_raw[flux_valid]
        n = len(flux_jy)

        rows_list = list(catalog)
        valid_indices = np.where(flux_valid)[0]
        default_spindex = info["default_spindex"]
        alpha_arr = np.full(n, default_spindex, dtype=np.float64)

        if spindex_from_cols is not None:
            # Two-frequency log-slope (e.g. AT20G)
            s_low_col  = spindex_from_cols["s_low"]
            s_high_col = spindex_from_cols["s_high"]
            freq_low   = spindex_from_cols["freq_low_hz"]
            freq_high  = spindex_from_cols["freq_high_hz"]
            if s_low_col in catalog.colnames and s_high_col in catalog.colnames:
                for j, i in enumerate(valid_indices):
                    row = rows_list[i]
                    sl = row[s_low_col]; sh = row[s_high_col]
                    if (
                        not np.ma.is_masked(sl) and not np.ma.is_masked(sh)
                        and np.isfinite(float(sl)) and np.isfinite(float(sh))
                        and float(sl) > 0 and float(sh) > 0
                    ):
                        alpha_arr[j] = np.log(float(sl) / float(sh)) / np.log(freq_low / freq_high)
        elif info.get("spindex_col") and info["spindex_col"] in catalog.colnames:
            for j, i in enumerate(valid_indices):
                row = rows_list[i]
                val = row[info["spindex_col"]]
                if not np.ma.is_masked(val) and np.isfinite(float(val)):
                    alpha_arr[j] = float(val)

        logger.info(f"{catalog_key.upper()} loaded: {n:,} sources (flux >= {flux_limit} Jy)")

        sky = cls(
            _ra_rad=ra_rad,
            _dec_rad=dec_rad,
            _flux_ref=flux_jy,
            _alpha=alpha_arr,
            _stokes_q=np.zeros(n, dtype=np.float64),
            _stokes_u=np.zeros(n, dtype=np.float64),
            _stokes_v=np.zeros(n, dtype=np.float64),
            _native_format="point_sources",
            model_name=catalog_key,
            brightness_conversion=brightness_conversion,
            _precision=precision,
        )
        sky._ensure_dtypes()
        return sky

    @classmethod
    def from_vlssr(
        cls,
        flux_limit: float = 1.0,
        brightness_conversion: str = "planck",
        precision: Any = None,
    ) -> "SkyModel":
        """
        Load the VLSSr catalog from VizieR (73.8 MHz, ~92,964 sources).

        VLSSr (Cohen et al. 2007) is the redux of the VLA Low-frequency Sky
        Survey at 73.8 MHz. Reference: VIII/97 on VizieR.

        Parameters
        ----------
        flux_limit : float, default=1.0
            Minimum flux density in Jy.
        brightness_conversion : str, default="planck"
            Conversion method: "planck" or "rayleigh-jeans".
        precision : PrecisionConfig, optional
            Precision configuration for array dtypes.

        Returns
        -------
        SkyModel
        """
        return cls._load_from_vizier_catalog("vlssr", flux_limit, brightness_conversion, precision)

    @classmethod
    def from_tgss(
        cls,
        flux_limit: float = 0.1,
        brightness_conversion: str = "planck",
        precision: Any = None,
    ) -> "SkyModel":
        """
        Load the TGSS ADR1 catalog from VizieR (150 MHz, ~623,604 sources).

        TGSS ADR1 (Intema et al. 2017) is the GMRT 150 MHz All-Sky Radio Survey.
        Reference: J/A+A/598/A78 on VizieR. Flux densities are in mJy; this
        method converts to Jy automatically before applying ``flux_limit``.

        Parameters
        ----------
        flux_limit : float, default=0.1
            Minimum flux density in Jy.
        brightness_conversion : str, default="planck"
            Conversion method: "planck" or "rayleigh-jeans".
        precision : PrecisionConfig, optional
            Precision configuration for array dtypes.

        Returns
        -------
        SkyModel
        """
        return cls._load_from_vizier_catalog("tgss", flux_limit, brightness_conversion, precision)

    @classmethod
    def from_wenss(
        cls,
        flux_limit: float = 0.05,
        brightness_conversion: str = "planck",
        precision: Any = None,
    ) -> "SkyModel":
        """
        Load the WENSS catalog from VizieR (325 MHz, ~229,000 sources).

        WENSS (de Bruyn et al. 1998) is the Westerbork Northern Sky Survey
        at 325 MHz. Reference: VIII/62 on VizieR.

        Parameters
        ----------
        flux_limit : float, default=0.05
            Minimum flux density in Jy.
        brightness_conversion : str, default="planck"
            Conversion method: "planck" or "rayleigh-jeans".
        precision : PrecisionConfig, optional
            Precision configuration for array dtypes.

        Returns
        -------
        SkyModel
        """
        return cls._load_from_vizier_catalog("wenss", flux_limit, brightness_conversion, precision)

    @classmethod
    def from_sumss(
        cls,
        flux_limit: float = 0.008,
        brightness_conversion: str = "planck",
        precision: Any = None,
    ) -> "SkyModel":
        """
        Load the SUMSS catalog from VizieR (843 MHz, ~210,412 sources).

        SUMSS (Mauch et al. 2003) is the Sydney University Molonglo Sky
        Survey at 843 MHz. Reference: VIII/81B on VizieR.

        Parameters
        ----------
        flux_limit : float, default=0.008
            Minimum flux density in Jy.
        brightness_conversion : str, default="planck"
            Conversion method: "planck" or "rayleigh-jeans".
        precision : PrecisionConfig, optional
            Precision configuration for array dtypes.

        Returns
        -------
        SkyModel
        """
        return cls._load_from_vizier_catalog("sumss", flux_limit, brightness_conversion, precision)

    @classmethod
    def from_nvss(
        cls,
        flux_limit: float = 0.0025,
        brightness_conversion: str = "planck",
        precision: Any = None,
    ) -> "SkyModel":
        """
        Load the NVSS catalog from VizieR (1400 MHz, ~1.8M sources).

        NVSS (Condon et al. 1998) is the NRAO VLA Sky Survey at 1.4 GHz.
        Reference: VIII/65 on VizieR. Warning: the full catalog has ~1.8M
        rows — consider using a high flux_limit to reduce memory usage.

        Parameters
        ----------
        flux_limit : float, default=0.0025
            Minimum flux density in Jy.
        brightness_conversion : str, default="planck"
            Conversion method: "planck" or "rayleigh-jeans".
        precision : PrecisionConfig, optional
            Precision configuration for array dtypes.

        Returns
        -------
        SkyModel
        """
        return cls._load_from_vizier_catalog("nvss", flux_limit, brightness_conversion, precision)

    @classmethod
    def from_first(
        cls,
        flux_limit: float = 0.001,
        brightness_conversion: str = "planck",
        precision: Any = None,
    ) -> "SkyModel":
        """
        Load the FIRST catalog from VizieR (1400 MHz, ~946,432 sources).

        FIRST (White et al. 1997) is the Faint Images of the Radio Sky at
        Twenty Centimeters survey at 1.4 GHz. Reference: VIII/92 on VizieR.

        Parameters
        ----------
        flux_limit : float, default=0.001
            Minimum flux density in Jy.
        brightness_conversion : str, default="planck"
            Conversion method: "planck" or "rayleigh-jeans".
        precision : PrecisionConfig, optional
            Precision configuration for array dtypes.

        Returns
        -------
        SkyModel
        """
        return cls._load_from_vizier_catalog("first", flux_limit, brightness_conversion, precision)

    @classmethod
    def from_lotss(
        cls,
        release: str = "dr2",
        flux_limit: float = 0.001,
        brightness_conversion: str = "planck",
        precision: Any = None,
    ) -> "SkyModel":
        """
        Load the LoTSS catalog from VizieR (144 MHz, DR1: ~325k, DR2: ~4.4M sources).

        LoTSS (Shimwell et al.) is the LOFAR Two-metre Sky Survey at 144 MHz.
        DR1: J/A+A/622/A1, DR2: J/A+A/659/A1 on VizieR.

        Parameters
        ----------
        release : str, default="dr2"
            Data release: "dr1" or "dr2".
        flux_limit : float, default=0.001
            Minimum flux density in Jy.
        brightness_conversion : str, default="planck"
            Conversion method: "planck" or "rayleigh-jeans".
        precision : PrecisionConfig, optional
            Precision configuration for array dtypes.

        Returns
        -------
        SkyModel

        Raises
        ------
        ValueError
            If ``release`` is not "dr1" or "dr2".
        """
        key = f"lotss_{release.lower()}"
        if key not in VIZIER_POINT_CATALOGS:
            raise ValueError(
                f"Unknown LoTSS release '{release}'. "
                f"Available: 'dr1', 'dr2'."
            )
        return cls._load_from_vizier_catalog(key, flux_limit, brightness_conversion, precision)

    @classmethod
    def from_at20g(
        cls,
        flux_limit: float = 0.04,
        brightness_conversion: str = "planck",
        precision: Any = None,
    ) -> "SkyModel":
        """
        Load the AT20G catalog from VizieR (20 GHz, ~5,890 sources).

        AT20G (Murphy et al. 2010) is the Australia Telescope 20 GHz Survey.
        Spectral indices are computed from multi-frequency flux measurements
        (4.8 and 8.6 GHz) where available. Reference: VIII/83 on VizieR.

        Parameters
        ----------
        flux_limit : float, default=0.04
            Minimum flux density in Jy.
        brightness_conversion : str, default="planck"
            Conversion method: "planck" or "rayleigh-jeans".
        precision : PrecisionConfig, optional
            Precision configuration for array dtypes.

        Returns
        -------
        SkyModel
        """
        return cls._load_from_vizier_catalog("at20g", flux_limit, brightness_conversion, precision)

    @classmethod
    def from_3c(
        cls,
        flux_limit: float = 1.0,
        brightness_conversion: str = "planck",
        precision: Any = None,
    ) -> "SkyModel":
        """
        Load the 3CR catalog from VizieR (178 MHz, ~471 sources).

        3CR (Edge et al. 1959, revised) is the Third Cambridge Catalogue.
        Coordinates are B1950 FK4 and are automatically converted to ICRS.
        Reference: VIII/1 on VizieR.

        Parameters
        ----------
        flux_limit : float, default=1.0
            Minimum flux density in Jy.
        brightness_conversion : str, default="planck"
            Conversion method: "planck" or "rayleigh-jeans".
        precision : PrecisionConfig, optional
            Precision configuration for array dtypes.

        Returns
        -------
        SkyModel
        """
        return cls._load_from_vizier_catalog("3c", flux_limit, brightness_conversion, precision)

    @classmethod
    def from_gb6(
        cls,
        flux_limit: float = 0.018,
        brightness_conversion: str = "planck",
        precision: Any = None,
    ) -> "SkyModel":
        """
        Load the GB6 catalog from VizieR (4850 MHz, ~75,162 sources).

        GB6 (Gregory et al. 1996) is the Green Bank 6 cm Radio Source Catalog
        at 4.85 GHz. Reference: VIII/40 on VizieR.

        Parameters
        ----------
        flux_limit : float, default=0.018
            Minimum flux density in Jy.
        brightness_conversion : str, default="planck"
            Conversion method: "planck" or "rayleigh-jeans".
        precision : PrecisionConfig, optional
            Precision configuration for array dtypes.

        Returns
        -------
        SkyModel
        """
        return cls._load_from_vizier_catalog("gb6", flux_limit, brightness_conversion, precision)

    @classmethod
    def from_racs(
        cls,
        band: str = "low",
        flux_limit: float = 1.0,
        max_rows: int = 1_000_000,
        brightness_conversion: str = "planck",
        precision: Any = None,
    ) -> "SkyModel":
        """
        Load a RACS catalog via CASDA TAP (887.5 / 1367.5 / 1655.5 MHz).

        RACS (McConnell et al. 2020) is the Rapid ASKAP Continuum Survey.
        Data are retrieved via CASDA TAP (astroquery). The column names used
        here are best-effort — verify against the live CASDA schema if errors
        occur.

        Parameters
        ----------
        band : str, default="low"
            Survey band: "low" (887.5 MHz), "mid" (1367.5 MHz), or
            "high" (1655.5 MHz).
        flux_limit : float, default=1.0
            Minimum flux density in Jy. Converted to mJy internally for
            the TAP query.
        max_rows : int, default=1_000_000
            Maximum rows to retrieve (TOP N in ADQL).
        brightness_conversion : str, default="planck"
            Conversion method: "planck" or "rayleigh-jeans".

        Returns
        -------
        SkyModel

        Raises
        ------
        ValueError
            If ``band`` is not "low", "mid", or "high".
        """
        band = band.lower()
        if band not in RACS_CATALOGS:
            raise ValueError(
                f"Unknown RACS band '{band}'. "
                f"Available: {sorted(RACS_CATALOGS.keys())}"
            )

        info = RACS_CATALOGS[band]
        flux_limit_mjy = flux_limit * 1000.0
        model_name = f"racs_{band}"

        logger.info(f"Fetching {info['description']} via CASDA TAP")

        try:
            tap = TapPlus(url=CASDA_TAP_URL)
            adql = (
                f"SELECT TOP {max_rows} "
                f"{info['ra_col']}, {info['dec_col']}, {info['flux_col']} "
                f"FROM {info['tap_table']} "
                f"WHERE {info['flux_col']} >= {flux_limit_mjy}"
            )
            job = tap.launch_job(adql)
            result = job.get_results()

        except Exception as e:
            logger.error(f"Failed to fetch RACS-{band}: {e}")
            return cls._empty_sky(model_name, brightness_conversion, precision)

        freq_hz = info["freq_mhz"] * 1e6

        ra_list, dec_list, flux_list = [], [], []
        for row in result:
            try:
                flux_mjy = row[info["flux_col"]]
                if np.ma.is_masked(flux_mjy) or not np.isfinite(float(flux_mjy)):
                    continue
                flux_jy = float(flux_mjy) * 1e-3
                if flux_jy < flux_limit:
                    continue
                ra_val  = row[info["ra_col"]]
                dec_val = row[info["dec_col"]]
                if np.ma.is_masked(ra_val) or np.ma.is_masked(dec_val):
                    continue
                ra_list.append(float(ra_val))
                dec_list.append(float(dec_val))
                flux_list.append(flux_jy)
            except Exception as e:
                logger.debug(f"Skipping RACS row: {e}")
                continue

        n = len(flux_list)
        logger.info(
            f"RACS-{band.upper()} loaded: {n:,} sources "
            f"(flux >= {flux_limit} Jy, freq={info['freq_mhz']} MHz)"
        )

        if n == 0:
            return cls._empty_sky(model_name, brightness_conversion, precision)

        sky = cls(
            _ra_rad=np.deg2rad(np.array(ra_list, dtype=np.float64)),
            _dec_rad=np.deg2rad(np.array(dec_list, dtype=np.float64)),
            _flux_ref=np.array(flux_list, dtype=np.float64),
            _alpha=np.full(n, -0.7, dtype=np.float64),  # No multi-freq data in RACS
            _stokes_q=np.zeros(n, dtype=np.float64),
            _stokes_u=np.zeros(n, dtype=np.float64),
            _stokes_v=np.zeros(n, dtype=np.float64),
            _native_format="point_sources",
            model_name=model_name,
            frequency=freq_hz,
            brightness_conversion=brightness_conversion,
            _precision=precision,
        )
        sky._ensure_dtypes()
        return sky

    @classmethod
    def from_pyradiosky_file(
        cls,
        filename: str,
        filetype: str | None = None,
        flux_limit: float = 0.0,
        reference_frequency_hz: float | None = None,
        brightness_conversion: str = "planck",
        precision: Any = None,
        frequencies: np.ndarray | None = None,
        obs_frequency_config: dict[str, Any] | None = None,
    ) -> "SkyModel":
        """
        Load a local sky model file via pyradiosky.

        Supports SkyH5, VOTable, text, and FHD formats (as handled by
        pyradiosky). Both ``component_type='point'`` and
        ``component_type='healpix'`` are supported.

        For HEALPix files, observation frequencies can be provided explicitly
        via ``frequencies`` or ``obs_frequency_config``. If the file has
        ``spectral_type='full'`` or ``'subband'`` and no explicit frequencies
        are given, the file's own frequency array is used.

        Parameters
        ----------
        filename : str
            Path to the sky model file.
        filetype : str, optional
            File format: "skyh5", "votable", "text", "fhd", etc.
            If None, pyradiosky infers from the file extension.
        flux_limit : float, default=0.0
            Minimum Stokes I flux in Jy at the reference frequency.
            Only used for point-source files.
        reference_frequency_hz : float, optional
            Reference frequency for Stokes I extraction (Hz).
            If None, uses the first frequency channel in the file.
            Only used for point-source files.
        brightness_conversion : str, default="planck"
            Conversion method: "planck" or "rayleigh-jeans".
        frequencies : np.ndarray, optional
            Array of observation frequencies in Hz for HEALPix files.
            Takes precedence over ``obs_frequency_config``.
        obs_frequency_config : dict, optional
            Frequency configuration dict (keys: starting_frequency,
            frequency_interval, frequency_bandwidth, frequency_unit).
            Used for HEALPix files when ``frequencies`` is None.

        Returns
        -------
        SkyModel

        Raises
        ------
        FileNotFoundError
            If ``filename`` does not exist.
        ValueError
            If the file has an unsupported ``component_type``, or if a
            HEALPix file with ``spectral_type='spectral_index'`` or
            ``'flat'`` is loaded without explicit frequencies.
        """
        if not os.path.exists(filename):
            raise FileNotFoundError(f"Sky model file not found: {filename}")

        sky = PyRadioSkyModel()
        sky.read(filename, filetype=filetype)

        if sky.component_type == "healpix":
            return cls._load_pyradiosky_healpix(
                sky, filename, frequencies, obs_frequency_config,
                brightness_conversion, precision,
            )
        elif sky.component_type != "point":
            raise ValueError(
                f"Unsupported component_type: '{sky.component_type}'. "
                "Only 'point' and 'healpix' are supported."
            )

        ref_freq_hz = reference_frequency_hz
        if ref_freq_hz is None:
            if sky.freq_array is not None and len(sky.freq_array) > 0:
                ref_freq_hz = float(sky.freq_array[0])
            else:
                raise ValueError(
                    "Cannot determine reference frequency. "
                    "Provide reference_frequency_hz explicitly."
                )

        if sky.freq_array is not None and len(sky.freq_array) > 1:
            ref_chan_idx = int(np.argmin(np.abs(np.array(sky.freq_array) - ref_freq_hz)))
        else:
            ref_chan_idx = 0

        # stokes shape: (4, Nfreqs, Ncomponents) or (4, 1, Ncomponents)
        stokes = sky.stokes
        stokes_i_ref = np.array(stokes[0, ref_chan_idx, :], dtype=np.float64)

        n_stokes = stokes.shape[0]
        stokes_q = np.array(stokes[1, ref_chan_idx, :], dtype=np.float64) if n_stokes > 1 else np.zeros_like(stokes_i_ref)
        stokes_u = np.array(stokes[2, ref_chan_idx, :], dtype=np.float64) if n_stokes > 2 else np.zeros_like(stokes_i_ref)
        stokes_v = np.array(stokes[3, ref_chan_idx, :], dtype=np.float64) if n_stokes > 3 else np.zeros_like(stokes_i_ref)

        if sky.spectral_type == "spectral_index":
            spectral_indices = np.asarray(sky.spectral_index, dtype=np.float64)
        elif sky.spectral_type == "flat":
            spectral_indices = np.zeros(sky.Ncomponents, dtype=np.float64)
        else:
            # "full" or "subband": log power-law fit between first and last channel
            if sky.freq_array is not None and len(sky.freq_array) >= 2:
                s_first = np.array(stokes[0, 0, :], dtype=np.float64)
                s_last = np.array(stokes[0, -1, :], dtype=np.float64)
                freq_first = float(sky.freq_array[0])
                freq_last = float(sky.freq_array[-1])
                spectral_indices = np.zeros(sky.Ncomponents, dtype=np.float64)
                valid = (s_first > 0) & (s_last > 0)
                if np.any(valid):
                    spectral_indices[valid] = (
                        np.log(s_first[valid] / s_last[valid])
                        / np.log(freq_first / freq_last)
                    )
            else:
                spectral_indices = np.zeros(sky.Ncomponents, dtype=np.float64)

        ra_arr = np.array(sky.ra.rad if hasattr(sky.ra, "rad") else sky.ra, dtype=np.float64)
        dec_arr = np.array(sky.dec.rad if hasattr(sky.dec, "rad") else sky.dec, dtype=np.float64)

        valid = np.isfinite(stokes_i_ref) & (stokes_i_ref >= flux_limit)
        n = int(valid.sum())

        model_name = f"pyradiosky:{os.path.basename(filename)}"
        logger.info(f"pyradiosky file loaded: {n:,} sources from {filename}")

        if n == 0:
            empty = cls._empty_sky(model_name, brightness_conversion, precision)
            empty.frequency = ref_freq_hz
            return empty

        sky = cls(
            _ra_rad=ra_arr[valid],
            _dec_rad=dec_arr[valid],
            _flux_ref=stokes_i_ref[valid],
            _alpha=spectral_indices[valid],
            _stokes_q=stokes_q[valid],
            _stokes_u=stokes_u[valid],
            _stokes_v=stokes_v[valid],
            _native_format="point_sources",
            model_name=model_name,
            frequency=ref_freq_hz,
            brightness_conversion=brightness_conversion,
            _precision=precision,
        )
        sky._ensure_dtypes()
        return sky

    @classmethod
    def _load_pyradiosky_healpix(
        cls,
        psky: Any,
        filename: str,
        frequencies: np.ndarray | None,
        obs_frequency_config: dict[str, Any] | None,
        brightness_conversion: str,
        precision: Any,
    ) -> "SkyModel":
        """
        Load a pyradiosky HEALPix sky model as multi-frequency HEALPix maps.

        This is called internally by ``from_pyradiosky_file()`` when the file
        has ``component_type='healpix'``.

        Parameters
        ----------
        psky : pyradiosky.SkyModel
            Already-read pyradiosky SkyModel with ``component_type='healpix'``.
        filename : str
            Original file path (for logging and model_name).
        frequencies : np.ndarray or None
            Explicit observation frequencies in Hz.
        obs_frequency_config : dict or None
            Frequency configuration dict.
        brightness_conversion : str
            Conversion method: "planck" or "rayleigh-jeans".
        precision : Any
            Precision configuration.

        Returns
        -------
        SkyModel
            Sky model in healpix_multifreq mode.

        Raises
        ------
        ValueError
            If frequencies cannot be determined.
        """
        # --- Determine observation frequencies ---
        if frequencies is not None:
            obs_freqs = np.asarray(frequencies, dtype=np.float64)
        elif obs_frequency_config is not None:
            obs_freqs = cls._parse_frequency_config(obs_frequency_config)
        elif (
            psky.spectral_type in ("full", "subband")
            and psky.freq_array is not None
            and len(psky.freq_array) > 0
        ):
            obs_freqs = np.asarray(psky.freq_array.to(u.Hz).value, dtype=np.float64)
        else:
            raise ValueError(
                f"Cannot determine observation frequencies for HEALPix file "
                f"with spectral_type='{psky.spectral_type}'. "
                "Provide 'frequencies' or 'obs_frequency_config' explicitly."
            )

        n_freq = len(obs_freqs)
        nside = psky.nside

        logger.info(
            f"Loading pyradiosky HEALPix file: {n_freq} frequencies "
            f"({obs_freqs[0]/1e6:.1f}–{obs_freqs[-1]/1e6:.1f} MHz), "
            f"nside={nside}, from {filename}"
        )

        # --- Evaluate stokes at observation frequencies ---
        psky_eval = psky.at_frequencies(obs_freqs * u.Hz, inplace=False)

        # --- Convert units to Kelvin if needed ---
        if psky_eval.stokes.unit.is_equivalent(u.Jy / u.sr):
            psky_eval.jansky_to_kelvin()

        # --- Determine coordinate frame and pixel handling ---
        is_galactic = False
        if hasattr(psky_eval, "frame") and psky_eval.frame is not None:
            frame_name = str(psky_eval.frame).lower()
            if "galactic" in frame_name:
                is_galactic = True

        rot = Rotator(coord=["G", "C"]) if is_galactic else None

        # Check for nested ordering
        is_nested = False
        if hasattr(psky_eval, "hpx_order") and psky_eval.hpx_order is not None:
            if psky_eval.hpx_order.lower() == "nested":
                is_nested = True

        # Check for sparse map (partial sky)
        hpx_inds = None
        if hasattr(psky_eval, "hpx_inds") and psky_eval.hpx_inds is not None:
            hpx_inds = np.asarray(psky_eval.hpx_inds)

        npix = hp.nside2npix(nside)

        # --- Extract Stokes I and build full-sky maps ---
        # psky_eval.stokes shape: (4, Nfreqs, Ncomponents)
        stokes_data = np.asarray(psky_eval.stokes.value)

        healpix_maps: dict[float, np.ndarray] = {}
        for i_freq in range(n_freq):
            stokes_i = stokes_data[0, i_freq, :]

            if hpx_inds is not None:
                # Sparse map: fill only specified pixels
                full_map = np.zeros(npix, dtype=np.float64)
                pixel_indices = hpx_inds
                if is_nested:
                    pixel_indices = hp.nest2ring(nside, pixel_indices)
                full_map[pixel_indices] = stokes_i
            else:
                # Full-sky map
                full_map = np.array(stokes_i, dtype=np.float64)
                if is_nested:
                    full_map = hp.reorder(full_map, n2r=True)

            if rot is not None:
                full_map = rot.rotate_map_pixel(full_map)

            healpix_maps[float(obs_freqs[i_freq])] = full_map.astype(np.float32)

        model_name = f"pyradiosky:{os.path.basename(filename)}"
        logger.info(
            f"pyradiosky HEALPix loaded: {npix} pixels × {n_freq} frequencies"
        )

        sky_model = cls(
            _healpix_maps=healpix_maps,
            _healpix_nside=nside,
            _observation_frequencies=obs_freqs,
            _native_format="healpix",
            frequency=float(obs_freqs[0]),
            model_name=model_name,
            brightness_conversion=brightness_conversion,
            _precision=precision,
        )
        sky_model._ensure_dtypes()
        return sky_model

    @classmethod
    def from_pysm3(
        cls,
        components: str | list[str] = "s1",
        nside: int = 64,
        frequencies: np.ndarray | None = None,
        obs_frequency_config: dict[str, Any] | None = None,
        brightness_conversion: str = "planck",
        precision: Any = None,
    ) -> "SkyModel":
        """
        Load a PySM3 diffuse sky model as multi-frequency HEALPix maps.

        Generates one brightness temperature map per observation frequency
        using PySM3's native per-channel computation. Maps are rotated from
        Galactic to Equatorial (ICRS) coordinates and stored as float32.

        Parameters
        ----------
        components : str or list of str, default="s1"
            PySM3 preset string(s) (e.g. "s1", "d1", ["s1", "d1", "f1"]).
            See PySM3 documentation for available presets.
        nside : int, default=64
            HEALPix NSIDE resolution.
        frequencies : np.ndarray, optional
            Array of observation frequencies in Hz. Takes precedence over
            ``obs_frequency_config`` when both are provided.
        obs_frequency_config : dict, optional
            Frequency configuration dict (keys: starting_frequency,
            frequency_interval, frequency_bandwidth, frequency_unit).
        brightness_conversion : str, default="planck"
            Conversion method for T_b → Jy: "planck" or "rayleigh-jeans".

        Returns
        -------
        SkyModel
            Sky model in healpix_multifreq mode.

        Raises
        ------
        ValueError
            If neither ``frequencies`` nor ``obs_frequency_config`` is provided.
        """
        if frequencies is None and obs_frequency_config is None:
            raise ValueError(
                "Either 'frequencies' or 'obs_frequency_config' must be provided. "
                "Example: from_pysm3(components='s1', nside=64, "
                "frequencies=np.linspace(100e6, 120e6, 20))"
            )

        if frequencies is None:
            frequencies = cls._parse_frequency_config(obs_frequency_config)
        frequencies = np.asarray(frequencies, dtype=np.float64)

        components_list = [components] if isinstance(components, str) else list(components)
        n_freq = len(frequencies)

        logger.info(
            f"Loading PySM3 components {components_list}: {n_freq} frequencies "
            f"({frequencies[0]/1e6:.1f}–{frequencies[-1]/1e6:.1f} MHz), nside={nside}"
        )

        sky = pysm3.Sky(nside=nside, preset_strings=components_list)
        rot = Rotator(coord=["G", "C"])

        healpix_maps: dict[float, np.ndarray] = {}
        for freq in frequencies:
            emission = sky.get_emission(freq * pysm3_u.Hz)
            emission_krj = emission.to(
                pysm3_u.K_RJ,
                equivalencies=pysm3_u.cmb_equivalencies(freq * pysm3_u.Hz),
            )
            temp_map = np.array(emission_krj[0])  # Stokes I

            current_nside = hp.get_nside(temp_map)
            if current_nside != nside:
                temp_map = hp.ud_grade(temp_map, nside_out=nside)

            temp_map = rot.rotate_map_pixel(temp_map)
            healpix_maps[float(freq)] = temp_map.astype(np.float32)

        model_name = f"pysm3:{'+'.join(components_list)}"
        logger.info(
            f"PySM3 {components_list} loaded: {hp.nside2npix(nside)} pixels × {n_freq} frequencies"
        )

        sky = cls(
            _healpix_maps=healpix_maps,
            _healpix_nside=nside,
            _observation_frequencies=frequencies,
            _native_format="healpix",
            frequency=float(frequencies[0]),
            model_name=model_name,
            brightness_conversion=brightness_conversion,
            _precision=precision,
        )
        sky._ensure_dtypes()
        return sky

    @classmethod
    def from_ulsa(
        cls,
        nside: int = 64,
        frequencies: np.ndarray | None = None,
        obs_frequency_config: dict[str, Any] | None = None,
        brightness_conversion: str = "planck",
        precision: Any = None,
    ) -> "SkyModel":
        """
        Load the ULSA ultra-low-frequency sky model as multi-frequency HEALPix maps.

        ULSA (Cong et al.) provides global sky brightness temperature maps
        below ~100 MHz. Maps are rotated from Galactic to Equatorial coordinates.

        Parameters
        ----------
        nside : int, default=64
            HEALPix NSIDE resolution.
        frequencies : np.ndarray, optional
            Array of observation frequencies in Hz. Takes precedence over
            ``obs_frequency_config`` when both are provided.
        obs_frequency_config : dict, optional
            Frequency configuration dict (keys: starting_frequency,
            frequency_interval, frequency_bandwidth, frequency_unit).
        brightness_conversion : str, default="planck"
            Conversion method for T_b → Jy: "planck" or "rayleigh-jeans".

        Returns
        -------
        SkyModel
            Sky model in healpix_multifreq mode.

        Raises
        ------
        ImportError
            If ``ULSA`` is not installed.
        ValueError
            If neither ``frequencies`` nor ``obs_frequency_config`` is provided.
        """
        try:
            import ULSA as ulsa
        except ImportError as err:
            raise ImportError(
                "ULSA is required for from_ulsa(). "
                "Install it with: "
                "pip install git+https://github.com/Yanping-Cong/ULSA.git"
            ) from err

        if frequencies is None and obs_frequency_config is None:
            raise ValueError(
                "Either 'frequencies' or 'obs_frequency_config' must be provided. "
                "Example: from_ulsa(nside=64, frequencies=np.linspace(1e6, 100e6, 20))"
            )

        if frequencies is None:
            frequencies = cls._parse_frequency_config(obs_frequency_config)
        frequencies = np.asarray(frequencies, dtype=np.float64)

        n_freq = len(frequencies)
        logger.info(
            f"Loading ULSA: {n_freq} frequencies "
            f"({frequencies[0]/1e6:.3f}–{frequencies[-1]/1e6:.3f} MHz), nside={nside}"
        )

        rot = Rotator(coord=["G", "C"])
        healpix_maps: dict[float, np.ndarray] = {}

        for freq in frequencies:
            freq_mhz = freq / 1e6
            try:
                # Try modern API: ulsa.generate(freq_mhz, nside=nside)
                temp_map = ulsa.generate(freq_mhz, nside=nside)
            except (AttributeError, TypeError):
                try:
                    # Fallback to older API: ulsa.Sky(nside).generate(freq_mhz)
                    temp_map = ulsa.Sky(nside).generate(freq_mhz)
                except Exception as e:
                    logger.error(f"ULSA generation failed at {freq_mhz:.3f} MHz: {e}")
                    npix = hp.nside2npix(nside)
                    temp_map = np.zeros(npix, dtype=np.float32)

            if len(temp_map) != hp.nside2npix(nside):
                temp_map = hp.ud_grade(temp_map, nside_out=nside)

            temp_map = rot.rotate_map_pixel(temp_map)
            healpix_maps[float(freq)] = temp_map.astype(np.float32)

        logger.info(
            f"ULSA loaded: {hp.nside2npix(nside)} pixels × {n_freq} frequencies"
        )

        sky = cls(
            _healpix_maps=healpix_maps,
            _healpix_nside=nside,
            _observation_frequencies=frequencies,
            _native_format="healpix",
            frequency=float(frequencies[0]),
            model_name="ulsa",
            brightness_conversion=brightness_conversion,
            _precision=precision,
        )
        sky._ensure_dtypes()
        return sky

    @classmethod
    def from_point_sources(
        cls,
        sources: list[dict[str, Any]],
        model_name: str = "custom",
        brightness_conversion: str = "planck",
        precision: Any = None,
    ) -> "SkyModel":
        """
        Create SkyModel from existing point source list.

        Parameters
        ----------
        sources : list of dict
            Point source list. Each dict must have 'coords' (SkyCoord) and 'flux' (float).
            Optional: 'spectral_index', 'stokes_q', 'stokes_u', 'stokes_v'.
        model_name : str, default="custom"
            Name for the model.
        brightness_conversion : str, default="planck"
            Conversion method: "planck" or "rayleigh-jeans".
        precision : PrecisionConfig, optional
            Precision configuration for array dtypes.

        Returns
        -------
        SkyModel
        """
        n = len(sources)
        if n == 0:
            return cls(
                _ra_rad=np.zeros(0, dtype=np.float64),
                _dec_rad=np.zeros(0, dtype=np.float64),
                _flux_ref=np.zeros(0, dtype=np.float64),
                _alpha=np.zeros(0, dtype=np.float64),
                _stokes_q=np.zeros(0, dtype=np.float64),
                _stokes_u=np.zeros(0, dtype=np.float64),
                _stokes_v=np.zeros(0, dtype=np.float64),
                _native_format="point_sources",
                model_name=model_name,
                brightness_conversion=brightness_conversion,
                _precision=precision,
            )

        ra_rad  = np.array([s["coords"].ra.rad  for s in sources], dtype=np.float64)
        dec_rad = np.array([s["coords"].dec.rad for s in sources], dtype=np.float64)
        flux_ref = np.array([s["flux"]                         for s in sources], dtype=np.float64)
        alpha    = np.array([s.get("spectral_index", -0.7)     for s in sources], dtype=np.float64)
        stokes_q = np.array([s.get("stokes_q", 0.0)            for s in sources], dtype=np.float64)
        stokes_u = np.array([s.get("stokes_u", 0.0)            for s in sources], dtype=np.float64)
        stokes_v = np.array([s.get("stokes_v", 0.0)            for s in sources], dtype=np.float64)

        sky = cls(
            _ra_rad=ra_rad,
            _dec_rad=dec_rad,
            _flux_ref=flux_ref,
            _alpha=alpha,
            _stokes_q=stokes_q,
            _stokes_u=stokes_u,
            _stokes_v=stokes_v,
            _native_format="point_sources",
            model_name=model_name,
            brightness_conversion=brightness_conversion,
            _precision=precision,
        )
        sky._ensure_dtypes()
        return sky

    # =========================================================================
    # Combination
    # =========================================================================

    @classmethod
    def combine(
        cls,
        models: list["SkyModel"],
        representation: str = "point_sources",
        nside: int = 64,
        frequency: float | None = None,
        obs_frequency_config: dict[str, Any] | None = None,
        ref_frequency: float = 76e6,
        brightness_conversion: str = "planck",
        precision: Any = None,
    ) -> "SkyModel":
        """
        Combine multiple sky models into one.

        The combination always works by first concatenating all models as point
        sources, preserving each source's individual properties (flux, spectral
        index, coordinates). If healpix_map representation is requested, the
        concatenated sources are then converted to multi-frequency HEALPix maps.

        Parameters
        ----------
        models : list of SkyModel
            Sky models to combine.
        representation : str, default="point_sources"
            Output representation: "point_sources" or "healpix_map".
        nside : int, default=64
            HEALPix NSIDE for healpix_map output.
        frequency : float, optional
            Frequency for HEALPix-to-point-source conversions.
        obs_frequency_config : dict, optional
            Required for healpix_map representation. Observation frequency
            configuration with keys: starting_frequency, frequency_interval,
            frequency_bandwidth, frequency_unit.
        ref_frequency : float, default=76e6
            Reference frequency for spectral extrapolation (Hz).

        Returns
        -------
        SkyModel
            Combined sky model.

        Raises
        ------
        ValueError
            If healpix_map representation is requested without obs_frequency_config.

        Warns
        -----
        UserWarning
            When combining catalog sources (GLEAM, MALS) with diffuse models (GSM),
            as this can result in double-counting of bright sources.

        Notes
        -----
        **How concatenation works**:

        Each model is converted to point sources, then all sources are collected
        into a single list. Each source retains its individual properties:

        - coords: SkyCoord (RA, Dec position)
        - flux: float (Jy at reference frequency)
        - spectral_index: float (power-law exponent)
        - stokes_q, stokes_u, stokes_v: float (polarization)

        No averaging occurs during concatenation - if two sources happen to be
        at similar positions, they remain as separate sources. This preserves
        the correct spectral behavior when later converting to HEALPix maps.

        **Double-counting warning**: Diffuse sky models (GSM, LFSM, Haslam) include
        integrated emission from all sources, including bright ones that also appear
        in catalogs like GLEAM. Naive combination will double-count these sources.

        Examples
        --------
        >>> # Combine as point sources (default)
        >>> gleam = SkyModel.from_gleam(flux_limit=1.0)
        >>> test = SkyModel.from_test_sources(num_sources=10)
        >>> combined = SkyModel.combine([gleam, test])

        >>> # Combine and convert to multi-frequency HEALPix
        >>> obs_config = {
        ...     "starting_frequency": 100.0,
        ...     "frequency_interval": 1.0,
        ...     "frequency_bandwidth": 20.0,
        ...     "frequency_unit": "MHz"
        ... }
        >>> combined = SkyModel.combine(
        ...     [gleam, test],
        ...     representation="healpix_map",
        ...     nside=64,
        ...     obs_frequency_config=obs_config
        ... )
        """
        if not models:
            return cls(
                _ra_rad=np.zeros(0, dtype=np.float64),
                _dec_rad=np.zeros(0, dtype=np.float64),
                _flux_ref=np.zeros(0, dtype=np.float64),
                _alpha=np.zeros(0, dtype=np.float64),
                _stokes_q=np.zeros(0, dtype=np.float64),
                _stokes_u=np.zeros(0, dtype=np.float64),
                _stokes_v=np.zeros(0, dtype=np.float64),
                model_name="combined_empty",
                brightness_conversion=brightness_conversion,
                _precision=precision,
            )

        has_catalog = any(m.mode == "point_sources" and m._has_point_sources()
                         for m in models)
        has_diffuse = any(m.mode == "healpix_multifreq" for m in models)

        if has_catalog and has_diffuse:
            warnings.warn(
                "Combining catalog sources (GLEAM/MALS) with diffuse models (GSM/LFSM/Haslam) "
                "may result in double-counting of bright sources. Diffuse models already include "
                "integrated emission from bright sources. Consider using only one model type "
                "or implementing source subtraction for accurate results.",
                UserWarning,
                stacklevel=2
            )

        freq = frequency
        if freq is None:
            for m in models:
                if m.frequency is not None:
                    freq = m.frequency
                    break

        has_healpix_multifreq = any(m.mode == "healpix_multifreq" for m in models)

        # ==================================================================
        # HEALPIX_MAP REPRESENTATION WITH MULTIFREQ MODELS
        # Combine maps element-wise in Jy space (not T_b space), per
        # frequency. T_b addition is nonlinear under Planck:
        # T_b(I1+I2) != T_b(I1) + T_b(I2)
        # ==================================================================
        if representation == "healpix_map" and has_healpix_multifreq:
            ref_model = next(m for m in models if m.mode == "healpix_multifreq")
            _, ref_nside, ref_freqs = ref_model.get_multifreq_maps()

            # Validate all healpix_multifreq models share the same nside and
            # frequency grid before doing element-wise arithmetic.  Mismatched
            # nside causes shape errors; mismatched frequency grids cause silent
            # scientific corruption via nearest-frequency fallback.
            for m in models:
                if m.mode != "healpix_multifreq":
                    continue
                _, m_nside, m_freqs = m.get_multifreq_maps()
                if m_nside != ref_nside:
                    raise ValueError(
                        f"Cannot combine HEALPix models with different nside values: "
                        f"reference has nside={ref_nside}, model '{m.model_name}' has "
                        f"nside={m_nside}. Resample to a common nside with hp.ud_grade() "
                        f"before combining."
                    )
                if not np.array_equal(m_freqs, ref_freqs):
                    raise ValueError(
                        f"Cannot combine HEALPix models with different frequency grids: "
                        f"reference has {len(ref_freqs)} channels "
                        f"({ref_freqs[0]/1e6:.3f}–{ref_freqs[-1]/1e6:.3f} MHz), "
                        f"model '{m.model_name}' has {len(m_freqs)} channels "
                        f"({m_freqs[0]/1e6:.3f}–{m_freqs[-1]/1e6:.3f} MHz). "
                        f"All models must share the same observation frequency grid."
                    )

            npix = hp.nside2npix(ref_nside)
            omega_pixel = 4 * np.pi / npix

            ps_models_data = []
            for m in models:
                if m.mode == "point_sources" and m._has_point_sources():
                    ipix_m = hp.ang2pix(ref_nside, np.pi / 2 - m._dec_rad, m._ra_rad)
                    ps_models_data.append((ipix_m, m._flux_ref, m._alpha))

            combined_maps: dict[float, np.ndarray] = {}
            for freq_hz in ref_freqs:
                combined_flux = np.zeros(npix, dtype=np.float64)

                for m in models:
                    if m.mode == "healpix_multifreq":
                        t_map = m.get_map_at_frequency(freq_hz).astype(np.float64)
                        pos = t_map > 0
                        if np.any(pos):
                            combined_flux[pos] += brightness_temp_to_flux_density(
                                t_map[pos], freq_hz, omega_pixel,
                                method=brightness_conversion,
                            )

                for ipix_m, flux_ref_m, alpha_m in ps_models_data:
                    scale = (float(freq_hz) / ref_frequency) ** alpha_m
                    flux_at_f = flux_ref_m * scale
                    combined_flux += np.bincount(ipix_m, weights=flux_at_f, minlength=npix)

                combined_T_b = np.zeros(npix, dtype=np.float64)
                pos_flux = combined_flux > 0
                if np.any(pos_flux):
                    combined_T_b[pos_flux] = flux_density_to_brightness_temp(
                        combined_flux[pos_flux], freq_hz, omega_pixel,
                        method=brightness_conversion,
                    )
                combined_maps[freq_hz] = combined_T_b.astype(np.float32)

            logger.info(
                f"Combined {len(models)} models into healpix_multifreq "
                f"({len(ref_freqs)} channels, nside={ref_nside})"
            )

            sky = cls(
                _healpix_maps=combined_maps,
                _healpix_nside=ref_nside,
                _observation_frequencies=ref_freqs,
                _native_format="healpix",
                frequency=float(ref_freqs[0]) if len(ref_freqs) > 0 else freq,
                model_name="combined",
                brightness_conversion=brightness_conversion,
                _precision=precision,
            )
            sky._ensure_dtypes()
            return sky

        # ==================================================================
        # ALL OTHER CASES: Concatenate as point sources first
        # ==================================================================
        all_sources = []
        for m in models:
            sources = m.to_point_sources(frequency=freq)
            all_sources.extend(sources)

        logger.info(f"Combined {len(models)} models: {len(all_sources)} total sources")

        combined = cls.from_point_sources(
            all_sources,
            model_name="combined",
            brightness_conversion=brightness_conversion,
            precision=precision,
        )
        combined.frequency = freq

        if representation == "healpix_map":
            if obs_frequency_config is None:
                raise ValueError(
                    "obs_frequency_config is required for healpix_map representation. "
                    "Provide a dict with: starting_frequency, frequency_interval, "
                    "frequency_bandwidth, frequency_unit."
                )

            combined.to_healpix_for_observation(
                nside=nside,
                obs_frequency_config=obs_frequency_config,
                ref_frequency=ref_frequency,
            )

        return combined

    # =========================================================================
    # String Representation
    # =========================================================================

    def __repr__(self) -> str:
        """Return a human-readable summary of the sky model.

        Returns
        -------
        str
            Summary string including native format, model name, mode,
            and either source count or HEALPix parameters.
        """
        if self._healpix_maps is not None:
            freqs = self._observation_frequencies
            freq_range = (
                f"{freqs[0]/1e6:.1f}-{freqs[-1]/1e6:.1f}"
                if len(freqs) > 1 else f"{freqs[0]/1e6:.1f}"
            )
            mem_info = self.estimate_healpix_memory(self._healpix_nside, len(freqs), np.float32)
            return (
                f"SkyModel(native='{self._native_format}', model='{self.model_name}', "
                f"mode='healpix_multifreq', nside={self._healpix_nside}, "
                f"n_freq={len(freqs)}, freq_range={freq_range}MHz, "
                f"memory={mem_info['total_mb']:.1f}MB)"
            )
        return (
            f"SkyModel(native='{self._native_format}', model='{self.model_name}', "
            f"n_sources={self.n_sources})"
        )
