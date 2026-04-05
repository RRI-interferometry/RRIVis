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
        "description": (
            "VLSSr: VLA Low-Frequency Sky Survey Redux (Lane, Cotton, van Velzen, "
            "Taylor, Perley, Kassim, Condon, MNRAS 440, 327, 2014). "
            "https://ui.adsabs.harvard.edu/abs/2014MNRAS.440..327L/abstract "
            "A 73.8 MHz continuum survey using the VLA in B and BnA configurations with "
            "1.56 MHz bandwidth, observed 2001-2007. The redux reprocessed the "
            "original VLSS data (Cohen et al. 2007, AJ 134, 1245) with improved "
            "ionospheric calibration and wide-field imaging. Resolution 75 arcsec, "
            "median rms noise ~0.1 Jy/beam. Sky coverage: 9.3 sr (the entire sky "
            "above an irregular southern boundary near declination -30 deg to -40 deg, "
            "depending on hour angle). Catalog contains 92,964 sources. Peak flux "
            "density (Sp) in Jy/beam; no integrated flux reported. Clean bias is "
            "0.66 * sigma. Via VizieR VIII/97."
        ),
        "ra_col": "RAJ2000",
        "dec_col": "DEJ2000",
        "flux_col": "Sp",
        "flux_unit": "Jy/beam",
        "spindex_col": None,
        "default_spindex": -0.7,
        "freq_mhz": 73.8,
        "coords_sexagesimal": True,
        "coord_frame": "icrs",
        "major_col": None,
        "minor_col": None,
        "pa_col": None,
    },
    "tgss": {
        "vizier_id": "J/A+A/598/A78",
        "table": None,
        "description": (
            "TGSS ADR1: TIFR GMRT Sky Survey Alternative Data Release 1 "
            "(Intema, Jagannathan, Mooley, Frail, A&A 598, A78, 2017). "
            "https://ui.adsabs.harvard.edu/abs/2017A&A...598A..78I/abstract "
            "A 150 MHz continuum survey using the Giant Metrewave Radio Telescope "
            "(GMRT, Khodad, Maharashtra, India), observed April 2010 to March 2012 "
            "over 2000+ hours. The ADR1 pipeline applied direction-dependent "
            "ionospheric phase calibration, antenna pointing corrections, and "
            "improved primary beam models. Resolution ~25 arcsec, median rms noise "
            "3.5 mJy/beam. Sky coverage: 36,900 deg^2 (90% of sky) between "
            "declination -53 deg and +90 deg. Catalog contains 623,604 sources "
            "above a 7-sigma peak-to-noise threshold. Stotal is the total "
            "integrated flux density in mJy. Via VizieR J/A+A/598/A78."
        ),
        "ra_col": "RAJ2000",
        "dec_col": "DEJ2000",
        "flux_col": "Stotal",
        "flux_unit": "mJy",
        "spindex_col": None,
        "default_spindex": -0.7,
        "freq_mhz": 150.0,
        "coords_sexagesimal": False,
        "coord_frame": "icrs",
        "major_col": "Maj",
        "minor_col": "Min",
        "pa_col": "PA",
    },
    "wenss": {
        "vizier_id": "VIII/62",
        "table": None,
        "description": (
            "WENSS: Westerbork Northern Sky Survey (Rengelink, Tang, de Bruyn, "
            "Miley, Bremer, Roettgering, Bremer, A&AS 124, 259, 1997; "
            "full survey: de Bruyn et al., in prep). "
            "https://ui.adsabs.harvard.edu/abs/1997A&AS..124..259R/abstract "
            "A 325 MHz (92 cm) continuum survey using the Westerbork Synthesis "
            "Radio Telescope (WSRT, Westerbork, Netherlands). Resolution "
            "54 arcsec x 54 arcsec * cosec(delta), positional accuracy ~1.5 arcsec "
            "for strong sources. Limiting flux density ~18 mJy (5-sigma). "
            "Sky coverage: the entire sky north of declination +28 deg "
            "(3.14 sr, ~10,300 deg^2). The main catalog (declination +28 deg to "
            "+76 deg) contains ~211,234 sources and the polar catalog (above "
            "+72 deg) contains ~18,186 sources, totalling ~229,420 sources. "
            "Sint is the integrated flux density in mJy. A collaboration between "
            "ASTRON and Leiden Observatory. Via VizieR VIII/62."
        ),
        "ra_col": "RAJ2000",
        "dec_col": "DEJ2000",
        "flux_col": "Sint",
        "flux_unit": "mJy",
        "spindex_col": None,
        "default_spindex": -0.7,
        "freq_mhz": 325.0,
        "coords_sexagesimal": True,
        "coord_frame": "icrs",
        "major_col": None,
        "minor_col": None,
        "pa_col": None,
    },
    "sumss": {
        "vizier_id": "VIII/81B",
        "table": None,
        "description": (
            "SUMSS: Sydney University Molonglo Sky Survey (Bock, Large, Sadler, "
            "AJ 117, 1578, 1999 [survey design]; Mauch, Murphy, Buttery, Curran, "
            "Hunstead, Piestrzynski, Robertson, Sadler, MNRAS 342, 1117, 2003 "
            "[source catalog]). "
            "https://ui.adsabs.harvard.edu/abs/2003MNRAS.342.1117M/abstract "
            "An 843 MHz continuum survey using the Molonglo Observatory Synthesis "
            "Telescope (MOST, near Canberra, Australia) in its upgraded wide-field "
            "configuration. Resolution 45 arcsec x 45 arcsec * cosec(delta), "
            "rms noise ~1 mJy/beam. Mosaic images are 4.3 deg x 4.3 deg. "
            "Sky coverage: the southern sky from declination -30 deg southward, "
            "excluding |b| < 10 deg (~8,000 deg^2). Designed to complement the "
            "NVSS with matched resolution and sensitivity. Catalog version 2.1 "
            "contains ~211,000 sources (v2.1) from 633 mosaics. St is the total "
            "integrated flux density in mJy. Via VizieR VIII/81B."
        ),
        "ra_col": "RAJ2000",
        "dec_col": "DEJ2000",
        "flux_col": "St",
        "flux_unit": "mJy",
        "spindex_col": None,
        "default_spindex": -0.7,
        "freq_mhz": 843.0,
        "coords_sexagesimal": False,
        "coord_frame": "icrs",
        "major_col": "dMajAxis",
        "minor_col": "dMinAxis",
        "pa_col": "dPA",
    },
    "nvss": {
        "vizier_id": "VIII/65",
        "table": None,
        "description": (
            "NVSS: NRAO VLA Sky Survey (Condon, Cotton, Greisen, Yin, Perley, "
            "Taylor, Broderick, AJ 115, 1693, 1998). "
            "https://ui.adsabs.harvard.edu/abs/1998AJ....115.1693C/abstract "
            "A 1.4 GHz continuum survey using the VLA in D and DnC configurations "
            "(Socorro, New Mexico, USA). Resolution 45 arcsec FWHM, rms noise "
            "~0.45 mJy/beam (Stokes I), ~0.29 mJy/beam (Stokes Q and U). "
            "Sky coverage: the entire sky north of declination -40 deg (82% of "
            "the celestial sphere, 33,885 deg^2). The survey produced 2326 "
            "4 deg x 4 deg continuum cubes (Stokes I, Q, U) and a catalog of "
            "~1.8 million discrete sources. 99% complete at 3.5 mJy integrated "
            "flux density. Positional accuracy <1 arcsec for sources >15 mJy, "
            "~7 arcsec at the survey limit. S1.4 is the integrated flux density "
            "at 1.4 GHz in mJy. Via VizieR VIII/65."
        ),
        "ra_col": "RAJ2000",
        "dec_col": "DEJ2000",
        "flux_col": "S1.4",
        "flux_unit": "mJy",
        "spindex_col": None,
        "default_spindex": -0.7,
        "freq_mhz": 1400.0,
        "coords_sexagesimal": False,
        "coord_frame": "icrs",
        "major_col": "MajAxis",
        "minor_col": "MinAxis",
        "pa_col": "PA",
    },
    "vlass": {
        "vizier_id": "J/ApJS/255/30",
        "table": "comp",
        "description": (
            "VLASS QL Ep.1: VLA Sky Survey Quick Look Epoch 1 Component Catalog "
            "(Gordon, Boyce, O'Dea, Rudnick, Andernach, et al., "
            "ApJS 255, 30, 2021). "
            "https://ui.adsabs.harvard.edu/abs/2021ApJS..255...30G/abstract "
            "The VLA Sky Survey (Lacy et al. 2020, PASP 132, 035001) observes "
            "the sky north of declination -40 deg in S-band (2-4 GHz, centre "
            "~3 GHz) using the VLA in BnA and B configurations. Angular resolution "
            "~2.5 arcsec — the highest of any all-sky radio continuum survey. "
            "The Quick Look Epoch 1 catalog contains ~1.9 million reliably detected "
            "radio components from the CIRADA pipeline. Caution: Quick Look flux "
            "densities are systematically underestimated by ~15% at peak flux "
            ">3 mJy/beam and are often unreliable for fainter components. "
            "Ftot is the total integrated flux density in mJy. "
            "Via VizieR J/ApJS/255/30."
        ),
        "ra_col": "RAJ2000",
        "dec_col": "DEJ2000",
        "flux_col": "Ftot",
        "flux_unit": "mJy",
        "spindex_col": None,
        "default_spindex": -0.7,
        "freq_mhz": 3000.0,
        "coords_sexagesimal": False,
        "coord_frame": "icrs",
        "major_col": "DCMaj",
        "minor_col": "DCMin",
        "pa_col": "DCPA",
    },
    "lotss_dr1": {
        "vizier_id": "J/A+A/622/A1",
        "table": None,
        "description": (
            "LoTSS DR1: LOFAR Two-metre Sky Survey First Data Release "
            "(Shimwell, Tasse, Hardcastle, Mechev, Williams, Best, Rottgering, "
            "et al., A&A 622, A1, 2019). "
            "https://ui.adsabs.harvard.edu/abs/2019A&A...622A...1S/abstract "
            "A sensitive 120-168 MHz survey using LOFAR (LOw-Frequency ARray, "
            "core in Exloo, Netherlands) High Band Antennas. Resolution 6 arcsec, "
            "median rms noise 71 uJy/beam, positional accuracy ~0.2 arcsec. "
            "DR1 covers 424 deg^2 in the HETDEX Spring Field "
            "(RA 10h45m-15h30m, Dec +45 deg to +57 deg). Catalog contains "
            "325,694 sources detected at >=5-sigma. Sint is the total integrated "
            "flux density in mJy. Via VizieR J/A+A/622/A1."
        ),
        "ra_col": "RAJ2000",
        "dec_col": "DEJ2000",
        "flux_col": "Sint",
        "flux_unit": "mJy",
        "spindex_col": None,
        "default_spindex": -0.7,
        "freq_mhz": 144.0,
        "coords_sexagesimal": False,
        "coord_frame": "icrs",
        "major_col": None,
        "minor_col": None,
        "pa_col": None,
    },
    "lotss_dr2": {
        "vizier_id": "J/A+A/659/A1",
        "table": None,
        "description": (
            "LoTSS DR2: LOFAR Two-metre Sky Survey Second Data Release "
            "(Shimwell, Hardcastle, Tasse, Best, Rottgering, Williams, Botteon, "
            "et al., A&A 659, A1, 2022). "
            "https://ui.adsabs.harvard.edu/abs/2022A&A...659A...1S/abstract "
            "Covers 27% of the northern sky (5,634 deg^2) in two regions "
            "centred near 12h45m +44d30m and 1h00m +28d00m spanning 4,178 and "
            "1,457 deg^2 respectively. 120-168 MHz, resolution 6 arcsec, "
            "median rms noise 83 uJy/beam. Catalog contains 4,396,228 radio "
            "components. SpeakTot is the total integrated flux density of the source "
            "in mJy/beam. Via VizieR J/A+A/659/A1."
        ),
        "ra_col": "RAJ2000",
        "dec_col": "DEJ2000",
        "flux_col": "SpeakTot",
        "flux_unit": "mJy",
        "spindex_col": None,
        "default_spindex": -0.7,
        "freq_mhz": 144.0,
        "coords_sexagesimal": False,
        "coord_frame": "icrs",
        "major_col": None,
        "minor_col": None,
        "pa_col": None,
    },
    "3c": {
        "vizier_id": "VIII/1",
        "table": "3cr",
        "description": (
            "3CR: Revised Third Cambridge Catalogue of Radio Sources "
            "(Edge, Shakeshaft, McAdam, Baldwin, Archer, MmRAS 68, 37, 1959 "
            "[original 3C]; Bennett, MmRAS 68, 163, 1962 [3CR revision]). "
            "https://ui.adsabs.harvard.edu/abs/1962MmRAS..68..163B/abstract "
            "The foundational radio source catalog at 178 MHz from the Cambridge "
            "radio interferometer. The 3CR revision (Bennett 1962) provides a "
            "complete and reliable list of all sources north of declination "
            "-5 deg with flux density >9 Jy at 178 MHz, excluding regions near "
            "the Galactic ridge. Contains 328 sources and is the "
            "best-studied sample of powerful radio-loud AGN, quasars, and radio "
            "galaxies. Historically foundational for radio galaxy classification "
            "(Fanaroff-Riley types) and AGN unification models. Coordinates "
            "are B1950 (FK4), automatically converted to ICRS by RRIVis. "
            "S178MHz is the flux density at 178 MHz in Jy. Via VizieR VIII/1."
        ),
        "ra_col": "RA1950",
        "dec_col": "DE1950",
        "flux_col": "S178MHz",
        "flux_unit": "Jy",
        "spindex_col": None,
        "default_spindex": -0.7,
        "freq_mhz": 178.0,
        "coords_sexagesimal": True,
        "coord_frame": "fk4",  # B1950 → converted to ICRS via .icrs
        "major_col": None,
        "minor_col": None,
        "pa_col": None,
    },
    # --- GLEAM family ---
    "gleam_egc": {
        "vizier_id": "VIII/100/gleamegc",
        "table": None,
        "description": (
            "GLEAM EGC: GaLactic and Extragalactic All-sky MWA Survey — "
            "Extragalactic Catalogue (Hurley-Walker, Callingham, Hancock, "
            "Franzen, Hindson, Kapinska, et al., MNRAS 464, 1146, 2017). "
            "https://ui.adsabs.harvard.edu/abs/2017MNRAS.464.1146H/abstract "
            "A 72-231 MHz continuum survey using the Murchison Widefield Array "
            "(MWA Phase I, 128 tiles, Murchison Radio-astronomy Observatory, "
            "Western Australia). Sources selected from a time- and frequency-"
            "integrated wideband image centred at 200 MHz with ~2 arcmin "
            "resolution. 20 separate flux density measurements across the "
            "72-231 MHz band. Sky coverage: 24,831 deg^2 — the entire southern "
            "sky excluding the Magellanic Clouds and |b| < 10 deg. "
            "Catalog contains 307,455 sources. 90% complete at 170 mJy, 50% "
            "complete at 55 mJy, reliability 99.97% above 5-sigma (~50 mJy). "
            "Fpwide is the wideband (170-231 MHz) peak flux density in Jy. "
            "The alpha column provides per-source spectral indices fitted "
            "across the 72-231 MHz band. Via VizieR VIII/100/gleamegc."
        ),
        "ra_col": "RAJ2000",
        "dec_col": "DEJ2000",
        "flux_col": "Fpwide",
        "flux_unit": "Jy",
        "spindex_col": "alpha",
        "default_spindex": -0.8,
        "freq_mhz": 200.0,
        "coords_sexagesimal": False,
        "coord_frame": "icrs",
        "major_col": None,
        "minor_col": None,
        "pa_col": None,
    },
    "gleam_x_dr1": {
        "vizier_id": "VIII/110/catalog",
        "table": None,
        "description": (
            "GLEAM-X DR1: GaLactic and Extragalactic All-sky MWA eXtended — "
            "Data Release 1 (Hurley-Walker, Hancock, Franzen, Callingham, "
            "Duchesne, et al., PASA 39, e035, 2022). "
            "https://ui.adsabs.harvard.edu/abs/2022PASA...39...35H/abstract "
            "Observations at 72-231 MHz using the MWA Phase II extended "
            "configuration (2018-2020), providing higher resolution than "
            "the original GLEAM survey. DR1 covers 1,447 deg^2 "
            "(4h <= RA <= 13h, -32.7 deg <= Dec <= -20.7 deg). Resolution "
            "~45 arcsec to 2 arcmin depending on frequency band. "
            "Catalog contains 78,967 components, of which 71,320 are "
            "spectrally fitted. Fpwide is the wideband (170-231 MHz) peak "
            "flux density in Jy. The alpha-SP column provides per-source "
            "spectral indices. Via VizieR VIII/110/catalog."
        ),
        "ra_col": "RAJ2000",
        "dec_col": "DEJ2000",
        "flux_col": "Fpwide",
        "flux_unit": "Jy",
        "spindex_col": "alpha-SP",
        "default_spindex": -0.8,
        "freq_mhz": 200.0,
        "coords_sexagesimal": False,
        "coord_frame": "icrs",
        "major_col": None,
        "minor_col": None,
        "pa_col": None,
    },
    "gleam_x_dr2": {
        "vizier_id": "VIII/113/catalog2",
        "table": None,
        "description": (
            "GLEAM-X DR2: GaLactic and Extragalactic All-sky MWA eXtended — "
            "Data Release 2 (Ross, Hurley-Walker, Hancock, Duchesne, Riseley, "
            "et al., PASA 41, e054, 2024). "
            "https://ui.adsabs.harvard.edu/abs/2024PASA...41...54R/abstract "
            "Covers 12,892 deg^2 around the South Galactic Pole "
            "(20h40m <= RA <= 6h40m, -90 deg <= Dec <= +30 deg) at 72-231 MHz "
            "using over 1,000 hours of MWA Phase II observations (2020). "
            "Source finding performed on a 170-231 MHz wideband mosaic with "
            "median rms noise 1.5 mJy/beam. Catalog contains 624,866 "
            "components, of which 562,302 are spectrally fitted. "
            "98% complete at 50 mJy, reliability 98.7% at 5-sigma. "
            "Fpwide is the wideband peak flux density in Jy. "
            "Via VizieR VIII/113/catalog2."
        ),
        "ra_col": "RAJ2000",
        "dec_col": "DEJ2000",
        "flux_col": "Fpwide",
        "flux_unit": "Jy",
        "spindex_col": "alpha-SP",
        "default_spindex": -0.8,
        "freq_mhz": 200.0,
        "coords_sexagesimal": False,
        "coord_frame": "icrs",
        "major_col": None,
        "minor_col": None,
        "pa_col": None,
    },
    "gleam_gal": {
        "vizier_id": "VIII/102/gleamgal",
        "table": None,
        "description": (
            "GLEAM Galactic Plane: GaLactic and Extragalactic All-sky MWA "
            "Survey II — Galactic Plane (Hurley-Walker, Hancock, Franzen, "
            "Callingham, Hindson, et al., PASA 36, e047, 2019). "
            "https://ui.adsabs.harvard.edu/abs/2019PASA...36...47H/abstract "
            "The Galactic plane component of GLEAM covering |b| < 10 deg "
            "for two longitude ranges: 345 deg < l < 67 deg (inner Galaxy) "
            "and 180 deg < l < 240 deg (outer Galaxy), at 72-231 MHz using "
            "MWA Phase I. Unlike the extragalactic catalog, this analysis "
            "used multi-scale CLEAN to better deconvolve large-scale Galactic "
            "structure. Source finding on a 60 MHz bandwidth image centred at "
            "200 MHz. Catalog contains 22,037 compact source components with "
            "rms position accuracy better than 2 arcsec. Fpwide is the wideband "
            "peak flux density in Jy. Via VizieR VIII/102/gleamgal."
        ),
        "ra_col": "RAJ2000",
        "dec_col": "DEJ2000",
        "flux_col": "Fpwide",
        "flux_unit": "Jy",
        "spindex_col": "alpha",
        "default_spindex": -0.8,
        "freq_mhz": 200.0,
        "coords_sexagesimal": False,
        "coord_frame": "icrs",
        "major_col": None,
        "minor_col": None,
        "pa_col": None,
    },
    # --- MALS family ---
    "mals_dr1": {
        "vizier_id": "J/ApJS/270/33",
        "table": "catalog",
        "description": (
            "MALS DR1: MeerKAT Absorption Line Survey Data Release 1 — "
            "Stokes I Image Catalogs (Deka, Gupta, Jagannathan, Sekhar, "
            "Momjian, et al., ApJS 270, 33, 2024). "
            "https://ui.adsabs.harvard.edu/abs/2024ApJS..270...33D/abstract "
            "L-band (900-1670 MHz) continuum survey using MeerKAT (64-dish "
            "radio interferometer, Karoo, South Africa) from 391 pointings. "
            "Median angular resolution ~12 arcsec at 1006 MHz and ~8 arcsec "
            "at 1381 MHz. Median rms noise ~25 uJy/beam (1006 MHz) and "
            "~22 uJy/beam (1381 MHz). DR1 covers 2,289 deg^2 and contains "
            "495,325 sources detected at >5-sigma. Primary science goal is "
            "HI and OH absorption line studies, with continuum catalogs as "
            "a key data product. Flux is the integrated flux density in mJy. "
            "SpMALS provides in-band spectral indices. Via VizieR J/ApJS/270/33."
        ),
        "ra_col": "RAJ2000",
        "dec_col": "DEJ2000",
        "flux_col": "Flux",
        "flux_unit": "mJy",
        "spindex_col": "SpMALS",
        "default_spindex": 0.0,
        "freq_mhz": 1200.0,
        "coords_sexagesimal": False,
        "coord_frame": "icrs",
        "major_col": None,
        "minor_col": None,
        "pa_col": None,
    },
    "mals_dr2": {
        "vizier_id": "J/A+A/690/A163",
        "table": "all",
        "description": (
            "MALS DR2: MeerKAT Absorption Line Survey Data Release 2 — "
            "Wideband Continuum Catalogues (Wagenveld, Gupta, Chen, Sekhar, "
            "Jagannathan, et al., A&A 690, A163, 2024). "
            "https://ui.adsabs.harvard.edu/abs/2024A&A...690A.163W/abstract "
            "Full wideband L-band continuum catalog from 391 MeerKAT pointings "
            "covering 4,344 deg^2, reaching a depth of ~10 uJy/beam. Contains "
            "971,980 sources spanning five orders of magnitude in flux density "
            "down to ~200 uJy, providing a robust view of the extragalactic "
            "radio source population. FluxTot is the total integrated flux "
            "density in mJy. SpIndex provides wideband spectral indices. "
            "Via VizieR J/A+A/690/A163."
        ),
        "ra_col": "RAJ2000",
        "dec_col": "DEJ2000",
        "flux_col": "FluxTot",
        "flux_unit": "mJy",
        "spindex_col": "SpIndex",
        "default_spindex": 0.0,
        "freq_mhz": 1284.0,
        "coords_sexagesimal": False,
        "coord_frame": "icrs",
        "major_col": None,
        "minor_col": None,
        "pa_col": None,
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
        "description": (
            "RACS-Low DR1: Rapid ASKAP Continuum Survey at 887.5 MHz "
            "(McConnell, Hale, Lenc, Banfield, George, et al., PASA 37, e048, "
            "2020 [survey design]; Hale, McConnell, Thomson, Lenc, Vernstrom, "
            "et al., PASA 38, e058, 2021 [source catalog]). "
            "https://ui.adsabs.harvard.edu/abs/2020PASA...37...48M/abstract "
            "First all-sky survey with the Australian Square Kilometre Array "
            "Pathfinder (ASKAP, 36 x 12-m antennas, Murchison Radio-astronomy "
            "Observatory, Western Australia). 288 MHz bandwidth centred at "
            "887.5 MHz, resolution ~15 arcsec (convolved to 25 arcsec for the "
            "catalog). Sky coverage: declination -80 deg to +30 deg, excluding "
            "|b| < 5 deg (~2.1 million sources). Via CASDA TAP."
        ),
    },
    "mid": {
        "freq_mhz": 1367.5,
        "tap_table": "casda.racs_mid_dr1_components_v01",
        "ra_col": "ra_deg_cont",
        "dec_col": "dec_deg_cont",
        "flux_col": "flux_peak",
        "flux_unit": "mJy",
        "description": (
            "RACS-Mid DR1: Rapid ASKAP Continuum Survey at 1367.5 MHz "
            "(Duchesne, Thomson, Hale, Whiting, Grundy, et al., PASA 41, "
            "e003, 2024). "
            "https://ui.adsabs.harvard.edu/abs/2024PASA...41....3D/abstract "
            "Second epoch of RACS using ASKAP (36 x 12-m antennas), 288 MHz "
            "bandwidth centred at 1367.5 MHz. Sky coverage: declination south "
            "of +49 deg. Via CASDA TAP."
        ),
    },
    "high": {
        "freq_mhz": 1655.5,
        "tap_table": "casda.racs_high_dr1_components_v01",
        "ra_col": "ra_deg_cont",
        "dec_col": "dec_deg_cont",
        "flux_col": "flux_peak",
        "flux_unit": "mJy",
        "description": (
            "RACS-High DR1: Rapid ASKAP Continuum Survey at 1655.5 MHz "
            "(Duchesne, Ross, Thomson, Lenc, Murphy, Galvin, Hotan, Moss, "
            "Whiting, PASA 42, e038, 2025). "
            "https://ui.adsabs.harvard.edu/abs/2025PASA...42...38D/abstract "
            "Third epoch of RACS using ASKAP (36 x 12-m antennas), 288 MHz "
            "bandwidth centred at 1655.5 MHz. Median resolution "
            "~11.8 arcsec x 8.1 arcsec. Observed 2021-2022. Catalog contains "
            "~2.7 million sources with 99.2% reliability. Via CASDA TAP."
        ),
    },
}
