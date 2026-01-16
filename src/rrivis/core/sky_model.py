# rrivis/core/sky_model.py
"""
Unified Sky Model for RRIVis.

This module provides a unified `SkyModel` class that can represent sky brightness
in two formats with **bidirectional conversion**:

1. **Point Sources**: List of discrete sources with (RA, Dec, flux, spectral_index)
2. **HEALPix Maps**: Pixelized brightness temperature maps (for diffuse emission)

The key feature is that ANY sky model can be converted to EITHER representation:
- Point sources (GLEAM, MALS, test) can be converted to HEALPix maps
- HEALPix maps (GSM, LFSM, Haslam) can be converted to point sources

This allows uniform treatment of all sky models regardless of their native format.

Examples
--------
>>> # Load GLEAM as point sources (native format)
>>> sky = SkyModel.from_gleam(flux_limit=1.0)
>>> sources = sky.to_point_sources()

>>> # Load GLEAM as HEALPix map (converts from point sources)
>>> sky = SkyModel.from_gleam(flux_limit=1.0)
>>> temp_map, nside, spec_map = sky.to_healpix(nside=64, frequency=150e6)

>>> # Load GSM as HEALPix map (native format)
>>> sky = SkyModel.from_diffuse_sky(model="gsm2008", frequency=100e6)
>>> temp_map, nside, spec_map = sky.get_healpix_data()

>>> # Load GSM as point sources (converts from HEALPix)
>>> sky = SkyModel.from_diffuse_sky(model="gsm2008", frequency=100e6)
>>> sources = sky.to_point_sources(flux_limit=1.0)

>>> # Combine multiple sky models
>>> combined = SkyModel.combine([gleam_sky, gsm_sky], frequency=100e6)

References
----------
- GLEAM: Hurley-Walker et al. (2017), MNRAS, 464, 1146
- MALS DR1: Deka et al. (2024), ApJS, 270, 33
- MALS DR2: Wagenveld et al. (2024), A&A, 690, A163
- GSM2008: de Oliveira-Costa et al. (2008), MNRAS, 388, 247
- GSM2016: Zheng et al. (2016), MNRAS, 464, 3486
- LFSM: Dowell et al. (2017), MNRAS, 469, 4537
- Haslam: Remazeilles et al. (2015), MNRAS, 451, 4311
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
import logging

import numpy as np
import healpy as hp
from astropy.coordinates import SkyCoord
import astropy.units as u
from healpy.rotator import Rotator

from pygdsm import GlobalSkyModel, GlobalSkyModel16, LowFrequencySkyModel, HaslamSkyModel


logger = logging.getLogger(__name__)


# =============================================================================
# Physical Constants
# =============================================================================

K_BOLTZMANN = 1.380649e-23  # Boltzmann constant (J/K)
C_LIGHT = 299792458  # Speed of light (m/s)


# =============================================================================
# Catalog Metadata
# =============================================================================

# Available GLEAM catalogs in VizieR
GLEAM_CATALOGS = {
    "VIII/100/gleamegc": "GLEAM EGC catalog, version 2 (307,455 rows)",
    "VIII/113/catalog2": "GLEAM-X DR2 (624,866 rows)",
    "VIII/102/gleamgal": "GLEAM Galactic plane (22,037 rows)",
    "VIII/109/gleamsgp": "GLEAM SGP region (108,851 rows)",
    "VIII/110/catalog": "GLEAM-X DR1 (78,967 rows)",
    "VIII/105/catalog": "G4Jy catalog (1,960 rows)",
}

# MALS catalog metadata
MALS_CATALOGS = {
    "dr1": {
        "vizier_id": "J/ApJS/270/33",
        "table": "catalog",
        "description": "MALS DR1: Stokes I at 1-1.4 GHz (495,325 sources)",
        "ra_col": "RAJ2000",
        "dec_col": "DEJ2000",
        "flux_col": "Flux",
        "spindex_col": "SpMALS",
        "freq_mhz": 1200,
    },
    "dr2": {
        "vizier_id": "J/A+A/690/A163",
        "table": "all",
        "description": "MALS DR2: Wideband continuum (971,980 sources)",
        "ra_col": "RAJ2000",
        "dec_col": "DEJ2000",
        "flux_col": "FluxTot",
        "spindex_col": "SpIndex",
        "freq_mhz": 1284,
    },
    "dr3": {
        "vizier_id": "J/A+A/698/A120",
        "table": "catalog",
        "description": "MALS DR3: HI 21-cm absorption (3,640 features)",
        "ra_col": "RAJ2000",
        "dec_col": "DEJ2000",
        "flux_col": "PeakFlux",
        "spindex_col": None,
        "freq_mhz": 1420,
        "coords_sexagesimal": True,
    },
}

# Diffuse sky models
DIFFUSE_MODELS = {
    "gsm2008": {
        "class": GlobalSkyModel,
        "description": "Global Sky Model 2008 (10 MHz - 100 GHz)",
        "freq_range": (10e6, 100e9),
        "init_kwargs": {"freq_unit": "Hz"},
    },
    "gsm2016": {
        "class": GlobalSkyModel16,
        "description": "Global Sky Model 2016 (10 MHz - 5 THz)",
        "freq_range": (10e6, 5e12),
        "init_kwargs": {"freq_unit": "Hz", "data_unit": "TRJ"},
    },
    "lfsm": {
        "class": LowFrequencySkyModel,
        "description": "Low Frequency Sky Model (10 - 408 MHz)",
        "freq_range": (10e6, 408e6),
        "init_kwargs": {"freq_unit": "Hz"},
    },
    "haslam": {
        "class": HaslamSkyModel,
        "description": "Haslam 408 MHz map with spectral scaling",
        "freq_range": (10e6, 100e9),
        "init_kwargs": {"freq_unit": "Hz", "spectral_index": -2.6},
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

    Attributes
    ----------
    _point_sources : list of dict, optional
        Point source representation. Each dict contains:
        - coords: SkyCoord
        - flux: float (Jy)
        - spectral_index: float
        - stokes_q, stokes_u, stokes_v: float
    _healpix_map : np.ndarray, optional
        HEALPix brightness temperature map in Kelvin.
    _healpix_nside : int, optional
        HEALPix NSIDE parameter.
    _spectral_index_map : np.ndarray, optional
        Per-pixel spectral indices (for HEALPix mode).
    _native_format : str
        Original format: "point_sources" or "healpix"
    frequency : float, optional
        Reference frequency in Hz.
    model_name : str, optional
        Name of the sky model.
    """

    # Internal storage
    _point_sources: Optional[List[Dict[str, Any]]] = None
    _healpix_map: Optional[np.ndarray] = None
    _healpix_nside: Optional[int] = None
    _spectral_index_map: Optional[np.ndarray] = None

    # Track native format
    _native_format: str = "point_sources"

    # Metadata
    frequency: Optional[float] = None
    model_name: Optional[str] = None

    # Cached data
    _pixel_coords: Optional[SkyCoord] = field(default=None, repr=False)
    _pixel_solid_angle: Optional[float] = field(default=None, repr=False)

    # =========================================================================
    # Properties
    # =========================================================================

    @property
    def mode(self) -> str:
        """Return the current representation mode."""
        if self._healpix_map is not None:
            return "healpix"
        return "point_sources"

    @property
    def native_format(self) -> str:
        """Return the native/original format of this sky model."""
        return self._native_format

    @property
    def n_sources(self) -> int:
        """Return the number of sources/pixels."""
        if self._healpix_map is not None:
            return len(self._healpix_map)
        return len(self._point_sources) if self._point_sources else 0

    @property
    def pixel_solid_angle(self) -> Optional[float]:
        """Return the solid angle per HEALPix pixel in steradians."""
        if self._pixel_solid_angle is None and self._healpix_nside is not None:
            npix = hp.nside2npix(self._healpix_nside)
            self._pixel_solid_angle = 4 * np.pi / npix
        return self._pixel_solid_angle

    @property
    def pixel_coords(self) -> Optional[SkyCoord]:
        """Return SkyCoord for all HEALPix pixel centers (computed lazily)."""
        if self._pixel_coords is None and self._healpix_nside is not None:
            self._pixel_coords = self._compute_pixel_coords()
        return self._pixel_coords

    def _compute_pixel_coords(self) -> SkyCoord:
        """Compute equatorial coordinates for all HEALPix pixels."""
        nside = self._healpix_nside
        npix = hp.nside2npix(nside)
        theta, phi = hp.pix2ang(nside, np.arange(npix))
        ra = np.rad2deg(phi)
        dec = 90 - np.rad2deg(theta)
        return SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame="icrs")

    # =========================================================================
    # Conversion Methods
    # =========================================================================

    def to_point_sources(
        self,
        flux_limit: float = 0.0,
        frequency: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Get sky model as point sources.

        If native format is point sources, returns them directly.
        If native format is HEALPix, converts each pixel to a point source.

        Parameters
        ----------
        flux_limit : float, default=0.0
            Minimum flux in Jy to include.
        frequency : float, optional
            Frequency for T_b to Jy conversion (required for HEALPix conversion).

        Returns
        -------
        list of dict
            Point sources with coords, flux, spectral_index, stokes.
        """
        # If we have point sources, return them
        if self._point_sources is not None:
            if flux_limit > 0:
                return [s for s in self._point_sources if s["flux"] >= flux_limit]
            return self._point_sources

        # Convert from HEALPix
        if self._healpix_map is None:
            return []

        freq = frequency or self.frequency
        if freq is None:
            raise ValueError("Frequency required for HEALPix to point source conversion")

        return self._healpix_to_point_sources(flux_limit, freq)

    def _healpix_to_point_sources(
        self,
        flux_limit: float,
        frequency: float
    ) -> List[Dict[str, Any]]:
        """Convert HEALPix map to point sources."""
        coords = self.pixel_coords
        omega = self.pixel_solid_angle

        # Rayleigh-Jeans: S = (2 k_B T ν²) / c² × Ω
        flux_jy = (
            (2 * K_BOLTZMANN * self._healpix_map * (frequency ** 2)) / (C_LIGHT ** 2)
        ) * omega / 1e-26

        # Get spectral indices
        if self._spectral_index_map is not None:
            spec_indices = self._spectral_index_map
        else:
            spec_indices = np.zeros(len(flux_jy))

        # Apply flux limit
        valid = flux_jy >= flux_limit
        if not np.any(valid):
            logger.warning(f"No sources above flux limit {flux_limit} Jy")
            return []

        # Create source list
        sources = []
        for i in np.where(valid)[0]:
            sources.append({
                "coords": coords[i],
                "flux": float(flux_jy[i]),
                "spectral_index": float(spec_indices[i]),
                "stokes_q": 0.0,
                "stokes_u": 0.0,
                "stokes_v": 0.0,
            })

        return sources

    def to_healpix(
        self,
        nside: int = 64,
        frequency: Optional[float] = None
    ) -> Tuple[np.ndarray, int, Optional[np.ndarray]]:
        """
        Get sky model as HEALPix map.

        If native format is HEALPix, returns the map (possibly resampled).
        If native format is point sources, converts by binning into pixels.

        Parameters
        ----------
        nside : int, default=64
            HEALPix NSIDE parameter for output.
        frequency : float, optional
            Frequency for Jy to T_b conversion (required for point source conversion).

        Returns
        -------
        temp_map : np.ndarray
            Brightness temperature map in Kelvin.
        nside : int
            Output NSIDE.
        spec_idx_map : np.ndarray or None
            Per-pixel spectral indices.
        """
        # If we have HEALPix data
        if self._healpix_map is not None:
            if nside == self._healpix_nside:
                return self._healpix_map, self._healpix_nside, self._spectral_index_map
            # Resample to requested nside
            temp_map = hp.ud_grade(self._healpix_map, nside_out=nside)
            spec_map = None
            if self._spectral_index_map is not None:
                spec_map = hp.ud_grade(self._spectral_index_map, nside_out=nside)
            return temp_map, nside, spec_map

        # Convert from point sources
        if self._point_sources is None or len(self._point_sources) == 0:
            npix = hp.nside2npix(nside)
            return np.zeros(npix), nside, np.zeros(npix)

        freq = frequency or self.frequency
        if freq is None:
            raise ValueError("Frequency required for point source to HEALPix conversion")

        return self._point_sources_to_healpix(nside, freq)

    def _point_sources_to_healpix(
        self,
        nside: int,
        frequency: float
    ) -> Tuple[np.ndarray, int, np.ndarray]:
        """
        Convert point sources to HEALPix brightness temperature map.

        For each source:
        1. Find pixel containing source coords
        2. Convert flux (Jy) to brightness temperature (K)
        3. Add to pixel (accumulate if multiple sources in same pixel)
        """
        npix = hp.nside2npix(nside)
        temp_map = np.zeros(npix)
        spec_idx_map = np.zeros(npix)
        source_count = np.zeros(npix)

        omega_pixel = 4 * np.pi / npix  # Pixel solid angle in steradians

        for source in self._point_sources:
            # Get pixel index
            ra = source["coords"].ra.deg
            dec = source["coords"].dec.deg
            theta = np.radians(90 - dec)  # Colatitude
            phi = np.radians(ra)
            ipix = hp.ang2pix(nside, theta, phi)

            # Convert Jy to K (inverse Rayleigh-Jeans)
            # T_b = S × c² / (2 k_B ν² Ω) × 10⁻²⁶
            flux_jy = source["flux"]
            T_b = flux_jy * (C_LIGHT ** 2) / (2 * K_BOLTZMANN * frequency ** 2 * omega_pixel) * 1e-26

            temp_map[ipix] += T_b
            spec_idx_map[ipix] += source.get("spectral_index", 0.0)
            source_count[ipix] += 1

        # Average spectral index for pixels with multiple sources
        mask = source_count > 0
        spec_idx_map[mask] /= source_count[mask]

        logger.info(f"Converted {len(self._point_sources)} point sources to HEALPix (nside={nside})")

        return temp_map, nside, spec_idx_map

    def get_healpix_data(self) -> Tuple[np.ndarray, int, Optional[np.ndarray]]:
        """
        Get HEALPix map data for direct visibility calculation.

        Returns
        -------
        temperature_map : np.ndarray
            Brightness temperature in Kelvin.
        nside : int
            HEALPix NSIDE parameter.
        spectral_index_map : np.ndarray or None

        Raises
        ------
        ValueError
            If no HEALPix data is available.
        """
        if self._healpix_map is None:
            raise ValueError("No HEALPix data. Use to_healpix() to convert from point sources.")
        return self._healpix_map, self._healpix_nside, self._spectral_index_map

    def get_for_visibility(
        self,
        representation: str,
        nside: int = 64,
        flux_limit: float = 0.0,
        frequency: Optional[float] = None
    ) -> "SkyModel":
        """
        Ensure sky model is in the requested representation.

        This is the main method to call before visibility calculation.
        It converts the sky model to the requested format if needed.

        Parameters
        ----------
        representation : str
            "point_sources" or "healpix_map"
        nside : int, default=64
            HEALPix NSIDE for healpix_map mode.
        flux_limit : float, default=0.0
            Minimum flux for point_sources mode.
        frequency : float, optional
            Frequency for conversion.

        Returns
        -------
        SkyModel
            Self with the requested representation populated.
        """
        freq = frequency or self.frequency

        if representation == "point_sources":
            if self._point_sources is None:
                self._point_sources = self._healpix_to_point_sources(flux_limit, freq)
        elif representation == "healpix_map":
            if self._healpix_map is None:
                self._healpix_map, self._healpix_nside, self._spectral_index_map = \
                    self._point_sources_to_healpix(nside, freq)

        return self

    # =========================================================================
    # Factory Methods
    # =========================================================================

    @classmethod
    def from_test_sources(
        cls,
        num_sources: int = 100,
        flux_range: Tuple[float, float] = (2.0, 8.0),
        dec_deg: float = -30.72,
        spectral_index: float = -0.8
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

        Returns
        -------
        SkyModel
            Sky model with test sources.
        """
        sources = []

        if num_sources == 1:
            sources.append({
                "coords": SkyCoord(ra=0 * u.deg, dec=dec_deg * u.deg),
                "flux": (flux_range[0] + flux_range[1]) / 2,
                "spectral_index": spectral_index,
                "stokes_q": 0.0,
                "stokes_u": 0.0,
                "stokes_v": 0.0,
            })
        else:
            for i in range(num_sources):
                ra_deg = (360.0 / num_sources) * i
                flux = flux_range[0] + (flux_range[1] - flux_range[0]) * i / (num_sources - 1)

                sources.append({
                    "coords": SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg),
                    "flux": flux,
                    "spectral_index": spectral_index,
                    "stokes_q": 0.0,
                    "stokes_u": 0.0,
                    "stokes_v": 0.0,
                })

        logger.info(f"Generated {num_sources} test sources")

        return cls(
            _point_sources=sources,
            _native_format="point_sources",
            model_name="test_sources",
        )

    @classmethod
    def from_gleam(
        cls,
        flux_limit: float = 1.0,
        catalog: str = "VIII/100/gleamegc"
    ) -> "SkyModel":
        """
        Load GLEAM catalog from VizieR.

        Parameters
        ----------
        flux_limit : float, default=1.0
            Minimum flux density in Jy.
        catalog : str, default="VIII/100/gleamegc"
            VizieR catalog ID.

        Returns
        -------
        SkyModel
            Sky model with GLEAM sources.
        """
        from astroquery.vizier import Vizier

        if catalog not in GLEAM_CATALOGS:
            logger.warning(f"Unknown catalog {catalog}. Available: {list(GLEAM_CATALOGS.keys())}")

        Vizier.ROW_LIMIT = -1

        logger.info(f"Fetching GLEAM catalog: {catalog}")
        logger.info("Downloading from VizieR...")

        try:
            catalog_data = Vizier.get_catalogs(catalog)[0]
        except Exception as e:
            logger.error(f"Failed to fetch GLEAM: {e}")
            return cls(_point_sources=[], model_name="gleam")

        logger.info(f"Downloaded {len(catalog_data)} sources, processing...")

        sources = []
        for row in catalog_data:
            flux = row["Fp076"]
            if flux >= flux_limit:
                ra = row["RAJ2000"] * u.deg
                dec = row["DEJ2000"] * u.deg

                # Get spectral index
                spindex = 0.0
                if "alpha" in row.colnames:
                    val = row["alpha"]
                    if isinstance(val, (float, np.floating)) and np.isfinite(val):
                        spindex = float(val)

                sources.append({
                    "coords": SkyCoord(ra=ra, dec=dec),
                    "flux": float(flux),
                    "spectral_index": spindex,
                    "stokes_q": 0.0,
                    "stokes_u": 0.0,
                    "stokes_v": 0.0,
                })

        logger.info(f"GLEAM loaded: {len(sources)} sources (flux >= {flux_limit} Jy)")

        return cls(
            _point_sources=sources,
            _native_format="point_sources",
            model_name="gleam",
        )

    @classmethod
    def from_mals(
        cls,
        flux_limit: float = 1.0,
        release: str = "dr2"
    ) -> "SkyModel":
        """
        Load MALS catalog from VizieR.

        Parameters
        ----------
        flux_limit : float, default=1.0
            Minimum flux density in mJy.
        release : str, default="dr2"
            Data release: "dr1", "dr2", or "dr3".

        Returns
        -------
        SkyModel
            Sky model with MALS sources.
        """
        from astroquery.vizier import Vizier

        release = release.lower()
        if release not in MALS_CATALOGS:
            logger.error(f"Invalid MALS release. Available: {list(MALS_CATALOGS.keys())}")
            return cls(_point_sources=[], model_name=f"mals_{release}")

        info = MALS_CATALOGS[release]
        Vizier.ROW_LIMIT = -1

        logger.info(f"Fetching {info['description']}")
        logger.info("Downloading from VizieR...")

        try:
            tables = Vizier.get_catalogs(info["vizier_id"])
            if not tables:
                raise ValueError("No tables returned")

            # Find the correct table
            catalog = None
            for t in tables:
                if info["table"] in t.meta.get("name", ""):
                    catalog = t
                    break
            if catalog is None:
                catalog = tables[0]

        except Exception as e:
            logger.error(f"Failed to fetch MALS: {e}")
            return cls(_point_sources=[], model_name=f"mals_{release}")

        logger.info(f"Downloaded {len(catalog)} sources, processing...")

        sources = []
        is_sexagesimal = info.get("coords_sexagesimal", False)

        for row in catalog:
            try:
                flux_mjy = row[info["flux_col"]]
                if np.isnan(flux_mjy) or flux_mjy < flux_limit:
                    continue

                flux_jy = flux_mjy / 1000.0

                # Parse coordinates
                if is_sexagesimal:
                    coords = SkyCoord(
                        str(row[info["ra_col"]]),
                        str(row[info["dec_col"]]),
                        unit=(u.hourangle, u.deg),
                        frame="icrs"
                    )
                else:
                    ra = row[info["ra_col"]]
                    dec = row[info["dec_col"]]
                    if np.isnan(ra) or np.isnan(dec):
                        continue
                    coords = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame="icrs")

                # Get spectral index
                spindex = 0.0
                if info["spindex_col"] and info["spindex_col"] in row.colnames:
                    val = row[info["spindex_col"]]
                    if not np.ma.is_masked(val) and np.isfinite(val):
                        spindex = float(val)

                sources.append({
                    "coords": coords,
                    "flux": float(flux_jy),
                    "spectral_index": spindex,
                    "stokes_q": 0.0,
                    "stokes_u": 0.0,
                    "stokes_v": 0.0,
                })

            except Exception as e:
                logger.debug(f"Skipping row: {e}")
                continue

        logger.info(f"MALS {release.upper()} loaded: {len(sources)} sources (flux >= {flux_limit} mJy)")

        return cls(
            _point_sources=sources,
            _native_format="point_sources",
            model_name=f"mals_{release}",
        )

    @classmethod
    def from_diffuse_sky(
        cls,
        model: str = "gsm2008",
        frequency: float = 100e6,
        nside: int = 32,
        compute_spectral_index: bool = True,
        reference_frequency: Optional[float] = None,
        include_cmb: bool = False,
    ) -> "SkyModel":
        """
        Load a diffuse sky model (GSM, LFSM, Haslam).

        Parameters
        ----------
        model : str, default="gsm2008"
            Model name: "gsm2008", "gsm2016", "lfsm", "haslam".
        frequency : float, default=100e6
            Observation frequency in Hz.
        nside : int, default=32
            HEALPix NSIDE resolution.
        compute_spectral_index : bool, default=True
            Compute per-pixel spectral indices.
        reference_frequency : float, optional
            Second frequency for spectral index (default: 2× frequency).
        include_cmb : bool, default=False
            Include CMB contribution.

        Returns
        -------
        SkyModel
            Sky model with HEALPix data.
        """
        model = model.lower()
        if model not in DIFFUSE_MODELS:
            raise ValueError(f"Unknown model '{model}'. Available: {list(DIFFUSE_MODELS.keys())}")

        info = DIFFUSE_MODELS[model]
        model_class = info["class"]

        logger.info(f"Loading {model.upper()} at {frequency/1e6:.1f} MHz, nside={nside}")

        # Initialize sky model
        init_kwargs = info["init_kwargs"].copy()
        init_kwargs["include_cmb"] = include_cmb

        sky = model_class(**init_kwargs)

        # Generate temperature map
        temp_map = sky.generate(frequency)

        # Compute spectral index if requested
        spec_idx_map = None
        if compute_spectral_index:
            ref_freq = reference_frequency or frequency * 2.0
            logger.info(f"Computing spectral indices ({frequency/1e6:.1f} → {ref_freq/1e6:.1f} MHz)")

            temp_map_ref = sky.generate(ref_freq)

            with np.errstate(divide='ignore', invalid='ignore'):
                beta = np.log(temp_map / temp_map_ref) / np.log(frequency / ref_freq)
                spec_idx_map = beta + 2.0  # Temperature to flux spectral index

            # Handle invalid values
            bad = ~np.isfinite(spec_idx_map)
            if np.any(bad):
                spec_idx_map[bad] = -0.7

            logger.info(f"Spectral index: mean={spec_idx_map.mean():.3f}")

        # Downgrade to requested nside
        temp_map = hp.ud_grade(temp_map, nside_out=nside)
        if spec_idx_map is not None:
            spec_idx_map = hp.ud_grade(spec_idx_map, nside_out=nside)

        # Rotate from Galactic to Equatorial
        rot = Rotator(coord=["G", "C"])
        temp_map = rot.rotate_map_pixel(temp_map)
        if spec_idx_map is not None:
            spec_idx_map = rot.rotate_map_pixel(spec_idx_map)

        logger.info(f"{model.upper()} loaded: {hp.nside2npix(nside)} pixels")

        return cls(
            _healpix_map=temp_map,
            _healpix_nside=nside,
            _spectral_index_map=spec_idx_map,
            _native_format="healpix",
            frequency=frequency,
            model_name=model,
        )

    @classmethod
    def from_point_sources(
        cls,
        sources: List[Dict[str, Any]],
        model_name: str = "custom"
    ) -> "SkyModel":
        """
        Create SkyModel from existing point source list.

        Parameters
        ----------
        sources : list of dict
            Point source list.
        model_name : str, default="custom"
            Name for the model.

        Returns
        -------
        SkyModel
        """
        return cls(
            _point_sources=sources,
            _native_format="point_sources",
            model_name=model_name,
        )

    # =========================================================================
    # Combination
    # =========================================================================

    @classmethod
    def combine(
        cls,
        models: List["SkyModel"],
        representation: str = "point_sources",
        nside: int = 64,
        frequency: Optional[float] = None
    ) -> "SkyModel":
        """
        Combine multiple sky models into one.

        Parameters
        ----------
        models : list of SkyModel
            Sky models to combine.
        representation : str, default="point_sources"
            Output representation: "point_sources" or "healpix_map".
        nside : int, default=64
            HEALPix NSIDE for healpix_map output.
        frequency : float, optional
            Frequency for conversions.

        Returns
        -------
        SkyModel
            Combined sky model.
        """
        if not models:
            return cls(_point_sources=[], model_name="combined_empty")

        # Get frequency from first model if not specified
        freq = frequency
        if freq is None:
            for m in models:
                if m.frequency is not None:
                    freq = m.frequency
                    break

        if representation == "point_sources":
            # Concatenate all point sources
            all_sources = []
            for m in models:
                sources = m.to_point_sources(frequency=freq)
                all_sources.extend(sources)

            logger.info(f"Combined {len(models)} models: {len(all_sources)} total sources")

            return cls(
                _point_sources=all_sources,
                _native_format="point_sources",
                frequency=freq,
                model_name="combined",
            )

        else:  # healpix_map
            # Add all HEALPix maps
            npix = hp.nside2npix(nside)
            combined_temp = np.zeros(npix)
            combined_spec = np.zeros(npix)
            spec_count = np.zeros(npix)

            for m in models:
                temp, _, spec = m.to_healpix(nside=nside, frequency=freq)
                combined_temp += temp
                if spec is not None:
                    combined_spec += spec
                    spec_count += (spec != 0).astype(float)

            # Average spectral indices
            mask = spec_count > 0
            combined_spec[mask] /= spec_count[mask]

            logger.info(f"Combined {len(models)} models into HEALPix (nside={nside})")

            return cls(
                _healpix_map=combined_temp,
                _healpix_nside=nside,
                _spectral_index_map=combined_spec,
                _native_format="healpix",
                frequency=freq,
                model_name="combined",
            )

    # =========================================================================
    # String representation
    # =========================================================================

    def __repr__(self) -> str:
        if self._healpix_map is not None:
            return (
                f"SkyModel(native='{self._native_format}', model='{self.model_name}', "
                f"healpix_nside={self._healpix_nside}, n_pixels={self.n_sources}, "
                f"freq={self.frequency/1e6 if self.frequency else 'N/A'}MHz)"
            )
        return (
            f"SkyModel(native='{self._native_format}', model='{self.model_name}', "
            f"n_sources={self.n_sources})"
        )
