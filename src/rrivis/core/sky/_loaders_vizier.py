# rrivis/core/sky/_loaders_vizier.py
"""VizieR and CASDA TAP loader mixin for SkyModel."""

import logging
from typing import Any

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord
from astroquery.utils.tap.core import TapPlus
from astroquery.vizier import Vizier

from .catalogs import CASDA_TAP_URL, RACS_CATALOGS, VIZIER_POINT_CATALOGS

logger = logging.getLogger(__name__)


class _VizierLoadersMixin:
    """Mixin providing VizieR and CASDA TAP factory classmethods for SkyModel."""

    @classmethod
    def _load_from_vizier_catalog(
        cls,
        catalog_key: str,
        flux_limit: float = 1.0,
        brightness_conversion: str = "planck",
        precision: Any = None,
    ) -> "SkyModel":  # noqa: F821
        """
        Load a point-source catalog from VizieR using unified metadata.

        This private helper is called by all public from_*() VizieR wrappers.
        It handles flux unit conversion (mJy->Jy), coordinate parsing
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
        coord_frame = info.get("coord_frame", "icrs")
        flux_unit = info.get("flux_unit", "Jy")
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
            logger.info(
                f"{catalog_key.upper()}: no sources above flux limit {flux_limit} Jy"
            )
            return _empty()

        if is_sexagesimal:
            ra_strs = [
                str(row[info["ra_col"]])
                for i, row in enumerate(catalog)
                if flux_valid[i]
            ]
            dec_strs = [
                str(row[info["dec_col"]])
                for i, row in enumerate(catalog)
                if flux_valid[i]
            ]
            sc = SkyCoord(
                ra_strs, dec_strs, unit=(u.hourangle, u.deg), frame=coord_frame
            )
        else:
            ra_list, dec_list = [], []
            coord_ok = np.ones(n_rows, dtype=bool)
            for i, row in enumerate(catalog):
                if not flux_valid[i]:
                    coord_ok[i] = False
                    ra_list.append(0.0)
                    dec_list.append(0.0)
                    continue
                ra_val = row[info["ra_col"]]
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

        ra_rad = sc.ra.rad
        dec_rad = sc.dec.rad
        flux_jy = flux_jy_raw[flux_valid]
        n = len(flux_jy)

        rows_list = list(catalog)
        valid_indices = np.where(flux_valid)[0]
        default_spindex = info["default_spindex"]
        alpha_arr = np.full(n, default_spindex, dtype=np.float64)

        if spindex_from_cols is not None:
            # Two-frequency log-slope (e.g. AT20G)
            s_low_col = spindex_from_cols["s_low"]
            s_high_col = spindex_from_cols["s_high"]
            freq_low = spindex_from_cols["freq_low_hz"]
            freq_high = spindex_from_cols["freq_high_hz"]
            if s_low_col in catalog.colnames and s_high_col in catalog.colnames:
                for j, i in enumerate(valid_indices):
                    row = rows_list[i]
                    sl = row[s_low_col]
                    sh = row[s_high_col]
                    if (
                        not np.ma.is_masked(sl)
                        and not np.ma.is_masked(sh)
                        and np.isfinite(float(sl))
                        and np.isfinite(float(sh))
                        and float(sl) > 0
                        and float(sh) > 0
                    ):
                        alpha_arr[j] = np.log(float(sl) / float(sh)) / np.log(
                            freq_low / freq_high
                        )
        elif info.get("spindex_col") and info["spindex_col"] in catalog.colnames:
            for j, i in enumerate(valid_indices):
                row = rows_list[i]
                val = row[info["spindex_col"]]
                if not np.ma.is_masked(val) and np.isfinite(float(val)):
                    alpha_arr[j] = float(val)

        logger.info(
            f"{catalog_key.upper()} loaded: {n:,} sources (flux >= {flux_limit} Jy)"
        )

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

    # -----------------------------------------------------------------
    # Public thin wrappers
    # -----------------------------------------------------------------

    @classmethod
    def from_gleam(
        cls,
        flux_limit: float = 1.0,
        catalog: str = "gleam_egc",
        brightness_conversion: str = "planck",
        precision: Any = None,
    ) -> "SkyModel":  # noqa: F821
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
        return cls._load_from_vizier_catalog(
            key, flux_limit, brightness_conversion, precision
        )

    @classmethod
    def from_mals(
        cls,
        flux_limit: float = 1.0,
        release: str = "dr2",
        brightness_conversion: str = "planck",
        precision: Any = None,
    ) -> "SkyModel":  # noqa: F821
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
        return cls._load_from_vizier_catalog(
            key, flux_limit, brightness_conversion, precision
        )

    @classmethod
    def from_vlssr(
        cls,
        flux_limit: float = 1.0,
        brightness_conversion: str = "planck",
        precision: Any = None,
    ) -> "SkyModel":  # noqa: F821
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
        return cls._load_from_vizier_catalog(
            "vlssr", flux_limit, brightness_conversion, precision
        )

    @classmethod
    def from_tgss(
        cls,
        flux_limit: float = 0.1,
        brightness_conversion: str = "planck",
        precision: Any = None,
    ) -> "SkyModel":  # noqa: F821
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
        return cls._load_from_vizier_catalog(
            "tgss", flux_limit, brightness_conversion, precision
        )

    @classmethod
    def from_wenss(
        cls,
        flux_limit: float = 0.05,
        brightness_conversion: str = "planck",
        precision: Any = None,
    ) -> "SkyModel":  # noqa: F821
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
        return cls._load_from_vizier_catalog(
            "wenss", flux_limit, brightness_conversion, precision
        )

    @classmethod
    def from_sumss(
        cls,
        flux_limit: float = 0.008,
        brightness_conversion: str = "planck",
        precision: Any = None,
    ) -> "SkyModel":  # noqa: F821
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
        return cls._load_from_vizier_catalog(
            "sumss", flux_limit, brightness_conversion, precision
        )

    @classmethod
    def from_nvss(
        cls,
        flux_limit: float = 0.0025,
        brightness_conversion: str = "planck",
        precision: Any = None,
    ) -> "SkyModel":  # noqa: F821
        """
        Load the NVSS catalog from VizieR (1400 MHz, ~1.8M sources).

        NVSS (Condon et al. 1998) is the NRAO VLA Sky Survey at 1.4 GHz.
        Reference: VIII/65 on VizieR. Warning: the full catalog has ~1.8M
        rows -- consider using a high flux_limit to reduce memory usage.

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
        return cls._load_from_vizier_catalog(
            "nvss", flux_limit, brightness_conversion, precision
        )

    @classmethod
    def from_first(
        cls,
        flux_limit: float = 0.001,
        brightness_conversion: str = "planck",
        precision: Any = None,
    ) -> "SkyModel":  # noqa: F821
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
        return cls._load_from_vizier_catalog(
            "first", flux_limit, brightness_conversion, precision
        )

    @classmethod
    def from_lotss(
        cls,
        release: str = "dr2",
        flux_limit: float = 0.001,
        brightness_conversion: str = "planck",
        precision: Any = None,
    ) -> "SkyModel":  # noqa: F821
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
                f"Unknown LoTSS release '{release}'. Available: 'dr1', 'dr2'."
            )
        return cls._load_from_vizier_catalog(
            key, flux_limit, brightness_conversion, precision
        )

    @classmethod
    def from_at20g(
        cls,
        flux_limit: float = 0.04,
        brightness_conversion: str = "planck",
        precision: Any = None,
    ) -> "SkyModel":  # noqa: F821
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
        return cls._load_from_vizier_catalog(
            "at20g", flux_limit, brightness_conversion, precision
        )

    @classmethod
    def from_3c(
        cls,
        flux_limit: float = 1.0,
        brightness_conversion: str = "planck",
        precision: Any = None,
    ) -> "SkyModel":  # noqa: F821
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
        return cls._load_from_vizier_catalog(
            "3c", flux_limit, brightness_conversion, precision
        )

    @classmethod
    def from_gb6(
        cls,
        flux_limit: float = 0.018,
        brightness_conversion: str = "planck",
        precision: Any = None,
    ) -> "SkyModel":  # noqa: F821
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
        return cls._load_from_vizier_catalog(
            "gb6", flux_limit, brightness_conversion, precision
        )

    @classmethod
    def from_racs(
        cls,
        band: str = "low",
        flux_limit: float = 1.0,
        max_rows: int = 1_000_000,
        brightness_conversion: str = "planck",
        precision: Any = None,
    ) -> "SkyModel":  # noqa: F821
        """
        Load a RACS catalog via CASDA TAP (887.5 / 1367.5 / 1655.5 MHz).

        RACS (McConnell et al. 2020) is the Rapid ASKAP Continuum Survey.
        Data are retrieved via CASDA TAP (astroquery). The column names used
        here are best-effort -- verify against the live CASDA schema if errors
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
                f"Unknown RACS band '{band}'. Available: {sorted(RACS_CATALOGS.keys())}"
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
                ra_val = row[info["ra_col"]]
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

    # -----------------------------------------------------------------
    # Listing helpers
    # -----------------------------------------------------------------

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
        return {
            name: info["description"] for name, info in VIZIER_POINT_CATALOGS.items()
        }

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
