# rrivis/core/sky/_loaders_vizier.py
"""VizieR and CASDA TAP loader mixin for SkyModel."""

import functools
import logging
from typing import Any

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord
from astroquery.utils.tap.core import TapPlus
from astroquery.vizier import Vizier

from .catalogs import CASDA_TAP_URL, RACS_CATALOGS, VIZIER_POINT_CATALOGS

logger = logging.getLogger(__name__)


def _require_service(service: str, action: str) -> None:
    """Check network connectivity for *service* and raise if unavailable.

    Parameters
    ----------
    service : str
        Service name recognised by ``rrivis.utils.network`` (e.g.
        ``"vizier"``, ``"casda"``).
    action : str
        Human-readable description of the operation that needs the
        service (used in the error message).

    Raises
    ------
    ConnectionError
        If general internet or the specific service is unreachable.
    """
    from rrivis.utils.network import (
        SERVICE_DISPLAY_NAMES,
        check_service,
        is_online,
    )

    display = SERVICE_DISPLAY_NAMES.get(service, service)

    if not is_online():
        raise ConnectionError(
            f"No internet connection. Cannot {action}.\n"
            f"Hint: use offline metadata methods like "
            f"SkyModel.get_catalog_info(key) or SkyModel.list_point_catalogs() "
            f"which work without network."
        )

    if not check_service(service):
        raise ConnectionError(
            f"{display} ({service}) is unreachable. Cannot {action}.\n"
            f"The service may be temporarily down. Try again later, or use "
            f"SkyModel.get_catalog_info(key) for offline metadata."
        )


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

        _require_service("vizier", f"download catalog '{catalog_key}' from VizieR")
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

        if info.get("spindex_col") and info["spindex_col"] in catalog.colnames:
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
            ``"gleam_gal"``.

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
            "VIII/110/catalog": "gleam_x_dr1",
        }
        key = _LEGACY_GLEAM_IDS.get(catalog, catalog)
        _gleam_keys = sorted(k for k in VIZIER_POINT_CATALOGS if k.startswith("gleam"))
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
    def from_vlass(
        cls,
        flux_limit: float = 0.001,
        brightness_conversion: str = "planck",
        precision: Any = None,
    ) -> "SkyModel":  # noqa: F821
        """
        Load the VLASS Quick Look catalog from VizieR (3000 MHz, ~1.9M sources).

        VLASS (Lacy et al. 2020) is the VLA Sky Survey at 2-4 GHz (S-band).
        The Quick Look Epoch 1 catalog (Gordon et al. 2021) covers 33,885 deg^2
        at 2.5 arcsec resolution. Reference: J/ApJS/255/30 on VizieR.

        Note: Quick Look flux densities may be systematically underestimated
        by approximately 15%.

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
            "vlass", flux_limit, brightness_conversion, precision
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

        _require_service("casda", f"download RACS-{band} catalog from CASDA")

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
            _ra_rad=cls._deg_to_rad_at_precision(
                np.array(ra_list, dtype=np.float64), precision
            ),
            _dec_rad=cls._deg_to_rad_at_precision(
                np.array(dec_list, dtype=np.float64), precision
            ),
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

    @staticmethod
    def get_point_catalog_metadata(catalog_key: str) -> dict[str, Any]:
        """Get static metadata for a VizieR point-source catalog (no network).

        Returns all locally-stored metadata for the catalog without
        making any network calls. For live column queries from VizieR,
        use ``get_catalog_columns()`` instead.

        Parameters
        ----------
        catalog_key : str
            Key in ``VIZIER_POINT_CATALOGS`` (e.g. ``"gleam_egc"``, ``"nvss"``).

        Returns
        -------
        dict
            Keys:

            - ``"vizier_id"`` : str — VizieR catalog identifier
            - ``"description"`` : str — catalog description
            - ``"freq_mhz"`` : float — survey reference frequency
            - ``"flux_col"`` : str — flux column name
            - ``"flux_unit"`` : str — unit of the flux column
            - ``"ra_col"`` : str — RA column name
            - ``"dec_col"`` : str — Dec column name
            - ``"spindex_col"`` : str or None — spectral index column
            - ``"default_spindex"`` : float — default spectral index
            - ``"coord_frame"`` : str — coordinate frame (e.g. ``"icrs"``, ``"fk4"``)

        Raises
        ------
        ValueError
            If ``catalog_key`` is not recognized.

        Examples
        --------
        >>> info = SkyModel.get_point_catalog_metadata("nvss")
        >>> print(info["freq_mhz"])
        1400.0
        >>> print(info["flux_col"], info["flux_unit"])
        S1.4 mJy
        """
        if catalog_key not in VIZIER_POINT_CATALOGS:
            raise ValueError(
                f"Unknown catalog key '{catalog_key}'. "
                f"Available: {sorted(VIZIER_POINT_CATALOGS.keys())}"
            )
        info = VIZIER_POINT_CATALOGS[catalog_key]
        return {
            "vizier_id": info["vizier_id"],
            "description": info["description"],
            "freq_mhz": info["freq_mhz"],
            "flux_col": info["flux_col"],
            "flux_unit": info.get("flux_unit", "Jy"),
            "ra_col": info["ra_col"],
            "dec_col": info["dec_col"],
            "spindex_col": info.get("spindex_col"),
            "default_spindex": info["default_spindex"],
            "coord_frame": info.get("coord_frame", "icrs"),
        }

    @staticmethod
    def get_racs_metadata(band: str) -> dict[str, Any]:
        """Get static metadata for a RACS catalog band (no network).

        Returns all locally-stored metadata for the RACS band without
        making any network calls. For live column queries from CASDA TAP,
        use ``get_racs_columns()`` instead.

        Parameters
        ----------
        band : str
            RACS band: ``"low"``, ``"mid"``, or ``"high"``.

        Returns
        -------
        dict
            Keys:

            - ``"description"`` : str — band description
            - ``"freq_mhz"`` : float — survey frequency
            - ``"tap_table"`` : str — CASDA TAP table name
            - ``"ra_col"`` : str — RA column name
            - ``"dec_col"`` : str — Dec column name
            - ``"flux_col"`` : str — flux column name
            - ``"flux_unit"`` : str — unit of the flux column

        Raises
        ------
        ValueError
            If ``band`` is not recognized.

        Examples
        --------
        >>> info = SkyModel.get_racs_metadata("low")
        >>> print(info["freq_mhz"])
        887.5
        """
        band = band.lower()
        if band not in RACS_CATALOGS:
            raise ValueError(
                f"Unknown RACS band '{band}'. Available: {sorted(RACS_CATALOGS.keys())}"
            )
        info = RACS_CATALOGS[band]
        return {
            "description": info["description"],
            "freq_mhz": info["freq_mhz"],
            "tap_table": info["tap_table"],
            "ra_col": info["ra_col"],
            "dec_col": info["dec_col"],
            "flux_col": info["flux_col"],
            "flux_unit": info.get("flux_unit", "mJy"),
        }

    @staticmethod
    @functools.lru_cache(maxsize=32)
    def get_catalog_columns(catalog_key: str) -> dict[str, Any]:
        """Query VizieR for all available columns in a point-source catalog.

        Fetches one row from VizieR and returns the full list of column names
        along with metadata about which columns RRIVis uses. Column
        descriptions and units are extracted from VizieR's own metadata.

        Parameters
        ----------
        catalog_key : str
            Key in ``VIZIER_POINT_CATALOGS`` (e.g. ``"gleam_egc"``, ``"nvss"``).

        Returns
        -------
        dict
            Keys:

            - ``"columns"`` : list[str] — all column names in the catalog
            - ``"column_details"`` : dict[str, dict] — per-column metadata
              from VizieR with ``"description"`` and ``"unit"`` keys
            - ``"used_by_rrivis"`` : dict[str, str | None] — columns used by
              RRIVis (``"ra"``, ``"dec"``, ``"flux"``, ``"spectral_index"``)
            - ``"vizier_id"`` : str — VizieR catalog identifier
            - ``"freq_mhz"`` : float — survey reference frequency
            - ``"flux_unit"`` : str — unit of the flux column
            - ``"description"`` : str — catalog description

            If the query fails, the dict contains an ``"error"`` key instead.

        Examples
        --------
        >>> info = SkyModel.get_catalog_columns("nvss")
        >>> print(info["columns"][:5])
        ['recno', 'Field', 'Xpos', 'Ypos', 'NVSS']
        >>> print(info["used_by_rrivis"])
        {'ra': 'RAJ2000', 'dec': 'DEJ2000', 'flux': 'S1.4', 'spectral_index': None}
        >>> print(info["column_details"]["S1.4"])
        {'description': '...', 'unit': 'mJy'}
        """
        if catalog_key not in VIZIER_POINT_CATALOGS:
            raise ValueError(
                f"Unknown catalog key '{catalog_key}'. "
                f"Available: {sorted(VIZIER_POINT_CATALOGS.keys())}"
            )

        info = VIZIER_POINT_CATALOGS[catalog_key]

        _require_service(
            "vizier", f"query live columns for '{catalog_key}' from VizieR"
        )

        try:
            v = Vizier(columns=["**"], row_limit=1)
            tables = v.get_catalogs(info["vizier_id"])
            if not tables:
                return {"error": f"No tables returned from VizieR for '{catalog_key}'"}

            catalog = None
            if info.get("table") is not None:
                for t in tables:
                    if info["table"] in t.meta.get("name", ""):
                        catalog = t
                        break
            if catalog is None:
                catalog = tables[0]

            columns = list(catalog.colnames)

            column_details = {}
            for col_name in columns:
                col = catalog.columns[col_name]
                column_details[col_name] = {
                    "description": getattr(col, "description", None) or "",
                    "unit": str(col.unit) if col.unit else None,
                }
        except Exception as e:
            logger.error(f"Failed to query VizieR columns for {catalog_key}: {e}")
            return {"error": str(e)}

        return {
            "columns": columns,
            "column_details": column_details,
            "used_by_rrivis": {
                "ra": info["ra_col"],
                "dec": info["dec_col"],
                "flux": info["flux_col"],
                "spectral_index": info.get("spindex_col"),
            },
            "vizier_id": info["vizier_id"],
            "freq_mhz": info["freq_mhz"],
            "flux_unit": info.get("flux_unit", "Jy"),
            "description": info["description"],
        }

    @staticmethod
    @functools.lru_cache(maxsize=8)
    def get_racs_columns(band: str) -> dict[str, Any]:
        """Query CASDA TAP for available columns in a RACS catalog.

        Parameters
        ----------
        band : str
            RACS band: ``"low"``, ``"mid"``, or ``"high"``.

        Returns
        -------
        dict
            Keys:

            - ``"columns"`` : list[str] — all column names in the TAP table
            - ``"used_by_rrivis"`` : dict[str, str] — columns used by RRIVis
            - ``"tap_table"`` : str — CASDA TAP table name
            - ``"freq_mhz"`` : float — survey frequency
            - ``"description"`` : str — band description

            If the query fails, the dict contains an ``"error"`` key instead.

        Examples
        --------
        >>> info = SkyModel.get_racs_columns("low")
        >>> print(info["freq_mhz"])
        887.5
        """
        band = band.lower()
        if band not in RACS_CATALOGS:
            raise ValueError(
                f"Unknown RACS band '{band}'. Available: {sorted(RACS_CATALOGS.keys())}"
            )

        info = RACS_CATALOGS[band]

        _require_service("casda", f"query live columns for RACS-{band} from CASDA")

        try:
            tap = TapPlus(url=CASDA_TAP_URL)
            job = tap.launch_job(
                f"SELECT column_name, description, unit "
                f"FROM tap_schema.columns "
                f"WHERE table_name='{info['tap_table']}'"
            )
            result = job.get_results()
            columns = list(result["column_name"])

            column_details = {}
            for row in result:
                col_name = row["column_name"]
                column_details[col_name] = {
                    "description": str(row["description"])
                    if row["description"]
                    else "",
                    "unit": str(row["unit"]) if row["unit"] else None,
                }
        except Exception as e:
            logger.error(f"Failed to query CASDA TAP columns for RACS-{band}: {e}")
            return {"error": str(e)}

        return {
            "columns": columns,
            "column_details": column_details,
            "used_by_rrivis": {
                "ra": info["ra_col"],
                "dec": info["dec_col"],
                "flux": info["flux_col"],
            },
            "tap_table": info["tap_table"],
            "freq_mhz": info["freq_mhz"],
            "description": info["description"],
        }
