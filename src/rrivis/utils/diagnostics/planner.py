"""Observable strip computation and data assembly for diagnostics.

Computes the sky region visible to a radio telescope during an observation
and collects all data (background map, point sources, beam pattern) needed
for the interactive strip visualisation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Known diffuse model names whose HEALPix maps are already rotated to
# equatorial (ICRS) by the RRIVis loaders — no extra rotation needed.
_DIFFUSE_MODELS = frozenset(
    {
        "gsm2008",
        "gsm2016",
        "lfss",
        "haslam",
        "lfsm",
        "diffuse_sky",
        "gsm",
        "pysm3",
    }
)


@dataclass(frozen=True)
class ObservableStrip:
    """All computed data for an observable-strip diagnostic plot.

    Produced by :meth:`DiagnosticsPlanner.compute` and consumed by
    :class:`~rrivis.utils.diagnostics.strip_plotter.StripPlotter`.
    """

    # --- strip geometry ---
    ra_start_deg: float
    ra_end_deg: float
    dec_lower_deg: float
    dec_upper_deg: float
    latitude_deg: float
    fov_radius_deg: float
    frequency_hz: float

    # --- background map (2-D Cartesian projection, or None) ---
    projected_map: np.ndarray | None

    # --- point sources (capped to max_point_sources, flux-sorted) ---
    source_ra_deg: np.ndarray | None
    source_dec_deg: np.ndarray | None
    source_flux_jy: np.ndarray | None
    in_strip_mask: np.ndarray | None  # bool
    top_n_indices: np.ndarray | None  # int indices into the source arrays

    # --- beam projection ---
    beam_projection: Any | None  # BeamSkyProjection
    beam_rgba: dict | None
    beam_contours: list | None

    # --- metadata ---
    obstime_start_iso: str | None
    obstime_end_iso: str | None
    lst_start_hours: float | None
    lst_end_hours: float | None
    background_mode: str


class DiagnosticsPlanner:
    """Compute the observable sky strip and collect data for plotting.

    Accepts either individual parameters or an ``RRIvisConfig`` object.
    At minimum, provide a location and either an LST range or a UTC start
    time + duration.

    Parameters
    ----------
    latitude_deg, longitude_deg, height_m : float, optional
        Observer location.  Defaults to HERA site.
    lst_start_hours, lst_end_hours : float, optional
        LST range (hours).  Mutually exclusive with *start_time_iso*.
    start_time_iso : str, optional
        UTC start time (ISO-8601).  Requires *duration_seconds*.
    duration_seconds : float, optional
        Observation duration (seconds).
    frequency_mhz : float
        Observing frequency in MHz (default 150).
    fov_radius_deg : float, optional
        Override FOV radius.  If *None*, estimated from *beam_diameter_m*.
    beam_diameter_m : float, optional
        Antenna dish diameter for FOV estimation.
    beam_config : dict, optional
        Full beam configuration dict (aperture_shape, taper, …).
    beam_fits_path : str, optional
        Path to a FITS beam file.
    beam_lst_hours : float, optional
        LST where the beam is centred (default: strip centre).
    beam_vmin_db, beam_vmax_db : float
        Beam colour-scale limits in dB.
    sky_model : SkyModel, optional
        Pre-loaded sky model.
    sky_model_name : str, optional
        Registered loader name (e.g. ``"gsm2008"``, ``"gleam"``).
    sky_model_kwargs : dict, optional
        Extra kwargs forwarded to ``SkyModel.from_catalog``.
    max_point_sources : int
        Maximum point sources to include (brightest first, default 1000).
    top_n_sources : int
        Number of brightest in-strip sources to highlight (default 5).
    background_mode : str
        ``"gsm"`` | ``"reference"`` | ``"none"`` (default ``"gsm"``).
    config : RRIvisConfig, optional
        Full RRIVis configuration — overrides individual params where set.
    """

    def __init__(
        self,
        *,
        # Location
        latitude_deg: float | None = None,
        longitude_deg: float | None = None,
        height_m: float | None = None,
        # Time — LST mode
        lst_start_hours: float | None = None,
        lst_end_hours: float | None = None,
        # Time — UTC mode
        start_time_iso: str | None = None,
        duration_seconds: float | None = None,
        # Frequency
        frequency_mhz: float = 150.0,
        # Beam / FOV
        fov_radius_deg: float | None = None,
        beam_diameter_m: float | None = None,
        beam_config: dict | None = None,
        beam_fits_path: str | None = None,
        beam_lst_hours: float | None = None,
        beam_vmin_db: float = -40.0,
        beam_vmax_db: float = 0.0,
        # Sky
        sky_model: Any | None = None,
        sky_model_name: str | None = None,
        sky_model_kwargs: dict | None = None,
        max_point_sources: int = 1000,
        top_n_sources: int = 5,
        # Display
        background_mode: str = "gsm",
        # Config shortcut
        config: Any | None = None,
    ):
        # --- resolve from config if given ---
        if config is not None:
            cfg = config if isinstance(config, dict) else config.model_dump()
            loc = cfg.get("location", {})
            latitude_deg = latitude_deg or float(loc.get("lat", -30.72))
            longitude_deg = longitude_deg or float(loc.get("lon", 21.43))
            height_m = height_m or float(loc.get("height", 1073.0))

            obs_t = cfg.get("obs_time", {})
            if start_time_iso is None and obs_t.get("start_time"):
                start_time_iso = obs_t["start_time"]
            if duration_seconds is None and obs_t.get("duration_seconds"):
                duration_seconds = obs_t["duration_seconds"]

            obs_f = cfg.get("obs_frequency", {})
            if obs_f.get("starting_frequency"):
                unit = obs_f.get("frequency_unit", "MHz")
                _mult = {"Hz": 1e-6, "kHz": 1e-3, "MHz": 1.0, "GHz": 1e3}
                frequency_mhz = float(obs_f["starting_frequency"]) * _mult.get(
                    unit, 1.0
                )

            beams = cfg.get("beams", {})
            if beam_config is None and beams:
                beam_config = beams
            if beam_fits_path is None and beams.get("beam_file"):
                beam_fits_path = beams["beam_file"]

        self.latitude_deg = latitude_deg if latitude_deg is not None else -30.72
        self.longitude_deg = longitude_deg if longitude_deg is not None else 21.43
        self.height_m = height_m if height_m is not None else 1073.0

        self.lst_start_hours = lst_start_hours
        self.lst_end_hours = lst_end_hours
        self.start_time_iso = start_time_iso
        self.duration_seconds = duration_seconds

        self.frequency_mhz = frequency_mhz
        self.frequency_hz = frequency_mhz * 1e6
        self.fov_radius_deg = fov_radius_deg
        self.beam_diameter_m = beam_diameter_m
        self.beam_config = beam_config or {}
        self.beam_fits_path = beam_fits_path
        self.beam_lst_hours = beam_lst_hours
        self.beam_vmin_db = beam_vmin_db
        self.beam_vmax_db = beam_vmax_db

        self.sky_model = sky_model
        self.sky_model_name = sky_model_name
        self.sky_model_kwargs = sky_model_kwargs or {}
        self.max_point_sources = max_point_sources
        self.top_n_sources = top_n_sources
        self.background_mode = background_mode

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def compute(self) -> ObservableStrip:
        """Compute the observable strip and assemble all plot data."""
        import astropy.units as u
        from astropy.coordinates import EarthLocation
        from astropy.time import Time, TimeDelta

        # 1. Location
        location = EarthLocation(
            lat=self.latitude_deg * u.deg,
            lon=self.longitude_deg * u.deg,
            height=self.height_m * u.m,
        )

        # 2. FOV radius
        fov_radius = self._resolve_fov_radius()

        # 3. RA range + time metadata
        obstime_start_iso = None
        obstime_end_iso = None
        lst_start = self.lst_start_hours
        lst_end = self.lst_end_hours

        if self.lst_start_hours is not None and self.lst_end_hours is not None:
            ra_start = self.lst_start_hours * 15.0
            ra_end = self.lst_end_hours * 15.0
        elif self.start_time_iso is not None and self.duration_seconds is not None:
            t_start = Time(self.start_time_iso)
            t_end = t_start + TimeDelta(self.duration_seconds, format="sec")
            ra_start = self._zenith_ra(location, t_start)
            ra_end = self._zenith_ra(location, t_end)
            obstime_start_iso = t_start.iso
            obstime_end_iso = t_end.iso
            lst_start = ra_start / 15.0
            lst_end = ra_end / 15.0
        else:
            raise ValueError(
                "Provide either (lst_start_hours, lst_end_hours) "
                "or (start_time_iso, duration_seconds)."
            )

        # 4. Dec bounds
        dec_lower = self.latitude_deg - fov_radius
        dec_upper = self.latitude_deg + fov_radius

        # 5. Load sky model
        sky = self._resolve_sky_model()

        # 6. Background projection
        projected_map = self._project_background(sky)

        # 7. Point sources
        src_ra, src_dec, src_flux = self._extract_point_sources(sky)

        # 8. In-strip filtering
        in_strip_mask, top_n_idx = self._filter_sources(
            src_ra, src_dec, src_flux, ra_start, ra_end, dec_lower, dec_upper
        )

        # 9. Beam projection
        beam_proj, beam_rgba, beam_contours = self._compute_beam(
            location, ra_start, ra_end
        )

        return ObservableStrip(
            ra_start_deg=ra_start,
            ra_end_deg=ra_end,
            dec_lower_deg=dec_lower,
            dec_upper_deg=dec_upper,
            latitude_deg=self.latitude_deg,
            fov_radius_deg=fov_radius,
            frequency_hz=self.frequency_hz,
            projected_map=projected_map,
            source_ra_deg=src_ra,
            source_dec_deg=src_dec,
            source_flux_jy=src_flux,
            in_strip_mask=in_strip_mask,
            top_n_indices=top_n_idx,
            beam_projection=beam_proj,
            beam_rgba=beam_rgba,
            beam_contours=beam_contours,
            obstime_start_iso=obstime_start_iso,
            obstime_end_iso=obstime_end_iso,
            lst_start_hours=lst_start,
            lst_end_hours=lst_end,
            background_mode=self.background_mode,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _zenith_ra(location, obstime) -> float:
        """RA of the zenith at a given time and location (degrees)."""
        import astropy.units as u
        from astropy.coordinates import AltAz, SkyCoord

        zenith = SkyCoord(
            alt=90 * u.deg,
            az=0 * u.deg,
            frame=AltAz(obstime=obstime, location=location),
        )
        return zenith.transform_to("icrs").ra.deg

    def _resolve_fov_radius(self) -> float:
        """Return FOV radius in degrees."""
        if self.fov_radius_deg is not None:
            return self.fov_radius_deg

        if self.beam_diameter_m is not None:
            # Diffraction limit: HPBW ≈ 1.22 λ / D
            c = 299_792_458.0
            wavelength = c / self.frequency_hz
            hpbw_rad = 1.22 * wavelength / self.beam_diameter_m
            fov = np.degrees(hpbw_rad) / 2.0
            logger.info(
                "FOV radius from dish diameter %.1f m: %.1f deg "
                "(HPBW %.1f deg at %.1f MHz)",
                self.beam_diameter_m,
                fov,
                np.degrees(hpbw_rad),
                self.frequency_mhz,
            )
            return fov

        # Sensible default for a ~14 m dish at the given frequency
        c = 299_792_458.0
        wavelength = c / self.frequency_hz
        hpbw_rad = 1.22 * wavelength / 14.0
        fov = np.degrees(hpbw_rad) / 2.0
        logger.warning(
            "No FOV radius or dish diameter given; defaulting to 14 m "
            "dish estimate (%.1f deg at %.1f MHz).",
            fov,
            self.frequency_mhz,
        )
        return fov

    def _resolve_sky_model(self):
        """Return a SkyModel, loading one if necessary."""
        if self.sky_model is not None:
            return self.sky_model

        if self.sky_model_name is not None:
            from rrivis.core.sky.model import SkyModel

            logger.info("Loading sky model '%s' ...", self.sky_model_name)
            return SkyModel.from_catalog(self.sky_model_name, **self.sky_model_kwargs)

        return None

    def _project_background(self, sky) -> np.ndarray | None:
        """Project a HEALPix sky model to 2-D Cartesian for Bokeh."""
        if self.background_mode == "none":
            return None
        if sky is None:
            return None

        from rrivis.core.sky.model import SkyFormat

        if sky.mode != SkyFormat.HEALPIX:
            return None

        import healpy as hp
        import matplotlib.pyplot as plt

        # Pick the frequency channel closest to the observing frequency
        freq_idx = sky.resolve_frequency_index(self.frequency_hz)
        healpix_map = sky.healpix_maps[freq_idx]

        projected = hp.cartview(
            healpix_map,
            xsize=4000,
            norm="hist",
            coord="C",
            flip="astro",
            title="",
            unit="Brightness",
            return_projected_map=True,
            notext=True,
        )
        plt.close()
        return projected

    def _extract_point_sources(self, sky):
        """Extract point-source arrays, capped and flux-sorted."""
        if sky is None:
            return None, None, None

        from rrivis.core.sky.model import SkyFormat

        if sky.mode != SkyFormat.POINT_SOURCES:
            return None, None, None

        ra_rad = sky.ra_rad
        dec_rad = sky.dec_rad
        flux = sky.flux

        if ra_rad is None or len(ra_rad) == 0:
            return None, None, None

        # Sort by flux descending, cap
        order = np.argsort(flux)[::-1]
        n = min(len(order), self.max_point_sources)
        order = order[:n]

        ra_deg = np.degrees(ra_rad[order])
        dec_deg = np.degrees(dec_rad[order])
        flux_jy = flux[order].copy()

        return ra_deg, dec_deg, flux_jy

    def _filter_sources(
        self, ra_deg, dec_deg, flux, ra_start, ra_end, dec_lower, dec_upper
    ):
        """Return (in_strip_mask, top_n_indices)."""
        if ra_deg is None:
            return None, None

        from rrivis.core.sky.region import SkyRegion

        # BoxRegion expects RA in [0, 360] for centre
        ra_center = (ra_start + ra_end) / 2.0
        if ra_start > ra_end:
            # Wrapping: average in [0, 360] space
            s = ra_start if ra_start >= 0 else ra_start + 360
            e = ra_end if ra_end >= 0 else ra_end + 360
            if e < s:
                e += 360
            ra_center = ((s + e) / 2.0) % 360

        width = ra_end - ra_start
        if width < 0:
            width += 360.0
        height = dec_upper - dec_lower

        region = SkyRegion.box(
            ra_deg=ra_center % 360,
            dec_deg=(dec_lower + dec_upper) / 2.0,
            width_deg=width,
            height_deg=height,
        )

        # BoxRegion.contains expects radians
        ra_rad = np.deg2rad(ra_deg % 360)
        dec_rad = np.deg2rad(dec_deg)
        mask = region.contains(ra_rad, dec_rad)

        # Top-N brightest in-strip
        in_strip_fluxes = np.where(mask, flux, -np.inf)
        n = min(self.top_n_sources, int(mask.sum()))
        top_n = np.argsort(in_strip_fluxes)[::-1][:n]

        return mask, top_n

    def _compute_beam(self, location, ra_start, ra_end):
        """Compute beam sky projection if a beam is configured."""
        if self.beam_fits_path is None and not self.beam_config.get("beam_mode"):
            return None, None, None

        from rrivis.core.jones.beam.projection import (
            compute_beam_power_on_radec_grid,
            create_rgba_overlay,
            extract_contours,
        )

        # Zenith position for beam centre
        if self.beam_lst_hours is not None:
            zenith_ra = self.beam_lst_hours * 15.0
        else:
            zenith_ra = (ra_start + ra_end) / 2.0
            if ra_start > ra_end:
                s = ra_start if ra_start >= 0 else ra_start + 360
                e = ra_end if ra_end >= 0 else ra_end + 360
                if e < s:
                    e += 360
                zenith_ra = (s + e) / 2.0
                if zenith_ra > 180:
                    zenith_ra -= 360
        zenith_dec = self.latitude_deg

        beam_power_func = self._build_beam_power_func()
        if beam_power_func is None:
            return None, None, None

        proj = compute_beam_power_on_radec_grid(
            beam_power_func,
            zenith_ra_deg=zenith_ra,
            zenith_dec_deg=zenith_dec,
            max_za_deg=90.0,
            ra_resolution_deg=0.25,
            dec_resolution_deg=0.25,
        )
        rgba = create_rgba_overlay(
            proj,
            cmap="RdBu_r",
            vmin_db=self.beam_vmin_db,
            vmax_db=self.beam_vmax_db,
            alpha_scale=0.7,
        )
        contours = extract_contours(proj, levels_db=[-3.0, -10.0])

        return proj, rgba, contours

    def _build_beam_power_func(self):
        """Return a ``(za_rad, az_rad) -> power`` callable, or None."""
        if self.beam_fits_path is not None:
            return self._fits_beam_power_func()

        if self.beam_config.get("beam_mode") == "analytic" and self.beam_diameter_m:
            return self._analytic_beam_power_func()

        return None

    def _analytic_beam_power_func(self):
        """Build power function from RRIVis analytic beam."""
        from rrivis.core.jones.beam.analytic import compute_aperture_beam

        diameter = self.beam_diameter_m
        frequency_hz = self.frequency_hz
        cfg = self.beam_config

        aperture_shape = cfg.get("aperture_shape", "circular")
        taper = cfg.get("taper", "gaussian")
        edge_taper_dB = cfg.get("edge_taper_dB", 10.0)
        feed_model = cfg.get("feed_model", "none")
        feed_computation = cfg.get("feed_computation", "analytical")
        feed_params = cfg.get("feed_params", None)
        reflector_type = cfg.get("reflector_type", "prime_focus")
        magnification = cfg.get("magnification", 1.0)
        aperture_params = cfg.get("aperture_params", None)

        def power_func(za_rad: np.ndarray, az_rad: np.ndarray) -> np.ndarray:
            flat_za = za_rad.ravel()
            flat_az = az_rad.ravel()
            jones = compute_aperture_beam(
                flat_za,
                flat_az,
                frequency=frequency_hz,
                diameter=diameter,
                aperture_shape=aperture_shape,
                taper=taper,
                edge_taper_dB=edge_taper_dB,
                feed_model=feed_model,
                feed_computation=feed_computation,
                feed_params=feed_params,
                reflector_type=reflector_type,
                magnification=magnification,
                aperture_params=aperture_params,
            )
            # jones shape: (N, 2, 2) — power = |J00|^2 + |J11|^2
            power = np.abs(jones[:, 0, 0]) ** 2 + np.abs(jones[:, 1, 1]) ** 2
            peak = power.max()
            if peak > 0:
                power = power / peak
            return power.reshape(za_rad.shape)

        return power_func

    def _fits_beam_power_func(self):
        """Build power function from a FITS beam file via interpolation."""
        from scipy.interpolate import RegularGridInterpolator

        try:
            from pyuvdata import UVBeam
        except ImportError:
            logger.warning("pyuvdata not installed — cannot load FITS beam.")
            return None

        beam = UVBeam()
        beam.read_beamfits(self.beam_fits_path)

        # Select frequency
        freq_array_mhz = beam.freq_array / 1e6
        freq_idx = int(np.argmin(np.abs(freq_array_mhz - self.frequency_mhz)))

        az_array = beam.axis1_array  # 0 to 2*pi
        za_array = beam.axis2_array  # 0 to pi

        # Compute power
        if beam.beam_type == "efield":
            e_theta = beam.data_array[0, 0, freq_idx, :, :]
            e_phi = beam.data_array[1, 0, freq_idx, :, :]
            power = np.abs(e_theta) ** 2 + np.abs(e_phi) ** 2
        else:
            power = beam.data_array[0, 0, freq_idx, :, :]

        peak = np.nanmax(power)
        if peak > 0:
            power = power / peak

        # Handle azimuth wrap: append az = 2*pi (copy of az = 0)
        az_ext = np.append(az_array, 2.0 * np.pi)
        power_ext = np.column_stack([power, power[:, 0]])

        interpolator = RegularGridInterpolator(
            (za_array, az_ext),
            power_ext,
            method="linear",
            bounds_error=False,
            fill_value=0.0,
        )

        logger.info(
            "Loaded FITS beam %s at %.1f MHz (index %d)",
            self.beam_fits_path,
            freq_array_mhz[freq_idx],
            freq_idx,
        )

        def power_func(za_rad: np.ndarray, az_rad: np.ndarray) -> np.ndarray:
            pts = np.column_stack([za_rad.ravel(), az_rad.ravel()])
            return interpolator(pts).reshape(za_rad.shape)

        return power_func
