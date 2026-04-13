"""Sky-visibility planning and data assembly."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from .geometry import (
    SIDEREAL_DEG_PER_SECOND,
    angular_separation_deg,
    axis_from_ra_deg,
    circular_interval_width_deg,
    compute_beam_power_on_full_sky_grid,
    extract_contour_segments,
    normalize_ra_deg,
    unwrap_interval_end_deg,
    wrap_ra_deg,
)

logger = logging.getLogger(__name__)

AxisType = Literal["ra", "lst"]
BackgroundLayer = Literal["diffuse", "none"]
FootprintModel = Literal["swept_beam", "rectangular_approx"]
DisplayMode = Literal["summary", "snapshots"]


@dataclass(frozen=True)
class VisibilitySnapshot:
    """One instantaneous sky-visibility snapshot."""

    label: str
    utc_iso: str | None
    lst_hours: float
    zenith_ra_deg: float
    zenith_dec_deg: float
    footprint_mask: np.ndarray
    visible_source_mask: np.ndarray | None


@dataclass(frozen=True)
class VisibilitySourceMetrics:
    """Point-source positions plus visibility metrics."""

    ra_deg: np.ndarray
    dec_deg: np.ndarray
    flux_jy: np.ndarray
    x_coord: np.ndarray
    source_name: np.ndarray | None
    visible_any: np.ndarray
    visible_fraction: np.ndarray
    min_separation_deg: np.ndarray
    first_visible_index: np.ndarray
    last_visible_index: np.ndarray
    top_visible_indices: np.ndarray
    nearby_indices: np.ndarray


@dataclass(frozen=True)
class SkyVisibilityPlan:
    """Renderer-neutral sky-visibility description."""

    x_axis: AxisType
    mode: DisplayMode
    title: str
    frequency_hz: float
    field_radius_deg: float
    latitude_deg: float
    longitude_deg: float
    observation_start_iso: str | None
    observation_end_iso: str | None
    lst_start_hours: float | None
    lst_end_hours: float | None
    track_labels: tuple[str, ...]
    track_time_isos: tuple[str | None, ...]
    track_lst_hours: np.ndarray
    track_ra_deg: np.ndarray
    ra_grid_deg: np.ndarray
    dec_grid_deg: np.ndarray
    background_layer: BackgroundLayer
    projected_background: np.ndarray | None
    footprint_model: FootprintModel
    footprint_mask: np.ndarray
    footprint_contours: tuple[tuple[np.ndarray, ...], ...]
    snapshots: tuple[VisibilitySnapshot, ...]
    source_metrics: VisibilitySourceMetrics | None
    beam_projection: Any | None
    beam_contours: list[tuple[list[np.ndarray], float]] | None
    beam_reference_ra_deg: float | None


@dataclass(frozen=True)
class _TrackSamples:
    labels: tuple[str, ...]
    time_isos: tuple[str | None, ...]
    lst_hours: np.ndarray
    ra_deg: np.ndarray
    raw_ra_deg: np.ndarray


class SkyVisibilityPlanner:
    """Compute a reusable sky-visibility plan."""

    def __init__(
        self,
        *,
        latitude_deg: float | None = None,
        longitude_deg: float | None = None,
        height_m: float | None = None,
        lst_start_hours: float | None = None,
        lst_end_hours: float | None = None,
        start_time_iso: str | None = None,
        duration_seconds: float | None = None,
        frequency_mhz: float = 150.0,
        field_radius_deg: float | None = None,
        beam_diameter_m: float | None = None,
        beam_config: dict | None = None,
        beam_fits_path: str | None = None,
        beam_reference: Literal["midpoint", "start", "end"] | float = "midpoint",
        beam_vmin_db: float = -40.0,
        beam_vmax_db: float = 0.0,
        sky_model: Any | None = None,
        sky_model_name: str | None = None,
        sky_model_kwargs: dict | None = None,
        x_axis: AxisType = "ra",
        background_layer: BackgroundLayer = "diffuse",
        footprint_model: FootprintModel = "swept_beam",
        mode: DisplayMode = "summary",
        snapshot_step_seconds: float = 3600.0,
        footprint_step_seconds: float = 60.0,
        grid_resolution_deg: float = 1.0,
        max_point_sources: int = 1000,
        top_n_sources: int = 5,
        nearby_source_count: int = 3,
        nearby_buffer_deg: float = 10.0,
        include_source_metrics: bool = True,
        config: Any | None = None,
    ):
        if config is not None:
            cfg = config if isinstance(config, dict) else config.model_dump()
            loc = cfg.get("location", {})
            latitude_deg = (
                latitude_deg
                if latitude_deg is not None
                else float(loc.get("lat", -30.72))
            )
            longitude_deg = (
                longitude_deg
                if longitude_deg is not None
                else float(loc.get("lon", 21.43))
            )
            height_m = (
                height_m if height_m is not None else float(loc.get("height", 1073.0))
            )

            obs_t = cfg.get("obs_time", {})
            if start_time_iso is None and obs_t.get("start_time"):
                start_time_iso = obs_t["start_time"]
            if duration_seconds is None and obs_t.get("duration_seconds"):
                duration_seconds = obs_t["duration_seconds"]

            obs_f = cfg.get("obs_frequency", {})
            if obs_f.get("starting_frequency"):
                unit = obs_f.get("frequency_unit", "MHz")
                mult = {"Hz": 1e-6, "kHz": 1e-3, "MHz": 1.0, "GHz": 1e3}
                frequency_mhz = float(obs_f["starting_frequency"]) * mult.get(unit, 1.0)

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
        self.field_radius_deg = field_radius_deg
        self.beam_diameter_m = beam_diameter_m
        self.beam_config = beam_config or {}
        self.beam_fits_path = beam_fits_path
        self.beam_reference = beam_reference
        self.beam_vmin_db = beam_vmin_db
        self.beam_vmax_db = beam_vmax_db

        self.sky_model = sky_model
        self.sky_model_name = sky_model_name
        self.sky_model_kwargs = sky_model_kwargs or {}

        self.x_axis = x_axis
        self.background_layer = background_layer
        self.footprint_model = footprint_model
        self.mode = mode
        self.snapshot_step_seconds = snapshot_step_seconds
        self.footprint_step_seconds = footprint_step_seconds
        self.grid_resolution_deg = grid_resolution_deg

        self.max_point_sources = max_point_sources
        self.top_n_sources = top_n_sources
        self.nearby_source_count = nearby_source_count
        self.nearby_buffer_deg = nearby_buffer_deg
        self.include_source_metrics = include_source_metrics

    def build(self) -> SkyVisibilityPlan:
        """Build the sky-visibility plan."""
        import astropy.units as u
        from astropy.coordinates import EarthLocation

        location = EarthLocation(
            lat=self.latitude_deg * u.deg,
            lon=self.longitude_deg * u.deg,
            height=self.height_m * u.m,
        )
        field_radius_deg = self._resolve_field_radius_deg()
        ra_grid_deg = np.arange(
            -180.0, 180.0 + self.grid_resolution_deg, self.grid_resolution_deg
        )
        dec_grid_deg = np.arange(
            -90.0, 90.0 + self.grid_resolution_deg, self.grid_resolution_deg
        )

        track = self._build_track_samples(location, self.footprint_step_seconds)
        snapshots_track = self._build_track_samples(
            location, self.snapshot_step_seconds
        )

        sky = self._resolve_sky_model()
        projected_background = self._project_background(sky)

        source_metrics, snapshot_visibility = self._build_source_metrics(
            sky=sky,
            track=track,
            snapshots=snapshots_track,
            field_radius_deg=field_radius_deg,
        )

        footprint_mask = self._build_footprint_mask(
            ra_grid_deg=ra_grid_deg,
            dec_grid_deg=dec_grid_deg,
            track=track,
            field_radius_deg=field_radius_deg,
        )
        footprint_contours = extract_contour_segments(
            ra_grid_deg,
            dec_grid_deg,
            footprint_mask.astype(float),
            levels=[0.5],
        )

        snapshots = self._build_snapshots(
            snapshots_track=snapshots_track,
            source_visibility=snapshot_visibility,
            ra_grid_deg=ra_grid_deg,
            dec_grid_deg=dec_grid_deg,
            field_radius_deg=field_radius_deg,
        )

        beam_projection, beam_contours, beam_reference_ra_deg = (
            self._build_beam_projection(
                ra_grid_deg=ra_grid_deg,
                dec_grid_deg=dec_grid_deg,
                track=track,
            )
        )

        return SkyVisibilityPlan(
            x_axis=self.x_axis,
            mode=self.mode,
            title=self._build_title(track),
            frequency_hz=self.frequency_hz,
            field_radius_deg=field_radius_deg,
            latitude_deg=self.latitude_deg,
            longitude_deg=self.longitude_deg,
            observation_start_iso=track.time_isos[0],
            observation_end_iso=track.time_isos[-1],
            lst_start_hours=float(track.lst_hours[0]),
            lst_end_hours=float(track.lst_hours[-1]),
            track_labels=track.labels,
            track_time_isos=track.time_isos,
            track_lst_hours=track.lst_hours,
            track_ra_deg=track.ra_deg,
            ra_grid_deg=ra_grid_deg,
            dec_grid_deg=dec_grid_deg,
            background_layer=self.background_layer,
            projected_background=projected_background,
            footprint_model=self.footprint_model,
            footprint_mask=footprint_mask,
            footprint_contours=footprint_contours,
            snapshots=snapshots,
            source_metrics=source_metrics,
            beam_projection=beam_projection,
            beam_contours=beam_contours,
            beam_reference_ra_deg=beam_reference_ra_deg,
        )

    @staticmethod
    def _zenith_ra_deg(location, obstime) -> float:
        import astropy.units as u
        from astropy.coordinates import AltAz, SkyCoord

        zenith = SkyCoord(
            alt=90 * u.deg,
            az=0 * u.deg,
            frame=AltAz(obstime=obstime, location=location),
        )
        return float(zenith.transform_to("icrs").ra.deg)

    def _build_track_samples(
        self,
        location,
        step_seconds: float,
    ) -> _TrackSamples:
        from astropy.time import Time, TimeDelta

        if step_seconds <= 0:
            raise ValueError(f"step_seconds must be > 0, got {step_seconds}")

        if self.lst_start_hours is not None and self.lst_end_hours is not None:
            raw_start_ra = self.lst_start_hours * 15.0
            raw_end_ra = unwrap_interval_end_deg(
                raw_start_ra,
                self.lst_end_hours * 15.0,
            )
            width_deg = raw_end_ra - raw_start_ra
            step_deg = step_seconds * SIDEREAL_DEG_PER_SECOND
            offsets = np.arange(0.0, width_deg + 0.5 * step_deg, step_deg)
            raw_ra_deg = raw_start_ra + offsets
            if raw_ra_deg[-1] < raw_end_ra:
                raw_ra_deg = np.append(raw_ra_deg, raw_end_ra)
            ra_deg = normalize_ra_deg(raw_ra_deg)
            lst_hours = wrap_ra_deg(raw_ra_deg) / 15.0
            labels = tuple(f"LST {lst:.2f}h" for lst in lst_hours)
            time_isos = tuple(None for _ in labels)
            return _TrackSamples(
                labels=labels,
                time_isos=time_isos,
                lst_hours=np.asarray(lst_hours, dtype=float),
                ra_deg=np.asarray(ra_deg, dtype=float),
                raw_ra_deg=np.asarray(raw_ra_deg, dtype=float),
            )

        if self.start_time_iso is None or self.duration_seconds is None:
            raise ValueError(
                "Provide either (lst_start_hours, lst_end_hours) or "
                "(start_time_iso, duration_seconds)."
            )

        start = Time(self.start_time_iso)
        duration = float(self.duration_seconds)
        offsets = np.arange(0.0, duration + 0.5 * step_seconds, step_seconds)
        if offsets[-1] < duration:
            offsets = np.append(offsets, duration)

        times = [start + TimeDelta(offset, format="sec") for offset in offsets]
        raw_ra_deg = np.asarray(
            [self._zenith_ra_deg(location, obstime) for obstime in times],
            dtype=float,
        )
        ra_deg = normalize_ra_deg(raw_ra_deg)
        lst_hours = wrap_ra_deg(raw_ra_deg) / 15.0
        labels = tuple(obstime.iso for obstime in times)
        time_isos = tuple(obstime.iso for obstime in times)
        return _TrackSamples(
            labels=labels,
            time_isos=time_isos,
            lst_hours=lst_hours,
            ra_deg=ra_deg,
            raw_ra_deg=raw_ra_deg,
        )

    def _resolve_field_radius_deg(self) -> float:
        if self.field_radius_deg is not None:
            return self.field_radius_deg

        diameter_m = self.beam_diameter_m or 14.0
        wavelength_m = 299_792_458.0 / self.frequency_hz
        hpbw_rad = 1.22 * wavelength_m / diameter_m
        return float(np.degrees(hpbw_rad) / 2.0)

    def _resolve_sky_model(self):
        if self.sky_model is not None:
            return self.sky_model
        if self.sky_model_name is None:
            return None
        from rrivis.core.precision import PrecisionConfig
        from rrivis.core.sky.registry import loader_registry

        loader_name, kwargs = loader_registry.resolve_request(
            self.sky_model_name,
            self.sky_model_kwargs,
        )
        kwargs.setdefault("precision", PrecisionConfig.standard())
        return loader_registry.loader(loader_name)(**kwargs)

    def _project_background(self, sky) -> np.ndarray | None:
        if self.background_layer == "none" or sky is None or sky.healpix is None:
            return None

        import healpy as hp
        import matplotlib.pyplot as plt
        from healpy.rotator import Rotator

        healpix = sky.healpix.to_dense() if sky.healpix.is_sparse else sky.healpix
        coordinate_frame = getattr(healpix, "coordinate_frame", "icrs")
        freq_idx = sky.resolve_frequency_index(self.frequency_hz)
        healpix_map = np.asarray(healpix.maps[freq_idx], dtype=float)

        if coordinate_frame.lower() == "galactic":
            healpix_map = Rotator(coord=["G", "C"]).rotate_map_pixel(healpix_map)
        elif coordinate_frame.lower() != "icrs":
            raise ValueError(
                "SkyVisibilityPlanner only supports HEALPix backgrounds in "
                f"ICRS or Galactic frames, got {coordinate_frame!r}."
            )

        projected = hp.cartview(
            healpix_map,
            xsize=4000,
            norm="hist",
            coord="C",
            flip="geo",
            title="",
            unit="Brightness",
            return_projected_map=True,
            notext=True,
        )
        plt.close()
        return projected

    def _build_source_metrics(
        self,
        *,
        sky,
        track: _TrackSamples,
        snapshots: _TrackSamples,
        field_radius_deg: float,
    ) -> tuple[VisibilitySourceMetrics | None, np.ndarray | None]:
        if sky is None or sky.point is None or not self.include_source_metrics:
            return None, None

        point = sky.point
        if len(point.ra_rad) == 0:
            return None, None

        order = np.argsort(point.flux)[::-1][: self.max_point_sources]
        ra_deg = np.degrees(point.ra_rad[order])
        dec_deg = np.degrees(point.dec_rad[order])
        flux_jy = np.asarray(point.flux[order], dtype=float)
        source_name = (
            np.asarray(point.source_name[order]).astype(str)
            if point.source_name is not None
            else None
        )

        separations = angular_separation_deg(
            track.ra_deg[:, None],
            np.full((len(track.ra_deg), 1), self.latitude_deg),
            ra_deg[None, :],
            dec_deg[None, :],
        )
        visible = separations <= field_radius_deg
        visible_any = visible.any(axis=0)
        visible_fraction = visible.mean(axis=0)
        min_separation_deg = separations.min(axis=0)

        first_visible_index = np.full(len(ra_deg), -1, dtype=int)
        last_visible_index = np.full(len(ra_deg), -1, dtype=int)
        if np.any(visible_any):
            visible_cols = np.where(visible_any)[0]
            first_visible_index[visible_cols] = np.argmax(
                visible[:, visible_cols], axis=0
            )
            reversed_visible = visible[:, visible_cols][::-1]
            last_visible_index[visible_cols] = (
                len(track.labels) - 1 - np.argmax(reversed_visible, axis=0)
            )

        visible_indices = np.where(visible_any)[0]
        top_visible_indices = visible_indices[
            np.argsort(flux_jy[visible_indices])[::-1]
        ][: self.top_n_sources]

        nearby_mask = (~visible_any) & (
            min_separation_deg <= field_radius_deg + self.nearby_buffer_deg
        )
        nearby_indices = np.where(nearby_mask)[0]
        nearby_indices = nearby_indices[np.argsort(flux_jy[nearby_indices])[::-1]][
            : self.nearby_source_count
        ]

        snapshot_visibility = (
            angular_separation_deg(
                snapshots.ra_deg[:, None],
                np.full((len(snapshots.ra_deg), 1), self.latitude_deg),
                ra_deg[None, :],
                dec_deg[None, :],
            )
            <= field_radius_deg
        )

        return (
            VisibilitySourceMetrics(
                ra_deg=np.asarray(ra_deg, dtype=float),
                dec_deg=np.asarray(dec_deg, dtype=float),
                flux_jy=flux_jy,
                x_coord=np.asarray(axis_from_ra_deg(ra_deg, self.x_axis), dtype=float),
                source_name=source_name,
                visible_any=visible_any,
                visible_fraction=np.asarray(visible_fraction, dtype=float),
                min_separation_deg=np.asarray(min_separation_deg, dtype=float),
                first_visible_index=first_visible_index,
                last_visible_index=last_visible_index,
                top_visible_indices=np.asarray(top_visible_indices, dtype=int),
                nearby_indices=np.asarray(nearby_indices, dtype=int),
            ),
            snapshot_visibility,
        )

    def _build_footprint_mask(
        self,
        *,
        ra_grid_deg: np.ndarray,
        dec_grid_deg: np.ndarray,
        track: _TrackSamples,
        field_radius_deg: float,
    ) -> np.ndarray:
        ra_mesh, dec_mesh = np.meshgrid(ra_grid_deg, dec_grid_deg)

        if self.footprint_model == "rectangular_approx":
            dec_lower = self.latitude_deg - field_radius_deg
            dec_upper = self.latitude_deg + field_radius_deg
            ra_wrapped = wrap_ra_deg(ra_mesh)
            start = wrap_ra_deg(track.raw_ra_deg[0])
            end = wrap_ra_deg(track.raw_ra_deg[-1])
            width = circular_interval_width_deg(start, end)
            delta = (ra_wrapped - start) % 360.0
            ra_mask = delta <= width + 1e-9
            dec_mask = (dec_mesh >= dec_lower) & (dec_mesh <= dec_upper)
            return ra_mask & dec_mask

        mask = np.zeros_like(ra_mesh, dtype=bool)
        for center_ra in track.ra_deg:
            separation = angular_separation_deg(
                ra_mesh,
                dec_mesh,
                center_ra,
                self.latitude_deg,
            )
            mask |= separation <= field_radius_deg
        return mask

    def _build_snapshots(
        self,
        *,
        snapshots_track: _TrackSamples,
        source_visibility: np.ndarray | None,
        ra_grid_deg: np.ndarray,
        dec_grid_deg: np.ndarray,
        field_radius_deg: float,
    ) -> tuple[VisibilitySnapshot, ...]:
        ra_mesh, dec_mesh = np.meshgrid(ra_grid_deg, dec_grid_deg)
        snapshots: list[VisibilitySnapshot] = []
        for idx, (label, utc_iso, lst_hours, center_ra) in enumerate(
            zip(
                snapshots_track.labels,
                snapshots_track.time_isos,
                snapshots_track.lst_hours,
                snapshots_track.ra_deg,
                strict=False,
            )
        ):
            footprint_mask = (
                angular_separation_deg(
                    ra_mesh,
                    dec_mesh,
                    center_ra,
                    self.latitude_deg,
                )
                <= field_radius_deg
            )
            visible_source_mask = None
            if source_visibility is not None:
                visible_source_mask = np.asarray(source_visibility[idx], dtype=bool)
            snapshots.append(
                VisibilitySnapshot(
                    label=label,
                    utc_iso=utc_iso,
                    lst_hours=float(lst_hours),
                    zenith_ra_deg=float(center_ra),
                    zenith_dec_deg=self.latitude_deg,
                    footprint_mask=footprint_mask,
                    visible_source_mask=visible_source_mask,
                )
            )
        return tuple(snapshots)

    def _build_beam_projection(
        self,
        *,
        ra_grid_deg: np.ndarray,
        dec_grid_deg: np.ndarray,
        track: _TrackSamples,
    ):
        beam_power_func = self._build_beam_power_func()
        if beam_power_func is None:
            return None, None, None

        reference_ra_deg = self._resolve_beam_reference_ra_deg(track)
        projection = compute_beam_power_on_full_sky_grid(
            beam_power_func=beam_power_func,
            zenith_ra_deg=reference_ra_deg,
            zenith_dec_deg=self.latitude_deg,
            ra_grid_deg=ra_grid_deg,
            dec_grid_deg=dec_grid_deg,
            max_za_deg=90.0,
        )

        from rrivis.core.jones.beam.projection import extract_contours

        contours = extract_contours(projection, levels_db=[-3.0, -10.0])
        return projection, contours, reference_ra_deg

    def _resolve_beam_reference_ra_deg(self, track: _TrackSamples) -> float:
        if isinstance(self.beam_reference, (int, float)):
            return float(normalize_ra_deg(float(self.beam_reference) * 15.0))
        if self.beam_reference == "start":
            return float(track.ra_deg[0])
        if self.beam_reference == "end":
            return float(track.ra_deg[-1])
        if self.beam_reference == "midpoint":
            midpoint = 0.5 * (track.raw_ra_deg[0] + track.raw_ra_deg[-1])
            return float(normalize_ra_deg(midpoint))
        raise ValueError(f"Unknown beam_reference={self.beam_reference!r}")

    def _build_title(self, track: _TrackSamples) -> str:
        prefix = "Sky Visibility"
        if self.mode == "snapshots":
            prefix = "Sky Visibility Snapshots"
        if track.time_isos[0] and track.time_isos[-1]:
            return f"{prefix}: {track.time_isos[0]} to {track.time_isos[-1]}"
        return (
            f"{prefix}: LST {track.lst_hours[0] % 24:.2f}h to "
            f"{track.lst_hours[-1] % 24:.2f}h"
        )

    def _build_beam_power_func(self):
        if self.beam_fits_path is not None:
            return self._fits_beam_power_func()
        if self.beam_config.get("beam_mode") == "analytic" and self.beam_diameter_m:
            return self._analytic_beam_power_func()
        return None

    def _analytic_beam_power_func(self):
        from rrivis.core.jones.beam.analytic import compute_aperture_beam

        diameter = self.beam_diameter_m
        frequency_hz = self.frequency_hz
        cfg = self.beam_config

        aperture_shape = cfg.get("aperture_shape", "circular")
        taper = cfg.get("taper", "gaussian")
        edge_taper_db = cfg.get("edge_taper_dB", 10.0)
        feed_model = cfg.get("feed_model", "none")
        feed_computation = cfg.get("feed_computation", "analytical")
        feed_params = cfg.get("feed_params")
        reflector_type = cfg.get("reflector_type", "prime_focus")
        magnification = cfg.get("magnification", 1.0)
        aperture_params = cfg.get("aperture_params")

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
                edge_taper_dB=edge_taper_db,
                feed_model=feed_model,
                feed_computation=feed_computation,
                feed_params=feed_params,
                reflector_type=reflector_type,
                magnification=magnification,
                aperture_params=aperture_params,
            )
            power = np.abs(jones[:, 0, 0]) ** 2 + np.abs(jones[:, 1, 1]) ** 2
            peak = np.nanmax(power)
            if peak > 0:
                power = power / peak
            return power.reshape(za_rad.shape)

        return power_func

    def _fits_beam_power_func(self):
        from scipy.interpolate import RegularGridInterpolator

        try:
            from pyuvdata import UVBeam
        except ImportError:
            logger.warning("pyuvdata not installed; FITS beam overlay disabled.")
            return None

        beam = UVBeam()
        beam.read_beamfits(self.beam_fits_path)
        freq_array_mhz = beam.freq_array / 1e6
        freq_idx = int(np.argmin(np.abs(freq_array_mhz - self.frequency_mhz)))
        az_array = beam.axis1_array
        za_array = beam.axis2_array

        if beam.beam_type == "efield":
            e_theta = beam.data_array[0, 0, freq_idx, :, :]
            e_phi = beam.data_array[1, 0, freq_idx, :, :]
            power = np.abs(e_theta) ** 2 + np.abs(e_phi) ** 2
        else:
            power = beam.data_array[0, 0, freq_idx, :, :]

        peak = np.nanmax(power)
        if peak > 0:
            power = power / peak

        az_ext = np.append(az_array, 2.0 * np.pi)
        power_ext = np.column_stack([power, power[:, 0]])
        interpolator = RegularGridInterpolator(
            (za_array, az_ext),
            power_ext,
            method="linear",
            bounds_error=False,
            fill_value=0.0,
        )

        def power_func(za_rad: np.ndarray, az_rad: np.ndarray) -> np.ndarray:
            points = np.column_stack([za_rad.ravel(), az_rad.ravel()])
            return interpolator(points).reshape(za_rad.shape)

        return power_func
