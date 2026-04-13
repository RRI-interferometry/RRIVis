# rrivis/core/sky/_factories.py
"""Factory functions for SkyModel creation.

Extracted from model.py to keep SkyModel focused on data access and conversion.
"""

from __future__ import annotations

import concurrent.futures
import logging
import traceback
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from ._data import HealpixData, PointSourceData
from .constants import BrightnessConversion
from .model import SkyFormat

if TYPE_CHECKING:
    from rrivis.core.precision import PrecisionConfig

    from .model import SkyModel

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SkyLoadError:
    """Context for one failed sky-loader request."""

    loader_name: str
    kwargs: dict[str, Any]
    exception: Exception
    traceback_text: str


class SkyLoadAggregateError(RuntimeError):
    """Raised when strict parallel sky loading has one or more failures."""

    def __init__(self, failures: list[SkyLoadError]) -> None:
        self.failures = failures
        details = "\n".join(
            f"- {failure.loader_name}: {failure.exception}" for failure in failures
        )
        super().__init__(
            f"load_parallel: {len(failures)} loader(s) failed (strict=True):\n{details}"
        )


def _require_precision(precision: PrecisionConfig | None) -> PrecisionConfig:
    if precision is None:
        raise ValueError(
            "Sky model construction requires an explicit PrecisionConfig. "
            "Pass precision=... at the loader or constructor boundary."
        )
    return precision


def create_empty(
    model_name: str,
    brightness_conversion: BrightnessConversion = BrightnessConversion.PLANCK,
    *,
    precision: PrecisionConfig,
    reference_frequency: float | None = None,
) -> SkyModel:
    """Return an empty point-source SkyModel (zero-length arrays).

    Parameters
    ----------
    model_name : str
        Name for the model.
    brightness_conversion : BrightnessConversion
        Brightness conversion method.
    precision : PrecisionConfig
        Precision configuration.
    reference_frequency : float, optional
        Reference frequency in Hz.

    Returns
    -------
    SkyModel
    """
    from .model import SkyModel

    precision = _require_precision(precision)

    return SkyModel(
        point=PointSourceData.empty(),
        source_format=SkyFormat.POINT_SOURCES,
        model_name=model_name,
        brightness_conversion=brightness_conversion,
        _precision=precision,
        reference_frequency=reference_frequency,
    )


def create_from_arrays(
    ra_rad: np.ndarray,
    dec_rad: np.ndarray,
    flux: np.ndarray,
    spectral_index: np.ndarray | None = None,
    stokes_q: np.ndarray | None = None,
    stokes_u: np.ndarray | None = None,
    stokes_v: np.ndarray | None = None,
    rotation_measure: np.ndarray | None = None,
    major_arcsec: np.ndarray | None = None,
    minor_arcsec: np.ndarray | None = None,
    pa_deg: np.ndarray | None = None,
    spectral_coeffs: np.ndarray | None = None,
    ref_freq: np.ndarray | None = None,
    source_name: np.ndarray | None = None,
    source_id: np.ndarray | None = None,
    extra_columns: dict[str, np.ndarray] | None = None,
    model_name: str = "custom",
    reference_frequency: float | None = None,
    brightness_conversion: BrightnessConversion = BrightnessConversion.PLANCK,
    *,
    precision: PrecisionConfig,
) -> SkyModel:
    """Create a SkyModel from numpy arrays.

    This is the preferred numpy-native constructor for point-source models.
    """
    from .model import SkyModel

    precision = _require_precision(precision)

    # Resolve dtypes from precision config
    src_dt = precision.sky_model.get_dtype("source_positions")
    flux_dt = precision.sky_model.get_dtype("flux")
    si_dt = precision.sky_model.get_dtype("spectral_index")

    n = len(ra_rad)
    if spectral_index is None:
        spectral_index = np.full(n, -0.7, dtype=si_dt)
    if stokes_q is None:
        stokes_q = np.zeros(n, dtype=flux_dt)
    if stokes_u is None:
        stokes_u = np.zeros(n, dtype=flux_dt)
    if stokes_v is None:
        stokes_v = np.zeros(n, dtype=flux_dt)
    if ref_freq is None and reference_frequency is not None:
        ref_freq = np.full(n, reference_frequency, dtype=flux_dt)
    if ref_freq is None:
        ref_freq = np.full(n, reference_frequency or 0.0, dtype=flux_dt)

    point = PointSourceData(
        ra_rad=np.asarray(ra_rad, dtype=src_dt),
        dec_rad=np.asarray(dec_rad, dtype=src_dt),
        flux=np.asarray(flux, dtype=flux_dt),
        spectral_index=np.asarray(spectral_index, dtype=si_dt),
        stokes_q=np.asarray(stokes_q, dtype=flux_dt),
        stokes_u=np.asarray(stokes_u, dtype=flux_dt),
        stokes_v=np.asarray(stokes_v, dtype=flux_dt),
        ref_freq=np.asarray(ref_freq, dtype=flux_dt),
        rotation_measure=rotation_measure,
        major_arcsec=major_arcsec,
        minor_arcsec=minor_arcsec,
        pa_deg=pa_deg,
        spectral_coeffs=spectral_coeffs,
        source_name=source_name,
        source_id=source_id,
        extra_columns={} if extra_columns is None else extra_columns,
    )

    return SkyModel(
        point=point,
        source_format=SkyFormat.POINT_SOURCES,
        model_name=model_name,
        reference_frequency=reference_frequency,
        brightness_conversion=brightness_conversion,
        _precision=precision,
    )


def create_from_freq_dict_maps(
    i_maps: dict[float, np.ndarray],
    q_maps: dict[float, np.ndarray] | None,
    u_maps: dict[float, np.ndarray] | None,
    v_maps: dict[float, np.ndarray] | None,
    nside: int,
    **kwargs: Any,
) -> SkyModel:
    """Create a SkyModel from frequency-keyed dicts of HEALPix maps.

    Standard constructor for loaders that build dict[float, ndarray]
    during generation (pygdsm, pysm3, etc.).
    """
    from .model import SkyModel

    precision = _require_precision(
        kwargs.pop("precision", kwargs.pop("_precision", None))
    )

    sorted_freqs = np.sort(np.array(list(i_maps.keys()), dtype=np.float64))
    i_arr = np.stack([i_maps[f] for f in sorted_freqs])
    q_arr = np.stack([q_maps[f] for f in sorted_freqs]) if q_maps else None
    u_arr = np.stack([u_maps[f] for f in sorted_freqs]) if u_maps else None
    v_arr = np.stack([v_maps[f] for f in sorted_freqs]) if v_maps else None
    coordinate_frame = kwargs.pop("coordinate_frame", "icrs")

    healpix = HealpixData(
        maps=i_arr,
        nside=nside,
        frequencies=sorted_freqs,
        coordinate_frame=coordinate_frame,
        q_maps=q_arr,
        u_maps=u_arr,
        v_maps=v_arr,
    )

    return SkyModel(
        healpix=healpix,
        source_format=kwargs.pop("source_format", SkyFormat.HEALPIX),
        _precision=precision,
        **kwargs,
    )


def create_test_sources(
    num_sources: int = 100,
    flux_range: tuple[float, float] = (1.0, 10.0),
    dec_deg: float = -30.0,
    spectral_index: float = -0.7,
    distribution: str = "uniform",
    seed: int | None = None,
    dec_range_deg: float | None = None,
    brightness_conversion: BrightnessConversion = BrightnessConversion.PLANCK,
    *,
    precision: PrecisionConfig,
    polarization_fraction: float = 0.0,
    polarization_angle_deg: float = 0.0,
    stokes_v_fraction: float = 0.0,
) -> SkyModel:
    """Generate synthetic test sources."""
    from .model import SkyModel

    precision = _require_precision(precision)

    if distribution not in ("uniform", "random"):
        raise ValueError(
            f"distribution must be 'uniform' or 'random', got '{distribution}'"
        )

    n = num_sources

    if distribution == "random":
        rng = np.random.default_rng(seed)
        ra_deg_arr = rng.uniform(0.0, 360.0, size=n)
        half_width = dec_range_deg if dec_range_deg is not None else 10.0
        dec_lo = max(-90.0, dec_deg - half_width)
        dec_hi = min(90.0, dec_deg + half_width)
        dec_deg_arr = rng.uniform(dec_lo, dec_hi, size=n)
        flux_arr = rng.uniform(flux_range[0], flux_range[1], size=n)
        logger.debug(
            f"Generated {n} random test sources "
            f"(seed={seed}, dec=[{dec_lo:.1f}, {dec_hi:.1f}]deg)"
        )
    else:
        if n == 1:
            ra_deg_arr = np.array([0.0])
            flux_arr = np.array([(flux_range[0] + flux_range[1]) / 2])
        else:
            ra_deg_arr = np.array([(360.0 / n) * i for i in range(n)])
            flux_arr = np.linspace(flux_range[0], flux_range[1], n)
        dec_deg_arr = np.full(n, dec_deg)
        logger.debug(f"Generated {n} uniform test sources")

    if polarization_fraction > 0:
        chi_rad = np.deg2rad(polarization_angle_deg)
        stokes_q_arr = flux_arr * polarization_fraction * np.cos(2.0 * chi_rad)
        stokes_u_arr = flux_arr * polarization_fraction * np.sin(2.0 * chi_rad)
    else:
        stokes_q_arr = np.zeros(n, dtype=np.float64)
        stokes_u_arr = np.zeros(n, dtype=np.float64)

    if stokes_v_fraction > 0:
        stokes_v_arr = flux_arr * stokes_v_fraction
    else:
        stokes_v_arr = np.zeros(n, dtype=np.float64)

    return create_from_arrays(
        ra_rad=SkyModel.deg_to_rad_at_precision(ra_deg_arr, precision),
        dec_rad=SkyModel.deg_to_rad_at_precision(dec_deg_arr, precision),
        flux=flux_arr.astype(np.float64),
        spectral_index=np.full(n, float(spectral_index)),
        stokes_q=stokes_q_arr,
        stokes_u=stokes_u_arr,
        stokes_v=stokes_v_arr,
        model_name="test_sources",
        brightness_conversion=brightness_conversion,
        precision=precision,
    )


def load_models_parallel(
    loaders: list[tuple[str, dict[str, Any]]],
    max_workers: int = 8,
    precision: PrecisionConfig | None = None,
    strict: bool = True,
) -> list[SkyModel]:
    """Load multiple sky models in parallel using threads.

    Each loader is a (method_name, kwargs) tuple identifying a registered
    loader function.
    """

    def _load_one(method_name: str, kw: dict) -> SkyModel:
        from ._registry import get_loader

        return get_loader(method_name)(**kw)

    n = min(len(loaders), max_workers)
    results: list[SkyModel | None] = [None] * len(loaders)
    failures: list[SkyLoadError] = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=n) as pool:
        future_to_loader: dict[concurrent.futures.Future, tuple[int, str]] = {}
        for index, (method_name, kwargs) in enumerate(loaders):
            kw = dict(kwargs)
            if precision is not None and "precision" not in kw:
                kw["precision"] = precision
            f = pool.submit(_load_one, method_name, kw)
            future_to_loader[f] = (index, method_name)

        for future in concurrent.futures.as_completed(future_to_loader):
            index, name = future_to_loader[future]
            try:
                sky = future.result()
                if sky.available_formats:
                    results[index] = sky
                    try:
                        n_elements = sky.n_sky_elements
                    except ValueError:
                        n_elements = sky.n_sky_elements_for(SkyFormat.HEALPIX)
                    logger.info(
                        f"Parallel load complete: {name} ({n_elements:,} sky elements)"
                    )
                else:
                    logger.info(f"Parallel load: {name} returned empty model")
            except Exception as e:
                failures.append(
                    SkyLoadError(
                        loader_name=name,
                        kwargs=loaders[index][1],
                        exception=e,
                        traceback_text=traceback.format_exc(),
                    )
                )
                logger.warning(f"Parallel load failed for {name}: {e}")

    loaded = [sky for sky in results if sky is not None]

    logger.info(f"load_parallel: {len(loaded)}/{len(loaders)} loaders succeeded")

    if failures and strict:
        raise SkyLoadAggregateError(failures)

    return loaded
