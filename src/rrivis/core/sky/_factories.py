# rrivis/core/sky/_factories.py
"""Factory functions for SkyModel creation.

Extracted from model.py to keep SkyModel focused on data access and conversion.
"""

from __future__ import annotations

import concurrent.futures
import logging
from typing import TYPE_CHECKING, Any

import numpy as np

from .constants import BrightnessConversion
from .model import SkyFormat

if TYPE_CHECKING:
    from rrivis.core.precision import PrecisionConfig

    from .model import SkyModel

logger = logging.getLogger(__name__)


def create_empty(
    model_name: str,
    brightness_conversion: BrightnessConversion = BrightnessConversion.PLANCK,
    precision: PrecisionConfig | None = None,
    reference_frequency: float | None = None,
) -> SkyModel:
    """Return an empty point-source SkyModel (zero-length arrays).

    Parameters
    ----------
    model_name : str
        Name for the model.
    brightness_conversion : BrightnessConversion
        Brightness conversion method.
    precision : PrecisionConfig, optional
        Precision configuration.
    reference_frequency : float, optional
        Reference frequency in Hz.

    Returns
    -------
    SkyModel
    """
    from .model import SkyModel

    if precision is None:
        from rrivis.core.precision import PrecisionConfig

        precision = PrecisionConfig.standard()

    return SkyModel(
        _ra_rad=np.zeros(0, dtype=np.float64),
        _dec_rad=np.zeros(0, dtype=np.float64),
        _flux=np.zeros(0, dtype=np.float64),
        _spectral_index=np.zeros(0, dtype=np.float64),
        _stokes_q=np.zeros(0, dtype=np.float64),
        _stokes_u=np.zeros(0, dtype=np.float64),
        _stokes_v=np.zeros(0, dtype=np.float64),
        _ref_freq=np.zeros(0, dtype=np.float64),
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
    model_name: str = "custom",
    reference_frequency: float | None = None,
    brightness_conversion: BrightnessConversion = BrightnessConversion.PLANCK,
    precision: PrecisionConfig | None = None,
) -> SkyModel:
    """Create a SkyModel from numpy arrays.

    This is the preferred numpy-native constructor for point-source models.
    """
    from .model import SkyModel

    if precision is None:
        from rrivis.core.precision import PrecisionConfig

        precision = PrecisionConfig.standard()

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

    return SkyModel(
        _ra_rad=np.asarray(ra_rad, dtype=src_dt),
        _dec_rad=np.asarray(dec_rad, dtype=src_dt),
        _flux=np.asarray(flux, dtype=flux_dt),
        _spectral_index=np.asarray(spectral_index, dtype=si_dt),
        _stokes_q=np.asarray(stokes_q, dtype=flux_dt),
        _stokes_u=np.asarray(stokes_u, dtype=flux_dt),
        _stokes_v=np.asarray(stokes_v, dtype=flux_dt),
        _rotation_measure=rotation_measure,
        _major_arcsec=major_arcsec,
        _minor_arcsec=minor_arcsec,
        _pa_deg=pa_deg,
        _spectral_coeffs=spectral_coeffs,
        _ref_freq=ref_freq,
        _native_format=SkyFormat.POINT_SOURCES,
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

    if kwargs.get("_precision") is None:
        from rrivis.core.precision import PrecisionConfig

        kwargs["_precision"] = PrecisionConfig.standard()

    sorted_freqs = np.sort(np.array(list(i_maps.keys()), dtype=np.float64))
    i_arr = np.stack([i_maps[f] for f in sorted_freqs])
    q_arr = np.stack([q_maps[f] for f in sorted_freqs]) if q_maps else None
    u_arr = np.stack([u_maps[f] for f in sorted_freqs]) if u_maps else None
    v_arr = np.stack([v_maps[f] for f in sorted_freqs]) if v_maps else None

    kwargs.setdefault("_active_mode", SkyFormat.HEALPIX)

    return SkyModel(
        _healpix_maps=i_arr,
        _healpix_q_maps=q_arr,
        _healpix_u_maps=u_arr,
        _healpix_v_maps=v_arr,
        _healpix_nside=nside,
        _observation_frequencies=sorted_freqs,
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
    precision: PrecisionConfig | None = None,
    polarization_fraction: float = 0.0,
    polarization_angle_deg: float = 0.0,
    stokes_v_fraction: float = 0.0,
) -> SkyModel:
    """Generate synthetic test sources."""
    from .model import SkyModel

    if precision is None:
        from rrivis.core.precision import PrecisionConfig

        precision = PrecisionConfig.standard()

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

    return SkyModel(
        _ra_rad=SkyModel.deg_to_rad_at_precision(ra_deg_arr, precision),
        _dec_rad=SkyModel.deg_to_rad_at_precision(dec_deg_arr, precision),
        _flux=flux_arr.astype(np.float64),
        _spectral_index=np.full(n, float(spectral_index)),
        _stokes_q=stokes_q_arr,
        _stokes_u=stokes_u_arr,
        _stokes_v=stokes_v_arr,
        _native_format=SkyFormat.POINT_SOURCES,
        model_name="test_sources",
        brightness_conversion=brightness_conversion,
        _precision=precision,
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
    results: list[SkyModel] = []
    failures: list[tuple[str, Exception]] = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=n) as pool:
        future_to_name: dict[concurrent.futures.Future, str] = {}
        for method_name, kwargs in loaders:
            kw = dict(kwargs)
            if precision is not None and "precision" not in kw:
                kw["precision"] = precision
            f = pool.submit(_load_one, method_name, kw)
            future_to_name[f] = method_name

        for future in concurrent.futures.as_completed(future_to_name):
            name = future_to_name[future]
            try:
                sky = future.result()
                if sky.has_point_sources or sky._healpix_maps is not None:
                    results.append(sky)
                    logger.info(
                        f"Parallel load complete: {name} "
                        f"({sky.n_sky_elements:,} sky elements)"
                    )
                else:
                    logger.info(f"Parallel load: {name} returned empty model")
            except Exception as e:
                failures.append((name, e))
                logger.warning(f"Parallel load failed for {name}: {e}")

    logger.info(f"load_parallel: {len(results)}/{len(loaders)} loaders succeeded")

    if failures and strict:
        names = ", ".join(n for n, _ in failures)
        raise RuntimeError(
            f"load_parallel: {len(failures)}/{len(loaders)} loaders failed "
            f"(strict=True): {names}"
        )

    return results
