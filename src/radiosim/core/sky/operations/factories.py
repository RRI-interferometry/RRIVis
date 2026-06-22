"""Factory functions for SkyModel creation.

Extracted from model.py to keep SkyModel focused on data access and conversion.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from ..containers import (
    HealpixData,
    MonopoleConvention,
    PointSourceData,
    SkyCoverage,
    SkyProvenance,
    SourceSubtractionStatus,
)
from ..containers.constants import BrightnessConversion
from ..containers.model import SkyModel
from ..support.point_builder import point_source_data_from_mapping

if TYPE_CHECKING:
    from radiosim.core.precision import PrecisionConfig

logger = logging.getLogger(__name__)


def _require_precision(precision: PrecisionConfig | None) -> PrecisionConfig:
    if precision is None:
        raise ValueError(
            "Sky model construction requires an explicit PrecisionConfig. "
            "Pass precision=... at the loader or constructor boundary."
        )
    return precision


def create_empty(
    model_name: str,
    brightness_conversion: BrightnessConversion | str = BrightnessConversion.PLANCK,
    *,
    precision: PrecisionConfig,
    reference_frequency: float | None = None,
    provenance: SkyProvenance | None = None,
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
    provenance : SkyProvenance, optional
        Physical-correctness metadata.  Defaults to an UNKNOWN-sentinel.

    Returns
    -------
    SkyModel
    """
    precision = _require_precision(precision)
    brightness_conversion = BrightnessConversion(brightness_conversion)

    return SkyModel(
        point=PointSourceData.empty(),
        model_name=model_name,
        brightness_conversion=brightness_conversion,
        precision=precision,
        reference_frequency=reference_frequency,
        provenance=provenance if provenance is not None else SkyProvenance(),
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
    brightness_conversion: BrightnessConversion | str = BrightnessConversion.PLANCK,
    *,
    precision: PrecisionConfig,
    provenance: SkyProvenance | None = None,
) -> SkyModel:
    """Create a SkyModel from numpy arrays.

    This is the preferred numpy-native constructor for point-source models.
    Pass ``provenance=`` to declare physical-correctness metadata (flux
    completeness, angular resolution, monopole convention).
    """
    precision = _require_precision(precision)
    brightness_conversion = BrightnessConversion(brightness_conversion)

    # Resolve dtypes from precision config (defaults for omitted columns; the
    # core-column recast is applied centrally by point_source_data_from_mapping).
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

    point = point_source_data_from_mapping(
        {
            "ra_rad": ra_rad,
            "dec_rad": dec_rad,
            "flux": flux,
            "spectral_index": spectral_index,
            "stokes_q": stokes_q,
            "stokes_u": stokes_u,
            "stokes_v": stokes_v,
            "ref_freq": ref_freq,
            "rotation_measure": rotation_measure,
            "major_arcsec": major_arcsec,
            "minor_arcsec": minor_arcsec,
            "pa_deg": pa_deg,
            "spectral_coeffs": spectral_coeffs,
            "source_name": source_name,
            "source_id": source_id,
            "extra_columns": {} if extra_columns is None else extra_columns,
        },
        precision=precision,
    )

    return SkyModel(
        point=point,
        model_name=model_name,
        reference_frequency=reference_frequency,
        brightness_conversion=brightness_conversion,
        precision=precision,
        provenance=provenance if provenance is not None else SkyProvenance(),
    )


def create_from_freq_dict_maps(
    i_maps: dict[float, np.ndarray],
    q_maps: dict[float, np.ndarray] | None,
    u_maps: dict[float, np.ndarray] | None,
    v_maps: dict[float, np.ndarray] | None,
    nside: int,
    *,
    precision: PrecisionConfig,
    coordinate_frame: str = "icrs",
    model_name: str | None = None,
    reference_frequency: float | None = None,
    brightness_conversion: BrightnessConversion | str = BrightnessConversion.PLANCK,
    provenance: SkyProvenance | None = None,
) -> SkyModel:
    """Create a SkyModel from frequency-keyed dicts of HEALPix maps.

    Standard constructor for loaders that build dict[float, ndarray]
    during generation (pygdsm, pysm3, etc.).

    Parameters
    ----------
    i_maps, q_maps, u_maps, v_maps : dict[float, np.ndarray] or None
        Frequency-keyed HEALPix maps (Kelvin). ``i_maps`` is required;
        the polarized cubes are optional.
    nside : int
        HEALPix NSIDE resolution of the maps.
    precision : PrecisionConfig
        Precision configuration.
    coordinate_frame : str, default "icrs"
        Coordinate frame of the HEALPix pixelization.
    model_name : str, optional
        Name for the model.
    reference_frequency : float, optional
        Reference frequency in Hz.
    brightness_conversion : BrightnessConversion or str
        Brightness conversion method for Stokes I.
    provenance : SkyProvenance, optional
        Physical-correctness metadata. Defaults to an UNKNOWN sentinel.
    """
    precision = _require_precision(precision)
    brightness_conversion = BrightnessConversion(brightness_conversion)

    sorted_freqs = np.sort(np.array(list(i_maps.keys()), dtype=np.float64))
    i_arr = np.stack([i_maps[f] for f in sorted_freqs])
    q_arr = np.stack([q_maps[f] for f in sorted_freqs]) if q_maps else None
    u_arr = np.stack([u_maps[f] for f in sorted_freqs]) if u_maps else None
    v_arr = np.stack([v_maps[f] for f in sorted_freqs]) if v_maps else None

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
        precision=precision,
        model_name=model_name,
        reference_frequency=reference_frequency,
        brightness_conversion=brightness_conversion,
        provenance=provenance if provenance is not None else SkyProvenance(),
    )


def create_test_sources(
    num_sources: int = 100,
    flux_range: tuple[float, float] = (1.0, 10.0),
    dec_deg: float = -30.0,
    spectral_index: float = -0.7,
    distribution: str = "uniform",
    seed: int | None = None,
    dec_range_deg: float | None = None,
    brightness_conversion: BrightnessConversion | str = BrightnessConversion.PLANCK,
    *,
    precision: PrecisionConfig,
    polarization_fraction: float = 0.0,
    polarization_angle_deg: float = 0.0,
    stokes_v_fraction: float = 0.0,
    reference_frequency: float | None = None,
    provenance: SkyProvenance | None = None,
) -> SkyModel:
    """Generate synthetic test sources."""
    precision = _require_precision(precision)
    brightness_conversion = BrightnessConversion(brightness_conversion)

    if distribution not in ("uniform", "random"):
        raise ValueError(
            f"distribution must be 'uniform' or 'random', got '{distribution}'"
        )

    n = num_sources

    resolved_seed: int | None = None
    if distribution == "random":
        resolved_seed = (
            seed if seed is not None else int(np.random.SeedSequence().entropy)
        )
        rng = np.random.default_rng(resolved_seed)
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

    if provenance is None:
        provenance = SkyProvenance(
            flux_completeness_jy=(float(flux_range[0]), float(flux_range[1])),
            flux_completeness_freq_hz=reference_frequency,
            angular_resolution_rad=(0.0, float(np.pi)),
            sky_coverage=SkyCoverage.FULL_SKY,
            monopole_convention=MonopoleConvention.ABSOLUTE_NO_CMB,
            source_subtraction=SourceSubtractionStatus.NONE,
            notes="synthetic/test_sources",
            rng_seed=resolved_seed,
        )

    return create_from_arrays(
        ra_rad=SkyModel.deg_to_rad_at_precision(ra_deg_arr, precision),
        dec_rad=SkyModel.deg_to_rad_at_precision(dec_deg_arr, precision),
        flux=flux_arr.astype(np.float64),
        spectral_index=np.full(n, float(spectral_index)),
        stokes_q=stokes_q_arr,
        stokes_u=stokes_u_arr,
        stokes_v=stokes_v_arr,
        model_name="test_sources",
        reference_frequency=reference_frequency,
        brightness_conversion=brightness_conversion,
        precision=precision,
        provenance=provenance,
    )
