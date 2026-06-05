"""Synthetic sky-model loaders."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np

from ..containers import (
    MonopoleConvention,
    SkyCoverage,
    SkyProvenance,
    SourceSubtractionStatus,
)
from ..containers.constants import BrightnessConversion
from ..containers.model import SkyFormat, _coerce_format
from ..recipes.dnds_models import DNDSModel, resolve_dn_ds
from ..registry.facade import loader_registry

if TYPE_CHECKING:
    from radiosim.core.precision import PrecisionConfig

    from ..containers.model import SkyModel
    from ..operations.region import SkyRegion

logger = logging.getLogger(__name__)


@loader_registry.register_loader(
    "test_sources",
    config_section="test_sources",
    use_flag="use_test_sources",
    category="synthetic",
    aliases={"test": {}, "test_healpix": {"representation": "healpix_map"}},
    config_fields={
        "num_sources": "num_sources",
        "distribution": "distribution",
        "seed": "seed",
        "flux_min": "flux_min",
        "flux_max": "flux_max",
        "dec_deg": "dec_deg",
        "dec_range_deg": "dec_range_deg",
        "spectral_index": "spectral_index",
        "representation": "representation",
        "nside": "nside",
        "polarization_fraction": "polarization_fraction",
        "polarization_angle_deg": "polarization_angle_deg",
        "stokes_v_fraction": "stokes_v_fraction",
    },
)
def load_test_sources(
    num_sources: int = 100,
    flux_min: float | None = None,
    flux_max: float | None = None,
    dec_deg: float = -30.0,
    spectral_index: float = -0.7,
    distribution: str = "uniform",
    seed: int | None = None,
    dec_range_deg: float | None = None,
    representation: str = "point_sources",
    nside: int = 64,
    frequencies: np.ndarray | None = None,
    obs_frequency_config: dict[str, Any] | None = None,
    reference_frequency: float | None = None,
    brightness_conversion: str = "planck",
    *,
    precision: PrecisionConfig,
    polarization_fraction: float = 0.0,
    polarization_angle_deg: float = 0.0,
    stokes_v_fraction: float = 0.0,
    region: SkyRegion | None = None,
    memmap_path: str | None = None,
    provenance: SkyProvenance | None = None,
) -> SkyModel:
    """Generate synthetic test sources in point or HEALPix form."""
    from radiosim.utils.frequency import parse_frequency_config

    from ..operations.factories import create_test_sources
    from ..operations.operations import materialize_healpix_model

    brightness = BrightnessConversion(brightness_conversion)
    flux_range = (
        (flux_min, flux_max)
        if flux_min is not None and flux_max is not None
        else (1.0, 10.0)
    )
    sky = create_test_sources(
        num_sources=num_sources,
        flux_range=flux_range,
        dec_deg=dec_deg,
        spectral_index=spectral_index,
        distribution=distribution,
        seed=seed,
        dec_range_deg=dec_range_deg,
        brightness_conversion=brightness,
        precision=precision,
        polarization_fraction=polarization_fraction,
        polarization_angle_deg=polarization_angle_deg,
        stokes_v_fraction=stokes_v_fraction,
        reference_frequency=reference_frequency,
    )

    if region is not None:
        sky = sky.filter_region(region)

    if provenance is not None:
        sky = sky.replace(provenance=provenance)

    target = _coerce_format(representation)
    if target == SkyFormat.HEALPIX:
        if frequencies is None and obs_frequency_config is not None:
            frequencies = parse_frequency_config(obs_frequency_config)
        ref_frequency = reference_frequency or (
            float(frequencies[0])
            if frequencies is not None and len(frequencies) > 0
            else None
        )
        if ref_frequency is not None:
            sky = sky.with_reference_frequency(ref_frequency)
        sky = materialize_healpix_model(
            sky,
            nside=nside,
            frequencies=frequencies,
            obs_frequency_config=obs_frequency_config,
            ref_frequency=ref_frequency,
            memmap_path=memmap_path,
            clear_other=True,
        )

    return sky


# =============================================================================
# Poisson confusion loader
# =============================================================================


def _approximate_region_area_sr(region: SkyRegion | None) -> float:
    """Return an approximate solid angle (sr) covered by ``region``.

    Uses cone / box closed-form formulas when possible; falls back to a
    nside=256 HEALPix mask count for unions or custom regions.  When
    ``region`` is None, returns 4π.
    """
    if region is None:
        return 4.0 * np.pi
    return float(region.area_sr())


def _sample_points_on_region(
    n: int,
    region: SkyRegion | None,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Draw ``n`` positions uniformly on the unit sphere (optionally inside ``region``).

    Returns ``(ra_rad, dec_rad)``.  Positions outside ``region`` are rejected
    and re-drawn until ``n`` accepted samples have been produced.  A cap of
    100× ``n`` trial draws is enforced to avoid infinite loops on degenerate
    regions.
    """
    if n <= 0:
        empty = np.zeros(0, dtype=np.float64)
        return empty, empty

    accepted_ra = np.empty(0, dtype=np.float64)
    accepted_dec = np.empty(0, dtype=np.float64)
    if region is None:
        coverage_fraction = 1.0
        trial_cap = max(100 * n, 1_000_000)
    else:
        coverage_fraction = max(
            float(region.area_sr() / (4.0 * np.pi)),
            1.0e-6,
        )
        trial_cap = max(int(np.ceil(20.0 * n / coverage_fraction)), 1_000_000)
    total_trials = 0

    while accepted_ra.size < n:
        remaining = max(n - accepted_ra.size, 1)
        if region is None:
            n_try = remaining
        else:
            expected_needed = int(np.ceil(1.5 * remaining / coverage_fraction))
            n_try = min(max(expected_needed, 4096), 1_000_000)
        # Uniform on the sphere: u ∈ [0, 1) → dec = arcsin(2u − 1).
        u_dec = rng.uniform(0.0, 1.0, size=n_try)
        dec = np.arcsin(2.0 * u_dec - 1.0)
        ra = rng.uniform(0.0, 2.0 * np.pi, size=n_try)
        if region is None:
            accepted_ra = np.concatenate([accepted_ra, ra])
            accepted_dec = np.concatenate([accepted_dec, dec])
        else:
            inside = region.contains(ra, dec)
            accepted_ra = np.concatenate([accepted_ra, ra[inside]])
            accepted_dec = np.concatenate([accepted_dec, dec[inside]])
        total_trials += n_try
        if total_trials > trial_cap and accepted_ra.size < n:
            raise RuntimeError(
                "Region-constrained sampling did not reach the requested "
                f"count ({accepted_ra.size}/{n} after {total_trials} trials)."
            )

    return accepted_ra[:n], accepted_dec[:n]


@loader_registry.register_loader(
    "poisson_confusion",
    config_section="poisson_confusion",
    use_flag="use_poisson_confusion",
    category="synthetic",
    aliases={"confusion_background": {}},
    config_fields={
        "flux_range_jy": "flux_range_jy",
        "reference_frequency": "reference_frequency",
        "dn_ds": "dn_ds",
        "area_sr": "area_sr",
        "representation": "representation",
        "nside": "nside",
        "seed": "seed",
        "spectral_index_dist": "spectral_index_dist",
    },
)
def load_poisson_confusion(
    *,
    flux_range_jy: tuple[float, float],
    reference_frequency: float,
    dn_ds: str | DNDSModel = "franzen2019_gleam_154mhz",
    area_sr: float | None = None,
    region: SkyRegion | None = None,
    representation: str = "point_sources",
    nside: int = 64,
    frequencies: np.ndarray | None = None,
    obs_frequency_config: dict[str, Any] | None = None,
    seed: int | None = None,
    spectral_index_dist: tuple[float, float] = (-0.8, 0.2),
    brightness_conversion: str = "planck",
    precision: PrecisionConfig,
    memmap_path: str | None = None,
) -> SkyModel:
    """Sample a Poisson realisation of the sub-threshold confusion background.

    Fills the flux band ``flux_range_jy`` at ``reference_frequency`` using
    a validated differential-count model ``dN/dS``. The expected source count
    in the band is ``λ = ∫ dN/dS · dS · area_sr``; the actual count is drawn
    from ``Poisson(λ)``.  Fluxes are sampled from the validated ``dN/dS`` PDF
    via inverse-CDF sampling, positions drawn uniformly on the sphere
    (restricted by ``region`` if supplied), and spectral indices drawn from a
    truncated normal ``N(mean, σ)``.

    Parameters
    ----------
    flux_range_jy
        ``(S_min, S_max)`` band in Jy at ``reference_frequency``.
    reference_frequency
        Frequency (Hz) at which ``flux_range_jy`` is evaluated.  Must match the
        model's calibration frequency exactly.
    dn_ds
        Validated preset name or validated :class:`DNDSModel`.
    area_sr
        Sphere area to populate.  Defaults to the approximate solid angle
        of ``region`` (or ``4π`` when ``region`` is None).
    region
        Optional :class:`SkyRegion` restricting the position draw.
    representation
        ``"point_sources"`` (default) or ``"healpix_map"``.
    nside, frequencies, obs_frequency_config
        Forwarded to :func:`materialize_healpix_model` when
        ``representation == "healpix_map"``.
    seed
        RNG seed.  Set for reproducible confusion realisations.
    spectral_index_dist
        ``(mean, σ)`` of the per-source spectral-index draw.  Values are
        clipped to ``[-2.5, +0.5]``.
    brightness_conversion, precision
        Standard RadioSim loader arguments.

    Returns
    -------
    SkyModel
        Sampled confusion population with provenance
        ``flux_completeness_jy=flux_range_jy``,
        ``source_subtraction=NONE``, ``monopole_convention=ABSOLUTE_NO_CMB``.
    """
    from ..operations.factories import create_from_arrays
    from ..operations.operations import materialize_healpix_model

    brightness = BrightnessConversion(brightness_conversion)
    if flux_range_jy[0] <= 0.0 or flux_range_jy[1] <= flux_range_jy[0]:
        raise ValueError(
            f"flux_range_jy must satisfy 0 < S_min < S_max; got {flux_range_jy!r}."
        )
    if area_sr is not None and region is None and not np.isclose(area_sr, 4.0 * np.pi):
        raise ValueError(
            "area_sr without an explicit region is scientifically ambiguous. "
            "Provide region=... or omit area_sr for a full-sky realization."
        )

    model = resolve_dn_ds(dn_ds)
    model_freq = model.reference_frequency_hz
    if not np.isclose(reference_frequency, model_freq, rtol=0.0, atol=1.0):
        raise ValueError(
            "load_poisson_confusion requires reference_frequency to match the "
            f"validated dN/dS calibration frequency exactly. Requested "
            f"{reference_frequency / 1e6:.3f} MHz, but model '{model.name}' is "
            f"calibrated at {model_freq / 1e6:.3f} MHz."
        )

    effective_area = (
        float(area_sr) if area_sr is not None else _approximate_region_area_sr(region)
    )
    if effective_area <= 0.0:
        raise ValueError(
            f"Effective area must be positive, got {effective_area!r}. "
            "Check the supplied region or area_sr."
        )

    expected_n = (
        float(model.integrated_counts(flux_range_jy[0], flux_range_jy[1]))
        * effective_area
    )
    rng = np.random.default_rng(seed)
    n = int(rng.poisson(expected_n))
    logger.info(
        "load_poisson_confusion: dn_ds=%s, band=%.3g–%.3g Jy, area=%.3g sr, "
        "λ=%.1f, drew N=%d sources.",
        model.name,
        flux_range_jy[0],
        flux_range_jy[1],
        effective_area,
        expected_n,
        n,
    )

    if n == 0:
        # Empty realisation — build an empty point catalog with provenance.
        from ..operations.factories import create_empty

        coverage_footprint = region.footprint() if region is not None else None
        empty_prov = SkyProvenance(
            flux_completeness_jy=flux_range_jy,
            flux_completeness_freq_hz=float(reference_frequency),
            angular_resolution_rad=(0.0, float(np.pi)),
            sky_coverage=(
                SkyCoverage.FULL_SKY if region is None else SkyCoverage.PARTIAL_SKY
            ),
            coverage_fraction=(
                1.0
                if region is None
                else (
                    coverage_footprint.coverage_fraction
                    if coverage_footprint is not None
                    else None
                )
            ),
            coverage_footprint=coverage_footprint,
            monopole_convention=MonopoleConvention.ABSOLUTE_NO_CMB,
            monopole_k=None,
            source_subtraction=SourceSubtractionStatus.NONE,
            notes=f"poisson_{model.name}",
        )
        return create_empty(
            f"poisson_{model.name}",
            brightness,
            precision=precision,
            reference_frequency=float(reference_frequency),
            provenance=empty_prov,
        )

    # Sample fluxes from the dN/dS PDF, positions uniformly on the sphere.
    flux_jy = model.sample_flux(n, rng, flux_range_jy[0], flux_range_jy[1])
    ra_rad, dec_rad = _sample_points_on_region(n, region, rng)

    alpha_mean, alpha_sigma = spectral_index_dist
    if alpha_sigma > 0:
        alpha = rng.normal(alpha_mean, alpha_sigma, size=n)
    else:
        alpha = np.full(n, float(alpha_mean))
    _ = np.clip(alpha, -2.5, 0.5, out=alpha)

    sky = create_from_arrays(
        ra_rad=ra_rad,
        dec_rad=dec_rad,
        flux=flux_jy,
        spectral_index=alpha,
        ref_freq=np.full(n, float(reference_frequency), dtype=np.float64),
        model_name=f"poisson_{model.name}",
        reference_frequency=float(reference_frequency),
        brightness_conversion=brightness,
        precision=precision,
    )

    coverage_footprint = region.footprint() if region is not None else None
    provenance = SkyProvenance(
        flux_completeness_jy=(float(flux_range_jy[0]), float(flux_range_jy[1])),
        flux_completeness_freq_hz=float(reference_frequency),
        angular_resolution_rad=(0.0, float(np.pi)),
        sky_coverage=(
            SkyCoverage.FULL_SKY if region is None else SkyCoverage.PARTIAL_SKY
        ),
        coverage_fraction=(
            1.0
            if region is None
            else (
                coverage_footprint.coverage_fraction
                if coverage_footprint is not None
                else None
            )
        ),
        coverage_footprint=coverage_footprint,
        monopole_convention=MonopoleConvention.ABSOLUTE_NO_CMB,
        monopole_k=None,
        source_subtraction=SourceSubtractionStatus.NONE,
        notes=f"poisson_{model.name} (λ={expected_n:.1f}, N={n})",
    )
    sky = sky.replace(provenance=provenance)

    target = _coerce_format(representation)
    if target == SkyFormat.HEALPIX:
        if frequencies is None and obs_frequency_config is not None:
            from radiosim.utils.frequency import parse_frequency_config

            frequencies = parse_frequency_config(obs_frequency_config)
        if frequencies is None:
            frequencies = np.asarray([float(reference_frequency)])
        sky = materialize_healpix_model(
            sky,
            nside=nside,
            frequencies=np.asarray(frequencies, dtype=np.float64),
            ref_frequency=float(reference_frequency),
            memmap_path=memmap_path,
            clear_other=True,
        )

    return sky
