"""Extragalactic point-source foreground loader (Mittal et al. 2024).

Implements the isotropic mode of the extragalactic point-source population
model of Mittal, Kulkarni, Anstey & de Lera Acedo 2024 (MNRAS 534, 1317):
a Poisson realization of a validated differential source-count model
``dN/dS`` with per-source Gaussian spectral indices. The 2PACF-clustered
mode of the paper is a planned follow-up; this loader has no clustering
parameters yet.

Two deliberate deviations from the paper's reference implementation
(``epspy`` v1.0.1) are documented here once:

* Fluxes are drawn from the stated ``dN/dS`` with the correct integration
  measure (inverse-CDF sampling via :class:`~..support.dnds.DNDSModel`).
  ``epspy`` weights its discrete log-spaced flux grid by ``dN/dS`` alone —
  omitting the ``dS ∝ S`` cell widths — which tilts its realized flux
  distribution to ``dN/dS · S⁻¹`` and lowers the realized mean sky
  temperature (about 1.3 K instead of the ~17 K implied by the stated
  Gervasi et al. 2008 counts over 1 µJy–100 mJy at 150 MHz).
* Sources carry exact continuous sky positions instead of being accumulated
  at HEALPix pixel granularity.

The paper's temperature spectral index ``beta`` (``T ∝ nu^-beta``,
``beta ~ N(2.681, 0.5)``) maps to RadioSim's flux convention
``S(nu) = S0 (nu/nu0)^spectral_index`` as ``spectral_index = 2 - beta``;
indices are drawn unclipped, as in the paper.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

import numpy as np

from radiosim.core.precision import PrecisionConfig

from ..containers import (
    MonopoleConvention,
    SkyProvenance,
    SourceSubtractionStatus,
)
from ..containers.constants import (
    DEFAULT_EXTRAGALACTIC_SPECTRAL_INDEX_DIST,
    BrightnessConversion,
)
from ..containers.model import SkyFormat, SkyModel, _coerce_format
from ..operations.region import SkyRegion
from ..registry import loader_registry
from ..support.dnds import DNDSModel, resolve_dn_ds
from ..support.provenance_coverage import coverage_provenance

# Runtime (not TYPE_CHECKING) imports above are deliberate: the config layer
# resolves this loader's type hints with ``get_type_hints`` to validate YAML
# options generically, so every annotation must be importable at runtime.
from .synthetic import _approximate_region_area_sr, _sample_points_on_region

logger = logging.getLogger(__name__)

#: Default expected-count ceiling for the point-source representation. The
#: direct-sum RIME is O(N_src x N_bl x N_freq); populations beyond this size
#: need a brighter flux floor, a smaller region, or an explicit opt-in.
DEFAULT_MAX_SOURCES = 20_000_000


@loader_registry.register_loader(
    "extragalactic_point_sources",
    config_section="extragalactic_point_sources",
    use_flag="use_extragalactic_point_sources",
    category="synthetic",
    aliases={"eps": {}, "mittal2024": {}},
    config_fields=[
        "flux_range_jy",
        "reference_frequency",
        "dn_ds",
        "spectral_index_dist",
        "representation",
        "nside",
        "seed",
        "max_sources",
    ],
)
def load_extragalactic_point_sources(
    *,
    flux_range_jy: tuple[float, float] = (1e-2, 1e-1),
    reference_frequency: float = 150e6,
    dn_ds: str | DNDSModel = "gervasi2008_150mhz",
    spectral_index_dist: tuple[float, float] = (
        DEFAULT_EXTRAGALACTIC_SPECTRAL_INDEX_DIST
    ),
    region: SkyRegion | None = None,
    representation: str = "point_sources",
    nside: int = 64,
    frequencies: Sequence[float] | np.ndarray | None = None,
    seed: int | None = None,
    max_sources: int = DEFAULT_MAX_SOURCES,
    brightness_conversion: str = "rayleigh-jeans",
    precision: PrecisionConfig,
    memmap_path: str | None = None,
    provenance: SkyProvenance | None = None,
) -> SkyModel:
    """Sample an isotropic extragalactic point-source population.

    Realizes the Mittal et al. 2024 (MNRAS 534, 1317) extragalactic
    foreground model without clustering: the expected source count in the
    flux band ``flux_range_jy`` is ``λ = ∫ dN/dS · dS · Ω``; the actual
    count is drawn from ``Poisson(λ)``. Fluxes are sampled from the
    validated ``dN/dS`` PDF via inverse-CDF sampling, positions uniformly
    on the sphere (restricted by ``region`` if supplied), and spectral
    indices from an unclipped normal ``N(mean, σ)``.

    Parameters
    ----------
    flux_range_jy
        ``(S_min, S_max)`` band in Jy at ``reference_frequency``. The
        default ``(0.01, 0.1)`` matches the epspy package default; the
        paper's fiducial deep range is ``(1e-6, 0.1)``.
    reference_frequency
        Frequency (Hz) at which ``flux_range_jy`` is evaluated. Must match
        the model's calibration frequency exactly.
    dn_ds
        Validated preset name or validated :class:`DNDSModel`. The default
        ``"gervasi2008_150mhz"`` is the paper's fiducial choice;
        ``"mandal2021_lotss_150mhz"`` and ``"intema2017_tgss_150mhz"``
        are the paper's alternative forms.
    spectral_index_dist
        ``(mean, σ)`` of the per-source spectral-index draw in RadioSim's
        flux convention ``S(nu) = S0 (nu/nu0)^spectral_index``. The default
        ``(-0.681, 0.5)`` is the paper's temperature-index distribution
        ``beta ~ N(2.681, 0.5)`` mapped via ``spectral_index = 2 - beta``.
        Draws are not clipped.
    region
        Optional :class:`SkyRegion` restricting the position draw (and the
        expected count, via its solid angle).
    representation
        ``"point_sources"`` (default) or ``"healpix_map"``.
    nside, frequencies
        Forwarded to :func:`materialize_healpix_model` when
        ``representation == "healpix_map"``. ``frequencies`` is accepted
        and ignored for point output (the simulator injects it).
    seed
        RNG seed. Set for reproducible realizations; the resolved seed is
        always recorded in the provenance.
    max_sources
        Guardrail on the expected count ``λ``: exceeding it raises before
        anything is drawn. Deep flux ranges quickly reach 1e8-1e9 sources
        (the paper's fiducial range holds ~4.4e9), far beyond what the
        direct-sum RIME can consume as discrete sources.
    brightness_conversion
        Jy↔K convention for HEALPix materialization. Defaults to
        ``"rayleigh-jeans"`` as in Mittal et al. 2024 (indistinguishable
        from ``"planck"`` at these frequencies and temperatures).
    precision
        Standard RadioSim loader argument.
    memmap_path
        Forwarded to :func:`materialize_healpix_model`.
    provenance
        Optional replacement for the automatically built provenance.

    Returns
    -------
    SkyModel
        Sampled population with provenance
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
    alpha_mean, alpha_sigma = spectral_index_dist
    if not np.isfinite(alpha_mean) or not np.isfinite(alpha_sigma) or alpha_sigma < 0:
        raise ValueError(
            "spectral_index_dist must be a finite (mean, sigma) pair with "
            f"sigma >= 0; got {spectral_index_dist!r}."
        )
    if max_sources <= 0:
        raise ValueError(f"max_sources must be positive, got {max_sources!r}.")

    model = resolve_dn_ds(dn_ds)
    model_freq = model.reference_frequency_hz
    if not np.isclose(reference_frequency, model_freq, rtol=0.0, atol=1.0):
        raise ValueError(
            "load_extragalactic_point_sources requires reference_frequency to "
            "match the validated dN/dS calibration frequency exactly. "
            f"Requested {reference_frequency / 1e6:.3f} MHz, but model "
            f"'{model.name}' is calibrated at {model_freq / 1e6:.3f} MHz."
        )

    effective_area = _approximate_region_area_sr(region)
    if effective_area <= 0.0:
        raise ValueError(
            f"Effective area must be positive, got {effective_area!r}. "
            "Check the supplied region."
        )

    expected_n = (
        float(model.integrated_counts(flux_range_jy[0], flux_range_jy[1]))
        * effective_area
    )
    if expected_n > max_sources:
        raise ValueError(
            f"Expected source count λ={expected_n:.4g} exceeds "
            f"max_sources={max_sources}. Raise flux_range_jy[0], restrict "
            "the region, or raise max_sources explicitly."
        )

    # Resolve seed=None to a concrete drawn seed *before* use so the resulting
    # realization is reproducible from its own recorded provenance. RNG call
    # order is part of the reproducibility contract: count, fluxes, positions,
    # spectral indices.
    resolved_seed = seed if seed is not None else int(np.random.SeedSequence().entropy)
    rng = np.random.default_rng(resolved_seed)
    n = int(rng.poisson(expected_n))
    logger.info(
        "load_extragalactic_point_sources: dn_ds=%s, band=%.3g–%.3g Jy, "
        "area=%.3g sr, λ=%.1f, drew N=%d sources.",
        model.name,
        flux_range_jy[0],
        flux_range_jy[1],
        effective_area,
        expected_n,
        n,
    )

    model_name = f"extragalactic_{model.name}"
    if n == 0:
        # Empty realisation — build an empty point catalog with provenance.
        from ..operations.factories import create_empty

        coverage = coverage_provenance(is_full_sky=region is None, region=region)
        empty_prov = SkyProvenance(
            flux_completeness_jy=flux_range_jy,
            flux_completeness_freq_hz=float(reference_frequency),
            angular_resolution_rad=(0.0, float(np.pi)),
            sky_coverage=coverage.sky_coverage,
            coverage_fraction=coverage.coverage_fraction,
            coverage_footprint=coverage.coverage_footprint,
            monopole_convention=MonopoleConvention.ABSOLUTE_NO_CMB,
            monopole_k=None,
            source_subtraction=SourceSubtractionStatus.NONE,
            notes=f"{model_name}: isotropic Mittal et al. 2024 population",
            rng_seed=resolved_seed,
        )
        return create_empty(
            model_name,
            brightness,
            precision=precision,
            reference_frequency=float(reference_frequency),
            provenance=provenance if provenance is not None else empty_prov,
        )

    # Sample fluxes from the dN/dS PDF, positions uniformly on the sphere,
    # then unclipped Gaussian spectral indices (see module docstring).
    flux_jy = model.sample_flux(n, rng, flux_range_jy[0], flux_range_jy[1])
    ra_rad, dec_rad = _sample_points_on_region(n, region, rng)
    if alpha_sigma > 0:
        alpha = rng.normal(alpha_mean, alpha_sigma, size=n)
    else:
        alpha = np.full(n, float(alpha_mean))

    sky = create_from_arrays(
        ra_rad=ra_rad,
        dec_rad=dec_rad,
        flux=flux_jy,
        spectral_index=alpha,
        ref_freq=np.full(n, float(reference_frequency), dtype=np.float64),
        model_name=model_name,
        reference_frequency=float(reference_frequency),
        brightness_conversion=brightness,
        precision=precision,
    )

    coverage = coverage_provenance(is_full_sky=region is None, region=region)
    auto_provenance = SkyProvenance(
        flux_completeness_jy=(float(flux_range_jy[0]), float(flux_range_jy[1])),
        flux_completeness_freq_hz=float(reference_frequency),
        angular_resolution_rad=(0.0, float(np.pi)),
        sky_coverage=coverage.sky_coverage,
        coverage_fraction=coverage.coverage_fraction,
        coverage_footprint=coverage.coverage_footprint,
        monopole_convention=MonopoleConvention.ABSOLUTE_NO_CMB,
        monopole_k=None,
        source_subtraction=SourceSubtractionStatus.NONE,
        notes=(
            f"{model_name}: isotropic Mittal et al. 2024 population "
            f"(λ={expected_n:.1f}, N={n})"
        ),
        rng_seed=resolved_seed,
    )
    sky = sky.replace(
        provenance=provenance if provenance is not None else auto_provenance
    )

    target = _coerce_format(representation)
    if target == SkyFormat.HEALPIX:
        if frequencies is None:
            raise ValueError(
                "load_extragalactic_point_sources requires explicit "
                "'frequencies' in Hz for HEALPix output."
            )
        sky = materialize_healpix_model(
            sky,
            nside=nside,
            frequencies=np.asarray(frequencies, dtype=np.float64),
            ref_frequency=float(reference_frequency),
            memmap_path=memmap_path,
            clear_other=True,
        )

    return sky
