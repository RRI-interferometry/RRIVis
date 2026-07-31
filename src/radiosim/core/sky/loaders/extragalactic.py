"""Extragalactic point-source foreground loader (Mittal et al. 2024).

Implements the extragalactic point-source population model of Mittal,
Kulkarni, Anstey & de Lera Acedo 2024 (MNRAS 534, 1317): source counts
from a validated differential source-count model ``dN/dS``, per-source
Gaussian spectral indices, and (by default) angular clustering following
a power-law 2-point angular correlation function with the Rana & Bagla
2019 TGSS parameters (the paper's eqs. 5-9). Setting ``clustering_amp=0``
gives the isotropic mode.

Deliberate deviations from the paper's reference implementation
(``epspy`` v1.0.1) are documented here once:

* Fluxes are drawn from the stated ``dN/dS`` with the correct integration
  measure (inverse-CDF sampling via :class:`~..support.dnds.DNDSModel`).
  ``epspy`` weights its discrete log-spaced flux grid by ``dN/dS`` alone —
  omitting the ``dS ∝ S`` cell widths — which tilts its realized flux
  distribution to ``dN/dS · S⁻¹`` and lowers the realized mean sky
  temperature (about 1.3 K instead of the ~17 K implied by the stated
  Gervasi et al. 2008 counts over 1 µJy–100 mJy at 150 MHz).
* In the ``point_sources`` representation, sources carry continuous sky
  positions (clustered draws are dithered to 1/256 of the pixel scale)
  instead of being accumulated at HEALPix pixel granularity.
* Clustered per-pixel counts are Poisson-sampled from
  ``λ_p = nbar (1 + δ_p)`` (``epspy`` rounds the rate field, discarding
  shot noise), negative rates are clipped with a warning instead of
  terminating the process, and the clustering realization resolves
  multipoles up to ``3 nside - 1`` by default (``epspy`` is silently
  band-limited to ``ell <= 199``); see
  :mod:`radiosim.core.sky.support.clustering` for the monopole convention.

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
    flux_density_to_brightness_temp,
)
from ..containers.model import SkyFormat, SkyModel, _coerce_format
from ..operations.region import SkyRegion
from ..registry import loader_registry
from ..support.clustering import (
    clustered_pixel_rates,
    dither_positions_in_pixels,
    gaussian_overdensity_map,
    power_law_acf_to_cl,
)
from ..support.dnds import DNDSModel, resolve_dn_ds
from ..support.healpy import lazy_healpy as hp
from ..support.provenance_coverage import coverage_provenance

# Runtime (not TYPE_CHECKING) imports above are deliberate: the config layer
# resolves this loader's type hints with ``get_type_hints`` to validate YAML
# options generically, so every annotation must be importable at runtime.
from .synthetic import _approximate_region_area_sr, _sample_points_on_region

logger = logging.getLogger(__name__)

#: Default expected-count ceiling for the point-source representation. The
#: direct-sum RIME is O(N_src x N_bl x N_freq); populations beyond this size
#: need a brighter flux floor, a smaller region, an explicit opt-in, or the
#: streamed ``healpix_map`` representation (which has no count ceiling).
DEFAULT_MAX_SOURCES = 20_000_000

#: Power-law 2PACF fiducials of Mittal et al. 2024: Rana & Bagla 2019
#: (MNRAS 485, 5891) fitted to TGSS ADR1 at 150 MHz.
DEFAULT_CLUSTERING_AMP = 7.8e-3
DEFAULT_CLUSTERING_GAMMA = 0.821

#: Source-block size for the streamed HEALPix accumulation. Part of the
#: seeded-reproducibility contract for the ``healpix_map`` representation:
#: per-chunk RNG draws depend on how pixels are grouped into chunks.
_STREAM_CHUNK_SOURCES = 5_000_000


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
        "clustering_amp",
        "clustering_gamma",
        "clustering_lmax",
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
    clustering_amp: float = DEFAULT_CLUSTERING_AMP,
    clustering_gamma: float = DEFAULT_CLUSTERING_GAMMA,
    clustering_lmax: int | None = None,
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
    """Sample an extragalactic point-source population (Mittal et al. 2024).

    The expected source count in the flux band ``flux_range_jy`` is
    ``λ = ∫ dN/dS · dS · Ω``. With clustering enabled (the default), a
    Gaussian overdensity map ``δ`` realizes the power-law 2PACF and
    per-pixel counts are drawn from ``Poisson(nbar (1 + δ_p))``; with
    ``clustering_amp=0`` the population is isotropic and the total count
    is one ``Poisson(λ)`` draw. Fluxes are sampled from the validated
    ``dN/dS`` PDF via inverse-CDF sampling and spectral indices from an
    unclipped normal ``N(mean, σ)``.

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
    clustering_amp, clustering_gamma
        Power-law 2PACF ``C(chi) = A (chi/deg)^-gamma``. Defaults are the
        paper's Rana & Bagla 2019 TGSS values ``A=7.8e-3``,
        ``gamma=0.821``. Set ``clustering_amp=0`` for an isotropic sky.
    clustering_lmax
        Highest multipole of the clustering realization. Defaults to
        ``3 * nside - 1``.
    region
        Optional :class:`SkyRegion` restricting the population (and the
        expected count, via its solid angle). Clustered draws populate
        pixels whose centers lie in the region and drop the few dithered
        positions that land outside it.
    representation
        ``"point_sources"`` (default) or ``"healpix_map"``. The HEALPix
        form streams sources directly into ``(n_freq, n_pix)`` brightness
        maps in pixel-size chunks, never materializing per-source arrays,
        so it supports arbitrarily deep flux ranges (the paper's fiducial
        ~4.4e9 sources included) at pixel granularity.
    nside
        HEALPix resolution of the clustering realization and of the
        ``healpix_map`` output.
    frequencies
        Map frequencies in Hz; required for ``healpix_map`` output and
        ignored for point output (the simulator injects it).
    seed
        RNG seed. Set for reproducible realizations; the resolved seed is
        always recorded in the provenance. Draw order is part of the
        contract and differs per mode: isotropic point draws count, then
        fluxes, positions, spectral indices; clustered point draws alms,
        counts, dithers, fluxes, spectral indices; the streamed HEALPix
        form draws [alms,] counts, then per-chunk fluxes and indices.
    max_sources
        Guardrail on the expected count ``λ`` for the ``point_sources``
        representation: exceeding it raises before anything is drawn.
        Deep flux ranges quickly reach 1e8-1e9 sources, far beyond what
        the direct-sum RIME can consume as discrete sources. The streamed
        ``healpix_map`` representation is exempt.
    brightness_conversion
        Jy↔K convention for HEALPix output. Defaults to
        ``"rayleigh-jeans"`` as in Mittal et al. 2024 (indistinguishable
        from ``"planck"`` at these frequencies and temperatures).
    precision
        Standard RadioSim loader argument.
    memmap_path
        Scratch directory for HEALPix map allocation.
    provenance
        Optional replacement for the automatically built provenance.

    Returns
    -------
    SkyModel
        Sampled population with provenance
        ``flux_completeness_jy=flux_range_jy``,
        ``source_subtraction=NONE``, ``monopole_convention=ABSOLUTE_NO_CMB``.
    """
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
    if not np.isfinite(clustering_amp) or clustering_amp < 0.0:
        raise ValueError(
            f"clustering_amp must be finite and >= 0, got {clustering_amp!r}."
        )
    clustered = clustering_amp > 0.0
    if clustered and (
        not np.isfinite(clustering_gamma) or not 0.0 < clustering_gamma < 2.0
    ):
        raise ValueError(
            f"clustering_gamma must satisfy 0 < gamma < 2, got {clustering_gamma!r}."
        )
    if clustering_lmax is not None and clustering_lmax < 1:
        raise ValueError(
            f"clustering_lmax must be at least 1, got {clustering_lmax!r}."
        )

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

    counts_per_sr = float(model.integrated_counts(flux_range_jy[0], flux_range_jy[1]))
    expected_n = counts_per_sr * effective_area
    target = _coerce_format(representation)
    if target != SkyFormat.HEALPIX and expected_n > max_sources:
        raise ValueError(
            f"Expected source count λ={expected_n:.4g} exceeds "
            f"max_sources={max_sources}. Raise flux_range_jy[0], restrict "
            "the region, raise max_sources explicitly, or use the streamed "
            "representation='healpix_map'."
        )

    # Resolve seed=None to a concrete drawn seed *before* use so the resulting
    # realization is reproducible from its own recorded provenance.
    resolved_seed = seed if seed is not None else int(np.random.SeedSequence().entropy)
    rng = np.random.default_rng(resolved_seed)

    model_name = f"extragalactic_{model.name}"
    mode_note = (
        f"clustered 2PACF (A={clustering_amp:g}, gamma={clustering_gamma:g})"
        if clustered
        else "isotropic"
    )

    if target == SkyFormat.HEALPIX:
        if frequencies is None:
            raise ValueError(
                "load_extragalactic_point_sources requires explicit "
                "'frequencies' in Hz for HEALPix output."
            )
        return _stream_healpix_model(
            model=model,
            model_name=model_name,
            mode_note=mode_note,
            flux_range_jy=flux_range_jy,
            reference_frequency=float(reference_frequency),
            counts_per_sr=counts_per_sr,
            expected_n=expected_n,
            alpha_mean=float(alpha_mean),
            alpha_sigma=float(alpha_sigma),
            clustered=clustered,
            clustering_amp=float(clustering_amp),
            clustering_gamma=float(clustering_gamma),
            clustering_lmax=clustering_lmax,
            region=region,
            nside=nside,
            frequencies=np.asarray(frequencies, dtype=np.float64),
            rng=rng,
            resolved_seed=resolved_seed,
            brightness=brightness,
            precision=precision,
            memmap_path=memmap_path,
            provenance_override=provenance,
        )

    # ---- point_sources representation -------------------------------------
    if clustered:
        # RNG order: alms, per-pixel counts, sub-pixel dithers, fluxes,
        # spectral indices.
        lmax = clustering_lmax if clustering_lmax is not None else 3 * nside - 1
        cl = power_law_acf_to_cl(clustering_amp, clustering_gamma, lmax)
        delta = gaussian_overdensity_map(cl, nside, rng)
        nbar_pix = counts_per_sr * float(hp.nside2pixarea(nside))
        rates = clustered_pixel_rates(nbar_pix, delta)
        if region is not None:
            mask = np.asarray(
                region.healpix_mask(nside, coordinate_frame="icrs"), dtype=bool
            )
            rates = np.where(mask, rates, 0.0)
        counts = rng.poisson(rates)
        pixels = np.repeat(np.arange(counts.size, dtype=np.int64), counts)
        ra_rad, dec_rad = dither_positions_in_pixels(pixels, nside, rng)
        if region is not None and ra_rad.size:
            keep = region.contains(ra_rad, dec_rad)
            n_dropped = int(ra_rad.size - np.count_nonzero(keep))
            if n_dropped:
                logger.info(
                    "Dropped %d boundary-dithered sources outside the region.",
                    n_dropped,
                )
            ra_rad, dec_rad = ra_rad[keep], dec_rad[keep]
        n = int(ra_rad.size)
    else:
        # RNG order: count, fluxes, positions, spectral indices.
        n = int(rng.poisson(expected_n))

    logger.info(
        "load_extragalactic_point_sources: dn_ds=%s, band=%.3g–%.3g Jy, "
        "area=%.3g sr, %s, λ=%.1f, drew N=%d sources.",
        model.name,
        flux_range_jy[0],
        flux_range_jy[1],
        effective_area,
        mode_note,
        expected_n,
        n,
    )

    if n == 0:
        return _empty_point_model(
            model_name=model_name,
            mode_note=mode_note,
            flux_range_jy=flux_range_jy,
            reference_frequency=float(reference_frequency),
            region=region,
            resolved_seed=resolved_seed,
            brightness=brightness,
            precision=precision,
            provenance_override=provenance,
        )

    from ..operations.factories import create_from_arrays

    flux_jy = model.sample_flux(n, rng, flux_range_jy[0], flux_range_jy[1])
    if not clustered:
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

    auto_provenance = _build_provenance(
        flux_range_jy=flux_range_jy,
        reference_frequency=float(reference_frequency),
        region=region,
        resolved_seed=resolved_seed,
        notes=(
            f"{model_name}: {mode_note} Mittal et al. 2024 population "
            f"(λ={expected_n:.1f}, N={n})"
        ),
    )
    return sky.replace(
        provenance=provenance if provenance is not None else auto_provenance
    )


def _build_provenance(
    *,
    flux_range_jy: tuple[float, float],
    reference_frequency: float,
    region: SkyRegion | None,
    resolved_seed: int,
    notes: str,
) -> SkyProvenance:
    coverage = coverage_provenance(is_full_sky=region is None, region=region)
    return SkyProvenance(
        flux_completeness_jy=(float(flux_range_jy[0]), float(flux_range_jy[1])),
        flux_completeness_freq_hz=reference_frequency,
        angular_resolution_rad=(0.0, float(np.pi)),
        sky_coverage=coverage.sky_coverage,
        coverage_fraction=coverage.coverage_fraction,
        coverage_footprint=coverage.coverage_footprint,
        monopole_convention=MonopoleConvention.ABSOLUTE_NO_CMB,
        monopole_k=None,
        source_subtraction=SourceSubtractionStatus.NONE,
        notes=notes,
        rng_seed=resolved_seed,
    )


def _empty_point_model(
    *,
    model_name: str,
    mode_note: str,
    flux_range_jy: tuple[float, float],
    reference_frequency: float,
    region: SkyRegion | None,
    resolved_seed: int,
    brightness: BrightnessConversion,
    precision: PrecisionConfig,
    provenance_override: SkyProvenance | None,
) -> SkyModel:
    """Empty realisation — an empty point catalog with full provenance."""
    from ..operations.factories import create_empty

    empty_prov = _build_provenance(
        flux_range_jy=flux_range_jy,
        reference_frequency=reference_frequency,
        region=region,
        resolved_seed=resolved_seed,
        notes=f"{model_name}: {mode_note} Mittal et al. 2024 population",
    )
    return create_empty(
        model_name,
        brightness,
        precision=precision,
        reference_frequency=reference_frequency,
        provenance=provenance_override
        if provenance_override is not None
        else (empty_prov),
    )


def _stream_healpix_model(
    *,
    model: DNDSModel,
    model_name: str,
    mode_note: str,
    flux_range_jy: tuple[float, float],
    reference_frequency: float,
    counts_per_sr: float,
    expected_n: float,
    alpha_mean: float,
    alpha_sigma: float,
    clustered: bool,
    clustering_amp: float,
    clustering_gamma: float,
    clustering_lmax: int | None,
    region: SkyRegion | None,
    nside: int,
    frequencies: np.ndarray,
    rng: np.random.Generator,
    resolved_seed: int,
    brightness: BrightnessConversion,
    precision: PrecisionConfig,
    memmap_path: str | None,
    provenance_override: SkyProvenance | None,
) -> SkyModel:
    """Stream sources into ``(n_freq, n_pix)`` brightness-temperature maps.

    Sources are drawn per pixel and accumulated chunk-by-chunk (at most
    ``_STREAM_CHUNK_SOURCES`` per block), so per-source arrays never exceed
    the chunk size regardless of the population depth. Output pixels are
    RING-ordered Kelvin maps in the ICRS frame, at pixel granularity (the
    positional information inside a pixel is not represented, matching the
    epspy map product).
    """
    from ._healpix_builder import build_healpix_from_stokes_cube

    npix = int(hp.nside2npix(nside))
    pixarea = float(hp.nside2pixarea(nside))
    nbar_pix = counts_per_sr * pixarea

    # RNG order: [alms if clustered,] per-pixel counts, then per-chunk
    # fluxes and spectral indices.
    if clustered:
        lmax = clustering_lmax if clustering_lmax is not None else 3 * nside - 1
        cl = power_law_acf_to_cl(clustering_amp, clustering_gamma, lmax)
        delta = gaussian_overdensity_map(cl, nside, rng)
        rates = clustered_pixel_rates(nbar_pix, delta)
    else:
        rates = np.full(npix, nbar_pix, dtype=np.float64)
    if region is not None:
        mask = np.asarray(
            region.healpix_mask(nside, coordinate_frame="icrs"), dtype=bool
        )
        rates = np.where(mask, rates, 0.0)

    counts = rng.poisson(rates)
    total = int(counts.sum())
    n_freq = int(frequencies.size)
    cube = np.zeros((n_freq, npix), dtype=np.float64)
    ratio = frequencies / reference_frequency

    cumulative = np.cumsum(counts)
    start = 0
    while start < npix:
        base = int(cumulative[start - 1]) if start else 0
        # Largest end with (cumulative[end-1] - base) <= chunk; a single
        # pixel holding more than one chunk still advances by one pixel.
        end = int(
            np.searchsorted(cumulative, base + _STREAM_CHUNK_SOURCES, side="right")
        )
        end = min(max(end, start + 1), npix)
        block = counts[start:end]
        n_block = int(block.sum())
        if n_block:
            local_pix = np.repeat(np.arange(end - start, dtype=np.int64), block)
            flux = model.sample_flux(n_block, rng, flux_range_jy[0], flux_range_jy[1])
            if alpha_sigma > 0:
                alpha = rng.normal(alpha_mean, alpha_sigma, size=n_block)
            else:
                alpha = np.full(n_block, alpha_mean)
            for fi in range(n_freq):
                weights = flux * np.power(ratio[fi], alpha)
                cube[fi, start:end] += np.bincount(
                    local_pix, weights=weights, minlength=end - start
                )
        start = end

    logger.info(
        "load_extragalactic_point_sources: dn_ds=%s, band=%.3g–%.3g Jy, %s, "
        "λ=%.1f, streamed N=%d sources into %d x %d HEALPix maps.",
        model.name,
        flux_range_jy[0],
        flux_range_jy[1],
        mode_note,
        expected_n,
        total,
        n_freq,
        npix,
    )

    def _stokes_rows():
        for fi in range(n_freq):
            flux_row = cube[fi]
            frequency = float(frequencies[fi])
            if brightness is BrightnessConversion.PLANCK:
                # The exact Planck conversion rejects S <= 0; empty pixels
                # are exactly 0 Jy and stay exactly 0 K.
                temp = np.zeros_like(flux_row)
                positive = flux_row > 0.0
                if np.any(positive):
                    temp[positive] = flux_density_to_brightness_temp(
                        flux_row[positive], frequency, pixarea, brightness
                    )
            else:
                temp = flux_density_to_brightness_temp(
                    flux_row, frequency, pixarea, brightness
                )
            yield (temp,)

    healpix = build_healpix_from_stokes_cube(
        stokes_rows=_stokes_rows(),
        nside=nside,
        frequencies=frequencies,
        coordinate_frame="icrs",
        region=region,
        precision=precision,
        memmap_path=memmap_path,
        ordering="ring",
    )

    auto_provenance = _build_provenance(
        flux_range_jy=flux_range_jy,
        reference_frequency=reference_frequency,
        region=region,
        resolved_seed=resolved_seed,
        notes=(
            f"{model_name}: {mode_note} Mittal et al. 2024 population, "
            f"streamed HEALPix accumulation (λ={expected_n:.1f}, N={total})"
        ),
    )
    return SkyModel(
        healpix=healpix,
        model_name=model_name,
        brightness_conversion=brightness,
        provenance=(
            provenance_override if provenance_override is not None else auto_provenance
        ),
        precision=precision,
    )
