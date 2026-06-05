# radiosim/core/sky/recipes.py
"""High-level sky-model recipes that compose the RadioSim primitives into
physically-disjoint combinations suitable for visibility simulation.

The flagship recipe is :func:`realistic_foreground_sky`, which builds the
canonical three-layer low-frequency foreground described in the RadioSim
realistic-foreground research summary §8 (Jelić 2008 / SKA-Low 2025 style):

- a source-subtracted diffuse template (typically GSM2016 or Haslam);
- one or more bright point-source catalogs above a chosen flux threshold;
- a Poisson realisation of the sub-threshold confusion background filling
  the gap between the catalog's completeness limit and the diffuse-map
  subtraction threshold.

Each layer is tagged with explicit :class:`SkyProvenance` so that the
disjointness check inside :func:`prepare_sky_model` confirms no
double-counting and the combined sky carries a coherent monopole.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np

from ..combine.engine import MixedModelPolicy
from ..combine.pipeline import prepare_sky_model
from ..containers import SkyCoverage, SourceSubtractionStatus
from ..containers.model import SkyFormat, SkyModel
from ..registry.facade import loader_registry

if TYPE_CHECKING:
    from radiosim.core.precision import PrecisionConfig

    from ..operations.region import SkyRegion

logger = logging.getLogger(__name__)

_DEFAULT_ALPHA_FOR_THRESHOLD_SCALING = -0.7


def _scale_flux_with_alpha(
    flux_jy: float,
    from_freq_hz: float,
    to_freq_hz: float,
    alpha: float = _DEFAULT_ALPHA_FOR_THRESHOLD_SCALING,
) -> float:
    """Power-law flux scaling ``S · (ν_to / ν_from)^α``.  Clamps to input on
    non-positive frequencies."""
    if from_freq_hz <= 0.0 or to_freq_hz <= 0.0:
        return float(flux_jy)
    return float(flux_jy) * (to_freq_hz / from_freq_hz) ** alpha


def _load_bright_catalog(
    name: str,
    flux_min_jy: float,
    *,
    region: SkyRegion | None,
    precision: PrecisionConfig,
    brightness_conversion: str,
    extra_kwargs: dict[str, Any] | None,
) -> SkyModel:
    """Invoke a registered catalog loader with a flux floor.

    Different loaders use different kwarg names for the flux-floor
    parameter — VizieR / RACS catalogs use ``flux_limit``, synthetic
    loaders use ``flux_min``.  This helper inspects the loader's signature
    and injects only the kwargs it accepts; user-supplied
    ``extra_kwargs`` always take precedence.
    """
    import inspect

    resolved = loader_registry.resolve_callable(name)
    accepts: set[str]
    try:
        sig = inspect.signature(resolved.definition.loader)
        accepts = set(sig.parameters)
    except (TypeError, ValueError):
        accepts = set()

    kwargs: dict[str, Any] = {
        "brightness_conversion": brightness_conversion,
        "precision": precision,
    }
    if region is not None and "region" in accepts:
        kwargs["region"] = region
    if "flux_limit" in accepts:
        kwargs["flux_limit"] = float(flux_min_jy)
    elif "flux_min" in accepts:
        kwargs["flux_min"] = float(flux_min_jy)

    if extra_kwargs:
        kwargs.update(extra_kwargs)
    return resolved(**kwargs)


def _load_diffuse(
    name: str,
    *,
    nside: int,
    frequencies: np.ndarray,
    region: SkyRegion | None,
    include_cmb: bool,
    brightness_conversion: str,
    precision: PrecisionConfig,
    extra_kwargs: dict[str, Any] | None,
) -> SkyModel:
    loader = loader_registry.resolve_callable(name)
    kwargs: dict[str, Any] = {
        "nside": int(nside),
        "frequencies": np.asarray(frequencies, dtype=np.float64),
        "brightness_conversion": brightness_conversion,
        "precision": precision,
        "include_cmb": include_cmb,
    }
    if region is not None:
        kwargs["region"] = region
    if extra_kwargs:
        kwargs.update(extra_kwargs)
    return loader(**kwargs)


def _apply_recipe_coverage(
    sky: SkyModel,
    region: SkyRegion | None,
    *,
    nside: int | None = None,
) -> SkyModel:
    """Normalize recipe components to explicit full/partial sky coverage."""
    if region is None:
        return sky

    coverage_footprint = sky.provenance.coverage_footprint
    if coverage_footprint is not None:
        coverage_footprint = coverage_footprint.intersect_mask(
            region.healpix_mask(
                coverage_footprint.nside,
                coordinate_frame=coverage_footprint.coordinate_frame,
            )
        )
    elif sky.provenance.is_full_sky:
        coverage_footprint = region.footprint()
    elif sky.healpix is not None and nside is not None:
        coverage_footprint = None

    coverage_fraction = (
        coverage_footprint.coverage_fraction if coverage_footprint is not None else None
    )
    provenance = sky.provenance.replace(
        sky_coverage=SkyCoverage.PARTIAL_SKY,
        coverage_fraction=coverage_fraction,
        coverage_footprint=coverage_footprint,
        monopole_k=None,
    )
    return sky.replace(provenance=provenance)


def _require_scientific_diffuse_ready(
    diffuse_sky: SkyModel,
    *,
    diffuse_name: str,
    confusion_enabled: bool,
) -> None:
    """Reject diffuse inputs that are not already scientifically safe."""
    status = diffuse_sky.provenance.source_subtraction
    if confusion_enabled:
        if status is not SourceSubtractionStatus.ALL:
            raise ValueError(
                "confusion_flux_range_jy requires a strictly smooth diffuse "
                f"template with source_subtraction=ALL. Diffuse '{diffuse_name}' "
                f"declares {status.value!r}."
            )
        return

    if status not in (
        SourceSubtractionStatus.ALL,
        SourceSubtractionStatus.ABOVE_THRESHOLD,
    ):
        raise ValueError(
            "realistic_foreground_sky only accepts pre-subtracted diffuse "
            f"templates in scientific mode. Diffuse '{diffuse_name}' declares "
            f"source_subtraction={status.value!r}. Use a scientifically "
            "prepared template such as 'haslam', or compose the layers "
            "manually outside this recipe."
        )


def _check_threshold_chain(
    *,
    bright_catalog_flux_min_jy: float,
    bright_catalog_freq_hz: float,
    confusion_flux_range_jy: tuple[float, float] | None,
    diffuse_sky: SkyModel,
) -> None:
    """Validate that bright catalog / confusion / diffuse subtraction tile the
    flux axis without overlap.

    Raises
    ------
    ValueError
        If any adjacent layers overlap in flux at the catalog's reference
        frequency.
    """
    # First, the confusion band itself must be sane.
    if confusion_flux_range_jy is not None:
        s_conf_min, s_conf_max = confusion_flux_range_jy
        if s_conf_min <= 0.0 or s_conf_max <= s_conf_min:
            raise ValueError(
                f"confusion_flux_range_jy must satisfy 0 < s_min < s_max; "
                f"got {confusion_flux_range_jy!r}."
            )
        if s_conf_max > bright_catalog_flux_min_jy + 1e-12:
            raise ValueError(
                f"Threshold-chain violation: confusion band reaches "
                f"{s_conf_max:g} Jy at {bright_catalog_freq_hz / 1e6:.1f} MHz "
                f"but the bright catalog only covers down to "
                f"{bright_catalog_flux_min_jy:g} Jy. Raise "
                "bright_catalog_flux_min_jy or lower confusion_flux_range_jy[1]."
            )

    if (
        confusion_flux_range_jy is not None
        and confusion_flux_range_jy[1] < bright_catalog_flux_min_jy - 1e-12
    ):
        raise ValueError(
            "Threshold-chain violation: confusion band only reaches "
            f"{confusion_flux_range_jy[1]:g} Jy at "
            f"{bright_catalog_freq_hz / 1e6:.1f} MHz, leaving a gap below the "
            f"bright catalog floor of {bright_catalog_flux_min_jy:g} Jy."
        )

    diffuse_subtraction_threshold_jy = (
        diffuse_sky.provenance.source_subtraction_threshold_jy
        if diffuse_sky.provenance.source_subtraction
        is SourceSubtractionStatus.ABOVE_THRESHOLD
        else None
    )
    diffuse_subtraction_freq_hz = (
        diffuse_sky.provenance.source_subtraction_freq_hz
        if diffuse_sky.provenance.source_subtraction
        is SourceSubtractionStatus.ABOVE_THRESHOLD
        else None
    )
    if diffuse_subtraction_threshold_jy is not None:
        thresh_at_cat = _scale_flux_with_alpha(
            diffuse_subtraction_threshold_jy,
            from_freq_hz=diffuse_subtraction_freq_hz
            if diffuse_subtraction_freq_hz is not None
            else bright_catalog_freq_hz,
            to_freq_hz=bright_catalog_freq_hz,
        )
        if bright_catalog_flux_min_jy < thresh_at_cat - 1e-12:
            raise ValueError(
                "Threshold-chain violation: bright_catalog_flux_min_jy="
                f"{bright_catalog_flux_min_jy:g} Jy is below the diffuse "
                f"subtraction threshold scaled to the catalog's "
                f"{bright_catalog_freq_hz / 1e6:.1f} MHz reference "
                f"(~{thresh_at_cat:g} Jy).  Raise bright_catalog_flux_min_jy "
                "to at least the scaled threshold, or lower "
                "diffuse_subtraction_threshold_jy."
            )


@loader_registry.register_loader(
    "realistic_foreground",
    config_section="realistic_foreground",
    use_flag="use_realistic_foreground",
    representations=("healpix_map",),
    category="synthetic",
    config_fields={
        "diffuse": "diffuse",
        "diffuse_kwargs": "diffuse_kwargs",
        "bright_catalogs": "bright_catalogs",
        "bright_catalog_kwargs": "bright_catalog_kwargs",
        "bright_catalog_flux_min_jy": "bright_catalog_flux_min_jy",
        "confusion_flux_range_jy": "confusion_flux_range_jy",
        "confusion_dn_ds": "confusion_dn_ds",
        "confusion_spectral_index_dist": "confusion_spectral_index_dist",
        "nside": "nside",
        "include_cmb": "include_cmb",
        "seed": "seed",
        "mixed_model_policy": "mixed_model_policy",
    },
)
def realistic_foreground_sky(
    *,
    diffuse: str = "haslam",
    diffuse_kwargs: dict[str, Any] | None = None,
    bright_catalogs: str = "gleam",
    bright_catalog_kwargs: dict[str, Any] | None = None,
    bright_catalog_flux_min_jy: float = 2.0,
    confusion_flux_range_jy: tuple[float, float] | None = None,
    confusion_dn_ds: str = "franzen2019_gleam_154mhz",
    confusion_spectral_index_dist: tuple[float, float] = (-0.8, 0.2),
    frequencies: np.ndarray,
    nside: int = 128,
    region: SkyRegion | None = None,
    include_cmb: bool = False,
    seed: int | None = None,
    brightness_conversion: str = "planck",
    precision: PrecisionConfig,
    mixed_model_policy: MixedModelPolicy = "error",
    memmap_path: str | None = None,
) -> SkyModel:
    """Build a three-layer physically-disjoint foreground sky.

    The returned model is the composite

        B_total = diffuse + bright_catalogs + poisson_confusion

    with each layer tagged so that the disjointness check in
    :func:`prepare_sky_model` confirms no double-counting.

    Parameters
    ----------
    diffuse
        Registered diffuse loader name (``"gsm2016"``, ``"gsm2008"``,
        ``"haslam"``, ``"lfsm"``, ``"pysm3"`` …).
    diffuse_kwargs
        Extra keyword arguments forwarded to the diffuse loader.
    bright_catalogs
        Registered point-catalog loader name.  The catalog is loaded at
        ``flux_limit = bright_catalog_flux_min_jy``.
    bright_catalog_kwargs
        Extra keyword arguments forwarded to *every* bright-catalog loader.
    bright_catalog_flux_min_jy
        Flux floor (Jy) for the bright catalog(s) at their native reference
        frequencies.
    confusion_flux_range_jy
        Flux band (Jy) over which :func:`load_poisson_confusion` fills the
        sub-threshold population at the catalog's reference frequency.
        Defaults to ``None`` — no Poisson layer, the typical correct
        choice for observational diffuse maps (Haslam, GSM) that already
        contain the unresolved sub-threshold population as a smooth
        contribution.  Enable this layer ONLY when your diffuse template
        is *strictly smooth* (e.g. a synthetic Gaussian random field with
        ``source_subtraction=ALL`` declared); observational diffuse maps
        such as Haslam are rejected in this mode.
    confusion_dn_ds
        Validated dN/dS preset for the Poisson layer.
    confusion_spectral_index_dist
        ``(mean, σ)`` for per-source spectral-index draw in the Poisson layer.
    frequencies
        Observation frequency grid (Hz).
    nside
        Output HEALPix nside.
    region
        Optional sky region restricting every loader.
    include_cmb
        If True, the diffuse loader is asked to include the CMB directly.
    seed
        RNG seed forwarded to the Poisson loader.
    mixed_model_policy
        Forwarded to :func:`prepare_sky_model`.  Defaults to ``"error"`` —
        this recipe is constructed to be disjoint and the checker should
        pass without any tolerance.

    Returns
    -------
    SkyModel
        Combined model in ``SkyFormat.HEALPIX`` representation.

    Raises
    ------
    ValueError
        If the threshold chain is physically inconsistent, if the diffuse
        template is not already scientifically prepared, or if confusion is
        requested with a non-smooth diffuse layer.
    """
    freqs = np.asarray(frequencies, dtype=np.float64)
    if freqs.size == 0:
        raise ValueError("frequencies must be non-empty.")

    # --- 1) Diffuse layer ---
    diffuse_sky = _load_diffuse(
        diffuse,
        nside=nside,
        frequencies=freqs,
        region=region,
        include_cmb=include_cmb,
        brightness_conversion=brightness_conversion,
        precision=precision,
        extra_kwargs=diffuse_kwargs,
    )
    diffuse_sky = _apply_recipe_coverage(diffuse_sky, region, nside=nside)
    _require_scientific_diffuse_ready(
        diffuse_sky,
        diffuse_name=diffuse,
        confusion_enabled=confusion_flux_range_jy is not None,
    )

    # --- 2) Bright catalogs ---
    catalog_skies: list[SkyModel] = []
    catalog_freqs: list[float] = []
    cat_sky = _load_bright_catalog(
        bright_catalogs,
        flux_min_jy=bright_catalog_flux_min_jy,
        region=region,
        precision=precision,
        brightness_conversion=brightness_conversion,
        extra_kwargs=bright_catalog_kwargs,
    )
    cat_sky = _apply_recipe_coverage(cat_sky, region, nside=nside)
    catalog_skies.append(cat_sky)
    nu = cat_sky.provenance.flux_completeness_freq_hz or cat_sky.reference_frequency
    if nu is None or nu <= 0:
        nu = float(freqs[0])
    catalog_freqs.append(float(nu))

    # --- 3) Threshold-chain check (per catalog frequency) ---
    _check_threshold_chain(
        bright_catalog_flux_min_jy=bright_catalog_flux_min_jy,
        bright_catalog_freq_hz=catalog_freqs[0],
        confusion_flux_range_jy=confusion_flux_range_jy,
        diffuse_sky=diffuse_sky,
    )

    # --- 4) Poisson confusion layer ---
    confusion_skies: list[SkyModel] = []
    if confusion_flux_range_jy is not None:
        # Use the first catalog's reference frequency as the confusion
        # reference — simplest and most physically meaningful choice.
        conf_freq = catalog_freqs[0]
        from ..loaders.synthetic import load_poisson_confusion

        conf_sky = load_poisson_confusion(
            flux_range_jy=tuple(confusion_flux_range_jy),
            reference_frequency=conf_freq,
            dn_ds=confusion_dn_ds,
            region=region,
            representation="point_sources",
            seed=seed,
            spectral_index_dist=confusion_spectral_index_dist,
            brightness_conversion=brightness_conversion,
            precision=precision,
        )
        conf_sky = _apply_recipe_coverage(conf_sky, region, nside=nside)
        if conf_sky.n_point_sources > 0:
            confusion_skies.append(conf_sky)
        else:
            logger.info(
                "realistic_foreground_sky: Poisson confusion drew 0 sources — "
                "skipping empty layer."
            )

    # --- 5) Combine via prepare_sky_model (healpix_map representation) ---
    components = [diffuse_sky, *catalog_skies, *confusion_skies]
    combined = prepare_sky_model(
        components,
        representation=SkyFormat.HEALPIX,
        nside=nside,
        frequencies=freqs,
        mixed_model_policy=mixed_model_policy,
        brightness_conversion=brightness_conversion,
        precision=precision,
        memmap_path=memmap_path,
    )

    return combined


__all__ = ["realistic_foreground_sky"]
