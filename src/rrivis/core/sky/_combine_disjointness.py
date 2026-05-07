# rrivis/core/sky/_combine_disjointness.py
"""Physical disjointness rules for combined sky models.

These checks live in their own module because the rule set is the most
likely future-churn area: new diffuse/point catalogs may need new
provenance signals, scale-separation thresholds, or alpha defaults.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Literal

from ._data import MonopoleConvention, SourceSubtractionStatus
from .constants import BrightnessConversion
from .model import SkyFormat

if TYPE_CHECKING:
    from .model import SkyModel


MixedModelPolicy = Literal["error", "warn", "allow"]


_DEFAULT_SUBTRACTION_SCALING_ALPHA = -0.7


def _scale_threshold_to_frequency(
    threshold_jy: float,
    from_freq_hz: float,
    to_freq_hz: float,
    alpha: float = _DEFAULT_SUBTRACTION_SCALING_ALPHA,
) -> float:
    """Scale a flux threshold from one reference frequency to another.

    Uses a simple power-law ``(to_freq/from_freq)**alpha``.  Returns the
    threshold unchanged if either frequency is non-positive (robust
    fallback).
    """
    if from_freq_hz <= 0.0 or to_freq_hz <= 0.0:
        return float(threshold_jy)
    return float(threshold_jy) * (to_freq_hz / from_freq_hz) ** alpha


def classify_model(sky: SkyModel) -> frozenset[SkyFormat]:
    """Return the set of populated payloads on this model.

    Hybrid models return a frozenset containing both
    :data:`SkyFormat.POINT_SOURCES` and :data:`SkyFormat.HEALPIX`. Empty
    point payloads (zero sources) are not considered populated.
    """
    formats: set[SkyFormat] = set()
    if sky.point is not None and not sky.point.is_empty:
        formats.add(SkyFormat.POINT_SOURCES)
    if sky.healpix is not None:
        formats.add(SkyFormat.HEALPIX)
    return frozenset(formats)


def _disjoint_pair_failures(
    diffuse: SkyModel,
    point: SkyModel,
    *,
    alpha: float,
) -> list[str]:
    """Return a list of human-readable reasons why *diffuse* + *point* overlap.

    Empty list means the pair is disjoint (physically safe to combine).  A
    non-empty list is returned when *every* one of the three pass rules
    fails — each entry reports one failed rule with numerics.
    """
    reasons: list[str] = []
    d_prov = diffuse.provenance
    p_prov = point.provenance

    # Rule 2.1 — diffuse is fully source-subtracted.
    if d_prov.source_subtraction == SourceSubtractionStatus.ALL:
        return []

    # Rule 2.2 — diffuse is source-subtracted at S* ≥ catalog completeness min.
    if d_prov.source_subtraction == SourceSubtractionStatus.ABOVE_THRESHOLD:
        t_d = d_prov.source_subtraction_threshold_jy
        nu_d = d_prov.source_subtraction_freq_hz
        completeness = p_prov.flux_completeness_jy
        nu_p = p_prov.flux_completeness_freq_hz
        if (
            t_d is not None
            and completeness is not None
            and nu_d is not None
            and nu_p is not None
        ):
            t_d_at_p = _scale_threshold_to_frequency(t_d, nu_d, nu_p, alpha=alpha)
            if t_d_at_p <= completeness[0]:
                return []  # diffuse subtraction tiles the flux axis below catalog
            reasons.append(
                f"diffuse '{diffuse.model_name}' is source-subtracted at "
                f"{t_d:g} Jy@{nu_d / 1e6:.1f} MHz "
                f"(scaled α={alpha:+.2f} to {t_d_at_p:g} Jy @ "
                f"{nu_p / 1e6:.1f} MHz), but catalog '{point.model_name}' "
                f"completeness starts at {completeness[0]:g} Jy — sources in "
                f"({completeness[0]:g}, {t_d_at_p:g}] Jy are double-counted."
            )
        else:
            reasons.append(
                f"diffuse '{diffuse.model_name}' declares ABOVE_THRESHOLD but "
                "threshold / completeness metadata is incomplete (cannot verify)."
            )
    elif d_prov.source_subtraction == SourceSubtractionStatus.NONE:
        reasons.append(
            f"diffuse '{diffuse.model_name}' has source_subtraction=NONE — it "
            f"still contains the bright extragalactic population that "
            f"catalog '{point.model_name}' also supplies."
        )
    else:
        reasons.append(
            f"diffuse '{diffuse.model_name}' has source_subtraction=UNKNOWN "
            "(declare provenance.source_subtraction to verify disjointness)."
        )

    # Rule 2.3 — angular-scale disjointness (scale separation).
    if (
        d_prov.angular_resolution_rad is not None
        and p_prov.angular_resolution_rad is not None
    ):
        d_theta_max = d_prov.angular_resolution_rad[1]
        p_theta_min = p_prov.angular_resolution_rad[0]
        if d_theta_max < p_theta_min:
            return []  # scale-separated by construction
        reasons.append(
            f"angular-scale ranges overlap: diffuse θ_max="
            f"{d_theta_max:.3g} rad ≥ point θ_min={p_theta_min:.3g} rad "
            "(not a valid scale-separation recipe)."
        )
    else:
        reasons.append(
            "angular-resolution metadata missing on at least one model "
            "(cannot verify scale separation)."
        )

    return reasons


def _check_monopole_consistency(models: list[SkyModel]) -> None:
    """Raise if ``monopole_convention`` is incompatible across inputs.

    Incompatible = two conventions drawn from
    ``{ABSOLUTE_WITH_CMB, ABSOLUTE_NO_CMB, MEAN_SUBTRACTED}`` such that their
    combination is mathematically wrong.  ``UNKNOWN`` is tolerated here and
    flagged separately by the disjointness checker.
    """
    declared = [
        m.provenance.monopole_convention
        for m in models
        if m.provenance.monopole_convention != MonopoleConvention.UNKNOWN
    ]
    incompat_pairs = {
        (MonopoleConvention.ABSOLUTE_WITH_CMB, MonopoleConvention.MEAN_SUBTRACTED),
        (MonopoleConvention.ABSOLUTE_NO_CMB, MonopoleConvention.MEAN_SUBTRACTED),
    }
    for i, conv_i in enumerate(declared):
        for conv_j in declared[i + 1 :]:
            pair = tuple(sorted((conv_i, conv_j), key=lambda c: c.value))
            if tuple(pair) in incompat_pairs:
                raise ValueError(
                    "Cannot combine sky models with incompatible monopole "
                    f"conventions: {conv_i.value!r} and {conv_j.value!r}. "
                    "Mean-subtract the absolute model (with_monopole_subtracted) "
                    "or add the monopole back to the mean-subtracted model "
                    "(with_monopole) before combining."
                )


def check_physical_disjointness(
    models: list[SkyModel],
    mixed_model_policy: MixedModelPolicy,
    *,
    alpha: float = _DEFAULT_SUBTRACTION_SCALING_ALPHA,
) -> None:
    """Validate that ``models`` can be physically summed without double-counting.

    Implements the three pass rules from the realistic-foreground research:

    1. Diffuse is fully source-subtracted (``SourceSubtractionStatus.ALL``).
    2. Diffuse is source-subtracted above a threshold that, scaled to the
       point catalog's reference frequency, sits at or below the catalog's
       flux-completeness minimum (layers tile the flux axis disjointly).
    3. Diffuse's maximum angular scale is strictly below the point model's
       minimum angular scale (scale-separation recipe).

    Monopole-convention compatibility is checked separately and raises
    under every policy when violated (numerically wrong, not merely suspect).

    The ``mixed_model_policy`` argument controls behavior on failure:

    - ``"error"`` — raise ``ValueError`` with the full diagnostic.  Unknown
      provenance on any cross-type pair counts as a failure (fail-closed).
    - ``"warn"``  — emit a ``UserWarning`` with the diagnostic and continue.
    - ``"allow"`` — suppress (caller asserts responsibility).

    Parameters
    ----------
    models
        Models about to be combined.
    mixed_model_policy
        Enforcement level.
    alpha
        Power-law spectral index used to scale source-subtraction thresholds
        between the diffuse-map reference frequency and the catalog's
        completeness frequency.  Default −0.7.
    """
    # Monopole consistency is unambiguously wrong — always enforce.
    _check_monopole_consistency(models)

    if mixed_model_policy == "allow":
        return

    # Pair each diffuse model against each point model and collect failures.
    diffuse_only = frozenset({SkyFormat.HEALPIX})
    diffuse_models = [m for m in models if classify_model(m) == diffuse_only]
    point_models = [m for m in models if SkyFormat.POINT_SOURCES in classify_model(m)]

    if not diffuse_models or not point_models:
        return  # same-type combinations: no point-vs-diffuse overlap possible

    all_reasons: list[str] = []
    for d in diffuse_models:
        for p in point_models:
            pair_reasons = _disjoint_pair_failures(d, p, alpha=alpha)
            if pair_reasons:
                all_reasons.append(
                    f"[{d.model_name} + {p.model_name}]: " + "; ".join(pair_reasons)
                )

    if not all_reasons:
        return

    header = (
        "Sky models are not physically disjoint — combining them would lead "
        "to double-counting of sources. Each diffuse+point pair below failed "
        "all three disjointness rules (source-subtracted ≥ catalog_min OR "
        "angular disjoint OR fully subtracted):"
    )
    hint = (
        "\nFix: (a) use a source-subtracted diffuse template "
        "(rrivis.core.sky.operations.subtract_bright_sources or a "
        "pre-subtracted catalog like 'haslam'), (b) raise the catalog's "
        "flux_limit above the diffuse threshold, (c) declare disjointness "
        "via SkyProvenance on each model, or (d) set "
        "sky_model.mixed_model_policy='warn' or 'allow' to override."
    )
    message = header + "\n  - " + "\n  - ".join(all_reasons) + hint
    if mixed_model_policy == "error":
        raise ValueError(message)
    warnings.warn(message, UserWarning, stacklevel=3)


def resolve_brightness_conversion(
    models: list[SkyModel],
    requested: BrightnessConversion | str | None,
) -> BrightnessConversion:
    """Resolve output brightness conversion without silent clobbering."""
    values = {m.brightness_conversion for m in models}
    if requested is None:
        if not values:
            return BrightnessConversion.PLANCK
        if len(values) == 1:
            return next(iter(values))
        raise ValueError(
            "Cannot combine sky models with different brightness_conversion "
            f"settings without an explicit brightness_conversion target: {values}."
        )
    return BrightnessConversion(requested)


def resolve_combination_params(
    models: list[SkyModel],
    representation: SkyFormat | str | None,
    frequency: float | None,
    ref_frequency: float | None,
) -> tuple[SkyFormat, float | None, float | None]:
    """Auto-detect representation and resolve frequency defaults.

    Returns (representation, frequency, ref_frequency).
    """
    # Coerce string to SkyFormat
    if isinstance(representation, str) and not isinstance(representation, SkyFormat):
        representation = SkyFormat(representation)

    # Auto-detect representation
    if representation is None:
        representation = (
            SkyFormat.HEALPIX
            if any(m.healpix is not None for m in models)
            else SkyFormat.POINT_SOURCES
        )

    freq = frequency
    if ref_frequency is None:
        ref_frequency = freq

    return representation, freq, ref_frequency
