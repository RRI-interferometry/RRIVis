# rrivis/core/sky/_dnds_models.py
"""Validated differential-source-count ``dN/dS`` models.

The scientific confusion workflow only accepts literature-traceable presets
with explicit calibration frequency and published flux-density validity range.
Unsupported extrapolation is rejected rather than guessed.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

_N_NUMERIC_TAB = 4096


@dataclass(frozen=True)
class DNDSModel:
    """Container describing a validated differential-source-count model."""

    name: str
    reference_frequency_hz: float
    flux_valid_range_jy: tuple[float, float]
    dn_ds: Callable[[np.ndarray], np.ndarray]
    sample_flux: Callable[[int, np.random.Generator, float, float], np.ndarray]
    integrated_counts: Callable[[float, float], float]
    notes: str = ""
    validated: bool = True

    def validate_flux_range(self, s_min: float, s_max: float) -> None:
        """Raise if ``[s_min, s_max]`` lies outside the published validity band."""
        lo_valid, hi_valid = self.flux_valid_range_jy
        if s_min <= 0.0 or s_max <= s_min:
            raise ValueError(
                f"{self.name}: require 0 < s_min < s_max, got "
                f"s_min={s_min!r}, s_max={s_max!r}."
            )
        if s_min < lo_valid or s_max > hi_valid:
            raise ValueError(
                f"{self.name}: requested flux band {s_min:g}–{s_max:g} Jy lies "
                f"outside the validated range {lo_valid:g}–{hi_valid:g} Jy."
            )


def _logspace_grid(
    s_min: float, s_max: float, *, n: int = _N_NUMERIC_TAB
) -> np.ndarray:
    return np.asarray(
        np.logspace(np.log10(s_min), np.log10(s_max), n),
        dtype=np.float64,
    )


def _numeric_integral(
    fn: Callable[[np.ndarray], np.ndarray],
    s_min: float,
    s_max: float,
) -> float:
    grid = _logspace_grid(s_min, s_max)
    vals = np.asarray(fn(grid), dtype=np.float64)
    return float(np.trapezoid(vals, grid))


def _numeric_sampler(
    fn: Callable[[np.ndarray], np.ndarray],
    *,
    n: int,
    rng: np.random.Generator,
    s_min: float,
    s_max: float,
) -> np.ndarray:
    if n <= 0:
        return np.zeros(0, dtype=np.float64)
    grid = _logspace_grid(s_min, s_max)
    pdf = np.asarray(fn(grid), dtype=np.float64)
    if np.any(pdf < 0.0):
        raise ValueError("Validated dN/dS models must stay non-negative.")
    d_s = np.diff(grid)
    masses = 0.5 * (pdf[:-1] + pdf[1:]) * d_s
    cdf = np.concatenate(([0.0], np.cumsum(masses)))
    if cdf[-1] <= 0.0:
        raise ValueError("dN/dS integrates to zero on the requested range.")
    cdf /= cdf[-1]
    u = rng.uniform(0.0, 1.0, size=n)
    return np.interp(u, cdf, grid)


def _make_numeric_model(
    *,
    name: str,
    reference_frequency_hz: float,
    flux_valid_range_jy: tuple[float, float],
    dn_ds: Callable[[np.ndarray], np.ndarray],
    notes: str,
) -> DNDSModel:
    def _integrated(s_min: float, s_max: float) -> float:
        model.validate_flux_range(s_min, s_max)
        return _numeric_integral(dn_ds, s_min, s_max)

    def _sample(
        n: int,
        rng: np.random.Generator,
        s_min: float,
        s_max: float,
    ) -> np.ndarray:
        model.validate_flux_range(s_min, s_max)
        return _numeric_sampler(dn_ds, n=n, rng=rng, s_min=s_min, s_max=s_max)

    model = DNDSModel(
        name=name,
        reference_frequency_hz=reference_frequency_hz,
        flux_valid_range_jy=flux_valid_range_jy,
        dn_ds=dn_ds,
        sample_flux=_sample,
        integrated_counts=_integrated,
        notes=notes,
        validated=True,
    )
    return model


def _franzen2019_154mhz_dn_ds(s: np.ndarray) -> np.ndarray:
    """154 MHz counts from Franzen et al. 2019, PASA 36:e004, Eq. (11).

    The paper fits the Euclidean-normalized counts with

        log10(S^2.5 dN/dS) = Σ a_i [log10(S)]^i

    over the published validity range 1 mJy–75 Jy, with coefficients

        a0=3.52, a1=0.307, a2=0.388, a3=-0.0404, a4=0.0351, a5=0.00600

    where S is in Jy and dN/dS has units sr^-1 Jy^-1.
    """
    s_arr = np.asarray(s, dtype=np.float64)
    if np.any(s_arr <= 0.0):
        raise ValueError("dN/dS is only defined for strictly positive flux density.")
    log_s = np.log10(s_arr)
    coeffs = np.array([3.52, 0.307, 0.388, -0.0404, 0.0351, 0.00600])
    poly = np.polynomial.polynomial.polyval(log_s, coeffs)
    euclid = np.power(10.0, poly)
    return euclid / np.power(s_arr, 2.5)


FRANZEN2019_GLEAM_154MHZ = _make_numeric_model(
    name="franzen2019_gleam_154mhz",
    reference_frequency_hz=154e6,
    flux_valid_range_jy=(1e-3, 75.0),
    dn_ds=_franzen2019_154mhz_dn_ds,
    notes=(
        "Franzen et al. 2019, PASA 36:e004, Eq. (11): 154 MHz polynomial fit "
        "to Euclidean-normalized differential counts, valid for 1 mJy–75 Jy."
    ),
)


DNDS_MODELS: dict[str, DNDSModel] = {
    FRANZEN2019_GLEAM_154MHZ.name: FRANZEN2019_GLEAM_154MHZ,
}


def resolve_dn_ds(spec: object) -> DNDSModel:
    """Resolve a validated ``dN/dS`` spec into a :class:`DNDSModel`."""
    if isinstance(spec, DNDSModel):
        if not spec.validated:
            raise ValueError(
                f"dN/dS model {spec.name!r} is not marked as validated and is "
                "not accepted by the scientific confusion workflow."
            )
        return spec
    if isinstance(spec, str):
        if spec not in DNDS_MODELS:
            raise KeyError(
                f"Unknown dN/dS preset '{spec}'. Available: {sorted(DNDS_MODELS)}"
            )
        return DNDS_MODELS[spec]
    raise TypeError(
        "dN/dS spec must be a validated preset name or a validated DNDSModel; "
        f"got {type(spec).__name__}."
    )


__all__ = [
    "DNDSModel",
    "DNDS_MODELS",
    "FRANZEN2019_GLEAM_154MHZ",
    "resolve_dn_ds",
]
