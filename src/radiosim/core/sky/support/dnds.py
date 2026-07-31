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


def _gervasi2008_150mhz_dn_ds(s: np.ndarray) -> np.ndarray:
    """150 MHz counts from Gervasi et al. 2008, ApJ 682, 223, Eq. (2).

    The paper fits the Euclidean-normalized counts with a sum of two
    inverse double power laws,

        S^2.5 dN/dS = 1/(A1 S^a1 + B1 S^b1) + 1/(A2 S^a2 + B2 S^b2)

    where S is in Jy and dN/dS has units sr^-1 Jy^-1. The parameters used
    here are the weighted-average slope values with the 151 MHz
    normalizations, as adopted at 150 MHz by Mittal et al. 2024
    (MNRAS 534, 1317), Table 1:

        A1 = 1.65e-4, B1 = 1.14e-4, a1 = -0.854, b1 = 0.37,
        A2/A1 = 0.24, B2/B1 = 1.8e7, a2 = -0.856, b2 = 1.47
    """
    s_arr = np.asarray(s, dtype=np.float64)
    if np.any(s_arr <= 0.0):
        raise ValueError("dN/dS is only defined for strictly positive flux density.")
    slope_a1, slope_b1 = -0.854, 0.37
    slope_a2, slope_b2 = -0.856, 1.47
    norm_a1, norm_b1 = 1.65e-4, 1.14e-4
    norm_a2, norm_b2 = 0.24 * norm_a1, 1.8e7 * norm_b1
    euclid = 1.0 / (norm_a1 * s_arr**slope_a1 + norm_b1 * s_arr**slope_b1) + 1.0 / (
        norm_a2 * s_arr**slope_a2 + norm_b2 * s_arr**slope_b2
    )
    return euclid / np.power(s_arr, 2.5)


GERVASI2008_150MHZ = _make_numeric_model(
    name="gervasi2008_150mhz",
    reference_frequency_hz=150e6,
    flux_valid_range_jy=(1e-6, 1e2),
    dn_ds=_gervasi2008_150mhz_dn_ds,
    notes=(
        "Gervasi et al. 2008, ApJ 682, 223, Eq. (2): sum of two inverse double "
        "power laws fitted to 151 MHz differential counts; weighted-average "
        "slopes with 151 MHz normalizations as adopted at 150 MHz by Mittal "
        "et al. 2024 (MNRAS 534, 1317), Table 1. Data-supported range "
        "1 uJy-100 Jy."
    ),
)


def _mandal2021_150mhz_dn_ds(s: np.ndarray) -> np.ndarray:
    """150 MHz counts from Mandal et al. 2021, A&A 648, A5, Eq. (13).

    The paper fits the Euclidean-normalized counts with

        log10(S^2.5 dN/dS) = sum_i a_i [log10(S / 1 mJy)]^i

    over the published validity range 0.2 mJy-10 Jy (Table 4), with
    coefficients

        a0=1.655, a1=-0.1150, a2=0.2272, a3=0.51788, a4=-0.449661,
        a5=0.160265, a6=-0.028541, a7=0.002041

    The polynomial argument is the flux density in mJy while the
    Euclidean-normalized counts stay in Jy^1.5 sr^-1, so dN/dS is
    returned in sr^-1 Jy^-1 for S in Jy.
    """
    s_arr = np.asarray(s, dtype=np.float64)
    if np.any(s_arr <= 0.0):
        raise ValueError("dN/dS is only defined for strictly positive flux density.")
    log_s_mjy = np.log10(s_arr * 1e3)
    coeffs = np.array(
        [1.655, -0.1150, 0.2272, 0.51788, -0.449661, 0.160265, -0.028541, 0.002041]
    )
    poly = np.polynomial.polynomial.polyval(log_s_mjy, coeffs)
    euclid = np.power(10.0, poly)
    return euclid / np.power(s_arr, 2.5)


MANDAL2021_LOTSS_150MHZ = _make_numeric_model(
    name="mandal2021_lotss_150mhz",
    reference_frequency_hz=150e6,
    flux_valid_range_jy=(2e-4, 10.0),
    dn_ds=_mandal2021_150mhz_dn_ds,
    notes=(
        "Mandal et al. 2021, A&A 648, A5, Eq. (13) and Table 4: 150 MHz "
        "7th-order polynomial fit to Euclidean-normalized differential counts "
        "from the LoTSS Deep Fields combined with TGSS-ADR1, valid for "
        "0.2 mJy-10 Jy."
    ),
)


def _intema2017_150mhz_dn_ds(s: np.ndarray) -> np.ndarray:
    """150 MHz counts from Intema et al. 2017, A&A 598, A78, Eq. (4).

    The paper fits the Euclidean-normalized counts with

        log10(S^2.5 dN/dS) = C0 + sum_{i=1..5} C_i [log10(S)]^i

    over 5 mJy-100 Jy (best constrained between 100 mJy and 10 Jy), with
    coefficients (Table 6)

        C0=3.5142, C1=0.3738, C2=-0.3138, C3=-0.0717, C4=0.0213, C5=0.0097

    where S is in Jy and dN/dS has units sr^-1 Jy^-1.
    """
    s_arr = np.asarray(s, dtype=np.float64)
    if np.any(s_arr <= 0.0):
        raise ValueError("dN/dS is only defined for strictly positive flux density.")
    log_s = np.log10(s_arr)
    coeffs = np.array([3.5142, 0.3738, -0.3138, -0.0717, 0.0213, 0.0097])
    poly = np.polynomial.polynomial.polyval(log_s, coeffs)
    euclid = np.power(10.0, poly)
    return euclid / np.power(s_arr, 2.5)


INTEMA2017_TGSS_150MHZ = _make_numeric_model(
    name="intema2017_tgss_150mhz",
    reference_frequency_hz=150e6,
    flux_valid_range_jy=(5e-3, 100.0),
    dn_ds=_intema2017_150mhz_dn_ds,
    notes=(
        "Intema et al. 2017, A&A 598, A78, Eq. (4) and Table 6: 150 MHz "
        "5th-order polynomial fit to TGSS ADR1 Euclidean-normalized "
        "differential counts, fitted over 5 mJy-100 Jy and best constrained "
        "between 100 mJy and 10 Jy."
    ),
)


DNDS_MODELS: dict[str, DNDSModel] = {
    FRANZEN2019_GLEAM_154MHZ.name: FRANZEN2019_GLEAM_154MHZ,
    GERVASI2008_150MHZ.name: GERVASI2008_150MHZ,
    MANDAL2021_LOTSS_150MHZ.name: MANDAL2021_LOTSS_150MHZ,
    INTEMA2017_TGSS_150MHZ.name: INTEMA2017_TGSS_150MHZ,
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
    "GERVASI2008_150MHZ",
    "INTEMA2017_TGSS_150MHZ",
    "MANDAL2021_LOTSS_150MHZ",
    "resolve_dn_ds",
]
