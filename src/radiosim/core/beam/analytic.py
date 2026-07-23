"""Private canonical scalar analytic-beam evaluators for Tier 3E."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
from scipy.special import j0, j1, jv  # pyright: ignore[reportMissingTypeStubs]

from radiosim.core.beam.errors import (
    BeamAngularDomainError,
    BeamFrequencyDomainError,
    BeamSamplingDerivationError,
    NonFiniteBeamResponseError,
    UnsupportedBeamPrecisionError,
)
from radiosim.core.beam.models import (
    LoadedBeamHandlerState,
    ResolvedAnalyticalIlluminationBeamModel,
    ResolvedAnalyticBeamDefinition,
    ResolvedCassegrainReflector,
    ResolvedCircularApertureBeamModel,
    ResolvedCorrugatedHornIllumination,
    ResolvedCosineTaper,
    ResolvedDerivedGaussianTaper,
    ResolvedDerivedParabolicSquaredTaper,
    ResolvedDerivedParabolicTaper,
    ResolvedDipoleGroundPlaneIllumination,
    ResolvedEllipticalApertureBeamModel,
    ResolvedGaussianTaper,
    ResolvedNumericalIlluminationBeamModel,
    ResolvedOpenWaveguideIllumination,
    ResolvedParabolicSquaredTaper,
    ResolvedParabolicTaper,
    ResolvedRectangularApertureBeamModel,
    ResolvedUniformTaper,
    _canonical_digest,  # pyright: ignore[reportPrivateUsage]
    _effective_assignment_dimensions,  # pyright: ignore[reportPrivateUsage]
)
from radiosim.core.precision import COMPLEX256_AVAILABLE, PrecisionConfig

_C_M_PER_S = 299_792_458.0
_ANALYTIC_CONTRACT = "tier3-scalar-v1"


def _uniform_taper(u_beam: np.ndarray) -> np.ndarray:
    u = np.asarray(u_beam, dtype=np.float64)
    argument = np.pi * u
    result = np.ones_like(argument)
    nonzero = argument != 0.0
    result[nonzero] = 2.0 * j1(argument[nonzero]) / argument[nonzero]
    return result


def _gaussian_taper(u_beam: np.ndarray, edge_taper_db: float) -> np.ndarray:
    u = np.asarray(u_beam, dtype=np.float64)
    pedestal = 10.0 ** (-edge_taper_db / 20.0)
    alpha = edge_taper_db * np.log(10.0) / 20.0
    return pedestal * _uniform_taper(u) + (1.0 - pedestal) * np.exp(-alpha * u**2)


def _parabolic_taper(
    u_beam: np.ndarray,
    edge_taper_db: float,
    *,
    squared: bool,
) -> np.ndarray:
    u = np.asarray(u_beam, dtype=np.float64)
    argument = np.pi * u
    pedestal = 10.0 ** (-edge_taper_db / 20.0)
    tapered = np.ones_like(argument)
    nonzero = argument != 0.0
    if squared:
        tapered[nonzero] = 48.0 * jv(3, argument[nonzero]) / argument[nonzero] ** 3
    else:
        tapered[nonzero] = 8.0 * jv(2, argument[nonzero]) / argument[nonzero] ** 2
    return pedestal * _uniform_taper(u) + (1.0 - pedestal) * tapered


def _cosine_taper(u_beam: np.ndarray) -> np.ndarray:
    u = np.asarray(u_beam, dtype=np.float64)
    result = np.empty_like(u)
    zero = u == 0.0
    singular = np.abs(np.abs(u) - 1.0) < 1e-12
    regular = ~zero & ~singular
    result[zero] = 1.0
    result[singular] = np.pi / 4.0
    result[regular] = np.cos(np.pi * u[regular] / 2.0) / (1.0 - u[regular] ** 2)
    return result


def _corrugated_horn(theta_feed: np.ndarray, q: float) -> np.ndarray:
    return np.cos(np.asarray(theta_feed, dtype=np.float64)) ** q


def _open_waveguide(
    theta_feed: np.ndarray,
    b_over_lambda: float,
) -> tuple[np.ndarray, np.ndarray]:
    theta = np.asarray(theta_feed, dtype=np.float64)
    e_plane = np.cos(theta)
    x = b_over_lambda * np.sin(theta)
    denominator = 1.0 - (2.0 * x) ** 2
    singular = np.abs(denominator) < 1e-12
    safe_denominator = np.where(singular, 1.0, denominator)
    h_plane = np.where(
        singular,
        np.pi / 4.0,
        np.cos(np.pi * x) / safe_denominator,
    )
    return e_plane, h_plane


def _dipole_ground_plane(
    theta_feed: np.ndarray,
    height_wavelengths: float,
) -> np.ndarray:
    theta = np.asarray(theta_feed, dtype=np.float64)
    return np.cos(theta) * np.sin(2.0 * np.pi * height_wavelengths * np.cos(theta))


@dataclass(frozen=True, slots=True)
class _AnalyticScience:
    diameter_max_m: float
    derived_edge_taper_db: float | None
    n_radial: int | None


def _result_dtype(precision: PrecisionConfig) -> np.dtype[Any]:
    requested = precision.jones.beam
    if requested == "float32":
        return np.dtype(np.complex64)
    if requested == "float64":
        return np.dtype(np.complex128)
    if requested == "float128" and COMPLEX256_AVAILABLE:
        return np.dtype(np.complex256)
    raise UnsupportedBeamPrecisionError(
        "Analytic beam evaluation requested float128 beam precision, but this "
        "NumPy runtime does not provide a distinct complex256 dtype."
    )


def _observation_frequencies(
    value: tuple[float, ...],
) -> tuple[float, ...]:
    if type(value) is not tuple or not value:
        raise BeamFrequencyDomainError(
            "observation_frequencies_hz must be a nonempty exact tuple."
        )
    copied: list[float] = []
    previous: float | None = None
    for frequency in value:
        if type(frequency) is not float or not math.isfinite(frequency):
            raise NonFiniteBeamResponseError(
                "observation_frequencies_hz must contain exact finite Python floats."
            )
        if frequency <= 0.0:
            raise BeamFrequencyDomainError(
                "observation_frequencies_hz must contain positive frequencies."
            )
        if previous is not None and frequency <= previous:
            raise BeamFrequencyDomainError(
                "observation_frequencies_hz must be strictly increasing."
            )
        copied.append(frequency)
        previous = frequency
    return tuple(copied)


def _feed_response(
    illumination: object,
    theta_feed: np.ndarray,
) -> np.ndarray:
    if type(illumination) is ResolvedCorrugatedHornIllumination:
        return _corrugated_horn(theta_feed, q=illumination.q)
    if type(illumination) is ResolvedOpenWaveguideIllumination:
        e_plane, h_plane = _open_waveguide(
            theta_feed,
            b_over_lambda=illumination.b_over_lambda,
        )
        return np.sqrt(np.abs(e_plane) * np.abs(h_plane))
    if type(illumination) is ResolvedDipoleGroundPlaneIllumination:
        return _dipole_ground_plane(
            theta_feed,
            height_wavelengths=illumination.height_wavelengths,
        )
    raise TypeError("analytic definition contains an unsupported illumination")


def _feed_angles(
    model: (
        ResolvedAnalyticalIlluminationBeamModel | ResolvedNumericalIlluminationBeamModel
    ),
    rho_m: np.ndarray,
    diameter_m: float,
) -> np.ndarray:
    focal_length_m = model.illumination.focal_ratio * diameter_m
    if type(model.reflector) is ResolvedCassegrainReflector:
        return 2.0 * np.arctan(
            rho_m / (2.0 * model.reflector.magnification * focal_length_m)
        )
    return 2.0 * np.arctan(rho_m / (2.0 * focal_length_m))


def _derived_edge_taper_db(
    model: ResolvedAnalyticalIlluminationBeamModel,
    diameter_m: float,
) -> float:
    edge_angle = _feed_angles(
        model,
        np.array([diameter_m / 2.0], dtype=np.float64),
        diameter_m,
    )
    edge_response = float(np.abs(_feed_response(model.illumination, edge_angle)[0]))
    if not math.isfinite(edge_response) or edge_response <= 0.0:
        raise BeamSamplingDerivationError(
            "Analytic illumination has no finite positive voltage at the "
            "reflector edge, so an edge taper cannot be derived."
        )
    result = float(-20.0 * np.log10(edge_response))
    if not math.isfinite(result):
        raise BeamSamplingDerivationError(
            "Analytic illumination produced a non-finite derived edge taper."
        )
    return result


def _science(
    definition: ResolvedAnalyticBeamDefinition,
    antenna_diameter_m: float,
) -> _AnalyticScience:
    model = definition.model
    if type(model) is ResolvedCircularApertureBeamModel:
        return _AnalyticScience(antenna_diameter_m, None, None)
    if type(model) is ResolvedRectangularApertureBeamModel:
        return _AnalyticScience(
            max(model.north_length_m, model.east_length_m),
            None,
            None,
        )
    if type(model) is ResolvedEllipticalApertureBeamModel:
        return _AnalyticScience(
            max(model.north_diameter_m, model.east_diameter_m),
            None,
            None,
        )
    if type(model) is ResolvedAnalyticalIlluminationBeamModel:
        return _AnalyticScience(
            antenna_diameter_m,
            _derived_edge_taper_db(model, antenna_diameter_m),
            None,
        )
    if type(model) is ResolvedNumericalIlluminationBeamModel:
        return _AnalyticScience(
            antenna_diameter_m,
            None,
            model.n_radial,
        )
    raise TypeError("analytic definition contains an unsupported beam model")


def _analytic_preload_key(  # pyright: ignore[reportUnusedFunction]
    definition: ResolvedAnalyticBeamDefinition,
    antenna_diameter_m: float,
) -> str:
    return _canonical_digest(
        {
            "kind": "analytic_preload",
            "definition_fingerprint": definition.definition_fingerprint,
            "effective_dimensions": _effective_assignment_dimensions(
                definition,
                antenna_diameter_m,
            ),
            "contract": _ANALYTIC_CONTRACT,
        }
    )


def _scientific_fingerprint(
    definition: ResolvedAnalyticBeamDefinition,
    antenna_diameter_m: float,
    science: _AnalyticScience,
    frequencies: tuple[float, ...],
    feature_scales: tuple[tuple[float, float], ...],
) -> str:
    return _canonical_digest(
        {
            "schema_version": "tier3-beam-v1",
            "kind": "analytic_handler",
            "contract": _ANALYTIC_CONTRACT,
            "model": definition.model,
            "effective_dimensions": _effective_assignment_dimensions(
                definition,
                antenna_diameter_m,
            ),
            "derived_edge_taper_db": science.derived_edge_taper_db,
            "n_radial": science.n_radial,
            "observation_frequencies_hz": frequencies,
            "voltage_feature_scale_by_frequency": feature_scales,
        }
    )


def _direct_taper_voltage(
    model: ResolvedCircularApertureBeamModel,
    u_beam: np.ndarray,
) -> np.ndarray:
    taper = model.taper
    if type(taper) is ResolvedUniformTaper:
        return _uniform_taper(u_beam)
    if type(taper) is ResolvedGaussianTaper:
        return _gaussian_taper(u_beam, taper.edge_taper_db)
    if type(taper) is ResolvedParabolicTaper:
        return _parabolic_taper(u_beam, taper.edge_taper_db, squared=False)
    if type(taper) is ResolvedParabolicSquaredTaper:
        return _parabolic_taper(u_beam, taper.edge_taper_db, squared=True)
    if type(taper) is ResolvedCosineTaper:
        return _cosine_taper(u_beam)
    raise TypeError("circular analytic definition contains an unsupported taper")


def _derived_taper_voltage(
    model: ResolvedAnalyticalIlluminationBeamModel,
    u_beam: np.ndarray,
    edge_taper_db: float,
) -> np.ndarray:
    taper = model.taper_profile
    if type(taper) is ResolvedDerivedGaussianTaper:
        return _gaussian_taper(u_beam, edge_taper_db)
    if type(taper) is ResolvedDerivedParabolicTaper:
        return _parabolic_taper(u_beam, edge_taper_db, squared=False)
    if type(taper) is ResolvedDerivedParabolicSquaredTaper:
        return _parabolic_taper(u_beam, edge_taper_db, squared=True)
    raise TypeError("analytical illumination contains an unsupported taper profile")


def _numerical_voltage(
    model: ResolvedNumericalIlluminationBeamModel,
    theta_rad: np.ndarray,
    diameter_m: float,
    frequency_hz: float,
) -> np.ndarray:
    radius_m = diameter_m / 2.0
    rho_m = np.linspace(0.0, radius_m, model.n_radial, dtype=np.float64)
    illumination = _feed_response(
        model.illumination,
        _feed_angles(model, rho_m, diameter_m),
    )
    normalization = float(np.trapezoid(illumination * rho_m, rho_m))
    if not math.isfinite(normalization) or abs(normalization) <= np.finfo(float).tiny:
        raise BeamSamplingDerivationError(
            "Numerical illumination has a zero or non-finite Hankel normalization."
        )
    u_beam = diameter_m * np.sin(theta_rad) * frequency_hz / _C_M_PER_S
    argument = 2.0 * np.pi * u_beam[:, None] * rho_m[None, :] / diameter_m
    integrand = illumination[None, :] * j0(argument) * rho_m[None, :]
    return np.asarray(
        np.trapezoid(integrand, rho_m, axis=1) / normalization,
        dtype=np.float64,
    )


class _AnalyticScalarEvaluator:
    __slots__ = (
        "_antenna_diameter_m",
        "_definition",
        "_identity",
        "_result_dtype",
        "_science",
    )

    def __init__(
        self,
        *,
        definition: ResolvedAnalyticBeamDefinition,
        antenna_diameter_m: float,
        identity: str,
        result_dtype: np.dtype[Any],
        science: _AnalyticScience,
    ) -> None:
        self._definition = definition
        self._antenna_diameter_m = antenna_diameter_m
        self._identity = identity
        self._result_dtype = result_dtype
        self._science = science

    def voltage_feature_scale_rad(self, frequency_hz: float) -> float:
        if type(frequency_hz) is not float or not math.isfinite(frequency_hz):
            raise NonFiniteBeamResponseError(
                "frequency_hz must be an exact finite Python float."
            )
        if frequency_hz <= 0.0:
            raise BeamFrequencyDomainError(
                f"{self._identity}: frequency_hz must be positive."
            )
        return _C_M_PER_S / frequency_hz / self._science.diameter_max_m

    def evaluate_numpy(
        self,
        altitude_rad: np.ndarray,
        azimuth_rad: np.ndarray,
        frequency_hz: float,
        time_mjd: float,
    ) -> np.ndarray:
        if type(altitude_rad) is not np.ndarray or altitude_rad.ndim != 1:
            raise BeamAngularDomainError(
                "altitude_rad must be a one-dimensional array."
            )
        if type(azimuth_rad) is not np.ndarray or azimuth_rad.ndim != 1:
            raise BeamAngularDomainError("azimuth_rad must be a one-dimensional array.")
        if altitude_rad.shape != azimuth_rad.shape:
            raise BeamAngularDomainError(
                "altitude_rad and azimuth_rad must have identical shapes."
            )
        try:
            if (
                altitude_rad.dtype.kind not in "fiu"
                or azimuth_rad.dtype.kind not in "fiu"
            ):
                raise TypeError("direction arrays must have real numeric dtypes")
            altitude = np.asarray(altitude_rad, dtype=np.float64)
            azimuth = np.asarray(azimuth_rad, dtype=np.float64)
        except (TypeError, ValueError, OverflowError) as exc:
            raise BeamAngularDomainError(
                "altitude_rad and azimuth_rad must contain real numeric values."
            ) from exc
        if not np.all(np.isfinite(altitude)) or not np.all(np.isfinite(azimuth)):
            raise NonFiniteBeamResponseError(
                "altitude_rad and azimuth_rad must contain finite values."
            )
        if np.any(altitude < -np.pi / 2.0) or np.any(altitude > np.pi / 2.0):
            raise BeamAngularDomainError(
                "altitude_rad values must lie in [-pi/2, pi/2]."
            )
        if type(frequency_hz) is not float or not math.isfinite(frequency_hz):
            raise NonFiniteBeamResponseError(
                "frequency_hz must be an exact finite Python float."
            )
        if frequency_hz <= 0.0:
            raise BeamFrequencyDomainError(
                f"{self._identity}: frequency_hz must be positive."
            )
        if type(time_mjd) is not float or not math.isfinite(time_mjd):
            raise NonFiniteBeamResponseError(
                "time_mjd must be an exact finite Python float."
            )

        voltage = np.zeros(altitude.shape, dtype=np.float64)
        visible = altitude >= 0.0
        if np.any(visible):
            theta = np.pi / 2.0 - altitude[visible]
            visible_azimuth = azimuth[visible]
            wavelength_m = _C_M_PER_S / frequency_hz
            model = self._definition.model
            if type(model) is ResolvedCircularApertureBeamModel:
                u_beam = self._antenna_diameter_m * np.sin(theta) / wavelength_m
                evaluated = _direct_taper_voltage(model, u_beam)
            elif type(model) is ResolvedRectangularApertureBeamModel:
                direction_radius = np.sin(theta)
                north_u = (
                    model.north_length_m
                    * direction_radius
                    * np.cos(visible_azimuth)
                    / wavelength_m
                )
                east_u = (
                    model.east_length_m
                    * direction_radius
                    * np.sin(visible_azimuth)
                    / wavelength_m
                )
                evaluated = np.sinc(north_u) * np.sinc(east_u)
            elif type(model) is ResolvedEllipticalApertureBeamModel:
                effective_diameter = 1.0 / np.sqrt(
                    (np.cos(visible_azimuth) / model.north_diameter_m) ** 2
                    + (np.sin(visible_azimuth) / model.east_diameter_m) ** 2
                )
                evaluated = _uniform_taper(
                    effective_diameter * np.sin(theta) / wavelength_m
                )
            elif type(model) is ResolvedAnalyticalIlluminationBeamModel:
                u_beam = self._antenna_diameter_m * np.sin(theta) / wavelength_m
                evaluated = _derived_taper_voltage(
                    model,
                    u_beam,
                    cast(float, self._science.derived_edge_taper_db),
                )
            elif type(model) is ResolvedNumericalIlluminationBeamModel:
                evaluated = _numerical_voltage(
                    model,
                    theta,
                    self._antenna_diameter_m,
                    frequency_hz,
                )
            else:
                raise TypeError("analytic definition contains an unsupported model")
            evaluated = np.asarray(evaluated, dtype=np.float64)
            evaluated[theta == 0.0] = 1.0
            voltage[visible] = evaluated

        if not np.all(np.isfinite(voltage)):
            raise NonFiniteBeamResponseError(
                f"{self._identity}: analytic evaluation produced non-finite values."
            )
        result = np.zeros(
            (altitude.size, 2, 2),
            dtype=self._result_dtype,
            order="C",
        )
        result[:, 0, 0] = voltage
        result[:, 1, 1] = voltage
        result.setflags(write=False)
        return result


@dataclass(frozen=True, slots=True)
class _LoadedAnalyticHandler:
    state: LoadedBeamHandlerState
    evaluator: _AnalyticScalarEvaluator


def _load_analytic_handler(  # pyright: ignore[reportUnusedFunction]
    definition: ResolvedAnalyticBeamDefinition,
    *,
    antenna_diameter_m: float,
    observation_frequencies_hz: tuple[float, ...],
    precision: PrecisionConfig,
    handler_ordinal: int,
) -> _LoadedAnalyticHandler:
    if type(definition) is not ResolvedAnalyticBeamDefinition:
        raise TypeError("definition must be an exact ResolvedAnalyticBeamDefinition")
    definition.__post_init__()
    if type(antenna_diameter_m) is not float or not math.isfinite(antenna_diameter_m):
        raise ValueError("antenna_diameter_m must be an exact finite float")
    if antenna_diameter_m <= 0.0:
        raise ValueError("antenna_diameter_m must be positive")
    if type(precision) is not PrecisionConfig:
        raise TypeError("precision must be an exact PrecisionConfig")
    if type(handler_ordinal) is not int or handler_ordinal < 0:
        raise ValueError("handler_ordinal must be a nonnegative exact integer")

    frequencies = _observation_frequencies(observation_frequencies_hz)
    result_dtype = _result_dtype(precision)
    science = _science(definition, antenna_diameter_m)
    feature_scales = tuple(
        (
            frequency,
            _C_M_PER_S / frequency / science.diameter_max_m,
        )
        for frequency in frequencies
    )
    scientific_fingerprint = _scientific_fingerprint(
        definition,
        antenna_diameter_m,
        science,
        frequencies,
        feature_scales,
    )
    handler_id = f"beam-{handler_ordinal:04d}-{scientific_fingerprint[:12]}"
    state = LoadedBeamHandlerState(
        handler_id=handler_id,
        kind="analytic",
        definition_fingerprint=definition.definition_fingerprint,
        scientific_fingerprint=scientific_fingerprint,
        file=None,
        voltage_feature_scale_by_frequency=feature_scales,
    )
    evaluator = _AnalyticScalarEvaluator(
        definition=definition,
        antenna_diameter_m=antenna_diameter_m,
        identity=handler_id,
        result_dtype=result_dtype,
        science=science,
    )
    return _LoadedAnalyticHandler(state=state, evaluator=evaluator)


__all__: list[str] = []
