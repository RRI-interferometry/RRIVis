"""Canonical drift-scan lightcurves using one selected BeamSystem handler."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, cast

import numpy as np

from radiosim.core.beam import BeamSystem
from radiosim.core.instrument import AntennaId, ResolvedEarthLocation

from .errors import (
    InvalidObservabilityContextError,
    InvalidObservabilityReferenceError,
    ObservabilitySkyUnavailableError,
)
from .geometry import compute_beam_map_on_healpix

if TYPE_CHECKING:
    from radiosim.core.sky.containers.model import SkyModel


def _owned_array(
    value: np.ndarray,
    *,
    field_name: str,
    dtype: np.dtype | type,
) -> np.ndarray:
    if type(value) is not np.ndarray:
        raise TypeError(f"{field_name} must be an exact ndarray")
    result = np.array(value, dtype=dtype, copy=True, order="C")
    if result.ndim != 1:
        raise ValueError(f"{field_name} must be one-dimensional")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{field_name} must contain only finite values")
    result.setflags(write=False)
    return result


@dataclass(frozen=True, slots=True, eq=False)
class DriftScanLightcurve:
    """Immutable drift-scan integration result for one exact channel."""

    lst_hours: np.ndarray
    integrated_flux: np.ndarray
    mean_brightness: np.ndarray | None
    horizon_masked: Literal[True]
    frequency_hz: float
    nside: int
    beam_evaluation_time_mjd: float
    reference_antenna: AntennaId
    reference_handler_id: str
    reference_scientific_fingerprint: str
    power_convention: Literal["half_trace_unpolarized"]

    def __post_init__(self) -> None:
        lst = _owned_array(
            self.lst_hours,
            field_name="lst_hours",
            dtype=np.float64,
        )
        integrated = _owned_array(
            self.integrated_flux,
            field_name="integrated_flux",
            dtype=np.float64,
        )
        mean = (
            None
            if self.mean_brightness is None
            else _owned_array(
                self.mean_brightness,
                field_name="mean_brightness",
                dtype=np.float64,
            )
        )
        if len(integrated) != len(lst) or (mean is not None and len(mean) != len(lst)):
            raise ValueError("lightcurve arrays must have matching lengths")
        if len(lst) == 0:
            raise ValueError("lightcurve arrays must be nonempty")
        if np.any((lst < 0.0) | (lst >= 24.0)):
            raise ValueError("lst_hours values must be in [0, 24)")
        if self.horizon_masked is not True:
            raise ValueError("horizon_masked must be literal True")
        if type(self.frequency_hz) is not float or not math.isfinite(self.frequency_hz):
            raise TypeError("frequency_hz must be an exact finite float")
        if self.frequency_hz <= 0.0:
            raise ValueError("frequency_hz must be positive")
        if (
            type(self.nside) is not int
            or self.nside <= 0
            or self.nside & (self.nside - 1)
        ):
            raise TypeError("nside must be a strict positive HEALPix NSIDE")
        if type(self.beam_evaluation_time_mjd) is not float or not math.isfinite(
            self.beam_evaluation_time_mjd
        ):
            raise TypeError("beam_evaluation_time_mjd must be an exact finite float")
        if type(self.reference_antenna) is not AntennaId:
            raise TypeError("reference_antenna must be an exact AntennaId")
        if (
            type(self.reference_handler_id) is not str
            or not self.reference_handler_id
            or self.reference_handler_id != self.reference_handler_id.strip()
        ):
            raise TypeError(
                "reference_handler_id must be a nonblank stripped exact string"
            )
        if (
            type(self.reference_scientific_fingerprint) is not str
            or re.fullmatch(
                r"[0-9a-f]{64}",
                self.reference_scientific_fingerprint,
            )
            is None
        ):
            raise ValueError(
                "reference_scientific_fingerprint must be a SHA-256 string"
            )
        if self.power_convention != "half_trace_unpolarized":
            raise ValueError("power_convention must be 'half_trace_unpolarized'")
        object.__setattr__(self, "lst_hours", lst)
        object.__setattr__(self, "integrated_flux", integrated)
        object.__setattr__(self, "mean_brightness", mean)
        object.__setattr__(
            self,
            "reference_antenna",
            AntennaId(
                self.reference_antenna.number,
                self.reference_antenna.name,
            ),
        )

    __hash__ = None  # type: ignore[assignment]


def _reference_handler(
    beam_system: BeamSystem,
    reference_antenna: AntennaId,
    frequency_hz: float,
) -> tuple[str, str]:
    if type(beam_system) is not BeamSystem:
        raise TypeError("beam_system must be an exact BeamSystem")
    if type(reference_antenna) is not AntennaId:
        raise InvalidObservabilityReferenceError(
            "reference_antenna must be an exact canonical AntennaId."
        )
    assignment_map = dict(beam_system.state.assignment_handler_ids)
    handler_id = assignment_map.get(reference_antenna)
    if handler_id is None:
        raise InvalidObservabilityReferenceError(
            "reference_antenna is not covered by the canonical BeamSystem."
        )
    handler = {value.handler_id: value for value in beam_system.state.handlers}[
        handler_id
    ]
    frequency_axis = tuple(
        value for value, _scale in handler.voltage_feature_scale_by_frequency
    )
    if frequency_hz not in frequency_axis:
        raise InvalidObservabilityContextError(
            "frequency_hz must exactly match a BeamSystem observation channel."
        )
    return handler_id, handler.scientific_fingerprint


def compute_drift_scan_lightcurve(
    sky: SkyModel,
    *,
    beam_system: BeamSystem,
    reference_antenna: AntennaId,
    location: ResolvedEarthLocation,
    frequency_hz: float,
    lst_hours: np.ndarray,
    beam_evaluation_time_mjd: float,
    area_normalize: bool = False,
) -> DriftScanLightcurve:
    """Integrate an exact HEALPix channel against one canonical reference beam."""
    from radiosim.core.sky.containers.model import SkyModel as SkyModelType

    if type(sky) is not SkyModelType:
        raise TypeError("sky must be an exact SkyModel")
    if sky.healpix is None:
        raise ObservabilitySkyUnavailableError(
            "compute_drift_scan_lightcurve requires a prepared HEALPix payload."
        )
    if type(location) is not ResolvedEarthLocation:
        raise TypeError("location must be an exact ResolvedEarthLocation")
    if type(frequency_hz) is not float or not math.isfinite(frequency_hz):
        raise InvalidObservabilityContextError(
            "frequency_hz must be an exact finite float."
        )
    if type(beam_evaluation_time_mjd) is not float or not math.isfinite(
        beam_evaluation_time_mjd
    ):
        raise InvalidObservabilityContextError(
            "beam_evaluation_time_mjd must be an exact finite float."
        )
    if type(cast(object, area_normalize)) is not bool:
        raise InvalidObservabilityContextError("area_normalize must be an exact bool.")
    if type(lst_hours) is not np.ndarray:
        raise InvalidObservabilityContextError("lst_hours must be an exact ndarray.")
    lsts = np.array(lst_hours, dtype=np.float64, copy=True, order="C")
    if lsts.ndim != 1 or not np.all(np.isfinite(lsts)):
        raise InvalidObservabilityContextError(
            "lst_hours must be a finite one-dimensional array."
        )
    if np.any((lsts < 0.0) | (lsts >= 24.0)):
        raise InvalidObservabilityContextError("lst_hours samples must be in [0, 24).")
    handler_id, scientific_fingerprint = _reference_handler(
        beam_system,
        reference_antenna,
        frequency_hz,
    )

    healpix = sky.healpix.require_dense("compute_drift_scan_lightcurve")
    matches = np.flatnonzero(
        np.asarray(healpix.frequencies, dtype=np.float64) == frequency_hz
    )
    if len(matches) != 1:
        raise InvalidObservabilityContextError(
            "Prepared HEALPix sky does not contain the exact selected channel."
        )
    sky_map = np.asarray(healpix.maps[int(matches[0])], dtype=np.float64)
    nside = int(healpix.nside)
    integrated = np.empty(lsts.shape, dtype=np.float64)
    mean_brightness = np.empty(lsts.shape, dtype=np.float64) if area_normalize else None

    for index, lst in enumerate(lsts):
        zenith_ra = float(((lst * 15.0) + 180.0) % 360.0 - 180.0)
        beam_map = compute_beam_map_on_healpix(
            beam_system=beam_system,
            reference_antenna=reference_antenna,
            nside=nside,
            zenith_ra_deg=zenith_ra,
            zenith_dec_deg=float(location.latitude_deg),
            frequency_hz=frequency_hz,
            time_mjd=beam_evaluation_time_mjd,
        )
        product = sky_map * beam_map
        integrated[index] = float(np.sum(product))
        if mean_brightness is not None:
            denominator = float(np.sum(beam_map))
            mean_brightness[index] = (
                float(np.sum(product) / denominator)
                if denominator > 0.0
                else float("nan")
            )

    return DriftScanLightcurve(
        lst_hours=lsts,
        integrated_flux=integrated,
        mean_brightness=mean_brightness,
        horizon_masked=True,
        frequency_hz=frequency_hz,
        nside=nside,
        beam_evaluation_time_mjd=beam_evaluation_time_mjd,
        reference_antenna=reference_antenna,
        reference_handler_id=handler_id,
        reference_scientific_fingerprint=scientific_fingerprint,
        power_convention="half_trace_unpolarized",
    )


def fractional_horizon_excess(
    masked: DriftScanLightcurve,
    unmasked: DriftScanLightcurve,
) -> np.ndarray:
    """Return the relative difference between two compatible lightcurves."""
    if (
        type(masked) is not DriftScanLightcurve
        or type(unmasked) is not DriftScanLightcurve
    ):
        raise TypeError("both inputs must be exact DriftScanLightcurve values")
    if masked.lst_hours.shape != unmasked.lst_hours.shape or not np.array_equal(
        masked.lst_hours,
        unmasked.lst_hours,
    ):
        raise ValueError(
            "fractional_horizon_excess requires both lightcurves to share "
            "the same LST grid."
        )
    if masked.frequency_hz != unmasked.frequency_hz:
        raise ValueError(
            "fractional_horizon_excess requires both lightcurves at the same frequency."
        )
    with np.errstate(divide="ignore", invalid="ignore"):
        result = (
            unmasked.integrated_flux - masked.integrated_flux
        ) / masked.integrated_flux
    result = np.array(result, dtype=np.float64, copy=True, order="C")
    result[masked.integrated_flux == 0.0] = np.nan
    result.setflags(write=False)
    return result


__all__ = [
    "DriftScanLightcurve",
    "compute_drift_scan_lightcurve",
    "fractional_horizon_excess",
]
