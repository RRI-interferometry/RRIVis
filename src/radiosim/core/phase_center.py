"""Canonical phase-center identity for Tier 4 zenith-drift simulation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

from radiosim.core.result import InvalidPhaseCenterError
from radiosim.core.runtime_config import FrozenMapping, json_safe_mapping


@dataclass(frozen=True, slots=True)
class PhaseCenter:
    """The exact immutable zenith-drift phase convention."""

    schema_version: Literal["radiosim.phase-center.v1"] = "radiosim.phase-center.v1"
    kind: Literal["zenith_drift"] = "zenith_drift"
    frame: Literal["altaz"] = "altaz"
    azimuth_rad: float = 0.0
    altitude_rad: float = math.pi / 2.0
    time_dependent: Literal[True] = True
    geometric_phase_sign: Literal[-1] = -1
    w_reference: Literal["n_minus_one"] = "n_minus_one"

    def __init_subclass__(cls, **kwargs: object) -> None:
        raise TypeError("PhaseCenter cannot be subclassed")

    def __post_init__(self) -> None:
        canonical = (
            type(self.schema_version) is str
            and self.schema_version == "radiosim.phase-center.v1"
            and type(self.kind) is str
            and self.kind == "zenith_drift"
            and type(self.frame) is str
            and self.frame == "altaz"
            and type(self.azimuth_rad) is float
            and type(self.altitude_rad) is float
            and type(self.time_dependent) is bool
            and self.time_dependent is True
            and type(self.geometric_phase_sign) is int
            and self.geometric_phase_sign == -1
            and type(self.w_reference) is str
            and self.w_reference == "n_minus_one"
        )
        try:
            azimuth = float(self.azimuth_rad)
            altitude = float(self.altitude_rad)
        except (TypeError, ValueError, OverflowError) as exc:
            raise InvalidPhaseCenterError(
                "phase-center angles must be finite real numbers"
            ) from exc
        if (
            not canonical
            or isinstance(self.azimuth_rad, bool)
            or isinstance(self.altitude_rad, bool)
            or not math.isfinite(azimuth)
            or not math.isfinite(altitude)
            or not 0.0 <= azimuth < 2.0 * math.pi
            or not -math.pi / 2.0 <= altitude <= math.pi / 2.0
            or azimuth != 0.0
            or altitude != math.pi / 2.0
        ):
            raise InvalidPhaseCenterError(
                "PhaseCenter must use the canonical zenith-drift convention"
            )
        object.__setattr__(self, "azimuth_rad", 0.0)
        object.__setattr__(self, "altitude_rad", math.pi / 2.0)

    def to_snapshot(self) -> FrozenMapping:
        """Return an immutable JSON-safe identity snapshot."""
        return json_safe_mapping(
            {
                "schema_version": self.schema_version,
                "kind": self.kind,
                "frame": self.frame,
                "azimuth_rad": self.azimuth_rad,
                "altitude_rad": self.altitude_rad,
                "time_dependent": self.time_dependent,
                "geometric_phase_sign": self.geometric_phase_sign,
                "w_reference": self.w_reference,
            }
        )
