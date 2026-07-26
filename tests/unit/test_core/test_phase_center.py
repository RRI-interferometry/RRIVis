"""Contracts for the canonical Tier 4 phase center."""

from __future__ import annotations

import math

import numpy as np
import pytest

from radiosim.core.phase_center import PhaseCenter
from radiosim.core.result import InvalidPhaseCenterError
from radiosim.core.runtime_config import FrozenMapping


def test_phase_center_is_the_exact_immutable_zenith_drift_contract():
    phase_center = PhaseCenter()

    assert type(phase_center) is PhaseCenter
    assert phase_center.schema_version == "radiosim.phase-center.v1"
    assert phase_center.kind == "zenith_drift"
    assert phase_center.frame == "altaz"
    assert phase_center.azimuth_rad == 0.0
    assert phase_center.altitude_rad == math.pi / 2.0
    assert phase_center.time_dependent is True
    assert phase_center.geometric_phase_sign == -1
    assert phase_center.w_reference == "n_minus_one"
    assert type(phase_center.to_snapshot()) is FrozenMapping

    with pytest.raises((AttributeError, TypeError)):
        phase_center.altitude_rad = 0.0


def test_phase_center_rejects_subclasses_and_noncanonical_values():
    with pytest.raises(TypeError):

        class MutablePhaseCenter(PhaseCenter):
            pass

    for field, value in (
        ("schema_version", "other"),
        ("kind", "tracking"),
        ("frame", "icrs"),
        ("azimuth_rad", 0),
        ("azimuth_rad", np.float64(0.0)),
        ("azimuth_rad", math.nan),
        ("altitude_rad", np.float64(math.pi / 2.0)),
        ("altitude_rad", 0.0),
        ("time_dependent", False),
        ("geometric_phase_sign", 1),
        ("w_reference", "w"),
    ):
        arguments = {field: value}
        with pytest.raises(InvalidPhaseCenterError):
            PhaseCenter(**arguments)
