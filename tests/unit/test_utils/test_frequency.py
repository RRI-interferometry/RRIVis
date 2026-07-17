"""Tests for strict typed and resolved frequency configuration."""

import numpy as np
import pytest
from pydantic import TypeAdapter, ValidationError

from radiosim.io.config import (
    ExplicitFrequencyConfig,
    FrequencyGridConfig,
    ObsFrequencyConfig,
    RadioSimConfig,
    dump_config,
    load_config,
)
from tests.fixtures.configs import valid_config_mapping


def _nonuniform_frequency_config():
    return {
        "mode": "explicit",
        "channel_frequencies_hz": [100e6, 101.5e6, 108e6],
    }


def test_typed_model_preserves_nonuniform_explicit_frequencies(tmp_path):
    data = valid_config_mapping(tmp_path, frequency=_nonuniform_frequency_config())

    config = RadioSimConfig.model_validate(data)
    dumped_frequency = config.obs_frequency.model_dump()

    assert config.obs_frequency.channel_frequencies_hz == (
        100e6,
        101.5e6,
        108e6,
    )
    assert dumped_frequency["channel_frequencies_hz"] == (
        100e6,
        101.5e6,
        108e6,
    )


def test_yaml_round_trip_preserves_nonuniform_explicit_frequencies(tmp_path):
    data = valid_config_mapping(tmp_path, frequency=_nonuniform_frequency_config())
    config = RadioSimConfig.model_validate(data)
    dumped = tmp_path / "dumped.yaml"
    dump_config(config, dumped)
    reloaded = load_config(dumped)

    assert reloaded.runtime.frequency.channel_frequencies_hz == (
        config.obs_frequency.channel_frequencies_hz
    )


def test_explicit_one_channel_is_valid():
    frequency = ExplicitFrequencyConfig(channel_frequencies_hz=[100e6])

    assert frequency.channel_frequencies_hz == (100e6,)


@pytest.mark.parametrize(
    "values",
    [
        [],
        [[100e6, 101e6]],
        np.array([[100e6, 101e6]]),
        [100e6, np.nan],
        [100e6, np.inf],
        [0.0],
        [-1.0],
        [101e6, 100e6],
        [100e6, 100e6],
        [True],
        [100e6, False],
    ],
)
def test_explicit_rejects_invalid_sequences(values):
    with pytest.raises(ValidationError):
        ExplicitFrequencyConfig(channel_frequencies_hz=values)


def test_explicit_numpy_input_is_copied_to_tuple():
    values = np.array([100e6, 101.5e6, 108e6])

    frequency = ExplicitFrequencyConfig(channel_frequencies_hz=values)
    values[1] = 999e6

    assert frequency.channel_frequencies_hz == (100e6, 101.5e6, 108e6)


def test_discriminated_union_requires_mode_and_forbids_mixed_fields():
    adapter = TypeAdapter(ObsFrequencyConfig)

    with pytest.raises(ValidationError, match="union_tag_not_found"):
        adapter.validate_python({"channel_frequencies_hz": [100e6]})
    with pytest.raises(ValidationError, match="starting_frequency"):
        adapter.validate_python(
            {
                "mode": "explicit",
                "channel_frequencies_hz": [100e6],
                "starting_frequency": 100.0,
            }
        )
    with pytest.raises(ValidationError, match="channel_frequencies_hz"):
        adapter.validate_python(
            {
                "mode": "grid",
                "starting_frequency": 100.0,
                "frequency_interval": 1.0,
                "frequency_bandwidth": 2.0,
                "channel_frequencies_hz": [100e6],
            }
        )


@pytest.mark.parametrize(
    ("unit", "starting", "interval", "bandwidth"),
    [
        ("Hz", 100e6, 1e6, 2e6),
        ("kHz", 100_000.0, 1_000.0, 2_000.0),
        ("MHz", 100.0, 1.0, 2.0),
        ("GHz", 0.1, 0.001, 0.002),
    ],
)
def test_grid_all_units_validate(unit, starting, interval, bandwidth):
    frequency = FrequencyGridConfig(
        starting_frequency=starting,
        frequency_interval=interval,
        frequency_bandwidth=bandwidth,
        frequency_unit=unit,
    )

    assert frequency.n_channels == 3


def test_grid_rejects_nonintegral_interval_count():
    with pytest.raises(ValidationError, match="must be an integer"):
        FrequencyGridConfig(
            starting_frequency=100.0,
            frequency_interval=0.3,
            frequency_bandwidth=1.0,
        )


def test_grid_validation_does_not_call_linspace(monkeypatch):
    monkeypatch.setattr(
        np,
        "linspace",
        lambda *args, **kwargs: pytest.fail("schema validation called linspace"),
    )

    frequency = FrequencyGridConfig(
        starting_frequency=100.0,
        frequency_interval=0.25,
        frequency_bandwidth=1.0,
    )

    assert frequency.n_channels == 5


def test_explicit_tuple_dumps_as_json_list():
    frequency = ExplicitFrequencyConfig(channel_frequencies_hz=[100e6, 101.5e6, 108e6])

    assert frequency.model_dump(mode="json")["channel_frequencies_hz"] == [
        100e6,
        101.5e6,
        108e6,
    ]
