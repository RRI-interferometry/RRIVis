"""Tier 1G target contract for removing dictionary configuration paths."""

from __future__ import annotations

import importlib
import inspect

import numpy as np
import pytest
from pydantic import ValidationError

import radiosim.utils as public_utils
from radiosim.core.sky.combine.engine import _combine_models
from radiosim.core.sky.combine.options import PrepareSkyOptions
from radiosim.core.sky.combine.regrid import regrid_healpix_model
from radiosim.core.sky.loaders.diffuse import load_diffuse_sky, load_pysm3
from radiosim.core.sky.loaders.pyradiosky import (
    _load_pyradiosky_healpix,
    load_pyradiosky_file,
)
from radiosim.core.sky.loaders.skyh5_multifile import load_skyh5_multifile
from radiosim.core.sky.loaders.synthetic import (
    load_poisson_confusion,
    load_test_sources,
)
from radiosim.core.sky.operations.operations import materialize_healpix_model
from radiosim.core.sky.support.frequencies import validate_observation_frequencies
from radiosim.io.config import RadioSimConfig, SkySourceConfig


def test_obsolete_public_configuration_utilities_are_removed():
    assert not hasattr(public_utils, "parse_frequency_config")
    assert not hasattr(public_utils, "validate_config")


def test_former_frequency_utility_has_no_dictionary_parser():
    try:
        frequency_module = importlib.import_module("radiosim.utils.frequency")
    except ModuleNotFoundError:
        return
    assert not hasattr(frequency_module, "parse_frequency_config")


def test_input_model_has_no_obsolete_workflow_naming_helper():
    assert not hasattr(RadioSimConfig, "generate_output_subdir")


@pytest.mark.parametrize(
    "callable_object",
    [
        SkySourceConfig.to_loader_request,
        PrepareSkyOptions,
        _combine_models,
        regrid_healpix_model,
        materialize_healpix_model,
        load_diffuse_sky,
        load_pysm3,
        load_pyradiosky_file,
        _load_pyradiosky_healpix,
        load_skyh5_multifile,
        load_test_sources,
        load_poisson_confusion,
    ],
)
def test_scientific_signatures_have_no_frequency_dictionary_alternative(
    callable_object,
):
    assert "obs_frequency_config" not in inspect.signature(callable_object).parameters


def test_prepare_options_forbids_removed_frequency_dictionary_field():
    with pytest.raises(ValidationError, match="obs_frequency_config"):
        PrepareSkyOptions(obs_frequency_config={"frequencies_hz": [100e6]})


def test_explicit_frequency_boundary_preserves_values_and_owns_its_array():
    caller = np.array([100e6, 101.25e6, 109e6])

    resolved = validate_observation_frequencies(caller)
    caller[1] = 999e6

    np.testing.assert_array_equal(resolved, [100e6, 101.25e6, 109e6])
    assert resolved.dtype == np.float64
    assert resolved.flags.owndata


@pytest.mark.parametrize(
    "frequencies",
    [
        [],
        [[100e6, 101e6]],
        [100e6, np.nan],
        [100e6, np.inf],
        [0.0],
        [-1.0],
        [101e6, 100e6],
        [100e6, 100e6],
    ],
)
def test_explicit_frequency_boundary_rejects_invalid_axes(frequencies):
    with pytest.raises(ValueError):
        validate_observation_frequencies(frequencies)


def test_explicit_frequency_boundary_accepts_one_channel():
    np.testing.assert_array_equal(
        validate_observation_frequencies([100e6]),
        [100e6],
    )


def test_removed_raw_frequency_dictionary_argument_is_rejected():
    with pytest.raises(TypeError):
        validate_observation_frequencies(
            obs_frequency_config={"frequencies_hz": [100e6, 101e6]}
        )
