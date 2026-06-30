"""Tests for ``PrepareSkyOptions`` cross-field validation and merge."""

from __future__ import annotations

import numpy as np
import pytest

from radiosim.core.sky import PrepareSkyOptions


class TestValidation:
    def test_default_options_validate(self) -> None:
        opts = PrepareSkyOptions()
        assert opts.representation is None
        assert opts.nside_safety_factor == 5.0
        assert opts.mixed_model_policy == "error"
        assert opts.assume_disjoint is False

    def test_frequencies_xor_obs_freq_config(self) -> None:
        with pytest.raises(ValueError, match="mutually exclusive"):
            PrepareSkyOptions(
                frequencies=np.array([1e8]),
                obs_frequency_config={"starting_frequency": 100},
            )

    def test_frequency_must_be_positive(self) -> None:
        with pytest.raises(ValueError, match="frequency must be strictly positive"):
            PrepareSkyOptions(frequency=0.0)
        with pytest.raises(ValueError, match="frequency must be strictly positive"):
            PrepareSkyOptions(frequency=-1.0)

    def test_safety_factor_must_be_positive(self) -> None:
        with pytest.raises(ValueError, match="nside_safety_factor"):
            PrepareSkyOptions(nside_safety_factor=0.0)
        with pytest.raises(ValueError, match="nside_safety_factor"):
            PrepareSkyOptions(nside_safety_factor=-1.0)

    def test_mixed_model_policy_whitelist(self) -> None:
        for policy in ("error", "warn", "allow"):
            opts = PrepareSkyOptions(mixed_model_policy=policy)
            assert opts.mixed_model_policy == policy
        with pytest.raises(ValueError, match="mixed_model_policy"):
            PrepareSkyOptions(mixed_model_policy="silent")


class TestMerge:
    def test_merged_returns_new_instance(self) -> None:
        opts = PrepareSkyOptions(nside=32)
        out = opts.merged(nside=64)
        assert opts.nside == 32
        assert out.nside == 64
        assert out is not opts

    def test_merged_revalidates(self) -> None:
        opts = PrepareSkyOptions(frequencies=np.array([1e8]))
        with pytest.raises(ValueError, match="mutually exclusive"):
            opts.merged(obs_frequency_config={"starting_frequency": 100})

    def test_merged_no_change_returns_self(self) -> None:
        opts = PrepareSkyOptions(nside=32)
        assert opts.merged() is opts

    def test_merged_unknown_field_raises(self) -> None:
        opts = PrepareSkyOptions()
        with pytest.raises(TypeError, match="unsupported fields"):
            opts.merged(bogus=1)
