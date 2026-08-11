"""PERF-001 P-a acceptance tests for truthful solver memory estimates."""

from __future__ import annotations

import inspect

import pytest

from radiosim.simulator.base import VisibilitySimulator
from radiosim.simulator.rime import RIMESimulator


def test_kernel_source_count_is_a_backward_compatible_keyword_only_input() -> None:
    for method in (
        VisibilitySimulator.get_memory_estimate,
        RIMESimulator.get_memory_estimate,
    ):
        parameter = inspect.signature(method).parameters["kernel_n_sources"]
        assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
        assert parameter.default is None


@pytest.mark.parametrize("polarized", [False, True])
def test_rime_estimate_names_full_matrix_inputs_bounded_leaf_and_assembly(
    polarized: bool,
) -> None:
    estimate = RIMESimulator().get_memory_estimate(
        n_antennas=5,
        n_baselines=10,
        n_sources=100,
        n_frequencies=4,
        n_times=3,
        polarized=polarized,
        kernel_n_sources=128,
    )

    details = estimate["details"]
    assert details["logical_n_sources"] == 100
    assert details["kernel_n_sources"] == 128
    assert details["target_kernel_pairs"] == 131_072
    assert details["max_kernel_baselines"] == 10
    assert details["max_kernel_pair_count"] == 1_280

    breakdown = estimate["breakdown_bytes"]
    assert breakdown["caller_jones_inputs"] == 2 * 10 * 128 * 4 * 16
    assert breakdown["caller_phase_input"] == 10 * 128 * 16
    assert breakdown["caller_array_envelope"] == 10 * 128 * 16
    assert breakdown["contraction_leaf_working"] == 1_280 * 208
    assert breakdown["contraction_output_assembly"] == 2 * 10 * 4 * 16
    assert estimate["output_bytes"] == 10 * 4 * 3 * 4 * 16
    assert breakdown["beam_patterns"] == 5 * 128 * 4 * 16
    assert breakdown["source_only_coherency_or_stokes"] == 128 * 4 * 16
    assert estimate["working_bytes"] == sum(breakdown.values())
    assert estimate["total_bytes"] == (
        estimate["output_bytes"] + estimate["working_bytes"]
    )


def test_bucket_expansion_changes_full_inputs_but_not_the_logical_catalog() -> None:
    simulator = RIMESimulator()
    logical = simulator.get_memory_estimate(4, 9, 65, 3)
    bucketed = simulator.get_memory_estimate(
        4,
        9,
        65,
        3,
        kernel_n_sources=128,
    )

    assert (
        logical["breakdown_bytes"]["source_arrays"]
        == bucketed["breakdown_bytes"]["source_arrays"]
    )
    assert (
        logical["breakdown_bytes"]["direction_cosines"]
        == bucketed["breakdown_bytes"]["direction_cosines"]
    )
    assert (
        logical["breakdown_bytes"]["caller_jones_inputs"]
        < bucketed["breakdown_bytes"]["caller_jones_inputs"]
    )
    for name in (
        "padded_host_directions",
        "direction_batch_host_arrays",
        "padded_host_signal_metadata",
        "backend_source_only_arrays",
        "source_only_coherency_or_stokes",
    ):
        assert logical["breakdown_bytes"][name] < bucketed["breakdown_bytes"][name]
    assert logical["details"]["logical_n_sources"] == 65
    assert bucketed["details"]["logical_n_sources"] == 65
    assert (
        "variable-width spectral coefficients"
        in bucketed["details"]["estimate_limitations"]
    )


def test_leaf_bound_does_not_claim_to_bound_full_inputs_or_output_assembly() -> None:
    simulator = RIMESimulator()
    small = simulator.get_memory_estimate(16, 2_000, 1_024, 1)
    large = simulator.get_memory_estimate(16, 4_000, 1_024, 1)

    assert small["details"]["max_kernel_pair_count"] <= 131_072
    assert large["details"]["max_kernel_pair_count"] <= 131_072
    assert (
        small["breakdown_bytes"]["contraction_leaf_working"]
        == large["breakdown_bytes"]["contraction_leaf_working"]
    )
    assert (
        large["breakdown_bytes"]["caller_jones_inputs"]
        == 2 * small["breakdown_bytes"]["caller_jones_inputs"]
    )
    assert (
        large["breakdown_bytes"]["contraction_output_assembly"]
        == 2 * small["breakdown_bytes"]["contraction_output_assembly"]
    )


def test_one_baseline_exception_reports_source_axis_overshoot_truthfully() -> None:
    estimate = RIMESimulator().get_memory_estimate(
        3,
        4,
        131_073,
        1,
    )

    assert estimate["details"]["max_kernel_baselines"] == 1
    assert estimate["details"]["max_kernel_pair_count"] == 131_073
    assert estimate["details"]["max_kernel_pair_count"] > 131_072


@pytest.mark.parametrize("kernel_n_sources", [False, -1, 4.5, 4])
def test_kernel_source_count_rejects_invalid_or_contracting_values(
    kernel_n_sources: object,
) -> None:
    with pytest.raises((TypeError, ValueError)):
        RIMESimulator().get_memory_estimate(
            3,
            3,
            5,
            1,
            kernel_n_sources=kernel_n_sources,  # type: ignore[arg-type]
        )
