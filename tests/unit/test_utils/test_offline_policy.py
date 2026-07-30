"""Tier 6C: the resolved offline policy is the only loader network authority.

``Tier6HybridRuntimePlan.md`` Section 16.1 makes ``execution.offline`` real:
:func:`radiosim.utils.network.set_offline_policy` installs a process-wide policy,
:func:`radiosim.utils.network.is_online` consults it *before* any socket probe,
and the policy is propagated into every loader worker under both executors.
Invariant S12 (Section 21) is "an offline run performs no socket probe and fails
network-requiring loaders under both executors"; the tests below prove each half.
"""

from __future__ import annotations

import pytest

from radiosim.core.precision import PrecisionConfig
from radiosim.core.sky.operations.parallel import load_models_parallel
from radiosim.utils import network as network_module
from radiosim.utils.network import (
    check_all_services,
    check_service,
    clear_cache,
    is_online,
    offline_policy,
    require_service,
    set_offline_policy,
)


@pytest.fixture(autouse=True)
def _reset_offline_policy():
    """Never leak an installed offline policy into another test."""
    clear_cache()
    set_offline_policy(False)
    yield
    clear_cache()
    set_offline_policy(False)


@pytest.fixture
def forbidden_socket(monkeypatch):
    """Fail the test if any code path attempts a real socket probe."""
    probes: list[tuple[str, int]] = []

    def probe(host: str, port: int, timeout: float) -> bool:
        probes.append((host, port))
        return True

    monkeypatch.setattr(network_module, "_check_socket", probe)
    return probes


# ---------------------------------------------------------------------------
# The policy itself
# ---------------------------------------------------------------------------


def test_the_installed_policy_defaults_to_online():
    assert offline_policy() is False


def test_setting_the_policy_is_observable_and_reversible():
    set_offline_policy(True)
    assert offline_policy() is True
    set_offline_policy(False)
    assert offline_policy() is False


def test_the_policy_rejects_a_non_boolean():
    with pytest.raises(TypeError, match="offline must be a boolean"):
        set_offline_policy(1)  # type: ignore[arg-type]


def test_clear_cache_reinstalls_the_online_policy():
    set_offline_policy(True)
    clear_cache()
    assert offline_policy() is False


# ---------------------------------------------------------------------------
# S12 first half: no socket probe
# ---------------------------------------------------------------------------


def test_forced_offline_short_circuits_is_online_without_a_probe(forbidden_socket):
    set_offline_policy(True)
    assert is_online() is False
    assert forbidden_socket == []


def test_forced_offline_short_circuits_check_service_without_a_probe(forbidden_socket):
    set_offline_policy(True)
    assert check_service("vizier") is False
    assert forbidden_socket == []


def test_forced_offline_short_circuits_check_all_services_without_a_probe(
    forbidden_socket,
):
    set_offline_policy(True)
    status = check_all_services()
    assert status.forced_offline is True
    assert status.internet is False
    assert status.vizier is False
    assert status.casda is False
    assert status.is_online is False
    assert forbidden_socket == []


def test_forced_offline_defeats_a_populated_online_cache(forbidden_socket):
    assert is_online() is True
    assert forbidden_socket  # the online path did probe, and cached the answer
    set_offline_policy(True)
    assert is_online() is False


def test_require_service_raises_from_the_policy_without_a_probe(forbidden_socket):
    set_offline_policy(True)
    with pytest.raises(ConnectionError, match="No internet connection"):
        require_service("vizier", "download catalog 'gleam' from VizieR")
    assert forbidden_socket == []


def test_online_policy_still_probes(forbidden_socket):
    set_offline_policy(False)
    assert is_online() is True
    assert forbidden_socket == [("8.8.8.8", 53)]


# ---------------------------------------------------------------------------
# S12 second half: network-requiring loaders fail under both executors
# ---------------------------------------------------------------------------


_RACS_REQUEST = [("racs", {"band": "low", "flux_limit": 1000.0, "max_rows": 1})]


@pytest.mark.parametrize("executor", ["thread", "process"])
def test_offline_network_loader_fails_under_both_executors(executor, monkeypatch):
    """S12 / Section 27 W5 -- the policy reaches thread *and* process workers."""
    probes: list[tuple[str, int]] = []

    def probe(host: str, port: int, timeout: float) -> bool:
        probes.append((host, port))
        return True

    monkeypatch.setattr(network_module, "_check_socket", probe)
    set_offline_policy(True)

    with pytest.raises(Exception) as excinfo:
        load_models_parallel(
            _RACS_REQUEST,
            1,
            precision=PrecisionConfig.standard(),
            strict=True,
            executor=executor,
        )

    aggregate = excinfo.value
    assert type(aggregate).__name__ == "SkyLoadAggregateError"
    [failure] = aggregate.failures  # type: ignore[attr-defined]
    assert failure.loader_name == "racs"
    assert isinstance(failure.exception, ConnectionError)
    assert "No internet connection" in str(failure.exception)
    # No probe in *this* process; the worker installed the policy before the
    # loader resolved, so it never reached a socket either.
    assert probes == []


def test_online_policy_is_also_propagated_into_a_process_worker():
    """The propagated value is the installed policy, not a hard-coded ``True``."""
    set_offline_policy(False)
    models, record = load_models_parallel(
        [("test_sources", {"num_sources": 2, "distribution": "uniform", "seed": 3})],
        1,
        precision=PrecisionConfig.standard(),
        strict=True,
        executor="process",
    )
    assert record.actual_executor == "process"
    assert len(models) == 1
    assert offline_policy() is False
