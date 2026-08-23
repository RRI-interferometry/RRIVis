"""SCI-004 phase-M1 red oracles for the whole-``SkyModel`` strategy boundary.

``docs/development/sci004_mmode_design.md`` Section 2 records why adding
``MModeSimulator.calculate_visibilities(SourceArrays, ...)`` to the existing
registry would be *false architecture*: at the reviewed source
``VisibilitySimulator.calculate_visibilities`` accepts ``SourceArrays``,
``Simulator.run()`` always calls ``core.hybrid.solve_sky`` itself, and
``solve_sky`` dispatches point sources through the registered object while
calling ``calculate_visibility_healpix`` directly -- so HEALPix and hybrid runs
would bypass the registered strategy entirely.

Phase M1 therefore introduces one immutable ``SkySolveRequest`` and one
``SkySolveOutcome`` at the **whole ``SkyModel``** boundary.
``RIMESimulator.solve(request)`` becomes a thin wrapper around the maintained
point/HEALPix/hybrid path whose arithmetic, component order, source reduction,
result bytes and fingerprints are unchanged; ``MModeSimulator.solve(request)``
consumes the same whole request and never calls the direct kernels. The standing
invariant stays exact::

    accepted values of execution.simulator == simulator registry keys

and after M1 that set is ``{"rime", "mmode"}``.

Section 9 makes capability truth *phase-local*: ``MModeSimulator``
``supports_polarization`` is explicitly overridden to ``False`` in M1, its
``supports_gpu`` is ``False`` without an independently accepted exact-solver
accelerator record, and its request validator rejects any sky with non-zero
Q, U or V using ``mmode_m1_scalar_only``. Only accepted M2 may flip the property
and its named Tier 7 characterization assertion.

Section 14.2 freezes the exact node IDs of the three Stokes rejection cases as
``test_mmode_m1_rejects_nonzero_stokes[Q]``, ``[U]`` and ``[V]``; each sets one
named Stokes value to binary64 one and requires
``radiosim.io.config_resolution.UnsupportedConfigError``, issue
``mmode_m1_scalar_only``, and the exact Section 8 message.

This module also declares the red cases for the two authoritative Tier 7 nodes,
because Section 14.2 groups all six M1 ``capability_cases`` together and the
characterization files themselves carry no red-record machinery. Imports are
function-local so each node yields its own Section 14.1 outcome.
"""

from __future__ import annotations

from typing import Any

import pytest

#: Section 2's post-M1 registry, which is also the accepted config value set.
EXPECTED_REGISTRY_KEYS: frozenset[str] = frozenset({"mmode", "rime"})

#: Section 8's exact ``mmode_m1_scalar_only`` message.
MMODE_M1_SCALAR_ONLY_MESSAGE = (
    "MModeSimulator phase M1 accepts Stokes I only; non-zero Q, U, or V "
    "requires accepted phase M2."
)
MMODE_M1_SCALAR_ONLY_CODE = "mmode_m1_scalar_only"

#: Section 8's exact ``mmode_point_morphology`` message.
MMODE_POINT_MORPHOLOGY_MESSAGE = (
    "execution.simulator='mmode' does not yet support Gaussian point-source "
    "morphology; use rime or remove the morphology."
)
MMODE_POINT_MORPHOLOGY_CODE = "mmode_point_morphology"

#: Section 14.2's two authoritative Tier 7 node IDs.
TIER7_PROPERTY_NODEID = (
    "tests/characterization/test_tier7_current_behavior.py::"
    "test_mmode_m1_capability_truth"
)
TIER7_REGISTRY_NODEID = (
    "tests/unit/test_tier7_jones_acceptance.py::"
    "test_the_accepted_simulator_values_equal_the_registry_keys"
)

_STRATEGY_FIXTURE = b"""\
execution:
  simulator: mmode
sky_representation: point
stokes:
  I: 1.0
  Q: 0.0
  U: 0.0
  V: 0.0
"""


def _stokes_fixture(field: str) -> bytes:
    """One Stokes value set to binary64 one, the others exactly zero."""
    values = {"I": "1.0", "Q": "0.0", "U": "0.0", "V": "0.0"}
    values[field] = "1.0"
    body = "\n".join(f"  {name}: {value}" for name, value in values.items())
    return (
        f"execution:\n  simulator: mmode\nsky_representation: point\nstokes:\n{body}\n"
    ).encode()


_MORPHOLOGY_FIXTURE = b"""\
execution:
  simulator: mmode
sky_representation: point
stokes:
  I: 1.0
  Q: 0.0
  U: 0.0
  V: 0.0
morphology:
  major_arcsec: 120.0
  minor_arcsec: 60.0
  pa_deg: 30.0
"""

_REGISTRY_FIXTURE = b"""\
execution:
  simulator: mmode
expected_registry_keys: ["mmode", "rime"]
"""

_CAPABILITY_FIXTURE = b"""\
capability_cases:
  - {case_kind: property, simulator: mmode, property: supports_polarization, expected: false}
  - {case_kind: property, simulator: rime, property: supports_polarization, expected: true}
"""

_REGISTRY_ORACLE = (
    "tests/unit/test_simulator/test_sci004_strategy.py::"
    "test_the_registry_resolves_the_maintained_rime_strategy_today"
)

_MISSING_SYMBOL_PATTERN = (
    r"ImportError: cannot import name 'MModeSimulator' from 'radiosim\.simulator'"
)


def _case(
    case_id: str,
    requirement_id: str,
    nodeid: str,
    kind: str,
    pattern: str,
    fixture: bytes,
) -> dict[str, Any]:
    return {
        "case_id": case_id,
        "requirement_id": requirement_id,
        "test_nodeid": nodeid,
        "expected_failure_kind": kind,
        "expected_failure_pattern": pattern,
        "fixture_defect_excluded_by": _REGISTRY_ORACLE,
        "fixture_bytes": fixture,
    }


def _local(function: str) -> str:
    return f"tests/unit/test_simulator/test_sci004_strategy.py::{function}"


SCI004_RED_CASES: tuple[dict[str, Any], ...] = (
    _case(
        "m1.strategy.request-outcome-boundary",
        "sci004.section-2.whole-skymodel-request-and-outcome",
        _local("test_the_whole_skymodel_request_and_outcome_types_exist"),
        "missing-symbol",
        (
            r"ImportError: cannot import name 'SkySolve(Outcome|Request)' from "
            r"'radiosim\.simulator\.base'"
        ),
        _STRATEGY_FIXTURE,
    ),
    _case(
        "m1.strategy.registry-equals-config",
        "sci004.section-2.registry-equals-accepted-simulator-values",
        _local("test_the_registry_keys_equal_the_accepted_execution_simulator_values"),
        "assertion",
        r"AssertionError",
        _REGISTRY_FIXTURE,
    ),
    _case(
        "m1.strategy.rime-solve-wrapper",
        "sci004.section-2.rime-solve-thin-wrapper",
        _local("test_rime_simulator_exposes_the_whole_skymodel_solve_wrapper"),
        "assertion",
        r"AssertionError",
        _STRATEGY_FIXTURE,
    ),
    _case(
        "m1.strategy.mmode-registry-entry",
        "sci004.section-2.mmode-resolves-from-the-registry",
        _local("test_mmode_simulator_resolves_from_the_registry"),
        "exception",
        r"ValueError",
        _REGISTRY_FIXTURE,
    ),
    _case(
        "m1.strategy.scalar-only-capability",
        "sci004.section-9.mmode-supports-polarization-false-in-m1",
        _local("test_mmode_simulator_reports_scalar_only_support_in_m1"),
        "missing-symbol",
        _MISSING_SYMBOL_PATTERN,
        _CAPABILITY_FIXTURE,
    ),
    _case(
        "m1.strategy.no-gpu-claim",
        "sci004.section-9.mmode-supports-gpu-false",
        _local("test_mmode_simulator_does_not_claim_gpu_support"),
        "missing-symbol",
        _MISSING_SYMBOL_PATTERN,
        _CAPABILITY_FIXTURE,
    ),
    _case(
        "m1.capability.mmode-rejects-nonzero-q",
        "sci004.section-8.mmode_m1_scalar_only",
        _local("test_mmode_m1_rejects_nonzero_stokes[Q]"),
        "missing-symbol",
        _MISSING_SYMBOL_PATTERN,
        _stokes_fixture("Q"),
    ),
    _case(
        "m1.capability.mmode-rejects-nonzero-u",
        "sci004.section-8.mmode_m1_scalar_only",
        _local("test_mmode_m1_rejects_nonzero_stokes[U]"),
        "missing-symbol",
        _MISSING_SYMBOL_PATTERN,
        _stokes_fixture("U"),
    ),
    _case(
        "m1.capability.mmode-rejects-nonzero-v",
        "sci004.section-8.mmode_m1_scalar_only",
        _local("test_mmode_m1_rejects_nonzero_stokes[V]"),
        "missing-symbol",
        _MISSING_SYMBOL_PATTERN,
        _stokes_fixture("V"),
    ),
    _case(
        "m1.strategy.gaussian-morphology-rejection",
        "sci004.section-8.mmode_point_morphology",
        _local("test_mmode_m1_rejects_gaussian_point_morphology"),
        "missing-symbol",
        _MISSING_SYMBOL_PATTERN,
        _MORPHOLOGY_FIXTURE,
    ),
    # Section 14.2's two authoritative Tier 7 capability nodes. Their fixtures
    # are the capability and registry declarations above, because that is what
    # the two characterization assertions actually pin.
    _case(
        "m1.capability.tier7-property-truth",
        "sci004.section-9.tier7-pins-both-polarization-properties",
        TIER7_PROPERTY_NODEID,
        "missing-symbol",
        _MISSING_SYMBOL_PATTERN,
        _CAPABILITY_FIXTURE,
    ),
    _case(
        "m1.capability.tier7-registry-truth",
        "sci004.section-2.tier7-pins-the-registry-config-equality",
        TIER7_REGISTRY_NODEID,
        "assertion",
        r"AssertionError",
        _REGISTRY_FIXTURE,
    ),
)

SCI004_RED_GREEN_CONTROLS: tuple[str, ...] = (_REGISTRY_ORACLE,)

#: Nodes declared here whose file is *not* this module. The generator groups
#: commands by file, so it needs to know they belong to the Tier 7 files.
SCI004_FOREIGN_RED_NODEIDS: tuple[str, ...] = (
    TIER7_PROPERTY_NODEID,
    TIER7_REGISTRY_NODEID,
)


# --- green control ------------------------------------------------------------


def test_the_registry_resolves_the_maintained_rime_strategy_today() -> None:
    """Section 2: direct RIME stays a maintained strategy, and is resolvable.

    Every red node below asserts something about a *second* registry entry. This
    control proves the registry surface itself works at ``G1``, so the red
    failures are the absence of ``mmode`` and nothing else.
    """
    from typing import get_args

    from radiosim.io.config import ExecutionConfig
    from radiosim.simulator import _SIMULATORS, get_simulator, get_simulator_names

    accepted = set(get_args(ExecutionConfig.model_fields["simulator"].annotation))
    assert accepted == set(_SIMULATORS) == set(get_simulator_names())
    assert "rime" in accepted

    simulator = get_simulator("rime")
    assert simulator.name == "rime"
    assert simulator.supports_polarization is True
    assert simulator.supports_gpu is False

    with pytest.raises(ValueError):
        get_simulator("definitely-not-registered")


# --- Section 2 / 9 red oracles ------------------------------------------------


def test_the_whole_skymodel_request_and_outcome_types_exist() -> None:
    """Section 2: one immutable request and one outcome at the whole-sky boundary."""
    import dataclasses

    from radiosim.simulator.base import SkySolveOutcome, SkySolveRequest

    for candidate in (SkySolveRequest, SkySolveOutcome):
        assert dataclasses.is_dataclass(candidate)
        assert candidate.__dataclass_params__.frozen is True

    fields = {field.name for field in dataclasses.fields(SkySolveRequest)}
    assert {
        "sky_model",
        "source_arrays",
        "instrument",
        "beam_system",
        "location",
        "time_grid",
        "frequencies",
        "receptors",
        "jones",
        "backend",
        "worker_policy",
    } <= fields


def test_the_registry_keys_equal_the_accepted_execution_simulator_values() -> None:
    """Section 2's standing invariant, after M1 registers the second strategy."""
    from typing import get_args

    from radiosim.io.config import ExecutionConfig
    from radiosim.simulator import _SIMULATORS, get_simulator_names

    accepted = set(get_args(ExecutionConfig.model_fields["simulator"].annotation))

    assert accepted == set(_SIMULATORS)
    assert accepted == set(get_simulator_names())
    assert accepted == set(EXPECTED_REGISTRY_KEYS)


def test_rime_simulator_exposes_the_whole_skymodel_solve_wrapper() -> None:
    """Section 2: ``RIMESimulator.solve(request)`` wraps the maintained path."""
    import inspect

    from radiosim.simulator.rime import RIMESimulator

    solve = getattr(RIMESimulator, "solve", None)
    assert callable(solve), "RIMESimulator.solve(request) is absent"
    signature = inspect.signature(solve)
    assert list(signature.parameters) == ["self", "request"]


def test_mmode_simulator_resolves_from_the_registry() -> None:
    """Section 2: the high-level API calls only the selected registered strategy."""
    from radiosim.simulator import get_simulator

    simulator = get_simulator("mmode")

    assert simulator.name == "mmode"
    assert callable(getattr(simulator, "solve", None))


def test_mmode_simulator_does_not_claim_gpu_support() -> None:
    """Section 9: no end-to-end accelerator record exists, so the flag is False."""
    from radiosim.simulator import MModeSimulator

    assert MModeSimulator.supports_gpu is False
    assert MModeSimulator().transform_execution_policy == (
        "host_harmonics_backend_native_dense_v1"
    )


@pytest.mark.parametrize("stokes_field", ["Q", "U", "V"])
def test_mmode_m1_rejects_nonzero_stokes(stokes_field: str) -> None:
    """Section 8/9/14.2: one binary64 one in Q, U or V is ``mmode_m1_scalar_only``.

    Section 14.2 freezes these three node IDs, the exception class, the issue
    code, and the exact message, so an M2-flipped or inherited-base answer here
    fails M1 rather than quietly widening the accepted payload.
    """
    from radiosim.io.config_resolution import UnsupportedConfigError
    from radiosim.simulator import MModeSimulator

    simulator = MModeSimulator()
    stokes = {"I": 1.0, "Q": 0.0, "U": 0.0, "V": 0.0}
    stokes[stokes_field] = 1.0

    with pytest.raises(UnsupportedConfigError) as excinfo:
        simulator.validate_scalar_sky_payload(stokes)

    issues = [
        issue
        for issue in excinfo.value.issues
        if issue.code == MMODE_M1_SCALAR_ONLY_CODE
    ]
    assert issues
    assert issues[0].message == MMODE_M1_SCALAR_ONLY_MESSAGE


def test_mmode_m1_rejects_gaussian_point_morphology() -> None:
    """Section 7.1/8: Gaussian morphology is not one common sky field in M1."""
    from radiosim.io.config_resolution import UnsupportedConfigError
    from radiosim.simulator import MModeSimulator

    simulator = MModeSimulator()

    with pytest.raises(UnsupportedConfigError) as excinfo:
        simulator.validate_point_morphology(
            major_arcsec=120.0, minor_arcsec=60.0, pa_deg=30.0
        )

    issues = [
        issue
        for issue in excinfo.value.issues
        if issue.code == MMODE_POINT_MORPHOLOGY_CODE
    ]
    assert issues
    assert issues[0].message == MMODE_POINT_MORPHOLOGY_MESSAGE
