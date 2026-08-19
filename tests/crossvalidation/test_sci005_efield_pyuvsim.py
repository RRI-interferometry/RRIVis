"""SCI-005 Stage-3 full-efield comparison against ``pyuvsim`` 1.4.0.

``docs/development/sci005_beam_physics_plan.md`` Section 5.5 grants one
optional Stage-3 comparison: the same full-efield UVBeam file, full-Stokes
point sky, times, frequencies, antennas and accepted east-X/fringe mapping in
both simulators, recording per-correlation absolute and relative residuals, the
exact source commit, input hashes and every convention mapping.

**It never gates.**  Every test here is marked ``crossval`` *and* ``slow``, so
``pixi run test -- -m "not slow"`` -- the standard gate, and the two CI suites
-- deselect the module entirely, and the full-suite run in the ``default``
environment skips it at import because ``pyuvsim`` is not installed there.  It
runs only under::

    pixi run --environment crossval -- python -m pytest \\
      tests/crossvalidation/ -m crossval

The comparison itself lives in ``tools/sci005_stage3_crossvalidation.py``,
which Section 7.4 grants as the non-gating artifact generator and standalone
validator.  This module imports that tool and asserts the bounds it declares,
so the numbers a generated artifact records and the numbers asserted here come
from one measurement and one implementation rather than two that can drift.
The grant's own wording contemplates exactly this: the tool "is never imported
by production or by a gating test", and nothing in this module gates.

Section 5.5's language discipline applies to every sentence written from this
module's result.  It licenses only "compared against pyuvsim for the named
fixture, with the recorded agreements and open disagreements", and never an
unqualified "validated against pyuvsim".
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

pytestmark = [pytest.mark.crossval, pytest.mark.slow]

pyuvsim = pytest.importorskip(
    "pyuvsim",
    reason=(
        "pyuvsim is only present in the optional `crossval` pixi environment; "
        "run `pixi run --environment crossval -- python -m pytest "
        "tests/crossvalidation/ -m crossval`"
    ),
)

REPO_ROOT = Path(__file__).resolve().parents[2]
TOOL_PATH = REPO_ROOT / "tools" / "sci005_stage3_crossvalidation.py"


def _load_tool() -> ModuleType:
    """Import the granted generator/validator by path.

    ``tools/`` is not an importable package, so the module is loaded from its
    exact repository path.  Its own module-level imports are standard library
    only; every scientific dependency it uses is imported lazily inside the
    comparison, which is what lets its ``validate`` form run in the standard
    environment.
    """
    name = "_radiosim_sci005_stage3_crossvalidation"
    existing = sys.modules.get(name)
    if existing is not None:
        return existing
    specification = importlib.util.spec_from_file_location(name, TOOL_PATH)
    assert specification is not None and specification.loader is not None, TOOL_PATH
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    try:
        specification.loader.exec_module(module)
    except Exception:
        sys.modules.pop(name, None)
        raise
    return module


tool = _load_tool()


@pytest.fixture(scope="module")
def comparison(tmp_path_factory):
    """Run the Stage-3 comparison exactly once for the whole module."""
    directory = tmp_path_factory.mktemp("sci005-stage3-crossval")
    return tool.run_comparison(Path(directory))


def test_installed_reference_versions_are_the_recorded_pins():
    """The artifact records exact versions, never "latest"."""
    import pyuvdata

    assert pyuvsim.__version__ == tool.REFERENCE_VERSION
    assert pyuvdata.__version__ == tool.PYUVDATA_VERSION


def test_correlation_residuals_are_the_complete_output_basis_label_set(comparison):
    """Every reported correlation carries a complete, well-formed residual row.

    The residual set is exactly the label set ``core/polarization_basis.py``
    gives the resolved output basis, and each row is finite, non-negative and
    internally consistent.  ``reference_max_abs > 0`` is what makes the
    relative residual meaningful rather than a ratio of two near-zero numbers.
    """
    from radiosim.core.polarization_basis import CORRELATION_LABELS

    expected = CORRELATION_LABELS[comparison.output_basis]
    assert comparison.correlations == expected
    assert set(comparison.residuals) == set(expected)

    for label in expected:
        absolute, relative, reference_max = comparison.residuals[label]
        assert np.isfinite([absolute, relative, reference_max]).all(), label
        assert absolute >= 0.0, label
        assert relative >= 0.0, label
        assert reference_max > 0.0, label
        assert relative == pytest.approx(absolute / reference_max, rel=1e-12), label


def test_the_comparison_is_not_vacuous(comparison):
    """Undoing the fringe convention mapping must move the cubes by order unity.

    Without this control the residual assertions below could be satisfied by
    two arrays that happen to be small, rather than by two simulators that
    agree about the fringe.
    """
    assert comparison.reference_scale > 1.0
    assert (
        comparison.control_relative_without_fringe_mapping > tool.CONTROL_RELATIVE_FLOOR
    ), comparison.control_relative_without_fringe_mapping


def test_total_intensity_agrees_within_the_declared_frame_invariant_bound(comparison):
    """``XX + YY`` cannot carry the tangent-frame difference.

    A unitary rotation ``R`` of the tangent basis sends ``J -> J R`` and
    ``B -> R^H B R``, which leaves ``Tr(J_1 B J_2^H)`` unchanged.  The residual
    frame difference between RadioSim's parallactic angle and pyradiosky's
    exact tangent-basis rotation (WP-6 SCI-007) therefore cannot enter the
    trace, so what this bounds is the full-efield beam evaluation, the fringe,
    the flux normalization and double-precision round-off.
    """
    assert comparison.trace_relative < tool.TRACE_RELATIVE_BOUND, (
        comparison.trace_relative
    )


def test_per_correlation_residuals_meet_the_declared_bound(comparison):
    """Each individual correlation agrees to the declared relative bound.

    An individual correlation is *not* frame invariant, so this bound sits
    above the known milli-radian frame effect SCI-007 measured and far below
    the order-unity disagreement a wrong convention mapping produces.  A
    measured excess is not silently tolerated: the generator records it in the
    artifact's ``open_disagreements``, and this assertion makes it loud.
    """
    exceeded = {
        label: values[1]
        for label, values in comparison.residuals.items()
        if values[1] > tool.PER_CORRELATION_RELATIVE_BOUND
    }
    assert not exceeded, exceeded


def test_open_disagreements_enumerate_exactly_the_exceeded_bounds(comparison):
    """The recorded disagreement list is sorted, unique and exhaustive."""
    entries = list(comparison.open_disagreements)
    assert entries == sorted(set(entries))

    expected = {
        f"{label}: max_rel_residual exceeds the declared per-correlation bound "
        f"{tool.PER_CORRELATION_RELATIVE_BOUND!r}"
        for label, values in comparison.residuals.items()
        if values[1] > tool.PER_CORRELATION_RELATIVE_BOUND
    }
    if comparison.trace_relative > tool.TRACE_RELATIVE_BOUND:
        expected.add(
            "total_intensity: the frame-invariant XX + YY residual exceeds the "
            f"declared bound {tool.TRACE_RELATIVE_BOUND!r}"
        )
    assert set(entries) == expected


def test_every_named_input_is_hashed(comparison):
    """The exact bytes fed to both simulators are recorded by content digest."""
    names = [name for name, _ in comparison.input_hashes]
    assert names == sorted(set(names))
    assert set(tool.REQUIRED_INPUT_NAMES) <= set(names)
    for _, digest in comparison.input_hashes:
        assert len(digest) == 64
        assert set(digest) <= set("0123456789abcdef")


def test_the_measured_comparison_builds_an_artifact_its_validator_accepts(comparison):
    """The generator and the standalone validator agree about this measurement.

    The record is assembled and validated in memory: no artifact is written,
    because the only admissible way for those bytes to exist is the frozen
    ``generate`` command run from a clean approved source commit.
    """
    approved = "0" * 40
    record = tool.build_record(
        approved_source_sha=approved,
        comparison=comparison,
        commands=[
            {
                "argv": ["git", "rev-parse", "HEAD"],
                "cwd": ".",
                "pixi_environment": "crossval",
                "started_at_utc": "2026-01-01T00:00:00Z",
                "duration_seconds": 0.0,
                "exit_code": 0,
                "stdout_sha256": "0" * 64,
                "stderr_sha256": "0" * 64,
            }
        ],
    )
    assert tuple(record) == tool.ARTIFACT_KEY_ORDER
    assert record["gating"] is False
    assert record["reference_package"] == "pyuvsim"
    assert record["output_basis"] == comparison.output_basis
    tool.validate_record(record, approved_source_sha=approved)
