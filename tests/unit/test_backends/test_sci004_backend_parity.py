r"""SCI-004 phase-M2 red oracles for the m-mode backend contract.

``docs/development/sci004_mmode_design.md`` Section 9 is deliberately narrow
about what a backend may do here:

    NumPy is the scientific reference.  Astropy frame work, IERS mapping, beam
    sampling, HEALPix geometry, and scalar/spin harmonic transforms are host-side
    NumPy work for every backend.  JAX and Dask may execute **only** the dense
    per-``m`` contractions and time synthesis.

That split is recorded as ``host_harmonics_backend_native_dense_v1`` and is
explicitly *not* an end-to-end accelerator claim: ``MModeSimulator.supports_gpu``
stays ``False`` without an independently accepted exact-solver accelerator
record, and register row ``PERF-001`` governs every performance statement.  This
module therefore binds *behavioural parity*, never speed.

The predicates are fixed by Section 9 and may not be widened to admit a backend:

* complex128 -- ``rtol = 1e-12``, ``atol = 1e-12 * max(1, max(abs(reference)))``;
* complex64 -- a *separately named* row at ``rtol = 5e-5``,
  ``atol = 5e-6 * max(1, max(abs(reference)))``, which "is a new low-precision
  contract and cannot replace the complex128 acceptance row".

Complex128 on JAX requires x64 and must fail *explicitly* if it is unavailable
rather than silently demoting to complex64 -- a silent demotion would turn the
acceptance row into the low-precision row without saying so.

``execution.solver.workers`` owns independent frequency-block construction and is
clamped to the frequency count; blocks are assembled in canonical frequency
order, so one worker and many workers must meet the same predicate.

The Section 13.4 owner is ``radiosim.core.mmode.solver``, whose backend-routed
dense entry points do not exist at ``A1``; imports are function-local so each
node yields its own Section 14.1 outcome.
"""

from __future__ import annotations

from typing import Any

import numpy as np

#: Section 9's fixed complex128 acceptance predicate.
COMPLEX128_RTOL = 1e-12
COMPLEX128_ATOL_FACTOR = 1e-12

#: Section 9's separately named complex64 row.  It never replaces the row above.
COMPLEX64_RTOL = 5e-5
COMPLEX64_ATOL_FACTOR = 5e-6

#: Section 9's recorded transform-execution policy literal.
MMODE_EXECUTION_POLICY = "host_harmonics_backend_native_dense_v1"

#: The backends Section 9 admits for the dense work, NumPy first.
BACKEND_NAMES: tuple[str, ...] = ("numpy", "jax", "dask")

#: Section 5.3's science field order.
FIELD_ORDER: tuple[str, ...] = ("I", "+2", "-2", "V")

N_BASELINE = 3
N_FREQUENCY = 4
N_PACKED = 11
N_SIGNED_M = 5

_PARITY_FIXTURE = f"""\
transform_execution_policy: {MMODE_EXECUTION_POLICY}
backends: ["numpy", "jax", "dask"]
predicate_id: sci004_backend_complex128.v1
rtol: {COMPLEX128_RTOL}
atol_factor: {COMPLEX128_ATOL_FACTOR}
accumulation_dtype: complex128
result_dtype: complex128
""".encode()

_SYNTHESIS_FIXTURE = f"""\
transform_execution_policy: {MMODE_EXECUTION_POLICY}
backends: ["numpy", "jax", "dask"]
stage: time_synthesis
signed_m: {N_SIGNED_M}
rtol: {COMPLEX128_RTOL}
""".encode()

_COMPLEX64_FIXTURE = f"""\
transform_execution_policy: {MMODE_EXECUTION_POLICY}
backends: ["numpy", "jax", "dask"]
predicate_id: sci004_backend_complex64.v1
rtol: {COMPLEX64_RTOL}
atol_factor: {COMPLEX64_ATOL_FACTOR}
accumulation_dtype: complex64
result_dtype: complex64
""".encode()

_WORKER_FIXTURE = f"""\
transform_execution_policy: {MMODE_EXECUTION_POLICY}
workers: [1, 2, 4, 64]
n_frequencies: {N_FREQUENCY}
block_order: canonical_frequency
""".encode()

_X64_FIXTURE = b"""\
backend: jax
accumulation_dtype: complex128
requires_x64: true
silent_demotion_allowed: false
"""

_PARITY_ORACLE = (
    "tests/unit/test_backends/test_sci004_backend_parity.py::"
    "test_the_three_backends_and_the_section_9_predicate_hold_today"
)

_SOLVER_IMPORT_PATTERN = (
    r"ImportError: cannot import name '\w+' from 'radiosim\.core\.mmode\.solver'"
)


def _local(function: str) -> str:
    return f"tests/unit/test_backends/test_sci004_backend_parity.py::{function}"


def _case(
    case_id: str,
    requirement_id: str,
    function: str,
    fixture: bytes,
) -> dict[str, Any]:
    return {
        "case_id": case_id,
        "requirement_id": requirement_id,
        "test_nodeid": _local(function),
        "expected_failure_kind": "missing-symbol",
        "expected_failure_pattern": _SOLVER_IMPORT_PATTERN,
        "fixture_defect_excluded_by": _PARITY_ORACLE,
        "fixture_bytes": fixture,
    }


SCI004_PHASE2_RED_CASES: tuple[dict[str, Any], ...] = (
    _case(
        "m2.backend.per-m-contraction-parity",
        "sci004.section-9.per-m-contraction-complex128-parity",
        "test_the_per_m_contraction_meets_the_complex128_predicate_on_every_backend",
        _PARITY_FIXTURE,
    ),
    _case(
        "m2.backend.time-synthesis-parity",
        "sci004.section-9.time-synthesis-complex128-parity",
        "test_the_time_synthesis_meets_the_complex128_predicate_on_every_backend",
        _SYNTHESIS_FIXTURE,
    ),
    _case(
        "m2.backend.complex64-named-row",
        "sci004.section-9.complex64-is-a-separate-named-row",
        "test_the_named_complex64_row_uses_its_own_wider_predicate",
        _COMPLEX64_FIXTURE,
    ),
    _case(
        "m2.backend.worker-invariance",
        "sci004.section-9.worker-count-does-not-change-the-result",
        "test_the_worker_count_never_changes_the_contracted_result",
        _WORKER_FIXTURE,
    ),
    _case(
        "m2.backend.jax-x64-explicit-failure",
        "sci004.section-9.complex128-jax-requires-x64-explicitly",
        "test_complex128_on_jax_requires_x64_and_fails_explicitly",
        _X64_FIXTURE,
    ),
)

SCI004_PHASE2_RED_GREEN_CONTROLS: tuple[str, ...] = (_PARITY_ORACLE,)


# --- helpers ------------------------------------------------------------------


def _blocks(dtype: str = "complex128") -> tuple[np.ndarray, np.ndarray]:
    """A deterministic transfer/sky block pair for the dense per-``m`` product."""
    rng = np.random.default_rng(20260823)
    transfer = (
        rng.normal(size=(N_BASELINE, N_FREQUENCY, 4, len(FIELD_ORDER), N_PACKED))
        + 1j * rng.normal(size=(N_BASELINE, N_FREQUENCY, 4, len(FIELD_ORDER), N_PACKED))
    ).astype(dtype)
    sky = (
        rng.normal(size=(N_FREQUENCY, len(FIELD_ORDER), N_PACKED))
        + 1j * rng.normal(size=(N_FREQUENCY, len(FIELD_ORDER), N_PACKED))
    ).astype(dtype)
    return (transfer, sky)


def _assert_within(
    candidate: np.ndarray,
    reference: np.ndarray,
    *,
    rtol: float,
    atol_factor: float,
    label: str,
) -> None:
    """Assert Section 9's scale-aware predicate, recomputed here from its text."""
    scale = max(1.0, float(np.max(np.abs(reference))))
    atol = atol_factor * scale
    deviation = np.abs(candidate - reference)
    assert bool(np.all(deviation <= atol + rtol * np.abs(reference))), (
        f"{label}: max deviation {float(np.max(deviation))!r} exceeds {atol!r}"
    )


# --- green control ------------------------------------------------------------


def test_the_three_backends_and_the_section_9_predicate_hold_today() -> None:
    """All three backends resolve at ``A1`` and already meet the predicate.

    Section 9's split is that JAX and Dask may execute only the dense work.  The
    dense work itself -- a complex128 contraction over the packed axis -- already
    agrees across the three backends to well inside ``rtol = 1e-12``, so a red
    failure below is the absence of the m-mode entry points that would route
    through them, not a broken backend or an unattainable tolerance.

    The two claims Section 9 forbids are pinned negatively here: the recorded
    policy literal is not an accelerator claim, and ``supports_gpu`` is ``False``.
    """
    from radiosim.backends import get_backend, list_backends
    from radiosim.simulator import MModeSimulator

    available = list_backends()
    for name in BACKEND_NAMES:
        assert available.get(name) is True, name

    transfer, sky = _blocks()
    subscripts = "bfcxp,fxp->bfc"
    reference = np.einsum(subscripts, transfer, sky, optimize=True)
    for name in BACKEND_NAMES:
        backend = get_backend(name)
        candidate = np.asarray(
            backend.to_numpy(
                backend.xp.einsum(
                    subscripts,
                    backend.asarray(transfer),
                    backend.asarray(sky),
                )
            )
        )
        _assert_within(
            candidate,
            reference,
            rtol=COMPLEX128_RTOL,
            atol_factor=COMPLEX128_ATOL_FACTOR,
            label=name,
        )

    assert MModeSimulator.supports_gpu is False
    assert MModeSimulator().transform_execution_policy == MMODE_EXECUTION_POLICY


# --- Section 9 red oracles ----------------------------------------------------


def test_the_per_m_contraction_meets_the_complex128_predicate_on_every_backend() -> (
    None
):
    """Section 9: the dense per-``m`` contraction, NumPy as the reference."""
    from radiosim.backends import get_backend
    from radiosim.core.mmode.solver import contract_per_m_block

    transfer, sky = _blocks()
    reference = np.asarray(
        contract_per_m_block(
            transfer_block=transfer,
            sky_block=sky,
            field_order=FIELD_ORDER,
            backend=get_backend("numpy"),
        )
    )
    assert reference.dtype == np.complex128
    assert reference.shape == (N_BASELINE, N_FREQUENCY, 4)

    for name in ("jax", "dask"):
        candidate = np.asarray(
            contract_per_m_block(
                transfer_block=transfer,
                sky_block=sky,
                field_order=FIELD_ORDER,
                backend=get_backend(name),
            )
        )
        _assert_within(
            candidate,
            reference,
            rtol=COMPLEX128_RTOL,
            atol_factor=COMPLEX128_ATOL_FACTOR,
            label=name,
        )


def test_the_time_synthesis_meets_the_complex128_predicate_on_every_backend() -> None:
    """Section 6/9: ``V_k = sum_m w_m v_m exp(+i 2 pi m u_k)`` on each backend.

    The exposure ``sinc`` is a diagonal ``w_m`` factor rather than a spectral
    taper, and the turns ``u_k`` come from the retained exact-turn grid; a
    backend may execute this sum but may not regenerate the topology from ``k``,
    ``N``, radians, or ``tau``.
    """
    from radiosim.backends import get_backend
    from radiosim.core.mmode.solver import synthesize_time_series

    rng = np.random.default_rng(20260824)
    modes = (
        rng.normal(size=(N_BASELINE, N_FREQUENCY, 4, N_SIGNED_M))
        + 1j * rng.normal(size=(N_BASELINE, N_FREQUENCY, 4, N_SIGNED_M))
    ).astype(np.complex128)
    samples = 2 * N_SIGNED_M + 1
    turns = [f"{index}/{samples}" for index in range(samples)]

    reference = np.asarray(
        synthesize_time_series(
            mode_cube=modes, era_turns=turns, backend=get_backend("numpy")
        )
    )
    assert reference.shape == (samples, N_BASELINE, N_FREQUENCY, 4)

    for name in ("jax", "dask"):
        candidate = np.asarray(
            synthesize_time_series(
                mode_cube=modes, era_turns=turns, backend=get_backend(name)
            )
        )
        _assert_within(
            candidate,
            reference,
            rtol=COMPLEX128_RTOL,
            atol_factor=COMPLEX128_ATOL_FACTOR,
            label=name,
        )


def test_the_named_complex64_row_uses_its_own_wider_predicate() -> None:
    """Section 9: complex64 is a separate row and never replaces the acceptance row.

    The wider tolerance is licensed only for the separately named low-precision
    contract, so the same complex64 result is required to *fail* the complex128
    acceptance predicate: if it passed, the two rows would be indistinguishable
    and the wider tolerance would have silently become the acceptance one.
    """
    from radiosim.backends import get_backend
    from radiosim.core.mmode.solver import contract_per_m_block

    transfer, sky = _blocks()
    reference = np.asarray(
        contract_per_m_block(
            transfer_block=transfer,
            sky_block=sky,
            field_order=FIELD_ORDER,
            backend=get_backend("numpy"),
        )
    )
    low = np.asarray(
        contract_per_m_block(
            transfer_block=transfer.astype("complex64"),
            sky_block=sky.astype("complex64"),
            field_order=FIELD_ORDER,
            backend=get_backend("numpy"),
            accumulation_dtype="complex64",
        )
    )
    assert low.dtype == np.complex64

    _assert_within(
        low.astype(np.complex128),
        reference,
        rtol=COMPLEX64_RTOL,
        atol_factor=COMPLEX64_ATOL_FACTOR,
        label="complex64",
    )
    scale = max(1.0, float(np.max(np.abs(reference))))
    deviation = np.abs(low.astype(np.complex128) - reference)
    assert bool(
        np.any(
            deviation
            > COMPLEX128_ATOL_FACTOR * scale + COMPLEX128_RTOL * np.abs(reference)
        )
    ), "a complex64 row that passes the complex128 predicate is not a separate row"


def test_the_worker_count_never_changes_the_contracted_result() -> None:
    """Section 9: workers own independent frequency blocks, in canonical order.

    ``workers`` is clamped to the frequency count, and blocks are assembled in
    canonical frequency order, so one worker and many workers meet the same
    predicate -- bit-identically, since the same values are summed in the same
    order rather than merely to within a tolerance.
    """
    from radiosim.backends import get_backend
    from radiosim.core.mmode.solver import contract_per_m_block

    transfer, sky = _blocks()
    backend = get_backend("numpy")
    reference = np.asarray(
        contract_per_m_block(
            transfer_block=transfer,
            sky_block=sky,
            field_order=FIELD_ORDER,
            backend=backend,
            workers=1,
        )
    )
    for workers in (2, 4, 64):
        candidate = np.asarray(
            contract_per_m_block(
                transfer_block=transfer,
                sky_block=sky,
                field_order=FIELD_ORDER,
                backend=backend,
                workers=workers,
            )
        )
        assert np.array_equal(candidate, reference), workers


def test_complex128_on_jax_requires_x64_and_fails_explicitly() -> None:
    """Section 9: "Complex128 JAX requires x64 and fails explicitly if unavailable".

    The failure has to be explicit because the silent alternative -- demoting to
    complex64 -- would substitute the separately named low-precision contract for
    the complex128 acceptance row without any record that it happened.
    """
    from radiosim.core.mmode.solver import require_backend_complex128

    resolved = require_backend_complex128("jax")
    assert resolved.dtype_name == "complex128"
    assert resolved.x64_enabled is True

    raised = None
    try:
        require_backend_complex128("jax", x64_enabled=False)
    except RuntimeError as error:  # pragma: no cover - the red path
        raised = error
    assert raised is not None, (
        "complex128 on JAX without x64 is an explicit failure, never a demotion"
    )
    assert "x64" in str(raised)
