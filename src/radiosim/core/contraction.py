"""The one compiled solver kernel: the per-(time, frequency) contraction.

``Tier6HybridRuntimePlan.md`` Section 13.6 authorizes exactly **one** compiled
kernel in RadioSim, and this module is it: the baseline-batched contraction that
turns the per-antenna Jones matrices, the coherency matrices, the geometric
phase, and the Gaussian envelope for one ``(time, frequency)`` pair into the one
``(B, 2, 2)`` block that Section 13.3's accumulation assembles.

Three properties make the private six-argument leaf the right and only
compilation boundary:

- it is pure -- every input is an array, nothing is read from or written to
  enclosing state;
- P-b stabilizes its source shape and P-a presents ordered baseline chunks; a
  full chunk and uneven tail may have different baseline shapes, but each
  recurring leaf signature is reusable;
- it is where essentially all of the solver's floating-point work happens, while
  everything around it (astropy coordinate transforms, the horizon mask, the
  Planck conversion, pyuvdata beam interpolation) is host-side by nature and is
  explicitly out of scope for compilation.

The uncompiled :func:`baseline_contraction` function is the reference leaf and
is what NumPy and Dask execute. :func:`baseline_contraction_for` always returns
an uncompiled Python scheduling wrapper. The wrapper chunks only baseline-
bearing operands, calls one private six-argument leaf in baseline order, and
concatenates its ``(chunk_B, 2, 2)`` outputs. A backend whose
``supports_compilation`` is ``True`` gets ``backend.compile`` applied exactly
once to that private leaf, never to the scheduling wrapper; there is no separate
"enable jit" switch.

``vmap`` is deliberately not used: each leaf is already expressed as batched
array operations over a leading baseline axis, which XLA fuses directly, and
Section 13.6 permits ``vmap`` only *inside* that leaf, never over the wrapper,
time, or frequency axes.
"""

import operator
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from radiosim.backends.base import ArrayBackend

__all__ = ["baseline_contraction", "baseline_contraction_for"]


#: Maximum desired baseline-source pairs in one contraction leaf invocation.
#: It is a target rather than a hard cap because preserving the accepted source
#: reduction order means a source axis larger than this value still executes one
#: complete baseline at a time.
_TARGET_KERNEL_PAIRS = 131_072


def baseline_contraction(
    jones_p: Any,
    jones_q: Any,
    coherency: Any,
    phase: Any,
    envelope: Any,
    stokes_i: Any,
    *,
    backend: "ArrayBackend",
) -> Any:
    """Contract one ``(time, frequency)`` step over sources, for all baselines.

    Computes, for every baseline ``b``::

        V[b] = sum_s  weight[b, s] * (J_p[b, s] @ C[s] @ J_q[b, s]^H)

    Parameters
    ----------
    jones_p, jones_q : array
        Per-baseline, per-source antenna Jones matrices, shape ``(B, S, 2, 2)``.
    coherency : array or None
        Per-source coherency matrices, shape ``(S, 2, 2)``. ``None`` selects the
        unpolarized specialization, which skips both coherency products and
        applies the Stokes I scaling as a scalar instead. The two forms are not
        interchangeable bit-for-bit -- floating-point multiplication is not
        associative -- so which one runs is part of the scientific identity of a
        run, not an optimization detail.
    phase : array
        Geometric phase ``exp(-2*pi*i*b.s)``, shape ``(B, S)``.
    envelope : array or float
        Gaussian morphology attenuation, shape ``(B, S)``, or the scalar ``1.0``
        when no source is resolved.
    stokes_i : array or None
        Per-source Stokes I in Jy, shape ``(S,)``. Required exactly when
        ``coherency`` is ``None``.
    backend : ArrayBackend
        Array backend supplying ``matmul``, ``conjugate_transpose``, and ``sum``.

    Returns
    -------
    array
        One ``(B, 2, 2)`` block, in the backend's own array domain and in
        selected-baseline order.

    Notes
    -----
    ``coherency`` and ``stokes_i`` are ``None``-switched rather than passed as a
    boolean flag on purpose: ``None`` is pytree *structure* to JAX, so the branch
    is resolved once at trace time and never becomes a traced value inside the
    compiled graph.
    """
    jones_q_hermitian = backend.conjugate_transpose(jones_q)

    if coherency is None:
        if stokes_i is None:
            raise ValueError(
                "baseline_contraction requires stokes_i when coherency is None"
            )
        product = backend.matmul(jones_p, jones_q_hermitian)
        weight = stokes_i * phase * envelope / 2.0
    else:
        product = backend.matmul(backend.matmul(jones_p, coherency), jones_q_hermitian)
        weight = phase * envelope

    return backend.sum(product * weight[..., None, None], axis=-3)


def _require_target_kernel_pairs(value: int | None) -> int | None:
    """Return one valid leaf pair target, or the explicit unbounded control."""
    if value is None:
        return None
    if isinstance(value, bool):
        raise TypeError("target_kernel_pairs must be a positive integer or None")
    try:
        target = operator.index(value)
    except TypeError as exc:
        raise TypeError(
            "target_kernel_pairs must be a positive integer or None"
        ) from exc
    if target <= 0:
        raise ValueError("target_kernel_pairs must be positive")
    return target


def _baseline_contraction_for_policy(
    backend: "ArrayBackend",
    *,
    target_kernel_pairs: int | None,
) -> Any:
    """Build the contraction scheduler for one explicit pair-count policy.

    ``None`` is the retained unbounded control used by PERF-001 comparisons.
    A positive target chunks only the baseline-bearing operands. The source
    axis is never split because doing so would change its accepted floating-
    point reduction order.
    """
    target = _require_target_kernel_pairs(target_kernel_pairs)

    def leaf(
        jones_p: Any,
        jones_q: Any,
        coherency: Any,
        phase: Any,
        envelope: Any,
        stokes_i: Any,
    ) -> Any:
        return baseline_contraction(
            jones_p,
            jones_q,
            coherency,
            phase,
            envelope,
            stokes_i,
            backend=backend,
        )

    compiled_leaf = backend.compile(leaf) if backend.supports_compilation else leaf

    def kernel(
        jones_p: Any,
        jones_q: Any,
        coherency: Any,
        phase: Any,
        envelope: Any,
        stokes_i: Any,
    ) -> Any:
        n_baselines = int(jones_p.shape[0])
        n_sources = int(jones_p.shape[1])
        if target is None or n_baselines == 0 or n_sources == 0:
            return compiled_leaf(
                jones_p,
                jones_q,
                coherency,
                phase,
                envelope,
                stokes_i,
            )

        chunk_baselines = max(1, min(n_baselines, target // n_sources))
        if chunk_baselines == n_baselines:
            return compiled_leaf(
                jones_p,
                jones_q,
                coherency,
                phase,
                envelope,
                stokes_i,
            )

        envelope_shape = getattr(envelope, "shape", None)
        envelope_is_scalar = envelope_shape is None or len(envelope_shape) == 0
        chunks: list[Any] = []
        for start in range(0, n_baselines, chunk_baselines):
            stop = min(start + chunk_baselines, n_baselines)
            chunk_envelope = envelope if envelope_is_scalar else envelope[start:stop]
            chunks.append(
                compiled_leaf(
                    jones_p[start:stop],
                    jones_q[start:stop],
                    coherency,
                    phase[start:stop],
                    chunk_envelope,
                    stokes_i,
                )
            )
        return backend.xp.concatenate(chunks, axis=0)

    return kernel


def baseline_contraction_for(backend: "ArrayBackend") -> Any:
    """Return the production contraction scheduler for one solver call.

    Build this **once per solver call**, above the time loop, and reuse it. A
    freshly created closure on every step would defeat the compilation cache and
    turn a one-off compile into a per-step compile.

    Returns
    -------
    callable
        ``kernel(jones_p, jones_q, coherency, phase, envelope, stokes_i)``. It
        schedules baseline chunks in Python around exactly one compiled leaf
        when the backend supports compilation, or one plain leaf otherwise.
    """
    return _baseline_contraction_for_policy(
        backend,
        target_kernel_pairs=_TARGET_KERNEL_PAIRS,
    )
