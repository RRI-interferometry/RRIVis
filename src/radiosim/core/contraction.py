"""The one compiled solver kernel: the per-(time, frequency) contraction.

``Tier6HybridRuntimePlan.md`` Section 13.6 authorizes exactly **one** compiled
kernel in RadioSim, and this module is it: the baseline-batched contraction that
turns the per-antenna Jones matrices, the coherency matrices, the geometric
phase, and the Gaussian envelope for one ``(time, frequency)`` pair into the one
``(B, 2, 2)`` block that Section 13.3's accumulation assembles.

Three properties make it the right and only compilation boundary:

- it is pure -- every input is an array, nothing is read from or written to
  enclosing state;
- it is shape-stable within a ``(time, frequency)`` step and dtype-stable within
  a run, so a compiled form is reusable;
- it is where essentially all of the solver's floating-point work happens, while
  everything around it (astropy coordinate transforms, the horizon mask, the
  Planck conversion, pyuvdata beam interpolation) is host-side by nature and is
  explicitly out of scope for compilation.

The uncompiled function is the reference implementation and is what the NumPy
and Dask backends always execute; :func:`baseline_contraction_for` returns it
unchanged for any backend whose ``supports_compilation`` is ``False``. A backend
that reports ``True`` gets ``backend.compile`` applied to it and nothing else --
there is no separate "enable jit" switch, because a backend that advertised
compilation and then did not compile would be exactly the kind of unfulfilled
capability claim Tier 6 exists to remove.

``vmap`` is deliberately not used: the contraction is already expressed as
batched array operations over a leading baseline axis, which XLA fuses directly,
and Section 13.6 permits ``vmap`` only *inside* this kernel, never over the time
or frequency axes.
"""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from radiosim.backends.base import ArrayBackend

__all__ = ["baseline_contraction", "baseline_contraction_for"]


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


def baseline_contraction_for(backend: "ArrayBackend") -> Any:
    """Return the contraction callable to use for one solver call.

    Build this **once per solver call**, above the time loop, and reuse it. A
    freshly created closure on every step would defeat the compilation cache and
    turn a one-off compile into a per-step compile.

    Returns
    -------
    callable
        ``kernel(jones_p, jones_q, coherency, phase, envelope, stokes_i)``,
        compiled when the backend supports compilation and the plain reference
        function otherwise.
    """

    def kernel(
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

    if backend.supports_compilation:
        return backend.compile(kernel)
    return kernel
