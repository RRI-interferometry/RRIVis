"""Section 27 row B3 and the Section 13.6 compilation boundary.

Two things are being held here:

1. the boundary itself -- ``ArrayBackend`` exposes ``supports_compilation`` and
   ``compile`` with safe defaults, and exactly **one** kernel in the package is
   ever compiled;
2. the kernel's correctness under compilation -- the compiled form agrees with
   its uncompiled reference within the Section 13.5 tolerance and produces the
   *identical* dtype. A dtype difference is a failure, not a tolerance question
   (Section 15 rule 5).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from radiosim.backends import get_backend
from radiosim.backends.base import ArrayBackend
from radiosim.backends.dask_backend import DaskBackend
from radiosim.backends.numpy_backend import NumPyBackend
from radiosim.core.contraction import baseline_contraction, baseline_contraction_for
from radiosim.core.precision import PrecisionConfig
from tests.support.repo_scan import PYTHON_SUFFIXES, iter_tracked_files

RTOL = 1e-12
ATOL_SCALE = 1e-12


def _inputs(rng: np.random.Generator, n_baselines: int, n_sources: int, dtype):
    real = np.float32 if np.dtype(dtype).itemsize == 8 else np.float64

    def _complex(*shape):
        return (rng.normal(size=shape) + 1j * rng.normal(size=shape)).astype(dtype)

    jones_p = _complex(n_baselines, n_sources, 2, 2)
    jones_q = _complex(n_baselines, n_sources, 2, 2)
    coherency = _complex(n_sources, 2, 2)
    phase = np.exp(-2j * np.pi * rng.normal(size=(n_baselines, n_sources))).astype(
        dtype
    )
    envelope = np.exp(-rng.uniform(0.0, 3.0, size=(n_baselines, n_sources))).astype(
        real
    )
    stokes_i = rng.uniform(0.1, 5.0, size=n_sources).astype(real)
    return jones_p, jones_q, coherency, phase, envelope, stokes_i


def test_base_backend_defaults_are_safe() -> None:
    """The base default must not claim a capability it does not have."""
    assert hasattr(ArrayBackend, "supports_compilation")
    assert hasattr(ArrayBackend, "compile")

    def reference() -> str:
        return "reference"

    for backend in (NumPyBackend(), DaskBackend(mode="cpu")):
        assert backend.supports_compilation is False
        # The identity default, so a caller can apply it unconditionally.
        assert backend.compile(reference) is reference


def test_jax_backend_reports_and_performs_compilation() -> None:
    backend = get_backend("jax", device="cpu")
    assert backend.supports_compilation is True

    compiled = backend.compile(lambda x: x * 2)
    # ``jax.jit`` returns a wrapper, never the original function.
    assert callable(compiled)
    assert np.allclose(
        backend.to_numpy(compiled(backend.asarray([1.0, 2.0]))), [2.0, 4.0]
    )


def test_only_a_compiling_backend_gets_a_compiled_kernel() -> None:
    """The outer scheduler preserves exactly one compiled six-input leaf."""
    numpy_kernel = baseline_contraction_for(NumPyBackend())
    dask_kernel = baseline_contraction_for(DaskBackend(mode="cpu"))
    jax_kernel = baseline_contraction_for(get_backend("jax", device="cpu"))

    # Every returned object is now the plain Python baseline scheduler. The JAX
    # compilation boundary is its private leaf rather than the outer wrapper.
    assert numpy_kernel.__closure__ is not None
    assert dask_kernel.__closure__ is not None
    assert jax_kernel.__closure__ is not None
    assert not hasattr(numpy_kernel, "lower")
    assert not hasattr(jax_kernel, "lower")


@pytest.mark.parametrize("polarized", [True, False])
@pytest.mark.parametrize(
    ("preset", "complex_dtype"),
    [("standard", np.complex128), ("fast", np.complex64)],
)
def test_b3_compiled_kernel_matches_the_uncompiled_reference(
    polarized: bool, preset: str, complex_dtype
) -> None:
    """B3: same values within Section 13.5 tolerance, and the identical dtype."""
    precision = (
        PrecisionConfig.fast() if preset == "fast" else PrecisionConfig.standard()
    )
    backend = get_backend("jax", device="cpu", precision=precision)
    rng = np.random.default_rng(6_08)
    jones_p, jones_q, coherency, phase, envelope, stokes_i = _inputs(
        rng, 5, 7, complex_dtype
    )

    arrays = [backend.asarray(value) for value in (jones_p, jones_q, phase, envelope)]
    device_p, device_q, device_phase, device_envelope = arrays
    device_coherency = backend.asarray(coherency) if polarized else None
    device_stokes = None if polarized else backend.asarray(stokes_i)

    reference = baseline_contraction(
        device_p,
        device_q,
        device_coherency,
        device_phase,
        device_envelope,
        device_stokes,
        backend=backend,
    )
    compiled = backend.compile(
        lambda p, q, c, ph, en, si: baseline_contraction(
            p, q, c, ph, en, si, backend=backend
        )
    )(
        device_p,
        device_q,
        device_coherency,
        device_phase,
        device_envelope,
        device_stokes,
    )

    reference_np = backend.to_numpy(reference)
    compiled_np = backend.to_numpy(compiled)

    # Section 15 rule 5: a dtype difference is a failure, not a tolerance question.
    assert compiled_np.dtype == reference_np.dtype
    assert compiled_np.shape == reference_np.shape == (5, 2, 2)

    scale = max(1.0, float(np.max(np.abs(reference_np))))
    if np.dtype(complex_dtype).itemsize == 8:
        rtol, atol = 1e-5, 1e-5 * scale
    else:
        rtol, atol = RTOL, ATOL_SCALE * scale
    assert np.all(
        np.abs(compiled_np - reference_np) <= atol + rtol * np.abs(reference_np)
    )


def test_the_unpolarized_specialization_requires_its_stokes_argument() -> None:
    backend = NumPyBackend()
    rng = np.random.default_rng(11)
    jones_p, jones_q, _, phase, envelope, _ = _inputs(rng, 2, 3, np.complex128)

    with pytest.raises(ValueError, match="stokes_i"):
        baseline_contraction(
            jones_p, jones_q, None, phase, envelope, None, backend=backend
        )


def test_exactly_one_kernel_is_compiled_in_the_package() -> None:
    """Section 13.6: one compiled kernel, and ``vmap`` nowhere."""
    source_root = Path(__file__).resolve().parents[3] / "src" / "radiosim"
    compile_sites: list[str] = []
    vmap_sites: list[str] = []
    for path in iter_tracked_files(source_root, suffixes=PYTHON_SUFFIXES):
        text = path.read_text(encoding="utf-8")
        relative = str(path.relative_to(source_root.parents[1]))
        if "backend.compile(" in text:
            compile_sites.append(relative)
        if path.name == "jax_backend.py":
            continue
        if ".vmap(" in text:
            vmap_sites.append(relative)

    assert compile_sites == ["src/radiosim/core/contraction.py"]
    assert vmap_sites == []
