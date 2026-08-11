"""Section 27 rows B4-B7: registry truthfulness, precision, and synchronization.

The single property under test is the one ``Tier6HybridRuntimePlan.md``
Section 13.1 calls registry truthfulness: **every selectable backend name
describes what actually executes**, and no provenance value the registry
produces misreports the implementation.
"""

from __future__ import annotations

import subprocess
import sys
import warnings

import numpy as np
import pytest

from radiosim.backends import (
    DaskBackend,
    JAXBackend,
    NumPyBackend,
    get_backend,
    get_backend_info,
    is_jax_available,
)
from radiosim.backends.base import BackendNotAvailableError
from radiosim.core.precision import PrecisionConfig
from radiosim.simulator.rime import RIMESimulator


def test_b4_auto_is_deterministic_numpy() -> None:
    """B4: ``auto`` names NumPy because NumPy is what runs (defect D9).

    Before Tier 6H, ``auto`` returned the NumPy-delegating ``NumbaBackend``,
    so every ``actual_backend`` value it produced said ``numba-cpu`` for a run
    that executed plain NumPy. ``auto`` is now deterministic NumPy and leaves
    optional runtime discovery to explicit operations.
    """
    backend = get_backend("auto")

    assert isinstance(backend, NumPyBackend)
    assert backend.name == "numpy-cpu"
    assert backend.backend_type == "numpy"
    assert backend.device_kind == "cpu"
    assert backend.xp is np


def test_auto_never_selects_the_dask_backend() -> None:
    """It delegates to NumPy, so auto-selecting it would misreport the run."""
    for _ in range(3):
        assert not isinstance(get_backend("auto"), DaskBackend)


def test_b5_registry_truthfulness() -> None:
    """B5: no name promises what is not delivered."""
    # ``numba`` is unknown to the factory.
    with pytest.raises(ValueError, match="removed before v1.0"):
        get_backend("numba")

    # The Dask backend reports a ``dask-*`` name.
    assert get_backend("dask").name.startswith("dask-")

    # No measured accelerator run exists, so no GPU claim is made.
    assert RIMESimulator().supports_gpu is False


def test_the_jax_backend_name_keeps_its_pre_existing_doubled_suffix() -> None:
    """Section 39: ``jax-cpu-cpu`` is truthful, only repetitive.

    ``JAXBackend.name`` is ``f"jax-{device.platform}-{jax.default_backend()}"``
    and both are ``"cpu"`` on a CPU host. The plan's risk register requires this
    exact string to be asserted rather than quietly replaced with a cleaner name
    invented for the occasion; changing the format would be its own decision,
    and Tier 6H does not make it.
    """
    backend = get_backend("jax", device="cpu")

    assert isinstance(backend, JAXBackend)
    assert backend.name == "jax-cpu-cpu"
    assert backend.backend_type == "jax"
    assert backend.device_kind == "cpu"


def test_b6_precision_rejection_is_uniform_across_explicit_backends() -> None:
    """B6: explicit backends reject float128; ``auto`` diverts to NumPy."""
    precise = PrecisionConfig.precise()

    for name in ("dask", "jax"):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with pytest.raises(BackendNotAvailableError, match="requested precision"):
                get_backend(name, precision=precise)
        assert [w for w in caught if "float128" in str(w.message)] == []

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            diverted = get_backend("auto", precision=precise)
        except BackendNotAvailableError:
            # Platforms without a real ``float128`` (every arm64 macOS host)
            # reject rather than downgrade, which is the same rule.
            diverted = None
    if diverted is not None:
        assert isinstance(diverted, NumPyBackend)
    assert [w for w in caught if "float128" in str(w.message)] == []


def test_b7_synchronize_blocks_on_the_given_array() -> None:
    """B7: the argument form orders the caller's own work, not a throwaway."""
    backend = get_backend("jax", device="cpu")

    pending = backend.exp(backend.asarray(np.linspace(0.0, 1.0, 64)))
    ready = backend.synchronize(pending)

    assert ready is not None
    assert np.allclose(backend.to_numpy(ready), np.exp(np.linspace(0.0, 1.0, 64)))
    # The no-argument form is still accepted and still returns nothing.
    assert backend.synchronize() is None

    # Every backend accepts the widened signature (breaking change C17).
    for other in (NumPyBackend(), DaskBackend(mode="cpu")):
        array = other.asarray([1.0, 2.0])
        assert np.array_equal(other.synchronize(array), array)
        assert other.synchronize() is None


def test_backend_info_reports_the_renamed_backend() -> None:
    info = get_backend_info()

    assert "numba" not in info
    assert info["dask"]["backend"] == "dask"
    assert info["dask"]["device"] == "CPU"
    assert info["numpy"]["device"] == "CPU"


def test_jax_is_installed_but_stays_out_of_the_base_import_graph() -> None:
    """The declared CPU JAX must not become an import-time cost for every run.

    Tier 6H made ``jax``/``jaxlib`` a dependency of every environment. Importing
    it eagerly from ``radiosim.backends`` would put roughly a second of XLA
    start-up into the import graph of every caller -- including point-source runs
    on the NumPy backend that never touch it -- so it is detected by module spec
    and imported on first construction, exactly like ``healpy`` and ``pyuvdata``.
    """
    assert is_jax_available() is True

    code = (
        "import sys\n"
        "import radiosim, radiosim.api, radiosim.io, radiosim.backends\n"
        "from radiosim.backends import get_backend, list_backends\n"
        "assert 'jax' not in sys.modules, sorted(sys.modules)\n"
        "get_backend('numpy')\n"
        "get_backend('dask')\n"
        "assert 'jax' not in sys.modules\n"
        # ``list_backends`` is an explicit capability query and may load JAX to
        # enumerate devices; construction is what must stay lazy for everyone
        # else.
        "assert list_backends()['jax'] is True\n"
        "get_backend('jax', device='cpu')\n"
        "assert 'jax' in sys.modules\n"
        "print('ok')\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert completed.returncode == 0, completed.stderr or completed.stdout
    assert "ok" in completed.stdout
