"""Characterize the Tier 6 hybrid-runtime, worker, and backend baseline.

Every test in this module pins behavior that exists on ``main`` **today**, before
any Tier 6 production change.  Each test docstring names the slice that owns the
deliberate flip (``OWNED BY: Tier 6x``); a later slice must update the named test
in the same commit that changes the behavior.  A test with no ``OWNED BY`` line
pins behavior Tier 6 preserves.

Tier 6A evidence record
=======================

Slice 6A is the evidence gate for ``Tier6HybridRuntimePlan.md`` Sections 28 and
41.  Section 33 grants 6A exactly one writable file -- this module -- so the
recorded evidence lives here rather than in the plan document.

Q1 -- is a CPU-only JAX installable on all three locked platforms under the
existing NumPy pin? (blocks 6H)  **Yes, from conda-forge, on all three.**
Measured 2026-07-30 on ``osx-arm64`` (macOS 26.5.2, Apple M1 Max), pixi 0.5x
resolver, channel ``conda-forge``, with ``pixi.toml`` and ``pixi.lock``
untouched: three throwaway pixi workspaces outside the repository declared
``channels = ["conda-forge"]``, ``platforms = ["linux-64", "osx-64",
"osx-arm64"]``, the repository's own pins ``numpy >=1.24,<2.5`` and
``numba >=0.64,<0.67`` (``pixi.toml`` lines 29-32), a pinned Python, and
``jax``/``jaxlib``.  ``pixi lock`` resolved every platform:

===========  ======  ============  ==============================  ==========
Platform     Python  jax           jaxlib (build)                  numpy
===========  ======  ============  ==============================  ==========
linux-64     3.11    0.10.2        0.10.2 cpu_py311hceffaa2_0      2.4.6
osx-64       3.11    0.10.2        0.10.2 cpu_py311h3ac22a7_0      2.4.6
osx-arm64    3.11    0.10.2        0.10.2 cpu_py311h001ef46_0      2.4.6
linux-64     3.12    0.10.2        0.10.2 cpu_py312hc81e8bd_0      2.4.6
osx-64       3.12    0.10.2        0.10.2 cpu_py312h1bb425c_0      2.4.6
osx-arm64    3.12    0.10.2        0.10.2 cpu_py312h8d61f43_0      2.4.6
===========  ======  ============  ==============================  ==========

Every selected ``jaxlib`` build string carries the ``cpu_`` prefix, so no
accelerator variant is involved; ``jax`` itself is the ``noarch`` package
``jax-0.10.2-pyhd8ed1ab_0``.  The conda-forge release index for ``jaxlib``
0.10.2 (https://api.anaconda.org/release/conda-forge/jaxlib/0.10.2, retrieved
2026-07-30) lists CPU builds for ``linux-64``, ``linux-aarch64``, ``osx-64`` and
``osx-arm64`` for cp311/cp312/cp313/cp314, with CUDA variants on the Linux
subdirs only and none for macOS.  So the plan's Section 28 "adopted" position
is achievable exactly as written, with no platform exclusion and no fallback
under Q1.

The **conda-forge channel is load-bearing** and PyPI is not a substitute:

* ``https://pypi.org/pypi/jaxlib/json`` (retrieved 2026-07-30) reports latest
  ``jaxlib`` 0.11.0 with ``requires_python >=3.12`` and wheels only for
  ``manylinux_2_27_x86_64``, ``macosx_11_0_arm64`` and ``win_amd64`` -- there is
  **no macOS x86_64 wheel at all**, and no cp311 wheel, so a PyPI-based feature
  would fail both ``osx-64`` and the ``py311`` environment.
* ``https://pypi.org/pypi/jaxlib/0.10.2/json`` (retrieved 2026-07-30) reports
  ``requires_python >=3.11`` and wheels for ``macosx_11_0_arm64`` and
  ``manylinux_2_27_x86_64``/``aarch64`` -- still no ``macosx x86_64``.

The NumPy pin interacts but does not block.  ``jaxlib``'s own metadata requires
``numpy>=2.0``, so a solve that also allows the lower half of the repository pin
selects an older JAX rather than failing: a fourth throwaway workspace pinning
``numpy >=1.24,<2.0`` resolved ``jax``/``jaxlib`` **0.7.1** (builds
``cpu_py311he1b3896_0`` linux-64, ``cpu_py311hb985efb_0`` osx-64,
``cpu_py311hb8efb21_0`` osx-arm64) against ``numpy 1.26.4`` on all three
platforms.  The whole declared NumPy range is therefore satisfiable with a
CPU-only JAX.

Q1 sub-question 2 -- does ``jax_enable_x64`` yield true float64/complex128 for
the solver dtypes?  **Yes.**  On the resolved ``jax``/``jaxlib`` 0.10.2 CPU
build (Python 3.11.15, numpy 2.4.6, device ``CpuDevice(id=0)``), before
``jax.config.update("jax_enable_x64", True)`` an explicit ``float64``/
``complex128`` request is truncated to ``float32``/``complex64`` with a
``UserWarning``; after it, ``zeros(float64)`` is ``float64``, ``zeros
(complex128)`` is ``complex128``, and ``matmul``, ``exp`` and ``sum`` over
``complex128`` all return ``complex128``.  ``float32``/``complex64`` remain
available, so the ``fast`` preset islands are unaffected.
``backends/jax_backend.py`` enables x64 unconditionally in ``__init__``, so the
solver dtypes are correct without caller action.

Q1 sub-question 3 -- what do the six currently-skipping tests do when JAX is
present?  **All six pass; none reveals a defect.**  Measured by installing
``jax==0.10.2``/``jaxlib==0.10.2`` (plus ``ml_dtypes`` and ``opt_einsum``) into
a throwaway ``--target`` directory and running the five affected modules with
that directory on ``PYTHONPATH`` inside the repository's own ``default``
environment (Python 3.11.13, numpy 2.3.2, scipy 1.17.0, ``jax.devices()`` ==
``[CpuDevice(id=0)]``): ``tests/unit/test_backends/test_jax_backend.py``,
``tests/unit/test_jones/test_backend_jones.py``,
``tests/unit/test_core/test_sky_backend.py``,
``tests/unit/test_core/test_visibility_backend.py`` and
``tests/unit/test_core/test_sky_spectral.py`` collected 55 tests and reported
``55 passed`` with **zero** skips, against ``6 skipped`` in the same modules
without JAX.

Q1 sub-question 4 -- do the resolved JAX-CPU numbers meet the Section 13.5
tolerance on a small workload, before any restructure?  **Yes, and in fact
bit-identically.**  Using the existing solver fixtures from
``tests/unit/test_core/test_visibility_backend.py`` with
``get_backend("jax", device="cpu")`` against ``get_backend("numpy")``:

=============================  ============  ==========  =============
Workload                       dtype         max abs dev  bit-identical
=============================  ============  ==========  =============
point sources, Q and U non-0   complex128    0.0          yes
HEALPix, scalar (I only)       complex128    0.0          yes
HEALPix, polarized             complex128    0.0          yes
=============================  ============  ==========  =============

The Section 13.5 predicate ``|V_jax - V_numpy| <= atol + rtol*|V_numpy|`` with
``rtol = 1e-12`` and ``atol = 1e-12 * max(1, max|V_numpy|)`` holds with margin.
Note this is a *pre-restructure* measurement on tiny workloads where XLA has
little to fuse; Section 13.5's refusal to assert bit-identity across backends
remains the right rule for 6H, and this record must not be read as licence to
tighten B1 to equality.

Q2 -- are the FITS beam handlers and the ``BeamSystem`` safe to share across
solver threads? (blocks 6E)  **No thread-safety failure was observed.**  The
class the plan names, ``BeamFITSHandler``, no longer exists anywhere in
``src/radiosim`` -- the current shapes are ``core/beam/fits.py``
``_LoadedFITSHandler`` (which wraps pyuvdata ``UVBeam.interp``) reached through
``core/beam/runtime.py`` ``BeamSystem.evaluate_jones``.  Probe: one
``shared_fits`` beam (a single ``handler_id`` shared by both antennas, written
by ``tests/fixtures/beamfits.write_scalar_efield_beamfits``) evaluated from a
four-thread ``ThreadPoolExecutor`` over 64 distinct
``(antenna, altitude, azimuth, frequency, time)`` cases, compared case by case
against a serial evaluation of the identical inputs: 0 mismatches, max absolute
deviation 0.0.  A repeat probe with 16 concurrent evaluations of one identical
input was likewise bit-identical.  This is positive evidence from one platform
and one pyuvdata version (3.2.1), not a proof; 6E should keep the per-worker
handler construction option in reserve if it observes any divergence.

Baseline fingerprint scope
==========================

Section 32.1 asks 6A to record ``scientific_sha256`` for *every* shipped
configuration as the reference values Section 27 R1 will check.  Two of the
three shipped configurations are pinned below as executable tests.  The third,
``configs/realistic_foreground_example.yaml``, **cannot be run at this gate**,
for two independent reasons:

1. it needs a 12 MB network download of the Remazeilles/Haslam 408 MHz map from
   ``lambda.gsfc.nasa.gov`` through the astropy download cache, so it can never
   be a hermetic test; and
2. more seriously, its ``realistic_foreground`` recipe fails before any
   visibility is computed with
   ``TypeError: _load_from_vizier_catalog() takes from 1 to 3 positional
   arguments but 4 positional arguments (and 3 keyword-only arguments) were
   given``.  Commit ``7b02bb2`` ("refactor(sky): normalize loader contracts",
   2026-06-25) made ``precision`` keyword-only on
   ``core/sky/loaders/vizier/core.py`` ``_load_from_vizier_catalog`` without
   updating the four positional call sites in
   ``core/sky/loaders/vizier/point_catalogs.py`` (the ``gleam``, ``mals`` and
   ``lotss`` wrappers and the data-driven loader factory that backs ``vlssr``,
   ``tgss``, ``wenss``, ``sumss``, ``nvss``, ``3c`` and ``vlass``).  Every
   VizieR point-catalog loader has therefore been dead since that commit.

That defect is a production bug outside 6A's Section 33 file grant and outside
Tier 6's stated scope; it is recorded here for the acceptance reviewer and needs
its own issue row and remediation slice.  Until it is fixed, R1 cannot cover
``configs/realistic_foreground_example.yaml``.  To keep the diffuse solver path
under fingerprint coverage anyway, this module also pins hermetic raw-cube
digests for the reachable Section 13.4 workloads, including both HEALPix rows.
The hybrid Section 13.4 row is unreachable by construction at this gate -- that
is precisely defect D2 -- so 6F owns adding it.

Reproducibility scope -- R1 is per-environment
==============================================

**Every fingerprint recorded here differs between the two locked Python
environments, and the plan's Section 27 R1 does not say so.**  Each digest is
reproducible to the bit *within* one environment (verified by repeated runs) but
not *across* them.  The cause is isolated, and it is not nondeterminism and not
the solver: the ``default``/py311 environment resolves astropy 7.1.0 while
``py312`` resolves astropy 8.0.1, and the ICRS -> AltAz transform of the same
source at the same instant differs between them --

* astropy 7.1.0: ``alt = 1.5668104524223887``, ``az = 1.8421809886140983``
* astropy 8.0.1: ``alt = 1.5668104524079418``, ``az = 1.8421809682045285``

-- a ~1.4e-11 rad altitude and ~2.0e-8 rad azimuth difference which the
geometric phase amplifies into every visibility.  ``numpy`` also differs (2.3.2
vs 2.4.6), but the coordinate difference alone is sufficient and was measured
directly.

Consequence for later slices: R1 ("post-restructure ``scientific_sha256`` equals
the pinned pre-restructure value") is only meaningful when the comparison runs in
the *same* environment as the pin.  Section 31 runs the gate in both, so 6D must
compare py311 against the py311 pin and py312 against the py312 pin.  Both are
recorded below, keyed by ``_ENVIRONMENT_KEY``; a third environment must be
measured and added rather than have the assertion relaxed.  This also means no
Tier 6 fingerprint may be treated as an environment-independent constant in
documentation.  The environment is now the *only* axis of variation for the
shipped-config ``scientific_sha256`` pins: since the RUN-005 fix
(``fix(result): exclude filesystem transport facts from scientific_sha256``)
those digests no longer depend on where the tree is checked out, so a given pin
is reproducible on any machine running the same locked environment.
"""

from __future__ import annotations

import hashlib
import importlib
import inspect
import re
import sys
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml
from astropy import units as u
from astropy.coordinates import EarthLocation
from astropy.time import Time

from radiosim.api import Simulator
from radiosim.backends import get_backend
from radiosim.backends.base import ArrayBackend
from radiosim.backends.dask_backend import DaskBackend
from radiosim.backends.numpy_backend import NumPyBackend
from radiosim.core.precision import PrecisionConfig
from radiosim.core.sky import (
    HealpixData,
    SkyFormat,
    SkyModel,
    create_from_arrays,
    materialize_healpix_model,
)
from radiosim.core.sky.combine.engine import _combine_models
from radiosim.core.sky.operations.parallel import load_models_parallel
from radiosim.core.time_grid import build_observation_time_grid
from radiosim.core.visibility import calculate_visibility
from radiosim.core.visibility_healpix import calculate_visibility_healpix
from radiosim.io.config import ExecutionConfig, VisibilityConfig
from radiosim.simulator.rime import RIMESimulator
from radiosim.utils import network as network_module
from tests.fixtures.configs import hybrid_config_mapping, valid_config_mapping

REPO_ROOT = Path(__file__).resolve().parents[2]


def _source(relative_path: str) -> str:
    """Return the text of a repository file, for source-truth pins."""
    return (REPO_ROOT / relative_path).read_text(encoding="utf-8")


def _shipped_config_mapping(name: str, tmp_path: Path) -> dict[str, Any]:
    """Load a shipped configuration, redirecting all output into ``tmp_path``."""
    data = yaml.safe_load((REPO_ROOT / "configs" / name).read_text(encoding="utf-8"))
    workflow = dict(data.get("workflow") or {})
    workflow["output_dir"] = str(tmp_path)
    workflow["save_results"] = False
    workflow["save_log"] = False
    workflow["plot_results"] = False
    data["workflow"] = workflow
    return data


def _run_shipped_config(name: str, tmp_path: Path):
    """Run a shipped configuration exactly as the CLI would, minus artifacts."""
    data = _shipped_config_mapping(name, tmp_path)
    simulator = Simulator.from_mapping(data, base_dir=REPO_ROOT / "configs")
    simulator.setup()
    return simulator.run(progress=False)


def _cube_digest(cube: Any) -> str:
    """Stable digest of a visibility cube.

    The recipe is fixed so a later slice can reproduce it: cast to a C-contiguous
    ``complex128`` array, hash the raw little-endian buffer together with the
    shape, and return the hex ``sha256``.  This is the raw-solver analogue of
    ``SimulationResult.scientific_sha256`` for the Section 13.4 workloads, which
    are low-level solver calls rather than whole runs.
    """
    array = np.ascontiguousarray(np.asarray(cube, dtype=np.complex128))
    digest = hashlib.sha256()
    digest.update(repr(array.shape).encode("utf-8"))
    digest.update(array.tobytes())
    return digest.hexdigest()


# Every fingerprint below is a function of the *environment*, not only of the
# source: astropy's ICRS->AltAz transform changed between 7.1.0 (``default``,
# py311) and 8.0.1 (``py312``), so the geometric phase differs in its last bits
# and every digest differs between the two locked environments.  See the
# "Reproducibility scope" note in this module's docstring.
_ENVIRONMENT_KEY = f"py{sys.version_info[0]}{sys.version_info[1]}"
_MEASURED_ENVIRONMENTS = {
    "py311": "python 3.11.13, numpy 2.3.2, astropy 7.1.0, scipy 1.17.0",
    "py312": "python 3.12.13, numpy 2.4.6, astropy 8.0.1, scipy 1.18.0",
}


def _expected_for_environment(table: dict[str, Any], what: str) -> Any:
    """Return this environment's pinned reference, or fail loudly."""
    if _ENVIRONMENT_KEY not in table:
        pytest.fail(
            f"No Tier 6A reference fingerprint recorded for {what} in "
            f"environment {_ENVIRONMENT_KEY}.  6A measured only "
            f"{sorted(_MEASURED_ENVIRONMENTS)}; record the new environment "
            "explicitly rather than relaxing the assertion."
        )
    return table[_ENVIRONMENT_KEY]


class _SetAtCountingBackend(NumPyBackend):
    """A NumPy backend that counts accumulation calls, for the D11 pin.

    Tier 6D flipped the two D11 pins that use this wrapper, so it now counts the
    block assemblies that replaced the per-cell writes as well as the writes
    themselves, and the pins assert that the ``set_at`` count is zero.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.set_at_calls = 0
        self.stack_calls = 0
        self.cube_assemblies: list[tuple[int, ...]] = []

    def set_at(self, arr: Any, index: Any, value: Any) -> Any:
        self.set_at_calls += 1
        return super().set_at(arr, index, value)

    def stack(self, arrays: Any, axis: int = 0) -> Any:
        result = super().stack(arrays, axis=axis)
        self.stack_calls += 1
        shape = tuple(np.asarray(result).shape)
        if len(shape) == 5:
            self.cube_assemblies.append(shape)
        return result


# =========================================================================
# Section 13.4 / R1 workload fixtures (hermetic, no network)
# =========================================================================


def _solver_components(tmp_path: Path, **overrides: Any):
    data = valid_config_mapping(tmp_path, **overrides)
    simulator = Simulator.from_mapping(data, base_dir=tmp_path)
    simulator._ensure_instrument_state()
    simulator._ensure_receptor_set()
    simulator._ensure_beam_system()
    from radiosim.core.instrument_adapters import SolverInstrumentView

    return (
        SolverInstrumentView.from_state(simulator._instrument_state),
        simulator.beam_system,
        simulator.receptors,
    )


_WORKLOAD_FREQS = np.array([100e6, 101e6], dtype=np.float64)
# Matches ``tests.fixtures.configs.valid_config_mapping`` exactly, so the
# low-level Section 13.4 workloads run on the same instrument and time grid the
# fixture would have produced through ``Simulator.setup()``.
WORKLOAD_LOCATION = EarthLocation.from_geodetic(
    21.4283 * u.deg, -30.72152 * u.deg, 1073.0 * u.m
)
WORKLOAD_OBSTIME = Time("2025-01-01T00:00:00")
WORKLOAD_TIME_GRID = build_observation_time_grid(
    start_time=WORKLOAD_OBSTIME.isot,
    duration_seconds=2.0,
    cadence_seconds=1.0,
)
WORKLOAD_SINGLE_TIME_GRID = build_observation_time_grid(
    start_time=WORKLOAD_OBSTIME.isot,
    duration_seconds=1.0,
    cadence_seconds=1.0,
)
_WORKLOAD_LST_RAD = WORKLOAD_OBSTIME.sidereal_time(
    "apparent", longitude=WORKLOAD_LOCATION.lon
).rad


def _workload_point_sources(*, polarized: bool, gaussian: bool) -> dict[str, Any]:
    n = 2
    zeros = np.zeros(n, dtype=np.float64)
    return {
        "ra_rad": np.array(
            [_WORKLOAD_LST_RAD, _WORKLOAD_LST_RAD + 0.01], dtype=np.float64
        ),
        "dec_rad": np.array([-0.536, -0.526], dtype=np.float64),
        "flux": np.array([2.0, 1.0], dtype=np.float64),
        "spectral_index": np.array([-0.7, -0.8], dtype=np.float64),
        "stokes_q": np.array([0.2, 0.0]) if polarized else zeros.copy(),
        "stokes_u": np.array([0.0, 0.1]) if polarized else zeros.copy(),
        "stokes_v": np.array([0.05, 0.0]) if polarized else zeros.copy(),
        "ref_freq": np.full(n, 100e6, dtype=np.float64),
        "rotation_measure": zeros.copy(),
        "spectral_coeffs": None,
        "per_channel_flux": None,
        "per_channel_stokes_q": None,
        "per_channel_stokes_u": None,
        "per_channel_stokes_v": None,
        "channel_frequencies": None,
        "major_arcsec": np.full(n, 120.0) if gaussian else zeros.copy(),
        "minor_arcsec": np.full(n, 60.0) if gaussian else zeros.copy(),
        "pa_deg": np.full(n, 30.0) if gaussian else zeros.copy(),
    }


def _workload_healpix_model(*, polarized: bool) -> SkyModel:
    npix = 12
    maps = np.linspace(1.0, 2.0, npix, dtype=np.float64)
    maps = np.vstack([maps, maps * 1.1])
    return SkyModel(
        healpix=HealpixData(
            maps=maps,
            nside=1,
            frequencies=_WORKLOAD_FREQS,
            coordinate_frame="icrs",
            q_maps=np.full_like(maps, 0.1) if polarized else None,
            u_maps=np.full_like(maps, 0.05) if polarized else None,
            v_maps=np.full_like(maps, 0.02) if polarized else None,
        ),
        model_name="tier6a-workload",
        brightness_conversion="rayleigh-jeans",
        precision=PrecisionConfig.standard(),
    )


def _run_point_workload(
    tmp_path: Path,
    *,
    polarized: bool = True,
    gaussian: bool = False,
    heterogeneous: bool = False,
    single_time: bool = False,
):
    overrides: dict[str, Any] = {}
    if heterogeneous:
        overrides["receptors"] = {
            "default": {"basis": "linear", "feed_rotation_deg": 0.0},
            "overrides": [
                {"antenna": {"kind": "number", "number": 1}, "basis": "circular"}
            ],
            "output_basis": "linear",
        }
    instrument, beam_system, receptors = _solver_components(tmp_path, **overrides)
    return calculate_visibility(
        instrument=instrument,
        beam_system=beam_system,
        source_arrays=_workload_point_sources(polarized=polarized, gaussian=gaussian),
        location=WORKLOAD_LOCATION,
        time_grid=(WORKLOAD_SINGLE_TIME_GRID if single_time else WORKLOAD_TIME_GRID),
        frequencies=_WORKLOAD_FREQS,
        backend=get_backend("numpy"),
        receptors=receptors,
    )


def _run_healpix_workload(tmp_path: Path, *, polarized: bool):
    instrument, beam_system, receptors = _solver_components(tmp_path)
    return calculate_visibility_healpix(
        _workload_healpix_model(polarized=polarized),
        instrument=instrument,
        beam_system=beam_system,
        location=WORKLOAD_LOCATION,
        time_grid=WORKLOAD_TIME_GRID,
        frequencies=_WORKLOAD_FREQS,
        backend=get_backend("numpy"),
        receptors=receptors,
        include_polarization=polarized,
    )


# =========================================================================
# D1-D5 -- the hybrid sky cannot reach the solver
# =========================================================================


def test_sky_representation_admits_the_hybrid_literal() -> None:
    """Records the closure of D1.

    Flipped by Tier 6F from the 6A pin
    ``test_sky_representation_admits_only_two_literals``, which asserted the
    two-literal set and the absence of ``allow_lossy_point_rasterization``.
    The high-level API can now express a hybrid sky, and the point-to-HEALPix
    rasterization it replaces has an explicit opt-in.  No ``OWNED BY`` line
    remains.
    """
    field = VisibilityConfig.model_fields["sky_representation"]
    assert field.annotation is not None
    literals = set(getattr(field.annotation, "__args__", ()))
    assert literals == {"point_sources", "healpix_map", "hybrid"}
    assert VisibilityConfig().sky_representation == "point_sources"
    assert "allow_lossy_point_rasterization" in VisibilityConfig.model_fields
    assert VisibilityConfig().allow_lossy_point_rasterization is False


def test_each_representation_runs_exactly_its_own_components(
    tmp_path, monkeypatch
) -> None:
    """Records the closure of D2.

    Flipped by Tier 6F from the 6A pin ``test_exactly_one_solver_runs_per_run_call``,
    which asserted that ``run()`` dispatches to exactly one solver and that no
    summation was therefore possible.  A ``hybrid`` run now calls both solvers,
    in the fixed Section 8.3 order, and sums their cubes.  No ``OWNED BY`` line
    remains.
    """
    point_module = importlib.import_module("radiosim.core.visibility")
    healpix_module = importlib.import_module("radiosim.core.visibility_healpix")
    calls: list[str] = []

    original_point = point_module.calculate_visibility
    original_healpix = healpix_module.calculate_visibility_healpix

    def record_point(*args: Any, **kwargs: Any):
        calls.append("point")
        return original_point(*args, **kwargs)

    def record_healpix(*args: Any, **kwargs: Any):
        calls.append("healpix")
        return original_healpix(*args, **kwargs)

    monkeypatch.setattr(point_module, "calculate_visibility", record_point)
    monkeypatch.setattr(healpix_module, "calculate_visibility_healpix", record_healpix)

    for representation, expected in (
        ("point_sources", ["point"]),
        ("healpix_map", ["healpix"]),
    ):
        calls.clear()
        data = valid_config_mapping(
            tmp_path,
            visibility={
                "calculation_type": "direct_sum",
                "sky_representation": representation,
                "allow_lossy_point_rasterization": representation == "healpix_map",
            },
        )
        Simulator.from_mapping(data, base_dir=tmp_path).run(progress=False)
        assert calls == expected

    calls.clear()
    hybrid_dir = tmp_path / "hybrid"
    hybrid_dir.mkdir(exist_ok=True)
    Simulator.from_mapping(hybrid_config_mapping(hybrid_dir), base_dir=hybrid_dir).run(
        progress=False
    )
    assert calls == ["point", "healpix"]


def test_hybrid_model_under_point_representation_is_now_rejected(
    tmp_path,
) -> None:
    """Records the closure of D3.

    Flipped by Tier 6F from the 6A pin
    ``test_hybrid_model_under_point_representation_is_silently_discarded``,
    which built a hybrid model whose HEALPix payload was inflated to an absurd
    brightness and asserted that the run was *bit-identical* to the point-only
    run, with no error and no warning.  A surviving HEALPix payload can no
    longer contribute nothing in silence: the request is rejected with the exact
    Section 18.3 message, and ``hybrid`` is the mode that sums both payloads.
    No ``OWNED BY`` line remains.
    """
    from radiosim.core.hybrid import HybridSkyError
    from radiosim.core.sky.combine import pipeline

    data = valid_config_mapping(tmp_path)
    baseline = np.asarray(
        Simulator.from_mapping(data, base_dir=tmp_path).run(progress=False).visibilities
    )

    hybrid_run = Simulator.from_mapping(data, base_dir=tmp_path)
    original_prepare = pipeline.prepare_sky_model
    frequencies = np.asarray(
        hybrid_run._resolved.frequency.channel_frequencies_hz, dtype=np.float64
    )

    def hybridize(*args: Any, **kwargs: Any):
        resolved = original_prepare(*args, **kwargs)
        hybrid_sky = materialize_healpix_model(
            resolved,
            nside=8,
            frequencies=frequencies,
            ref_frequency=float(frequencies[0]),
            clear_other=False,
        )
        inflated = np.full_like(hybrid_sky.healpix.maps, 1.0e6)
        return hybrid_sky.replace(healpix=hybrid_sky.healpix.replace(maps=inflated))

    monkeypatch_target = pipeline
    saved = monkeypatch_target.prepare_sky_model
    monkeypatch_target.prepare_sky_model = hybridize
    try:
        with pytest.raises(HybridSkyError) as excinfo:
            hybrid_run.setup()
    finally:
        monkeypatch_target.prepare_sky_model = saved

    assert str(excinfo.value) == (
        "visibility.sky_representation=point_sources would discard the HEALPix "
        "payload carried by the resolved sky model. Request hybrid to sum both "
        "components, or set "
        "visibility.allow_lossy_point_materialization=true to convert the "
        "HEALPix payload to point sources."
    )
    # The point-only run itself is untouched by the new gate.
    assert baseline.shape == (2, 3, 3, 4)


def test_setup_publishes_the_payloads_the_requested_mode_solves(tmp_path) -> None:
    """Records the closure of D3's setup-side fork.

    Flipped by Tier 6F from the 6A pin
    ``test_setup_keeps_exactly_one_payload_per_representation``.  ``setup`` no
    longer forks to exactly one payload: a ``hybrid`` request publishes both,
    and the fork that remains is the one the mode genuinely implies.  No
    ``OWNED BY`` line remains.
    """
    healpix_data = valid_config_mapping(
        tmp_path,
        visibility={
            "calculation_type": "direct_sum",
            "sky_representation": "healpix_map",
            "allow_lossy_point_rasterization": True,
        },
    )
    healpix_sim = Simulator.from_mapping(healpix_data, base_dir=tmp_path)
    healpix_sim.setup()
    assert healpix_sim._source_arrays is None
    assert healpix_sim._sky_model.healpix is not None

    point_sim = Simulator.from_mapping(
        valid_config_mapping(tmp_path), base_dir=tmp_path
    )
    point_sim.setup()
    assert point_sim._source_arrays is not None
    assert point_sim._sky_model.healpix is None

    hybrid_dir = tmp_path / "hybrid_setup"
    hybrid_dir.mkdir(exist_ok=True)
    hybrid_sim = Simulator.from_mapping(
        hybrid_config_mapping(hybrid_dir), base_dir=hybrid_dir
    )
    hybrid_sim.setup()
    assert hybrid_sim._source_arrays is not None
    assert hybrid_sim._sky_model.healpix is not None
    assert hybrid_sim._sky_model.formats == {
        SkyFormat.POINT_SOURCES,
        SkyFormat.HEALPIX,
    }


def test_point_target_combine_still_drops_a_hybrid_contributor_maps() -> None:
    """Records the closure of D4 at the boundary that reaches a user.

    Flipped by Tier 6F from the 6A pin
    ``test_point_target_combine_silently_drops_a_hybrid_contributor_maps``.
    The low-level combine *primitive* is deliberately unchanged -- Section 10.1
    reuses the existing gate, and Section 33 grants 6F neither ``engine.py`` nor
    ``healpix.py`` -- so a direct ``_combine_models`` call still drops the maps.
    What changed is that no configuration can reach that drop any more: the
    simulator rejects the request first, with the exact Section 18.3 message
    (``tests/unit/test_core/test_hybrid_visibility.py``).  This test now pins
    both halves of that boundary.  No ``OWNED BY`` line remains.
    """
    precision = PrecisionConfig.standard()
    frequencies = np.asarray([100e6, 150e6], dtype=np.float64)
    npix = 12
    point_only = create_from_arrays(
        ra_rad=np.asarray([0.1, 0.2]),
        dec_rad=np.asarray([0.0, 0.0]),
        flux=np.asarray([1.0, 2.0]),
        reference_frequency=150e6,
        precision=precision,
    )
    hybrid = point_only.replace(
        healpix=HealpixData(
            maps=np.full((frequencies.size, npix), 9.0, dtype=np.float64),
            nside=1,
            frequencies=frequencies,
            coordinate_frame="icrs",
        )
    )
    assert hybrid.formats == {SkyFormat.POINT_SOURCES, SkyFormat.HEALPIX}

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        combined = _combine_models(
            [hybrid, point_only],
            representation=SkyFormat.POINT_SOURCES,
            precision=precision,
            mixed_model_policy="allow",
        )

    assert combined.formats == {SkyFormat.POINT_SOURCES}
    assert combined.healpix is None
    assert combined.n_point_sources == 4
    assert [str(w.message) for w in caught] == []

    # The same shape is now unreachable from a configuration: the Section 20.1
    # step-9 gate rejects it before any solver runs.
    from radiosim.core.hybrid import HybridSkyError, check_representation_compatibility

    with pytest.raises(HybridSkyError, match="would discard the HEALPix payload"):
        check_representation_compatibility(
            sky_representation="point_sources",
            contributed_models=[hybrid, point_only],
            resolved_model=combined,
            allow_lossy_point_rasterization=False,
        )


def test_healpix_target_combine_rasterizes_only_behind_the_new_opt_in() -> None:
    """Records the closure of D5's rasterization half.

    Flipped by Tier 6F from the 6A pin
    ``test_healpix_target_combine_rasterizes_a_point_contributor``.  The
    capability is unchanged and still folds point flux into the map cube -- the
    combine primitive is not in 6F's Section 33 grant -- but a configuration can
    only reach it by setting ``visibility.allow_lossy_point_rasterization``.
    No ``OWNED BY`` line remains.
    """
    precision = PrecisionConfig.standard()
    frequencies = np.asarray([100e6, 150e6], dtype=np.float64)
    npix = 12
    point_only = create_from_arrays(
        ra_rad=np.asarray([0.1]),
        dec_rad=np.asarray([0.0]),
        flux=np.asarray([5.0]),
        reference_frequency=150e6,
        precision=precision,
    )
    healpix_only = SkyModel(
        healpix=HealpixData(
            maps=np.zeros((frequencies.size, npix), dtype=np.float64),
            nside=1,
            frequencies=frequencies,
            coordinate_frame="icrs",
        ),
        precision=precision,
    )

    combined = _combine_models(
        [point_only, healpix_only],
        representation=SkyFormat.HEALPIX,
        frequencies=frequencies,
        precision=precision,
        mixed_model_policy="allow",
    )

    assert combined.formats == {SkyFormat.HEALPIX}
    assert combined.point is None
    assert float(np.max(np.abs(combined.healpix.maps))) > 0.0

    from radiosim.core.hybrid import HybridSkyError, check_representation_compatibility

    with pytest.raises(HybridSkyError, match="would rasterize 1 point source"):
        check_representation_compatibility(
            sky_representation="healpix_map",
            contributed_models=[point_only, healpix_only],
            resolved_model=combined,
            allow_lossy_point_rasterization=False,
        )
    check_representation_compatibility(
        sky_representation="healpix_map",
        contributed_models=[point_only, healpix_only],
        resolved_model=combined,
        allow_lossy_point_rasterization=True,
    )


def test_point_target_combine_rejects_a_healpix_only_contributor() -> None:
    """Records D5's hard-error half, which Tier 6F preserves and extends.

    Flipped by Tier 6F only in its message assertion: the existing rejection is
    unchanged in condition and in effect, but now also names ``hybrid`` as the
    lossless alternative (Section 8.2 rule 2).  No ``OWNED BY`` line remains.
    """
    precision = PrecisionConfig.standard()
    frequencies = np.asarray([100e6, 150e6], dtype=np.float64)
    point_only = create_from_arrays(
        ra_rad=np.asarray([0.1]),
        dec_rad=np.asarray([0.0]),
        flux=np.asarray([5.0]),
        reference_frequency=150e6,
        precision=precision,
    )
    healpix_only = SkyModel(
        healpix=HealpixData(
            maps=np.full((frequencies.size, 12), 3.0, dtype=np.float64),
            nside=1,
            frequencies=frequencies,
            coordinate_frame="icrs",
        ),
        precision=precision,
    )

    with pytest.raises(ValueError) as excinfo:
        _combine_models(
            [point_only, healpix_only],
            representation=SkyFormat.POINT_SOURCES,
            precision=precision,
            mixed_model_policy="allow",
        )
    assert str(excinfo.value) == (
        "Point-source combination requires converting a HEALPix-only model to "
        "point sources, which is lossy. Request "
        "visibility.sky_representation=hybrid to sum a point component and a "
        "HEALPix component without converting either, or re-run with "
        "allow_lossy_point_materialization=True to opt in."
    )


def test_memory_estimate_counts_every_solved_component(tmp_path) -> None:
    """Records the closure of D19.

    Flipped by Tier 6F from the 6A pin
    ``test_memory_estimate_counts_only_point_sources``, which asserted that
    ``get_memory_estimate`` read ``self._source_arrays`` alone and therefore
    reported zero sky elements for a HEALPix run.  The estimate now sums every
    component the requested mode solves (Section 17).  No ``OWNED BY`` line
    remains.
    """
    data = valid_config_mapping(
        tmp_path,
        visibility={
            "calculation_type": "direct_sum",
            "sky_representation": "healpix_map",
            "allow_lossy_point_rasterization": True,
        },
    )
    simulator = Simulator.from_mapping(data, base_dir=tmp_path)
    simulator.setup()
    assert simulator._source_arrays is None
    assert simulator._sky_model.healpix is not None
    assert simulator._solved_sky_element_count() == (
        simulator._sky_model.n_healpix_pixels
    )
    estimate = simulator.get_memory_estimate()
    assert isinstance(estimate, dict)

    hybrid_dir = tmp_path / "hybrid_memory"
    hybrid_dir.mkdir(exist_ok=True)
    hybrid_sim = Simulator.from_mapping(
        hybrid_config_mapping(hybrid_dir), base_dir=hybrid_dir
    )
    hybrid_sim.setup()
    assert hybrid_sim._solved_sky_element_count() == (
        hybrid_sim._sky_model.n_point_sources + hybrid_sim._sky_model.n_healpix_pixels
    )


# =========================================================================
# D6, D7 -- worker policy is hard-coded and unexpressible
# =========================================================================


def test_sky_loading_consumes_the_resolved_worker_count() -> None:
    """Records the closure of D6's behavior half.

    Flipped by Tier 6C from the 6A pin ``test_sky_loading_hard_codes_eight_workers``,
    which asserted the literal ``max_workers=8,`` in ``api/simulator.py`` and the
    ``max_workers: int = 8`` default on ``load_models_parallel``.  Tier 6B supplied
    the typed policy; Tier 6C removed both numbers (plan Section 11.2), so the
    driver has no default a caller can silently inherit and the only call site
    passes the resolved value.
    """
    simulator_source = _source("src/radiosim/api/simulator.py")
    assert "max_workers=8," not in simulator_source
    assert "max_workers=sky_loading.max_workers," in simulator_source

    signature = inspect.signature(load_models_parallel)
    assert signature.parameters["max_workers"].default is inspect.Parameter.empty


def test_execution_config_expresses_worker_policy_in_two_typed_blocks() -> None:
    """Records the Tier 6B schema that closed the D6/D7 expressibility half.

    Flipped by Tier 6B from the 6A pin
    ``test_execution_config_has_no_worker_or_concurrency_field``, which asserted
    that ``ExecutionConfig`` carried exactly ``{backend, precision, simulator,
    offline}`` and therefore had nowhere to express a worker count.  The typed
    blocks now exist and are resolved; they are not yet *consumed* -- see
    ``tests/unit/test_simulator/test_worker_policy.py`` for the interim
    boundary owned by 6C (loader) and 6E (solver).
    """
    assert set(ExecutionConfig.model_fields) == {
        "backend",
        "precision",
        "simulator",
        "offline",
        "sky_loading",
        "solver",
    }
    execution = ExecutionConfig()
    assert execution.sky_loading.max_workers is None
    assert execution.sky_loading.executor == "auto"
    assert execution.solver.workers == 1
    assert execution.solver.executor == "thread"


def test_execution_config_backend_literal_now_offers_dask() -> None:
    """Flipped by Tier 6H, closing the config half of D8/D9.

    The 6A pin recorded the defect: ``execution.backend`` offered a ``numba``
    literal naming a backend that never compiled a kernel.  Tier 6H replaced it
    with ``dask``, the name of what the class actually is, and gave the removed
    literal the verbatim Section 18.3 rejection asserted here and in
    ``tests/unit/test_io/test_config.py`` (Section 27 row E4).  The literal
    changes in the same commit as the rename so no state ever exposes a config
    name the registry cannot construct (plan Sections 32.2, 32.8, 33).
    """
    backend_literals = set(
        getattr(ExecutionConfig.model_fields["backend"].annotation, "__args__", ())
    )
    assert backend_literals == {"auto", "numpy", "jax", "dask"}
    with pytest.raises(Exception, match="removed before v1.0"):
        ExecutionConfig(backend="numba")


def test_run_no_longer_advertises_n_workers(tmp_path) -> None:
    """Flipped by Tier 6E, closing D7.

    The 6A pin recorded the defect: ``run()`` advertised and documented an
    ``n_workers`` parameter that could never be used, raising
    ``NotImplementedError`` for every value.  Tier 6E deleted the parameter
    outright (plan Sections 12.1-12.3), so the rejection is now Python's own
    ``TypeError`` naming the removed keyword, ``progress`` is keyword-only, and
    solver concurrency is expressed once, in ``execution.solver.workers``.
    """
    signature = inspect.signature(Simulator.run)
    assert list(signature.parameters) == ["self", "progress"]
    assert signature.parameters["progress"].kind is inspect.Parameter.KEYWORD_ONLY
    docstring = Simulator.run.__doc__ or ""
    assert "Number of parallel workers (default: auto)" not in docstring
    assert "execution.solver.workers" in docstring

    simulator = Simulator.from_mapping(
        valid_config_mapping(tmp_path), base_dir=tmp_path
    )
    with pytest.raises(TypeError, match="n_workers"):
        simulator.run(n_workers=1)  # type: ignore[call-arg]


def test_no_worker_value_is_recorded_in_provenance(tmp_path) -> None:
    """Pins D6: no resolved worker count reaches the bounded result snapshot.

    Formerly ``OWNED BY: Tier 6E``; closed by 6E, see the disposition below.

    Scope note added by Tier 6B: the resolved worker policy now *does* reach
    ``SimulationResult.resolved_config`` and therefore ``provenance_sha256``,
    the HDF5 ``resolved_config_json`` and the summary JSON
    (``tests/unit/test_simulator/test_worker_policy.py``).  What this pin still
    records is the narrower fact 6C and 6E own: ``to_summary_snapshot()`` is a
    bounded metadata view that embeds no resolved configuration, so no
    *executed* worker count -- no loader execution record, no per-run solver
    thread count -- is reported there yet.

    Disposition of the 6C half (Tier 6C implementation): 6C surfaced the executed
    loader policy in ``SimulationResult.history`` and in the summary-JSON
    document's ``execution`` block (plan Section 19), neither of which is
    ``to_summary_snapshot()``.  ``core/result.py`` is not in 6C's Section 33
    grant, and the bounded snapshot is deliberately left free of runtime worker
    values, so the assertions below are unchanged and the pin now carries only
    the 6E half.

    Disposition of the 6E half (Tier 6E implementation): **closed with the
    assertions unchanged, deliberately.**  6E made ``execution.solver.workers``
    effective, and the count the solver executes is *exactly* the resolved,
    already-clamped ``ResolvedSolverExecutionConfig.workers``
    (``api/simulator.py`` passes ``self._resolved.execution.solver`` straight to
    both solvers; the partition applies no second clamp of its own).  There is
    therefore no separate *executed* solver worker value that could diverge from
    the resolved one, and the resolved one has been in ``resolved_config``,
    ``provenance_sha256``, the HDF5 ``resolved_config_json`` and the summary
    JSON since 6B.  ``core/result.py`` is not in 6E's Section 33 grant either,
    and adding a runtime worker count to the bounded snapshot would be a
    scope-free change with nothing new to report.  This pin now records a
    standing invariant rather than a defect: the bounded snapshot stays free of
    worker values.  No ``OWNED BY`` line remains.
    """
    result = Simulator.from_mapping(
        valid_config_mapping(tmp_path), base_dir=tmp_path
    ).run(progress=False)
    snapshot = result.to_summary_snapshot()
    flattened = repr(snapshot)
    assert "max_workers" not in flattened
    assert "n_workers" not in flattened
    assert "workers" not in flattened


# =========================================================================
# D8-D15 -- backend truthfulness, accumulation, and missing harness
# =========================================================================


def test_get_backend_auto_returns_the_numpy_backend_on_a_cpu_only_host() -> None:
    """Flipped by Tier 6H, closing D9 (Section 27 row B4).

    The 6A pin recorded the defect: ``auto`` returned a ``NumbaBackend`` whose
    ``xp`` was plain ``numpy``, so every ``actual_backend`` provenance value it
    produced said ``numba-cpu`` for a run that executed NumPy.  The corrected
    precedence is JAX **only** when the installed runtime exposes a non-CPU
    device, otherwise NumPy; the Dask backend is never auto-selected, because
    it too would misreport a NumPy run (plan Section 14.1).

    The declared JAX is CPU-only by design, so on every environment this
    repository locks the answer is the NumPy backend.
    """
    from radiosim.backends import _has_non_cpu_jax_device

    assert _has_non_cpu_jax_device() is False
    backend = get_backend("auto")
    assert isinstance(backend, NumPyBackend)
    assert backend.name == "numpy-cpu"
    assert backend.xp is np


def test_no_numba_kernel_decorator_exists_in_the_package() -> None:
    """Flipped by Tier 6H, closing D8's code half.

    The 6A pin recorded the defect: the backend named ``numba`` imported
    ``jit`` and ``prange``, never called either, and carried a ``jit_compile``
    helper with no caller.  Tier 6H deleted the import, the helper, and the
    name (plan Section 14.1); Section 14.2 records why no real Numba kernel was
    written instead.

    The decorator sweep is unchanged and still asserts zero matches: Tier 6's
    one compiled kernel is compiled by **JAX**, through ``ArrayBackend.compile``
    at a call site, not by a Numba decorator (plan Section 13.6).
    """
    pattern = re.compile(r"@(njit|jit|vectorize|guvectorize|cuda\.jit)\b")
    offenders = [
        str(path.relative_to(REPO_ROOT))
        for path in sorted((REPO_ROOT / "src" / "radiosim").rglob("*.py"))
        if pattern.search(path.read_text(encoding="utf-8"))
    ]
    assert offenders == []

    assert not (
        REPO_ROOT / "src" / "radiosim" / "backends" / "numba_backend.py"
    ).exists()
    dask_source = _source("src/radiosim/backends/dask_backend.py")
    assert "import numba" not in dask_source
    assert "prange(" not in dask_source
    assert "def jit_compile" not in dask_source


def test_dask_backend_docstring_makes_no_compilation_claim() -> None:
    """Flipped by Tier 6H, closing D8's documentation half.

    The 6A pin recorded the defect: the class docstring advertised
    ``"'cpu': Local CPU with JIT and parallel loops"`` for a class that compiled
    nothing.  The renamed class states what it is and, per the Section 39 risk
    row, states explicitly that no compilation ever occurred and none is added.
    """
    docstring = DaskBackend.__doc__ or ""
    assert "JIT" not in docstring
    assert "parallel loops" not in docstring
    assert "It compiles nothing." in docstring
    module_doc = sys.modules[DaskBackend.__module__].__doc__ or ""
    assert "it never" in module_doc and "compiled a single kernel" in module_doc
    assert DaskBackend(mode="cpu").name == "dask-cpu"
    assert DaskBackend(mode="cpu").xp is np


def test_rime_simulator_no_longer_claims_gpu_support() -> None:
    """Flipped by Tier 6H, closing D10 (Section 27 row B5).

    The 6A pin recorded the defect: ``supports_gpu`` returned ``True``
    unconditionally, on the strength of a JAX backend existing rather than of
    any measured accelerator run.  Tier 6 produces no such run and therefore
    makes no such claim (plan Sections 4, 14.1).
    """
    simulator = RIMESimulator()
    assert simulator.supports_gpu is False
    source = inspect.getsource(type(simulator).supports_gpu.fget)
    assert "return False" in source


def test_rime_simulator_docstring_states_the_canonical_chain_order() -> None:
    """Flipped by Tier 6H, closing D20.

    The 6A pin recorded the defect: the class docstring still advertised the
    pre-Tier-5 order ``J = B @ G @ D @ P @ E @ T @ Z @ K``, which omits ``C``
    and ``H`` entirely and reverses the composition sense.  The canonical order
    is ``Tier5ReceptorFeedPlan.md`` Section 19.1's, and it is what
    ``_build_jones_chain`` actually builds.
    """
    docstring = RIMESimulator.__doc__ or ""
    assert "J = B @ G @ D @ P @ E @ T @ Z @ K" not in docstring
    assert "J = H @ G @ B @ D @ P @ C @ E @ T @ Z" in docstring


def test_point_solver_accumulates_one_set_at_per_time_baseline_frequency(
    tmp_path,
) -> None:
    """Pins D11 for the point solver.

    OWNED BY: Tier 6D.  FLIPPED BY: Tier 6D -- the per-``(t, b, f)`` ``set_at``
    accumulation is gone.  The solver now assembles one ``(B, 2, 2)`` block per
    ``(time, frequency)``, one ``(B, F, 2, 2)`` block per time, and exactly one
    ``(T, B, F, 2, 2)`` cube per call (Section 13.3, test R2), so the call count
    drops from ``T*B*F`` functional whole-cube copies to ``T*F + T + 1``
    assemblies and zero ``set_at`` calls.  The shape itself is asserted in
    ``tests/unit/test_core/test_visibility_accumulation.py``; this pin only
    records that the old shape is truly gone.
    """
    instrument, beam_system, receptors = _solver_components(tmp_path)
    backend = _SetAtCountingBackend()
    cube = calculate_visibility(
        instrument=instrument,
        beam_system=beam_system,
        source_arrays=_workload_point_sources(polarized=True, gaussian=False),
        location=WORKLOAD_LOCATION,
        time_grid=WORKLOAD_TIME_GRID,
        frequencies=_WORKLOAD_FREQS,
        backend=backend,
        receptors=receptors,
    )
    n_times, n_baselines, n_freqs = cube.shape[:3]
    assert backend.set_at_calls == 0
    assert backend.stack_calls == n_times * n_freqs + n_times + 1
    assert backend.cube_assemblies == [tuple(np.asarray(cube).shape)]


def test_healpix_solver_accumulates_one_set_at_per_time_baseline_frequency(
    tmp_path,
) -> None:
    """Pins D11 for the HEALPix solver.

    OWNED BY: Tier 6D.  FLIPPED BY: Tier 6D -- same restructure as the point
    solver above, in both the scalar and the polarized HEALPix path.
    """
    instrument, beam_system, receptors = _solver_components(tmp_path)
    backend = _SetAtCountingBackend()
    cube = calculate_visibility_healpix(
        _workload_healpix_model(polarized=False),
        instrument=instrument,
        beam_system=beam_system,
        location=WORKLOAD_LOCATION,
        time_grid=WORKLOAD_TIME_GRID,
        frequencies=_WORKLOAD_FREQS,
        backend=backend,
        receptors=receptors,
        include_polarization=False,
    )
    n_times, n_baselines, n_freqs = cube.shape[:3]
    assert backend.set_at_calls == 0
    assert backend.stack_calls == n_times * n_freqs + n_times + 1
    assert backend.cube_assemblies == [tuple(np.asarray(cube).shape)]


def test_healpix_solver_rebuilds_the_constant_receptor_transforms_per_time() -> None:
    """Pins D12: ``H_p @ C_p`` is recomputed inside the time loop.

    OWNED BY: Tier 6D.  FLIPPED BY: Tier 6D -- the constant ``H_p @ C_p``
    product, together with the selected-antenna tuple it is keyed by, is now
    built once above the time loop instead of once per time sample.  The
    frequency loop no longer enumerates: the restructure removed the last use of
    ``freq_idx``, which only existed to index the per-cell output write.

    Anchor updated by Tier 6E: the per-time body became the ``_time_block``
    closure that solver workers call over contiguous time ranges (plan Section
    11.3), so the literal ``for time_idx in range(n_times):`` statement no
    longer exists.  The property being pinned is unchanged -- the constant
    transform is built once, above everything per-time -- and is now anchored on
    the closure that replaced the statement.
    """
    source = _source("src/radiosim/core/visibility_healpix.py")
    time_loop = source.index("def _time_block(time_idx: int")
    transforms = source.index("receptor_transforms = _receptor_transforms(")
    frequency_loop = source.index("for freq in frequencies:")
    assert transforms < time_loop < frequency_loop
    assert source.count("def _time_block(") == 1
    assert source.count("receptor_transforms = _receptor_transforms(") == 1
    assert "for freq_idx, freq in enumerate(frequencies):" not in source


def test_backend_surface_exposes_the_compilation_boundary() -> None:
    """Flipped by Tier 6H, closing D14 (Section 27 row B3).

    The 6A pin recorded the defect: ``jit``/``vmap``/``jit_compile`` were
    backend-private, so no backend-agnostic caller could opt into compilation,
    and nothing in the package called any of them.  ``ArrayBackend`` now carries
    ``supports_compilation`` (default ``False``) and ``compile`` (default
    identity), the two members plan Section 13.6 specifies, so the solvers can
    request compilation without importing JAX.
    """
    for attribute in ("compile", "supports_compilation"):
        assert hasattr(ArrayBackend, attribute), attribute
    assert ArrayBackend.supports_compilation.fget(None) is False  # type: ignore[arg-type]

    marker = object()

    def _reference() -> object:
        return marker

    numpy_backend = NumPyBackend()
    dask_backend = DaskBackend(mode="cpu")
    for backend in (numpy_backend, dask_backend):
        assert backend.supports_compilation is False
        assert backend.compile(_reference) is _reference

    # The removed Numba helper answers with its replacement rather than a bare
    # attribute miss.
    with pytest.raises(AttributeError, match="removed before v1.0"):
        dask_backend.jit_compile  # noqa: B018

    jax_backend = get_backend("jax", device="cpu")
    assert jax_backend.supports_compilation is True


def test_nothing_in_the_package_calls_jit_vmap_or_jit_compile() -> None:
    """Pins D14's second half: the compilation helpers have no callers.

    OWNED BY: Tier 6H, which wires ``ArrayBackend.compile`` to exactly one
    kernel; the call site arrives with that kernel, not with the rename.
    """
    callers: list[str] = []
    pattern = re.compile(r"\.(jit|vmap|jit_compile)\s*\(")
    for path in sorted((REPO_ROOT / "src" / "radiosim").rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        for match in pattern.finditer(text):
            line = text[: match.start()].count("\n") + 1
            if path.name in {"jax_backend.py", "dask_backend.py"}:
                continue
            callers.append(f"{path.relative_to(REPO_ROOT)}:{line}")
    assert callers == []


def test_jax_synchronize_blocks_on_the_callers_array() -> None:
    """Flipped by Tier 6H, closing D13 (Section 27 row B7).

    The 6A pin recorded the defect by source text, because JAX was not
    installable then: ``synchronize()`` blocked on
    ``jax.block_until_ready(jnp.array(0))``, a freshly constructed throwaway
    constant that completes immediately and orders none of the caller's work,
    which made every JAX timing number meaningless.  JAX is now a declared
    dependency, so this is asserted by execution rather than by grep.
    """
    import inspect as _inspect

    backend = get_backend("jax", device="cpu")
    signature = _inspect.signature(type(backend).synchronize)
    assert list(signature.parameters) == ["self", "arr"]
    assert signature.parameters["arr"].default is None

    pending = backend.exp(backend.asarray([1.0, 2.0, 3.0]))
    ready = backend.synchronize(pending)
    assert ready is not None
    assert np.allclose(backend.to_numpy(ready), np.exp([1.0, 2.0, 3.0]))
    assert backend.synchronize() is None


def test_jax_is_a_cpu_only_dependency_of_every_pixi_environment() -> None:
    """Flipped by Tier 6H, closing D16.

    The 6A pin recorded the defect: ``jax`` appeared nowhere in ``pixi.toml``,
    so the NumPy/JAX parity evidence ``Fix.md`` §15 mandates could not be
    produced and six tests skipped instead.  Tier 6H added the ``jax-cpu``
    feature with the exact versions the Q1 record above resolved
    (``jax``/``jaxlib`` 0.10.2, ``cpu_*`` builds) and carried it into **both**
    declared environments rather than into a separate one, because plan
    Section 31 requires the six skips to disappear from the two gate
    environments' own counts -- a third environment would have left them
    skipping there.  See the Section 33 correction recorded in
    ``Tier6HybridRuntimePlan.md``.

    The ``cpu*`` build constraint is asserted because conda-forge also ships
    CUDA ``jaxlib`` variants on the Linux subdirs, and Tier 6 makes no
    accelerator claim (plan Sections 4, 14.1).
    """
    pixi_toml = _source("pixi.toml")
    assert "[feature.jax-cpu.dependencies]" in pixi_toml
    assert 'jax = ">=0.10.2,<0.11"' in pixi_toml
    assert 'build = "cpu*"' in pixi_toml
    assert 'default = ["py311", "jax-cpu"]' in pixi_toml
    assert 'py312 = ["py312", "jax-cpu"]' in pixi_toml
    assert 'numpy = ">=1.24,<2.5"' in pixi_toml
    assert "[feature.py311.dependencies]" in pixi_toml
    assert "[feature.py312.dependencies]" in pixi_toml

    # The dependency is real, not just declared: the six formerly-skipping
    # tests can only run because this import succeeds in the gate environment.
    import jax

    assert jax.devices()[0].platform == "cpu"


def test_there_is_no_benchmark_harness_task_or_performance_test() -> None:
    """Pins D15.

    OWNED BY: Tier 6I, which adds ``src/radiosim/benchmarks/``,
    ``tests/performance/test_backend_benchmarks.py`` and the ``bench`` task.
    """
    performance = sorted(
        p.name for p in (REPO_ROOT / "tests" / "performance").glob("*.py")
    )
    integration = sorted(
        p.name for p in (REPO_ROOT / "tests" / "integration").glob("*.py")
    )
    assert performance == ["__init__.py"]
    # ``tests/integration/test_hybrid_end_to_end.py`` is Tier 6F's own Section
    # 33 grant (Section 25.4 lists it as a new test file), so 6F narrowed this
    # assertion from "the directory is empty" to "the directory holds nothing
    # that belongs to Tier 6I".  The performance directory, the benchmarks
    # package, and the ``bench`` task are still pinned absent, and those are
    # what D15 is about.
    assert integration == ["__init__.py", "test_hybrid_end_to_end.py"]
    assert not (REPO_ROOT / "src" / "radiosim" / "benchmarks").exists()
    assert "bench" not in _source("pixi.toml")


# =========================================================================
# D17, D18 -- offline policy and executor degradation
# =========================================================================


def test_forced_offline_policy_short_circuits_the_socket_probe(
    monkeypatch,
) -> None:
    """Records the closure of D17's detection half.

    Flipped by Tier 6C from the 6A pin
    ``test_forced_offline_status_does_not_populate_the_module_cache``, which
    asserted that a forced-offline ``NetworkStatus`` left the module cache empty
    and that ``is_online()`` then probed a live socket anyway.  Tier 6C added
    ``set_offline_policy`` (plan Section 16.1); the installed policy -- not the
    TTL cache -- is now the authority, so no probe happens at all.
    """
    monkeypatch.setattr(network_module, "_cached_status", None)
    status = network_module.get_network_status(offline=True)
    assert status.forced_offline is True
    assert status.internet is False
    assert network_module._cached_status is None

    probed: list[tuple[str, int]] = []

    def fake_socket(host: str, port: int, timeout: float) -> bool:
        probed.append((host, port))
        return True

    monkeypatch.setattr(network_module, "_check_socket", fake_socket)
    monkeypatch.setattr(network_module, "_offline_policy", True)
    assert network_module.is_online() is False
    assert probed == []


def test_require_service_consults_the_installed_offline_policy(
    monkeypatch,
) -> None:
    """Records the closure of D17's enforcement half.

    Flipped by Tier 6C from the 6A pin
    ``test_require_service_consults_is_online_not_a_resolved_policy``.  The gate
    still reads ``is_online()`` -- that is deliberate, because Section 16.1 makes
    ``is_online()`` itself consult the policy first -- but the policy now reaches
    it, so a forced-offline run fails a network-requiring loader without a probe.
    """
    source = inspect.getsource(network_module.require_service)
    assert "if not is_online():" in source
    assert "offline" not in inspect.signature(network_module.require_service).parameters

    probed: list[tuple[str, int]] = []
    monkeypatch.setattr(network_module, "_cached_status", None)
    monkeypatch.setattr(
        network_module,
        "_check_socket",
        lambda host, port, timeout: probed.append((host, port)) or True,
    )
    monkeypatch.setattr(network_module, "_offline_policy", True)
    with pytest.raises(ConnectionError, match="No internet connection"):
        network_module.require_service("vizier", "download catalog 'gleam'")
    assert probed == []


def test_an_explicit_process_request_is_rejected_and_auto_degradation_recorded(
    caplog, monkeypatch
) -> None:
    """Records the closure of D18.

    Flipped by Tier 6C from the 6A pin
    ``test_process_executor_degrades_to_threads_with_only_a_log_warning``, which
    asserted that a failed pickle probe silently degraded an *explicit*
    ``executor="process"`` request to threads with only a ``logger.warning`` and
    no returned record.  Section 11.2 now rejects the explicit request outright
    and records an ``auto`` degradation in ``LoaderExecutionRecord``.
    """
    parallel_module = importlib.import_module("radiosim.core.sky.operations.parallel")
    monkeypatch.setattr(
        parallel_module,
        "_pickle_probe",
        lambda *args, **kwargs: ("test_sources", "cannot pickle 'function' object"),
    )

    loaders = [
        ("test_sources", {"num_sources": 1, "distribution": "uniform", "seed": 1})
    ]
    with pytest.raises(parallel_module.WorkerPolicyError, match="requested explicitly"):
        load_models_parallel(
            loaders,
            2,
            precision=PrecisionConfig.standard(),
            strict=True,
            executor="process",
        )

    with caplog.at_level("WARNING", logger=parallel_module.__name__):
        models, record = load_models_parallel(
            loaders,
            2,
            precision=PrecisionConfig.standard(),
            strict=True,
            executor="auto",
        )

    assert len(models) == 1
    assert record.requested_executor == "auto"
    assert record.actual_executor == "thread"
    assert record.degraded_reason is not None
    assert any(
        "Falling back to thread pool" in message.message for message in caplog.records
    )
    assert "LoaderExecutionRecord" in _source(
        "src/radiosim/core/sky/operations/parallel.py"
    )


def test_recommend_executor_is_registry_driven() -> None:
    """Records the executor-selection rule Tier 6C must preserve."""
    from radiosim.core.sky.operations.parallel import recommend_executor_for_loaders

    assert recommend_executor_for_loaders([("test_sources", {})]) == "process"
    assert recommend_executor_for_loaders([("gleam", {})]) == "thread"


# =========================================================================
# Baseline fingerprints -- the reference values Section 27 R1 will check
# =========================================================================


# Re-pinned for the RUN-005 fix ("fix(result): exclude filesystem transport
# facts from scientific_sha256"), which stopped the scientific digest hashing
# the absolute antenna-layout path carried by the instrument snapshot and the
# filesystem-transport keys of the beam snapshot.  The digests below are
# therefore *checkout-independent*: the same commit checked out at any path
# reproduces them.  They remain per-environment, because the astropy version
# difference described in this module's docstring still moves the last bits of
# every visibility.  The pre-fix, checkout-local values were
# ``302deb27...`` / ``161fc98c...`` and ``b3c1a93e...`` / ``e670c35f...``.
#
# Re-pinned again for Tier 6F under the declared breaking change ``C11``
# (Section 36): ``SolverResultProvenance`` gained ``components`` and
# ``component_element_counts``, both deterministic, both deliberately inside
# ``scientific_sha256`` so a hybrid result can never collide with a
# single-component result over the same instrument and sky numbers
# (Section 9.4).  Every result therefore gets a new scientific digest, including
# the single-component ones below.  **The visibilities themselves did not
# move**: the raw ``sha256`` of the C-contiguous ``complex128`` cube is
# byte-identical at ``6708b0e`` and at this commit, measured in both
# environments --
#
#   ============================  ==================  ==================
#   Run                           py311 cube sha256   py312 cube sha256
#   ============================  ==================  ==================
#   configs/config.yaml           ``cce1bfe8...``     ``7560d2f2...``
#   receptor_circular_example     ``95890bc6...``     ``ff26cb85...``
#   ============================  ==================  ==================
#
# -- so this is a provenance-surface change, not a scientific one.  The
# immediately preceding (post-RUN-005, pre-6F) values were:
#
#   config.yaml                py311 ``b702a202...``  py312 ``e570a9bc...``
#   receptor_circular_example  py311 ``92ce5ce1...``  py312 ``7dd9e7a7...``
#
# Per the Section 36 note on ``C11`` and ``RUN-005``, those -- not any earlier
# acceptance record's values -- are the correct "before" baseline for this diff.
_SHIPPED_CONFIG_FINGERPRINTS: dict[str, dict[str, str]] = {
    "config.yaml": {
        "py311": "4bbb74035b3d700fa7638dca6b854a8c9110bc2abe8d418c7b180f527b947f2b",
        "py312": "9e4f4e164074ad7acf71a6c2c518b1d481a131054445b97e4b1b111be0838e28",
    },
    "receptor_circular_example.yaml": {
        "py311": "be1e86fba57821a95f13f527a72b2ffd42edd4494cc68b0fde68d0f24d042203",
        "py312": "a1ea03d8cf5286149b07543736b3e4cdef90091f8464fc9a04b20f38a736ecab",
    },
}

#: The raw visibility-cube digests recorded above, asserted directly so the
#: "``C11`` moved the fingerprint but not the science" claim is a test, not a
#: comment.  Recipe: ``sha256`` of the C-contiguous ``complex128`` buffer.
_SHIPPED_CONFIG_CUBE_DIGESTS: dict[str, dict[str, str]] = {
    "config.yaml": {
        "py311": "cce1bfe86dc8b3fe81e5c6064a8449afa5bbab95866ec6bc352681dbf1e5ffae",
        "py312": "7560d2f267f372e19ef735afca0cb9ec05ca9f75e2f2ca62a35c52843660f9df",
    },
    "receptor_circular_example.yaml": {
        "py311": "95890bc680c21057c5c23245dc8b67eb7e8662559b3d965905862148a75dd2f8",
        "py312": "ff26cb85289e77cda59a7508dae2e38afeb32bbfb4aff1b98315ac33e2c0177b",
    },
}


def _raw_cube_digest(cube: Any) -> str:
    """Digest a published cube exactly as the pre-6F baseline was digested."""
    array = np.ascontiguousarray(np.asarray(cube))
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def test_shipped_default_config_scientific_fingerprint(tmp_path) -> None:
    """Records the R1 reference for ``configs/config.yaml``.

    The Tier 6D restructure must reproduce this digest bit-for-bit *in the same
    environment*, from any checkout location.  A change here is a scientific
    change and must be justified, never re-pinned silently.
    """
    expected = _expected_for_environment(
        _SHIPPED_CONFIG_FINGERPRINTS["config.yaml"], "configs/config.yaml"
    )
    result = _run_shipped_config("config.yaml", tmp_path)
    assert result.visibilities.shape == (60, 15, 101, 4)
    assert str(result.visibilities.dtype) == "complex128"
    assert result.solver.sky_representation == "point_sources"
    assert result.solver.execution_path == "polarized"
    assert result.solver.components == ("point",)
    assert result.scientific_sha256 == expected
    assert _raw_cube_digest(result.visibilities) == _expected_for_environment(
        _SHIPPED_CONFIG_CUBE_DIGESTS["config.yaml"], "configs/config.yaml"
    )


def test_shipped_circular_receptor_config_scientific_fingerprint(tmp_path) -> None:
    """Records the R1 reference for ``configs/receptor_circular_example.yaml``."""
    expected = _expected_for_environment(
        _SHIPPED_CONFIG_FINGERPRINTS["receptor_circular_example.yaml"],
        "configs/receptor_circular_example.yaml",
    )
    result = _run_shipped_config("receptor_circular_example.yaml", tmp_path)
    assert result.visibilities.shape == (6, 15, 3, 4)
    assert str(result.visibilities.dtype) == "complex128"
    assert result.solver.sky_representation == "point_sources"
    assert result.solver.components == ("point",)
    assert result.scientific_sha256 == expected
    assert _raw_cube_digest(result.visibilities) == _expected_for_environment(
        _SHIPPED_CONFIG_CUBE_DIGESTS["receptor_circular_example.yaml"],
        "configs/receptor_circular_example.yaml",
    )


def test_shipped_realistic_foreground_config_cannot_run_at_this_gate() -> None:
    """Records why R1 cannot cover the third shipped configuration.

    Every VizieR point-catalog loader raises ``TypeError`` because
    ``_load_from_vizier_catalog`` takes ``precision`` keyword-only while all four
    wrapper call sites pass it positionally.  This is a production defect outside
    Tier 6's scope; see this module's docstring.  When it is fixed, this test
    must be replaced by a real fingerprint (network-marked) rather than deleted.
    """
    from radiosim.core.sky.loaders.vizier.point_catalogs import load_gleam

    with pytest.raises(TypeError, match="positional arguments"):
        load_gleam(flux_limit=1000.0, precision=PrecisionConfig.standard())

    config = yaml.safe_load(
        (REPO_ROOT / "configs" / "realistic_foreground_example.yaml").read_text(
            encoding="utf-8"
        )
    )
    assert config["sky_model"]["sources"][0]["bright_catalogs"] == "gleam"
    assert config["visibility"]["sky_representation"] == "healpix_map"


_WORKLOAD_RUNNERS: dict[str, Any] = {
    # Section 13.4 row 1
    "point_unpolarized_1time_2freq": lambda tmp: _run_point_workload(
        tmp, polarized=False, single_time=True
    ),
    # Section 13.4 row 2
    "point_polarized_2times": lambda tmp: _run_point_workload(tmp, polarized=True),
    # Section 13.4 row 3
    "point_gaussian_morphology": lambda tmp: _run_point_workload(
        tmp, polarized=True, gaussian=True
    ),
    # Section 13.4 row 4
    "healpix_scalar": lambda tmp: _run_healpix_workload(tmp, polarized=False),
    # Section 13.4 row 5
    "healpix_polarized": lambda tmp: _run_healpix_workload(tmp, polarized=True),
    # Section 13.4 row 7 (row 6, hybrid, is unreachable -- defect D2)
    "heterogeneous_receptor_bases": lambda tmp: _run_point_workload(
        tmp, polarized=True, heterogeneous=True
    ),
}

_WORKLOAD_SHAPES: dict[str, tuple[int, ...]] = {
    "healpix_polarized": (2, 3, 2, 2, 2),
    "healpix_scalar": (2, 3, 2, 2, 2),
    "heterogeneous_receptor_bases": (2, 3, 2, 2, 2),
    "point_gaussian_morphology": (2, 3, 2, 2, 2),
    "point_polarized_2times": (2, 3, 2, 2, 2),
    "point_unpolarized_1time_2freq": (1, 3, 2, 2, 2),
}

_WORKLOAD_DIGESTS: dict[str, dict[str, str]] = {
    "healpix_polarized": {
        "py311": "201feac2a5d1c8173528a24629d53a4fa51d19ef2eee9bdff667c3eda3c836a5",
        "py312": "72c006b63a70230c7827ef5a618859c1541070bbdabdaada5e4b7edd0c40b1b3",
    },
    "healpix_scalar": {
        "py311": "ed6356f91b7277ad3ad494f6b37b2d78110a7af58eef770fbf7d6729b3af3f7b",
        "py312": "4a701c82b6f7608569dba79d797a531dde5bda54e26ceddc61b7a22ad6d62344",
    },
    "heterogeneous_receptor_bases": {
        "py311": "81055aff940d17817c66fb95ac760962af867ef4a9a3062b1e5bd80991803252",
        "py312": "d39cbe2fde4a3a54c518423ee4c7ee0db2b2664c5caabdf88dbd3d7c7979537d",
    },
    "point_gaussian_morphology": {
        "py311": "9cd139554a45920f6338c4552544e2c490c8597bcd46f915a3f3855d867ae384",
        "py312": "370f7f353ec8ced7f09a8322b0867b6f8e7c2fc3ecf51f160ca8fc9d21939941",
    },
    "point_polarized_2times": {
        "py311": "1140e5917a671af77233b3b244cc0bd7fb15c814a8f5fb70d22cd9c16cd5b9cd",
        "py312": "dabe4c4bc678276a98d03a266ae2e1a9ec39f949bd263ee4da15247bb83f7431",
    },
    "point_unpolarized_1time_2freq": {
        "py311": "b4cc91e5852ef3ad5992c76a770950a68580da7ba73142b920cbcdc28d4f2510",
        "py312": "93cd8c728e387e0e0d24eee5101403b02f8fa44d8556f1644e4904e5feff2f14",
    },
}


@pytest.mark.parametrize("workload", sorted(_WORKLOAD_RUNNERS))
def test_section_13_4_workload_fingerprints(tmp_path, workload: str) -> None:
    """Records raw-cube digests for the reachable Section 13.4 workloads.

    The hybrid row is unreachable at this gate (defect D2) and is 6F's to add.
    ``_cube_digest`` fixes the reproduction recipe, so Tier 6D can prove the
    accumulation restructure changed no number without rerunning a whole config.
    """
    expected_shape = _WORKLOAD_SHAPES[workload]
    expected_digest = _expected_for_environment(
        _WORKLOAD_DIGESTS[workload], f"Section 13.4 workload {workload!r}"
    )
    cube = _WORKLOAD_RUNNERS[workload](tmp_path)
    array = np.asarray(cube)

    assert array.shape == expected_shape
    assert str(array.dtype) == "complex128"
    # A digest of an all-zero cube would pin nothing.
    assert float(np.max(np.abs(array))) > 0.0
    assert _cube_digest(array) == expected_digest


def test_run_records_the_numpy_backend_as_the_actual_backend(tmp_path) -> None:
    """Records the provenance fields Tier 6H must keep truthful."""
    result = Simulator.from_mapping(
        valid_config_mapping(tmp_path), base_dir=tmp_path
    ).run(progress=False)
    assert result.backend.requested_backend == "numpy"
    assert result.backend.actual_backend == "numpy-cpu"


def test_solver_seconds_is_componentized_but_stays_out_of_the_fingerprints(
    tmp_path,
) -> None:
    """Records the closure of the missing per-component timing.

    Flipped by Tier 6F from the 6A pin
    ``test_solver_seconds_is_a_single_uncomponentized_measurement``.
    ``ResultPerformance`` now carries ``solver_point_seconds`` and
    ``solver_healpix_seconds``; because timings are nondeterministic they stay
    outside ``scientific_sha256``, ``provenance_sha256``, and the bounded
    metadata snapshot (Section 9.4).  No ``OWNED BY`` line remains.
    """
    result = Simulator.from_mapping(
        valid_config_mapping(tmp_path), base_dir=tmp_path
    ).run(progress=False)

    assert result.performance.solver_point_seconds > 0.0
    assert result.performance.solver_healpix_seconds == 0.0
    assert result.performance.solver_point_seconds <= result.performance.solver_seconds

    flattened = repr(result.to_summary_snapshot())
    assert "solver_point_seconds" not in flattened
    assert "solver_healpix_seconds" not in flattened
    assert isinstance(time.perf_counter(), float)
