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

Update (standalone SKY-001 remediation slice, after Tier 6 was accepted):
blocker 2 above is **fixed**.  The four call sites in
``core/sky/loaders/vizier/point_catalogs.py`` now pass ``precision`` by keyword,
all ten registered VizieR point-catalog loaders reach the VizieR fetch boundary
again, and ``tests/unit/test_core/test_sky_vizier_loader.py`` guards every one
of them offline.  ``test_shipped_realistic_foreground_config_cannot_run_at_this_gate``
below was flipped in the same commit.  Blocker 1 is unchanged, so
``configs/realistic_foreground_example.yaml`` remains outside hermetic R1
coverage for the network reason alone.

Reproducibility scope -- R1 is per (platform, Python) environment
=================================================================

**Every fingerprint recorded here differs between the two locked Python
environments *and* between the three locked pixi platforms, and the plan's
Section 27 R1 did not say so.**  Each digest is reproducible to the bit *within*
one ``(platform, python)`` environment, and is not reproducible across either
axis.  ``_ENVIRONMENT_KEY`` therefore names both axes -- ``linux-64-py311``,
``osx-64-py312``, ``osx-arm64-py311`` and so on -- one key per cell of the
Section 31 / CI matrix.  Neither axis is nondeterminism, and neither is the
solver.

Axis 1 -- Python environment (astropy).  The ``default``/py311 environment
resolves astropy 7.1.0 while ``py312`` resolves astropy 8.0.1, and the
ICRS -> AltAz transform of the same source at the same instant differs between
them --

* astropy 7.1.0: ``alt = 1.5668104524223887``, ``az = 1.8421809886140983``
* astropy 8.0.1: ``alt = 1.5668104524079418``, ``az = 1.8421809682045285``

-- a ~1.4e-11 rad altitude and ~2.0e-8 rad azimuth difference which the
geometric phase amplifies into every visibility.  ``numpy`` also differs (2.3.2
vs 2.4.6), but the coordinate difference alone is sufficient and was measured
directly.  This axis was diagnosed by Tier 6A's own acceptance.

Axis 2 -- platform (CPU architecture and platform math libraries).  Diagnosed by
the Tier 6J whole-tier rejection (``Fix.md``, 2026-07-31) and repaired here.  The
three locked pixi platforms produce three *different* raw visibility cubes for
the identical source, the identical locked package versions and the identical
Python version.  ``osx-arm64`` (``arm64``) is the only platform any Tier 6
implementer or reviewer ran locally, so it was the only one pinned, and CI was
red on the other two for the entire tier.  Crucially, ``linux-64`` and
``osx-64`` -- both ``x86_64`` -- do **not** agree with each other either, so the
divergence is not a bare ``arm64``-vs-``x86_64`` split and a machine-only axis
would not have been enough; the platform's own libm/BLAS build participates.

The evidence that this is inherent to the code the tier inherited, not something
Tier 6 introduced, is that the six Section 13.4 raw-cube digests measured by CI
on ``linux-64``/py311 at the *6A characterization commit itself* (run
``30531414992``, which added these pins and no production code at all) are
byte-identical to the ones measured at the end of the tier (run
``30628921601``), and likewise for ``osx-64``/py311 -- while the ``osx-arm64``
pins they are compared against never moved either.  Every Tier 6 production
restructure (6D block assembly, 6E solver workers, 6F hybrid summation, 6H
batched contraction) therefore changed *no* number on *any* of the three
platforms; the platform spread was already there before the first line of Tier 6
production code and is architecture-level floating-point non-associativity in the
vectorized trig/BLAS paths, not a determinism defect.  Two further checks support
pinning it rather than relaxing it: the two ``linux-64`` runs above ran on
different host CPUs (AMD EPYC 7763 and AMD EPYC 9V74) and still agreed to the
byte, and the pins are bit-stable across repeated runs within a platform.

Axis 3 -- the individual x86_64 runner.  Found immediately after the axis-2
repair went in, and the reason every pin below is a *set* of observed digests
rather than one value.  On ``linux-64``/py312, CI run ``30640039816`` measured
digests that differ from the ones runs ``30628921601`` and ``30631837095``
measured for the identical source -- and the CPU *model* string does not explain
it: the divergent run and one of the two agreeing runs both report
``AMD EPYC 9V74``, while the other agreeing run reports ``AMD EPYC 7763``.  Two
of the three runs agree across different CPU models; two runs on the same CPU
model disagree.  The discriminating machine property is therefore something the
model string does not capture -- most plausibly the vectorized code path NumPy
actually dispatches to, which depends on the CPU feature set the virtual machine
exposes (an AVX-512-capable part may or may not expose it) rather than on the
part number.  That has not been proven, only narrowed, which is exactly why
``_machine_fingerprint`` now attaches NumPy's dispatched feature set to every pin
failure: the next divergence must arrive with the evidence needed to name this
axis instead of narrowing it again.  That evidence has since arrived through the
raw-cube pins (``_SHIPPED_CONFIG_CUBE_DIGESTS``): the same runner class measured
byte-identical second-observation cubes on an ``AMD EPYC 9V74`` (run
``30646860127``) and an ``Intel(R) Xeon(R) Platinum 8370C`` (run
``30651948058``), both dispatching the AVX-512 tiers -- the class crosses CPU
vendors and models but tracks the dispatched feature set.

The variance is *per digest*, not per environment: run ``30640039816`` moved the
two shipped-config fingerprints and the ``heterogeneous_receptor_bases``
workload, while the other five Section 13.4 workloads were unaffected, as one
would expect when only some kernels differ between dispatch paths.

Axis 3 is emphatically **not** nondeterminism within a run, and the same failing
job proves it: every within-process reproducibility test passed there -- solver
worker invariance at 1/2/3/4 workers, loader worker invariance across
``{1,2,4,8}`` x ``{thread,process}``, per-solver bit-identity under workers, and
hybrid additivity.  A race or a hash-seed-ordered reduction would have made those
flaky. Inside one process the computation is bit-reproducible; it is the machine
that varies.

Consequence for later slices and for reviewers: R1 ("post-restructure
``scientific_sha256`` equals the pinned pre-restructure value") is only
meaningful when the comparison runs in the *same* ``(platform, python)``
environment as the pin, and even then only up to the recorded set of digests
that environment's fleet has been observed to produce.  Cross-environment
agreement is a **tolerance-level** claim (Section 13.5), never a bit-level one,
and no Tier 6 fingerprint may be treated as an environment-independent constant
in documentation.  An unrecorded digest -- in an uncharacterized environment, or
a value never seen in a characterized one -- is a hard, loud failure that prints
what it measured together with the machine fingerprint, so the observation can be
adjudicated and recorded deliberately.  A set never grows to make a failure go
away: a new value is either a real regression or a newly observed machine class,
and which one it is must be decided before it is written down.  Within one
environment the pins remain independent of *where* the tree is checked out: since
the RUN-005 fix (``fix(result): exclude filesystem transport facts from
scientific_sha256``) those digests no longer depend on the checkout path.

Tier 8A instrumentation (``CI-001``)
====================================

Axis 3's stated discriminator -- the dispatched vector feature set -- was
**falsified** by the observations recorded in ``Fix.md``'s ``CI-001`` row: a
second byte-stable digest class on ``linux-64-py311`` reproduces identically
across three CI runs, two CPU vendors and three CPU models whose
``numpy.__cpu_features__`` lists differ from one another.  The prose above is
left standing as the record of what was believed when the pins were written;
``CI-001`` is the correction, and it is the register row that owns the unnamed
discriminator.

Slice 8A changed exactly two things here, both on the *evidence* path and
neither on any assertion:

1. ``_record_machine_fingerprint`` writes the machine fingerprint to
   ``output/characterization/`` **unconditionally**, on pass as well as on
   failure.  Until now ``_machine_fingerprint()`` was reachable only from
   ``_assert_pinned_digests``'s ``pytest.fail`` branch, so the fleet described a
   runner only when that runner *disagreed*; nothing was ever recorded about a
   runner that produced an accepted digest, which is the single largest reason
   this divergence is undiagnosable.  The fingerprint itself was widened from
   (CPU model, dispatched features) to add the thread environment and the BLAS
   build, since the feature set alone is now known to be insufficient.
2. ``_assert_pinned_digests`` accepts an optional fourth element per check --
   the array the digest was taken over.  A pass captures it as a reference cube
   for this ``(pin, environment)``; a failure reports ``max|dV|``, the maximum
   relative delta, the differing-element count and the first differing index
   against every captured reference, and names the nearest recorded observation.
   The gate could not previously tell one ULP from one hundred percent, and no
   failing log in the last twenty-five CI runs contained a single number.

Both write only into ``output/`` (gitignored) and both swallow every error: a
diagnostic that can fail a test is worse than no diagnostic.  No digest table
grew, and 8A deliberately did **not** append the divergent ``linux-64-py311``
class -- see ``Fix.md`` ``CI-001`` and ``Tier8ReleasePlan.md`` Section 14 for
why appending under a falsified rationale is forbidden.

The post-Tier-8 WP-2 extension (``PostTier8RemediationPlan.md`` Section 5.2)
widened the fingerprint again, on the same evidence-only terms as 8A: it adds
the libc/glibc version (targets ``libm``), the GitHub runner image identity
(``ImageOS``/``ImageVersion``), NumPy's runtime BLAS report including the
OpenBLAS runtime core name (the "OpenBLAS runtime dispatch" datum from
``CI-001``'s remaining hypothesis space), the CPU count and scheduler
affinity, and the cache topology (OpenBLAS blocking follows detected cache
sizes, which vary across VM SKUs with identical CPU model strings).  Every
field is best-effort, cross-platform, and exception-swallowed; the extension
changes no assertion, no digest, and no test outcome.
"""

from __future__ import annotations

import contextlib
import hashlib
import importlib
import inspect
import io
import os
import platform
import re
import subprocess
import sys
import time
import tomllib
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml
from astropy import units as u
from astropy.coordinates import EarthLocation
from astropy.table import Table
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
from tests.support.repo_scan import PYTHON_SUFFIXES, iter_tracked_files

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
# source, along two independent axes -- the locked Python environment (astropy's
# ICRS->AltAz transform changed between 7.1.0/py311 and 8.0.1/py312) and the
# platform (the three locked pixi platforms disagree in the last bits of the
# vectorized trig/BLAS paths, and the two x86_64 platforms disagree with each
# other as well as with arm64).  See the "Reproducibility scope" note in this
# module's docstring for the evidence that the platform axis predates every Tier
# 6 production change.
_PIXI_ARCHITECTURES = {
    "x86_64": "64",
    "AMD64": "64",
    "amd64": "64",
    "arm64": "arm64",
    "aarch64": "aarch64",
}


def _platform_key() -> str:
    """Return the pixi platform name for the running interpreter.

    The three names this can produce in the locked matrix are ``linux-64``,
    ``osx-64`` and ``osx-arm64``, exactly the ``pixi.toml`` ``platforms`` list
    and exactly the CI job matrix.  Anything else falls through to a descriptive
    key that no pin table contains, which is the intended loud failure.
    """
    operating_system = {"linux": "linux", "darwin": "osx", "win32": "win"}.get(
        sys.platform, sys.platform
    )
    machine = platform.machine()
    return f"{operating_system}-{_PIXI_ARCHITECTURES.get(machine, machine)}"


_PLATFORM_KEY = _platform_key()
_PYTHON_KEY = f"py{sys.version_info[0]}{sys.version_info[1]}"
_ENVIRONMENT_KEY = f"{_PLATFORM_KEY}-{_PYTHON_KEY}"
_MEASURED_ENVIRONMENTS = {
    "linux-64-py311": "linux-64, python 3.11.13, numpy 2.3.2, astropy 7.1.0",
    "linux-64-py312": "linux-64, python 3.12.13, numpy 2.4.6, astropy 8.0.1",
    "osx-64-py311": "osx-64, python 3.11.13, numpy 2.3.2, astropy 7.1.0",
    "osx-64-py312": "osx-64, python 3.12.13, numpy 2.4.6, astropy 8.0.1",
    "osx-arm64-py311": "osx-arm64, python 3.11.13, numpy 2.3.2, astropy 7.1.0",
    "osx-arm64-py312": "osx-arm64, python 3.12.13, numpy 2.4.6, astropy 8.0.1",
}


def _thread_environment() -> str:
    """Report the thread-count environment a vectorized reduction can depend on."""
    names = (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "GOTO_NUM_THREADS",
    )
    settings = [f"{name}={os.environ[name]}" for name in names if name in os.environ]
    if not settings:
        settings = ["<none set>"]
    return f"{' '.join(settings)} (os.cpu_count()={os.cpu_count()})"


def _blas_build() -> str:
    """Report the BLAS/LAPACK build NumPy is actually linked against.

    ``numpy.__config__.show(mode="dicts")`` is the only machine-readable form of
    this fact; it is a NumPy-internal shape rather than a stability guarantee, so
    every access is defensive and any failure degrades to a marker string rather
    than masking the pin failure this text accompanies.
    """
    try:
        config = np.__config__.show(mode="dicts")  # type: ignore[call-arg]
        libraries = config.get("Build Dependencies", {})
        parts = []
        for key in ("blas", "lapack"):
            entry = libraries.get(key)
            if isinstance(entry, dict):
                parts.append(
                    f"{key}={entry.get('name', '?')} {entry.get('version', '?')}"
                    f" [openblas configuration: "
                    f"{entry.get('openblas configuration', '?')}]"
                )
        if parts:
            return " ".join(parts)
    except Exception:  # pragma: no cover - a NumPy internal, not public API
        pass
    return "unavailable"


def _libc_fingerprint() -> str:
    """Report the C library the process runs against; targets ``libm``.

    ``CI-001``'s remaining hypothesis space names ``libm`` dispatch explicitly.
    ``platform.libc_ver()`` inspects the interpreter binary; on Linux,
    ``os.confstr("CS_GNU_LIBC_VERSION")`` adds the glibc the process actually
    loaded.  Best-effort on every axis (WP-2, evidence path only).
    """
    parts: list[str] = []
    try:
        library, version = platform.libc_ver()
        if library or version:
            parts.append(f"{library} {version}".strip())
    except Exception:  # pragma: no cover - diagnostics must never fail a run
        pass
    try:
        runtime = os.confstr("CS_GNU_LIBC_VERSION")
        if runtime:
            parts.append(f"runtime {runtime}")
    except Exception:  # pragma: no cover - Linux-only constant
        pass
    return "; ".join(parts) if parts else "unavailable"


def _runner_image() -> str:
    """Report the GitHub-hosted runner image identity, when present.

    The cheapest possible ``CI-001`` discriminator: if digest-class membership
    tracks ``ImageOS``/``ImageVersion``, the axis is the image's ``libm`` and
    the search ends there.  Off CI both variables are unset, and saying so is
    itself the datum.
    """
    image_os = os.environ.get("ImageOS")
    image_version = os.environ.get("ImageVersion")
    if image_os or image_version:
        return (
            f"ImageOS={image_os or '<unset>'} ImageVersion={image_version or '<unset>'}"
        )
    return "<ImageOS/ImageVersion unset: not a GitHub-hosted runner>"


def _numpy_runtime() -> str:
    """Report NumPy's runtime BLAS state, including the OpenBLAS core name.

    OpenBLAS picks its kernels at *runtime* from the CPU it lands on -- the
    "OpenBLAS runtime dispatch" datum ``CI-001`` names as uncaptured.  Three
    best-effort sources, most direct first: ``openblas_get_corename()`` /
    ``openblas_get_config()`` via ``ctypes`` on the BLAS the process already
    loaded (no new dependency); ``threadpoolctl`` (the same source
    ``numpy.show_runtime()`` uses), whose ``architecture`` field is the core
    name; and finally the captured ``numpy.show_runtime()`` text.
    """
    entries: list[str] = []
    try:
        import ctypes

        for soname in (
            "libblas.so.3",
            "libopenblas.so.0",
            "libblas.3.dylib",
            "libopenblas.0.dylib",
        ):
            try:
                library = ctypes.CDLL(soname)
                library.openblas_get_corename.restype = ctypes.c_char_p
                library.openblas_get_config.restype = ctypes.c_char_p
                core = library.openblas_get_corename()
                config = library.openblas_get_config()
                entries.append(
                    f"openblas runtime core="
                    f"{(core or b'?').decode('ascii', 'replace')}"
                    f" config={(config or b'?').decode('ascii', 'replace')!r}"
                    f" (via {soname})"
                )
                break
            except (OSError, AttributeError):
                continue
    except Exception:  # pragma: no cover - diagnostics must never fail a run
        pass
    try:
        from threadpoolctl import threadpool_info  # type: ignore[import-not-found]

        for pool in threadpool_info():
            entries.append(
                f"{pool.get('internal_api', '?')} {pool.get('version', '?')}"
                f" core={pool.get('architecture', '?')}"
                f" threads={pool.get('num_threads', '?')}"
            )
    except Exception:  # pragma: no cover - optional dependency
        pass
    if entries:
        return "; ".join(entries)
    try:
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            np.show_runtime()
        flattened = " ".join(buffer.getvalue().split())
        if flattened:
            return flattened
    except Exception:  # pragma: no cover - diagnostics must never fail a run
        pass
    return "unavailable"


def _cpu_topology() -> str:
    """Report the CPU count and the scheduler affinity actually granted."""
    try:
        count = os.cpu_count()
    except Exception:  # pragma: no cover - diagnostics must never fail a run
        count = None
    try:
        affinity: object = len(os.sched_getaffinity(0))
    except Exception:  # pragma: no cover - Linux-only API
        affinity = "unavailable"
    return f"os.cpu_count()={count} sched_getaffinity={affinity}"


def _cache_topology() -> str:
    """Report the CPU cache topology (``lscpu`` on Linux, ``sysctl`` on macOS).

    OpenBLAS blocking follows detected cache sizes, which vary across VM SKUs
    with identical CPU model strings -- one of the few axes that crosses
    vendors while staying byte-stable per class (WP-2 rationale).
    """
    if sys.platform.startswith("linux"):
        command = ["lscpu"]
    elif sys.platform == "darwin":
        command = ["sysctl", "hw"]
    else:  # pragma: no cover - no locked platform reaches this
        return "unavailable"
    try:
        completed = subprocess.run(
            command, capture_output=True, text=True, timeout=10, check=False
        )
        lines = [
            " ".join(line.split())
            for line in completed.stdout.splitlines()
            if "cache" in line.lower()
        ]
        if lines:
            return "; ".join(lines)
    except Exception:  # pragma: no cover - diagnostics must never fail a run
        pass
    return "unavailable"


def _machine_fingerprint() -> str:
    """Describe the machine facts a raw-cube digest is actually a function of.

    Attached to every pin failure, and -- since Tier 8A -- also written to disk
    unconditionally by ``_record_machine_fingerprint`` on pass as well as fail.
    The whole reason the third correction to Section 27 R1 exists is that a
    divergence was observed without any evidence of *what* differed between the
    two runners, so the next divergence must arrive with that evidence already
    attached.  NumPy's dispatched CPU feature set is the primary suspect and the
    primary datum; the CPU model string is recorded too, having already been
    proven insufficient on its own, and Tier 8A added the thread environment and
    the BLAS build after the CI-001 observation falsified the feature-set
    explanation as well (``Fix.md`` register row ``CI-001``).

    The WP-2 extension (``PostTier8RemediationPlan.md`` Section 5.2) added the
    libc/glibc version, the runner image identity, NumPy's runtime BLAS report
    with the OpenBLAS core name, the CPU topology, and the cache topology --
    the axes ``CI-001`` names as uncaptured (hypervisor CPU-feature masking
    and ``libm``/OpenBLAS runtime dispatch).  Evidence path only: every field
    is best-effort and the extension changes no assertion, no digest, and no
    test outcome.
    """
    try:
        from numpy._core._multiarray_umath import (  # type: ignore[import-not-found]
            __cpu_features__ as cpu_features,
        )

        features = ",".join(sorted(name for name, on in cpu_features.items() if on))
    except Exception:  # pragma: no cover - a NumPy internal, not public API
        features = "unavailable"
    try:
        from radiosim.utils.device import get_device_resources

        model = get_device_resources().cpu.model or platform.processor()
    except Exception:  # pragma: no cover - diagnostics must never mask a failure
        model = platform.processor()
    return (
        f"environment key: {_ENVIRONMENT_KEY}\n"
        f"cpu model: {model!r}\n"
        f"numpy dispatched features: {features}\n"
        f"thread environment: {_thread_environment()}\n"
        f"blas build: {_blas_build()}\n"
        f"libc: {_libc_fingerprint()}\n"
        f"runner image: {_runner_image()}\n"
        f"numpy runtime: {_numpy_runtime()}\n"
        f"cpu topology: {_cpu_topology()}\n"
        f"cache topology: {_cache_topology()}\n"
        f"python {platform.python_version()}, numpy {np.__version__}, "
        f"platform {platform.platform()}"
    )


#: Where the unconditional machine-fingerprint record and the reference cubes
#: land.  ``output/`` is gitignored (``.gitignore:176``) apart from two named
#: exceptions, so nothing written here can dirty a working tree or enter a
#: commit.  Setting the environment variable to a path relocates the directory
#: (which is how a CI job stages the record as a build artifact); setting it to
#: the empty string disables every write.
_RECORD_DIR_ENV = "RADIOSIM_CHARACTERIZATION_RECORD_DIR"

#: Reference cubes larger than this are not captured.  The published
#: ``configs/config.yaml`` cube is 5.8 MB; the cap exists so that a future,
#: larger pinned workload cannot silently fill a contributor's disk.
_MAX_REFERENCE_CUBE_BYTES = 64 * 1024 * 1024


def _record_dir() -> Path | None:
    """Return the diagnostics directory, or ``None`` when recording is disabled."""
    raw = os.environ.get(_RECORD_DIR_ENV)
    if raw is not None and not raw.strip():
        return None
    base = Path(raw) if raw else REPO_ROOT / "output" / "characterization"
    try:
        base.mkdir(parents=True, exist_ok=True)
    except Exception:  # pragma: no cover - diagnostics must never fail a run
        return None
    return base


def _record_slug(what: str) -> str:
    """Turn a pin's human description into a filesystem-safe directory name."""
    return re.sub(r"[^A-Za-z0-9]+", "-", what).strip("-").lower()


def _record_machine_fingerprint() -> None:
    """Write this session's machine fingerprint, on pass as well as on failure.

    Tier 8A, ``CI-001``.  Before this, ``_machine_fingerprint()`` was reachable
    only from ``_assert_pinned_digests``'s ``pytest.fail`` branch, so the fleet
    left a record of a runner *only when that runner disagreed*.  That is exactly
    backwards for adjudicating a second digest class: the question "what was
    different about the machines that produced the recorded value?" had no
    recorded answer at all, because no passing ``linux-64-py311`` runner had ever
    described itself.  The file is per (environment, xdist worker) so parallel
    workers cannot clobber each other, and every failure mode is swallowed --
    a diagnostic that can fail a test is worse than no diagnostic.

    This changes no assertion, no digest, and no test outcome; it only makes the
    pass path leave evidence behind.
    """
    base = _record_dir()
    if base is None:
        return
    worker = os.environ.get("PYTEST_XDIST_WORKER", "main")
    path = base / f"machine-fingerprint-{_ENVIRONMENT_KEY}-{worker}.txt"
    try:
        path.write_text(
            f"# RadioSim characterization machine fingerprint (CI-001)\n"
            f"# written unconditionally, on pass as well as on failure\n"
            f"recorded (UTC): {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}\n"
            f"xdist worker: {worker}\n"
            f"characterized as: "
            f"{_MEASURED_ENVIRONMENTS.get(_ENVIRONMENT_KEY, 'never characterized')}\n"
            f"{_machine_fingerprint()}\n",
            encoding="utf-8",
        )
    except Exception:  # pragma: no cover - diagnostics must never fail a run
        return


_record_machine_fingerprint()


def _reference_cube_dir(what: str) -> Path | None:
    """Return the directory holding reference cubes for one pin, if enabled."""
    base = _record_dir()
    if base is None:
        return None
    return base / "reference_cubes" / _record_slug(what) / _ENVIRONMENT_KEY


def _capture_reference_cube(what: str, digest: str, cube: Any) -> None:
    """Store a cube whose digest is a *recorded* observation, for later deltas.

    Tier 8A, ``CI-001``.  A digest gate cannot tell 1 ULP from 100%: the failing
    logs of the last 25 CI runs contain hex strings and not one number.  A
    numeric delta needs something to subtract, and the only honest reference is a
    cube that was measured while its digest still matched the pin -- so the
    capture happens on the *pass* path, keyed by the digest it matched.  Writes
    are one-shot (an existing file is never rewritten), size-capped, and entirely
    best-effort.
    """
    if cube is None:
        return
    directory = _reference_cube_dir(what)
    if directory is None:
        return
    path = directory / f"{digest}.npy"
    if path.exists():
        return
    try:
        array = np.ascontiguousarray(np.asarray(cube))
        if array.nbytes > _MAX_REFERENCE_CUBE_BYTES:
            return
        directory.mkdir(parents=True, exist_ok=True)
        np.save(path, array)
    except Exception:  # pragma: no cover - diagnostics must never fail a run
        return


def _cube_delta(measured: Any, reference: Any) -> str:
    """Report ``max|dV|``, ``max relative d``, and the first differing element."""
    left = np.ascontiguousarray(np.asarray(measured))
    right = np.ascontiguousarray(np.asarray(reference))
    if left.shape != right.shape:
        return f"shape differs: measured {left.shape} vs recorded {right.shape}"
    difference = np.abs(left - right)
    max_absolute = float(np.max(difference)) if difference.size else 0.0
    scale = np.maximum(np.abs(right), np.abs(left))
    with np.errstate(divide="ignore", invalid="ignore"):
        relative = np.where(scale > 0.0, difference / scale, 0.0)
    max_relative = float(np.max(relative)) if relative.size else 0.0
    differing = np.flatnonzero(left.ravel() != right.ravel())
    count = int(differing.size)
    if count == 0:
        return "identical to the byte (0 differing elements)"
    first = tuple(int(axis) for axis in np.unravel_index(int(differing[0]), left.shape))
    return (
        f"max|dV| = {max_absolute!r}, max relative d = {max_relative!r}, "
        f"{count} of {left.size} elements differ, first at index {first} "
        f"(measured {left[first].item()!r} vs recorded {right[first].item()!r})"
    )


def _cube_delta_report(
    table: dict[str, tuple[str, ...]], what: str, cube: Any
) -> list[str]:
    """Compare a measured cube against every recorded observation held on disk.

    Tier 8A, ``CI-001``, design item 3.  The delta is reported against *each*
    recorded observation that has a captured reference, and the nearest -- the
    one with the smallest ``max|dV|`` -- is named, because the adjudication
    question is "how far is this class from the class we already accepted?".
    When nothing has been captured the report says so and states the recipe that
    would produce one, so a reader is never left with an unexplained silence.
    """
    if cube is None:
        return []
    directory = _reference_cube_dir(what)
    recorded = table.get(_ENVIRONMENT_KEY, ())
    if directory is None:
        return ["  numeric delta: recording is disabled via " + _RECORD_DIR_ENV]
    available = [
        (digest, directory / f"{digest}.npy")
        for digest in recorded
        if (directory / f"{digest}.npy").exists()
    ]
    if not available:
        return [
            "  numeric delta: unavailable -- no reference cube has been captured "
            f"for {_ENVIRONMENT_KEY} in {directory}.  A reference is written "
            "automatically the next time this pin *passes* on this machine; to "
            "compare a divergent runner against an accepted one, stage that "
            f"directory (see {_RECORD_DIR_ENV}) from a passing run.",
        ]
    lines = ["  numeric delta against each captured reference cube:"]
    distances: list[tuple[float, str]] = []
    for digest, path in available:
        try:
            reference = np.load(path)
        except Exception:  # pragma: no cover - diagnostics must never fail a run
            lines.append(f"    {digest[:16]}...: reference unreadable at {path}")
            continue
        lines.append(f"    {digest[:16]}...: {_cube_delta(cube, reference)}")
        try:
            left = np.ascontiguousarray(np.asarray(cube))
            if left.shape == reference.shape:
                distances.append((float(np.max(np.abs(left - reference))), digest))
        except Exception:  # pragma: no cover - diagnostics must never fail a run
            continue
    if distances:
        distance, digest = min(distances)
        lines.append(
            f"  nearest recorded observation: {digest} at max|dV| = {distance!r}"
        )
    return lines


def _pin_problem(table: dict[str, tuple[str, ...]], what: str, measured: str) -> str:
    """Describe how a measured digest fails its pin, or return ``""`` if it does not.

    Each pin is a *recorded observation set*, not a single value: within one
    ``(platform, python)`` environment the x86_64 CI fleet has been observed to
    produce more than one digest for the same source, and pretending otherwise
    would mean either a permanently red gate or a silently relaxed one.  A digest
    that has been observed and recorded before passes; anything else -- an
    uncharacterized environment, or a value never seen in this one -- fails
    loudly and prints what it measured.  The set only ever grows deliberately,
    one reviewed CI observation at a time; see the "Reproducibility scope" note
    in this module's docstring.
    """
    if _ENVIRONMENT_KEY not in table:
        return (
            f"{what}: no digest has ever been recorded for environment "
            f"{_ENVIRONMENT_KEY} "
            f"({_MEASURED_ENVIRONMENTS.get(_ENVIRONMENT_KEY, 'never characterized')})"
            f".\n  measured:  {measured}\n  recorded environments: "
            f"{sorted(table)}"
        )
    recorded = table[_ENVIRONMENT_KEY]
    if measured not in recorded:
        return (
            f"{what}: digest not among those recorded for environment "
            f"{_ENVIRONMENT_KEY}.\n  measured:  {measured}\n  recorded:  "
            + "\n             ".join(recorded)
        )
    return ""


def _assert_pinned_digests(
    *checks: (
        tuple[dict[str, tuple[str, ...]], str, str]
        | tuple[dict[str, tuple[str, ...]], str, str, Any]
    ),
) -> None:
    """Check every ``(table, what, measured)`` triple and report all failures at once.

    Deliberately non-short-circuiting.  When these checks were plain chained
    ``assert`` statements, the first one to fail hid every later measurement in
    the same test, and each hidden value cost a whole CI round to harvest.  One
    run must now surface everything a reviewer needs.

    Tier 8A (``CI-001``) added an **optional fourth element**: the array the
    digest was computed over.  When it is supplied, a pass captures that cube as
    a reference for this ``(pin, environment)`` and a failure reports a numeric
    delta against every captured reference instead of two bare hex strings.  The
    element is optional so that ``test_tier7_current_behavior.py``'s call sites,
    which are outside this slice, keep working unchanged; a three-element check
    behaves exactly as it did before.
    """
    problems: list[str] = []
    for check in checks:
        table, what, measured = check[0], check[1], check[2]
        cube = check[3] if len(check) == 4 else None
        problem = _pin_problem(table, what, measured)
        if not problem:
            _capture_reference_cube(what, measured, cube)
            continue
        problems.append("\n".join([problem, *_cube_delta_report(table, what, cube)]))
    if problems:
        pytest.fail(
            "\n".join(problems)
            + "\n\nEvery pin here is a recorded set of observed digests.  A value "
            "that is not in the set is either a real regression -- a change that "
            "moved the numbers -- or a runner whose vectorized floating-point "
            "behaviour has never been observed in this environment.  Decide "
            "which before recording it, and never record a value you cannot "
            "explain.  Never relax the assertion and never reuse another "
            "environment's digest.\n" + _machine_fingerprint()
        )


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
        for path in iter_tracked_files(
            REPO_ROOT / "src" / "radiosim", suffixes=PYTHON_SUFFIXES
        )
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
    is what ``_build_jones_chain`` actually builds.

    ANCHOR UPDATED BY: Tier 7F, which moved ``P`` sky-side of ``C``
    (``Tier7JonesSciencePlan.md`` Section 12.2, defect D12) and added the three
    Tier 7E diagonal terms to the written-out order.  The Tier 6H property --
    that the class docstring states the order the solver builds rather than a
    superseded one -- is what is re-asserted, against the current canonical
    order and against both stale ones.
    """
    docstring = RIMESimulator.__doc__ or ""
    assert "J = B @ G @ D @ P @ E @ T @ Z @ K" not in docstring
    assert "J = H @ G @ B @ D @ P @ C @ E @ T @ Z" not in docstring
    assert "J = H @ G @ B @ Rc @ Kd @ X @ D @ C @ E @ P @ T @ Z" in " ".join(
        docstring.split()
    )


def test_point_solver_accumulates_one_set_at_per_time_baseline_frequency(
    tmp_path,
) -> None:
    """Pins D11 for the point solver.

    OWNED BY: Tier 6D.  FLIPPED BY: Tier 6D -- the per-``(t, b, f)`` ``set_at``
    accumulation is gone.  The solver now assembles one ``(B, 2, 2)`` block per
    ``(time, frequency)``, one ``(B, F, 2, 2)`` block per time, and exactly one
    ``(T, B, F, 2, 2)`` cube per call (Section 13.3, test R2), so the call count
    drops from ``T*B*F`` functional whole-cube copies to a handful of
    assemblies and zero ``set_at`` calls.  The shape itself is asserted in
    ``tests/unit/test_core/test_visibility_accumulation.py``; this pin only
    records that the old shape is truly gone.

    Count narrowed by Tier 6H: Section 13.6's compiled kernel is
    baseline-batched and returns the whole ``(B, 2, 2)`` block from one call, so
    the ``T*F`` per-``(time, frequency)`` assemblies of Tier 6D disappear and
    only ``T + 1`` remain.  Strictly fewer assemblies; the D11 property being
    pinned -- zero ``set_at`` calls and exactly one whole-cube assembly -- is
    unchanged.
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
    assert backend.stack_calls == n_times + 1
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
    assert backend.stack_calls == n_times + 1
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

    Anchor updated again by Tier 7B, which closed defect D4 by routing this path
    through the shared Jones chain.  What used to be a constant ``H_p @ C_p``
    *matrix product* is now the pair of run-constant chain terms that produce
    it, and the Tier 6D property is preserved by hoisting those terms to exactly
    the same place: ``_resolved_receptor_terms`` is called once, above the time
    loop.  The one assertion that could not survive is the frequency loop's
    non-enumeration: the direction-batched contract passes a frequency index to
    every term, so the loop now enumerates.  That was never the property being
    pinned -- Tier 6D's sentence about it records only that the *old* reason for
    ``freq_idx`` (indexing a per-cell output write) is gone, and it still is:
    the index is passed to terms, never used to write into a cube.
    """
    source = _source("src/radiosim/core/visibility_healpix.py")
    time_loop = source.index("def _time_block(time_idx: int")
    transforms = source.index("receptor_terms = _resolved_receptor_terms(")
    frequency_loop = source.index("for freq_idx, freq in enumerate(frequencies):")
    assert transforms < time_loop < frequency_loop
    assert source.count("def _time_block(") == 1
    assert source.count("receptor_terms = _resolved_receptor_terms(") == 1
    assert "_receptor_transforms" not in source
    # No per-cell output write survives: the index reaches terms, not a cube.
    assert "set_at" not in source


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


def test_exactly_one_solver_call_site_requests_compilation() -> None:
    """Flipped by Tier 6H, closing D14's second half.

    The 6A pin recorded the defect: the compilation helpers had no caller
    anywhere, so ``jit``/``vmap`` were decoration rather than capability.  Plan
    Section 13.6 authorizes exactly **one** compiled kernel -- the
    per-(time, frequency) baseline-batched contraction -- so this pin now
    asserts the boundary in the other direction: ``backend.compile`` is called
    from the solver contraction module and nowhere else, and no direct
    ``.jit(``/``.vmap(`` call escapes the JAX backend.
    """
    compile_callers: list[str] = []
    private_callers: list[str] = []
    compile_pattern = re.compile(r"\bbackend\.compile\s*\(")
    private_pattern = re.compile(r"\.(jit|vmap|jit_compile)\s*\(")
    for path in iter_tracked_files(
        REPO_ROOT / "src" / "radiosim", suffixes=PYTHON_SUFFIXES
    ):
        text = path.read_text(encoding="utf-8")
        relative = str(path.relative_to(REPO_ROOT))
        for match in compile_pattern.finditer(text):
            line = text[: match.start()].count("\n") + 1
            compile_callers.append(f"{relative}:{line}")
        if path.name == "jax_backend.py":
            continue
        for match in private_pattern.finditer(text):
            line = text[: match.start()].count("\n") + 1
            private_callers.append(f"{relative}:{line}")

    assert private_callers == []
    assert [caller.split(":")[0] for caller in compile_callers] == [
        "src/radiosim/core/contraction.py"
    ], compile_callers


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

    AMENDED BY: Tier 7J.  ``Tier7JonesSciencePlan.md`` Section 34 authorizes an
    optional ``crossval`` environment for the Section 29 Tier-2 comparison, so
    ``default``'s declaration gained a ``solve-group`` and is no longer the bare
    list form this test spelled out.  The property Tier 6H wrote it to protect
    is what is asserted instead, and it is now asserted over *every* declared
    environment rather than over two named strings: each one carries the
    ``jax-cpu`` feature, so the parity evidence runs wherever the suite runs and
    a future environment cannot quietly drop it.
    """
    pixi_toml = _source("pixi.toml")
    assert "[feature.jax-cpu.dependencies]" in pixi_toml
    assert 'jax = ">=0.10.2,<0.11"' in pixi_toml
    assert 'build = "cpu*"' in pixi_toml
    assert 'numpy = ">=1.24,<2.5"' in pixi_toml

    manifest = tomllib.loads(pixi_toml)
    environments = manifest["environments"]
    assert set(environments) == {"default", "py312", "crossval"}
    for name, declaration in environments.items():
        features = (
            declaration if isinstance(declaration, list) else declaration["features"]
        )
        assert "jax-cpu" in features, name
    # The optional environment is `default` plus one feature, in `default`'s own
    # solve group, so it cannot resolve a different stack (Section 29).
    assert (
        environments["crossval"]["solve-group"]
        == (environments["default"]["solve-group"])
    )
    assert set(environments["default"]["features"]) < set(
        environments["crossval"]["features"]
    )
    assert "[feature.py311.dependencies]" in pixi_toml
    assert "[feature.py312.dependencies]" in pixi_toml

    # The dependency is real, not just declared: the six formerly-skipping
    # tests can only run because this import succeeds in the gate environment.
    import jax

    assert jax.devices()[0].platform == "cpu"


def test_the_benchmark_harness_task_and_performance_test_now_exist() -> None:
    """Closes D15.

    FLIPPED BY: Tier 6I -- the defect this pinned was that RadioSim carried
    performance disclaimers with no way to produce the evidence they asked for:
    an empty ``tests/performance/``, no benchmark package, and no task to run
    one. All three now exist (Sections 22, 23, 32.9), so the pin becomes its own
    inverse rather than being deleted: the assertions still name exactly the
    three surfaces D15 was about.

    ``tests/integration/test_hybrid_end_to_end.py`` was added by Tier 6F, which
    narrowed this test's integration-directory assertion at the time; the
    assertion is kept so a stray new file in either directory is still visible
    in a diff.  Tier 7D added ``test_jones_end_to_end.py``, the one integration
    file ``Tier7JonesSciencePlan.md`` Section 30 names.  Tier 8D added
    ``test_cli_end_to_end.py``, the CLI-to-artifact test
    ``Tier8ReleasePlan.md`` Section 17's 8D item 3 names -- the one thing this
    directory did not do, since both older files drive the Python API rather
    than the command line.  Each addition widens this list by exactly the file
    its own plan names, which is what makes an unplanned one visible.
    """
    performance = sorted(
        p.name for p in (REPO_ROOT / "tests" / "performance").glob("*.py")
    )
    integration = sorted(
        p.name for p in (REPO_ROOT / "tests" / "integration").glob("*.py")
    )
    assert performance == ["__init__.py", "test_backend_benchmarks.py"]
    assert integration == [
        "__init__.py",
        "test_cli_end_to_end.py",
        "test_hybrid_end_to_end.py",
        "test_jones_end_to_end.py",
    ]

    benchmarks = REPO_ROOT / "src" / "radiosim" / "benchmarks"
    assert benchmarks.is_dir()
    assert sorted(p.name for p in benchmarks.glob("*.py")) == [
        "__init__.py",
        "harness.py",
        "record.py",
    ]

    pixi_toml = _source("pixi.toml")
    assert 'bench = "python -m pytest tests/performance/ -m performance"' in pixi_toml

    # The benchmarks must not become a gate. Section 22.3: performance tests
    # never gate; CI continues to run only ``-m "not slow"``.
    performance_source = _source("tests/performance/test_backend_benchmarks.py")
    assert (
        "pytestmark = [pytest.mark.performance, pytest.mark.slow]" in performance_source
    )


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
#
# Re-keyed for the Tier 6J repair (2026-07-31): the ``osx-arm64`` rows are the
# values above, unchanged; the ``linux-64`` and ``osx-64`` rows were harvested
# from the CI logs of run ``30628921601`` (jobs ``91150529919``, ``91150529912``,
# ``91150529972``, ``91150529971``), which print the full measured digest in the
# pytest string diff.  They were *not* re-derived locally -- this repository has
# no x86_64 host -- so CI is their verification, and run ``30631837095``
# confirmed every one of them.
#
# Each value is now a recorded *observation*, and each cell holds the set of
# observations made in it (axis 3 in this module's docstring).  Provenance of the
# second observations, both on ``linux-64``/py312, both from run ``30640039816``
# job ``91187338402`` on an ``AMD EPYC 9V74``:
#
#   config.yaml                 ``b576167d...`` (first  ``94ed2fd1...``)
#   receptor_circular_example   ``de4ced01...`` (first  ``e5075437...``)
#
# The first observation in each cell was measured on runs ``30628921601`` (9V74)
# and ``30631837095`` (7763 for py312, 9V74 for py311), so the model string is
# known not to be the discriminator.  Adding a third observation to any cell
# requires deciding, on the evidence in the failure message, that it is a machine
# class and not a regression.
_SHIPPED_CONFIG_FINGERPRINTS: dict[str, dict[str, tuple[str, ...]]] = {
    "config.yaml": {
        "linux-64-py311": (
            "65a1b2b4248d8f479656a32682f2399162d58518b95e163c400b4eba55408a12",
        ),
        "linux-64-py312": (
            "94ed2fd18d5c23d31a1bf9bcabaefb4ebb4b213258cf000ada4297052783a4ca",
            "b576167d143bee69217e91f17f5371b4e7a1005bd1cec639e70cf8f32601ebef",
        ),
        "osx-64-py311": (
            "a984750776d29ee149f04a6c5815d0c99582a9cd1700240472f3b6d3ea2108db",
        ),
        "osx-64-py312": (
            "fccde411c77b7cd4e347689ce10fad0dd12c3e28e3b89ecb774aad779a0711da",
        ),
        "osx-arm64-py311": (
            "4bbb74035b3d700fa7638dca6b854a8c9110bc2abe8d418c7b180f527b947f2b",
        ),
        "osx-arm64-py312": (
            "9e4f4e164074ad7acf71a6c2c518b1d481a131054445b97e4b1b111be0838e28",
        ),
    },
    "receptor_circular_example.yaml": {
        "linux-64-py311": (
            "c257c96e4bee3eaea28e367590398c6fa20d9f71d1bf5534854569dd62e85ca0",
        ),
        "linux-64-py312": (
            "e50754376c095ecec9615016b78137067b286efc95b5cf6eb646e6d6e76bcede",
            "de4ced0186c9d3ec51c8df3883b857e82fd049da68b41a96f4938bfc366e7c92",
        ),
        "osx-64-py311": (
            "abac0e11c50bd9098c3b136dc86c6c5866d9aff6f2e1821c9210d769019b2d32",
        ),
        "osx-64-py312": (
            "768f1b2f7eac091451bca6f69e8b3a623955b28e76ad5134494b1cb65791f4a0",
        ),
        "osx-arm64-py311": (
            "be1e86fba57821a95f13f527a72b2ffd42edd4494cc68b0fde68d0f24d042203",
        ),
        "osx-arm64-py312": (
            "a1ea03d8cf5286149b07543736b3e4cdef90091f8464fc9a04b20f38a736ecab",
        ),
    },
}

#: The raw visibility-cube digests recorded above, asserted directly so the
#: "``C11`` moved the fingerprint but not the science" claim is a test, not a
#: comment.  Recipe: ``sha256`` of the C-contiguous ``complex128`` buffer.
#:
#: These two digests were the last x86_64 pins to be measured, and they took an
#: extra CI round to obtain: for the whole of Tier 6 the ``scientific_sha256``
#: assertion in the same test short-circuited ahead of them on all four x86_64
#: jobs, so the cube assertion below was never reached there and no
#: ``linux-64``/``osx-64`` value existed in any log to harvest.  Inventing them
#: was not an option.  ``_assert_pinned_digest`` was the harvest mechanism:
#: once the ``scientific_sha256`` pins above were correct on every platform, run
#: ``30631837095`` reached this assertion on each x86_64 job, found no entry for
#: its environment, and failed printing the digest it had just measured (jobs
#: ``91159779076``, ``91159778993``, ``91159779102``, ``91159779044``).  All
#: eight values are distinct, which is the same per-``(platform, python)``
#: structure every other pin family in this module shows.
#:
#: The ``linux-64``/py312 cells hold two observations.  The runner class that
#: produced the second shipped-config ``scientific_sha256`` observations also
#: produces different raw cubes -- the cube is what that digest is computed
#: over -- and, because ``_assert_pinned_digests`` no longer short-circuits,
#: run ``30646860127`` (job ``91210265306``, ``AMD EPYC 9V74``) surfaced both
#: cube values in a single failure, exactly as the previous revision of this
#: comment predicted.  Run ``30651948058`` (job ``91227058667``) then measured
#: the identical two values on an ``Intel(R) Xeon(R) Platinum 8370C``, so the
#: second observation in each cell is byte-stable across two runs and two CPU
#: vendors.  Both jobs dispatched the AVX-512 tiers (``AVX512_SKX`` through
#: ``AVX512_ICL``; the 9V74 additionally ``AVX512BF16``, immaterial for
#: ``complex128`` work) -- the strongest evidence yet that the axis-3
#: discriminator is the dispatched vector feature set, not the CPU model.
#: Adjudicated as a machine class, not a regression: both failing jobs passed
#: every within-process reproducibility test, and the scientific digests for
#: the same class were accepted on the same evidence (commit ``e5b20d1``).
_SHIPPED_CONFIG_CUBE_DIGESTS: dict[str, dict[str, tuple[str, ...]]] = {
    "config.yaml": {
        "linux-64-py311": (
            "9d770ec675b52d352aea6cf750cdba5056cc0517aad3d87b84ef5ed47e48997f",
        ),
        "linux-64-py312": (
            "f7df2b44c374b7ffc86d631ae33f0398538ff77ec5dfc4d80ed3f5266fe35f5d",
            "51c26634c3fec9242885f8ffbbb5a8cecd4aba4562203d4e23f21833c2cee12d",
        ),
        "osx-64-py311": (
            "5d147191625b3317cba05dfd330c04b0cdd0ff24ec6e3792935c7df31f8fcb75",
        ),
        "osx-64-py312": (
            "debe780775f8a101d942a4b1746e822dab44094d2ac393378208dd04ac160fa7",
        ),
        "osx-arm64-py311": (
            "cce1bfe86dc8b3fe81e5c6064a8449afa5bbab95866ec6bc352681dbf1e5ffae",
        ),
        "osx-arm64-py312": (
            "7560d2f267f372e19ef735afca0cb9ec05ca9f75e2f2ca62a35c52843660f9df",
        ),
    },
    "receptor_circular_example.yaml": {
        "linux-64-py311": (
            "1fb5cedd8635dc66a8e51000772b50fb8a4ec3305980f2c01dcb415f85f43f5b",
        ),
        "linux-64-py312": (
            "57c6a9dbe57c97c2a2b5307a3a530da5896b17f6393c77dc08a38fc4b4f48ce4",
            "9e95838cf6aca5fc219a07bb70f2f91ed4c33088a88de67abdbd618c38603ba3",
        ),
        "osx-64-py311": (
            "2bdc9994e53f3d89417f1d2d5c2ddd5cfc08b44d94e86feef90595e96130b389",
        ),
        "osx-64-py312": (
            "925c9b92d762e2d8919bb25356b01f85d33faa0ebe6e5608d6a373fc69ec6c15",
        ),
        "osx-arm64-py311": (
            "95890bc680c21057c5c23245dc8b67eb7e8662559b3d965905862148a75dd2f8",
        ),
        "osx-arm64-py312": (
            "ff26cb85289e77cda59a7508dae2e38afeb32bbfb4aff1b98315ac33e2c0177b",
        ),
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
    result = _run_shipped_config("config.yaml", tmp_path)
    assert result.visibilities.shape == (60, 15, 101, 4)
    assert str(result.visibilities.dtype) == "complex128"
    assert result.solver.sky_representation == "point_sources"
    assert result.solver.execution_path == "polarized"
    assert result.solver.components == ("point",)
    _assert_pinned_digests(
        (
            _SHIPPED_CONFIG_FINGERPRINTS["config.yaml"],
            "configs/config.yaml scientific_sha256",
            result.scientific_sha256,
        ),
        (
            _SHIPPED_CONFIG_CUBE_DIGESTS["config.yaml"],
            "configs/config.yaml raw cube sha256",
            _raw_cube_digest(result.visibilities),
            result.visibilities,
        ),
    )


def test_shipped_circular_receptor_config_scientific_fingerprint(tmp_path) -> None:
    """Records the R1 reference for ``configs/receptor_circular_example.yaml``."""
    result = _run_shipped_config("receptor_circular_example.yaml", tmp_path)
    assert result.visibilities.shape == (6, 15, 3, 4)
    assert str(result.visibilities.dtype) == "complex128"
    assert result.solver.sky_representation == "point_sources"
    assert result.solver.components == ("point",)
    _assert_pinned_digests(
        (
            _SHIPPED_CONFIG_FINGERPRINTS["receptor_circular_example.yaml"],
            "configs/receptor_circular_example.yaml scientific_sha256",
            result.scientific_sha256,
        ),
        (
            _SHIPPED_CONFIG_CUBE_DIGESTS["receptor_circular_example.yaml"],
            "configs/receptor_circular_example.yaml raw cube sha256",
            _raw_cube_digest(result.visibilities),
            result.visibilities,
        ),
    )


def test_shipped_realistic_foreground_config_cannot_run_at_this_gate(
    monkeypatch,
) -> None:
    """Records why R1 cannot cover the third shipped configuration.

    Originally this pinned *two* independent blockers, both named in this
    module's docstring: (1) the 12 MB Remazeilles/Haslam network download, and
    (2) the SKY-001 production defect that made every VizieR point-catalog
    loader raise ``TypeError`` before any network access.

    FLIPPED BY: the standalone SKY-001 remediation slice.  Blocker (2) is gone:
    the four wrapper call sites in
    ``core/sky/loaders/vizier/point_catalogs.py`` now pass ``precision`` by
    keyword, so ``load_gleam`` runs all the way to the VizieR fetch boundary.
    The assertion below is inverted accordingly -- with the fetch mocked, the
    wrapper returns a real :class:`SkyModel` instead of raising.

    Blocker (1) stands unchanged, so this test keeps its name and its purpose:
    the configuration still needs a network download and can therefore never be
    a hermetic R1 fingerprint.  The 6A docstring's instruction to replace this
    with a "network-marked" fingerprint is not actionable -- the repository has
    no ``network`` pytest marker and R1 requires hermeticity -- so the
    configuration stays outside R1 and this test records why.
    """
    from radiosim.core.sky.loaders.vizier import core as vizier_core
    from radiosim.core.sky.loaders.vizier.point_catalogs import load_gleam

    catalog = Table(
        {
            "RAJ2000": [180.0],
            "DEJ2000": [-30.0],
            "Fpwide": [4000.0],
            "alpha": [-0.8],
        }
    )
    monkeypatch.setattr(
        vizier_core,
        "_fetch_vizier_catalog",
        lambda **kwargs: catalog,
    )

    sky = load_gleam(
        flux_limit=1000.0,
        max_rows=1,
        precision=PrecisionConfig.standard(),
    )
    assert sky.n_point_sources == 1

    config = yaml.safe_load(
        (REPO_ROOT / "configs" / "realistic_foreground_example.yaml").read_text(
            encoding="utf-8"
        )
    )
    assert config["sky_model"]["sources"][0]["bright_catalogs"] == "gleam"
    assert config["visibility"]["sky_representation"] == "healpix_map"
    # Blocker (1): the diffuse layer this recipe composes is a network download.
    assert config["sky_model"]["sources"][0]["diffuse"] == "haslam"


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

# The ``linux-64`` and ``osx-64`` rows were harvested from the CI logs of run
# ``30628921601``.  Every one of them is byte-identical to the value the *same*
# platform measured at the 6A characterization commit (run ``30531414992``,
# which added no production code), which is the direct evidence that the platform
# spread predates the tier rather than being introduced by 6D/6F/6H -- see the
# "Reproducibility scope" note in this module's docstring.
#
# ``heterogeneous_receptor_bases`` on ``linux-64``/py312 carries a second
# observation, ``11d4c0a5...``, from run ``30640039816`` job ``91187338402``
# (``AMD EPYC 9V74``).  It is the only workload of the six that moved on that
# runner, which is why these pins are sets per digest rather than per cell.
_WORKLOAD_DIGESTS: dict[str, dict[str, tuple[str, ...]]] = {
    "healpix_polarized": {
        "linux-64-py311": (
            "c839098b725fc8cebd4ef2c93cc9b67aa59be8b7aea5446a1ee44b3c9fafbc94",
        ),
        "linux-64-py312": (
            "e7ba84d489507cec6ea43cf8425e52a0a4faf31575c419a46d312a829ddccad6",
        ),
        "osx-64-py311": (
            "8c5e5fd7a8b30881e7e6de833f15bfcf0583500610bdc077620093fa6b898cad",
        ),
        "osx-64-py312": (
            "f0e26ad0f436758c0c3b153578ce0242fbe8f98feab51fffd0497d7ffb53e6ef",
        ),
        "osx-arm64-py311": (
            "201feac2a5d1c8173528a24629d53a4fa51d19ef2eee9bdff667c3eda3c836a5",
        ),
        "osx-arm64-py312": (
            "72c006b63a70230c7827ef5a618859c1541070bbdabdaada5e4b7edd0c40b1b3",
        ),
    },
    "healpix_scalar": {
        "linux-64-py311": (
            "98dd5ca861a2970992a7adc047ac878cab51c6b674afcea7d4edd397409895bc",
        ),
        "linux-64-py312": (
            "5efc9724d30f28acfa6e6f193f2da8a733137d9e6d2c4c23ec5a069aea3f5fa3",
        ),
        "osx-64-py311": (
            "faa5b00af524da277d3ba66147c48aaf725677242a61177edec2e491849d28f0",
        ),
        "osx-64-py312": (
            "f915b97067b180a5d006c1c03d8ebe8c4bdc3dd561e3f3d684317f2252d03d04",
        ),
        "osx-arm64-py311": (
            "ed6356f91b7277ad3ad494f6b37b2d78110a7af58eef770fbf7d6729b3af3f7b",
        ),
        "osx-arm64-py312": (
            "4a701c82b6f7608569dba79d797a531dde5bda54e26ceddc61b7a22ad6d62344",
        ),
    },
    "heterogeneous_receptor_bases": {
        "linux-64-py311": (
            "1b7674f0c8c0b6561ea06929a55ecab797d609a150b31e8ad72bf6a88c7f3b7b",
        ),
        "linux-64-py312": (
            "73f340f1726163987eef8a387c7634a1e990264c8b23211918eea883749d54b7",
            "11d4c0a5afd60d1682d62e5d85dcd3cde7c45d8e6b29411e22ccc35425847c46",
        ),
        "osx-64-py311": (
            "afc26b47933ea3964416dae8ac6cb5d242e133f6068d91617c5ae493ffd97702",
        ),
        "osx-64-py312": (
            "7a730aa3e8e0e035c32847efe1cc7354439c9e2a8a267251c4c36f44629c7d74",
        ),
        "osx-arm64-py311": (
            "81055aff940d17817c66fb95ac760962af867ef4a9a3062b1e5bd80991803252",
        ),
        "osx-arm64-py312": (
            "d39cbe2fde4a3a54c518423ee4c7ee0db2b2664c5caabdf88dbd3d7c7979537d",
        ),
    },
    "point_gaussian_morphology": {
        "linux-64-py311": (
            "638a6efa57aedf732d76e251726e59055c4a8c92c6b74340f598b546239ac097",
        ),
        "linux-64-py312": (
            "0a52ca6ab8542a87928b31e029ee366d8e49baa7c18d0693ba10b9c3b2f512ea",
        ),
        "osx-64-py311": (
            "f88aefc2a3323d462f2b324533b64b6934f3ad2667fa0fcec5f0f2a432e5df9e",
        ),
        "osx-64-py312": (
            "019ec56ed275b4d27af64338192d4b2890dd2bab860fa4f44392f3cdd5f6f723",
        ),
        "osx-arm64-py311": (
            "9cd139554a45920f6338c4552544e2c490c8597bcd46f915a3f3855d867ae384",
        ),
        "osx-arm64-py312": (
            "370f7f353ec8ced7f09a8322b0867b6f8e7c2fc3ecf51f160ca8fc9d21939941",
        ),
    },
    "point_polarized_2times": {
        "linux-64-py311": (
            "4de5b348e06b3fd3e3fb457487fa8a21e7f10671d2c06ea56866f89b7f717f65",
        ),
        "linux-64-py312": (
            "46d011077d62cc26fdf44cb4c1d7a99e724b028181f527d699d8e7fb917c1230",
        ),
        "osx-64-py311": (
            "76705c96687157dc96048fd7ee607d1e978fe018018f6931123579340c446a69",
        ),
        "osx-64-py312": (
            "a86e1bea97af310480d62b94da733b0a5793cddbb37187e74fd2df31f0a2461d",
        ),
        "osx-arm64-py311": (
            "1140e5917a671af77233b3b244cc0bd7fb15c814a8f5fb70d22cd9c16cd5b9cd",
        ),
        "osx-arm64-py312": (
            "dabe4c4bc678276a98d03a266ae2e1a9ec39f949bd263ee4da15247bb83f7431",
        ),
    },
    "point_unpolarized_1time_2freq": {
        "linux-64-py311": (
            "3b4b340e461c1886e495e4592bb7bc299bd432b2e784a44378c263d9b5629311",
        ),
        "linux-64-py312": (
            "2af6aa20bfb49580b2e2f2a25d1fe8a3ad0e614b44d4a0b7df116faf53d46a55",
        ),
        "osx-64-py311": (
            "91b57e6405fff2c0b4206f0b976a2a91f5e67335e2c21118160c05692ee8c4f6",
        ),
        "osx-64-py312": (
            "90ab8f2d7b59229df1aebf5c2606b43e386156b5b7f44231682f7d964394bb38",
        ),
        "osx-arm64-py311": (
            "b4cc91e5852ef3ad5992c76a770950a68580da7ba73142b920cbcdc28d4f2510",
        ),
        "osx-arm64-py312": (
            "93cd8c728e387e0e0d24eee5101403b02f8fa44d8556f1644e4904e5feff2f14",
        ),
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
    cube = _WORKLOAD_RUNNERS[workload](tmp_path)
    array = np.asarray(cube)

    assert array.shape == expected_shape
    assert str(array.dtype) == "complex128"
    # A digest of an all-zero cube would pin nothing.
    assert float(np.max(np.abs(array))) > 0.0
    _assert_pinned_digests(
        (
            _WORKLOAD_DIGESTS[workload],
            f"Section 13.4 workload {workload!r}",
            _cube_digest(array),
            array,
        ),
    )


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
