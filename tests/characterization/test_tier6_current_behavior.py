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
Tier 6 fingerprint may be treated as a portable constant in documentation.
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
from radiosim.backends.numba_backend import NumbaBackend
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
from tests.fixtures.configs import valid_config_mapping

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
    """A NumPy backend that counts ``set_at`` calls, for the D11 accumulation pin."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.set_at_calls = 0

    def set_at(self, arr: Any, index: Any, value: Any) -> Any:
        self.set_at_calls += 1
        return super().set_at(arr, index, value)


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


def test_sky_representation_admits_only_two_literals() -> None:
    """Pins D1: the high-level API cannot express a hybrid sky.

    OWNED BY: Tier 6F, which adds the ``hybrid`` literal and
    ``allow_lossy_point_rasterization``.
    """
    field = VisibilityConfig.model_fields["sky_representation"]
    assert field.annotation is not None
    literals = set(getattr(field.annotation, "__args__", ()))
    assert literals == {"point_sources", "healpix_map"}
    assert "hybrid" not in literals
    assert "allow_lossy_point_rasterization" not in VisibilityConfig.model_fields


def test_exactly_one_solver_runs_per_run_call(tmp_path, monkeypatch) -> None:
    """Pins D2: ``run()`` dispatches to one solver, so no summation is possible.

    OWNED BY: Tier 6F, which introduces ``V_total = V_point + V_healpix``.
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
            },
        )
        Simulator.from_mapping(data, base_dir=tmp_path).run(progress=False)
        assert calls == expected


def test_hybrid_model_under_point_representation_is_silently_discarded(
    tmp_path,
) -> None:
    """Pins D3: a surviving HEALPix payload contributes nothing, silently.

    A hybrid model whose HEALPix payload is inflated to an absurd brightness
    produces visibilities that are *bit-identical* to the point-only run, with no
    error and no warning.

    OWNED BY: Tier 6F, after which the same model must either sum both payloads
    or be rejected by the Section 18.3 message.
    """
    data = valid_config_mapping(tmp_path)

    point_only = Simulator.from_mapping(data, base_dir=tmp_path)
    point_only.setup()
    assert point_only._sky_model.healpix is None
    baseline = np.asarray(point_only.run(progress=False).visibilities)

    hybrid_run = Simulator.from_mapping(data, base_dir=tmp_path)
    hybrid_run.setup()
    frequencies = np.asarray(hybrid_run._frequencies_hz, dtype=np.float64)
    hybrid_sky = materialize_healpix_model(
        hybrid_run._sky_model,
        nside=8,
        frequencies=frequencies,
        ref_frequency=float(frequencies[0]),
        clear_other=False,
    )
    inflated = np.full_like(hybrid_sky.healpix.maps, 1.0e6)
    hybrid_sky = hybrid_sky.replace(healpix=hybrid_sky.healpix.replace(maps=inflated))
    assert hybrid_sky.formats == {SkyFormat.POINT_SOURCES, SkyFormat.HEALPIX}

    hybrid_run._sky_model = hybrid_sky
    hybrid_run._source_arrays = hybrid_sky.as_point_source_arrays()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        hybrid = np.asarray(hybrid_run.run(progress=False).visibilities)

    assert np.array_equal(hybrid, baseline)
    assert [
        str(entry.message)
        for entry in caught
        if "discard" in str(entry.message).lower()
        or "ignored" in str(entry.message).lower()
    ] == []


def test_setup_keeps_exactly_one_payload_per_representation(tmp_path) -> None:
    """Pins D3's setup-side fork at ``api/simulator.py`` steps 6.

    OWNED BY: Tier 6F.
    """
    healpix_data = valid_config_mapping(
        tmp_path,
        visibility={
            "calculation_type": "direct_sum",
            "sky_representation": "healpix_map",
        },
    )
    healpix_sim = Simulator.from_mapping(healpix_data, base_dir=tmp_path)
    healpix_sim.setup()
    assert healpix_sim._source_arrays is None

    point_sim = Simulator.from_mapping(
        valid_config_mapping(tmp_path), base_dir=tmp_path
    )
    point_sim.setup()
    assert point_sim._source_arrays is not None


def test_point_target_combine_silently_drops_a_hybrid_contributor_maps() -> None:
    """Pins D4: a hybrid contributor loses its maps with no diagnostic.

    OWNED BY: Tier 6F.
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

    import warnings

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


def test_healpix_target_combine_rasterizes_a_point_contributor() -> None:
    """Pins D5's rasterization half: point flux is folded into the map cube.

    OWNED BY: Tier 6F, which makes this opt-in via
    ``allow_lossy_point_rasterization``.
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


def test_point_target_combine_rejects_a_healpix_only_contributor() -> None:
    """Pins D5's hard-error half.

    OWNED BY: Tier 6F.
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

    with pytest.raises(ValueError, match="allow_lossy_point_materialization=True"):
        _combine_models(
            [point_only, healpix_only],
            representation=SkyFormat.POINT_SOURCES,
            precision=precision,
            mixed_model_policy="allow",
        )


def test_memory_estimate_counts_only_point_sources(tmp_path) -> None:
    """Pins D19: the estimate under-reports for HEALPix and would for hybrid.

    OWNED BY: Tier 6F.
    """
    data = valid_config_mapping(
        tmp_path,
        visibility={
            "calculation_type": "direct_sum",
            "sky_representation": "healpix_map",
        },
    )
    simulator = Simulator.from_mapping(data, base_dir=tmp_path)
    simulator.setup()
    assert simulator._source_arrays is None
    assert simulator._sky_model.healpix is not None

    source = inspect.getsource(Simulator.get_memory_estimate)
    assert 'len(self._source_arrays["ra_rad"]) if self._source_arrays else 0' in source
    estimate = simulator.get_memory_estimate()
    assert isinstance(estimate, dict)


# =========================================================================
# D6, D7 -- worker policy is hard-coded and unexpressible
# =========================================================================


def test_sky_loading_hard_codes_eight_workers() -> None:
    """Pins D6: the only loader call site passes a literal ``max_workers=8``.

    OWNED BY: Tier 6C, which removes the literal, and Tier 6B, which introduces
    the typed policy that replaces it.
    """
    simulator_source = _source("src/radiosim/api/simulator.py")
    assert "max_workers=8," in simulator_source

    signature = inspect.signature(load_models_parallel)
    assert signature.parameters["max_workers"].default == 8


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


def test_execution_config_backend_literal_still_offers_numba() -> None:
    """Pins the un-renamed backend literal, split out of the 6A worker pin.

    OWNED BY: Tier 6H, which renames the backend and changes this literal to
    ``dask`` with its Section 18.3 rejection message.  The 6A pin asserted this
    together with the worker-field set; the two halves were separated when the
    literal change moved from 6B to 6H (plan Sections 32.2, 32.8, 33).
    """
    backend_literals = set(
        getattr(ExecutionConfig.model_fields["backend"].annotation, "__args__", ())
    )
    assert backend_literals == {"auto", "numpy", "jax", "numba"}


def test_run_still_advertises_and_then_rejects_n_workers(tmp_path) -> None:
    """Pins D7: the public parameter exists, is documented, and always raises.

    OWNED BY: Tier 6E, which deletes the parameter outright.
    """
    signature = inspect.signature(Simulator.run)
    assert list(signature.parameters) == ["self", "progress", "n_workers"]
    assert signature.parameters["n_workers"].default is None
    docstring = Simulator.run.__doc__ or ""
    assert "Number of parallel workers (default: auto)" in docstring

    simulator = Simulator.from_mapping(
        valid_config_mapping(tmp_path), base_dir=tmp_path
    )
    with pytest.raises(NotImplementedError, match="Target remediation: Tier 6"):
        simulator.run(n_workers=1)


def test_no_worker_value_is_recorded_in_provenance(tmp_path) -> None:
    """Pins D6: no resolved worker count reaches the bounded result snapshot.

    OWNED BY: Tier 6C and Tier 6E.

    Scope note added by Tier 6B: the resolved worker policy now *does* reach
    ``SimulationResult.resolved_config`` and therefore ``provenance_sha256``,
    the HDF5 ``resolved_config_json`` and the summary JSON
    (``tests/unit/test_simulator/test_worker_policy.py``).  What this pin still
    records is the narrower fact 6C and 6E own: ``to_summary_snapshot()`` is a
    bounded metadata view that embeds no resolved configuration, so no
    *executed* worker count -- no loader execution record, no per-run solver
    thread count -- is reported there yet.
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


def test_get_backend_auto_returns_a_numba_backend_whose_xp_is_numpy() -> None:
    """Pins D9: ``auto`` misreports the executing implementation.

    OWNED BY: Tier 6H, which corrects the precedence so ``auto`` returns the
    NumPy backend when no non-CPU JAX device exists.
    """
    backend = get_backend("auto")
    assert isinstance(backend, NumbaBackend)
    assert backend.name == "numba-cpu"
    assert backend.xp is np


def test_no_numba_kernel_decorator_exists_in_the_package() -> None:
    """Pins D8: the ``numba`` backend compiles nothing of its own.

    OWNED BY: Tier 6H, which adds exactly one compiled kernel behind
    ``supports_compilation``.
    """
    pattern = re.compile(r"@(njit|jit|vectorize|guvectorize|cuda\.jit)\b")
    offenders = [
        str(path.relative_to(REPO_ROOT))
        for path in sorted((REPO_ROOT / "src" / "radiosim").rglob("*.py"))
        if pattern.search(path.read_text(encoding="utf-8"))
    ]
    assert offenders == []

    numba_source = _source("src/radiosim/backends/numba_backend.py")
    assert "from numba import jit, prange" in numba_source
    assert "prange(" not in numba_source  # imported, advertised, never called


def test_numba_backend_docstring_claims_jit_and_parallel_loops() -> None:
    """Pins D8's documentation half.

    OWNED BY: Tier 6H, which renames the class to ``DaskBackend`` and deletes
    the claim.
    """
    docstring = NumbaBackend.__doc__ or ""
    assert "'cpu': Local CPU with JIT and parallel loops" in docstring


def test_rime_simulator_reports_unconditional_gpu_support() -> None:
    """Pins D10.

    OWNED BY: Tier 6H, which makes ``supports_gpu`` ``False``.
    """
    simulator = RIMESimulator()
    assert simulator.supports_gpu is True
    source = inspect.getsource(type(simulator).supports_gpu.fget)
    assert "return True" in source


def test_rime_simulator_docstring_advertises_the_pre_tier5_chain_order() -> None:
    """Pins D20.

    OWNED BY: Tier 6H (documentation truth).
    """
    docstring = RIMESimulator.__doc__ or ""
    assert "J = B @ G @ D @ P @ E @ T @ Z @ K" in docstring


def test_point_solver_accumulates_one_set_at_per_time_baseline_frequency(
    tmp_path,
) -> None:
    """Pins D11 for the point solver.

    OWNED BY: Tier 6D, which replaces per-cell ``set_at`` with one per-time block
    assembly and asserts a single whole-cube assembly per call (R2).
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
    assert backend.set_at_calls == n_times * n_baselines * n_freqs
    assert backend.set_at_calls > 1


def test_healpix_solver_accumulates_one_set_at_per_time_baseline_frequency(
    tmp_path,
) -> None:
    """Pins D11 for the HEALPix solver.

    OWNED BY: Tier 6D.
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
    assert backend.set_at_calls == n_times * n_baselines * n_freqs
    assert backend.set_at_calls > 1


def test_healpix_solver_rebuilds_the_constant_receptor_transforms_per_time() -> None:
    """Pins D12: ``H_p @ C_p`` is recomputed inside the time loop.

    OWNED BY: Tier 6D, which hoists it above the loop.
    """
    source = _source("src/radiosim/core/visibility_healpix.py")
    time_loop = source.index("for time_idx in range(n_times):")
    transforms = source.index("receptor_transforms = _receptor_transforms(")
    frequency_loop = source.index("for freq_idx, freq in enumerate(frequencies):")
    assert time_loop < transforms < frequency_loop


def test_backend_abstract_surface_omits_jit_vmap_and_jit_compile() -> None:
    """Pins D14: compilation is backend-private and uncallable generically.

    OWNED BY: Tier 6H, which adds ``supports_compilation`` and ``compile``.
    """
    for attribute in ("jit", "vmap", "jit_compile", "compile", "supports_compilation"):
        assert not hasattr(ArrayBackend, attribute), attribute
    assert hasattr(NumbaBackend, "jit_compile")
    jax_source = _source("src/radiosim/backends/jax_backend.py")
    assert "    def jit(self, func):" in jax_source
    assert "    def vmap(" in jax_source


def test_nothing_in_the_package_calls_jit_vmap_or_jit_compile() -> None:
    """Pins D14's second half: the compilation helpers have no callers.

    OWNED BY: Tier 6H.
    """
    callers: list[str] = []
    pattern = re.compile(r"\.(jit|vmap|jit_compile)\s*\(")
    for path in sorted((REPO_ROOT / "src" / "radiosim").rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        for match in pattern.finditer(text):
            line = text[: match.start()].count("\n") + 1
            if path.name in {"jax_backend.py", "numba_backend.py"}:
                continue
            callers.append(f"{path.relative_to(REPO_ROOT)}:{line}")
    assert callers == []


def test_jax_synchronize_blocks_on_a_throwaway_constant() -> None:
    """Pins D13 by source text, since JAX is not installed at this gate.

    OWNED BY: Tier 6H, which changes the signature to ``synchronize(arr=None)``
    and blocks on the caller's array.
    """
    source = _source("src/radiosim/backends/jax_backend.py")
    assert "    def synchronize(self) -> None:" in source
    assert "jax.block_until_ready(jnp.array(0))" in source


def test_jax_is_not_a_dependency_of_any_pixi_environment() -> None:
    """Pins D16: the mandated NumPy/JAX parity evidence needs a dependency change.

    See the Q1 evidence record in this module's docstring for the exact versions
    that resolve on all three locked platforms.

    OWNED BY: Tier 6H, which adds the jax-cpu feature and environment.
    """
    pixi_toml = _source("pixi.toml")
    assert "jax" not in pixi_toml
    assert 'numpy = ">=1.24,<2.5"' in pixi_toml
    assert "[feature.py311.dependencies]" in pixi_toml
    assert "[feature.py312.dependencies]" in pixi_toml


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
    assert integration == ["__init__.py"]
    assert not (REPO_ROOT / "src" / "radiosim" / "benchmarks").exists()
    assert "bench" not in _source("pixi.toml")


# =========================================================================
# D17, D18 -- offline policy and executor degradation
# =========================================================================


def test_forced_offline_status_does_not_populate_the_module_cache(
    monkeypatch,
) -> None:
    """Pins D17: ``execution.offline`` never reaches loader enforcement.

    OWNED BY: Tier 6C, which adds ``set_offline_policy`` and installs it in
    workers.
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
    assert network_module.is_online() is True
    assert probed  # a live probe happened despite the forced-offline status


def test_require_service_consults_is_online_not_a_resolved_policy(
    monkeypatch,
) -> None:
    """Pins D17's enforcement path.

    OWNED BY: Tier 6C.
    """
    source = inspect.getsource(network_module.require_service)
    assert "if not is_online():" in source
    assert "offline" not in inspect.signature(network_module.require_service).parameters

    monkeypatch.setattr(network_module, "_cached_status", None)
    monkeypatch.setattr(network_module, "is_online", lambda *a, **k: False)
    with pytest.raises(ConnectionError, match="No internet connection"):
        network_module.require_service("vizier", "download catalog 'gleam'")


def test_process_executor_degrades_to_threads_with_only_a_log_warning(
    caplog, monkeypatch
) -> None:
    """Pins D18: the degradation succeeds silently and is recorded nowhere.

    The pickle probe is forced to fail so the ``executor="process"`` request
    takes the fallback branch.  The load still succeeds, the only trace is a
    ``logger.warning``, and the returned value carries no record of either the
    request or the degradation.

    OWNED BY: Tier 6C, which adds ``LoaderExecutionRecord`` and an explicit
    rejection for an explicit process request.
    """
    parallel_module = importlib.import_module("radiosim.core.sky.operations.parallel")
    monkeypatch.setattr(
        parallel_module, "_kwargs_picklable", lambda *args, **kwargs: False
    )

    loaders = [
        ("test_sources", {"num_sources": 1, "distribution": "uniform", "seed": 1})
    ]
    with caplog.at_level("WARNING", logger=parallel_module.__name__):
        models = load_models_parallel(
            loaders,
            max_workers=2,
            precision=PrecisionConfig.standard(),
            strict=True,
            executor="process",
        )

    assert len(models) == 1
    assert any(
        "Falling back to thread pool" in record.message for record in caplog.records
    )
    assert "LoaderExecutionRecord" not in _source(
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


_SHIPPED_CONFIG_FINGERPRINTS: dict[str, dict[str, str]] = {
    "config.yaml": {
        "py311": "302deb27aebed7fd9db23a51bf8e3ad038258de3b4752021d823c86e6ba8e685",
        "py312": "161fc98c4d6a58303d31100648a2f5ec4794ed4307a32542752cb04bf31cb82e",
    },
    "receptor_circular_example.yaml": {
        "py311": "b3c1a93e7a6910593292871b0945bc2981a7250c8a171a1600baf1c495e988bf",
        "py312": "e670c35f60d0c3094271a3e50d0ee8fc7020802ef02b4e3aa5e1a33f586a93cd",
    },
}


def test_shipped_default_config_scientific_fingerprint(tmp_path) -> None:
    """Records the R1 reference for ``configs/config.yaml``.

    The Tier 6D restructure must reproduce this digest bit-for-bit *in the same
    environment*.  A change here is a scientific change and must be justified,
    never re-pinned silently.
    """
    expected = _expected_for_environment(
        _SHIPPED_CONFIG_FINGERPRINTS["config.yaml"], "configs/config.yaml"
    )
    result = _run_shipped_config("config.yaml", tmp_path)
    assert result.visibilities.shape == (60, 15, 101, 4)
    assert str(result.visibilities.dtype) == "complex128"
    assert result.solver.sky_representation == "point_sources"
    assert result.solver.execution_path == "polarized"
    assert result.scientific_sha256 == expected


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
    assert result.scientific_sha256 == expected


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


def test_solver_seconds_is_a_single_uncomponentized_measurement(tmp_path) -> None:
    """Pins the absence of per-component timing.

    OWNED BY: Tier 6F, which adds ``solver_point_seconds`` and
    ``solver_healpix_seconds``.
    """
    result = Simulator.from_mapping(
        valid_config_mapping(tmp_path), base_dir=tmp_path
    ).run(progress=False)
    snapshot = result.to_summary_snapshot()
    flattened = repr(snapshot)
    assert "solver_point_seconds" not in flattened
    assert "solver_healpix_seconds" not in flattened
    assert isinstance(time.perf_counter(), float)
