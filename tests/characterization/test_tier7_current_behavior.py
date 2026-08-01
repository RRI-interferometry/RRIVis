"""Characterize the Tier 7 Jones surface, solver integration, and config baseline.

Every test in this module pins behavior that exists on ``main`` **today**, before
any Tier 7 production change.  Each test docstring names the slice that owns the
deliberate flip (``OWNED BY: Tier 7x``); a later slice must update the named test
in the same commit that changes the behavior.  A test with no ``OWNED BY`` line
pins behavior Tier 7 preserves.

The gate commit is ``997aba5`` ("docs(jones): accept Tier 7 design").  Its
production tree is identical to ``ac4fe41``: ``git diff ac4fe41 997aba5`` touches
only ``Fix.md`` and ``Tier7JonesSciencePlan.md``.  Every source-line reference in
``Tier7JonesSciencePlan.md`` Sections 5 to 7 was therefore taken at the same tree
these tests measure.

Tier 7A evidence record
=======================

Slice 7A is the evidence gate for ``Tier7JonesSciencePlan.md`` Section 41
questions Q1 and Q2.  Section 34 grants 7A exactly one production-adjacent
writable file -- this module -- so the recorded evidence lives here, following
the Tier 5A and Tier 6A precedent.

Q1 -- does a cross-validation reference resolve against the locked
``pyuvdata ==3.2.1``?  (blocks the Tier-2 half of Section 29; 7J acts on it)
**Yes.  ``pyuvsim 1.4.0`` resolves, installs, and imports against
``pyuvdata ==3.2.1`` on all three locked platforms and both locked Python
versions.**  Measured 2026-08-01 on ``osx-arm64`` (macOS 26.5.2, Apple M1 Max),
pixi 0.75.0, with ``pixi.toml`` and ``pixi.lock`` untouched: throwaway pixi
workspaces outside the repository, one of which replicated the repository's
entire ``[dependencies]`` and ``[pypi-dependencies]`` list plus the ``jax-cpu``
feature and added ``pyuvsim`` in a separate ``crossval`` feature.  ``pixi lock``
resolved every environment:

============  ===========  ======  ==========  =======  ==========  ==========
Environment   Platform     Python  pyuvdata    pyuvsim  pyradiosky  astropy
============  ===========  ======  ==========  =======  ==========  ==========
crossval      linux-64     3.11    3.2.1       1.4.0    1.1.0       8.0.1
crossval      osx-64       3.11    3.2.1       1.4.0    1.1.0       8.0.1
crossval      osx-arm64    3.11    3.2.1       1.4.0    1.1.0       8.0.1
crossval312   linux-64     3.12    3.2.1       1.4.0    1.1.0       8.0.1
crossval312   osx-64       3.12    3.2.1       1.4.0    1.1.0       8.0.1
crossval312   osx-arm64    3.12    3.2.1       1.4.0    1.1.0       8.0.1
============  ===========  ======  ==========  =======  ==========  ==========

Three facts qualify that answer, and 7J must carry all three:

1. **``pyuvsim`` is not a conda-forge package.**  A first throwaway workspace
   that declared ``pyuvsim`` as a *conda* dependency failed with "No candidates
   were found for pyuvsim \\*" on every platform, and
   ``https://api.anaconda.org/package/conda-forge/pyuvsim`` (retrieved
   2026-08-01) returns HTTP 404.  The optional feature Section 34 authorizes for
   7J must therefore declare ``pyuvsim`` under ``[feature.crossval.
   pypi-dependencies]``, not under conda dependencies.  The wheel is
   ``pyuvsim-1.4.0-py3-none-any.whl``, pure Python, so no platform has a wheel
   gap -- this is the opposite of the Tier 6A ``jaxlib`` situation, where
   conda-forge was load-bearing and PyPI had no ``osx-64`` wheel.
2. **The version is pinned down by ``pyuvdata`` and it is not the latest.**
   ``https://pypi.org/pypi/pyuvsim/json`` (retrieved 2026-08-01) reports latest
   ``pyuvsim 1.4.2`` requiring ``pyuvdata>=3.2.3``, which the repository's
   ``==3.2.1`` pin excludes.  The resolver selected ``1.4.0``, whose metadata
   requires ``pyuvdata>=3.1.0``, ``pyradiosky>=1.0.1``, ``astropy>=6.0``,
   ``numpy>=1.23``, ``scipy>=1.8``, ``psutil`` and ``python_requires >=3.10``.
   A cross-validation artifact must record ``pyuvsim 1.4.0``, never "latest".
3. **Resolution was verified by installation and import, not only by locking.**
   The ``crossval`` and ``crossval312`` environments were realized and
   ``import pyuvsim`` succeeded on both, reporting ``pyuvsim 1.4.0`` against
   ``pyuvdata 3.2.1``/``pyradiosky 1.1.0``/``numpy 2.4.6``/``astropy 8.0.1``
   under Python 3.11.15 and 3.12.13, with ``pyuvsim.simsetup.SkyModelData``,
   ``pyuvsim.uvsim.UVTask``, ``pyuvsim.uvsim.run_uvdata_uvsim``,
   ``pyuvsim.UVEngine`` and ``pyuvsim.initialize_uvdata_from_params`` all
   present.  The feature is also **additive to the byte**: diffing the resolved
   package list of ``crossval`` against ``default`` and of ``crossval312``
   against ``py312``, on all three platforms, the only difference in any of the
   six comparisons is the single added ``pyuvsim-1.4.0-py3-none-any.whl``.  No
   package is removed and no version moves -- ``psutil`` and ``pyradiosky`` are
   already inside the repository's resolved closure.  So 7J's optional feature
   cannot perturb the gating environments.

So Section 41's "if a version resolves" branch is live: 7J may add the optional
``crossval`` pixi feature and the marked, non-gating comparison test, and the
"recorded manual run" fallback is **not** needed.  Section 29.2's forbidden
claims stay forbidden regardless -- a resolvable reference is not a validated
one.

Q2 -- what is the host-memory cost of direction-batched DDE evaluation on the
largest shipped HEALPix configuration?  (blocks 7B's acceptance framing)
**Measured peak 690,207,014 bytes (658.24 MiB); the projected worst case with
``P``, ``Z`` and ``T`` enabled is 784,584,614 bytes (748.24 MiB), a factor of
1.137.  That is far below Section 41's factor-of-two trigger, so 7B does NOT
need a chunked HEALPix direction batch.**

Method, measured 2026-08-01 on ``osx-arm64``/py311 inside the repository's own
``default`` environment: ``configs/realistic_foreground_example.yaml`` was run
through ``Simulator.from_mapping(...).setup()`` and ``run(progress=False)``
with output redirected to a scratch directory and artifact writing disabled,
with ``tracemalloc`` started immediately before ``run()`` and stopped
immediately after, so the peak covers the solver and excludes sky loading.  The
run needs the network (see
``test_shipped_realistic_foreground_config_is_not_hermetic`` below) and is
therefore recorded here rather than asserted as a test.

Resolved shape of that configuration:

* ``nside = 128``, dense (not sparse), ``npix = 196608``, ``maps`` of shape
  ``(11, 196608)`` float64 = 8,650,752 bytes;
* 5 antennas, 15 baselines, 11 frequency channels, 10 time samples;
* published cube ``(10, 15, 11, 4)`` complex128 = 105,600 bytes;
* pixels above the horizon per time step, instrumented at
  ``visibility_healpix._host_preprocess_time_step``: 98308, 98306, 98310,
  98309, 98308, 98307, 98309, 98310, 98306, 98306 -- so the **direction batch
  is 98,310 at its largest**, not the 196,608 pixel count, because the horizon
  mask halves it before any Jones evaluation.

Arithmetic, using Section 41's own unit (``64 * n_dir`` bytes per antenna per
``(time, frequency)`` step):

* one ``(n_dir, 2, 2)`` complex128 batch = ``64 * 98310`` = 6,291,840 bytes
  (6.00 MiB);
* the three DDE terms Tier 7 adds (``P``, ``Z``, ``T``) held simultaneously for
  all five antennas -- the worst case the contract can produce, and strictly
  worse than the real one, because ``visibility.py``'s antenna loop is
  sequential and each term's batch is a temporary consumed by the next
  ``matmul`` -- cost ``3 * 5 * 6,291,840`` = 94,377,600 bytes (90.00 MiB);
* projected peak ``690,207,014 + 94,377,600`` = 784,584,614 bytes, i.e.
  **1.137x** the measured peak.  The realistic figure, one antenna's term batch
  plus its running product, is ``2 * 6,291,840`` = 12,583,680 bytes, i.e.
  1.018x.

For reference, the existing per-``(time, frequency)`` direction-batch working
set at this gate is already ~239 MiB of the 658 MiB peak: the per-antenna beam
Jones cache (5 x 6.00 MiB), the stacked ``J_p``/``J_q`` kernel inputs
(2 x 15 x 6.00 MiB), the coherency batch (6.00 MiB) and the phase array
(15 x 98310 x 16 bytes = 22.50 MiB).  A DDE term is small next to the
baseline-batched kernel inputs, which is the substantive finding: **the
batch-size risk in the HEALPix path is the ``(B, n_dir, 2, 2)`` stack, which
already exists and which Tier 7 does not change**, not the per-antenna
``(n_dir, 2, 2)`` term output that Q2 asked about.  Crossing the factor-of-two
line at this configuration would take about 22 simultaneous full-array DDE
terms; Tier 7 adds three.  Section 41's DIE ``(1, 2, 2)`` contract (I3) keeps
``G``, ``B``, ``D``, ``X``, ``Kd`` and ``Rc`` out of this budget entirely.

Baseline fingerprint scope
==========================

Section 33.2 asks 7A to record the cube digests and ``scientific_sha256`` for
"all four shipped configs".  What is actually achievable at this gate, and why,
differs per configuration:

* ``configs/config.yaml`` and ``configs/receptor_circular_example.yaml`` are
  pinned in **all six** ``(platform, python)`` environments by
  ``test_tier6_current_behavior``'s ``_SHIPPED_CONFIG_FINGERPRINTS`` and
  ``_SHIPPED_CONFIG_CUBE_DIGESTS``, whose x86_64 rows were harvested from CI.
  This module asserts those same tables rather than copying the 24 digests into
  a second maintenance site, so a Tier 7 restructure that moves a number fails a
  Tier-7-owned test.
* ``configs/hybrid_sky_example.yaml`` has **no** pin in any environment, and
  7A cannot create one that is green everywhere: every digest here is a
  function of ``(platform, python)`` and of the individual runner (see the
  "Reproducibility scope" note in ``test_tier6_current_behavior``), this
  repository has no x86_64 host, and inventing the four missing cells is not an
  option.  Rather than land a table that is knowingly red on four of six CI
  jobs, this module pins the shipped hybrid configuration with an
  **environment-independent bit-level** invariant instead: its cube is exactly
  the sum of the point-only and HEALPix-only cubes of the same configuration.
  That is a stronger 7B gate than an absolute digest on one machine class,
  because it holds on every runner.  For the record, the values measured on
  ``osx-arm64``/py311 at this commit, published cube shape ``(5, 15, 4, 4)``
  complex128, are ``scientific_sha256``
  ``65777deecea484de327d4f524db6ee8fda1751749890bb047f0781ec0ec3808a`` and raw
  cube ``sha256``
  ``bdd866b1936949a18bb1705ae8111a65a7b0e8e86a9eea7b641f8eccd58d281a``, with
  the point-only and HEALPix-only controls at
  ``4b41ff36b1f12da26af9611b0c797ae9eef81a6b08c024f91a20f1e7198a1b3a`` and
  ``4a3669f65b1236b1d2d1ee25078fac003bf5fd6b73cdcc4677c6bce3b229afae`` and a
  hybrid-minus-sum maximum absolute deviation of exactly 0.0.  Whether to spend
  a CI round harvesting the four missing x86_64 cells into an absolute table is
  the acceptance reviewer's call, not 7A's.
* ``configs/realistic_foreground_example.yaml`` cannot be a hermetic test at
  all: it downloads the Remazeilles/Haslam 408 MHz map and queries VizieR for
  GLEAM.  Tier 6A recorded the same conclusion.  It is pinned here only by the
  source facts that make it non-hermetic, plus the Q2 measurement above, which
  is the whole reason 7A ran it.

Nothing in this module re-pins a digest that Tier 6 already owns; the two
delegating tests call Tier 6's tables so that Tier 7 has its own failure site.
"""

from __future__ import annotations

import inspect
import re
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml

from radiosim.backends import get_backend
from radiosim.core.jones import chain as jones_chain
from radiosim.core.jones.base import JonesTerm
from radiosim.core.jones.baseline_errors import JonesBaselineTerm
from radiosim.core.jones.chain import JonesChain
from radiosim.core.precision import JonesPrecision, PrecisionConfig
from radiosim.core.visibility import calculate_visibility
from radiosim.core.visibility_healpix import calculate_visibility_healpix
from radiosim.io.config import (
    RadioSimConfig,
    VisibilityConfig,
    collect_unsupported_issues,
    load_config,
)
from tests.characterization.test_tier6_current_behavior import (
    _SHIPPED_CONFIG_CUBE_DIGESTS,
    _SHIPPED_CONFIG_FINGERPRINTS,
    _WORKLOAD_FREQS,
    WORKLOAD_LOCATION,
    WORKLOAD_TIME_GRID,
    _assert_pinned_digests,
    _raw_cube_digest,
    _run_shipped_config,
    _shipped_config_mapping,
    _solver_components,
    _workload_point_sources,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = REPO_ROOT / "src" / "radiosim"
JONES_ROOT = SOURCE_ROOT / "core" / "jones"

IDENTITY = np.eye(2, dtype=np.complex128)


def _source(relative_path: str) -> str:
    """Return the text of a repository file, for source-truth pins."""
    return (REPO_ROOT / relative_path).read_text(encoding="utf-8")


# =========================================================================
# Section 5.1 -- the exported Jones surface
# =========================================================================

#: The exact ``__all__`` of ``radiosim.core.jones``, in file order.
#: At the gate this was 43 names: three base names plus 40 concrete term
#: classes, with the ``CLAUDE.md`` "46" claim recorded as defect D0.
#:
#: PARTLY FLIPPED BY: Tier 7B, which removed ``GeometricPhaseJones`` -- K is
#: per-*baseline* and cannot be a chain term -- and added ``geometric_phase()``
#: plus the two names the batched contract introduces, taking the count to 45.
#:
#: FLIPPED BY: Tier 7C, which executed Section 23's removal ledger: 26 classes
#: deleted and ``CrosshandPhaseJones`` renamed ``CrosshandJones``.  What remains
#: is Section 9.1's public surface exactly -- 13 concrete terms, the three base
#: names, and the three non-class exports of 7B.
EXPORTED_JONES_NAMES: tuple[str, ...] = (
    "JonesTerm",
    "JonesChain",
    "JonesBaselineTerm",
    "DirectionBatch",
    "evaluate_antenna_jones",
    "geometric_phase",
    "GainJones",
    "BandpassJones",
    "PolarizationLeakageJones",
    "ParallacticAngleJones",
    "IonosphereJones",
    "TroposphereJones",
    "ReceptorConfigJones",
    "BasisTransformJones",
    "DelayJones",
    "CableReflectionJones",
    "CrosshandJones",
    "BaselineMultiplicativeJones",
    "SmearingFactorJones",
)

#: The exported names that carry real physics.  ``E`` is deliberately absent:
#: the solver's beam term is the private ``_ResolvedBeamJones`` adapter, not an
#: exported class.
#:
#: FLIPPED BY: Tier 7B, which turned the K class into ``geometric_phase()``.
REAL_PHYSICS_EXPORTS: tuple[str, ...] = (
    "geometric_phase",
    "ReceptorConfigJones",
    "BasisTransformJones",
)

#: The exported names that are not term classes at all: the two base classes,
#: the baseline base class, and the batched-contract support surface.
NON_TERM_EXPORTS: tuple[str, ...] = (
    "JonesTerm",
    "JonesChain",
    "JonesBaselineTerm",
    "DirectionBatch",
    "evaluate_antenna_jones",
)

#: The exported terms that carry real physics, with the slice that implemented
#: each.  FLIPPED BY: Tier 7D, which moved ``G`` and ``B`` out of
#: ``PLANNED_TERMS`` -- the first two rows of the 11-row planned table to become
#: numbers.  FLIPPED BY: Tier 7E, which moved the four calibration terms ``D``,
#: ``X``, ``Kd`` and ``Rc``, completing workstream A.  FLIPPED BY: Tier 7F,
#: which moved ``P`` and opened workstream B.  FLIPPED BY: Tier 7G, which moved
#: ``Z`` and ``T`` and closed workstream B -- after it every exported
#: ``JonesTerm`` is here.  Tier 7H moves the remaining two, which are
#: ``JonesBaselineTerm`` and not chain terms at all.
IMPLEMENTED_TERMS: dict[str, str] = {
    "GainJones": "7D",
    "BandpassJones": "7D",
    "PolarizationLeakageJones": "7E",
    "CrosshandJones": "7E",
    "DelayJones": "7E",
    "CableReflectionJones": "7E",
    "ParallacticAngleJones": "7F",
    "IonosphereJones": "7G",
    "TroposphereJones": "7G",
}

#: The exported terms still at ``term_status == "planned"``, with the slice that
#: implements each.  Every one of them raises when evaluated; none is an
#: identity.  Section 5.1's 37-stub table became an 11-row one plus the 26
#: deletions below; Tier 7D left nine rows, Tier 7E five, Tier 7F four, and
#: Tier 7G two -- both of them ``JonesBaselineTerm``, so no ``JonesTerm``
#: subclass is planned any more and ``compute_jones_batch`` could become
#: ``@abstractmethod``.
PLANNED_TERMS: dict[str, str] = {
    "BaselineMultiplicativeJones": "7H",
    "SmearingFactorJones": "7H",
}

#: Section 23's removal ledger, executed by Tier 7C (25 of these) and Tier 7B
#: (``GeometricPhaseJones``).  Twenty-six classes in all, plus the rename.
REMOVED_JONES_CLASSES: tuple[str, ...] = (
    "GeometricPhaseJones",
    "TimeVariableGainJones",
    "ElevationGainJones",
    "PolynomialBandpassJones",
    "SplineBandpassJones",
    "RFIFlaggedBandpassJones",
    "IXRLeakageJones",
    "MuellerLeakageJones",
    "BeamSquintLeakageJones",
    "FieldRotationJones",
    "VLBIFeedRotationJones",
    "TurbulentIonosphereJones",
    "GPSIonosphereJones",
    "SaastamoinenTroposphereJones",
    "TurbulentTroposphereJones",
    "TroposphericOpacityJones",
    "FaradayRotationJones",
    "DifferentialFaradayJones",
    "WPhaseJones",
    "WProjectionJones",
    "WidefieldPolarimetricJones",
    "ElementBeamJones",
    "ArrayFactorJones",
    "DifferentialBeamJones",
    "FringeFitJones",
    "CrosshandPhaseJones",
    "CrosshandDelayJones",
    "FrequencyDependentLeakageJones",
)

#: The modules that held only deleted classes and were deleted with them.
#:
#: Section 33.2 says "five now-empty modules"; the arithmetic gives three, and
#: Section 34's writable list marks exactly these three "(delete)".  The other
#: nine former stub modules keep one or two planned terms each.
REMOVED_STUB_MODULES: tuple[str, ...] = (
    "faraday.py",
    "wterm.py",
    "element_beam.py",
)

#: The modules that were stub modules at the gate and now hold planned terms.
SURVIVING_TERM_MODULES: tuple[str, ...] = (
    "gain.py",
    "bandpass.py",
    "polarization_leakage.py",
    "parallactic.py",
    "ionosphere.py",
    "troposphere.py",
    "delay.py",
    "crosshand.py",
    "baseline_errors.py",
)


def test_jones_package_exports_exactly_the_recorded_names() -> None:
    """Pins ``__all__``, in order (Section 5.1).

    OWNED BY: Tier 7J, which rebuilds the documentation around these names.

    FLIPPED BY: Tier 7B (43 -> 45) and Tier 7C (45 -> 19).
    """
    import radiosim.core.jones as jones_package

    assert tuple(jones_package.__all__) == EXPORTED_JONES_NAMES
    assert len(EXPORTED_JONES_NAMES) == 19
    assert len(set(EXPORTED_JONES_NAMES)) == 19
    assert len(EXPORTED_JONES_NAMES) == (
        len(NON_TERM_EXPORTS)
        + 1  # geometric_phase
        + len(PLANNED_TERMS)
        + len(IMPLEMENTED_TERMS)
        + 2  # ReceptorConfigJones and BasisTransformJones
    )


def test_every_exported_jones_name_resolves_through_lazy_getattr() -> None:
    """Pins that every exported name binds lazily and none is eagerly imported.

    OWNED BY: Tier 7C.  The lazy table shrinks with the class list.

    PARTLY FLIPPED BY: Tier 7B, after which two exported names are functions
    rather than classes -- ``geometric_phase`` because K is per-baseline, and
    ``evaluate_antenna_jones`` because chain evaluation is not a term.
    """
    import radiosim.core.jones as jones_package

    functions = {"geometric_phase", "evaluate_antenna_jones"}
    for name in EXPORTED_JONES_NAMES:
        resolved = getattr(jones_package, name)
        if name in functions:
            assert callable(resolved) and not isinstance(resolved, type), name
        else:
            assert isinstance(resolved, type), name
    assert set(EXPORTED_JONES_NAMES).issubset(set(jones_package.__dir__()))
    with pytest.raises(AttributeError, match="has no attribute 'NotAJonesTerm'"):
        jones_package.NotAJonesTerm  # noqa: B018


def test_claude_md_claims_forty_six_exported_jones_classes() -> None:
    """Pins defect D0: the documented count disagrees with ``__all__``.

    OWNED BY: Tier 7J, which rewrites the ``CLAUDE.md`` Implementation Status
    and Jones sections around the true surviving name count.  Tier 7C's writable
    list does not include ``CLAUDE.md``, so the claim is left stale here
    deliberately and the gap is recorded rather than quietly closed.
    """
    assert "46 exported classes" in _source("CLAUDE.md")
    assert len(EXPORTED_JONES_NAMES) == 19


def test_every_exported_term_is_real_physics_or_a_declared_plan() -> None:
    """Pins Section 5.1's three-real / 37-stub split at its resolution.

    FLIPPED BY: Tier 7C.  The 37 identity stubs became 26 deletions and 11
    planned terms.  A planned term is not a stub: it returns nothing at all.

    FLIPPED BY: Tier 7D, which implemented ``G`` and ``B``: the planned table
    is nine rows, and two names moved into ``IMPLEMENTED_TERMS``.

    FLIPPED BY: Tier 7E, which implemented ``D``, ``X``, ``Kd`` and ``Rc``: the
    planned table is five rows, and four more names moved.

    FLIPPED BY: Tier 7F, which implemented ``P`` -- the first direction- and
    time-dependent propagation term, and the one that opens workstream B.

    FLIPPED BY: Tier 7G, which implemented ``Z`` and ``T`` and closed
    workstream B.  The planned table is two rows, and both are
    ``JonesBaselineTerm``: every per-antenna term in the chain now carries
    physics, which is what let ``compute_jones_batch`` become
    ``@abstractmethod`` in the same slice.

    OWNED BY: Tier 7H, which turns the last two rows into real physics.
    """
    assert len(REMOVED_JONES_CLASSES) == 28  # 26 deletions + K + the renamed X
    assert len(IMPLEMENTED_TERMS) == 9
    assert len(PLANNED_TERMS) == 2
    assert all(
        issubclass(
            getattr(__import__("radiosim.core.jones", fromlist=[name]), name),
            JonesBaselineTerm,
        )
        for name in PLANNED_TERMS
    )
    assert set(PLANNED_TERMS).isdisjoint(IMPLEMENTED_TERMS)
    assert set(PLANNED_TERMS).isdisjoint(REAL_PHYSICS_EXPORTS)
    assert set(IMPLEMENTED_TERMS).isdisjoint(REAL_PHYSICS_EXPORTS)
    assert set(PLANNED_TERMS) | set(IMPLEMENTED_TERMS) | set(
        REAL_PHYSICS_EXPORTS
    ) == set(EXPORTED_JONES_NAMES) - set(NON_TERM_EXPORTS)


@pytest.mark.parametrize("class_name", sorted(PLANNED_TERMS))
def test_a_planned_term_raises_instead_of_returning_the_identity(
    class_name: str,
) -> None:
    """Pins the resolution of defect D1, one class at a time.

    At the gate each of these returned ``xp.eye(2, dtype=np.complex128)`` from
    ``compute_jones(...)``, for every antenna, direction, frequency and time.
    Asserted one class at a time, deliberately: Section 33.2 requires each
    stub's later implementation to be a visible, deliberate flip of a named test
    rather than one aggregate assertion quietly losing rows.

    FLIPPED BY: Tier 7C.  There is no ``compute_jones`` and no identity return;
    there is a name, a documented effect, and a refusal.

    OWNED BY: Tier 7D through Tier 7H.
    """
    import radiosim.core.jones as jones_package

    term_class = getattr(jones_package, class_name)
    term = term_class()
    is_baseline = isinstance(term, JonesBaselineTerm)
    assert isinstance(term, JonesTerm) is not is_baseline

    assert not hasattr(term, "compute_jones")
    assert not hasattr(term, "compute_baseline_term")
    assert term.term_status == "planned"

    method = term.compute_baseline_factor if is_baseline else term.compute_jones_batch
    kwargs: dict[str, Any] = {
        "directions": _planned_term_directions(),
        "frequency_hz": 1.5e8,
        "freq_idx": 0,
        "time_mjd": 60_000.0,
        "time_idx": 0,
        "backend": get_backend("numpy"),
        "dtype": np.complex128,
    }
    if is_baseline:
        kwargs |= {"baseline_idx": 0, "antenna_p": 0, "antenna_q": 1}
    else:
        kwargs["antenna_idx"] = 0

    with pytest.raises(NotImplementedError) as excinfo:
        method(**kwargs)
    assert class_name in str(excinfo.value)


def _planned_term_directions() -> Any:
    from radiosim.core.jones import DirectionBatch

    values = np.linspace(0.2, 1.2, 3)
    return DirectionBatch(
        alt_rad=values,
        az_rad=values / 2.0,
        dir_l=np.cos(values) * np.sin(values / 2.0),
        dir_m=np.cos(values) * np.cos(values / 2.0),
        dir_n=np.sin(values),
        ra_rad=values,
        dec_rad=-values,
        hour_angle_rad=values / 3.0,
        n_dir=3,
    )


@pytest.mark.parametrize("module_name", SURVIVING_TERM_MODULES)
def test_a_surviving_term_module_carries_no_stub_marker(module_name: str) -> None:
    """Pins the removal of the ``TODO: implement properly`` marker.

    At the gate every one of these twelve modules carried the module-level line
    ``"Stub implementation: returns identity matrix. TODO: implement properly."``
    and a class docstring beginning ``"Stub: ..."``.

    FLIPPED BY: Tier 7C.  Each surviving module now states the term's
    mathematics, its units and signs, its citation, and the slice that
    implements it -- which is what makes the class worth keeping while its
    physics does not exist yet.
    """
    text = (JONES_ROOT / module_name).read_text(encoding="utf-8")
    assert "TODO" not in text
    assert "Stub" not in text
    assert "xp.eye(2, dtype=np.complex128)" not in text
    assert "References" in text or "planned" in text


@pytest.mark.parametrize("module_name", REMOVED_STUB_MODULES)
def test_a_module_of_only_deleted_classes_is_deleted(module_name: str) -> None:
    """``faraday.py``, ``wterm.py`` and ``element_beam.py`` held nothing else.

    FLIPPED BY: Tier 7C.
    """
    assert not (JONES_ROOT / module_name).exists()


@pytest.mark.parametrize("class_name", REMOVED_JONES_CLASSES)
def test_a_removed_class_is_absent_from_every_access_path(class_name: str) -> None:
    """Pins the removal ledger, one name at a time.

    FLIPPED BY: Tier 7B (``GeometricPhaseJones``) and Tier 7C (the rest).
    ``docs/migration_guide.md`` names the replacement for every one of them,
    which is the Tier 5H review's requirement made mechanical.
    """
    import radiosim.core.jones as jones_package

    assert class_name not in jones_package.__all__
    assert class_name not in jones_package.__dir__()
    with pytest.raises(AttributeError):
        getattr(jones_package, class_name)

    for path in sorted(SOURCE_ROOT.rglob("*.py")):
        assert class_name not in path.read_text(encoding="utf-8"), path

    assert class_name in _source("docs/migration_guide.md")


def test_todo_markers_outside_the_stub_modules() -> None:
    """Pins the true ``TODO`` inventory of ``src/radiosim``.

    Section 5.1 states that "a repository-wide search finds **no** ``TODO``
    marker anywhere in ``src/radiosim`` outside these twelve stub modules".
    That was **not** true at the gate commit: ``cli/main.py:6``
    ("TODO: Future enhancements for v0.3.0+", present since ``be231d2``) and
    ``core/sky/registry/catalogs.py:595``
    ("TODO(scientific-coverage): ...", present since ``8372dec``) both predate
    ``ac4fe41``.  Neither is a Jones stub and neither weakens ``SCI-001``, but
    7C's residual scan (I20) must exclude them explicitly rather than assert an
    empty set and then be relaxed when it fails.

    FLIPPED BY: Tier 7C, after which those two are the *only* ``TODO`` carriers
    in the package: the twelve stub modules' markers are gone.

    The load-bearing half of the claim -- that the **beam** subsystem is
    TODO-free, which is why Section 19's ``SCI-003`` disposition rests on
    ``beam/TODO.md`` rather than on in-code markers -- does hold, and is
    asserted here.
    """
    carriers = {
        path.relative_to(SOURCE_ROOT).as_posix()
        for path in sorted(SOURCE_ROOT.rglob("*.py"))
        if "TODO" in path.read_text(encoding="utf-8")
    }
    assert carriers == {"cli/main.py", "core/sky/registry/catalogs.py"}

    for name in ("cli/main.py", "core/sky/registry/catalogs.py"):
        assert "TODO: implement properly" not in (SOURCE_ROOT / name).read_text(
            encoding="utf-8"
        )
    assert not any(
        path.name.endswith(".py") and "TODO" in path.read_text(encoding="utf-8")
        for path in (JONES_ROOT / "beam").rglob("*.py")
    )


def test_no_planned_term_accepts_physics_it_would_discard() -> None:
    """Pins the resolution of defect D2: real physics silently dropped.

    At the gate a caller could hand a stub a TEC map, D-terms, a gain sigma, a
    bandpass table, a feed-angle offset or an elevation array and get no error,
    no warning and no effect.  That is the concrete harm ``SCI-001`` names, and
    it was materially worse than "returns identity".

    FLIPPED BY: Tier 7C.  A planned term declares no constructor at all, so
    every one of those calls is now a ``TypeError``.  Each term slice introduces
    its real constructor together with the resolution that validates it.

    FLIPPED BY: Tier 7D for ``G`` and ``B``, which now have real constructors
    that accept resolved values and reject everything else -- so their former
    silently-discarded keywords are gone from this table, not from the contract.

    FLIPPED BY: Tier 7E for ``D``, ``X``, ``Kd`` and ``Rc``.  ``D``'s row is the
    one that leaves visibly: a caller could once hand a stub an array of D-terms
    and get no error, no warning and no effect, and the row is gone because the
    constructor now takes resolved leakage coefficients and validates them.

    FLIPPED BY: Tier 7F for ``P``, whose constructor now requires the site
    latitude and one mount type per antenna row and rejects an unmodelled mount.

    FLIPPED BY: Tier 7G for ``Z`` and ``T``.  Both rows leave visibly: a caller
    could once hand the ionosphere stub a TEC array, or the troposphere stub an
    array of elevations, and get no error, no warning and no effect.  Both
    constructors now require resolved values -- a TEC model, antenna positions,
    a shell height and rotation measures; zenith delays, a mapping function and
    a site -- and reject everything else.  The table below is what is left:
    the two baseline terms, probed with the physics keyword each would have
    swallowed.

    OWNED BY: Tier 7H.
    """
    import radiosim.core.jones as jones_package

    discarded = {
        "BaselineMultiplicativeJones": {"matrices": np.zeros((1, 2, 2))},
        "SmearingFactorJones": {"channel_width_hz": 1.0e6},
    }
    for class_name, kwargs in discarded.items():
        term_class = getattr(jones_package, class_name)
        with pytest.raises(TypeError):
            term_class(**kwargs)
        term = term_class()
        assert vars(term) == {}
        assert "__init__" not in vars(term_class)

    # And the two terms this slice implemented now refuse the keywords their
    # stubs used to swallow, because their constructors take resolved values.
    for class_name, kwargs in (
        ("IonosphereJones", {"tec": np.array([1.0e17, 2.0e17])}),
        ("TroposphereJones", {"elevations": np.array([0.5, 0.9])}),
    ):
        with pytest.raises(TypeError):
            getattr(jones_package, class_name)(**kwargs)


def test_capability_flags_are_declared_only_where_they_can_be_verified() -> None:
    """Pins the resolution of defect D10: unverified hints about identities.

    At the gate ``FaradayRotationJones`` and ``WPhaseJones`` claimed unitarity,
    and ``WPhaseJones`` and ``ArrayFactorJones`` claimed scalarity, about a
    matrix that was the 2x2 identity -- true only because the identity is
    trivially both, which is exactly the vacuity the Tier 5H review adjudicated
    as ``SCI-001`` material.

    FLIPPED BY: Tier 7B, which added the flag-verification harness (D10, I2),
    and Tier 7C, which deleted all four of those classes and stripped every flag
    from the terms that survive: invariant I2's sweep cannot verify a claim
    about a matrix that cannot be computed, so a planned term declares none.
    Each term slice adds its flags with its physics and its own I2 case.

    FLIPPED BY: Tier 7D for ``G`` and ``B``, which declare ``is_diagonal``,
    ``is_scalar`` and ``is_unitary`` computed from their own resolved numbers,
    and are swept by invariant I2 in ``test_gain.py`` and ``test_bandpass.py``.

    FLIPPED BY: Tier 7E for ``D``, ``X``, ``Kd`` and ``Rc``, each of which
    computes all three flags from its own resolved parameters and is swept by
    I2 in its own test module.

    FLIPPED BY: Tier 7F for ``P``, which declares ``is_unitary`` unconditionally
    -- a real rotation is orthogonal -- and computes the other two from its
    resolved mount types.

    FLIPPED BY: Tier 7G for ``Z`` and ``T``.  ``Z`` declares ``is_unitary``
    unconditionally and computes the other two from whether a rotation measure
    was configured; ``T`` declares ``is_scalar`` and ``is_diagonal``
    unconditionally and computes ``is_unitary`` from whether an opacity was.

    OWNED BY: Tier 7H, whose two baseline terms are all that is left.
    """
    import radiosim.core.jones as jones_package

    for class_name in PLANNED_TERMS:
        term_class = getattr(jones_package, class_name)
        base = (
            JonesBaselineTerm
            if issubclass(term_class, JonesBaselineTerm)
            else JonesTerm
        )
        for flag in ("is_diagonal", "is_scalar", "is_unitary"):
            assert getattr(term_class, flag, None) is getattr(base, flag, None), (
                class_name,
                flag,
            )
            assert flag not in vars(term_class)

    # ``get_config`` reports the status alongside the flags, so a consumer
    # reading a term's configuration cannot miss whether it runs.  The probe was
    # a *planned* term until this slice; there is no planned ``JonesTerm`` left
    # to read it from, so it reads the term that used to be the probe and
    # asserts the flip itself -- same keys, and a status that now says the
    # physics is there.
    from radiosim.core.jones.ionosphere import ResolvedTecModel

    ionosphere = jones_package.IonosphereJones(
        tec_model=ResolvedTecModel(vertical_tec_tecu=10.0),
        antenna_positions_enu_m=np.zeros((2, 3)),
        shell_height_m=350_000.0,
        rotation_measures_rad_m2=np.zeros(2),
        minimum_elevation_deg=0.0,
    )
    config = ionosphere.get_config()
    assert set(config) >= {
        "name",
        "term_status",
        "is_direction_dependent",
        "is_time_dependent",
        "is_frequency_dependent",
        "is_diagonal",
        "is_scalar",
        "is_unitary",
    }
    assert config["term_status"] == "implemented"


# =========================================================================
# Section 5.3 -- the chain and the evaluation contract
# =========================================================================


class _CountingIdentityTerm(JonesTerm):
    """A direction-dependent probe that counts batched evaluations.

    FLIPPED BY: Tier 7B.  At the gate this probe counted one ``compute_jones``
    call *per direction*; it now counts one ``compute_jones_batch`` call for the
    whole batch, which is the defect-D5 change made visible.
    """

    def __init__(self) -> None:
        self.calls: list[tuple[int, int, int, int]] = []

    @property
    def name(self) -> str:
        return "probe"

    @property
    def is_direction_dependent(self) -> bool:
        return True

    def compute_jones_batch(
        self,
        *,
        antenna_idx: int,
        directions: Any,
        frequency_hz: float,
        freq_idx: int,
        time_mjd: float,
        time_idx: int,
        backend: Any,
        dtype: Any,
    ) -> Any:
        self.calls.append((antenna_idx, directions.n_dir, freq_idx, time_idx))
        return backend.batch_eye((directions.n_dir,), 2, dtype=dtype)


def _probe_directions(n_dir: int) -> Any:
    """A direction batch for the Tier 7B contract pins."""
    from radiosim.core.jones import DirectionBatch

    alt = np.full(n_dir, 1.0)
    az = np.linspace(0.0, 1.0, n_dir)
    return DirectionBatch.from_horizontal(
        alt_rad=alt,
        az_rad=az,
        dir_l=np.cos(alt) * np.sin(az),
        dir_m=np.cos(alt) * np.cos(az),
        dir_n=np.sin(alt),
        latitude_rad=-0.536,
        local_sidereal_time_rad=0.0,
    )


def _evaluate_chain(chain: JonesChain, *, n_dir: int = 3, dtype: Any = None) -> Any:
    return chain.compute_antenna_jones_batch(
        antenna_idx=0,
        directions=_probe_directions(n_dir),
        frequency_hz=1.0e8,
        freq_idx=0,
        time_mjd=60_000.0,
        time_idx=0,
        dtype=np.complex128 if dtype is None else dtype,
    )


def test_jones_term_contract_is_direction_batched() -> None:
    """Defect D5, flipped: one call carries the whole direction batch.

    At the gate ``compute_jones`` took ``source_idx: int | None`` and the default
    ``compute_jones_all_sources`` was a Python list comprehension calling it once
    per direction -- one Python call per HEALPix pixel, which is why the diffuse
    solver bypassed the chain entirely.

    OWNED BY: Tier 7B.  FLIPPED BY: Tier 7B, which replaced the contract with
    ``DirectionBatch`` and ``compute_jones_batch``.
    """
    assert not hasattr(JonesTerm, "compute_jones")
    assert not hasattr(JonesTerm, "compute_jones_all_sources")

    signature = inspect.signature(JonesTerm.compute_jones_batch)
    assert list(signature.parameters) == [
        "self",
        "antenna_idx",
        "directions",
        "frequency_hz",
        "freq_idx",
        "time_mjd",
        "time_idx",
        "backend",
        "dtype",
    ]
    # Every argument after ``self`` is keyword-only: a mis-ordered call is
    # impossible rather than merely discouraged.
    assert all(
        parameter.kind is inspect.Parameter.KEYWORD_ONLY
        for name, parameter in signature.parameters.items()
        if name != "self"
    )

    probe = _CountingIdentityTerm()
    result = probe.compute_jones_batch(
        antenna_idx=3,
        directions=_probe_directions(7),
        frequency_hz=1.0e8,
        freq_idx=1,
        time_mjd=60_000.0,
        time_idx=2,
        backend=get_backend("numpy"),
        dtype=np.complex128,
    )
    assert np.asarray(result).shape == (7, 2, 2)
    assert probe.calls == [(3, 7, 1, 2)]


def test_jones_chain_add_term_rejects_a_baseline_term() -> None:
    """Defect D7, flipped: ``add_term`` now enforces its own docstring.

    At the gate the docstring said "Only ``JonesTerm`` subclasses may be added
    here" while ``add_term`` performed no check at all, so a
    ``JonesBaselineTerm`` was accepted and then failed with an
    ``AttributeError`` deep inside evaluation instead of a typed rejection at
    the point of the mistake.

    OWNED BY: Tier 7B.  FLIPPED BY: Tier 7B, which added the isinstance guard.
    """
    import radiosim.core.jones as jones_package

    assert "Only ``JonesTerm`` subclasses may be added here" in (
        JonesChain.__doc__ or ""
    )

    chain = JonesChain(get_backend("numpy"))
    baseline_term = jones_package.BaselineMultiplicativeJones()
    with pytest.raises(TypeError, match="JonesBaselineTerm"):
        chain.add_term(baseline_term)
    assert chain.terms == []


def test_jones_chain_seed_dtype_comes_from_the_caller() -> None:
    """Defect D8, flipped: the identity seeds no longer come from a literal.

    At the gate both seeds were a literal ``np.complex128``, so a chain whose
    every term was ``complex64`` still produced a ``complex128`` product and
    ``PrecisionConfig`` was not even reachable from here.

    OWNED BY: Tier 7B.  FLIPPED BY: Tier 7B, which passes the resolved dtype
    into ``compute_antenna_jones_batch`` -- the solver resolves it once from the
    precision model, so the chain never chooses.  ``precision`` is deliberately
    still not a constructor argument: a chain rebuilt per ``(time, frequency)``
    would otherwise carry a second copy of the run's precision policy.
    """
    assert "precision" not in inspect.signature(JonesChain.__init__).parameters

    backend = get_backend("numpy")
    empty = JonesChain(backend)
    for dtype in (np.complex64, np.complex128):
        assert np.asarray(_evaluate_chain(empty, dtype=dtype)).dtype == dtype

    loaded = JonesChain(backend)
    loaded.add_term(_CountingIdentityTerm())
    for dtype in (np.complex64, np.complex128):
        assert np.asarray(_evaluate_chain(loaded, dtype=dtype)).dtype == dtype

    text = _source("src/radiosim/core/jones/chain.py")
    assert "dtype=np.complex128" not in text

    # The precision model that used to be ignored really does offer other dtypes.
    assert PrecisionConfig.fast().jones.get_dtype("gain") == np.complex64


def test_receptor_config_jones_returns_the_dtype_it_is_given(tmp_path) -> None:
    """Defect D9, flipped: the C term honours ``PrecisionConfig`` too.

    OWNED BY: Tier 7B.  FLIPPED BY: Tier 7B, which made C and H dtype-correct.
    The default preset resolves ``complex128``, which is what the removed
    literal said, so every shipped configuration is bit-identical; a preset that
    resolves anything else is where the fix becomes observable.
    """
    import radiosim.core.jones as jones_package
    from radiosim.core.instrument_adapters import SolverInstrumentView

    instrument, _, receptors = _solver_components(tmp_path)
    assert type(instrument) is SolverInstrumentView

    term = jones_package.ReceptorConfigJones(receptors=receptors, instrument=instrument)
    for dtype in (np.complex64, np.complex128):
        matrix = np.asarray(
            term.compute_jones_batch(
                antenna_idx=0,
                directions=_probe_directions(3),
                frequency_hz=1.0e8,
                freq_idx=0,
                time_mjd=60_000.0,
                time_idx=0,
                backend=get_backend("numpy"),
                dtype=dtype,
            )
        )
        assert matrix.dtype == dtype
        # A direction-independent term returns one broadcastable matrix.
        assert matrix.shape == (1, 2, 2)
        # The default homogeneous-linear, zero-rotation case is exactly I2.
        np.testing.assert_array_equal(matrix[0], IDENTITY.astype(dtype))


def test_jones_chain_docstring_records_the_designed_chain_order() -> None:
    """Defect D11, flipped; defect D12 still recorded.

    At the gate the "extended" line was undesigned: it placed ``W`` sky-side of
    ``Z`` and declared the diagonal terms ``Kd``/``Rc`` "applied separately",
    neither of which is what Section 20.12 designs.

    OWNED BY: Tier 7B (which replaces the extended line) and Tier 7F (which
    corrects the ``P``/``C`` order).  FLIPPED BY: Tier 7B for the extended line
    only -- the canonical Tier 5 line, with ``P`` correlator-side of ``C``, was
    deliberately left alone, because moving ``P`` is a change to an accepted
    Tier 5 decision and belongs to the slice that makes ``P`` real.

    FLIPPED BY: Tier 7F, which is that slice.  ``P`` now sits sky-side of ``C``,
    the superseded Tier 5 line is gone from the docstring entirely, and the
    class says *why* the placement is physical rather than only what it is.
    D12 is closed here.
    """
    docstring = JonesChain.__doc__ or ""
    # Corrected: the Tier 5 line no longer states what the chain is.
    assert "J_total = H @ G @ B @ D @ P @ C @ E @ T @ Z" not in docstring

    # Replaced: the undesigned extended line is gone, and the Section 12.2
    # canonical order is in its place, with P sky-side of C.
    assert "@ F @ T @ Z @ W" not in docstring
    assert "(K, Kd, Rc applied separately)" not in docstring
    assert "J_total = H @ G @ B @ Rc @ Kd @ X @ D @ C @ E @ P @ T @ Z" in docstring
    assert "J_total = H @ G @ B @ Rc @ Kd @ X @ D @ P @ C @ E @ T @ Z" not in docstring
    assert "commute" in docstring
    collapsed = " ".join(docstring.split())
    assert "M(basis) R(chi + psi) = C R(psi)" in collapsed

    # The composition really is terms[0] @ ... @ terms[-1], reversed at
    # evaluation time, which is what makes the add order the chain order.
    chain_source = inspect.getsource(jones_chain.JonesChain.compute_antenna_jones_batch)
    assert "for term in reversed(self.terms)" in chain_source


# =========================================================================
# Section 5.4 -- solver integration
# =========================================================================


#: The six optional term blocks the gate ``jones_config`` accepted, kept as a
#: record of what the removed dictionary could express.  Nothing constructs one
#: any more: Tier 7C removed the parameter, and Tier 7D replaces it with the
#: typed ``jones:`` schema whose rejections Section 24 fixes verbatim.
GATE_OPTIONAL_JONES_TERMS: dict[str, dict[str, Any]] = {
    "G": {"enabled": True, "sigma": 0.4},
    "B": {"enabled": True, "bandpass_gains": [0.5, 2.0]},
    "D": {"enabled": True, "d_terms": [0.1, 0.2]},
    "P": {"enabled": True, "mount_type": "altaz"},
    "T": {"enabled": True},
    "Z": {"enabled": True, "tec": 5.0e17, "include_faraday": True},
}


def test_no_solver_or_simulator_accepts_a_jones_config(tmp_path) -> None:
    """Defects D1 and D3, closed by removal rather than by validation.

    At the gate, enabling G, B, D, P, T and Z with physically meaningful
    parameters produced a **bit-identical** cube: the most direct statement of
    ``SCI-001``, since a user who configured instrumental gains, a bandpass,
    leakage, parallactic rotation, a troposphere and an ionosphere got exactly
    the unmodelled sky back, silently.  Tier 7B made that a loud
    ``NotImplementedError``.

    FLIPPED BY: Tier 7C, which removes the parameter outright (Section 33.2).
    An untyped dictionary that could only ever reach an identity stub is not a
    configuration surface, and keeping it while deleting the stubs would leave a
    keyword whose every accepted value fails.  Tier 7D introduces the typed
    ``jones:`` section that replaces it, wired through ``ResolvedJonesTerms``.

    DISCHARGED BY: Tier 7D, which added the typed replacement.  The pin itself
    stands unchanged -- no signature accepts the untyped dictionary, and that is
    still true -- and the discharge is the assertion below that every one of
    those four signatures now takes a ``ResolvedJonesTerms`` instead.

    OWNED BY: Tier 7D.
    """
    from radiosim.core.jones_terms import EMPTY_JONES_TERMS, ResolvedJonesTerms
    from radiosim.simulator.base import VisibilitySimulator
    from radiosim.simulator.rime import RIMESimulator

    for function in (
        calculate_visibility,
        calculate_visibility_healpix,
        VisibilitySimulator.calculate_visibilities,
        RIMESimulator.calculate_visibilities,
    ):
        assert "jones_config" not in inspect.signature(function).parameters

    assert "jones_config" not in _source("src/radiosim/core/hybrid.py")
    assert "jones_config" not in _source("src/radiosim/api/simulator.py")
    assert "jones_config" not in _source("src/radiosim/simulator/rime.py")
    assert "jones_config" not in _source("src/radiosim/simulator/base.py")

    # FLIPPED BY: Tier 7F, which deleted the unreachable D17 guard.  Three of
    # the point solver's four mentions of ``jones_config`` were that guard's
    # signature, docstring and body; the one that survives is the sentence in
    # ``_build_jones_chain`` explaining what the removed parameter was replaced
    # by, which is a historical note rather than a live surface.
    point = _source("src/radiosim/core/visibility.py")
    assert "_reject_parallactic_rotation" not in point
    assert "jones_config.get(" not in point
    assert point.count("jones_config") == 1

    # The discharge: a typed, defaulted ``jones_terms`` on all four signatures.
    for function in (
        calculate_visibility,
        calculate_visibility_healpix,
        VisibilitySimulator.calculate_visibilities,
        RIMESimulator.calculate_visibilities,
    ):
        parameter = inspect.signature(function).parameters["jones_terms"]
        assert parameter.default is EMPTY_JONES_TERMS
        assert type(parameter.default) is ResolvedJonesTerms
        # The annotation is compared as text: two of the four modules declare it
        # under ``TYPE_CHECKING``, so resolving it would need every solver-only
        # name they import to be resolvable here too.
        annotation = parameter.annotation
        if not isinstance(annotation, str):
            annotation = getattr(annotation, "__name__", str(annotation))
        assert "ResolvedJonesTerms" in annotation

    # And the cube is unchanged, because nothing removed was ever reachable.
    instrument, beam_system, receptors = _solver_components(tmp_path)
    kwargs: dict[str, Any] = {
        "instrument": instrument,
        "beam_system": beam_system,
        "source_arrays": _workload_point_sources(polarized=True, gaussian=False),
        "location": WORKLOAD_LOCATION,
        "time_grid": WORKLOAD_TIME_GRID,
        "frequencies": _WORKLOAD_FREQS,
        "backend": get_backend("numpy"),
        "receptors": receptors,
    }
    cube = np.asarray(calculate_visibility(**kwargs))
    assert float(np.max(np.abs(cube))) > 0.0

    for term_name, term_config in GATE_OPTIONAL_JONES_TERMS.items():
        with pytest.raises(TypeError, match="jones_config"):
            calculate_visibility(  # type: ignore[call-arg]
                **kwargs,
                jones_config={term_name: term_config},
            )


def test_build_jones_chain_carries_only_the_terms_that_exist(
    tmp_path,
) -> None:
    """Pins the add order, and with it defect D12's observability status.

    At the gate, with every optional term enabled the chain was
    H, G, B, D, P, C, E, T, Z -- ``P`` correlator-side of ``C``, which Section 12
    calls wrong for a circular receptor.  The error was unobservable only
    because ``P`` was the identity.

    FLIPPED BY: Tier 7C.  Six of those nine slots held identity stubs and are
    now empty, so the chain is exactly H, C, E and the builder takes no
    configuration.

    FLIPPED BY: Tier 7F, which moved ``P`` sky-side of ``C`` in the solver's own
    documented factorization and made the term real.  What the pin recorded --
    that the documented order was the uncorrected one -- is discharged, and what
    replaces it is the corrected order plus the assertion that the superseded
    one is gone.  The chain with no ``jones:`` section is still exactly H, C, E:
    the correction moves a slot no default run occupies.
    """
    from radiosim.core.visibility import _build_jones_chain

    instrument, beam_system, receptors = _solver_components(tmp_path)
    chain = _build_jones_chain(
        get_backend("numpy"),
        instrument,
        np.array([0.9, 1.0]),
        np.array([1.4, 1.5]),
        100e6,
        0,
        2,
        WORKLOAD_LOCATION,
        time_mjd=60000.0,
        beam_system=beam_system,
        receptors=receptors,
    )
    assert [term.name for term in chain.terms] == ["H", "C", "E"]
    assert "jones_config" not in inspect.signature(_build_jones_chain).parameters

    # The corrected order is the documented one, and the superseded one is gone.
    documented = " ".join(_build_jones_chain.__doc__.split())
    assert "J = H @ G @ B @ Rc @ Kd @ X @ D @ C @ E @ P @ T @ Z" in documented
    assert "J = H @ G @ B @ D @ P @ C @ E @ T @ Z" not in documented


def test_production_supplies_no_jones_parameter_at_all(tmp_path) -> None:
    """Pins defect D3 at its resolution rather than at its symptom.

    At the gate ``core/hybrid.py`` hard-coded ``jones_config=None`` at the one
    production call site, ``RIMESimulator.simulate`` and
    ``VisibilitySimulator.simulate`` declared the parameter, and
    ``api/simulator.py`` never mentioned it -- so no supported entry point could
    enable a term.  Tier 7C removed the parameter, so there is no hard-coded
    ``None`` left to find.

    FLIPPED BY: Tier 7D, which put the typed ``jones_terms`` at exactly this
    call site.  What survives from the pin is the property that mattered: there
    is no hard-coded ``None``, and what the production path passes is a resolved
    inventory rather than raw configuration.
    """
    from radiosim.simulator.rime import RIMESimulator

    hybrid = _source("src/radiosim/core/hybrid.py")
    assert "jones_config" not in hybrid
    assert "jones_config=None" not in hybrid
    assert "jones_terms=jones_terms" in hybrid
    assert "jones_terms: ResolvedJonesTerms = EMPTY_JONES_TERMS" in hybrid

    # The hybrid path still reaches the point solver, and still gets a cube.
    instrument, beam_system, receptors = _solver_components(tmp_path)
    cube = np.asarray(
        RIMESimulator().calculate_visibilities(
            instrument=instrument,
            beam_system=beam_system,
            source_arrays=_workload_point_sources(polarized=False, gaussian=False),
            frequencies=_WORKLOAD_FREQS,
            backend=get_backend("numpy"),
            location=WORKLOAD_LOCATION,
            time_grid=WORKLOAD_TIME_GRID,
            receptors=receptors,
        )
    )
    assert float(np.max(np.abs(cube))) > 0.0


def test_the_ad_hoc_jones_validation_surface_is_gone(tmp_path) -> None:
    """Pins the removal of the three checks that stood in for a schema.

    At the gate ``calculate_visibility`` carried a type check, a "no beam key"
    check and the parallactic guard, and no field validation of any kind below
    the top level -- ``{"G": {"enabled": "yes please"}}`` was accepted and
    ignored.  Tier 7B made a truthy value fail loudly; Tier 7C removed the
    parameter and, with it, all three checks.

    DISCHARGED BY: Tier 7D, whose typed ``jones:`` schema replaced the three
    with strict parsing plus the verbatim rejections R2, R4-R7 and R11, each
    asserted by exact string in ``tests/unit/test_core/test_jones_resolution.py``
    and ``tests/unit/test_io/test_jones_config.py``.  The pin stands: the ad-hoc
    checks are still gone, and none came back.

    OWNED BY: Tier 7D.
    """
    from radiosim.core.jones_errors import (
        IdentityJonesTermError,
        InvalidJonesConfigError,
        JonesAssignmentError,
    )

    assert issubclass(IdentityJonesTermError, InvalidJonesConfigError)
    assert issubclass(JonesAssignmentError, Exception)

    point = _source("src/radiosim/core/visibility.py")
    assert "jones_config must be a dict or None" not in point
    assert "must not contain a beam entry" not in point

    instrument, beam_system, receptors = _solver_components(tmp_path)
    kwargs: dict[str, Any] = {
        "instrument": instrument,
        "beam_system": beam_system,
        "source_arrays": _workload_point_sources(polarized=False, gaussian=False),
        "location": WORKLOAD_LOCATION,
        "time_grid": WORKLOAD_TIME_GRID,
        "frequencies": _WORKLOAD_FREQS,
        "backend": get_backend("numpy"),
        "receptors": receptors,
    }

    # Every former spelling is now one ordinary unexpected-keyword TypeError.
    for value in ([("G", True)], {"beam": {}}, {"G": {"enabled": "yes please"}}, None):
        with pytest.raises(TypeError, match="jones_config"):
            calculate_visibility(**kwargs, jones_config=value)  # type: ignore[call-arg]

    assert float(np.max(np.abs(np.asarray(calculate_visibility(**kwargs))))) > 0.0


def test_the_parallactic_rotation_guard_is_gone_and_the_combination_is_legal(
    tmp_path,
) -> None:
    """Pins defect D17 at its resolution.

    The guard fired only when ``jones_config["P"]["enabled"]`` was true, which
    no supported entry point could arrange, so its message could never be seen
    by a user of the public API.

    FLIPPED BY: Tier 7F.  The guard is deleted, and what it forbade is now the
    physics: with ``P`` real and sky-side of ``C``,
    ``C_p P_p = M(basis) R(chi + psi)`` is the full time-dependent receptor
    orientation, which is exactly what ``Tier5ReceptorFeedPlan.md`` Section 12.3
    said would happen "when Tier 7 implements ``P``".  A rotated receptor on a
    rotating mount is therefore accepted and carried, not rejected.
    """
    from radiosim.core.jones_terms import resolve_jones_terms

    assert "_reject_parallactic_rotation" not in _source(
        "src/radiosim/core/visibility.py"
    )
    assert "a non-zero feed_rotation_deg cannot be combined" not in _source(
        "src/radiosim/core/visibility.py"
    )
    assert callable(resolve_jones_terms)


def test_healpix_solver_shares_the_one_chain_and_the_one_evaluator() -> None:
    """Defect D4, closed.

    At the gate the diffuse path never touched ``JonesChain``: it built its own
    constant ``H_p @ C_p`` and left-multiplied it onto the beam batch, so a term
    added to the point path would silently not apply to a HEALPix sky.

    OWNED BY: Tier 7B.  FLIPPED BY: Tier 7B, which routes both solvers through
    ``_build_jones_chain`` and ``evaluate_antenna_jones``.  The typed
    ``jones_terms`` parameter is still Tier 7D's to add -- what 7B guarantees is
    that when it arrives there is exactly one place for it to reach.
    """
    assert (
        "jones_config" not in inspect.signature(calculate_visibility_healpix).parameters
    )

    text = _source("src/radiosim/core/visibility_healpix.py")
    assert "_build_jones_chain" in text
    assert "evaluate_antenna_jones" in text
    assert "_receptor_transforms" not in text
    assert "_evaluate_beam_batch_by_antenna" not in text

    # Exactly one chain-composition site in the whole package.  The gate count
    # was nine ``add_term`` calls; Tier 7C deleted the six that added an
    # identity stub, leaving three; Tier 7D replaced those three with a single
    # call inside one walk of the canonical order, which is a stronger form of
    # the same property -- there is now one statement in the package that puts a
    # term into a chain, and it cannot treat a configured term differently from
    # an always-on one.
    point = _source("src/radiosim/core/visibility.py")
    assert point.count("chain.add_term(") == 1
    assert "chain.add_term(" not in text


def test_geometric_phase_is_implemented_exactly_once() -> None:
    """Defect D6, closed: one function, no class, no inline copy.

    At the gate the same formula existed three times: an exported
    ``GeometricPhaseJones`` class that no solver constructed, and one inline copy
    in each solver, each carrying the exact non-coplanar ``w * (n - 1)`` term.

    OWNED BY: Tier 7B.  FLIPPED BY: Tier 7B, which extracted
    ``geometric_phase()`` and deleted the class.  ``K`` is per-baseline, so it is
    a function the solver applies beside the compiled contraction rather than a
    chain term.
    """
    import radiosim.core.jones as jones_package

    with pytest.raises(AttributeError):
        jones_package.GeometricPhaseJones  # noqa: B018
    assert callable(jones_package.geometric_phase)

    point = _source("src/radiosim/core/visibility.py")
    healpix = _source("src/radiosim/core/visibility_healpix.py")
    for text in (point, healpix):
        assert "GeometricPhaseJones" not in text
        # The formula is called, never re-spelled.
        assert "backend.exp(-2j * np.pi" not in text
        assert "geometric_phase(" in text

    geometric = _source("src/radiosim/core/jones/geometric.py")
    assert geometric.count("backend.exp(-2j * np.pi") == 1
    assert "bl_w * (dir_n - 1.0)" in geometric

    assert (SOURCE_ROOT / "core" / "jones" / "evaluate.py").exists()
    assert (SOURCE_ROOT / "core" / "jones" / "directions.py").exists()


def test_the_non_coplanar_w_contribution_is_already_exact(tmp_path) -> None:
    """Pins defect D19: an enabled ``W`` term would double-count.

    The inline geometric phase already carries ``w * (n - 1)`` exactly, in both
    solvers, so the W-term Jones class models an effect the forward model has
    always applied.  The source pin is the decisive half; the behavioural half
    below records that the ``n`` direction cosine is live -- moving the sources
    well off the phase centre changes the cube -- so the term is not a
    no-op that a later slice could reintroduce harmlessly.

    FLIPPED BY: Tier 7C, which deleted ``wterm.py`` rather than implementing it,
    so the double-count hazard is closed by construction: there is no W term to
    enable.  ``Fix.md`` Section 16 Workstream C's "W/non-coplanar effects" item
    is answered by this pin plus the documentation, not by a term.

    ANCHOR UPDATED BY: Tier 7B, which extracted the two inline copies into the
    one ``geometric_phase()`` function (defect D6).  The ``w (n - 1)`` term the
    pin is about is unchanged; there is now one place to read it rather than
    two, which makes the double-count hazard easier to see, not harder.
    """
    assert "bl_w * (dir_n - 1.0)" in _source("src/radiosim/core/jones/geometric.py")
    for solver in ("visibility.py", "visibility_healpix.py"):
        assert "geometric_phase(" in _source(f"src/radiosim/core/{solver}")

    instrument, beam_system, receptors = _solver_components(tmp_path)
    kwargs: dict[str, Any] = {
        "instrument": instrument,
        "beam_system": beam_system,
        "location": WORKLOAD_LOCATION,
        "time_grid": WORKLOAD_TIME_GRID,
        "frequencies": _WORKLOAD_FREQS,
        "backend": get_backend("numpy"),
        "receptors": receptors,
    }
    near = _workload_point_sources(polarized=False, gaussian=False)
    far = dict(near)
    far["dec_rad"] = near["dec_rad"] + 0.35

    near_cube = np.asarray(calculate_visibility(source_arrays=near, **kwargs))
    far_cube = np.asarray(calculate_visibility(source_arrays=far, **kwargs))
    assert _raw_cube_digest(near_cube) != _raw_cube_digest(far_cube)

    # And there is no W term left to double-count with.
    assert not (JONES_ROOT / "wterm.py").exists()
    for name in ("WPhaseJones", "WProjectionJones"):
        assert name not in _source("src/radiosim/core/jones/__init__.py")
    guide = " ".join(_source("docs/user_guide/jones_matrices.rst").split())
    assert "w-phase and w-projection" in guide


def test_rotation_measure_is_already_applied_by_the_point_solver(tmp_path) -> None:
    """Pins defect D18: an enabled ``F`` term would double-count.

    ``source_rm`` flows into ``evaluate_point_flux_at_freq`` inside the
    frequency loop, so line-of-sight Faraday rotation of the *intrinsic* sky is
    already modelled.  A separately configured F term would rotate it twice,
    with no guard anywhere.

    OWNED BY: Tier 7G, whose ``Z`` term owns ionospheric rotation only, and
    Tier 7C, which deletes ``faraday.py``.
    """
    instrument, beam_system, receptors = _solver_components(tmp_path)
    kwargs: dict[str, Any] = {
        "instrument": instrument,
        "beam_system": beam_system,
        "location": WORKLOAD_LOCATION,
        "time_grid": WORKLOAD_TIME_GRID,
        "frequencies": _WORKLOAD_FREQS,
        "backend": get_backend("numpy"),
        "receptors": receptors,
    }
    unrotated = _workload_point_sources(polarized=True, gaussian=False)
    rotated = dict(unrotated)
    rotated["rotation_measure"] = np.array([25.0, -10.0], dtype=np.float64)

    unrotated_cube = np.asarray(calculate_visibility(source_arrays=unrotated, **kwargs))
    rotated_cube = np.asarray(calculate_visibility(source_arrays=rotated, **kwargs))
    assert _raw_cube_digest(unrotated_cube) != _raw_cube_digest(rotated_cube)

    assert "source_rm_t" in _source("src/radiosim/core/visibility.py")


# =========================================================================
# Section 5.6 -- configuration, precision, provenance
# =========================================================================


def test_the_jones_section_is_the_eleventh_configuration_section() -> None:
    """Pins the Jones configuration surface at its arrival.

    FLIPPED BY: Tier 7D, which added the ``jones:`` section.  At the gate there
    was none at all, so no supported entry point could enable a term; the
    surface is now typed, strict, and defaults to ``None`` rather than to an
    empty model, because an absent section and an empty one are different
    statements (R2).

    OWNED BY: Tier 7D.
    """
    assert set(RadioSimConfig.model_fields) == {
        "instrument",
        "beams",
        "receptors",
        "jones",
        "baseline_selection",
        "sky_model",
        "obs_time",
        "obs_frequency",
        "visibility",
        "execution",
        "workflow",
    }
    assert RadioSimConfig.model_fields["jones"].default is None
    assert (SOURCE_ROOT / "io" / "jones_config.py").exists()
    assert (SOURCE_ROOT / "core" / "jones_terms.py").exists()
    assert (SOURCE_ROOT / "core" / "jones_errors.py").exists()


def test_calculation_type_reaches_no_consumer_because_it_no_longer_exists() -> None:
    """Pins defect D13 at its resolution: a validated field that nothing read.

    At the gate ``io/config.py`` was the only module in ``src/radiosim`` that
    mentioned ``calculation_type``, and there it appeared three times -- the
    field declaration and the two halves of the spherical-harmonic rejection.
    No solver, simulator, resolver or runtime model read it, so ``direct_sum``
    was a silent no-op.

    FLIPPED BY: Tier 7C, which removed the field (Section 33.2).  The two
    surviving mentions are the removed-field guidance that tells a user with an
    old document what to do, and the class docstring that says why it is gone --
    neither is a field and neither is read by the runtime.
    """
    carriers = {
        path.relative_to(SOURCE_ROOT).as_posix()
        for path in sorted(SOURCE_ROOT.rglob("*.py"))
        if "calculation_type" in path.read_text(encoding="utf-8")
    }
    assert carriers == {"io/config.py"}

    text = _source("src/radiosim/io/config.py")
    assert "calculation_type: Literal" not in text
    assert "config.visibility.calculation_type" not in text
    assert '"visibility.calculation_type": (' in text
    assert "calculation_type" not in VisibilityConfig.model_fields

    # The honored strategy selector is a different field in a different section,
    # and it is now the only one (invariant I15).
    assert "get_simulator(self._simulator_name)" in _source(
        "src/radiosim/api/simulator.py"
    )


def test_spherical_harmonic_is_no_longer_a_value_or_a_promise(tmp_path) -> None:
    """Pins the removal of the rejection that named this tier.

    FLIPPED BY: Tier 7C, which removed both the value and the message.  Setting
    the key is now a schema-stage removed-field rejection carrying R1's exact
    guidance, not an ``unsupported``-stage promise that Tier 7 will implement a
    spherical-harmonic transform.  Tier 7 does not: Section 18 descopes m-mode
    to register row ``SCI-004``, and Section 24 fixes the replacement text.
    """
    from tests.fixtures.configs import valid_config_mapping

    mapping = valid_config_mapping(tmp_path)
    mapping["visibility"] = {
        **mapping["visibility"],
        "calculation_type": "spherical_harmonic",
    }
    path = tmp_path / "spherical.yaml"
    path.write_text(yaml.safe_dump(mapping), encoding="utf-8")

    with pytest.raises(Exception) as excinfo:
        RadioSimConfig.model_validate(mapping)
    assert "extra" in str(excinfo.value).lower()

    assert "spherical-harmonic calculation is not implemented until Tier 7" not in (
        _source("src/radiosim/io/config.py")
    )
    assert (
        collect_unsupported_issues(
            RadioSimConfig.model_validate(valid_config_mapping(tmp_path))
        )
        == ()
    )

    with pytest.raises(Exception) as excinfo:
        load_config(path)
    assert (
        "visibility.calculation_type was removed before v1.0; the solver "
        "strategy is selected by 'execution.simulator' (currently only 'rime')."
    ) in str(excinfo.value)


@pytest.mark.parametrize(
    "config_name",
    (
        "config.yaml",
        "receptor_circular_example.yaml",
        "hybrid_sky_example.yaml",
        "realistic_foreground_example.yaml",
    ),
)
def test_no_shipped_config_sets_calculation_type(
    config_name: str,
) -> None:
    """Pins the Q3 answer: four shipped configs, none of them sets the key.

    Section 41's Q3 blocked 7C on confirming that no consumer used the value.
    It did not, so 7C deleted the field and the key, and every shipped document
    still validates and still resolves.

    FLIPPED BY: Tier 7C.
    """
    text = (REPO_ROOT / "configs" / config_name).read_text(encoding="utf-8")
    assert "calculation_type" not in text

    data = yaml.safe_load(text)
    assert "calculation_type" not in data["visibility"]
    assert set(data["visibility"]) <= set(VisibilityConfig.model_fields)
    assert RadioSimConfig.model_validate(data).execution.simulator == "rime"


def test_jones_precision_declares_every_term() -> None:
    """Pins defect D15 at its resolution.

    At the gate ``JonesPrecision`` declared eight fields, so ``C`` and ``H`` --
    which are in *every* chain -- and every extended term had no precision of
    their own and silently inherited whatever the caller passed.

    FLIPPED BY: Tier 7D, which added the seven missing fields and resolves all
    fifteen into ``ResolvedJonesDtypes``.

    OWNED BY: Tier 7D.
    """
    from radiosim.core.jones_terms import PRECISION_FIELD_BY_TERM
    from radiosim.io.config import JonesPrecisionInput

    expected = {
        "geometric_phase",
        "beam",
        "ionosphere",
        "troposphere",
        "parallactic",
        "gain",
        "bandpass",
        "polarization_leakage",
        "receptor_config",
        "basis_transform",
        "crosshand",
        "delay",
        "cable_reflection",
        "baseline_multiplicative",
        "smearing",
    }
    assert set(JonesPrecision.model_fields) == expected
    assert set(JonesPrecisionInput.model_fields) == expected
    # Every declared field belongs to exactly one term letter, and every term
    # letter has one: the mapping is what makes "no term without a precision" a
    # checkable claim rather than a count.
    assert set(PRECISION_FIELD_BY_TERM.values()) == expected

    precision = PrecisionConfig.standard()
    assert precision.jones.get_dtype("gain") == np.complex128
    assert precision.jones.get_dtype("receptor_config") == np.complex128
    # ``get_dtype`` is a bare ``getattr``, so a name that is not a term is an
    # ``AttributeError`` rather than a typed rejection.
    with pytest.raises(AttributeError):
        precision.jones.get_dtype("receptor")


def test_mount_types_other_than_fixed_are_no_longer_rejected_by_receptors() -> None:
    """Pins defect D16 at its resolution.

    The Tier 5 rejection was a blanket one -- it fired for *every* non-``fixed``
    mount, with or without a receptor rotation -- so a scientifically ordinary
    alt-az array could not be simulated at all.  The capability gap was held
    open only by ``P`` being a stub.

    FLIPPED BY: Tier 7F.  ``resolve_receptors`` no longer looks at
    ``mount_type``; the rule that replaces the blanket rejection lives with the
    term that discharges it, as rejections R12 and R15 in ``resolve_jones_terms``
    (breaking-change ledger row B14).  R15 is a strictly better contract than
    what it replaces: it names the fix rather than the tier.
    """
    from radiosim.core.jones.parallactic import SUPPORTED_MOUNT_TYPES
    from radiosim.core.receptor import resolve_receptors

    receptor_source = " ".join(_source("src/radiosim/core/receptor.py").split())
    assert "is unsupported by Tier 5 receptors;" not in receptor_source
    assert "time-dependent feed orientation requires the parallactic-angle" not in (
        receptor_source
    )
    assert "_SUPPORTED_MOUNT_TYPE" not in receptor_source
    assert "An antenna mount type Tier 5 defers to Tier 7." not in receptor_source
    assert callable(resolve_receptors)

    # And the five mounts the successor rule names are the five P models.
    assert set(SUPPORTED_MOUNT_TYPES) == {
        "alt-az",
        "equatorial",
        "fixed",
        "alt-az+nasmyth-l",
        "alt-az+nasmyth-r",
    }
    # Both messages are written across adjacent string literals in the source,
    # so they are pinned in halves rather than as one line-wrapped substring.
    jones_source = " ".join(_source("src/radiosim/core/jones_terms.py").split())
    assert "whose feeds rotate " in jones_source
    assert "with the sky; enable 'jones.P' or the simulation would " in jones_source
    assert "which the " in jones_source
    assert "parallactic-angle term does not model; supported mounts are " in (
        jones_source
    )


def test_documentation_no_longer_records_a_stub_surface() -> None:
    """Pins the resolution of defect D21: docs that describe what is there.

    At the gate ``docs/api/jones.rst`` told the reader that many exported terms
    were "identity scaffolds", that "A returned identity matrix is not a modeled
    physical effect", and that the rest of the package was "documented for
    development and inspection" -- statements that became false the moment the
    stubs went.

    FLIPPED BY: Tier 7C.  The reference now says which terms are implemented,
    which are planned, and that a planned one raises.  The full documentation
    rebuild around the surviving names is Tier 7J's.

    FLIPPED BY: Tier 7F for one assertion.  The guide's last "When Tier 7
    implements ..." promise was the one about ``P``, and this slice discharges
    it: ``P`` is implemented, the parallactic-angle boundary now states the
    composition ``C_p P_p = M(basis) R(chi + psi)`` rather than a deferral, and
    the guide carries no outstanding promise at all.  The property being pinned
    -- that the guide describes what is there rather than what is coming -- is
    strengthened, so the assertion is inverted rather than deleted.

    OWNED BY: Tier 7J.
    """
    api_docs = " ".join(_source("docs/api/jones.rst").split())
    for stale in (
        "identity scaffolds",
        "A returned identity matrix is not a modeled physical effect.",
        "documented for development and inspection",
    ):
        assert stale not in api_docs, stale
    assert "term_status" in api_docs
    assert "``compute_jones_batch`` **raises**" in api_docs
    for module_name in (
        "ionosphere",
        "troposphere",
        "parallactic",
        "gain",
        "bandpass",
        "polarization_leakage",
    ):
        assert f"radiosim.core.jones.{module_name}" in api_docs
    for removed in ("faraday", "wterm", "element_beam"):
        assert f"radiosim.core.jones.{removed}" not in api_docs

    guide = " ".join(_source("docs/user_guide/jones_matrices.rst").split())
    assert "When Tier 7 implements" not in guide
    assert "identity scaffolds or later-tier work" not in guide
    assert "A class returning an identity matrix is a scaffold" not in guide
    # What replaces the promise: the terms that really are still planned, named,
    # and the statement that each of them raises.
    #
    # FLIPPED BY: Tier 7G, which implemented ``Z`` and ``T``.  The guide names
    # the two that remain -- both ``JonesBaselineTerm`` -- and says so, and it
    # also records that no ``JonesTerm`` is planned any more, which is the
    # statement the ``@abstractmethod`` flip rests on.
    assert "``M`` and ``Q`` — are ``term_status: planned``" in guide
    assert "each **raises** when evaluated" in guide
    assert "No ``JonesTerm`` is planned any more" in guide


def test_beam_todo_markdown_is_the_sci_003_artifact() -> None:
    """Pins defect D20: an in-source wish list with no dispositions.

    OWNED BY: Tier 7I, which replaces it with
    ``docs/development/beam_physics_scope.md``.
    """
    todo = (JONES_ROOT / "beam" / "TODO.md").read_text(encoding="utf-8")
    lines = todo.splitlines()
    assert lines[0] == "# Beam System — Future Work (v5.0+)"

    # Seven numbered second-level items, whose numbering is itself a leftover:
    # it runs 2, 3, 4, 5, 6, 9, 13, so items were deleted without renumbering.
    top_level = [line for line in lines if line.startswith("## ")]
    assert top_level == [
        "## 2. Cross-Polarization Models",
        "## 3. Near/Far Field Regime",
        "## 4. Aperture Blockage",
        "## 5. Random Surface Errors (Ruze Effect)",
        "## 6. Systematic Aberrations",
        "## 9. Beam Squint",
        "## 13. Pointing Errors",
    ]
    assert len(top_level) == 7
    # No item carries a disposition, a register row, or an owner.
    assert not re.search(r"SCI-00\d", todo)
    assert not (REPO_ROOT / "docs" / "development" / "beam_physics_scope.md").exists()


# =========================================================================
# Baseline fingerprints -- the reference values Tier 7B must reproduce
# =========================================================================


def test_shipped_default_config_fingerprint_is_unchanged(tmp_path) -> None:
    """Re-asserts Tier 6's ``configs/config.yaml`` pins under Tier 7 ownership.

    Tier 7B claims bit-identity with this gate for every shipped configuration.
    Delegating to Tier 6's tables gives that claim coverage in all six
    ``(platform, python)`` environments from day one; copying the digests into
    a second table would only create a second place to get them wrong.
    """
    result = _run_shipped_config("config.yaml", tmp_path)
    assert result.visibilities.shape == (60, 15, 101, 4)
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
        ),
    )


def test_shipped_circular_receptor_config_fingerprint_is_unchanged(tmp_path) -> None:
    """Re-asserts Tier 6's ``receptor_circular_example.yaml`` pins for Tier 7.

    This is the configuration whose ``C``/``H`` terms are not the identity, so
    it is the one that would catch a 7B dtype or ordering regression in the
    receptor path.
    """
    result = _run_shipped_config("receptor_circular_example.yaml", tmp_path)
    assert result.visibilities.shape == (6, 15, 3, 4)
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
        ),
    )


def test_shipped_hybrid_config_is_exactly_the_sum_of_its_components(
    tmp_path,
) -> None:
    """Pins ``configs/hybrid_sky_example.yaml`` without an absolute digest.

    No ``(platform, python)`` digest exists for this configuration in any
    table, and 7A cannot create one that is green on the four x86_64 CI jobs
    from a machine that has no x86_64 host -- see this module's "Baseline
    fingerprint scope" note.  What *is* environment-independent, and is what
    Tier 7B actually has to preserve, is that the hybrid cube is the exact
    backend-domain sum of the point-only and HEALPix-only cubes of the same
    configuration.  Both solvers therefore stay under a bit-level Tier 7 pin on
    every runner.

    The two single-component controls each keep one of the shipped file's two
    sources rather than only switching ``sky_representation``: asking the
    combiner to fold the HEALPix source into a point-source model raises
    "Point-source combination requires converting a HEALPix-only model to point
    sources, which is lossy", which is the Tier 6F opt-in working as designed.
    """
    from radiosim.api import Simulator

    def run(representation: str, source_index: int | None):
        data = _shipped_config_mapping("hybrid_sky_example.yaml", tmp_path)
        sky = dict(data["sky_model"])
        if source_index is not None:
            sky["sources"] = [sky["sources"][source_index]]
        data["sky_model"] = sky
        data["visibility"] = dict(data["visibility"])
        data["visibility"]["sky_representation"] = representation
        simulator = Simulator.from_mapping(data, base_dir=REPO_ROOT / "configs")
        simulator.setup()
        return simulator.run(progress=False)

    hybrid = run("hybrid", None)
    point = run("point_sources", 0)
    healpix = run("healpix_map", 1)

    assert hybrid.visibilities.shape == (5, 15, 4, 4)
    assert str(hybrid.visibilities.dtype) == "complex128"
    assert hybrid.solver.components == ("point", "healpix")
    assert point.solver.components == ("point",)
    assert healpix.solver.components == ("healpix",)
    assert float(np.max(np.abs(np.asarray(point.visibilities)))) > 0.0
    assert float(np.max(np.abs(np.asarray(healpix.visibilities)))) > 0.0

    total = np.asarray(point.visibilities) + np.asarray(healpix.visibilities)
    assert float(np.max(np.abs(np.asarray(hybrid.visibilities) - total))) == 0.0
    assert _raw_cube_digest(hybrid.visibilities) == _raw_cube_digest(total)


def test_shipped_realistic_foreground_config_is_not_hermetic() -> None:
    """Records why the fourth shipped configuration carries no pinned digest.

    Its ``realistic_foreground`` recipe downloads the Remazeilles/Haslam
    408 MHz map and queries VizieR for GLEAM, and the configuration declares
    ``offline: false`` accordingly, so it can never be a hermetic test.  Tier 6A
    reached the same conclusion for the same configuration.  7A did run it
    once, off-gate and online, to answer Section 41's Q2; the measured numbers
    are in this module's docstring.
    """
    config = yaml.safe_load(
        (REPO_ROOT / "configs" / "realistic_foreground_example.yaml").read_text(
            encoding="utf-8"
        )
    )
    source = config["sky_model"]["sources"][0]
    assert source["kind"] == "realistic_foreground"
    assert source["diffuse"] == "haslam"
    assert source["bright_catalogs"] == "gleam"
    assert source["nside"] == 128
    assert config["visibility"]["sky_representation"] == "healpix_map"
    assert config["execution"]["offline"] is False
