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

#: The exact ``__all__`` of ``radiosim.core.jones`` at this gate, in file order.
#: Three base names plus 40 concrete term classes.  Section 5.1 counts 43 and
#: records the ``CLAUDE.md`` "46" claim as defect D0.
EXPORTED_JONES_NAMES: tuple[str, ...] = (
    "JonesTerm",
    "JonesChain",
    "JonesBaselineTerm",
    "GeometricPhaseJones",
    "GainJones",
    "TimeVariableGainJones",
    "ElevationGainJones",
    "BandpassJones",
    "PolynomialBandpassJones",
    "SplineBandpassJones",
    "RFIFlaggedBandpassJones",
    "PolarizationLeakageJones",
    "IXRLeakageJones",
    "MuellerLeakageJones",
    "BeamSquintLeakageJones",
    "ParallacticAngleJones",
    "FieldRotationJones",
    "VLBIFeedRotationJones",
    "IonosphereJones",
    "TurbulentIonosphereJones",
    "GPSIonosphereJones",
    "TroposphereJones",
    "SaastamoinenTroposphereJones",
    "TurbulentTroposphereJones",
    "TroposphericOpacityJones",
    "FaradayRotationJones",
    "DifferentialFaradayJones",
    "WPhaseJones",
    "WProjectionJones",
    "WidefieldPolarimetricJones",
    "ReceptorConfigJones",
    "BasisTransformJones",
    "ElementBeamJones",
    "ArrayFactorJones",
    "DifferentialBeamJones",
    "DelayJones",
    "CableReflectionJones",
    "FringeFitJones",
    "CrosshandPhaseJones",
    "CrosshandDelayJones",
    "FrequencyDependentLeakageJones",
    "BaselineMultiplicativeJones",
    "SmearingFactorJones",
)

#: The three exported classes that implement real physics (Section 5.1 table).
#: ``E`` is deliberately absent: the solver's beam term is the private
#: ``_ResolvedBeamJones`` adapter, not an exported class.
REAL_PHYSICS_EXPORTS: tuple[str, ...] = (
    "GeometricPhaseJones",
    "ReceptorConfigJones",
    "BasisTransformJones",
)


def _stub_freqs() -> np.ndarray:
    return np.array([100e6, 150e6], dtype=np.float64)


#: Every one of the 35 exported ``JonesTerm`` identity stubs, with a constructor
#: call that satisfies its required arguments.  Section 5.1's table lists 37
#: stubs; the two baseline-dependent ones (``M``, ``Q``) are not ``JonesTerm``
#: subclasses and are pinned separately below.
JONES_TERM_STUBS: dict[str, Any] = {
    "GainJones": lambda m: m.GainJones(n_antennas=2),
    "TimeVariableGainJones": lambda m: m.TimeVariableGainJones(2, 3),
    "ElevationGainJones": lambda m: m.ElevationGainJones(n_antennas=2),
    "BandpassJones": lambda m: m.BandpassJones(2, _stub_freqs()),
    "PolynomialBandpassJones": lambda m: m.PolynomialBandpassJones(2, _stub_freqs()),
    "SplineBandpassJones": lambda m: m.SplineBandpassJones(2, _stub_freqs()),
    "RFIFlaggedBandpassJones": lambda m: m.RFIFlaggedBandpassJones(2, _stub_freqs()),
    "PolarizationLeakageJones": lambda m: m.PolarizationLeakageJones(2),
    "IXRLeakageJones": lambda m: m.IXRLeakageJones(2),
    "MuellerLeakageJones": lambda m: m.MuellerLeakageJones(2),
    "BeamSquintLeakageJones": lambda m: m.BeamSquintLeakageJones(2),
    "ParallacticAngleJones": lambda m: m.ParallacticAngleJones(
        antenna_latitudes=np.array([-0.536, -0.536]),
        source_positions=np.array([[1.0, -0.5], [1.1, -0.4]]),
        times=np.array([0.0]),
    ),
    "FieldRotationJones": lambda m: m.FieldRotationJones(np.array([-0.536, -0.536])),
    "VLBIFeedRotationJones": lambda m: m.VLBIFeedRotationJones(
        [{"latitude": -0.536}, {"latitude": 0.1}],
        np.array([[1.0, -0.5]]),
        np.array([0.0]),
    ),
    "IonosphereJones": lambda m: m.IonosphereJones(frequencies=_stub_freqs()),
    "TurbulentIonosphereJones": lambda m: m.TurbulentIonosphereJones(
        2, 2, _stub_freqs()
    ),
    "GPSIonosphereJones": lambda m: m.GPSIonosphereJones(frequencies=_stub_freqs()),
    "TroposphereJones": lambda m: m.TroposphereJones(
        n_antennas=2, frequencies=_stub_freqs()
    ),
    "SaastamoinenTroposphereJones": lambda m: m.SaastamoinenTroposphereJones(
        2, 2, _stub_freqs()
    ),
    "TurbulentTroposphereJones": lambda m: m.TurbulentTroposphereJones(
        n_antennas=2, frequencies=_stub_freqs()
    ),
    "TroposphericOpacityJones": lambda m: m.TroposphericOpacityJones(
        n_antennas=2, frequencies=_stub_freqs()
    ),
    "FaradayRotationJones": lambda m: m.FaradayRotationJones(
        rotation_measure=12.0, frequencies=_stub_freqs()
    ),
    "DifferentialFaradayJones": lambda m: m.DifferentialFaradayJones(
        2, 2, frequencies=_stub_freqs()
    ),
    "WPhaseJones": lambda m: m.WPhaseJones(
        source_lmn=np.zeros((2, 3)), wavelengths=np.array([2.0, 3.0])
    ),
    "WProjectionJones": lambda m: m.WProjectionJones(2),
    "WidefieldPolarimetricJones": lambda m: m.WidefieldPolarimetricJones(
        source_lmn=np.zeros((2, 3))
    ),
    "ElementBeamJones": lambda m: m.ElementBeamJones(n_antennas=2),
    "ArrayFactorJones": lambda m: m.ArrayFactorJones(n_antennas=2, n_elements=4),
    "DifferentialBeamJones": lambda m: m.DifferentialBeamJones(n_antennas=2),
    "DelayJones": lambda m: m.DelayJones(
        n_antennas=2, delays=np.array([1e-9, 2e-9]), frequencies=_stub_freqs()
    ),
    "CableReflectionJones": lambda m: m.CableReflectionJones(
        n_antennas=2, reflection_coeff=0.1, cable_delay=1e-7
    ),
    "FringeFitJones": lambda m: m.FringeFitJones(n_antennas=2),
    "CrosshandPhaseJones": lambda m: m.CrosshandPhaseJones(phase_offset=0.4),
    "CrosshandDelayJones": lambda m: m.CrosshandDelayJones(delay=1e-9),
    "FrequencyDependentLeakageJones": lambda m: m.FrequencyDependentLeakageJones(
        n_antennas=2, frequencies=_stub_freqs()
    ),
}

#: The two exported ``JonesBaselineTerm`` identity stubs (Section 5.1, M and Q).
BASELINE_TERM_STUBS: dict[str, Any] = {
    "BaselineMultiplicativeJones": lambda m: m.BaselineMultiplicativeJones(),
    "SmearingFactorJones": lambda m: m.SmearingFactorJones(),
}

#: The twelve modules whose every class is an identity stub (Section 5.1).
STUB_MODULES: tuple[str, ...] = (
    "gain.py",
    "bandpass.py",
    "polarization_leakage.py",
    "parallactic.py",
    "ionosphere.py",
    "troposphere.py",
    "faraday.py",
    "wterm.py",
    "element_beam.py",
    "delay.py",
    "crosshand.py",
    "baseline_errors.py",
)


def test_jones_package_exports_exactly_forty_three_names() -> None:
    """Pins ``__all__`` at 43 names, in order (Section 5.1).

    OWNED BY: Tier 7C, which deletes 26 stub classes and renames
    ``CrosshandPhaseJones`` to ``CrosshandJones``, and Tier 7J, which rebuilds
    the documentation around the surviving 16 names.
    """
    import radiosim.core.jones as jones_package

    assert tuple(jones_package.__all__) == EXPORTED_JONES_NAMES
    assert len(EXPORTED_JONES_NAMES) == 43
    assert len(set(EXPORTED_JONES_NAMES)) == 43


def test_every_exported_jones_name_resolves_through_lazy_getattr() -> None:
    """Pins that all 43 names bind lazily and none is eagerly imported.

    OWNED BY: Tier 7C.  The lazy table shrinks with the class list.
    """
    import radiosim.core.jones as jones_package

    for name in EXPORTED_JONES_NAMES:
        assert isinstance(getattr(jones_package, name), type), name
    assert set(EXPORTED_JONES_NAMES).issubset(set(jones_package.__dir__()))
    with pytest.raises(AttributeError, match="has no attribute 'NotAJonesTerm'"):
        jones_package.NotAJonesTerm  # noqa: B018


def test_claude_md_claims_forty_six_exported_jones_classes() -> None:
    """Pins defect D0: the documented count disagrees with ``__all__``.

    OWNED BY: Tier 7J, which rewrites the ``CLAUDE.md`` Implementation Status
    and Jones sections around the true surviving name count.
    """
    assert "46 exported classes" in _source("CLAUDE.md")
    assert len(EXPORTED_JONES_NAMES) == 43


def test_only_three_exported_classes_implement_real_physics() -> None:
    """Pins Section 5.1's three-real / 37-stub split.

    OWNED BY: Tier 7C through Tier 7H, each of which converts stubs it owns
    into real terms or deletes them.
    """
    stub_names = set(JONES_TERM_STUBS) | set(BASELINE_TERM_STUBS)
    assert len(stub_names) == 37
    assert stub_names.isdisjoint(REAL_PHYSICS_EXPORTS)
    assert stub_names | set(REAL_PHYSICS_EXPORTS) == set(EXPORTED_JONES_NAMES) - {
        "JonesTerm",
        "JonesChain",
        "JonesBaselineTerm",
    }


@pytest.mark.parametrize("class_name", sorted(JONES_TERM_STUBS))
def test_jones_term_stub_returns_the_two_by_two_identity(class_name: str) -> None:
    """Pins the identity return of one stub, individually (defect D1).

    Asserted one class at a time, deliberately: Section 33.2 requires each
    stub's later deletion or implementation to be a visible, deliberate flip of
    a named test rather than one aggregate assertion quietly losing rows.

    OWNED BY: Tier 7C (deletion of the 26 out-of-scope classes) and Tier 7D
    through Tier 7G (real implementations of G, B, D, X, Kd, Rc, P, Z, T).
    """
    import radiosim.core.jones as jones_package

    term = JONES_TERM_STUBS[class_name](jones_package)
    assert isinstance(term, JonesTerm)

    backend = get_backend("numpy")
    matrix = term.compute_jones(0, 0, 0, 0, backend)
    assert np.asarray(matrix).shape == (2, 2)
    assert str(np.asarray(matrix).dtype) == "complex128"
    np.testing.assert_array_equal(np.asarray(matrix), IDENTITY)

    # The direction-independent call path returns the same identity.
    np.testing.assert_array_equal(
        np.asarray(term.compute_jones(1, None, 1, 1, backend)), IDENTITY
    )


@pytest.mark.parametrize("class_name", sorted(BASELINE_TERM_STUBS))
def test_baseline_term_stub_returns_the_two_by_two_identity(class_name: str) -> None:
    """Pins the identity return of M and Q (defect D1, baseline half).

    OWNED BY: Tier 7H, which implements both on the Hadamard path.
    """
    import radiosim.core.jones as jones_package

    term = BASELINE_TERM_STUBS[class_name](jones_package)
    assert isinstance(term, JonesBaselineTerm)
    assert not isinstance(term, JonesTerm)

    matrix = term.compute_baseline_term(0, 1, 0, 0, 0, get_backend("numpy"))
    np.testing.assert_array_equal(np.asarray(matrix), IDENTITY)


@pytest.mark.parametrize("module_name", STUB_MODULES)
def test_stub_module_carries_the_todo_marker(module_name: str) -> None:
    """Pins the ``TODO: implement properly`` marker in each stub module.

    Section 5.1 records that no ``TODO`` marker exists anywhere in
    ``src/radiosim`` outside these twelve modules; the complementary half of
    that claim is pinned by
    ``test_no_todo_marker_exists_outside_the_stub_modules``.

    OWNED BY: Tier 7C (which deletes five of these modules outright) and the
    term slices, whose Section 31 step 5 removes each term's own stub warning.
    """
    text = (JONES_ROOT / module_name).read_text(encoding="utf-8")
    assert (
        "Stub implementation: returns identity matrix. TODO: implement properly."
        in (text)
    )
    assert "TODO: implement properly" in text


def test_todo_markers_outside_the_stub_modules() -> None:
    """Pins the true ``TODO`` inventory of ``src/radiosim``.

    Section 5.1 states that "a repository-wide search finds **no** ``TODO``
    marker anywhere in ``src/radiosim`` outside these twelve stub modules".
    That is **not** true at the gate commit: ``cli/main.py:6``
    ("TODO: Future enhancements for v0.3.0+", present since ``be231d2``) and
    ``core/sky/registry/catalogs.py:595``
    ("TODO(scientific-coverage): ...", present since ``8372dec``) both predate
    ``ac4fe41``.  Neither is a Jones stub and neither weakens ``SCI-001``, but
    7C's residual scan (I20) must exclude them explicitly rather than assert an
    empty set and then be relaxed when it fails.

    The load-bearing half of the claim -- that the **beam** subsystem is
    TODO-free, which is why Section 19's ``SCI-003`` disposition rests on
    ``beam/TODO.md`` rather than on in-code markers -- does hold, and is
    asserted here.

    OWNED BY: Tier 7C.
    """
    carriers = {
        path.relative_to(SOURCE_ROOT).as_posix()
        for path in sorted(SOURCE_ROOT.rglob("*.py"))
        if "TODO" in path.read_text(encoding="utf-8")
    }
    stub_carriers = {f"core/jones/{name}" for name in STUB_MODULES}
    assert carriers == stub_carriers | {
        "cli/main.py",
        "core/sky/registry/catalogs.py",
    }

    # Every stub carrier uses the marker phrase; neither outsider does.
    for name in ("cli/main.py", "core/sky/registry/catalogs.py"):
        assert "TODO: implement properly" not in (SOURCE_ROOT / name).read_text(
            encoding="utf-8"
        )
    assert not any(
        path.name.endswith(".py") and "TODO" in path.read_text(encoding="utf-8")
        for path in (JONES_ROOT / "beam").rglob("*.py")
    )


def test_stub_constructors_discard_physically_meaningful_parameters() -> None:
    """Pins defect D2: real physics can be handed in and silently dropped.

    A caller supplying a TEC map, D-terms, a gain sigma, a bandpass table, a
    feed-angle offset or an elevation array gets no error, no warning, and no
    effect.  This is the concrete harm ``SCI-001`` names, and it is materially
    worse than "returns identity".

    OWNED BY: Tier 7C (the classes whose constructors vanish) and Tier 7D
    through Tier 7G (the classes that start honoring their arguments).
    """
    import radiosim.core.jones as jones_package

    tec = np.array([1.0e17, 2.0e17])
    ionosphere = jones_package.IonosphereJones(tec=tec, frequencies=_stub_freqs())
    assert not any(
        np.array_equal(np.asarray(value), tec)
        for value in vars(ionosphere).values()
        if isinstance(value, np.ndarray)
    )
    assert not hasattr(ionosphere, "tec")

    elevations = np.array([0.5, 0.9])
    troposphere = jones_package.TroposphereJones(n_antennas=2, elevations=elevations)
    assert troposphere.elevations is elevations  # stored ...
    np.testing.assert_array_equal(
        np.asarray(troposphere.compute_jones(0, 0, 0, 0, get_backend("numpy"))),
        IDENTITY,  # ... and never read
    )

    gains = jones_package.GainJones(n_antennas=2, gain_sigma=0.35, seed=7)
    assert gains.gain_sigma == 0.35
    np.testing.assert_array_equal(
        np.asarray(gains.compute_jones(0, None, 0, 0, get_backend("numpy"))), IDENTITY
    )

    bandpass = jones_package.BandpassJones(
        2, _stub_freqs(), bandpass_gains=np.array([0.5, 2.0])
    )
    assert not hasattr(bandpass, "bandpass_gains")

    leakage = jones_package.PolarizationLeakageJones(
        2, d_terms=np.array([0.1 + 0.2j, 0.3])
    )
    assert not hasattr(leakage, "d_terms")

    parallactic = jones_package.ParallacticAngleJones(
        antenna_latitudes=np.array([-0.536]),
        source_positions=np.array([[1.0, -0.5]]),
        times=np.array([0.0]),
        feed_angle_offset=np.array([0.7]),
    )
    assert not hasattr(parallactic, "feed_angle_offset")


def test_capability_flags_are_self_reported_and_vacuously_true() -> None:
    """Pins defect D10: unverified capability hints about identity matrices.

    ``FaradayRotationJones`` and ``WPhaseJones`` claim unitarity, and
    ``WPhaseJones`` and ``ArrayFactorJones`` claim scalarity, about a matrix
    that is the 2x2 identity.  Each claim is true only because the identity is
    trivially unitary and trivially scalar, which is exactly the vacuity the
    Tier 5H review adjudicated as ``SCI-001`` material.

    OWNED BY: Tier 7B, which adds the flag-verification harness (D10, I2), and
    Tier 7C, which deletes ``F``, ``W`` and ``a`` outright.
    """
    import radiosim.core.jones as jones_package

    backend = get_backend("numpy")

    faraday = jones_package.FaradayRotationJones(rotation_measure=50.0)
    assert faraday.is_unitary() is True
    np.testing.assert_array_equal(
        np.asarray(faraday.compute_jones(0, 0, 0, 0, backend)), IDENTITY
    )

    wphase = jones_package.WPhaseJones()
    assert wphase.is_unitary() is True
    assert wphase.is_scalar() is True
    np.testing.assert_array_equal(
        np.asarray(wphase.compute_jones(0, 0, 0, 0, backend)), IDENTITY
    )

    array_factor = jones_package.ArrayFactorJones()
    assert array_factor.is_scalar() is True

    # The base-class defaults are equally unverified: they are plain returns.
    assert JonesTerm.is_diagonal(faraday) is False
    assert JonesTerm.is_scalar(faraday) is False
    assert JonesTerm.is_unitary(wphase) is False
    assert set(faraday.get_config()) == {
        "name",
        "is_direction_dependent",
        "is_time_dependent",
        "is_frequency_dependent",
        "is_diagonal",
        "is_scalar",
        "is_unitary",
    }


# =========================================================================
# Section 5.3 -- the chain and the evaluation contract
# =========================================================================


class _CountingIdentityTerm(JonesTerm):
    """A direction-dependent probe that counts scalar ``compute_jones`` calls."""

    def __init__(self) -> None:
        self.calls: list[tuple[int, int | None, int, int]] = []

    @property
    def name(self) -> str:
        return "probe"

    @property
    def is_direction_dependent(self) -> bool:
        return True

    def compute_jones(
        self,
        antenna_idx: int,
        source_idx: int | None,
        freq_idx: int,
        time_idx: int,
        backend: Any,
        **kwargs: Any,
    ) -> Any:
        self.calls.append((antenna_idx, source_idx, freq_idx, time_idx))
        return backend.xp.eye(2, dtype=np.complex64)


def test_jones_term_contract_is_scalar_per_direction_with_a_python_loop() -> None:
    """Pins defect D5: one direction at a time, by integer index.

    ``compute_jones`` takes ``source_idx: int | None`` and the default
    ``compute_jones_all_sources`` is a Python list comprehension that calls it
    once per direction.  At HEALPix scale that is one Python call per pixel.

    OWNED BY: Tier 7B, which replaces the contract with ``DirectionBatch`` and
    ``compute_jones_batch``.
    """
    signature = inspect.signature(JonesTerm.compute_jones)
    assert list(signature.parameters) == [
        "self",
        "antenna_idx",
        "source_idx",
        "freq_idx",
        "time_idx",
        "backend",
        "kwargs",
    ]
    assert not hasattr(JonesTerm, "compute_jones_batch")

    probe = _CountingIdentityTerm()
    result = probe.compute_jones_all_sources(3, 7, 1, 2, get_backend("numpy"))
    assert np.asarray(result).shape == (7, 2, 2)
    assert probe.calls == [(3, source, 1, 2) for source in range(7)]

    source = inspect.getsource(JonesTerm.compute_jones_all_sources)
    assert "for s in range(n_sources)" in source


def test_jones_chain_add_term_accepts_a_baseline_term() -> None:
    """Pins defect D7: ``add_term`` contradicts its own docstring.

    ``JonesChain``'s class docstring states "Only ``JonesTerm`` subclasses may
    be added here", but ``add_term`` performs no isinstance check, so a
    ``JonesBaselineTerm`` -- which is not a ``JonesTerm`` -- is accepted and
    then blows up inside ``compute_antenna_jones`` with an ``AttributeError``
    rather than a typed rejection.

    OWNED BY: Tier 7B, which adds the isinstance guard.
    """
    import radiosim.core.jones as jones_package

    assert "Only ``JonesTerm`` subclasses may be added here" in (
        JonesChain.__doc__ or ""
    )

    chain = JonesChain(get_backend("numpy"))
    baseline_term = jones_package.BaselineMultiplicativeJones()
    chain.add_term(baseline_term)
    assert chain.terms == [baseline_term]

    with pytest.raises(AttributeError):
        chain.compute_antenna_jones(0, None, 0, 0)


def test_jones_chain_hard_codes_complex128_for_both_identity_seeds() -> None:
    """Pins defect D8: ``PrecisionConfig`` is ignored by the chain seeds.

    Both seeds are literal ``np.complex128``, so a chain whose every term is
    ``complex64`` still produces a ``complex128`` product.  ``PrecisionConfig``
    is not even a constructor argument.

    OWNED BY: Tier 7B, which resolves the seed dtype from the precision model.
    """
    assert "precision" not in inspect.signature(JonesChain.__init__).parameters

    backend = get_backend("numpy")
    empty = JonesChain(backend)
    assert str(np.asarray(empty.compute_antenna_jones(0, None, 0, 0)).dtype) == (
        "complex128"
    )
    assert (
        str(np.asarray(empty.compute_antenna_jones_all_sources(0, 3, 0, 0)).dtype)
        == "complex128"
    )

    loaded = JonesChain(backend)
    loaded.add_term(_CountingIdentityTerm())
    assert str(np.asarray(loaded.compute_antenna_jones(0, 0, 0, 0)).dtype) == (
        "complex128"
    )

    text = _source("src/radiosim/core/jones/chain.py")
    assert text.count("dtype=np.complex128") == 2
    assert "PrecisionConfig" not in text

    # The precision model that is ignored here really does offer other dtypes.
    assert PrecisionConfig.fast().jones.get_dtype("gain") == np.complex64


def test_receptor_config_jones_hard_codes_complex128(tmp_path) -> None:
    """Pins defect D9: the C term ignores ``PrecisionConfig`` as well.

    OWNED BY: Tier 7B, which makes C and H dtype-correct.
    """
    text = _source("src/radiosim/core/jones/receptor.py")
    assert "dtype=np.complex128" in text

    import radiosim.core.jones as jones_package
    from radiosim.core.instrument_adapters import SolverInstrumentView

    instrument, _, receptors = _solver_components(tmp_path)
    assert type(instrument) is SolverInstrumentView

    term = jones_package.ReceptorConfigJones(receptors=receptors, instrument=instrument)
    matrix = term.compute_jones(0, None, 0, 0, get_backend("numpy"))
    assert str(np.asarray(matrix).dtype) == "complex128"
    # The default homogeneous-linear, zero-rotation case is exactly the identity.
    np.testing.assert_array_equal(np.asarray(matrix), IDENTITY)


def test_jones_chain_docstring_records_two_chain_orders() -> None:
    """Pins defects D11 and D12: the canonical and the undesigned order.

    The canonical Tier 5 order places ``P`` correlator-side of ``C`` (D12); the
    "extended" line places ``W`` sky-side of ``Z`` and declares the diagonal
    terms ``Kd``/``Rc`` "applied separately" (D11).

    OWNED BY: Tier 7B (which replaces the extended line) and Tier 7F (which
    corrects the ``P``/``C`` order).
    """
    docstring = JonesChain.__doc__ or ""
    assert "J_total = H @ G @ B @ D @ P @ C @ E @ T @ Z" in docstring
    assert "@ F @ T @ Z @ W" in docstring
    assert "``Kd``, ``Rc``" not in docstring
    assert "(K, Kd, Rc applied separately)" in docstring

    canonical = docstring.split("J_total = H @ G @ B @ D @ P @ C @ E @ T @ Z")[1]
    assert "K applied separately" in canonical.split("\n")[0]

    # The composition really is terms[0] @ ... @ terms[-1], reversed at
    # evaluation time, which is what makes the add order the chain order.
    chain_source = inspect.getsource(jones_chain.JonesChain.compute_antenna_jones)
    assert "for term in reversed(self.terms)" in chain_source


# =========================================================================
# Section 5.4 -- solver integration
# =========================================================================


ALL_OPTIONAL_JONES_TERMS: dict[str, dict[str, Any]] = {
    "G": {"enabled": True, "sigma": 0.4},
    "B": {"enabled": True, "bandpass_gains": [0.5, 2.0]},
    "D": {"enabled": True, "d_terms": [0.1, 0.2]},
    "P": {"enabled": True, "mount_type": "altaz"},
    "T": {"enabled": True},
    "Z": {"enabled": True, "tec": 5.0e17, "include_faraday": True},
}


def test_enabling_every_optional_jones_term_changes_no_visibility(tmp_path) -> None:
    """Pins defects D1 and D3 behaviorally: the whole optional chain is inert.

    G, B, D, P, T and Z are the only terms ``_build_jones_chain`` can reach at
    all, and enabling all six with physically meaningful parameters produces a
    **bit-identical** cube.  This is the single most direct statement of
    ``SCI-001``: a user who configures instrumental gains, a bandpass, leakage,
    parallactic rotation, a troposphere and an ionosphere gets exactly the
    unmodelled sky back.

    OWNED BY: Tier 7D through Tier 7G.  Each slice that implements one of these
    terms must flip this pin, because from then on that term changes the
    numbers -- which is invariant I7.
    """
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

    baseline = np.asarray(calculate_visibility(**kwargs, jones_config=None))
    empty = np.asarray(calculate_visibility(**kwargs, jones_config={}))
    loaded = np.asarray(
        calculate_visibility(**kwargs, jones_config=dict(ALL_OPTIONAL_JONES_TERMS))
    )

    assert float(np.max(np.abs(baseline))) > 0.0
    assert _raw_cube_digest(baseline) == _raw_cube_digest(empty)
    assert _raw_cube_digest(baseline) == _raw_cube_digest(loaded)


def test_build_jones_chain_adds_terms_in_the_uncorrected_canonical_order(
    tmp_path,
) -> None:
    """Pins the add order, and with it defect D12's observability status.

    With every optional term enabled the chain is H, G, B, D, P, C, E, T, Z --
    ``P`` correlator-side of ``C``, which Section 12 calls wrong for a circular
    receptor.  The error is unobservable today only because ``P`` is the
    identity; this pin records the *order*, so 7F's correction is a visible
    flip rather than an invisible one.

    OWNED BY: Tier 7F.
    """
    from radiosim.core.visibility import _build_jones_chain

    instrument, beam_system, receptors = _solver_components(tmp_path)
    chain = _build_jones_chain(
        get_backend("numpy"),
        dict(ALL_OPTIONAL_JONES_TERMS),
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
    assert [term.name for term in chain.terms] == [
        "H",
        "G",
        "B",
        "D",
        "P",
        "C",
        "E",
        "T",
        "Z",
    ]

    default_chain = _build_jones_chain(
        get_backend("numpy"),
        {},
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
    assert [term.name for term in default_chain.terms] == ["H", "C", "E"]


def test_production_never_supplies_a_non_empty_jones_config() -> None:
    """Pins defect D3: the single production call site passes a literal ``None``.

    ``RIMESimulator.simulate`` and ``VisibilitySimulator.simulate`` declare the
    parameter, ``core/hybrid.py`` hard-codes ``None``, and ``api/simulator.py``
    never mentions it, so no supported entry point can enable a term.

    OWNED BY: Tier 7D, which replaces the hard-coded ``None`` with the resolved
    ``jones`` model, and Tier 7C, which removes the parameter entirely.
    """
    hybrid = _source("src/radiosim/core/hybrid.py")
    assert hybrid.count("jones_config=None") == 1
    assert "jones_config" not in _source("src/radiosim/api/simulator.py")

    from radiosim.simulator.base import VisibilitySimulator
    from radiosim.simulator.rime import RIMESimulator

    for method in (
        RIMESimulator.calculate_visibilities,
        VisibilitySimulator.calculate_visibilities,
    ):
        parameter = inspect.signature(method).parameters["jones_config"]
        assert parameter.default is None

    solver_parameter = inspect.signature(calculate_visibility).parameters[
        "jones_config"
    ]
    assert solver_parameter.default is None
    assert str(solver_parameter.annotation) == "dict[str, typing.Any] | None"


def test_jones_config_is_an_untyped_dict_with_ad_hoc_rejections(tmp_path) -> None:
    """Pins the whole validation surface a typed ``jones`` section will replace.

    Three ad-hoc checks stand in for a schema: a type check, a "no beam key"
    check, and the parallactic guard.  There is no field validation of any
    kind below the top level -- ``{"G": {"enabled": "yes please"}}`` is
    accepted and ignored.

    OWNED BY: Tier 7D, which introduces the typed schema, and Tier 7C, which
    removes the parameter.
    """
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

    with pytest.raises(TypeError, match="jones_config must be a dict or None"):
        calculate_visibility(**kwargs, jones_config=[("G", True)])  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="must not contain a beam entry"):
        calculate_visibility(**kwargs, jones_config={"beam": {}})

    # Nonsense inside a term block is accepted without comment.
    nonsense = np.asarray(
        calculate_visibility(**kwargs, jones_config={"G": {"enabled": "yes please"}})
    )
    plain = np.asarray(calculate_visibility(**kwargs, jones_config=None))
    assert _raw_cube_digest(nonsense) == _raw_cube_digest(plain)


def test_reject_parallactic_rotation_guards_an_unreachable_combination(
    tmp_path,
) -> None:
    """Pins defect D17: dead defensive code behind an empty ``jones_config``.

    The guard fires only when ``jones_config["P"]["enabled"]`` is true, which no
    supported entry point can arrange, so its message can never be seen by a
    user of the public API.  Reached directly it does raise, and the exact
    string is pinned here so 7F's removal is deliberate.

    OWNED BY: Tier 7F, which deletes the guard once ``P`` is real.
    """
    from radiosim.core.receptor import UnsupportedFeedGeometryError
    from radiosim.core.visibility import _reject_parallactic_rotation

    _, _, receptors = _solver_components(
        tmp_path,
        receptors={
            "default": {"basis": "linear", "feed_rotation_deg": 30.0},
            "overrides": [],
            "output_basis": "linear",
        },
    )

    # Silent for every configuration the public API can produce.
    _reject_parallactic_rotation({}, receptors)
    _reject_parallactic_rotation({"P": {"enabled": False}}, receptors)

    with pytest.raises(UnsupportedFeedGeometryError) as excinfo:
        _reject_parallactic_rotation({"P": {"enabled": True}}, receptors)
    assert str(excinfo.value) == (
        "a non-zero feed_rotation_deg cannot be combined with an enabled "
        "parallactic-angle term until Tier 7 implements it."
    )


def test_healpix_solver_has_no_jones_config_and_no_chain() -> None:
    """Pins defect D4: the diffuse path never touches ``JonesChain``.

    A term added to the point path would silently not apply to a HEALPix sky.
    The HEALPix path builds its own constant ``H_p @ C_p`` and left-multiplies
    it onto the beam batch.

    OWNED BY: Tier 7B, which routes both solvers through
    ``evaluate_antenna_jones``.
    """
    assert (
        "jones_config" not in inspect.signature(calculate_visibility_healpix).parameters
    )

    text = _source("src/radiosim/core/visibility_healpix.py")
    assert "JonesChain" not in text
    assert "jones_config" not in text
    assert "_receptor_transforms" in text
    assert "_evaluate_beam_batch_by_antenna" in text


def test_geometric_phase_is_implemented_three_times() -> None:
    """Pins defect D6: one unused class and two inline copies.

    ``GeometricPhaseJones`` is real physics that no solver constructs, and both
    solvers compute the same formula inline, including the exact non-coplanar
    ``w * (n - 1)`` term.

    OWNED BY: Tier 7B, which extracts ``geometric_phase()`` and deletes the
    class.
    """
    import radiosim.core.jones as jones_package

    assert isinstance(jones_package.GeometricPhaseJones, type)

    point = _source("src/radiosim/core/visibility.py")
    healpix = _source("src/radiosim/core/visibility_healpix.py")
    for text in (point, healpix):
        assert "GeometricPhaseJones" not in text
        assert "backend.exp(-2j * np.pi" in text
    assert "bl_w * (n_dir - 1.0)" in point
    assert "(dir_n_xp - 1.0)" in healpix

    assert not (SOURCE_ROOT / "core" / "jones" / "evaluate.py").exists()
    assert not (SOURCE_ROOT / "core" / "jones" / "directions.py").exists()


def test_the_non_coplanar_w_contribution_is_already_exact(tmp_path) -> None:
    """Pins defect D19: an enabled ``W`` term would double-count.

    The inline geometric phase already carries ``w * (n - 1)`` exactly, in both
    solvers, so the W-term Jones class models an effect the forward model has
    always applied.  The source pin is the decisive half; the behavioural half
    below records that the ``n`` direction cosine is live -- moving the sources
    well off the phase centre changes the cube -- so the term is not a
    no-op that a later slice could reintroduce harmlessly.

    OWNED BY: Tier 7C, which deletes ``wterm.py`` rather than implementing it.
    """
    assert "bl_w * (n_dir - 1.0)" in _source("src/radiosim/core/visibility.py")
    assert "(dir_n_xp - 1.0)" in _source("src/radiosim/core/visibility_healpix.py")

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

    text = _source("src/radiosim/core/jones/wterm.py")
    assert "K_W = exp(-2πi·w·(n-1)) * I" in text


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


def test_no_jones_section_exists_in_the_configuration_schema() -> None:
    """Pins the absence of any Jones configuration surface.

    OWNED BY: Tier 7D, which adds the ``jones:`` section.
    """
    assert set(RadioSimConfig.model_fields) == {
        "instrument",
        "beams",
        "receptors",
        "baseline_selection",
        "sky_model",
        "obs_time",
        "obs_frequency",
        "visibility",
        "execution",
        "workflow",
    }
    assert "jones" not in RadioSimConfig.model_fields
    assert not (SOURCE_ROOT / "io" / "jones_config.py").exists()
    assert not (SOURCE_ROOT / "core" / "jones_terms.py").exists()
    assert not (SOURCE_ROOT / "core" / "jones_errors.py").exists()


def test_calculation_type_reaches_no_consumer() -> None:
    """Pins defect D13: a validated field that nothing reads.

    ``io/config.py`` is the only module in ``src/radiosim`` that mentions it,
    and there it appears three times: the field declaration and the two halves
    of the spherical-harmonic rejection.  No solver, simulator, resolver or
    runtime model reads it, so ``direct_sum`` is a silent no-op.

    OWNED BY: Tier 7C, which removes the field.
    """
    carriers = {
        path.relative_to(SOURCE_ROOT).as_posix()
        for path in sorted(SOURCE_ROOT.rglob("*.py"))
        if "calculation_type" in path.read_text(encoding="utf-8")
    }
    assert carriers == {"io/config.py"}

    text = _source("src/radiosim/io/config.py")
    # Three occurrences: the field declaration, the rejection's comparison, and
    # the dotted field name in the rejection payload.
    assert text.count("calculation_type") == 3

    field = VisibilityConfig.model_fields["calculation_type"]
    assert field.default == "direct_sum"

    # The honored strategy selector is a different field in a different section.
    assert "get_simulator(self._simulator_name)" in _source(
        "src/radiosim/api/simulator.py"
    )


def test_spherical_harmonic_is_rejected_with_the_tier7_promise(tmp_path) -> None:
    """Pins the exact rejection message that names this tier.

    OWNED BY: Tier 7C, which removes both the value and the message.
    """
    from tests.fixtures.configs import valid_config_mapping

    mapping = valid_config_mapping(
        tmp_path,
        visibility={"calculation_type": "spherical_harmonic"},
    )
    path = tmp_path / "spherical.yaml"
    path.write_text(yaml.safe_dump(mapping), encoding="utf-8")

    config = RadioSimConfig.model_validate(mapping)
    issues = collect_unsupported_issues(config)
    matching = [
        issue for issue in issues if issue.code == "spherical_harmonic_unsupported"
    ]
    assert len(matching) == 1
    assert matching[0].path == "visibility.calculation_type"
    assert matching[0].stage == "unsupported"
    assert matching[0].category == "unsupported"
    assert matching[0].message == (
        "spherical-harmonic calculation is not implemented until Tier 7"
    )

    with pytest.raises(Exception) as excinfo:
        load_config(path)
    assert "spherical-harmonic calculation is not implemented until Tier 7" in str(
        excinfo.value
    )


@pytest.mark.parametrize(
    "config_name",
    (
        "config.yaml",
        "receptor_circular_example.yaml",
        "hybrid_sky_example.yaml",
        "realistic_foreground_example.yaml",
    ),
)
def test_every_shipped_config_sets_calculation_type_direct_sum(
    config_name: str,
) -> None:
    """Pins the Q3 input: four shipped configs, all ``direct_sum``.

    Section 41's Q3 blocks 7C on confirming that no consumer uses the value.
    7A records the starting state so 7C's mechanical edit is auditable.

    OWNED BY: Tier 7C, which deletes the key from all four files.
    """
    data = yaml.safe_load(
        (REPO_ROOT / "configs" / config_name).read_text(encoding="utf-8")
    )
    assert data["visibility"]["calculation_type"] == "direct_sum"


def test_jones_precision_declares_exactly_eight_terms() -> None:
    """Pins defect D15: no precision field for C, H, or any extended term.

    OWNED BY: Tier 7D, which extends the precision model.
    """
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
    }
    assert set(JonesPrecision.model_fields) == expected
    assert set(JonesPrecisionInput.model_fields) == expected

    precision = PrecisionConfig.standard()
    assert precision.jones.get_dtype("gain") == np.complex128
    # ``get_dtype`` is a bare ``getattr``, so an unmodelled term is an
    # ``AttributeError`` rather than a typed rejection.
    with pytest.raises(AttributeError):
        precision.jones.get_dtype("receptor")


def test_mount_types_other_than_fixed_are_rejected() -> None:
    """Pins defect D16: no alt-az array can carry a rotated receptor yet.

    The rejection is a blanket one -- it fires for *every* non-``fixed`` mount,
    with or without a receptor rotation -- so a scientifically ordinary alt-az
    array cannot be simulated at all under a non-default mount type.  The
    capability gap is held open only by ``P`` being a stub.

    OWNED BY: Tier 7F, which replaces the blanket rejection with R15 once
    ``P`` exists.
    """
    from radiosim.core.receptor import _SUPPORTED_MOUNT_TYPE, resolve_receptors

    assert _SUPPORTED_MOUNT_TYPE == "fixed"

    # The message is written across two adjacent string literals in the source,
    # so it is pinned in halves rather than as one line-wrapped substring.  The
    # end-to-end rejection itself is owned by
    # ``tests/unit/test_core/test_receptor_resolution.py``
    # ``test_non_fixed_mount_type_is_rejected``; what 7A records here is that
    # the gate exists and that its message names this tier.
    text = " ".join(_source("src/radiosim/core/receptor.py").split())
    assert "is unsupported by Tier 5 receptors;" in text
    assert "time-dependent feed orientation requires the parallactic-angle" in text
    assert "term (Tier 7)." in text
    assert "An antenna mount type Tier 5 defers to Tier 7." in text
    assert callable(resolve_receptors)


def test_documentation_records_the_stub_surface_as_inspectable() -> None:
    """Pins defect D21: docs that become false the moment the stubs go.

    OWNED BY: Tier 7C and Tier 7J.
    """
    api_docs = " ".join(_source("docs/api/jones.rst").split())
    assert "identity scaffolds" in api_docs
    assert "A returned identity matrix is not a modeled physical effect." in api_docs
    assert "documented for development and inspection" in api_docs
    for module_name in (
        "ionosphere",
        "troposphere",
        "parallactic",
        "gain",
        "bandpass",
        "polarization_leakage",
    ):
        assert f"radiosim.core.jones.{module_name}" in api_docs

    guide = " ".join(_source("docs/user_guide/jones_matrices.rst").split())
    assert "When Tier 7 implements" in guide


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
