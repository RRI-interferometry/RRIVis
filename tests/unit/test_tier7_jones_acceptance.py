"""Tier 7 acceptance invariants that Tier 7C establishes.

``Tier7JonesSciencePlan.md`` Section 30 assigns this module invariants I1, I14,
I16, I18 and I20, plus the whole-tier gate assertions of Section 37.  Tier 7C
creates it with the two invariants 7C owns outright:

* **I20 -- no identity survives.**  The residual scan.  After 7C no exported
  Jones class silently multiplies by the identity: every exported term either
  carries real physics or refuses to be evaluated, and the source markers that
  advertised the identity scaffolds (``TODO: implement properly``, the
  ``Stub:`` docstring prefix, and the unconditional
  ``xp.eye(2, dtype=np.complex128)`` return) are gone from ``src/radiosim``.
* **I15 -- strategy registry equals config surface.**  The accepted values of
  ``execution.simulator`` equal the keys of the simulator registry, as a set.
  ``visibility.calculation_type`` was a second, *unread* strategy selector;
  removing it leaves exactly one, and this invariant is what makes the
  divergence structurally impossible to reintroduce (Section 18).

I1, I14, I16 and I18 land with the slices that own them (7D onward); 7K
re-asserts everything here.

I20's residual scan carries two documented exclusions, both ratified by the
7A independent-acceptance correction to Section 5.1: ``cli/main.py`` and
``core/sky/registry/catalogs.py`` each carry a ``TODO`` marker that predates
``ac4fe41`` and is neither a Jones stub nor ``SCI-001`` material.  Neither uses
the ``TODO: implement properly`` phrase, and the scan asserts that too, so the
exclusion cannot quietly widen.
"""

from __future__ import annotations

import ast
import cmath
import inspect
import math
from pathlib import Path

import pytest

from radiosim.core.jones.base import JonesTerm
from radiosim.core.jones.baseline_errors import JonesBaselineTerm

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = REPOSITORY_ROOT / "src" / "radiosim"
JONES_ROOT = SOURCE_ROOT / "core" / "jones"

#: Section 9.1's surviving public surface: 16 names, plus the three non-class
#: exports Tier 7B added (``geometric_phase``, ``DirectionBatch``,
#: ``evaluate_antenna_jones``).
SURVIVING_JONES_NAMES = (
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

#: Section 23's removal ledger, executed by Tier 7C.  ``GeometricPhaseJones``
#: left at 7B (it became the ``geometric_phase`` function) and is listed here
#: too, because an absence pin is an absence pin whichever slice created it.
REMOVED_JONES_NAMES = (
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

#: Modules deleted outright, because every class they held was deleted.
REMOVED_JONES_MODULES = ("faraday.py", "wterm.py", "element_beam.py")

#: The two pre-existing, non-Jones ``TODO`` carriers ratified by the 7A
#: acceptance correction to Section 5.1.
NON_JONES_TODO_CARRIERS = frozenset({"cli/main.py", "core/sky/registry/catalogs.py"})

#: Terms whose physics exists today: Tier 5's receptor pair, the two Tier 7D
#: implemented, the four Tier 7E implemented, Tier 7F's ``P``, and Tier 7G's
#: ``Z`` and ``T``.  Everything else exported is ``"planned"`` until its own
#: slice implements it, and this set grows by exactly the terms a slice made
#: real -- which is what makes I20's eventual "every term is implemented" a
#: sequence of visible steps rather than one flip at the end.
#:
#: After Tier 7H it contains **every** exported term of either kind: the two
#: names it gained are ``BaselineMultiplicativeJones`` and
#: ``SmearingFactorJones``, which are ``JonesBaselineTerm`` and apply by
#: Hadamard product rather than in the chain.  Nothing is planned any more, and
#: Section 37 criterion 2 is met.
IMPLEMENTED_TERM_NAMES = frozenset(
    {
        "ReceptorConfigJones",
        "BasisTransformJones",
        "GainJones",
        "BandpassJones",
        "PolarizationLeakageJones",
        "CrosshandJones",
        "DelayJones",
        "CableReflectionJones",
        "ParallacticAngleJones",
        "IonosphereJones",
        "TroposphereJones",
        "BaselineMultiplicativeJones",
        "SmearingFactorJones",
    }
)


def _source(relative: str) -> str:
    return (REPOSITORY_ROOT / relative).read_text(encoding="utf-8")


def _python_sources() -> list[Path]:
    return sorted(SOURCE_ROOT.rglob("*.py"))


def _exported_term_classes() -> list[tuple[str, type]]:
    """Every exported concrete term class, base classes excluded."""
    import radiosim.core.jones as jones_package

    bases = (JonesTerm, JonesBaselineTerm)
    classes: list[tuple[str, type]] = []
    for name in jones_package.__all__:
        value = getattr(jones_package, name)
        if not isinstance(value, type) or value in bases:
            continue
        if issubclass(value, bases):
            classes.append((name, value))
    return classes


def _term_status(term_class: type) -> str:
    """Read a class's declared ``term_status`` without constructing it.

    It is a constant-valued property on every term, so the descriptor's getter
    can be read directly -- which is what lets this check ``C`` and ``H``, whose
    constructors need a resolved receptor set, alongside the planned terms.
    """
    descriptor = inspect.getattr_static(term_class, "term_status")
    assert isinstance(descriptor, property), term_class
    assert descriptor.fget is not None
    return descriptor.fget(None)


def _planned_term_classes() -> list[tuple[str, type]]:
    """Every exported term whose physics does not exist yet."""
    return [
        (name, term_class)
        for name, term_class in _exported_term_classes()
        if name not in IMPLEMENTED_TERM_NAMES
    ]


# ---------------------------------------------------------------------------
# I20 -- no identity survives
# ---------------------------------------------------------------------------


def test_the_jones_package_exports_exactly_the_surviving_names() -> None:
    """Section 9.1's 16 names plus the three non-class exports of Tier 7B."""
    import radiosim.core.jones as jones_package

    assert tuple(jones_package.__all__) == SURVIVING_JONES_NAMES
    assert len(SURVIVING_JONES_NAMES) == 19
    term_names = {name for name, _ in _exported_term_classes()}
    assert len(term_names) == 13
    assert len(IMPLEMENTED_TERM_NAMES) == 13
    assert len(_planned_term_classes()) == 0
    assert IMPLEMENTED_TERM_NAMES == term_names
    assert term_names | {
        "JonesTerm",
        "JonesChain",
        "JonesBaselineTerm",
        "DirectionBatch",
        "evaluate_antenna_jones",
        "geometric_phase",
    } == set(SURVIVING_JONES_NAMES)


@pytest.mark.parametrize("name", REMOVED_JONES_NAMES)
def test_a_removed_jones_name_is_gone_from_every_access_path(name: str) -> None:
    """The removal is real: not exported, not lazily bound, not importable."""
    import radiosim.core.jones as jones_package

    assert name not in jones_package.__all__
    assert name not in jones_package.__dir__()
    with pytest.raises(AttributeError, match=f"has no attribute {name!r}"):
        getattr(jones_package, name)


@pytest.mark.parametrize("name", REMOVED_JONES_NAMES)
def test_a_removed_jones_name_appears_nowhere_in_the_package_source(
    name: str,
) -> None:
    """No stale import, docstring row, or lazy-table entry survives."""
    carriers = [
        path.relative_to(SOURCE_ROOT).as_posix()
        for path in _python_sources()
        if name in path.read_text(encoding="utf-8")
    ]
    assert carriers == []


@pytest.mark.parametrize("module_name", REMOVED_JONES_MODULES)
def test_a_module_whose_every_class_was_deleted_is_deleted(module_name: str) -> None:
    assert not (JONES_ROOT / module_name).exists()


def test_no_stub_marker_survives_anywhere_in_the_package() -> None:
    """Section 37 criterion 3, and the load-bearing half of I20.

    The two ``TODO`` carriers excluded here are the ones the 7A acceptance
    correction to Section 5.1 ratified; the assertion that neither uses the
    ``TODO: implement properly`` phrase is what stops the exclusion widening.
    """
    todo_carriers = set()
    for path in _python_sources():
        text = path.read_text(encoding="utf-8")
        relative = path.relative_to(SOURCE_ROOT).as_posix()
        assert "TODO: implement properly" not in text, relative
        assert "Stub:" not in text, relative
        assert "Stub implementation" not in text, relative
        assert "xp.eye(2, dtype=np.complex128)" not in text, relative
        if "TODO" in text:
            todo_carriers.add(relative)
    assert todo_carriers == NON_JONES_TODO_CARRIERS


def test_the_beam_subsystem_carries_no_in_code_todo_marker() -> None:
    """Section 19's ``SCI-003`` disposition rests on ``beam/TODO.md`` alone."""
    for path in sorted((JONES_ROOT / "beam").rglob("*.py")):
        assert "TODO" not in path.read_text(encoding="utf-8"), path


@pytest.mark.parametrize(
    "name,term_class", _exported_term_classes(), ids=lambda value: str(value)[:40]
)
def test_every_exported_term_declares_a_truthful_status(
    name: str, term_class: type
) -> None:
    """``term_status`` is ``"implemented"`` or ``"planned"``, and it is checked.

    Section 23 gives the eventual value; at 7C only ``C`` and ``H`` have
    physics, so a base-class default of ``"implemented"`` would be a lie on
    eleven classes -- exactly the vacuous-``True`` failure mode invariant I2
    exists to prevent.  The default is therefore ``"planned"``, each implemented
    term overrides it (Section 31 step 5), and Section 37 criterion 2 is the
    assertion that at 7K no ``"planned"`` remains.
    """
    status = _term_status(term_class)
    assert status in {"implemented", "planned"}
    assert (status == "implemented") is (name in IMPLEMENTED_TERM_NAMES), name


def test_no_term_is_planned_any_more() -> None:
    """Section 37 criterion 2, reached: the planned table is empty.

    FLIPPED BY: Tier 7H.  This file carried three parametrized tests over the
    planned terms -- that each refuses to be evaluated, that each declares no
    unverifiable capability flag, and that each accepts no physics keyword it
    would discard.  ``M`` and ``Q`` were the last two rows, so those three tests
    have no subject left and are replaced by this one, which asserts the state
    that made them subjectless rather than quietly collecting nothing.

    What replaces each of them is stronger, not weaker, and lives elsewhere:
    the evaluation contract is now enforced at *construction* by two
    ``@abstractmethod`` declarations (``test_term_contract.py``); the capability
    flags of every implemented term are swept numerically by invariant I2 in
    that term's own module; and every constructor now takes resolved values and
    rejects everything else, asserted per term in ``test_closure_error.py`` and
    ``test_smearing.py``.
    """
    assert _planned_term_classes() == []
    assert {name for name, _ in _exported_term_classes()} == IMPLEMENTED_TERM_NAMES

    # The contract is enforced where a subclass is written, not where it is
    # first evaluated: neither base class can be instantiated without it.
    assert "compute_jones_batch" in JonesTerm.__abstractmethods__
    assert "compute_baseline_factor" in JonesBaselineTerm.__abstractmethods__
    assert "hadamard_target" in JonesBaselineTerm.__abstractmethods__


def test_every_baseline_term_declares_where_its_factor_attaches() -> None:
    """The two Hadamard terms, and the property that keeps them apart.

    ``M`` multiplies the contracted ``(B, 2, 2)`` block and ``Q`` the kernel's
    ``(B, n_dir)`` envelope.  Declaring which is which is what lets the solvers
    dispatch without an ``isinstance`` ladder, and asserting it here is what
    stops a third baseline term from inheriting the wrong attachment point by
    accident.
    """
    import radiosim.core.jones as jones_package
    from radiosim.core.jones.baseline_errors import BASELINE_FACTOR_TARGETS

    targets = {
        "BaselineMultiplicativeJones": "correlation",
        "SmearingFactorJones": "envelope",
    }
    baseline_names = {
        name
        for name, term_class in _exported_term_classes()
        if issubclass(term_class, JonesBaselineTerm)
    }
    assert baseline_names == set(targets)
    for name, expected in targets.items():
        term_class = getattr(jones_package, name)
        descriptor = inspect.getattr_static(term_class, "hadamard_target")
        assert descriptor.fget(None) == expected
        assert expected in BASELINE_FACTOR_TARGETS


def test_no_jones_module_returns_an_unconditional_identity() -> None:
    """The structural half of I20, read from the syntax tree rather than text.

    A term that returns an identity for every input is indistinguishable from
    no term at all.  This walks every ``compute_*`` body in the package and
    asserts that none of them is a bare ``return <eye-like>``.
    """
    offenders: list[str] = []
    for path in sorted(JONES_ROOT.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef):
                continue
            if not node.name.startswith("compute_"):
                continue
            body = [item for item in node.body if not _is_docstring(item)]
            if len(body) != 1 or not isinstance(body[0], ast.Return):
                continue
            rendered = ast.unparse(body[0])
            if "eye" in rendered:
                offenders.append(f"{path.name}:{node.name}")
    assert offenders == []


def _is_docstring(node: ast.stmt) -> bool:
    return isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant)


# ---------------------------------------------------------------------------
# I15 -- strategy registry equals config surface
# ---------------------------------------------------------------------------


def test_the_accepted_simulator_values_equal_the_registry_keys() -> None:
    """I15, and the reason ``calculation_type`` could be removed rather than wired.

    ``execution.simulator`` is the one strategy selector, and it is honored:
    ``api/simulator.py`` passes it to ``get_simulator``.  Pinning the two sets
    equal makes a second, unread selector structurally impossible to
    reintroduce (Section 18).
    """
    from typing import get_args

    from radiosim.io.config import ExecutionConfig
    from radiosim.simulator import _SIMULATORS, get_simulator_names

    accepted = set(get_args(ExecutionConfig.model_fields["simulator"].annotation))
    assert accepted == set(_SIMULATORS)
    assert accepted == set(get_simulator_names())
    assert accepted == {"rime"}
    assert "get_simulator(self._simulator_name)" in _source(
        "src/radiosim/api/simulator.py"
    )


def test_calculation_type_is_absent_from_the_schema_and_the_package() -> None:
    """Section 37 criterion 15, the removal half."""
    from radiosim.io.config import RadioSimConfig, VisibilityConfig

    assert "calculation_type" not in VisibilityConfig.model_fields
    assert set(VisibilityConfig.model_fields) == {
        "sky_representation",
        "allow_lossy_point_materialization",
        "allow_lossy_point_rasterization",
    }
    assert set(
        RadioSimConfig.model_fields["visibility"].annotation.model_fields
    ) == set(VisibilityConfig.model_fields)

    # The only surviving mentions in ``src`` are prose: the removed-field
    # guidance a user with an old document sees, and the two docstrings that
    # record why the field and its rejection are gone.  None is a field, and
    # nothing in the runtime reads any of them.
    carriers = {
        path.relative_to(SOURCE_ROOT).as_posix()
        for path in _python_sources()
        if "calculation_type" in path.read_text(encoding="utf-8")
    }
    assert carriers == {"io/config.py"}
    config_source = _source("src/radiosim/io/config.py")
    assert "calculation_type: Literal" not in config_source
    assert "config.visibility.calculation_type" not in config_source
    assert '"visibility.calculation_type": (' in config_source

    # And no shipped configuration still sets it.
    for path in sorted((REPOSITORY_ROOT / "configs").glob("*.yaml")):
        assert "calculation_type" not in path.read_text(encoding="utf-8"), path


def test_the_removed_calculation_type_field_carries_the_r1_guidance(tmp_path) -> None:
    """Rejection R1, by exact string."""
    import yaml

    from radiosim.io.config import collect_schema_issues
    from tests.fixtures.configs import valid_config_mapping

    mapping = valid_config_mapping(tmp_path)
    mapping["visibility"] = dict(mapping["visibility"])
    mapping["visibility"]["calculation_type"] = "direct_sum"

    issues = collect_schema_issues(mapping)
    matching = [
        issue for issue in issues if issue.path == "visibility.calculation_type"
    ]
    assert len(matching) == 1
    assert matching[0].code == "removed_field"
    assert matching[0].message == (
        "visibility.calculation_type was removed before v1.0; the solver "
        "strategy is selected by 'execution.simulator' (currently only 'rime')."
    )

    path = tmp_path / "removed.yaml"
    path.write_text(yaml.safe_dump(mapping), encoding="utf-8")
    from radiosim.io.config import load_config

    with pytest.raises(Exception) as excinfo:
        load_config(path)
    assert "visibility.calculation_type was removed before v1.0" in str(excinfo.value)


def test_spherical_harmonic_is_no_longer_a_value_or_a_rejection() -> None:
    """The field is gone, so both halves of its rejection go with it."""
    text = _source("src/radiosim/io/config.py")
    assert '"spherical_harmonic"' not in text
    assert "spherical-harmonic calculation is not implemented until Tier 7" not in text

    from radiosim.io.config import RadioSimConfig, collect_unsupported_issues

    # The unsupported stage survives as an empty contract, with no live trigger.
    body = inspect.getsource(collect_unsupported_issues)
    body = body[body.index('"""', body.index('"""') + 3) + 3 :]
    assert "calculation_type" not in body
    assert "spherical" not in body

    fields = RadioSimConfig.model_fields["visibility"].annotation.model_fields
    for field in fields.values():
        assert "spherical_harmonic" not in str(field.annotation)


# ---------------------------------------------------------------------------
# I14 -- point/HEALPix agreement with every implemented term enabled
# ---------------------------------------------------------------------------


def test_the_two_sky_paths_agree_with_every_implemented_term_enabled(
    tmp_path,
) -> None:
    """I14: the proof that the shared evaluator really is shared.

    The point sky and the HEALPix sky differ in flux normalization by
    construction -- a pixel is not a delta function -- so the scientific claim
    is not that the two paths produce the same number.  It is that both apply
    **the same Jones factor**: with every implemented term enabled and no
    per-antenna override, each path's corrupted cube must satisfy
    ``V' = M(nu) V M(nu)^H`` for the *same* ``M``.  If a term reached one solver
    and not the other, this is where it would show, and it is exactly the defect
    (D4) the shared ``_build_jones_chain`` closes.

    ``M`` is written out here from Sections 20.1-20.6's closed forms rather than
    read back from the resolved terms (Section 29.1), so the test is an oracle
    and not a tautology.

    FLIPPED BY: Tier 7E.  Until this slice the assertion compared the *set of
    per-element ratios* between the two paths, which is only well defined while
    every enabled term is diagonal.  ``D`` mixes the two feeds, so a
    per-element ratio stopped being a single number and stopped being
    comparable between two different skies.  The matrix form is what the ratio
    form was approximating, and it is strictly stronger.
    """
    import numpy as np

    from radiosim.core.visibility import calculate_visibility
    from radiosim.core.visibility_healpix import calculate_visibility_healpix
    from tests.characterization.test_tier6_current_behavior import (
        WORKLOAD_LOCATION,
        WORKLOAD_TIME_GRID,
        _workload_healpix_model,
        _workload_point_sources,
    )
    from tests.unit.test_core.test_jones_resolution import (
        solver_components_with_jones,
    )

    # Every term that carries physics today, enabled at once.  Extended by each
    # term slice, which is what keeps "every implemented term" honest rather
    # than frozen at whatever was implemented when the test was written.
    jones = {
        "G": {"amplitude_error": 0.3, "phase_error_rad": 0.4},
        "B": {"model": {"kind": "polynomial", "coefficients": [1.0, 0.25]}},
        "Rc": {"amplitude": 0.2, "cable_delay_s": 1.5e-7, "phase_rad": 0.3},
        "Kd": {"delay_s": 4.0e-9},
        "X": {"phase_rad": 0.5, "delay_s": 2.0e-9},
        "D": {"d_terms": {"kind": "explicit", "d0": [0.05, 0.02], "d1": [-0.03, 0.04]}},
    }
    from radiosim.backends import get_backend

    backend = get_backend("numpy")

    def cubes(configuration):
        instrument, beams, receptors, terms, frequencies = solver_components_with_jones(
            tmp_path, configuration
        )
        point = np.asarray(
            calculate_visibility(
                instrument=instrument,
                beam_system=beams,
                source_arrays=_workload_point_sources(polarized=True, gaussian=False),
                location=WORKLOAD_LOCATION,
                time_grid=WORKLOAD_TIME_GRID,
                frequencies=frequencies,
                backend=backend,
                receptors=receptors,
                jones_terms=terms,
            )
        )
        diffuse = np.asarray(
            calculate_visibility_healpix(
                _workload_healpix_model(polarized=True),
                instrument=instrument,
                beam_system=beams,
                location=WORKLOAD_LOCATION,
                time_grid=WORKLOAD_TIME_GRID,
                frequencies=frequencies,
                backend=backend,
                receptors=receptors,
                jones_terms=terms,
                include_polarization=True,
            )
        )
        return point, diffuse, np.asarray(frequencies, dtype=np.float64)

    plain_point, plain_diffuse, frequencies = cubes(None)
    jones_point, jones_diffuse, _ = cubes(jones)

    def expected_jones(frequency: float) -> np.ndarray:
        """``M(nu) = G B(nu) Rc(nu) Kd(nu) X(nu) D`` from the published forms."""
        centre = 0.5 * (float(frequencies[0]) + float(frequencies[-1]))
        half_bandwidth = 0.5 * (float(frequencies[-1]) - float(frequencies[0]))
        gain = (1.0 + 0.3) * cmath.exp(1j * 0.4)
        bandpass = 1.0 + 0.25 * (frequency - centre) / half_bandwidth
        reflection = 1.0 + 0.2 * cmath.exp(
            -2j * math.pi * frequency * 1.5e-7 + 1j * 0.3
        )
        delay = cmath.exp(-2j * math.pi * frequency * 4.0e-9)
        crosshand = np.diag(
            np.array(
                [1.0, cmath.exp(1j * (0.5 + 2.0 * math.pi * frequency * 2.0e-9))],
                dtype=np.complex128,
            )
        )
        leakage = np.array(
            [[1.0, 0.05 + 0.02j], [-(-0.03 + 0.04j).conjugate(), 1.0]],
            dtype=np.complex128,
        )
        return (gain * bandpass * reflection * delay) * (crosshand @ leakage)

    for label, plain, corrupted in (
        ("point", plain_point, jones_point),
        ("healpix", plain_diffuse, jones_diffuse),
    ):
        assert float(np.max(np.abs(plain))) > 0.0, label
        for index in range(frequencies.size):
            matrix = expected_jones(float(frequencies[index]))
            np.testing.assert_allclose(
                corrupted[:, :, index, :, :],
                np.einsum(
                    "ij,tbjk,lk->tbil",
                    matrix,
                    plain[:, :, index, :, :],
                    matrix.conjugate(),
                ),
                rtol=1e-11,
                atol=1e-18,
                err_msg=label,
            )


def test_both_sky_paths_carry_the_direction_dependent_term(tmp_path) -> None:
    """I14 for ``P``, which the factorized form above cannot express.

    The test above proves the shared evaluator by factoring the corrupted cube
    as ``V' = M V M^H`` for one direction-independent ``M``.  ``P`` is
    direction-*dependent*, so no such ``M`` exists: the rotation differs per
    source and the sum over sources does not factor.  A test that added ``P`` to
    that oracle would have to widen a tolerance until the assertion stopped
    saying anything, which is the failure mode Section 29.1 names.

    What is asserted instead is the property I14 exists for -- that a configured
    term reaches *both* solvers and reaches them identically -- through three
    statements that hold on each path independently:

    1. enabling ``P`` on an alt-az array changes the cube (so the term reached
       that path at all);
    2. with an unpolarized sky and one mount for the whole array it does *not*
       change the cube (so both antennas received the same rotation, which is
       what a shared evaluator keyed by instrument row guarantees and what a
       per-solver copy would be free to get wrong);
    3. with two different mounts it changes the cube even for an unpolarized
       sky (so the per-antenna mount really is per antenna on both paths).

    Together those pin the same defect D4 the factorized form does, for the
    class of term the factorized form cannot reach.
    """
    import numpy as np

    from radiosim.backends import get_backend
    from radiosim.core.visibility import calculate_visibility
    from radiosim.core.visibility_healpix import calculate_visibility_healpix
    from tests.characterization.test_tier6_current_behavior import (
        WORKLOAD_LOCATION,
        WORKLOAD_TIME_GRID,
        _workload_healpix_model,
        _workload_point_sources,
    )
    from tests.unit.test_core.test_jones_resolution import (
        solver_components_with_jones,
    )

    backend = get_backend("numpy")

    def cubes(configuration, mount_types, *, polarized):
        instrument, beams, receptors, terms, frequencies = solver_components_with_jones(
            tmp_path, configuration, mount_types=mount_types
        )
        point = np.asarray(
            calculate_visibility(
                instrument=instrument,
                beam_system=beams,
                source_arrays=_workload_point_sources(
                    polarized=polarized, gaussian=False
                ),
                location=WORKLOAD_LOCATION,
                time_grid=WORKLOAD_TIME_GRID,
                frequencies=frequencies,
                backend=backend,
                receptors=receptors,
                jones_terms=terms,
            )
        )
        diffuse = np.asarray(
            calculate_visibility_healpix(
                _workload_healpix_model(polarized=polarized),
                instrument=instrument,
                beam_system=beams,
                location=WORKLOAD_LOCATION,
                time_grid=WORKLOAD_TIME_GRID,
                frequencies=frequencies,
                backend=backend,
                receptors=receptors,
                jones_terms=terms,
                include_polarization=polarized,
            )
        )
        return point, diffuse

    enabled = {"P": {"enabled": True}}

    polarized_absent = cubes(None, "fixed", polarized=True)
    polarized_present = cubes(enabled, "alt-az", polarized=True)
    plain_absent = cubes(None, "fixed", polarized=False)
    plain_present = cubes(enabled, "alt-az", polarized=False)
    mixed_present = cubes(enabled, ("alt-az", "fixed"), polarized=False)

    for index, label in ((0, "point"), (1, "healpix")):
        scale = float(np.max(np.abs(polarized_absent[index])))
        assert scale > 0.0, label

        # 1 -- the term reached this path.
        moved = float(
            np.max(np.abs(polarized_present[index] - polarized_absent[index]))
        )
        assert moved / scale > 1e-10, label

        # 2 -- and it is the *same* rotation on both antennas.
        plain_scale = float(np.max(np.abs(plain_absent[index])))
        np.testing.assert_allclose(
            plain_present[index],
            plain_absent[index],
            rtol=1e-12,
            atol=1e-14 * plain_scale,
            err_msg=label,
        )

        # 3 -- unless the two antennas carry different mounts.
        heterogeneous = float(
            np.max(np.abs(mixed_present[index] - plain_absent[index]))
        )
        assert heterogeneous / plain_scale > 1e-6, label


def test_both_sky_paths_carry_the_two_propagation_terms(tmp_path) -> None:
    """I14 for ``T`` and ``Z``, the other two direction-dependent terms.

    Like ``P``, neither can join the factorized ``V' = M V M^H`` oracle above:
    ``T``'s delay and opacity and ``Z``'s dispersive phase all vary per
    direction, so no single ``M`` exists and a test that invented one would have
    to widen a tolerance until it stopped saying anything (Section 29.1).

    What is asserted instead is the property I14 exists for, in the form each
    term's physics makes checkable on *both* paths independently:

    1. ``Z``'s Faraday rotation changes a polarized cube and leaves an
       unpolarized one untouched -- a rotation shared by two antennas is
       ``F C F^H``, which moves ``(Q, U)`` and nothing else;
    2. ``Z``'s dispersive phase with a *gradient* screen changes the cube, while
       the same screen with no gradient leaves it exactly unchanged -- an
       antenna-common scalar phase cancels source by source, which is why the
       gradient model exists;
    3. ``T``'s opacity attenuates both paths by the same power factor.

    Each statement holds separately on the point path and on the HEALPix path,
    which is what a shared evaluator guarantees and a per-solver copy would be
    free to get wrong.
    """
    import numpy as np

    from radiosim.backends import get_backend
    from radiosim.core.visibility import calculate_visibility
    from radiosim.core.visibility_healpix import calculate_visibility_healpix
    from tests.characterization.test_tier6_current_behavior import (
        WORKLOAD_LOCATION,
        WORKLOAD_TIME_GRID,
        _workload_healpix_model,
        _workload_point_sources,
    )
    from tests.unit.test_core.test_jones_resolution import (
        solver_components_with_jones,
    )

    backend = get_backend("numpy")

    def cubes(configuration, *, polarized):
        instrument, beams, receptors, terms, frequencies = solver_components_with_jones(
            tmp_path, configuration
        )
        point = np.asarray(
            calculate_visibility(
                instrument=instrument,
                beam_system=beams,
                source_arrays=_workload_point_sources(
                    polarized=polarized, gaussian=False
                ),
                location=WORKLOAD_LOCATION,
                time_grid=WORKLOAD_TIME_GRID,
                frequencies=frequencies,
                backend=backend,
                receptors=receptors,
                jones_terms=terms,
            )
        )
        diffuse = np.asarray(
            calculate_visibility_healpix(
                _workload_healpix_model(polarized=polarized),
                instrument=instrument,
                beam_system=beams,
                location=WORKLOAD_LOCATION,
                time_grid=WORKLOAD_TIME_GRID,
                frequencies=frequencies,
                backend=backend,
                receptors=receptors,
                jones_terms=terms,
                include_polarization=polarized,
            )
        )
        return point, diffuse

    faraday = {
        "Z": {
            "tec": {"kind": "constant", "vertical_tec_tecu": 0.0},
            "minimum_elevation_deg": 0.0,
            "faraday": {"rotation_measure_rad_m2": 1.5},
        }
    }
    uniform = {
        "Z": {
            "tec": {"kind": "constant", "vertical_tec_tecu": 40.0},
            "minimum_elevation_deg": 0.0,
        }
    }
    gradient = {
        "Z": {
            "tec": {
                "kind": "gradient",
                "vertical_tec_tecu": 40.0,
                "gradient_east_tecu_per_km": 0.6,
            },
            "minimum_elevation_deg": 0.0,
        }
    }
    opacity = {
        "T": {
            "zenith_delay": {"kind": "explicit"},
            "mapping_function": "simple",
            "minimum_elevation_deg": 0.0,
            "opacity": {"zenith_opacity": 0.3},
        }
    }

    polarized_absent = cubes(None, polarized=True)
    polarized_faraday = cubes(faraday, polarized=True)
    plain_absent = cubes(None, polarized=False)
    plain_faraday = cubes(faraday, polarized=False)
    plain_uniform = cubes(uniform, polarized=False)
    plain_gradient = cubes(gradient, polarized=False)
    plain_opacity = cubes(opacity, polarized=False)

    for index, label in ((0, "point"), (1, "healpix")):
        scale = float(np.max(np.abs(polarized_absent[index])))
        plain_scale = float(np.max(np.abs(plain_absent[index])))
        assert scale > 0.0 and plain_scale > 0.0, label

        # 1 -- Z's rotation reaches this path, and only through polarization.
        assert (
            float(np.max(np.abs(polarized_faraday[index] - polarized_absent[index])))
            / scale
            > 1e-10
        ), label
        assert (
            float(np.max(np.abs(plain_faraday[index] - plain_absent[index])))
            / plain_scale
            < 1e-14
        ), label

        # 2 -- Z's dispersive phase reaches it only through a gradient.
        assert (
            float(np.max(np.abs(plain_uniform[index] - plain_absent[index])))
            / plain_scale
            < 1e-14
        ), label
        assert (
            float(np.max(np.abs(plain_gradient[index] - plain_absent[index])))
            / plain_scale
            > 1e-10
        ), label

        # 3 -- T's opacity attenuates, and by the power factor rather than the
        # voltage one.
        ratios = np.abs(plain_opacity[index]) / np.where(
            np.abs(plain_absent[index]) > 1e-12 * plain_scale,
            np.abs(plain_absent[index]),
            np.inf,
        )
        finite = ratios[np.isfinite(ratios) & (ratios > 0.0)]
        assert finite.size > 0, label
        assert float(np.max(finite)) <= math.exp(-0.3) + 1e-12, label
        assert float(np.min(finite)) > math.exp(-0.6), label


# ---------------------------------------------------------------------------
# I18 -- observability is inert
# ---------------------------------------------------------------------------


def test_enabling_a_jones_term_leaves_observability_bit_identical(
    tmp_path,
) -> None:
    """I18.

    The observability product evaluates beams, not chains.  An observability
    plot that silently changed when a bandpass was configured would be a new
    instance of the same class of defect this tier exists to remove, so the
    inertness is asserted rather than assumed.
    """
    from dataclasses import fields as dataclass_fields
    from dataclasses import is_dataclass

    import numpy as np

    from radiosim.api.simulator import Simulator
    from tests.fixtures.configs import valid_config_mapping

    def plan_snapshot(jones):
        work = tmp_path / ("with" if jones else "without")
        work.mkdir(parents=True, exist_ok=True)
        data = valid_config_mapping(work)
        if jones is not None:
            data["jones"] = jones
        simulator = Simulator.from_mapping(data, base_dir=work)
        return simulator.plan_observability(channel_index=0)

    without = plan_snapshot(None)
    with_terms = plan_snapshot(
        {
            "G": {"amplitude_error": 0.4, "phase_error_rad": 1.1},
            "B": {"model": {"kind": "polynomial", "coefficients": [1.0, 0.5]}},
        }
    )

    # Every declared field, compared bit for bit -- arrays with
    # ``assert_array_equal`` and everything else with ``==``.  Field-by-field
    # rather than through one serialization, so a field that grows later is
    # covered automatically instead of being silently dropped by a snapshot
    # helper that does not know about it.
    def identical(left, right, path: str) -> None:
        if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
            np.testing.assert_array_equal(left, right, err_msg=path)
            return
        if isinstance(left, (tuple, list)):
            assert type(left) is type(right), path
            assert len(left) == len(right), path
            for index, (one, other) in enumerate(zip(left, right, strict=True)):
                identical(one, other, f"{path}[{index}]")
            return
        if is_dataclass(left) and not isinstance(left, type):
            assert type(left) is type(right), path
            for nested in dataclass_fields(left):
                identical(
                    getattr(left, nested.name),
                    getattr(right, nested.name),
                    f"{path}.{nested.name}",
                )
            return
        assert left == right, path

    names = [field.name for field in dataclass_fields(without)]
    assert names
    for name in names:
        identical(getattr(without, name), getattr(with_terms, name), name)


# ---------------------------------------------------------------------------
# Tier 7H -- the baseline-dependent Hadamard path
# ---------------------------------------------------------------------------


def _baseline_path_cubes(tmp_path, configuration):
    """Return the point and HEALPix cubes for one ``jones:`` configuration."""
    import numpy as np

    from radiosim.backends import get_backend
    from radiosim.core.visibility import calculate_visibility
    from radiosim.core.visibility_healpix import calculate_visibility_healpix
    from tests.characterization.test_tier6_current_behavior import (
        WORKLOAD_LOCATION,
        WORKLOAD_TIME_GRID,
        _workload_healpix_model,
        _workload_point_sources,
    )
    from tests.unit.test_core.test_jones_resolution import (
        solver_components_with_jones,
    )

    backend = get_backend("numpy")
    instrument, beams, receptors, terms, frequencies = solver_components_with_jones(
        tmp_path, configuration
    )
    point = np.asarray(
        calculate_visibility(
            instrument=instrument,
            beam_system=beams,
            source_arrays=_workload_point_sources(polarized=True, gaussian=False),
            location=WORKLOAD_LOCATION,
            time_grid=WORKLOAD_TIME_GRID,
            frequencies=frequencies,
            backend=backend,
            receptors=receptors,
            jones_terms=terms,
        )
    )
    diffuse = np.asarray(
        calculate_visibility_healpix(
            _workload_healpix_model(polarized=True),
            instrument=instrument,
            beam_system=beams,
            location=WORKLOAD_LOCATION,
            time_grid=WORKLOAD_TIME_GRID,
            frequencies=frequencies,
            backend=backend,
            receptors=receptors,
            jones_terms=terms,
            include_polarization=True,
        )
    )
    return point, diffuse, list(instrument.selected_pairs)


def test_both_sky_paths_carry_the_closure_error(tmp_path) -> None:
    """I14 for ``M``, in the exact form its direction independence allows.

    ``M`` multiplies the finished ``(B, 2, 2)`` block, so the oracle is an
    equality rather than a bound: on *both* paths the corrupted cube is the
    clean cube times the configured matrix, element by element, for every time
    and every channel.  A term that reached one solver and not the other would
    fail on the path it missed, which is exactly what defect D4 was.
    """
    import numpy as np

    matrix = np.array([[1.4 + 0.0j, 0.6 - 0.3j], [1.2 + 0.5j, 0.7 + 0.0j]])
    jones = {
        "M": {
            "matrix": [
                [
                    [float(matrix[row, col].real), float(matrix[row, col].imag)]
                    for col in (0, 1)
                ]
                for row in (0, 1)
            ]
        }
    }

    clean_point, clean_diffuse, _ = _baseline_path_cubes(tmp_path, None)
    dirty_point, dirty_diffuse, _ = _baseline_path_cubes(tmp_path, jones)

    for clean, dirty, label in (
        (clean_point, dirty_point, "point"),
        (clean_diffuse, dirty_diffuse, "healpix"),
    ):
        scale = float(np.max(np.abs(clean)))
        assert scale > 0.0, label
        np.testing.assert_array_equal(dirty, clean * matrix[None, None, None, :, :])
        assert float(np.max(np.abs(dirty - clean))) / scale > 1e-10, label


def test_both_sky_paths_carry_the_smearing_envelope(tmp_path) -> None:
    """I14 for ``Q``, whose factor is per baseline **and** per direction.

    No single factorized oracle exists for it -- that is what
    direction-dependent means -- so what is asserted is the property I14 exists
    for, in the form ``Q``'s physics makes checkable on both paths:

    1. it moves each path, by more than the noise floor;
    2. it leaves each path's **autocorrelations** bit-identical, because a
       zero-length baseline has neither a residual delay nor a fringe rate and
       ``numpy.sinc(0)`` is exactly one;
    3. the diffuse path, whose directions span the visible hemisphere, loses far
       more amplitude than the point path, whose two sources sit within a degree
       of the phase centre -- which is the direction dependence itself.
    """
    import numpy as np

    jones = {"Q": {"bandwidth_smearing": True, "time_smearing": True}}
    clean_point, clean_diffuse, pairs = _baseline_path_cubes(tmp_path, None)
    dirty_point, dirty_diffuse, _ = _baseline_path_cubes(tmp_path, jones)

    autos = [index for index, (p, q) in enumerate(pairs) if p == q]
    cross = [index for index, (p, q) in enumerate(pairs) if p != q]
    assert autos and cross, pairs

    losses = []
    for clean, dirty, label in (
        (clean_point, dirty_point, "point"),
        (clean_diffuse, dirty_diffuse, "healpix"),
    ):
        scale = float(np.max(np.abs(clean)))
        assert scale > 0.0, label
        assert float(np.max(np.abs(dirty - clean))) / scale > 1e-10, label
        for index in autos:
            np.testing.assert_array_equal(dirty[:, index], clean[:, index])
        losses.append(
            float(np.max(np.abs(dirty[:, cross] - clean[:, cross])))
            / float(np.max(np.abs(clean[:, cross])))
        )

    assert losses[1] > 100.0 * losses[0]


def test_the_compiled_kernel_is_untouched_by_the_hadamard_path() -> None:
    """Invariant **I16**, re-asserted at the slice Section 27 names.

    The whole structural claim of Section 15 is that ``M`` and ``Q`` attach to a
    signature that already existed: ``Q`` to the ``envelope`` argument and ``M``
    to the ``(B, 2, 2)`` return.  So the kernel must be *byte for byte* the same
    boundary it was at ``ac4fe41``: one ``backend.compile`` call site in ``src/``,
    the same six positional parameters in the same order, and no ``vmap``.
    """
    import inspect as _inspect

    from radiosim.core.contraction import baseline_contraction, baseline_contraction_for

    compile_sites = [
        path.relative_to(SOURCE_ROOT).as_posix()
        for path in _python_sources()
        if "backend.compile(" in path.read_text(encoding="utf-8")
    ]
    assert compile_sites == ["core/contraction.py"]

    signature = _inspect.signature(baseline_contraction)
    assert list(signature.parameters) == [
        "jones_p",
        "jones_q",
        "coherency",
        "phase",
        "envelope",
        "stokes_i",
        "backend",
    ]
    inner = _inspect.signature(baseline_contraction_for)
    assert list(inner.parameters) == ["backend"]

    for path in _python_sources():
        if path.name == "jax_backend.py":
            continue
        assert ".vmap(" not in path.read_text(encoding="utf-8"), path


def test_the_closure_error_does_not_move_the_accumulation(tmp_path) -> None:
    """Section 41's **Q5**, answered structurally as well as numerically.

    ``M`` multiplies the kernel's output *before* the cast to the output dtype,
    so closure errors participate at accumulation precision (Section 15.2).  The
    question Q5 asks is whether that disturbs the Tier 6 per-time assembly: it
    does not, and the evidence is that the two ``backend.stack`` accumulation
    sites per solver are still exactly two and still bracket the same lists, and
    that the multiply appears strictly between the kernel call and the cast.
    """
    for name in ("visibility.py", "visibility_healpix.py"):
        source = _source(f"src/radiosim/core/{name}")
        assert source.count("backend.stack(") == 2, name

        multiply = source.index("block = block * baseline_factors.correlation")
        kernel = source.index("block = contraction(")
        cast = source.index("freq_blocks.append(backend.asarray(block")
        assert kernel < multiply < cast, name

    # And numerically: with ``M`` enabled the cube is still assembled once per
    # time step and once per run, so a worker split changes nothing (Tier 6's
    # own invariant, re-run with a baseline term in the chain).
    import numpy as np

    from radiosim.backends import get_backend
    from radiosim.core.runtime_config import ResolvedSolverExecutionConfig
    from radiosim.core.visibility import calculate_visibility
    from tests.characterization.test_tier6_current_behavior import (
        WORKLOAD_LOCATION,
        WORKLOAD_TIME_GRID,
        _workload_point_sources,
    )
    from tests.unit.test_core.test_jones_resolution import (
        solver_components_with_jones,
    )

    jones = {
        "M": {"matrix": [[[1.2, 0.0], [0.8, 0.1]], [[0.9, -0.2], [1.1, 0.0]]]},
        "Q": {"bandwidth_smearing": True, "time_smearing": True},
    }
    instrument, beams, receptors, terms, frequencies = solver_components_with_jones(
        tmp_path, jones
    )

    def cube(workers: int):
        return np.asarray(
            calculate_visibility(
                instrument=instrument,
                beam_system=beams,
                source_arrays=_workload_point_sources(polarized=True, gaussian=False),
                location=WORKLOAD_LOCATION,
                time_grid=WORKLOAD_TIME_GRID,
                frequencies=frequencies,
                backend=get_backend("numpy"),
                receptors=receptors,
                jones_terms=terms,
                solver_execution=ResolvedSolverExecutionConfig(
                    workers=workers, executor="thread"
                ),
            )
        )

    np.testing.assert_array_equal(cube(2), cube(1))
