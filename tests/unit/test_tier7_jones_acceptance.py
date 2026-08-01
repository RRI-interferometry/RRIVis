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
#: implemented, and the four Tier 7E implemented.  Everything else exported is
#: ``"planned"`` until its own slice implements it, and this set grows by
#: exactly the terms a slice made real -- which is what makes I20's eventual
#: "every term is implemented" a sequence of visible steps rather than one flip
#: at the end.
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
    assert len(IMPLEMENTED_TERM_NAMES) == 8
    assert len(_planned_term_classes()) == 5
    assert IMPLEMENTED_TERM_NAMES < term_names
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


@pytest.mark.parametrize(
    "name,term_class", _planned_term_classes(), ids=lambda value: str(value)[:40]
)
def test_a_planned_term_refuses_to_be_evaluated(name: str, term_class: type) -> None:
    """No exported class returns an identity for all inputs.

    A ``"planned"`` term inherits the base contract, which raises.  It cannot
    silently multiply by the identity, which is exactly what ``Fix.md``
    Section 16 asks for and what Section 33.2 claims is already true after 7C.
    """
    method_name = (
        "compute_baseline_factor"
        if issubclass(term_class, JonesBaselineTerm)
        else "compute_jones_batch"
    )
    base = JonesBaselineTerm if issubclass(term_class, JonesBaselineTerm) else JonesTerm
    # The planned term does not override the raising base contract ...
    assert getattr(term_class, method_name) is getattr(base, method_name)
    # ... and constructing one is possible, but evaluating one is not.
    with pytest.raises(NotImplementedError) as excinfo:
        getattr(term_class(), method_name)(
            **_contract_kwargs(method_name),
        )
    assert method_name in str(excinfo.value)
    assert term_class.__name__ in str(excinfo.value)


def _contract_kwargs(method_name: str) -> dict[str, object]:
    import numpy as np

    from radiosim.backends import get_backend
    from radiosim.core.jones import DirectionBatch

    values = np.linspace(0.2, 1.2, 3)
    directions = DirectionBatch(
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
    common: dict[str, object] = {
        "directions": directions,
        "frequency_hz": 1.5e8,
        "freq_idx": 0,
        "time_mjd": 60_000.0,
        "time_idx": 0,
        "backend": get_backend("numpy"),
        "dtype": np.complex128,
    }
    if method_name == "compute_baseline_factor":
        return {"baseline_idx": 0, "antenna_p": 0, "antenna_q": 1, **common}
    return {"antenna_idx": 0, **common}


@pytest.mark.parametrize(
    "name,term_class", _planned_term_classes(), ids=lambda value: str(value)[:40]
)
def test_a_planned_term_declares_no_unverifiable_capability_flag(
    name: str, term_class: type
) -> None:
    """A flag that cannot be swept is a claim about numbers with no numbers.

    Defect D10 was terms declaring unitarity and scalarity about a matrix that
    was the 2x2 identity.  A ``"planned"`` term cannot be evaluated, so
    invariant I2's sweep cannot verify any flag it declares; it therefore
    declares none, and each term slice adds its flags together with its physics
    and its own I2 case (Section 31 steps 3-5).
    """
    base = JonesBaselineTerm if issubclass(term_class, JonesBaselineTerm) else JonesTerm
    for flag in ("is_diagonal", "is_scalar", "is_unitary", "is_frequency_dependent"):
        declared = getattr(term_class, flag, None)
        inherited = getattr(base, flag, None)
        assert declared is inherited, f"{name}.{flag} is declared but unverifiable"


@pytest.mark.parametrize(
    "name,term_class", _planned_term_classes(), ids=lambda value: str(value)[:40]
)
def test_a_planned_term_accepts_no_physics_it_would_discard(
    name: str, term_class: type
) -> None:
    """Defect D2, closed for every surviving term.

    The stub constructors took a TEC map, D-terms, a gain sigma, a bandpass
    table and a feed-angle offset, stored some of them, read none of them, and
    reported nothing.  A planned term takes no parameters at all, so there is
    no argument left for it to swallow.  Each term slice introduces its real
    constructor together with the resolution that validates it.
    """
    # No constructor of its own at all: nothing to store, nothing to drop.
    assert "__init__" not in vars(term_class), name
    assert term_class.__init__ is object.__init__, name

    term = term_class()
    assert vars(term) == {}, name

    # And every physics keyword the old stubs swallowed is now a TypeError.
    for keyword in ("tec", "d_terms", "gain_sigma", "delays", "elevations"):
        with pytest.raises(TypeError):
            term_class(**{keyword: 1.0})


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
