"""Tier 3H.2 cleanup contracts for the final public beam surface."""

from __future__ import annotations

import importlib
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from tests.support.repo_scan import PYTHON_SUFFIXES, iter_tracked_files

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DELETED_MODULES = {
    "radiosim.core.jones.beam.fits": (
        REPOSITORY_ROOT / "src/radiosim/core/jones/beam/fits/__init__.py"
    ),
    "radiosim.core.jones.beam.fits.handler": (
        REPOSITORY_ROOT / "src/radiosim/core/jones/beam/fits/handler.py"
    ),
    "radiosim.core.jones.beam.analytic.composed": (
        REPOSITORY_ROOT / "src/radiosim/core/jones/beam/analytic/composed.py"
    ),
    "radiosim.core.jones.beam.analytic.plotting": (
        REPOSITORY_ROOT / "src/radiosim/core/jones/beam/analytic/plotting.py"
    ),
}
REMOVED_NAMES = {
    "BeamManager",
    "BeamFITSHandler",
    "FITSBeamJones",
    "BeamJones",
    "AnalyticBeamJones",
    "compute_aperture_beam",
    "APERTURE_SHAPES",
    "TAPER_FUNCTIONS",
    "FEED_MODELS",
    "REFLECTOR_TYPES",
    "compute_edge_taper_from_feed",
    "feed_to_taper",
    "feed_to_farfield_numerical",
    "plot_beam_pattern",
    "plot_beam_comparison",
    "plot_beam_2d",
    "plot_feed_illumination",
}
ANALYTIC_EXPORTS = [
    "compute_u_beam",
    "airy_voltage_pattern",
    "sinc_voltage_pattern",
    "elliptical_airy_voltage_pattern",
    "uniform_taper",
    "gaussian_taper_pattern",
    "parabolic_taper",
    "parabolic_squared_taper",
    "cosine_taper",
    "corrugated_horn_illumination",
    "open_waveguide_illumination",
    "dipole_ground_plane_illumination",
    "prime_focus_angle",
    "cassegrain_angle",
    "compute_edge_angle",
    "compute_hpbw_numerical",
]


def test_exact_legacy_module_sources_are_deleted() -> None:
    for path in DELETED_MODULES.values():
        assert not path.exists(), path

    assert (REPOSITORY_ROOT / "src/radiosim/core/beam/fits.py").is_file()


@pytest.mark.parametrize("module_name", DELETED_MODULES)
def test_deleted_legacy_modules_fail_in_fresh_process(module_name: str) -> None:
    script = f"""
import importlib

try:
    importlib.import_module({module_name!r})
except ModuleNotFoundError as exc:
    assert (
        exc.name == {module_name!r}
        or {module_name!r}.startswith(exc.name + ".")
    ), (exc.name, {module_name!r})
else:
    raise AssertionError({module_name!r} + " unexpectedly remains importable")
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


def test_removed_names_are_absent_from_public_module_surfaces() -> None:
    module_names = (
        "radiosim.core",
        "radiosim.core.jones",
        "radiosim.core.jones.beam",
        "radiosim.core.jones.beam.analytic",
        "radiosim.core.jones.beam.analytic.aperture",
        "radiosim.core.jones.beam.analytic.taper",
        "radiosim.core.jones.beam.analytic.illumination",
    )
    for module_name in module_names:
        module = importlib.import_module(module_name)
        public = set(getattr(module, "__all__", ()))
        lazy = set(getattr(module, "_LAZY_EXPORTS", ()))
        discovered = set(dir(module))

        assert REMOVED_NAMES.isdisjoint(public), (module_name, REMOVED_NAMES & public)
        assert REMOVED_NAMES.isdisjoint(lazy), (module_name, REMOVED_NAMES & lazy)
        assert REMOVED_NAMES.isdisjoint(discovered), (
            module_name,
            REMOVED_NAMES & discovered,
        )
        for name in REMOVED_NAMES:
            assert not hasattr(module, name), (module_name, name)


def test_analytic_public_api_is_exact_and_preserves_object_identity() -> None:
    from radiosim.core.jones.beam import analytic
    from radiosim.core.jones.beam.analytic import (
        aperture,
        illumination,
        numerical_hpbw,
        taper,
    )

    defining_modules = (aperture, taper, illumination, numerical_hpbw)
    assert analytic.__all__ == ANALYTIC_EXPORTS
    for name in ANALYTIC_EXPORTS:
        defining_module = next(
            module for module in defining_modules if hasattr(module, name)
        )
        assert getattr(analytic, name) is getattr(defining_module, name)


def test_analytic_public_api_uses_no_type_suppression() -> None:
    initializer = (
        REPOSITORY_ROOT / "src/radiosim/core/jones/beam/analytic/__init__.py"
    ).read_text(encoding="utf-8")

    assert "pyright: ignore" not in initializer


def test_retained_aperture_primitives_cover_scalar_array_and_shape_behavior() -> None:
    from radiosim.core.jones.beam.analytic import (
        airy_voltage_pattern,
        compute_u_beam,
        elliptical_airy_voltage_pattern,
        sinc_voltage_pattern,
    )

    assert compute_u_beam(np.array(0.0), 14.0, 150e6) == pytest.approx(0.0)
    u = compute_u_beam(np.array([0.0, 0.1]), 14.0, 150e6)
    assert u.shape == (2,)
    assert u[1] > 0.0

    assert float(airy_voltage_pattern(np.array(0.0))) == pytest.approx(1.0)
    np.testing.assert_allclose(
        airy_voltage_pattern(np.array([0.0, 0.25])),
        np.array([1.0, airy_voltage_pattern(np.array(0.25))]),
    )
    assert sinc_voltage_pattern(np.array(0.0), np.array(0.0)) == pytest.approx(1.0)
    assert sinc_voltage_pattern(np.array(1.0), np.array(0.0)) == pytest.approx(0.0)

    theta = np.array([0.0, 0.1, 0.1])
    phi = np.array([0.0, 0.0, np.pi / 2.0])
    elliptical = elliptical_airy_voltage_pattern(
        theta,
        phi,
        diameter_x=14.0,
        diameter_y=10.0,
        wavelength=2.0,
    )
    assert elliptical.shape == theta.shape
    assert elliptical[0] == pytest.approx(1.0)
    assert elliptical[1] != pytest.approx(elliptical[2])


def test_retained_taper_primitives_are_normalized_for_scalars_and_arrays() -> None:
    from radiosim.core.jones.beam.analytic import (
        cosine_taper,
        gaussian_taper_pattern,
        parabolic_squared_taper,
        parabolic_taper,
        uniform_taper,
    )

    for taper_function in (
        uniform_taper,
        gaussian_taper_pattern,
        parabolic_taper,
        parabolic_squared_taper,
        cosine_taper,
    ):
        assert float(taper_function(0.0)) == pytest.approx(1.0)
        values = taper_function(np.array([0.0, 0.25]))
        assert values.shape == (2,)
        assert values[0] == pytest.approx(1.0)
        assert np.all(np.isfinite(values))


def test_retained_illumination_and_reflector_primitives_are_numerically_sane() -> None:
    from radiosim.core.jones.beam.analytic import (
        cassegrain_angle,
        compute_edge_angle,
        corrugated_horn_illumination,
        dipole_ground_plane_illumination,
        open_waveguide_illumination,
        prime_focus_angle,
    )

    angles = np.array([0.0, 0.1])
    assert corrugated_horn_illumination(angles)[0] == pytest.approx(1.0)
    e_plane, h_plane = open_waveguide_illumination(angles)
    assert e_plane[0] == pytest.approx(1.0)
    assert h_plane[0] == pytest.approx(1.0)
    assert dipole_ground_plane_illumination(angles)[0] == pytest.approx(1.0)

    rho = np.array([0.0, 1.0])
    prime = prime_focus_angle(rho, focal_length=5.0)
    cassegrain = cassegrain_angle(rho, focal_length=5.0, magnification=2.0)
    assert prime[0] == pytest.approx(0.0)
    assert cassegrain[0] == pytest.approx(0.0)
    assert 0.0 < cassegrain[1] < prime[1]
    assert compute_edge_angle(14.0, 5.0, "prime_focus") == pytest.approx(
        prime_focus_angle(np.array([7.0]), 5.0)[0]
    )
    assert compute_edge_angle(14.0, 5.0, "cassegrain", 2.0) == pytest.approx(
        cassegrain_angle(np.array([7.0]), 5.0, 2.0)[0]
    )


def test_compute_hpbw_numerical_remains_importable_and_finite() -> None:
    from radiosim.core.jones.beam.analytic import (
        compute_hpbw_numerical,
        uniform_taper,
    )

    hpbw = compute_hpbw_numerical(
        uniform_taper,
        np.array([100e6, 150e6]),
        diameter=14.0,
        n_samples=2000,
    )
    assert hpbw.shape == (2,)
    assert np.all(np.isfinite(hpbw))
    assert np.all(hpbw > 0.0)
    assert hpbw[1] < hpbw[0]


def test_numeric_imports_are_pure_and_do_not_restore_deleted_modules() -> None:
    script = f"""
import importlib
import sys

analytic = importlib.import_module("radiosim.core.jones.beam.analytic")
aperture = importlib.import_module("radiosim.core.jones.beam.analytic.aperture")
taper = importlib.import_module("radiosim.core.jones.beam.analytic.taper")
illumination = importlib.import_module(
    "radiosim.core.jones.beam.analytic.illumination"
)
hpbw = importlib.import_module("radiosim.core.jones.beam.analytic.numerical_hpbw")

for name, module in (
    ("compute_u_beam", aperture),
    ("uniform_taper", taper),
    ("corrugated_horn_illumination", illumination),
    ("compute_hpbw_numerical", hpbw),
):
    assert getattr(analytic, name) is getattr(module, name)

deleted = set({tuple(DELETED_MODULES)!r})
assert deleted.isdisjoint(sys.modules), deleted & set(sys.modules)
for forbidden in (
    "matplotlib",
    "pyuvdata",
    "jax",
    "numba",
    "requests",
    "httpx",
    "urllib3",
    "webbrowser",
):
    assert forbidden not in sys.modules, (forbidden, sorted(sys.modules))
assert not any(name.startswith("radiosim.devices") for name in sys.modules)
assert not any(name.startswith("radiosim.backends") for name in sys.modules)
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


def test_jones_source_has_no_legacy_import_or_definition_residue() -> None:
    source_paths = (
        REPOSITORY_ROOT / "src/radiosim/core/__init__.py",
        *iter_tracked_files(
            REPOSITORY_ROOT / "src/radiosim/core/jones", suffixes=PYTHON_SUFFIXES
        ),
    )
    deleted_imports = (
        "radiosim.core.jones.beam.fits",
        "radiosim.core.jones.beam.analytic.composed",
        "radiosim.core.jones.beam.analytic.plotting",
    )
    removed_word_pattern = re.compile(
        r"\b(?:"
        + "|".join(
            re.escape(name) for name in sorted(REMOVED_NAMES - {"REFLECTOR_TYPES"})
        )
        + r")\b"
    )

    for path in source_paths:
        text = path.read_text(encoding="utf-8")
        assert not any(module_name in text for module_name in deleted_imports), path
        assert removed_word_pattern.search(text) is None, path
        assert re.search(r"(?<!_)REFLECTOR_TYPES\b", text) is None, path


# Tier 5G decision 6: the illumination and receptor vocabularies are disjoint,
# enforced by name.  ``core/beam/`` owns illumination/taper/edge-angle;
# ``core/receptor.py`` owns receptor/feed/basis/feed_rotation.  Neither file may
# use the other's words, or "feed" silently means two different things again.
_RECEPTOR_VOCABULARY = re.compile(r"[Ff]eed|[Rr]eceptor")
_ILLUMINATION_VOCABULARY = re.compile(r"[Ii]llumination|[Tt]aper|edge_angle")


def test_beam_analytic_source_uses_no_receptor_vocabulary() -> None:
    text = (REPOSITORY_ROOT / "src/radiosim/core/beam/analytic.py").read_text(
        encoding="utf-8"
    )

    assert _RECEPTOR_VOCABULARY.search(text) is None
    for renamed in ("_feed_response", "_feed_angles", "theta_feed"):
        assert renamed not in text
    for replacement in (
        "_illumination_response",
        "_illumination_edge_angles",
        "theta_illumination",
    ):
        assert replacement in text


def test_receptor_source_uses_no_illumination_vocabulary() -> None:
    text = (REPOSITORY_ROOT / "src/radiosim/core/receptor.py").read_text(
        encoding="utf-8"
    )

    assert _ILLUMINATION_VOCABULARY.search(text) is None
    assert "feed_rotation_rad" in text


def test_illumination_module_replaced_the_feed_module() -> None:
    package = REPOSITORY_ROOT / "src/radiosim/core/jones/beam/analytic"

    assert not (package / "feed.py").exists()
    illumination = (package / "illumination.py").read_text(encoding="utf-8")
    assert "theta_feed" not in illumination
    assert "theta_illumination" in illumination
    for retired in (
        "corrugated_horn_pattern",
        "open_waveguide_pattern",
        "dipole_ground_plane_pattern",
    ):
        assert retired not in illumination

    from radiosim.core.jones.beam import analytic

    assert not hasattr(analytic, "feed")
    for retired in (
        "corrugated_horn_pattern",
        "open_waveguide_pattern",
        "dipole_ground_plane_pattern",
    ):
        assert not hasattr(analytic, retired)


def test_retired_illumination_module_path_fails_in_fresh_process() -> None:
    script = """
import importlib

try:
    importlib.import_module("radiosim.core.jones.beam.analytic.feed")
except ModuleNotFoundError as exc:
    assert exc.name == "radiosim.core.jones.beam.analytic.feed", exc.name
else:
    raise AssertionError("the retired feed module is unexpectedly importable")

illumination = importlib.import_module(
    "radiosim.core.jones.beam.analytic.illumination"
)
assert illumination.corrugated_horn_illumination is not None
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
