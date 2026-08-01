"""Tier 7B: the backend-parity harness for Jones evaluation.

``Tier7JonesSciencePlan.md`` Section 28 keeps ``Tier6HybridRuntimePlan.md``
Sections 13.4-13.5's tolerance rule unchanged and applies it per term:

===========  ===================================  ==============================
Backend      Tolerance vs NumPy                   Scope
===========  ===================================  ==============================
Dask         **bit-identical**                    full ``(T, B, F, 2, 2)`` cube
JAX-CPU      ``rtol=1e-12``, ``atol=0``           full cube
===========  ===================================  ==============================

7B owns the *harness*, not the per-term cases: :func:`assert_backend_parity` runs
one cube builder on every available backend and applies the rule, so a later
slice adds a parity case for its own term in three lines rather than
re-deriving the fixture.  A term whose parity fails is a defect in the term, not
a tolerance to widen.

The cases here cover what 7B actually restructured: the shared evaluator on the
point path, on the HEALPix path (scalar and polarized), and the extracted
geometric phase.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

from radiosim.backends import get_backend
from radiosim.backends.base import BackendNotAvailableError
from radiosim.core.visibility import calculate_visibility
from radiosim.core.visibility_healpix import calculate_visibility_healpix
from tests.characterization.test_tier6_current_behavior import (
    _WORKLOAD_FREQS,
    WORKLOAD_LOCATION,
    WORKLOAD_TIME_GRID,
    _solver_components,
    _workload_healpix_model,
    _workload_point_sources,
)

#: Section 28's tolerance rule, as data.
PARITY_TOLERANCE: dict[str, dict[str, float]] = {
    "dask": {"rtol": 0.0, "atol": 0.0},
    "jax": {"rtol": 1e-12, "atol": 0.0},
}


def _optional_backend(name: str):
    if name == "jax":
        kwargs: dict[str, Any] = {"device": "cpu"}
    elif name == "dask":
        pytest.importorskip("dask")
        kwargs = {"mode": "cpu"}
    else:
        kwargs = {}
    try:
        return get_backend(name, **kwargs)
    except BackendNotAvailableError as exc:
        pytest.skip(str(exc))


def assert_backend_parity(
    build_cube: Callable[[Any], Any],
    *,
    backend_name: str,
) -> None:
    """Assert Section 28's parity rule for one cube builder on one backend.

    Parameters
    ----------
    build_cube
        Callable taking an ``ArrayBackend`` and returning a ``(T, B, F, 2, 2)``
        cube.  A later slice's parity case is this callable with its own term
        enabled at a physically large value, so the term is what is being tested
        rather than the noise floor.
    backend_name
        ``"dask"`` or ``"jax"``.  NumPy is the reference and is always run.
    """
    reference_backend = get_backend("numpy")
    reference = np.asarray(reference_backend.to_numpy(build_cube(reference_backend)))

    backend = _optional_backend(backend_name)
    actual = np.asarray(backend.to_numpy(build_cube(backend)))

    assert actual.shape == reference.shape
    tolerance = PARITY_TOLERANCE[backend_name]
    if tolerance["rtol"] == 0.0 and tolerance["atol"] == 0.0:
        np.testing.assert_array_equal(actual, reference)
    else:
        np.testing.assert_allclose(actual, reference, **tolerance)


# ---------------------------------------------------------------------------
# The 7B cases
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend_name", ["dask", "jax"])
@pytest.mark.parametrize("polarized", [False, True])
def test_point_path_jones_evaluation_parity(
    tmp_path,
    backend_name: str,
    polarized: bool,
) -> None:
    """The shared evaluator on the point path, unpolarized and polarized."""
    instrument, beam_system, receptors = _solver_components(tmp_path)
    sources = _workload_point_sources(polarized=polarized, gaussian=False)

    def build(backend):
        return calculate_visibility(
            instrument=instrument,
            beam_system=beam_system,
            source_arrays=sources,
            location=WORKLOAD_LOCATION,
            time_grid=WORKLOAD_TIME_GRID,
            frequencies=_WORKLOAD_FREQS,
            backend=backend,
            receptors=receptors,
        )

    assert_backend_parity(build, backend_name=backend_name)


@pytest.mark.parametrize("backend_name", ["dask", "jax"])
@pytest.mark.parametrize("polarized", [False, True])
def test_healpix_path_jones_evaluation_parity(
    tmp_path,
    backend_name: str,
    polarized: bool,
) -> None:
    """The same evaluator on the diffuse path -- the half that had none before."""
    instrument, beam_system, receptors = _solver_components(tmp_path)
    sky = _workload_healpix_model(polarized=polarized)

    def build(backend):
        return calculate_visibility_healpix(
            sky,
            instrument=instrument,
            beam_system=beam_system,
            location=WORKLOAD_LOCATION,
            time_grid=WORKLOAD_TIME_GRID,
            frequencies=_WORKLOAD_FREQS,
            backend=backend,
            receptors=receptors,
            include_polarization=polarized,
        )

    assert_backend_parity(build, backend_name=backend_name)


@pytest.mark.parametrize("backend_name", ["dask", "jax"])
def test_point_path_parity_with_a_circular_receptor(
    tmp_path,
    backend_name: str,
) -> None:
    """A non-identity ``C`` and ``H``, so parity is testing the chain product.

    With the default linear receptors both terms are exactly ``I2`` and the
    chain product is the beam alone; a circular receptor reported in a linear
    basis makes both factors non-trivial and non-commuting.
    """
    instrument, beam_system, receptors = _solver_components(
        tmp_path,
        receptors={
            "default": {"basis": "circular", "feed_rotation_deg": 23.0},
            "output_basis": "linear",
        },
    )
    sources = _workload_point_sources(polarized=True, gaussian=False)

    def build(backend):
        return calculate_visibility(
            instrument=instrument,
            beam_system=beam_system,
            source_arrays=sources,
            location=WORKLOAD_LOCATION,
            time_grid=WORKLOAD_TIME_GRID,
            frequencies=_WORKLOAD_FREQS,
            backend=backend,
            receptors=receptors,
        )

    assert_backend_parity(build, backend_name=backend_name)


@pytest.mark.parametrize("backend_name", ["dask", "jax"])
def test_healpix_path_parity_with_a_circular_receptor(
    tmp_path,
    backend_name: str,
) -> None:
    """The diffuse path with the same non-identity receptor factors."""
    instrument, beam_system, receptors = _solver_components(
        tmp_path,
        receptors={
            "default": {"basis": "circular", "feed_rotation_deg": 23.0},
            "output_basis": "linear",
        },
    )
    sky = _workload_healpix_model(polarized=True)

    def build(backend):
        return calculate_visibility_healpix(
            sky,
            instrument=instrument,
            beam_system=beam_system,
            location=WORKLOAD_LOCATION,
            time_grid=WORKLOAD_TIME_GRID,
            frequencies=_WORKLOAD_FREQS,
            backend=backend,
            receptors=receptors,
            include_polarization=True,
        )

    assert_backend_parity(build, backend_name=backend_name)


# ---------------------------------------------------------------------------
# The 7D cases: one per implemented term, at a large parameter value
# ---------------------------------------------------------------------------
#
# Section 28 requires each term slice to add a parity case with **that term
# alone** enabled at a large value.  Large, deliberately: a 1% gain error would
# leave the term's own contribution near the floating-point noise floor, and a
# parity test that passes because nothing happened is a parity test of nothing.


_PARITY_GAIN = {
    "G": {
        "amplitude_error": 0.35,
        "phase_error_rad": 0.8,
        "per_antenna": [{"antenna": 1, "feed": 0, "amplitude_error": -0.4}],
        "time_model": {"kind": "sinusoidal", "depth": 0.3, "period_hours": 0.5},
    }
}

_PARITY_BANDPASS = {
    "B": {
        "model": {
            "kind": "polynomial",
            "coefficients": [[1.0, 0.0], [0.4, 0.2], [-0.3, 0.0]],
        },
        "per_antenna": [
            {
                "antenna": 0,
                "feed": 1,
                "model": {"kind": "polynomial", "coefficients": [0.5, -0.25]},
            }
        ],
    }
}


@pytest.mark.parametrize("backend_name", ["dask", "jax"])
@pytest.mark.parametrize(
    ("label", "jones"),
    [
        ("G", _PARITY_GAIN),
        ("B", _PARITY_BANDPASS),
        ("G+B", {**_PARITY_GAIN, **_PARITY_BANDPASS}),
    ],
)
def test_point_path_parity_with_a_configured_term(
    tmp_path,
    backend_name: str,
    label: str,
    jones: dict[str, Any],
) -> None:
    """Each Tier 7D term alone, and both together, on the point path."""
    from tests.unit.test_core.test_jones_resolution import (
        solver_components_with_jones,
    )

    instrument, beam_system, receptors, jones_terms, frequencies = (
        solver_components_with_jones(tmp_path, jones)
    )
    sources = _workload_point_sources(polarized=True, gaussian=False)

    def build(backend):
        return calculate_visibility(
            instrument=instrument,
            beam_system=beam_system,
            source_arrays=sources,
            location=WORKLOAD_LOCATION,
            time_grid=WORKLOAD_TIME_GRID,
            frequencies=frequencies,
            backend=backend,
            receptors=receptors,
            jones_terms=jones_terms,
        )

    assert_backend_parity(build, backend_name=backend_name)


@pytest.mark.parametrize("backend_name", ["dask", "jax"])
@pytest.mark.parametrize(
    ("label", "jones"),
    [("G", _PARITY_GAIN), ("B", _PARITY_BANDPASS)],
)
def test_healpix_path_parity_with_a_configured_term(
    tmp_path,
    backend_name: str,
    label: str,
    jones: dict[str, Any],
) -> None:
    """The same terms on the diffuse path, through the one shared evaluator."""
    from tests.unit.test_core.test_jones_resolution import (
        solver_components_with_jones,
    )

    instrument, beam_system, receptors, jones_terms, frequencies = (
        solver_components_with_jones(tmp_path, jones)
    )
    sky = _workload_healpix_model(polarized=True)

    def build(backend):
        return calculate_visibility_healpix(
            sky,
            instrument=instrument,
            beam_system=beam_system,
            location=WORKLOAD_LOCATION,
            time_grid=WORKLOAD_TIME_GRID,
            frequencies=frequencies,
            backend=backend,
            receptors=receptors,
            jones_terms=jones_terms,
            include_polarization=True,
        )

    assert_backend_parity(build, backend_name=backend_name)
