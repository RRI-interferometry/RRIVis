"""Tier 7B: the extracted geometric phase (the K term), defect D6.

Before this slice the same formula existed three times: an exported
``GeometricPhaseJones`` class that no solver constructed, and one inline copy in
each solver.  This module owns the single surviving implementation: its value
against a reference written out here, the ``w (n - 1)`` sign (the Workstream C
deliverable), and bit-exact equality with **both** former inline copies, which
is what makes the extraction a refactor rather than a change.
"""

from __future__ import annotations

import numpy as np
import pytest

from radiosim.backends import get_backend
from radiosim.backends.base import BackendNotAvailableError
from radiosim.core.jones import geometric_phase
from radiosim.core.jones.geometric import uvw_in_wavelengths
from radiosim.core.sky.containers.constants import C_LIGHT

#: Three baselines with a deliberately non-zero ``w``.
BASELINES_M = np.array(
    [
        [100.0, 0.0, 0.0],
        [0.0, 250.0, 30.0],
        [-75.0, 40.0, -12.5],
    ],
    dtype=np.float64,
)
FREQUENCY_HZ = 150.0e6

#: Four directions, including the phase centre (zenith) itself.
ALTITUDE_RAD = np.array([np.pi / 2, 1.2, 0.8, 0.4])
AZIMUTH_RAD = np.array([0.0, 0.7, 3.1, 5.0])
DIR_L = np.cos(ALTITUDE_RAD) * np.sin(AZIMUTH_RAD)
DIR_M = np.cos(ALTITUDE_RAD) * np.cos(AZIMUTH_RAD)
DIR_N = np.sin(ALTITUDE_RAD)


def _get_optional_backend(name: str):
    if name == "jax":
        kwargs = {"device": "cpu"}
    elif name == "dask":
        pytest.importorskip("dask")
        kwargs = {"mode": "cpu"}
    else:
        kwargs = {}
    try:
        return get_backend(name, **kwargs)
    except BackendNotAvailableError as exc:
        pytest.skip(str(exc))


def _uvw(backend) -> np.ndarray:
    return uvw_in_wavelengths(
        baseline_vectors_m=backend.asarray(
            BASELINES_M,
            dtype=backend.default_real_dtype,
        ),
        wavelength_m=float(C_LIGHT) / FREQUENCY_HZ,
    )


def _phase(backend):
    return geometric_phase(
        uvw_wavelengths=_uvw(backend),
        dir_l=backend.asarray(DIR_L, dtype=backend.default_real_dtype),
        dir_m=backend.asarray(DIR_M, dtype=backend.default_real_dtype),
        dir_n=backend.asarray(DIR_N, dtype=backend.default_real_dtype),
        backend=backend,
    )


# ---------------------------------------------------------------------------
# Value, against a reference written out here
# ---------------------------------------------------------------------------


def test_geometric_phase_matches_the_written_out_expression() -> None:
    backend = get_backend("numpy")

    wavelength = float(C_LIGHT) / FREQUENCY_HZ
    expected = np.empty((3, 4), dtype=np.complex128)
    for baseline in range(3):
        u, v, w = BASELINES_M[baseline] / wavelength
        for direction in range(4):
            argument = (
                u * DIR_L[direction]
                + v * DIR_M[direction]
                + w * (DIR_N[direction] - 1.0)
            )
            expected[baseline, direction] = np.exp(-2.0j * np.pi * argument)

    np.testing.assert_allclose(
        np.asarray(_phase(backend)),
        expected,
        rtol=1e-14,
        atol=0.0,
    )


def test_the_phase_is_exactly_unity_at_the_phase_centre() -> None:
    """``n = 1`` and ``l = m = 0`` must give exactly ``1 + 0j``, on every baseline.

    This is the ``-1`` in ``w (n - 1)``: without it the zenith would carry a
    non-zero ``w`` phase and the whole cube would be referenced to the wrong
    point.
    """
    backend = get_backend("numpy")
    phase = np.asarray(
        geometric_phase(
            uvw_wavelengths=_uvw(backend),
            dir_l=np.array([0.0]),
            dir_m=np.array([0.0]),
            dir_n=np.array([1.0]),
            backend=backend,
        )
    )
    np.testing.assert_array_equal(phase[:, 0], np.ones(3, dtype=np.complex128))


def test_the_phase_sign_is_negative_for_a_positive_path_difference() -> None:
    """A direction east of the phase centre on an east-west baseline.

    ``exp(-2 pi i b.s)``: a positive ``u l`` must produce a *negative* phase
    argument.  Every delay-like Tier 7 term (``Kd``, ``Rc``, ``T``, ``Z``) has to
    match this sign (invariant I4), so it is pinned explicitly rather than left
    implicit in a cube digest.
    """
    backend = get_backend("numpy")
    uvw = uvw_in_wavelengths(
        baseline_vectors_m=np.array([[10.0, 0.0, 0.0]]),
        wavelength_m=1.0,
    )
    phase = np.asarray(
        geometric_phase(
            uvw_wavelengths=uvw,
            dir_l=np.array([0.01]),
            dir_m=np.array([0.0]),
            dir_n=np.array([1.0]),
            backend=backend,
        )
    )

    assert np.angle(phase[0, 0]) == pytest.approx(-2.0 * np.pi * 0.1)


def test_the_non_coplanar_term_is_live() -> None:
    """Changing ``w`` alone changes the phase: ``w (n - 1)`` is not dropped."""
    backend = get_backend("numpy")
    flat = np.asarray(
        geometric_phase(
            uvw_wavelengths=np.array([[100.0, 50.0, 0.0]]),
            dir_l=DIR_L,
            dir_m=DIR_M,
            dir_n=DIR_N,
            backend=backend,
        )
    )
    tilted = np.asarray(
        geometric_phase(
            uvw_wavelengths=np.array([[100.0, 50.0, 20.0]]),
            dir_l=DIR_L,
            dir_m=DIR_M,
            dir_n=DIR_N,
            backend=backend,
        )
    )

    # Identical at the phase centre, different everywhere else.
    assert flat[0, 0] == tilted[0, 0]
    assert not np.allclose(flat[0, 1:], tilted[0, 1:])


# ---------------------------------------------------------------------------
# Equality with the two former inline copies
# ---------------------------------------------------------------------------


def _former_point_solver_inline(backend) -> np.ndarray:
    """``visibility.py`` at ``e1ae149``, transcribed verbatim."""
    baseline_vectors = backend.asarray(
        BASELINES_M,
        dtype=backend.default_real_dtype,
    )
    l_dir = backend.asarray(DIR_L, dtype=backend.default_real_dtype)
    m_dir = backend.asarray(DIR_M, dtype=backend.default_real_dtype)
    n_dir = backend.asarray(DIR_N, dtype=backend.default_real_dtype)

    uvw_wavelengths = baseline_vectors / (float(C_LIGHT) / float(FREQUENCY_HZ))
    bl_u = uvw_wavelengths[:, 0:1]
    bl_v = uvw_wavelengths[:, 1:2]
    bl_w = uvw_wavelengths[:, 2:3]
    b_dot_s = bl_u * l_dir + bl_v * m_dir + bl_w * (n_dir - 1.0)
    return backend.exp(-2j * np.pi * b_dot_s)


def _former_healpix_solver_inline(backend) -> np.ndarray:
    """``visibility_healpix.py`` at ``e1ae149``, transcribed verbatim."""
    baseline_vectors = backend.asarray(
        BASELINES_M,
        dtype=backend.default_real_dtype,
    )
    dir_l_xp = backend.asarray(DIR_L, dtype=backend.default_real_dtype)
    dir_m_xp = backend.asarray(DIR_M, dtype=backend.default_real_dtype)
    dir_n_xp = backend.asarray(DIR_N, dtype=backend.default_real_dtype)

    wavelength_m = float(C_LIGHT) / float(FREQUENCY_HZ)
    uvw_wavelengths = baseline_vectors / wavelength_m
    delay = (
        uvw_wavelengths[:, 0:1] * dir_l_xp
        + uvw_wavelengths[:, 1:2] * dir_m_xp
        + uvw_wavelengths[:, 2:3] * (dir_n_xp - 1.0)
    )
    return backend.exp(-2j * np.pi * delay)


@pytest.mark.parametrize("backend_name", ["numpy", "dask", "jax"])
def test_the_extracted_function_equals_both_former_inline_copies(
    backend_name: str,
) -> None:
    """Bit-exact on every backend: the extraction changed no arithmetic."""
    backend = _get_optional_backend(backend_name)

    extracted = backend.to_numpy(_phase(backend))
    point = backend.to_numpy(_former_point_solver_inline(backend))
    healpix = backend.to_numpy(_former_healpix_solver_inline(backend))

    np.testing.assert_array_equal(extracted, point)
    np.testing.assert_array_equal(extracted, healpix)


def test_the_k_class_is_gone() -> None:
    """``GeometricPhaseJones`` was replaced by the function, not kept beside it."""
    import radiosim.core.jones as jones_package
    import radiosim.core.jones.geometric as geometric_module

    assert not hasattr(geometric_module, "GeometricPhaseJones")
    assert "GeometricPhaseJones" not in jones_package.__all__
    with pytest.raises(AttributeError):
        jones_package.GeometricPhaseJones  # noqa: B018
