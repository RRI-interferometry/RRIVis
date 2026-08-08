"""WP-4 analytic oracles for the SCI-006 east-X convention ruling.

These tests pin the convention facts and closed-form basis transformation that
WP-5 must implement.  They deliberately do not assert that the current Jones
chain already applies the transformation: doing so here would prematurely make
the WP-5 behavioural change.
"""

from __future__ import annotations

import numpy as np

from radiosim.core.polarization import stokes_to_coherency
from radiosim.core.polarization_basis import CORRELATION_LABELS

# The sky-linear brightness matrix is ordered (North, East).  RadioSim declares
# its zero-rotation linear output feeds as (X=east, Y=north), so the ideal
# feed-by-sky response is this permutation.  It is kept test-local until WP-5
# gives the mapping one production owner in polarization_basis.py.
SKY_NORTH_EAST_TO_EAST_X_NORTH_Y = np.array(
    [[0.0, 1.0], [1.0, 0.0]],
    dtype=np.complex128,
)


def _local_stokes_from_linear_visibility(
    visibility: np.ndarray,
) -> tuple[complex, complex, complex, complex]:
    xx = visibility[0, 0]
    xy = visibility[0, 1]
    yx = visibility[1, 0]
    yy = visibility[1, 1]
    return (xx + yy, xx - yy, xy + yx, -1j * (xy - yx))


def test_positive_iau_q_seen_by_east_x_has_negative_xx_minus_yy() -> None:
    """For +Q along North, an east X feed sees less power than North Y."""
    brightness = np.asarray(stokes_to_coherency(1.0, 0.6, 0.0, 0.0))
    permutation = SKY_NORTH_EAST_TO_EAST_X_NORTH_Y

    visibility = permutation @ brightness @ permutation.conj().T

    assert CORRELATION_LABELS["linear_xy"] == ("XX", "XY", "YX", "YY")
    np.testing.assert_allclose(
        visibility,
        np.array([[0.2, 0.0], [0.0, 0.8]], dtype=np.complex128),
        rtol=0.0,
        atol=1e-15,
    )
    np.testing.assert_allclose(
        visibility[0, 0] - visibility[1, 1],
        -0.6 + 0.0j,
        rtol=0.0,
        atol=1e-15,
    )


def test_east_x_axis_permutation_flips_q_and_v_but_not_i_or_u() -> None:
    """The ruled basis change maps local ``(I,Q,U,V)`` to ``(I,-Q,U,-V)``."""
    stokes = (5.0, 1.25, -0.75, 0.5)
    brightness = np.asarray(stokes_to_coherency(*stokes))
    permutation = SKY_NORTH_EAST_TO_EAST_X_NORTH_Y

    visibility = permutation @ brightness @ permutation.conj().T
    observed = _local_stokes_from_linear_visibility(visibility)

    np.testing.assert_allclose(
        observed,
        (stokes[0], -stokes[1], stokes[2], -stokes[3]),
        rtol=0.0,
        atol=1e-15,
    )


def test_east_x_axis_permutation_leaves_unpolarized_brightness_bit_identical() -> None:
    """A pure-I brightness matrix is invariant under the ruled permutation."""
    brightness = np.asarray(stokes_to_coherency(3.0, 0.0, 0.0, 0.0))
    permutation = SKY_NORTH_EAST_TO_EAST_X_NORTH_Y

    transformed = permutation @ brightness @ permutation.conj().T

    np.testing.assert_array_equal(transformed, brightness)
