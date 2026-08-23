r"""The scalar baseline transfer ``B_lm`` and its rigid-rotation law.

``docs/development/sci004_mmode_design.md`` Section 6 defines the reference-phase
response from *the same* Jones and fringe factors as the direct RIME,

.. math::

    K^X_{pqfc}(\hat n)=
    \left[J_{p,\theta\phi}(\hat n)P^X_{\theta\phi}
    J^H_{q,\theta\phi}(\hat n)\right]_c
    K_{pq}(\hat n)M_{pq}(\hat n)Q_{pq}(\hat n)H(\hat n),

with the existing geometric phase ``K`` at its accepted sign, the accepted
baseline closure factor ``M``, ``Q``'s accepted bandwidth smearing, and the
strict horizon factor

.. math::

    H(\hat n)=\mathbf 1[\operatorname{alt}(\hat n)>0].

Equality is excluded, matching both maintained direct solvers: no epsilon, beam
cutoff, or half-weight at the horizon is allowed, and ``H`` is part of the
transfer function rather than an after-the-fact time mask.  Every SCI-004 cube
therefore has ``C = 4`` in the exact resolved matrix order.

The scalar transfer this module builds is

.. math:: B^I_{pqfc,lm}=\int K^I_{pqfc}\,Y_{lm}\,d\Omega,

with the harmonic placed *unconjugated* against the conjugated sky expansion
``a_lm = integral(I conj(Y_lm) dOmega)``, so ``v = sum_lm B_lm a_lm`` reproduces
``integral(K I dOmega)`` exactly.  Section 4.1's rigid group composition then
gives ``B_lm(alpha) = B_lm(0) exp(+i m alpha)``.

Quadrature note
---------------
The integral is evaluated on Section 7.3's **iso-Gauss** grid: ``3 * nside``
Gauss-Legendre colatitude rings, each carrying ``4 * nside`` uniform azimuths
from ``phi = 0``, for exactly the ``12 * nside**2`` nodes the declared
``quadrature_nside`` names.  Two properties of the hard horizon force that
choice.  A uniform-azimuth ring kills every ``m != 0`` response exactly, which
an irregular grid does not.  And ``3 * nside`` is even for every accepted
``nside``, so no node lands on the equator, the strictly visible half of the
sphere carries exactly half of the total quadrature weight, and the hemisphere
area integrates exactly.  An equal-area HEALPix pixel-centre sum cannot: its
visible-area error is exactly ``1/(3 * nside)`` -- ``2.08e-2`` at
``nside = 16`` -- because the equatorial ring lies on the horizon and giving it
half weight is exactly what Section 6 forbids.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Final

import numpy as np

from radiosim.core.mmode.frame import strict_horizon_visible
from radiosim.core.mmode.harmonics import (
    field_columns,
    packed_conjugate_harmonics,
    packed_polarized_conjugate_harmonics,
    polarized_packed_block_table,
    scalar_packed_block_table,
)
from radiosim.core.mmode.types import (
    PolarizedPackedCube,
    PolarizedPackedTable,
    ScalarPackedCube,
    ScalarPackedTable,
)

__all__ = [
    "FRINGE_CONVENTION",
    "PolarizedBaselineTransfer",
    "ScalarBaselineTransfer",
    "bridged_stokes_matrices",
    "build_polarized_baseline_transfer",
    "build_scalar_baseline_transfer",
    "quadrature_grid",
    "receptor_component_matrices",
    "receptor_row_indices",
    "resolved_receptor_matrices",
]

#: Section 6: ``K`` is the *existing* geometric phase at its accepted sign, so
#: the m-mode transfer introduces no second fringe convention.
FRINGE_CONVENTION: Final = "existing_geometric_phase_v1"

#: Section 6: every SCI-004 cube has exactly four correlations.
N_CORRELATIONS: Final = 4

_SPEED_OF_LIGHT_M_PER_S: Final[float] = 299792458.0


def quadrature_grid(nside: int) -> tuple[np.ndarray, np.ndarray]:
    """Return the Section 7.3 **iso-Gauss** grid directions and weights.

    A transfer grid of resolution ``nside`` has ``3 * nside`` Gauss-Legendre
    colatitude rings -- the nodes and weights of the ``3 * nside``-point rule in
    ``z = cos(theta)`` on ``[-1, 1]`` -- each carrying ``4 * nside`` uniformly
    spaced azimuths starting at ``phi = 0``, for exactly ``12 * nside**2``
    nodes.  A node's quadrature weight is its Gauss-Legendre weight times
    ``2 * pi / (4 * nside)``.

    ``3 * nside`` is even for every accepted ``nside``, so no node lies on the
    horizon-critical equator, the visible hemisphere carries exactly half the
    total weight under any strict horizon through the equator, and the uniform
    azimuths annihilate every ``m != 0`` mode of an azimuthally constant
    integrand exactly.  Equal-area HEALPix pixel-centre quadrature is rejected
    as a transfer quadrature: its equatorial ring sits *on* the horizon, and its
    visible-area error under the strict ``alt > 0`` factor is exactly
    ``1/(3 * nside)`` -- ``2.08e-2`` at ``nside = 16``.

    Node enumeration is ring-major from the north pole, then ascending azimuth
    index.  Directions are returned as ``(East, North, Up)`` components, which
    are the direction cosines ``(l, m, n)`` the existing geometric phase
    consumes.
    """
    resolution = int(nside)
    if resolution < 2 or resolution & (resolution - 1):
        raise ValueError("quadrature_nside must be a power of two, at least 2")
    ring_count = 3 * resolution
    azimuth_count = 4 * resolution
    nodes, node_weights = np.polynomial.legendre.leggauss(ring_count)
    # ``leggauss`` returns ascending ``z``; the canonical enumeration is
    # ring-major *from the north pole*, so the rings are taken in descending
    # ``z`` -- ascending colatitude.
    descending = np.argsort(-nodes, kind="stable")
    nodes = nodes[descending]
    node_weights = node_weights[descending]
    azimuths = (
        2.0 * math.pi * np.arange(azimuth_count, dtype=np.float64) / azimuth_count
    )
    sin_theta = np.sqrt(np.maximum(0.0, 1.0 - nodes * nodes))
    up = np.repeat(nodes, azimuth_count)
    radial = np.repeat(sin_theta, azimuth_count)
    phi = np.tile(azimuths, ring_count)
    directions = np.stack((radial * np.cos(phi), radial * np.sin(phi), up), axis=-1)
    weights = np.repeat(node_weights, azimuth_count) * (2.0 * math.pi / azimuth_count)
    return directions, weights


def _scalar_beam_response(
    directions: np.ndarray, *, beam: str, diameter_m: float, frequency_hz: float
) -> np.ndarray:
    """Return the scalar voltage response ``e`` of one antenna on the grid."""
    if beam == "unit":
        return np.ones(directions.shape[0], dtype=np.complex128)
    if beam == "heterogeneous":
        from radiosim.core.jones.beam.analytic.aperture import (
            airy_voltage_pattern,
            compute_u_beam,
        )

        theta = np.arccos(np.clip(directions[:, 2], -1.0, 1.0))
        u_beam = compute_u_beam(theta, float(diameter_m), float(frequency_hz))
        return np.asarray(airy_voltage_pattern(u_beam), dtype=np.complex128)
    raise ValueError(f"unsupported transfer beam {beam!r}")


@dataclass(frozen=True, slots=True)
class ScalarBaselineTransfer:
    """One baseline's scalar transfer cube and the grid it was built on."""

    table: ScalarPackedTable
    blm: ScalarPackedCube
    quadrature_directions_enu: np.ndarray
    quadrature_weights: np.ndarray
    horizon_mask: np.ndarray
    fringe: np.ndarray
    correlation_labels: tuple[str, str, str, str]
    per_antenna_beam_identities: tuple[str, str]
    baseline_enu_m: tuple[float, float, float]
    frequency_hz: float

    @property
    def fringe_convention(self) -> str:
        """Return the existing geometric-phase convention identifier."""
        return FRINGE_CONVENTION

    def correlation_index(self, label: str) -> int:
        """Return the row-major index of one resolved correlation label."""
        return self.correlation_labels.index(str(label))

    def at_relative_phase(self, relative_phase_rad: float) -> ScalarBaselineTransfer:
        """Return ``B_lm(alpha) = B_lm(0) exp(+i m alpha)``.

        Section 4.1's ``T(alpha) = T(0) R3(alpha)`` makes this exact rather than
        approximate: the whole instrument is rigid in the terrestrial frame and
        the sky rotates by one parameter about the retained polar axis.
        """
        alpha = float(relative_phase_rad)
        values = np.array(self.blm.values, copy=True)
        for row in self.table.block_rows:
            order = int(row["m"])
            start = int(row["value_start"])
            stop = int(row["value_stop"])
            values[:, :, :, start:stop] *= complex(
                math.cos(order * alpha), math.sin(order * alpha)
            )
        return ScalarBaselineTransfer(
            table=self.table,
            blm=ScalarPackedCube(table=self.table, values=values),
            quadrature_directions_enu=self.quadrature_directions_enu,
            quadrature_weights=self.quadrature_weights,
            horizon_mask=self.horizon_mask,
            fringe=self.fringe,
            correlation_labels=self.correlation_labels,
            per_antenna_beam_identities=self.per_antenna_beam_identities,
            baseline_enu_m=self.baseline_enu_m,
            frequency_hz=self.frequency_hz,
        )


def build_scalar_baseline_transfer(
    *,
    beam: str,
    baseline_enu_m: Sequence[float],
    frequency_hz: float,
    lmax: int,
    mmax: int,
    quadrature_nside: int,
    antenna_diameters_m: Sequence[float] = (14.0, 14.0),
    correlation_labels: Sequence[str] = ("XX", "XY", "YX", "YY"),
    table: ScalarPackedTable | None = None,
) -> ScalarBaselineTransfer:
    """Build the Stokes-``I`` scalar transfer ``B_lm`` of one baseline.

    Parameters
    ----------
    beam : str
        ``"unit"`` for the analytic identity response, or ``"heterogeneous"``
        for a per-antenna uniformly illuminated circular aperture.
    baseline_enu_m : sequence of float
        The baseline vector in metres, as ``(East, North, Up)``.
    frequency_hz : float
        The channel frequency.
    lmax, mmax, quadrature_nside : int
        Section 7.3's declared truncation dimensions.
    antenna_diameters_m : sequence of float
        The two antenna diameters, used only by the heterogeneous beam.
    correlation_labels : sequence of str
        The resolved row-major correlation labels; exactly four are required.
    table : ScalarPackedTable, optional
        Reuse an already built block table instead of rebuilding it.

    Returns
    -------
    ScalarBaselineTransfer
        The ``(baseline, frequency, correlation, packed_value)`` cube together
        with the quadrature grid, horizon mask and fringe it was built from.
    """
    from radiosim.backends import get_backend
    from radiosim.core.jones import geometric_phase

    labels = tuple(str(label) for label in correlation_labels)
    if len(labels) != N_CORRELATIONS:
        raise ValueError(
            "an SCI-004 transfer cube has exactly four correlations; "
            "an omitted cross-hand is a rejection, not a supported subset"
        )
    resolved = (
        table if table is not None else scalar_packed_block_table(lmax=lmax, mmax=mmax)
    )
    directions, weights = quadrature_grid(quadrature_nside)

    # Section 6: ``H`` is part of the transfer function, equality at the horizon
    # is excluded, and the predicate is the one shared code object the direct
    # oracles invoke -- never a re-derivation of the same formula.
    horizon = strict_horizon_visible(directions[:, 2])

    baseline = np.asarray(baseline_enu_m, dtype=np.float64).reshape(1, 3)
    wavelength = _SPEED_OF_LIGHT_M_PER_S / float(frequency_hz)
    fringe = np.asarray(
        geometric_phase(
            uvw_wavelengths=baseline / wavelength,
            dir_l=directions[:, 0],
            dir_m=directions[:, 1],
            dir_n=directions[:, 2],
            backend=get_backend("numpy"),
        ),
        dtype=np.complex128,
    )

    diameters = tuple(float(value) for value in antenna_diameters_m)
    if len(diameters) != 2:
        raise ValueError("a baseline has exactly two antenna diameters")
    responses = [
        _scalar_beam_response(
            directions, beam=beam, diameter_m=diameter, frequency_hz=frequency_hz
        )
        for diameter in diameters
    ]
    identities = tuple(
        "unit_response_v1"
        if beam == "unit"
        else f"circular_aperture_uniform_illumination_v1:diameter_m={diameter!r}"
        for diameter in diameters
    )

    # ``P^I`` is the Stokes-I coherency ``(1/2) I_2`` (CLAUDE-normative half
    # factor), and both Jones matrices are the scalar response times ``I_2`` in
    # the M1 scalar subset, so only the two parallel hands respond.
    coherency = np.zeros((N_CORRELATIONS, directions.shape[0]), dtype=np.complex128)
    scalar_product = 0.5 * responses[0] * np.conjugate(responses[1])
    coherency[0] = scalar_product
    coherency[3] = scalar_product

    integrand = coherency * fringe[0][None, :] * horizon[None, :].astype(np.float64)

    # ``B_lm = integral(K Y_lm dOmega)``: the harmonic is unconjugated here,
    # against the conjugated sky expansion, so ``sum_lm B_lm a_lm`` reproduces
    # ``integral(K I dOmega)``.
    theta = np.arccos(np.clip(directions[:, 2], -1.0, 1.0))
    phi = np.arctan2(directions[:, 1], directions[:, 0]) % (2.0 * math.pi)
    harmonics = np.conjugate(packed_conjugate_harmonics(resolved, theta, phi))

    packed = np.empty(
        (1, 1, N_CORRELATIONS, resolved.packed_value_count), dtype=np.complex128
    )
    weighted = integrand * weights[None, :]
    for index in range(N_CORRELATIONS):
        packed[0, 0, index] = weighted[index] @ harmonics

    return ScalarBaselineTransfer(
        table=resolved,
        blm=ScalarPackedCube(table=resolved, values=packed),
        quadrature_directions_enu=directions,
        quadrature_weights=weights,
        horizon_mask=horizon,
        fringe=fringe,
        correlation_labels=(labels[0], labels[1], labels[2], labels[3]),
        per_antenna_beam_identities=(identities[0], identities[1]),
        baseline_enu_m=(
            float(baseline[0, 0]),
            float(baseline[0, 1]),
            float(baseline[0, 2]),
        ),
        frequency_hz=float(frequency_hz),
    )


# ---------------------------------------------------------------------------
# Section 6 polarized baseline transfer
# ---------------------------------------------------------------------------


#: Shaw's four ``(theta, phi)``-basis brightness component matrices.  ``P^V``
#: carries the *opposite* matrix sign from RadioSim's ``P^V_RS`` in one unchanged
#: ordered basis, which is exactly what Section 5.2's bridge reconciles: after
#: ``D`` the physical IAU ``V`` field has the same sign.
_SHAW_COMPONENTS: Final[dict[str, np.ndarray]] = {
    "I": 0.5 * np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.complex128),
    "Q": 0.5 * np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128),
    "U": 0.5 * np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128),
    "V": 0.5 * np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=np.complex128),
}


def receptor_row_indices(correlation_labels: Sequence[str]) -> dict[str, int]:
    r"""Return each receptor label's row index in the ``2 x 2`` coherency.

    The four correlations are in the existing row-major receptor-product order
    ``(00, 01, 10, 11)``, so the first label's leading character names receptor
    row ``0`` and the last label's leading character names row ``1``.  Deriving
    the pair from the resolved labels is what lets the same code serve
    ``("XX", "XY", "YX", "YY")`` and ``("RR", "RL", "LR", "LL")`` -- a
    hard-coded ``X``/``Y`` table silently rejects the circular basis the
    accepted receptor contracts support.
    """
    labels = tuple(str(label) for label in correlation_labels)
    if len(labels) != N_CORRELATIONS:
        raise ValueError(
            "an SCI-004 correlation set has exactly four row-major labels; "
            "an omitted cross-hand is a rejection, not a supported subset"
        )
    first, second = labels[0][0], labels[-1][0]
    if first == second:
        raise ValueError(
            f"correlation labels {labels} do not name two distinct receptors"
        )
    return {first: 0, second: 1}


def bridged_stokes_matrices() -> dict[str, np.ndarray]:
    r"""Return ``D P^X D`` -- Shaw's brightness matrices in the RadioSim basis.

    Section 5.2's bridge is ``J_{\theta\phi} = J_{NE} D`` with
    ``D = diag(-1, 1)``, so Section 6's reference-phase response

    .. math::

        K^X=\bigl[J_{p,\theta\phi}P^X_{\theta\phi}J^H_{q,\theta\phi}\bigr]_c

    is exactly ``[J_{p,NE} (D P^X D) J^H_{q,NE}]_c``.  These four matrices are
    therefore the *only* place the Shaw convention enters a kernel built from
    RadioSim ``(North, East)`` Jones matrices, and they are the one shared code
    object both the production kernel and the fixture composition consume.

    ``D`` is its own inverse and its own conjugate transpose, so the bridge is
    applied on both sides of the same matrix rather than as a fitted sign.

    Examples
    --------
    >>> import numpy as np
    >>> bridged = bridged_stokes_matrices()
    >>> bool(np.array_equal(bridged["I"], 0.5 * np.eye(2)))
    True
    >>> bool(np.array_equal(bridged["U"], 0.5 * np.array([[0.0, -1.0], [-1.0, 0.0]])))
    True
    """
    from radiosim.core.polarization import shaw_basis_bridge

    bridge = np.asarray(shaw_basis_bridge(), dtype=np.complex128)
    return {
        name: bridge @ matrix @ bridge.conj().T
        for name, matrix in _SHAW_COMPONENTS.items()
    }


def resolved_receptor_matrices(*, receptors: Any, instrument: Any) -> tuple[Any, ...]:
    r"""Return each solver antenna row's constant sky-basis receptor matrix.

    The canonical chain is ``J_p = H_p ... C_p E_p P_p ...`` with ``P_p`` exactly
    the identity for the shipped ``fixed`` and unspecified mounts, so for the
    accepted M2 scope an antenna's ``(North, East)`` Jones matrix is
    ``J_{p,NE} = M_p E_p(\hat n)`` with the **constant**

    .. math:: M_p = H_p C_p .

    Both factors come from the maintained direct-RIME code objects --
    :func:`~radiosim.core.jones.receptor.receptor_matrix` and
    :func:`~radiosim.core.jones.receptor.basis_transform_matrix` -- so a
    receptor, feed-rotation or output-basis change cannot mean one thing to the
    direct solver and another to the m-mode kernel.

    Section 6 anchors these cells to the direct RIME's own basis: they are
    constant coefficients acting in the celestial tangent basis of each
    direction, which preserves the integrand's spin weight and keeps the
    Section 7.3 spin-``+-2`` quadrature spectrally exact.  No tangent transport
    enters for this scope.

    Parameters
    ----------
    receptors : ResolvedReceptorSet
        The resolved array-wide receptor inventory and its one ``output_basis``.
    instrument : SolverInstrumentView
        The solver view whose antenna row order indexes the returned tuple.
    """
    from radiosim.core.jones.receptor import basis_transform_matrix, receptor_matrix

    by_number = {
        identifier.number: resolved
        for identifier, resolved in receptors.receptor_by_antenna.items()
    }
    matrices: list[np.ndarray] = []
    for number in instrument.antenna_numbers:
        resolved = by_number[int(number)]
        config = receptor_matrix(resolved.basis, float(resolved.feed_rotation_rad))
        transform = basis_transform_matrix(resolved.basis, receptors.output_basis)
        matrices.append(np.asarray(transform @ config, dtype=np.complex128))
    return tuple(matrices)


def receptor_component_matrices(
    correlation_labels: Sequence[str],
) -> dict[str, np.ndarray]:
    r"""Return ``M P^X M^H`` for the four Stokes components, ``M = C D``.

    Section 5.2 puts the accepted SCI-006 receptor permutation **inside**
    ``J_NE`` and bridges to Shaw's basis with ``J_thetaphi = J_NE D``; the
    receptor matrix ``C`` maps the IAU ``(North, East)`` sky basis to the
    resolved receptor pair.  Composing the two gives ``M = C D``, and the
    component matrices the transfer integrand needs are ``M P^X M^H``.

    These are **constant** matrices in the celestial tangent basis, which is the
    basis the Section 5.3 spin expansions use, and Section 6 anchors that to the
    accepted direct RIME: the coherency is built in the celestial North/East
    tangent basis, the direction-independent chain terms right-multiply it as
    constants there, and every mount-dependent tangent rotation belongs to the
    ``P`` term -- exactly the identity for the shipped ``fixed`` and unspecified
    mounts.  Constant coefficients on spin-weighted fields preserve the
    integrand's spin weight, which is what keeps the Section 7.3 spin-``+-2``
    quadrature spectrally exact.  Transporting a constant matrix into the
    rotating local basis instead is the inadmissible object Section 6 rejects.

    For the accepted east-X linear pair this reproduces SCI-006's ruling that
    "positive IAU ``Q`` produces negative ``XX-YY`` for east X", and it is
    derived here rather than tabulated so a receptor or bridge change cannot
    silently leave a hard-coded matrix behind.

    Examples
    --------
    >>> import numpy as np
    >>> matrices = receptor_component_matrices(("XX", "XY", "YX", "YY"))
    >>> bool(np.allclose(matrices["Q"], 0.5 * np.array([[-1.0, 0.0], [0.0, 1.0]])))
    True
    """
    from radiosim.core.polarization_basis import basis_for_correlations

    basis = basis_for_correlations(tuple(str(label) for label in correlation_labels))
    if basis == "linear_xy":
        # SCI-006: ``P`` maps ``(North, East)`` to ``(X = east, Y = north)``.
        receptor = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
    elif basis == "circular_rl":
        receptor = np.array(
            [[1.0, 1.0j], [1.0, -1.0j]], dtype=np.complex128
        ) / math.sqrt(2.0)
    else:  # pragma: no cover - the table has exactly two bases
        raise ValueError(f"unsupported correlation basis {basis!r}")
    return {
        name: receptor @ matrix @ receptor.conj().T
        for name, matrix in bridged_stokes_matrices().items()
    }


@dataclass(frozen=True, slots=True)
class PolarizedBaselineTransfer:
    """One baseline's full-Stokes transfer cube and the grid it was built on."""

    table: PolarizedPackedTable
    blm: PolarizedPackedCube
    quadrature_directions_enu: np.ndarray
    quadrature_weights: np.ndarray
    horizon_mask: np.ndarray
    fringe: np.ndarray
    correlation_labels: tuple[str, str, str, str]
    per_antenna_beam_identities: tuple[str, str]
    baseline_enu_m: tuple[float, float, float]
    frequency_hz: float

    @property
    def fringe_convention(self) -> str:
        """Return the existing geometric-phase convention identifier."""
        return FRINGE_CONVENTION

    def correlation_index(self, label: str) -> int:
        """Return the row-major index of one resolved correlation label."""
        return self.correlation_labels.index(str(label))

    def coefficient(
        self,
        baseline: int,
        frequency: int,
        correlation: int,
        field: str,
        degree: int,
        order: int,
    ) -> complex:
        """Return one ``B^X_lm`` cell of the retained cube."""
        return self.blm.coefficient(
            baseline, frequency, correlation, field, degree, order
        )

    def at_relative_phase(self, relative_phase_rad: float) -> PolarizedBaselineTransfer:
        """Return ``B^X_lm(alpha) = B^X_lm(0) exp(+i m alpha)`` for every field.

        Section 4.1's ``T(alpha) = T(0) R3(alpha)`` makes this exact rather than
        approximate for all four fields alike: the whole instrument is rigid in
        the terrestrial frame and the sky rotates by one parameter about the
        retained polar axis, which multiplies every signed-``m`` block by the
        same scalar regardless of its spin.
        """
        alpha = float(relative_phase_rad)
        values = np.array(self.blm.values, copy=True)
        for row in self.table.block_rows:
            start, stop = int(row["value_start"]), int(row["value_stop"])
            if stop <= start:
                continue
            values[:, :, :, start:stop] *= complex(
                math.cos(int(row["m"]) * alpha), math.sin(int(row["m"]) * alpha)
            )
        return PolarizedBaselineTransfer(
            table=self.table,
            blm=PolarizedPackedCube(table=self.table, values=values),
            quadrature_directions_enu=self.quadrature_directions_enu,
            quadrature_weights=self.quadrature_weights,
            horizon_mask=self.horizon_mask,
            fringe=self.fringe,
            correlation_labels=self.correlation_labels,
            per_antenna_beam_identities=self.per_antenna_beam_identities,
            baseline_enu_m=self.baseline_enu_m,
            frequency_hz=self.frequency_hz,
        )


def build_polarized_baseline_transfer(
    *,
    beam: str,
    baseline_enu_m: Sequence[float],
    frequency_hz: float,
    lmax: int,
    mmax: int,
    quadrature_nside: int,
    antenna_diameters_m: Sequence[float] = (14.0, 14.0),
    correlation_labels: Sequence[str] = ("XX", "XY", "YX", "YY"),
    table: PolarizedPackedTable | None = None,
    horizon_free: bool = False,
) -> PolarizedBaselineTransfer:
    r"""Build the full-Stokes transfer ``B^X_lm`` of one baseline.

    Section 6 defines the reference-phase response from *the same* Jones and
    fringe factors as the direct RIME and then expands it in **conjugate**
    harmonics on the transfer side:

    .. math::

        B^I_{pqfc,lm}=\int K^I Y_{lm}\,d\Omega,\qquad
        B^V_{pqfc,lm}=\int K^V Y_{lm}\,d\Omega,

    .. math::

        B^{(+2)}_{pqfc,lm}=\int(K^Q-iK^U)\,{}_{+2}Y_{lm}\,d\Omega,\qquad
        B^{(-2)}_{pqfc,lm}=\int(K^Q+iK^U)\,{}_{-2}Y_{lm}\,d\Omega.

    The horizon factor ``H = 1[alt > 0]`` is part of the transfer function, and
    it is applied through :func:`radiosim.core.mmode.frame.strict_horizon_visible`
    -- the one shared code object Section 6 mandates the private direct oracles
    invoke too, "never a re-derivation of the same formula".

    The receptor cells are the constant celestial-tangent-basis matrices of
    :func:`receptor_component_matrices`; no tangent transport enters, per
    Section 6's direct-RIME anchoring for the identity-``P`` scope.

    Parameters
    ----------
    horizon_free : bool
        Section 7.3's tier-1a ablation.  When true, **every** horizon truncation
        is removed and the beam is sampled at its exact even continuation
        ``abs(alt)`` -- an aperture pattern depends on the zenith angle through
        the even ``sin(theta) = cos(alt)`` -- while the fringe, entire in the
        direction cosines, stays on the true direction.  Removing the explicit
        ``H`` factor alone is insufficient, because the resolved beam applies its
        own below-horizon cut.  The result is a tier-1 internal and never a
        result.
    """
    from radiosim.backends import get_backend
    from radiosim.core.jones import geometric_phase

    labels = tuple(str(label) for label in correlation_labels)
    if len(labels) != N_CORRELATIONS:
        raise ValueError(
            "an SCI-004 transfer cube has exactly four correlations; "
            "an omitted cross-hand is a rejection, not a supported subset"
        )
    resolved = (
        table
        if table is not None
        else polarized_packed_block_table(lmax=lmax, mmax=mmax)
    )
    directions, weights = quadrature_grid(quadrature_nside)

    # Section 6: ``H`` is part of the transfer function and the predicate is the
    # one shared code object the direct oracles invoke.
    horizon = strict_horizon_visible(directions[:, 2])
    horizon_weight = (
        np.ones(directions.shape[0], dtype=np.float64)
        if horizon_free
        else horizon.astype(np.float64)
    )

    baseline = np.asarray(baseline_enu_m, dtype=np.float64).reshape(1, 3)
    wavelength = _SPEED_OF_LIGHT_M_PER_S / float(frequency_hz)
    fringe = np.asarray(
        geometric_phase(
            uvw_wavelengths=baseline / wavelength,
            dir_l=directions[:, 0],
            dir_m=directions[:, 1],
            dir_n=directions[:, 2],
            backend=get_backend("numpy"),
        ),
        dtype=np.complex128,
    )

    diameters = tuple(float(value) for value in antenna_diameters_m)
    if len(diameters) != 2:
        raise ValueError("a baseline has exactly two antenna diameters")
    # Section 7.3: the tier-1a ablation samples the beam on ``abs(alt)``, its
    # unique smooth even continuation, rather than on the true direction.
    beam_directions = directions
    if horizon_free:
        beam_directions = np.array(directions, copy=True)
        beam_directions[:, 2] = np.abs(beam_directions[:, 2])
    responses = [
        _scalar_beam_response(
            beam_directions, beam=beam, diameter_m=diameter, frequency_hz=frequency_hz
        )
        for diameter in diameters
    ]
    identities = tuple(
        "unit_response_v1"
        if beam == "unit"
        else f"circular_aperture_uniform_illumination_v1:diameter_m={diameter!r}"
        for diameter in diameters
    )

    components = receptor_component_matrices(labels)
    scalar_product = responses[0] * np.conjugate(responses[1])
    common = scalar_product * fringe[0] * horizon_weight

    theta = np.arccos(np.clip(directions[:, 2], -1.0, 1.0))
    phi = np.arctan2(directions[:, 1], directions[:, 0]) % (2.0 * math.pi)
    # ``B_lm`` expands the transfer in *unconjugated* harmonics against the
    # conjugated sky expansion, so ``sum_lm B_lm a_lm`` reproduces the integral.
    harmonics = np.conjugate(packed_polarized_conjugate_harmonics(resolved, theta, phi))

    rows = receptor_row_indices(labels)
    packed = np.zeros(
        (1, 1, N_CORRELATIONS, resolved.packed_value_count), dtype=np.complex128
    )
    columns = {name: field_columns(resolved, name) for name in resolved.field_order}
    for index, label in enumerate(labels):
        row, column = rows[label[0]], rows[label[1]]
        cell = {
            name: complex(matrix[row][column]) for name, matrix in components.items()
        }
        kernel = {
            "I": cell["I"],
            "V": cell["V"],
            "+2": cell["Q"] - 1j * cell["U"],
            "-2": cell["Q"] + 1j * cell["U"],
        }
        for name in resolved.field_order:
            selected = columns[name]
            if selected.size == 0:
                continue
            weighted = (common * weights) * kernel[name]
            packed[0, 0, index, selected] = weighted @ harmonics[:, selected]

    return PolarizedBaselineTransfer(
        table=resolved,
        blm=PolarizedPackedCube(table=resolved, values=packed),
        quadrature_directions_enu=directions,
        quadrature_weights=weights,
        horizon_mask=horizon,
        fringe=fringe,
        correlation_labels=(labels[0], labels[1], labels[2], labels[3]),
        per_antenna_beam_identities=(identities[0], identities[1]),
        baseline_enu_m=(
            float(baseline[0, 0]),
            float(baseline[0, 1]),
            float(baseline[0, 2]),
        ),
        frequency_hz=float(frequency_hz),
    )
