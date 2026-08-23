# radiosim/core/polarization.py
"""
Polarization utilities for Jones matrices and coherency matrices.

Implements the Radio Interferometer Measurement Equation (RIME):
    V_ij = E_i(θ,φ) @ C_source @ E_j^H(θ,φ) @ exp[-2πi(uvw·lmn)]

CONVENTION CHOICES (Critical for correctness):

1. **Stokes → Coherency: "Half-Power" / Density Matrix Convention**

   C = (1/2) * [[I+Q,    U+iV  ],
                [U-iV,   I-Q   ]]

   The matrix axes are the canonical sky order ``(North, East)``:
   - B_NN = (I+Q)/2  (north-directed power)
   - B_EE = (I-Q)/2  (east-directed power)

   The receptor term maps this sky matrix to the requested products.  For the
   ideal default ``(X=east, Y=north)`` matched unit-response output,
   ``XX=(I-Q)/2`` and ``YY=(I+Q)/2``; their sum is still I.  Heterogeneous or
   non-unitary Jones chains can change that observed sum.

   This ensures the sky coherency of a 1 Jy source has trace 1 Jy, not 2 Jy.

2. **Stokes V Sign: IAU / Hamaker-Bregman-Sault Convention**

   C[0,1] = (U + iV) / 2

   The upper-right sky ``(North, East)`` element carries ``+iV``; the
   lower-left carries ``-iV``.  After the east-X permutation, the reported
   ``XY`` product carries ``-iV`` and ``YX`` carries ``+iV``.

   This is the convention of the primary references, verified term by term
   during the Tier 5A evidence gate:

   - Hamaker, Bregman & Sault 1996, A&AS 117, 137: Eq. (3) fixes the coherency
     ordering ``(e_x e_x*, e_x e_y*, e_y e_x*, e_y e_y*)`` and Eq. (9), the
     inverse of the Eq. (8) Stokes map, gives ``<e_x e_y*> = (U + iV)/2``.
   - Smirnov 2011, A&A 527, A106, Eq. (7): ``B = [[I+Q, U+iV], [U-iV, I-Q]]``,
     with the circular form of Section 6.3 giving ``RR = I + V``.
   - codex-africanus (``africanus/model/coherency/conversion.py``):
     ``"XY": u + v*1j``, ``"YX": u - v*1j``, ``"RR": i + v``, ``"LL": i - v``.

   Observable consequence: under the linear-to-circular basis matrix
   ``S = (1/sqrt 2) [[1, i], [1, -i]]`` this yields ``RR = (I+V)/2`` and
   ``LL = (I-V)/2``, so a source with ``V = +I`` emerges as pure RR.

   **Divergence from pyradiosky.** ``pyradiosky.utils.stokes_to_coherency``
   builds the mirror of the matrix above, ``C[0,1] = (U - iV) / 2``, as does
   Hamaker 2006, A&A 456, 395 Eq. (3). RadioSim deliberately follows
   the HBS 1996 / Smirnov 2011 / africanus sign instead. No RadioSim data path
   mixes the two: the ``pyradiosky_file`` loader reads Stokes I/Q/U/V columns
   and never a pyradiosky-built coherency matrix. A user combining RadioSim
   cross-hand visibilities with a pyradiosky-computed coherency must flip the
   sign of V. All V = 0 results are identical under either convention.

3. **Stokes I Extraction: Simple Sum (No Division)**

   I = B_NN + B_EE

   For an ideal matched unit-response system this is also ``V_XX + V_YY``.
   Arbitrary per-antenna Jones factors need not preserve that visibility sum.

4. **Jones Matrix: RIME Standard**

   J[feed, sky_basis]: rows are receptor feeds, columns are ``(North, East)``

References:
- Hamaker, Bregman & Sault 1996, A&AS 117, 137: Eqs. (3), (8), (9)
- Smirnov 2011, A&A 527, A106: "Revisiting the RIME I", Eq. (7)
- codex-africanus: Stokes/correlation mapping
- Price 2015: "Bayesian optimal mapping" (basis rotations)
"""

import numpy as np


def stokes_to_coherency(stokes_I, stokes_Q=0, stokes_U=0, stokes_V=0, *, xp=np):
    """
    Convert Stokes parameters to 2×2 coherency matrix.

    Uses the "half-power" / density matrix convention with the IAU/HBS V sign:

    C = (1/2) * [[I+Q,    U+iV  ],
                 [U-iV,   I-Q   ]]

    ENERGY CONSERVATION: For a source with flux I, this produces a sky
    coherency matrix whose trace is I (not 2I).  Post-Jones visibility traces
    need not remain I for heterogeneous or non-unitary chains.  Receptor/output
    matrices determine product labels and axis order.

    Parameters
    ----------
    stokes_I : float or array
        Total intensity (Stokes I) in Jy. Required, must be ≥ 0.
    stokes_Q : float or array, optional
        Linear polarization (Stokes Q) in Jy. Default 0 (unpolarized).
    stokes_U : float or array, optional
        Linear polarization at 45° (Stokes U) in Jy. Default 0.
    stokes_V : float or array, optional
        Circular polarization (Stokes V) in Jy. Default 0.
        Note: Sign convention follows HBS 1996 / Smirnov 2011 / africanus,
        i.e. ``C[0,1] = (U + iV)/2``. This is the mirror of pyradiosky's.

    Returns
    -------
    coherency : ndarray
        2×2 complex coherency matrix (density matrix form)
        Shape: (2, 2) for scalar inputs, or (..., 2, 2) for array inputs

    Notes
    -----
    - Coherency matrix is Hermitian: C = C^H (conjugate transpose)
    - For physical sources: C must be positive semi-definite
    - Trace(C) = I (total intensity), NOT 2I
    - C[0,0] = (I+Q)/2, C[1,1] = (I-Q)/2 → Sum = I ✓

    Broadcasting:

    - The four Stokes inputs broadcast against each other under the usual
      NumPy rules, so scalars — including the Q/U/V defaults — combine
      freely with arrays. Genuinely incompatible shapes still raise
      ``ValueError``.
    - Output adds (2, 2) dimensions at the end of the broadcast shape
    - Example: all four of shape (100,) → C.shape=(100, 2, 2)

    >>> stokes_to_coherency(np.ones(5)).shape  # scalar defaults broadcast
    (5, 2, 2)

    Examples
    --------
    >>> # Unpolarized 1 Jy source; check energy conservation
    >>> C = stokes_to_coherency(stokes_I=1.0)
    >>> bool(np.allclose(C[0, 0] + C[1, 1], 1.0))
    True

    >>> # Fully Q-polarized: all power lands on the North sky axis
    >>> C = stokes_to_coherency(stokes_I=10.0, stokes_Q=10.0)
    >>> float(C[0, 0].real), float(C[1, 1].real)
    (10.0, 0.0)

    >>> # Circular polarization (IAU/HBS: U+iV -> +iV)
    >>> C = stokes_to_coherency(stokes_I=5.0, stokes_V=2.0)
    >>> float(C[0, 1].imag)
    1.0
    """
    # Convert to arrays for consistent handling.
    stokes_I = xp.asarray(stokes_I, dtype=float)
    stokes_Q = xp.asarray(stokes_Q, dtype=float)
    stokes_U = xp.asarray(stokes_U, dtype=float)
    stokes_V = xp.asarray(stokes_V, dtype=float)

    # Broadcast to one shared shape so scalar inputs (including the Q/U/V
    # defaults) combine with array inputs; genuinely incompatible shapes
    # raise ValueError here. Broadcasting already-equal shapes is an
    # identity view, so previously valid inputs are untouched.
    stokes_I, stokes_Q, stokes_U, stokes_V = xp.broadcast_arrays(
        stokes_I, stokes_Q, stokes_U, stokes_V
    )

    # Fill coherency matrix with the IAU/HBS half-power convention.
    row_x = xp.stack(
        [
            stokes_I + stokes_Q,
            stokes_U + 1j * stokes_V,
        ],
        axis=-1,
    )
    row_y = xp.stack(
        [
            stokes_U - 1j * stokes_V,
            stokes_I - stokes_Q,
        ],
        axis=-1,
    )
    coherency = xp.stack([row_x, row_y], axis=-2)

    # Normalize: Divide by 2 for density matrix / half-power convention
    # CRITICAL: This ensures Tr(C) = I, not 2I (energy conservation)
    coherency = coherency / 2.0

    return coherency


def apply_jones_matrices(jones_i, coherency, jones_j):
    """
    Apply Jones matrices to coherency matrix for visibility.

    Computes: V = E_i @ C @ E_j^H

    where E_j^H is the Hermitian conjugate of E_j.

    Parameters
    ----------
    jones_i : ndarray
        Jones matrix for antenna i
        Shape: (2, 2) or (..., 2, 2)
        Convention: jones[feed, sky_basis]
    coherency : ndarray
        Coherency matrix for source
        Shape: (2, 2) or (..., 2, 2)
    jones_j : ndarray
        Jones matrix for antenna j
        Shape: (2, 2) or (..., 2, 2)

    Returns
    -------
    visibility : ndarray
        2×2 complex visibility matrix
        Shape: (2, 2) or (..., 2, 2)
        Elements are ordered by the rows of the supplied Jones matrices.

    Broadcasting Notes
    ------------------
    CRITICAL: All inputs must broadcast to compatible shapes!

    Safe patterns:
    1. All same shape: (N, 2, 2) @ (N, 2, 2) @ (N, 2, 2) → (N, 2, 2) ✓
    2. Scalar propagation: (2, 2) @ (2, 2) @ (2, 2) → (2, 2) ✓
    3. Broadcast dimension: (1, 2, 2) @ (N, 2, 2) → (N, 2, 2) ✓

    Dangerous patterns (will fail or give wrong results):
    - jones_i: (Ntime, 2, 2), coherency: (Nsources, 2, 2)
      → Broadcast fails: incompatible (Ntime ≠ Nsources)

    Best practice: explicitly reshape/broadcast before calling.

    .. code-block:: python

        # Vectorize over sources
        jones_i_all = jones_i[..., None, :, :]  # (Ntime, 1, 2, 2)
        coherency_all = coherency[None, ...]  # (1, Nsources, 2, 2)
        # Now broadcasts to (Ntime, Nsources, 2, 2)

    Notes
    -----
    - For single source per call: all inputs (2, 2) → output (2, 2)
    - For vectorized sources: typically all (Nsources, 2, 2)
    - The @ operator handles the last two dimensions as matrices

    Examples
    --------
    >>> # Single source, single baseline
    >>> jones_i = np.array([[1.0 + 0j, 0.05], [0.03, 0.95]])
    >>> jones_j = np.array([[0.98, 0.02], [0.01, 0.99]])
    >>> C = stokes_to_coherency(stokes_I=10.0)
    >>> vis = apply_jones_matrices(jones_i, C, jones_j)
    >>> vis.shape
    (2, 2)

    >>> # Multiple sources (vectorized)
    >>> Nsrc = 100
    >>> jones_i_all = np.tile(jones_i, (Nsrc, 1, 1))  # (100, 2, 2)
    >>> zeros = np.zeros(Nsrc)
    >>> C_all = stokes_to_coherency(np.ones(Nsrc), zeros, zeros, zeros)
    >>> jones_j_all = np.tile(jones_j, (Nsrc, 1, 1))
    >>> vis_all = apply_jones_matrices(jones_i_all, C_all, jones_j_all)
    >>> vis_all.shape
    (100, 2, 2)
    """
    # E_j^H = Hermitian conjugate (conjugate transpose)
    # np.swapaxes on last two axes works for any leading dimensions
    jones_j_H = np.conj(np.swapaxes(jones_j, -2, -1))

    # Matrix multiplication: E_i @ C @ E_j^H
    # @ operator broadcasts and operates on last two dimensions
    visibility = jones_i @ coherency @ jones_j_H

    return visibility


def stokes_I_only_visibility(jones_i, jones_j, intensity):
    """
    Simplified calculation for unpolarized source (Stokes I only).

    For unpolarized: C = (I/2) * Identity
    Result: V = (I/2) * (J_i @ J_j^H)

    More efficient than full coherency when Q = U = V = 0.

    Parameters
    ----------
    jones_i, jones_j : ndarray
        Jones matrices, shape (2, 2) or (..., 2, 2)
    intensity : float or array
        Stokes I (total flux) in Jy

    Returns
    -------
    visibility : ndarray
        2×2 visibility matrix, shape (2, 2) or (..., 2, 2)

    Notes
    -----
    Even for unpolarized sources, vis_matrix can have non-zero XY, YX
    due to instrumental polarization leakage (off-diagonal Jones terms).

    This is equivalent to:

    .. code-block:: python

        C = stokes_to_coherency(stokes_I=intensity, stokes_Q=0, stokes_U=0, stokes_V=0)
        vis = apply_jones_matrices(jones_i, C, jones_j)

    Examples
    --------
    >>> J_i = np.array([[0.95, 0.05], [0.02, 0.98]])  # Leaky beam
    >>> J_j = np.eye(2)  # Ideal
    >>> vis = stokes_I_only_visibility(J_i, J_j, intensity=10.0)
    >>> float(vis[0, 1])  # Non-zero! Leakage creates cross-pol
    0.25
    """
    jones_j_H = np.conj(np.swapaxes(jones_j, -2, -1))
    visibility = (intensity / 2.0) * (jones_i @ jones_j_H)
    return visibility


def coherency_to_stokes(coherency):
    """
    Convert coherency matrix back to Stokes parameters.

    Inverse of stokes_to_coherency() for validation/testing.  The input must be
    in the canonical sky ``(North, East)`` order; apply the inverse receptor
    transform before passing an east-X output visibility matrix.

    Parameters
    ----------
    coherency : ndarray
        2×2 complex coherency matrix (half-power convention)
        Shape: (2, 2) or (..., 2, 2)

    Returns
    -------
    stokes_I, stokes_Q, stokes_U, stokes_V : float or array
        Stokes parameters in Jy

    Notes
    -----
    Round-trip property (up to numerical precision), for any Stokes vector:

    .. code-block:: python

        C = stokes_to_coherency(stokes_I, stokes_Q, stokes_U, stokes_V)
        I2, Q2, U2, V2 = coherency_to_stokes(C)
        assert np.allclose([stokes_I, stokes_Q, stokes_U, stokes_V], [I2, Q2, U2, V2])

    The Examples section below executes exactly that round trip.

    With half-power convention C = [[I+Q, U+iV], [U-iV, I-Q]] / 2:
    - I: C[0,0] + C[1,1] = (I+Q)/2 + (I-Q)/2 = I (no factor needed!)
    - Q: C[0,0] - C[1,1] = (I+Q)/2 - (I-Q)/2 = Q (no factor needed!)
    - U: C[0,1] + C[1,0] = (U+iV)/2 + (U-iV)/2 = U (no factor needed!)
    - V: Im(C[0,1]) = Im((U+iV)/2) = V/2, so V = 2*Im(C[0,1]) (factor of 2!)

    The /2 in the coherency definition causes terms to cancel for I, Q, U.
    Only V genuinely needs the factor of 2.

    Examples
    --------
    >>> sI, sQ, sU, sV = 10.0, 2.0, -1.0, 0.5
    >>> C = stokes_to_coherency(sI, sQ, sU, sV)
    >>> I2, Q2, U2, V2 = coherency_to_stokes(C)
    >>> np.allclose([sI, sQ, sU, sV], [I2, Q2, U2, V2])
    True
    """
    # Key insight: The /2 in coherency causes cancellation when adding/subtracting diagonals
    # Sum of halved parts = whole (no factor of 2 needed for I, Q, U)

    # stokes_I = Tr(C) = (I+Q)/2 + (I-Q)/2 = I
    stokes_I = coherency[..., 0, 0].real + coherency[..., 1, 1].real

    # stokes_Q = (I+Q)/2 - (I-Q)/2 = Q
    stokes_Q = coherency[..., 0, 0].real - coherency[..., 1, 1].real

    # stokes_U = (U+iV)/2 + (U-iV)/2 = U (taking real part)
    stokes_U = coherency[..., 0, 1].real + coherency[..., 1, 0].real

    # stokes_V: Im(C[0,1]) = Im((U+iV)/2) = V/2, so multiply by 2
    stokes_V = 2 * coherency[..., 0, 1].imag

    return stokes_I, stokes_Q, stokes_U, stokes_V


def jones_matrix_power(jones):
    """
    Calculate power beam from E-field Jones matrix.

    Power response: ``P = |E|²`` (square-law detector)

    Parameters
    ----------
    jones : ndarray
        2×2 complex Jones matrix (E-field)
        Shape: (2, 2) or (..., 2, 2)

    Returns
    -------
    power_x : float or array
        Power for X polarization: ``|J_Xθ|² + |J_Xφ|²``
    power_y : float or array
        Power for Y polarization: ``|J_Yθ|² + |J_Yφ|²``

    Notes
    -----
    Power beam = what a square-law detector measures (loses phase info).
    E-field beam = includes phase (needed for interferometry).

    Examples
    --------
    >>> J = np.array([[0.9 + 0.1j, 0.05], [0.03, 0.95 - 0.05j]])
    >>> px, py = jones_matrix_power(J)
    >>> round(float(px), 4)  # |0.9+0.1j|² + |0.05|²
    0.8225
    >>> round(float(py), 4)  # |0.03|² + |0.95-0.05j|²
    0.9059
    """
    power_x = np.abs(jones[..., 0, 0]) ** 2 + np.abs(jones[..., 0, 1]) ** 2
    power_y = np.abs(jones[..., 1, 0]) ** 2 + np.abs(jones[..., 1, 1]) ** 2
    return power_x, power_y


# ---------------------------------------------------------------------------
# SCI-004 Section 5.2: the RadioSim-to-Shaw basis bridge
# ---------------------------------------------------------------------------

#: Section 10's ``stokes_v_basis_bridge`` literal, which is never nullable.
STOKES_V_BASIS_BRIDGE = "radiosim.stokes-ne-theta-phi.v1"


def shaw_basis_bridge(*, xp=np):
    r"""Return Section 5.2's exact basis bridge ``D = diag(-1, 1)``.

    ``docs/development/sci004_mmode_design.md`` Section 5.2: RadioSim's sky
    electric vector is ordered ``(North, East)`` while Shaw et al. use the
    spherical ``(theta, phi)`` basis with ``theta`` pointing *South* and ``phi``
    East, so

    .. math::

        D=\operatorname{diag}(-1,1),\qquad e_{\theta\phi}=De_{NE},\qquad
        J_{\theta\phi}=J_{NE}D.

    ``D`` is **not** the SCI-006 east-X permutation and does not replace it: the
    receptor permutation stays inside ``J_NE`` and is antidiagonal, while ``D``
    is diagonal.  Applying ``D`` in place of the permutation is precisely the
    defect SCI-006 ruled on.

    Examples
    --------
    >>> import numpy as np
    >>> bridge = shaw_basis_bridge()
    >>> bool(np.array_equal(bridge, np.array([[-1.0, 0.0], [0.0, 1.0]])))
    True
    >>> bool(np.array_equal(bridge @ bridge, np.eye(2)))
    True
    """
    return xp.asarray([[-1.0, 0.0], [0.0, 1.0]], dtype=xp.float64)


def stokes_to_shaw_fields(stokes_I, stokes_Q, stokes_U, stokes_V):
    r"""Return ``(I_H, Q_H, U_H, V_H)`` for Section 5.2's Shaw-form equations.

    Transporting RadioSim's ``(North, East)`` brightness matrix with
    ``D P D^T`` and reading the result against Shaw's own ``(theta, phi)``
    matrix -- whose ``P^V`` carries the opposite matrix sign in one unchanged
    ordered basis -- gives

    .. math:: I_H=I_{RS},\quad Q_H=Q_{RS},\quad U_H=-U_{RS},\quad V_H=V_{RS}.

    Only ``U`` flips.  Section 5.2 forbids any additional fitted or configurable
    ``V`` flip: after the bridge the physical IAU ``V`` field has the same sign.

    Parameters
    ----------
    stokes_I, stokes_Q, stokes_U, stokes_V : float or array
        The RadioSim-convention Stokes payload.  Arrays broadcast elementwise
        and the caller's dtype is preserved.

    Returns
    -------
    tuple
        ``(I_H, Q_H, U_H, V_H)`` in the Shaw ``(theta, phi)`` basis.

    Examples
    --------
    >>> stokes_to_shaw_fields(1.0, 0.2, 0.3, 0.4)
    (1.0, 0.2, -0.3, 0.4)
    """
    return (stokes_I, stokes_Q, -stokes_U, stokes_V)
