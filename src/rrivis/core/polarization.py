# rrivis/core/polarization.py
"""
Polarization utilities for Jones matrices and coherency matrices.

Implements the Radio Interferometer Measurement Equation (RIME):
    V_ij = E_i(θ,φ) @ C_source @ E_j^H(θ,φ) @ exp[-2πi(uvw·lmn)]

CONVENTION CHOICES (Critical for correctness):

1. **Stokes → Coherency: "Half-Power" / Density Matrix Convention**

   C = (1/2) * [[I+Q,    U-iV  ],
                [U+iV,   I-Q   ]]

   Physical meaning: A source with flux I splits power between two feeds:
   - V_XX = (I+Q)/2  (half the power in X feed)
   - V_YY = (I-Q)/2  (half the power in Y feed)
   - V_XX + V_YY = I (energy conserved!)

   This ensures a 1 Jy source produces 1 Jy total visibility, not 2 Jy.

2. **Stokes V Sign: Africanus/Pauli Convention**

   C[0,1] = (U - iV) / 2  (Africanus/Pauli)
   NOT: (U + iV) / 2      (Smirnov 2011 alternative)

   Matches: Codex-Africanus, matvis, pfb-imaging, Wikipedia

3. **Stokes I Extraction: Simple Sum (No Division)**

   I = V_XX + V_YY  (sum of parts = whole)

   Intuitive for debugging and consistent with half-power convention.

4. **Jones Matrix: RIME Standard**

   J[feed, sky_basis]: rows=feeds (X,Y), columns=sky basis (θ,φ)

References:
- Smirnov 2011: "Revisiting the RIME I" (Eq. 4 for brightness matrix)
- Hamaker & Bregman 1996: IAU polarization conventions
- Africanus docs: Stokes/correlation mapping
- Price 2015: "Bayesian optimal mapping" (basis rotations)
"""

import numpy as np


def stokes_to_coherency(I, Q=0, U=0, V=0):
    """
    Convert Stokes parameters to 2×2 coherency matrix.

    Uses "half-power" / density matrix convention with Africanus/Pauli V sign:

    C = (1/2) * [[I+Q,    U-iV  ],
                 [U+iV,   I-Q   ]]

    ENERGY CONSERVATION: For a source with flux I, this produces visibilities
    where V_XX + V_YY = I (not 2I), ensuring physical correctness.

    Parameters
    ----------
    I : float or array
        Total intensity (Stokes I) in Jy. Required, must be ≥ 0.
    Q : float or array, optional
        Linear polarization (Stokes Q) in Jy. Default 0 (unpolarized).
        Range: -I to +I
    U : float or array, optional
        Linear polarization at 45° (Stokes U) in Jy. Default 0.
        Range: -I to +I
    V : float or array, optional
        Circular polarization (Stokes V) in Jy. Default 0.
        Range: -I to +I (positive = left-circular for Africanus)
        Note: Sign convention matches Africanus, opposite of Smirnov 2011.

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
    - All inputs broadcast to common shape
    - Output adds (2, 2) dimensions at end
    - Example: I.shape=(100,) → C.shape=(100, 2, 2)

    Examples
    --------
    >>> # Unpolarized 1 Jy source
    >>> C = stokes_to_coherency(I=1.0)
    >>> # Check energy conservation
    >>> np.allclose(C[0,0] + C[1,1], 1.0)  # → True

    >>> # Fully Q-polarized
    >>> C = stokes_to_coherency(I=10.0, Q=10.0)  # All in X feed
    >>> C[0,0], C[1,1]  # → (10.0, 0.0)

    >>> # Circular polarization
    >>> C = stokes_to_coherency(I=5.0, V=2.0)
    >>> C[0,1].imag  # → -1.0 (Africanus: U-iV → -iV)
    """
    # Convert to arrays for consistent handling
    I = np.asarray(I, dtype=float)
    Q = np.asarray(Q, dtype=float)
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)

    # Broadcast to common shape
    I, Q, U, V = np.broadcast_arrays(I, Q, U, V)

    # Initialize coherency matrix
    # Shape: (..., 2, 2) for array inputs, (2, 2) for scalars
    shape = I.shape + (2, 2)
    coherency = np.zeros(shape, dtype=complex)

    # Fill coherency matrix with Africanus/half-power convention
    coherency[..., 0, 0] = I + Q  # XX: (I+Q) before normalization
    coherency[..., 0, 1] = U - 1j * V  # XY: U - iV (Africanus)
    coherency[..., 1, 0] = U + 1j * V  # YX: U + iV (conjugate)
    coherency[..., 1, 1] = I - Q  # YY: (I-Q) before normalization

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
        Elements: [[V_XX, V_XY], [V_YX, V_YY]]

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

    Best practice: Explicitly reshape/broadcast before calling:
    >>> # Vectorize over sources
    >>> jones_i_all = jones_i[..., None, :, :]  # (Ntime, 1, 2, 2)
    >>> coherency_all = coherency[None, ...]    # (1, Nsources, 2, 2)
    >>> # Now broadcasts to (Ntime, Nsources, 2, 2)

    Notes
    -----
    - For single source per call: all inputs (2, 2) → output (2, 2)
    - For vectorized sources: typically all (Nsources, 2, 2)
    - The @ operator handles the last two dimensions as matrices

    Examples
    --------
    >>> # Single source, single baseline
    >>> jones_i = np.array([[1.0+0j, 0.05], [0.03, 0.95]])
    >>> jones_j = np.array([[0.98, 0.02], [0.01, 0.99]])
    >>> C = stokes_to_coherency(I=10.0)
    >>> vis = apply_jones_matrices(jones_i, C, jones_j)
    >>> vis.shape  # → (2, 2)

    >>> # Multiple sources (vectorized)
    >>> Nsrc = 100
    >>> jones_i_all = np.tile(jones_i, (Nsrc, 1, 1))  # (100, 2, 2)
    >>> C_all = stokes_to_coherency(I=np.ones(Nsrc))  # (100, 2, 2)
    >>> jones_j_all = np.tile(jones_j, (Nsrc, 1, 1))
    >>> vis_all = apply_jones_matrices(jones_i_all, C_all, jones_j_all)
    >>> vis_all.shape  # → (100, 2, 2)
    """
    # E_j^H = Hermitian conjugate (conjugate transpose)
    # np.swapaxes on last two axes works for any leading dimensions
    jones_j_H = np.conj(np.swapaxes(jones_j, -2, -1))

    # Matrix multiplication: E_i @ C @ E_j^H
    # @ operator broadcasts and operates on last two dimensions
    visibility = jones_i @ coherency @ jones_j_H

    return visibility


def visibility_to_correlations(vis_matrix):
    """
    Extract correlation products from visibility matrix.

    Converts 2×2 visibility matrix → correlation dictionary

    Parameters
    ----------
    vis_matrix : ndarray
        2×2 complex visibility matrix
        Shape: (2, 2) or (..., 2, 2)

    Returns
    -------
    correlations : dict
        Dictionary with keys:
        - 'XX': V_XX, parallel-hand (X feed × X feed)
        - 'XY': V_XY, cross-hand (X feed × Y feed)
        - 'YX': V_YX, cross-hand (Y feed × X feed)
        - 'YY': V_YY, parallel-hand (Y feed × Y feed)
        - 'I': Stokes I visibility = V_XX + V_YY (no division!)

    Notes
    -----
    CONVENTION: With half-power coherency normalization:
    - V_XX = (I+Q)/2, V_YY = (I-Q)/2
    - Therefore: I = V_XX + V_YY (sum, not average!)
    - This is "sum of parts = whole" (energy conservation)

    For linear feeds (X=E, Y=N in typical radio astronomy):
    - XX: East-West linear correlation
    - YY: North-South linear correlation
    - XY, YX: Cross-hand (measure U ± iV)

    Unpolarized source (Q=U=V=0):
    - V_XX ≈ I/2, V_YY ≈ I/2
    - V_XY ≈ V_YX ≈ 0 (ideal case)
    - With leaky beams: V_XY, V_YX ≠ 0 even for unpolarized source

    Examples
    --------
    >>> # Perfect 10 Jy unpolarized source, ideal instrument
    >>> C = stokes_to_coherency(I=10.0)
    >>> J = np.eye(2)  # Ideal Jones matrix
    >>> vis = apply_jones_matrices(J, C, J)
    >>> corr = visibility_to_correlations(vis)
    >>> corr['XX']  # → 5.0 (half the flux)
    >>> corr['YY']  # → 5.0 (other half)
    >>> corr['I']   # → 10.0 (total intensity recovered!)

    >>> # Fully Q-polarized
    >>> C = stokes_to_coherency(I=10.0, Q=10.0)
    >>> vis = apply_jones_matrices(J, C, J)
    >>> corr = visibility_to_correlations(vis)
    >>> corr['XX']  # → 10.0 (all in X)
    >>> corr['YY']  # → 0.0 (none in Y)
    >>> corr['I']   # → 10.0 ✓
    """
    # Extract elements from visibility matrix
    correlations = {
        "XX": vis_matrix[..., 0, 0],
        "XY": vis_matrix[..., 0, 1],
        "YX": vis_matrix[..., 1, 0],
        "YY": vis_matrix[..., 1, 1],
        # CRITICAL: No division by 2! Sum = whole with half-power convention
        "I": vis_matrix[..., 0, 0] + vis_matrix[..., 1, 1],
    }

    return correlations


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
    >>> C = stokes_to_coherency(I=intensity, Q=0, U=0, V=0)
    >>> vis = apply_jones_matrices(jones_i, C, jones_j)

    Examples
    --------
    >>> J_i = np.array([[0.95, 0.05], [0.02, 0.98]])  # Leaky beam
    >>> J_j = np.eye(2)  # Ideal
    >>> vis = stokes_I_only_visibility(J_i, J_j, intensity=10.0)
    >>> vis[0,1]  # Non-zero! Leakage creates cross-pol
    """
    jones_j_H = np.conj(np.swapaxes(jones_j, -2, -1))
    visibility = (intensity / 2.0) * (jones_i @ jones_j_H)
    return visibility


def coherency_to_stokes(coherency):
    """
    Convert coherency matrix back to Stokes parameters.

    Inverse of stokes_to_coherency() for validation/testing.

    Parameters
    ----------
    coherency : ndarray
        2×2 complex coherency matrix (half-power convention)
        Shape: (2, 2) or (..., 2, 2)

    Returns
    -------
    I, Q, U, V : float or array
        Stokes parameters in Jy

    Notes
    -----
    Round-trip property (up to numerical precision):
    >>> C = stokes_to_coherency(I, Q, U, V)
    >>> I2, Q2, U2, V2 = coherency_to_stokes(C)
    >>> np.allclose([I,Q,U,V], [I2,Q2,U2,V2])  # → True

    With half-power convention C = [[I+Q, U-iV], [U+iV, I-Q]] / 2:
    - I: C[0,0] + C[1,1] = (I+Q)/2 + (I-Q)/2 = I (no factor needed!)
    - Q: C[0,0] - C[1,1] = (I+Q)/2 - (I-Q)/2 = Q (no factor needed!)
    - U: C[0,1] + C[1,0] = (U-iV)/2 + (U+iV)/2 = U (no factor needed!)
    - V: Im(C[1,0]) = Im((U+iV)/2) = V/2, so V = 2*Im(C[1,0]) (factor of 2!)

    The /2 in the coherency definition causes terms to cancel for I, Q, U.
    Only V genuinely needs the factor of 2.

    Examples
    --------
    >>> I, Q, U, V = 10.0, 2.0, -1.0, 0.5
    >>> C = stokes_to_coherency(I, Q, U, V)
    >>> I2, Q2, U2, V2 = coherency_to_stokes(C)
    >>> np.allclose([I, Q, U, V], [I2, Q2, U2, V2])
    True
    """
    # Key insight: The /2 in coherency causes cancellation when adding/subtracting diagonals
    # Sum of halved parts = whole (no factor of 2 needed for I, Q, U)

    # I = Tr(C) = (I+Q)/2 + (I-Q)/2 = I
    I = coherency[..., 0, 0].real + coherency[..., 1, 1].real

    # Q = (I+Q)/2 - (I-Q)/2 = Q
    Q = coherency[..., 0, 0].real - coherency[..., 1, 1].real

    # U = (U-iV)/2 + (U+iV)/2 = U (taking real part)
    U = coherency[..., 0, 1].real + coherency[..., 1, 0].real

    # V: Im(C[1,0]) = Im((U+iV)/2) = V/2, so multiply by 2
    V = 2 * coherency[..., 1, 0].imag

    return I, Q, U, V


def jones_matrix_power(jones):
    """
    Calculate power beam from E-field Jones matrix.

    Power response: P = |E|² (square-law detector)

    Parameters
    ----------
    jones : ndarray
        2×2 complex Jones matrix (E-field)
        Shape: (2, 2) or (..., 2, 2)

    Returns
    -------
    power_x : float or array
        Power for X polarization: |J_Xθ|² + |J_Xφ|²
    power_y : float or array
        Power for Y polarization: |J_Yθ|² + |J_Yφ|²

    Notes
    -----
    Power beam = what a square-law detector measures (loses phase info).
    E-field beam = includes phase (needed for interferometry).

    Examples
    --------
    >>> J = np.array([[0.9+0.1j, 0.05], [0.03, 0.95-0.05j]])
    >>> px, py = jones_matrix_power(J)
    >>> px  # Power in X: |0.9+0.1j|² + |0.05|² ≈ 0.8225
    """
    power_x = np.abs(jones[..., 0, 0]) ** 2 + np.abs(jones[..., 0, 1]) ** 2
    power_y = np.abs(jones[..., 1, 0]) ** 2 + np.abs(jones[..., 1, 1]) ** 2
    return power_x, power_y


def mueller_from_jones(jones):
    """
    Convert Jones matrix to Mueller matrix (placeholder).

    Mueller matrix M relates Stokes vectors: S_out = M @ S_in
    Jones matrix J relates electric fields: E_out = J @ E_in

    Parameters
    ----------
    jones : ndarray
        2×2 Jones matrix, shape (2, 2) or (..., 2, 2)

    Returns
    -------
    mueller : ndarray
        4×4 real Mueller matrix (NOT IMPLEMENTED)

    Notes
    -----
    For interferometry, we use Jones matrices directly (RIME).
    Mueller matrices useful for:
    - Instrumental polarization calibration
    - Single-dish polarimetry
    - Some imaging algorithms

    References
    ----------
    - Hamaker+ 1996: "Understanding radio polarimetry"
    - van Straten 2009: "High-fidelity polarimetry using SAM"

    Raises
    ------
    NotImplementedError
        Mueller conversion deferred to future version.
    """
    raise NotImplementedError(
        "Mueller matrix conversion not implemented. "
        "For RIME, use Jones matrices directly via apply_jones_matrices()."
    )
