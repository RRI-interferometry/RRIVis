# test_polarization.py
"""
Tests for polarization module (Jones matrices and coherency matrices).

Validates:
1. Stokes ↔ coherency round-trip
2. Energy conservation (half-power convention)
3. Africanus/Pauli V sign convention
4. Broadcasting behavior
5. Edge cases (unpolarized, fully polarized, etc.)
"""

import numpy as np
import pytest
from polarization import (
    stokes_to_coherency,
    coherency_to_stokes,
    apply_jones_matrices,
    visibility_to_correlations,
    stokes_I_only_visibility,
    jones_matrix_power,
)


class TestStokesToCoherency:
    """Test Stokes parameter to coherency matrix conversion."""

    def test_unpolarized_source(self):
        """Unpolarized source: Q=U=V=0."""
        I = 10.0
        C = stokes_to_coherency(I)

        # Should be diagonal with I/2 on each element
        assert np.allclose(C[0, 0].real, I / 2), "C[0,0] should be I/2"
        assert np.allclose(C[1, 1].real, I / 2), "C[1,1] should be I/2"
        assert np.allclose(C[0, 1], 0), "C[0,1] should be zero"
        assert np.allclose(C[1, 0], 0), "C[1,0] should be zero"

        # Trace should equal I (energy conservation)
        assert np.allclose(np.trace(C).real, I), "Trace(C) must equal I"

    def test_fully_Q_polarized(self):
        """Fully Q-polarized: Q=I, U=V=0 (all in X feed)."""
        I, Q = 10.0, 10.0
        C = stokes_to_coherency(I, Q=Q)

        # All power in XX
        assert np.allclose(C[0, 0].real, I), "C[0,0] = I (all in X)"
        assert np.allclose(C[1, 1].real, 0), "C[1,1] = 0 (none in Y)"
        assert np.allclose(C[0, 1], 0), "Off-diagonal zero"
        assert np.allclose(np.trace(C).real, I), "Trace = I"

    def test_africanus_V_convention(self):
        """Verify Africanus convention: C[0,1] = (U - iV)/2."""
        I, U, V = 1.0, 0.0, 1.0  # Pure circular polarization

        C = stokes_to_coherency(I, U=U, V=V)

        # Africanus: C[0,1] = (U - iV)/2 = (0 - i)/2 = -i/2
        expected_01 = (U - 1j * V) / 2
        assert np.allclose(C[0, 1], expected_01), "Africanus: C[0,1] = (U-iV)/2"

        # C[1,0] should be conjugate
        assert np.allclose(C[1, 0], np.conj(expected_01)), "C[1,0] = conj(C[0,1])"

    def test_energy_conservation(self):
        """Trace(C) = I for any polarization state."""
        test_cases = [
            (10.0, 0, 0, 0),  # Unpolarized
            (10.0, 5.0, 0, 0),  # Q polarized
            (10.0, 0, 3.0, 0),  # U polarized
            (10.0, 0, 0, 2.0),  # V polarized
            (10.0, 2.0, -3.0, 1.5),  # Mixed
        ]

        for I, Q, U, V in test_cases:
            C = stokes_to_coherency(I, Q, U, V)
            trace = np.trace(C).real
            assert np.allclose(
                trace, I, rtol=1e-10
            ), f"Energy not conserved for I={I}, Q={Q}, U={U}, V={V}"

    def test_hermitian_property(self):
        """Coherency matrix must be Hermitian: C = C^H."""
        I, Q, U, V = 5.0, 1.0, -0.5, 0.3
        C = stokes_to_coherency(I, Q, U, V)

        C_H = np.conj(C.T)
        assert np.allclose(C, C_H), "Coherency must be Hermitian"

    def test_array_broadcasting(self):
        """Test broadcasting with array inputs."""
        Nsrc = 100
        I = np.random.uniform(1, 10, Nsrc)
        Q = np.random.uniform(-I, I)
        U = np.zeros(Nsrc)
        V = np.zeros(Nsrc)

        C = stokes_to_coherency(I, Q, U, V)

        assert C.shape == (Nsrc, 2, 2), f"Expected shape (100, 2, 2), got {C.shape}"

        # Check energy conservation for all sources
        traces = np.trace(C, axis1=-2, axis2=-1).real
        assert np.allclose(traces, I), "Energy not conserved in array"


class TestCoherencyToStokes:
    """Test coherency to Stokes conversion (inverse operation)."""

    def test_roundtrip_unpolarized(self):
        """Stokes → coherency → Stokes should be identity."""
        I, Q, U, V = 10.0, 0.0, 0.0, 0.0

        C = stokes_to_coherency(I, Q, U, V)
        I2, Q2, U2, V2 = coherency_to_stokes(C)

        assert np.allclose([I, Q, U, V], [I2, Q2, U2, V2], rtol=1e-10)

    def test_roundtrip_polarized(self):
        """Round-trip with full polarization."""
        I, Q, U, V = 10.0, 2.0, -3.0, 1.5

        C = stokes_to_coherency(I, Q, U, V)
        I2, Q2, U2, V2 = coherency_to_stokes(C)

        assert np.allclose([I, Q, U, V], [I2, Q2, U2, V2], rtol=1e-10)

    def test_roundtrip_array(self):
        """Round-trip with array inputs."""
        Nsrc = 50
        I = np.random.uniform(1, 10, Nsrc)
        Q = np.random.uniform(-I, I)
        U = np.random.uniform(-np.sqrt(I**2 - Q**2), np.sqrt(I**2 - Q**2))
        V = np.random.uniform(-1, 1, Nsrc)

        C = stokes_to_coherency(I, Q, U, V)
        I2, Q2, U2, V2 = coherency_to_stokes(C)

        assert np.allclose(I, I2, rtol=1e-10)
        assert np.allclose(Q, Q2, rtol=1e-10)
        assert np.allclose(U, U2, rtol=1e-10)
        assert np.allclose(V, V2, rtol=1e-10)


class TestApplyJonesMatrices:
    """Test Jones matrix application to coherency."""

    def test_ideal_instrument_unpolarized(self):
        """Ideal instrument (J=I) with unpolarized source."""
        I = 10.0
        C = stokes_to_coherency(I)
        J_ideal = np.eye(2, dtype=complex)

        V = apply_jones_matrices(J_ideal, C, J_ideal)

        # Should equal coherency matrix for ideal instrument
        assert np.allclose(V, C)

    def test_leaky_beam_creates_crosspol(self):
        """Leaky Jones matrix creates cross-pol even for unpolarized source."""
        I = 10.0
        C = stokes_to_coherency(I)  # Unpolarized: C[0,1] = 0

        # Leaky beam (off-diagonal elements)
        J_leaky = np.array([[0.95, 0.05], [0.03, 0.97]], dtype=complex)

        V = apply_jones_matrices(J_leaky, C, J_leaky)

        # Cross-pol should be non-zero due to leakage
        assert not np.allclose(V[0, 1], 0), "Leakage should create cross-pol"
        assert not np.allclose(V[1, 0], 0)

    def test_hermitian_visibility(self):
        """Visibility matrix should be Hermitian for autocorrelations (same antenna).

        For autocorrelations where J_i = J_j, the visibility V = J * C * J^H
        should be Hermitian since C is Hermitian.
        """
        I, Q, U, V = 5.0, 1.0, -0.5, 0.3
        C = stokes_to_coherency(I, Q, U, V)

        # Use same Jones matrix for autocorrelation
        J = np.array([[0.9 + 0.1j, 0.05], [0.02, 0.95 - 0.05j]])

        vis = apply_jones_matrices(J, C, J)

        # Autocorrelation visibility should be Hermitian
        vis_H = np.conj(vis.T)
        assert np.allclose(vis, vis_H, atol=1e-10), "Autocorrelation visibility must be Hermitian"

    def test_array_broadcasting(self):
        """Test broadcasting with multiple sources."""
        Nsrc = 10
        I = np.random.uniform(1, 10, Nsrc)
        C = stokes_to_coherency(I)  # Shape: (10, 2, 2)

        # Same Jones matrices for all sources
        J_i = np.eye(2, dtype=complex)
        J_j = np.eye(2, dtype=complex)

        # Should broadcast J across sources
        V = apply_jones_matrices(J_i, C, J_j)

        assert V.shape == (Nsrc, 2, 2)
        # For ideal instrument, V = C
        assert np.allclose(V, C)


class TestVisibilityToCorrelations:
    """Test extraction of correlation products."""

    def test_unpolarized_ideal_instrument(self):
        """Unpolarized source, ideal instrument."""
        I = 10.0
        C = stokes_to_coherency(I)
        J = np.eye(2, dtype=complex)
        V = apply_jones_matrices(J, C, J)

        corr = visibility_to_correlations(V)

        # XX and YY should each be I/2
        assert np.allclose(corr["XX"], I / 2), "XX should be I/2"
        assert np.allclose(corr["YY"], I / 2), "YY should be I/2"

        # Cross-pols should be zero
        assert np.allclose(corr["XY"], 0), "XY should be zero"
        assert np.allclose(corr["YX"], 0), "YX should be zero"

        # CRITICAL: Stokes I = XX + YY (no division!)
        assert np.allclose(corr["I"], I), "I = XX + YY (energy conservation)"

    def test_fully_Q_polarized(self):
        """Fully Q-polarized source (all in X)."""
        I, Q = 10.0, 10.0
        C = stokes_to_coherency(I, Q=Q)
        J = np.eye(2, dtype=complex)
        V = apply_jones_matrices(J, C, J)

        corr = visibility_to_correlations(V)

        # All power in XX
        assert np.allclose(corr["XX"], I), "XX = I for Q=I"
        assert np.allclose(corr["YY"], 0), "YY = 0 for Q=I"
        assert np.allclose(corr["I"], I), "Total intensity still I"

    def test_stokes_I_no_division_by_2(self):
        """Verify Stokes I = XX + YY (not (XX+YY)/2)."""
        # This is the critical test for half-power convention
        test_cases = [
            (10.0, 0, 0, 0),  # Unpolarized
            (10.0, 5.0, 0, 0),  # Q
            (10.0, 0, 3.0, 0),  # U
            (10.0, 0, 0, 2.0),  # V
        ]

        J = np.eye(2, dtype=complex)

        for I, Q, U, V in test_cases:
            C = stokes_to_coherency(I, Q, U, V)
            vis = apply_jones_matrices(J, C, J)
            corr = visibility_to_correlations(vis)

            # CRITICAL: Must be sum, not average
            assert np.allclose(
                corr["I"], I, rtol=1e-10
            ), f"Failed for I={I}, Q={Q}, U={U}, V={V}"
            assert np.allclose(corr["I"], corr["XX"] + corr["YY"], rtol=1e-10)


class TestStokesIOnlyVisibility:
    """Test optimized unpolarized source calculation."""

    def test_matches_full_calculation(self):
        """Stokes_I_only should match full coherency for unpolarized."""
        I = 10.0
        J_i = np.array([[0.95 + 0.05j, 0.02], [0.01, 0.98 - 0.03j]])
        J_j = np.array([[0.97, 0.03], [0.02, 0.99]])

        # Full calculation
        C = stokes_to_coherency(I)
        V_full = apply_jones_matrices(J_i, C, J_j)

        # Optimized calculation
        V_fast = stokes_I_only_visibility(J_i, J_j, I)

        assert np.allclose(V_full, V_fast, rtol=1e-10)

    def test_leakage_with_unpolarized(self):
        """Even unpolarized source shows cross-pol with leaky beam."""
        I = 10.0
        J_leaky = np.array([[0.9, 0.1], [0.08, 0.92]])

        V = stokes_I_only_visibility(J_leaky, J_leaky, I)

        # Should have non-zero cross-pol due to leakage
        assert not np.allclose(V[0, 1], 0)


class TestJonesMatrixPower:
    """Test E-field to power beam conversion."""

    def test_ideal_dipole(self):
        """Ideal dipole: J = I → power = 1."""
        J = np.eye(2, dtype=complex)
        px, py = jones_matrix_power(J)

        assert np.allclose(px, 1.0)
        assert np.allclose(py, 1.0)

    def test_attenuated_beam(self):
        """Attenuated beam."""
        J = np.array([[0.9, 0], [0, 0.8]], dtype=complex)
        px, py = jones_matrix_power(J)

        assert np.allclose(px, 0.81)  # |0.9|^2
        assert np.allclose(py, 0.64)  # |0.8|^2

    def test_phase_ignored_in_power(self):
        """Phase is ignored in power calculation."""
        J_no_phase = np.array([[1.0, 0], [0, 1.0]])
        J_with_phase = np.array([[1.0 * np.exp(1j * np.pi / 4), 0], [0, 1.0]])

        px1, py1 = jones_matrix_power(J_no_phase)
        px2, py2 = jones_matrix_power(J_with_phase)

        # Power should be same regardless of phase
        assert np.allclose(px1, px2)
        assert np.allclose(py1, py2)


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_zero_intensity(self):
        """Zero intensity source."""
        C = stokes_to_coherency(I=0.0)
        assert np.allclose(C, 0)

    def test_negative_intensity_allowed(self):
        """Negative I can occur in some formalisms (residuals, etc)."""
        # Should not raise error, but violates physical positivity
        C = stokes_to_coherency(I=-1.0)
        assert C is not None

    def test_scalar_vs_array_consistency(self):
        """Scalar and 1-element array should give same result."""
        I_scalar = 10.0
        I_array = np.array([10.0])

        C_scalar = stokes_to_coherency(I_scalar)
        C_array = stokes_to_coherency(I_array)

        assert C_scalar.shape == (2, 2)
        assert C_array.shape == (1, 2, 2)
        assert np.allclose(C_scalar, C_array[0])


class TestConventionValidation:
    """Validate our convention choices against known results."""

    def test_africanus_convention_sign(self):
        """
        Verify Africanus convention explicitly.

        Africanus: XY correlation = (U - iV)/2
        Smirnov:   XY correlation = (U + iV)/2  (opposite V sign)

        We use Africanus.
        """
        I, U, V = 10.0, 2.0, 1.0
        C = stokes_to_coherency(I, U=U, V=V)

        # Africanus: C[0,1] = (U - iV)/2
        expected = (U - 1j * V) / 2
        assert np.allclose(C[0, 1], expected), "Must match Africanus convention"

        # NOT Smirnov: C[0,1] = (U + iV)/2
        smirnov_wrong = (U + 1j * V) / 2
        assert not np.allclose(C[0, 1], smirnov_wrong), "Must NOT match Smirnov"

    def test_half_power_convention_validated(self):
        """
        Validate half-power convention thoroughly.

        A 1 Jy unpolarized source should produce:
        - V_XX = 0.5 Jy
        - V_YY = 0.5 Jy
        - V_XX + V_YY = 1.0 Jy (energy conserved)

        NOT:
        - V_XX = 1.0 Jy, V_YY = 1.0 Jy → sum = 2 Jy (WRONG!)
        """
        I = 1.0  # 1 Jy source
        C = stokes_to_coherency(I)
        J = np.eye(2, dtype=complex)  # Ideal instrument
        V = apply_jones_matrices(J, C, J)

        # XX and YY should each be 0.5 (half-power)
        assert np.allclose(V[0, 0], 0.5), "V_XX must be 0.5 for 1 Jy unpolarized"
        assert np.allclose(V[1, 1], 0.5), "V_YY must be 0.5 for 1 Jy unpolarized"

        # Total must be 1.0 (energy conserved)
        total = V[0, 0] + V[1, 1]
        assert np.allclose(total, 1.0), "Total must be 1.0 Jy (energy conserved)"

        # Extract Stokes I
        corr = visibility_to_correlations(V)
        assert np.allclose(
            corr["I"], 1.0
        ), "Extracted I must be 1.0 Jy (no factor of 2 error)"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
