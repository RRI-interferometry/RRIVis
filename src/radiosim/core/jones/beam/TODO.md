# Beam System — Future Work (v5.0+)

## 2. Cross-Polarization Models
Previously implemented in `crosspol.py`, removed to reduce complexity. To be
re-added when calibration/polarimetry features require it.

### Quadrupolar Model
- Dominant cross-pol pattern for linearly-polarized feeds
- `epsilon(theta) = epsilon_0 * (theta / theta_HPBW)^2`
- Azimuthal dependence: `cross_pol = epsilon(theta) * sin(2*phi)`
- Vanishes on principal planes (phi = 0, pi/2), peaks at phi = pi/4
- Parameter: `epsilon_0` (peak leakage amplitude at HPBW, e.g. 0.01 = 1%)

### IXR (Intrinsic Cross-polarization Ratio) Conversion
- Basis-independent polarimetric fidelity measure (Carozzi & Woan 2011)
- Convert IXR in dB to leakage epsilon: `|eps| = (sqrt(IXR_lin) - 1) / (sqrt(IXR_lin) + 1)`
- Feeds into quadrupolar model with IXR-derived epsilon
- Parameter: `target_ixr_dB` (e.g. 30 dB)

### Ludwig-3 Decomposition
- Standard co/cross definition for linearly-polarized antennas (Ludwig 1973)
- `E_co = E_theta * cos(phi) - E_phi * sin(phi)`
- `E_cross = E_theta * sin(phi) + E_phi * cos(phi)`
- Preserves total power: `|E_co|^2 + |E_cross|^2 = |E_theta|^2 + |E_phi|^2`

### Jones Matrix Assembly
- Without cross-pol: `J = [[co, 0], [0, co]]` (diagonal)
- With cross-pol: `J = [[co, cross], [-cross, co]]` (opposite signs for physical consistency)
- Y-feed cross-pol has opposite parity to X-feed

### Config Fields (when re-added)
- `crosspol_model`: `"none"`, `"quadrupolar"`, `"ixr"`, `"ludwig3"`
- `crosspol_params`: `{"epsilon_0": 0.01}` or `{"target_ixr_dB": 30.0}`

### References
- Carozzi & Woan (2011) IEEE Trans. Antennas Propag. 59, 2058
- Ludwig (1973) IEEE Trans. Antennas Propag. AP-21, 116-119
- Hamaker, Bregman & Sault (1996) A&AS 117, 137-147

## 3. Near/Far Field Regime
- Fresnel diffraction integral for R < 2D²/λ
- Fresnel number N_F = D²/(4λR)
- On-axis oscillation: I(R) = 4·sin²(π·N_F/2)
- Not needed for astronomical sources (always far field)
- Relevant for: holography, antenna test ranges, drone calibration

## 4. Aperture Blockage
- Subreflector/feed shadow: removes central region from aperture integral
- CASA model: E_blocked = E_aperture - (d/D)² × E_subreflector_pattern
- η_b ≈ (1 − (d_block/D)²)²
- Support legs: 4-fold diffraction spikes
- Requires: blockage_diameter, n_support_legs, support_leg_width config fields

## 5. Random Surface Errors (Ruze Effect)
- Ruze equation: η_s = exp(−(4πσ/λ)²)
- Decomposes beam: B_total = exp(−σ_φ²)·B_main + error_beam
- Error beam: Gaussian with FWHM ≈ 0.53λ/L (L = correlation length / panel size)
- Rule of thumb: λ_min ≈ 10σ
- Requires: rms_surface_error_m, correlation_length_m config fields

## 6. Systematic Aberrations
- Defocus: quadratic phase error from axial feed displacement
- Coma: asymmetric beam from lateral feed displacement
- Astigmatism: elliptical beam from dish warping
- Gravitational sag: elevation-dependent deformation
- Zernike polynomial expansion of phase error across aperture
- Requires: aberration_coefficients or gravity_model config

## 9. Beam Squint
- RCP/LCP beam offset: θ_squint ≈ d_offset/(2f)
- Spurious Stokes V: V_meas = V_true + (1/2)·I·(dA/dn)·θ_squint
- Rotates with parallactic angle for alt-az mounts
- Requires: squint_angle_rad, squint_pa_deg config fields
- Reference: Cotton & Uson 2008 (arXiv:0807.0026)

## 13. Pointing Errors
- Gain reduction: <G/G₀> = [1 + 4·ln2·(σ_p/θ_HPBW)²]⁻¹
- For 5% accuracy: σ_p < 0.10·θ_HPBW per axis
- Sources: structural flexure, encoder errors, atmospheric refraction
- Requires: pointing_rms_rad config field
