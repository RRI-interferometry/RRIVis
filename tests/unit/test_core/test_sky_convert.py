"""Tests for rrivis.core.sky.convert — bidirectional HEALPix / point-source conversion."""

import healpy as hp
import numpy as np

from rrivis.core.sky.constants import (
    C_LIGHT,
    K_BOLTZMANN,
    brightness_temp_to_flux_density,
    flux_density_to_brightness_temp,
)
from rrivis.core.sky.convert import (
    healpix_map_to_point_arrays,
    point_sources_to_healpix_maps,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

FREQ_408MHZ = 408e6  # Hz
FREQ_100MHZ = 100e6
FREQ_200MHZ = 200e6


def _omega(nside: int) -> float:
    """Pixel solid angle for a HEALPix grid."""
    npix = hp.nside2npix(nside)
    return 4 * np.pi / npix


# ===========================================================================
# TestHealpixMapToPointArrays
# ===========================================================================


class TestHealpixMapToPointArrays:
    """Tests for healpix_map_to_point_arrays."""

    # 1. Uniform temperature map
    def test_uniform_temperature_map(self):
        nside = 8
        npix = hp.nside2npix(nside)
        temp_map = np.full(npix, 100.0)

        result = healpix_map_to_point_arrays(temp_map, FREQ_408MHZ, "rayleigh-jeans")

        assert len(result["ra_rad"]) == npix
        assert len(result["flux_ref"]) == npix
        # All fluxes should be identical (uniform temperature)
        np.testing.assert_allclose(
            result["flux_ref"],
            result["flux_ref"][0],
            rtol=1e-12,
            err_msg="Uniform map should produce identical flux for every pixel",
        )

    # 2. Single hot pixel
    def test_single_hot_pixel(self):
        nside = 8
        npix = hp.nside2npix(nside)
        hot_pixel = 42
        temp_map = np.zeros(npix)
        temp_map[hot_pixel] = 1000.0

        result = healpix_map_to_point_arrays(temp_map, FREQ_408MHZ, "rayleigh-jeans")

        assert len(result["flux_ref"]) == 1

        # Verify coordinate matches hp.pix2ang
        theta, phi = hp.pix2ang(nside, hot_pixel)
        expected_ra = phi
        expected_dec = np.pi / 2 - theta
        np.testing.assert_allclose(result["ra_rad"][0], expected_ra, atol=1e-14)
        np.testing.assert_allclose(result["dec_rad"][0], expected_dec, atol=1e-14)

        # Verify flux matches the direct conversion
        omega = _omega(nside)
        expected_flux = brightness_temp_to_flux_density(
            np.array([1000.0]), FREQ_408MHZ, omega, method="rayleigh-jeans"
        )
        np.testing.assert_allclose(result["flux_ref"][0], expected_flux[0], rtol=1e-12)

    # 3. Empty map (all zeros)
    def test_empty_map_all_zeros(self):
        nside = 8
        npix = hp.nside2npix(nside)
        temp_map = np.zeros(npix)

        result = healpix_map_to_point_arrays(temp_map, FREQ_408MHZ, "rayleigh-jeans")

        for key in (
            "ra_rad",
            "dec_rad",
            "flux_ref",
            "alpha",
            "stokes_q",
            "stokes_u",
            "stokes_v",
        ):
            assert len(result[key]) == 0, (
                f"Key {key!r} should be empty for all-zero map"
            )

    # 4. Negative temperatures ignored
    def test_negative_temperatures_ignored(self):
        nside = 8
        npix = hp.nside2npix(nside)
        temp_map = np.full(npix, -100.0)
        # Set exactly 10 pixels positive
        positive_count = 10
        temp_map[:positive_count] = 50.0

        result = healpix_map_to_point_arrays(temp_map, FREQ_408MHZ, "rayleigh-jeans")

        assert len(result["flux_ref"]) == positive_count
        assert np.all(result["flux_ref"] > 0)

    # 5. Coordinate accuracy
    def test_coordinate_accuracy(self):
        nside = 16
        npix = hp.nside2npix(nside)
        pixel_idx = 42
        temp_map = np.zeros(npix)
        temp_map[pixel_idx] = 500.0

        result = healpix_map_to_point_arrays(temp_map, FREQ_408MHZ, "rayleigh-jeans")

        assert len(result["ra_rad"]) == 1

        theta, phi = hp.pix2ang(nside, pixel_idx)
        expected_ra = phi
        expected_dec = np.pi / 2 - theta
        np.testing.assert_allclose(result["ra_rad"][0], expected_ra, atol=1e-14)
        np.testing.assert_allclose(result["dec_rad"][0], expected_dec, atol=1e-14)

    # 6. Polarization extraction
    def test_polarization_extraction(self):
        nside = 8
        npix = hp.nside2npix(nside)
        freq = FREQ_408MHZ
        omega = _omega(nside)

        # All pixels hot so all are valid
        temp_map = np.full(npix, 200.0)

        # Q map with known values (shape (1, npix) -- single frequency)
        q_values = np.linspace(0.1, 1.0, npix).reshape(1, npix)

        result = healpix_map_to_point_arrays(
            temp_map,
            freq,
            "rayleigh-jeans",
            healpix_q_maps=q_values,
            freq_index=0,
        )

        rj_factor = (2 * K_BOLTZMANN * freq**2 / C_LIGHT**2) * omega / 1e-26

        # All npix pixels are valid and ordered by pixel index (0..npix-1)
        expected_q = q_values[0] * rj_factor
        np.testing.assert_allclose(result["stokes_q"], expected_q, rtol=1e-10)

    # 7. Planck vs RJ conversion
    def test_planck_vs_rj_conversion(self):
        nside = 8
        npix = hp.nside2npix(nside)
        temp_map = np.full(npix, 100.0)

        result_rj = healpix_map_to_point_arrays(temp_map, FREQ_408MHZ, "rayleigh-jeans")
        result_planck = healpix_map_to_point_arrays(temp_map, FREQ_408MHZ, "planck")

        # At 408 MHz/100K, h*nu / k*T ~ 2e-4, so RJ is very close to Planck
        # but they should NOT be identical.
        flux_rj = result_rj["flux_ref"]
        flux_planck = result_planck["flux_ref"]

        assert not np.allclose(flux_rj, flux_planck, atol=0), (
            "Planck and RJ should give numerically different results"
        )

        # At these low frequencies/high temperatures, RJ slightly overestimates
        # because it neglects the exponential cutoff. Planck < RJ.
        assert np.all(flux_rj > flux_planck), (
            "RJ should give slightly higher flux than Planck at 408 MHz / 100 K"
        )


# ===========================================================================
# TestPointSourcesToHealpixMaps
# ===========================================================================


class TestPointSourcesToHealpixMaps:
    """Tests for point_sources_to_healpix_maps."""

    # 8. Single source single pixel
    def test_single_source_single_pixel(self):
        nside = 16
        npix = hp.nside2npix(nside)
        ra = np.array([0.0])
        dec = np.array([0.0])
        flux = np.array([5.0])
        alpha = np.array([0.0])
        freq = np.array([FREQ_100MHZ])

        i_maps, q_maps, u_maps, v_maps = point_sources_to_healpix_maps(
            ra,
            dec,
            flux,
            alpha,
            spectral_coeffs=None,
            stokes_q=None,
            stokes_u=None,
            stokes_v=None,
            rotation_measure=None,
            nside=nside,
            frequencies=freq,
            ref_frequency=FREQ_100MHZ,
            brightness_conversion="rayleigh-jeans",
        )

        assert i_maps.shape == (1, npix)

        # Only one pixel should be non-zero
        nonzero_pixels = np.count_nonzero(i_maps[0])
        assert nonzero_pixels == 1

        # The non-zero pixel index should match hp.ang2pix for (RA=0, Dec=0)
        expected_pix = hp.ang2pix(nside, np.pi / 2 - 0.0, 0.0)
        assert i_maps[0, expected_pix] > 0

    # 9. Two sources same pixel
    def test_two_sources_same_pixel(self):
        nside = 16
        # Both sources at the same position
        ra = np.array([1.0, 1.0])
        dec = np.array([0.5, 0.5])
        flux = np.array([3.0, 2.0])
        alpha = np.array([0.0, 0.0])
        freq = np.array([FREQ_100MHZ])

        i_maps, _, _, _ = point_sources_to_healpix_maps(
            ra,
            dec,
            flux,
            alpha,
            spectral_coeffs=None,
            stokes_q=None,
            stokes_u=None,
            stokes_v=None,
            rotation_measure=None,
            nside=nside,
            frequencies=freq,
            ref_frequency=FREQ_100MHZ,
            brightness_conversion="rayleigh-jeans",
        )

        pix = hp.ang2pix(nside, np.pi / 2 - 0.5, 1.0)
        omega = _omega(nside)
        expected_temp = flux_density_to_brightness_temp(
            np.array([5.0]), FREQ_100MHZ, omega, method="rayleigh-jeans"
        )
        np.testing.assert_allclose(
            i_maps[0, pix],
            expected_temp[0],
            rtol=1e-4,
        )

    # 10. Spectral scaling across channels
    def test_spectral_scaling_across_channels(self):
        nside = 16
        ra = np.array([0.0])
        dec = np.array([0.0])
        flux = np.array([10.0])  # 10 Jy at ref
        alpha_val = -0.7
        alpha = np.array([alpha_val])
        ref_freq = FREQ_100MHZ
        freqs = np.array([FREQ_100MHZ, FREQ_200MHZ])

        i_maps, _, _, _ = point_sources_to_healpix_maps(
            ra,
            dec,
            flux,
            alpha,
            spectral_coeffs=None,
            stokes_q=None,
            stokes_u=None,
            stokes_v=None,
            rotation_measure=None,
            nside=nside,
            frequencies=freqs,
            ref_frequency=ref_freq,
            brightness_conversion="rayleigh-jeans",
        )

        pix = hp.ang2pix(nside, np.pi / 2 - 0.0, 0.0)

        # At ref_freq the flux is 10 Jy. At 200 MHz it should be 10 * 2^(-0.7).
        # The map stores T_b, but in RJ regime T_b ~ S / (2 k_B nu^2 / c^2 * omega)
        # so T_b_200 / T_b_100 = (S_200 / S_100) * (nu_100 / nu_200)^2
        # S_200/S_100 = 2^(-0.7), nu_100/nu_200 = 0.5
        # ratio = 2^(-0.7) * 0.25
        expected_ratio = (2**alpha_val) * (FREQ_100MHZ / FREQ_200MHZ) ** 2
        actual_ratio = i_maps[1, pix] / i_maps[0, pix]
        np.testing.assert_allclose(actual_ratio, expected_ratio, rtol=1e-4)

    # 11. Empty sources
    def test_empty_sources(self):
        nside = 8
        npix = hp.nside2npix(nside)
        n_freq = 3
        freqs = np.array([100e6, 150e6, 200e6])

        i_maps, q_maps, u_maps, v_maps = point_sources_to_healpix_maps(
            ra_rad=np.array([]),
            dec_rad=np.array([]),
            flux_ref=np.array([]),
            alpha=np.array([]),
            spectral_coeffs=None,
            stokes_q=None,
            stokes_u=None,
            stokes_v=None,
            rotation_measure=None,
            nside=nside,
            frequencies=freqs,
            ref_frequency=100e6,
            brightness_conversion="rayleigh-jeans",
        )

        assert i_maps.shape == (n_freq, npix)
        assert np.all(i_maps == 0)
        assert q_maps is None
        assert u_maps is None
        assert v_maps is None

    # 12. Output shape
    def test_output_shape(self):
        nside = 32
        npix = hp.nside2npix(nside)
        n_freq = 5
        freqs = np.linspace(100e6, 200e6, n_freq)

        ra = np.array([0.5])
        dec = np.array([0.2])
        flux = np.array([1.0])
        alpha = np.array([-0.7])

        i_maps, _, _, _ = point_sources_to_healpix_maps(
            ra,
            dec,
            flux,
            alpha,
            spectral_coeffs=None,
            stokes_q=None,
            stokes_u=None,
            stokes_v=None,
            rotation_measure=None,
            nside=nside,
            frequencies=freqs,
            ref_frequency=100e6,
            brightness_conversion="rayleigh-jeans",
        )

        assert i_maps.shape == (n_freq, npix)
        assert npix == 12 * 32**2  # 12288

    # 13. Unpolarized returns None
    def test_unpolarized_returns_none(self):
        nside = 8
        freqs = np.array([100e6])

        ra = np.array([0.0, 1.0])
        dec = np.array([0.0, 0.5])
        flux = np.array([5.0, 3.0])
        alpha = np.array([0.0, 0.0])

        # All stokes Q/U/V are zero
        _, q_maps, u_maps, v_maps = point_sources_to_healpix_maps(
            ra,
            dec,
            flux,
            alpha,
            spectral_coeffs=None,
            stokes_q=np.zeros(2),
            stokes_u=np.zeros(2),
            stokes_v=np.zeros(2),
            rotation_measure=None,
            nside=nside,
            frequencies=freqs,
            ref_frequency=100e6,
            brightness_conversion="rayleigh-jeans",
        )

        assert q_maps is None
        assert u_maps is None
        assert v_maps is None

    # 14. Polarized sources generate maps
    def test_polarized_sources_generate_maps(self):
        nside = 8
        npix = hp.nside2npix(nside)
        freqs = np.array([100e6])

        ra = np.array([0.0, 1.0])
        dec = np.array([0.0, 0.5])
        flux = np.array([5.0, 3.0])
        alpha = np.array([0.0, 0.0])

        i_maps, q_maps, u_maps, v_maps = point_sources_to_healpix_maps(
            ra,
            dec,
            flux,
            alpha,
            spectral_coeffs=None,
            stokes_q=np.array([0.1, 0.2]),
            stokes_u=np.array([0.05, 0.0]),
            stokes_v=np.array([0.0, 0.0]),
            rotation_measure=None,
            nside=nside,
            frequencies=freqs,
            ref_frequency=100e6,
            brightness_conversion="rayleigh-jeans",
        )

        assert q_maps is not None
        assert u_maps is not None
        assert v_maps is not None
        assert q_maps.shape == (1, npix)
        assert u_maps.shape == (1, npix)
        assert v_maps.shape == (1, npix)
        # The maps should match i_maps shape
        assert q_maps.shape == i_maps.shape

    # 15. Round-trip flux conservation
    def test_round_trip_flux_conservation(self):
        """Convert point sources -> healpix -> point sources. Total flux conserved."""
        nside = 16
        rng = np.random.default_rng(42)
        n_sources = 10
        freq = FREQ_100MHZ
        ref_freq = FREQ_100MHZ

        # Random sky positions spread across the sky
        ra = rng.uniform(0, 2 * np.pi, n_sources)
        dec = rng.uniform(-np.pi / 2, np.pi / 2, n_sources)
        flux = rng.uniform(1.0, 10.0, n_sources)
        alpha = np.zeros(n_sources)
        total_flux_in = np.sum(flux)

        # Forward: point sources -> healpix
        i_maps, _, _, _ = point_sources_to_healpix_maps(
            ra,
            dec,
            flux,
            alpha,
            spectral_coeffs=None,
            stokes_q=None,
            stokes_u=None,
            stokes_v=None,
            rotation_measure=None,
            nside=nside,
            frequencies=np.array([freq]),
            ref_frequency=ref_freq,
            brightness_conversion="rayleigh-jeans",
        )

        # Reverse: healpix -> point sources
        result = healpix_map_to_point_arrays(i_maps[0], freq, "rayleigh-jeans")

        total_flux_out = np.sum(result["flux_ref"])

        # Flux should be conserved within 5% (pixelization can merge sources)
        np.testing.assert_allclose(
            total_flux_out,
            total_flux_in,
            rtol=0.05,
            err_msg="Round-trip total flux should be conserved within 5%",
        )

    # 16. Bincount accumulation
    def test_bincount_accumulation(self):
        """Three sources mapped to the same pixel accumulate correctly."""
        nside = 16
        # All three sources at the exact same position
        ra = np.array([0.5, 0.5, 0.5])
        dec = np.array([0.3, 0.3, 0.3])
        flux = np.array([2.0, 3.0, 5.0])  # total = 10 Jy
        alpha = np.zeros(3)
        freq = np.array([FREQ_100MHZ])

        i_maps, _, _, _ = point_sources_to_healpix_maps(
            ra,
            dec,
            flux,
            alpha,
            spectral_coeffs=None,
            stokes_q=None,
            stokes_u=None,
            stokes_v=None,
            rotation_measure=None,
            nside=nside,
            frequencies=freq,
            ref_frequency=FREQ_100MHZ,
            brightness_conversion="rayleigh-jeans",
        )

        pix = hp.ang2pix(nside, np.pi / 2 - 0.3, 0.5)
        omega = _omega(nside)

        # Temperature for summed flux (10 Jy)
        expected_temp = flux_density_to_brightness_temp(
            np.array([10.0]), FREQ_100MHZ, omega, method="rayleigh-jeans"
        )
        np.testing.assert_allclose(
            i_maps[0, pix],
            expected_temp[0],
            rtol=1e-4,
        )

        # Exactly one non-zero pixel
        assert np.count_nonzero(i_maps[0]) == 1

    # 17. Unoccupied pixels are zero
    def test_unoccupied_pixels_are_zero(self):
        """All pixels except the source pixel must be exactly 0.0."""
        nside = 16
        npix = hp.nside2npix(nside)
        ra = np.array([1.0])
        dec = np.array([0.5])
        flux = np.array([7.0])
        alpha = np.array([0.0])
        freq = np.array([FREQ_100MHZ])

        i_maps, _, _, _ = point_sources_to_healpix_maps(
            ra,
            dec,
            flux,
            alpha,
            spectral_coeffs=None,
            stokes_q=None,
            stokes_u=None,
            stokes_v=None,
            rotation_measure=None,
            nside=nside,
            frequencies=freq,
            ref_frequency=FREQ_100MHZ,
            brightness_conversion="rayleigh-jeans",
        )

        source_pix = hp.ang2pix(nside, np.pi / 2 - 0.5, 1.0)
        assert i_maps[0, source_pix] > 0, "Source pixel should have positive T_b"

        # Every other pixel must be exactly zero
        mask = np.ones(npix, dtype=bool)
        mask[source_pix] = False
        assert np.all(i_maps[0, mask] == 0.0), (
            "All unoccupied pixels should be exactly 0.0"
        )


# ===========================================================================
# TestHealpixMapToPointArrays — polarization via observation_frequencies
# ===========================================================================


class TestHealpixPolarizationFreqResolution:
    """Tests that hit the observation_frequencies / freq_index branches."""

    def test_polarization_with_observation_frequencies(self):
        """Providing observation_frequencies (no explicit freq_index) resolves
        the nearest index and extracts Q correctly (line 114)."""
        nside = 8
        npix = hp.nside2npix(nside)
        freq = FREQ_100MHZ
        omega = _omega(nside)

        # Build a map with a few hot pixels
        temp_map = np.zeros(npix)
        hot_pixels = np.array([10, 20, 30])
        temp_map[hot_pixels] = 500.0

        # Q map: shape (1, npix) -- single frequency slice
        q_data = np.zeros((1, npix))
        q_data[0, hot_pixels] = np.array([0.3, 0.6, 0.9])

        # observation_frequencies provided, freq_index=None -> argmin path
        result = healpix_map_to_point_arrays(
            temp_map,
            freq,
            "rayleigh-jeans",
            healpix_q_maps=q_data,
            observation_frequencies=np.array([FREQ_100MHZ]),
            freq_index=None,
        )

        rj_factor = (2 * K_BOLTZMANN * freq**2 / C_LIGHT**2) * omega / 1e-26

        # valid_idx are the hot pixels (sorted, which hp pixel indices already are)
        expected_q = q_data[0, hot_pixels] * rj_factor
        np.testing.assert_allclose(result["stokes_q"], expected_q, rtol=1e-10)

        # U and V should be zero (not provided)
        np.testing.assert_array_equal(result["stokes_u"], 0.0)
        np.testing.assert_array_equal(result["stokes_v"], 0.0)

    def test_full_quv_extraction(self):
        """Provide all 3 polarization maps (Q, U, V) with observation_frequencies;
        verify all three are extracted and correctly scaled (lines 118, 123, 128)."""
        nside = 8
        npix = hp.nside2npix(nside)
        freq = FREQ_100MHZ
        omega = _omega(nside)

        temp_map = np.zeros(npix)
        hot_pixels = np.array([5, 15, 25, 35])
        temp_map[hot_pixels] = 300.0

        q_data = np.zeros((1, npix))
        u_data = np.zeros((1, npix))
        v_data = np.zeros((1, npix))
        q_data[0, hot_pixels] = np.array([0.1, 0.2, 0.3, 0.4])
        u_data[0, hot_pixels] = np.array([0.05, 0.15, 0.25, 0.35])
        v_data[0, hot_pixels] = np.array([0.01, 0.02, 0.03, 0.04])

        result = healpix_map_to_point_arrays(
            temp_map,
            freq,
            "rayleigh-jeans",
            healpix_q_maps=q_data,
            healpix_u_maps=u_data,
            healpix_v_maps=v_data,
            observation_frequencies=np.array([FREQ_100MHZ]),
            freq_index=None,
        )

        rj_factor = (2 * K_BOLTZMANN * freq**2 / C_LIGHT**2) * omega / 1e-26

        np.testing.assert_allclose(
            result["stokes_q"], q_data[0, hot_pixels] * rj_factor, rtol=1e-10
        )
        np.testing.assert_allclose(
            result["stokes_u"], u_data[0, hot_pixels] * rj_factor, rtol=1e-10
        )
        np.testing.assert_allclose(
            result["stokes_v"], v_data[0, hot_pixels] * rj_factor, rtol=1e-10
        )

        # All three should be non-zero
        assert np.all(result["stokes_q"] != 0)
        assert np.all(result["stokes_u"] != 0)
        assert np.all(result["stokes_v"] != 0)
