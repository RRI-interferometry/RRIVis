"""Tests for ``PointSpectrum`` propagation through ``_combine_models`` to HEALPix.

Before the fix, ``_combine_healpix.combine_healpix`` ignored
``m.point.spectrum`` and silently degraded to power-law extrapolation when
combining a per-channel point catalog into a HEALPix cube.  These tests
verify that the per-channel table is now consulted on both the I and the
polarization paths and that the result agrees with the standalone
``materialize_healpix_model`` (which has always read the spectrum).
"""

from __future__ import annotations

import numpy as np
import pytest

from radiosim.core.precision import PrecisionConfig
from radiosim.core.sky import (
    SkyFormat,
    SkyModel,
    create_from_arrays,
    materialize_healpix_model,
)
from radiosim.core.sky.combine.engine import _combine_models
from radiosim.core.sky.containers.constants import (
    BrightnessConversion,
    rayleigh_jeans_factor,
)
from radiosim.core.sky.containers.data import PointSourceData, PointSpectrum

# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture
def precision() -> PrecisionConfig:
    return PrecisionConfig.standard()


def _make_spectrum_sky(
    *,
    precision: PrecisionConfig,
    flux_per_chan: np.ndarray,
    frequencies: np.ndarray,
    q_per_chan: np.ndarray | None = None,
    u_per_chan: np.ndarray | None = None,
    ra_rad: float = 0.5,
    dec_rad: float = 0.1,
    model_name: str = "spectrum_src",
) -> SkyModel:
    """Build a single-source sky carrying a PointSpectrum table.

    Reference flux/Stokes are set to *deliberately wrong* values (zero or a
    very different number) so any test that returns the right answer can
    only have done so by reading the per-channel table, not the reference
    arrays.
    """
    sky = create_from_arrays(
        ra_rad=np.array([ra_rad]),
        dec_rad=np.array([dec_rad]),
        flux=np.array([0.0]),  # reference deliberately wrong
        spectral_index=np.array([0.0]),
        stokes_q=np.array([0.0]),
        stokes_u=np.array([0.0]),
        stokes_v=np.array([0.0]),
        ref_freq=np.array([frequencies[0]]),
        reference_frequency=float(frequencies[0]),
        precision=precision,
        model_name=model_name,
    )
    n_src = 1
    spectrum = PointSpectrum(
        flux=flux_per_chan.reshape(-1, n_src).astype(np.float64),
        frequencies=frequencies.astype(np.float64),
        stokes_q=(
            q_per_chan.reshape(-1, n_src).astype(np.float64)
            if q_per_chan is not None
            else None
        ),
        stokes_u=(
            u_per_chan.reshape(-1, n_src).astype(np.float64)
            if u_per_chan is not None
            else None
        ),
    )
    new_point = PointSourceData(
        ra_rad=sky.point.ra_rad,
        dec_rad=sky.point.dec_rad,
        flux=sky.point.flux,
        spectral_index=sky.point.spectral_index,
        stokes_q=sky.point.stokes_q,
        stokes_u=sky.point.stokes_u,
        stokes_v=sky.point.stokes_v,
        ref_freq=sky.point.ref_freq,
        spectrum=spectrum,
    )
    return sky.replace(point=new_point)


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #


class TestCombineHealpixUsesSpectrumI:
    def test_per_channel_flux_recovered_at_each_channel(self, precision):
        """Combine a spectrum-bearing source by itself; the resulting Jy
        accumulated into the source's pixel must equal the per-channel
        flux table at every channel.  Power-law extrapolation would give
        a flat-spectrum source (since spectral_index=0 and flux=0), so any
        per-channel variation is proof the spectrum was read.
        """
        nside = 8
        npix = 12 * nside * nside
        omega_pixel = 4 * np.pi / npix
        frequencies = np.array([100e6, 150e6, 200e6, 250e6])
        # Distinct per-channel flux values that no power law could produce
        # from flux=0 at ref_freq.
        flux_per_chan = np.array([3.0, 7.5, 1.25, 11.0])
        sky = _make_spectrum_sky(
            precision=precision,
            flux_per_chan=flux_per_chan,
            frequencies=frequencies,
        )
        combined = _combine_models(
            [sky],
            representation=SkyFormat.HEALPIX,
            nside=nside,
            frequencies=frequencies,
            precision=precision,
            brightness_conversion=BrightnessConversion.RAYLEIGH_JEANS,
        )
        # Convert the K_RJ map back to Jy at each frequency and read out
        # the source's pixel.
        from radiosim.core.sky.containers.constants import (
            brightness_temp_to_flux_density,
        )

        for fi, freq in enumerate(frequencies):
            t_map = combined.healpix.maps[fi]
            jy_map = brightness_temp_to_flux_density(
                t_map.astype(np.float64),
                float(freq),
                omega_pixel,
                method="rayleigh-jeans",
            )
            # All flux concentrated in the single source's pixel.
            recovered = float(jy_map.sum())
            assert recovered == pytest.approx(flux_per_chan[fi], rel=1e-4)

    def test_combine_matches_materialize(self, precision):
        """_combine_models([sky]) into HEALPix must equal
        materialize_healpix_model(sky) when the input carries a spectrum —
        the two paths share exactly one source of truth for spectral
        evaluation.  Both must use the same brightness conversion (the
        sky's own Planck default here) for the comparison to be apples-
        to-apples.
        """
        nside = 8
        frequencies = np.array([100e6, 150e6, 200e6, 250e6])
        flux_per_chan = np.array([3.0, 7.5, 1.25, 11.0])
        sky = _make_spectrum_sky(
            precision=precision,
            flux_per_chan=flux_per_chan,
            frequencies=frequencies,
        )
        via_combine = _combine_models(
            [sky],
            representation=SkyFormat.HEALPIX,
            nside=nside,
            frequencies=frequencies,
            precision=precision,
            # Inherit the sky's Planck brightness convention; passing
            # an override here would compare two different physics paths.
        )
        via_materialize = materialize_healpix_model(
            sky,
            nside=nside,
            frequencies=frequencies,
        )
        np.testing.assert_allclose(
            np.asarray(via_combine.healpix.maps),
            np.asarray(via_materialize.healpix.maps),
            rtol=1e-5,
            atol=1e-6,
        )


class TestCombineHealpixUsesSpectrumPolarization:
    def test_per_channel_q_u_recovered(self, precision):
        """A spectrum carrying per-channel Q and U must propagate to the
        combined Q/U HEALPix maps without re-applying Faraday rotation
        (per-channel Q/U are observed values, not reference values).
        """
        nside = 8
        npix = 12 * nside * nside
        omega_pixel = 4 * np.pi / npix
        frequencies = np.array([100e6, 200e6, 300e6])
        flux_per_chan = np.array([5.0, 5.0, 5.0])
        q_per_chan = np.array([1.0, -2.0, 3.0])
        u_per_chan = np.array([-0.5, 0.7, -1.4])
        sky = _make_spectrum_sky(
            precision=precision,
            flux_per_chan=flux_per_chan,
            frequencies=frequencies,
            q_per_chan=q_per_chan,
            u_per_chan=u_per_chan,
        )
        combined = _combine_models(
            [sky],
            representation=SkyFormat.HEALPIX,
            nside=nside,
            frequencies=frequencies,
            precision=precision,
            brightness_conversion=BrightnessConversion.RAYLEIGH_JEANS,
        )
        assert combined.healpix.q_maps is not None
        assert combined.healpix.u_maps is not None
        for fi, freq in enumerate(frequencies):
            rj_factor = rayleigh_jeans_factor(float(freq), omega_pixel)
            q_jy = combined.healpix.q_maps[fi].astype(np.float64) * rj_factor
            u_jy = combined.healpix.u_maps[fi].astype(np.float64) * rj_factor
            assert float(q_jy.sum()) == pytest.approx(q_per_chan[fi], rel=1e-4)
            assert float(u_jy.sum()) == pytest.approx(u_per_chan[fi], rel=1e-4)


class TestCombinePowerLawAndSpectrumDisjointly:
    def test_power_law_layer_unchanged_when_summed_with_spectrum(self, precision):
        """When a power-law-only model is summed with a spectrum-bearing model
        (placed in distinct pixels), each layer's contribution at each freq
        should match what it would produce alone."""
        nside = 16
        npix = 12 * nside * nside
        omega_pixel = 4 * np.pi / npix
        frequencies = np.array([100e6, 200e6])
        # Spectrum source at one location, power-law source at a different
        # location far enough apart to land in distinct pixels at nside=16.
        spec_flux = np.array([6.0, 9.0])
        spec_sky = _make_spectrum_sky(
            precision=precision,
            flux_per_chan=spec_flux,
            frequencies=frequencies,
            ra_rad=0.5,
            dec_rad=0.1,
            model_name="spec",
        )
        # A plain power-law model: flux 4 Jy at ref_freq=100 MHz, alpha=-1
        # so at 200 MHz the flux is 2 Jy.
        pl_sky = create_from_arrays(
            ra_rad=np.array([2.0]),
            dec_rad=np.array([-0.4]),
            flux=np.array([4.0]),
            spectral_index=np.array([-1.0]),
            stokes_q=np.array([0.0]),
            stokes_u=np.array([0.0]),
            stokes_v=np.array([0.0]),
            ref_freq=np.array([100e6]),
            reference_frequency=100e6,
            precision=precision,
            model_name="pl",
        )
        combined = _combine_models(
            [spec_sky, pl_sky],
            representation=SkyFormat.HEALPIX,
            nside=nside,
            frequencies=frequencies,
            precision=precision,
            brightness_conversion=BrightnessConversion.RAYLEIGH_JEANS,
            mixed_model_policy="allow",
        )
        from radiosim.core.sky.containers.constants import (
            brightness_temp_to_flux_density,
        )

        expected_pl = np.array([4.0, 2.0])  # alpha=-1 power-law
        for fi, freq in enumerate(frequencies):
            jy_map = brightness_temp_to_flux_density(
                combined.healpix.maps[fi].astype(np.float64),
                float(freq),
                omega_pixel,
                method="rayleigh-jeans",
            )
            total = float(jy_map.sum())
            assert total == pytest.approx(spec_flux[fi] + expected_pl[fi], rel=1e-4)
