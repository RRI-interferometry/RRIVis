"""Tests for per-channel point-source data model and the short-circuit path.

Covers:

* ``PointSourceData`` invariants on its optional ``spectrum`` (PointSpectrum)
  field and its paired Stokes tables.
* ``spectral.evaluate_point_flux_at_freq`` nearest-channel behaviour and
  consistency with the extrapolation path when per-channel data is absent.
* ``convert.bin_sources_to_flux`` short-circuit when per-channel inputs are
  supplied.
"""

from __future__ import annotations

import numpy as np
import pytest

from radiosim.core.sky._data import PointSourceData, PointSpectrum
from radiosim.core.sky.convert import bin_sources_to_flux
from radiosim.core.sky.spectral import (
    compute_spectral_scale,
    evaluate_point_flux_at_freq,
    nearest_channel_index,
)

# --------------------------------------------------------------------------- #
# PointSourceData invariants
# --------------------------------------------------------------------------- #


def _make_point(n: int, n_chan: int | None = None) -> PointSourceData:
    rng = np.random.default_rng(0)
    flux_ref = rng.uniform(0.5, 2.0, size=n)
    kwargs: dict = {
        "ra_rad": rng.uniform(0.0, 2 * np.pi, size=n),
        "dec_rad": rng.uniform(-0.5, 0.5, size=n),
        "flux": flux_ref,
        "spectral_index": np.full(n, -0.7),
        "stokes_q": np.zeros(n),
        "stokes_u": np.zeros(n),
        "stokes_v": np.zeros(n),
        "ref_freq": np.full(n, 150e6),
    }
    if n_chan is not None:
        freqs = np.linspace(100e6, 200e6, n_chan)
        pcf = np.array(
            [(freqs[i] / freqs[0]) ** -0.7 * flux_ref for i in range(n_chan)]
        )
        kwargs["spectrum"] = PointSpectrum(flux=pcf, frequencies=freqs)
    return PointSourceData(**kwargs)


class TestPointSourceDataInvariants:
    def test_construction_without_per_channel(self) -> None:
        p = _make_point(n=4, n_chan=None)
        assert p.n_sources == 4
        assert p.spectrum is None

    def test_construction_with_per_channel(self) -> None:
        p = _make_point(n=3, n_chan=5)
        assert p.spectrum is not None
        assert p.spectrum.flux.shape == (5, 3)
        assert len(p.spectrum.frequencies) == 5

    def test_rejects_shape_mismatch(self) -> None:
        with pytest.raises(ValueError, match="does not match"):
            PointSpectrum(
                flux=np.ones((3, 2)),
                frequencies=np.array([1e8, 2e8]),  # length 2 != 3
            )

    def test_rejects_non_ascending_frequencies(self) -> None:
        with pytest.raises(ValueError, match="strictly ascending"):
            PointSpectrum(
                flux=np.ones((3, 1)),
                frequencies=np.array([2e8, 1e8, 3e8]),
            )

    def test_rejects_unpaired_polarization(self) -> None:
        pcf = np.ones((2, 3))
        freqs = np.array([1e8, 2e8])
        with pytest.raises(ValueError, match="must be set together"):
            PointSpectrum(
                flux=pcf,
                frequencies=freqs,
                stokes_q=np.ones((2, 3)),
                # stokes_u missing — must be paired with q
            )

    def test_spectrum_source_count_must_match(self) -> None:
        spectrum = PointSpectrum(
            flux=np.ones((2, 5)),  # 5 sources in spectrum
            frequencies=np.array([1e8, 2e8]),
        )
        with pytest.raises(ValueError, match="spectrum has"):
            PointSourceData(
                ra_rad=np.zeros(3),  # but only 3 here
                dec_rad=np.zeros(3),
                flux=np.zeros(3),
                spectral_index=np.zeros(3),
                stokes_q=np.zeros(3),
                stokes_u=np.zeros(3),
                stokes_v=np.zeros(3),
                ref_freq=np.full(3, 1e8),
                spectrum=spectrum,
            )

    def test_masked_preserves_per_channel_tables(self) -> None:
        p = _make_point(n=4, n_chan=3)
        mask = np.array([True, False, True, False])
        m = p.masked(mask)
        assert m.n_sources == 2
        assert m.spectrum is not None
        assert m.spectrum.flux.shape == (3, 2)
        assert p.spectrum is not None
        np.testing.assert_array_equal(m.spectrum.flux, p.spectrum.flux[:, mask])


# --------------------------------------------------------------------------- #
# evaluate_point_flux_at_freq short-circuit
# --------------------------------------------------------------------------- #


class TestEvaluateAtFreq:
    def test_nearest_channel_exact_match(self) -> None:
        freqs = np.array([100e6, 150e6, 200e6], dtype=np.float64)
        pc_flux = np.array(
            [
                [1.0, 2.0],
                [2.0, 4.0],
                [4.0, 8.0],
            ],
            dtype=np.float64,
        )
        pc_q = pc_flux * 0.1
        pc_u = pc_flux * 0.2
        pc_v = pc_flux * 0.05
        stokes_i = np.array([1.0, 2.0])
        zeros = np.zeros(2)
        alpha = np.full(2, -999.0)  # would break on the spectral path

        i, q, u, v = evaluate_point_flux_at_freq(
            stokes_i=stokes_i,
            stokes_q=zeros,
            stokes_u=zeros,
            stokes_v=zeros,
            spectral_index=alpha,
            spectral_coeffs=None,
            ref_freq=150e6,
            rotation_measure=None,
            per_channel_flux=pc_flux,
            per_channel_stokes_q=pc_q,
            per_channel_stokes_u=pc_u,
            per_channel_stokes_v=pc_v,
            channel_frequencies=freqs,
            freq=150e6,
        )
        np.testing.assert_array_equal(i, [2.0, 4.0])
        np.testing.assert_array_equal(q, [0.2, 0.4])
        np.testing.assert_array_equal(u, [0.4, 0.8])
        np.testing.assert_array_equal(v, [0.1, 0.2])

    def test_nearest_channel_off_grid(self) -> None:
        freqs = np.array([100e6, 150e6, 200e6], dtype=np.float64)
        pc_flux = np.array([[1.0], [2.0], [4.0]], dtype=np.float64)
        i, _, _, _ = evaluate_point_flux_at_freq(
            stokes_i=np.array([1.0]),
            stokes_q=np.array([0.0]),
            stokes_u=np.array([0.0]),
            stokes_v=np.array([0.0]),
            spectral_index=np.array([-0.7]),
            spectral_coeffs=None,
            ref_freq=100e6,
            rotation_measure=None,
            per_channel_flux=pc_flux,
            per_channel_stokes_q=None,
            per_channel_stokes_u=None,
            per_channel_stokes_v=None,
            channel_frequencies=freqs,
            freq=160e6,  # nearest to 150 MHz channel
        )
        np.testing.assert_array_equal(i, [2.0])

    def test_fallback_matches_spectral_scale(self) -> None:
        alpha = np.array([-0.7, 0.0, 0.5])
        stokes_i = np.array([1.0, 2.0, 3.0])
        ref_freq = 150e6
        freq = 200e6
        scale = compute_spectral_scale(alpha, None, freq, ref_freq)
        i, q, u, v = evaluate_point_flux_at_freq(
            stokes_i=stokes_i,
            stokes_q=np.zeros(3),
            stokes_u=np.zeros(3),
            stokes_v=np.zeros(3),
            spectral_index=alpha,
            spectral_coeffs=None,
            ref_freq=ref_freq,
            rotation_measure=None,
            per_channel_flux=None,
            per_channel_stokes_q=None,
            per_channel_stokes_u=None,
            per_channel_stokes_v=None,
            channel_frequencies=None,
            freq=freq,
        )
        np.testing.assert_allclose(i, stokes_i * scale)
        np.testing.assert_array_equal(q, np.zeros(3))
        np.testing.assert_array_equal(u, np.zeros(3))
        np.testing.assert_array_equal(v, np.zeros(3))

    def test_per_channel_skips_faraday_rotation(self) -> None:
        # With per-channel Q/U present, apply_faraday_rotation should NOT run;
        # the returned Q/U are the stored channel values verbatim.
        freqs = np.array([100e6, 200e6], dtype=np.float64)
        pc_q = np.array([[5.0], [6.0]])
        pc_u = np.array([[-3.0], [-2.0]])
        i, q, u, _ = evaluate_point_flux_at_freq(
            stokes_i=np.array([1.0]),
            stokes_q=np.array([999.0]),  # should be ignored
            stokes_u=np.array([999.0]),
            stokes_v=np.array([0.0]),
            spectral_index=np.array([-0.7]),
            spectral_coeffs=None,
            ref_freq=150e6,
            rotation_measure=np.array([1000.0]),  # would rotate massively
            per_channel_flux=np.array([[1.0], [2.0]]),
            per_channel_stokes_q=pc_q,
            per_channel_stokes_u=pc_u,
            per_channel_stokes_v=None,
            channel_frequencies=freqs,
            freq=100e6,
        )
        np.testing.assert_array_equal(q, [5.0])
        np.testing.assert_array_equal(u, [-3.0])


# --------------------------------------------------------------------------- #
# bin_sources_to_flux short-circuit
# --------------------------------------------------------------------------- #


class TestBinSourcesShortCircuit:
    def test_per_channel_bin_uses_channel_row(self) -> None:
        ipix = np.array([0, 1, 0], dtype=np.int64)
        pc_flux = np.array(
            [
                [1.0, 2.0, 3.0],  # 100 MHz
                [10.0, 20.0, 30.0],  # 200 MHz
            ],
            dtype=np.float64,
        )
        freqs = np.array([100e6, 200e6])
        out = bin_sources_to_flux(
            ipix,
            flux=np.array([0.0, 0.0, 0.0]),  # would produce zero via scale path
            spectral_index=np.full(3, -0.7),
            spectral_coeffs=None,
            freq=200e6,
            ref_frequency=100e6,
            npix=4,
            per_channel_flux=pc_flux,
            channel_frequencies=freqs,
        )
        # Row at 200 MHz = [10, 20, 30]; bin into pixels [0, 1, 0] => pix 0 = 40, pix 1 = 20
        np.testing.assert_array_equal(out, [40.0, 20.0, 0.0, 0.0])

    def test_nearest_channel_index(self) -> None:
        freqs = np.array([100e6, 150e6, 200e6])
        assert nearest_channel_index(freqs, 149e6) == 1
        assert nearest_channel_index(freqs, 175e6) in (1, 2)  # tie broken low
        assert nearest_channel_index(freqs, 201e6) == 2
