"""Tests for the io-support owner-group of the core/sky Phase-2 cleanup.

Covers spec items:

- I1: ``_schema`` version field on the serialized ``radiosim_provenance``
  attribute; the reader warns on a missing/unknown version and the field
  round-trips.
- I2: ``_sanitize_extra_column`` warns when it stringifies a mixed-type column.
- I3: lossy point-source export warns listing the dropped fields (RM,
  morphology, higher-order spectral terms) and emits ``spectral_type="full"``
  when a per-channel spectrum is available.
- J1: the eager memmap zero-fill is lazy/opt-in behind a flag/platform check
  while keeping the cross-platform zero guarantee available.
- J2: ``get_sky_storage_dtype`` stays the single, stable dtype lookup.
"""

from __future__ import annotations

import contextlib
import os
import warnings

import numpy as np
import pytest

from radiosim.core.precision import PrecisionConfig
from radiosim.core.sky.io.serialization import (
    PROVENANCE_SCHEMA_VERSION,
    _sanitize_extra_column,
    _unwrap_provenance_payload,
    _warn_dropped_point_fields,
)
from radiosim.core.sky.support.allocation import (
    _ZERO_FILL_GUARANTEED_BY_PLATFORM,
    allocate_cube,
    ensure_scratch_dir,
)
from radiosim.core.sky.support.precision import get_sky_storage_dtype


@contextlib.contextmanager
def assert_no_user_warning():
    """Fail if a ``UserWarning`` is emitted inside the block.

    ``pytest.warns(None)`` was removed in pytest 8+, so we promote
    ``UserWarning`` to an error for the duration of the block instead.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        yield


# ---------------------------------------------------------------------------
# I1 — provenance schema version field
# ---------------------------------------------------------------------------
class TestProvenanceSchemaVersion:
    def test_schema_version_is_one(self):
        assert PROVENANCE_SCHEMA_VERSION == 1

    def test_unwrap_returns_inner_provenance_dict(self):
        inner = {"sky_coverage": "full_sky", "coverage_fraction": 1.0}
        payload = {"_schema": PROVENANCE_SCHEMA_VERSION, "provenance": inner}
        out = _unwrap_provenance_payload(payload, "file.skyh5")
        assert out == inner

    def test_unwrap_warns_on_missing_schema(self):
        # A legacy / foreign payload that is itself the bare provenance dict.
        bare = {"sky_coverage": "unknown"}
        with pytest.warns(UserWarning, match="no '_schema'"):
            out = _unwrap_provenance_payload(bare, "legacy.skyh5")
        # Best-effort decode returns the bare dict as the provenance body.
        assert out == bare

    def test_unwrap_warns_on_unknown_schema(self):
        payload = {"_schema": 999, "provenance": {"sky_coverage": "unknown"}}
        with pytest.warns(UserWarning, match="unknown"):
            out = _unwrap_provenance_payload(payload, "future.skyh5")
        assert out == {"sky_coverage": "unknown"}

    def test_unwrap_missing_body_returns_empty(self):
        payload = {"_schema": PROVENANCE_SCHEMA_VERSION}
        out = _unwrap_provenance_payload(payload, "broken.skyh5")
        assert out == {}

    def test_schema_round_trip_through_skyh5(self, tmp_path):
        """The written attribute carries _schema and round-trips losslessly."""
        pytest.importorskip("pyradiosky")
        pytest.importorskip("h5py")
        import json

        import h5py

        from radiosim.core.sky import (
            MonopoleConvention,
            SkyCoverage,
            SkyProvenance,
            SourceSubtractionStatus,
            create_from_arrays,
            load_skyh5,
            save_skyh5,
        )

        precision = PrecisionConfig.standard()
        provenance = SkyProvenance(
            sky_coverage=SkyCoverage.FULL_SKY,
            coverage_fraction=1.0,
            monopole_convention=MonopoleConvention.ABSOLUTE_NO_CMB,
            monopole_k=42.0,
            source_subtraction=SourceSubtractionStatus.ABOVE_THRESHOLD,
            source_subtraction_threshold_jy=1.5,
            source_subtraction_freq_hz=150e6,
            notes="schema-version-test",
        )
        n = 4
        sky = create_from_arrays(
            ra_rad=np.linspace(0.0, 1.0, n),
            dec_rad=np.linspace(-0.3, 0.3, n),
            flux=np.linspace(1.0, 4.0, n),
            spectral_index=np.full(n, -0.7),
            reference_frequency=150e6,
            precision=precision,
            provenance=provenance,
        )

        out = tmp_path / "schema.skyh5"
        save_skyh5(sky, str(out))

        # The serialized attribute is wrapped with the schema version.
        with h5py.File(str(out), "r") as fh:
            raw = fh.attrs["radiosim_provenance"]
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        payload = json.loads(raw)
        assert payload["_schema"] == PROVENANCE_SCHEMA_VERSION
        assert "provenance" in payload

        # Round-trip does not warn (schema matches) and preserves provenance.
        with assert_no_user_warning():
            round_tripped = load_skyh5(str(out), precision=precision)
        assert round_tripped.provenance.notes == "schema-version-test"
        assert round_tripped.provenance.monopole_k == pytest.approx(42.0)


# ---------------------------------------------------------------------------
# I2 — mixed-type extra column stringification warning
# ---------------------------------------------------------------------------
class TestSanitizeExtraColumnWarning:
    def test_warns_on_mixed_type_column(self):
        col = np.array(["a", 1, 2.5, None], dtype=object)
        with pytest.warns(UserWarning, match="mixes string and numeric"):
            out = _sanitize_extra_column(col, "mixed")
        assert out.dtype.kind in ("U", "S")

    def test_warning_names_the_column(self):
        col = np.array([1, "two"], dtype=object)
        with pytest.warns(UserWarning, match="'culprit'"):
            _sanitize_extra_column(col, "culprit")

    def test_all_string_column_does_not_warn(self):
        col = np.array(["x", "y", None], dtype=object)
        with assert_no_user_warning():
            out = _sanitize_extra_column(col, "labels")
        assert out.tolist() == ["x", "y", ""]

    def test_all_numeric_column_does_not_warn(self):
        col = np.array([1, 2.0, None], dtype=object)
        with assert_no_user_warning():
            out = _sanitize_extra_column(col, "nums")
        assert out.dtype == np.float64
        assert np.isnan(out[-1])

    def test_native_dtype_column_passes_through_unchanged(self):
        col = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        with assert_no_user_warning():
            out = _sanitize_extra_column(col, "native")
        np.testing.assert_array_equal(out, col)


# ---------------------------------------------------------------------------
# I3 — lossy point export warns on dropped fields / emits multi-term spectral
# ---------------------------------------------------------------------------
class _FakePoint:
    """Minimal stand-in exposing the attributes _warn_dropped_point_fields reads."""

    def __init__(
        self,
        *,
        rotation_measure=None,
        major_arcsec=None,
        spectral_coeffs=None,
        spectrum=None,
    ):
        self.polarization = (
            None
            if rotation_measure is None
            else type("Pol", (), {"rotation_measure": rotation_measure})()
        )
        self.morphology = (
            None
            if major_arcsec is None
            else type("Morph", (), {"major_arcsec": major_arcsec})()
        )
        self.spectral_coeffs = spectral_coeffs
        self.spectrum = spectrum


class TestWarnDroppedPointFields:
    def test_warns_listing_rotation_measure(self):
        point = _FakePoint(rotation_measure=np.array([0.0, 12.0]))
        with pytest.warns(UserWarning, match="rotation measure"):
            dropped = _warn_dropped_point_fields(point)
        assert "rotation measure (RM)" in dropped

    def test_warns_listing_morphology(self):
        point = _FakePoint(major_arcsec=np.array([10.0, 0.0]))
        with pytest.warns(UserWarning, match="Gaussian morphology"):
            dropped = _warn_dropped_point_fields(point)
        assert "Gaussian morphology" in dropped

    def test_warns_listing_higher_order_spectral_terms(self):
        # (N, N_terms) with N_terms > 1 and no per-channel spectrum.
        coeffs = np.array([[1.0, -0.7, 0.1], [2.0, -0.5, 0.05]])
        point = _FakePoint(spectral_coeffs=coeffs, spectrum=None)
        with pytest.warns(UserWarning, match="log-polynomial"):
            dropped = _warn_dropped_point_fields(point)
        assert any("spectral coefficients" in d for d in dropped)

    def test_no_warning_for_clean_power_law_point(self):
        point = _FakePoint(
            rotation_measure=np.array([0.0, 0.0]),
            major_arcsec=np.array([0.0, 0.0]),
            spectral_coeffs=np.array([[1.0], [2.0]]),
        )
        with assert_no_user_warning():
            dropped = _warn_dropped_point_fields(point)
        assert dropped == []

    def test_spectrum_present_suppresses_spectral_drop_warning(self):
        # A per-channel spectrum is exported losslessly as "full", so the
        # multi-term coefficients are NOT counted as dropped.
        coeffs = np.array([[1.0, -0.7, 0.1], [2.0, -0.5, 0.05]])
        point = _FakePoint(spectral_coeffs=coeffs, spectrum=object())
        with assert_no_user_warning():
            dropped = _warn_dropped_point_fields(point)
        assert dropped == []


class TestPointExportSpectralType:
    def test_power_law_export_uses_spectral_index(self):
        pytest.importorskip("pyradiosky")
        from radiosim.core.sky import create_from_arrays
        from radiosim.core.sky.io.serialization import to_pyradiosky

        precision = PrecisionConfig.standard()
        n = 3
        sky = create_from_arrays(
            ra_rad=np.linspace(0.1, 0.3, n),
            dec_rad=np.linspace(-0.2, 0.2, n),
            flux=np.linspace(1.0, 3.0, n),
            spectral_index=np.full(n, -0.7),
            reference_frequency=150e6,
            precision=precision,
        )
        psky = to_pyradiosky(sky)
        assert psky.spectral_type == "spectral_index"

    def test_spectrum_export_uses_full_spectral_type(self):
        pytest.importorskip("pyradiosky")
        from radiosim.core.sky import (
            PointSourceData,
            PointSpectrum,
            create_from_arrays,
        )
        from radiosim.core.sky.io.serialization import to_pyradiosky

        precision = PrecisionConfig.standard()
        n = 3
        sky = create_from_arrays(
            ra_rad=np.linspace(0.1, 0.3, n),
            dec_rad=np.linspace(-0.2, 0.2, n),
            flux=np.linspace(1.0, 3.0, n),
            spectral_index=np.full(n, -0.7),
            reference_frequency=150e6,
            precision=precision,
        )
        freqs = np.array([100e6, 150e6, 200e6])
        flux_table = np.outer(np.array([1.0, 0.9, 0.8]), sky.point.flux)
        spectrum = PointSpectrum(flux=flux_table, frequencies=freqs)
        point_with_spectrum = PointSourceData(
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
        sky = sky.replace(point=point_with_spectrum)

        psky = to_pyradiosky(sky)
        assert psky.spectral_type == "full"
        # The exported channel axis matches the spectrum frequencies.
        assert psky.freq_array.size == len(freqs)


# ---------------------------------------------------------------------------
# J1 — lazy / opt-in memmap zero-fill
# ---------------------------------------------------------------------------
class TestLazyMemmapZeroFill:
    def test_ram_allocation_is_always_zero(self):
        arr = allocate_cube((3, 8), np.float32, None, "i_maps")
        assert not isinstance(arr, np.memmap)
        assert np.all(arr == 0)

    def test_default_memmap_reads_zero_where_guaranteed(self, tmp_path):
        scratch = ensure_scratch_dir(str(tmp_path))
        arr = allocate_cube((2, 16), np.float32, scratch, "i_maps")
        assert isinstance(arr, np.memmap)
        # On every platform the default allocation must read as zero: POSIX via
        # the ftruncate guarantee (no eager write), Windows via the eager fill.
        assert np.all(arr == 0)

    def test_zero_fill_true_forces_zero(self, tmp_path):
        scratch = ensure_scratch_dir(str(tmp_path))
        arr = allocate_cube((2, 16), np.float32, scratch, "i_maps", zero_fill=True)
        assert np.all(arr == 0)

    def test_opt_out_skips_full_write(self, tmp_path, monkeypatch):
        # zero_fill=False must NOT call mm[:] = 0 (no full-cube write).
        writes = {"count": 0}
        real_setitem = np.memmap.__setitem__

        def counting_setitem(self, key, value):
            writes["count"] += 1
            return real_setitem(self, key, value)

        monkeypatch.setattr(np.memmap, "__setitem__", counting_setitem)
        scratch = ensure_scratch_dir(str(tmp_path))
        allocate_cube((2, 16), np.float32, scratch, "i_maps", zero_fill=False)
        assert writes["count"] == 0

    def test_default_skips_full_write_on_posix(self, tmp_path, monkeypatch):
        if not _ZERO_FILL_GUARANTEED_BY_PLATFORM:
            pytest.skip("platform does not guarantee zero-on-grow")
        writes = {"count": 0}
        real_setitem = np.memmap.__setitem__

        def counting_setitem(self, key, value):
            writes["count"] += 1
            return real_setitem(self, key, value)

        monkeypatch.setattr(np.memmap, "__setitem__", counting_setitem)
        scratch = ensure_scratch_dir(str(tmp_path))
        allocate_cube((2, 16), np.float32, scratch, "i_maps")
        assert writes["count"] == 0

    def test_zero_fill_guarantee_flag_matches_platform(self):
        assert _ZERO_FILL_GUARANTEED_BY_PLATFORM == (os.name == "posix")


# ---------------------------------------------------------------------------
# J2 — get_sky_storage_dtype stays the single, stable dtype lookup
# ---------------------------------------------------------------------------
class TestGetSkyStorageDtype:
    def test_none_precision_returns_default(self):
        assert get_sky_storage_dtype(None, "flux") == np.dtype(np.float32)

    def test_none_precision_honors_explicit_default(self):
        assert get_sky_storage_dtype(None, "flux", np.float64) == np.dtype(np.float64)

    def test_precision_drives_dtype(self):
        precise = PrecisionConfig.precise()
        dt = get_sky_storage_dtype(precise, "flux")
        expected = np.dtype(precise.sky_model.get_dtype("flux"))
        assert dt == expected

    def test_fast_precision_gives_float32_flux(self):
        fast = PrecisionConfig.fast()
        assert get_sky_storage_dtype(fast, "flux") == np.dtype(
            fast.sky_model.get_dtype("flux")
        )

    def test_returns_numpy_dtype_instance(self):
        out = get_sky_storage_dtype(PrecisionConfig.standard(), "healpix_maps")
        assert isinstance(out, np.dtype)

    def test_module_still_importable_at_stable_path(self):
        # The public import path other owners rely on must remain stable.
        from radiosim.core.sky.support.precision import (
            get_sky_storage_dtype as imported,
        )

        assert imported is get_sky_storage_dtype
