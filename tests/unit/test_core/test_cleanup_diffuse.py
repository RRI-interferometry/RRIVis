"""Characterization + behavior tests for the loaders-diffuse cleanup.

Covers Phase-2 Task 2.7 (spec A4, A5, B5/B6/B7/B8 (diffuse side), H1, H2, H3).

The network/optional-dependency loaders are exercised via monkeypatched,
in-memory fixtures so the assertions stay offline and deterministic. The
characterization assertions are invariant-style (shapes, dtypes, frequency
ordering, provenance fields) so that an incorrect decomposition violates an
invariant rather than relying on a brittle golden snapshot.
"""

from __future__ import annotations

import numpy as np
import pytest

from radiosim.core.precision import PrecisionConfig
from radiosim.core.sky.containers import SkyCoverage, SourceSubtractionStatus

# ---------------------------------------------------------------------------
# load_diffuse_sky — offline path via a fake pygdsm-like model class
# ---------------------------------------------------------------------------


class _FakeGSM:
    """Minimal pygdsm-like stand-in: ``generate(freq)`` returns a full map."""

    def __init__(self, nside: int, **_kwargs):
        self._nside = nside
        self._call_count = 0

    def generate(self, freq):
        import healpy as hp

        npix = hp.nside2npix(self._nside)
        # A frequency-dependent but deterministic full-sky map (Kelvin).
        self._call_count += 1
        base = np.linspace(1.0, 2.0, npix)
        return base * (freq / 100e6)


@pytest.fixture
def patched_diffuse(monkeypatch):
    nside = 8
    from radiosim.core.sky.loaders import diffuse as diffuse_mod

    monkeypatch.setattr(
        diffuse_mod,
        "_resolve_model_class",
        lambda _path: (lambda **kw: _FakeGSM(nside=nside, **kw)),
    )
    # Avoid the network preflight side effects.
    monkeypatch.setattr(diffuse_mod, "require_service", lambda *a, **k: None)
    return nside


def test_load_diffuse_sky_emits_ring_ordering(patched_diffuse):
    """Diffuse loaders are ring-native; output must declare ordering='ring'."""
    from radiosim.core.sky.loaders.diffuse import load_diffuse_sky

    sky = load_diffuse_sky(
        model="gsm2008",
        nside=patched_diffuse,
        frequencies=np.array([100e6]),
        precision=PrecisionConfig.standard(),
    )
    assert sky.healpix is not None
    assert sky.healpix.ordering == "ring"
    assert not sky.healpix.is_nested


def test_load_diffuse_sky_ud_grade_uses_ring_order_in_out(monkeypatch, patched_diffuse):
    """When pygdsm returns a coarser map, ud_grade must stay RING throughout."""
    import healpy as hp

    from radiosim.core.sky.loaders import diffuse as diffuse_mod
    from radiosim.core.sky.loaders.diffuse import load_diffuse_sky

    target_nside = patched_diffuse
    source_nside = max(4, target_nside // 2)
    captured: list[dict] = []
    real_ud_grade = hp.ud_grade

    def _recording_ud_grade(arr, **kwargs):
        captured.append(dict(kwargs))
        return real_ud_grade(arr, **kwargs)

    monkeypatch.setattr(hp, "ud_grade", _recording_ud_grade)
    monkeypatch.setattr(
        diffuse_mod,
        "_resolve_model_class",
        lambda _path: (lambda **kw: _FakeGSM(nside=source_nside, **kw)),
    )
    monkeypatch.setattr(diffuse_mod, "require_service", lambda *a, **k: None)

    sky = load_diffuse_sky(
        model="gsm2008",
        nside=target_nside,
        frequencies=np.array([100e6]),
        precision=PrecisionConfig.standard(),
    )
    assert sky.healpix is not None
    assert sky.healpix.ordering == "ring"
    assert captured, "expected ud_grade when source nside < target nside"
    for call in captured:
        assert call.get("order_in") == "RING"
        assert call.get("order_out") == "RING"


def test_load_diffuse_sky_full_sky_invariants(patched_diffuse):
    from radiosim.core.sky.loaders.diffuse import load_diffuse_sky

    nside = patched_diffuse
    freqs = np.array([100e6, 120e6, 150e6])
    precision = PrecisionConfig.standard()

    sky = load_diffuse_sky(
        model="gsm2008",
        nside=nside,
        frequencies=freqs,
        precision=precision,
    )

    assert sky.healpix is not None
    hpx = sky.healpix
    # Explicit nonuniform frequencies remain unchanged.
    np.testing.assert_array_equal(hpx.frequencies, np.array([100e6, 120e6, 150e6]))
    # map cube shape (n_freq, npix) for a full-sky load.
    import healpy as hp

    maps, _ = hpx.get_multifreq_maps()
    assert maps.shape == (3, hp.nside2npix(nside))
    # B5: full-sky coverage provenance.
    prov = sky.provenance
    assert prov.sky_coverage == SkyCoverage.FULL_SKY
    assert prov.coverage_fraction == 1.0
    assert prov.coverage_footprint is None
    # monopole_k is the first (now ascending => 100 MHz) full-sky mean, finite.
    assert prov.monopole_k is not None and np.isfinite(prov.monopole_k)


def test_load_diffuse_sky_haslam_source_subtraction_provenance(patched_diffuse):
    from radiosim.core.sky.loaders.diffuse import load_diffuse_sky

    sky = load_diffuse_sky(
        model="haslam",
        nside=patched_diffuse,
        frequencies=np.array([100e6, 200e6]),
        precision=PrecisionConfig.standard(),
    )
    prov = sky.provenance
    # Haslam carries source-subtraction metadata above 2 Jy at 408 MHz.
    assert prov.source_subtraction == SourceSubtractionStatus.ABOVE_THRESHOLD
    assert prov.source_subtraction_threshold_jy == pytest.approx(2.0)
    assert prov.source_subtraction_freq_hz == pytest.approx(408e6)


def test_load_diffuse_sky_requires_exactly_one_freq_source(patched_diffuse):
    from radiosim.core.sky.loaders.diffuse import load_diffuse_sky

    with pytest.raises(ValueError):
        load_diffuse_sky(
            model="gsm2008",
            nside=patched_diffuse,
            precision=PrecisionConfig.standard(),
        )


def test_load_diffuse_sky_rejects_descending_explicit_frequencies(patched_diffuse):
    """The direct loader enforces the ordered explicit-Hz contract."""
    from radiosim.core.sky.loaders.diffuse import load_diffuse_sky

    with pytest.raises(ValueError, match="strictly ascending"):
        load_diffuse_sky(
            model="gsm2008",
            nside=patched_diffuse,
            frequencies=np.array([120e6, 100e6]),
            precision=PrecisionConfig.standard(),
        )


def test_load_diffuse_sky_provenance_override(patched_diffuse):
    from radiosim.core.sky.containers import (
        MonopoleConvention,
        SkyProvenance,
    )
    from radiosim.core.sky.loaders.diffuse import load_diffuse_sky

    override = SkyProvenance(
        sky_coverage=SkyCoverage.FULL_SKY,
        coverage_fraction=1.0,
        monopole_convention=MonopoleConvention.ABSOLUTE_NO_CMB,
        notes="explicit-override",
    )
    sky = load_diffuse_sky(
        model="gsm2008",
        nside=patched_diffuse,
        frequencies=np.array([100e6]),
        precision=PrecisionConfig.standard(),
        provenance=override,
    )
    assert sky.provenance.notes == "explicit-override"


# ---------------------------------------------------------------------------
# _load_point_branch — in-memory pyradiosky-like fixture
# ---------------------------------------------------------------------------


class _FakeAngle:
    """Mimics an astropy Longitude/Latitude: ``to_value(u.rad)`` returns rad."""

    def __init__(self, rad):
        self.rad = np.asarray(rad, dtype=np.float64)

    def to_value(self, _unit):
        return self.rad


class _FakeStokes:
    """Mimics an astropy Quantity stokes cube with ``to_value``."""

    def __init__(self, arr):
        self._arr = np.asarray(arr, dtype=np.float64)

    def to_value(self, _unit):
        return self._arr


class _FakePSky:
    """Stand-in for a single-channel pyradiosky point SkyModel."""

    def __init__(self, ra, dec, stokes, names=None):
        self.Ncomponents = len(ra)
        self.ra = _FakeAngle(ra)
        self.dec = _FakeAngle(dec)
        # stokes shape: (n_stokes, 1, N)
        self._stokes = np.asarray(stokes, dtype=np.float64)
        self.name = np.asarray(names) if names is not None else None

    @property
    def stokes(self):
        return _FakeStokes(self._stokes)

    def read(self, _path):  # already populated; multifile re-reads per file
        return None


def _make_point_psky_factory():
    """Return (factory, per-file stokes registry) for a 3-channel point set."""
    ra = [0.1, 0.2, 0.3]
    dec = [0.0, 0.1, -0.1]
    names = ["a", "b", "c"]
    # 3 files; per-file Stokes I scaled so channels are distinguishable.
    files = {
        "f100.skyh5": np.array([[[1.0, 2.0, 3.0]]]),  # I-only
        "f150.skyh5": np.array([[[1.5, 3.0, 4.5]]]),
        "f200.skyh5": np.array([[[2.0, 4.0, 6.0]]]),
    }

    state = {"path": None}

    class _Factory:
        def __call__(self):
            return self

        def read(self, path):
            state["path"] = path
            self._stokes = files[path]

        @property
        def Ncomponents(self):
            return 3

        @property
        def ra(self):
            return _FakeAngle(ra)

        @property
        def dec(self):
            return _FakeAngle(dec)

        @property
        def name(self):
            return np.asarray(names)

        @property
        def stokes(self):
            return _FakeStokes(self._stokes)

    return _Factory, list(files.keys())


def test_load_point_branch_invariants(monkeypatch):
    from radiosim.core.sky.loaders import skyh5_multifile as mf

    factory, paths = _make_point_psky_factory()
    monkeypatch.setattr(mf, "_pyradiosky_cls", lambda: factory)

    precision = PrecisionConfig.standard()
    sorted_freqs = np.array([100e6, 150e6, 200e6])

    sky = mf._load_point_branch(
        sorted_paths=paths,
        sorted_freqs=sorted_freqs,
        n_stokes_avail=1,
        reference_frequency_hz=None,
        precision=precision,
        region=None,
        memmap_path=None,
        provenance=None,
    )

    pt = sky.point
    assert pt is not None
    assert pt.ra_rad.shape == (3,)
    # reference channel default = first (100 MHz): flux == [1, 2, 3].
    np.testing.assert_allclose(pt.flux, [1.0, 2.0, 3.0])
    np.testing.assert_allclose(pt.ref_freq, 100e6)
    # per-channel spectrum preserved, shape (n_freq, n_src), ascending freqs.
    spec = pt.spectrum
    assert spec is not None
    assert spec.flux.shape == (3, 3)
    np.testing.assert_array_equal(spec.frequencies, sorted_freqs)
    np.testing.assert_allclose(spec.flux[:, 0], [1.0, 1.5, 2.0])


def test_load_point_branch_reference_channel_selection(monkeypatch):
    from radiosim.core.sky.loaders import skyh5_multifile as mf

    factory, paths = _make_point_psky_factory()
    monkeypatch.setattr(mf, "_pyradiosky_cls", lambda: factory)

    sky = mf._load_point_branch(
        sorted_paths=paths,
        sorted_freqs=np.array([100e6, 150e6, 200e6]),
        n_stokes_avail=1,
        reference_frequency_hz=200e6,
        precision=PrecisionConfig.standard(),
        region=None,
        memmap_path=None,
        provenance=None,
    )
    # ref channel = 200 MHz => flux == [2, 4, 6].
    np.testing.assert_allclose(sky.point.flux, [2.0, 4.0, 6.0])
    np.testing.assert_allclose(sky.point.ref_freq, 200e6)


def test_load_point_branch_precision_dtype(monkeypatch):
    """A5/precision: stored arrays honor the configured storage dtype."""
    from radiosim.core.sky.loaders import skyh5_multifile as mf
    from radiosim.core.sky.support.precision import get_sky_storage_dtype

    factory, paths = _make_point_psky_factory()
    monkeypatch.setattr(mf, "_pyradiosky_cls", lambda: factory)

    precision = PrecisionConfig.fast()
    sky = mf._load_point_branch(
        sorted_paths=paths,
        sorted_freqs=np.array([100e6, 150e6, 200e6]),
        n_stokes_avail=1,
        reference_frequency_hz=None,
        precision=precision,
        region=None,
        memmap_path=None,
        provenance=None,
    )
    flux_dt = get_sky_storage_dtype(precision, "flux")
    assert sky.point.flux.dtype == flux_dt
    assert sky.point.spectrum.flux.dtype == flux_dt


# ---------------------------------------------------------------------------
# H1 — _pysm3 lazy import helper
# ---------------------------------------------------------------------------


def test_pysm3_lazy_helper_exists_and_raises_friendly(monkeypatch):
    from radiosim.core.sky.loaders import diffuse as diffuse_mod

    assert hasattr(diffuse_mod, "_pysm3")
    import builtins

    real_import = builtins.__import__

    def _blocked(name, *a, **k):
        if name == "pysm3" or name.startswith("pysm3."):
            raise ImportError("blocked")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", _blocked)
    with pytest.raises(ImportError, match="pysm3"):
        diffuse_mod._pysm3()


# ---------------------------------------------------------------------------
# H2 — FITS provenance inference
# ---------------------------------------------------------------------------


def _write_fits_image(path, *, with_beam):
    from astropy.io import fits

    n = 16
    data = np.ones((n, n), dtype=np.float64)
    hdu = fits.PrimaryHDU(data=data)
    h = hdu.header
    h["CTYPE1"] = "RA---SIN"
    h["CTYPE2"] = "DEC--SIN"
    h["CRVAL1"] = 0.0
    h["CRVAL2"] = 0.0
    h["CRPIX1"] = n / 2
    h["CRPIX2"] = n / 2
    h["CDELT1"] = -0.1
    h["CDELT2"] = 0.1
    h["CUNIT1"] = "deg"
    h["CUNIT2"] = "deg"
    h["RESTFRQ"] = 150e6
    h["RADESYS"] = "ICRS"
    if with_beam:
        h["BUNIT"] = "Jy/beam"
        h["BMAJ"] = 0.5
        h["BMIN"] = 0.25
    hdu.writeto(path, overwrite=True)


def test_fits_provenance_inferred(tmp_path):
    pytest.importorskip("reproject")
    from radiosim.core.sky.loaders.fits import load_fits_image

    path = str(tmp_path / "img.fits")
    _write_fits_image(path, with_beam=True)
    sky = load_fits_image(
        path,
        nside=16,
        precision=PrecisionConfig.standard(),
    )
    prov = sky.provenance
    # Inferred from reprojection: a FITS image becomes a full-sky ICRS grid.
    assert prov.sky_coverage == SkyCoverage.FULL_SKY
    # Angular resolution inferred from the beam (BMAJ/BMIN present).
    assert prov.angular_resolution_rad is not None
    theta_min, theta_max = prov.angular_resolution_rad
    assert theta_min > 0.0
    # File origin recorded in notes.
    assert prov.notes is not None and "fits" in prov.notes.lower()


def test_fits_provenance_caller_override(tmp_path):
    pytest.importorskip("reproject")
    from radiosim.core.sky.containers import MonopoleConvention, SkyProvenance
    from radiosim.core.sky.loaders.fits import load_fits_image

    path = str(tmp_path / "img2.fits")
    _write_fits_image(path, with_beam=False)
    override = SkyProvenance(
        sky_coverage=SkyCoverage.FULL_SKY,
        coverage_fraction=1.0,
        monopole_convention=MonopoleConvention.ABSOLUTE_NO_CMB,
        notes="caller-supplied",
    )
    sky = load_fits_image(
        path,
        nside=16,
        precision=PrecisionConfig.standard(),
        provenance=override,
    )
    assert sky.provenance.notes == "caller-supplied"


# ---------------------------------------------------------------------------
# H3 — explicit frequency cross-check in load_skyh5_multifile
# ---------------------------------------------------------------------------


def test_skyh5_multifile_explicit_frequency_cross_check(monkeypatch, tmp_path):
    """An explicit frequency array is cross-checked against the file grid."""
    from radiosim.core.sky.loaders import skyh5_multifile as mf

    # Build 3 fake header files on disk plus a header reader stub.
    paths = []
    file_freqs = {}
    for i, fhz in enumerate([100e6, 150e6, 200e6]):
        p = tmp_path / f"chan{i}.skyh5"
        p.write_bytes(b"stub")
        paths.append(str(p))
        file_freqs[str(p)] = fhz

    # Per-file Stokes I keyed by the on-disk channel path.
    ra = [0.1, 0.2, 0.3]
    dec = [0.0, 0.1, -0.1]
    names = ["a", "b", "c"]
    stokes_by_path = {
        paths[0]: np.array([[[1.0, 2.0, 3.0]]]),
        paths[1]: np.array([[[1.5, 3.0, 4.5]]]),
        paths[2]: np.array([[[2.0, 4.0, 6.0]]]),
    }

    class _Factory:
        def __call__(self):
            return self

        def read(self, path):
            self._stokes = stokes_by_path[path]

        @property
        def Ncomponents(self):
            return 3

        @property
        def ra(self):
            return _FakeAngle(ra)

        @property
        def dec(self):
            return _FakeAngle(dec)

        @property
        def name(self):
            return np.asarray(names)

        @property
        def stokes(self):
            return _FakeStokes(self._stokes)

    monkeypatch.setattr(mf, "_pyradiosky_cls", lambda: _Factory())

    def _fake_read_header(path):
        return {
            "filename": path,
            "component_type": "point",
            "spectral_type": "full",
            "Nfreqs": 1,
            "Ncomponents": 3,
            "freq_array": np.array([file_freqs[path]], dtype=np.float64),
            "stokes_shape": (1, 1, 3),
            "stokes_unit": "Jy",
        }

    monkeypatch.setattr(mf, "_read_header", _fake_read_header)

    # A matching explicit array should pass.
    frequencies = np.array([100e6, 150e6, 200e6])
    sky = mf.load_skyh5_multifile(
        filenames=paths,
        frequencies=frequencies,
        precision=PrecisionConfig.standard(),
    )
    assert sky.point is not None

    # A mismatching explicit array should raise.
    bad = np.array([100e6, 150e6, 999e6])
    with pytest.raises(ValueError):
        mf.load_skyh5_multifile(
            filenames=paths,
            frequencies=bad,
            precision=PrecisionConfig.standard(),
        )
