"""Generate packaged HEALPix footprint assets for catalog provenance."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import healpy as hp
import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "src" / "radiosim" / "core" / "sky" / "data" / "footprints"
NSIDE = 256
COORDINATE_FRAME = "icrs"
GENERATOR_VERSION = "catalog_footprints_v1"


def _pixel_centers() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    npix = hp.nside2npix(NSIDE)
    ipix = np.arange(npix, dtype=np.int64)
    theta, phi = hp.pix2ang(NSIDE, ipix)
    ra_deg = np.degrees(phi)
    dec_deg = np.degrees(np.pi / 2.0 - theta)
    return ipix, ra_deg, dec_deg


def _ra_between(ra_deg: np.ndarray, lo_deg: float, hi_deg: float) -> np.ndarray:
    if lo_deg <= hi_deg:
        return (ra_deg >= lo_deg) & (ra_deg <= hi_deg)
    return (ra_deg >= lo_deg) | (ra_deg <= hi_deg)


def _mask_declination(
    dec_deg: np.ndarray,
    *,
    dec_min_deg: float | None = None,
    dec_max_deg: float | None = None,
) -> np.ndarray:
    mask = np.ones(dec_deg.shape, dtype=bool)
    if dec_min_deg is not None:
        mask &= dec_deg >= dec_min_deg
    if dec_max_deg is not None:
        mask &= dec_deg <= dec_max_deg
    return mask


def _sumss_mask(ra_deg: np.ndarray, dec_deg: np.ndarray) -> np.ndarray:
    coords = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg, frame="icrs")
    gal_b_deg = coords.galactic.b.deg
    return (dec_deg <= -30.0) & (np.abs(gal_b_deg) > 10.0)


def _lotss_dr1_mask(ra_deg: np.ndarray, dec_deg: np.ndarray) -> np.ndarray:
    return _ra_between(ra_deg, 161.0, 231.0) & (dec_deg >= 45.5) & (dec_deg <= 57.0)


def _survey_payloads() -> dict[str, dict[str, object]]:
    ipix, ra_deg, dec_deg = _pixel_centers()
    return {
        "nvss.npz": {
            "release": "NVSS",
            "reference_url": "https://www.cv.nrao.edu/nvss/",
            "hpx_inds": ipix[_mask_declination(dec_deg, dec_min_deg=-40.0)],
        },
        "vlass.npz": {
            "release": "VLASS Quick Look Epoch 1",
            "reference_url": "https://science.nrao.edu/science/surveys/vlass",
            "hpx_inds": ipix[_mask_declination(dec_deg, dec_min_deg=-40.0)],
        },
        "tgss.npz": {
            "release": "TGSS ADR1",
            "reference_url": "https://tgssadr.strw.leidenuniv.nl/",
            "hpx_inds": ipix[_mask_declination(dec_deg, dec_min_deg=-53.0)],
        },
        "wenss.npz": {
            "release": "WENSS",
            "reference_url": "https://www.sao.ru/cats/doc/WENSS.html",
            "hpx_inds": ipix[_mask_declination(dec_deg, dec_min_deg=30.0)],
        },
        "sumss.npz": {
            "release": "SUMSS",
            "reference_url": "https://ui.adsabs.harvard.edu/abs/2003MNRAS.342.1117M/abstract",
            "hpx_inds": ipix[_sumss_mask(ra_deg, dec_deg)],
        },
        "lotss_dr1.npz": {
            "release": "LoTSS DR1 HETDEX Spring Field",
            "reference_url": "https://www.aanda.org/articles/aa/full_html/2019/02/aa33973-18/aa33973-18.html",
            "hpx_inds": ipix[_lotss_dr1_mask(ra_deg, dec_deg)],
        },
        "racs_low.npz": {
            "release": "RACS-low1 DR1 curated catalogue",
            "reference_url": "https://research.csiro.au/racs/home/data-2/racs-low-dr1-data/",
            "hpx_inds": ipix[
                _mask_declination(dec_deg, dec_min_deg=-80.0, dec_max_deg=30.0)
            ],
        },
        "racs_mid.npz": {
            "release": "RACS-mid1 DR2 curated catalogue",
            "reference_url": "https://research.csiro.au/racs/home/data-2/racs-mid1-dr2/",
            "hpx_inds": ipix[_mask_declination(dec_deg, dec_max_deg=49.0)],
        },
        "racs_high.npz": {
            "release": "RACS-high DR1 curated catalogue",
            "reference_url": "https://www.atnf.csiro.au/daily-picture/2025/01/31/a-radio-galaxy-from-racs-high/",
            "hpx_inds": ipix[_mask_declination(dec_deg, dec_max_deg=48.0)],
        },
    }


def _expected_payload(raw: dict[str, object]) -> dict[str, object]:
    return {
        "nside": np.array(NSIDE, dtype=np.int64),
        "coordinate_frame": np.array(COORDINATE_FRAME),
        "hpx_inds": np.asarray(raw["hpx_inds"], dtype=np.int64),
        "release": np.array(str(raw["release"])),
        "reference_url": np.array(str(raw["reference_url"])),
        "generator_version": np.array(GENERATOR_VERSION),
    }


def _check_asset(path: Path, expected: dict[str, object]) -> bool:
    if not path.exists():
        print(f"missing: {path}", file=sys.stderr)
        return False
    with np.load(path, allow_pickle=False) as current:
        for key, expected_value in expected.items():
            actual_value = current[key]
            if isinstance(expected_value, np.ndarray):
                if not np.array_equal(actual_value, expected_value):
                    print(f"mismatch: {path.name}:{key}", file=sys.stderr)
                    return False
            elif actual_value != expected_value:
                print(f"mismatch: {path.name}:{key}", file=sys.stderr)
                return False
    return True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--check",
        action="store_true",
        help="Validate that committed assets match the generator output.",
    )
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ok = True
    for filename, raw_payload in sorted(_survey_payloads().items()):
        expected = _expected_payload(raw_payload)
        path = OUT_DIR / filename
        if args.check:
            ok &= _check_asset(path, expected)
            continue
        np.savez_compressed(path, **expected)
        print(path.relative_to(ROOT))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
