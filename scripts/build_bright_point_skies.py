#!/usr/bin/env python3
"""Build bright-point EoR-style sky cubes mirroring /Volumes/CrucialX8/EOR_sky_222/.

Two output cubes are produced:

1. ``zenith_bright_point``        — bright pixel at HERA zenith (LST = 0 h:
                                    ICRS RA = 0°, Dec = -30.72152°).
2. ``sidelobe_airy_bright_point`` — bright pixel on the analytic Airy 7 m
                                    first-sidelobe ring at LST = 0
                                    (AZ = 0°, ZA = 61.07° → RA = 0°, Dec ≈ +30.36°).

Each output cube has 89 single-channel skyh5 files with metadata identical to
the source set (RING ordering, ICRS frame, nside = 256, Jy/sr units, IQUV
layout, full-sky dense). Stokes I is set to the global minimum of Stokes I
in EOR_sky_222 everywhere, with one pixel raised to ``--bright-jysr``.
Stokes Q, U, V are zeroed so the bright pixel is a clean unpolarized source.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import h5py
import healpy as hp
import numpy as np

C_LIGHT = 299_792_458.0

# HERA reference values used by the notebook.
HERA_LAT_DEG = -30.72152
HERA_LON_DEG = 21.42831

# Airy first-sidelobe peak: J1(x) maximum after the first null.
AIRY_FIRST_SIDELOBE_X = 5.1356

# Default integrated flux density of the single bright pixel, in Jy. At
# nside=256 (Ωpix ≈ 1.598e-5 sr) this is ≈ 1.252e6 Jy/sr ≈ 6.37 kK at 80 MHz.
DEFAULT_BRIGHT_JY = 20.0


def topo_to_icrs(
    za_deg: float, az_deg: float, lst_hours: float, lat_deg: float
) -> tuple[float, float]:
    """(ZA, AZ) topocentric at LST → (RA, Dec) ICRS, in degrees.

    AZ measured clockwise from north (N = 0°, E = 90°). Refraction and
    precession ignored — accurate to far better than one HEALPix pixel.
    """
    za = np.deg2rad(za_deg)
    az = np.deg2rad(az_deg)
    lat = np.deg2rad(lat_deg)
    sin_dec = np.cos(za) * np.sin(lat) + np.sin(za) * np.cos(lat) * np.cos(az)
    dec = np.arcsin(np.clip(sin_dec, -1.0, 1.0))
    cos_dec = np.cos(dec)
    if abs(cos_dec) < 1e-12:
        ha = 0.0
    else:
        sin_h = -np.sin(za) * np.sin(az) / cos_dec
        cos_h = (np.cos(za) - np.sin(lat) * sin_dec) / (np.cos(lat) * cos_dec)
        ha = float(np.arctan2(sin_h, cos_h))
    ra_rad = np.deg2rad(lst_hours * 15.0) - ha
    return float(np.rad2deg(ra_rad) % 360.0), float(np.rad2deg(dec))


def airy_first_sidelobe_za_deg(diameter_m: float, freq_hz: float) -> float:
    """Analytic ZA of the Airy power pattern's first sidelobe peak."""
    lam = C_LIGHT / freq_hz
    return float(
        np.rad2deg(np.arcsin(AIRY_FIRST_SIDELOBE_X * lam / (np.pi * diameter_m)))
    )


def list_source_files(src_dir: Path) -> list[Path]:
    files = sorted(src_dir.glob("fch*.skyh5"))
    if not files:
        raise FileNotFoundError(f"No fch*.skyh5 files in {src_dir!s}")
    return files


def compute_global_min_stokes_i(files: list[Path]) -> float:
    """Return min over all (channels × pixels) of Stokes I, in Jy/sr."""
    g_min = np.inf
    for p in files:
        with h5py.File(p, "r") as f:
            i_slice = np.asarray(f["Data/stokes"][0, 0, :])
        m = float(i_slice.min())
        if m < g_min:
            g_min = m
    return float(g_min)


def resolve_target_pixel(ra_deg: float, dec_deg: float, nside: int) -> int:
    """ICRS (RA, Dec) → RING-ordered HEALPix pixel index."""
    theta = np.deg2rad(90.0 - dec_deg)
    phi = np.deg2rad(ra_deg % 360.0)
    return int(hp.ang2pix(nside, theta, phi, nest=False))


def build_one_cube(
    *,
    files: list[Path],
    out_dir: Path,
    target_pixel: int,
    floor_jysr: float,
    bright_jysr: float,
    note: str,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    n = len(files)
    for i, src in enumerate(files):
        dst = out_dir / src.name
        shutil.copy2(src, dst)
        with h5py.File(dst, "r+") as f:
            stokes = f["Data/stokes"]
            n_stokes, n_freq, n_pix = stokes.shape
            new_stokes = np.zeros((n_stokes, n_freq, n_pix), dtype=stokes.dtype)
            i_map = np.full(n_pix, floor_jysr, dtype=stokes.dtype)
            i_map[target_pixel] = bright_jysr
            new_stokes[0, 0, :] = i_map
            stokes[...] = new_stokes
            stokes.attrs["rrivis_provenance"] = note
        if (i + 1) % 10 == 0 or i + 1 == n:
            print(f"    [{i + 1:3d}/{n}] {dst.name}", flush=True)


def verify_one(out_dir: Path, target_pixel: int, expected_bright: float) -> None:
    """Read back the lowest-channel file and confirm the bright pixel value."""
    files = sorted(out_dir.glob("fch*.skyh5"))
    if not files:
        raise SystemExit(f"verify: no files in {out_dir!s}")
    with h5py.File(files[0], "r") as f:
        s = f["Data/stokes"]
        i_map = np.asarray(s[0, 0, :])
        q_max = float(np.abs(np.asarray(s[1, 0, :])).max())
        u_max = float(np.abs(np.asarray(s[2, 0, :])).max())
        v_max = float(np.abs(np.asarray(s[3, 0, :])).max())
    bright = float(i_map[target_pixel])
    other_max = float(np.delete(i_map, target_pixel).max())
    other_min = float(np.delete(i_map, target_pixel).min())
    print(f"    {files[0].name}:")
    print(
        f"      I[bright_pixel] = {bright:.6e} Jy/sr  (expected {expected_bright:.6e})"
    )
    print(f"      I[other pixels]  range = [{other_min:.6e}, {other_max:.6e}]")
    print(f"      |Q|max={q_max}  |U|max={u_max}  |V|max={v_max}  (expect 0, 0, 0)")


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Generate bright-point EoR-style sky cubes."
    )
    ap.add_argument(
        "--src-dir", type=Path, default=Path("/Volumes/CrucialX8/EOR_sky_222")
    )
    ap.add_argument("--out-root", type=Path, default=Path("/Volumes/CrucialX8"))
    bright_group = ap.add_mutually_exclusive_group()
    bright_group.add_argument(
        "--bright-jy",
        type=float,
        default=None,
        help=(
            "Integrated flux density of the bright pixel (Jy). Converted to "
            "Jy/sr via the HEALPix pixel solid angle of the source cube."
        ),
    )
    bright_group.add_argument(
        "--bright-jysr",
        type=float,
        default=None,
        help="Specific intensity of the bright pixel directly (Jy/sr).",
    )
    ap.add_argument("--lst-hours", type=float, default=0.0)
    ap.add_argument("--lat-deg", type=float, default=HERA_LAT_DEG)
    ap.add_argument("--airy-diameter-m", type=float, default=7.0)
    ap.add_argument(
        "--airy-freq-hz",
        type=float,
        default=80e6,
        help="Frequency at which the Airy first-sidelobe ZA is computed.",
    )
    ap.add_argument(
        "--airy-az-deg",
        type=float,
        default=0.0,
        help="Azimuth (deg, N=0, E=90) at which to place the sidelobe source.",
    )
    args = ap.parse_args()

    files = list_source_files(args.src_dir)
    print(f"Source: {args.src_dir!s} ({len(files)} files)")

    # Validate metadata invariants from a sample file.
    with h5py.File(files[0], "r") as f:
        nside = int(f["Header/nside"][()])
        order = f["Header/hpx_order"][()]
        order = order.decode() if isinstance(order, bytes) else str(order)
        frame_grp = f["Header/hpx_frame/frame"][()]
        frame = frame_grp.decode() if isinstance(frame_grp, bytes) else str(frame_grp)
        if order.lower() != "ring":
            raise SystemExit(f"expected ring ordering, got {order!r}")
        if frame.lower() != "icrs":
            raise SystemExit(f"expected icrs frame, got {frame!r}")

    omega_pix = 4.0 * np.pi / hp.nside2npix(nside)

    if args.bright_jysr is not None:
        bright_jysr = float(args.bright_jysr)
        bright_jy = bright_jysr * omega_pix
    else:
        bright_jy = (
            float(args.bright_jy) if args.bright_jy is not None else DEFAULT_BRIGHT_JY
        )
        bright_jysr = bright_jy / omega_pix

    print("Computing global min(Stokes I) over the source cube ...", flush=True)
    floor = compute_global_min_stokes_i(files)
    print(f"  floor (global min)        = {floor:.6e} Jy/sr")
    print(f"  bright-pixel intensity    = {bright_jysr:.6e} Jy/sr")
    print(
        f"  bright-pixel integrated S = {bright_jy:.6f} Jy  "
        f"(nside={nside}, Ω_pix={omega_pix:.6e} sr)"
    )

    # Target 1: HERA zenith at LST = 0.
    ra_z, dec_z = topo_to_icrs(0.0, 0.0, args.lst_hours, args.lat_deg)
    pix_z = resolve_target_pixel(ra_z, dec_z, nside)
    print("\n=== Cube 1: zenith_bright_point ===")
    print(
        f"  zenith @ LST={args.lst_hours} h, lat={args.lat_deg}°  →  "
        f"ICRS (RA={ra_z:.4f}°, Dec={dec_z:.4f}°)"
    )
    print(f"  HEALPix pixel (nside={nside}, RING) = {pix_z}")

    # Target 2: First Airy sidelobe ring.
    za_sl = airy_first_sidelobe_za_deg(args.airy_diameter_m, args.airy_freq_hz)
    ra_s, dec_s = topo_to_icrs(za_sl, args.airy_az_deg, args.lst_hours, args.lat_deg)
    pix_s = resolve_target_pixel(ra_s, dec_s, nside)
    print("\n=== Cube 2: sidelobe_airy_bright_point ===")
    print(
        f"  Airy first sidelobe @ D={args.airy_diameter_m} m, "
        f"ν={args.airy_freq_hz / 1e6:.1f} MHz  →  ZA={za_sl:.4f}°"
    )
    print(
        f"  topocentric (ZA={za_sl:.4f}°, AZ={args.airy_az_deg}°) "
        f"@ LST={args.lst_hours} h  →  ICRS (RA={ra_s:.4f}°, Dec={dec_s:.4f}°)"
    )
    print(f"  HEALPix pixel (nside={nside}, RING) = {pix_s}")

    print("\nWriting cube 1 ...")
    build_one_cube(
        files=files,
        out_dir=args.out_root / "zenith_bright_point",
        target_pixel=pix_z,
        floor_jysr=floor,
        bright_jysr=bright_jysr,
        note=(
            f"rrivis bright-point cube: floor=global_min(EOR_sky_222)={floor:.3e} Jy/sr, "
            f"bright pixel={bright_jysr:.3e} Jy/sr at HERA zenith "
            f"(RA={ra_z:.4f}, Dec={dec_z:.4f}), "
            f"LST={args.lst_hours} h, lat={args.lat_deg} deg."
        ),
    )

    print("\nWriting cube 2 ...")
    build_one_cube(
        files=files,
        out_dir=args.out_root / "sidelobe_airy_bright_point",
        target_pixel=pix_s,
        floor_jysr=floor,
        bright_jysr=bright_jysr,
        note=(
            f"rrivis bright-point cube: floor=global_min(EOR_sky_222)={floor:.3e} Jy/sr, "
            f"bright pixel={bright_jysr:.3e} Jy/sr at Airy 7m first sidelobe "
            f"(AZ={args.airy_az_deg}, ZA={za_sl:.4f})  →  "
            f"ICRS (RA={ra_s:.4f}, Dec={dec_s:.4f}), "
            f"LST={args.lst_hours} h, lat={args.lat_deg} deg."
        ),
    )

    print("\nVerifying ...")
    print("  zenith_bright_point:")
    verify_one(
        args.out_root / "zenith_bright_point",
        target_pixel=pix_z,
        expected_bright=bright_jysr,
    )
    print("  sidelobe_airy_bright_point:")
    verify_one(
        args.out_root / "sidelobe_airy_bright_point",
        target_pixel=pix_s,
        expected_bright=bright_jysr,
    )

    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
