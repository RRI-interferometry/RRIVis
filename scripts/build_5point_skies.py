#!/usr/bin/env python3
"""Build 5-source EoR-style sky cubes mirroring /Volumes/CrucialX8/EOR_sky_222/.

Three output cubes (89 single-channel skyh5 files each, format identical to
the source set):

1. ``zenith_5_point`` — 5 sources on the HERA zenith-transit line
   (Dec = HERA latitude). Default RA offsets {-100, -50, 0, +50, +100}°
   give pairwise separations ≥ 42.7°, well above the Airy 7 m HPBW of
   ~32° at 80 MHz. Each source transits the zenith at LST = RA/15h,
   producing 5 distinct main-beam peaks in a 24h drift-scan lightcurve.

2. ``sidelobe_5_point`` — 5 sources on the Airy first-sidelobe ring
   (ZA = 61.07° from boresight at 80 MHz, D = 7 m), spaced 72° in
   azimuth around the boresight at LST = 0. Pairwise separations on
   the sky ≥ 61.9°, all simultaneously on the sidelobe at LST = 0.

3. ``sidelobe_lst_5_point`` — 5 sources at the fixed Dec where the
   AZ = 0° (due-north), ZA = 61.07° topocentric direction crosses the
   sky (Dec ≈ +30.36° at HERA latitude). RA offsets default to the same
   {-100, -50, 0, +50, +100}° as ``zenith_5_point`` so each source
   transits the Airy first-sidelobe (north-meridian) one-by-one at
   LST = RA/15h — the LST-spread sidelobe analogue of ``zenith_5_point``.

Each source has the same per-pixel intensity (default 20 Jy through
Ω_pix at the source cube's nside). Stokes Q/U/V are zeroed.
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
HERA_LAT_DEG = -30.72152
HERA_LON_DEG = 21.42831
AIRY_FIRST_SIDELOBE_X = 5.1356
AIRY_HPBW_FACTOR = 1.029  # HPBW ≈ 1.029 * λ/D for the Airy disk
DEFAULT_BRIGHT_JY = 20.0


def topo_to_icrs(
    za_deg: float, az_deg: float, lst_hours: float, lat_deg: float
) -> tuple[float, float]:
    """(ZA, AZ) topocentric at LST → (RA, Dec) ICRS, in degrees."""
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
    lam = C_LIGHT / freq_hz
    return float(
        np.rad2deg(np.arcsin(AIRY_FIRST_SIDELOBE_X * lam / (np.pi * diameter_m)))
    )


def airy_hpbw_deg(diameter_m: float, freq_hz: float) -> float:
    """Airy half-power beam width (full width) in degrees, small-angle limit."""
    lam = C_LIGHT / freq_hz
    return float(np.rad2deg(AIRY_HPBW_FACTOR * lam / diameter_m))


def angular_sep_deg(ra1: float, dec1: float, ra2: float, dec2: float) -> float:
    a1, d1, a2, d2 = (np.deg2rad(v) for v in (ra1, dec1, ra2, dec2))
    cos_d = np.sin(d1) * np.sin(d2) + np.cos(d1) * np.cos(d2) * np.cos(a1 - a2)
    return float(np.rad2deg(np.arccos(np.clip(cos_d, -1.0, 1.0))))


def list_source_files(src_dir: Path) -> list[Path]:
    files = sorted(src_dir.glob("fch*.skyh5"))
    if not files:
        raise FileNotFoundError(f"No fch*.skyh5 files in {src_dir!s}")
    return files


def compute_global_min_stokes_i(files: list[Path]) -> float:
    g_min = np.inf
    for p in files:
        with h5py.File(p, "r") as f:
            i_slice = np.asarray(f["Data/stokes"][0, 0, :])
        m = float(i_slice.min())
        if m < g_min:
            g_min = m
    return float(g_min)


def resolve_target_pixel(ra_deg: float, dec_deg: float, nside: int) -> int:
    theta = np.deg2rad(90.0 - dec_deg)
    phi = np.deg2rad(ra_deg % 360.0)
    return int(hp.ang2pix(nside, theta, phi, nest=False))


def build_one_cube(
    *,
    files: list[Path],
    out_dir: Path,
    target_pixels: list[int],
    floor_jysr: float,
    bright_jysr: float,
    note: str,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    n = len(files)
    target_arr = np.asarray(target_pixels, dtype=np.int64)
    if len(set(target_pixels)) != len(target_pixels):
        raise SystemExit(
            f"target_pixels has duplicates: {target_pixels} — pick farther-"
            "apart sources or a higher nside."
        )
    for i, src in enumerate(files):
        dst = out_dir / src.name
        shutil.copy2(src, dst)
        with h5py.File(dst, "r+") as f:
            stokes = f["Data/stokes"]
            n_stokes, n_freq, n_pix = stokes.shape
            new_stokes = np.zeros((n_stokes, n_freq, n_pix), dtype=stokes.dtype)
            i_map = np.full(n_pix, floor_jysr, dtype=stokes.dtype)
            i_map[target_arr] = bright_jysr
            new_stokes[0, 0, :] = i_map
            stokes[...] = new_stokes
            stokes.attrs["rrivis_provenance"] = note
        if (i + 1) % 10 == 0 or i + 1 == n:
            print(f"    [{i + 1:3d}/{n}] {dst.name}", flush=True)


def verify_one(out_dir: Path, target_pixels: list[int], expected_bright: float) -> None:
    files = sorted(out_dir.glob("fch*.skyh5"))
    if not files:
        raise SystemExit(f"verify: no files in {out_dir!s}")
    with h5py.File(files[0], "r") as f:
        s = f["Data/stokes"]
        i_map = np.asarray(s[0, 0, :])
        q_max = float(np.abs(np.asarray(s[1, 0, :])).max())
        u_max = float(np.abs(np.asarray(s[2, 0, :])).max())
        v_max = float(np.abs(np.asarray(s[3, 0, :])).max())
    target_arr = np.asarray(target_pixels, dtype=np.int64)
    bright_vals = i_map[target_arr]
    others = np.delete(i_map, target_arr)
    print(f"    {files[0].name}:")
    print(
        f"      I[bright pixels] = {bright_vals.tolist()}  "
        f"(expected {expected_bright:.6e})"
    )
    print(f"      I[other pixels]   range = [{others.min():.6e}, {others.max():.6e}]")
    print(f"      |Q|max={q_max}  |U|max={u_max}  |V|max={v_max}  (expect 0)")


def make_zenith_sources(
    lat_deg: float, ra_offsets_deg: list[float]
) -> list[tuple[float, float]]:
    return [(((ra % 360.0) + 360.0) % 360.0, lat_deg) for ra in ra_offsets_deg]


def make_sidelobe_sources(
    za_deg: float,
    az_offsets_deg: list[float],
    lst_hours: float,
    lat_deg: float,
) -> list[tuple[float, float]]:
    return [topo_to_icrs(za_deg, az, lst_hours, lat_deg) for az in az_offsets_deg]


def make_sidelobe_lst_sources(
    za_deg: float,
    az_deg: float,
    ra_offsets_deg: list[float],
    lat_deg: float,
) -> list[tuple[float, float]]:
    """5 sources at the Dec where (AZ, ZA) lands on the sky, with chosen RAs.

    The Dec at which the topocentric direction ``(AZ, ZA)`` points is
    independent of LST — only the RA changes as the Earth rotates. So
    fixing all 5 sources at that Dec and spreading their RAs makes each
    source transit the same topocentric (AZ, ZA) point at LST = RA/15 h.
    """
    _, dec = topo_to_icrs(za_deg, az_deg, 0.0, lat_deg)
    return [(((ra % 360.0) + 360.0) % 360.0, dec) for ra in ra_offsets_deg]


def report_pairs(
    label: str, sources: list[tuple[float, float]], hpbw_deg: float
) -> None:
    seps = []
    for i in range(len(sources)):
        for j in range(i + 1, len(sources)):
            seps.append((i + 1, j + 1, angular_sep_deg(*sources[i], *sources[j])))
    s_min = min(s[2] for s in seps)
    s_max = max(s[2] for s in seps)
    print(
        f"  pairwise sep (deg): min={s_min:.2f}, max={s_max:.2f}  "
        f"(HPBW reference = {hpbw_deg:.2f}°)"
    )
    if s_min <= hpbw_deg:
        print(
            f"  WARNING: minimum pairwise separation ({s_min:.2f}°) ≤ HPBW "
            f"({hpbw_deg:.2f}°).  Sources are not separated by >1 beam."
        )


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate 5-source EoR-style sky cubes.")
    ap.add_argument(
        "--src-dir", type=Path, default=Path("/Volumes/CrucialX8/EOR_sky_222")
    )
    ap.add_argument("--out-root", type=Path, default=Path("/Volumes/CrucialX8"))
    ap.add_argument(
        "--bright-jy",
        type=float,
        default=DEFAULT_BRIGHT_JY,
        help="Integrated flux density per source (Jy).",
    )
    ap.add_argument("--lst-hours", type=float, default=0.0)
    ap.add_argument("--lat-deg", type=float, default=HERA_LAT_DEG)
    ap.add_argument("--airy-diameter-m", type=float, default=7.0)
    ap.add_argument("--airy-freq-hz", type=float, default=80e6)
    ap.add_argument(
        "--zenith-ra-offsets",
        type=float,
        nargs="*",
        default=[-100.0, -50.0, 0.0, 50.0, 100.0],
        help="RA values (deg) of zenith-line sources (Dec = lat-deg).",
    )
    ap.add_argument(
        "--sidelobe-az-offsets",
        type=float,
        nargs="*",
        default=[0.0, 72.0, 144.0, 216.0, 288.0],
        help="Azimuth values (deg) for sources on the sidelobe ring.",
    )
    ap.add_argument(
        "--sidelobe-lst-az-deg",
        type=float,
        default=0.0,
        help=(
            "Topocentric azimuth (deg, N=0, E=90) of the sidelobe-transit "
            "point used by the sidelobe_lst_5_point cube."
        ),
    )
    ap.add_argument(
        "--sidelobe-lst-ra-offsets",
        type=float,
        nargs="*",
        default=[-100.0, -50.0, 0.0, 50.0, 100.0],
        help=(
            "RA values (deg) of the 5 sidelobe-LST sources. Each transits "
            "the AZ=sidelobe-lst-az-deg / ZA=Airy-first-sidelobe point at "
            "LST = RA/15 h."
        ),
    )
    ap.add_argument(
        "--targets",
        nargs="*",
        choices=("zenith", "sidelobe", "sidelobe_lst", "all"),
        default=["all"],
        help="Which cube(s) to build. Default: all three.",
    )
    args = ap.parse_args()
    selected_targets = (
        {"zenith", "sidelobe", "sidelobe_lst"}
        if "all" in args.targets
        else set(args.targets)
    )

    files = list_source_files(args.src_dir)
    print(f"Source: {args.src_dir} ({len(files)} files)")

    with h5py.File(files[0], "r") as f:
        nside = int(f["Header/nside"][()])
    omega_pix = 4.0 * np.pi / hp.nside2npix(nside)
    bright_jysr = args.bright_jy / omega_pix

    floor = compute_global_min_stokes_i(files)
    print(f"  floor (global min)      = {floor:.6e} Jy/sr")
    print(
        f"  per-source intensity    = {bright_jysr:.6e} Jy/sr "
        f"({args.bright_jy} Jy through Ω_pix={omega_pix:.3e} sr)"
    )

    hpbw = airy_hpbw_deg(args.airy_diameter_m, args.airy_freq_hz)
    za_sl = airy_first_sidelobe_za_deg(args.airy_diameter_m, args.airy_freq_hz)
    print(
        f"\nAiry beam: D={args.airy_diameter_m} m, ν={args.airy_freq_hz / 1e6:.1f} MHz "
        f"→ HPBW ≈ {hpbw:.2f}°, first sidelobe ZA ≈ {za_sl:.4f}°"
    )

    z_sources = make_zenith_sources(args.lat_deg, args.zenith_ra_offsets)
    z_pixels = [resolve_target_pixel(ra, dec, nside) for ra, dec in z_sources]
    if "zenith" in selected_targets:
        print("\n=== zenith_5_point ===")
        for i, ((ra, dec), pix) in enumerate(zip(z_sources, z_pixels, strict=True)):
            print(
                f"  src #{i + 1}: (RA={ra:8.4f}°, Dec={dec:8.4f}°)  pixel={pix}  "
                f"transit LST={(ra / 15.0) % 24:.2f} h"
            )
        report_pairs("zenith_5_point", z_sources, hpbw)

    s_sources = make_sidelobe_sources(
        za_sl, args.sidelobe_az_offsets, args.lst_hours, args.lat_deg
    )
    s_pixels = [resolve_target_pixel(ra, dec, nside) for ra, dec in s_sources]
    if "sidelobe" in selected_targets:
        print("\n=== sidelobe_5_point ===")
        for i, ((ra, dec), pix, az) in enumerate(
            zip(s_sources, s_pixels, args.sidelobe_az_offsets, strict=True)
        ):
            print(
                f"  src #{i + 1}: AZ={az:6.1f}°, ZA={za_sl:.4f}° → "
                f"(RA={ra:8.4f}°, Dec={dec:8.4f}°)  pixel={pix}"
            )
        report_pairs("sidelobe_5_point", s_sources, hpbw)

    sl_sources = make_sidelobe_lst_sources(
        za_sl,
        args.sidelobe_lst_az_deg,
        args.sidelobe_lst_ra_offsets,
        args.lat_deg,
    )
    sl_pixels = [resolve_target_pixel(ra, dec, nside) for ra, dec in sl_sources]
    if "sidelobe_lst" in selected_targets:
        print("\n=== sidelobe_lst_5_point ===")
        for i, ((ra, dec), pix) in enumerate(zip(sl_sources, sl_pixels, strict=True)):
            print(
                f"  src #{i + 1}: (RA={ra:8.4f}°, Dec={dec:8.4f}°)  pixel={pix}  "
                f"transits AZ={args.sidelobe_lst_az_deg}°/ZA={za_sl:.4f}° at "
                f"LST={(ra / 15.0) % 24:.2f} h"
            )
        report_pairs("sidelobe_lst_5_point", sl_sources, hpbw)

    if "zenith" in selected_targets:
        print("\nWriting zenith_5_point ...")
        z_note = (
            f"rrivis 5-source cube: floor=global_min(EOR_sky_222)={floor:.3e} Jy/sr; "
            f"5 bright pixels at Dec={args.lat_deg} deg, RA={args.zenith_ra_offsets} "
            f"each {bright_jysr:.3e} Jy/sr ({args.bright_jy} Jy)."
        )
        build_one_cube(
            files=files,
            out_dir=args.out_root / "zenith_5_point",
            target_pixels=z_pixels,
            floor_jysr=floor,
            bright_jysr=bright_jysr,
            note=z_note,
        )

    if "sidelobe" in selected_targets:
        print("\nWriting sidelobe_5_point ...")
        s_note = (
            f"rrivis 5-source cube: floor=global_min(EOR_sky_222)={floor:.3e} Jy/sr; "
            f"5 bright pixels on Airy first-sidelobe ring (ZA={za_sl:.4f}°) "
            f"at LST={args.lst_hours} h, AZ={args.sidelobe_az_offsets}; each "
            f"{bright_jysr:.3e} Jy/sr ({args.bright_jy} Jy)."
        )
        build_one_cube(
            files=files,
            out_dir=args.out_root / "sidelobe_5_point",
            target_pixels=s_pixels,
            floor_jysr=floor,
            bright_jysr=bright_jysr,
            note=s_note,
        )

    if "sidelobe_lst" in selected_targets:
        print("\nWriting sidelobe_lst_5_point ...")
        sl_dec = sl_sources[0][1] if sl_sources else float("nan")
        sl_note = (
            f"rrivis 5-source cube: floor=global_min(EOR_sky_222)={floor:.3e} Jy/sr; "
            f"5 bright pixels at Dec={sl_dec:.4f}° (Airy first-sidelobe Dec at "
            f"AZ={args.sidelobe_lst_az_deg}°, ZA={za_sl:.4f}° from HERA); "
            f"RA={args.sidelobe_lst_ra_offsets}; each {bright_jysr:.3e} Jy/sr "
            f"({args.bright_jy} Jy). Each source transits the (AZ, ZA) sidelobe "
            f"point at LST = RA/15 h."
        )
        build_one_cube(
            files=files,
            out_dir=args.out_root / "sidelobe_lst_5_point",
            target_pixels=sl_pixels,
            floor_jysr=floor,
            bright_jysr=bright_jysr,
            note=sl_note,
        )

    print("\nVerifying ...")
    if "zenith" in selected_targets:
        print("  zenith_5_point:")
        verify_one(args.out_root / "zenith_5_point", z_pixels, bright_jysr)
    if "sidelobe" in selected_targets:
        print("  sidelobe_5_point:")
        verify_one(args.out_root / "sidelobe_5_point", s_pixels, bright_jysr)
    if "sidelobe_lst" in selected_targets:
        print("  sidelobe_lst_5_point:")
        verify_one(args.out_root / "sidelobe_lst_5_point", sl_pixels, bright_jysr)

    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
