# VisSim — Exhaustive Reference

> Source: `/Users/kartikmandar/RRIVis/simulators/VisSim/` (git submodule, upstream
> `https://github.com/samirchoudhuri/VisSim.git`).
> All claims below are derived directly from the files in that directory.

---

## 1. Overview

**VisSim** is a small, self-contained C package that simulates radio
interferometric visibilities and writes them in **AIPS-compatible UVFITS
format**. It is written from a low-frequency / EoR (Epoch of Reionisation)
perspective and is templated on the **GMRT** (Giant Metrewave Radio Telescope).

The repository advertises itself in `simulators/VisSim/README.md` (the entire
file) as:

> VisSim — The visibility simulator for radio interferometric observations.
> It generates the visibilities for the point sources and diffuse foreground
> components. The instrumental noise can also be added in the data.

The codebase splits into two cooperating tools, each in its own subdirectory:

| Subdir | Binary(s) | Role |
|--------|-----------|------|
| `simulators/VisSim/point_src/` | `visfits` | Generate UVFITS file: point sources + (optional) diffuse FITS image cube as sky, with antenna gain/phase noise, system noise, and gridded random‑UV options. Supports the **K** (geometric phase) + simple analytic primary beam only. |
| `simulators/VisSim/diffuse/` | `grf`, `grf_unity`, `grf_rea`, `grf_unity_rea`, `visfitsgrid`, `gen_bubble` | Generate diffuse foreground FITS image cubes (Gaussian random fields with a prescribed angular power spectrum + frequency scaling), and grid such image-cube visibilities into an existing UVFITS template. |

Only `grf`, `visfitsgrid` (and `gen_bubble`, partially) have C sources actually
present in the tree; `grf_unity`, `grf_rea`, `grf_unity_rea`, `gen_bubble` are
referenced by `simulators/VisSim/diffuse/makefile` but their `.c` files are
**absent** in the checked-in tree (see §11 — Limitations).

### 1.1 Authorship & licence

From `simulators/VisSim/point_src/COPYLEFT.INFO`:

| Field | Value |
|-------|-------|
| Code name | `visfits` |
| Developed by | **Prasun Dutta** — Indian Institute of Science Education and Research (IISER), Bhopal, ITI Gas Rahat Building, Govindpura, Bhopal, MP, PIN 462023, India |
| Email | `prasun@iiserb.ac.in` |
| Purpose | "Generate uv fits data format for different input data" |
| Licence | Free software, distributed under **GPL** (verbatim from the file: "Distributed under GPL"); credit-with-distribution clause |
| Acknowledgements | Ue-Li Pen, Gregory Paciga (CITA); Jayaram N. Chengalure, Jayanta Roy (NCRA); Wasim Raja (RRI); Nissim Kanekar (NCRA); Somnath Bharadwaj, Samir Chowdhury (IIT Kgp) |

The current GitHub remote is `samirchoudhuri/VisSim`; the most recent commit
author is **Samir Choudhuri** (`samir.svc@gmail.com`), who appears to be the
present maintainer. There is no `LICENSE` file in the tree itself — only the
GPL claim inside `COPYLEFT.INFO`.

### 1.2 Version

Versioning is handled at compile-time:

```c
// simulators/VisSim/point_src/version.h
# define HVERSION 0   // Higher version number
# define LVERSION 1   // Lower version number
```

`simulators/VisSim/point_src/visfits.h` also declares `# define VERSION "0.01"`.
`get_version()` in `funcs.c` formats this together with the IATUTC offset (34 s)
into a string `"00 34.01"` that is printed at run-time. There are no git tags.

### 1.3 Languages

| Language | Lines (approx.) | Files |
|----------|----------------:|-------|
| C | ~2.5 k (current) + ~1.2 k (legacy `funcs.c.nrcp`) | 13 |
| Make | ~30 | 2 (`Makefile`, `makefile`) |
| Bash | ~20 | 2 (`run.sh`) |
| Plain-text data / config | a few | `INP.VISFITS`, `*.vf`, `input.grf`, `GMRT_ANT.DAT`, `pttest.dat`, `COPYLEFT.INFO` |

There is **no** Python, no `pyproject.toml`, no `setup.py`, no
`requirements.txt`, no `environment.yml`, no `Dockerfile`, no CI config, and
no test harness.

### 1.4 Git history (`git -C simulators/VisSim log --oneline`)

```
69011e9 check image dimension 3              (Sat Dec 13 22:35:56 2025) — Samir Choudhuri
723bd64 CDELT negative value corrected
9e3bc37 Write_SU_TABLE updated for CASA new version
d83597c point source catalogue added
6855e24 Update README.md
e9afddf point_src and diffuse added
d429706 Initial commit
```

Seven commits, no tags. The "check image dimension 3" patch (HEAD) added a
3-axis sanity check inside `SREAD_HDR()` of `diffuse/fitsprog.c`.

---

## 2. Repository layout

```
simulators/VisSim/
├── .git                         # gitdir pointer — submodule of RRIVis
├── README.md                    # 215 bytes — one paragraph, see §1
│
├── point_src/                   # → builds the `visfits` binary
│   ├── Makefile                 # gcc rule, hard-codes /home/samir/astro/{lib,include}
│   ├── visfits.h                # public symbols + compile-time switches (POINT/DIFF/NOISE/GERR)
│   ├── eor2fits.h               # legacy header (almost identical to visfits.h)
│   ├── version.h                # HVERSION / LVERSION macros
│   ├── version.c                # standalone "version" tool
│   ├── visfits.c                # main(): orchestrates the simulation
│   ├── funcs.c                  # ~1275 lines: header writing, scan/source/ant readers, RIME inner loop
│   ├── funcs.c.nrcp             # older variant of funcs.c (NR-Cambridge style RNG)
│   ├── beam.c                   # Bessel-J1 Airy primary beam
│   ├── utils.c                  # CHKFILE, JD↔GST, JD→IAU date, HMS→deg, COPYLEFT printer
│   ├── COPYLEFT.c               # tiny tool that prints lines beginning with '$' from COPYLEFT.INFO
│   ├── COPYLEFT                 # Mach-O x86_64 binary built from COPYLEFT.c
│   ├── COPYLEFT.INFO            # licence + author block (printed by `Legals()`)
│   ├── INP.VISFITS              # main input config (telescope/freq/observation/noise)
│   ├── scan.vf                  # AIPS scan table input (one scan per row)
│   ├── source.vf                # AIPS source table input (one source per row)
│   ├── GMRT_ANT.DAT             # 30-row GMRT antenna XYZ baselines (metres)
│   ├── pttest.dat               # 1-source test point-source catalogue (NCHAN-channel SED)
│   └── run.sh                   # `make && time ./visfits ...`
│
└── diffuse/                     # → builds the `grf`/`visfitsgrid` binaries
    ├── makefile                 # gcc rules; references several missing .c files
    ├── fitsprog.h               # naxis = 3, declarations
    ├── fitsprog.c               # FITS image header read/write helpers (BWRITE/BREAD/SWRITE/SREAD)
    ├── grf.c                    # main: 2-D Gaussian random field generator with prescribed P(U)
    ├── beam.c                   # identical PB code to point_src/beam.c
    ├── visfitsgrid.c            # main: FFT image cube → grid into UVFITS template
    ├── read_fits_func.c         # UVFITS reader used by visfitsgrid
    ├── read_fits_func.h         #   declarations
    ├── funcs_comov.c            # cosmology utilities (E(z), conformal distance, dr/dν)
    ├── funcs_comov.h            #   declarations + (mismatched) `cera` proto
    ├── simp.c                   # composite Simpson 1/3 quadrature
    ├── input.grf                # parameter file consumed by `grf`
    └── run.sh                   # commented-out driver loop
```

Total disk usage: **224 KB** (`du -sh` of the tree).

---

## 3. Build & runtime architecture

### 3.1 Compilation model

Everything is plain `gcc`. Two tiny GNU make files:

```make
# simulators/VisSim/point_src/Makefile
LINKLIB = -L/home/samir/astro/lib -fopenmp -lfftw3 -lcfitsio -lnrcp -lm -lgsl -lgslcblas
INCLUDE = -I/home/samir/astro/include/

turget: clean visfits

clean:
	rm -rf visfits *~

visfits: beam.c visfits.c funcs.c utils.c
	gcc -g -o visfits $(INCLUDE) visfits.c beam.c funcs.c utils.c ${LINKLIB}
```

```make
# simulators/VisSim/diffuse/makefile
LINKLIB = -L/home/samir/astro/lib -fopenmp -lfftw3 -lcfitsio -lm -lnrcp -lgsl -lgslcblas
INCLUDE = -I/home/samir/astro/include/

target: grf grf_unity grf_rea grf_unity_rea visfitsgrid gen_bubble

grf:           grf.c fitsprog.c beam.c
	gcc -g -o grf $(INCLUDE) fitsprog.c beam.c grf.c $(LINKLIB)
grf_unity:     grf_unity.c fitsprog.c beam.c                    # source missing
grf_rea:       grf_rea.c fitsprog.c beam.c                      # source missing
grf_unity_rea: grf_unity_rea.c fitsprog.c beam.c                # source missing
visfitsgrid:   fitsprog.c visfitsgrid.c read_fits_func.c
gen_bubble:    gen_bubble.c funcs_comov.c                       # source missing
```

Hard-coded paths: both makefiles point at `/home/samir/astro/{lib,include}` —
this is the original developer's workstation. Anyone else must edit
`LINKLIB` / `INCLUDE` (or rely on system-default `-l...`).

### 3.2 External library dependencies

Pulled in by both makefiles:

| Library | Used for | Where |
|---------|----------|-------|
| `cfitsio` | All FITS I/O (UVFITS write, image read/write) | every `*.c` that touches FITS |
| `fftw3` | 2-D real ↔ complex DFT for diffuse field & gridding | `grf.c`, `visfitsgrid.c` |
| `gsl` (`-lgsl -lgslcblas`) | RNG (`gsl_rng_cmrg`), Gaussian draws (`gsl_ran_gaussian`), Bessel J₁ (`gsl_sf_bessel_J1`) | `funcs.c`, `grf.c`, `beam.c` |
| `nrcp` (Numerical Recipes for C, third-party) | `julday()`, `nrutil.h`, `SQR()` macro | `funcs.c`, `read_fits_func.c` |
| `m` (libm) | trig, sqrt, pow | everywhere |
| `OpenMP` (`-fopenmp`) | thread-parallel diffuse-image inner loop | `funcs.c::Gen_Vis()` |

The `nrcp` link dependency is the one most likely to be absent on a generic
machine — Numerical Recipes is non-redistributable and must be provided by the
user. The only NR symbol actually used in the tree is `julday(month, day, year)`
inside `Init_PARM()` and `SQR(x)` inside `read_fits_func.c`.

### 3.3 Runtime model

Both binaries are command-line, one-shot, single-process executables:

* `visfits` — single process; spawns `NT` OpenMP threads only inside
  `Gen_Vis()` for the diffuse summation when `# define DIFF` is active.
* `grf` — single-threaded; uses FFTW serial plans (`FFTW_ESTIMATE`).
* `visfitsgrid` — single-threaded.

There is no MPI, no GPU, no on-the-fly checkpointing, no logging library —
progress is emitted via `printf`.

---

## 4. The `visfits` driver — entry point

`simulators/VisSim/point_src/visfits.c` (75 lines) is the entire `main()`. Its
calling convention:

```
Usage: visfits <input file> <scan file> <source file> <input image fits file>
                <point source file> <output fits file> <No of Threads>
```

That is **7 positional arguments** — `argv[1..7]` — checked by `argc!=8`.

| `argv[i]` | Variable | Meaning |
|-----------|----------|---------|
| 1 | input file | the `INP.VISFITS`-style key/value file (read by `Read_Inputs`) |
| 2 | scan file | AIPS-style scan table (`scan.vf`) |
| 3 | source file | AIPS-style source table (`source.vf`) |
| 4 | input image fits file | diffuse-sky FITS image cube (only used if `#define DIFF` is on) |
| 5 | point source file | text catalogue (`pttest.dat` style) — direction-cosines + per-channel flux |
| 6 | output fits file | UVFITS path to be created (must NOT exist) |
| 7 | NT | number of OpenMP threads for the diffuse loop |

Driver flow (literal sequence in `visfits.c`):

1. `Legals()` — print COPYLEFT block and version banner.
2. `CHKFILE(...)` for each of `argv[1..5]`.
3. `Read_Inputs(argv[1])` — parse `INP.VISFITS`.
4. `Init_PARM()` — count scans, sources; allocate antenna XYZ; compute
   reference hour-angle window.
5. `Init_HDR(argv[6])` — create the empty UVFITS file & write the random-group
   header.
6. `Read_Image_FITS(argv[4])` — read the diffuse image cube (always called,
   though only used when `DIFF` is compiled in).
7. `Read_Point_Sources(argv[5])` — read the ASCII point catalogue.
8. `Write_FITS_data(argv[6])` — main loop: write all visibility groups.
9. `Write_AG_TABLE` / `Write_FQ_TABLE` / `Write_SU_TABLE` — append AIPS
   antenna, frequency-table and source-table HDUs.
10. Print elapsed wall time.

There is no error-recovery path: any FITS error is fatal via `printerror()`
which calls `exit(status)`.

### 4.1 Compile-time switches (`visfits.h`)

```c
//# define POINT   // include point sources in Gen_Vis
//# define DIFF    // include diffuse cube in Gen_Vis (OpenMP loop)
# define NOISE    // add Gaussian noise σ = SIG_N to Re/Im
//# define GERR    // multiply by per-antenna complex gain (1+δ)e^{iφ}
```

Default state of the file as committed: **only `NOISE` is enabled**. To
actually generate sky visibilities the user must uncomment `POINT` and/or
`DIFF` and rebuild. `Gen_Vis()` returns `Vre = Vim = 0` if both are commented
out, so the default build produces noise-only data.

### 4.2 Other compile-time constants

From `visfits.h`:

| Macro | Value | Meaning |
|-------|-------|---------|
| `VERSION` | `"0.01"` | string version |
| `IATUTC` | `34.0` | seconds, fixed leap-second offset (frozen for 2007 epoch) |
| `REF_LON` | `82.5` | reference longitude (deg) — GMRT |
| `C` | `299792458.0` | speed of light (m/s) |
| `DR` | `M_PI/180` | deg→rad |
| `EPOACH` | `2000.0` | epoch of RA/Dec |
| `MAX_VAL` | `1.0e7` | unused except as a guard |
| `NANTE` | 30 | number of antennas (compile-time, GMRT) |
| `NSTOKES` | 2 | output Stokes / pol products written (XX, YY) |
| `NCMPLX` | 3 | each pixel is `{re, im, weight}` |
| `PCOUNT` | 8 | UV random-group parameters: U, V, W, BASELINE, DATE, DATE, SOURCE, FREQSEL |
| `NAXIS` | 7 | total UVFITS NAXIS — 0 (groups) + 6 image-axes |
| `NSIDE` | 1 | one IF |
| `enum {RE=0, IM=1, WT=2}` | | indexing inside the visibility triplet |

Changing the array (e.g. for non-GMRT) requires recompilation, *not* a config
change.

---

## 5. Inputs

### 5.1 `INP.VISFITS` — the master parameter file

Parsed sequentially by `Read_Inputs()` in `funcs.c` (lines 116-216). The
parser is **strictly positional** — it reads three header lines, one value
line, three header lines, one value line, etc., using `fgets` + `sscanf`. Any
deviation from this layout will silently mis-bind values.

The file shipped in `simulators/VisSim/point_src/INP.VISFITS`:

```
# ==================== TELESCOPE PARAMEERS ===========================
# VALUE ----------------  PARAM  --TYP-- COMMENT ---------------------
#
  GMRT             : TELESC      (S)   TELESCOPE name
  GMRT_ANT.DAT     : ANTFILE     (S)   TELE antenna parameter file
  19 06 00         : TELE_LAT_   (F)   Tel latitude     (DD MnMn SS)
  74 03 00         : TELE_LON_   (F)   Tel longitude    (DD MnMn SS)
# ____________________________________________________________________
# ==================== FREQENCY PARAMEERS ============================
# VALUE ----------------  PARAM  --TYP-- COMMENT ---------------------
#
  150.0            : FREQOBS     (F)   1st freq channel (UNIT : MHz)
  16.              : BWOBS       (F)   Obs. bandwidth   (UNIT : MHz)
  150.0            : RESTFREQ    (D)   Line rest freq   (UNIT : MHz)
# ____________________________________________________________________
# ==================== OBSERVATION PARAMEERS =========================
# VALUE ----------------  PARAM  --TYP-- COMMENT ---------------------
#
  PRASUN           : NAMEOBS     (S)   Observer Name
  128              : NCHAN       (I)   No. of channels
  16.              : INTIME      (F)   Integration time (UNIT : Sec)
  12 02 2007       : DOS         (I)   Date of observ.  (DD MM YYYY)
  21 00 00         : STTIM       (F)   Start OBS (GMT)  (HH MnMn SS)
  0.0 -950         : SIG_A seedA (F L) Antenna based gain error amp
  0.0 -1050        : SIG_P seedP (F L) do Phase
  1.03 -850        : SIG_N seedN (F L) Noise standard dev
  0.0              : SIG_Acont
  0.0              : SIG_Pcont
  8                : NTbin
  0.0 -955         : sigpos seedpos
  3000. -855 0.    : Umax in lambda seedran flagran(1=randomUV)
```

Per-field semantics inferred from the code:

| Variable | Type | Source line | Meaning |
|----------|------|-------------|---------|
| `TELESC` | `char[128]` | line 4 | Telescope name string written to `TELESCOP` keyword. |
| `ANTFILE` | `char[128]` | line 5 | Path to antenna XYZ file (`GMRT_ANT.DAT`). 30 rows expected (`NANTE`). |
| `TELE_LAT_DD/Mm/SS` | float×3 | line 6 | Latitude HMS-style triple — **note**: it is read with `HMS2D` and then divided by 15, i.e. interpreted as deg–min–sec. |
| `TELE_LON_DD/Mm/SS` | float×3 | line 7 | Longitude triple. Same `HMS2D/15` convention. |
| `FREQOBS` | float | | Reference frequency (MHz). Used in CRVAL3 of the data and in PSCAL for U/V/W. |
| `BWOBS` | float | | Total bandwidth (MHz). |
| `RESTFREQ` | double | | Rest-frame line freq (MHz). Recorded but commented out in `Init_HDR()`. |
| `NAMEOBS` | `char[8]` | | Observer name. |
| `NCHAN` | int | | Number of frequency channels. Channel width is `CWOBS = BWOBS/NCHAN`. |
| `INTIME` | float | | Integration time per record (s). |
| `OBS_DD/MM/YY` | int×3 | | Calendar date of observation. |
| `ST_HH/MM/SS` | float×3 | | UTC start time. |
| `SIG_A`, `seedA` | double, ulong | | Per-antenna gain-amplitude std-dev (random per record) and RNG seed. |
| `SIG_P`, `seedP` | double, ulong | | Same for gain phase (radians). |
| `SIG_N`, `seedN` | double, ulong | | System-noise std-dev (added to Re and Im independently if `NOISE`). |
| `SIG_Acont` | double | | Std-dev of the **constant-per-time-bin** gain-amplitude offset. |
| `SIG_Pcont` | double | | Same, phase. |
| `NTbin` | int | | Number of constant-gain time bins per scan. |
| `SIG_Pos`, `seedPos` | double, ulong | | Std-dev of the per-record *position* shift in (x,y) (zeroed in current code). |
| `Umax`, `seedran`, `flagran` | double, ulong, double | | If `flagran==1.` the (u,v,w) is *replaced* by a uniform random draw in `[-Umax, Umax]` (in λ); `w` is forced to 0. Used to generate scrambled-baseline test data. |

Five RNG streams are allocated, all of type `gsl_rng_cmrg`:
`rA` (gain amp), `rP` (gain phase), `rN` (noise), `rPos` (position), `rran`
(random UV).

### 5.2 `scan.vf` — scan list

```
# ==================== SCAN PARAMEERS ===============================
# SC_NO    SUID    FQID    BRECORD     ERECORD
  1        1       1       1           1800
```

Two fixed comment header lines, then one row per scan. Columns:

| Col | Meaning |
|-----|---------|
| `SC_NO` | Scan number (printed at end of each scan). |
| `SUID` | Source ID — index into `source.vf` (1-based). |
| `FQID` | Frequency-table ID — written verbatim into the FREQSEL random parameter. |
| `BRECORD` | First record index of the scan (1-based, inclusive). |
| `ERECORD` | Last record (inclusive). |

`NSCAN` is determined by counting non-blank rows (`feof` loop in
`Init_PARM`); each scan contributes `(ERECORD - BRECORD + 1) × NBASELINE`
random groups, with `NBASELINE = NANTE*(NANTE-1)/2 = 435` for GMRT.

### 5.3 `source.vf` — source list

```
# ================== SOURCE PARAMEERS =======(EPOCH:2000)=====
# SUNAM      SUID     CALCOD    SOURA          SOUDEC
  0823+26    1        C         10 46 00.0     +59 00 59.9
```

| Col | Meaning |
|-----|---------|
| `SUNAM` | Source name (≤16 chars; copied to AIPS SU table). |
| `SUID` | Source ID. |
| `CALCOD` | 4-char calibrator code; if it doesn't begin with `C` the writer overwrites it with `" "`. |
| `SOURA` | RA in **hours min sec** triple. Decoded with `HMS2D` (no `/15`). |
| `SOUDEC` | Dec in **deg arcmin arcsec** triple. Decoded with `HMS2D/15`. |

In the AIPS SU table written by `Write_SU_TABLE()`, `RAAPP` and `DECAPP` are
hard-aliased to `RAEPO` and `DECEPO` (no precession applied).

### 5.4 `GMRT_ANT.DAT` — antenna positions

30 rows, format `name X Y Z` (whitespace-separated, metres). The shipped file
is the canonical GMRT layout:

```
C00  6.950000E+00   6.878800E+02   -2.004000E+01
C01  1.324000E+01   3.264300E+02   -4.035000E+01
C02  0.000000E+00   0.000000E+00    0.000000E+00     # array reference
C03 -5.110000E+01  -3.727200E+02    1.335900E+02
...
W06 -3.102110E+03  -1.124560E+04    8.916260E+03
```

Read in `Init_PARM()` into `bx[ii], by[ii], bz[ii]` (doubles, metres). The
station name is later truncated to 6 chars and appended with `:NN` for the
AIPS AN table:

```c
sprintf(buffer2, ":%02d", row);
sprintf(anname, "%s%s", buffer1, buffer2);
anname[6] = ' '; anname[7] = ' ';     // forced 8-char ANNAME
```

Hard-coded array reference (X,Y,Z) written into `ARRAYX/Y/Z` of the AN table:
`(1657004.6290, 5797894.3801, 2073303.1705)` metres, i.e. **GMRT** —
non-configurable.

### 5.5 `pttest.dat` — point-source catalogue

Read by `Read_Point_Sources()`. The number of sources is determined by
counting newline characters. The format per row, parsed by

```c
fscanf(fp, "%f%f%*f%f", &xPS0[ii], &yPS0[ii], &alphapt[ii]);
for(jj=0;jj<NCHAN;++jj) fscanf(fp, "%f", &IPS[ii][jj]);
```

is therefore:

| Field | Type | Meaning |
|-------|------|---------|
| `xPS0` | float | direction-cosine `l = cos(δ) sin(Δα)` (radians) — **already projected** onto the tangent plane of the phase centre |
| `yPS0` | float | direction-cosine `m` (radians) |
| (skip) | float | — read-and-discarded with `%*f` |
| `alphapt` | float | spectral index (informational only — actual SED is given by the per-channel fluxes that follow) |
| `IPS[ii][0..NCHAN-1]` | float×NCHAN | brightness in Jy at each channel |

The shipped file `pttest.dat` is a **single source** at `(l,m) = (0.118, 0.0)`
rad, spectral index 0.8, and a 128-channel SED smoothly spanning 1.044 → 0.96
Jy that traces a roughly `(ν₀/ν)^α` law. The total number of fluxes per row
must equal the `NCHAN` in `INP.VISFITS`.

### 5.6 Diffuse image cube (UVFITS-side input)

`Read_Image_FITS()` opens `argv[4]` (e.g. `difftest.fits`), checks
`naxes[0] == naxes[1]`, reads `CDELT1, CDELT2` (interpreted as **arcmin**,
converted to radians), and then `naxes[0]*naxes[1]*naxes[2]` floats into
`Idata[]`. Channels axis is `naxes[2]`, frequency-corrected through
`freqcorr = 1 + ((chan - NCHAN/2) * CWOBS) / FREQOBS` inside `Gen_Vis`.

### 5.7 `input.grf` — diffuse generator parameter file

Single-line, whitespace-separated; from the shipped sample
`simulators/VisSim/diffuse/input.grf`:

```
-850 512 0.4046 150.0e6 62.5e3 129.0 256 20 95.0 0.8 0.1 513.0e-06 2.34
psnobeamunity.dat psbeamunity.dat
```

Format (parsed by `grf.c`):

| # | Symbol | Meaning |
|---|--------|---------|
| 1 | `seed` | GSL CMRG RNG seed (long) |
| 2 | `N` | grid size (must be even) |
| 3 | `L` | pixel resolution (arcmin) |
| 4 | `nu0` | reference frequency (Hz) |
| 5 | `deltanu` | channel width (Hz) |
| 6 | `chan0` | reference channel (FORTRAN-style fractional pixel) |
| 7 | `Nchan` | number of channels |
| 8 | `NB` | number of bins for power-spectrum diagnostic |
| 9 | `theta0` | normalisation angle (arcmin) — `A = π θ₀² / 2` |
| 10 | `spindex` | spectral index `α` for `(ν₀/ν)^α` scaling |
| 11 | `delspindex` | running of α: `α + Δα · log₁₀(ν/ν₀)` |
| 12 | `A_150` | overall amplitude (mK² at 150 MHz, in C_ℓ-style units) |
| 13 | `betav` | power-law slope of `P(U) ∝ (1000/(2π U))^β` |
| 14 | `psnobeam` | filename to write the no-beam power spectrum |
| 15 | `pswbeam` | filename to write the with-beam power spectrum |

---

## 6. Outputs

### 6.1 UVFITS produced by `visfits`

`Init_HDR()` writes a primary HDU with **GROUPS** layout and 7 NAXIS axes:

| NAXIS | Value | CTYPE | CRVAL | CDELT | CRPIX |
|-------|------:|-------|------:|------:|------:|
| 1 | 0 | (random groups) | — | — | — |
| 2 | 3 | `COMPLEX` | 1 | 1 | 1 |
| 3 | 2 | `STOKES` | -1 | -1 | 1 |
| 4 | NCHAN | `FREQ` | `FREQOBS·1e6` | `CWOBS·1e6` | `NCHAN/2 + 1` |
| 5 | 1 | `IF` | 1 | 1 | 1 |
| 6 | 1 | `RA` | 0 | 1 | 1 |
| 7 | 1 | `DEC` | 0 | 1 | 1 |

Random-group parameters (`PCOUNT=8`):

```c
PTYPE = {"UU---SIN","VV---SIN","WW---SIN","BASELINE","DATE","DATE","SOURCE","FREQSEL"}
PSCAL = {1e-6/FREQOBS, 1e-6/FREQOBS, 1e-6/FREQOBS, 1, 1, 1, 1, 1}
PZERO = {0, 0, 0, 0, DATEOBS, 0, 0, 0}
```

So U, V, W are written as wavelengths *divided by* `FREQOBS·1e6` and
multiplied by `1e-6`, i.e. effectively in units of seconds (`light-travel
time`). They are recovered by AIPS as wavelengths via `PSCAL × FREQOBS`. The
two `DATE` parameters split the integer JD (PZERO) and the fractional UT
respectively. `BASELINE = 256·a1 + a2` (1-based), per AIPS convention — in
the historical "no array number" form (no `/10` term).

Stokes axis is **−1, −2** ⇒ `RR, LL` (AIPS sign convention) although the AN
table records the receptor frames as `POLTYA = 'X'` and `POLTYB = 'Y'`. This
is an inconsistency baked into the code.

After the primary HDU, three binary tables are appended in this order:

| Table | EXTNAME | Source function |
|-------|---------|-----------------|
| Antenna | `AIPS AN` | `Write_AG_TABLE` — 12 columns, `NANTE` rows |
| Frequency | `AIPS FQ` | `Write_FQ_TABLE` — 5 columns, 1 row |
| Source | `AIPS SU` | `Write_SU_TABLE` — 19 columns, `NSOUR` rows |

Notable AN-table keywords (hard-coded GMRT geometry):

| Keyword | Value |
|---------|------:|
| `ARRAYX` | `1657004.6290` m |
| `ARRAYY` | `5797894.3801` m |
| `ARRAYZ` | `2073303.1705` m |
| `DEGPDY` | `360.98564497436661` |
| `IATUTC` | `34.0` |
| `ARRNAM` | `GMRT` |
| `POLTYPE` | `APPROX` |
| `P_REFANT` | 5 |

### 6.2 FITS image cube produced by `grf`

3-D `DOUBLE_IMG`, `(N, N, Nchan)`, written by `SWRITE_HDR()` of
`fitsprog.c`. Header keywords:

| Axis | CTYPE | CRVAL | CDELT | CRPIX |
|------|-------|------:|------:|------:|
| 1 | `thetax` | 0 | `pixel` (arcmin) | `N/2 + 1` |
| 2 | `thetay` | 0 | `pixel` (arcmin) | `N/2 + 1` |
| 3 | `3IMAGE` | `nu0` (Hz) | `deltanu` (Hz) | `chan0` |

(The stray `"3IMAGE"` literal is a typo; `BWRITE_HDR` uses `"RE_IM3"` for the
Fourier-side cube. Both paths write `naxis = 3` headers.)

### 6.3 Diagnostic power spectra (`grf`)

After running, `grf` writes two two-column ASCII files specified by
`psnbeam` / `pswbeam`. Each row is

```
binu  num   P_I(nu0, U)   measured_power
```

— bin-centre `U` (wavelengths), counts in bin, theory `P_I` and measurement,
respectively without and with primary-beam multiplication.

---

## 7. Algorithms — verbatim from the source

### 7.1 Earth-rotation U,V,W transform (`Calc_uvw` in `funcs.c`)

For a given Hour-Angle `H` (deg) and source declination `δ`, the standard
rotation is applied to each antenna's local XYZ:

```
u = sin H · bx + cos H · by
v = -sin δ · cos H · bx + sin δ · sin H · by + cos δ · bz
w =  cos δ · cos H · bx - cos δ · sin H · by + sin δ · bz
```

(no precession, no nutation, no aberration). This matches Thompson, Moran &
Swenson up to GMRT's local frame conventions.

The hour-angle of a record is computed once at the start of each scan from

```
gst   = jd2gst(JD₀ + STTIM)                # in degrees
gst   = gst - floor(gst) + ((int)gst)/360. # wrap
H     = gst + TELE_LON - sourceRA[0]       # NB: uses RA of FIRST source
H    += 360 · (record · INTIME) / 86400    # increment per record
```

Subtle detail: the hour-angle is anchored to **`souRA[0]`** even when other
sources are processed (`Calc_uvw(H, SUID)` then uses `souDC[SUID-1]`). This
is OK only when all listed sources share roughly the same RA, which is the
case in the shipped `source.vf`.

### 7.2 Sidereal-time helper (`utils.c::jd2gst`)

```
T   = (JD - 2451545.0) / 36525
GST = (24110.54841 + 8640184.812866·T + 0.093104·T² - 6.2e-6·T³) / 240
```

Returned value is in **degrees**. Coefficients are the IAU-1982 expression.

### 7.3 Visibility kernel (`Gen_Vis` in `funcs.c`)

Conditioned on the compile switches:

```
freqcorr = 1 + ((chan - NCHAN/2) · CWOBS) / FREQOBS
ν        = FREQOBS · freqcorr · 1e6
(u',v',w') = (u, v, w) · freqcorr     # convert to current channel's wavelength
U        = sqrt(u'² + v'² + w'²)

if DIFF and U < Umax:
    omp parallel for index in [0, npixels):
        i = index % NGridI; j = index / NGridI
        θx = (i - NGridI/2) · DIx
        θy = (j - NGridI/2) · DIy
        φ  = 2π · (u'·θx + v'·θy + w'·(sqrt(1 - θx² - θy²) - 1))
        VR += I_diffuse[chan, index] · cos φ
        VI += I_diffuse[chan, index] · sin φ
    # diffuse contribution is **NOT** primary-beam-multiplied here
    # (the beam was already baked into the input image by grf)

if POINT:
    for each point source s:
        θ  = sqrt(l² + m²)
        b  = Beam(θ, ν)                    # Bessel-J1 Airy
        φ  = 2π · (u'·l + v'·m + w'·(sqrt(1 - l² - m²) - 1))
        VR += I_s[chan] · b · cos φ
        VI += I_s[chan] · b · sin φ
```

Sign of the phase: `exp(+i·2π(u·l + v·m + w·(n−1)))` (positive convention).
This is *opposite* to the radio-astronomy Wiener convention used by AIPS/CASA,
which use `exp(−2πi · ...)`. Whether downstream packages flip it is
out-of-scope here — call it a code-internal convention.

`w·(sqrt(1 − l² − m²) − 1)` is the full **w-term**, so VisSim is not
restricted to small fields of view *for the integrand* — only to `l² + m² < 1`.

### 7.4 Primary beam (`beam.c`, identical in both subdirs)

```
D = 45 m (dish diameter)
x = π · θ · D · ν / c
b(θ) = (2 J₁(x) / x)²       (Airy)
b(0) = 1
```

— exactly what is expected for an unblocked circular aperture. The code
**ignores the secondary `d=1` parameter** (would-be central blockage
diameter); a commented-out Gaussian alternative
`b = exp(-θ²/(9.8e-3)²)` is also present.

### 7.5 Gain errors (compile switch `GERR`)

For every (record, antenna):

```
δAi  = δAi_const(timebin) + N(0, SIG_A)
δφi  = δφi_const(timebin) + N(0, SIG_P)
g_re = (1 + δAi) cos(δφi)
g_im = (1 + δAi) sin(δφi)
```

The constant-per-time-bin component is drawn once at the top of `Write_scan`;
the random per-record component is drawn inside the antenna loop. Then for
each baseline (a, b):

```
v_re = ga_re·gb_re·V_re + ga_im·gb_im·V_re + ga_im·gb_re·V_im - ga_re·gb_im·V_im
v_im = ga_im·gb_im·V_im + ga_re·gb_re·V_im + ga_re·gb_im·V_re - ga_im·gb_re·V_re
```

This is the real-part / imag-part form of `V' = g_a · V · conj(g_b)` (note
**conj(g_b)**, not `g_b` — the signs match `g_b* = (gb_re - i·gb_im)`). In
the absence of `GERR`, the visibility is written verbatim.

### 7.6 System noise (`NOISE`)

```
V_re += N(0, SIG_N)
V_im += N(0, SIG_N)
```

Independent Gaussian draws on the two parts. There is no per-baseline /
per-channel weighting — `SIG_N` is a single scalar.

### 7.7 Random-UV mode (`flagran == 1`)

If enabled (in `INP.VISFITS`), every (u, v) pair is **replaced** by a fresh
uniform draw in `[-Umax, +Umax]` (in λ), and `w := 0`. This is useful for
generating Monte-Carlo testing data with controlled UV coverage but no real
geometry.

### 7.8 Diffuse generator `grf.c` — Gaussian random field with prescribed P(U)

The model implemented is a **2-D zero-mean isotropic Gaussian random field**
on a square `N × N` grid with physical side `length = N · L` rad. Its
expected angular power spectrum at `ν₀` is

```
P_I(ν, U) = (2 k_B ν² / c²)² · A_150 · (ν₀/ν)^(2α) · (1000 / (2π U))^β
```

where `k_B = 1.38e3 Jy/K` (this odd unit is so that a 1 K brightness on the
RJ tail at GHz scales gives Jy/sr — see the conversion below), `A_150 = 513e-6`
in the shipped config, `α = 2.8`, `β = 2.34`. The amplitude factor in
front converts the temperature-fluctuation power spectrum to specific-intensity
power spectrum on the Rayleigh-Jeans tail.

**Construction** (lines 124-173 of `grf.c`):

1. For every Fourier mode `(i, j)` in the upper-half plane (real-DFT layout
   `ydim = N/2 + 1`), set
   `U = sqrt(i² + j²) / length`,
   `amp = fac · sqrt(P_I(ν₀, U))`, with
   `fac = L / (sqrt(2)·N)`,
   then draw `Re, Im ~ amp · N(0,1)` and store
   `in[k] = (-1)^(i+j) · (Re + i·Im)` (the `(-1)^(i+j)` recentres the image
   so DC sits at `(N/2, N/2)`).
2. Enforce Hermitian symmetry along the `j=0` and `j=N/2` axes by mirroring.
3. The four real-only corners are set with imaginary part 0.
4. `fftw_execute(p)` performs `c2r` 2-D inverse DFT.
5. Per output channel `k = 0..Nchan-1`:
     ν = ν₀ + (k+1.5−chan0)·Δν,
     for every pixel multiply by
     `Beam(θ, ν) · (ν₀/ν)^(α + Δα · log₁₀(ν/ν₀))`
   and write that 2-D plane as channel `k+1` of the output FITS cube.

The diagnostic `power_spec()` routine bins `|Ĩ|²` logarithmically in `U` and
writes both the raw and the post-beam spectra to the two ASCII files.

### 7.9 Gridded-vis injection — `visfitsgrid.c`

For each plane of the input image cube:

```
out[i, j]   ← img[j, i]                       # transpose
in          ← FFT_r2c(out)                    # FFTW r2c
reim[k, i, j] ← (-1)^(i+j) · (Re,  -Im)       # de-shift, conjugate (r2c sign)
```

Then for every existing UVFITS group:

```
factor = 1 + (Δν / ν₀) · (chan + 1.5 − chan0)
uu = sign(v) · u · factor · ν₀
vv = sign(v) · v · factor · ν₀     # NOTE: the source uses randpar[0] *twice* — see below
ii  = round(uu / dU); if ii<0 then ii += N
jj  = round(vv / dU)
data[chan, stokes, RE/IM] = fac · data + reim[chan, ii, jj]·{(+,sign(v))}
```

`fac` is the third command-line argument: 0 ⇒ overwrite, 1 ⇒ add to existing.
`dU` is computed from the image `CDELT1` (deg) as `180 / (π · N · CDELT1)`.

> **Bug to be aware of (lines 161-162):**
> ```c
> uuc = signv*randpar[0]*corrfact[chan]*nu_chan0;
> vvc = signv*randpar[0]*corrfact[chan]*nu_chan0;   // should be randpar[1]
> ```
> Both U and V are taken from `randpar[0]`. This is faithfully reproduced
> here because we are documenting the code as it exists.

### 7.10 Cosmology utilities (`funcs_comov.c`)

For an `(Ωₘ, Ωₖ)` cosmology with `h = H₀/100`, the module exposes:

* `func_E(a) = sqrt(Ωₘ a⁻³ + Ωₖ a⁻² + Ω_Λ)` where `Ω_Λ = 1 − Ωₘ − Ωₖ`.
* `ceta(a) = ∫_a^1 (a²·E(a))⁻¹ da · (c/H₀)` via `simp()`.
* `rnu(a)` — comoving distance for flat / open / closed:
  `LL · sinh(ceta/LL)`, `ceta`, `LL · sin(ceta/LL)` respectively, with
  `LL = (c/H₀)/sqrt(|Ωₖ|)`.
* `rnup(a)` — `dr/dν` evaluated at the 21-cm rest frequency 1420 MHz.

These functions are referenced only by the missing `gen_bubble.c`; nothing in
`grf` / `visfits` / `visfitsgrid` calls them. `simp.c` is a textbook composite
Simpson 1/3 rule for `n` panels — also only used by `funcs_comov.c`.

---

## 8. Public API / CLI surface

There is no library — everything is a CLI executable. Quick reference:

| Binary | Where | Args (positional) |
|--------|-------|-------------------|
| `visfits` | `point_src/` | `INP.VISFITS scan.vf source.vf diff_image.fits ptsrc.dat out.uvfits NTHREADS` |
| `version` | `point_src/version.c` (separate, not in the Makefile) | — — prints version string |
| `COPYLEFT` | `point_src/` | `COPYLEFT.INFO` (its built form is committed as a Mach-O binary) |
| `grf` | `diffuse/` | `input.grf out_image.fits` |
| `grf_unity`, `grf_rea`, `grf_unity_rea` | declared in `makefile`, not present | — |
| `visfitsgrid` | `diffuse/` | `input_image.fits uvfits_template factor` (factor=0 overwrite, 1 add) |
| `gen_bubble` | declared, not present | — |

Sample driver scripts:

```bash
# simulators/VisSim/point_src/run.sh
NTHREAD=25
PS=pttest.dat
make
time ./visfits INP.VISFITS scan.vf source.vf difftest.fits ${PS} visdiff.fits ${NTHREAD}
```

```bash
# simulators/VisSim/diffuse/run.sh   (the body is commented out)
path="/home/samir/simmulti/diffmulti"
export path
for i in `seq 1 1`; do
    # sed "s/-850/-2$i"79"/" input.grf > input.grfcp
    # $path/grf input.grfcp img_diff$i.fits
    # cp R53D22.RES.UVFITS tmp/visdiff$i.fits
    # $path/visfitsgrid tmp/img_diff$i.fits tmp/visdiff$i.fits 0.
done
```

---

## 9. File-by-file reference

### 9.1 `point_src/`

| File | Type | Contents |
|------|------|----------|
| `Makefile` | make | Builds `visfits`. Hard-codes `/home/samir/astro/{lib,include}`. |
| `visfits.c` | C | `main()`. Parses 7 argv, invokes the pipeline. 75 lines. |
| `visfits.h` | C header | Macros & function decls. Compile switches `POINT/DIFF/NOISE/GERR`. |
| `eor2fits.h` | C header | Older near-duplicate of `visfits.h` (still referenced by `utils.c` and `version.c` for the `COPYLEFTFILE` macro). |
| `version.h` | C header | `HVERSION 0`, `LVERSION 1`. |
| `version.c` | C | Tiny standalone executable that prints the formatted version string. **Not** wired into the Makefile. |
| `funcs.c` | C, ~1275 lines | Implements `Calc_uvw`, `Read_Inputs`, `Init_PARM`, `Init_HDR`, `Write_SU_TABLE`, `Write_AG_TABLE`, `Write_FQ_TABLE`, `Write_scan`, `Write_FITS_data`, `Gen_Vis`, `Read_Image_FITS`, `Read_Point_Sources`, `printerror`, `get_version`, `Legals`. |
| `funcs.c.nrcp` | C, 1222 lines | Older variant: uses Numerical Recipes `gasdev` / `ran1` rather than GSL CMRG; SU-table form `2E` for fluxes (i.e. 2 IFs); no random-UV switch; missing the `freqcorr` rescaling of (u,v,w) in `Gen_Vis`. **Not used by the build** — kept for reference only. |
| `beam.c` | C, 33 lines | `double Beam(double θ, double ν)` — Airy J₁ pattern for D = 45 m. |
| `utils.c` | C, 114 lines | `CHKFILE`, `COPYLEFT`, `replace_nulls`, `jd2gst`, `jd2iau_date`, `HMS2D`. |
| `COPYLEFT.c` | C, 25 lines | Standalone tool that prints lines beginning with `$` from a file — the same logic now embedded in `utils.c::COPYLEFT`. |
| `COPYLEFT` | bin | Mach-O 64-bit executable built from `COPYLEFT.c` (committed accidentally — usually a build artefact). |
| `COPYLEFT.INFO` | text | Author / licence block, see §1. |
| `INP.VISFITS` | text | Master parameter file (see §5.1). |
| `scan.vf` | text | One scan, IDs `1,1,1`, records 1 to 1800. |
| `source.vf` | text | One source, name `0823+26`, RA 10ʰ46ᵐ, Dec +59°. |
| `GMRT_ANT.DAT` | text | 30-row antenna XYZ table. |
| `pttest.dat` | text | One source, 128-channel SED. |
| `run.sh` | bash | `make && time ./visfits INP.VISFITS scan.vf source.vf difftest.fits pttest.dat visdiff.fits 25`. |

### 9.2 `diffuse/`

| File | Type | Contents |
|------|------|----------|
| `makefile` | make | Builds 6 binaries (only `grf`, `visfitsgrid` and partially `gen_bubble` have sources). |
| `grf.c` | C, 309 lines | Generates a Gaussian random field FITS image cube with prescribed angular power spectrum + frequency scaling + (Airy) primary beam. |
| `fitsprog.c` | C, 122 lines | Helper module: `BWRITE_HDR`, `BREAD_HDR`, `SWRITE_HDR`, `SREAD_HDR` — read / write 3-axis FITS image headers. `naxis = 3` is hard-coded in `fitsprog.h`. |
| `fitsprog.h` | C header | `naxis = 3`; declarations for `printerror` and the four header helpers. |
| `beam.c` | C, 32 lines | Same Airy beam as `point_src/beam.c`. |
| `visfitsgrid.c` | C, 184 lines | Reads diffuse image cube, FFTs each channel, then for every existing UVFITS group rounds (u,v) onto the FFT grid and adds (or overwrites) the Re/Im. |
| `read_fits_func.c` | C, 290 lines | UVFITS reader: `read_fits_header`, `read_ranpar`, `read_data`, `close_fits`, and a meta-`readfits()` that fills `RR/LL` arrays for all groups, accumulating mean/RMS as it goes. |
| `read_fits_func.h` | C header | Declarations for the above. |
| `funcs_comov.c` | C, 96 lines | Cosmology helpers (`E(z)`, `ceta`, `rnu`, `rnup`, `initialize`). |
| `funcs_comov.h` | C header | Declarations + the proto `double cera(double x);` (typo for `ceta`). |
| `simp.c` | C, 24 lines | Composite Simpson's rule. |
| `input.grf` | text | Sample diffuse-generator parameters. |
| `run.sh` | bash | Driver loop, fully commented out. |

---

## 10. Integration & extension points

VisSim has no Python bindings, no shared library and no plug-in system; it
talks to other tools strictly through file formats:

| Boundary | Format | Direction |
|----------|--------|-----------|
| Sky → simulator | ASCII point catalogue (`pttest.dat`) | in |
| Sky → simulator | FITS image cube (any 3-D `(N,N,Nchan)` with `CDELT1 == CDELT2`, in arcmin) | in |
| Simulator → user | UVFITS (random-groups primary HDU + AIPS AN/FQ/SU tables) | out |
| Diffuse generator → simulator / gridder | FITS image cube produced by `grf` is a drop-in for the `argv[4]` of `visfits`, or the input of `visfitsgrid` | out → in |

Because VisSim writes **standard AIPS-compatible UVFITS**, downstream tools
that consume UVFITS (AIPS, CASA's `importuvfits`, `pyuvdata`, RRIVis itself
via `pyuvdata`-style readers) can read its output. Caveats listed in §11.

To extend the simulator one would typically:

* Add a new sky term — drop it inside `Gen_Vis()` along the same pattern as
  the `POINT` and `DIFF` blocks, behind a new compile-time switch.
* Add a new beam — replace the body of `point_src/beam.c::Beam(θ, ν)`; the
  signature is fixed (one scalar in / one scalar out, two arguments).
* Add a new array — replace `GMRT_ANT.DAT`, change `NANTE` in `visfits.h`
  and the hard-coded `ARRAYX/Y/Z` in `Write_AG_TABLE`, then recompile.
* Add a new instrumental error — extend `Write_scan()` (the visibility
  inner-most loop is the only place where calibration / noise are applied).

---

## 11. Known limitations & TODOs

1. **No build/runtime portability.** Both makefiles assume
   `/home/samir/astro/{lib,include}` and require Numerical Recipes
   (`-lnrcp`). On a fresh machine the user must fetch / vendor NR or rip out
   `julday()` and `SQR()`.
2. **Hard-coded GMRT.** `NANTE = 30`, `ARRAYX/Y/Z`, dish diameter `D = 45`
   in the Airy beam, and the `REF_LON = 82.5` constant are all bound at
   compile time. There is no telescope-abstraction layer.
3. **Stokes confusion.** Output declares `STOKES = -1, -2 (RR, LL)` in the
   primary HDU but the AN table writes `POLTYA = 'X'`, `POLTYB = 'Y'`. Users
   pulling Stokes I from the file must pick a convention manually.
4. **Hour-angle anchored to source 0.** All scans use `souRA[0]` as the
   phase-tracking RA when computing `H` (see §7.1). Multi-source observations
   that span widely separated RAs will be wrong.
5. **No precession / nutation / aberration / refraction / Earth rotation
   variation** beyond the IAU-1982 GST polynomial.
6. **Single-IF only.** `NSIDE = 1` is hard-coded; the FQ table writes only
   one row.
7. **`POINT` and `DIFF` are switched off by default** in `visfits.h`. Without
   recompiling with at least one of them on, the simulator emits noise-only
   visibilities. This is a footgun.
8. **Output file must not exist** — `Init_HDR()` exits with a hard error on
   pre-existing path, so re-runs require explicit `rm`.
9. **`visfitsgrid.c` line 162** uses `randpar[0]` for both `uu` and `vv` (see
   §7.9). This appears to be a copy-paste bug.
10. **Missing source files** referenced by `diffuse/makefile`:
    `grf_unity.c`, `grf_rea.c`, `grf_unity_rea.c`, `gen_bubble.c`. The
    `target:` in the makefile will therefore fail on a clean checkout.
11. **No tests.** No unit, regression or smoke tests anywhere in the tree.
12. **No CI / lint / formatter / style guide.**
13. **Mach-O binary committed.** `point_src/COPYLEFT` is a compiled x86_64
    macOS executable accidentally checked in; it should be a build artefact,
    not a tracked file.
14. **`cera` vs `ceta` typo** in `funcs_comov.h` — only matters for the
    missing `gen_bubble.c`.
15. **No Python wrapper.** The package as it stands is not a candidate for
    direct embedding in RRIVis; integration would have to be at the UVFITS
    boundary.

---

## 12. Quick-start (best-effort)

Assuming `cfitsio`, `fftw3`, `gsl`, and a Numerical-Recipes-for-C build are
available, on the original developer's Linux box:

```bash
cd simulators/VisSim/point_src
# edit Makefile so INCLUDE / LINKLIB point at your installed libs
./run.sh                  # runs `make` then `./visfits ...` with NTHREAD=25
```

For a diffuse-only workflow:

```bash
cd simulators/VisSim/diffuse
make grf visfitsgrid
./grf input.grf img_diff.fits
# Then either:
#   (a) feed img_diff.fits as the 4th argument to visfits, or
#   (b) ./visfitsgrid img_diff.fits some_template.uvfits 0
```

The visibilities of shipped `INP.VISFITS` cover **JD 12 Feb 2007, 21:00:00 UT,
GMRT, 150 MHz centre, 16 MHz bandwidth, 128 channels, 1800 records × 16 s ≈
8 h, 30 antennas, 435 baselines** — i.e. ≈ 783 000 random groups in the
output UVFITS.
