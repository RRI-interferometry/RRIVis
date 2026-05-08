# Obit — Exhaustive Technical Reference

> Location of source: `simulators/Obit/` (git submodule).
> All file paths in this document are given relative to the repository root
> `/Users/RRI-interferometry/RadioSim/simulators/Obit/`.

---

## 1. Overview

**Obit** is a development environment for radio (and to a lesser degree
single-dish / OTF) astronomy data reduction. It is authored and maintained by
**W. D. Cotton (Bill Cotton, NRAO)** with contributions from Mark Kettenis
(JIVE) and Daniele Biancu (IRA). Obit pre-dates and complements NRAO AIPS:
it reuses the **AIPS catalog and table format on disk**, can call AIPS
tasks, and is interactively driven through a Python REPL called **ObitTalk**.
Beyond a CLEAN/imaging/calibration pipeline, the package includes:

- a large C class library (~245 `.c` files, ~700 headers) implementing UV
  data, images, tables, sky models, calibration solvers, beams, plotting, etc.;
- SWIG-generated Python bindings to most of those classes;
- a set of stand-alone task executables (98+ in `Obit/tasks/`, plus 12 in
  `ObitSD/tasks/`) — each task has a `.TDF` adverb-definition file and a
  matching C `main()`;
- **ObitTalk** — a ParselTongue-derived Python shell that exposes both
  AIPS tasks and Obit tasks under a common interface (run remotely via
  XML-RPC or locally);
- **ObitView** — a Motif/X11 FITS image viewer (`XFITSview`) with a
  companion **ObitMess** task-message server;
- **ObitSD** — single-dish "On-The-Fly" (OTF) processing for the Green
  Bank Telescope (GBT). The top-level `InstallObit.sh` now marks `ObitSD`
  as `defunct` (`doObitSD=no`), although the source tree is still shipped.
- pipelines for EVLA continuum/low-band, ALMA, VLBA, MeerKAT, LinPolCont
  written as Python ObitTalk scripts (`ObitSystem/Obit/python/EVLAContPipe.py`,
  `MeerKATPipeline.py`, `ALMAPipe.py`, `VLBAContPipe.py`, ...).

### Purpose

Obit is positioned as the modern successor to AIPS POPS-style processing.
It targets:

- general radio interferometric calibration (gain, bandpass, polarization,
  Faraday rotation, ionospheric, baseline-based);
- imaging and CLEAN — 2-D/3-D, faceted, Cotton-Schwab, multi-scale,
  multi-frequency synthesis (MFS / wideband), peeling, primary-beam
  correction, mosaicking;
- spectral fitting, Faraday/RM synthesis, holography (`MapBeam`), source
  finding, ionospheric calibration, RFI excision;
- bridging to AIPS data files (`AIPSDir`, `AIPSCat`, `IOImageAIPS`,
  `IOUVAIPS`) and BDF/ASDM ingest (`ObitBDFData`, `ObitSDMData`,
  `BDFIn` task) used by JVLA/ALMA;
- GPU acceleration via CUDA (`ObitCUDAUtil.cu`, `ObitCUDAGrid.cu`,
  `ObitGPUGrid.c`, `ObitGPUSkyModel.c`).

### License

GNU **GPL v2 (or later)** — `simulators/Obit/ObitSystem/Obit/LICENSE`,
`ObitSystem/ObitView/LICENSE`, `ObitSystem/ObitSD/LICENSE`,
`ObitSystem/ObitTalk/COPYING`. Each `.c`/`.h`/`.TDF` file repeats the GPL
notice and lists Bill Cotton (NRAO) as the contact.

### Languages

| Language | Where | Approximate role |
|----------|-------|------------------|
| **C (ANSI/C99)** | `ObitSystem/Obit/src/`, `ObitSystem/Obit/tasks/`, `ObitSystem/Obit/include/`, `ObitSystem/ObitSD/src/`, `ObitSystem/ObitView/src/` | Core class library, task `main()`s, image viewer |
| **CUDA C++ (`.cu`)** | `Obit/src/CUDAFArray.cu`, `CUDAFInterpolate.cu`, `ObitCUDAGrid.cu`, `ObitCUDASkyModel.cu`, `ObitCUDAUtil.cu` | Optional GPU kernels for gridding and DFT sky model |
| **Python 2/3** | `Obit/python/`, `ObitTalk/python/`, `ObitSD/python/`, `Obit/share/scripts/` | High-level API, tasks driver, pipelines, scripts |
| **SWIG** | `Obit/python/Obit.swig`, `Obit/python/ObitTypeMaps.swig`, `Obit/python/Obit_wrap.c` (generated), `*.inc` interface fragments | Python ↔ C bindings |
| **Fortran 77** | `Obit/AIPS/*.FOR` (`GTPARM.FOR`, `INPUT.FOR`, `OBTST.FOR`) | AIPS interop helpers (built inside an AIPS install) |
| **Perl** | `Obit/bin/ObitTables.pl`, `Obit/bin/f2c.pl` | Code generators (table boilerplate, Fortran-to-C glue) |
| **Shell (sh/csh)** | `InstallObit.sh`, `makeDist.sh`, `other/Install3rdParty.sh`, `other/Clean3rdParty.sh`, `ObitSD/scripts/*.csh` | Build and pipeline driver scripts |
| **TDF / "POPS-DEF"** | `Obit/TDF/*.TDF`, `ObitSD/TDF/*.TDF` | Adverb-definition files describing each task's CLI parameters |
| **GNU Autoconf m4** | `Obit/m4/`, `ObitTalk/m4`, `ObitView/m4/`, `ObitSD/m4/` | Detect cfitsio, fftw, gsl, glib, motif, plplot, pgplot, python, xmlrpc, wvr, zlib, SSE, fPIC |
| **LaTeX** | `Obit/doc/OBITdoc.tex` (~6700 lines), `ObitTalk/doc/ObitTalk.tex`, `ObitTalk/doc/cookbook.xml` | Reference/cookbook documents |
| **Doxygen** | `Obit/doc/doxygen/` | Auto-generated API browser |
| **XML / DocBook** | `ObitTalk/doc/cookbook.xml` | Cookbook source |

### Version

`simulators/Obit/ObitSystem/ObitVersion` records `1.1.670` (the prior
top-of-line svn revision; the project moved from NRAO-svn to GitHub in
the commit `592b2f04 "switch from svn to git"` — see also
`Obit/share/scripts/getVersion.py`, which now uses
`git rev-list master --count` to derive the version stamp). The autoconf
`AC_INIT(Obit, 1.0, bcotton@nrao.edu)` (`configure.ac:33`) carries the
nominal release version `1.0`. The package is therefore a continuously
revisioned trunk and does **not** publish numbered Git tags
(`git tag` returns empty in `simulators/Obit`).

### Recent commits (head of `simulators/Obit`)

| Hash | Subject (first words) |
|------|----------------------|
| e46d4b4e | Switch to RHEL9, plotting/scripting, bug fixes |
| c4e72dad | Major fixes to MFBeam, XYPass, Bug fixes |
| 42ace5df | "Get sincos right" — `ObitSinCos` workaround |
| 5ce526d5 | Bug fixes, MeerKAT pipeline / SDM |
| 9287da5d | MeerKAT pipeline, imaging tools |
| 91d67bb1 | XYSim, XYDly, polarisation calibration updates |
| 3c22d593 | RMSyn efficiency upgrade, BDFIn fixes |
| 786f4904 | New `Slice.py`, `Blank.py`, GPU gridding fixes |
| 8235fedd | `keepLin` — keep poln-calibrated data in linear basis |
| e1d84da2 | New JI/JT (Jones) AIPS-style tables |
| 592b2f04 | Switch from svn to git |

There is one upstream remote (https://github.com/bill-cotton/Obit per
`migrateObit.text`).

---

## 2. Repository layout

```
simulators/Obit/
├── README                 # top-level install notes
├── InstallObit.sh         # master build script (sh)
├── makeDist.sh            # tarball builder
├── migrateObit.text       # svn → git migration recipe
├── Notes.test             # personal env-var note
├── ObitSystem/
│   ├── ObitVersion        # "1.1.670"
│   ├── Obit/              # the main C library + tasks + python bindings
│   ├── ObitSD/            # GBT / OTF single-dish (marked defunct)
│   ├── ObitTalk/          # python REPL & XML-RPC task driver
│   └── ObitView/          # Motif image viewer + ObitMess
└── other/
    ├── Install3rdParty.sh
    ├── Clean3rdParty.sh
    ├── distro_tools/      # files, userdistro.obit
    └── tarballs/          # 18 vendored 3rd-party source tarballs
```

### Per-component breakdown (top of `ObitSystem/`)

| Component | Size | Highlights |
|-----------|------|------------|
| `Obit/` | C library + 98 tasks + 231 python files | The main system |
| `ObitSD/` (~13 MB) | OTF single-dish (defunct) | GBT DCR/CCB/PAR/VEGAS/SP receivers |
| `ObitTalk/` (~3.7 MB) | Python | ObitTalk/ObitTalkServer/ObitTalk3 binaries, AIPSTask/ObitTask classes |
| `ObitView/` (~1.4 MB) | C/Motif | XFITSview, blink, movie, optionbox, XMLRPC server hooks |

### `ObitSystem/Obit/` directory tree

| Subdirectory | Size | Purpose |
|--------------|------|---------|
| `src/` | 12 MB | 245 `.c`/`.cu` files implementing every Obit class |
| `include/` | 4.4 MB | 709 headers (each class triplet `.h` + `Def.h` + `ClassDef.h`) |
| `python/` | 6.8 MB | 231 files: SWIG `.swig`/`.inc` interface fragments and `.py` shims |
| `tasks/` | 5.5 MB | 108 task `main()` C files, `Makefile.in` |
| `TDF/` | — | 96 `.TDF` adverb-definition files (one per task) |
| `bin/` | — | `f2c.pl`, `ObitTables.pl`, `obitinclude.py` (codegen helpers) |
| `lib/` | — | empty placeholder for built `libObit.a` |
| `share/scripts/` | — | 74 user scripts (calibration recipes, plotting, mosaic, RM fits) |
| `share/data/` | — | calibrator FITS models (3C48, 3C123, 3C138, 3C147, 3C196, 3C286, 3C295, 3C380, 0408-65, 1934-638), Perley-Butler 2013 CSV, NVSS/MGPS/SUMSS VZ tables, MeerKAT/KAT-7 templates, polarisation calibration tables |
| `m4/` | — | 16 macros for `cfitsio`, `curl`, `fftw[3]`, `glib-2.0`, `gsl`, `motif`, `pgplot`, `plplot`, `python`, `wvr`, `xmlrpc`, `zlib`, `fPIC`, `sse` |
| `AIPS/` | — | Fortran wrappers for embedding inside AIPS (`OBTST.FOR`, `GTPARM.FOR`, `INPUT.FOR`, `OBTST.HLP`, `LOGIN.CSH`, `AIPSlibs.CSH`, `ZOINTD.c`) |
| `dummy_cfitsio/` | — | Minimal stub used when CFITSIO is absent |
| `dummy_xmlrpc/` | — | Minimal stub used when xmlrpc-c is absent |
| `doc/` | — | `OBITdoc.tex` (6 776 lines), `OBITdoc.sty`, Doxygen output, `Obit.html` |
| `test/` | — | C-level unit tests (`testCleanVis.c`, `testFeather.c`, ...) |
| `testScripts/` | — | Python smoke tests |
| `testData/` | — | Sample data used by tests |
| `changes.log` | 5 328 lines | Hand-maintained change journal |

### `ObitSystem/ObitTalk/`

| Path | Purpose |
|------|---------|
| `python/ObitTalk.py` | Interactive shell entry: imports `AIPS`, `FITS`, `AIPSTask`, `ObitTask`, `AIPSData`, `FITSData`, `ObitScript`; sets up readline / `otcompleter` |
| `python/AIPS.py`, `AIPSData.py`, `AIPSTask.py`, `AIPSTV.py`, `AIPSUtil.py` | AIPS task / data abstractions |
| `python/FITS.py`, `FITSData.py` | FITS task / data abstractions |
| `python/ObitTask.py`, `ObitScript.py`, `Task.py` | Obit task driver, script runner, base Task |
| `python/MinimalMatch.py` | Adverb minimum-match parsing |
| `python/LocalProxy.py`, `XMLRPCServer.py`, `otcompleter.py` | Local/remote dispatch and tab completion |
| `python/Proxy/` | `AIPS.py`, `AIPSData.py`, `AIPSTask.py`, `FITSData.py`, `ObitScriptP.py`, `ObitTask.py`, `Popsdat.py`, `Task.py` (XML-RPC server side) |
| `python/Wizardry/` | `AIPSData.py` — direct-access "Wizardry" interface |
| `bin/ObitTalk.in`, `ObitTalk3.in`, `ObitTalkServer.in` | Driver scripts (autoconf-substituted) |
| `doc/ObitTalk.tex`, `ObitTalk.pdf`, `cookbook.xml`, `Report.bib` | User-facing documentation, EPS figures |
| `test/imean.py`, `jmfit.py`, `mandl.py`, `template.py`, `ObitTest.in` | ObitTalk regression tests |
| `AUTHORS` | "Mark Kettenis <kettenis@jive.nl> / Bill Cotton <bcotton@nrao.edu> / Daniele Biancu <dbiancu@ira.cnr.it>" |
| `INSTALL`, `NEWS`, `TODO`, `ChangeLog`, `ENVIRONMENT` | Stock GNU autotools docs |

### `ObitSystem/ObitView/`

| Path | Purpose |
|------|---------|
| `ObitView.c` | `XFITSview`/`ObitView` main: Motif-based FITS viewer |
| `ObitMess.c` | `ObitMess` task-message server (separate executable) |
| `clientFCopy.c`, `clientTest.c`, `testObitMess.py` | XML-RPC client samples |
| `src/imagedisp.c`, `Image2Pix.c`, `color.c`, `histo.c` | Pixel rendering, colour bar, histogram |
| `src/blinkbox.c`, `moviebox.c` | Blink and movie modes (multi-image / 3-D) |
| `src/cursor.c`, `markpos.c`, `lookpos.c`, `poslabel.c`, `drawbox.c` | Position read-out / overlay |
| `src/control.c`, `menu.c`, `optionbox.c`, `requestbox.c`, `messagebox.c`, `infobox.c`, `aboutbox.c`, `helpbox.c` | UI widgets |
| `src/AIPSfilebox.c`, `saveimwindow.c`, `textfile.c`, `scrolltext.c` | File-dialog & log-window helpers |
| `src/graph.c` | Graph overlay |
| `src/XMLRPCserver.c`, `XMLRPCTaskMessServer.c` | Remote-control hooks (used by ObitTalk) |
| `src/TMessMenu.c`, `taskmessage.c`, `taskmessagebox.c` | Task message integration with ObitMess |
| `src/logger.c`, `FStrng.c`, `toolbox.c` | Logging / string / tool palette |
| `aaaSomeFile.fits` | test asset |
| `ObitView.hlp` | online help |

### `ObitSystem/ObitSD/` (single-dish, defunct)

| Path | Purpose |
|------|---------|
| `src/ObitOTF.c`, `ObitOTFDesc.c`, `ObitOTFSel.c`, `ObitOTFArrayGeom.c`, `ObitOTFCal.c`, `ObitOTFCalBandpass.c`, `ObitOTFCalFlag.c`, `ObitOTFCalUtil.c`, `ObitOTFFlagUtil.c`, `ObitOTFGetAtmCor.c`, `ObitOTFGetSoln.c`, `ObitOTFGrid.c`, `ObitOTFSkyModel.c`, `ObitOTFSoln2Cal.c`, `ObitOTFUtil.c`, `ObitIOOTFFITS.c` | Core OTF data / calibration / gridding classes |
| `src/ObitDConCleanOTF.c`, `ObitDConCleanOTFRec.c` | Hogbom CLEAN for OTF |
| `src/ObitGBT*Info.c` (DCR, CCB, IF, VEGAS, SP, BeamOff) | GBT receiver back-end metadata parsers |
| `src/ObitTableGBT*.c`, `ObitTableOTF*.c` | GBT/OTF AIPS-style table classes |
| `src/Obit2DLegendre.c`, `ObitPennArrayAtmFit.c`, `ObitPennArrayUtil.c`, `ObitVEGASUtil.c`, `ObitGBTCCBUtil.c`, `ObitGBTDCROTF.c`, `ObitTableGBTPARSENSORUtil.c`, `ObitTableOTFTargetUtil.c` | Domain-specific utilities |
| `tasks/CCBCalib.c`, `CCBFix.c`, `CCBOTF.c`, `DCROTF.c`, `OTFImage.c`, `OTFSCal.c`, `PAROTF.c`, `PAROTF2.c`, `QuadDet.c`, `VEGASOTF.c` | OTF tasks |
| `python/CleanOTF.py`, `OTF.py`, `OTFDesc.py`, `OTFGetSoln.py`, `OTFGetAtmCor.py`, `OTFRec.py`, `OTFSoln2Cal.py`, `OTFUtil.py`, `GBTUtil.py`, `CCBUtil.py`, `VEGASUtil.py`, `PARCal.py` | Python wrappers + GBT helpers |
| `scripts/*.csh`, `scripts/*.py` | GBT calibration recipes (C/L/P/X-band, Daisy scans, etc.) |
| `TDF/*.TDF` | OTF task adverb files |

### `other/`

| Path | Purpose |
|------|---------|
| `Install3rdParty.sh` | Build all bundled third-party tarballs (8 327 bytes) |
| `Clean3rdParty.sh` | Tear-down counterpart |
| `tarballs/` | `cfitsio2490`, `cfitsio3100`, `curl-7.17.0`, `fftw-3.1.2`, `glib-2.2.0`, `gsl-1.6`, `gsl-1.13`, `openmotif-2.3.0`, `pkgconfig-0.14.0`, `plplot-5.8.0`, `Python-2.7.1`, `swig1.1-883`, `w3c-libwww-5.4.0`, `xmlrpc-c-1.06.18`, `zlib-1.2.3`, `boostheader_1_41_0`, `bnmin1-1.11`, `libair-1.0.1` |
| `distro_tools/` | `files`, `userdistro.obit` (manifest used by the distro builder) |

---

## 3. Installation & dependencies

### One-shot installer

`simulators/Obit/InstallObit.sh` is the canonical entry point. It:

1. Builds bundled third-party software in `other/tarballs/` via
   `other/Install3rdParty.sh` (unless `-without THIRD`).
2. Configures and `make`s `ObitSystem/Obit`, `ObitView`, and `ObitTalk`.
   `ObitSD` is hard-disabled (`doObitSD=no  # defunct`,
   `InstallObit.sh:23`).
3. Writes out POSIX/csh shell-rc files (`setup.sh`, `setup.csh`) that
   set `OBIT`, `OBITINSTALL`, `PYTHONPATH`,
   `PLPLOT_DRV_DIR`, `PATH`, `LD_LIBRARY_PATH`.

Selectable knobs (`InstallObit.sh -without …`):

```
PLPLOT  CFITSIO  GLIB  FFTW  GSL  ZLIB  MOTIF  PYTHON  CURL  XMLRPC
BOOST   WVR      THIRD Obit  ObitView  ObitTalk
```

### Dependencies

| Dependency | Detected by | Mandatory? | Notes |
|------------|-------------|------------|-------|
| `glib-2.0` | `m4/glib-2.0.m4` | Yes | Used everywhere; provides threads via `gthread-2.0` (controls `OBIT_THREADS_ENABLED`) |
| `cfitsio` | `m4/cfitsio.m4` | Effectively yes | Falls back to `dummy_cfitsio/` stubs |
| `xmlrpc-c` | `m4/xmlrpc.m4` | Effectively yes | Falls back to `dummy_xmlrpc/`; needed for ObitTalk/ObitView IPC |
| `fftw3` (or `fftw2`) | `m4/fftw3.m4`, `fftw.m4` | Yes | Float build |
| `gsl` | `m4/gsl.m4` | Yes | Numerics + special functions |
| `zlib` | `m4/zlib.m4` | Yes | gzipped FITS |
| `curl` | — | Optional | Used by xmlrpc-c HTTP transport |
| `python` (2.7+ legacy, 3 supported) | `m4/python.m4`, `python/setup.py` | Optional but standard | Required for ObitTalk |
| `swig` | configure | Build-time only | Regenerate `Obit_wrap.c` |
| `motif`/`lesstif` | `m4/motif.m4` | ObitView only | X11 widget toolkit |
| `plplot` (preferred) / `pgplot` | `m4/plplot.m4`, `pgplot.m4` | Plotting | `Obit/python/MK_Tools.py` is also adding a Matplotlib alternative (see commit `e46d4b4e`) |
| `libwww` | — | **Disabled by default** (`doWWW=no` in `InstallObit.sh`) |
| `boost` headers + `bnmin1` | `m4/wvr.m4` | WVR only | Bojan Nikolic's ALMA WVR library |
| CUDA toolkit | (manual) | GPU-only | See `src/ObitCUDAUtil.cu` and `Obit.h ObitHaveGPU` |
| AIPS install | `Obit/AIPS/LOGIN.CSH`, `AIPSlibs.CSH` | Optional | Required to invoke AIPS tasks via ObitTalk; `AIPS_ROOT` & `AIPS_VERSION` env vars |

Each component carries its own GNU autotools `configure`/`Makefile.in`
trees; configure macros live in
`Obit/m4/`, `ObitView/m4/`, `ObitSD/m4/`, and `ObitTalk/m4`.

### Manual fall-back

> "If you cannot get the InstallObit.sh script to work, the fallback is
> to install the needed third party manually (if not done by
> `InstallObit.sh`) and run the configure script in the subdirectories
> under `ObitSystem` Obit, ObitView, ObitTalk and ObitSD" — `README:71-78`.

For each subdirectory:

```sh
cd ObitSystem/<component>
./configure --exec_prefix=$OBITINSTALL --with-obit=$OBIT \
            --with-cfitsio=… --with-fftw3=… …
make all
make install      # ObitView and ObitTalk only
```

### Run-time environment

After install:

```sh
export OBIT=$OBITINSTALL/ObitSystem/Obit
export PYTHONPATH=$OBIT/python:$OBITINSTALL/opt/share/obittalk/python
export PATH=$OBITINSTALL/bin:$PATH
export LD_LIBRARY_PATH=$OBITINSTALL/other/lib:$LD_LIBRARY_PATH
# AIPS interop
export AIPS_ROOT=…
export AIPS_VERSION=…
```

The setup files `setup.sh` / `setup.csh` are auto-generated by
`InstallObit.sh:231-260`.

### AIPS interoperability

- `Obit/AIPS/LOGIN.CSH` is sourced by AIPS' `LOGIN.CSH` to add Obit
  shared libraries to the AIPS load path, so an AIPS user can invoke
  `OBTST.FOR` (the in-AIPS Obit test task) from the regular AIPS
  prompt.
- ObitTalk reads `AIPS_VERSION/SYSTEM/UNIX/POPSDAT.HLP` (via
  `ObitTalk/python/Proxy/Popsdat.py`) to find AIPS adverb defaults.
- Disk catalogs use the AIPS layout (`Obit/src/ObitAIPSDir.c`,
  `ObitAIPSCat.c`).

---

## 4. Build & runtime architecture

### Build artefacts

`ObitSystem/Obit/Makefile` (top-level, hand-written, not generated)
descends into `src`, `lib`, `python`, `tasks`. Targets:

| Target | Effect |
|--------|--------|
| `versionupdate` | runs `python share/scripts/getVersion.py $OBIT` (now a `git rev-list --count`) |
| `srcupdate` | builds `.o` files in `src/` |
| `libupdate` | archives `libObit.a` in `lib/` |
| `pythonupdate` | builds the SWIG extension `_Obit.so` |
| `taskupdate` | links each task `main()` against `libObit.a` |
| `tableupdate` | regenerates AIPS-table boilerplate via `bin/ObitTables.pl` |
| `docupdate` | LaTeX → PDF of `OBITdoc.tex` |
| `backuptar`, `distrib`, `copy` | tarball makers |
| `clean` | recursive clean |

`Obit/dummy_cfitsio/Makefile.in` and `Obit/dummy_xmlrpc/Makefile.in`
provide minimal stubs so the build succeeds when those libraries are
absent (the resulting binary then refuses runtime FITS/XMLRPC ops).

### Runtime

```
┌──────────────────────────────────────────────────────────────────┐
│  USER                                                            │
│                                                                  │
│  $ ObitTalk                       (interactive Python shell)     │
│  >>> task = ObitTask("MFImage")                                  │
│  >>> task.inFile = "src.uvfits"                                  │
│  >>> task.go()                                                   │
└────────────────────────┬─────────────────────────────────────────┘
                         │ XML-RPC (xmlrpc-c) or local fork/exec
                         ▼
┌──────────────────────────────────────────────────────────────────┐
│  ObitTalk python  (ObitSystem/ObitTalk/python/)                  │
│   AIPSTask · ObitTask · Task · Proxy/* · Wizardry/*              │
│   ObitScript (run-script remotely)                               │
└────────────────────────┬─────────────────────────────────────────┘
                         │ exec($OBIT/bin/<TaskName>) with InfoList
                         │ or call into _Obit (SWIG) directly
                         ▼
┌──────────────────────────────────────────────────────────────────┐
│  Obit task binary  ($OBIT/bin/MFImage, etc.)                     │
│   parses $TASK.in   (ObitParser → ObitInfoList)                  │
│   calls into libObit                                             │
└────────────────────────┬─────────────────────────────────────────┘
                         │ libObit.a (C, GPL)
                         ▼
┌──────────────────────────────────────────────────────────────────┐
│  Obit C class library  (ObitSystem/Obit/src/ + include/)         │
│                                                                  │
│   Obit (root) ──► ObitData ──► ObitImage / ObitUV                │
│   ObitDCon (deconvolution) ──► ObitDConClean ──► …MF, …WB        │
│   ObitSkyModel (point/grid/MF/VM[Beam,Squint,Ion,Poln])          │
│   ObitTable ──► ObitTableAN, BP, CC, CL, FG, FQ, GC, JI, JT, …    │
│   ObitUVCal[Bandpass|Calibrate|Flag|Jones|Polarization|Selection]│
│   ObitDoppler · ObitPrecess · ObitSkyGeom · ObitPosition         │
│   ObitFFT · ObitFArray · ObitCArray · ObitFInterpolate           │
│   ObitMatx (2x2 complex) · ObitVecFunc · ObitExp · ObitSinCos    │
│   ObitPlot (PLPLOT/PGPLOT)                                       │
│   ObitMultiProc · ObitThread · ObitThreadGrid (gthreads)         │
│   ObitCUDAUtil/ObitGPUGrid/ObitGPUSkyModel (CUDA)                │
└────────────────────────┬─────────────────────────────────────────┘
                         │ libcfitsio · libfftw3f · libgsl · libglib
                         │ libxmlrpc (IPC) · libplplot/libpgplot
                         │ libcuda (optional)
                         ▼
                   On-disk data (AIPS catalog or FITS)
                   AIPS$DA01:…  /  *.uvfits  /  *.fits
```

### Class system

Each Obit C class follows a Glib-style triplet of headers (a fact called
out in `Obit.h:39`):

```
ObitFoo.h          # public API (function prototypes)
ObitFooDef.h       # data members (struct fields)
ObitFooClassDef.h  # class function pointers (vtable)
```

A subclass `ObitBar` `#includes` the parent's `Def.h`/`ClassDef.h`
inside its own `Def.h`/`ClassDef.h` to obtain inheritance via nested
struct expansion. The base type ID is stored as the first
`gint32` of every object (see comment in `Obit.h:60`).

`ObitAll.h` is the convenience aggregate include for application code.

---

## 5. Public API and CLIs

### 5.1 Command-line entry points

| Binary | Source | Purpose |
|--------|--------|---------|
| `ObitTalk` | `ObitTalk/bin/ObitTalk.in` | Python 2 interactive shell |
| `ObitTalk3` | `ObitTalk/bin/ObitTalk3.in` | Python 3 variant |
| `ObitTalkServer` | `ObitTalk/bin/ObitTalkServer.in` | XML-RPC daemon for remote task launch |
| `XFITSview` / `ObitView` | `ObitView/ObitView.c` | Motif image viewer |
| `ObitMess` | `ObitView/ObitMess.c` | Task-message GUI server |
| `<TaskName>` (98+) | `Obit/tasks/<Task>.c` | Each task is a stand-alone C executable consuming a `<Task>.in` parameter file generated by `ObitParser` |

A typical task invocation (POPS-equivalent) is:

```sh
$ MFImage -input MFImage.in
```

The `.in` file is human-editable; ObitTalk normally writes one before
calling `os.system(taskname)`.

### 5.2 ObitTalk

Once started:

```python
>>> AIPS.userno = 1234            # AIPS user number
>>> from AIPSData import AIPSUVData, AIPSImage
>>> from FITSData import FITSUVData, FITSImage
>>> from ObitTask import ObitTask

>>> uv  = AIPSUVData("MYDATA", "UVDATA", 1, 1)
>>> task = ObitTask("MFImage")        # any C task
>>> task.inName  = "MYDATA"
>>> task.inSeq   = 1
>>> task.Sources = ["3C286"]
>>> log = task.go()                   # synchronous run, returns log
>>> task.inputs()                     # show adverbs
>>> task.help();  task.explain()      # POPS-style help
```

`AIPSTask("IMEAN")` works the same way and dispatches to the AIPS
binary. Both classes inherit minimum-match adverb names
(`MinimalMatch.py`).

### 5.3 OTObit utility module

`ObitSystem/Obit/python/OTObit.py` is the catch-all "POPS-equivalent"
shim loaded at ObitTalk startup. It exposes:

- `imhead`, `uvhead`, `tvlod`, `tvall`, `tabget`, `tabput`, `tput`,
  `tget`;
- `AIPSHelp("...")`, `ObitHelp("...")`;
- `AIPSUVData / AIPSImage / FITSUVData / FITSImage` factory glue;
- pre-imported `History`, `InfoList`, `Image`, `UV`, `OData`, `OErr`,
  `OSystem`, `OPlot`, `OPrinter`, `OWindow`, `ODisplay`,
  `Catalog`, `OSurvey`, `Table`, `TableUtil`, `TableList`.

### 5.4 Major C objects (with python counterparts)

| C class (`include/`) | Python wrapper (`python/`) | Responsibility |
|----------------------|---------------------------|---------------|
| `Obit` | `Obit.py` (SWIG) | Root virtual base, `ObitInitAll`, `ObitHaveGPU` |
| `ObitSystem` | `OSystem.py` | Global init/shutdown (AIPS dirs, FITS dirs, threading) |
| `ObitErr` | `OErr.py` | Error / log message stack |
| `ObitInfoList` | `InfoList.py` | Associative array used to pass adverbs into every routine |
| `ObitInfoElem` | (internal) | Element of an InfoList |
| `ObitData` (abstract) | `OData.py` | Common ancestor of `ObitImage` and `ObitUV` |
| `ObitImage` | `Image.py` | FITS/AIPS image, magic-blanking, `FArray` pixel container |
| `ObitImageMF` | `ImageMF.py` | "MF" wide-band (multi-frequency) image with sub-band planes |
| `ObitImageWB` | (internal) | Wide-band Sault-Wieringa imaging |
| `ObitImageDesc` | `ImageDesc.py` | WCS-/FITS-style image header |
| `ObitImageMosaic[MF/WB]` | `ImageMosaic.py` | Faceted mosaic of `ObitImage`s |
| `ObitImageInterp` | `ImageInterp.py` | Image cube interpolation in frequency |
| `ObitUV` | `UV.py` | Visibility data set (FITS or AIPS), I/O buffered |
| `ObitUVDesc` | `UVDesc.py` | UV data header/descriptor |
| `ObitUVSel` | (internal) | Selection state |
| `ObitUVCal[*]` | `UVSoln.py`, `UVSelfCal.py`, `UVGSolve.py`, `UVSoln2Cal.py` | Calibration (Cal, Bandpass, Baseline, Calibrate, Flag, Jones, Polarization, Select) |
| `ObitUVImager[/MF/WB/Ion/Squint]` | `UVImager.py` | Visibility-domain imager (gridding + FFT) |
| `ObitUVGrid[/MF/WB]` | (internal) | Convolutional gridder |
| `ObitUVWeight` | (internal) | Briggs/uniform/natural weighting |
| `ObitUVPeelUtil` | `PeelWork.py`, `PeelScripts.py` | Source peeling |
| `ObitUVRFIXize` | `UVRFIXize.py` | RFI excision |
| `ObitUVSelfCal` | `UVSelfCal.py` | Self-cal driver |
| `ObitDCon`, `ObitDConClean`, `ObitDConCleanImage`, `ObitDConCleanVis[/MF/WB/Line]` | `CleanImage.py`, `CleanVis.py` | Hogbom / Cotton-Schwab / multi-scale CLEAN |
| `ObitDConCleanWindow`, `ObitDConCleanPxList[/MF/WB]`, `ObitDConCleanPxHist`, `ObitDConCleanBmHist` | `OWindow.py` | CLEAN windows / pixel/beam histograms |
| `ObitSkyModel` | `SkyModel.py` | Predict-from-CC-table sky models |
| `ObitSkyModelMF` | (internal) | Wide-band point-source model |
| `ObitSkyModelVM` | (internal) | Time-variable model base |
| `ObitSkyModelVMBeam[MF]` | `SkyModelVMBeam.py` | Direction-dependent voltage-beam corrections |
| `ObitSkyModelVMSquint`, `ObitSkyModelVMIon`, `ObitSkyModelVMPoln` | `SkyModelVMIon.py` | Beam-squint, ionospheric, polarisation models |
| `ObitFullBeam` | `FullBeam.py` | Time-variable EVLA full-Stokes beam |
| `ObitBeamShape` | `BeamShape.py` | Analytic primary beam (Gaussian, cos², UV-derived) |
| `ObitTable` | `Table.py` | Generic AIPS-style table base |
| `ObitTable<XX>` (49 subclasses: AN, AT, BL, BP, CC, CD, CL, CP, CQ, CT, FG, FQ, FS, GC, IM, JI, JT, MC, MF, NI, NX, OB, OF, OT, PC, PD, PO, PS, PT, SN, SU, SY, TY, VL, VZ, WX, plus IDI_*) | `Table*.inc` / `TableUtil.py` | One per AIPS extension table |
| `ObitTableHistory` | `History.py` | Image/UV history records |
| `ObitTableCC`, `ObitTableCCUtil` | (internal) | CLEAN component tables |
| `ObitTableJI`, `ObitTableJT` | `TableJI.inc`, `TableJT.inc` | Jones-matrix calibration tables (new in commit `e1d84da2`) |
| `ObitTableIDI_*` | `TableIDI_*.inc` | FITS-IDI table classes (ANTENNA, ARRAY_GEOMETRY, BANDPASS, CALIBRATION, FLAG, FREQUENCY, GAIN_CURVE, INTERFEROMETER_MODEL, PHASE_CAL, SOURCE, SYSTEM_TEMPERATURE, UV_DATA, WEATHER) |
| `ObitFArray`, `ObitFArrayUtil`, `ObitCArray`, `ObitCInterpolate`, `ObitFInterpolate` | `FArray.py`, `FArrayUtil.py`, `CArray.py`, `FInterpolate.py` | Float / complex N-D arrays + interpolation |
| `ObitGPUFArray`, `ObitGPUFInterpolate` | `GPUFArray.py`, `GPUFInterpolate.py` | CUDA versions |
| `ObitMatx` | (internal) | 2×2 complex Jones matrix utilities |
| `ObitFFT` | `FFT.py` | FFTW3 wrapper |
| `ObitFaraSyn` | (internal, used by `Farad` task) | Faraday synthesis |
| `ObitRMFit` | `RMFit.py` | Rotation-measure fitting |
| `ObitSpectrumFit`, `ObitSpectrumInterp`, `ObitSpectrumMF` | `SpectrumFit.py` | Multi-term spectral fits |
| `ObitImageFit`, `ObitImageFitData`, `ObitFitModel`, `ObitFitRegion`, `ObitFitRegionList` | `ImageFit.py`, `FitModel.py`, `FitRegion.py` | Gaussian/CLEAN-component fitting |
| `ObitConvUtil`, `ObitFeatherUtil` | `ConvUtil.py`, `FeatherUtil.py` | Convolution and feather (combine) operators |
| `ObitTimeFilter` | `TimeFilter.py` | Time-domain filter |
| `ObitDoppler` | `Doppler.py` | Doppler/topocentric corrections |
| `ObitPrecess`, `ObitSkyGeom`, `ObitPosition`, `ObitPosLabelUtil` | `SkyGeom.py` | Coordinates, precession, geometry |
| `ObitPlot` | `OPlot.py` | PLPLOT/PGPLOT interface |
| `ObitDisplay` | `ODisplay.py` | XML-RPC bridge to ObitView |
| `ObitPrinter` | `OPrinter.py` | Multi-line text output |
| `ObitParser`, `ObitReturn` | `ParserUtil.py` | `.in` parameter-file reader / `.out` writer |
| `ObitMultiProc`, `ObitThread`, `ObitThreadGrid` | (internal) | gthread parallelism |
| `ObitMem`, `memwatch.h` | — | Memory tracking |
| `ObitFile`, `ObitFileFITS`, `ObitIO*` | (internal) | I/O abstraction |
| `ObitIOImageAIPS`, `ObitIOImageFITS`, `ObitIOUVAIPS`, `ObitIOUVFITS`, `ObitIOTableAIPS`, `ObitIOTableFITS`, `ObitIOHistoryAIPS`, `ObitIOHistoryFITS` | — | AIPS-vs-FITS backends per data type |
| `ObitAIPSDir`, `ObitAIPSCat`, `ObitAIPSFortran`, `ObitAIPSObject` | `AIPSDir.py` | AIPS catalog management |
| `ObitFITS` | `FITSDir.py` | FITS disk directory |
| `ObitBDFData`, `ObitSDMData` | `OASDM.py` | ASDM/BDF (JVLA, ALMA) reader |
| `ObitPCal`, `ObitPolnCalFit`, `ObitPolCalList`, `ObitPolnUnwind` | (internal, drives `PCal`/`PolCal`) | Polarisation calibration solver |
| `ObitGainCal`, `ObitVLAGain` | (internal) | Gain solver |
| `ObitOpacity`, `ObitWeather`, `ObitTsys`, `ObitSwPower`, `ObitEVLASysPower` | (internal) | Atmospheric / system-temperature corrections |
| `ObitWVRCoef`, `ObitALMACalAtm` | (internal) | ALMA atmospheric calibration |
| `ObitSourceEphemerus`, `ObitSource`, `ObitSourceList` | `Source.py` | Source bookkeeping |
| `ObitAntenna`, `ObitAntennaList` | `AntennaList.py` | Array geometry |
| `ObitSurveyUtil`, `ObitTablePSUtil`, `ObitTableMFUtil`, `ObitTableFSUtil`, `ObitTableNXUtil`, `ObitTableSUUtil`, `ObitTableSNUtil`, `ObitTableVLUtil`, `ObitTableNIUtil`, `ObitTableCLUtil`, `ObitTableBPUtil`, `ObitTableFQUtil`, `ObitTableANUtil`, `ObitTableCCUtil`, `ObitTableUtil` | `Catalog.py`, `OSurvey.py` | Survey/catalog utilities |
| `ObitIonCal`, `ObitIoN2SolNTable` | `IonCal.py` | Ionospheric calibration |
| `ObitVecFunc`, `ObitExp`, `ObitSinCos` (with avx_mathfun headers) | (internal) | SIMD/AVX fast math |
| `ObitZernike` | `ZernikeUtil.py` | Zernike polynomials (used in field-based ionospheric cal) |
| `ObitPixHisto` | `PixHistFDR.py` | Pixel histogram + FDR thresholding |
| `ObitPointing` | (internal) | Antenna pointing model |
| `ObitUVWCalc` | (internal) | u,v,w computation |
| `ObitUVRLDelay` | (internal, drives `RLDly`) | R-L delay calibration |
| `ObitVersion` | `Version.inc` | Reports build version |

### 5.5 Python-only modules

Beyond SWIG wrappers, `Obit/python/` provides higher-level helpers:

| Module | Purpose |
|--------|---------|
| `OTObit.py` | POPS-style command set (data factories, `tvlod`, `imhead`) |
| `ObitInit.py`, `OSystem.py`, `OErr.py` | Lifecycle and error stack |
| `ObitTasks.py`, `OTWindow.py`, `OTWindow2.py`, `MsgWin.py`, `TaskWindow.py`, `TaskMsgBuffer.py` | Task / window management |
| `Obit_wrap.c`, `Obit_wrap.doc` | Generated SWIG wrapper |
| `Obit.swig`, `ObitTypeMaps.swig` | SWIG sources |
| `Image.py`, `ImageMF.py`, `ImageDesc.py`, `ImageUtil.py`, `ImageMosaic.py`, `ImageInterp.py`, `ImageFit.py` | Image-side facade |
| `UV.py`, `UVDesc.py`, `UVImager.py`, `UVGSolve.py`, `UVSelfCal.py`, `UVSoln.py`, `UVSoln2Cal.py`, `UVRFIXize.py`, `UVVis.py`, `UVPolnUtil.py` | UV-side facade |
| `Doppler.py`, `Source.py`, `AntennaList.py`, `BeamShape.py`, `FullBeam.py`, `SkyModel.py`, `SkyModelVMBeam.py`, `SkyModelVMIon.py`, `SkyGeom.py` | Physical helpers |
| `Catalog.py`, `OSurvey.py`, `NVSSWebUtil.py`, `TableSTar.py`, `TableUtil.py`, `TableList.py`, `TableDesc.py`, `History.py` | Catalog / table tools |
| `Slice.py`, `Blank.py`, `MakeIFs.py`, `MergeCal.py`, `mjd.py`, `MosaicUtil.py`, `ParserUtil.py`, `PipeUtil.py`, `PipePlots.py`, `ONumpyPlot.py`, `ODisplay.py`, `OPlot.py`, `OPrinter.py`, `OWindow.py`, `OASDM.py`, `IDIFix.py`, `KATH5toAIPS.py`, `MeerKATH5toAIPS.py` | Domain utilities (I/O conversion, plotting, slicing, blanking) |
| Pipelines: `ALMACal.py`, `ALMAPipe.py`, `EVLACal.py`, `EVLAContPipe.py`, `EVLALowBandPipe.py`, `EVLAPipeline.py`, `EVLAPolnScripts.py`, `LinPContPipe.py`, `MeerKATCal.py`, `MeerKATPipeline.py`, `VLACal.py`, `VLBACal.py`, `VLBAContPipe.py`, `VLBAContPipeWrap.py`, `VLBALinePipe.py` | End-to-end calibration & imaging pipelines |
| `ObitDebug.py`, `testObit.py`, `testWeight.py` | Smoke tests |
| `setup.py`, `setupdata.py`, `makesetup.py`, `Makefile.in` | Build glue for the SWIG extension |
| `six.py` | Vendored Py2/3 compat shim |

---

## 6. Tasks — file-by-file

Every task in `simulators/Obit/ObitSystem/Obit/tasks/` has a matching
TDF file in `simulators/Obit/ObitSystem/Obit/TDF/`. The list below
covers all 98 task `.c` files currently shipped (TDF docstrings cited
where consulted).

### Editing & flagging

| Task | File | Description |
|------|------|-------------|
| `AutoFlag` | `tasks/AutoFlag.c` | Automated visibility flagging |
| `MednFlag` | `tasks/MednFlag.c` | Median-filter flagging |
| `OTFFlag` | `tasks/OTFFlag.c` | OTF flagging in UV time-series |
| `Quack` | `tasks/Quack.c` | Edge-of-scan flag |
| `UVFlag` | `tasks/UVFlag.c` | Apply ad-hoc flag selection |
| `LowFRFI` | `tasks/LowFRFI.c` | Low-frequency RFI excision |
| `SrvrEdt` | `tasks/SrvrEdt.c` | Server-side edit |
| `SNFilt` | `tasks/SNFilt.c` | Fit instrumental phases in SN table |

### Calibration solvers

| Task | File | Description |
|------|------|-------------|
| `Calib` | `tasks/Calib.c` | Amp/phase gain calibration |
| `BPass` | `tasks/BPass.c` | Bandpass solver |
| `BPCal` | `tasks/BPCal.c` | Bandpass calibrator handling |
| `CLCal` | `tasks/CLCal.c` | Apply SN solutions to a CL table |
| `CLCor` | `tasks/CLCor.c` | Modify CL tables |
| `SNCor` | `tasks/SNCor.c` | Modify SN tables |
| `SNCpy` | `tasks/SNCpy.c` | Copy SN |
| `SNSmo` | `tasks/SNSmo.c` | Smooth SN tables |
| `GetJy` | `tasks/GetJy.c` | Calibrator flux density bootstrapping |
| `SetJy` | `tasks/SetJy.c` | Modify SU table source flux |
| `SYGain` | `tasks/SYGain.c` | SY-table-based gain |
| `WVRCal` | `tasks/WVRCal.c` | ALMA WVR calibration |
| `SysPowerView` | `tasks/SysPowerView.c` | Inspect EVLA SysPower table |
| `DPCal` | `tasks/DPCal.c` | Differential polarisation calibration |
| `PCal` | `tasks/PCal.c` | Polarisation gain solver |
| `PCCor` | `tasks/PCCor.c` | PC table corrections |
| `XYDly` | `tasks/XYDly.c` | X–Y delay solver (linear feeds) |
| `XYPass` | `tasks/XYPass.c` | X–Y bandpass / phase solver |
| `XYSim` | `tasks/XYSim.c` | Simulate UV with linear feeds |
| `MKXPhase` | `tasks/MKXPhase.c` | MeerKAT cross-hand phase |
| `RLDly` | `tasks/RLDly.c` | R–L delay solver |
| `RLPass` | `tasks/RLPass.c` | R–L bandpass |
| `Lin2Cir` | `tasks/Lin2Cir.c` | Convert linear feeds to circular basis |
| `SwaPol` | `tasks/SwaPol.c` | Swap polarisation labels |

### UV utilities & manipulation

| Task | File | Description |
|------|------|-------------|
| `UVCopy` | `tasks/UVCopy.c` | Copy UV (with selection) |
| `UVAppend` | `tasks/UVAppend.c` | Append two UV files |
| `UVBlAvg` | `tasks/UVBlAvg.c` | Baseline-time averaging |
| `UVFix` | `tasks/UVFix.c` | Fix-up UV header/inflate w |
| `UVShift` | `tasks/UVShift.c` | Shift phase center to a given position |
| `UVSub` | `tasks/UVSub.c` | Subtract sky model from UV |
| `UVSim` | `tasks/UVSim.c` | Generate synthetic UV data from a model |
| `UVPolCor` | `tasks/UVPolCor.c` | UV-domain polarisation correction |
| `UVXpnd` | `tasks/UVXpnd.c` | Expand spectral resolution |
| `Splat` / `Split` / `SplitCh` | `tasks/Splat.c`, `Split.c`, `SplitCh.c` | Single-source / channel split |
| `Hann` | `tasks/Hann.c` | Hanning smooth |
| `AvgBL` | `tasks/AvgBL.c` | Baseline-based average / shift |
| `AvgCh` | `tasks/AvgCh.c` | Average channels |
| `noFQId` | `tasks/noFQId.c` | Strip FQ-id from UV |
| `Lister` | `tasks/Lister.c` | List visibilities |
| `BASMangle` | `tasks/BASMangle.c` | Baseline mangling utility |
| `TabCopy` | `tasks/TabCopy.c` | Copy AIPS table |
| `IDIIn` / `IDIOut` | `tasks/IDIIn.c`, `IDIOut.c` | FITS-IDI ↔ AIPS catalog |
| `ASDMList` / `BDFIn` | `tasks/ASDMList.c`, `BDFIn.c` | List / load ALMA-ASDM, JVLA-BDF data |

### Imaging / deconvolution

| Task | File | Description |
|------|------|-------------|
| `Imager` | `tasks/Imager.c` | Generic continuum imager |
| `MFImage` | `tasks/MFImage.c` | Wide-band ("MF") imager + CLEAN + self-cal (`TDF/MFImage.TDF`) |
| `MFBeam` | `tasks/MFBeam.c` | Wide-band beam-corrected imager (MeerKAT-friendly) |
| `SCMap` | `tasks/SCMap.c` | Self-cal imaging |
| `SWImag` | `tasks/SWImag.c` | Sault-Wieringa wide-band imager |
| `Squint` | `tasks/Squint.c` | Beam-squint corrected imaging |
| `BeamCor` | `tasks/BeamCor.c` | Voltage-beam correction |
| `IonImage` | `tasks/IonImage.c` | Field-based ionospheric imaging |
| `IonMovie` | `tasks/IonMovie.c` | Ionospheric movie |
| `IonSF` | `tasks/IonSF.c` | Iono structure function |
| `IonCal` | `tasks/IonCal.c` | Iono calibration |
| `MapBeam` / `MapBeam2` | `tasks/MapBeam.c`, `MapBeam2.c` | Holography / beam-mapping |
| `ComBeam` | `tasks/ComBeam.c` | Combine beams |
| `Restore` | `tasks/Restore.c` | Restore CLEAN components onto a residual |
| `DFTIm` / `DftPl` | `tasks/DFTIm.c`, `DftPl.c` | DFT imager / DFT-plot diagnostic |
| `Faces` | `tasks/Faces.c` | Faceted image management |
| `FAlign` | `tasks/FAlign.c` | Align facets |
| `HGeom` | `tasks/HGeom.c` | Reproject one image to another's geometry |
| `Feather` | `tasks/Feather.c` | Feather two images of differing resolution |
| `Convol` | `tasks/Convol.c` | Convolve image |
| `MCube` | `tasks/MCube.c` | Construct multi-cube |
| `CubeClip` / `CubeVel` | `tasks/CubeClip.c`, `CubeVel.c` | Cube clipping / velocity |
| `Squish` | `tasks/Squish.c` | Compress 3-D cube |
| `MaskCube` | `tasks/MaskCube.c` | Mask a cube |
| `SubImage` | `tasks/SubImage.c` | Cut a sub-cube |
| `Bloat` | `tasks/Bloat.c` | "Bloat" image (resize) |
| `ImPlot` | `tasks/ImPlot.c` | Image contour/grayscale plot |
| `CVel` | `tasks/CVel.c` | Convert velocity reference |

### Spectral / Faraday

| Task | File | Description |
|------|------|-------------|
| `RMSyn` | `tasks/RMSyn.c` | RM synthesis (efficiency-upgraded in `3c22d593`) |
| `Farad` | `tasks/Farad.c` | Faraday analysis (uses `ObitFaraSyn`) |

### Polarisation calibration / imaging

| Task | File | Description |
|------|------|-------------|
| `MazrCal` | `tasks/MazrCal.c` | Mazr poln calibration |
| `VLPoln` | `tasks/VLPoln.c` | VLA polarisation processing |
| `VL2FS` / `VL2VZ` | `tasks/VL2FS.c`, `VL2VZ.c` | VL-format catalog conversions |
| `VLSSFix` | `tasks/VLSSFix.c` | Fix VLSS ionospheric residual geometry |
| `FndSou` | `tasks/FndSou.c` | Source finding |
| `Ringer` | `tasks/Ringer.c` | Ring-source treatment |
| `SubImage` | (above) | (re-use) |

### Misc / tooling

| Task | File | Description |
|------|------|-------------|
| `FuncContainer` | `tasks/FuncContainer.c` | Generic function-call container |
| `Template` | `tasks/Template.c` | Skeleton/example task |

(96 TDF files cover those 98 tasks; some auxiliary docs `Imager.doc`,
`SubImage.doc`, `Squish.doc`, etc. live alongside.)

### TDF format

A TDF file is the parameter-definition source consumed both by
`ObitParser` (in the C task) and by `ObitTask`/`AIPSTask` (in
ObitTalk). For example `simulators/Obit/ObitSystem/Obit/TDF/UVSim.TDF`:

```
UVSim     LLLLLLLLLLLLUUUUUUUUUUUU CCCCCCCCCCCCCCCCCCCCCCCCCCCCC
UVSim     Task to generate simulates UV data
**PARAM** str 4
DataType                         "FITS" or "AIPS" type
**PARAM** str 10  **DEF** 2000-01-01
refDate                          reference day YYYY-MM-DD
**PARAM** float 2
timeRange                        Time range to add.
**PARAM** float 1 **DEF** 1.0
delTime                          Time increment (sec)
…
```

Field tokens are `**PARAM**` (type and shape), `**DEF**` (default), an
adverb name, optional numeric range, and a help string.

---

## 7. Core capabilities

The list below summarises capabilities documented in source comments,
TDF help, and `OBITdoc.tex`:

- **General radio interferometric calibration** — Calib + CLCal +
  BPass + PCal + RLPass + RLDly + XYPass + XYDly + Jones-matrix tables
  (JI/JT). Polarisation calibration supports both circular and linear
  feeds, with a `keepLin` flag (commit `8235fedd`) to preserve the
  linear basis.
- **Ionospheric calibration** — `IonCal`, `IonImage`, `IonMovie`,
  `IonSF`, Zernike-polynomial correction, `ObitTableNI`/`NIUtil`.
- **CLEAN deconvolution** — Hogbom (`ObitDConCleanImage`),
  Cotton-Schwab (`ObitDConCleanVis`), multi-frequency
  (`ObitDConCleanVisMF`), Sault-Wieringa wide-band
  (`ObitDConCleanVisWB`), line (`ObitDConCleanVisLine`), with
  per-component (`PxList[/MF/WB]`), pixel-histogram (`PxHist`), and
  beam-history (`BmHist`) classes.
- **Multi-frequency synthesis (MFS)** — `ObitImageMF`, `ObitImageMosaicMF`,
  `ObitUVImagerMF`, `ObitDConCleanVisMF`, `ObitSkyModelMF`, with
  effective-frequency tracking on subbands (commit `c4e72dad`).
- **Wide-band imaging** — Sault-Wieringa pipeline
  (`ObitImageMosaicWB`, `ObitUVImagerWB`, `ObitDConCleanVisWB`,
  `ObitImageWB`).
- **Self-calibration** — `ObitUVSelfCal`, `ObitUVGSolve`, `SCMap`
  task; Imager / MFImage do internal self-cal loops.
- **Peeling** — `ObitUVPeelUtil`, `python/PeelWork.py`,
  `python/PeelScripts.py` (subtract bright sources, solve, image
  residuals).
- **Beam handling** — analytic primary beams (`ObitBeamShape`:
  Gaussian, generalized cos², UV-derived), full-Stokes voltage beams
  (`ObitFullBeam`), VLA/EVLA squint, MeerKAT cosine² beams
  (`Cos2Beam.py`, `MKCosBeam.py`).
- **Holography** — `MapBeam`/`MapBeam2` tasks (use
  `ObitSkyModelVMBeam` and BDF holography hack; see commit
  `eeadb6e7`).
- **Sky models** — point-source CC tables, CC-grid, multi-frequency,
  variable-model with beam-corrected (`VMBeam[MF]`), squint (`VMSquint`),
  ionospheric (`VMIon`), polarisation (`VMPoln`).
- **Mosaicing** — `ObitImageMosaic[/MF/WB]`, `MosaicUtil.py`,
  `share/scripts/Mosaic.py`, `mosaicBasic.py`, `CheckPos.py`.
- **Faraday / RM analysis** — `ObitFaraSyn`, `ObitRMFit`, `RMSyn` /
  `Farad` tasks, `FitFarRot.py`, `MKRMFit.py`, `MKPlotRM.py`.
- **Spectral fitting** — `ObitSpectrumFit`, `ObitSpectrumInterp`,
  `ObitSpectrumMF`, `ImageMFFitSpec` API.
- **Source finding & cataloguing** — `ObitSurveyUtil`,
  `ObitTableVL`/`VLUtil`, NVSS/SUMSS/MGPS VZ catalogs in
  `share/data`, `FndSou` task, `Catalog.py`, `OSurvey.py`,
  `NVSSWebUtil.py`.
- **Image fitting** — `ObitImageFit`, `ObitFitModel`, `ObitFitRegion`,
  `share/scripts/CheckPos.py`, `Targ.py`.
- **Image arithmetic / utility** — `Convol`, `Feather`, `HGeom`,
  `Restore`, `SubImage`, `Bloat`, `MaskCube`, `CubeClip`, `CubeVel`,
  `Squish`, plus python `Slice.py`, `Blank.py`, `MakeIFs.py`.
- **Plotting** — `ObitPlot` (PLPLOT/PGPLOT), Matplotlib alternatives
  (`MK_Tools.py`, `ONumpyPlot.py`, `PipePlots.py`).
- **GPU acceleration** — Optional CUDA paths in
  `ObitCUDAGrid`/`ObitCUDASkyModel`/`ObitCUDAUtil`/`CUDAFArray`/`CUDAFInterpolate`,
  driven through `ObitGPUGrid`/`ObitGPUSkyModel`/`ObitGPUFArray`. The
  `Obit.h` macro `ObitHaveGPU` detects compile-in support.
- **Threading** — `ObitMultiProc`, `ObitThread`, `ObitThreadGrid`,
  `OBIT_THREADS_ENABLED` set when `gthread-2.0` is detected.
- **AVX / AVX2 / AVX-512 SIMD** — `ObitVecFunc.c`, `ObitExp.c`,
  `ObitSinCos.c` (with `sincos2` fallback after compiler bug),
  `include/avx_mathfun.h`, `avx2_mathfun.h`, `avx512_mathfun.h`.
- **Pipelines** — End-to-end Python recipes for EVLA continuum &
  low-band, ALMA, VLBA, LinPCont, MeerKAT (see `python/*Pipe.py`,
  `share/scripts/*TemplateParm.py`, `MeerKATPipeline.py`).

---

## 8. Input and output formats

### Visibility (UV) data

| Format | Source | Notes |
|--------|--------|-------|
| **AIPS catalog UV** | `ObitIOUVAIPS.c`, `ObitAIPSCat.c`, `ObitAIPSDir.c` | Native AIPS disk format with `MA`/`UV` files; addressed by `(name, class, seq, disk)` |
| **FITS UV (FITS-IDI / random-groups)** | `ObitIOUVFITS.c` | Columnar (FITS-IDI) and random-groups |
| **ALMA ASDM / BDF** | `ObitSDMData.c`, `ObitBDFData.c`, `tasks/BDFIn.c`, `ASDMList.c`, `python/OASDM.py` | Read JVLA/ALMA archive |
| **MeerKAT KATH5** | `python/KATH5toAIPS.py`, `MeerKATH5toAIPS.py` | Convert to AIPS UV |

### Image data

| Format | Source | Notes |
|--------|--------|-------|
| **AIPS catalog image** | `ObitIOImageAIPS.c` | Native AIPS catalog |
| **FITS image** | `ObitIOImageFITS.c` | Multi-extension; magic-blanking via `OBIT_MAGIC` |
| **MFImage cube** | `ObitImageMF.c` | Sub-band planes with frequency assignments / break points |

### OTF (single-dish)

| Format | Source | Notes |
|--------|--------|-------|
| **OTF FITS** | `ObitSD/src/ObitIOOTFFITS.c` | Time-ordered single-dish data |

### Tables

All AIPS-style tables (49 in `Obit/include/`) plus the FITS-IDI tables
are first-class objects: `AN`, `AT`, `BL`, `BP`, `CC`, `CD`, `CL`,
`CP`, `CQ`, `CT`, `FG`, `FQ`, `FS`, `GC`, `IM`, `JI`, `JT`, `MC`, `MF`,
`NI`, `NX`, `OB`, `OF`, `OT`, `PC`, `PD`, `PO`, `PS`, `PT`, `SN`,
`SU`, `SY`, `TY`, `VL`, `VZ`, `WX`, plus `IDI_*`. Each has a Python
`.inc` SWIG file in `Obit/python/`.

### Provided sky-model data

`Obit/share/data/` ships ready-made calibrator FITS models for AIPS
calibration: 0408-65, 1934-638 (L, UHF), 3C48 (C/L/Ka/K/P/Q/S/X),
3C123 (C/L/P/S), 3C138 (C/Ka/K/L/P/Q/S/X), 3C147 (C/L/P/S/X), 3C196
(C/L/P/S/X), 3C286 (full L/C/X plus per-band CalModel), 3C295
(C/L/P/S/X), 3C380 (C/L/P/S/X), Cygnus-A 74 MHz; CSV
`PERLEY_BUTLER_2013.csv`; "VZ" tables for AllSky, NVSS, MGPS, SUMSS;
KAT-7 / MeerKAT UV templates; MeerKAT polarisation calibration tables
(`MKMedAvg*.PolCalTab.uvtab.gz`).

---

## 9. Testing & examples

### C test programs (`Obit/test/`)

Each is a stand-alone executable built by `make testupdate`:

| File | What it exercises |
|------|-------------------|
| `test.c`, `test2.c`, `test3.c` | Smoke-tests for the core (`ObitInfoList`, `ObitFArray`) |
| `testCleanVis.c` | UV-domain CLEAN |
| `testClean.c` | Image-domain Hogbom |
| `testCCMerge.c` | CC table merging |
| `testFeather.c` | Feather operator |
| `testHGeom.c` | Reprojection |
| `testPre.c` | Precession |
| `testSelfCal.c` | Self-calibration |
| `testUVImage.c` | UV→image roundtrip |
| `testUVSub.c` | UV subtraction |
| `AIPS2FITS.c` | AIPS↔FITS conversion |
| `dump.c`, `doofus.c` | Header dumpers / scratch |

### Python smoke tests (`Obit/testScripts/`)

`testCleanVis.py`, `testContourPlot.py`, `testFeather.py`,
`testHGeom.py`, `testObit.py`, `testUVImage.py`, `testUVSub.py`. Run
via `pixi`-style scripts or directly through ObitTalk.

### ObitTalk regression tests (`ObitTalk/test/`)

`imean.py` (run AIPS IMEAN), `jmfit.py` (run AIPS JMFIT), `mandl.py`
(Mandelbrot-style FITS exercise), `template.py`, plus `ObitTest.in`.

### Worked examples

`Obit/share/scripts/` ships a substantial library of recipes:

```
Mosaic.py / mosaicBasic.py / CheckPos.py / Targ.py / PointingList.py   # Mosaicking
MK_PolCal_XY.py / MKPolCal.py / MK_PolnPlot.py                          # MeerKAT polarization
MK_Tools.py / MKCosBeam.py / Cos2Beam.py                                # MeerKAT imaging tools
MKPlotRM.py / MKRMFit.py / MKPlotSpec.py                                # MeerKAT RM/spectrum
EVLAContTemplateParm.py / EVLATemplateParm.py / EVLALowBandTemplateParm.py
ALMATemplateParm.py / VLBAContTemplateParm.py / MKTemplateParm.py
HugoPerley_3C286.py / FixEVPA.py                                        # Calibrator models
CleanWindowEdit.py / RMClip.py / BlankImage.py / ConcatCC.py
scriptFeather.py / scriptHGeom.py / scriptUVImage.py / scriptMakeCube.py
scriptMakeMFCube.py / scriptPBApply.py / scriptPBUndo.py / scriptQuantize.py
scriptDifImage.py / scriptCopyVLTab.py / scriptFFT.py / scriptFixVLSS.py
scriptFetchNVSS.py / scriptHi2Header.py / scriptBlowup.py / scriptAvgImage.py
scriptVPolCal.py / GetFluxDensity.py / FitFarRot.py / FitSpec.py / FitError.py
AvgPolInt.py / AvgWeights.py / CombineBeam.py / CopyHistory.py
CopyMFPlane.py / Descimate.py / CLEANMask.py / SumImage.py / SpectrumUtils.py
PangleOTF.py / PBCorImageMF.py / PointingList.py / debug.py
AIPSSetup.py / obitrc.py / logging.conf / getVersion.py
```

`testFuncCont.c` is the matching C-side example for `FuncContainer`.

### Documentation

- `Obit/doc/OBITdoc.tex` — 6 776-line reference (LaTeX) covering the
  data model, Jones tables, calibration math, and per-task adverbs.
- `Obit/doc/Obit.html` — HTML mini-page.
- `Obit/doc/doxygen/` — auto-generated class browser.
- `ObitTalk/doc/ObitTalk.tex`, `ObitTalk.pdf`, `cookbook.xml`,
  `Report.bib` — ObitTalk user guide & cookbook.
- `simulators/Obit/README` — top-level install README.
- `migrateObit.text` — recipe for moving an svn install to GitHub.
- `changes.log` — 5 328-line developer journal in `Obit/`.

---

## 10. Integration & extension points

### Adding a new task

1. Drop a new `tasks/MyTask.c` (use `tasks/Template.c` as a skeleton —
   reads inputs via `MyTaskIn(argc, argv, err)` then calls into
   `libObit`).
2. Drop a matching `TDF/MyTask.TDF` defining adverbs.
3. Add `MyTask` to `tasks/Makefile.in`'s task list.
4. After build, ObitTalk picks it up automatically via
   `ObitTask("MyTask")` (which reads the TDF file at task-construction
   time).

### Adding a new C class

Follow the triple-header convention:

```
include/ObitFoo.h         # public prototypes (use Obit/include/ObitTEMPLATE-style)
include/ObitFooDef.h      # struct fields
include/ObitFooClassDef.h # class function-pointer table
src/ObitFoo.c             # implementation
```

Then expose it to Python:

```
python/Foo.inc            # SWIG fragment
python/Foo.py             # high-level facade
python/Obit.swig          # add %include "Foo.inc"
```

`bin/ObitTables.pl` regenerates the `Table*` triplets when AIPS table
formats change (`Makefile :: tableupdate`).

### Adding an AIPS table

Edit the relevant section of `OBITdoc.tex` (which doubles as the table
specification), then `make tableupdate`. The Perl tool reads the doc
tables and emits `ObitTableXX.{h,Def.h,ClassDef.h,c}` plus the
matching `TableXX.inc` SWIG fragment. JI / JT (Jones) tables were
introduced this way in commit `e1d84da2`.

### Calling Obit from external Python

Any external Python program with `PYTHONPATH` set to
`$OBIT/python:$OBITINSTALL/opt/share/obittalk/python` can:

```python
import OErr, OSystem, UV, Image, ObitTask
err = OErr.OErr()
ObitSys = OSystem.OSystem("MyApp", 1, 0, 0, [], 0, [], True, False, err)
uv = UV.newPAUV("name", "uvdata", 1, 1, True, err)
mf = ObitTask.ObitTask("MFImage")
mf.inFile = "src.uvfits"; mf.go()
```

### Remote execution

`ObitTalkServer` (Python XML-RPC daemon) lets a workstation run Obit
or AIPS tasks on a remote compute node (`ObitTalk/python/Proxy/*`).
ObitView can also be controlled by ObitTalk over XML-RPC via
`ObitDisplay` (`src/ObitDisplay.c`,
`ObitView/src/XMLRPCserver.c`).

---

## 11. Notable internals

- **InfoList everywhere.** Almost every public function in `libObit`
  takes an `ObitInfoList*` carrying its adverbs. The python
  `InfoList.py` plus `Info2List` / `List2Info` (in `InfoList.inc`)
  marshal Python dicts in/out of `ObitInfoList`.
- **`ObitErr` is the universal status carrier** (`OErr.py`); every
  routine that can fail takes one.
- **AIPS-style class IDs.** First `gint32` of each object is a class
  ID stamp (`Obit.h:60`). Allows runtime type checks without RTTI.
- **`memwatch.h`** (vendored) tracks heap to debug leaks
  (`Obit.h:32`).
- **Threading is opt-in.** `OBIT_THREADS_ENABLED` is set at configure
  time only when `gthread-2.0` is available; otherwise
  `ObitMultiProc.c` falls back to single-threaded paths
  (`#ifdef OBIT_THREADS_ENABLED` guards in each kernel).
- **CFITSIO and xmlrpc-c are *soft* dependencies** thanks to
  `dummy_cfitsio/` and `dummy_xmlrpc/` stubs (links at build time so
  the binary still runs when those subsystems are unused).
- **SWIG output is checked in.** `python/Obit_wrap.c` is generated by
  SWIG 4.0.1 but committed (`Obit_wrap.c` line 1) so a developer
  doesn't need SWIG to build.
- **AVX / AVX-512 paths.** `ObitFArray.c::Sum` and
  `ObitSkyModelVMBeamMF.c::SkyModelVMul` carry both fast-AVX and
  scalar implementations; some AVX-512 paths are currently disabled
  due to "mysterious segv" (commit `0b10f527`, on `ObitExp.c`).
- **`sincos` workaround.** `ObitSinCos.c` defines `sincos2` /
  `sincos2f` because old `gcc` builds disagreed about the
  `glibc`-vs-Apple `sincos` extension; the affected tasks (`AvgBL`,
  `RMSyn`, `UVShift`, `XYDly`, `Farad`) were retro-fitted across
  several commits (`42ace5df`, `5ce526d5`, `c4e72dad`).
- **Interactive display revival.** `ObitDisplay.c` allows a task to
  re-enable the X11 ObitView display by `touch`-ing
  `/tmp/<TaskName>.doTV` (see `changes.log:18-20`).
- **GPU support detection.** `Obit.h` defines `ObitHaveGPU()`
  (`ObitParser.c` honours it); a task asking for `doGPUGrid=True`
  silently falls back if CUDA was not compiled in.
- **Multiple imager / cleaner classes.** `2D` and `3D` modes coexist:
  `PickNext2D` / `PickNext3D` selectors in `ObitDConCleanVis.c`.
- **AIPS dir caching.** `ObitAIPSDir.c::DirAlloc` updates time tags
  even if a file already exists (the file is being rewritten, commit
  `c4e72dad`).

---

## 12. Known limitations & TODOs

- **`ObitSD` is marked defunct** in the installer (`InstallObit.sh:23`).
  The source tree is shipped but neither built nor exercised by the
  default workflow; AUTHORS / scripts in `ObitSD/scripts/` reference an
  early GBT campaign rather than current operations.
- **`ObitTalk/TODO`** ships an empty / minimal TODO; documentation
  rebuild requires LaTeX (`README:181-184`).
- **`doWWW=no`** is forced off — the bundled `w3c-libwww-5.4.0.tgz` no
  longer builds reliably.
- **64-bit & Ubuntu caveats** in `README:133-179` describe
  build-time problems with the vendored `glib`, `motif`, and `zlib`
  tarballs and recommend the system packages instead.
- **AVX-512 paths** are intentionally disabled in
  `ObitSkyModelVMBeamMF.c::SkyModelVMul` and `ObitExp.c::ExpVec`
  ("mysterious segv", `0b10f527`).
- **Old `gcc` `sincos` clash** prompted the `sincos2` workaround
  (`include/sincos.h`).
- **`ObitView` documentation** still references the historical
  `XFITSview` name (`ObitView/README:36`).
- **No semantic version tags** — only `ObitVersion` (`1.1.670`)
  computed from git revision count.
- **Python 2 legacy** — `ObitTalk` (`bin/ObitTalk.in`) is the Python 2
  driver; `ObitTalk3.in` is the modern Python 3 alternative. Some
  pipeline scripts still hand-import `six.moves` for compatibility.
- **Documentation cookbook** is in DocBook XML
  (`ObitTalk/doc/cookbook.xml`); the rendered PDF has not been
  refreshed in step with all recent task additions.
- **`changes.log`** is hand-maintained and currently 5 328 lines —
  authoritative but unstructured.

---

## 13. Quick reference — selected file paths

```
simulators/Obit/InstallObit.sh
simulators/Obit/ObitSystem/ObitVersion                                        # 1.1.670
simulators/Obit/ObitSystem/Obit/Makefile                                      # top build
simulators/Obit/ObitSystem/Obit/configure.ac                                  # autoconf
simulators/Obit/ObitSystem/Obit/include/Obit.h                                # base class
simulators/Obit/ObitSystem/Obit/include/ObitAll.h                             # umbrella include
simulators/Obit/ObitSystem/Obit/src/ObitSystem.c                              # global init
simulators/Obit/ObitSystem/Obit/src/ObitUV.c                                  # visibility class
simulators/Obit/ObitSystem/Obit/src/ObitImage.c                               # image class
simulators/Obit/ObitSystem/Obit/src/ObitImageMF.c                             # MFImage class
simulators/Obit/ObitSystem/Obit/src/ObitDConCleanVisMF.c                      # MFS CLEAN
simulators/Obit/ObitSystem/Obit/src/ObitFaraSyn.c                             # Faraday synth
simulators/Obit/ObitSystem/Obit/src/ObitSkyModelVMBeamMF.c                    # MFS DDE sky model
simulators/Obit/ObitSystem/Obit/src/ObitCUDAGrid.cu                           # CUDA gridder
simulators/Obit/ObitSystem/Obit/tasks/MFImage.c                               # wide-band imaging task
simulators/Obit/ObitSystem/Obit/tasks/UVSim.c                                 # UV simulator
simulators/Obit/ObitSystem/Obit/TDF/MFImage.TDF                               # adverb file
simulators/Obit/ObitSystem/Obit/python/OTObit.py                              # ObitTalk shell prelude
simulators/Obit/ObitSystem/Obit/python/UV.py                                  # UV facade
simulators/Obit/ObitSystem/Obit/python/MeerKATPipeline.py                     # MeerKAT pipeline
simulators/Obit/ObitSystem/Obit/python/EVLAContPipe.py                        # EVLA pipeline
simulators/Obit/ObitSystem/Obit/share/data/3C286LBand.CalModel.fits.gz        # bundled cal model
simulators/Obit/ObitSystem/ObitTalk/python/ObitTalk.py                        # interactive shell
simulators/Obit/ObitSystem/ObitTalk/python/ObitTask.py                        # task driver
simulators/Obit/ObitSystem/ObitTalk/python/Proxy/ObitTask.py                  # remote-task XML-RPC
simulators/Obit/ObitSystem/ObitView/ObitView.c                                # XFITSview
simulators/Obit/ObitSystem/ObitView/ObitMess.c                                # task-message GUI
simulators/Obit/other/Install3rdParty.sh                                      # 3rd-party builder
simulators/Obit/other/tarballs/                                               # 18 vendored tarballs
```

---

*Generated from a fresh inspection of the `simulators/Obit` submodule
(commit `e46d4b4e`). All section claims are grounded in the files
cited above; any quoted strings come directly from those sources.*
