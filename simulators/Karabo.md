# Karabo-Pipeline — Exhaustive Technical Reference

> Location of source under inspection: `simulators/Karabo/`
> Upstream: <https://github.com/i4Ds/Karabo-Pipeline>
> Documentation: <https://i4ds.github.io/Karabo-Pipeline/>
> All paths in this document are relative to `simulators/Karabo/` unless otherwise noted.

---

## 1. Overview

**Karabo** (a.k.a. *Karabo-Pipeline*) is a Python-only **radio-astronomy software distribution** developed by the FHNW Institute for Data Science (i4Ds, Windisch, Switzerland) and positioned by the SKA Observatory community as a **starting point for the SKA Digital Twin pipeline**. Its self-described goal (`README.md` line 13) is "validation and benchmarking of radio telescopes and algorithms" — it can simulate the SKA and ~20 other supported telescopes, run imaging, perform source detection, and evaluate the results.

Karabo is *not* a primary engine; it is a high-level orchestration layer that wraps and re-exposes a battery of established radio-astronomy tools through a uniform Python (and Jupyter-friendly) API:

| External engine                | Role inside Karabo                                                |
|--------------------------------|--------------------------------------------------------------------|
| **OSKAR** (`oskarpy=2.8.3`)    | Default interferometer simulation backend; visibility I/O          |
| **RASCIL** (`rascil=1.0.0`)    | Alternate simulator (DFT) and primary continuum imaging / cleaning |
| **WSClean**                    | CLI-driven dirty / clean imaging                                   |
| **PyBDSF** (`bdsf=1.10.2`)     | Source-detection (Gaussian fitting)                                |
| **ska-sdp-datamodels / -func-python** | Visibility & image data models + SKA SDP processing functions |
| **casacore / python-casacore** | Measurement-Set I/O                                                |
| **healpy / pyradiosky** (impl.) | HEALPix sky utilities                                              |
| **Eidos** (`eidos=1.1.0`)      | Polarised primary-beam generation                                  |
| **Bluebild** (`bluebild=0.1.0`)| Alternative imaging                                                |
| **katbeam** (`katbeam=0.1.0`)  | MeerKAT-style beam evaluation                                      |
| **ARatmospy** (`aratmospy=1.0.0`) | Ionospheric TEC-screen FITS generation                          |
| **tools21cm** (`tools21cm=2.0.3`) | 21-cm signal cubes (EoR, lightcones)                            |
| **pyuvdata\[casa\]**           | MWA / UVFITS / HDF5 visibility I/O                                 |
| **ska-gridder-nifty-cuda**     | CUDA-accelerated NIFTY gridder                                     |
| **dask / dask-mpi / mpich / mpi4py** | Local + cluster + MPI parallelism                            |
| **MIGHTEE / GLEAM / MALS DR1V3 / Haslam408 / HI** (datasets) | Bundled / on-demand surveys |

The codebase is *exclusively Python* (no C/C++/Fortran sources are shipped — those live in the wrapped libraries). Telescope-model `*.tm` directory bundles ship as data assets.

### 1.1 Project metadata (cited)

| Field          | Value | Source |
|----------------|-------|--------|
| Package name   | `Karabo-Pipeline` | `pyproject.toml` line 2 |
| Description    | "A data-driven pipeline for Radio Astronomy from i4ds for the SKA Telescope." | `pyproject.toml` line 3 |
| Author         | Simon Felix `<simon.felix@fhnw.ch>` (FHNW) | `pyproject.toml` line 5; `LICENSE` line 3 ("Copyright (c) 2022 Simon Felix") |
| Latest tag     | `v0.34.0` (Sphinx-built `CITATION.cff`); newest commit `ca7f333` "Update git clone URL to use HTTPS" | `git log` |
| Released tags (selected) | v0.2.0 → v0.34.0 (50+ tags) | `git tag` |
| Python         | `>=3.9`, currently constrained `<3.11` in conda env | `pyproject.toml` line 9; `environment.yaml` line 12 |
| License        | MIT (note: `pyproject.toml` and `LICENSE` say MIT; the README also states MIT — but a stale "BSD 3-Clause" badge is rendered from `anaconda.org/i4ds/karabo-pipeline/badges/license.svg` in `README.md` line 7. The authoritative file `LICENSE` is MIT.) |
| Versioning     | `versioneer` PEP 440 from git tags (`tag_prefix = "v"`) | `pyproject.toml` lines 33–38; `karabo/_version.py` (auto-generated) |
| Status classifier | "Development Status :: 3 - Alpha" | `pyproject.toml` line 12 |
| Anaconda channel | `i4ds` (also `nvidia/label/cuda-11.7.0`, `conda-forge`) | `environment.yaml` lines 1–4 |
| Maintainer (conda recipe) | `lukas.gehrig@fhnw.ch` | `conda/meta.yaml` line 89 |

### 1.2 Citation

`CITATION.cff` lines 1–27 declare the canonical citation:

> University of Applied Sciences Northwestern Switzerland (FHNW), Windisch, CH-5210; software, MIT, version `v0.34.0`, repository <https://github.com/i4Ds/Karabo-Pipeline>.

### 1.3 Contributors

`CONTRIBUTORS.md` lists committers all from FHNW: Simon Felix (technical PM), Christoph Vögele, Lukas Gehrig, Rohit Sharma, Filip Schramka, Vincenzo Timmel.

---

## 2. Repository layout

Generated from `find . -type f -not -path './.git/*' | wc -l` — **~20 274 files** (the bulk of which are the 225 telescope-model `.tm` directories and their per-station `layout.txt` files). Top-level tree:

```
simulators/Karabo/
├── README.md                  # marketing + quickstart
├── LICENSE                    # MIT (Simon Felix, 2022)
├── CITATION.cff               # software citation (v0.34.0)
├── CONTRIBUTORS.md            # committer roster
├── pyproject.toml             # build system, deps, versioneer
├── setup.py                   # versioneer thin wrapper
├── setup.cfg                  # mypy/flake8/isort/black/pydocstyle/coverage
├── MANIFEST.in
├── modules.rst                # Sphinx index hook
├── codecov.yml
├── Makefile                   # Sphinx documentation build (`make html`)
├── Dockerfile                 # CUDA 11.7.1 + miniconda + karabo install
├── environment.yaml           # full conda dep pin
├── conda/
│   ├── meta.yaml              # conda-build recipe
│   ├── build.sh               # `pip install --no-deps .`
│   └── conda_build_config.yaml
├── doc/src/                   # Sphinx documentation source
│   ├── conf.py
│   ├── index.rst
│   ├── installation_user.md
│   ├── container.md
│   ├── parallel_processing.md
│   ├── development.md
│   ├── main_features/
│   │   ├── simulation.rst
│   │   ├── imaging.rst
│   │   ├── base_imaging.rst
│   │   ├── oskar_imaging.rst
│   │   ├── rascil_imaging.rst
│   │   ├── wsclean_imaging.rst
│   │   └── sourcedetection.rst
│   ├── examples/
│   │   ├── examples.md
│   │   ├── example_structure.md
│   │   ├── combine_examples.py
│   │   └── _example_scripts/
│   │       ├── example_tel_set.py
│   │       └── example_interfe_simu.py
│   └── utilities/utils.rst
├── k8s/SRCNet_SKA-MID_AAstar/  # Kubernetes job + cronjob YAMLs (SRCNet)
│   ├── job.yaml
│   ├── cronjob.yaml
│   ├── tmp-pvc.yaml
│   ├── README.md
│   └── data-ingestor/{values.yaml,secrets.yaml}
└── karabo/                    # Python package (PEP 420 layout)
    ├── __init__.py            # versioneer hook + WSL LD path patch + SLURM dask
    ├── _version.py            # versioneer auto-generated
    ├── error.py               # Exception hierarchy
    ├── warning.py             # Warning hierarchy & _DEV_ERROR_MSG
    ├── karabo_resource.py     # NumpyHandleError, HiddenPrints, CaptureSpam
    ├── simulator_backend.py   # SimulatorBackend = OSKAR | RASCIL enum
    ├── simulation/            # high-level science API
    │   ├── telescope.py       # Telescope class + .tm I/O
    │   ├── telescope_versions.py # 17 enums for ALMA/ATCA/SKA versions
    │   ├── station.py         # Station class
    │   ├── east_north_coordinate.py
    │   ├── coordinate_helper.py # WGS84 ↔ ECEF
    │   ├── observation.py     # Observation / ObservationLong / ObservationParallelized
    │   ├── interferometer.py  # InterferometerSimulation (OSKAR + RASCIL backends)
    │   ├── visibility.py      # Visibility + VisibilityFormat helpers
    │   ├── sky_model.py       # SkyModel (xarray-backed) + SkyPrefixMapping/SkySourcesUnits
    │   ├── beam.py            # Gaussian / Airy / Eidos primary-beams
    │   ├── line_emission.py   # line-emission pipeline
    │   ├── line_emission_helpers.py
    │   ├── sample_simulation.py # canonical end-to-end smoke run
    │   └── signal/            # 21-cm / EoR / synchrotron / superpixel signals
    │       ├── base_signal.py / base_segmentation.py
    │       ├── signal_21_cm.py / eor_profile.py
    │       ├── synchroton_signal.py / galactic_foreground.py
    │       ├── superimpose.py / seg_u_net_segmentation.py / superpixel_segmentation.py
    │       ├── helpers.py / typing.py / plotting.py
    ├── imaging/
    │   ├── image.py           # Image class (~973 LOC) + ImageMosaicker
    │   ├── imager_base.py     # DirtyImager, ImageCleaner ABCs
    │   ├── imager_oskar.py    # OskarDirtyImager
    │   ├── imager_rascil.py   # RascilDirtyImager + RascilImageCleaner
    │   ├── imager_wsclean.py  # WscleanDirtyImager + WscleanImageCleaner
    │   └── util.py            # MGCLS download, beam guess, project_sky_to_image
    ├── sourcedetection/
    │   ├── result.py          # PyBDSF / abstract SourceDetectionResult(s)
    │   └── evaluation.py      # truth-prediction assignment + metrics
    ├── data/
    │   ├── external_data.py   # CSCS S3 download infrastructure
    │   ├── casa.py            # MS subtable readers (Antenna, Polarization, …)
    │   ├── obscore.py         # IVOA ObsCore metadata model (~1151 LOC)
    │   ├── src.py             # SKA SRC / Rucio metadata
    │   ├── *.tm/              # 225 OSKAR telescope-model bundles
    │   │     (ACA, ALMA cycles 0–11, ATCA configs, CARMA, NGVLA, PDBI,
    │   │      SKA-LOW-AA0.5/AA1/AA2/AA4/AAstar, SKA-MID-AA*, SMA, VLA, etc.)
    │   ├── lofar.tm  meerkat.tm  mkatplus.tm  ska1low.tm  ska1mid.tm
    │   ├── askap.tm  WSRT.tm  vlba.tm  telescope.tm  (EXAMPLE)
    │   └── _add_oskar_alma_layouts/, _add_oskar_ska_layouts/
    ├── util/
    │   ├── _types.py          # TypeAlias zoo (NPFloatLike, BeamType, PrecisionType)
    │   ├── data_util.py       # path / string utils (~307 LOC)
    │   ├── file_handler.py    # FileHandler short/long-term cache (~513 LOC)
    │   ├── dask.py            # DaskHandler{Basic,Slurm} (~749 LOC)
    │   ├── gpu_util.py        # is_cuda_available / get_gpu_memory via nvidia-smi
    │   ├── math_util.py       # poisson-disk sampling, Voigt, etc.
    │   ├── hdf5_util.py       # healpix file loading
    │   ├── jupyter.py         # Jupyter helpers
    │   ├── plotting_util.py
    │   ├── helpers.py         # Environment, get_rnd_str, ...
    │   ├── rascil_util.py     # filter_data_dir_warning_message
    │   ├── config_util.py
    │   ├── survey.py
    │   ├── testing.py
    │   └── ska_sdp_datamodels/__init__.py # vendored shim
    ├── examples/              # Jupyter notebooks + Python scripts
    │   ├── Sky_Simulation.ipynb / imaging.ipynb / source_detection*.ipynb
    │   ├── ImageMosaicker.ipynb / signal_simulation_segmentation.ipynb
    │   ├── LineEmissionSimulation_RASCIL.ipynb
    │   ├── SRCNet_simulation_walkthrough.ipynb
    │   ├── SRCNet_v0.1_simulation_1_MeerKAT.py / 2_AAstar.py
    │   ├── SRCNet_rucio_meta.py
    │   └── data/point_sources_OSKAR1_diluted5000.h5
    ├── workflows/
    │   ├── __init__.py
    │   └── SRCNet_SKA-MID_AAstar.py
    ├── performance_test/
    │   ├── time_karabo.py / time_karabo_reconstruction.py
    │   ├── time_karabo_parallelization_by_channel.py
    │   ├── time_karabo_slurm_h5.py / sbatch_script.sh
    │   └── paper/  (benchmark scripts and reporting for the i4ds paper)
    └── test/                  # pytest suite (~30 modules)
        ├── conftest.py / util.py
        ├── test_telescope.py / test_skymodel.py / test_observation.py / test_simulation.py
        ├── test_imager_oskar.py / test_imager_rascil.py / test_imager_wsclean.py
        ├── test_image.py / test_imaging_util.py / test_mosaic.py
        ├── test_source_detection.py / test_obscore.py / test_casa.py
        ├── test_dask.py / test_mpi.py / test_filehandler.py / test_data_util.py
        ├── test_beam.py / test_beam_ska-low.py / test_ionosphere.py
        ├── test_line_emission.py / test_long_observation.py / test_mock_mightee.py
        ├── test_notebooks.py / test_examples.py / test_superimpose.py
        ├── test_system_noise.py / test_telescope_baselines.py
        ├── test_rascil_telescope_setup.py / test_sim_format_imager_combos.py
        ├── test_coordinate_helper.py / test_utils.py
        └── data/  (blank_image.fits, detection.csv, run5.cst, cst_like_beam_*.txt, …)
```

`karabo/.experiments/` (hidden directory) holds throwaway research scripts: `source_detection_exp.py`, `plot_source_detection_exp.py`, `README.md`.

---

## 3. Installation & dependencies

### 3.1 Recommended path — conda (per `doc/src/installation_user.md`)

```bash
# 1. Miniconda + libmamba solver
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
source ~/miniconda3/bin/activate
conda init bash
conda install -n base conda-libmamba-solver

# 2. Dedicated env with Python 3.10
conda create -n karabo python=3.10
conda activate karabo
conda config --env --set solver libmamba
conda config --env --set channel_priority true

# 3. Install Karabo (from i4ds Anaconda channel)
conda install -c nvidia/label/cuda-12.9.1 -c i4ds -c conda-forge karabo-pipeline

# WSL2 only:
conda env config vars set LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/wsl/lib
```

System requirements (`installation_user.md` lines 3–7): Linux or Windows-WSL (macOS users are explicitly directed to the Docker image starting v0.18.1), 8 GB RAM, 10 GB disk, optional NVIDIA CUDA ≥ 11 (the docs as of `v0.34.0` reference CUDA 12.9; the in-tree `Dockerfile` line 1 is still pinned to CUDA `11.7.1` and `environment.yaml` to `cuda-version=11.7`).

### 3.2 Direct pip / source install

`pyproject.toml` declares the build system as `setuptools>=56.0 + wheel + versioneer[toml]`. `setup.py` is a 9-line versioneer thin wrapper; the conda `build.sh` is a single line:

```bash
$PYTHON -m pip install --no-deps .
```

i.e. dependencies are *expected* to come from conda — `pip install karabo-pipeline` is **not** the supported path because the runtime depends on conda-only binaries (OSKAR, casacore-with-MPICH, WSClean, ska-gridder-nifty-cuda, etc.).

### 3.3 Pinned runtime dependencies (`environment.yaml`, `conda/meta.yaml`)

```
python >=3.9,<3.11
aratmospy 1.0.0   astropy            bdsf 1.10.2     bluebild 0.1.0
casacore           cuda-cudart 11.7   cuda-version 11.7
dask 2022.12.1     dask-mpi           distributed
eidos 1.1.0        healpy             h5py *=mpi_mpich*
ipython            katbeam 0.1.0      libcufft         matplotlib
montagepy 6.0.0    mpi4py             mpich            nbformat / nbconvert
numpy >=1.21,!=1.24.0,<2.0
oskarpy 2.8.3      packaging          pandas           psutil
rascil 1.0.0       reproject >=0.9,<=10.0
requests           rfc3986 >=2.0.0    scipy >=1.10.1,<1.14
ska-gridder-nifty-cuda 0.3.0
ska-sdp-datamodels 0.1.3              ska-sdp-func-python 0.1.4
tools21cm 2.0.3    wsclean            xarray >=2022.11
fftw =*=mpi_mpich* (conda-forge)
setuptools >=69.0.0,<70.0.0  (env)  /  != 71.0.0,71.0.1,71.0.2  (recipe)
pyuvdata[casa] >=2.4,<3
```

Optional `[dev]` (`pyproject.toml` lines 44–63): black 23.10.0 (with `[jupyter]`), flake8 6.1.0, isort 5.12.0, pre-commit 3.5.0, pydocstyle 6.3.0, pytest 7.4.2, pytest-cov 4.1.0, mypy 1.6.1, sphinx, sphinx_rtd_theme, myst-parser, ipykernel, nest_asyncio, podmena (commit emojis), versioneer.

### 3.4 Container

`Dockerfile` (51 lines) builds on `nvidia/cuda:11.7.1-cudnn8-devel-ubuntu22.04`:

1. Install `git gcc gfortran libarchive13 wget curl nano` via apt.
2. Install Miniconda (`Miniconda3-py311_23.5.2-0-Linux-x86_64.sh`) into `/opt/conda`, activate libmamba solver.
3. Create env `karabo` (default `PYTHON_VERSION=3.10`, overridable via build-arg).
4. Either `BUILD=user` → `conda install -c i4ds -c conda-forge -c nvidia/label/cuda-11.7.1 karabo-pipeline=$KARABO_VERSION`, or `BUILD=test` → install from local source.
5. Copies `karabo/examples/*` to `/workspace/karabo-examples`, installs JupyterLab + ipykernel + pytest, registers an `ipykernel` named `karabo`.
6. `BASH_ENV=/opt/etc/conda_init_script` so non-interactive shells auto-activate the env; `ldconfig` registers the conda MPICH so the MPI hook resolves.
7. `ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "karabo"]`, `WORKDIR /workspace`.

Build args: `GIT_REV` (default `main`), `BUILD` (`user`|`test`), `KARABO_VERSION` (semver without `v`), `PYTHON_VERSION` (default `3.10`).

### 3.5 Kubernetes assets

`k8s/SRCNet_SKA-MID_AAstar/` ships `job.yaml`, `cronjob.yaml`, `tmp-pvc.yaml` plus a Helm-style `data-ingestor/{values.yaml,secrets.yaml}` for SKA SRCNet ingestion runs. Documented in its own `README.md`.

---

## 4. Runtime architecture

Karabo is a **wrapper-and-glue layer**. Most modules instantiate objects from the wrapped libraries, marshal them through Karabo's homogenised dataclasses, and re-emit to disk in standard formats (Measurement Set, OSKAR `.vis`, FITS, HDF5).

```
                     ┌──────────────────────────────────────────────┐
                     │  USER  (Python script · Jupyter notebook)    │
                     └──────────────────────────────────────────────┘
                                          │
                                          ▼
┌────────────────────────────────────────────────────────────────────────────┐
│                       karabo  high-level Python API                        │
│                                                                            │
│  ┌────────────┐  ┌──────────────┐  ┌─────────────┐  ┌──────────────────┐  │
│  │ Telescope  │  │  SkyModel    │  │ Observation │  │ Interferometer-  │  │
│  │ (.tm dirs) │  │ (xarray)     │  │ (+Long/     │  │ Simulation       │  │
│  │            │  │              │  │  Parallelized)│  │                  │  │
│  └────────────┘  └──────────────┘  └─────────────┘  └──────────────────┘  │
│         │              │                  │                   │            │
│         └──────────────┴──────────────────┴───────────────────┘            │
│                                  │                                         │
│                    SimulatorBackend = {OSKAR | RASCIL}                     │
│                                  │                                         │
│   ┌──────────────────────────────┴───────────────────────────┐             │
│   │ Visibility (path + format ∈ {MS, OSKAR_VIS, UVFITS})    │             │
│   └──────────────────────────────┬───────────────────────────┘             │
│                                  │                                         │
│  ┌──────────┐  ┌──────────────┐  ▼  ┌────────────────┐  ┌──────────────┐ │
│  │ Image    │  │ DirtyImager / Image Cleaner (ABC)    │  │ SourceDet.   │ │
│  │ Mosaick. │  │  ├─ OskarDirtyImager                 │  │ Result       │ │
│  │          │  │  ├─ RascilDirtyImager / Cleaner      │  │ (PyBDSF)     │ │
│  │          │  │  └─ WscleanDirty/CleanImager         │  │ + Evaluation │ │
│  └──────────┘  └─────────────────────────────────────┘  └──────────────┘ │
│                                                                            │
│  Cross-cutting: util/dask.py · util/file_handler.py · data/external_data   │
└────────────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌────────────────────────────────────────────────────────────────────────────┐
│           Wrapped engines (conda-installed, called as Python or CLI)        │
│   oskarpy → liboskar / CUDA · rascil · ska-sdp-{datamodels,func-python}    │
│   wsclean (CLI) · pybdsf · casacore · pyuvdata · healpy · katbeam · eidos  │
│   aratmospy · tools21cm · bluebild · ska-gridder-nifty-cuda                │
│   dask / dask-mpi / mpich · MPI4PY                                          │
└────────────────────────────────────────────────────────────────────────────┘
```

The **strategy point** of the architecture is `karabo/simulator_backend.py`:

```python
# karabo/simulator_backend.py
class SimulatorBackend(enum.Enum):
    OSKAR = "OSKAR"
    RASCIL = "RASCIL"
```

Almost every public class accepts `backend: SimulatorBackend` and dispatches to a private `__run_simulation_oskar` / `__run_simulation_rascil` method, with `assert_never(backend)` exhaustiveness checks.

### 4.1 Package import side-effects (`karabo/__init__.py`)

The root `__init__.py` is intentionally thin (it must run during build-time as `versioneer` calls into it):

1. Imports `filter_data_dir_warning_message` from `util/rascil_util.py` (silences a noisy RASCIL warning).
2. Reads `__version__` from `_version.py` (versioneer).
3. **WSL fix-up**: if `platform.release()` contains `"WSL"` and `LD_LIBRARY_PATH` lacks `/usr/lib/wsl/lib`, it appends it and `os.execv`'s the interpreter to take effect.
4. **SLURM hook**: if `SLURM_JOB_ID` is in env, imports `DaskHandlerSlurm` and calls `_prepare_slurm_nodes_for_dask()` so SLURM ranks see a coherent dask cluster from script start.

---

## 5. Public API — class-by-class

The following sections cover every primary user-facing class. Signatures are condensed where defaults are obvious; cite `karabo/<module>.py` for source line numbers.

### 5.1 `karabo.simulation.telescope.Telescope` (`karabo/simulation/telescope.py`, 942 LOC)

The central representation of an array. Carries WGS-84 centre and a list of `Station` objects (each with east/north/up coordinates and a per-station antenna list).

```python
class Telescope:
    def __init__(self, longitude: float, latitude: float, altitude: float = 0.0)

    # Primary factory:
    @classmethod
    def constructor(
        cls,
        name: Union[OSKARTelescopesWithVersionType,
                    OSKARTelescopesWithoutVersionType,
                    RASCILTelescopes],
        version: Optional[enum.Enum] = None,
        backend: SimulatorBackend = SimulatorBackend.OSKAR,
    ) -> Telescope                                            # lines 246–340
```

`OSKARTelescopesWithVersionType` (lines 66–85) is a `Literal[...]` covering: `ACA`, `ALMA`, `ATCA`, `CARMA`, `NGVLA`, `PDBI`, `SKA-LOW-AA{0.5,1,2,4,star}`, `SKA-MID-AA{0.5,1,2,4,star}`, `SMA`, `VLA`. Each accepts a `version` enum from `karabo/simulation/telescope_versions.py` (one enum class per telescope, e.g. `ALMAVersions.CYCLE_10_3` resolves to filename `alma.cycle10.3.tm`).

`OSKARTelescopesWithoutVersionType` (lines 86–96): `EXAMPLE`, `MeerKAT`, `ASKAP`, `LOFAR`, `MKATPlus`, `SKA1LOW`, `SKA1MID`, `VLBA`, `WSRT`.

`RASCILTelescopes` (lines 99–114): `LOWBD2`, `LOWBD2-CORE`, `LOW`, `LOWR3`, `LOWR4`, `LOW-AA0.5`, `MID`, `MIDR5`, `MID-AA0.5`, `MEERKAT+`, `ASKAP`, `LOFAR`, `VLAA`, `VLAA_north` (delegated to `ska_sdp_datamodels.configuration.config_create.create_named_configuration`).

The mapping `OSKAR_TELESCOPE_TO_FILENAMES` (lines 116–147) ties each name to a `*.tm` directory inside `karabo/data/`. The 225 `.tm` directories cover ALMA cycles 0–11, ACA cycles 1–10, ATCA, CARMA, NGVLA configurations, PDBI, SMA, VLA, and the 10 SKA-LOW/MID AA0.5–AAstar configurations (`SKA-LOW-AA0.5.ska-ost-array-config-2.3.1.tm`, etc.).

Other notable methods:

| Method                                  | Purpose | Source |
|----------------------------------------|---------|--------|
| `read_OSKAR_tm_file(path)`             | Load any `.tm` directory (parses `position.txt`, `layout.txt`, `station<NNN>/layout.txt`) | lines 623–725 |
| `write_to_disk(dir_name, *, overwrite)` | Write the in-memory model back to a `.tm` directory; enforces `.tm` suffix | lines 575–599 |
| `get_OSKAR_telescope()`                 | Round-trip into `oskar.telescope.Telescope` (writes a temp `.tm`) | lines 556–573 |
| `add_station(...)` / `add_antenna_to_station(...)` | Programmatic build-up | lines 426–502 |
| `plot_telescope(file=None)`             | Matplotlib lon/lat scatter; routes to `plot_telescope_OSKAR` for both backends | lines 503–554 |
| `get_cartesian_position()`              | Centre in ECEF                            | lines 620–621 |
| `get_stations_wgs84()` / `get_baseline_lengths(stations_wgs84)` | Baseline statistics in metres (Euclidean ECEF, not geodesic) | lines 875–913 |
| `max_baseline()`                        | Longest baseline (metres) | lines 915–923 |
| `ang_res(freq, b)`                      | θ = c/(f·B), returns arcsec | lines 925–942 |
| `create_baseline_cut_telescope(lcut, hcut, tel, tm_path=None)` | Returns a new `.tm` containing only baselines with `lcut < b < hcut` (m); also returns a station-name mapping | lines 798–873 |
| `_get_station_infos(tel_path)` (cls)    | Returns a `pd.DataFrame[station-nr, station-path, x, y]` | lines 753–796 |
| `__convert_to_karabo_telescope(name)` (cls, RASCIL) | Wraps RASCIL `Configuration` so `max_baseline()`/baseline-cut still work | lines 343–390 |

### 5.2 `karabo.simulation.station.Station` and `EastNorthCoordinate`

`station.py` (29 LOC) defines `Station` with fields `position: EastNorthCoordinate`, `longitude/latitude/altitude` (the centre of the telescope), and `antennas: List[EastNorthCoordinate]`. `east_north_coordinate.py` (29 LOC) defines an `EastNorthCoordinate` dataclass with `x, y, z, x_error, y_error, z_error`. `coordinate_helper.py` (52 LOC) provides `wgs84_to_cartesian(lon, lat, alt)`.

### 5.3 `karabo.simulation.observation.Observation*`

```python
class ObservationAbstract(ABC):                               # observation.py:14
    def __init__(*, mode="Tracking", start_frequency_hz=0,
                 start_date_and_time: Union[datetime, str],
                 length: timedelta = timedelta(hours=4),
                 number_of_channels=1, frequency_increment_hz=0,
                 phase_centre_ra_deg=0, phase_centre_dec_deg=0,
                 number_of_time_steps=1)
    def set_length_of_observation(self, hours, minutes, seconds, milliseconds)
    def get_OSKAR_settings_tree(self) -> OskarSettingsTreeType  # OSKAR settings dict
    @staticmethod
    def create_observations_oskar_from_lists(settings_tree,
        central_frequencies_hz, channel_bandwidths_hz, n_channels)
    def get_phase_centre(self) -> List[float]
    def compute_hour_angles_of_observation(self) -> NDArray[np.float_]  # for RASCIL

class Observation(ObservationAbstract): ...  # plain
class ObservationLong(ObservationAbstract):
    # adds number_of_days >= 2; length capped at 12 h
class ObservationParallelized(ObservationAbstract):
    # accepts list[float] center_frequencies_hz / channel_bandwidths_hz / n_channels
```

`get_OSKAR_settings_tree` (lines 99–126) constructs the `oskar` dict with date format `"%d-%m-%Y %H:%M:%S.%f"` (truncated to ms) and `length` as `H:M:S:ms`.

`compute_hour_angles_of_observation` (lines 217–240) computes hour angles for RASCIL DFT given `length` and `number_of_time_steps`.

### 5.4 `karabo.simulation.sky_model.SkyModel` (2318 LOC — the largest module)

A wrapper around an `xarray.DataArray` with **14 columns** (constant `SOURCES_COLS = 14`):

| Idx | Field            | Unit / dtype     |
|----:|------------------|------------------|
|  0  | right ascension  | deg              |
|  1  | declination      | deg              |
|  2  | Stokes I flux    | Jy               |
|  3  | Stokes Q flux    | Jy (default 0)   |
|  4  | Stokes U flux    | Jy               |
|  5  | Stokes V flux    | Jy               |
|  6  | reference frequency | Hz            |
|  7  | spectral index   | dimensionless    |
|  8  | rotation measure | rad/m²           |
|  9  | major axis FWHM  | arcsec           |
| 10  | minor axis FWHM  | arcsec           |
| 11  | position angle   | deg              |
| 12  | true redshift    | —                |
| 13  | observed redshift| — (line emission)|
| 14  | source id (object)| placed in `xr.coords`, not a column |

Constants `_STOKES_IDX` and `COL_IDX` enable string→column lookups (`sky_model.py` lines 536–559).

Construction:

```python
SkyModel(sources=None,            # xr.DataArray | np.ndarray | None
         wcs=None,
         precision=np.float64,    # np.dtype
         h5_file_connection=None) # h5py.File for lazy / dask backed reads
```

Selected methods (alphabetic):

| Method | Purpose | Source |
|--------|---------|--------|
| `add_point_sources(sources)` | Append rows; auto-pads to 14 cols, treats column 14 as source-id coord | lines 731–782 |
| `close()` | Closes the underlying h5py file | lines 611–618 |
| `compute()` | Forces a Dask computation, materialises the array | lines 634–649 |
| `convert_to_backend(backend, desired_frequencies_hz, channel_bandwidth_hz)` | OSKAR → returns self; RASCIL → builds `List[SkyComponent]` (Stokes-I, all-channel flat continuum) | lines 2183–2318 |
| `copy_sky(sky)` (static) | Deep copy keeping the h5 connection alive | lines 620–632 |
| `explore_sky(phase_center, stokes, ...)` | Matplotlib scatter plot via WCS (`AIR` projection by default) | lines 1146–1289 |
| `filter_by_radius(inner, outer, ra0, dec0, indices=False)` | Annular spherical filter via `astropy.SphericalCircle` | lines 871–940 |
| `filter_by_radius_euclidean_flat_approximation(...)` | Flat-Euclid version (Dask-friendly, large catalogs) | lines 942–1041 |
| `filter_by_column(col_idx, min_val, max_val)` | Generic two-sided filter | lines 1043–1075 |
| `filter_by_flux(min_jy, max_jy)` / `filter_by_frequency(min_hz, max_hz)` | Wrappers | lines 1077–1089 |
| `get_OSKAR_sky(sky, precision)` (static) | Convert to `oskar.Sky.from_array(..., 'single'\|'double')`; trims to 12 cols | lines 1291–1305 |
| `read_healpix_file_to_sky_model_array(file, channel, polarisation)` (static) | HDF5/HEALPix → xarray RA/DEC/flux | lines 1307–1329 |
| `read_from_file(path)` (cls) | CSV reader (15 named columns) | lines 787–828 |
| `save_sky_model_as_csv(path)` / `write_to_file(path)` | Round-trip CSV | lines 1478–1509 |
| `to_np_array(with_obj_ids=False)` | Materialise (and optionally append id column) | lines 830–858 |
| `setup_default_wcs(phase_center=[0,0])` / `set_wcs(wcs)` / `get_wcs()` | astropy WCS setup (`RA---AIR`, `DEC--AIR`) | lines 1091–1131 |
| `get_cartesian_sky()` | Unit vectors per source | lines 1524–1536 |
| `get_random_poisson_disk_sky(min_size, max_size, flux_min, flux_max, r=3)` (cls) | Synthetic Poisson-disk catalogue | lines 2121–2133 |
| `sky_test()` / `sky_test_LE()` (cls) | 81-source test grids around (RA=20°, Dec=-30°); `sky_test_LE` adds redshifts in [0.8,1.0] | lines 2135–2181 |

Survey loaders — these download once via the `external_data` framework (cached under `XDG_CACHE_HOME` or `$HOME/.cache`) and parse the FITS catalogue against a `SkyPrefixMapping`/`SkySourcesUnits` declarative schema:

| Class method | Catalogue | Sources | Frequencies | Citation |
|--------------|-----------|---------|-------------|----------|
| `get_GLEAM_Sky(min_freq, max_freq)` | GLEAM (MWA) | >300 000 (south of dec +30) | 72–231 MHz, encoded as col-name suffixes `Fp{0}`/`a{0}`/`b{0}` | lines 1919–1972 |
| `get_MIGHTEE_Sky(...)` | MIGHTEE Continuum Early Science L1 (MeerKAT, COSMOS) | 9896 | 1304–1375 MHz | lines 2007–2059 |
| `get_MALS_DR1V3_Sky(...)` | MALS DR1V3 (MeerKAT) | 715 760 | 902–1644 MHz | lines 2061–2119 |
| `get_sample_simulated_catalog()` | HI sources small simulated catalogue (~8 MB, courtesy ETHZ Cosmology) | — | — | lines 1974–2005 |
| `get_sky_model_from_h5_to_xarray(path, prefix_mapping=None, load_as="dask_array", chunksize="auto")` | Generic HDF5 reader (lazy-loadable via Dask) | — | — | lines 1538–1598 |
| `get_sky_model_from_fits(*, fits_file, prefix_mapping, unit_mapping, units_sources=None, min_freq=None, max_freq=None, encoded_freq=None, chunks="auto", memmap=False)` | Generic FITS reader supporting both column-frequency-encoded and per-row ref-freq tables | lines 1827–1917 |

Supporting dataclasses for FITS parsing:

* `SkyPrefixMapping` (lines 116–140) maps the 14 SkyModel fields to one-or-many FITS column names.
* `SkySourcesUnits` (lines 143–321) defines `astropy.units` defaults for each field and provides `format_sky_prefix_freq_mapping(...)`, `get_unit_scales(...)`, `extract_names_and_freqs(...)`, `get_pos_ids_to_ra_dec(pos_ids)` (decodes `JHHMMSS±DDMMSS` IDs to ra/dec degrees).

### 5.5 `karabo.simulation.interferometer.InterferometerSimulation` (`interferometer.py`, 984 LOC)

The principal simulator façade. Constructor (lines 217–325) exposes ~30 parameters mirroring OSKAR's settings tree:

```python
InterferometerSimulation(
    channel_bandwidth_hz=0,           time_average_sec=0,
    max_time_per_samples=8,
    correlation_type=CorrelationType.Cross_Correlations,   # / Auto / Both
    uv_filter_min=0, uv_filter_max=float("inf"),
    uv_filter_units=FilterUnits.WaveLengths,               # / Metres
    force_polarised_ms=False, ignore_w_components=False,
    noise_enable=False,  noise_seed="time",
    noise_start_freq=1e9, noise_inc_freq=1e8, noise_number_freq=24,
    noise_rms_start=0,   noise_rms_end=0,
    noise_rms="Range",   noise_freq="Range",      # see Literal types below
    enable_array_beam=False, enable_numerical_beam=False,
    use_gpus=None,                                # auto: gpu_util.is_cuda_available()
    use_dask=None, split_observation_by_channels=False,
    n_split_channels="each", client=None,
    precision="single",                           # / "double"
    station_type="Isotropic beam",                # / "Aperture array" / "Gaussian beam" / "VLA (PBCOR)"
    enable_power_pattern=False,                   # if True: gauss_beam_fwhm *= sqrt(2)
    gauss_beam_fwhm_deg=0.0, gauss_ref_freq_hz=0.0,
    ionosphere_fits_path=None, ionosphere_screen_type=None,
    ionosphere_screen_height_km=300, ionosphere_screen_pixel_size_m=0,
    ionosphere_isoplanatic_screen=False,
)
```

Type literals: `NoiseRmsType = Literal["Range","Data file","Telescope model"]`, `NoiseFreqType = Literal["Range","Data file","Observation settings","Telescope model"]`, `StationTypeType = Literal["Aperture array","Isotropic beam","Gaussian beam","VLA (PBCOR)"]` (lines 105–110).

Primary entry-point (overloaded for type-safety):

```python
def run_simulation(
    self,
    telescope: Telescope,
    sky: SkyModel,
    observation: ObservationAbstract,
    backend: SimulatorBackend = SimulatorBackend.OSKAR,
    primary_beam: Optional[RASCILImage] = None,        # only honoured for RASCIL
    visibility_format: VisibilityFormat = "MS",        # "MS" | "OSKAR_VIS" | "UVFITS"
    visibility_path: Optional[Union[DirPath, FilePath]] = None,
) -> Union[Visibility, List[Visibility]]
```

Behaviour (lines 420–526):

* OSKAR + `Observation` → `__setup_run_simulation_oskar` builds an OSKAR settings tree (`__get_OSKAR_settings_tree`, lines 913–975), then calls `oskar.Interferometer.run`. Output is one `.MS` or `.vis` file.
* OSKAR + `ObservationLong` → `__run_simulation_long` (lines 795–864) loops over days, writes `<date>.vis` per day, then calls `Visibility.combine_vis(...)`. Only `MS` output supported.
* OSKAR + `ObservationParallelized` → `__run_simulation_parallelized_observation` (lines 645–720) uses Dask: `client.scatter` the sky array, `delayed(__run_simulation_oskar)` per channel-group, returns a `List[Visibility]`. Picks `start_freq_<Hz>.<MS|vis>` filenames.
* RASCIL → `__run_simulation_rascil` (lines 538–643) builds a `ska_sdp_datamodels.visibility.create_visibility` with hour-angles + frequency channels, converts the SkyModel to `SkyComponent` list via `SkyModel.convert_to_backend`, optionally `apply_beam_to_skycomponent(skycomponents, primary_beam)`, then `dft_skycomponent_visibility(... dft_compute_kernel="cpu_looped")` and `export_visibility_to_ms`.

Other methods:

* `simulate_foreground_vis(telescope, foreground, foreground_observation, foreground_vis_file)` — runs the simulation, then opens the OSKAR `.vis` to expose `(visibility, foreground_cross_correlation_list, fg_header, fg_handle, fg_block, ff_uu, ff_vv, ff_ww)`.
* `set_ionosphere(file_path)` — register a TEC FITS produced by ARatmospy.
* `yes_double_precision()` — `self.precision != "single"`.
* `__interpret_uv_filter(value)` — sentinels `inf`→`"max"`, `<=0`→`"min"`.

`format_timedelta(td: timedelta) -> str` (lines 56–83) is a helper formatting time differences as `[-]HH h MM m SS.fff s`.

### 5.6 `karabo.simulation.visibility` (228 LOC)

```python
VisibilityFormat = Literal["MS", "OSKAR_VIS", "UVFITS"]   # line 15

class VisibilityFormatUtil:
    @classmethod is_valid_path_for_format(cls, path, format) -> bool
    @classmethod parse_visibility_format_from_path(cls, path) -> Optional[VisibilityFormat]

class Visibility:
    def __init__(self, path: Union[DirPath, FilePath]) -> None
    # auto-infers self.format from path extension (.ms / .vis / .uvfits|.uvf)

    @classmethod combine_vis(cls,
        visibilities: List[Visibility],
        combined_ms_filepath: Optional[DirPathType] = None,
        group_by: str = "day",
    ) -> DirPathType
    # OSKAR_VIS-only (others raise NotImplementedError); reads each via
    # oskar.VisHeader/VisBlock and writes a single combined CASA MS

    @classmethod num_measurements(cls, *,
        start_freq_hz, end_freq_hz, freq_inc_hz,
        num_time_stamps, num_stations, num_polarizations,
    ) -> int  # estimator for MS table size
```

Validation lives in the `_VISIBILITY_FORMAT_VALIDATORS` dict — each format has a lambda enforcing extension. `combine_vis` writes coords/visibilities for `group_by="day"` (per-block per-time) or any other value (mean across files).

### 5.7 `karabo.simulation.beam`

`beam.py` (130 LOC) exposes pure functions:

* `gaussian_beam_fwhm_for_frequency(desired_frequency, reference_fwhm_degrees=1.8, reference_frequency_Hz=8e8)` — uses MeerKAT-like reference values (`REFERENCE_FWHM_DEGREES`, `REFERENCE_FREQUENCY_HZ`).
* `generate_gaussian_beam_data(fwhm_pixels, x_size, y_size)` — uses `astropy.convolution.Gaussian2DKernel` and normalises peak to 1.
* `generate_eidos_beam(npixels, image_width_degrees, frequencies, stokes="I")` — shells out to the `eidos` CLI per frequency (writes FITS files in the working directory).
* `generate_airy_beam_data(fwhm_pixels, x_size, y_size)` — Airy pattern via `scipy.special.j1`.
* `airy_beam_fwhm_for_frequency(frequency_hz, dish_diameter_m)` — `0.61 λ / D` in degrees.

### 5.8 `karabo.simulation.line_emission`

`line_emission.py` (250 LOC). One pipeline function:

```python
def line_emission_pipeline(
    output_base_directory: Union[Path, str],
    pointings: List[CircleSkyRegion],     # named-tuple (center: SkyCoord, radius: u.Quantity)
    sky_model: SkyModel,
    observation_details: Observation,
    telescope: Telescope,
    interferometer: InterferometerSimulation,
    simulator_backend: SimulatorBackend,
    dirty_imager_config: DirtyImagerConfig,
    primary_beams: Optional[List[NDArray[np.float_]]] = None,
    should_perform_primary_beam_correction: Optional[bool] = True,
) -> Tuple[List[List[Visibility]], List[List[Image]]]
```

For each `(frequency_channel, pointing)`: filter the sky by annular radius and by **observed redshift** column 13 (mapped to the channel via `convert_frequency_to_z`), set `number_of_channels=1`, run `interferometer.run_simulation`, then `RascilDirtyImager.create_dirty_image` (filename `dirty_<backend>_<index_p>.fits`).

`line_emission_helpers.py` (91 LOC) contains `convert_frequency_to_z(freq, rest_freq=21cm hyperfine)` and helpers for HI line cosmology.

### 5.9 `karabo.simulation.signal.*`

Composable building blocks for diffuse/EoR/foreground cubes:

| File | Class | Role |
|------|-------|------|
| `base_signal.py` | `BaseSignal[ImageT]` (Generic ABC) | `simulate() → list[ImageT]` |
| `base_segmentation.py` | `BaseSegmentation` (ABC) | EoR-cube segmentation interface |
| `signal_21_cm.py` | `Signal21cm(BaseSignal[Image3D])` | Wraps `tools21cm.t2c`; downloads xfrac/dens cubes from `https://ttt.astro.su.se/~gmell/244Mpc`. Provides `available_redshifts()`, `get_xfrac_dens_file(z, box_dims)`, `randomized_lightcones(N, n_z)`. |
| `eor_profile.py` | `EORProfile` | Global 21-cm signal profile |
| `synchroton_signal.py` | `SynchrotronSignal` (Image2D) | Galactic synchrotron foreground |
| `galactic_foreground.py` | `GalacticForeground` | combined foreground |
| `superimpose.py` | `Superimpose` | Adds 2D/3D images: `_combine_2_images_2d`, `_combine_images_3d_2d`, `_combine_2_images_3d`, public `combine(*signals)` |
| `seg_u_net_segmentation.py` | `SegUNetSegmentation(BaseSegmentation)` | U-Net binary mask |
| `superpixel_segmentation.py` | `SuperpixelSegmentation` | classic CV segmentation |
| `helpers.py` | numerous geometry/cosmology helpers |
| `typing.py` | `BaseImage`, `Image2D`, `Image3D`, `XFracDensFilePair` (frozen dataclasses) |
| `plotting.py` | `SignalPlotting` (matplotlib renderers) |

### 5.10 Imaging API

`imaging/imager_base.py` (128 LOC) defines two ABC pairs and configs (`@dataclass`):

```python
@dataclass class DirtyImagerConfig:    imaging_npixel: int; imaging_cellsize: float
                                       combine_across_frequencies: bool = True
class DirtyImager(ABC):
    config: DirtyImagerConfig
    @abstractmethod create_dirty_image(self, visibility, /, *, output_fits_path=None) -> Image

@dataclass class ImageCleanerConfig:   imaging_npixel: int; imaging_cellsize: float
class ImageCleaner(ABC):
    @abstractmethod create_cleaned_image(self, visibility, /, *,
                                          dirty_fits_path=None, output_fits_path=None) -> Image
```

#### 5.10.1 `OskarDirtyImager` (`imager_oskar.py`, 121 LOC)

```python
@dataclass class OskarDirtyImagerConfig(DirtyImagerConfig):
    imaging_phase_centre: Optional[str] = None  # SkyCoord string

class OskarDirtyImager(DirtyImager):
    def create_dirty_image(self, visibility, /, *, output_fits_path=None) -> Image
```

Internally calls `oskar.Imager().set(input_file=..., output_root=..., cellsize_arcsec=..., image_size=...)`, runs with `return_images=1`, renames OSKAR's `<root>_I.fits` to the requested path, then post-processes the FITS header (`NAXIS=4`, `NAXIS4=1`, fixes `CDELT3` which OSKAR sets to 0). Constraint: `combine_across_frequencies=False` is `NotImplementedError` (OSKAR always integrates across frequency).

#### 5.10.2 `RascilDirtyImager` / `RascilImageCleaner` (`imager_rascil.py`, 519 LOC)

Dirty path uses `rascil.processing_components.create_visibility_from_ms`, masks autocorrelations (`antenna1==antenna2`), then `ska_sdp_func_python.imaging.{create_image_from_visibility, invert_visibility}` with `context="2d"`. If `combine_across_frequencies=True`, sums across the frequency axis and rewrites the FITS.

`RascilImageCleanerConfig` (lines 181–281) is a dense dataclass exposing the RASCIL clean pipeline (msclean / hogbom / mmclean), facets, taper, n-iter/major, multi-scale, gain, threshold, restored-output mode, `imaging_dft_kernel` (`cpu_looped`/`gpu_raw`), `use_dask`, `n_threads`, `use_cuda` (which forces `img_context='wg'` / Nifty WGridder). Drives `rascil.workflows.continuum_imaging_skymodel_list_rsexecute_workflow` via `rsexecute`.

`create_cleaned_image_variants` (lines 343–430) returns `(deconvolved, restored, residual)` Images.

#### 5.10.3 `WscleanDirtyImager` / `WscleanImageCleaner` (`imager_wsclean.py`, 299 LOC)

Both shell out to the `wsclean` CLI under `subprocess.run(shell=True, capture_output=True, check=True)` after `cd <tmpdir>` and exporting `OPENBLAS_NUM_THREADS=1` (a workaround per WSClean's own multi-thread warning). Only `MS` visibilities and `combine_across_frequencies=True` are supported. The cleaner config exposes `niter` (default 50000), `mgain` (0.8), `auto_threshold` (3) — set to `None` to omit the flag.

`create_image_custom_command(command, output_filenames)` (lines 240–299) lets a user pass an arbitrary `wsclean ...` command string and harvest specific output FITS files.

#### 5.10.4 `Image` and `ImageMosaicker` (`imaging/image.py`, 973 LOC)

`Image` (lines 46–832) wraps `(data: ndarray, header: astropy.io.fits.Header)`. Constructed either from `path=` (FITS) or `data=,header=`. Always reshaped to **4D** `(frequency, polarisation, x, y)` (2D / 3D inputs are auto-broadcast with a warning). Major methods:

| Method | Purpose | Source |
|--------|---------|--------|
| `read_from_file(path)` (static) | FITS read | line 130 |
| `write_to_file(path, overwrite=False)` | FITS write | line 143 |
| `header_has_parameters(parameters)` / `has_beam_parameters()` / `get_beam_parameters() -> BeamType` | Header probing | lines 165–668 |
| `get_squeezed_data()` | first freq + first pol slice | line 174 |
| `resample(shape, **kwargs)` | bilinear resample with `RegularGridInterpolator`; in-place; updates `CDELT*` | lines 177–223 |
| `cutout(center_xy, size_xy)` | astropy `Cutout2D` (returns new Image) | line 225 |
| `circle()` | masks pixels outside an inscribed circle (in-place per channel/pol) | lines 249–272 |
| `split_image(N, overlap=0)` | `N×N` cutouts | lines 302–367 |
| `to_NNData()` / `to_2dNNData(ra_dec_axis=(1,2))` | `astropy.nddata.NDData` adapters | lines 369–375 |
| `plot(...)` | WCS-aware plot via matplotlib `imshow` | lines 377–466 |
| `overplot_with_skymodel(sky, filename=None, channel_index=0, stokes_index=0, vmin_image=None, vmax_image=None)` | Overlay catalogue | line 468 |
| `plot_side_by_side_with_skymodel(...)` | 1×2 panel | line 532 |
| `get_dimensions_of_image()` / `get_phase_center()` | — | lines 610–626 |
| `get_quality_metric()` | RMS, peak, etc. | line 670 |
| `get_power_spectrum() / plot_power_spectrum()` | wraps `rascil.apps.imaging_qa.imaging_qa_diagnostics.power_spectrum` | lines 708–767 |
| `get_cellsize()` | radians, from `CDELT2` | line 768 |
| `get_wcs() / get_2d_wcs(ra_dec_axis=(1,2))` | astropy WCS | lines 782–790 |
| `get_corners_in_world(header)` (cls) | corner WCS coords as ICRS DataFrame | lines 790–831 |

`ImageMosaicker` (lines 834–973) wraps `reproject.mosaicking.{find_optimal_celestial_wcs, reproject_and_coadd}`:

```python
ImageMosaicker(reproject_function=reproject_interp,
               combine_function="mean",       # or "sum"
               match_background=False,
               background_reference=None)
.get_optimal_wcs(images, projection="SIN", **kwargs) -> (WCS, (int,int))
.mosaic(images, wcs=None, input_weights=None, hdu_in=None, hdu_weights=None,
        shape_out=None, image_for_header=None, **kwargs) -> (Image, footprint)
```

#### 5.10.5 `imaging/util.py`

* `get_MGCLS_images(regex_pattern, verbose=False) -> List[SkaSdpImage]` — downloads MeerKAT Galaxy Cluster Legacy Survey DR1 cubes from CSCS via `MGCLSContainerDownloadObject` and imports each with `rascil.processing_components.image.operations.import_image_from_fits`.
* `_convert_clean_beam_to_degrees(im, beam_pixels) -> BeamType`, `guess_beam_parameters(img) -> BeamType` — astropy 2-D Gaussian fit on a 15×15 PSF cutout.
* `project_sky_to_image(sky, phase_center, imaging_cellsize, imaging_npixel, filter_outlier=True, invert_ra=True) -> (img_coords, idxs)` — converts a `SkyModel` into pixel coordinates against an `RA---AIR / DEC--AIR` WCS.

### 5.11 Source detection (`karabo/sourcedetection/`)

`result.py` (749 LOC) declares:

* `ISourceDetectionResult` (ABC) — `detected_sources`, `has_source_image`, `get_source_image()`.
* `SourceDetectionResult(ISourceDetectionResult)` — generic wrapper around an `(N, ≥7)` array `[index, ra, dec, pos_x_pix, pos_y_pix, total_flux, peak_flux, ...]`.
  * `detect_sources_in_image(image, beam=None, verbose=False, **kwargs) -> Optional[SourceDetectionResult]` (cls) — invokes `bdsf.process_image`. If `beam` is None, tries `image.get_beam_parameters()`; otherwise calls `imaging.util.guess_beam_parameters`. `kwargs` forwarded to PyBDSF.
  * `write_to_file(path)` — zips the source FITS + catalog CSV.
* `PyBDSFSourceDetectionResult(SourceDetectionResult)` — built from a `bdsf.image.Image`. Writes a Gaussian-list CSV (`sources.csv`), exports the BDSF result image, and exposes 14 image-type accessors: `get_RMS_map_image`, `get_mean_map_image`, `get_polarised_intensity_image`, `get_gaussian_residual_image`, `get_gaussian_model_image`, `get_shapelet_residual_image`, `get_shapelet_model_image`, `get_major_axis_FWHM_variation_image`, `get_minor_axis_FWHM_variation_image`, `get_position_angle_variation_image`, `get_peak_to_total_flux_variation_image`, `get_peak_to_aperture_flux_variation_image`, `get_island_mask_image`, plus the 7-column reduced array indexed by `BDSFResultIdxsToUseForKarabo = [0,4,6,12,14,8,9]` (Gauss_id, RA, DEC, Total_flux, Peak_flux, RA_max, E_RA_max).
* `PyBDSFSourceDetectionResultList(ISourceDetectionResult)` — aggregates per-tile detections with a `min_pixel_distance_between_sources` deduplication step (prefers higher total flux), useful with `Image.split_image(...)`-driven parallel detection.

`evaluation.py` (506 LOC) defines `SourceDetectionEvaluation`:

* Constructor takes `(sky, ground_truth, assignments, sky_idxs, source_detection)` and computes `tp/fp/fn`.
* `automatic_assignment_of_ground_truth_and_prediction(ground_truth, detected, max_dist, top_k=3)` (cls method) implements the algorithm of Masias-Moyset 2014 using `scipy.spatial.KDTree`. Returns an `(N,3)` array `[gt_idx, pred_idx, distance]`; `-1` indicates unassigned.
* `calculate_evaluation_measures(assignments)` (cls) — TP/FP/FN.
* Plus plotting helpers (`plot_assignments`, etc.).

### 5.12 Data utilities (`karabo/data/`)

* **`external_data.py`** — `DownloadObject(remote_base_url)` with classmethod `download(url, local_file_path, verify=True, verbose=True)` (uses `requests.get(stream=True, timeout=60)` + `tqdm`). `SingleFileDownloadObject` is the abstract single-resource handler. Concrete classes:
  * `GLEAMSurveyDownloadObject` → `surveys/gleam/GLEAM_EGC_v0.fits`
  * `HISourcesSmallCatalogDownloadObject` → `surveys/hi-small/HI_sources_small_catalog_v0.h5`
  * `MIGHTEESurveyDownloadObject`, `MALSSurveyV3DownloadObject`, `ExampleHDF5Map`
  * `DiffuseEmissionHaslam408DownloadObject` → NASA LAMBDA `lambda_mollweide_haslam408_dsds.fits`
  * `ContainerContents(remote_url, regexr_pattern)` + `MGCLSContainerDownloadObject(regexr_pattern)` for bulk fetches against the CSCS object store. Base URLs: `cscs_base_url = "https://rgw.cscs.ch/ska"`, public bucket `karabo-public`, testing prefix `karabo-public/testing`.
* **`casa.py`** (594 LOC) — Strict casacore MS table readers. `_get_cols(table, *, subtable_id=None)` extracts colvalues into a lower-case dict; `_create_table(table, classtype, *, subtable_id=None)` materialises dataclasses (e.g. `MSPolarizationTable`) from MS subtables. Used by `obscore.py`.
* **`obscore.py`** (1151 LOC) — `ObsCoreMeta` dataclass implementing the IVOA ObsCore data model. `FitsHeaderAxis` lookup helpers; constants for `_DataProductTypeType`, `_CalibLevelType`, `_PolStatesType`. Knows how to derive ObsCore metadata from OSKAR `VisHeader`, `pyuvdata.UVData`, and `casacore.tables.table` directly (so a Karabo-produced MS or `.vis` can be ingested by an ObsCore TAP service).
* **`src.py`** — `RucioMeta(namespace, name, lifetime, dataset_name=None, meta=Optional[Union[dict, ObsCoreMeta]])` for SKA SRCNet ingestion (matches `gitlab.com/ska-telescope/src/ska-src-ingestion`). `to_dict(fpath, *, ignore_none=True)` serialises to JSON. The example in `karabo/examples/SRCNet_rucio_meta.py` demonstrates the end-to-end SRCNet workflow.
* **Telescope `.tm` data** (`data/*.tm/`) — Each is an OSKAR-format directory: `position.txt` (lon, lat, alt), `layout.txt` (per-station east/north/up + errors), `station<NNN>/layout.txt` (per-antenna). 225 such bundles are shipped (ALMA cycles 0–11, ACA, ATCA, CARMA, NGVLA configs, PDBI, SMA, VLA, plus single-version directories `lofar.tm`, `meerkat.tm`, `mkatplus.tm`, `askap.tm`, `WSRT.tm`, `vlba.tm`, `ska1low.tm`, `ska1mid.tm`, `telescope.tm` (EXAMPLE), and the SKA-OST `SKA-{LOW,MID}-AA{0.5,1,2,4,star}.ska-ost-array-config-2.3.1.tm`). The helper directories `data/_add_oskar_alma_layouts/` and `data/_add_oskar_ska_layouts/` contain provisioning scripts/data for keeping these refreshed.

### 5.13 Utilities (`karabo/util/`)

| File | Highlights |
|------|-----------|
| `_types.py` | `NPFloatLike`, `NPFloatInpBroadType`, `IntFloat`, `IntFloatList`, `PrecisionType = Literal["single","double"]`, `OskarSettingsTreeType = Dict[str, Dict[str, Any]]`, `BeamType = TypedDict({bmaj,bmin,bpa})`, `MISSING` sentinel. |
| `data_util.py` (307 LOC) | `get_module_absolute_path()`, `get_module_path_of_module(module)`, `extract_digit_from_string`, `extract_chars_from_string`, `read_CSV_to_ndarray`, Voigt etc. |
| `file_handler.py` (513 LOC) | `FileHandler` short/long-term cache. `_get_disk_cache_root(term)` honours `TMPDIR`/`TMP`/`SCRATCH`/`/tmp` for STM and `XDG_CACHE_HOME`/`$HOME/.cache`/`/tmp` for LTM. Provides `get_tmp_dir(prefix, term="short", purpose=None, unique=None, mkdir=True)` plus `write_dir(dir, overwrite=False)` context manager and `assert_valid_ending(path, ending)`. Singleton-like (uses `purpose`, `unique`, `prefix` to bucket caches). |
| `dask.py` (749 LOC) | `DaskHandlerBasic` class-level singleton (`dask_client`, `memory_limit`, `n_threads_per_worker`, `use_dask`, `use_processes`); creates either an `MPI4PY`/`dask_mpi.initialize` cluster (`MPI.COMM_WORLD.Get_size() > 1`) or a local `LocalCluster` via `Nanny`/`Worker`. `parallelize_with_dask(iterate_function, iterable, *args, **kwargs)`. `_calc_num_of_workers()` derives concurrency from `psutil.virtual_memory()` and `psutil.cpu_count()`. `DaskHandlerSlurm._prepare_slurm_nodes_for_dask()` is invoked from `karabo/__init__.py` if `SLURM_JOB_ID` is set. `DaskHandler` is the environment-aware alias used everywhere. |
| `gpu_util.py` | `is_cuda_available()` and `get_gpu_memory()` parse `nvidia-smi`'s text output; the script doubles as a `__main__`. |
| `math_util.py` (278 LOC) | Voigt profile, `get_poisson_disk_sky(min_size, max_size, flux_min, flux_max, r)`, `long_lat_to_cartesian`, etc. |
| `hdf5_util.py` | `get_healpix_image(file)`, `convert_healpix_2_radec(filtered)`. |
| `helpers.py` (114 LOC) | `Environment.get(name, type)` typed env-var reader, `get_rnd_str(...)`. |
| `plotting_util.py` | `Font` constants, `get_slices(wcs)`. |
| `rascil_util.py` (40 LOC) | `filter_data_dir_warning_message()` — patches `logging` filter for noisy RASCIL `data_dir` warnings. |
| `jupyter.py` | Jupyter detection helpers. |
| `survey.py` (114 LOC, excluded from coverage) | Misc survey helpers. |
| `testing.py` (39 LOC) | testing helpers (mocking-related). |
| `config_util.py` | Config-loading helpers. |
| `ska_sdp_datamodels/__init__.py` | Vendored shim for `export_visibility_to_ms` (re-exports the upstream symbol; insulates Karabo from upstream import-path drift). |

---

## 6. Core flows

### 6.1 End-to-end smoke test (`karabo/simulation/sample_simulation.py`)

```python
phase_center = [250, -80]
sky = SkyModel.get_GLEAM_Sky(min_freq=72e6, max_freq=80e6) \
              .filter_by_radius(0, 0.55, *phase_center)
sky.setup_default_wcs(phase_center=phase_center)

telescope = Telescope.constructor("ASKAP", version=None, backend=SimulatorBackend.OSKAR)
observation = Observation(start_frequency_hz=100e6,
                          start_date_and_time=datetime(2024, 3, 15, 10, 46, 0),
                          phase_centre_ra_deg=250, phase_centre_dec_deg=-80,
                          number_of_channels=16, number_of_time_steps=24)
sim = InterferometerSimulation(channel_bandwidth_hz=1e6)
visibility = sim.run_simulation(telescope, sky, observation,
                                backend=SimulatorBackend.OSKAR,
                                visibility_format="MS")
```

`run_sample_simulation(...)` returns the tuple `(visibility, phase_center, sky, telescope, observation, interferometer_sim)` for downstream tests/notebooks.

### 6.2 Full pipeline: simulate → image → detect → evaluate

```python
# 1. Simulate (OSKAR or RASCIL)
vis = InterferometerSimulation(...).run_simulation(telescope, sky, observation,
                                                   backend=SimulatorBackend.OSKAR,
                                                   visibility_format="MS")

# 2. Image (choose backend)
imager = WscleanDirtyImager(DirtyImagerConfig(imaging_npixel=2048, imaging_cellsize=...))
dirty = imager.create_dirty_image(vis, output_fits_path="/tmp/dirty.fits")

cleaner = WscleanImageCleaner(WscleanImageCleanerConfig(imaging_npixel=2048,
                                                        imaging_cellsize=...,
                                                        niter=50000, mgain=0.8))
clean = cleaner.create_cleaned_image(vis, output_fits_path="/tmp/clean.fits")

# 3. Source detection
sd = PyBDSFSourceDetectionResult.detect_sources_in_image(clean)

# 4. Truth assignment + metrics
gt_pix, idxs = project_sky_to_image(sky, phase_center, cellsize, npixel)
assignments  = SourceDetectionEvaluation.automatic_assignment_of_ground_truth_and_prediction(
    ground_truth=gt_pix.T, detected=sd.detected_sources[:, 3:5], max_dist=5)
eval_ = SourceDetectionEvaluation(sky, gt_pix, assignments, idxs, sd)
print(eval_.tp, eval_.fp, eval_.fn)
```

Backends are interchangeable: e.g. swap `WscleanDirtyImager` → `OskarDirtyImager` or `RascilDirtyImager` (each has its own config dataclass).

### 6.3 Long observations & multi-day combining

`InterferometerSimulation.run_simulation` with an `ObservationLong` (`number_of_days >= 2`, per-day `length <= 12 h`) loops over days, writes one OSKAR `.vis` per day under a temp dir, then calls `Visibility.combine_vis(visibilities, combined_ms_filepath, group_by="day")` to merge into a single CASA MS.

### 6.4 Distributed / channel-parallel observations

`ObservationParallelized(center_frequencies_hz=[…], n_channels=[…], channel_bandwidths_hz=[…], …)` triggers the Dask-distributed path. `__run_simulation_parallelized_observation` (a) ensures a dask client (`DaskHandler.get_dask_client()`), (b) `client.scatter`s the sky array, (c) submits one `delayed(__run_simulation_oskar)` per channel-group, (d) collects with `dask.compute(*delayed_results, scheduler="distributed")`, returning a `List[Visibility]` keyed by `start_freq_<Hz>.<ext>`.

### 6.5 Line-emission pipeline

`line_emission_pipeline(...)` (`karabo/simulation/line_emission.py`) runs once per (channel × pointing), filtering the sky by *radius and observed redshift*, simulating, dirty-imaging via RASCIL, and returning matrices of `Visibility`s and `Image`s. Optionally accepts pre-computed primary beams (one FITS-array per channel).

### 6.6 Mosaicking

`ImageMosaicker(reproject_function=reproject_interp, combine_function="mean", match_background=False).mosaic(images_list, wcs=...)` produces a single re-projected `Image` plus a footprint map.

### 6.7 21-cm / EoR signal generation

```python
from karabo.simulation.signal.signal_21_cm import Signal21cm
from karabo.simulation.signal.plotting import SignalPlotting

z1, z2 = Signal21cm.available_redshifts()[:2]
files = [Signal21cm.get_xfrac_dens_file(z=z, box_dims=244/0.7) for z in (z1, z2)]
sig = Signal21cm(files)
images = sig.simulate()                          # list[Image3D] in Kelvin
fig = SignalPlotting.brightness_temperature(images[0])
```

The xfrac/dens cubes are downloaded once from `https://ttt.astro.su.se/~gmell/244Mpc/...` and processed with `tools21cm`. `Superimpose.combine(*signals)` merges multiple signals (broadcasting 2D over 3D layers as needed).

### 6.8 SRCNet / Rucio ingestion

`karabo/examples/SRCNet_simulation_walkthrough.ipynb`, `SRCNet_rucio_meta.py`, `SRCNet_v0.1_simulation_1_MeerKAT.py`, `SRCNet_v0.1_simulation_2_AAstar.py` and `karabo/workflows/SRCNet_SKA-MID_AAstar.py` form a complete chain: simulate visibilities, image with WSClean, build `ObsCoreMeta` from the resulting `.MS`/`.fits`, and emit a `RucioMeta` JSON for SKA SRC ingestion. `k8s/SRCNet_SKA-MID_AAstar/` packages the same as a Kubernetes `Job`/`CronJob`.

---

## 7. Input & output formats

| Direction | Format | Notes |
|-----------|--------|-------|
| In | OSKAR `.tm` directory | `Telescope.read_OSKAR_tm_file`; bundled in `karabo/data/`. |
| In | CSV catalogue (15 cols) | `SkyModel.read_from_file`. |
| In | FITS catalogue | `SkyModel.get_sky_model_from_fits`; supports column-frequency-encoded format. |
| In | HDF5 catalogue | `SkyModel.get_sky_model_from_h5_to_xarray`; lazy via Dask. |
| In | HEALPix HDF5 | `SkyModel.read_healpix_file_to_sky_model_array`. |
| In | TEC FITS (ARatmospy) | Ionosphere screen — `InterferometerSimulation(ionosphere_fits_path=...)`. |
| In | FITS image | `Image(path=...)`. |
| In | xfrac/dens binary cubes | `Signal21cm.get_xfrac_dens_file`. |
| Out | CASA Measurement Set (`.MS` directory) | Default for both OSKAR and RASCIL backends. |
| Out | OSKAR visibility binary (`.vis`) | Selected via `visibility_format="OSKAR_VIS"`. |
| Out | UVFITS (`.uvfits`/`.uvf`) | Recognised by `VisibilityFormatUtil` but disabled with `KaraboInterferometerSimulationError("unexpected uvfits")` in the OSKAR / RASCIL paths — i.e. read-only via `pyuvdata`, not written by Karabo. |
| Out | FITS image | All imagers, plus PyBDSF sub-images. |
| Out | CSV | `SkyModel.save_sky_model_as_csv`, BDSF Gaussian list. |
| Out | ZIP | `SourceDetectionResult.write_to_file` (FITS + CSV bundle). |
| Out | JSON | `RucioMeta.to_dict(fpath, ignore_none=True)`. |

---

## 8. Telescope inventory (cited from the codebase)

| OSKAR-name | Versioned? | `OSKAR_TELESCOPE_TO_FILENAMES` value | Versions enum |
|-----------|------------|--------------------------------------|---------------|
| `EXAMPLE` | no  | `telescope.tm` | — |
| `MeerKAT` | no  | `meerkat.tm`   | — |
| `ACA`     | yes | `aca.{0}.tm`   | `ACAVersions` |
| `ALMA`    | yes | `alma.{0}.tm`  | `ALMAVersions` (cycles 0–11, ALL) |
| `ASKAP`   | no  | `askap.tm`     | — |
| `ATCA`    | yes | `atca_{0}.tm`  | `ATCAVersions` |
| `CARMA`   | yes | `carma.{0}.tm` | `CARMAVersions` |
| `LOFAR`   | no  | `lofar.tm`     | — |
| `MKATPlus`| no  | `mkatplus.tm`  | — |
| `NGVLA`   | yes | `ngvla-{0}.tm` | `NGVLAVersions` |
| `PDBI`    | yes | `pdbi-{0}.tm`  | `PDBIVersions` |
| `SKA-LOW-AA0.5` … `AAstar` | yes | `SKA-LOW-AA{0.5,1,2,4,star}.{0}.tm` | `SKALowAA*Versions` |
| `SKA-MID-AA0.5` … `AAstar` | yes | `SKA-MID-AA{0.5,1,2,4,star}.{0}.tm` | `SKAMidAA*Versions` |
| `SKA1LOW` | no  | `ska1low.tm`   | — |
| `SKA1MID` | no  | `ska1mid.tm`   | — |
| `SMA`     | yes | `sma.{0}.tm`   | `SMAVersions` |
| `VLA`     | yes | `vla.{0}.tm`   | `VLAVersions` |
| `VLBA`    | no  | `vlba.tm`      | — |
| `WSRT`    | no  | `WSRT.tm`      | — |

RASCIL telescope strings (delegated to `ska_sdp_datamodels.configuration.config_create.create_named_configuration`): `LOWBD2`, `LOWBD2-CORE`, `LOW`, `LOWR3`, `LOWR4`, `LOW-AA0.5`, `MID`, `MIDR5`, `MID-AA0.5`, `MEERKAT+`, `ASKAP`, `LOFAR`, `VLAA`, `VLAA_north`.

---

## 9. Testing

* `pytest` is the only test driver. `pyproject.toml` (`[tool.pytest.ini_options] testpaths = "karabo/test"`).
* `setup.cfg` configures *coverage*, *flake8* (max-line 88, ignores `E203 W503 E704`), *isort* (`profile = black`), *mypy* (strict-ish: `disallow_any_generics=true`, `warn_return_any=true`, etc.), *pydocstyle* (Google convention, `D105` ignored), *green*. Line length 88.
* Pre-commit hook config in `.pre-commit-config.yaml`.
* Coverage `omit` skips `*/test/*`, `*/.experiments/*`, `*/examples/*`, `setup.py`, `*/_version.py`, `*/survey.py`.
* Tests live in `karabo/test/` (~30 modules):
  * Sim/IO smoke: `test_telescope.py`, `test_skymodel.py`, `test_observation.py`, `test_simulation.py`, `test_long_observation.py`, `test_telescope_baselines.py`, `test_rascil_telescope_setup.py`.
  * Imaging: `test_imager_oskar.py`, `test_imager_rascil.py`, `test_imager_wsclean.py`, `test_image.py`, `test_imaging_util.py`, `test_mosaic.py`, `test_sim_format_imager_combos.py`.
  * Source detection: `test_source_detection.py`.
  * I/O / metadata: `test_obscore.py`, `test_casa.py`, `test_filehandler.py`, `test_data_util.py`, `test_utils.py`.
  * Beam / ionosphere: `test_beam.py`, `test_beam_ska-low.py`, `test_ionosphere.py`, `test_system_noise.py`.
  * Notebooks / examples: `test_notebooks.py`, `test_examples.py` (run notebooks via `nbformat`/`nbconvert`).
  * Parallelism: `test_dask.py`, `test_mpi.py`.
  * Signals: `test_superimpose.py`, `test_line_emission.py`, `test_mock_mightee.py`, `test_coordinate_helper.py`.
  * Test fixtures in `karabo/test/data/`: `blank_image.fits`, `detection.csv`, `detection.zip`, `detection_result_512px.csv`, `cst_like_beam_port_{1,2}.txt`, `run5.cst`, `filtered_sky.csv`.

GitHub Actions workflows under `.github/workflows/`:

| Workflow | Purpose |
|----------|---------|
| `test.yml` | Unit tests (the primary CI). |
| `dev-workflow.yml` | Dev branch checks. |
| `conda-build.yml`, `conda-build-test.yml` | Build & test conda package. |
| `test-conda-wheel.yml` | Cross-check against the conda-built artifact. |
| `build-docker-image.yml` | Build the Karabo Docker image. |
| `build-docs.yml` | Sphinx docs → GitHub Pages (`i4ds.github.io/Karabo-Pipeline`). |
| `release.yml` | Versioned releases. |
| `update-citation-cff.yml` | Auto-bump `CITATION.cff`. |

---

## 10. Performance / parallel architecture

`doc/src/parallel_processing.md` (and `karabo/util/dask.py`) document Karabo's three layers of parallelism:

1. **Local Dask** — `DaskHandlerBasic` builds a `LocalCluster` via `Nanny`/`Worker`. Threading vs. processes is configurable (`use_processes=False` is the default because `pybdsf` cannot be forked).
2. **MPI + Dask** — when `MPI.COMM_WORLD.Get_size() > 1`, `dask_mpi.initialize(comm=MPI.COMM_WORLD, [nthreads=...])` is called and a `Client(processes=cls.use_processes)` is returned. The conda env pins `mpich` and packages every MPI-tagged variant of casacore/h5py/fftw to keep ABIs aligned.
3. **SLURM hooks** — `DaskHandlerSlurm._prepare_slurm_nodes_for_dask()` is run by `karabo/__init__.py` whenever `SLURM_JOB_ID` is set, so a salloc/sbatch invocation has a working dask cluster on first import.

GPU paths:

* `gpu_util.is_cuda_available()` → `InterferometerSimulation` defaults `use_gpus=True`.
* OSKAR uses CUDA when the conda CUDA stack is present.
* RASCIL `RascilImageCleanerConfig.use_cuda=True` switches the gridder to `wg` (Nifty W-Gridder, `ska-gridder-nifty-cuda`).

`karabo/performance_test/` contains scripts used for the i4ds paper benchmark:

* `time_karabo.py` — single-node walltime.
* `time_karabo_parallelization_by_channel.py` — channel-parallel sweep with Dask.
* `time_karabo_reconstruction.py` — full clean reconstruction benchmark.
* `time_karabo_slurm_h5.py` + `sbatch_script.sh` — SLURM script.
* `paper/karabo_benchmark.py`, `paper/process_results.py`, `paper/reporting.py`, `paper/skymodel_reader.py`, `paper/run_paper_benchmark_sing.sh`, `paper/README.md`.

---

## 11. Extension points

| To add … | Touch | Notes |
|----------|-------|-------|
| A new telescope (OSKAR-style) | drop a `<name>.tm` directory under `karabo/data/`, add the literal to `OSKARTelescopesWith{out,}VersionType`, add the filename to `OSKAR_TELESCOPE_TO_FILENAMES`, add an enum to `telescope_versions.py` if versioned. |
| A new RASCIL telescope | extend `RASCILTelescopes` literal once `ska_sdp_datamodels.configuration.config_create.create_named_configuration` learns about it. |
| A new visibility format | add to `VisibilityFormat` literal in `simulation/visibility.py` and register a path validator in `_VISIBILITY_FORMAT_VALIDATORS`. The assertions on lines 24–25 lock the two in sync. |
| A new dirty imager | subclass `DirtyImager` in `imaging/imager_base.py`; create a sibling `<Foo>DirtyImagerConfig(DirtyImagerConfig)` dataclass. |
| A new image cleaner | subclass `ImageCleaner` similarly. |
| A new sky catalogue | add a `SkyPrefixMapping`, `SkySourcesUnits` and a thin `@classmethod get_<NAME>_Sky(cls, min_freq=None, max_freq=None)` on `SkyModel`; back it with a `SingleFileDownloadObject` if the catalogue is remote. |
| A new signal | subclass `BaseSignal[Image2D|Image3D]` in `simulation/signal/`. |
| A new download object | subclass `DownloadObject` (file or container) in `data/external_data.py`. The remote naming convention is `<dirname><file|dir>_v<version>` (see line 22 of that file). |
| A new ObsCore field | add a dataclass field to `data/obscore.py:ObsCoreMeta` and update the relevant from-`*` factory. |

---

## 12. Notable internals & caveats

* **Frozen-by-design caching.** `FileHandler` deliberately caches by a tuple of `(prefix, term, purpose, unique)`; if you call `FileHandler().get_tmp_dir(unique=self)` with the same `self` you get the same directory across calls — this is intentional but surprising.
* **Auto-correlations are dropped at imaging time** by `RascilDirtyImager` (sets `flags=1` for `antenna1==antenna2`).
* **RASCIL backend currently ignores Stokes Q/U/V** (`SkyModel.convert_to_backend(SimulatorBackend.RASCIL, ...)` hard-codes `PolarisationFrame("stokesI")` and a `flux_array[:, 0]`). The TODO comments at lines 597–599 of `interferometer.py` and 2289–2295 of `sky_model.py` flag this.
* **OSKAR cannot accept a custom primary beam directly** — `run_simulation` warns and ignores `primary_beam` for OSKAR; users must instead set the `gauss_beam_fwhm_deg`/`gauss_ref_freq_hz` constructor parameters or use `enable_array_beam`/`enable_numerical_beam`. RASCIL accepts an explicit `RASCILImage` primary beam.
* **Power vs. field pattern.** `enable_power_pattern=True` triggers `gauss_beam_fwhm_deg *= sqrt(2)` (`interferometer.py` line 317) — careful when comparing to OSKAR's native settings.
* **Visibility format mismatch with `Observation*` types.** `ObservationLong` and `ObservationParallelized` only support `MS` output; OSKAR_VIS and UVFITS raise `KaraboInterferometerSimulationError`/`NotImplementedError`. UVFITS is never produced.
* **`Image.__init__` reshapes silently.** 2D / 3D inputs are auto-broadcast into `(freq, pol, x, y)` with a `warnings.warn` — if you depend on the original shape, beware.
* **WSClean integration is purely subprocess-based** and assumes `wsclean` is on `$PATH` (the conda recipe ships it). `OPENBLAS_NUM_THREADS=1` is forced.
* **`scipy < 1.14` is pinned** because `healpy 1.15.0` imports `scipy.integrate.trapz` (removed later); see comment at `environment.yaml` lines 44–47.
* **`numpy < 2.0`** is pinned (`!=1.24.0`) — see issue #584 (`environment.yaml` line 35).
* **WSL automatic restart.** `karabo/__init__.py` lines 17–30 auto-`os.execv`'s the interpreter once `LD_LIBRARY_PATH` is patched. Wrapping Karabo inside a long-running daemon may surprise you.
* **MIT vs. BSD-3 badge mismatch in README.** The badge is from anaconda.org and may be stale; `LICENSE` and `pyproject.toml` declare MIT. The README footer (line 50) also says MIT.
* **Static "Alpha" classifier despite v0.34.** Karabo retains `Development Status :: 3 - Alpha` in `pyproject.toml` line 12, and the install docs explicitly state "we don't guarantee" stable APIs across minor releases (`installation_user.md` lines 36–38).
* **Long-term cache is hard-coded to one host.** Surveys are downloaded from `https://rgw.cscs.ch/ska:karabo-public/...` (CSCS S3, Switzerland). There is no mirror; if CSCS is unreachable, `get_GLEAM_Sky` etc. fail.
* **The vendored `karabo/util/ska_sdp_datamodels/__init__.py`** is a single-line shim that re-exports `export_visibility_to_ms`. It exists solely to insulate Karabo from upstream renames — there is no actual vendored copy of `ska-sdp-datamodels`.

---

## 13. Headline classes — quick reference

```python
# Telescopes
karabo.simulation.telescope.Telescope
karabo.simulation.station.Station
karabo.simulation.east_north_coordinate.EastNorthCoordinate

# Sky
karabo.simulation.sky_model.SkyModel
karabo.simulation.sky_model.SkyPrefixMapping
karabo.simulation.sky_model.SkySourcesUnits

# Observation
karabo.simulation.observation.Observation
karabo.simulation.observation.ObservationLong
karabo.simulation.observation.ObservationParallelized

# Simulation
karabo.simulator_backend.SimulatorBackend          # Enum {OSKAR, RASCIL}
karabo.simulation.interferometer.InterferometerSimulation
karabo.simulation.interferometer.CorrelationType   # Enum
karabo.simulation.interferometer.FilterUnits       # Enum
karabo.simulation.visibility.Visibility
karabo.simulation.visibility.VisibilityFormat      # Literal
karabo.simulation.visibility.VisibilityFormatUtil

# Imaging
karabo.imaging.image.Image
karabo.imaging.image.ImageMosaicker
karabo.imaging.imager_base.DirtyImager / DirtyImagerConfig
karabo.imaging.imager_base.ImageCleaner / ImageCleanerConfig
karabo.imaging.imager_oskar.OskarDirtyImager / OskarDirtyImagerConfig
karabo.imaging.imager_rascil.RascilDirtyImager / RascilDirtyImagerConfig
karabo.imaging.imager_rascil.RascilImageCleaner / RascilImageCleanerConfig
karabo.imaging.imager_wsclean.WscleanDirtyImager
karabo.imaging.imager_wsclean.WscleanImageCleaner / WscleanImageCleanerConfig
karabo.imaging.imager_wsclean.create_image_custom_command

# Source detection
karabo.sourcedetection.result.ISourceDetectionResult
karabo.sourcedetection.result.SourceDetectionResult
karabo.sourcedetection.result.PyBDSFSourceDetectionResult
karabo.sourcedetection.result.PyBDSFSourceDetectionResultList
karabo.sourcedetection.evaluation.SourceDetectionEvaluation

# Signal generation
karabo.simulation.signal.signal_21_cm.Signal21cm
karabo.simulation.signal.eor_profile.EORProfile
karabo.simulation.signal.synchroton_signal.SynchrotronSignal
karabo.simulation.signal.galactic_foreground.GalacticForeground
karabo.simulation.signal.superimpose.Superimpose
karabo.simulation.signal.seg_u_net_segmentation.SegUNetSegmentation
karabo.simulation.signal.superpixel_segmentation.SuperpixelSegmentation

# Data, metadata & I/O
karabo.data.external_data.{DownloadObject, SingleFileDownloadObject,
    GLEAMSurveyDownloadObject, MIGHTEESurveyDownloadObject,
    MALSSurveyV3DownloadObject, HISourcesSmallCatalogDownloadObject,
    DiffuseEmissionHaslam408DownloadObject, MGCLSContainerDownloadObject,
    ContainerContents, ExampleHDF5Map}
karabo.data.casa.MSPolarizationTable          # plus other MS table dataclasses
karabo.data.obscore.ObsCoreMeta
karabo.data.obscore.FitsHeaderAxis
karabo.data.src.RucioMeta

# Utilities
karabo.util.dask.DaskHandler / DaskHandlerBasic / DaskHandlerSlurm
karabo.util.file_handler.FileHandler
karabo.util.gpu_util.{is_cuda_available, get_gpu_memory}
karabo.util.helpers.Environment
karabo.util._types.{NPFloatLike, IntFloat, BeamType, PrecisionType,
                    OskarSettingsTreeType, DirPathType, FilePathType}

# Errors and warnings
karabo.error.{KaraboError, KaraboDaskError, KaraboInterferometerSimulationError,
              KaraboSkyModelError, KaraboSourceDetectionEvaluationError, NodeTermination}
karabo.warning.{KaraboWarning, InterferometerSimulationWarning}
karabo.karabo_resource.{NumpyHandleError, HiddenPrints, CaptureSpam}
```

There is **no console-script CLI** — `pyproject.toml` declares no `[project.scripts]` entry-points. All interaction is via Python or Jupyter notebooks. The closest things to CLIs are:

* `karabo/examples/SRCNet_v0.1_simulation_*.py` — argparse-based scripts.
* `karabo/examples/SRCNet_rucio_meta.py` — argparse driver.
* `karabo/workflows/SRCNet_SKA-MID_AAstar.py` — the `Environment.get(...)` env-driven pipeline used inside the K8s `Job`.
* `karabo/util/gpu_util.py` (when run as `__main__`).

Everything else is a Python library.

---

*Generated by reading the source under `simulators/Karabo/` only — no external assumptions, no dependence on the wrapped libraries' upstream documentation.*
