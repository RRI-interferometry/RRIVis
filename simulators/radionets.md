# radionets — Exhaustive Technical Reference

> Source path: `simulators/radionets/` (git submodule of RRIVis).
> All citations below use paths relative to the repository root unless prefixed.

---

## 1. Overview & Purpose

**radionets** is a PyTorch-based deep-learning framework for **imaging radio interferometric data** with convolutional neural networks. The goal is to reconstruct calibrated radio interferometer observations from incomplete/sampled `(u, v)` Fourier-space data, producing high-resolution sky-brightness images. It bundles dataset simulation, model training, and quantitative evaluation utilities behind three CLI entry points.

The package historically also includes a small in-repo simulation pipeline (Gaussian sources, point sources, VLBA-like sampling), though further development of simulations has migrated to companion repositories `radiosim` and `pyvisgen` (see `simulators/radionets/README.md` lines 16–17, 62–67).

| Field | Value |
|---|---|
| Name | `radionets` |
| Latest tag | `v0.4.1` (2025-08-08) — see `simulators/radionets/CHANGES.rst` line 1 |
| Status | Beta (`Development Status :: 4 - Beta`, `pyproject.toml` line 28) |
| License | MIT (`simulators/radionets/LICENSE`) — Copyright (c) 2019 Kevin Schmidt |
| Languages | Python (≥ 3.11), TOML (configs) |
| Build backend | `hatchling` + `hatch-vcs` (`pyproject.toml` lines 1–3, 110–114) |
| Source layout | `src/radionets/` (PEP 621, hatch wheel sources mapping line 117–120) |
| Maintainers | Kevin Schmitz, Anno Knierim (TU Dortmund) — `pyproject.toml` lines 13–16 |
| Reference paper | Schmidt et al., A&A — DOI `10.1051/0004-6361/202142113` (`CITATION.cff` line 43) |
| Zenodo DOI | `205111370` (`README.md` badge) |
| Repository | https://github.com/radionets-project/radionets |
| Docs | https://radionets.readthedocs.io/en/latest/ |

Total Python LOC under `src/radionets/`: **9 580 lines** across **42 files** (`wc -l` aggregate).

---

## 2. Repository Layout

```
simulators/radionets/
├── README.md                  # Overview + structure (see lines 51–88)
├── LICENSE                    # MIT
├── CITATION.cff               # Author/DOI metadata
├── CHANGES.rst                # Towncrier-managed changelog
├── pyproject.toml             # Hatchling build, deps, scripts, ruff, towncrier
├── environment-dev.yml        # Conda/mamba dev env (pytorch + cudatoolkit + cartopy)
├── uv.lock                    # uv lockfile (664 KB)
├── .pre-commit-config.yaml    # pre-commit hooks (ruff etc.)
├── .readthedocs.yml           # Read-the-Docs build config
├── .zenodo.json               # Zenodo metadata
├── .github/                   # CI workflows
├── assets/                    # README & paper images, logos (PNG)
├── configs/
│   └── radionets_default_train_config.toml   # Shipped at install time → share/configs
├── docs/                      # Sphinx documentation
│   ├── conf.py
│   ├── index.rst
│   ├── changelog.rst
│   ├── citeus.md / glossary.md / references.{md,bib}
│   ├── api-reference/         # core / architecure / training / plotting / evaluation
│   ├── user-guide/  developer-guide/  changes/  _static/  _templates/
│   └── Makefile  make.bat
├── examples/                  # Tutorial notebooks + matplotlib RC files
│   ├── 00_observation_simulation.ipynb
│   ├── 01_dataset_simulation.ipynb
│   ├── 02_model_training.ipynb
│   ├── 03_model_evaluation.ipynb
│   ├── 07_benchmark_testset.ipynb
│   ├── Archs&Losses.ipynb
│   ├── default_simulation_config.toml
│   ├── default_eval_config.toml
│   ├── nb.py
│   └── matplotlib_rcs/        # paper_*.rc style sheets
├── src/radionets/             # Python package
│   ├── __init__.py            # Registers PuOr cmap; rich tracebacks
│   ├── version.py             # _version-file fallback
│   ├── core/                  # data, learner, callbacks, model, losses, logging
│   ├── architecture/          # archs, blocks, layers, activation, unc_archs
│   ├── training/              # CLI + utils
│   ├── evaluation/            # CLI + utils + analysis (jet/blob/contour/dr)
│   ├── plotting/              # Hist class, visualization, inspection, PuOr cmap
│   ├── simulations/           # gaussians, point_sources, uv_simulations, sampling
│   │   ├── layouts/           # VLBA antenna text file + loader
│   │   └── scripts/           # simulate_images CLI
│   └── tools/                 # quickstart CLI
└── tests/
    ├── conftest.py            # Session cleanup of ./tests/build/
    ├── simulate.toml / training.toml / evaluate.toml
    ├── test_simulation.py     # Order: first
    ├── test_training.py       # Databunch / training / save / pre-load / plot_loss
    ├── test_evaluation.py     # Order: last; full pipeline
    ├── test_architecture_layers.py  # Locally/Complex Conv/InstNorm/PReLU
    └── model/                 # Sample HDF5 + .model artifacts for tests
```

---

## 3. Installation & Dependencies

### 3.1 Runtime dependencies (`pyproject.toml` lines 33–50)

```
astropy >= 7.1.0          comet-ml >= 3.50.0     fastai >= 2.8.2
h5py >= 3.14.0            kornia >= 0.8.1        natsort >= 8.4.0
numba >= 0.61.2           numpy >= 2.2.6         pandas >= 2.3.1
pytorch-msssim >= 1.0.0   rich-click >= 1.8.9    rich >= 14.1.0
scikit-image >= 0.25.2    toml >= 0.10.2         torch >= 2.7.1
tqdm >= 4.67.1
```

Optional `plot` extra: `matplotlib >= 3.10.5`.

### 3.2 Dev / docs / tests groups (`pyproject.toml` lines 61–102)

| Group | Notable contents |
|---|---|
| `tests` | `pytest >= 7.0`, `pytest-cov`, `pytest-order`, `pytest-xdist`, `coverage`, `restructuredtext-lint`, `tomli` |
| `docs`  | `sphinx >= 8.1.3`, `pydata-sphinx-theme`, `sphinx-{automodapi,changelog,copybutton,design,gallery,tippy,togglebutton}`, `sphinx-autobuild`, `myst-parser`, `nbsphinx`, `numpydoc`, `sphinxcontrib-bibtex`, `linkify-it-py`, `graphviz`, `jupyter`, `notebook`, `ipython`, `matplotlib` |
| `dev` | All of the above + `pre-commit >= 4.2.0`, `ipython`, `jupyter` |

### 3.3 Conda env (`environment-dev.yml`)

```yaml
channels: [fastai, pytorch, defaults, conda-forge]
dependencies:
  - python, pytorch, cudatoolkit, cartopy, numpy, numba
  - pip:
      - towncrier
      - -e .
```

Cartopy is required at runtime only for `simulations/visualize_simulations.py` (cartopy.crs / cartopy.io.img_tiles) and `simulations/uv_plots.py`. It is **not** declared in `pyproject.toml` runtime deps, so plotting paths require a conda install.

### 3.4 Python / hardware requirements

* `requires-python = ">=3.11"` (line 31), classifiers cover 3.11–3.12.
* GPU optional: `CudaCallback._order = 3` does `self.model.cuda()` in `before_fit` (`core/callbacks.py` lines 271–286). `eval_model` auto-detects via `torch.cuda.is_available()` (`evaluation/utils.py` lines 322–332). README notes `cudatoolkit >= 11.3` (line 108).

### 3.5 Install steps (`README.md` lines 21–33)

```bash
mamba env create -f environment.yml      # README references environment.yml; current file is environment-dev.yml
pre-commit install
```

---

## 4. Build & Runtime Architecture

### 4.1 Pipeline

```
┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│ 1. SIMULATION    │ →   │ 2. TRAINING      │ →   │ 3. EVALUATION    │ →   │ 4. RECONSTRUCTION│
│  Gaussians/Point │     │  fastai Learner  │     │  Inference + Hist│     │   ifft(pred)     │
│  uv-coverage     │     │  callbacks       │     │  blobs/dr/area/  │     │   image-space    │
│  fft sampling    │     │  loss switching  │     │  ms_ssim/jet     │     │   reconstruction │
│  HDF5 bundles    │     │  .model pickle   │     │  predictions.h5  │     │                  │
└──────────────────┘     └──────────────────┘     └──────────────────┘     └──────────────────┘
   simulations/             core/ + training/         evaluation/             evaluation.utils.get_ifft
   scripts/simulate_images  scripts/start_training    scripts/start_evaluation
```

* The training input `x` is a sampled (u, v) Fourier-space tensor with two channels (real/imag or amp/phase). The target `y` is either the sampled-truth Fourier image or the source image (controlled by `[general] fourier` and `amp_phase`, see `tests/training.toml` lines 21–22).
* Networks predict 2- or 4-channel Fourier maps. 4 channels = (mean₁, var₁, mean₂, var₂) for uncertainty heads (`core/loss_functions.py::beta_nll_loss` lines 79–115).
* Final image is obtained by `ifft2(fftshift(...))` followed by `abs(...)` in `evaluation/utils.py::get_ifft` (lines 335–362) and `core/utils.py::get_ifft_torch` (lines 10–30).

### 4.2 Layered code architecture

| Layer | Location | Role |
|---|---|---|
| CLI entrypoints | `simulations/scripts/`, `training/scripts/`, `evaluation/scripts/`, `tools/quickstart.py` | Click + rich-click TOML-driven commands |
| Pipeline orchestration | `core/learner.py`, `training/utils.py`, `evaluation/train_inspection.py` | fastai `Learner`, databunch, normalization, inspection |
| Models | `architecture/{archs,unc_archs,blocks,layers,activation}.py` | nn.Module subclasses; complex-valued layers |
| Losses / callbacks | `core/{loss_functions,callbacks,model}.py` | Custom losses, Comet logging, normalization, save/load |
| Data IO | `core/data.py` | `H5DataSet`, `DataBunch`, save/open HDF5 helpers |
| Simulation | `simulations/{gaussians,point_sources,uv_simulations,sampling}.py` | Source generation + sampling masks |
| Plotting | `plotting/{hist,visualization,inspection,_puor}.py` | Matplotlib helpers, custom PuOr cmap |
| Evaluation analysis | `evaluation/{jet_angle,blob_detection,contour,dynamic_range,jets,pointsources}.py` | Quantitative metrics over test set |

### 4.3 Package `__init__` side effects (`src/radionets/__init__.py`)

```python
colormaps.register(cmap=PuOr); colormaps.register(cmap=PuOr_r)
add_safe_globals([L])                  # fastcore.foundation.L for torch.load
install(show_locals=False)             # rich tracebacks
```

This means **importing radionets at all** registers two matplotlib colormaps (`radionets.PuOr`, `radionets.PuOr_r`), enables rich tracebacks globally, and whitelists fastcore’s `L` for torch’s safe-loading. Affected callers: `core/callbacks.py` (CometCallback uses `cmap="radionets.PuOr"`).

---

## 5. Public API / CLI

### 5.1 `[project.scripts]` (`pyproject.toml` lines 104–108)

| Console script | Module | Function | Purpose |
|---|---|---|---|
| `radionets-simulation` | `radionets.simulations.scripts.simulate_images` | `main` | Generate fft + sampled HDF5 bundles |
| `radionets-training`   | `radionets.training.scripts.start_training`     | `main` | Train / lr_find / fine_tune / plot_loss |
| `radionets-evaluation` | `radionets.evaluation.scripts.start_evaluation` | `main` | Inference + metrics + plotting |
| `radionets-quickstart` | `radionets.tools.quickstart`                    | `quickstart` | Copy default config TOML |

Each command consumes a single TOML config (Click argument `configuration_path`).

### 5.2 README usage block (`README.md` lines 35–49)

```
radionets_simulations  <config.toml>
radionets_training     <config.toml>
radionets_evaluation   <config.toml>
```

The README references underscore variants (`radionets_simulations`); current scripts table uses **dashes** (`radionets-simulation`, etc.) — note the discrepancy.

### 5.3 Training subcommand modes (`training/scripts/start_training.py` lines 28–34)

```
--mode {train, lr_find, plot_loss, fine_tune}    (default: train)
```

* `train` — full training loop; saves model + loss plot; optional inspection plots.
* `fine_tune` — uses `Learner.fine_tune`; requires `pre_model`.
* `lr_find` — runs `Learner.lr_find()` and saves an `lr_loss.png/pdf`.
* `plot_loss` — re-renders saved train/valid loss curves from a `.model` checkpoint.

### 5.4 Quickstart (`tools/quickstart.py`)

Reads default config from `share/configs/radionets_default_train_config.toml` (installed via `[tool.hatch.build.targets.wheel.shared-data]`, `pyproject.toml` lines 122–123) and copies it to a user-supplied path, with a `-y/--yes` flag for non-interactive overwrite.

### 5.5 Public Python API surface (selected `__all__` exports)

| Module | Exports |
|---|---|
| `radionets.core` (`core/__init__.py`) | `AvgLossCallback, CometCallback, CudaCallback, DataAug, DataBunch, GradientCallback, H5DataSet, Normalize, PredictionImageGradient, SaveTempCallback, SwitchLoss, define_learner, get_bundles, get_dls, get_learner, init_cnn, load_data, load_pre_model, open_bundle, open_bundle_pack, open_fft_bundle, save_bundle, save_fft_pair, save_model, setup_logger` |
| `radionets.architecture` (`architecture/__init__.py`) | `BottleneckResBlock, Decoder, Encoder, GeneralELU, GeneralReLU, Lambda, LocallyConnected2d, NNBlock, SRBlock, SRResNet, SRResNet18, SRResNet18Complex, SRResNet18AmpPhase, SRResNet34, SRResNet34AmpPhase, SRResNet34_unc, SRResNet34_unc_no_grad, Uncertainty, UncertaintyWrapper` |
| `radionets.plotting` (`plotting/__init__.py`) | `Hist, plot_loss, plot_lr, plot_lr_loss` |
| `radionets.core.loss_functions` | `beta_nll_loss, create_circular_mask, jet_seg, l1, mse, splitted_L1, splitted_L1_masked` |

---

## 6. File-by-file Source Walkthrough

### 6.1 `src/radionets/core/`

| File | LOC | Key symbols | Notes |
|---|---:|---|---|
| `__init__.py` | 54 | re-exports | Aggregates user-facing API |
| `data.py` | 223 | `H5DataSet`, `DataBunch`, `get_bundles`, `get_dls`, `load_data`, `open_bundle`, `open_bundle_pack`, `open_fft_bundle`, `save_bundle`, `save_fft_pair` | HDF5 dataset using bundle files; key `x`/`y`/`z*` |
| `learner.py` | 97 | `get_learner`, `define_learner` | Constructs fastai `Learner` with callback list driven by config |
| `callbacks.py` | 521 | `CometCallback`, `AvgLossCallback`, `CudaCallback`, `DataAug`, `Normalize`, `SaveTempCallback`, `SwitchLoss`, `GradientCallback`, `PredictionImageGradient` | Fastai callbacks; ordering via `_order` |
| `model.py` | 104 | `init_cnn`, `_init_cnn`, `load_pre_model`, `save_model` | Kaiming init, full optimizer-state checkpointing, norm_dict serialization |
| `loss_functions.py` | 134 | `l1`, `mse`, `splitted_L1`, `splitted_L1_masked`, `beta_nll_loss`, `jet_seg`, `create_circular_mask` | Operates on dict input `x={"pred": tensor}` |
| `utils.py` | 46 | `_maybe_item`, `get_ifft_torch`, `split_real_imag`, `split_amp_phase` | Torch ifft helper supports `amp_phase`, log-scale, and uncertainty (4-ch) inputs |
| `logging.py` | 40 | `setup_logger` | Wraps `rich.logging.RichHandler` |

#### 6.1.1 H5DataSet design (`core/data.py` lines 23–90)

* Each bundle is an HDF5 file containing keys `x{i}`, `y{i}` and optional `z{i}` (point-source list).
* `__len__` = `num_bundles × num_img_per_bundle`. `__getitem__(i)` opens only the bundles needed for the indices.
* If `tar_fourier=False` but data has 2 channels → raises `ValueError("Two channeled data is used despite Fourier being False…")`.

#### 6.1.2 Loss functions

```python
# core/loss_functions.py
def splitted_L1(x, y):                          # amp + phase L1
    pred = x["pred"]; ...
    return l1(pred[:, 0], y[:, 0]) + l1(pred[:, 1], y[:, 1])

def beta_nll_loss(x, y, beta=0.5):              # heteroscedastic NLL
    # 4-channel pred = [mean_amp, var_amp, mean_phase, var_phase]
    loss = 0.5 * ((target - mean) ** 2 / variance + variance.log())
    if beta > 0: loss = loss * variance.detach() ** beta
```

`splitted_L1_masked` applies a circular pixel mask (radius=50, image 256×256) and downweights outside-mask values by 0.3.

#### 6.1.3 Callback `_order` schedule

| Callback | `_order` |
|---|---:|
| `CudaCallback`        | 3 |
| `DataAug`             | 3 |
| `Normalize`           | 4 |
| `SwitchLoss`          | 5 |
| `SaveTempCallback`    | 95 (writes `temp_{epoch+1}.model` every 10 epochs) |

`Normalize` supports three modes — `max`, `mean`, `all` (per-image z-score) — selected via `train_conf["normalize"]` (`callbacks.py` lines 318–371).

`SwitchLoss` swaps the loss function once `epoch+1 > when_switch` (defaults: `comb_likelihood` second loss, `callbacks.py` lines 399–421).

`CometCallback` (lines 42–218) initializes a `comet_ml.Experiment`, logs train/valid metrics and produces 4-panel prediction + 3-panel FFT figures every `plot_n_epochs`. Uses `radionets.PuOr` cmap.

`GradientCallback` cancels backward and re-runs it manually after raising `CancelBackwardException`, capturing per-parameter gradients on the last iteration of the last epoch (lines 424–481).

### 6.2 `src/radionets/architecture/`

| File | LOC | Classes |
|---|---:|---|
| `__init__.py` | 36 | aggregator |
| `activation.py` | 48 | `Lambda`, `GeneralReLU`, `GeneralELU` |
| `archs.py` | 260 | `SRResNet`, `SRResNetComplex`, `SRResNet18`, `SRResNet18Complex`, `SRResNet18AmpPhase`, `SRResNet34`, `SRResNet34AmpPhase`, `SRResNet34_unc`, `SRResNet34_unc_no_grad` |
| `blocks.py` | 641 | `NNBlock` (ABC), `SRBlock`, `ComplexSRBlock`, `BottleneckResBlock`, `Encoder`, `Decoder` |
| `layers.py` | 424 | `LocallyConnected2d`, `ComplexConv2d`, `ComplexInstanceNorm2d`, `ComplexPReLU` |
| `unc_archs.py` | 85 | `Uncertainty`, `UncertaintyWrapper` |

#### 6.2.1 SRResNet family (`architecture/archs.py`)

```python
class SRResNet(nn.Module):
    channels = 64
    preBlock  = Conv2d(2 → 64, k=9, p=4, groups=2) → PReLU
    blocks    = N × SRBlock(64, 64)
    postBlock = Conv2d(64 → 64, k=3, p=1, bias=False) → InstanceNorm2d
    final     = Conv2d(64 → 2, k=9, p=4, groups=2)

    forward(x): return {"pred": final( preBlock(x) + postBlock(blocks(...)) )}
```

| Class | Block count | Output activation |
|---|---:|---|
| `SRResNet18` | 8 | identity |
| `SRResNet34` | 16 | identity |
| `SRResNet18AmpPhase`, `SRResNet34AmpPhase` | 8 / 16 | amp = `ReLU`, phase = `Hardtanh(-π, π)` |
| `SRResNet18Complex` | 8 (Complex blocks) | complex |
| `SRResNet34_unc`, `SRResNet34_unc_no_grad` | 16 | reshapes to half-image (`s//2 + 1, s`); 4 output channels with `GeneralReLU(sub=-1e-10)` on the 2 variance channels; `_no_grad` variant detaches variance through `torch.no_grad()` |

`SRResNetComplex` uses `channels = 128` and stacks `ComplexSRBlock`s.

#### 6.2.2 Blocks (`architecture/blocks.py`)

* `NNBlock` — abstract; sets `idconv = Conv2d(1×1)` if channels mismatch and `pool = AvgPool2d(2, ceil_mode=True)` if `stride != 1`.
* `SRBlock` — two 3×3 reflect-padded Conv2d → InstanceNorm2d, PReLU after first; residual = `convs(x) + idconv(pool(x))`. Optional `dropout` parameter inserts `nn.Dropout` between conv blocks.
* `ComplexSRBlock` — same shape but with `ComplexConv2d`, `ComplexInstanceNorm2d`, `ComplexPReLU`.
* `BottleneckResBlock` — three 1×1 / 3×3 / 1×1 conv (channels `c → c//4 → c//4 → c`), BatchNorm + Dropout, residual + final PReLU. Used for deep ResNet experiments (added in v0.4.0, see `CHANGES.rst` lines 71–83).
* `Encoder` / `Decoder` — UNet primitives. Encoder = Conv2d → optional BatchNorm → PReLU; Decoder = ConvTranspose2d → PReLU.

#### 6.2.3 Layers (`architecture/layers.py`)

* `LocallyConnected2d` — uses `unfold(2)` and `unfold(3)` to extract patches, then position-specific weights of shape `(1, out_c, in_c, H, W, k²)`. Supports stride and optional bias of shape `(1, out_c, H, W)`.
* `ComplexConv2d` — implements `(a+bi)·(c+di)` via two real `nn.Conv2d` halves operating on real/imag channel chunks; default `padding="same"`.
* `ComplexInstanceNorm2d` — separate per-channel mean/variance for real and imag; affine learnables `weight_{real,imag}`, `bias_{real,imag}`.
* `ComplexPReLU` — torch.where positive/negative on each chunk; supports shared (`num_parameters=1`) or per-channel modes.

#### 6.2.4 Uncertainty heads (`architecture/unc_archs.py`)

`Uncertainty` is a 3-Conv → `LocallyConnected2d(64 → 2, output_size=[s//2+1, s])` head producing variance predictions, post-activated with `GeneralELU(add=1+1e-7)` to ensure positivity.

`UncertaintyWrapper` extends `SRResNet34`, runs the base prediction, concatenates `[pred, input]` (4 channels) into the uncertainty head, and stitches outputs to `[pred_amp, unc_amp, pred_phase, unc_phase]`. Note: `super.forward` (vs. `super().forward`) on line 67 looks like a bug — calls the unbound method.

### 6.3 `src/radionets/training/`

| File | LOC | Purpose |
|---|---:|---|
| `scripts/start_training.py` | 178 | Click CLI; loads TOML, builds databunch, defines arch + learner, runs `learn.fit`/`fine_tune`/`lr_find`/`plot_loss`, handles `KeyboardInterrupt` via `pop_interrupt` |
| `utils.py` | 152 | `create_databunch`, `read_config`, `check_outpath`, `define_arch`, `pop_interrupt`, `end_training`, `get_normalisation_factors` |

`define_arch` dispatches by `arch_name`: anything containing `"filter_deep"`, `"resnet"`, or `"Uncertainty"` is instantiated with `(img_size,)`; otherwise zero-arg.

`get_normalisation_factors` iterates `data.train_ds`, accumulates per-batch real/imag means and stds, then returns `{mean_real, mean_imag, std_real, std_imag}` (used when `normalize == "mean"`).

### 6.4 `src/radionets/evaluation/`

| File | LOC | Highlights |
|---|---:|---|
| `scripts/start_evaluation.py` | 164 | Dispatches inspection/eval flags from TOML to `evaluate_*` functions |
| `utils.py` | 844 | dataloader builders, normalization, ifft, symmetry application, sampling, numba truncated-normal generators |
| `train_inspection.py` | 839 | All `create_*` plot routines and `evaluate_*` metrics; `save_sampled` for uncertainty pipelines |
| `jet_angle.py` | 154 | PCA-based jet-angle estimator (`pca`, `calc_jet_angle`, `bmul`) |
| `blob_detection.py` | 86 | `calc_blobs` (uses `skimage.feature.blob_log`), `crop_first_component`, `corners` |
| `contour.py` | 93 | `compute_area_ratio`, `area_of_contour`, `analyse_intensity` |
| `dynamic_range.py` | 73 | RMS-based dynamic-range computation over corner boxes |
| `jets.py` | 116 | Astropy `Gaussian2D` fitting (`fitgaussian_crop`, `fitgaussian_iterativ`) |
| `pointsources.py` | 123 | `flux_comparison`, `get_length_extended`, `get_length_point` for source-list metrics |

#### 6.4.1 Truncated normal sampling (`evaluation/utils.py` lines 468–522)

```python
@vectorize(["float64(float64, float64, float64, float64)"], target="cpu")
def tn_numba_vec_cpu(mu, sig, a, b): ...

@vectorize(["float64(float64, float64, float64, float64)"], target="parallel")
def tn_numba_vec_parallel(mu, sig, a, b): ...
```

`trunc_rvs(mu, sig, num_samples, mode, target, nthreads)` selects:

* `mode="amp"` → `[0, ∞)`
* `mode="phase"` → `[-π, π]`
* `mode in {"real","imag"}` → `(-∞, ∞)`

Used by `sample_images` to draw `(mean, std) → image` samples from the network's predicted distributions, then applies `symmetry` and inverse FFT and reduces to `mean`/`std` over `num_samples=100` (`evaluation/train_inspection.py::save_sampled`, lines 483–564).

#### 6.4.2 Symmetry stitching (`evaluation/utils.py` lines 403–465)

`symmetry(image, key)` builds the lower half of an image from rotated/conjugate-flipped upper half. For `key=="unc"` the second channel is **not** sign-flipped (since variance is symmetric); otherwise the imaginary/phase channel is negated. `apply_symmetry(img_dict)` first F.pads `(0, 0, 0, half-5)` and then calls `symmetry`. This mechanism enables training on **half-sized images** (introduced in v0.2.0, see `CHANGES.rst` lines 216–237) and reconstructing full images at inference.

#### 6.4.3 Evaluation metric routines (`train_inspection.py`)

| Function | Output | Method |
|---|---|---|
| `evaluate_viewing_angle`   | jet-angle offset histogram | PCA over high-flux pixels |
| `evaluate_dynamic_range`   | DR truth/pred histograms   | RMS via 4 corner boxes |
| `evaluate_ms_ssim`(`_sampled`) | MS-SSIM histogram   | `pytorch_msssim.ms_ssim`, win=7 |
| `evaluate_intensity`(`_sampled`) | sum/peak ratio   | `analyse_intensity` 5%-threshold mask |
| `evaluate_mean_diff`       | flux % diff in core component | `calc_blobs` + `crop_first_component` |
| `evaluate_area`(`_sampled`) | pred/truth contour area | matplotlib `plt.contour` 5% level |
| `evaluate_point`           | point-source flux comparison | `flux_comparison` |
| `evaluate_unc`             | uncertainty histograms | masked source pixels |
| `evaluate_gan_sources`     | GAN-source diagnostics | source vs. truth pixel statistics |
| `save_sampled`             | `sampled_imgs_<model>.h5` | trunc-normal sampling × 100 |

### 6.5 `src/radionets/simulations/`

| File | LOC | Role |
|---|---:|---|
| `simulate.py` | 80 | `create_fft_images` and `sample_fft_images` orchestrators |
| `gaussians.py` | 411 | Random Gaussian-component jet/source generation |
| `point_sources.py` | 218 | Point-like Gaussians + extended jets, source-list output |
| `uv_simulations.py` | 453 | `Source`, `Antenna`, `get_uv_coverage`, `create_mask`, `sample_freqs` |
| `sampling.py` | 96 | Iterates `train/valid/test` bundles, applies sampling mask, optional interpolation, saves outputs |
| `utils.py` | 269 | `check_outpath`, `read_config`, `prepare_fft_images`, noise helpers, `interpol` |
| `uv_plots.py` | 298 | UV/baseline plotting helpers (Cartopy + Matplotlib) |
| `visualize_simulations.py` | 369 | Antenna-on-globe rendering, `OrBu` colormap (legacy) |
| `layouts/layouts.py` | 11 | `vlba()` → reads `vlba.txt` antenna positions |
| `layouts/vlba.txt` | 11 | 10 VLBA stations: BR, FD, HN, KP, LA, MK, NL, OV, PT, SC (geocentric XYZ + dish_dia) |
| `scripts/simulate_images.py` | 47 | Click CLI invoking simulate + sample |

#### 6.5.1 Gaussian source generation (`simulations/gaussians.py` lines 81–303)

* `gauss_paramters()` randomises:
  * components: `randint(4, 7)`
  * peak amplitude: `randint(0,100) * random() / 10`, logarithmic decrease over components
  * spacing: linear `(0,5,10,…) px` along jet, rotation `0–360°`, sides `0|1` (one- or two-sided)
* `gaussian_component` adds a 2D Gaussian to a meshgrid using a 2×2 rotation matrix.
* `create_ext_gauss_bundle(grid)` produces `bundle_size` images via `gaussian_source(grid)`.
* Final pipeline: `simulate_gaussian_sources` → optional noise (`add_noise`) → `np.fft.fft2` (with `fftshift/ifftshift`) → optional `add_white_noise` → `save_fft_pair(path, bundle_fft, bundle, list_sources)`.

#### 6.5.2 Point sources + extended jets (`simulations/point_sources.py`)

`create_point_source_img(...)` per bundle:

1. Generate extended Gaussian jet (`gaussian_source`) padded by random `x_off, y_off`.
2. Add 2–4 randomized point sources (`create_gauss`).
3. Build `source_list = [list_x, list_y, list_sx, list_sy, list_tag]` where `tag=0` → point, `tag=1` → extended.
4. Save HDF5 with keys `x{i}` (FFT), `y{i}` (image), `z{i}` (source list).

#### 6.5.3 UV simulation (`simulations/uv_simulations.py`)

* `Source(lon, lat).propagate(num_steps)` emits propagated `(lon, lat)` arrays simulating sky tracking; `mod_delete(a, n, m)` punches holes for multi-pointing scenarios.
* `Source.to_ecef()` uses `astropy.coordinates.EarthLocation`.
* `Antenna(X, Y, Z)` exposes `to_geodetic`, `to_enu`, `get_baselines`, `get_uv` (returns `(u, v, steps)`).
* `get_uv_coverage(source, antenna, multi_channel, bandwidths, iterate)` optionally repeats `(u, v)` across `bandwidths` channels with linearly scaled wavelengths.
* `create_mask(u, v, size=64)` builds a 2D histogram, zeros out the central low-frequency square (`size//2 ± 2|3`), and rotates 90°. The mask is applied multiplicatively to the FFT image.
* `sample_freqs` in `train/valid/test` and either fixed `(lon, lat, steps)` or randomised per-image masks.

#### 6.5.4 Sampling pipeline (`simulations/sampling.py`)

```python
for mode in ["train", "valid", "test"]:
    for path in get_fft_bundle_paths(data_path, "fft", mode):
        fft, truth = open_fft_bundle(path)              # or open_bundle_pack for points
        fft_scaled       = prepare_fft_images(fft.copy(), amp_phase, real_imag)
        fft_scaled_truth = prepare_fft_images(np.fft(truth), amp_phase, real_imag)
        fft_samp         = sample_freqs(fft_scaled, antenna_config, ...)
        if interpolation: fft_samp = interpol(fft_samp)
        save_fft_pair(out, fft_samp, fft_scaled_truth, source_list)
```

`prepare_fft_images` (in `simulations/utils.py` lines 156–169) computes `amp = log10(|F| + 1e-10)/10 + 1`, leaving `phase = angle(F)`; for 511-px images applies an amplitude>0.1 mask to phase.

### 6.6 `src/radionets/plotting/`

| File | LOC | Highlights |
|---|---:|---|
| `_puor.py` | 270 | Hand-coded `PuOr` and `PuOr_r` `LinearSegmentedColormap`s; registered globally on import |
| `hist.py` | 411 | `Hist` class — methods `jet_angles`, `dynamic_ranges`, `ms_ssim`, `area`, `mean_diff`, etc. |
| `visualization.py` | 873 | `plot_target`, `plot_inp_tar`, `visualize_with_fourier(_diff)`, `visualize_source_reconstruction`, `visualize_uncertainty`, `visualize_sampled_unc`, `plot_contour`, `plot_length_point`, `plot_fitgaussian` |
| `inspection.py` | 98 | `plot_loss`, `plot_lr`, `plot_lr_loss` for `Learner` artefacts |

### 6.7 `src/radionets/tools/`

* `quickstart.py` (89 LOC) — copies bundled TOML to user-supplied destination; uses `rich_click` for nicer help.

---

## 7. Configuration Reference

### 7.1 Simulation TOML schema (`examples/default_simulation_config.toml`)

```toml
[mode]   quiet = true
[paths]  data_path = "./example_data/"   data_format = "h5"
[gaussians]      simulate = true   num_components = [4, 10]
[point_sources]  simulate = false  add_extended = false
[image_options]
  bundles_train = 5  bundles_valid = 1  bundles_test = 1
  bundle_size   = 200  img_size = 63
  noise = false  noise_level = 5  white_noise = false
  mean_real = 0.85 std_real = 0.0425 mean_imag = 0.2 std_imag = 0.01
[sampling_options]
  fourier = true real_imag = false amp_phase = true
  antenna_config = "vlba"   specific_mask = true
  lon = -80  lat = 50  steps = 50  bandwidths = 1  multi_channel = false
  keep_fft_files = true  source_list = false  compressed = false  interpolation = false
```

### 7.2 Training TOML schema (`configs/radionets_default_train_config.toml`)

```toml
[mode]    quiet = true  gpu = false
[logging] comet_ml = true  project_name = "VLA"  plot_n_epochs = 2  scale = true
[paths]   data_path = "./example_data/"  model_path = "./build/example_model/example.model"  pre_model = "none"
[general]
  fourier = true  amp_phase = true  normalize = false  source_list = false
  arch_name = "filter_deep"                     # Note: filter_deep was removed; tests use SRResNet18
  loss_func = "splitted_L1"  num_epochs = 5   inspection = true  output_format = "png"
  switch_loss = false  when_switch = 25
[hypers]   batch_size = 100  lr = 1e-3
[param_scheduling]
  use = true  lr_start = 7e-2  lr_max = 3e-1  lr_stop = 5e-2  lr_ratio = 0.25
```

### 7.3 Evaluation TOML schema (`examples/default_eval_config.toml`)

```toml
[paths]   data_path  model_path  model_path_2  ("none" disables a 2nd model)
[general] fourier amp_phase source_list arch_name arch_name_2 output_format diff
[inspection] visualize_prediction visualize_source_reconstruction visualize_contour
             visualize_dynamic_range visualize_ms_ssim sample_uncertainty
             visualize_uncertainty random num_images
[eval]    batch_size save_vals save_path
          evaluate_viewing_angle evaluate_dynamic_range evaluate_ms_ssim
          evaluate_intensity evaluate_mean_diff evaluate_area
          evaluate_point predict_grad evaluate_gan
```

---

## 8. Input & Output Formats

* **HDF5 bundles** (saved by `save_fft_pair` in `core/data.py` lines 164–173):

  | Key prefix | Content |
  |---|---|
  | `x{i}` | Sampled / sub-sampled Fourier image (2 ch) |
  | `y{i}` | Truth — image-space or full-Fourier |
  | `z{i}` | Per-source list `[x, y, sx, sy, tag]` (only if `source_list=true`) |

  File-name convention (see `simulations/sampling.py` line 87, `train_inspection.py` line 168): `fft_{mode}{N}.h5`, `samp_{mode}{N}.h5`, `predictions_{model_stem}.h5`, `sampled_imgs_{model_stem}.h5`.

* **Compressed npz**: `sampling.py` lines 89–92 use `numpy.savez_compressed(out, x=fft_samp, y=fft_scaled)` if `compressed=true`.

* **`.model` checkpoint** (PyTorch `torch.save` dict, `core/model.py` lines 71–104):

  ```
  {model, opt, epoch, iters, vals, train_loss, valid_loss, lrs, norm_dict}
  ```

  `norm_dict` carries either `{mean_real, mean_imag, std_real, std_imag}`, `{max_scaling: 0}`, `{all: 0}`, or `{}` depending on `Normalize.mode`.

* **Predictions HDF5** (`evaluation.utils.save_pred`, lines 365–369): keys `pred`, `inp`, `true`, optionally `unc`, `indices`.

* **Antenna layouts**: plain text `# x y z dish_dia station` rows. Currently only `vlba.txt` is shipped.

---

## 9. Testing & Tutorials

### 9.1 Test layout (`tests/`)

| File | Purpose | Notes |
|---|---|---|
| `conftest.py` | Session-scoped cleanup of `./tests/build/` | Auto-used fixture |
| `test_simulation.py` | `radionets-simulation` end-to-end with `simulate.toml` | Marked `pytest.mark.order("first")` |
| `test_training.py` | DataBunch / `define_learner` / `radionets-training` / `save_model` / pre-trained reload / `plot_loss` | Sequential dependencies on the simulation step |
| `test_evaluation.py` | `TestEvaluation` class (marked `order("last")`) covering `get_images`, `get_prediction`, `get_ifft`, `area_of_contour`, jet-angle pipeline (`im_to_array_value`, `bmul`, `pca`, `calc_jet_angle`), blob detection, GAN sources, symmetry, sample_images, normalization, and a full CLI invocation. Plus a parametrized `test_trunc_rv` over `(mode, target)` pairs validating Numba vectorized truncated-normal vs `scipy.stats.truncnorm`. |
| `test_architecture_layers.py` | Extensive unit tests for `LocallyConnected2d`, `ComplexConv2d`, `ComplexInstanceNorm2d`, `ComplexPReLU` (init, forward, gradient flow, device, edge cases) |
| `tests/model/` | Fixture HDF5 + `.model` files: `predictions_unc.h5`, `samp_test0.h5`, `instance.model`, `model_eval.model` |

`pyproject.toml` line 136: `addopts = "--verbose"`.

### 9.2 Tutorial notebooks (`examples/`)

| Notebook | Topic |
|---|---|
| `00_observation_simulation.ipynb` | UV-coverage and observation simulation walkthrough |
| `01_dataset_simulation.ipynb` | Generating training/validation/test HDF5 bundles |
| `02_model_training.ipynb` | Training pipeline, callbacks, lr_find |
| `03_model_evaluation.ipynb` | Loading checkpoints, generating evaluation plots |
| `07_benchmark_testset.ipynb` | Benchmarking on a held-out test set |
| `Archs&Losses.ipynb` | Architecture & loss-function exploration |
| `nb.py` | Notebook helper module |
| `matplotlib_rcs/` | `paper_*.rc` style sheets (`paper_long`, `paper_small`, `paper_large`, `paper_large_3`, `paper_large_3_2`) |

---

## 10. Integration & Extension Points

### 10.1 Adding a new architecture

1. Implement an `nn.Module` subclass in `architecture/archs.py` (or `unc_archs.py`). Forward must return `{"pred": tensor}` to be loss-compatible.
2. Add the class to `architecture/__init__.py` `__all__`.
3. The dispatch in `training/utils.py::define_arch` (and `evaluation/utils.py::load_pretrained_model`) instantiates the class either with `(img_size,)` or zero-arg based on substring match (`"filter_deep"`, `"resnet"`, `"Uncertainty"`). Names containing none of these will be called with `()`.

### 10.2 Adding a new loss

Add a function in `core/loss_functions.py` using signature `loss(x, y)` where `x` is the model output dict (`x["pred"]`) and `y` the target. List the function name in `__all__`. It is then selectable from TOML via `[general] loss_func = "your_loss"`. To use **feature loss** instead, set `loss_func = "feature_loss"` and provide an `init_feature_loss()` (referenced in `core/learner.py` line 89 but **not implemented in the current source** — known TODO).

### 10.3 Adding a new callback

* Subclass `fastai.callback.core.Callback`.
* Set `_order` to position relative to existing callbacks (3 = after CUDA, 4 = Normalize, 5 = SwitchLoss, 95 = late).
* Append to the list inside `core/learner.py::define_learner`.

### 10.4 Adding a new antenna layout

Add a function `<name>()` in `simulations/layouts/layouts.py` returning `np.array([X, Y, Z])`. Set `[sampling_options] antenna_config = "<name>"`. Layouts are looked up via `getattr(layouts, ant_config)` (`simulations/uv_simulations.py` line 418).

### 10.5 Adding new evaluation metrics

* Add an `evaluate_<name>(conf)` to `evaluation/train_inspection.py` using `preprocessing(conf)` for boilerplate (model load + DataLoader + norm_dict).
* Wire a config flag into `[eval]` inside the TOML and `evaluation/utils.py::read_config`.
* Dispatch from `evaluation/scripts/start_evaluation.py::main`.

---

## 11. Notable Internals & Caveats

| Topic | Detail |
|---|---|
| **Image symmetry trick** | Models trained on a half image (`s//2 + 1, s`) and the bottom half is reconstructed via Hermitian symmetry of FFTs. Switched in v0.2.0; uncertainty channels use a different sign convention (`evaluation/utils.py::symmetry`). |
| **Custom PuOr cmap** | Registered globally on `import radionets`; used as `cmap="radionets.PuOr"`. Defined in `plotting/_puor.py`. |
| **`add_safe_globals([L])`** | Whitelists fastcore’s `L` for `torch.load` (post-2.5 safe-load enforcement). |
| **CometCallback hard-coded** | Always instantiates a `comet_ml.Experiment`. Disable via `[logging] comet_ml = false`. |
| **Hatch shared-data** | `configs/radionets_default_train_config.toml` is installed to `<env>/share/configs/` and read by `quickstart`. |
| **FFT convention** | `np.fft.ifftshift(np.fft.fft2(np.fft.fftshift(...)))` for forward, `fftshift(ifft2(ifftshift(...)))` for inverse. Magnitude is taken at the end (`abs`), discarding final phase. |
| **Numba acceleration** | `tn_numba_vec_{cpu,parallel}` provide CPU and parallel truncated-normal samplers (added in v0.3.0, `CHANGES.rst` line 213). |
| **Half-/full-image dispatch** | `apply_symmetry` is invoked when `pred.shape[-2] < pred.shape[-1]` (`evaluation/train_inspection.py` line 108, `evaluation/utils.py` line 812). |
| **`fitgaussian_iterativ`** | Hard-coded 10-iteration cap and `data.max() > 0.05` threshold for source extraction (`evaluation/jets.py`). |
| **Quickstart message** | Logs `pyvisgen` instead of `radionets` (`tools/quickstart.py` line 46). Cosmetic. |

---

## 12. Known Bugs / TODOs

| Location | Issue |
|---|---|
| `architecture/blocks.py` line 203 | Typo `nn.Rropout` instead of `nn.Dropout` — only triggered when `dropout=True` on `SRBlock` |
| `architecture/unc_archs.py` line 67 | `super.forward(x)` should be `super().forward(x)` — `UncertaintyWrapper` is currently broken |
| `core/learner.py` line 89 | `loss_functions.init_feature_loss()` referenced but not defined |
| `evaluation/scripts/start_evaluation.py` line 141 | `PredictionImageGradient(test_data=...)` kwarg mismatches `__init__(validation_data=...)` |
| `simulations/gaussians.py` line 409 | `np.float` deprecated — will fail on NumPy 2.x |
| `simulations/point_sources.py` line 122 | Same `np.float` issue |
| `README.md` | Suggests `mamba env create -f environment.yml`; only `environment-dev.yml` ships |
| `README.md` | Documents `radionets_*` (underscore) script names; entry-points use dashes |
| `simulations/README.md` lines 14–20 | Documented TODOs: power-law per jet component, Lorentz factor, FRI/FRII variation, image-size scaling |

---

## 13. Git History (last 25 commits)

```
d165444 Merge pull request #209 from radionets-project/fix-callbacks
fc3ecc5 [pre-commit.ci] auto fixes from pre-commit.com hooks
93cc6aa add change log fragment
52252c7 Remove "all" from plt.close() calls
85c5687 Handle learn.normalize = False in model.py
5fb1a3c Merge pull request #205 from radionets-project/env-dev
3c19db1 Merge pull request #206 from radionets-project/ignore-up038
d3709c9 Merge pull request #208 from radionets-project/pre-commit-ci-update-config
...
f8ce30a Merge pull request #198 from radionets-project/complex_archs
f382dd6 Move dropout to blocks list
1a6b245 Merge pull request #200 from radionets-project/update_symmetry
```

Released tags: `v0.1.0, v0.1.1, v0.1.12, v0.1.14, v0.1.16, v0.1.18, v0.2.0, v0.3.0, v0.4.0, v0.4.1`.

---

## 14. Citing

Per `CITATION.cff` and `README.md`, cite the A&A paper:

> Schmidt et al., **A&A**, DOI [`10.1051/0004-6361/202142113`](https://www.aanda.org/component/article?access=doi&doi=10.1051/0004-6361/202142113), and the Zenodo record.

Authors (CITATION.cff): K. Schmitz, A. Knierim, P.-S. W. Blomenkamp, S. Fröse, F. Geyer, O. Locke, A. Poggenpohl, E. Zaldivar.

---

*End of reference.*
