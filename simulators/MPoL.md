# MPoL — Million Points of Light

> Exhaustive technical reference for the **MPoL** package as vendored at
> `simulators/MPoL/`.
> Sources cited inline as relative paths under `simulators/MPoL/`.
> All claims trace to source code, `pyproject.toml`, `README.md`, `LICENSE`,
> `docs/changelog.md`, `docs/units-and-conventions.md`, `paper/paper.md`, and
> the git log/tags of the vendored submodule.

---

## 1. Overview

**MPoL** ("Million Points of Light") is a **PyTorch library** for
**Regularized Maximum Likelihood (RML) imaging and Bayesian inference** of
interferometric visibility datasets, primarily aimed at facilities such as
ALMA and the JVLA (`README.md`, lines 1–16; `docs/index.md`, lines 1–14).

Quoting the upstream README (`simulators/MPoL/README.md`, lines 8–12):

> MPoL is a PyTorch *library* built for Regularized Maximum Likelihood (RML)
> imaging and Bayesian Inference with datasets from interferometers like the
> Atacama Large Millimeter/Submillimeter Array (ALMA) and the Karl G. Jansky
> Very Large Array (VLA). [...] MPoL is *not* an imaging application nor a
> pipeline.

Key design facts (extracted from source):

- All core modeling primitives are subclasses of `torch.nn.Module`
  (`BaseCube`, `HannConvCube`, `ImageCube`, `FourierCube`, `NuFFT`,
  `NuFFTCached`, `GriddedDataset`, `GriddedNet`).  
  See `src/mpol/images.py` lines 21, 107, 194, 332, 413; `src/mpol/fourier.py`
  lines 14, 111, 436; `src/mpol/datasets.py` line 14;
  `src/mpol/precomposed.py` line 9.
- Forward-mode autodiff (`loss.backward()`) is the imaging engine.
- Non-uniform FFT is delegated to **TorchKbNufft**
  (`src/mpol/fourier.py`, lines 6, 139, 419, 493).
- Gridding helpers are CPU/numpy and use **fast-histogram**
  (`src/mpol/gridding.py`, lines 10, 312).
- Image cubes are stored in physical units of $\mathrm{Jy\,arcsec^{-2}}$
  (`src/mpol/images.py` lines 416–417, 492; `docs/units-and-conventions.md`
  line 48).
- Spatial frequencies are in cycles per radian (i.e. units of
  $\lambda$, *not* $\mathrm{k}\lambda$ since v0.3.0;
  `docs/changelog.md` lines 14, 22–31).

### License, version, language

| Item | Value | Source |
|------|-------|--------|
| License | MIT | `LICENSE` lines 1–3 |
| Copyright | Ian Czekala and contributors, 2019–2025 | `LICENSE` line 3 |
| Language | Python ≥ 3.10 (pyproject says ≥ 3.8 but docs require 3.10) | `pyproject.toml` line 26; `docs/installation.md` line 3 |
| Latest tag (vendored) | `v0.3.0-alpha` | `git tag` output |
| Last commit (vendored) | `24c0bbc` "Update paper.md authors with Jane Huang, finally!" | `git log -1` |
| Build backend | `hatchling.build` (with `hatch-vcs`) | `pyproject.toml` lines 1–3 |
| Distribution | PyPI as `MPoL` | `pyproject.toml` line 24 |
| Zenodo concept DOI | `10.5281/zenodo.3594081` | `README.md` lines 5, 38 |
| Pinned Zenodo record (latest in code) | `10064221` | `src/mpol/__init__.py` line 1 |

### Authors

Ian Czekala, Jeff Jennings, Brianna Zawadzki, Ryan Loomis, Kadri Nizam,
Megan Delamer, Kaylee de Soto, Robert Frazier, Hannah Grzybowski, Jane Huang,
Mary Ogborn, Tyler Quinn, Kristin Hopley
(`CONTRIBUTORS.md`; `paper/paper.md` lines 13–62; `README.md` lines 22–32).

### Companion publications

- Zawadzki et al. 2023, *PASP* **135**, 064503 — DOI
  `10.1088/1538-3873/acdf84` (`README.md` lines 44–61).
- JOSS draft `paper/paper.md` (`paper/paper.md` lines 1–80).

### Companion repositories

Tutorials and end-to-end workflows live in a separate repo
`MPoL-dev/examples` (`docs/index.md` line 14;
`docs/changelog.md` line 6).

---

## 2. Repository layout

Directory tree of `simulators/MPoL/` (depth-2):

```
MPoL/
├── CODE_OF_CONDUCT.md
├── CONTRIBUTORS.md
├── LICENSE
├── README.md
├── pyproject.toml
├── docs/
│   ├── api/                     # Sphinx API stubs (one file per module)
│   ├── _static/                 # mermaid sources, baseline tables, fftshift figs
│   ├── background.md            # rml-intro / units companion
│   ├── changelog.md
│   ├── conf.py
│   ├── developer-documentation.md
│   ├── favicon.ico, logo.png
│   ├── index.md
│   ├── installation.md
│   ├── units-and-conventions.md
│   ├── Makefile / make.bat
├── paper/
│   ├── paper.md, paper.bib, fig.pdf      # JOSS submission
├── src/
│   └── mpol/
│       ├── __init__.py          # only defines zenodo_record = 10064221
│       ├── constants.py
│       ├── coordinates.py       # GridCoords (one of the two pivot classes)
│       ├── crossval.py          # k-fold splitters
│       ├── datasets.py          # GriddedDataset, Dartboard
│       ├── exceptions.py
│       ├── fourier.py           # FourierCube, NuFFT, NuFFTCached, generate_fake_data
│       ├── geometry.py          # flat_to_observer / observer_to_flat
│       ├── gridding.py          # GridderBase, DataAverager, DirtyImager
│       ├── images.py            # BaseCube, HannConvCube, GaussConv*, ImageCube
│       ├── input_output.py      # ProcessFitsImage
│       ├── losses.py            # χ², log-likelihood, regularizers
│       ├── onedim.py            # radial profiles
│       ├── plot.py              # diagnostic figures
│       ├── precomposed.py       # GriddedNet (was SimpleNet)
│       ├── training.py          # train_to_dirty_image (rest is commented)
│       ├── utils.py             # cube reshapers, gaussian helpers
│       ├── tests.mplstyle
│       └── data/mock_data.npz
└── test/                        # pytest test suite (see §10)
```

### Source line counts (`wc -l src/mpol/*.py`)

| File | Lines |
|------|------|
| `gridding.py` | 1186 |
| `plot.py` | 1167 |
| `utils.py` | 648 |
| `fourier.py` | 644 |
| `crossval.py` | 603 |
| `images.py` | 582 |
| `losses.py` | 551 |
| `training.py` | 394 |
| `coordinates.py` | 362 |
| `datasets.py` | 296 |
| `onedim.py` | 170 |
| `geometry.py` | 161 |
| `precomposed.py` | 109 |
| `input_output.py` | 100 |
| `exceptions.py` | 21 |
| `constants.py` | 11 |
| `__init__.py` | 1 |
| **Total** | **7006** |

---

## 3. Installation & dependencies

Source: `simulators/MPoL/pyproject.toml`, `simulators/MPoL/docs/installation.md`.

### 3.1 Runtime dependencies (`pyproject.toml` lines 12–21)

```toml
dependencies = [
    "numpy",
    "fast-histogram",
    "scipy",
    "torch>=1.8.0",
    "torchvision",
    "torchaudio",
    "torchkbnufft",
    "astropy",
]
```

### 3.2 Optional `dev` extras (`pyproject.toml` lines 28–55)

`pytest, pytest-cov, matplotlib, requests, astropy, tensorboard, mypy,
frank>=1.2.1, sphinx>=7.2.0, jupytext, ipython!=8.7.0, nbsphinx,
sphinx_book_theme>=0.9.3, sphinx_copybutton, jupyter, nbconvert,
sphinxcontrib-mermaid>=0.8.1, myst-nb, jupyter-cache, Pillow, asdf, pyro-ppl,
arviz[all], visread>=0.0.4, ruff`.

### 3.3 Optional `test` extras (`pyproject.toml` lines 56–66)

`pytest, pytest-cov, matplotlib, requests, tensorboard, mypy, visread,
frank>=1.2.1, ruff`.

### 3.4 Notable consumed packages (by import)

| Package | Used by |
|---------|--------|
| `torch`, `torch.nn`, `torch.fft` | All core modules |
| `torchkbnufft` | `src/mpol/fourier.py` (lines 6, 139, 419, 493) |
| `numpy`, `numpy.fft`, `numpy.typing` | `coordinates.py`, `gridding.py`, `utils.py`, … |
| `fast_histogram` | `src/mpol/gridding.py` line 10 (gridding loop) |
| `scipy` | (declared; not imported in core) |
| `astropy.constants`, `astropy.io.fits`, `astropy.wcs`, `astropy.visualization` | `constants.py`, `images.py:to_FITS`, `input_output.py`, `plot.py` |
| `frank` (optional) | `src/mpol/onedim.py` line 126 (radial deprojection) |
| `visread` (test only) | `test/conftest.py` lines 8, 64 |

### 3.5 Install commands (`docs/installation.md`)

```bash
pip install MPoL                  # stable from PyPI
pip install MPoL==0.2.0           # pinned version
git clone https://github.com/MPoL-dev/MPoL.git && cd MPoL && pip install .
pip install --upgrade MPoL
```

There is **no** `setup.py` (removed in v0.3.0; `docs/changelog.md` line 53);
versioning is via `hatch-vcs` from git tags (`pyproject.toml` lines 72–76).

### 3.6 Hardware

GPU support is automatic via PyTorch — every `nn.Module` (`BaseCube`,
`ImageCube`, `FourierCube`, `NuFFT(Cached)`, `GriddedDataset`, `GriddedNet`)
inherits `.to(device)`. Tensors registered as `register_buffer(...)`
(e.g. `vis`, `mask`, `vis_gridded`, `weight_gridded`, `taper_2D`,
`k_traj`, `real_interp_mat`, `imag_interp_mat`) move with the module.

---

## 4. Architecture

### 4.1 Layered model

The MPoL pipeline is a chain of `nn.Module`s. The `GriddedNet`
precomposed module (`src/mpol/precomposed.py` lines 9–110) wires them
together as:

```
              ┌───────────────────────────┐
              │ GriddedNet (nn.Module)    │
              └────────────┬──────────────┘
   trainable  │            │
   ─────────▶ │ BaseCube  ──▶  HannConvCube  ──▶  ImageCube  ──▶  FourierCube
              │ (Parameter)   (Hann apodise)    (passthrough +    (FFT, packed)
              │                                  to_FITS, flux)
              │                                       │
              │                                       └─ NuFFT (loose vis at uu,vv)
              └───────────────────────────────────────────────────────────────────
```

Loss is computed by feeding the FFT/NuFFT output into a `GriddedDataset`
(or directly into `r_chi_squared` / `log_likelihood`) plus regularizers
from `mpol.losses`.

### 4.2 Sky vs. packed cube convention

Two layouts coexist (`docs/units-and-conventions.md` lines 42–88;
`src/mpol/utils.py` lines 31–106):

- **Sky cube**: orientation as on the sky — RA increases to the *left*,
  no `fftshift`. Used by `BaseCube` initial values, `ImageCube.sky_cube`,
  `to_FITS`, image-plane regularizers (`TV_image`, `TSV`).
- **Packed cube**: `fftshift`-applied along axes `(1,2)`, RA flipped.
  This is the layout consumed by `torch.fft.fftn` and the NuFFT.
  Conversions: `sky_cube_to_packed_cube`, `packed_cube_to_sky_cube`,
  `ground_cube_to_packed_cube`, `packed_cube_to_ground_cube`.
- "Ground cube" = visibility cube as one would `imshow` it
  (centered on `(u,v) = 0`; i.e. `fftshift` applied to packed visibility).
  See `src/mpol/fourier.py:FourierCube.ground_vis` (line 70).

### 4.3 Gridding vs. NuFFT — two RML modes

| Mode | Forward layer | Loss | When to use |
|------|---------------|------|-------------|
| Gridded | `FourierCube` | `r_chi_squared_gridded`, `log_likelihood_gridded` (via `GriddedDataset.forward` mask) | Continuum imaging, full-batch gradient descent |
| Loose / Non-uniform | `NuFFT` (or `NuFFTCached`) | `r_chi_squared`, `log_likelihood`, `neg_log_likelihood_avg` directly on `(uu,vv)` | Stochastic gradient descent, residuals, self-cal |

Both share `GridCoords` for grid geometry and `BaseCube → ImageCube`
for the image-plane parameterization.

### 4.4 Data flow diagram

```
              MS / .npz (uu,vv,weight,re,im) [numpy]
                              │
                              ▼
                  DataAverager / DirtyImager (numpy; fast_histogram)
                              │
                              ▼              ┌─────────────────────────┐
                  GriddedDataset ◀──────────│ register_buffer mask    │
                  (torch.nn.Module, GPU)     │ vis_gridded, weight_grd │
                              │              │ vis_indexed, weight_idx │
                              │              └─────────────────────────┘
                              │  (training loop)
   ┌─────────────────────┐    │
   │ BaseCube parameter  │    │
   │ (softplus/identity) │    │
   └──────────┬──────────┘    │
              ▼               │
        HannConvCube          │
              ▼               │
         ImageCube ────► ImageCube.sky_cube  (regularizers act here)
              ▼               
         FourierCube ────► loss = r_chi_squared_gridded(model_vis, dataset)
              │                + λ_TV * TV_image + λ_S * sparsity + …
              │
              ▼ (or)
            NuFFT(cube, uu, vv) ────► loss on un-gridded visibilities
```

### 4.5 Mermaid diagrams

Authoritative mermaid sources are checked into the docs:

- `docs/_static/mmd/src/GriddedNet.mmd` — the precomposed gridded model
- `docs/_static/mmd/src/SkyModel.mmd`
- `docs/_static/mmd/src/SingleDish.mmd`
- `docs/_static/mmd/src/Parametric.mmd`
- `docs/_static/mmd/src/BaseCube.mmd`
- `docs/_static/mmd/src/ImageCube.mmd`

(Listed by `find` output.)

---

## 5. Public API — by module

### 5.1 `mpol.constants` (`src/mpol/constants.py`, 11 lines)

```python
arcsec : float = π / (180·3600)        # radians per arcsec
deg    : float = π / 180               # radians per degree
kB     : float = astropy.constants.k_B.cgs.value     # erg / K
cc     : float = astropy.constants.c.cgs.value       # cm / s
c_ms   : float = astropy.constants.c.value           # m / s
```

### 5.2 `mpol.exceptions`

Five subclasses of `Exception` (`src/mpol/exceptions.py` lines 4–21):
`CellSizeError`, `WrongDimensionError`, `DataError`,
`DimensionMismatchError`, `ThresholdExceededError`.

### 5.3 `mpol.coordinates`

Pivot class **`GridCoords(cell_size: float, npix: int)`**
(`src/mpol/coordinates.py` lines 16–362). Defines the dual image / Fourier
grid. `npix` must be even and positive (line 86).

Public properties (read-only via `@property` / `@cached_property`):

| Property | Units | Description |
|----------|-------|------|
| `cell_size`, `npix` | arcsec, — | echo of constructor |
| `dl`, `dm` | radians | image pixel width (= `cell_size·arcsec`) |
| `l_centers`, `m_centers` | radians | 1D pixel centers |
| `du`, `dv` | $\lambda$ | UV pixel width = `1 / (npix·dl)` |
| `uv_edges`, `u_edges`, `v_edges` | $\lambda$ | length-`npix+1` edges |
| `u_centers`, `v_centers` | $\lambda$ | length-`npix` centers |
| `u_bin_min/max`, `v_bin_min/max`, `max_uv_grid_value` | $\lambda$ | bounds |
| `img_ext` | arcsec | `[lmax, lmin, lmin, lmax]` for `imshow` |
| `vis_ext` | $\lambda$ | UV bounds for `imshow` |
| `vis_ext_Mlam` | M$\lambda$ | (added in v0.3.0; `docs/changelog.md` line 10) |
| `ground_u_centers_2D`, `ground_v_centers_2D` | $\lambda$ | 2D non-shifted UV grid |
| `ground_q_centers_2D`, `sky_phi_centers_2D` | $\lambda$, rad | polar UV |
| `packed_u_centers_2D`, `packed_v_centers_2D`, `packed_q_centers_2D`, `packed_phi_centers_2D` | $\lambda$/rad | fftshift-applied versions |
| `q_max` | $\lambda$ | outer edge of UV grid |
| `x_centers_2D`, `y_centers_2D`, `sky_x_centers_2D`, `sky_y_centers_2D`, `packed_x_centers_2D`, `packed_y_centers_2D` | arcsec | image-plane meshgrids |

Method **`check_data_fit(uu, vv) -> bool`** raises
`CellSizeError` if the dataset extends outside the supported UV grid
(lines 299–353), suggesting a maximum allowable `cell_size` via
`utils.get_maximum_cell_size`.

`__eq__` is defined: two `GridCoords` are equal iff `cell_size` and
`npix` match (line 355).

### 5.4 `mpol.images`

| Class / function | Inherits | Purpose | Source |
|------------------|----------|---------|--------|
| `BaseCube(coords, nchan=1, pixel_mapping=None, base_cube=None)` | `nn.Module` | Holds the `nn.Parameter` of shape `(nchan, npix, npix)` and applies a positivity-enforcing `pixel_mapping` (default `nn.Softplus()`). Default init is constant `-3` so post-softplus output is near zero. | lines 21–104 |
| `HannConvCube(nchan, requires_grad=False)` | `nn.Module` | `Conv2d` with a fixed 3×3 Hann kernel applied on a sky-format cube (apodisation in the Fourier domain). | lines 107–191 |
| `GaussConvImage(coords, nchan, FWHM_maj, FWHM_min, Omega=0, requires_grad=False)` | `nn.Module` | Convolves a *sky-format* cube with a 2D rotated Gaussian (image-domain `Conv2d`). Raises `RuntimeError` if `FWHM_maj` is unresolved (< 3 pixels). Added in v0.3.0 (`docs/changelog.md` line 7). | lines 194–330 |
| `GaussConvFourier(coords, FWHM_maj, FWHM_min, Omega=0)` | `nn.Module` | Convolves a *packed* cube via UV-plane Gaussian taper; fast for large kernels. Pre-computes `taper_2D` buffer. | lines 332–410 |
| `ImageCube(coords, nchan=1)` | `nn.Module` | Identity passthrough that stores the cube on a `register_buffer("packed_cube")`. Provides `sky_cube`, `flux` (Jy/channel), and `to_FITS(fname, overwrite, header_kwargs)` (uses `astropy.wcs`). | lines 413–536 |
| `uv_gaussian_taper(coords, FWHM_maj, FWHM_min, Omega) -> np.ndarray` | function | Computes a packed UV taper normalized to amplitude 1.0 at the origin. Used by `GaussConvFourier`. | lines 538–583 |

`ImageCube.flux` reads (line 481):

```python
return self.coords.cell_size**2 * torch.sum(self.packed_cube, dim=(1, 2))
```

i.e. converts $\mathrm{Jy/arcsec^2}$ to Jy/channel.

### 5.5 `mpol.fourier`

Three layers + helper:

1. **`FourierCube(coords, persistent_vis=False)`** — gridded FFT layer
   (`src/mpol/fourier.py` lines 14–108). Forward applies
   `cell_size**2 * torch.fft.fftn(packed_cube, dim=(1,2))` (line 65, derived
   from TMS Eqn A8.18). Buffers: `vis`. Properties:
   `ground_vis`, `ground_amp`, `ground_phase`.

2. **`NuFFT(coords, nchan=1)`** — non-uniform FFT layer
   (lines 111–433). Wraps `torchkbnufft.KbNufft`. Forward signature:

   ```python
   forward(packed_cube: Tensor[(nchan,npix,npix)],
           uu: Tensor, vv: Tensor,
           sparse_matrices: bool = False) -> Tensor[complex128, (nchan, nvis)]
   ```

   Behaviour:
   - 1D `uu, vv` of shape `(nvis,)` → "same_uv" mode, parallelizes
     over the *coil* dimension (line 228).
   - 2D `uu, vv` of shape `(nchan, nvis)` → batches over `nchan`
     (line 244). `sparse_matrices=True` is incompatible with batch mode
     and is silently downgraded with a `RuntimeWarning` (lines 403–415).
   - Internally converts $\lambda$ → "rad/sky-pixel" via `_lambda_to_radpix`
     (lines 143–190).

3. **`NuFFTCached(coords, uu, vv, nchan=1, sparse_matrices=True)`** —
   sub-class of `NuFFT` (lines 436–569). Pre-computes
   `k_traj`, `real_interp_mat`, `imag_interp_mat` as buffers; ideal for
   repeated evaluation at fixed `(uu, vv)`.

4. **`generate_fake_data(packed_cube, coords, uu, vv, weight)`** — utility
   (lines 572–644). Returns `(vis_noise, vis_noiseless)` torch
   `complex128` tensors of shape `(nchan, nvis)` after adding Gaussian
   noise with $\sigma = 1/\sqrt{w}$ (real and imaginary parts independently).
   Replaces the deprecated `make_fake_data` (`docs/changelog.md` line 21).

### 5.6 `mpol.gridding`

Numpy-based UV gridding (no PyTorch except for the final hand-off).

| Symbol | Lines | Purpose |
|--------|-------|---------|
| `_check_data_inputs_2d(...)` | 18–78 | Validate shapes, dtypes, weight positivity, no Hermitian pairs |
| `verify_no_hermitian_pairs(uu,vv,data,test_vis=5,test_channel=0)` | 81–186 | Heuristically detect Hermitian-augmented datasets |
| `GridderBase` | 189–556 | Abstract base; computes cell indices, sums via `fast_histogram.histogram2d`, supports `_check_scatter_error` for outlier-weight detection |
| `DataAverager(GridderBase)` | 558–674 | Uniform-weighted gridding for likelihood evaluation; `to_pytorch_dataset(check_visibility_scatter=True, max_scatter=1.2) -> GriddedDataset` |
| `DirtyImager(GridderBase)` | 677–1186 | Diagnostic dirty image / dirty beam producer. Augments dataset with Hermitian conjugates internally (lines 794–811). Supports `weighting={"natural","uniform","briggs"}`, `robust ∈ [-2,2]`, `taper_function`, `unit ∈ {"Jy/beam","Jy/arcsec^2"}` (the latter requires a beam-area calculation; lines 1075–1186). Provides `from_tensors` classmethod for residual visibilities (lines 770–791). |

`DataAverager.to_pytorch_dataset` is the canonical entry point from numpy
data into the PyTorch graph (`src/mpol/gridding.py` lines 631–674).

### 5.7 `mpol.datasets`

| Symbol | Lines | Purpose |
|--------|-------|---------|
| `GriddedDataset(*, coords, vis_gridded, weight_gridded, mask, nchan=1)` | 14–144 | `nn.Module` holding gridded reference data. Buffers: `vis_gridded`, `weight_gridded`, `mask`, plus pre-indexed `vis_indexed`, `weight_indexed` (1D tensors). `forward(modelVisibilityCube)` masks the model FFT to align with the data (handles complex via real/imag split because `masked_select` lacks complex grad support). `add_mask(mask)` further restricts via logical AND. `ground_mask` returns the fftshifted mask. |
| `Dartboard(coords, q_edges=None, phi_edges=None)` | 147–296 | A polar-coordinate UV partitioner used for k-fold cross-validation. Default `q_edges` come from `loglinspace(0, q_max, N_log=8, M_linear=5)`, default `phi_edges` are 8 equal bins on $[0,\pi]$. Provides `get_polar_histogram`, `get_nonzero_cell_indices`, `build_grid_mask_from_cells`. |

### 5.8 `mpol.crossval`

| Class | Lines | Purpose |
|-------|-------|---------|
| `RandomCellSplitGridded(dataset, k=5, seed=None, channel=0)` | 333–440 | Random k-fold splits of UV cells, holding the top 1% highest-weight cells in *every* training set. Iterator yields `(train, test)` `GriddedDataset` pairs. |
| `DartboardSplitGridded(gridded_dataset, k, dartboard=None, seed=None, verbose=True)` | 443–603 | Dartboard-guided splits; the smallest-`q` bin always remains in the training set. Has `from_dartboard_properties` classmethod. |

A `CrossValidate` orchestrator class is **commented out**
(lines 18–331) — the active code only exposes the splitters.

### 5.9 `mpol.losses`

All functions live in `src/mpol/losses.py`. None of them are
methods — every regularizer is a free function that returns a scalar
torch tensor, ready for `loss.backward()`.

| Function | Equation (from docstring) | Lines |
|----------|--------------------------|-------|
| `_chi_squared(model, data, weight)` *private* | $\chi^2 = \sum_i w_i\|V_i - M_i\|^2$ | 10–42 |
| `r_chi_squared(model, data, weight)` | $\chi^2_R = \frac{1}{2N}\chi^2$ (EHT-IV 2019) | 45–94 |
| `r_chi_squared_gridded(modelVisibilityCube, griddedDataset)` | gridded variant | 97–126 |
| `log_likelihood(model, data, weight)` | $\ln\mathcal L = -N\ln 2\pi + \sum_i \ln w_i - \tfrac12\chi^2$ | 129–188 |
| `log_likelihood_gridded(modelVisibilityCube, griddedDataset)` | gridded variant | 191–222 |
| `neg_log_likelihood_avg(model, data, weight)` | $-\ln\mathcal L / (2N)$ — appropriate when self-calibrating amplitudes/weights | 225–262 |
| `entropy(cube, prior_intensity, tot_flux=10)` | $\frac{1}{\zeta}\sum_i I_i \ln(I_i/p_i)$ (EHT-IV) | 265–298 |
| `TV_image(sky_cube, epsilon=1e-10)` | TV in image (l,m) | 301–336 |
| `TV_channel(cube, epsilon=1e-10)` | TV across channel axis | 339–367 |
| `TSV(sky_cube)` | $\sum (I_{l+1}-I_l)^2 + (I_{m+1}-I_m)^2$ (Kuramochi+18) | 370–404 |
| `sparsity(cube, mask=None)` | $L_1$ norm, optionally over `mask==True` pixels | 407–439 |
| `UV_sparsity(vis, qs, q_max)` | $L_1$ on visibilities outside `q_max` | 442–478 |
| `PSD(qs, psd, l)` | Gaussian-process PSD prior | 481–525 |
| `edge_clamp(cube)` | $L_2$ on image-edge pixels | 528–551 |

These align with the standard RML loss menu (entropy, TV, TSV, $L_1$);
v0.3.0 renamed `nll`→`r_chi_squared`, `nll_gridded`→`r_chi_squared_gridded`
(`docs/changelog.md` line 34). The legacy `mpol.losses.nll` no longer exists.

### 5.10 `mpol.precomposed`

Single class **`GriddedNet(coords, nchan=1, base_cube=None)`**
(`src/mpol/precomposed.py` lines 9–110). Composes
`BaseCube → HannConvCube → ImageCube → FourierCube` and additionally
holds a `NuFFT` for `predict_loose_visibilities(uu, vv) -> Tensor`.
The class docstring discourages over-reliance:

> This module is provided as a starting point. However, we recommend
> that you don't get too comfortable using it and instead write your
> own (custom) modules following PyTorch idioms (`precomposed.py`,
> lines 12–17).

Renamed from `SimpleNet` in v0.3.0 (`docs/changelog.md` line 16).

### 5.11 `mpol.training`

Only one *active* function:

```python
train_to_dirty_image(model, imager, robust=0.5, learn_rate=100, niter=1000)
```

(`src/mpol/training.py` lines 6–56). Initializes a `GriddedNet` to a
Briggs-weighted dirty image from a `DirtyImager` using SGD and the
square-root MSE loss. Useful as a warm-start for RML.

The remainder of the file (a `TrainTest` class) is **commented out**
(lines 59–394) — the active workflow is for users to write their own
optimization loop, mirroring PyTorch idioms.

### 5.12 `mpol.geometry`

Two pure functions for sky/observer frame conversion
(`src/mpol/geometry.py` lines 8–161):

```python
flat_to_observer(x, y, omega=0.0, incl=0.0, Omega=0.0) -> (X, Y)
observer_to_flat(X, Y, omega=0.0, incl=0.0, Omega=0.0) -> (x, y)
```

Inputs accept `torch.Tensor` so that the operation is autodifferentiable.
Conventions follow exoplanet-dev/exoplanet's keplerian module
(`src/mpol/geometry.py` lines 28–36).

### 5.13 `mpol.onedim`

| Function | Purpose |
|----------|---------|
| `radialI(icube, geom, chan=0, bins=None)` | Azimuthally-averaged $I(r)$ from an `ImageCube`. `geom` dict keys: `incl`, `Omega`, `omega`, `dRA`, `dDec` (deg & arcsec). |
| `radialV(fcube, geom, rescale_flux, chan=0, bins=None)` | Radial visibility profile $V(q)$ via deprojection; **requires** `frank` (`from frank.geometry import apply_phase_shift, deproject`). |

Both return `(bin_centers, masked_array)` of arcsec / klambda + brightness.

### 5.14 `mpol.input_output`

Single utility class **`ProcessFitsImage(filename, channel=0)`** that
loads CASA-style FITS images and exposes `get_extent`, `get_beam`,
`get_image(beam=True)` (`src/mpol/input_output.py` lines 5–101).
Note `get_image` multiplies pixel values by `1e3` (line 88) — i.e. it
assumes Jy → mJy conversion.

### 5.15 `mpol.utils` — selected helpers

(`src/mpol/utils.py`)

| Function | Lines | Purpose |
|----------|-------|---------|
| `torch2npy(t)` | 11–28 | `t.detach().cpu().numpy()` — for plotting / non-Torch libs |
| `ground_cube_to_packed_cube`, `packed_cube_to_ground_cube` | 31–65 | `fftshift` toggles in vis space |
| `sky_cube_to_packed_cube`, `packed_cube_to_sky_cube` | 68–106 | RA-flip + `fftshift` toggles for image cubes |
| `get_Jy_arcsec2(T_b, nu=230e9)` | 108–130 | Rayleigh–Jeans brightness conversion |
| `loglinspace(start, end, N_log, M_linear=3)` | 133–174 | Dartboard / radial bin edges with linear stretch near zero |
| `fftspace(width, N)` | 177–197 | symmetric coordinate array centered on 0 (even N) |
| `check_baselines(q, min_feasible_q=1e3, max_feasible_q=1e8)` | 200–235 | Warn if user supplied $\mathrm{k}\lambda$ instead of $\lambda$ |
| `get_max_spatial_freq(cell_size, npix)` | 238–260 | Nyquist UV bound for an image grid |
| `get_maximum_cell_size(uu_vv_point)` | 263–276 | Inverse: max cell to Nyquist a given baseline |
| `get_optimal_image_properties(image_width, u, v)` | 279–331 | Auto-design `(cell_size, npix)` for desired image extent and dataset |
| `sky_gaussian_radians`, `sky_gaussian_arcsec` | 334–438 | Analytic 2D rotated Gaussians (sky plane) |
| `fourier_gaussian_lambda_radians`, `fourier_gaussian_lambda_arcsec` | 440–649 | Analytic Fourier counterparts (with full derivation in docstring) |

`mpol.utils` does **not** define `convolve_packed_cube` even though
the v0.3.0 changelog mentions it (`docs/changelog.md` line 9); the
functionality is supplied by `images.GaussConvFourier` instead.

### 5.16 `mpol.plot`

(`src/mpol/plot.py` — 1167 lines)

Top-level plotting helpers:

- `get_image_cmap_norm(image, stretch="power", gamma=1.0, asinh_a=0.02, symmetric=False)`
- `plot_image(...)`
- `vis_histogram_fig(...)`
- `split_diagnostics_fig(splitter, channel=0, save_prefix=None)`
- `train_diagnostics_fig(...)`
- `crossval_diagnostics_fig(cv, title="", save_prefix=None)`
- `vis_1d_fig(...)`
- `radial_fig(...)`

A `get_residual_image` helper exists as commented-out scaffolding
(lines 52–100).

---

## 6. Core mathematics

### 6.1 RIME / measurement equation

MPoL operates on the standard small-field-of-view 2D Fourier transform
of the sky brightness $I(l,m)$ (`docs/units-and-conventions.md` lines 32–36):

$$
\mathcal V(u,v) = \int\!\!\int I(l,m)\,e^{-2\pi i(ul+vm)}\,\mathrm dl\,\mathrm dm
$$

Discrete forward operator on a packed cube of $\mathrm{Jy/arcsec^2}$
(`src/mpol/fourier.py` line 65, derived from TMS Eqn A8.18):

$$
V_{u,v} = (\Delta l)(\Delta m)\;\texttt{FFT}(I_{l,m})
\quad\equiv\quad \texttt{cell\_size**2 * torch.fft.fftn(...)}.
$$

Inverse transform for the dirty image (`docs/units-and-conventions.md`
lines 107–117):

$$
I_{l,m} = U V (\Delta u)(\Delta v)\;\texttt{iFFT}(\mathcal V_{u,v}).
$$

`DirtyImager` implements this as
`coords.npix**2 * np.fft.ifft2(C * vis_gridded)` after `fftshift`-ing
(line 1156–1162), where `C` is the chosen weighting normalization.

### 6.2 Likelihood

For complex visibilities (`losses.py` lines 129–188),

$$
\ln \mathcal L = -N\ln 2\pi + \sum_i \ln w_i
   - \tfrac12 \sum_i w_i |V_i - M_i|^2.
$$

The factor differs from the real-valued multivariate Gaussian by 2
because each visibility is two independent Gaussian samples (real
and imaginary). The reduced $\chi^2_R$ definition uses $1/(2N)$
(`losses.py` lines 53–94).

### 6.3 RML regularizers

| Regularizer | Functional form | Reference |
|-------------|-----------------|-----------|
| Entropy | $\frac{1}{\zeta}\sum_i I_i\ln(I_i/p_i)$ | EHT-IV 2019 |
| TV (image) | $\sum\sqrt{(\Delta_l I)^2+(\Delta_m I)^2+\epsilon}$ | EHT-IV 2019 |
| TV (channel) | $\sum\sqrt{(\Delta_v I)^2+\epsilon}$ | (custom) |
| TSV | $\sum (\Delta_l I)^2 + (\Delta_m I)^2$ | Kuramochi 2018 |
| Sparsity ($L_1$) | $\sum |I_i|$ | (standard) |
| UV sparsity | $L_1$ on $V$ for $q > q_\max$ | (custom) |
| PSD | $\sum \mathrm{psd}/P(q)$ with $P(q)=2\pi\ell^2 e^{-2\pi^2\ell^2 q^2}$ | (custom GP) |
| Edge clamp | $L_2$ on the four image edges | (custom) |

Total loss is built additively by the user, e.g.

```python
loss = losses.r_chi_squared_gridded(model_vis, dataset)
loss = loss + 1e-3 * losses.TSV(model.icube.sky_cube)
loss = loss + 1e-4 * losses.entropy(model.icube.sky_cube,
                                    prior_intensity=1e-10, tot_flux=0.25)
```

### 6.4 NuFFT vs. Gridded forward model

- **Gridded** (`FourierCube`): $\mathcal O(N_\mathrm{pix}^2 \log N_\mathrm{pix})$
  per gradient step; interpolation error embedded in cell averaging.
- **NuFFT** (`NuFFT(Cached)` via TorchKbNufft, Beatty-style Kaiser–Bessel):
  $\mathcal O(N_\mathrm{pix}^2 \log N_\mathrm{pix} + N_\mathrm{vis}\cdot K)$;
  exact at the visibility points, supports batch- and coil-parallelism
  (`src/mpol/fourier.py` lines 226–260, 376–399).

### 6.5 Gridding weights (`DirtyImager._grid_visibilities`,
`gridding.py` lines 813–923)

- `weighting="natural"` → density weights = 1
- `weighting="uniform"` → density weight per visibility =
  `1 / cell_weight[cell]`
- `weighting="briggs"` → CASA definition with
  $f^2 = (5\cdot 10^{-\mathrm{robust}})^2 / (\sum w_g^2 / \sum w_i)$,
  per-cell weight = $1/(1 + w_g f^2)$.

Briggs `robust` accepts $[-2, 2]$ (line 866).

---

## 7. Inputs and outputs

### 7.1 Inputs

| Input style | How it's loaded | Used by |
|-------------|----------------|---------|
| Numpy `(uu, vv, weight, data_re, data_im)` | Direct constructor of `DataAverager` / `DirtyImager` | Most workflows |
| Torch tensors of `(uu, vv, weight, data)` | `DirtyImager.from_tensors(coords, uu, vv, weight, data)` | Residual visibilities |
| FITS image | `mpol.input_output.ProcessFitsImage` | Plot comparisons |
| Bundled `.npz` | `src/mpol/data/mock_data.npz` (loaded via `importlib.resources.files("mpol.data") / "mock_data.npz"`) | Test fixtures (`test/conftest.py` line 15) |

There is **no built-in MS reader**. Conversion from a CASA Measurement
Set is delegated to the companion package
[`visread`](https://github.com/MPoL-dev/visread) (used in the test
suite — `test/conftest.py` line 8). MPoL does not import
`casatools` itself.

### 7.2 Outputs

| Output | Provider |
|--------|----------|
| FITS cube | `ImageCube.to_FITS(fname, overwrite=False, header_kwargs=None)` (uses `astropy.wcs`; `images.py` lines 494–535) |
| Dirty image / dirty beam (numpy cube) | `DirtyImager.get_dirty_image(...)` returns `(image, beam)` |
| `GriddedDataset` (PyTorch buffers) | `DataAverager.to_pytorch_dataset()` |
| Numpy from torch | `mpol.utils.torch2npy(tensor)` |
| Diagnostic figures | `mpol.plot.*` |

Persistence of the model itself uses the standard
`torch.save(model.state_dict(), …)` / `torch.load(...)` pattern
(`training.py` line 38 example: `optimizer = torch.optim.SGD(...)`).

---

## 8. Notable internals

### 8.1 Default tensor dtype

As of v0.3.0, MPoL no longer casts to `float64`/`complex128` internally —
it follows PyTorch's default (`docs/changelog.md` line 8). The user
must ensure all inputs share a precision; otherwise PyTorch promotes.

### 8.2 PyTorch buffers

Module state that is **not** an optimization parameter is stored via
`register_buffer` so it travels with `.to(device)` and shows up in
`state_dict`:

| Module | Buffer(s) |
|--------|-----------|
| `FourierCube` | `vis` (default non-persistent) |
| `NuFFTCached` | `k_traj`, optionally `real_interp_mat`, `imag_interp_mat` |
| `ImageCube` | `packed_cube` |
| `GaussConvFourier` | `taper_2D` |
| `GriddedDataset` | `vis_gridded`, `weight_gridded`, `mask`, `vis_indexed`, `weight_indexed` |

### 8.3 Hermitian-pair handling

- Loose-input gridders **forbid** Hermitian-augmented datasets and
  raise `DataError` if detected
  (`gridding.py:verify_no_hermitian_pairs` lines 81–186).
- `DirtyImager` *internally* augments with conjugates via
  `@property` accessors (lines 793–811) so the inverse FFT yields
  a real image.

### 8.4 Visibility-scatter sanity check

`GridderBase._check_scatter_error` estimates the per-cell standard
deviation of visibility residuals (in $\sigma$ units). If the median
exceeds `max_scatter` (default 1.2), `DataAverager.to_pytorch_dataset`
raises `RuntimeError` and `DirtyImager.get_dirty_image` emits a
`RuntimeWarning` (`gridding.py` lines 418–540, 631–662, 1075–1145).

### 8.5 NuFFT interpolation modes

`NuFFT.forward(..., sparse_matrices=False)` uses table interpolation
(default, fastest); `True` uses pre-computed sparse interpolation
matrices (more accurate, no batch parallelism in TorchKbNuFFT
v1.4.0 — automatic warning + downgrade path,
`src/mpol/fourier.py` lines 403–415).

### 8.6 Mock data fixture

`src/mpol/data/mock_data.npz` ships a downsampled IM Lup dataset
used by the test fixtures (`test/conftest.py` lines 14–32, 53–124).
External Zenodo records (controlled by `mpol.zenodo_record = 10064221`)
back the 1D mock disk used by `radialV` tests (`test/conftest.py`
lines 159–168).

### 8.7 Plotting style

`src/mpol/plot.py` and tests load the bundled
`src/mpol/tests.mplstyle` via `plt.style.use("mpol.tests")`
(`test/conftest.py` line 12).

---

## 9. Extension points

### 9.1 Custom regularizers

Any function `loss(*tensors) -> torch.Tensor` is admissible — there is no
class hierarchy. Compose by adding to the total `loss` before
`loss.backward()` (see `losses.py` for templates).

### 9.2 Custom modules

Sub-class `torch.nn.Module`. The convention is:

- Take a `coords: GridCoords` argument.
- Produce a packed cube (image side) or a packed visibility cube.
- Use `register_buffer` for non-trainable state.

The README specifically positions the package as a *library*: users
extend it by writing new `nn.Module`s, not by configuring a pipeline
(`README.md` lines 8–12; `docs/index.md` lines 6–10).

### 9.3 Custom samplers / inference

`pyro-ppl` and `arviz[all]` are listed as `dev` extras
(`pyproject.toml` lines 51–52) — the developer documentation calls out
Pyro as the supported avenue for variational inference and HMC.

### 9.4 Custom cross-validation splitters

Implement an iterator that yields `(train_set, test_set)` pairs of
`GriddedDataset` (mirroring `RandomCellSplitGridded` /
`DartboardSplitGridded`).

---

## 10. Tests, tutorials, examples

### 10.1 Test layout (`simulators/MPoL/test/`)

Files (one module per source-module):

| Test | Targets |
|------|---------|
| `coordinates_test.py` | `GridCoords` |
| `crossval_test.py` | `RandomCellSplitGridded`, `DartboardSplitGridded` |
| `datasets_test.py` | `GriddedDataset`, `Dartboard` |
| `fftshift_test.py` | conventions in `utils` |
| `fourier_test.py` | `FourierCube`, `NuFFT`, `NuFFTCached` |
| `geometry_test.py` | `flat_to_observer`, `observer_to_flat` |
| `gridder_dataset_export_test.py` | `DataAverager` → `GriddedDataset` |
| `gridder_gridding_test.py` | `DirtyImager` core gridding |
| `gridder_imager_test.py` | dirty image / beam |
| `gridder_init_test.py` | constructor validation |
| `images_test.py` | `BaseCube`, `Hann`/`Gauss` conv, `ImageCube` |
| `input_output_test.py` | `ProcessFitsImage` |
| `losses_test.py` | every loss function |
| `onedim_test.py` | radial profiles |
| `plot_test.py` | diagnostics |
| `train_test_test.py` | `train_to_dirty_image` |
| `utils_test.py` | helpers |
| `conftest.py` | session-scoped fixtures (mock cube, baselines, dataset) |
| `plot_utils.py` | shared figure helpers (`imshow_two`, …) |
| `README.md` | tester notes |

### 10.2 Tutorials

The on-tree docs (`docs/`) provide background, units, installation,
and API stubs only. Long-form tutorials (mock data, dirty-image
initialization, optimization, cross-validation, parametric inference
with Pyro) were *moved out of this repository* to
`MPoL-dev/examples` in v0.3.0 (`docs/changelog.md` lines 6, 32).

---

## 11. CLI / entry points

There are **no console scripts** declared in `pyproject.toml`. MPoL
is consumed exclusively as a Python library (no `[project.scripts]`
section). All workflows are user-written PyTorch scripts.

---

## 12. Version history (key items, `docs/changelog.md`)

### v0.3.0 highlights

- Tutorials moved to `MPoL-dev/examples`.
- New layer `images.GaussConvImage`; new helper
  `utils.convolve_packed_cube` mentioned (actually delivered via
  `images.GaussConvFourier`).
- Removed explicit `float64`/`complex128` casting — defaults to
  PyTorch's default dtype (`docs/changelog.md` line 8).
- Renamed: `SimpleNet` → `GriddedNet`,
  `cube` → `packed_cube`, `make_fake_data` → `generate_fake_data`,
  `nll` → `r_chi_squared`, `nll_gridded` → `r_chi_squared_gridded`.
- Spatial-frequency unit changed from k$\lambda$ to $\lambda$.
- Removed `from_image_properties` classmethods everywhere — pass
  `GridCoords` instead.
- Added `vis_ext_Mlam` to `GridCoords`.
- Added `DirtyImager.from_tensors`.
- Added `losses.neg_log_likelihood_avg`.
- Removed custom `spheroidal_gridding` (replaced by TorchKbNuFFT).
- Refactored `NuFFT` API: `(uu, vv)` are now passed at *forward*,
  not *init*; `NuFFTCached` preserves the old behaviour.
- Removed `setup.py`; switched to `hatch`/`hatch-vcs`.
- Added type hints across core modules.

### v0.2.0 highlights

- New programs `mpol.crossval`, `mpol.geometry`, `mpol.onedim`,
  `mpol.training`.
- Replaced `Gridder` with `GridderBase` + `DataAverager` + `DirtyImager`.
- `GriddedDataset` became an `nn.Module` with buffers.
- Added `frank` as test/analysis extra.
- Added `mpol.exceptions`, `mpol.protocols` (the latter has since been
  removed from this tree — no `protocols.py` exists in the current
  source).

### Earlier

- v0.1.4: removed `GriddedResidualConnector`.
- v0.1.3: added `make_fake_data` (now `generate_fake_data`).
- v0.1.2: added `NuFFT` layer.
- v0.1.1: added `HannConvCube`, `Dartboard`,
  `KFoldCrossValidatorGridded` (now `DartboardSplitGridded`).
- v0.1.0: introduced `Gridder` and `GridCoords`.
- v0.0.5: introduced changelog; entropy follows EHT-IV; image cube
  optimized in log-space.

---

## 13. Known limitations / TODOs

Drawn from in-source comments and the changelog (no fabrication):

1. **`mpol.training.TrainTest` and `mpol.crossval.CrossValidate`
   orchestrators are commented out**
   (`training.py` lines 59–394; `crossval.py` lines 18–331).
   Only the splitter classes and `train_to_dirty_image` are usable.
2. **`mpol.utils.convolve_packed_cube`** is mentioned in
   `docs/changelog.md` (line 9) but does not exist in the current
   source — the equivalent functionality is in
   `images.GaussConvFourier` and `images.GaussConvImage`.
3. **`mpol.protocols`** referenced in `docs/changelog.md` (line 87)
   no longer ships.
4. **TorchKbNuFFT v1.4.0 limitation**: `sparse_matrices=True` is
   incompatible with `(nchan, nvis)` batch parallelism — handled by
   warning and downgrade (`fourier.py` lines 403–415, 471–482).
5. **No native MS reader**: users must convert with
   [`visread`](https://github.com/MPoL-dev/visread) or CASA before
   feeding the gridders.
6. **`GaussConvImage` rotation-bug fix**: a recent commit
   (`15088bd refactored convolution tests and discovered rotation bug`)
   in the vendored history indicates the rotation parameter `Omega`
   has been freshly verified.
7. **CASA baseline convention**: per
   `docs/units-and-conventions.md` line 24, if your image is
   "upside down and mirrored" you must `np.conj` the visibilities.
8. **Even-`npix` requirement**: enforced by `GridCoords` (line 86) —
   odd-sided images are not supported.
9. **`scipy` is declared but is not actually imported** anywhere in
   `src/mpol/`; this is a vestigial dependency.
10. **`from frank.geometry import …`** is a *runtime* import inside
    `onedim.radialV` — the function fails at call time if `frank`
    is not installed.

---

## 14. Quick-reference: end-to-end RML imaging recipe

Synthesizing the documented APIs above, a minimal MPoL imaging script
looks like:

```python
import numpy as np, torch
from mpol import coordinates, gridding, precomposed, losses, utils

# 1. Define the image / Fourier grid
coords = coordinates.GridCoords(cell_size=0.005, npix=512)   # arcsec, even

# 2. Convert (uu, vv [lambda], weight, data_re, data_im) to a PyTorch dataset
averager = gridding.DataAverager(
    coords=coords,
    uu=uu, vv=vv, weight=weight,
    data_re=np.real(vis), data_im=np.imag(vis),
)
dset = averager.to_pytorch_dataset()        # GriddedDataset (nn.Module)

# 3. Build the model
model = precomposed.GriddedNet(coords=coords, nchan=dset.nchan).to("cuda")

# 4. Optimization loop
optim = torch.optim.Adam(model.parameters(), lr=0.3)
for epoch in range(2000):
    optim.zero_grad()
    model_vis = model()
    loss  = losses.r_chi_squared_gridded(model_vis, dset)
    loss += 1e-3 * losses.TSV(model.icube.sky_cube)
    loss += 1e-4 * losses.sparsity(model.icube.sky_cube)
    loss.backward()
    optim.step()

# 5. Export
model.icube.to_FITS("rml.fits", overwrite=True)
```

This recipe is consistent with `precomposed.GriddedNet.forward`
(`precomposed.py` lines 69–84), the gridded loss
(`losses.py` lines 97–126), and FITS export
(`images.py` lines 494–535).

---

## 15. File-citation index (for cross-checking)

- License & copyright: `simulators/MPoL/LICENSE`
- Identity & citation: `simulators/MPoL/README.md`,
  `simulators/MPoL/CONTRIBUTORS.md`, `simulators/MPoL/paper/paper.md`
- Build & deps: `simulators/MPoL/pyproject.toml`
- Versioning: `git -C simulators/MPoL log/tag`,
  `simulators/MPoL/docs/changelog.md`
- Conventions: `simulators/MPoL/docs/units-and-conventions.md`
- Modules:
  - `simulators/MPoL/src/mpol/__init__.py`
  - `simulators/MPoL/src/mpol/constants.py`
  - `simulators/MPoL/src/mpol/exceptions.py`
  - `simulators/MPoL/src/mpol/coordinates.py`
  - `simulators/MPoL/src/mpol/images.py`
  - `simulators/MPoL/src/mpol/fourier.py`
  - `simulators/MPoL/src/mpol/gridding.py`
  - `simulators/MPoL/src/mpol/datasets.py`
  - `simulators/MPoL/src/mpol/crossval.py`
  - `simulators/MPoL/src/mpol/losses.py`
  - `simulators/MPoL/src/mpol/precomposed.py`
  - `simulators/MPoL/src/mpol/training.py`
  - `simulators/MPoL/src/mpol/geometry.py`
  - `simulators/MPoL/src/mpol/onedim.py`
  - `simulators/MPoL/src/mpol/input_output.py`
  - `simulators/MPoL/src/mpol/plot.py`
  - `simulators/MPoL/src/mpol/utils.py`
- Tests: `simulators/MPoL/test/*.py`
