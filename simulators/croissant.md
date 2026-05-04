# CROISSANT — exhaustive reference

> *spheriCal haRmOnics vISibility SimulAtor iN pyThon*

This document is an exhaustive technical reference for the CROISSANT codebase as it exists in `simulators/croissant/` (vendored from `https://github.com/christianhbye/croissant`, package name `croissant-sim`, version **5.2.1**, MIT-licensed, author Christian Hellum Bye, `chbye@berkeley.edu`). It is intended for engineers integrating, comparing, or replacing CROISSANT inside RRIVis. Every public function/class is described with its signature, semantics, side effects, expected shapes, dtypes, and the physics it implements. Where helpful, the relevant test that locks the behavior is named.

---

## 1. What CROISSANT is, in one paragraph

CROISSANT is a small (≈1.5k lines of Python source, ≈2.5k lines of tests), fully-JAX, fully-differentiable single-element/per-pair *visibility* (a.k.a. *antenna temperature*) simulator for low-frequency radio astronomy. It does **not** simulate baseline-resolved interferometric visibilities of point/diffuse sources via the Measurement Equation the way DP3/RASCIL/pyuvsim/WODEN do. Instead, it computes the *autocorrelation* (or the cross-correlation of two beam patterns) of an antenna power beam against a sky brightness temperature map, by representing both as spherical harmonic expansions and turning the convolution `T_ant(t,ν) = ∫dΩ A(θ,φ,ν) T_sky(θ,φ,ν,t)` into a dot product over `(ℓ,m)` indices. Time evolution is a phase rotation `exp(-i·m·φ(t))` applied to the sky `aₗᵐ` coefficients. CROISSANT supports both Earth and Moon as observing world (the latter via the MEPA / Mean Earth / Polar Axis frame) and is built end-to-end on Equinox + JAX so the entire pipeline is `jax.jit`/`jax.grad`-friendly.

The spiritual niche is **single-element antenna-temperature forecasting** for global-21cm / total-power experiments (REACH, MIST, EDGES-style, FARSIDE/lunar-orbit concepts), where you care about how the integrated sky power varies as the Earth/Moon rotates. It is *not* a per-baseline visibility simulator and does not compute fringe-rate / `(u,v,w)` geometry.

---

## 2. Repository layout

```
simulators/croissant/
├── pyproject.toml            # build (setuptools), pinned deps, ruff config
├── uv.lock                   # uv lockfile (~400 KB)
├── requirements-dev.txt      # dev pins
├── README.md                 # quickstart
├── CONTRIBUTING.md           # 6 lines
├── CLAUDE.md                 # agent-facing project doc
├── CHANGELOG.md              # release-please managed
├── IMPROVEMENTS.md           # known limitations + future work
├── LICENSE                   # MIT (Christian Hellum Bye, 2022-)
├── .pre-commit-config.yaml   # ruff hook
├── .release-please-manifest.json
├── release-please-config.json
├── docs/
│   └── math.md               # 30-line derivation of the SHT convolution
├── notebooks/
│   ├── example_sim.ipynb     # main demo: full sidereal day Moon + Earth
│   ├── croissant_jax.ipynb   # legacy JAX entry-point demo
│   ├── multipair_example.ipynb # two dipole pairs
│   ├── single_pixel.ipynb    # point-source sky impulse response
│   ├── mepa_precession.ipynb # MEPA reference epoch effects
│   ├── test_analytic_beams.ipynb
│   ├── beam.npy              # cached sample beam
│   └── ulsa.npy              # cached ULSA sky map
├── src/croissant/
│   ├── __init__.py           # top-level exports + __version__
│   ├── constants.py          # sidereal_day, Y00, deprecated PIX_WEIGHTS_NSIDE
│   ├── utils.py              # SHT indexing, healpix helpers, time array
│   ├── sphere.py             # SphBase eqx.Module + compute_alm
│   ├── beam.py               # Beam(SphBase): horizon, beam_rot, beam_tilt
│   ├── sky.py                # Sky(SphBase): coord systems
│   ├── rotations.py          # frame transforms, Wigner-D Euler angles
│   ├── simulator.py          # Simulator class, rot_alm_z, convolve, ground loss
│   ├── multipair.py          # vmap'd multi-pair convolve
│   ├── alm.py                # 18-line deprecation shim → utils
│   └── jax/__init__.py       # 9-line deprecation shim → top-level
└── tests/                    # 9 test modules, ~2.5k lines, pytest-cov 70% target
```

The `src/croissant/jax/` submodule and `src/croissant/alm.py` exist solely to keep pre-v5 imports (`from croissant.jax import multipair`, `from croissant import alm`) working with a `DeprecationWarning`. New code should use `croissant.multipair`, `croissant.utils` etc.

---

## 3. Package metadata (pyproject.toml)

* `name = "croissant-sim"`, `version = "5.2.1"` (release-please manages bumps).
* Python `>=3.10, <3.13`. 3.13 is documented as experimental; `<3.13` is hard-pinned.
* Runtime deps (unpinned, installed transitively by uv lock): `astropy`, `equinox`, `jax`, `lunarsky`, `numpy`, `s2fft`. NB `spiceypy` is required at runtime but is pulled in via `lunarsky` rather than declared explicitly — it is imported directly in `rotations.py`.
* Dev deps (`[dependency-groups].dev`): `healpy>=1.19.0`, `ipympl`, `jupyterlab`, `jupyterlab-vim`, `matplotlib`, `pre-commit`, `pygdsm>=1.6.4`, `pytest`, `pytest-cov`, `pytest-timeout`, `ruff`. Healpy and pygdsm are dev-only because production code uses `s2fft` for SHTs, but tests cross-check against healpy and use pygdsm’s `GlobalSkyModel16` as a realistic sky.
* `pytest` config: `--cov=src --cov-report=term-missing --junitxml=junit.xml -o junit_family=legacy --timeout=120`. So every test has a 120 s timeout.
* Ruff: line length 79, rules `E, F, W, I` (pycodestyle, pyflakes, isort).
* Per-file ignores: `__init__.py: E402, F401`; `notebooks/*: E402, E501`; `src/croissant/jax/__init__.py: F403`; `src/croissant/alm.py: F401`.
* Test bootstrap: `tests/conftest.py` calls `jax.config.update("jax_enable_x64", True)` globally, so the entire test suite runs in float64. Production code does not enable x64 — callers must do so themselves if they want it.

---

## 4. Mathematical foundation

Two pieces of math drive everything:

### 4.1 Convolution as a dot product in `(ℓ,m)` space

Antenna temperature is

`V(ν, t) = ∫ dΩ  A(θ, φ, ν) · T_sky(θ, φ, ν, t)`

Expand both in spherical harmonics:
`A = Σ aₗᵐ Yₗᵐ`, `T_sky = Σ bₗᵐ Yₗᵐ`. Orthonormality of `Yₗᵐ` gives

`V = Σ_ℓ aₗ⁰ bₗ⁰ + 2 · Σ_{ℓ, m>0} Re(aₗᵐ · (bₗᵐ)*)`,

which is the relation derived in `docs/math.md`. Because the `s2fft` backend stores both positive and negative `m`, the implementation uses the symmetric form `V = Σₗ Σₘ aₗᵐ · (bₗᵐ)*` — i.e. an einsum `"flm,tm,flm->tf"` over (freq, ℓ, m). Negative-`m` modes are reconstructed via the reality condition `aₗ⁻ᵐ = (-1)ᵐ (aₗᵐ)*` (verified in `utils.is_real`). `s2fft.forward(reality=True)` exploits this so only positive-m modes are *transformed*, but the stored alm array is full-width `2·lmax+1`.

The DPSS section of `docs/math.md` describes a frequency-axis decomposition using Discrete Prolate Spheroidal Sequences via `hera_filters` — but **DPSS is not implemented in v5.x**. It is purely speculative future work in `docs/math.md`. There is no `dpss` module, no `hera_filters` import anywhere in the repo, and no test exercises that path.

### 4.2 Time evolution as a Z-axis phase rotation

In a frame whose Z-axis is aligned with the rotation axis of the host body (FK5 for Earth, MEPA for the Moon), the rotation of the sky over time is a pure Z rotation, which acts on `aₗᵐ` as `aₗᵐ(t) = aₗᵐ(0) · exp(-i·m·φ(t))` where `φ(t) = 2π·(t-t₀)/T_sid`. `T_sid` is `23.9345 h` for Earth (sidereal_day_earth) and `655.720 h ≈ 27.32 d` for the Moon (sidereal_day_moon), constants taken straight from NSSDC fact sheets. This is implemented in `simulator.rot_alm_z` and is exact only as long as the simulation Z-axis is the spin axis — see §13 limitations.

The `(t-t₀)` is converted to seconds inside `Simulator.__init__` from `times_jd` via `dt_sec = (times_jd - times_jd[0]) * 86400`, so phases are computed relative to the *first* time sample. The user may pass any monotonic JD array; non-uniform spacing works (verified by `test_rot_alm_z_with_times_parameter`).

---

## 5. Architecture: Equinox modules + JAX

Three classes — `SphBase`, `Beam`, `Sky`, `Simulator` — all subclass `eqx.Module`. This gives them automatic pytree registration, so you can pass them directly to `jax.jit`, `jax.grad`, `jax.vmap` without writing custom flatten/unflatten. Static fields (anything not differentiated) are marked with `eqx.field(static=True)`; dynamic fields are JAX arrays that flow as pytree leaves. Concretely:

* In `SphBase`: dynamic `data, freqs, theta, phi`; static `sampling, lmax, _L, _niter, nside`.
* In `Beam`: adds dynamic `horizon, beam_rot, beam_tilt`.
* In `Sky`: adds static `coord`.
* In `Simulator`: dynamic `beam, sky, times_jd, freqs, lon, lat, alt, dl_topo, Tgnd, phases`; static `lmax, _L, eul_topo (tuple of floats), world, _et_ref`.

Because `lmax` is static, the JAX trace specializes per `lmax` — changing `lmax` triggers re-JIT but no per-call recompilation when only `data`/`freqs` change.

Functions wrapped with `@jax.jit` (or `@eqx.filter_jit`):
* Module-level: `sphere.compute_alm` (`@eqx.filter_jit`), `simulator.convolve`, `simulator.correct_ground_loss`, `multipair.multi_convolve`, `multipair.compute_visibilities`.
* Methods: `Beam.compute_norm`, `Beam.compute_fgnd`, `Beam.compute_alm`, `Sky.compute_alm`, `Simulator.compute_beam_eq`, `Simulator.compute_ground_contribution`, `Simulator.sim`.

`Simulator.sim` therefore *recomputes* the forward SHT every call — this is intentional so that `jax.grad(sim, sky.data)` and `jax.grad(sim, beam.data)` flow gradients through the SHT (see IMPROVEMENTS.md §4). For users who don't need gradients, `Simulator.precompute_sky_alm()` returns the simulator-frame sky `aₗᵐ` once, and `sim(sky_alm=...)` skips the forward SHT and rotation.

---

## 6. Module-by-module deep dive

### 6.1 `croissant/__init__.py` (15 lines)

Top-level surface:

```python
from . import constants, multipair, rotations, simulator, utils
from .beam import Beam
from .simulator import Simulator
from .sky import Sky
from . import alm   # deprecated, but importable
__author__ = "Christian Hellum Bye"
__version__ = importlib.metadata.version("croissant-sim")  # falls back to "unknown"
```

So `cro.Beam`, `cro.Sky`, `cro.Simulator`, `cro.utils.*`, `cro.constants.*`, `cro.simulator.*`, `cro.multipair.*`, `cro.rotations.*` are all the public surface. `croissant.sphere`, `croissant.beam`, `croissant.sky` are *not* re-exported as submodules — they are accessible as `cro.beam`, `cro.sky`, `cro.simulator` because Python imports them implicitly via `from .beam import Beam` etc. Notebook code regularly uses `cro.sphere.compute_alm`, but `sphere` is only available because tests/notebooks import it directly.

### 6.2 `constants.py` (29 lines)

```python
sidereal_day_earth = 23.9345 * 3600              # 86 164.2 s
sidereal_day_moon  = 655.720 * 3600              # 2.36e6 s ≈ 27.32 days
sidereal_day       = {"earth": ..., "moon": ...}
Y00                = 1 / sqrt(4π)                # the (0,0) spherical harmonic value
_PIX_WEIGHTS_NSIDE = (32, 64, 128, 512, 1024, 2048, 4096)  # legacy healpy weights
```

`PIX_WEIGHTS_NSIDE` is exposed via `__getattr__` and emits a `FutureWarning` when accessed. It is unused in production code; it exists only because old healpy-based code referenced it (`tests/test_constants.py` locks the warning).

### 6.3 `utils.py` (507 lines)

A grab-bag of helpers. Functions:

| Function | Purpose |
|---|---|
| `valid_nside(nside) -> bool` | True iff `nside > 0` and is a power of two (`nside & (nside-1) == 0`). |
| `hp_npix2nside(npix) -> int` | `int(sqrt(npix/12))`; raises `ValueError` if `npix <= 0` or `npix % 12 != 0`. |
| `hp_valid_npix(npix) -> bool` | `valid_nside(hp_npix2nside(npix))` with the `ValueError` swallowed. |
| `time_array(t_start, t_end, N_times, delta_t)` | Uniformly-spaced `astropy.time.Time` (or `lunarsky.Time` if `t_start` is one) array. Accepts any 3 of the 4 arguments. Returns a `Time` if `t_start` is given, else an `astropy.units.Quantity` of seconds. Logs a `UserWarning` if `delta_t` has no units. |
| `generate_phi(lmax, sampling, nside)` | Wraps `s2fft.sampling.s2_samples.phis_equiang` for non-healpix samplings. For healpix it concatenates `phis_ring(t, nside)` for every ring. Validates that the right pair (lmax for non-healpix, nside for healpix) is provided. |
| `generate_theta(lmax, sampling, nside)` | Same wrapping for `s2fft.sampling.s2_samples.thetas`. For healpix, since s2fft returns one θ per ring, it `np.repeat`s by `nphi_ring(t, nside)` so the output is one θ per pixel. |
| `getidx(lmax, ell, emm) -> (l_ix, m_ix)` | `(ell, emm + lmax)`. The alm array column index for a given m is `m + lmax`, so column `lmax` corresponds to `m=0`. |
| `getlm(lmax, ix) -> (ell, emm)` | Inverse of `getidx`; subtracts `lmax` from the m column. Accepts vectorized 2-row index arrays. |
| `total_power(alm, lmax) -> float` | `Re(alm[..., 0, lmax]) / Y00`. Only the monopole `(0,0)` integrates to non-zero on the sphere, and the integral equals `a₀⁰ · √(4π)`, which is `a₀⁰ / Y00`. Used to normalize beams. Vectorizes over leading axes. |
| `lmax_from_shape(shape) -> int` | `shape[-2] - 1`; alm shape is `(..., lmax+1, 2lmax+1)`. |
| `is_real(alm) -> bool` | Returns `jnp.allclose(alm[..., :lmax][..., ::-1], (-1)**m · conj(alm[..., lmax+1:])).item()`. The Hermitian-symmetry check for real signals. Works on any leading-batch shape. |
| `reduce_lmax(alm, new_lmax)` | Truncates an alm array to `new_lmax`. Returns `alm[..., :-d, d:-d]` where `d = lmax - new_lmax`. Raises `ValueError` if `new_lmax > lmax`. Returns the original array if `d == 0`. |
| `shape_from_lmax(lmax) -> (lmax+1, 2*lmax+1)` | Two-element tuple. |
| `lmax_from_ntheta(ntheta, sampling) -> int` | Inverse of the SHT bandlimit relation per sampling: `mw/gl: ntheta-1`, `mwss: ntheta-2`, `dh: ntheta//2 - 1`, `healpix: 2·hp_npix2nside(ntheta)`. Raises `ValueError` for unknown sampling. |
| `__getattr__(name)` | Backwards compatibility shim. Accessing `utils.get_rot_mat`, `utils.rotmat_to_euler`, `utils.rotmat_to_eulerZYX`, `utils.rotmat_to_eulerZYZ` returns the function from `rotations` wrapped in a `FutureWarning`. |

Implementation notes:
* `total_power`: index for monopole is `(0, lmax)`. `Re` returns Python float; arrays remain JAX. Used by `Simulator.sim` for beam normalization and by `multipair.compute_normalization`.
* `is_real`: ends with `.item()`, so it is a Python `bool` and breaks JIT tracing — never call inside a JIT'd function.
* `reduce_lmax`: pure index-slice, JIT-safe, used heavily in `Simulator.sim` to trim mismatched lmax between sky/beam.

### 6.4 `sphere.py` (137 lines)

```python
@eqx.filter_jit
def compute_alm(data, lmax, sampling, nside=None, niter=0):
    m2alm = partial(s2fft.forward, L=lmax+1, spin=0, nside=nside,
                    sampling=sampling, method="jax", reality=True,
                    precomps=None, spmd=False, L_lower=0, iter=niter)
    return jax.vmap(m2alm)(data)
```

Always uses `method="jax"`, `reality=True`, `spin=0`, vmaps over the leading frequency axis.

`SphBase(eqx.Module)` is the parent of `Beam` and `Sky`. Its `__init__`:
* Coerces `data` to `jnp.array`, `freqs` to at least 1-D.
* If sampling is `"healpix"`, validates `npix` (`hp_valid_npix(data.shape[1])` else `ValueError`).
* Stores `_niter`, `sampling`.
* Computes `lmax = utils.lmax_from_ntheta(data.shape[1], sampling)`. So lmax is *derived from the data shape and sampling*, not an independent parameter — there is no way to bandlimit an MWSS map below its natural lmax via the constructor.
* `_L = lmax + 1` (for `s2fft`, which uses `L = lmax + 1` everywhere).
* `nside = hp_npix2nside(data.shape[1])` for healpix, else `None`.
* Pre-computes `phi` and `theta` arrays from `utils.generate_phi`/`generate_theta`.

Default `niter` is 0 across the codebase since v5.1.3 (per CHANGELOG and IMPROVEMENTS.md §5). The previous default for healpix was `niter=3`; setting it to 0 cut JIT compile time dramatically at the cost of approximate forward SHT for healpix.

### 6.5 `beam.py` (179 lines) — `Beam(SphBase)`

Power beam pattern in local antenna ENU frame.

Constructor: `Beam(data, freqs, sampling="mwss", horizon=None, beam_rot=0.0, beam_tilt=0.0, niter=0)`. `data` is a power beam (not voltage), so it is intrinsically real and unpolarized.

Extra fields:
* `horizon`: boolean array. Default is `theta <= π/2` (i.e. the upper hemisphere is "above horizon"). For non-healpix samplings the default is expanded to shape `(ntheta, 1)` so it broadcasts over `phi`.
* `beam_rot`: float, degrees. **Astronomical convention**: measured from local North towards East. `beam_rot=0` leaves the beam unrotated (φ=0 axis aligned with local East in ENU). `beam_rot=90` rotates the beam so its φ=0 axis points South. Implemented as a phase factor `exp(+i·m·beam_rot)` applied to the alm — *not* a Wigner-D rotation, since a pure azimuthal rotation is a Z rotation. Note the `+` sign — this was changed in v5.1.4 to follow N→E convention; the test `test_beam_rot_direction` locks the sign.
* `beam_tilt`: float, degrees. **NotImplementedError** if non-zero. The intent is a tilt of the boresight from local zenith; IMPROVEMENTS.md §3 spells out the design (Wigner-D rotation about the local Y / East–West axis). Currently raises in `__init__`.

Methods:

* `_compute_norm(use_horizon=True)` — Numerical sphere integral of the beam pattern (per frequency). Uses `s2fft.utils.quadrature_jax.quad_weights(L, sampling, nside)` for non-healpix, else uniform weights `4π/npix` (HEALPix is equal-area). With `use_horizon=True`, the data is multiplied by `self.horizon[None]` first. Returns shape `(N_freqs,)`.
* `compute_norm()` — `_compute_norm(use_horizon=False)`. The full-sphere integral, used as the denominator in `Simulator.sim`.
* `compute_fgnd()` — `1 - norm_above_horizon / norm_total`. Ground fraction of the beam pattern (one number per freq). Used to compute the ground contribution `T_gnd · f_gnd`.
* `compute_alm()` — (1) zero out below-horizon pixels via `data * horizon[None]`, (2) `sphere.compute_alm` over freq axis, (3) apply `exp(+i·m·beam_rot)` phase factor to all m (uses `jnp.arange(-lmax, lmax+1)` so emm covers the full alm column index). Returns `(N_freqs, lmax+1, 2*lmax+1)`.

Tests in `tests/test_beam.py` lock:
* Norm of an all-ones beam over the full sphere is `4π` to `1e-3` for every sampling.
* Norm scales linearly with the constant.
* `fgnd ≈ 0.5` for default horizon on isotropic beam; `fgnd = 0` if `horizon=ones`; `fgnd = 1` if `horizon=zeros`.
* `fgnd + fsky = 1` exactly.
* Reality condition holds for the alm of a constant beam.
* `beam_rot=0` and `beam_rot=360` give equal alm.
* `beam_rot=90` permutes a `cos(φ)` beam (peak at East) so the peak is at South (`-sin(φ)`), not at North — confirming the N→E convention.

### 6.6 `sky.py` (115 lines) — `Sky(SphBase)`

Sky brightness temperature map in one of three frames.

Constructor: `Sky(data, freqs, sampling="healpix", coord="galactic", niter=0)`.
* `coord ∈ {"galactic", "equatorial", "mepa"}`. Else raises `ValueError`.
* `equatorial` means **FK5**, mean-equatorial-of-J2000. The astropy frame name is `fk5` and that's what `rotations.gal2eq` rotates to.
* `mepa` (Mean Earth / Polar Axis) is a Moon-fixed inertial frame whose Z-axis is aligned with the lunar spin axis at a reference epoch. Equivalent to `MOON_ME` in SPICE evaluated at that epoch. Replaced the previous MCMF frame in v5.1.0 — the *physical* difference is that MEPA is inertial (frozen at a chosen epoch) so the sky-rotation phase model is exact, while MCMF is body-fixed and drifts.

Methods:
* `compute_alm()` — direct SHT in the input coordinate system. Just wraps `sphere.compute_alm`.
* `compute_alm_eq(world="moon", et=None)` — alm in the *simulation* frame (FK5 if `world="earth"`, MEPA if `world="moon"`). If the sky is already native (`coord=="equatorial"` and `world=="earth"`, or `coord=="mepa"` and `world=="moon"`), just returns `compute_alm()`. If `coord=="galactic"`, calls `rotations.gal2eq` or `rotations.gal2mepa(et=et)`. Mismatched combos (`coord="mepa"` + `world="earth"`, `coord="equatorial"` + `world="moon"`) raise `ValueError`. There is *no* equatorial↔mepa transformation: those are physically different frames anchored to different bodies, so calling them out is correct.

`et` is the SPICE ephemeris time (seconds past J2000 TDB) used as the MEPA reference epoch. Defaults to `0.0` (J2000) inside `gal2mepa`. Per `Simulator.__init__`, when `world="moon"` the simulator passes `_et_ref = jd_to_et(times_jd[0])`, so the MEPA frame is anchored to the *first* observation epoch — this makes the sky-rotation phase exact for that observation (because Moon's spin axis is by definition aligned with the MEPA Z-axis at the reference epoch).

### 6.7 `rotations.py` (465 lines)

Coordinate transforms, expressed as 3×3 rotation matrices, ZYZ Euler angles, and Wigner-D arrays for s2fft. No SHT happens here; this module produces inputs for `s2fft.utils.rotation.rotate_flms`.

Public functions:

* `jd_to_et(jd) -> float` — `(jd - 2451545.0) * 86400.0`. SPICE ET seconds past J2000.
* `get_rot_mat(from_frame, to_frame, et=None)` — 3×3 rotation matrix between any two of `{galactic, fk5, AltAz, LunarTopo, mcmf, mepa}`. Implementation:
  * If `to_frame == "mepa"` or `from_frame == "mepa"`, dispatches to `_rot_mat_to_mepa(other, et)` (and transposes if from-MEPA).
  * Otherwise, since `astropy` doesn't transform from galactic via the cartesian representation, it builds three unit vectors with `frame=from_frame` and transforms them. If `from_frame` is galactic, it inverts the direction and transposes at the end.
  * **ENU vs. NEU**: AltAz and LunarTopo natively use NEU (X=North, Y=East, Z=Up) cartesian. The function swaps X↔Y for those frames so the matrix is in right-handed ENU (X=East, Y=North, Z=Up), giving `det = +1`. This was a v5.1.2 fix; `tests/test_rotations.py::test_get_rot_mat` locks both the determinant and the relationship to the raw NEU SkyCoord output.
* `rotmat_to_euler(mat, eulertype)` — dispatches to `rotmat_to_eulerZYX` (healpy convention, Tait-Bryan) or `rotmat_to_eulerZYZ` (Wigner / s2fft convention). The codebase prefers ZYZ.
* `rotmat_to_eulerZYZ(mat)` — extracts ZYZ Euler angles `(α, β, γ)`. Handles gimbal lock at β=0 and β=π by setting γ=0 and absorbing into α. The v5.2.1 fix in `15f6d21` corrected `np.arctan2` argument order at β=π. Tested by `test_rotmat_to_eulerZYZ_gimbal_lock`.
* `rotmat_to_eulerZYX(mat)` — Healpy/Tait-Bryan convention `(α, -β, γ) = (yaw, -pitch, roll)`. Returns the input that `healpy.rotator.Rotator(eul, eulertype="ZYX")` expects. Mostly retained for backwards compatibility.
* `generate_euler_dl(lmax, from_frame, to_frame, et=None)` — Returns `(euler, dl_array)` where `euler = rotmat_to_eulerZYZ(get_rot_mat(...))` and `dl_array = s2fft.generate_rotate_dls(lmax+1, euler[1])`. The `dl_array` is the reduced Wigner d-matrix at fixed β; `s2fft.utils.rotation.rotate_flms` consumes both.
* `generate_euler_dl_from_rotmat(lmax, rotmat)` — Same but caller already has the matrix.
* `topo_to_mepa_euler_dl(lmax, topo_frame)` — convenience around `generate_euler_dl(lmax, topo_frame, "mepa")`.
* `get_mepa_rotation_matrix(et=0.0)` — `spice.pxform("J2000", "MOON_ME", et)`. **`@functools.lru_cache(maxsize=1024)`** — capped to avoid unbounded memory in long Bayesian sweeps. The 1024 cap was added in `0661eb2`.
* `_rot_mat_to_mepa(from_frame, et)` — internal. For `LunarTopo`, composes `topo → MCMF → J2000 → MEPA` using the *frame's* obstime for the topo→MCMF and MCMF→J2000 steps and `et` (defaulting to obstime) for the J2000→MEPA step. For other frames, just `frame → FK5/J2000 → MEPA`. The decoupling of "obstime" (used for the time-varying parts of the chain) vs. "MEPA reference epoch" (used to anchor the MEPA Z-axis) is what enables `test_topo_to_mepa_beta_constant` (β unchanged across observation times when `et` defaults to obstime) and `test_topo_to_mepa_time_dependent` (β does change when `et` is fixed at J2000).
* `_gal_to_sim_frame(alm, eul=None, dl_array=None, world="moon", et=None)` — generic galactic→sim-frame rotation. Uses `s2fft.utils.rotation.rotate_flms` vmapped over the leading frequency axis.
* `gal2eq(alm, eul=None, dl_array=None)` — `_gal_to_sim_frame(world="earth")`.
* `gal2mepa(alm, eul=None, dl_array=None, et=None)` — `_gal_to_sim_frame(world="moon", et=et)`.

Caching strategy: `get_rot_mat` is **not** cached — every call re-runs astropy frame transforms. `get_mepa_rotation_matrix` is cached on `et`. The Euler/dl computation in `generate_euler_dl` is not cached. `Simulator.__init__` calls it once and stores `eul_topo, dl_topo` on the module, so it pays the cost once per simulator construction.

### 6.8 `simulator.py` (406 lines)

The orchestration layer.

#### 6.8.1 Module-level functions

* **`rot_alm_z(lmax, N_times=None, delta_t=None, times=None, world="moon")`** — Returns `phases` of shape `(N_times, 2*lmax+1)`, where `phases[t, m] = exp(-i·m·φ(t))`, `φ(t) = 2π·(t-t₀)/T_sid`. Two calling modes: pass `N_times + delta_t` (uniform sampling, `dt = arange(N_times)*delta_t`), or pass `times` (any 1-D array of seconds, treated as absolute, then internally subtracted from `times[0]`). If both `N_times` and `delta_t` are missing and `times` is None, raises `ValueError`. Empty `times` array also raises. Note the `m` array spans `[-lmax, +lmax]` — same range as the alm column index minus `lmax`. Tested by `test_rot_alm_z` (against a direct s2fft Wigner rotation about Z) and `test_rot_alm_z_with_times_parameter` (offset-invariance, single-time edge case, non-uniform spacing).

* **`convolve(beam_alm, sky_alm, phases)`** — The core engine. Returns `jnp.einsum("flm,tm,flm->tf", sky_alm.conjugate(), phases, beam_alm)`. Output shape `(N_times, N_freqs)`, dtype complex128 (in tests with `x64=True`). Note that this `einsum` correctly executes `Σ_{ℓ,m} sky*ₗᵐ · phase_m · beam_ₗᵐ`, which combines both `±m` and falls back to the math.md formula. **No normalization is applied** — that is the caller's job (`Simulator.sim` divides by `beam.compute_norm()`).

* **`correct_ground_loss(vis, fgnd, Tgnd)`** — Inverts the simple ground-loss model: `T_sky = (vis - f_gnd · T_gnd) / (1 - f_gnd)`. Three `jax.jit`'d arithmetic ops. Used to recover sky temperature from a measured antenna temperature when `f_gnd` and `T_gnd` are estimated.

#### 6.8.2 `class Simulator(eqx.Module)`

Constructor `Simulator(beam, sky, times_jd, freqs, lon, lat, alt=0, lmax=None, world="moon", Tgnd=300.0)`:

1. Verifies `beam.freqs ≈ freqs ≈ sky.freqs` via `jnp.allclose`, else `ValueError`.
2. `lmax = min(beam.lmax, sky.lmax)` if not provided. If provided and exceeds either, `ValueError`.
3. Builds the topocentric → simulation-frame Euler angles & Wigner-d array based on the *first* observation time:
   * **Earth**: `EarthLocation(lon, lat, height=alt)` + `astropy.time.Time(times_jd[0], format='jd')` → `AltAz(location, obstime)`. `eul_topo, dl_topo = generate_euler_dl(beam.lmax, AltAz, "fk5")`. `_et_ref = 0.0`.
   * **Moon**: `MoonLocation(lon, lat, height=alt)` + `lunarsky.Time(times_jd[0], format='jd')` → `LunarTopo(location, obstime)`. `_et_ref = jd_to_et(t0.tdb.jd)`. `eul_topo, dl_topo = generate_euler_dl(beam.lmax, LunarTopo, "mepa")`.
   * `world="saturn"` etc. raises `ValueError`.
4. Note: `eul_topo` is computed at `beam.lmax`, not `self.lmax`. Truncation via `reduce_lmax` happens at convolve time in `sim()`. The `dl_array` shape only depends on `lmax+1`, so this is fine.
5. Pre-computes `phases = rot_alm_z(self.lmax, times=dt_sec, world=world)` where `dt_sec = (times_jd - times_jd[0]) * 86400`. So phases are computed at the *simulation* lmax, not `beam.lmax`.
6. `Tgnd` stored as a scalar `jnp.array`. Only constant-temperature ground supported (IMPROVEMENTS.md §2).
7. `alt` is stored but never used downstream (rotation matrices are translation-free; altitude only matters for parallax/aberration which are negligible at radio frequencies).

Methods:

* `compute_beam_eq()` (`@jax.jit`) — `compute_alm()` on the beam, then `s2fft.utils.rotation.rotate_flms` with the cached `eul_topo, dl_topo`, vmapped over freq. Output `(N_freqs, beam.lmax+1, 2*beam.lmax+1)` — *not yet* truncated to `self.lmax`. The "_eq" name is historical; on the Moon this is the MEPA frame, not equatorial.
* `compute_ground_contribution()` (`@jax.jit`) — `beam.compute_fgnd() * Tgnd`. Returns `(N_freqs,)`.
* `precompute_sky_alm()` — `sky.compute_alm_eq(world=self.world, et=self._et_ref)`. **Not** JIT-jit'd (callable from outside JAX). Returns the sky alm in the simulation frame at full sky lmax. Caveats are documented at length in the docstring: (1) tied to this simulator's `world` and start time, so don't reuse across simulators with different start times; (2) computed *outside* `jax.grad`, so it severs the gradient chain through `sky.data` — for differentiable workflows, omit `sky_alm` and let `sim()` recompute.
* `sim(sky_alm=None)` (`@jax.jit`) — The main entry point.
  1. `beam_eq_alm = compute_beam_eq()`.
  2. If `sky_alm` is None, `sky_eq_alm = sky.compute_alm_eq(world, et=_et_ref)`. Else validates ndim==3, `shape[0] == len(freqs)`, `lmax_from_shape(sky_alm.shape) >= self.lmax`, then uses it.
  3. `beam_eq_alm = reduce_lmax(beam_eq_alm, self.lmax)`, same for `sky_eq_alm`.
  4. `vis_sky = convolve(beam_eq_alm, sky_eq_alm, phases) / beam.compute_norm()[None, :]`.
  5. `vis_gnd = compute_ground_contribution()`.
  6. `vis = (vis_sky + vis_gnd).real`. The imaginary part is mathematically zero for real beam + real sky (locked at numerical noise level by tests in `test_physics.py::TestTimeDomain::test_axial_mode_constant_in_time`, `TestMultipair::test_*`).

Returns shape `(N_times, N_freqs)`, real dtype (float64 if x64 enabled).

The `sim()` method does *not* take time/frequency arguments — those are baked in at simulator construction. To re-run with new times/freqs, build a new `Simulator`. Most of the cost is in `compute_alm_eq` (the SHT), which is recompiled per `(lmax, sampling, niter)` combination but cached otherwise.

### 6.9 `multipair.py` (135 lines)

Multi-antenna-pair extension. The `convolve` function is `vmap`'d over a leading "pair" axis on the beam tensor:

```python
_multi_convolve = jax.vmap(convolve, in_axes=(0, None, None))

@jax.jit
def multi_convolve(beam_alm, sky_alm, phases):
    """beam_alm: (N_pairs, N_freqs, lmax+1, 2*lmax+1)
       returns:  (N_pairs, N_times, N_freqs)"""
    return _multi_convolve(beam_alm, sky_alm, phases)
```

`compute_visibilities(beam_alm, sky_alm, phases, norm)` builds on top:
* `vis_raw = multi_convolve(...)`
* Broadcasts `norm` (shape `(N_pairs,)` or `(N_pairs, N_freqs)`) against the time axis.
* Returns `vis = transpose(vis_raw / norm_broadcast, (1, 0, 2))` — shape `(N_times, N_pairs, N_freqs)`. Note the transpose: this is the only shape difference from `multi_convolve`. The dtype is complex; for auto-correlations the imaginary part is at noise level (locked by `test_auto_correlation_matches_convolve`).

Normalization helpers:
* `compute_normalization(auto_beam_alm)` — vmapped `total_power` across the antenna axis. Input `(N_antennas, N_freqs, lmax+1, 2lmax+1)`, output `(N_antennas, N_freqs)`. The "auto_beam_alm" is the beam alm of each antenna's auto-correlation pattern (i.e. `|E|²`).
* `pair_normalization(antenna_powers, pairs)` — for each `(p, q) ∈ pairs`, returns `sqrt(power_p · power_q)`. Accepts `pairs` as any iterable of `(int, int)`, internally `jnp.array`'s it. Supports both shape `(N_antennas,)` and `(N_antennas, N_freqs)`. Returns matching `(N_pairs,)` or `(N_pairs, N_freqs)`.

The geometric-mean normalization comes from the Cauchy-Schwarz inequality for cross-correlations of two beam patterns: `|⟨A_p, A_q⟩| ≤ √(⟨A_p, A_p⟩ · ⟨A_q, A_q⟩)`. For identical antennas, the cross- and auto-correlation values match exactly (locked by `test_identical_antennas_cross_equals_auto`).

### 6.10 `alm.py` (18 lines)

```python
from .utils import getidx, getlm, is_real, lmax_from_shape, \
                   reduce_lmax, shape_from_lmax, total_power
warnings.warn("The alm module is deprecated and will be removed ...",
              DeprecationWarning, stacklevel=2)
```

Pure deprecation surface. New code should use `croissant.utils`.

### 6.11 `jax/__init__.py` (9 lines)

```python
logger.warning("The croissant.jax interface is deprecated...")
from .. import *
```

Old `from croissant.jax import multipair, alm, simulator` style. Logs a warning at import.

---

## 7. Coordinate-system reference

CROISSANT works with five frames:

| Frame | Astropy/SPICE name | Notes |
|---|---|---|
| Galactic | `galactic` | Sky models from pygdsm/ULSA are native here. |
| Equatorial (FK5) | `fk5` | Mean equatorial of J2000. Earth simulation frame. Z-axis = celestial pole. |
| Topocentric (Earth) | `astropy.coordinates.AltAz` | Antenna ENU at observer location. Built from `EarthLocation + obstime`. |
| Topocentric (Moon) | `lunarsky.LunarTopo` | Lunar antenna ENU. Built from `MoonLocation + obstime`. |
| MEPA | SPICE `MOON_ME` evaluated at a reference epoch | Moon-fixed inertial frame. Z-axis = lunar spin axis. Moon simulation frame. Replaced MCMF in v5.1.0. |
| MCMF | SPICE `MOON_ME` (body-fixed) | Used internally during topo→MEPA chain on the Moon. Not a simulation frame. |

ENU vs. NEU: AltAz/LunarTopo natively store cartesian as NEU. `get_rot_mat` always returns matrices in ENU (X=East, Y=North, Z=Up) so that `det = +1` and downstream code is in a single convention. The notebooks add a brief explainer that beam patterns from FEKO/HFSS are typically ENU-native, so no manual reorientation is needed before passing to `Beam`.

The simulation frame is FK5 for `world="earth"` and MEPA for `world="moon"`. In both, the Z-axis is the rotation axis, which is what makes `rot_alm_z` exact.

---

## 8. Sampling schemes

Inherited from s2fft. Five supported:

| Sampling | Description | `ntheta` | `nphi` | `lmax_from_ntheta` |
|---|---|---|---|---|
| `mw` | McEwen & Wiaux equiangular | `lmax+1` | `2lmax+1` | `ntheta-1` |
| `mwss` | MW-symmetric (poles included). 1° equiangular at lmax≈180. | `lmax+2` | `2lmax+2` | `ntheta-2` |
| `dh` | Driscoll-Healy | `2lmax+2` | `2lmax+2` | `ntheta/2 - 1` |
| `gl` | Gauss-Legendre | `lmax+1` | `2lmax+1` | `ntheta-1` |
| `healpix` | HEALPix RING. `lmax = 2·nside`. | `npix` (flat) | (per-ring) | `2·hp_npix2nside(ntheta)` |

The default for `Beam` is `"mwss"`; the default for `Sky` is `"healpix"` (because pygdsm/ULSA produce healpix maps natively). Mixed-sampling sims work — the SHT happens once per object.

The s2fft `mw`/`mwss` forward transforms call `spin.size`, which fails on Python `int`s when JIT is disabled. `tests/test_sphere.py` documents this and only runs jit-disabled tests on `dh`, `gl`, `healpix`.

For HEALPix at `niter=0` (the default), the forward transform is approximate; `niter=3` is roughly 100× slower to compile but is band-limited-exact. `test_compute_alm_healpix_niter_reduces_error` locks the monotonicity.

---

## 9. The `Simulator.sim()` data-flow, end to end

```
Beam.data (N_freq, ...)        Sky.data (N_freq, ...)
       │                              │
       │  Beam.compute_alm            │  Sky.compute_alm
       │  (mask horizon, SHT,         │  (SHT)
       │   apply beam_rot phase)      │
       ▼                              ▼
  beam_topo_alm (N_freq, L, 2L-1)   sky_native_alm
       │                              │
       │ s2fft.rotate_flms            │ rotations.gal2eq or gal2mepa
       │ (eul_topo, dl_topo)          │  (no-op if already in sim frame)
       ▼                              ▼
  beam_eq_alm                       sky_eq_alm
       │                              │
       │   utils.reduce_lmax          │   utils.reduce_lmax
       ▼                              ▼
  beam_eq_alm[:lmax]                sky_eq_alm[:lmax]
       │                              │
       └────────────┬─────────────────┘
                    ▼
        convolve(beam, sky, phases)            phases = rot_alm_z(lmax, dt_sec, world)
                    │                          (precomputed in __init__)
                    ▼
           (N_times, N_freq)
                    │
            / beam.compute_norm()
                    │
        + beam.compute_fgnd() * Tgnd
                    │
                  .real
                    ▼
              vis (N_times, N_freq) [K]
```

Every arrow represents a JAX traced operation, so the whole pipeline is differentiable end-to-end.

---

## 10. Ground-loss model

Two pieces:

* **In the forward model** (`Simulator.sim`): `vis = vis_sky + f_gnd · T_gnd`, where `f_gnd = 1 - ∫_{above} A dΩ / ∫_{full} A dΩ` and `T_gnd` is a constant scalar.
* **For inversion** (`correct_ground_loss(vis, fgnd, Tgnd)`): `T_sky_recovered = (vis - f_gnd · T_gnd) / (1 - f_gnd)`.

Limitations (IMPROVEMENTS.md §2):
* `T_gnd` is constant in space and frequency.
* No reflection / scattering of sky onto ground.
* No frequency dependence in `T_gnd`.

A spatially-varying or scattering ground model would require its own SHT pipeline and is explicitly out of scope.

---

## 11. Multipair / cross-correlation behavior

Although CROISSANT is single-element at heart, `multipair` lets you simulate multiple beam patterns (or multiple antenna pairs in cross-correlation) sharing a single sky. The pair convention:

* For an *autocorrelation* of antenna `p`, supply the power beam `|E_p|²` as a "pair beam" entry.
* For a *cross-correlation* of antennas `p×q`, supply the *cross-power beam* `E_p · E_q*` — CROISSANT does not compute it for you. The user must construct this.
* Normalization is `sqrt(⟨A_p A_p⟩ · ⟨A_q A_q⟩)`, computed via `compute_normalization(auto_beam_alm)` + `pair_normalization(powers, pairs)`.
* For real (auto) pair beams, `convolve` returns a complex value with imaginary part at numerical noise level (`< 1e-12`).
* For identical antennas, cross-pair (0,1) ≡ auto-pair (0,0). This is what `test_identical_antennas_cross_equals_auto` locks.
* The 90° beam rotation = ¼-sidereal-day time shift relation is locked by `test_dipole_azimuth_rotation_time_shift`, illustrating that azimuth and time are dual variables in this representation.

There is **no fringe / `(u,v,w)` term** anywhere in the multipair code: `multipair.compute_visibilities` produces `T_pq(t,ν)`, not `V_pq` in the conventional interferometric sense. To use this for a real interferometer, you would need to bake the geometric phase into the cross-power beam yourself.

---

## 12. Test suite (≈2.5k lines)

| Test file | LOC | What it covers |
|---|---|---|
| `test_constants.py` | 14 | `PIX_WEIGHTS_NSIDE` deprecation warning, missing-attribute behavior. |
| `test_utils.py` | 324 | All `utils.*` helpers; cross-checks `generate_phi`/`generate_theta` against healpy and s2fft; verifies `is_real` against `s2fft.signal_generator.generate_flm(reality=...)`; `time_array`; `lmax_from_ntheta`. |
| `test_sphere.py` | 202 | `SphBase` init shape/range; `compute_alm` shape and monopole; HEALPix `niter` reduces reconstruction error. |
| `test_beam.py` | 317 | Init across all samplings/lmaxes; `beam_tilt!=0` raises; default horizon shape; isotropic norm = 4π; norm scales with amplitude; `fgnd` extremes; alm reality; `beam_rot` phase formula; `beam_rot=90` permutes a `cos(φ)` beam to `-sin(φ)` (N→E direction lock). |
| `test_sky.py` | 134 | Defaults; invalid coord raises; alm shape; uniform sky → monopole `T/Y00`; `compute_alm_eq` is a no-op for native frames, preserves monopole when galactic→sim, raises for unsupported (mepa,earth) and (equatorial,moon). Multifrequency monopole scales with `T(ν)`. |
| `test_rotations.py` | 197 | `get_rot_mat` matches healpy for galactic↔FK5; ENU swap on AltAz; det=+1 for LunarTopo→MEPA; round-trip; ZYZ Euler round-trips for permutation matrix and gimbal-lock cases (β=0 and β=π); MEPA matrix is a proper rotation; topo→MEPA β depends on time when `et` fixed but is constant when `et` defaults to obstime (the v5.1.0 fix). |
| `test_simulator.py` | 172 | `rot_alm_z` matches a direct s2fft Wigner Z rotation across `(lmax, world, N_times)`; `convolve` recovers monopole sky T(ν) when normalized by `total_power`; `convolve` over a 5-mode beam matches manual sum; `times` parameter handles non-uniform spacing, single time, offset invariance. |
| `test_sim_class.py` | 426 | Simulator init shape/lmax/world validation; `sim()` shape/dtype; `sim(sky_alm=...)` matches `sim()`; `precompute_sky_alm` shape; raises for sky_alm with wrong ndim/freq/lmax; **monopole sky → constant visibility over time**; ground contribution increases vis by `f_gnd · T_gnd`; ground-loss round-trip and the bias when `f_gnd` or `T_gnd` are misspecified; **Moon sim depends on start time** (regression test for the v5.1.0 MEPA fix); ENU east-direction beam test (cos(φ) beam sees source at East stronger than at North by cos(alt) ratio). |
| `test_multipair.py` | 208 | Single auto-correlation packed into multipair matches `convolve` to 1e-12; `jax.grad` through vmap is finite/non-zero; normalization helpers shape & values for both 1-D and 2-D antenna_powers. |
| `test_physics.py` | 1254 | The "physical invariants" suite. Sub-classes: `TestLinearitySuperposition` (sky scaling, superposition, T_gnd linearity); `TestTimeDomain` (sidereal periodicity, sidereal-time offset = roll, sky dipole gives 1-cycle/day FFT peak, m=0 modes time-independent, m=2 modes 2-cycles/day); `TestSpectralBehavior` (achromatic beam preserves power-law); `TestBeamProperties` (isotropic beam recovers monopole, 360° rotation = identity, 180° symmetric beam invariant under 180° rotation, non-symmetric beam: 90° rotation changes vis); `TestGroundLoss` (round-trip recovers same T_sky for two T_gnd; horizon=ones gives no ground); `TestMultipair` (auto matches convolve to 1e-10; identical antennas cross == auto; 90° azimuth rotation = ¼-sidereal-day time shift). |

Per `CLAUDE.md`, *physics tests must always pass*; if they break after a code change, the code is wrong, not the test. Several of these tests pin behavior down to `rtol=1e-10` against analytic answers, which is unusual for radio-astro test suites.

---

## 13. Known limitations and design choices

(verbatim from `IMPROVEMENTS.md`, condensed)

1. **Fixed beam-orientation epoch.** The topo→sim Euler angles are computed once at `times_jd[0]` and never updated. On Earth this introduces a ~50″/year precession error (cumulative) plus ~17″ nutation (bounded). Negligible for typical 21-cm runs (days–weeks); flagged as low-priority.

2. **Ground model is too simple.** Constant-T, isotropic, no scattering. Dedicated multi-pixel ground SHT pipeline would be a major design change and is out of scope.

3. **`beam_tilt` is not implemented.** Constructor raises. The IMPROVEMENTS.md doc spells out the design (Wigner-D about local Y, then re-apply horizon, then forward SHT).

4. **`sim()` recomputes the SHT every call.** Intentional, so `jax.grad` flows through `sky.data` / `beam.data`. Power users who don't need gradients can call `convolve` directly with cached `beam_eq_alm` and `sky_eq_alm`. `precompute_sky_alm` provides a partial shortcut but severs gradient flow through pixel data.

5. **HEALPix SHT compile time.** `s2fft method="jax"` with `niter=3` is "very slow" to compile but accurate, GPU-OK, gradient-OK. Default is `niter=0` (since v5.1.3) — fast compile, approximate, GPU-OK, gradient-OK. There is also a `method="jax_healpy"` path (CPU-only, partial gradients) not exposed via `Beam`/`Sky` constructors.

6. **`alt` is unused.** `EarthLocation`/`MoonLocation` are constructed with it, but the rotation matrices are translation-free. Aberration & parallax are negligible at radio frequencies. Documented as physically correct but the docstring should explain.

---

## 14. Version history (CHANGELOG.md highlights)

* **5.0.0** — Major: dropped numpy/healpy backend entirely; package is now JAX-only. `Beam` rewritten in JAX, `Sky` becomes a class (was previously a function), MCMF replaced by MEPA, switched to `uv` / `pyproject.toml`.
* **5.1.0** — `niter` keyword added to `SphBase`/`Beam`/`Sky`. ZYZ Euler convention default. `SphBase`/`Sky` classes split out. Various import/circular-dependency cleanups.
* **5.1.1** — `mepa et` parameter; capped `lru_cache` on `get_mepa_rotation_matrix`; doc fixes; notebooks updated to MEPA frame; uses `time.tdb.jd` for accurate phase.
* **5.1.2** — Swapped NEU→ENU axes in `get_rot_mat` for topocentric frames (`#110`). Determinant becomes +1; this changes the sign of any beam pattern that wasn't already in ENU.
* **5.1.3** — Default `niter=0` (was 3 for healpix). Major perf improvement at the cost of approximate HEALPix SHT.
* **5.1.4** — `beam_az_rot` (now `beam_rot`) follows N→E astronomical convention (`b8dbf34`). The phase factor is `exp(+i·m·beam_rot)`.
* **5.2.0** — `sim(sky_alm=...)` accepts a precomputed sky alm (`#117`).
* **5.2.1** — Gimbal-lock fix in `rotmat_to_eulerZYZ` at β=π (`15f6d21`, `#120`). The currently-checked-out version.

---

## 15. Notebooks

| Notebook | Purpose |
|---|---|
| `example_sim.ipynb` | Canonical demo. Builds a healpix Y₁⁰ + Y₀⁰ beam, fetches GSM2016 from pygdsm, runs both Moon and Earth sidereal day, makes waterfall + spectrum + temperature-vs-time plots. The "first thing you should run." |
| `croissant_jax.ipynb` | Older JAX-interface demo (pre-v5.0). |
| `multipair_example.ipynb` | Two short-dipole beams (X and Y) on the Moon, with ULSA sky. Demonstrates `compute_normalization` + `pair_normalization` + `compute_visibilities` and shows that beam norms are invariant under coordinate rotation. |
| `single_pixel.ipynb` | Single-pixel-source-impulse test of the system response. Compares two start times to demonstrate the MEPA-anchored phase rotation. |
| `mepa_precession.ipynb` | Visualizes how the MEPA reference epoch affects long-duration simulations. |
| `test_analytic_beams.ipynb` | Sanity-checks closed-form Gaussian / dipole beams against numerical SHT. |
| `beam.npy` / `ulsa.npy` | Cached inputs (~couple of MB). `ulsa.npy` is loaded by `single_pixel.ipynb` and `multipair_example.ipynb`. |

The CLAUDE.md says "tests enable 64-bit JAX precision globally via `conftest.py`" — notebooks need to call `jax.config.update("jax_enable_x64", True)` themselves, and the example notebooks all do this on their first cell.

---

## 16. Performance characteristics

Empirically (from notebooks and IMPROVEMENTS.md):

* Forward SHT compile times: `dh, gl` < `mw, mwss` < `healpix (niter=0)` < `healpix (niter=3)`.
* Run-time of `sim()` is dominated by the SHT, not the einsum. The einsum is `O(N_freqs · (lmax+1) · (2lmax+1) · N_times)`.
* `Simulator.sim` is `@jax.jit`'d, so the first call compiles (seconds–minutes) and subsequent calls with the same shape are sub-second. Changing `lmax`, `N_times`, or `N_freqs` retriggers compile.
* The phases array is static once `times_jd` is fixed.
* GPU support is automatic via JAX, since every op is `jax.numpy` / `s2fft`'s JAX backend. `method="jax_healpy"` is the only CPU-only path and is not used in production.
* Memory: `(N_pairs, N_freqs, lmax+1, 2lmax+1)` for multipair beam arrays at complex128 — at `lmax=128, N_freqs=50, N_pairs=8` this is ≈170 MB and easily fits on a laptop GPU.

---

## 17. Where CROISSANT fits relative to the other simulators in `simulators/`

* `pyuvsim`, `WODEN`, `RIMEz`, `matvis`, `fftvis`, `OSKAR`, `RASCIL`, `wsclean`, `DP3`, `meqtrees-cattery` — all per-baseline visibility simulators that follow the RIME `V_pq = ∫ J_p · S · J_q^H exp(-2πi (u·l + v·m + w·n)) dΩ`.
* `hera_sim`, `PRISim` — higher-level wrappers around RIME-style sims plus systematics injection.
* `pyradiosky` — sky-model I/O, used as input to several of the above.
* `healvis` — a closer analogue: HEALPix sky × beam → visibilities, but it computes a per-baseline sum (not a spherical-harmonic dot product) and uses healpy directly.

CROISSANT differs in three structural ways:
1. **Spherical-harmonic dot product** instead of pixel-by-pixel weighted sum. `O(L²)` operations per (time, freq) instead of `O(N_pixels)`.
2. **Antenna temperature, not interferometric V_pq.** No `(u, v, w)` geometry, no fringe term. (multipair gives you `T_pq(t,ν)`, the cross-power-beam-weighted sky.)
3. **Differentiable.** `jax.grad` flows through every parameter (beam pixel data, sky pixel data, `Tgnd`, frequency, etc.).

For RRIVis users: CROISSANT is the right reference for *single-element / autocorrelation forecasting* and for *gradient-based fits* of beam/sky parameters against an integrated-power dataset. It is the wrong reference for baseline-resolved Stokes-aware simulation — that work is done by `pyuvsim`/`WODEN`/`matvis` etc.

---

## 18. Quick API cheat-sheet

```python
import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import croissant as cro
from lunarsky import Time

freqs = jnp.linspace(50.0, 100.0, 5)   # MHz
nside = 16
npix = 12 * nside ** 2

# 1) Beam in local antenna ENU coordinates (HEALPix here, MWSS also fine)
beam = cro.Beam(
    data=jnp.ones((len(freqs), npix)),     # (N_freqs, npix), real-valued power beam
    freqs=freqs,
    sampling="healpix",                    # or "mwss", "mw", "dh", "gl"
    horizon=None,                          # default: theta <= pi/2
    beam_rot=0.0,                          # azimuthal rotation, deg, N→E
    beam_tilt=0.0,                         # not implemented; must be 0.0
    niter=0,                               # SHT iterations
)

# 2) Sky in galactic / equatorial / mepa coords, healpix or any equiangular
sky = cro.Sky(
    data=jnp.ones((len(freqs), npix)),     # (N_freqs, npix)
    freqs=freqs,
    sampling="healpix",
    coord="galactic",                      # or "equatorial", "mepa"
    niter=0,
)

# 3) Build a Simulator (Earth or Moon)
t0 = Time("2022-01-01 00:00:00")
times_jd = jnp.linspace(t0.jd, t0.jd + 1.0, 24)  # 24 steps over 1 day
sim = cro.Simulator(
    beam, sky, times_jd, freqs,
    lon=0.0, lat=0.0, alt=0.0,
    lmax=None,                             # default: min(beam.lmax, sky.lmax)
    world="moon",                          # or "earth"
    Tgnd=300.0,                            # K
)

# 4) Run
vis = sim.sim()                            # (N_times, N_freqs), real, K
# or, with precomputed sky alm (skips SHT, but breaks grad flow through sky.data)
sky_alm = sim.precompute_sky_alm()
vis = sim.sim(sky_alm=sky_alm)

# 5) Recover sky temperature
fgnd = sim.beam.compute_fgnd()
T_sky = cro.simulator.correct_ground_loss(vis, fgnd, sim.Tgnd)

# 6) Multi-pair (e.g. two beams)
beam_alm_pair = jnp.stack([sim.compute_beam_eq(), sim.compute_beam_eq()], 0)
auto_powers = cro.multipair.compute_normalization(beam_alm_pair)
norm = cro.multipair.pair_normalization(auto_powers, [(0, 0), (0, 1)])
phases = cro.simulator.rot_alm_z(sim.lmax, times=jnp.arange(24)*3600., world="moon")
sky_alm_eq = sim.sky.compute_alm_eq(world="moon", et=sim._et_ref)
vis_pair = cro.multipair.compute_visibilities(beam_alm_pair, sky_alm_eq, phases, norm)
# (N_times, N_pairs, N_freqs)
```

---

## 19. Gotchas and integration footnotes for RRIVis

1. **Power beams, not voltage beams.** `Beam.data` is `|E|²`, real-valued. RRIVis stores Jones matrices `J(θ,φ)` from FITS or analytic models; to feed CROISSANT you would compute `|E_x|² + |E_y|²` (or per-Stokes coherency element) before constructing `Beam`. There is no Stokes/polarization machinery in CROISSANT — it is implicitly Stokes-I.
2. **Single antenna, no `(u,v,w)`.** As above. RRIVis's `core/visibility.py` (`V_pq = Σ J_p C J_q^H`) is *strictly* more general; CROISSANT corresponds to the special case `p == q` and `w·n` term collapsed to zero (because there is no baseline geometry).
3. **Frequency in MHz.** `Simulator.freqs` are MHz. RRIVis tends to use Hz internally (especially in `core/visibility.py`); convert before passing.
4. **Times in JD.** `Simulator.times_jd` are float JD (TDB or UTC — both are accepted by `astropy.time.Time(format='jd')`; v5.1.1 made the internal phase use `time.tdb.jd` for better accuracy). RRIVis uses `astropy.time.Time` instances.
5. **HEALPix is the only payload format that survives a SHT cleanly across both pyuvsim/WODEN-style and CROISSANT.** The MWSS grid is the s2fft "best" choice but is not natively supported by RRIVis's diffuse-sky pipeline. If integrating, the easiest path is to push HEALPix into `croissant.Sky(sampling="healpix")` directly.
6. **Reference epoch on the Moon matters.** `Simulator(world="moon", times_jd=...)` anchors MEPA at `times_jd[0]`. Do *not* mix outputs across simulators that have different start times — see `precompute_sky_alm` docstring and `test_moon_sim_depends_on_start_time`.
7. **`niter=0` HEALPix SHT is approximate.** When comparing CROISSANT outputs to other simulators with bandlimited-exact SHTs, set `niter=3` or use an MWSS grid — otherwise discrepancies of `~1e-3` are normal at moderate `lmax`.
8. **Output dtype.** `vis.real` is float32 unless the user enables `jax_enable_x64`. RRIVis tests typically run float64. Set `jax.config.update("jax_enable_x64", True)` before instantiating any CROISSANT object.
9. **No backwards compatibility shims for v4.x.** `croissant.alm` and `croissant.jax` exist only for v4→v5 import compat. Anything more substantial than `from croissant.utils import getidx` should not be expected to work.
10. **Spice kernel files** are pulled by `lunarsky` lazily. First-time MEPA usage will download SPICE kernels (~10 MB) into `~/.spiceypy/` or wherever `spiceypy` keeps them. Run a small Moon test once during environment setup to pre-fetch.

---

## 20. One-line summary

CROISSANT = a JAX-native, differentiable, sphere-harmonic-domain antenna-temperature simulator for global-21cm-style experiments on Earth or the Moon, built around `T_ant(t,ν) = (Σₗₘ aₗᵐ_beam(ν) · phase_m(t) · aₗᵐ*_sky(ν)) / ∫ A dΩ + f_gnd · T_gnd` evaluated via `s2fft` and `equinox`.
