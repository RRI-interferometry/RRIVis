# Tier 5 Receptor, Feed, and Polarization Basis Integration Plan

## 1. Identity, status, and governing sources

| Fact | Value |
|---|---|
| Status | Design accepted; 5A independently accepted after bounded corrections (`568855f`, and this Tier 5A acceptance); 5B authorized. Implementation not started. |
| Date | 2026-07-29 |
| Repository | `/Users/kartikmandar/MacProjects/RadioSim` |
| Branch | `main` |
| Baseline | `1472c3c` (`docs(output): accept Tier 4 integration`) |
| Baseline parent | `93bff96` (`docs(output): accept Tier 4H obsolete-path removal`) |
| Governing roadmap | `Fix.md` §4, §5, §6.7, §7.5, §14 |
| Prior accepted architecture | `Tier2InstrumentPlan.md`, `Tier3BeamObservabilityPlan.md`, `Tier4ResultOutputPlan.md` |
| Repository policy | `CLAUDE.md`, `AGENTS.md`, including the pre-v1 direct-replacement policy |
| Issues in scope | `POL-001` (OPEN), `POL-002` (ROADMAP) |

This document is the governing implementation specification for Tier 5. It is
not Tier 5 implementation. Every characterization statement below was taken
from the working tree at `1472c3c` and every cited line number is true at that
commit. Where a fact could not be established from source at this gate it is
recorded as an open question in Section 43 rather than asserted.

## 2. Design-only authority

This gate changes exactly two tracked files:

- this file, `Tier5ReceptorFeedPlan.md`, newly added at the repository root;
- one current-status note appended to `Fix.md`.

It adds no production behavior, no test, no fixture, no configuration value,
no dependency, no lockfile change, no generated artifact, and no CI behavior.
It does not modify the `Fix.md` §5 issue register rows and does not modify any
prior acceptance record. Every implementation slice in Section 34 requires a
separate authorization and an independent acceptance after its implementation
commit.

## 3. Tier 4 dependency and acceptance state

Tier 4 is independently accepted as a whole at the baseline (Tier 4I,
2026-07-29). `OUT-001` through `OUT-006` are `DONE`. Tier 5 preserves without
reopening:

- the canonical `ObservationTimeGrid`, `PhaseCenter`, and authoritative time
  grid;
- the immutable `SimulationResult` / `LoadedSimulationResult` lifecycle,
  ownership, and equality semantics;
- the `(time, baseline, frequency, correlation)` dimension order and the
  `(T, B, F, 4)` public array shape;
- the versioned safe HDF5 transaction, hostile-input limits, and the
  no-dynamic-evaluation rule;
- the single standard-format projection shared by Measurement Set and UVFITS,
  including the explicit zenith-to-ICRS phase projection and the
  `RADIOSIM_PROJECTION_JSON=` history record;
- the four collision policies, staged run directory, atomic publish, manifest,
  and browser-last ordering;
- failure-before-side-effect ordering for configuration, instrument, beam,
  observability, solver, result, and writer work.

Tier 5 changes what the correlation axis *means* and what populates it. It does
not change the axis position, the array shape, the time grid, the phase
contract, the workflow transaction, or the collision policy.

Tier 3 remains accepted. Tier 5 preserves the single Simulator-local
`BeamSystem`, the immutable `LoadedBeamState`, and the accepted scalar E-Jones
boundary described in Section 6.2.

## 4. Current architecture and the terminology split

### 4.1 The two meanings of "feed"

`Fix.md` §6.7 records that RadioSim historically used "feed" for two unrelated
physical concepts. At the baseline the split is *partially* resolved:

1. **Aperture illumination** — how a horn, waveguide, or dipole illuminates a
   reflector. Tier 3 already renamed this at the configuration boundary. The
   live path is `beams.model.kind ∈ {analytical_illumination,
   numerical_illumination}` with `beams.model.illumination.kind ∈
   {corrugated_horn, open_waveguide, dipole_ground_plane}`
   (`src/radiosim/io/beam_config.py:128-157`, `:205-221`). The legacy names
   `beams.feed_model`, `beams.feed_computation`, and `beams.feed_params` are
   rejected with typed guidance
   (`src/radiosim/io/config.py:1629-1641`, `:1679-1716`).
2. **Receiving receptors** — the orthogonal pair of polarization-sensitive
   elements that convert the incident field into two voltage streams. This is
   Tier 5's subject. It has **no** configuration surface at the baseline.

The rename is therefore complete at the *configuration* boundary and incomplete
at the *identifier* boundary. `src/radiosim/core/beam/analytic.py` still names
the illumination evaluator `_feed_response` (line 302), the illumination edge
angle helper `_feed_angles` (line 330), and its argument `theta_feed`
(lines 195, 204, 224, 304), and an orphaned duplicate of the same physics lives
in `src/radiosim/core/jones/beam/analytic/feed.py`. Those identifiers are
illumination physics and must stop using receptor vocabulary. Conversely
`feed_array` and `feed_angle` in `src/radiosim/core/beam/fits.py:473-505` and
`src/radiosim/core/beam/models.py:662,707-709` are pyuvdata dependency
attribute names describing the *receiving receptor* and are correctly named;
Tier 5 must not rename them.

### 4.2 What POL-001 actually is at the baseline

`Fix.md` §5 states POL-001 as "Top-level feed/receptor config is ignored". That
wording describes the pre-Tier-1 tree. At `1472c3c` the defect has changed
shape and must be restated truthfully:

- there is **no** `FeedsConfig` class anywhere in `src/`;
- `RadioSimConfig` (`src/radiosim/io/config.py:1695-1708`) declares exactly nine
  sections — `instrument`, `beams`, `baseline_selection`, `sky_model`,
  `obs_time`, `obs_frequency`, `visibility`, `execution`, `workflow` — and no
  `feeds`;
- `RadioSimConfig` is a `StrictFrozenModel` with `extra="forbid"`
  (`src/radiosim/io/model_base.py:9`), so a `feeds:` block is a hard schema
  failure, mapped to a `removed_field` `ConfigIssue`
  (`src/radiosim/io/config.py:2259-2263`) using the guidance at
  `src/radiosim/io/config.py:2023-2026`.

So POL-001 is no longer "silently ignored"; Tier 1 converted it to state 3 of
the §4.2 truthfulness rule (unsupported and rejected). Tier 5 closes POL-001 by
moving receptor configuration from state 3 to state 1 (implemented and tested).

One live truthfulness defect survives in that guidance text. The `feeds` hint
reads:

```text
Analytic-beam illumination remains under 'beams.feed_model'; receptor physics belongs to Tier 5.
```

`beams.feed_model` is itself rejected (`src/radiosim/io/config.py:1629-1632`).
The hint therefore directs the user to a second removed field. Tier 5 repairs
it.

### 4.3 What POL-002 actually is at the baseline

`src/radiosim/core/jones/receptor.py` is 103 lines and defines two public
classes exported from `radiosim.core.jones`
(`src/radiosim/core/jones/__init__.py:118-119`, `:183-184`):

- `ReceptorConfigJones` (line 20), `name` = `"C"` (line 37),
  `is_direction_dependent` = `False`, `is_unitary()` = `True` (line 44),
  constructor `__init__(self, feed_type: str = "linear", **kwargs)` (line 31)
  which stores `self.feed_type` and never reads it;
  `compute_jones()` (lines 46-57) returns `xp.eye(2, dtype=np.complex128)`
  unconditionally.
- `BasisTransformJones` (line 60), `name` = `"H"` (line 82), constructor
  `__init__(self, from_basis="linear", to_basis="circular", **kwargs)`
  (lines 73-78) which stores both and never reads them; `compute_jones()`
  (lines 91-102) returns `xp.eye(2, dtype=np.complex128)` unconditionally.

The module docstring (lines 1-11) already states the intended circular matrix
`C = [[1, i], [i, 1]] / sqrt(2)`, which is documentation of unimplemented
physics and is not the matrix this plan adopts (Section 18). Both classes are
`is_unitary() -> True` while returning identity, which is vacuously true and
misleading. Neither class is instantiated anywhere in `src/`.

## 5. Target architecture

```
receptors: (new top-level typed config section)
        |
        v
resolve_receptors()  ->  ResolvedReceptorSet  (frozen, AntennaId-keyed,
        |                                       own receptor_sha256)
        |                                       sibling of BeamSystem,
        |                                       does NOT reopen instrument_sha256
        v
_build_jones_chain()  ->  JonesChain with explicit canonical order including
        |                 C (ReceptorConfigJones) and H (BasisTransformJones)
        v
core/visibility.py, core/visibility_healpix.py
        |                 V_pq = J_p C_sky J_q^H, unchanged equation
        v
SimulationResult.correlations / .polarization_basis  (data-driven, not literals)
        |
        +--> io/hdf5.py            (schema 2.0.0, basis recorded and validated)
        +--> io/standard_visibility.py -> MS, UVFITS  (basis-aware AIPS codes)
        +--> io/summary_json.py    (already data-driven; gains receptor block)
        +--> visualization/        (parallel hands derived, not indexed 0/3)
```

The sky brightness basis is fixed and single (Section 10). Every polarization
degree of freedom that varies per antenna lives in `ResolvedReceptorSet` and
reaches the solver only through the Jones chain.

## 6. Current source and test inventory

### 6.1 Configuration inventory

| Fact | Location |
|---|---|
| Nine top-level sections, no `feeds` | `src/radiosim/io/config.py:1695-1708` |
| `extra="forbid"` strict frozen base | `src/radiosim/io/model_base.py:9` |
| `feeds` removed-field guidance (stale hint) | `src/radiosim/io/config.py:2023-2026` |
| `feed_model` / `feed_computation` / `feed_params` beam guidance | `src/radiosim/io/config.py:1629-1641` |
| Removed-beam-field validator | `src/radiosim/io/config.py:1710-1716`, helper `_legacy_beam_fields` at `:1679`, text at `:1688` |
| `extra_forbidden` → `removed_field` mapping | `src/radiosim/io/config.py:2259-2263` |
| Close-match "Did you mean" hinting | `src/radiosim/io/config.py:2108`, `:2271` |
| Illumination configs (live) | `src/radiosim/io/beam_config.py:128-157` |
| Illumination-bearing beam models | `src/radiosim/io/beam_config.py:205-221` |
| `VisibilityConfig` (3 fields, none polarization) | `src/radiosim/io/config.py:1370-1377` |
| Tagged antenna reference pattern to mirror | `src/radiosim/io/instrument_config.py:165-188` |
| Per-antenna override pattern to mirror | `src/radiosim/io/instrument_config.py:184-209` |

`baseline_selection.correlations` is a `Literal["all","auto","cross"]` antenna-pair
filter and is **not** a polarization control. It must not be conflated with the
new receptor surface.

No file under `configs/` or `examples/` sets any feed or receptor field.

### 6.2 Polarization and Jones inventory

**`src/radiosim/core/polarization.py`** (465 lines). Public surface:
`stokes_to_coherency` (49), `apply_jones_matrices` (136),
`visibility_to_correlations` (217), `stokes_I_only_visibility` (288),
`coherency_to_stokes` (330), `jones_matrix_power` (391), `mueller_from_jones`
(426, raises `NotImplementedError` at 461-464).

Coherency construction, lines 112-131:

```python
row_x = xp.stack([stokes_I + stokes_Q, stokes_U - 1j * stokes_V], axis=-1)
row_y = xp.stack([stokes_U + 1j * stokes_V, stokes_I - stokes_Q], axis=-1)
coherency = xp.stack([row_x, row_y], axis=-2)
coherency = coherency / 2.0
```

The half-power factor is line 131. The module docstring (lines 22-27) asserts
`C[0,1] = (U - iV) / 2` and labels it the "Africanus/Pauli" convention,
explicitly contrasting it with "Smirnov 2011". The inverse
`coherency_to_stokes` (lines 377-386) is consistent with that choice, deriving
`stokes_V = 2 * coherency[..., 1, 0].imag`. `visibility_to_correlations`
returns a dict hard-keyed `"XX"`, `"XY"`, `"YX"`, `"YY"`, `"I"` (lines 277-282).
There is no basis parameter anywhere in the module and no circular path.

**`src/radiosim/core/jones/base.py`.** `JonesTerm.compute_jones` abstract
signature (lines 132-141):

```python
def compute_jones(
    self,
    antenna_idx: int,
    source_idx: int | None,
    freq_idx: int,
    time_idx: int,
    backend: Any,
    **kwargs,
) -> Any:
```

Its docstring (lines 142-160) states the return is `(2, 2)` "in linear
polarization basis [X, Y]". There is no antenna batch dimension; per-antenna
variation is expressed by calling once per `antenna_idx`. The optional
`compute_jones_all_sources` (lines 199-231) vectorizes only the source axis.

**`src/radiosim/core/jones/chain.py`.** `compute_antenna_jones` starts from
`xp.eye(2)` (line 159), iterates `for term in reversed(self.terms):` (line 166)
and accumulates `J_total = self.backend.matmul(J_term, J_total)` (line 184).
For `terms = [t0, t1, ..., tn]` this yields `J_total = t0 @ t1 @ ... @ tn`, so
the **first-added** term is leftmost and is applied **last** to the sky field.
`compute_antenna_jones_all_sources` (line 188) is the batched equivalent using
`backend.batch_eye` (line 215) and the same reversed accumulation (line 246).

**`src/radiosim/core/visibility.py`.** `_build_jones_chain` (lines 597-721)
constructs `JonesChain(backend)` at line 644 and adds terms in the order
`Z` (647-657), `T` (659-667), `E` (669-678, **always**), `P` (680-691),
`D` (693-700), `G` (702-709), `B` (711-719). Combined with the chain semantics
above this produces `J = Z T E P D G B`, i.e. the bandpass is applied to the
sky field first and the ionosphere last. That is inverted relative to the
Hamaker–Bregman–Sault / Smirnov ordering `J = G B D P E T Z K`. It is currently
unobservable because every term except `E` is an identity stub, every optional
term defaults to disabled, and `E` is constrained to a scalar multiple of the
identity (below), so all present factors commute. Section 19 fixes it.

The chain configuration is a raw untyped `dict`
(`src/radiosim/core/visibility.py:210`, validated only for type and for a
forbidden `"beam"` key at `:258-264`), and the high-level API passes
`jones_config=None` (`src/radiosim/api/simulator.py:945`). No typed
configuration reaches any Jones term today.

**E-Jones scalar boundary.** `src/radiosim/core/beam/runtime.py:372-388`
rejects any interpolated beam whose cross-polar terms exceed tolerance or whose
`X`/`Y` diagonals differ, then rebuilds the response as `scalar * I2`. The
accepted E term is therefore exactly `e·I₂`. This is load-bearing for Tier 5:
**E commutes with every receptor matrix**, so basis transforms are exact and
order-independent with respect to E, and cross-hand visibilities in Tier 5 arise
solely from sky `Q`/`U`/`V` and from receptor geometry — never from the beam.

**Beam FITS receptor constraints.** `src/radiosim/core/beam/fits.py:473-505`
requires `feed_array == ("x", "y")`, `x_orientation == "east"`, and
`feed_angle ≈ (π/2, 0)` radians within `_FEED_ANGLE_TOLERANCE_RAD`.
`src/radiosim/core/beam/models.py:707-709` repeats the `("x","y")` requirement
on the resolved metadata.

**Solver shapes.** `src/radiosim/core/visibility.py:292-295` allocates
`(n_times, n_baselines, n_freq, 2, 2)`. The per-antenna Jones cache
(lines 524-539) holds `(n_sources, 2, 2)` per antenna number. The RIME is
applied at lines 572-587 as `V = J_p @ C @ J_q^H` with an unpolarized fast path.
`src/radiosim/core/visibility_healpix.py` mirrors this with a polarized path
(lines 341-423) and a scalar `C = (I/2) I₂` path (lines 425-485); it calls
`beam_system.evaluate_jones` directly and **never constructs a `JonesChain`**.

**Instrument.** `ResolvedAntenna`
(`src/radiosim/core/instrument.py:309-318`) carries `id`, `position_enu_m`,
`diameter_m`, `mount_type`, `beam_id`, `provenance`. `ResolvedInstrument`
(`:638-645`) carries `name`, `location`, `antennas`, `provenance` and validates
`instrument_sha256` in `__post_init__` (`:647-671`) against
`_canonical_instrument_fingerprint_payload` (`:535-609`). Per-antenna beams are
**not** stored on the instrument; they are keyed separately by `AntennaId`
inside `BeamSystem` (`src/radiosim/core/beam/runtime.py:471-506`,
`LoadedBeamState.assignment_handler_ids` in
`src/radiosim/core/beam/models.py:1237`). Tier 5 follows the beam precedent,
not the instrument precedent (Section 17).

### 6.3 Result and serialization inventory

The correlation contract is currently fixed at **four independent hard-coded
sites** that stay in lockstep only by convention:

| Site | Location | Content |
|---|---|---|
| Result model | `src/radiosim/core/result.py:32` | `_CORRELATIONS = ("XX", "XY", "YX", "YY")` |
| HDF5 | `src/radiosim/io/hdf5.py:58-60` | `CORRELATIONS`, `AIPS_CODES = (-5, -7, -8, -6)` |
| Standard formats | `src/radiosim/io/standard_visibility.py:29-31` | `CANONICAL_CORRELATIONS`, `CANONICAL_CODES = [-5,-7,-8,-6]`, `FILE_CODES = [-5,-6,-7,-8]` |
| pyuvdata construction | `src/radiosim/io/standard_visibility.py:887`, `:898` | `feeds=["x","y"]`, `polarization_array=["xx","xy","yx","yy"]` |

Additional hard-coded polarization facts:

- `src/radiosim/core/result.py:408` hashes the literal string `"linear_xy"`
  into the scientific fingerprint rather than the field value.
- `src/radiosim/core/result.py:951`, `:987-988`, `:1117`, `:1145-1146` assign
  `correlations=_CORRELATIONS` and `polarization_basis="linear_xy"` as literals
  in every construction path.
- `src/radiosim/core/result.py:512-519` `stokes_i()` returns
  `visibilities[..., 0] + visibilities[..., 3]` without consulting
  `self.correlations`.
- `src/radiosim/core/result.py:827` requires `visibilities.shape[-1] == 4`;
  `:920` performs the `(2,2) → 4` row-major reshape that implicitly defines
  `[0,0],[0,1],[1,0],[1,1] → XX,XY,YX,YY`.
- `src/radiosim/core/result.py:1043-1044` rejects any loaded correlation tuple
  that is not exactly `("XX","XY","YX","YY")`.
- `src/radiosim/io/hdf5.py:671-679` writes labels and AIPS codes from the module
  constants, not from `result.correlations`; `:1833-1842` rejects any file whose
  stored labels or codes differ from those constants; `:1404-1405` requires
  exactly four correlations.
- `src/radiosim/io/standard_visibility.py:478-482` and `:686-688` reject any
  correlation tuple other than the canonical four, including reorderings;
  `:727` iterates `for correlation_index in (0, 3):` to fix parallel-hand
  autocorrelations; `:925-932` asserts pyuvdata preserved `CANONICAL_CODES`;
  `:1085`, `:1114`, `:1465` gate the read path on the same four codes.
- `src/radiosim/io/measurement_set.py:595-604` rejects `NUM_CORR != 4`.
- `src/radiosim/io/uvfits.py:125-126`, `:305-307` reject anything but the
  canonical four.
- `src/radiosim/io/summary_json.py:257` reports the correlation axis count and
  `:278-281` already emits `{"labels": list(result.correlations), "basis":
  result.polarization_basis}` — this site is already data-driven and requires
  no structural change.
- `src/radiosim/visualization/bokeh_plots.py:93-106` obtains Stokes I only via
  `result.stokes_i()` and never indexes correlations directly, so the plot layer
  inherits correctness from the result model.

No AIPS circular code (`-1`, `-2`, `-3`, `-4`) and no casacore circular Stokes
enum value appears anywhere in `src/`.

### 6.4 Test and truth surfaces

`tests/unit/test_jones/` contains only `test_backend_jones.py` and
`test_beam_analysis.py`. **No test exercises `ReceptorConfigJones` or
`BasisTransformJones`.** `tests/integration/` contains only `__init__.py`.

Tests that pin the current polarization contract and will need coordinated
change:

| Test | Pin |
|---|---|
| `tests/unit/test_core/test_result.py::test_result_factory_flattens_correlations_once_and_hardens_all_arrays` (lines 354-360) | `(2,2)→4` mapping, `correlations`, `polarization_basis == "linear_xy"` |
| `tests/characterization/test_tier4_current_behavior.py::test_run_publishes_one_immutable_canonical_result` (lines 158-165) | end-to-end labels and `I = XX + YY` |
| `tests/unit/test_simulator/test_result_integration.py::test_run_publishes_one_canonical_result_with_exact_axes` (lines 60-67) | same, via `Simulator.run()` |
| `tests/unit/test_core/test_beam_solver_integration.py` (lines 424-426, 809-812, 854-857) | unpolarized `XX == YY`, `XY == YX == 0`, half-power parallel hands, flat index order |
| `tests/unit/test_io/test_hdf5_result.py` (lines 621-622, 1171-1177, 1212) | stored labels/codes and hostile reordering rejection |
| `tests/unit/test_io/test_standard_visibility.py` (lines 198, 338-351, 430) | canonical tuple, reorder rejection, in-memory AIPS order |
| `tests/unit/test_io/test_measurement_set.py:117`, `tests/unit/test_io/test_uvfits.py:66,78` | round-trip labels and on-disk code order |

`tests/unit/test_core/test_beam_solver_integration.py:279` is the only test that
calls `stokes_to_coherency`; the repository contains **no test that pins the
Stokes `V` sign convention against an external reference**. That is precisely
why the discrepancy in Section 10.2 has survived.

## 7. Current data-flow trace

1. `load_config()` validates `RadioSimConfig`. A `feeds:` block fails here with
   a `removed_field` issue. No receptor information exists past this point.
2. `Simulator.setup()` resolves the instrument (Tier 2) and the `BeamSystem`
   (Tier 3). Neither carries receptor state.
3. `Simulator.run()` calls the solver with `jones_config=None`
   (`src/radiosim/api/simulator.py:945`).
4. `_build_jones_chain` adds only `E` (`src/radiosim/core/visibility.py:669-678`).
   `E` is `e·I₂`.
5. The solver evaluates `V_pq = J_p C J_q^H` into a `(T, B, F, 2, 2)` cube.
6. `build_simulation_result` reshapes to `(T, B, F, 4)`
   (`src/radiosim/core/result.py:920`) and stamps the literal labels and basis.
7. Every writer re-asserts the same literals independently.

The consequence is that RadioSim always produces four linear correlations
because the beam is always a scalar and the labels are always literals, not
because any receptor was modelled.

## 8. Confirmed defect matrix

| # | Defect | Evidence | Closes |
|---|---|---|---|
| D1 | No receptor configuration surface exists; `feeds:` is rejected with guidance that points at a second removed field | `src/radiosim/io/config.py:1695-1708`, `:2023-2026`, `:1629-1632` | POL-001 |
| D2 | `ReceptorConfigJones` and `BasisTransformJones` accept basis arguments and return identity, while advertising `is_unitary() -> True` | `src/radiosim/core/jones/receptor.py:31-57`, `:73-102` | POL-002 |
| D3 | `stokes_to_coherency` uses `C[0,1] = (U - iV)/2`, opposite to the RIME literature's `(U + iV)/2`, and no test pins it | `src/radiosim/core/polarization.py:22-27`, `:112-131`, `:377-386` | POL-002 |
| D4 | Correlation labels, AIPS codes, and basis are hard-coded at four independent sites and cannot express any other basis | `src/radiosim/core/result.py:32`, `src/radiosim/io/hdf5.py:58-60`, `src/radiosim/io/standard_visibility.py:29-31`, `:887-898` | POL-001 |
| D5 | `polarization_basis` is hashed as a literal, so the scientific fingerprint cannot distinguish two bases | `src/radiosim/core/result.py:408` | POL-001 |
| D6 | `stokes_i()` indexes correlations `0` and `3` without consulting `self.correlations` | `src/radiosim/core/result.py:512-519` | POL-001 |
| D7 | `JonesChain` composition order is inverted relative to HBS/Smirnov; currently unobservable because all present factors commute | `src/radiosim/core/jones/chain.py:166,184`, `src/radiosim/core/visibility.py:647-719` | POL-002 |
| D8 | Illumination physics still uses receptor vocabulary in identifiers, and an orphaned duplicate implementation exists | `src/radiosim/core/beam/analytic.py:302,330`, `src/radiosim/core/jones/beam/analytic/feed.py` | POL-001 |
| D9 | `visibility_to_correlations` returns hard-keyed `"XX"/"XY"/"YX"/"YY"` and cannot express a circular basis | `src/radiosim/core/polarization.py:277-282` | POL-002 |

## 9. Scientific conventions and citations

The following references govern every decision in Sections 10-15. Each is named
so that an independent reviewer can check the resulting matrices against a
published table rather than against this document.

| Ref | Source | Used for |
|---|---|---|
| R1 | Hamaker, Bregman & Sault 1996, A&AS 117, 137 (RIME Paper I) | measurement equation form, Jones factor ordering |
| R2 | Hamaker & Bregman 1996, A&AS 117, 161 (Paper III) | linear/circular receptor response tables |
| R3 | Smirnov 2011, A&A 527, A106 (Revisiting the RIME, Paper I) | brightness matrix `B`, Jones chain order `J = G B D P E T Z K` |
| R4 | Thompson, Moran & Swenson, *Interferometry and Synthesis in Radio Astronomy*, 3rd ed., §4.7 | Stokes-to-correlation relations for linear and circular feeds |
| R5 | IAU 1973 Commission 40 resolution; IEEE Std 145 | sign of Stokes `V`, sense of RCP/LCP |
| R6 | AIPS Memo 117 (Greisen), polarization code table | `RR=-1, LL=-2, RL=-3, LR=-4, XX=-5, YY=-6, XY=-7, YX=-8` |
| R7 | pyuvdata 3.2.1 source, installed in this workspace | `feed_array`, `feed_angle`, `polarization_array` semantics |
| R8 | casacore `Stokes` enumeration | Measurement Set `CORR_TYPE` values |

### 9.1 The conventions this plan adopts

- **Stokes definition (R5).** `I` total intensity, `Q`/`U` linear referenced to
  the celestial frame with position angle measured from North through East,
  `V = RCP − LCP` in the IAU sense.
- **Brightness matrix (R3, R4).** In a right-handed linear sky basis
  `(X → North, Y → East)` mapped onto the receptor's nominal `(x, y)`:

  ```
  B = 1/2 * [[ I + Q,  U + iV ],
             [ U - iV,  I - Q ]]
  ```

  The `1/2` is RadioSim's existing half-power convention and is retained, so
  `V_xx + V_yy = I` and `V_RR + V_LL = I`.
- **Jones ordering (R1, R3).** `J_p = G_p B_p D_p P_p C_p E_p T_p Z_p K_p`,
  leftmost factor nearest the correlator, applied last to the sky field.
- **Circular basis (R2, R4).** `R` right-hand circular, `L` left-hand circular,
  in the IAU sense consistent with `V = RCP − LCP`.
- **Polarization codes (R6).** AIPS codes as tabulated above. `pyuvdata`
  reproduces them exactly (`utils/pol.py:37,43,58` in the installed 3.2.1).

## 10. Design decision 1 — internal sky polarization basis

### 10.1 Decision

**RadioSim keeps exactly one internal sky polarization basis: the linear
`(x, y)` basis, with the IAU/HBS brightness matrix**

```
B = 1/2 * [[I + Q, U + iV], [U - iV, I - Q]]
```

The sky basis is never rewritten for a circular array. Receptor basis is
expressed entirely on the receptor side, as a per-antenna Jones factor. There is
exactly one function that constructs `B` — `stokes_to_coherency` in
`src/radiosim/core/polarization.py` — and exactly one function that inverts it —
`coherency_to_stokes`.

Rationale:

- Both solver paths already build `B` in the linear basis
  (`src/radiosim/core/visibility.py:505-507`,
  `src/radiosim/core/visibility_healpix.py:398`), and both sky payloads store
  `I/Q/U/V` directly. Keeping the sky basis fixed means Q/U/V sign conventions
  have exactly one home.
- Heterogeneous arrays (Section 13) require per-antenna receptor matrices in any
  case. Once those exist, rewriting the sky basis buys nothing and doubles the
  number of sign conventions that must be tested.
- The accepted E-Jones is `e·I₂` (Section 6.2), so `E` commutes with any
  receptor factor and the decomposition
  "sky-linear brightness × receptor Jones" is exact, not approximate.

### 10.2 The Stokes V sign correction (defect D3)

The baseline builds `C[0,1] = (U − iV)/2`
(`src/radiosim/core/polarization.py:112-113`) and documents this as the
"Africanus/Pauli" convention, explicitly contrasted with Smirnov 2011
(`:22-27`). References R2, R3, and R4 all give `XY = U + iV` for linear feeds
under the IAU `V` definition (R5). The baseline matrix is therefore the mirror
of the literature matrix under `V → −V`.

In the linear basis this is only observable in `XY`/`YX` for sources with
`V ≠ 0`, and no test pins it (Section 6.4), which is why it has survived. In a
circular basis it is immediately observable: a source with `V = +I` would emerge
as pure `LL` instead of pure `RR` under any standard basis transform.

**Decision.** Tier 5 corrects `stokes_to_coherency` to
`row_x = [I + Q, U + 1j*V]`, `row_y = [U - 1j*V, I - Q]`, and correspondingly
corrects `coherency_to_stokes` to derive `V` from the `[0,1]` element:
`stokes_V = 2 * coherency[..., 0, 1].imag`. Both must change together; changing
one alone breaks the existing round-trip invariant.

This is an intentional pre-v1 breaking scientific change. Its blast radius is
exactly: visibilities of sources with non-zero Stokes `V`, in the cross-hand
correlations only. All `V = 0` results, all parallel hands, and all
`stokes_i()` values are bit-identical before and after.

Slice 5A must produce written evidence for this correction — a reproduction of
the R2/R4 table and an explicit statement of what `codex-africanus` actually
implements — before slice 5C changes the sign. If 5A's evidence contradicts this
section, the design must be amended and re-accepted, not silently followed. See
Section 43, open question Q1.

## 11. Design decision 2 — supported receptor bases

### 11.1 Decision

Tier 5 supports exactly two receptor bases:

| Value | Feeds | Nominal `feed_array` | Meaning |
|---|---|---|---|
| `linear` | `x`, `y` | `("x", "y")` | two orthogonal linear dipoles/probes |
| `circular` | `r`, `l` | `("r", "l")` | right- and left-hand circular receptors, IAU sense |

Every antenna has exactly two feeds. Both feeds of an antenna share the same
basis. Both feeds of an antenna are ideal and orthogonal.

### 11.2 Explicitly rejected

Each rejection is a typed error with an actionable message (Section 25, 27):

| Rejected | Reason | Deferred to |
|---|---|---|
| any `basis` value other than `linear` / `circular` | not implemented | — |
| `stokes` as a receptor basis | Stokes is a sky representation, not a receptor | — |
| single-feed (`Nfeeds = 1`) antennas | a homogeneous four-correlation axis is undefined | Tier 7 |
| more than two feeds per antenna | same | Tier 7 |
| elliptical or non-orthogonal feed pairs | requires the D-term (leakage) to be real | Tier 7 (SCI-001) |
| independent per-feed angles on one antenna | equivalent to a non-orthogonal pair | Tier 7 |
| frequency-dependent or time-dependent receptor basis | requires B/G terms to be real | Tier 7 |
| `mount_type` other than `fixed` | requires the P-term (parallactic angle), a Tier 7 stub | Tier 7 |

The last row is load-bearing. `src/radiosim/core/beam/models.py:696-706`
already requires `mount_type == "fixed"` for FITS beams, and
`src/radiosim/io/standard_visibility.py:889` writes `mount_type="fixed"`.
Tier 5 makes that requirement explicit for receptors as well, because a
steerable mount makes the feed angle time-dependent through the parallactic
angle, and `ParallacticAngleJones` returns identity at the baseline.

### 11.3 Modelling assumption that must be documented

Converting a circular-native antenna into a linear output basis (or the
reverse) is exact **only** when both feeds are ideal, orthogonal, and share a
common complex gain. That is true in Tier 5 because `D` (leakage) and `G`
(gains) are identity stubs and disabled. When Tier 7 implements `D`, the
conversion becomes approximate and must be re-examined. The plan requires this
statement to appear in `docs/user_guide/jones_matrices.rst` and in the
`ResolvedReceptorSet` docstring, not only here.

## 12. Design decision 3 — feed-angle convention and frame

### 12.1 Decision

RadioSim adopts the pyuvdata 3.2.1 feed-angle convention (R7) verbatim, so that
no translation layer is needed at the writer boundary:

- **Frame.** Topocentric horizontal (az/alt) frame of the antenna, which is the
  frame in which the accepted beams are already defined
  (`pixel_coordinate_system == "az_za"`,
  `src/radiosim/core/beam/fits.py:465-471`).
- **Definition.** `feed_angle` is the position angle of one feed, in radians,
  measured from **North toward East**, for a fixed-mount antenna. A feed angle
  of `0` points North.
- **Nominal linear pair.** `feed_array = ("x", "y")` with
  `feed_angle = (π/2, 0)` — the `x` feed toward East, the `y` feed toward North.
  This is exactly what `src/radiosim/core/beam/fits.py:490-505` already enforces
  and what `x_orientation="east"`
  (`src/radiosim/io/standard_visibility.py:887`) encodes. pyuvdata's own
  reference telescopes use the same values
  (`pyuvdata/telescopes.py:45,82,93,100,111`).
- **Nominal circular pair.** `feed_array = ("r", "l")` with
  `feed_angle = (0.0, 0.0)`, matching pyuvdata's default map for `r`/`l` feeds
  (`pyuvdata/uvbeam/uvbeam.py:673`).

### 12.2 The configured quantity

The user configures **one rotation offset per antenna**, not two absolute
angles:

```
feed_rotation_deg : float, default 0.0
```

The resolved absolute pyuvdata feed angles are then

```
linear:   feed_angle = (pi/2 + chi, 0 + chi)
circular: feed_angle = (chi, chi)
```

where `chi = radians(feed_rotation_deg)`. A single offset is used because
Section 11.2 rejects non-orthogonal pairs; two independent angles would let a
user express a pair that Tier 5 cannot model. `feed_rotation_deg` is normalized
into `(-180, 180]` at resolution and stored in radians.

The name keeps the word "feed" deliberately: on the *receptor* side "feed" is
the correct pyuvdata/AIPS vocabulary. It is the *illumination* side that must
stop using it (Section 15).

### 12.3 Parallactic-angle boundary

Because `ParallacticAngleJones` is a Tier 7 identity stub and Tier 5 accepts
only `mount_type == "fixed"`, `feed_rotation_deg` is a **static** rotation in
the topocentric frame for the whole observation. The `C` term is therefore
time-independent and `is_time_dependent()` returns `False`.

Tier 5 must reject, with an explicit error, any configuration that enables the
`P` term while a non-zero `feed_rotation_deg` is present, because the composed
result would silently omit the time-dependent part of the rotation. When Tier 7
implements `P`, the composition `P_p(t) · C_p` becomes the full time-dependent
receptor orientation and this rejection is removed.

## 13. Design decision 4 — common output basis for heterogeneous arrays

### 13.1 Decision

A `SimulationResult` carries **exactly one** output basis and **exactly four**
correlation labels. The output basis is resolved once, before any solver work,
by this rule:

| `receptors.output_basis` | Native bases present | Resolved output basis |
|---|---|---|
| `auto` (default) | all `linear` | `linear_xy` |
| `auto` | all `circular` | `circular_rl` |
| `auto` | mixed | **rejected** — `AmbiguousOutputBasisError` |
| `linear` | any combination | `linear_xy` |
| `circular` | any combination | `circular_rl` |

Under `auto`, a mixed array is rejected rather than silently defaulted, because
either choice would be a scientific decision made on the user's behalf. Naming
`output_basis` explicitly makes the array legal.

### 13.2 Why the transform is physically meaningful

For a dual-feed antenna the correlator receives both feed voltages coherently.
The pair `(v_r, v_l)` is a fixed unitary combination of `(v_x, v_y)`, so
expressing an antenna's response in the other basis is a change of
representation of two simultaneously sampled voltages, not a fabrication. This
is the same operation performed by hybrid-array correlators and by CASA's
basis handling. It is exact under the assumptions of Section 11.3.

For a *single*-feed antenna the operation is impossible, which is why
Section 11.2 rejects `Nfeeds = 1`.

### 13.3 What remains rejected until implemented

- mixed arrays under `output_basis: auto` (above);
- any antenna whose native basis is neither `linear` nor `circular`;
- any request for a mixed-hand correlation axis such as
  `("XR", "XL", "YR", "YL")` — Tier 5 has no such labels, no AIPS codes for
  them, and no writer support;
- any request for a Stokes-labelled output axis (`I`, `Q`, `U`, `V` as
  correlations) — that is an imaging product, not a correlator product.

## 14. Design decision 5 — correlation labels and file-format codes

### 14.1 Basis-independent index mapping

The `(2, 2) → 4` row-major reshape at `src/radiosim/core/result.py:920` defines
the mapping and is basis independent:

| Flat index | Matrix element | `linear_xy` | `circular_rl` |
|---|---|---|---|
| 0 | `[0,0]` | `XX` | `RR` |
| 1 | `[0,1]` | `XY` | `RL` |
| 2 | `[1,0]` | `YX` | `LR` |
| 3 | `[1,1]` | `YY` | `LL` |

Indices `0` and `3` are the parallel hands in both bases; indices `1` and `2`
are the cross hands in both bases. `stokes_i()` remains `v[...,0] + v[...,3]` in
both bases — but Tier 5 requires it to *derive* those indices from
`self.correlations` and assert the result, rather than assume them (defect D6).

### 14.2 The two accepted correlation coordinate sets

| Basis token | Labels | In-memory AIPS codes | Round-trip AIPS code order | pyuvdata `feed_array` | pyuvdata `polarization_array` |
|---|---|---|---|---|---|
| `linear_xy` | `("XX","XY","YX","YY")` | `(-5, -7, -8, -6)` | `(-5, -6, -7, -8)` = `XX,YY,XY,YX` | `["x","y"]` | `["xx","xy","yx","yy"]` |
| `circular_rl` | `("RR","RL","LR","LL")` | `(-1, -3, -4, -2)` | `(-1, -2, -3, -4)` = `RR,LL,RL,LR` | `["r","l"]` | `["rr","rl","lr","ll"]` |

The linear row reproduces the existing constants exactly
(`src/radiosim/io/hdf5.py:59-60`,
`src/radiosim/io/standard_visibility.py:29-31`), so no linear behavior changes.
The circular row follows the same two rules: in-memory order is the row-major
matrix order; the descending code order is what a `UVData` in either basis
reports in `polarization_array` after a UVFITS write/read round trip, and what
pyuvdata's *reader* produces from a Measurement Set on read-back.

**Correction (Tier 5A, Q3).** This column is labelled "On-disk AIPS code
order" in the design-gate text; Tier 5A's independent pyuvdata 3.2.1 probe
(`tests/characterization/test_pyuvdata_321_polarization_contract.py`) found
that label is only literally true for UVFITS, whose polarization axis is a
monotonic `CRVAL`/`CDELT` sequence and is genuinely written in this descending
order. For a Measurement Set, the `POLARIZATION` table's `CORR_TYPE` column
preserves the **in-memory** order passed to `UVData.new` (verified: circular
`CORR_TYPE = [5, 6, 7, 8]` = `RR,RL,LR,LL`, not the descending row above); the
descending order only appears after `UVData.read_ms()` canonicalizes the axis
on the way back into memory. Tier 5F must not expect `(-1,-2,-3,-4)` when
inspecting `CORR_TYPE` directly, only when inspecting `UVData.polarization_array`
post-read. The `pyuvdata feeds` column is also corrected below (§14.4); no
correlation-coordinate contract, decision, or slice boundary changes as a
result — see Section 43 Q3.

This table becomes **one** module-level constant, exported once and imported by
`core/result.py`, `io/hdf5.py`, and `io/standard_visibility.py`, replacing the
four independent copies (defect D4).

### 14.3 Measurement Set `CORR_TYPE`

RadioSim does not write `CORR_TYPE` directly; pyuvdata derives it from
`polarization_array` using the casacore `Stokes` enumeration (R8):
`RR=5, RL=6, LR=7, LL=8, XX=9, XY=10, YX=11, YY=12`. Tier 5 keeps that
delegation and validates it on read-back. This is the in-memory order, not the
Section 14.2 descending order (Tier 5A, Q3): a circular write keeps
`CORR_TYPE = [5, 6, 7, 8]`, and only `UVData.read_ms()` reorders it to
`(-1,-2,-3,-4)` in `polarization_array`.
`src/radiosim/io/measurement_set.py:595-604` currently requires
`NUM_CORR == 4`, which remains correct for both bases; the downstream label
check in `src/radiosim/io/standard_visibility.py` becomes basis-aware.

### 14.4 What `feed_array` records in written files

pyuvdata's `Telescope.feed_array` has shape `(Nants, Nfeeds)` and accepts
`["x","y","r","l"]` (`pyuvdata/telescopes.py:404-417`), so a heterogeneous
array is representable in the dependency. Tier 5 nonetheless writes the
**output** basis uniformly for every antenna, with the nominal `feed_angle` for
that basis, because `feed_array` and `polarization_array` must describe the same
basis for any downstream reader to interpret the data correctly. The per-antenna
*native* basis, feed rotation, and applied transform are recorded in RadioSim
provenance — the HDF5 receptor group, the summary JSON receptor block, and the
`RADIOSIM_PROJECTION_JSON=` history record — never inferred by a reader from
`feed_array`.

**Correction (Tier 5A, Q3).** Tier 5F must construct `Telescope.new(...)` with
an explicit `feed_array` of shape `(Nants, Nfeeds)` (and `feed_angle` of the
same shape), not the convenience `feeds=[...]` parameter this section
previously implied for both bases. The installed pyuvdata 3.2.1
(`pyuvdata/telescopes.py:884-950`) routes `feeds` only through
`set_feeds_from_x_orientation`, which `Telescope.new` invokes solely when
`x_orientation` is also supplied; without it, `feeds` is silently ignored and
`feed_array` stays `None`. The existing linear writer
(`src/radiosim/io/standard_visibility.py:887`) works today only because it
also passes `x_orientation="east"`; Tier 5F's circular path has no
`x_orientation` to supply (§22.1) and so must pass `feed_array` directly. This
corrects the construction form only; the recorded output-basis and
`feed_array`/`polarization_array` contract above is otherwise unchanged.

See Section 43, open question Q3, on validating `feed_angle` for `r`/`l` feeds
against pyuvdata 3.2.1 `check()`.

## 15. Design decision 6 — illumination versus receptor schema separation

### 15.1 Decision

Two disjoint vocabularies, enforced by name:

| Concept | Owner | Configuration path | Identifier vocabulary |
|---|---|---|---|
| Aperture **illumination** | beam subsystem (`core/beam/`) | `beams.model.illumination.*` | `illumination`, `taper`, `edge_angle` |
| Receiving **receptor** | new receptor subsystem (`core/receptor.py`) | `receptors.*` | `receptor`, `feed`, `basis`, `feed_rotation` |

### 15.2 Work implied at the configuration boundary

Already complete at the baseline (Tier 3): `beams.feed_model`,
`beams.feed_computation`, and `beams.feed_params` are rejected with typed
guidance pointing at `beams.model.illumination`
(`src/radiosim/io/config.py:1629-1641`). Tier 5 adds no new rejection there.

Tier 5 repairs the one stale message: `_REMOVED_FIELD_GUIDANCE["feeds"]`
(`src/radiosim/io/config.py:2023-2026`) currently points at the also-removed
`beams.feed_model`. The `feeds` key is repurposed as an alias-rejection that
points at the new `receptors` section (Section 27).

### 15.3 Work implied at the identifier boundary

Rename, in `src/radiosim/core/beam/analytic.py`:

| Current | Replacement | Lines |
|---|---|---|
| `_feed_response` | `_illumination_response` | 302, 369, 546 |
| `_feed_angles` | `_illumination_edge_angles` | 330, 361, 548 |
| `theta_feed` parameter | `theta_illumination` | 195, 204, 224, 304, 310, 316, 323 |

Rename `src/radiosim/core/jones/beam/analytic/feed.py` to
`illumination.py` with its functions renamed
`corrugated_horn_pattern` → `corrugated_horn_illumination` and similarly for
the other two, updating `src/radiosim/core/jones/beam/analytic/__init__.py`.
These are private or analysis-only symbols; no public API changes.

Do **not** rename `feed_array`, `feed_angle`, `x_orientation`, or
`UnsupportedBeamFeedError` in `src/radiosim/core/beam/fits.py`,
`src/radiosim/core/beam/models.py`, or `src/radiosim/core/beam/errors.py`.
Those describe the receiving receptor and are correctly named.

## 16. Exact receptor configuration schema

A new module `src/radiosim/io/receptor_config.py` defines strict frozen
Pydantic v2 models, mirroring `src/radiosim/io/instrument_config.py`. It reuses
`AntennaReference` (`src/radiosim/io/instrument_config.py:178-181`) rather than
redefining antenna tagging.

```python
ReceptorBasis = Literal["linear", "circular"]
OutputBasisRequest = Literal["auto", "linear", "circular"]

class ReceptorDefinitionConfig(StrictFrozenModel):
    basis: ReceptorBasis = "linear"
    feed_rotation_deg: _StrictFiniteFloat = 0.0

class ReceptorOverrideConfig(StrictFrozenModel):
    antenna: AntennaReference
    basis: ReceptorBasis | None = None
    feed_rotation_deg: _StrictFiniteFloat | None = None

class ReceptorsConfig(StrictFrozenModel):
    default: ReceptorDefinitionConfig = Field(
        default_factory=ReceptorDefinitionConfig
    )
    overrides: tuple[ReceptorOverrideConfig, ...] = ()
    output_basis: OutputBasisRequest = "auto"
```

`RadioSimConfig` gains a tenth section:

```python
receptors: ReceptorsConfig = Field(default_factory=ReceptorsConfig)
```

Model-level validation in `ReceptorsConfig`:

1. every `overrides[i]` must set at least one of `basis` / `feed_rotation_deg`;
2. no two overrides may reference the same antenna (compared after the
   `AntennaReference` normalization already used by
   `src/radiosim/io/instrument_config.py:151-175`);
3. an override that mixes reference kinds (`number` and `name`) resolving to the
   same antenna is a duplicate and is rejected at resolution, not at schema
   validation, because antenna identity is not known until Tier 2 resolution.

The default `ReceptorsConfig()` reproduces exactly the baseline behavior: every
antenna linear, zero rotation, output basis `linear_xy`. Every existing config
therefore keeps bit-identical results for `V = 0` skies.

### 16.1 Accepted YAML for every mode

Omitted entirely — homogeneous linear, the default:

```yaml
# no receptors: section; equivalent to the block below
```

Explicit homogeneous linear:

```yaml
receptors:
  default:
    basis: linear
    feed_rotation_deg: 0.0
  output_basis: auto
```

Homogeneous circular:

```yaml
receptors:
  default:
    basis: circular
  output_basis: auto      # resolves to circular_rl
```

Homogeneous linear rotated 45 degrees:

```yaml
receptors:
  default:
    basis: linear
    feed_rotation_deg: 45.0
```

Heterogeneous rotations within one basis:

```yaml
receptors:
  default:
    basis: linear
  overrides:
    - antenna: {kind: number, number: 3}
      feed_rotation_deg: 30.0
    - antenna: {kind: name, name: HERA-11}
      feed_rotation_deg: -15.0
```

Heterogeneous bases, explicit common output basis:

```yaml
receptors:
  default:
    basis: linear
  overrides:
    - antenna: {kind: number, number: 7}
      basis: circular
  output_basis: circular    # every antenna transformed into R/L
```

Circular native, linear output:

```yaml
receptors:
  default:
    basis: circular
  output_basis: linear
```

## 17. Exact resolved receptor model and precedence

### 17.1 Placement decision

Receptor state lives in a **sibling** resolved object, not inside
`ResolvedInstrument`.

Chosen: a new `src/radiosim/core/receptor.py` defining `ResolvedReceptor`,
`ResolvedReceptorSet`, `ReceptorProvenance`, and `resolve_receptors()`, keyed by
`AntennaId`, exactly as Tier 3 keys beams by `AntennaId` inside `BeamSystem`
(`src/radiosim/core/beam/runtime.py:471-506`).

Rejected alternative: adding fields to `ResolvedAntenna`
(`src/radiosim/core/instrument.py:309-318`). That would change
`_canonical_instrument_fingerprint_payload`
(`src/radiosim/core/instrument.py:535-609`) and therefore every
`instrument_sha256` in every accepted Tier 2 test and record, for a quantity
that is not an instrument geometry property. The beam precedent — where
`ResolvedAntenna` holds only a `beam_id` *reference* and the beam itself lives
elsewhere — is the established architecture, and receptors need no reference at
all because `receptors.overrides` names antennas directly.

### 17.2 Models

```python
@dataclass(frozen=True, slots=True)
class ResolvedReceptor:
    basis: Literal["linear", "circular"]
    feed_rotation_rad: float          # normalized into (-pi, pi]
    feed_array: tuple[str, str]       # ("x","y") or ("r","l")
    feed_angle_rad: tuple[float, float]   # absolute pyuvdata angles
    source: AntennaFieldSource        # "default" or "override"

@dataclass(frozen=True, slots=True)
class ResolvedReceptorSet:
    output_basis: Literal["linear_xy", "circular_rl"]
    receptor_by_antenna: Mapping[AntennaId, ResolvedReceptor]  # MappingProxyType
    provenance: ReceptorProvenance

    def to_snapshot(self) -> dict[str, object]: ...
```

`ReceptorProvenance` records the requested `output_basis`, the resolution rule
that produced the resolved basis, the ordered override applications, and
`receptor_sha256` — a SHA-256 over a canonical JSON payload of
`(output_basis, [(antenna_number, antenna_name, basis, feed_rotation_rad,
feed_array, feed_angle_rad, source), ...])` in canonical antenna order.
`__post_init__` recomputes and checks it, exactly as
`ResolvedInstrument.__post_init__` does
(`src/radiosim/core/instrument.py:647-671`).

### 17.3 Precedence, centralized and explicit

`resolve_receptors(config: ReceptorsConfig, instrument: ResolvedInstrument)
-> ResolvedReceptorSet` is the **only** place receptor precedence is decided
(governing decision `Fix.md` §4.3). Order:

1. start from `config.default` for every antenna in canonical instrument order;
2. apply `config.overrides` in declared order; each override sets only the
   fields it declares; a later override for the same antenna is a duplicate and
   is rejected;
3. every override must resolve to exactly one antenna present in the resolved
   instrument, else `ReceptorAssignmentError`;
4. compute `feed_array` and `feed_angle_rad` from `basis` and
   `feed_rotation_rad` per Section 12.2;
5. resolve `output_basis` per the Section 13.1 table;
6. validate the Section 11.2 rejections;
7. compute `receptor_sha256`.

Resolution is pure, runs before any beam load, backend selection, filesystem
access, or solver work, and produces no side effect.

## 18. Exact receptor and basis-transform Jones mathematics

All matrices are `2×2` complex, in the sky-linear basis of Section 10, with the
row index the receptor feed and the column index the sky component
(`jones[feed, sky_basis]`, matching the existing convention statement at
`src/radiosim/core/polarization.py:149`).

### 18.1 Building blocks

Rotation of the receptor pair by `χ` within the sky-linear plane:

```
R(chi) = [[ cos chi,  sin chi],
          [-sin chi,  cos chi]]
```

Linear-to-circular basis matrix, rows ordered `(R, L)`, columns `(x, y)`:

```
S = (1/sqrt(2)) * [[1,  i],
                   [1, -i]]
```

`S` is unitary: `S S^H = S^H S = I₂`.

### 18.2 `ReceptorConfigJones` (term `C`)

```
C_p = M(basis_p) @ R(chi_p)

M(linear)   = I2
M(circular) = S
```

Properties: `name = "C"`, `is_direction_dependent = False`,
`is_time_dependent = False`, `is_frequency_dependent = False`,
`is_unitary() = True` (now truthfully, since `M` and `R` are both unitary).
`is_diagonal()` is `True` only when `basis == "linear"` and `chi == 0`.

The term is constructed from `ResolvedReceptorSet` plus a
`SolverInstrumentView`, and `compute_jones(antenna_idx, ...)` resolves
`AntennaId` from the view exactly as `_ResolvedBeamJones._antenna_id` does
(`src/radiosim/core/visibility.py:119-129`).

### 18.3 `BasisTransformJones` (term `H`)

```
H_p = T(basis_p -> output_basis)

T(linear   -> linear_xy)   = I2
T(circular -> circular_rl) = I2
T(linear   -> circular_rl) = S
T(circular -> linear_xy)   = S^H
```

Properties: `name = "H"`, direction/time/frequency independent, unitary.
Note that `H_p @ C_p` collapses: for a circular-native antenna with circular
output, `H C = S R(χ)`; for a linear-native antenna with circular output,
`H C = S R(χ)`. The two terms are kept separate anyway because they answer two
different questions — what the receptor physically is (`C`) and what basis the
result is reported in (`H`) — and because Tier 7's `D` term must be inserted
*between* them.

### 18.4 Derived correlation relations (the test oracle)

With `B` from Section 9.1 and `E = e·I₂`, for two antennas with the same basis
and zero rotation:

**Linear output** (`C = H = I₂`):

```
V_xx = (I + Q)/2      V_xy = (U + iV)/2
V_yx = (U - iV)/2     V_yy = (I - Q)/2
```

**Circular output** (`C = S`, `H = I₂`), obtained as `S B S^H`:

```
V_RR = (I + V)/2      V_RL = (Q + iU)/2
V_LR = (Q - iU)/2     V_LL = (I - V)/2
```

These reproduce R2/R4 exactly. They are the reference table for the Tier 5 tests
and the reason the Section 10.2 sign correction is required: with the baseline
`B`, the same `S` yields `V_RR = (I − V)/2`, i.e. a `V = +I` source would appear
as pure `LL`.

### 18.5 Rotation invariants (the second test oracle)

**Linear.** `V' = R(χ) B R(χ)^T` rotates the linear Stokes parameters by `2χ`:

```
Q' = Q cos 2chi + U sin 2chi
U' = -Q sin 2chi + U cos 2chi
I' = I    V' = V
```

so `V_xx + V_yy` and `V_RR + V_LL` are invariant.

**Circular.** `S R(chi) = diag(e^{-i chi}, e^{+i chi}) @ S`, therefore

```
V_RR, V_LL  invariant under chi
V_RL -> e^{-2 i chi} V_RL
V_LR -> e^{+2 i chi} V_LR
```

Both identities are exact, analytic, and independent of the implementation.

### 18.6 Energy conservation

For unpolarized `Q = U = V = 0`, `B = (I/2) I₂`, so for any unitary receptor
`J`, `V = J (I/2) I₂ J^H = (I/2) I₂`. Hence in **every** supported basis and for
**every** feed rotation:

```
V[0,0] + V[1,1] = I     V[0,1] = V[1,0] = 0
```

This is the invariant `Fix.md` §14 calls "unpolarized energy conservation", and
it holds identically for `linear_xy` and `circular_rl`.

### 18.7 Round trip

`T(linear→circular) @ T(circular→linear) = S S^H = I₂` to machine precision, and
`(H C)^H (H C) = I₂` for every accepted `(basis, χ, output_basis)` combination.

## 19. Jones chain order and solver integration contract

### 19.1 Canonical order

Tier 5 makes the chain order explicit and correct (defect D7). The canonical
factorization, leftmost nearest the correlator (R1, R3):

```
J_p = H_p  G_p  B_p  D_p  P_p  C_p  E_p  T_p  Z_p    (K applied separately)
```

`H` is leftmost because it is a reporting-basis change performed at the
correlator. `C` sits between the sky-side DDEs (`E`, `T`, `Z`) and the
electronics-side DIEs (`D`, `G`, `B`), because leakage and gains are defined in
the receptor's own basis.

Because `JonesChain` composes `terms[0] @ terms[1] @ ... @ terms[-1]`
(`src/radiosim/core/jones/chain.py:166,184`), the add order becomes:

```python
chain.add_term(H)   # first added, leftmost, applied last
chain.add_term(G)
chain.add_term(B)
chain.add_term(D)
chain.add_term(P)
chain.add_term(C)
chain.add_term(E)   # always
chain.add_term(T)
chain.add_term(Z)
```

This reverses the current add order in
`src/radiosim/core/visibility.py:647-719`. The change is currently
unobservable — all optional terms are disabled by default and identity when
enabled, and `E = e·I₂` commutes — but it must be fixed now, because `C` and `H`
are the first non-commuting factors RadioSim will ever compose. The plan
requires a dedicated test that composes two deliberately non-commuting synthetic
terms and asserts the documented product order.

`C` and `H` are always added, exactly as `E` always is. When the resolved
receptor set is homogeneous linear with zero rotation, both are `I₂` and the
result is bit-identical to the baseline; the terms are still present so that the
chain has one shape.

### 19.2 Point-source path

`_build_jones_chain` gains a `receptors: ResolvedReceptorSet` parameter and
constructs `C` and `H` from it. No other change to
`src/radiosim/core/visibility.py` is required: the per-antenna Jones cache
(lines 524-539) and the RIME application (lines 572-587) are already
basis-agnostic, and the unpolarized fast path at line 576 remains valid because
`J_p J_q^H` is exactly the right factor for `B = (I/2) I₂` under any unitary
receptor (Section 18.6).

### 19.3 HEALPix path

`src/radiosim/core/visibility_healpix.py` does **not** use `JonesChain`; it
calls `beam_system.evaluate_jones` directly (lines 91-135) and applies the RIME
at lines 409-414 (polarized) and 472-475 (scalar). Tier 5 threads receptors into
this path by left-multiplying the cached per-antenna beam Jones by the constant
`H_p @ C_p` for that antenna, once per antenna per frequency:

```
J_p_effective = (H_p @ C_p) @ J_p_beam
```

This is exact because `C` and `H` are direction-, time-, and
frequency-independent, and because the cache is already keyed by antenna handler
(lines 103-134). The scalar path at lines 457-463 builds `C = (I/2) I₂`
explicitly and therefore also remains correct under Section 18.6, but it must
still apply `H_p @ C_p` so that cross-hand outputs are zero *in the reported
basis* rather than by assumption.

Both paths must produce identical correlations for the same inputs; the plan
requires a point-vs-HEALPix cross-check for at least one circular case.

## 20. Exact correlation coordinate contract in the result model

### 20.1 One shared constant

A new module-level table, defined once and imported everywhere:

```python
# src/radiosim/core/polarization_basis.py
PolarizationBasis = Literal["linear_xy", "circular_rl"]

CORRELATION_LABELS: Final[Mapping[PolarizationBasis, tuple[str, str, str, str]]]
AIPS_CODES_CANONICAL: Final[Mapping[PolarizationBasis, tuple[int, int, int, int]]]
AIPS_CODES_FILE_ORDER: Final[Mapping[PolarizationBasis, tuple[int, int, int, int]]]
PYUVDATA_FEEDS: Final[Mapping[PolarizationBasis, tuple[str, str]]]
PYUVDATA_POLARIZATIONS: Final[Mapping[PolarizationBasis, tuple[str, ...]]]

def basis_for_correlations(correlations: tuple[str, ...]) -> PolarizationBasis: ...
def parallel_hand_indices(correlations: tuple[str, ...]) -> tuple[int, int]: ...
```

populated from the Section 14.2 table. `src/radiosim/core/result.py:32`,
`src/radiosim/io/hdf5.py:58-60`, and
`src/radiosim/io/standard_visibility.py:29-31` all import from it. The four
independent copies are removed (defect D4).

### 20.2 Result model changes

| Site | Baseline | Tier 5 |
|---|---|---|
| `core/result.py:951`, `:987-988` | `correlations=_CORRELATIONS`, `polarization_basis="linear_xy"` | both derived from `receptors.output_basis` |
| `core/result.py:1117`, `:1145-1146` | same literals | same derivation |
| `core/result.py:408` | `_hash_json(digest, "polarization_basis", "linear_xy")` | hashes the actual basis, plus a new `receptor` snapshot entry |
| `core/result.py:512-519` | `v[...,0] + v[...,3]` | `i0, i3 = parallel_hand_indices(self.correlations)` then sum, with an explicit check |
| `core/result.py:1043-1044` | `!= _CORRELATIONS` → `InvalidResultError` | must be one of the two accepted tuples; error names both |
| `core/result.py:827`, `:920` | unchanged | unchanged; the `4` and the row-major reshape are basis independent |

`SimulationResult` and `LoadedSimulationResult` gain one field,
`receptors: ResolvedReceptorSet` (or, on the loaded side, its snapshot), placed
alongside `beam_state`, and `to_summary_snapshot()` gains a `receptor` block.
`polarization_basis` is narrowed from `str` to the `PolarizationBasis` literal
and validated on construction and on load.

### 20.3 Visualization

`src/radiosim/visualization/bokeh_plots.py:93-106` already delegates to
`result.stokes_i()` and never indexes correlations, so once `stokes_i()` derives
its indices (Section 20.2) the plot layer is correct in both bases with no
structural change. Axis and legend text that names "XX + YY" must be replaced by
text derived from `result.correlations`, so a circular run does not display
linear labels. The plot layer must not gain a second basis table.

## 21. HDF5 schema change

`src/radiosim/io/hdf5.py` currently writes labels and codes from module
constants (`:671-679`) and rejects any file whose stored values differ
(`:1833-1842`). Tier 5:

1. writes `coordinates/correlation/labels` and
   `coordinates/correlation/aips_codes` from `result.correlations` and the
   Section 14.2 table for `result.polarization_basis`;
2. adds a required fixed-width dataset `coordinates/correlation/basis` holding
   `linear_xy` or `circular_rl`;
3. adds a required group `receptors/` holding the `ResolvedReceptorSet`
   snapshot: `output_basis`, `receptor_sha256`, and the per-antenna
   `(antenna_number, antenna_name, basis, feed_rotation_rad, feed_angle_rad)`
   arrays in canonical antenna order;
4. validates on read that the `(labels, aips_codes, basis)` triple is exactly
   one row of the Section 14.2 table, rejecting mismatches with
   `UnsafeResultInputError` as today;
5. bumps `SCHEMA_VERSION` from `"1.0.0"` to `"2.0.0"` and rejects `"1.0.0"`
   files with `UnsupportedSchemaVersionError` naming Tier 5 as the boundary.

The bump is a version replacement, not a compatibility shim, per the pre-v1
policy. All Tier 4 safety properties are preserved unchanged: fixed byte widths,
NUL checks, bounded string limits, no dynamic evaluation, temporary-write /
read-back / atomic-publish ordering.

## 22. Standard visibility format mapping

### 22.1 Write path

`src/radiosim/io/standard_visibility.py`:

| Line | Baseline | Tier 5 |
|---|---|---|
| `:29-31` | three linear-only constants | import the Section 20.1 table |
| `:478-482`, `:686-688` | reject anything but the linear tuple | accept either accepted tuple; reject reorderings and mixed labels |
| `:727` | `for correlation_index in (0, 3):` | `for correlation_index in parallel_hand_indices(correlations):` |
| `:887` | `feeds=["x","y"]` | `feed_array=tile(PYUVDATA_FEEDS[basis])` (corrected construction form, Tier 5A Q3 — `feeds=` alone is silently ignored by pyuvdata 3.2.1 without `x_orientation`) |
| `:887` `x_orientation="east"` | fixed | retained for `linear_xy`; for `circular_rl` the plan uses explicit `feed_array`/`feed_angle` and omits the deprecated `x_orientation` (see Q3) |
| `:898` | `polarization_array=["xx","xy","yx","yy"]` | `list(PYUVDATA_POLARIZATIONS[basis])` |
| `:925-932` | assert `CANONICAL_CODES` | assert `AIPS_CODES_CANONICAL[basis]` |
| `:1085`, `:1114`, `:1465` | linear-only code gates | basis-aware gates against both accepted rows |

The autocorrelation normalization at `:727` is load-bearing: in a circular basis
the parallel hands `RR` and `LL` are still the real-valued autocorrelation
products, so deriving the indices rather than assuming `(0, 3)` is correct as
well as cleaner.

### 22.2 Read path

`src/radiosim/io/measurement_set.py:595-604` keeps its `NUM_CORR == 4` check.
The label reconstruction in `src/radiosim/io/standard_visibility.py:1040`,
`:1465` becomes basis-aware: read the on-disk codes, match them against both
`AIPS_CODES_FILE_ORDER` rows, reject anything else with
`FormatRepresentationError`, then reorder to the canonical in-memory order for
that basis.

`src/radiosim/io/uvfits.py:125-126`, `:305-307` become basis-aware with the same
rule.

### 22.3 Projection history

The `RADIOSIM_PROJECTION_JSON=` record
(`src/radiosim/io/standard_visibility.py:28`, prefix at `:28`) gains
`polarization_basis` and `receptor_sha256`. Its bounded-length and
depth limits (`_PROJECTION_HISTORY_LIMIT`, `_MAX_PROJECTION_JSON_DEPTH` at
`:33-34`) are unchanged; the added keys are two short scalars.

## 23. Summary JSON, provenance, and fingerprint policy

`src/radiosim/io/summary_json.py:278-281` already emits
`{"labels": list(result.correlations), "basis": result.polarization_basis}` and
therefore requires **no** structural change — it becomes truthful automatically
once the result model is data-driven. Tier 5 adds one bounded block:

```json
"receptors": {
  "output_basis": "circular_rl",
  "receptor_sha256": "<64 hex>",
  "native_basis_counts": {"linear": 4, "circular": 1},
  "distinct_feed_rotations_deg": [0.0, 30.0]
}
```

Per-antenna receptor rows are **not** embedded in the summary; the summary is
bounded metadata by Tier 4 contract, and the complete per-antenna set lives in
the HDF5 `receptors/` group. The summary's existing "explicitly incomplete"
exclusion list gains "per-antenna receptor definitions".

**Fingerprint policy.** `receptor_sha256` and `polarization_basis` enter the
*scientific* hash (`src/radiosim/core/result.py:387-412`), because changing
either changes the visibilities or their meaning. They do **not** enter the
provenance-only hash separately. Two consequences that must be stated in the
breaking-change ledger:

- every existing `scientific_sha256` changes, because `polarization_basis` is
  now hashed as a value and a `receptor` snapshot entry is added, even for a
  default configuration;
- `instrument_sha256` does **not** change, because receptors are a sibling of
  the instrument (Section 17.1).

## 24. Public API additions, removals, exports, and signatures

Additions, lazily exported following the existing pattern in
`src/radiosim/core/jones/__init__.py:183-184`:

```python
# radiosim.core
PolarizationBasis
CORRELATION_LABELS
ResolvedReceptor
ResolvedReceptorSet
ReceptorProvenance
resolve_receptors
ReceptorError, InvalidReceptorConfigError, UnsupportedReceptorBasisError,
UnsupportedFeedGeometryError, AmbiguousOutputBasisError,
UnsupportedBasisTransformError, ReceptorAssignmentError

# radiosim.io
ReceptorsConfig, ReceptorDefinitionConfig, ReceptorOverrideConfig
```

Changed signatures:

```python
ReceptorConfigJones(*, receptors: ResolvedReceptorSet,
                    instrument: SolverInstrumentView)
BasisTransformJones(*, receptors: ResolvedReceptorSet,
                    instrument: SolverInstrumentView)
```

The permissive stub constructors `ReceptorConfigJones(feed_type="linear")` and
`BasisTransformJones(from_basis=..., to_basis=...)`
(`src/radiosim/core/jones/receptor.py:31`, `:73-75`) are removed outright.
Passing `feed_type`, `from_basis`, or `to_basis` raises `TypeError` with a
message naming `receptors:` configuration as the replacement.

```python
_build_jones_chain(..., receptors: ResolvedReceptorSet, ...)
calculate_visibility(..., receptors: ResolvedReceptorSet, ...)
calculate_visibility_healpix(..., receptors: ResolvedReceptorSet, ...)
build_simulation_result(..., receptors: ResolvedReceptorSet, ...)
SimulationResult.receptors -> ResolvedReceptorSet
SimulationResult.polarization_basis -> PolarizationBasis  # narrowed
```

`visibility_to_correlations` (`src/radiosim/core/polarization.py:217`) is
changed to accept the basis and return basis-correct keys, or removed if 5A
shows it has no production caller. Slice 5H decides on evidence, not assumption.

## 25. Error taxonomy and side-effect ordering

### 25.1 Typed errors

`src/radiosim/core/receptor.py` defines:

- `ReceptorError(RuntimeError)`;
- `InvalidReceptorConfigError(ReceptorError)`;
- `UnsupportedReceptorBasisError(InvalidReceptorConfigError)`;
- `UnsupportedFeedGeometryError(InvalidReceptorConfigError)`;
- `AmbiguousOutputBasisError(InvalidReceptorConfigError)`;
- `ReceptorAssignmentError(InvalidReceptorConfigError)`;
- `UnsupportedBasisTransformError(ReceptorError)`.

`src/radiosim/io/result_errors.py` gains:

- `UnsupportedPolarizationBasisError(FormatRepresentationError)`.

No existing error class is renamed or removed.

### 25.2 Mandatory order

Receptor work inserts into the Tier 4 order (`Tier4ResultOutputPlan.md` §23.2)
at exactly one place, between configuration validation and instrument/beam
resolution:

1. schema, semantic, time-count, width, and phase validation precede setup;
2. **receptor schema validation and `resolve_receptors()` run next, after
   instrument resolution and before any beam load, backend selection, device
   transfer, filesystem access, or network access**;
3. instrument and beam resolution retain their Tier 3 order;
4. result shape, dtype, finite, coordinate, correlation-label, basis, and
   identity validation precede publication;
5. every writer step retains its Tier 4 ordering, with the added basis check
   performed in the pure-validation phase before any path creation.

`resolve_receptors()` therefore fails before the first beam file is opened, and
an unsupported basis or ambiguous output basis leaves no new output path, no
loaded beam, and no backend allocation.

## 26. Scientific invariants

Every one of these must be asserted by a test, and each is checkable against
Section 18 without reading the implementation.

| # | Invariant |
|---|---|
| S1 | Default configuration (`linear`, `χ=0`, `auto`) yields `C = H = I₂` and visibilities bit-identical to the baseline for any `V = 0` sky |
| S2 | `stokes_to_coherency` reproduces `B = ½[[I+Q, U+iV],[U−iV, I−Q]]` exactly |
| S3 | `coherency_to_stokes(stokes_to_coherency(s)) == s` to machine precision for arbitrary `I,Q,U,V` |
| S4 | Circular output reproduces `RR=(I+V)/2, RL=(Q+iU)/2, LR=(Q−iU)/2, LL=(I−V)/2` |
| S5 | Unpolarized source: `V[0,0] + V[1,1] = I`, `V[0,1] = V[1,0] = 0`, in both bases, for every rotation |
| S6 | `S S^H = S^H S = I₂`; `(H C)^H (H C) = I₂` for every accepted combination |
| S7 | Linear rotation by `χ` rotates `(Q,U)` by `2χ` and leaves `I`, `V` unchanged |
| S8 | Circular rotation by `χ` leaves `RR`, `LL` unchanged and multiplies `RL` by `e^{−2iχ}` |
| S9 | Round trip `T(lin→circ) ∘ T(circ→lin) = I₂` |
| S10 | Circular-native + `output_basis: linear` equals linear-native + `output_basis: linear` for the same sky, to machine precision (exactness of the change of representation under Section 11.3) |
| S11 | `stokes_i()` equals `I` summed over sources in both bases |
| S12 | Point and HEALPix paths agree on a common circular case within the established solver tolerance |
| S13 | The Jones chain composes in the Section 19.1 order, proven with two non-commuting synthetic terms |
| S14 | `receptor_sha256` is stable under reordering of `overrides` that produces the same resolved set, and changes whenever any resolved receptor changes |

## 27. Configuration additions, removals, and exact rejection messages

Added: the `receptors:` section of Section 16.

Removed/repurposed: `_REMOVED_FIELD_GUIDANCE["feeds"]`
(`src/radiosim/io/config.py:2023-2026`) is rewritten from its current stale text
to a pointer at the new section.

Exact messages (message, then hint, matching the existing `ConfigIssue` shape at
`src/radiosim/io/config.py:2259-2263`):

```text
feeds: top-level 'feeds' was replaced by the Tier 5 receptor model
  Use the 'receptors' section with 'default.basis', 'default.feed_rotation_deg', and 'output_basis'.

receptors.default.feed_type: removed before v1.0; use 'basis'
  Set receptors.default.basis to 'linear' or 'circular'.

receptors.default.basis: input should be 'linear' or 'circular'
  Tier 5 supports exactly two receptor bases; elliptical and mixed-feed receptors are Tier 7.

receptors.default.n_feeds: removed before v1.0; every antenna has exactly two feeds
  Single-feed and multi-feed antennas are rejected until Tier 7 implements them.

receptors.default.feed_angle_deg: removed before v1.0; use 'feed_rotation_deg'
  feed_rotation_deg is an offset from the nominal orientation for the selected basis.

receptors.output_basis: input should be 'auto', 'linear' or 'circular'
  Use 'auto' for a homogeneous array; name a basis explicitly for a mixed array.
```

Runtime rejections raised by `resolve_receptors()`:

```text
AmbiguousOutputBasisError: receptors.output_basis='auto' cannot resolve a mixed array
  (linear antennas: 4, circular antennas: 1); set receptors.output_basis to 'linear' or 'circular'.

ReceptorAssignmentError: receptors.overrides[2] references antenna number 91, which is
  absent from the resolved instrument.

ReceptorAssignmentError: receptors.overrides[3] duplicates antenna 'HERA-11', already set
  by receptors.overrides[1].

UnsupportedFeedGeometryError: mount_type='alt-az' is unsupported by Tier 5 receptors;
  time-dependent feed orientation requires the parallactic-angle term (Tier 7).

UnsupportedFeedGeometryError: a non-zero feed_rotation_deg cannot be combined with an
  enabled parallactic-angle term until Tier 7 implements it.
```

Every message names the field path, states the boundary, and names the
replacement or the deferring tier.

## 28. Backward compatibility and pre-v1 migration policy

There is none, by `Fix.md` §4.1. Specifically:

- the stub constructors `ReceptorConfigJones(feed_type=...)` and
  `BasisTransformJones(from_basis=..., to_basis=...)` are removed, not
  deprecated;
- the HDF5 schema goes to `2.0.0` and `1.0.0` files are rejected, not upgraded;
- `_CORRELATIONS`, `CORRELATIONS`, `AIPS_CODES`, `CANONICAL_CORRELATIONS`,
  `CANONICAL_CODES`, and `FILE_CODES` are removed as independent constants, not
  aliased;
- the Stokes `V` sign changes with no compatibility flag;
- `scientific_sha256` values change for every result;
- solver and result factory signatures gain a required `receptors` parameter
  with no default.

Every removal is documented in `docs/migration_guide.md` with the exact
replacement, and `docs/migration_guide.md:139` (which currently reads that
top-level `feeds` has "no replacement") is corrected.

## 29. Exact test matrix

`Fix.md` §14 lists eight required tests. Each maps to concrete cases:

| §14 requirement | Test file | Cases |
|---|---|---|
| identity for linear-to-linear | `tests/unit/test_jones/test_receptor.py` | `C` and `H` are exactly `I₂` for `linear`/`χ=0`/`linear_xy`; solver output bit-identical to a receptor-free reference |
| analytic linear/circular transforms | `tests/unit/test_jones/test_receptor.py` | `S` matches Section 18.1 elementwise; `C` for circular equals `S R(χ)`; S4 table reproduced |
| transform inverse/round trip | `tests/unit/test_jones/test_basis_transform.py` | S6, S9 |
| unpolarized energy conservation | `tests/unit/test_core/test_polarization.py`, `tests/unit/test_core/test_receptor_solver.py` | S5 across both bases × `χ ∈ {0, 30°, 45°, 90°, −15°}` |
| fully Q/U/V polarized reference cases | `tests/unit/test_core/test_polarization.py` | S2, S3, S4 with pure `Q`, pure `U`, pure `V`, and a mixed case, against the Section 18.4 table |
| correct linear and circular labels | `tests/unit/test_core/test_result.py`, `tests/unit/test_io/test_hdf5_result.py` | both label tuples, both code sets, both file orders, rejection of any reordering |
| heterogeneous arrays into one output basis | `tests/unit/test_core/test_receptor_resolution.py`, `tests/unit/test_core/test_receptor_solver.py` | mixed array under explicit `output_basis`; `auto` rejection; S10 |
| format polarization metadata round trips | `tests/unit/test_io/test_measurement_set.py`, `test_uvfits.py`, `test_standard_visibility.py` | circular MS and UVFITS round trips with raw pyuvdata inspection of `feed_array`, `feed_angle`, `polarization_array`, and `CORR_TYPE` |

Additional required tests not enumerated in §14 but implied by the design:

| Area | Test |
|---|---|
| Chain order | S13, two non-commuting synthetic terms |
| Provenance | S14; `instrument_sha256` unchanged; `scientific_sha256` changes with basis |
| Solver agreement | S12, point vs HEALPix circular |
| Config rejection | every message in Section 27, asserted verbatim |
| Ordering | `resolve_receptors()` failure leaves no loaded beam, no backend allocation, no output path |
| Terminology | no `feed`-named identifier survives in `core/beam/analytic.py`; no `illumination`-named identifier appears in `core/receptor.py` |

## 30. Exact implementation file inventory

### 30.1 New production files

```text
src/radiosim/core/polarization_basis.py
src/radiosim/core/receptor.py
src/radiosim/io/receptor_config.py
src/radiosim/core/jones/beam/analytic/illumination.py   (renamed from feed.py)
```

### 30.2 Modified production files

```text
src/radiosim/__init__.py
src/radiosim/api/simulator.py
src/radiosim/core/__init__.py
src/radiosim/core/beam/analytic.py
src/radiosim/core/instrument_adapters.py
src/radiosim/core/jones/__init__.py
src/radiosim/core/jones/beam/analytic/__init__.py
src/radiosim/core/jones/chain.py
src/radiosim/core/jones/receptor.py
src/radiosim/core/polarization.py
src/radiosim/core/result.py
src/radiosim/core/visibility.py
src/radiosim/core/visibility_healpix.py
src/radiosim/io/__init__.py
src/radiosim/io/config.py
src/radiosim/io/config_resolution.py
src/radiosim/io/hdf5.py
src/radiosim/io/measurement_set.py
src/radiosim/io/result_errors.py
src/radiosim/io/standard_visibility.py
src/radiosim/io/summary_json.py
src/radiosim/io/uvfits.py
src/radiosim/simulator/base.py
src/radiosim/simulator/rime.py
src/radiosim/visualization/bokeh_plots.py
```

### 30.3 Removed production file

```text
src/radiosim/core/jones/beam/analytic/feed.py   (renamed, not deleted outright)
```

### 30.4 New test files

```text
tests/characterization/test_tier5_current_behavior.py
tests/characterization/test_pyuvdata_321_polarization_contract.py
tests/unit/test_core/test_polarization_basis.py
tests/unit/test_core/test_receptor_resolution.py
tests/unit/test_core/test_receptor_solver.py
tests/unit/test_io/test_receptor_config.py
tests/unit/test_jones/test_receptor.py
tests/unit/test_jones/test_basis_transform.py
tests/unit/test_jones/test_chain_order.py
tests/unit/test_tier5_receptor_acceptance.py
```

### 30.5 Modified tests and fixtures

```text
tests/characterization/test_tier4_current_behavior.py
tests/fixtures/configs.py
tests/unit/test_core/test_beam_solver_integration.py
tests/unit/test_core/test_polarization.py
tests/unit/test_core/test_result.py
tests/unit/test_io/test_config.py
tests/unit/test_io/test_config_resolution.py
tests/unit/test_io/test_hdf5_result.py
tests/unit/test_io/test_measurement_set.py
tests/unit/test_io/test_standard_visibility.py
tests/unit/test_io/test_summary_json.py
tests/unit/test_io/test_uvfits.py
tests/unit/test_simulator/test_api.py
tests/unit/test_simulator/test_result_integration.py
tests/unit/test_visualization/
```

### 30.6 Configuration, examples, documentation, manifests

```text
configs/config.yaml
configs/receptor_circular_example.yaml        (new)
docs/migration_guide.md
docs/user_guide/configuration.rst
docs/user_guide/configuration_support.rst
docs/user_guide/jones_matrices.rst
docs/user_guide/beam_models.rst
docs/index.rst
CLAUDE.md
README.md
```

## 31. Dependency characterization results

Established by direct inspection of the installed pyuvdata in
`.pixi/envs/default` at this gate:

| Fact | Evidence |
|---|---|
| AIPS circular codes present and correct | `pyuvdata/utils/pol.py:37` — `"rr": -1, "ll": -2, "rl": -3, "lr": -4` |
| Reverse map present | `pyuvdata/utils/pol.py:43` |
| `feed_array` accepts `["x","y","r","l"]`, shape `(Nants, Nfeeds)` | `pyuvdata/telescopes.py:404-417` |
| `feed_angle` shape `(Nants, Nfeeds)`, radians, `0` toward North for fixed mounts | `pyuvdata/telescopes.py:418-431` |
| `Nfeeds` acceptable values are `[1, 2]` | `pyuvdata/telescopes.py:397-403` |
| `r`/`l` feeds default to `feed_angle = 0.0` | `pyuvdata/uvbeam/uvbeam.py:673` |
| Reference telescopes use `feed_angle = [π/2, 0]` for `x,y` | `pyuvdata/telescopes.py:45,82,93,100,111` |
| `x_orientation` is deprecated in favour of `feed_array`/`feed_angle` | `pyuvdata/telescopes.py:440-457` |
| `feed_array`, `feed_angle`, and `mount_type` must be set together | `pyuvdata/telescopes.py:567-579` |
| Tier 3 pins pyuvdata `3.2.1` in resolved beam metadata | `src/radiosim/core/beam/models.py:692-694` |
| `pyproject.toml` declares `pyuvdata>=2.4` | `pyproject.toml:49` |

Not established at this gate and required from slice 5A: whether pyuvdata
3.2.1 `UVData.check()` and the MS/UVFITS writers accept
`polarization_array=["rr","rl","lr","ll"]` together with
`Telescope.new(feeds=["r","l"], feed_angle=[[0.0, 0.0], ...])` and no
`x_orientation`; and the exact on-disk code ordering pyuvdata produces for
circular data. See Section 43, Q3.

## 32. Tests-first implementation strategy

Every slice writes its failing tests first and commits production changes only
after those tests fail for the intended reason. Characterization tests written
in 5A pin the *current* behavior, including the current `V` sign, so that the
5C correction is a visible, reviewed diff rather than an unnoticed drift. Tests
that must change when a later slice lands are marked with the owning slice, as
Tier 1 and Tier 4 did; no test is left silently xfailed past its owning slice.

## 33. Common verification gate

Run before every slice commit and before every acceptance:

```bash
pixi run test -- tests/unit/test_jones/ -k "receptor or basis or polarization or chain"
pixi run test -- tests/unit/test_core/ -k "polarization or receptor or result"
pixi run test -- tests/unit/test_io/
pixi run test -- tests/characterization/
pixi run test -- -m "not slow"
pixi run lint
pixi run format
```

`pixi run typecheck` is run only at the whole-tier acceptance, per `CLAUDE.md`.
The Pyright ceiling and the whitespace, YAML-validation, offline-example, and
clean-copy Sphinx checks established by Tiers 1-4 apply unchanged.

## 34. Tier 5 implementation slices

### 34.1 Tier 5A — characterization, convention evidence, and dependency contract

**Objective.** Pin current polarization behavior and establish the scientific
and dependency facts that Sections 10.2 and 14 depend on. Tests and evidence
only.

**Tests-first evidence.** New characterization tests assert the *current*
`C[0,1] = (U − iV)/2`, the current four hard-coded constant sites, the current
identity return of both receptor stubs, and the current chain composition order.
All pass at the baseline by construction.

**Production changes.** None.

**Scientific invariants.** Produce a written reproduction of the R2/R4
Stokes-to-correlation tables for linear and circular feeds and an explicit
statement of what `codex-africanus` implements, resolving Q1. Produce offline
pyuvdata 3.2.1 probes for circular `feeds`, `feed_angle`, `polarization_array`,
MS `CORR_TYPE`, and UVFITS on-disk code order, resolving Q3. Probe artifacts are
written outside the repository and removed.

**Breaking changes.** None.

**Exclusions.** No schema, no Jones math, no result change, no rename.

**Stop.** Commit only after both open questions are resolved with recorded
evidence. If the evidence contradicts Section 10.2 or Section 14.2, stop and
return the plan for amendment; do not proceed to 5B.

**Independent acceptance.** Re-run the probes independently; confirm each
characterization test fails if the corresponding constant is perturbed.

**Suggested commit.** `test(pol): characterize Tier 5 polarization baseline`

**Next slice.** Tier 5B.

### 34.2 Tier 5B — receptor configuration schema and resolution

**Objective.** Add the typed `receptors:` section, `ResolvedReceptorSet`, and
`resolve_receptors()`, with every rejection. No physics and no result change.

**Tests-first evidence.** Schema acceptance for every YAML in Section 16.1;
every rejection message in Section 27 asserted verbatim; precedence order;
`receptor_sha256` stability (S14); ordering (resolution fails before beam load).

**Production changes.** Add `src/radiosim/io/receptor_config.py` and
`src/radiosim/core/receptor.py`; add `receptors` to `RadioSimConfig`; wire
`resolve_receptors()` into `config_resolution` / `Simulator.setup()` at the
Section 25.2 position; repair the `feeds` guidance text; export the new symbols.
The resolved set is produced and validated but not yet consumed by the solver.

**Scientific invariants.** S14. No visibility changes; every existing result is
bit-identical.

**Breaking changes.** `feeds:` guidance text changes; `RadioSimConfig` gains a
tenth section.

**Exclusions.** No Jones math, no correlation-label change, no writer change, no
`V`-sign change.

**Stop.** Commit only after the full non-slow suite is unchanged apart from the
new tests. Tier 5C remains unauthorized pending independent 5B acceptance.

**Independent acceptance.** Author every rejected YAML by hand and confirm the
exact message; confirm `instrument_sha256` is unchanged; confirm a resolution
failure leaves no loaded beam and no output path.

**Suggested commit.** `refactor(config): separate illumination and receptor models`

**Next slice.** Tier 5C.

### 34.3 Tier 5C — polarization convention and receptor Jones mathematics

**Objective.** Correct the Stokes `V` sign and implement `ReceptorConfigJones`
and `BasisTransformJones` as real unitary physics. Not yet wired into the
solver.

**Tests-first evidence.** S2, S3, S4, S6, S7, S8, S9 fail at the start of the
slice: the `V`-sign tests fail against the current construction, and every
receptor-matrix test fails against the identity stubs.

**Production changes.** Add `src/radiosim/core/polarization_basis.py`; correct
`stokes_to_coherency` and `coherency_to_stokes` in
`src/radiosim/core/polarization.py`; replace both classes in
`src/radiosim/core/jones/receptor.py` with the Section 18 mathematics and the
Section 24 signatures; update the 5A characterization test that pinned the old
sign, with the change visible in the diff.

**Scientific invariants.** S2, S3, S4, S6, S7, S8, S9, and the analytic
statement that every accepted `H C` is unitary.

**Breaking changes.** Stokes `V` sign; removal of the permissive stub
constructors.

**Exclusions.** No chain wiring, no result-model change, no writer change, no
rename.

**Stop.** Commit only after every matrix is checked elementwise against
Section 18 and against the R2/R4 evidence recorded in 5A. Tier 5D remains
unauthorized pending independent 5C acceptance.

**Independent acceptance.** Recompute `S B S^H` symbolically or numerically from
the published table without reading the implementation; confirm the `V`-sign
change alters only cross-hands of `V ≠ 0` sources.

**Suggested commit.** `feat(jones): implement receptor basis transforms`

**Next slice.** Tier 5D.

### 34.4 Tier 5D — chain order and solver integration

**Objective.** Fix the chain composition order, thread `ResolvedReceptorSet`
into both solver paths, and make receptor configuration change the computed
visibilities.

**Tests-first evidence.** S13 fails against the current add order; S1, S5, S10,
S12 fail because the solver ignores receptors.

**Production changes.** Reverse the add order in
`src/radiosim/core/visibility.py:_build_jones_chain` to Section 19.1; always add
`C` and `H`; add the `receptors` parameter to `_build_jones_chain`,
`calculate_visibility`, `calculate_visibility_healpix`, `VisibilitySimulator`,
`RIMESimulator`, and `Simulator.run()`; left-multiply the HEALPix per-antenna
Jones cache by `H_p @ C_p`; document the order in the `JonesChain` docstring.

**Scientific invariants.** S1, S5, S10, S12, S13. Point and HEALPix agree.

**Breaking changes.** Solver signatures gain a required `receptors` parameter;
the chain composition order changes.

**Exclusions.** No result-model label change yet — the result is still stamped
`linear_xy` in this slice, which is why 5D's circular tests assert on the raw
`(2,2)` cube rather than on `result.correlations`. No writer change, no rename.

**Stop.** Commit only after a circular run is proven to differ from a linear run
on a `V`-polarized source and to be identical on an unpolarized one. Tier 5E
remains unauthorized pending independent 5D acceptance.

**Independent acceptance.** Construct the chain with two non-commuting synthetic
terms and confirm the product order; run point and HEALPix on the same circular
case and compare.

**Suggested commit.** `feat(solver): apply resolved receptors in the Jones chain`

**Next slice.** Tier 5E.

### 34.5 Tier 5E — data-driven correlation coordinates and HDF5 schema 2.0.0

**Objective.** Make the result model's correlation coordinates, basis, and
fingerprint data-driven, and version the HDF5 schema.

**Tests-first evidence.** Both label tuples through the result model; derived
`stokes_i()`; basis in the scientific hash; HDF5 `2.0.0` write/read with the
`receptors/` group; `1.0.0` rejection; hostile reordering rejection for both
bases.

**Production changes.** Import the Section 20.1 table into
`src/radiosim/core/result.py` and `src/radiosim/io/hdf5.py`, removing their
local constants; derive `correlations` and `polarization_basis` from
`receptors.output_basis`; hash the real basis and a receptor snapshot; derive
`stokes_i()` indices; accept both tuples on load; bump the HDF5 schema and add
the `basis` dataset and `receptors/` group.

**Scientific invariants.** S11, S14; `instrument_sha256` unchanged;
`scientific_sha256` changes for every result and differs between bases.

**Breaking changes.** HDF5 `1.0.0` rejected; `scientific_sha256` values change;
`polarization_basis` narrowed to a literal.

**Exclusions.** No MS/UVFITS change, no summary block, no plot text, no rename.

**Stop.** Commit only after a circular result round-trips through HDF5 with an
independently inspected file. Tier 5F remains unauthorized pending independent
5E acceptance.

**Independent acceptance.** Open the file with raw h5py; confirm labels, codes,
basis, and receptor group; confirm a hand-edited basis/label mismatch is
rejected.

**Suggested commit.** `feat(result): support linear and circular correlations`

**Next slice.** Tier 5F.

### 34.6 Tier 5F — standard formats, summary, and plots

**Objective.** Carry the resolved basis into Measurement Set, UVFITS, the
summary JSON, and every renderer.

**Tests-first evidence.** Circular MS and UVFITS round trips with raw pyuvdata
inspection; basis-aware autocorrelation normalization; summary receptor block;
plot axis text derived from `result.correlations`.

**Production changes.** Replace the local constants in
`src/radiosim/io/standard_visibility.py` with the shared table; make the write
and read paths, the `(0, 3)` normalization, the `feeds`/`polarization_array`
construction, and the code-order checks basis-aware; extend the projection
history record; add the summary receptor block; derive plot labels.

**Scientific invariants.** Format metadata round trips for both bases; the
autocorrelation normalization acts on the parallel hands in both bases.

**Breaking changes.** Basis-aware writer signatures and read gates.

**Exclusions.** No rename, no documentation sweep, no obsolete-path removal, no
issue closure.

**Stop.** Commit only after both formats round-trip in both bases in both
Python environments. Tier 5G remains unauthorized pending independent 5F
acceptance.

**Independent acceptance.** Open MS and UVFITS with raw pyuvdata and casacore;
verify `feed_array`, `feed_angle`, `polarization_array`, `CORR_TYPE`, and the
on-disk code order against Section 14.2.

**Suggested commit.** `feat(output): record polarization basis in standard formats`

**Next slice.** Tier 5G.

### 34.7 Tier 5G — illumination terminology, configuration guidance, documentation, and samples

**Objective.** Complete the terminology split and align every truth surface.

**Tests-first evidence.** A residual-scan test asserting no receptor-named
identifier survives in `core/beam/analytic.py` and no illumination-named
identifier appears in `core/receptor.py`; documentation residual tests in the
established `tests/unit/test_tier1h_documentation.py` style.

**Production changes.** Perform the Section 15.3 renames; rename
`core/jones/beam/analytic/feed.py` to `illumination.py` and update its
`__init__`; add `configs/receptor_circular_example.yaml`; update
`configs/config.yaml`, `docs/migration_guide.md` (including the stale line 139),
`docs/user_guide/configuration.rst`, `configuration_support.rst`,
`jones_matrices.rst`, `beam_models.rst`, `docs/index.rst`, `README.md`, and the
`CLAUDE.md` implementation-status section, which must now state that K, E, C,
and H implement real physics.

**Scientific invariants.** Documentation states the Section 11.3 modelling
assumption and the Section 12.3 parallactic boundary explicitly.

**Breaking changes.** Private identifier renames; one new sample config.

**Exclusions.** No obsolete-path removal, no issue closure.

**Stop.** Commit only after clean-copy Sphinx and every YAML validation pass.

**Independent acceptance.** Validate both sample configs; run the offline
example; build Sphinx from a clean copy; grep for every removed term.

**Suggested commit.** `docs(feeds): complete illumination and receptor split`

**Next slice.** Tier 5H.

### 34.8 Tier 5H — obsolete-path removal

**Objective.** Delete every path the earlier slices superseded, on evidence.

**Tests-first evidence.** Import-failure and attribute-failure tests for every
removed symbol.

**Production changes.** Remove the duplicated correlation constants if any
survive; remove or convert `visibility_to_correlations`
(`src/radiosim/core/polarization.py:217`) on the evidence gathered in 5A; remove
the `feed_type` / `from_basis` / `to_basis` kwargs and the vacuous
`is_unitary()` overrides; remove the stale `mueller_from_jones`
`NotImplementedError` stub or gate it explicitly as Tier 7, per the §4.2
truthfulness rule; remove any dead illumination duplicate left after 5G.

**Scientific invariants.** No behavior change; the full non-slow suite is
unchanged.

**Breaking changes.** Every removed symbol fails with a documented migration
message.

**Exclusions.** No new behavior, no issue closure.

**Stop.** Commit only after a repository-wide grep proves no reference to any
removed symbol survives in `src/`, `tests/`, `configs/`, `examples/`, or
tracked `docs/`.

**Independent acceptance.** Independently grep and independently import.

**Suggested commit.** `refactor(pol): remove superseded polarization paths`

**Next slice.** Tier 5I.

### 34.9 Tier 5I — independent whole-tier acceptance

**Objective.** Accept or reject Tier 5 as one indivisible gate against
Section 42, and close `POL-001` and `POL-002` only on the complete row set of
Section 43.

**Production changes.** None. `Fix.md` register rows are updated only if every
criterion passes.

**Suggested commit.** `docs(feeds): accept Tier 5 integration`

## 35. Exact writable file list for every slice

### Tier 5A

```text
tests/characterization/test_tier5_current_behavior.py
tests/characterization/test_pyuvdata_321_polarization_contract.py
```

### Tier 5B

```text
src/radiosim/__init__.py
src/radiosim/api/simulator.py
src/radiosim/core/__init__.py
src/radiosim/core/receptor.py
src/radiosim/io/__init__.py
src/radiosim/io/config.py
src/radiosim/io/config_resolution.py
src/radiosim/io/receptor_config.py
tests/fixtures/configs.py
tests/unit/test_core/test_receptor_resolution.py
tests/unit/test_io/test_config.py
tests/unit/test_io/test_config_resolution.py
tests/unit/test_io/test_receptor_config.py
tests/unit/test_simulator/test_api.py
```

### Tier 5C

```text
src/radiosim/core/__init__.py
src/radiosim/core/jones/__init__.py
src/radiosim/core/jones/receptor.py
src/radiosim/core/polarization.py
src/radiosim/core/polarization_basis.py
tests/characterization/test_tier5_current_behavior.py
tests/unit/test_core/test_polarization.py
tests/unit/test_core/test_polarization_basis.py
tests/unit/test_jones/test_basis_transform.py
tests/unit/test_jones/test_receptor.py
```

### Tier 5D

```text
src/radiosim/api/simulator.py
src/radiosim/core/instrument_adapters.py
src/radiosim/core/jones/chain.py
src/radiosim/core/visibility.py
src/radiosim/core/visibility_healpix.py
src/radiosim/simulator/base.py
src/radiosim/simulator/rime.py
tests/unit/test_core/test_beam_solver_integration.py
tests/unit/test_core/test_receptor_solver.py
tests/unit/test_jones/test_chain_order.py
tests/unit/test_simulator/test_api.py
```

### Tier 5E

```text
src/radiosim/core/result.py
src/radiosim/io/hdf5.py
tests/characterization/test_tier4_current_behavior.py
tests/unit/test_core/test_result.py
tests/unit/test_io/test_hdf5_result.py
tests/unit/test_simulator/test_result_integration.py
```

### Tier 5F

```text
src/radiosim/io/measurement_set.py
src/radiosim/io/result_errors.py
src/radiosim/io/standard_visibility.py
src/radiosim/io/summary_json.py
src/radiosim/io/uvfits.py
src/radiosim/visualization/bokeh_plots.py
tests/unit/test_io/test_measurement_set.py
tests/unit/test_io/test_standard_visibility.py
tests/unit/test_io/test_summary_json.py
tests/unit/test_io/test_uvfits.py
tests/unit/test_visualization/
```

### Tier 5G

```text
src/radiosim/core/beam/analytic.py
src/radiosim/core/jones/beam/analytic/__init__.py
src/radiosim/core/jones/beam/analytic/feed.py
src/radiosim/core/jones/beam/analytic/illumination.py
CLAUDE.md
README.md
configs/config.yaml
configs/receptor_circular_example.yaml
docs/index.rst
docs/migration_guide.md
docs/user_guide/beam_models.rst
docs/user_guide/configuration.rst
docs/user_guide/configuration_support.rst
docs/user_guide/jones_matrices.rst
tests/unit/test_tier1h_documentation.py
```

### Tier 5H

```text
src/radiosim/core/jones/receptor.py
src/radiosim/core/polarization.py
src/radiosim/core/result.py
src/radiosim/io/hdf5.py
src/radiosim/io/standard_visibility.py
tests/unit/test_core/test_polarization.py
tests/unit/test_tier5_receptor_acceptance.py
```

### Tier 5I

```text
Fix.md
Tier5ReceptorFeedPlan.md
```

## 36. Independent acceptance gate after every slice

Every slice acceptance is performed by a reviewer who did not write the slice
and who works from source, not from the implementation summary. Each acceptance
must independently:

1. confirm the commit contains only its exact Section 35 file list;
2. re-derive the slice's scientific claims from Section 18 or from the R2/R4
   tables, not from the implementation;
3. author at least one rejected input by hand and confirm the exact message;
4. run the Section 33 gate in both Python environments;
5. confirm no later-tier behavior entered the slice;
6. record the result in `Fix.md` as a dated acceptance note without rewriting
   any prior record.

## 37. Stop boundary after every slice

After each commit the implementer stops. The next slice is unauthorized until
its predecessor is independently accepted. No slice may be split across two
commits, and no two slices may share a commit.

## 38. Breaking-change ledger

| # | Change | Slice | Blast radius |
|---|---|---|---|
| B1 | `RadioSimConfig` gains `receptors` | 5B | additive; default reproduces baseline |
| B2 | `feeds:` guidance text replaced | 5B | error text only |
| B3 | Stokes `V` sign corrected to `C[0,1] = (U + iV)/2` | 5C | cross-hands of `V ≠ 0` sources only |
| B4 | `ReceptorConfigJones` / `BasisTransformJones` constructors replaced | 5C | both classes had no production caller |
| B5 | Jones chain composition order corrected | 5D | currently unobservable; all present factors commute |
| B6 | Solver and result-factory signatures require `receptors` | 5D, 5E | every internal caller |
| B7 | HDF5 schema `1.0.0` → `2.0.0`, `1.0.0` rejected | 5E | every previously written file |
| B8 | `scientific_sha256` changes for every result | 5E | every recorded fingerprint |
| B9 | Four duplicated correlation constants removed | 5E, 5F | internal imports |
| B10 | Writer read gates become basis-aware | 5F | none for linear files |
| B11 | Illumination identifiers renamed | 5G | private symbols only |
| B12 | Superseded polarization helpers removed | 5H | documented migration errors |

## 39. Final whole-tier acceptance criteria

Tier 5I accepts Tier 5 only when all criteria pass as one indivisible gate:

1. The implementation range is linear, every slice was independently accepted,
   and every commit contains only its exact Section 35 file list.
2. All six design decisions in Sections 10-15 are implemented as specified, or
   the plan was amended and re-accepted before the deviation.
3. `stokes_to_coherency` and `coherency_to_stokes` implement the Section 9.1
   brightness matrix and round-trip exactly (S2, S3).
4. `ReceptorConfigJones` and `BasisTransformJones` implement Section 18 exactly,
   are unitary in fact and not only by declaration, and no receptor option
   returns identity except the ones that are analytically identity.
5. The circular correlation table (S4) is reproduced against R2/R4 by an
   independent computation.
6. Unpolarized energy conservation (S5) holds in both bases for every tested
   feed rotation.
7. Rotation invariants S7 and S8 and round trips S6 and S9 hold to machine
   precision.
8. A heterogeneous array transforms into one named output basis (S10), and a
   heterogeneous array under `output_basis: auto` is rejected with the exact
   Section 27 message.
9. Point and HEALPix agree on a common circular case (S12).
10. The Jones chain composes in the Section 19.1 order, proven with
    non-commuting terms (S13).
11. `SimulationResult.correlations` and `.polarization_basis` are derived from
    the resolved receptor set at every construction site, and `stokes_i()`
    derives its indices (S11).
12. HDF5 schema `2.0.0` round-trips both bases, records and validates the basis
    and the receptor group, and rejects `1.0.0` and every hostile reordering.
13. Measurement Set and UVFITS round-trip both bases with independently
    inspected `feed_array`, `feed_angle`, `polarization_array`, `CORR_TYPE`, and
    on-disk code order.
14. The summary JSON reports the true basis, labels, and receptor block, and
    lists per-antenna receptor definitions among its exclusions.
15. Every renderer derives its polarization text from `result.correlations`.
16. `receptor_sha256` and `polarization_basis` enter the scientific fingerprint;
    `instrument_sha256` is unchanged (S14).
17. Every removed input, constructor argument, constant, symbol, and schema
    version fails with its documented migration boundary.
18. No receptor-named identifier survives in illumination code and no
    illumination-named identifier appears in receptor code.
19. Every message in Section 27 is asserted verbatim by a test.
20. `resolve_receptors()` failure leaves no loaded beam, no backend allocation,
    and no output path.
21. Dual-Python focused and full non-slow suites pass with only independently
    classified unavailable-backend skips and established warnings.
22. Ruff, formatting, Pyright under the unchanged ceiling, lock metadata, YAML
    validation, offline example, clean-copy Sphinx, whitespace, fresh imports,
    and generated-artifact checks pass.
23. CI succeeds for quality and all six locked OS/Python jobs on the exact
    acceptance SHA.
24. No physical GPU, network, registry, external-data, or production claim
    appears without direct evidence.
25. No Tier 6 through Tier 8 implementation enters the range.

Any failed criterion keeps `POL-001` and `POL-002` open.

## 40. Evidence required to close POL-001 and POL-002

| Issue | Tier 5I evidence |
|---|---|
| `POL-001` | criteria 1, 2, 8, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20 — a typed top-level receptor section exists, changes the calculated correlations, is recorded in every output format and fingerprint, and every unsupported option is rejected with an actionable message; the illumination/receptor terminology split is complete in configuration *and* identifiers |
| `POL-002` | criteria 3, 4, 5, 6, 7, 9, 10 — `ReceptorConfigJones` and `BasisTransformJones` implement real, unitary, analytically verified physics; the brightness matrix follows the IAU/HBS convention; both solver paths agree; the chain order is correct |

`POL-002` is closed **only** for the C and H terms. `SCI-001` (Z, T, P, D, G, B,
F, W, Ee/a/dE, Kd/Rc/ff, X/Kx/DF, M/Q) remains `ROADMAP` for Tier 7 and is not
touched by Tier 5. `Fix.md` §5 is updated only after the complete row set and
Section 39 pass. A partial success closes no issue.

## 41. Risk register

| Risk | Control and acceptance evidence |
|---|---|
| The Section 10.2 `V`-sign correction is itself wrong | 5A must reproduce R2/R4 and state what `codex-africanus` implements before 5C changes anything; Q1 blocks the slice |
| pyuvdata 3.2.1 rejects circular `feeds` in some writer | 5A probes MS and UVFITS with circular data before 5F depends on it; Q3 blocks 5F, not 5A |
| The chain-order correction changes existing results | proven unobservable by S1 and by the fact that all present factors commute; a bit-identical baseline comparison is required in 5D |
| Basis conversion is silently approximate | Section 11.3 is stated in code, docs, and the plan; `D` and `G` are identity in Tier 5; Tier 7 must revisit |
| `scientific_sha256` churn breaks recorded evidence | B8 is declared in the ledger; Tier 4 acceptance records are historical and are not rewritten |
| HDF5 `2.0.0` bump strands previously written files | pre-v1 policy; the rejection message names Tier 5 explicitly and there is no upgrade path by design |
| Scope creep into other Jones terms | Section 42 exclusions; slice file lists exclude every other term file |
| After 5C, RadioSim's coherency `V` sign diverges from pyradiosky's (added Tier 5A) | `pyradiosky`'s own `stokes_to_coherency` builds `(U - iV)/2` (the sign RadioSim is leaving) and Hamaker 2006 A&A 456, 395 Eq. (3) prints the same form, so the two packages will disagree on cross-hand `V` sign post-5C. Verified no RadioSim data-path defect results: the `pyradiosky_file` loader (`src/radiosim/core/sky/loaders/pyradiosky.py`) reads Stokes `I/Q/U/V` columns from `pyradiosky`'s sky model, never a `pyradiosky`-built coherency matrix, so no conversion in RadioSim's own pipeline uses the opposing convention. 5C's docstring replacement and 5G's documentation sweep must state the divergence explicitly so a user combining RadioSim visibilities with a pyradiosky-based coherency computation does not assume a shared `V` sign. |

## 42. Explicit exclusions and Tier 6+ boundary

Tier 5 does not implement:

- any Jones term other than `C` and `H` — `Z`, `T`, `P`, `D`, `G`, `B`, `F`,
  `W`, `Ee`/`a`/`dE`, `Kd`/`Rc`/`ff`, `X`/`Kx`/`DF`, and baseline `M`/`Q` remain
  identity stubs owned by Tier 7 (`SCI-001`);
- parallactic-angle rotation, time-dependent feed orientation, or steerable
  mounts;
- polarization leakage, `D`-term calibration, or any calibration solve;
- non-scalar E-Jones, cross-polar beams, or heterogeneous beam bases — the
  Tier 3 scalar `e·I₂` boundary is preserved unchanged;
- elliptical, non-orthogonal, single-feed, or multi-feed receptors;
- mixed-hand correlation axes or Stokes-labelled output axes;
- hybrid-sky scheduling, worker control, backend performance, or Tier 6 work;
- spherical harmonics, m-mode, or Tier 7 algorithms;
- a repository-wide documentation rewrite or Tier 8 cleanup;
- physical GPU validation, live network or registry validation, deployment, or
  release.

## 43. Open questions

These are unresolved at the design gate and are recorded rather than assumed.
Each names the slice that must resolve it and what happens if the evidence
contradicts this plan.

**Q1 — The Stokes `V` convention (blocks 5C). RESOLVED by Tier 5A; Section 10.2
stands.** Section 10.2 asserts that the mainstream RIME literature (R2, R3, R4)
uses `C[0,1] = (U + iV)/2` under the IAU `V` definition (R5), and that
`src/radiosim/core/polarization.py:112-113` is the mirror of it. The docstring
at `:22-27` attributes the current form to "Africanus/Pauli". Slice 5A must
reproduce the R2/R4 table from the sources and state explicitly what
`codex-africanus` implements. If the evidence supports the current form, this
plan's Section 10.2, Section 18.4, and the sign of `S` in Section 18.1 must be
amended and re-accepted before 5C proceeds.

**Correction (Tier 5A independent acceptance).** 5A resolved this using R1
(Hamaker, Bregman & Sault 1996, A&AS 117, 137, Paper I, Eq. 8/9) and R3
(Smirnov 2011, A&A 527, A106, Eq. 7 and §6.3) rather than R2/R4: R2 was not
needed, and R4 (Thompson, Moran & Swenson §4.7) could not be retrieved from any
open-access route tried and is recorded as unconfirmed. Both retrieved sources
give `C[0,1] = (U + iV)/2` and `RR = I + V`, independently re-derived by the
5A acceptance reviewer by direct matrix arithmetic from the quoted equations
(not by trusting 5A's prose), confirming Section 10.2 stands. Separately, 5A's
fetch of `codex-africanus` (`ska-sa/codex-africanus`,
`africanus/model/coherency/conversion.py`) found it implements
`"XY": u + v*1j` and `"RR": i + v` — the corrected sign this plan moves
*to*, not the sign RadioSim's current source docstring claims it matches. That
docstring's own "Matches: Codex-Africanus" attribution
(`src/radiosim/core/polarization.py:22-27`) is therefore false; this plan never
asserted otherwise (`codex-africanus` is not one of R1-R8 and Section 10.2 only
compares the baseline against R2-R4), so no Section 10.2/18.1/18.4 amendment is
required — the false attribution is a source-code defect Tier 5C's planned
docstring replacement already corrects. Recorded contrary evidence: pyradiosky
(`stokes_to_coherency`) and Hamaker 2006 A&A 456, 395 Eq. (3) both use the
current `(U - iV)/2` form; see the new pyradiosky-divergence risk in Section
41.

**Q2 — Where `resolve_receptors()` is invoked (blocks 5B).** Section 25.2 places
it after instrument resolution and before beam loading. Whether that is inside
`config_resolution` or inside `Simulator.setup()` depends on where the resolved
instrument first exists in the Tier 1/Tier 2 pipeline. 5B must establish the
exact call site from source and record it; the *ordering* requirement is fixed
by this plan and is not negotiable, only its host function is.

**Q3 — pyuvdata circular-feed acceptance (blocks 5F, not 5A). RESOLVED by Tier
5A; Section 14.2/22.1 hold with two corrections applied above.** Section 14.4
and Section 22.1 assume that `Telescope.new(feeds=["r","l"],
feed_angle=[[0,0],...], mount_type="fixed")` together with
`polarization_array=["rr","rl","lr","ll"]` passes `UVData.check()` and writes
valid MS and UVFITS in pyuvdata 3.2.1, and that the on-disk AIPS order is
descending code order `(-1,-2,-3,-4)`. The installed source supports the
parameter values (`pyuvdata/telescopes.py:404-431`, `utils/pol.py:37`) but the
writer round trip was not executed at this gate. 5A must probe it. If pyuvdata
requires `x_orientation` even for circular feeds, or produces a different
on-disk order, Section 14.2 and Section 22.1 must be amended.

**Correction (Tier 5A independent acceptance).** 5A's live probes
(`tests/characterization/test_pyuvdata_321_polarization_contract.py`, 11
tests, re-run independently on both py311 and py312) confirm the *outcome*
holds — circular `feed_array`/`feed_angle`/`polarization_array` pass
`check()`, and both writers round-trip — with two construction-level
corrections, both applied directly to Sections 14.2 and 14.4 above rather than
left only in this open question: (a) the `feeds=["r","l"]` convenience
argument does not configure anything by itself in pyuvdata 3.2.1 — it is
silently ignored unless `x_orientation` is also supplied
(`pyuvdata/telescopes.py:884-950`), so Tier 5F must construct `feed_array`
directly, not `feeds`; (b) the "on-disk" descending code order
`(-1,-2,-3,-4)` is literally true for UVFITS but for a Measurement Set it is
produced by pyuvdata's *reader* canonicalizing `polarization_array` on
read-back — the MS `POLARIZATION.CORR_TYPE` column itself preserves the
in-memory order (verified: circular `CORR_TYPE = [5,6,7,8]`, not
`(-1,-2,-3,-4)`). Neither correction changes the Q3 verdict, the slice
boundary, or the decision that 5F may proceed on this contract; `x_orientation`
is also confirmed not required for, and not rejected alongside, circular feeds,
and pyuvdata performs no `feed_array`/`polarization_array` cross-validation, so
RadioSim must enforce that coupling itself (already implied by Section 22.1's
basis-aware rejection of mixed labels).

**Q4 — Fate of `visibility_to_correlations` (resolved in 5H).** The function
(`src/radiosim/core/polarization.py:217`) returns hard-keyed linear labels plus
an `"I"` entry. Whether it has any production caller was not established at this
gate. 5A records its callers; 5H either makes it basis-aware or removes it, on
that evidence.

**Q5 — Fate of `mueller_from_jones` (resolved in 5H).** It raises
`NotImplementedError` (`src/radiosim/core/polarization.py:461-464`) while being
publicly exported, which is state 4 of the §4.2 truthfulness rule masquerading
as state 1. 5H either removes it or gates it explicitly as Tier 7. It is not
Tier 5 physics either way.

**Correction (Tier 5A independent acceptance) — premise corrected, resolution
unchanged.** 5A found the "publicly exported" premise false: `mueller_from_jones`
is absent from `radiosim.core.__all__` and from the `radiosim.core` namespace
(`hasattr(radiosim.core, "mueller_from_jones")` is `False`); it is reachable
only as `radiosim.core.polarization.mueller_from_jones`, an undecorated public
module-level name with no leading underscore. `jones_matrix_power` and
`stokes_I_only_visibility` are in the same state; `apply_jones_matrices`,
`visibility_to_correlations`, and `stokes_to_coherency` are the three names
`radiosim.core` actually re-exports. The accurate description is therefore: a
module-level public name, unreachable from the package's advertised surface,
that raises `NotImplementedError` — closer to an unadvertised stub than to
"state 4 masquerading as state 1", since nothing at the `radiosim.core` import
boundary claims it works. This does not change 5H's task: it still either
removes the name or gates it explicitly as Tier 7, on the same evidence.

## 44. Design-gate verification evidence

### 44.1 Starting state

The gate was authored on clean `main` at `1472c3c`
(`docs(output): accept Tier 4 integration`), parent `93bff96`
(`docs(output): accept Tier 4H obsolete-path removal`). `git status` reported no
staged, unstaged, or untracked path at the start of the gate.

### 44.2 What was observed

Every characterization statement in Sections 4, 6, 7, 8, and 31 was taken by
direct read of the working tree at that commit, and every cited line number was
verified there. The installed pyuvdata under `.pixi/envs/default` was inspected
read-only for the facts in Section 31.

One baseline suite run was executed in the default pixi environment. Because
the `test` pixi task is `python -m pytest tests/` (`pixi.toml:23`), the
appended focused paths widened rather than narrowed the selection, so the run
was the **full** suite: **3359 passed, 6 skipped, 26 warnings** in 335 s,
exit code 0, with no failure, xfail, or xpass. That is the Tier 5 starting
baseline.

### 44.3 What was not observed

No CI run was checked for this SHA. The dual-Python boundary (only the default
environment was exercised), Pyright, Sphinx, Ruff, the YAML validations, and the
offline example were **not** executed at this gate. The `-m "not slow"` marker
selection was not exercised separately; the single run above was unmarked and
covered the whole tree. No pyuvdata write probe with circular polarizations was
executed
(open question Q3). No literature PDF was retrieved; the R1-R8 citations are
given so that an independent reviewer can check them, and Q1 requires slice 5A
to do exactly that before any sign change.

### 44.4 Scope

This was documentation-only design work. No production code, test, fixture,
configuration, dependency, lockfile, CI definition, or generated artifact was
changed. `POL-001` remains **OPEN** and `POL-002` remains **ROADMAP**. Tier 5A
remains unauthorized. The next task is an independent review and acceptance of
this document, not implementation.
