# Committed benchmark reference records

This directory holds the curated benchmark records the repository keeps under
version control. Every ordinary `pixi run bench` run writes
`output/benchmarks/<UTC timestamp>-<host tag>.json`, which stays ignored so a
local benchmark never dirties the tree; only the records below are retained,
and each is retained because a written statement cites it.

## What lives here

| path | schema | governed by |
|------|--------|-------------|
| `<UTC>-<host>.json` | `radiosim.benchmark.v2` | `Tier6HybridRuntimePlan.md` Sections 22.1, 32.9 |
| `perf001/<UTC>-<host>.json` | `radiosim.benchmark.perf001.v1` | the accepted `PERF-001` records |
| `sci004/<UTC>-<host>.json` | `radiosim.benchmark.sci004.v1` | `docs/development/sci004_mmode_design.md` Section 11 |

The `.gitignore` exception is deliberately narrow in all three cases: the
`*.json` records and this file are tracked, and nothing else in the directory
is. A stray file or a nested directory stays ignored.

## What a record is evidence of, and what it is not

A record is evidence of the rows it actually measured, on the machine and in the
environment its own `provenance` block names, and of nothing else. Each schema
carries a per-row `claims_not_licensed` array that says so inside the record.

For the SCI-004 m-mode record the array is exactly:

```text
general_speedup
gpu_or_accelerator_support
perf001_evidence_or_closure
performance_regression_gate
unmeasured_workloads
```

Section 11 states the consequence in the memo's own words: "A record is evidence
only of these nine measured CPU rows. Timing values never gate CI and license
neither a speedup nor a memory/accelerator advantage. `PERF-001` statements
remain governed by separate accepted PERF-001 records."

None of these records gates CI. The suite runs `-m "not slow"`, and the
performance tests are marked `performance` and `slow`; a timing number here has
never failed a build and is not permitted to.

No accelerator run of RadioSim has ever been measured. Nothing in this directory
claims one, and the JAX rows it does contain were measured on CPU builds.

## The SCI-004 inventory

`radiosim.benchmark.sci004.v1` defines its own schema rather than extending the
PERF-001 inventory, because every SCI-004 row joins a frame certificate, a
scientific identity, a deterministic block schedule, and a direct/backend
comparison for which the PERF-001 record has no analogue. Its `workloads` array
is the exact nine-row Cartesian product of three fixtures and three backends:

```text
fixture: mmode_single_scalar_mode, mmode_point_stokes_i, mmode_point_full_stokes
backend within each fixture: numpy, jax, dask
```

All three fixtures are point-representation runs, which is the capability the
accepted phase M2 licenses through the public solve path. The three rows in a
fixture group share their input, frame-certificate, dimension, precision, worker
and memory-budget fields; their scientific and result-cube identities stay
backend-qualified and are compared by the fixed `sci004_backend_complex128.v1`
predicate at `rtol = 1e-12`. Each row additionally carries the every-run
`sci004_two_tier_direct.v3` comparison against the frozen direct reference,
whose deficit is a disclosed, budget-bounded truncation difference and is never
asserted as agreement.

Backend *correctness* parity is what these rows establish. Backend *performance*
remains a roadmap item, and the measured JAX-CPU timings in the existing
`radiosim.benchmark.v2` records are slower than NumPy on every benchmarked
workload.
