# RadioSim examples

The examples use the strict Tier 1 configuration and the public `Simulator`
API. The script and the notebook are deterministic, local, and NumPy-backed;
they need no network access.

## Script

`scripts/simple_simulation.py` runs a small offline simulation and prints the
public result surfaces. It writes no output artifacts, which is what makes it
safe to run unconditionally. Saving and plotting are not part of it; those are
demonstrated with their real signatures in
[`docs/quickstart.rst`](../docs/quickstart.rst).

The parser defines exactly three options — `--config`, `--backend` and
`--progress` — plus argparse's own `--help`. From the repository root:

```bash
pixi run python examples/scripts/simple_simulation.py --help
pixi run python examples/scripts/simple_simulation.py
```

`--progress` adds the simulation progress header and summary:

```bash
pixi run python examples/scripts/simple_simulation.py --progress
```

`--config` runs a shipped YAML document instead of the built-in example,
without executing that document's CLI workflow section:

```bash
pixi run python examples/scripts/simple_simulation.py --config configs/config.yaml
```

`--backend` is an explicit override. Omitting it preserves a YAML document's
backend; the built-in example uses NumPy.

## Notebook

Open `notebooks/01_basic_usage.ipynb` from the repository root. Its cells use
the bundled HERA layout and synthetic sources, and the notebook writes no
artifacts. It is checked by executing it, not by reading it:

```bash
pixi run jupyter nbconvert --to notebook --execute --stdout \
  examples/notebooks/01_basic_usage.ipynb
```

## Shipped configurations

`configs/` holds four documents; all four validate offline with
`pixi run radiosim validate <path>`.

- `configs/config.yaml` — the default offline sample: the five-antenna HERA
  layout, an analytic circular-aperture beam, and 200 synthetic point sources.
- `configs/hybrid_sky_example.yaml` — `visibility.sky_representation: hybrid`,
  summing a point-source cube and a HEALPix cube on one shared grid. Both
  payloads are synthetic, so it runs fully offline.
- `configs/realistic_foreground_example.yaml` — a Haslam diffuse template
  combined with bright GLEAM sources. Validation is local, but **simulation
  needs network access** and the optional sky-model dependencies.
- `configs/receptor_circular_example.yaml` — circular receptors on every
  antenna, so the reported correlations are `RR, RL, LR, LL` instead of
  `XX, XY, YX, YY`.

## Backends

NumPy is the deterministic default. The selectable names are `numpy`, `jax`,
`dask` and `auto`; `numba` was removed in Tier 6H because the backend behind
that name never compiled a kernel. Selecting a backend is a correctness-parity
choice, not a performance claim: the measured records in
[`output/benchmarks/reference/`](../output/benchmarks/reference/) show Dask
bit-identical to NumPy and JAX-CPU slower than NumPy on every benchmarked
workload, and no accelerator has been exercised. The full statement, with the
numbers and their provenance, is in the repository
[README's backend section](../README.md#backends-and-performance).
