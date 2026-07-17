# RadioSim examples

The examples use the strict Tier 1 configuration and the public `Simulator`
API. The default script and notebook are deterministic, local, and
NumPy-backed; they do not require network access.

## Script

From the repository root:

```bash
pixi run python examples/scripts/simple_simulation.py --help
pixi run python examples/scripts/simple_simulation.py --no-plot
```

The smoke run writes nothing. Saving and plotting are explicit:

```bash
pixi run python examples/scripts/simple_simulation.py \
  --save --output-dir simulation_output

pixi run python examples/scripts/simple_simulation.py \
  --plot --output-dir simulation_output
```

To run a shipped YAML document without executing its CLI workflow section:

```bash
pixi run python examples/scripts/simple_simulation.py \
  --config configs/config.yaml --no-plot
```

`--backend` is an explicit override. Omitting it preserves a YAML document's
backend; the built-in example uses NumPy.

## Notebook

Open `notebooks/01_basic_usage.ipynb` from the repository root. Its default
cells use the bundled HERA layout and synthetic sources. Plotting and result
saving are disabled unless you change their explicit flags.

## Shipped configurations

- `configs/config.yaml` is an offline synthetic point-source example.
- `configs/realistic_foreground_example.yaml` describes a Haslam and GLEAM
  foreground recipe. Validation is local, but simulation may require network
  access and optional sky-model dependencies.

NumPy is the deterministic default. JAX and Numba can be selected when their
optional dependencies are installed, but backend selection alone is not a
claim of end-to-end GPU acceleration.
