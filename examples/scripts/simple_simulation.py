#!/usr/bin/env python
"""Run a small deterministic RadioSim visibility simulation.

The built-in example uses NumPy, local test sources, and the bundled five-
antenna layout. It needs no network access and writes nothing unless ``--save``
or ``--plot`` is supplied.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a small offline RadioSim visibility simulation."
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="Use a strict RadioSim YAML document instead of the built-in example.",
    )
    parser.add_argument(
        "--backend",
        choices=("auto", "numpy", "jax", "numba"),
        default=None,
        help=(
            "Override the YAML backend. With the built-in example, omission uses "
            "the deterministic NumPy default."
        ),
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save HDF5 results to --output-dir.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("simulation_output"),
        help="Directory used only when --save or --plot is requested.",
    )
    plot_group = parser.add_mutually_exclusive_group()
    plot_group.add_argument(
        "--plot",
        action="store_true",
        help="Write antenna plots without opening a browser.",
    )
    plot_group.add_argument(
        "--no-plot",
        action="store_true",
        help="Explicitly disable plotting (the default).",
    )
    parser.add_argument(
        "--progress",
        action="store_true",
        help="Show the simulation progress header and summary.",
    )
    return parser


def _built_in_simulator(backend: str | None):
    from radiosim import Simulator
    from radiosim.io.config import ExecutionConfig, PrecisionInput

    repository_root = Path(__file__).resolve().parents[2]
    antenna_file = repository_root / "antenna_layout_examples" / "hera_5.txt"
    execution = ExecutionConfig(
        backend=backend or "numpy",
        precision=PrecisionInput(preset="standard"),
        offline=True,
    )
    return Simulator.from_parameters(
        antenna_layout=antenna_file,
        antenna_file_format="radiosim",
        antenna_diameter_m=14.0,
        channel_frequencies_hz=(100_000_000.0, 105_000_000.0),
        location={"lat": -30.72152, "lon": 21.4283, "height": 1073.0},
        start_time="2025-01-01T00:00:00",
        duration_seconds=1.0,
        time_step_seconds=1.0,
        sky_model={
            "sources": [
                {
                    "kind": "test_sources",
                    "num_sources": 3,
                    "distribution": "uniform",
                    "seed": 7,
                    "flux_min": 1.0,
                    "flux_max": 2.0,
                    "dec_deg": -30.0,
                }
            ]
        },
        execution=execution,
    )


def _simulator_from_args(args: argparse.Namespace):
    from radiosim import Simulator
    from radiosim.io.config_resolution import SimulationOverrides

    if args.config is None:
        return _built_in_simulator(args.backend)
    overrides = (
        None if args.backend is None else SimulationOverrides(backend=args.backend)
    )
    return Simulator.from_yaml(args.config, overrides=overrides)


def main() -> int:
    """Run the example and report public result surfaces."""
    args = _parser().parse_args()
    simulator = _simulator_from_args(args)

    print(f"Requested backend: {simulator.config.execution.backend_strategy}")
    simulator.setup()
    print(f"Antennas: {len(simulator.antennas or {})}")
    print(f"Baselines: {len(simulator.baselines or {})}")
    print(
        f"Frequency channels: {len(simulator.config.frequency.channel_frequencies_hz)}"
    )

    estimate = simulator.get_memory_estimate()
    print(f"Estimated memory: {estimate.get('total_human', 'unavailable')}")

    results = simulator.run(progress=args.progress)
    visibilities = results["visibilities"]
    sample_baseline = next(iter(visibilities))
    products = visibilities[sample_baseline]
    shapes = {name: value.shape for name, value in products.items()}
    print(f"Visibility baselines: {len(visibilities)}")
    print(f"Sample baseline: {sample_baseline}")
    print(f"Sample product shapes: {shapes}")

    if args.save:
        saved = simulator.save(args.output_dir, format="hdf5")
        print(f"Saved results: {saved}")
    if args.plot:
        paths = simulator.plot(
            plot_type="antenna",
            output_dir=args.output_dir,
            show=False,
        )
        print(f"Saved plots: {len(paths)}")

    print("Simulation complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
