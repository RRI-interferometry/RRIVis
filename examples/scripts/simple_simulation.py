#!/usr/bin/env python
"""Run a small deterministic RadioSim visibility simulation.

The built-in example uses NumPy, local test sources, and the bundled five-
antenna layout. It needs no network access and writes no output artifacts.
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
        choices=("auto", "numpy", "jax", "dask"),
        default=None,
        help=(
            "Override the YAML backend. With the built-in example, omission uses "
            "the deterministic NumPy default."
        ),
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
    from radiosim.io.instrument_config import (
        BaselineSelectionConfig,
        InstrumentConfig,
        InstrumentLocationConfig,
        LayoutFileSourceConfig,
    )

    repository_root = Path(__file__).resolve().parents[2]
    antenna_file = repository_root / "antenna_layout_examples" / "hera_5.txt"
    execution = ExecutionConfig(
        backend=backend or "numpy",
        precision=PrecisionInput(preset="standard"),
        offline=True,
    )
    instrument = InstrumentConfig(
        source=LayoutFileSourceConfig(
            path=antenna_file,
            format="radiosim",
            telescope_name="HERA",
        ),
        location=InstrumentLocationConfig(
            longitude_deg=21.4283,
            latitude_deg=-30.72152,
            height_m=1073.0,
        ),
        default_diameter_m=14.0,
    )
    return Simulator.from_parameters(
        instrument=instrument,
        baseline_selection=BaselineSelectionConfig(correlations="all"),
        channel_frequencies_hz=(100_000_000.0, 105_000_000.0),
        channel_widths_hz=(1_000_000.0, 1_000_000.0),
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
    print(f"Antennas: {len(simulator.antennas)}")
    print(f"Baselines: {len(simulator.baselines)}")
    print(
        f"Frequency channels: {len(simulator.config.frequency.channel_frequencies_hz)}"
    )

    estimate = simulator.get_memory_estimate()
    print(f"Estimated memory: {estimate.get('total_human', 'unavailable')}")

    result = simulator.run(progress=args.progress)
    assert result is simulator.result
    if args.config is None:
        # Only the built-in example has fixed dimensions: five HERA antennas
        # (fifteen baselines), two channels, one time sample.  A document
        # supplied with --config defines its own grid, so asserting these here
        # would make every shipped configuration fail.
        assert result.visibilities.shape == (1, 15, 2, 4)
    print(f"Visibility shape (T, B, F, C): {result.visibilities.shape}")
    print(f"Correlations: {', '.join(result.correlations)}")
    print(f"Stokes-I shape: {result.stokes_i().shape}")
    print(f"Scientific fingerprint: {result.scientific_sha256}")
    print(f"Provenance fingerprint: {result.provenance_sha256}")

    print("Simulation complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
