#!/usr/bin/env python
"""Simple RRIvis visibility simulation example.

This script demonstrates the basic usage of RRIvis for simulating
radio interferometry visibilities using the RIME (Radio Interferometer
Measurement Equation).

Usage:
    python simple_simulation.py
    python simple_simulation.py --backend jax  # Use GPU acceleration
    python simple_simulation.py --config path/to/config.yaml
"""

import argparse
from pathlib import Path

import numpy as np


def main():
    """Run a simple visibility simulation."""
    parser = argparse.ArgumentParser(
        description="Simple RRIvis visibility simulation example"
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="auto",
        choices=["auto", "numpy", "jax", "numba"],
        help="Computation backend (default: auto)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="simulation_output",
        help="Output directory for results",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Disable plotting",
    )
    args = parser.parse_args()

    # Import RRIvis
    import rrivis
    from rrivis import Simulator
    from rrivis.backends import list_backends

    print(f"RRIvis version: {rrivis.__version__}")
    print(f"Available backends: {list_backends()}")
    print()

    # Option 1: From configuration file
    if args.config:
        print(f"Loading configuration from: {args.config}")
        sim = Simulator.from_config(args.config)
    else:
        # Option 2: Programmatic configuration
        print("Using programmatic configuration...")

        # Get the antenna layout example path
        examples_dir = Path(__file__).parent.parent.parent
        antenna_file = examples_dir / "antenna_layout_examples" / "example_rrivis_format.txt"

        if not antenna_file.exists():
            # Fallback: create a simple inline antenna layout
            print("Creating simple test antenna layout...")
            antenna_file = None
            # Will use default test configuration

        sim = Simulator(
            antenna_layout=str(antenna_file) if antenna_file else None,
            frequencies=[100, 120, 140, 160, 180, 200],  # MHz
            sky_model="test",  # Use test sources
            location={
                "lat": -30.72,  # HERA latitude
                "lon": 21.43,   # HERA longitude
                "height": 1073.0,
            },
            start_time="2025-01-15T00:00:00",
            backend=args.backend,
        )

    # Show simulation configuration
    print("=" * 60)
    print("Simulation Configuration")
    print("=" * 60)
    print(f"  Backend: {args.backend}")

    # Setup the simulation (loads antennas, generates baselines, loads sources)
    print("\nSetting up simulation...")
    sim.setup()

    # Show setup information
    print(f"  Antennas: {len(sim._antennas) if sim._antennas else 'N/A'}")
    print(f"  Baselines: {len(sim._baselines) if sim._baselines else 'N/A'}")
    print(f"  Sources: {len(sim._sources) if sim._sources else 'N/A'}")
    print(f"  Frequencies: {len(sim._frequencies_hz) if sim._frequencies_hz is not None else 'N/A'}")

    # Estimate memory usage
    memory_estimate = sim.get_memory_estimate()
    print(f"\nEstimated memory usage: {memory_estimate:.2f} MB")

    # Run the simulation
    print("\nRunning simulation...")
    print("-" * 60)
    results = sim.run(progress=True)
    print("-" * 60)

    # Display results summary
    print("\nResults Summary")
    print("=" * 60)
    if results and "visibilities" in results:
        vis = results["visibilities"]
        n_baselines = len(vis)
        print(f"  Number of baselines: {n_baselines}")

        # Get sample visibility info
        sample_key = list(vis.keys())[0]
        sample_vis = vis[sample_key]
        print(f"  Sample baseline: {sample_key}")
        print(f"  Visibility shape per baseline: {sample_vis.shape}")

        # Show available correlation products
        if hasattr(results, "correlation_products"):
            print(f"  Correlation products: {results.get('correlation_products', ['XX', 'XY', 'YX', 'YY'])}")

    # Save results
    output_dir = Path(args.output_dir)
    print(f"\nSaving results to: {output_dir}")
    sim.save(str(output_dir), format="hdf5")
    print("  Results saved successfully!")

    # Plot results (if not disabled)
    if not args.no_plot:
        print("\nGenerating plots...")
        try:
            sim.plot(plot_type="all")
            print("  Plots generated successfully!")
        except Exception as e:
            print(f"  Warning: Could not generate plots: {e}")

    print("\nSimulation complete!")
    return 0


if __name__ == "__main__":
    exit(main())
