"""Command-line interface for RRIvis.

Provides the main entry point for running simulations from the command line.
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

from rrivis.__about__ import __version__, __description__


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser for the CLI."""
    parser = argparse.ArgumentParser(
        prog="rrivis",
        description=__description__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with config file (v0.1.x compatible)
  rrivis config.yaml

  # Run with CLI arguments
  rrivis simulate --antenna-layout HERA65.csv --frequencies 100,150,200

  # Show version
  rrivis --version

For more information, see https://github.com/kartikmandar/RRIvis
        """,
    )

    parser.add_argument(
        "--version",
        action="version",
        version=f"rrivis {__version__}",
    )

    # Subcommands
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Config file mode (default, backward compatible)
    parser.add_argument(
        "config",
        nargs="?",
        type=str,
        help="Path to YAML configuration file",
    )

    parser.add_argument(
        "--config",
        dest="config_flag",
        type=str,
        help="Path to YAML configuration file (alternative syntax)",
    )

    parser.add_argument(
        "--antenna-file",
        type=str,
        help="Path to antenna positions file (overrides config)",
    )

    parser.add_argument(
        "--sim-data-dir",
        type=str,
        help="Directory for simulation output (overrides config)",
    )

    parser.add_argument(
        "--backend",
        choices=["auto", "numpy", "jax", "numba"],
        default="auto",
        help="Computation backend (default: auto)",
    )

    parser.add_argument(
        "--verbose", "-v",
        action="count",
        default=0,
        help="Increase verbosity (use -vv for debug)",
    )

    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress non-error output",
    )

    # Simulate subcommand (new CLI arguments mode)
    simulate_parser = subparsers.add_parser(
        "simulate",
        help="Run simulation with CLI arguments",
    )

    simulate_parser.add_argument(
        "--antenna-layout",
        type=str,
        required=True,
        help="Path to antenna positions file",
    )

    simulate_parser.add_argument(
        "--frequencies",
        type=str,
        required=True,
        help="Frequencies in MHz (comma-separated, e.g., '100,150,200')",
    )

    simulate_parser.add_argument(
        "--sky-model",
        choices=["test", "gleam", "gsm"],
        default="test",
        help="Sky model to use (default: test)",
    )

    simulate_parser.add_argument(
        "--output",
        type=str,
        default="output/",
        help="Output directory (default: output/)",
    )

    simulate_parser.add_argument(
        "--format",
        choices=["hdf5", "json", "ms"],
        default="hdf5",
        help="Output format (default: hdf5)",
    )

    simulate_parser.add_argument(
        "--backend",
        choices=["auto", "numpy", "jax", "numba"],
        default="auto",
        help="Computation backend (default: auto)",
    )

    # Init subcommand (create default config)
    init_parser = subparsers.add_parser(
        "init",
        help="Create a default configuration file",
    )

    init_parser.add_argument(
        "--output", "-o",
        type=str,
        default="config.yaml",
        help="Output config file path (default: config.yaml)",
    )

    # Validate subcommand
    validate_parser = subparsers.add_parser(
        "validate",
        help="Validate a configuration file",
    )

    validate_parser.add_argument(
        "config",
        type=str,
        help="Path to configuration file to validate",
    )

    return parser


def run_config_mode(args: argparse.Namespace) -> int:
    """Run simulation using configuration file (v0.1.x compatible)."""
    from rrivis.utils.logging import setup_logging, get_logger

    # Setup logging based on verbosity
    import logging
    if args.quiet:
        level = logging.ERROR
    elif args.verbose >= 2:
        level = logging.DEBUG
    elif args.verbose >= 1:
        level = logging.INFO
    else:
        level = logging.INFO

    setup_logging(level=level)
    logger = get_logger("rrivis.cli")

    # Determine config file path
    config_path = args.config or args.config_flag

    if not config_path:
        logger.error("No configuration file specified")
        logger.info("Usage: rrivis config.yaml")
        logger.info("       rrivis --config config.yaml")
        logger.info("       rrivis simulate --antenna-layout file.csv --frequencies 100,200")
        return 1

    config_path = Path(config_path)
    if not config_path.exists():
        logger.error(f"Configuration file not found: {config_path}")
        return 1

    logger.info(f"Loading configuration from: {config_path}")

    try:
        # Load and validate config
        from rrivis.io.config import load_config
        config = load_config(config_path)

        # Apply CLI overrides
        if args.antenna_file:
            config.antenna_layout.antenna_positions_file = args.antenna_file

        if args.sim_data_dir:
            config.output.simulation_data_dir = args.sim_data_dir

        # Run simulation
        from rrivis.api.simulator import Simulator
        sim = Simulator(config=config.to_dict())
        results = sim.run()

        logger.info("Simulation completed successfully")
        return 0

    except Exception as e:
        logger.error(f"Simulation failed: {e}")
        if args.verbose >= 2:
            import traceback
            traceback.print_exc()
        return 1


def run_simulate_mode(args: argparse.Namespace) -> int:
    """Run simulation using CLI arguments (new mode)."""
    from rrivis.utils.logging import setup_logging, get_logger

    setup_logging()
    logger = get_logger("rrivis.cli")

    # Parse frequencies
    try:
        frequencies = [float(f.strip()) for f in args.frequencies.split(",")]
    except ValueError:
        logger.error(f"Invalid frequencies format: {args.frequencies}")
        logger.info("Expected comma-separated numbers, e.g., '100,150,200'")
        return 1

    logger.info(f"Running simulation with {len(frequencies)} frequencies")
    logger.info(f"Antenna layout: {args.antenna_layout}")
    logger.info(f"Sky model: {args.sky_model}")

    try:
        from rrivis.api.simulator import Simulator

        sim = Simulator(
            antenna_layout=args.antenna_layout,
            frequencies=frequencies,
            sky_model=args.sky_model,
            backend=args.backend,
        )

        results = sim.run()
        sim.save(args.output, format=args.format)

        logger.info(f"Results saved to: {args.output}")
        return 0

    except Exception as e:
        logger.error(f"Simulation failed: {e}")
        return 1


def run_init_mode(args: argparse.Namespace) -> int:
    """Create default configuration file."""
    from rrivis.io.config import create_default_config

    output_path = Path(args.output)
    if output_path.exists():
        print(f"File already exists: {output_path}")
        response = input("Overwrite? [y/N] ")
        if response.lower() != "y":
            print("Aborted.")
            return 1

    create_default_config(output_path)
    print(f"Created default configuration: {output_path}")
    return 0


def run_validate_mode(args: argparse.Namespace) -> int:
    """Validate configuration file."""
    from rrivis.io.config import load_config

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"File not found: {config_path}")
        return 1

    try:
        config = load_config(config_path)
        print(f"Configuration is valid: {config_path}")
        print(f"  Telescope: {config.telescope.telescope_name}")
        print(f"  Antenna file: {config.antenna_layout.antenna_positions_file}")
        print(f"  Frequency range: {config.obs_frequency.starting_frequency} - "
              f"{config.obs_frequency.starting_frequency + config.obs_frequency.frequency_bandwidth} "
              f"{config.obs_frequency.frequency_unit}")
        return 0
    except Exception as e:
        print(f"Configuration is INVALID: {e}")
        return 1


def main(argv: Optional[list] = None) -> int:
    """Main entry point for the CLI."""
    parser = create_parser()
    args = parser.parse_args(argv)

    # Handle subcommands
    if args.command == "simulate":
        return run_simulate_mode(args)
    elif args.command == "init":
        return run_init_mode(args)
    elif args.command == "validate":
        return run_validate_mode(args)
    else:
        # Default: config file mode (backward compatible)
        if args.config or args.config_flag:
            return run_config_mode(args)
        else:
            parser.print_help()
            return 0


if __name__ == "__main__":
    sys.exit(main())
