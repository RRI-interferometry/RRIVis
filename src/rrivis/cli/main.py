"""Command-line interface for RRIvis.

Provides the main entry point for running simulations from the command line.
"""

# TODO: Future enhancements for v0.3.0+
# =====================================
# The following features were present in the legacy src/main.py (v0.1.x)
# and should be implemented in the new modular CLI/API:
#
# 1. Batch Simulation Support
#    - Run multiple simulations in sequence with automatic naming
#    - batch_index parameter for output folder organization
#    - Summary reporting across all batches
#
# 3. File-Based Logging
#    - Log all console output to simulation folder
#    - Custom stdout/stderr redirection during simulation
#    - Automatic temp folder cleanup of old simulations
#
# 5. Smart Output Folder Management
#    - Base path selection: Explicit → XDG_DOWNLOAD_DIR → ~/Downloads
#    - Timestamp-based folder organization (YYYY-MM-DD_HH-MM-SS)
#    - Auto-save YAML config to simulation folder for reproducibility
#
# 6. PyUVData Telescope Integration
#    - Load known telescope metadata from pyuvdata.Telescope
#    - Auto-populate location, antennas, diameters, mount types, feeds
#    - Per-field control flags (use_pyuvdata_location, use_pyuvdata_antennas, etc.)
#
# 7. Advanced Baseline Filtering
#    - Filter by baseline length with tolerance
#    - Filter by azimuth angle ranges
#    - Toggle autocorrelations/crosscorrelations
#
# See LEGACY_CODE.md in project memory for detailed feature descriptions.

import argparse
import logging
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
  # Run with config file
  rrivis --config config.yaml

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

    parser.add_argument(
        "--config",
        dest="config_flag",
        type=str,
        help="Path to YAML configuration file",
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
    from rrivis.utils.logging import setup_logging, get_logger, print_info, print_success, print_warning, console

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
    config_path = args.config_flag

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

    print_info(f"Loading configuration from: {config_path}")

    try:
        # Load and validate config
        from rrivis.io.config import load_config
        config = load_config(config_path)

        # Pre-flight: collect ALL config errors before doing any work
        errors = config.validate()
        if errors:
            from rich.panel import Panel
            console.print()
            lines = "\n".join(
                f"  [bold red][{i}][/bold red]  {e}"
                for i, e in enumerate(errors, 1)
            )
            console.print(Panel(
                lines,
                title=f"[bold red]Config invalid \u2014 {len(errors)} error(s) found[/bold red]",
                border_style="red",
            ))
            console.print("  Fix the above in your config file and re-run.\n")
            return 1

        # Apply CLI overrides
        if args.antenna_file:
            config.antenna_layout.antenna_positions_file = args.antenna_file

        if args.sim_data_dir:
            config.output.simulation_data_dir = args.sim_data_dir

        # Handle output based on config settings
        output_config = config.output

        # Determine output directory early (before simulation)
        sim_data_dir = output_config.simulation_data_dir or "output"
        sim_subdir = output_config.simulation_subdir

        # Auto-generate subdirectory name if not specified
        if not sim_subdir:
            sim_subdir = config.generate_output_subdir()
            logger.debug(f"Auto-generated output subdirectory: {sim_subdir}")

        output_dir = Path(sim_data_dir) / sim_subdir

        # Handle output folder conflict: prompt whenever the folder already exists
        any_output = output_config.save_simulation_data or output_config.plot_results
        do_overwrite = False  # resolved below based on config + user input
        if any_output and output_dir.exists() and sorted(output_dir.iterdir()):
            existing_files = sorted(output_dir.iterdir())
            if output_config.skip_overwrite_confirmation and not output_config.overwrite_output:
                print_warning(
                    "Conflicting config: overwrite_output is false but skip_overwrite_confirmation is true. "
                    "Aborting to avoid accidental data loss."
                )
                console.print(
                    "  To overwrite silently:       set [bold]overwrite_output: true[/bold] and [bold]skip_overwrite_confirmation: true[/bold]"
                )
                console.print(
                    "  To get the confirmation prompt: set [bold]overwrite_output: true[/bold] and [bold]skip_overwrite_confirmation: false[/bold]"
                )
                return 0
            elif output_config.skip_overwrite_confirmation:
                do_overwrite = True
                print_warning("Output folder exists (confirmation skipped). All existing files will be overwritten.")
            else:
                print_warning(f"Output folder already exists: {output_dir}")
                console.print("  Existing files:")
                for f in existing_files:
                    console.print(f"    [dim]→[/dim]  [cyan]{f.name}[/cyan]")
                console.print("\n  [bold][y][/bold] Overwrite existing files")
                console.print("  [bold][n][/bold] Abort")
                console.print("  [bold][s][/bold] Save to a new folder with a suffix\n")
                try:
                    answer = input("Enter choice [y/n/s]: ").strip().lower()
                except (EOFError, KeyboardInterrupt):
                    answer = "n"
                if answer in ("s", "suffix"):
                    try:
                        suffix = input("Enter suffix to append to folder name: ").strip()
                    except (EOFError, KeyboardInterrupt):
                        suffix = ""
                    if not suffix:
                        print_warning("No suffix entered. Aborted.")
                        return 0
                    sim_subdir = f"{sim_subdir}_{suffix}"
                    output_dir = Path(sim_data_dir) / sim_subdir
                    print_info(f"Saving to new folder: {output_dir}")
                elif answer in ("y", "yes"):
                    do_overwrite = True
                    print_info("Proceeding — all existing files in the output directory will be overwritten.")
                else:
                    print_warning("Aborted. No files were modified.")
                    return 0

        # Set up file logging if save_log_data is enabled
        log_file: Path | None = None
        if output_config.save_log_data:
            output_dir.mkdir(parents=True, exist_ok=True)
            log_file = output_dir / "simulation.log"
            setup_logging(
                level=logging.DEBUG if args.verbose >= 2 else logging.INFO,
                log_file=str(log_file),
            )
            print_info(f"Logging to: {log_file}")

        # Run simulation — CLI --backend flag overrides config compute.backend
        backend = args.backend if args.backend != "auto" else config.compute.backend
        from rrivis.api.simulator import Simulator
        sim = Simulator(config=config.to_dict(), backend=backend)
        sim.run()

        saved_files: list[Path] = []
        if log_file is not None:
            saved_files.append(log_file)

        # Always save the config for reproducibility whenever any output is written
        if any_output:
            try:
                output_dir.mkdir(parents=True, exist_ok=True)
                from rrivis.io.writers import save_config_yaml
                saved_config_path = output_dir / "config.yaml"
                save_config_yaml(config.to_dict(), saved_config_path)
                saved_files.append(saved_config_path)
                logger.debug(f"Config saved to: {saved_config_path}")
            except Exception as e:
                logger.warning(f"Failed to save config: {e}")

        # Save simulation data if requested
        if output_config.save_simulation_data:
            try:
                output_dir.mkdir(parents=True, exist_ok=True)
                data_path = sim.save(
                    output_dir,
                    format="hdf5",
                    overwrite=do_overwrite,
                )
                if data_path:
                    saved_files.append(data_path)
            except Exception as e:
                logger.warning(f"Failed to save results: {e}")

        # Generate plots if requested
        if output_config.plot_results:
            try:
                plot_paths = sim.plot(
                    plot_type="all",
                    output_dir=output_dir,
                    backend=output_config.plotting_backend or "bokeh",
                    show=output_config.open_plots_in_browser,
                    overwrite=output_config.overwrite_output,
                )
                if plot_paths:
                    saved_files.extend(plot_paths)
            except Exception as e:
                logger.warning(f"Failed to generate plots: {e}")

        # Final output summary panel
        if saved_files:
            from rich.panel import Panel
            from rich.table import Table
            table = Table(show_header=False, box=None, padding=(0, 1))
            table.add_column("", style="dim")
            table.add_column("", style="cyan")
            for f in saved_files:
                table.add_row("→", str(f))
            console.print(Panel(table, title="[bold]Output Files[/bold]", border_style="green"))

        print_success("Done.")
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
        if args.config_flag:
            return run_config_mode(args)
        else:
            parser.print_help()
            return 0


if __name__ == "__main__":
    sys.exit(main())
