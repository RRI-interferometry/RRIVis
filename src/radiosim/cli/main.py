"""Command-line interface for RadioSim.

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
# See LEGACY_CODE.md in project memory for detailed feature descriptions.

import logging
import sys
from pathlib import Path
from typing import Literal, cast

import click

from radiosim.__about__ import __description__, __version__

_BACKEND_CHOICES = click.Choice(["auto", "numpy", "jax", "numba"])
BackendStrategy = Literal["auto", "numpy", "jax", "numba"]


@click.group(
    invoke_without_command=True,
    help=__description__,
    epilog=(
        "Examples:\n\n"
        "  # Run with config file\n"
        "  radiosim --config config.yaml\n\n"
        "  # Run with CLI arguments\n"
        "  radiosim simulate --antenna-layout "
        "antenna_layout_examples/hera_5.txt --frequencies 100,150,200 "
        "--telescope-name HERA --default-diameter-m 14 "
        "--latitude -30.7 --longitude 21.4 --height 1073 "
        "--start-time 2025-01-01T00:00:00\n\n"
        "  # Show version\n"
        "  radiosim --version\n\n"
        "For more information, see https://github.com/RRI-interferometry/RadioSim"
    ),
)
@click.version_option(version=__version__, prog_name="radiosim")
@click.option(
    "--config", "config_flag", type=click.Path(), help="Path to YAML configuration file"
)
@click.option(
    "--antenna-file",
    type=click.Path(path_type=Path),
    default=None,
    help="Path override for a config layout_file instrument source",
)
@click.option(
    "--sim-data-dir",
    type=click.Path(path_type=Path),
    default=None,
    help="Directory for simulation output (overrides config)",
)
@click.option(
    "--backend",
    type=_BACKEND_CHOICES,
    default=None,
    show_default=False,
    help="Computation backend override",
)
@click.option(
    "-v", "--verbose", count=True, help="Increase verbosity (use -vv for debug)"
)
@click.option("-q", "--quiet", is_flag=True, help="Suppress non-error output")
@click.option(
    "--offline/--online",
    default=None,
    help="Override network mode; omit to preserve the configuration",
)
@click.pass_context
def cli(
    ctx: click.Context,
    config_flag: str | None,
    antenna_file: Path | None,
    sim_data_dir: Path | None,
    backend: BackendStrategy | None,
    verbose: int,
    quiet: bool,
    offline: bool | None,
) -> None:
    """RadioSim — Radio Interferometer Visibility Simulator."""
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose
    ctx.obj["quiet"] = quiet
    ctx.obj["backend"] = backend
    ctx.obj["antenna_file"] = antenna_file
    ctx.obj["sim_data_dir"] = sim_data_dir
    ctx.obj["offline"] = offline

    if ctx.invoked_subcommand is None:
        if config_flag:
            sys.exit(
                run_config_mode(
                    config_flag=config_flag,
                    antenna_file=antenna_file,
                    sim_data_dir=sim_data_dir,
                    backend=backend,
                    verbose=verbose,
                    quiet=quiet,
                    offline=offline,
                )
            )
        else:
            click.echo(ctx.get_help())


@cli.command()
@click.option(
    "--antenna-layout", required=True, type=str, help="Path to antenna positions file"
)
@click.option(
    "--telescope-name", required=True, type=str, help="Canonical telescope name"
)
@click.option(
    "--default-diameter-m",
    type=click.FloatRange(min=0.0, min_open=True),
    default=None,
    help="Positive fallback diameter in metres; omit when the layout is complete",
)
@click.option(
    "--correlations",
    type=click.Choice(["all", "cross", "auto"]),
    default="all",
    show_default=True,
    help="Baseline correlation selection",
)
@click.option(
    "--frequencies",
    required=True,
    type=str,
    help="Frequencies in MHz (comma-separated, e.g., '100,150,200')",
)
@click.option(
    "--sky-model",
    type=click.Choice(["test", "gleam", "gsm"]),
    default="test",
    show_default=True,
    help="Sky model to use",
)
@click.option("--output", default="output/", show_default=True, help="Output directory")
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["hdf5", "json", "ms"]),
    default="hdf5",
    show_default=True,
    help="Output format",
)
@click.option(
    "--backend",
    type=_BACKEND_CHOICES,
    default="auto",
    show_default=True,
    help="Computation backend",
)
@click.option(
    "--latitude", required=True, type=float, help="Latitude in degrees (required)"
)
@click.option(
    "--longitude", required=True, type=float, help="Longitude in degrees (required)"
)
@click.option("--height", required=True, type=float, help="Height in meters (required)")
@click.option(
    "--start-time",
    required=True,
    type=str,
    help="Observation start time (required)",
)
@click.option(
    "--duration-seconds",
    type=float,
    default=1.0,
    show_default=True,
    help="Observation duration in seconds",
)
@click.option(
    "--time-step-seconds",
    type=float,
    default=1.0,
    show_default=True,
    help="Observation cadence in seconds",
)
def simulate(
    antenna_layout: str,
    telescope_name: str,
    default_diameter_m: float | None,
    correlations: Literal["all", "cross", "auto"],
    frequencies: str,
    sky_model: str,
    output: str,
    output_format: str,
    backend: BackendStrategy,
    latitude: float,
    longitude: float,
    height: float,
    start_time: str,
    duration_seconds: float,
    time_step_seconds: float,
) -> None:
    """Run simulation with CLI arguments."""
    sys.exit(
        run_simulate_mode(
            antenna_layout=antenna_layout,
            telescope_name=telescope_name,
            default_diameter_m=default_diameter_m,
            correlations=correlations,
            frequencies=frequencies,
            sky_model=sky_model,
            output=output,
            output_format=output_format,
            backend=backend,
            latitude=latitude,
            longitude=longitude,
            height=height,
            start_time=start_time,
            duration_seconds=duration_seconds,
            time_step_seconds=time_step_seconds,
        )
    )


@cli.command()
@click.option(
    "-o",
    "--output",
    default="config.yaml",
    show_default=True,
    help="Output config file path",
)
def init(output: str) -> None:
    """Create a default configuration file."""
    sys.exit(run_init_mode(output=output))


@cli.command()
@click.argument("config", type=click.Path())
def validate(config: str) -> None:
    """Validate a configuration file."""
    sys.exit(run_validate_mode(config=config))


@cli.command()
@click.pass_context
def status(ctx: click.Context) -> None:
    """Check network connectivity and service availability."""
    sys.exit(run_status_mode(verbose=ctx.obj.get("verbose", 0)))


# ---------------------------------------------------------------------------
# Handler functions — parameters are direct (no Namespace)
# ---------------------------------------------------------------------------


def run_config_mode(
    config_flag: str | None,
    antenna_file: str | Path | None,
    sim_data_dir: str | Path | None,
    backend: BackendStrategy | None,
    verbose: int,
    quiet: bool,
    offline: bool | None = None,
) -> int:
    """Resolve one document, run its scientific state, then its workflow."""
    from radiosim.utils.logging import (
        get_logger,
        print_info,
        print_success,
        setup_logging,
    )

    if quiet:
        level = logging.ERROR
    elif verbose >= 2:
        level = logging.DEBUG
    elif verbose >= 1:
        level = logging.INFO
    else:
        level = logging.INFO

    setup_logging(level=level)
    logger = get_logger("radiosim.cli")

    if not config_flag:
        logger.error("No configuration file specified")
        logger.info("Usage: radiosim --config config.yaml")
        return 1

    config_path = Path(config_flag)
    print_info(f"Loading configuration from: {config_path}")

    try:
        from radiosim.io.config import load_config
        from radiosim.io.config_resolution import (
            ConfigResolutionError,
            InstrumentSourcePathOverride,
            SimulationOverrides,
            WorkflowOverrides,
        )

        antenna_override = (
            InstrumentSourcePathOverride(path=Path(antenna_file))
            if antenna_file is not None
            else None
        )
        bundle = load_config(
            config_path,
            overrides=SimulationOverrides(
                backend=backend,
                offline=offline,
                instrument_source=antenna_override,
            ),
            workflow_overrides=WorkflowOverrides(
                output_dir=None if sim_data_dir is None else Path(sim_data_dir),
            ),
            check_input_paths=True,
        )
    except ConfigResolutionError as error:
        from radiosim.cli.workflow import render_configuration_error

        render_configuration_error(error, command="config mode")
        return 1
    except Exception as error:
        logger.error(f"Configuration setup failed: {error}")
        if verbose >= 2:
            import traceback

            traceback.print_exc()
        return 1

    from radiosim.api.simulator import Simulator

    simulator = Simulator(bundle.runtime)
    try:
        simulator.run()
    except Exception as error:
        logger.error(f"Simulation failed: {error}")
        if verbose >= 2:
            import traceback

            traceback.print_exc()
        return 1

    try:
        from radiosim.cli.workflow import run_cli_workflow

        saved_files = run_cli_workflow(
            simulator,
            bundle.workflow,
            runtime=bundle.runtime,
            provenance=bundle.provenance,
            verbose=verbose,
        )
    except Exception as error:
        logger.error(f"CLI workflow failed: {error}")
        if verbose >= 2:
            import traceback

            traceback.print_exc()
        return 1

    for path in saved_files:
        print_info(f"Output: {path}")
    print_success("Done.")
    return 0


def run_simulate_mode(
    antenna_layout: str,
    telescope_name: str,
    default_diameter_m: float | None,
    correlations: Literal["all", "cross", "auto"],
    frequencies: str,
    sky_model: str,
    output: str,
    output_format: str,
    backend: BackendStrategy,
    latitude: float,
    longitude: float,
    height: float,
    start_time: str,
    duration_seconds: float = 1.0,
    time_step_seconds: float = 1.0,
) -> int:
    """Resolve typed scientific parameters, run, and save explicitly."""
    from radiosim.utils.logging import get_logger, setup_logging

    setup_logging()
    logger = get_logger("radiosim.cli")

    try:
        frequency_mhz = [float(value.strip()) for value in frequencies.split(",")]
    except ValueError:
        logger.error(f"Invalid frequencies format: {frequencies}")
        logger.info("Expected comma-separated numbers, e.g., '100,150,200'")
        return 1
    channel_frequencies_hz = tuple(value * 1e6 for value in frequency_mhz)

    logger.info(f"Running simulation with {len(frequency_mhz)} frequencies")
    logger.info(f"Antenna layout: {antenna_layout}")
    logger.info(f"Sky model: {sky_model}")

    try:
        from pydantic import ValidationError

        from radiosim.api.simulator import Simulator
        from radiosim.core.instrument_resolution import DiameterResolutionError
        from radiosim.core.sky.registry import loader_registry
        from radiosim.io.config import (
            ExecutionConfig,
            PrecisionInput,
            SkyModelConfig,
            VisibilityConfig,
            schema_issues_from_validation_error,
        )
        from radiosim.io.config_resolution import (
            ConfigResolutionError,
            ConfigSchemaError,
        )
        from radiosim.io.instrument_config import (
            BaselineSelectionConfig,
            InstrumentConfig,
            InstrumentLocationConfig,
            LayoutFileSourceConfig,
        )

        loader_name, defaults = loader_registry.resolve_request(sky_model, {})
        meta = loader_registry.meta(sky_model)
        try:
            instrument = InstrumentConfig(
                source=LayoutFileSourceConfig(
                    path=antenna_layout,
                    format="radiosim",
                    telescope_name=telescope_name,
                ),
                location=InstrumentLocationConfig(
                    latitude_deg=latitude,
                    longitude_deg=longitude,
                    height_m=height,
                ),
                default_diameter_m=default_diameter_m,
            )
            baseline_selection = BaselineSelectionConfig(correlations=correlations)
            sky_input = SkyModelConfig.model_validate(
                {
                    "flux_unit": "Jy",
                    "sources": [{"kind": loader_name, **defaults}],
                }
            )
            visibility = VisibilityConfig(
                sky_representation=cast(
                    Literal["point_sources", "healpix_map"],
                    "healpix_map"
                    if meta["representations"][0] == "healpix_map"
                    else "point_sources",
                )
            )
            execution = ExecutionConfig(
                backend=backend,
                precision=PrecisionInput(),
            )
        except ValidationError as error:
            raise ConfigSchemaError(
                schema_issues_from_validation_error(error)
            ) from error

        simulator = Simulator.from_parameters(
            instrument=instrument,
            baseline_selection=baseline_selection,
            channel_frequencies_hz=channel_frequencies_hz,
            start_time=start_time,
            duration_seconds=duration_seconds,
            time_step_seconds=time_step_seconds,
            sky_model=sky_input,
            visibility=visibility,
            execution=execution,
        )

        simulator.run()
        simulator.save(output, format=output_format, overwrite=False)

        logger.info(f"Results saved to: {output}")
        return 0

    except ConfigResolutionError as error:
        from radiosim.cli.workflow import render_configuration_error

        render_configuration_error(error, command="simulate")
        return 1
    except DiameterResolutionError as error:
        logger.error(
            f"Simulation failed: {error}. Add a positive Diameter column or pass "
            "--default-diameter-m."
        )
        return 1
    except Exception as error:
        logger.error(f"Simulation failed: {error}")
        return 1


def run_init_mode(output: str) -> int:
    """Create default configuration file."""
    from radiosim.io.config import create_default_config

    output_path = Path(output)
    if output_path.exists():
        print(f"File already exists: {output_path}")
        response = input("Overwrite? [y/N] ")
        if response.lower() != "y":
            print("Aborted.")
            return 1

    create_default_config(output_path)
    print(f"Created default configuration: {output_path}")
    return 0


def run_validate_mode(config: str) -> int:
    """Resolve a document and summarize it without constructing runtime."""
    from radiosim.io.config import load_config
    from radiosim.io.config_resolution import ConfigResolutionError

    try:
        bundle = load_config(config, check_input_paths=True)
    except ConfigResolutionError as error:
        from radiosim.cli.workflow import render_configuration_error

        render_configuration_error(error, command="validate")
        return 1

    from radiosim.cli.workflow import render_resolved_summary

    render_resolved_summary(bundle)
    return 0


def run_status_mode(verbose: int = 0) -> int:
    """Check and display network connectivity status."""
    from rich.panel import Panel
    from rich.table import Table

    from radiosim.utils.logging import console, setup_logging
    from radiosim.utils.network import (
        SERVICE_DISPLAY_NAMES,
        SERVICE_ENDPOINTS,
        check_all_services,
    )

    setup_logging(level=logging.DEBUG if verbose >= 2 else logging.INFO)

    console.print()
    with console.status("Checking network connectivity..."):
        net_status = check_all_services()

    table = Table(show_header=True, header_style="bold cyan", padding=(0, 1))
    table.add_column("Service", style="white")
    table.add_column("Endpoint", style="dim")
    table.add_column("Status")

    # General internet row
    if net_status.internet:
        inet_status = "[bold green]Online[/bold green]"
    else:
        inet_status = "[bold red]Offline[/bold red]"
    table.add_row("Internet", "8.8.8.8:53", inet_status)

    # Per-service rows
    for name, (host, _port) in SERVICE_ENDPOINTS.items():
        available = net_status.service_available(name)
        display = SERVICE_DISPLAY_NAMES.get(name, name)
        if available is True:
            svc_status = "[green]Reachable[/green]"
        elif available is False:
            svc_status = "[red]Unreachable[/red]"
        else:
            svc_status = "[dim]Not checked[/dim]"
        table.add_row(display, host, svc_status)

    console.print(
        Panel(
            table,
            title="[bold]RadioSim Network Status[/bold]",
            border_style="cyan",
        )
    )

    # Device resources panel
    from radiosim.utils.device import get_device_resources

    with console.status("Detecting device resources..."):
        dev = get_device_resources()

    dev_table = Table(show_header=True, header_style="bold cyan", padding=(0, 1))
    dev_table.add_column("Resource", style="white")
    dev_table.add_column("Value")

    # OS
    dev_table.add_row(
        "OS",
        f"{dev.os_info.name} {dev.os_info.version} "
        f"({dev.os_info.architecture}, {dev.os_info.bits})",
    )

    # CPU
    cores_str = ""
    if dev.cpu.physical_cores is not None:
        cores_str = (
            f"{dev.cpu.physical_cores} physical / {dev.cpu.logical_cores} logical"
        )
    else:
        cores_str = f"{dev.cpu.logical_cores} logical"
    cpu_label = dev.cpu.model or dev.cpu.architecture
    dev_table.add_row("CPU", f"{cpu_label} ({dev.cpu.architecture}, {cores_str})")

    # RAM
    if dev.memory.total_gb is not None:
        ram_parts = [f"{dev.memory.total_gb:.2f} GB total"]
        if dev.memory.available_gb is not None:
            ram_parts.append(f"{dev.memory.available_gb:.2f} GB available")
        dev_table.add_row("RAM", ", ".join(ram_parts))
    else:
        dev_table.add_row("RAM", "[dim]Unknown (install psutil for detection)[/dim]")

    # Storage
    dev_table.add_row(
        "Storage",
        f"{dev.storage.free_gb:.2f} GB free / {dev.storage.total_gb:.2f} GB total",
    )

    # GPU(s)
    if dev.gpus:
        for i, gpu in enumerate(dev.gpus):
            label = "GPU" if i == 0 else ""
            parts = [gpu.name or gpu.vendor]
            details: list[str] = []
            if gpu.vram_total_gb is not None:
                details.append(f"{gpu.vram_total_gb:.1f} GB VRAM")
            if gpu.cores is not None:
                details.append(f"{gpu.cores} cores")
            if gpu.metal_support:
                details.append(gpu.metal_support)
            if gpu.cuda_driver:
                details.append(f"Driver {gpu.cuda_driver}")
            if details:
                parts.append(f"({', '.join(details)})")
            dev_table.add_row(label, " ".join(parts))
    else:
        dev_table.add_row("GPU", "[dim]None detected[/dim]")

    console.print(
        Panel(
            dev_table,
            title="[bold]RadioSim Device Resources[/bold]",
            border_style="cyan",
        )
    )
    console.print()
    return 0


def main() -> None:
    """Main entry point for the CLI."""
    cli()


if __name__ == "__main__":
    main()
