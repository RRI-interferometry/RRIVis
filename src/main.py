# src/main.py

# Standard Library imports
import os
import tempfile
import time
from datetime import datetime, timezone
import sys
import logging
import argparse
from pathlib import Path

# Third-party imports
import yaml
from bokeh.io import show, reset_output, output_file
import numpy as np
from astropy.constants import c
from astropy.time import TimeDelta
import matplotlib.pyplot as plt
import h5py
import shutil

# Local module imports
from beams import (
    calculate_hpbw_radians,
    calculate_gaussian_beam_area_EBeam,
)
from antenna import read_antenna_positions
from baseline import generate_baselines
from source import get_sources
from observation import get_location_and_time
from visibility import (
    calculate_visibility,
    calculate_modulus_phase,
)
from gsm_map import diffused_sky_model
from plot import (
    plot_visibility,
    plot_heatmaps,
    plot_modulus_vs_frequency,
    plot_antenna_layout,
)


"""
These are the default config values that will be used if the user does not
provide a config file or the config file is missing some parameters.
"""
DEFAULT_CONFIG = {
    # If PYUVData telescope is used, then all the possible inputs will be
    # used from the PYUVData telescope object. And if not present there,
    # will be taken from the antenna, baseline, location,
    # beams parameters.
    # TODO: Add support for using Telescope objects and custom telescopes parameters.
    "telescope": {
        "telescope_type": {
            "use_PYUVData_telescope": False,
            "use_custom_telescope": False,
        },
        # Known telescope names: ATA, HERA, MWA, OVRO-LWA, PAPER, SMA, SZA
        "telescope_name": "Unknown",
        # Control which fields to import from pyuvdata Telescope when enabled
        # If False, values from config/CLI are used instead.
        "use_pyuvdata_location": False,
        "use_pyuvdata_antennas": False,
        "use_pyuvdata_diameters": False,
    },
    "antenna": {
        # User must provide an absolute path via --antenna-file or config.
        "antenna_positions_file": None,
        "all_antenna_diameter": "",
        "use_different_diameters": False,
        "diameters": {},
        "fixed_hpbw": None,
        # Antenna optical/mechanical type (e.g., parabolic, spherical, phased-array, dipole, other)
        # By default a single type applies to all antennas. Optionally allow per-antenna overrides.
        "physical_antenna_type": "",  # Alternative for aperture
        "use_different_antenna_types": False,
        "types_per_antenna": {},  # {antenna_number: "parabolic"}
        # Illumination taper of the aperture (affects beam sidelobes in advanced models)
        # Common options: uniform, cosine, gaussian, taper10dB, taper20dB, custom
        "illumination": "",  # Alternative for feed type
        "use_different_illumination": False,
        "illumination_per_antenna": {},  # {antenna_number: "uniform"}
    },
    "feeds": {
        # TODO: To be added later
        "use_polarized_feeds": False,
        "use_different_feed_types": False,
        "all_same_feed_type": "dual",
        "feed_types_per_antenna": {},
    },
    "beams": {
        # Either EBeam or PowerBeam
        "use_different_beams_for_each_antenna": False,
        "all_same_beam_type": "EBeam",
        "beams_per_antenna": {},
        # TODO: Add support for PYUVData analytic beams and other beam types and mixing beam types.
        "use_different_beam_responses_for_each_antenna": False,
        # TODO: Add support these options: analytic_beams(Gaussian, Airy, etc.), healpix_analytic_beams, PYUVData_analytic_beams, UVBeam, Beam_interface etc.
        "all_same_beam_response": "GaussianHealpix",
        "beam_response_per_antenna": {},
    },
    "baseline": {
        "use_autocorrelations": True,
        "use_crosscorrelations": True,
        "only_selective_baseline_length": False,
        "selective_baseline_lengths": [],
        # TODO: Add support for inputting angle for the unit vector of the baseline for trimming!
    },
    "location": {
        "lat": "",
        "lon": "",
        "height": "",
    },
    "visibility": {
        # TODO: Polarization support will be added later.
        # TODO: Add support for faster visibility calculations.
    },
    "sky_model": {
        "test_sources": {
            "use_test_sources": False,
            "num_sources": 100,
            "flux_limit": 50,
        },
        "test_sources_healpix": {
            "use_test_sources": False,
            "num_sources": 100,
            "flux_limit": 50,
            "nside": 32,
        },
        # pyGSM has the following available catalogues:
        # GlobalSkyModel2016 (use gsm2016 to select this)
        # GlobalSkyModel (use gsm2008 to select this)
        "gsm_healpix": {
            "use_gsm": False,
            "gsm_catalogue": "gsm2008",
            "flux_limit": 50,
            "nside": 32,
        },
        # GLEAM has the following available catalogues:
        # VIII/100/table1 → GLEAM first year observing parameters (28 rows)
        # VIII/100/gleamegc → GLEAM EGC catalog, version 2 (307455 rows)
        # VIII/102/gleamgal → GLEAM Galactic plane catalog (22037 rows)
        # VIII/105/catalog → G4Jy catalogue (18/01/2020) (1960 rows)
        # VIII/109/gleamsgp → GLEAM SGP catalogue (108851 rows)
        # VIII/110/catalog → First data release of GLEAM-X (78967 rows)
        # VIII/113/catalog2 → Second data release of GLEAM-X (624866 rows)
        "gleam": {
            "use_gleam": False,
            "gleam_catalogue": "VIII/105/catalog",  # G4Jy (smallest set)
            "flux_limit": 50,
        },
        "gleam_healpix": {
            "use_gleam": False,
            "gleam_catalogue": "VIII/100/gleamegc",
            "flux_limit": 50,
            "nside": 32,
        },
        "gsm+gleam_healpix": {
            "use_gsm_gleam": False,
            "gsm_catalogue": "gsm2008",
            "gleam_catalogue": "VIII/100/gleamegc",
            "flux_limit": 50,
            "nside": 32,
        },
    },
    "obs_time": {
        "time_interval": 1.0,
        "time_interval_unit": "hours",
        "total_duration": 1,
        "total_duration_unit": "days",
        "start_time": "2025-01-01T00:00:00",
    },
    "frequency": {
        "starting_frequency": 50.0,
        "frequency_interval": 1.0,
        "frequency_bandwidth": 100.0,
        "frequency_unit": "MHz",
    },
    # Data output formats for visibility data:
    # HDF5 file format
    # CASA Measurement Set (MS)
    # UVFITS
    "output": {
        "output_file_name": "complex_visibility",
        "output_file_format": "HDF5",
        "save_simulation_data": False,
        "plot_results_in_bokeh": True,
        "save_log_data": False,
        # Angle unit for display output - must be specified by user
        # Valid values: "degrees" or "radians"
        "angle_unit": "",
    },
    # Use different simulators for cross-checking the results
    # Available simulators: fftvis, matvis, pyuvsim, healvis
    "simulators": {
        "use_different_simulator_for_cross_check": False,
        "name": "",
    },
    # Sky model plotting options
    "plot_skymodel_every_hour": True,  # Enable sky model plots by default
    "skymodel_frequency": 150,  # Frequency in MHz for sky model plots
    "fov_radius_deg": 10,  # Field of view radius in degrees for sky model plots
}


def load_config(yaml_file):
    """
    Load configuration from YAML and merge with default values.
    """
    with open(yaml_file, "r") as CONFIG:
        user_config = yaml.safe_load(CONFIG) or {}
    return merge_config(DEFAULT_CONFIG, user_config), user_config


# Note: relative paths in user configs are not supported; expect absolute paths.


def merge_config(default, user):
    """
    Recursively merge user configuration with default configuration.
    Ensures all default values are retained unless explicitly overridden.
    """
    merged = default.copy()
    for key, value in user.items():
        if isinstance(value, dict) and key in merged and isinstance(merged[key], dict):
            merged[key] = merge_config(merged[key], value)
        else:
            merged[key] = value
    return merged


def create_simulation_folder(config):
    """
    Create a simulation folder in a user-specified or default location
    based on the current UTC date and time.

    Precedence for base directory (highest to lowest):
      1) config['output']['simulation_data_dir'] (or --sim-data-dir)
      2) $XDG_DOWNLOAD_DIR if set and exists
      3) $HOME/Downloads
    Subfolder name under the base directory can be customized via
      config['output']['simulation_subdir'] (defaults to 'RRIVis_simulation_data').
    """
    if config["output"]["save_simulation_data"]:
        # 1) Use explicit directory from config if provided
        explicit_dir = config.get("output", {}).get("simulation_data_dir")

        if explicit_dir:
            base_path = os.path.abspath(os.path.expanduser(explicit_dir))
            print(f"Using simulation data base directory from config/CLI: {base_path}")
        else:
            # 2) Check XDG_DOWNLOAD_DIR environment variable
            xdg_download_dir = os.environ.get("XDG_DOWNLOAD_DIR")
            simulation_subdir = config.get("output", {}).get(
                "simulation_subdir", "RRIVis_simulation_data"
            )
            if xdg_download_dir and os.path.isdir(os.path.expanduser(xdg_download_dir)):
                base_path = os.path.join(
                    os.path.expanduser(xdg_download_dir), simulation_subdir
                )
                print(
                    f"Using $XDG_DOWNLOAD_DIR as base: {os.path.expanduser(xdg_download_dir)}"
                )
            else:
                # 3) Default to the user's $HOME/Downloads
                downloads_dir = Path.home() / "Downloads"
                base_path = str(downloads_dir / simulation_subdir)
                if xdg_download_dir:
                    print(
                        f"$XDG_DOWNLOAD_DIR not found or invalid → Defaulting to: {downloads_dir}"
                    )
                else:
                    print(f"No $XDG_DOWNLOAD_DIR set → Defaulting to: {downloads_dir}")

        # Use timezone-aware datetime in UTC with human-friendly separators
        # Example: 2025-09-03_18-15-36_UTC
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S") + "_UTC"
        folder_path = os.path.join(base_path, timestamp)
        try:
            os.makedirs(folder_path, exist_ok=True)
            print(f"Simulation data will be saved under: {folder_path}")
            return folder_path
        except PermissionError:
            print(f"Error: Insufficient permissions to create folder '{folder_path}'.")
        except OSError as e:
            print(f"Error: Failed to create folder '{folder_path}' due to: {e}")
    return None


def _defaults_used(default: dict, user: dict) -> dict:
    """
    Return a dictionary of default values that are still in effect
    (i.e., keys not provided by the user). For nested dicts, compute recursively.
    """
    result = {}
    for key, def_val in default.items():
        if key not in user:
            result[key] = def_val
        else:
            user_val = user[key]
            if isinstance(def_val, dict) and isinstance(user_val, dict):
                nested = _defaults_used(def_val, user_val)
                if nested:
                    result[key] = nested
    return result


def save_yaml_config(config, folder_path):
    """
    Save the final YAML configuration file in the simulation folder with proper spacing.
    """
    yaml_file_path = os.path.join(folder_path, "config.yaml")

    # Customizing the YAML dumper for better readability
    class MyDumper(yaml.Dumper):
        def write_line_break(self, data=None):
            # Add a blank line between top-level sections
            if self.indent == 0:
                self.stream.write("\n")
            super().write_line_break(data)

    with open(yaml_file_path, "w") as f:
        yaml.dump(config, f, Dumper=MyDumper, default_flow_style=False, sort_keys=True)
    print(f"Configuration saved to {yaml_file_path}")


def init_console_logging():
    """
    Ensure a console logger exists and register a custom SUCCESS level.
    Does not create any file handlers. Safe to call multiple times.
    """
    SUCCESS_LEVEL_NUM = 25
    if not hasattr(logging, "SUCCESS"):
        logging.addLevelName(SUCCESS_LEVEL_NUM, "SUCCESS")

        def success(self, message, *args, **kws):
            if self.isEnabledFor(SUCCESS_LEVEL_NUM):
                self._log(SUCCESS_LEVEL_NUM, message, args, **kws)

        logging.Logger.success = success  # type: ignore[attr-defined]

    logger = logging.getLogger()
    if not logger.handlers:
        logger.setLevel(logging.INFO)

        console_stream = sys.__stdout__
        console_handler = logging.StreamHandler(console_stream)
        console_handler.setLevel(logging.INFO)

        class ColorFormatter(logging.Formatter):
            RESET = "\033[0m"
            COLORS = {
                logging.INFO: "\033[37m",  # white
                logging.WARNING: "\033[33m",  # yellow
                logging.ERROR: "\033[31m",  # red
                logging.CRITICAL: "\033[31m",  # red
                SUCCESS_LEVEL_NUM: "\033[32m",  # green
            }

            def format(self, record):
                msg = super().format(record)
                try:
                    is_tty = (
                        hasattr(sys.__stdout__, "isatty") and sys.__stdout__.isatty()
                    )
                except Exception:
                    is_tty = False
                if is_tty:
                    color = self.COLORS.get(record.levelno, "")
                    if color:
                        return f"{color}{msg}{self.RESET}"
                return msg

        console_formatter = ColorFormatter("%(message)s")
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

    return logger


def cleanup_old_temp_outputs(prefix: str = "rrivis_", retention_seconds: float = 300.0):
    """
    Remove stale temporary files/directories created by this app in the system temp folder.

    - Scans `tempfile.gettempdir()` for entries starting with `prefix`.
    - Deletes those older than `retention_seconds` based on modification time.
    - Ignores errors to avoid impacting normal startup.
    """
    try:
        temp_root = tempfile.gettempdir()
        now = time.time()
        cutoff = now - retention_seconds
        for name in os.listdir(temp_root):
            if not name.startswith(prefix):
                continue
            path = os.path.join(temp_root, name)
            try:
                mtime = os.path.getmtime(path)
            except OSError:
                continue
            if mtime >= cutoff:
                continue
            try:
                if os.path.isdir(path):
                    shutil.rmtree(path, ignore_errors=True)
                else:
                    os.remove(path)
                logging.info(f"Cleaned old temp output: {path}")
            except Exception:
                # Ignore cleanup failures
                pass
    except Exception:
        # Never fail startup due to cleanup
        pass


def setup_logging(simulation_folder_path):
    """
    Set up logging to log both to a file and the console.
    """
    log_file = os.path.join(simulation_folder_path, "simulation.log")

    # Define a custom SUCCESS level for green completion messages
    SUCCESS_LEVEL_NUM = 25
    if not hasattr(logging, "SUCCESS"):
        logging.addLevelName(SUCCESS_LEVEL_NUM, "SUCCESS")

        def success(self, message, *args, **kws):
            if self.isEnabledFor(SUCCESS_LEVEL_NUM):
                self._log(SUCCESS_LEVEL_NUM, message, args, **kws)

        logging.Logger.success = success  # type: ignore[attr-defined]

    # Build separate console and file handlers with different formats for readability
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Clear existing handlers to avoid duplicate logs on repeated runs
    for h in list(logger.handlers):
        logger.removeHandler(h)

    # File handler with timestamps
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(file_formatter)

    # Console handler with cleaner format (no timestamps) and colors by level
    # IMPORTANT: use sys.__stdout__ to avoid recursion when we later redirect sys.stdout
    console_stream = sys.__stdout__
    console_handler = logging.StreamHandler(console_stream)
    console_handler.setLevel(logging.INFO)

    class ColorFormatter(logging.Formatter):
        RESET = "\033[0m"
        COLORS = {
            logging.INFO: "\033[37m",  # white
            logging.WARNING: "\033[33m",  # yellow
            logging.ERROR: "\033[31m",  # red
            logging.CRITICAL: "\033[31m",  # red
            SUCCESS_LEVEL_NUM: "\033[32m",  # green
        }

        def format(self, record):
            msg = super().format(record)
            # Only colorize if original stdout is a TTY
            try:
                is_tty = hasattr(sys.__stdout__, "isatty") and sys.__stdout__.isatty()
            except Exception:
                is_tty = False
            if is_tty:
                color = self.COLORS.get(record.levelno, "")
                if color:
                    return f"{color}{msg}{self.RESET}"
            return msg

    console_formatter = ColorFormatter("%(message)s")
    console_handler.setFormatter(console_formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return log_file


def section(title: str):
    """Pretty section separator with special colors for start/end headings."""
    line = "." * 80
    print("")
    try:
        # Choose color: default magenta, green for STARTING, green for ENDING
        color = _BOLD + _MAGENTA
        t_upper = title.upper()
        if "STARTING SIMULATION" in t_upper:
            color = _BOLD + _GREEN
        elif "ENDING SIMULATION" in t_upper:
            color = _BOLD + _GREEN

        print(_c(line, _DIM))
        print(_c(title, color))
        print(_c(line, _DIM))
    except Exception:
        # Fallback to plain text if color helpers are unavailable
        print(line)
        print(title)
        print(line)
    print("")


class LoggerWriter:
    """
    Redirect `print` statements to the logger.
    """

    def __init__(self, level):
        self.level = level
        self.buffer = ""

    def write(self, message):
        if message != "\n":  # Avoid empty lines
            self.level(message.strip())

    def flush(self):
        pass


# --- Pretty console helpers (minimalistic, TTY-aware colors) ---
def _supports_color() -> bool:
    try:
        return hasattr(sys.__stdout__, "isatty") and sys.__stdout__.isatty()
    except Exception:
        return False


_TTY = _supports_color()
_RESET = "\033[0m"
_BOLD = "\033[1m"
_DIM = "\033[2m"
_CYAN = "\033[36m"
_GREEN = "\033[32m"
_YELLOW = "\033[33m"
_MAGENTA = "\033[35m"
_WHITE = "\033[37m"
_LIGHT_BLUE = "\033[94m"
_DARK_GREY = "\033[90m"


def _c(text: str, style: str) -> str:
    if not _TTY:
        return text
    return f"{style}{text}{_RESET}"


def _indent(text: str, spaces: int = 2) -> str:
    if not text:
        return ""
    pad = " " * spaces
    return "\n".join(pad + line for line in text.rstrip().splitlines())


# --- Colored nested config pretty-printer ---
# Level 0 (top): cyan, Level 1: yellow, Level 2: light blue; repeat beyond.
_LEVEL_STYLES = [
    _BOLD + _CYAN,
    _BOLD + _YELLOW,
    _BOLD + _LIGHT_BLUE,
]


def _level_style(level: int) -> str:
    return _LEVEL_STYLES[level % len(_LEVEL_STYLES)]


def _fmt_scalar(val):
    if isinstance(val, str):
        return repr(val)
    return str(val)


def _config_lines(data, level: int = 0, indent: int = 2):
    lines = []
    pad = " " * (level * indent)
    if isinstance(data, dict):
        for key in sorted(data.keys(), key=lambda k: str(k)):
            ksty = _level_style(level)
            key_txt = _c(f"{key}:", ksty)
            val = data[key]
            is_leaf_container = (isinstance(val, dict) and not val) or (
                isinstance(val, list) and not val
            )
            if isinstance(val, (dict, list)) and not is_leaf_container:
                # Non-leaf container: print key in level color and recurse
                lines.append(pad + key_txt)
                lines.extend(_config_lines(val, level + 1, indent))
            else:
                # Leaf (scalar or empty container): key in darker grey, value (if scalar) normal
                leaf_key_txt = _c(f"{key}:", _DARK_GREY)
                if isinstance(val, (dict, list)):
                    # Empty dict/list
                    lines.append(pad + leaf_key_txt)
                else:
                    lines.append(pad + leaf_key_txt + " " + _fmt_scalar(val))
    elif isinstance(data, list):
        for item in data:
            bullet = pad + "- "
            if isinstance(item, (dict, list)):
                lines.append(bullet)
                lines.extend(_config_lines(item, level + 1, indent))
            else:
                lines.append(bullet + _fmt_scalar(item))
    else:
        lines.append(pad + _fmt_scalar(data))
    return lines


def print_colored_config(data, base_indent: int = 2):
    for line in _config_lines(data, level=0, indent=base_indent):
        print(line)


def _parse_positive_float(value, field_name: str) -> float:
    """Parse a value into a positive finite float with clear errors.

    Accepts ints/floats and numeric strings (ignoring surrounding whitespace).
    Rejects empty strings, non-numeric strings, non-finite, and non-positive values.
    """
    if isinstance(value, (int, float)):
        f = float(value)
    elif isinstance(value, str):
        s = value.strip()
        if not s:
            raise ValueError(f"{field_name} is empty; provide a positive number.")
        try:
            f = float(s)
        except ValueError:
            raise ValueError(f"{field_name} must be a positive number; got '{value}'.")
    else:
        raise ValueError(
            f"{field_name} must be a number or numeric string, got {type(value).__name__}."
        )

    if not np.isfinite(f) or f <= 0:
        raise ValueError(f"{field_name} must be a positive, finite number; got {f}.")
    return f


def _validate_angle_unit(config):
    """Validate that angle_unit is properly configured.

    The angle_unit must be specified by the user and must be either 'degrees' or 'radians'.
    """
    angle_unit = config.get("output", {}).get("angle_unit", "")

    # Check if angle_unit is empty (not specified by user)
    if not angle_unit:
        raise ValueError(
            "output.angle_unit must be specified in the configuration. "
            "Valid values are 'degrees' or 'radians'."
        )

    # Check if angle_unit is valid
    if angle_unit not in ["degrees", "radians"]:
        raise ValueError(
            f"Invalid output.angle_unit: '{angle_unit}'. "
            "Valid values are 'degrees' or 'radians'."
        )

    return angle_unit


def _convert_angle_for_display(angle_radians, angle_unit):
    """Convert angle from radians to the user's preferred unit.

    Args:
        angle_radians: Angle value in radians
        angle_unit: Target unit ('degrees' or 'radians')

    Returns:
        Angle value in the specified unit
    """
    if angle_unit == "degrees":
        return np.degrees(angle_radians)
    return angle_radians  # Already in radians


def main():
    """
    Main function to run the Visibility Simulation.
    """

    # Parse CLI arguments
    parser = argparse.ArgumentParser(
        description="RRIviz - Radio Astronomy Visibility Simulator"
    )
    parser.add_argument(
        "-c",
        "--config",
        default=None,
        help=(
            "Absolute path to a configuration YAML file. "
            "If omitted, built-in defaults are used."
        ),
    )
    parser.add_argument(
        "--sim-data-dir",
        dest="sim_data_dir",
        default=None,
        help=(
            "Directory to store simulation data (overrides config output.simulation_data_dir). "
            "If not provided, defaults to the user's Downloads folder."
        ),
    )
    parser.add_argument(
        "--antenna-file",
        dest="antenna_file",
        default=None,
        help=(
            "Absolute path to antenna positions file (CSV/TXT). Overrides config antenna.antenna_positions_file."
        ),
    )
    args = parser.parse_args()

    # Resolve configuration:
    # - If --config is provided: require an absolute path and merge with defaults.
    # - If not provided: use DEFAULT_CONFIG as-is (with internal defaults).
    user_config_raw = {}
    if args.config:
        if not os.path.isabs(args.config):
            raise ValueError(
                "--config must be an absolute path; relative paths are not supported."
            )
        if not os.path.exists(args.config):
            raise FileNotFoundError(f"Configuration file not found: {args.config}")
        config, user_config_raw = load_config(args.config)
    else:
        config = DEFAULT_CONFIG.copy()

    # CLI override for simulation data directory
    if args.sim_data_dir:
        sim_dir_abs = os.path.abspath(os.path.expanduser(args.sim_data_dir))
        config.setdefault("output", {})["simulation_data_dir"] = sim_dir_abs

    # Determine antenna positions file path (if provided): CLI overrides config
    configured_ant_path = args.__dict__.get("antenna_file") or config.get(
        "antenna", {}
    ).get("antenna_positions_file")
    if configured_ant_path:
        configured_ant_path = os.path.abspath(os.path.expanduser(configured_ant_path))
        if not os.path.isabs(configured_ant_path):
            raise ValueError("antenna.antenna_positions_file must be an absolute path.")
        if not os.path.exists(configured_ant_path):
            raise FileNotFoundError(
                f"Antenna positions file not found: {configured_ant_path}"
            )

    # Validate angle unit configuration
    angle_unit = _validate_angle_unit(config)

    # Print a clear configuration summary to console (add leading spacing)
    print("")
    print("")
    section("CONFIGURATION")
    if args.config:
        print(
            f"{_c('Config file:', _CYAN)} {args.config} {_c('(unspecified fields use in-code defaults)', _DIM)}"
        )
    else:
        print(
            f"{_c('Config file:', _CYAN)} none {_c('→ using in-code defaults', _DIM)}"
        )
    if configured_ant_path:
        source_str = "CLI" if args.__dict__.get("antenna_file") else "config"
        print(f"{_c(f'Antenna file ({source_str}):', _CYAN)} {configured_ant_path}")
    else:
        print(
            f"{_c('Antenna file:', _CYAN)} (not provided; may use pyuvdata Telescope)"
        )

    # Print a clear, readable breakdown of user-provided vs defaulted configuration
    print("")
    print(_c("User-provided overrides (from config.yaml):", _BOLD + _GREEN))
    if user_config_raw:
        print_colored_config(user_config_raw, base_indent=2)
    else:
        print(_indent("(none)", 2))

    defaults_active = _defaults_used(DEFAULT_CONFIG, user_config_raw)
    print("")
    print(_c("Defaults in effect (from in-code defaults):", _BOLD + _GREEN))
    if defaults_active:
        # Pretty-print nested defaults with level-based colors
        print_colored_config(defaults_active, base_indent=2)
    else:
        print(_indent("(none)", 2))

    # Ensure console logging with SUCCESS level, even when not saving to disk
    init_console_logging()
    # Visual separation between config dump and maintenance logs
    print("")
    cleanup_old_temp_outputs(prefix="rrivis_", retention_seconds=300.0)

    # Determine whether to save data and initialize folder path
    save_simulation_data_flag = bool(
        config.get("output", {}).get("save_simulation_data", True)
    )
    simulation_folder_path = None

    if save_simulation_data_flag:
        simulation_folder_path = create_simulation_folder(config)
        if simulation_folder_path:
            save_yaml_config(config, simulation_folder_path)
            log_file = setup_logging(simulation_folder_path)
            print(f"Logging to {log_file}")

            sys.stdout = LoggerWriter(logging.info)
            sys.stderr = LoggerWriter(logging.error)

    section("STARTING SIMULATION")

    # Optionally load telescope metadata from pyuvdata and use available fields
    antennas = None
    used_from_tel = {}
    diameters_from_tel = None  # None, scalar float, or dict{antenna_number: diameter}
    if (
        config.get("telescope", {})
        .get("telescope_type", {})
        .get("use_PYUVData_telescope", False)
    ):
        tel_name = config.get("telescope", {}).get("telescope_name")
        if tel_name:
            try:
                from pyuvdata import Telescope as PyuvTelescope

                tel = PyuvTelescope.from_known_telescopes(
                    name=tel_name, run_check=False
                )
                # Attempt to fill missing params from known database without overwriting user values
                tel.update_params_from_known_telescopes(
                    overwrite=False, run_check=False
                )

                section("TELESCOPE (pyuvdata)")
                kv = lambda label, msg: print(f"{_c(label + ':', _BOLD + _CYAN)} {msg}")
                kv("Loaded known telescope", tel_name)
                use_tel_location = bool(
                    config.get("telescope", {}).get("use_pyuvdata_location", False)
                )
                use_tel_antennas = bool(
                    config.get("telescope", {}).get("use_pyuvdata_antennas", False)
                )
                use_tel_diameters = bool(
                    config.get("telescope", {}).get("use_pyuvdata_diameters", False)
                )
                print(
                    f"{_c('Options →', _BOLD + _CYAN)} use_pyuvdata_location={use_tel_location}, use_pyuvdata_antennas={use_tel_antennas}, use_pyuvdata_diameters={use_tel_diameters}"
                )

                # Location
                if getattr(tel, "location", None) is not None and use_tel_location:
                    try:
                        geodetic = tel.location.to_geodetic()
                        lat_deg = float(geodetic.lat.deg)
                        lon_deg = float(geodetic.lon.deg)
                        height_m = float(geodetic.height.to_value())
                        used_from_tel["location"] = True
                        # Override config location
                        config.setdefault("location", {})
                        config["location"]["lat"] = lat_deg
                        config["location"]["lon"] = lon_deg
                        config["location"]["height"] = height_m
                        kv(
                            "Location",
                            f"from pyuvdata (lat={lat_deg}, lon={lon_deg}, h={height_m} m)",
                        )
                    except Exception:
                        kv(
                            "Location",
                            "present on Telescope but could not convert to geodetic; keeping config/defaults.",
                        )
                        used_from_tel["location"] = False
                else:
                    if getattr(tel, "location", None) is None:
                        kv("Location", "not present; using config/defaults.")
                    else:
                        kv(
                            "Location",
                            "present on Telescope but not used (use_pyuvdata_location=False); using config/defaults.",
                        )
                    used_from_tel["location"] = False

                # Helper: convert ITRF deltas to ENU at lat/lon
                def _ecef_deltas_to_enu(lat_rad, lon_rad, dxyz):
                    slat, clat = np.sin(lat_rad), np.cos(lat_rad)
                    slon, clon = np.sin(lon_rad), np.cos(lon_rad)
                    x = dxyz[:, 0]
                    y = dxyz[:, 1]
                    z = dxyz[:, 2]
                    e = -slon * x + clon * y
                    n = -slat * clon * x - slat * slon * y + clat * z
                    u = clat * clon * x + clat * slon * y + slat * z
                    return np.vstack([e, n, u]).T

                # Antenna positions, names, numbers
                antpos = getattr(tel, "antenna_positions", None)
                names = getattr(tel, "antenna_names", None)
                numbers = getattr(tel, "antenna_numbers", None)
                nants = getattr(tel, "Nants", None)
                if (
                    antpos is not None
                    and names is not None
                    and numbers is not None
                    and nants
                    and use_tel_antennas
                ):
                    try:
                        dxyz = np.array(antpos, dtype=float)
                        # Convert to ENU using the (possibly updated) config location
                        lat_rad = np.deg2rad(config["location"]["lat"])
                        lon_rad = np.deg2rad(config["location"]["lon"])
                        enu = _ecef_deltas_to_enu(lat_rad, lon_rad, dxyz)
                        # Build antennas dict in expected format
                        antennas = {}
                        for i in range(int(nants)):
                            name = names[i] if i < len(names) else str(numbers[i])
                            num = int(numbers[i])
                            e, n, u = map(float, enu[i])
                            antennas[i] = {
                                "Name": name,
                                "Number": num,
                                "BeamID": 0,
                                "Position": (e, n, u),
                            }
                        used_from_tel["antenna_positions"] = True
                        kv(
                            "Antenna positions",
                            f"from pyuvdata (converted ITRF→ENU), Nants={nants}",
                        )
                    except Exception as e:
                        kv(
                            "Antenna positions",
                            f"present but could not be used ({e}); will fall back.",
                        )
                        used_from_tel["antenna_positions"] = False
                else:
                    if antpos is None or names is None or numbers is None or not nants:
                        kv(
                            "Antenna positions",
                            "not present; will fall back to antenna file.",
                        )
                    else:
                        kv(
                            "Antenna positions",
                            "present on Telescope but not used (use_pyuvdata_antennas=False); will fall back to antenna file.",
                        )
                    used_from_tel["antenna_positions"] = False

                # Antenna diameters
                diam = getattr(tel, "antenna_diameters", None)
                if use_tel_diameters and diam is not None:
                    try:
                        arr = np.array(diam)
                        if arr.ndim == 0:
                            dval = float(arr)
                            diameters_from_tel = dval
                            kv("Antenna diameters", f"from pyuvdata (scalar {dval} m)")
                        elif arr.ndim == 1:
                            tel_nums = list(
                                map(int, getattr(tel, "antenna_numbers", []))
                            )
                            if len(tel_nums) != len(arr):
                                kv(
                                    "Antenna diameters",
                                    "present but length mismatch vs antenna_numbers; not using.",
                                )
                            else:
                                diameters_from_tel = {
                                    int(tel_nums[i]): float(arr[i])
                                    for i in range(len(tel_nums))
                                }
                                # Print concise summary
                                vals = np.array(
                                    list(diameters_from_tel.values()), dtype=float
                                )
                                vmin, vmax = float(vals.min()), float(vals.max())
                                nunique = int(len(set(np.round(vals, 6))))
                                if np.allclose(vmin, vmax):
                                    kv(
                                        "Antenna diameters",
                                        f"uniform = {vmin} m across {len(vals)} antennas",
                                    )
                                else:
                                    print(
                                        f"{_c('Antenna diameters:', _BOLD + _CYAN)} per-antenna ({len(vals)}); unique={nunique}, min={vmin} m, max={vmax} m"
                                    )
                                    # Show a small sample for transparency
                                    sample = list(sorted(diameters_from_tel.items()))[
                                        :10
                                    ]
                                    kv(
                                        "Sample diameters (first 10)",
                                        ", ".join(f"{k}:{v}" for k, v in sample),
                                    )
                        else:
                            kv(
                                "Antenna diameters",
                                "present but invalid shape; not using.",
                            )
                    except Exception as e:
                        kv(
                            "Antenna diameters",
                            f"present but could not parse ({e}); not using.",
                        )
                else:
                    if diam is None:
                        kv(
                            "Antenna diameters",
                            "not present on Telescope; using config/user values.",
                        )
                    else:
                        kv(
                            "Antenna diameters",
                            "present on Telescope but not used (use_pyuvdata_diameters=False). Using config/user values.",
                        )

                # Feeds
                nfeeds = getattr(tel, "Nfeeds", None)
                feed_array = getattr(tel, "feed_array", None)
                feed_angle = getattr(tel, "feed_angle", None)
                if nfeeds is not None:
                    kv(
                        "Feeds",
                        f"Nfeeds={nfeeds} {'(feed_array available)' if feed_array is not None else ''}",
                    )
                else:
                    kv("Feeds", "not present; using config/defaults.")

                # Mount type (concise summary)
                mtype = getattr(tel, "mount_type", None)
                if mtype is not None:
                    try:
                        if isinstance(mtype, (list, tuple, np.ndarray)):
                            # Flatten to strings
                            vals = [
                                str(v)
                                for v in (
                                    mtype.tolist()
                                    if hasattr(mtype, "tolist")
                                    else mtype
                                )
                            ]
                            total = len(vals)
                            unique_vals = {}
                            for v in vals:
                                unique_vals[v] = unique_vals.get(v, 0) + 1
                            if len(unique_vals) == 1:
                                only = next(iter(unique_vals.keys()))
                                kv(
                                    "Mount type",
                                    f"uniform='{only}' across {total} antennas",
                                )
                            else:
                                # Show up to top 4 counts
                                top = sorted(
                                    unique_vals.items(), key=lambda x: (-x[1], x[0])
                                )[:4]
                                summary = ", ".join(f"{k}:{c}" for k, c in top)
                                kv(
                                    "Mount type",
                                    f"per-antenna; unique={len(unique_vals)}; top={summary}",
                                )
                        else:
                            kv("Mount type", f"'{mtype}'")
                    except Exception:
                        kv("Mount type", "present but could not summarize.")
                else:
                    kv("Mount type", "not present.")

                # Instrument and citation (simple reporting)
                instr = getattr(tel, "instrument", None)
                if instr is not None and str(instr).strip():
                    kv("Instrument", str(instr))
                else:
                    kv("Instrument", "not present.")

                citation = getattr(tel, "citation", None)
                if citation is not None and str(citation).strip():
                    kv("Citation", str(citation))
                else:
                    kv("Citation", "not present.")

            except Exception as e:
                print(
                    f"Could not load pyuvdata Telescope for '{tel_name}': {e}. Will use config/antenna file."
                )
        else:
            print(
                "use_PYUVData_telescope=True but telescope_name not provided; using config/antenna file."
            )

    # Convenience flags/defaults
    plotting_backend = config.get("plotting", "bokeh")
    # save_simulation_data_flag already determined above
    # Enable sky model plots by default unless explicitly disabled
    plot_skymodel_every_hour = bool(config.get("plot_skymodel_every_hour", True))
    skymodel_frequency = config.get("skymodel_frequency", 150)
    fov_radius_deg = config.get("fov_radius_deg", 10)

    # Access antennas either from pyuvdata Telescope or from antenna file
    if antennas is None:
        # Fall back to antenna file
        antenna_file = configured_ant_path
        if not antenna_file:
            raise ValueError(
                "Antenna positions not available from pyuvdata and no antenna file provided."
            )
        section("ANTENNA METADATA & POSITIONS")
        try:
            antennas = read_antenna_positions(antenna_file)
        except (FileNotFoundError, ValueError) as e:
            print(f"Error while reading antenna positions file: {e}")
            raise
    else:
        # We constructed antennas from Telescope; print a concise summary similar to file path reader
        section("ANTENNA METADATA & POSITIONS")
        print(_c("Antenna metadata and positions (from pyuvdata):", _BOLD + _CYAN))
        for idx, data in antennas.items():
            print(f"{_c(f'Antenna {idx}:', _BOLD + _CYAN)} {data}")
        total_memory_bytes = sys.getsizeof(antennas) + sum(
            sys.getsizeof(value) + sum(sys.getsizeof(v) for v in value.values())
            for value in antennas.values()
        )
        total_memory_mb = total_memory_bytes / (1024 * 1024)
        print(
            f"{_c('Total memory used by antennas:', _BOLD + _CYAN)} {total_memory_mb:.4f} MB"
        )
        # Note: antenna layout plots are generated once later after diameters validation

    # Assign antenna diameters with strict validation (no hidden defaults)
    per_antenna_config = config["antenna"].get("use_different_diameters", False)
    config_diameters_map = config["antenna"].get("diameters") or {}
    config_global_diameter = config["antenna"].get("all_antenna_diameter")

    # If diameters came from pyuvdata (dict), use them regardless of per_antenna_config
    if isinstance(diameters_from_tel, dict) and diameters_from_tel:
        missing = []
        for ant in antennas.values():
            num = int(ant["Number"])
            if num not in diameters_from_tel:
                missing.append(num)
            else:
                ant["diameter"] = float(diameters_from_tel[num])
        if missing:
            raise ValueError(
                "pyuvdata diameters enabled but missing for some antenna numbers: "
                f"{sorted(missing)}. Please provide diameters for all antennas."
            )
    elif per_antenna_config:
        # Expect per-antenna diameters to be provided in file/header or config['antenna']['diameters']
        missing = []
        for ant in antennas.values():
            if "diameter" in ant and ant["diameter"] is not None:
                continue
            num = int(ant["Number"])
            if num in config_diameters_map:
                ant["diameter"] = float(config_diameters_map[num])
            else:
                missing.append(num)
        if missing:
            raise ValueError(
                "use_different_diameters=True but diameters are missing for antenna numbers: "
                f"{sorted(missing)}. Add per-antenna diameters in the file or in antenna.diameters."
            )
    else:
        # Prefer scalar diameter from pyuvdata Telescope if provided
        if isinstance(diameters_from_tel, (int, float)):
            antenna_diameter = float(diameters_from_tel)
        else:
            # Require a user-provided scalar; empty or None is not allowed
            antenna_diameter = _parse_positive_float(
                config_global_diameter,
                "antenna.all_antenna_diameter",
            )
        for ant in antennas.values():
            ant["diameter"] = antenna_diameter

    # Validate antenna diameters before printing/using them
    invalid = []
    for ant in antennas.values():
        d = ant.get("diameter")
        if d is None:
            invalid.append((ant.get("Number", "?"), "missing"))
            continue
        try:
            d_float = float(d)
        except Exception:
            invalid.append((ant.get("Number", "?"), "not a number"))
            continue
        if not np.isfinite(d_float) or d_float <= 0:
            invalid.append((ant.get("Number", "?"), f"invalid value {d_float}"))
        else:
            ant["diameter"] = d_float

    if invalid:
        details = ", ".join([f"{num}: {reason}" for num, reason in invalid])
        raise ValueError(
            "Invalid antenna diameters detected. Ensure all diameters are positive, finite numbers. "
            f"Problems for antenna numbers → {details}"
        )

    # Debug output
    section("ANTENNA DIAMETERS")
    for idx, data in antennas.items():
        print(f"{_c(f'Antenna {idx}:', _BOLD + _CYAN)} {data}")

    # Antenna types reporting
    section("ANTENNA TYPES")
    ant_cfg = config.get("antenna", {})
    if ant_cfg.get("use_different_antenna_types", False):
        types_map = ant_cfg.get("types_per_antenna", {}) or {}
        # Print per-antenna type if provided; otherwise report as unknown
        for ant in antennas.values():
            num = int(ant.get("Number", -1))
            atype = types_map.get(num, "unknown")
            print(f"{_c(f'Antenna {num}:', _BOLD + _CYAN)} type={atype}")
    else:
        atype = ant_cfg.get("antenna_type", "unknown")
        print(f"{_c('Antenna type (all):', _BOLD + _CYAN)} {atype}")

    # Antenna illumination reporting
    section("ANTENNA ILLUMINATION")
    if ant_cfg.get("use_different_illumination", False):
        illum_map = ant_cfg.get("illumination_per_antenna", {}) or {}
        for ant in antennas.values():
            num = int(ant.get("Number", -1))
            illum = illum_map.get(num, "uniform")
            print(f"{_c(f'Antenna {num}:', _BOLD + _CYAN)} illumination={illum}")
    else:
        illum = ant_cfg.get("illumination", "uniform")
        print(f"{_c('Illumination (all):', _BOLD + _CYAN)} {illum}")
    section("ANTENNA LAYOUT")
    print(_c("Antenna layout:", _BOLD + _CYAN), "see browser for interactive view.")
    plot_antenna_layout(
        antennas,
        plotting=plotting_backend,
        save_simulation_data=save_simulation_data_flag,
        folder_path=simulation_folder_path,
        open_in_browser=config.get("output", {}).get("plot_results_in_bokeh", True),
    )
    _al_outdir2 = simulation_folder_path or tempfile.mkdtemp(prefix="rrivis_")
    from plot import plot_antenna_layout_3d_plotly as _plot3d

    _plot3d(
        antennas,
        save_simulation_data=True,
        folder_path=_al_outdir2,
        open_in_browser=config.get("output", {}).get("plot_results_in_bokeh", True),
    )

    # Check if the HPBW is fixed for all frequencies (configured in degrees).
    # Convert once to radians for internal use.
    fixed_hpbw_deg = config["antenna"]["fixed_hpbw"]
    fixed_hpbw_rad = (
        None if fixed_hpbw_deg is None else np.radians(float(fixed_hpbw_deg))
    )

    # Convert frequency inputs to Hz based on the unit
    unit_conversion = {
        "Hz": 1,
        "kHz": 1e3,
        "MHz": 1e6,
        "GHz": 1e9,
    }

    # Direct nested access to frequency parameters
    try:
        # Values are specified in the units given by frequency_unit
        start_frequency = float(config["frequency"]["starting_frequency"])  # >= 0
        frequency_interval = float(config["frequency"]["frequency_interval"])  # > 0
        frequency_bandwidth = float(config["frequency"]["frequency_bandwidth"])  # > 0
        frequency_unit = config["frequency"]["frequency_unit"]
    except KeyError as e:
        raise ValueError(f"Missing required configuration key: {e}")

    if frequency_unit not in unit_conversion:
        raise ValueError(
            f"Invalid frequency unit: {frequency_unit}. Must be one of {list(unit_conversion.keys())}."
        )

    # Validate
    if start_frequency < 0:
        raise ValueError("frequency.starting_frequency must be >= 0")
    if frequency_interval <= 0:
        raise ValueError("frequency.frequency_interval must be > 0")
    if frequency_bandwidth <= 0:
        raise ValueError("frequency.frequency_bandwidth must be > 0")

    # Convert start/interval/bandwidth to Hz based on unit
    start_frequency_hz = start_frequency * unit_conversion[frequency_unit]
    interval_hz = frequency_interval * unit_conversion[frequency_unit]
    total_bandwidth_hz = frequency_bandwidth * unit_conversion[frequency_unit]

    # Compute the end frequency and the array of frequencies
    end_frequency_hz = start_frequency_hz + total_bandwidth_hz
    frequencies = np.arange(
        start_frequency_hz, end_frequency_hz, interval_hz, dtype=float
    )
    wavelengths = c / frequencies

    # Debug output
    section("FREQUENCY GRID")
    print(f"{_c('Start Frequency:', _BOLD + _CYAN)} {start_frequency} {frequency_unit}")
    print(
        f"{_c('Frequency Interval:', _BOLD + _CYAN)} {frequency_interval} {frequency_unit}"
    )
    print(
        f"{_c('Total Bandwidth:', _BOLD + _CYAN)} {frequency_bandwidth} {frequency_unit}"
    )
    print(
        f"{_c('End Frequency:', _BOLD + _CYAN)} {end_frequency_hz / unit_conversion[frequency_unit]} {frequency_unit}"
    )
    print(f"{_c('Number of Frequency Channels:', _BOLD + _CYAN)} {len(frequencies)}")

    # Loop through each antenna to calculate HPBW (radians) and beam areas
    hpbw_per_antenna = {}
    beam_area_per_antenna = {}

    for ant in antennas.values():
        antenna_number = ant["Number"]
        diameter = ant["diameter"]

        # Calculate HPBW (radians) for each frequency for this antenna
        hpbw_values_radians = calculate_hpbw_radians(
            frequencies_hz=frequencies,
            dish_diameter=diameter,
            fixed_hpbw_radians=fixed_hpbw_rad,
        )
        hpbw_per_antenna[antenna_number] = hpbw_values_radians

        # TODO: Currently only EBeam is used! Add support for Power Beams.
        # TODO: Only Gaussian Beam Area is supported! Add support for other response types.
        # Calculate the beam area for each frequency
        beam_area = calculate_gaussian_beam_area_EBeam(
            nside=512, theta_HPBW=hpbw_values_radians
        )
        beam_area_per_antenna[antenna_number] = beam_area

    # Debugging: Print HPBW and beam areas for each antenna
    section("HPBW & BEAM AREAS PER ANTENNA")
    for ant in antennas.values():
        antenna_number = ant["Number"]
        # Pretty print arrays with limited length and spacing
        # Internally stored in radians; convert to degrees only for display
        h_arr = np.degrees(np.asarray(hpbw_per_antenna[antenna_number]))
        b_arr = np.asarray(beam_area_per_antenna[antenna_number], dtype=float)
        np.set_printoptions(linewidth=120, threshold=12, edgeitems=3, suppress=True)
        print(
            f"{_c(f'Antenna {antenna_number}:', _BOLD + _CYAN)} HPBW (degrees) = {np.array2string(h_arr, precision=3, separator=', ')}"
        )
        print(
            f"{_c(f'Antenna {antenna_number}:', _BOLD + _CYAN)} Beam Area (steradians) = {np.array2string(b_arr, precision=3, separator=', ')}"
        )

    # Beams type and beams response for antennas
    if config["beams"]["use_different_beams_for_each_antenna"]:
        beams_per_antenna = config["beams"]["beams_per_antenna"]
    else:
        # Map beam types using antenna numbers
        beams_per_antenna = {
            ant["Number"]: config["beams"]["all_same_beam_type"]
            for ant in antennas.values()
        }

    if config["beams"]["use_different_beam_responses_for_each_antenna"]:
        beam_response_per_antenna = config["beams"]["beam_response_per_antenna"]
    else:
        # Map beam responses using antenna numbers
        beam_response_per_antenna = {
            ant["Number"]: config["beams"]["all_same_beam_response"]
            for ant in antennas.values()
        }

    # Generate baselines from the antennas
    section("GENERATING BASELINES")
    try:
        baselines = generate_baselines(
            antennas, beams_per_antenna, beam_response_per_antenna
        )
        print("Baselines generated successfully.")
    except ValueError as ve:
        print(f"ValueError: {ve}")
        raise
    except KeyError as ke:
        print(f"KeyError: {ke}")
        raise
    except Exception as e:
        print(f"An unexpected error occurred while generating baselines: {e}")
        raise

    # Trim baselines based on baseline configuration
    if baselines:
        trimmed_baselines = {}

        # Extract baseline configuration
        use_autocorrelations = config["baseline"]["use_autocorrelations"]
        use_crosscorrelations = config["baseline"]["use_crosscorrelations"]
        only_selective_baselines = config["baseline"]["only_selective_baseline_length"]
        selective_lengths = config["baseline"]["selective_baseline_lengths"]
        tolerance = 2.0  # Tolerance for selective baseline lengths (in meters)

        # Iterate through the baselines and apply filters
        for (ant1, ant2), baseline_data in baselines.items():
            baseline_length = baseline_data["Length"]

            # Filter out autocorrelations if disabled
            if not use_autocorrelations and ant1 == ant2:
                continue

            # Filter out crosscorrelations if disabled
            if not use_crosscorrelations and ant1 != ant2:
                continue

            # Filter for selective baseline lengths
            if only_selective_baselines:
                # Check if the baseline length is within the acceptable range of any selective length
                if not any(
                    abs(baseline_length - length) <= tolerance
                    for length in selective_lengths
                ):
                    continue

            # Add the baseline to the trimmed dictionary if it passes all filters
            trimmed_baselines[(ant1, ant2)] = baseline_data

        # Check if the trimmed baselines are empty
        if not trimmed_baselines:
            error_message = "No baselines match the specified criteria. Trimmed baselines are empty."
            print(error_message)
            raise ValueError(error_message)
        else:
            # Debugging: Print trimmed baselines
            section("TRIMMED BASELINES")
            for key, value in trimmed_baselines.items():
                print(f"{_c(f'Baseline {key}:', _BOLD + _CYAN)} {value}")

        # Replace the original baselines with the trimmed version
        baselines = trimmed_baselines
    else:
        print("No baselines to process!")

    # Get observation location and start time
    section("OBSERVATION SETUP")
    location, obstime_start = get_location_and_time(
        config["location"]["lat"],
        config["location"]["lon"],
        config["location"]["height"],
        config["obs_time"]["start_time"],
    )

    # (pyuvdata Telescope integration handled earlier)

    # Ensure only one sky model is enabled
    sky_model_config = config["sky_model"]

    enabled_sky_models = {
        "test_sources": sky_model_config["test_sources"]["use_test_sources"],
        "test_sources_healpix": sky_model_config["test_sources_healpix"][
            "use_test_sources"
        ],
        "gsm_healpix": sky_model_config["gsm_healpix"]["use_gsm"],
        "gleam": sky_model_config["gleam"]["use_gleam"],
        "gleam_healpix": sky_model_config["gleam_healpix"]["use_gleam"],
        "gsm+gleam_healpix": sky_model_config["gsm+gleam_healpix"]["use_gsm_gleam"],
    }

    # Count enabled sky models
    enabled_count = sum(enabled_sky_models.values())

    if enabled_count > 1:
        conflicting_models = [
            key for key, enabled in enabled_sky_models.items() if enabled
        ]
        raise ValueError(
            f"Conflicting sky models enabled: {', '.join(conflicting_models)}. Please enable only one sky model at a time."
        )

    # Auto-select a sky model when none is enabled
    if enabled_count == 0:
        section("SKY MODEL SELECTION")
        print(
            _c("No sky model enabled in config:", _BOLD + _CYAN),
            "attempting auto-selection...",
        )
        # simple connectivity check to VizieR host
        import socket

        def _has_internet(host="vizier.cds.unistra.fr", port=443, timeout=3):
            try:
                with socket.create_connection((host, port), timeout=timeout):
                    return True
            except Exception:
                return False

        if _has_internet():
            print(
                _c("Internet available →", _BOLD + _CYAN),
                "selecting GLEAM catalog sources.",
            )
            sky_model_config["gleam"]["use_gleam"] = True
        else:
            print(
                _c("No internet detected →", _BOLD + _CYAN),
                "falling back to test sources.",
            )
            sky_model_config["test_sources"]["use_test_sources"] = True

    use_gleam = sky_model_config["gleam"]["use_gleam"]
    use_gsm = sky_model_config["gsm_healpix"]["use_gsm"]
    use_gleam_healpix = sky_model_config["gleam_healpix"]["use_gleam"]
    use_gsm_gleam_healpix = sky_model_config["gsm+gleam_healpix"]["use_gsm_gleam"]
    use_test_sources_healpix = sky_model_config["test_sources_healpix"][
        "use_test_sources"
    ]
    use_test_sources = sky_model_config["test_sources"]["use_test_sources"]

    flux_limit = None
    nside = None

    # Load sources
    section("SKY MODEL LOADING")
    if sky_model_config["gleam"]["use_gleam"]:
        flux_limit = sky_model_config["gleam"]["flux_limit"]
    elif sky_model_config["gsm_healpix"]["use_gsm"]:
        flux_limit = sky_model_config["gsm_healpix"]["flux_limit"]
        nside = sky_model_config["gsm_healpix"]["nside"]
    elif sky_model_config["gleam_healpix"]["use_gleam"]:
        flux_limit = sky_model_config["gleam_healpix"]["flux_limit"]
        nside = sky_model_config["gleam_healpix"]["nside"]
    elif sky_model_config["gsm+gleam_healpix"]["use_gsm_gleam"]:
        flux_limit = sky_model_config["gsm+gleam_healpix"]["flux_limit"]
        nside = sky_model_config["gsm+gleam_healpix"]["nside"]
    elif sky_model_config["test_sources_healpix"]["use_test_sources"]:
        flux_limit = sky_model_config["test_sources_healpix"]["flux_limit"]
        nside = sky_model_config["test_sources_healpix"]["nside"]
    else:
        flux_limit = sky_model_config["test_sources"]["flux_limit"]

    try:
        sources, spectral_indices = get_sources(
            use_test_sources=use_test_sources,
            use_test_sources_healpix=use_test_sources_healpix,
            use_gleam=use_gleam,
            use_gsm=use_gsm,
            use_gleam_healpix=use_gleam_healpix,
            use_gsm_gleam_healpix=use_gsm_gleam_healpix,
            gleam_catalogue=sky_model_config["gleam"]["gleam_catalogue"],
            gsm_catalogue=sky_model_config["gsm_healpix"]["gsm_catalogue"],
            flux_limit=flux_limit,
            frequency=config["frequency"]["starting_frequency"] * 1e6,
            nside=nside,
        )
    except Exception as e:
        if sky_model_config["gleam"]["use_gleam"]:
            print(f"GLEAM loading failed ({e}); falling back to test sources.")
            sources, spectral_indices = get_sources(
                use_test_sources=True,
                use_test_sources_healpix=False,
                use_gleam=False,
                use_gsm=False,
                use_gleam_healpix=False,
                use_gsm_gleam_healpix=False,
                flux_limit=None,
                frequency=config["frequency"]["starting_frequency"] * 1e6,
                nside=None,
            )
        else:
            raise

    # Convert time_interval to seconds (support legacy key time_interval_between_observation)
    obs_cfg = config["obs_time"]
    time_interval_value = obs_cfg.get("time_interval")
    if time_interval_value is None:
        time_interval_value = obs_cfg.get("time_interval_between_observation")
    if time_interval_value is None:
        raise ValueError(
            "Missing obs_time.time_interval (or time_interval_between_observation)"
        )

    if obs_cfg["time_interval_unit"] == "hours":
        time_interval_seconds = time_interval_value * 3600
    elif obs_cfg["time_interval_unit"] == "seconds":
        time_interval_seconds = time_interval_value
    elif obs_cfg["time_interval_unit"] == "minutes":
        time_interval_seconds = time_interval_value * 60
    else:
        raise ValueError(f"Invalid time_interval_unit: {obs_cfg['time_interval_unit']}")
    print(f"Time Interval between each observation: {time_interval_seconds} seconds")

    # Convert total_duration to seconds
    if obs_cfg["total_duration_unit"] == "days":
        total_duration_seconds = obs_cfg["total_duration"] * 86400
        print(
            f"Total duration of the simulation: {total_duration_seconds / 86400} days"
        )
    elif obs_cfg["total_duration_unit"] == "hours":
        total_duration_seconds = obs_cfg["total_duration"] * 3600
        print(
            f"Total duration of the simulation: {total_duration_seconds / 3600} hours"
        )
    elif obs_cfg["total_duration_unit"] == "seconds":
        total_duration_seconds = obs_cfg["total_duration"]
        print(f"Total duration of the simulation: {total_duration_seconds} seconds")
    else:
        raise ValueError(
            f"Invalid total_duration_unit: {obs_cfg['total_duration_unit']}"
        )

    # Time points for simulation
    time_points = np.arange(0, total_duration_seconds, time_interval_seconds)

    # Convert time points to MJD format
    mjd_time_points = (obstime_start + TimeDelta(time_points, format="sec")).mjd

    # Initialize dictionaries to store modulus and phase over time for each baseline
    moduli_over_time = {key: [] for key in baselines.keys()}
    phases_over_time = {key: [] for key in baselines.keys()}

    start_time = time.time()

    # Calculate visibility based on the selected sky model
    visibility_function = calculate_visibility

    # Initialize dictionaries to store complex visibility over time for each baseline
    visibilities = {key: [] for key in baselines.keys()}

    # Loop over time points to calculate visibility
    section("SIMULATION PROGRESS")
    total_steps = len(time_points)
    # Adaptive progress cadence: print every step for small runs; otherwise ~10% increments
    progress_interval = 1 if total_steps <= 50 else max(1, total_steps // 10)
    for idx, current_time in enumerate(time_points):
        # Update observation time
        obstime = obstime_start + TimeDelta(current_time, format="sec")

        # Calculate visibility for current time
        visibility_dict = visibility_function(
            antennas,
            baselines,
            sources,
            spectral_indices,
            location,
            obstime,
            wavelengths,
            frequencies,
            hpbw_per_antenna,
            nside=nside,
        )

        # Append visibility data for each baseline
        for key in baselines.keys():
            visibilities[key].append(visibility_dict[key])

        # # Calculate modulus and phase of visibility
        moduli, phases = calculate_modulus_phase(visibility_dict)

        # # Append modulus and phase to the time series data
        for key in baselines.keys():
            moduli_over_time[key].append(moduli[key])
            phases_over_time[key].append(phases[key])

        # Adaptive progress reporting
        if (idx % progress_interval == 0) or (idx == total_steps - 1):
            elapsed_time = time.time() - start_time
            if idx > 0:
                time_per_step = elapsed_time / idx
                remaining_steps = total_steps - idx
                estimated_remaining_time = time_per_step * remaining_steps
                print(
                    f"{_c('Time step', _BOLD + _CYAN)} {idx+1}/{total_steps}: "
                    f"Estimated remaining time: {estimated_remaining_time:.2f} seconds"
                )
            else:
                print(
                    f"{_c('Time step', _BOLD + _CYAN)} {idx+1}/{total_steps}: Time elapsed: {elapsed_time:.2f} seconds"
                )

    # Plot the gsm2008 map along with gleam sources
    if plot_skymodel_every_hour:
        section("SKY MAP PLOTS (GSM)")
        diffused_sky_model(
            location=location,
            obstime_start=obstime_start,
            total_seconds=total_duration_seconds,
            frequency=skymodel_frequency,
            fov_radius_deg=fov_radius_deg,
            gleam_sources=sources if use_gleam else None,
            save_simulation_data=save_simulation_data_flag,
            folder_path=simulation_folder_path,
            open_in_browser=config.get("output", {}).get("plot_results_in_bokeh", True),
        )
    # Save computed data to an HDF5 file (derive name if not provided)
    output_file_name = config.get("output_file") or (
        config.get("output", {}).get("output_file_name", "complex_visibility") + ".h5"
    )
    if save_simulation_data_flag and simulation_folder_path and output_file_name:
        section("SAVING OUTPUTS")
        output_file_path = os.path.join(simulation_folder_path, output_file_name)
        with h5py.File(output_file_path, "w") as h5file:
            # Create a group for each baseline
            for key, vis in visibilities.items():
                baseline_group = h5file.create_group(
                    f"baseline_{key}"
                )  # Create a group for the baseline

                # Save complex visibility
                vis_array = np.stack(vis)  # Convert to 2D NumPy array
                baseline_group.create_dataset(
                    "complex_visibility",
                    data=vis_array.astype(np.complex128),
                    dtype="complex128",
                )

            # Save frequencies and time points
            h5file.create_dataset("frequencies", data=frequencies)
            h5file.create_dataset("time_points_mjd", data=mjd_time_points)

            # Save metadata as attributes (optional)
            if "gleam_flux_limit" in config:
                h5file.attrs["gleam_flux_limit"] = config["gleam_flux_limit"]
            if "theta_HPBW" in config:
                h5file.attrs["theta_HPBW"] = config["theta_HPBW"]
            h5file.attrs["num_antennas"] = len(
                antennas
            )  # Calculated from the antennas list
            h5file.attrs["num_baselines"] = len(
                baselines
            )  # Calculated from the baselines dictionary
            h5file.attrs["location"] = str(location)

        print(f"Simulation data saved to {output_file_path}")
    elif save_simulation_data_flag and not simulation_folder_path:
        print(
            "Warning: save_simulation_data=True but simulation folder could not be created; skipping file save."
        )

    # Calculate total memory usage for visibilities in MB
    total_memory_mb = 0
    for key in visibilities.keys():
        # Convert the list of arrays for each baseline into a stacked 2D NumPy array
        vis_array = np.stack(visibilities[key])  # Shape: (time_steps, frequencies)
        total_memory_mb += vis_array.nbytes / (1024**2)  # Convert bytes to MB

    print(
        f"{_c('Total memory used by visibility data:', _BOLD + _CYAN)} {total_memory_mb:.2f} MB"
    )

    # After the loop, execute all the sky model plot functions
    # if args.plot_skymodel_every_hour:
    #     for i, plot_func in enumerate(sky_model_plots):
    #         plt.figure()  # Create a new figure for each plot
    #         plot_func()  # Generate each sky model plot

    # Convert lists to numpy arrays for plotting
    for key in baselines.keys():
        moduli_over_time[key] = np.array(moduli_over_time[key])
        phases_over_time[key] = np.array(phases_over_time[key])

    # Convert time_points to desired units for plotting, e.g., hours
    # time_points_hours = time_points / 3600  # Convert seconds to hours for plotting

    # Generate plots based on the selected plotting library
    fig1 = plot_visibility(
        moduli_over_time,
        phases_over_time,
        baselines,
        mjd_time_points,  # MJD for plotting
        frequencies,
        total_duration_seconds,
        plotting=plotting_backend,
        save_simulation_data=save_simulation_data_flag,
        folder_path=simulation_folder_path,
        angle_unit=angle_unit,
    )
    fig2 = plot_heatmaps(
        moduli_over_time,
        phases_over_time,
        baselines,
        frequencies,
        total_duration_seconds,
        mjd_time_points,
        plotting=plotting_backend,
        save_simulation_data=save_simulation_data_flag,
        folder_path=simulation_folder_path,
    )
    fig3 = plot_modulus_vs_frequency(
        moduli_over_time,
        phases_over_time,
        baselines,
        frequencies,
        mjd_time_points,
        plotting=plotting_backend,
        save_simulation_data=save_simulation_data_flag,
        folder_path=simulation_folder_path,
    )

    # Display plots (optional)
    if plotting_backend == "bokeh" and config.get("output", {}).get(
        "plot_results_in_bokeh", True
    ):
        # Explicitly control Bokeh output files. If not saving, use a temp directory.
        output_dir = simulation_folder_path or tempfile.mkdtemp(prefix="rrivis_")
        try:
            output_file(
                os.path.join(output_dir, "show_visibility-phase-lsts.html"),
                title="Visibility and Phase vs LST",
            )
            show(fig1)
        finally:
            reset_output()
        try:
            output_file(
                os.path.join(output_dir, "show_modulus-phase-freq.html"),
                title="Modulus and Phase vs Frequency",
            )
            show(fig3)
        finally:
            reset_output()
        try:
            output_file(
                os.path.join(output_dir, "show_heatmaps-freq-time.html"),
                title="Visibility Heatmaps",
            )
            show(fig2)
        finally:
            reset_output()
    else:
        plt.show()

    # Ending section and final success message
    section("ENDING SIMULATION")
    logger = logging.getLogger()
    if hasattr(logger, "success"):
        logger.success("Simulation completed successfully.")
    else:
        print("Simulation completed successfully.")
    # Trailing spacing after end of output
    print("")
    print("")


if __name__ == "__main__":
    main()
