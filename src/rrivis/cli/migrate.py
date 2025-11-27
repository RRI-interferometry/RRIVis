"""Configuration migration tool for v0.1.x -> v0.2.0.

Provides automated migration of configuration files from the old format
to the new Pydantic-validated format.
"""

import argparse
import sys
from pathlib import Path
from typing import Any, Dict

import yaml


def migrate_config(old_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Migrate v0.1.x config to v0.2.0 format.

    Args:
        old_config: v0.1.x configuration dictionary

    Returns:
        v0.2.0 configuration dictionary

    Changes:
    - Adds backend selection (default: "auto")
    - Adds Jones chain configuration
    - Maps old beam format to new E Jones config
    - Preserves all existing settings
    """
    new_config = old_config.copy()

    # Add backend selection (new in v0.2.0)
    if "backend" not in new_config:
        new_config["backend"] = "auto"  # Auto-detect best available

    # Add Jones chain configuration (new in v0.2.0)
    if "jones" not in new_config:
        # Default: Enable K (geometric) and E (beam) only
        # Preserve v0.1.x behavior (no atmospheric/instrumental effects)
        new_config["jones"] = {
            "K": {"enabled": True},   # Geometric phase (always on)
            "E": {"enabled": True},   # Primary beam (always on)
            "Z": {"enabled": False},  # Ionosphere (opt-in)
            "T": {"enabled": False},  # Troposphere (opt-in)
            "P": {"enabled": False},  # Parallactic (opt-in)
            "G": {"enabled": False},  # Gains (opt-in)
            "B": {"enabled": False},  # Bandpass (opt-in)
            "D": {"enabled": False},  # Pol leakage (opt-in)
        }

        # Migrate old beam format to new E Jones config
        if "beam_file" in old_config:
            new_config["jones"]["E"]["beam_file"] = old_config.pop("beam_file")
            new_config["jones"]["E"]["provider"] = "custom_fits"

    # Add output format options (new in v0.2.0)
    if "output" in new_config and "format" not in new_config.get("output", {}):
        new_config["output"]["format"] = "hdf5"  # Default for v0.1.x compatibility

    # Add multiprocessing options (new in v0.2.0)
    if "parallel" not in new_config:
        new_config["parallel"] = {
            "enabled": False,  # Preserve v0.1.x single-threaded behavior
            "n_workers": None,  # Will auto-detect if enabled
        }

    # Add config version marker
    new_config["_config_version"] = "0.2.0"

    return new_config


def write_migrated_config(
    old_config_path: Path,
    new_config_path: Path,
    backup: bool = True,
) -> None:
    """
    Migrate config file and write to disk.

    Args:
        old_config_path: Path to v0.1.x config
        new_config_path: Path for v0.2.0 config
        backup: Create backup of old config (default: True)
    """
    # Load old config
    with open(old_config_path) as f:
        old_config = yaml.safe_load(f) or {}

    # Migrate
    new_config = migrate_config(old_config)

    # Backup old config
    if backup:
        backup_path = old_config_path.with_suffix(".yaml.v0.1.x.bak")
        with open(backup_path, "w") as f:
            yaml.dump(old_config, f, default_flow_style=False)
        print(f"Backed up old config to: {backup_path}")

    # Write new config
    with open(new_config_path, "w") as f:
        yaml.dump(new_config, f, default_flow_style=False, sort_keys=False)
        f.write("\n# Migrated from v0.1.x to v0.2.0\n")
        f.write("# Added fields:\n")
        f.write("#   - backend: Backend selection (numpy/numba/jax/auto)\n")
        f.write("#   - jones: Jones matrix configuration\n")
        f.write("#   - parallel: Multiprocessing options\n")

    print(f"Migrated config written to: {new_config_path}")
    print(f"\nAdded new fields:")
    print(f"  - backend: '{new_config.get('backend', 'auto')}'")
    print(f"  - jones: 8 Jones terms (K, E enabled by default)")
    print(f"  - parallel: multiprocessing options")


def validate_migration(
    old_config_path: Path,
    new_config_path: Path,
) -> bool:
    """
    Validate that migration preserves original settings.

    Args:
        old_config_path: Path to original config
        new_config_path: Path to migrated config

    Returns:
        True if migration is valid
    """
    with open(old_config_path) as f:
        old_config = yaml.safe_load(f) or {}

    with open(new_config_path) as f:
        new_config = yaml.safe_load(f) or {}

    # Check that all original keys are preserved
    def check_keys(old: Dict, new: Dict, path: str = "") -> bool:
        for key, value in old.items():
            new_path = f"{path}.{key}" if path else key
            if key not in new:
                print(f"  Missing key: {new_path}")
                return False
            if isinstance(value, dict):
                if not check_keys(value, new[key], new_path):
                    return False
        return True

    return check_keys(old_config, new_config)


def cli_migrate(argv: list = None) -> int:
    """Command-line entry point for config migration."""
    parser = argparse.ArgumentParser(
        prog="rrivis-migrate",
        description="Migrate RRIvis config from v0.1.x to v0.2.0",
    )

    parser.add_argument(
        "input",
        type=Path,
        help="Input config file (v0.1.x format)",
    )

    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=None,
        help="Output config file (default: input.v0.2.0.yaml)",
    )

    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Don't create backup of old config",
    )

    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate migration after completion",
    )

    args = parser.parse_args(argv)

    # Check input exists
    if not args.input.exists():
        print(f"Error: Input file not found: {args.input}")
        return 1

    # Determine output path
    if args.output is None:
        output_path = args.input.with_suffix(".v0.2.0.yaml")
    else:
        output_path = args.output

    # Migrate
    try:
        write_migrated_config(
            args.input,
            output_path,
            backup=not args.no_backup,
        )
    except Exception as e:
        print(f"Error during migration: {e}")
        return 1

    # Validate if requested
    if args.validate:
        print("\nValidating migration...")
        if validate_migration(args.input, output_path):
            print("Validation passed: All original settings preserved")
        else:
            print("Validation WARNING: Some settings may not have been preserved")

    print(f"\nMigration complete!")
    print(f"You can now run: rrivis {output_path}")

    return 0


if __name__ == "__main__":
    sys.exit(cli_migrate())
