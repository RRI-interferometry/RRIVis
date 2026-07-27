# radiosim/io/writers.py
"""Retained resolved-configuration artifact writer for RadioSim workflows."""

from pathlib import Path
from typing import Any

import yaml
from typing_extensions import override


def save_config_yaml(
    config: dict[str, Any],
    output_path: str | Path,
) -> str:
    """Save configuration to YAML format with proper formatting.

    Parameters
    ----------
    config : dict
        Configuration dictionary to save
    output_path : str or Path
        Path to output YAML file

    Returns
    -------
    str
        Path to the saved file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Custom YAML dumper for better readability
    class FormattedDumper(yaml.Dumper):
        @override
        def write_line_break(self, data: str | None = None) -> None:
            # Add a blank line between top-level sections
            if self.indent == 0:
                _ = self.stream.write("\n")
            super().write_line_break(data)

    with open(output_path, "w") as f:
        yaml.dump(
            config, f, Dumper=FormattedDumper, default_flow_style=False, sort_keys=True
        )

    return str(output_path)
