# src/antenna.py
import sys
import os


def _supports_color() -> bool:
    try:
        return hasattr(sys.__stdout__, "isatty") and sys.__stdout__.isatty()
    except Exception:
        return False


_TTY = _supports_color()
_RESET = "\033[0m"
_BOLD = "\033[1m"
_CYAN = "\033[36m"


def _c(text: str, style: str) -> str:
    if not _TTY:
        return text
    return f"{style}{text}{_RESET}"


def read_antenna_positions(file_path):
    """
    Reads antenna positions and metadata from a file, with support for additional information.

    The file should contain antenna information in the format:
    Name  Number  BeamID  E  N  U  [Diameter]
    Additional columns may be present and are ignored, except an optional
    column named 'Diameter' in the header (case-insensitive), which will be
    parsed to attach per-antenna dish diameters.

    Parameters:
    file_path (str): Path to the antenna position file.

    Returns:
    dict: Dictionary with antenna indices as keys and dictionaries of metadata and positions as values.

    Raises:
    ValueError: If the file is empty, has invalid data, or the file format is incorrect.
    FileNotFoundError: If the file path does not exist.
    """
    # Check if file path is provided
    if not file_path:
        raise ValueError("Antenna positions file path is not provided.")

    # Check if file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file '{file_path}' does not exist.")

    antennas = {}
    try:
        with open(file_path, "r") as f:
            lines = f.readlines()

        # Ensure the file has more than just the header
        if len(lines) <= 1:
            raise ValueError(
                "The antenna positions file is empty or has no valid data."
            )

        # Detect header and optional Diameter column index
        header_idx = None
        diameter_col_idx = None
        for idx, line in enumerate(lines):
            if line.strip():
                header_idx = idx
                header_tokens = line.strip().split()
                for j, tok in enumerate(header_tokens):
                    if tok.lower() == "diameter":
                        diameter_col_idx = j
                        break
                break

        for i, line in enumerate(lines):
            # Skip the header line or empty lines
            if i == header_idx or not line.strip():
                continue
            parts = line.strip().split()
            if len(parts) < 6:
                raise ValueError(f"Invalid antenna position in line {i+1}: {line}")
            # Extract metadata and positions
            try:
                name = parts[0]
                number = int(parts[1])
                beam_id = int(parts[2])
                e, n, u = map(float, parts[3:6])
                ant = {
                    "Name": name,
                    "Number": number,
                    "BeamID": beam_id,
                    "Position": (e, n, u),
                }
                # Optional diameter if header specified a Diameter column and value present
                if diameter_col_idx is not None and len(parts) > diameter_col_idx:
                    try:
                        ant["diameter"] = float(parts[diameter_col_idx])
                    except Exception:
                        # Ignore if not parseable; diameter assignment handled in main
                        pass
                antennas[i - 1] = ant
            except ValueError:
                raise ValueError(f"Could not parse data in line {i+1}: {line}")

    except Exception as e:
        raise ValueError(f"Failed to read antenna positions file '{file_path}': {e}")

    # Debug output
    print("")
    print(_c("Antenna metadata and positions:", _BOLD + _CYAN))
    for idx, data in antennas.items():
        print(f"{_c(f'Antenna {idx}:', _BOLD + _CYAN)} {data}")

    # Calculate total memory usage in MB
    total_memory_bytes = sys.getsizeof(antennas) + sum(
        sys.getsizeof(value) + sum(sys.getsizeof(v) for v in value.values())
        for value in antennas.values()
    )
    total_memory_mb = total_memory_bytes / (1024 * 1024)
    print(
        f"{_c('Total memory used by antennas:', _BOLD + _CYAN)} {total_memory_mb:.4f} MB"
    )

    return antennas
