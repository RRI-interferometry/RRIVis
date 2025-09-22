# src/observation.py

from astropy.coordinates import EarthLocation
from astropy.time import Time
import astropy.units as u
import sys


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


def get_location_and_time(lat=None, lon=None, height=None, starttime=None):
    """
    Returns the observation location and start time.

    Parameters:
    lat (float): Latitude in degrees. Defaults to HERA latitude.
    lon (float): Longitude in degrees. Defaults to HERA longitude.
    height (float): Height in meters. Defaults to HERA height.
    starttime (str): Observation start time in ISO format. Defaults to the current UTC time.

    Returns:
    tuple: EarthLocation object and Time object for observation start time.

    Raises:
    ValueError: If latitude, longitude, or height are out of valid range, or if starttime is invalid.
    """
    # Default HERA coordinates if not provided
    default_lat = -30.72152777777791
    default_lon = 21.428305555555557
    default_height = 1073.0

    # Validate latitude
    if lat is not None:
        if not (-90 <= lat <= 90):
            raise ValueError(
                f"Invalid latitude: {lat}. Must be between -90 and 90 degrees."
            )
    else:
        lat = default_lat

    # Validate longitude
    if lon is not None:
        if not (-180 <= lon <= 180):
            raise ValueError(
                f"Invalid longitude: {lon}. Must be between -180 and 180 degrees."
            )
    else:
        lon = default_lon

    # Validate height
    if height is not None:
        if not (isinstance(height, (int, float)) and height >= 0):
            raise ValueError(
                f"Invalid height: {height}. Must be a non-negative number."
            )
    else:
        height = default_height

    # Validate and parse starttime
    try:
        obstime_start = (
            Time(starttime, format="isot", scale="utc") if starttime else Time.now()
        )
    except ValueError as e:
        raise ValueError(
            f"Invalid start time: {starttime}. Expected ISO format (e.g., '2025-01-01T00:00:00')."
        ) from e

    # Create EarthLocation object
    location = EarthLocation(lat=lat * u.deg, lon=lon * u.deg, height=height * u.m)

    # Debug output
    print(
        f"{_c('Observation Location:', _BOLD + _CYAN)} Latitude={lat}, Longitude={lon}, Height={height} meters"
    )
    print(f"{_c('Observation Start Time:', _BOLD + _CYAN)} {obstime_start.isot}")

    return location, obstime_start
