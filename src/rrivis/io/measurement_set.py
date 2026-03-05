# rrivis/io/measurement_set.py
"""Measurement Set I/O for RRIvis.

This module provides functions for reading and writing CASA Measurement Set
format, enabling interoperability with standard radio astronomy tools like
CASA, QuartiCal, WSClean, and other calibration pipelines.

The implementation uses pyuvdata as the primary backend, with optional
dask-ms support for very large datasets.

Requirements
------------
- pyuvdata >= 2.4 (required)
- python-casacore >= 3.5 (required for MS format)
- dask-ms >= 0.2.20 (optional, for large datasets)

Examples
--------
Write simulation results to MS:

>>> from rrivis.io.measurement_set import write_ms
>>> write_ms(
...     output_path="simulation.ms",
...     visibilities=results["visibilities"],
...     frequencies=results["frequencies"],
...     antennas=results["antennas"],
...     baselines=results["baselines"],
...     location=results["location"],
...     obstime=results["obstime"],
... )

Read MS back into memory:

>>> from rrivis.io.measurement_set import read_ms
>>> data = read_ms("simulation.ms")
>>> print(data["visibilities"].shape)

References
----------
- CASA Measurement Set format: https://casa.nrao.edu/
- pyuvdata documentation: https://pyuvdata.readthedocs.io/
- dask-ms documentation: https://dask-ms.readthedocs.io/
- Africanus I paper (dask-ms): https://arxiv.org/abs/2412.12052
"""

from __future__ import annotations

import importlib.util
import warnings
from pathlib import Path
from typing import Any

import numpy as np
from astropy.coordinates import EarthLocation
from astropy.time import Time

# Check for pyuvdata availability
try:
    from pyuvdata import Telescope, UVData

    PYUVDATA_AVAILABLE = True
except ImportError:
    PYUVDATA_AVAILABLE = False
    warnings.warn(
        "pyuvdata not available. Install with: pip install pyuvdata\n"
        "MS I/O functionality will be disabled.",
        stacklevel=2,
    )

# Check for python-casacore availability
CASACORE_AVAILABLE = importlib.util.find_spec("casacore.tables") is not None
if not CASACORE_AVAILABLE and PYUVDATA_AVAILABLE:
    warnings.warn(
        "python-casacore not available. Install with: pip install python-casacore\n"
        "MS format support will be disabled.",
        stacklevel=2,
    )

# Check for dask-ms availability (optional)
try:
    from daskms import xds_from_ms

    DASKMS_AVAILABLE = True
except ImportError:
    DASKMS_AVAILABLE = False


def _check_ms_dependencies():
    """Check that MS dependencies are available."""
    if not PYUVDATA_AVAILABLE:
        raise ImportError(
            "pyuvdata is required for MS I/O. Install with:\n  pip install pyuvdata"
        )
    if not CASACORE_AVAILABLE:
        raise ImportError(
            "python-casacore is required for MS format. Install with:\n"
            "  pip install python-casacore\n"
            "Or install RRIvis with MS support:\n"
            "  pip install rrivis[ms]"
        )


def write_ms(
    output_path: str | Path,
    visibilities: dict[tuple[int, int], dict[str, np.ndarray] | np.ndarray],
    frequencies: np.ndarray,
    antennas: dict[str, dict[str, Any]],
    baselines: dict[tuple[int, int], dict[str, Any]],
    location: EarthLocation,
    obstime: Time,
    telescope_name: str = "RRIvis",
    instrument_name: str = "RRIvis Simulator",
    polarizations: list[str] | None = None,
    phase_center_ra: float = 0.0,
    phase_center_dec: float = -30.0,
    channel_width: float | None = None,
    integration_time: float = 1.0,
    overwrite: bool = False,
) -> Path:
    """Write visibility data to CASA Measurement Set format.

    This function converts RRIvis simulation results to the standard
    Measurement Set format used by CASA, QuartiCal, WSClean, and other
    radio astronomy software.

    Parameters
    ----------
    output_path : str or Path
        Path to output MS file (will be a directory).
    visibilities : dict
        Dictionary mapping baseline tuples (ant1, ant2) to visibility data.
        Each value can be either:
        - Dict with polarization keys ("XX", "XY", "YX", "YY" or "I", "Q", "U", "V")
        - Single numpy array (assumed to be Stokes I or XX)
    frequencies : np.ndarray
        Array of observation frequencies in Hz.
    antennas : dict
        Dictionary of antenna information with keys like "0000", "0001".
        Each entry should have "Number", "Position" (ENU), and optionally "diameter".
    baselines : dict
        Dictionary of baseline information from generate_baselines().
    location : EarthLocation
        Observatory location.
    obstime : Time
        Observation time (can be single Time or array of Times).
    telescope_name : str, optional
        Name of the telescope (default: "RRIvis").
    instrument_name : str, optional
        Name of the instrument (default: "RRIvis Simulator").
    polarizations : list of str, optional
        List of polarization labels. If None, auto-detected from visibility keys.
        Common options: ["XX", "XY", "YX", "YY"] or ["RR", "RL", "LR", "LL"].
    phase_center_ra : float, optional
        Right ascension of phase center in degrees (default: 0.0).
    phase_center_dec : float, optional
        Declination of phase center in degrees (default: -30.0).
    channel_width : float, optional
        Channel width in Hz. If None, inferred from frequency array.
    integration_time : float, optional
        Integration time in seconds (default: 1.0).
    overwrite : bool, optional
        Overwrite existing MS file (default: False).

    Returns
    -------
    Path
        Path to the created MS file.

    Raises
    ------
    ImportError
        If pyuvdata or python-casacore is not installed.
    FileExistsError
        If output path exists and overwrite=False.
    ValueError
        If input data is invalid.

    Examples
    --------
    >>> from rrivis import Simulator
    >>> sim = Simulator.from_config("config.yaml")
    >>> results = sim.run()
    >>> from rrivis.io.measurement_set import write_ms
    >>> write_ms(
    ...     "output.ms",
    ...     visibilities=results["visibilities"],
    ...     frequencies=results["frequencies"],
    ...     antennas=results["antennas"],
    ...     baselines=results["baselines"],
    ...     location=results["location"],
    ...     obstime=results["obstime"],
    ... )

    Notes
    -----
    The MS format is the standard for radio interferometry data exchange.
    After creating an MS, you can:

    - View in CASA: ``casabrowser output.ms``
    - Calibrate with QuartiCal: ``goquartical output.ms``
    - Image with WSClean: ``wsclean -name image output.ms``

    See Also
    --------
    read_ms : Read visibility data from MS format.
    """
    _check_ms_dependencies()

    output_path = Path(output_path)

    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"MS already exists: {output_path}\nUse overwrite=True to replace it."
        )

    # Remove existing MS if overwriting
    if output_path.exists() and overwrite:
        import shutil

        shutil.rmtree(output_path)

    # Parse antenna information
    antenna_positions = {}
    antenna_names = []
    antenna_diameters = []

    # Sort by antenna number to ensure consistent ordering
    sorted_ant_keys = sorted(
        antennas.keys(), key=lambda k: antennas[k].get("Number", 0)
    )

    for ant_key in sorted_ant_keys:
        ant_data = antennas[ant_key]
        ant_num = ant_data.get("Number", int(ant_key) if ant_key.isdigit() else 0)
        position = np.array(ant_data["Position"], dtype=float)
        diameter = ant_data.get("diameter", 14.0)

        antenna_positions[ant_num] = position
        antenna_names.append(f"ANT{ant_num:03d}")
        antenna_diameters.append(diameter)

    len(antenna_positions)

    # Determine polarizations from visibility data
    sample_vis = next(iter(visibilities.values()))
    if isinstance(sample_vis, dict):
        if polarizations is None:
            # Auto-detect polarization type
            vis_keys = set(sample_vis.keys())
            if vis_keys & {"XX", "XY", "YX", "YY"}:
                polarizations = ["XX", "XY", "YX", "YY"]
            elif vis_keys & {"RR", "RL", "LR", "LL"}:
                polarizations = ["RR", "RL", "LR", "LL"]
            elif vis_keys & {"I", "Q", "U", "V"}:
                # Convert Stokes to linear feeds for MS compatibility
                polarizations = ["XX", "XY", "YX", "YY"]
            else:
                polarizations = list(vis_keys)[:4]
    else:
        # Single array - assume unpolarized (XX only or Stokes I)
        polarizations = ["XX"]

    n_pols = len(polarizations)
    n_freqs = len(frequencies)

    # Calculate channel width
    if channel_width is None:
        if n_freqs > 1:
            channel_width = np.abs(frequencies[1] - frequencies[0])
        else:
            channel_width = 1e6  # Default 1 MHz

    # Handle time - ensure it's an array
    if isinstance(obstime, Time):
        if obstime.isscalar:
            times = np.array([obstime.jd])
        else:
            times = obstime.jd
    else:
        times = np.array([obstime])

    n_times = len(times)

    # Build baseline list and data arrays
    baseline_list = []
    for bl_key in sorted(visibilities.keys()):
        ant1, ant2 = bl_key
        baseline_list.append((ant1, ant2))

    n_baselines = len(baseline_list)
    n_blts = n_baselines * n_times

    # Create antenna pair arrays
    ant_1_array = np.zeros(n_blts, dtype=int)
    ant_2_array = np.zeros(n_blts, dtype=int)
    time_array = np.zeros(n_blts)
    uvw_array = np.zeros((n_blts, 3))

    # Create data array
    # Shape: (Nblts, Nfreqs, Npols)
    data_array = np.zeros((n_blts, n_freqs, n_pols), dtype=np.complex128)
    flag_array = np.zeros((n_blts, n_freqs, n_pols), dtype=bool)
    nsample_array = np.ones((n_blts, n_freqs, n_pols), dtype=float)

    # Fill arrays
    blt_idx = 0
    for _t_idx, time_jd in enumerate(times):
        for _bl_idx, (ant1, ant2) in enumerate(baseline_list):
            ant_1_array[blt_idx] = ant1
            ant_2_array[blt_idx] = ant2
            time_array[blt_idx] = time_jd

            # Get UVW from baseline info
            bl_key = (ant1, ant2)
            if bl_key in baselines:
                uvw = baselines[bl_key].get("BaselineVector", np.zeros(3))
                uvw_array[blt_idx] = uvw
            else:
                # Calculate from antenna positions
                pos1 = antenna_positions.get(ant1, np.zeros(3))
                pos2 = antenna_positions.get(ant2, np.zeros(3))
                uvw_array[blt_idx] = pos2 - pos1

            # Get visibility data for this baseline
            vis_data = visibilities.get(bl_key, visibilities.get((ant2, ant1), None))
            if vis_data is not None:
                if isinstance(vis_data, dict):
                    for p_idx, pol in enumerate(polarizations):
                        if pol in vis_data:
                            vis_values = vis_data[pol]
                            if np.isscalar(vis_values):
                                data_array[blt_idx, :, p_idx] = vis_values
                            elif len(vis_values) == n_freqs:
                                data_array[blt_idx, :, p_idx] = vis_values
                            else:
                                # Handle shape mismatch
                                data_array[blt_idx, :, p_idx] = np.resize(
                                    vis_values, n_freqs
                                )
                        elif pol == "XX" and "I" in vis_data:
                            # Convert Stokes I to XX (approximate)
                            vis_values = vis_data["I"]
                            if np.isscalar(vis_values):
                                data_array[blt_idx, :, p_idx] = vis_values / 2
                            else:
                                data_array[blt_idx, :, p_idx] = (
                                    np.resize(vis_values, n_freqs) / 2
                                )
                else:
                    # Single array - assume first polarization
                    if np.isscalar(vis_data):
                        data_array[blt_idx, :, 0] = vis_data
                    else:
                        data_array[blt_idx, :, 0] = np.resize(vis_data, n_freqs)

            blt_idx += 1

    # Convert ENU positions to ECEF for pyuvdata
    # pyuvdata expects antenna_positions relative to telescope location in ECEF
    lat = location.lat.rad
    lon = location.lon.rad

    # Rotation matrix from ENU to ECEF
    R_enu_to_ecef = np.array(
        [
            [-np.sin(lon), -np.sin(lat) * np.cos(lon), np.cos(lat) * np.cos(lon)],
            [np.cos(lon), -np.sin(lat) * np.sin(lon), np.cos(lat) * np.sin(lon)],
            [0, np.cos(lat), np.sin(lat)],
        ]
    )

    # Convert antenna positions from ENU to ECEF (relative to array center)
    antenna_positions_ecef = {}
    for ant_num, enu_pos in antenna_positions.items():
        ecef_rel = R_enu_to_ecef @ enu_pos
        antenna_positions_ecef[ant_num] = ecef_rel

    # Create Telescope object
    telescope = Telescope.new(
        name=telescope_name,
        location=location,
        antenna_positions=antenna_positions_ecef,
        antenna_names=antenna_names,
        antenna_diameters=antenna_diameters,
        instrument=instrument_name,
    )

    # Create UVData object
    uvd = UVData.new(
        freq_array=frequencies,
        polarization_array=polarizations,
        times=times,
        antpairs=baseline_list,
        telescope=telescope,
        do_blt_outer=True,  # Cartesian product of times × baselines
        time_axis_faster_than_bls=False,
    )

    # Set the data
    uvd.data_array = data_array
    uvd.flag_array = flag_array
    uvd.nsample_array = nsample_array

    # Set UVW coordinates (in meters)
    uvd.uvw_array = uvw_array

    # Set phase center
    uvd.phase_center_ra = phase_center_ra * np.pi / 180  # Convert to radians
    uvd.phase_center_dec = phase_center_dec * np.pi / 180

    # Set integration time
    uvd.integration_time = np.full(n_blts, integration_time)

    # Set channel width
    uvd.channel_width = np.full(n_freqs, channel_width)

    # Validate the UVData object
    try:
        uvd.check()
    except Exception as e:
        warnings.warn(f"UVData validation warning: {e}", stacklevel=2)

    # Write to MS format
    uvd.write_ms(str(output_path), clobber=overwrite, force_phase=True)

    return output_path


def read_ms(
    input_path: str | Path,
    data_column: str = "DATA",
    include_flags: bool = True,
) -> dict[str, Any]:
    """Read visibility data from CASA Measurement Set format.

    Parameters
    ----------
    input_path : str or Path
        Path to input MS file.
    data_column : str, optional
        Name of the data column to read (default: "DATA").
        Other options: "CORRECTED_DATA", "MODEL_DATA".
    include_flags : bool, optional
        Whether to include flag data (default: True).

    Returns
    -------
    dict
        Dictionary containing:
        - visibilities : np.ndarray with shape (Nblts, Nfreqs, Npols)
        - frequencies : np.ndarray with shape (Nfreqs,)
        - times : np.ndarray with shape (Ntimes,) in JD
        - ant_1_array : np.ndarray with shape (Nblts,)
        - ant_2_array : np.ndarray with shape (Nblts,)
        - uvw_array : np.ndarray with shape (Nblts, 3)
        - polarizations : list of str
        - flags : np.ndarray (if include_flags=True)
        - antenna_names : list of str
        - antenna_positions : np.ndarray with shape (Nants, 3)
        - telescope_name : str
        - telescope_location : EarthLocation

    Raises
    ------
    ImportError
        If pyuvdata or python-casacore is not installed.
    FileNotFoundError
        If input MS does not exist.

    Examples
    --------
    >>> from rrivis.io.measurement_set import read_ms
    >>> data = read_ms("observation.ms")
    >>> print(f"Shape: {data['visibilities'].shape}")
    >>> print(f"Frequencies: {data['frequencies'] / 1e6} MHz")

    Notes
    -----
    This function uses pyuvdata to read the MS file, which handles
    all the complexity of the MS format including subtables.

    See Also
    --------
    write_ms : Write visibility data to MS format.
    """
    _check_ms_dependencies()

    input_path = Path(input_path)

    if not input_path.exists():
        raise FileNotFoundError(f"MS not found: {input_path}")

    # Read MS using pyuvdata
    uvd = UVData()
    uvd.read(str(input_path), data_column=data_column)

    # Extract data
    result = {
        "visibilities": uvd.data_array,
        "frequencies": uvd.freq_array.flatten(),
        "times": np.unique(uvd.time_array),
        "time_array": uvd.time_array,
        "ant_1_array": uvd.ant_1_array,
        "ant_2_array": uvd.ant_2_array,
        "uvw_array": uvd.uvw_array,
        "polarizations": uvd.get_pols(),
        "antenna_names": uvd.telescope.antenna_names,
        "antenna_positions": uvd.telescope.antenna_positions,
        "telescope_name": uvd.telescope.name,
        "telescope_location": uvd.telescope.location,
        "n_antennas": uvd.telescope.Nants,
        "n_baselines": uvd.Nbls,
        "n_times": uvd.Ntimes,
        "n_frequencies": uvd.Nfreqs,
        "n_polarizations": uvd.Npols,
    }

    if include_flags:
        result["flags"] = uvd.flag_array
        result["nsample_array"] = uvd.nsample_array

    return result


def read_ms_dask(
    input_path: str | Path,
    columns: list[str] | None = None,
    chunks: dict[str, int] | None = None,
) -> list[Any]:
    """Read MS using dask-ms for large datasets.

    This function uses dask-ms to read MS files lazily, which is more
    memory-efficient for very large datasets.

    Parameters
    ----------
    input_path : str or Path
        Path to input MS file.
    columns : list of str, optional
        Columns to read. If None, reads common columns.
    chunks : dict, optional
        Chunking specification for dask arrays.
        Default: {"row": 100000}

    Returns
    -------
    list of xarray.Dataset
        List of xarray datasets, one per DATA_DESC_ID partition.

    Raises
    ------
    ImportError
        If dask-ms is not installed.

    Examples
    --------
    >>> from rrivis.io.measurement_set import read_ms_dask
    >>> datasets = read_ms_dask("large_observation.ms")
    >>> # Process lazily
    >>> for ds in datasets:
    ...     data = ds.DATA.data  # dask array
    ...     result = data.mean().compute()  # compute on demand

    Notes
    -----
    dask-ms is recommended for MS files larger than available RAM.
    Install with: pip install dask-ms

    See Also
    --------
    read_ms : Read MS into memory using pyuvdata.
    """
    if not DASKMS_AVAILABLE:
        raise ImportError(
            "dask-ms is required for lazy reading. Install with:\n  pip install dask-ms"
        )

    input_path = Path(input_path)

    if not input_path.exists():
        raise FileNotFoundError(f"MS not found: {input_path}")

    if columns is None:
        columns = ["DATA", "FLAG", "UVW", "TIME", "ANTENNA1", "ANTENNA2"]

    if chunks is None:
        chunks = {"row": 100000}

    # Read MS using dask-ms
    datasets = xds_from_ms(str(input_path), columns=columns, chunks=chunks)

    return list(datasets)


def ms_info(input_path: str | Path) -> dict[str, Any]:
    """Get summary information about a Measurement Set.

    Parameters
    ----------
    input_path : str or Path
        Path to input MS file.

    Returns
    -------
    dict
        Dictionary with MS summary information including:
        - n_rows: Total number of rows
        - n_antennas: Number of antennas
        - n_baselines: Number of baselines
        - n_times: Number of time stamps
        - n_channels: Number of frequency channels
        - n_polarizations: Number of polarizations
        - frequencies: Frequency array (Hz)
        - time_range: (min_time, max_time) in MJD
        - telescope_name: Name of telescope
        - antenna_names: List of antenna names

    Examples
    --------
    >>> from rrivis.io.measurement_set import ms_info
    >>> info = ms_info("observation.ms")
    >>> print(f"Antennas: {info['n_antennas']}")
    >>> print(f"Channels: {info['n_channels']}")
    >>> print(f"Time range: {info['time_range']} MJD")
    """
    _check_ms_dependencies()

    input_path = Path(input_path)

    if not input_path.exists():
        raise FileNotFoundError(f"MS not found: {input_path}")

    # Use pyuvdata for quick info (reads only metadata)
    uvd = UVData()

    # Read just header info
    uvd.read(str(input_path), read_data=False)

    time_jd = uvd.time_array
    time_mjd = time_jd - 2400000.5  # JD to MJD

    info = {
        "n_rows": uvd.Nblts,
        "n_antennas": uvd.telescope.Nants,
        "n_baselines": uvd.Nbls,
        "n_times": uvd.Ntimes,
        "n_channels": uvd.Nfreqs,
        "n_polarizations": uvd.Npols,
        "frequencies": uvd.freq_array.flatten(),
        "time_range": (time_mjd.min(), time_mjd.max()),
        "telescope_name": uvd.telescope.name,
        "antenna_names": uvd.telescope.antenna_names,
        "polarizations": uvd.get_pols(),
        "channel_width": uvd.channel_width[0]
        if uvd.channel_width is not None
        else None,
        "integration_time": uvd.integration_time[0]
        if uvd.integration_time is not None
        else None,
    }

    return info
