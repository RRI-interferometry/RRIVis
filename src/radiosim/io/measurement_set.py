# radiosim/io/measurement_set.py
"""Measurement Set I/O for RadioSim.

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
Write simulation results to MS through the public Simulator boundary:

>>> simulator.run(progress=False)
>>> simulator.save("output", format="ms", filename="simulation")

Read MS back into memory:

>>> from radiosim.io.measurement_set import read_ms
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
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from radiosim.core.instrument import (
        ResolvedBaselineSelection,
        ResolvedInstrument,
    )

import numpy as np
from astropy.coordinates import EarthLocation
from astropy.time import Time

# Check for pyuvdata availability
try:
    from pyuvdata import Telescope, UVData
    from pyuvdata.utils import ECEF_from_ENU

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
            "Or install RadioSim with MS support:\n"
            "  pip install radiosim[ms]"
        )


def write_ms(
    output_path: str | Path,
    visibilities: dict[tuple[int, int], dict[str, np.ndarray] | np.ndarray],
    frequencies: np.ndarray,
    instrument: ResolvedInstrument,
    selection: ResolvedBaselineSelection,
    obstime: Time,
    polarizations: list[str] | None = None,
    channel_width: float | None = None,
    integration_time: float = 1.0,
    overwrite: bool = False,
) -> Path:
    """Write canonical instrument state and selected visibilities to an MS.

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
    instrument : ResolvedInstrument
        Complete canonical telescope identity, location, antennas, and diameters.
    selection : ResolvedBaselineSelection
        Exact canonical baseline subset represented by ``visibilities``.
    obstime : Time
        Observation time (can be single Time or array of Times).
    polarizations : list of str, optional
        List of polarization labels. If None, auto-detected from visibility keys.
        Common options: ["XX", "XY", "YX", "YY"] or ["RR", "RL", "LR", "LL"].
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

    from radiosim.core.instrument import (
        ResolvedBaselineSelection,
        ResolvedInstrument,
    )

    if type(instrument) is not ResolvedInstrument:
        raise TypeError("instrument must be a ResolvedInstrument")
    if type(selection) is not ResolvedBaselineSelection:
        raise TypeError("selection must be a ResolvedBaselineSelection")
    if selection.provenance.instrument_sha256 != (
        instrument.provenance.instrument_sha256
    ):
        raise ValueError("selection does not belong to instrument")

    output_path = Path(output_path)

    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"MS already exists: {output_path}\nUse overwrite=True to replace it."
        )

    # Remove existing MS if overwriting
    if output_path.exists() and overwrite:
        import shutil

        shutil.rmtree(output_path)

    antenna_numbers = [antenna.id.number for antenna in instrument.antennas]
    antenna_names = [antenna.id.name for antenna in instrument.antennas]
    antenna_diameters = [antenna.diameter_m for antenna in instrument.antennas]
    antenna_positions_enu = np.asarray(
        [antenna.position_enu_m for antenna in instrument.antennas],
        dtype=np.float64,
    )

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
    baseline_list = list(selection.provenance.selected_ids)
    if set(visibilities) != set(baseline_list):
        raise ValueError(
            "visibilities must contain exactly the selected canonical baseline pairs"
        )

    n_baselines = len(baseline_list)
    n_blts = n_baselines * n_times

    # Create antenna pair arrays
    ant_1_array = np.zeros(n_blts, dtype=int)
    ant_2_array = np.zeros(n_blts, dtype=int)
    time_array = np.zeros(n_blts)

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

            bl_key = (ant1, ant2)
            vis_data = visibilities[bl_key]
            if isinstance(vis_data, dict):
                for p_idx, pol in enumerate(polarizations):
                    values = vis_data.get(pol)
                    scale = 1.0
                    if values is None and pol == "XX" and "I" in vis_data:
                        values = vis_data["I"]
                        scale = 0.5
                    if values is None:
                        continue
                    array = np.asarray(values)
                    if array.ndim >= 2:
                        array = array[_t_idx]
                    data_array[blt_idx, :, p_idx] = np.resize(array, n_freqs) * scale
            else:
                array = np.asarray(vis_data)
                if array.ndim >= 2:
                    array = array[_t_idx]
                data_array[blt_idx, :, 0] = np.resize(array, n_freqs)

            blt_idx += 1

    location = EarthLocation.from_geocentric(
        *instrument.location.itrs_xyz_m,
        unit="m",
    )
    absolute_ecef = np.asarray(
        ECEF_from_ENU(antenna_positions_enu, center_loc=location),
        dtype=np.float64,
    )
    center_ecef = np.asarray(instrument.location.itrs_xyz_m, dtype=np.float64)
    relative_ecef = absolute_ecef - center_ecef
    antenna_positions_ecef = {
        number: np.array(relative_ecef[index], dtype=np.float64, copy=True)
        for index, number in enumerate(antenna_numbers)
    }

    # Create Telescope object
    telescope = Telescope.new(
        name=instrument.name,
        location=location,
        antenna_positions=antenna_positions_ecef,
        antenna_names=antenna_names,
        antenna_numbers=antenna_numbers,
        antenna_diameters=antenna_diameters,
        instrument=instrument.name,
        update_from_known=False,
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
        update_telescope_from_known=False,
    )

    # Set the data
    uvd.data_array = data_array
    uvd.flag_array = flag_array
    uvd.nsample_array = nsample_array

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
    >>> from radiosim.io.measurement_set import read_ms
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
    >>> from radiosim.io.measurement_set import read_ms_dask
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
    >>> from radiosim.io.measurement_set import ms_info
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
