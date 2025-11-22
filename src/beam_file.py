# beam_file.py
"""
Beam FITS file handling using pyuvdata UVBeam.

CRITICAL FIXES APPLIED:
1. za_range must be in DEGREES, not radians (UVBeam API requirement)
2. Jones matrix index order: rows=feeds, cols=sky_basis (RIME standard)
3. Azimuth convention conversion: Astropy (N=0) → UVBeam (E=0)
4. beam_ids None check in BeamManager (avoid crash)
5. Extensive documentation on conventions

Loads beam files, performs coordinate transformations, and interpolates
to provide 2×2 Jones matrices for visibility calculations.

ASSUMPTIONS (Phase 1):
- Beam E-field basis aligned with linear feed basis (X, Y)
- No explicit basis vector rotation (TODO: use basis_vector_array)
- See beam.basis_vector_array for future full implementation

References:
- pyuvdata UVBeam documentation
- Price 2015: Basis rotations in RIME
- Carozzi & Woan 2009: Generalized measurement equation
"""

import numpy as np
from pyuvdata import UVBeam
from astropy.coordinates import SkyCoord, AltAz, EarthLocation
from astropy.time import Time
import astropy.units as u
import os


def astropy_az_to_uvbeam_az(astropy_az_rad):
    """
    Convert Astropy azimuth to UVBeam azimuth convention.

    CRITICAL COORDINATE CONVENTION FIX!

    Astropy AltAz: North=0°, East=90° (measured from North toward East)
    UVBeam:        East=0°, North=90° (measured from East toward North)

    Conversion: uvbeam_az = π/2 - astropy_az

    Parameters
    ----------
    astropy_az_rad : float or array
        Astropy azimuth in radians (North=0, East=π/2)

    Returns
    -------
    uvbeam_az_rad : float or array
        UVBeam azimuth in radians (East=0, North=π/2)
        Wrapped to [0, 2π]

    Examples
    --------
    >>> # North in Astropy (0°) → North in UVBeam (90°)
    >>> astropy_az_to_uvbeam_az(0.0)
    1.5707963...  # π/2

    >>> # East in Astropy (90°) → East in UVBeam (0°)
    >>> astropy_az_to_uvbeam_az(np.pi/2)
    0.0

    >>> # South in Astropy (180°) → West in UVBeam (270°)
    >>> astropy_az_to_uvbeam_az(np.pi)
    4.7123889...  # 3π/2

    Notes
    -----
    Without this conversion, the beam would be rotated 90° relative
    to the sky, producing completely wrong polarization!

    References
    ----------
    - pyuvdata UVBeam docs: "az runs from East to North"
    - Astropy AltAz: "Azimuth is measured from North toward East"
    """
    uvbeam_az_rad = np.pi / 2 - astropy_az_rad

    # Wrap to [0, 2π]
    uvbeam_az_rad = np.mod(uvbeam_az_rad, 2 * np.pi)

    return uvbeam_az_rad


class BeamFITSHandler:
    """
    Handler for a single beam FITS file.

    Provides Jones matrix interpolation for given sky positions.

    ASSUMPTIONS (Phase 1):
    - Beam E-field basis aligned with linear feed basis (X, Y)
    - No explicit basis vector rotation needed
    - Future: implement full basis handling using beam.basis_vector_array

    Attributes
    ----------
    beam : UVBeam
        Loaded beam object from pyuvdata
    beam_file_path : str
        Path to beam FITS file
    freq_interp_kind : str
        Frequency interpolation method (default: 'cubic')
    """

    def __init__(self, beam_file_path, config, logger):
        """
        Load beam FITS file with optional partial reading.

        Parameters
        ----------
        beam_file_path : str
            Path to .beamfits file
        config : dict
            RRIvis configuration dictionary
        logger : logging.Logger
            Logger instance

        Raises
        ------
        FileNotFoundError
            If beam file doesn't exist
        ValueError
            If beam file is invalid or unsupported format
        """
        self.logger = logger
        self.config = config
        self.beam_file_path = beam_file_path

        # Check file exists
        if not os.path.exists(beam_file_path):
            raise FileNotFoundError(f"Beam file not found: {beam_file_path}")

        # Load beam file
        self.beam = UVBeam()
        self._load_beam()

        # Validate beam
        self._validate_beam()

        # Cache interpolation settings
        self.freq_interp_kind = config["beams"].get("beam_freq_interp", "cubic")

    def _load_beam(self):
        """
        Load beam with optional partial reading for memory efficiency.

        CRITICAL FIX: za_range must be in DEGREES, not radians!
        """
        # Get observation frequency range (Hz)
        freq_min = self.config["obs_frequency"]["freq_min_MHz"] * 1e6
        freq_max = self.config["obs_frequency"]["freq_max_MHz"] * 1e6
        freq_buffer = self.config["beams"].get("beam_freq_buffer_mhz", 10.0) * 1e6

        # Get zenith angle limit IN DEGREES (CRITICAL!)
        za_max_deg = self.config["beams"].get("beam_za_max_deg", 90.0)

        self.logger.info(f"Loading beam from: {self.beam_file_path}")
        self.logger.info(
            f"  Frequency range: {freq_min/1e6:.1f} - {freq_max/1e6:.1f} MHz "
            f"(buffer: {freq_buffer/1e6:.1f} MHz)"
        )
        self.logger.info(f"  Zenith angle limit: {za_max_deg:.1f}° (DEGREES for UVBeam)")

        try:
            # CRITICAL: za_range expects DEGREES, not radians!
            # BUG FIX: Was passing radians → loaded wrong angles!
            self.beam.read_beamfits(
                self.beam_file_path,
                freq_range=[freq_min - freq_buffer, freq_max + freq_buffer],
                za_range=[0, za_max_deg],  # DEGREES!
            )

            self.logger.info("  Loaded successfully")
            self.logger.info(f"  Beam type: {self.beam.beam_type}")
            self.logger.info(f"  Coordinate system: {self.beam.pixel_coordinate_system}")
            self.logger.info(f"  Frequencies: {self.beam.Nfreqs}")
            self.logger.info(f"  Basis vectors: {self.beam.Naxes_vec}")

            if hasattr(self.beam, "Nfeeds"):
                self.logger.info(f"  Feeds: {self.beam.Nfeeds}")

        except Exception as e:
            self.logger.error(f"Failed to load beam file: {e}")
            self.logger.error("Check:")
            self.logger.error("  1. File exists and is valid BeamFITS format")
            self.logger.error("  2. Frequency range overlaps with observation")
            self.logger.error("  3. za_max_deg is in degrees (not radians!)")
            raise

    def _validate_beam(self):
        """Validate beam is suitable for use."""
        # Check if E-field beam
        if self.beam.beam_type != "efield":
            self.logger.warning(
                f"Beam type is '{self.beam.beam_type}', expected 'efield' "
                "for full Jones matrix"
            )
            self.logger.warning(
                "Continuing, but polarization may not be fully captured"
            )

        # Check coordinate system
        if self.beam.pixel_coordinate_system not in ["az_za", "healpix"]:
            raise ValueError(
                f"Unsupported coordinate system: {self.beam.pixel_coordinate_system}"
            )

        # Log basis vector info (for future full implementation)
        if (
            hasattr(self.beam, "basis_vector_array")
            and self.beam.basis_vector_array is not None
        ):
            self.logger.info("  Beam has basis_vector_array (not used in Phase 1)")
            self.logger.info(
                "  TODO: Implement full basis rotation for arbitrary orientations"
            )

    def get_jones_matrix(self, alt_rad, az_rad, freq_hz, location=None, time=None):
        """
        Get 2×2 Jones matrix at given position(s).

        CRITICAL FIXES APPLIED:
        1. Azimuth conversion: Astropy (N=0) → UVBeam (E=0)
        2. Jones matrix index order: rows=feeds, cols=sky_basis

        Parameters
        ----------
        alt_rad : float or array
            Altitude angle(s) in radians
        az_rad : float or array
            Azimuth angle(s) in radians (ASTROPY CONVENTION: N=0, E=π/2)
        freq_hz : float
            Frequency in Hz (scalar for now)
        location : EarthLocation, optional
            Observatory location (for future parallactic angle)
        time : Time, optional
            Observation time (for future parallactic angle)

        Returns
        -------
        jones : ndarray
            2×2 complex Jones matrix
            Shape: (2, 2) for scalar inputs, or (Nsources, 2, 2) for arrays
            Convention: jones[feed_idx, sky_basis_idx]
            - jones[0, :]: X feed response to sky basis components
            - jones[1, :]: Y feed response to sky basis components

        Notes
        -----
        The Jones matrix follows RIME standard:
        - Rows: feed polarizations (0=X, 1=Y for linear)
        - Columns: sky basis components (θ, φ or aligned)

        For RIME visibility calculation:
            V = J_i @ C @ J_j^H

        Examples
        --------
        >>> handler = BeamFITSHandler('beam.beamfits', config, logger)
        >>> # Single source
        >>> jones = handler.get_jones_matrix(
        ...     alt_rad=0.5, az_rad=0.0, freq_hz=150e6
        ... )
        >>> jones.shape  # → (2, 2)

        >>> # Multiple sources (vectorized)
        >>> alts = np.array([0.3, 0.5, 0.7])
        >>> azs = np.array([0.0, np.pi/4, np.pi/2])
        >>> jones_all = handler.get_jones_matrix(alts, azs, 150e6)
        >>> jones_all.shape  # → (3, 2, 2)
        """
        # CRITICAL FIX: Convert Astropy azimuth to UVBeam convention
        # Without this, beam would be rotated 90° relative to sky!
        az_uvbeam = astropy_az_to_uvbeam_az(az_rad)

        # Convert alt/az to beam coordinates
        if self.beam.pixel_coordinate_system == "az_za":
            za_rad = np.pi / 2 - alt_rad
            axis1_array = az_uvbeam  # Use converted azimuth
            axis2_array = za_rad

        elif self.beam.pixel_coordinate_system == "healpix":
            theta = np.pi / 2 - alt_rad
            phi = az_uvbeam  # Use converted azimuth
            # For HEALPix, interp uses theta/phi directly
            axis1_array = az_uvbeam
            axis2_array = theta

        # Interpolate beam
        # UVBeam.interp returns:
        # interp_data: (Naxes_vec, Nspws, Nfeeds, Nfreqs, Npoints)
        # interp_basis_vector: basis vector info (not used in Phase 1)

        try:
            interp_data, interp_basis_vector = self.beam.interp(
                az_array=np.atleast_1d(az_uvbeam),
                za_array=np.atleast_1d(za_rad if self.beam.pixel_coordinate_system == "az_za" else theta),
                freq_array=np.atleast_1d(freq_hz),
                freq_interp_kind=self.freq_interp_kind,
                reuse_spline=True,  # Performance optimization
            )

        except Exception as e:
            self.logger.error(f"Beam interpolation failed: {e}")
            self.logger.error(f"  alt_rad: {alt_rad}, az_rad: {az_rad}")
            self.logger.error(f"  az_uvbeam: {az_uvbeam}")
            self.logger.error(f"  freq_hz: {freq_hz}")
            raise

        # Extract Jones matrix with CORRECT index ordering
        # CRITICAL FIX: rows = feeds (X, Y), columns = sky basis (θ, φ)
        #
        # interp_data[axis_vec, spw, feed, freq, point]
        # axis_vec: 0=θ or x, 1=φ or y (depending on beam basis)
        # feed: 0=X, 1=Y for linear feeds
        #
        # RIME convention: J[feed, sky_basis]
        # J = [[E_Xθ, E_Xφ],   (feed X response to θ and φ components)
        #      [E_Yθ, E_Yφ]]   (feed Y response to θ and φ components)
        #
        # BUG FIX: Was [basis, feed] → WRONG!
        # Now: [feed, basis] → CORRECT!

        if interp_data.shape[-1] == 1:
            # Single point
            jones = np.array(
                [
                    # Row 0: X feed responses to basis 0 and basis 1
                    [interp_data[0, 0, 0, 0, 0], interp_data[1, 0, 0, 0, 0]],
                    # Row 1: Y feed responses to basis 0 and basis 1
                    [interp_data[0, 0, 1, 0, 0], interp_data[1, 0, 1, 0, 0]],
                ],
                dtype=complex,
            )
        else:
            # Multiple points
            Npoints = interp_data.shape[-1]
            jones = np.zeros((Npoints, 2, 2), dtype=complex)

            # Fill with correct index order: [source, feed, basis]
            jones[:, 0, 0] = interp_data[0, 0, 0, 0, :]  # X feed, basis 0
            jones[:, 0, 1] = interp_data[1, 0, 0, 0, :]  # X feed, basis 1
            jones[:, 1, 0] = interp_data[0, 0, 1, 0, :]  # Y feed, basis 0
            jones[:, 1, 1] = interp_data[1, 0, 1, 0, :]  # Y feed, basis 1

        return jones


class BeamManager:
    """
    Manages multiple beams and antenna-to-beam assignments.

    Supports three modes:
    1. Analytic: No beam files (use existing analytic beams)
    2. Shared: All antennas use same beam file
    3. Per-antenna: Different beams per antenna (from layout or config)

    FIXED: Logic bug in has_beam_ids_in_layout check (was missing None check)

    Attributes
    ----------
    mode : str
        'analytic', 'shared', or 'per_antenna'
    beam_handlers : dict
        Mapping beam_id → BeamFITSHandler
    antenna_to_beam : dict
        Mapping antenna_number → beam_id
    """

    def __init__(self, config, antenna_data, logger):
        """
        Initialize beam manager based on configuration.

        Parameters
        ----------
        config : dict
            RRIvis configuration dictionary
        antenna_data : dict
            Antenna data from antenna.py, including:
            - antenna_numbers: list of antenna indices
            - beam_ids: list of beam IDs (or None)
        logger : logging.Logger
            Logger instance
        """
        self.config = config
        self.antenna_data = antenna_data
        self.logger = logger

        # Storage for beam handlers
        self.beam_handlers = {}  # beam_id → BeamFITSHandler
        self.antenna_to_beam = {}  # antenna_number → beam_id

        # Determine mode and load beams
        self._initialize_beams()

    def _initialize_beams(self):
        """Determine beam mode and load appropriate beams."""
        beams_config = self.config["beams"]

        if not beams_config.get("use_beam_file", False):
            self.logger.info("Using analytic beam models (no beam files)")
            self.mode = "analytic"
            return

        # Check if per-antenna beams requested
        use_different_beams = beams_config.get("use_different_beams", False)

        if not use_different_beams:
            self._load_shared_beam()
        else:
            self._load_per_antenna_beams()

    def _load_shared_beam(self):
        """Load single beam for all antennas."""
        self.mode = "shared"
        beam_path = self.config["beams"]["beam_file_path"]

        if not beam_path:
            raise ValueError("use_beam_file=True but beam_file_path is empty!")

        self.logger.info("Loading shared beam for all antennas")

        # Load single beam with ID 0
        self.beam_handlers[0] = BeamFITSHandler(beam_path, self.config, self.logger)

        # All antennas use beam 0
        for ant_num in self.antenna_data["antenna_numbers"]:
            self.antenna_to_beam[ant_num] = 0

        self.logger.info(
            f"  All {len(self.antenna_to_beam)} antennas assigned to shared beam"
        )

    def _load_per_antenna_beams(self):
        """Load per-antenna beams from layout file or config."""
        self.mode = "per_antenna"

        # CRITICAL FIX: Check for both existence AND non-None
        # BUG WAS: Only checked 'beam_ids' in dict, not if it was None
        # CAUSED: Crash when beam_ids key exists but value is None
        has_beam_ids_in_layout = (
            "beam_ids" in self.antenna_data
            and self.antenna_data["beam_ids"] is not None
        )

        has_beam_mapping_in_config = bool(
            self.config["beams"].get("beams_per_antenna", {})
        )

        if has_beam_ids_in_layout:
            self.logger.info("Using beam IDs from antenna layout file")
            self._load_from_layout()
        elif has_beam_mapping_in_config:
            self.logger.info("Using beam assignments from config")
            self._load_from_config()
        else:
            raise ValueError(
                "use_different_beams=True but no beam assignments found!\n"
                "Either:\n"
                "  1. Add BeamID column to antenna layout file, OR\n"
                "  2. Specify beams_per_antenna in config YAML"
            )

    def _load_from_layout(self):
        """Load beams based on IDs in antenna layout file."""
        beam_files = self.config["beams"].get("beam_files", {})

        if not beam_files:
            raise ValueError(
                "Beam IDs in layout but no beam_files mapping in config!"
            )

        # Get unique beam IDs from antenna data
        beam_ids = set(self.antenna_data["beam_ids"])

        # Load each unique beam
        for beam_id in beam_ids:
            if beam_id not in beam_files:
                raise ValueError(
                    f"Beam ID {beam_id} in layout but not in config beam_files!"
                )

            beam_path = beam_files[beam_id]
            self.logger.info(f"Loading beam {beam_id}: {beam_path}")
            self.beam_handlers[beam_id] = BeamFITSHandler(
                beam_path, self.config, self.logger
            )

        # Map antennas to beams
        for ant_num, beam_id in zip(
            self.antenna_data["antenna_numbers"], self.antenna_data["beam_ids"]
        ):
            self.antenna_to_beam[ant_num] = beam_id

        self.logger.info(f"Loaded {len(self.beam_handlers)} unique beams")

    def _load_from_config(self):
        """Load beams based on antenna → beam mapping in config."""
        beam_files = self.config["beams"].get("beam_files", {})
        beams_per_antenna = self.config["beams"].get("beams_per_antenna", {})

        # Load unique beam files
        unique_beam_ids = set(beams_per_antenna.values())

        for beam_id in unique_beam_ids:
            if beam_id not in beam_files:
                raise ValueError(
                    f"Beam ID {beam_id} referenced but not in beam_files!"
                )

            beam_path = beam_files[beam_id]
            self.logger.info(f"Loading beam {beam_id}: {beam_path}")
            self.beam_handlers[beam_id] = BeamFITSHandler(
                beam_path, self.config, self.logger
            )

        # Map antennas to beams
        for ant_num in self.antenna_data["antenna_numbers"]:
            if ant_num in beams_per_antenna:
                self.antenna_to_beam[ant_num] = beams_per_antenna[ant_num]
            else:
                # Use default if not specified
                default_beam = self.config["beams"].get("default_beam_id", 0)
                self.antenna_to_beam[ant_num] = default_beam

        self.logger.info(f"Loaded {len(self.beam_handlers)} unique beams")

    def get_jones_matrix(
        self, antenna_number, alt_rad, az_rad, freq_hz, location, time
    ):
        """
        Get Jones matrix for specific antenna at given position.

        Parameters
        ----------
        antenna_number : int
            Antenna index
        alt_rad, az_rad : float or array
            Sky coordinates in radians (Astropy convention)
        freq_hz : float
            Frequency in Hz
        location : EarthLocation
            Observatory location
        time : Time
            Observation time

        Returns
        -------
        jones : ndarray or None
            2×2 Jones matrix (2, 2) or (Nsources, 2, 2)
            Returns None if mode='analytic' (signals use analytic beams)
        """
        if self.mode == "analytic":
            # Return None to signal use of analytic beams
            return None

        # Get beam ID for this antenna
        beam_id = self.antenna_to_beam[antenna_number]

        # Get Jones matrix from appropriate beam
        jones = self.beam_handlers[beam_id].get_jones_matrix(
            alt_rad, az_rad, freq_hz, location, time
        )

        return jones
