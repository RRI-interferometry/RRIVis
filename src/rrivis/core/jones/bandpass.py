"""
Bandpass Jones term (B) for frequency-dependent instrumental response.

The B-Jones term models the frequency-dependent complex gain of the signal
path, including:
- Analog filter responses
- Digital filter characteristics
- Cable frequency response
- Receiver bandpass shape

This is a direction-independent effect that varies with frequency and antenna.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from abc import abstractmethod

from .base import JonesTerm


class BandpassJones(JonesTerm):
    """
    Abstract base class for bandpass calibration.

    The bandpass describes the frequency-dependent complex gain of each
    antenna/polarization combination across the observing band.

    Parameters
    ----------
    n_antennas : int
        Number of antennas in the array.
    frequencies : np.ndarray
        Frequency channels in Hz, shape (n_freq,).
    bandpass_gains : np.ndarray, optional
        Pre-computed bandpass gains with shape (n_antennas, n_freq, 2, 2).
        If None, unity bandpass is assumed.
    """

    def __init__(
        self,
        n_antennas: int,
        frequencies: np.ndarray,
        bandpass_gains: Optional[np.ndarray] = None
    ):
        self.n_antennas = n_antennas
        self.frequencies = np.asarray(frequencies)
        self.n_freq = len(self.frequencies)

        if bandpass_gains is not None:
            self.bandpass_gains = np.asarray(bandpass_gains)
            if self.bandpass_gains.shape != (n_antennas, self.n_freq, 2, 2):
                raise ValueError(
                    f"bandpass_gains shape {self.bandpass_gains.shape} does not match "
                    f"expected ({n_antennas}, {self.n_freq}, 2, 2)"
                )
        else:
            # Unity bandpass (no frequency-dependent gain)
            self.bandpass_gains = np.zeros((n_antennas, self.n_freq, 2, 2), dtype=np.complex128)
            self.bandpass_gains[:, :, 0, 0] = 1.0
            self.bandpass_gains[:, :, 1, 1] = 1.0

    @property
    def name(self) -> str:
        return "B"

    @property
    def is_direction_dependent(self) -> bool:
        return False

    def compute_jones(
        self,
        antenna_idx: int,
        source_idx: int,
        freq_idx: int,
        time_idx: int,
        backend: Any,
        **kwargs
    ) -> Any:
        """
        Compute bandpass Jones matrix for given antenna and frequency.

        Parameters
        ----------
        antenna_idx : int
            Antenna index.
        source_idx : int
            Source index (not used, bandpass is direction-independent).
        freq_idx : int
            Frequency channel index.
        time_idx : int
            Time index (not used, bandpass is assumed time-invariant).
        backend : ArrayBackend
            Compute backend.
        **kwargs : dict
            Additional parameters (unused).

        Returns
        -------
        jones : array
            2x2 complex Jones matrix for bandpass.
        """
        xp = backend.xp
        jones = backend.asarray(self.bandpass_gains[antenna_idx, freq_idx])
        return jones

    def set_bandpass(
        self,
        antenna_idx: int,
        gains: np.ndarray,
        freq_indices: Optional[np.ndarray] = None
    ) -> None:
        """
        Set bandpass gains for a specific antenna.

        Parameters
        ----------
        antenna_idx : int
            Antenna index.
        gains : np.ndarray
            Complex gains with shape (n_freq, 2, 2) or (n_selected_freq, 2, 2).
        freq_indices : np.ndarray, optional
            Frequency channel indices if updating subset. If None, updates all.
        """
        if freq_indices is None:
            self.bandpass_gains[antenna_idx] = gains
        else:
            self.bandpass_gains[antenna_idx, freq_indices] = gains

    def get_amplitude_response(self, antenna_idx: int, pol: int = 0) -> np.ndarray:
        """
        Get amplitude response for a polarization.

        Parameters
        ----------
        antenna_idx : int
            Antenna index.
        pol : int
            Polarization index (0 or 1).

        Returns
        -------
        amplitude : np.ndarray
            Amplitude response vs frequency, shape (n_freq,).
        """
        return np.abs(self.bandpass_gains[antenna_idx, :, pol, pol])

    def get_phase_response(self, antenna_idx: int, pol: int = 0) -> np.ndarray:
        """
        Get phase response for a polarization.

        Parameters
        ----------
        antenna_idx : int
            Antenna index.
        pol : int
            Polarization index (0 or 1).

        Returns
        -------
        phase : np.ndarray
            Phase response in radians vs frequency, shape (n_freq,).
        """
        return np.angle(self.bandpass_gains[antenna_idx, :, pol, pol])


class PolynomialBandpassJones(BandpassJones):
    """
    Bandpass model using polynomial representation.

    Models the bandpass as a polynomial in normalized frequency, which is
    useful for smooth bandpass shapes and for interpolation.

    Parameters
    ----------
    n_antennas : int
        Number of antennas.
    frequencies : np.ndarray
        Frequency channels in Hz.
    poly_order : int
        Order of the polynomial (default: 3).
    amplitude_coeffs : np.ndarray, optional
        Amplitude polynomial coefficients, shape (n_antennas, 2, poly_order+1).
    phase_coeffs : np.ndarray, optional
        Phase polynomial coefficients, shape (n_antennas, 2, poly_order+1).
    """

    def __init__(
        self,
        n_antennas: int,
        frequencies: np.ndarray,
        poly_order: int = 3,
        amplitude_coeffs: Optional[np.ndarray] = None,
        phase_coeffs: Optional[np.ndarray] = None
    ):
        self.poly_order = poly_order
        frequencies = np.asarray(frequencies)

        # Normalized frequency for polynomial evaluation
        self.freq_min = frequencies.min()
        self.freq_max = frequencies.max()
        self.freq_normalized = 2 * (frequencies - self.freq_min) / (self.freq_max - self.freq_min) - 1

        n_freq = len(frequencies)
        n_coeffs = poly_order + 1

        # Default: unity amplitude, zero phase
        if amplitude_coeffs is None:
            amplitude_coeffs = np.zeros((n_antennas, 2, n_coeffs))
            amplitude_coeffs[:, :, 0] = 1.0  # Constant term = 1

        if phase_coeffs is None:
            phase_coeffs = np.zeros((n_antennas, 2, n_coeffs))

        self.amplitude_coeffs = amplitude_coeffs
        self.phase_coeffs = phase_coeffs

        # Compute bandpass from polynomials
        bandpass_gains = self._compute_from_polynomials(n_antennas, n_freq)

        super().__init__(n_antennas, frequencies, bandpass_gains)

    def _compute_from_polynomials(
        self,
        n_antennas: int,
        n_freq: int
    ) -> np.ndarray:
        """Evaluate polynomials to get bandpass gains."""
        bandpass = np.zeros((n_antennas, n_freq, 2, 2), dtype=np.complex128)

        for ant in range(n_antennas):
            for pol in range(2):
                # Evaluate amplitude polynomial
                amp = np.polyval(self.amplitude_coeffs[ant, pol, ::-1], self.freq_normalized)
                # Evaluate phase polynomial
                phase = np.polyval(self.phase_coeffs[ant, pol, ::-1], self.freq_normalized)
                # Combine to complex gain
                bandpass[ant, :, pol, pol] = amp * np.exp(1j * phase)

        return bandpass

    def fit_from_data(
        self,
        antenna_idx: int,
        measured_gains: np.ndarray,
        weights: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fit polynomial bandpass from measured data.

        Parameters
        ----------
        antenna_idx : int
            Antenna index.
        measured_gains : np.ndarray
            Measured complex gains, shape (n_freq, 2).
        weights : np.ndarray, optional
            Weights for fitting, shape (n_freq, 2).

        Returns
        -------
        amplitude_coeffs : np.ndarray
            Fitted amplitude coefficients, shape (2, poly_order+1).
        phase_coeffs : np.ndarray
            Fitted phase coefficients, shape (2, poly_order+1).
        """
        if weights is None:
            weights = np.ones_like(measured_gains, dtype=float)

        for pol in range(2):
            # Fit amplitude
            amp = np.abs(measured_gains[:, pol])
            valid = weights[:, pol] > 0
            if np.sum(valid) > self.poly_order:
                self.amplitude_coeffs[antenna_idx, pol] = np.polyfit(
                    self.freq_normalized[valid], amp[valid], self.poly_order
                )[::-1]

            # Fit unwrapped phase
            phase = np.unwrap(np.angle(measured_gains[:, pol]))
            if np.sum(valid) > self.poly_order:
                self.phase_coeffs[antenna_idx, pol] = np.polyfit(
                    self.freq_normalized[valid], phase[valid], self.poly_order
                )[::-1]

        # Recompute bandpass
        self.bandpass_gains[antenna_idx] = self._compute_from_polynomials(1, self.n_freq)[0]

        return self.amplitude_coeffs[antenna_idx], self.phase_coeffs[antenna_idx]


class SplineBandpassJones(BandpassJones):
    """
    Bandpass model using cubic spline interpolation.

    Uses a sparse set of control points to define the bandpass shape,
    with cubic spline interpolation between them. Useful for modeling
    bandpasses with sharp features.

    Parameters
    ----------
    n_antennas : int
        Number of antennas.
    frequencies : np.ndarray
        Frequency channels in Hz.
    control_frequencies : np.ndarray
        Frequencies of control points in Hz.
    control_gains : np.ndarray, optional
        Complex gains at control points, shape (n_antennas, n_control, 2).
    """

    def __init__(
        self,
        n_antennas: int,
        frequencies: np.ndarray,
        control_frequencies: np.ndarray,
        control_gains: Optional[np.ndarray] = None
    ):
        self.control_frequencies = np.asarray(control_frequencies)
        n_control = len(self.control_frequencies)

        if control_gains is None:
            # Unity gains at control points
            control_gains = np.ones((n_antennas, n_control, 2), dtype=np.complex128)

        self.control_gains = np.asarray(control_gains)

        # Interpolate to get full bandpass
        bandpass_gains = self._interpolate_bandpass(n_antennas, frequencies)

        super().__init__(n_antennas, frequencies, bandpass_gains)

    def _interpolate_bandpass(
        self,
        n_antennas: int,
        frequencies: np.ndarray
    ) -> np.ndarray:
        """Interpolate control points to full bandpass."""
        from scipy.interpolate import CubicSpline

        n_freq = len(frequencies)
        bandpass = np.zeros((n_antennas, n_freq, 2, 2), dtype=np.complex128)

        for ant in range(n_antennas):
            for pol in range(2):
                # Interpolate amplitude and phase separately
                amp = np.abs(self.control_gains[ant, :, pol])
                phase = np.unwrap(np.angle(self.control_gains[ant, :, pol]))

                # Create splines
                amp_spline = CubicSpline(self.control_frequencies, amp)
                phase_spline = CubicSpline(self.control_frequencies, phase)

                # Evaluate at all frequencies
                amp_interp = amp_spline(frequencies)
                phase_interp = phase_spline(frequencies)

                bandpass[ant, :, pol, pol] = amp_interp * np.exp(1j * phase_interp)

        return bandpass

    def update_control_point(
        self,
        antenna_idx: int,
        control_idx: int,
        gain: complex,
        pol: int = 0
    ) -> None:
        """
        Update a single control point and reinterpolate.

        Parameters
        ----------
        antenna_idx : int
            Antenna index.
        control_idx : int
            Control point index.
        gain : complex
            New complex gain value.
        pol : int
            Polarization (0 or 1).
        """
        self.control_gains[antenna_idx, control_idx, pol] = gain
        # Reinterpolate for this antenna
        new_bandpass = self._interpolate_bandpass(1, self.frequencies)
        self.bandpass_gains[antenna_idx] = new_bandpass[0]


class RFIFlaggedBandpassJones(BandpassJones):
    """
    Bandpass with RFI flagging support.

    Tracks which frequency channels are flagged due to RFI and sets
    their gains to zero.

    Parameters
    ----------
    n_antennas : int
        Number of antennas.
    frequencies : np.ndarray
        Frequency channels in Hz.
    bandpass_gains : np.ndarray, optional
        Pre-computed bandpass gains.
    flags : np.ndarray, optional
        Boolean flag array, shape (n_antennas, n_freq) or (n_freq,).
        True = flagged (RFI contaminated).
    """

    def __init__(
        self,
        n_antennas: int,
        frequencies: np.ndarray,
        bandpass_gains: Optional[np.ndarray] = None,
        flags: Optional[np.ndarray] = None
    ):
        super().__init__(n_antennas, frequencies, bandpass_gains)

        if flags is None:
            self.flags = np.zeros((n_antennas, self.n_freq), dtype=bool)
        else:
            flags = np.asarray(flags, dtype=bool)
            if flags.ndim == 1:
                # Same flags for all antennas
                self.flags = np.tile(flags, (n_antennas, 1))
            else:
                self.flags = flags

        # Apply flags
        self._apply_flags()

    def _apply_flags(self) -> None:
        """Zero out flagged channels."""
        for ant in range(self.n_antennas):
            self.bandpass_gains[ant, self.flags[ant]] = 0.0

    def flag_channel(
        self,
        freq_idx: int,
        antenna_idx: Optional[int] = None
    ) -> None:
        """
        Flag a frequency channel.

        Parameters
        ----------
        freq_idx : int
            Frequency channel index to flag.
        antenna_idx : int, optional
            Antenna index. If None, flags for all antennas.
        """
        if antenna_idx is None:
            self.flags[:, freq_idx] = True
            self.bandpass_gains[:, freq_idx] = 0.0
        else:
            self.flags[antenna_idx, freq_idx] = True
            self.bandpass_gains[antenna_idx, freq_idx] = 0.0

    def flag_frequency_range(
        self,
        freq_min: float,
        freq_max: float,
        antenna_idx: Optional[int] = None
    ) -> None:
        """
        Flag a range of frequencies.

        Parameters
        ----------
        freq_min : float
            Minimum frequency to flag (Hz).
        freq_max : float
            Maximum frequency to flag (Hz).
        antenna_idx : int, optional
            Antenna index. If None, flags for all antennas.
        """
        mask = (self.frequencies >= freq_min) & (self.frequencies <= freq_max)
        if antenna_idx is None:
            self.flags[:, mask] = True
            self.bandpass_gains[:, mask] = 0.0
        else:
            self.flags[antenna_idx, mask] = True
            self.bandpass_gains[antenna_idx, mask] = 0.0

    def unflag_channel(
        self,
        freq_idx: int,
        antenna_idx: Optional[int] = None,
        restore_gains: Optional[np.ndarray] = None
    ) -> None:
        """
        Unflag a frequency channel.

        Parameters
        ----------
        freq_idx : int
            Frequency channel index to unflag.
        antenna_idx : int, optional
            Antenna index. If None, unflags for all antennas.
        restore_gains : np.ndarray, optional
            Gains to restore. If None, sets to unity.
        """
        unity = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.complex128)

        if antenna_idx is None:
            self.flags[:, freq_idx] = False
            if restore_gains is not None:
                self.bandpass_gains[:, freq_idx] = restore_gains
            else:
                self.bandpass_gains[:, freq_idx] = unity
        else:
            self.flags[antenna_idx, freq_idx] = False
            if restore_gains is not None:
                self.bandpass_gains[antenna_idx, freq_idx] = restore_gains
            else:
                self.bandpass_gains[antenna_idx, freq_idx] = unity

    def get_flag_fraction(self, antenna_idx: Optional[int] = None) -> float:
        """
        Get fraction of flagged channels.

        Parameters
        ----------
        antenna_idx : int, optional
            Antenna index. If None, returns average over all antennas.

        Returns
        -------
        fraction : float
            Fraction of channels flagged (0 to 1).
        """
        if antenna_idx is None:
            return np.mean(self.flags)
        return np.mean(self.flags[antenna_idx])
