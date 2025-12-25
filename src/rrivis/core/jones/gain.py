"""Complex gain Jones term (G matrix).

The G term represents electronic gains including amplitude and phase
variations in the signal chain. It is direction-independent and typically
diagonal (unless there's coupling between polarizations).
"""

from typing import Any, Dict, Optional, Union
import numpy as np

from rrivis.core.jones.base import JonesTerm


class GainJones(JonesTerm):
    """Complex electronic gains Jones matrix.

    G = [[g_x, 0  ],
         [0,   g_y]]

    where g_x and g_y are complex gains for X and Y polarizations.

    Gains can be:
    - Per-antenna (different gains for each antenna)
    - Time-variable (gains change with time)
    - Frequency-dependent (if not using separate bandpass term)

    Args:
        gains: Complex gain values.
              Shape: (N_ant,) for static gains, or
              Shape: (N_ant, N_time) for time-variable gains, or
              Shape: (N_ant, N_time, 2) for per-polarization gains
        gain_sigma: Optional standard deviation for gain perturbations
    """

    def __init__(
        self,
        gains: Optional[np.ndarray] = None,
        n_antennas: int = 1,
        gain_sigma: float = 0.0,
        seed: Optional[int] = None,
    ):
        """Initialize gain Jones term.

        Args:
            gains: Pre-computed gain values (complex). If None, uses ideal gains.
            n_antennas: Number of antennas (used if gains is None)
            gain_sigma: Standard deviation for random gain perturbations
            seed: Random seed for reproducibility
        """
        self.n_antennas = n_antennas
        self.gain_sigma = gain_sigma
        self._rng = np.random.default_rng(seed)

        if gains is not None:
            self.gains = np.asarray(gains, dtype=np.complex128)
            self.n_antennas = self.gains.shape[0]
        else:
            # Default to ideal gains (unity)
            self.gains = np.ones(n_antennas, dtype=np.complex128)

        # Add perturbations if requested
        if gain_sigma > 0 and gains is None:
            amp_pert = 1 + self._rng.normal(0, gain_sigma, n_antennas)
            phase_pert = self._rng.normal(0, gain_sigma, n_antennas)
            self.gains = amp_pert * np.exp(1j * phase_pert)

    @property
    def name(self) -> str:
        return "G"

    @property
    def is_direction_dependent(self) -> bool:
        return False  # Gains are per-antenna, not per-direction

    @property
    def is_time_dependent(self) -> bool:
        return self.gains.ndim > 1  # Time-variable if 2D+

    @property
    def is_frequency_dependent(self) -> bool:
        return False  # Use BandpassJones for frequency-dependent gains

    def is_diagonal(self) -> bool:
        return True  # Standard gains are diagonal

    def compute_jones(
        self,
        antenna_idx: int,
        source_idx: Optional[int],
        freq_idx: int,
        time_idx: int,
        backend: Any,
        **kwargs
    ) -> Any:
        """Compute gain Jones matrix.

        Args:
            antenna_idx: Antenna index
            source_idx: Not used (direction-independent)
            freq_idx: Not used (frequency-independent)
            time_idx: Time index (if time-variable gains)
            backend: Array backend
            **kwargs: Additional parameters

        Returns:
            2x2 diagonal complex Jones matrix
        """
        xp = backend.xp

        # Get gain for this antenna
        if self.gains.ndim == 1:
            # Static gains: shape (N_ant,)
            g = self.gains[antenna_idx]
            g_x = g
            g_y = g
        elif self.gains.ndim == 2:
            # Time-variable gains: shape (N_ant, N_time)
            g = self.gains[antenna_idx, time_idx]
            g_x = g
            g_y = g
        elif self.gains.ndim == 3:
            # Per-polarization gains: shape (N_ant, N_time, 2)
            g_x = self.gains[antenna_idx, time_idx, 0]
            g_y = self.gains[antenna_idx, time_idx, 1]
        else:
            raise ValueError(f"Unexpected gains shape: {self.gains.shape}")

        # Build diagonal Jones matrix
        G = xp.array([
            [g_x, 0],
            [0, g_y],
        ], dtype=np.complex128)

        return G

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({
            "n_antennas": self.n_antennas,
            "gains_shape": self.gains.shape,
            "gain_sigma": self.gain_sigma,
        })
        return config


class TimeVariableGainJones(GainJones):
    """Gains that vary smoothly with time.

    Generates realistic time-variable gains using a random walk
    or smooth interpolation model.
    """

    def __init__(
        self,
        n_antennas: int,
        n_times: int,
        amp_sigma: float = 0.02,
        phase_sigma: float = 0.05,
        correlation_time: float = 600.0,
        time_step: float = 10.0,
        seed: Optional[int] = None,
    ):
        """Initialize time-variable gains.

        Args:
            n_antennas: Number of antennas
            n_times: Number of time samples
            amp_sigma: Standard deviation of amplitude variations (fractional)
            phase_sigma: Standard deviation of phase variations (radians)
            correlation_time: Time scale for gain variations (seconds)
            time_step: Time between samples (seconds)
            seed: Random seed
        """
        self.amp_sigma = amp_sigma
        self.phase_sigma = phase_sigma
        self.correlation_time = correlation_time
        self.time_step = time_step

        rng = np.random.default_rng(seed)

        # Generate correlated random walks
        # Correlation coefficient between adjacent samples
        rho = np.exp(-time_step / correlation_time)

        gains = np.ones((n_antennas, n_times), dtype=np.complex128)

        for ant in range(n_antennas):
            # Generate correlated amplitude and phase variations
            amp = np.ones(n_times)
            phase = np.zeros(n_times)

            for t in range(1, n_times):
                # AR(1) process
                amp[t] = rho * amp[t-1] + np.sqrt(1 - rho**2) * rng.normal(1, amp_sigma)
                phase[t] = rho * phase[t-1] + np.sqrt(1 - rho**2) * rng.normal(0, phase_sigma)

            gains[ant, :] = amp * np.exp(1j * phase)

        super().__init__(gains=gains, n_antennas=n_antennas)

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({
            "amp_sigma": self.amp_sigma,
            "phase_sigma": self.phase_sigma,
            "correlation_time": self.correlation_time,
            "time_step": self.time_step,
        })
        return config