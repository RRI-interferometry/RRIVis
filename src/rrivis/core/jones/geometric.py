"""Geometric phase delay Jones term (K matrix).

The K term represents the geometric phase delay from the baseline-source geometry:
    K = exp(-2πi(ul + vm + w(n-1))) * I

where (u,v,w) are baseline coordinates in wavelengths and (l,m,n) are
direction cosines of the source.
"""

from typing import Any, Dict, Optional, Union
import numpy as np

from rrivis.core.jones.base import JonesTerm


class GeometricPhaseJones(JonesTerm):
    """Geometric phase delay from baseline-source geometry.

    K = exp(-2πi(ul + vm + w(n-1))) * I

    This is a scalar Jones term (proportional to identity matrix), so it
    commutes with all other terms.

    Note: The phase is actually baseline-dependent, but we compute it
    per-antenna for consistency with the JonesChain interface. The full
    fringe term is applied when computing visibilities.

    Args:
        source_lmn: Array of direction cosines per source.
                   Shape: (N_sources, 3) for (l, m, n), or
                   Shape: (N_sources, 2) for (l, m) only - n will be calculated.
        wavelengths: Array of wavelengths in meters
                    Shape: (N_freq,)
    """

    def __init__(
        self,
        source_lmn: np.ndarray,
        wavelengths: np.ndarray,
    ):
        """Initialize geometric phase term.

        Args:
            source_lmn: Direction cosines array.
                       Shape (N_sources, 3) for (l, m, n), or
                       Shape (N_sources, 2) for (l, m) - n computed automatically.
            wavelengths: Wavelengths in meters (N_freq,)
        """
        source_lmn = np.asarray(source_lmn)
        self.wavelengths = np.asarray(wavelengths)

        # Handle case where only (l, m) is provided - compute n
        if source_lmn.ndim == 1:
            # Single source
            source_lmn = source_lmn.reshape(1, -1)

        if source_lmn.shape[1] == 2:
            # Only (l, m) provided - calculate n = sqrt(1 - l² - m²)
            l = source_lmn[:, 0]
            m = source_lmn[:, 1]
            n = np.sqrt(1 - l**2 - m**2)
            self.source_lmn = np.column_stack([l, m, n])
        elif source_lmn.shape[1] == 3:
            self.source_lmn = source_lmn
        else:
            raise ValueError(
                f"source_lmn must have shape (N, 2) or (N, 3), "
                f"got shape {source_lmn.shape}"
            )

    @property
    def name(self) -> str:
        return "K"

    @property
    def is_direction_dependent(self) -> bool:
        return True  # Depends on source position

    @property
    def is_frequency_dependent(self) -> bool:
        return True  # Phase depends on wavelength

    def is_scalar(self) -> bool:
        return True  # Proportional to identity

    def is_unitary(self) -> bool:
        return True  # Pure phase (|K| = 1)

    def compute_jones(
        self,
        antenna_idx: int,
        source_idx: Optional[int],
        freq_idx: int,
        time_idx: int,
        backend: Any,
        **kwargs
    ) -> Any:
        """Compute geometric phase Jones matrix.

        For the K term, we return identity matrix here. The actual
        phase is applied at the visibility level since it requires
        baseline coordinates.

        If 'baseline_uvw' is provided in kwargs, we compute the full
        phase term.

        Args:
            antenna_idx: Antenna index (not used for K)
            source_idx: Source index
            freq_idx: Frequency index
            time_idx: Time index (not used)
            backend: Array backend
            **kwargs: May contain 'baseline_uvw' for full phase.
                     baseline_uvw can be shape (3,) for single baseline
                     or (N_baselines, 3) for multiple baselines.

        Returns:
            2x2 Jones matrix (identity or phase * identity)
        """
        xp = backend.xp

        # Check if baseline UVW is provided for full phase calculation
        baseline_uvw = kwargs.get('baseline_uvw')

        if baseline_uvw is None:
            # Return identity - phase applied at visibility level
            return xp.eye(2, dtype=np.complex128)

        # Get source direction cosines
        l, m, n = self.source_lmn[source_idx]

        # Get baseline coordinates - handle both (3,) and (N, 3) shapes
        baseline_uvw = np.asarray(baseline_uvw)
        if baseline_uvw.ndim == 1:
            # Single baseline: shape (3,)
            u, v, w = baseline_uvw
        else:
            # Multiple baselines: shape (N, 3)
            u, v, w = baseline_uvw.T  # Transpose to unpack columns

        # Compute phase: -2π(ul + vm + w(n-1))
        # Note: baseline_uvw should already be in wavelengths
        phase = -2.0 * np.pi * (u * l + v * m + w * (n - 1.0))

        # Geometric phase term
        phase_term = xp.exp(1j * phase)

        # Return scalar matrix: phase * I
        K = phase_term * xp.eye(2, dtype=np.complex128)

        return K

    def compute_fringe(
        self,
        source_idx: int,
        freq_idx: int,
        baseline_uvw: Union[np.ndarray, Any],
        backend: Any,
    ) -> Any:
        """Compute fringe term for a source-baseline pair.

        This is a convenience method for computing just the phase term
        without the full Jones matrix machinery.

        Args:
            source_idx: Source index
            freq_idx: Frequency index
            baseline_uvw: Baseline (u, v, w) in wavelengths.
                         Shape (3,) for single baseline, or
                         Shape (N_baselines, 3) for multiple baselines.
            backend: Array backend

        Returns:
            Complex phase factor exp(-2πi(ul + vm + w(n-1)))
            Scalar for single baseline, array for multiple baselines.
        """
        xp = backend.xp

        l, m, n = self.source_lmn[source_idx]

        # Handle both (3,) and (N, 3) shapes for baseline_uvw
        baseline_uvw = np.asarray(baseline_uvw)
        if baseline_uvw.ndim == 1:
            # Single baseline: shape (3,)
            u, v, w = baseline_uvw
        else:
            # Multiple baselines: shape (N, 3)
            u, v, w = baseline_uvw.T  # Transpose to unpack columns

        phase = -2.0 * np.pi * (u * l + v * m + w * (n - 1.0))
        return xp.exp(1j * phase)

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({
            "n_sources": len(self.source_lmn),
            "n_frequencies": len(self.wavelengths),
        })
        return config
