"""FITS beam file handling for the E-Jones (primary beam) in the RIME.

This sub-package provides the :class:`FITSBeamJones` Jones term and the
underlying :class:`BeamFITSHandler` and :class:`BeamManager` for loading
and interpolating beam patterns from FITS files via pyuvdata UVBeam.

Modules
-------
handler
    FITS beam file handling (``BeamFITSHandler``, ``BeamManager``).
"""

from typing import Any

import numpy as np

from rrivis.core.jones.beam import BeamJones
from rrivis.core.jones.beam.fits.handler import (
    BeamFITSHandler,
    BeamManager,
    astropy_az_to_uvbeam_az,
)


class FITSBeamJones(BeamJones):
    """Beam Jones term using FITS beam files via BeamManager.

    Wraps a :class:`BeamManager` instance to provide FITS-based beam
    responses within the JonesChain framework. Falls back to identity
    if the BeamManager returns ``None``.

    Parameters
    ----------
    beam_manager : BeamManager
        BeamManager instance for FITS beam interpolation.
    source_altaz : np.ndarray
        Source coordinates in alt-az, shape ``(N_sources, 2)``
        as ``[altitude, azimuth]`` in radians.
    frequencies : np.ndarray
        Observation frequencies in Hz, shape ``(N_freq,)``.
    """

    def __init__(
        self,
        beam_manager: Any,
        source_altaz: np.ndarray,
        frequencies: np.ndarray,
    ):
        self._beam_manager = beam_manager

        def beam_model(
            antenna_idx: int,
            zenith_angle: float,
            azimuth: float,
            frequency: float,
            time_idx: int,
            **kw: Any,
        ) -> np.ndarray:
            alt = np.pi / 2 - zenith_angle
            antenna_number = kw.get("antenna_number", antenna_idx)
            jones = self._beam_manager.get_jones_matrix(
                antenna_number=antenna_number,
                alt_rad=alt,
                az_rad=azimuth,
                freq_hz=frequency,
                location=None,
                time=None,
            )
            if jones is None:
                return np.eye(2, dtype=np.complex128)
            return jones

        super().__init__(beam_model, source_altaz, frequencies)

    def compute_jones_all_sources(
        self,
        antenna_idx: int,
        n_sources: int,
        freq_idx: int,
        time_idx: int,
        backend: Any,
        **kwargs: Any,
    ) -> Any:
        """Vectorized FITS beam lookup for all sources.

        Parameters
        ----------
        antenna_idx : int
            Antenna index.
        n_sources : int
            Number of sources to evaluate.
        freq_idx : int
            Frequency index.
        time_idx : int
            Time step index.
        backend : ArrayBackend
            Array backend for type conversion.
        **kwargs
            Must include ``antenna_number`` for BeamManager lookup.

        Returns
        -------
        np.ndarray
            Jones matrices, shape ``(n_sources, 2, 2)``.
        """
        alts = self.source_altaz[:n_sources, 0]
        azs = self.source_altaz[:n_sources, 1]
        frequency = self.frequencies[freq_idx]
        antenna_number = kwargs.get("antenna_number", antenna_idx)

        jones = self._beam_manager.get_jones_matrix(
            antenna_number=antenna_number,
            alt_rad=alts,
            az_rad=azs,
            freq_hz=frequency,
            location=None,
            time=None,
        )

        if jones is None:
            result = np.zeros((n_sources, 2, 2), dtype=np.complex128)
            result[:, 0, 0] = 1.0
            result[:, 1, 1] = 1.0
            return backend.asarray(result, dtype=np.complex128)

        if jones.ndim == 2:
            jones = np.tile(jones, (n_sources, 1, 1))
        return backend.asarray(jones, dtype=np.complex128)

    def is_diagonal(self) -> bool:
        return False  # FITS beams can have cross-pol

    def get_config(self) -> dict[str, Any]:
        config = super().get_config()
        config["beam_source"] = "fits"
        return config


__all__ = [
    "FITSBeamJones",
    "astropy_az_to_uvbeam_az",
    "BeamFITSHandler",
    "BeamManager",
]
