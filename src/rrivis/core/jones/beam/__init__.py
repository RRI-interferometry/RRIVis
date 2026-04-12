"""Primary beam Jones term (E matrix) and beam pattern models.

This sub-package implements the E-Jones (primary beam) for the RIME,
representing how antenna sensitivity varies across the sky.

Sub-packages
------------
analytic
    Analytic aperture beam models: composed beam, aperture shapes,
    illumination tapers, feed models, numerical HPBW.
fits
    FITS beam file handling via pyuvdata UVBeam.

Classes
-------
BeamJones
    Base beam Jones term wrapping a callable beam model.
AnalyticBeamJones
    Aperture-based analytic beam with configurable shape, taper, feed.
FITSBeamJones
    FITS-file-based beam via ``BeamManager``.
"""

from collections.abc import Callable
from typing import Any

import numpy as np

from rrivis.core.jones.base import JonesTerm


class BeamJones(JonesTerm):
    """Primary beam voltage pattern Jones matrix (E term in the RIME).

    The beam Jones matrix describes how antenna sensitivity varies across
    the sky::

        E = [[E_xx, E_xy], [E_yx, E_yy]]

    This is generally a full 2x2 matrix (non-diagonal) due to
    cross-polarization coupling, beam squint, and asymmetric feeds.

    Parameters
    ----------
    beam_model : callable
        Function returning a 2x2 Jones matrix. Signature:
        ``(antenna_idx, zenith_angle, azimuth, frequency, time_idx, **kw) -> (2, 2)``
    source_altaz : np.ndarray
        Source coordinates in alt-az, shape ``(N_sources, 2)`` as
        ``[altitude, azimuth]`` in radians.
    frequencies : np.ndarray
        Observation frequencies in Hz, shape ``(N_freq,)``.
    """

    def __init__(
        self,
        beam_model: Callable,
        source_altaz: np.ndarray,
        frequencies: np.ndarray,
    ):
        self.beam_model = beam_model
        self.source_altaz = np.asarray(source_altaz)
        self.frequencies = np.asarray(frequencies)

    @property
    def name(self) -> str:
        return "E"

    @property
    def is_direction_dependent(self) -> bool:
        return True  # Beam pattern varies across sky

    @property
    def is_time_dependent(self) -> bool:
        return True  # For alt-az, source position in beam frame changes

    @property
    def is_frequency_dependent(self) -> bool:
        return True  # Beam pattern scales with frequency

    def is_diagonal(self) -> bool:
        return False  # Generally has cross-pol terms

    def compute_jones(
        self,
        antenna_idx: int,
        source_idx: int | None,
        freq_idx: int,
        time_idx: int,
        backend: Any,
        **kwargs,
    ) -> Any:
        """Compute primary beam Jones matrix for a single source.

        Parameters
        ----------
        antenna_idx : int
            Antenna index.
        source_idx : int or None
            Source index into ``source_altaz``.
        freq_idx : int
            Frequency index into ``frequencies``.
        time_idx : int
            Time step index.
        backend : ArrayBackend
            Array backend for type conversion.
        **kwargs
            Additional parameters forwarded to ``beam_model``.

        Returns
        -------
        np.ndarray
            Complex Jones matrix, shape ``(2, 2)``.
        """
        # Get source coordinates
        alt, az = self.source_altaz[source_idx]
        zenith_angle = np.pi / 2 - alt  # Convert altitude to zenith angle

        # Get frequency
        frequency = self.frequencies[freq_idx]

        # Compute beam Jones matrix using provided model
        E = self.beam_model(
            antenna_idx=antenna_idx,
            zenith_angle=zenith_angle,
            azimuth=az,
            frequency=frequency,
            time_idx=time_idx,
            **kwargs,
        )

        # Convert to backend array
        return backend.asarray(E, dtype=np.complex128)

    def get_config(self) -> dict[str, Any]:
        config = super().get_config()
        config.update(
            {
                "n_sources": len(self.source_altaz),
                "n_frequencies": len(self.frequencies),
                "beam_model": self.beam_model.__name__
                if hasattr(self.beam_model, "__name__")
                else str(type(self.beam_model)),
            }
        )
        return config


# Import from sub-packages AFTER BeamJones is defined (sub-packages inherit from it)
from rrivis.core.jones.beam.analytic import (  # noqa: E402
    AnalyticBeamJones,
    compute_aperture_beam,
    plot_beam_2d,
    plot_beam_comparison,
    plot_beam_pattern,
    plot_feed_illumination,
)
from rrivis.core.jones.beam.fits import (  # noqa: E402
    BeamFITSHandler,
    BeamManager,
    FITSBeamJones,
    astropy_az_to_uvbeam_az,
)
from rrivis.core.jones.beam.projection import (  # noqa: E402
    BeamSkyProjection,
    compute_beam_power_on_radec_grid,
    create_rgba_overlay,
    extract_contours,
)

__all__ = [
    # Jones matrix classes
    "BeamJones",
    "AnalyticBeamJones",
    "FITSBeamJones",
    # Analytic beam
    "compute_aperture_beam",
    # Beam plotting
    "plot_beam_pattern",
    "plot_beam_comparison",
    "plot_beam_2d",
    "plot_feed_illumination",
    # FITS beam handling
    "astropy_az_to_uvbeam_az",
    "BeamFITSHandler",
    "BeamManager",
    # Beam sky projection
    "BeamSkyProjection",
    "compute_beam_power_on_radec_grid",
    "create_rgba_overlay",
    "extract_contours",
]
