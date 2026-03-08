"""Primary beam Jones term (E matrix) and beam pattern models.

This sub-package implements the E-Jones (primary beam) for the RIME,
representing how antenna sensitivity varies across the sky.

Modules
-------
analytic
    Composed aperture beam model (``compute_aperture_beam``).
aperture
    Aperture shape far-field patterns (Airy, sinc, elliptical Airy).
taper
    Illumination taper functions (uniform, Gaussian, parabolic, cosine).
feed
    Feed pattern models and reflector geometry (prime-focus, Cassegrain).
numerical_hpbw
    Numerical HPBW finder for arbitrary beam patterns.
fits
    FITS beam file handling via pyuvdata UVBeam.

Classes
-------
BeamJones
    Base beam Jones term wrapping a callable beam model.
AnalyticBeamJones
    Aperture-based analytic beam with configurable shape, taper, feed,
    and feed.
FITSBeamJones
    FITS-file-based beam via ``BeamManager``.
"""

from collections.abc import Callable
from typing import Any

import numpy as np

from rrivis.core.jones.base import JonesTerm
from rrivis.core.jones.beam.analytic import compute_aperture_beam

# Re-export FITS beam handling
from rrivis.core.jones.beam.fits import (
    BeamFITSHandler,
    BeamManager,
    astropy_az_to_uvbeam_az,
)


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


class AnalyticBeamJones(BeamJones):
    """Beam Jones term using aperture-based analytic beam patterns.

    Combines aperture shape, illumination taper, feed model, and
    cross-polarization model into a full 2x2 Jones matrix via
    :func:`~rrivis.core.jones.beam.analytic.compute_aperture_beam`.

    Parameters
    ----------
    source_altaz : np.ndarray
        Source alt-az coordinates, shape ``(N_sources, 2)`` in radians.
    frequencies : np.ndarray
        Frequencies in Hz, shape ``(N_freq,)``.
    diameter : float
        Default antenna diameter in metres.
    aperture_shape : str
        Aperture geometry: ``'circular'``, ``'rectangular'``, ``'elliptical'``.
    taper : str
        Illumination taper: ``'uniform'``, ``'gaussian'``, ``'parabolic'``,
        ``'parabolic_squared'``, ``'cosine'``.
    edge_taper_dB : float
        Edge taper in dB (default 10.0).
    feed_model : str
        Feed pattern model: ``'none'``, ``'corrugated_horn'``,
        ``'open_waveguide'``, ``'dipole_ground_plane'``.
    feed_computation : str
        ``'analytical'`` (derive edge taper) or ``'numerical'`` (Hankel transform).
    feed_params : dict or None
        Feed-specific parameters (e.g. ``q``, ``focal_ratio``).
    reflector_type : str
        ``'prime_focus'`` or ``'cassegrain'``.
    magnification : float
        Cassegrain magnification ``M = (e+1)/(e-1)``.
    diameter_per_antenna : dict or None
        Per-antenna diameter overrides ``{antenna_number: diameter_m}``.
    aperture_params : dict or None
        Aperture-specific parameters (e.g. ``length_x``/``length_y`` for
        rectangular, ``diameter_x``/``diameter_y`` for elliptical).
    """

    def __init__(
        self,
        source_altaz: np.ndarray,
        frequencies: np.ndarray,
        diameter: float,
        aperture_shape: str = "circular",
        taper: str = "gaussian",
        edge_taper_dB: float = 10.0,
        feed_model: str = "none",
        feed_computation: str = "analytical",
        feed_params: dict | None = None,
        reflector_type: str = "prime_focus",
        magnification: float = 1.0,
        diameter_per_antenna: dict[Any, float] | None = None,
        aperture_params: dict | None = None,
    ):
        self.diameter = diameter
        self.aperture_shape = aperture_shape
        self.taper = taper
        self.edge_taper_dB = edge_taper_dB
        self.feed_model = feed_model
        self.feed_computation = feed_computation
        self.feed_params = feed_params or {}
        self.reflector_type = reflector_type
        self.magnification = magnification
        self.diameter_per_antenna = diameter_per_antenna
        self.aperture_params = aperture_params or {}

        def beam_model(
            antenna_idx: int,
            zenith_angle: float,
            azimuth: float,
            frequency: float,
            time_idx: int,
            **kw: Any,
        ) -> np.ndarray:
            ant_num = kw.get("antenna_number", antenna_idx)
            d = self._get_diameter_for_antenna(ant_num)
            return compute_aperture_beam(
                theta=zenith_angle,
                phi=azimuth,
                frequency=frequency,
                diameter=d,
                aperture_shape=self.aperture_shape,
                taper=self.taper,
                edge_taper_dB=self.edge_taper_dB,
                feed_model=self.feed_model,
                feed_computation=self.feed_computation,
                feed_params=self.feed_params,
                reflector_type=self.reflector_type,
                magnification=self.magnification,
                aperture_params=self.aperture_params,
            )

        super().__init__(beam_model, source_altaz, frequencies)

    def _get_diameter_for_antenna(self, ant_num: Any) -> float:
        """Get diameter for a specific antenna, falling back to default."""
        if (
            self.diameter_per_antenna is not None
            and ant_num in self.diameter_per_antenna
        ):
            return self.diameter_per_antenna[ant_num]
        return self.diameter

    def compute_jones_all_sources(
        self,
        antenna_idx: int,
        n_sources: int,
        freq_idx: int,
        time_idx: int,
        backend: Any,
        **kwargs: Any,
    ) -> Any:
        """Vectorized beam Jones computation for all sources.

        Returns
        -------
        np.ndarray
            Jones matrices, shape ``(n_sources, 2, 2)``.
        """
        alts = self.source_altaz[:n_sources, 0]
        azs = self.source_altaz[:n_sources, 1]
        zenith_angles = np.pi / 2 - alts
        frequency = self.frequencies[freq_idx]

        ant_num = kwargs.get("antenna_number", antenna_idx)
        d = self._get_diameter_for_antenna(ant_num)

        jones = compute_aperture_beam(
            theta=zenith_angles,
            phi=azs,
            frequency=frequency,
            diameter=d,
            aperture_shape=self.aperture_shape,
            taper=self.taper,
            edge_taper_dB=self.edge_taper_dB,
            feed_model=self.feed_model,
            feed_computation=self.feed_computation,
            feed_params=self.feed_params,
            reflector_type=self.reflector_type,
            magnification=self.magnification,
            aperture_params=self.aperture_params,
        )

        return backend.asarray(jones, dtype=np.complex128)

    def is_diagonal(self) -> bool:
        return True

    def get_config(self) -> dict[str, Any]:
        config = super().get_config()
        config.update(
            {
                "diameter": self.diameter,
                "aperture_shape": self.aperture_shape,
                "taper": self.taper,
                "edge_taper_dB": self.edge_taper_dB,
                "feed_model": self.feed_model,
            }
        )
        return config


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
    # Jones matrix classes
    "BeamJones",
    "AnalyticBeamJones",
    "FITSBeamJones",
    # Analytic beam
    "compute_aperture_beam",
    # FITS beam handling
    "astropy_az_to_uvbeam_az",
    "BeamFITSHandler",
    "BeamManager",
]
