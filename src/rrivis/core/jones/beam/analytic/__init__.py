"""Analytic aperture beam models for the E-Jones (primary beam) in the RIME.

This sub-package provides the :class:`AnalyticBeamJones` Jones term and its
underlying components: aperture shapes, illumination tapers, feed models,
and the composed beam function :func:`compute_aperture_beam`.

Modules
-------
composed
    Composed aperture beam model (``compute_aperture_beam``).
aperture
    Aperture shape far-field patterns (Airy, sinc, elliptical Airy).
taper
    Illumination taper functions (uniform, Gaussian, parabolic, cosine).
feed
    Feed pattern models and reflector geometry (prime-focus, Cassegrain).
numerical_hpbw
    Numerical HPBW finder for arbitrary beam patterns.
"""

from typing import Any

import numpy as np

from rrivis.core.jones.beam import BeamJones
from rrivis.core.jones.beam.analytic.composed import compute_aperture_beam


class AnalyticBeamJones(BeamJones):
    """Beam Jones term using aperture-based analytic beam patterns.

    Combines aperture shape, illumination taper, feed model, and
    cross-polarization model into a full 2x2 Jones matrix via
    :func:`~rrivis.core.jones.beam.analytic.composed.compute_aperture_beam`.

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

    def plot(
        self,
        frequency: float | None = None,
        **kwargs: Any,
    ) -> Any:
        """Plot the beam pattern for this Jones term configuration.

        Convenience method that calls
        :func:`~rrivis.core.jones.beam.analytic.plotting.plot_beam_pattern`
        with this instance's beam parameters.

        Parameters
        ----------
        frequency : float or None
            Frequency in Hz. Defaults to ``self.frequencies[0]``.
        **kwargs
            Additional keyword arguments forwarded to
            :func:`plot_beam_pattern`.

        Returns
        -------
        matplotlib.figure.Figure
            The figure containing the beam pattern plot.
        """
        from rrivis.core.jones.beam.analytic.plotting import plot_beam_pattern

        if frequency is None:
            frequency = float(self.frequencies[0])

        return plot_beam_pattern(
            diameter=self.diameter,
            frequency=frequency,
            aperture_shape=self.aperture_shape,
            taper=self.taper,
            edge_taper_dB=self.edge_taper_dB,
            feed_model=self.feed_model,
            feed_computation=self.feed_computation,
            feed_params=self.feed_params,
            reflector_type=self.reflector_type,
            magnification=self.magnification,
            aperture_params=self.aperture_params,
            **kwargs,
        )


from rrivis.core.jones.beam.analytic.plotting import (  # noqa: E402
    plot_beam_2d,
    plot_beam_comparison,
    plot_beam_pattern,
    plot_feed_illumination,
)

__all__ = [
    "AnalyticBeamJones",
    "compute_aperture_beam",
    "plot_beam_pattern",
    "plot_beam_comparison",
    "plot_beam_2d",
    "plot_feed_illumination",
]
