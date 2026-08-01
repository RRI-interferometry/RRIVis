"""The tropospheric term (T).

``T_p(s, nu, t)`` carries the neutral atmosphere's two effects in one
antenna-side factor: excess path delay, as a zenith delay times an elevation
mapping function, and opacity attenuation::

    T = exp(-2 pi i nu * ZD * m(el)) * exp(-tau(el) / 2) * I2

The ``1/2`` in the opacity exponent is the voltage convention: the visibility
*power* is attenuated by ``exp(-tau)`` on a baseline of two identical antennas.
Both effects are direction-dependent through the elevation.

Planned, not implemented.  Tier 7G implements delay and opacity as one term
with ``zenith_delay`` and ``opacity`` sub-blocks, which is why there is no
separate opacity class and no separate Saastamoinen class -- the model is a
field, not a subclass (``Tier7JonesSciencePlan.md`` Section 9.1).  Stochastic
turbulent screens are out of scope (Section 4).

References
----------
Saastamoinen (1972), in *The Use of Artificial Satellites for Geodesy*,
Geophys. Monogr. Ser. 15, 247.
Niell (1996), J. Geophys. Res. 101, 3227 -- the mapping functions.
Thompson, Moran & Swenson (2017), 3rd ed., Chapter 13.
"""

from .base import JonesTerm


class TroposphereJones(JonesTerm):
    """Tropospheric delay and opacity ``T`` (planned; Tier 7G implements it).

    ``term_status`` is ``"planned"``: constructing it is allowed, evaluating it
    raises.  See :class:`~radiosim.core.jones.gain.GainJones` for why it takes
    no parameters yet.
    """

    @property
    def name(self) -> str:
        return "T"

    @property
    def is_direction_dependent(self) -> bool:
        return True
