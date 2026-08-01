"""The ionospheric term (Z).

``Z_p(s, nu, t)`` carries both ionospheric effects in one factor: the
dispersive phase from the total electron content along the line of sight, and
the ionospheric Faraday rotation from the electron content weighted by the
line-of-sight magnetic field::

    phi_disp = -2 pi k_TEC * sTEC / nu,      k_TEC = 1.3445e9 Hz TECU^-1
    Z        = exp(i phi_disp) * R(RM_ion lambda^2)

Both are direction-dependent through the thin-shell slant mapping.

Planned, not implemented.  Tier 7G implements it.  ``Z`` owns *ionospheric*
Faraday rotation only: the intrinsic rotation measure of a source is already
applied by the sky model inside the solver's frequency loop, so a separate,
free-standing ``F`` term would rotate the same emission twice
(``Tier7JonesSciencePlan.md`` Section 11, defect D18).  Stochastic turbulent
screens and IONEX/GPS ingestion are out of scope (Section 4).

References
----------
Thompson, Moran & Swenson (2017), 3rd ed., eq. 13.128.
Intema et al. (2009), A&A 501, 1185.
"""

from .base import JonesTerm


class IonosphereJones(JonesTerm):
    """Ionospheric TEC phase and Faraday rotation ``Z`` (planned; Tier 7G).

    ``term_status`` is ``"planned"``: constructing it is allowed, evaluating it
    raises.  See :class:`~radiosim.core.jones.gain.GainJones` for why it takes
    no parameters yet.
    """

    @property
    def name(self) -> str:
        return "Z"

    @property
    def is_direction_dependent(self) -> bool:
        return True
