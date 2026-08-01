"""The polarization leakage term (D).

``D_p`` carries the first-order cross-coupling between an antenna's two feeds::

    D_p = [[1, d_p0], [-d_p1, 1]]

with complex, generally frequency-dependent leakage coefficients.  It is
direction-independent.

Planned, not implemented.  Tier 7D-7E implements it with a ``d_terms`` field
whose kinds subsume the former IXR parameterization; there is no separate
Mueller class, because a Mueller matrix is a derived 4x4 view of this same 2x2
Jones, and no separate frequency-dependent class, because ``D`` is
frequency-capable by construction.  Beam squint is a *beam* property and is
routed to the beam subsystem rather than modelled as a direction-dependent
D-term, which would create the second beam pathway
``Tier7JonesSciencePlan.md`` Section 4 forbids.

References
----------
Hamaker, Bregman & Sault (1996), A&AS 117, 137, Section 4.
Sault, Hamaker & Bregman (1996), A&AS 117, 149.
"""

from .base import JonesTerm


class PolarizationLeakageJones(JonesTerm):
    """Polarization leakage D-terms ``D`` (planned; Tier 7E implements it).

    ``term_status`` is ``"planned"``: constructing it is allowed, evaluating it
    raises.  See :class:`~radiosim.core.jones.gain.GainJones` for why it takes
    no parameters yet.
    """

    @property
    def name(self) -> str:
        return "D"

    @property
    def is_direction_dependent(self) -> bool:
        return False
