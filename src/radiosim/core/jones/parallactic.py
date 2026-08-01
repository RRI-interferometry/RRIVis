"""The parallactic-angle term (P).

``P_p(s, t)`` is the rotation of an antenna's feeds relative to the sky frame::

    P_p = [[cos psi, sin psi], [-sin psi, cos psi]]

with ``psi`` the parallactic angle, which for an alt-azimuth mount varies with
hour angle, declination and site latitude -- and therefore *across the field*,
which is why ``P`` is direction-dependent rather than a per-antenna scalar
rotation.

Planned, not implemented.  Tier 7F implements it, direction-batched, with the
five mount types (alt-az, equatorial, fixed, alt-az+nasmyth-l,
alt-az+nasmyth-r), and unlocks ``instrument`` mount types beyond ``fixed``
(``Tier7JonesSciencePlan.md`` Section 9.1, defect D16).  A per-direction ``P``
subsumes what used to be a separate field-rotation class exactly, and
heterogeneous VLBI mounts are a per-antenna ``mount_type`` that
``ResolvedInstrument`` already carries, not a separate class.

Tier 7F also moves ``P`` sky-side of ``C`` in the canonical chain (design
decision D12): for a circular receptor the Tier 5 order ``C P`` and the correct
order ``P C`` differ, and the error is unobservable today only because ``P``
does not exist.

References
----------
Thompson, Moran & Swenson (2017), *Interferometry and Synthesis in Radio
Astronomy*, 3rd ed., Section 4.6.
Smirnov (2011), A&A 527, A106 (Paper I), Section 6.4.
"""

from .base import JonesTerm


class ParallacticAngleJones(JonesTerm):
    """Parallactic-angle rotation ``P`` (planned; Tier 7F implements it).

    ``term_status`` is ``"planned"``: constructing it is allowed, evaluating it
    raises.  See :class:`~radiosim.core.jones.gain.GainJones` for why it takes
    no parameters yet.
    """

    @property
    def name(self) -> str:
        return "P"

    @property
    def is_direction_dependent(self) -> bool:
        return True
