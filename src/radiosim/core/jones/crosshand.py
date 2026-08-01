"""The cross-hand term (X).

``X`` is the relative phase between an antenna's two feed paths, constant in
frequency or linear in it::

    X = diag(exp(+i (phi_x + 2 pi nu tau_x) / 2),
             exp(-i (phi_x + 2 pi nu tau_x) / 2))

Cross-hand phase and cross-hand delay are the same diagonal matrix -- one
constant term and one linear-in-frequency term -- so they are one term with two
parameters rather than two classes (``Tier7JonesSciencePlan.md`` Section 9.1).

Planned, not implemented.  Tier 7E implements it.  There is no separate
frequency-dependent leakage class: ``D`` is frequency-capable by construction,
and a second class for the same matrix with a frequency axis is duplication.

References
----------
Sault, Hamaker & Bregman (1996), A&AS 117, 149.
Thompson, Moran & Swenson (2017), 3rd ed., Chapter 7.
"""

from radiosim.core.jones.base import JonesTerm


class CrosshandJones(JonesTerm):
    """Cross-hand phase and delay ``X`` (planned; Tier 7E implements it).

    Tier 7C renamed this term and folded the former separate cross-hand delay
    class into it, because the two are the same diagonal matrix; see
    ``docs/migration_guide.md`` for both old names.

    ``term_status`` is ``"planned"``: constructing it is allowed, evaluating it
    raises.  See :class:`~radiosim.core.jones.gain.GainJones` for why it takes
    no parameters yet.
    """

    @property
    def name(self) -> str:
        return "X"

    @property
    def is_direction_dependent(self) -> bool:
        return False
