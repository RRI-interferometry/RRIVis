"""Delay-like instrumental terms: the electronic delay (Kd) and cable
reflection (Rc).

``Kd`` is a per-antenna, per-feed instrumental delay offset::

    Kd = diag(exp(-2 pi i nu tau_0), exp(-2 pi i nu tau_1))

whose negative-exponent sign matches the geometric phase's own, so that a
positive delay produces ``exp(-i * positive)`` on both paths (invariant I4).

``Rc`` is the standing-wave ripple from a reflection in the RF cable::

    Rc = diag(1 + A exp(-2 pi i nu tau_c), ...),      0 < |A| < 1

which is a distinct, non-pure-phase frequency structure from the bandpass, and
is why it is a term of its own rather than a bandpass model.

Planned, not implemented.  Tier 7E implements both.  There is no
fringe-fitting term: fringe fitting is a *calibration solution*, and its
forward-model content is exactly ``G`` times ``Kd`` times a phase rate
(``Tier7JonesSciencePlan.md`` Section 9.1, Section 4).

References
----------
Thompson, Moran & Swenson (2017), 3rd ed., Chapter 7.
Ewall-Wice et al. (2016), MNRAS 460, 4320 -- cable reflections in HERA/PAPER.
"""

from radiosim.core.jones.base import JonesTerm


class DelayJones(JonesTerm):
    """Instrumental delay ``Kd`` (planned; Tier 7E implements it).

    ``term_status`` is ``"planned"``: constructing it is allowed, evaluating it
    raises.  See :class:`~radiosim.core.jones.gain.GainJones` for why it takes
    no parameters yet.
    """

    @property
    def name(self) -> str:
        return "Kd"

    @property
    def is_direction_dependent(self) -> bool:
        return False


class CableReflectionJones(JonesTerm):
    """Cable-reflection ripple ``Rc`` (planned; Tier 7E implements it).

    ``term_status`` is ``"planned"``: constructing it is allowed, evaluating it
    raises.  See :class:`~radiosim.core.jones.gain.GainJones` for why it takes
    no parameters yet.
    """

    @property
    def name(self) -> str:
        return "Rc"

    @property
    def is_direction_dependent(self) -> bool:
        return False
