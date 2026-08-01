"""The bandpass term (B).

``B_p(nu)`` is the per-antenna, per-feed complex frequency response of the
signal chain::

    B_p(nu) = diag(b_p0(nu), b_p1(nu))

It is the frequency-dependent counterpart of ``G``: the same diagonal matrix,
with structure across the band rather than across time.

Planned, not implemented.  Tier 7D implements it with a ``model`` field whose
kinds replace the former polynomial and spline subclasses.  There is no
RFI-flagging variant: flagging is a data-quality product, not a voltage-domain
Jones factor, and RadioSim's result contract has no flag array for it to write
(``Tier7JonesSciencePlan.md`` Section 9.1).

References
----------
Hamaker, Bregman & Sault (1996), A&AS 117, 137.
Smirnov (2011), A&A 527, A106 (Paper I), Section 6.
"""

from .base import JonesTerm


class BandpassJones(JonesTerm):
    """Frequency-dependent bandpass ``B`` (planned; Tier 7D implements it).

    ``term_status`` is ``"planned"``: constructing it is allowed, evaluating it
    raises.  See :class:`~radiosim.core.jones.gain.GainJones` for why it takes
    no parameters yet.
    """

    @property
    def name(self) -> str:
        return "B"

    @property
    def is_direction_dependent(self) -> bool:
        return False
