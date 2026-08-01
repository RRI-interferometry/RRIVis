"""The complex electronic gain term (G).

``G_p`` is the per-antenna, per-feed complex voltage gain of the receiving
chain downstream of the feed::

    G_p = diag(g_p0, g_p1)

one complex number per feed.  It is direction-independent and, absent a time or
elevation model, constant over the observation.

Planned, not implemented.  Tier 7D implements it, absorbing what used to be
three separate classes: a time model becomes a ``time_model`` field and an
elevation gain curve becomes an ``elevation_curve`` field, because both multiply
the same diagonal matrix (``Tier7JonesSciencePlan.md`` Section 9.1).

References
----------
Hamaker, Bregman & Sault (1996), A&AS 117, 137 -- the ``G`` factorization of the
measurement equation.
Smirnov (2011), A&A 527, A106 (Paper I), Section 6.
"""

from radiosim.core.jones.base import JonesTerm


class GainJones(JonesTerm):
    """Complex electronic gains ``G`` (planned; Tier 7D implements it).

    ``term_status`` is ``"planned"``: the term has a name, a chain position and
    a documented physical effect, and
    :meth:`~radiosim.core.jones.base.JonesTerm.compute_jones_batch` raises.  It
    takes no parameters, because a parameter it cannot honour is a parameter it
    would silently discard (defect D2); Tier 7D introduces the real constructor
    together with the ``jones.G`` configuration that validates it, and declares
    the capability flags that its invariant sweep can then verify.
    """

    @property
    def name(self) -> str:
        return "G"

    @property
    def is_direction_dependent(self) -> bool:
        return False
