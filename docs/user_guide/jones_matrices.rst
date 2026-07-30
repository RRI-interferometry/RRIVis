Jones Matrix Framework
======================

RadioSim exposes a broad Jones-term framework, but public class availability is
not the same as implemented high-level science. The current high-level
``Simulator`` uses geometric phase (K), the canonical scalar E-Jones primary
beam, the receptor configuration (C), and the output basis transform (H) as
substantive forward-model effects. Many other exported terms remain identity
scaffolds or later-tier work.

RIME context
------------

For baseline :math:`ij`, RadioSim uses the usual matrix form

.. math::

   V_{ij} = \sum_s J_i(\vec{s}) C_s J_j^H(\vec{s}).

Here :math:`C_s` is the source coherency (brightness) matrix. It is unrelated
to the receptor-configuration term named ``C`` below; the Jones-term letters and
the RIME symbols are separate namespaces.

The framework organizes geometric, beam, propagation, gain, bandpass,
polarization, and other terms. Only terms with implemented calculations and
scientific tests should be interpreted as changing simulated visibilities.

Current high-level effects
--------------------------

Geometric phase is calculated by the direct-sum visibility path. Primary beams
use the strict tagged configuration, for example:

.. code-block:: yaml

   beams:
     mode: analytic
     model:
       kind: circular_aperture
       taper:
         kind: gaussian
         edge_taper_db: 10.0

``analytic``, ``shared_fits``, ``per_antenna_fits``, and ``mixed`` all resolve
to one canonical per-antenna ``BeamSystem``. The point-source, HEALPix, and
observability paths use the same evaluator. Within the accepted FITS subset,
the E-Jones response is a scalar complex voltage on the diagonal of a 2x2
matrix. It does not claim arbitrary cross-polarization or receptor coupling.

The high-level API does not accept a caller-supplied Jones chain. General
polarized BeamFITS remains outside the accepted scalar subset; a circular
receptor basis is expressed by the receptor terms below rather than by the beam
evaluator.

Polarization conventions
------------------------

Stokes :math:`I` is total intensity, :math:`Q` and :math:`U` are linear
referenced to the celestial frame with position angle measured from North
through East, and :math:`V = \mathrm{RCP} - \mathrm{LCP}` in the IAU sense. In a
right-handed linear sky basis mapped onto an antenna's nominal ``(x, y)`` feeds,
the brightness matrix is

.. math::

   B = \frac{1}{2}
   \begin{bmatrix} I + Q & U + iV \\ U - iV & I - Q \end{bmatrix}.

The one-half factor is RadioSim's half-power convention, so
:math:`V_{xx} + V_{yy} = I` and :math:`V_{RR} + V_{LL} = I` rather than
:math:`2I`. The ``U + iV`` placement in the ``[0, 1]`` element is the
literature convention for linear feeds under the IAU definition of :math:`V`;
earlier RadioSim releases carried the mirror-image sign, which is only
observable in the cross hands of a source with non-zero :math:`V`. See
:doc:`../migration_guide`.

Receptor and basis terms
------------------------

The ``receptors`` configuration section (see :doc:`configuration`) resolves once
into an array-wide receptor inventory and exactly one output basis. Two chain
terms carry it.

``C`` (``ReceptorConfigJones``) is what the receptor physically is. For an
antenna with feed rotation :math:`\chi`,

.. math::

   C_p = M(\mathrm{basis}_p)\, R(\chi_p), \qquad
   R(\chi) = \begin{bmatrix} \cos\chi & \sin\chi \\
                            -\sin\chi & \cos\chi \end{bmatrix},

with :math:`M(\mathrm{linear}) = I_2` and :math:`M(\mathrm{circular}) = S`,
where

.. math::

   S = \frac{1}{\sqrt{2}}
   \begin{bmatrix} 1 & i \\ 1 & -i \end{bmatrix}

is the unitary linear-to-circular matrix with rows ordered :math:`(R, L)`.

``H`` (``BasisTransformJones``) is which basis the result is reported in:
:math:`I_2` when an antenna's native basis already matches the output basis,
:math:`S` for linear native into ``circular_rl``, and :math:`S^H` for circular
native into ``linear_xy``. Both terms are unitary, and both are always present
in the chain. When every antenna is linear with zero rotation and the output
basis is ``linear_xy``, both are exactly :math:`I_2` and results are
bit-identical to a run with no ``receptors`` section at all.

For two antennas sharing one basis with zero rotation, the resulting
correlations are

.. math::

   \begin{aligned}
   V_{xx} &= (I + Q)/2, & V_{xy} &= (U + iV)/2, \\
   V_{yx} &= (U - iV)/2, & V_{yy} &= (I - Q)/2,
   \end{aligned}

in the linear basis and

.. math::

   \begin{aligned}
   V_{RR} &= (I + V)/2, & V_{RL} &= (Q + iU)/2, \\
   V_{LR} &= (Q - iU)/2, & V_{LL} &= (I - V)/2,
   \end{aligned}

in the circular basis. A linear feed rotation by :math:`\chi` rotates
:math:`(Q, U)` by :math:`2\chi` and leaves :math:`I` and :math:`V` unchanged; a
circular rotation leaves :math:`RR` and :math:`LL` unchanged and multiplies
:math:`RL` by :math:`e^{-2i\chi}`. For an unpolarized source,
:math:`V[0,0] + V[1,1] = I` and the cross hands vanish in every supported basis
and at every feed rotation.

Modelling assumption
~~~~~~~~~~~~~~~~~~~~

Converting a circular-native antenna into a linear output basis, or the reverse,
is exact **only** when both feeds are ideal, orthogonal, and share a common
complex gain. That holds today because the polarization-leakage term ``D`` and
the gain term ``G`` are disabled identity stubs. When Tier 7 implements ``D``,
the conversion becomes approximate and this statement must be re-examined.
Elliptical and non-orthogonal feed pairs are rejected rather than approximated
for the same reason.

Parallactic-angle boundary
~~~~~~~~~~~~~~~~~~~~~~~~~~

``ParallacticAngleJones`` (``P``) is an identity stub, and only
``mount_type: fixed`` is accepted, so ``feed_rotation_deg`` is a **static**
rotation in the topocentric frame for the whole observation. ``C`` is therefore
time-independent. Enabling ``P`` while a non-zero ``feed_rotation_deg`` is
present is rejected with an explicit error, because the composed result would
silently omit the time-dependent part of the rotation. When Tier 7 implements
``P``, the composition :math:`P_p(t)\,C_p` becomes the full time-dependent
receptor orientation and that rejection is removed.

Chain order
-----------

The canonical factorization, leftmost factor nearest the correlator, is

.. math::

   J_p = H_p\, G_p\, B_p\, D_p\, P_p\, C_p\, E_p\, T_p\, Z_p,

with the geometric phase K applied separately by the solver. ``H`` is leftmost
because it is a reporting-basis change performed at the correlator, and ``C``
sits between the sky-side direction-dependent terms (``E``, ``T``, ``Z``) and
the electronics-side direction-independent terms (``D``, ``G``, ``B``), because
leakage and gains are defined in the receptor's own basis. ``JonesChain``
composes ``terms[0] @ terms[1] @ ... @ terms[-1]``, so terms are added in that
same left-to-right order and the leftmost factor is applied last.

Scaffolded terms
----------------

Ionosphere, troposphere, parallactic rotation, gain, bandpass, polarization
leakage, and other exported terms are not advertised as complete high-level
effects. Until Tier 7 and the later scientific tiers add conventions, analytic
invariants, reference comparisons, backend parity, and a test proving a
configured effect changes visibility, users should not include them in
scientific claims.

Low-level framework use
-----------------------

The API reference exposes low-level classes for development and inspection.
Read their individual docstrings and tests before use. A class returning an
identity matrix is a scaffold, not a physical model. Beam ownership remains
with the high-level canonical system; numeric analytic primitives do not form
a second configuration or runtime API.
