Jones Matrix Framework
======================

RadioSim exposes a Jones-term framework in which public class availability is
not the same as implemented high-level science, and every term says which it
is. The current high-level ``Simulator`` always applies geometric phase (K), the
canonical scalar E-Jones primary beam, the receptor configuration (C), and the
output basis transform (H). Six further terms — gain (G), bandpass (B), cable
reflection (Rc), instrumental delay (Kd), cross-hand phase and delay (X), and
polarization leakage (D) — carry real physics and are applied when the
``jones:`` section configures them; see :doc:`jones_terms` for each one's
mathematics, units, citation, and configuration. All ten are
``term_status: implemented``.

The remaining exported terms — ``P``, ``Z``, ``T``, ``M`` and ``Q`` — are
``term_status: planned``: each has a documented physical effect and a position
in the chain below, and each **raises** when evaluated. None of them multiplies
by the identity, so a term cannot silently do nothing.

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
complex gain. Two configurations now break that condition, and both are
reachable:

* ``jones.D`` — any non-zero leakage. ``D`` is not diagonal, so it does not
  commute with the basis change ``H``, and the reported correlations carry
  ``H D H^{H}`` rather than ``D``. The discrepancy between the reported
  quantity and an ideal-feed one is first order in :math:`|d|`.
* ``jones.G``, ``jones.B``, ``jones.Kd`` or ``jones.Rc`` configured **per feed**
  — a gain, bandpass, delay or reflection that differs between an antenna's two
  feeds. Each is diagonal, so it commutes with ``H`` only when its two diagonal
  entries agree; a feed-asymmetric one does not, and the error is first order in
  the feed ratio :math:`(g_0 - g_1)/(g_0 + g_1)`. A feed-symmetric value is
  scalar and remains exact. ``jones.X`` is per construction a *relative* phase
  between the two feeds, so it is never feed-symmetric and never commutes with
  ``H``.

None of this is an approximation RadioSim makes silently: the forward model
applies each term in the antenna's own basis at its own place in the chain
(Section 12.2 of ``Tier7JonesSciencePlan.md``), and the reported cube is the
exact result of that chain. What becomes approximate is the *interpretation* of
a circular-native run reported in a linear basis — or the reverse — as though it
were a linear-native one. If you need the exact receptor-frame quantities, set
``receptors.output_basis`` to the antennas' own basis, which makes ``H`` the
identity and removes the question.

Elliptical and non-orthogonal feed pairs are still rejected rather than
approximated: those would break the *receptor* model itself, not merely the
reporting basis.

Parallactic-angle boundary
~~~~~~~~~~~~~~~~~~~~~~~~~~

``ParallacticAngleJones`` (``P``) is planned rather than implemented, and only
``mount_type: fixed`` is accepted, so ``feed_rotation_deg`` is a **static**
rotation in the topocentric frame for the whole observation. ``C`` is therefore
time-independent, and no run can carry a time-dependent feed orientation: the
mount-type rejection is what enforces that. When Tier 7 implements
``P``, the composition :math:`P_p(t)\,C_p` becomes the full time-dependent
receptor orientation, ``P`` moves sky-side of ``C`` in the factorization below,
and the blanket mount-type rejection is replaced by one that names the fix.

Chain order
-----------

The canonical factorization, leftmost factor nearest the correlator, is

.. math::

   J_p = H_p\, G_p\, B_p\, Rc_p\, Kd_p\, X_p\, D_p\, P_p\, C_p\, E_p\, T_p\, Z_p,

with the geometric phase K applied separately by the solver. ``H`` is leftmost
because it is a reporting-basis change performed at the correlator, and ``C``
sits between the sky-side direction-dependent terms (``E``, ``T``, ``Z``) and
the electronics-side direction-independent terms (``D``, ``X``, ``Kd``, ``Rc``,
``B``, ``G``), because leakage, delays and gains are defined in the receptor's
own basis. ``JonesChain`` composes ``terms[0] @ terms[1] @ ... @ terms[-1]``, so
terms are added in that same left-to-right order and the leftmost factor is
applied last.

What in that order is physical, and what is convention
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* **Physical, and tested.** ``D`` sits correlator-side of ``C``: leakage is a
  property of the receiving hardware and is defined in the receptor's own basis,
  which is why its coefficients are indexed by feed 0/1 rather than by
  ``x``/``y``. The same applies to ``X``, ``Kd``, ``Rc``, ``B`` and ``G``.
* **Convention, because the factors commute.** The relative order of ``G``,
  ``B``, ``Rc``, ``Kd`` and ``X`` among themselves. All five are diagonal 2×2
  matrices in the same basis, and diagonal matrices commute. Their mutual order
  is fixed here so the chain has one shape, one provenance string and one test;
  it is **not** a physical claim. ``D`` is *not* in that set — it is
  off-diagonal, and it does not commute with a feed-asymmetric ``G``, ``B``,
  ``Kd`` or ``Rc``, nor with ``X``.
* **Not yet observable.** The placement of ``P`` shown above is Tier 5's, and it
  is unobservable while ``P`` is planned. Implementing ``P`` moves it sky-side of
  ``C``; see the parallactic-angle boundary above.

Planned terms
-------------

Ionosphere (``Z``), troposphere (``T``), parallactic rotation (``P``), and the
two baseline-Hadamard terms (``M``, ``Q``) are exported, documented, and
**not implemented**. Each declares ``term_status: planned`` and raises when
evaluated, so none of them can enter a result. Until each gains its conventions,
analytic invariants, reference comparisons, backend parity, and a test proving a
configured effect changes the visibilities, it supports no scientific claim.

Every other Jones class that this package once exported has been removed rather
than kept as a placeholder: turbulent and GPS ionospheres, w-phase and
w-projection, element beams and array factors, fringe fitting, and the leakage,
bandpass, gain and parallactic variants that were parameterizations of a term
rather than terms of their own. See :doc:`../migration_guide` for the
replacement for each name.

Low-level framework use
-----------------------

The API reference exposes low-level classes for development and inspection.
Read their individual docstrings and tests before use. Beam ownership remains
with the high-level canonical system; numeric analytic primitives do not form
a second configuration or runtime API.
