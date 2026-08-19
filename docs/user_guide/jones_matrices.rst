Jones Matrix Framework
======================

RadioSim exposes a Jones-term framework in which every exported term carries
real physics and says so. The high-level ``Simulator`` always applies geometric
phase (K, a separate per-baseline function), the private solver-owned E-Jones
primary beam — a scalar complex voltage on the diagonal, except when SCI-005
Stage-2 beam squint is configured, where it is generally full — and the
exported receptor configuration (C) and output basis transform (H). Nine
further exported Jones terms —
gain (G), bandpass (B), cable reflection (Rc), instrumental delay (Kd),
cross-hand phase and delay (X), polarization leakage (D), parallactic angle (P),
troposphere (T), and ionosphere (Z) — are applied when the ``jones:`` section
configures them; see :doc:`jones_terms` for each one's mathematics, units,
citation, and configuration. The per-antenna matrix chain contains the three
solver-owned factors ``H``, ``C``, and ``E`` plus any configured optional term;
K remains outside that chain.

Two further exported terms — ``M`` (per-baseline closure error) and ``Q`` (time
and bandwidth smearing) — are **not** ``JonesTerm`` at all: both are
``JonesBaselineTerm``, applied by Hadamard product to finished visibilities
rather than by matrix multiplication. They are implemented too. All thirteen
exported concrete terms (eleven ``JonesTerm`` classes plus ``M`` and ``Q``)
declare ``term_status: implemented``; none is an unconditional identity stub,
and no optional block may resolve to an identity Jones or baseline factor.
Parameter-dependent identity cases remain where an always-present physical
transform is genuinely neutral. Tier 7 of the remediation programme is what
turned each term from a name into physics, one slice at a time, and ``M`` and
``Q`` were the last two.

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
the accepted scalar E-Jones response is a complex voltage on the diagonal of
a 2x2
matrix. It does not claim arbitrary cross-polarization or receptor coupling.

SCI-005 Stage 2's ``beams.squint`` (:ref:`stage2-beam-squint`) is the one
exception, and it is accepted only on the ``analytic`` mode: the two native
feeds sample the same scalar pattern at oppositely displaced directions, and
the beam runtime composes the resulting :math:`E = C^\dagger D_b\,C` from
those samples and the antenna's own resolved receptor matrix. An antenna
without squint keeps today's byte-identical scalar response.

The high-level API does not accept a caller-supplied Jones chain. General
polarized BeamFITS remains outside the accepted scalar subset; a circular
receptor basis is expressed by the receptor terms below rather than by the beam
evaluator.

Polarization conventions
------------------------

Stokes :math:`I` is total intensity, :math:`Q` and :math:`U` are linear
referenced to the celestial frame with position angle measured from North
through East, and :math:`V = \mathrm{RCP} - \mathrm{LCP}` in the IAU sense. In a
canonical sky basis ordered ``(North, East)``, the brightness matrix is

.. math::

   B = \frac{1}{2}
   \begin{bmatrix} I + Q & U + iV \\ U - iV & I - Q \end{bmatrix}.

The one-half factor is RadioSim's half-power convention, so
:math:`\operatorname{Tr}(B)=I` rather than :math:`2I`. In an ideal matched
unit-response system this also gives :math:`V_{xx} + V_{yy} = I` and
:math:`V_{RR} + V_{LL} = I`; heterogeneous or non-unitary Jones chains need not
preserve those sums. The ``U + iV`` placement in the ``[0, 1]`` element is the
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

with :math:`M(\mathrm{linear}) = P` and :math:`M(\mathrm{circular}) = S`, where

.. math::

   P = \begin{bmatrix} 0 & 1 \\ 1 & 0 \end{bmatrix}

maps ``(North, East)`` sky columns to ``(X=east, Y=north)`` receptor rows, and

.. math::

   S = \frac{1}{\sqrt{2}}
   \begin{bmatrix} 1 & i \\ 1 & -i \end{bmatrix}

maps the same sky columns to unitary circular rows ordered :math:`(R, L)`.

``H`` (``BasisTransformJones``) is which basis the result is reported in:
:math:`H=M_\mathrm{output}M_\mathrm{native}^H`.  This gives :math:`I_2` when
native and output bases match, :math:`SP` for linear native into
``circular_rl``, and :math:`PS^H` for circular native into ``linear_xy``. Both
terms are unitary and always present. For the default unrotated linear array,
:math:`C=P` and :math:`H=I_2`.

For two antennas sharing one basis with zero rotation, the resulting
correlations are

.. math::

   \begin{aligned}
   V_{xx} &= (I - Q)/2, & V_{xy} &= (U - iV)/2, \\
   V_{yx} &= (U + iV)/2, & V_{yy} &= (I + Q)/2,
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
:math:`RL` by :math:`e^{-2i\chi}`. For an unpolarized source with a matched
unitary receptor/reporting transform and common scalar response :math:`c`, the
reported matrix is :math:`c I_2`: its trace is :math:`2c` and the cross hands
vanish in every supported basis. The unit-response case has :math:`c=I/2`.

Cross-basis reporting and interpretation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Converting a circular-native antenna into a linear output basis, or the reverse,
is always an exact unitary coordinate change applied after the antenna's
native-feed effects. What is conditional is a different interpretation: the
reported result is equivalent to a hypothetical antenna whose same component
matrices acted natively in the output basis only when those matrices commute
with ``H``. Reachable non-commuting cases include:

* ``jones.D`` — any non-zero leakage. ``D`` is not diagonal, so it does not
  commute with the basis change ``H``, and the reported correlations carry
  ``H D H^{H}`` rather than ``D``. The difference from the hypothetical
  output-native instrument is first order in :math:`|d|`.
* ``jones.G``, ``jones.B``, ``jones.Kd`` or ``jones.Rc`` configured **per feed**
  — a gain, bandpass, delay or reflection that differs between an antenna's two
  feeds. Each is diagonal, so it commutes with ``H`` only when its two diagonal
  entries agree; a feed-asymmetric one does not, and the error is first order in
  the feed ratio :math:`(g_0 - g_1)/(g_0 + g_1)`. A feed-symmetric value is
  scalar and commutes with ``H``. ``jones.X`` is per construction a *relative*
  phase between the two feeds; whenever its two diagonal entries differ it is
  not feed-symmetric and generally does not commute with ``H``. If a resolved
  ``X`` matrix is exactly :math:`I_2`, it is neutral and commutes trivially.

RadioSim makes no approximation here: the forward model applies each term in
the antenna's own basis at its own place in the chain (Section 12.2 of
``Tier7JonesSciencePlan.md``), and the reported cube is the exact coordinate
transform of that result. The only invalid shortcut is interpreting a
cross-basis result as though the same non-commuting electronics belonged to a
native output-basis antenna. To inspect receptor-frame quantities directly, set
``receptors.output_basis`` to the antennas' own basis, which makes ``H`` the
identity.

Elliptical and non-orthogonal feed pairs are still rejected rather than
approximated: those would break the *receptor* model itself, not merely the
reporting basis.

Parallactic-angle boundary
~~~~~~~~~~~~~~~~~~~~~~~~~~

``feed_rotation_deg`` is the **static** part of the receptor orientation, in the
topocentric frame, for the whole observation. ``C`` is therefore
time-independent, and the time-dependent part is a separate term: ``P``
(``ParallacticAngleJones``), which is implemented and documented in
:doc:`jones_terms`.

The two compose, and this is the composition the ordering below exists for:

.. math::

   \begin{aligned}
   \alpha_p &= \eta_p\psi_p(s,t)+\nu_p\mathrm{el}(s,t), \\
   C_p\,P_p &= M(\mathrm{basis}_p)R(\chi_p)R(\alpha_p) \\
             &= M(\mathrm{basis}_p)R(\chi_p+\alpha_p).
   \end{aligned}

so the static feed rotation and the field rotation **add**. There is no
double-rotation and no dropped one: enabling ``jones.P`` on an array with a
non-zero ``feed_rotation_deg`` is legal, and the composite is the receptor at
:math:`\chi_p+\alpha_p`. Ordinary alt-az has :math:`\alpha_p=\psi_p`; Nasmyth
right/left retain the signed elevation term. Earlier releases rejected that
combination outright; see :doc:`../migration_guide`.

Which antennas rotate is a property of the **instrument**, not of the
``receptors`` section: each antenna's ``mount_type`` decides it. ``alt-az`` and
the two Nasmyth variants rotate; ``equatorial``, ``fixed`` and an unspecified
mount do not. An array with a rotating mount and no ``jones.P`` is rejected, and
an array with no rotating mount that configures ``jones.P`` is rejected as well,
because the term would be exactly :math:`I_2`. See :doc:`jones_terms` for both
messages.

Chain order
-----------

The canonical factorization, leftmost factor nearest the correlator, is

.. math::

   J_p = H_p\, G_p\, B_p\, Rc_p\, Kd_p\, X_p\, D_p\, C_p\, E_p\, P_p\, T_p\, Z_p,

with the geometric phase K applied separately by the solver. ``H`` is leftmost
because it is a reporting-basis change performed at the correlator, and ``C``
sits between the sky-side direction-dependent terms (``E``, ``P``, ``T``, ``Z``)
and the electronics-side direction-independent terms (``D``, ``X``, ``Kd``,
``Rc``, ``B``, ``G``), because leakage, delays and gains are defined in the
receptor's own basis. ``JonesChain`` composes
``terms[0] @ terms[1] @ ... @ terms[-1]``, so terms are added in that same
left-to-right order and the leftmost factor is applied last.

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
* **Physical, and tested.** ``P`` sits **sky-side** of ``C`` and ``E``. A field
  rotation acts on the incoming field in the linear topocentric frame, before
  the receptor sees it, so
  :math:`C_p P_p=M(\mathrm{basis})R(\chi_p+\alpha_p)` is a single rotation of
  the receptor pair. Earlier releases placed ``P``
  correlator-side of ``C``, following Tier 5's factorization; that is wrong for
  a circular receptor, where it would apply a real 2x2 rotation to the
  :math:`(R, L)` pair. Since
  :math:`S R(\alpha_p)=\mathrm{diag}(e^{-i\alpha_p},e^{+i\alpha_p})S`, the
  correct effect on circular polarizations is a pair of opposite phases, which
  multiplies :math:`V_{RL}` by :math:`e^{-2i\alpha_p}` and :math:`V_{LR}` by
  :math:`e^{+2i\alpha_p}`. The retired pre-SCI-006 linear binding
  used :math:`M_{\mathrm{old}}=I_2`; that special case made the two placements
  agree and hid the error while ``P`` did not exist. The current east-X binding
  uses :math:`M(\mathrm{linear})=P`, and
  :math:`P R(\alpha_p) \ne R(\alpha_p)P` for a generic rotation with
  :math:`\sin(\alpha_p) \ne 0`. The executable oracle
  therefore distinguishes the orders for both linear and circular receptors.
  See :doc:`../migration_guide`.
* **Physical, and tested when beam squint is enabled.** The relative order of
  ``E`` and ``P``. For every beam declaration except SCI-005 Stage-2
  ``beams.squint``, ``E`` is a scalar complex voltage on the diagonal, so it
  commutes with everything and ``C E P`` and ``C P E`` remain numerically
  identical — the order was fixed at ``C E P`` in anticipation of a future
  non-scalar ``E``. Beam squint is that non-scalar ``E``: the chain order does
  not change, but it is no longer a convention with nothing to test. On a
  rotated **linear** receptor, ``C E P`` (physically correct, because a field
  rotation acts on the incoming field before the receptor and the beam see
  it — the same reasoning as the ``P``/``C`` bullet above) and ``C P E``
  differ, and this is an executable order-matters oracle. On *any* **circular**
  receptor the composed
  :math:`E = C^\dagger \operatorname{diag}(b_0, b_1)\, C` reduces to
  :math:`\frac{b_0+b_1}{2}I_2 - \frac{b_0-b_1}{2}\sigma_y`,
  independent of the feed rotation :math:`\chi`, which commutes exactly with
  every real rotation :math:`R(\theta) = \exp(i\theta\sigma_y)`; a
  circular-receptor order control is therefore identically zero by this exact
  algebraic identity, not because the physical effect is absent — squint still
  leaves :math:`|E_{01}| = |b_0 - b_1|/2 > 0`. See :ref:`stage2-beam-squint`.

The two terms outside the chain
-------------------------------

``M`` (per-baseline closure error) and ``Q`` (time and bandwidth smearing) are
implemented, and neither is in the chain above. They are baseline-dependent, so
they cannot be written as :math:`J_p C J_q^H` at all, and they apply by
**Hadamard product** — element by element — rather than by matrix
multiplication:

* ``Q`` is a real attenuation per ``(baseline, direction)``, multiplied into the
  visibility sum beside the Gaussian morphology envelope;
* ``M`` is a complex 2×2 per baseline, multiplied into the finished correlation
  matrix.

They descend from ``JonesBaselineTerm`` rather than ``JonesTerm``, and
``JonesChain.add_term`` rejects them by type: a category error that used to
surface as an ``AttributeError`` deep inside evaluation is now a ``TypeError``
at the point of the mistake. See :doc:`jones_terms` for the mathematics of each.

No term is planned any more
---------------------------

Every exported term declares ``term_status: implemented``. Both evaluation
contracts — ``JonesTerm.compute_jones_batch`` and
``JonesBaselineTerm.compute_baseline_factor`` — are ``@abstractmethod``, so a
term that does not implement the contract cannot be constructed at all, and
there is no exported class whose physics is a promise.

Every other Jones class that this package once exported has been removed rather
than kept as a placeholder: turbulent and GPS ionospheres, w-phase and
w-projection, element beams and array factors, fringe fitting, and the leakage,
bandpass, gain and field-rotation variants that were parameterizations of a term
rather than terms of their own -- a per-direction ``P`` subsumes the last of
those exactly. See :doc:`../migration_guide` for the
replacement for each name.

Low-level framework use
-----------------------

The API reference exposes low-level classes for development and inspection.
Read their individual docstrings and tests before use. Beam ownership remains
with the high-level canonical system; numeric analytic primitives do not form
a second configuration or runtime API.
