Jones terms
===========

This page documents every Jones term RadioSim can actually apply: its
mathematics, its units, its citation, and the configuration that enables it.
A term that is not on this page is not implemented, and configuring it is a
parse error rather than a silent no-op.

See :doc:`jones_matrices` for the chain algebra and the receptor terms, and
:doc:`configuration` for where the ``jones:`` section sits in a document.

.. contents::
   :local:
   :depth: 2


What is implemented today
-------------------------

.. list-table::
   :header-rows: 1
   :widths: 10 30 25 35

   * - Term
     - Effect
     - Configured by
     - Status
   * - ``K``
     - Geometric phase delay
     - not configurable
     - always applied, per baseline
   * - ``E``
     - Primary beam voltage pattern
     - ``beams:``
     - always applied
   * - ``C``
     - Receptor basis and static feed rotation
     - ``receptors:``
     - always applied
   * - ``H``
     - Reporting-basis transform
     - ``receptors.output_basis``
     - always applied
   * - ``G``
     - Complex electronic gain
     - ``jones.G``
     - implemented
   * - ``B``
     - Frequency-dependent bandpass
     - ``jones.B``
     - implemented
   * - ``Rc``
     - RF cable reflection ripple
     - ``jones.Rc``
     - implemented
   * - ``Kd``
     - Instrumental delay offset
     - ``jones.Kd``
     - implemented
   * - ``X``
     - Cross-hand phase and delay
     - ``jones.X``
     - implemented
   * - ``D``
     - Polarization leakage
     - ``jones.D``
     - implemented

Five further terms — ``P``, ``Z``, ``T``, ``M`` and ``Q`` — exist as named
classes with documented physics and **no implementation**: constructing one is
allowed, evaluating one raises. They have no configuration block, deliberately.
A schema field for a term that cannot be honoured would accept a value and
discard it, which is worse than refusing.


The ``jones:`` section
----------------------

Every term is absent by default, and an absent term is not in the chain at all.
A configuration with no ``jones:`` section produces exactly the visibilities it
produced before the section existed — the same numbers and the same
``scientific_sha256``.

Two rules follow from that, and both are enforced:

* There is no ``enabled: false``. To disable a term, remove its block.
* A block whose resolved parameters make the term exactly the identity is
  **rejected**. A term that cannot change the visibilities is indistinguishable
  from no term, and accepting one would reintroduce the silent-no-op behaviour
  this section exists to remove.

A ``jones:`` key that is present but configures nothing is likewise rejected,
because an empty section is a statement of intent the document does not carry
out — usually a deleted term or a mis-indented key.

Units are never implicit: every angle field ends in ``_rad`` or ``_deg``, every
frequency in ``_hz``. A complex number is written ``[re, im]``, never as a
Python complex literal and never as a string. A field that accepts a complex
value also accepts a bare real number.

.. note::

   YAML 1.1 requires a **signed** exponent for a float in scientific notation.
   Write ``1.0e+8``, not ``1.0e8``: the second parses as a *string* and is
   rejected with a type error.


G — complex electronic gain
---------------------------

``G`` is the per-antenna, per-feed complex voltage gain of the receiving chain
downstream of the feed. It is direction-independent, frequency-independent, and
diagonal:

.. math::

   G_p(t) = g_\mathrm{el}(\mathrm{el}_\mathrm{ref})
            \, \mathrm{diag}\bigl(g_{p0}(t),\, g_{p1}(t)\bigr),
   \qquad
   g_{pf}(t) = (1 + a_{pf}) \, e^{i \phi_{pf}} \, s_{pf}(t)

with :math:`a_{pf}` a fractional amplitude error, :math:`\phi_{pf}` a phase
error in radians, and :math:`f \in \{0, 1\}` the feed index **in the antenna's
own receptor basis**. ``G`` sits correlator-side of ``C``, which is why it is
defined per feed index rather than per ``x``/``y``.

*Reference:* Hamaker, Bregman & Sault (1996), A&AS **117**, 137; Smirnov (2011),
A&A **527**, A106, §6.

Configuration
~~~~~~~~~~~~~

.. code-block:: yaml

   jones:
     G:
       amplitude_error: 0.02          # fractional, array-wide default
       phase_error_rad: 0.0
       per_antenna:                   # optional; overrides the defaults
         - antenna: 12
           feed: 0                    # 0 or 1, in the antenna's own basis
           amplitude_error: 0.05
           phase_error_rad: 0.13
       elevation_curve: [1.0, -1.0e-4]  # optional; polynomial in elevation (deg)
       time_model:                    # optional; default {kind: constant}
         kind: linear_drift           # constant | linear_drift | sinusoidal
         rate_per_hour: 0.01

``per_antenna`` entries are keyed by **antenna number** and validated against
the resolved instrument: an unknown number is rejected, as is a repeated
``(antenna, feed)`` pair or a feed index outside ``{0, 1}``. An explicit entry
beats the array-wide default, and an entry may override only one of the two
values.

Time models
~~~~~~~~~~~

Three closed forms, none of which draws a random number, so two runs of the same
configuration produce the same gains bit for bit. ``t`` is measured in hours
from the **first sample of the observation**.

.. list-table::
   :header-rows: 1
   :widths: 20 50 30

   * - ``kind``
     - :math:`s(t)`
     - Fields
   * - ``constant``
     - :math:`1`
     - none
   * - ``linear_drift``
     - :math:`1 + r\,t`
     - ``rate_per_hour``
   * - ``sinusoidal``
     - :math:`1 + d \sin(2\pi t / P + \varphi)`
     - ``depth``, ``period_hours``, ``phase_rad``

The elevation gain curve
~~~~~~~~~~~~~~~~~~~~~~~~

``elevation_curve`` gives the coefficients :math:`c_k` of
:math:`g_\mathrm{el}(\mathrm{el}) = \sum_k c_k \, \mathrm{el}^k`, with the
elevation in **degrees**, lowest order first. The elevation is that of the
*pointing centre*, which is a direction-independent quantity — enabling the
curve does not make ``G`` direction-dependent.

.. warning::

   RadioSim's only phase convention is zenith drift, so the pointing elevation
   is exactly 90 degrees at every time sample and the curve evaluates to a
   single constant for the whole run. It is a real, non-identity gain — it
   scales every visibility by :math:`g_\mathrm{el}(90)` — but it does not vary,
   and it will not until RadioSim gains a steerable phase centre. If you want a
   gain that changes over the observation today, use ``time_model``.

What ``G`` does and does not do
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* ``G`` is diagonal, always.
* ``G`` is **not** unitary unless every amplitude error is zero, the time model
  is constant, and there is no elevation curve. A gain that attenuates cannot
  preserve power.
* A common real amplitude error :math:`a` on both feeds of both antennas scales
  every correlation of a baseline by exactly :math:`(1+a)^2`.
* A pure phase error leaves all four correlation amplitudes unchanged and
  changes only phases; a phase common to the whole array cancels entirely.
* Antenna-based gains leave the **closure phase** invariant. That is what
  distinguishes a Jones term from a baseline-dependent error.
* ``G`` commutes with ``B``.


B — bandpass
------------

``B`` is the per-antenna, per-feed complex frequency response of the signal
chain. It is the frequency-dependent counterpart of ``G``: the same diagonal
matrix, with structure across the band rather than across time.

.. math::

   B_p(\nu) = \mathrm{diag}\bigl(b_{p0}(\nu),\, b_{p1}(\nu)\bigr)

*Reference:* Smirnov (2011), A&A **527**, A106, §6; CASA ``bandpass``
conventions.

Configuration
~~~~~~~~~~~~~

.. code-block:: yaml

   jones:
     B:
       model:
         kind: polynomial               # polynomial | tabulated
         coefficients: [1.0, 0.0, -0.05]
         reference_frequency_hz: null   # null = band centre
         scale_frequency_hz: null       # null = half-bandwidth
       per_antenna: []                  # same (antenna, feed) keying as G

A per-antenna override replaces that feed's whole ``model``.

``polynomial``
~~~~~~~~~~~~~~

.. math::

   b(\nu) = \sum_k c_k x^k,
   \qquad
   x = \frac{\nu - \nu_\mathrm{ref}}{\nu_\mathrm{scale}}

``reference_frequency_hz`` defaults to the band centre and
``scale_frequency_hz`` to the half-bandwidth, so :math:`x` runs from exactly
:math:`-1` to :math:`+1` across the observed band and a low-order polynomial
stays well conditioned wherever in the spectrum the band sits. Coefficients may
be complex.

A single-channel observation has no half-bandwidth. Rather than substitute some
other number — which would make the same configuration mean different things at
different band widths — resolution rejects it and asks for
``scale_frequency_hz`` explicitly.

``tabulated``
~~~~~~~~~~~~~

.. code-block:: yaml

   jones:
     B:
       model:
         kind: tabulated
         node_frequencies_hz: [9.0e+7, 1.0e+8, 1.1e+8, 1.2e+8]
         gains: [[0.9, 0.0], [1.0, 0.05], [0.98, -0.02], [0.9, 0.0]]

Complex gains at explicit node frequencies, interpolated by cubic spline in the
real and imaginary parts separately. At least four strictly increasing nodes are
required, and one gain per node.

Frequencies outside the node range are **rejected, not extrapolated**. A
bandpass continued past its own measurement is a fabricated number, and
RadioSim will not fabricate one silently — if the nodes do not span every
observed channel, the run stops and says so.

What ``B`` does and does not do
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* ``B`` is diagonal, always, and frequency-dependent by definition.
* ``B`` is not unitary for any realistic configuration: a bandpass is an
  attenuation profile.
* A real bandpass :math:`b(\nu)` on both antennas takes :math:`V(\nu)` to
  :math:`b(\nu)^2 V(\nu)`, independently at each channel.
* A real, frequency-flat bandpass is exactly a ``G`` amplitude error. The two
  stay separate terms because one is *defined* to carry frequency structure and
  the other is not.


Rc — cable reflection
---------------------

``Rc`` is the standing-wave ripple a reflection in an antenna's RF cable puts
across the band. It is direction-independent, frequency-dependent, diagonal, and
**not** unitary:

.. math::

   Rc_p(\nu) = \mathrm{diag}\bigl(r_{p0}(\nu),\, r_{p1}(\nu)\bigr),
   \qquad
   r_{pf}(\nu) = 1 + A_{pf}\,
        e^{-2\pi i \nu \tau_{\mathrm{cable},pf} + i\phi_{pf}}

:math:`A` is a dimensionless reflection amplitude, :math:`\tau_\mathrm{cable}`
the **round-trip** cable delay in seconds, and :math:`\phi` a phase offset. This
is the first-order, single-bounce reflection; multiple bounces would add terms in
:math:`A^2 e^{-4\pi i\nu\tau_c}` and are out of scope.

*Reference:* Kern et al. (2020), ApJ **888**, 70; Beardsley et al. (2016),
ApJ **833**, 102; Ewall-Wice et al. (2016), MNRAS **460**, 4320.

Configuration
~~~~~~~~~~~~~

.. code-block:: yaml

   jones:
     Rc:
       amplitude: 0.01          # dimensionless; 0 < |A| < 1
       cable_delay_s: 1.5e-7    # round-trip, seconds
       phase_rad: 0.0
       per_antenna:             # same (antenna, feed) keying as G
         - antenna: 3
           feed: 1
           amplitude: 0.04

An amplitude outside :math:`0 < |A| < 1` is **rejected** — a reflection cannot
return more power than it receives, and a zero one is not a reflection. The
rejection names the physics rather than reporting a bounds error, and it applies
to a per-antenna override as well as to the array-wide default.

What ``Rc`` does and does not do
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* :math:`|r|` oscillates between :math:`1 - A` and :math:`1 + A` with frequency
  period :math:`1/\tau_\mathrm{cable}`.
* The delay-domain (frequency-Fourier) transform of a corrupted spectrum carries
  a secondary peak at **exactly** :math:`\tau_\mathrm{cable}` with relative
  amplitude :math:`A`. That peak is why ``Rc`` is a term of its own rather than a
  bandpass shape: a smooth bandpass is compact around zero delay, and a
  reflection is not.
* With one cable on both feeds of both antennas, every correlation is scaled by
  the real factor :math:`|r(\nu)|^2` and no phase moves — the exact opposite of
  ``Kd``'s common-mode behaviour.
* A reflection with ``cable_delay_s: 0.0`` is legal and is a constant complex
  offset rather than a ripple. It is reported as frequency-**independent**,
  because it is.


Kd — instrumental delay
-----------------------

``Kd`` is a per-antenna, per-feed delay offset in the signal chain. It is
direction-independent, frequency-dependent, diagonal, and unitary:

.. math::

   Kd_p(\nu) = \mathrm{diag}\bigl(e^{-2\pi i \nu \tau_{p0}},\,
                                  e^{-2\pi i \nu \tau_{p1}}\bigr)

The **negative** exponent is RadioSim's one delay-sign convention, the same one
the geometric phase :math:`e^{-2\pi i\, \vec b \cdot \vec s}` uses: a positive
delay produces :math:`e^{-i\,\text{positive}}` everywhere.

*Reference:* Thompson, Moran & Swenson (2017), 3rd ed., Chapter 7; CASA ``K``
Jones.

Configuration
~~~~~~~~~~~~~

.. code-block:: yaml

   jones:
     Kd:
       delay_s: 1.0e-9          # array-wide default, per feed
       per_antenna:             # same (antenna, feed) keying as G
         - antenna: 7
           feed: 0
           delay_s: 4.5e-9

``delay_s`` defaults to zero, so a block written with ``per_antenna`` alone means
"these feeds only". A block that resolves to zero everywhere is rejected as the
identity.

What ``Kd`` does and does not do
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* A delay **common to both feeds of every antenna** cancels exactly on every
  cross-correlation: the term enters as :math:`e^{-2\pi i\nu\tau_p}` on antenna
  :math:`p` and as its conjugate on antenna :math:`q`, leaving
  :math:`e^{-2\pi i\nu(\tau_p - \tau_q)} = 1`. If you want a whole-array delay to
  do something, it has to be *differential*.
* A differential delay is a pure baseline phase slope: every correlation
  amplitude is unchanged and the phase is exactly linear in frequency.
* ``Kd`` is antenna-based, so it leaves the **closure phase** invariant. A fringe
  slope looks like a baseline property and is not one.
* A delay shared by an antenna's two feeds is a *scalar* phase on that antenna,
  and RadioSim reports it as such.


X — cross-hand phase and delay
------------------------------

``X`` is the relative phase between an antenna's two feed paths, constant in
frequency or linear in it. It is direction-independent, diagonal, and unitary:

.. math::

   X_p(\nu) = \mathrm{diag}\bigl(1,\;
              e^{\,i(\phi_x + 2\pi\nu\tau_x)}\bigr)

Cross-hand phase and cross-hand delay are the same matrix — one
frequency-constant term and one frequency-linear one — so they are one term with
two parameters rather than two terms.

The first entry is exactly :math:`1`, not a second free parameter. Only the
*relative* phase between the two feeds is physical: any pair of feed phases
factorizes into a common phase, which ``G`` owns and which cancels on every
baseline, times a relative phase, which is this term. A second parameter here
would be exactly degenerate with ``G``.

*Reference:* CASA ``crosshand phase`` (``Xf``) and ``KCROSS`` conventions;
Sault, Hamaker & Bregman (1996), A&AS **117**, 149; Smirnov (2011), §6.

Configuration
~~~~~~~~~~~~~

.. code-block:: yaml

   jones:
     X:
       phase_rad: 0.1
       delay_s: 0.0
       per_antenna:             # keyed by antenna number ALONE
         - antenna: 3
           phase_rad: -0.4

``X`` is the one term whose ``per_antenna`` entries carry **no** ``feed`` key.
The parameter is the phase *between* the two feeds, so there is one number per
antenna and a feed index would have to name the feed the phase is not on. A
repeated antenna is rejected, as is an antenna the instrument does not have.

What ``X`` does and does not do
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* With linear receptors, a cross-hand phase :math:`\phi_x` leaves both parallel
  hands untouched and multiplies :math:`V_{xy}` by :math:`e^{-i\phi_x}` and
  :math:`V_{yx}` by :math:`e^{+i\phi_x}` — that is, it rotates Stokes :math:`U`
  into Stokes :math:`V` by exactly :math:`\phi_x`, the classic X-Y phase
  signature.
* Unlike a ``G`` phase, a cross-hand phase common to the whole array does **not**
  cancel: it is a phase on one feed only, so it survives on the cross hands.
* ``X`` is defined per feed *index* in the antenna's own basis. On a circular
  receptor the phased feed is ``L``, and the affected correlations are the
  :math:`(RL, LR)` pair rather than :math:`(xy, yx)`.
* ``X`` commutes with ``G``, ``B``, ``Kd`` and ``Rc``, all of which are diagonal
  in the same basis. It does **not** commute with ``D``.


D — polarization leakage
------------------------

``D`` is the first-order cross-coupling between an antenna's two feed chains. It
is direction-independent, optionally frequency-dependent, **non-diagonal** and
**non-unitary**:

.. math::

   D_p(\nu) = \begin{bmatrix}
       1 & d_{p0}(\nu) \\
       -d_{p1}(\nu)^{*} & 1
   \end{bmatrix}

:math:`d_{p0}` is the leakage of feed 1's signal into feed 0's chain and
:math:`d_{p1}` the converse. Both are dimensionless and complex;
:math:`|d| \sim 0.01`–:math:`0.05` is typical of a well built receiver. The
conjugate-and-negate on the lower left is the Hamaker, Bregman & Sault
convention, and it is what makes ``D`` reduce to a scaled rotation for real,
equal leakages.

*Reference:* Hamaker, Bregman & Sault (1996), A&AS **117**, 137, §4; Sault,
Hamaker & Bregman (1996), A&AS **117**, 149; Smirnov (2011), §6.4; Carozzi &
Woan (2011), IEEE TAP **59**, 2058 (IXR).

Configuration
~~~~~~~~~~~~~

.. code-block:: yaml

   jones:
     D:
       d_terms:                 # the array-wide model, naming BOTH feeds
         kind: explicit         # explicit | ixr | frequency_polynomial
         d0: [0.02, 0.0]        # [re, im]
         d1: [0.0, 0.02]
       per_antenna:
         - antenna: 3
           feed: 1
           d_term:              # a per-antenna override names ONE feed
             kind: ixr
             ixr_db: 30.0
             phase_rad: 0.0

The array-wide field is ``d_terms`` and the override field is ``d_term``. The
names differ by one letter because the shapes do: the array-wide block names both
feeds, while an override is keyed by a feed index and therefore names one. An
override that had to restate both feeds would make the index it is keyed by
meaningless.

The three kinds
~~~~~~~~~~~~~~~

``explicit``
    Complex ``d0`` and ``d1``, each written ``[re, im]`` or as a bare real
    number. Both default to zero, so a block may name one feed only.

``ixr``
    An intrinsic cross-polarization ratio in decibels, converted by

    .. math::

       |d| = \frac{1}{\sqrt{\mathrm{IXR}_\mathrm{lin}}},
       \qquad
       \mathrm{IXR}_\mathrm{lin} = 10^{\mathrm{IXR}_\mathrm{dB}/10}

    equivalently :math:`\mathrm{IXR}_\mathrm{dB} = -20\log_{10}|d|`. A *larger*
    IXR is a *smaller* leakage: 30 dB is about 3 per cent, 20 dB about 10 per
    cent. ``ixr_db`` must be positive — :math:`0` dB is a completely
    depolarizing receptor (:math:`|d| = 1`), and a negative value would mean a
    leakage larger than the direct path, which the first-order form does not
    describe. Use ``per_antenna`` to give the two feeds different phases.

``frequency_polynomial``
    :math:`d(\nu) = \sum_k c_k x^k` with
    :math:`x = (\nu - \nu_\mathrm{ref})/\nu_\mathrm{scale}`, written as
    ``coefficients0`` and ``coefficients1``. ``reference_frequency_hz`` and
    ``scale_frequency_hz`` default to the band centre and half-bandwidth, exactly
    as for ``B``.

What ``D`` does and does not do
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* :math:`D(0) = I_2` exactly, which is why a zero-leakage block is rejected.
* :math:`\det D = 1 + d_{p0} d_{p1}^{*}`, so ``D`` is invertible for every
  physical leakage — it is a calibratable corruption, not a loss of information.
* ``D`` is **not** unitary for any non-zero leakage. A receptor that moves power
  between its two chains while preserving :math:`J J^{H} = I` would be a
  rotation, not a leakage.
* For an **unpolarized** source the corrupted cross hand is exactly

  .. math::

     V_{01} = \tfrac{I}{2}\,\bigl(d_{p0} - d_{q1}\bigr)

  — note the second antenna contributes :math:`-d_{q1}`, not
  :math:`+d_{q1}^{*}`. This is the sharpest available check that a leakage model
  is right, and RadioSim's test suite asserts it at machine precision.
* ``D`` does **not** commute with a feed-asymmetric ``G``, ``B``, ``Kd`` or
  ``Rc``, nor with ``X``. The canonical chain puts all of those nearer the
  correlator than ``D``, and that order is observable.
* ``D`` is direction-independent by construction. A leakage that varied across
  the beam is *beam squint*, which belongs to the beam subsystem; modelling it
  here would create a second beam pathway.


Where the record goes
---------------------

An enabled Jones term is recorded everywhere the run is recorded:

* ``scientific_sha256`` changes. Two runs differing in any single Jones
  parameter have different fingerprints; two runs with ``jones:`` absent have
  the fingerprint they had before the section existed.
* HDF5 results gain a ``jones/`` group with the enabled terms, the composed
  chain order, each term's resolved parameters, the resolved mount types, and
  ``jones_sha256``. The group is written only when a term was enabled.
* The summary JSON gains a bounded ``jones`` block. A run that enabled nothing
  reports empty lists and a ``null`` digest rather than omitting the block, so
  "no terms" is distinguishable from "an older summary".
* Measurement Set and UVFITS output is **unchanged**. A corrupted visibility is
  still a visibility; RadioSim does not write calibration tables.
* Observability plots are unchanged. They evaluate beams, not chains.
