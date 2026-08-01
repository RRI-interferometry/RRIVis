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

Nine further terms — ``D``, ``P``, ``Z``, ``T``, ``X``, ``Kd``, ``Rc``, ``M``
and ``Q`` — exist as named classes with documented physics and **no
implementation**: constructing one is allowed, evaluating one raises. They have
no configuration block, deliberately. A schema field for a term that cannot be
honoured would accept a value and discard it, which is worse than refusing.


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
