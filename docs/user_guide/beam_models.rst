Beam Models
===========

``beams`` is a strict discriminated union. Each document selects one complete
mode; unknown fields and incomplete mode shapes are rejected.

Illumination is not the receptor
--------------------------------

``beams`` owns aperture **illumination**: how a reflector is fed with power, in
the vocabulary ``illumination``, ``taper``, and edge angle. The receiving
**receptor** — which orthogonal feed pair each antenna has and which
polarization basis results are reported in — is owned by the separate
``receptors`` section documented in :doc:`configuration`. The two vocabularies
are deliberately disjoint, in configuration and in source identifiers, so that
"feed" never means two things at once. Choosing ``circular`` receptors does not
change any beam model, and choosing an illumination model does not change any
correlation label.

Runtime boundary
----------------

The high-level ``Simulator`` activates all four modes below. Source resolution
first creates immutable definitions, instrument resolution supplies canonical
antenna identities, and setup then resolves complete assignments and atomically
loads one canonical per-antenna ``BeamSystem``. Beam assignment, file,
metadata, frequency-domain,
and sampling-characterization failures occur before device, backend, network,
or sky work. There is no analytic fallback for a FITS declaration.

Analytic mode
-------------

The runnable direct-circular form is:

.. code-block:: yaml

   beams:
     mode: analytic
     model:
       kind: circular_aperture
       taper:
         kind: gaussian
         edge_taper_db: 10.0

Direct circular tapers are ``uniform``, ``gaussian``, ``parabolic``,
``parabolic_squared``, and ``cosine``. Gaussian, parabolic, and
parabolic-squared tapers accept a finite nonnegative ``edge_taper_db``.
Antenna diameters come from canonical instrument resolution; there is no beam
diameter field or hidden diameter fallback.

The other active analytic variants are:

.. code-block:: yaml

   # Rectangular aperture
   beams:
     mode: analytic
     model:
       kind: rectangular_aperture
       north_length_m: 14.0
       east_length_m: 12.0

   # Elliptical aperture
   beams:
     mode: analytic
     model:
       kind: elliptical_aperture
       north_diameter_m: 14.0
       east_diameter_m: 12.0

   # Analytically derived illumination
   beams:
     mode: analytic
     model:
       kind: analytical_illumination
       illumination:
         kind: corrugated_horn
         focal_ratio: 0.4
         q: 1.15
       taper_profile:
         kind: gaussian
       reflector:
         kind: prime_focus

   # Numerically integrated illumination
   beams:
     mode: analytic
     model:
       kind: numerical_illumination
       illumination:
         kind: open_waveguide
         focal_ratio: 0.4
         b_over_lambda: 0.7
       reflector:
         kind: cassegrain
         magnification: 2.0

Analytical illumination supports ``corrugated_horn`` (``focal_ratio``, ``q``),
``open_waveguide`` (``focal_ratio``, ``b_over_lambda``), and
``dipole_ground_plane`` (``focal_ratio``, ``height_wavelengths``). Its derived
taper profile is ``gaussian``, ``parabolic``, or ``parabolic_squared``.
Reflectors are ``prime_focus`` or ``cassegrain``; Cassegrain magnification must
be greater than one. Numerical illumination uses a fixed 256-point radial
resolution, not a user-authored tuning field.

FITS and assignment modes
-------------------------

A FITS source has ``kind: fits``, ``path``, ``normalization: peak`` or
``normalization: uvbeam_peak_common_v1``,
``angular_interpolation: bilinear``, and ``frequency_interpolation: cubic`` or
``linear``. The first three option values shown are defaults where applicable.
That one ``normalization`` field selects which of **two** accepted subsets of
the same file RadioSim reads, and nothing else in the document does.

The default ``peak`` subset is deliberately scalar. RadioSim accepts finite
``efield`` or ``simple`` data on a regular full-visible-hemisphere ``az_za``
grid, a fixed antenna mount, east-oriented linear X/Y feeds, a finite identity
basis transform, unit bandpass, peak normalization, and a strictly increasing
frequency axis. The evaluated voltage is the scalar complex response on the
diagonal of a 2x2 E-Jones matrix. Power beams, circular feeds in the *file*,
non-identity bases, other coordinate systems or mounts, and arbitrary
cross-polarization are rejected. Angular interpolation is bilinear; frequency
interpolation is exactly linear or cubic with no extrapolation or method
fallback.

Rejecting circular feeds in a ``peak`` BeamFITS file does not restrict the
receptor model. ``receptors`` supplies the receptor basis and any static feed
rotation independently of the beam, and the scalar E-Jones response multiplies
both bases identically. Polarization leakage is the ``D`` term, configured
under ``jones`` and documented in :doc:`jones_terms`; a beam that genuinely
differs between the two feeds is a *non-scalar* E-Jones, which is the second
accepted subset below (:ref:`stage3-full-efield`). Everything that remains out
of scope is dispositioned in ``docs/development/beam_physics_scope.md``.

.. code-block:: yaml

   # One shared source
   beams:
     mode: shared_fits
     beam:
       kind: fits
       path: beams/shared.beamfits
       normalization: peak
       angular_interpolation: bilinear
       frequency_interpolation: cubic

   # Ordered per-antenna FITS assignments
   beams:
     mode: per_antenna_fits
     assignments:
       - antenna: {kind: number, number: 0}
         beam: {kind: fits, path: beams/antenna-0.beamfits}

   # Ordered analytic/FITS choices with one shared analytic definition
   beams:
     mode: mixed
     analytic_model:
       kind: circular_aperture
       taper: {kind: uniform}
     assignments:
       - antenna: {kind: name, name: ANT0}
         beam: {kind: analytic}
       - antenna: {kind: number, number: 1}
         beam: {kind: fits, path: beams/antenna-1.beamfits}

Assignments are ordered and nonempty. Antenna references are tagged by
``kind: number`` or ``kind: name``. Configuration resolution preserves those
references without reading FITS content. ``Simulator.setup`` resolves every
reference against the canonical instrument, requires complete coverage, and
loads the resulting handlers atomically.

Path and provenance rules
-------------------------

YAML-relative FITS paths use the YAML file's parent. Mapping, typed-model, and
parameter construction require ``base_dir`` for relative FITS paths. ``~`` is
expanded, environment-variable syntax is rejected, and every checked FITS path
must exist, be a readable regular file, and is normalized through symlinks.
``check_input_paths=False`` skips existence/type/readability checks but still
normalizes the path. Each source records its indexed logical path, such as
``beams.assignments[2].beam.path``, in configuration provenance.

Source resolution constructs immutable definitions with deterministic
fingerprints from the complete normalized analytic model or FITS source
options. Path validation does not read FITS content. During setup, canonical
assignment and loading validate the complete scientific subset and publish the
immutable loaded state only after every handler succeeds. Point visibility,
HEALPix visibility, sampling advice, observability, and result provenance all
consume this same ``BeamSystem`` and its detached state.

Pointing offsets and surface errors
-----------------------------------

Two optional ``beams`` blocks describe the mount and the dish rather than the
beam model, so both are accepted in all four modes and both are per-antenna.
Their absence adds no optional pointing or surface-error effect or snapshot, so
otherwise equivalent current runs have the same cube, beam fingerprints, and
``scientific_sha256``.

``beams.pointing`` is a **deterministic mount mispointing**. The two angles are
a fixed rotation of that antenna's beam frame relative to the topocentric
horizontal frame, composed as the two encoder errors of an alt-az mount:
``azimuth_offset_deg`` rotates about the local vertical (North through East),
then ``elevation_offset_deg`` tilts the boresight away from the zenith. Because
RadioSim's boresight *is* the zenith, the mispointed boresight lands at
topocentric azimuth ``azimuth_offset_deg`` and zenith angle
``elevation_offset_deg``, and the beam is evaluated at the direction expressed
in that rotated frame.

Two consequences are exact rather than small-angle:

- the beam's peak moves by a great-circle angle of exactly
  ``elevation_offset_deg``;
- a pure azimuth offset rotates the pattern about the boresight without moving
  it. That is the alt-az keyhole degeneracy, and it is real physics at a
  zenith-pointed mount, not an approximation: a pure azimuth offset is
  therefore inert for a circular aperture and is *not* inert for the
  rectangular and elliptical ones.

The horizon gate is unchanged and stays on the true topocentric altitude — a
rotation of the beam frame does not move the ground.

``beams.surface_error`` is the **Ruze random-surface RMS**, in metres. Ruze
(1966) gives the reflector's *power* efficiency,

.. math::

   \eta_s(\lambda) = \exp\!\left[-\left(\frac{4 \pi \sigma}{\lambda}\right)^{2}\right],

and RadioSim's ``E`` is a voltage beam, so the factor applied to the beam is
:math:`\sqrt{\eta_s}`. The visibility amplitude on a baseline of two antennas
sharing the same :math:`\sigma` is then scaled by exactly :math:`\eta_s`, which
is what the published equation states. Both closed forms are public as
``radiosim.core.beam.runtime.ruze_power_efficiency`` and ``ruze_voltage_factor``.

Neither effect changes beam loading or deduplication: two mispointed antennas of
the same diameter and model still share one loaded handler, because both effects
are applied around the evaluator rather than inside it. They do enter the
per-antenna ``assignment_fingerprint`` and therefore the beam state fingerprint
and ``scientific_sha256``.

What RadioSim does **not** model, and who owns it, is written out item by item
in ``docs/development/beam_physics_scope.md``: full cross-polarization and the
Ruze error-beam decomposition. SCI-005 Stage 2 removed beam squint from that
list; see `Beam squint`_ below.

.. _stage1-aperture-physics:

Aperture blockage and deterministic surface error
-------------------------------------------------

``beams.aperture_physics`` is the optional array-wide block that turns the
scalar analytic beam from a closed-form far field into **one normalized
aperture transform**. For unmodified pupil :math:`\mathcal P_0`, radial
illumination :math:`A(\mathbf u)`, obstruction mask :math:`M(\mathbf u)` and
deterministic surface height :math:`h(\mathbf u)`, the voltage response is

.. math::

   e(\mathbf q,\lambda)=\frac{1}{N_0}
   \int_{\mathcal P_0} A(\mathbf u)\,M(\mathbf u)\,
   \exp\!\left[-i\frac{4\pi}{\lambda}h(\mathbf u)\right]
   \exp(-i\mathbf q\cdot\mathbf u)\,d^{2}u,
   \qquad
   N_0=\int_{\mathcal P_0} A(\mathbf u)\,d^{2}u.

Three properties of that single integral matter in practice:

- :math:`N_0` is **always** the unmodified ideal-aperture integral. It is not
  recomputed after masking and the modified beam is never re-peak-normalized,
  so blockage and aberration loss appear exactly once in ``E`` — the boresight
  response of a blocked uniform aperture really is :math:`1-\epsilon^{2}`, not
  one.
- the mask and the phase are applied *inside* the integral. Two separately
  evaluated far-field patterns must never be multiplied together, because the
  Fourier transform does not distribute over aperture multiplication.
- the aperture axes are ``(north, east)`` and aperture azimuth
  :math:`\varphi = 0` points north and increases through east, matching
  RadioSim's topocentric azimuth.

.. code-block:: yaml

   beams:
     mode: analytic
     model:
       kind: circular_aperture
       taper: {kind: uniform}
     aperture_physics:
       normalization: unmodified_ideal_aperture_v1
       blockage:
         central_diameter_ratio: 0.15     # 0 < epsilon < 1
         support_legs:
           - {position_angle_deg: 0.0, width_m: 0.3}
           - {position_angle_deg: 180.0, width_m: 0.3}
       zernike_surface:
         convention: radiosim.real_unit_rms_disk_surface_height.v1
         modes:
           - {n: 2, m: 0, surface_height_coefficient_m: 0.0005}

**Supported pupils.** The transform needs one exact compact aperture-plane
profile, which only ``circular_aperture`` with a ``uniform``, ``parabolic`` or
``parabolic_squared`` taper, and ``analytical_illumination`` with a
``parabolic`` or ``parabolic_squared`` taper profile, provide. The direct and
derived Gaussian shortcut does not uniquely specify a compact disk pupil, the
direct cosine shortcut declares no radial-pupil inverse, and the numerical
illumination response is a fixed 256-node trapezoidal Hankel rule rather than
the continuum transform of a retained pupil. Those, along with the rectangular
and elliptical families and every BeamFITS source, are rejected with a typed
``UnsupportedConfigError``. A FITS file already contains its own aperture
physics; applying this block on top would double count it.

**Blockage geometry.** A support leg is the closed radial strip of physical
width ``width_m``, running from the edge of the central shadow to the ideal
pupil edge and centred on ``position_angle_deg`` measured North through East.
It is one *outward half-strip*, so a structure crossing the whole dish is
authored as two records 180 degrees apart. Masks combine by set union, so an
overlap is removed once rather than twice. For uniform illumination, no legs,
and :math:`x = \pi D \sin\theta / \lambda`, the response reduces to the
published blocked-aperture form

.. math::

   e_\epsilon(x)=\frac{2\left[J_1(x)-\epsilon J_1(\epsilon x)\right]}{x},
   \qquad e_\epsilon(0)=1-\epsilon^{2},
   \qquad \eta_b=(1-\epsilon^{2})^{2},

(`NASA TM X-63186 <https://ntrs.nasa.gov/citations/19680013447>`_;
`ITU-R SA.2401-0 <https://www.itu.int/pub/R-REP-SA.2401-2017>`_). For a tapered
illumination the boresight loss is illumination-weighted and is **not**
:math:`1-\epsilon^{2}`. A leg wider than an antenna's resolved aperture
diameter is rejected at beam-assignment resolution with
``InvalidBeamGeometryError``, because per-antenna diameters exist only after
the instrument resolves.

**Surface height.** ``zernike_surface`` takes exactly ``convention`` and a
non-empty ``modes`` sequence, and every mode has exactly the three keys ``n``,
``m`` and ``surface_height_coefficient_m``. The basis is R. J. Noll's real
unit-RMS *disk* basis (`JOSA 66, 207 (1976)
<https://opg.optica.org/josa/abstract.cfm?uri=josa-66-3-207>`_,
DOI 10.1364/JOSA.66.000207), normalized so that
:math:`(1/\pi)\int Z_n^m Z_{n'}^{m'}\rho\,d\rho\,d\varphi = \delta`. Validation
requires :math:`0 \le n \le 32`, :math:`|m| \le n` and :math:`n-|m|` even;
duplicate ``(n, m)`` pairs are rejected, and piston ``(0, 0)`` and tip/tilt
``(1, \pm 1)`` are rejected because instrumental delay and deterministic
pointing already own those effects. No Noll or OSA single index is accepted.

Each coefficient is signed **aperture-equivalent** reflector surface-height
error in metres, that is one half of the reflected optical-path difference. A
physical normal displacement :math:`\delta_n` at incidence angle :math:`i` maps
to :math:`h = \delta_n\cos i`; RadioSim never invents :math:`i` from the beam
model, so :math:`h=\delta_n` only at normal incidence. Because the signed
excess path is exactly :math:`2h`, the positive-delay convention gives the
phase factor :math:`\exp(-i\,4\pi h/\lambda)`.

After a blockage mask is applied these ordinary disk functions are no longer
orthogonal over the transmitting annulus, so the quadrature sum of coefficients
is *not* the RMS over that annulus — the annular basis is a different one
(V. N. Mahajan, `JOSA 71, 75 (1981) <https://doi.org/10.1364/JOSA.71.000075>`_).

**Numerics.** The integral is evaluated by a boundary-fitted polar
Gauss-Legendre rule whose radial panels split at the central-shadow radius,
every leg saturation radius, every support-topology radius and every periodic
cut, and whose per-panel orders are seeded from both the far-field bandwidth
and the surface-phase bandwidth. Every tolerance, order and resource cap is
fixed by the design and cannot be authored in YAML. Extended-precision beams
keep their width: nodes, weights and accumulation never pass through float64
when the resolved dtype is wider.

Ruze scattered-power diagnostic
-------------------------------

``beams.surface_error`` keeps its accepted coherent-voltage meaning exactly.
On top of it, one antenna may declare an optional *ensemble-power* diagnostic:

.. code-block:: yaml

     surface_error:
       default:
         rms_surface_error_m: 0.001
         error_beam_diagnostic:
           kind: gaussian_covariance_power
           correlation_length_m: 0.25

The literal ``gaussian_covariance_power`` names a complete field law, not just a
radial function: a real, zero-mean, jointly Gaussian, second-order stationary
aperture-equivalent surface-error field with
:math:`\rho_h(\Delta)=\exp[-(|\Delta|/L)^{2}]`. It is that law's
*characteristic function*, not its covariance alone, that licenses the
mutual-coherence kernel :math:`\exp\{-s^{2}[1-\rho_h]\}`; a
covariance-matched non-Gaussian field gives a different kernel.
``rms_surface_error_m`` is the pointwise standard deviation of the random
residual left *after* the configured deterministic Zernike map, never a value
inferred from those coefficients.

Read the record with
:meth:`~radiosim.core.beam.runtime.BeamSystem.evaluate_ruze_power_diagnostic`.
It reports coherent-main power, total ensemble power, and their non-negative
scattered difference at every requested direction, together with a convergence
record whose Poisson tail, separation-truncation bound, separation residuals,
paired-pupil residuals and imaginary residual are all retained separately.

Three properties are worth stating plainly:

- it is **not a Jones voltage**. ``sqrt(B_main + B_error)`` would invent a phase
  and perfectly correlated structure, so no complex field is derived from it, it
  never enters a cross-correlation Jones matrix, and requesting it neither calls
  nor changes ``evaluate_jones``. Configuring it *does* change the scientific
  fingerprint; evaluating it repeatedly changes nothing;
- the whole algorithm is host-side, so the method takes no backend argument; and
- version 1 requires an **unobstructed** pupil. With a blockage authored, the
  paired region is the intersection of two shifted copies of the support mask,
  which needs a second boundary and topology-root family this version does not
  freeze, so the diagnostic is refused for that antenna with issue code
  ``beam.ruze_power_diagnostic.unsupported_obstruction``.

The evaluation works in the *separation* variable. Writing
:math:`f = A M e^{-i\phi_{\rm det}}` and
:math:`C(\boldsymbol\Delta)=\int f(\mathbf r) f^{*}(\mathbf r-\boldsymbol\Delta)\,d^{2}r`,
each Poisson mixture term is

.. math::

   P_m(\mathbf q)=\frac{1}{|N_0|^{2}}\int_{\mathbb R^{2}}
   C(\boldsymbol\Delta)\,e^{-i\mathbf q\cdot\boldsymbol\Delta}\,
   e^{-|\boldsymbol\Delta|^{2}/\ell_m^{2}}\,d^{2}\Delta,
   \qquad \ell_m=L/\sqrt m,

which is the same integral as the shifted-wavevector form, evaluated in a
variable where it costs :math:`O(1)` in :math:`D/L`: :math:`C` carries no
far-field oscillation, the Gaussian confines the separation to a few
:math:`\ell_m`, and one :math:`C` array serves every retained order and every
requested direction (Ruze 1952, DOI 10.1007/BF02903409; Ruze 1966,
DOI 10.1109/PROC.1966.4784).

An absent ``aperture_physics`` block changes nothing at all — the same resolved
configuration, fingerprints, result bytes and logs as before.

.. _stage2-beam-squint:

Beam squint
-----------

``beams.squint`` turns the scalar analytic beam into **two native-feed
samples of the same scalar pattern, oppositely displaced**, then composes
them back into the receptor's ``E`` factor. It follows the exact Cotton/Uson
law rather than the small-angle approximation often quoted for it (J. M.
Uson and W. D. Cotton, `Beam squint and Stokes V with off-axis feeds
<https://arxiv.org/abs/0807.0026>`_, 2008). :doc:`configuration` gives the
five authored fields, their domains, and every rejection; this section gives
the physics.

.. code-block:: yaml

   beams:
     mode: analytic
     model:
       kind: circular_aperture
       taper: {kind: uniform}
     squint:
       default:
         convention: cotton_uson_exact_v1
         reference_frequency_hz: 1.5e8
         per_feed_offset_deg_at_reference: 2.0
         mechanical_feed_position_angle_deg: 35.0
         positive_native_feed: x

**Frequency law.** ``per_feed_offset_deg_at_reference`` is the displacement
:math:`\delta_{\rm ref}` of *one* hand at ``reference_frequency_hz``
:math:`\nu_{\rm ref}`, so the nominal pointing is the midpoint of the two
feeds and their total feed-to-feed separation is :math:`2\delta(\nu)`. The
exact scaling is

.. math::

   \delta(\nu) = \sin^{-1}\!\left[\frac{\nu_{\rm ref}}{\nu}\sin\delta_{\rm ref}\right],

evaluated as one binary64 expression at both beam-system load (a preflight
over every observation channel) and at evaluation time, so an argument the
preflight accepted cannot leave :math:`[-1, 1]` later. RadioSim rejects an
out-of-domain channel; it never clips a displacement, because a clipped value
would silently report a different telescope. The approximation
:math:`\delta \propto 1/\nu` is only this law's small-angle limit and is
never the production computation.

**Direction convention.** The squint direction is orthogonal to the
optical-axis/feed plane, not along the physical feed-location ray. Writing
the feed-ray unit vector as
:math:`\mathbf u_{\rm feed}(\beta) = \cos\beta\,\hat{\mathbf N} + \sin\beta\,\hat{\mathbf E}`,
the v1 handedness fixes the squint direction as that ray rotated a further
:math:`+\pi/2`:

.. math::

   \mathbf u_{{\rm squint},+}(\beta) =
   -\sin\beta\,\hat{\mathbf N} + \cos\beta\,\hat{\mathbf E} =
   \mathbf u_{\rm feed}(\beta + \pi/2).

The ``positive_native_feed`` label is evaluated at :math:`+\delta(\nu)` along
this direction and its basis partner (``x``/``y`` or ``r``/``l``, fixed by
the label pair) at :math:`-\delta(\nu)`; swapping the positive feed reverses
the sign of the Stokes-V leakage this produces. Both evaluations are exact
great-circle (Rodrigues) rotations of the already pointing-transformed beam
direction about the horizontal axis
:math:`\hat{\mathbf a}_p = \sin\beta_{{\rm squint},p}\,\hat{\mathbf N}
- \cos\beta_{{\rm squint},p}\,\hat{\mathbf E}`, so that a rotation of
:math:`+\delta` about :math:`\hat{\mathbf a}_p` moves the beam-frame zenith
along :math:`+\mathbf u_{{\rm squint},+}`; the horizon gate stays on the true
topocentric altitude exactly as it does for a mispointed boresight.

**Mount field rotation.** ``mechanical_feed_position_angle_deg`` describes
the physical off-axis feed location in the antenna beam frame, North through
East — it is not ``receptors.*.feed_rotation_deg``, which is the electrical
receptor-axis convention used to build ``C`` and cannot rotate a physical
feed displacement. On a rotating mount the feed ray follows the antenna's
resolved boresight exactly as :math:`P`'s field rotation does:

.. math::

   \beta_{{\rm feed},p} = \operatorname{wrap}\!\left(
   \beta_{\rm mechanical} + \eta_p\,\psi_p + \nu_p\,\mathrm{alt}_p\right),
   \qquad
   \beta_{{\rm squint},p} = \operatorname{wrap}\!\left(
   \beta_{{\rm feed},p} + \pi/2\right),

where :math:`\psi_p` and :math:`\mathrm{alt}_p` are the parallactic angle and
true altitude of the antenna's resolved boresight (the beam-frame zenith
mapped through any configured pointing offset, or the topocentric zenith
otherwise) and :math:`(\eta_p, \nu_p)` are the same accepted mount factors
``P`` uses: :math:`(1, 0)` for ``alt-az``, :math:`(0, 0)` for
``equatorial``/``fixed``, :math:`(1, +1)` for Nasmyth-right, and
:math:`(1, -1)` for Nasmyth-left. For :math:`\eta_p = 0` the parallactic
angle is taken as exactly ``0.0``, which the formula multiplies away; for
:math:`\eta_p \ne 0` at a boresight exactly at zenith the parallactic angle
is undefined, and RadioSim raises rather than substituting
:math:`\operatorname{atan2}(0, 0)` — a non-rotating mount, or a rotating one
with a nonzero pointing offset, avoids this.

**Factorization.** Writing the two displaced native-feed voltage samples as
:math:`D_b = \operatorname{diag}(b_0, b_1)`, the physical local response is
:math:`D_b C P`. RadioSim's chain stays fixed at ``C E P``
(:doc:`jones_matrices`), so squint is folded into ``E`` rather than into a
new chain position:

.. math::

   E = C^\dagger D_b\,C, \qquad C E = D_b C.

``E`` is **generally full** in RadioSim's sky-side space once squint is
enabled — the old scalar-only ``E`` is retired for a squint-carrying antenna
— including for a rotated linear receptor. The one exception is exact: for
*any* circular receptor :math:`C = S\,R(\chi)`, and any :math:`b_0 \ne b_1`,

.. math::

   E = C^\dagger \operatorname{diag}(b_0, b_1)\, C =
   \frac{b_0 + b_1}{2}\, I_2 - \frac{b_0 - b_1}{2}\, \sigma_y,

independent of :math:`\chi`, whose diagonal entries are exactly equal and
which commutes exactly with every real rotation because
:math:`R(\theta) = \exp(i\theta\sigma_y)`. A circular receptor's squint is
therefore still physically present — :math:`|E_{01}| = |b_0 - b_1|/2 > 0` —
but it is unable to make ``C E P`` and ``C P E`` differ; only a rotated
*linear* receptor makes the chain order observable.

**Precision and scope.** Displacement geometry (the feed ray, the squint
rotation, and the boresight computation) is binary64 throughout, matching
``DirectionBatch`` and the accepted pointing rotation. The two feed samples,
the receptor matrix ``C``, and the ``E`` composition are evaluated at the
resolved beam dtype and never narrow through ``complex128`` when that dtype
is wider. ``beams.squint`` is accepted only on the analytic beams mode: a
measured BeamFITS pattern may already contain the physical feed displacement,
and the accepted scalar subset carries no metadata by which RadioSim could
prove it does not, so every ``shared_fits``, ``per_antenna_fits``, and
``mixed`` document that also carries a squint block is rejected. An antenna
without squint keeps today's byte-identical response, call surface, and
result.

.. _stage3-full-efield:

Full-efield BeamFITS
--------------------

``normalization: uvbeam_peak_common_v1`` reads the **complete complex**
``data_array`` of the same ``beam_type: efield`` file the scalar ``peak``
subset reads only the diagonal of. :doc:`configuration` gives the authored
field, the strict file contract, and every rejection; this section gives the
physics.

.. code-block:: yaml

   beams:
     mode: shared_fits
     beam:
       kind: fits
       path: beams/shared.beamfits
       normalization: uvbeam_peak_common_v1

**Two subsets, not a widening.** The literal names an accepted *interpretation*
of the committed bytes, not an operation applied to them: RadioSim renormalizes
nothing under either value. A file the scalar subset accepts is generally
rejected by the full-efield subset — its zenith row does not satisfy the
de-spin predicate below once the two vector components are read — and a
full-efield file is generally rejected by the scalar subset. The two literals
therefore name two readings of one file format rather than a repair of one by
the other.

**Basis conversion.** pyuvdata stores an ``az_za`` E-field beam as two vector
components per feed, azimuth first and zenith angle second. RadioSim converts
them into the **chain's own** sky tangent pair with the fixed real orthogonal
*constant*

.. math::

   M = \begin{pmatrix}0 & 1\\ -1 & 0\end{pmatrix},
   \qquad \det M = +1,
   \qquad
   J_{\rm native}[f, c] = \sum_a \mathrm{data}[a, f]\, M[a, c],

so that :math:`J_{\rm native}[f, 0] = -E_\theta` and
:math:`J_{\rm native}[f, 1] = +E_{\mathrm{az_{uv}}}`. The target basis is
fixed by the ``P`` term rather than chosen here. ``E`` sits between ``C`` and
``P``, so its columns must be whatever tangent basis ``P`` delivers the sky
coherency into, and that is the **mixed-sign** pair
:math:`(-\hat{\mathbf e}_\theta, +\hat{\mathbf e}_{\mathrm{az_{uv}}})`:
:math:`\hat{\mathbf e}_\theta = -(\cos\psi\,\hat{\mathbf N} +
\sin\psi\,\hat{\mathbf E})` while
:math:`\hat{\mathbf e}_{\mathrm{az_{uv}}} = -\sin\psi\,\hat{\mathbf N} +
\cos\psi\,\hat{\mathbf E}` is unnegated. The mixed sign is structurally
necessary: :math:`\hat{\mathbf N}\times\hat{\mathbf E} = -\hat{\mathbf r}`
while :math:`\hat{\mathbf e}_\theta \times
\hat{\mathbf e}_{\mathrm{az_{uv}}} = +\hat{\mathbf r}`, so the two frames
carry opposite handedness and a proper rotation cannot deliver a common-sign
copy of one from the other. :math:`M` is constant, orthogonal, **antisymmetric**
with :math:`M^{\mathsf T} = -M`, and proper, so the conversion preserves total
field power exactly, introduces no reflection into the chain, and leaves every
complex phase in the data; direction geometry is binary64 throughout. The
accepted direction mapping ``az_uv = (pi/2 - az_radiosim) mod 2*pi`` is
unchanged and still fixes only *where* the pattern is sampled.

Ludwig's third definition (A. C. Ludwig, *The definition of cross
polarization*, IEEE Trans. Antennas Propag. **21**, 116, 1973, DOI
10.1109/TAP.1973.1140406) remains RadioSim's language for **diagnostics and
oracles**, and only there. With :math:`\varphi` measured from North through
East, the chain-basis to Ludwig-3 map is the *proper* rotation

.. math::

   S(\varphi) =
   \begin{pmatrix}-\cos\varphi & -\sin\varphi\\
   \ \ \sin\varphi & -\cos\varphi\end{pmatrix},
   \qquad \det S = +1,
   \qquad J_{\rm chain}\,S(\varphi) = J_{\rm L3},

so a co/cross oracle is mapped into the chain basis by :math:`S(\varphi)`
before it is compared with production.

**The zenith.** Coordinate azimuth is singular at the pole, and the chain
tangent pair *spins* with it while the physical response is one fixed map.
Requiring the converted ``za = 0`` row to be equal across azimuth would
therefore reject a perfectly valid file. What is single valued is the de-spun
matrix:

.. math::

   J(\mathrm{az_{uv}}) = J(\mathrm{az_{ref}})\,
   R(\mathrm{az_{uv}} - \mathrm{az_{ref}}),
   \qquad
   R(x) = \begin{pmatrix}\ \ \cos x & \sin x\\
   -\sin x & \cos x\end{pmatrix},

equivalently that :math:`J(\mathrm{az_{uv}})\,R(\mathrm{az_{uv}})^{\mathsf T}`
is constant across the row. A file whose de-spun zenith row is not constant
declares a physical matrix that depends on an arbitrary coordinate and is
rejected. At ``az_radiosim = 0`` the chain pair is exactly
:math:`-(\hat{\mathbf N}, \hat{\mathbf E})` — a genuinely common sign at
that one point — so the North/East tangent limit survives there.

**Azimuth-wrap continuity.** The azimuth axis is endpoint-excluded and closes
:math:`2\pi`, and the seam predicate is a **second** difference compared
against the interior maximum, not a first difference against one adjacent
sample. Only the second difference is scale-consistent: for a
twice-differentiable periodic row sampled at step :math:`h`, every
:math:`\Delta^2_k` equals :math:`h^2 J''(\xi_k)`, so seam and interior second
differences are the same order at every sampling density. The predicate is
:math:`|\Delta^2_0|_{\max} \leq 8\max_{k\ \rm interior}|\Delta^2_k|_{\max} +
(\mathrm{atol} + \mathrm{rtol}\cdot\mathrm{scale})`, evaluated per intrinsic
frequency and per zenith-angle row. It is deliberately not claimed to detect a
seam jump smaller than the row's own local curvature scale: on a coarse grid
such a jump is genuinely indistinguishable from smooth variation.

The stored ``basis_vector_array`` is **validated, never composed**: pyuvdata
3.2.1 builds the array it returns from ``numpy.ones`` and ``numpy.zeros`` and
discards the stored one, so RadioSim requires the committed array to be
exactly the native identity and then verifies that the returned array is that
identity too.

**Factorization.** :math:`J_{\rm native}` maps the incident tangent field to
the file's own native feed voltages. RadioSim's chain stays fixed at ``C E P``
(:doc:`jones_matrices`), so the file's response is folded into ``E`` using the
antenna's own resolved receptor matrix :math:`C`:

.. math::

   E = C^{\dagger} J_{\rm native}, \qquad C\,E = J_{\rm native}.

That :math:`C` is read from the same resolved receptor set the chain's own
``C`` term is built from, so the two cannot disagree — which is why the file's
feed pair, feed angles and derived x-orientation must match **every** antenna
it is assigned to. ``E`` is generally full for both receptor bases and for a
rotated linear receptor, so ``C E P`` and ``C P E`` genuinely differ.

**Cross-polar diagnostic.** The intrinsic cross-polarization ratio of the
accepted :math:`J_{\rm native}` follows Carozzi and Woan's polarimetric
definition (T. D. Carozzi and G. Woan, *A generalized measure of the
intrinsic cross-polarization ratio*, IEEE Trans. Antennas Propag. **59**,
2058, 2011, DOI 10.1109/TAP.2011.2143664): from the singular values
:math:`\sigma_{\max} \ge \sigma_{\min}` and
:math:`\kappa = \sigma_{\max}/\sigma_{\min}`,

.. math::

   \mathrm{IXR} = \left(\frac{\kappa + 1}{\kappa - 1}\right)^{2},
   \qquad
   \mathrm{IXR}_{\rm dB} = 10\log_{10}\mathrm{IXR}.

It is a **diagnostic and never a configuration field, a public method, or a
result record**: an exactly unitary-scaled matrix has infinite IXR and a
degenerate one has none, so RadioSim classifies the matrix by a fixed relative
tolerance and reports the derived numbers only where they are finite and well
defined, rather than writing an infinity or a ``NaN`` into any output.

**Scope and exclusions.** ``beams.squint`` (:ref:`stage2-beam-squint`) and
``beams.aperture_physics`` (:ref:`stage1-aperture-physics`) are accepted only
on the ``analytic`` beams mode, so neither can be combined with either
BeamFITS subset. That exclusion is deliberate for squint in particular: a
measured pattern may already contain the physical feed displacement, and no
BeamFITS metadata lets RadioSim prove it does not, so applying the analytic
Cotton/Uson displacement to a measured file would double-count it (J. M. Uson
and W. D. Cotton, `Beam squint and Stokes V with off-axis feeds
<https://arxiv.org/abs/0807.0026>`_, 2008). The full-efield subset is one
antenna's element response: it is not a station, array-factor, or
mutual-coupling model, and it says nothing about near-field or Fresnel-regime
behaviour. An antenna reading a ``peak`` file keeps today's byte-identical
scalar response, fingerprints, and result.

HEALPix sampling advice
-----------------------

Each loaded analytic handler stores its conservative voltage feature scale at
every exact observation frequency. For a circular or illumination aperture the
scale is :math:`\lambda / D`; rectangular and elliptical models use their
largest effective dimension.

An accepted azimuth/zenith-angle BeamFITS handler instead stores twice the
smallest validated native-grid angular spacing. This
``native_grid_representation_bound`` describes the sampled/interpolated
representation that RadioSim can evaluate. It is not a measured FWHM, a
physical beam bandwidth, or proof that the source beam was adequately sampled.

For every selected canonical baseline :math:`(p,q)` and exact observation
frequency :math:`\nu`, RadioSim forms the voltage-product feature scale

.. math::

   s_{pq}(\nu) =
   \left(s_p(\nu)^{-1} + s_q(\nu)^{-1}\right)^{-1}.

Only baselines retained by Tier 2 selection participate. The global minimum
therefore accounts for analytic aperture differences, different or shared FITS
handlers, and mixed analytic/FITS products. An autocorrelation uses the same
formula and naturally yields :math:`s_p/2`; an auto-only selection evaluates
every selected auto. Stable selected-baseline order followed by exact frequency
order breaks equal-scale ties.

The allowed HEALPix pixel scale is the minimum product scale divided by the
fixed engineering safety factor five. The recommendation is the smallest
power-of-two NSIDE, no larger than 65536, that satisfies that limit. Advice is
logging-only: neither the requested NSIDE nor an already loaded payload is
resampled, mutated, or changed automatically. A coarse grid produces:

.. code-block:: text

   HEALPix nside={actual} has pixel scale {pixel_rad:.6g} rad, above the Tier 3
   beam-product limit {limit_rad:.6g} rad (smallest feature {feature_rad:.6g} rad,
   safety factor 5, baseline {p}-{q}, frequency {frequency_hz:.6g} Hz). Use at least
   nside={recommended}; the requested NSIDE is unchanged.

The baseline, frequency, handler identities, metric kind, feature scale, pixel
limit, and recommendation identify the exact limiting canonical product.
Missing, ambiguous, non-finite, non-positive, or unmatched state raises
``BeamSamplingDerivationError`` rather than disabling advice.

Visibility-result provenance
----------------------------

Every successful point-source or HEALPix run publishes the exact immutable
loaded beam state:

.. code-block:: python

   result = simulator.run()
   beam_snapshot = result.beam_state.to_snapshot()

``to_snapshot()`` returns a fresh detached JSON-safe snapshot. It records mode,
canonical antenna assignments, analytic dimensions and
parameters, FITS resolved transport provenance and validated domains, handler
IDs, deduplication relationships, feature scales, and deterministic
fingerprints. It contains no ``UVBeam``, evaluator, data or backend array,
``BeamSystem``, lock, logger, renderer state, observability reference choice, or
``BeamSamplingRequirement``. Mutating a returned snapshot cannot change the
Simulator state or a later result.
