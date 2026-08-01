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

A FITS source has ``kind: fits``, ``path``, ``normalization: peak``,
``angular_interpolation: bilinear``, and ``frequency_interpolation: cubic`` or
``linear``. The first three option values shown are defaults where applicable.

The accepted FITS subset is deliberately scalar. RadioSim accepts finite
``efield`` or ``simple`` data on a regular full-visible-hemisphere ``az_za``
grid, a fixed antenna mount, east-oriented linear X/Y feeds, a finite identity
basis transform, unit bandpass, peak normalization, and a strictly increasing
frequency axis. The evaluated voltage is the scalar complex response on the
diagonal of a 2x2 E-Jones matrix. Power beams, circular feeds in the *file*,
non-identity bases, other coordinate systems or mounts, and arbitrary
cross-polarization are rejected. Angular interpolation is bilinear; frequency
interpolation is exactly linear or cubic with no extrapolation or method
fallback.

Rejecting circular feeds in a BeamFITS file does not restrict the receptor
model. ``receptors`` supplies the receptor basis and any static feed rotation
independently of the beam, and the scalar E-Jones response multiplies both bases
identically. Polarization leakage and a beam that genuinely differs between the
two feeds remain Tier 7 work.

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
Absent, they change nothing: the cube, every beam fingerprint, and
``scientific_sha256`` are what they were without them.

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
in ``docs/development/beam_physics_scope.md``: polarized and cross-polar beams,
beam squint, aperture blockage, Zernike aberrations, and the Ruze error-beam
decomposition.

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
