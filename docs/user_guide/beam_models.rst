Beam Models
===========

``beams`` is a strict discriminated union. Each document selects one complete
mode; unknown fields and incomplete mode shapes are rejected.

Runtime boundary
----------------

Tier 3B accepts and source-resolves all four modes below. The high-level
``Simulator`` currently activates only ``mode: analytic`` with
``model.kind: circular_aperture``. FITS-backed modes fail with
``beam_runtime_fits_pending``. Rectangular, elliptical, analytical-
illumination, and numerical-illumination models fail with
``beam_runtime_analytic_variant_pending``. These guards run before device,
backend, network, UVBeam, output, plotting, or browser work. There is no
analytic fallback for a FITS declaration.

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

The other declared analytic variants are complete input and resolution
contracts, but are not yet Simulator-active:

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
``kind: number`` or ``kind: name``. Tier 3B preserves those references; it does
not resolve them against an instrument or load BeamFITS content.

Path and provenance rules
-------------------------

YAML-relative FITS paths use the YAML file's parent. Mapping, typed-model, and
parameter construction require ``base_dir`` for relative FITS paths. ``~`` is
expanded, environment-variable syntax is rejected, and every checked FITS path
must exist, be a readable regular file, and is normalized through symlinks.
``check_input_paths=False`` skips existence/type/readability checks but still
normalizes the path. Each source records its indexed logical path, such as
``beams.assignments[2].beam.path``, in configuration provenance.

Resolution constructs immutable definitions with deterministic fingerprints
from the complete normalized analytic model or FITS source options. It does not
read FITS content, import UVBeam, resolve antenna assignments, create a
``BeamSystem``, or promise solver/observability FITS support.

``Simulator.plot_observability`` is a planning visualization, not a simulation
product. In Tier 3B it uses the same direct-circular projection as simulation
and still rejects heterogeneous-diameter footprint semantics.
