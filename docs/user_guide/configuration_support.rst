Configuration Support Matrix
============================

All public configuration sources use the same strict resolver.

Entry points
------------

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - Entry point
     - Input
     - Result
   * - ``load_config(path)``
     - YAML with paths based at its parent
     - Resolved runtime, workflow, and provenance
   * - ``resolve_config(config, source=...)``
     - Mapping or ``RadioSimConfig`` with source context
     - The same resolved bundle
   * - ``Simulator(resolved)``
     - ``ResolvedSimulationConfig`` only
     - Runtime object without workflow state
   * - ``Simulator.from_yaml(path)``
     - YAML document
     - Simulator through the common resolver
   * - ``Simulator.from_config(model, base_dir=...)``
     - Strict input model
     - Simulator through the common resolver
   * - ``Simulator.from_mapping(data, base_dir=...)``
     - Python mapping
     - Simulator through the common resolver
   * - ``Simulator.from_parameters(...)``
     - Typed instrument, typed baseline selection, and scientific values
     - Simulator with explicit-Hz frequency input
   * - ``radiosim validate``
     - YAML document
     - Resolved summary without runtime/output work

Scientific ownership
--------------------

.. list-table::
   :header-rows: 1
   :widths: 24 76

   * - Section
     - Active behavior
   * - ``instrument``
     - One discriminated source resolves canonical identities, positions,
       location, diameters, and deterministic provenance
   * - ``baseline_selection``
     - Typed correlation, length-target/range, and axial-azimuth filtering
   * - ``beams``
     - ``analytic``, ``shared_fits``, ``per_antenna_fits``, and ``mixed`` all
       resolve and run through one canonical Simulator beam system
   * - ``receptors``
     - Linear or circular two-feed receptors per antenna, static
       ``feed_rotation_deg``, and one resolved array-wide ``output_basis`` that
       names the reported correlation labels
   * - ``sky_model``
     - Strict source requests for point or HEALPix preparation
   * - ``obs_time`` / ``obs_frequency``
     - Canonical UTC sample centers and exposures, exact frequency centers,
       and required positive channel widths
   * - ``visibility``
     - Point-source, HEALPix, or summed ``hybrid`` direct sum
   * - ``jones``
     - One optional block per enabled term — ``G``, ``B``, ``Rc``, ``Kd``,
       ``X``, ``D``, ``P``, ``Z``, ``T``, ``M``, ``Q`` — applied in the
       canonical chain order regardless of key order; absence selects the
       current empty optional-term inventory while the always-present factors
       remain active
   * - ``execution``
     - Backend, precision, RIME simulator, and offline policy
   * - ``workflow``
     - CLI-only saving, logging, plotting, prompting, and browser policy

Instrument source support
-------------------------

``layout_file`` supports ``radiosim``, ``casa_loc``, ``measurement_set``,
``uvfits``, and ``mwa_metafits``. ``known_telescope`` uses a named registry
source with an explicit offline/network policy. It is a source kind, not a
file format. See :doc:`instrument_resolution`.

Feature boundaries
------------------

Heterogeneous positive antenna diameters are used by both point and HEALPix
visibility paths. Observability selects the same canonical beam evaluator and
requires an explicit reference antenna for scientifically heterogeneous
assignments.

Receptor and polarization-basis physics is implemented for ideal orthogonal
two-feed receptors: the receptor-configuration term ``C`` and the
basis-transform term ``H`` are substantive, both bases run end to end, and the
resolved basis names the correlation labels in memory, in HDF5, in the summary
JSON, in Measurement Set and UVFITS exports, and in every renderer. Polarization
leakage (``D``), parallactic rotation (``P``), gains (``G``) and bandpass
(``B``) are implemented too, together with cable reflection (``Rc``),
instrumental delay (``Kd``), cross-hand phase and delay (``X``), troposphere
(``T``), ionosphere (``Z``), and the two baseline-dependent terms ``M`` and
``Q``; all of them are configured under ``jones`` and documented in
:doc:`jones_terms`. Elliptical or non-orthogonal feed pairs, single-feed and
multi-feed antennas, and a frequency- or time-dependent receptor basis are not
implemented. Arbitrary BeamFITS variants, explicit Measurement Set phase
centres, and spherical-harmonic simulation are also not implemented; worker
policy is configurable through ``execution.sky_loading`` and
``execution.solver``.

Receptor support by mode
------------------------

.. list-table::
   :header-rows: 1
   :widths: 34 22 44

   * - Declaration
     - Resolved ``output_basis``
     - Reported correlations
   * - omitted section, or ``basis: linear`` with ``output_basis: auto``
     - ``linear_xy``
     - ``XX, XY, YX, YY``
   * - ``basis: circular`` with ``output_basis: auto``
     - ``circular_rl``
     - ``RR, RL, LR, LL``
   * - any array with ``output_basis: linear``
     - ``linear_xy``
     - ``XX, XY, YX, YY``
   * - any array with ``output_basis: circular``
     - ``circular_rl``
     - ``RR, RL, LR, LL``
   * - mixed bases with ``output_basis: auto``
     - rejected
     - ``AmbiguousOutputBasisError`` naming both antenna counts

A non-zero ``feed_rotation_deg`` is the **static** topocentric part of the
receptor orientation for the whole observation. The time-dependent part is the
separate ``jones.P`` term, and the two compose: the static feed rotation and
the mount-dependent field rotation add. With
:math:`\alpha_p=\eta_p\psi_p+\nu_p\mathrm{el}`, an antenna at
``feed_rotation_deg`` is the receptor at :math:`\chi_p+\alpha_p`. Ordinary
alt-az has :math:`\alpha_p=\psi_p`; Nasmyth mounts retain their signed elevation
term. Which antennas rotate is a property of the instrument's per-antenna
``mount_type``, not of the
``receptors`` section: an array with a rotating mount and no ``jones.P`` is
rejected, and an array with no rotating mount that configures ``jones.P`` is
rejected as an identity.

Beam support by stage
---------------------

.. list-table::
   :header-rows: 1
   :widths: 30 20 20 30

   * - Beam declaration
     - Schema
     - Path resolution
     - Simulator runtime
   * - ``analytic``: ``circular_aperture``
     - Supported
     - Supported
     - Supported
   * - ``analytic``: ``rectangular_aperture``,
       ``elliptical_aperture``, ``analytical_illumination``, or
       ``numerical_illumination``
     - Supported
     - Supported
     - Supported
   * - ``shared_fits``
     - Supported
     - Supported
     - Supported within the accepted scalar or full-efield subset
   * - ``per_antenna_fits``
     - Supported
     - Supported
     - Supported within the accepted scalar or full-efield subset
   * - ``mixed``
     - Supported
     - Supported
     - Supported within the accepted scalar or full-efield subset
   * - ``beams.aperture_physics``
     - Supported
     - Not applicable
     - Supported on the declared circular pupils only
   * - ``beams.surface_error.*.error_beam_diagnostic``
     - Supported
     - Not applicable
     - Supported on an unobstructed circular pupil only
   * - ``beams.squint``
     - Supported
     - Not applicable
     - Supported on ``analytic`` beams only

``beams.aperture_physics`` composes a central blockage, support-leg shadows and
a deterministic real unit-RMS disk Zernike surface height inside one normalized
aperture transform. It is accepted only where the design declares an exact
compact aperture-plane profile: ``circular_aperture`` with a ``uniform``,
``parabolic`` or ``parabolic_squared`` taper, and ``analytical_illumination``
with a ``parabolic`` or ``parabolic_squared`` taper profile. Gaussian, cosine
and numerical illuminations, the rectangular and elliptical families, and every
BeamFITS source are rejected with ``UnsupportedConfigError``.

``beams.squint`` (SCI-005 Stage 2) samples the analytic beam at two oppositely
displaced native-feed directions and composes them into a generally full ``E``
following the exact Cotton/Uson arcsine law; see :ref:`stage2-beam-squint`. It
is accepted only when the resolved beams mode is ``analytic`` — every
``shared_fits``, ``per_antenna_fits``, and ``mixed`` document carrying a
squint block is rejected with ``UnsupportedConfigError`` before any
antenna-reference matching, because a measured BeamFITS pattern may already
contain the physical feed displacement and the accepted scalar subset carries
no metadata by which RadioSim could prove it does not.

``beams.beam.normalization`` (SCI-005 Stage 3) selects which accepted subset of
a BeamFITS file is read. The default ``peak`` is the accepted scalar subset,
whose evaluated response is one complex voltage on the diagonal of ``E``.
``uvbeam_peak_common_v1`` is the accepted full-efield subset: the file's
complete complex ``data_array`` is converted by the frozen constant matrix
``M = [[0, 1], [-1, 0]]`` into the chain's own sky tangent pair and factorized
against the antenna's own resolved receptor, so ``E`` is a generally full 2x2
matrix; see
:ref:`stage3-full-efield`. The literal names an accepted interpretation of the
committed bytes rather than a normalizing operation, and the two subsets are
different readings of the same ``beam_type: efield`` file rather than a strict
widening of one another. A phased-array antenna response, a station or
array-factor model, mutual coupling between elements, and near-field or
Fresnel-regime behaviour all remain unimplemented.

The nested ``error_beam_diagnostic`` declaration is validated, resolved and
fingerprinted, and it is deliberately *not* a Jones voltage: it can never change
a cross-baseline visibility. Its ensemble-power record is read through
``BeamSystem.evaluate_ruze_power_diagnostic``. Version 1 requires an
*unobstructed* pupil: attaching it to an antenna whose aperture physics declares
a blockage is refused with ``UnsupportedConfigError`` and issue code
``beam.ruze_power_diagnostic.unsupported_obstruction``, because the paired
region of two shifted copies of the support mask needs boundary and topology
families this version does not freeze. Nothing else is refused -- the coherent
``surface_error`` loss and the blockage mask in ``E`` keep their accepted
behaviour.

FITS path validation checks and records sources but does not read BeamFITS
content. ``Simulator.setup`` resolves canonical antenna references, loads and
validates the accepted subset named by the source's ``normalization`` literal,
and publishes state atomically. This does not imply arbitrary BeamFITS variants,
GPU interpolation, automatic NSIDE mutation, or resampling support.

NumPy is the deterministic backend default. Selecting ``jax``, ``dask``, or
``auto`` does not establish accelerator coverage for the high-level workflow;
no accelerator has been measured, and ``numba`` is no longer a selectable
value.

Output boundary
---------------

``Simulator.save`` accepts an exact final artifact path and a typed
``ResultFormat`` for HDF5, summary JSON, Measurement Set, or UVFITS. All four
accept either solver arm: a direct ``rime`` run and an ``execution.simulator:
mmode`` run publish the same ``(time, baseline, frequency, correlation)`` cube
in the same four correlation labels, and the writers differ only in the solver
provenance they carry.  HDF5 preserves the complete tagged m-mode snapshot and
reconstructs it on read; summary JSON publishes the same snapshot as bounded
metadata, together with the ERA-derived exposure rule and the extremes of the
synthesized integration widths, which for an m-mode grid are not one repeated
cadence; UVFITS and Measurement Set carry HISTORY lines naming the m-mode,
time-grid, frame, harmonic and Stokes-``V``-bridge conventions beside the
projection record.  All five paths publish the same synthesized UTC sample
centres and integration widths.  Direct
Python and ``simulate`` calls never prompt or suffix. Config mode preflights
``collision_policy`` and manifest ownership before runtime, builds the complete
run in sibling staging, and publishes atomically. Summary JSON is explicitly
incomplete metadata; HDF5 is the complete reconstructable result.
``Simulator.plot`` renders the published result into one explicit directory
from the canonical coordinate arrays.  Configured workflow plotting is
preflighted before runtime — only ``plotting_backend: bokeh`` is implemented —
then staged with the run and opened from published paths last.

Retained visualization controls are ``plot_results``, ``open_plots_in_browser``,
``plotting_backend``, and ``visibility_phase_unit`` (exactly ``radians`` or
``degrees``).  Every other visualization input was removed and is rejected with
exact migration text.
