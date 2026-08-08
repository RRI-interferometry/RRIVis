I/O and Configuration API
=========================

Configuration boundaries
------------------------

.. autofunction:: radiosim.io.config.load_config

.. autofunction:: radiosim.io.config_resolution.resolve_config

.. autofunction:: radiosim.io.config.dump_config

``load_config`` and ``resolve_config`` return
``ResolvedConfiguration(runtime, workflow, provenance)``. ``dump_config``
accepts a strict user-input model.

Input models
------------

.. autoclass:: radiosim.io.config.RadioSimConfig
   :members:

.. autoclass:: radiosim.io.instrument_config.InstrumentConfig
   :members:

.. autoclass:: radiosim.io.instrument_config.LayoutFileSourceConfig
   :members:

.. autoclass:: radiosim.io.instrument_config.KnownTelescopeSourceConfig
   :members:

.. autoclass:: radiosim.io.instrument_config.InstrumentLocationConfig
   :members:

.. autoclass:: radiosim.io.instrument_config.AntennaDiameterOverrideConfig
   :members:

.. autoclass:: radiosim.io.instrument_config.BaselineSelectionConfig
   :members:

.. autoclass:: radiosim.io.instrument_config.LengthTargetsConfig
   :members:

.. autoclass:: radiosim.io.instrument_config.LengthRangesConfig
   :members:

.. autoclass:: radiosim.io.instrument_config.AzimuthRangeConfig
   :members:

.. autoclass:: radiosim.io.config.FrequencyGridConfig
   :members:

.. autoclass:: radiosim.io.config.ExplicitFrequencyConfig
   :members:

Resolved models
---------------

.. autoclass:: radiosim.core.runtime_config.ResolvedConfiguration
   :members:

.. autoclass:: radiosim.core.runtime_config.ResolvedSimulationConfig
   :members:

Instrument sources
------------------

.. automodule:: radiosim.io.instrument_sources
   :members:
   :undoc-members:
   :show-inheritance:

Versioned HDF5 results
----------------------

``radiosim.visibility`` schema version ``4.0.0`` is the complete,
reconstructable result format.  Its canonical extension is ``.h5``.
The direct APIs are:

.. autofunction:: radiosim.io.hdf5.write_result_hdf5

.. autofunction:: radiosim.io.hdf5.load_result_hdf5

.. autoclass:: radiosim.io.hdf5.HDF5ReadLimits
   :members:

``write_result_hdf5(result, path, *, overwrite=False)`` accepts an exact
:class:`~radiosim.core.result.SimulationResult`.
``load_result_hdf5(path, *, limits=HDF5ReadLimits())`` returns an immutable
:class:`~radiosim.core.result.LoadedSimulationResult`.  A loaded result
contains detached identity and provenance snapshots rather than live
instrument, beam, backend, or Simulator services.

The schema losslessly preserves complex64 or complex128 visibilities in
``time, baseline, frequency, correlation`` order.  The correlation labels are
data, not a constant: a ``linear_xy`` result stores ``XX, XY, YX, YY`` and a
``circular_rl`` result stores ``RR, RL, LR, LL``, always in the row-major
flattening of the 2x2 visibility matrix, so indices ``0`` and ``3`` are the
parallel hands in both bases.  The written ``polarization_basis`` and
``correlations`` are validated against each other on read, and only those two
label tuples in that order are accepted.

Schema ``4.0.0`` also stores a ``receptors`` group: the resolved
``output_basis``, the ``receptor_sha256`` fingerprint, and per-antenna
``antenna_number``, ``antenna_name``, ``basis``, ``feed_rotation_rad``, and
``feed_angle_rad`` in canonical instrument antenna order.  A loaded result
therefore reconstructs which receptor produced which correlation.

The schema also preserves flags, weights, two-part UTC sample
centres, integration durations, frequency centres and channel widths,
canonical antenna and baseline identity and geometry, location, phase centre,
and immutable provenance.  Both the scientific fingerprint and the
provenance fingerprint are stored and recomputed on read.  Complex256 is
rejected instead of being silently cast.

Publication uses an exclusively created same-directory temporary regular file.
The writer flushes and fsyncs it, closes it, performs a complete read-back,
verifies scientific equality and both fingerprints, then uses atomic
no-clobber or replacement publication and fsyncs the parent directory.  Python
writer APIs never prompt.

The reader treats every file as hostile.  It validates the exact object and
attribute allowlists, links, ranks, shapes, dtypes, byte order, dimension
labels, units, chunks, filters, and :class:`HDF5ReadLimits` before allocating
payloads.  RadioSim-authored text and compact JSON use inspectable fixed-width
UTF-8 datasets: scalar widths are encoded byte widths, array widths are the
maximum encoded element width, and short values use trailing NUL padding only.
VLEN strings, ASCII-tagged text where UTF-8 is required, oversized fixed
items, oversized fixed datasets, and aggregate JSON widths are rejected from
metadata before any value-read API or payload allocation.  After that preflight,
the reader validates padding and strict UTF-8 and parses JSON without dynamic
evaluation.

The legacy unversioned files are rejected because they had unsafe baseline-name
parsing and incomplete scientific fields; there is no legacy reader.  Files
written by the rejected VLEN implementation are also unsafe inputs and
are rejected; there is no VLEN compatibility reader, migration shim, or
fallback.  Schema ``1.0.0`` files predate the receptor group and the
basis-driven correlation labels, and schema ``2.0.0`` files predate the solved
sky components, their element counts, and the per-component solver timings, so
a summed hybrid result cannot be told from a single-component one.  Schema
``3.0.0`` files predate the Jones record, so a run that corrupted its
visibilities with a configured gain or bandpass could not be told from a clean
one.  All three are rejected with ``UnsupportedSchemaVersionError`` and none is
upgraded in place.  Re-run the simulation to obtain a ``4.0.0`` file.

Schema ``4.0.0`` stores an **optional** ``jones`` group: ``enabled_terms``,
``chain_order``, the per-term resolved configuration as ``term_snapshots_json``,
the resolved antenna mount types as ``mount_types_json``, and the
``jones_sha256`` that enters the scientific fingerprint.  The group is written
only when a run enabled a Jones term, and a file without it is read as "no terms
enabled" rather than rejected — which is why a run with no ``jones:`` section
produces the file it always produced, apart from the version.  It is the one
optional group in the schema; every other group is mandatory, and a partially
present ``jones`` group is rejected like any other allowlist mismatch.

Schema ``4.0.0`` records the components in ``provenance/solver_json``
(``components`` and ``component_element_counts``, in the canonical
``point``, ``healpix`` order) and their wall times in
``provenance/performance_json`` (``solver_point_seconds`` and
``solver_healpix_seconds``).  The reader validates both records against the
canonical dataclass field sets, bounds the component list, and cross-checks the
declared ``sky_representation`` against the embedded resolved configuration,
all before any science payload is allocated.  Component names and counts are
deterministic and enter the scientific fingerprint; the timings are not and do
not.

The legacy unsafe HDF5 function pair was removed immediately with no
compatibility aliases.  High-level ``Simulator.save`` dispatches this writer
only for ``ResultFormat.HDF5``.

Truthful summary JSON
---------------------

.. autofunction:: radiosim.io.summary_json.write_result_summary_json

``ResultFormat.SUMMARY_JSON`` uses the canonical ``.summary.json`` extension
and schema ``radiosim.result-summary`` version ``1.2.0``.  It reports result
shape, dtype, units, fingerprints, flags/weights summaries, canonical axes,
detached identity/provenance snapshots, performance, and history.  Its
``correlation`` block names the exact labels and the resolved
``polarization_basis``, and its bounded ``receptors`` block reports
``output_basis``, ``receptor_sha256``, ``native_basis_counts``, and the sorted
``distinct_feed_rotations_deg``.  Per-antenna receptor rows are deliberately
absent; the complete set lives in the HDF5 ``receptors`` group.  It
explicitly excludes visibility samples, full flags and weights, full
coordinates, and antenna/baseline geometry.  The complete UTF-8 payload is
limited to 16 MiB before filesystem mutation and uses the atomic regular-file
publication policy.  It has no reader and cannot reconstruct a result; HDF5 is
the complete reconstructable RadioSim format.

Standard visibility exchange
----------------------------

Measurement Set and UVFITS exports use one shared projection of an exact
``SimulationResult`` and return an immutable ``StandardVisibilityData`` from
their readers.  Neither reader reconstructs a native ``SimulationResult`` or
``LoadedSimulationResult``.  The projection preserves the canonical time-major,
baseline-inner layout and explicitly maps the result's own correlation labels —
``XX, XY, YX, YY`` for ``linear_xy`` or ``RR, RL, LR, LL`` for ``circular_rl`` —
into each file's standard ordering.  It derives the ICRS first-time zenith from
the first two-part UTC centre, phases exactly once before writing, records both
source fingerprints, the resolved ``polarization_basis``, the
``receptor_sha256``, and the original zenith-drift semantics in HISTORY, and
never mutates the source result.

Polarization mapping
~~~~~~~~~~~~~~~~~~~~

One shared table, ``radiosim.core.polarization_basis``, owns every label and
code for both bases.  No writer or reader hard-codes a second copy; the
format-specific maps are derived from it.

.. list-table::
   :header-rows: 1
   :widths: 32 34 34

   * - Surface
     - ``linear_xy``
     - ``circular_rl``
   * - In-memory labels
     - ``XX, XY, YX, YY``
     - ``RR, RL, LR, LL``
   * - Canonical AIPS codes
     - ``-5, -7, -8, -6``
     - ``-1, -3, -4, -2``
   * - UVFITS stored axis order
     - ``-5, -6, -7, -8``
     - ``-1, -2, -3, -4``
   * - pyuvdata ``feed_array``
     - ``x, y``
     - ``r, l``
   * - Measurement Set ``CORR_TYPE``
     - casacore ``Stokes`` 9-12
     - casacore ``Stokes`` 5-8

The canonical order is the row-major flattening of the 2x2 visibility matrix,
which is what a freshly written Measurement Set's ``POLARIZATION`` table
contains because pyuvdata preserves the order it is handed.  A UVFITS file
stores its polarization axis in descending code order instead, and pyuvdata's
Measurement Set reader canonicalizes ``polarization_array`` into that same
descending order on read-back.  The two orders must not be confused.

``feed_array`` records the nominal receptor orientation for the resolved output
basis, written uniformly for every antenna: an unrotated linear pair has
``feed_angle_rad`` ``(pi/2, 0)`` and an unrotated circular pair ``(0, 0)``.  The
per-antenna native basis and feed rotation are RadioSim provenance and are never
inferred by a reader from ``feed_array``.  Because pyuvdata does not
cross-validate ``feed_array`` against ``polarization_array``, RadioSim enforces
that coupling itself and rejects an input whose declared feeds disagree with its
polarization axis.  Readers accept exactly the two label tuples above, in
exactly those orders; a reordering is rejected rather than permuted, because the
correlation axis order is part of the contract.

.. autoclass:: radiosim.io.standard_visibility.StandardVisibilityData
   :members:

.. autoclass:: radiosim.io.standard_visibility.ProjectedPhaseCenter
   :members:

.. autoclass:: radiosim.io.standard_visibility.StandardReadLimits
   :members:

Measurement Set
~~~~~~~~~~~~~~~

.. autofunction:: radiosim.io.measurement_set.write_measurement_set

.. autofunction:: radiosim.io.measurement_set.read_measurement_set

Measurement Set support is loaded only for a requested operation.  Its
optional dependencies are installed with ``radiosim[ms]``.  DATA storage is
complex64: complex64 input is retained within the documented tolerance, while
complex128 input is explicitly converted and recorded as lossy in HISTORY.
The writer passes ``force_phase=False`` and publishes a verified sibling
temporary directory with atomic no-replace or directory exchange.  Readers
inspect metadata and enforce ``StandardReadLimits`` before loading science
arrays.  Arbitrary canonical selected-baseline subsets, autos, crosses, and
explicit per-channel widths use the same standard projection.  The former
generic Measurement Set surface and its availability booleans are removed.

UVFITS
~~~~~~

.. autofunction:: radiosim.io.uvfits.write_uvfits

.. autofunction:: radiosim.io.uvfits.read_uvfits

UVFITS preserves supported complex64 and complex128 visibility storage, but it
requires one to 255 canonically numbered antennas and a regular spectral grid
whose equal channel width matches the channel spacing.  Unsupported results
are rejected before optional dependencies or filesystem mutation; use HDF5 or
Measurement Set instead.  UVFITS is not lossless for arbitrary nonuniform
channels, complex256, the original time-varying AltAz phase, the complete
configuration tree, or provenance beyond bounded HISTORY records.

The writer uses a fresh sibling regular file, passes ``force_phase=False``,
performs a complete read-back, and then uses the regular-file atomic
publication contract.  The reader validates FITS random-group and antenna
table headers before pyuvdata data allocation and then returns only validated
``StandardVisibilityData``.

``Simulator.save`` dispatches these writers only for ``ResultFormat.MS`` and
``ResultFormat.UVFITS``.  Standard-format optional dependencies stay lazy, and
the atomic writers never prompt.

Owned CLI workflow transaction
------------------------------

Config-mode CLI output is one staged run directory containing
``manifest.json``, ``resolved-config.yaml``, optional ``simulation.log``, and
the selected result artifact.  The strict
``radiosim.workflow-manifest.v1`` manifest lists sorted safe relative paths and
SHA-256 hashes.  A nonempty run is replaceable only when that manifest validates
the exact contained artifacts; malformed, traversing, linked, aliased, or
unlisted content never authorizes replacement.

``collision_policy`` is exactly ``error``, ``replace``, ``suffix``, or
``prompt``.  Prompting occurs only for a valid owned run on a TTY and before
simulation or filesystem mutation.  Python and direct ``simulate`` never
prompt.  The CLI writes, verifies, closes, hashes, and fsyncs staging before one
atomic directory publish.  Failure before publication removes staging and
preserves the old run.  Configured plots are rendered into the same staged
directory with browser presentation disabled, recorded in the ownership
manifest, and opened from their published paths only after the atomic publish
succeeds.
