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

Measurement Set I/O
-------------------

Measurement Set support requires ``python-casacore`` or ``radiosim[ms]``.
Its current generic API is retained unchanged until the later standard-format
slice.  Measurement Set has not been migrated to the canonical result model.

.. automodule:: radiosim.io.measurement_set
   :members:
   :show-inheritance:

Versioned HDF5 results
----------------------

``radiosim.visibility`` schema version ``1.0.0`` is the complete,
reconstructable Tier 4 result format.  Its canonical extension is ``.h5``.
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
``time, baseline, frequency, correlation`` order, with the exact correlations
``XX, XY, YX, YY``.  It also preserves flags, weights, two-part UTC sample
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
science arrays.  UTF-8 strings and compact JSON are bounded and parsed without
dynamic evaluation.  The legacy unversioned files are rejected because they had
unsafe baseline-name parsing and incomplete scientific fields; there is no
legacy reader.

The legacy unsafe HDF5 function pair was removed immediately with no
compatibility aliases.  High-level ``Simulator.save`` integration remains
unavailable until a later slice.  Summary JSON, canonical Measurement Set
migration, and UVFITS are also later slices.

Resolved-configuration workflow artifact
----------------------------------------

``radiosim.io.writers`` temporarily retains only the resolved-configuration
YAML artifact helper needed by the pre-workflow state.
