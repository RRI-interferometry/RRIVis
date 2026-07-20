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

.. automodule:: radiosim.io.measurement_set
   :members:
   :show-inheritance:

Writers and readers
-------------------

.. automodule:: radiosim.io.writers
   :members:
   :show-inheritance:
