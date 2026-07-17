I/O and Configuration API
=========================

Configuration boundaries
------------------------

.. autofunction:: radiosim.io.config.load_config

.. autofunction:: radiosim.io.config_resolution.resolve_config

.. autofunction:: radiosim.io.config.dump_config

``load_config`` and ``resolve_config`` return
``ResolvedConfiguration(runtime, workflow, provenance)``. ``dump_config``
accepts a strict ``RadioSimConfig`` input model and writes only user-facing
input state.

Input models
------------

.. autoclass:: radiosim.io.config.RadioSimConfig
   :members:
   :show-inheritance:

.. autoclass:: radiosim.io.config.FrequencyGridConfig
   :members:
   :show-inheritance:

.. autoclass:: radiosim.io.config.ExplicitFrequencyConfig
   :members:
   :show-inheritance:

``ObsFrequencyConfig`` is an annotated discriminated union rather than a class,
so its two concrete variants are documented above.

Resolved models
---------------

.. autoclass:: radiosim.core.runtime_config.ResolvedConfiguration
   :members:

.. autoclass:: radiosim.core.runtime_config.ResolvedSimulationConfig
   :members:

.. autoclass:: radiosim.core.runtime_config.ResolvedFrequencyConfig
   :members:

Example
-------

.. code-block:: python

   from radiosim.io import dump_config, load_config
   from radiosim.io.config import RadioSimConfig

   bundle = load_config("configs/config.yaml")
   print(bundle.runtime.execution.backend_strategy)
   print(bundle.workflow.output_dir)

   input_model = RadioSimConfig.model_validate(document_mapping)
   dump_config(input_model, "copied-config.yaml")

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
