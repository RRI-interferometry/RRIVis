Simulator API
=============

``Simulator`` accepts one already resolved scientific runtime object. The
classmethods provide the public source-specific construction paths.

.. autoclass:: radiosim.api.simulator.Simulator
   :members:
   :show-inheritance:
   :special-members: __init__

Construction
------------

.. code-block:: python

   from radiosim import Simulator

   yaml_simulator = Simulator.from_yaml("configs/config.yaml")
   mapping_simulator = Simulator.from_mapping(mapping, base_dir=project_dir)
   model_simulator = Simulator.from_config(input_model, base_dir=project_dir)
   direct_simulator = Simulator(resolved_runtime)

The direct constructor rejects mappings and input models. Mapping/model inputs
with relative paths require ``base_dir``. ``from_parameters`` is the concise
programmatic path for complete inputs and exact channel values in Hz.

Workflow separation
-------------------

No constructor executes CLI ``workflow`` actions. ``run`` computes results;
``save``, ``plot``, and ``plot_observability`` are explicit helpers.
``plot_observability`` is a visualization capability associated with the
Simulator, not a separate engine or product.
