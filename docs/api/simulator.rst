Simulator API
=============

``Simulator`` accepts one resolved runtime. Its classmethods provide disjoint
YAML, typed-model, mapping, and typed-parameter construction paths.

.. autoclass:: radiosim.api.simulator.Simulator
   :members:
   :show-inheritance:
   :special-members: __init__

.. code-block:: python

   from radiosim import Simulator

   yaml_simulator = Simulator.from_yaml("configs/config.yaml")
   mapping_simulator = Simulator.from_mapping(mapping, base_dir=project_dir)
   model_simulator = Simulator.from_config(input_model, base_dir=project_dir)
   parameter_simulator = Simulator.from_parameters(
       instrument=instrument,
       baseline_selection=baseline_selection,
       channel_frequencies_hz=(100_000_000.0, 101_500_000.0),
       start_time="2025-01-01T00:00:00",
       sky_model=sky_model,
   )
   direct_simulator = Simulator(resolved_runtime)

The direct constructor rejects mappings and input models. Mapping/model inputs
with relative paths require ``base_dir``. No constructor executes CLI workflow
actions. ``run`` computes results; ``save``, ``plot``, and
``plot_observability`` are explicit helpers.

After setup, ``instrument`` returns the canonical resolved object, while
``antennas`` and ``baselines`` return its exact immutable tuples. Access before
resolution raises ``RuntimeError``.
