Quick Start Guide
=================

Validate the shipped offline example before running it:

.. code-block:: bash

   pixi run radiosim validate configs/config.yaml
   pixi run radiosim --config configs/config.yaml

The validation command resolves schema, semantics, paths, backend strategy,
precision, and frequency samples without constructing ``Simulator``, loading
scientific data, or creating output directories.

Python API
----------

Use ``from_yaml`` for a YAML path:

.. code-block:: python

   from radiosim import Simulator

   simulator = Simulator.from_yaml("configs/config.yaml")
   results = simulator.run(progress=False)
   print(f"Baselines: {len(results['visibilities'])}")

   # Saving is explicit in Python.
   simulator.save("output", format="hdf5")

``from_yaml`` resolves only scientific runtime configuration. It does not run
the document's CLI ``workflow`` actions.

Small programmatic simulation
-----------------------------

``from_parameters`` accepts exact channel values in Hz and builds the explicit
frequency variant:

.. code-block:: python

   from radiosim import Simulator
   from radiosim.io.config import ExecutionConfig, PrecisionInput

   simulator = Simulator.from_parameters(
       antenna_layout="antenna_layout_examples/hera_5.txt",
       antenna_file_format="radiosim",
       antenna_diameter_m=14.0,
       channel_frequencies_hz=(100_000_000.0, 101_500_000.0, 108_000_000.0),
       location={"lat": -30.72152, "lon": 21.4283, "height": 1073.0},
       start_time="2025-01-01T00:00:00",
       duration_seconds=1.0,
       time_step_seconds=1.0,
       sky_model={
           "sources": [{"kind": "test_sources", "num_sources": 3}]
       },
       execution=ExecutionConfig(
           backend="numpy",
           precision=PrecisionInput(preset="standard"),
           offline=True,
       ),
   )
   results = simulator.run(progress=False)

Result structure
----------------

The current result is a dictionary. ``results["visibilities"]`` maps a
baseline key to correlation-product arrays:

.. code-block:: python

   products_by_baseline = results["visibilities"]
   baseline = next(iter(products_by_baseline))
   products = products_by_baseline[baseline]
   print({name: values.shape for name, values in products.items()})

Configuration file
------------------

A complete small document looks like this:

.. code-block:: yaml

   antenna_layout:
     antenna_positions_file: ../antenna_layout_examples/hera_5.txt
     antenna_file_format: radiosim
     all_antenna_diameter: 14.0

   location:
     lat: -30.72152
     lon: 21.4283
     height: 1073.0

   obs_time:
     start_time: "2025-01-01T00:00:00"
     duration_seconds: 1.0
     time_step_seconds: 1.0

   obs_frequency:
     mode: grid
     starting_frequency: 100.0
     frequency_interval: 1.0
     frequency_bandwidth: 2.0
     frequency_unit: MHz

   sky_model:
     sources:
       - kind: test_sources
         num_sources: 3

   execution:
     backend: numpy
     precision:
       preset: standard
     simulator: rime
     offline: true

   workflow:
     output_dir: output
     save_results: false
     plot_results: false
     open_plots_in_browser: false
     save_log: false

Backend selection
-----------------

NumPy is the deterministic default. Configure ``execution.backend`` as
``numpy``, ``jax``, ``numba``, or ``auto``. The latter is a real selection
strategy. Backend resolution does not imply that every high-level scientific
kernel runs on a GPU; see :doc:`user_guide/backends`.

Next steps
----------

- :doc:`user_guide/configuration`
- :doc:`user_guide/configuration_support`
- :doc:`user_guide/backends`
- :doc:`user_guide/beam_models`
- :doc:`api/simulator`
