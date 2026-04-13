Quick Start Guide
=================

This guide will help you get started with RRIvis for radio interferometry
visibility simulations.

Basic Simulation
----------------

The simplest way to run a simulation is using the high-level ``Simulator`` class:

.. code-block:: python

   from rrivis import Simulator

   # Create simulator from a tagged config file
   sim = Simulator.from_config("config.yaml")

   # Setup and run
   sim.setup()
   results = sim.run(progress=True)

   # Save results
   sim.save("output/")

Using a Configuration File
--------------------------

For more complex simulations, use a YAML configuration file:

.. code-block:: yaml

   # config.yaml
   telescope:
     telescope_name: "HERA"

   antenna_layout:
     antenna_positions_file: "antennas.txt"
     all_antenna_diameter: 14.0

   obs_frequency:
     starting_frequency: 100.0
     frequency_bandwidth: 50.0
     frequency_interval: 1.0
     frequency_unit: "MHz"

   sky_model:
     sources:
       - kind: gleam
         flux_limit: 1.0
         max_rows: 10000

Then load and run:

.. code-block:: python

   from rrivis import Simulator

   sim = Simulator.from_config("config.yaml")
   sim.setup()
   results = sim.run()

GPU Acceleration
----------------

Enable GPU acceleration for faster simulations:

.. code-block:: python

   from rrivis import Simulator

   config = {
       "antenna_layout": {
           "antenna_positions_file": "antennas.txt",
           "antenna_file_format": "rrivis",
           "all_antenna_diameter": 14.0,
       },
       "obs_frequency": {
           "frequencies_hz": [100e6, 150e6, 200e6],
           "frequency_unit": "MHz",
       },
       "sky_model": {"sources": [{"kind": "gleam"}]},
       "visibility": {"sky_representation": "point_sources"},
   }

   # Auto-detect best backend
   sim = Simulator(config=config, backend="auto")

   # Or explicitly use JAX
   sim = Simulator(config=config, backend="jax")

   sim.setup()
   results = sim.run()  # Uses GPU if available

Check available backends:

.. code-block:: python

   from rrivis.backends import list_backends
   print(list_backends())  # ['numpy', 'jax', 'numba']

Accessing Results
-----------------

The simulation results contain visibility data:

.. code-block:: python

   results = sim.run()

   # Access visibilities by baseline (results is a dict)
   vis_01 = results["visibilities"][(0, 1)]  # Shape: (n_times, n_freqs)

   # Get metadata
   print(f"Number of baselines: {len(results['visibilities'])}")
   print(f"Frequencies: {results['frequencies']}")

Using Jones Matrices
--------------------

Add instrumental effects using Jones matrices:

.. code-block:: python

   from rrivis import Simulator
   from rrivis.core.jones import (
       JonesChain,
       GeometricDelayJones,
       BeamJones,
       GainJones,
   )

   # Create Jones chain
   jones = JonesChain([
       GeometricDelayJones(),
       BeamJones(beam_type="gaussian"),
       GainJones(amplitude_std=0.01),
   ])

   # Use in simulation (pass jones_config to run())
   sim = Simulator.from_config("config.yaml")
   sim.setup()
   results = sim.run()

Command Line Interface
----------------------

Run simulations from the command line:

.. code-block:: bash

   # Basic usage
   rrivis --config config.yaml --antenna-file antennas.txt

   # With specific backend
   rrivis --config config.yaml --backend jax

   # Show help
   rrivis --help

Low-Level API
-------------

For more control, use the low-level API:

.. code-block:: python

   from rrivis.core import calculate_visibility, generate_baselines
   from rrivis.io import read_antenna_positions
   from rrivis.backends import get_backend

   # Setup
   backend = get_backend("numpy")
   antennas = read_antenna_positions("antennas.txt")
   baselines = generate_baselines(antennas)

   # Calculate visibilities
   vis = calculate_visibility(
       baselines=baselines,
       sources=sources,
       frequencies=frequencies,
       backend=backend,
   )

Next Steps
----------

- :doc:`user_guide/configuration` - Detailed configuration options
- :doc:`user_guide/backends` - Backend selection and GPU setup
- :doc:`user_guide/jones_matrices` - Jones matrix framework
- :doc:`api/simulator` - Full API reference
