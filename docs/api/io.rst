I/O Module
==========

The ``rrivis.io`` module handles configuration, file I/O, and data persistence.

.. automodule:: rrivis.io
   :members:
   :undoc-members:
   :show-inheritance:

Configuration
-------------

Pydantic-based configuration models with validation.

.. automodule:: rrivis.io.config
   :members:
   :undoc-members:
   :show-inheritance:

RRIvisConfig
^^^^^^^^^^^^

.. autoclass:: rrivis.io.config.RRIvisConfig
   :members:
   :undoc-members:
   :show-inheritance:

TelescopeConfig
^^^^^^^^^^^^^^^

.. autoclass:: rrivis.io.config.TelescopeConfig
   :members:
   :undoc-members:
   :show-inheritance:

AntennaLayoutConfig
^^^^^^^^^^^^^^^^^^^

.. autoclass:: rrivis.io.config.AntennaLayoutConfig
   :members:
   :undoc-members:
   :show-inheritance:

ObsFrequencyConfig
^^^^^^^^^^^^^^^^^^

.. autoclass:: rrivis.io.config.ObsFrequencyConfig
   :members:
   :undoc-members:
   :show-inheritance:

Utility Functions
^^^^^^^^^^^^^^^^^

.. autofunction:: rrivis.io.config.load_config

.. autofunction:: rrivis.io.config.create_default_config

Writers
-------

.. automodule:: rrivis.io.writers
   :members:
   :undoc-members:
   :show-inheritance:

Antenna Readers
---------------

.. automodule:: rrivis.io.antenna_readers
   :members:
   :undoc-members:
   :show-inheritance:

Measurement Set I/O
-------------------

CASA Measurement Set format support for interoperability with standard
radio astronomy tools (CASA, QuartiCal, WSClean).

.. note::

   Measurement Set support requires additional dependencies::

       pip install rrivis[ms]

   Or install python-casacore directly::

       pip install python-casacore

.. automodule:: rrivis.io.measurement_set
   :members:
   :undoc-members:
   :show-inheritance:

write_ms
^^^^^^^^

.. autofunction:: rrivis.io.measurement_set.write_ms

read_ms
^^^^^^^

.. autofunction:: rrivis.io.measurement_set.read_ms

ms_info
^^^^^^^

.. autofunction:: rrivis.io.measurement_set.ms_info

Example Usage
-------------

Configuration Loading
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from rrivis.io.config import load_config, RRIvisConfig

   # Load from YAML file
   config = load_config("config.yaml")

   # Access validated configuration
   print(config.telescope.telescope_name)
   print(config.obs_frequency.n_channels)

   # Create programmatically
   config = RRIvisConfig(
       telescope={"telescope_name": "HERA"},
       obs_frequency={"starting_frequency": 100.0},
   )

   # Save to YAML
   config.to_yaml("output_config.yaml")

Measurement Set Export
^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from rrivis import Simulator
   from rrivis.io import write_ms, read_ms, ms_info, MS_AVAILABLE

   # Check if MS support is available
   if MS_AVAILABLE:
       # Run simulation
       sim = Simulator.from_config("config.yaml")
       results = sim.run()

       # Save as Measurement Set
       sim.save("output/", format="ms")

       # Or use write_ms directly
       write_ms(
           "simulation.ms",
           visibilities=results["visibilities"],
           frequencies=results["frequencies"],
           antennas=results["antennas"],
           baselines=results["baselines"],
           location=results["location"],
           obstime=results["obstime"],
       )

       # Get MS info
       info = ms_info("simulation.ms")
       print(f"Antennas: {info['n_antennas']}")
       print(f"Channels: {info['n_channels']}")

       # Read back
       data = read_ms("simulation.ms")

After creating a Measurement Set, you can:

- View in CASA: ``casabrowser simulation.ms``
- Calibrate with QuartiCal: ``goquartical simulation.ms``
- Image with WSClean: ``wsclean -name image simulation.ms``
