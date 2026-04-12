Configuration Guide
===================

RRIvis uses Pydantic-based configuration for type-safe, validated settings.

Configuration File Format
-------------------------

Configuration files use YAML format:

.. code-block:: yaml

   telescope:
     telescope_name: "HERA"
     use_pyuvdata_telescope: false

   antenna_layout:
     antenna_positions_file: "antennas.txt"
     antenna_file_format: "rrivis"
     all_antenna_diameter: 14.0

   beams:
     beam_mode: "analytic"
     all_beam_response: "gaussian"
     beam_za_max_deg: 90.0

   location:
     lat: -30.72
     lon: 21.43
     height: 1073.0

   obs_frequency:
     starting_frequency: 100.0
     frequency_interval: 1.0
     frequency_bandwidth: 50.0
     frequency_unit: "MHz"

   obs_time:
     time_interval: 1.0
     time_interval_unit: "hours"
     total_duration: 1.0
     total_duration_unit: "days"
     start_time: "2025-01-01T00:00:00"

   sky_model:
     flux_unit: "Jy"
     mixed_model_policy: "error"
     sources:
       - kind: gleam
         flux_limit: 1.0
       - kind: diffuse_sky
         model: gsm2008
         nside: 64

   visibility:
     sky_representation: "healpix_map"
     allow_lossy_point_materialization: false

   output:
     output_file_format: "HDF5"
     save_simulation_data: true

Configuration Sections
----------------------

The ``sky_model`` section is source-driven: ``sky_model.sources`` is a list of
loader specifications, each with a required ``kind`` and loader-specific
arguments. Global options such as ``flux_unit`` and
``brightness_conversion`` live alongside that list.

Two policy fields are especially important:

- ``sky_model.mixed_model_policy`` controls whether point catalogs may be
  combined with diffuse HEALPix models. The default is ``"error"``.
- ``visibility.allow_lossy_point_materialization`` controls whether the
  simulator may convert HEALPix maps into point sources when
  ``visibility.sky_representation`` is ``"point_sources"``. The default
  is ``false``.

Telescope
^^^^^^^^^

.. list-table::
   :header-rows: 1

   * - Parameter
     - Type
     - Default
     - Description
   * - telescope_name
     - str
     - "Unknown"
     - Name of telescope
   * - use_pyuvdata_telescope
     - bool
     - false
     - Use pyuvdata telescope definition
   * - use_pyuvdata_location
     - bool
     - false
     - Use pyuvdata location
   * - use_pyuvdata_antennas
     - bool
     - false
     - Use pyuvdata antenna positions

Antenna Layout
^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1

   * - Parameter
     - Type
     - Default
     - Description
   * - antenna_positions_file
     - str
     - None
     - Path to antenna positions file
   * - antenna_file_format
     - str
     - "rrivis"
     - File format: rrivis, casa, uvfits, mwa, pyuvdata
   * - all_antenna_diameter
     - float
     - 14.0
     - Default antenna diameter (m)

Observation Frequency
^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1

   * - Parameter
     - Type
     - Default
     - Description
   * - starting_frequency
     - float
     - 50.0
     - Starting frequency
   * - frequency_interval
     - float
     - 1.0
     - Frequency channel width
   * - frequency_bandwidth
     - float
     - 100.0
     - Total bandwidth
   * - frequency_unit
     - str
     - "MHz"
     - Unit: Hz, kHz, MHz, GHz

Observation Time
^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1

   * - Parameter
     - Type
     - Default
     - Description
   * - time_interval
     - float
     - 1.0
     - Time between samples
   * - time_interval_unit
     - str
     - "hours"
     - Unit: seconds, minutes, hours, days
   * - total_duration
     - float
     - 1.0
     - Total observation duration
   * - total_duration_unit
     - str
     - "days"
     - Unit: seconds, minutes, hours, days
   * - start_time
     - str
     - None
     - ISO format start time

Loading Configuration
---------------------

From YAML file:

.. code-block:: python

   from rrivis.io.config import load_config

   config = load_config("config.yaml")
   print(config.telescope.telescope_name)

Programmatic creation:

.. code-block:: python

   from rrivis.io.config import RRIvisConfig, TelescopeConfig

   config = RRIvisConfig(
       telescope=TelescopeConfig(telescope_name="HERA"),
       obs_frequency={"starting_frequency": 100.0},
   )

Create default config file:

.. code-block:: python

   from rrivis.io.config import create_default_config

   create_default_config("my_config.yaml")

Validation
----------

Configuration is validated automatically:

.. code-block:: python

   from rrivis.io.config import RRIvisConfig
   from pydantic import ValidationError

   try:
       config = RRIvisConfig(
           obs_frequency={"starting_frequency": -100}  # Invalid!
       )
   except ValidationError as e:
       print(e)  # Helpful error message

Exporting Configuration
-----------------------

Save to YAML:

.. code-block:: python

   config.to_yaml("output_config.yaml")

Convert to dictionary:

.. code-block:: python

   config_dict = config.to_dict()
