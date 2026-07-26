Quick Start Guide
=================

Validate and run the shipped offline example:

.. code-block:: bash

   pixi run radiosim validate configs/config.yaml
   pixi run radiosim --config configs/config.yaml

Validation resolves schema, semantics, source paths, backend policy,
precision, and frequency samples without constructing a backend, loading the
instrument or sky, creating output, or opening a browser.

YAML construction
-----------------

.. code-block:: python

   from radiosim import Simulator

   simulator = Simulator.from_yaml("configs/config.yaml")
   result = simulator.run(progress=False)
   print(f"Result shape: {result.visibilities.shape}")
   print(f"Stokes-I shape: {result.stokes_i().shape}")
   assert result is simulator.result

``from_yaml`` resolves scientific configuration but never executes CLI
``workflow`` actions.

The canonical result also exposes ``flags``, ``weights``, ``time_grid``,
``frequencies_hz``, ``channel_widths_hz``, ``correlations``,
``scientific_sha256``, and ``provenance_sha256``. Saving and result plotting
are deliberately unavailable in this slice: ``Simulator.save`` and
``Simulator.plot`` raise ``ResultUnavailableError`` before output mutation.
Config-mode save/plot requests and direct ``simulate`` requests fail before
runtime; output transactions, standard formats, and canonical rendering are
implemented in later slices.

Typed parameter construction
----------------------------

.. code-block:: python

   from pathlib import Path

   from radiosim import Simulator
   from radiosim.io.config import ExecutionConfig, PrecisionInput
   from radiosim.io.instrument_config import (
       BaselineSelectionConfig,
       InstrumentConfig,
       InstrumentLocationConfig,
       LayoutFileSourceConfig,
   )

   instrument = InstrumentConfig(
       source=LayoutFileSourceConfig(
           path=Path("antenna_layout_examples/hera_5.txt"),
           format="radiosim",
           telescope_name="HERA",
       ),
       location=InstrumentLocationConfig(
           longitude_deg=21.4283,
           latitude_deg=-30.72152,
           height_m=1073.0,
       ),
       default_diameter_m=14.0,
   )

   simulator = Simulator.from_parameters(
       instrument=instrument,
       baseline_selection=BaselineSelectionConfig(correlations="all"),
       channel_frequencies_hz=(100_000_000.0, 101_500_000.0),
       channel_widths_hz=(1_000_000.0, 1_000_000.0),
       start_time="2025-01-01T00:00:00",
       sky_model={
           "sources": [{"kind": "test_sources", "num_sources": 3}]
       },
       execution=ExecutionConfig(
           backend="numpy",
           precision=PrecisionInput(preset="standard"),
           offline=True,
       ),
   )
   result = simulator.run(progress=False)
   assert result is simulator.result

After ``setup`` or ``run``, ``simulator.instrument`` is the canonical immutable
instrument, and ``antennas`` and ``baselines`` are canonical tuples. Results
include those exact tuples and a detached provenance snapshot.

Configuration file
------------------

.. code-block:: yaml

   instrument:
     source:
       kind: layout_file
       path: ../antenna_layout_examples/hera_5.txt
       format: radiosim
       telescope_name: HERA
     location:
       longitude_deg: 21.4283
       latitude_deg: -30.72152
       height_m: 1073.0
     default_diameter_m: 14.0

   baseline_selection:
     correlations: all
     length_filter: null
     azimuth_ranges_deg: []

   obs_time:
     start_time: "2025-01-01T00:00:00"
     duration_seconds: 1.0
     time_step_seconds: 1.0

   obs_frequency:
     mode: explicit
     channel_frequencies_hz: [100000000.0, 101500000.0]
     channel_widths_hz: [1000000.0, 1000000.0]

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

NumPy is the deterministic default. JAX, Numba, and ``auto`` are selectable,
but selection does not prove full GPU coverage.

Next steps
----------

- :doc:`user_guide/configuration`
- :doc:`user_guide/instrument_resolution`
- :doc:`user_guide/configuration_support`
- :doc:`user_guide/beam_models`
- :doc:`api/simulator`
