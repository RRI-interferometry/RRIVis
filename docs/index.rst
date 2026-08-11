RadioSim Documentation
======================

RadioSim simulates radio-interferometric visibilities through a strict,
source-aware configuration system and a high-level ``Simulator`` API. NumPy is
the deterministic backend default. Optional backend selection is available,
but the current high-level scientific path does not promise end-to-end GPU
acceleration.

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   installation
   quickstart
   migration_guide

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   user_guide/configuration
   user_guide/instrument_resolution
   user_guide/configuration_support
   user_guide/backends
   user_guide/jones_matrices
   user_guide/jones_terms
   user_guide/sky_models
   user_guide/beam_models

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/simulator
   api/result
   api/algorithms
   api/core
   api/sky
   api/jones
   api/backends
   api/io
   api/observability
   api/visualization
   api/utils

.. toctree::
   :maxdepth: 1
   :caption: Development

   contributing
   changelog
   development/ci001_adjudication
   development/sci006_polarization_convention
   development/sci007_frame_accuracy_bound
   development/perf001_runtime_mitigations
   development/sci005_beam_physics_plan
   HERA_VSIM_ANALYSIS

Current high-level support
--------------------------

- strict YAML, mapping, and typed-model configuration;
- point-source and HEALPix direct-sum visibility paths;
- canonical instruments with typed baseline selection;
- analytic beams with heterogeneous positive antenna diameters;
- linear or circular two-feed receptors with a static feed rotation and one
  resolved array-wide output polarization basis;
- a typed ``jones:`` section carrying gains, bandpass, cable reflection,
  instrumental delay, cross-hand phase and delay, polarization leakage,
  parallactic rotation, troposphere, ionosphere, baseline closure error, and
  time and bandwidth smearing;
- exact grid or explicit-Hz frequency input;
- backend and precision selection through resolved configuration; and
- observability plotting as a ``Simulator`` helper.

FITS/mixed/per-antenna beams, HDF5, summary JSON, Measurement Set, and UVFITS
output are supported within their documented contracts, and each records the
resolved polarization basis. Elliptical or non-orthogonal feed pairs,
single-feed and multi-feed antennas, a frequency- or time-dependent receptor
basis, a non-scalar E-Jones, and later simulator modes remain separate work.

Quick example
-------------

.. code-block:: python

   from radiosim import Simulator

   simulator = Simulator.from_yaml("configs/config.yaml")
   result = simulator.run(progress=True)
   print(result.visibilities.shape)
   assert result is simulator.result

The YAML ``workflow`` section is executed only by config-mode CLI orchestration.
``Simulator.save`` and direct ``simulate --output`` use one exact final artifact
path and never prompt. Config mode publishes one owned, manifested run directory
under the selected ``error``, ``replace``, ``suffix``, or TTY-only ``prompt``
collision policy. ``Simulator.plot`` renders the published result into one
explicit directory from the canonical coordinate arrays, and configured
workflow plots are staged with the run and opened only after publication.

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
