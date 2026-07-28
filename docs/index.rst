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
   user_guide/sky_models
   user_guide/beam_models

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/simulator
   api/core
   api/backends
   api/io
   api/jones

.. toctree::
   :maxdepth: 1
   :caption: Development

   contributing
   changelog

Current high-level support
--------------------------

- strict YAML, mapping, and typed-model configuration;
- point-source and HEALPix direct-sum visibility paths;
- canonical instruments with typed baseline selection;
- analytic beams with heterogeneous positive antenna diameters;
- exact grid or explicit-Hz frequency input;
- backend and precision selection through resolved configuration; and
- observability plotting as a ``Simulator`` helper.

FITS/mixed/per-antenna beams, HDF5, summary JSON, Measurement Set, and UVFITS
output are supported within their documented contracts. Receptor physics and
later simulator modes remain separate work.

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
collision policy. Canonical result plotting remains a later Tier 4G slice.

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
