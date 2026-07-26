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

FITS/mixed/per-antenna beams, receptor physics, heterogeneous observability,
UVFITS output, and later simulator modes are not supported.

Quick example
-------------

.. code-block:: python

   from radiosim import Simulator

   simulator = Simulator.from_yaml("configs/config.yaml")
   result = simulator.run(progress=True)
   print(result.visibilities.shape)
   assert result is simulator.result

The YAML ``workflow`` section is executed only by config-mode CLI orchestration.
During the canonical-result cutover, result saving and plotting are unavailable:
``Simulator.save`` and ``Simulator.plot`` raise ``ResultUnavailableError``
before side effects. Config-mode save/plot requests and direct ``simulate``
requests fail before runtime. Canonical output and rendering are later slices.

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
