RRIvis Documentation
====================

**RRIvis** is a Python package for simulating radio interferometry visibilities
with GPU acceleration support. It implements the Radio Interferometer Measurement
Equation (RIME) with full polarization support.

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

Features
--------

- **GPU Acceleration**: Universal GPU support via JAX (NVIDIA/AMD/Apple Silicon/TPU) and Numba
- **Full Polarization**: Complete RIME implementation with 2x2 Jones matrices
- **Jones Matrix Framework**: 8 Jones terms for comprehensive instrumental modeling
- **Multiple Sky Models**: GLEAM, GSM, HEALPix, and custom point sources
- **Flexible Beam Models**: Analytic and FITS-based beam patterns
- **High-Level API**: Simple ``Simulator`` class for notebooks and scripts
- **Type-Safe Configuration**: Pydantic-based validation

Quick Example
-------------

.. code-block:: python

   from rrivis import Simulator

   # Create simulator with auto-detected backend
   sim = Simulator(backend="auto")

   # Configure simulation
   sim.setup(
       antenna_layout="antennas.txt",
       frequencies=[100, 150, 200],  # MHz
       sky_model="gleam",
   )

   # Run simulation
   results = sim.run(progress=True)

   # Save results
   sim.save("output.h5")

Installation
------------

.. code-block:: bash

   # Basic installation
   pip install rrivis

   # With GPU support (NVIDIA)
   pip install rrivis[gpu-cuda]

   # With all optional dependencies
   pip install rrivis[all]

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
