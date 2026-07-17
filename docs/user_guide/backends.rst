Compute Backends
================

RadioSim resolves a requested backend strategy before runtime setup. NumPy is
the deterministic default. JAX and Numba are optional; ``auto`` is a real
selection strategy.

Configuration
-------------

.. code-block:: yaml

   execution:
     backend: numpy  # numpy | jax | numba | auto
     precision:
       preset: standard

For a YAML document, a call-site override is explicit:

.. code-block:: python

   from radiosim import Simulator
   from radiosim.io.config_resolution import SimulationOverrides

   simulator = Simulator.from_yaml(
       "configs/config.yaml",
       overrides=SimulationOverrides(backend="auto"),
   )

Omitting the override preserves the document value. ``None`` is the only
no-override sentinel; ``auto`` never means “keep the document.”

Direct backend API
------------------

.. code-block:: python

   from radiosim.backends import get_backend, list_backends

   print(list_backends())
   backend = get_backend("numpy")
   values = backend.asarray([1.0, 2.0, 3.0])
   print(backend.sum(values))

The backend factory constructs an explicit backend or resolves ``auto`` from
the installed, precision-compatible options. An unavailable explicit backend
fails rather than silently switching to another backend.

Precision
---------

A precision override replaces the complete precision tree; it is not deeply
merged with the document. Presets and custom leaves are mutually exclusive in
one value. Explicit JAX/Numba plus ``float128`` is rejected during configuration
resolution, before importing the optional backend.

Runtime truth
-------------

Backend selection is wired into resolved configuration and the backend factory.
It does not imply end-to-end acceleration of every high-level calculation.
Current orchestration still includes host-side Python loops, Astropy coordinate
work, transfers, and paths with incomplete backend coverage.

Consequently:

- successful JAX or Numba selection is not proof that the complete simulation
  ran on a GPU;
- optional-backend correctness tests do not establish performance;
- HEALPix and point-source paths must be assessed separately; and
- RadioSim publishes no unverified speedup multiplier.

A performance claim needs a reproducible workload, hardware and accelerator,
backend version, precision, problem dimensions, setup/compile time,
steady-state time, transfer time, peak memory, and correctness tolerance against
NumPy.

Installation
------------

NumPy ships with the base installation. Optional extras install backend
dependencies only:

.. code-block:: bash

   pip install radiosim[gpu-cuda]
   pip install radiosim[gpu]
   pip install radiosim[numba]

Consult the selected backend's own installation documentation for supported
hardware and platform details. Installation alone does not change the
high-level support boundary above.
