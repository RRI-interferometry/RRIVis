Compute Backends
================

RRIvis supports multiple compute backends for CPU and GPU execution.

Available Backends
------------------

NumPy (Default)
^^^^^^^^^^^^^^^

The NumPy backend is always available and serves as the baseline implementation.

- **Pros**: Always available, well-tested, portable
- **Cons**: CPU-only, single-threaded
- **Best for**: Development, small simulations, testing

JAX
^^^

The JAX backend provides GPU acceleration with automatic differentiation.

- **Pros**: GPU support (NVIDIA/AMD/Apple/TPU), automatic differentiation, JIT compilation
- **Cons**: Requires JAX installation, larger memory footprint
- **Best for**: Large simulations, gradient-based optimization, Apple Silicon

Numba
^^^^^

The Numba backend provides GPU acceleration via CUDA/ROCm.

- **Pros**: CUDA and ROCm support, parallel CPU execution, low-level optimization
- **Cons**: Requires Numba installation, CUDA toolkit for GPU
- **Best for**: Production workloads, NVIDIA/AMD GPUs

Backend Selection
-----------------

Auto-detection:

.. code-block:: python

   from rrivis.backends import get_backend

   # Auto-detect best available backend
   backend = get_backend("auto")

Explicit selection:

.. code-block:: python

   # Specific backend
   backend = get_backend("numpy")
   backend = get_backend("jax")
   backend = get_backend("numba")

Check available backends:

.. code-block:: python

   from rrivis.backends import list_backends

   available = list_backends()
   print(available)  # ['numpy', 'jax', 'numba']

Using with Simulator
--------------------

Pass backend to Simulator:

.. code-block:: python

   from rrivis import Simulator

   # Auto-detect
   sim = Simulator(backend="auto")

   # Specific backend
   sim = Simulator(backend="jax")

Installation
------------

NumPy backend is included by default.

JAX Backend
^^^^^^^^^^^

For NVIDIA GPU (CUDA 12):

.. code-block:: bash

   pip install rrivis[gpu-cuda]

For AMD GPU (ROCm):

.. code-block:: bash

   pip install rrivis[gpu-rocm]

For Apple Silicon:

.. code-block:: bash

   pip install rrivis[gpu]

For TPU:

.. code-block:: bash

   pip install rrivis[tpu]

Numba Backend
^^^^^^^^^^^^^

.. code-block:: bash

   pip install rrivis[numba]

Backend API
-----------

All backends implement a common interface:

.. code-block:: python

   from rrivis.backends import get_backend

   backend = get_backend("numpy")

   # Array creation
   x = backend.array([1, 2, 3])
   zeros = backend.zeros((10, 10))
   ones = backend.ones((5, 5), dtype=backend.complex128)

   # Math operations
   y = backend.sin(x)
   z = backend.exp(1j * x)

   # Linear algebra
   result = backend.matmul(A, B)
   conj = backend.conj(z)

   # Reductions
   total = backend.sum(x)
   mean = backend.mean(x)

Performance Comparison
----------------------

Approximate speedups (vs NumPy baseline):

.. list-table::
   :header-rows: 1

   * - Simulation Size
     - JAX (GPU)
     - Numba (GPU)
   * - 100 antennas, 100 sources
     - 5x
     - 3x
   * - 500 antennas, 1000 sources
     - 20x
     - 15x
   * - 1000 antennas, 10000 sources
     - 50x
     - 40x

GPU Memory Management
---------------------

For large simulations, manage GPU memory:

.. code-block:: python

   import os

   # JAX memory allocation
   os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
   os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.8"

   from rrivis import Simulator

   sim = Simulator(backend="jax")
   # ...

Troubleshooting
---------------

JAX not detecting GPU:

.. code-block:: python

   import jax
   print(jax.devices())  # Should show GPU devices

Numba CUDA issues:

.. code-block:: bash

   # Check CUDA installation
   nvcc --version

   # Set CUDA path if needed
   export CUDA_HOME=/usr/local/cuda
