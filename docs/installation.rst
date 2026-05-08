Installation
============

Requirements
------------

- Python 3.11 or higher
- NumPy >= 1.24
- Astropy >= 5.0

Basic Installation
------------------

Install RadioSim from PyPI:

.. code-block:: bash

   pip install radiosim

GPU Support
-----------

RadioSim supports GPU acceleration through JAX and Numba backends.

NVIDIA GPU (CUDA 12)
^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   pip install radiosim[gpu-cuda]

AMD GPU (ROCm)
^^^^^^^^^^^^^^

.. code-block:: bash

   pip install radiosim[gpu-rocm]

Apple Silicon (Metal)
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   pip install radiosim[gpu]

This automatically detects and uses Metal on M1/M2/M3/M4 Macs.

Google TPU
^^^^^^^^^^

.. code-block:: bash

   pip install radiosim[tpu]

Optional Dependencies
---------------------

Numba Backend
^^^^^^^^^^^^^

For the Numba backend with Dask support:

.. code-block:: bash

   pip install radiosim[numba]

Measurement Set I/O
^^^^^^^^^^^^^^^^^^^

For CASA Measurement Set support:

.. code-block:: bash

   pip install radiosim[ms]

Development
^^^^^^^^^^^

For development tools (pytest, ruff, pyright):

.. code-block:: bash

   pip install radiosim[dev]

Documentation
^^^^^^^^^^^^^

For building documentation:

.. code-block:: bash

   pip install radiosim[docs]

All Dependencies
^^^^^^^^^^^^^^^^

Install everything:

.. code-block:: bash

   pip install radiosim[all]

Development Installation
------------------------

For development, use pixi:

.. code-block:: bash

   # Clone repository
   git clone https://github.com/kartikmandar/RadioSim.git
   cd RadioSim

   # Install with pixi
   pixi install

   # Activate environment
   pixi shell

   # Run tests
   pytest

Verifying Installation
----------------------

Verify your installation:

.. code-block:: python

   import radiosim
   print(radiosim.__version__)  # Should print "0.2.0"

   # Check available backends
   from radiosim.backends import list_backends
   print(list_backends())  # ['numpy', 'jax', 'numba'] depending on install

   # Quick test
   from radiosim import Simulator
   sim = Simulator()
   print("RadioSim installed successfully!")
