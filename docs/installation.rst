Installation
============

Requirements
------------

- Python 3.11 or higher
- NumPy >= 1.24
- Astropy >= 5.0

Basic Installation
------------------

Install RRIvis from PyPI:

.. code-block:: bash

   pip install rrivis

GPU Support
-----------

RRIvis supports GPU acceleration through JAX and Numba backends.

NVIDIA GPU (CUDA 12)
^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   pip install rrivis[gpu-cuda]

AMD GPU (ROCm)
^^^^^^^^^^^^^^

.. code-block:: bash

   pip install rrivis[gpu-rocm]

Apple Silicon (Metal)
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   pip install rrivis[gpu]

This automatically detects and uses Metal on M1/M2/M3/M4 Macs.

Google TPU
^^^^^^^^^^

.. code-block:: bash

   pip install rrivis[tpu]

Optional Dependencies
---------------------

Numba Backend
^^^^^^^^^^^^^

For the Numba backend with Dask support:

.. code-block:: bash

   pip install rrivis[numba]

Measurement Set I/O
^^^^^^^^^^^^^^^^^^^

For CASA Measurement Set support:

.. code-block:: bash

   pip install rrivis[ms]

Development
^^^^^^^^^^^

For development tools (pytest, ruff, pyright):

.. code-block:: bash

   pip install rrivis[dev]

Documentation
^^^^^^^^^^^^^

For building documentation:

.. code-block:: bash

   pip install rrivis[docs]

All Dependencies
^^^^^^^^^^^^^^^^

Install everything:

.. code-block:: bash

   pip install rrivis[all]

Development Installation
------------------------

For development, use pixi:

.. code-block:: bash

   # Clone repository
   git clone https://github.com/kartikmandar/RRIvis.git
   cd RRIvis

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

   import rrivis
   print(rrivis.__version__)  # Should print "0.2.0"

   # Check available backends
   from rrivis.backends import list_backends
   print(list_backends())  # ['numpy', 'jax', 'numba'] depending on install

   # Quick test
   from rrivis import Simulator
   sim = Simulator()
   print("RRIvis installed successfully!")
