Installation
============

Requirements
------------

- Python 3.11 or 3.12
- NumPy and Astropy from the package dependency set

Basic installation
------------------

.. code-block:: bash

   pip install radiosim

Optional dependencies
---------------------

Backend extras install optional JAX or Dask dependencies:

.. code-block:: bash

   pip install radiosim[gpu-cuda]
   pip install radiosim[gpu]
   pip install radiosim[dask]

These extras make backend implementations available when the platform is
supported. RadioSim has measured no accelerator, so none of them establishes
that any RIME calculation runs on a GPU. See :doc:`user_guide/backends` for the
execution boundary, the compilation boundary, and the measured NumPy/JAX-CPU/Dask
comparison.

Measurement Set support requires its optional dependency:

.. code-block:: bash

   pip install radiosim[ms]

Development installation
------------------------

The repository uses Pixi for repeatable development environments:

.. code-block:: bash

   git clone https://github.com/RRI-interferometry/RadioSim.git
   cd RadioSim
   pixi install
   pixi run test

Other quality commands are:

.. code-block:: bash

   pixi run lint
   pixi run check-format
   pixi run typecheck
   make -C docs clean html

Verify installation
-------------------

.. code-block:: python

   import radiosim
   from radiosim.backends import list_backends

   print(radiosim.__version__)
   print(list_backends())

Run a configuration-only smoke test without backend/device or scientific-loader
construction:

.. code-block:: bash

   radiosim validate configs/config.yaml
