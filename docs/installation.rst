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

Backend extras install optional JAX or Numba dependencies:

.. code-block:: bash

   pip install radiosim[gpu-cuda]
   pip install radiosim[gpu]
   pip install radiosim[numba]

These extras make backend implementations available when the platform is
supported. They do not guarantee that every high-level RIME calculation runs
on a GPU. See :doc:`user_guide/backends` for the current execution boundary.

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
