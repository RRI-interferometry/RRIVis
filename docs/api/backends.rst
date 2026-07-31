Backends Module
===============

The ``radiosim.backends`` package provides a common array/backend abstraction.
Availability and selection do not by themselves prove end-to-end GPU execution
of a high-level simulation.

Backend Selection
-----------------

.. autofunction:: radiosim.backends.get_backend

.. autofunction:: radiosim.backends.list_backends

Base Backend
------------

.. automodule:: radiosim.backends.base
   :members:
   :exclude-members: precision
   :undoc-members:
   :show-inheritance:

NumPy Backend
-------------

.. automodule:: radiosim.backends.numpy_backend
   :members:
   :undoc-members:
   :show-inheritance:

JAX Backend
-----------

.. automodule:: radiosim.backends.jax_backend
   :members:
   :undoc-members:
   :show-inheritance:

Dask Backend
------------

.. automodule:: radiosim.backends.dask_backend
   :members:
   :undoc-members:
   :show-inheritance:

Example Usage
-------------

.. code-block:: python

   from radiosim.backends import get_backend, list_backends

   # Check availability in this environment.
   print(list_backends())

   # Resolve the current automatic backend strategy.
   backend = get_backend("auto")

   # Use the common array interface.
   x = backend.asarray([1, 2, 3])
   y = backend.sin(x)
   result = backend.sum(y)
