Backends Module
===============

The ``radiosim.backends`` module provides compute backend abstraction for
CPU and GPU execution.

.. automodule:: radiosim.backends
   :members:
   :undoc-members:
   :show-inheritance:

Backend Selection
-----------------

.. autofunction:: radiosim.backends.get_backend

.. autofunction:: radiosim.backends.list_backends

Base Backend
------------

.. automodule:: radiosim.backends.base
   :members:
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

Numba Backend
-------------

.. automodule:: radiosim.backends.numba_backend
   :members:
   :undoc-members:
   :show-inheritance:

Example Usage
-------------

.. code-block:: python

   from radiosim.backends import get_backend, list_backends

   # Check available backends
   print(list_backends())  # ['numpy', 'jax', 'numba']

   # Get auto-detected backend
   backend = get_backend("auto")

   # Use backend operations
   x = backend.array([1, 2, 3])
   y = backend.sin(x)
   result = backend.sum(y)
