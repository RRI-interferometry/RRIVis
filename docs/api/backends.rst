Backends Module
===============

The ``rrivis.backends`` module provides compute backend abstraction for
CPU and GPU execution.

.. automodule:: rrivis.backends
   :members:
   :undoc-members:
   :show-inheritance:

Backend Selection
-----------------

.. autofunction:: rrivis.backends.get_backend

.. autofunction:: rrivis.backends.list_backends

Base Backend
------------

.. automodule:: rrivis.backends.base
   :members:
   :undoc-members:
   :show-inheritance:

NumPy Backend
-------------

.. automodule:: rrivis.backends.numpy_backend
   :members:
   :undoc-members:
   :show-inheritance:

JAX Backend
-----------

.. automodule:: rrivis.backends.jax_backend
   :members:
   :undoc-members:
   :show-inheritance:

Numba Backend
-------------

.. automodule:: rrivis.backends.numba_backend
   :members:
   :undoc-members:
   :show-inheritance:

Example Usage
-------------

.. code-block:: python

   from rrivis.backends import get_backend, list_backends

   # Check available backends
   print(list_backends())  # ['numpy', 'jax', 'numba']

   # Get auto-detected backend
   backend = get_backend("auto")

   # Use backend operations
   x = backend.array([1, 2, 3])
   y = backend.sin(x)
   result = backend.sum(y)
