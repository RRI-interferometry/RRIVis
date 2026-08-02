Sky Model
=========

``radiosim.core.sky`` is the sky-model subpackage. ``SkyModel`` is a frozen
container holding two optional payloads -- ``sky.point`` (``PointSourceData``)
and ``sky.healpix`` (``HealpixData``) -- and a model may carry either or both.
Mutate a model with ``.replace(**changes)``; the arrays themselves are
read-only after construction.

The package facade re-exports the public surface of the subpackages below, so
``from radiosim.core.sky import SkyModel`` and
``from radiosim.core.sky.containers import SkyModel`` name the same object.
Each symbol is documented once, under the subpackage that defines it.

.. automodule:: radiosim.core.sky
   :no-members:
   :no-special-members:

Containers
----------

Frozen dataclasses: ``SkyModel`` itself, the ``PointSourceData`` columnar
arrays with their morphology, polarization, metadata and per-channel spectrum
sub-blocks, ``HealpixData`` with first-class sparse support, the coverage
footprint, and ``SkyProvenance``.

.. automodule:: radiosim.core.sky.containers
   :members:
   :undoc-members:
   :show-inheritance:

Loaders
-------

Module-level loader functions, each registered with the loader registry. Every
loader requires an explicit ``precision=PrecisionConfig(...)``.

.. automodule:: radiosim.core.sky.loaders
   :members:
   :undoc-members:
   :show-inheritance:

Registry and catalog metadata
-----------------------------

The loader registry is the single source of truth for config fields, aliases,
network services, source category, and source representation; the catalog
parameter tables live beside it.

.. automodule:: radiosim.core.sky.registry
   :members:
   :undoc-members:
   :show-inheritance:

Combination
-----------

``prepare_sky_model()`` combines and materializes contributed models, applying
the physical-disjointness checks and the optional ``assume_disjoint`` escape.

.. automodule:: radiosim.core.sky.combine
   :members:
   :undoc-members:
   :show-inheritance:

Operations
----------

Mutation-free transforms, the ``create_from_arrays()`` /
``create_test_sources()`` factories, representation conversion, and region
selection.

.. automodule:: radiosim.core.sky.operations
   :members:
   :undoc-members:
   :show-inheritance:

Recipes
-------

Composite sky recipes that call more than one loader.

.. automodule:: radiosim.core.sky.recipes
   :members:
   :undoc-members:
   :show-inheritance:

Diagnostics
-----------

Analysis, discovery, and polarization diagnostics over a resolved model.

.. automodule:: radiosim.core.sky.diagnostics
   :members:
   :undoc-members:
   :show-inheritance:

Serialization
-------------

.. automodule:: radiosim.core.sky.io
   :members:
   :undoc-members:
   :show-inheritance:

Support
-------

Shared helpers used by the loaders and operations, including the lazy healpy
accessor that keeps point-only import paths from loading healpy.

.. automodule:: radiosim.core.sky.support
   :members:
   :undoc-members:
   :show-inheritance:
