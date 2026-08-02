Utilities
=========

``radiosim.utils`` holds the cross-cutting helpers the configuration, sky, and
solver layers share: coordinate conversion, cosmology, device inspection,
HEALPix helpers, logging configuration, the network-service policy, and the
pre-flight validator that collects every configuration error at once.

Package surface
---------------

.. automodule:: radiosim.utils
   :members:
   :undoc-members:
   :show-inheritance:

Configuration validation
------------------------

.. automodule:: radiosim.utils.validation
   :members:
   :undoc-members:
   :show-inheritance:

Network policy
--------------

``get_required_services`` reports which network-backed services a resolved sky
model needs, from the declarations the loader registry carries. It is what the
``Simulator`` pre-flight prints.

.. automodule:: radiosim.utils.network
   :members:
   :undoc-members:
   :show-inheritance:

Coordinates and cosmology
-------------------------

.. automodule:: radiosim.utils.coordinates
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: radiosim.utils.cosmology
   :members:
   :undoc-members:
   :show-inheritance:

HEALPix helpers
---------------

.. automodule:: radiosim.utils.healpix
   :members:
   :undoc-members:
   :show-inheritance:

Device inspection and logging
-----------------------------

.. automodule:: radiosim.utils.device
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: radiosim.utils.logging
   :members:
   :undoc-members:
   :show-inheritance:
