Core Module
===========

The ``radiosim.core`` package contains the core astronomy calculations including
visibility computation, antenna handling, baseline generation, and beam models.

Visibility
----------

.. automodule:: radiosim.core.visibility
   :members:
   :undoc-members:
   :show-inheritance:

Antenna
-------

.. automodule:: radiosim.core.antenna
   :members:
   :undoc-members:
   :show-inheritance:

Baseline
--------

.. automodule:: radiosim.core.baseline
   :members:
   :undoc-members:
   :show-inheritance:

Beam and sky APIs
-----------------

The current high-level ``Simulator`` connects analytic beams and strict sky
source requests. See :doc:`jones` for the implemented/scaffold distinction and
:doc:`../user_guide/sky_models` for the lower-level sky preparation API. FITS
and per-antenna beam execution remain rejected by the strict high-level
configuration resolver.

Observation
-----------

.. automodule:: radiosim.core.observation
   :members:
   :undoc-members:
   :show-inheritance:

Polarization
------------

.. automodule:: radiosim.core.polarization
   :members:
   :undoc-members:
   :show-inheritance:
