Jones Matrix API
================

The modules below expose the Jones framework. Exported classes do not all
represent implemented high-level effects. Geometric phase and the analytic
primary beam are the substantive effects in the current ``Simulator`` path;
many other terms remain identity scaffolds or later-tier work.

Base and chain
--------------

.. automodule:: radiosim.core.jones.base
   :members:
   :show-inheritance:

.. automodule:: radiosim.core.jones.chain
   :members:
   :show-inheritance:

Implemented high-level term modules
-----------------------------------

.. automodule:: radiosim.core.jones.geometric
   :members:
   :show-inheritance:

.. automodule:: radiosim.core.jones.beam.analytic
   :members:
   :show-inheritance:

Scaffolded and low-level modules
--------------------------------

The remaining Jones modules are documented for development and inspection.
Check each implementation and its scientific tests before use. A returned
identity matrix is not a modeled physical effect, and low-level FITS beam
classes do not make FITS beams supported by the high-level resolver.

.. automodule:: radiosim.core.jones.ionosphere
   :members:
   :show-inheritance:

.. automodule:: radiosim.core.jones.troposphere
   :members:
   :show-inheritance:

.. automodule:: radiosim.core.jones.parallactic
   :members:
   :show-inheritance:

.. automodule:: radiosim.core.jones.gain
   :members:
   :show-inheritance:

.. automodule:: radiosim.core.jones.bandpass
   :members:
   :show-inheritance:

.. automodule:: radiosim.core.jones.polarization_leakage
   :members:
   :show-inheritance:
