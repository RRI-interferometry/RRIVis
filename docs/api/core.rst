Core Module
===========

Canonical instrument
--------------------

.. automodule:: radiosim.core.instrument
   :members:
   :undoc-members:
   :show-inheritance:

Instrument resolution
---------------------

.. automodule:: radiosim.core.instrument_resolution
   :members:
   :undoc-members:
   :show-inheritance:

Solver instrument adapter
-------------------------

.. automodule:: radiosim.core.instrument_adapters
   :members:
   :undoc-members:
   :show-inheritance:

Baselines
---------

.. automodule:: radiosim.core.baseline_resolution
   :members:
   :undoc-members:
   :show-inheritance:

Visibility
----------

.. automodule:: radiosim.core.visibility
   :members:
   :undoc-members:
   :show-inheritance:

Beam and sky APIs
-----------------

The public beam boundary includes strict source-resolved definitions, canonical
assignment state, typed errors, immutable ``LoadedBeamState``,
``resolve_beam_assignments``, ``BeamSystem``, and ``load_beam_system``. All
four beam modes run through that single evaluator in the high-level
``Simulator``.

.. automodule:: radiosim.core.beam
   :members:
   :undoc-members:
   :show-inheritance:

See :doc:`../user_guide/beam_models`, :doc:`jones`, and
:doc:`../user_guide/sky_models` for the active boundaries.

Observation and polarization
----------------------------

.. automodule:: radiosim.core.observation
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: radiosim.core.polarization
   :members:
   :undoc-members:
   :show-inheritance:
