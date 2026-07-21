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

Tier 3B exposes immutable source-resolved beam definitions and four resolved
mode inputs. These values describe validated intent; only direct-circular
analytic input is active in the high-level Simulator. No loaded beam state,
assignment state, or canonical beam evaluator is public yet.

.. automodule:: radiosim.core.beam.models
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
