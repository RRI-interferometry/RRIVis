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

Visibility solvers
------------------

``radiosim.core.visibility`` solves the point-source RIME and
``radiosim.core.visibility_healpix`` the HEALPix diffuse RIME; both route every
array operation through the selected backend. ``radiosim.core.hybrid`` decides
which components a run solves and sums their cubes while they are still backend
arrays. ``radiosim.core.contraction`` holds the one compiled kernel in the
package -- the baseline-batched per-``(time, frequency)`` contraction -- and
``radiosim.core.solver_partition`` splits a run into the blocks the solvers
iterate.

.. automodule:: radiosim.core.visibility
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: radiosim.core.visibility_healpix
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: radiosim.core.hybrid
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: radiosim.core.contraction
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: radiosim.core.solver_partition
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

Observation geometry and time
-----------------------------

.. automodule:: radiosim.core.observation
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: radiosim.core.time_grid
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: radiosim.core.phase_center
   :members:
   :undoc-members:
   :show-inheritance:

Polarization, receptors, and precision
--------------------------------------

``radiosim.core.polarization`` owns the Stokes-to-coherency conversion with its
1/2 factor, ``radiosim.core.polarization_basis`` is the single canonical
correlation-coordinate table for ``linear_xy`` and ``circular_rl``,
``radiosim.core.receptor`` resolves the per-antenna receptor set, and
``radiosim.core.precision`` carries the dtype contract every loader and solver
honours.

.. automodule:: radiosim.core.polarization
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: radiosim.core.polarization_basis
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: radiosim.core.receptor
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: radiosim.core.precision
   :members:
   :undoc-members:
   :show-inheritance:
