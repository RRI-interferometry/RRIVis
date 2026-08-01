Jones Matrix API
================

The modules below expose the Jones framework. Every exported term declares
``term_status``, which is exactly ``"implemented"`` or ``"planned"``:

- **implemented** — the term's physics exists, is exercised by the solver, and
  is covered by analytic invariant tests. Today: the receptor terms ``C`` and
  ``H`` in ``radiosim.core.jones.receptor``, the geometric phase
  ``geometric_phase()`` (the ``K`` term, a function rather than a class because
  it is per-baseline), and the ``E`` term, which is the canonical
  ``BeamSystem`` reached through a private solver-owned adapter.
- **planned** — the term has a name, a documented physical effect and a
  position in the canonical chain, and ``compute_jones_batch`` **raises**. It
  never multiplies by the identity, it declares no capability flag that cannot
  be verified, and it accepts no parameter it would discard.

There are no other exported terms. Twenty-six classes that returned the 2x2
identity for every input, for effects RadioSim does not plan to model — turbulent
and GPS ionospheres, w-projection, element beams and array factors, fringe
fitting, Mueller and IXR leakage variants, and the rest — were removed before
v1.0;
:doc:`../migration_guide` names the replacement for each. See
:doc:`../user_guide/jones_matrices` for the receptor mathematics and the chain
order.

Base and chain
--------------

.. automodule:: radiosim.core.jones.base
   :members:
   :show-inheritance:

.. automodule:: radiosim.core.jones.chain
   :members:
   :show-inheritance:

.. automodule:: radiosim.core.jones.directions
   :members:
   :show-inheritance:

.. automodule:: radiosim.core.jones.evaluate
   :members:
   :show-inheritance:

Implemented terms
-----------------

.. automodule:: radiosim.core.jones.geometric
   :members:
   :show-inheritance:

.. automodule:: radiosim.core.jones.receptor
   :members:
   :show-inheritance:

Planned terms
-------------

Each module below documents one term's mathematics, its units and signs, its
citation, and the slice that implements it. Constructing one of these classes
is allowed; evaluating one raises ``NotImplementedError``.

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

.. automodule:: radiosim.core.jones.delay
   :members:
   :show-inheritance:

.. automodule:: radiosim.core.jones.crosshand
   :members:
   :show-inheritance:

.. automodule:: radiosim.core.jones.baseline_errors
   :members:
   :show-inheritance:

Beam primitives
---------------

.. automodule:: radiosim.core.jones.beam.analytic
   :members:
   :show-inheritance:

The analytic module exposes formula functions only. It is not a configuration,
assignment, composition, plotting, or runtime ownership surface. Canonical
beam models and evaluation are documented in :doc:`core` and
:doc:`../user_guide/beam_models`.
