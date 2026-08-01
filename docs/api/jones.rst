Jones Matrix API
================

``radiosim.core.jones`` exports nineteen names: three base classes
(:class:`~radiosim.core.jones.base.JonesTerm`,
:class:`~radiosim.core.jones.chain.JonesChain` and
:class:`~radiosim.core.jones.baseline_errors.JonesBaselineTerm`), thirteen
concrete terms, and three non-class exports —
:class:`~radiosim.core.jones.directions.DirectionBatch`,
:func:`~radiosim.core.jones.evaluate.evaluate_antenna_jones` and
:func:`~radiosim.core.jones.geometric.geometric_phase`.

Every exported term declares ``term_status``, and since Tier 7 of the
remediation programme every one of them reads ``"implemented"``: the term's
physics exists, the solver evaluates it, and it is covered by analytic
invariant tests, a backend-parity case, and an effect-changes-visibility case.
There is no ``"planned"`` term left. Both evaluation contracts —
``JonesTerm.compute_jones_batch`` and
``JonesBaselineTerm.compute_baseline_factor`` — are ``@abstractmethod``, so a
class that does not implement its physics cannot be constructed at all, and
nothing in this package multiplies by the identity in silence.

Twenty-six classes that returned the 2x2 identity for every input, for effects
RadioSim does not plan to model — turbulent and GPS ionospheres, w-projection,
element beams and array factors, fringe fitting, Mueller and IXR leakage
variants, and the rest — were removed before v1.0, and
``CrosshandPhaseJones`` was renamed ``CrosshandJones``.
:doc:`../migration_guide` names the replacement for each.

See :doc:`../user_guide/jones_terms` for each term's mathematics, units,
signs, citation, and configuration, and
:doc:`../user_guide/jones_matrices` for the receptor mathematics and the
canonical chain order.

Base classes, the chain, and the shared evaluator
-------------------------------------------------

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

Direction-dependent terms
-------------------------

``K`` is a module-level function rather than a class, because the geometric
phase is per-baseline and was never a chain term.

.. automodule:: radiosim.core.jones.geometric
   :members:
   :show-inheritance:

.. automodule:: radiosim.core.jones.parallactic
   :members:
   :show-inheritance:

.. automodule:: radiosim.core.jones.ionosphere
   :members:
   :show-inheritance:

.. automodule:: radiosim.core.jones.troposphere
   :members:
   :show-inheritance:

The primary beam ``E`` is the canonical ``BeamSystem``, reached through a
private solver-owned adapter rather than an exported Jones class; it is
documented in :doc:`core` and :doc:`../user_guide/beam_models`.

Direction-independent terms
---------------------------

.. automodule:: radiosim.core.jones.receptor
   :members:
   :show-inheritance:

.. automodule:: radiosim.core.jones.polarization_leakage
   :members:
   :show-inheritance:

.. automodule:: radiosim.core.jones.crosshand
   :members:
   :show-inheritance:

.. automodule:: radiosim.core.jones.delay
   :members:
   :show-inheritance:

.. automodule:: radiosim.core.jones.bandpass
   :members:
   :show-inheritance:

.. automodule:: radiosim.core.jones.gain
   :members:
   :show-inheritance:

Baseline-dependent terms
------------------------

``M`` and ``Q`` are not ``JonesTerm`` at all. They descend from
``JonesBaselineTerm``, apply by Hadamard product to finished visibilities
rather than by matrix multiplication, and ``JonesChain.add_term`` rejects them
by type.

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
