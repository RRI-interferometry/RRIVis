Observability
=============

``radiosim.core.observability`` answers "what is up, when, and how well is the
beam pointed at it" for a resolved instrument and observation. It is a
``Simulator`` helper (``Simulator.plan_observability`` /
``Simulator.plot_observability``) and is independent of simulation-result
rendering: nothing in this subpackage runs the solver or reads a
``SimulationResult``.

The subpackage re-exports its whole public surface -- the window and option
types, the planner and its plan, the beam-projection and light-curve helpers,
and the typed errors -- so each symbol is documented once here rather than
again under its defining module.

.. automodule:: radiosim.core.observability
   :members:
   :undoc-members:
   :show-inheritance:
