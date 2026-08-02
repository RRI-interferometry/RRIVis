Solver Strategies
=================

``radiosim.simulator`` holds the swappable solver strategies.
``VisibilitySimulator`` is the abstract contract; ``RIMESimulator`` is the one
registered implementation, a direct RIME summation with cost
``O(N_sources x N_baselines x N_frequencies)``.

``execution.simulator`` accepts exactly the keys of the simulator registry,
which today is ``rime`` alone. A spherical-harmonic or m-mode solver would be a
new registration, not a value on a removed configuration field.

Registry
--------

.. automodule:: radiosim.simulator
   :no-members:
   :no-special-members:

.. autofunction:: radiosim.simulator.get_simulator

.. autofunction:: radiosim.simulator.list_simulators

Solver contract
---------------

.. automodule:: radiosim.simulator.base
   :members:
   :undoc-members:
   :show-inheritance:

Direct RIME summation
---------------------

.. automodule:: radiosim.simulator.rime
   :members:
   :undoc-members:
   :show-inheritance:
