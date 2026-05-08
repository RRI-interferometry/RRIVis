Simulator API
=============

The high-level ``Simulator`` class provides a simple interface for running
radio interferometry visibility simulations.

.. automodule:: radiosim.api.simulator
   :members:
   :undoc-members:
   :show-inheritance:

Simulator Class
---------------

.. autoclass:: radiosim.api.simulator.Simulator
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

Example Usage
-------------

Basic simulation:

.. code-block:: python

   from radiosim import Simulator

   sim = Simulator.from_config("config.yaml")
   sim.setup()
   results = sim.run(progress=True)
   sim.save("output.h5")

With Jones matrices:

.. code-block:: python

   from radiosim import Simulator
   from radiosim.core.jones import JonesChain, BeamJones, GainJones

   jones = JonesChain([BeamJones(), GainJones()])

   sim = Simulator()
   sim.setup(antenna_layout="antennas.txt", jones_chain=jones)
   results = sim.run()
