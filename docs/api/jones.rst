Jones Matrix Framework
======================

The ``radiosim.core.jones`` module implements the complete Jones matrix framework
for radio interferometry, providing 8 Jones terms for comprehensive instrumental
and propagation modeling.

.. automodule:: radiosim.core.jones
   :members:
   :undoc-members:
   :show-inheritance:

Jones Terms
-----------

The RIME (Radio Interferometer Measurement Equation) is modeled as:

.. math::

   V_{ij} = \sum_s J_i \cdot C_s \cdot J_j^H

where :math:`J` is the Jones chain and :math:`C_s` is the source coherency matrix.

The full Jones chain is:

.. math::

   J = K \cdot E \cdot Z \cdot T \cdot P \cdot G \cdot B \cdot D

Base Classes
------------

.. automodule:: radiosim.core.jones.base
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: radiosim.core.jones.chain
   :members:
   :undoc-members:
   :show-inheritance:

K - Geometric Delay
-------------------

Phase delay due to geometric path length.

.. automodule:: radiosim.core.jones.geometric
   :members:
   :undoc-members:
   :show-inheritance:

E - Primary Beam
----------------

Direction-dependent antenna response.

.. automodule:: radiosim.core.jones.beam
   :members:
   :undoc-members:
   :show-inheritance:

Z - Ionosphere
--------------

Ionospheric effects including Faraday rotation.

.. automodule:: radiosim.core.jones.ionosphere
   :members:
   :undoc-members:
   :show-inheritance:

T - Troposphere
---------------

Tropospheric phase and amplitude effects.

.. automodule:: radiosim.core.jones.troposphere
   :members:
   :undoc-members:
   :show-inheritance:

P - Parallactic Angle
---------------------

Feed rotation with parallactic angle.

.. automodule:: radiosim.core.jones.parallactic
   :members:
   :undoc-members:
   :show-inheritance:

G - Gain
--------

Electronic gain errors.

.. automodule:: radiosim.core.jones.gain
   :members:
   :undoc-members:
   :show-inheritance:

B - Bandpass
------------

Frequency-dependent bandpass response.

.. automodule:: radiosim.core.jones.bandpass
   :members:
   :undoc-members:
   :show-inheritance:

D - Polarization Leakage
------------------------

Polarization leakage (D-terms).

.. automodule:: radiosim.core.jones.polarization_leakage
   :members:
   :undoc-members:
   :show-inheritance:

Example Usage
-------------

Creating a Jones chain:

.. code-block:: python

   from radiosim.core.jones import (
       JonesChain,
       GeometricDelayJones,
       BeamJones,
       IonosphereJones,
       GainJones,
       BandpassJones,
   )

   # Full instrumental chain
   jones = JonesChain([
       GeometricDelayJones(),                    # K
       BeamJones(beam_type="gaussian"),          # E
       IonosphereJones(tec=10.0),                # Z
       GainJones(amplitude_std=0.01),            # G
       BandpassJones(ripple_amplitude=0.02),     # B
   ])

   # Use in simulation
   from radiosim import Simulator

   sim = Simulator()
   sim.setup(
       antenna_layout="antennas.txt",
       jones_chain=jones,
   )
   results = sim.run()

Individual Jones term:

.. code-block:: python

   from radiosim.core.jones import GainJones
   import numpy as np

   # Create gain Jones term with 1% amplitude errors
   gain = GainJones(amplitude_std=0.01, phase_std=np.deg2rad(1))

   # Compute Jones matrix for an antenna
   J = gain.compute(antenna_idx=0, frequency=150e6)
   print(J.shape)  # (2, 2) complex
