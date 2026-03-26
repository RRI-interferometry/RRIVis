Jones Matrix Framework
======================

RRIvis implements a complete Jones matrix framework for modeling instrumental
and propagation effects in radio interferometry.

The RIME Equation
-----------------

The Radio Interferometer Measurement Equation (RIME) is:

.. math::

   V_{ij} = \sum_s J_i(\vec{s}) \cdot C_s \cdot J_j^H(\vec{s})

where:

- :math:`V_{ij}` is the 2x2 visibility matrix for baseline :math:`ij`
- :math:`J_i` is the Jones matrix chain for antenna :math:`i`
- :math:`C_s` is the source coherency matrix
- :math:`\vec{s}` is the source direction

Full Jones Chain
----------------

The complete Jones chain is:

.. math::

   J = K \cdot E \cdot Z \cdot T \cdot P \cdot G \cdot B \cdot D

.. list-table::
   :header-rows: 1

   * - Term
     - Name
     - Effect
   * - K
     - Geometric Delay
     - Phase from path length
   * - E
     - Primary Beam
     - Direction-dependent gain
   * - Z
     - Ionosphere
     - Faraday rotation, phase
   * - T
     - Troposphere
     - Amplitude, phase delay
   * - P
     - Parallactic Angle
     - Feed rotation
   * - G
     - Gain
     - Electronic gain errors
   * - B
     - Bandpass
     - Frequency response
   * - D
     - Polarization Leakage
     - D-terms

Creating a Jones Chain
----------------------

.. code-block:: python

   from rrivis.core.jones import (
       JonesChain,
       GeometricDelayJones,
       BeamJones,
       IonosphereJones,
       GainJones,
       BandpassJones,
       PolarizationLeakageJones,
   )

   # Create chain with selected terms
   jones = JonesChain([
       GeometricDelayJones(),
       BeamJones(beam_type="gaussian"),
       GainJones(amplitude_std=0.01),
   ])

   # Use in simulation
   from rrivis import Simulator

   sim = Simulator()
   sim.setup(
       antenna_layout="antennas.txt",
       jones_chain=jones,
   )
   results = sim.run()

Individual Jones Terms
----------------------

Geometric Delay (K)
^^^^^^^^^^^^^^^^^^^

Phase delay from geometric path length:

.. code-block:: python

   from rrivis.core.jones import GeometricDelayJones

   K = GeometricDelayJones()
   # Automatically computed from UVW coordinates

Primary Beam (E)
^^^^^^^^^^^^^^^^

Direction-dependent antenna response:

.. code-block:: python

   from rrivis.core.jones import BeamJones

   # Analytic beam
   E = BeamJones(beam_type="gaussian", fwhm_deg=10.0)

   # From FITS file
   E = BeamJones(beam_file="beam.fits")

Ionosphere (Z)
^^^^^^^^^^^^^^

Ionospheric Faraday rotation and phase:

.. code-block:: python

   from rrivis.core.jones import IonosphereJones

   Z = IonosphereJones(
       tec=10.0,  # TEC units
       rotation_measure=1.0,  # rad/m^2
   )

Troposphere (T)
^^^^^^^^^^^^^^^

Tropospheric amplitude and phase effects:

.. code-block:: python

   from rrivis.core.jones import TroposphereJones

   T = TroposphereJones(
       zenith_delay=2.3,  # meters
       scale_height=8000,  # meters
   )

Parallactic Angle (P)
^^^^^^^^^^^^^^^^^^^^^

Feed rotation with parallactic angle:

.. code-block:: python

   from rrivis.core.jones import ParallacticJones

   P = ParallacticJones()
   # Computed from antenna latitude and source position

Gain (G)
^^^^^^^^

Electronic gain errors:

.. code-block:: python

   from rrivis.core.jones import GainJones
   import numpy as np

   G = GainJones(
       amplitude_std=0.01,  # 1% amplitude error
       phase_std=np.deg2rad(1),  # 1 degree phase error
       time_scale=3600,  # seconds
   )

Bandpass (B)
^^^^^^^^^^^^

Frequency-dependent response:

.. code-block:: python

   from rrivis.core.jones import BandpassJones

   B = BandpassJones(
       ripple_amplitude=0.02,  # 2% ripple
       ripple_period_mhz=10.0,
   )

Polarization Leakage (D)
^^^^^^^^^^^^^^^^^^^^^^^^

Polarization leakage (D-terms):

.. code-block:: python

   from rrivis.core.jones import PolarizationLeakageJones

   D = PolarizationLeakageJones(
       d_term_amplitude=0.01,  # 1% leakage
   )

Custom Jones Terms
------------------

Create custom Jones terms by subclassing:

.. code-block:: python

   from rrivis.core.jones.base import JonesTerm
   import numpy as np

   class MyJonesTerm(JonesTerm):
       def __init__(self, my_parameter: float):
           self.my_parameter = my_parameter

       def compute(
           self,
           antenna_idx: int,
           frequency: float,
           time: float = None,
           direction: tuple = None,
           **kwargs
       ) -> np.ndarray:
           # Return 2x2 complex Jones matrix
           J = np.eye(2, dtype=np.complex128)
           J *= np.exp(1j * self.my_parameter)
           return J

   # Use in chain
   jones = JonesChain([MyJonesTerm(my_parameter=0.1)])

21cm Cosmology Considerations
-----------------------------

For 21cm cosmology, systematic control is critical:

- **Foregrounds are 10^4x stronger** than the EoR signal
- **Mode mixing** from chromatic beams causes foreground leakage
- **End-to-end forward modeling** is essential

Recommended Jones chain for 21cm:

.. code-block:: python

   jones = JonesChain([
       GeometricDelayJones(),
       BeamJones(beam_file="measured_beam.fits"),  # Use measured beams
       IonosphereJones(tec=10.0),
       GainJones(amplitude_std=0.001),  # Tight gain control
       BandpassJones(ripple_amplitude=0.001),  # Smooth bandpass
   ])
