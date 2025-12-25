Beam Models
===========

RRIvis supports both analytic and measured beam patterns.

Beam Modes
----------

Analytic (Default)
^^^^^^^^^^^^^^^^^^

Use mathematical beam models:

.. code-block:: yaml

   beams:
     beam_mode: "analytic"
     all_beam_response: "gaussian"

Available analytic beams:

- ``gaussian`` - Gaussian beam pattern
- ``airy`` - Airy disk pattern
- ``cosine`` - Cosine pattern
- ``exponential`` - Exponential pattern

Shared Beam
^^^^^^^^^^^

All antennas use the same FITS beam file:

.. code-block:: yaml

   beams:
     beam_mode: "shared"
     beam_file: "/path/to/beam.fits"
     beam_za_max_deg: 90.0

Per-Antenna Beams
^^^^^^^^^^^^^^^^^

Different beams for different antennas:

From layout file:

.. code-block:: yaml

   beams:
     beam_mode: "per_antenna"
     beam_assignment: "from_layout"

The antenna file should include a ``BeamID`` column:

.. code-block:: text

   Name    Number  BeamID       E      N      U
   ANT001  0       beam_dish    0.0    0.0    0.0
   ANT002  1       beam_dipole  10.0   0.0    0.0

From configuration:

.. code-block:: yaml

   beams:
     beam_mode: "per_antenna"
     beam_assignment: "from_config"
     antenna_beam_map:
       "ANT001": "/path/to/beam1.fits"
       "ANT002": "/path/to/beam2.fits"

Analytic Beam Parameters
------------------------

Gaussian Beam
^^^^^^^^^^^^^

.. code-block:: python

   from rrivis.core.beams import gaussian_beam

   # Beam response at zenith angle theta
   response = gaussian_beam(theta, fwhm_rad)

The Gaussian beam pattern is:

.. math::

   A(\theta) = \exp\left(-\frac{\theta^2}{2\sigma^2}\right)

where :math:`\sigma = \text{FWHM} / (2\sqrt{2\ln 2})`.

HPBW Calculation
^^^^^^^^^^^^^^^^

Half Power Beam Width depends on antenna type:

.. math::

   \text{HPBW} = k \cdot \frac{\lambda}{D}

where:

- :math:`k` is a coefficient depending on antenna type
- :math:`\lambda` is wavelength
- :math:`D` is antenna diameter

.. list-table::
   :header-rows: 1

   * - Antenna Type
     - k value
   * - Parabolic (uniform)
     - 1.02
   * - Parabolic (cosine taper)
     - 1.10
   * - Parabolic (Gaussian taper)
     - 1.18
   * - Spherical reflector
     - 1.05
   * - Phased array
     - 1.10

FITS Beam Files
---------------

Requirements
^^^^^^^^^^^^

Beam FITS files must:

- Be in pyuvdata UVBeam format
- Contain E-field (not power) beam data
- Cover the observation's zenith angle range
- Cover the observation's frequency range

Configuration
^^^^^^^^^^^^^

.. code-block:: yaml

   beams:
     beam_mode: "shared"
     beam_file: "/path/to/beam.fits"
     beam_za_max_deg: 90.0      # Maximum zenith angle
     beam_za_buffer_deg: 5.0    # Buffer around observation range
     beam_freq_buffer_hz: 1e6   # Frequency buffer (1 MHz)

Loading Beam Files
^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from rrivis.core.beam_file import load_beam

   beam = load_beam(
       "beam.fits",
       za_max_deg=90.0,
       freq_buffer_hz=1e6,
   )

   # Interpolate beam at specific direction/frequency
   response = beam.interpolate(
       zenith_angle=30.0,  # degrees
       azimuth=45.0,       # degrees
       frequency=150e6,    # Hz
   )

Polarization
------------

Full polarization beams use 2x2 Jones matrices:

.. code-block:: python

   # E-field beam returns Jones matrix
   J = beam.jones_matrix(theta, phi, frequency)
   # J.shape = (2, 2), complex

The visibility equation with polarized beams:

.. math::

   V_{ij} = E_i \cdot C \cdot E_j^H

where:

- :math:`E_i` is the beam Jones matrix for antenna i
- :math:`C` is the source coherency matrix

Beam Solid Angle
----------------

Calculate beam solid angle using HEALPix:

.. code-block:: python

   from rrivis.core.beams import beam_solid_angle

   omega = beam_solid_angle(beam_pattern, nside=256)
   print(f"Beam solid angle: {omega} sr")

Using Beams in Simulation
-------------------------

With Simulator class:

.. code-block:: python

   from rrivis import Simulator

   sim = Simulator()
   sim.setup(
       antenna_layout="antennas.txt",
       beam_mode="shared",
       beam_file="beam.fits",
   )

With Jones chain:

.. code-block:: python

   from rrivis.core.jones import BeamJones, JonesChain

   beam = BeamJones(
       beam_type="gaussian",
       fwhm_deg=10.0,
   )

   # Or from FITS
   beam = BeamJones(beam_file="beam.fits")

   jones = JonesChain([beam])

Performance Tips
----------------

- Use analytic beams for quick testing
- Pre-compute beam interpolation tables for large simulations
- Use lower ``beam_za_max_deg`` if sources are near zenith
- GPU backends accelerate beam calculations significantly
