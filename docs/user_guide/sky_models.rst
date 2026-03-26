Sky Models
==========

RRIvis supports multiple sky models for visibility simulations.

Available Sky Models
--------------------

GLEAM Catalog
^^^^^^^^^^^^^

The GaLactic and Extragalactic All-sky MWA (GLEAM) catalog provides
point source positions and fluxes.

.. code-block:: python

   from rrivis import Simulator

   sim = Simulator()
   sim.setup(
       antenna_layout="antennas.txt",
       sky_model="gleam",
       flux_limit=1.0,  # Jy
   )

Configuration:

.. code-block:: yaml

   sky_model:
     gleam:
       use_gleam: true
       flux_limit: 1.0
       gleam_catalogue: "VIII/100/gleamegc"

Global Sky Model (GSM)
^^^^^^^^^^^^^^^^^^^^^^

The Global Sky Model provides diffuse emission.

.. code-block:: python

   sim.setup(
       antenna_layout="antennas.txt",
       sky_model="gsm",
       nside=64,  # HEALPix resolution
   )

Configuration:

.. code-block:: yaml

   sky_model:
     gsm_healpix:
       use_gsm: true
       nside: 64
       flux_limit: 0.1

Combined GSM + GLEAM
^^^^^^^^^^^^^^^^^^^^

Combine diffuse and point source emission:

.. code-block:: yaml

   sky_model:
     gsm+gleam_healpix:
       use_gsm_gleam: true
       nside: 64
       flux_limit: 1.0

Test Sources
^^^^^^^^^^^^

Simple point sources for testing:

.. code-block:: python

   sim.setup(
       antenna_layout="antennas.txt",
       sky_model="test",
       num_sources=100,
       flux_limit=50.0,
   )

Configuration:

.. code-block:: yaml

   sky_model:
     test_sources:
       use_test_sources: true
       num_sources: 100
       flux_limit: 50.0

Custom Point Sources
--------------------

Define custom point sources programmatically:

.. code-block:: python

   from rrivis.core.source import PointSource

   sources = [
       PointSource(
           ra=0.0,          # degrees
           dec=-30.0,       # degrees
           flux=10.0,       # Jy
           spectral_index=-0.7,
       ),
       PointSource(
           ra=15.0,
           dec=-30.0,
           flux=5.0,
           spectral_index=-0.8,
       ),
   ]

   sim.setup(
       antenna_layout="antennas.txt",
       sky_model=sources,
   )

Polarized Sources
^^^^^^^^^^^^^^^^^

Include polarization (Stokes I, Q, U, V):

.. code-block:: python

   from rrivis.core.source import PointSource

   source = PointSource(
       ra=0.0,
       dec=-30.0,
       stokes_i=10.0,    # Total intensity (Jy)
       stokes_q=1.0,     # Linear polarization
       stokes_u=0.5,
       stokes_v=0.0,     # Circular polarization
       spectral_index=-0.7,
   )

HEALPix Sky Maps
----------------

Load sky models in HEALPix format:

.. code-block:: python

   import healpy as hp

   # Load HEALPix map
   sky_map = hp.read_map("sky_model.fits")

   sim.setup(
       antenna_layout="antennas.txt",
       sky_model=sky_map,
       nside=64,
   )

Flux Limits
-----------

Control simulation speed with flux limits:

.. code-block:: python

   # Only sources brighter than 1 Jy
   sim.setup(sky_model="gleam", flux_limit=1.0)

   # Include fainter sources (slower)
   sim.setup(sky_model="gleam", flux_limit=0.1)

Spectral Index
--------------

Source fluxes are scaled with frequency using spectral index:

.. math::

   S(\nu) = S_0 \left(\frac{\nu}{\nu_0}\right)^{\alpha}

where :math:`\alpha` is the spectral index (typically -0.7 to -0.8).

Sky Model Selection Guide
-------------------------

.. list-table::
   :header-rows: 1

   * - Use Case
     - Recommended Model
   * - Quick testing
     - test_sources
   * - Point source calibration
     - gleam
   * - Diffuse emission
     - gsm
   * - Full sky simulation
     - gsm+gleam
   * - Custom science
     - Custom PointSource list

Performance Considerations
--------------------------

- **Number of sources**: More sources = longer runtime
- **HEALPix resolution**: Higher nside = more pixels = slower
- **Flux limit**: Higher limit = fewer sources = faster

For large simulations, use GPU backends:

.. code-block:: python

   sim = Simulator(backend="jax")
   sim.setup(sky_model="gleam", flux_limit=0.1)
   results = sim.run()  # GPU accelerated
