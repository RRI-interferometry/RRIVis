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

   from rrivis.core.sky import SkyModel
   from rrivis.core.precision import PrecisionConfig

   precision = PrecisionConfig.standard()
   sky = SkyModel.from_catalog("gleam", flux_limit=1.0, precision=precision)

Configuration:

.. code-block:: yaml

   sky_model:
     flux_unit: "Jy"
     gleam:
       use_gleam: true
       flux_limit: 1.0
       gleam_catalogue: "VIII/100/gleamegc"

Global Sky Model (GSM)
^^^^^^^^^^^^^^^^^^^^^^

The Global Sky Model provides diffuse emission.

.. code-block:: python

   import numpy as np
   from rrivis.core.sky import SkyModel

   frequencies = np.linspace(100e6, 200e6, 11)  # Hz
   sky = SkyModel.from_catalog("diffuse_sky", model="gsm2008", frequencies=frequencies, nside=64)

Configuration:

.. code-block:: yaml

   sky_model:
     flux_unit: "Jy"
     gsm_healpix:
       use_gsm: true
       nside: 64

Combined Models
^^^^^^^^^^^^^^^

Combine diffuse and point source emission:

.. code-block:: python

   from rrivis.core.sky import SkyModel
   from rrivis.core.precision import PrecisionConfig
   import numpy as np

   precision = PrecisionConfig.standard()
   frequencies = np.linspace(100e6, 200e6, 11)

   gleam = SkyModel.from_catalog("gleam", flux_limit=1.0, precision=precision)
   gsm = SkyModel.from_catalog("diffuse_sky", model="gsm2008", frequencies=frequencies, nside=64, precision=precision)
   combined = SkyModel.combine(
       [gleam, gsm], representation="healpix_map", precision=precision,
   )

Test Sources
^^^^^^^^^^^^

Simple point sources for testing:

.. code-block:: python

   from rrivis.core.sky import SkyModel
   from rrivis.core.precision import PrecisionConfig

   precision = PrecisionConfig.standard()
   sky = SkyModel.from_test_sources(
       num_sources=100,
       flux_range=(2.0, 8.0),
       dec_deg=-30.0,
       spectral_index=-0.8,
       precision=precision,
   )

Configuration:

.. code-block:: yaml

   sky_model:
     flux_unit: "Jy"
     test_sources:
       use_test_sources: true
       num_sources: 100

Custom Point Sources
--------------------

Define custom point sources programmatically using ``SkyModel.from_arrays()``:

.. code-block:: python

   import numpy as np
   from rrivis.core.sky import SkyModel
   from rrivis.core.precision import PrecisionConfig

   precision = PrecisionConfig.standard()
   sky = SkyModel.from_arrays(
       ra_rad=np.deg2rad([0.0, 15.0]),
       dec_rad=np.deg2rad([-30.0, -30.0]),
       flux=np.array([10.0, 5.0]),
       spectral_index=np.array([-0.7, -0.8]),
       precision=precision,
   )

Or using ``from_point_sources()`` with dictionaries:

.. code-block:: python

   from astropy.coordinates import SkyCoord

   sources = [
       {
           "coords": SkyCoord(ra=0.0, dec=-30.0, unit="deg"),
           "flux": 10.0,
           "spectral_index": -0.7,
       },
       {
           "coords": SkyCoord(ra=15.0, dec=-30.0, unit="deg"),
           "flux": 5.0,
           "spectral_index": -0.8,
       },
   ]
   sky = SkyModel.from_point_sources(sources, precision=precision)

Polarized Sources
^^^^^^^^^^^^^^^^^

Include polarization (Stokes I, Q, U, V):

.. code-block:: python

   sky = SkyModel.from_arrays(
       ra_rad=np.deg2rad([0.0]),
       dec_rad=np.deg2rad([-30.0]),
       flux=np.array([10.0]),
       spectral_index=np.array([-0.7]),
       stokes_q=np.array([1.0]),
       stokes_u=np.array([0.5]),
       stokes_v=np.array([0.0]),
       precision=precision,
   )

HEALPix Sky Maps
----------------

Convert point sources to multi-frequency HEALPix maps:

.. code-block:: python

   import numpy as np

   frequencies = np.linspace(100e6, 200e6, 11)
   sky_healpix = sky.with_healpix_maps(nside=64, frequencies=frequencies)

Flux Limits
-----------

Control simulation speed with flux limits:

.. code-block:: python

   # Only sources brighter than 1 Jy
   sky = SkyModel.from_catalog("gleam", flux_limit=1.0, precision=precision)

   # Include fainter sources (slower)
   sky = SkyModel.from_catalog("gleam", flux_limit=0.1, precision=precision)

Spectral Index
--------------

Source fluxes are scaled with frequency using spectral index:

.. math::

   S(\nu) = S_0 \left(\frac{\nu}{\nu_0}\right)^{\alpha}

where :math:`\alpha` is the spectral index (typically -0.7 to -0.8) and
:math:`\nu_0` is the catalog reference frequency (stored per model).

Sky Model Selection Guide
-------------------------

.. list-table::
   :header-rows: 1

   * - Use Case
     - Recommended Model
   * - Quick testing
     - ``from_test_sources()``
   * - Point source calibration
     - ``from_catalog("gleam")``
   * - Diffuse emission
     - ``from_catalog("diffuse_sky", model="gsm2008")``
   * - Full sky simulation
     - ``SkyModel.combine([gleam, gsm])``
   * - Custom science
     - ``from_arrays()`` or ``from_point_sources()``

Performance Considerations
--------------------------

- **Number of sources**: More sources = longer runtime
- **HEALPix resolution**: Higher nside = more pixels = slower
- **Flux limit**: Higher limit = fewer sources = faster

For large simulations, use GPU backends:

.. code-block:: python

   sim = Simulator(
       antenna_layout="antennas.txt",
       sky_model="gleam",
       backend="jax",
   )
   sim.setup()
   results = sim.run()  # GPU accelerated
