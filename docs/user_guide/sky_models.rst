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

   from rrivis.core.precision import PrecisionConfig
   from rrivis.core.sky.loaders import load_gleam

   precision = PrecisionConfig.standard()
   sky = load_gleam(flux_limit=1.0, max_rows=10000, precision=precision)

Configuration:

.. code-block:: yaml

   sky_model:
     flux_unit: "Jy"
     sources:
       - kind: gleam
         flux_limit: 1.0
         max_rows: 10000
         catalog: gleam_egc

Global Sky Model (GSM)
^^^^^^^^^^^^^^^^^^^^^^

The Global Sky Model provides diffuse emission.

.. code-block:: python

   import numpy as np
   from rrivis.core.precision import PrecisionConfig
   from rrivis.core.sky.loaders import load_diffuse_sky

   precision = PrecisionConfig.standard()
   frequencies = np.linspace(100e6, 200e6, 11)  # Hz
   sky = load_diffuse_sky(
       model="gsm2008",
       frequencies=frequencies,
       nside=64,
       precision=precision,
   )

Configuration:

.. code-block:: yaml

   sky_model:
     flux_unit: "Jy"
     sources:
       - kind: gsm2008
         nside: 64

Alias forms resolve through the loader registry. For example,
``kind: gsm2008`` becomes ``diffuse_sky`` with ``model: gsm2008``,
and explicit fields still win:

.. code-block:: yaml

   sky_model:
     sources:
       - kind: gsm2016
         nside: 128
       - kind: gsm2016
         model: haslam
         nside: 64

Combined Models
^^^^^^^^^^^^^^^

Combine diffuse and point source emission:

.. code-block:: python

   from rrivis.core.sky import combine_models
   from rrivis.core.sky.loaders import load_diffuse_sky, load_gleam
   from rrivis.core.precision import PrecisionConfig
   import numpy as np

   precision = PrecisionConfig.standard()
   frequencies = np.linspace(100e6, 200e6, 11)

   gleam = load_gleam(flux_limit=1.0, max_rows=10000, precision=precision)
   gsm = load_diffuse_sky(
       model="gsm2008",
       frequencies=frequencies,
       nside=64,
       precision=precision,
   )
   combined = combine_models(
       [gleam, gsm],
       representation="healpix_map",
       nside=64,
       frequencies=frequencies,
       mixed_model_policy="warn",
       precision=precision,
   )

Mixing point catalogs with diffuse HEALPix models is blocked by default
because it can double-count bright sources. Set
``mixed_model_policy="warn"`` or ``"allow"`` only when that tradeoff is
intentional.

Test Sources
^^^^^^^^^^^^

Simple point sources for testing:

.. code-block:: python

   from rrivis.core.sky import create_test_sources
   from rrivis.core.precision import PrecisionConfig

   precision = PrecisionConfig.standard()
   sky = create_test_sources(
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
     sources:
       - kind: test_sources
         num_sources: 100

Custom Point Sources
--------------------

Define custom point sources programmatically using ``create_from_arrays()``:

.. code-block:: python

   import numpy as np
   from rrivis.core.sky import create_from_arrays
   from rrivis.core.precision import PrecisionConfig

   precision = PrecisionConfig.standard()
   sky = create_from_arrays(
       ra_rad=np.deg2rad([0.0, 15.0]),
       dec_rad=np.deg2rad([-30.0, -30.0]),
       flux=np.array([10.0, 5.0]),
       spectral_index=np.array([-0.7, -0.8]),
       precision=precision,
   )

RRIvis keeps custom catalogs in columnar arrays rather than per-source
dictionaries, so ``create_from_arrays()`` is the direct construction API.

Polarized Sources
^^^^^^^^^^^^^^^^^

Include polarization (Stokes I, Q, U, V):

.. code-block:: python

   sky = create_from_arrays(
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
   from rrivis.core.sky import materialize_healpix_model

   frequencies = np.linspace(100e6, 200e6, 11)
   sky_healpix = materialize_healpix_model(
       sky,
       nside=64,
       frequencies=frequencies,
   )

Convert a HEALPix-only model back to a point-source view explicitly:

.. code-block:: python

   from rrivis.core.sky import materialize_point_sources_model

   point_view = materialize_point_sources_model(
       sky_healpix,
       frequency=100e6,
       lossy=True,
   )

Lossy HEALPix-to-point conversion is never implicit. Simulator configs
must opt in with ``visibility.allow_lossy_point_materialization: true``
before requesting ``visibility.sky_representation: point_sources`` for a
HEALPix-only model.

Public Sky API
--------------

The root ``rrivis.core.sky`` package is intentionally small. The stable
entry points are:

- constructors: ``create_empty()``, ``create_from_arrays()``, ``create_test_sources()``
- transforms: ``combine_models()``, ``materialize_healpix_model()``,
  ``materialize_point_sources_model()``, ``with_memmap_backing()``
- IO: ``load_skyh5()``, ``save_skyh5()``, ``to_pyradiosky()``, ``write_bbs()``
- discovery: ``estimate_healpix_memory()``, ``list_all_models()``,
  ``get_catalog_info()``

Lower-level implementation helpers remain in their defining modules and
are not part of the root public contract.

Flux Limits
-----------

Control simulation speed with flux limits:

.. code-block:: python

   # Only sources brighter than 1 Jy
   sky = load_gleam(flux_limit=1.0, max_rows=10000, precision=precision)

   # Include fainter sources (slower)
   sky = load_gleam(flux_limit=0.1, max_rows=10000, precision=precision)

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
     - ``create_test_sources()``
   * - Point source calibration
     - ``load_gleam()``
   * - Diffuse emission
     - ``load_diffuse_sky(model="gsm2008")``
   * - Full sky simulation
     - ``combine_models([gleam, gsm])``
   * - Custom science
     - ``create_from_arrays()``

Performance Considerations
--------------------------

- **Number of sources**: More sources = longer runtime
- **HEALPix resolution**: Higher nside = more pixels = slower
- **Flux limit**: Higher limit = fewer sources = faster

For large point-source simulations, use GPU backends:

.. code-block:: python

   sim = Simulator(
       config={
           "antenna_layout": {
               "antenna_positions_file": "antennas.txt",
               "antenna_file_format": "rrivis",
               "all_antenna_diameter": 14.0,
           },
           "obs_frequency": {
               "frequencies_hz": [100e6, 150e6],
               "frequency_unit": "MHz",
           },
           "sky_model": {"sources": [{"kind": "gleam"}]},
           "visibility": {"sky_representation": "point_sources"},
       },
       backend="jax",
   )
   sim.setup()
   results = sim.run()  # GPU accelerated for point-source visibility

HEALPix direct visibility currently uses a NumPy CPU path. If a GPU backend is
configured with ``visibility.sky_representation: healpix_map``, RRIVis warns and
runs the HEALPix calculation on CPU.
