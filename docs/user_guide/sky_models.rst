Sky Models
==========

RadioSim supports multiple sky models for visibility simulations.

Available Sky Models
--------------------

GLEAM Catalog
^^^^^^^^^^^^^

The GaLactic and Extragalactic All-sky MWA (GLEAM) catalog provides
point source positions and fluxes.

.. code-block:: python

   from radiosim.core.precision import PrecisionConfig
   from radiosim.core.sky.loaders import load_gleam

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
   from radiosim.core.precision import PrecisionConfig
   from radiosim.core.sky.loaders import load_diffuse_sky

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
         options:
           nside: 128
       - kind: gsm2016
         options:
           model: haslam
           nside: 64

Combined Models
^^^^^^^^^^^^^^^

Combine diffuse and point source emission:

.. code-block:: python

   from radiosim.core.sky import prepare_sky_model
   from radiosim.core.sky.loaders import load_diffuse_sky, load_gleam
   from radiosim.core.precision import PrecisionConfig
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
   combined = prepare_sky_model(
       [gleam, gsm],
       representation="healpix_map",
       nside=64,
       frequencies=frequencies,
       mixed_model_policy="warn",
       precision=precision,
   )

Mixing point catalogs with diffuse HEALPix models is blocked by default
because it can double-count bright sources. If disjointness was verified
out of band, pass ``assume_disjoint=True`` (or ``sky_model.assume_disjoint:
true`` in YAML) to skip only the double-counting rules while keeping
monopole checks. Set ``mixed_model_policy="warn"`` or ``"allow"`` only
when you need the broader override that also relaxes UNKNOWN monopole
escalation.

Test Sources
^^^^^^^^^^^^

Simple point sources for testing:

.. code-block:: python

   from radiosim.core.sky import create_test_sources
   from radiosim.core.precision import PrecisionConfig

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

Extragalactic Point Sources (Mittal et al. 2024)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A statistical extragalactic foreground population following Mittal,
Kulkarni, Anstey & de Lera Acedo 2024 (MNRAS 534, 1317; the model behind
the ``epspy`` package — cite that paper when you use this loader): source
counts drawn from a validated ``dN/dS`` preset (Gervasi et al. 2008 by
default; Mandal et al. 2021 and Intema et al. 2017 also ship), unclipped
Gaussian spectral indices (default ``(-0.681, 0.5)``, the paper's
``beta ~ N(2.681, 0.5)`` in RadioSim's ``S ∝ nu^alpha`` convention via
``alpha = 2 - beta``), and by default angular clustering from the paper's
power-law 2PACF (Rana & Bagla 2019: ``A=7.8e-3``, ``gamma=0.821``; pass
``clustering_amp=0`` for an isotropic sky):

.. code-block:: python

   import numpy as np
   from radiosim.core.sky import load_extragalactic_point_sources
   from radiosim.core.precision import PrecisionConfig

   precision = PrecisionConfig.standard()

   # Discrete sources for the point-source RIME (exact positions).
   sky = load_extragalactic_point_sources(
       flux_range_jy=(1e-2, 1e-1),
       seed=42,
       precision=precision,
   )

   # Deep populations stream directly into HEALPix brightness maps in
   # bounded memory (no per-source arrays, no max_sources ceiling).
   maps = load_extragalactic_point_sources(
       flux_range_jy=(1e-6, 1e-1),
       representation="healpix_map",
       nside=128,
       frequencies=1e6 * np.arange(50, 201),
       seed=42,
       precision=precision,
   )

Configuration (aliases ``eps`` and ``mittal2024`` also work):

.. code-block:: yaml

   sky_model:
     flux_unit: "Jy"
     sources:
       - kind: extragalactic_point_sources
         options:
           flux_range_jy: [0.01, 0.1]
           clustering_amp: 0.0078
           seed: 42

The realization is reproducible from the seed recorded in the model's
provenance. Note one deliberate difference from the ``epspy`` reference
implementation: fluxes are sampled from the stated ``dN/dS`` with the
correct integration measure, whereas ``epspy``'s log-grid draw omits the
``dS`` cell widths and therefore realizes a fainter ``dN/dS · S⁻¹``
distribution (its documented ~1.3 K mean sky temperature at 150 MHz for
the deep fiducial range corresponds to ~17 K under the stated counts).
Expect this loader to produce the brighter, count-consistent sky; see
``radiosim/core/sky/loaders/extragalactic.py`` for the full deviation
list.

Custom Point Sources
--------------------

Define custom point sources programmatically using ``create_from_arrays()``:

.. code-block:: python

   import numpy as np
   from radiosim.core.sky import create_from_arrays
   from radiosim.core.precision import PrecisionConfig

   precision = PrecisionConfig.standard()
   sky = create_from_arrays(
       ra_rad=np.deg2rad([0.0, 15.0]),
       dec_rad=np.deg2rad([-30.0, -30.0]),
       flux=np.array([10.0, 5.0]),
       spectral_index=np.array([-0.7, -0.8]),
       precision=precision,
   )

RadioSim keeps custom catalogs in columnar arrays rather than per-source
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

A source's ``rotation_measure`` is its **intrinsic**, source-frame Faraday
rotation and is applied here, to the source's own ``(Q, U)``; the *ionospheric*
rotation measure along the line of sight belongs to the ``jones.Z`` term
(:doc:`jones_terms`), is configured separately, and composes with this one
rather than duplicating it.

HEALPix Sky Maps
----------------

Convert point sources to multi-frequency HEALPix maps:

.. code-block:: python

   import numpy as np
   from radiosim.core.sky import materialize_healpix_model

   frequencies = np.linspace(100e6, 200e6, 11)
   sky_healpix = materialize_healpix_model(
       sky,
       nside=64,
       frequencies=frequencies,
   )

Convert a HEALPix-only model back to a point-source view explicitly:

.. code-block:: python

   from radiosim.core.sky import materialize_point_sources_model

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

The root ``radiosim.core.sky`` package is intentionally small. The stable
entry points are:

- constructors: ``create_empty()``, ``create_from_arrays()``, ``create_test_sources()``
- transforms: ``prepare_sky_model()``, ``materialize_healpix_model()``,
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
     - ``prepare_sky_model([gleam, gsm])``
   * - Custom science
     - ``create_from_arrays()``

Performance Considerations
--------------------------

- **Number of sources**: More sources = longer runtime
- **HEALPix resolution**: Higher nside = more pixels = slower
- **Flux limit**: Higher limit = fewer sources = faster

Backend selection is resolved consistently, but it does not establish
end-to-end GPU acceleration for either high-level sky representation. Benchmark
point-source and HEALPix workloads separately, report the actual backend and
device, and compare numerical results with NumPy before making a performance
claim.
