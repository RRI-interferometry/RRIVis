Compute Backends
================

RadioSim resolves a requested backend strategy before runtime setup. NumPy is
the deterministic default. JAX and Dask are optional; ``auto`` is a real
selection strategy, not a synonym for "keep the document value."

Available backends
------------------

.. list-table::
   :header-rows: 1
   :widths: 12 30 12 14 32

   * - Name
     - What actually executes
     - Device
     - Compilation
     - Notes
   * - ``numpy``
     - NumPy on the host
     - CPU
     - none
     - Always available; the reference every other backend is compared against.
   * - ``jax``
     - JAX/XLA
     - runtime default,
       or a strict request
     - one kernel
     - Generic JAX follows its runtime; named ``cpu``, ``gpu``, and ``tpu``
       devices never fall back.
   * - ``dask``
     - NumPy, optionally
       through Dask arrays
     - CPU
     - none
     - Renamed from ``numba`` before v1.0.
   * - ``auto``
     - NumPy on the host
     - CPU
     - none
     - Deterministic and import-free; it never probes JAX or selects Dask.

``execution.backend: numba`` and ``get_backend("numba")`` are removed. The class
behind that name never compiled a kernel: it called the same NumPy operations
the NumPy backend calls, so the name described a capability that did not exist.
It is now ``dask``, and the rename adds no compilation and no acceleration --
see :doc:`../migration_guide`.

``auto`` selection
------------------

``auto`` never imports or probes JAX. It asks only NumPy to honor the requested
precision and never selects Dask. The resulting
``result.provenance.backend.actual_backend`` therefore names NumPy. Accelerator
inventory and JAX capability are explicit discovery operations, not hidden
selection side effects (``PERF-001``).

``float128`` narrows this further: JAX and Dask cannot honor it, so an explicit
``jax`` or ``dask`` request with ``float128`` is rejected during configuration
resolution, before the optional backend is imported. ``auto`` returns NumPy
only when NumPy can honor that precision; otherwise it raises
``BackendNotAvailableError`` rather than downgrading a dtype.

Configuration
-------------

.. code-block:: yaml

   execution:
     backend: numpy  # numpy | jax | dask | auto
     precision:
       preset: standard

For a YAML document, a call-site override is explicit:

.. code-block:: python

   from radiosim import Simulator
   from radiosim.io.config_resolution import SimulationOverrides

   simulator = Simulator.from_yaml(
       "configs/config.yaml",
       overrides=SimulationOverrides(backend="auto"),
   )

Omitting the override preserves the document value. ``None`` is the only
no-override sentinel; ``auto`` never means "keep the document."

Direct backend API
------------------

.. code-block:: python

   from radiosim.backends import get_backend, list_backends

   print(list_backends())
   backend = get_backend("numpy")
   values = backend.asarray([1.0, 2.0, 3.0])
   print(backend.sum(values))

The backend factory constructs an explicit backend or resolves ``auto`` to
NumPy. ``list_backends()`` and ``get_backend_info()`` are explicit discovery
operations and may import and probe JAX; GPU and TPU plugin failures are
isolated so one does not erase the other's truthful availability
(``PERF-001``).

``get_backend("jax")`` delegates device choice to the JAX runtime. Passing
``device="cpu"``, ``"gpu"``, or ``"tpu"`` makes that device a strict
requirement. The direct ``get_backend("gpu")`` and ``get_backend("tpu")``
aliases are strict too. An unavailable or broken runtime raises
``BackendNotAvailableError`` with the runtime failure retained as its cause;
there is no CPU fallback.

Generic :func:`radiosim.utils.device.get_device_resources` reports physical
hardware through platform APIs and vendor tools. It never imports JAX as a
fallback. JAX device discovery belongs to the explicit backend-discovery calls,
which keeps a minimal ``Simulator.setup()`` with ``backend: auto`` free of JAX
initialization (``PERF-001``).

Simulator accelerator capability
--------------------------------

The inherited :attr:`radiosim.simulator.VisibilitySimulator.supports_gpu`
value is ``False``, and :class:`radiosim.simulator.RIMESimulator` states the
same value explicitly. A future simulator may return ``True`` only when an
independently accepted end-to-end accelerator record names that exact
implementation. No such record exists yet (``PERF-001``).

The compilation boundary
------------------------

``ArrayBackend`` exposes two capability members: ``supports_compilation`` and
``compile``. The base implementations are ``False`` and the identity function,
so NumPy and Dask inherit "no compilation" without importing anything. The JAX
backend overrides them with ``True`` and ``jax.jit``.

Exactly **one** kernel is compiled --
:func:`radiosim.core.contraction.baseline_contraction`, the baseline-batched
per-``(time, frequency)`` contraction that turns the per-antenna Jones matrices,
the coherency matrices, the geometric phase, and the Gaussian envelope into one
``(B, 2, 2)`` block. It is the only place in RadioSim that calls
``backend.compile``. The uncompiled function stays the reference implementation
and is what NumPy and Dask always execute.

There is no separate "enable jit" switch. Compilation follows from
``execution.backend: jax`` and nothing else, because a backend that advertised
compilation and then did not compile would be another unfulfilled claim.

Two measured consequences of compiling this kernel, both recorded in the
benchmark records described below:

- **Retracing.** Both solvers mask sources and pixels by ``above_horizon`` at
  every time step, so the kernel's source axis changes size whenever the visible
  sky changes. XLA recompiles per distinct shape. Measured on the reference host,
  the first call at a newly seen source count cost 39-44 ms against 0.09-0.11 ms
  for a repeat call at the same count -- a factor of about 494 -- so a seven-step
  sequence over three distinct source counts spent 0.122 s of its 0.123 s total
  in recompilation. A long observation whose above-horizon set changes at most
  steps will pay this repeatedly.
- **Working set.** The kernel materializes ``(B, S, 2, 2)`` antenna-Jones batches,
  so its peak host allocation grows with ``baselines x sources``, not with
  sources alone. Measured at about 208 bytes per ``(baseline, source)`` pair and
  linear across four sizes (2.21 MB at 100x100, 133.25 MB at 800x800). Shipped
  configurations and the parity workloads use a handful of baselines, so this is
  invisible in normal use; a large array against a populated catalogue is not
  normal use, and the figure is published so the limit can be computed rather
  than discovered.

Host-side stages
----------------

"Partially integrated" is not a useful disclaimer, so here is the complete list
of stages that run on the host regardless of the selected backend, and why:

- **Astropy coordinate transforms** (``ICRS`` to ``AltAz``, LST, parallactic
  geometry). Astropy is the accepted source of truth for coordinates; a
  hand-rolled device transform would be a scientific change disguised as a
  performance change.
- **Horizon masking.** A per-time-step selection of the visible source or pixel
  set, feeding the compiled kernel its source axis.
- **Planck brightness conversion** in the HEALPix solver. A masked scalar
  transform of sky data, not a hot array operation.
- **FITS beam interpolation.** ``pyuvdata``'s ``UVBeam.interp``, which is
  host-side by nature.
- **HEALPix direction cosines.** Computed on the host from astropy output, then
  handed to the backend as a single array per time step.

Everything else in both solvers -- the Jones chain composition, the geometric
phase, the coherency construction, the contraction, and the accumulation -- goes
through the backend.

Measured position
-----------------

The following are measurements, not claims. They were produced by
``pixi run bench`` and are committed in full at
``output/benchmarks/reference/``; the numbers below are from
``20260731T104303Z-darwin-arm64.json``, taken at commit ``ea48d2c`` on an
Apple M1 Max (macOS 26.5.2, arm64, 10 logical CPUs) with ``numpy 2.3.2``,
``jax 0.10.2`` (CPU-only ``jaxlib``) and ``dask 2025.7.0``, at the ``standard``
precision preset (float64 coordinates, Jones terms, source tables,
accumulation, and output with float32 HEALPix map storage, producing
``complex128`` results).

**Correctness.** Across all eight benchmarked workloads -- the seven-row
point/HEALPix/hybrid/heterogeneous-receptor parity matrix plus a 4096-source,
four-time point workload:

- Dask is **bit-identical** to NumPy. Maximum absolute deviation ``0`` on every
  workload, as it must be: it delegates to the same NumPy operations.
- JAX-CPU agrees with NumPy within the stated tolerance
  ``|dV| <= atol + rtol*|V|``, ``rtol = 1e-12``,
  ``atol = 1e-12 * max(1, max|V|)``. Worst observed absolute deviation
  ``1.7e-11`` (relative ``4.0e-15``) on the 4096-source workload, against an
  allowed ``atol`` of ``5.2e-9``; four of the eight workloads deviated by
  ``0``. Bit-identity is neither required nor asserted: XLA may fuse and reorder
  the source reduction.

**Speed.** On this CPU-only host, JAX is **slower** than NumPy on every
workload measured, by roughly 3x on the 4096-source workload (0.121 s versus
0.040 s steady-state median) and by 10-20x on the small parity workloads, where
per-call dispatch dominates. Dask matches NumPy to within noise, as expected
from delegation. First-call compilation added 0.002-0.80 s depending on workload.

That is the honest position and it is not a defect: Tier 6 completed backend
*correctness* parity and the compilation boundary, not device-resident
orchestration. The solver still drives its time and frequency axes from host-side
Python, and coordinate transforms and beam interpolation still run on the host,
so an XLA backend pays dispatch and transfer costs it cannot yet amortize.

**No accelerator was exercised.** Every record carries ``accelerator: "none"``
and lists ``gpu``, ``tpu`` and ``distributed`` under ``unmeasured``. RadioSim
publishes no GPU, TPU, or distributed performance number, because none has been
measured. A GPU claim requires a real accelerator run.

Reproducing the records
-----------------------

.. code-block:: bash

   pixi run bench

The performance tests are marked ``performance`` and ``slow``, so
``pixi run test`` filtered by ``-m "not slow"`` and CI both exclude them; they
never gate. They assert record completeness, record honesty, and correctness
against NumPy -- never a time threshold, because a timing assertion on shared
hardware is a flake generator rather than a guarantee.

Each run writes ``output/benchmarks/<UTC timestamp>-<host tag>.json``, which is
gitignored. The committed reference set under ``output/benchmarks/reference/``
is a copy of one such run. Every record states the hardware, the accelerator (or
its absence), the backend and its version, the full precision tree, the antenna,
baseline, source, pixel, time and frequency counts, setup versus steady-state
timing, compilation time, host-transfer time, peak host memory, and the
correctness tolerance against NumPy. A record missing any of these is a
``BenchmarkRecordError``, not a partial record.

Any documentation sentence in RadioSim that asserts a speed, a GPU capability,
or a distributed capability must cite a record file. If it does not, it should
not be believed, and it should be reported as a bug.

Precision
---------

A precision override replaces the complete precision tree; it is not deeply
merged with the document. Presets and custom leaves are mutually exclusive in
one value. Explicit JAX/Dask plus ``float128`` is rejected during configuration
resolution, before importing the optional backend.

Installation
------------

NumPy ships with the base installation. Optional extras install backend
dependencies only:

.. code-block:: bash

   pip install radiosim[jax]       # JAX for supported platforms
   pip install radiosim[dask]      # NumPy/Dask backend

The standard pixi gates declare a **CPU-only** ``jax``/``jaxlib`` so the
NumPy/JAX parity evidence above is actually measured rather than skipped. The
isolated Linux ``gpu`` environment is readiness infrastructure with a strict
preflight; it is not an accelerator measurement. The device-named PyPI extras
``gpu``, ``gpu-cuda``, ``gpu-rocm`` and ``tpu`` were removed before ``0.3.0``:
RadioSim has measured no accelerator (``PERF-001``), so an installable extra
named for one advertised a capability this page explicitly does not claim. A
user with their own accelerator hardware installs the vendor's JAX wheel
directly, and nothing on this page changes when they do.
