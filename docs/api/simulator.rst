Simulator API
=============

``Simulator`` accepts one resolved runtime. Its classmethods provide disjoint
YAML, typed-model, mapping, and typed-parameter construction paths.

.. autoclass:: radiosim.api.simulator.Simulator
   :members:
   :show-inheritance:
   :special-members: __init__

.. code-block:: python

   from radiosim import Simulator

   yaml_simulator = Simulator.from_yaml("configs/config.yaml")
   mapping_simulator = Simulator.from_mapping(mapping, base_dir=project_dir)
   model_simulator = Simulator.from_config(input_model, base_dir=project_dir)
   parameter_simulator = Simulator.from_parameters(
       instrument=instrument,
       baseline_selection=baseline_selection,
       channel_frequencies_hz=(100_000_000.0, 101_500_000.0),
       channel_widths_hz=(1_000_000.0, 1_000_000.0),
       start_time="2025-01-01T00:00:00",
       sky_model=sky_model,
   )
   direct_simulator = Simulator(resolved_runtime)

The direct constructor rejects mappings and input models. Mapping/model inputs
with relative paths require ``base_dir``. No constructor executes CLI workflow
actions. ``run`` computes a canonical result. ``plot_observability`` remains an
explicit helper independent of simulation-result rendering.

``Simulator.run()`` returns an immutable
:class:`~radiosim.core.result.SimulationResult`, and ``Simulator.result``
returns the identical last successfully published object (or ``None`` before
the first successful run). The visibility array shape is
``(time, baseline, frequency, correlation)`` and the correlation order is
``XX, XY, YX, YY``. Use ``result.stokes_i()`` for derived Stokes I.

``Simulator.save(path, /, *, format=ResultFormat.HDF5, overwrite=False)``
treats ``path`` as the exact final artifact target.  It supports HDF5,
metadata-only summary JSON, Measurement Set, and UVFITS through the typed
``ResultFormat`` enum.  Missing canonical extensions are appended; conflicting
extensions and string format arguments are rejected.  Python APIs never
prompt, suffix, or read CLI workflow policy.

.. code-block:: python

   from radiosim import ResultFormat

   simulator.save("results/run", format=ResultFormat.HDF5)
   simulator.save("results/run", format=ResultFormat.SUMMARY_JSON)

``Simulator.plot()`` renders the published canonical result:

.. code-block:: python

   plots = simulator.plot(
       plot_type="all",
       output_dir="output/plots",
       backend="bokeh",
       show=True,
       overwrite=False,
       visibility_phase_unit="radians",
   )

Every parameter is keyword-only.  ``plot_type`` is exactly ``all``, ``antenna``,
``visibility``, ``heatmap``, or ``frequency``; ``output_dir`` is required and
explicit; only the ``bokeh`` backend is implemented.  The renderers consume the
published coordinate arrays directly — MJD time centers from
``result.time_grid``, channel centers in hertz from ``result.frequencies_hz``,
and the exact published baseline order — and never reconstruct an axis from a
duration, cadence, or scalar start time.  Stokes I is derived explicitly as
``XX + YY`` through ``SimulationResult.stokes_i``.

``visibility_phase_unit`` is exactly ``radians`` or ``degrees`` and affects only
the displayed phase axis.  Contract validation and collision checks precede all
filesystem work: an unknown plot family, backend, or phase unit raises
``ResultPlotContractError``, a missing ``output_dir`` raises ``OutputPathError``,
and an existing declared file without ``overwrite=True`` raises
``OutputCollisionError``.  A browser is opened only after every declared file is
published, and a browser failure raises ``ResultBrowserError`` without removing
published output.

After setup, ``instrument`` returns the canonical resolved object, while
``antennas`` and ``baselines`` return its exact immutable tuples. Access before
resolution raises ``RuntimeError``.

``beam_system`` returns the one loaded per-antenna ``BeamSystem`` used by both
visibility solvers and observability. ``beam_state`` returns its immutable
``LoadedBeamState``. Both are read-only properties and raise before successful
setup; neither exposes mutable evaluators through result provenance.

Beam sampling lifecycle
-----------------------

``setup`` resolves the instrument, canonical beam assignments, and complete
loaded ``BeamSystem`` before deriving HEALPix advice. The pre-sky derivation
uses only selected canonical baselines, exact loaded handler feature scales,
and every exact observation frequency. It runs before device inspection,
backend initialization, network policy, or sky loading. Invalid or incomplete
canonical state raises ``BeamSamplingDerivationError`` and retains only the
instrument state, so retry rebuilds the beam system.

When the resolved workflow uses HEALPix, the configured target NSIDE is checked
before sky loading. After preparation, an existing HEALPix payload is checked
again using its actual NSIDE. Point-only payloads have no post-load NSIDE
advisory, and an unchanged requested/actual grid does not receive duplicate
coarse-grid messages. Both checks are advisory only: ``Simulator`` never
changes or resamples the requested or loaded NSIDE.

Result beam provenance
----------------------

Every successful ``run`` includes the exact immutable loaded state at
``result.beam_state``. Its ``to_snapshot()`` method returns a fresh detached,
JSON-safe snapshot for analytic, shared-FITS, per-antenna-FITS, and mixed modes
in both point-source and HEALPix execution, including transport provenance
where designed. Scientific fingerprints remain path-independent where
designed.

The result contains no live beam evaluator, ``BeamSystem``, ``UVBeam``, array,
lock, logger, callable, observability reference choice, renderer state, or
``BeamSamplingRequirement``. Snapshot mutation cannot alter
``Simulator.beam_state`` or a later result. A failure before result
construction publishes no partial beam metadata.
