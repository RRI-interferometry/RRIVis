Configuration Guide
===================

RadioSim has one strict Tier 1 configuration contract. YAML documents, Python
mappings, and ``RadioSimConfig`` models pass through the same schema, semantic,
unsupported-feature, path, override, and runtime-resolution stages.

Unknown fields are rejected. This is an intentionally breaking pre-v1 API; the
resolver does not translate removed sections or constructor patterns.

Complete document
-----------------

The scientific inputs ``antenna_layout``, ``location``, ``sky_model``,
``obs_time``, and ``obs_frequency`` are required:

.. code-block:: yaml

   telescope:
     telescope_name: HERA

   antenna_layout:
     antenna_positions_file: ../antenna_layout_examples/hera_5.txt
     antenna_file_format: radiosim
     all_antenna_diameter: 14.0

   beams:
     beam_mode: analytic
     aperture_shape: circular
     taper: gaussian
     edge_taper_dB: 10.0
     feed_model: none
     reflector_type: prime_focus

   baseline_selection:
     use_autocorrelations: true
     use_crosscorrelations: true

   location:
     lat: -30.72152
     lon: 21.4283
     height: 1073.0

   obs_time:
     start_time: "2025-01-01T00:00:00"
     duration_seconds: 60.0
     time_step_seconds: 10.0

   obs_frequency:
     mode: explicit
     channel_frequencies_hz: [100000000.0, 101500000.0, 108000000.0]

   sky_model:
     flux_unit: Jy
     sources:
       - kind: test_sources
         num_sources: 3
         seed: 7

   visibility:
     calculation_type: direct_sum
     sky_representation: point_sources

   execution:
     backend: numpy
     precision:
       preset: standard
     simulator: rime
     offline: true

   workflow:
     output_dir: output
     run_subdir: null
     result_filename: visibilities
     result_format: hdf5
     save_results: false
     overwrite: false
     skip_overwrite_confirmation: false
     prompt_for_output_suffix: false
     plot_results: false
     open_plots_in_browser: false
     plotting_backend: bokeh
     save_log: false

Frequency modes
---------------

``obs_frequency`` is a discriminated union.

Use ``mode: explicit`` for one or more immutable, strictly increasing channel
values in Hz. Nonuniform spacing is valid and preserved:

.. code-block:: yaml

   obs_frequency:
     mode: explicit
     channel_frequencies_hz: [100000000.0, 101250000.0, 109000000.0]

Use ``mode: grid`` for a uniform grid:

.. code-block:: yaml

   obs_frequency:
     mode: grid
     starting_frequency: 100.0
     frequency_interval: 0.25
     frequency_bandwidth: 1.0
     frequency_unit: MHz

Grid bandwidth is the inclusive span from the first to last channel. The ratio
``frequency_bandwidth / frequency_interval`` must be integral within the
documented tolerance. Resolution uses ``start + index * interval`` and never
adjusts the requested interval to fit an endpoint.

Execution and workflow
----------------------

``execution`` selects scientific runtime policy:

- ``backend``: ``numpy``, ``jax``, ``numba``, or ``auto``;
- ``precision``: a complete preset or custom precision tree;
- ``simulator``: ``rime`` in the current high-level contract; and
- ``offline``: network policy for runtime sky loading.

``workflow`` is CLI-only orchestration. It controls output directory and run
name, result filename and format, save/overwrite/skip behavior, prompting,
plotting, browser opening, plotting backend, and log saving.

``ResolvedSimulationConfig`` never contains ``workflow``. Calling
``Simulator.from_yaml`` resolves scientific state but does not save, log, plot,
prompt, skip, or open a browser. Config-mode CLI performs those actions only
after a successful run.

Loading and serialization
-------------------------

``load_config`` returns a ``ResolvedConfiguration`` bundle, not the mutable
input model:

.. code-block:: python

   from radiosim.io import load_config

   bundle = load_config("configs/config.yaml")
   runtime = bundle.runtime
   workflow = bundle.workflow
   provenance = bundle.provenance
   print(runtime.execution.backend_strategy)

``dump_config`` accepts only a strict ``RadioSimConfig`` input model and writes
the user-facing document. Runtime paths and provenance are not serialized:

.. code-block:: python

   from radiosim.io import dump_config
   from radiosim.io.config import RadioSimConfig

   input_model = RadioSimConfig.model_validate(document_mapping)
   dump_config(input_model, "copied-config.yaml")

Use standard ``input_model.model_dump(mode="json")`` when an ordinary Python
mapping is needed.

Mapping and model resolution
----------------------------

Mapping and typed-model inputs use ``resolve_config`` with an explicit source:

.. code-block:: python

   from pathlib import Path

   from radiosim.io import resolve_config
   from radiosim.io.config_resolution import ConfigurationSource

   project_dir = Path("/absolute/project/directory")
   bundle = resolve_config(
       document_mapping,
       source=ConfigurationSource.for_mapping(base_dir=project_dir),
   )

The high-level wrappers are ``Simulator.from_mapping(mapping, base_dir=...)``
and ``Simulator.from_config(input_model, base_dir=...)``.

Path rules
----------

- A YAML document's relative paths are based at the YAML file's parent.
- Relative mapping/model paths require an explicit ``base_dir``.
- Explicit call-site override paths use the captured invocation directory.
- ``~`` is expanded; hidden ``$VARIABLE`` substitution is rejected.
- Input paths are normalized and checked before backend initialization.
- Validation never creates an output directory.
- Glob expansion is deterministic and sorted.

The rules cover antenna files, file-backed sky sources, skyh5 lists/globs,
registry-declared file options, future beam paths, and workflow output paths.

Overrides and precedence
------------------------

The precedence rule is:

``explicit override > document value > declared default``

``None`` means no override. ``auto`` is an actual backend choice. Frequency and
location overrides replace complete logical values; they are not deep merged.
A precision override replaces the complete precision tree. Explicit
backend/precision incompatibilities fail before optional backend imports, and
unavailable explicit backends fail during backend construction rather than
silently falling back.

Current feature boundary
------------------------

The high-level path currently consumes analytic beam fields, one uniform
antenna diameter, all generated baselines, explicit location and paths, the
direct-sum simulator, resolved backend selection, and observability plotting as
a ``Simulator`` helper.

These declared later-tier settings are rejected before runtime side effects:

- FITS, mixed, and per-antenna beam execution;
- per-antenna diameter maps;
- baseline-subset changes;
- top-level receptor/feed fields;
- pyuvdata telescope flags;
- UVFITS workflow output; and
- spherical-harmonic simulation.

A lower-level class or schema field is not evidence that a high-level feature
is connected.

Default template
----------------

Generate a strict template with explicit placeholders:

.. code-block:: bash

   radiosim init --output my-config.yaml

The generated antenna path is a placeholder. Replace it before resolving the
document.
