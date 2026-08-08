Instrument Resolution
=====================

RadioSim resolves one instrument source into a canonical immutable
``Instrument``. That object owns telescope identity, Earth location, antenna
names and numbers, ENU positions, positive diameters, inert source metadata,
and a deterministic provenance snapshot.

Sources and formats
-------------------

``instrument.source`` is a discriminated union:

- ``layout_file`` accepts ``radiosim``, ``casa_loc``, ``measurement_set``,
  ``uvfits``, or ``mwa_metafits``;
- ``known_telescope`` accepts a registry name and ``offline`` or
  ``allow_network`` policy.

Local ``radiosim``, ``casa_loc``, and ``mwa_metafits`` sources require
``source.telescope_name`` and ``instrument.location``. Dataset sources can
provide embedded identity and location. An explicit matching value takes
precedence; a conflicting identity or location fails resolution, while both
source and explicit facts remain recorded in provenance.

Global ``execution.offline`` is authoritative. A source that explicitly
requests network access is rejected under global offline mode. Offline
resolution uses an injected/local registry seam and does not enumerate or
download the live registry.

Diameter precedence
-------------------

Resolution applies values in this order:

1. matching typed ``diameter_overrides``;
2. a positive diameter from the selected source;
3. ``default_diameter_m``.

Each override references an antenna with a tagged name or number. Resolution
rejects missing, ambiguous, duplicate, or contradictory references and fails
if any final diameter is absent or non-positive. Source and chosen values are
both represented in provenance.

Typed baseline selection
------------------------

``baseline_selection.correlations`` is ``all``, ``cross``, or ``auto``.
``length_filter`` is either typed ``targets`` with a tolerance or typed
inclusive ``ranges``. ``azimuth_ranges_deg`` contains axial ranges on
``[0, 180)`` and supports wrapped intervals. Selection returns immutable
canonical ``Baseline`` objects and a stable source-row map.

Lifecycle and public state
--------------------------

Construction resolves configuration and paths but does not load instrument,
backend, sky, output, or browser state. ``setup`` resolves and selects the
instrument first, then resolves receptors, Jones terms, and beams before it
creates the backend, solver adapter, observation, and sky state. A failed
instrument resolution publishes no partial public state; retry starts cleanly.
Later setup failures keep the already resolved canonical instrument available
for inspection and retry.

Before instrument resolution, ``instrument``, ``antennas``, and ``baselines``
raise ``RuntimeError``. Afterwards they return the exact immutable canonical
object and tuples. Solver adapters copy canonical values into fresh C-contiguous
``float64`` arrays and make every exposed array read-only.

Results and output
------------------

The immutable in-memory result exposes the exact canonical antenna and
selected-baseline tuples plus detached, JSON-safe instrument and beam
snapshots. After a successful run, ``Simulator.save`` publishes that result as
HDF5, summary JSON, Measurement Set, or UVFITS through the corresponding typed
``ResultFormat``. HDF5 is the reconstructable native result; summary JSON is a
bounded metadata view; Measurement Set and UVFITS are projected standard-format
exports. ``Simulator.plot`` publishes deterministic Bokeh result views into a
required explicit directory. Both methods raise ``ResultUnavailableError``
before side effects when no successful result has been published.

Direct CLI and config mode
--------------------------

Config mode accepts every supported typed source and baseline criterion from
YAML. The direct CLI intentionally exposes a local RadioSim layout, explicit
telescope name/location/diameter, correlation choice, and scientific runtime
controls. Use config mode for dataset sources, registry sources, per-antenna
overrides, or length/azimuth baseline filters.
