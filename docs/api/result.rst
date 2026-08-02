Result
======

``Simulator.run()`` returns one immutable ``SimulationResult``. It owns the
canonical visibility cube with shape ``(time, baseline, frequency,
correlation)``, the coordinate arrays those axes were computed from, the
resolved correlation labels and polarization basis, the beam-state snapshot,
and the two content digests.

``scientific_sha256`` hashes the raw little-endian bytes of the visibilities,
flags, weights, time grid, frequencies, and channel widths.
``provenance_sha256`` additionally covers run metadata, including the package
version, so a version bump moves the provenance digest and never the scientific
one.

.. automodule:: radiosim.core.result
   :members:
   :undoc-members:
   :show-inheritance:
