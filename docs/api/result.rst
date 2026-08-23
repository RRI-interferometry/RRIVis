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

The solver record is a strict tagged union
------------------------------------------

``SimulationResult.solver`` is one of two arms, discriminated by its ``solver``
field. A direct run keeps the unchanged ``SolverResultProvenance``: its six
fields, its serialization, and therefore every ``rime`` fingerprint are
byte-identical to what they were before ``SCI-004`` introduced the union.

An m-mode run carries ``MModeSolverResultProvenance`` instead. Its snapshot has
the same six common fields -- ``solver``, ``sky_representation``,
``convention``, ``execution_path``, ``components`` and
``component_element_counts`` -- followed by the m-mode block: the time-grid,
frame and harmonic conventions, the four truncation dimensions, the quadrature
and truncation policies, ``tangent_polarization_frame``,
``stokes_v_basis_bridge``, the bundled-IERS table digest, the frame-certificate
digest, and the transform execution policy. Neither
``tangent_polarization_frame`` nor ``stokes_v_basis_bridge`` is nullable: in
phase M1 the first is the exact literal ``not_applicable_scalar_m1``, because
the phase carries no polarized payload at all, and the second is always
``radiosim.stokes-ne-theta-phi.v1``.

Point, HEALPix and hybrid remain **solver provenance**, not separate output
products, on either arm: the public result keeps its single ``(time, baseline,
frequency, correlation)`` visibility array in the four row-major correlation
labels, which is the exact row-major flattening of the strategy's
``(time, baseline, frequency, 2, 2)`` receptor cube. Readers reconstruct and
authenticate whichever arm was written; a reader that silently relabels an
m-mode record as ``rime`` is a failure, not a fallback.

.. automodule:: radiosim.core.result
   :members:
   :undoc-members:
   :show-inheritance:
