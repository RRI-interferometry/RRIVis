Jones Matrix Framework
======================

RadioSim exposes a broad Jones-term framework, but public class availability is
not the same as implemented high-level science. The current high-level
``Simulator`` uses geometric phase (K) and the canonical scalar E-Jones primary
beam as substantive forward-model effects. Many other exported terms remain
identity scaffolds or later-tier work.

RIME context
------------

For baseline :math:`ij`, RadioSim uses the usual matrix form

.. math::

   V_{ij} = \sum_s J_i(\vec{s}) C_s J_j^H(\vec{s}).

The framework organizes geometric, beam, propagation, gain, bandpass,
polarization, and other terms. Only terms with implemented calculations and
scientific tests should be interpreted as changing simulated visibilities.

Current high-level effects
--------------------------

Geometric phase is calculated by the direct-sum visibility path. Primary beams
use the strict tagged configuration, for example:

.. code-block:: yaml

   beams:
     mode: analytic
     model:
       kind: circular_aperture
       taper:
         kind: gaussian
         edge_taper_db: 10.0

``analytic``, ``shared_fits``, ``per_antenna_fits``, and ``mixed`` all resolve
to one canonical per-antenna ``BeamSystem``. The point-source, HEALPix, and
observability paths use the same evaluator. Within the accepted FITS subset,
the E-Jones response is a scalar complex voltage on the diagonal of a 2x2
matrix. It does not claim arbitrary cross-polarization or receptor coupling.

The high-level API does not accept a caller-supplied Jones chain. Full
receptor/basis/polarization physics, including general polarized BeamFITS, is a
Tier 5 boundary rather than an implicit property of the scalar beam evaluator.

Scaffolded terms
----------------

Ionosphere, troposphere, parallactic rotation, gain, bandpass, polarization
leakage, receptor/basis transforms, and other exported terms are not advertised
as complete high-level effects. Until their later scientific tiers add
conventions, analytic invariants, reference comparisons, backend parity, and a
test proving a configured effect changes visibility, users should not include
them in scientific claims.

Low-level framework use
-----------------------

The API reference exposes low-level classes for development and inspection.
Read their individual docstrings and tests before use. A class returning an
identity matrix is a scaffold, not a physical model. Beam ownership remains
with the high-level canonical system; numeric analytic primitives do not form
a second configuration or runtime API.
