Jones Matrix Framework
======================

RadioSim exposes a broad Jones-term framework, but public class availability is
not the same as implemented high-level science. The current high-level
``Simulator`` uses geometric phase (K) and the analytic primary beam (E) as
substantive forward-model effects. Many other exported terms remain identity
scaffolds or later-tier work.

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

Geometric phase is calculated by the direct-sum visibility path. The analytic
beam is configured through the strict document:

.. code-block:: yaml

   beams:
     beam_mode: analytic
     aperture_shape: circular
     taper: gaussian
     edge_taper_dB: 10.0

The current high-level API does not accept a caller-supplied Jones chain. It
also rejects FITS/per-antenna beam configuration and non-default top-level
receptor fields.

Scaffolded terms
----------------

Ionosphere, troposphere, parallactic rotation, gain, bandpass, polarization
leakage, receptor/basis transforms, and other exported terms are not advertised
as complete high-level effects. Until their later scientific tiers add
conventions, analytic invariants, reference comparisons, backend parity, and a
test proving a configured effect changes visibility, users should not include
them in scientific claims.

Low-level use
-------------

The API reference exposes low-level classes for development and inspection.
Read their individual docstrings and tests before use. A class returning an
identity matrix is a scaffold, not a physical model. Low-level FITS beam types
likewise do not bypass the strict high-level resolver's rejection boundary.
