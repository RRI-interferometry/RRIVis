Beam Models
===========

The current high-level ``Simulator`` supports one analytic beam configuration
and one uniform antenna diameter. FITS, mixed, and per-antenna beam fields exist
for a later tier but are rejected by configuration resolution today.

Analytic beam configuration
---------------------------

.. code-block:: yaml

   antenna_layout:
     antenna_positions_file: antennas.txt
     antenna_file_format: radiosim
     all_antenna_diameter: 14.0

   beams:
     beam_mode: analytic
     aperture_shape: circular
     taper: gaussian
     edge_taper_dB: 10.0
     feed_model: none
     feed_computation: analytical
     feed_params: {}
     reflector_type: prime_focus
     magnification: 1.0
     aperture_params: {}

The supported aperture shapes are ``circular``, ``rectangular``, and
``elliptical``. Rectangular and elliptical shapes require their corresponding
positive dimensions in ``aperture_params``. Supported tapers are ``uniform``,
``gaussian``, ``parabolic``, ``parabolic_squared``, and ``cosine``.

``feed_model`` and related fields describe illumination of the analytic
aperture. They are not the top-level ``feeds`` receptor configuration. The
latter is rejected until receptor/basis physics is implemented.

Uniform-array behavior
----------------------

``all_antenna_diameter`` is applied to every loaded antenna by the current
high-level setup. File-provided per-antenna diameters and the deferred
``diameters`` mapping are not active high-level behavior. Enabling
``use_different_diameters`` or supplying a nonempty map is rejected before
backend or loader setup.

FITS and per-antenna fields
---------------------------

The strict schema names later-tier fields such as ``beam_file``,
``antenna_beam_map``, ``per_antenna``, interpolation controls, zenith-angle
limits, frequency buffers, and peak normalization. These fields are deliberately
rejected because the modern resolved configuration is not yet connected to the
high-level FITS ``BeamManager``.

Do not treat the presence of low-level UVBeam/FITS modules as high-level
support. Current YAML must use ``beam_mode: analytic``. There is no silent FITS
fallback to analytic behavior.

Observability
-------------

``Simulator.plot_observability()`` uses the resolved analytic beam and uniform
diameter to build its visualization. It is a helper attached to the Simulator,
not another beam engine or simulation product. Heterogeneous-beam display
semantics remain undefined until the later beam/instrument tiers.

Performance
-----------

No blanket GPU or beam speedup is documented. Backend selection does not prove
that the complete beam and visibility calculation ran on an accelerator.
