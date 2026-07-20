Beam Models
===========

The high-level ``Simulator`` supports analytic aperture illumination. FITS,
mixed, and per-antenna beam assignment remain rejected.

.. code-block:: yaml

   instrument:
     source:
       kind: layout_file
       path: antennas.txt
       format: radiosim
       telescope_name: Example Array
     location:
       longitude_deg: 21.4283
       latitude_deg: -30.7215
       height_m: 1050.0
     default_diameter_m: 14.0

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
``elliptical``. Rectangular and elliptical shapes require their positive
dimensions in ``aperture_params``. ``feed_model`` describes analytic aperture
illumination; it does not implement receptor/feed physics.

Antenna diameters come from canonical instrument resolution. The point and
HEALPix visibility solvers receive the same per-antenna diameter vectors, so
heterogeneous arrays are supported for analytic simulation. There is no hidden
diameter fallback.

``Simulator.plot_observability`` uses one uniform resolved diameter. It raises
a dedicated error for a heterogeneous array before optional sky preparation,
because heterogeneous footprint semantics are not defined. This restriction
does not reduce heterogeneous visibility support.

Low-level UVBeam/FITS modules do not make FITS beams a supported high-level
feature. Backend selection also does not prove complete GPU beam execution.
