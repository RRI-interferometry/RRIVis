Visualization
=============

``radiosim.visualization`` renders published results and sky models. Only the
``bokeh`` backend is implemented. The result renderers consume the published
coordinate arrays directly -- MJD time centres from ``result.time_grid``,
channel centres in hertz from ``result.frequencies_hz``, and the exact
published baseline order -- and never reconstruct an axis from a duration,
cadence, or scalar start time. Stokes I is derived explicitly as the sum of the
two parallel hands, and the axis label names them for the result's own
polarization basis.

The subpackage re-exports its whole public surface, so each renderer and typed
error is documented once here rather than again under its defining module.

.. automodule:: radiosim.visualization
   :members:
   :undoc-members:
   :show-inheritance:
