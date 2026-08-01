Changelog
=========

All notable changes to RadioSim are documented here.

[Unreleased]
------------

Removed
^^^^^^^

- **Twenty-six Jones classes**: every exported Jones term that returned the 2x2
  identity for every input, and could therefore be configured without changing
  anything, was removed rather than left as a scaffold. ``GeometricPhaseJones``
  became the ``geometric_phase()`` function (K is per-baseline),
  ``CrosshandPhaseJones`` was renamed ``CrosshandJones``, and the modules
  ``faraday``, ``wterm`` and ``element_beam`` are gone.
  :doc:`migration_guide` names the replacement for each. The nineteen remaining
  exports each declare ``term_status``: ``"implemented"`` for the receptor terms
  ``C`` and ``H``, ``"planned"`` for the rest, and a planned term **raises**
  when evaluated instead of multiplying by the identity.
- **jones_config solver parameter**: removed from every solver and simulator
  signature. It was an untyped dictionary hard-coded to ``None`` at the only
  production call site, and every term it could enable was one of the removed
  identity stubs. A typed ``jones:`` configuration section replaces it.
- **visibility.calculation_type**: removed before v1.0. It validated
  ``direct_sum`` and ``spherical_harmonic``, and nothing in the package read
  either value. The solver strategy is selected by ``execution.simulator``,
  whose accepted values are exactly the keys of the simulator registry. A
  document that still sets the key is rejected with removed-field guidance
  naming the replacement. See :doc:`migration_guide`.

[0.2.0] - 2025-12-15
--------------------

Major release with package restructuring and GPU support.

Added
^^^^^

- **Package Structure**: Proper Python package installable via ``pip install radiosim``
- **GPU Support**: Universal GPU acceleration via JAX and Numba backends
  - NVIDIA (CUDA 12)
  - AMD (ROCm)
  - Apple Silicon (Metal)
  - Google TPU
- **High-Level API**: New ``Simulator`` class for easy notebook/script usage
- **Backend Abstraction**: Write once, run on CPU or GPU
- **Jones Matrix Framework**: Complete 8-term Jones chain
  - K: Geometric delay
  - E: Primary beam
  - Z: Ionosphere
  - T: Troposphere
  - P: Parallactic angle
  - G: Gain
  - B: Bandpass
  - D: Polarization leakage
- **Precision Control**: Granular control over numerical precision
  - Presets: ``standard`` (float64), ``fast`` (mixed), ``precise`` (float128), ``ultra``
  - Per-component control for coordinates, Jones matrices, accumulation, output
  - float128/complex256 support on NumPy (platform-dependent)
  - Automatic fallback with warnings on unsupported backends
  - YAML configuration support in config files
- **Pydantic Configuration**: Type-safe configuration with validation
- **CLI Commands**: ``radiosim`` and ``radiosim-migrate``
- **Measurement Set I/O**: Export to CASA MS format for QuartiCal/WSClean
  - ``write_ms()``: Write visibilities to MS format
  - ``read_ms()``: Read MS files back into memory
  - ``ms_info()``: Quick metadata summary
  - Install with: ``pip install radiosim[ms]``
- **Test Suite**: 376+ tests covering unit, integration, and performance
- **CI/CD**: GitHub Actions with multi-Python, multi-OS testing
- **Documentation**: Sphinx documentation with autodoc

Changed
^^^^^^^

- Module structure reorganized into subpackages (``core/``, ``backends/``, ``api/``, etc.)
- Import paths changed from ``src.*`` to ``radiosim.*``
- ``file_format`` parameter renamed to ``format_type`` in antenna readers

Fixed
^^^^^

- Numerous bug fixes and performance improvements
- MWA antenna FITS file parsing
- Polarization support edge cases

[0.1.x] - 2024
--------------

Initial development releases.

Added
^^^^^

- Basic visibility calculation
- GLEAM and GSM sky models
- Analytic beam patterns
- Beam FITS file support
- Full polarization RIME
- Bokeh visualization
- HDF5 output

Migration
---------

See :doc:`migration_guide` for upgrading from v0.1.x to v0.2.0.
