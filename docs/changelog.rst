Changelog
=========

All notable changes to RRIvis are documented here.

[0.2.0] - 2025-12-15
--------------------

Major release with package restructuring and GPU support.

Added
^^^^^

- **Package Structure**: Proper Python package installable via ``pip install rrivis``
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
- **CLI Commands**: ``rrivis`` and ``rrivis-migrate``
- **Measurement Set I/O**: Export to CASA MS format for QuartiCal/WSClean
  - ``write_ms()``: Write visibilities to MS format
  - ``read_ms()``: Read MS files back into memory
  - ``ms_info()``: Quick metadata summary
  - Install with: ``pip install rrivis[ms]``
- **Test Suite**: 376+ tests covering unit, integration, and performance
- **CI/CD**: GitHub Actions with multi-Python, multi-OS testing
- **Documentation**: Sphinx documentation with autodoc

Changed
^^^^^^^

- Module structure reorganized into subpackages (``core/``, ``backends/``, ``api/``, etc.)
- Import paths changed from ``src.*`` to ``rrivis.*``
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
