Contributing
============

We welcome contributions to RRIvis! This guide will help you get started.

Development Setup
-----------------

1. Fork and clone the repository:

   .. code-block:: bash

      git clone https://github.com/YOUR_USERNAME/RRIvis.git
      cd RRIvis

2. Install with pixi:

   .. code-block:: bash

      pixi install
      pixi shell

3. Run tests to verify setup:

   .. code-block:: bash

      pytest

Code Style
----------

We use the following tools:

- **ruff** - Linting
- **ruff format** - Code formatting
- **pyright** - Type checking

Run formatting and linting:

.. code-block:: bash

   ruff format src/ tests/
   ruff check src/ tests/
   pyright src/rrivis

Testing
-------

Write tests for all new functionality:

.. code-block:: bash

   # Run all tests
   pytest

   # Run with coverage
   pytest --cov=rrivis --cov-report=html

   # Run specific tests
   pytest tests/unit/test_visibility.py
   pytest -k "test_gaussian"

Test markers:

- ``@pytest.mark.slow`` - Slow tests
- ``@pytest.mark.gpu`` - GPU-required tests
- ``@pytest.mark.integration`` - Integration tests

Documentation
-------------

Update documentation for new features:

1. Add docstrings (NumPy style)
2. Update relevant .rst files
3. Build and verify:

   .. code-block:: bash

      cd docs
      make html
      open _build/html/index.html

Docstring Format
^^^^^^^^^^^^^^^^

Use NumPy-style docstrings:

.. code-block:: python

   def calculate_visibility(
       baselines: np.ndarray,
       sources: List[Source],
       frequencies: np.ndarray,
   ) -> np.ndarray:
       """
       Calculate visibilities for given baselines and sources.

       Parameters
       ----------
       baselines : np.ndarray
           Baseline coordinates, shape (n_baselines, 3).
       sources : List[Source]
           List of source objects.
       frequencies : np.ndarray
           Frequency array in Hz.

       Returns
       -------
       np.ndarray
           Complex visibilities, shape (n_baselines, n_frequencies).

       Examples
       --------
       >>> vis = calculate_visibility(baselines, sources, freqs)
       >>> print(vis.shape)
       (100, 50)

       Notes
       -----
       Uses the RIME formalism for visibility calculation.
       """

Pull Request Process
--------------------

1. Create a feature branch:

   .. code-block:: bash

      git checkout -b feature/amazing-feature

2. Make changes and commit:

   .. code-block:: bash

      git add .
      git commit -m "Add amazing feature"

3. Push and create PR:

   .. code-block:: bash

      git push origin feature/amazing-feature

4. Ensure CI passes
5. Request review

Commit Messages
^^^^^^^^^^^^^^^

Use conventional commits:

- ``feat:`` New feature
- ``fix:`` Bug fix
- ``docs:`` Documentation
- ``test:`` Tests
- ``refactor:`` Code refactoring
- ``perf:`` Performance improvement

Example: ``feat: add ionosphere Jones term``

Issue Reports
-------------

When reporting issues, include:

1. RRIvis version (``rrivis --version``)
2. Python version
3. Operating system
4. Minimal reproducible example
5. Full error traceback

Feature Requests
----------------

For feature requests:

1. Check existing issues first
2. Describe the use case
3. Propose implementation if possible

Code of Conduct
---------------

- Be respectful and inclusive
- Focus on constructive feedback
- Help others learn and grow

Questions?
----------

- Open a GitHub issue
- Check existing documentation
- Review closed issues for solutions
