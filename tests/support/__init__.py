"""Shared, non-collected helpers for the RadioSim test suite.

Modules here are imported by tests; they define no tests themselves and pytest
collects nothing from this package (``python_files = ["test_*.py"]``).

The one member today is :mod:`tests.support.repo_scan`, the git-scoped file
lister every repository-wide scan uses instead of :meth:`pathlib.Path.rglob`
(``Tier8ReleasePlan.md`` Section 12).
"""
