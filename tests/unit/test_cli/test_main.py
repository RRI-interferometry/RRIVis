# tests/test_main.py
"""Tests for the CLI entry point."""


def test_main_module_importable():
    """Test that the rrivis CLI entry point is importable."""
    from rrivis.cli import main  # noqa: F401
