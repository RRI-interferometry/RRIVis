# tests/test_main.py
"""Tests for the legacy main.py entry point (deprecated)."""

import pytest
import warnings


def test_main_execution():
    """Test that the legacy main.py cannot be imported (rrivis.core.source removed)."""
    # Note: src/main.py is deprecated and depends on rrivis.core.source
    # which was removed during the v0.2.0 refactoring.
    import sys
    import os

    # Add src to path temporarily for legacy import
    src_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "src")
    sys.path.insert(0, src_path)

    try:
        with pytest.raises((ImportError, ModuleNotFoundError)):
            import main
    finally:
        sys.path.remove(src_path)
