# tests/test_main.py
"""Tests for the legacy main.py entry point (deprecated)."""

import pytest
import warnings


def test_main_execution():
    """Test that the legacy main.py can be imported with deprecation warning."""
    # Note: src/main.py is deprecated and will be removed in v0.3.0
    # This test verifies it still works during the transition period
    import sys
    import os

    # Add src to path temporarily for legacy import
    src_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "src")
    sys.path.insert(0, src_path)

    try:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            import main
            # Check that deprecation warning was raised
            assert any("deprecated" in str(warning.message).lower() for warning in w)
    finally:
        sys.path.remove(src_path)
