"""Tests for rrivis.core.sky._registry — loader registration system."""

import pytest

from rrivis.core.sky._registry import (
    _LOADERS,
    get_loader,
    list_loaders,
    register_loader,
)

# ---------------------------------------------------------------------------
# Registration and retrieval
# ---------------------------------------------------------------------------


class TestRegistry:
    def test_register_and_get_loader(self):
        """Register a dummy function, retrieve it by name."""

        @register_loader("_test_dummy_loader")
        def _dummy_loader(**kwargs):
            return "dummy"

        try:
            retrieved = get_loader("_test_dummy_loader")
            assert retrieved is _dummy_loader
            assert retrieved() == "dummy"
        finally:
            # Clean up so we don't pollute the global registry
            _LOADERS.pop("_test_dummy_loader", None)

    def test_get_unknown_loader_raises(self):
        """Requesting an unknown loader name raises ValueError."""
        with pytest.raises(ValueError, match="Unknown sky model loader"):
            get_loader("__nonexistent_loader_xyz__")

    def test_list_loaders_sorted(self):
        """list_loaders() returns a sorted list."""
        names = list_loaders()
        assert names == sorted(names)
        assert isinstance(names, list)
        assert len(names) > 0

    def test_all_expected_loaders_registered(self):
        """After importing sky, all expected loaders exist in the registry."""
        # Force import of the sky package which triggers all @register_loader
        import rrivis.core.sky  # noqa: F401

        names = list_loaders()
        expected = {
            "gleam",
            "mals",
            "vlssr",
            "tgss",
            "wenss",
            "sumss",
            "nvss",
            "lotss",
            "3c",
            "vlass",
            "racs",
            "diffuse_sky",
            "pysm3",
            "pyradiosky_file",
            "bbs",
            "fits_image",
        }
        missing = expected - set(names)
        assert not missing, f"Expected loaders not registered: {missing}"
