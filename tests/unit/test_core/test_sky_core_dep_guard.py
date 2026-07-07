"""Tests for healpy lazy-import guard and point-only import hygiene."""

from __future__ import annotations

import builtins
import subprocess
import sys
from pathlib import Path

import pytest

from radiosim.core.sky.support.healpy import (
    HEALPY_IMPORT_ERROR_MESSAGE,
    _healpy,
)


def _reset_healpy_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    import radiosim.core.sky.support.healpy as healpy_support

    monkeypatch.setattr(healpy_support, "_HEALPY_MODULE", None, raising=False)


def _block_healpy_import(monkeypatch: pytest.MonkeyPatch) -> None:
    original_import = builtins.__import__

    def guarded_import(name: str, globals=None, locals=None, fromlist=(), level=0):
        if name == "healpy" or name.startswith("healpy."):
            raise ModuleNotFoundError(f"No module named {name!r}")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    monkeypatch.delitem(sys.modules, "healpy", raising=False)


class TestHealpyGuard:
    def test_missing_healpy_raises_actionable_import_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _reset_healpy_cache(monkeypatch)
        _block_healpy_import(monkeypatch)

        with pytest.raises(ImportError, match="pixi install") as exc_info:
            _healpy()

        assert HEALPY_IMPORT_ERROR_MESSAGE in str(exc_info.value)

    def test_lazy_proxy_triggers_same_guard(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from radiosim.core.sky.support.healpy import lazy_healpy

        _reset_healpy_cache(monkeypatch)
        _block_healpy_import(monkeypatch)

        with pytest.raises(ImportError, match="pixi install"):
            lazy_healpy.nside2npix(8)


class TestPointPathImportHygiene:
    def test_import_sky_without_loading_healpy_subprocess(self) -> None:
        repo_root = Path(__file__).resolve().parents[3]
        code = """
import sys
from radiosim.core.precision import PrecisionConfig
import radiosim.core.sky as sky

assert "healpy" not in sys.modules
sky.create_test_sources(precision=PrecisionConfig.standard())
region = sky.SkyRegion.cone(0.0, 0.0, 5.0)
import numpy as np
region.contains(np.array([0.0]), np.array([0.0]))
assert "healpy" not in sys.modules
print("ok")
"""
        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr or result.stdout
        assert "ok" in result.stdout

    def test_healpix_operation_loads_healpy_when_present(self) -> None:
        import radiosim.core.sky.support.healpy as healpy_support

        healpy_support._HEALPY_MODULE = None
        sys.modules.pop("healpy", None)

        import radiosim.core.sky as sky

        region = sky.SkyRegion.cone(0.0, 0.0, 5.0)
        region.healpix_mask(nside=8)
        assert "healpy" in sys.modules
