"""Tests for BBS sky loader edge cases."""

from __future__ import annotations

import pytest

from radiosim.core.precision import PrecisionConfig
from radiosim.core.sky.loaders.bbs import load_bbs


def test_bbs_loader_missing_file_has_actionable_error(tmp_path):
    missing = tmp_path / "missing.skymodel"

    with pytest.raises(OSError, match="Could not open BBS sky model file"):
        load_bbs(str(missing), precision=PrecisionConfig.standard())
