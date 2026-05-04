"""HEALPix ordering field tests for HealpixData."""

from __future__ import annotations

import healpy as hp
import numpy as np
import pytest

from rrivis.core.sky import HealpixData


def _basic_kwargs(nside: int = 8) -> dict:
    npix = hp.nside2npix(nside)
    return {
        "maps": np.zeros((1, npix), dtype=np.float32),
        "nside": nside,
        "frequencies": np.asarray([100e6], dtype=np.float64),
    }


class TestHealpixDataOrdering:
    def test_default_is_ring(self) -> None:
        data = HealpixData(**_basic_kwargs())
        assert data.ordering == "ring"

    def test_nest_is_accepted(self) -> None:
        data = HealpixData(ordering="nest", **_basic_kwargs())
        assert data.ordering == "nest"

    def test_invalid_ordering_raises(self) -> None:
        with pytest.raises(ValueError, match="ordering must be"):
            HealpixData(ordering="weird", **_basic_kwargs())

    def test_ordering_is_lowercased(self) -> None:
        data = HealpixData(ordering="NEST", **_basic_kwargs())
        assert data.ordering == "nest"

    def test_ordering_differs_breaks_equality(self) -> None:
        from rrivis.core.precision import PrecisionConfig
        from rrivis.core.sky import SkyModel

        precision = PrecisionConfig.standard()
        a = SkyModel(
            healpix=HealpixData(ordering="ring", **_basic_kwargs()),
            _precision=precision,
        )
        b = SkyModel(
            healpix=HealpixData(ordering="nest", **_basic_kwargs()),
            _precision=precision,
        )
        assert a != b
