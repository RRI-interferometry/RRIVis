"""Resolved-owner attachment controls; no loader or public solver admission."""

import numpy as np
import pytest

from radiosim.core.precision import PrecisionConfig
from radiosim.core.sky.containers._polarization_materialization import (
    complete_model_native_identity,
    require_native_identity,
)
from radiosim.core.sky.containers.constants import BrightnessConversion as BC
from radiosim.core.sky.containers.healpix import HealpixData
from radiosim.core.sky.containers.model import SkyModel
from radiosim.core.sky.containers.point import TangentPolarizationFrame


@pytest.mark.parametrize("precise", [False, True])
def test_attachment_uses_final_owner_and_preserves_noop(precise: bool) -> None:
    raw = HealpixData(
        maps=np.array([[2.0]], dtype=np.float64),
        q_maps=np.array([[1.0]], dtype=np.float64),
        frequencies=np.array([1.0]),
        nside=1,
        hpx_inds=np.array([4]),
    )
    model = SkyModel(
        healpix=raw,
        precision=PrecisionConfig.precise() if precise else PrecisionConfig.fast(),
    )
    assert raw.i_brightness_conversion is None
    assert model.healpix is not None
    assert model.healpix.i_brightness_conversion == "planck"
    assert model.healpix.maps.dtype == np.dtype("float64" if precise else "float32")
    assert model.healpix.polarization_materialization is None
    frame = TangentPolarizationFrame.canonical("icrs")
    bound = complete_model_native_identity(
        model, source_profile="radiosim_ne_iau_v1", tangent_frame=frame
    )
    assert bound.healpix is not None
    receipt = bound.healpix.polarization_materialization
    assert receipt is not None
    require_native_identity(
        bound.healpix,
        brightness_conversion=BC.PLANCK,
        source_profile="radiosim_ne_iau_v1",
        tangent_frame=frame,
        expected=receipt,
    )
    repeated = complete_model_native_identity(
        bound, source_profile="radiosim_ne_iau_v1", tangent_frame=frame
    )
    assert repeated.healpix is not None
    assert repeated.healpix.polarization_materialization is receipt
    assert repeated == bound
    with pytest.raises(ValueError, match="materialization"):
        _ = bound.replace(brightness_conversion=BC.RAYLEIGH_JEANS)
    with pytest.raises(ValueError, match="materialization"):
        _ = bound.replace(healpix=raw)
    with pytest.raises(ValueError, match="materialization"):
        _ = bound.healpix.replace(polarization_materialization=None)
    changed = bound.healpix.maps.copy()
    changed[0, 0] += 1
    with pytest.raises(ValueError, match="materialization"):
        _ = bound.healpix.replace(maps=changed)
    assert bound.healpix.polarization_materialization is receipt
    assert bound.healpix.maps[0, 0] == 2
