"""Point-source builder helper.

Consolidates the ``PointSourceData(...)``-from-column-dict construction
that appears in ``operations/factories.py::create_from_arrays`` and three
times in ``combine/engine.py`` (spec item B9). The shared builder applies
the same precision dtypes as ``create_from_arrays`` (RA/Dec at
``source_positions`` precision, flux/Stokes/ref_freq at ``flux`` precision,
spectral index at ``spectral_index`` precision) and forwards the optional
morphology / polarization / metadata / spectral / per-channel sub-blocks
untouched.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from ..containers import PointSourceData

if TYPE_CHECKING:
    from radiosim.core.precision import PrecisionConfig

#: Core columns and the precision sub-component whose dtype they take.
_POSITION_KEYS = ("ra_rad", "dec_rad")
_FLUX_KEYS = ("flux", "stokes_q", "stokes_u", "stokes_v", "ref_freq")
_SPECTRAL_INDEX_KEYS = ("spectral_index",)

#: Optional columns forwarded verbatim (no precision recast applied; they
#: keep their incoming dtype, matching create_from_arrays / engine.py).
_OPTIONAL_KEYS = (
    "spectral_coeffs",
    "rotation_measure",
    "major_arcsec",
    "minor_arcsec",
    "pa_deg",
    "source_name",
    "source_id",
    "extra_columns",
    "morphology",
    "polarization",
    "metadata",
    "spectrum",
)


def point_source_data_from_mapping(
    data: dict[str, Any],
    precision: PrecisionConfig,
) -> PointSourceData:
    """Build a :class:`PointSourceData` from a column mapping at ``precision``.

    Core columns are cast to the precision dtypes used by
    ``create_from_arrays``:

    * ``ra_rad`` / ``dec_rad`` → ``sky_model.source_positions``
    * ``flux`` / ``stokes_q`` / ``stokes_u`` / ``stokes_v`` / ``ref_freq``
      → ``sky_model.flux``
    * ``spectral_index`` → ``sky_model.spectral_index``

    Optional sub-block columns (``rotation_measure``, ``major_arcsec``,
    ``minor_arcsec``, ``pa_deg``, ``spectral_coeffs``, ``source_name``,
    ``source_id``, ``extra_columns``) and pre-built nested blocks
    (``morphology``, ``polarization``, ``metadata``, ``spectrum``) are
    forwarded unchanged when present.

    Parameters
    ----------
    data : dict
        Column mapping. Must contain the eight core keys (``ra_rad``,
        ``dec_rad``, ``flux``, ``spectral_index``, ``stokes_q``,
        ``stokes_u``, ``stokes_v``, ``ref_freq``).
    precision : PrecisionConfig
        Precision configuration driving the core-column dtypes.

    Returns
    -------
    PointSourceData
    """
    src_dt = precision.sky_model.get_dtype("source_positions")
    flux_dt = precision.sky_model.get_dtype("flux")
    si_dt = precision.sky_model.get_dtype("spectral_index")

    kwargs: dict[str, Any] = {}
    for key in _POSITION_KEYS:
        kwargs[key] = np.asarray(data[key], dtype=src_dt)
    for key in _FLUX_KEYS:
        kwargs[key] = np.asarray(data[key], dtype=flux_dt)
    for key in _SPECTRAL_INDEX_KEYS:
        kwargs[key] = np.asarray(data[key], dtype=si_dt)

    for key in _OPTIONAL_KEYS:
        if key in data and data[key] is not None:
            kwargs[key] = data[key]

    return PointSourceData(**kwargs)
