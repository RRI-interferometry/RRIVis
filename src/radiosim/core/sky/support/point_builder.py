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

from ..containers import (
    PointMetadata,
    PointMorphology,
    PointPolarization,
    PointSourceData,
)

if TYPE_CHECKING:
    from radiosim.core.precision import PrecisionConfig

#: Core columns and the precision sub-component whose dtype they take.
_POSITION_KEYS = ("ra_rad", "dec_rad")
_FLUX_KEYS = ("flux", "stokes_q", "stokes_u", "stokes_v", "ref_freq")
_SPECTRAL_INDEX_KEYS = ("spectral_index",)

#: Flat per-source column keys that pack into the nested sub-dataclasses.
_FLAT_MORPHOLOGY_FIELDS = ("major_arcsec", "minor_arcsec", "pa_deg")
_FLAT_POLARIZATION_FIELDS = ("rotation_measure",)
_FLAT_METADATA_FIELDS = ("source_name", "source_id", "extra_columns")

#: Optional keys forwarded verbatim to the nested-only constructor (no
#: precision recast; they keep their incoming dtype). After flat-key packing
#: these are the only non-core keys the constructor sees.
_OPTIONAL_KEYS = (
    "spectral_coeffs",
    "morphology",
    "polarization",
    "metadata",
    "spectrum",
)


def _pack_flat_blocks(data: dict[str, Any]) -> dict[str, Any]:
    """Pack flat per-source column keys into the nested sub-dataclasses.

    ``major_arcsec`` / ``minor_arcsec`` / ``pa_deg`` collapse into a
    :class:`PointMorphology` (all-or-none); ``rotation_measure`` into a
    :class:`PointPolarization`; ``source_name`` / ``source_id`` /
    ``extra_columns`` into a :class:`PointMetadata`. A pre-built nested block
    in ``data`` wins over its flat counterparts (passing both is a TypeError).

    Runs on a shallow copy of ``data`` so the caller's mapping is never
    mutated; returns a new dict carrying only nested blocks (no flat keys).
    """
    out = dict(data)

    def _pop_flat(names: tuple[str, ...]) -> dict[str, Any]:
        popped: dict[str, Any] = {}
        for name in names:
            if name in out:
                popped[name] = out.pop(name)
        return popped

    morph_flat = _pop_flat(_FLAT_MORPHOLOGY_FIELDS)
    if morph_flat:
        if out.get("morphology") is not None:
            raise TypeError(
                "PointSourceData: pass either 'morphology' or the flat "
                "morphology columns (major_arcsec/minor_arcsec/pa_deg), "
                "not both."
            )
        present = {k for k, v in morph_flat.items() if v is not None}
        if present and len(present) != 3:
            raise ValueError(
                "PointSourceData: major_arcsec, minor_arcsec, pa_deg must "
                "be all set or all None."
            )
        if present:
            out["morphology"] = PointMorphology(
                major_arcsec=morph_flat["major_arcsec"],
                minor_arcsec=morph_flat["minor_arcsec"],
                pa_deg=morph_flat["pa_deg"],
            )

    pol_flat = _pop_flat(_FLAT_POLARIZATION_FIELDS)
    if pol_flat:
        if out.get("polarization") is not None:
            raise TypeError(
                "PointSourceData: pass either 'polarization' or the flat "
                "rotation_measure column, not both."
            )
        rm = pol_flat["rotation_measure"]
        if rm is not None:
            out["polarization"] = PointPolarization(rotation_measure=rm)

    meta_flat = _pop_flat(_FLAT_METADATA_FIELDS)
    if meta_flat:
        if out.get("metadata") is not None:
            raise TypeError(
                "PointSourceData: pass either 'metadata' or the flat metadata "
                "columns (source_name/source_id/extra_columns), not both."
            )
        non_empty = (
            meta_flat.get("source_name") is not None
            or meta_flat.get("source_id") is not None
            or meta_flat.get("extra_columns")
        )
        if non_empty:
            out["metadata"] = PointMetadata(
                source_name=meta_flat.get("source_name"),
                source_id=meta_flat.get("source_id"),
                extra_columns=meta_flat.get("extra_columns") or {},
            )

    return out


def point_source_data_from_mapping(
    data: dict[str, Any],
    precision: PrecisionConfig,
) -> PointSourceData:
    """Build a :class:`PointSourceData` from a column mapping at ``precision``.

    This is the single column-oriented construction route for point sources.
    The raw :class:`PointSourceData` constructor is nested-only; this helper
    packs flat per-source columns into the matching nested sub-dataclass
    (:func:`_pack_flat_blocks`) before constructing.

    Core columns are cast to the precision dtypes used by
    ``create_from_arrays``:

    * ``ra_rad`` / ``dec_rad`` → ``sky_model.source_positions``
    * ``flux`` / ``stokes_q`` / ``stokes_u`` / ``stokes_v`` / ``ref_freq``
      → ``sky_model.flux``
    * ``spectral_index`` → ``sky_model.spectral_index``

    Flat optional columns (``rotation_measure``, ``major_arcsec``,
    ``minor_arcsec``, ``pa_deg``, ``source_name``, ``source_id``,
    ``extra_columns``) are packed into :class:`PointMorphology` /
    :class:`PointPolarization` / :class:`PointMetadata` (morphology is
    all-or-none). ``spectral_coeffs`` and pre-built nested blocks
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
    packed = _pack_flat_blocks(data)

    src_dt = precision.sky_model.get_dtype("source_positions")
    flux_dt = precision.sky_model.get_dtype("flux")
    si_dt = precision.sky_model.get_dtype("spectral_index")

    kwargs: dict[str, Any] = {}
    for key in _POSITION_KEYS:
        kwargs[key] = np.asarray(packed[key], dtype=src_dt)
    for key in _FLUX_KEYS:
        kwargs[key] = np.asarray(packed[key], dtype=flux_dt)
    for key in _SPECTRAL_INDEX_KEYS:
        kwargs[key] = np.asarray(packed[key], dtype=si_dt)

    for key in _OPTIONAL_KEYS:
        if key in packed and packed[key] is not None:
            kwargs[key] = packed[key]

    return PointSourceData(**kwargs)
