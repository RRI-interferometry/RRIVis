"""Serialization helpers for SkyModel (SkyH5 via pyradiosky).

Extracted from model.py to keep SkyModel focused on data access.
"""

from __future__ import annotations

import logging
import warnings
from typing import TYPE_CHECKING, Any

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord

if TYPE_CHECKING:
    from radiosim.core.precision import PrecisionConfig

    from ..containers.model import SkyModel

logger = logging.getLogger(__name__)

# Schema version of the ``radiosim_provenance`` HDF5 attribute payload. Bump
# this when the on-disk provenance encoding changes incompatibly; readers warn
# on a missing or unrecognised version so drift surfaces instead of silently
# producing a half-decoded SkyProvenance.
PROVENANCE_SCHEMA_VERSION = 1


def _resolve_serialization_format(
    sky: SkyModel,
    representation: Any,
) -> Any:
    from ..containers.model import SkyFormat

    if representation is not None:
        target = (
            SkyFormat(representation)
            if isinstance(representation, str)
            else representation
        )
        if target not in sky.formats:
            raise ValueError(
                f"Cannot serialize as {target.value!r}; payload is not available. "
                f"Available: {[fmt.value for fmt in sky.formats]}"
            )
        return target
    if len(sky.formats) == 1:
        return next(iter(sky.formats))
    raise ValueError(
        "SkyModel contains both point and HEALPix payloads. "
        "Pass representation='point_sources' or representation='healpix_map' "
        "to to_pyradiosky() or save_skyh5()."
    )


def _sanitize_extra_column(values: np.ndarray, name: str | None = None) -> np.ndarray:
    """Convert metadata columns to pyradiosky-friendly array dtypes.

    Object columns that are uniformly string-like or uniformly numeric are
    converted losslessly.  A column that mixes string and numeric (or other)
    types cannot be represented as a single non-object pyradiosky array, so it
    is stringified — and a :class:`UserWarning` is emitted, because that path
    loses the original element types.
    """
    arr = np.asarray(values)
    if arr.dtype.kind != "O":
        return arr

    flat = arr.tolist()
    present = [value for value in flat if value is not None]
    if not present:
        return np.full(len(arr), "", dtype=str)
    if all(isinstance(value, (str, bytes, np.str_)) for value in present):
        return np.asarray(
            ["" if value is None else str(value) for value in flat],
            dtype=str,
        )
    if all(
        isinstance(
            value,
            (
                bool,
                int,
                float,
                np.bool_,
                np.integer,
                np.floating,
            ),
        )
        for value in present
    ):
        return np.asarray(
            [np.nan if value is None else float(value) for value in flat],
            dtype=np.float64,
        )
    label = f" {name!r}" if name is not None else ""
    warnings.warn(
        f"to_pyradiosky: extra metadata column{label} mixes string and numeric "
        "(or other) types; stringifying every element to fit pyradiosky's "
        "extra_column storage. Original element types are not recoverable on "
        "round-trip.",
        UserWarning,
        stacklevel=2,
    )
    return np.asarray(
        ["" if value is None else str(value) for value in flat],
        dtype=str,
    )


def _warn_dropped_point_fields(point: Any) -> list[str]:
    """Warn about point-source physics that the pyradiosky export cannot carry.

    pyradiosky's point model has no representation for Faraday rotation
    measure or Gaussian morphology, and (absent a per-channel spectrum, which
    is exported losslessly as ``spectral_type="full"``) it cannot carry the
    higher-order terms of a log-polynomial spectral model.  These fields are
    silently lost on export today; this surfaces them as an explicit warning
    naming exactly what was dropped.

    Returns the list of dropped field labels (also useful for tests).
    """
    dropped: list[str] = []

    polarization = getattr(point, "polarization", None)
    rotation_measure = (
        polarization.rotation_measure if polarization is not None else None
    )
    if rotation_measure is not None and np.any(rotation_measure != 0):
        dropped.append("rotation measure (RM)")

    morphology = getattr(point, "morphology", None)
    major_arcsec = morphology.major_arcsec if morphology is not None else None
    if major_arcsec is not None and np.any(major_arcsec > 0):
        dropped.append("Gaussian morphology")

    # Higher-order spectral terms only survive via the per-channel ("full")
    # export path; with no spectrum we collapse to a single power-law index.
    if point.spectrum is None:
        spectral_coeffs = getattr(point, "spectral_coeffs", None)
        if spectral_coeffs is not None and spectral_coeffs.shape[1] > 1:
            dropped.append("higher-order (log-polynomial) spectral coefficients")

    if dropped:
        warnings.warn(
            "to_pyradiosky: dropping point-source fields not representable in "
            f"pyradiosky: {', '.join(dropped)}. Use "
            "radiosim.core.sky.write_bbs() for lossless export.",
            UserWarning,
            stacklevel=2,
        )
    return dropped


def to_pyradiosky(sky: SkyModel, representation: Any = None) -> Any:
    """Convert a SkyModel to a pyradiosky.SkyModel for serialization.

    ``representation`` is required when both point and HEALPix payloads
    are populated.
    """
    from pyradiosky import SkyModel as PyRadioSkyModel

    from ..containers.model import SkyFormat

    target = _resolve_serialization_format(sky, representation)

    if target == SkyFormat.HEALPIX:
        if sky.healpix is None:
            raise ValueError("Cannot serialize missing HEALPix SkyModel payload.")
        healpix = sky.healpix
        healpix_maps = healpix.maps
        nside = healpix.nside
        observation_frequencies = healpix.frequencies
        npix = healpix.n_pixels
        sorted_indices = np.argsort(observation_frequencies)
        sorted_freqs = observation_frequencies[sorted_indices]
        n_freq = len(sorted_freqs)

        # Honor the model's storage precision instead of forcing float32 — a
        # user on PrecisionConfig.precise()/ultra() would otherwise silently
        # lose half their mantissa on every HEALPix round-trip.
        stokes_arr = np.zeros((4, n_freq, npix), dtype=healpix_maps.dtype)
        q_maps = healpix.q_maps
        u_maps = healpix.u_maps
        v_maps = healpix.v_maps
        for out_i, src_i in enumerate(sorted_indices):
            stokes_arr[0, out_i, :] = healpix_maps[src_i]
            if q_maps is not None:
                stokes_arr[1, out_i, :] = q_maps[src_i]
            if u_maps is not None:
                stokes_arr[2, out_i, :] = u_maps[src_i]
            if v_maps is not None:
                stokes_arr[3, out_i, :] = v_maps[src_i]

        from astropy.coordinates import ICRS, Galactic

        frame = Galactic() if healpix.coordinate_frame == "galactic" else ICRS()

        return PyRadioSkyModel(
            nside=nside,
            hpx_order=healpix.ordering,
            hpx_inds=(
                healpix.hpx_inds if healpix.hpx_inds is not None else np.arange(npix)
            ),
            stokes=stokes_arr * u.K,
            spectral_type="full",
            freq_array=sorted_freqs * u.Hz,
            component_type="healpix",
            frame=frame,
            history=f"RadioSim SkyModel: {sky.model_name or 'unknown'}, "
            f"brightness_conversion={sky.brightness_conversion}",
        )

    if sky.point is not None and not sky.point.is_empty:
        point = sky.point
        n = point.n_sources
        metadata = point.metadata
        source_name = metadata.source_name if metadata is not None else None
        if source_name is not None:
            names = np.asarray(
                ["" if value is None else str(value) for value in source_name],
                dtype=str,
            )
        else:
            names = np.full(n, "", dtype=str)

        skycoord = SkyCoord(
            ra=point.ra_rad, dec=point.dec_rad, unit="rad", frame="icrs"
        )

        extra_column_dict = {
            name: _sanitize_extra_column(values, name)
            for name, values in (
                metadata.extra_columns.items() if metadata is not None else ()
            )
        }
        source_id = metadata.source_id if metadata is not None else None
        if source_id is not None and "source_id" not in extra_column_dict:
            extra_column_dict["source_id"] = _sanitize_extra_column(
                source_id, "source_id"
            )

        _warn_dropped_point_fields(point)

        common = {
            "name": names,
            "skycoord": skycoord,
            "component_type": "point",
            "extra_column_dict": extra_column_dict or None,
            "history": (
                f"RadioSim SkyModel: {sky.model_name or 'unknown'}, "
                f"brightness_conversion={sky.brightness_conversion}"
            ),
        }

        spectrum = point.spectrum
        if spectrum is not None:
            # Lossless multi-frequency export: pyradiosky supports a per-channel
            # ("full") spectral type for point components, so emit the whole
            # Stokes-flux table instead of collapsing it to a power-law index.
            sorted_indices = np.argsort(spectrum.frequencies)
            sorted_freqs = spectrum.frequencies[sorted_indices]
            n_freq = len(sorted_freqs)
            stokes_arr = np.zeros((4, n_freq, n), dtype=spectrum.flux.dtype)
            stokes_arr[0] = spectrum.flux[sorted_indices]
            if spectrum.stokes_q is not None:
                stokes_arr[1] = spectrum.stokes_q[sorted_indices]
            if spectrum.stokes_u is not None:
                stokes_arr[2] = spectrum.stokes_u[sorted_indices]
            if spectrum.stokes_v is not None:
                stokes_arr[3] = spectrum.stokes_v[sorted_indices]
            return PyRadioSkyModel(
                stokes=stokes_arr * u.Jy,
                spectral_type="full",
                freq_array=sorted_freqs * u.Hz,
                **common,
            )

        stokes_arr = np.zeros((4, 1, n), dtype=np.float64)
        stokes_arr[0, 0, :] = point.flux
        stokes_arr[1, 0, :] = point.stokes_q
        stokes_arr[2, 0, :] = point.stokes_u
        stokes_arr[3, 0, :] = point.stokes_v

        ref_freq = point.ref_freq
        if ref_freq is not None and len(ref_freq) == n:
            ref_freq_arr = ref_freq * u.Hz
        else:
            scalar_ref_freq = sky.reference_frequency
            if scalar_ref_freq is None:
                scalar_ref_freq = 1e8
                warnings.warn(
                    "to_pyradiosky: no per-source ref_freq and no model "
                    "reference_frequency; falling back to 100 MHz for the "
                    "exported reference frequency. Set reference_frequency "
                    "explicitly to avoid a silent assumption.",
                    stacklevel=2,
                )
            ref_freq_arr = np.full(n, scalar_ref_freq) * u.Hz

        return PyRadioSkyModel(
            stokes=stokes_arr * u.Jy,
            spectral_type="spectral_index",
            spectral_index=point.spectral_index.copy(),
            reference_frequency=ref_freq_arr,
            **common,
        )

    raise ValueError("Cannot save an empty SkyModel (no sources or maps).")


def save_skyh5(
    sky: SkyModel,
    filename: str,
    *,
    representation: Any = None,
    clobber: bool = False,
    compression: str | None = "gzip",
) -> None:
    """Save a SkyModel to SkyH5 format (HDF5 via pyradiosky).

    Parameters
    ----------
    sky : SkyModel
        The model to save.
    filename : str
        Output file path (typically ``*.skyh5``).
    representation : SkyFormat or str, optional
        Required when both point-source and HEALPix payloads are present.
    clobber : bool, default False
        Overwrite an existing file.
    compression : str or None, default "gzip"
        HDF5 compression for data arrays.

    Notes
    -----
    Point-source fields that pyradiosky cannot represent (rotation measure,
    Gaussian morphology, higher-order log-polynomial spectral terms) are
    dropped; ``to_pyradiosky`` emits a ``UserWarning`` naming exactly what was
    lost. Use ``radiosim.core.sky.write_bbs()`` for lossless export.
    """
    psky = to_pyradiosky(sky, representation=representation)
    psky.write_skyh5(filename, clobber=clobber, data_compression=compression)

    # Stash SkyProvenance as a JSON-encoded HDF5 root attribute so that
    # round-tripping a source-subtracted Haslam template (or any sky with
    # tagged provenance) preserves the source_subtraction / monopole /
    # completeness metadata that pyradiosky does not model.  Non-radiosim
    # readers ignore the attribute; our load_skyh5 reads it back below.
    import json

    import h5py

    payload = {
        "_schema": PROVENANCE_SCHEMA_VERSION,
        "provenance": sky.provenance.to_dict(),
    }
    with h5py.File(filename, "a") as fh:
        fh.attrs["radiosim_provenance"] = json.dumps(payload)

    logger.info(f"SkyModel saved to {filename}")


def _read_skyh5_provenance(filename: str):  # type: ignore[no-untyped-def]
    """Return the SkyProvenance stashed in ``filename`` by ``save_skyh5``.

    Returns ``None`` if the file has no radiosim-written provenance attribute
    (e.g. it was written by a different pyradiosky-based tool).
    """
    import json

    import h5py

    from ..containers import SkyProvenance

    try:
        with h5py.File(filename, "r") as fh:
            raw = fh.attrs.get("radiosim_provenance")
    except OSError:
        return None
    if raw is None:
        return None
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    try:
        payload = json.loads(raw)
    except (TypeError, ValueError):
        logger.warning(
            "Failed to decode radiosim_provenance attribute in %s; provenance "
            "will be reconstructed from defaults.",
            filename,
        )
        return None
    provenance_dict = _unwrap_provenance_payload(payload, filename)
    return SkyProvenance.from_dict(provenance_dict)


def _unwrap_provenance_payload(payload: object, filename: str) -> dict:
    """Validate the ``_schema`` version and return the inner provenance dict.

    The ``radiosim_provenance`` attribute is a JSON object of the form
    ``{"_schema": <int>, "provenance": {...}}`` (see ``save_skyh5``).  A
    missing ``_schema`` key (e.g. a legacy or foreign payload that is itself
    the bare provenance dict) or an unrecognised version is warned about and
    decoded best-effort so a schema bump surfaces loudly instead of silently
    corrupting the round-trip.
    """
    if not isinstance(payload, dict):
        logger.warning(
            "radiosim_provenance attribute in %s is not a JSON object "
            "(got %s); attempting best-effort decode.",
            filename,
            type(payload).__name__,
        )
        return payload if isinstance(payload, dict) else {}

    if "_schema" not in payload:
        warnings.warn(
            f"radiosim_provenance attribute in {filename!r} has no '_schema' "
            f"version field (expected {PROVENANCE_SCHEMA_VERSION}); treating the "
            "payload as an unversioned provenance dict. The file may have been "
            "written by an older or non-radiosim tool.",
            UserWarning,
            stacklevel=3,
        )
        return payload

    schema = payload["_schema"]
    if schema != PROVENANCE_SCHEMA_VERSION:
        warnings.warn(
            f"radiosim_provenance attribute in {filename!r} has unknown "
            f"_schema={schema!r} (this build understands "
            f"{PROVENANCE_SCHEMA_VERSION}); decoding best-effort, some fields "
            "may be missing or misinterpreted.",
            UserWarning,
            stacklevel=3,
        )

    inner = payload.get("provenance")
    if not isinstance(inner, dict):
        logger.warning(
            "radiosim_provenance attribute in %s is missing its 'provenance' "
            "body; provenance will be reconstructed from defaults.",
            filename,
        )
        return {}
    return inner


def load_skyh5(
    filename: str,
    *,
    precision: PrecisionConfig,
    **kwargs: Any,
) -> SkyModel:
    """Load a SkyModel from a SkyH5 file.

    Convenience wrapper around load_pyradiosky_file().  When the file was
    written by ``save_skyh5``, the SkyProvenance round-trips via the
    ``radiosim_provenance`` HDF5 root attribute; for files written by other
    tools the provenance falls back to whatever pyradiosky/loader defaults
    produce.
    """
    from ..loaders.pyradiosky import load_pyradiosky_file

    sky = load_pyradiosky_file(
        filename, filetype="skyh5", precision=precision, **kwargs
    )
    stashed = _read_skyh5_provenance(filename)
    if stashed is not None:
        sky = sky.replace(provenance=stashed)
    return sky
