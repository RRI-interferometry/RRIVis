"""Pyradiosky file loader functions for SkyModel.

Point-source ``region`` filtering uses the client-side convention documented
in :mod:`radiosim.core.sky.support.region_filter`. HEALPix payloads crop to
region masks during cube assembly in :mod:`radiosim.core.sky.loaders._healpix_builder`.
"""

from __future__ import annotations

import logging
import os
import warnings
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Literal

import astropy.units as u
import numpy as np

from ..containers.model import SkyModel
from ..containers.point import PointSpectrum
from ..registry import loader_registry
from ..support.frequencies import validate_observation_frequencies
from ..support.healpix_geometry import ordered_row
from ..support.healpy import lazy_healpy as hp
from ..support.point_builder import point_source_data_from_mapping
from ..support.quantities import to_value
from ..support.region_filter import apply_point_region_filter
from ._healpix_builder import build_healpix_from_stokes_cube, extract_stokes_component

if TYPE_CHECKING:
    from radiosim.core.precision import PrecisionConfig

    from ..containers import SkyProvenance
    from ..operations.region import SkyRegion

logger = logging.getLogger(__name__)

#: Lazily-populated ``pyradiosky.SkyModel`` class. Kept as a module attribute
#: (not a top-level import) so the sky package imports cleanly even when the
#: optional ``pyradiosky`` dependency is absent; tests may monkeypatch this.
PyRadioSkyModel: Any = None


def _pyradiosky_cls() -> Any:
    """Return the ``pyradiosky.SkyModel`` class, importing it on first use."""
    global PyRadioSkyModel
    if PyRadioSkyModel is None:
        try:
            from pyradiosky import SkyModel as _cls
        except ImportError as exc:
            raise ImportError(
                "Reading pyradiosky files requires the optional 'pyradiosky' "
                "package. Install it with `pip install pyradiosky`."
            ) from exc
        PyRadioSkyModel = _cls
    return PyRadioSkyModel


class LossyConversionWarning(UserWarning):
    """Warn when pyradiosky import drops higher-order spectral information."""


@loader_registry.register_loader(
    "pyradiosky_file",
    config_section="pyradiosky",
    use_flag="use_pyradiosky",
    category="file",
    requires_file=True,
    network_service=None,
    # pyradiosky files can carry point or HEALPix payloads; declare both
    # explicitly rather than relying on a loader-name special-case in the
    # registry.
    representations=("point_sources", "healpix_map"),
    aliases=["pyradiosky"],
    config_fields=[
        "filename",
        "filetype",
        "flux_limit",
        "reference_frequency_hz",
        "spectral_loss_policy",
    ],
)
def load_pyradiosky_file(
    filename: str,
    filetype: str | None = None,
    flux_limit: float = 0.0,
    reference_frequency_hz: float | None = None,
    spectral_loss_policy: Literal["warn", "error"] = "warn",
    brightness_conversion: str = "planck",
    *,
    precision: PrecisionConfig,
    frequencies: Sequence[float] | np.ndarray | None = None,
    region: SkyRegion | None = None,
    memmap_path: str | None = None,
    provenance: SkyProvenance | None = None,
) -> SkyModel:
    """
    Load a local sky model file via pyradiosky.

    Supports SkyH5, VOTable, text, and FHD formats (as handled by
    pyradiosky). Both ``component_type='point'`` and
    ``component_type='healpix'`` are supported.

    For HEALPix files, observation frequencies can be provided explicitly via
    ``frequencies``. If the file has ``spectral_type='full'`` or ``'subband'``
    and no explicit frequencies are given, the file's own frequency array is
    used.

    Parameters
    ----------
    filename : str
        Path to the sky model file.
    filetype : str, optional
        File format: "skyh5", "votable", "text", "fhd", etc.
        If None, pyradiosky infers from the file extension.
    flux_limit : float, default=0.0
        Minimum Stokes I flux in Jy at the reference frequency.
        Only used for point-source files.
    reference_frequency_hz : float, optional
        Reference frequency for Stokes I extraction (Hz).
        If None, uses the first frequency channel in the file.
        Only used for point-source files.
    brightness_conversion : str, default="planck"
        Conversion method: "planck" or "rayleigh-jeans".
    frequencies : np.ndarray, optional
        Explicit ordered observation frequencies in Hz for HEALPix files.

    Returns
    -------
    SkyModel

    Raises
    ------
    FileNotFoundError
        If ``filename`` does not exist.
    ValueError
        If the file has an unsupported ``component_type``, or if a
        HEALPix file with ``spectral_type='spectral_index'`` or
        ``'flat'`` is loaded without explicit frequencies.
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Sky model file not found: {filename}")

    sky = _pyradiosky_cls()()
    sky.read(filename, filetype=filetype)

    if sky.component_type == "healpix":
        sky_out = _load_pyradiosky_healpix(
            sky,
            filename,
            frequencies,
            brightness_conversion,
            precision,
            region=region,
            memmap_path=memmap_path,
        )
        if provenance is not None:
            sky_out = sky_out.replace(provenance=provenance)
        return sky_out
    elif sky.component_type != "point":
        raise ValueError(
            f"Unsupported component_type: '{sky.component_type}'. "
            "Only 'point' and 'healpix' are supported."
        )

    ref_freq_hz = reference_frequency_hz
    if ref_freq_hz is None:
        if sky.freq_array is not None and len(sky.freq_array) > 0:
            ref_freq_hz = float(to_value(sky.freq_array, u.Hz)[0])
        elif sky.reference_frequency is not None and len(sky.reference_frequency) > 0:
            # spectral_index type: frequency stored per-component
            ref_freq_hz = float(to_value(sky.reference_frequency, u.Hz)[0])
        else:
            raise ValueError(
                "Cannot determine reference frequency. "
                "Provide reference_frequency_hz explicitly."
            )

    if sky.freq_array is not None and len(sky.freq_array) > 1:
        freq_vals = np.asarray(to_value(sky.freq_array, u.Hz), dtype=np.float64)
        ref_chan_idx = int(np.argmin(np.abs(freq_vals - ref_freq_hz)))
    else:
        ref_chan_idx = 0

    # stokes shape: (4, Nfreqs, Ncomponents) or (4, 1, Ncomponents)
    stokes = np.asarray(to_value(sky.stokes, u.Jy), dtype=np.float64)
    stokes_i_ref = np.array(stokes[0, ref_chan_idx, :], dtype=np.float64)

    n_stokes = stokes.shape[0]
    stokes_q = (
        np.array(stokes[1, ref_chan_idx, :], dtype=np.float64)
        if n_stokes > 1
        else np.zeros_like(stokes_i_ref)
    )
    stokes_u = (
        np.array(stokes[2, ref_chan_idx, :], dtype=np.float64)
        if n_stokes > 2
        else np.zeros_like(stokes_i_ref)
    )
    stokes_v = (
        np.array(stokes[3, ref_chan_idx, :], dtype=np.float64)
        if n_stokes > 3
        else np.zeros_like(stokes_i_ref)
    )

    point_spectrum: PointSpectrum | None = None
    if sky.spectral_type == "spectral_index":
        spectral_indices = np.asarray(sky.spectral_index, dtype=np.float64)
    elif sky.spectral_type == "flat":
        spectral_indices = np.zeros(sky.Ncomponents, dtype=np.float64)
    else:
        # "full" / "subband": preserve the per-channel spectrum losslessly in a
        # PointSpectrum (downstream materialization uses nearest-channel lookup)
        # rather than collapsing it to a single fitted index. A fitted index is
        # still stored as a fallback for code paths that ignore the spectrum.
        if sky.freq_array is not None and len(sky.freq_array) >= 2:
            freq_vals = np.asarray(to_value(sky.freq_array, u.Hz), dtype=np.float64)
            order = np.argsort(freq_vals)  # PointSpectrum requires ascending
            sorted_freqs = freq_vals[order]
            spec_i = np.asarray(stokes[0], dtype=np.float64)[order]
            spec_q = (
                np.asarray(stokes[1], dtype=np.float64)[order] if n_stokes > 1 else None
            )
            spec_u = (
                np.asarray(stokes[2], dtype=np.float64)[order] if n_stokes > 2 else None
            )
            spec_v = (
                np.asarray(stokes[3], dtype=np.float64)[order] if n_stokes > 3 else None
            )
            # PointSpectrum pairs Q and U (both or neither).
            if (spec_q is None) != (spec_u is None):
                spec_q = spec_u = None
            point_spectrum = PointSpectrum(
                flux=spec_i,
                frequencies=sorted_freqs,
                stokes_q=spec_q,
                stokes_u=spec_u,
                stokes_v=spec_v,
            )

            s_first = spec_i[0]
            s_last = spec_i[-1]
            spectral_indices = np.zeros(sky.Ncomponents, dtype=np.float64)
            si_valid = (s_first > 0) & (s_last > 0)
            if np.any(si_valid):
                spectral_indices[si_valid] = np.log(
                    s_first[si_valid] / s_last[si_valid]
                ) / np.log(sorted_freqs[0] / sorted_freqs[-1])
        else:
            spectral_indices = np.zeros(sky.Ncomponents, dtype=np.float64)

        if point_spectrum is None:
            # No usable channel axis to preserve — this is the genuinely lossy
            # case the policy guards.
            message = (
                "Loading a pyradiosky point file with spectral_type="
                f"{sky.spectral_type!r} but no usable multi-channel frequency "
                "axis; the spectrum cannot be preserved."
            )
            if spectral_loss_policy == "error":
                raise ValueError(message)
            warnings.warn(message, LossyConversionWarning, stacklevel=2)

    # Build per-source reference frequency array
    per_source_ref_freq = None
    if sky.spectral_type == "spectral_index" and sky.reference_frequency is not None:
        per_source_ref_freq = np.asarray(
            to_value(sky.reference_frequency, u.Hz), dtype=np.float64
        )

    ra_arr = np.array(to_value(sky.ra, u.rad), dtype=np.float64)
    dec_arr = np.array(to_value(sky.dec, u.rad), dtype=np.float64)
    source_name = None
    if getattr(sky, "name", None) is not None:
        source_name = np.asarray(sky.name)

    source_id = None
    extra_columns: dict[str, np.ndarray] = {}
    extra = getattr(sky, "extra_columns", None)
    if extra is not None and getattr(extra.dtype, "names", None) is not None:
        for name in extra.dtype.names:
            values = np.asarray(extra[name])
            if name == "source_id":
                source_id = values
            else:
                extra_columns[name] = values

    flux_valid = np.isfinite(stokes_i_ref) & (stokes_i_ref >= flux_limit)
    cols = apply_point_region_filter(
        {
            "ra_rad": ra_arr[flux_valid],
            "dec_rad": dec_arr[flux_valid],
            "flux": stokes_i_ref[flux_valid],
            "spectral_index": spectral_indices[flux_valid],
            "stokes_q": stokes_q[flux_valid],
            "stokes_u": stokes_u[flux_valid],
            "stokes_v": stokes_v[flux_valid],
            "ref_freq": (
                per_source_ref_freq[flux_valid]
                if per_source_ref_freq is not None
                else np.full(int(flux_valid.sum()), ref_freq_hz, dtype=np.float64)
            ),
            "source_name": (
                source_name[flux_valid] if source_name is not None else None
            ),
            "source_id": source_id[flux_valid] if source_id is not None else None,
            **{name: values[flux_valid] for name, values in extra_columns.items()},
        },
        region,
    )
    n = len(cols["flux"])

    model_name = f"pyradiosky:{os.path.basename(filename)}"
    logger.info(f"pyradiosky file loaded: {n:,} sources from {filename}")

    if n == 0:
        from ..operations.factories import create_empty

        return create_empty(
            model_name,
            brightness_conversion,
            precision=precision,
            reference_frequency=ref_freq_hz,
        )

    spectrum = None
    if point_spectrum is not None:
        spectrum = point_spectrum.masked_sources(flux_valid)
        if region is not None and n < len(spectrum.flux):
            region_on_flux = region.contains(ra_arr[flux_valid], dec_arr[flux_valid])
            spectrum = spectrum.masked_sources(region_on_flux)

    sky_model = SkyModel(
        point=point_source_data_from_mapping(
            {
                "ra_rad": cols["ra_rad"],
                "dec_rad": cols["dec_rad"],
                "flux": cols["flux"],
                "spectral_index": cols["spectral_index"],
                "stokes_q": cols["stokes_q"],
                "stokes_u": cols["stokes_u"],
                "stokes_v": cols["stokes_v"],
                "ref_freq": cols["ref_freq"],
                "source_name": cols["source_name"],
                "source_id": cols["source_id"],
                "extra_columns": {name: cols[name] for name in extra_columns},
                "spectrum": spectrum,
            },
            precision=precision,
        ),
        model_name=model_name,
        reference_frequency=ref_freq_hz,
        brightness_conversion=brightness_conversion,
        precision=precision,
    )
    if provenance is not None:
        sky_model = sky_model.replace(provenance=provenance)
    return sky_model


def _load_pyradiosky_healpix(
    psky: Any,
    filename: str,
    frequencies: Sequence[float] | np.ndarray | None,
    brightness_conversion: str,
    precision: PrecisionConfig,
    region: SkyRegion | None = None,
    memmap_path: str | None = None,
) -> SkyModel:
    """
    Load a pyradiosky HEALPix sky model as multi-frequency HEALPix maps.

    This is called internally by ``load_pyradiosky_file()`` when the file
    has ``component_type='healpix'``.

    Parameters
    ----------
    psky : pyradiosky.SkyModel
        Already-read pyradiosky SkyModel with ``component_type='healpix'``.
    filename : str
        Original file path (for logging and model_name).
    frequencies : np.ndarray or None
        Explicit observation frequencies in Hz.
    brightness_conversion : str
        Conversion method: "planck" or "rayleigh-jeans".
    precision : PrecisionConfig
        Precision configuration.

    Returns
    -------
    SkyModel
        Sky model in healpix_map mode.

    Raises
    ------
    ValueError
        If frequencies cannot be determined.
    """
    # --- Determine observation frequencies ---
    # The file's own frequency array remains authoritative when the caller
    # does not provide an explicit materialization axis.
    if frequencies is not None:
        obs_freqs = validate_observation_frequencies(
            frequencies,
            label="load_pyradiosky_file frequencies",
        )
    elif (
        psky.spectral_type in ("full", "subband")
        and psky.freq_array is not None
        and len(psky.freq_array) > 0
    ):
        obs_freqs = validate_observation_frequencies(
            to_value(psky.freq_array, u.Hz),
            label="pyradiosky file frequencies",
        )
    else:
        raise ValueError(
            f"Cannot determine observation frequencies for HEALPix file "
            f"with spectral_type='{psky.spectral_type}'. "
            "Provide 'frequencies' explicitly."
        )

    n_freq = len(obs_freqs)
    nside = psky.nside

    logger.info(
        f"Loading pyradiosky HEALPix file: {n_freq} frequencies "
        f"({obs_freqs[0] / 1e6:.1f}\u2013{obs_freqs[-1] / 1e6:.1f} MHz), "
        f"nside={nside}, from {filename}"
    )

    # --- Evaluate stokes at observation frequencies ---
    psky_eval = psky.at_frequencies(obs_freqs * u.Hz, inplace=False)

    # --- Convert units to Kelvin if needed ---
    if psky_eval.stokes.unit.is_equivalent(u.Jy / u.sr):
        psky_eval.jansky_to_kelvin()

    # --- Determine coordinate frame and pixel handling ---
    coordinate_frame = "icrs"
    if hasattr(psky_eval, "frame") and psky_eval.frame is not None:
        frame_name = str(psky_eval.frame).lower()
        if "galactic" in frame_name:
            coordinate_frame = "galactic"
        elif "icrs" not in frame_name:
            raise ValueError(
                "Unsupported pyradiosky HEALPix frame. "
                f"Expected ICRS or Galactic, got {psky_eval.frame!r}."
            )

    # Check for nested ordering
    is_nested = False
    if hasattr(psky_eval, "hpx_order") and psky_eval.hpx_order is not None:
        if psky_eval.hpx_order.lower() == "nested":
            is_nested = True

    # Check for sparse map (partial sky)
    hpx_inds = None
    if hasattr(psky_eval, "hpx_inds") and psky_eval.hpx_inds is not None:
        hpx_inds = np.asarray(psky_eval.hpx_inds)

    # --- Extract Stokes and build full-sky maps ---
    # psky_eval.stokes shape: (n_stokes, Nfreqs, Ncomponents)
    stokes_data = np.asarray(psky_eval.stokes.value)
    n_stokes_avail = stokes_data.shape[0]
    has_pol = n_stokes_avail >= 3

    if has_pol:
        if brightness_conversion != "rayleigh-jeans":
            logger.info(
                "Using Rayleigh-Jeans conversion (required: polarized K_RJ data)"
            )
        brightness_conversion = "rayleigh-jeans"

    pix = None
    if hpx_inds is not None:
        pix = np.array(hpx_inds, copy=True)
        if is_nested:
            pix = hp.nest2ring(nside, pix)

    npix = hp.nside2npix(nside)
    builder_hpx_inds = (
        pix if pix is not None and (len(pix) < npix or region is not None) else None
    )

    def _ring_ordered_row(data_1d: np.ndarray) -> np.ndarray:
        return ordered_row(
            data_1d,
            builder_handles_scatter=builder_hpx_inds is not None,
            pix=pix,
            npix=npix,
            is_nested=is_nested,
        )

    def _iter_stokes_rows():
        for fi in range(n_freq):
            i_row = extract_stokes_component(stokes_data[:, fi, :], "I", n_stokes_avail)
            if i_row is None:
                raise ValueError("pyradiosky HEALPix data is missing Stokes I.")
            i_map = _ring_ordered_row(i_row)
            q_map = (
                _ring_ordered_row(
                    extract_stokes_component(stokes_data[:, fi, :], "Q", n_stokes_avail)
                )
                if has_pol
                else None
            )
            u_map = (
                _ring_ordered_row(
                    extract_stokes_component(stokes_data[:, fi, :], "U", n_stokes_avail)
                )
                if has_pol
                else None
            )
            v_row = extract_stokes_component(stokes_data[:, fi, :], "V", n_stokes_avail)
            v_map = _ring_ordered_row(v_row) if v_row is not None else None
            yield (i_map, q_map, u_map, v_map)

    healpix = build_healpix_from_stokes_cube(
        stokes_rows=_iter_stokes_rows(),
        nside=nside,
        frequencies=obs_freqs,
        coordinate_frame=coordinate_frame,
        hpx_inds=builder_hpx_inds,
        region=region,
        precision=precision,
        memmap_path=memmap_path,
    )

    model_name = f"pyradiosky:{os.path.basename(filename)}"
    stokes_label = "I"
    if has_pol:
        stokes_label = "IQU" + ("V" if n_stokes_avail >= 4 else "")
    logger.info(
        f"pyradiosky HEALPix loaded: {healpix.n_pixels} pixels \u00d7 {n_freq} frequencies, "
        f"stokes={stokes_label}"
    )

    return SkyModel(
        healpix=healpix,
        model_name=model_name,
        brightness_conversion=brightness_conversion,
        precision=precision,
    )
