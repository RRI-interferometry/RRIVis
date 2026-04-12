# rrivis/core/sky/_data.py
"""Inner data containers for SkyModel.

This module is a LEAF dependency — it imports only numpy and healpy.
No imports from model.py, convert.py, combine.py, or loaders.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypedDict

import healpy as hp
import numpy as np

# =============================================================================
# SourceArrays TypedDict
# =============================================================================


class SourceArrays(TypedDict):
    """Return type for point-source array extraction.

    Keys match the interface consumed by ``visibility.py`` and align with
    ``PointSourceData`` field names.
    """

    ra_rad: np.ndarray
    dec_rad: np.ndarray
    flux: np.ndarray
    spectral_index: np.ndarray
    stokes_q: np.ndarray
    stokes_u: np.ndarray
    stokes_v: np.ndarray
    ref_freq: np.ndarray
    rotation_measure: np.ndarray | None
    major_arcsec: np.ndarray | None
    minor_arcsec: np.ndarray | None
    pa_deg: np.ndarray | None
    spectral_coeffs: np.ndarray | None


def empty_source_arrays() -> SourceArrays:
    """Return an empty ``SourceArrays`` dict (zero-length float64 arrays)."""
    z = np.zeros(0, dtype=np.float64)
    return {
        "ra_rad": z.copy(),
        "dec_rad": z.copy(),
        "flux": z.copy(),
        "spectral_index": z.copy(),
        "stokes_q": z.copy(),
        "stokes_u": z.copy(),
        "stokes_v": z.copy(),
        "ref_freq": z.copy(),
        "rotation_measure": None,
        "major_arcsec": None,
        "minor_arcsec": None,
        "pa_deg": None,
        "spectral_coeffs": None,
    }


# =============================================================================
# PointSourceData
# =============================================================================


@dataclass(frozen=True)
class PointSourceData:
    """Columnar arrays for point-source sky model.

    All core arrays have shape ``(N,)``.  This is always fully populated
    (even if zero-length for an empty sky).  No field is ever None for
    the core arrays — an empty model uses zero-length arrays.

    Optional extension arrays (rotation_measure, morphology, spectral_coeffs)
    are None when that feature is absent for ALL sources.
    """

    ra_rad: np.ndarray
    dec_rad: np.ndarray
    flux: np.ndarray
    spectral_index: np.ndarray
    stokes_q: np.ndarray
    stokes_u: np.ndarray
    stokes_v: np.ndarray
    ref_freq: np.ndarray

    # Optional per-source extensions
    rotation_measure: np.ndarray | None = None
    major_arcsec: np.ndarray | None = None
    minor_arcsec: np.ndarray | None = None
    pa_deg: np.ndarray | None = None
    spectral_coeffs: np.ndarray | None = None  # shape (N, N_terms) or None

    def __post_init__(self) -> None:
        """Validate array consistency."""
        n = len(self.ra_rad)
        core_fields = {
            "ra_rad": self.ra_rad,
            "dec_rad": self.dec_rad,
            "flux": self.flux,
            "spectral_index": self.spectral_index,
            "stokes_q": self.stokes_q,
            "stokes_u": self.stokes_u,
            "stokes_v": self.stokes_v,
            "ref_freq": self.ref_freq,
        }
        for name, arr in core_fields.items():
            if len(arr) != n:
                raise ValueError(
                    f"PointSourceData: {name} has length {len(arr)}, "
                    f"expected {n} (must match ra_rad)."
                )

        # Morphology: all-or-none
        morph = (self.major_arcsec, self.minor_arcsec, self.pa_deg)
        morph_present = sum(1 for m in morph if m is not None)
        if morph_present not in (0, 3):
            raise ValueError(
                "PointSourceData: major_arcsec, minor_arcsec, pa_deg must be "
                "all set or all None."
            )

        # Optional arrays must match length
        for name, arr in [
            ("rotation_measure", self.rotation_measure),
            ("major_arcsec", self.major_arcsec),
            ("minor_arcsec", self.minor_arcsec),
            ("pa_deg", self.pa_deg),
        ]:
            if arr is not None and len(arr) != n:
                raise ValueError(
                    f"PointSourceData: {name} has length {len(arr)}, expected {n}."
                )
        if self.spectral_coeffs is not None and self.spectral_coeffs.shape[0] != n:
            raise ValueError(
                f"PointSourceData: spectral_coeffs has {self.spectral_coeffs.shape[0]} "
                f"rows, expected {n}."
            )

    @property
    def n_sources(self) -> int:
        """Number of point sources."""
        return len(self.ra_rad)

    @property
    def is_empty(self) -> bool:
        """True if no sources are present."""
        return len(self.ra_rad) == 0

    @classmethod
    def empty(cls) -> PointSourceData:
        """Create an empty PointSourceData (zero-length arrays)."""
        z = np.zeros(0, dtype=np.float64)
        return cls(
            ra_rad=z.copy(),
            dec_rad=z.copy(),
            flux=z.copy(),
            spectral_index=z.copy(),
            stokes_q=z.copy(),
            stokes_u=z.copy(),
            stokes_v=z.copy(),
            ref_freq=z.copy(),
        )

    def masked(self, mask: np.ndarray) -> PointSourceData:
        """Return new instance with boolean mask applied to all arrays.

        Parameters
        ----------
        mask : np.ndarray
            Boolean mask of shape ``(n_sources,)``.

        Returns
        -------
        PointSourceData
        """
        return PointSourceData(
            ra_rad=self.ra_rad[mask],
            dec_rad=self.dec_rad[mask],
            flux=self.flux[mask],
            spectral_index=self.spectral_index[mask],
            stokes_q=self.stokes_q[mask],
            stokes_u=self.stokes_u[mask],
            stokes_v=self.stokes_v[mask],
            ref_freq=self.ref_freq[mask],
            rotation_measure=(
                self.rotation_measure[mask]
                if self.rotation_measure is not None
                else None
            ),
            major_arcsec=(
                self.major_arcsec[mask] if self.major_arcsec is not None else None
            ),
            minor_arcsec=(
                self.minor_arcsec[mask] if self.minor_arcsec is not None else None
            ),
            pa_deg=self.pa_deg[mask] if self.pa_deg is not None else None,
            spectral_coeffs=(
                self.spectral_coeffs[mask] if self.spectral_coeffs is not None else None
            ),
        )

    def as_source_arrays(
        self, flux_limit: float = 0.0, reference_frequency: float = 0.0
    ) -> SourceArrays:
        """Convert to SourceArrays dict for visibility calculation.

        Parameters
        ----------
        flux_limit : float, default 0.0
            Minimum flux in Jy.
        reference_frequency : float, default 0.0
            Fallback reference frequency (used only if ref_freq is all-zero).

        Returns
        -------
        SourceArrays
        """
        if self.is_empty:
            return empty_source_arrays()

        if flux_limit > 0:
            mask = self.flux >= flux_limit
            n = int(mask.sum())
            if n == 0:
                return empty_source_arrays()
        else:
            mask = np.ones(self.n_sources, dtype=bool)
            n = self.n_sources

        return {
            "ra_rad": self.ra_rad[mask],
            "dec_rad": self.dec_rad[mask],
            "flux": self.flux[mask],
            "spectral_index": self.spectral_index[mask],
            "stokes_q": self.stokes_q[mask],
            "stokes_u": self.stokes_u[mask],
            "stokes_v": self.stokes_v[mask],
            "ref_freq": self.ref_freq[mask],
            "rotation_measure": (
                self.rotation_measure[mask]
                if self.rotation_measure is not None
                else None
            ),
            "major_arcsec": (
                self.major_arcsec[mask] if self.major_arcsec is not None else None
            ),
            "minor_arcsec": (
                self.minor_arcsec[mask] if self.minor_arcsec is not None else None
            ),
            "pa_deg": self.pa_deg[mask] if self.pa_deg is not None else None,
            "spectral_coeffs": (
                self.spectral_coeffs[mask] if self.spectral_coeffs is not None else None
            ),
        }

    # Tuple of all per-source 1-D array field names (for iteration).
    _CORE_FIELDS: tuple[str, ...] = (
        "ra_rad",
        "dec_rad",
        "flux",
        "spectral_index",
        "stokes_q",
        "stokes_u",
        "stokes_v",
        "ref_freq",
    )

    _OPTIONAL_FIELDS: tuple[str, ...] = (
        "rotation_measure",
        "major_arcsec",
        "minor_arcsec",
        "pa_deg",
    )


# =============================================================================
# HealpixData
# =============================================================================


@dataclass(frozen=True)
class HealpixData:
    """Multi-frequency HEALPix brightness temperature maps.

    All Stokes maps have shape ``(n_freq, npix)`` where
    ``npix = hp.nside2npix(nside)``.  The ``frequencies`` array
    provides the frequency axis in Hz.
    """

    maps: np.ndarray  # Stokes I, shape (n_freq, npix), in Kelvin
    nside: int
    frequencies: np.ndarray  # shape (n_freq,), in Hz

    q_maps: np.ndarray | None = None
    u_maps: np.ndarray | None = None
    v_maps: np.ndarray | None = None

    def __post_init__(self) -> None:
        """Validate array shapes."""
        expected_npix = hp.nside2npix(self.nside)
        if self.maps.ndim != 2:
            raise ValueError(
                f"HealpixData: maps must be 2-D (n_freq, npix), "
                f"got shape {self.maps.shape}"
            )
        if self.maps.shape[1] != expected_npix:
            raise ValueError(
                f"HealpixData: maps has {self.maps.shape[1]} pixels per map, "
                f"expected {expected_npix} for nside={self.nside}"
            )
        n_freq = self.maps.shape[0]
        if len(self.frequencies) != n_freq:
            raise ValueError(
                f"HealpixData: frequencies has {len(self.frequencies)} entries "
                f"but maps has {n_freq} frequency channels."
            )
        for name, arr in [
            ("q_maps", self.q_maps),
            ("u_maps", self.u_maps),
            ("v_maps", self.v_maps),
        ]:
            if arr is not None and arr.shape != self.maps.shape:
                raise ValueError(
                    f"HealpixData: {name} shape {arr.shape} does not match "
                    f"maps shape {self.maps.shape}"
                )

    @property
    def n_frequencies(self) -> int:
        """Number of frequency channels."""
        return len(self.frequencies)

    @property
    def n_pixels(self) -> int:
        """Number of HEALPix pixels per map."""
        return self.maps.shape[1]

    @property
    def pixel_solid_angle(self) -> float:
        """Solid angle per pixel in steradians."""
        return 4 * np.pi / hp.nside2npix(self.nside)

    @property
    def has_polarization(self) -> bool:
        """True if any Stokes Q/U/V maps are populated."""
        return any(m is not None for m in (self.q_maps, self.u_maps, self.v_maps))

    def masked_region(self, healpix_mask: np.ndarray) -> HealpixData:
        """Return new HealpixData with out-of-region pixels zeroed.

        Parameters
        ----------
        healpix_mask : np.ndarray
            Boolean mask of shape ``(npix,)`` — True for pixels to keep.

        Returns
        -------
        HealpixData
        """
        inv_mask = ~healpix_mask
        new_maps = self.maps.copy()
        new_maps[:, inv_mask] = 0.0

        new_q = None
        new_u = None
        new_v = None
        if self.q_maps is not None:
            new_q = self.q_maps.copy()
            new_q[:, inv_mask] = 0.0
        if self.u_maps is not None:
            new_u = self.u_maps.copy()
            new_u[:, inv_mask] = 0.0
        if self.v_maps is not None:
            new_v = self.v_maps.copy()
            new_v[:, inv_mask] = 0.0

        return HealpixData(
            maps=new_maps,
            nside=self.nside,
            frequencies=self.frequencies,
            q_maps=new_q,
            u_maps=new_u,
            v_maps=new_v,
        )
