"""Shared plotting helpers for sky-model visualization.

Module-level helpers (no class hierarchy). Each function takes the
:class:`~radiosim.core.sky.SkyModel` it operates on as its first
argument; the previous ``_SkyPlotterBase`` mixin and its
construction-time ``self._sky`` binding have been removed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from matplotlib.figure import Figure

    from radiosim.core.sky.containers.model import SkyModel


_COLORBAR_LABELS = {
    "flux": "Flux density (Jy)",
    "flux_log": "log$_{10}$(Flux density / Jy)",
    "spectral_index": "Spectral index",
    "stokes_q": "Stokes Q (Jy)",
    "stokes_u": "Stokes U (Jy)",
    "stokes_v": "Stokes V (Jy)",
    "tb": "T$_b$ (K)",
    "tb_log": "log$_{10}$(T$_b$ / K)",
}

_HEALPY_PROJECTIONS = {
    "mollweide": "mollview",
    "cartesian": "cartview",
    "gnomonic": "gnomview",
    "orthographic": "orthview",
}

_MPL_PROJECTIONS = {"mollweide", "aitoff", "hammer", "cartesian"}

_COORD_LABELS = {
    None: ("RA", "Dec"),
    "C": ("RA", "Dec"),
    "G": ("Galactic $l$", "Galactic $b$"),
    "E": ("Ecliptic lon", "Ecliptic lat"),
}


def _freq_label(freq_hz: float) -> str:
    """Format a frequency in Hz for display."""
    if freq_hz >= 1e9:
        return f"{freq_hz / 1e9:.1f} GHz"
    return f"{freq_hz / 1e6:.1f} MHz"


def _auto_title(sky: SkyModel, suffix: str, frequency: float | None = None) -> str:
    """Build a plot title from model metadata."""
    parts: list[str] = []
    if sky.model_name:
        parts.append(sky.model_name.upper().replace("_", " "))
    if frequency is not None:
        parts.append(_freq_label(frequency))
    if sky.healpix is not None:
        parts.append(f"nside={sky.healpix.nside}")
    meta = " — ".join(parts) if parts else "Sky Model"
    return f"{meta} — {suffix}"


def _validate_plot_mode(sky: SkyModel, required: str) -> None:
    """Raise ValueError if the model lacks the required data."""
    if required == "point_sources":
        if not sky.has_point_sources:
            raise ValueError(
                f"Plot requires point-source data, but this SkyModel "
                f"(model='{sky.model_name}') has no point payload. "
                "Load a point-source catalog or use "
                "radiosim.core.sky.materialize_point_sources_model("
                "sky, ..., lossy=True) first."
            )
    elif required == "healpix":
        if sky.healpix is None:
            raise ValueError(
                f"Plot requires HEALPix maps, but this SkyModel "
                f"(model='{sky.model_name}') has no HEALPix payload. "
                "Load a diffuse model or materialize a HEALPix payload first."
            )


def _resolve_plot_frequency(sky: SkyModel, frequency: float | None) -> float:
    """Pick the best frequency for plotting from what is available."""
    if frequency is not None:
        return float(frequency)
    if sky.reference_frequency is not None:
        return float(sky.reference_frequency)
    if sky.healpix is not None and len(sky.healpix.frequencies) > 0:
        return float(sky.healpix.frequencies[0])
    raise ValueError("No frequency specified and none available in the model.")


def _get_color_data(sky: SkyModel, color_by: str) -> tuple[np.ndarray, str]:
    """Return point-source color values and colorbar label."""
    mapping = {
        "flux": (
            sky.point.flux if sky.point is not None else None,
            _COLORBAR_LABELS["flux"],
        ),
        "spectral_index": (
            sky.point.spectral_index if sky.point is not None else None,
            _COLORBAR_LABELS["spectral_index"],
        ),
        "stokes_q": (
            sky.point.stokes_q if sky.point is not None else None,
            _COLORBAR_LABELS["stokes_q"],
        ),
        "stokes_u": (
            sky.point.stokes_u if sky.point is not None else None,
            _COLORBAR_LABELS["stokes_u"],
        ),
        "stokes_v": (
            sky.point.stokes_v if sky.point is not None else None,
            _COLORBAR_LABELS["stokes_v"],
        ),
    }
    if color_by not in mapping:
        raise ValueError(
            f"Unknown color_by='{color_by}'. Choose from: {list(mapping.keys())}"
        )
    values, label = mapping[color_by]
    if values is None:
        raise ValueError(
            f"color_by='{color_by}' requested but the data is not available "
            f"in this SkyModel."
        )
    return values, label


def _rotate_coords(
    ra_rad: np.ndarray,
    dec_rad: np.ndarray,
    coord: str | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Rotate point-source coordinates from Equatorial to another frame."""
    if coord is None or coord == "C":
        lon = ra_rad.copy()
        lon[lon > np.pi] -= 2 * np.pi
        return lon, dec_rad

    from healpy.rotator import Rotator

    rotator = Rotator(coord=["C", coord])
    theta = np.pi / 2 - dec_rad
    theta_rot, phi_rot = rotator(theta, ra_rad)
    lat = np.pi / 2 - theta_rot
    lon = phi_rot.copy()
    lon[lon > np.pi] -= 2 * np.pi
    return lon, lat


def _healpy_view(
    map_data: np.ndarray,
    projection: str = "mollweide",
    coord: str | list[str] | None = None,
    rot: tuple[float, ...] | None = None,
    **kwargs,
) -> None:
    """Dispatch to the appropriate healpy view function."""
    import healpy as hp

    func_name = _HEALPY_PROJECTIONS.get(projection)
    if func_name is None:
        raise ValueError(
            f"Unknown projection='{projection}'. "
            f"Choose from: {list(_HEALPY_PROJECTIONS.keys())}"
        )
    view_func = getattr(hp, func_name)
    view_func(map_data, coord=coord, rot=rot, **kwargs)


def _get_stokes_map(
    sky: SkyModel,
    stokes: str,
    frequency: float,
) -> tuple[np.ndarray | None, str]:
    """Retrieve a single Stokes map and its label."""
    healpix = sky.healpix
    if healpix is not None:
        healpix = healpix.require_dense("plotter")

    if stokes == "I":
        if healpix is None:
            return None, "Stokes I"
        idx = sky.healpix.resolve_frequency_index(frequency)
        return healpix.maps[idx], "Stokes I"
    if healpix is None:
        return None, f"Stokes {stokes}"

    maps_attr = {
        "Q": healpix.q_maps,
        "U": healpix.u_maps,
        "V": healpix.v_maps,
    }
    if stokes not in maps_attr:
        raise ValueError(f"Unknown stokes='{stokes}'. Choose from 'I', 'Q', 'U', 'V'.")
    stokes_maps = maps_attr[stokes]
    if stokes_maps is None:
        return None, f"Stokes {stokes}"

    idx = sky.healpix.resolve_frequency_index(frequency)
    if idx < len(stokes_maps):
        return stokes_maps[idx], f"Stokes {stokes}"
    return None, f"Stokes {stokes}"


def _get_stokes_cube(sky: SkyModel, stokes: str) -> np.ndarray:
    """Return the full ``(n_freq, npix)`` cube for a Stokes parameter."""
    if sky.healpix is None:
        cube = None
    elif stokes == "I":
        cube = sky.healpix.maps
    else:
        maps_attr = {
            "Q": sky.healpix.q_maps,
            "U": sky.healpix.u_maps,
            "V": sky.healpix.v_maps,
        }
        if stokes not in maps_attr:
            raise ValueError(
                f"Unknown stokes='{stokes}'. Choose from 'I', 'Q', 'U', 'V'."
            )
        cube = maps_attr[stokes]
    if cube is None:
        raise ValueError(f"Stokes {stokes} cube is not available for this SkyModel.")
    return np.asarray(cube)


def _scatter_sky(
    lon: np.ndarray,
    lat: np.ndarray,
    values: np.ndarray,
    cbar_label: str,
    *,
    cmap: str = "inferno",
    marker_size: float = 0.1,
    alpha: float = 0.3,
    vmin: float | None = None,
    vmax: float | None = None,
    projection: str = "mollweide",
    coord: str | None = None,
    title: str = "",
    figsize: tuple[float, float] = (14, 7),
) -> Figure:
    """Shared scatter plot on a sky projection."""
    import matplotlib.pyplot as plt

    if projection not in _MPL_PROJECTIONS:
        raise ValueError(
            f"Unknown projection='{projection}'. "
            f"Choose from: {sorted(_MPL_PROJECTIONS)}"
        )

    if projection == "cartesian":
        fig, ax = plt.subplots(figsize=figsize)
        lon_deg = np.degrees(lon)
        lat_deg = np.degrees(lat)
        sc = ax.scatter(
            lon_deg,
            lat_deg,
            c=values,
            s=marker_size,
            alpha=alpha,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            rasterized=True,
        )
        ax.invert_xaxis()
        ax.set_aspect("equal")
    else:
        fig, ax = plt.subplots(figsize=figsize, subplot_kw={"projection": projection})
        sc = ax.scatter(
            lon,
            lat,
            c=values,
            s=marker_size,
            alpha=alpha,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            rasterized=True,
        )
    fig.colorbar(sc, ax=ax, label=cbar_label, shrink=0.6)
    ax.grid(True, alpha=0.3)
    xlabel, ylabel = _COORD_LABELS.get(coord, ("RA", "Dec"))
    ax.set_xlabel(xlabel + " [deg]" if projection == "cartesian" else xlabel)
    ax.set_ylabel(ylabel + " [deg]" if projection == "cartesian" else ylabel)
    ax.set_title(title)
    return fig
