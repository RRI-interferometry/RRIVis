"""Linear-polarisation diagnostics for sky models.

Relocated from ``operations/operations.py`` (spec item F2): deriving the
linear-polarisation amplitude, angle, and fraction from a model's Stokes
Q/U is a diagnostic, not a mutation-free transform, so it lives alongside
the other ``diagnostics`` helpers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

_FRACTIONAL_POL_I_ATOL = 1e-12

if TYPE_CHECKING:
    from ..containers.model import SkyModel


def compute_linear_polarization(
    sky: SkyModel,
    *,
    frequency: float | None = None,
) -> dict[str, np.ndarray]:
    """Derive ``(P, χ, P/|I|)`` from a SkyModel's Stokes Q/U.

    For a HEALPix payload, returns dense maps shaped ``(npix,)`` when
    ``frequency`` is given (the closest channel is selected) or
    ``(n_freq, npix)`` when ``frequency=None``.  For a point-source
    payload, returns ``(n_sources,)`` arrays — Q/U here are intrinsic
    Stokes parameters, no per-frequency scaling is applied.

    Parameters
    ----------
    sky
        Sky model carrying Stokes Q and U.  ``ValueError`` is raised if
        either is absent.
    frequency
        Optional frequency (Hz) at which to slice a HEALPix payload.
        Ignored for point-source payloads.

    Returns
    -------
    dict
        Keys:

        - ``"P"`` : ``sqrt(Q² + U²)`` (linear polarisation amplitude).
        - ``"chi_deg"`` : ``0.5 · atan2(U, Q)`` in degrees, range
          ``(-90°, 90°]``.
        - ``"frac_pol"`` : ``P / |I|`` (fractional linear polarisation).
          ``nan`` where ``I = 0``.

    Raises
    ------
    ValueError
        If neither payload carries Q and U.
    """
    if sky.healpix is not None:
        if sky.healpix.q_maps is None or sky.healpix.u_maps is None:
            raise ValueError(
                "compute_linear_polarization requires Stokes Q and U HEALPix "
                "maps; the input has none.  Load a polarised template (e.g. "
                "PySM3 with synchrotron) or supply Q/U arrays explicitly."
            )
        if frequency is None:
            i_maps = sky.healpix.maps
            q_maps = sky.healpix.q_maps
            u_maps = sky.healpix.u_maps
        else:
            idx = sky.healpix.resolve_frequency_index(float(frequency))
            i_maps = sky.healpix.maps[idx]
            q_maps = sky.healpix.q_maps[idx]
            u_maps = sky.healpix.u_maps[idx]
        return _linear_pol_arrays(i_maps, q_maps, u_maps)

    if sky.point is not None:
        if sky.point.stokes_q is None or sky.point.stokes_u is None:
            raise ValueError(
                "compute_linear_polarization requires Stokes Q and U "
                "components on the point payload; got neither."
            )
        return _linear_pol_arrays(
            sky.point.flux,
            sky.point.stokes_q,
            sky.point.stokes_u,
        )

    raise ValueError("SkyModel carries no payload; cannot derive polarisation.")


def _linear_pol_arrays(
    i: np.ndarray,
    q: np.ndarray,
    u: np.ndarray,
) -> dict[str, np.ndarray]:
    q_arr = np.asarray(q, dtype=float)
    u_arr = np.asarray(u, dtype=float)
    i_arr = np.asarray(i, dtype=float)
    p = np.hypot(q_arr, u_arr)
    chi_rad = 0.5 * np.arctan2(u_arr, q_arr)
    chi_deg = np.degrees(chi_rad)
    with np.errstate(divide="ignore", invalid="ignore"):
        frac_pol = p / np.abs(i_arr)
    frac_pol = np.where(
        np.isclose(i_arr, 0.0, rtol=0.0, atol=_FRACTIONAL_POL_I_ATOL),
        np.nan,
        frac_pol,
    )
    return {"P": p, "chi_deg": chi_deg, "frac_pol": frac_pol}
