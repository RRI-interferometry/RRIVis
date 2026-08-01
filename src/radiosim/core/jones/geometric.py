"""The geometric phase (the K term), as a function.

    K = exp(-2*pi*i*(u*l + v*m + w*(n - 1)))

with ``(u, v, w)`` the baseline vector **in wavelengths** and ``(l, m, n)`` the
direction cosines of the sky direction.  The ``w*(n - 1)`` form is the exact
non-coplanar phase, not the small-field ``w`` approximation: the ``-1`` is what
makes the phase vanish at the phase centre, and carrying it exactly is why no
separate ``W`` term is needed or wanted -- an enabled one would double-count
(``Tier7JonesSciencePlan.md`` defect D19).

Why a function and not a class
------------------------------
``K`` is the one term in the standard eight that is **not** per-antenna: it
depends on the baseline vector, so it cannot be composed into a per-antenna
Jones chain.  Until Tier 7B the repository had three implementations of it -- an
exported ``GeometricPhaseJones`` class that no solver ever constructed, and one
inline copy in each solver -- which is defect D6.  The class could only ever
return the identity unless a caller smuggled ``baseline_uvw`` in through
``**kwargs``, which nothing did.

The physics lives here once, as two small functions both solvers call, and the
scalar phase is applied by the solver alongside the compiled contraction rather
than multiplied into a ``(n_dir, 2, 2)`` matrix per antenna: it is proportional
to the identity, so it commutes with everything and needs no matrix at all.
"""

from typing import Any

import numpy as np

__all__ = ["geometric_phase", "uvw_in_wavelengths"]


def uvw_in_wavelengths(*, baseline_vectors_m: Any, wavelength_m: float) -> Any:
    """Return the selected baseline vectors in wavelengths.

    Parameters
    ----------
    baseline_vectors_m : array
        ``(n_baselines, 3)`` local ENU baseline vectors in metres, already in
        the backend's array domain.
    wavelength_m : float
        The observing wavelength, ``c / nu``.

    Returns
    -------
    array
        ``(n_baselines, 3)``, same array domain and dtype rules as the input.
    """
    return baseline_vectors_m / wavelength_m


def geometric_phase(
    *,
    uvw_wavelengths: Any,
    dir_l: Any,
    dir_m: Any,
    dir_n: Any,
    backend: Any,
) -> Any:
    """Return ``exp(-2*pi*i*(u*l + v*m + w*(n - 1)))`` for every baseline.

    Parameters
    ----------
    uvw_wavelengths : array
        ``(n_baselines, 3)`` baseline vectors in wavelengths, from
        :func:`uvw_in_wavelengths`.
    dir_l, dir_m, dir_n : array
        ``(n_dir,)`` direction cosines **already in the backend's array domain
        and in the backend's resolved real dtype**.  They are passed as arrays
        rather than as a :class:`~radiosim.core.jones.directions.DirectionBatch`
        deliberately: the batch is the host-side description of the directions
        (``Tier6HybridRuntimePlan.md`` Section 13.3), while the phase is the
        hottest array expression in the solver and must run in whatever real
        precision ``PrecisionConfig`` resolved -- ``float32`` under the ``fast``
        preset -- not silently in the batch's ``float64``.
    backend : ArrayBackend
        The backend whose ``exp`` evaluates the phase.

    Returns
    -------
    array
        ``(n_baselines, n_dir)`` complex phase factors.
    """
    bl_u = uvw_wavelengths[:, 0:1]
    bl_v = uvw_wavelengths[:, 1:2]
    bl_w = uvw_wavelengths[:, 2:3]
    b_dot_s = bl_u * dir_l + bl_v * dir_m + bl_w * (dir_n - 1.0)
    return backend.exp(-2j * np.pi * b_dot_s)
