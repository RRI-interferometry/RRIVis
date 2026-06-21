"""Sky-model diagnostics.

Pure compute helpers for inspecting sky models: angular/delay power
spectra and Gaussianity statistics (:mod:`.analysis`), catalog discovery
and memory estimation (:mod:`.discovery`), and linear-polarisation
derivation (:mod:`.polarization`).
"""

from __future__ import annotations

from .analysis import (
    compute_angular_power_spectrum,
    compute_cross_cell,
    compute_delay_spectrum,
    compute_frequency_correlation,
    compute_kparallel,
    filter_ell_band,
    gaussianity_stats,
)
from .discovery import (
    estimate_healpix_memory,
    get_catalog_info,
    list_all_models,
)
from .polarization import compute_linear_polarization

__all__ = [
    "compute_angular_power_spectrum",
    "compute_cross_cell",
    "compute_delay_spectrum",
    "compute_frequency_correlation",
    "compute_kparallel",
    "filter_ell_band",
    "gaussianity_stats",
    "estimate_healpix_memory",
    "get_catalog_info",
    "list_all_models",
    "compute_linear_polarization",
]
