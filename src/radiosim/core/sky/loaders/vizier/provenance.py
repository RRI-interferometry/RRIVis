"""Provenance builder for point-source catalogs loaded from VizieR or CASDA."""

from __future__ import annotations

import numpy as np

from ...containers import (
    MonopoleConvention,
    SkyCoverage,
    SkyProvenance,
    SourceSubtractionStatus,
)
from ...operations.region import SkyRegion
from ...registry import (
    RacsCatalogEntry,
    VizierCatalogEntry,
    load_catalog_footprint_asset,
)


def _build_point_catalog_provenance(
    *,
    info: VizierCatalogEntry | RacsCatalogEntry,
    flux_limit_jy: float,
    flux_jy: np.ndarray | None,
    catalog_key: str,
    region: SkyRegion | None = None,
) -> SkyProvenance:
    """Build a ``SkyProvenance`` for a point-source catalog from its metadata.

    The returned provenance declares the catalog's flux-completeness band,
    angular resolution, and ABSOLUTE_NO_CMB monopole (point catalogs carry no
    isotropic background).  ``flux_completeness_jy`` uses ``flux_limit_jy``
    as the lower bound (the requested loader cut, also the catalog's effective
    detection floor for this query) and the catalog's saturation/brightest-
    sources cutoff (``info.flux_saturation_jy`` if known, else ``inf``) as the
    upper bound.  This is a metadata band, not a sample statistic — the
    brightest source actually present in the loaded subset is irrelevant.
    """
    saturation = getattr(info, "flux_saturation_jy", None)
    upper = float(saturation) if saturation is not None else float("inf")
    flux_completeness: tuple[float, float] | None = (float(flux_limit_jy), upper)

    # Angular resolution: beam FWHM at the low end, full sky at the high end.
    beam_arcsec = getattr(info, "beam_fwhm_arcsec", None)
    if beam_arcsec is not None:
        beam_rad = float(beam_arcsec) * (np.pi / 180.0) / 3600.0
        angular_resolution_rad: tuple[float, float] | None = (
            beam_rad,
            float(np.pi),
        )
    else:
        angular_resolution_rad = None

    coverage_footprint = None
    footprint_asset = getattr(info, "footprint_asset", None)
    if footprint_asset:
        coverage_footprint = load_catalog_footprint_asset(footprint_asset)
        if region is not None:
            coverage_footprint = coverage_footprint.intersect_mask(
                region.healpix_mask(
                    coverage_footprint.nside,
                    coordinate_frame=coverage_footprint.coordinate_frame,
                )
            )

    if coverage_footprint is None:
        sky_coverage = SkyCoverage.PARTIAL_SKY
        coverage_fraction = None
    else:
        sky_coverage = (
            SkyCoverage.FULL_SKY
            if coverage_footprint.is_full_sky
            else SkyCoverage.PARTIAL_SKY
        )
        coverage_fraction = coverage_footprint.coverage_fraction

    # Monopole: treat the point catalog's integrated I flux over 4π sr as the
    # contribution of discrete sources to the sky mean (in Jy/sr), which is a
    # separate bookkeeping axis from the diffuse-map monopole. ``monopole_k``
    # stays None here and is filled by the combiner once a frequency is known.
    return SkyProvenance(
        flux_completeness_jy=flux_completeness,
        flux_completeness_freq_hz=float(info.freq_mhz) * 1e6,
        angular_resolution_rad=angular_resolution_rad,
        sky_coverage=sky_coverage,
        coverage_fraction=coverage_fraction,
        coverage_footprint=coverage_footprint,
        monopole_convention=MonopoleConvention.ABSOLUTE_NO_CMB,
        monopole_k=None,
        source_subtraction=SourceSubtractionStatus.NONE,
        notes=f"vizier/{catalog_key}"
        if isinstance(info, VizierCatalogEntry)
        else f"racs/{catalog_key}",
    )
