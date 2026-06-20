"""Coverage-provenance helper.

Consolidates the FULL_SKY / PARTIAL_SKY coverage block that was inlined
four times across ``loaders/diffuse.py`` (GSM + PySM3) and
``loaders/synthetic.py`` (Poisson empty + non-empty) (spec item B5).

Each of those sites computes the same three coverage fields
(``sky_coverage``, ``coverage_fraction``, ``coverage_footprint``) from a
:class:`~radiosim.core.sky.operations.region.SkyRegion` and then embeds
them into a :class:`~radiosim.core.sky.containers.SkyProvenance` together
with site-specific fields (e.g. ``monopole_k``, ``flux_completeness_jy``).
This helper returns just that shared coverage triple as a
:class:`CoverageProvenance`; the caller splices it into its
``SkyProvenance(...)`` construction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from ..containers import SkyCoverage, SkyFootprint
from ..containers.footprint import DEFAULT_COVERAGE_FOOTPRINT_NSIDE

if TYPE_CHECKING:
    from ..operations.region import SkyRegion


@dataclass(frozen=True)
class CoverageProvenance:
    """The shared coverage fields spliced into a :class:`SkyProvenance`.

    Attributes
    ----------
    sky_coverage : SkyCoverage
        ``FULL_SKY`` or ``PARTIAL_SKY``.
    coverage_fraction : float or None
        Fraction of the full sky covered (``1.0`` for full sky; the
        footprint's coverage fraction for a partial region; ``None`` when
        a partial region's fraction is unknown).
    coverage_footprint : SkyFootprint or None
        The sparse HEALPix footprint for a partial region; ``None`` for
        full sky.
    """

    sky_coverage: SkyCoverage
    coverage_fraction: float | None
    coverage_footprint: SkyFootprint | None


def coverage_provenance(
    *,
    is_full_sky: bool,
    nside: int = DEFAULT_COVERAGE_FOOTPRINT_NSIDE,
    observed_fraction: float | None = None,
    region: SkyRegion | None = None,
) -> CoverageProvenance:
    """Build the shared FULL_SKY / PARTIAL_SKY coverage fields once.

    Reproduces the inline decision at the four current call sites:

    * ``region is None`` (full sky): ``FULL_SKY``, fraction ``1.0``, no
      footprint.
    * ``region is not None`` (partial): ``PARTIAL_SKY``; the footprint is
      ``region.footprint()`` and the fraction is its ``coverage_fraction``
      (or ``observed_fraction`` if the caller supplies one explicitly).

    Parameters
    ----------
    is_full_sky : bool
        Whether the payload covers the full sky. When a ``region`` is
        given it is the authoritative driver (``region is None`` ⇒ full
        sky), matching the existing call sites; ``is_full_sky`` is used
        only when ``region is None``.
    nside : int
        HEALPix NSIDE — used as the footprint grid resolution when a
        ``region`` is given.
    observed_fraction : float or None, optional
        Explicit partial-sky coverage fraction. When ``None`` and a
        ``region`` is given, the footprint's ``coverage_fraction`` is used.
    region : SkyRegion or None, optional
        The angular region. ``None`` means full sky.

    Returns
    -------
    CoverageProvenance
        The shared ``sky_coverage`` / ``coverage_fraction`` /
        ``coverage_footprint`` triple.
    """
    full_sky = region is None and is_full_sky

    if region is None:
        if full_sky:
            return CoverageProvenance(
                sky_coverage=SkyCoverage.FULL_SKY,
                coverage_fraction=1.0,
                coverage_footprint=None,
            )
        return CoverageProvenance(
            sky_coverage=SkyCoverage.PARTIAL_SKY,
            coverage_fraction=observed_fraction,
            coverage_footprint=None,
        )

    coverage_footprint = region.footprint(nside=nside)
    coverage_fraction = (
        observed_fraction
        if observed_fraction is not None
        else coverage_footprint.coverage_fraction
    )
    return CoverageProvenance(
        sky_coverage=SkyCoverage.PARTIAL_SKY,
        coverage_fraction=coverage_fraction,
        coverage_footprint=coverage_footprint,
    )
