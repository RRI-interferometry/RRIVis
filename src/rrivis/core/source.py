# rrivis/core/source.py
"""
Sky Model Source Loading Module for RRIVis.

.. deprecated:: 0.3.0
    This module is deprecated. Use :class:`rrivis.core.sky_model.SkyModel` instead.

    Migration guide:
    - `generate_test_sources(n)` -> `SkyModel.from_test_sources(num_sources=n).to_point_sources()`
    - `load_gleam(flux_limit=x)` -> `SkyModel.from_gleam(flux_limit=x).to_point_sources()`
    - `load_mals(flux_limit=x)` -> `SkyModel.from_mals(flux_limit=x).to_point_sources()`
    - `load_diffuse_sky(...)` -> `SkyModel.from_diffuse_sky(...).to_point_sources()`
    - `get_sources(...)` -> Use SkyModel factory methods directly

The functions in this module are kept for backward compatibility but will be
removed in a future version.
"""

import logging
import warnings
from typing import Dict, List, Optional, Tuple, Any

from rrivis.core.sky_model import (
    SkyModel,
    GLEAM_CATALOGS,
    MALS_CATALOGS,
    DIFFUSE_MODELS,
)


logger = logging.getLogger(__name__)

# Type alias for source dictionaries
SourceDict = Dict[str, Any]

# Re-export catalog metadata for backward compatibility
DIFFUSE_SKY_MODELS = {
    name: {
        "class": info["class"],
        "description": info["description"],
        "freq_range": info["freq_range"],
        "reference": info.get("reference", ""),
        "init_kwargs": info["init_kwargs"],
    }
    for name, info in DIFFUSE_MODELS.items()
}


def _deprecation_warning(old_func: str, new_method: str) -> None:
    """Emit deprecation warning."""
    warnings.warn(
        f"{old_func}() is deprecated. Use SkyModel.{new_method}() instead. "
        "See rrivis.core.sky_model for the new unified API.",
        DeprecationWarning,
        stacklevel=3
    )


def get_sources(
    use_test_sources: bool = False,
    use_test_sources_healpix: bool = False,
    use_gleam: bool = False,
    use_gsm: bool = False,
    use_gleam_healpix: bool = False,
    use_gsm_gleam_healpix: bool = False,
    use_mals: bool = False,
    mals_release: str = "dr2",
    gleam_catalogue: str = "VIII/100/gleamegc",
    gsm_catalogue: str = "gsm2008",
    diffuse_model: str = "gsm2008",
    flux_limit: Optional[float] = None,
    frequency: Optional[float] = None,
    nside: Optional[int] = None,
    num_sources: Optional[int] = None,
) -> Tuple[List[SourceDict], int]:
    """
    Load sources from various sky model catalogs.

    .. deprecated:: 0.3.0
        Use SkyModel factory methods instead.

    Parameters
    ----------
    use_test_sources : bool
        Generate test sources.
    use_gleam : bool
        Load GLEAM catalog.
    use_gsm : bool
        Load diffuse sky model.
    use_mals : bool
        Load MALS catalog.
    mals_release : str
        MALS release: dr1, dr2, dr3.
    flux_limit : float, optional
        Minimum flux in Jy (mJy for MALS).
    frequency : float, optional
        Frequency in Hz for GSM.
    nside : int, optional
        HEALPix NSIDE for GSM.
    num_sources : int, optional
        Number of test sources.

    Returns
    -------
    sources : list of dict
        Source list.
    status : int
        Status code (0 = success).
    """
    _deprecation_warning("get_sources", "from_*")

    # Dispatch to appropriate loader
    if use_test_sources or use_test_sources_healpix:
        return generate_test_sources(num_sources), 0

    elif use_gleam:
        return load_gleam(flux_limit=flux_limit or 1.0, gleam_catalogue=gleam_catalogue)

    elif use_mals:
        return load_mals(flux_limit=flux_limit or 1.0, release=mals_release)

    elif use_gsm:
        model = diffuse_model if diffuse_model != "gsm2008" else gsm_catalogue
        return load_diffuse_sky(
            frequency=frequency or 100e6,
            nside=nside or 32,
            flux_limit=flux_limit or 1.0,
            model=model
        )

    else:
        return generate_test_sources(num_sources), 0


def generate_test_sources(num_sources: Optional[int] = 3) -> List[SourceDict]:
    """
    Generate synthetic test sources.

    .. deprecated:: 0.3.0
        Use `SkyModel.from_test_sources()` instead.

    Parameters
    ----------
    num_sources : int, optional
        Number of sources. Defaults to 3.

    Returns
    -------
    list of dict
        Source list.
    """
    _deprecation_warning("generate_test_sources", "from_test_sources")

    if num_sources is None:
        num_sources = 3

    sky = SkyModel.from_test_sources(num_sources=num_sources)
    return sky.to_point_sources()


def load_gleam(
    flux_limit: float,
    gleam_catalogue: str = "VIII/100/gleamegc"
) -> Tuple[List[SourceDict], int]:
    """
    Load GLEAM catalog from VizieR.

    .. deprecated:: 0.3.0
        Use `SkyModel.from_gleam()` instead.

    Parameters
    ----------
    flux_limit : float
        Minimum flux in Jy.
    gleam_catalogue : str
        VizieR catalog ID.

    Returns
    -------
    sources : list of dict
    status : int
    """
    _deprecation_warning("load_gleam", "from_gleam")

    sky = SkyModel.from_gleam(flux_limit=flux_limit, catalog=gleam_catalogue)
    return sky.to_point_sources(), 0


def load_mals(
    flux_limit: float = 1.0,
    release: str = "dr2"
) -> Tuple[List[SourceDict], int]:
    """
    Load MALS catalog from VizieR.

    .. deprecated:: 0.3.0
        Use `SkyModel.from_mals()` instead.

    Parameters
    ----------
    flux_limit : float
        Minimum flux in mJy.
    release : str
        Data release: dr1, dr2, dr3.

    Returns
    -------
    sources : list of dict
    status : int
    """
    _deprecation_warning("load_mals", "from_mals")

    sky = SkyModel.from_mals(flux_limit=flux_limit, release=release)
    return sky.to_point_sources(), 0


def load_mals_dr1(flux_limit: float = 1.0) -> Tuple[List[SourceDict], int]:
    """Load MALS DR1. Deprecated: use SkyModel.from_mals(release='dr1')."""
    return load_mals(flux_limit=flux_limit, release="dr1")


def load_mals_dr2(flux_limit: float = 1.0) -> Tuple[List[SourceDict], int]:
    """Load MALS DR2. Deprecated: use SkyModel.from_mals(release='dr2')."""
    return load_mals(flux_limit=flux_limit, release="dr2")


def load_mals_dr3(flux_limit: float = 1.0) -> Tuple[List[SourceDict], int]:
    """Load MALS DR3. Deprecated: use SkyModel.from_mals(release='dr3')."""
    return load_mals(flux_limit=flux_limit, release="dr3")


def load_diffuse_sky(
    frequency: float = 76e6,
    nside: int = 32,
    flux_limit: float = 1.0,
    beam_area: Optional[float] = None,
    model: str = "gsm2008",
    compute_spectral_index: bool = True,
    reference_frequency: Optional[float] = None,
    include_cmb: bool = False
) -> Tuple[List[SourceDict], int]:
    """
    Load diffuse sky model and convert to point sources.

    .. deprecated:: 0.3.0
        Use `SkyModel.from_diffuse_sky()` instead.

    Parameters
    ----------
    frequency : float
        Frequency in Hz.
    nside : int
        HEALPix NSIDE.
    flux_limit : float
        Minimum flux in Jy.
    model : str
        Model name: gsm2008, gsm2016, lfsm, haslam.
    compute_spectral_index : bool
        Compute per-pixel spectral indices.

    Returns
    -------
    sources : list of dict
    status : int
    """
    _deprecation_warning("load_diffuse_sky", "from_diffuse_sky")

    sky = SkyModel.from_diffuse_sky(
        model=model,
        frequency=frequency,
        nside=nside,
        compute_spectral_index=compute_spectral_index,
        reference_frequency=reference_frequency,
        include_cmb=include_cmb,
    )

    sources = sky.to_point_sources(flux_limit=flux_limit, frequency=frequency)
    return sources, 0


def load_gleam_in_healpix(
    flux_limit: float = 50,
    nside: int = 32,
    ref_freq: float = 76e6
) -> Tuple:
    """
    Load GLEAM catalog into HEALPix format.

    .. deprecated:: 0.3.0
        Use `SkyModel.from_gleam().to_healpix()` instead.
    """
    _deprecation_warning("load_gleam_in_healpix", "from_gleam().to_healpix()")

    sky = SkyModel.from_gleam(flux_limit=flux_limit)
    temp_map, nside_out, spec_map = sky.to_healpix(nside=nside, frequency=ref_freq)
    return temp_map, spec_map


def load_gsm_gleam_in_healpix(
    frequency: Optional[float] = None,
    nside: Optional[int] = None,
    flux_limit: Optional[float] = None
) -> Tuple[Optional, Optional]:
    """
    Combined GSM+GLEAM in HEALPix.

    .. deprecated:: 0.3.0
        Use `SkyModel.combine()` instead.
    """
    _deprecation_warning("load_gsm_gleam_in_healpix", "combine()")
    return None, None
