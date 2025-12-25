# tests/fixtures/sample_sky_models.py
"""
Sample sky model fixtures for RRIvis tests.

These fixtures can be imported directly for use in tests or examples.
"""

import numpy as np


def get_single_bright_source():
    """
    Single bright unpolarized source at zenith.

    Returns
    -------
    list
        List with single source dictionary.
    """
    from astropy.coordinates import SkyCoord
    import astropy.units as u

    return [
        {
            "coords": SkyCoord(ra=0.0 * u.deg, dec=45.0 * u.deg, frame="icrs"),
            "flux": 10.0,
            "spectral_index": -0.7,
        },
    ]


def get_multiple_unpolarized_sources():
    """
    Multiple unpolarized sources at different positions.

    Returns
    -------
    list
        List of source dictionaries.
    """
    from astropy.coordinates import SkyCoord
    import astropy.units as u

    return [
        {
            "coords": SkyCoord(ra=0.0 * u.deg, dec=45.0 * u.deg, frame="icrs"),
            "flux": 10.0,
            "spectral_index": -0.7,
        },
        {
            "coords": SkyCoord(ra=45.0 * u.deg, dec=30.0 * u.deg, frame="icrs"),
            "flux": 5.0,
            "spectral_index": -0.8,
        },
        {
            "coords": SkyCoord(ra=90.0 * u.deg, dec=60.0 * u.deg, frame="icrs"),
            "flux": 2.0,
            "spectral_index": -0.6,
        },
        {
            "coords": SkyCoord(ra=180.0 * u.deg, dec=20.0 * u.deg, frame="icrs"),
            "flux": 1.0,
            "spectral_index": -0.9,
        },
    ]


def get_polarized_sources():
    """
    Polarized sources with Stokes Q, U, V.

    Returns
    -------
    list
        List of source dictionaries with polarization info.
    """
    from astropy.coordinates import SkyCoord
    import astropy.units as u

    return [
        {
            # Strongly linearly polarized (Q)
            "coords": SkyCoord(ra=0.0 * u.deg, dec=45.0 * u.deg, frame="icrs"),
            "flux": 10.0,
            "spectral_index": -0.7,
            "stokes_q": 2.0,  # 20% polarized
            "stokes_u": 0.0,
            "stokes_v": 0.0,
        },
        {
            # Circularly polarized (V)
            "coords": SkyCoord(ra=60.0 * u.deg, dec=30.0 * u.deg, frame="icrs"),
            "flux": 5.0,
            "spectral_index": -0.8,
            "stokes_q": 0.0,
            "stokes_u": 0.0,
            "stokes_v": 1.0,  # 20% circular polarization
        },
        {
            # Mixed polarization
            "coords": SkyCoord(ra=120.0 * u.deg, dec=50.0 * u.deg, frame="icrs"),
            "flux": 8.0,
            "spectral_index": -0.7,
            "stokes_q": 0.5,
            "stokes_u": 0.5,
            "stokes_v": 0.2,
        },
    ]


def get_random_sources(n_sources=100, flux_min=0.1, flux_max=10.0, seed=42):
    """
    Random sky model with specified number of sources.

    Parameters
    ----------
    n_sources : int
        Number of sources to generate.
    flux_min : float
        Minimum flux in Jy.
    flux_max : float
        Maximum flux in Jy.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    list
        List of source dictionaries.
    """
    from astropy.coordinates import SkyCoord
    import astropy.units as u

    np.random.seed(seed)
    sources = []

    for _ in range(n_sources):
        # Random position on sphere
        ra = np.random.uniform(0, 360)
        dec = np.degrees(np.arcsin(np.random.uniform(-1, 1)))

        # Power-law flux distribution (more faint sources)
        flux = flux_min * (flux_max / flux_min) ** np.random.random()

        # Random spectral index (typical range)
        spectral_index = np.random.uniform(-1.0, -0.5)

        sources.append({
            "coords": SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame="icrs"),
            "flux": flux,
            "spectral_index": spectral_index,
        })

    return sources


def get_grid_sources(n_ra=5, n_dec=5, flux=1.0, ra_center=0.0, dec_center=45.0, spacing=5.0):
    """
    Regular grid of sources for testing.

    Parameters
    ----------
    n_ra : int
        Number of sources in RA direction.
    n_dec : int
        Number of sources in Dec direction.
    flux : float
        Flux for all sources in Jy.
    ra_center : float
        Center RA in degrees.
    dec_center : float
        Center Dec in degrees.
    spacing : float
        Spacing between sources in degrees.

    Returns
    -------
    list
        List of source dictionaries.
    """
    from astropy.coordinates import SkyCoord
    import astropy.units as u

    sources = []

    ra_offsets = np.linspace(-(n_ra - 1) / 2 * spacing, (n_ra - 1) / 2 * spacing, n_ra)
    dec_offsets = np.linspace(-(n_dec - 1) / 2 * spacing, (n_dec - 1) / 2 * spacing, n_dec)

    for ra_off in ra_offsets:
        for dec_off in dec_offsets:
            sources.append({
                "coords": SkyCoord(
                    ra=(ra_center + ra_off) * u.deg,
                    dec=(dec_center + dec_off) * u.deg,
                    frame="icrs"
                ),
                "flux": flux,
                "spectral_index": -0.7,
            })

    return sources


def get_point_source_at_phase_center():
    """
    Single point source exactly at the phase center.

    Useful for testing fringe patterns and baseline response.

    Returns
    -------
    list
        List with single source at phase center.
    """
    from astropy.coordinates import SkyCoord
    import astropy.units as u

    return [
        {
            "coords": SkyCoord(ra=0.0 * u.deg, dec=0.0 * u.deg, frame="icrs"),
            "flux": 1.0,
            "spectral_index": 0.0,  # Flat spectrum for simplicity
        },
    ]


def get_two_point_sources_symmetric():
    """
    Two point sources symmetric about the phase center.

    Useful for testing baseline response and UV coverage.

    Returns
    -------
    list
        List with two symmetric sources.
    """
    from astropy.coordinates import SkyCoord
    import astropy.units as u

    return [
        {
            "coords": SkyCoord(ra=1.0 * u.deg, dec=0.0 * u.deg, frame="icrs"),
            "flux": 1.0,
            "spectral_index": -0.7,
        },
        {
            "coords": SkyCoord(ra=-1.0 * u.deg, dec=0.0 * u.deg, frame="icrs"),
            "flux": 1.0,
            "spectral_index": -0.7,
        },
    ]


# Convenience dictionary for quick access
SKY_MODELS = {
    "single": get_single_bright_source,
    "multiple": get_multiple_unpolarized_sources,
    "polarized": get_polarized_sources,
    "random": get_random_sources,
    "grid": get_grid_sources,
    "phase_center": get_point_source_at_phase_center,
    "symmetric": get_two_point_sources_symmetric,
}
