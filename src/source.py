# src/source.py

from astropy.coordinates import SkyCoord
import astropy.units as au
from astroquery.vizier import Vizier
from pympler import asizeof
import numpy as np
import healpy as hp
from healpy.rotator import Rotator
from pygdsm import GlobalSkyModel
from tqdm import tqdm
import sys


def _supports_color() -> bool:
    try:
        return hasattr(sys.__stdout__, "isatty") and sys.__stdout__.isatty()
    except Exception:
        return False


_TTY = _supports_color()
_RESET = "\033[0m"
_BOLD = "\033[1m"
_CYAN = "\033[36m"


def _c(text: str, style: str) -> str:
    if not _TTY:
        return text
    return f"{style}{text}{_RESET}"


def get_sources(
    use_test_sources=False,
    use_test_sources_healpix=False,
    use_gleam=False,
    use_gsm=False,
    use_gleam_healpix=False,
    use_gsm_gleam_healpix=False,
    gleam_catalogue="VIII/100/gleamegc",
    gsm_catalogue="gsm2008",
    flux_limit=None,
    frequency=None,
    nside=None,
    num_sources=None,
):
    """
    Returns either test sources, GLEAM catalog sources, or GSM2008 sources based on the arguments.

    Parameters:
    - use_test_sources (bool): If True, uses test sources.
    - use_test_sources_healpix (bool): If True, uses test sources in HEALPix format
    - use_gleam (bool): If True, loads the GLEAM catalog.
    - use_gsm (bool): If True, loads GSM2008 sources.
    - use_gleam_healpix (bool): If True, loads GLEAM catalog and maps to HEALPix.
    - use_gsm_gleam_healpix (bool): If True, loads GSM2008 sources and maps to HEALPix.
    - flux_limit (float): Flux limit for filtering GLEAM sources.
    - frequency (float): Frequency for GSM sources in Hz.
    - nside (int): Healpix resolution parameter for GSM sources.

    Returns:
    list: List of sources with "coords", "flux", and "spectral_index".
    """

    if use_test_sources:
        sources = generate_test_sources(num_sources)
        print(f"Using {len(sources)} dynamic test sources...")
        return sources, 0
    elif use_test_sources_healpix:
        sources = generate_test_sources(num_sources)
        print(f"Using {len(sources)} dynamic test sources in HEALPix format...")
        return sources, 0
    elif use_gleam:
        print(_c("Loading GLEAM catalog:", _BOLD + _CYAN))
        return load_gleam(flux_limit=flux_limit, gleam_catalogue=gleam_catalogue)
    elif use_gsm:
        print("Loading GSM2008 sources...")
        return load_gsm2008(flux_limit=flux_limit, frequency=frequency, nside=nside)
    elif use_gleam_healpix:
        print(_c("Loading GLEAM catalog and mapping to HEALPix:", _BOLD + _CYAN))
        return load_gleam_in_healpix(flux_limit, nside)
    elif use_gsm_gleam_healpix:
        print("Loading GSM2008 sources and mapping to HEALPix...")
        return load_gsm_gleam_in_healpix(
            frequency=frequency, nside=nside, flux_limit=flux_limit
        )
    else:
        sources = generate_test_sources(num_sources)
        print(f"Using {len(sources)} dynamic test sources...")
        return sources, 0


def generate_test_sources(num_sources=3):
    """
    Generate test sources dynamically based on the specified number.

    Parameters:
    - num_sources (int): Number of test sources to generate. Defaults to 3.

    Returns:
    list: List of test sources with coordinates, flux, and spectral index.
    """
    if num_sources is None:
        num_sources = 3  # Default fallback

    sources = []

    # If only 1 source, place it at zenith
    if num_sources == 1:
        sources.append({
            "coords": SkyCoord(ra=0 * au.deg, dec=-30.72152777777791 * au.deg),
            "flux": 4,
            "spectral_index": -0.8,
        })
        return sources

    # Generate sources distributed evenly in RA
    for i in range(num_sources):
        ra_deg = (360.0 / num_sources) * i  # Evenly distribute in RA
        dec_deg = -30.72152777777791  # Fixed declination for all test sources

        # Vary flux to create some diversity (2-8 Jy range)
        flux = 2 + (6.0 * i / (num_sources - 1)) if num_sources > 1 else 4

        sources.append({
            "coords": SkyCoord(ra=ra_deg * au.deg, dec=dec_deg * au.deg),
            "flux": flux,
            "spectral_index": -0.8,  # Same spectral index for all test sources
        })

    return sources


# Legacy test_sources variable for backward compatibility (now generated dynamically)
test_sources = generate_test_sources()


# Test sources in healpix
def load_test_sources_in_healpix(nside):
    """
    Converts test sources to a HEALPix map.

    Parameters:
    - nside (int): HEALPix NSIDE parameter (resolution).

    Returns:
    tuple: Two HEALPix maps (flux_map, spectral_index_map).
    """
    npix = hp.nside2npix(nside)
    healpix_flux_map = np.zeros(npix)
    healpix_spectral_index_map = np.zeros(npix)

    for source in test_sources:
        ra = source["coords"].ra.deg
        dec = source["coords"].dec.deg
        flux = source["flux"]
        spectral_index = source["spectral_index"]

        # Convert RA/Dec to HEALPix pixel index
        theta = np.radians(90 - dec)
        phi = np.radians(ra)
        pixel_index = hp.ang2pix(nside, theta, phi)

        # Add flux to the corresponding HEALPix pixel
        healpix_flux_map[pixel_index] += flux
        healpix_spectral_index_map[pixel_index] = spectral_index

    return healpix_flux_map, healpix_spectral_index_map


# GLEAM catalog loading function
def load_gleam(flux_limit, gleam_catalogue):
    """
    Loads the GLEAM Extragalactic catalog from VizieR using astroquery.
    Applies a flux limit to reduce the number of sources.

    Parameters:
    - flux_limit (float): Minimum flux (in Jy) to include a source.
    - gleam_catalogue (str): Catalogue ID to fetch (e.g., "VIII/100/gleamegc", "VIII/113/catalog2").

    Returns:
    list: List of sources with their coordinates, fluxes, and spectral indices.
    """
    # Set up Vizier query
    Vizier.ROW_LIMIT = -1

    # Validate if the provided catalogue exists in VizieR
    available_catalogs = {
        "VIII/100/table1": "GLEAM first year observing parameters (28 rows)",
        "VIII/100/gleamegc": "GLEAM EGC catalog, version 2 (307455 rows)",
        "VIII/102/gleamgal": "GLEAM Galactic plane catalog (22037 rows)",
        "VIII/105/catalog": "G4Jy catalogue (18/01/2020) (1960 rows)",
        "VIII/109/gleamsgp": "GLEAM SGP catalogue (108851 rows)",
        "VIII/110/catalog": "First data release of GLEAM-X (78967 rows)",
        "VIII/113/catalog2": "Second data release of GLEAM-X (624866 rows)",
    }

    if gleam_catalogue not in available_catalogs:
        print(f"Invalid GLEAM catalogue '{gleam_catalogue}' selected!")
        print("Available GLEAM catalogues:")
        for cat, desc in available_catalogs.items():
            print(f"{cat} → {desc}")
        return []

    # Retrieve the selected GLEAM catalogue
    catalog_list = Vizier.find_catalogs(gleam_catalogue)
    if not catalog_list:
        raise Exception("GLEAM catalog not found in VizieR.")
    print(
        _c("Fetching GLEAM Catalogue:", _BOLD + _CYAN),
        f"{gleam_catalogue} → {available_catalogs[gleam_catalogue]}",
    )
    print(
        _c("Downloading from VizieR...", _BOLD + _CYAN),
        "(this may take a while for large catalogs)",
    )
    catalog = Vizier.get_catalogs(gleam_catalogue)[0]
    print(_c("Download complete:", _BOLD + _CYAN), f"Retrieved {len(catalog)} sources.")

    print(
        _c("Processing:", _BOLD + _CYAN),
        f"{len(catalog)} sources from GLEAM catalog...",
    )

    # Apply flux limit and extract positions and fluxes
    sources = []
    # Disable tqdm when logging redirects output (avoids messy progress bars)
    for row in tqdm(
        catalog, desc="Processing GLEAM sources", unit="sources", disable=True
    ):
        flux = row["Fp076"]  # Wide-band peak flux density at 76 MHz in Jy
        if flux >= flux_limit:
            ra = row["RAJ2000"] * au.deg
            dec = row["DEJ2000"] * au.deg
            coords = SkyCoord(ra=ra, dec=dec)

            # Retrieve the spectral index from the 'Alpha' column
            if (
                "alpha" in row.colnames
                and isinstance(row["alpha"], (float, np.float32, np.float64))
                and np.isfinite(row["alpha"])
            ):
                spectral_index = row["alpha"]
            else:
                spectral_index = 0.0  # Assign a default value if missing

            print(
                _c("Source:", _BOLD + _CYAN),
                f"RA: {ra}, Dec: {dec}, Flux: {flux}, Spectral Index: {spectral_index}",
            )

            # Append source dictionary to the list
            sources.append(
                {"coords": coords, "flux": flux, "spectral_index": spectral_index}
            )

    # Measure the memory size of the loaded GLEAM catalog
    gleam_size = asizeof.asizeof(sources)
    print(
        _c("GLEAM catalog loaded:", _BOLD + _CYAN),
        f"{len(sources)} sources. Total size: {gleam_size / (1024**2):.2f} MB",
    )

    return sources, 0


def load_gsm2008(frequency=76e6, nside=32, flux_limit=1.0, beam_area=1.0):
    """
    Processes the GSM2008 Healpix map and converts it into sources compatible with the app.

    Parameters:
    - frequency (float): Frequency in Hz for flux density calculations.
    - nside (int): Healpix resolution parameter.
    - flux_limit (float): Minimum flux (in Jy) to include a source.
    - beam_area (float): Beam solid angle in steradians for flux conversion.

    Returns:
    list: List of sources with "coords", "flux", and "spectral_index" keys.
    """
    gsm = GlobalSkyModel(
        freq_unit="Hz"
    )  # Healpix map in Galactic coordinates with nside=512
    sky_map = gsm.generate(frequency)  # Generate map at the given frequency (in Kelvin)

    # Create a rotator to transform from Galactic to Equatorial coordinates
    rot = Rotator(coord=["G", "C"])

    # Downgrade the map
    downgraded_map = hp.ud_grade(sky_map, nside_out=nside)

    # Rotate the map to Equatorial coordinates
    equatorial = rot.rotate_map_pixel(downgraded_map)

    # Get pixel indices for nside
    npix = hp.nside2npix(nside)
    pixel_indices = np.arange(npix)

    # Get theta and phi values for each pixel in Galactic coordinates
    theta_gal, phi_gal = hp.pix2ang(nside, pixel_indices)

    # Rotate the pixel to Equatorial coordinates
    theta_eq, phi_eq = rot(theta_gal, phi_gal)

    # Convert to RA/Dec
    ra = np.rad2deg(phi_eq)
    dec = 90 - np.rad2deg(theta_eq)

    # Constants
    k_B = 1.380649e-23  # Boltzmann constant in J/K
    c = 299792458  # Speed of light in m/s

    # Calculate flux density using beam area
    flux_density = (
        (2 * k_B * equatorial * (frequency**2)) / (c**2)
    ) * beam_area  # Flux density in W/m^2/Hz

    # Convert flux density to Jy (1 Jy = 1e-26 W/m^2/Hz)
    flux_density_jy = flux_density / 1e-26

    # Debugging: Log the range of flux densities
    print(
        f"Flux density range: {flux_density_jy.min():.2f} Jy to {flux_density_jy.max():.2f} Jy"
    )

    # Apply flux limit condition
    valid_indices = flux_density_jy >= flux_limit
    if valid_indices.sum() == 0:
        print("Warning: No sources meet the flux limit condition.")
    else:
        print(f"{valid_indices.sum()} sources meet the flux limit condition.")

    # Filter valid sources
    ra = ra[valid_indices]
    dec = dec[valid_indices]
    flux_density_jy = flux_density_jy[valid_indices]

    # Assign a default spectral index for GSM2008 sources
    spectral_index = 0.0

    # Create source dictionaries
    sources = [
        {
            "coords": SkyCoord(ra=r * au.deg, dec=d * au.deg, frame="icrs"),
            "flux": f,
            "spectral_index": spectral_index,
        }
        for r, d, f in zip(ra, dec, flux_density_jy)
    ]

    print(
        f"GSM2008 sources loaded with {len(sources)} sources after applying flux limit of {flux_limit} Jy."
    )

    # Measure the memory size of the GSM2008 sources
    gsm_size = asizeof.asizeof(sources)
    print(f"Total size: {gsm_size / (1024**2):.2f} MB")

    return sources, 0


def load_gleam_in_healpix(flux_limit=50, nside=32, ref_freq=76e6):
    """
    Load GLEAM catalog and map sources to HEALPix with flux and spectral index.

    Parameters:
    flux_limit (float): Minimum flux (in Jy) to include a source.
    nside (int): HEALPix NSIDE parameter (resolution).
    ref_freq (float): Reference frequency in Hz.

    Returns:
    tuple: Two HEALPix maps (flux_map, alpha_map).
    """
    Vizier.ROW_LIMIT = -1
    catalog = Vizier.get_catalogs("VIII/100/gleamegc")[0]

    # Initialize HEALPix maps
    npix = hp.nside2npix(nside)
    healpix_flux_map = np.zeros(npix)
    healpix_alpha_map = (
        np.zeros(npix) + np.nan
    )  # Start with NaNs for undefined spectral indices
    sources = []
    for row in catalog:
        flux = row["Fp076"]  # Flux at 76 MHz
        if flux >= flux_limit:
            ra = row["RAJ2000"] * au.deg
            dec = row["DEJ2000"] * au.deg
            coords = SkyCoord(ra=ra, dec=dec)
            spectral_index = row.get("alpha", 0.0)  # Default alpha = 0.0 if missing

            sources.append(
                {"coords": coords, "flux": flux, "spectral_index": spectral_index}
            )

    # Map each source to a HEALPix pixel
    for source in sources:
        ra, dec = source["coords"].ra.deg, source["coords"].dec.deg
        pixel_index = hp.ang2pix(nside, np.radians(90 - dec), np.radians(ra))
        healpix_flux_map[pixel_index] += source["flux"]
        healpix_alpha_map[pixel_index] = source["spectral_index"]

    return healpix_flux_map, healpix_alpha_map


def load_gsm_gleam_in_healpix():
    return None, None
