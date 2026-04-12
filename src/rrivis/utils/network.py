"""Network connectivity detection for RRIvis.

Provides utilities to check internet connectivity and reachability of
specific services (VizieR, CASDA, pygdsm data, PySM3 data) that RRIVis
depends on for sky model downloads.
"""

import socket
import time
from dataclasses import dataclass, field
from typing import Any

from rrivis.utils.logging import get_logger

logger = get_logger("rrivis.utils.network")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_GENERAL_HOST = "8.8.8.8"
_GENERAL_PORT = 53
_GENERAL_TIMEOUT = 2.0
_SERVICE_TIMEOUT = 3.0
_CACHE_TTL = 300.0  # seconds

SERVICE_ENDPOINTS: dict[str, tuple[str, int]] = {
    "vizier": ("vizier.cds.unistra.fr", 443),
    "casda": ("casda.csiro.au", 443),
    "pygdsm_data": ("zenodo.org", 443),
    "pysm3_data": ("portal.nersc.gov", 443),
}

# Maps source kind -> service name for sky models that need network.
_SKY_MODEL_SERVICES_CACHE: dict[str, str] | None = None


def get_sky_model_services() -> dict[str, str]:
    """Return the source-kind -> network-service mapping."""
    global _SKY_MODEL_SERVICES_CACHE
    if _SKY_MODEL_SERVICES_CACHE is None:
        from rrivis.core.sky.registry import build_network_services_map

        _SKY_MODEL_SERVICES_CACHE = build_network_services_map()
    return _SKY_MODEL_SERVICES_CACHE


# Human-readable display names for services.
SERVICE_DISPLAY_NAMES: dict[str, str] = {
    "vizier": "VizieR",
    "casda": "CASDA",
    "pygdsm_data": "pygdsm data",
    "pysm3_data": "PySM3 data",
}

# ---------------------------------------------------------------------------
# NetworkStatus dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NetworkStatus:
    """Result of a network connectivity check.

    Attributes
    ----------
    internet : bool
        Whether general internet connectivity is available.
    vizier : bool or None
        Whether VizieR is reachable (None if not checked).
    casda : bool or None
        Whether CASDA is reachable (None if not checked).
    pygdsm_data : bool or None
        Whether Zenodo (pygdsm data host) is reachable (None if not checked).
    pysm3_data : bool or None
        Whether NERSC portal (PySM3 data host) is reachable (None if not checked).
    timestamp : float
        Monotonic time when the check was performed.
    forced_offline : bool
        True if offline mode was forced via --offline flag or config.
    """

    internet: bool
    vizier: bool | None = None
    casda: bool | None = None
    pygdsm_data: bool | None = None
    pysm3_data: bool | None = None
    timestamp: float = field(default_factory=time.monotonic)
    forced_offline: bool = False

    @property
    def is_online(self) -> bool:
        """Whether the system is considered online.

        Returns False if offline mode is forced, or if general internet
        connectivity is unavailable.
        """
        return not self.forced_offline and self.internet

    def service_available(self, name: str) -> bool | None:
        """Get the availability of a named service.

        Parameters
        ----------
        name : str
            Service name (one of: vizier, casda, pygdsm_data, pysm3_data).

        Returns
        -------
        bool or None
            True/False if checked, None if not checked.

        Raises
        ------
        ValueError
            If the service name is unknown.
        """
        if name not in SERVICE_ENDPOINTS:
            raise ValueError(
                f"Unknown service {name!r}. "
                f"Known services: {', '.join(SERVICE_ENDPOINTS)}"
            )
        return getattr(self, name)

    def to_dict(self) -> dict[str, Any]:
        """Convert to a plain dictionary."""
        return {
            "internet": self.internet,
            "vizier": self.vizier,
            "casda": self.casda,
            "pygdsm_data": self.pygdsm_data,
            "pysm3_data": self.pysm3_data,
            "timestamp": self.timestamp,
            "forced_offline": self.forced_offline,
            "is_online": self.is_online,
        }


# ---------------------------------------------------------------------------
# Module-level cache
# ---------------------------------------------------------------------------

_cached_status: NetworkStatus | None = None


# ---------------------------------------------------------------------------
# Low-level check
# ---------------------------------------------------------------------------


def _check_socket(host: str, port: int, timeout: float) -> bool:
    """Attempt a TCP connection to *host*:*port*.

    Returns True if the connection succeeds within *timeout* seconds,
    False otherwise.
    """
    try:
        conn = socket.create_connection((host, port), timeout=timeout)
        conn.close()
        logger.debug("Socket check %s:%d — reachable", host, port)
        return True
    except OSError:
        logger.debug("Socket check %s:%d — unreachable", host, port)
        return False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def is_online(timeout: float = _GENERAL_TIMEOUT) -> bool:
    """Quick check for general internet connectivity.

    Uses a cached result if available and not expired (TTL = 300 s).

    Parameters
    ----------
    timeout : float
        TCP connection timeout in seconds (default 2).

    Returns
    -------
    bool
        True if the internet appears reachable.
    """
    global _cached_status

    now = time.monotonic()
    if (
        _cached_status is not None
        and not _cached_status.forced_offline
        and (now - _cached_status.timestamp) < _CACHE_TTL
    ):
        return _cached_status.internet

    result = _check_socket(_GENERAL_HOST, _GENERAL_PORT, timeout)
    _cached_status = NetworkStatus(internet=result, timestamp=now)
    return result


def check_service(service: str, timeout: float = _SERVICE_TIMEOUT) -> bool:
    """Check whether a specific service is reachable.

    Parameters
    ----------
    service : str
        Service name (one of: vizier, casda, pygdsm_data, pysm3_data).
    timeout : float
        TCP connection timeout in seconds (default 3).

    Returns
    -------
    bool
        True if the service endpoint is reachable.

    Raises
    ------
    ValueError
        If *service* is not a known service name.
    """
    if service not in SERVICE_ENDPOINTS:
        raise ValueError(
            f"Unknown service {service!r}. "
            f"Known services: {', '.join(SERVICE_ENDPOINTS)}"
        )
    host, port = SERVICE_ENDPOINTS[service]
    return _check_socket(host, port, timeout)


def check_all_services(timeout: float = _SERVICE_TIMEOUT) -> NetworkStatus:
    """Check general internet connectivity and all known services.

    Always performs fresh checks (ignores cache). Updates the module-level
    cache with the result.

    Parameters
    ----------
    timeout : float
        TCP connection timeout per endpoint in seconds (default 3).

    Returns
    -------
    NetworkStatus
        Full status with all fields populated.
    """
    global _cached_status

    now = time.monotonic()
    internet = _check_socket(
        _GENERAL_HOST, _GENERAL_PORT, min(timeout, _GENERAL_TIMEOUT)
    )

    service_results: dict[str, bool] = {}
    for name, (host, port) in SERVICE_ENDPOINTS.items():
        service_results[name] = _check_socket(host, port, timeout)

    status = NetworkStatus(
        internet=internet,
        timestamp=now,
        **service_results,
    )
    _cached_status = status
    return status


def get_network_status(offline: bool = False) -> NetworkStatus:
    """Get network status, respecting the offline flag.

    This is the primary entry point used by :class:`~rrivis.api.simulator.Simulator`.

    Parameters
    ----------
    offline : bool
        If True, return a forced-offline status with no network I/O.

    Returns
    -------
    NetworkStatus
        Current network status.
    """
    if offline:
        return NetworkStatus(
            internet=False,
            forced_offline=True,
            timestamp=time.monotonic(),
        )
    online = is_online()
    return NetworkStatus(internet=online, timestamp=time.monotonic())


def get_required_services(sky_config: dict) -> dict[str, list[str]]:
    """Determine which network services are needed by enabled sky models.

    Parameters
    ----------
    sky_config : dict
        The ``sky_model`` section of the RRIVis configuration (as a dict).

    Returns
    -------
    dict[str, list[str]]
        Mapping of service name to list of model names that require it.
        Only services needed by at least one enabled model are included.
    """
    required: dict[str, list[str]] = {}
    sources = sky_config.get("sources")
    if isinstance(sources, list):
        alias_map = {}
        try:
            from rrivis.core.sky.registry import build_alias_map

            alias_map = build_alias_map()
        except Exception:
            alias_map = {}

        services = get_sky_model_services()
        for entry in sources:
            if not isinstance(entry, dict):
                continue
            raw_kind = entry.get("kind")
            if not raw_kind:
                continue
            kind = alias_map.get(raw_kind, raw_kind)
            service = services.get(kind)
            if service is not None:
                required.setdefault(service, []).append(kind)
        return required

    # Fallback for legacy config shape.
    for kind, service in get_sky_model_services().items():
        sub = sky_config.get(kind, {})
        if isinstance(sub, dict):
            enabled = any(
                bool(v)
                for k, v in sub.items()
                if isinstance(k, str) and k.startswith("use_")
            )
            if enabled:
                required.setdefault(service, []).append(kind)
    return required


def require_service(service: str, action: str, *, strict: bool = True) -> None:
    """Check network connectivity for a service.

    When ``strict=True`` (default), raises :class:`ConnectionError` if the
    service is unreachable -- appropriate for loaders that *must* download
    data (e.g. VizieR catalogs).  When ``strict=False``, logs a warning
    instead -- appropriate for services whose data may already be cached
    locally (e.g. pygdsm, PySM3).

    Parameters
    ----------
    service : str
        Service name (e.g. ``"vizier"``, ``"casda"``, ``"pygdsm_data"``,
        ``"pysm3_data"``).
    action : str
        Human-readable description of what needs the service
        (e.g. ``"download catalog 'gleam' from VizieR"``).
    strict : bool, default True
        If True, raise :class:`ConnectionError` when unavailable.
        If False, log a warning instead (for services with local caches).

    Raises
    ------
    ConnectionError
        Only when ``strict=True`` and the service is unreachable.
    """
    display = SERVICE_DISPLAY_NAMES.get(service, service)

    if not is_online():
        msg = (
            f"No internet connection. Cannot {action}.\n"
            f"Hint: use offline metadata methods like "
            f"SkyModel.get_catalog_info(key) or SkyModel.list_point_catalogs() "
            f"which work without network."
        )
        if strict:
            raise ConnectionError(msg)
        logger.warning(
            f"No internet connection. {action.capitalize()} may fail if data "
            f"files have not been downloaded previously."
        )
        return

    if not check_service(service):
        msg = (
            f"{display} ({service}) is unreachable. Cannot {action}.\n"
            f"The service may be temporarily down. Try again later, or use "
            f"SkyModel.get_catalog_info(key) for offline metadata."
        )
        if strict:
            raise ConnectionError(msg)
        logger.warning(
            f"{display} is unreachable. {action.capitalize()} may fail if data "
            f"files have not been cached locally from a previous run."
        )


def clear_cache() -> None:
    """Reset the module-level cached network status.

    Intended for use in tests.
    """
    global _cached_status, _SKY_MODEL_SERVICES_CACHE
    _cached_status = None
    _SKY_MODEL_SERVICES_CACHE = None
