"""Tests for rrivis.utils.network connectivity detection."""

from unittest.mock import MagicMock, patch

import pytest

from rrivis.utils.network import (
    SERVICE_ENDPOINTS,
    NetworkStatus,
    _check_socket,
    check_all_services,
    check_service,
    clear_cache,
    get_network_status,
    get_required_services,
    get_sky_model_services,
    is_online,
)


@pytest.fixture(autouse=True)
def _clear_network_cache():
    """Ensure a clean cache for every test."""
    clear_cache()
    yield
    clear_cache()


# ---------------------------------------------------------------------------
# _check_socket
# ---------------------------------------------------------------------------


class TestCheckSocket:
    @patch("rrivis.utils.network.socket.create_connection")
    def test_reachable(self, mock_conn):
        mock_sock = MagicMock()
        mock_conn.return_value = mock_sock
        assert _check_socket("example.com", 443, 2.0) is True
        mock_conn.assert_called_once_with(("example.com", 443), timeout=2.0)
        mock_sock.close.assert_called_once()

    @patch("rrivis.utils.network.socket.create_connection", side_effect=OSError("fail"))
    def test_unreachable(self, mock_conn):
        assert _check_socket("example.com", 443, 2.0) is False

    @patch(
        "rrivis.utils.network.socket.create_connection",
        side_effect=TimeoutError("timed out"),
    )
    def test_timeout(self, mock_conn):
        assert _check_socket("example.com", 443, 2.0) is False


# ---------------------------------------------------------------------------
# is_online
# ---------------------------------------------------------------------------


class TestIsOnline:
    @patch("rrivis.utils.network._check_socket", return_value=True)
    def test_online(self, mock_check):
        assert is_online() is True
        mock_check.assert_called_once()

    @patch("rrivis.utils.network._check_socket", return_value=False)
    def test_offline(self, mock_check):
        assert is_online() is False

    @patch("rrivis.utils.network._check_socket", return_value=True)
    def test_cached(self, mock_check):
        assert is_online() is True
        assert is_online() is True
        # Second call should use cache, not check again.
        assert mock_check.call_count == 1

    @patch("rrivis.utils.network._check_socket", return_value=True)
    @patch("rrivis.utils.network.time.monotonic")
    def test_cache_expiry(self, mock_time, mock_check):
        mock_time.return_value = 0.0
        assert is_online() is True
        assert mock_check.call_count == 1

        # Advance time beyond TTL (300s)
        mock_time.return_value = 301.0
        assert is_online() is True
        assert mock_check.call_count == 2


# ---------------------------------------------------------------------------
# check_service
# ---------------------------------------------------------------------------


class TestCheckService:
    @patch("rrivis.utils.network._check_socket", return_value=True)
    def test_known_service(self, mock_check):
        for name in SERVICE_ENDPOINTS:
            assert check_service(name) is True

    @patch("rrivis.utils.network._check_socket", return_value=False)
    def test_unreachable_service(self, mock_check):
        assert check_service("vizier") is False

    def test_unknown_service(self):
        with pytest.raises(ValueError, match="Unknown service"):
            check_service("nonexistent")


# ---------------------------------------------------------------------------
# check_all_services
# ---------------------------------------------------------------------------


class TestCheckAllServices:
    @patch("rrivis.utils.network._check_socket", return_value=True)
    def test_all_online(self, mock_check):
        status = check_all_services()
        assert status.internet is True
        assert status.vizier is True
        assert status.casda is True
        assert status.pygdsm_data is True
        assert status.pysm3_data is True
        assert status.is_online is True
        assert status.forced_offline is False
        # General + 4 services = 5 calls
        assert mock_check.call_count == 5

    @patch("rrivis.utils.network._check_socket", return_value=False)
    def test_all_offline(self, mock_check):
        status = check_all_services()
        assert status.internet is False
        assert status.vizier is False
        assert status.is_online is False

    @patch("rrivis.utils.network._check_socket")
    def test_partial_reachability(self, mock_check):
        def side_effect(host, port, timeout):
            if host == "8.8.8.8":
                return True
            if host == "vizier.cds.unistra.fr":
                return True
            return False

        mock_check.side_effect = side_effect
        status = check_all_services()
        assert status.internet is True
        assert status.vizier is True
        assert status.casda is False
        assert status.pygdsm_data is False
        assert status.pysm3_data is False
        assert status.is_online is True


# ---------------------------------------------------------------------------
# get_network_status
# ---------------------------------------------------------------------------


class TestGetNetworkStatus:
    @patch("rrivis.utils.network._check_socket", return_value=True)
    def test_online(self, mock_check):
        status = get_network_status()
        assert status.internet is True
        assert status.is_online is True
        assert status.forced_offline is False
        # Services should not be checked (general check only)
        assert status.vizier is None

    def test_forced_offline(self):
        status = get_network_status(offline=True)
        assert status.internet is False
        assert status.is_online is False
        assert status.forced_offline is True
        # No network I/O should have occurred — nothing to mock.

    @patch("rrivis.utils.network._check_socket", return_value=True)
    def test_forced_offline_skips_network_io(self, mock_check):
        status = get_network_status(offline=True)
        assert status.forced_offline is True
        mock_check.assert_not_called()


# ---------------------------------------------------------------------------
# NetworkStatus
# ---------------------------------------------------------------------------


class TestNetworkStatus:
    def test_is_online_true(self):
        s = NetworkStatus(internet=True)
        assert s.is_online is True

    def test_is_online_false_no_internet(self):
        s = NetworkStatus(internet=False)
        assert s.is_online is False

    def test_is_online_false_forced_offline(self):
        s = NetworkStatus(internet=True, forced_offline=True)
        assert s.is_online is False

    def test_service_available(self):
        s = NetworkStatus(internet=True, vizier=True, casda=False)
        assert s.service_available("vizier") is True
        assert s.service_available("casda") is False
        assert s.service_available("pygdsm_data") is None

    def test_service_available_unknown(self):
        s = NetworkStatus(internet=True)
        with pytest.raises(ValueError, match="Unknown service"):
            s.service_available("unknown")

    def test_to_dict(self):
        s = NetworkStatus(internet=True, vizier=False, timestamp=100.0)
        d = s.to_dict()
        assert d["internet"] is True
        assert d["vizier"] is False
        assert d["is_online"] is True
        assert d["timestamp"] == 100.0

    def test_frozen(self):
        s = NetworkStatus(internet=True)
        with pytest.raises(AttributeError):
            s.internet = False


# ---------------------------------------------------------------------------
# get_required_services
# ---------------------------------------------------------------------------


class TestGetRequiredServices:
    def test_no_network_models(self):
        sky_config = {
            "sources": [{"kind": "test_sources"}],
        }
        assert get_required_services(sky_config) == {}

    def test_vizier_model(self):
        sky_config = {
            "sources": [{"kind": "gleam"}, {"kind": "nvss"}],
        }
        result = get_required_services(sky_config)
        assert "vizier" in result
        assert "gleam" in result["vizier"]
        assert "nvss" in result["vizier"]

    def test_casda_model(self):
        sky_config = {"sources": [{"kind": "racs"}]}
        result = get_required_services(sky_config)
        assert result == {"casda": ["racs"]}

    def test_diffuse_models(self):
        sky_config = {
            "sources": [{"kind": "gsm2008"}, {"kind": "pysm3"}],
        }
        result = get_required_services(sky_config)
        assert "pygdsm_data" in result
        assert "pysm3_data" in result

    def test_disabled_models_excluded(self):
        sky_config = {
            "sources": [],
        }
        assert get_required_services(sky_config) == {}

    def test_missing_config_keys(self):
        assert get_required_services({}) == {}

    def test_all_vizier_models_mapped(self):
        """Every model in get_sky_model_services()
        should map to a known service endpoint."""
        for _key, service in get_sky_model_services().items():
            assert service in SERVICE_ENDPOINTS, (
                f"get_sky_model_services() maps {_key!r} to unknown service {service!r}"
            )


# ---------------------------------------------------------------------------
# clear_cache
# ---------------------------------------------------------------------------


class TestClearCache:
    @patch("rrivis.utils.network._check_socket", return_value=True)
    def test_clear_forces_recheck(self, mock_check):
        is_online()
        assert mock_check.call_count == 1
        clear_cache()
        is_online()
        assert mock_check.call_count == 2
