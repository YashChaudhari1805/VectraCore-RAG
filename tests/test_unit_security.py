"""
tests/test_unit_security.py
---------------------------
Unit tests for ``api.security`` — API key authentication and rate limiting.
"""

from __future__ import annotations

import time
from collections import deque
from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException

from api.security import _rate_windows, check_rate_limit, verify_api_key


# ---------------------------------------------------------------------------
# verify_api_key
# ---------------------------------------------------------------------------


class TestVerifyApiKey:
    """Tests for the ``verify_api_key`` dependency."""

    def test_passes_when_auth_disabled(self) -> None:
        """No exception raised when ``auth_enabled`` is False."""
        with patch("api.security.settings") as mock_settings:
            mock_settings.auth_enabled = False
            verify_api_key(x_api_key=None)

    def test_raises_401_when_key_missing(self) -> None:
        """401 raised when auth is enabled and no key is provided."""
        with patch("api.security.settings") as mock_settings:
            mock_settings.auth_enabled = True
            mock_settings.api_keys = ["secret-key"]
            with pytest.raises(HTTPException) as exc_info:
                verify_api_key(x_api_key=None)
            assert exc_info.value.status_code == 401

    def test_raises_403_when_key_invalid(self) -> None:
        """403 raised when auth is enabled and the key does not match."""
        with patch("api.security.settings") as mock_settings:
            mock_settings.auth_enabled = True
            mock_settings.api_keys = ["correct-key"]
            with pytest.raises(HTTPException) as exc_info:
                verify_api_key(x_api_key="wrong-key")
            assert exc_info.value.status_code == 403

    def test_passes_with_valid_key(self) -> None:
        """No exception raised when auth is enabled and the correct key is supplied."""
        with patch("api.security.settings") as mock_settings:
            mock_settings.auth_enabled = True
            mock_settings.api_keys = ["valid-key"]
            verify_api_key(x_api_key="valid-key")


# ---------------------------------------------------------------------------
# check_rate_limit
# ---------------------------------------------------------------------------


class TestCheckRateLimit:
    """Tests for the ``check_rate_limit`` dependency."""

    def _make_request(self, ip: str = "127.0.0.1") -> MagicMock:
        req = MagicMock()
        req.headers = {}
        req.client.host = ip
        return req

    def test_allows_requests_under_limit(self) -> None:
        """Requests below the limit should pass without error."""
        _rate_windows.clear()
        request = self._make_request(ip="10.0.0.1")
        with patch("api.security.settings") as mock_settings:
            mock_settings.rate_limit_per_minute = 5
            for _ in range(5):
                check_rate_limit(request)

    def test_blocks_when_limit_exceeded(self) -> None:
        """429 raised once the per-minute limit is exceeded."""
        _rate_windows.clear()
        request = self._make_request(ip="10.0.0.2")
        with patch("api.security.settings") as mock_settings:
            mock_settings.rate_limit_per_minute = 3
            for _ in range(3):
                check_rate_limit(request)
            with pytest.raises(HTTPException) as exc_info:
                check_rate_limit(request)
            assert exc_info.value.status_code == 429

    def test_window_expires(self) -> None:
        """Requests older than 60 seconds should not count toward the limit."""
        _rate_windows.clear()
        ip = "10.0.0.3"
        old_time = time.monotonic() - 61.0
        _rate_windows[ip] = deque([old_time, old_time, old_time])

        request = self._make_request(ip=ip)
        with patch("api.security.settings") as mock_settings:
            mock_settings.rate_limit_per_minute = 3
            check_rate_limit(request)

    def test_different_ips_tracked_separately(self) -> None:
        """Rate limit windows must be independent per client IP."""
        _rate_windows.clear()
        req_a = self._make_request(ip="192.168.1.1")
        req_b = self._make_request(ip="192.168.1.2")

        with patch("api.security.settings") as mock_settings:
            mock_settings.rate_limit_per_minute = 2
            for _ in range(2):
                check_rate_limit(req_a)
            check_rate_limit(req_b)

    def test_respects_x_forwarded_for(self) -> None:
        """Client IP should be read from X-Forwarded-For when present."""
        _rate_windows.clear()
        req = MagicMock()
        req.headers = {"X-Forwarded-For": "203.0.113.5, 10.0.0.1"}
        req.client.host = "10.0.0.1"

        with patch("api.security.settings") as mock_settings:
            mock_settings.rate_limit_per_minute = 10
            check_rate_limit(req)

        assert "203.0.113.5" in _rate_windows
