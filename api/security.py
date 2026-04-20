"""
api/security.py
---------------
Authentication and rate-limiting middleware for the Grid07 API.

Authentication:
    When ``settings.auth_enabled`` is ``True``, every request must supply a
    valid API key via the ``X-API-Key`` header.  Keys are configured through
    the ``API_KEYS`` environment variable (comma-separated list).

Rate limiting:
    A sliding-window in-memory counter limits each client IP to
    ``settings.rate_limit_per_minute`` requests per 60-second window.
    This is intentionally simple — for production at scale, replace with
    a Redis-backed solution (e.g. ``slowapi`` + Redis backend).
"""

import time
from collections import defaultdict, deque
from typing import Optional

from fastapi import Header, HTTPException, Request, status

from core.config import settings
from core.logging_config import get_logger

log = get_logger(__name__)

_rate_windows: dict[str, deque[float]] = defaultdict(deque)


def _client_ip(request: Request) -> str:
    """
    Extract the real client IP from the request.

    Respects the ``X-Forwarded-For`` header when the app runs behind a
    reverse proxy (nginx, AWS ALB, etc.).

    Args:
        request: The incoming FastAPI ``Request`` object.

    Returns:
        The client IP address as a string.
    """
    forwarded_for: Optional[str] = request.headers.get("X-Forwarded-For")
    if forwarded_for:
        return forwarded_for.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def check_rate_limit(request: Request) -> None:
    """
    Enforce a sliding-window rate limit on the requesting client IP.

    Args:
        request: The incoming FastAPI ``Request`` object.

    Raises:
        HTTPException 429: When the client exceeds the configured request rate.
    """
    ip = _client_ip(request)
    window = _rate_windows[ip]
    now = time.monotonic()
    cutoff = now - 60.0

    while window and window[0] < cutoff:
        window.popleft()

    if len(window) >= settings.rate_limit_per_minute:
        log.warning("rate_limit_exceeded", client_ip=ip, window_size=len(window))
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit exceeded: max {settings.rate_limit_per_minute} requests/minute.",
        )

    window.append(now)


def verify_api_key(x_api_key: Optional[str] = Header(default=None)) -> None:
    """
    FastAPI dependency that validates the ``X-API-Key`` header.

    When ``settings.auth_enabled`` is ``False`` (no keys configured), all
    requests are permitted without authentication.

    Args:
        x_api_key: Value of the ``X-API-Key`` request header.

    Raises:
        HTTPException 401: When auth is enabled and no key is supplied.
        HTTPException 403: When auth is enabled and the key is invalid.
    """
    if not settings.auth_enabled:
        return

    if x_api_key is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing X-API-Key header.",
        )

    if x_api_key not in settings.api_keys:
        log.warning("invalid_api_key", key_prefix=x_api_key[:6])
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid API key.",
        )
