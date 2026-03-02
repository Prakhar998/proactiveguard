from __future__ import annotations


class ProactiveGuardError(Exception):
    """Base exception for all ProactiveGuard SDK errors."""


class AuthenticationError(ProactiveGuardError):
    """Raised when the API key is missing or invalid (HTTP 401/403)."""

    def __init__(self) -> None:
        super().__init__(
            "Invalid or missing API key. "
            "Pass api_key= or set the PROACTIVEGUARD_API_KEY environment variable."
        )


class APIError(ProactiveGuardError):
    """Raised when the API returns an unexpected error response."""

    def __init__(self, status_code: int, message: str) -> None:
        self.status_code = status_code
        super().__init__(f"API error {status_code}: {message}")


class RateLimitError(ProactiveGuardError):
    """Raised when the API rate limit is exceeded (HTTP 429)."""

    def __init__(self) -> None:
        super().__init__("Rate limit exceeded. Slow down requests or upgrade your plan.")


class InsufficientDataError(ProactiveGuardError):
    """Raised when not enough observations have been collected for a prediction."""

    def __init__(self, node_id: str, have: int, need: int) -> None:
        super().__init__(f"Node '{node_id}' has {have} observations, needs {need} to predict.")
