"""Low-level HTTP client — single requests.Session, shared across calls."""

from __future__ import annotations

from typing import Any, Dict

import requests

from .exceptions import APIError, AuthenticationError, RateLimitError

_DEFAULT_BASE_URL = "https://api.proactiveguard.io/v1"
_DEFAULT_TIMEOUT = 30


class HTTPClient:
    def __init__(self, api_key: str, base_url: str, timeout: int) -> None:
        self._session = requests.Session()
        self._session.headers.update(
            {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "User-Agent": "proactiveguard-python/0.2.1",
            }
        )
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def get(self, path: str, **params: Any) -> Dict:
        resp = self._session.get(
            f"{self.base_url}{path}", params=params or None, timeout=self.timeout
        )
        return self._handle(resp)

    def post(self, path: str, body: Dict) -> Dict:
        resp = self._session.post(f"{self.base_url}{path}", json=body, timeout=self.timeout)
        return self._handle(resp)

    def delete(self, path: str) -> Dict:
        resp = self._session.delete(f"{self.base_url}{path}", timeout=self.timeout)
        return self._handle(resp)

    @staticmethod
    def _handle(resp: requests.Response) -> Dict:
        if resp.status_code in (401, 403):
            raise AuthenticationError()
        if resp.status_code == 429:
            raise RateLimitError()
        if not resp.ok:
            try:
                msg = resp.json().get("detail", resp.text)
            except Exception:
                msg = resp.text
            raise APIError(resp.status_code, msg)
        if resp.status_code == 204:
            return {}
        return resp.json()
