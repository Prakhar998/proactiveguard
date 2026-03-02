"""
ProactiveGuard — Predictive failure detection for distributed consensus systems.

Quick start
-----------
    from proactiveguard import ProactiveGuard

    pg = ProactiveGuard(api_key="pg-...")

    # Streaming monitoring
    pg.observe("node-0", metrics)
    result = pg.status("node-0")
    if result and result.is_pre_failure:
        print(f"Warning: {result.status} — {result.time_to_failure:.0f}s to failure")

    # Batch prediction
    labels = pg.predict(X)           # (n, 50, 32) numpy array
    probs  = pg.predict_proba(X)

Copyright (c) 2025 Maya Plus. All rights reserved.
"""

from __future__ import annotations

import os
from typing import List, Optional

import numpy as np

from ._http import _DEFAULT_BASE_URL, _DEFAULT_TIMEOUT, HTTPClient
from .exceptions import APIError, AuthenticationError, InsufficientDataError, RateLimitError
from .types import PredictionResult

__version__ = "0.1.0"
__all__ = [
    "ProactiveGuard",
    "PredictionResult",
    "AuthenticationError",
    "APIError",
    "RateLimitError",
    "InsufficientDataError",
]


class ProactiveGuard:
    """
    ProactiveGuard API client.

    Parameters
    ----------
    api_key:
        Your ProactiveGuard API key (``pg-...``).
        Falls back to the ``PROACTIVEGUARD_API_KEY`` environment variable.
    base_url:
        Override the API base URL (useful for self-hosted deployments).
    timeout:
        HTTP request timeout in seconds.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        *,
        base_url: str = _DEFAULT_BASE_URL,
        timeout: int = _DEFAULT_TIMEOUT,
    ) -> None:
        resolved_key = api_key or os.environ.get("PROACTIVEGUARD_API_KEY", "")
        if not resolved_key:
            raise AuthenticationError()
        self._http = HTTPClient(resolved_key, base_url, timeout)

    # ── Streaming / monitoring API ─────────────────────────────────────────────

    def observe(
        self,
        node_id: str,
        metrics: dict,
    ) -> Optional[PredictionResult]:
        """
        Feed one timestep of metrics for a node.

        Returns a :class:`PredictionResult` once enough observations have been
        collected server-side, or ``None`` if the window is still filling.

        Parameters
        ----------
        node_id:
            Cluster node identifier, e.g. ``"etcd-0"``.
        metrics:
            Dict of metric values.  Recognised keys mirror the ``Observation``
            fields: ``heartbeat_latency_ms``, ``messages_sent``,
            ``messages_received``, ``messages_dropped``, ``missed_heartbeats``,
            ``response_rate``, ``term``, ``commit_index``, ``is_leader``, etc.
        """
        body = {"node_id": node_id, "metrics": metrics}
        data = self._http.post("/observe", body)
        if data.get("ready") is False:
            return None
        return self._parse_result(data)

    def status(self, node_id: str) -> Optional[PredictionResult]:
        """Return the latest prediction for ``node_id``, or None if not ready."""
        data = self._http.get(f"/status/{node_id}")
        if not data or data.get("ready") is False:
            return None
        return self._parse_result(data)

    def reset(self, node_id: Optional[str] = None) -> None:
        """
        Clear observation windows server-side.

        Parameters
        ----------
        node_id:
            If given, reset only that node. If None, reset all nodes.
        """
        path = f"/reset/{node_id}" if node_id else "/reset"
        self._http.delete(path)

    # ── Batch prediction API ───────────────────────────────────────────────────

    def predict(self, X: "np.ndarray") -> "np.ndarray":
        """
        Predict failure class labels for a batch of observation windows.

        Parameters
        ----------
        X:
            Shape ``(n_samples, window_size, n_features)`` or a single window
            ``(window_size, n_features)``.

        Returns
        -------
        np.ndarray
            String labels of shape ``(n_samples,)``.
        """
        probs = self.predict_proba(X)
        class_ids = np.argmax(probs, axis=1)
        labels_map: List[str] = self._http.get("/meta/horizon-names")["names"]
        return np.array([labels_map[i] for i in class_ids])

    def predict_proba(self, X: "np.ndarray") -> "np.ndarray":
        """
        Return class probability array.

        Parameters
        ----------
        X:
            Shape ``(n_samples, window_size, n_features)`` or ``(window_size, n_features)``.

        Returns
        -------
        np.ndarray
            Shape ``(n_samples, n_classes)`` — each row sums to 1.
        """
        X = np.asarray(X, dtype=np.float32)
        if X.ndim == 2:
            X = X[np.newaxis]
        body = {"windows": X.tolist()}
        data = self._http.post("/predict", body)
        return np.array(data["probabilities"], dtype=np.float32)

    def predict_with_ttf(self, X: "np.ndarray") -> "tuple[np.ndarray, np.ndarray, np.ndarray]":
        """
        Return ``(class_labels, time_to_failure_seconds, confidence)``.

        Parameters
        ----------
        X:
            Shape ``(n_samples, window_size, n_features)`` or ``(window_size, n_features)``.
        """
        X = np.asarray(X, dtype=np.float32)
        if X.ndim == 2:
            X = X[np.newaxis]
        body = {"windows": X.tolist()}
        data = self._http.post("/predict/ttf", body)
        labels = np.array(data["labels"])
        ttf = np.array(data["time_to_failure"], dtype=np.float32)
        conf = np.array(data["confidence"], dtype=np.float32)
        return labels, ttf, conf

    # ── Internal helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _parse_result(data: dict) -> PredictionResult:
        return PredictionResult(
            node_id=data["node_id"],
            status=data["status"],
            risk_score=float(data["risk_score"]),
            confidence=float(data["confidence"]),
            time_to_failure=data.get("time_to_failure"),
            failure_type=data.get("failure_type"),
            probabilities=data.get("probabilities", {}),
            timestamp=float(data.get("timestamp", 0)),
        )

    def __repr__(self) -> str:
        return f"ProactiveGuard(base_url={self._http.base_url!r})"
