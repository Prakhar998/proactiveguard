"""
Tests for the proactiveguard SDK (API client).
Uses `responses` to mock HTTP calls — no real server needed.
"""

import numpy as np
import pytest
import responses as rsps_lib

from proactiveguard import ProactiveGuard, PredictionResult
from proactiveguard.exceptions import (
    APIError,
    AuthenticationError,
    RateLimitError,
)

BASE = "https://api.proactiveguard.io/v1"


# ── Fixtures ───────────────────────────────────────────────────────────────────


@pytest.fixture
def pg():
    return ProactiveGuard(api_key="pg-test-key")


@pytest.fixture
def healthy_result():
    return {
        "node_id": "node-0",
        "status": "healthy",
        "risk_score": 0.02,
        "confidence": 0.97,
        "time_to_failure": None,
        "failure_type": None,
        "probabilities": {"healthy": 0.97, "degraded_30s": 0.02, "degraded_20s": 0.01},
        "ready": True,
        "timestamp": 1700000000.0,
    }


# ── Auth tests ─────────────────────────────────────────────────────────────────


def test_missing_api_key_raises():
    import os
    os.environ.pop("PROACTIVEGUARD_API_KEY", None)
    with pytest.raises(AuthenticationError):
        ProactiveGuard()


def test_env_var_api_key(monkeypatch):
    monkeypatch.setenv("PROACTIVEGUARD_API_KEY", "pg-env-key")
    pg = ProactiveGuard()
    assert pg is not None


# ── observe() tests ────────────────────────────────────────────────────────────


@rsps_lib.activate
def test_observe_returns_none_when_not_ready(pg):
    rsps_lib.add(rsps_lib.POST, f"{BASE}/observe", json={"ready": False}, status=200)
    result = pg.observe("node-0", {"heartbeat_latency_ms": 20.0})
    assert result is None


@rsps_lib.activate
def test_observe_returns_result_when_ready(pg, healthy_result):
    rsps_lib.add(rsps_lib.POST, f"{BASE}/observe", json=healthy_result, status=200)
    result = pg.observe("node-0", {"heartbeat_latency_ms": 20.0})
    assert isinstance(result, PredictionResult)
    assert result.node_id == "node-0"
    assert result.is_healthy


# ── status() tests ─────────────────────────────────────────────────────────────


@rsps_lib.activate
def test_status_returns_none_when_not_ready(pg):
    rsps_lib.add(rsps_lib.GET, f"{BASE}/status/node-0", json={"ready": False}, status=200)
    assert pg.status("node-0") is None


@rsps_lib.activate
def test_status_returns_result(pg, healthy_result):
    rsps_lib.add(rsps_lib.GET, f"{BASE}/status/node-0", json=healthy_result, status=200)
    result = pg.status("node-0")
    assert isinstance(result, PredictionResult)
    assert result.status == "healthy"


# ── reset() tests ──────────────────────────────────────────────────────────────


@rsps_lib.activate
def test_reset_specific_node(pg):
    rsps_lib.add(rsps_lib.DELETE, f"{BASE}/reset/node-0", body=b"", status=204)
    pg.reset("node-0")  # should not raise


@rsps_lib.activate
def test_reset_all_nodes(pg):
    rsps_lib.add(rsps_lib.DELETE, f"{BASE}/reset", body=b"", status=204)
    pg.reset()  # should not raise


# ── predict() tests ────────────────────────────────────────────────────────────


@rsps_lib.activate
def test_predict_batch(pg):
    horizon_names = [
        "healthy", "degraded_30s", "degraded_20s", "degraded_10s", "degraded_5s",
        "failed_crash", "failed_slow", "failed_byzantine", "failed_partition",
    ]
    probs = [[0.9, 0.05, 0.02, 0.01, 0.01, 0.0, 0.0, 0.0, 0.01]] * 3

    rsps_lib.add(rsps_lib.GET, f"{BASE}/meta/horizon-names", json={"names": horizon_names})
    rsps_lib.add(rsps_lib.POST, f"{BASE}/predict", json={"probabilities": probs})

    X = np.random.randn(3, 50, 32).astype("float32")
    labels = pg.predict(X)
    assert labels.shape == (3,)
    assert all(l == "healthy" for l in labels)


@rsps_lib.activate
def test_predict_single_window_auto_expanded(pg):
    horizon_names = ["healthy", "degraded_30s", "degraded_20s", "degraded_10s",
                     "degraded_5s", "failed_crash", "failed_slow", "failed_byzantine",
                     "failed_partition"]
    probs = [[0.9, 0.05, 0.02, 0.01, 0.01, 0.0, 0.0, 0.0, 0.01]]

    rsps_lib.add(rsps_lib.GET, f"{BASE}/meta/horizon-names", json={"names": horizon_names})
    rsps_lib.add(rsps_lib.POST, f"{BASE}/predict", json={"probabilities": probs})

    X = np.random.randn(50, 32).astype("float32")  # single window, no batch dim
    labels = pg.predict(X)
    assert labels.shape == (1,)


@rsps_lib.activate
def test_predict_proba_shape(pg):
    probs = np.random.dirichlet(np.ones(9), size=4).tolist()
    rsps_lib.add(rsps_lib.POST, f"{BASE}/predict", json={"probabilities": probs})

    X = np.random.randn(4, 50, 32).astype("float32")
    result = pg.predict_proba(X)
    assert result.shape == (4, 9)
    np.testing.assert_allclose(result.sum(axis=1), 1.0, atol=1e-5)


@rsps_lib.activate
def test_predict_with_ttf(pg):
    rsps_lib.add(rsps_lib.POST, f"{BASE}/predict/ttf", json={
        "labels": ["healthy", "degraded_30s", "failed_crash"],
        "time_to_failure": [0.0, 30.0, 0.0],
        "confidence": [0.95, 0.88, 0.99],
    })

    X = np.random.randn(3, 50, 32).astype("float32")
    labels, ttf, conf = pg.predict_with_ttf(X)
    assert labels.shape == (3,)
    assert ttf.shape == (3,)
    assert conf.shape == (3,)
    assert (conf >= 0).all() and (conf <= 1).all()


# ── Error handling tests ───────────────────────────────────────────────────────


@rsps_lib.activate
def test_auth_error_on_401(pg):
    rsps_lib.add(rsps_lib.GET, f"{BASE}/status/node-0", status=401)
    with pytest.raises(AuthenticationError):
        pg.status("node-0")


@rsps_lib.activate
def test_rate_limit_error_on_429(pg):
    rsps_lib.add(rsps_lib.POST, f"{BASE}/observe", status=429)
    with pytest.raises(RateLimitError):
        pg.observe("node-0", {})


@rsps_lib.activate
def test_api_error_on_500(pg):
    rsps_lib.add(rsps_lib.POST, f"{BASE}/predict", json={"detail": "internal error"}, status=500)
    X = np.random.randn(2, 50, 32).astype("float32")
    with pytest.raises(APIError) as exc_info:
        pg.predict_proba(X)
    assert exc_info.value.status_code == 500


# ── PredictionResult tests ─────────────────────────────────────────────────────


class TestPredictionResult:
    def test_healthy_flags(self):
        r = PredictionResult(node_id="n", status="healthy", risk_score=0.01, confidence=0.95)
        assert r.is_healthy
        assert not r.is_pre_failure
        assert not r.is_failed

    def test_degraded_flags(self):
        for status in ["degraded_30s", "degraded_20s", "degraded_10s", "degraded_5s"]:
            r = PredictionResult(node_id="n", status=status, risk_score=0.5, confidence=0.8)
            assert r.is_pre_failure
            assert not r.is_healthy
            assert not r.is_failed

    def test_failed_flags(self):
        for status in ["failed_crash", "failed_slow", "failed_byzantine", "failed_partition"]:
            r = PredictionResult(node_id="n", status=status, risk_score=0.99, confidence=0.97)
            assert r.is_failed
            assert not r.is_healthy
            assert not r.is_pre_failure

    def test_to_dict_keys(self):
        r = PredictionResult(node_id="n", status="healthy", risk_score=0.0, confidence=1.0)
        d = r.to_dict()
        for key in ("node_id", "status", "risk_score", "confidence",
                    "is_healthy", "is_pre_failure", "is_failed", "timestamp"):
            assert key in d

    def test_repr(self):
        r = PredictionResult(node_id="n", status="healthy", risk_score=0.0, confidence=1.0)
        assert "healthy" in repr(r)


# ── repr test ──────────────────────────────────────────────────────────────────


def test_pg_repr(pg):
    assert "ProactiveGuard" in repr(pg)
    assert "api.proactiveguard.io" in repr(pg)
