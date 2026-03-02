# ProactiveGuard

**Predictive failure detection for distributed consensus systems.**

ProactiveGuard uses machine learning to predict node failures in etcd, Raft, and CockroachDB clusters — giving you a warning window of up to 30 seconds before a crash or partition occurs.

## Install

```bash
pip install proactiveguard
```

Requires Python 3.9+ and no heavy ML dependencies — just `requests` and `numpy`.

## Authentication

Get your API key from [app.proactiveguard.io](https://app.proactiveguard.io) and set it as an environment variable:

```bash
export PROACTIVEGUARD_API_KEY="pg-..."
```

Or pass it directly when creating the client:

```python
from proactiveguard import ProactiveGuard

pg = ProactiveGuard(api_key="pg-...")
```

## Streaming Monitoring (Real-Time)

Feed one timestep of metrics per node at each collection interval. Once enough
observations accumulate, the API returns a prediction automatically.

```python
from proactiveguard import ProactiveGuard

pg = ProactiveGuard()  # reads PROACTIVEGUARD_API_KEY from environment

# Call this every collection interval (e.g. every second)
result = pg.observe("etcd-0", {
    "heartbeat_latency_ms": 22.5,
    "messages_sent": 12,
    "messages_received": 11,
    "messages_dropped": 0,
    "missed_heartbeats": 0,
    "response_rate": 1.0,
    "term": 4,
    "commit_index": 1042,
    "is_leader": False,
})

# result is None while the observation window is filling up
if result and result.is_pre_failure:
    print(f"[ALERT] {result.node_id}: {result.status}")
    print(f"        Risk score : {result.risk_score:.2f}")
    print(f"        Time left  : {result.time_to_failure:.0f}s")
    print(f"        Failure type: {result.failure_type}")

# Or poll the latest prediction at any time
result = pg.status("etcd-0")
if result and result.is_failed:
    print(f"Node {result.node_id} has already failed: {result.status}")
```

### PredictionResult fields

| Field | Type | Description |
|---|---|---|
| `node_id` | `str` | Node identifier |
| `status` | `str` | One of `healthy`, `degraded_30s`, `degraded_20s`, `degraded_10s`, `degraded_5s`, `failed_crash`, `failed_slow`, `failed_byzantine`, `failed_partition` |
| `risk_score` | `float` | 0.0 = healthy, 1.0 = certain failure |
| `confidence` | `float` | Model confidence in this prediction (0–1) |
| `time_to_failure` | `float \| None` | Estimated seconds until failure, `None` if healthy |
| `failure_type` | `str \| None` | `crash`, `slow`, `byzantine`, or `partition` |
| `is_healthy` | `bool` | `True` when `status == "healthy"` |
| `is_pre_failure` | `bool` | `True` when the node is degraded but not yet failed |
| `is_failed` | `bool` | `True` when the node has already failed |
| `probabilities` | `dict` | Full probability distribution over all nine classes |

## Batch Prediction

If you have pre-collected observation windows (e.g. for offline analysis or model evaluation),
you can run batch predictions directly:

```python
import numpy as np
from proactiveguard import ProactiveGuard

pg = ProactiveGuard()

# X shape: (n_samples, window_size=50, n_features=32)
X = np.random.rand(10, 50, 32).astype("float32")

labels = pg.predict(X)          # → array of strings, shape (10,)
probs  = pg.predict_proba(X)    # → float32 array, shape (10, 9)

# With time-to-failure and confidence
labels, ttf, conf = pg.predict_with_ttf(X)
for i in range(len(labels)):
    print(f"Sample {i}: {labels[i]}  ttf={ttf[i]:.0f}s  conf={conf[i]:.2f}")
```

## Resetting State

```python
pg.reset("etcd-0")   # reset one node
pg.reset()           # reset all nodes
```

## Error Handling

```python
from proactiveguard import ProactiveGuard
from proactiveguard.exceptions import AuthenticationError, APIError, RateLimitError

try:
    pg = ProactiveGuard()
    result = pg.observe("etcd-0", metrics)
except AuthenticationError:
    print("Bad API key — check PROACTIVEGUARD_API_KEY")
except RateLimitError:
    print("Rate limit hit — slow down or upgrade plan")
except APIError as e:
    print(f"API error {e.status_code}: {e}")
```

## Self-Hosted / Custom Endpoint

Point the client at your own deployment:

```python
pg = ProactiveGuard(
    api_key="pg-...",
    base_url="https://pg.internal.mycompany.com/v1",
    timeout=10,
)
```

## Links

- [Documentation](https://docs.proactiveguard.io)
- [Dashboard](https://app.proactiveguard.io)
- [GitHub](https://github.com/Prakhar998/proactiveguard)
- [Report an issue](https://github.com/Prakhar998/proactiveguard/issues)

---

Copyright (c) 2025 Maya Plus. All rights reserved.
