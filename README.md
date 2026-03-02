# ProactiveGuard

Predictive failure detection for distributed consensus systems (etcd, Raft, CockroachDB).

## Install

```bash
pip install proactiveguard
```

## Quick Start

```python
from proactiveguard import ProactiveGuard

pg = ProactiveGuard(api_key="pg-...")

# Stream metrics from your cluster nodes
pg.observe("etcd-0", {
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

result = pg.status("etcd-0")
if result and result.is_pre_failure:
    print(f"Warning: {result.status} — {result.time_to_failure:.0f}s to failure")

# Batch prediction
labels = pg.predict(X)           # X: (n, 50, 32) numpy array
probs  = pg.predict_proba(X)
labels, ttf, conf = pg.predict_with_ttf(X)
```

## Authentication

Pass your API key directly or via environment variable:

```bash
export PROACTIVEGUARD_API_KEY="pg-..."
```

```python
pg = ProactiveGuard()  # picks up from env
```

## Links

- [Documentation](https://docs.proactiveguard.io)
- [Dashboard](https://app.proactiveguard.io)

Copyright (c) 2025 Maya Plus. All rights reserved.
