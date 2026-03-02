from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class PredictionResult:
    """
    Result returned by the ProactiveGuard API for a single node.

    Attributes
    ----------
    node_id:
        The node this prediction refers to.
    status:
        One of: ``healthy``, ``degraded_30s``, ``degraded_20s``, ``degraded_10s``,
        ``degraded_5s``, ``failed_crash``, ``failed_slow``, ``failed_byzantine``,
        ``failed_partition``.
    risk_score:
        Float in [0, 1] — 0 = fully healthy, 1 = certain failure.
    confidence:
        Model confidence in this prediction, float in [0, 1].
    time_to_failure:
        Estimated seconds until failure, or None if healthy.
    failure_type:
        High-level failure category (``"crash"``, ``"slow"``, ``"byzantine"``,
        ``"partition"``), or None if healthy/degraded.
    probabilities:
        Per-class probability dict, e.g. ``{"healthy": 0.92, "degraded_30s": 0.05, ...}``.
    timestamp:
        Unix timestamp when this result was generated.
    """

    node_id: str
    status: str
    risk_score: float
    confidence: float
    time_to_failure: Optional[float] = None
    failure_type: Optional[str] = None
    probabilities: Dict[str, float] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    # ── Convenience flags ──────────────────────────────────────────────────────

    @property
    def is_healthy(self) -> bool:
        return self.status == "healthy"

    @property
    def is_pre_failure(self) -> bool:
        return self.status.startswith("degraded_")

    @property
    def is_failed(self) -> bool:
        return self.status.startswith("failed_")

    def to_dict(self) -> dict:
        return {
            "node_id": self.node_id,
            "status": self.status,
            "risk_score": self.risk_score,
            "confidence": self.confidence,
            "time_to_failure": self.time_to_failure,
            "failure_type": self.failure_type,
            "probabilities": self.probabilities,
            "is_healthy": self.is_healthy,
            "is_pre_failure": self.is_pre_failure,
            "is_failed": self.is_failed,
            "timestamp": self.timestamp,
        }

    def __repr__(self) -> str:
        ttf = f", ttf={self.time_to_failure:.0f}s" if self.time_to_failure is not None else ""
        return (
            f"PredictionResult(node={self.node_id!r}, status={self.status!r}, "
            f"risk={self.risk_score:.2f}, conf={self.confidence:.2f}{ttf})"
        )
