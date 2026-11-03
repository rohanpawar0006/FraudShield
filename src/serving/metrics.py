"""Operational telemetry and rolling latency statistics tracker for FraudShield."""

import threading
import time
from collections import deque
from typing import Deque

import numpy as np

from src.common.config import settings
from src.serving.schemas import MetricsResponse


class MetricsTracker:
    """Thread-safe rolling window metrics tracker for serving endpoints."""

    def __init__(self, window_size: int = settings.METRICS_WINDOW_SIZE) -> None:
        self._lock = threading.Lock()
        self.window_size = window_size
        self.request_count: int = 0
        self.error_count: int = 0
        self.feature_fallback_count: int = 0
        self.gnn_fallback_count: int = 0
        self.latencies: Deque[float] = deque(maxlen=window_size)
        self.start_time: float = time.time()

    def record_request(
        self,
        latency_ms: float,
        feature_fallback: bool,
        gnn_fallback: bool = False,
    ) -> None:
        """Record a successful scoring request and its observed latency."""
        with self._lock:
            self.request_count += 1
            if feature_fallback:
                self.feature_fallback_count += 1
            if gnn_fallback:
                self.gnn_fallback_count += 1
            self.latencies.append(latency_ms)

    def record_error(self) -> None:
        """Record an error encounter (validation, internal error, etc.)."""
        with self._lock:
            self.error_count += 1

    def get_metrics(self) -> MetricsResponse:
        """Calculate and return operational metrics including p50/p95 latencies and throughput."""
        with self._lock:
            req_count = self.request_count
            err_count = self.error_count
            fb_count = self.feature_fallback_count
            gnn_fb_count = self.gnn_fallback_count
            current_latencies = list(self.latencies)

        elapsed = max(time.time() - self.start_time, 1e-3)
        throughput = req_count / elapsed

        if current_latencies:
            p50 = float(np.percentile(current_latencies, 50))
            p95 = float(np.percentile(current_latencies, 95))
        else:
            p50 = 0.0
            p95 = 0.0

        return MetricsResponse(
            request_count=req_count,
            error_count=err_count,
            feature_fallback_count=fb_count,
            gnn_fallback_count=gnn_fb_count,
            throughput_rps=round(throughput, 2),
            p50_latency_ms=round(p50, 3),
            p95_latency_ms=round(p95, 3),
            window_size=len(current_latencies),
        )

    def reset(self) -> None:
        """Reset all counters and rolling windows (used in test isolation)."""
        with self._lock:
            self.request_count = 0
            self.error_count = 0
            self.feature_fallback_count = 0
            self.gnn_fallback_count = 0
            self.latencies.clear()
            self.start_time = time.time()


# Global telemetry tracker
metrics_tracker = MetricsTracker()

