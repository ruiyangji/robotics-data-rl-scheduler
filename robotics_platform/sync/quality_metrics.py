"""
Data Quality Metrics Tracker.
Calculates drop rate, late rate, p50/p95 ingest-to-query latency, and per-stream statistics.
"""

from typing import Dict, List, Any
from collections import defaultdict
import numpy as np


class DataQualityTracker:
    def __init__(self):
        self.total_received = 0
        self.status_counts = defaultdict(int)
        self.stream_counts = defaultdict(lambda: defaultdict(int))
        self.latencies_ms: List[float] = []

    def record_event(
        self,
        stream_type: str,
        status: str,
        latency_ms: float,
    ):
        self.total_received += 1
        self.status_counts[status] += 1
        self.stream_counts[stream_type][status] += 1
        if status == "VALID":
            self.latencies_ms.append(latency_ms)

    def get_summary(self) -> Dict[str, Any]:
        total = max(1, self.total_received)
        valid = self.status_counts["VALID"]
        stale = self.status_counts["STALE_DROPPED"]
        quarantined = self.status_counts["OUT_OF_ORDER_QUARANTINED"]

        lat_arr = np.array(self.latencies_ms) if self.latencies_ms else np.array([0.0])
        p50 = float(np.percentile(lat_arr, 50))
        p95 = float(np.percentile(lat_arr, 95))
        p99 = float(np.percentile(lat_arr, 99))

        return {
            "total_received": self.total_received,
            "valid_count": valid,
            "stale_dropped_count": stale,
            "quarantined_count": quarantined,
            "drop_rate_pct": (stale / total) * 100.0,
            "quarantine_rate_pct": (quarantined / total) * 100.0,
            "success_rate_pct": (valid / total) * 100.0,
            "latency_p50_ms": p50,
            "latency_p95_ms": p95,
            "latency_p99_ms": p99,
            "stream_breakdown": {
                stream: dict(counts) for stream, counts in self.stream_counts.items()
            },
        }
