"""
Stale Packet Policy and Sequence Verification.

Applies domain-specific freshness deadlines:
- High-rate joint/IMU: stale if older than 500 ms (real-time control limit)
- Camera metadata: stale if older than 2000 ms
- Heartbeat: stale if older than 5000 ms
- Error events: never dropped for staleness
"""

from typing import Dict, Tuple, Optional
from robotics_platform.ingest.schema import TelemetryEvent


class StalePacketPolicy:
    def __init__(self, thresholds: Optional[Dict[str, float]] = None):
        # Maximum allowed age (in seconds) between event corrected time and ingest time
        self.thresholds = thresholds or {
            "joint_imu": 0.500,        # 500ms
            "camera_meta": 2.000,      # 2.0s
            "status_heartbeat": 5.000, # 5.0s
            "error_event": float("inf"), # Never drop critical error telemetry
        }
        # Last received sequence number per (robot_id, stream_type)
        self.last_seq: Dict[Tuple[str, str], int] = {}

    def evaluate_packet(
        self,
        event: TelemetryEvent,
        watermark: float,
    ) -> Tuple[str, Optional[str]]:
        """
        Evaluates event freshness and sequence order.
        Returns (status, drop_reason).
        status: VALID, STALE_DROPPED, OUT_OF_ORDER_QUARANTINED
        """
        threshold = self.thresholds.get(event.stream_type, 1.0)
        timestamp = event.corrected_timestamp or event.event_timestamp
        age = event.ingest_timestamp - timestamp

        # Check staleness threshold
        if age > threshold:
            reason = f"Stale {event.stream_type}: age {age*1000:.1f}ms exceeds {threshold*1000:.1f}ms threshold"
            return "STALE_DROPPED", reason

        # Sequence ordering check
        key = (event.robot_id, event.stream_type)
        prev_seq = self.last_seq.get(key, -1)

        if event.sequence_number <= prev_seq:
            # Out of order or duplicate packet arriving after newer sequence number
            reason = f"Out-of-order/duplicate seq {event.sequence_number} <= previous {prev_seq}"
            self.last_seq[key] = max(prev_seq, event.sequence_number)
            return "OUT_OF_ORDER_QUARANTINED", reason

        # Valid in-order packet
        self.last_seq[key] = event.sequence_number
        return "VALID", None
