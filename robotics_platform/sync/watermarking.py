"""
Stream Watermarking Tracker.

Tracks the latest timestamp that is safe to treat as complete for each stream.
Downstream consumers can read up to the watermark with high confidence that
earlier packets will no longer arrive.
"""

from typing import Dict, Tuple, Optional


class WatermarkTracker:
    def __init__(self, default_slack_seconds: float = 0.15):
        self.default_slack_seconds = default_slack_seconds
        # Stream-specific slack allowances (e.g. video frames have larger latency allowance)
        self.stream_slack = {
            "joint_imu": 0.08,        # 80ms slack for fast control loop
            "camera_meta": 0.25,      # 250ms slack for vision processing
            "status_heartbeat": 0.50, # 500ms slack for telemetry
            "error_event": 0.0,
        }
        # Highest observed corrected timestamp per (robot_id, stream_type)
        self.max_observed: Dict[Tuple[str, str], float] = {}

    def update(self, robot_id: str, stream_type: str, corrected_timestamp: float) -> float:
        """
        Updates the highest observed timestamp and returns the updated watermark.
        """
        key = (robot_id, stream_type)
        prev = self.max_observed.get(key, 0.0)
        self.max_observed[key] = max(prev, corrected_timestamp)
        return self.get_watermark(robot_id, stream_type)

    def get_watermark(self, robot_id: str, stream_type: str) -> float:
        """Returns the current safe-to-read watermark timestamp."""
        key = (robot_id, stream_type)
        max_t = self.max_observed.get(key, 0.0)
        slack = self.stream_slack.get(stream_type, self.default_slack_seconds)
        return max(0.0, max_t - slack)

    def is_past_watermark(self, robot_id: str, stream_type: str, timestamp: float) -> bool:
        """Checks if a packet's timestamp is older than the current safe watermark."""
        wm = self.get_watermark(robot_id, stream_type)
        return timestamp < wm
