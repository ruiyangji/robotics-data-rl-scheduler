"""
Telemetry Ingestion Pipeline.
Coordinates clock offset synchronization, watermarking, stale packet policy,
quality metric recording, and persistent storage.
"""

from typing import List, Dict, Any, Optional
from robotics_platform.ingest.schema import TelemetryEvent
from robotics_platform.ingest.storage import TelemetryStorage
from robotics_platform.sync.clock_sync import ClockSyncTracker
from robotics_platform.sync.watermarking import WatermarkTracker
from robotics_platform.sync.staleness import StalePacketPolicy
from robotics_platform.sync.quality_metrics import DataQualityTracker


class IngestionPipeline:
    def __init__(
        self,
        storage: Optional[TelemetryStorage] = None,
        stale_thresholds: Optional[Dict[str, float]] = None,
    ):
        self.storage = storage or TelemetryStorage(":memory:")
        self.clock_sync = ClockSyncTracker()
        self.watermark_tracker = WatermarkTracker()
        self.stale_policy = StalePacketPolicy(thresholds=stale_thresholds)
        self.quality_tracker = DataQualityTracker()

    def process_event(self, event: TelemetryEvent) -> TelemetryEvent:
        """Processes a single telemetry event through synchronization, watermarking, and quality policy."""
        # 1. Correct hardware clock offset to server-synchronized time
        t_corr = self.clock_sync.record_packet(
            event.robot_id, event.event_timestamp, event.ingest_timestamp
        )
        event.corrected_timestamp = t_corr

        # 2. Update stream watermark
        wm = self.watermark_tracker.update(event.robot_id, event.stream_type, t_corr)

        # 3. Evaluate freshness and sequence order
        status, drop_reason = self.stale_policy.evaluate_packet(event, wm)
        event.status = status
        event.drop_reason = drop_reason

        # 4. Record data quality metrics
        latency_ms = max(0.0, (event.ingest_timestamp - t_corr) * 1000.0)
        self.quality_tracker.record_event(event.stream_type, status, latency_ms)

        return event

    def process_batch(self, events: List[TelemetryEvent]) -> List[TelemetryEvent]:
        processed = [self.process_event(e) for e in events]
        self.storage.insert_batch(processed)
        return processed

    def get_metrics_summary(self) -> Dict[str, Any]:
        return self.quality_tracker.get_summary()
