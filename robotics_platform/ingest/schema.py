"""
Pydantic Schemas for Robotics Telemetry Ingest.
"""

from typing import Dict, Any, Optional
from pydantic import BaseModel, Field


class TelemetryEvent(BaseModel):
    event_id: str
    robot_id: str
    stream_type: str
    event_timestamp: float           # Robot hardware timestamp
    ingest_timestamp: float          # Server receive monotonic timestamp
    sequence_number: int             # Monotonic sequence counter per (robot_id, stream_type)
    payload: Dict[str, Any] = Field(default_factory=dict)
    corrected_timestamp: Optional[float] = None  # Offset-corrected event timestamp
    status: str = "VALID"            # VALID, STALE_DROPPED, OUT_OF_ORDER_QUARANTINED
    drop_reason: Optional[str] = None

    model_config = {"arbitrary_types_allowed": True}
