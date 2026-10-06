"""
Fault injection engine for robotics telemetry:
- Network latency & jitter
- Out-of-order delivery
- Dropped packets
- Clock skew & drift
"""

from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np


@dataclass
class NetworkFaultConfig:
    min_latency_ms: float = 10.0      # Base network latency in ms
    jitter_std_ms: float = 25.0       # Jitter standard deviation
    drop_rate: float = 0.05           # 5% packet loss probability
    clock_skew_std_ms: float = 80.0   # Per-robot initial clock offset std in ms
    clock_drift_ppm: float = 30.0     # Clock drift in parts per million (microsec / sec)


class FaultInjector:
    def __init__(self, config: Optional[NetworkFaultConfig] = None, seed: Optional[int] = 42):
        self.config = config or NetworkFaultConfig()
        self.rng = np.random.default_rng(seed)

    def sample_clock_parameters(self) -> Tuple[float, float]:
        """Returns (initial_offset_seconds, drift_rate)."""
        offset_sec = self.rng.normal(0, self.config.clock_skew_std_ms / 1000.0)
        drift_rate = self.rng.normal(0, self.config.clock_drift_ppm * 1e-6)
        return float(offset_sec), float(drift_rate)

    def apply_network_transmission(
        self,
        event_ground_truth_time: float,
    ) -> Tuple[bool, float]:
        """
        Simulates packet transit over the network.
        Returns:
            is_dropped: bool (whether packet was dropped by network)
            arrival_time: float (ground truth timestamp when packet hits ingest server)
        """
        # Packet drop check
        if self.rng.uniform(0, 1) < self.config.drop_rate:
            return True, 0.0

        # Latency with log-normal or clipped normal jitter
        jitter = max(0.0, self.rng.normal(0, self.config.jitter_std_ms / 1000.0))
        latency = (self.config.min_latency_ms / 1000.0) + jitter
        arrival_time = event_ground_truth_time + latency

        return False, arrival_time
