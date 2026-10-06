"""
Fleet Simulator.
Manages a virtual fleet of 8 to 32 robots emitting multi-rate telemetry
with fault injection (jitter, drops, clock skew, out of order).
"""

from typing import List, Dict, Any, Optional, Tuple
import uuid
import numpy as np

from robotics_platform.fleet.robot import VirtualRobot, RobotStatus
from robotics_platform.fleet.streams import StreamGenerator
from robotics_platform.fleet.faults import FaultInjector, NetworkFaultConfig
from robotics_platform.ingest.schema import TelemetryEvent


class FleetSimulator:
    def __init__(
        self,
        num_robots: int = 16,
        fault_config: Optional[NetworkFaultConfig] = None,
        seed: Optional[int] = 42,
    ):
        self.num_robots = num_robots
        self.fault_injector = FaultInjector(fault_config, seed=seed)
        self.rng = np.random.default_rng(seed)
        self.robots: List[VirtualRobot] = []
        self._init_fleet()

    def _init_fleet(self):
        self.robots = []
        for i in range(self.num_robots):
            robot_id = f"robot_{i:02d}"
            # Sample initial clock offset and drift
            offset, drift = self.fault_injector.sample_clock_parameters()
            robot = VirtualRobot(
                robot_id=robot_id,
                x=float(self.rng.uniform(-30.0, 30.0)),
                y=float(self.rng.uniform(-30.0, 30.0)),
                battery_pct=float(self.rng.uniform(70.0, 100.0)),
                status=RobotStatus.IDLE if i % 4 != 0 else RobotStatus.COLLECTING,
                clock_offset=offset,
                clock_drift_rate=drift,
            )
            self.robots.append(robot)

    def simulate_step(
        self,
        current_time: float,
        dt: float,
    ) -> List[TelemetryEvent]:
        """
        Simulates one time step of duration dt across all robots.
        Returns all generated telemetry events that arrived at the ingest service
        (excluding network dropped packets), sorted by ingest arrival timestamp.
        """
        arrived_events: List[Tuple[float, TelemetryEvent]] = []

        for robot in self.robots:
            robot.update(dt, self.rng)
            t_hw = robot.get_hardware_time(current_time)

            # 1. Joint / IMU stream (e.g. 50 Hz -> every 0.02s)
            # In a simulation step dt, sample with probability dt * 50
            if self.rng.uniform(0, 1) < min(1.0, dt * 50.0):
                payload = StreamGenerator.generate_joint_imu(robot, t_hw, self.rng)
                self._dispatch_event(
                    robot, "joint_imu", t_hw, current_time, payload, arrived_events
                )

            # 2. Camera metadata stream (15 Hz)
            if self.rng.uniform(0, 1) < min(1.0, dt * 15.0):
                payload = StreamGenerator.generate_camera_meta(robot, t_hw, self.rng)
                self._dispatch_event(
                    robot, "camera_meta", t_hw, current_time, payload, arrived_events
                )

            # 3. Status heartbeat (2 Hz)
            if self.rng.uniform(0, 1) < min(1.0, dt * 2.0):
                payload = StreamGenerator.generate_status_heartbeat(robot, t_hw, self.rng)
                self._dispatch_event(
                    robot, "status_heartbeat", t_hw, current_time, payload, arrived_events
                )

            # 4. Error events (sporadic)
            err_payload = StreamGenerator.generate_error_event(robot, t_hw, self.rng)
            if err_payload is not None:
                self._dispatch_event(
                    robot, "error_event", t_hw, current_time, err_payload, arrived_events
                )

        # Sort packets by arrival time at ingest server (simulates out-of-order jitter arrivals)
        arrived_events.sort(key=lambda x: x[0])
        return [evt for _, evt in arrived_events]

    def _dispatch_event(
        self,
        robot: VirtualRobot,
        stream_type: str,
        t_hw: float,
        ground_truth_time: float,
        payload: Dict[str, Any],
        out_list: List[Tuple[float, TelemetryEvent]],
    ):
        seq = robot.get_next_seq(stream_type)
        is_dropped, arrival_time = self.fault_injector.apply_network_transmission(ground_truth_time)

        if is_dropped:
            # Dropped in flight by simulated wireless network
            return

        event = TelemetryEvent(
            event_id=f"{robot.robot_id}_{stream_type}_{seq}_{uuid.uuid4().hex[:6]}",
            robot_id=robot.robot_id,
            stream_type=stream_type,
            event_timestamp=t_hw,
            ingest_timestamp=arrival_time,
            sequence_number=seq,
            payload=payload,
        )
        out_list.append((arrival_time, event))
