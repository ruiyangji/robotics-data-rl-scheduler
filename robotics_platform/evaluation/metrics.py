"""
Metrics calculation for robotics collection scheduling evaluation.
"""

from typing import Dict, Any, List
import numpy as np


def compute_scenario_metrics(
    info: Dict[str, Any],
    policy_name: str,
    scenario_id: int,
    total_reward: float,
) -> Dict[str, Any]:
    return {
        "scenario_id": scenario_id,
        "policy": policy_name,
        "total_reward": round(total_reward, 2),
        "total_value_collected": round(info.get("total_value_collected", 0.0), 2),
        "completion_rate_pct": round(info.get("completion_rate_pct", 0.0), 2),
        "deadline_miss_rate_pct": round(info.get("deadline_miss_rate_pct", 0.0), 2),
        "error_coverage_pct": round(info.get("error_coverage_pct", 0.0), 2),
        "bandwidth_util_pct": round(info.get("bandwidth_util_pct", 0.0), 2),
        "makespan_sec": round(info.get("current_time", 0.0), 2),
    }
