"""
Standard Cluster Scheduling Heuristics:
- FIFO (First-In, First-Out First-Fit)
- SJF (Shortest Job First)
- Best-Fit / Tetris (Packing heuristic)
- DRF (Dominant Resource Fairness)
- Random (Uniform over valid masked actions)
"""

from typing import Dict, Any, Optional
import numpy as np
from cluster_scheduler.env.cluster_env import ClusterEnv


class BaseScheduler:
    def __init__(self, name: str):
        self.name = name

    def pick_action(self, env: ClusterEnv, obs: Dict[str, np.ndarray]) -> int:
        raise NotImplementedError


class FIFOScheduler(BaseScheduler):
    """
    First-In First-Out: Places candidate job on the lowest-indexed node that can fit it.
    If no node fits, chooses DEFER.
    """

    def __init__(self):
        super().__init__("FIFO")

    def pick_action(self, env: ClusterEnv, obs: Dict[str, np.ndarray]) -> int:
        mask = obs["action_mask"]
        num_nodes = env.num_nodes
        # Check nodes in order 0..num_nodes-1
        for i in range(num_nodes):
            if mask[i] == 1:
                return i
        return num_nodes  # DEFER


class SJFScheduler(BaseScheduler):
    """
    Shortest Job First heuristic:
    Prefers placement on nodes that maximize packing for short jobs.
    """

    def __init__(self):
        super().__init__("SJF")

    def pick_action(self, env: ClusterEnv, obs: Dict[str, np.ndarray]) -> int:
        mask = obs["action_mask"]
        num_nodes = env.num_nodes
        job = env.candidate_job
        if job is None:
            return num_nodes

        # Pick node with least currently queued/running jobs that fits
        best_node = None
        min_load = float("inf")
        for i in range(num_nodes):
            if mask[i] == 1:
                load = len(env.nodes[i].running_jobs)
                if load < min_load:
                    min_load = load
                    best_node = i

        return best_node if best_node is not None else num_nodes


class BestFitScheduler(BaseScheduler):
    """
    Best-Fit / Tetris heuristic:
    Selects the valid node that leaves the smallest residual capacity (closest match)
    to minimize multi-dimensional resource fragmentation.
    """

    def __init__(self):
        super().__init__("BestFit")

    def pick_action(self, env: ClusterEnv, obs: Dict[str, np.ndarray]) -> int:
        mask = obs["action_mask"]
        num_nodes = env.num_nodes
        job = env.candidate_job
        if job is None:
            return num_nodes

        best_node = None
        min_residual = float("inf")

        for i in range(num_nodes):
            if mask[i] == 1:
                node = env.nodes[i]
                rem_cpu = node.cpu_free - job.cpu_req
                rem_mem = node.mem_free - job.mem_req
                residual = (rem_cpu / node.cpu_capacity) ** 2 + (rem_mem / node.mem_capacity) ** 2
                if residual < min_residual:
                    min_residual = residual
                    best_node = i

        return best_node if best_node is not None else num_nodes


class DRFScheduler(BaseScheduler):
    """
    Dominant Resource Fairness:
    Selects the node that achieves the most balanced dominant resource utilization.
    """

    def __init__(self):
        super().__init__("DRF")

    def pick_action(self, env: ClusterEnv, obs: Dict[str, np.ndarray]) -> int:
        mask = obs["action_mask"]
        num_nodes = env.num_nodes
        job = env.candidate_job
        if job is None:
            return num_nodes

        best_node = None
        min_dom_share = float("inf")

        for i in range(num_nodes):
            if mask[i] == 1:
                node = env.nodes[i]
                dom_share = max(node.cpu_utilization, node.mem_utilization)
                if dom_share < min_dom_share:
                    min_dom_share = dom_share
                    best_node = i

        return best_node if best_node is not None else num_nodes


class RandomScheduler(BaseScheduler):
    """
    Random scheduler:
    Uniformly samples among valid actions in the action mask.
    """

    def __init__(self, seed: Optional[int] = None):
        super().__init__("Random")
        self.rng = np.random.default_rng(seed)

    def pick_action(self, env: ClusterEnv, obs: Dict[str, np.ndarray]) -> int:
        mask = obs["action_mask"]
        valid_actions = np.where(mask == 1)[0]
        if len(valid_actions) == 0:
            return env.num_nodes
        return int(self.rng.choice(valid_actions))
