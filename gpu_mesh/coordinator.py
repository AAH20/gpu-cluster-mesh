import time
import threading
from typing import Any, List, Dict, Optional
from dataclasses import dataclass, field

@dataclass
class ClusterNode:
    node_id: int
    hostname: str
    gpu_indices: List[int]
    is_healthy: bool = True
    last_heartbeat: float = field(default_factory=time.time)
    # Caller-supplied value passed to register_heartbeat(); this class does
    # not measure RDMA/network latency itself.
    heartbeat_latency_us: float = 2.5

@dataclass
class RingState:
    active_node_ids: List[int]
    topology_version: int
    quorum_size: int

class ElasticCommunicator:
    """
    In-memory cluster membership tracker. Tracks node health via
    application-level heartbeats and evicts a node once it misses its
    heartbeat deadline, bumping a topology version counter.

    This models the membership decision a real NCCL/RDMA ring rebuild would
    act on. It does not call NCCL, use RDMA verbs, or touch any GPU or
    network hardware — there is no PyTorch or CUDA dependency in this
    package at all. Wire scan_and_reconfigure()'s output into your own
    NCCL process-group rebuild logic if you need the real thing.
    """
    def __init__(self, initial_nodes: List[ClusterNode], heartbeat_timeout_s: float = 0.5):
        self.nodes: Dict[int, ClusterNode] = {n.node_id: n for n in initial_nodes}
        self.heartbeat_timeout_s = heartbeat_timeout_s
        self.topology_version = 1
        self._lock = threading.Lock()
        self.eviction_history: List[Dict[str, Any]] = []

    def get_active_ring(self) -> RingState:
        with self._lock:
            active_ids = [n.node_id for n in self.nodes.values() if n.is_healthy]
            return RingState(
                active_node_ids=sorted(active_ids),
                topology_version=self.topology_version,
                quorum_size=len(active_ids)
            )

    def register_heartbeat(self, node_id: int, latency_us: float = 2.5):
        with self._lock:
            if node_id in self.nodes:
                node = self.nodes[node_id]
                node.last_heartbeat = time.time()
                node.heartbeat_latency_us = latency_us
                node.is_healthy = True

    def scan_and_reconfigure(self) -> Optional[RingState]:
        """
        Evicts any node that has missed its heartbeat deadline and bumps the
        topology version. Returns the new RingState, or None if nothing
        changed. Callers are responsible for acting on the eviction (e.g.
        actually rebuilding an NCCL communicator) — this method only updates
        in-memory membership state.
        """
        now = time.time()
        evicted = []
        with self._lock:
            for node_id, node in self.nodes.items():
                if node.is_healthy and (now - node.last_heartbeat > self.heartbeat_timeout_s):
                    node.is_healthy = False
                    evicted.append(node_id)

            if evicted:
                self.topology_version += 1
                active_ids = [n.node_id for n in self.nodes.values() if n.is_healthy]
                self.eviction_history.append({
                    "evicted_nodes": evicted,
                    "new_topology_version": self.topology_version,
                    "timestamp": now
                })
                print(f"[gpu_mesh] Evicted nodes {evicted}. Topology bumped to v{self.topology_version} (active nodes: {len(active_ids)}).")
                return RingState(
                    active_node_ids=sorted(active_ids),
                    topology_version=self.topology_version,
                    quorum_size=len(active_ids)
                )
        return None
