import time
from typing import Dict, Any, List
from .coordinator import ElasticCommunicator

class ClusterHealthTelemetry:
    """
    Computes a health/membership summary dict from an ElasticCommunicator's
    current in-memory state. Returns a plain dict — it does not export,
    transmit, or attest anything; wire the returned dict into whatever
    metrics/monitoring pipeline you actually use.
    """
    def __init__(self, coordinator: ElasticCommunicator):
        self.coordinator = coordinator

    def export_telemetry_digest(self) -> Dict[str, Any]:
        ring = self.coordinator.get_active_ring()
        total_nodes = len(self.coordinator.nodes)
        healthy_nodes = ring.quorum_size

        cluster_goodput_efficiency = (healthy_nodes / max(total_nodes, 1)) * 100.0
        avg_heartbeat_latency = sum(n.heartbeat_latency_us for n in self.coordinator.nodes.values()) / max(total_nodes, 1)

        return {
            "cluster_topology_version": ring.topology_version,
            "total_cluster_nodes": total_nodes,
            "active_quorum_nodes": healthy_nodes,
            "training_goodput_efficiency_pct": round(cluster_goodput_efficiency, 2),
            "average_heartbeat_latency_us": round(avg_heartbeat_latency, 2),
            "total_hot_reconfigs": len(self.coordinator.eviction_history),
        }
