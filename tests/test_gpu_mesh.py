import time
import unittest
from gpu_mesh.coordinator import ElasticCommunicator, ClusterNode, RingState
from gpu_mesh.tier_snapshot import MultiTierGradientWAL
from gpu_mesh.telemetry import ClusterHealthTelemetry

class TestGPUClusterMesh(unittest.TestCase):
    def setUp(self):
        # 16 nodes * 8 GPUs = 128 GPU cluster setup
        self.initial_nodes = [
            ClusterNode(node_id=i, hostname=f"gpu-node-{i:02d}", gpu_indices=list(range(8)))
            for i in range(16)
        ]
        self.communicator = ElasticCommunicator(self.initial_nodes, heartbeat_timeout_s=0.1)

    def test_initial_ring_quorum(self):
        ring = self.communicator.get_active_ring()
        self.assertEqual(ring.quorum_size, 16)
        self.assertEqual(ring.topology_version, 1)

    def test_dynamic_partition_eviction_and_zero_restart_hot_swap(self):
        time.sleep(0.12) # Let initial heartbeats expire
        
        # All nodes heartbeat EXCEPT node 7 (simulating InfiniBand switch port drop)
        for i in range(16):
            if i != 7:
                self.communicator.register_heartbeat(i, latency_us=2.1)

        new_ring = self.communicator.scan_and_reconfigure()

        # Invariant: Node 7 evicted, ring hot-rebuilt to 15 nodes, version bumped to 2
        self.assertIsNotNone(new_ring)
        self.assertEqual(new_ring.quorum_size, 15)
        self.assertEqual(new_ring.topology_version, 2)
        self.assertNotIn(7, new_ring.active_node_ids)

    def test_multi_tier_gradient_snapshot_recovery(self):
        wal = MultiTierGradientWAL(max_in_memory_frames=3)
        
        # Record 5 training step gradient checkpoints
        for step in range(1, 6):
            wal.record_step_gradient(
                step_id=step,
                state_hash=f"hash_step_{step}",
                grad_norm=0.45 + (step * 0.01)
            )

        # Invariant: Latest recovery frame is step 5 from L1 Host RAM in sub-15ms
        latest = wal.get_latest_recovery_frame()
        self.assertIsNotNone(latest)
        self.assertEqual(latest.step_id, 5)
        self.assertEqual(latest.tier, "L1_HOST_RAM")

    def test_telemetry_goodput_export(self):
        telemetry = ClusterHealthTelemetry(self.communicator)
        digest = telemetry.export_telemetry_digest()

        self.assertEqual(digest["total_cluster_nodes"], 16)
        self.assertEqual(digest["training_goodput_efficiency_pct"], 100.0)
        self.assertIn("average_heartbeat_latency_us", digest)

    def test_telemetry_reflects_eviction(self):
        time.sleep(0.12)
        for i in range(16):
            if i != 3:
                self.communicator.register_heartbeat(i, latency_us=2.1)
        self.communicator.scan_and_reconfigure()

        telemetry = ClusterHealthTelemetry(self.communicator)
        digest = telemetry.export_telemetry_digest()

        self.assertEqual(digest["active_quorum_nodes"], 15)
        self.assertEqual(digest["total_hot_reconfigs"], 1)
        self.assertLess(digest["training_goodput_efficiency_pct"], 100.0)

if __name__ == "__main__":
    unittest.main()
