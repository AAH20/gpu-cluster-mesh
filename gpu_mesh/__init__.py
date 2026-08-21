"""
GPU-Cluster-Mesh: an in-memory, pure-Python simulation of heartbeat-based
cluster membership tracking and tiered gradient-checkpoint bookkeeping for
distributed training coordination logic.

This package does not use NCCL, RDMA, CUDA, or PyTorch, and has no GPU or
network dependency. It models the membership/eviction/versioning decisions a
real elastic training coordinator needs to make, so that logic can be built
and tested independently of real GPU hardware. It is not itself a
replacement for NCCL fault tolerance, RDMA link management, or a checkpoint
storage system.
"""

from .coordinator import ElasticCommunicator, ClusterNode, RingState
from .tier_snapshot import MultiTierGradientWAL, SnapshotFrame
from .telemetry import ClusterHealthTelemetry

__version__ = "0.2.0"
__all__ = [
    "ElasticCommunicator",
    "ClusterNode",
    "RingState",
    "MultiTierGradientWAL",
    "SnapshotFrame",
    "ClusterHealthTelemetry",
]
