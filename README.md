# GPU-Cluster-Mesh (`gpu-cluster-mesh`)

**An in-memory simulation of heartbeat-based cluster membership and tiered gradient-checkpoint bookkeeping for distributed training coordination logic.**

[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

---

## What this actually is

Large-scale distributed training clusters need to decide, quickly, which
nodes are still healthy and rebuild their communication topology around the
survivors when one drops. `gpu-cluster-mesh` implements that *decision
logic* — heartbeat tracking, timeout-based eviction, topology versioning,
and a tiered checkpoint eviction buffer — as plain, testable, in-memory
Python.

- `ElasticCommunicator` — tracks per-node heartbeats; `scan_and_reconfigure()`
  evicts any node that misses its deadline and bumps a topology version
  counter.
- `MultiTierGradientWAL` — a bounded in-memory ring buffer for checkpoint
  metadata. Recent steps are tagged `"L1_HOST_RAM"`; once the buffer is
  full, the oldest step is moved to a list tagged `"L2_LOCAL_NVME"`.
- `ClusterHealthTelemetry` — computes a summary dict (topology version,
  quorum size, goodput %) from an `ElasticCommunicator`'s current state.

## What this is not

**This package has no NCCL, RDMA, CUDA, or PyTorch dependency, and none of
its code talks to real GPU or network hardware.** It is a model of the
membership/eviction/versioning decisions a real elastic-training
coordinator has to make, so that decision logic can be built and unit
tested independently of a real cluster. Specifically:

- `ElasticCommunicator` does not create, rebuild, or touch an NCCL
  communicator or process group. It decides *which nodes should be in the
  ring* based on heartbeats you feed it; wiring that decision into an
  actual `torch.distributed`/NCCL rebuild is left to the caller.
- `heartbeat_latency_us` on `ClusterNode` is a value you pass in via
  `register_heartbeat()`, not a measured RDMA/InfiniBand link latency.
- The `"L1_HOST_RAM"` / `"L2_LOCAL_NVME"` tier labels describe intended
  targets for a caller-provided writer. Nothing here performs actual RAM,
  NVMe, or cloud-storage I/O — both tiers are ordinary Python objects in
  the same process.
- No dollar-cost or outage-frequency figures are claimed anywhere in this
  README, because nothing in this package measures them.

If you're building real elastic-training fault tolerance, use this as the
membership state machine underneath your own NCCL/RDMA integration — not as
a replacement for it.

## Install

Not published to PyPI. Install from source:

```bash
git clone https://github.com/AAH20/gpu-cluster-mesh.git
cd gpu-cluster-mesh
pip install -e .
```

## Usage

```python
from gpu_mesh import ElasticCommunicator, ClusterNode, MultiTierGradientWAL

nodes = [ClusterNode(node_id=i, hostname=f"node-{i:02d}", gpu_indices=list(range(8))) for i in range(16)]
communicator = ElasticCommunicator(nodes, heartbeat_timeout_s=0.5)

# Nodes report in; anything that doesn't gets evicted on the next scan
communicator.register_heartbeat(node_id=0, latency_us=2.1)

new_ring = communicator.scan_and_reconfigure()
if new_ring:
    print(f"Active nodes: {new_ring.quorum_size} (topology v{new_ring.topology_version})")
    # Wire new_ring.active_node_ids into your own NCCL rebuild here.

wal = MultiTierGradientWAL(max_in_memory_frames=5)
wal.record_step_gradient(step_id=1024, state_hash="sha256_hash", grad_norm=0.48)
```

## Tests

```bash
python -m unittest discover tests -v
```

5/5 pass locally on Python 3.10–3.12 (`.github/workflows/ci.yml` runs the
same command on push once enabled on GitHub).

## Architecture

```
gpu-cluster-mesh/
├── gpu_mesh/
│   ├── __init__.py            # exports
│   ├── coordinator.py         # ElasticCommunicator, ClusterNode, RingState
│   ├── tier_snapshot.py       # MultiTierGradientWAL, SnapshotFrame
│   └── telemetry.py           # ClusterHealthTelemetry
└── tests/
    └── test_gpu_mesh.py
```

## License

Apache-2.0

## Author

**Ahmed Hassan**
* LinkedIn: [Ahmed Hassan](https://eg.linkedin.com/in/ahmed-hassan-f11)
* Platform: [A2Z SOC](https://a2zsoc.com)
