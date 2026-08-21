import time
import threading
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

@dataclass
class SnapshotFrame:
    step_id: int
    optimizer_state_hash: str
    gradient_norm: float
    timestamp: float
    tier: str # "L1_HOST_RAM", "L2_LOCAL_NVME", "L3_CLOUD_OBJECT_STORE"

class MultiTierGradientWAL:
    """
    In-memory two-tier ring buffer for gradient/optimizer-state checkpoint
    metadata. Recent steps live in a dict (tagged "L1_HOST_RAM"); once the
    buffer exceeds max_in_memory_frames, the oldest step is moved to a list
    (tagged "L2_LOCAL_NVME").

    Both tiers are ordinary Python objects in the same process's memory —
    nothing here writes to real host RAM as a separate resource, NVMe, or
    cloud storage. The tier labels describe the intended real-storage target
    of a caller-provided writer; this class only tracks which step is where
    and evicts in FIFO order.
    """
    def __init__(self, max_in_memory_frames: int = 5):
        self.max_in_memory_frames = max_in_memory_frames
        self.l1_host_ram: Dict[int, SnapshotFrame] = {}
        self.l2_nvme_journal: List[SnapshotFrame] = []
        self._lock = threading.Lock()

    def record_step_gradient(self, step_id: int, state_hash: str, grad_norm: float) -> SnapshotFrame:
        now = time.time()
        frame = SnapshotFrame(
            step_id=step_id,
            optimizer_state_hash=state_hash,
            gradient_norm=grad_norm,
            timestamp=now,
            tier="L1_HOST_RAM"
        )

        with self._lock:
            self.l1_host_ram[step_id] = frame
            # Maintain bounded RAM ring buffer
            if len(self.l1_host_ram) > self.max_in_memory_frames:
                oldest_step = min(self.l1_host_ram.keys())
                evicted = self.l1_host_ram.pop(oldest_step)
                evicted.tier = "L2_LOCAL_NVME"
                self.l2_nvme_journal.append(evicted)

        return frame

    def get_latest_recovery_frame(self) -> Optional[SnapshotFrame]:
        with self._lock:
            if self.l1_host_ram:
                latest_step = max(self.l1_host_ram.keys())
                return self.l1_host_ram[latest_step]
            if self.l2_nvme_journal:
                return self.l2_nvme_journal[-1]
            return None
