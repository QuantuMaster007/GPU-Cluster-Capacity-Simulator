from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, List


@dataclass(frozen=True)
class ClusterConfig:
    name: str
    nodes: int
    gpus_per_node: int
    gpu_type: str
    interconnect: str  # "NVLINK" or "PCIE"
    efficiency_factor: float = 0.9


@dataclass(frozen=True)
class PricingConfig:
    gpu_hour_cost_usd: float
    overhead_multiplier: float = 1.0


@dataclass(frozen=True)
class SimulationConfig:
    duration_minutes: int
    time_step_minutes: int = 1


@dataclass(frozen=True)
class SchedulerConfig:
    policy: str = "FFD"
    enable_inference_priority: bool = True
    enable_preemption: bool = False


@dataclass(frozen=True)
class JobStreamConfig:
    name: str
    job_type: str  # "training" | "inference"
    arrivals_per_hour: float
    gpus_required_min: int
    gpus_required_max: int
    duration_minutes_min: int
    duration_minutes_max: int
    priority: int
    sla_wait_minutes: Optional[int] = None


@dataclass
class Job:
    job_id: int
    job_type: str
    gpus_required: int
    duration_minutes: int
    arrival_minute: int
    priority: int
    sla_wait_minutes: Optional[int] = None

    start_minute: Optional[int] = None
    end_minute: Optional[int] = None
    assigned_node: Optional[int] = None


@dataclass
class NodeState:
    node_id: int
    gpus_total: int
    gpus_free: int
    running_jobs: List[int] = field(default_factory=list)


@dataclass
class ClusterState:
    cfg: ClusterConfig
    nodes: List[NodeState]

    @property
    def total_gpus(self) -> int:
        return self.cfg.nodes * self.cfg.gpus_per_node

    @property
    def free_gpus(self) -> int:
        return sum(n.gpus_free for n in self.nodes)
