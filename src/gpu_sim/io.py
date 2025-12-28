from __future__ import annotations

from typing import Any, Dict, List

import yaml

from gpu_sim.models import (
    ClusterConfig,
    PricingConfig,
    SimulationConfig,
    SchedulerConfig,
    JobStreamConfig,
)


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_config(cfg: Dict[str, Any]):
    cluster = ClusterConfig(**cfg["cluster"])
    pricing = PricingConfig(**cfg["pricing"])
    sim = SimulationConfig(**cfg["simulation"])
    sched = SchedulerConfig(**cfg["scheduler"])

    streams: List[JobStreamConfig] = []
    for s in cfg["workload"]["job_streams"]:
        streams.append(JobStreamConfig(**s))

    seed = int(cfg.get("seed", 0))
    return seed, cluster, pricing, sim, sched, streams
