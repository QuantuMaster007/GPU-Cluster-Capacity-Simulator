from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Any

import pandas as pd

from gpu_sim.models import ClusterState, PricingConfig, Job


@dataclass
class TickMetrics:
    minute: int
    total_gpus: int
    busy_gpus: int
    utilization: float


def compute_tick_metrics(minute: int, cluster: ClusterState) -> TickMetrics:
    total = cluster.total_gpus
    busy = total - cluster.free_gpus
    util = busy / total if total > 0 else 0.0
    return TickMetrics(minute=minute, total_gpus=total, busy_gpus=busy, utilization=util)


def jobs_to_table(jobs: List[Job]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for j in jobs:
        wait = None
        if j.start_minute is not None:
            wait = j.start_minute - j.arrival_minute

        sla_violation = None
        if j.job_type == "inference" and j.sla_wait_minutes is not None and wait is not None:
            sla_violation = wait > j.sla_wait_minutes

        rows.append(
            {
                "job_id": j.job_id,
                "job_type": j.job_type,
                "priority": j.priority,
                "gpus_required": j.gpus_required,
                "arrival_minute": j.arrival_minute,
                "start_minute": j.start_minute,
                "end_minute": j.end_minute,
                "wait_minutes": wait,
                "assigned_node": j.assigned_node,
                "sla_wait_minutes": j.sla_wait_minutes,
                "sla_violation": sla_violation,
            }
        )
    return pd.DataFrame(rows)


def summarize_cost(
    tick_df: pd.DataFrame,
    pricing: PricingConfig,
    efficiency_factor: float,
    time_step_minutes: int,
) -> Dict[str, float]:
    busy_gpus = tick_df["busy_gpus"].sum()
    busy_gpu_minutes = busy_gpus * time_step_minutes
    busy_gpu_hours = busy_gpu_minutes / 60.0

    raw = busy_gpu_hours * pricing.gpu_hour_cost_usd
    total = (raw * pricing.overhead_multiplier) / max(efficiency_factor, 1e-9)

    return {
        "busy_gpu_hours": float(busy_gpu_hours),
        "raw_gpu_cost_usd": float(raw),
        "total_cost_usd": float(total),
    }
