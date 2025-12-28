from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Any, Tuple

import numpy as np
import pandas as pd

from gpu_sim.models import (
    ClusterConfig,
    PricingConfig,
    SimulationConfig,
    SchedulerConfig,
    JobStreamConfig,
    Job,
    NodeState,
    ClusterState,
)
from gpu_sim.scheduler import allocate_jobs
from gpu_sim.metrics import compute_tick_metrics, jobs_to_table, summarize_cost


@dataclass
class SimulationResult:
    tick_df: pd.DataFrame
    jobs_df: pd.DataFrame
    summary: Dict[str, Any]


def _init_cluster(cluster_cfg: ClusterConfig) -> ClusterState:
    nodes = [
        NodeState(node_id=i, gpus_total=cluster_cfg.gpus_per_node, gpus_free=cluster_cfg.gpus_per_node)
        for i in range(cluster_cfg.nodes)
    ]
    return ClusterState(cfg=cluster_cfg, nodes=nodes)


def _sample_jobs_for_minute(
    rng: np.random.Generator,
    minute: int,
    stream: JobStreamConfig,
    job_id_start: int,
) -> Tuple[List[Job], int]:
    lam_per_min = stream.arrivals_per_hour / 60.0
    arrivals = int(rng.poisson(lam=lam_per_min))

    jobs: List[Job] = []
    jid = job_id_start

    for _ in range(arrivals):
        gpus = int(rng.integers(stream.gpus_required_min, stream.gpus_required_max + 1))
        dur = int(rng.integers(stream.duration_minutes_min, stream.duration_minutes_max + 1))
        jobs.append(
            Job(
                job_id=jid,
                job_type=stream.job_type,
                gpus_required=gpus,
                duration_minutes=dur,
                arrival_minute=minute,
                priority=stream.priority,
                sla_wait_minutes=stream.sla_wait_minutes,
            )
        )
        jid += 1

    return jobs, jid


def run_simulation(
    seed: int,
    cluster_cfg: ClusterConfig,
    pricing: PricingConfig,
    sim_cfg: SimulationConfig,
    sched_cfg: SchedulerConfig,
    streams: List[JobStreamConfig],
) -> SimulationResult:
    rng = np.random.default_rng(seed)
    cluster = _init_cluster(cluster_cfg)

    queue: List[Job] = []
    running: Dict[int, Job] = {}
    completed: List[Job] = []

    tick_rows: List[Dict[str, Any]] = []

    next_job_id = 1
    duration = sim_cfg.duration_minutes
    step = sim_cfg.time_step_minutes

    for minute in range(0, duration, step):
        # complete finished jobs
        ended_ids = [jid for jid, job in running.items() if job.end_minute is not None and job.end_minute <= minute]
        for jid in ended_ids:
            job = running.pop(jid)
            node = cluster.nodes[job.assigned_node]  # type: ignore[index]
            node.gpus_free += job.gpus_required
            if jid in node.running_jobs:
                node.running_jobs.remove(jid)
            completed.append(job)

        # arrivals
        for stream in streams:
            new_jobs, next_job_id = _sample_jobs_for_minute(rng, minute, stream, next_job_id)
            queue.extend(new_jobs)

        # schedule
        started = allocate_jobs(cluster, queue, inference_priority=sched_cfg.enable_inference_priority)
        started_ids = {j.job_id for j in started}
        queue = [j for j in queue if j.job_id not in started_ids]

        for j in started:
            j.start_minute = minute
            j.end_minute = minute + j.duration_minutes
            running[j.job_id] = j

        # metrics
        tm = compute_tick_metrics(minute, cluster)
        tick_rows.append(
            {
                "minute": tm.minute,
                "total_gpus": tm.total_gpus,
                "busy_gpus": tm.busy_gpus,
                "utilization": tm.utilization,
                "queue_depth": len(queue),
                "running_jobs": len(running),
                "completed_jobs": len(completed),
            }
        )

    tick_df = pd.DataFrame(tick_rows)
    jobs_df = jobs_to_table(completed)

    cost = summarize_cost(
        tick_df=tick_df,
        pricing=pricing,
        efficiency_factor=cluster_cfg.efficiency_factor,
        time_step_minutes=step,
    )

    wait_p50 = float(jobs_df["wait_minutes"].dropna().median()) if not jobs_df.empty else 0.0
    wait_p95 = float(jobs_df["wait_minutes"].dropna().quantile(0.95)) if not jobs_df.empty else 0.0
    util_avg = float(tick_df["utilization"].mean()) if not tick_df.empty else 0.0

    sla_viol_rate = None
    if not jobs_df.empty:
        inf = jobs_df[jobs_df["job_type"] == "inference"]
        if len(inf) > 0 and "sla_violation" in inf.columns:
            sla_viol_rate = float(inf["sla_violation"].fillna(False).astype(bool).mean())

    summary: Dict[str, Any] = {
        "cluster_name": cluster_cfg.name,
        "gpu_type": cluster_cfg.gpu_type,
        "interconnect": cluster_cfg.interconnect,
        "total_gpus": cluster.total_gpus,
        "avg_utilization": util_avg,
        "jobs_completed": int(len(jobs_df)),
        "wait_p50_min": wait_p50,
        "wait_p95_min": wait_p95,
        "inference_sla_violation_rate": sla_viol_rate,
        **cost,
    }

    return SimulationResult(tick_df=tick_df, jobs_df=jobs_df, summary=summary)
