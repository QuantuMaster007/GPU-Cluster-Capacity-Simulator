from __future__ import annotations

from typing import Optional, List, Tuple

from gpu_sim.models import ClusterState, Job


def sort_queue(queue: List[Job], inference_priority: bool) -> List[Job]:
    # Lower priority number = higher priority (1 beats 2)
    def key(j: Job) -> Tuple[int, int, int]:
        inference_boost = -1 if (inference_priority and j.job_type == "inference") else 0
        # pack big jobs first (FFD) to reduce fragmentation
        return (j.priority, inference_boost, -j.gpus_required)

    return sorted(queue, key=key)


def try_place_job(cluster: ClusterState, job: Job) -> Optional[int]:
    for node in cluster.nodes:
        if node.gpus_free >= job.gpus_required:
            return node.node_id
    return None


def allocate_jobs(cluster: ClusterState, queue: List[Job], inference_priority: bool) -> List[Job]:
    started: List[Job] = []
    ordered = sort_queue(queue, inference_priority=inference_priority)

    for job in ordered:
        node_id = try_place_job(cluster, job)
        if node_id is None:
            continue
        node = cluster.nodes[node_id]
        node.gpus_free -= job.gpus_required
        node.running_jobs.append(job.job_id)
        job.assigned_node = node_id
        started.append(job)

    return started
