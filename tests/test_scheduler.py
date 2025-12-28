from gpu_sim.models import ClusterConfig, NodeState, ClusterState, Job
from gpu_sim.scheduler import allocate_jobs


def test_allocate_jobs_first_fit():
    cluster_cfg = ClusterConfig(
        name="t",
        nodes=2,
        gpus_per_node=4,
        gpu_type="X",
        interconnect="PCIE",
        efficiency_factor=0.9,
    )
    cluster = ClusterState(
        cfg=cluster_cfg,
        nodes=[NodeState(0, 4, 4), NodeState(1, 4, 4)],
    )

    queue = [
        Job(job_id=1, job_type="training", gpus_required=4, duration_minutes=10, arrival_minute=0, priority=2),
        Job(job_id=2, job_type="training", gpus_required=2, duration_minutes=10, arrival_minute=0, priority=2),
    ]

    started = allocate_jobs(cluster=cluster, queue=queue, inference_priority=False)

    assert len(started) == 2
    assert cluster.free_gpus == 2  # 8 total - (4+2) = 2
