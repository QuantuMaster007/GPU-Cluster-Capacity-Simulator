GPU Cluster Capacity Simulator
Overview
A lightweight simulator for GPU cluster capacity planning for AI workloads.

This project models utilization, queueing delay, packing/fragmentation effects, and a simplified cost view. It is designed to demonstrate infrastructure-level thinking rather than ML model optimization.
What This Simulator Covers
- GPU utilization and saturation behavior
- Queue depth and wait-time dynamics (P50 / P95)
- Inference SLA violations
- Cost efficiency and capacity tradeoffs

Architecture
The simulator models the interaction between workload arrivals, scheduling policies, and cluster topology to estimate real-world behavior.

Flow:
1. Job arrivals (training and inference)
2. Scheduler assigns GPUs (packing + fragmentation)
3. Simulator tracks time-series metrics
4. Reports utilization, queueing, SLA, and cost

Quickstart
1. Create and activate environment
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
pip install -e .[dev]

2. Run a simulation
gpu-sim run --config configs/base.yaml --out results/base_tick.csv --out-jobs results/base_jobs.csv

Scenarios Included
- Base: Balanced training and inference baseline
- Mixed Workloads: Fragmentation and contention effects
- Inference Spike: SLA pressure during bursts
- NVLink vs PCIe: Interconnect efficiency sensitivity

Results & Insights
Key observations from simulation results:

- High utilization does not guarantee low latency
- Queue depth grows rapidly under bursty inference
- SLA violations often appear before GPUs are fully saturated

Streamlit Dashboard
An interactive dashboard enables live simulation and visualization.

Run:
pip install -r requirements.txt
streamlit run app.py

Features:
- Scenario selection
- Live simulation execution
- KPI cards (utilization, P95 wait, SLA rate, cost)
- Time-series charts

Why This Project Exists
Modern AI infrastructure challenges are dominated by capacity planning, scheduling tradeoffs, and cost efficiency rather than raw GPU count alone.

This project highlights why queueing theory, workload mix, and scheduler design matter in real systems.
Roadmap
- Pool partitioning (training vs inference)
- Priority scheduling and preemption
- Trace-driven workloads
- Multi-cluster comparison
- Advanced cost modeling

License
MIT License
