GPU Cluster Capacity Simulator

A lightweight simulator for GPU cluster capacity planning for AI workloads.
Models utilization, queueing delay, packing/fragmentation, and a simplified cost view.

This project demonstrates AI infrastructure–level thinking: capacity planning, scheduling tradeoffs, queueing/SLA behavior, and cost efficiency.

ARCHITECTURE
The simulator models how job arrivals, scheduling policies, and cluster topology interact to determine:
- GPU utilization
- Queue depth & wait time
- Inference SLA violations
- Cost efficiency

QUICKSTART
1) Create environment & install
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
pip install -e ".[dev]"

2) Run a simulation
gpu-sim run --config configs/base.yaml --out results/base_tick.csv --out-jobs results/base_jobs.csv

SCENARIOS INCLUDED
- base.yaml – Balanced training + inference baseline
- mixed_workloads.yaml – Training + inference contention
- inference_spike.yaml – SLA pressure during bursts
- nvl_vs_pcie.yaml – Interconnect sensitivity

RESULTS & INSIGHTS
GPU Utilization:
High average utilization does not guarantee good latency or SLA behavior.

Queue Depth / SLA Pressure:
Inference SLA violations appear before GPUs look fully saturated.

STREAMLIT DASHBOARD
Run:
pip install -r requirements.txt
streamlit run app.py

Allows scenario comparison, live simulation, KPI visualization.

DEVELOPMENT
pytest -q
ruff check .

WHY THIS PROJECT
Shows how AI compute systems behave in practice, focusing on infra tradeoffs rather than ML optimization.

ROADMAP
- Pool partitioning
- Preemption
- Workload traces
- Multi-cluster comparison
- Cost optimization

LICENSE
MIT License

