# GPU Cluster Capacity Simulator

A lightweight simulator for **GPU cluster capacity planning** for AI workloads.  
Models **utilization, queueing delay, packing / fragmentation**, and a simplified **cost view**.

This project demonstrates **AI compute + infrastructure-level thinking**: capacity planning, scheduling tradeoffs, queueing/SLA behavior, and cost efficiency.

---

## Architecture
![Architecture](docs/images/architecture.svg)

---

## Quickstart

### 1) Create environment & install
```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
pip install -e .[dev]
