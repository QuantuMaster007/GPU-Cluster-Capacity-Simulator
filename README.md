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

---

## Results & Insights

### GPU Utilization
![GPU Utilization](docs/images/utilization.png)

### Queue Depth / SLA Pressure
![Queue & SLA](docs/images/queue_sla.png)

---

## Results & Insights

### GPU Utilization
![GPU Utilization](docs/images/utilization.png)

### Queue Depth / SLA Pressure
![Queue & SLA](docs/images/queue_sla.png)

---

## Streamlit Dashboard

Run an interactive dashboard to compare scenarios and visualize utilization + queue depth.

```bash
pip install streamlit plotly
streamlit run app.py


```bash
git add README.md
git commit -m "Add Streamlit run instructions to README"
git push

![CI](https://github.com/QuantuMaster007/GPU-Cluster-Capacity-Simulator/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.11%2B-blue)

---

## Streamlit Dashboard

Run an interactive dashboard to compare scenarios and visualize utilization + queue depth.

```bash
pip install streamlit plotly
streamlit run app.py


Verify:
```bash
tail -n 30 README.md

---

## Streamlit Dashboard

Run an interactive dashboard to compare scenarios and visualize utilization + queue depth.

```bash
pip install streamlit plotly
streamlit run app.py
