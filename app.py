import json
import subprocess
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

RESULTS = Path("results")
CONFIGS = Path("configs")


def run_sim(config_path: Path):
    RESULTS.mkdir(parents=True, exist_ok=True)
    stem = config_path.stem
    tick = RESULTS / f"{stem}_tick.csv"
    jobs = RESULTS / f"{stem}_jobs.csv"

    cmd = [
        "gpu-sim",
        "run",
        "--config", str(config_path),
        "--out", str(tick),
        "--out-jobs", str(jobs),
    ]
    out = subprocess.check_output(cmd, text=True)
    return tick, jobs, out


def load_summary_from_stdout(stdout: str):
    marker = "=== Simulation Summary ==="
    if marker not in stdout:
        return {}
    tail = stdout.split(marker, 1)[1].strip()
    start = tail.find("{")
    end = tail.rfind("}")
    if start == -1 or end == -1:
        return {}
    return json.loads(tail[start : end + 1])


st.set_page_config(page_title="GPU Cluster Capacity Simulator", layout="wide")
st.title("GPU Cluster Capacity Simulator")
st.caption("Scenario-driven capacity planning: utilization, queueing/SLA pressure, and cost.")

scenario_files = sorted(CONFIGS.glob("*.yaml"))
scenario_map = {p.stem: p for p in scenario_files}

colA, colB = st.columns([2, 1])
with colA:
    scenario = st.selectbox("Scenario", options=list(scenario_map.keys()), index=0)
with colB:
    run_btn = st.button("Run simulation")

config_path = scenario_map[scenario]
tick_path = RESULTS / f"{scenario}_tick.csv"
jobs_path = RESULTS / f"{scenario}_jobs.csv"

summary = {}
if run_btn or (not tick_path.exists()) or (not jobs_path.exists()):
    with st.spinner("Running simulation..."):
        tick_path, jobs_path, stdout = run_sim(config_path)
        summary = load_summary_from_stdout(stdout)

tick_df = pd.read_csv(tick_path)
jobs_df = pd.read_csv(jobs_path)

avg_util = float(tick_df["utilization"].mean())
jobs_completed = int(len(jobs_df))
wait_p95 = float(jobs_df["wait_minutes"].dropna().quantile(0.95)) if jobs_completed else 0.0

inf = jobs_df[jobs_df["job_type"] == "inference"] if "job_type" in jobs_df.columns else pd.DataFrame()
sla_rate = None
if len(inf) > 0 and "sla_violation" in inf.columns:
    sla_rate = float(inf["sla_violation"].fillna(False).astype(bool).mean())

total_cost = summary.get("total_cost_usd", None)

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Avg Utilization", f"{avg_util:.2%}")
k2.metric("Jobs Completed", f"{jobs_completed}")
k3.metric("P95 Wait (min)", f"{wait_p95:.1f}")
k4.metric("Inference SLA Violation", "n/a" if sla_rate is None else f"{sla_rate:.2%}")
k5.metric("Total Cost (USD)", "n/a" if total_cost is None else f"${total_cost:,.0f}")

st.divider()

left, right = st.columns(2)
with left:
    fig1 = px.line(tick_df, x="minute", y="utilization", title="Utilization over time")
    st.plotly_chart(fig1, use_container_width=True)

with right:
    fig2 = px.line(tick_df, x="minute", y="queue_depth", title="Queue depth over time")
    st.plotly_chart(fig2, use_container_width=True)

with st.expander("Show data"):
    st.subheader("Tick metrics")
    st.dataframe(tick_df, use_container_width=True, height=260)
    st.subheader("Jobs")
    st.dataframe(jobs_df, use_container_width=True, height=260)
