import pandas as pd

from gpu_sim.models import PricingConfig
from gpu_sim.metrics import summarize_cost


def test_summarize_cost_basic():
    tick_df = pd.DataFrame({"busy_gpus": [4, 4, 0]})
    pricing = PricingConfig(gpu_hour_cost_usd=2.0, overhead_multiplier=1.0)
    out = summarize_cost(tick_df=tick_df, pricing=pricing, efficiency_factor=1.0, time_step_minutes=1)
    assert out["busy_gpu_hours"] > 0
    assert out["total_cost_usd"] > 0
