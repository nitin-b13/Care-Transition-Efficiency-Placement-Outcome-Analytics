"""Module 4 - KPI calculation. Input: outputs/features.csv. Output: outputs/kpi_summary.json, kpi_monthly.csv"""
import json, numpy as np, pandas as pd
from pathlib import Path

BASE = Path(__file__).parent
OUT = BASE / "outputs"

def kpis(d: pd.DataFrame) -> dict:
    """KPIs for any slice of the features table (used by the dashboard too)."""
    if len(d) == 0:
        return {k: np.nan for k in ["transfer_eff", "discharge_eff", "throughput", "backlog_rate", "stability", "net_hhs_flow", "n_days"]}
    entries, exits = d["intake"].sum(), d["discharges"].sum()
    return {
        "transfer_eff": d["transfers"].sum() / d["cbp_custody"].sum(),        # Transfers / CBP custody
        "discharge_eff": d["discharges"].sum() / d["hhs_care"].sum(),         # Discharges / HHS care
        "throughput": exits / entries if entries else np.nan,                 # Total exits / total entries
        "backlog_rate": (d["transfers"].sum() - d["discharges"].sum()) / d["hhs_care"].mean() * 100 / max(len(d) / 7, 1),  # % of HHS load added per week
        "stability": float(d["stability_score"].mean()),                      # mean 30-day stability score
        "net_hhs_flow": int(d["transfers"].sum() - d["discharges"].sum()),
        "n_days": len(d),
    }

if __name__ == "__main__":
    d = pd.read_csv(OUT / "features.csv", parse_dates=["date"])
    summary = {"overall": kpis(d), **{str(y): kpis(g) for y, g in d.groupby("year")}}
    json.dump(summary, open(OUT / "kpi_summary.json", "w"), indent=2, default=float)
    monthly = pd.DataFrame({m: kpis(g) for m, g in d.groupby("month")}).T
    monthly.index.name = "month"; monthly.to_csv(OUT / "kpi_monthly.csv")
    for k, v in summary.items():
        print(k, {a: round(b, 3) if isinstance(b, float) else b for a, b in v.items()})
