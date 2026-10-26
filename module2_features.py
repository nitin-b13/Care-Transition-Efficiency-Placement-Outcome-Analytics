"""Module 2 - Feature engineering. Input: outputs/clean.csv. Output: outputs/features.csv"""
import numpy as np, pandas as pd
from pathlib import Path

BASE = Path(__file__).parent
OUT = BASE / "outputs"

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy().sort_values("date").reset_index(drop=True)
    d["year"] = d["date"].dt.year
    d["month"] = d["date"].dt.to_period("M").astype(str)
    d["weekday"] = d["date"].dt.day_name()
    d["is_weekend"] = d["date"].dt.dayofweek >= 5
    d["gap_days"] = d["date"].diff().dt.days.fillna(1).astype(int)

    # daily ratios
    d["transfer_eff"] = d["transfers"] / d["cbp_custody"]          # Transfers / CBP custody
    d["discharge_eff"] = d["discharges"] / d["hhs_care"]           # Discharges / HHS care
    d["net_hhs_flow"] = d["transfers"] - d["discharges"]           # + = HHS load building
    d["net_cbp_flow"] = d["intake"] - d["transfers"]               # + = CBP load building
    d["hhs_change"] = d["hhs_care"].diff().fillna(0)

    # 7-reporting-day rolling windows (smooths zero-days and reporting gaps)
    for c in ["intake", "transfers", "discharges", "net_hhs_flow"]:
        d[f"{c}_7"] = d[c].rolling(7, min_periods=3).sum()
    d["throughput_7"] = d["discharges_7"] / d["intake_7"].replace(0, np.nan)   # exits / entries
    d["transfer_eff_7"] = d["transfers_7"] / d["cbp_custody"].rolling(7, min_periods=3).sum()
    d["discharge_eff_7"] = d["discharges_7"] / d["hhs_care"].rolling(7, min_periods=3).sum()
    d["backlog_rate_7"] = d["net_hhs_flow_7"] / d["hhs_care"] * 100            # % of HHS load added per week

    # sustained imbalance: consecutive reporting days where 7-day net HHS flow > 0
    pos = (d["net_hhs_flow_7"] > 0).astype(int)
    run = pos.groupby((pos != pos.shift()).cumsum()).cumsum()
    d["imbalance_run"] = np.where(pos == 1, run, 0)

    # outcome stability: 100 * (1 - coefficient of variation of discharges, 30-day window)
    roll = d["discharges"].rolling(30, min_periods=15)
    cv = roll.std() / roll.mean().replace(0, np.nan)
    d["stability_score"] = (100 * (1 - cv.clip(upper=1))).round(1)

    # sudden drop in discharges: 7-day average < 70% of the prior 30-day average
    recent = d["discharges"].rolling(7, min_periods=3).mean()
    prior = d["discharges"].rolling(30, min_periods=15).mean().shift(7)
    d["discharge_drop"] = (recent < 0.7 * prior).fillna(False)
    return d

if __name__ == "__main__":
    f = add_features(pd.read_csv(OUT / "clean.csv", parse_dates=["date"]))
    f.to_csv(OUT / "features.csv", index=False)
    print(f"{len(f)} rows, {f.shape[1]} columns -> outputs/features.csv")
    print(f[["transfer_eff", "discharge_eff", "throughput_7", "backlog_rate_7", "stability_score"]].describe().round(3).T)
    print("weekend rows:", int(f.is_weekend.sum()), "| sudden-drop days:", int(f.discharge_drop.sum()),
          "| longest imbalance run:", int(f.imbalance_run.max()))
