"""Module 5 - Bottleneck detection. Rule-based alerts + a simple model that predicts whether the HHS backlog will keep building.
Input: outputs/features.csv. Output: outputs/bottleneck_flags.csv, model_report.json"""
import json, numpy as np, pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, roc_auc_score

BASE = Path(__file__).parent
OUT = BASE / "outputs"
HORIZON = 14        # reporting days ahead
RISE_PCT = 2.0      # "backlog building" = HHS care up by more than 2% over the horizon
FEATURES = ["transfer_eff_7", "discharge_eff_7", "backlog_rate_7", "cbp_custody", "intake_7", "stability_score", "hhs_trend_14"]

def rule_flags(d, backlog_thr=2.0, transfer_thr=0.5, discharge_thr=0.01):
    """Default thresholds are also the dashboard's alert defaults."""
    f = pd.DataFrame({"date": d["date"]})
    f["backlog_alert"] = d["backlog_rate_7"] > backlog_thr
    f["slow_transfer_alert"] = d["transfer_eff_7"] < transfer_thr
    f["low_discharge_alert"] = d["discharge_eff_7"] < discharge_thr
    f["drop_alert"] = d["discharge_drop"]
    f["alerts"] = f[["backlog_alert", "slow_transfer_alert", "low_discharge_alert", "drop_alert"]].sum(axis=1)
    return f

if __name__ == "__main__":
    d = pd.read_csv(OUT / "features.csv", parse_dates=["date"])
    flags = rule_flags(d); flags.to_csv(OUT / "bottleneck_flags.csv", index=False)

    m = d.copy()
    m["hhs_trend_14"] = m["hhs_care"].pct_change(HORIZON) * 100
    m["target"] = ((m["hhs_care"].shift(-HORIZON) / m["hhs_care"] - 1) * 100 > RISE_PCT).astype(int)
    m = m.dropna(subset=FEATURES).iloc[:-HORIZON]
    cut = int(len(m) * 0.7)                                  # chronological split, no shuffling
    tr, te = m.iloc[:cut], m.iloc[cut:]
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced"))
    model.fit(tr[FEATURES], tr["target"])
    p = model.predict_proba(te[FEATURES])[:, 1]
    baseline = max(te["target"].mean(), 1 - te["target"].mean())
    coefs = dict(zip(FEATURES, model[-1].coef_[0].round(3)))
    rep = {"horizon_reporting_days": HORIZON, "rise_threshold_pct": RISE_PCT, "train_rows": len(tr), "test_rows": len(te),
           "test_period": [str(te.date.min().date()), str(te.date.max().date())],
           "share_of_test_days_with_rising_backlog": round(float(te["target"].mean()), 3),
           "accuracy": round(accuracy_score(te["target"], p > .5), 3), "baseline_accuracy": round(float(baseline), 3),
           "roc_auc": round(roc_auc_score(te["target"], p), 3) if te["target"].nunique() > 1 else None,
           "standardised_coefficients": coefs,
           "rule_alert_days": {c: int(flags[c].sum()) for c in flags.columns if c.endswith("_alert")}}
    json.dump(rep, open(OUT / "model_report.json", "w"), indent=2)
    print(json.dumps(rep, indent=2))
