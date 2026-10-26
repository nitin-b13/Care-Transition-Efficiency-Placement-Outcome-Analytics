"""Module 3 - Exploratory data analysis. Input: outputs/features.csv. Output: figures/*.png, outputs/eda_stats.json"""
import json, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

BASE = Path(__file__).parent
FIG = BASE / "figures"; FIG.mkdir(exist_ok=True)
BLUE, ORANGE, GREEN, RED, GREY = "#2563eb", "#f59e0b", "#16a34a", "#dc2626", "#64748b"
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "font.size": 10, "figure.dpi": 130})

def save(name):
    plt.tight_layout(); plt.savefig(FIG / name, bbox_inches="tight"); plt.close()

def run():
    d = pd.read_csv(BASE / "outputs" / "features.csv", parse_dates=["date"])
    m = d.groupby("month").agg(intake=("intake", "sum"), transfers=("transfers", "sum"), discharges=("discharges", "sum"),
                               cbp=("cbp_custody", "mean"), hhs=("hhs_care", "mean")).reset_index()
    m["month_dt"] = pd.to_datetime(m["month"])

    # 1 pipeline load over time
    fig, ax = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    ax[0].plot(d["date"], d["hhs_care"], color=BLUE); ax[0].set_title("Children in HHS care (daily)"); ax[0].set_ylabel("Children")
    ax[1].plot(d["date"], d["cbp_custody"], color=ORANGE); ax[1].set_title("Children in CBP custody (daily)"); ax[1].set_ylabel("Children")
    save("01_pipeline_load.png")

    # 2 monthly flows
    fig, ax = plt.subplots(figsize=(9, 4))
    for c, col, lab in [("intake", ORANGE, "Apprehended into CBP"), ("transfers", BLUE, "Transferred CBP to HHS"), ("discharges", GREEN, "Discharged to sponsors")]:
        ax.plot(m["month_dt"], m[c], label=lab, color=col, lw=2)
    ax.set_title("Monthly flows through the pipeline"); ax.set_ylabel("Children (sum of reported days)"); ax.legend(frameon=False)
    save("02_monthly_flows.png")

    # 3 monthly efficiency ratios
    me = d.groupby("month").agg(t=("transfers", "sum"), c=("cbp_custody", "sum"), dc=("discharges", "sum"), h=("hhs_care", "sum")).reset_index()
    me["transfer_eff"] = me.t / me.c; me["discharge_eff"] = me.dc / me.h; me["month_dt"] = pd.to_datetime(me["month"])
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    ax[0].plot(me.month_dt, me.transfer_eff, color=BLUE, lw=2); ax[0].set_title("Transfer efficiency (transfers / CBP custody)")
    ax[1].plot(me.month_dt, me.discharge_eff * 100, color=GREEN, lw=2); ax[1].set_title("Discharge effectiveness (% of HHS care per day)")
    for a in ax: a.tick_params(axis="x", labelrotation=30); a.title.set_fontsize(9)
    save("03_efficiency_ratios.png")

    # 4 weekday vs weekend
    wk = d.groupby("is_weekend")[["intake", "transfers", "discharges"]].mean().rename(index={False: "Weekday", True: "Weekend"})
    wk.plot(kind="bar", figsize=(7, 3.8), color=[ORANGE, BLUE, GREEN], rot=0)
    plt.title("Average daily flow: weekday vs weekend reports"); plt.xlabel(""); plt.ylabel("Children per reported day"); plt.legend(frameon=False)
    save("04_weekday_weekend.png")

    # 5 backlog
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(d["date"], d["backlog_rate_7"], color=GREY, lw=1)
    ax.fill_between(d["date"], 0, d["backlog_rate_7"], where=d["backlog_rate_7"] > 0, color=RED, alpha=.5, label="Backlog building")
    ax.fill_between(d["date"], 0, d["backlog_rate_7"], where=d["backlog_rate_7"] <= 0, color=GREEN, alpha=.35, label="Backlog clearing")
    ax.axhline(0, color="black", lw=.6); ax.set_title("Backlog accumulation rate (7-day net HHS flow, % of HHS care)"); ax.legend(frameon=False)
    save("05_backlog_rate.png")

    # 6 outcome stability
    fig, ax = plt.subplots(figsize=(9, 3.8))
    ax.plot(d["date"], d["stability_score"], color=BLUE)
    ax.scatter(d.loc[d.discharge_drop, "date"], d.loc[d.discharge_drop, "stability_score"], color=RED, s=10, label="Sudden discharge drop")
    ax.set_title("Outcome stability score (30-day) and sudden drops"); ax.set_ylabel("Score (0-100)"); ax.legend(frameon=False)
    save("06_outcome_stability.png")

    # 7 correlation
    cols = ["intake", "cbp_custody", "transfers", "hhs_care", "discharges"]
    corr = d[cols].corr()
    fig, ax = plt.subplots(figsize=(5.5, 4.5)); im = ax.imshow(corr, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(5)); ax.set_xticklabels(["Intake", "CBP", "Transfers", "HHS", "Discharges"], rotation=30)
    ax.set_yticks(range(5)); ax.set_yticklabels(["Intake", "CBP", "Transfers", "HHS", "Discharges"])
    for i in range(5):
        for j in range(5): ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8, color="white" if corr.iloc[i, j] > .55 else "black")
    ax.set_title("Correlation of pipeline stages"); save("07_correlation.png")

    peak = d.loc[d.hhs_care.idxmax()]; low = d.loc[d.hhs_care.idxmin()]
    yr = d.groupby("year").agg(intake=("intake", "sum"), transfers=("transfers", "sum"), discharges=("discharges", "sum"), hhs=("hhs_care", "mean")).round(0)
    stats = {
        "peak_hhs": [str(peak.date.date()), int(peak.hhs_care)], "low_hhs": [str(low.date.date()), int(low.hhs_care)],
        "peak_cbp": [str(d.loc[d.cbp_custody.idxmax(), "date"].date()), int(d.cbp_custody.max())],
        "weekday_avg": wk.loc["Weekday"].round(1).to_dict(), "weekend_avg": wk.loc["Weekend"].round(1).to_dict(),
        "weekend_rows": int(d.is_weekend.sum()),
        "corr_transfers_discharges": round(float(d[["transfers", "discharges"]].corr().iloc[0, 1]), 2),
        "corr_intake_transfers": round(float(d[["intake", "transfers"]].corr().iloc[0, 1]), 2),
        "yearly": yr.to_dict(orient="index"),
        "pct_days_backlog_building": round(float((d.backlog_rate_7 > 0).mean() * 100), 1),
        "longest_imbalance_run": int(d.imbalance_run.max()),
        "longest_imbalance_end": str(d.loc[d.imbalance_run.idxmax(), "date"].date()),
        "sudden_drop_days": int(d.discharge_drop.sum()),
    }
    json.dump(stats, open(BASE / "outputs" / "eda_stats.json", "w"), indent=2, default=str)
    return stats

if __name__ == "__main__":
    s = run()
    for k, v in s.items(): print(k, "->", v)
