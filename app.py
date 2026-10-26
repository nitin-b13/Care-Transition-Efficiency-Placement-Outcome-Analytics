"""Module 6 - Streamlit dashboard. Run:  streamlit run app.py"""
import numpy as np, pandas as pd, plotly.graph_objects as go, plotly.express as px, streamlit as st
from module1_cleaning import clean
from module2_features import add_features
from module4_kpis import kpis
from module5_bottleneck_model import rule_flags

st.set_page_config(page_title="UAC Care Pipeline Analytics", page_icon="📊", layout="wide")
st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700&display=swap');
html, body, [class*="css"] {font-family:'Poppins',sans-serif;}
[data-testid="stMetric"] {background:#f8fafc;border:1px solid #e2e8f0;border-radius:14px;padding:14px 16px;}
[data-testid="stMetricLabel"] p {font-size:.8rem;color:#475569;}
h1,h2,h3 {color:#0f172a;}
</style>""", unsafe_allow_html=True)
BLUE, ORANGE, GREEN, RED = "#2563eb", "#f59e0b", "#16a34a", "#dc2626"

@st.cache_data
def load():
    df, log = clean()
    return add_features(df), log

data, cleaning_log = load()

# ---------------- sidebar ----------------
st.sidebar.header("Controls")
lo, hi = data["date"].min().date(), data["date"].max().date()
rng = st.sidebar.date_input("Date range", (lo, hi), min_value=lo, max_value=hi)
start, end = (rng if isinstance(rng, tuple) and len(rng) == 2 else (lo, hi))
view = st.sidebar.radio("Ratio view", ["Daily", "7-day rolling", "Monthly"], index=1)
st.sidebar.markdown("**Ratio metrics to show**")
show_t = st.sidebar.checkbox("Transfer efficiency", True)
show_d = st.sidebar.checkbox("Discharge effectiveness", True)
show_p = st.sidebar.checkbox("Pipeline throughput", True)
st.sidebar.markdown("**Alert thresholds**")
thr_b = st.sidebar.slider("Backlog rate alert (% of HHS care added per week, above)", -10.0, 15.0, 2.0, 0.5)
thr_t = st.sidebar.slider("Transfer efficiency alert (7-day, below)", 0.0, 1.5, 0.50, 0.05)
thr_d = st.sidebar.slider("Discharge effectiveness alert (7-day, below)", 0.0, 0.06, 0.010, 0.001, format="%.3f")

d = data[(data["date"].dt.date >= start) & (data["date"].dt.date <= end)].copy()
if d.empty:
    st.warning("No reported days in the selected range."); st.stop()
flags = rule_flags(d, thr_b, thr_t, thr_d)

# ---------------- header + KPIs ----------------
st.title("UAC Care Pipeline: Process Efficiency and Outcome Analytics")
st.caption(f"CBP custody → HHS care → sponsor placement · {len(d):,} reported days · {start:%d %b %Y} to {end:%d %b %Y}")
k = kpis(d)
c = st.columns(5)
c[0].metric("Transfer Efficiency Ratio", f"{k['transfer_eff']:.2f}", help="Transfers ÷ CBP custody")
c[1].metric("Discharge Effectiveness", f"{k['discharge_eff']*100:.2f}%", help="Discharges ÷ HHS care (per reported day)")
c[2].metric("Pipeline Throughput", f"{k['throughput']:.2f}", help="Total exits (discharges) ÷ total entries (apprehensions)")
c[3].metric("Backlog Accumulation", f"{k['backlog_rate']:+.1f}% / wk", help="Net HHS inflow (transfers − discharges) as % of average HHS care, per week")
c[4].metric("Outcome Stability", f"{k['stability']:.0f} / 100", help="100 × (1 − variability of discharges over 30 days)")

# ---------------- visual alerts (latest 7 reported days in range) ----------------
last = d.iloc[-1]; lf = flags.iloc[-1]
st.subheader("Alerts (latest reported day in range)")
a = st.columns(4)
def alert(col, name, on, text):
    (col.error if on else col.success)(f"**{name}**\n\n{text}")
alert(a[0], "Backlog", lf.backlog_alert, f"Backlog rate {last.backlog_rate_7:+.1f}% / wk")
alert(a[1], "Transfer speed", lf.slow_transfer_alert, f"7-day transfer efficiency {last.transfer_eff_7:.2f}")
alert(a[2], "Discharges", lf.low_discharge_alert, f"7-day discharge effectiveness {last.discharge_eff_7*100:.2f}%")
alert(a[3], "Sudden drop", lf.drop_alert, "Discharges fell sharply vs prior month" if lf.drop_alert else "No sudden drop")

tabs = st.tabs(["Care pipeline flow", "Transfer & discharge efficiency", "Bottleneck detection", "Outcome trends", "Data"])

# ---------------- tab 1 ----------------
with tabs[0]:
    tot_i, tot_t, tot_d = d.intake.sum(), d.transfers.sum(), d.discharges.sum()
    sk = go.Figure(go.Sankey(
        node=dict(label=["Apprehended into CBP", "CBP custody", "HHS care", "Sponsor placement"],
                  color=[ORANGE, ORANGE, BLUE, GREEN], pad=30, thickness=22),
        link=dict(source=[0, 1, 2], target=[1, 2, 3], value=[tot_i, tot_t, tot_d],
                  color=["rgba(245,158,11,.35)", "rgba(37,99,235,.35)", "rgba(22,163,74,.35)"],
                  label=["Intake", "Transfers", "Discharges"])))
    sk.update_layout(title=f"Total flow through the pipeline: {tot_i:,} in → {tot_t:,} transferred → {tot_d:,} discharged", height=380, margin=dict(t=60, b=10))
    st.plotly_chart(sk)
    l, r = st.columns(2)
    f1 = go.Figure([go.Scatter(x=d.date, y=d.hhs_care, name="HHS care", line=dict(color=BLUE))])
    f1.update_layout(title="Children in HHS care", height=330, margin=dict(t=50, b=10)); l.plotly_chart(f1)
    f2 = go.Figure([go.Scatter(x=d.date, y=d.cbp_custody, name="CBP custody", line=dict(color=ORANGE))])
    f2.update_layout(title="Children in CBP custody", height=330, margin=dict(t=50, b=10)); r.plotly_chart(f2)

# ---------------- tab 2 ----------------
with tabs[1]:
    if view == "Monthly":
        g = d.groupby("month").agg(t=("transfers", "sum"), c=("cbp_custody", "sum"), dc=("discharges", "sum"), h=("hhs_care", "sum"), i=("intake", "sum")).reset_index()
        g["x"] = pd.to_datetime(g["month"]); g["te"] = g.t / g.c; g["de"] = g.dc / g.h; g["tp"] = g.dc / g.i.replace(0, np.nan)
    else:
        g = d.copy(); g["x"] = g["date"]
        suf = "_7" if view == "7-day rolling" else ""
        g["te"] = g["transfer_eff" + suf]; g["de"] = g["discharge_eff" + suf]
        g["tp"] = g["throughput_7"] if suf else (g["discharges"] / g["intake"].replace(0, np.nan))
    if show_t or show_d:
        fig = go.Figure()
        if show_t: fig.add_scatter(x=g.x, y=g.te, name="Transfer efficiency (transfers ÷ CBP custody)", line=dict(color=BLUE))
        if show_d: fig.add_scatter(x=g.x, y=g.de * 20, name="Discharge effectiveness ×20 (scaled to share the axis)", line=dict(color=GREEN))
        if show_t: fig.add_hline(y=thr_t, line_dash="dot", line_color=RED, annotation_text="transfer alert threshold")
        fig.update_layout(title=f"Efficiency ratios ({view.lower()})", height=420, legend=dict(orientation="h", y=-0.2)); st.plotly_chart(fig)
    if show_p:
        fp = go.Figure(go.Scatter(x=g.x, y=g.tp.clip(upper=8), name="Throughput", line=dict(color=ORANGE)))
        fp.add_hline(y=1, line_dash="dot", line_color="#64748b", annotation_text="exits = entries")
        fp.update_layout(title=f"Pipeline throughput: exits ÷ entries ({view.lower()}, capped at 8)", height=340); st.plotly_chart(fp)
    if not (show_t or show_d or show_p): st.info("Switch on at least one ratio metric in the sidebar.")

# ---------------- tab 3 ----------------
with tabs[2]:
    colors = np.where(d.backlog_rate_7 > thr_b, RED, np.where(d.backlog_rate_7 > 0, ORANGE, GREEN))
    fb = go.Figure(go.Bar(x=d.date, y=d.backlog_rate_7, marker_color=colors, name="Backlog rate"))
    fb.add_hline(y=thr_b, line_dash="dot", line_color=RED, annotation_text="alert threshold")
    fb.update_layout(title="Backlog accumulation rate (7-day net HHS flow, % of HHS care) · red = above alert threshold", height=380); st.plotly_chart(fb)
    l, r = st.columns(2)
    fn = go.Figure([go.Scatter(x=d.date, y=d.net_hhs_flow_7, fill="tozeroy", line=dict(color=BLUE), name="Net HHS flow")])
    fn.update_layout(title="7-day net inflow to HHS (transfers − discharges)", height=320); l.plotly_chart(fn)
    runs = d[d.imbalance_run > 0].copy()
    runs["episode"] = (runs.imbalance_run == 1).cumsum()
    ep = runs.groupby("episode").agg(start=("date", "min"), end=("date", "max"), days=("imbalance_run", "max"), peak_hhs=("hhs_care", "max")).query("days >= 7").sort_values("days", ascending=False).head(10)
    r.markdown("**Longest sustained-imbalance periods** (transfers outpacing discharges, 7+ reported days)")
    r.dataframe(ep.assign(start=ep.start.dt.date, end=ep.end.dt.date), hide_index=True)
    st.caption(f"Alert days in range: backlog {int(flags.backlog_alert.sum())} · slow transfers {int(flags.slow_transfer_alert.sum())} · low discharges {int(flags.low_discharge_alert.sum())} · sudden drops {int(flags.drop_alert.sum())}")

# ---------------- tab 4 ----------------
with tabs[3]:
    m = d.groupby("month").agg(intake=("intake", "sum"), transfers=("transfers", "sum"), discharges=("discharges", "sum")).reset_index()
    fm = px.line(m, x="month", y=["intake", "transfers", "discharges"], color_discrete_sequence=[ORANGE, BLUE, GREEN])
    fm.update_layout(title="Month-over-month flows", height=360, legend_title=""); st.plotly_chart(fm)
    l, r = st.columns(2)
    fs = go.Figure([go.Scatter(x=d.date, y=d.stability_score, line=dict(color=BLUE), name="Stability")])
    dd = d[d.discharge_drop]; fs.add_scatter(x=dd.date, y=dd.stability_score, mode="markers", marker=dict(color=RED, size=6), name="Sudden drop")
    fs.update_layout(title="Outcome stability score and sudden discharge drops", height=330); l.plotly_chart(fs)
    wk = d.groupby("is_weekend")[["intake", "transfers", "discharges"]].mean().rename(index={False: "Weekday", True: "Weekend"}).reset_index()
    fw = px.bar(wk.melt("is_weekend"), x="is_weekend", y="value", color="variable", barmode="group", color_discrete_sequence=[ORANGE, BLUE, GREEN])
    fw.update_layout(title="Weekday vs weekend reports (avg per reported day)", height=330, xaxis_title="", yaxis_title="Children", legend_title=""); r.plotly_chart(fw)

# ---------------- tab 5 ----------------
with tabs[4]:
    q = st.text_input("Search (date, month, weekday, e.g. 'March 2024' or '2024-03' or 'Saturday')")
    show = d[["date", "weekday", "intake", "cbp_custody", "transfers", "hhs_care", "discharges", "transfer_eff", "discharge_eff", "net_hhs_flow"]].copy()
    if q:
        key = show["date"].dt.strftime("%B %Y %Y-%m-%d %A")
        show = show[key.str.contains(q, case=False, na=False)]
    st.write(f"{len(show):,} rows"); st.dataframe(show.assign(date=show.date.dt.date), hide_index=True)
    st.download_button("Download filtered rows (CSV)", show.to_csv(index=False), "uac_filtered.csv", "text/csv")
    with st.expander("Cleaning log"): st.json(cleaning_log)
