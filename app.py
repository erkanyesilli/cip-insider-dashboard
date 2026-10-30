"""Dashboard: Insider purchases and subsequent stock returns (CIP group 201).

Run:   streamlit run app.py
Data:  data/*.parquet (created once with build_data.py)

All charts show aggregated values (median, mean, distributions), never single observations.
Filters and aggregations are vectorized (NumPy/pandas), so the dashboard stays fast.
"""
import json
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

DATA = Path(__file__).resolve().parent / "data"
HORIZONS = (30, 90)
DAYS = np.arange(-20, 64)          # trading days relative to entry
GREEN, RED, BLUE, GRAY, ORANGE = "#16a34a", "#dc2626", "#2563eb", "#6b7280", "#ea580c"
ROLE_COLORS = {"CEO": "#b91c1c", "CFO": "#ea580c", "Other officer": "#7c3aed", "Director": "#0d9488",
               "10% owner": "#854d0e", "Other": "#db2777"}

# The cleaning steps are stored in German in data/funnel.parquet: translated here
STEP_EN = {
    "Alle Aktien-Transaktionen (Form 3/4/5)": ("All stock transactions (Forms 3/4/5)", "Everything in the SEC tables"),
    "Nur Code P (Kauf an der Börse)": ("Code P only (open-market purchase)",
                                       "Sales, gifts, option exercises etc. are not voluntary purchases"),
    "Nur Form 4": ("Form 4 only", "4/A = amendment of an earlier filing (would double count), Form 5 = annual report"),
    "Stückzahl und Preis > 0": ("Shares and price > 0", "Without price or shares there is no purchase value"),
    "Meldung 0-30 Tage nach dem Kauf": ("Filed 0–30 days after the trade",
                                         "Negative or > 30 days = typo in the date (e.g. year 0024) or very late"),
    "Doppelte Meldungen entfernt": ("Duplicate reports removed",
                                    "Same trade reported by several related owners (e.g. fund + its manager)"),
    "Nur Stamm-Aktien": ("Common stock only", "Preferred stock, warrants, notes trade at a different price"),
    "Ohne Fonds": ("Funds excluded", "No corporate inside knowledge; price follows the fund's net assets"),
    "Meldungen (Filings) mit Käufen": ("Filings with purchases", "Several trades in one filing are combined"),
    "Ticker mit Kursdaten": ("Ticker with price data", "No ticker or no Yahoo prices = no return (mostly delisted)"),
    "Events (Firma x Meldetag)": ("Events (company × filing day)",
                                  "Same company on the same day = same price path, counted once"),
    "Kurs am Einstiegstag vorhanden": ("Price on entry day", "Stock traded on the day after the filing"),
    "Kurs nach 30 und 90 Tagen vorhanden = Stichprobe": ("Price after 30 and 90 days = sample",
                                                         "Missing end price usually means delisted or acquired"),
}
LEVEL_EN = {"Transaktion": "Transaction", "Meldung": "Filing", "Event": "Event"}

st.set_page_config(page_title="Insider Purchases & Returns", page_icon="📈", layout="wide")


# ------------------------------------------------------------------ data
@st.cache_data
def load():
    ev = pd.read_parquet(DATA / "events.parquet")
    paths = pd.read_parquet(DATA / "paths.parquet").to_numpy(dtype="float32")
    filings = pd.read_parquet(DATA / "filings.parquet")
    prices = pd.read_parquet(DATA / "prices.parquet")
    funnel = pd.read_parquet(DATA / "funnel.parquet")
    meta = json.loads((DATA / "meta.json").read_text(encoding="utf-8"))
    # winsorize the paths once (bounds per day from the full sample)
    paths_w = np.clip(paths, np.array(meta["paths_lo"], "float32"), np.array(meta["paths_hi"], "float32"))
    return ev, paths, paths_w, filings, prices, funnel, meta


ev, PATHS, PATHS_W, filings, prices, funnel, meta = load()
ROLES = [r for r in meta["roles"] if r in set(ev["TOP_ROLE"])]


# ------------------------------------------------------------------ helpers
def n_fmt(x):
    return f"{int(round(x)):,}"


def pct(x, d=1, sign=True):
    if pd.isna(x):
        return "–"
    return (f"{x * 100:+.{d}f}" if sign else f"{x * 100:.{d}f}") + "%"


def median_ci(x):
    """Median with 95% confidence interval from order statistics: distribution-free, no bootstrap.
    Bounds = sorted values at ranks n/2 ± 1.96·√n/2."""
    x = np.sort(x[~np.isnan(x)])
    n = len(x)
    if n < 10:
        return np.nan, np.nan, np.nan
    half = 1.96 * np.sqrt(n) / 2
    lo, hi = int(np.floor(n / 2 - half)), int(np.ceil(n / 2 + half))
    return float(np.median(x)), float(x[max(lo, 0)]), float(x[min(hi, n - 1)])


def summary(df, by, h):
    """Statistics per group: n, median with CI, winsorized mean, share beating the market."""
    rows = []
    for name, g in df.groupby(by, observed=True, sort=True):
        x = g[f"EX_{h}"].to_numpy()
        med, lo, hi = median_ci(x)
        rows.append({"Group": str(name), "n": len(g), "Median": med, "CI low": lo, "CI high": hi,
                     "Mean (winsorized)": g[f"EX_{h}_W"].mean(), "Share > market": (x > 0).mean()})
    return pd.DataFrame(rows)


def path_stats(idx):
    """Average path over all selected events (vectorized over the whole matrix)."""
    p, pw = PATHS[idx], PATHS_W[idx]
    cnt = np.sum(~np.isnan(pw), axis=0)
    mean = np.nanmean(pw, axis=0)
    se = np.nanstd(pw, axis=0, ddof=1) / np.sqrt(cnt)
    return pd.DataFrame({"Day": DAYS, "Mean": mean, "lo": mean - 1.96 * se, "hi": mean + 1.96 * se,
                         "Median": np.nanmedian(p, axis=0)})


def group_column(df, kind):
    if kind == "Role (highest on filing day)":
        return pd.Categorical(df["TOP_ROLE"], ROLES, ordered=True)
    if kind == "Cluster vs. single purchase":
        return np.where(df["CLUSTER"], "Cluster (≥ 2 insiders in 30 d)", "Single purchase")
    if kind == "Purchase value":
        return pd.cut(df["VALUE_USD"], [0, 1e4, 1e5, 1e6, np.inf],
                      labels=["< $10k", "$10k–100k", "$100k–1M", "> $1M"])
    if kind == "Share price at purchase":
        return pd.cut(df["AVG_PRICE"], [0, 1, 5, 20, np.inf],
                      labels=["< $1 (penny)", "$1–5", "$5–20", "> $20"])
    if kind == "Quarter":
        return df["FILING_DATE"].dt.to_period("Q").astype(str)
    return np.where(df["ANY_10B5_1"], "Planned (10b5-1)", "Not planned")


# ------------------------------------------------------------------ filters
with st.sidebar:
    st.header("Filters")
    dmin, dmax = ev["FILING_DATE"].min().date(), ev["FILING_DATE"].max().date()
    period = st.slider("Filing date", min_value=dmin, max_value=dmax, value=(dmin, dmax), format="MMM D, YYYY")
    roles = st.multiselect("Role (highest on filing day)", ROLES, default=ROLES)
    kind = st.radio("Purchase type", ["All", "Cluster purchases only", "Single purchases only"])
    no_penny = st.checkbox("Exclude penny stocks (< $1)")
    no_plan = st.checkbox("Exclude planned purchases (10b5-1)")
    h = st.radio("Horizon", HORIZONS, index=1, horizontal=True, format_func=lambda x: f"{x} days")

mask = (ev["FILING_DATE"].dt.date.between(*period) & ev["TOP_ROLE"].isin(roles))
if kind == "Cluster purchases only":
    mask &= ev["CLUSTER"]
elif kind == "Single purchases only":
    mask &= ~ev["CLUSTER"]
if no_penny:
    mask &= ~ev["PENNY"]
if no_plan:
    mask &= ~ev["ANY_10B5_1"]
sel = ev[mask]
idx = sel.index.to_numpy()
st.sidebar.caption(f"**{n_fmt(len(sel))}** of {n_fmt(len(ev))} events selected")

st.title("Insider Purchases and Stock Returns")
st.caption("SEC Form 4 (code P), 2024–2025 · Prices: Yahoo Finance · Excess return = stock return − SPY return · "
           "Entry at the close of the first trading day after the filing")

if len(sel) < 30:
    st.warning("Too few events for a meaningful analysis. Please relax the filters.")
    st.stop()

tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Comparison (drill-down)", "Single company", "Data & method"])

# ------------------------------------------------------------------ tab 1: overview
with tab1:
    x = sel[f"EX_{h}"].to_numpy()
    med, lo, hi = median_ci(x)
    c = st.columns(5)
    c[0].metric("Events", n_fmt(len(sel)))
    c[1].metric("Companies", n_fmt(sel["ISSUERCIK"].nunique()))
    c[2].metric(f"Median excess return {h} d", pct(med), help=f"95% CI: {pct(lo)} to {pct(hi)}")
    c[3].metric("Mean (winsorized)", pct(sel[f"EX_{h}_W"].mean()))
    c[4].metric("Share beating the market", pct((x > 0).mean(), 0, False))

    ps = path_stats(idx)
    long = ps.melt(id_vars="Day", value_vars=["Mean", "Median"], var_name="Statistic", value_name="Value")
    color = alt.Scale(domain=["Mean", "Median"], range=[GREEN, RED])
    band = alt.Chart(ps).mark_area(opacity=0.15, color=GREEN).encode(
        x="Day:Q", y=alt.Y("lo:Q", axis=alt.Axis(format="%")), y2="hi:Q")
    lines = alt.Chart(long).mark_line(strokeWidth=2.5).encode(
        x=alt.X("Day:Q", title="Trading days relative to entry"),
        y=alt.Y("Value:Q", title="Cumulative excess return", axis=alt.Axis(format="%")),
        color=alt.Color("Statistic:N", scale=color, legend=alt.Legend(orient="top-right", title=None)),
        tooltip=["Day:Q", "Statistic:N", alt.Tooltip("Value:Q", format="+.2%")])
    rules = (alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color="black").encode(y="y:Q")
             + alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(strokeDash=[4, 4]).encode(x="x:Q"))

    left, right = st.columns([3, 2])
    with left:
        st.subheader("Path around the purchase")
        st.altair_chart((band + rules + lines).properties(height=360), width="stretch")
        st.caption("Band = 95% confidence interval of the mean. Left of 0: price development before entry.")
    with right:
        st.subheader(f"Distribution after {h} days")
        counts, edges = np.histogram(np.clip(x, -1, 1), bins=40, range=(-1, 1))
        hist = pd.DataFrame({"from": edges[:-1], "to": edges[1:], "Count": counts})
        bars = alt.Chart(hist).mark_bar(color=BLUE, opacity=0.8, binSpacing=0).encode(
            x=alt.X("from:Q", bin="binned", title=f"Excess return {h} d (capped at ±100%)",
                    axis=alt.Axis(format="%")), x2="to:Q", y=alt.Y("Count:Q", title="Number of events"),
            tooltip=[alt.Tooltip("from:Q", format=".0%"), alt.Tooltip("to:Q", format=".0%"), "Count:Q"])
        marks = pd.DataFrame({"Value": [med, sel[f"EX_{h}_W"].mean()], "Statistic": ["Median", "Mean"]})
        mrule = alt.Chart(marks).mark_rule(strokeWidth=2.5).encode(
            x="Value:Q", color=alt.Color("Statistic:N", scale=color, legend=None),
            tooltip=["Statistic:N", alt.Tooltip("Value:Q", format="+.2%")])
        st.altair_chart((bars + mrule).properties(height=360), width="stretch")
        st.caption("Red = median, green = mean (winsorized).")

    st.info("**How to read:** Insiders typically buy *after* a price decline (curve above 0 on the left). "
            "Afterwards the typical purchase (median) slightly lags the market, while the average is slightly "
            "positive because a few stocks rise very strongly.")

# ------------------------------------------------------------------ tab 2: comparison
with tab2:
    by = st.selectbox("Group by", ["Role (highest on filing day)", "Cluster vs. single purchase", "Purchase value",
                                   "Share price at purchase", "Quarter", "Planned purchase (10b5-1)"])
    g = sel.assign(Group=group_column(sel, by))
    tab = summary(g, "Group", h)
    order = tab["Group"].tolist()

    left, right = st.columns(2)
    with left:
        st.subheader(f"Median with 95% CI after {h} days")
        y = alt.Y("Group:N", sort=order, title=None)
        ci = alt.Chart(tab).mark_rule(strokeWidth=2.5, color=BLUE).encode(
            y=y, x=alt.X("CI low:Q", title=f"Median excess return {h} d", axis=alt.Axis(format="%")),
            x2="CI high:Q")
        pts = alt.Chart(tab).mark_circle(size=110, color=BLUE).encode(
            y=y, x="Median:Q",
            tooltip=["Group:N", "n:Q", alt.Tooltip("Median:Q", format="+.2%"),
                     alt.Tooltip("CI low:Q", format="+.2%"), alt.Tooltip("CI high:Q", format="+.2%")])
        zero = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(color="black").encode(x="x:Q")
        st.altair_chart((zero + ci + pts).properties(height=max(300, 45 * len(tab))), width="stretch")
    with right:
        st.subheader("Median path per group")
        frames = []
        for name, gg in g.groupby("Group", observed=True):
            if len(gg) >= 30:
                frames.append(pd.DataFrame({"Day": DAYS, "Median": np.nanmedian(PATHS[gg.index.to_numpy()], axis=0),
                                            "Group": str(name)}))
        if frames:
            pl = pd.concat(frames)
            st.altair_chart(alt.Chart(pl).mark_line(strokeWidth=2).encode(
                x=alt.X("Day:Q", title="Trading days relative to entry"),
                y=alt.Y("Median:Q", title="Median excess return (cumulative)", axis=alt.Axis(format="%")),
                color=alt.Color("Group:N", sort=order, legend=alt.Legend(orient="right", title=None)),
                tooltip=["Group:N", "Day:Q", alt.Tooltip("Median:Q", format="+.2%")]
            ).properties(height=max(300, 45 * len(tab))), width="stretch")
        st.caption("Only groups with at least 30 events.")

    show = tab.copy()
    pct_cols = ["Median", "CI low", "CI high", "Mean (winsorized)", "Share > market"]
    show[pct_cols] = show[pct_cols] * 100
    st.dataframe(show, hide_index=True, width="stretch",
                 column_config={c: st.column_config.NumberColumn(c, format="%.1f%%") for c in pct_cols})
    st.caption("If the confidence intervals of two groups overlap strongly, the difference is not reliable.")

# ------------------------------------------------------------------ tab 3: single company
with tab3:
    firms = (sel.groupby(["ISSUERCIK", "ISSUERNAME", "TICKER"]).size().rename("n").reset_index()
                .sort_values("n", ascending=False))
    labels = {r.ISSUERCIK: f"{r.ISSUERNAME} ({r.TICKER}) · {r.n} events" for r in firms.itertuples()}
    cik = st.selectbox("Company (sorted by number of purchase events)", list(labels), format_func=labels.get)
    fe = sel[sel["ISSUERCIK"] == cik]
    ff = filings[filings["EVENT_ID"].isin(fe["EVENT_ID"])].sort_values("FILING_DATE")
    ticker = fe["TICKER"].iloc[0]

    c = st.columns(4)
    c[0].metric("Purchase events", n_fmt(len(fe)))
    c[1].metric("Different insiders", n_fmt(ff["OWNER_NAME"].nunique()))
    c[2].metric("Total purchase value", f"${ff['VALUE_USD'].sum() / 1e6:,.1f}M")
    c[3].metric(f"Median excess return {h} d", pct(fe[f"EX_{h}"].median()))

    px = prices[[ticker, "SPY"]].dropna()
    px = px.loc[px.index >= pd.Timestamp("2023-12-01")]
    idxd = (px / px.iloc[0] * 100).reset_index().melt(id_vars="Date", var_name="Security", value_name="Index")
    idxd["Security"] = idxd["Security"].map({ticker: f"{ticker} (stock)", "SPY": "SPY (market)"})
    stock_idx = (px[ticker] / px[ticker].iloc[0] * 100)
    marks = ff.assign(Index=stock_idx.reindex(ff["FILING_DATE"], method="ffill").to_numpy(),
                      Value=ff["VALUE_USD"], Ex=ff[f"EX_{h}"])
    line = alt.Chart(idxd).mark_line(strokeWidth=2).encode(
        x=alt.X("Date:T", title=None, axis=alt.Axis(format="%b %Y")), y=alt.Y("Index:Q", title="Price (start = 100)"),
        color=alt.Color("Security:N", scale=alt.Scale(domain=[f"{ticker} (stock)", "SPY (market)"],
                                                      range=[BLUE, "#9ca3af"]),
                        legend=alt.Legend(orient="top", title=None)))
    dots = alt.Chart(marks).mark_circle(opacity=0.85, stroke="white").encode(
        x="FILING_DATE:T", y="Index:Q",
        size=alt.Size("Value:Q", title="Purchase value ($)", scale=alt.Scale(range=[40, 600])),
        color=alt.Color("ROLE:N", title="Role", scale=alt.Scale(domain=list(ROLE_COLORS), range=list(ROLE_COLORS.values()))),
        tooltip=[alt.Tooltip("FILING_DATE:T", title="Filing date"), alt.Tooltip("OWNER_NAME:N", title="Insider"),
                 alt.Tooltip("ROLE:N", title="Role"), alt.Tooltip("Value:Q", title="Value ($)", format=",.0f"),
                 alt.Tooltip("Ex:Q", title=f"Excess return {h} d", format="+.1%")])
    st.altair_chart(alt.layer(line, dots).resolve_scale(color="independent").properties(height=380), width="stretch")
    st.caption("Dots = purchase filings (size = value, color = role). Hover for details.")

    tbl = ff[["FILING_DATE", "TRANS_DATE", "OWNER_NAME", "ROLE", "VALUE_USD", "EX_30", "EX_90"]].copy()
    tbl[["EX_30", "EX_90"]] *= 100
    st.dataframe(tbl, hide_index=True, width="stretch", column_config={
        "FILING_DATE": st.column_config.DateColumn("Filing date", format="YYYY-MM-DD"),
        "TRANS_DATE": st.column_config.DateColumn("Trade date", format="YYYY-MM-DD"),
        "OWNER_NAME": "Insider", "ROLE": "Role",
        "VALUE_USD": st.column_config.NumberColumn("Value", format="dollar"),
        "EX_30": st.column_config.NumberColumn("Excess 30 d", format="%+.1f%%"),
        "EX_90": st.column_config.NumberColumn("Excess 90 d", format="%+.1f%%")})

# ------------------------------------------------------------------ tab 4: data & method
with tab4:
    left, right = st.columns([3, 2])
    with left:
        st.subheader("From raw data to the sample")
        fun = funnel[funnel.index >= funnel.index[funnel["Schritt"].str.startswith("Nur Code P")][0]].copy()
        fun["Step"] = fun["Schritt"].map(lambda s: STEP_EN.get(s, (s, ""))[0])
        fun["Why"] = fun["Schritt"].map(lambda s: STEP_EN.get(s, (s, ""))[1])
        fun["Level"] = fun["Ebene"].map(LEVEL_EN)
        fun = fun.rename(columns={"Anzahl": "Rows"})
        base = alt.Chart(fun).encode(
            y=alt.Y("Step:N", sort=None, title=None, axis=alt.Axis(labelLimit=320, labelOverlap=False)))
        txt = base.mark_text(align="left", dx=4, fontSize=11).encode(
            x="Rows:Q", text=alt.Text("Rows:Q", format=","))
        st.altair_chart((base.mark_bar().encode(
            x=alt.X("Rows:Q", title="Number of rows"),
            color=alt.Color("Level:N", scale=alt.Scale(domain=["Transaction", "Filing", "Event"],
                                                       range=[BLUE, ORANGE, GREEN]),
                            legend=alt.Legend(orient="bottom", title=None)),
            tooltip=["Level:N", "Step:N", alt.Tooltip("Rows:Q", format=","), "Why:N"]
        ) + txt).properties(height=430), width="stretch")
        st.caption(f"Start: {n_fmt(funnel['Anzahl'].iloc[0])} transactions of all kinds. Hover over a bar for the reason.")
    with right:
        st.subheader("Price data from Yahoo Finance")
        a, b = st.columns(2)
        a.metric("Filings with prices (before)", pct(meta["coverage_repo"], 0, False))
        b.metric("Filings with prices (after)", pct(meta["coverage_now"], 0, False))
        st.markdown(
            f"- The first download returned no prices for **{n_fmt(meta['n_tickers_missing'])}** of "
            f"{n_fmt(meta['n_tickers_repo'])} tickers, including **SPY** (Yahoo rate limit).\n"
            f"- Re-downloaded slowly (small batches, pauses) and tickers corrected: **{n_fmt(meta['n_tickers_fixed'])}** recovered.\n"
            f"- Remaining: mostly delisted companies → **survivorship bias** (returns rather too good).")
        st.subheader("Method in brief")
        st.markdown(
            "- **Event** = company × filing day (several insiders on the same day count once).\n"
            "- **Entry** = close on the first trading day *after* the filing (no look-ahead).\n"
            "- **Excess return** = stock return − SPY return over the same period.\n"
            "- **Cluster** = ≥ 2 different insiders of the same company in the 30 days *up to* the filing day.\n"
            "- **Means winsorized** (1%/99%), **median CI** from order statistics.")
