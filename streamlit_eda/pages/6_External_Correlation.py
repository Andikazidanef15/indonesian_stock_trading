import streamlit as st
import pandas as pd
import numpy as np

from src.sidebar import render_sidebar
from src.utils import load_ohlcv, load_external_returns, combine_returns
from src.stats import calculate_daily_price_metrics
from src.visualization import visualize_correlation_heatmap, visualize_rolling_correlation

ticker, period = render_sidebar()

st.title(":material/trending_up: External Correlation Analysis")
st.caption(f"Analysing **{ticker}** over **{period}**")
st.write(
    "Examines how the selected stock moves relative to external market factors. "
    "Choose any combination of indices, currencies, commodities, or other stocks below."
)
st.divider()

# ── Comparison ticker configuration ───────────────────────────────────────────
PRESETS: dict[str, str] = {
    "IHSG (^JKSE)":   "^JKSE",
    "USD/IDR":         "USDIDR=X",
    "S&P 500":         "^GSPC",
    "VIX":             "^VIX",
    "Gold":            "GC=F",
    "Crude Oil (WTI)": "CL=F",
    "BBRI.JK":         "BBRI.JK",
    "BMRI.JK":         "BMRI.JK",
    "TLKM.JK":         "TLKM.JK",
    "ASII.JK":         "ASII.JK",
    "ANTM.JK":         "ANTM.JK",
}
COLORS = ["steelblue", "seagreen", "darkorange", "purple",
          "crimson",   "teal",     "saddlebrown", "slateblue",
          "olive",     "deeppink",  "coral"]

with st.container(border=True):
    st.markdown("**Comparison tickers**")
    col_sel, col_custom = st.columns([2, 1])

    with col_sel:
        selected_labels = st.multiselect(
            "Select from presets",
            options=list(PRESETS.keys()),
            default=["IHSG (^JKSE)", "USD/IDR"],
            help="Pick one or more indices, currencies, commodities, or IDX stocks.",
        )

    with col_custom:
        custom_input = st.text_input(
            "Add custom tickers (comma-separated)",
            placeholder="e.g. NVDA, BTC-USD, ^N225",
            help="Any valid Yahoo Finance ticker symbol.",
        )

# Build the final list of (label, symbol) pairs to compare against
compare: list[tuple[str, str]] = [(lbl, PRESETS[lbl]) for lbl in selected_labels]

if custom_input.strip():
    for raw in custom_input.split(","):
        sym = raw.strip().upper()
        if sym and sym != ticker.upper():
            compare.append((sym, sym))   # label = symbol for custom entries

if not compare:
    st.info("Select at least one comparison ticker above.")
    st.stop()

# ── Data loading ───────────────────────────────────────────────────────────────
try:
    df = load_ohlcv(ticker, period)
except Exception as e:
    st.error(str(e))
    st.stop()

log_returns = calculate_daily_price_metrics(df)["log_returns"]
start_date  = df.index.min()
end_date    = df.index.max()

# Load each comparison ticker; skip ones that fail
loaded_labels:  list[str]        = []
loaded_returns: list[pd.Series]  = []

with st.spinner("Fetching comparison data…"):
    for label, symbol in compare:
        try:
            ret = load_external_returns(symbol, start=start_date, end=end_date)
            loaded_labels.append(label)
            loaded_returns.append(ret)
        except Exception:
            st.warning(f"Could not load data for **{label}** (`{symbol}`) — skipped.")

if not loaded_labels:
    st.error("None of the comparison tickers could be loaded. Try different symbols.")
    st.stop()

combined = combine_returns(
    [ticker] + [sym for _, sym in compare[:len(loaded_labels)]],
    [log_returns] + loaded_returns,
)
combined.columns = [ticker] + loaded_labels

if len(combined) < 30:
    st.warning(
        "Fewer than 30 aligned trading days available. "
        "Results may be unreliable — try a longer time horizon."
    )

# ── Section 1: Correlation Matrix ─────────────────────────────────────────────
st.subheader("Correlation Matrix")
st.caption("Based on log returns over the selected period.")

corr = combined.corr()

if corr.isna().any().any():
    st.warning("Correlation matrix contains NaN — some pairs have insufficient overlapping data.")

st.altair_chart(visualize_correlation_heatmap(corr), use_container_width=True)

with st.expander("Raw correlation values"):
    st.dataframe(corr.round(4), use_container_width=True)

# Per-factor interpretation
st.subheader("Interpretation")
cols = st.columns(min(len(loaded_labels), 3))
for i, label in enumerate(loaded_labels):
    val = corr.loc[ticker, label]
    col = cols[i % len(cols)]
    if pd.isna(val):
        col.warning(f"**vs {label}:** N/A")
    elif abs(val) > 0.7:
        col.warning(f"**vs {label} ({val:.3f}):** Strong {'positive' if val > 0 else 'negative'} relationship.")
    elif abs(val) > 0.4:
        col.info(f"**vs {label} ({val:.3f}):** Moderate correlation.")
    else:
        col.success(f"**vs {label} ({val:.3f}):** Low correlation — mostly independent.")

st.divider()

# ── Section 2: Rolling Correlation ────────────────────────────────────────────
st.subheader("Rolling Correlation")

rolling_window = st.slider(
    "Rolling window (days)",
    min_value=10, max_value=120, value=60, step=5,
    help="Number of trading days for the rolling correlation window.",
)

effective_window = rolling_window
if len(combined) <= rolling_window:
    effective_window = max(10, len(combined) // 5)
    st.info(f"Not enough data for {rolling_window}-day window — using {effective_window} days instead.")

# Build roll_df with one column per comparison ticker
roll_data = {"Date": combined.index}
for label in loaded_labels:
    roll_data[label] = combined[ticker].rolling(effective_window).corr(combined[label]).values

roll_df = pd.DataFrame(roll_data).dropna()
roll_df["Date"] = pd.to_datetime(roll_df["Date"]).dt.tz_localize(None)

for i, label in enumerate(loaded_labels):
    color     = COLORS[i % len(COLORS)]
    overall   = corr.loc[ticker, label]
    st.altair_chart(
        visualize_rolling_correlation(
            roll_df, label, color, overall,
            f"Rolling {effective_window}-Day Correlation: {ticker} vs {label}",
        ),
        use_container_width=True,
    )

# Summary table
if len(roll_df) > 0:
    with st.expander("Rolling correlation summary statistics"):
        summary = pd.DataFrame(
            {
                "Mean": roll_df[loaded_labels].mean(),
                "Std":  roll_df[loaded_labels].std(),
                "Min":  roll_df[loaded_labels].min(),
                "Max":  roll_df[loaded_labels].max(),
            }
        ).rename_axis("Factor")
        st.dataframe(summary.round(4), use_container_width=True)

st.divider()

# ── Section 3: Market Stress Decoupling (IHSG only) ───────────────────────────
IHSG_LABEL = next((lbl for lbl in loaded_labels if "JKSE" in lbl or lbl == "IHSG (^JKSE)"), None)

if IHSG_LABEL:
    st.subheader("Decoupling Analysis During Market Stress")
    st.write(
        f"Compares **{ticker}–{IHSG_LABEL}** correlation during **crash periods** "
        "(IHSG cumulative 20-day return < −10%) vs. normal periods."
    )

    ihsg_sym = PRESETS.get(IHSG_LABEL, "^JKSE")
    try:
        ihsg_raw = load_external_returns(ihsg_sym, start=start_date, end=end_date)
    except Exception:
        ihsg_raw = pd.Series(dtype=float)

    if len(ihsg_raw) > 20:
        ihsg_cum   = ihsg_raw.rolling(20).sum()
        crash_mask = ihsg_cum < -0.10

        combined_idx       = combined.copy()
        combined_idx.index = pd.to_datetime(combined_idx.index).normalize()

        crash_common  = combined_idx.index.intersection(
            pd.to_datetime(ihsg_cum[crash_mask].index).normalize()
        )
        normal_common = combined_idx.index.intersection(
            pd.to_datetime(ihsg_cum[~crash_mask].dropna().index).normalize()
        )

        c1, c2, c3 = st.columns(3)
        c1.metric("Crash days identified", len(crash_common))
        c2.metric("Normal days",           len(normal_common))

        if len(crash_common) > 10:
            crash_corr  = combined_idx.loc[crash_common,  [ticker, IHSG_LABEL]].corr().iloc[0, 1]
            normal_corr = combined_idx.loc[normal_common, [ticker, IHSG_LABEL]].corr().iloc[0, 1]
            diff        = crash_corr - normal_corr

            c3.metric("Crash − Normal correlation", f"{diff:+.4f}")
            ic1, ic2 = st.columns(2)
            ic1.metric(f"{ticker}–{IHSG_LABEL} (crash)",  f"{crash_corr:.4f}")
            ic2.metric(f"{ticker}–{IHSG_LABEL} (normal)", f"{normal_corr:.4f}")

            if diff < -0.1:
                st.success(
                    f"**{ticker} decouples during crashes** (−{abs(diff):.3f}). "
                    "Defensive characteristics — may outperform during market stress."
                )
            elif diff > 0.1:
                st.warning(
                    f"**Correlation rises during crashes** (+{diff:.3f}). "
                    "Higher systematic risk under stress — diversification benefit is reduced."
                )
            else:
                st.info(
                    f"**No significant decoupling** ({diff:+.3f}). "
                    "Correlation is stable across market regimes."
                )
        else:
            st.info("Fewer than 10 crash-period days detected — decoupling analysis not reliable.")
    else:
        st.info("Insufficient IHSG data for crash analysis — try a longer time horizon.")
