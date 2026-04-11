import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import altair as alt
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.stats.diagnostic import breaks_cusumolsresid
import statsmodels.api as sm

def visualize_price_chart(df: pd.DataFrame):
    price_df = df[["Open", "High", "Low", "Close"]].reset_index()
    price_df["Date"] = pd.to_datetime(price_df["Date"]).dt.tz_localize(None)
    price_df["Color"] = np.where(price_df["Close"] >= price_df["Open"], "green", "red")

    base = alt.Chart(price_df).encode(
        alt.X("Date:T", title=None),
        color=alt.Color("Color:N", scale=None),
        tooltip=[
            alt.Tooltip("Date:T", format="%Y-%m-%d"),
            alt.Tooltip("Open:Q", format=",.2f"),
            alt.Tooltip("High:Q", format=",.2f"),
            alt.Tooltip("Low:Q", format=",.2f"),
            alt.Tooltip("Close:Q", format=",.2f"),
        ],
    )

    rule = base.mark_rule().encode(
        alt.Y("Low:Q", scale=alt.Scale(zero=False), title="Price"),
        alt.Y2("High:Q"),
    )

    bar = base.mark_bar(size=5).encode(
        alt.Y("Open:Q"),
        alt.Y2("Close:Q"),
    )

    price_chart = (rule + bar).properties(height=280).interactive()

    vol_df = df[["Volume"]].reset_index()[["Date", "Volume"]]
    vol_df["Date"] = pd.to_datetime(vol_df["Date"]).dt.tz_localize(None)

    vol_chart = (
        alt.Chart(vol_df)
        .mark_bar(color="lightsteelblue", opacity=0.6)
        .encode(
            alt.X("Date:T", title="Date"),
            alt.Y("Volume:Q", title="Volume"),
            tooltip=[alt.Tooltip("Date:T", format="%Y-%m-%d"), alt.Tooltip("Volume:Q", format=",")],
        )
        .properties(height=100)
        .interactive()
    )

    return alt.vconcat(price_chart, vol_chart).resolve_scale(x="shared")

def visualize_log_return_distribution(log_returns:pd.Series, mu:float, std:float):
    # Create histogram data
    hist_df = pd.DataFrame({"Log Return": log_returns})
    
    # Create normal PDF line data
    x = np.linspace(log_returns.min(), log_returns.max(), 300)
    pdf_df = pd.DataFrame({
        "Log Return": x,
        "Density": stats.norm.pdf(x, mu, std)
    })
    
    # Histogram
    histogram = (
        alt.Chart(hist_df)
        .transform_bin(
            "binned_return",
            field="Log Return",
            bin=alt.Bin(maxbins=50)
        )
        .transform_aggregate(
            count="count()",
            groupby=["binned_return", "binned_return_end"]
        )
        .transform_calculate(
            density=f"datum.count / ({len(log_returns)} * (datum.binned_return_end - datum.binned_return))"
        )
        .mark_bar(color="skyblue", opacity=0.75)
        .encode(
            alt.X("binned_return:Q", title="Log Return", bin="binned"),
            alt.X2("binned_return_end:Q"),
            alt.Y("density:Q", title="Density"),
            tooltip=[
                alt.Tooltip("binned_return:Q", title="Bin Start", format=".4f"),
                alt.Tooltip("binned_return_end:Q", title="Bin End", format=".4f"),
                alt.Tooltip("density:Q", title="Density", format=".4f")
            ]
        )
    )
    
    # Normal PDF line
    pdf_line = (
        alt.Chart(pdf_df)
        .mark_line(color="red", strokeWidth=2)
        .encode(
            alt.X("Log Return:Q"),
            alt.Y("Density:Q"),
            tooltip=[
                alt.Tooltip("Log Return:Q", format=".4f"),
                alt.Tooltip("Density:Q", format=".4f")
            ]
        )
    )
    
    chart = (
        (histogram + pdf_line)
        .properties(
            height=300,
            title="Distribution of Log Returns"
        )
        .interactive()
    )
    
    return chart

def visualize_qq_plot(log_returns:pd.Series):
    (osm, osr), (slope, intercept, _) = stats.probplot(log_returns, dist="norm")
    
    # Create DataFrame for scatter points
    qq_df = pd.DataFrame({
        "Theoretical Quantiles": osm,
        "Sample Quantiles": osr
    })
    
    # Create DataFrame for reference line
    line_df = pd.DataFrame({
        "Theoretical Quantiles": [osm.min(), osm.max()],
        "Sample Quantiles": [slope * osm.min() + intercept, slope * osm.max() + intercept]
    })
    
    # Scatter plot for sample quantiles
    scatter = (
        alt.Chart(qq_df)
        .mark_circle(color="steelblue", size=30, opacity=0.6)
        .encode(
            alt.X("Theoretical Quantiles:Q", title="Theoretical Quantiles"),
            alt.Y("Sample Quantiles:Q", title="Sample Quantiles"),
            tooltip=[
                alt.Tooltip("Theoretical Quantiles:Q", format=".4f"),
                alt.Tooltip("Sample Quantiles:Q", format=".4f")
            ]
        )
    )
    
    # Reference line
    line = (
        alt.Chart(line_df)
        .mark_line(color="red", strokeWidth=2)
        .encode(
            alt.X("Theoretical Quantiles:Q"),
            alt.Y("Sample Quantiles:Q")
        )
    )
    
    chart = (
        (scatter + line)
        .properties(
            height=300,
            title="Q-Q Plot of Log Returns"
        )
        .interactive()
    )
    
    return chart

def visualize_rolling_skewness_kurtosis(roll_df:pd.DataFrame, roll_window:int):
    zero_rule = (
        alt.Chart(pd.DataFrame({"y": [0]}))
        .mark_rule(strokeDash=[4, 4], opacity=0.6)
        .encode(y="y:Q")
    )

    skew_chart = (
        alt.Chart(roll_df)
        .mark_line(color="dodgerblue")
        .encode(
            alt.X("Date:T", title=None),
            alt.Y("Skewness:Q"),
            tooltip=[alt.Tooltip("Date:T", format="%Y-%m-%d"), alt.Tooltip("Skewness:Q", format=".3f")],
        )
        .properties(height=180, title=f"Rolling {roll_window}-Day Skewness")
    )

    kurt_chart = (
        alt.Chart(roll_df)
        .mark_line(color="crimson")
        .encode(
            alt.X("Date:T", title="Date"),
            alt.Y("Kurtosis:Q", title="Excess Kurtosis"),
            tooltip=[alt.Tooltip("Date:T", format="%Y-%m-%d"), alt.Tooltip("Kurtosis:Q", format=".3f")],
        )
        .properties(height=180, title=f"Rolling {roll_window}-Day Excess Kurtosis")
    )

    return alt.vconcat(skew_chart + zero_rule, kurt_chart + zero_rule).resolve_scale(x="shared")

def visualize_cusum_chart(cusum_df:pd.DataFrame):
    # Plot 1: CUSUM with boundaries
    cusum_line = alt.Chart(cusum_df).mark_line(color='steelblue', strokeWidth=1.5).encode(
        x=alt.X('Date:T', title='Date'),
        y=alt.Y('CUSUM:Q', title='CUSUM Statistic'),
        tooltip=['Date:T', alt.Tooltip('CUSUM:Q', format='.4f')]
    )

    upper_line = alt.Chart(cusum_df).mark_line(color='red', strokeDash=[5, 5], strokeWidth=1.5).encode(
        x='Date:T',
        y='Upper Bound:Q'
    )

    lower_line = alt.Chart(cusum_df).mark_line(color='red', strokeDash=[5, 5], strokeWidth=1.5).encode(
        x='Date:T',
        y='Lower Bound:Q'
    )

    band = alt.Chart(cusum_df).mark_area(opacity=0.1, color='green').encode(
        x='Date:T',
        y='Lower Bound:Q',
        y2='Upper Bound:Q'
    )

    zero_line = alt.Chart(pd.DataFrame({'y': [0]})).mark_rule(color='gray', opacity=0.5).encode(
        y='y:Q'
    )

    cusum_chart = alt.layer(band, zero_line, upper_line, lower_line, cusum_line).properties(
        title='CUSUM Test for Structural Breaks in Returns',
        width='container',
        height=300
    )

    return cusum_chart

def visualize_returns_with_breaks(breaks_detected:list, cusum_df:pd.DataFrame):
    returns_line = alt.Chart(cusum_df).mark_line(color='steelblue', strokeWidth=0.8, opacity=0.7).encode(
        x=alt.X('Date:T', title='Date'),
        y=alt.Y('Log Returns:Q', title='Log Return'),
        tooltip=['Date:T', alt.Tooltip('Log Returns:Q', format='.4f')]
    )

    returns_zero = alt.Chart(pd.DataFrame({'y': [0]})).mark_rule(color='gray', opacity=0.5).encode(
        y='y:Q'
    )

    # Add break regions if detected
    if len(breaks_detected) > 0:
        # Find contiguous break regions
        break_starts = [breaks_detected[0]]
        break_ends = []
        for i in range(1, len(breaks_detected)):
            if breaks_detected[i] - breaks_detected[i-1] > 1:
                break_ends.append(breaks_detected[i-1])
                break_starts.append(breaks_detected[i])
        break_ends.append(breaks_detected[-1])
        
        break_regions = pd.DataFrame({
            'start': [cusum_df['Log Returns'].index[s] for s in break_starts],
            'end': [cusum_df['Log Returns'].index[e] for e in break_ends]
        })
        
        break_rect = alt.Chart(break_regions).mark_rect(opacity=0.3, color='red').encode(
            x='start:T',
            x2='end:T'
        )
        
        returns_chart = alt.layer(break_rect, returns_zero, returns_line).properties(
            title='Log Returns with Structural Break Periods Highlighted',
            width='container',
            height=300
        )
    else:
        returns_chart = alt.layer(returns_zero, returns_line).properties(
            title='Log Returns with Structural Break Periods Highlighted',
            width='container',
            height=300
        )
    
    return returns_chart

def visualize_agg_returns_boxplot(
    agg_returns: pd.DataFrame,
    key_column: str = 'DayOfWeek',
    label_column: str = 'DayLabel',
    labels: list = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'],
    key_offset: int = 0,
):
    agg_returns = agg_returns.copy()
    agg_returns[label_column] = agg_returns[key_column].map(lambda d: labels[d - key_offset])
    agg_boxplot = alt.Chart(agg_returns).mark_boxplot().encode(
        x=alt.X(f'{label_column}:N', title=key_column, sort=labels[:5]),
        y=alt.Y('Log Return:Q', title='Log Return'),
        tooltip=[f'{label_column}:N']
    ).properties(
        title=f'{key_column} Effect on Log Returns (Boxplot)',
        height=300
    )
    return agg_boxplot

def visualize_pre_post_earnings_behavior(earnings_df: pd.DataFrame):
    # Create vertical line data for earnings release
    vline_df = pd.DataFrame({'Days from Earnings': [0], 'Mean Log Return': [np.nan]})

    earnings_chart = (
        alt.Chart(earnings_df)
        .mark_line(point=alt.OverlayMarkDef(color='navy', filled=True, size=50))
        .encode(
            x=alt.X('Days from Earnings:O', title='Days from Earnings', scale=alt.Scale(zero=False)),
            y=alt.Y('Mean Log Return:Q', title='Log Return'),
            tooltip=['Days from Earnings', alt.Tooltip('Mean Log Return', format=".5f")]
        )
        .properties(
            width=600,
            height=350,
            title='Average Log Return Around Earnings Releases'
        )
    ) + (
        alt.Chart(vline_df)
        .mark_rule(color='red', strokeDash=[6, 3])
        .encode(
            x='Days from Earnings:O'
        )
        .properties()
    )

    return earnings_chart


def visualize_correlation_heatmap(corr: pd.DataFrame) -> alt.Chart:
    """Altair heatmap of a correlation matrix with value annotations."""
    corr_long = (
        corr.reset_index()
        .melt(id_vars="index", var_name="Variable 2", value_name="Correlation")
        .rename(columns={"index": "Variable 1"})
    )

    heatmap = (
        alt.Chart(corr_long)
        .mark_rect()
        .encode(
            x=alt.X("Variable 1:N", title=None),
            y=alt.Y("Variable 2:N", title=None),
            color=alt.Color(
                "Correlation:Q",
                scale=alt.Scale(scheme="redblue", domain=[-1, 1]),
                legend=alt.Legend(title="Correlation"),
            ),
            tooltip=[
                "Variable 1",
                "Variable 2",
                alt.Tooltip("Correlation:Q", format=".4f"),
            ],
        )
    )

    text = (
        alt.Chart(corr_long)
        .mark_text(fontSize=14, fontWeight="bold")
        .encode(
            x="Variable 1:N",
            y="Variable 2:N",
            text=alt.Text("Correlation:Q", format=".3f"),
            color=alt.condition(
                "abs(datum.Correlation) > 0.5",
                alt.value("white"),
                alt.value("black"),
            ),
        )
    )

    return (heatmap + text).properties(height=220, title="Correlation Heatmap")


def visualize_rolling_correlation(
    roll_df: pd.DataFrame,
    col: str,
    color: str,
    overall_val: float,
    title: str,
) -> alt.Chart:
    """
    Altair area + line chart for a rolling correlation series.

    Parameters
    ----------
    roll_df      : DataFrame with 'Date' column and one correlation column named `col`.
    col          : Name of the correlation column in roll_df (e.g. 'IHSG').
    color        : Line / area colour (e.g. 'steelblue').
    overall_val  : Static overall correlation value shown as a dashed reference line.
    title        : Chart title.
    """
    zero_rule = (
        alt.Chart(pd.DataFrame({"y": [0]}))
        .mark_rule(color="gray", strokeDash=[4, 4], opacity=0.6)
        .encode(y="y:Q")
    )

    line = (
        alt.Chart(roll_df)
        .mark_line(color=color)
        .encode(
            x=alt.X("Date:T", title=None),
            y=alt.Y(f"{col}:Q", scale=alt.Scale(domain=[-1, 1]), title="Correlation"),
            tooltip=[
                alt.Tooltip("Date:T", format="%Y-%m-%d"),
                alt.Tooltip(f"{col}:Q", format=".3f"),
            ],
        )
        .properties(height=200, title=title)
    )

    area = (
        alt.Chart(roll_df)
        .mark_area(color=color, opacity=0.15)
        .encode(x="Date:T", y=f"{col}:Q", y2=alt.value(0))
    )

    if not pd.isna(overall_val):
        overall_rule = (
            alt.Chart(pd.DataFrame({"y": [overall_val]}))
            .mark_rule(color=color, strokeDash=[6, 3], opacity=0.7)
            .encode(y="y:Q")
        )
        return area + zero_rule + overall_rule + line

    return area + zero_rule + line