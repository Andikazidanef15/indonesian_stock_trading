import streamlit as st

st.set_page_config(
    page_title="Stock EDA Dashboard",
    page_icon=":chart_with_upwards_trend:",
    layout="wide",
)

pages = st.navigation([
    st.Page("pages/1_Daily_Stock.py",           title="Daily Stock",          icon=":material/home:"),
    st.Page("pages/2_Return_Distribution.py", title="Return Distribution",  icon=":material/bar_chart:"),
    st.Page("pages/3_Structural_Breaks.py",   title="Structural Breaks",    icon=":material/timeline:"),
    st.Page("pages/4_Seasonality.py",         title="Seasonality",          icon=":material/calendar_month:"),
    st.Page("pages/5_Volatility_Structure.py", title="Volatility Structure",  icon=":material/waves:"),
    st.Page("pages/6_External_Correlation.py", title="External Correlation",  icon=":material/trending_up:"),
    st.Page("pages/7_Strategy_Recommendation.py", title="Strategy Recommendation", icon=":material/recommend:"),
])

pages.run()
