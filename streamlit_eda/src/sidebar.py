import streamlit as st
from src.constant import STOCKS, HORIZON_MAP


def render_sidebar() -> tuple[str, str]:
    """
    Render shared sidebar controls on every page.

    Widget-owned session state keys are cleared by Streamlit on page navigation,
    so we keep a separate set of *backing store* keys (prefixed with '_') that are
    set manually and therefore survive navigation.  The widgets read their initial
    display value from the backing store via `index=`, and `on_change` callbacks
    keep the two in sync whenever the user makes a change.

    Returns (ticker, period_string).
    """
    # Backing store — manually managed, never owned by a widget → survives navigation
    st.session_state.setdefault("_ticker",  STOCKS[0])
    st.session_state.setdefault("_horizon", "1 Year")
    st.session_state.setdefault("_custom",  "")

    def _save_ticker():
        st.session_state["_ticker"] = st.session_state["_w_ticker"]

    def _save_horizon():
        st.session_state["_horizon"] = st.session_state["_w_horizon"]

    def _save_custom():
        st.session_state["_custom"] = st.session_state["_w_custom"]

    with st.sidebar:
        st.header("Settings")

        # index= reads from backing store (NOT from the widget key) so there is
        # no circular dependency.  If _w_ticker was cleared on navigation,
        # index= re-initialises the widget to the correct option.
        st.selectbox(
            "Stock ticker",
            options=STOCKS,
            index=STOCKS.index(st.session_state["_ticker"])
                  if st.session_state["_ticker"] in STOCKS else 0,
            key="_w_ticker",
            on_change=_save_ticker,
            help="Select an IDX stock to analyze.",
        )

        st.text_input(
            "Or type a custom ticker",
            value=st.session_state["_custom"],
            placeholder="e.g. AAPL, NVDA",
            key="_w_custom",
            on_change=_save_custom,
        )

        st.radio(
            "Time horizon",
            options=list(HORIZON_MAP.keys()),
            index=list(HORIZON_MAP.keys()).index(st.session_state["_horizon"]),
            key="_w_horizon",
            on_change=_save_horizon,
        )

    # Derive final ticker: custom input overrides the selectbox
    ticker = (
        st.session_state["_custom"].upper().strip()
        if st.session_state["_custom"].strip()
        else st.session_state["_ticker"]
    )
    period = HORIZON_MAP[st.session_state["_horizon"]]
    return ticker, period
