import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from streamlit_autorefresh import st_autorefresh

from models.fx_pricer import GarmanKohlhagen
from models.montercarlo import MonteCarloFX  # <-- check the actual filename (montercarlo vs montecarlo)
from models.binomial import BinomialFX

from data.marketdata import get_market_quotes


st.set_page_config(
    page_title="FX Option Pricer",
    page_icon="💱",
    layout="wide",
)

# ============================================================
# Dark theme styling — single file, no config.toml.
# This is CSS ONLY (for colors/fonts/spacing on native Streamlit
# widgets). No custom HTML content blocks are used anywhere below;
# all data is rendered with st.dataframe / st.metric / st.columns.
# ============================================================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&family=JetBrains+Mono:wght@500;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ---- Base app background ---- */
html, body, .stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
    background: radial-gradient(circle at top left, #151a26 0%, #0b0d13 55%, #08090d 100%) !important;
    color: #e8e9ee !important;
}
[data-testid="stHeader"] { background: transparent !important; }
[data-testid="stToolbar"] { background: transparent !important; }
[data-testid="stMainBlockContainer"] { background: transparent !important; padding-top: 2rem; }

p, span, label, li { color: #e6e6e6; }

/* ---- Title ---- */
h1 {
    font-weight: 800 !important;
    letter-spacing: -0.5px;
    background: linear-gradient(90deg, #60a5fa, #a78bfa 60%, #f472b6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-shadow: 0 0 30px rgba(96,165,250,0.15);
}
h2, h3 { color: #eef0f5 !important; font-weight: 700 !important; }
[data-testid="stCaptionContainer"] { color: #8b92a5 !important; }
hr { border-color: rgba(255,255,255,0.08) !important; }

/* ---- Widget labels ---- */
[data-testid="stWidgetLabel"] p {
    color: #9aa3b8 !important;
    font-weight: 600 !important;
    font-size: 0.82rem !important;
    text-transform: uppercase;
    letter-spacing: 0.04em;
}

/* ---- st.metric ---- */
[data-testid="stMetric"] {
    background: linear-gradient(160deg, #171c28 0%, #12151f 100%);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 14px;
    padding: 18px 20px;
    box-shadow: 0 4px 18px rgba(0,0,0,0.35);
    transition: border-color 0.2s ease, transform 0.2s ease;
}
[data-testid="stMetric"]:hover {
    border-color: rgba(96,165,250,0.4);
    transform: translateY(-1px);
}
[data-testid="stMetricLabel"] {
    color: #9aa3b8 !important;
    font-weight: 600 !important;
    text-transform: uppercase;
    font-size: 0.72rem !important;
    letter-spacing: 0.06em;
}
/* Fix: Streamlit truncates metric values with "..." on narrow columns.
   Allow the value to size to content instead of clipping. */
[data-testid="stMetricValue"] {
    color: #f4f6fb !important;
    font-weight: 800 !important;
    font-family: 'JetBrains Mono', monospace !important;
    white-space: normal !important;
    overflow: visible !important;
    text-overflow: unset !important;
    font-size: 1.6rem !important;
    line-height: 1.3 !important;
}
[data-testid="stMetricDelta"] {
    font-family: 'JetBrains Mono', monospace !important;
    font-weight: 700 !important;
}

/* ---- Dataframe (glide-data-grid canvas widget) ----
   This widget is drawn on a <canvas>, so regular CSS rules on its
   descendants don't work (or worse, break the rendering entirely).
   It reads a set of CSS custom properties instead — override those. */
[data-testid="stDataFrame"] {
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 12px;
    overflow: hidden;
    --gdg-bg-cell: #12151f;
    --gdg-bg-cell-medium: #171c28;
    --gdg-bg-header: #171c28;
    --gdg-bg-header-has-focus: #1f2937;
    --gdg-bg-header-hovered: #1f2937;
    --gdg-text-dark: #f4f6fb;
    --gdg-text-medium: #cfd3de;
    --gdg-text-light: #9aa3b8;
    --gdg-text-header: #9aa3b8;
    --gdg-text-header-selected: #f4f6fb;
    --gdg-bg-bubble: #171c28;
    --gdg-bg-bubble-selected: #232a3b;
    --gdg-bg-search-result: #3b82f6;
    --gdg-border-color: rgba(255,255,255,0.08);
    --gdg-horizontal-border-color: rgba(255,255,255,0.08);
    --gdg-drilldown-border: rgba(255,255,255,0.2);
    --gdg-link-color: #60a5fa;
    --gdg-accent-color: #3b82f6;
    --gdg-accent-fg: #ffffff;
    --gdg-accent-light: rgba(59,130,246,0.2);
}

/* ---- Inputs ---- */
div[data-baseweb="select"] > div,
input[type="number"],
input[type="text"] {
    background-color: #12151f !important;
    border: 1px solid rgba(255,255,255,0.12) !important;
    border-radius: 8px !important;
    color: #e8e9ee !important;
}
div[data-baseweb="select"] span { color: #e8e9ee !important; }

div[data-baseweb="popover"] ul,
div[data-baseweb="menu"] {
    background-color: #171c28 !important;
    color: #e8e9ee !important;
}
li[data-baseweb="menu-item"]:hover { background-color: #232a3b !important; }

button[data-testid="stNumberInputStepUp"],
button[data-testid="stNumberInputStepDown"] {
    background-color: #171c28 !important;
    border-color: rgba(255,255,255,0.12) !important;
    color: #e8e9ee !important;
}

[data-testid="stRadio"] label span { color: #e6e6e6 !important; }
[data-testid="stCheckbox"] label span { color: #e6e6e6 !important; }

/* ---- Buttons ---- */
.stButton > button {
    background: linear-gradient(90deg, #3b82f6, #8b5cf6);
    color: white;
    border: none;
    border-radius: 10px;
    font-weight: 700;
    padding: 0.6rem 1rem;
    box-shadow: 0 4px 16px rgba(59,130,246,0.35);
    transition: transform 0.15s ease, box-shadow 0.15s ease;
}
.stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 6px 20px rgba(139,92,246,0.45);
    color: white;
}

</style>
""", unsafe_allow_html=True)

# ============================================================
# Title
# ============================================================

st.title("💱 FX OPTION PRICER")
st.caption("European FX Option Pricing • Live Market Data powered by Twelve Data")

st.markdown("---")

# ============================================================
# FX dashboard
# ============================================================

quotes = {}  # <-- initialized BEFORE the try block to avoid a NameError further down if the API call fails

try:

    quotes = get_market_quotes()

    rows = []
    for pair, info in quotes.items():
        rows.append({
            "Pair": pair,
            "Spot": float(info["close"]),
            "Change (%)": float(info["percent_change"]),
            "High": float(info["high"]),
            "Low": float(info["low"]),
        })

    df = pd.DataFrame(rows)

    def color_change(v):
        if v > 0:
            return "color:#4ade80;font-weight:bold;"
        elif v < 0:
            return "color:#f87171;font-weight:bold;"
        return "color:#9aa3b8;"

    st.subheader("📈 Live FX Market")

    styled_df = (
        df.style
        .map(color_change, subset=["Change (%)"])
        .format({
            "Spot": "{:.5f}",
            "High": "{:.5f}",
            "Low": "{:.5f}",
            "Change (%)": "{:+.3f}%",
        })
    )

    st.dataframe(
        styled_df,
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("### Major Currency Pairs")

    c1, c2, c3, c4 = st.columns(4)

    pairs = ["EUR/USD", "GBP/USD", "USD/JPY", "USD/CHF"]

    for col, pair in zip([c1, c2, c3, c4], pairs):
        with col:
            st.metric(
                label=pair,
                value=f"{float(quotes[pair]['close']):.5f}",
                delta=f"{float(quotes[pair]['percent_change']):+.3f}%",
            )

except Exception as e:

    st.error(f"Unable to load market data: {e}")

st.markdown("---")

# ============================================================
# Auto-refresh (every 30 seconds)
# ============================================================

st_autorefresh(
    interval=30000,
    key="fx_refresh"
)

# ============================================================
# Columns
# ============================================================

left, right = st.columns([1, 1])


with left:

    st.subheader("Pricing Parameters")

    option_type = st.selectbox(
        "Option Type",
        ["Call", "Put"]
    )

    model_type = st.selectbox(
        "Pricing Model",
        [
            "Garman-Kohlhagen",
            "Monte Carlo",
            "Binomial"
        ]
    )

    st.markdown("### Market Data")

    # ============================================================
    # Data source selection
    # ============================================================

    data_source = st.radio(
        "Data Source",
        [
            "Manual",
            "Live Market"
        ]
    )

    pair = st.selectbox(
        "Currency Pair",
        [
            "EUR/USD",
            "GBP/USD",
            "USD/JPY",
            "USD/CHF"
        ]
    )

    # <-- safety net: if the API call failed above, `quotes` may be empty.
    # Automatically fall back to Manual mode to avoid a crash.
    if data_source == "Live Market" and pair not in quotes:
        st.warning("Live data unavailable for this pair, switching to Manual input.")
        data_source = "Manual"

    if data_source == "Manual":

        S = st.number_input(
            "Spot Price",
            value=1.1200,
            format="%.5f"
        )

    else:

        live_spot = float(quotes[pair]["close"])

        st.metric(
            label="Live Spot",
            value=f"{live_spot:.5f}"
        )

        S = st.number_input(
            "Spot Price",
            value=live_spot,
            format="%.5f"
        )

    K = st.number_input(
        "Strike",
        value=1.1500,
        format="%.4f"
    )

    T = st.number_input(
        "Maturity (Years)",
        value=1.0
    )

    rd = st.number_input(
        "Domestic Rate",
        value=0.03,
        format="%.4f"
    )

    rf = st.number_input(
        "Foreign Rate",
        value=0.01,
        format="%.4f"
    )

    sigma = st.number_input(
        "Volatility",
        value=0.12,
        format="%.4f"
    )

    if model_type == "Monte Carlo":

        simulations = st.number_input(
            "Number of Simulations",
            min_value=1000,
            value=100000,
            step=1000
        )

    if model_type == "Binomial":

        steps = st.number_input(
            "Number of Steps",
            min_value=10,
            value=500,
            step=10
        )

    calculate = st.button(
        "Calculate Price",
        use_container_width=True
    )

    # <-- Without this, clicking the "Theta decay" checkbox further down
    # would make the whole results section disappear: st.button() only
    # stays True for a single rerun. We store the click in session_state
    # so the results keep showing afterwards.
    if calculate:
        st.session_state["has_calculated"] = True

    calculate = st.session_state.get("has_calculated", False)

with right:

    st.subheader("Results")

    if calculate:

        if model_type == "Garman-Kohlhagen":

            model = GarmanKohlhagen(
                S,
                K,
                T,
                rd,
                rf,
                sigma
            )

        elif model_type == "Monte Carlo":

            model = MonteCarloFX(
                S,
                K,
                T,
                rd,
                rf,
                sigma,
                simulations
            )

        elif model_type == "Binomial":

            model = BinomialFX(
                S,
                K,
                T,
                rd,
                rf,
                sigma,
                steps
            )

        if option_type == "Call":

            price = model.call_price()

        else:

            price = model.put_price()

        st.metric(
            label="Option Price",
            value=f"{price:.6f}"
        )

# ============================================================
# Greeks — displayed FULL WIDTH (outside the left/right columns)
# to have enough room and avoid truncated values.
# Only available for Garman-Kohlhagen.
# ============================================================

if calculate and model_type == "Garman-Kohlhagen":

    st.markdown("---")
    st.subheader("Greeks")

    delta, gamma, vega, theta, rho = model.greeks(option_type)

    greeks_data = [
        ("Delta", delta, "Price sensitivity to a +1 unit move in spot."),
        ("Gamma", gamma, "Rate of change of Delta as spot moves."),
        ("Vega", vega, "Price sensitivity to a +1 point move in volatility."),
        ("Theta", theta, "Value lost per day that passes (time decay)."),
        ("Rho", rho, "Price sensitivity to a +1 point move in the domestic rate."),
    ]

    g_cols = st.columns(5)

    for col, (name, val, desc) in zip(g_cols, greeks_data):
        with col:
            st.metric(
                label=name,
                value=f"{val:.4f}",
                help=desc,
            )

# ============================================================
# Payoff — also displayed full width
# ============================================================

if calculate:

    st.markdown("---")

    st.subheader("Payoff & Option Value")

    S_range = np.linspace(
        0.5 * S,
        1.5 * S,
        200
    )

    if option_type == "Call":

        payoff = np.maximum(
            S_range - K,
            0
        )
        breakeven = K + price

    else:

        payoff = np.maximum(
            K - S_range,
            0
        )
        breakeven = K - price

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=S_range,
            y=payoff,
            mode="lines",
            name="Payoff at maturity",
            line=dict(color="#60a5fa", width=2.5),
            hovertemplate="Spot: %{x:.4f}<br>Payoff: %{y:.5f}<extra></extra>",
        )
    )

    # --- Theoretical value today (before maturity) ---
    # Garman-Kohlhagen only: closed-form formula, so cheap to recompute
    # over 200 points. For Monte Carlo / Binomial, the whole simulation
    # would need to rerun at every point -> too slow for an interactive
    # chart, so it's omitted for those models.
    if model_type == "Garman-Kohlhagen":

        value_today = []
        for s in S_range:
            m = GarmanKohlhagen(s, K, T, rd, rf, sigma)
            value_today.append(m.call_price() if option_type == "Call" else m.put_price())

        fig.add_trace(
            go.Scatter(
                x=S_range,
                y=value_today,
                mode="lines",
                name=f"Value today (T={T:.2f} yr)",
                line=dict(color="#fb923c", width=2, dash="dot"),
                hovertemplate="Spot: %{x:.4f}<br>Value: %{y:.5f}<extra></extra>",
            )
        )

        # --- Interactive time decay (Theta decay) ---
        show_decay = st.checkbox(
            "📉 Show time decay (Theta decay)",
            value=False,
            help="Overlays the value curve at several maturity horizons to see "
                 "how the option loses time value as expiry approaches.",
        )

        if show_decay and T > 0:
            decay_fractions = [0.75, 0.5, 0.25, 0.05]
            decay_colors = ["#4ade80", "#a78bfa", "#f472b6", "#facc15"]

            for frac, color in zip(decay_fractions, decay_colors):
                t_step = T * frac
                if t_step <= 0.001:
                    continue
                values_at_t = []
                for s in S_range:
                    m = GarmanKohlhagen(s, K, t_step, rd, rf, sigma)
                    values_at_t.append(m.call_price() if option_type == "Call" else m.put_price())

                fig.add_trace(
                    go.Scatter(
                        x=S_range,
                        y=values_at_t,
                        mode="lines",
                        name=f"T = {t_step:.2f} yr ({frac:.0%} remaining)",
                        line=dict(color=color, width=1.3, dash="dash"),
                        opacity=0.75,
                        hovertemplate="Spot: %{x:.4f}<br>Value: %{y:.5f}<extra></extra>",
                    )
                )

    else:
        st.caption(
            "ℹ️ The theoretical value curve before maturity is only shown for "
            "Garman-Kohlhagen (fast to compute). For Monte Carlo / Binomial, only "
            "the payoff at maturity is shown here."
        )

    # --- Useful reference lines for a trader ---
    fig.add_vline(
        x=K,
        line_dash="dash",
        line_color="#8b92a5",
        annotation_text="Strike (K)",
        annotation_position="top",
        annotation_font_color="#f4f6fb",
    )

    fig.add_vline(
        x=S,
        line_dash="dot",
        line_color="#4ade80",
        annotation_text=f"Current spot ({S:.4f})",
        annotation_position="bottom",
        annotation_font_color="#f4f6fb",
    )

    fig.add_vline(
        x=breakeven,
        line_dash="dashdot",
        line_color="#f87171",
        annotation_text=f"Breakeven ({breakeven:.4f})",
        annotation_position="top",
        annotation_font_color="#f4f6fb",
    )

    fig.update_layout(
        title=dict(text="Option Payoff & Value", font=dict(color="#f4f6fb", size=18)),
        xaxis_title="Spot Price",
        yaxis_title="Value / Payoff",
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.02)",
        font=dict(color="#f4f6fb"),
        height=550,
        hovermode="x unified",
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
            font=dict(color="#f4f6fb", size=12),
            bgcolor="rgba(0,0,0,0)",
        ),
    )
    fig.update_xaxes(
        gridcolor="rgba(255,255,255,0.06)",
        title_font=dict(color="#e6e6e6"),
        tickfont=dict(color="#cfd3de"),
    )
    fig.update_yaxes(
        gridcolor="rgba(255,255,255,0.06)",
        title_font=dict(color="#e6e6e6"),
        tickfont=dict(color="#cfd3de"),
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # ============================================================
    # Greeks sensitivity to Spot — useful for anticipating hedging
    # ============================================================
    if model_type == "Garman-Kohlhagen":

        st.subheader("Greeks Sensitivity to Spot")
        st.caption("How Delta and Gamma evolve as the market moves — useful for anticipating hedge rebalancing.")

        deltas, gammas = [], []
        for s in S_range:
            m = GarmanKohlhagen(s, K, T, rd, rf, sigma)
            deltas.append(m.delta(option_type))
            gammas.append(m.gamma())

        fig_greeks = go.Figure()

        fig_greeks.add_trace(
            go.Scatter(
                x=S_range, y=deltas, mode="lines", name="Delta",
                line=dict(color="#60a5fa", width=2.5), yaxis="y1",
                hovertemplate="Spot: %{x:.4f}<br>Delta: %{y:.4f}<extra></extra>",
            )
        )
        fig_greeks.add_trace(
            go.Scatter(
                x=S_range, y=gammas, mode="lines", name="Gamma",
                line=dict(color="#f87171", width=2, dash="dot"), yaxis="y2",
                hovertemplate="Spot: %{x:.4f}<br>Gamma: %{y:.4f}<extra></extra>",
            )
        )

        fig_greeks.add_vline(x=S, line_dash="dot", line_color="#4ade80")
        fig_greeks.add_vline(x=K, line_dash="dash", line_color="#8b92a5")

        fig_greeks.update_layout(
            title=dict(text="Delta & Gamma vs Spot", font=dict(color="#f4f6fb", size=18)),
            xaxis_title="Spot Price",
            yaxis=dict(title="Delta", side="left", title_font=dict(color="#e6e6e6"), tickfont=dict(color="#cfd3de")),
            yaxis2=dict(title="Gamma", overlaying="y", side="right", title_font=dict(color="#e6e6e6"), tickfont=dict(color="#cfd3de")),
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(255,255,255,0.02)",
            font=dict(color="#f4f6fb"),
            height=450,
            hovermode="x unified",
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                font=dict(color="#f4f6fb", size=12),
                bgcolor="rgba(0,0,0,0)",
            ),
        )
        fig_greeks.update_xaxes(
            gridcolor="rgba(255,255,255,0.06)",
            title_font=dict(color="#e6e6e6"),
            tickfont=dict(color="#cfd3de"),
        )
        fig_greeks.update_yaxes(gridcolor="rgba(255,255,255,0.06)")

        st.plotly_chart(fig_greeks, use_container_width=True)