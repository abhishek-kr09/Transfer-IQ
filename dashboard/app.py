# ============================================================
# FILE: dashboard/app.py
# Week 8 — Final UI with navbar, search, player detail page
# ============================================================

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import joblib

st.set_page_config(
    page_title="TransferIQ",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ------------------------------------------------
# Custom CSS — navbar, footer, search card
# ------------------------------------------------
st.markdown("""
<style>
:root {
    --tiq-blue: #0f3f89;
    --tiq-blue-2: #1c5fc5;
    --tiq-cyan: #2ea7d8;
    --tiq-green: #17704b;
    --tiq-ink-0: #050914;
    --tiq-ink-1: #0a1224;
    --tiq-ink-2: #111c32;
    --tiq-text: #e7eefc;
    --tiq-muted: #9aa9c4;
    --tiq-border: rgba(100, 136, 204, 0.28);
    --tiq-shadow: 0 18px 38px rgba(2, 7, 19, 0.55);
}

#MainMenu, footer, header { visibility: hidden; }

.stApp {
    background:
        radial-gradient(1050px 560px at 4% -12%, rgba(28, 95, 197, 0.28) 0%, transparent 56%),
        radial-gradient(900px 380px at 100% 0%, rgba(23, 112, 75, 0.22) 0%, transparent 58%),
        linear-gradient(180deg, var(--tiq-ink-0) 0%, var(--tiq-ink-1) 56%, #0d182c 100%);
    color: var(--tiq-text);
}

.block-container {
    padding-top: 0.2rem;
    padding-bottom: 3.2rem;
    max-width: 1100px;
}

.navbar {
    background: linear-gradient(120deg, #113a7d 0%, #1a56b2 52%, #166a48 100%);
    box-shadow: 0 12px 24px rgba(3, 10, 24, 0.45);
    padding: 12px 28px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    border-bottom: 1px solid rgba(203, 220, 255, 0.22);
    position: sticky;
    top: 0;
    z-index: 100;
}

.navbar-brand {
    color: white;
    font-size: 20px;
    font-weight: 800;
    letter-spacing: 0.4px;
}

.navbar-sub {
    color: rgba(232, 241, 255, 0.9);
    font-size: 12px;
    font-weight: 500;
    letter-spacing: 0.4px;
}

.search-title {
    color: #edf4ff;
    margin-bottom: 0.35rem;
    font-size: 1.9rem;
}

/* dark selectbox surface and text */
div[data-baseweb="select"] > div:first-child {
    background: linear-gradient(145deg, #10192c 0%, #0b1322 100%) !important;
    border: 1px solid rgba(120, 154, 224, 0.34) !important;
    border-radius: 10px !important;
    min-height: 44px !important;
    box-shadow: inset 0 1px 0 rgba(190, 210, 255, 0.08);
}

div[data-baseweb="select"] span,
div[data-baseweb="select"] input {
    color: #e7eeff !important;
}

div[data-baseweb="select"] svg {
    color: #b8cdf9 !important;
}

.metric-card {
    border: 1px solid var(--tiq-border);
    border-left: 5px solid #2a7cf3;
    border-radius: 10px;
    padding: 14px 16px;
    margin-bottom: 12px;
    background: linear-gradient(145deg, rgba(16, 28, 49, 0.95), rgba(12, 21, 39, 0.95));
    box-shadow: var(--tiq-shadow);
}

.metric-label {
    font-size: 12px;
    color: var(--tiq-muted);
    margin-bottom: 6px;
    font-weight: 600;
    letter-spacing: 0.3px;
    text-transform: uppercase;
}

.metric-value {
    font-size: 30px;
    font-weight: 800;
    color: #dbe8ff;
    line-height: 1.08;
}

.metric-sub {
    font-size: 12px;
    color: #8fa1bf;
    margin-top: 7px;
}

.player-header {
    background:
        radial-gradient(420px 220px at 88% 18%, rgba(255, 255, 255, 0.16) 0%, transparent 62%),
        linear-gradient(128deg, #0f3f89 0%, #1c5fc5 44%, #17704b 100%);
    color: white;
    padding: 20px 22px;
    border-radius: 12px;
    margin-bottom: 14px;
    box-shadow: 0 14px 34px rgba(8, 34, 73, 0.24);
}

.player-name-big {
    font-size: 27px;
    font-weight: 800;
    margin-bottom: 6px;
    letter-spacing: 0.3px;
}

.player-meta {
    font-size: 13px;
    opacity: 0.92;
    letter-spacing: 0.25px;
}

div[data-testid="stButton"] > button {
    border-radius: 12px;
    border-width: 1px;
    font-weight: 700;
    min-height: 41px;
    padding-top: 0.3rem;
    padding-bottom: 0.3rem;
    font-size: 0.95rem;
    transition: all 0.2s ease;
}

/* force season active button to be blue — override Streamlit's red primary */
div[data-testid="stButton"] > button[kind="primary"] {
    background: linear-gradient(120deg, #1449a3 0%, #1c62cd 64%, #2190cf 100%) !important;
    border-color: rgba(155, 191, 255, 0.36) !important;
    color: #f4f8ff !important;
    box-shadow: 0 10px 20px rgba(10, 40, 95, 0.45);
}

div[data-testid="stButton"] > button[kind="primary"]:hover {
    background: linear-gradient(120deg, #0f3d8e 0%, #1955b1 64%, #1f7eb6 100%) !important;
    border-color: rgba(155, 191, 255, 0.5) !important;
    transform: translateY(-1px);
}

div[data-testid="stButton"] > button[kind="secondary"] {
    background: linear-gradient(140deg, #0e1627 0%, #0a1220 100%) !important;
    border-color: rgba(119, 145, 198, 0.34) !important;
    color: #dce8ff !important;
    box-shadow: none;
}

div[data-testid="stButton"] > button[kind="secondary"]:hover {
    background: linear-gradient(140deg, #121d34 0%, #0d1629 100%) !important;
    border-color: rgba(143, 174, 238, 0.48) !important;
    color: #f0f6ff !important;
}

.footer {
    position: fixed;
    left: 0;
    bottom: 0;
    width: 100%;
    background: linear-gradient(100deg, #0f3f89 0%, #1a56b3 50%, #17704b 100%);
    color: rgba(239, 245, 255, 0.9);
    text-align: center;
    padding: 8px 10px;
    font-size: 11px;
    z-index: 999;
    border-top: 1px solid rgba(255, 255, 255, 0.22);
}

.home-wrap {
    min-height: 70vh;
    display: flex;
    align-items: center;
    justify-content: center;
}

@media (max-width: 900px) {
    .navbar {
        padding: 14px 18px;
        flex-direction: column;
        gap: 3px;
        align-items: flex-start;
    }

    .navbar-brand {
        font-size: 20px;
    }

    .navbar-sub {
        font-size: 12px;
    }

    div[data-baseweb="select"] > div:first-child {
        min-height: 42px !important;
    }

    .player-header {
        padding: 16px 14px;
        border-radius: 12px;
    }

    .player-name-big {
        font-size: 22px;
    }

    .player-meta {
        font-size: 12px;
        line-height: 1.5;
    }

    .metric-card {
        padding: 12px 12px;
    }

    .metric-value {
        font-size: 24px;
    }

    .footer {
        position: static;
        margin-top: 22px;
    }

    .block-container {
        padding-bottom: 1rem;
    }
}

@media (max-width: 600px) {
    .block-container {
        padding-left: 0.8rem;
        padding-right: 0.8rem;
    }

    .navbar {
        padding: 12px 14px;
    }

    .navbar-brand {
        font-size: 17px;
    }

    .navbar-sub {
        font-size: 11px;
    }

    .metric-value {
        font-size: 22px;
    }

    div[data-testid="stButton"] > button {
        min-height: 40px;
        font-size: 0.9rem;
    }
}
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------
# Navbar
# ------------------------------------------------
st.markdown("""
<div class="navbar">
    <div class="navbar-brand">⚽ TransferIQ</div>
    <div class="navbar-sub">Football Player Market Value Prediction</div>
</div>
""", unsafe_allow_html=True)

# ------------------------------------------------
# Load data & models (cached)
# ------------------------------------------------
@st.cache_resource
def load_models():
    xgb = joblib.load("./dashboard/xgb_model.pkl")
    return xgb

@st.cache_data
def load_data():
    df = pd.read_csv("./data/processed/player_transfer_value_with_sentiment.csv")
    df = df.sort_values(["player_name", "season_encoded"]).reset_index(drop=True)
    season_map = {1:"2019/20", 2:"2020/21", 3:"2021/22", 4:"2022/23", 5:"2023/24"}
    df["season_label"] = df["season_encoded"].map(season_map)
    return df

xgb_model = load_models()
df = load_data()

xgb_features  = ["lstm_pred","current_age","age_decay_factor","position_encoded",
                  "season_encoded","attacking_output_index","injury_burden_index",
                  "availability_rate","goals_per90","assists_per90",
                  "goal_contributions_per90","minutes_played","pass_accuracy_pct",
                  "vader_compound_score","log_social_buzz"]


all_players = sorted(df["player_name"].unique())

# ------------------------------------------------
# Session state — track which page we're on
# ------------------------------------------------
if "page" not in st.session_state:
    st.session_state.page   = "home"
if "player" not in st.session_state:
    st.session_state.player = None
if "season" not in st.session_state:
    st.session_state.season = "2024/25"

# ------------------------------------------------
# Ensemble prediction helper
# ------------------------------------------------
def get_ensemble_preds(player_name, future_seasons_all):
    p_df = df[df["player_name"] == player_name].sort_values("season_encoded").reset_index(drop=True)

    last_val = float(p_df["market_value_eur"].iloc[-1])
    latest_s = p_df["season_label"].iloc[-1]
    latest_row = p_df.iloc[-1]

    preds = []

    for i in range(3):
        fd = {
            "lstm_pred": last_val,  # dummy (model expects it)
            "current_age": latest_row.get("current_age", np.nan),
            "age_decay_factor": latest_row.get("age_decay_factor", np.nan),
            "position_encoded": latest_row.get("position_encoded", np.nan),
            "season_encoded": 5 + i + 1,
            "attacking_output_index": latest_row.get("attacking_output_index", np.nan),
            "injury_burden_index": latest_row.get("injury_burden_index", np.nan),
            "availability_rate": latest_row.get("availability_rate", np.nan),
            "goals_per90": latest_row.get("goals_per90", np.nan),
            "assists_per90": latest_row.get("assists_per90", np.nan),
            "goal_contributions_per90": latest_row.get("goal_contributions_per90", np.nan),
            "minutes_played": latest_row.get("minutes_played", np.nan),
            "pass_accuracy_pct": latest_row.get("pass_accuracy_pct", np.nan),
            "vader_compound_score": latest_row.get("vader_compound_score", np.nan),
            "log_social_buzz": float(np.log1p(latest_row.get("social_buzz_score", 0))),
        }

        xin = pd.DataFrame([fd])[xgb_features]

        pred = float(max(xgb_model.predict(xin)[0], 0))
        pred = float(np.clip(pred, last_val * 0.6, last_val * 1.4))

        preds.append(pred)

    return p_df, last_val, latest_s, preds

# ================================================
# HOME PAGE
# ================================================
if st.session_state.page == "home":

    st.markdown("<div style='height:0.2rem'></div>", unsafe_allow_html=True)

    col_l, col_c, col_r = st.columns([1.25, 2.5, 1.25])
    with col_c:
        st.markdown("<h3 class='search-title'>🔍 Search Player</h3>", unsafe_allow_html=True)

        # Native selectbox — has built-in search/filter as you type
        selected_player = st.selectbox(
            "Player",
            options=all_players,
            index=None,
            placeholder="Type to search player...",
            label_visibility="collapsed",
        )
        st.markdown("<div style='height:0.3rem'></div>", unsafe_allow_html=True)

        # Season selector with active color
        st.markdown("**Select Future Season**")
        season_cols = st.columns(3)
        for i, s in enumerate(["2024/25", "2025/26", "2026/27"]):
            with season_cols[i]:
                is_active = (st.session_state.season == s)
                label = f"✓ {s}" if is_active else s
                btn_t = "primary" if is_active else "secondary"
                if st.button(label, key=f"season_{s}",
                             type=btn_t, use_container_width=True):
                    st.session_state.season = s
                    st.rerun()

        st.markdown("<div style='height:0.25rem'></div>", unsafe_allow_html=True)

        if selected_player:
            if st.button("🔎 View Player Analysis", type="primary", use_container_width=True):
                st.session_state.player = selected_player
                st.session_state.page   = "player"
                st.rerun()
        else:
            st.button("🔎 View Player Analysis", type="primary",
                      use_container_width=True, disabled=True)

# ================================================
# PLAYER DETAIL PAGE
# ================================================
elif st.session_state.page == "player":

    player  = st.session_state.player
    season  = st.session_state.season
    fut_all = ["2024/25", "2025/26", "2026/27"]
    fut_idx = fut_all.index(season)

    # Back button
    if st.button("← Back to Search"):
        st.session_state.page = "home"
        st.rerun()

    # Get predictions
    with st.spinner("Generating prediction..."):
        p_df, last_val, latest_s, ens_preds = get_ensemble_preds(player, fut_all)

    pred_val    = ens_preds[fut_idx]
    pct_change  = (pred_val - last_val) / last_val * 100
    pct_str     = f"+{pct_change:.1f}%" if pct_change >= 0 else f"{pct_change:.1f}%"
    position    = p_df["position"].iloc[-1] if "position" in p_df.columns else "Player"
    age         = int(p_df["current_age"].iloc[-1]) if "current_age" in p_df.columns else ""

    # ── PLAYER HEADER ──
    st.markdown(f"""
    <div class="player-header">
        <div class="player-name-big">{player}</div>
        <div class="player-meta">{position} &nbsp;|&nbsp; Age: {age} &nbsp;|&nbsp; Season: {season}</div>
    </div>
    """, unsafe_allow_html=True)

    # ── TOP METRICS ──
    m1, m2, m3 = st.columns(3)
    with m1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Predicted Market Value ({season})</div>
            <div class="metric-value">€{pred_val/1e6:.1f}M</div>
            <div class="metric-sub">Ensemble model prediction</div>
        </div>""", unsafe_allow_html=True)
    with m2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Last Known Value (2023/24)</div>
            <div class="metric-value">€{last_val/1e6:.1f}M</div>
            <div class="metric-sub">Season 5 actual value</div>
        </div>""", unsafe_allow_html=True)
    with m3:
        color = "#1a7340" if pct_change >= 0 else "#b41f1f"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Projected Change</div>
            <div class="metric-value" style="color:{color}">{pct_str}</div>
            <div class="metric-sub">vs last known value</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # ── GRAPHS ──
    col1, col2 = st.columns(2)

    # GRAPH 1 — Historical Market Value
    with col1:
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(
            x=p_df["season_label"], y=p_df["market_value_eur"],
            mode="lines+markers", fill="tozeroy",
            line=dict(color="#1a7340", width=2), marker=dict(size=9),
            fillcolor="rgba(26,115,64,0.08)",
            hovertemplate="Season: %{x}<br>Value: €%{y:,.0f}<extra></extra>",
        ))
        for _, row in p_df.iterrows():
            fig1.add_annotation(x=row["season_label"], y=row["market_value_eur"],
                text=f"€{row['market_value_eur']/1e6:.1f}M", showarrow=False,
                yshift=14, font=dict(size=11, color="#1a7340"))
        fig1.update_layout(title="Historical Market Value", xaxis_title="Season",
            yaxis_title="Value (EUR)", showlegend=False, hovermode="x unified")
        st.plotly_chart(fig1, width='stretch')

    # GRAPH 2 — Ensemble Forecast
    with col2:
        fore_x = [latest_s] + fut_all
        fore_y = [last_val]  + ens_preds
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=fore_x, y=fore_y, mode="lines+markers",
            name="Ensemble Forecast",
            line=dict(color="#1a50a3", width=2.5),
            marker=dict(size=9, symbol="diamond"),
            hovertemplate="Season: %{x}<br>Forecast: €%{y:,.0f}<extra></extra>",
        ))
        for sx, sy in zip(fore_x[1:], fore_y[1:]):
            fig2.add_annotation(x=sx, y=sy, text=f"€{sy/1e6:.1f}M",
                showarrow=False, yshift=14, font=dict(size=11, color="#1a50a3"))
        fig2.add_vrect(x0=season, x1=season,
            fillcolor="rgba(26,80,163,0.08)", layer="below", line_width=0)
        fig2.update_layout(title="Ensemble Market Value Forecast",
            xaxis_title="Season", yaxis_title="Value (EUR)",
            showlegend=False, hovermode="x unified")
        st.plotly_chart(fig2, width='stretch')

    # GRAPH 3 — Performance Trends
    with col1:
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(x=p_df["season_label"], y=p_df["goals_per90"],
            mode="lines+markers", name="Goals/90",
            line=dict(color="#b41f1f", width=2), marker=dict(size=7),
            hovertemplate="%{x}: %{y:.2f}<extra>Goals/90</extra>"))
        fig3.add_trace(go.Scatter(x=p_df["season_label"], y=p_df["assists_per90"],
            mode="lines+markers", name="Assists/90",
            line=dict(color="#1a50a3", width=2), marker=dict(size=7),
            hovertemplate="%{x}: %{y:.2f}<extra>Assists/90</extra>"))
        fig3.add_trace(go.Scatter(x=p_df["season_label"], y=p_df["availability_rate"],
            mode="lines+markers", name="Availability",
            line=dict(color="#1a7340", width=2, dash="dot"), marker=dict(size=7),
            yaxis="y2",
            hovertemplate="%{x}: %{y:.2f}<extra>Availability</extra>"))
        fig3.update_layout(title="Performance Trends", xaxis_title="Season",
            yaxis=dict(title="Goals / Assists per 90"),
            yaxis2=dict(title="Availability", overlaying="y", side="right", range=[0,1.2]),
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5))
        st.plotly_chart(fig3, width='stretch')

    # GRAPH 4 — Sentiment Trend
    with col2:
        fig4 = go.Figure()
        fig4.add_trace(go.Bar(
            x=p_df["season_label"], y=p_df["vader_compound_score"],
            marker_color=["#1a7340" if v >= 0 else "#b41f1f"
                          for v in p_df["vader_compound_score"]],
            hovertemplate="Season: %{x}<br>Sentiment: %{y:.3f}<extra></extra>",
        ))
        fig4.add_hline(y=0, line_dash="dash", line_color="#888", line_width=1)
        fig4.update_layout(title="Public Sentiment by Season",
            xaxis_title="Season", yaxis_title="Sentiment Score",
            showlegend=False)
        st.plotly_chart(fig4, width='stretch')

# ------------------------------------------------
# Footer
# ------------------------------------------------
st.markdown("""
<div class="footer">
    TransferIQ &nbsp;|&nbsp; Football Player Market Value Prediction &nbsp;|&nbsp;
    XGBoost Prediction Model &nbsp;|&nbsp; 2026
</div>
""", unsafe_allow_html=True)