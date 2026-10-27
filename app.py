"""
app.py  —  Football Player Transfer Value Predictor Dashboard
=============================================================
A Streamlit application that lets users:
  1. Predict a player's estimated market value by filling in attributes.
  2. Compare two players side-by-side.
  3. Explore model performance metrics and feature importances.
  4. Browse the underlying dataset (top-valued players, distributions).

Run:
  streamlit run app.py

Data source: Transfermarkt via github.com/dcaribou/transfermarkt-datasets
Note: Values are *estimated* market values, not actual transfer fees.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------------------
# Path setup — allow running from project root without installing the package
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

# ---------------------------------------------------------------------------
# Page config  (must be first Streamlit call)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="⚽ Football Transfer Value Predictor",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    /* Hero banner */
    .hero-banner {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 2rem 2.5rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        color: #ffffff;
    }
    .hero-banner h1 { font-size: 2.2rem; margin: 0; }
    .hero-banner p  { font-size: 1rem; color: #a8c8e8; margin-top: 0.4rem; }

    /* Metric cards */
    .metric-card {
        background: #1e1e2e;
        border: 1px solid #333355;
        border-radius: 10px;
        padding: 1.2rem 1.4rem;
        text-align: center;
    }
    .metric-card .label { font-size: 0.8rem; color: #8888aa; text-transform: uppercase; }
    .metric-card .value { font-size: 1.8rem; font-weight: 700; color: #e0e0f0; }
    .metric-card .delta { font-size: 0.85rem; color: #66cc88; }

    /* Value prediction box */
    .prediction-box {
        background: linear-gradient(135deg, #0f3460, #533483);
        border-radius: 14px;
        padding: 2rem;
        text-align: center;
        color: white;
        margin-top: 1rem;
    }
    .prediction-box .pred-label { font-size: 0.95rem; color: #ccc; }
    .prediction-box .pred-value { font-size: 3rem; font-weight: 800; }
    .prediction-box .pred-note  { font-size: 0.78rem; color: #aaa; margin-top: 0.5rem; }

    /* Section headers */
    .section-header {
        font-size: 1.25rem;
        font-weight: 600;
        color: #e0e0f0;
        border-bottom: 2px solid #0f3460;
        padding-bottom: 0.3rem;
        margin-bottom: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
POSITIONS = [
    "Attack", "Midfield", "Defender", "Goalkeeper"
]
SUB_POSITIONS = {
    "Attack":     ["Centre-Forward", "Left Winger", "Right Winger",
                   "Second Striker", "Attack"],
    "Midfield":   ["Central Midfield", "Attacking Midfield",
                   "Defensive Midfield", "Left Midfield", "Right Midfield", "Midfield"],
    "Defender":   ["Centre-Back", "Left-Back", "Right-Back",
                   "Left Wing-Back", "Right Wing-Back", "Defender"],
    "Goalkeeper": ["Goalkeeper"],
}
FEET = ["right", "left", "both"]


# ---------------------------------------------------------------------------
# Caching helpers
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading model…")
def load_predictor():
    """Load the trained Predictor once and cache it."""
    try:
        from predictor import Predictor
        return Predictor(), None
    except Exception as exc:
        return None, str(exc)


@st.cache_data(show_spinner="Loading dataset…")
def load_features_df():
    path = ROOT / "data" / "processed" / "features.csv"
    if not path.exists():
        return None
    return pd.read_csv(path, low_memory=False)


@st.cache_data(show_spinner="Loading players…")
def load_players_df():
    path = ROOT / "data" / "raw" / "players.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path, low_memory=False)
    return df[df["market_value_in_eur"].notna()]


@st.cache_data
def load_metrics():
    path = ROOT / "models" / "metrics.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


@st.cache_data
def load_feature_importance():
    path = ROOT / "models" / "feature_importance.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


@st.cache_data
def load_encodings():
    path = ROOT / "data" / "processed" / "label_encodings.json"
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def fmt_eur(value: float) -> str:
    """Format a value in EUR with M/K suffix."""
    if value >= 1_000_000:
        return f"€{value / 1_000_000:.1f}M"
    elif value >= 1_000:
        return f"€{value / 1_000:.0f}K"
    return f"€{value:.0f}"


def build_feature_dict(predictor, inputs: dict) -> dict:
    """
    Map human-readable widget values into the integer-encoded feature dict
    that the model expects.
    """
    enc = predictor.encodings
    feat = {}

    for col in predictor.feature_columns:
        if col in enc:
            # Categorical — translate label → code
            label = inputs.get(col, "")
            feat[col] = enc[col].get(label, -1)
        else:
            # Numeric — pass through directly
            feat[col] = inputs.get(col, 0)

    return feat


# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------

def render_sidebar():
    with st.sidebar:
        st.markdown("## ⚽ Transfer Value Predictor")
        st.divider()
        st.markdown("### Navigation")
        page = st.radio(
            "Go to",
            ["🏠 Home",
             "🔮 Predict Value",
             "⚖️ Compare Players",
             "📊 Model Metrics",
             "📈 Data Explorer",
             "ℹ️ About & Limitations"],
            label_visibility="collapsed",
        )
        st.divider()
        st.markdown(
            "<small>Data: [dcaribou/transfermarkt-datasets](https://github.com/dcaribou/transfermarkt-datasets)<br>"
            "Values are **estimated**, not actual fees.</small>",
            unsafe_allow_html=True,
        )
    return page


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------

def page_home(predictor, players_df, metrics):
    st.markdown(
        """
        <div class="hero-banner">
            <h1>⚽ Football Transfer Value Predictor</h1>
            <p>Estimate a player's market value using a Random Forest model trained on
            50,000+ real Transfermarkt player profiles.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    model_ready = predictor is not None
    data_ready  = players_df is not None

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        n_players = f"{len(players_df):,}" if data_ready else "—"
        st.metric("Players in Dataset", n_players)
    with c2:
        st.metric("Model", "Random Forest" if model_ready else "Not trained")
    with c3:
        if metrics:
            rf = next((m for m in metrics if "Random" in m["model"]), None)
            st.metric("Test R²", f"{rf['R2']:.3f}" if rf else "—")
        else:
            st.metric("Test R²", "—")
    with c4:
        if metrics:
            rf = next((m for m in metrics if "Random" in m["model"]), None)
            st.metric("Test MAE", fmt_eur(rf["MAE_EUR"]) if rf else "—")
        else:
            st.metric("Test MAE", "—")

    st.divider()

    if not model_ready:
        st.warning(
            "**Model not trained yet.**  "
            "Follow the setup steps in the sidebar to run the training pipeline.",
            icon="⚠️",
        )

    col_a, col_b = st.columns([1, 1])

    with col_a:
        st.markdown("### 🚀 Quick Start")
        st.markdown(
            """
            1. **Download data** — `python src/download_data.py`
            2. **Preprocess** — `python src/preprocess.py`
            3. **Train model** — `python src/train.py`
            4. **Explore** — Use the navigation menu on the left

            The full pipeline takes ~5–10 minutes depending on your internet speed
            and machine.
            """
        )

    with col_b:
        st.markdown("### 📋 What does it predict?")
        st.markdown(
            """
            The model estimates a player's **current market value in EUR** as
            listed on Transfermarkt.

            **This is not a transfer fee prediction.**  Market value is
            Transfermarkt's editorial estimate of what a player would fetch
            on the open market.  Actual transfer fees can differ significantly.

            Features used:
            - Age, position, foot, nationality
            - Recent goals, assists, minutes played (last 2 seasons)
            - Club league / competition tier
            """
        )

    if data_ready:
        st.divider()
        st.markdown("### 🏆 Top 10 Most Valuable Players in Dataset")
        top10 = (
            players_df.nlargest(10, "market_value_in_eur")[
                ["name", "position", "current_club_name",
                 "country_of_citizenship", "market_value_in_eur"]
            ]
            .reset_index(drop=True)
        )
        top10.index += 1
        top10["market_value_in_eur"] = top10["market_value_in_eur"].apply(fmt_eur)
        top10.columns = ["Name", "Position", "Club", "Nationality", "Market Value"]
        st.dataframe(top10, use_container_width=True, height=380)


def page_predict(predictor):
    st.markdown('<div class="section-header">🔮 Predict Player Market Value</div>',
                unsafe_allow_html=True)

    if predictor is None:
        st.error("Model not loaded. Run `python src/train.py` first.")
        return

    encodings = load_encodings()

    # ---- Input form -------------------------------------------------------
    with st.form("prediction_form"):
        st.markdown("#### Player Attributes")

        col1, col2, col3 = st.columns(3)

        with col1:
            age = st.slider("Age", min_value=15, max_value=45, value=24, step=1)

            positions_available = sorted(encodings.get("position", {}).keys())
            if not positions_available:
                positions_available = POSITIONS
            position_label = st.selectbox("Position", positions_available)

            sub_pos_pool = SUB_POSITIONS.get(position_label, ["—"])
            sub_positions_available = sorted(encodings.get("sub_position", {}).keys())
            if sub_positions_available:
                # Filter to options that exist in the sub_positions pool if possible
                relevant = [s for s in sub_positions_available if
                            any(p.lower() in s.lower() for p in sub_pos_pool)]
                sub_pos_opts = relevant if relevant else sub_positions_available
            else:
                sub_pos_opts = sub_pos_pool
            sub_position_label = st.selectbox("Sub-position", sub_pos_opts)

        with col2:
            feet_available = sorted(encodings.get("foot", {}).keys())
            if not feet_available:
                feet_available = FEET
            foot_label = st.selectbox("Preferred Foot", feet_available)

            height_in_cm = st.slider("Height (cm)", 155, 210, 181, step=1)

            countries_available = sorted(encodings.get("country_of_citizenship", {}).keys())
            if not countries_available:
                countries_available = ["Unknown"]
            country = st.selectbox("Nationality", countries_available)

        with col3:
            st.markdown("**Recent Form (last 2 seasons)**")
            total_appearances = st.number_input("Appearances", 0, 150, 40, step=1)
            total_goals       = st.number_input("Goals",       0, 100, 10, step=1)
            total_assists     = st.number_input("Assists",     0, 100, 8,  step=1)
            total_minutes     = st.number_input("Minutes Played", 0, 10000, 3200, step=50)

        st.markdown("#### Club Context")
        col4, col5 = st.columns(2)
        with col4:
            total_yellow_cards = st.number_input("Yellow Cards", 0, 30, 4, step=1)
            total_red_cards    = st.number_input("Red Cards",    0, 10, 0, step=1)
        with col5:
            comp_types_available = sorted(encodings.get("competition_type", {}).keys())
            if not comp_types_available:
                comp_types_available = ["domestic_league", "domestic_cup",
                                        "international_cup", "other"]
            comp_type_label = st.selectbox("Competition Type", comp_types_available)

            squad_size         = st.number_input("Club Squad Size",   10, 60, 28, step=1)
            national_tm_players = st.number_input("Intl. Team Players in Club", 0, 30, 8, step=1)

        submitted = st.form_submit_button("🔮 Estimate Market Value", type="primary",
                                          use_container_width=True)

    if submitted:
        # Compute derived features
        safe_min      = max(total_minutes, 90)
        goals_per_90  = round(total_goals   / safe_min * 90, 4)
        assists_per_90 = round(total_assists / safe_min * 90, 4)

        # Build input dict with both numeric and categorical values
        raw_inputs = {
            "age":                    age,
            "height_in_cm":           height_in_cm,
            "total_appearances":      total_appearances,
            "total_goals":            total_goals,
            "total_assists":          total_assists,
            "total_minutes":          total_minutes,
            "total_yellow_cards":     total_yellow_cards,
            "total_red_cards":        total_red_cards,
            "goals_per_90":           goals_per_90,
            "assists_per_90":         assists_per_90,
            "squad_size":             squad_size,
            "national_team_players":  national_tm_players,
            # Categorical labels — build_feature_dict will encode them
            "position":               position_label,
            "sub_position":           sub_position_label,
            "foot":                   foot_label,
            "country_of_citizenship": country,
            "country_of_birth":       country,   # fallback same value
            "competition_type":       comp_type_label,
        }

        feat_dict = build_feature_dict(predictor, raw_inputs)
        value_eur = predictor.predict(feat_dict)

        st.markdown(
            f"""
            <div class="prediction-box">
                <div class="pred-label">Estimated Market Value</div>
                <div class="pred-value">{fmt_eur(value_eur)}</div>
                <div class="pred-note">
                    This is a model estimate, not an official Transfermarkt valuation or a transfer fee.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Context gauge — where does this fall in the dataset?
        features_df = load_features_df()
        if features_df is not None:
            pct = (features_df["market_value_in_eur"] < value_eur).mean() * 100
            st.markdown(f"**Percentile:** This estimated value is higher than **{pct:.0f}%** of players in the dataset.")

            # Mini histogram
            fig = px.histogram(
                features_df, x="market_value_in_eur",
                nbins=60,
                title="Where Your Player Falls in the Value Distribution",
                labels={"market_value_in_eur": "Market Value (EUR)"},
                color_discrete_sequence=["#3a86ff"],
            )
            fig.add_vline(x=value_eur, line_color="#ff6b6b",
                          line_width=2.5, line_dash="dash",
                          annotation_text="Your player",
                          annotation_position="top right")
            fig.update_xaxes(range=[0, features_df["market_value_in_eur"].quantile(0.99)])
            fig.update_layout(template="plotly_dark", height=320)
            st.plotly_chart(fig, use_container_width=True)


def page_compare(predictor):
    st.markdown('<div class="section-header">⚖️ Compare Two Players</div>',
                unsafe_allow_html=True)

    if predictor is None:
        st.error("Model not loaded. Run `python src/train.py` first.")
        return

    encodings = load_encodings()
    positions_available = sorted(encodings.get("position", {}).keys()) or POSITIONS
    feet_available      = sorted(encodings.get("foot",     {}).keys()) or FEET
    countries_available = sorted(encodings.get("country_of_citizenship", {}).keys()) or ["Unknown"]

    def player_form(label: str, defaults: dict):
        st.markdown(f"#### {label}")
        age               = st.slider(f"Age##{label}", 15, 45, defaults["age"], key=f"age_{label}")
        pos               = st.selectbox(f"Position##{label}", positions_available, key=f"pos_{label}",
                                          index=positions_available.index(defaults["pos"]) if defaults["pos"] in positions_available else 0)
        foot              = st.selectbox(f"Foot##{label}", feet_available, key=f"foot_{label}")
        height            = st.slider(f"Height (cm)##{label}", 155, 210, defaults["height"], key=f"h_{label}")
        goals             = st.number_input(f"Goals (last 2 seasons)##{label}", 0, 100, defaults["goals"], key=f"g_{label}")
        assists           = st.number_input(f"Assists##{label}", 0, 100, defaults["assists"], key=f"a_{label}")
        minutes           = st.number_input(f"Minutes Played##{label}", 0, 10000, defaults["minutes"], step=50, key=f"m_{label}")
        appearances       = st.number_input(f"Appearances##{label}", 0, 150, defaults["apps"], key=f"ap_{label}")

        safe_min = max(minutes, 90)
        return {
            "age": age, "position": pos, "foot": foot,
            "height_in_cm": height, "total_goals": goals,
            "total_assists": assists, "total_minutes": minutes,
            "total_appearances": appearances,
            "total_yellow_cards": 0, "total_red_cards": 0,
            "goals_per_90":   round(goals   / safe_min * 90, 4),
            "assists_per_90": round(assists  / safe_min * 90, 4),
            "squad_size": 25, "national_team_players": 8,
            "competition_type": "domestic_league",
            "country_of_citizenship": countries_available[0],
            "country_of_birth": countries_available[0],
            "sub_position": pos,
        }

    col_left, col_right = st.columns(2)
    with col_left:
        p1 = player_form("Player A", {"age": 22, "pos": positions_available[0],
                                       "height": 178, "goals": 18, "assists": 7,
                                       "minutes": 3200, "apps": 38})
    with col_right:
        p2 = player_form("Player B", {"age": 29, "pos": positions_available[0],
                                       "height": 182, "goals": 12, "assists": 9,
                                       "minutes": 2800, "apps": 32})

    if st.button("⚖️ Compare Market Values", type="primary", use_container_width=True):
        v1 = predictor.predict(build_feature_dict(predictor, p1))
        v2 = predictor.predict(build_feature_dict(predictor, p2))

        c1, c2, c3 = st.columns([2, 1, 2])
        with c1:
            st.markdown(
                f'<div class="prediction-box"><div class="pred-label">Player A</div>'
                f'<div class="pred-value">{fmt_eur(v1)}</div></div>',
                unsafe_allow_html=True,
            )
        with c2:
            st.markdown("<div style='text-align:center;padding-top:2.5rem;font-size:1.5rem;'>vs</div>",
                        unsafe_allow_html=True)
        with c3:
            st.markdown(
                f'<div class="prediction-box"><div class="pred-label">Player B</div>'
                f'<div class="pred-value">{fmt_eur(v2)}</div></div>',
                unsafe_allow_html=True,
            )

        # Radar chart comparison
        categories = ["Age (norm)", "Goals", "Assists", "Minutes", "Appearances"]

        def norm(val, lo, hi):
            return (val - lo) / max(hi - lo, 1)

        r1 = [norm(p1["age"], 15, 45),
              norm(p1["total_goals"], 0, 50),
              norm(p1["total_assists"], 0, 30),
              norm(p1["total_minutes"], 0, 7000),
              norm(p1["total_appearances"], 0, 80)]
        r2 = [norm(p2["age"], 15, 45),
              norm(p2["total_goals"], 0, 50),
              norm(p2["total_assists"], 0, 30),
              norm(p2["total_minutes"], 0, 7000),
              norm(p2["total_appearances"], 0, 80)]

        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=r1 + [r1[0]], theta=categories + [categories[0]],
            fill="toself", name=f"Player A  {fmt_eur(v1)}",
            line_color="#3a86ff",
        ))
        fig.add_trace(go.Scatterpolar(
            r=r2 + [r2[0]], theta=categories + [categories[0]],
            fill="toself", name=f"Player B  {fmt_eur(v2)}",
            line_color="#ff6b6b",
        ))
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            template="plotly_dark",
            title="Attribute Radar (normalised)",
            height=420,
        )
        st.plotly_chart(fig, use_container_width=True)

        # Difference summary
        diff = v1 - v2
        winner = "Player A" if diff > 0 else "Player B"
        st.info(
            f"**{winner}** is estimated to be worth **{fmt_eur(abs(diff))} more** "
            f"({abs(diff/max(v2,1)*100):.0f}% difference)."
        )


def page_metrics(metrics, feature_importance):
    st.markdown('<div class="section-header">📊 Model Performance Metrics</div>',
                unsafe_allow_html=True)

    if metrics is None:
        st.warning("No metrics file found.  Run `python src/train.py` first.")
        return

    # Metrics table
    st.markdown("#### Test-Set Evaluation (held-out 20%)")
    df_m = pd.DataFrame(metrics)
    df_m["MAE_EUR"]  = df_m["MAE_EUR"].apply(fmt_eur)
    df_m["RMSE_EUR"] = df_m["RMSE_EUR"].apply(fmt_eur)
    st.dataframe(df_m, use_container_width=True)

    rf_row = next((m for m in metrics if "Random" in m["model"]), None)
    bs_row = next((m for m in metrics if "Baseline" in m["model"]), None)

    if rf_row and bs_row:
        improvement = (1 - rf_row["MAE_EUR"] / bs_row["MAE_EUR"]) * 100
        st.success(
            f"The Random Forest reduces MAE by **{improvement:.1f}%** over the median baseline, "
            f"achieving R² = **{rf_row['R2']}** on the test set."
        )

        # Bar chart comparison
        bar_df = pd.DataFrame({
            "Model":     [m["model"] for m in metrics],
            "MAE (EUR)": [m["MAE_EUR"] for m in metrics],
            "R²":        [m["R2"]     for m in metrics],
        })
        col1, col2 = st.columns(2)
        with col1:
            fig_mae = px.bar(bar_df, x="Model", y="MAE (EUR)",
                             title="MAE by Model (lower = better)",
                             color="Model", template="plotly_dark",
                             color_discrete_sequence=["#ff6b6b", "#3a86ff"])
            st.plotly_chart(fig_mae, use_container_width=True)
        with col2:
            fig_r2 = px.bar(bar_df, x="Model", y="R²",
                            title="R² by Model (higher = better)",
                            color="Model", template="plotly_dark",
                            color_discrete_sequence=["#ff6b6b", "#3a86ff"])
            fig_r2.update_yaxes(range=[0, 1])
            st.plotly_chart(fig_r2, use_container_width=True)

    if rf_row and "CV_R2_mean" in rf_row:
        st.markdown(
            f"**5-fold Cross-Validation R²** (training data):  "
            f"**{rf_row['CV_R2_mean']:.4f}** ± {rf_row['CV_R2_std']:.4f}"
        )

    # Feature importance
    if feature_importance is not None:
        st.markdown("#### Feature Importances (Random Forest)")
        top20 = feature_importance.head(20)
        fig_fi = px.bar(
            top20.sort_values("importance"),
            x="importance", y="feature", orientation="h",
            title="Top 20 Feature Importances",
            template="plotly_dark",
            color="importance",
            color_continuous_scale="Blues",
        )
        fig_fi.update_layout(height=500, showlegend=False)
        st.plotly_chart(fig_fi, use_container_width=True)


def page_explorer(features_df, players_df):
    st.markdown('<div class="section-header">📈 Data Explorer</div>',
                unsafe_allow_html=True)

    if features_df is None:
        st.warning("Processed data not found.  Run `python src/preprocess.py` first.")
        return

    df = features_df.copy()

    # --------------- Distribution of market values ----------------------
    st.markdown("#### Market Value Distribution")
    fig_dist = px.histogram(
        df, x="market_value_in_eur",
        nbins=80,
        title="Distribution of Player Market Values (EUR)",
        labels={"market_value_in_eur": "Market Value (EUR)"},
        template="plotly_dark",
        color_discrete_sequence=["#3a86ff"],
    )
    # Cap x-axis at 99th pct for readability
    p99 = df["market_value_in_eur"].quantile(0.99)
    fig_dist.update_xaxes(range=[0, p99])
    st.plotly_chart(fig_dist, use_container_width=True)

    # --------------- Value vs Age ----------------------------------------
    st.markdown("#### Market Value vs Age")
    fig_age = px.scatter(
        df.sample(min(5000, len(df)), random_state=42),
        x="age", y="market_value_in_eur",
        opacity=0.5,
        trendline="lowess",
        title="Market Value vs Age (random 5,000 sample)",
        labels={"age": "Age", "market_value_in_eur": "Market Value (EUR)"},
        template="plotly_dark",
        color_discrete_sequence=["#3a86ff"],
    )
    fig_age.update_yaxes(range=[0, df["market_value_in_eur"].quantile(0.97)])
    st.plotly_chart(fig_age, use_container_width=True)

    # --------------- Value by Position -----------------------------------
    if players_df is not None and "position" in players_df.columns:
        st.markdown("#### Median Market Value by Position")
        pos_df = (
            players_df.dropna(subset=["position", "market_value_in_eur"])
            .groupby("position")["market_value_in_eur"]
            .median()
            .reset_index()
            .sort_values("market_value_in_eur", ascending=False)
        )
        pos_df.columns = ["Position", "Median Market Value (EUR)"]
        fig_pos = px.bar(
            pos_df, x="Position", y="Median Market Value (EUR)",
            title="Median Market Value by Position",
            template="plotly_dark",
            color="Median Market Value (EUR)",
            color_continuous_scale="Blues",
        )
        st.plotly_chart(fig_pos, use_container_width=True)

    # --------------- Top nationalities ----------------------------------
    if players_df is not None and "country_of_citizenship" in players_df.columns:
        st.markdown("#### Top 20 Nationalities by Total Market Value")
        nat_df = (
            players_df.dropna(subset=["country_of_citizenship", "market_value_in_eur"])
            .groupby("country_of_citizenship")["market_value_in_eur"]
            .sum()
            .reset_index()
            .nlargest(20, "market_value_in_eur")
        )
        nat_df.columns = ["Country", "Total Market Value (EUR)"]
        fig_nat = px.bar(
            nat_df.sort_values("Total Market Value (EUR)"),
            x="Total Market Value (EUR)", y="Country",
            orientation="h",
            template="plotly_dark",
            color="Total Market Value (EUR)",
            color_continuous_scale="Blues",
        )
        fig_nat.update_layout(height=520)
        st.plotly_chart(fig_nat, use_container_width=True)

    # --------------- Raw sample -----------------------------------------
    st.markdown("#### Raw Feature Sample (first 100 rows)")
    st.dataframe(df.head(100), use_container_width=True, height=320)


def page_about():
    st.markdown('<div class="section-header">ℹ️ About & Limitations</div>',
                unsafe_allow_html=True)

    st.markdown(
        """
        ### What this app does
        This dashboard predicts a football player's **estimated market value in EUR**
        using a Random Forest regression model trained on real Transfermarkt data.

        ### Data source
        Data is sourced from the open-source project
        [dcaribou/transfermarkt-datasets](https://github.com/dcaribou/transfermarkt-datasets),
        which scrapes and publishes structured data from [Transfermarkt](https://www.transfermarkt.com/).

        The dataset covers 50,000+ players, 88,000+ games, and 1.9M+ appearances
        from Europe's top leagues and national competitions.

        ### Model
        | Step | Detail |
        |------|--------|
        | Algorithm | Random Forest Regressor (300 trees) |
        | Target | log₁(market_value_in_eur) — exponentiated at prediction time |
        | Features | Age, position, foot, nationality, appearance stats, club context |
        | Split | 80% train / 20% test, stratified by value quartile |
        | Baseline | Median predictor (DummyRegressor) |

        ### ⚠️ Known Limitations
        1. **Not actual transfer fees.** Transfermarkt market values are editorial
           estimates. Real transfer fees can be far higher (add-ons, agent fees)
           or lower (free transfers, loan deals).
        2. **Recency lag.** The dataset snapshot is current to July 2026. Very recent
           form or injuries are not reflected.
        3. **Position granularity.** The model uses Transfermarkt's position taxonomy.
           Fine-grained tactical roles (e.g., press-forward vs. target striker) are
           not captured.
        4. **Club prestige proxy.** Club market value / reputation is approximated
           by squad size and national-team players, not a true prestige score.
        5. **No psychological or injury history features.** Injury-prone players or
           contract situations are not modelled.
        6. **Regression to the mean.** Random Forests tend to under-predict extreme
           values (€100M+ superstars) and over-predict very low values.
        7. **Nationality encoding.** Countries with few players in the dataset
           (code = -1) reduce prediction accuracy for those players.

        ### Licence & Attribution
        - Dataset: [CC-BY-SA 4.0 attribution required](https://github.com/dcaribou/transfermarkt-datasets)
        - Original data owner: [Transfermarkt GmbH & Co. KG](https://www.transfermarkt.com/) — non-commercial use only
        """
    )


# ---------------------------------------------------------------------------
# App entry point
# ---------------------------------------------------------------------------

def main():
    predictor, load_err = load_predictor()
    players_df          = load_players_df()
    features_df         = load_features_df()
    metrics             = load_metrics()
    feature_importance  = load_feature_importance()

    page = render_sidebar()

    if load_err and page not in ("🏠 Home", "ℹ️ About & Limitations",
                                  "📈 Data Explorer", "📊 Model Metrics"):
        st.warning(f"⚠️ {load_err}", icon="⚠️")

    if   page == "🏠 Home":
        page_home(predictor, players_df, metrics)
    elif page == "🔮 Predict Value":
        page_predict(predictor)
    elif page == "⚖️ Compare Players":
        page_compare(predictor)
    elif page == "📊 Model Metrics":
        page_metrics(metrics, feature_importance)
    elif page == "📈 Data Explorer":
        page_explorer(features_df, players_df)
    elif page == "ℹ️ About & Limitations":
        page_about()


if __name__ == "__main__":
    main()
