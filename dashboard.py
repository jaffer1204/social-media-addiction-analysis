from pathlib import Path
import html
import re

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


# ============================================================
# PAGE SETUP
# ============================================================
st.set_page_config(
    page_title="Social Media Addiction Intelligence",
    page_icon="🧠",
    layout="wide",
)


# ============================================================
# DATA LOADING
# ============================================================
@st.cache_data(show_spinner=False)
def load_data():
    """Load the dataset from the project folder."""
    dataset_paths = [
        Path("dataset") / "Students Social Media Addiction.csv",
        Path("Students Social Media Addiction.csv"),
    ]

    for dataset_path in dataset_paths:
        if dataset_path.exists():
            return pd.read_csv(dataset_path)

    st.error("Dataset file not found. Place it inside the dataset folder.")
    st.stop()


df = load_data()


# ============================================================
# CONSTANTS
# ============================================================
FEATURE_COLUMNS = [
    "Age",
    "Avg_Daily_Usage_Hours",
    "Sleep_Hours_Per_Night",
    "Mental_Health_Score",
    "Most_Used_Platform",
]

NUMERIC_FEATURES = [
    "Age",
    "Avg_Daily_Usage_Hours",
    "Sleep_Hours_Per_Night",
    "Mental_Health_Score",
]

CATEGORICAL_FEATURES = ["Most_Used_Platform"]

PLATFORM_STYLES = {
    "Instagram": ("IG", "linear-gradient(135deg, #f97316, #ec4899, #8b5cf6)"),
    "TikTok": ("TT", "linear-gradient(135deg, #06b6d4, #111827, #f43f5e)"),
    "YouTube": ("YT", "linear-gradient(135deg, #ef4444, #b91c1c)"),
    "Twitter": ("X", "linear-gradient(135deg, #38bdf8, #0f172a)"),
    "Facebook": ("f", "linear-gradient(135deg, #2563eb, #1d4ed8)"),
    "LinkedIn": ("in", "linear-gradient(135deg, #0ea5e9, #0369a1)"),
    "Snapchat": ("SC", "linear-gradient(135deg, #fde047, #f59e0b)"),
    "WhatsApp": ("WA", "linear-gradient(135deg, #22c55e, #15803d)"),
    "Discord": ("DC", "linear-gradient(135deg, #818cf8, #4f46e5)"),
    "Reddit": ("RD", "linear-gradient(135deg, #fb923c, #ea580c)"),
}


# ============================================================
# SIDEBAR THEME CONTROL
# ============================================================
st.sidebar.title("ðŸ§­ Control Center")

theme_mode = st.sidebar.radio(
    "Theme",
    ["Dark", "Light"],
    horizontal=True,
    index=0,
)

if theme_mode == "Dark":
    app_background = """
        radial-gradient(circle at 12% 12%, rgba(56, 189, 248, 0.24), transparent 28%),
        radial-gradient(circle at 82% 6%, rgba(244, 114, 182, 0.17), transparent 25%),
        radial-gradient(circle at 70% 86%, rgba(45, 212, 191, 0.14), transparent 29%),
        linear-gradient(135deg, #050812 0%, #0b1220 48%, #111827 100%)
    """
    sidebar_background = "rgba(2, 6, 23, 0.88)"
    hero_background = """
        linear-gradient(135deg, rgba(15, 23, 42, 0.88), rgba(14, 116, 144, 0.34)),
        linear-gradient(90deg, rgba(56, 189, 248, 0.14), rgba(244, 114, 182, 0.08))
    """
    prediction_background = """
        linear-gradient(135deg, rgba(15, 23, 42, 0.88), rgba(8, 47, 73, 0.62)),
        linear-gradient(90deg, rgba(56, 189, 248, 0.16), rgba(244, 114, 182, 0.10))
    """
    panel = "rgba(15, 23, 42, 0.70)"
    panel_strong = "rgba(15, 23, 42, 0.90)"
    text_color = "#f8fafc"
    muted_color = "#b6c6dc"
    soft_text = "#cbd5e1"
    title_color = "#ffffff"
    note_color = "#9fb1ca"
    metric_background = "rgba(15, 23, 42, 0.68)"
    tab_background = "rgba(15, 23, 42, 0.62)"
    hover_label_bg = "#020617"
    grid_color = "rgba(148, 163, 184, 0.14)"
    chart_template = "plotly_dark"
else:
    app_background = """
        radial-gradient(circle at 12% 12%, rgba(14, 165, 233, 0.18), transparent 28%),
        radial-gradient(circle at 82% 6%, rgba(236, 72, 153, 0.12), transparent 25%),
        radial-gradient(circle at 70% 86%, rgba(20, 184, 166, 0.12), transparent 29%),
        linear-gradient(135deg, #f8fbff 0%, #eef7ff 48%, #f8fafc 100%)
    """
    sidebar_background = "rgba(255, 255, 255, 0.86)"
    hero_background = """
        linear-gradient(135deg, rgba(255, 255, 255, 0.88), rgba(224, 242, 254, 0.72)),
        linear-gradient(90deg, rgba(14, 165, 233, 0.16), rgba(236, 72, 153, 0.10))
    """
    prediction_background = """
        linear-gradient(135deg, rgba(255, 255, 255, 0.90), rgba(224, 242, 254, 0.76)),
        linear-gradient(90deg, rgba(14, 165, 233, 0.14), rgba(236, 72, 153, 0.09))
    """
    panel = "rgba(255, 255, 255, 0.74)"
    panel_strong = "rgba(255, 255, 255, 0.92)"
    text_color = "#0f172a"
    muted_color = "#475569"
    soft_text = "#334155"
    title_color = "#0f172a"
    note_color = "#64748b"
    metric_background = "rgba(255, 255, 255, 0.70)"
    tab_background = "rgba(255, 255, 255, 0.72)"
    hover_label_bg = "#0f172a"
    grid_color = "rgba(71, 85, 105, 0.16)"
    chart_template = "plotly_white"


# ============================================================
# PREMIUM RESPONSIVE UI
# ============================================================
st.markdown(
    """
    <style>
    :root {
        --app-bg: {text_color};
        --panel: {panel};
        --panel-strong: {panel_strong};
        --border: rgba(125, 211, 252, 0.18);
        --text: {text_color};
        --muted: {muted_color};
        --soft-text: {soft_text};
        --title-text: {title_color};
        --note-text: {note_color};
        --metric-bg: {metric_background};
        --tab-bg: {tab_background};
        --cyan: #38bdf8;
        --teal: #2dd4bf;
        --pink: #f472b6;
        --amber: #f59e0b;
    }

    .stApp {
        color: var(--text);
        background: {app_background};
    }

    .block-container {
        padding-top: 1.4rem;
        padding-bottom: 2.4rem;
        max-width: 1480px;
        position: relative;
        z-index: 1;
    }

    section[data-testid="stSidebar"] {
        background: {sidebar_background};
        border-right: 1px solid rgba(56, 189, 248, 0.16);
        z-index: 3;
    }

    .particle-field {
        position: fixed;
        inset: 0;
        z-index: 0;
        overflow: hidden;
        pointer-events: none;
    }

    .particle-field span {
        position: absolute;
        width: 6px;
        height: 6px;
        border-radius: 999px;
        background: rgba(56, 189, 248, 0.58);
        box-shadow: 0 0 18px rgba(56, 189, 248, 0.76);
        animation: particleFloat 18s linear infinite;
        opacity: 0.58;
    }

    .particle-field span:nth-child(1) { left: 7%; top: 78%; animation-delay: -1s; animation-duration: 20s; }
    .particle-field span:nth-child(2) { left: 16%; top: 34%; animation-delay: -7s; animation-duration: 24s; }
    .particle-field span:nth-child(3) { left: 26%; top: 88%; animation-delay: -4s; animation-duration: 19s; }
    .particle-field span:nth-child(4) { left: 38%; top: 22%; animation-delay: -10s; animation-duration: 26s; }
    .particle-field span:nth-child(5) { left: 49%; top: 68%; animation-delay: -5s; animation-duration: 21s; }
    .particle-field span:nth-child(6) { left: 58%; top: 40%; animation-delay: -12s; animation-duration: 25s; }
    .particle-field span:nth-child(7) { left: 67%; top: 82%; animation-delay: -3s; animation-duration: 23s; }
    .particle-field span:nth-child(8) { left: 78%; top: 18%; animation-delay: -9s; animation-duration: 22s; }
    .particle-field span:nth-child(9) { left: 88%; top: 62%; animation-delay: -6s; animation-duration: 27s; }
    .particle-field span:nth-child(10) { left: 94%; top: 34%; animation-delay: -14s; animation-duration: 20s; }

    @keyframes particleFloat {
        0% {
            transform: translate3d(0, 40px, 0) scale(0.7);
            opacity: 0;
        }
        18% {
            opacity: 0.62;
        }
        100% {
            transform: translate3d(24px, -130vh, 0) scale(1.12);
            opacity: 0;
        }
    }

    .hero {
        position: relative;
        overflow: hidden;
        border: 1px solid rgba(125, 211, 252, 0.23);
        border-radius: 18px;
        padding: 32px 34px;
        margin-bottom: 24px;
        background: {hero_background};
        box-shadow: 0 24px 70px rgba(0, 0, 0, 0.32);
        backdrop-filter: blur(18px);
    }

    .hero::after {
        content: "";
        position: absolute;
        inset: auto -10% -80% 35%;
        height: 220px;
        background: radial-gradient(circle, rgba(56, 189, 248, 0.22), transparent 70%);
        pointer-events: none;
    }

    .eyebrow {
        color: #bae6fd;
        font-size: 0.83rem;
        font-weight: 800;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        margin-bottom: 10px;
    }

    .hero-title {
        color: var(--title-text);
        font-size: clamp(2rem, 4vw, 3.7rem);
        line-height: 1.02;
        font-weight: 900;
        letter-spacing: 0;
        margin-bottom: 12px;
    }

    .hero-subtitle {
        max-width: 980px;
        color: var(--soft-text);
        font-size: 1.02rem;
        line-height: 1.62;
    }

    .section-title {
        color: var(--text);
        font-size: 1.38rem;
        font-weight: 850;
        margin: 1.3rem 0 0.8rem 0;
    }

    .glass-card {
        background: var(--panel);
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 20px 21px;
        min-height: 132px;
        backdrop-filter: blur(18px);
        box-shadow: 0 16px 42px rgba(0, 0, 0, 0.23);
        transition: transform 0.22s ease, border-color 0.22s ease, box-shadow 0.22s ease;
    }

    .glass-card:hover {
        transform: translateY(-5px);
        border-color: rgba(56, 189, 248, 0.68);
        box-shadow: 0 22px 58px rgba(14, 165, 233, 0.18);
    }

    .kpi-icon {
        color: var(--cyan);
        font-size: 1.35rem;
        margin-bottom: 10px;
        text-shadow: 0 0 18px rgba(56, 189, 248, 0.72);
    }

    .kpi-label {
        color: var(--muted);
        font-size: 0.84rem;
        font-weight: 750;
        margin-bottom: 8px;
    }

    .kpi-value {
        color: var(--title-text);
        font-size: 1.65rem;
        line-height: 1.18;
        font-weight: 900;
        overflow-wrap: anywhere;
    }

    .kpi-note {
        color: var(--note-text);
        font-size: 0.78rem;
        margin-top: 9px;
    }

    .insight-card {
        background: var(--panel);
        border: 1px solid rgba(125, 211, 252, 0.18);
        border-left: 4px solid var(--cyan);
        border-radius: 14px;
        padding: 15px 17px;
        color: var(--text);
        margin-bottom: 0.9rem;
        box-shadow: 0 14px 34px rgba(0, 0, 0, 0.22);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }

    .insight-card:hover {
        transform: translateY(-3px);
        border-color: rgba(45, 212, 191, 0.55);
    }

    .insight-title {
        color: var(--title-text);
        font-weight: 850;
        margin-bottom: 4px;
    }

    .insight-body {
        color: var(--soft-text);
        font-size: 0.95rem;
        line-height: 1.55;
    }

    .prediction-panel {
        background: {prediction_background};
        border: 1px solid rgba(56, 189, 248, 0.34);
        border-radius: 18px;
        padding: 22px 24px;
        margin-bottom: 1rem;
        box-shadow: 0 20px 60px rgba(14, 165, 233, 0.15);
        backdrop-filter: blur(18px);
    }

    .prediction-score {
        color: var(--title-text);
        font-size: 2.2rem;
        font-weight: 900;
        line-height: 1.1;
    }

    .prediction-meta {
        color: var(--soft-text);
        font-size: 0.95rem;
        margin-top: 8px;
    }

    .metric-highlight {
        color: #bae6fd;
        font-weight: 850;
    }

    div[data-testid="stMetric"] {
        background: var(--metric-bg);
        border: 1px solid rgba(125, 211, 252, 0.16);
        border-radius: 14px;
        padding: 14px 15px;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        border-bottom: 1px solid rgba(125, 211, 252, 0.14);
    }

    .stTabs [data-baseweb="tab"] {
        background: var(--tab-bg);
        border: 1px solid rgba(125, 211, 252, 0.14);
        border-radius: 999px;
        color: var(--soft-text);
        padding: 8px 18px;
    }

    .stTabs [aria-selected="true"] {
        color: var(--title-text);
        border-color: rgba(56, 189, 248, 0.55);
        box-shadow: 0 0 24px rgba(56, 189, 248, 0.14);
    }

    div[data-testid="stDataFrame"] {
        border: 1px solid rgba(125, 211, 252, 0.14);
        border-radius: 14px;
        overflow: hidden;
    }

    .platform-strip {
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
        margin: 0.2rem 0 1rem 0;
    }

    .platform-pill {
        display: inline-flex;
        align-items: center;
        gap: 9px;
        padding: 9px 12px;
        border: 1px solid rgba(125, 211, 252, 0.22);
        border-radius: 999px;
        background: var(--panel);
        color: var(--text);
        box-shadow: 0 10px 26px rgba(14, 165, 233, 0.10);
        backdrop-filter: blur(14px);
    }

    .platform-logo {
        display: inline-grid;
        place-items: center;
        width: 28px;
        height: 28px;
        border-radius: 9px;
        color: #ffffff;
        font-size: 0.72rem;
        font-weight: 900;
        letter-spacing: 0;
        box-shadow: 0 0 18px rgba(56, 189, 248, 0.24);
    }

    .platform-name {
        font-size: 0.88rem;
        font-weight: 760;
    }

    @media (max-width: 900px) {
        .hero {
            padding: 24px 22px;
        }

        .glass-card {
            min-height: 116px;
        }
    }
    </style>
    """
    .replace("{app_background}", app_background)
    .replace("{sidebar_background}", sidebar_background)
    .replace("{hero_background}", hero_background)
    .replace("{prediction_background}", prediction_background)
    .replace("{panel_strong}", panel_strong)
    .replace("{panel}", panel)
    .replace("{text_color}", text_color)
    .replace("{muted_color}", muted_color)
    .replace("{soft_text}", soft_text)
    .replace("{title_color}", title_color)
    .replace("{note_color}", note_color)
    .replace("{metric_background}", metric_background)
    .replace("{tab_background}", tab_background),
    unsafe_allow_html=True,
)


# ============================================================
# HEADER COMPONENT
# ============================================================
def render_live_header():
    """Render the dashboard header with a live clock."""
    components.html(
        f"""
        <!doctype html>
        <html>
        <head>
            <style>
                html, body {{
                    margin: 0;
                    padding: 0;
                    background: transparent;
                    font-family: Arial, sans-serif;
                }}

                .hero {{
                    position: relative;
                    overflow: hidden;
                    border: 1px solid rgba(125, 211, 252, 0.23);
                    border-radius: 18px;
                    padding: 32px 34px;
                    color: {text_color};
                    background: {hero_background};
                    box-shadow: 0 24px 70px rgba(0, 0, 0, 0.24);
                    backdrop-filter: blur(18px);
                    min-height: 154px;
                }}

                .hero::after {{
                    content: "";
                    position: absolute;
                    inset: auto -10% -80% 35%;
                    height: 220px;
                    background: radial-gradient(circle, rgba(56, 189, 248, 0.22), transparent 70%);
                    pointer-events: none;
                }}

                .hero-grid {{
                    position: relative;
                    z-index: 1;
                    display: grid;
                    grid-template-columns: minmax(0, 1fr) auto;
                    gap: 24px;
                    align-items: center;
                }}

                .eyebrow {{
                    color: #38bdf8;
                    font-size: 0.83rem;
                    font-weight: 800;
                    letter-spacing: 0.12em;
                    text-transform: uppercase;
                    margin-bottom: 10px;
                }}

                .title {{
                    color: {title_color};
                    font-size: clamp(2rem, 4vw, 3.55rem);
                    line-height: 1.02;
                    font-weight: 900;
                    letter-spacing: 0;
                    margin-bottom: 12px;
                }}

                .subtitle {{
                    max-width: 980px;
                    color: {soft_text};
                    font-size: 1.02rem;
                    line-height: 1.62;
                }}

                .clock-card {{
                    min-width: 235px;
                    border: 1px solid rgba(125, 211, 252, 0.24);
                    border-radius: 16px;
                    padding: 16px 18px;
                    background: {panel};
                    box-shadow: 0 16px 42px rgba(14, 165, 233, 0.14);
                    backdrop-filter: blur(18px);
                    text-align: right;
                }}

                .clock-label {{
                    color: {muted_color};
                    font-size: 0.76rem;
                    font-weight: 800;
                    letter-spacing: 0.12em;
                    text-transform: uppercase;
                    margin-bottom: 7px;
                }}

                .clock-time {{
                    color: {title_color};
                    font-size: 1.8rem;
                    font-weight: 900;
                    line-height: 1.05;
                    text-shadow: 0 0 18px rgba(56, 189, 248, 0.26);
                }}

                .clock-date {{
                    color: {soft_text};
                    font-size: 0.88rem;
                    margin-top: 8px;
                }}

                @media (max-width: 760px) {{
                    .hero {{
                        padding: 24px 22px;
                    }}

                    .hero-grid {{
                        grid-template-columns: 1fr;
                    }}

                    .clock-card {{
                        width: auto;
                        min-width: 0;
                        text-align: left;
                    }}
                }}
            </style>
        </head>
        <body>
            <div class="hero">
                <div class="hero-grid">
                    <div>
                        <div class="eyebrow">AI Analytics Platform</div>
                        <div class="title">Social Media Addiction Intelligence</div>
                        <div class="subtitle">
                            A premium Streamlit analytics workspace for exploring student behavior,
                            usage intensity, sleep health, platform patterns, and AI-powered addiction risk.
                        </div>
                    </div>
                    <div class="clock-card">
                        <div class="clock-label">Live System Time</div>
                        <div class="clock-time" id="clock-time">--:--:--</div>
                        <div class="clock-date" id="clock-date">Loading date</div>
                    </div>
                </div>
            </div>

            <script>
                const timeElement = document.getElementById("clock-time");
                const dateElement = document.getElementById("clock-date");

                function updateClock() {{
                    const now = new Date();
                    timeElement.textContent = now.toLocaleTimeString([], {{
                        hour: "2-digit",
                        minute: "2-digit",
                        second: "2-digit"
                    }});
                    dateElement.textContent = now.toLocaleDateString([], {{
                        weekday: "long",
                        year: "numeric",
                        month: "long",
                        day: "numeric"
                    }});
                }}

                updateClock();
                setInterval(updateClock, 1000);
            </script>
        </body>
        </html>
        """,
        height=230,
    )


# ============================================================
# BACKGROUND PARTICLES
# ============================================================
st.markdown(
    """
    <div class="particle-field" aria-hidden="true">
        <span></span><span></span><span></span><span></span><span></span>
        <span></span><span></span><span></span><span></span><span></span>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SIDEBAR FILTERS AND PREDICTION INPUTS
# ============================================================
with st.sidebar.expander("Interactive Filters", expanded=True):
    gender_options = sorted(df["Gender"].dropna().unique())
    platform_options = sorted(df["Most_Used_Platform"].dropna().unique())
    academic_options = sorted(df["Academic_Level"].dropna().unique())

    selected_gender = st.multiselect("Gender", gender_options, default=gender_options)
    selected_platform = st.multiselect(
        "Most Used Platform",
        platform_options,
        default=platform_options,
    )
    selected_academic = st.multiselect(
        "Academic Level",
        academic_options,
        default=academic_options,
    )
    selected_age = st.slider(
        "Age Range",
        min_value=int(df["Age"].min()),
        max_value=int(df["Age"].max()),
        value=(int(df["Age"].min()), int(df["Age"].max())),
    )
    selected_addiction_score = st.slider(
        "Addiction Score Range",
        min_value=int(df["Addicted_Score"].min()),
        max_value=int(df["Addicted_Score"].max()),
        value=(int(df["Addicted_Score"].min()), int(df["Addicted_Score"].max())),
    )

with st.sidebar.expander("AI Prediction Inputs", expanded=True):
    prediction_age = st.number_input(
        "Age",
        min_value=int(df["Age"].min()),
        max_value=int(df["Age"].max()),
        value=int(df["Age"].median()),
        step=1,
    )
    prediction_usage = st.number_input(
        "Daily usage hours",
        min_value=float(df["Avg_Daily_Usage_Hours"].min()),
        max_value=float(df["Avg_Daily_Usage_Hours"].max()),
        value=float(round(df["Avg_Daily_Usage_Hours"].median(), 1)),
        step=0.1,
        format="%.1f",
    )
    prediction_sleep = st.number_input(
        "Sleep hours",
        min_value=float(df["Sleep_Hours_Per_Night"].min()),
        max_value=float(df["Sleep_Hours_Per_Night"].max()),
        value=float(round(df["Sleep_Hours_Per_Night"].median(), 1)),
        step=0.1,
        format="%.1f",
    )
    prediction_mental_health = st.number_input(
        "Mental health score",
        min_value=int(df["Mental_Health_Score"].min()),
        max_value=int(df["Mental_Health_Score"].max()),
        value=int(df["Mental_Health_Score"].median()),
        step=1,
    )
    prediction_platform = st.selectbox("Most used platform", platform_options)
    predict_clicked = st.button("🔮 Predict Addiction Risk", width="stretch")


# ============================================================
# DATA FILTERING
# ============================================================
filtered_df = df[
    (df["Gender"].isin(selected_gender))
    & (df["Most_Used_Platform"].isin(selected_platform))
    & (df["Academic_Level"].isin(selected_academic))
    & (df["Age"].between(selected_age[0], selected_age[1]))
    & df["Addicted_Score"].between(
        selected_addiction_score[0],
        selected_addiction_score[1],
    )
].copy()

st.sidebar.markdown("---")
st.sidebar.metric("Filtered Records", f"{len(filtered_df):,}")
st.sidebar.caption(f"Original dataset: {len(df):,} records")

if filtered_df.empty:
    st.warning("No records match the selected filters. Adjust the sidebar filters.")
    st.stop()


# ============================================================
# HELPER FUNCTIONS
# ============================================================
def platform_style(platform):
    """Return display initials and gradient for a platform badge."""
    return PLATFORM_STYLES.get(
        platform,
        (platform[:2].upper(), "linear-gradient(135deg, #38bdf8, #2dd4bf)"),
    )


def platform_badge(platform):
    """Build a compact platform logo badge with text."""
    initials, gradient = platform_style(platform)
    return (
        f"<span class='platform-pill'>"
        f"<span class='platform-logo' style='background:{gradient};'>"
        f"{html.escape(initials)}</span>"
        f"<span class='platform-name'>{html.escape(platform)}</span>"
        f"</span>"
    )


def render_platform_strip(platforms):
    """Show selected platforms as premium icon pills."""
    if not platforms:
        return

    badges = "".join(platform_badge(platform) for platform in platforms)
    st.markdown(f"<div class='platform-strip'>{badges}</div>", unsafe_allow_html=True)


def platform_axis_label(platform):
    """Create a readable chart label with compact platform initials."""
    initials, _ = platform_style(platform)
    return f"{initials} | {platform}"


def split_counter_value(value):
    """Split a KPI value into countable number parts when possible."""
    value_text = str(value)
    match = re.search(r"-?\d[\d,]*(?:\.\d+)?", value_text)
    if not match:
        return None

    number_text = match.group(0)
    target = float(number_text.replace(",", ""))
    decimals = len(number_text.split(".")[1]) if "." in number_text else 0

    return {
        "target": target,
        "decimals": decimals,
        "prefix": value_text[: match.start()],
        "suffix": value_text[match.end() :],
    }


def kpi_card(icon, label, value, note):
    """Render a glowing KPI card with animated count-up values."""
    counter = split_counter_value(value)
    icon_html = str(icon)
    escaped_label = html.escape(str(label))
    escaped_value = html.escape(str(value))
    escaped_note = html.escape(str(note))

    if counter:
        animation_script = f"""
            const valueElement = document.querySelector(".kpi-value");
            const target = {counter["target"]};
            const decimals = {counter["decimals"]};
            const prefix = {counter["prefix"]!r};
            const suffix = {counter["suffix"]!r};
            const duration = 1150;
            const startTime = performance.now();

            function formatNumber(number) {{
                return number.toLocaleString(undefined, {{
                    minimumFractionDigits: decimals,
                    maximumFractionDigits: decimals
                }});
            }}

            function animateCounter(now) {{
                const progress = Math.min((now - startTime) / duration, 1);
                const eased = 1 - Math.pow(1 - progress, 3);
                valueElement.textContent = prefix + formatNumber(target * eased) + suffix;

                if (progress < 1) {{
                    requestAnimationFrame(animateCounter);
                }}
            }}

            requestAnimationFrame(animateCounter);
        """
    else:
        animation_script = """
            document.querySelector(".kpi-value").classList.add("text-reveal");
        """

    components.html(
        f"""
        <!doctype html>
        <html>
        <head>
            <style>
                html, body {{
                    margin: 0;
                    padding: 0;
                    background: transparent;
                    font-family: Arial, sans-serif;
                }}

                .glass-card {{
                    box-sizing: border-box;
                    width: 100%;
                    min-height: 132px;
                    background: {panel};
                    border: 1px solid rgba(125, 211, 252, 0.18);
                    border-radius: 16px;
                    padding: 20px 21px;
                    backdrop-filter: blur(18px);
                    box-shadow: 0 16px 42px rgba(0, 0, 0, 0.23);
                    transition: transform 0.22s ease, border-color 0.22s ease, box-shadow 0.22s ease;
                    overflow: hidden;
                    position: relative;
                }}

                .glass-card::after {{
                    content: "";
                    position: absolute;
                    inset: auto -18% -55% 42%;
                    height: 110px;
                    background: radial-gradient(circle, rgba(56, 189, 248, 0.22), transparent 70%);
                    pointer-events: none;
                }}

                .glass-card:hover {{
                    transform: translateY(-5px);
                    border-color: rgba(56, 189, 248, 0.68);
                    box-shadow: 0 22px 58px rgba(14, 165, 233, 0.18);
                }}

                .kpi-icon {{
                    color: #38bdf8;
                    font-size: 1.35rem;
                    margin-bottom: 10px;
                    text-shadow: 0 0 18px rgba(56, 189, 248, 0.72);
                }}

                .kpi-label {{
                    color: {muted_color};
                    font-size: 0.84rem;
                    font-weight: 750;
                    margin-bottom: 8px;
                }}

                .kpi-value {{
                    color: {title_color};
                    font-size: 1.65rem;
                    line-height: 1.18;
                    font-weight: 900;
                    overflow-wrap: anywhere;
                }}

                .kpi-note {{
                    color: {note_color};
                    font-size: 0.78rem;
                    margin-top: 9px;
                }}

                .platform-pill {{
                    display: inline-flex;
                    align-items: center;
                    gap: 8px;
                    padding: 6px 9px;
                    border: 1px solid rgba(125, 211, 252, 0.22);
                    border-radius: 999px;
                    background: {panel};
                    color: {title_color};
                }}

                .platform-logo {{
                    display: inline-grid;
                    place-items: center;
                    width: 24px;
                    height: 24px;
                    border-radius: 8px;
                    color: #ffffff;
                    font-size: 0.68rem;
                    font-weight: 900;
                    letter-spacing: 0;
                }}

                .platform-name {{
                    font-size: 0.76rem;
                    font-weight: 800;
                }}

                .text-reveal {{
                    animation: textReveal 0.9s ease both;
                }}

                @keyframes textReveal {{
                    from {{ opacity: 0; transform: translateY(8px); filter: blur(4px); }}
                    to {{ opacity: 1; transform: translateY(0); filter: blur(0); }}
                }}
            </style>
        </head>
        <body>
            <div class="glass-card">
                <div class="kpi-icon">{icon_html}</div>
                <div class="kpi-label">{escaped_label}</div>
                <div class="kpi-value">{escaped_value}</div>
                <div class="kpi-note">{escaped_note}</div>
            </div>
            <script>
                {animation_script}
            </script>
        </body>
        </html>
        """,
        height=150,
    )


def style_chart(fig, height=430):
    """Apply consistent futuristic Plotly styling."""
    fig.update_layout(
        template=chart_template,
        height=height,
        margin=dict(l=25, r=25, t=62, b=40),
        font=dict(family="Arial", size=13, color=soft_text),
        title_font=dict(size=19, color=text_color),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        hoverlabel=dict(bgcolor=hover_label_bg, font_color="#ffffff", font_size=13),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )
    fig.update_xaxes(gridcolor=grid_color)
    fig.update_yaxes(gridcolor=grid_color)
    return fig


def risk_label(risk_percentage):
    """Convert risk percentage into a readable risk label."""
    if risk_percentage < 40:
        return "Low"
    if risk_percentage < 70:
        return "Moderate"
    return "High"


def render_insight(title, body):
    """Display one automated insight card."""
    st.markdown(
        f"""
        <div class="insight-card">
            <div class="insight-title">{title}</div>
            <div class="insight-body">{body}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def build_insights(data):
    """Generate smart insights from the currently filtered dataset."""
    numeric_data = data.select_dtypes(include=["number"])
    correlation = numeric_data.corr(numeric_only=True)

    sleep_corr = correlation.loc["Sleep_Hours_Per_Night", "Addicted_Score"]
    usage_corr = correlation.loc["Avg_Daily_Usage_Hours", "Addicted_Score"]
    mental_corr = correlation.loc["Mental_Health_Score", "Addicted_Score"]

    platform_risk = (
        data.groupby("Most_Used_Platform", as_index=False)
        .agg(
            Average_Addiction=("Addicted_Score", "mean"),
            Average_Usage=("Avg_Daily_Usage_Hours", "mean"),
            Users=("Most_Used_Platform", "count"),
        )
        .sort_values("Average_Addiction", ascending=False)
    )
    riskiest_platform = platform_risk.iloc[0]

    high_risk_share = (data["Addicted_Score"] >= 7).mean() * 100
    peak_age = (
        data.groupby("Age", as_index=False)["Addicted_Score"]
        .mean()
        .sort_values("Addicted_Score", ascending=False)
        .iloc[0]
    )

    sleep_message = (
        "Lower sleep is associated with higher addiction scores in this filtered view."
        if sleep_corr < -0.15
        else "Sleep and addiction show a weak relationship in this filtered view."
    )
    mental_message = (
        "Lower mental health scores tend to align with higher addiction scores."
        if mental_corr < -0.15
        else "Mental health and addiction have a limited linear relationship here."
    )

    return [
        (
            "💤 Sleep Risk Signal",
            f"{sleep_message} Correlation with addiction score: "
            f"<span class='metric-highlight'>{sleep_corr:.2f}</span>.",
        ),
        (
            "📱 Platform Pattern",
            f"<span class='metric-highlight'>{riskiest_platform['Most_Used_Platform']}</span> "
            f"has the highest average addiction score "
            f"({riskiest_platform['Average_Addiction']:.2f}) and average usage of "
            f"{riskiest_platform['Average_Usage']:.2f} hours.",
        ),
        (
            "🧠 Mental Health Link",
            f"{mental_message} Correlation with addiction score: "
            f"<span class='metric-highlight'>{mental_corr:.2f}</span>.",
        ),
        (
            "⏱ Usage Pressure",
            f"Daily usage has a correlation of "
            f"<span class='metric-highlight'>{usage_corr:.2f}</span> with addiction score. "
            f"High-risk users make up <span class='metric-highlight'>{high_risk_share:.1f}%</span> "
            "of this filtered dataset.",
        ),
        (
            "📈 Age Trend",
            f"Age <span class='metric-highlight'>{int(peak_age['Age'])}</span> has the highest "
            f"average addiction score in the current filters "
            f"({peak_age['Addicted_Score']:.2f}).",
        ),
    ]


# ============================================================
# MACHINE LEARNING
# ============================================================
@st.cache_resource(show_spinner=False)
def train_prediction_model(data):
    """Train a Random Forest model for addiction score prediction."""
    model_data = data[FEATURE_COLUMNS + ["Addicted_Score"]].dropna().copy()
    X = model_data[FEATURE_COLUMNS]
    y = model_data["Addicted_Score"]

    stratify_target = y if y.value_counts().min() >= 2 else None

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=stratify_target,
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", "passthrough", NUMERIC_FEATURES),
            ("platform", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=350,
                    random_state=42,
                    class_weight="balanced",
                ),
            ),
        ]
    )
    model.fit(X_train, y_train)

    test_predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, test_predictions)

    feature_names = model.named_steps["preprocessor"].get_feature_names_out()
    readable_feature_names = [
        feature_name.split("__", 1)[-1].replace("Most_Used_Platform_", "Platform: ")
        for feature_name in feature_names
    ]
    importance_df = pd.DataFrame(
        {
            "Feature": readable_feature_names,
            "Importance": model.named_steps["classifier"].feature_importances_,
        }
    )
    importance_df["Feature"] = importance_df["Feature"].apply(
        lambda value: "Most Used Platform" if value.startswith("Platform: ") else value
    )
    importance_df = (
        importance_df.groupby("Feature", as_index=False)["Importance"]
        .sum()
        .sort_values("Importance", ascending=False)
    )

    prediction_summary = pd.DataFrame(
        {
            "Actual Score": y_test,
            "Predicted Score": test_predictions,
        }
    )

    return model, accuracy, importance_df, prediction_summary


model, model_accuracy, importance_df, prediction_summary = train_prediction_model(df)

prediction_input = pd.DataFrame(
    [
        {
            "Age": prediction_age,
            "Avg_Daily_Usage_Hours": prediction_usage,
            "Sleep_Hours_Per_Night": prediction_sleep,
            "Mental_Health_Score": prediction_mental_health,
            "Most_Used_Platform": prediction_platform,
        }
    ]
)

if predict_clicked:
    predicted_score = int(model.predict(prediction_input)[0])
    class_labels = model.named_steps["classifier"].classes_
    probabilities = model.predict_proba(prediction_input)[0]
    expected_score = float(sum(label * prob for label, prob in zip(class_labels, probabilities)))
    min_score = int(df["Addicted_Score"].min())
    max_score = int(df["Addicted_Score"].max())
    risk_percentage = ((expected_score - min_score) / (max_score - min_score)) * 100
    risk_percentage = max(0, min(100, risk_percentage))

    st.session_state["prediction_result"] = {
        "score": predicted_score,
        "expected_score": expected_score,
        "risk_percentage": risk_percentage,
        "risk_label": risk_label(risk_percentage),
        "input": prediction_input,
    }

if "prediction_result" in st.session_state:
    result = st.session_state["prediction_result"]
    st.sidebar.success(
        f"Risk: {result['risk_percentage']:.1f}% | "
        f"Score: {result['score']} | {result['risk_label']}"
    )


# ============================================================
# HEADER AND KPI VALUES
# ============================================================
render_live_header()

total_users = len(filtered_df)
avg_addiction = filtered_df["Addicted_Score"].mean()
avg_usage = filtered_df["Avg_Daily_Usage_Hours"].mean()
avg_sleep = filtered_df["Sleep_Hours_Per_Night"].mean()
most_used_platform = filtered_df["Most_Used_Platform"].mode()[0]

st.markdown('<div class="section-title">Executive KPI Overview</div>', unsafe_allow_html=True)
kpi_cols = st.columns(5)

with kpi_cols[0]:
    kpi_card("&#128101;", "Total Users", f"{total_users:,}", "Filtered student records")
with kpi_cols[1]:
    kpi_card("&#129504;", "Average Addiction Score", f"{avg_addiction:.2f}", "Mean score")
with kpi_cols[2]:
    kpi_card("&#9201;", "Average Daily Usage", f"{avg_usage:.2f} hrs", "Usage per day")
with kpi_cols[3]:
    kpi_card(platform_badge(most_used_platform), "Most Used Platform", most_used_platform, "Dominant platform")
with kpi_cols[4]:
    kpi_card("&#127769;", "Average Sleep Hours", f"{avg_sleep:.2f} hrs", "Sleep per night")

# ============================================================
# PLATFORM BADGES
# ============================================================
st.markdown('<div class="section-title">Active Platform Mix</div>', unsafe_allow_html=True)
render_platform_strip(sorted(filtered_df["Most_Used_Platform"].dropna().unique()))


# ============================================================
# NAVIGATION
# ============================================================
overview_tab, analytics_tab, prediction_tab, insights_tab, dataset_tab = st.tabs(
    ["🏠 Overview", "📊 Analytics", "🔮 AI Prediction", "💡 Insights", "🗂 Dataset"]
)


# ============================================================
# OVERVIEW
# ============================================================
with overview_tab:
    st.markdown('<div class="section-title">Snapshot</div>', unsafe_allow_html=True)
    metric_1, metric_2, metric_3, metric_4 = st.columns(4)
    high_risk_users = int((filtered_df["Addicted_Score"] >= 7).sum())
    avg_conflicts = filtered_df["Conflicts_Over_Social_Media"].mean()

    metric_1.metric("High-Risk Users", f"{high_risk_users:,}")
    metric_2.metric("Avg Conflicts", f"{avg_conflicts:.2f}")
    metric_3.metric("Countries", f"{filtered_df['Country'].nunique():,}")
    metric_4.metric("Platforms", f"{filtered_df['Most_Used_Platform'].nunique():,}")

    overview_col_1, overview_col_2 = st.columns([1.15, 1])

    with overview_col_1:
        st.markdown('<div class="section-title">Filtered Summary Statistics</div>', unsafe_allow_html=True)
        summary_columns = [
            "Age",
            "Avg_Daily_Usage_Hours",
            "Sleep_Hours_Per_Night",
            "Mental_Health_Score",
            "Conflicts_Over_Social_Media",
            "Addicted_Score",
        ]
        st.dataframe(filtered_df[summary_columns].describe().round(2), width="stretch")

    with overview_col_2:
        platform_share = filtered_df["Most_Used_Platform"].value_counts().reset_index()
        platform_share.columns = ["Platform", "Users"]
        platform_share["Platform Display"] = platform_share["Platform"].apply(platform_axis_label)
        fig = px.pie(
            platform_share,
            names="Platform Display",
            values="Users",
            hole=0.48,
            title="Platform Share",
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        fig.update_traces(
            textposition="inside",
            textinfo="percent+label",
            hovertemplate="<b>%{label}</b><br>Users: %{value}<br>Share: %{percent}<extra></extra>",
        )
        st.plotly_chart(style_chart(fig, height=390), width="stretch")

    with st.expander("🧾 Quick Dataset Preview", expanded=False):
        st.dataframe(filtered_df.head(20), width="stretch")


# ============================================================
# ANALYTICS
# ============================================================
with analytics_tab:
    st.markdown('<div class="section-title">Interactive Plotly Analytics</div>', unsafe_allow_html=True)
    render_platform_strip(sorted(filtered_df["Most_Used_Platform"].dropna().unique()))

    row1_col1, row1_col2 = st.columns(2)

    with row1_col1:
        platform_counts = filtered_df["Most_Used_Platform"].value_counts().reset_index()
        platform_counts.columns = ["Platform", "Users"]
        platform_counts["Platform Display"] = platform_counts["Platform"].apply(platform_axis_label)
        fig = px.bar(
            platform_counts,
            x="Platform Display",
            y="Users",
            color="Platform Display",
            text="Users",
            title="Platform Usage Analysis",
            color_discrete_sequence=px.colors.qualitative.Bold,
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(showlegend=False, xaxis_title="Platform")
        st.plotly_chart(style_chart(fig), width="stretch")

    with row1_col2:
        fig = px.histogram(
            filtered_df,
            x="Addicted_Score",
            nbins=10,
            title="Addiction Score Distribution",
            color_discrete_sequence=["#38bdf8"],
        )
        fig.update_layout(xaxis_title="Addiction Score", yaxis_title="Users")
        st.plotly_chart(style_chart(fig), width="stretch")

    row2_col1, row2_col2 = st.columns(2)

    with row2_col1:
        fig = px.scatter(
            filtered_df,
            x="Avg_Daily_Usage_Hours",
            y="Sleep_Hours_Per_Night",
            color="Addicted_Score",
            size="Addicted_Score",
            hover_data=["Age", "Gender", "Most_Used_Platform", "Mental_Health_Score"],
            title="Sleep vs Daily Usage",
            color_continuous_scale="Turbo",
        )
        fig.update_layout(
            xaxis_title="Daily Usage Hours",
            yaxis_title="Sleep Hours Per Night",
        )
        st.plotly_chart(style_chart(fig), width="stretch")

    with row2_col2:
        fig = px.scatter(
            filtered_df,
            x="Mental_Health_Score",
            y="Addicted_Score",
            color="Most_Used_Platform",
            size="Avg_Daily_Usage_Hours",
            hover_data=["Age", "Gender", "Sleep_Hours_Per_Night"],
            title="Mental Health Analysis",
        )
        fig.update_layout(
            xaxis_title="Mental Health Score",
            yaxis_title="Addiction Score",
        )
        st.plotly_chart(style_chart(fig), width="stretch")

    row3_col1, row3_col2 = st.columns(2)

    with row3_col1:
        gender_summary = (
            filtered_df.groupby("Gender", as_index=False)
            .agg(
                Average_Addiction=("Addicted_Score", "mean"),
                Average_Usage=("Avg_Daily_Usage_Hours", "mean"),
                Users=("Gender", "count"),
            )
            .sort_values("Average_Addiction", ascending=False)
        )
        fig = px.bar(
            gender_summary,
            x="Gender",
            y=["Average_Addiction", "Average_Usage"],
            barmode="group",
            hover_data=["Users"],
            title="Gender Comparison",
            color_discrete_sequence=["#38bdf8", "#2dd4bf"],
        )
        fig.update_layout(yaxis_title="Average Value", legend_title="Metric")
        st.plotly_chart(style_chart(fig), width="stretch")

    with row3_col2:
        age_trend = (
            filtered_df.groupby("Age", as_index=False)
            .agg(
                Average_Addiction=("Addicted_Score", "mean"),
                Average_Usage=("Avg_Daily_Usage_Hours", "mean"),
            )
            .sort_values("Age")
        )
        fig = px.line(
            age_trend,
            x="Age",
            y=["Average_Addiction", "Average_Usage"],
            markers=True,
            title="Trend Chart by Age",
            color_discrete_sequence=["#f472b6", "#38bdf8"],
        )
        fig.update_layout(yaxis_title="Average Value", legend_title="Metric")
        st.plotly_chart(style_chart(fig), width="stretch")

    with st.expander("🔥 Correlation Matrix", expanded=True):
        numeric_df = filtered_df.select_dtypes(include=["number"])
        correlation = numeric_df.corr(numeric_only=True).round(2)
        fig = go.Figure(
            data=go.Heatmap(
                z=correlation.values,
                x=correlation.columns,
                y=correlation.columns,
                colorscale="RdBu",
                zmin=-1,
                zmax=1,
                text=correlation.values,
                texttemplate="%{text}",
                hovertemplate="<b>%{y}</b> vs <b>%{x}</b><br>Correlation: %{z}<extra></extra>",
            )
        )
        fig.update_layout(title="Heatmap Correlation Matrix")
        st.plotly_chart(style_chart(fig, height=560), width="stretch")


# ============================================================
# AI PREDICTION
# ============================================================
with prediction_tab:
    st.markdown('<div class="section-title">AI Prediction System</div>', unsafe_allow_html=True)

    with st.expander("🤖 Model Details", expanded=True):
        st.write(
            "The RandomForestClassifier predicts `Addicted_Score` from age, daily usage, "
            "sleep hours, mental health score, and most used platform. The risk percentage "
            "uses the model probability distribution across score classes."
        )

    pred_col1, pred_col2 = st.columns([1.05, 1])

    with pred_col1:
        if "prediction_result" in st.session_state:
            result = st.session_state["prediction_result"]
            st.markdown(
                f"""
                <div class="prediction-panel">
                    <div class="kpi-label">Prediction Result</div>
                    <div class="prediction-score">
                        Score {result["score"]} · {result["risk_label"]} Risk
                    </div>
                    <div class="prediction-meta">
                        Addiction risk percentage:
                        <span class="metric-highlight">{result["risk_percentage"]:.1f}%</span>
                    </div>
                    <div class="prediction-meta">
                        Expected model score: {result["expected_score"]:.2f}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.info("Enter values in the sidebar and click Predict Addiction Risk.")

        metric_a, metric_b = st.columns(2)
        metric_a.metric("Model Accuracy", f"{model_accuracy * 100:.2f}%")
        exact_predictions = (
            prediction_summary["Actual Score"] == prediction_summary["Predicted Score"]
        ).sum()
        metric_b.metric("Correct Test Predictions", f"{exact_predictions}/{len(prediction_summary)}")

        with st.expander("Prediction Input Values", expanded=False):
            st.dataframe(prediction_input, width="stretch")

    with pred_col2:
        risk_value = (
            st.session_state["prediction_result"]["risk_percentage"]
            if "prediction_result" in st.session_state
            else 0
        )
        fig = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=risk_value,
                number={"suffix": "%", "font": {"size": 46, "color": title_color}},
                title={
                    "text": "Addiction Risk Gauge",
                    "font": {"size": 20, "color": soft_text},
                },
                gauge={
                    "axis": {"range": [0, 100], "tickcolor": soft_text},
                    "bar": {"color": "#38bdf8"},
                    "bgcolor": panel,
                    "borderwidth": 1,
                    "bordercolor": "rgba(125, 211, 252, 0.35)",
                    "steps": [
                        {"range": [0, 40], "color": "rgba(45, 212, 191, 0.35)"},
                        {"range": [40, 70], "color": "rgba(245, 158, 11, 0.38)"},
                        {"range": [70, 100], "color": "rgba(244, 114, 182, 0.42)"},
                    ],
                    "threshold": {
                        "line": {"color": "#ffffff", "width": 4},
                        "thickness": 0.78,
                        "value": risk_value,
                    },
                },
            )
        )
        st.plotly_chart(style_chart(fig, height=420), width="stretch")

    st.markdown('<div class="section-title">Model Feature Importance</div>', unsafe_allow_html=True)
    fig = px.bar(
        importance_df,
        x="Importance",
        y="Feature",
        orientation="h",
        title="Random Forest Feature Importance",
        color="Importance",
        color_continuous_scale="Teal",
    )
    fig.update_layout(yaxis=dict(autorange="reversed"))
    st.plotly_chart(style_chart(fig, height=440), width="stretch")

    with st.expander("Prediction Sample", expanded=False):
        st.dataframe(prediction_summary.head(25), width="stretch")


# ============================================================
# AI INSIGHTS
# ============================================================
with insights_tab:
    st.markdown('<div class="section-title">AI-Generated Smart Insights</div>', unsafe_allow_html=True)

    for title, body in build_insights(filtered_df):
        render_insight(title, body)

    platform_profile = (
        filtered_df.groupby("Most_Used_Platform", as_index=False)
        .agg(
            Users=("Most_Used_Platform", "count"),
            Average_Addiction=("Addicted_Score", "mean"),
            Average_Usage=("Avg_Daily_Usage_Hours", "mean"),
            Average_Sleep=("Sleep_Hours_Per_Night", "mean"),
            Average_Mental_Health=("Mental_Health_Score", "mean"),
        )
        .sort_values("Average_Addiction", ascending=False)
        .round(2)
    )

    with st.expander("📱 Platform Risk Profile", expanded=True):
        st.dataframe(platform_profile, width="stretch")


# ============================================================
# DATASET
# ============================================================
with dataset_tab:
    st.markdown('<div class="section-title">Dataset Workspace</div>', unsafe_allow_html=True)

    csv_data = filtered_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Download Filtered Dataset CSV",
        data=csv_data,
        file_name="filtered_social_media_addiction_data.csv",
        mime="text/csv",
        width="stretch",
    )

    st.dataframe(filtered_df, width="stretch")

    with st.expander("Dataset Information", expanded=False):
        info_col1, info_col2 = st.columns(2)
        with info_col1:
            st.write("Dataset Shape")
            st.write(f"Rows: {filtered_df.shape[0]}")
            st.write(f"Columns: {filtered_df.shape[1]}")
        with info_col2:
            st.write("Missing Values")
            missing_values = filtered_df.isnull().sum().reset_index()
            missing_values.columns = ["Column", "Missing Values"]
            st.dataframe(missing_values, width="stretch")

    with st.expander("Full Summary Statistics", expanded=False):
        full_summary = filtered_df.describe(include="all").round(2).astype(str)
        st.dataframe(full_summary, width="stretch")
