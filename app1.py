from datetime import datetime
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import shap
from backend.core.data import load_dataset, split_dataset
from backend.core.model import load_model, load_features
from backend.core.simulation import get_random_packet, predict, simulate_window
from backend.core.evaluation import evaluate
from backend.services.SHAP_explainer import create_explainer, generate_shap_analysis

# File paths
MODEL_PATH   = "backend/model/rf_model.pkl"
FEATURE_PATH = "backend/model/rf_features.pkl"
DATA_FILE    = "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"

st.set_page_config(
    page_title="AI-NIDS",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Global CSS injection
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;600&family=Space+Grotesk:wght@300;400;500;600;700&family=Outfit:wght@300;400;500;600;700;800&display=swap');

:root {
    --bg-primary: #080D14;
    --bg-secondary: #0D1520;
    --bg-card: #101A27;
    --bg-card-hover: #152030;
    --border-subtle: rgba(99, 179, 237, 0.1);
    --border-default: rgba(99, 179, 237, 0.2);
    --border-active: rgba(99, 179, 237, 0.5);
    --accent-blue: #63B3ED;
    --accent-blue-bright: #90CDF4;
    --accent-green: #68D391;
    --accent-yellow: #F6AD55;
    --accent-red: #FC8181;
    --text-primary: #E2EBF5;
    --text-secondary: #A0B4C8;
    --text-muted: #4A6480;
    --text-dim: #2D4A60;
    --mono: 'JetBrains Mono', monospace;
    --sans: 'Outfit', sans-serif;
}

html, body, [class*="css"] {
    font-family: var(--sans) !important;
}

.stApp {
    background-color: var(--bg-primary);
    background-image:
        radial-gradient(ellipse 80% 50% at 50% -20%, rgba(99,179,237,0.06) 0%, transparent 60%);
}

#MainMenu, footer, header { visibility: hidden; }
.block-container { 
    padding: 0 3rem 5rem !important; 
    max-width: 1400px !important; 
}

/* Dividers */
hr { 
    border-color: var(--border-subtle) !important; 
    margin: 2rem 0 !important; 
}

/* ── Metric Cards ── */
[data-testid="metric-container"] {
    background: var(--bg-card);
    border: 1px solid var(--border-subtle);
    border-radius: 14px;
    padding: 24px 28px !important;
    position: relative;
    overflow: hidden;
    transition: all 0.25s ease;
    height: 115px;
    display: flex;
    flex-direction: column;
    justify-content: center;
}
[data-testid="metric-container"]:hover {
    border-color: var(--border-default);
    background: var(--bg-card-hover);
    transform: translateY(-1px);
    box-shadow: 0 8px 32px rgba(0,0,0,0.3);
}
[data-testid="metric-container"]::after {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(99,179,237,0.4), transparent);
}
[data-testid="metric-container"] label {
    font-family: var(--mono) !important;
    font-size: 0.6rem !important;
    letter-spacing: 0.18em !important;
    text-transform: uppercase !important;
    color: var(--text-muted) !important;
    font-weight: 500 !important;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    font-family: var(--sans) !important;
    font-weight: 700 !important;
    font-size: 2.2rem !important;
    color: var(--text-primary) !important;
    line-height: 1.2 !important;
    letter-spacing: -0.02em !important;
}

/* ── Buttons ── */
.stButton > button {
    font-family: var(--mono) !important;
    font-size: 0.66rem !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
    background: rgba(99,179,237,0.06) !important;
    color: var(--accent-blue-bright) !important;
    border: 1px solid var(--border-default) !important;
    border-radius: 8px !important;
    padding: 11px 22px !important;
    transition: all 0.2s ease !important;
    width: 100% !important;
    height: 42px !important;
    font-weight: 500 !important;
}
.stButton > button:hover {
    background: rgba(99,179,237,0.12) !important;
    border-color: var(--border-active) !important;
    box-shadow: 0 0 20px rgba(99,179,237,0.15) !important;
    transform: translateY(-1px) !important;
}
.stButton > button:active { 
    transform: translateY(0) !important; 
}

.btn-danger > button {
    background: rgba(252,129,129,0.05) !important;
    color: var(--accent-red) !important;
    border-color: rgba(252,129,129,0.25) !important;
}
.btn-danger > button:hover {
    background: rgba(252,129,129,0.1) !important;
    border-color: rgba(252,129,129,0.5) !important;
    box-shadow: 0 0 20px rgba(252,129,129,0.12) !important;
}

/* ── DataFrames & Tables ── */
.stDataFrame {
    border: 1px solid var(--border-subtle) !important;
    border-radius: 12px !important;
    overflow: hidden;
}
.stDataFrame thead th {
    font-family: var(--mono) !important;
    background: var(--bg-secondary) !important;
    color: var(--text-muted) !important;
    font-size: 0.6rem !important;
    letter-spacing: 0.14em !important;
    text-transform: uppercase !important;
    font-weight: 500 !important;
    padding: 12px 16px !important;
}
.stDataFrame tbody td {
    font-family: var(--mono) !important;
    font-size: 0.75rem !important;
    color: var(--text-secondary) !important;
    padding: 10px 16px !important;
}
.stDataFrame tbody tr:hover { 
    background: rgba(99,179,237,0.03) !important; 
}

.stTable {
    font-family: var(--mono) !important;
    font-size: 0.75rem !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: 12px !important;
    overflow: hidden;
}
.stTable th {
    background: var(--bg-secondary) !important;
    color: var(--text-muted) !important;
    font-size: 0.6rem !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
    padding: 12px 16px !important;
    font-weight: 500 !important;
}
.stTable td {
    color: var(--text-secondary) !important;
    padding: 10px 16px !important;
    border-bottom: 1px solid var(--border-subtle) !important;
}

/* Spinner */
.stSpinner > div { border-top-color: var(--accent-blue) !important; }

/* Headings */
h1, h2, h3 {
    font-family: var(--sans) !important;
    color: var(--text-primary) !important;
    font-weight: 700 !important;
}
</style>
""", unsafe_allow_html=True)


# Section heading
def section_header(title: str, subtitle: str = ""):
    sub_html = (
        f"<span style='font-family:var(--mono,JetBrains Mono,monospace);font-size:0.6rem;"
        f"color:var(--text-dim,#2D4A60);margin-left:14px;letter-spacing:0.12em;font-weight:400;"
        f"text-transform:uppercase;'>{subtitle}</span>"
    ) if subtitle else ""
    st.markdown(f"""
    <div style="margin:2.2rem 0 1.2rem;display:flex;align-items:baseline;
                border-left:2px solid rgba(99,179,237,0.6);padding-left:16px;">
        <span style="font-family:'Outfit',sans-serif;font-weight:700;font-size:1.05rem;
                     color:#E2EBF5;letter-spacing:0.01em;">{title}</span>
        {sub_html}
    </div>
    """, unsafe_allow_html=True)


# Divider
def neon_divider():
    st.markdown("""
    <div style="height:1px;margin:2rem 0;
        background:linear-gradient(90deg,transparent,
        rgba(99,179,237,0.18) 30%,rgba(99,179,237,0.06) 70%,transparent);">
    </div>
    """, unsafe_allow_html=True)


# Severity badge
def severity_chip(severity: str):
    palette = {
        "HIGH":   ("#FC8181", "rgba(252,129,129,0.1)",  "rgba(252,129,129,0.3)"),
        "MEDIUM": ("#F6AD55", "rgba(246,173,85,0.1)",   "rgba(246,173,85,0.3)"),
        "LOW":    ("#68D391", "rgba(104,211,145,0.08)", "rgba(104,211,145,0.28)"),
    }
    c, bg, border = palette.get(severity, ("#63B3ED", "rgba(99,179,237,0.08)", "rgba(99,179,237,0.3)"))
    st.markdown(f"""
    <style>@keyframes blink{{0%,100%{{opacity:1}}50%{{opacity:0.15}}}}</style>
    <div style="margin-top:1.1rem;">
        <span style="display:inline-flex;align-items:center;gap:8px;
            font-family:'JetBrains Mono',monospace;font-size:0.66rem;font-weight:500;
            letter-spacing:0.12em;color:{c};background:{bg};
            border:1px solid {border};border-radius:20px;padding:7px 18px;">
            <span style="width:5px;height:5px;border-radius:50%;background:{c};
                display:inline-block;animation:blink 1.6s step-start infinite;"></span>
            {severity} SEVERITY
        </span>
    </div>
    """, unsafe_allow_html=True)


#Status banner
def status_banner(severity):
    cfg = {
        None:     {"bg":"rgba(99,179,237,0.04)","border":"rgba(99,179,237,0.15)",
                   "color":"#4A6480","icon":"◈",
                   "msg":"SYSTEM INITIALIZED — AWAITING TRAFFIC SIMULATION"},
        "LOW":    {"bg":"rgba(104,211,145,0.05)","border":"rgba(104,211,145,0.25)",
                   "color":"#68D391","icon":"✓",
                   "msg":"GATEWAY STATUS: STABLE — ALL SYSTEMS NOMINAL"},
        "MEDIUM": {"bg":"rgba(246,173,85,0.05)","border":"rgba(246,173,85,0.28)",
                   "color":"#F6AD55","icon":"⚠",
                   "msg":"GATEWAY STATUS: ELEVATED RISK — MONITORING ACTIVE"},
        "HIGH":   {"bg":"rgba(252,129,129,0.06)","border":"rgba(252,129,129,0.4)",
                   "color":"#FC8181","icon":"◉",
                   "msg":"GATEWAY STATUS: HIGH THREAT DETECTED — IMMEDIATE ACTION REQUIRED"},
    }
    c = cfg.get(severity, cfg[None])
    anim = "animation:flash 1.8s ease-in-out infinite;" if severity == "HIGH" else ""
    st.markdown(f"""
    <style>@keyframes flash{{0%,100%{{opacity:1}}50%{{opacity:0.55}}}}</style>
    <div style="background:{c['bg']};border:1px solid {c['border']};
        border-radius:12px;padding:14px 24px;margin-bottom:1.6rem;
        font-family:'JetBrains Mono',monospace;font-size:0.68rem;font-weight:500;
        color:{c['color']};letter-spacing:0.08em;
        display:flex;align-items:center;gap:14px;{anim}">
        <span style="font-size:0.9rem;opacity:0.9;">{c['icon']}</span>
        {c['msg']}
    </div>
    """, unsafe_allow_html=True)


#Classification result card
def result_card(benign: bool):
    if benign:
        st.markdown("""
        <div style="background:rgba(104,211,145,0.05);border:1px solid rgba(104,211,145,0.25);
            border-radius:14px;padding:32px 24px;text-align:center;height:180px;
            display:flex;flex-direction:column;align-items:center;justify-content:center;">
            <div style="width:44px;height:44px;border-radius:50%;
                background:rgba(104,211,145,0.12);border:1.5px solid rgba(104,211,145,0.4);
                display:flex;align-items:center;justify-content:center;
                font-size:1.1rem;margin-bottom:14px;">✓</div>
            <div style="font-family:'Outfit',sans-serif;font-weight:700;font-size:1.15rem;
                color:#68D391;letter-spacing:0.05em;">BENIGN TRAFFIC</div>
            <div style="font-family:'JetBrains Mono',monospace;font-size:0.57rem;
                color:rgba(104,211,145,0.45);margin-top:8px;letter-spacing:0.14em;">
                NO THREAT DETECTED</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <style>@keyframes threat{0%,100%{opacity:1}50%{opacity:0.55}}</style>
        <div style="background:rgba(252,129,129,0.06);border:1px solid rgba(252,129,129,0.4);
            border-radius:14px;padding:32px 24px;text-align:center;height:180px;
            display:flex;flex-direction:column;align-items:center;justify-content:center;
            animation:threat 1.8s ease-in-out infinite;">
            <div style="width:44px;height:44px;border-radius:50%;
                background:rgba(252,129,129,0.1);border:1.5px solid rgba(252,129,129,0.4);
                display:flex;align-items:center;justify-content:center;
                font-size:1.1rem;margin-bottom:14px;color:#FC8181;">⚠</div>
            <div style="font-family:'Outfit',sans-serif;font-weight:700;font-size:1.15rem;
                color:#FC8181;letter-spacing:0.05em;">MALICIOUS TRAFFIC</div>
            <div style="font-family:'JetBrains Mono',monospace;font-size:0.57rem;
                color:rgba(252,129,129,0.45);margin-top:8px;letter-spacing:0.14em;">
                ATTACK DETECTED — REVIEW SHAP DRIVERS</div>
        </div>
        """, unsafe_allow_html=True)


# SHAP feature row
def feature_card(feature: str, impact: float, is_attack: bool):
    c      = "#FC8181" if is_attack else "#68D391"
    bg     = "rgba(252,129,129,0.05)" if is_attack else "rgba(104,211,145,0.04)"
    border = "rgba(252,129,129,0.2)"  if is_attack else "rgba(104,211,145,0.18)"
    sign   = f"+{impact:.4f}" if impact > 0 else f"{impact:.4f}"
    st.markdown(f"""
    <div style="background:{bg};border:1px solid {border};
        border-left:2px solid {c};border-radius:8px;
        padding:12px 16px;margin-bottom:8px;
        display:flex;align-items:center;justify-content:space-between;min-height:44px;">
        <span style="font-family:'JetBrains Mono',monospace;font-size:0.72rem;
            color:{c};font-weight:400;letter-spacing:0.01em;">
            {feature}
        </span>
        <span style="font-family:'JetBrains Mono',monospace;font-size:0.67rem;
                     color:{c};opacity:0.6;margin-left:16px;white-space:nowrap;font-weight:500;">
            {sign}
        </span>
    </div>
    """, unsafe_allow_html=True)


# Empty state
def empty_state(msg: str):
    st.markdown(f"""
    <div style="border:1px dashed rgba(99,179,237,0.1);border-radius:12px;
        padding:40px;text-align:center;margin-top:0.5rem;
        font-family:'JetBrains Mono',monospace;font-size:0.67rem;
        color:#2D4A60;letter-spacing:0.14em;">{msg}</div>
    """, unsafe_allow_html=True)


# Navbar
st.markdown(f"""
<div style="display:flex;align-items:center;justify-content:space-between;
    padding:26px 0 22px;border-bottom:1px solid rgba(99,179,237,0.08);
    margin-bottom:2rem;">
    <div style="display:flex;align-items:center;gap:16px;">
        <div style="width:42px;height:42px;border:1px solid rgba(99,179,237,0.4);
            border-radius:11px;display:flex;align-items:center;justify-content:center;
            background:rgba(99,179,237,0.06);
            box-shadow:0 0 20px rgba(99,179,237,0.1);">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none"
                 stroke="#63B3ED" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round">
                <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>
                <path d="M9 12l2 2 4-4"/>
            </svg>
        </div>
        <div>
            <div style="font-family:'Outfit',sans-serif;font-weight:800;
                font-size:1.05rem;letter-spacing:0.2em;color:#E2EBF5;">AI-NIDS</div>
            <div style="font-family:'JetBrains Mono',monospace;font-size:0.55rem;
                color:#2D4A60;letter-spacing:0.14em;margin-top:2px;">
                IOT GATEWAY INTRUSION DETECTION SYSTEM</div>
        </div>
    </div>
    <div style="display:flex;align-items:center;gap:14px;">
        <div style="font-family:'JetBrains Mono',monospace;font-size:0.62rem;
            color:#4A6480;border:1px solid rgba(99,179,237,0.1);background:rgba(99,179,237,0.03);
            padding:7px 14px;border-radius:7px;letter-spacing:0.07em;">
            {datetime.now().strftime("%Y-%m-%d  %H:%M:%S")}
        </div>
        <div style="font-family:'JetBrains Mono',monospace;font-size:0.57rem;
            color:#2D4A60;letter-spacing:0.1em;
            border:1px solid rgba(99,179,237,0.07);padding:7px 12px;border-radius:7px;">
            RF MODEL · v1.0
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# Session init
if "alert_log" not in st.session_state:
    st.session_state["alert_log"] = []

last_severity = st.session_state.get("last_event", {}).get("severity", None)
status_banner(last_severity)

# KPIs
section_header("System Overview")

total_events = len(st.session_state["alert_log"])
high_alerts  = sum(1 for e in st.session_state["alert_log"] if e["severity"] == "HIGH")
avg_risk     = (
    round(sum(e["mean_risk_score"] for e in st.session_state["alert_log"]) / total_events, 2)
    if total_events > 0 else 0.00
)

k1, k2, k3 = st.columns(3)
k1.metric("Windows Analyzed", total_events)
k2.metric("High Severity Alerts", high_alerts)
k3.metric("Avg Risk Score", avg_risk)

neon_divider()

#Load resources
model         = load_model(MODEL_PATH)
feature_names = load_features(FEATURE_PATH)
explainer     = create_explainer(model)

if not model or not feature_names:
    st.markdown("""
    <div style="background:rgba(252,129,129,0.06);border:1px solid rgba(252,129,129,0.3);
        border-radius:10px;padding:16px 24px;font-family:'JetBrains Mono',monospace;
        font-size:0.72rem;color:#FC8181;letter-spacing:0.08em;">
        ⚠ MODEL OR FEATURE SCHEMA NOT FOUND — CHECK BACKEND PATHS
    </div>
    """, unsafe_allow_html=True)
    st.stop()

df = load_dataset(DATA_FILE)
X_train, X_test, y_train, y_test = split_dataset(df, feature_names)


#Section 1 Simulation 
section_header("Traffic Analysis Simulation", "sliding window engine")

btn_c1, btn_c2, _ = st.columns([1, 1, 4])

with btn_c1:
    run_sim = st.button("▶  Run Simulation", key="run_sim", use_container_width=True)

with btn_c2:
    st.markdown('<div class="btn-danger">', unsafe_allow_html=True)
    reset_sim = st.button("↺  Reset", key="reset_sim", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

if run_sim:
    with st.spinner("Analyzing traffic window..."):
        event = simulate_window(model, X_test, y_test)
    st.session_state["last_event"] = event
    st.session_state["alert_log"].append(event)
    st.rerun()

if reset_sim:
    st.session_state["alert_log"] = []
    st.session_state.pop("last_event", None)
    st.rerun()

if "last_event" in st.session_state:
    event = st.session_state["last_event"]
    st.markdown("<div style='margin-top:1.8rem;'></div>", unsafe_allow_html=True)

    wk1, wk2, wk3, wk4 = st.columns(4)
    wk1.metric("Packets Analyzed",  event["window_size"])
    wk2.metric("Malicious Packets", event["attack_count"])
    wk3.metric("Mean Risk Score",   f"{event['mean_risk_score']:.3f}")
    wk4.metric("Alert Triggered",   "YES" if event["alert_triggered"] else "NO")

    severity_chip(event["severity"])


# Section 2 Chart + Log
neon_divider()

THRESHOLD = 0.6

if st.session_state["alert_log"]:
    log_df = pd.DataFrame(st.session_state["alert_log"])
    section_header("Risk Score Trend", f"{len(log_df)} windows captured")

    # ── Dynamic Y-axis: start capped at threshold; expand when scores exceed it ──
    max_score = log_df["mean_risk_score"].max()
    if max_score > THRESHOLD:
        y_max = min(1.05, max_score + 0.08)
    else:
        # Cap slightly above threshold so chart always shows the threshold line
        y_max = THRESHOLD + 0.08

    fig = go.Figure()

    # Shaded fill
    fig.add_trace(go.Scatter(
        x=log_df["timestamp"], y=log_df["mean_risk_score"],
        fill="tozeroy", fillcolor="rgba(99,179,237,0.04)",
        line=dict(color="rgba(0,0,0,0)"),
        showlegend=False, hoverinfo="skip",
    ))

    severity_colors = log_df["severity"].map(
        {"LOW": "#68D391", "MEDIUM": "#F6AD55", "HIGH": "#FC8181"}
    ).fillna("#63B3ED")

    fig.add_trace(go.Scatter(
        x=log_df["timestamp"],
        y=log_df["mean_risk_score"],
        mode="lines+markers",
        line=dict(color="#63B3ED", width=2),
        marker=dict(
            color=severity_colors, size=8,
            line=dict(color="#080D14", width=2)
        ),
        name="Risk Score",
        hovertemplate="<b>%{x}</b><br>Risk: %{y:.3f}<extra></extra>",
    ))

    fig.add_hline(
        y=THRESHOLD, line_dash="dot",
        line_color="rgba(252,129,129,0.5)", line_width=1.5,
        annotation_text="Alert Threshold (0.6)",
        annotation_font=dict(color="#FC8181", family="JetBrains Mono", size=10),
        annotation_position="top left",
    )

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(13,21,32,0.95)",
        font=dict(family="JetBrains Mono", color="#4A6480", size=10),
        margin=dict(t=16, b=44, l=58, r=24),
        height=280,
        xaxis=dict(
            gridcolor="rgba(99,179,237,0.05)",
            linecolor="rgba(99,179,237,0.1)",
            tickfont=dict(size=9, color="#4A6480"),
            tickangle=-20,
        ),
        yaxis=dict(
            gridcolor="rgba(99,179,237,0.05)",
            linecolor="rgba(99,179,237,0.1)",
            tickfont=dict(size=9, color="#4A6480"),
            range=[0, y_max],
            dtick=0.1,
        ),
        showlegend=False,
        hoverlabel=dict(
            bgcolor="#0D1520",
            font=dict(family="JetBrains Mono", color="#90CDF4", size=11),
            bordercolor="rgba(99,179,237,0.25)",
        ),
    )

    st.plotly_chart(fig, use_container_width=True)

    # Alert log 
    section_header("Gateway Alert Log", "sorted by most recent")

    st.dataframe(
        log_df.sort_values(by="timestamp", ascending=False),
        use_container_width=True,
        hide_index=True,
        column_config={
            "timestamp":       st.column_config.TextColumn("Timestamp"),
            "window_size":     st.column_config.NumberColumn("Window Size",  format="%d"),
            "attack_count":    st.column_config.NumberColumn("Attacks",      format="%d"),
            "mean_risk_score": st.column_config.NumberColumn("Risk Score",   format="%.3f"),
            "severity":        st.column_config.TextColumn("Severity"),
            "alert_triggered": st.column_config.CheckboxColumn("Alert"),
        },
    )

else:
    empty_state("NO ALERTS LOGGED — RUN A SIMULATION TO POPULATE THE LOG")


# Section 3 Packet Analysis 
neon_divider()
section_header("Real-Time Packet Analysis", "SHAP-powered classification")

pc1, pc2, _ = st.columns([1, 1, 4])

with pc1:
    if st.button("⬡  Capture Random Packet", use_container_width=True):
        packet, actual = get_random_packet(X_test, y_test)
        st.session_state["packet"] = packet
        st.session_state["actual"] = actual

with pc2:
    if "packet" in st.session_state:
        st.markdown('<div class="btn-danger">', unsafe_allow_html=True)
        if st.button("✕  Close Analysis", use_container_width=True):
            st.session_state.pop("packet", None)
            st.session_state.pop("actual", None)
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)


if "packet" in st.session_state:
    packet     = st.session_state["packet"]
    prediction = predict(model, packet)
    packet_df  = packet.to_frame().T

    neon_divider()

    important_features = [
        "Source Port", "Destination Port", "Protocol", "Flow Duration",
        "Total Fwd Packets", "Total Backward Packets",
        "Total Length of Fwd Packets", "Total Length of Bwd Packets",
    ]
    available = [f for f in important_features if f in packet_df.columns]

    left, right = st.columns([1.3, 1])

    with left:
        section_header("Packet Summary")
        st.table(packet_df[available].T)

    with right:
        section_header("Classification Result")
        st.markdown("<div style='margin-top:0.5rem;'></div>", unsafe_allow_html=True)
        result_card(prediction == 0)

    shap_vector, explanation_text = generate_shap_analysis(
        explainer, packet_df, feature_names, prediction
    )

    impact_df = pd.DataFrame({"Feature": feature_names, "Impact": shap_vector})
    impact_df["AbsImpact"] = impact_df["Impact"].abs()
    impact_df = impact_df.sort_values(by="AbsImpact", ascending=False).head(5)

    positive_features = impact_df[impact_df["Impact"] > 0]
    negative_features = impact_df[impact_df["Impact"] <= 0]

    neon_divider()
    section_header("Key Risk Drivers", "top-5 SHAP features")

    d1, d2 = st.columns(2)

    with d1:
        st.markdown("""
        <div style="font-family:'JetBrains Mono',monospace;font-size:0.6rem;font-weight:500;
            letter-spacing:0.15em;color:#FC8181;margin-bottom:14px;margin-top:4px;
            text-transform:uppercase;">
            Attack Drivers
        </div>
        """, unsafe_allow_html=True)
        if not positive_features.empty:
            for _, row in positive_features.iterrows():
                feature_card(row["Feature"], row["Impact"], is_attack=True)
        else:
            st.markdown(
                "<span style='font-family:JetBrains Mono,monospace;"
                "font-size:0.7rem;color:#2D4A60;'>None identified</span>",
                unsafe_allow_html=True,
            )

    with d2:
        st.markdown("""
        <div style="font-family:'JetBrains Mono',monospace;font-size:0.6rem;font-weight:600;
            letter-spacing:0.15em;color:#68D391;margin-bottom:14px;margin-top:4px;
            text-transform:uppercase;">
            Benign Indicators
        </div>
        """, unsafe_allow_html=True)
        if not negative_features.empty:
            for _, row in negative_features.iterrows():
                feature_card(row["Feature"], row["Impact"], is_attack=False)
        else:
            st.markdown(
                "<span style='font-family:JetBrains Mono,monospace;"
                "font-size:0.7rem;color:#2D4A60;'>None identified</span>",
                unsafe_allow_html=True,
            )

    neon_divider()
    section_header("SHAP Explanation", "model interpretability")

    st.markdown(f"""
    <div style="padding:24px 28px;border-radius:12px;
        background:var(--bg-card,#101A27);
        border:1px solid rgba(99,179,237,0.12);
        border-left:2px solid rgba(99,179,237,0.45);
        line-height:1.9;font-family:'Outfit',sans-serif;font-size:0.9rem;
        color:#A0B4C8;letter-spacing:0.005em;font-weight:400;">
        {explanation_text}
    </div>
    """, unsafe_allow_html=True)


#Footer 
st.markdown("""
<div style="margin-top:4rem;padding-top:22px;
    border-top:1px solid rgba(99,179,237,0.06);
    display:flex;align-items:center;justify-content:space-between;">
    <span style="font-family:'JetBrains Mono',monospace;font-size:0.57rem;
        color:#1E3040;letter-spacing:0.1em;">
        © 2026 AI-NIDS — AI-DRIVEN NETWORK INTRUSION DETECTION SYSTEM PROTOTYPE
    </span>
    <span style="font-family:'JetBrains Mono',monospace;font-size:0.57rem;
        color:#1E3040;letter-spacing:0.1em;">
        RANDOM FOREST · SHAP · CICIDS2017
    </span>
</div>
""", unsafe_allow_html=True)