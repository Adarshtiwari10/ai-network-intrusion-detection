from datetime import datetime
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import shap
from backend.core.data import load_dataset, split_dataset
from backend.core.model import load_model, load_features
from backend.core.simulation import get_random_packet, predict, simulate_window
from backend.core.evaluation import evaluate
from backend.services.SHAP_explainer import create_explainer, generate_shap_analysis

MODEL_PATH = "backend/model/rf_model.pkl"
FEATURE_PATH = "backend/model/rf_features.pkl"
DATA_FILE = "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"

st.set_page_config(page_title="AI-NIDS", layout="wide")
st.title("AI-Based IoT Gateway  Intrusion Detection System")
st.caption("Real-time packet analysis with SHAP explainability and sliding window simulation")

# Display current system status based on last simulation event
if "last_event" in st.session_state:
    current_severity = st.session_state["last_event"]["severity"]

    if current_severity == "LOW":
        st.success("Gateway Status: STABLE")
    elif current_severity == "MEDIUM":
        st.warning("Gateway Status: ELEVATED RISK")
    else:
        st.error("Gateway Status: HIGH THREAT DETECTED")
else:
    st.info("System initialized. Awaiting traffic simulation...")

if "alert_log" not in st.session_state:
    st.session_state["alert_log"] = []

st.divider()

st.subheader("System Overview")

col1, col2, col3 = st.columns(3)

total_events = len(st.session_state["alert_log"])
high_alerts = sum(1 for e in st.session_state["alert_log"] if e["severity"] == "HIGH")
avg_risk = (
    round(sum(e["mean_risk_score"] for e in st.session_state["alert_log"]) / total_events, 2)
    if total_events > 0 else 0
)

col1.metric("Total Windows Analyzed", total_events)
col2.metric("High Severity Alerts", high_alerts)
col3.metric("Average Risk Score", avg_risk)
st.divider()

model = load_model(MODEL_PATH)
feature_names = load_features(FEATURE_PATH)
explainer = create_explainer(model)

if not model or not feature_names:
    st.error("Model or feature schema not found.")
    st.stop()

df = load_dataset(DATA_FILE)
X_train, X_test, y_train, y_test = split_dataset(df, feature_names)

# Sliding Window Simulation
with st.container():
    st.subheader("Gateway Traffic Analysis Simulation")
    if st.button("Run Sliding Window Simulation"):
        event = simulate_window(model, X_test, y_test)
        st.session_state["last_event"] = event
        st.session_state["alert_log"].append(event)
        st.rerun()
    
    if st.button("Reset Simulation"):
        st.session_state["alert_log"] = []
        st.session_state.pop("last_event", None)
        st.rerun()

        st.subheader("Gateway Traffic Analysis Window")

        col1, col2, col3 = st.columns(3)

        col1.metric("Packets Analyzed", event["window_size"])
        col2.metric("Malicious Packets", event["attack_count"])
        col3.metric("Mean Risk Score", f"{event['mean_risk_score']:.2f}")

        if event["severity"] == "HIGH":
            st.error(" HIGH SEVERITY ALERT")
        elif event["severity"] == "MEDIUM":
            st.warning(" MEDIUM SEVERITY ALERT")
        else:        
            st.success(" LOW SEVERITY ALERT")

        if event["alert_triggered"]:
            st.error("Gateway Alert Triggered")
        else:
            st.success("No Gateway Alert")
        st.session_state["alert_log"].append(event)

# gateway alert log
# st.subheader("Gateway Alert Log")
if st.session_state["alert_log"]:
    log_df = pd.DataFrame(st.session_state["alert_log"])
    
    st.subheader("📈 Risk Score Trend Over Time")

    fig = px.line(
        log_df,
        x="timestamp",
        y="mean_risk_score",
        markers=True
    )

    # 🔴 ADD THIS PART
    fig.add_hline(
        y=0.6,
        line_dash="dash",
        annotation_text="Alert Threshold",
        annotation_position="top left"
    )
    st.plotly_chart(fig, use_container_width=True)
    st.subheader("Gateway Alert Log")
    log_df = pd.DataFrame(st.session_state["alert_log"])
    st.dataframe(
        log_df.sort_values(by="timestamp", ascending=False),
        use_container_width=True
    )
else:
    st.info("No alerts logged yet.")

## Real-time Packet Analysis
st.divider()
st.subheader("Real-time Packet Analysis")
col1, col2 = st.columns(2)

with col1:
    if st.button("Capture Random Packet"):
        packet, actual = get_random_packet(X_test, y_test)
        st.session_state["packet"] = packet
        st.session_state["actual"] = actual

if "packet" in st.session_state:
    close_col1, close_col2 = st.columns([8, 1])
    with close_col1:
        if st.button("Close Analysis"):
            st.session_state.pop("packet", None)
            st.session_state.pop("actual", None)
            st.rerun()
    packet = st.session_state["packet"]
    prediction = predict(model, packet)
    st.divider()

    packet_df = packet.to_frame().T


    important_display_features = [
        "Source Port",
        "Destination Port",
        "Protocol",
        "Flow Duration",
        "Total Fwd Packets",
        "Total Backward Packets",
        "Total Length of Fwd Packets",
        "Total Length of Bwd Packets"
    ]

    available_features = [f for f in important_display_features if f in packet_df.columns]
    filtered_packet = packet_df[available_features]

    left, right = st.columns([1, 1])

    with left:
        st.subheader("Packet Summary")
        st.table(filtered_packet.T)

    with right:
        st.subheader(" Classification Result")

        if prediction == 0:
            st.success(" BENIGN TRAFFIC")
        else:
            st.error("MALICIOUS TRAFFIC")


    shap_vector, explanation_text = generate_shap_analysis(
        explainer,
        packet_df,
        feature_names,
        prediction
    )

    # Build impact dataframe for clean display
    impact_df = pd.DataFrame({
        "Feature": feature_names,
        "Impact": shap_vector
    })

    impact_df["AbsImpact"] = impact_df["Impact"].abs()
    impact_df = impact_df.sort_values(by="AbsImpact", ascending=False).head(5)

    positive_features = impact_df[impact_df["Impact"] > 0]
    negative_features = impact_df[impact_df["Impact"] <= 0]

    st.subheader("🔍 Key Risk Drivers")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🔴 Attack Drivers")
        if not positive_features.empty:
            for _, row in positive_features.iterrows():
                st.error(f"{row['Feature']}")
        else:
            st.write("None")

    with col2:
        st.markdown("### 🟢 Benign Indicators")
        if not negative_features.empty:
            for _, row in negative_features.iterrows():
                st.success(f"{row['Feature']}")
        else:
            st.write("None")

    st.divider()
    st.subheader(" SHAP Explanation")
    st.markdown(
        f"""
        <div style="
            padding:15px;
            border-radius:10px;
            background-color:#161B22;
            border:1px solid #30363d;
            line-height:1.6;
            font-size:15px;
        ">
            {explanation_text}
        </div>
        """,
        unsafe_allow_html=True
    )

st.divider()
st.caption("© 2026 AI-driven Network Intrusion Detection System Prototype")




