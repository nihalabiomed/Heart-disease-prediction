import streamlit as st
import pandas as pd
import joblib
import time
import numpy as np

# ─────────────────────────────────────────────
# 1. PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="CardioScan AI | Cardiac Risk Engine",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# 2. GLOBAL CSS (Clinical Cyberpunk Theme)
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Exo+2:wght@300;400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Exo 2', sans-serif;
    background-color: #060b14;
    color: #c9d6e3;
}
.main { background-color: #060b14; }

/* Sidebar Styling */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #07111f 0%, #0b1a2e 100%);
    border-right: 1px solid rgba(0,229,255,0.12);
}

/* Cyber Button */
.stButton > button {
    background: linear-gradient(135deg, #00e5ff, #0057ff);
    color: #000;
    font-family: 'Share Tech Mono', monospace;
    font-size: 14px;
    font-weight: 700;
    letter-spacing: 2px;
    border: none;
    border-radius: 8px;
    height: 3.2em;
    width: 100%;
    transition: all 0.3s ease;
    box-shadow: 0 0 20px rgba(0,229,255,0.3);
}
.stButton > button:hover {
    background: linear-gradient(135deg, #ff2e63, #ff6b35);
    box-shadow: 0 0 24px rgba(255,46,99,0.4);
    transform: translateY(-2px);
}

/* Metric Cards */
[data-testid="stMetric"] {
    background: linear-gradient(135deg, #0a1628, #0f2040);
    border: 1px solid rgba(0,229,255,0.2);
    border-radius: 12px;
    padding: 18px 16px;
}
[data-testid="stMetricValue"] {
    font-family: 'Share Tech Mono', monospace !important;
    color: #00e5ff !important;
}

h1 {
    font-family: 'Share Tech Mono', monospace !important;
    color: #00e5ff !important;
    letter-spacing: 3px;
    text-shadow: 0 0 20px rgba(0,229,255,0.4);
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# 3. LOAD ASSETS (Model & Scaler)
# ─────────────────────────────────────────────
@st.cache_resource
def load_assets():
    model = joblib.load('heart_model.pkl')
    scaler = joblib.load('scaler.pkl')
    return model, scaler

try:
    model, scaler = load_assets()
except FileNotFoundError:
    st.error("⚠️ Model or Scaler files not found! Please ensure 'heart_model.pkl' and 'scaler.pkl' are in the repository.")

# ─────────────────────────────────────────────
# 4. SIDEBAR INPUTS
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 10px 0 24px 0;'>
        <div style='font-size: 32px;'>🫀</div>
        <div style='font-family: Share Tech Mono, monospace; font-size: 20px; color: #00e5ff; letter-spacing: 4px;'>CARDIOSCAN</div>
        <div style='font-size: 10px; color: #3d6680; letter-spacing: 3px;'>NEURAL RISK ENGINE v3.1</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("**🩻 Patient Parameters**")
    age = st.number_input("Age (years)", 1, 110, 52)
    sex = st.radio("Biological Sex", ["Male", "Female"], horizontal=True)
    sex_val = 1 if sex == "Male" else 0

    st.markdown("---")
    st.markdown("**📋 Clinical Signals**")
    cp = st.selectbox("Chest Pain Type", [0, 1, 2, 3], format_func=lambda x: f"Type {x}")
    trestbps = st.number_input("Resting BP (mm Hg)", 90, 200, 130)
    chol = st.slider("Cholesterol (mg/dL)", 100, 500, 240)
    fbs = st.selectbox("Fasting Blood Sugar > 120", [0, 1])
    restecg = st.selectbox("Resting ECG", [0, 1, 2])
    thalach = st.slider("Max Heart Rate (bpm)", 60, 220, 155)
    exang = st.radio("Exercise Induced Angina", [0, 1], horizontal=True)
    oldpeak = st.number_input("ST Depression", 0.0, 6.0, 1.0, 0.1)
    slope = st.selectbox("Slope of ST", [0, 1, 2])
    ca = st.selectbox("Major Vessels (0–3)", [0, 1, 2, 3])
    thal = st.selectbox("Thalassemia", [0, 1, 2, 3])

# ─────────────────────────────────────────────
# 5. HEADER & METRICS
# ─────────────────────────────────────────────
st.markdown("<h1>🫀 CARDIAC RISK ANALYTICS</h1>", unsafe_allow_html=True)
st.markdown("---")

# Quick Analytics
target_hr = 220 - age
chol_status = "ELEVATED" if chol > 240 else "NORMAL"

c1, c2, c3, c4 = st.columns(4)
c1.metric("🫀 Target HR", f"{target_hr} bpm", "Age-Max")
c2.metric("🩸 Cholesterol", f"{chol} mg/dL", chol_status)
c3.metric("📈 Peak HR", f"{thalach} bpm", f"Limit: {target_hr}")
c4.metric("🔬 Vessels", f"{ca}", "Fluoroscopy")

st.markdown("---")

# ─────────────────────────────────────────────
# 6. MAIN PANEL & INFERENCE
# ─────────────────────────────────────────────
left, right = st.columns([3, 2])

with right:
    st.markdown("### ⚕️ Inference Engine")
    st.markdown(f"""
    <div style='background:#0a1628; border:1px solid rgba(0,229,255,0.15); border-radius:10px; padding:20px; font-family: Share Tech Mono, monospace; font-size:12px; color:#5f8aaa; line-height:2;'>
        MODEL &nbsp; → &nbsp; K-Nearest Neighbors (KNN)<br>
        ACCURACY → &nbsp; 90.16%<br>
        STATUS &nbsp; → &nbsp; Ready for Computation
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    run = st.button("🩺 EXECUTE RISK ANALYSIS")

with left:
    if run:
        with st.spinner("🔬 Running diagnostic inference pipeline..."):
            time.sleep(1)

        # 1. Create Input Array (Exactly 13 features in order)
        raw_features = np.array([[age, sex_val, cp, trestbps, chol, fbs,
                                  restecg, thalach, exang, oldpeak, slope, ca, thal]])

        # 2. Scale the input using the saved scaler
        features_scaled = scaler.transform(raw_features)

        # 3. Predict
        prediction = model.predict(features_scaled)

        st.markdown("### 📊 Diagnostic Report")
        if prediction[0] == 1:
            st.error("🚨 POSITIVE — CARDIAC RISK DETECTED")
            st.warning("⚠️ Immediate cardiology consultation advised based on clinical markers.")
        else:
            st.success("✅ NEGATIVE — STABLE CARDIAC PROFILE")
            st.info("📋 Metrics reside within safety margins. Routine monitoring recommended.")
    else:
        st.markdown("""
        <div style='background:#0a1628; border:1px dashed rgba(0,229,255,0.2); border-radius:12px; padding:60px; text-align:center;'>
            <div style='font-size:40px;'>🩻</div>
            <p style='font-family: Share Tech Mono, monospace; color:#00e5ff; letter-spacing:2px;'>AWAITING PATIENT DATA...</p>
            <p style='font-size:11px; color:#3d6680; letter-spacing:1px;'>INPUT PARAMETERS VIA SIDEBAR · THEN EXECUTE ANALYSIS</p>
        </div>
        """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# 7. FOOTER
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown("<div style='text-align:center; font-size:10px; color:#3d6680; letter-spacing:2px;'>⚕️ &nbsp; BIO-ARCHITECT ENGINE v3.1 &nbsp;·&nbsp; DIGITAL UNIVERSITY KERALA &nbsp;·&nbsp; RESEARCH USE ONLY</div>", unsafe_allow_html=True)