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

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #07111f 0%, #0b1a2e 100%);
    border-right: 1px solid rgba(0,229,255,0.12);
}

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

/* ── Cyan particle burst ── */
@keyframes rise {
    0%   { transform: translateY(0) scale(1);   opacity: 1; }
    100% { transform: translateY(-100vh) scale(0.4); opacity: 0; }
}
.particle {
    position: fixed;
    bottom: -20px;
    width: 10px; height: 10px;
    border-radius: 50%;
    background: #00e5ff;
    box-shadow: 0 0 8px #00e5ff, 0 0 16px #00e5ff88;
    animation: rise linear forwards;
    z-index: 9999;
    pointer-events: none;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# 3. LOAD ASSETS
# ─────────────────────────────────────────────
@st.cache_resource
def load_assets():
    model = joblib.load('heart_model.pkl')
    scaler = joblib.load('scaler.pkl')
    return model, scaler

try:
    model, scaler = load_assets()
except FileNotFoundError:
    st.error("⚠️ Config error: Regenerate your heart_model.pkl via your notebook using the 5 features.")

# ─────────────────────────────────────────────
# 4. SIDEBAR INPUTS (High Importance Features Only)
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 10px 0 24px 0;'>
        <div style='font-size: 32px;'>🏥</div>
        <div style='font-family: Share Tech Mono, monospace; font-size: 20px;
                    color: #00e5ff; letter-spacing: 4px;'>CARDIOSCAN</div>
        <div style='font-size: 10px; color: #3d6680; letter-spacing: 3px;'>MINIMALIST CORE ENGINE v4.0</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("**🩻 Vital Demographics**")
    age = st.number_input("Age (years)", 1, 110, 52)
    sex = st.radio("Biological Sex", ["Male", "Female"], horizontal=True)
    sex_val = 1 if sex == "Male" else 0

    st.markdown("---")
    st.markdown("**📋 Primary Bio-Markers**")
    cp = st.selectbox("Chest Pain Severity", [0, 1, 2, 3], format_func=lambda x: f"Type {x}")
    chol = st.slider("Cholesterol level (mg/dL)", 100, 500, 240)
    thalach = st.slider("Maximum Heart Rate (bpm)", 60, 220, 155)

# ─────────────────────────────────────────────
# 5. HEADER & METRICS
# ─────────────────────────────────────────────
st.markdown("<h1>🫀 CARDIAC RISK ANALYTICS</h1>", unsafe_allow_html=True)
st.markdown("---")

target_hr   = 220 - age
chol_status = "ELEVATED" if chol > 240 else "NORMAL"

c1, c2, c3 = st.columns(3)
c1.metric("🫀 Age Target HR", f"{target_hr} bpm",  "Calculated Max")
c2.metric("🩸 Serum Cholesterol", f"{chol} mg/dL", chol_status)
c3.metric("📈 Peak Met HR",   f"{thalach} bpm",     f"Limit: {target_hr}")

st.markdown("---")

# ─────────────────────────────────────────────
# 6. MAIN PANEL & INFERENCE
# ─────────────────────────────────────────────
left, right = st.columns([3, 2])

with right:
    st.markdown("### ⚕️ Core Inference Engine")
    st.markdown(f"""
    <div style='background:#0a1628; border:1px solid rgba(0,229,255,0.15);
                border-radius:10px; padding:20px;
                font-family: Share Tech Mono, monospace;
                font-size:12px; color:#5f8aaa; line-height:2;'>
        MODEL &nbsp; → &nbsp; Optimized KNN Cluster<br>
        FEATURES → &nbsp; 5 Primary Vitals<br>
        STATUS &nbsp; → &nbsp; Ready for Spatial Check
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    run = st.button("🩺 EXECUTE RISK ANALYSIS")

with left:
    if run:
        with st.spinner("🔬 Computing clean vector neighborhood match..."):
            time.sleep(1)

        # ─── EXACT 5 FEATURE DATAFRAME MATCH ───
        feature_names = ['age', 'sex', 'cp', 'chol', 'thalach']
                         
        raw_features_df = pd.DataFrame([[age, sex_val, cp, chol, thalach]], 
                                       columns=feature_names)
                                       
        # Scale and predict using only the 5 core markers
        features_scaled = scaler.transform(raw_features_df)
        prediction      = model.predict(features_scaled)

        st.markdown("### 📊 Diagnostic Report")

        if prediction[0] == 1:
            st.error("🚨 POSITIVE — CARDIAC RISK DETECTED")
            st.warning("⚠️ Critical thresholds crossed in primary parameters. Clinical evaluation suggested.")
        else:
            st.success("✅ NEGATIVE — STABLE CARDIAC PROFILE")
            st.info("📋 Core metrics reside within safe parameter vectors.")

            # ── Cyan particle burst ──
            import random
            particles_html = ""
            for i in range(60):
                left_pct  = random.randint(0, 100)
                duration  = round(random.uniform(1.5, 3.5), 2)
                delay     = round(random.uniform(0, 1.5), 2)
                size      = random.randint(6, 14)
                shade     = random.choice(["#00e5ff", "#00cfff", "#00b8e6", "#7fffff", "#00fff7"])
                particles_html += f"""
                <div class='particle' style='
                    left:{left_pct}%;
                    width:{size}px; height:{size}px;
                    background:{shade};
                    box-shadow: 0 0 8px {shade};
                    animation-duration:{duration}s;
                    animation-delay:{delay}s;
                '></div>"""
            st.markdown(particles_html, unsafe_allow_html=True)

    else:
        st.markdown("""
        <div style='background:#0a1628; border:1px dashed rgba(0,229,255,0.2);
                    border-radius:12px; padding:60px; text-align:center;'>
            <div style='font-size:40px;'>🏥</div>
            <p style='font-family: Share Tech Mono, monospace; color:#00e5ff;
                      letter-spacing:2px;'>AWAITING CORE VITALS...</p>
            <p style='font-size:11px; color:#3d6680; letter-spacing:1px;'>
                INPUT PARAMETERS VIA SIDEBAR · THEN RUN ANALYSIS
            </p>
        </div>
        """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# 7. FOOTER
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style='text-align:center; font-size:10px; color:#3d6680; letter-spacing:2px;'>
    ⚕️ &nbsp; BIO-ARCHITECT ENGINE v4.0 &nbsp;·&nbsp; DIGITAL UNIVERSITY KERALA &nbsp;·&nbsp; RESEARCH USE ONLY
</div>
""", unsafe_allow_html=True)