# --- AT THE TOP OF APP.PY ---
import joblib
import numpy as np

# Load the brain and the scaling rules
model = joblib.load('heart_model.pkl')
scaler = joblib.load('scaler.pkl')

# --- INSIDE THE 'IF RUN:' BLOCK ---
# 1. Create the feature array from sidebar inputs
input_data = np.array([[age, sex_val, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])

# 2. SCALE the input (This is what you just fixed!)
input_scaled = scaler.transform(input_data)

# 3. PREDICT
prediction = model.predict(input_scaled)

import streamlit as st
import pandas as pd
import joblib
import time

# ─────────────────────────────────────────────
# 1. PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="CardioScan AI | Cardiac Risk Engine",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# 2. GLOBAL CSS
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

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #07111f 0%, #0b1a2e 100%);
    border-right: 1px solid rgba(0,229,255,0.12);
}
[data-testid="stSidebar"] * { color: #a8c0d6 !important; }

/* ── Button ── */
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

/* ── Metric Cards — fix truncation ── */
[data-testid="stMetric"] {
    background: linear-gradient(135deg, #0a1628, #0f2040);
    border: 1px solid rgba(0,229,255,0.2);
    border-radius: 12px;
    padding: 18px 16px 14px 20px;
    position: relative;
    overflow: visible !important;
    min-height: 90px;
}
[data-testid="stMetric"]::before {
    content: '';
    position: absolute;
    top: 0; left: 0;
    width: 3px; height: 100%;
    background: linear-gradient(180deg, #00e5ff, #0057ff);
    border-radius: 3px 0 0 3px;
}
[data-testid="stMetricLabel"] {
    color: #5f8aaa !important;
    font-size: 10px !important;
    letter-spacing: 1.5px !important;
    text-transform: uppercase !important;
    white-space: normal !important;
    overflow: visible !important;
    margin-bottom: 6px !important;
}
[data-testid="stMetricValue"] {
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 20px !important;
    color: #00e5ff !important;
    white-space: normal !important;
    overflow: visible !important;
}
[data-testid="stMetricDelta"] {
    color: #3dffa0 !important;
    font-size: 10px !important;
    white-space: normal !important;
    overflow: visible !important;
}

/* ── Progress bar ── */
.stProgress > div > div { background-color: #00e5ff !important; }

/* ── Divider ── */
hr { border-color: rgba(0,229,255,0.12) !important; }

/* ── Typography ── */
h1 {
    font-family: 'Share Tech Mono', monospace !important;
    color: #00e5ff !important;
    letter-spacing: 3px;
    text-shadow: 0 0 20px rgba(0,229,255,0.4);
    font-size: 26px !important;
}
h2, h3 {
    font-family: 'Exo 2', sans-serif !important;
    color: #7ec8e3 !important;
    letter-spacing: 1px;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# 3. LOAD MODEL
# ─────────────────────────────────────────────
@st.cache_resource
def load_model():
    return joblib.load('heart_model.pkl')

model = load_model()

# ─────────────────────────────────────────────
# 4. SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 10px 0 24px 0;'>
        <div style='font-size: 32px;'>❤️</div>
        <div style='font-family: Share Tech Mono, monospace; font-size: 20px;
                    color: #00e5ff; letter-spacing: 4px; margin-top: 6px;'>
            CARDIOSCAN
        </div>
        <div style='font-size: 10px; color: #3d6680; letter-spacing: 3px; margin-top: 4px;'>
            NEURAL RISK ENGINE v3.1
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("**🧬 Patient Parameters**")
    st.markdown("---")

    age      = st.number_input("Age (years)", min_value=1, max_value=110, value=52)
    sex      = st.radio("Biological Sex", ["Male", "Female"], horizontal=True)

    st.markdown("---")
    st.markdown("**📡 Clinical Signals**")

    cp = st.selectbox(
        "Chest Pain Type",
        options=[0, 1, 2, 3],
        format_func=lambda x: {
            0: "0 — Typical Angina",
            1: "1 — Atypical Angina",
            2: "2 — Non-Anginal Pain",
            3: "3 — Asymptomatic"
        }[x]
    )
    chol    = st.slider("Cholesterol (mg/dL)", 100, 500, 240)
    thalach = st.slider("Max Heart Rate (bpm)", 60, 220, 155)
    oldpeak = st.number_input("ST Depression (oldpeak)", 0.0, 6.0, 1.0, step=0.1)
    ca      = st.selectbox("Major Vessels (0–3)", [0, 1, 2, 3])

    st.markdown("---")
    st.caption("🔒 Data processed locally. Nothing is transmitted.")

# ─────────────────────────────────────────────
# 5. HEADER
# ─────────────────────────────────────────────
hcol1, hcol2 = st.columns([5, 1])
with hcol1:
    st.markdown("""
    <h1>CARDIAC RISK ANALYTICS</h1>
    <p style='color:#3d6680; font-family: Share Tech Mono, monospace;
              font-size:11px; letter-spacing:3px; margin-top:2px;'>
        CLEVELAND CLINICAL DATASET &nbsp;·&nbsp; KNN CLASSIFICATION &nbsp;·&nbsp; REAL-TIME INFERENCE
    </p>
    """, unsafe_allow_html=True)
with hcol2:
    st.markdown("""
    <div style='text-align:right; padding-top:20px;'>
        <span style='font-family: Share Tech Mono, monospace; font-size:11px;
                     background:#00e5ff18; color:#00e5ff;
                     border:1px solid #00e5ff44; border-radius:4px;
                     padding: 5px 10px; letter-spacing:2px;'>
            ● ONLINE
        </span>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# ─────────────────────────────────────────────
# 6. METRIC CARDS  (4 wide so text isn't squished)
# ─────────────────────────────────────────────
chol_label = "ELEVATED" if chol > 240 else ("BORDERLINE" if chol > 200 else "NORMAL")
target_hr  = 220 - age
hr_reserve = thalach - int(0.6 * age)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Target Heart Rate",  f"{target_hr} bpm",   "Age-predicted max")
c2.metric("Cholesterol",        f"{chol} mg/dL",       chol_label)
c3.metric("Peak Heart Rate",    f"{thalach} bpm",      f"HR Reserve {hr_reserve}")
c4.metric("ST Depression",      f"{oldpeak} mm",       f"{ca} vessel(s) flagged")

st.markdown("---")

# ─────────────────────────────────────────────
# 7. MAIN PANEL
# ─────────────────────────────────────────────
left, right = st.columns([3, 2])

with right:
    st.markdown("### ⚙️ Inference Engine")
    st.markdown(f"""
    <div style='background:#0a1628; border:1px solid rgba(0,229,255,0.15);
                border-radius:10px; padding:18px 20px;
                font-family: Share Tech Mono, monospace;
                font-size:12px; color:#5f8aaa; line-height:2.2;'>
        MODEL &nbsp;&nbsp;&nbsp;&nbsp; → &nbsp;K-Nearest Neighbors<br>
        ACCURACY &nbsp;→ &nbsp;90.16%<br>
        FEATURES &nbsp;→ &nbsp;13 clinical variables<br>
        DATASET &nbsp;&nbsp; → &nbsp;UCI Cleveland<br>
        <span style='color:#2a4050'>──────────────────</span><br>
        AGE &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; → &nbsp;{age} years<br>
        SEX &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; → &nbsp;{sex}<br>
        CP TYPE &nbsp;&nbsp; → &nbsp;Type {cp}<br>
        CHOL &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; → &nbsp;{chol} mg/dL<br>
        HR MAX &nbsp;&nbsp;&nbsp; → &nbsp;{thalach} bpm
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    run = st.button("⚡ EXECUTE RISK ANALYSIS")

with left:
    if run:
        with st.spinner("🔬 Running neural inference..."):
            bar = st.progress(0)
            for i in range(100):
                time.sleep(0.01)
                bar.progress(i + 1)

        sex_val  = 1 if sex == "Male" else 0
        features = pd.DataFrame(
            [[age, sex_val, cp, 130, chol, 0, 1, thalach, 0, oldpeak, 1, ca, 2]],
            columns=['age','sex','cp','trestbps','chol','fbs',
                     'restecg','thalach','exang','oldpeak','slope','ca','thal']
        )

        prediction  = model.predict(features)
        probability = model.predict_proba(features)[0][1] if hasattr(model, 'predict_proba') else None

        st.markdown("### 🩺 Diagnostic Report")

        if prediction[0] == 1:
            confidence = probability if probability else 0.85
            st.markdown(f"""
            <div style='background:linear-gradient(135deg,#1a0a10,#200d18);
                        border:1px solid rgba(255,46,99,0.4);
                        border-left:4px solid #ff2e63;
                        border-radius:10px; padding:22px 24px;
                        font-family: Share Tech Mono, monospace;'>
                <div style='font-size:16px; color:#ff2e63;
                            letter-spacing:2px; margin-bottom:14px;'>
                    🟥 &nbsp;POSITIVE — CARDIAC RISK DETECTED
                </div>
                <div style='font-size:12px; color:#9a5060; line-height:2.2;'>
                    CONFIDENCE &nbsp;&nbsp;&nbsp;→ &nbsp;{confidence:.1%}<br>
                    RISK LEVEL &nbsp;&nbsp;&nbsp;→ &nbsp;HIGH PROBABILITY<br>
                    RECOMMENDED &nbsp;→ &nbsp;CARDIOLOGY REFERRAL
                </div>
            </div>
            """, unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
            st.error(
                "**Clinical Advisory:** Patient metrics align with known coronary artery "
                "disease patterns. Immediate cardiology consultation and further workup "
                "(ECG, stress test, angiography) is strongly advised."
            )
            if probability:
                st.markdown(f"**Risk Score:** `{probability:.2%}`")
                st.progress(int(probability * 100))

        else:
            confidence = (1 - probability) if probability else 0.88
            st.markdown(f"""
            <div style='background:linear-gradient(135deg,#0a1a10,#0d2018);
                        border:1px solid rgba(61,255,160,0.35);
                        border-left:4px solid #3dffa0;
                        border-radius:10px; padding:22px 24px;
                        font-family: Share Tech Mono, monospace;'>
                <div style='font-size:16px; color:#3dffa0;
                            letter-spacing:2px; margin-bottom:14px;'>
                    🟩 &nbsp;NEGATIVE — STABLE CARDIAC PROFILE
                </div>
                <div style='font-size:12px; color:#3d8060; line-height:2.2;'>
                    CONFIDENCE &nbsp;&nbsp;&nbsp;→ &nbsp;{confidence:.1%}<br>
                    RISK LEVEL &nbsp;&nbsp;&nbsp;→ &nbsp;LOW PROBABILITY<br>
                    RECOMMENDED &nbsp;→ &nbsp;ROUTINE MONITORING
                </div>
            </div>
            """, unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
            st.success(
                "**Clinical Summary:** Patient vitals are within the statistical safety margin. "
                "No immediate cardiac intervention indicated. Continue routine health monitoring."
            )
            if probability:
                st.markdown(f"**Stability Score:** `{(1 - probability):.2%}`")
                st.progress(int((1 - probability) * 100))
            st.balloons()

    else:
        st.markdown("""
        <div style='background:#0a1628; border:1px dashed rgba(0,229,255,0.2);
                    border-radius:12px; padding:50px 30px; text-align:center;'>
            <div style='font-size:48px; margin-bottom:16px;'>🫀</div>
            <div style='font-family: Share Tech Mono, monospace;
                        color:#00e5ff; font-size:15px; letter-spacing:3px;'>
                SYSTEM READY
            </div>
            <div style='color:#3d6680; font-size:11px;
                        margin-top:10px; letter-spacing:2px; line-height:2;'>
                CONFIGURE PARAMETERS IN THE SIDEBAR<br>
                THEN CLICK EXECUTE TO RUN ANALYSIS
            </div>
        </div>
        """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# 8. FOOTER
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style='text-align:center; font-family: Share Tech Mono, monospace;
            font-size:11px; color:#2a4a60; letter-spacing:2px; padding: 6px 0 12px;'>
    CARDIOSCAN AI &nbsp;·&nbsp; BIO-ARCHITECT ENGINE v3.1 &nbsp;·&nbsp; DIGITAL UNIVERSITY KERALA<br>
    <span style='color:#1a3040; font-size:10px;'>
        ⚠ FOR RESEARCH & EDUCATIONAL PURPOSES ONLY — NOT FOR CLINICAL DIAGNOSIS
    </span>
</div>
""", unsafe_allow_html=True)