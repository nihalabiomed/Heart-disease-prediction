import streamlit as st
import pandas as pd
import joblib

# 1. Page Configuration & Custom CSS for a "Digital" feel
st.set_page_config(page_title="Bio-Architect | Heart AI", layout="wide")

st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #ff4b4b;
        color: white;
    }
    </style>
    """, unsafe_allow_stdio=True)

# 2. Load the Model
# Ensure your heart_model.pkl is in the same folder!
model = joblib.load('heart_model.pkl')

# 3. Header Section
st.title("🧬 Bio-Architect: Cardiac Analysis Lab")
st.markdown("---")

# 4. Dashboard Layout
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("📋 Patient Metrics")
    age = st.number_input("Age", 1, 120, 50)
    sex = st.selectbox("Sex", ["Male", "Female"])
    cp = st.selectbox("Chest Pain Type (0-3)", [0, 1, 2, 3])
    chol = st.slider("Cholesterol (mg/dl)", 100, 600, 240)
    thalach = st.slider("Max Heart Rate Achieved", 60, 220, 150)
    ca = st.slider("Major Vessels (0-3)", 0, 3, 0)

with col2:
    st.subheader("🩺 Diagnostic Output")
    st.info("Adjust the metrics on the left and click 'Run Analysis' to see the AI prediction.")
    
    if st.button("🚀 Run Neural Analysis"):
        # Convert inputs to model format
        sex_val = 1 if sex == "Male" else 0
        
        # We use clinical defaults for the features not in the UI
        # Order: age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal
        features = pd.DataFrame([[age, sex_val, cp, 120, chol, 0, 1, thalach, 0, 1.0, 1, ca, 2]], 
                                columns=['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
                                         'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'])
        
        prediction = model.predict(features)
        probability = model.predict_proba(features)[0][1] if hasattr(model, 'predict_proba') else None

        st.markdown("---")
        
        if prediction[0] == 1:
            st.error(f"### ⚠️ Assessment: HIGH RISK")
            if probability:
                st.write(f"**Confidence Level:** {probability:.2%}")
            st.warning("Clinical recommendation: Immediate cardiology consultation required.")
        else:
            st.success(f"### ✅ Assessment: LOW RISK")
            if probability:
                st.write(f"**Confidence Level:** {(1-probability):.2%}")
            st.balloons()
            st.write("Clinical metrics appear stable. Patient is within the predicted healthy range.")

# 5. Footer
st.markdown("---")
st.caption("Bio-Architect Lab v2.0 | Digital University Kerala | Data Science & Bio AI")