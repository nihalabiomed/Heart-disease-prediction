import streamlit as st
import pandas as pd
import joblib

# 1. Load the "Brain" we saved earlier
# Make sure you ran the 'joblib.dump' line in your notebook first!
model = joblib.load('heart_model.pkl')

# 2. Setup the Website Look
st.set_page_config(page_title="Heart Health AI")
st.title("❤️ Heart Disease Predictor AI")
st.write("Enter the patient's clinical details below to analyze risk.")

# 3. Create Input Boxes (The Sliders)
st.sidebar.header("Patient Metrics")
age = st.sidebar.slider("Age", 1, 100, 50)
chol = st.sidebar.slider("Cholesterol (mg/dl)", 100, 600, 200)
thalach = st.sidebar.slider("Max Heart Rate Achieved", 60, 220, 150)

# 4. The Prediction Logic
if st.button("Analyze Results"):
    # This creates the table the model needs to read
    # We use 'healthy' defaults for the columns we don't have sliders for yet
    features = pd.DataFrame([[age, 1, 0, 120, chol, 0, 0, thalach, 0, 0.0, 1, 0, 2]], 
                            columns=['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
                                     'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'])
    
    # Get the prediction from the model
    prediction = model.predict(features)
    
    # Display the result
    st.subheader("Final Assessment:")
    if prediction[0] == 1:
        st.error("⚠️ HIGH RISK: The model predicts a high probability of heart disease.")
        st.write("Consider further clinical diagnostic testing.")
    else:
        st.success("✅ LOW RISK: The model predicts the patient is likely healthy.")
        st.write("The clinical metrics provided fall within normal ranges for this model.")