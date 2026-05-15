Team Members
  
1. Nihala Thajudeen

 2. Savin jees v

3. Pavithra KM

Course:

Predictive Analytics | Academic Year 2025-26

# Heart Disease Prediction AI 🩺
https://nihalabiomed-heart-disease-prediction-app-vnrlqu.streamlit.app

This project uses Machine Learning to predict the presence or absence of heart disease based on 13 clinical attributes from the Cleveland Heart Disease Dataset.

## 🚀 Project Overview
This clinical dashboard uses Machine Learning to predict cardiac risk based on the **Cleveland Heart Disease Dataset**. This project demonstrates a full Bio-AI pipeline: from raw data preprocessing and feature selection to live deployment.

## 📊 Analytics & Model Performance
We evaluated three different algorithms to find the most accurate "diagnostic brain":

* **KNN (K-Nearest Neighbors):** 🏆 **90.16% Accuracy** (Current Production Model)
* **Logistic Regression:** 85.25% Accuracy
* **Gradient Boosting:** 77.05% Accuracy

### 🔍 Feature Selection (Stage 5)
Using **Mutual Information** and **Chi-Square** tests, we identified the most critical clinical markers:
1. **cp** (Chest Pain Type)
2. **thalach** (Max Heart Rate Achieved)
3. **ca** (Number of Major Vessels)
4. **thal** (Thalassemia/Blood Flow)

---

## 💻 Technical Stack
* **Language:** Python 3.14
* **Framework:** Streamlit (Web UI)
* **ML Libraries:** Scikit-Learn, Pandas, NumPy
* **Visualization:** Seaborn, Matplotlib

---

## 🛠️ How to run locally
1. Clone the repository:
   ```bash
   git clone [https://github.com/nihalabiomed/Heart-disease-prediction.git](https://github.com/nihalabiomed/Heart-disease-prediction.git)
Install dependencies:

Bash
pip install -r requirements.txt
Launch the dashboard:

Bash
streamlit run app.py
