Team Members

1. Nihala Thajudeen

2. Savin jees

3. Pavithra KM

Course:

Predictive Analytics | Academic Year 2025-26

# Heart Disease Prediction AI 🩺

This project uses Machine Learning to predict the presence or absence of heart disease based on 13 clinical attributes from the Cleveland Heart Disease Dataset.

## 🚀 Project Overview
This was developed as part of the Predictive Analytics course (Academic Year 2025-26). It covers the full data science life cycle, from data cleaning to live model deployment.

### Team Members
* **Nihala Thajudeen** - Lead Developer & Data Scientist

---

## 📊 Data Science Life Cycle Stages

### 1. Data Collection & Understanding
Used the **Cleveland Heart Disease Dataset** containing 303 patient records.
* **Key Features:** Age, Cholesterol, Chest Pain Type (cp), Max Heart Rate (thalach).

### 2. Exploratory Data Analysis (EDA)
I performed EDA to find patterns. For example, I found that as age and cholesterol increase, the likelihood of heart disease tends to rise.

### 3. Model Building & Comparison
I tested three different algorithms to find the best "brain" for this task:
* **Logistic Regression:** 88.52% Accuracy (🏆 Winner)
* **Gradient Boosting:** 77.00% Accuracy
* **KNN:** 68.00% Accuracy

### 4. Model Interpretation
Using feature importance, I discovered that **Chest Pain Type (cp)** and **Max Heart Rate (thalach)** are the most significant predictors in our smartest model.

---

## 💻 Deployment
The model is deployed using **Streamlit**. It provides a user-friendly interface for clinicians to input patient metrics and receive an instant risk assessment.

### How to run locally:
1. Clone the repo: `git clone [YOUR_REPO_LINK]`
2. Install requirements: `pip install -r requirements.txt`
3. Run app: `streamlit run app.py`

---

## ✅ Final Results
Our Logistic Regression model achieved an accuracy of **88.52%**, making it a viable tool for preliminary health screening support.
