Team Members

1. Nihala Thajudeen

2. Savin jees

3. Pavithra KM

Course:

Predictive Analytics | Academic Year 2025-26

# Heart Disease Prediction AI 🩺
https://nihalabiomed-heart-disease-prediction-app-vnrlqu.streamlit.app

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
<img width="1440" height="900" alt="Screenshot 2026-04-08 at 8 49 55 PM" src="https://github.com/user-attachments/assets/1c9e6bb5-c69f-4f13-8de8-04e0da2e8df3" />

<img width="881" height="547" alt="6ab73deb-d442-46ab-8290-1892c96d6cfb" src="https://github.com/user-attachments/assets/6cc4212b-8cbd-477d-9114-0e08be2a97fe" />

<img width="567" height="435" alt="9497f973-47ec-4d19-aede-60c79c91d90d" src="https://github.com/user-attachments/assets/b6191987-5512-4c7a-a53a-d7f601cbd055" />

<img width="819" height="725" alt="b1a87226-eeb3-4012-98e6-63e85b965328" src="https://github.com/user-attachments/assets/e86fecd4-661e-46aa-94ca-13fa8d3e4383" />

### How to run locally:
1. Clone the repo: `git clone [YOUR_REPO_LINK]`
2. Install requirements: `pip install -r requirements.txt`
3. Run app: `streamlit run app.py`

---

## ✅ Final Results
Our Logistic Regression model achieved an accuracy of **88.52%**, making it a viable tool for preliminary health screening support.
