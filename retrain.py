import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier

# 1. Load data
data = pd.read_csv('heart-disease.csv')

# 2. Prep
X = data.drop('target', axis=1)
y = data['target']

# 3. Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. Train Gradient Boosting
model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
model.fit(X_train_scaled, y_train)

# 6. Print accuracy
from sklearn.metrics import accuracy_score
y_pred = model.predict(X_test_scaled)
print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")

# 7. Save
joblib.dump(model, 'heart_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
print("Success! Gradient Boosting trained and saved.")