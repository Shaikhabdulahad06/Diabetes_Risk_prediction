# train_risk_model_rf.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import pickle

# 1. Load the Pima Indians Diabetes Dataset
data = pd.read_csv("diabetes.csv")

# 2. Keep only the features we want
features = ["Pregnancies", "BMI", "DiabetesPedigreeFunction", "Age"]
X = data[features]
y = data["Outcome"]

# 3. Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=42
)

# 4. Train Random Forest Classifier
model = RandomForestClassifier(
    n_estimators=200, max_depth=6, random_state=42
)
model.fit(X_train, y_train)

# 5. Evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Example probability output
y_prob = model.predict_proba(X_test)[:, 1]
print("\nFirst 10 probability predictions:", y_prob[:10])

# 6. Save model
with open("risk_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Risk Model trained and saved as risk_model.pkl")
