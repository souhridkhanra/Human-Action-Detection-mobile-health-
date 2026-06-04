import pandas as pd
import numpy as np


df = pd.read_csv("mHealth_raw_data.csv")

print("Original Shape:", df.shape)


df = df.dropna()

# Drop non-useful columns
df = df.drop(columns=['subject', 'timestamp'], errors='ignore')

print("Cleaned Shape:", df.shape)


X = df.drop('Activity', axis=1)
y = df['Activity']

# Encode labels (safe)
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

print("Data Ready")

from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(
    n_estimators=100,
    n_jobs=-1,
    random_state=42
)

print("Training Random Forest...")
rf_model.fit(X_train, y_train)


from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

y_pred = rf_model.predict(X_test)

print("\n===== FINAL RESULTS =====")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))


import seaborn as sns
import matplotlib.pyplot as plt

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(10, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

import joblib

joblib.dump(rf_model, "rf_activity_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(le, "label_encoder.pkl")

print("Model Saved")


sample = X_test[0].reshape(1, -1)

prediction = rf_model.predict(sample)
predicted_label = le.inverse_transform(prediction)

print("\nSample Prediction:", predicted_label)