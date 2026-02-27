import pandas as pd
import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

DATA_FILE = "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"

MODEL_DIR = "model"
MODEL_FILE = os.path.join(MODEL_DIR, "rf_model.pkl")
FEATURE_FILE = os.path.join(MODEL_DIR, "rf_features.pkl")

os.makedirs(MODEL_DIR, exist_ok=True)

print("Loading dataset...")
df = pd.read_csv(DATA_FILE)

df.columns = df.columns.str.strip()
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)

columns_to_drop = [
    "Flow ID",
    "Source IP",
    "Destination IP",
    "Timestamp"
]

for col in columns_to_drop:
    if col in df.columns:
        df.drop(col, axis=1, inplace=True)

if "Label" not in df.columns:
    raise ValueError("Label column not found.")

X = df.drop("Label", axis=1)
y = df["Label"].apply(lambda x: 0 if x == "BENIGN" else 1)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print("Training RandomForest...")

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

print("\nModel Evaluation:")
print(classification_report(y_test, model.predict(X_test)))

joblib.dump(model, MODEL_FILE)
joblib.dump(X.columns.tolist(), FEATURE_FILE)

print("\nModel and feature schema saved in /model directory.")

print("\nTop Feature Importances:")
for name, importance in zip(X.columns, model.feature_importances_):
    print(f"{name}: {importance:.4f}")