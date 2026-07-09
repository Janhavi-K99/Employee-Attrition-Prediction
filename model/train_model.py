import pandas as pd
import numpy as np
import joblib
import os
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "WA_Fn-UseC_-HR-Employee-Attrition.csv")
MODEL_DIR = os.path.dirname(__file__)

if not os.path.exists(DATA_PATH):
    DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "..", "WA_Fn-UseC_-HR-Employee-Attrition.csv")

df = pd.read_csv(DATA_PATH)

columns_to_drop = ["EmployeeNumber", "Over18", "StandardHours", "EmployeeCount"]
df.drop(columns=columns_to_drop, inplace=True, errors="ignore")

df["Attrition"] = df["Attrition"].map({"Yes": 1, "No": 0})

categorical_cols = df.select_dtypes(include="object").columns.tolist()
numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
numeric_cols.remove("Attrition")

df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

feature_columns = [col for col in df_encoded.columns if col != "Attrition"]
X = df_encoded[feature_columns]
y = df_encoded["Attrition"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = LogisticRegression(class_weight="balanced", random_state=42, max_iter=2000)
model.fit(X_scaled, y)

joblib.dump(model, os.path.join(MODEL_DIR, "attrition_model.pkl"))
joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))
joblib.dump(feature_columns, os.path.join(MODEL_DIR, "feature_columns.pkl"))

print("Model training complete.")
print(f"Features: {len(feature_columns)}")
print(f"Model coefficients: {len(model.coef_[0])}")
