import pandas as pd
import numpy as np
import joblib
import os
import re

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "model")

MODEL_REQUIRED_COLS = [
    "Age", "BusinessTravel", "DailyRate", "Department",
    "DistanceFromHome", "Education", "EducationField",
    "EnvironmentSatisfaction", "Gender", "HourlyRate",
    "JobInvolvement", "JobLevel", "JobRole", "JobSatisfaction",
    "MaritalStatus", "MonthlyIncome", "MonthlyRate", "NumCompaniesWorked",
    "OverTime", "PercentSalaryHike", "PerformanceRating",
    "RelationshipSatisfaction", "StockOptionLevel",
    "TotalWorkingYears", "TrainingTimesLastYear", "WorkLifeBalance",
    "YearsAtCompany", "YearsInCurrentRole", "YearsSinceLastPromotion",
    "YearsWithCurrManager"
]

def load_model_artifacts():
    model = joblib.load(os.path.join(MODEL_DIR, "attrition_model.pkl"))
    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
    feature_columns = joblib.load(os.path.join(MODEL_DIR, "feature_columns.pkl"))
    return model, scaler, feature_columns

def find_column(df, candidates):
    candidates = [candidates] if isinstance(candidates, str) else candidates
    for c in df.columns:
        c_clean = c.strip().lower().replace(" ", "").replace("_", "").replace("-", "")
        for cand in candidates:
            cand_clean = cand.strip().lower().replace(" ", "").replace("_", "").replace("-", "")
            if c_clean == cand_clean:
                return c
    for c in df.columns:
        c_clean = c.strip().lower().replace(" ", "").replace("_", "").replace("-", "")
        for cand in candidates:
            cand_clean = cand.strip().lower().replace(" ", "").replace("_", "").replace("-", "")
            if cand_clean in c_clean or c_clean in cand_clean:
                return c
    return None

def normalize_column_names(df):
    df = df.copy()
    mapping = {}
    for col in df.columns:
        normalized = col.strip()
        mapping[col] = normalized
    return df.rename(columns=mapping)

def get_available_model_cols(df):
    norm = normalize_column_names(df)
    found = {}
    for col in MODEL_REQUIRED_COLS:
        matched = find_column(norm, [col])
        if matched is not None:
            found[col] = matched
    return found

def can_predict(found_map):
    return len(found_map) >= 10

def preprocess_for_prediction(df_raw):
    model, scaler, feature_columns = load_model_artifacts()
    df = df_raw.copy()
    col_map = get_available_model_cols(df)
    rename = {v: k for k, v in col_map.items()}
    df = df.rename(columns=rename)
    available = [c for c in MODEL_REQUIRED_COLS if c in df.columns]
    drop_cols = ["EmployeeNumber", "Over18", "StandardHours", "EmployeeCount", "Attrition"]
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True, errors="ignore")
    cat_cols = df.select_dtypes(include="object").columns.tolist()
    df_enc = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    for col in feature_columns:
        if col not in df_enc.columns:
            df_enc[col] = 0
    for col in df_enc.columns:
        if col not in feature_columns:
            df_enc.drop(columns=[col], inplace=True)
    df_enc = df_enc[feature_columns]
    X_scaled = scaler.transform(df_enc)
    return X_scaled, model

def get_column_type(col, df):
    if pd.api.types.is_numeric_dtype(df[col]):
        return "numeric"
    return "categorical"

def detect_groups(df):
    groups = []
    age_map = {"age", "employeeage", "employee_age", "ageofemployee"}
    dept_map = {"department", "dept", "businessunit", "division", "unit"}
    role_map = {"jobrole", "role", "job_title", "position", "title", "designation"}
    gender_map = {"gender", "sex"}
    salary_map = {"monthlyincome", "income", "salary", "monthly_income", "compensation"}
    tenure_map = {"yearsatcompany", "tenure", "years_of_service", "serviceyears", "employmentyears"}
    education_map = {"education", "edu", "educationlevel", "degreename"}
    overtime_map = {"overtime", "over_time", "ot"}
    satisfaction_map = {"jobsatisfaction", "job_satisfaction", "satisfaction"}
    balance_map = {"worklifebalance", "work_life_balance", "wlb"}
    marital_map = {"maritalstatus", "marital_status", "marital"}
    travel_map = {"businesstravel", "business_travel", "travel"}
    id_map = {"employeeid", "employee_id", "empid", "employee_number", "employeenumber"}
    perf_map = {"performancerating", "performance_rating", "performance"}
    all_maps = {
        "Age": age_map, "Department": dept_map, "JobRole": role_map, "Gender": gender_map,
        "MonthlyIncome": salary_map, "YearsAtCompany": tenure_map, "Education": education_map,
        "OverTime": overtime_map, "JobSatisfaction": satisfaction_map,
        "WorkLifeBalance": balance_map, "MaritalStatus": marital_map,
        "BusinessTravel": travel_map, "EmployeeNumber": id_map, "PerformanceRating": perf_map,
    }
    for group, aliases in all_maps.items():
        matched = find_column(df, list(aliases))
        if matched:
            groups.append((group, matched))
    return groups
