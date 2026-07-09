import pandas as pd
import numpy as np
from utils.preprocessing import preprocess_for_prediction

def predict_attrition(df_raw):
    X_scaled, model = preprocess_for_prediction(df_raw)
    predictions = model.predict(X_scaled)
    probabilities = model.predict_proba(X_scaled)[:, 1]
    risk_labels = []
    for prob in probabilities:
        if prob < 0.3:
            risk_labels.append("Low Risk")
        elif prob < 0.6:
            risk_labels.append("Medium Risk")
        else:
            risk_labels.append("High Risk")
    return predictions, probabilities, risk_labels
