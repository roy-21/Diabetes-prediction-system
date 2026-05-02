import numpy as np
import pandas as pd
import shap
import joblib
import os
import torch
from pytorch_tabnet.tab_model import TabNetClassifier
import matplotlib
matplotlib.use('Agg') # For headless plotting
import matplotlib.pyplot as plt

ARTIFACTS_DIR = "artifacts"
DATA_PATH = "diabetes.csv"

# Global lazy-loading for models
models = {
    "ensemble": None,
    "tabnet": None,
    "scaler": None,
    "explainer": None,
    "feature_cols": None,
    "global_median": None
}

def load_artifacts():
    if models["scaler"] is None:
        models["scaler"] = joblib.load(os.path.join(ARTIFACTS_DIR, "scaler.joblib"))
        models["feature_cols"] = np.load(os.path.join(ARTIFACTS_DIR, "feature_columns.npy"), allow_pickle=True)
        models["explainer"] = joblib.load(os.path.join(ARTIFACTS_DIR, "shap_explainer.joblib"))
        
        # Load median values
        df = pd.read_csv(DATA_PATH)
        zero_as_missing = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
        df[zero_as_missing] = df[zero_as_missing].replace(0, np.nan)
        models["global_median"] = df.median(numeric_only=True)
        
    if models["ensemble"] is None:
        models["ensemble"] = joblib.load(os.path.join(ARTIFACTS_DIR, "diabetes_ensemble_model.joblib"))
        
    if models["tabnet"] is None:
        tabnet_path = os.path.join(ARTIFACTS_DIR, "tabnet_model.zip")
        if os.path.exists(tabnet_path):
            clf = TabNetClassifier()
            clf.load_model(tabnet_path)
            models["tabnet"] = clf

def predict_diabetes_api(input_dict, model_type="ensemble"):
    load_artifacts()
    
    # Feature Engineering
    df_input = pd.DataFrame([input_dict])
    zero_as_missing = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
    df_input[zero_as_missing] = df_input[zero_as_missing].replace(0, np.nan)
    
    for col in zero_as_missing:
        if col in df_input.columns:
            df_input[col].fillna(models["global_median"][col], inplace=True)
            
    df_input["Glucose_Insulin_ratio"] = df_input["Glucose"] / (df_input["Insulin"] + 1e-6)
    df_input["BMI_Age"] = df_input["BMI"] * df_input["Age"]
    df_input["Glucose_BMI"] = df_input["Glucose"] * df_input["BMI"]
    df_input["Pregnancies_Age"] = df_input["Pregnancies"] / (df_input["Age"] + 1e-6)
    df_input["Insulin_per_Pregnancy"] = df_input["Insulin"] / (df_input["Pregnancies"] + 1)
    
    df_input["Glucose_status"] = pd.cut(df_input["Glucose"],
                                        bins=[0, 100, 125, 200],
                                        labels=["normal", "prediabetes", "diabetes"])
    
    df_input["BMI_status"] = pd.cut(df_input["BMI"],
                                    bins=[0, 18.5, 25, 30, 100],
                                    labels=["underweight", "normal", "overweight", "obese"])
    
    df_input = pd.get_dummies(df_input, columns=["Glucose_status", "BMI_status"], drop_first=True)
    df_input = df_input.reindex(columns=models["feature_cols"], fill_value=0)
    
    input_scaled = models["scaler"].transform(df_input)
    
    # Prediction
    if model_type == "tabnet" and models["tabnet"] is not None:
        prediction = models["tabnet"].predict(input_scaled)[0]
        probability = models["tabnet"].predict_proba(input_scaled)[0, 1]
    else:
        # Fallback to ensemble
        model_type = "ensemble"
        prediction = models["ensemble"].predict(input_scaled)[0]
        probability = models["ensemble"].predict_proba(input_scaled)[0, 1]
        
    # SHAP Explanation (Always using explainer which is tree explainer for ensemble)
    # For TabNet, we can just return the same SHAP values for consistency in UI, 
    # as TabNet has its own interpretability but it requires a different setup.
    shap_values = models["explainer"].shap_values(input_scaled)
    shap_contrib = pd.DataFrame({
        'Feature': models["feature_cols"],
        'SHAP_Value': shap_values[0]
    }).sort_values('SHAP_Value', key=abs, ascending=False).head(5)
    
    return {
        "prediction": int(prediction),
        "probability": float(probability),
        "message": "Diabetes Detected" if prediction == 1 else "No Diabetes",
        "model_used": model_type,
        "top_contributors": shap_contrib.to_dict('records')
    }
