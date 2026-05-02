import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import joblib
import os

ARTIFACTS_DIR = "artifacts"
DATA_PATH = "diabetes.csv" 

# Columns to be considered missing if zero
zero_as_missing = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

# Load training data and prepare global median
df = pd.read_csv(DATA_PATH)
df[zero_as_missing] = df[zero_as_missing].replace(0, np.nan)


def predict_diabetes_with_explanation(input_dict):
    """
    Predict diabetes with SHAP explanation
    """
    # Load saved objects
    model = joblib.load(os.path.join(ARTIFACTS_DIR, "diabetes_ensemble_model.joblib"))
    scaler = joblib.load(os.path.join(ARTIFACTS_DIR, "scaler.joblib"))
    explainer = joblib.load(os.path.join(ARTIFACTS_DIR, "shap_explainer.joblib"))
    feature_cols = np.load(os.path.join(ARTIFACTS_DIR, "feature_columns.npy"), allow_pickle=True)
    
    # Create dataframe from input
    df_input = pd.DataFrame([input_dict])
    
    # Convert Zero to NaN by treating it as missing
    df_input[zero_as_missing] = df_input[zero_as_missing].replace(0, np.nan)
    
    # Impute with global median
    for col in zero_as_missing:
        if col in df_input.columns:
            df_input[col].fillna(df[col].median(), inplace=True)
    
    # Feature Engineering
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
    
    # Align columns
    df_input = df_input.reindex(columns=feature_cols, fill_value=0)
    
    # Scale & predict
    input_scaled = scaler.transform(df_input)
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0, 1]
    
    # SHAP Explanation
    shap_values = explainer.shap_values(input_scaled)
    
    shap_exp = shap.Explanation(
        values=shap_values[0],
        base_values=explainer.expected_value,
        data=df_input.iloc[0],
        feature_names=feature_cols.tolist()
    )
    
    # SHAP Waterfall Plot 
    fig = plt.figure(figsize=(12, 8))
    shap.waterfall_plot(shap_exp)
    plt.title(f"SHAP Explanation - Prediction: {'Diabetes' if prediction == 1 else 'No Diabetes'} (Prob: {probability:.2%})",
              fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Top 5 contributing features
    shap_contrib = pd.DataFrame({
        'Feature': feature_cols,
        'SHAP_Value': shap_values[0]
    }).sort_values('SHAP_Value', key=abs, ascending=False).head(5)
    
    return {
        "prediction": int(prediction),
        "probability": float(probability),
        "message": " Diabetes Detected" if prediction == 1 else " No Diabetes",
        "top_contributors": shap_contrib.to_dict('records'),
        "shap_fig": fig
    }