import numpy as np
import pandas as pd
import joblib
import os
import torch
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score

ARTIFACTS_DIR = "artifacts"
DATA_PATH = "diabetes.csv"

def prepare_data():
    df = pd.read_csv(DATA_PATH)
    
    # Missing value handling
    zero_as_missing = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
    df[zero_as_missing] = df[zero_as_missing].replace(0, np.nan)
    
    for col in zero_as_missing:
        df[col].fillna(df[col].median(), inplace=True)
        
    # Feature Engineering
    df["Glucose_Insulin_ratio"] = df["Glucose"] / (df["Insulin"] + 1e-6)
    df["BMI_Age"] = df["BMI"] * df["Age"]
    df["Glucose_BMI"] = df["Glucose"] * df["BMI"]
    df["Pregnancies_Age"] = df["Pregnancies"] / (df["Age"] + 1e-6)
    df["Insulin_per_Pregnancy"] = df["Insulin"] / (df["Pregnancies"] + 1)
    
    df["Glucose_status"] = pd.cut(df["Glucose"], bins=[0, 100, 125, 200], labels=["normal", "prediabetes", "diabetes"])
    df["BMI_status"] = pd.cut(df["BMI"], bins=[0, 18.5, 25, 30, 100], labels=["underweight", "normal", "overweight", "obese"])
    
    df = pd.get_dummies(df, columns=["Glucose_status", "BMI_status"], drop_first=True)
    
    # Align with saved scaler features
    feature_cols = np.load(os.path.join(ARTIFACTS_DIR, "feature_columns.npy"), allow_pickle=True)
    
    X = df.drop("Outcome", axis=1)
    y = df["Outcome"]
    
    # Ensure columns match exact order of feature_cols
    X = X.reindex(columns=feature_cols, fill_value=0)
    
    # Scale
    scaler = joblib.load(os.path.join(ARTIFACTS_DIR, "scaler.joblib"))
    X_scaled = scaler.transform(X)
    
    return X_scaled, y.values

def train_and_save_tabnet():
    print("Loading and preparing data...")
    X, y = prepare_data()
    
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print("Initializing TabNet Classifier...")
    clf = TabNetClassifier(
        optimizer_fn=torch.optim.Adam,
        optimizer_params=dict(lr=2e-2),
        scheduler_params={"step_size":50, "gamma":0.9},
        scheduler_fn=torch.optim.lr_scheduler.StepLR,
        mask_type='entmax'
    )
    
    print("Training TabNet...")
    clf.fit(
        X_train=X_train, y_train=y_train,
        eval_set=[(X_train, y_train), (X_valid, y_valid)],
        eval_name=['train', 'valid'],
        eval_metric=['auc'],
        max_epochs=100 , patience=20,
        batch_size=256, virtual_batch_size=128,
        num_workers=0,
        drop_last=False
    )
    
    # Evaluate
    preds = clf.predict(X_valid)
    probs = clf.predict_proba(X_valid)[:, 1]
    
    print(f"Validation Accuracy: {accuracy_score(y_valid, preds):.4f}")
    print(f"Validation AUC: {roc_auc_score(y_valid, probs):.4f}")
    
    # Save model
    model_path = os.path.join(ARTIFACTS_DIR, "tabnet_model")
    saved_filepath = clf.save_model(model_path)
    print(f"TabNet model saved successfully at {saved_filepath}")

if __name__ == "__main__":
    train_and_save_tabnet()
