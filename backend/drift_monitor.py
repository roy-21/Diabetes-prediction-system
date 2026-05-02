import pandas as pd
import joblib
import os
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
from evidently import ColumnMapping

ARTIFACTS_DIR = "artifacts"
DATA_PATH = "diabetes.csv"

def generate_drift_report(current_data_df: pd.DataFrame):
    """
    Generates a data drift report comparing incoming current data with reference data
    """
    if not os.path.exists("drift_reports"):
        os.makedirs("drift_reports")

    # Load Reference Data
    reference_data = pd.read_csv(DATA_PATH)
    reference_data.drop(columns=["Outcome"], inplace=True) # Exclude target
    
    # Prepare column mapping
    column_mapping = ColumnMapping()
    column_mapping.numerical_features = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]
    
    # Run Drift Report
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference_data, current_data=current_data_df, column_mapping=column_mapping)
    
    # Save Report
    report_path = "drift_reports/data_drift_report.html"
    report.save_html(report_path)
    
    return report_path
