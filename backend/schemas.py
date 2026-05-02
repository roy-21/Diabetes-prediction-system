from pydantic import BaseModel, Field
from typing import Dict, Any, List

class PredictionRequest(BaseModel):
    Pregnancies: int = Field(..., ge=0, le=20)
    Glucose: float = Field(..., ge=0.0)
    BloodPressure: float = Field(..., ge=0.0)
    SkinThickness: float = Field(..., ge=0.0)
    Insulin: float = Field(..., ge=0.0)
    BMI: float = Field(..., ge=0.0)
    DiabetesPedigreeFunction: float = Field(..., ge=0.0)
    Age: int = Field(..., ge=1, le=120)
    model_type: str = Field(default="ensemble", description="ensemble or tabnet")

class TopContributor(BaseModel):
    Feature: str
    SHAP_Value: float

class PredictionResponse(BaseModel):
    prediction: int
    probability: float
    message: str
    model_used: str
    top_contributors: List[TopContributor]
