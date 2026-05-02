from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from backend.schemas import PredictionRequest, PredictionResponse
from backend.predict import predict_diabetes_api

app = FastAPI(
    title="Diabetes Prediction API",
    description="Advanced ML API for Diabetes Prediction with TabNet & Ensemble models.",
    version="2.0.0"
)

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health_check():
    return {"status": "healthy", "version": "2.0.0"}

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    try:
        input_data = request.dict(exclude={"model_type"})
        model_type = request.model_type
        
        result = predict_diabetes_api(input_data, model_type)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
