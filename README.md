# 🚀 Advanced Diabetes Prediction System

A production-ready, full-stack AI platform designed to predict diabetes risk using ensemble machine learning and deep learning (TabNet) architectures, backed by SHAP explainability and MLOps principles.

## ✨ Key Features

- **🧠 Multi-Model Architecture**: Switch dynamically between Gradient Boosting Ensemble (XGBoost + LightGBM) and Deep Learning (Google's TabNet).
- **📊 SHAP Explainability**: Understand model decisions with built-in SHAP (SHapley Additive exPlanations) values to identify key contributing factors.
- **🎨 Glassmorphism UI**: Premium React + TailwindCSS frontend offering a dynamic, responsive, and visually stunning user experience.
- **⚡ High-Performance API**: Fully decoupled backend powered by FastAPI for rapid inference and scalability.
- **⚙️ MLOps Integrated**: Automated data drift detection using Evidently AI.
- **🐳 Dockerized**: Fully containerized setup (Frontend & Backend) via `docker-compose` for seamless local deployment.
- **🔄 CI/CD**: Pre-configured GitHub Actions pipeline for automated building and testing.

## 🛠️ Technology Stack

| Component | Technology |
|---|---|
| **Backend API** | FastAPI, Uvicorn, Pydantic |
| **Frontend** | React (Vite), TailwindCSS |
| **Machine Learning** | Scikit-Learn, XGBoost, LightGBM, PyTorch, TabNet |
| **Explainability** | SHAP |
| **MLOps** | Evidently AI |
| **Deployment** | Docker, Docker Compose, GitHub Actions |

## 🚀 Quick Start (Docker)

The easiest way to run the entire application stack is via Docker.

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/diabetes-prediction.git
cd diabetes-prediction

# 2. Build and run the containers
docker-compose up --build
```
- **Frontend App:** `http://localhost:5173`
- **Backend API:** `http://localhost:8000`
- **API Docs (Swagger UI):** `http://localhost:8000/docs`

## 💻 Manual Installation (Local Dev)

If you prefer to run the components separately:

### Backend
```bash
# Install requirements
pip install -r requirements.txt

# Train the TabNet model (Optional)
python train_tabnet.py

# Run the API
uvicorn backend.main:app --reload --port 8000
```

### Frontend
```bash
cd frontend
npm install
npm run dev
```

## 🧠 Model Information

1. **Ensemble Model**: An optimized blend of XGBoost and LightGBM models trained on engineered features (e.g., Glucose/Insulin ratio, BMI*Age).
2. **TabNet Model**: An advanced attentive neural network designed by Google, specifically tailored for high performance on tabular datasets. 

## 🛡️ License
This project is licensed under the MIT License.
