import http
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Project paths
PROJECT_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_DIR / "fraud_detection_model.pkl"

# Load the trained model pipeline
model = joblib.load(MODEL_PATH)

app = FastAPI(title="Fraud Detection API",
            description="An API for detecting fraudulent transactions using a trained machine learning model.",
            version="1.0.0")

# Allow CORS for all origins (you can restrict this in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
                "http://localhost:5500", 
                "http://127.0.0.1:5500",
                "http://localhost:8000",
                "http://127.0.0.1:8000"
    ],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TransactionData(BaseModel):
    type: str = Field(..., description="Type of transaction (e.g., 'TRANSFER', 'CASH_OUT', etc.)")
    amount: float = Field(ge=0)
    oldbalanceOrg: float = Field(ge=0)
    newbalanceOrig: float = Field(ge=0)
    oldbalanceDest: float = Field(ge=0)
    newbalanceDest: float = Field(ge=0)
    balanceDiffOrig: float
    balanceDiffDest: float

class PredictionResponse(BaseModel):
    prediction: int
    label: str
    fraud_probability: float
    threshold: float

@app.get("/")
async def root():
    return {"message": "Fraud Detection API is running",
            "docs": "/docs",
    }

@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": model is not None}

@app.post("/predict", response_model=PredictionResponse)
async def predict(transaction: TransactionData):
    # Preserve the order of features as expected by the model
    features = pd.DataFrame(
        [
            {
                "type": transaction.type,
                "amount": transaction.amount,
                "oldbalanceOrg": transaction.oldbalanceOrg,
                "newbalanceOrig": transaction.newbalanceOrig,
                "oldbalanceDest": transaction.oldbalanceDest,
                "newbalanceDest": transaction.newbalanceDest,
                "balanceDiffOrig": transaction.balanceDiffOrig,
                "balanceDiffDest": transaction.balanceDiffDest
            }
        ]
    )

    fraud_probability = model.predict_proba(features)[0][1]

    # Initial decision threshold (can be adjusted based on business requirements)
    threshold = 0.5
    prediction = int(fraud_probability >= threshold)

    return PredictionResponse(
        prediction=prediction,
        label="Fraudulent" if prediction == 1 else "Legitimate",
        fraud_probability=round(fraud_probability,6),
        threshold=threshold,
    )  