from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

# Load saved models and scaler
scaler = joblib.load("credit_scaler.pkl")
credit_model = joblib.load("best_credit_model.pkl")
fraud_model = joblib.load("best_fraud_model.pkl")

# Define input schema
class CustomerData(BaseModel):
    # include the features you trained on
    age: int
    income: float
    tenure: int
    avg_transaction_amount: float
    transaction_count: int
    fraud_rate: float
    # add any other engineered features here

# Create FastAPI app
app = FastAPI(title="MTech ML API for IntelliScore", description="Credit Scoring & Fraud Detection Service", version="1.0")

@app.post("/predict/")
def predict(data: CustomerData):
    # Convert input to DataFrame
    df = pd.DataFrame([data.dict()])

    # Scale features for credit scoring (only if scaler was used during training)
    scaled_features = scaler.transform(df)

    # Predictions
    credit_pred = credit_model.predict(scaled_features)[0]
    credit_proba = credit_model.predict_proba(scaled_features)[0][1]

    fraud_pred = fraud_model.predict(df)[0]
    fraud_proba = fraud_model.predict_proba(df)[0][1]

    return {
        "credit_score_prediction": int(credit_pred),
        "credit_score_probability": float(credit_proba),
        "fraud_prediction": int(fraud_pred),
        "fraud_probability": float(fraud_proba)
    }