from pathlib import Path
from typing import Dict

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

BASE = Path(__file__).resolve().parent
ARTIFACT = joblib.load(BASE / "model.joblib")
import json
META = json.loads((BASE / "model_metadata.json").read_text(encoding="utf-8"))
MODEL = ARTIFACT["model"]
FEATURE_COLUMNS = ARTIFACT["feature_columns"]
MEANS = ARTIFACT["numeric_means"]
STATUS = ARTIFACT["status_mapping"]
LABELS = ARTIFACT["status_labels"]

app = FastAPI(
    title="Customer Delinquency Prediction API",
    version="2.0.0",
    description="FastAPI layer for the actual Random Forest delinquency model."
)

class PredictionRequest(BaseModel):
    age: int = Field(ge=18, le=100)
    income: float = Field(ge=0)
    credit_score: float = Field(ge=300, le=850)
    credit_utilization: float = Field(ge=0, le=1)
    missed_payments: int = Field(ge=0)
    delinquent_account: int = Field(ge=0, le=1)
    loan_balance: float = Field(ge=0)
    debt_to_income_ratio: float = Field(ge=0, le=1)
    employment_status: str
    account_tenure: int = Field(ge=0)
    credit_card_type: str
    location: str
    month_1: str
    month_2: str
    month_3: str
    month_4: str
    month_5: str

def make_features(req: PredictionRequest):
    if any(m not in STATUS for m in [
        req.month_1, req.month_2, req.month_3, req.month_4, req.month_5
    ]):
        raise HTTPException(status_code=400, detail="Invalid payment status.")

    row = pd.DataFrame([{
        "Age": req.age,
        "Income": req.income,
        "Credit_Score": req.credit_score,
        "Credit_Utilization": req.credit_utilization,
        "Missed_Payments": req.missed_payments,
        "Delinquent_Account": req.delinquent_account,
        "Loan_Balance": req.loan_balance,
        "Debt_to_Income_Ratio": req.debt_to_income_ratio,
        "Employment_Status": req.employment_status,
        "Account_Tenure": req.account_tenure,
        "Credit_Card_Type": req.credit_card_type,
        "Location": req.location,
        "Month_1": STATUS[req.month_1],
        "Month_2": STATUS[req.month_2],
        "Month_3": STATUS[req.month_3],
        "Month_4": STATUS[req.month_4],
        "Month_5": STATUS[req.month_5],
    }])

    # Match the original notebook's preprocessing.
    row["Income"] = row["Income"].fillna(MEANS["Income"])
    row["Credit_Score"] = row["Credit_Score"].fillna(MEANS["Credit_Score"])
    row["Loan_Balance"] = row["Loan_Balance"].fillna(MEANS["Loan_Balance"])

    row = pd.get_dummies(
        row,
        columns=["Employment_Status", "Credit_Card_Type", "Location"]
    )
    row = row.reindex(columns=FEATURE_COLUMNS, fill_value=0)
    return row

@app.get("/health")
def health():
    return {"status": "healthy", "model": "Random Forest", "target": "Month_6"}

@app.get("/model-info")
def model_info():
    return {
        "model": "Random Forest Classifier",
        "target": "Month_6",
        "training_rows": META["rows"],
        "accuracy": META["accuracy"],
        "note": "Accuracy is reported in the UI from model_metadata.json."
    }

@app.post("/predict")
def predict(req: PredictionRequest):
    X = make_features(req)
    prediction = int(MODEL.predict(X)[0])
    probabilities = MODEL.predict_proba(X)[0]
    result = {
        LABELS[prediction]: round(float(probabilities[list(MODEL.classes_).index(prediction)]), 4)
        for prediction in MODEL.classes_
    }
    confidence = float(max(probabilities))

    factors = sorted(
        zip(FEATURE_COLUMNS, MODEL.feature_importances_),
        key=lambda x: x[1],
        reverse=True
    )[:6]

    return {
        "prediction": LABELS[prediction],
        "confidence": round(confidence, 4),
        "probabilities": result,
        "top_model_factors": [
            {"feature": name, "importance": round(float(value), 4)}
            for name, value in factors
        ]
    }
