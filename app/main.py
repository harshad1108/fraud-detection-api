import joblib
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

model = joblib.load("models/model.pkl")
scaler = joblib.load("models/scaler.pkl")

class Transaction(BaseModel):
    features: list

@app.get("/")
def home():
    return {"message": "Fraud Detection API Running"}

@app.post("/predict")
def predict(data: Transaction):
    input_data = np.array(data.features).reshape(1, -1)
    input_data[:, [0, -1]] = scaler.transform(input_data[:, [0, -1]])
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    return {
        "prediction": int(prediction),
        "fraud_probability": float(probability)
    }
