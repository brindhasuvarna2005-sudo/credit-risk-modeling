from fastapi import FastAPI
import joblib
import pandas as pd
import os

app = FastAPI(title="Credit Risk Scoring API")

# -------------------------------
# PATHS
# -------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "xgb_model.pkl")
FEATURE_PATH = os.path.join(BASE_DIR, "..", "models", "feature_columns.pkl")

model = joblib.load(MODEL_PATH)
features = joblib.load(FEATURE_PATH)

THRESHOLD = 0.2


# -------------------------------
# ROUTES
# -------------------------------
@app.get("/")
def home():
    return {"message": "Credit Risk Model Running"}


@app.post("/predict")
def predict(data: dict):

    df = pd.DataFrame([data])
    df = df.reindex(columns=features, fill_value=0)

    prob = model.predict_proba(df)[:, 1][0]
    pred = int(prob > THRESHOLD)

    return {
        "default_probability": float(prob),
        "risk": "High Risk" if pred == 1 else "Low Risk"
    }