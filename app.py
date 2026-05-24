from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import pickle
from pathlib import Path

app = FastAPI(title="Obesity Prediction API")

MODEL_PATH = Path(__file__).parent / "obesity_model.pkl"

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)


class ObesityInput(BaseModel):
    Gender: str
    Age: float
    Height: float
    Weight: float
    family_history_with_overweight: str
    FAVC: str
    FCVC: float
    NCP: float
    CAEC: str
    SMOKE: str
    CH2O: float
    SCC: str
    FAF: float
    TUE: float
    CALC: str
    MTRANS: str


@app.get("/")
def home():
    return {"message": "Obesity Prediction API is running"}


@app.post("/predict")
def predict(data: ObesityInput):
    input_df = pd.DataFrame([data.model_dump()])

    prediction = model.predict(input_df)[0]

    result = {"prediction": str(prediction)}

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(input_df)[0]
        classes = model.classes_

        result["probabilities"] = {
            str(cls): float(prob) for cls, prob in zip(classes, probabilities)
        }

    return result
