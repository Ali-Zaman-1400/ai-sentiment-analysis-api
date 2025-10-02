from fastapi import FastAPI
from pydantic import BaseModel
import yaml
import joblib
from src.preprocessing import clean_text

with open("config.yaml", "r") as f:
    cfg = yaml.safe_load(f)

VEC = joblib.load(cfg["paths"]["vectorizer_file"])
MODEL = joblib.load(cfg["paths"]["model_file"])

app = FastAPI(title="Sentiment Analysis API")

class InputText(BaseModel):
    text: str

@app.post("/predict")
def predict(input: InputText):
    clean = clean_text(input.text)
    X = VEC.transform([clean])
    pred = MODEL.predict(X)[0]
    proba = None
    if hasattr(MODEL, "predict_proba"):
        proba = float(max(MODEL.predict_proba(X)[0]))
    return {"text": input.text, "prediction": pred, "confidence": proba}
