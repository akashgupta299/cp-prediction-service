import os
import re
import pickle
from pathlib import Path
from typing import Dict, Any

try:
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel
except Exception:  # pragma: no cover - fallback when fastapi is unavailable
    class HTTPException(Exception):
        def __init__(self, status_code: int, detail: str):
            self.status_code = status_code
            self.detail = detail
    class BaseModel:  # minimal stand-in
        pass
    class FastAPI:  # minimal stand-in
        def __init__(self):
            self.routes = {}
        def post(self, path, **kwargs):
            def decorator(func):
                self.routes[(path, 'POST')] = func
                return func
            return decorator

app = FastAPI()

class PredictionRequest(BaseModel):
    address: str
    shipment_id: str

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"


# Load all available models once during startup
def load_models() -> Dict[str, Dict[str, Any]]:
    models: Dict[str, Dict[str, Any]] = {}
    if MODEL_DIR.exists():
        for sub in MODEL_DIR.iterdir():
            if sub.is_dir():
                try:
                    with open(sub / "model.pkl", "rb") as f:
                        model = pickle.load(f)
                    with open(sub / "vectorizer.pkl", "rb") as f:
                        vectorizer = pickle.load(f)
                    with open(sub / "encoder.pkl", "rb") as f:
                        encoder = pickle.load(f)
                    models[sub.name] = {
                        "model": model,
                        "vectorizer": vectorizer,
                        "encoder": encoder,
                    }
                except Exception:
                    continue

    if not models:
        from .dummy_model import DummyModel, DummyVectorizer, DummyEncoder
        models["12"] = {
            "model": DummyModel(),
            "vectorizer": DummyVectorizer(),
            "encoder": DummyEncoder(),
        }

    return models


MODELS = load_models()

_PINCODE_REGEX = re.compile(r"\b(\d{6})\b")

def _extract_pincode(address: str) -> str:
    match = _PINCODE_REGEX.search(address)
    if not match:
        raise HTTPException(status_code=400, detail="Pincode not found")
    return match.group(1)

def predict_cp(address: str) -> Dict[str, Any]:
    pincode = _extract_pincode(address)
    prefix = pincode[:2]
    bundle = MODELS.get(prefix)
    if not bundle:
        raise HTTPException(status_code=404, detail="No model for pincode prefix")
    model = bundle['model']
    vectorizer = bundle['vectorizer']
    encoder = bundle['encoder']
    features = vectorizer.transform([address])
    probs = model.predict_proba(features)[0]
    best_idx = max(range(len(probs)), key=lambda i: probs[i])
    predicted_cp = encoder.inverse_transform([best_idx])[0]
    confidence = probs[best_idx]
    return {
        'predicted_cp': predicted_cp,
        'confidence': confidence,
        'pincode': pincode,
    }

@app.post("/predict")
def predict(req: PredictionRequest):
    result = predict_cp(req.address)
    return {
        'shipment_id': req.shipment_id,
        'predicted_cp': result['predicted_cp'],
        'confidence': result['confidence'],
    }
