import os
import re
import joblib
from pathlib import Path
from typing import Dict, Any
import pymysql
from datetime import datetime

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
                    model = joblib.load(sub / "address_model.joblib")
                    vectorizer = joblib.load(sub / "address_vectorizer.joblib")
                    encoder = joblib.load(sub / "address_encoder.joblib")
                    models[sub.name] = {
                        "model": model,
                        "vectorizer": vectorizer,
                        "encoder": encoder,
                    }
                except Exception:
                    continue

    if not models:
        from dummy_model import DummyModel, DummyVectorizer, DummyEncoder
        models["12"] = {
            "model": DummyModel(),
            "vectorizer": DummyVectorizer(),
            "encoder": DummyEncoder(),
        }

    return models


MODELS = load_models()

_PINCODE_REGEX = re.compile(r"\b(\d{6})\b")

def save_prediction_to_db(shipment_id: str, address: str, pincode: str, predicted_cp: str, confidence: float):
    """Save prediction to MySQL database"""
    try:
        connection = pymysql.connect(
            host='localhost',
            user='cp_service',
            password='$eRv!(e$@12',
            database='cp_predictions'
        )
        cursor = connection.cursor()
        
        insert_query = """
        INSERT INTO predictions (shipment_id, address, pincode, predicted_cp, confidence, model_version, cached, timestamp)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        """
        
        cursor.execute(insert_query, (
            shipment_id, 
            address, 
            pincode, 
            predicted_cp, 
            confidence,
            'v1',  # model_version
            False,  # cached
            datetime.now()
        ))
        
        connection.commit()
        connection.close()
        print(f"✅ Saved prediction for shipment {shipment_id} to database")
        
    except Exception as e:
        print(f"❌ Failed to save prediction to database: {e}")

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
    
    # Save prediction to database
    save_prediction_to_db(
        shipment_id=req.shipment_id,
        address=req.address,
        pincode=result['pincode'],
        predicted_cp=result['predicted_cp'],
        confidence=result['confidence']
    )
    
    return {
        'shipment_id': req.shipment_id,
        'predicted_cp': result['predicted_cp'],
        'confidence': result['confidence'],
    }
