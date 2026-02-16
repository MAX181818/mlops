import os
import joblib
import numpy as np
from typing import Literal, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

BASE_DIR = os.path.dirname(__file__)
MODELS_DIR = os.path.join(BASE_DIR, "..", "models")

RF_PATH = os.path.join(MODELS_DIR, "rf_model.joblib")
LR_PATH = os.path.join(MODELS_DIR, "lr_model.joblib")
LE_PATH = os.path.join(MODELS_DIR, "label_encoder.joblib")

missing = [p for p in [RF_PATH, LR_PATH, LE_PATH] if not os.path.exists(p)]
if missing:
    raise FileNotFoundError(f"Faltan archivos en /models: {missing}")

rf_model = joblib.load(RF_PATH)
lr_model = joblib.load(LR_PATH)
label_encoder = joblib.load(LE_PATH)

app = FastAPI(title="Penguins API (multi-model)", version="1.0.0")

# --------- INPUTS ---------
class Features(BaseModel):
    bill_length_mm: float = Field(..., example=39.1)
    bill_depth_mm: float = Field(..., example=18.2)
    flipper_length_mm: float = Field(..., example=181)
    body_mass_g: float = Field(..., example=3750)
    year: int = Field(..., example=2007)
    island: str = Field(..., example="Torgersen")  # Dream / Torgersen / Biscoe
    sex: str = Field(..., example="male")          # male / female
    model_name: Optional[Literal["rf", "lr"]] = Field("rf", description="Modelo a usar: rf o lr")

# --------- FEATURE BUILD ---------
def build_features_vector(f: Features) -> np.ndarray:
    island = f.island.strip().lower()
    sex = f.sex.strip().lower()

    island_Dream = 1 if island == "dream" else 0
    island_Torgersen = 1 if island == "torgersen" else 0
    sex_male = 1 if sex == "male" else 0

    if f.bill_depth_mm == 0:
        raise ValueError("bill_depth_mm no puede ser 0 (bill_ratio).")
    bill_ratio = f.bill_length_mm / f.bill_depth_mm

    if f.flipper_length_mm == 0:
        raise ValueError("flipper_length_mm no puede ser 0 (mass_per_flipper_length).")
    mass_per_flipper_length = f.body_mass_g / f.flipper_length_mm

    X = np.array([[
        f.bill_length_mm,
        f.bill_depth_mm,
        f.flipper_length_mm,
        f.body_mass_g,
        f.year,
        island_Dream,
        island_Torgersen,
        sex_male,
        bill_ratio,
        mass_per_flipper_length
    ]], dtype=float)

    return X

def pick_model(name: str):
    if name == "rf":
        return rf_model
    if name == "lr":
        return lr_model
    raise ValueError("model_name inválido. Usa 'rf' o 'lr'.")

# --------- ENDPOINTS ---------
@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/models")
def models():
    return {
        "available_models": ["rf", "lr"],
        "note": "Envía model_name en el body de /predict para escoger."
    }

@app.post("/predict")
def predict(features: Features):
    try:
        X = build_features_vector(features)
        m = pick_model(features.model_name or "rf")

        pred_id = int(m.predict(X)[0])
        pred_label = label_encoder.inverse_transform([pred_id])[0]

        return {
            "model_used": features.model_name or "rf",
            "prediction": str(pred_label),
            "class_id": pred_id
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))