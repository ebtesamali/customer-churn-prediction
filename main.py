from fastapi import FastAPI
from pydantic import BaseModel, Field
import pandas as pd
import joblib
import json
from typing import Any, Dict

app = FastAPI(title="Churn Prediction API", version="1.0")

# Load artifacts
model = joblib.load("model.joblib")

with open("feature_columns.json", "r") as f:
    FEATURE_COLUMNS = json.load(f)

with open("threshold.json", "r") as f:
    THRESHOLD = float(json.load(f)["threshold"])


class UserFeatures(BaseModel):
    # numeric
    num_events: float = Field(..., ge=0)
    num_sessions: float = Field(..., ge=0)
    events_per_session: float = Field(..., ge=0)
    num_songs: float = Field(..., ge=0)
    total_listen_time: float = Field(..., ge=0)
    num_downgrades: float = Field(0, ge=0)
    tenure_days: float = Field(..., ge=0)

    # counts of actions (use API-friendly names)
    Add_Friend: float = Field(0, ge=0)
    Add_to_Playlist: float = Field(0, ge=0)
    Thumbs_Down: float = Field(0, ge=0)
    Thumbs_Up: float = Field(0, ge=0)

    # categoricals (raw, before one-hot)
    last_level: str = Field(..., examples=["free", "paid"])
    device_type: str = Field(..., examples=["Mobile", "Desktop", "Tablet", "Unknown"])


@app.get("/health")
def health() -> Dict[str, Any]:
    return {"status": "ok", "threshold": THRESHOLD}


def build_model_input(payload: UserFeatures) -> pd.DataFrame:
    row = payload.model_dump()

    # map API keys to the original column names used before get_dummies
    rename_map = {
        "Add_Friend": "Add Friend",
        "Add_to_Playlist": "Add to Playlist",
        "Thumbs_Down": "Thumbs Down",
        "Thumbs_Up": "Thumbs Up",
    }
    row = {rename_map.get(k, k): v for k, v in row.items()}

    df = pd.DataFrame([row])

    # replicate training encoding
    df = pd.get_dummies(df, columns=["last_level", "device_type"], drop_first=True)

    # align to training features
    df = df.reindex(columns=FEATURE_COLUMNS, fill_value=0)

    return df


@app.post("/predict")
def predict(payload: UserFeatures) -> Dict[str, Any]:
    X = build_model_input(payload)

    proba = float(model.predict_proba(X)[:, 1][0])
    pred = int(proba >= THRESHOLD)

    return {
        "churn_probability": proba,
        "threshold": THRESHOLD,
        "churn_prediction": pred,
    }
