"""FastAPI service for the driver-anomaly-distance pipeline.

Local dev:
    uvicorn serve:app --reload
    -> http://localhost:8000/docs
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from pipeline_def import RAW_FEATURES, FormIndexEngineer  # noqa: F401 -- required to unpickle bundle

ARTIFACT_PATH = Path(__file__).parent / "pipeline.joblib"

app = FastAPI(title="NASCAR Driver Anomaly API")

_bundle: Optional[dict] = None
_load_error: Optional[str] = None

try:
    _bundle = joblib.load(ARTIFACT_PATH)
except Exception as exc:  # noqa: BLE001 -- any load failure -> 503, not a crash at import
    _load_error = str(exc)


class DriverStats(BaseModel):
    driver: str = Field(..., min_length=1, max_length=80, description="Driver name (for display only)")
    asp: float = Field(..., ge=1, le=45, description="Average Starting Position")
    arp: float = Field(..., ge=1, le=45, description="Average Running Position")
    afp: float = Field(..., ge=1, le=45, description="Average Finish Position")
    succ_pct: float = Field(..., ge=0, le=100, description="Success Rate percentage")
    pgae_per_100: float = Field(..., ge=-50, le=50, description="Positions Gained Above Expected per 100 laps")
    gr_lr: float = Field(..., ge=-10, le=10, description="Net Rating (Gain Rating - Loss Rating)")
    ss: float = Field(..., ge=-50, le=50, description="Speed Score")
    cpoms: float = Field(..., ge=0.5, le=1.5, description="Continuously graded Percent of Max Speed")


class NearestDriver(BaseModel):
    driver: str
    distance: float


class AnomalyResponse(BaseModel):
    driver: str
    anomaly_score: float
    nearest_drivers: list[NearestDriver]
    engineered: dict


def _require_bundle() -> dict:
    if _bundle is None:
        raise HTTPException(status_code=503, detail=f"Pipeline artifact unavailable: {_load_error}")
    return _bundle


@app.get("/")
def artifact_info():
    bundle = _require_bundle()
    return {
        "service": "nascar-driver-anomaly",
        "metadata": bundle["metadata"],
        "raw_features": RAW_FEATURES,
    }


@app.post("/anomaly-score", response_model=AnomalyResponse)
def anomaly_score(stats: DriverStats):
    bundle = _require_bundle()
    pipeline = bundle["pipeline"]
    neighbors = bundle["neighbors"]
    drivers = bundle["drivers"]

    row = np.array([[getattr(stats, f) for f in RAW_FEATURES]], dtype=float)
    transformed = pipeline.named_steps["engineer"].transform(row)
    scaled = pipeline.named_steps["scaler"].transform(transformed)

    distances, indices = neighbors.kneighbors(scaled, n_neighbors=5)
    distances, indices = distances[0], indices[0]

    engineered_names = list(RAW_FEATURES) + ["surge_index", "consistency_gap"]
    engineered = dict(zip(engineered_names, transformed[0].tolist()))

    return AnomalyResponse(
        driver=stats.driver,
        anomaly_score=float(np.mean(distances)),
        nearest_drivers=[
            NearestDriver(driver=drivers[i], distance=float(d))
            for i, d in zip(indices, distances)
        ],
        engineered=engineered,
    )
