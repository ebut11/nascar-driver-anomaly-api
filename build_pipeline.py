"""Fits the driver-anomaly pipeline and dumps pipeline.joblib.

Run this once (and any time the source CSV changes):

    python build_pipeline.py

Produces pipeline.joblib containing a dict bundle:
    {
        "pipeline": fitted sklearn Pipeline (FormIndexEngineer -> StandardScaler),
        "neighbors": fitted NearestNeighbors model over the scaled training field,
        "drivers": list of driver names, aligned row-for-row with the training matrix,
        "metadata": {"steps": [...], "built_at": "...", "sklearn_version": "...", "n_drivers": N},
    }
"""

from __future__ import annotations

import datetime
from pathlib import Path

import joblib
import pandas as pd
import sklearn
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from pipeline_def import RAW_FEATURES, FormIndexEngineer

CSV_PATH = (
    Path(__file__).parent.parent / "Advanced Short Track Data 2026.csv"
)
OUT_PATH = Path(__file__).parent / "pipeline.joblib"

CSV_TO_FEATURE = {
    "ASP": "asp",
    "ARP": "arp",
    "AFP": "afp",
    "Succ%": "succ_pct",
    "PGAE/100": "pgae_per_100",
    "GR-LR": "gr_lr",
    "SS": "ss",
    "cPOMS": "cpoms",
}


def load_training_frame() -> pd.DataFrame:
    df = pd.read_csv(CSV_PATH)
    df = df.rename(columns=CSV_TO_FEATURE)
    df = df.dropna(subset=RAW_FEATURES)
    return df


def main() -> None:
    df = load_training_frame()
    drivers = df["Driver"].tolist()
    X = df[RAW_FEATURES].to_numpy(dtype=float)

    pipeline = Pipeline(
        steps=[
            ("engineer", FormIndexEngineer(surge_weight=0.5)),
            ("scaler", StandardScaler()),
        ]
    )
    X_scaled = pipeline.fit_transform(X)

    neighbors = NearestNeighbors(n_neighbors=5)
    neighbors.fit(X_scaled)

    bundle = {
        "pipeline": pipeline,
        "neighbors": neighbors,
        "drivers": drivers,
        "metadata": {
            "steps": [name for name, _ in pipeline.steps] + ["neighbors"],
            "built_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "sklearn_version": sklearn.__version__,
            "n_drivers": len(drivers),
            "raw_features": RAW_FEATURES,
        },
    }

    joblib.dump(bundle, OUT_PATH)
    print(f"Wrote {OUT_PATH} ({len(drivers)} drivers, sklearn {sklearn.__version__})")


if __name__ == "__main__":
    main()
