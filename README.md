# NASCAR Driver Anomaly API — Assignment 4

A fitted scikit-learn Pipeline served with FastAPI and deployed on Modal.

## What it does

Fits on `Advanced Short Track Data 2026.csv` (40 drivers, 2026 season). A custom
transformer, `FormIndexEngineer` (in `pipeline_def.py`), learns robust
median/IQR statistics on the training field and engineers two extra features
per driver: a `surge_index` (positions gained above expected, blended with net
rating) and a `consistency_gap` (how much a driver's finish beats their
running position). The Pipeline (`engineer -> StandardScaler`) feeds a fitted
`NearestNeighbors(n_neighbors=5)` model. Given any driver's raw stat line, the
API returns an anomaly score (mean scaled distance to the 5 nearest drivers in
the fitted field) and the names of those nearest comparable drivers.

## Files

| file | what it does |
|---|---|
| `pipeline_def.py` | the custom `FormIndexEngineer` transformer (shared by build + serve) |
| `build_pipeline.py` | fits the pipeline + NearestNeighbors model, dumps `pipeline.joblib` |
| `pipeline.joblib` | the fitted bundle: pipeline, neighbors model, driver names, metadata |
| `serve.py` | FastAPI app — `GET /` (artifact info), `POST /anomaly-score` |
| `modal_serve.py` | wraps `serve.py` in a Modal ASGI app for deployment |
| `postman_collection.json` | Postman collection with assertions against the deployed URL |
| `requirements.txt` | local dev dependencies |

## Rebuild the artifact

```
pip install -r requirements.txt
python build_pipeline.py
```

## Run locally

```
uvicorn serve:app --reload
```
-> http://localhost:8000/docs

## Deploy

```
modal deploy modal_serve.py
```

## Live deployment

- Modal API: https://ebut11--nascar-driver-anomaly-api-fastapi-app.modal.run
- API docs: https://ebut11--nascar-driver-anomaly-api-fastapi-app.modal.run/docs
- Frontend: `/driver-check` page on the nascar-chase-dashboard Vercel site

## Write-up

I chose a nearest-neighbors anomaly-distance model because it's a natural fit
for the NASCAR stats I already track for my season-long Chase project: instead
of predicting a label, it answers "how unusual is this stat line, and who does
it resemble?" — useful for spotting breakout or fluke performances. The custom
transformer, `FormIndexEngineer`, engineers a robust (median/IQR) surge index
and consistency gap from raw loop-data stats before scaling and neighbor
search. Built with scikit-learn 1.9.0.
