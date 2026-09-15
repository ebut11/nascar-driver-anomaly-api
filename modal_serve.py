"""Deploys serve.py (the FastAPI driver-anomaly API) on Modal.

Local preview:   modal serve modal_serve.py   -> temporary URL, live reload
Deploy for real: modal deploy modal_serve.py  -> prints the permanent public URL

Ships pipeline_def.py, serve.py, and pipeline.joblib into the image so the
artifact and the class needed to unpickle it are both present at boot.
scikit-learn is pinned to the exact version recorded in the bundle metadata.
"""

from pathlib import Path

import modal

HERE = Path(__file__).parent

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "fastapi==0.141.1",
        "scikit-learn==1.9.0",
        "joblib",
        "numpy",
    )
    .add_local_file(HERE / "pipeline_def.py", "/root/pipeline_def.py")
    .add_local_file(HERE / "serve.py", "/root/serve.py")
    .add_local_file(HERE / "pipeline.joblib", "/root/pipeline.joblib")
)

app = modal.App(name="nascar-driver-anomaly-api", image=image)


@app.function(max_containers=1)
@modal.asgi_app()
def fastapi_app():
    import sys

    sys.path.insert(0, "/root")
    from serve import app as web_app

    return web_app
