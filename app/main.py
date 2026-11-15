"""Web API and page for the facies classifier."""
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from . import core

STATIC_DIR = Path(__file__).parent.parent / "static"
MAX_BYTES = 5 * 1024 * 1024

app = FastAPI(
    title="Facies Classifier API",
    description="Predicts lithofacies from the wireline logs of one well. Model trained on the SEG 2016 machine learning contest dataset; the example wells are synthetic.",
    version="0.1.0-beta",
)
model = core.load_model()


def run(content: bytes) -> dict:
    try:
        well = core.validate(core.read_csv(content), model)
    except core.InputError as err:
        raise HTTPException(status_code=422, detail=str(err)) from err
    return core.predict(well, model)


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "features": len(model.settings["features"]),
            "facies": len(model.settings["facies"])}


@app.get("/api/demo")
def list_demo() -> list[dict]:
    return [{"id": key, "name": key.upper()} for key in core.demo_wells()]


@app.get("/api/demo/{well_id}")
def predict_demo(well_id: str) -> dict:
    path = core.demo_wells().get(well_id)
    if path is None:
        raise HTTPException(status_code=404, detail=f"Unknown demo well: {well_id}")
    return run(path.read_bytes())


@app.get("/api/demo/{well_id}/csv")
def download_demo(well_id: str) -> FileResponse:
    path = core.demo_wells().get(well_id)
    if path is None:
        raise HTTPException(status_code=404, detail=f"Unknown demo well: {well_id}")
    return FileResponse(path, media_type="text/csv", filename=f"{well_id}.csv")


@app.post("/api/predict")
async def predict_upload(file: UploadFile = File(..., description="CSV file with the logs of one well")) -> dict:
    content = await file.read(MAX_BYTES + 1)
    if len(content) > MAX_BYTES:
        raise HTTPException(status_code=413, detail="The file is larger than 5 MB.")
    return run(content)


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
