from __future__ import annotations

import shutil
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app import geocode, rag, sandbox, weather

BASE_DIR = Path(__file__).resolve().parents[1]
WEB_DIR = BASE_DIR / "web"
UPLOAD_DIR = BASE_DIR / "uploads"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
INDEX_PATH = ARTIFACTS_DIR / "index.json"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Agentic RAG Chatbot")
app.mount("/static", StaticFiles(directory=WEB_DIR / "static"), name="static")

INDEX_STATE: dict = {"index": None, "files": []}


@app.get("/")
def index() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")


def _safe_filename(name: str) -> str:
    return Path(name).name


def _load_index() -> Optional[dict]:
    if INDEX_STATE.get("index") is not None:
        return INDEX_STATE["index"]
    if INDEX_PATH.exists():
        index = rag.load_index(INDEX_PATH)
        INDEX_STATE["index"] = index
        return index
    return None


@app.post("/api/ingest")
async def ingest(
    files: list[UploadFile] = File(...),
    use_embeddings: bool = Form(False),
    embed_model: str = Form(rag.EMBED_MODEL_DEFAULT),
) -> dict:
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")

    saved_paths: list[Path] = []
    for f in files:
        if not f.filename:
            continue
        filename = _safe_filename(f.filename)
        if not filename:
            continue
        path = UPLOAD_DIR / filename
        with path.open("wb") as buffer:
            shutil.copyfileobj(f.file, buffer)
        saved_paths.append(path)

    if not saved_paths:
        raise HTTPException(status_code=400, detail="No valid files to ingest")

    try:
        index = rag.build_index(saved_paths, use_embeddings=use_embeddings, embed_model=embed_model)
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    rag.save_index(index, INDEX_PATH)
    INDEX_STATE["index"] = index
    INDEX_STATE["files"] = [p.name for p in saved_paths]

    return {
        "ingested": len(saved_paths),
        "files": INDEX_STATE["files"],
        "has_embeddings": index.get("has_embeddings", False),
        "embed_model": index.get("embed_model"),
    }


@app.post("/api/ask")
async def ask(
    question: str = Form(...),
    top_k: int = Form(4),
    citations_k: int = Form(3),
    use_embeddings: bool = Form(False),
    embed_model: str = Form(rag.EMBED_MODEL_DEFAULT),
) -> dict:
    index = _load_index()
    if index is None:
        raise HTTPException(status_code=400, detail="No index found. Upload files first.")

    intent = rag.detect_intent(question)
    require_overlap = intent not in ("summary",)
    min_score = 0.0 if intent == "summary" else 0.1
    hits = rag.search(
        index,
        question,
        top_k=top_k,
        min_score=min_score,
        use_embeddings=use_embeddings,
        embed_model=embed_model,
        require_overlap=require_overlap,
    )
    answer, citations = rag.build_answer(hits, query=question, max_citations=citations_k)
    return {
        "question": question,
        "answer": answer,
        "citations": citations,
    }


@app.post("/api/weather")
async def weather_api(
    start: str = Form(...),
    end: str = Form(...),
    city: Optional[str] = Form(None),
    country: Optional[str] = Form(None),
    lat: Optional[float] = Form(None),
    lon: Optional[float] = Form(None),
    location_name: Optional[str] = Form(None),
    sandbox_mode: str = Form("none"),
) -> dict:
    if sandbox_mode not in ("none", "docker"):
        raise HTTPException(status_code=400, detail="Invalid sandbox mode")

    try:
        resolved_name = None
        if city:
            geo = geocode.geocode_city(city, country=country)
            lat = float(geo["latitude"])
            lon = float(geo["longitude"])
            name = geo.get("name") or city
            parts = [name]
            if geo.get("admin1"):
                parts.append(str(geo["admin1"]))
            if geo.get("country_code"):
                parts.append(str(geo["country_code"]))
            resolved_name = ", ".join(parts)

        if lat is None or lon is None:
            raise ValueError("Provide a city or explicit latitude/longitude.")

        if sandbox_mode == "docker":
            result = sandbox.run_weather_in_docker(lat, lon, start, end).payload
        else:
            result = weather.run_weather_analysis(lat, lon, start, end)
    except sandbox.SandboxError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if resolved_name:
        result["location_name"] = resolved_name
    elif location_name:
        result["location_name"] = location_name
    return result


@app.post("/api/geocode")
async def geocode_api(
    query: str = Form(...),
    country: Optional[str] = Form(None),
    limit: int = Form(5),
) -> dict:
    try:
        results = geocode.geocode_search(query, country=country, limit=limit)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"results": results}
