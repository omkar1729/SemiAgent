"""FastAPI service for the wafer RCA pipeline.

Agents are loaded once at startup (lifespan) and reused across requests — never
instantiated inside a handler (HARD RULE 3).
"""
import os
import sys
import tempfile
from contextlib import asynccontextmanager
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, File, Form, UploadFile

from config import settings
from agents.detection_agent import DetectionAgent
from agents.retrieval_agent import RetrievalAgent
from agents.diagnosis_agent import DiagnosisAgent
from pipeline.graph import build_graph


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.detection_agent = DetectionAgent()
    app.state.retrieval_agent = RetrievalAgent()
    app.state.diagnosis_agent = DiagnosisAgent()
    app.state.graph = build_graph(
        app.state.detection_agent,
        app.state.retrieval_agent,
        app.state.diagnosis_agent,
    )
    yield


app = FastAPI(title="Wafer RCA System", lifespan=lifespan)


@app.post("/diagnose")
async def diagnose(file: UploadFile = File(...), engineer_note: Optional[str] = Form(None)):
    tmp = tempfile.NamedTemporaryFile(suffix=".npy", delete=False)
    try:
        tmp.write(await file.read())
        tmp.close()
        state = {
            "wafer_image_path": tmp.name,
            "engineer_note": engineer_note or None,
            "detection_result": None, "detection_latency_ms": None,
            "retrieval_result": None, "retrieval_latency_ms": None,
            "diagnosis_result": None, "diagnosis_latency_ms": None,
            "total_latency_ms": None, "pipeline_status": "running", "error_message": None,
        }
        return app.state.graph.invoke(state)
    finally:
        if os.path.exists(tmp.name):
            os.remove(tmp.name)


@app.get("/health")
async def health():
    try:
        chroma_doc_count = app.state.retrieval_agent.retriever.vectorstore._collection.count()
    except Exception:
        chroma_doc_count = None
    return {
        "status": "ok",
        "detection_agent_loaded": hasattr(app.state, "detection_agent"),
        "retrieval_agent_loaded": hasattr(app.state, "retrieval_agent"),
        "diagnosis_agent_loaded": hasattr(app.state, "diagnosis_agent"),
        "detection_mode": "vit" if settings.use_trained_vit else "clip",
        "chroma_doc_count": chroma_doc_count,
    }


@app.get("/metrics")
async def metrics():
    import mlflow

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name("wafer-rca")
    if experiment is None:
        return {"message": "No runs found", "metrics": {}}
    runs = client.search_runs([experiment.experiment_id],
                              order_by=["attributes.start_time DESC"], max_results=1)
    if not runs:
        return {"message": "No runs found", "metrics": {}}
    return {"run_id": runs[0].info.run_id, "metrics": dict(runs[0].data.metrics)}
