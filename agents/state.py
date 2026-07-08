"""Shared LangGraph pipeline state."""
from typing import Optional, TypedDict


class DiagnosticState(TypedDict):
    wafer_image_path: str
    engineer_note: Optional[str]
    detection_result: Optional[dict]
    detection_latency_ms: Optional[float]
    retrieval_result: Optional[dict]
    retrieval_latency_ms: Optional[float]
    diagnosis_result: Optional[dict]
    diagnosis_latency_ms: Optional[float]
    total_latency_ms: Optional[float]
    pipeline_status: str
    error_message: Optional[str]
