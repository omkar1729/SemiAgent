"""CLI entry point: run the full RCA pipeline on a single wafer map.

    python pipeline/run_pipeline.py --image_path data/processed/sample.npy
    python pipeline/run_pipeline.py --image_path data/processed/sample.npy \
        --engineer_note "Edge-ring pattern observed after CMP step"
"""
import argparse
import json
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.detection_agent import DetectionAgent
from agents.retrieval_agent import RetrievalAgent
from agents.diagnosis_agent import DiagnosisAgent
from pipeline.graph import build_graph


def main():
    parser = argparse.ArgumentParser(description="Wafer defect RCA pipeline")
    parser.add_argument("--image_path", required=True, help="Path to a wafer map (.npy)")
    parser.add_argument("--engineer_note", default=None, help="Optional free-text note")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    log = logging.getLogger("run_pipeline")

    log.info("Initializing agents...")
    detection_agent = DetectionAgent()
    retrieval_agent = RetrievalAgent()
    diagnosis_agent = DiagnosisAgent()
    graph = build_graph(detection_agent, retrieval_agent, diagnosis_agent)

    initial_state = {
        "wafer_image_path": args.image_path,
        "engineer_note": args.engineer_note or None,
        "detection_result": None,
        "detection_latency_ms": None,
        "retrieval_result": None,
        "retrieval_latency_ms": None,
        "diagnosis_result": None,
        "diagnosis_latency_ms": None,
        "total_latency_ms": None,
        "pipeline_status": "running",
        "error_message": None,
    }

    log.info("Running pipeline on %s", args.image_path)
    final_state = graph.invoke(initial_state)

    print(json.dumps(final_state, indent=2, default=str))

    detection = final_state.get("detection_result") or {}
    diagnosis = final_state.get("diagnosis_result") or {}
    hypotheses = diagnosis.get("root_cause_hypotheses") or []
    top = hypotheses[0].get("hypothesis") if hypotheses else "(none)"
    print("\n=== SUMMARY ===")
    print(f"Status:         {final_state.get('pipeline_status')}")
    print(f"Defect type:    {detection.get('defect_type')}")
    print(f"Detection mode: {detection.get('detection_mode')}")
    print(f"Top hypothesis: {top}")
    print(f"Total latency:  {final_state.get('total_latency_ms')} ms")
    if final_state.get("error_message"):
        print(f"Error:          {final_state['error_message']}")


if __name__ == "__main__":
    main()
