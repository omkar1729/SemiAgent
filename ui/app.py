"""Streamlit UI for the wafer RCA pipeline.

    streamlit run ui/app.py

Upload a MixedWM38 wafer map (.npy), optionally add an engineer note, and get a
structured root cause analysis. Agents are loaded once and cached.
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import streamlit as st

from config import settings
from agents.detection_agent import DetectionAgent, wafer_array_to_pil
from agents.retrieval_agent import RetrievalAgent
from agents.diagnosis_agent import DiagnosisAgent
from pipeline.graph import build_graph

st.set_page_config(page_title="Wafer Defect RCA", page_icon="🔬", layout="wide")


@st.cache_resource(show_spinner="Loading agents (first run downloads models)...")
def load_pipeline():
    detection = DetectionAgent()
    retrieval = RetrievalAgent()
    diagnosis = DiagnosisAgent()
    graph = build_graph(detection, retrieval, diagnosis)
    return graph


def run_pipeline(graph, wafer_array, engineer_note):
    path = tempfile.NamedTemporaryFile(suffix=".npy", delete=False).name
    np.save(path, wafer_array)
    try:
        state = {
            "wafer_image_path": path, "engineer_note": engineer_note or None,
            "detection_result": None, "detection_latency_ms": None,
            "retrieval_result": None, "retrieval_latency_ms": None,
            "diagnosis_result": None, "diagnosis_latency_ms": None,
            "total_latency_ms": None, "pipeline_status": "running", "error_message": None,
        }
        return graph.invoke(state)
    finally:
        if os.path.exists(path):
            os.remove(path)


st.title("🔬 Semiconductor Wafer Defect — Root Cause Analysis")
st.caption(f"Detection mode: **{'ViT' if settings.use_trained_vit else 'CLIP zero-shot'}** · "
           "Detection → Hybrid Retrieval → Diagnosis (LLM)")

with st.sidebar:
    st.header("Input")
    uploaded = st.file_uploader("Wafer map (.npy)", type=["npy"])
    use_sample = st.button("Use bundled sample")
    engineer_note = st.text_area(
        "Engineer note (optional)",
        placeholder="e.g. Edge-ring pattern observed after CMP step, unusual slurry flow noted",
    )
    run = st.button("Run analysis", type="primary")

# Resolve the wafer array from upload / sample.
wafer = None
if uploaded is not None:
    wafer = np.load(uploaded)
elif use_sample or run:
    sample_path = os.path.join(settings.mixedwm38_data_dir, "sample.npy")
    if os.path.exists(sample_path):
        wafer = np.load(sample_path)
    else:
        st.warning("No bundled sample found at data/processed/sample.npy — upload a .npy.")

col_img, col_out = st.columns([1, 2])
if wafer is not None:
    with col_img:
        st.subheader("Wafer map")
        st.image(wafer_array_to_pil(wafer), use_container_width=True)

if run and wafer is not None:
    graph = load_pipeline()
    with st.spinner("Running pipeline..."):
        state = run_pipeline(graph, wafer, engineer_note)

    with col_out:
        det = state.get("detection_result") or {}
        st.subheader("1 · Detection")
        c1, c2 = st.columns(2)
        c1.metric("Defect type", det.get("defect_type", "—"))
        c2.metric("Confidence", f"{det.get('confidence', 0):.1%}")
        if det.get("active_defects"):
            st.write("Active patterns:", ", ".join(det["active_defects"]))

        retr = state.get("retrieval_result") or {}
        st.subheader("2 · Retrieved knowledge")
        for i, d in enumerate(retr.get("retrieved_docs", []), 1):
            with st.expander(f"[{i}] {d.get('source', 'doc')}"):
                st.write(d.get("text", ""))

        st.subheader("3 · Diagnosis")
        if state.get("pipeline_status") == "failed":
            st.error(f"Pipeline failed: {state.get('error_message')}")
        else:
            diag = state.get("diagnosis_result") or {}
            if diag.get("defect_summary"):
                st.markdown(f"**Summary.** {diag['defect_summary']}")
            if diag.get("process_step_analysis"):
                st.markdown(f"**Process analysis.** {diag['process_step_analysis']}")
            for h in diag.get("root_cause_hypotheses", []):
                st.markdown(
                    f"**#{h.get('rank', '?')} ({h.get('confidence_level', '?')})** "
                    f"{h.get('hypothesis', '')}  \n"
                    f"*Evidence:* {h.get('supporting_evidence', '')}")
            params = diag.get("investigation_parameters") or []
            if params:
                st.markdown("**Investigate:**")
                for p in params:
                    st.markdown(f"- {p}")

        if state.get("total_latency_ms"):
            st.caption(f"Total latency: {state['total_latency_ms']:.0f} ms")
elif run and wafer is None:
    st.warning("Provide a wafer map first (upload a .npy or click 'Use bundled sample').")
