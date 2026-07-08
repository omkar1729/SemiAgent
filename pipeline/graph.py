"""Sequential LangGraph pipeline: detection -> retrieval -> diagnosis.

Each node is a closure over its agent that measures wall-clock latency and writes
its result into the shared DiagnosticState. Any node that raises marks the
pipeline failed; the conditional router then short-circuits to END.
"""
import time

from langgraph.graph import StateGraph, END

from agents.state import DiagnosticState


def make_detection_node(agent):
    def detection_node(state: DiagnosticState) -> dict:
        try:
            start = time.perf_counter()
            result = agent.detect(state["wafer_image_path"])
            latency = (time.perf_counter() - start) * 1000.0
            return {"detection_result": result, "detection_latency_ms": latency}
        except Exception as exc:
            return {"pipeline_status": "failed",
                    "error_message": f"detection failed: {exc}"}

    return detection_node


def make_retrieval_node(agent):
    def retrieval_node(state: DiagnosticState) -> dict:
        try:
            start = time.perf_counter()
            result = agent.retrieve(state["detection_result"], state.get("engineer_note"))
            latency = (time.perf_counter() - start) * 1000.0
            return {"retrieval_result": result, "retrieval_latency_ms": latency}
        except Exception as exc:
            return {"pipeline_status": "failed",
                    "error_message": f"retrieval failed: {exc}"}

    return retrieval_node


def make_diagnosis_node(agent):
    def diagnosis_node(state: DiagnosticState) -> dict:
        try:
            start = time.perf_counter()
            result = agent.diagnose(
                state["detection_result"],
                state["retrieval_result"],
                state.get("engineer_note"),
            )
            latency = (time.perf_counter() - start) * 1000.0
            total = (
                (state.get("detection_latency_ms") or 0.0)
                + (state.get("retrieval_latency_ms") or 0.0)
                + latency
            )
            return {
                "diagnosis_result": result,
                "diagnosis_latency_ms": latency,
                "total_latency_ms": total,
                "pipeline_status": "complete",
            }
        except Exception as exc:
            return {"pipeline_status": "failed",
                    "error_message": f"diagnosis failed: {exc}"}

    return diagnosis_node


def route(state: DiagnosticState) -> str:
    return "end" if state.get("pipeline_status") == "failed" else "continue"


def build_graph(detection_agent, retrieval_agent, diagnosis_agent):
    graph = StateGraph(DiagnosticState)
    graph.add_node("detection", make_detection_node(detection_agent))
    graph.add_node("retrieval", make_retrieval_node(retrieval_agent))
    graph.add_node("diagnosis", make_diagnosis_node(diagnosis_agent))
    graph.set_entry_point("detection")
    graph.add_conditional_edges("detection", route, {"continue": "retrieval", "end": END})
    graph.add_conditional_edges("retrieval", route, {"continue": "diagnosis", "end": END})
    graph.add_conditional_edges("diagnosis", route, {"continue": END, "end": END})
    return graph.compile()
