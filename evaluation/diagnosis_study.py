"""Unified diagnosis-quality study (replaces eval_diagnosis + ablation + ablation_factorial).

One fixed, nested, stratified test sample (N = 10,30,50,100; sample[:10] ⊂ sample[:30]
⊂ … ⊂ sample[:100]) is scored ONCE and aggregated over the nested prefixes. Every
condition shares the same baseline so nothing is computed twice.

Baseline  B0 = ViT-trained detector · no engineer note · hybrid retrieval · GPT-4o-mini

Conditions per image (all GPT + hybrid unless stated):
  Detector×Note grid (factors 1 & 3, incl. interaction)
    CLIP|nonote   CLIP|note
    ViT_untrained|nonote   ViT_untrained|note
    ViT_trained|nonote (= B0)   ViT_trained|note
  Component knockout (factor 5 — vary ONE thing from B0)
    B0 + sparse-only (BM25) retrieval
    B0 + no retrieval
    B0 with oracle (ground-truth) detection
  LLM arm (optional, skipped if SemiKong unreachable)
    B0 with SemiKong-8B

Diagnosis quality (factor 4) is the rubric overall on a 0-10 scale; the per-dimension
breakdown is reported for B0. Each size logs the grid table + marginals, knockout
table, and Wilcoxon paired tests to MLflow and CSV.
"""
import itertools
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.seed import set_all_seeds
from config import settings

set_all_seeds(settings.random_seed)

import mlflow
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from agents.detection_agent import CLASS_NAMES, DetectionAgent
from agents.retrieval_agent import RetrievalAgent
from agents.diagnosis_agent import DiagnosisAgent
from evaluation.rubric import DiagnosticRubric, RUBRIC_DIMENSIONS

RESULTS_DIR = "evaluation/results"
EVAL_NS = sorted({int(x) for x in os.environ.get("EVAL_NS", "10,30,50,100").split(",")})
# Conditions per image are I/O-bound LLM calls -> run concurrently (bounded so we stay
# under the OpenAI 200K TPM limit; the rubric itself also fans out 5 calls and backs off
# 429s). 4 workers x full-doc prompts overran the TPM cap, so default to 2.
WORKERS = int(os.environ.get("STUDY_WORKERS", "2"))
SINGLE_CLASSES = list(range(1, 9))  # 8 single-defect classes, excluding Normal(0)
BASELINE_DET = "ViT_trained"
DETECTORS = ["CLIP", "ViT_untrained", "ViT_trained"]
NOTE_LEVELS = ["nonote", "note"]

# A defect-accurate engineer observation per single-defect class (CLASS_NAMES 1-8).
NOTE_TEMPLATES = {
    1: "Engineer note: a dense cluster of failing dies is concentrated at the wafer center, seen after the spin-coat/anneal steps.",
    2: "Engineer note: a ring-shaped band of failing dies surrounds the wafer center, following recent develop / edge-bead-removal changes.",
    3: "Engineer note: failing dies are localized along one section of the wafer edge, possibly a focus or handling issue.",
    4: "Engineer note: a complete ring of failing dies sits at the outer wafer edge, observed after the CMP step.",
    5: "Engineer note: a small localized cluster of failing dies appears in one region, consistent with a particle event.",
    6: "Engineer note: nearly the entire wafer is failing, seen after a wet-etch / deposition step.",
    7: "Engineer note: failing dies are scattered randomly across the wafer, suggesting particle or ESD events.",
    8: "Engineer note: a thin linear track of failing dies crosses the wafer, consistent with a handling scratch.",
}


def base_query(defect_type: str) -> str:
    return f"{defect_type} semiconductor wafer defect process cause mechanism"


# SemiKong-8B-GPTQ serves a 3072-token context, so the 5 full retrieved docs (≈1.3-2.6k
# tokens) plus the CoT prompt + 1024-token completion would overflow it. The SemiKong
# arm therefore receives truncated doc snippets; GPT (B0) keeps the full docs. This is a
# documented asymmetry driven by the smaller model's context window.
SEMIKONG_DOC_CHARS = int(os.environ.get("SEMIKONG_DOC_CHARS", "500"))


def truncate_retrieval(retr, cap=SEMIKONG_DOC_CHARS):
    return {"retrieved_docs": [{**d, "text": d.get("text", "")[:cap]} for d in retr["retrieved_docs"]],
            "query_used": retr["query_used"]}


class TemplateDiagnoser:
    """Degraded Diagnosis Agent (RQ3 'Template Diagnosis' baseline): builds a
    structured report from the detected defect + retrieved snippets with NO LLM
    reasoning, so the rubric gap between this and the LLM isolates the LLM's value."""
    model = "template"
    is_local = False

    def diagnose(self, detection_result, retrieval_result, engineer_note=None):
        defect = detection_result.get("defect_type", "unknown")
        docs = retrieval_result.get("retrieved_docs", [])
        hyps = []
        for i, d in enumerate(docs[:3]):
            snippet = (d.get("text", "") or "")[:200]
            hyps.append({"rank": i + 1,
                         "hypothesis": f"Cause associated with {defect}: {snippet}",
                         "supporting_evidence": f"[{i + 1}]",
                         "confidence_level": "Medium"})
        return {
            "defect_summary": f"Detected defect pattern: {defect}.",
            "process_step_analysis": f"Review process steps commonly linked to {defect} (see retrieved documents).",
            "root_cause_hypotheses": hyps,
            "investigation_parameters": ["Inspect the process step indicated by the cited documents",
                                         "Compare against baseline wafers",
                                         "Check tool logs for the relevant step"],
            "reasoning_chain": "[template diagnosis - no LLM]",
            "llm_model_used": "template",
        }


def oracle_detection(label_row) -> dict:
    return {
        "defect_type": CLASS_NAMES[int(np.argmax(label_row))],
        "confidence": 1.0,
        "active_defects": [CLASS_NAMES[i] for i in range(len(CLASS_NAMES)) if label_row[i] == 1],
        "all_probabilities": {CLASS_NAMES[i]: float(label_row[i]) for i in range(len(CLASS_NAMES))},
        "detection_mode": "oracle",
    }


def build_roundrobin_sample(primary, test_idx, n):
    """Stratified sample whose prefixes stay class-balanced (round-robin over the 8
    single-defect classes), so sample[:10] ⊂ sample[:30] ⊂ … ⊂ sample[:100]."""
    rng = np.random.default_rng(settings.random_seed)
    pools = {c: list(rng.permutation(test_idx[primary[test_idx] == c])) for c in SINGLE_CLASSES}
    sample, i = [], 0
    while len(sample) < n and any(pools.values()):
        c = SINGLE_CLASSES[i % len(SINGLE_CLASSES)]
        if pools[c]:
            sample.append(int(pools[c].pop()))
        i += 1
    return sample


def semikong_agent_or_none():
    """Return a reachable SemiKong DiagnosisAgent, else None (LLM arm is skipped)."""
    try:
        a = DiagnosisAgent(provider=settings.semikong_backend)
        if not a.is_local:
            a.client.with_options(timeout=20.0).chat.completions.create(
                model=a.model, messages=[{"role": "user", "content": "ping"}], max_tokens=2)
        return a
    except Exception as exc:
        print(f"SemiKong unavailable -> LLM arm skipped ({exc}).")
        return None


def wilcoxon_safe(a, b):
    try:
        return float(wilcoxon(a, b).pvalue)
    except Exception:
        return float("nan")


def build_conditions(has_semikong):
    """List of (key, label) condition descriptors evaluated per image."""
    conds = []
    for d in DETECTORS:                       # detector x note grid
        for n in NOTE_LEVELS:
            conds.append((f"grid|{d}|{n}", f"{d} · {n}"))
    conds += [
        ("knock|sparse", "B0 + sparse-only retrieval"),
        ("knock|noretr", "B0 + no retrieval"),
        ("knock|oracle", "B0 + oracle detection"),
        ("knock|template", "B0 + template diagnosis (no LLM)"),
        ("knock|gpt_trunc", "B0 + GPT on truncated docs"),
    ]
    if has_semikong:
        conds.append(("llm|semikong", "B0 with SemiKong (truncated docs)"))
    return conds


def main():
    data = np.load(os.path.join(settings.mixedwm38_data_dir, "data.npy"))
    labels = np.load(os.path.join(settings.mixedwm38_data_dir, "labels.npy"))
    test_idx = np.load(os.path.join(settings.mixedwm38_data_dir, "test_indices.npy"))
    primary = labels.argmax(axis=1)

    sample = build_roundrobin_sample(primary, test_idx, max(EVAL_NS))
    os.makedirs(RESULTS_DIR, exist_ok=True)
    for N in EVAL_NS:  # nested prefix index files (reused/inspectable)
        np.save(os.path.join(RESULTS_DIR, f"study_{N}_indices.npy"), np.array(sample[:N]))
    print(f"Scoring {len(sample)} images; reporting nested sizes {EVAL_NS}.")

    retrieval_agent = RetrievalAgent()
    rubric = DiagnosticRubric()
    bm25_index = retrieval_agent.retriever.bm25_index
    bm25_corpus = retrieval_agent.retriever.bm25_corpus
    detectors = {
        "CLIP": DetectionAgent(mode="clip"),
        "ViT_untrained": DetectionAgent(mode="vit", load_vit_checkpoint=False),
        "ViT_trained": DetectionAgent(mode="vit", load_vit_checkpoint=True),
    }
    gpt = DiagnosisAgent(provider="openai")
    template_agent = TemplateDiagnoser()
    semikong = semikong_agent_or_none()
    conditions = build_conditions(bool(semikong))

    scores = {k: [] for k, _ in conditions}          # overall /10
    dims = {k: [] for k, _ in conditions}            # per-dimension dicts
    latencies = {k: [] for k, _ in conditions}       # diagnose-call latency (ms)
    e2e_det, e2e_retr = [], []                        # B0 detection + retrieval latency (ms) for end-to-end
    per_image_rows = []

    for step, idx in enumerate(sample):
        wafer, label_row = data[idx], labels[idx]
        true_cls = int(primary[idx])
        note_text = NOTE_TEMPLATES.get(true_cls)
        det_lat = {}
        dets = {}
        for name, det in detectors.items():
            _t = time.perf_counter()
            dets[name] = det.detect(wafer)
            det_lat[name] = (time.perf_counter() - _t) * 1000.0
        oracle = oracle_detection(label_row)

        # Retrieval is local + cheap; compute every variant the conditions need.
        hybrid = {(d, n): retrieval_agent.retrieve(dets[d], note_text if n == "note" else None)
                  for d in DETECTORS for n in NOTE_LEVELS}
        hybrid_oracle = retrieval_agent.retrieve(oracle, None)
        # Time the B0 retrieval once for the end-to-end latency estimate.
        _t = time.perf_counter()
        retrieval_agent.retrieve(dets[BASELINE_DET], None)
        e2e_det.append(det_lat[BASELINE_DET])
        e2e_retr.append((time.perf_counter() - _t) * 1000.0)
        sparse_q = base_query(dets[BASELINE_DET]["defect_type"])
        sparse_texts = bm25_index.get_top_n(sparse_q.split(), bm25_corpus, n=settings.retrieval_top_k)
        sparse_retr = {"retrieved_docs": [{"text": t, "source": "bm25"} for t in sparse_texts],
                       "query_used": sparse_q}
        empty_retr = {"retrieved_docs": [], "query_used": ""}

        # (condition key) -> (detection dict, retrieval dict, note text, agent)
        plan = {}
        for d in DETECTORS:
            for n in NOTE_LEVELS:
                plan[f"grid|{d}|{n}"] = (dets[d], hybrid[(d, n)],
                                         note_text if n == "note" else None, gpt)
        b0_det = dets[BASELINE_DET]
        plan["knock|sparse"] = (b0_det, sparse_retr, None, gpt)
        plan["knock|noretr"] = (b0_det, empty_retr, None, gpt)
        plan["knock|oracle"] = (oracle, hybrid_oracle, None, gpt)
        plan["knock|template"] = (b0_det, hybrid[(BASELINE_DET, "nonote")], None, template_agent)
        # GPT on the SAME truncated docs SemiKong gets -> fair, apples-to-apples LLM compare.
        plan["knock|gpt_trunc"] = (b0_det, truncate_retrieval(hybrid[(BASELINE_DET, "nonote")]), None, gpt)
        if semikong:  # truncated docs to fit SemiKong's 3072-token context
            plan["llm|semikong"] = (b0_det, truncate_retrieval(hybrid[(BASELINE_DET, "nonote")]),
                                    None, semikong)

        row = {"image_index": int(idx), "true_defect": CLASS_NAMES[true_cls],
               "CLIP_pred": dets["CLIP"]["defect_type"],
               "ViT_untrained_pred": dets["ViT_untrained"]["defect_type"],
               "ViT_trained_pred": dets["ViT_trained"]["defect_type"]}

        def _score(key):
            detection, retrieval, note, agent = plan[key]
            # Resilience net on top of the SDK's own 8 retries: a sustained TPM 429
            # (which the SDK can't outlast) must not discard the whole run's work.
            for attempt in range(6):
                try:
                    t0 = time.perf_counter()
                    diagnosis = agent.diagnose(detection, retrieval, note)
                    lat = (time.perf_counter() - t0) * 1000.0
                    sc = rubric.auto_score(diagnosis, retrieval["retrieved_docs"])
                    return key, sc, lat
                except Exception as exc:
                    if attempt == 5:
                        raise
                    print(f"  [retry {attempt + 1}/5] {key}: {type(exc).__name__}; backing off")
                    time.sleep(min(60, 10 * (attempt + 1)))

        with ThreadPoolExecutor(max_workers=WORKERS) as pool:
            for key, sc, lat in pool.map(_score, list(plan.keys())):
                scores[key].append(sc["overall"])
                dims[key].append(sc)
                latencies[key].append(lat)
                row[f"score_{key}"] = sc["overall"]
        per_image_rows.append(row)
        b0 = scores[f"grid|{BASELINE_DET}|nonote"][-1]
        print(f"[{step + 1}/{len(sample)}] {CLASS_NAMES[true_cls]:10s} B0={b0}  "
              + " ".join(f"{d[:4]}:{np.mean([scores[f'grid|{d}|{n}'][-1] for n in NOTE_LEVELS]):.1f}"
                         for d in DETECTORS))

    for N in EVAL_NS:
        aggregate_and_log(N, conditions, scores, dims, latencies, per_image_rows, bool(semikong),
                          e2e_det, e2e_retr)


def aggregate_and_log(N, conditions, scores, dims, latencies, per_image_rows, has_semikong,
                      e2e_det, e2e_retr):
    sub = {k: scores[k][:N] for k, _ in conditions}
    lab = {k: l for k, l in conditions}
    B0 = f"grid|{BASELINE_DET}|nonote"

    # ---- detector x note grid table + marginals ----
    grid_rows = []
    for d in DETECTORS:
        for n in NOTE_LEVELS:
            k = f"grid|{d}|{n}"
            grid_rows.append({"detector": d, "note": n,
                              "mean_score": float(np.mean(sub[k])), "std_score": float(np.std(sub[k]))})
    marg_det = {d: float(np.mean([np.mean(sub[f"grid|{d}|{n}"]) for n in NOTE_LEVELS])) for d in DETECTORS}
    marg_note = {n: float(np.mean([np.mean(sub[f"grid|{d}|{n}"]) for d in DETECTORS])) for n in NOTE_LEVELS}

    def cat(keys):
        out = []
        for k in keys:
            out += sub[k]
        return out

    # detector pairwise (concatenate over note), note effect (concatenate over detector)
    p_clip_vs_trained = wilcoxon_safe(cat([f"grid|CLIP|{n}" for n in NOTE_LEVELS]),
                                      cat([f"grid|ViT_trained|{n}" for n in NOTE_LEVELS]))
    p_untr_vs_trained = wilcoxon_safe(cat([f"grid|ViT_untrained|{n}" for n in NOTE_LEVELS]),
                                      cat([f"grid|ViT_trained|{n}" for n in NOTE_LEVELS]))
    p_clip_vs_untr = wilcoxon_safe(cat([f"grid|CLIP|{n}" for n in NOTE_LEVELS]),
                                   cat([f"grid|ViT_untrained|{n}" for n in NOTE_LEVELS]))
    p_note = wilcoxon_safe(cat([f"grid|{d}|note" for d in DETECTORS]),
                           cat([f"grid|{d}|nonote" for d in DETECTORS]))

    # ---- knockout / llm arms vs B0 (with diagnose-call latency) ----
    arms = [("knock|sparse", lab["knock|sparse"]),
            ("knock|noretr", lab["knock|noretr"]),
            ("knock|oracle", lab["knock|oracle"]),
            ("knock|template", lab["knock|template"]),
            ("knock|gpt_trunc", lab["knock|gpt_trunc"])]
    if has_semikong:
        arms.append(("llm|semikong", lab["llm|semikong"]))

    def lat_ms(k):
        return float(np.mean(latencies[k][:N])) if latencies[k][:N] else float("nan")

    arm_rows = [{"condition": B0, "label": "B0 (ViT_trained · nonote · hybrid · GPT)",
                 "mean_score": float(np.mean(sub[B0])), "std_score": float(np.std(sub[B0])),
                 "mean_diag_latency_ms": lat_ms(B0), "p_value_vs_B0": float("nan")}]
    for k, l in arms:
        arm_rows.append({"condition": k, "label": l,
                         "mean_score": float(np.mean(sub[k])), "std_score": float(np.std(sub[k])),
                         "mean_diag_latency_ms": lat_ms(k),
                         "p_value_vs_B0": wilcoxon_safe(sub[B0], sub[k])})

    # fair LLM compare (both on truncated docs) + truncation effect on GPT
    p_gpt_trunc_vs_semikong = (wilcoxon_safe(sub["knock|gpt_trunc"], sub["llm|semikong"])
                               if has_semikong else float("nan"))
    p_full_vs_trunc = wilcoxon_safe(sub[B0], sub["knock|gpt_trunc"])

    # ---- end-to-end latency for the deployed B0 path (detection + retrieval + diagnosis) ----
    det_ms = float(np.mean(e2e_det[:N]))
    retr_ms = float(np.mean(e2e_retr[:N]))
    diag_ms = lat_ms(B0)
    e2e_ms = det_ms + retr_ms + diag_ms

    # ---- B0 per-dimension breakdown (diagnosis-quality detail) ----
    b0_dims = {d["key"]: float(np.mean([s[d["key"]] for s in dims[B0][:N]])) for d in RUBRIC_DIMENSIONS}

    # ---- persist ----
    suffix = f"_n{N}"
    grid_path = os.path.join(RESULTS_DIR, f"study_grid{suffix}.csv")
    arm_path = os.path.join(RESULTS_DIR, f"study_knockout{suffix}.csv")
    per_image_path = os.path.join(RESULTS_DIR, f"study_per_image{suffix}.csv")
    pd.DataFrame(grid_rows).to_csv(grid_path, index=False)
    pd.DataFrame(arm_rows).to_csv(arm_path, index=False)
    pd.DataFrame(per_image_rows[:N]).to_csv(per_image_path, index=False)

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment("wafer-rca")
    with mlflow.start_run(run_name=f"diagnosis_study_n{N}"):
        mlflow.log_params({"n_images": N, "baseline": "ViT_trained|nonote|hybrid|gpt",
                           "detectors": ",".join(DETECTORS), "notes": ",".join(NOTE_LEVELS),
                           "openai_model": settings.llm_model,
                           "semikong_model": settings.semikong_model if has_semikong else "n/a",
                           "retrieval_top_k": settings.retrieval_top_k})
        for r in grid_rows:
            mlflow.log_metric(f"grid_{r['detector']}_{r['note']}", r["mean_score"])
        for d, v in marg_det.items():
            mlflow.log_metric(f"marg_detector_{d}", v)
        for n, v in marg_note.items():
            mlflow.log_metric(f"marg_note_{n}", v)
        for r in arm_rows:
            cc = r["condition"].replace("|", "_")
            mlflow.log_metric(f"arm_{cc}", r["mean_score"])
            mlflow.log_metric(f"lat_{cc}_ms", r["mean_diag_latency_ms"])
            if not np.isnan(r["p_value_vs_B0"]):
                mlflow.log_metric(f"pvalue_{cc}_vs_B0", r["p_value_vs_B0"])
        for key, v in b0_dims.items():
            mlflow.log_metric(f"B0_dim_{key}", v)
        mlflow.log_metric("pvalue_CLIP_vs_trained", p_clip_vs_trained)
        mlflow.log_metric("pvalue_untrained_vs_trained", p_untr_vs_trained)
        mlflow.log_metric("pvalue_CLIP_vs_untrained", p_clip_vs_untr)
        mlflow.log_metric("pvalue_note_vs_nonote", p_note)
        mlflow.log_metric("pvalue_gpt_trunc_vs_semikong", p_gpt_trunc_vs_semikong)
        mlflow.log_metric("pvalue_gpt_full_vs_trunc", p_full_vs_trunc)
        mlflow.log_metric("e2e_latency_ms", e2e_ms)
        mlflow.log_metric("latency_detection_ms", det_ms)
        mlflow.log_metric("latency_retrieval_ms", retr_ms)
        mlflow.log_metric("latency_diagnosis_ms", diag_ms)
        for p in (grid_path, arm_path, per_image_path):
            mlflow.log_artifact(p)

    # ---- console ----
    print(f"\n================ Diagnosis study (N={N}) — rubric /10 ================")
    print("Detector × note grid (mean):")
    tbl = pd.DataFrame(grid_rows).pivot(index="detector", columns="note", values="mean_score")
    print(tbl.to_string(float_format=lambda v: f"{v:.2f}"))
    print("  marginals  detector: " + "  ".join(f"{d}={v:.2f}" for d, v in marg_det.items()))
    print("             note:     " + "  ".join(f"{n}={v:.2f}" for n, v in marg_note.items()))
    print("Knockout / LLM arms vs B0:")
    for r in arm_rows:
        p = "-" if np.isnan(r["p_value_vs_B0"]) else f"{r['p_value_vs_B0']:.4f}"
        print(f"  {r['label']:44s} {r['mean_score']:5.2f} ± {r['std_score']:.2f}"
              f"  diag={r['mean_diag_latency_ms']:7.0f}ms  p_vs_B0={p}")
    print("Wilcoxon paired tests:")
    print(f"  CLIP vs ViT_trained         p={p_clip_vs_trained:.4f}")
    print(f"  ViT_untrained vs trained    p={p_untr_vs_trained:.4f}")
    print(f"  CLIP vs ViT_untrained       p={p_clip_vs_untr:.4f}")
    print(f"  note vs no-note             p={p_note:.4f}")
    print(f"  GPT(full) vs GPT(truncated) p={p_full_vs_trunc:.4f}   <- truncation effect on GPT")
    print(f"  GPT(trunc) vs SemiKong      p={p_gpt_trunc_vs_semikong:.4f}   <- FAIR same-docs LLM compare")
    print("B0 diagnosis-quality breakdown (per dimension, 1-5): "
          + "  ".join(f"{k}={v:.2f}" for k, v in b0_dims.items()))
    print(f"End-to-end B0 latency/case: {e2e_ms:.0f} ms "
          f"(detection {det_ms:.0f} + retrieval {retr_ms:.0f} + diagnosis {diag_ms:.0f})")
    print(f"Saved -> {grid_path}, {arm_path}  (MLflow run diagnosis_study_n{N})")


if __name__ == "__main__":
    main()
