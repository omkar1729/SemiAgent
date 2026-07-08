"""Multi-judge robustness / bias check for the LLM-as-judge rubric.

Regenerates a fixed set of diagnosis reports from two writers (GPT-4o-mini and
SemiKong-8B) on the same wafers, then scores every report with THREE judges using the
identical rubric prompt: GPT-4o-mini (the original judge), Claude, and Gemini.

Outputs, per (writer x judge): mean rubric score /10; per-judge leniency; cross-judge
agreement (Spearman); and whether each judge independently ranks GPT > SemiKong
(the bias question — if only the GPT judge does, that's self-enhancement bias).

Keys are read from env only (ANTHROPIC_API_KEY, GEMINI_API_KEY) and never written out.
"""
import json
import os
import re
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.seed import set_all_seeds
from config import settings

set_all_seeds(settings.random_seed)

import mlflow
import numpy as np
import openai
import pandas as pd
from scipy.stats import spearmanr, wilcoxon

from agents.detection_agent import CLASS_NAMES, DetectionAgent
from agents.retrieval_agent import RetrievalAgent
from agents.diagnosis_agent import DiagnosisAgent
from evaluation.rubric import DiagnosticRubric, RUBRIC_DIMENSIONS

RESULTS_DIR = "evaluation/results"
N = int(os.environ.get("JUDGE_N", "20"))
CLAUDE_MODEL = os.environ.get("CLAUDE_MODEL", "claude-sonnet-4-5-20250929")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
ANTH = os.environ["ANTHROPIC_API_KEY"]
GEM = os.environ["GEMINI_API_KEY"]
DOC_CHARS = int(os.environ.get("SEMIKONG_DOC_CHARS", "500"))

rubric = DiagnosticRubric()  # GPT-4o-mini judge + shared prompt builder


def truncate(retr, cap=DOC_CHARS):
    return {"retrieved_docs": [{**d, "text": d.get("text", "")[:cap]} for d in retr["retrieved_docs"]],
            "query_used": retr["query_used"]}


def docs_block(docs):
    return "\n".join((d.get("text", "")[:300]) for d in docs) or "[none]"


def parse_scores(text):
    """Tolerant: a missing/garbled dimension (small models like SemiKong sometimes
    omit one) defaults to a neutral 3 instead of crashing the whole run."""
    m = re.search(r"\{.*\}", text, re.DOTALL)
    try:
        data = json.loads(m.group()) if m else {}
    except Exception:
        data = {}
    sc = {}
    for d in RUBRIC_DIMENSIONS:
        try:
            v = int(data.get(d["key"]))
        except (TypeError, ValueError):
            v = 3
        sc[d["key"]] = min(5, max(1, v))
    sc["overall"] = round(sum(sc[d["key"]] for d in RUBRIC_DIMENSIONS) / len(RUBRIC_DIMENSIONS) * 2.0, 2)
    return sc


def build_prompt(report, docs):
    """The identical rubric prompt every judge sees."""
    fields = ["defect_summary", "process_step_analysis", "root_cause_hypotheses", "investigation_parameters"]
    return rubric._prompt({k: report.get(k) for k in fields}, docs_block(docs))


# SemiKong-as-judge uses the same OpenAI-compatible server as the SemiKong writer.
_semikong_client = openai.OpenAI(base_url=settings.semikong_base_url,
                                 api_key=settings.semikong_api_key or "not-needed",
                                 max_retries=3, timeout=90.0)


def gpt_judge(report, docs):
    return rubric.auto_score(report, docs)  # GPT-4o-mini, batched rubric prompt


def _post(url, body, headers):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers=headers)
    with urllib.request.urlopen(req, timeout=90) as r:
        return json.load(r)


def claude_judge(report, docs):
    prompt = build_prompt(report, docs)
    for _ in range(4):
        try:
            d = _post("https://api.anthropic.com/v1/messages",
                      {"model": CLAUDE_MODEL, "max_tokens": 400, "temperature": 0,
                       "messages": [{"role": "user", "content": prompt}]},
                      {"x-api-key": ANTH, "anthropic-version": "2023-06-01",
                       "content-type": "application/json"})
            return parse_scores(d["content"][0]["text"])
        except Exception as e:
            last = e
            time.sleep(6)
    raise RuntimeError(f"claude judge failed: {last}")


def gemini_judge(report, docs):
    prompt = build_prompt(report, docs)
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={GEM}"
    for _ in range(4):
        try:
            d = _post(url,
                      {"contents": [{"parts": [{"text": prompt}]}],
                       "generationConfig": {"temperature": 0, "responseMimeType": "application/json",
                                            "maxOutputTokens": 2048, "thinkingConfig": {"thinkingBudget": 0}}},
                      {"content-type": "application/json"})
            return parse_scores(d["candidates"][0]["content"]["parts"][0]["text"])
        except Exception as e:
            last = e
            time.sleep(6)
    raise RuntimeError(f"gemini judge failed: {last}")


def semikong_judge(report, docs):
    prompt = build_prompt(report, docs)
    for _ in range(4):
        try:
            resp = _semikong_client.chat.completions.create(
                model=settings.semikong_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0, max_tokens=400)
            return parse_scores(resp.choices[0].message.content or "")
        except Exception as e:
            last = e
            time.sleep(5)
    raise RuntimeError(f"semikong judge failed: {last}")


JUDGES = {"gpt-4o-mini": gpt_judge, "claude": claude_judge,
          "gemini": gemini_judge, "semikong": semikong_judge}
WRITERS = ["gpt", "semikong"]


def main():
    data = np.load(os.path.join(settings.mixedwm38_data_dir, "data.npy"))
    labels = np.load(os.path.join(settings.mixedwm38_data_dir, "labels.npy"))
    idx_path = os.path.join(RESULTS_DIR, "study_50_indices.npy")
    sample = [int(x) for x in np.load(idx_path)][:N]
    primary = labels.argmax(axis=1)
    print(f"Multi-judge: {len(sample)} wafers x {len(WRITERS)} writers x {len(JUDGES)} judges "
          f"(claude={CLAUDE_MODEL}, gemini={GEMINI_MODEL})")

    detector = DetectionAgent(mode="vit", load_vit_checkpoint=True)
    retrieval = RetrievalAgent()
    writers = {"gpt": DiagnosisAgent(provider="openai"),
               "semikong": DiagnosisAgent(provider="semikong")}

    rows = []  # one row per (wafer, writer, judge)
    for step, i in enumerate(sample):
        det = detector.detect(data[i])
        retr = retrieval.retrieve(det, None)
        reports = {
            "gpt": (writers["gpt"].diagnose(det, retr, None), retr["retrieved_docs"]),
            "semikong": (writers["semikong"].diagnose(det, truncate(retr), None),
                         truncate(retr)["retrieved_docs"]),
        }
        for w in WRITERS:
            report, docs = reports[w]

            def _judge(kv):
                name, fn = kv
                try:
                    return name, fn(report, docs)
                except Exception as e:
                    print(f"  judge {name} failed on idx={i}/{w}: {e}")
                    return name, None

            with ThreadPoolExecutor(max_workers=len(JUDGES)) as pool:
                results = dict(pool.map(_judge, JUDGES.items()))
            for jname, sc in results.items():
                if sc is None:
                    continue
                rows.append({"image_index": i, "true_defect": CLASS_NAMES[int(primary[i])],
                             "writer": w, "judge": jname, "overall": sc["overall"],
                             **{d["key"]: sc[d["key"]] for d in RUBRIC_DIMENSIONS}})
        print(f"[{step + 1}/{len(sample)}] idx={i} "
              + " | ".join(f"{w}:" + ",".join(f"{j[:3]}={next(r['overall'] for r in rows if r['image_index']==i and r['writer']==w and r['judge']==j)}"
                                              for j in JUDGES) for w in WRITERS))

    df = pd.DataFrame(rows)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out = os.path.join(RESULTS_DIR, f"multi_judge_n{len(sample)}.csv")
    df.to_csv(out, index=False)
    aggregate(df, len(sample), out)


def aggregate(df, n, out):
    judges = list(JUDGES)
    print(f"\n================ Multi-judge results (N={n}) — rubric /10 ================")

    # writer x judge mean-overall table
    piv = df.pivot_table(index="writer", columns="judge", values="overall", aggfunc="mean")
    print("\nMean overall by writer x judge:")
    print(piv.to_string(float_format=lambda v: f"{v:.2f}"))

    print("\nJudge leniency (mean over all reports) and GPT-vs-SemiKong ranking per judge:")
    bias = {}
    for j in judges:
        gj = df[(df.judge == j) & (df.writer == "gpt")].set_index("image_index")["overall"]
        sj = df[(df.judge == j) & (df.writer == "semikong")].set_index("image_index")["overall"]
        paired = pd.concat([gj, sj], axis=1, keys=["g", "s"]).dropna()
        g, s = paired["g"].values, paired["s"].values
        try:
            p = float(wilcoxon(g, s).pvalue)
        except Exception:
            p = float("nan")
        delta = float(np.mean(g) - np.mean(s)) if len(g) else float("nan")
        bias[j] = delta
        leniency = float(df[df.judge == j]["overall"].mean())
        print(f"  {j:12s} leniency={leniency:5.2f}  GPT={np.mean(g):5.2f} SemiKong={np.mean(s):5.2f}  "
              f"GPT-SemiKong delta={delta:+.2f}  p={p:.4f}")

    # cross-judge agreement on the SAME reports (Spearman over all writer reports)
    print("\nCross-judge agreement (Spearman rho of per-report overall scores):")
    wide = df.pivot_table(index=["image_index", "writer"], columns="judge", values="overall")
    for a in range(len(judges)):
        for b in range(a + 1, len(judges)):
            ja, jb = judges[a], judges[b]
            sub = wide[[ja, jb]].dropna()
            rho = spearmanr(sub[ja], sub[jb]).correlation if len(sub) > 2 else float("nan")
            print(f"  {ja:12s} vs {jb:12s}  rho={rho:+.3f}  mean|diff|={np.mean(np.abs(sub[ja]-sub[jb])):.2f}")

    print("\nBias check: GPT-judge's GPT-favoring delta vs the other judges' deltas")
    print(f"  gpt-4o-mini delta={bias.get('gpt-4o-mini', float('nan')):+.2f}  "
          + "  ".join(f"{j} delta={bias[j]:+.2f}" for j in judges if j != "gpt-4o-mini"))

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment("wafer-rca")
    with mlflow.start_run(run_name=f"multi_judge_n{n}"):
        mlflow.log_params({"n_wafers": n, "writers": ",".join(WRITERS), "judges": ",".join(judges),
                           "claude_model": CLAUDE_MODEL, "gemini_model": GEMINI_MODEL})
        for w in WRITERS:
            for j in judges:
                mlflow.log_metric(f"mean_{w}_by_{j}", float(piv.loc[w, j]))
        for j, dlt in bias.items():
            mlflow.log_metric(f"gpt_minus_semikong_{j}", dlt)
        mlflow.log_artifact(out)
    print(f"\nSaved -> {out}  (MLflow run multi_judge_n{n})")


if __name__ == "__main__":
    main()
