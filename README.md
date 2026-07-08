# SemiAgent — Wafer Defect Root Cause Analysis

A three-agent system that turns a MixedWM38 wafer bin map (plus an optional
engineer note) into a structured root cause analysis report.

```
 wafer map (.npy) ─► Detection Agent ─► Retrieval Agent ─► Diagnosis Agent ─► RCA report
        + note          (CLIP / ViT)     (BM25 + dense)     (GPT-4o-mini CoT)
                         └──────────── orchestrated by LangGraph ───────────┘
```

1. **Detection Agent** — CLIP zero-shot (default, no training) or a fine-tuned
   ViT (`USE_TRAINED_VIT=true`). Classifies the map over 38 MixedWM38 classes.
2. **Retrieval Agent** — hybrid BM25 + dense (Chroma) retrieval over a corpus of
   arXiv abstracts, patents, and curated seed abstracts, fused with RRF.
3. **Diagnosis Agent** — GPT-4o-mini with chain-of-thought prompting, grounded in
   the retrieved docs and the optional engineer note, returns a structured JSON report.

---

## 1. Setup

```bash
python -m venv venv
venv\Scripts\activate            # Windows (use source venv/bin/activate on *nix)
pip install -r requirements.txt
cp .env.example .env             # then fill in keys (see below)
```

### API keys (put them in `.env`)

| Key | Needed for | How to get it |
|-----|------------|---------------|
| `OPENAI_API_KEY` | Diagnosis + rubric eval | platform.openai.com → **API keys** → *Create new secret key*. Requires billing/credit. |
| `SERPAPI_API_KEY` | Real patent abstracts (preferred) | serpapi.com → register → copy API key from the dashboard. |
| `USPTO_API_KEY` | Patent fallback (titles) | data.uspto.gov → create a free account → verify with ID.me → copy the key from the **MyODP** dashboard. |
| `GCP_PROJECT` | Patents via BigQuery (alt) | A GCP project id; also `pip install google-cloud-bigquery db-dtypes` and run `gcloud auth application-default login`. |

**Kaggle** (only if you don't already have the dataset): kaggle.com → *Settings* →
*API* → **Create New Token** → move the downloaded `kaggle.json` to `~/.kaggle/`.
The dataset (`Wafer_Map_Datasets.npz`) is already present under `data/processed/`.

---

## 2. Run order

```bash
python data/download_mixedwm38.py      # -> data/processed/{data,labels,*_indices}.npy
python data/build_corpus.py            # -> data/corpus.jsonl  (arXiv + patents + seeds)
python knowledge_base/build_index.py   # -> Chroma vector store + BM25 pickles

# optional: train the ViT detector (only if USE_TRAINED_VIT=true)
python models/train_vit.py
python models/evaluate_vit.py

# run the pipeline on one wafer
python pipeline/run_pipeline.py --image_path data/processed/sample.npy
python pipeline/run_pipeline.py --image_path data/processed/sample.npy \
    --engineer_note "Edge-ring pattern observed after CMP step, unusual slurry flow noted"

# or serve the API
uvicorn api.main:app --reload

# or launch the Streamlit UI (upload a wafer map + note -> RCA report)
streamlit run ui/app.py
```

API endpoints: `POST /diagnose` (multipart `file` + optional `engineer_note`),
`GET /health`, `GET /metrics`. Corpus size is controlled by `CORPUS_MAX_DOCS`
(default 1000).

---

## 3. Two detection modes

* **CLIP zero-shot (default)** — no setup beyond `pip install`. Encodes the 38
  class descriptions once and matches each wafer map against them.
* **ViT fine-tuned** — set `USE_TRAINED_VIT=true` in `.env`, then run
  `models/train_vit.py` to produce `models/checkpoints/vit_best.pt`. The pipeline
  loads this checkpoint automatically (and errors clearly if it is missing).

**GPU training.** `device` auto-detects CUDA. `train_vit.py` uses mixed precision
and is tuned via `VIT_BATCH_SIZE` / `VIT_EPOCHS` (and `VIT_MAX_TRAIN_SAMPLES` for a
quick subset run). Install the CUDA build of torch first (see `requirements.txt`).
On a 6 GB GPU, batch 32 fits (~4 GB) at roughly 10 min/epoch.

---

## 4. Evaluation

Three non-overlapping scripts (the diagnosis study replaces the old
`eval_diagnosis` + `ablation` + `ablation_factorial` — one scoring pass, no
duplicated conditions):

```bash
python evaluation/eval_detection.py    # top-1 + macro-F1 + per-class: CLIP / ViT-untrained / ViT-trained (500 maps)
python evaluation/eval_retrieval.py    # P@5 (hybrid/sparse/dense) + cosine-relevance (8 seed queries)
EVAL_NS=10,30,50 python evaluation/diagnosis_study.py   # rubric /10 (needs OPENAI)

# optional: multi-judge bias cross-check (GPT-4o-mini vs Claude/Gemini/SemiKong judges)
JUDGE_N=20 ANTHROPIC_API_KEY=... GEMINI_API_KEY=... \
  SEMIKONG_BASE_URL=... SEMIKONG_MODEL=semikong-8b \
  python evaluation/multi_judge.py
```

All runs log params, metrics, and result CSVs to the local MLflow file store
(`mlflow_runs/`, experiment `wafer-rca`).

**The diagnosis study** scores one fixed, nested, stratified sample (sizes from
`EVAL_NS`, default `10,30,50`; each size is a prefix of the next) ONCE and aggregates
over the prefixes. Everything shares a single baseline so nothing is computed twice:

> **Baseline B0** = ViT-trained detector · no engineer note · hybrid retrieval · GPT-4o-mini

* **Detector×note grid:** {CLIP, ViT-untrained, ViT-trained} × {no-note, note}
* **Component knockout** (one change from B0): sparse-only retrieval · no retrieval ·
  oracle (ground-truth) detection · **template diagnosis** (no-LLM baseline) ·
  **GPT on truncated docs** (fair, same-input comparison vs SemiKong)
* **LLM arm:** B0 with SemiKong-8B (skipped automatically if unreachable)
* **Latency:** per-stage and end-to-end (detection + retrieval + diagnosis) for B0

Each arm is compared to B0 with a Wilcoxon signed-rank test; the grid also reports
per-cell means, marginals, and detector/note pairwise tests. The rubric scores five
frozen dimensions 1–5 in **one batched call** (overall = mean × 2, on a 0–10 scale).
Tune concurrency with `STUDY_WORKERS` (default **2** — keeps GPT-4o-mini calls under
the 200K TPM cap; higher values thrash on the rate limit and run *slower*).

### Headline results (corrected labels, N=50)

| Experiment | Result |
|---|---|
| Detection (500 maps) | top-1 / macro-F1 — CLIP 5.0% / 0.008 · ViT-untrained 4.4% / 0.008 · **ViT-trained 98.4% / 0.982** |
| Retrieval | P@5: sparse **0.525** · dense 0.475 · hybrid 0.450 · cosine-relevance (hybrid top-5) **0.671** |
| Diagnosis B0 | **9.60/10** (causal 5.0, evidence 5.0, technical 5.0, actionable 5.0, completeness 4.0) |
| Knockout vs B0 | **template-diagnosis (no LLM) 7.52** (p<0.001) · no-retrieval 7.84 (p<0.001) · sparse-only 9.22 (p=0.007) · oracle-det 9.57 (p=0.59) |
| LLM (fair, same docs) | GPT-trunc 9.51 vs **SemiKong 9.33, p=0.008** (GPT robust to truncation: full 9.60 vs trunc 9.51, p=0.16) |
| End-to-end latency | **4.56 s/case** (detection 88 ms + retrieval 18 ms + diagnosis 4451 ms) |
| Judge robustness (N=20, 4 judges) | all judges rank GPT>SemiKong — GPT-4o-mini +0.40 (p=0.001), Claude +0.60, Gemini +0.86, SemiKong +0.44; GPT-judge is the **least** GPT-favoring |

Takeaway: the report quality rests on **two** load-bearing components — the **LLM's
reasoning** (template knockout is the largest drop) and **retrieval** (close second);
detector accuracy and the engineer note barely move the score; **GPT-4o-mini beats
SemiKong-8B even on identical truncated docs**, and this ranking holds across four
independent judges (GPT/Claude/Gemini/SemiKong) — not a self-grading artifact. See
`REPORT.txt` for the full writeup.

### Using SemiKong (open-source semiconductor LLM) for the LLM arm

[SemiKong](https://github.com/aitomatic/semikong) is a Llama-3.1-based model for
semiconductor manufacturing. The study adds the SemiKong arm when it is reachable and
skips it cleanly otherwise. Two ways to wire it in:

**a) OpenAI-compatible server** (vLLM / LMStudio / Ollama) — what these results used
(vLLM serving `semikong-8b`):

```bash
# .env  (or pass as env vars)
SEMIKONG_BACKEND=semikong
SEMIKONG_BASE_URL=http://<host>:8000/v1
SEMIKONG_MODEL=semikong-8b
```

Note: a quantized SemiKong-8B server often has a small context window (e.g. 3072
tokens), so the study feeds it **truncated doc snippets** (`SEMIKONG_DOC_CHARS`,
default 500) while GPT keeps the full docs — a documented asymmetry.

**b) In-process (transformers + bitsandbytes 4-bit)**:

```bash
# .env
SEMIKONG_BACKEND=semikong_local
SEMIKONG_HF_MODEL=pentagoniac/SEMIKONG-8B-chat   # gated on HF
HF_TOKEN=hf_...                                   # token with access granted
```

At 4-bit it needs ~5.4 GB of weights, so it wants a GPU with ≳8 GB free (a 6 GB card
is too small to hold it alongside other work). `device_map="auto"` spills to CPU if needed.

---

## 5. Notes & deviations from the original brief

The implementation follows the brief but adapts to the actual dataset and the
installed environment. These choices are deliberate:

* **CLIP backend.** The environment ships `open-clip-torch`, not the `openai/CLIP`
  package. The agent uses open_clip with the OpenAI `ViT-B-32` weights (same model);
  it still tries `import clip` first and falls back automatically.
* **Dataset format.** The real `Wafer_Map_Datasets.npz` stores maps as `(N,52,52)`
  with values `{0,1,2}` (0 background, 1 good die, 2 defect) and labels as an `(N,8)`
  multi-hot of the *base* defect types — not `(N,38)`. The 8-column → type order is
  the dataset authors' **official mapping** (`Junliangwangdhu/WaferMap`), verified
  against the documented single-type row ranges:
  `[Center, Donut, Edge-Loc, Edge-Ring, Local, Near-Full, Scratch, Random]`
  (`DATA_COLUMN_TYPES` in `agents/detection_agent.py`). Random appears only as a
  single type; Scratch appears in 18 of the 38 patterns. `labels.npy` is the derived
  `(N,38)` one-hot.
* **Patents.** The brief's `POST /patents/search` + `abstractText` schema does not
  match the live USPTO ODP API (its free key exposes metadata only, no abstracts).
  So patents are sourced, in order of preference: **SerpApi Google Patents** (real
  abstracts/snippets) → **BigQuery** Google Patents Public Data → **USPTO ODP**
  titles. Each is skipped cleanly when unconfigured.
* **Corpus size.** `CORPUS_MAX_DOCS` (default 1000) caps the corpus; the default
  build yields ~776 arXiv abstracts + 200 patents + 24 seeds. SerpApi patents use
  paginated search snippets (no per-patent detail calls) to stay within the free quota.
* **GPU.** `device` auto-detects CUDA. The shipped torch was CPU-only; install the
  matching CUDA build (`requirements.txt`) to train the ViT on GPU. Training uses
  AMP mixed precision.
* **Detection accuracy.** CLIP zero-shot on these abstract maps scores ~5% — far
  out-of-distribution for CLIP, which collapses most maps onto one class. The
  fine-tuned ViT reaches **98.4%** top-1 (100% per single-defect class). The diagnosis
  study's **oracle-detection** knockout exists precisely to isolate how little this
  detector gap actually affects the written report.
* **Infra.** MLflow 3.x requires opting back into the file store
  (`MLFLOW_ALLOW_FILE_STORE`, set in `config.py`); `python-multipart` is required for
  the API's form uploads. Both are handled.

---

## 6. Project layout

```
config.py                 pydantic-settings config (loads .env)
utils/seed.py             set_all_seeds()
data/                     download_mixedwm38.py, build_corpus.py
knowledge_base/           build_index.py, retriever.py  (Chroma + BM25)
agents/                   state, detection, retrieval, diagnosis agents
pipeline/                 graph.py (LangGraph), run_pipeline.py (CLI)
models/                   train_vit.py, evaluate_vit.py
evaluation/               eval_detection.py, eval_retrieval.py, diagnosis_study.py, rubric.py
api/main.py               FastAPI service
ui/app.py                 Streamlit UI
notebooks/demo.ipynb      end-to-end walkthrough
```
