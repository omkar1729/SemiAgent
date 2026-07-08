"""Intrinsic retrieval quality: Precision@5 for three retrieval modes — hybrid
(BM25+dense, RRF), sparse-only (BM25), and dense-only (Chroma) — over annotated
seed queries.

One judgment per single defect type (MixedWM38 has 8 single types, excluding
Normal). Each query's relevant set is that type's three seed documents; P@5 is the
fraction of the top-5 retrieved chunks whose source document is relevant. This
measures retrieval in isolation; the diagnosis study's sparse/no-retrieval knockouts
measure the downstream effect.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.seed import set_all_seeds
from config import settings

set_all_seeds(settings.random_seed)

import mlflow
import numpy as np

from knowledge_base.retriever import EMBED_MODEL, HybridRetriever

JUDGMENTS_PATH = "data/retrieval_judgments.json"

RELEVANCE_JUDGMENTS = [
    {"defect_type": "Center", "query": "Center wafer defect circular cluster at center process root cause",
     "relevant_ids": ["seed_center_001", "seed_center_002", "seed_center_003"]},
    {"defect_type": "Donut", "query": "Donut wafer defect ring shaped pattern around center process cause",
     "relevant_ids": ["seed_donut_001", "seed_donut_002", "seed_donut_003"]},
    {"defect_type": "Edge-Loc", "query": "Edge-Loc wafer defect localized along one edge process cause",
     "relevant_ids": ["seed_edgeloc_001", "seed_edgeloc_002", "seed_edgeloc_003"]},
    {"defect_type": "Edge-Ring", "query": "Edge-Ring wafer defect full ring outer edge process cause",
     "relevant_ids": ["seed_edgering_001", "seed_edgering_002", "seed_edgering_003"]},
    {"defect_type": "Local", "query": "Local wafer defect small localized cluster one region process cause",
     "relevant_ids": ["seed_local_001", "seed_local_002", "seed_local_003"]},
    {"defect_type": "Near-Full", "query": "Near-Full wafer defect covering entire wafer process cause",
     "relevant_ids": ["seed_nearfull_001", "seed_nearfull_002", "seed_nearfull_003"]},
    {"defect_type": "Random", "query": "Random wafer defect scattered particles across wafer process cause",
     "relevant_ids": ["seed_random_001", "seed_random_002", "seed_random_003"]},
    {"defect_type": "Scratch", "query": "Scratch wafer defect linear scratch line process cause",
     "relevant_ids": ["seed_scratch_001", "seed_scratch_002", "seed_scratch_003"]},
]


def _hybrid(retriever, query):
    return [r["source"] for r in retriever.retrieve(query, k=5)]


def _sparse(retriever, query):
    texts = retriever.bm25_index.get_top_n(query.split(), retriever.bm25_corpus, n=5)
    return [retriever._text_to_source.get(t, "unknown") for t in texts]


def _dense(retriever, query):
    docs = retriever.vectorstore.similarity_search(query, k=5)
    return [d.metadata.get("source_id", "unknown") for d in docs]


def _cosine_relevance(retriever, embedder, query):
    """Thesis Table-4 metric: mean cosine similarity of the hybrid top-5 docs to the query."""
    texts = [r["text"] for r in retriever.retrieve(query, k=5)]
    if not texts:
        return 0.0
    q = np.asarray(embedder.embed_query(query), dtype=float)
    D = np.asarray(embedder.embed_documents(texts), dtype=float)
    q /= (np.linalg.norm(q) + 1e-9)
    D /= (np.linalg.norm(D, axis=1, keepdims=True) + 1e-9)
    return float(np.mean(D @ q))


def main():
    os.makedirs(os.path.dirname(JUDGMENTS_PATH), exist_ok=True)
    with open(JUDGMENTS_PATH, "w", encoding="utf-8") as fh:
        json.dump(RELEVANCE_JUDGMENTS, fh, indent=2)

    retriever = HybridRetriever()
    modes = {"hybrid": _hybrid, "sparse": _sparse, "dense": _dense}

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment("wafer-rca")
    means = {}
    with mlflow.start_run(run_name="eval_retrieval"):
        for mode, fn in modes.items():
            precisions = []
            print(f"\n[{mode}] per-query Precision@5:")
            for j in RELEVANCE_JUDGMENTS:
                sources = fn(retriever, j["query"])
                relevant = set(j["relevant_ids"])
                p5 = sum(1 for s in sources if s in relevant) / 5.0
                precisions.append(p5)
                print(f"  {j['defect_type']:12s} P@5 = {p5:.2f}")
            mean_p5 = sum(precisions) / len(precisions)
            means[mode] = mean_p5
            mlflow.log_metric(f"mean_precision_at_5_{mode}", mean_p5)
            print(f"  -> mean P@5 ({mode}) = {mean_p5:.4f}")

        # Retrieval Relevance as cosine similarity (thesis Table-4 metric), hybrid top-5
        from langchain_community.embeddings import HuggingFaceEmbeddings
        embedder = HuggingFaceEmbeddings(model_name=EMBED_MODEL,
                                         model_kwargs={"device": settings.device})
        cos = [_cosine_relevance(retriever, embedder, j["query"]) for j in RELEVANCE_JUDGMENTS]
        mean_cos = float(sum(cos) / len(cos))
        mlflow.log_metric("retrieval_relevance_cosine_hybrid", mean_cos)
        print(f"\nRetrieval Relevance (mean cosine sim of hybrid top-5 to query) = {mean_cos:.4f}")

    print("\nMean Precision@5 by mode: " + "  ".join(f"{m}={v:.3f}" for m, v in means.items()))


if __name__ == "__main__":
    main()
