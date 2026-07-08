"""Hybrid BM25 + dense retrieval with Reciprocal Rank Fusion."""
import os
import pickle
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import chromadb
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

from config import settings

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
COLLECTION_NAME = "wafer_knowledge"
BM25_INDEX_PATH = "knowledge_base/bm25_index.pkl"
BM25_CORPUS_PATH = "knowledge_base/bm25_corpus.pkl"
BM25_SOURCES_PATH = "knowledge_base/bm25_sources.pkl"


class HybridRetriever:
    def __init__(self):
        embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL,
                                           model_kwargs={"device": settings.device})
        client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
        self.vectorstore = Chroma(
            client=client,
            collection_name=COLLECTION_NAME,
            embedding_function=embeddings,
        )
        with open(BM25_INDEX_PATH, "rb") as fh:
            self.bm25_index = pickle.load(fh)
        with open(BM25_CORPUS_PATH, "rb") as fh:
            self.bm25_corpus = pickle.load(fh)
        # Optional aligned source-id list (written by build_index.py) lets us
        # attribute sparse-only hits to their real source document.
        self._text_to_source = {}
        if os.path.exists(BM25_SOURCES_PATH):
            with open(BM25_SOURCES_PATH, "rb") as fh:
                sources = pickle.load(fh)
            self._text_to_source = dict(zip(self.bm25_corpus, sources))

    def retrieve(self, query: str, k: int = None) -> list:
        if k is None:
            k = settings.retrieval_top_k
        rrf_k = settings.rrf_k_constant

        # Dense ranking (top 10).
        dense_docs = self.vectorstore.similarity_search(query, k=10)
        dense_texts = [d.page_content for d in dense_docs]
        dense_meta = {d.page_content: d.metadata for d in dense_docs}

        # Sparse ranking (top 10).
        tokens = query.split()
        sparse_texts = self.bm25_index.get_top_n(tokens, self.bm25_corpus, n=10)

        # Reciprocal Rank Fusion across the two ranked lists.
        scores = {}
        for rank, text in enumerate(dense_texts):
            scores[text] = scores.get(text, 0.0) + 1.0 / (rrf_k + rank)
        for rank, text in enumerate(sparse_texts):
            scores[text] = scores.get(text, 0.0) + 1.0 / (rrf_k + rank)

        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:k]
        results = []
        for text, _ in ranked:
            source = dense_meta.get(text, {}).get("source_id") or \
                self._text_to_source.get(text, "unknown")
            results.append({"text": text, "source": source})
        return results
