"""Build the hybrid retrieval index from data/corpus.jsonl.

Produces, in lock-step from a single ordered chunk list so the two indexes stay
aligned (HARD RULE 10):
  * a persistent ChromaDB dense vector store (collection "wafer_knowledge"), and
  * a BM25 sparse index saved as knowledge_base/bm25_index.pkl plus its backing
    text list knowledge_base/bm25_corpus.pkl.

Both BM25 artifacts are always written together (HARD RULE 4).
"""
import json
import os
import pickle
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import chromadb
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rank_bm25 import BM25Okapi

from config import settings

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
COLLECTION_NAME = "wafer_knowledge"
BM25_INDEX_PATH = "knowledge_base/bm25_index.pkl"
BM25_CORPUS_PATH = "knowledge_base/bm25_corpus.pkl"
BM25_SOURCES_PATH = "knowledge_base/bm25_sources.pkl"


def load_corpus(path: str) -> list:
    docs = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                docs.append(json.loads(line))
    return docs


def main():
    corpus = load_corpus(settings.corpus_jsonl_path)
    print(f"Loaded {len(corpus)} documents from {settings.corpus_jsonl_path}")

    # Split each document, carrying source_id + title onto every chunk.
    splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
    documents = [
        Document(page_content=d["text"], metadata={"source_id": d["id"], "title": d.get("title", "")})
        for d in corpus
    ]
    chunks = splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks")

    # Single pass over the ordered chunk list builds the BM25 backing data; the
    # same list is then handed to Chroma, so the two indexes share chunk order.
    bm25_corpus = []
    tokenized_corpus = []
    bm25_sources = []  # source_id aligned to bm25_corpus order (for attribution)
    for chunk in chunks:
        text = chunk.page_content
        bm25_corpus.append(text)
        tokenized_corpus.append(text.split())
        bm25_sources.append(chunk.metadata.get("source_id", ""))

    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL,
                                       model_kwargs={"device": settings.device})
    client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
    # Start clean so re-running doesn't duplicate chunks in the collection.
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    vectorstore = Chroma(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
    )
    vectorstore.add_documents(chunks)

    bm25_index = BM25Okapi(tokenized_corpus)
    os.makedirs(os.path.dirname(BM25_INDEX_PATH), exist_ok=True)
    with open(BM25_INDEX_PATH, "wb") as fh:
        pickle.dump(bm25_index, fh)
    with open(BM25_CORPUS_PATH, "wb") as fh:
        pickle.dump(bm25_corpus, fh)
    with open(BM25_SOURCES_PATH, "wb") as fh:
        pickle.dump(bm25_sources, fh)

    print(f"Indexed {len(chunks)} chunks into ChromaDB ('{COLLECTION_NAME}') and BM25.")


if __name__ == "__main__":
    main()
