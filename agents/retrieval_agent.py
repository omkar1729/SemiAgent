"""Knowledge Retrieval Agent — wraps the hybrid retriever and enriches the query
with the detected defect type and any optional engineer note."""
from typing import Optional

from knowledge_base.retriever import HybridRetriever


class RetrievalAgent:
    def __init__(self):
        self.retriever = HybridRetriever()

    def retrieve(self, detection_result: dict, engineer_note: Optional[str] = None) -> dict:
        defect_type = detection_result.get("defect_type", "")
        query = f"{defect_type} semiconductor wafer defect process cause mechanism"
        if engineer_note:  # not None and not empty -> fold into the query
            query = f"{query} {engineer_note}"

        retrieved_docs = self.retriever.retrieve(query)
        return {"retrieved_docs": retrieved_docs, "query_used": query}
