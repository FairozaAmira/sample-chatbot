"""Core Retrieval-Augmented Generation pipeline orchestration."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import List

from langchain.chains import RetrievalQA
from langchain.docstore.document import Document
from langchain.prompts import PromptTemplate
from langchain.schema import BaseRetriever

from .config import get_settings
from .costs import estimate_cost
from .identifiers import generate_request_id
from .llm import get_llm
from .schemas import ChatbotResponse, Citation, ChatDomain, CostBreakdown
from .timing import track_duration
from .vectorstore import get_vector_store

LOGGER = logging.getLogger(__name__)

ANSWER_PROMPT = PromptTemplate.from_template(
    (
        "You are an enterprise compliance and procurement assistant. "
        "You must strictly answer using the given context.\n\n"
        "Context:\n{context}\n\n"
        "User question: {question}\n"
        "Guidelines:\n"
        "- Quote relevant facts from the context and cite their source IDs.\n"
        "- If the answer is unavailable, respond with 'I'm unable to locate that information in the current knowledge base.'\n"
        "- Maintain professional, concise tone."
    )
)


@dataclass
class RetrievedContext:
    """Wrapper containing retrieved documents and the generated answer."""

    answer: str
    documents: List[Document]


class DomainAwareRetriever(BaseRetriever):
    """Retriever that filters documents by requested domain before returning results."""

    def __init__(self, base_retriever: BaseRetriever, domain: ChatDomain) -> None:
        self._base_retriever = base_retriever
        self._domain = domain

    def get_relevant_documents(self, query: str) -> List[Document]:  # type: ignore[override]
        candidates = self._base_retriever.get_relevant_documents(query)
        filtered = [doc for doc in candidates if doc.metadata.get("domain") == self._domain.value]
        if filtered:
            return filtered
        return candidates

    async def aget_relevant_documents(self, query: str) -> List[Document]:  # type: ignore[override]
        candidates = await self._base_retriever.aget_relevant_documents(query)
        filtered = [doc for doc in candidates if doc.metadata.get("domain") == self._domain.value]
        if filtered:
            return filtered
        return candidates


class RAGPipeline:
    """High-level interface for answering questions using RAG."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self.llm = get_llm()

    def _build_chain(self, domain: ChatDomain, top_k: int) -> RetrievalQA:
        vector_store = get_vector_store()
        retriever = DomainAwareRetriever(vector_store.as_retriever(search_kwargs={"k": top_k}), domain)
        return RetrievalQA.from_chain_type(
            llm=self.llm,
            retriever=retriever,
            chain_type="stuff",
            chain_type_kwargs={"prompt": ANSWER_PROMPT},
            return_source_documents=True,
        )

    def answer(self, query: str, domain: ChatDomain, top_k: int) -> ChatbotResponse:
        with track_duration() as elapsed_ms:
            chain = self._build_chain(domain, top_k)
            LOGGER.info("Executing RAG chain for query: %s", query)
            result = chain.invoke({"query": query})
            answer: str = result.get("result", "No answer generated.")
            documents: List[Document] = result.get("source_documents", [])[:top_k]

        citations = [
            Citation(
                source=doc.metadata.get("source", "unknown"),
                snippet=(doc.page_content or "")[:400],
                score=doc.metadata.get("score"),
            )
            for doc in documents
        ]
        cost_breakdown = CostBreakdown()
        api_cost = None
        if self.settings.llm_api_key:
            estimate = estimate_cost(query, answer, rate_per_1k=self.settings.llm_cost_per_1k_tokens)
            cost_breakdown = CostBreakdown(**estimate.__dict__)
            api_cost = estimate.estimated_cost_usd

        response = ChatbotResponse(
            id=generate_request_id(),
            created_at=datetime.now(timezone.utc),
            time_taken_ms=elapsed_ms(),
            api_cost=api_cost,
            cost=cost_breakdown,
            answer=answer.strip(),
            citations=citations,
        )
        return response


__all__ = ["RAGPipeline"]
