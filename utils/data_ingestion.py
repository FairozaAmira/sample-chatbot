"""Data ingestion and QA generation utilities for the RAG system."""
from __future__ import annotations

import json
import logging
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from email import policy
from email.parser import BytesParser
from pathlib import Path
from typing import Iterable, List

import pandas as pd
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pypdf import PdfReader
from bs4 import BeautifulSoup

from .config import Settings, get_settings
from .identifiers import generate_request_id
from .llm import PlaceholderLLM, get_llm
from .schemas import ChatDomain
from .vectorstore import upsert_documents

LOGGER = logging.getLogger(__name__)


@dataclass
class QAPair:
    """Represents a generated question/answer pair."""

    question: str
    answer: str
    source: str
    domain: str


@dataclass
class CrawlSummary:
    """Summary returned after a data ingestion run."""

    documents_indexed: int
    qa_pairs_generated: int
    qa_output_path: Path | None


class DataCrawler:
    """Crawls local data sources, optional sample websites, and generates QA pairs."""

    SUPPORTED_SUFFIXES = {
        ".txt",
        ".md",
        ".csv",
        ".json",
        ".pdf",
        ".eml",
        ".log",
    }

    SAMPLE_WEBSITE_FILES = {
        "knowledge": "policy_portal.html",
        "tender": "tender_hub.html",
        "finance": "finance_updates.html",
    }

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.llm = get_llm()
        self.splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=120)

    def run(self, refresh_index: bool, max_qas_per_document: int) -> CrawlSummary:
        """Execute the end-to-end ingestion workflow."""

        LOGGER.info("Starting crawl: refresh_index=%s, max_qas=%s", refresh_index, max_qas_per_document)
        if refresh_index:
            self._reset_vector_store()

        raw_documents = list(self._load_documents())
        raw_documents.extend(self._load_sample_websites())
        LOGGER.info("Loaded %d raw documents from %s", len(raw_documents), self.settings.data_directory)

        chunked_documents = self._split_documents(raw_documents)
        LOGGER.info("Chunked documents into %d segments", len(chunked_documents))

        qa_pairs = self._generate_qa_pairs(raw_documents, max_qas_per_document)
        qa_documents = self._qa_pairs_to_documents(qa_pairs)

        total_upserted = upsert_documents([*chunked_documents, *qa_documents])
        LOGGER.info("Persisted %d documents into the vector store", total_upserted)

        qa_output_path = self._persist_qa_pairs(qa_pairs) if qa_pairs else None

        return CrawlSummary(
            documents_indexed=len(raw_documents),
            qa_pairs_generated=len(qa_pairs),
            qa_output_path=qa_output_path,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _reset_vector_store(self) -> None:
        """Remove the existing vector store persistence layer."""

        persist_directory = self.settings.chroma_persist_directory
        if persist_directory.exists():
            LOGGER.info("Removing existing Chroma directory at %s", persist_directory)
            shutil.rmtree(persist_directory)
        persist_directory.mkdir(parents=True, exist_ok=True)

    def _load_documents(self) -> Iterable[Document]:
        """Yield documents from the configured data directory."""

        data_dir = self.settings.data_directory
        if not data_dir.exists():
            LOGGER.warning("Data directory %s does not exist; no documents loaded.", data_dir)
            return []

        for path in sorted(data_dir.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in self.SUPPORTED_SUFFIXES:
                continue
            yield from self._load_file(path)

    def _infer_domain(self, path: Path) -> str:
        """Infer the domain from the file path."""

        try:
            relative_parts = path.relative_to(self.settings.data_directory).parts
        except ValueError:
            relative_parts = ()

        if relative_parts:
            candidate = relative_parts[0].lower()
            if candidate in {domain.value for domain in ChatDomain}:
                return candidate
        return ChatDomain.knowledge.value

    def _build_metadata(self, path: Path, file_type: str) -> dict[str, str]:
        domain = self._infer_domain(path)
        return {
            "id": generate_request_id(),
            "source": str(path),
            "file_name": path.name,
            "file_type": file_type,
            "domain": domain,
            "ingested_at": datetime.now(timezone.utc).isoformat(),
        }

    def _load_file(self, path: Path) -> List[Document]:
        suffix = path.suffix.lower()
        metadata = self._build_metadata(path, file_type=suffix.strip("."))

        try:
            if suffix in {".txt", ".md", ".log"}:
                content = path.read_text(encoding="utf-8", errors="ignore")
                return [Document(page_content=content, metadata=metadata)]
            if suffix == ".json":
                data = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
                serialized = json.dumps(data, indent=2, ensure_ascii=False)
                return [Document(page_content=serialized, metadata=metadata)]
            if suffix == ".csv":
                df = pd.read_csv(path)
                table_text = df.to_markdown(index=False)
                return [Document(page_content=table_text, metadata=metadata)]
            if suffix == ".pdf":
                reader = PdfReader(str(path))
                documents: List[Document] = []
                for page_number, page in enumerate(reader.pages, start=1):
                    text = page.extract_text() or ""
                    page_metadata = {**metadata, "page": page_number}
                    documents.append(Document(page_content=text, metadata=page_metadata))
                return documents
            if suffix == ".eml":
                with path.open("rb") as email_file:
                    message = BytesParser(policy=policy.default).parse(email_file)
                body = message.get_body(preferencelist=("plain", "html"))
                content = body.get_content() if body else ""
                headers = "\n".join(f"{k}: {v}" for k, v in message.items())
                full_text = f"{headers}\n\n{content}"
                return [Document(page_content=full_text, metadata=metadata)]
        except Exception as exc:  # pragma: no cover - ingestion failures are logged
            LOGGER.exception("Failed to load file %s: %s", path, exc)
            return []

        LOGGER.debug("Skipping unsupported file: %s", path)
        return []

    def _load_sample_websites(self) -> List[Document]:
        """Fetch sample web pages to demonstrate external data ingestion."""

        documents: List[Document] = []
        sample_directory = self.settings.data_directory / "web_samples"
        sample_directory.mkdir(parents=True, exist_ok=True)

        for domain, file_name in self.SAMPLE_WEBSITE_FILES.items():
            html_path = sample_directory / file_name
            if not html_path.exists():
                LOGGER.debug("Sample website file missing for domain %s: %s", domain, html_path)
                continue

            try:
                html_text = html_path.read_text(encoding="utf-8")
            except OSError as exc:
                LOGGER.warning("Unable to read sample website %s: %s", html_path, exc)
                continue

            soup = BeautifulSoup(html_text, "html.parser")
            text = soup.get_text(separator="\n")
            if not text.strip():
                continue

            metadata = {
                "id": generate_request_id(),
                "source": str(html_path.resolve()),
                "file_name": file_name,
                "file_type": "html",
                "domain": domain,
                "ingested_at": datetime.now(timezone.utc).isoformat(),
                "type": "web_sample",
            }
            documents.append(Document(page_content=text, metadata=metadata))
        return documents

    def _split_documents(self, documents: Iterable[Document]) -> List[Document]:
        chunks: List[Document] = []
        for document in documents:
            splits = self.splitter.split_documents([document])
            for idx, split in enumerate(splits):
                split.metadata.setdefault("chunk_index", idx)
                split.metadata.setdefault("parent_source", document.metadata.get("source"))
                split.metadata.setdefault("domain", document.metadata.get("domain"))
                split.metadata.setdefault("type", "chunk")
            chunks.extend(splits)
        return chunks

    def _generate_qa_pairs(self, documents: Iterable[Document], max_pairs: int) -> List[QAPair]:
        if max_pairs <= 0:
            return []

        qa_pairs: List[QAPair] = []
        for document in documents:
            content = document.page_content.strip()
            if not content:
                continue
            segments = [segment.strip() for segment in content.split("\n\n") if len(segment.strip()) > 40]
            if not segments:
                segments = [content]

            for index, segment in enumerate(segments[:max_pairs]):
                question = self._craft_question(document, index)
                answer = self._craft_answer(segment)
                qa_pairs.append(
                    QAPair(
                        question=question,
                        answer=answer,
                        source=document.metadata.get("source", "unknown"),
                        domain=document.metadata.get("domain", ChatDomain.knowledge.value),
                    )
                )

        return qa_pairs

    def _craft_question(self, document: Document, index: int) -> str:
        domain = document.metadata.get("domain", ChatDomain.knowledge.value)
        source_name = document.metadata.get("file_name", "document")
        return (
            f"What are the key insights from section {index + 1} of {source_name} "
            f"within the {domain} knowledge base?"
        )

    def _craft_answer(self, segment: str) -> str:
        if isinstance(self.llm, PlaceholderLLM):
            return segment[:600]

        try:
            prompt = (
                "You are generating knowledge-base Q&A pairs. Summarize the following segment "
                "into one concise answer limited to 5 sentences. Segment:\n\n{segment}"
            )
            response = self.llm.predict(prompt.format(segment=segment))
            return response.strip()
        except Exception as exc:  # pragma: no cover - LLM failures defer to heuristic answer
            LOGGER.warning("LLM QA generation failed; using heuristic answer: %s", exc)
            return segment[:600]

    def _qa_pairs_to_documents(self, qa_pairs: Iterable[QAPair]) -> List[Document]:
        documents: List[Document] = []
        for pair in qa_pairs:
            metadata = {
                "source": pair.source,
                "domain": pair.domain,
                "type": "generated_qa",
                "qa_id": generate_request_id(),
            }
            content = f"Question: {pair.question}\nAnswer: {pair.answer}"
            documents.append(Document(page_content=content, metadata=metadata))
        return documents

    def _persist_qa_pairs(self, qa_pairs: Iterable[QAPair]) -> Path | None:
        pairs = list(qa_pairs)
        if not pairs:
            return None

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output_path = self.settings.qa_cache_directory / f"qa_pairs_{timestamp}.json"
        payload = [pair.__dict__ for pair in pairs]
        output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        return output_path


__all__ = ["DataCrawler", "QAPair", "CrawlSummary"]
