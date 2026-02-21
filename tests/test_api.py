from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from main import create_app
from utils.config import get_settings


@pytest.fixture(autouse=True)
def reset_settings(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    qa_dir = tmp_path / "qa"
    chroma_dir = tmp_path / "chroma"
    data_dir.mkdir(parents=True, exist_ok=True)
    qa_dir.mkdir(parents=True, exist_ok=True)
    chroma_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setenv("CHATBOT_DATA_DIRECTORY", str(data_dir))
    monkeypatch.setenv("CHATBOT_QA_CACHE_DIRECTORY", str(qa_dir))
    monkeypatch.setenv("CHATBOT_CHROMA_PERSIST_DIRECTORY", str(chroma_dir))
    monkeypatch.setenv("CHATBOT_APP_ENV", "test")

    # Clear cached settings and app to load new environment variables per test.
    get_settings.cache_clear()
    create_app.cache_clear()


@pytest.fixture()
def client() -> TestClient:
    app = create_app()
    return TestClient(app)


def _seed_sample_documents(data_dir: Path) -> None:
    knowledge_dir = data_dir / "knowledge"
    knowledge_dir.mkdir(parents=True, exist_ok=True)
    (knowledge_dir / "policy.txt").write_text(
        "Procurement must follow policy A. Financial approvals require CFO sign-off.",
        encoding="utf-8",
    )

    tender_dir = data_dir / "tender"
    tender_dir.mkdir(parents=True, exist_ok=True)
    (tender_dir / "tender_notes.txt").write_text(
        "Tender drafting should highlight compliance with policy A and include risk mitigation steps.",
        encoding="utf-8",
    )


def test_health_endpoint(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"


def test_crawler_and_chatbot_flow(client: TestClient, tmp_path: Path) -> None:
    settings = get_settings()
    _seed_sample_documents(settings.data_directory)

    crawl_payload = {"refresh_index": True, "max_qas_per_document": 1}
    crawl_response = client.post("/api/crawler", json=crawl_payload)
    assert crawl_response.status_code == 200
    crawl_body = crawl_response.json()
    assert crawl_body["documents_indexed"] >= 2
    assert crawl_body["qa_pairs_generated"] >= 2
    assert "timeTaken" in crawl_body
    assert "apiCost" in crawl_body

    chatbot_payload = {"query": "What does policy A require?", "domain": "knowledge", "top_k": 3}
    chatbot_response = client.post("/api/chatbot", json=chatbot_payload)
    assert chatbot_response.status_code == 200
    chatbot_body = chatbot_response.json()
    assert chatbot_body["answer"]
    assert isinstance(chatbot_body["citations"], list)
    assert "timeTaken" in chatbot_body
    assert "apiCost" in chatbot_body


def test_evaluation_endpoint(client: TestClient) -> None:
    evaluation_payload = {
        "question": "What is policy A?",
        "answer": "Policy A ensures compliance with procurement rules.",
        "references": [{"context": "Policy A covers procurement compliance."}],
    }
    response = client.post("/api/evaluation", json=evaluation_payload)
    assert response.status_code == 200
    body = response.json()
    assert body["average_score"] >= 0
    assert any(metric["metric"] == "ReferenceOverlap" for metric in body["metrics"])
    assert "timeTaken" in body
    assert "apiCost" in body
