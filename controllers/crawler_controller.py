"""Controller responsible for orchestrating data crawling and indexing."""
from __future__ import annotations

from datetime import datetime, timezone

from utils.data_ingestion import DataCrawler
from utils.identifiers import generate_request_id
from utils.schemas import CostBreakdown, CrawlRequest, CrawlResponse
from utils.timing import track_duration

_crawler = DataCrawler()


def handle_crawl_request(payload: CrawlRequest) -> CrawlResponse:
    """Execute the crawler based on the provided payload."""

    with track_duration() as elapsed_ms:
        summary = _crawler.run(
            refresh_index=payload.refresh_index,
            max_qas_per_document=payload.max_qas_per_document,
        )

    return CrawlResponse(
        id=generate_request_id(),
        created_at=datetime.now(timezone.utc),
        time_taken_ms=elapsed_ms(),
        api_cost=None,
        cost=CostBreakdown(),
        documents_indexed=summary.documents_indexed,
        qa_pairs_generated=summary.qa_pairs_generated,
        qa_output_path=str(summary.qa_output_path) if summary.qa_output_path else None,
    )
