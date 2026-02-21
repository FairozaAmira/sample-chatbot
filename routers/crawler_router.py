"""API router for data crawling operations."""
from __future__ import annotations

from fastapi import APIRouter

from controllers import handle_crawl_request
from utils.schemas import CrawlRequest, CrawlResponse

router = APIRouter(prefix="/api/crawler", tags=["crawler"])


@router.post("", response_model=CrawlResponse, summary="Ingest data into the knowledge base")
async def crawler_endpoint(payload: CrawlRequest) -> CrawlResponse:
    """Trigger the data ingestion workflow."""

    return handle_crawl_request(payload)
