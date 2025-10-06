import time
from typing import Dict, List, Optional

import trafilatura
from ddgs import DDGS
from loguru import logger

from ..config.config import settings
from .chunking import chunk_text
from .qdrant_store import QdrantRAG


def ddg_search(query: str, max_results: int):
    with DDGS() as ddgs:
        logger.info("DDG search: query='{q}' max_results={n}", q=query, n=max_results)
        return list(
            ddgs.text(
                query,
                max_results=max_results,
                region=settings.ddg_region,
                safesearch=settings.ddg_safesearch,
            )
        )


def ddg_reddit_search(query: str, max_results: int):
    logger.info("DDG reddit search: query='{q}' max_results={n}", q=query, n=max_results)
    return ddg_search(f"site:old.reddit.com {query}", max_results=max_results)


def fetch_text(url: str) -> Optional[str]:
    logger.debug("Fetching URL: {url}", url=url)
    html = trafilatura.fetch_url(url, no_ssl=True)
    if not html:
        logger.warning("Failed to fetch URL or empty HTML: {url}", url=url)
        return None
    txt = trafilatura.extract(html, include_comments=False, include_links=False)
    if not txt:
        logger.debug("No extractable text from URL: {url}", url=url)
    return txt


def ingest_urls(urls: List[str], rag: QdrantRAG) -> List[Dict]:
    report = []
    for url in urls[:10]:
        try:
            txt = fetch_text(url)
            if not txt:
                continue
            chunks = chunk_text(txt, max_chars=settings.chunk_max_chars)
            if not chunks:
                continue
            logger.info("Ingesting {n} chunks from {url} into RAG", n=len(chunks), url=url)
            rag.upsert(chunks, [{"url": url, "title": url} for _ in chunks])
            report.append({"url": url, "chunk_count": len(chunks)})
            time.sleep(settings.polite_delay_sec)
        except Exception:
            logger.exception("Error ingesting URL: {url}", url=url)
            continue
    logger.info(
        "Ingest complete: processed {n} urls, ingested {m} items", n=len(urls[:10]), m=len(report)
    )
    return report
