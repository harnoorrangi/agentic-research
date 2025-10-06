from __future__ import annotations

from typing import Callable, Dict, List, Literal, Optional, TypedDict

from loguru import logger
from pydantic_ai import Tool

from ..config.config import settings
from ..rag.ingest import ddg_reddit_search, ddg_search, ingest_urls
from ..rag.qdrant_store import QdrantRAG

SourceType = Literal["web", "reddit", "both"]


class ToolBundle(TypedDict):
    tools: List[Tool]  # attach to LLM agents
    fns: Dict[str, Callable]  # plain callables for Python code


def _norm_ddg_items(items: List[dict], source: str) -> List[Dict]:
    """Normalize DDG results to a standard shape."""
    out: List[Dict] = []
    for it in items or []:
        title = it.get("title") or it.get("name") or it.get("href") or it.get("url") or ""
        url = it.get("href") or it.get("url") or ""
        snippet = it.get("body") or it.get("snippet") or ""
        date = it.get("date") or None
        if not url:
            continue

        out.append({"title": title, "url": url, "snippet": snippet, "date": date, "source": source})
    return out


def make_tools(rag: QdrantRAG) -> ToolBundle:
    """Creates both callable functions and Tool-wrapped variants for agents."""

    # -------- plain Python functions --------
    def _retrieve(query: str, k: int = settings.top_k) -> List[Dict]:
        logger.info("ToolFn: retrieve(query='{}', k={})", query, k)
        return rag.query(query, k)

    def _search(
        query: str, source: SourceType = "both", max_results: Optional[int] = None
    ) -> List[Dict]:
        logger.info("ToolFn: search(query='{}', source='{}')", query, source)
        n_web = max_results or settings.max_search_results
        n_red = max_results or settings.max_reddit_results
        web_hits = (
            _norm_ddg_items(ddg_search(query, n_web), "web") if source in ("web", "both") else []
        )
        red_hits = (
            _norm_ddg_items(ddg_reddit_search(query, n_red), "reddit")
            if source in ("reddit", "both")
            else []
        )
        seen, merged = set(), []
        for h in web_hits + red_hits:
            u = h.get("url")
            if not u or u in seen:
                continue
            seen.add(u)
            merged.append(h)
        logger.debug("ToolFn: search returned {} hits", len(merged))
        return merged

    def _ingest(
        query: str, source: SourceType = "both", max_results: Optional[int] = None
    ) -> List[Dict]:
        logger.info("ToolFn: ingest(query='{}', source='{}')", query, source)
        hits = _search(query, source=source, max_results=max_results)
        urls = [h["url"] for h in hits if h.get("url")]
        report = ingest_urls(urls, rag)
        logger.info("ToolFn: ingest processed {} urls", len(report))
        return report

    # -------- Tool wrappers (for LLMs) --------
    @Tool
    def retrieve(query: str, k: int = settings.top_k) -> List[Dict]:
        """Retrieve top-k passages from Qdrant."""
        return _retrieve(query, k)

    @Tool
    def search(
        query: str, source: SourceType = "both", max_results: Optional[int] = None
    ) -> List[Dict]:
        """Search web or reddit ('web'|'reddit'|'both')."""
        return _search(query, source, max_results)

    @Tool
    def ingest(
        query: str, source: SourceType = "both", max_results: Optional[int] = None
    ) -> List[Dict]:
        """Search → fetch → extract → chunk → upsert into RAG."""
        return _ingest(query, source, max_results)

    return {
        "tools": [retrieve, search, ingest],
        "fns": {"retrieve": _retrieve, "search": _search, "ingest": _ingest},
    }
