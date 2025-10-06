from __future__ import annotations

import itertools
import json
from typing import Any, Dict, List, Optional, TypedDict

from langgraph.graph import END, START, StateGraph
from loguru import logger
from pydantic_ai import Agent

from ..config.config import settings
from ..rag.qdrant_store import QdrantRAG
from ..tools.web_tools import make_tools


class DebateState(TypedDict):
    topic: str
    rounds: int
    log: List[Dict[str, Any]]
    summary: Optional[str]


def _truncate(s: str, n: int = 1200) -> str:
    return s if len(s) <= n else s[: n - 3] + "..."


def _bullets(hits: List[Dict], limit: int = 8) -> str:
    lines = []
    for h in hits[:limit]:
        t = h.get("title") or h.get("url")
        u = h.get("url")
        sn = (h.get("snippet") or "").strip()
        if not u:
            continue
        lines.append(f"- **{t}** — {u}\n  - {sn[:220]}{'...' if len(sn) > 220 else ''}")
    return "\n".join(lines) or "NONE"


def build_graph(researcher: Agent, critic: Agent, judge: Agent, rag: QdrantRAG):
    """Simple Researcher → Critic → Judge pipeline."""
    graph = StateGraph(DebateState)

    bundle = make_tools(rag)
    tools = bundle["tools"]
    fns = bundle["fns"]
    retrieve, search, ingest = fns["retrieve"], fns["search"], fns["ingest"]

    # Attach tools to researcher & critic agents
    for agent in (researcher, critic):
        try:
            agent.tools = tools
        except Exception:
            pass

    # RESEARCHER AGENT
    async def researcher_node(state: DebateState) -> Dict[str, Any]:
        q = state["topic"]
        logger.info("Researcher: topic='{}'", q)

        rag_hits = retrieve(q, settings.top_k) or []
        if len(rag_hits) < max(1, settings.top_k // 2):
            logger.info("Researcher: low RAG coverage; ingesting more")
            ingest(q, source="both", max_results=settings.max_search_results)
            rag_hits = retrieve(q, settings.top_k) or []

        web_hits = search(q, source="web", max_results=settings.max_search_results)
        red_hits = search(q, source="reddit", max_results=settings.max_reddit_results)

        prompt = (
            f"# Deep Research Task\n"
            f"**Topic:** {q}\n\n"
            "Produce a detailed, source-grounded report for any topic.\n\n"
            "### Output format (strict markdown):\n"
            "## Detailed Summary\n"
            "- 10–15 bullets of main insights with [n] markers.\n\n"
            "## What I Found on the Internet\n"
            "- 5 bullets summarizing factual findings from web sources with [n].\n\n"
            "## What I Found on Reddit\n"
            "- 5 bullets summarizing user experiences or opinions from Reddit with [n].\n\n"
            "## References\n"
            "- Numbered list of URLs matching [n] markers.\n\n"
            "Prefer authoritative and recent info; note disagreements or gaps explicitly.\n\n"
            "## RAG Passages\n"
            f"{_truncate(json.dumps(rag_hits, ensure_ascii=False), 2000)}\n\n"
            "## Web Hits\n"
            f"{_bullets(web_hits)}\n\n"
            "## Reddit Hits\n"
            f"{_bullets(red_hits)}\n"
        )

        res = await researcher.run(prompt)
        out = getattr(res, "output", "") or getattr(res, "content", "") or str(res)

        state["log"].append(
            {
                "role": "researcher",
                "content": out,
                "web_hits": web_hits,
                "reddit_hits": red_hits,
            }
        )
        return state

    # Critic
    async def critic_node(state: DebateState) -> Dict[str, Any]:
        q = state["topic"]
        last = next((m for m in reversed(state["log"]) if m["role"] == "researcher"), {})
        researcher_text = last.get("content", "") or q

        probe = researcher_text[:800]
        web_hits = search(probe, source="web", max_results=8)
        red_hits = search(probe, source="reddit", max_results=8)

        prompt = (
            "You are the Critic. Evaluate completeness, evidence strength, and missing angles.\n\n"
            "### Output format (strict markdown):\n"
            "## Key Weaknesses\n"
            "- Bullet list of flaws or gaps with [n].\n\n"
            "## Contradictions & Edge Cases\n"
            "- Note any disagreements or special cases with [n].\n\n"
            "## What To Verify Next\n"
            "- Suggested follow-up queries or missing aspects.\n\n"
            "## References\n"
            "- Numbered list of URLs cited.\n\n"
            f"## Researcher Report\n{_truncate(researcher_text, 4000)}\n\n"
            "## Extra Web Evidence\n"
            f"{_bullets(web_hits)}\n\n"
            "## Extra Reddit Evidence\n"
            f"{_bullets(red_hits)}\n"
        )

        res = await critic.run(prompt)
        out = getattr(res, "output", "") or getattr(res, "content", "") or str(res)

        state["log"].append(
            {
                "role": "critic",
                "content": out,
                "web_hits": web_hits,
                "reddit_hits": red_hits,
            }
        )
        return state

    # Judge
    async def judge_node(state: DebateState) -> Dict[str, Any]:
        res_entries = [m for m in state["log"] if m["role"] == "researcher"]
        crit_entries = [m for m in state["log"] if m["role"] == "critic"]

        researcher_text = "\n\n".join(e.get("content", "") for e in res_entries)
        critic_text = "\n\n".join(e.get("content", "") for e in crit_entries)

        urls: List[str] = []

        def _collect(hits):
            for h in hits or []:
                u = h.get("url")
                if u:
                    urls.append(u)

        for e in itertools.chain(res_entries, crit_entries):
            _collect(e.get("web_hits"))
            _collect(e.get("reddit_hits"))
        refs = list(dict.fromkeys(urls))

        prompt = (
            "You are the Judge. Combine findings into one coherent, well-structured brief.\n\n"
            "### Output format (strict markdown):\n"
            "## Detailed Summary\n"
            "- 6–10 bullets integrating Researcher + Critic insights.\n\n"
            "## What i Found on the Internet\n"
            "- Give factual bullets from web sources with [n].\n\n"
            "## What i Found on Reddit\n"
            "- Give detailed bullets from Reddit (user opinions, experiences) with [n].\n\n"
            "## References\n"
            "- Numbered list of URLs only.\n\n"
            f"## Researcher\n{_truncate(researcher_text, 5000)}\n\n"
            f"## Critic\n{_truncate(critic_text, 4000)}\n\n"
            f"## Known URLs\n{json.dumps(refs[:40], indent=2)}\n"
        )

        res = await judge.run(prompt)
        out = getattr(res, "output", "") or getattr(res, "content", "") or str(res)

        # Simple confidence heuristic
        total_hits = sum(
            len(e.get("web_hits") or []) + len(e.get("reddit_hits") or [])
            for e in itertools.chain(res_entries, crit_entries)
        )
        confidence = "High" if total_hits >= 10 else ("Medium" if total_hits >= 4 else "Low")

        state["summary"] = out
        state["log"].append(
            {"role": "judge", "content": out, "references": refs, "confidence": confidence}
        )
        return state

    # Build a graph
    graph.add_node("researcher", researcher_node)
    graph.add_node("critic", critic_node)
    graph.add_node("judge", judge_node)
    graph.add_edge(START, "researcher")
    graph.add_edge("researcher", "critic")
    graph.add_edge("critic", "judge")
    graph.add_edge("judge", END)
    return graph.compile()
