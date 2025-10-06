# src/agentic_research/app.py
from __future__ import annotations

import asyncio

import typer
from loguru import logger
from rich.console import Console
from rich.panel import Panel

from .agents.factory import make_agents
from .config.config import settings
from .graph.orchestrator import build_graph
from .rag.qdrant_store import QdrantRAG

app = typer.Typer(add_completion=False)
console = Console()


async def _run_async(topic: str, rounds: int) -> dict:
    logger.info("Starting async run topic='{t}' rounds={r}", t=topic, r=rounds)
    # init deps
    rag = QdrantRAG()
    researcher, critic, judge = make_agents(rag)
    graph = build_graph(researcher, critic, judge, rag)

    # initial state
    state = {
        "topic": topic,
        "rounds": max(1, int(rounds)),  # guard
        "log": [],
        "summary": None,
    }

    # run async graph
    return await graph.ainvoke(state)


@app.command()
def run(
    topic: str = typer.Argument(..., help="Debate topic/question."),
    rounds: int = typer.Option(settings.rounds, "--rounds", "-r", help="Number of R↔C rounds."),
    show_log: bool = typer.Option(False, "--show-log", help="Print the full debate log."),
):
    """
    Run the multi-agent debate (Researcher ↔ Critic) with a Judge summary.
    """
    try:
        result = asyncio.run(_run_async(topic, rounds))
    except KeyboardInterrupt:
        logger.warning("Run interrupted by user")
        console.print("[red]Interrupted.[/red]")
        raise typer.Exit(code=130)
    except Exception as e:
        logger.exception("Unhandled exception while running: {err}", err=e)
        console.print_exception()
        raise typer.Exit(code=1)

    # optionally show the transcript
    if show_log:
        for entry in result.get("log", []):
            role = entry.get("role", "unknown").upper()
            console.print(Panel(entry.get("content", ""), title=role))

    # judge summary (or fallback if missing)
    summary = (
        result.get("summary") or (result.get("log") or [{}])[-1].get("content") or ""
    ).strip()
    console.rule("[bold green]JUDGE SUMMARY")
    console.print(summary)
    console.rule("[bold green]JUDGE SUMMARY")


def main():
    app()


if __name__ == "__main__":
    main()
