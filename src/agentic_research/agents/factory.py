# agents/factory.py (only the prompt pieces changed)

from loguru import logger
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.ollama import OllamaProvider

from ..config.config import settings
from ..tools.web_tools import make_tools


def react_preamble(role: str) -> str:
    return (
        f"You are the {role} in a multi-agent **ReAct** deep-research workflow.\n"
        "You may call tools. Use this loop:\n"
        "Thought → decide what’s missing to answer the question.\n"
        "Action → call the best tool.\n"
        "Observation → read tool output and update your plan.\n"
        "Repeat until you have enough high-quality evidence.\n\n"
        "Tool policy:\n"
        "• Start with `retrieve(query, k)` to reuse local RAG.\n"
        "• If coverage is thin/outdated/biased → `web_search_tool` and/or `reddit_search_tool`.\n"
        "• When you find reusable sources → `web_search_and_ingest` or `reddit_search_and_ingest`, then `retrieve` again.\n"
        "• Reformulate queries (synonyms, entities, time/location filters) when results look off-topic.\n\n"
        "• Create a detailed summary of your findings.\n"
        "Citations & integrity:\n"
        "• Inline cite as [n] that maps to a numbered **References** list with title + URL.\n"
        "• Never fabricate links or statistics; if uncertain, say so and mark as a gap.\n"
        "• Prefer primary or high-quality secondary sources; avoid low-cred forums except for qualitative signals."
    ).strip()


def _researcher_rules() -> str:
    return (
        "Deep-search mandate:\n"
        "1) Use web + reddit to gather *current* evidence directly answering the user’s question.\n"
        "2) Extract concrete facts (metrics, dates, named entities), conflicting viewpoints, and practical takeaways.\n"
        "3) Ingest worthwhile sources and re-query RAG to consolidate.\n"
        "4) De-duplicate/cluster results; avoid repeating the same claim via many links.\n\n"
        "5) Focus on reddit search since we get first experiences from users there.\n"
        "Stop when: you have a coherent, well-cited set of findings with minimal gaps.\n\n"
        "Output format:\n"
        "• Detailed summary >=50 and <= 100 sentences, each with [n] if a claim)\n"
        "• Key findings (bulleted; keep each claim atomic, with [n])\n"
        "• Gaps & caveats (bulleted; what’s missing/ambiguous)\n"
        "• References: numbered list with title — URL"
    ).strip()


def _critic_rules() -> str:
    return (
        "Audit mandate:\n"
        "• Check factual accuracy, recency, and source quality; flag weak/duplicate/low-cred items.\n"
        "• Identify missing counter-evidence, confounders, and sampling/measurement issues.\n"
        "• Where weak, *use tools* to verify, replace, or upgrade sources (official stats, papers, reputable outlets).\n"
        "• Propose targeted follow-up queries or datasets to close gaps.\n\n"
        "Stop when: you’ve strengthened/trimmed the set and listed concrete follow-ups.\n\n"
        "Output format:\n"
        "• Quality review (bullets; cite with [n] where you bring evidence)\n"
        "• Fixes & replacements (bullets with suggested sources/queries)\n"
        "• References: numbered list (only the items you cite here)"
    ).strip()


def _judge_rules() -> str:
    return (
        "Synthesis mandate (no tool calls):\n"
        "• Produce a detailed brief that integrates the *final* evidence surfaced by Researcher & Critic.\n"
        "• Note confidence and the main residual uncertainties/gaps.\n"
        "• Keep neutral and pragmatic.\n\n"
        "Output format:\n"
        "• Answer (should be detailed, cite [n] where factual)\n"
        "• Why this answer (Write the key findings referencing [n])\n"
        "• Confidence (Low/Med/High) + 1–3 uncertainties"
    ).strip()


def make_agents(rag):
    model = OpenAIChatModel(
        model_name=settings.model_name,  # e.g., "llama3.1"
        provider=OllamaProvider(base_url=settings.openai_base_url),
    )
    bundle = make_tools(rag)
    tools = bundle["tools"]
    logger.info("Creating agents with model={m}; tools={n}", m=settings.model_name, n=len(tools))
    researcher_system = (
        react_preamble("Researcher")
        + "\n\n"
        + _researcher_rules()
        + "\n\nOperational rules:\n"
        + "1) Always begin with: retrieve(query=topic, k=settings.top_k).\n"
        + f"2) If results are thin or missing, call web_search_tool(query=topic, max_results={settings.max_search_results}) and/or reddit_search_tool(query=topic, max_results={settings.max_reddit_results}).\n"
        + "3) For any high-quality sources you find, call web_search_and_ingest(...) or reddit_search_and_ingest(...) so they are stored in RAG, then call retrieve(...) again to consolidate.\n"
        + "4) Produce a detailed, cited research summary answering what/why/how/pros/cons, and append a numbered References list mapping citations [n] -> title — URL.\n"
        + "5) Do NOT fabricate links or data. If evidence is lacking, mark it as a gap."
    )

    researcher = Agent(
        model=model,
        tools=tools,
        name="Researcher",
        system_prompt=researcher_system,
    )
    logger.debug("Researcher system prompt length={l}", l=len(researcher_system))

    critic_system = (
        react_preamble("Critic")
        + "\n\n"
        + _critic_rules()
        + "\n\nOperational rules:\n"
        + "1) Begin by retrieve(query=topic, k=settings.top_k) and read the Researcher's output from the execution log.\n"
        + f"2) If you find weak claims, use web_search_tool/reddit_search_tool (max_results={settings.max_search_results}) to verify; when you find better sources, call the ingest tools to update RAG and then retrieve again.\n"
        + "3) Produce a structured critique: Weaknesses, Fixes/Replacements, Suggested Follow-ups, and References (numbered).\n"
        + "4) Mark high-confidence vs low-confidence verifications."
    )

    critic = Agent(
        model=model,
        tools=tools,
        name="Critic",
        system_prompt=critic_system,
    )
    logger.debug("Critic system prompt length={l}", l=len(critic_system))

    judge_system = (
        react_preamble("Judge")
        + "\n\nDo NOT call tools. Base your synthesis strictly on what the Researcher and Critic surfaced in the execution log.\n\n"
        + _judge_rules()
    )

    judge = Agent(
        model=model,
        name="Judge",
        system_prompt=judge_system,
    )
    logger.debug("Judge system prompt length={l}", l=len(judge_system))

    logger.info("Agents created: Researcher, Critic, Judge")
    return researcher, critic, judge
