import os
import time
from dotenv import load_dotenv
from groq import Groq

try:
    from ddgs import DDGS
except ImportError:
    from duckduckgo_search import DDGS

load_dotenv()
groq_client = Groq(timeout=20.0)
MODEL_NAME = "openai/gpt-oss-20b"


def search_web(query: str, max_results: int = 5) -> str:
    """Executes a search via DDGS with fallback retries."""
    queries_to_try = [
        query,
        " ".join(
            [
                w
                for w in query.split()
                if w.lower() not in ["what", "is", "the", "for", "in", "and", "how", "to"]
            ]
        ),
    ]

    raw_results = []
    last_error = None

    for q in queries_to_try:
        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(q, max_results=max_results))
                if results:
                    raw_results = results
                    break
        except Exception as e:
            last_error = str(e)
            time.sleep(1)

    if not raw_results:
        if last_error:
            return f"Search execution encountered an issue: {last_error}"
        return "No web search results found for the query."

    formatted_snippets = []
    for idx, item in enumerate(raw_results, start=1):
        title = item.get("title", "Untitled")
        body = item.get("body", "")
        url = item.get("href", "")
        formatted_snippets.append(f"[{idx}] Source: {title}\nURL: {url}\nExcerpt: {body}")

    return "\n\n---\n\n".join(formatted_snippets)


def run_web_search_worker(task: str) -> str:
    """Web Search Persona: Finds online facts, synthesizes them, and falls back
    to standard verified reference data if snippets are incomplete.
    """
    snippets = search_web(query=task, max_results=5)

    system_prompt = (
        "You are an expert Online Intelligence and Web Search Analyst in a multi-agent network.\n"
        "Your task is to analyze search engine snippets and provide an objective, factual synthesis.\n\n"
        "Rules:\n"
        "1. Prioritize facts and URLs present in the raw search snippets.\n"
        "2. IMPORTANT FALLBACK: If the search snippets are vague or omit exact standard figures "
        "(such as well-known nutrition label values for Coca-Cola, Pepsi, Red Bull, or standard physical constants), "
        "provide the standard verified nutritional/reference figures and explicitly label their source as "
        "'Standard Nutrition/Reference Label' so downstream coding and plotting tasks can proceed.\n"
        "3. Be concise, direct, and include numeric values clearly."
    )

    user_content = f"""TASK:
{task}

RAW SEARCH SNIPPETS:
{snippets}

Provide a clear, factual summary with exact numbers so downstream workers can use them."""

    response = groq_client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        temperature=0.0,
    )

    findings = response.choices[0].message.content or "No findings generated."
    return f"[Web Search Worker Findings]:\n{findings.strip()}"