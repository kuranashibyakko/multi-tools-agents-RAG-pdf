import os
import time
from dotenv import load_dotenv
from groq import Groq

# Use the updated library
try:
    from ddgs import DDGS
except ImportError:
    from duckduckgo_search import DDGS

load_dotenv()
groq_client = Groq()
MODEL_NAME = "openai/gpt-oss-20b"


def search_web(query: str, max_results: int = 4) -> str:
    """Executes a search via DDGS with fallback retries."""
    queries_to_try = [
        query,
        # Stripped query fallback (e.g., removing question framing)
        " ".join([w for w in query.split() if w.lower() not in ["what", "is", "the", "for", "in", "and", "how", "to"]])
    ]

    raw_results = []
    last_error = None

    for q in queries_to_try:
        try:
            with DDGS() as ddgs:
                # text search with a slight delay if retrying
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
        formatted_snippets.append(
            f"[{idx}] Source: {title}\nURL: {url}\nExcerpt: {body}"
        )

    return "\n\n---\n\n".join(formatted_snippets)


def run_web_search_worker(task: str) -> str:
    """Web Search Persona: Finds online facts, synthesizes them,

    and returns citations with source URLs.
    """
    snippets = search_web(query=task, max_results=4)

    # If the search engine is blocked or empty, return a graceful, honest message
    if "No web search results found" in snippets or "encountered an issue" in snippets:
        return (
            f"[Web Search Worker Findings]: Could not retrieve live web snippets for query: '{task}'. "
            f"Details: {snippets}"
        )

    system_prompt = (
        "You are an expert Online Intelligence and Web Search Analyst in a multi-agent network.\n"
        "Your task is to analyze raw search engine snippets and provide an objective, factual synthesis.\n\n"
        "Rules:\n"
        "1. Answer ONLY using the facts present in the search snippets.\n"
        "2. Include the source URLs as markdown links or citations for every major claim.\n"
        "3. If snippets contradict each other, mention the disagreement.\n"
        "4. Be concise, direct, and factual. No conversational filler."
    )

    user_content = f"""TASK:
{task}

RAW SEARCH SNIPPETS:
{snippets}

Provide a verified, cited summary based on the snippets."""

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


if __name__ == "__main__":
    print("Testing Web Search Worker directly...\n")
    test_task = "WHO recommended daily limit for free sugar intake in grams"
    print(run_web_search_worker(test_task))