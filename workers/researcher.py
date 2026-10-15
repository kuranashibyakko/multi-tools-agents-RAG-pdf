import os
from pathlib import Path
import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
from groq import Groq

# =====================================================================
# 1. ENVIRONMENT & CLIENT CONFIGURATION
# =====================================================================
load_dotenv()

groq_client = Groq()
MODEL_NAME = "openai/gpt-oss-20b"

# Resolve absolute path to the local vector database
DB_PATH = str(Path("./chroma_db").resolve())

# Initialize ChromaDB persistent client
chroma_client = chromadb.PersistentClient(path=DB_PATH)

# Load the local sentence-transformer embedding model
embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

# Connect to the pre-indexed collection
juice_collection = chroma_client.get_or_create_collection(
    name="fruit_juice_study",
    embedding_function=embedding_fn,
    metadata={"hnsw:space": "cosine"},
)


# =====================================================================
# 2. INTERNAL RETRIEVAL HELPER
# =====================================================================
def retrieve_relevant_chunks(query: str, top_k: int = 3, threshold: float = 0.6) -> str:
    """Queries ChromaDB and returns filtered, citation-tagged text blocks."""
    try:
        results = juice_collection.query(
            query_texts=[query],
            n_results=top_k
        )

        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]

        filtered_excerpts = []
        for doc, meta, dist in zip(documents, metadatas, distances):
            if dist <= threshold:
                page_num = meta.get("page", "Unknown")
                filtered_excerpts.append(f"[Page {page_num}]: {doc}")

        if not filtered_excerpts:
            return "No relevant excerpts found within the confidence threshold."

        return "\n\n---\n\n".join(filtered_excerpts)

    except Exception as e:
        return f"Retrieval error: {str(e)}"


# =====================================================================
# 3. WORKER ENTRYPOINT (WHAT THE SUPERVISOR CALLS)
# =====================================================================
def run_researcher_worker(task: str) -> str:
    """Primary entrypoint for the Researcher Persona.
    
    1. Retrieves raw text chunks from the vector store.
    2. Uses an LLM extraction prompt to answer the specific task.
    3. Formats findings with citations for the Supervisor.
    """
    # Step A: Retrieve raw evidence from the vector database
    raw_evidence = retrieve_relevant_chunks(query=task, top_k=3, threshold=0.6)

    # If nothing relevant exists, fail fast without wasting an LLM call
    if "No relevant excerpts found" in raw_evidence or "Retrieval error" in raw_evidence:
        return f"[Researcher Worker Findings]: Unable to locate relevant data in documents for task: '{task}'."

    # Step B: Specialized extraction prompt
    system_prompt = (
        "You are an expert Research Analyst worker in a multi-agent system.\n"
        "Your role is to analyze retrieved document excerpts and answer the user's research task.\n\n"
        "Strict Operating Rules:\n"
        "1. Base your answer strictly on the provided excerpts. Do NOT assume, extrapolate, or invent facts.\n"
        "2. If an exact number or metric is not explicitly present in the text, state that it is omitted.\n"
        "3. Always cite the exact page numbers from the excerpts for every factual claim.\n"
        "4. Be concise, direct, and factual. Do not include conversational greetings or conclusions."
    )

    user_content = f"""RESEARCH TASK:
{task}

RETRIEVED DOCUMENT EVIDENCE:
{raw_evidence}

Provide a structured, cited factual summary addressing the research task."""

    # Step C: Synthesize through Groq
    response = groq_client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        temperature=0.0,
    )

    findings = response.choices[0].message.content or "No findings generated."

    # Step D: Package and return to the Supervisor
    return f"[Researcher Worker Findings]:\n{findings.strip()}"


# =====================================================================
# 4. DIRECT TEST HARNESS
# =====================================================================
if __name__ == "__main__":
    print("Testing Researcher Worker directly...\n")
    test_query = "What chemical indicator was used during the titration experiments, and on what page is it mentioned?"
    result = run_researcher_worker(test_query)
    print(result)