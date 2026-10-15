# Autonomous Multi-Agent Research & Analysis Network

An autonomous multi-agent system featuring hierarchical orchestration, local document RAG, live web intelligence, a sandboxed Python execution environment, and an adversarial Critic with self-healing feedback loops.

---

## Architecture Overview

```text
                           ┌────────────────────────┐
                           │      User Prompt       │
                           └───────────┬────────────┘
                                       │
                                       ▼
                           ┌────────────────────────┐
                           │    Supervisor Agent    │
                           │  (Planner & Evaluator) │
                           └───────────┬────────────┘
                                       │
        ┌───────────────────┬──────────┴──────────┬───────────────────┐
        │                   │                     │                   │
        ▼                   ▼                     ▼                   ▼
┌───────────────┐   ┌───────────────┐     ┌───────────────┐   ┌───────────────┐
│  Researcher   │   │  Web Search   │     │ Coder Sandbox │   │ Critic (QA)   │
│  (ChromaDB    │   │  (DuckDuckGo  │     │ (Subprocess   │   │ (Adversarial  │
│  MiniLM RAG)  │   │  Live DDGS)   │     │ Execution)    │   │ Verification) │
└───────┬───────┘   └───────┬───────┘     └───────┬───────┘   └───────┬───────┘
        │                   │                     │                   │
        └───────────────────┴──────────┬──────────┴───────────────────┘
                                       │ Tool Observations
                                       ▼
                           ┌────────────────────────┐
                           │   Supervisor Synthesis │
                           │  (Self-Healing Loop)   │
                           └────────────────────────┘


Specialized Personas
Supervisor (supervisor.py): Central orchestrator using Groq tool calling to plan tasks, route sub-goals, and recover automatically from Critic rejections.

Researcher (workers/researcher.py): Local semantic document RAG powered by ChromaDB and all-MiniLM-L6-v2 with distance thresholding and citation extraction.

Web Search Worker (workers/web_search.py): Live external search using DuckDuckGo (ddgs) with citation link formatting.

Coder Worker (workers/coder.py): Writes standalone Python scripts and executes them inside an isolated ./sandbox/ subprocess.

Critic Worker (workers/critic.py): Adversarial QA gate evaluating outputs for accuracy, completeness, and citations with PASSED / NEEDS_REVISION feedback.

Project Structure
Plaintext
multi-agent/
├── app.py                  # Streamlit visual dashboard with dynamic PDF upload
├── supervisor.py           # CLI orchestrator & agent decision loop
├── ingest.py               # Document chunking & ChromaDB ingestion
├── workers/
│   ├── __init__.py
│   ├── coder.py            # Code execution sandbox
│   ├── critic.py           # Quality assurance & verification
│   ├── researcher.py       # Local ChromaDB vector retrieval
│   └── web_search.py       # DuckDuckGo live search
├── docs/                   # PDF documents directory
├── sandbox/                # Subprocess execution folder
├── requirements.txt
├── .env.example
└── .gitignore
Setup & Installation
Clone the repository:

Bash
git clone [https://github.com/](https://github.com/)<your-username>/<your-repo-name>.git
cd <your-repo-name>
Create and activate a virtual environment:

Bash
python3 -m venv venv
source venv/bin/activate
Install dependencies:

Bash
pip install -r requirements.txt
Configure environment variables:

Bash
cp .env.example .env
Add your Groq API key in .env:

Ini, TOML
GROQ_API_KEY=gsk_your_actual_key_here
Index baseline documents into ChromaDB:
Place your reference PDF inside ./docs/ and run:

Bash
python ingest.py
Running the System
Option A: Interactive Streamlit Dashboard
Bash
streamlit run app.py
Open http://localhost:8501 to use the chat interface, upload new PDFs dynamically, and inspect live agent execution traces.

Option B: Terminal CLI
Bash
python supervisor.py
"""
with open("README.md", "w") as f:
f.write(content.strip() + "\n")
print("README.md created successfully!")
'


---

### Verify and Stage for Git

Check that the file exists and has size:

```bash
ls -lh README.md
git status