from pathlib import Path
import streamlit as st

from ingest import collection, get_indexed_sources, index_pdf_file
from supervisor import run_supervisor

st.set_page_config(
    page_title="Multi-Agent Supervisor Studio",
    page_icon="🤖",
    layout="wide",
)

st.title("🤖 Multi-Agent Supervisor-Worker Network")
st.caption("Hierarchical Swarm with Dynamic RAG, Web Search, Isolated Sandbox & Critic QA")

DOCS_DIR = Path("./docs").resolve()
DOCS_DIR.mkdir(exist_ok=True)

# =====================================================================
# SIDEBAR: DOCUMENT RE-INDEXING & SWARM STATUS
# =====================================================================
with st.sidebar:
    st.header("📚 Document Knowledge Base")
    st.caption("Upload PDFs to dynamically index them into ChromaDB for the Researcher.")

    uploaded_files = st.file_uploader(
        "Choose PDF document(s)",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload study papers, lab notes, or reports to expand the Researcher's local knowledge.",
    )

    if uploaded_files:
        if st.button("⚡ Index Uploaded PDFs", use_container_width=True):
            for uploaded_file in uploaded_files:
                save_path = DOCS_DIR / uploaded_file.name
                with open(save_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                with st.spinner(f"Chunking & embedding `{uploaded_file.name}`..."):
                    result = index_pdf_file(save_path)

                if result["success"]:
                    st.success(
                        f"**{uploaded_file.name}**: Indexed {result['chunks']} chunks! "
                        f"(Total DB size: {result['total']})"
                    )
                else:
                    st.error(f"Failed to index {uploaded_file.name}: {result['error']}")

    st.divider()

    # Display currently indexed files
    indexed_sources = get_indexed_sources()
    st.subheader(f"Indexed Documents ({len(indexed_sources)})")
    if indexed_sources:
        for src in indexed_sources:
            st.markdown(f"- 📄 `{src}`")
        st.caption(f"Total vector chunks in ChromaDB: **{collection.count()}**")
    else:
        st.info("No documents currently indexed.")

    st.divider()

    st.header("Active Agent Swarm")
    st.markdown("""
    - 👑 **Supervisor:** Orchestration & Step Planning
    - 📚 **Researcher:** ChromaDB Vector Retrieval (`all-MiniLM-L6-v2`)
    - 🌐 **Web Search:** DuckDuckGo Real-Time Search (`ddgs`)
    - ⚡ **Coder:** Subprocess-Isolated Python Sandbox
    - 🛡️ **Critic:** Adversarial QA & Self-Healing Verifier
    """)
    st.divider()
    st.markdown("**Local Stack:** WSL2 Linux | Groq `openai/gpt-oss-20b`")


# =====================================================================
# CHAT SESSION & ORCHESTRATION TIMELINE
# =====================================================================
if "messages" not in st.session_state:
    st.session_state.messages = []

# Render conversation history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User prompt bar
user_prompt = st.chat_input("Ask a question across indexed documents, the web, or code simulations...")

if user_prompt:
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    with st.chat_message("assistant"):
        status_box = st.status("🚀 Supervisor assembling worker swarm...", expanded=True)

        def visual_callback(event_type: str, data: str):
            if event_type == "dispatch":
                status_box.write(f"🔄 {data}")
            elif event_type == "observation":
                with status_box.expander("📋 Inspect Worker Output", expanded=False):
                    st.markdown(data)
            elif event_type == "final":
                status_box.update(
                    label="✅ All Worker Tasks Verified and Complete!",
                    state="complete",
                    expanded=False,
                )

        final_result = run_supervisor(
            user_prompt, max_iterations=6, step_callback=visual_callback
        )
        st.markdown(final_result)

    st.session_state.messages.append({"role": "assistant", "content": final_result})