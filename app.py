import shutil
import time
from pathlib import Path
import streamlit as st

from ingest import collection, get_indexed_sources, index_pdf_file
from supervisor import run_supervisor

st.set_page_config(
    page_title="Multi-Agent Supervisor Studio v2.0",
    page_icon="🤖",
    layout="wide",
)

st.title("🤖 Multi-Agent Supervisor-Worker Network (v2.0)")
st.caption("Hierarchical Swarm with Dynamic RAG, Web Search, Visual Code Sandbox & Critic QA")

DOCS_DIR = Path("./docs").resolve()
DOCS_DIR.mkdir(exist_ok=True)
SANDBOX_DIR = Path("./sandbox").resolve()
SANDBOX_DIR.mkdir(exist_ok=True)

# =====================================================================
# SIDEBAR: DOCUMENT RE-INDEXING & SWARM TELEMETRY
# =====================================================================
with st.sidebar:
    st.header("📚 Document Knowledge Base")
    st.caption("Upload PDFs to dynamically index them into ChromaDB for the Researcher.")

    uploaded_files = st.file_uploader(
        "Choose PDF document(s)",
        type=["pdf"],
        accept_multiple_files=True,
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
                        f"(Total DB: {result['total']})"
                    )
                else:
                    st.error(f"Failed to index {uploaded_file.name}: {result['error']}")

    st.divider()

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
    - 📊 **Coder (v2.0):** Python & Matplotlib Visual Sandbox
    - 🛡️ **Critic:** Adversarial QA & Self-Healing Verifier
    """)
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# =====================================================================
# CHAT SESSION & VISUAL ARTIFACT RENDERING
# =====================================================================
if "messages" not in st.session_state:
    st.session_state.messages = []

for idx, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("plot_path") and Path(msg["plot_path"]).exists():
            st.image(msg["plot_path"], caption="Generated Sandbox Visualization", use_container_width=True)
            with open(msg["plot_path"], "rb") as img_file:
                st.download_button(
                    label="📥 Download Chart (PNG)",
                    data=img_file,
                    file_name=Path(msg["plot_path"]).name,
                    mime="image/png",
                    key=f"dl_{idx}",
                )
        if msg.get("telemetry"):
            st.caption(msg["telemetry"])

user_prompt = st.chat_input("Ask a question, run a simulation, or request a data visualization...")

if user_prompt:
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    with st.chat_message("assistant"):
        status_box = st.status("🚀 Supervisor assembling worker swarm...", expanded=True)
        start_time = time.time()
        metrics = {"dispatches": 0}

        # Clear any lingering plot before starting a new turn
        active_plot = SANDBOX_DIR / "output_plot.png"
        if active_plot.exists():
            active_plot.unlink()

        def visual_callback(event_type: str, data: str):
            if event_type == "dispatch":
                metrics["dispatches"] += 1
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
        elapsed = time.time() - start_time

        st.markdown(final_result)

        # Check if the Coder generated a matplotlib plot during this run
        saved_plot_path = None
        if active_plot.exists():
            timestamp_name = f"chart_{int(time.time())}.png"
            archived_plot = SANDBOX_DIR / timestamp_name
            shutil.copy(active_plot, archived_plot)
            saved_plot_path = str(archived_plot)

            st.image(saved_plot_path, caption="Generated Sandbox Visualization", use_container_width=True)
            with open(saved_plot_path, "rb") as img_file:
                st.download_button(
                    label="📥 Download Chart (PNG)",
                    data=img_file,
                    file_name=timestamp_name,
                    mime="image/png",
                    key=f"dl_live_{int(time.time())}",
                )

        telemetry_str = (
            f"⏱️ **Execution Time:** {elapsed:.2f}s | "
            f"🔄 **Worker Dispatches:** {metrics['dispatches']} | "
            f"🧠 **Model:** `openai/gpt-oss-20b`"
        )
        st.caption(telemetry_str)

    st.session_state.messages.append({
        "role": "assistant",
        "content": final_result,
        "plot_path": saved_plot_path,
        "telemetry": telemetry_str,
    })