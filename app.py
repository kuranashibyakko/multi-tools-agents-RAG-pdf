import shutil
import time
from pathlib import Path
import streamlit as st

from ingest import (
    collection,
    delete_document_by_source,
    get_indexed_sources,
    index_pdf_file,
)
from supervisor import run_supervisor
from telemetry import get_telemetry_summary, log_agent_run

st.set_page_config(
    page_title="Multi-Agent Supervisor Studio v3.0",
    page_icon="🤖",
    layout="wide",
)

st.title("🤖 Multi-Agent Supervisor-Worker Network (v3.0)")
st.caption("Parallel Swarm Execution • Multi-Turn Memory • Visual Sandbox • SQLite Telemetry")

DOCS_DIR = Path("./docs").resolve()
DOCS_DIR.mkdir(exist_ok=True)
SANDBOX_DIR = Path("./sandbox").resolve()
SANDBOX_DIR.mkdir(exist_ok=True)

# =====================================================================
# SIDEBAR: DOCUMENT KNOWLEDGE BASE & MANAGEMENT
# =====================================================================
with st.sidebar:
    st.header("📚 Document Knowledge Base")
    uploaded_files = st.file_uploader(
        "Upload PDF(s) to ChromaDB",
        type=["pdf"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        if st.button("⚡ Index Uploaded PDFs", use_container_width=True):
            for uploaded_file in uploaded_files:
                save_path = DOCS_DIR / uploaded_file.name
                with open(save_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                with st.spinner(f"Embedding `{uploaded_file.name}`..."):
                    result = index_pdf_file(save_path)

                if result["success"]:
                    st.success(f"Indexed {result['chunks']} chunks from `{uploaded_file.name}`")
                else:
                    st.error(f"Failed: {result['error']}")

    st.divider()

    indexed_sources = get_indexed_sources()
    st.subheader(f"Indexed Documents ({len(indexed_sources)})")
    if indexed_sources:
        for src in indexed_sources:
            col1, col2 = st.columns([4, 1])
            col1.markdown(f"📄 `{src}`")
            if col2.button("✖", key=f"del_{src}", help=f"Remove {src} from ChromaDB"):
                deleted_count = delete_document_by_source(src)
                st.toast(f"Removed {deleted_count} chunks for {src}")
                st.rerun()
        st.caption(f"Total vector chunks: **{collection.count()}**")
    else:
        st.info("No documents indexed.")

    st.divider()
    if st.button("🗑️ Clear Chat Memory", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# =====================================================================
# MAIN WORKSPACE TABS
# =====================================================================
tab_chat, tab_telemetry = st.tabs(["💬 Agent Workspace", "📊 System Telemetry & Audit Logs"])

with tab_chat:
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

    user_prompt = st.chat_input("Ask a question, follow up on a previous turn, or plot data...")

    if user_prompt:
        prior_history = list(st.session_state.messages)
        st.session_state.messages.append({"role": "user", "content": user_prompt})
        with st.chat_message("user"):
            st.markdown(user_prompt)

        with st.chat_message("assistant"):
            status_box = st.status("🚀 Supervisor orchestrating worker swarm...", expanded=True)
            start_time = time.time()
            metrics = {"dispatches": 0}

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
                user_prompt,
                max_iterations=6,
                step_callback=visual_callback,
                chat_history=prior_history,
            )
            elapsed = time.time() - start_time

            st.markdown(final_result)

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

            log_agent_run(
                prompt=user_prompt,
                dispatches=metrics["dispatches"],
                latency_sec=elapsed,
                chart_generated=bool(saved_plot_path),
            )

            telemetry_str = (
                f"⏱️ **Execution Time:** {elapsed:.2f}s | "
                f"🔄 **Worker Dispatches:** {metrics['dispatches']} | "
                f"🧠 **Context Turns:** {len(prior_history) // 2} | "
                f"⚡ **Model:** `openai/gpt-oss-20b`"
            )
            st.caption(telemetry_str)

        st.session_state.messages.append({
            "role": "assistant",
            "content": final_result,
            "plot_path": saved_plot_path,
            "telemetry": telemetry_str,
        })

with tab_telemetry:
    st.subheader("📈 Historical Swarm Performance (SQLite)")
    summary = get_telemetry_summary()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Runs Logged", summary["total_runs"])
    c2.metric("Avg Latency (s)", f"{summary['avg_latency']}s")
    c3.metric("Avg Worker Dispatches", summary["avg_dispatches"])
    c4.metric("Visual Charts Generated", summary["total_charts"])

    st.divider()
    st.markdown("#### Recent Execution Traces")
    if summary["recent_runs"]:
        st.dataframe(summary["recent_runs"], use_container_width=True)
    else:
        st.info("No runs recorded yet. Submit a prompt in the Agent Workspace tab!")