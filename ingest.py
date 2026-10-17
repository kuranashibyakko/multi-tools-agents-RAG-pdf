import os
import re
from pathlib import Path
import chromadb
from chromadb.utils import embedding_functions
from pypdf import PdfReader

# 1. Paths and Client Setup
DB_PATH = str(Path("./chroma_db").resolve())
DOCS_DIR = Path("./docs").resolve()
DOCS_DIR.mkdir(exist_ok=True)

chroma_client = chromadb.PersistentClient(path=DB_PATH)
embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

collection = chroma_client.get_or_create_collection(
    name="fruit_juice_study",
    embedding_function=embedding_fn,
    metadata={"hnsw:space": "cosine"},
)


def extract_and_chunk_pdf(pdf_path: Path, chunk_size: int = 50, overlap: int = 10):
    """Extracts text from a PDF and chunks it with sliding windows and unique IDs."""
    reader = PdfReader(str(pdf_path))
    chunks = []
    metadatas = []
    ids = []

    # Clean filename stem for unique chunk IDs across different files
    doc_prefix = re.sub(r"[^a-zA-Z0-9_]", "_", pdf_path.stem)[:20]

    for page_idx, page in enumerate(reader.pages, start=1):
        raw_text = page.extract_text() or ""
        words = raw_text.split()
        if not words:
            continue

        stride = chunk_size - overlap
        chunk_num = 0

        for i in range(0, len(words), stride):
            chunk_words = words[i : i + chunk_size]
            chunk_text = " ".join(chunk_words)

            chunk_id = f"{doc_prefix}_p{page_idx}_c{chunk_num}"
            chunks.append(chunk_text)
            metadatas.append({"page": page_idx, "source": pdf_path.name})
            ids.append(chunk_id)
            chunk_num += 1

            if i + chunk_size >= len(words):
                break

    return ids, chunks, metadatas


def index_pdf_file(pdf_path: Path, chunk_size: int = 50, overlap: int = 10) -> dict:
    """Indexes or updates a single PDF into ChromaDB."""
    ids, documents, metadatas = extract_and_chunk_pdf(
        pdf_path, chunk_size=chunk_size, overlap=overlap
    )

    if not ids:
        return {"success": False, "chunks": 0, "total": collection.count(), "error": "No readable text extracted."}

    collection.upsert(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
    )

    return {
        "success": True,
        "chunks": len(ids),
        "total": collection.count(),
        "error": None,
    }


def get_indexed_sources() -> list[str]:
    """Returns a list of unique source PDF filenames currently indexed."""
    try:
        data = collection.get()
        if not data or not data.get("metadatas"):
            return []
        sources = {meta.get("source") for meta in data["metadatas"] if meta and "source" in meta}
        return sorted(list(sources))
    except Exception:
        return []

def delete_document_by_source(source_name: str) -> int:
    """Deletes all vector chunks belonging to a specific PDF filename."""
    try:
        existing = collection.get(where={"source": source_name})
        ids_to_delete = existing.get("ids", [])
        if ids_to_delete:
            collection.delete(ids=ids_to_delete)
        return len(ids_to_delete)
    except Exception:
        return 0


if __name__ == "__main__":
    default_pdf = DOCS_DIR / "Fruit_Juice_Sugar_Content_Transcription_AI.pdf"
    if default_pdf.exists():
        print(f"Reading and chunking: {default_pdf}")
        res = index_pdf_file(default_pdf)
        print(f"Ingested {res['chunks']} chunks. Total items in collection: {res['total']}")
    else:
        print(f"Default file not found at {default_pdf}. Ready for dynamic indexing.")