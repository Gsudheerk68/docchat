"""RAG pipeline: load -> split -> embed -> store (Chroma) -> retrieve."""
import json
import threading
import uuid
from pathlib import Path

from fastembed import TextEmbedding
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from app import config

DATA = Path(config.DATA_DIR)
REGISTRY = DATA / "registry.json"
ALLOWED = {".pdf", ".txt", ".md"}

splitter = RecursiveCharacterTextSplitter(
    chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP
)


class LocalEmbeddings(Embeddings):
    """Free, local embeddings (ONNX via fastembed). Model downloads once on first use."""

    def __init__(self, model_name: str):
        self.model = TextEmbedding(model_name=model_name, cache_dir=str(DATA / "models"))

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [v.tolist() for v in self.model.embed(texts)]

    def embed_query(self, text: str) -> list[float]:
        return next(iter(self.model.embed([text]))).tolist()


def load_pages(path: Path) -> list[Document]:
    """Return one Document per PDF page, or one for a text file."""
    if path.suffix.lower() == ".pdf":
        reader = PdfReader(str(path))
        return [
            Document(page_content=page.extract_text() or "", metadata={"page": i})
            for i, page in enumerate(reader.pages)
        ]
    return [Document(page_content=path.read_text(encoding="utf-8", errors="ignore"))]


_vs = None
_vs_lock = threading.Lock()


def get_vectorstore() -> Chroma:
    """Create the vector store once. The lock stops two request threads from creating
    the Chroma client at the same time (that race caused KeyError: '/data/chroma')."""
    global _vs
    with _vs_lock:
        if _vs is None:
            DATA.mkdir(parents=True, exist_ok=True)
            _vs = Chroma(
                collection_name="docuchat",
                embedding_function=LocalEmbeddings(config.EMBEDDING_MODEL),
                persist_directory=str(DATA / "chroma"),
            )
        return _vs


# ---- tiny JSON registry so the UI can list/delete documents ----
def _load_registry() -> dict:
    return json.loads(REGISTRY.read_text()) if REGISTRY.exists() else {}


def _save_registry(reg: dict) -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    REGISTRY.write_text(json.dumps(reg, indent=2))


def list_documents() -> list[dict]:
    return [{"id": k, **v} for k, v in _load_registry().items()]


def ingest_file(path: Path, original_name: str) -> dict:
    """Load a file, split into chunks, embed and store. Returns registry entry."""
    suffix = path.suffix.lower()
    if suffix not in ALLOWED:
        raise ValueError(f"Unsupported file type '{suffix}'. Use PDF, TXT or MD.")

    pages = load_pages(path)
    chunks = splitter.split_documents(pages)
    chunks = [c for c in chunks if c.page_content.strip()]
    if not chunks:
        raise ValueError("No readable text found in this file (scanned PDFs need OCR).")

    doc_id = uuid.uuid4().hex[:12]
    for c in chunks:
        c.metadata = {
            "doc_id": doc_id,
            "source": original_name,
            "page": int(c.metadata.get("page", 0)) + 1 if suffix == ".pdf" else None,
        }
        c.metadata = {k: v for k, v in c.metadata.items() if v is not None}  # Chroma rejects None

    get_vectorstore().add_documents(chunks, ids=[f"{doc_id}-{i}" for i in range(len(chunks))])

    reg = _load_registry()
    reg[doc_id] = {"name": original_name, "chunks": len(chunks)}
    _save_registry(reg)
    return {"id": doc_id, **reg[doc_id]}


def delete_document(doc_id: str) -> bool:
    reg = _load_registry()
    if doc_id not in reg:
        return False
    get_vectorstore()._collection.delete(where={"doc_id": doc_id})
    del reg[doc_id]
    _save_registry(reg)
    return True


def retrieve(question: str):
    """Return the top-k most similar chunks for the question."""
    if not _load_registry():
        return []
    return get_vectorstore().similarity_search(question, k=config.TOP_K)


def format_context(docs) -> str:
    parts = []
    for i, d in enumerate(docs, 1):
        page = f", page {d.metadata['page']}" if "page" in d.metadata else ""
        parts.append(f"[{i}] ({d.metadata.get('source', 'unknown')}{page})\n{d.page_content}")
    return "\n\n".join(parts)


def sources_from(docs) -> list[dict]:
    seen, out = set(), []
    for i, d in enumerate(docs, 1):
        key = (d.metadata.get("source"), d.metadata.get("page"))
        if key in seen:
            continue
        seen.add(key)
        out.append(
            {
                "n": i,
                "source": d.metadata.get("source", "unknown"),
                "page": d.metadata.get("page"),
                "snippet": d.page_content[:220].strip(),
            }
        )
    return out
