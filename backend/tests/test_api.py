from fastapi.testclient import TestClient
from langchain_core.documents import Document
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from app import chains, rag
from app.main import app

client = TestClient(app)


def fake_llm(text="Hello from fake LLM"):
    return lambda: FakeListChatModel(responses=[text])


def test_health():
    assert client.get("/health").status_code == 200


def test_plain_chat(monkeypatch):
    monkeypatch.setattr(chains, "get_llm", fake_llm())
    r = client.post("/api/chat", json={"message": "hi"})
    assert r.status_code == 200 and r.json()["reply"] == "Hello from fake LLM"


def test_rag_chat_returns_sources(monkeypatch):
    doc = Document(page_content="LangChain is a framework for LLM apps.", metadata={"source": "notes.txt"})
    monkeypatch.setattr(rag, "retrieve", lambda q: [doc])
    monkeypatch.setattr(chains, "get_llm", fake_llm("It is a framework [1]."))
    r = client.post("/api/chat", json={"message": "What is LangChain?", "use_docs": True})
    body = r.json()
    assert r.status_code == 200
    assert body["sources"][0]["source"] == "notes.txt"


def test_rag_without_docs(monkeypatch):
    monkeypatch.setattr(rag, "retrieve", lambda q: [])
    r = client.post("/api/chat", json={"message": "anything", "use_docs": True})
    assert "upload" in r.json()["reply"].lower()


def test_bad_extension():
    r = client.post("/api/documents", files={"file": ("x.exe", b"abc")})
    assert r.status_code == 400


def test_ingest_retrieve_delete(tmp_path, monkeypatch):
    """End-to-end through Chroma using a tiny fake embedder (no model download)."""
    from langchain_chroma import Chroma
    from langchain_core.embeddings import Embeddings

    class Fake(Embeddings):
        def _v(self, t):
            return [float(t.lower().count(c)) for c in "aeiou"] + [1.0]

        def embed_documents(self, texts):
            return [self._v(t) for t in texts]

        def embed_query(self, text):
            return self._v(text)

    monkeypatch.setattr(rag, "DATA", tmp_path)
    monkeypatch.setattr(rag, "REGISTRY", tmp_path / "registry.json")
    store = Chroma(collection_name="test_col", embedding_function=Fake(), persist_directory=str(tmp_path / "c"))
    monkeypatch.setattr(rag, "get_vectorstore", lambda: store)

    f = tmp_path / "a.txt"
    f.write_text("Alumni Connect is a networking platform for students and alumni. " * 5)
    info = rag.ingest_file(f, "a.txt")
    assert info["chunks"] >= 1
    assert rag.retrieve("alumni platform")[0].metadata["source"] == "a.txt"
    assert rag.delete_document(info["id"]) is True
    assert rag.list_documents() == []


def test_stream(monkeypatch):
    import json

    monkeypatch.setattr(chains, "get_llm", fake_llm("streamed answer"))
    with client.stream("POST", "/api/chat/stream", json={"message": "hi"}) as r:
        body = "".join(r.iter_text())
    events = [json.loads(l[6:]) for l in body.split("\n\n") if l.startswith("data: ")]
    assert "".join(e["data"] for e in events if e["type"] == "token") == "streamed answer"
    assert events[-1]["type"] == "done"


def test_list_content_is_handled(monkeypatch):
    """Models that return content as a list of blocks must still produce text."""
    from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
    from langchain_core.messages import AIMessage

    monkeypatch.setattr(
        chains, "get_llm",
        lambda: GenericFakeChatModel(messages=iter([AIMessage(content=[{"type": "text", "text": "block text"}])])),
    )
    r = client.post("/api/chat", json={"message": "hi"})
    assert r.json()["reply"] == "block text"


def test_stream_empty_response_reports_error(monkeypatch):
    import json
    from langchain_core.language_models.fake_chat_models import FakeListChatModel

    monkeypatch.setattr(chains, "get_llm", lambda: FakeListChatModel(responses=[""]))
    with client.stream("POST", "/api/chat/stream", json={"message": "hi"}) as r:
        body = "".join(r.iter_text())
    events = [json.loads(l[6:]) for l in body.split("\n\n") if l.startswith("data: ")]
    assert events[-1]["type"] == "error"
