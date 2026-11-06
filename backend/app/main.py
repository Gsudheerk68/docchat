import asyncio
import json
import logging
import shutil
import tempfile
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from app import config, rag
from app import llm as llm_mod
from app.chains import build_chat_chain, build_rag_chain, text_of, to_messages

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s: %(message)s')
log = logging.getLogger('docuchat')
from app.schemas import ChatRequest, ChatResponse, DocumentInfo

@asynccontextmanager
async def lifespan(_app):
    def warm_up():  # load the vector store + embedding model in the background at startup
        try:
            rag.get_vectorstore()
            log.info("Vector store ready")
        except Exception:
            log.exception("Vector store warm-up failed")

    threading.Thread(target=warm_up, daemon=True).start()
    yield


app = FastAPI(title="DocuChat API", version="4.1", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)

MAX_UPLOAD_MB = 20


@app.get("/health")
def health():
    return {"status": "ok", "provider": config.LLM_PROVIDER}


# ---------- documents ----------
@app.get("/api/documents", response_model=list[DocumentInfo])
def documents():
    return rag.list_documents()


@app.post("/api/documents", response_model=DocumentInfo)
async def upload_document(file: UploadFile = File(...)):
    name = Path(file.filename or "upload").name
    suffix = Path(name).suffix.lower()
    if suffix not in rag.ALLOWED:
        raise HTTPException(400, "Only PDF, TXT and MD files are supported.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = Path(tmp.name)
    try:
        if tmp_path.stat().st_size > MAX_UPLOAD_MB * 1024 * 1024:
            raise HTTPException(413, f"File too large (max {MAX_UPLOAD_MB} MB).")
        return await run_in_threadpool(rag.ingest_file, tmp_path, name)
    except ValueError as e:
        raise HTTPException(400, str(e))
    finally:
        tmp_path.unlink(missing_ok=True)


@app.delete("/api/documents/{doc_id}")
def remove_document(doc_id: str):
    if not rag.delete_document(doc_id):
        raise HTTPException(404, "Document not found")
    return {"deleted": doc_id}


# ---------- chat ----------
@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    history = to_messages([m.model_dump() for m in req.history])
    try:
        if req.use_docs:
            docs = await run_in_threadpool(rag.retrieve, req.message)
            if not docs:
                return ChatResponse(reply="I don't have any documents to search yet. Upload a PDF, TXT or MD file first.")
            chain = build_rag_chain()
            msg = await run_in_threadpool(
                chain.invoke,
                {"question": req.message, "history": history, "context": rag.format_context(docs)},
            )
            return ChatResponse(reply=text_of(msg), sources=rag.sources_from(docs))

        chain = build_chat_chain()
        msg = await run_in_threadpool(chain.invoke, {"question": req.message, "history": history})
        return ChatResponse(reply=text_of(msg))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"LLM error: {e}")


# ---------- Phase 4: streaming chat (Server-Sent Events) ----------
def sse(event_type: str, data) -> str:
    return f"data: {json.dumps({'type': event_type, 'data': data})}\n\n"


@app.post("/api/chat/stream")
async def chat_stream(req: ChatRequest):
    history = to_messages([m.model_dump() for m in req.history])

    async def pipeline(q: asyncio.Queue):
        """Does the work (search + LLM) and pushes SSE frames into a queue."""
        t0 = time.time()
        try:
            if req.use_docs:
                await q.put(sse("status", "Searching your documents"))
                docs = await run_in_threadpool(rag.retrieve, req.message)
                log.info("retrieval: %.1fs, %d chunks", time.time() - t0, len(docs))
                if not docs:
                    await q.put(sse("token", "I don't have any documents to search yet. Upload a PDF, TXT or MD file first."))
                    await q.put(sse("done", ""))
                    return
                await q.put(sse("sources", rag.sources_from(docs)))
                chain = build_rag_chain()
                inputs = {"question": req.message, "history": history, "context": rag.format_context(docs)}
            else:
                chain = build_chat_chain()
                inputs = {"question": req.message, "history": history}

            got_text = False
            async for chunk in chain.astream(inputs):
                token = text_of(chunk)
                if token:
                    if not got_text:
                        log.info("first token after %.1fs", time.time() - t0)
                    got_text = True
                    await q.put(sse("token", token))
            if not got_text:
                log.error("Model returned an empty response (provider=%s)", config.LLM_PROVIDER)
                await q.put(sse("error", "The model returned an empty response. Open /api/llm-test to check the model and API key, and see the backend logs."))
                return
            log.info("answer finished in %.1fs", time.time() - t0)
            await q.put(sse("done", ""))
        except asyncio.CancelledError:
            raise
        except Exception as e:
            log.exception("Streaming chat failed")
            await q.put(sse("error", f"LLM error: {e}"))
        finally:
            await q.put(None)  # sentinel: nothing more to send

    async def events():
        q: asyncio.Queue = asyncio.Queue()
        task = asyncio.create_task(pipeline(q))
        try:
            while True:
                try:
                    item = await asyncio.wait_for(q.get(), timeout=10)
                except asyncio.TimeoutError:
                    yield ": ping\n\n"  # keep-alive so slow models don't look like a dropped connection
                    continue
                if item is None:
                    break
                yield item
        finally:
            task.cancel()

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ---------- diagnostics ----------
@app.get("/api/llm-test")
async def llm_test():
    """Open in the browser to check the model + API key work, independent of the chat UI."""
    try:
        msg = await run_in_threadpool(llm_mod.get_llm().invoke, "Reply with the single word: ok")
        return {
            "ok": True,
            "provider": config.LLM_PROVIDER,
            "model": config.LLM_MODEL or config.DEFAULT_MODELS.get(config.LLM_PROVIDER),
            "text": text_of(msg),
            "raw_content": msg.content,
        }
    except Exception as e:
        return {"ok": False, "provider": config.LLM_PROVIDER, "error": str(e)}
