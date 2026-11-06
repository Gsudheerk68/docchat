
**Backend**
- `POST /api/chat/stream` — Server-Sent Events: `sources` → `token`… → `done` (or `error`), using `chain.astream()`
- Embedding model cache stored under `DATA_DIR` so it survives restarts
- `Dockerfile` for the API

**Frontend**
- `api.js` `streamChat()` reads the stream with `fetch` + `ReadableStream`
- `App.jsx` appends tokens live, shows sources first, supports **Stop** (`AbortController`)
- `Dockerfile` + `nginx.conf` (static build, proxies `/api`, buffering off for SSE)

**Root:** `docker-compose.yml`, `.gitignore`

## Run with Docker
```bash
cp backend/.env.example backend/.env     # add your API key
docker compose up --build
# open http://localhost:8080
```

## Run without Docker
Same as Phase 3 (`uvicorn` on :8000, `npm run dev` on :5173).

## Ideas for next steps
- Auth (login) and per-user document collections
- Hybrid search or a re-ranker for better retrieval
- Save chat history in SQLite/Postgres
- Swap Chroma for pgvector or Qdrant
- Evaluate answers (RAGAS) and add LangSmith tracing
