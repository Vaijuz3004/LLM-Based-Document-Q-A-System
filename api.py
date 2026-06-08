"""
api.py — FastAPI Service with Streaming
=========================================
Exposes the RAG chain as an HTTP API with:
  - POST /ask          → standard JSON response
  - POST /ask/stream   → token-by-token streaming (SSE)
  - GET  /health       → health check
  - GET  /docs         → auto-generated Swagger UI (free from FastAPI)

CONCEPT: Why FastAPI?
──────────────────────
FastAPI is the modern standard for Python ML/AI APIs because:
1. Async-first (ASGI) — handles concurrent requests efficiently
2. Pydantic validation — automatic request/response validation
3. Auto-generated OpenAPI docs at /docs — no extra work
4. Type hints everywhere — better IDE support and fewer bugs
5. ~3x faster than Flask for I/O-bound workloads

INTERVIEW Q: What is the difference between ASGI and WSGI?
INTERVIEW Q: What does Pydantic do and why is it useful?
INTERVIEW Q: How does FastAPI generate API documentation automatically?
"""

from dotenv import load_dotenv
load_dotenv()

import json
import time
import asyncio
from typing import AsyncGenerator

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from chain import build_chain, build_chain_with_sources


# ─── App Initialization ───────────────────────────────────────────────────────

app = FastAPI(
    title="RAG Document Q&A API",
    description="Ask natural language questions about uploaded documents. "
                "Answers are grounded in source documents with citations.",
    version="1.0.0",
    # Swagger UI available at /docs
    # ReDoc available at /redoc
)


# ─── CORS Middleware ──────────────────────────────────────────────────────────
# CONCEPT: CORS (Cross-Origin Resource Sharing)
# ──────────────────────────────────────────────
# Browsers block JavaScript from calling APIs on different domains
# (security feature). CORS headers tell the browser which origins
# are allowed to call this API.
#
# allow_origins=["*"] allows ALL origins — fine for development,
# restrict to specific domains in production.
#
# INTERVIEW Q: What is CORS and why does it exist?
# INTERVIEW Q: What is the difference between a preflight request and a regular request?

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # restrict in production: ["https://myapp.com"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Startup: Load Chain Once ─────────────────────────────────────────────────
# CONCEPT: Application Lifespan
# ──────────────────────────────
# Loading the FAISS index + cross-encoder model takes ~5-10 seconds.
# If we loaded per request, every user would wait 10 seconds.
# Loading ONCE at startup means all requests share the same loaded chain.
#
# FastAPI lifespan context manager runs setup code before the first
# request and teardown code after the last request.
#
# INTERVIEW Q: Why load ML models at startup rather than per request?
# INTERVIEW Q: What is the FastAPI lifespan context manager?

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load expensive resources once at startup."""
    print("Loading RAG chain (FAISS index + cross-encoder)...")
    app.state.chain              = build_chain()
    app.state.chain_with_sources = build_chain_with_sources()
    app.state.start_time         = time.time()
    print("RAG chain loaded. API ready.")
    yield  # ← server runs here
    # Cleanup on shutdown (if needed)
    print("Shutting down...")

app.router.lifespan_context = lifespan


# ─── Pydantic Models ──────────────────────────────────────────────────────────
# CONCEPT: Pydantic Request/Response Models
# ──────────────────────────────────────────
# Pydantic models define the shape of API inputs and outputs.
# FastAPI uses them to:
# 1. Validate incoming JSON (returns 422 if invalid)
# 2. Serialise outgoing Python objects to JSON
# 3. Generate OpenAPI schema for /docs automatically
#
# Field() adds metadata: description, example, constraints
# These appear in the /docs Swagger UI.
#
# INTERVIEW Q: What is Pydantic v2 and what changed from v1?
# INTERVIEW Q: What HTTP status code does FastAPI return for validation errors?

class QueryRequest(BaseModel):
    question: str = Field(
        ...,   # ... means required
        min_length=3,
        max_length=500,
        description="The question to ask about the documents",
        examples=["What are the payment terms?"]
    )

class SourceDocument(BaseModel):
    content: str
    source:  str
    page:    int | str

class QueryResponse(BaseModel):
    question: str
    answer:   str
    sources:  list[SourceDocument]
    latency_ms: float

class HealthResponse(BaseModel):
    status:   str
    uptime_s: float
    version:  str


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health(request: Request):
    """
    Health check endpoint.

    CONCEPT: Health Checks
    ───────────────────────
    Essential in production for:
    - Load balancers to detect unhealthy instances
    - Kubernetes liveness/readiness probes
    - Monitoring systems (Datadog, CloudWatch)

    INTERVIEW Q: What is the difference between liveness and readiness probes?
    """
    uptime = time.time() - request.app.state.start_time
    return HealthResponse(
        status="ok",
        uptime_s=round(uptime, 1),
        version="1.0.0"
    )


@app.post("/ask", response_model=QueryResponse, tags=["Q&A"])
async def ask(req: QueryRequest, request: Request):
    """
    Standard Q&A endpoint.
    Waits for the full answer before responding.

    Use this when:
    - Building a backend that processes answers programmatically
    - You don't need real-time streaming in the UI
    - Response time doesn't affect UX (batch processing, etc.)

    CONCEPT: async def vs def in FastAPI
    ──────────────────────────────────────
    FastAPI supports both sync and async route handlers.
    Use async def when the handler does I/O (database calls, API calls)
    so the event loop can handle other requests while waiting.
    Use def for CPU-bound work (FastAPI runs it in a thread pool).

    LangChain's chain.invoke() is synchronous I/O (OpenAI API call).
    We wrap it with asyncio.to_thread() to avoid blocking the event loop.

    INTERVIEW Q: What is Python's asyncio event loop?
    INTERVIEW Q: When would you use async def vs def in FastAPI?
    INTERVIEW Q: What is the Global Interpreter Lock (GIL)?
    """
    start = time.time()
    try:
        # Run synchronous LangChain call in a thread pool
        # This frees the event loop to handle other requests
        result = await asyncio.to_thread(
            request.app.state.chain_with_sources.invoke,
            req.question
        )

        latency_ms = round((time.time() - start) * 1000, 1)

        # Extract source documents
        sources = [
            SourceDocument(
                content=doc.page_content[:300],  # truncate for response
                source=doc.metadata.get("source", "unknown").split("/")[-1],
                page=doc.metadata.get("page", "?")
            )
            for doc in result.get("sources", [])
        ]

        return QueryResponse(
            question=req.question,
            answer=result["answer"],
            sources=sources,
            latency_ms=latency_ms
        )

    except FileNotFoundError:
        raise HTTPException(
            status_code=503,
            detail="FAISS index not found. Run python ingest.py first."
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ask/stream", tags=["Q&A"])
async def ask_stream(req: QueryRequest, request: Request):
    """
    Streaming Q&A endpoint using Server-Sent Events (SSE).
    Returns tokens as they are generated — no waiting for the full answer.

    Use this when:
    - Building a chat UI (tokens appear progressively like ChatGPT)
    - Answers are long and users should see progress
    - Low perceived latency is important for UX

    CONCEPT: Server-Sent Events (SSE)
    ───────────────────────────────────
    SSE is a simple HTTP protocol for server-to-client streaming:
    - Client makes ONE HTTP request
    - Connection stays open
    - Server pushes events as "data: <payload>\\n\\n" lines
    - Client receives events via EventSource API in browser

    SSE format rules:
        data: <payload>\\n\\n    ← one event
        data: token1\\n\\n
        data: token2\\n\\n
        data: [DONE]\\n\\n       ← sentinel to signal end of stream

    CONCEPT: SSE vs WebSockets
    ───────────────────────────
    SSE:
      - One-way: server → client only
      - Built on plain HTTP (works through proxies/CDNs)
      - Simpler: no special protocol
      - Auto-reconnects if connection drops
      - Perfect for LLM streaming

    WebSockets:
      - Two-way: full-duplex communication
      - Separate protocol (ws:// wss://)
      - More complex setup
      - Better for chat apps, multiplayer games, real-time collaboration

    INTERVIEW Q: What is the difference between SSE and WebSockets?
    INTERVIEW Q: How does EventSource work in the browser?
    INTERVIEW Q: What is the SSE data format?
    """
    async def token_generator() -> AsyncGenerator[str, None]:
        """
        CONCEPT: AsyncGenerator
        ────────────────────────
        An async generator is a function that yields values
        asynchronously. FastAPI's StreamingResponse consumes it
        and sends each yielded string as an HTTP chunk.

        We run chain.stream() (synchronous) in a thread executor
        and yield each token as an SSE event.
        """
        try:
            # Metadata event first — tells client what's coming
            meta = json.dumps({"question": req.question, "type": "start"})
            yield f"data: {meta}\n\n"

            # Stream tokens from the LangChain chain
            # chain.stream() is a synchronous generator
            # We run it in a thread pool so we don't block the event loop
            loop = asyncio.get_event_loop()
            queue: asyncio.Queue[str | None] = asyncio.Queue()

            def run_stream():
                """Runs in a thread pool — feeds tokens into the async queue."""
                for chunk in request.app.state.chain.stream(req.question):
                    loop.call_soon_threadsafe(queue.put_nowait, chunk)
                loop.call_soon_threadsafe(queue.put_nowait, None)  # sentinel

            # Start streaming in background thread
            loop.run_in_executor(None, run_stream)

            # Yield tokens as they arrive
            while True:
                token = await queue.get()
                if token is None:
                    break  # stream finished
                # Escape newlines in SSE payload
                payload = json.dumps({"token": token, "type": "token"})
                yield f"data: {payload}\n\n"

            # Done event
            yield f"data: {json.dumps({'type': 'done'})}\n\n"

        except Exception as e:
            error_payload = json.dumps({"type": "error", "message": str(e)})
            yield f"data: {error_payload}\n\n"

    return StreamingResponse(
        token_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control":    "no-cache",
            "X-Accel-Buffering": "no",    # disable nginx buffering
            "Connection":       "keep-alive",
        }
    )


@app.get("/", tags=["System"])
async def root():
    """Redirect hint to /docs."""
    return JSONResponse({
        "message": "RAG Q&A API is running",
        "docs":    "http://localhost:8000/docs",
        "health":  "http://localhost:8000/health"
    })


# ─── Run directly ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn

    # CONCEPT: uvicorn
    # ─────────────────
    # uvicorn is an ASGI server — it handles async Python web apps.
    # ASGI = Asynchronous Server Gateway Interface (successor to WSGI).
    #
    # --reload: watch for file changes and restart (development only)
    # --workers: number of worker processes (production: 4 per CPU core)
    # --host 0.0.0.0: listen on all network interfaces
    #
    # INTERVIEW Q: What is the difference between WSGI and ASGI?
    # INTERVIEW Q: How many workers should you run in production?

    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,     # set False in production
        log_level="info"
    )
