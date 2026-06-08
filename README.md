# LLM-Based Document Q&A System (RAG Pipeline)

End-to-end RAG pipeline: ingest PDFs → vector search → LLM answers with citations.

## Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| PDF loading | LangChain PyPDFLoader | Extract text + metadata from PDFs |
| Chunking | RecursiveCharacterTextSplitter | Split docs into focused windows |
| Embeddings | OpenAI text-embedding-3-small | Convert text → 1536-dim vectors |
| Vector store | FAISS | Store + search vectors |
| Re-ranking | cross-encoder/ms-marco-MiniLM-L-6-v2 | Re-rank top-k for accuracy |
| LLM | GPT-4o-mini | Generate grounded answers |
| API | FastAPI + uvicorn | HTTP service with SSE streaming |

## Project Structure

```
rag-qa-system/
├── docs/                   ← place your PDFs here
├── faiss_index/            ← auto-created by ingest.py
├── ingest.py               ← Step 1: offline ingestion pipeline
├── retriever.py            ← Step 2: FAISS + cross-encoder retrieval
├── prompt.py               ← Step 3: prompt templates
├── chain.py                ← Step 4: LCEL RAG chain
├── api.py                  ← Step 5: FastAPI service
├── test_rag.py             ← Tests
├── pyproject.toml          ← Dependencies
├── .env.example            ← Copy to .env and add your key
└── .gitignore
```

## Setup

```bash
# 1. Clone and enter project
cd LLM-Based-Document-Q-A-System

# 2. Create virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # Mac/Linux

# 3. Install dependencies
pip install -e .

# 4. Configure API key
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY

# 5. Add PDFs
# Place PDF files in the docs/ folder

# 6. Run ingestion (offline step - run once per document set)
python ingest.py

# 7. Start the API server
uvicorn api:app --reload --port 8000
```

## Usage

### Standard endpoint
```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the payment terms?"}'
```

### Streaming endpoint
```bash
curl -X POST http://localhost:8000/ask/stream \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the late payment fee?"}'
```

### Swagger UI
Open http://localhost:8000/docs in your browser.

## Run Tests

```bash
# Unit tests only (no API key or FAISS index needed)
pytest test_rag.py -v -k "not api_client"

# All tests (requires API key + python ingest.py run first)
pytest test_rag.py -v

# With coverage
pytest test_rag.py --cov=. --cov-report=html
```

## Pipeline Flow

```
PDF files
    ↓ PyPDFLoader
Document objects (page_content + metadata)
    ↓ RecursiveCharacterTextSplitter (chunk_size=500, overlap=50)
Chunks
    ↓ OpenAIEmbeddings (text-embedding-3-small)
float32[1536] vectors
    ↓ FAISS.from_documents()
FAISS index (saved to faiss_index/)

--- At query time ---

User question
    ↓ OpenAIEmbeddings (same model)
Query vector
    ↓ FAISS similarity search (top-6)
Candidate chunks
    ↓ CrossEncoderReranker (top-3)
Re-ranked chunks
    ↓ ChatPromptTemplate
Filled prompt [system + human messages]
    ↓ ChatOpenAI (gpt-4o-mini, temperature=0)
Answer with citations
    ↓ StrOutputParser
String response → HTTP response
```