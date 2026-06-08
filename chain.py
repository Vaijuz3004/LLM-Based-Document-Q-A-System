"""
chain.py — LangChain RAG Chain (LCEL)
======================================
Wires together retriever + prompt + LLM into a single
callable pipeline using LangChain Expression Language (LCEL).

CONCEPT: What is LCEL?
───────────────────────
LangChain Expression Language is a declarative way to compose
LangChain components using the pipe | operator (like Unix pipes).

Each component is a "Runnable" — it has:
    .invoke(input)          → returns output (blocking)
    .stream(input)          → yields output chunks (streaming)
    .batch([input1, input2]) → processes multiple inputs

The | operator chains Runnables:
    A | B means: output of A becomes input of B

INTERVIEW Q: What is LCEL and what problem does it solve vs legacy LangChain?
INTERVIEW Q: What is the difference between invoke(), stream(), and batch()?
INTERVIEW Q: What is a Runnable in LangChain?
"""

from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

from retriever import build_retriever
from prompt import get_rag_prompt


# ─── Helper: Format Retrieved Docs for the Prompt ─────────────────────────────
def format_docs(docs: list[Document]) -> str:
    """
    Convert a list of Document objects into a formatted string
    that gets injected into the {context} slot of the prompt.

    CONCEPT: Why Format Docs?
    ──────────────────────────
    The retriever returns Document objects. The prompt template
    expects a plain string for {context}.

    We include the source metadata (filename + page) in the formatted
    string so the LLM can cite them in its answer.

    Example output:
    ───────────────
    [Source: contract.pdf, page 1]
    Payment terms: all invoices are due net-30 from the date of issue.
    Late payments incur a 1.5% monthly fee.

    [Source: contract.pdf, page 2]
    The company reserves the right to suspend services for accounts
    overdue by more than 60 days.

    INTERVIEW Q: Why do we inject metadata into the context string?
    INTERVIEW Q: What happens to retrieval quality if metadata is missing?
    """
    if not docs:
        return "No relevant documents found."

    formatted_chunks = []
    for doc in docs:
        source   = doc.metadata.get("source", "unknown")
        page     = doc.metadata.get("page", "?")
        # Normalise the path — show just the filename, not the full path
        filename = source.split("/")[-1].split("\\")[-1]

        chunk = (
            f"[Source: {filename}, page {page}]\n"
            f"{doc.page_content.strip()}"
        )
        formatted_chunks.append(chunk)

    return "\n\n".join(formatted_chunks)


# ─── Build the LCEL Chain ─────────────────────────────────────────────────────
def build_chain(index_path: str = "./faiss_index"):
    """
    Assemble the full RAG chain.

    CHAIN ANATOMY:
    ──────────────

        User question (string)
              │
              ▼
        ┌─────────────────────────────────┐
        │  RunnableParallel               │
        │  ┌───────────────┐              │
        │  │ "context":    │              │
        │  │ retriever     │ → format_docs│  ← fetches + formats chunks
        │  │    |          │              │
        │  │ format_docs   │              │
        │  └───────────────┘              │
        │  ┌───────────────┐              │
        │  │ "question":   │              │
        │  │ Passthrough() │              │  ← passes question unchanged
        │  └───────────────┘              │
        └─────────────────────────────────┘
              │
              ▼  {"context": "...", "question": "..."}
        ChatPromptTemplate          ← fills in {context} and {question}
              │
              ▼  [SystemMessage, HumanMessage]
        ChatOpenAI (gpt-4o-mini)    ← generates the answer
              │
              ▼  AIMessage
        StrOutputParser             ← extracts .content string
              │
              ▼  "Answer with citations..."

    CONCEPT: RunnablePassthrough
    ──────────────────────────────
    Passes its input through unchanged. Used here so the question
    string flows through to fill the {question} slot in the prompt,
    while the retriever branch fills the {context} slot.

    CONCEPT: RunnableParallel
    ──────────────────────────
    Runs multiple branches in parallel with the same input.
    Both "context" branch and "question" branch receive the
    original user question as input simultaneously.

    CONCEPT: StrOutputParser
    ──────────────────────────
    ChatOpenAI returns an AIMessage object:
        AIMessage(content="The payment terms are...", ...)
    StrOutputParser extracts just the .content string.

    CONCEPT: temperature=0
    ───────────────────────
    Controls randomness in LLM generation:
        temperature=0.0  → always pick highest probability token
                          (deterministic, consistent, good for Q&A)
        temperature=0.7  → balanced creativity (good for writing)
        temperature=1.0  → maximum diversity (good for brainstorming)

    For document Q&A, you want deterministic answers — same question
    should always return the same answer.

    INTERVIEW Q: What does temperature control in an LLM?
    INTERVIEW Q: What is the difference between invoke() and stream()?
    INTERVIEW Q: Why use RunnableParallel instead of sequential steps?
    """
    retriever = build_retriever(index_path)
    prompt    = get_rag_prompt()
    llm       = ChatOpenAI(
        model="gpt-4o-mini",  # cheapest capable model
        temperature=0,        # deterministic for Q&A
        streaming=True        # enables .stream() token-by-token output
    )

    # Build the chain using LCEL pipe operator
    chain = (
        RunnableParallel({
            "context":  retriever | format_docs,  # retrieve → format
            "question": RunnablePassthrough()      # pass question through
        })
        | prompt          # fill {context} and {question} into template
        | llm             # generate answer (ChatOpenAI)
        | StrOutputParser()  # extract string from AIMessage
    )

    return chain


# ─── Convenience wrapper with source tracking ─────────────────────────────────
def build_chain_with_sources(index_path: str = "./faiss_index"):
    """
    Alternative chain that also returns the source documents.

    CONCEPT: Why Return Sources?
    ─────────────────────────────
    Returning source documents alongside the answer lets the API
    client display clickable citations or verify answers. This is
    the production pattern — don't just trust the LLM's in-text
    citations, also return the actual chunks programmatically.

    Returns a dict:
        {
            "answer":  "Payment is due net-30... [Source: ...]",
            "sources": [Document(...), Document(...)]
        }

    INTERVIEW Q: How do you implement source attribution in RAG?
    """
    retriever = build_retriever(index_path)
    prompt    = get_rag_prompt()
    llm       = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # Branch that retrieves docs AND passes them through to the output
    retrieve_docs = RunnableParallel({
        "docs":     retriever,
        "question": RunnablePassthrough()
    })

    def make_prompt_input(data: dict) -> dict:
        return {
            "context":  format_docs(data["docs"]),
            "question": data["question"]
        }

    answer_chain = (
        make_prompt_input
        | prompt
        | llm
        | StrOutputParser()
    )

    # Final chain returns both answer and source documents
    full_chain = retrieve_docs | RunnableParallel({
        "answer":  answer_chain,
        "sources": lambda x: x["docs"]
    })

    return full_chain


# ─── Quick test ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Building chain (loads FAISS index + cross-encoder model)...")
    chain = build_chain()

    questions = [
        "what is total contract value?",
        "What is the late payment fee?",
        "What encryption is used for data protection?",
        "What is the CEO's favourite colour?",  # unanswerable test
    ]

    for q in questions:
        print(f"\n{'='*55}")
        print(f"Q: {q}")
        print(f"A: ", end="", flush=True)

        # Stream token by token
        for chunk in chain.stream(q):
            print(chunk, end="", flush=True)
        print()
