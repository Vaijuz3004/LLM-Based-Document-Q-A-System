"""
retriever.py — Vector Retrieval & Re-ranking
=============================================
Loads the saved FAISS index and retrieves the most relevant
document chunks for a given query using a two-stage approach:

  Stage 1 (Bi-encoder): Fast vector similarity via FAISS
  Stage 2 (Cross-encoder): Accurate re-ranking of top candidates

CONCEPT: Why Two Stages?
─────────────────────────
You have 1000 chunks. You can't run a cross-encoder on all 1000
(too slow). You can't rely on just FAISS (less accurate).
Solution: use FAISS to narrow to 6 candidates fast, then
cross-encoder to re-rank those 6 accurately and return top 3.
"""

from dotenv import load_dotenv
load_dotenv()

from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

# ─── STAGE 1: Load FAISS + Bi-encoder Retriever ───────────────────────────────
def load_vectorstore(index_path : str = "./faiss_index"):

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    vectorstore = FAISS.load_local(folder_path=index_path, embeddings=embeddings, allow_dangerous_deserialization=True)
    print("@@@@@@@@@@@", vectorstore)
    return vectorstore

def get_base_retriever(vectorstore : FAISS, top_k : int = 6):
    return vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": top_k})

# ─── STAGE 2: Cross-encoder Re-ranker ────────────────────────────────────────
def build_reranker(top_n: int = 3):
    model = HuggingFaceCrossEncoder(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
    return CrossEncoderReranker(model=model, top_n=top_n)

# ─── COMBINED: Full Two-Stage Retriever ───────────────────────────────────────
def build_retriever(index_path : str = "./faiss_index", fetch_k : int = 6, top_n : int = 3):
    vectorstore = load_vectorstore(index_path)
    base_retriever = get_base_retriever(vectorstore, top_k = fetch_k)
    reranker = build_reranker(top_n=top_n)

    retriever = ContextualCompressionRetriever(base_compressor=reranker, base_retriever=base_retriever)
    return retriever

# ─── Utility: Quick Similarity Search (for debugging) ─────────────────────────
def quick_search(query: str, index_path: str = "./faiss_index", k: int = 3) -> None:
    """
    Run a quick similarity search and print results.
    Useful for debugging — run directly to test retrieval quality.

    Usage:
        python retriever.py
    """
    vectorstore = load_vectorstore(index_path)
    results = vectorstore.similarity_search_with_score(query, k=k)

    print(f"\nQuery: '{query}'")
    print(f"Top {k} results by cosine similarity:\n")

    for i, (doc, score) in enumerate(results):
        # FAISS returns L2 distance — lower = more similar
        # Convert to similarity: sim = 1 / (1 + score)
        similarity = round(1 / (1 + score), 3)
        print(f"[{i+1}] Score: {similarity} | "
              f"Source: {doc.metadata.get('source','?')}, "
              f"Page: {doc.metadata.get('page','?')}")
        print(f"     {doc.page_content[:120]}...")
        print()

if __name__ == "__main__":
    # quick_search("What are the payment terms?")
    quick_search("what is total contract value")