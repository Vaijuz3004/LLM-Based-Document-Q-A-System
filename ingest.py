from dotenv import load_dotenv
load_dotenv()
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

import os


def ingest_documents(docs_dir : str = "./docs", index_path : str = "./faiss_index"):

    #1. Load pdfs
    loader = DirectoryLoader(docs_dir, glob="**/*.pdf", show_progress=True, loader_cls=PyPDFLoader)
    raw_docs = loader.load()
    # print(f"Loaded {len(raw_docs)} documents")
    # print(raw_docs[0].page_content)

    #2. Split into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=50, separators=["\n\n", "\n", " ", ""])
    chunks = splitter.split_documents(raw_docs)
    print(f"Split into {len(chunks)} chunks")

    #3. Embed + Index -- one API call batches all chunks
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = FAISS.from_documents(chunks, embeddings)

    #4. Persist -- Save to disk so you don't have to re-embed every time
    vectorstore.save_local(index_path)
    print(f"Saved index to {index_path}")



if __name__ == "__main__":
    ingest_documents()

