"""
main.py

End-to-end runner for the Hybrid RAG pipeline, now using the full LangGraph
flow: load -> chunk -> index -> retrieve -> grade relevance -> generate
(with automatic query rewrite + retry on poor retrieval).

Run:
    python main.py
"""

import os
import logging
from dotenv import load_dotenv

# Centralized logging — suppress noisy HTTP logs from HuggingFace/httpx
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
for noisy_logger in ("httpx", "httpcore", "urllib3", "huggingface_hub"):
    logging.getLogger(noisy_logger).setLevel(logging.WARNING)

from pipeline.data_loader import load_documents
from pipeline.text_splitter import split_documents
from pipeline.vectorstore import get_or_build_vectorstore
from pipeline.search import get_hybrid_retriever
from pipeline.graph import build_graph, run_query

load_dotenv()

DATA_DIR = os.getenv("DATA_DIR", "./data")


def build_pipeline():
    """Load docs, chunk them, build/load the vectorstore, and return the compiled graph app."""
    print(f"\n[1/5] Loading documents from '{DATA_DIR}'...")
    raw_docs = load_documents(DATA_DIR)

    if not raw_docs:
        raise RuntimeError(
            f"No documents found in '{DATA_DIR}'. "
            f"Add some PDFs/CSV/DOCX/images/TXT files there first."
        )

    print(f"[2/5] Splitting {len(raw_docs)} document(s) into chunks...")
    chunks = split_documents(raw_docs)

    print(f"[3/5] Building/loading FAISS vector index for {len(chunks)} chunk(s)...")
    vectorstore = get_or_build_vectorstore(chunks)

    print("[4/5] Building hybrid retriever (FAISS + BM25)...")
    retriever = get_hybrid_retriever(chunks, vectorstore)

    print("[5/5] Compiling LangGraph RAG flow...")
    app = build_graph(retriever)

    print("\nPipeline ready.\n")
    return app


def main():
    app = build_pipeline()

    print("Ask a question about your documents (or type 'exit' to quit).\n")

    while True:
        question = input("Question: ").strip()
        if question.lower() in ("exit", "quit"):
            break
        if not question:
            continue

        result = run_query(app, question)
        answer = result["answer"] if isinstance(result, dict) else result

        print(f"\nAnswer:\n{answer}\n")
        if isinstance(result, dict):
            print(f"  Confidence: {'relevant' if result.get('is_relevant') else 'low'}")
            print(f"  Time: {result.get('timing_s', '?')}s")
            print(f"  Sources: {len(result.get('documents', []))} chunk(s)")
        print("-" * 60 + "\n")


if __name__ == "__main__":
    main()