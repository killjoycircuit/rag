"""
vectorstore.py

Builds, saves, and loads the FAISS vector store used for the semantic
side of the Hybrid RAG pipeline.

Workflow:
    1. build_vectorstore(chunks) -> embeds chunks + creates a FAISS index, saves to disk
    2. load_vectorstore()        -> loads a previously saved FAISS index from disk
    3. get_or_build_vectorstore(chunks) -> loads if it exists, otherwise builds fresh

Usage:
    from data_loader import load_documents
    from text_splitter import split_documents
    from vectorstore import get_or_build_vectorstore

    docs = load_documents("./data")
    chunks = split_documents(docs)
    vectorstore = get_or_build_vectorstore(chunks)

    results = vectorstore.similarity_search("your query here", k=4)
"""

import os
import logging
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

from pipeline.embedding import get_embedding_model

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DEFAULT_INDEX_DIR = os.getenv("FAISS_INDEX_DIR", "./faiss_index")


def build_vectorstore(
    chunks: List[Document],
    index_dir: str = DEFAULT_INDEX_DIR,
    embedding_model=None,
) -> FAISS:
    """
    Embed chunks and build a new FAISS index, then save it to disk.

    Args:
        chunks: List of chunked Document objects (from text_splitter.py).
        index_dir: Directory path where the FAISS index will be saved.
        embedding_model: Optional pre-loaded embedding model. If None,
            loads the default via embedding.get_embedding_model().

    Returns:
        A FAISS vectorstore instance, ready for similarity search.
    """
    if not chunks:
        raise ValueError("Cannot build vectorstore: no chunks provided.")

    embeddings = embedding_model or get_embedding_model()

    logger.info(f"Building FAISS index from {len(chunks)} chunk(s)...")
    vectorstore = FAISS.from_documents(documents=chunks, embedding=embeddings)

    Path(index_dir).mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(index_dir)
    logger.info(f"FAISS index built and saved to '{index_dir}'.")

    return vectorstore


def load_vectorstore(
    index_dir: str = DEFAULT_INDEX_DIR,
    embedding_model=None,
) -> Optional[FAISS]:
    """
    Load a previously saved FAISS index from disk, if it exists.

    Args:
        index_dir: Directory path where the FAISS index was saved.
        embedding_model: Optional pre-loaded embedding model. Must match
            the model used to build the index originally.

    Returns:
        A FAISS vectorstore instance, or None if no saved index is found.
    """
    index_path = Path(index_dir)
    faiss_file = index_path / "index.faiss"

    if not faiss_file.exists():
        logger.info(f"No existing FAISS index found at '{index_dir}'.")
        return None

    embeddings = embedding_model or get_embedding_model()

    logger.info(f"Loading FAISS index from '{index_dir}'...")
    vectorstore = FAISS.load_local(
        index_dir,
        embeddings,
        allow_dangerous_deserialization=True,  # safe here: we created this file ourselves
    )
    logger.info("FAISS index loaded successfully.")

    return vectorstore


def get_or_build_vectorstore(
    chunks: Optional[List[Document]] = None,
    index_dir: str = DEFAULT_INDEX_DIR,
    embedding_model=None,
    force_rebuild: bool = False,
) -> FAISS:
    """
    Convenience function: loads the FAISS index if it already exists,
    otherwise builds it fresh from the given chunks.

    Args:
        chunks: Chunked Documents to use if building is needed.
            Required if no saved index exists yet, or if force_rebuild=True.
        index_dir: Directory path for the FAISS index.
        embedding_model: Optional pre-loaded embedding model.
        force_rebuild: If True, ignores any saved index and rebuilds from chunks.

    Returns:
        A ready-to-query FAISS vectorstore instance.
    """
    embeddings = embedding_model or get_embedding_model()

    if not force_rebuild:
        existing = load_vectorstore(index_dir, embedding_model=embeddings)
        if existing is not None:
            return existing

    if not chunks:
        raise ValueError(
            "No saved FAISS index found and no chunks provided to build one. "
            "Pass `chunks` (from text_splitter.split_documents) to build a new index."
        )

    return build_vectorstore(chunks, index_dir=index_dir, embedding_model=embeddings)


def add_documents_to_vectorstore(
    vectorstore: FAISS,
    new_chunks: List[Document],
    index_dir: str = DEFAULT_INDEX_DIR,
) -> FAISS:
    """
    Add new chunks to an existing FAISS index (incremental update) and
    re-save to disk. Useful when new files are added later without
    rebuilding the whole index from scratch.

    Args:
        vectorstore: The existing FAISS vectorstore instance.
        new_chunks: New chunked Documents to add.
        index_dir: Directory path to re-save the updated index.

    Returns:
        The updated FAISS vectorstore instance.
    """
    if not new_chunks:
        logger.warning("No new chunks provided — nothing added.")
        return vectorstore

    logger.info(f"Adding {len(new_chunks)} new chunk(s) to existing FAISS index...")
    vectorstore.add_documents(new_chunks)

    vectorstore.save_local(index_dir)
    logger.info(f"Updated FAISS index saved to '{index_dir}'.")

    return vectorstore


# ---------------------------------------------------------------------------
# Manual test run
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from pipeline.data_loader import load_documents
    from pipeline.text_splitter import split_documents

    DATA_DIR = os.getenv("DATA_DIR", "./data")

    raw_docs = load_documents(DATA_DIR)
    doc_chunks = split_documents(raw_docs)

    store = get_or_build_vectorstore(doc_chunks)

    query = "test query"
    results = store.similarity_search(query, k=3)

    print(f"\nTop {len(results)} result(s) for query: '{query}'")
    for i, res in enumerate(results):
        print(f"\n--- Result {i+1} ---")
        print(f"Source: {res.metadata.get('source')}")
        print(f"Content preview: {res.page_content[:200]}")