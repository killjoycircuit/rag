"""
text_splitter.py

Splits loaded Documents (from data_loader.py) into smaller overlapping chunks
for embedding + retrieval in the Hybrid RAG pipeline.

Uses LangChain's RecursiveCharacterTextSplitter, which tries to split on
natural boundaries first (paragraphs -> sentences -> words) before falling
back to a hard character cut.

Usage:
    from data_loader import load_documents
    from text_splitter import split_documents

    docs = load_documents("./data")
    chunks = split_documents(docs)
"""

import logging
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_CHUNK_SIZE = 1000       # characters per chunk
DEFAULT_CHUNK_OVERLAP = 150     # ~15% overlap, keeps context across chunk boundaries

# File types that are already small/atomic and shouldn't be split further.
# e.g. a single CSV row or a short image-OCR snippet rarely benefits from splitting,
# and splitting it can actually break meaning.
NO_SPLIT_FILE_TYPES = {"csv"}


def get_text_splitter(
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> RecursiveCharacterTextSplitter:
    """
    Build a RecursiveCharacterTextSplitter with sensible separators.

    Tries to split on paragraph breaks first, then lines, then sentences,
    then words, then hard character cut as last resort.
    """
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )


def split_documents(
    documents: List[Document],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> List[Document]:
    """
    Split a list of Documents into smaller chunks.

    - Documents whose file_type is in NO_SPLIT_FILE_TYPES are kept as-is
      (e.g. CSV rows are already atomic units).
    - All other Documents are run through the recursive splitter.
    - Metadata (source, file_type) is preserved on every chunk, and a
      chunk_index is added so chunks can be traced back to their position
      within the original document.

    Args:
        documents: List of Document objects from data_loader.load_documents()
        chunk_size: Max characters per chunk.
        chunk_overlap: Overlapping characters between consecutive chunks.

    Returns:
        List of chunked Document objects, ready for embedding.
    """
    if not documents:
        logger.warning("No documents provided to split_documents().")
        return []

    splitter = get_text_splitter(chunk_size, chunk_overlap)

    all_chunks: List[Document] = []
    skipped_count = 0

    for doc in documents:
        file_type = doc.metadata.get("file_type", "unknown")

        # Skip splitting for already-atomic content types
        if file_type in NO_SPLIT_FILE_TYPES:
            skipped_count += 1
            doc.metadata["chunk_index"] = 0
            all_chunks.append(doc)
            continue

        # Skip splitting if content is already shorter than chunk_size
        if len(doc.page_content) <= chunk_size:
            doc.metadata["chunk_index"] = 0
            all_chunks.append(doc)
            continue

        chunks = splitter.split_text(doc.page_content)

        for idx, chunk_text in enumerate(chunks):
            chunk_doc = Document(
                page_content=chunk_text,
                metadata={
                    **doc.metadata,
                    "chunk_index": idx,
                },
            )
            all_chunks.append(chunk_doc)

    logger.info(
        f"Split {len(documents)} document(s) into {len(all_chunks)} chunk(s) "
        f"(chunk_size={chunk_size}, overlap={chunk_overlap}, "
        f"{skipped_count} doc(s) left unsplit by type)."
    )

    return all_chunks


# ---------------------------------------------------------------------------
# Manual test run
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import os
    from pipeline.data_loader import load_documents

    DATA_DIR = os.getenv("DATA_DIR", "./data")
    raw_docs = load_documents(DATA_DIR)
    chunks = split_documents(raw_docs)

    for i, chunk in enumerate(chunks[:5]):
        print(f"\n--- Chunk {i+1} ---")
        print(f"Source: {chunk.metadata.get('source')}")
        print(f"Type: {chunk.metadata.get('file_type')}")
        print(f"Chunk index: {chunk.metadata.get('chunk_index')}")
        print(f"Length: {len(chunk.page_content)} chars")
        print(f"Preview: {chunk.page_content[:200]}")