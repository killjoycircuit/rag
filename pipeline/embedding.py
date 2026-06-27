"""
embedding.py

Provides the embedding model used to convert text chunks into vectors
for the semantic (FAISS) side of the Hybrid RAG pipeline.

Model: sentence-transformers/all-MiniLM-L6-v2
  - Fast, lightweight, 384-dim embeddings, strong general-purpose default.

Device: auto-detects CUDA if available, falls back to CPU.
You can override this with the EMBEDDING_DEVICE env var ("cpu" or "cuda").

Usage:
    from embedding import get_embedding_model

    embeddings = get_embedding_model()
    vector = embeddings.embed_query("some text")
    vectors = embeddings.embed_documents(["text 1", "text 2"])
"""

import os
import logging

from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def _detect_device() -> str:
    """
    Decide whether to run embeddings on GPU (CUDA) or CPU.

    Priority:
      1. EMBEDDING_DEVICE env var, if set ("cpu" or "cuda")
      2. Auto-detect via torch.cuda.is_available()
      3. Fallback to "cpu"
    """
    override = os.getenv("EMBEDDING_DEVICE")
    if override in ("cpu", "cuda"):
        logger.info(f"Using device override from .env: {override}")
        return override

    try:
        import torch

        if torch.cuda.is_available():
            logger.info("CUDA GPU detected — using GPU for embeddings.")
            return "cuda"
    except ImportError:
        pass

    logger.info("No GPU detected (or torch not installed) — using CPU.")
    return "cpu"


def get_embedding_model(
    model_name: str = DEFAULT_MODEL_NAME,
    device: str = None,
    normalize_embeddings: bool = True,
) -> HuggingFaceEmbeddings:
    """
    Load and return a HuggingFace embedding model wrapped for LangChain.

    Args:
        model_name: HuggingFace model repo name.
        device: "cpu" or "cuda". If None, auto-detects.
        normalize_embeddings: L2-normalize vectors — recommended for
            cosine-similarity search in FAISS.

    Returns:
        A LangChain-compatible HuggingFaceEmbeddings instance.
    """
    resolved_device = device or _detect_device()

    logger.info(f"Loading embedding model '{model_name}' on device='{resolved_device}'...")

    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": resolved_device},
        encode_kwargs={"normalize_embeddings": normalize_embeddings},
    )

    logger.info("Embedding model loaded successfully.")
    return embeddings


# ---------------------------------------------------------------------------
# Manual test run
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    embedding_model = get_embedding_model()

    sample_texts = [
        "LangChain helps build LLM-powered applications.",
        "FAISS is a library for efficient vector similarity search.",
    ]

    vectors = embedding_model.embed_documents(sample_texts)

    print(f"\nGenerated {len(vectors)} embedding(s).")
    print(f"Embedding dimension: {len(vectors[0])}")
    print(f"First 5 values of vector 1: {vectors[0][:5]}")