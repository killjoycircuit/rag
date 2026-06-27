# """
# engine.py

# Backend orchestration layer for the Hybrid RAG application.
# Wraps the existing pipeline modules (data_loader, text_splitter,
# vectorstore, search, graph) with caching and structured result
# formatting for the UI.

# Supports OpenRouter as primary LLM with Groq as fallback.
# OpenRouter provides free access to models like DeepSeek, Llama, Mistral, etc.
# """

# import os
# import time
# import logging
# from pathlib import Path
# from typing import Any

# import streamlit as st
# from dotenv import load_dotenv

# from pipeline.data_loader import load_documents
# from pipeline.text_splitter import split_documents
# from pipeline.vectorstore import get_or_build_vectorstore
# from pipeline.search import get_hybrid_retriever
# from pipeline.graph import build_graph, run_query



# load_dotenv()

# logger = logging.getLogger(__name__)

# # ---------------------------------------------------------------------------
# # Constants
# # ---------------------------------------------------------------------------
# DATA_DIR = Path(os.getenv("DATA_DIR", "./data"))
# DATA_DIR.mkdir(parents=True, exist_ok=True)

# SUPPORTED_EXTENSIONS = ["pdf", "csv", "xlsx", "xls", "txt", "docx", "png", "jpg", "jpeg", "md"]

# # Available models per provider
# OPENROUTER_MODELS = [
#     "deepseek/deepseek-chat-v3-0324:free",
#     "meta-llama/llama-4-maverick:free",
#     "qwen/qwen3-235b-a22b:free",
#     "mistralai/mistral-small-3.1-24b-instruct:free",
#     "google/gemma-3-27b-it:free",
#     "microsoft/phi-4-reasoning-plus:free",
#     "nousresearch/deephermes-3-llama-3-8b-preview:free",
# ]

# GROQ_MODELS = [
#     "llama-3.3-70b-versatile",
#     "llama-3.1-8b-instant",
#     "mixtral-8x7b-32768",
#     "gemma2-9b-it",
# ]


# # ---------------------------------------------------------------------------
# # LLM provider setup — OpenRouter primary, Groq fallback
# # ---------------------------------------------------------------------------
# def _get_llm_for_provider(provider: str, model: str, temperature: float):
#     """
#     Build a LangChain-compatible LLM instance for the given provider.
#     OpenRouter uses the OpenAI-compatible API format.

#     Args:
#         provider: 'openrouter' or 'groq'
#         model: Model identifier string
#         temperature: Sampling temperature

#     Returns:
#         A LangChain chat model instance.
#     """
#     if provider == "openrouter":
#         try:
#             # from langchain_openai import ChatOpenAI
#             api_key = os.getenv("OPENROUTER_API_KEY")
#             if not api_key:
#                 raise EnvironmentError("OPENROUTER_API_KEY not set")
#             return ChatOpenAI(
#                 model=model,
#                 temperature=temperature,
#                 api_key=api_key,
#                 base_url="https://openrouter.ai/api/v1",
#                 default_headers={
#                     "HTTP-Referer": "http://localhost:8501",
#                     "X-Title": "Hybrid RAG",
#                 },
#             )
#         except Exception as e:
#             logger.warning(f"OpenRouter init failed ({e}), falling back to Groq")
#             return _get_llm_for_provider("groq", "llama-3.3-70b-versatile", temperature)
#     else:
#         from langchain_groq import ChatGroq
#         api_key = os.getenv("GROQ_API_KEY")
#         if not api_key:
#             raise EnvironmentError("GROQ_API_KEY not set")
#         return ChatGroq(
#             model=model,
#             temperature=temperature,
#             api_key=api_key,
#         )


# def get_configured_llm(settings: dict):
#     """Build the LLM from current user settings, with automatic fallback."""
#     provider = settings.get("llm_provider", "openrouter")
#     if provider == "openrouter":
#         model = settings.get("llm_model_openrouter", "deepseek/deepseek-chat-v3-0324:free")
#     else:
#         model = settings.get("llm_model_groq", "llama-3.3-70b-versatile")
#     temperature = settings.get("temperature", 0.2)
#     return _get_llm_for_provider(provider, model, temperature)


# # ---------------------------------------------------------------------------
# # File management
# # ---------------------------------------------------------------------------
# def save_uploaded_files(uploaded_files) -> list[dict]:
#     """
#     Persist uploaded files to DATA_DIR and return file metadata.

#     Returns:
#         List of dicts with name, size_bytes, file_type keys.
#     """
#     file_infos = []
#     for uf in uploaded_files:
#         dest = DATA_DIR / uf.name
#         data = uf.getbuffer()
#         with open(dest, "wb") as f:
#             f.write(data)
#         file_infos.append({
#             "name": uf.name,
#             "size_bytes": len(data),
#             "file_type": Path(uf.name).suffix.lstrip(".").lower(),
#         })
#     return file_infos


# def delete_document(filename: str) -> bool:
#     """Delete a document from DATA_DIR by filename."""
#     target = DATA_DIR / filename
#     if target.exists():
#         target.unlink()
#         return True
#     return False


# # ---------------------------------------------------------------------------
# # Pipeline orchestration
# # ---------------------------------------------------------------------------
# def index_documents(uploaded_files, settings: dict) -> list[dict]:
#     """
#     Full ingestion pipeline: save → load → split → index → build graph.

#     Rebuilds the full index from all files in DATA_DIR. Uses settings
#     for chunk_size, chunk_overlap, hybrid_weight, and top_k.

#     Returns:
#         List of document summary dicts [{name, file_type, chunk_count, size_bytes}].
#     """
#     # Save new files
#     file_infos = save_uploaded_files(uploaded_files)

#     # Load all documents from DATA_DIR
#     raw_docs = load_documents(str(DATA_DIR))

#     # Split with user settings
#     chunk_size = settings.get("chunk_size", 1000)
#     chunk_overlap = settings.get("chunk_overlap", 150)
#     chunks = split_documents(raw_docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

#     # Build vectorstore
#     vectorstore = get_or_build_vectorstore(chunks, force_rebuild=True)

#     # Build hybrid retriever with user weights
#     semantic_weight = settings.get("hybrid_weight", 0.6)
#     keyword_weight = round(1.0 - semantic_weight, 2)
#     top_k = settings.get("top_k", 4)
#     retriever = get_hybrid_retriever(
#         chunks, vectorstore,
#         semantic_weight=semantic_weight,
#         keyword_weight=keyword_weight,
#         top_k=top_k,
#     )

#     # Build LangGraph
#     app = build_graph(retriever)

#     # Store in session
#     st.session_state["pipeline_app"] = app
#     st.session_state["pipeline_ready"] = True
#     st.session_state["chunks_cache"] = chunks
#     st.session_state["total_chunks"] = len(chunks)
#     st.session_state["last_indexed"] = time.strftime("%Y-%m-%d %H:%M:%S")

#     # Build per-document summaries
#     summary_by_source: dict[str, dict] = {}
#     for chunk in chunks:
#         source = chunk.metadata.get("source", "unknown")
#         file_type = chunk.metadata.get("file_type", "unknown")
#         name = Path(source).name
#         if name not in summary_by_source:
#             size = 0
#             for fi in file_infos:
#                 if fi["name"] == name:
#                     size = fi["size_bytes"]
#                     break
#             if size == 0:
#                 disk_path = DATA_DIR / name
#                 if disk_path.exists():
#                     size = disk_path.stat().st_size
#             summary_by_source[name] = {
#                 "name": name,
#                 "file_type": file_type,
#                 "chunk_count": 0,
#                 "size_bytes": size,
#             }
#         summary_by_source[name]["chunk_count"] += 1

#     return list(summary_by_source.values())


# def rebuild_pipeline_from_disk(settings: dict) -> list[dict] | None:
#     """
#     Rebuild the pipeline from existing files in DATA_DIR (no new upload).
#     Returns document summaries or None if DATA_DIR is empty.
#     """
#     files_on_disk = [f for f in DATA_DIR.iterdir() if f.is_file()]
#     if not files_on_disk:
#         return None

#     raw_docs = load_documents(str(DATA_DIR))
#     if not raw_docs:
#         return None

#     chunk_size = settings.get("chunk_size", 1000)
#     chunk_overlap = settings.get("chunk_overlap", 150)
#     chunks = split_documents(raw_docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

#     vectorstore = get_or_build_vectorstore(chunks, force_rebuild=False)

#     semantic_weight = settings.get("hybrid_weight", 0.6)
#     keyword_weight = round(1.0 - semantic_weight, 2)
#     top_k = settings.get("top_k", 4)
#     retriever = get_hybrid_retriever(
#         chunks, vectorstore,
#         semantic_weight=semantic_weight,
#         keyword_weight=keyword_weight,
#         top_k=top_k,
#     )

#     app = build_graph(retriever)

#     st.session_state["pipeline_app"] = app
#     st.session_state["pipeline_ready"] = True
#     st.session_state["chunks_cache"] = chunks
#     st.session_state["total_chunks"] = len(chunks)
#     st.session_state["last_indexed"] = time.strftime("%Y-%m-%d %H:%M:%S")

#     summary_by_source: dict[str, dict] = {}
#     for chunk in chunks:
#         source = chunk.metadata.get("source", "unknown")
#         file_type = chunk.metadata.get("file_type", "unknown")
#         name = Path(source).name
#         if name not in summary_by_source:
#             size = 0
#             disk_path = DATA_DIR / name
#             if disk_path.exists():
#                 size = disk_path.stat().st_size
#             summary_by_source[name] = {
#                 "name": name,
#                 "file_type": file_type,
#                 "chunk_count": 0,
#                 "size_bytes": size,
#             }
#         summary_by_source[name]["chunk_count"] += 1

#     return list(summary_by_source.values())


# # ---------------------------------------------------------------------------
# # Query execution
# # ---------------------------------------------------------------------------
# def estimate_confidence(documents: list, is_relevant: bool) -> float:
#     """
#     Heuristic confidence score for display purposes.
#     Not a calibrated probability — just a UI indicator.
#     """
#     if not documents:
#         return 0.0
#     base = 0.55 if is_relevant else 0.15
#     chunk_bonus = min(len(documents) / 8, 0.35)
#     return round(min(base + chunk_bonus, 0.97), 2)


# def query(question: str) -> dict:
#     """
#     Run a question through the pipeline and return a structured result
#     for the UI including answer, sources, chunks, confidence, and timings.
#     """
#     app = st.session_state.get("pipeline_app")
#     if app is None:
#         return {
#             "answer": "Please upload and index documents first.",
#             "sources": [],
#             "chunks": [],
#             "confidence": 0.0,
#             "timing_s": 0.0,
#             "retry_count": 0,
#             "is_relevant": False,
#         }

#     # Run the graph
#     result = run_query(app, question)

#     documents = result.get("documents", [])
#     is_relevant = result.get("is_relevant", False)

#     # Build source list
#     sources = sorted({
#         Path(doc.metadata.get("source", "unknown")).name
#         for doc in documents
#     })

#     # Build chunk display data
#     chunks_display = []
#     for i, doc in enumerate(documents):
#         chunks_display.append({
#             "rank": i + 1,
#             "source": Path(doc.metadata.get("source", "unknown")).name,
#             "file_type": doc.metadata.get("file_type", "unknown"),
#             "chunk_index": doc.metadata.get("chunk_index", 0),
#             "content": doc.page_content,
#         })

#     return {
#         "answer": result.get("answer", ""),
#         "sources": sources,
#         "chunks": chunks_display,
#         "confidence": estimate_confidence(documents, is_relevant),
#         "timing_s": result.get("timing_s", 0.0),
#         "retry_count": result.get("retry_count", 0),
#         "is_relevant": is_relevant,
#     }


# def format_file_size(size_bytes: int) -> str:
#     """Format byte count as human-readable string."""
#     if size_bytes < 1024:
#         return f"{size_bytes} B"
#     elif size_bytes < 1024 * 1024:
#         return f"{size_bytes / 1024:.1f} KB"
#     else:
#         return f"{size_bytes / (1024 * 1024):.1f} MB"
