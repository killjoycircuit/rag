"""
llm.py

Sets up the Groq-hosted LLM used for answer generation in the Hybrid RAG
pipeline, plus a helper to generate answers grounded in retrieved chunks.

Model: openai/gpt-oss-20b (fast, lightweight, free-tier friendly on Groq)
Note: Groq's model lineup changes over time. If this model is ever
deprecated, check https://console.groq.com/docs/models for the current
recommended replacement and update DEFAULT_MODEL below.

Usage:
    from search import get_hybrid_retriever, search
    from llm import generate_answer

    results = search(hybrid_retriever, "your question")
    answer = generate_answer("your question", results)
    print(answer)
"""

import os
import logging
from typing import List, Optional

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DEFAULT_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
DEFAULT_TEMPERATURE = float(os.getenv("GROQ_TEMPERATURE", 0.2))

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    logger.warning(
        "GROQ_API_KEY not found in environment. Set it in your .env file "
        "before calling the LLM (get a free key at https://console.groq.com/keys)."
    )

# ---------------------------------------------------------------------------
# RAG prompt template
# ---------------------------------------------------------------------------
RAG_PROMPT_TEMPLATE = """You are a helpful assistant answering questions using ONLY the context provided below.

Rules:
- Answer using only the information in the context. Do not use outside knowledge.
- If the context doesn't contain enough information to answer, say so clearly — don't guess.
- Cite the source file name(s) you used when possible.
- Be concise and direct.

Context:
{context}

Question: {question}

Answer:"""


def get_llm(
    model: str = DEFAULT_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
) -> ChatGroq:
    """
    Build and return a ChatGroq LLM instance.

    Args:
        model: Groq model ID (e.g. "openai/gpt-oss-20b").
        temperature: Sampling temperature. Lower = more deterministic,
            better for factual RAG answers.

    Returns:
        A LangChain-compatible ChatGroq instance.
    """
    if not GROQ_API_KEY:
        raise EnvironmentError(
            "GROQ_API_KEY is not set. Add it to your .env file: "
            "GROQ_API_KEY=your_key_here"
        )

    logger.info(f"Initializing Groq LLM (model='{model}', temperature={temperature})...")

    llm = ChatGroq(
        model=model,
        temperature=temperature,
        api_key=GROQ_API_KEY,
    )

    logger.info("Groq LLM ready.")
    return llm


def _format_context(documents: List[Document]) -> str:
    """
    Format retrieved Documents into a single context string for the prompt,
    with source attribution per chunk.
    """
    if not documents:
        return "(no relevant context found)"

    formatted_chunks = []
    for i, doc in enumerate(documents, start=1):
        source = doc.metadata.get("source", "unknown source")
        formatted_chunks.append(f"[Chunk {i} | Source: {source}]\n{doc.page_content}")

    return "\n\n".join(formatted_chunks)


def generate_answer(
    question: str,
    retrieved_docs: List[Document],
    llm: Optional[ChatGroq] = None,
) -> str:
    """
    Generate a grounded answer to `question` using the retrieved chunks
    as context, via the Groq LLM.

    Args:
        question: The user's question.
        retrieved_docs: List of Document chunks from the hybrid retriever
            (search.py's search() function).
        llm: Optional pre-built ChatGroq instance. If None, builds one
            with default settings.

    Returns:
        The generated answer string.
    """
    model = llm or get_llm()

    context = _format_context(retrieved_docs)

    prompt = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)
    chain = prompt | model

    logger.info(f"Generating answer for question: '{question}'")
    response = chain.invoke({"context": context, "question": question})

    answer = response.content
    logger.info("Answer generated.")

    return answer


# ---------------------------------------------------------------------------
# Manual test run
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from pipeline.data_loader import load_documents
    from pipeline.text_splitter import split_documents
    from pipeline.vectorstore import get_or_build_vectorstore
    from pipeline.search import get_hybrid_retriever, search

    DATA_DIR = os.getenv("DATA_DIR", "./data")

    raw_docs = load_documents(DATA_DIR)
    chunks = split_documents(raw_docs)
    store = get_or_build_vectorstore(chunks)
    retriever = get_hybrid_retriever(chunks, store)

    test_question = "What is this document about?"
    found_docs = search(retriever, test_question)

    final_answer = generate_answer(test_question, found_docs)

    print(f"\nQuestion: {test_question}")
    print(f"\nAnswer:\n{final_answer}")