"""
state.py

Centralized session state management for the Hybrid RAG application.
Handles conversation history, active chat tracking, document registry,
pipeline state, and user settings — all via st.session_state.
"""

import uuid
from datetime import datetime
from typing import Any

import streamlit as st


# ---------------------------------------------------------------------------
# Default settings
# ---------------------------------------------------------------------------
DEFAULT_SETTINGS: dict[str, Any] = {
    "chunk_size": 1000,
    "chunk_overlap": 150,
    "top_k": 4,
    "hybrid_weight": 0.6,       # semantic weight (keyword = 1 - this)
    "temperature": 0.2,
    "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
    "llm_provider": "openrouter",  # "openrouter" or "groq"
    "llm_model_openrouter": "deepseek/deepseek-chat-v3-0324:free",
    "llm_model_groq": "llama-3.3-70b-versatile",
    "mmr_enabled": False,
}


# ---------------------------------------------------------------------------
# Conversation helpers
# ---------------------------------------------------------------------------
def _new_conversation() -> dict:
    """Create a blank conversation dict."""
    now = datetime.now().isoformat(timespec="seconds")
    return {
        "id": str(uuid.uuid4()),
        "title": "New Chat",
        "messages": [],          # [{role, content, result?}]
        "created_at": now,
        "updated_at": now,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def init_state() -> None:
    """Initialize all session-state keys if they don't exist yet."""
    defaults = {
        "conversations": [],                # list of conversation dicts
        "active_conversation_id": None,     # uuid str
        "indexed_documents": [],            # [{name, file_type, chunk_count, size_bytes}]
        "pipeline_app": None,               # compiled LangGraph app
        "pipeline_ready": False,
        "settings": {**DEFAULT_SETTINGS},
        "current_page": "chat",             # "chat" or "documents"
        "total_chunks": 0,
        "last_indexed": None,
        "upload_expanded": True,            # upload panel visibility
        "chunks_cache": None,               # cached chunk list for BM25
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

    # Ensure at least one conversation exists
    if not st.session_state["conversations"]:
        conv = _new_conversation()
        st.session_state["conversations"].append(conv)
        st.session_state["active_conversation_id"] = conv["id"]


def get_active_conversation() -> dict | None:
    """Return the currently active conversation dict."""
    cid = st.session_state.get("active_conversation_id")
    if cid is None:
        return None
    for conv in st.session_state["conversations"]:
        if conv["id"] == cid:
            return conv
    return None


def create_new_conversation() -> str:
    """Create a new conversation, set it active, return its id."""
    conv = _new_conversation()
    st.session_state["conversations"].insert(0, conv)
    st.session_state["active_conversation_id"] = conv["id"]
    return conv["id"]


def switch_conversation(conversation_id: str) -> None:
    """Switch the active conversation."""
    st.session_state["active_conversation_id"] = conversation_id


def add_message(role: str, content: str, result: dict | None = None) -> None:
    """Append a message to the active conversation."""
    conv = get_active_conversation()
    if conv is None:
        return
    conv["messages"].append({
        "role": role,
        "content": content,
        "result": result,
    })
    conv["updated_at"] = datetime.now().isoformat(timespec="seconds")

    # Auto-title from first user message
    if role == "user" and len(conv["messages"]) == 1:
        conv["title"] = content[:50] + ("…" if len(content) > 50 else "")


def delete_conversation(conversation_id: str) -> None:
    """Delete a conversation by id. If it was active, switch to another."""
    st.session_state["conversations"] = [
        c for c in st.session_state["conversations"]
        if c["id"] != conversation_id
    ]
    # If we deleted the active one, switch or create new
    if st.session_state["active_conversation_id"] == conversation_id:
        if st.session_state["conversations"]:
            st.session_state["active_conversation_id"] = \
                st.session_state["conversations"][0]["id"]
        else:
            create_new_conversation()


def get_settings() -> dict:
    """Return the current settings dict."""
    return st.session_state.get("settings", {**DEFAULT_SETTINGS})


def update_setting(key: str, value: Any) -> None:
    """Update a single setting."""
    st.session_state["settings"][key] = value
