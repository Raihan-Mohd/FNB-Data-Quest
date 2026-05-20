"""Cached markdown content loader for the Research page."""
from pathlib import Path
import streamlit as st

CONTENT_DIR = Path(__file__).resolve().parent.parent / "content"


@st.cache_data
def load_markdown(name: str) -> str:
    """Load a markdown file from /content by filename (with or without .md extension)."""
    if not name.endswith(".md"):
        name = f"{name}.md"
    path = CONTENT_DIR / name
    if not path.exists():
        return f"*Content file `{name}` not found.*"
    return path.read_text(encoding="utf-8")
