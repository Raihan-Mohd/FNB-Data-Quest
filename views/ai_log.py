"""AI Usage Log viewer. Renders the log from content/ai_log.md."""
import streamlit as st
from utils.content_loader import load_markdown

st.title("AI Usage Log")
st.caption("Every AI prompt used in this project, what we used the output for, and what we changed.")

st.markdown(load_markdown("ai_log"))
