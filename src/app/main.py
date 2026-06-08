"""Main entry point for the Ticketier Streamlit app."""

import streamlit as st

st.set_page_config(page_title="Cognitive App", page_icon="🎫", layout="wide")

st.title("🎫 Cognitive App")
st.subheader("Agentic AI App on Azure AI Foundry", divider="blue")

st.markdown(
    """
Ticketier is an intelligent ticket analysis and knowledge management application 
powered by Azure AI Foundry. Upload documents, search with AI, and analyze images 
— all from one place.
"""
)

# Card styling
st.markdown("""
<style>
div[data-testid="stColumn"] > div {
    border: 1px solid #444;
    border-radius: 10px;
    padding: 1.5rem;
    height: 100%;
}
</style>
""", unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("### 📘 Build Knowledge Base")
    st.markdown(
        "Upload PDF documents to create a searchable vector index "
        "with Azure AI Search and OpenAI embeddings."
    )
    st.page_link("pages/0_📘_Build_Knowledge_base.py", label="Go to Knowledge Base", icon="📘")

with col2:
    st.markdown("### 🔎 AI Search RAG")
    st.markdown(
        "Ask questions about your indexed documents and get "
        "AI-generated answers with source citations."
    )
    st.page_link("pages/1_🔎_AISearch_RAG.py", label="Go to AI Search", icon="🔎")

with col3:
    st.markdown("### 🖼️ Vision RAG")
    st.markdown(
        "Upload one or more images and chat about them using "
        "Azure OpenAI's vision-capable models."
    )
    st.page_link("pages/2_🖼️_Vision_RAG.py", label="Go to Vision RAG", icon="🖼️")

with col4:
    st.markdown("### 🧱 Databricks SQL")
    st.markdown(
        "Ask questions in natural language and get answers "
        "by running AI-generated SQL on Databricks."
    )
    st.page_link("pages/3_🧱_Databricks_SQL.py", label="Go to Databricks SQL", icon="🧱")

st.divider()

with st.expander("⚙️ Tech Stack"):
    st.markdown(
        """
| Component | Technology |
|-----------|------------|
| **UI** | Streamlit |
| **LLM** | Azure OpenAI (Responses API) |
| **Embeddings** | text-embedding-ada-002 |
| **Search** | Azure AI Search (HNSW vector) |
| **Database** | PostgreSQL + pgvector |
| **Agents** | Microsoft Agent Framework |
| **Auth** | Azure Identity / API Key |
"""
    )
