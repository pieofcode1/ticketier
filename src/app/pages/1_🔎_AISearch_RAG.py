import os
import streamlit as st
from pathlib import Path
from openai import AzureOpenAI
from dotenv import load_dotenv
load_dotenv()

from src.core.azure_search_helper import AzureSearchHelper
from src.core.chat_history_helper import save_chat_history


def display_chat_history():
    """Display the chat history with latest response first."""
    if st.session_state.aisearch_chat_history:
        # Show the most recent exchange at the top
        latest = st.session_state.aisearch_chat_history[-1]
        # Handle both 'query' and 'question' keys for compatibility
        user_msg = latest.get("query") or latest.get("question", "")
        with st.chat_message("user"):
            st.write(user_msg)
        with st.chat_message("assistant"):
            st.write(latest.get("answer", ""))
        
        # Show divider if there's more history
        if len(st.session_state.aisearch_chat_history) > 1:
            st.divider()
            st.caption("📜 Previous conversations")
            
            # Show remaining history in reverse order (newest to oldest)
            for message in reversed(st.session_state.aisearch_chat_history[:-1]):
                user_msg = message.get("query") or message.get("question", "")
                with st.chat_message("user"):
                    st.write(user_msg)
                with st.chat_message("assistant"):
                    st.write(message.get("answer", ""))


def handle_user_input(query: str):
    """Handle user input question and update chat history."""

    response = st.session_state.search_helper.rag_query(
                                        query=query,
                                        index_name=st.session_state.selected_index
                                    )
    st.session_state.aisearch_chat_history.append(response)


def main():
    """Streamlit app to chat with documents using Azure Cognitive Search."""
    st.set_page_config(page_title="Chatbot",
                       page_icon=":mag_right:", layout="wide")

    # Initialize Session state
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "aisearch_chat_history" not in st.session_state:
        st.session_state.aisearch_chat_history = []
    if "selected_index" not in st.session_state:
        st.session_state.selected_index = None
    if "search_helper" not in st.session_state:
        st.session_state.search_helper = AzureSearchHelper()

    st.header("Chat with your Data (AI Search) :mag_right:", divider='blue')
    
    # Chat input
    user_question = st.chat_input("Ask a question about your documents")
    if user_question:
        print(f"User Question: {user_question}")
        handle_user_input(user_question)

    # Display chat history
    if st.session_state.aisearch_chat_history:
        display_chat_history()

    with st.sidebar:
        st.markdown("#### Cognitive Search Vector Store")
        st.write(
            """Cognitive Search Indexes are populated with the domain specific knowledgebase.""")
        st.write("\n\n")

        st.session_state.use_az_search_vector_store = True
        indices = st.session_state.search_helper.get_all_indices()
        selected_index = st.selectbox(
            'Choose Vector Index to use',
            indices
        )
        st.write('You selected:', selected_index)

        if (selected_index != st.session_state.selected_index):
            st.session_state.aisearch_chat_history = []
            st.session_state.selected_index = selected_index
        
        st.divider()
        
        # Chat history actions
        if st.session_state.aisearch_chat_history:
            col1, col2 = st.columns(2)
            with col1:
                if st.button("💾 Save Chat", use_container_width=True):
                    metadata = {
                        "index_name": st.session_state.selected_index
                    }
                    filepath = save_chat_history(
                        st.session_state.aisearch_chat_history,
                        "aisearch_rag",
                        metadata
                    )
                    if filepath:
                        st.success(f"💾 Chat saved!")
                    else:
                        st.warning("No chat to save")
            with col2:
                if st.button("🗑️ Clear Chat", use_container_width=True):
                    st.session_state.aisearch_chat_history = []
                    st.rerun()


if __name__ == "__main__":
    main()
