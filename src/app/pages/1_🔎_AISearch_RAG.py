import os
import streamlit as st
from pathlib import Path
from openai import AzureOpenAI
from dotenv import load_dotenv
load_dotenv()

from src.core.azure_search_helper import AzureSearchHelper


def handle_user_input(query: str):
    """Handle user input question and update chat history."""

    response = st.session_state.search_helper.rag_query(
                                        query=query,
                                        index_name=st.session_state.selected_index
                                    )
    st.session_state.chat_history.append(response)
    st.write(response)
    # for i, message in enumerate(reversed(st.session_state.chat_history)):
    for i, message in enumerate(st.session_state.chat_history):
        print(f"Idx: {i}, Message: {message}")
        with st.chat_message("user"):
            st.write(message["query"])
        with st.chat_message("assistant"):
            st.write(message["answer"])


def main():
    """Streamlit app to chat with documents using Azure Cognitive Search."""
    st.set_page_config(page_title="Chatbot",
                       page_icon=":mag_right:", layout="wide")

    # Initialize Session state
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "selected_index" not in st.session_state:
        st.session_state.selected_index = None
    if "search_helper" not in st.session_state:
        st.session_state.search_helper = AzureSearchHelper()

    st.header("Chat with your Data (AI Search) :mag_right:", divider='blue')
    # user_question = st.text_input("Ask a question about your documents")
    user_question = st.chat_input("Ask a question about your documents")
    if user_question:
        print(f"User Question: {user_question}")
        handle_user_input(user_question)

    with st.sidebar:
        st.markdown("#### Cognitive Search Vector Store")
        st.write(
            """                 Cognitive Search Indexes are populated with the domain specific knowledgebase.""")
        st.write("\n\n")

        st.session_state.use_az_search_vector_store = True
        indices = st.session_state.search_helper.get_all_indices()
        selected_index = st.selectbox(
            'Choose Vector Index to use',
            indices
        )
        st.write('You selected:', selected_index)

        if (selected_index != st.session_state.selected_index):
            st.session_state.chat_history = []
            st.session_state.selected_index = selected_index


if __name__ == "__main__":
    main()
