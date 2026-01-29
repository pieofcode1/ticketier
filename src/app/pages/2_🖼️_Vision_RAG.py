"""
Vision RAG Page - Chat with Images using Azure OpenAI Vision

This page allows users to upload images and ask questions about them
using Azure OpenAI's vision-capable models via the Responses API.
"""

import os
import sys
import streamlit as st
from dotenv import load_dotenv
from src.core.openai_manager import OpenAIManager
from src.core.chat_history_helper import save_chat_history

# Load environment variables
load_dotenv()

# Custom CSS for chat styling
CUSTOM_CSS = """
<style>
.chat-message {
    padding: 1rem;
    border-radius: 0.5rem;
    margin-bottom: 1rem;
    display: flex;
    flex-direction: column;
}
.chat-message.user {
    background-color: #2b313e;
}
.chat-message.assistant {
    background-color: #475063;
}
</style>
"""


@st.cache_resource
def get_openai_manager():
    """Initialize and cache the OpenAI manager."""
    return OpenAIManager()


def get_image_media_type(file_name: str) -> str:
    """Determine the media type based on file extension."""
    extension = file_name.lower().split(".")[-1] if "." in file_name else "png"
    media_types = {
        "png": "image/png",
        "jpg": "image/jpeg",
        "jpeg": "image/jpeg",
        "gif": "image/gif",
        "webp": "image/webp",
        "bmp": "image/bmp",
    }
    return media_types.get(extension, "image/png")


def handle_user_input(question: str):
    """Process user question about the uploaded image."""
    if not st.session_state.image_data:
        st.warning("Please upload an image first.")
        return

    try:
        manager = get_openai_manager()
        
        # Get media type from uploaded file
        media_type = st.session_state.get("image_media_type", "image/png")
        
        # Analyze image using the OpenAI Manager
        result = manager.analyze_image(
            image=st.session_state.image_data,
            prompt=question,
            instructions="You are a helpful assistant that analyzes images and answers questions about them. "
                        "Provide detailed, accurate descriptions and insights based on the image content.",
            media_type=media_type,
            detail="high"
        )

        # Store the Q&A in chat history
        st.session_state.vision_chat_history.append({
            "question": question,
            "answer": result.content
        })

    except Exception as e:
        st.error(f"Error analyzing image: {str(e)}")
        return


def display_chat_history():
    """Display the chat history with latest response first."""
    # Display latest response first, then show history in reverse order
    if st.session_state.vision_chat_history:
        # Show the most recent exchange at the top
        latest = st.session_state.vision_chat_history[-1]
        with st.chat_message("user"):
            st.write(latest["question"])
        with st.chat_message("assistant"):
            st.write(latest["answer"])
        
        # Show divider if there's more history
        if len(st.session_state.vision_chat_history) > 1:
            st.divider()
            st.caption("📜 Previous conversations")
            
            # Show remaining history in reverse order (newest to oldest)
            for message in reversed(st.session_state.vision_chat_history[:-1]):
                with st.chat_message("user"):
                    st.write(message["question"])
                with st.chat_message("assistant"):
                    st.write(message["answer"])


def main():
    """Streamlit app to chat with images using Azure OpenAI Vision."""
    st.set_page_config(
        page_title="Vision RAG - Image Analysis",
        page_icon="🖼️",
        layout="wide"
    )
    st.write(CUSTOM_CSS, unsafe_allow_html=True)

    # Initialize session state
    if "image_data" not in st.session_state:
        st.session_state.image_data = None
    if "image_media_type" not in st.session_state:
        st.session_state.image_media_type = "image/png"
    if "vision_chat_history" not in st.session_state:
        st.session_state.vision_chat_history = []

    st.header("Chat with your Images 🖼️", divider="red")
    
    # Display info if no image uploaded
    if not st.session_state.image_data:
        st.info("👈 Upload an image in the sidebar to get started!")

    # Chat input
    user_question = st.chat_input("Ask a question about the image")
    if user_question:
        handle_user_input(user_question)

    # Display chat history
    if st.session_state.vision_chat_history:
        display_chat_history()

    # Sidebar for image upload
    with st.sidebar:
        st.subheader("📤 Upload Image")
        
        image_file = st.file_uploader(
            "Choose an image to analyze",
            type=["png", "jpg", "jpeg", "gif", "webp", "bmp"],
            accept_multiple_files=False,
            help="Supported formats: PNG, JPG, JPEG, GIF, WEBP, BMP"
        )

        if image_file is not None:
            # Get image data
            image_data = image_file.getvalue()
            
            # Check if it's a new image
            if image_data != st.session_state.image_data:
                st.session_state.vision_chat_history = []
                st.session_state.image_data = image_data
                st.session_state.image_media_type = get_image_media_type(image_file.name)

            # Display the uploaded image
            st.image(st.session_state.image_data, caption="Uploaded Image", width='stretch')
            
            # Display image info
            st.caption(f"📁 {image_file.name}")
            st.caption(f"📊 Size: {len(image_data) / 1024:.1f} KB")
            
            # Store image name for history metadata
            st.session_state.image_name = image_file.name
        
        st.divider()
        
        # Chat history actions
        if st.session_state.vision_chat_history:
            col1, col2 = st.columns(2)
            with col1:
                if st.button("💾 Save Chat", use_container_width=True):
                    metadata = {
                        "image_name": st.session_state.get("image_name", "unknown"),
                        "media_type": st.session_state.get("image_media_type", "image/png")
                    }
                    filepath = save_chat_history(
                        st.session_state.vision_chat_history,
                        "vision_rag",
                        metadata
                    )
                    if filepath:
                        st.success(f"💾 Chat saved!")
                    else:
                        st.warning("No chat to save")
            with col2:
                if st.button("🗑️ Clear Chat", use_container_width=True):
                    st.session_state.vision_chat_history = []
                    st.rerun()


if __name__ == "__main__":
    main()
