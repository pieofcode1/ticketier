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
    """Process user question about the uploaded images."""
    if not st.session_state.image_data_list:
        st.warning("Please upload at least one image first.")
        return

    try:
        manager = get_openai_manager()
        
        images = st.session_state.image_data_list
        media_types = st.session_state.image_media_types

        if len(images) == 1:
            # Single image — use analyze_image
            result = manager.analyze_image(
                image=images[0],
                prompt=question,
                instructions="You are a helpful assistant that analyzes images and answers questions about them. "
                            "Provide detailed, accurate descriptions and insights based on the image content.",
                media_type=media_types[0],
                detail="high"
            )
        else:
            # Multiple images — use analyze_multiple_images
            from src.core.openai_manager import ImageContent
            image_contents = [
                ImageContent(data=img, media_type=mt, detail="high")
                for img, mt in zip(images, media_types)
            ]
            result = manager.analyze_multiple_images(
                images=image_contents,
                prompt=question,
                instructions="You are a helpful assistant that analyzes images and answers questions about them. "
                            "Provide detailed, accurate descriptions and insights based on the image content.",
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
    if "image_data_list" not in st.session_state:
        st.session_state.image_data_list = []
    if "image_media_types" not in st.session_state:
        st.session_state.image_media_types = []
    if "image_names" not in st.session_state:
        st.session_state.image_names = []
    if "vision_chat_history" not in st.session_state:
        st.session_state.vision_chat_history = []

    st.header("Chat with your Images 🖼️", divider="red")
    
    # Display info if no image uploaded
    if not st.session_state.image_data_list:
        st.info("👈 Upload one or more images in the sidebar to get started!")

    # Chat input
    user_question = st.chat_input("Ask a question about the image")
    if user_question:
        handle_user_input(user_question)

    # Display chat history
    if st.session_state.vision_chat_history:
        display_chat_history()

    # Sidebar for image upload
    with st.sidebar:
        st.subheader("📤 Upload Images")
        
        image_files = st.file_uploader(
            "Choose images to analyze",
            type=["png", "jpg", "jpeg", "gif", "webp", "bmp"],
            accept_multiple_files=True,
            help="Supported formats: PNG, JPG, JPEG, GIF, WEBP, BMP"
        )

        if image_files:
            # Get all image data
            new_data = [f.getvalue() for f in image_files]
            
            # Check if images changed
            if new_data != st.session_state.image_data_list:
                st.session_state.vision_chat_history = []
                st.session_state.image_data_list = new_data
                st.session_state.image_media_types = [get_image_media_type(f.name) for f in image_files]
                st.session_state.image_names = [f.name for f in image_files]

            # Display uploaded images
            for i, img_data in enumerate(st.session_state.image_data_list):
                name = st.session_state.image_names[i]
                st.image(img_data, caption=name, use_container_width=True)
                st.caption(f"📁 {name}  •  📊 {len(img_data) / 1024:.1f} KB")
        else:
            if st.session_state.image_data_list:
                st.session_state.image_data_list = []
                st.session_state.image_media_types = []
                st.session_state.image_names = []
                st.session_state.vision_chat_history = []
        
        st.divider()
        
        # Chat history actions
        if st.session_state.vision_chat_history:
            col1, col2 = st.columns(2)
            with col1:
                if st.button("💾 Save Chat", use_container_width=True):
                    metadata = {
                        "image_names": st.session_state.image_names,
                        "media_types": st.session_state.image_media_types
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
