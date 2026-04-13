import os
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
load_dotenv()

import streamlit as st
from src.core.azure_search_helper import AzureSearchHelper


def example_basic_usage(index_name: str, pdf_file: Path):
    """Basic example: Index a PDF by pages using environment variables."""
    
    # Initialize the helper (reads from environment variables)
    helper = AzureSearchHelper()
    
    # Process and index a PDF document
    result = helper.process_and_index_pdf(
        pdf_path=pdf_file,
        index_name=index_name,
        title=pdf_file.name,
        chunking_strategy='page',
        metadata={
            'category': 'product_release_notes',
            'timestamp': datetime.now().isoformat()
        }
    )

    print(f"Indexing complete: {result}")


def create_cogsearch_index(index_name: str, pdf_files: list):
    """Example: Process multiple PDFs into the same index."""

    # Create index once
    st.session_state.search_helper.create_index(index_name, embedding_dimensions=1536)

    print (pdf_files)
    if not isinstance(pdf_files, list):
        pdf_files = [pdf_files]

    # Process multiple PDFs
    for file_content in pdf_files:
        print(f"\nProcessing: {file_content.name}")
        result = st.session_state.search_helper.process_and_index_pdf(
            pdf_path=file_content,
            index_name=index_name,
            create_index=st.session_state.create_new_index,
            title=file_content.name,
            chunking_strategy='page',
            metadata={
                'category': 'product_release_notes',
                'timestamp': datetime.now().isoformat()
            }
        )
        print(f"Completed: {result['uploaded']} documents uploaded")


def main():
    """Streamlit app to build a knowledge base using Azure Cognitive Search."""
    st.set_page_config(page_title="Build Knowledge Base", page_icon=":books:", layout="centered")

    # Initialize Session state
    if "index_name" not in st.session_state:
        st.session_state.index_name = None
    if "user_question" not in st.session_state:
        st.session_state.user_question = None
    if "create_new_index" not in st.session_state:
        st.session_state.create_new_index = True
    if "has_vectorized_data" not in st.session_state:
        st.session_state.has_vectorized_data = None
    if "search_helper" not in st.session_state:
        st.session_state.search_helper = AzureSearchHelper()

    st.header("Build your knowledge base :books:", divider='green')

    index_name = st.text_input(
        'Cognitive Search Index name', placeholder='Name of the index')
    if index_name:
        st.session_state.index_name = index_name

    st.session_state.create_new_index = st.toggle("Create New Index?", value=True, key="create_new_index_toggle")

    pdf_docs = st.file_uploader(
        "Upload your PDFs here and click on 'Process' ", accept_multiple_files=True)

    if st.button(":blue[Create Index]", type="secondary"):
        if len(pdf_docs) != 0:
            # Preview extracted text from uploaded PDFs
            with st.expander("PDF Text Preview", expanded=True):
                for pdf_file in pdf_docs:
                    st.subheader(f"📄 {pdf_file.name}")
                    pdf_file.seek(0)
                    try:
                        import PyPDF2, io
                        reader = PyPDF2.PdfReader(io.BytesIO(pdf_file.read()))
                        st.write(f"**Pages found:** {len(reader.pages)}")
                        for page_num, page in enumerate(reader.pages, start=1):
                            text = page.extract_text() or ""
                            if text.strip():
                                st.markdown(f"**Page {page_num}** ({len(text)} chars):")
                                st.text(text[:500] + ("..." if len(text) > 500 else ""))
                            else:
                                st.warning(f"Page {page_num}: No extractable text (image-based PDF?)")
                    except Exception as e:
                        st.error(f"Error reading {pdf_file.name}: {e}")

            # process the information from PDFs
            with st.spinner("Processing"):
                # Step 1: Create Azure Cognitive Search Index
                # Step 2: Upload PDFs to Azure Cognitive Search Index
                create_cogsearch_index(st.session_state.index_name, pdf_docs)
                st.success("Index created successfully")
        else:
            st.write("Please upload PDF documents")

    with st.sidebar:
        st.header(":blue[AI Search indices]")
        index_names = st.session_state.search_helper.get_all_indices()
        with st.container(height=300):
            for index_name in index_names:
                st.write(index_name)


if __name__ == "__main__":
    main()
