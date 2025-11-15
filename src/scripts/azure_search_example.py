"""
Example usage of the Azure Search Helper class for PDF indexing.

This script demonstrates how to use the AzureSearchHelper class to:
1. Process PDF documents
2. Chunk them by page or section
3. Generate embeddings
4. Create an Azure AI Search index
5. Upload documents with vector search capabilities

Note: All examples assume environment variables are set.
See README.md for required environment variables.
"""
import os
from datetime import datetime
from dotenv import load_dotenv
load_dotenv()

from src.core.azure_search_helper import AzureSearchHelper


def example_basic_usage():
    """Basic example: Index a PDF by pages using environment variables."""
    
    # Initialize the helper (reads from environment variables)
    helper = AzureSearchHelper()
    
    # Process and index a PDF document
    result = helper.process_and_index_pdf(
        pdf_path="C:\\Users\\keyurpatel\\Downloads\\Release_Notes_Technicolor_XB7_8.1p11s1.pdf",
        index_name="rel-notes-index",
        title="Release Notes Technicolor XB7 8.1p11s1",
        chunking_strategy='page',
        metadata={
            'category': 'product_release_notes',
            'timestamp': datetime.now().isoformat()
        }
    )
    
    print(f"Indexing complete: {result}")


def example_section_chunking():
    """Example: Index a PDF by sections."""
    
    # Initialize using environment variables
    helper = AzureSearchHelper()
    
    # Define section headers to split on
    section_headers = [
        'Introduction',
        'Background',
        'Methodology',
        'Results',
        'Discussion',
        'Conclusion',
        'References'
    ]
    
    result = helper.process_and_index_pdf(
        pdf_path="path/to/research_paper.pdf",
        index_name="research-papers-index",
        title="Research Paper Title",
        chunking_strategy='section',
        section_headers=section_headers,
        metadata={
            'document_type': 'research_paper',
            'field': 'computer_science'
        }
    )
    
    print(f"Indexing complete: {result}")


def example_with_azure_credential():
    """Example: Use DefaultAzureCredential for authentication."""
    
    # Set USE_AZURE_CREDENTIAL=true in environment or pass explicitly
    helper = AzureSearchHelper(use_azure_credential=True)
    
    result = helper.process_and_index_pdf(
        pdf_path="path/to/document.pdf",
        index_name="documents-index",
        title="Document Title",
        chunking_strategy='page'
    )
    
    print(f"Indexing complete: {result}")


def example_manual_workflow():
    """Example: Manually control each step of the workflow."""
    
    # Initialize using environment variables
    helper = AzureSearchHelper()
    
    # Step 1: Create chunks
    chunks = helper.chunk_by_page(
        pdf_path="path/to/document.pdf",
        title="My Document",
        metadata={'category': 'manual'}
    )
    print(f"Created {len(chunks)} chunks")
    
    # Step 2: Prepare search documents (generates embeddings)
    search_docs = helper.prepare_search_documents(chunks)
    print(f"Generated embeddings for {len(search_docs)} documents")
    
    # Step 3: Create index
    helper.create_index(
        index_name="my-custom-index",
        embedding_dimensions=1536,
        recreate_if_exists=False
    )
    
    # Step 4: Upload documents
    result = helper.upload_documents(
        index_name="my-custom-index",
        documents=search_docs
    )
    print(f"Upload result: {result}")


def example_batch_processing():
    """Example: Process multiple PDFs into the same index."""
    
    # Initialize using environment variables
    helper = AzureSearchHelper()
    
    # Create index once
    helper.create_index("batch-documents-index", embedding_dimensions=1536)
    
    # Process multiple PDFs
    pdf_files = [
        ("path/to/doc1.pdf", "Document 1"),
        ("path/to/doc2.pdf", "Document 2"),
        ("path/to/doc3.pdf", "Document 3"),
    ]
    
    for pdf_path, title in pdf_files:
        print(f"\nProcessing: {title}")
        result = helper.process_and_index_pdf(
            pdf_path=pdf_path,
            index_name="batch-documents-index",
            title=title,
            chunking_strategy='page',
            create_index=False  # Index already created
        )
        print(f"Completed: {result['uploaded']} documents uploaded")


def example_with_explicit_parameters():
    """Example: Override environment variables with explicit parameters."""
    
    # You can still override specific parameters if needed
    helper = AzureSearchHelper(
        embedding_deployment="text-embedding-3-large",  # Override default
        azure_openai_api_version="2024-08-01"  # Use newer API version
    )
    
    result = helper.process_and_index_pdf(
        pdf_path="C:\\Workspace\\Code\\Repos\\collab\\sql-agentic-app-with-fabric\\Data_Ingest\\RAG_Preparation\\SecureBank - Frequently Asked Questions.pdf",
        index_name="custom-config-index",
        title="Document with Custom Config",
        chunking_strategy='page'
    )
    
    print(f"Indexing complete: {result}")


if __name__ == "__main__":
    # Uncomment the example you want to run
    
    example_basic_usage()
    # example_section_chunking()
    # example_with_azure_credential()
    # example_manual_workflow()
    # example_batch_processing()
    # example_with_explicit_parameters()
