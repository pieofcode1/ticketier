"""
Azure AI Search Helper for PDF Document Processing and Indexing

This module provides a helper class for processing PDF documents, chunking them,
generating embeddings, and indexing them in Azure AI Search with vector search capabilities.
"""

import os
import uuid
from typing import List, Dict, Any, Optional, Literal
from dataclasses import dataclass
import PyPDF2
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchIndex,
    SearchField,
    SearchFieldDataType,
    VectorSearch,
    VectorSearchProfile,
    HnswAlgorithmConfiguration,
    SemanticConfiguration,
    SemanticField,
    SemanticPrioritizedFields,
    SemanticSearch,
)
from azure.search.documents.models import VectorizedQuery
from openai import AzureOpenAI
from azure.identity import DefaultAzureCredential, get_bearer_token_provider


@dataclass
class DocumentChunk:
    """Represents a chunk of a document with its metadata."""
    id: str
    title: str
    content: str
    metadata: Dict[str, Any]
    page_number: Optional[int] = None
    section_name: Optional[str] = None


@dataclass
class SearchDocument:
    """Represents a document ready for indexing in Azure AI Search."""
    id: str
    title: str
    title_vector: List[float]
    content: str
    content_vector: List[float]
    metadata: Dict[str, Any]


class AzureSearchHelper:
    """
    Helper class for Azure AI Search operations with PDF processing and vector embeddings.
    
    This class handles:
    - PDF document extraction and chunking (by page or section)
    - Embedding generation for text content
    - Index creation and management
    - Document upload to Azure AI Search
    """

    def __init__(
        self,
        search_endpoint: Optional[str] = None,
        search_key: Optional[str] = None,
        azure_openai_endpoint: Optional[str] = None,
        azure_openai_key: Optional[str] = None,
        azure_openai_api_version: Optional[str] = None,
        embedding_deployment: Optional[str] = None,
        use_azure_credential: Optional[bool] = None,
    ):
        """
        Initialize the Azure Search Helper.
        
        All parameters can be provided directly or read from environment variables.
        Environment variables take precedence if parameters are not provided.

        Args:
            search_endpoint: Azure AI Search service endpoint (env: AZURE_SEARCH_SERVICE_ENDPOINT)
            search_key: Azure AI Search admin key (env: AZURE_SEARCH_API_KEY)
            azure_openai_endpoint: Azure OpenAI endpoint for embeddings (env: AI_FOUNDRY_OPENAI_ENDPOINT)
            azure_openai_key: Azure OpenAI API key (env: AI_FOUNDRY_API_KEY)
            azure_openai_api_version: Azure OpenAI API version (env: AZURE_AI_API_VERSION, default: "2025-04-01-preview")
            embedding_deployment: Name of the embedding deployment (env: EMBEDDING_DEPLOYMENT_NAME, default: "text-embedding-ada-002")
            use_azure_credential: Whether to use DefaultAzureCredential (env: USE_AZURE_CREDENTIAL, default: False)
        """
        # Read from environment variables with fallback to provided parameters
        self.search_endpoint = search_endpoint or os.getenv("AI_SEARCH_SERVICE_ENDPOINT")
        search_key = search_key or os.getenv("AI_SEARCH_API_KEY")
        self.azure_openai_endpoint = azure_openai_endpoint or os.getenv("AI_FOUNDRY_OPENAI_ENDPOINT")
        azure_openai_key = azure_openai_key or os.getenv("AI_FOUNDRY_API_KEY")
        self.azure_openai_api_version = azure_openai_api_version or os.getenv("AZURE_AI_API_VERSION", "2025-04-01-preview")
        self.embedding_deployment = embedding_deployment or os.getenv("EMBEDDING_DEPLOYMENT_NAME", "text-embedding-ada-002")
        self.gpt_deployment = os.getenv("GPT_MODEL_DEPLOYMENT_NAME", "gpt-4.1")
        use_azure_credential = use_azure_credential if use_azure_credential is not None else os.getenv("USE_AZURE_CREDENTIAL", "false").lower() == "true"
        
        # Validate required parameters
        if not self.search_endpoint:
            raise ValueError("search_endpoint must be provided or set AZURE_SEARCH_SERVICE_ENDPOINT environment variable")
        
        if not use_azure_credential and not search_key:
            raise ValueError("search_key must be provided or set AZURE_SEARCH_API_KEY environment variable when not using Azure credential")
        
        print(f"Using Azure Search Endpoint: {self.search_endpoint}, type: {type(self.search_endpoint)}")
        
        # Initialize credentials
        if use_azure_credential:
            self.credential = DefaultAzureCredential()
            self.index_client = SearchIndexClient(
                endpoint=self.search_endpoint,
                credential=self.credential
            )
        else:
            self.credential = AzureKeyCredential(search_key)
            self.index_client = SearchIndexClient(
                endpoint=self.search_endpoint,
                credential=self.credential
            )
        
        # Initialize OpenAI client if endpoint is provided
        self.openai_client = None
        if self.azure_openai_endpoint:
            if use_azure_credential:
                token_provider = get_bearer_token_provider(
                    DefaultAzureCredential(),
                    "https://cognitiveservices.azure.com/.default"
                )
                self.openai_client = AzureOpenAI(
                    azure_endpoint=self.azure_openai_endpoint,
                    azure_ad_token_provider=token_provider,
                    api_version=self.azure_openai_api_version
                )
            elif azure_openai_key:
                self.openai_client = AzureOpenAI(
                    azure_endpoint=self.azure_openai_endpoint,
                    api_key=azure_openai_key,
                    api_version=self.azure_openai_api_version
                )

    def get_all_indices(self) -> List[str]:
        """
        Retrieve the list of existing Azure AI Search indices.

        Returns:
            List of index names
        """
        index_names = self.index_client.list_index_names()
        return list(index_names)

    def extract_text_from_pdf(self, pdf_path: str) -> List[Dict[str, Any]]:
        """
        Extract text from a PDF file, organizing by pages.

        Args:
            pdf_path: Path to the PDF file

        Returns:
            List of dictionaries containing page number and text content
        """
        pages = []
        
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            
            for page_num, page in enumerate(pdf_reader.pages, start=1):
                text = page.extract_text()
                if text.strip():  # Only include pages with content
                    pages.append({
                        'page_number': page_num,
                        'text': text,
                        'char_count': len(text)
                    })
        
        return pages

    def chunk_by_page(
        self,
        pdf_path: str,
        title: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[DocumentChunk]:
        """
        Chunk a PDF document by pages.

        Args:
            pdf_path: Path to the PDF file
            title: Title of the document
            metadata: Additional metadata to include

        Returns:
            List of DocumentChunk objects
        """
        pages = self.extract_text_from_pdf(pdf_path)
        chunks = []
        
        base_metadata = metadata or {}
        base_metadata.update({
            'source_file': os.path.basename(pdf_path),
            'chunking_strategy': 'by_page'
        })
        
        for page in pages:
            chunk_id = str(uuid.uuid4())
            chunk_metadata = {
                **base_metadata,
                'page_number': page['page_number'],
                'char_count': page['char_count']
            }
            
            chunk = DocumentChunk(
                id=chunk_id,
                title=f"{title} - Page {page['page_number']}",
                content=page['text'],
                metadata=chunk_metadata,
                page_number=page['page_number']
            )
            chunks.append(chunk)
        
        return chunks

    def chunk_by_section(
        self,
        pdf_path: str,
        title: str,
        section_headers: List[str],
        metadata: Optional[Dict[str, Any]] = None,
        max_chunk_size: int = 2000
    ) -> List[DocumentChunk]:
        """
        Chunk a PDF document by sections based on header patterns.

        Args:
            pdf_path: Path to the PDF file
            title: Title of the document
            section_headers: List of section header patterns to split on
            metadata: Additional metadata to include
            max_chunk_size: Maximum character count per chunk

        Returns:
            List of DocumentChunk objects
        """
        pages = self.extract_text_from_pdf(pdf_path)
        full_text = "\n\n".join([p['text'] for p in pages])
        
        chunks = []
        base_metadata = metadata or {}
        base_metadata.update({
            'source_file': os.path.basename(pdf_path),
            'chunking_strategy': 'by_section'
        })
        
        # Simple section splitting based on headers
        current_section = ""
        current_section_name = "Introduction"
        
        lines = full_text.split('\n')
        
        for line in lines:
            # Check if line is a section header
            is_header = any(header.lower() in line.lower() for header in section_headers)
            
            if is_header and current_section:
                # Save current section as chunk
                chunk_id = str(uuid.uuid4())
                chunk = DocumentChunk(
                    id=chunk_id,
                    title=f"{title} - {current_section_name}",
                    content=current_section.strip(),
                    metadata={**base_metadata, 'section': current_section_name},
                    section_name=current_section_name
                )
                chunks.append(chunk)
                
                # Start new section
                current_section = line + "\n"
                current_section_name = line.strip()
            else:
                current_section += line + "\n"
                
                # Handle max chunk size
                if len(current_section) > max_chunk_size:
                    chunk_id = str(uuid.uuid4())
                    chunk = DocumentChunk(
                        id=chunk_id,
                        title=f"{title} - {current_section_name}",
                        content=current_section.strip(),
                        metadata={**base_metadata, 'section': current_section_name},
                        section_name=current_section_name
                    )
                    chunks.append(chunk)
                    current_section = ""
        
        # Add final section
        if current_section.strip():
            chunk_id = str(uuid.uuid4())
            chunk = DocumentChunk(
                id=chunk_id,
                title=f"{title} - {current_section_name}",
                content=current_section.strip(),
                metadata={**base_metadata, 'section': current_section_name},
                section_name=current_section_name
            )
            chunks.append(chunk)
        
        return chunks

    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts using Azure OpenAI.

        Args:
            texts: List of text strings to embed

        Returns:
            List of embedding vectors

        Raises:
            ValueError: If OpenAI client is not initialized
        """
        if not self.openai_client:
            raise ValueError(
                "OpenAI client not initialized. "
                "Please provide azure_openai_endpoint and azure_openai_key."
            )
        
        # Generate embeddings in batches to avoid token limits
        batch_size = 16
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            response = self.openai_client.embeddings.create(
                input=batch,
                model=self.embedding_deployment
            )
            
            batch_embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(batch_embeddings)
        
        return all_embeddings

    def prepare_search_documents(
        self,
        chunks: List[DocumentChunk]
    ) -> List[SearchDocument]:
        """
        Prepare document chunks for indexing by generating embeddings.

        Args:
            chunks: List of DocumentChunk objects

        Returns:
            List of SearchDocument objects ready for indexing
        """
        # Extract texts for embedding
        titles = [chunk.title for chunk in chunks]
        contents = [chunk.content for chunk in chunks]
        
        # Generate embeddings
        print(f"Generating embeddings for {len(chunks)} chunks...")
        title_embeddings = self.generate_embeddings(titles)
        content_embeddings = self.generate_embeddings(contents)
        
        # Create search documents
        search_docs = []
        for i, chunk in enumerate(chunks):
            doc = SearchDocument(
                id=chunk.id,
                title=chunk.title,
                title_vector=title_embeddings[i],
                content=chunk.content,
                content_vector=content_embeddings[i],
                metadata=chunk.metadata
            )
            search_docs.append(doc)
        
        return search_docs

    def create_index(
        self,
        index_name: str,
        embedding_dimensions: int = 1536,
        recreate_if_exists: bool = False
    ) -> SearchIndex:
        """
        Create an Azure AI Search index with vector search capabilities.

        Args:
            index_name: Name of the index to create
            embedding_dimensions: Dimension of the embedding vectors
            recreate_if_exists: Whether to delete and recreate if index exists

        Returns:
            Created SearchIndex object
        """
        # Check if index exists
        try:
            existing_index = self.index_client.get_index(index_name)
            if recreate_if_exists:
                print(f"Deleting existing index: {index_name}")
                self.index_client.delete_index(index_name)
            else:
                print(f"Index {index_name} already exists")
                return existing_index
        except Exception:
            pass  # Index doesn't exist, continue with creation
        
        # Define index fields
        fields = [
            SearchField(
                name="id",
                type=SearchFieldDataType.String,
                key=True,
                filterable=True
            ),
            SearchField(
                name="title",
                type=SearchFieldDataType.String,
                searchable=True,
                filterable=True,
                sortable=True
            ),
            SearchField(
                name="title_vector",
                type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                searchable=True,
                vector_search_dimensions=embedding_dimensions,
                vector_search_profile_name="default-vector-profile"
            ),
            SearchField(
                name="content",
                type=SearchFieldDataType.String,
                searchable=True,
                analyzer_name="en.microsoft"
            ),
            SearchField(
                name="content_vector",
                type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                searchable=True,
                vector_search_dimensions=embedding_dimensions,
                vector_search_profile_name="default-vector-profile"
            ),
            SearchField(
                name="metadata",
                type=SearchFieldDataType.String,
                searchable=False,
                filterable=True
            ),
        ]
        
        # Configure vector search
        vector_search = VectorSearch(
            profiles=[
                VectorSearchProfile(
                    name="default-vector-profile",
                    algorithm_configuration_name="hnsw-config"
                )
            ],
            algorithms=[
                HnswAlgorithmConfiguration(
                    name="hnsw-config",
                    parameters={
                        "m": 4,
                        "efConstruction": 400,
                        "efSearch": 500,
                        "metric": "cosine"
                    }
                )
            ]
        )
        
        # Configure semantic search (optional but recommended)
        semantic_config = SemanticConfiguration(
            name="default-semantic-config",
            prioritized_fields=SemanticPrioritizedFields(
                title_field=SemanticField(field_name="title"),
                content_fields=[SemanticField(field_name="content")]
            )
        )
        
        semantic_search = SemanticSearch(
            configurations=[semantic_config]
        )
        
        # Create the index
        index = SearchIndex(
            name=index_name,
            fields=fields,
            vector_search=vector_search,
            semantic_search=semantic_search
        )
        
        print(f"Creating index: {index_name}")
        result = self.index_client.create_index(index)
        print(f"Index created successfully")
        
        return result

    def upload_documents(
        self,
        index_name: str,
        documents: List[SearchDocument]
    ) -> Dict[str, Any]:
        """
        Upload documents to an Azure AI Search index.

        Args:
            index_name: Name of the target index
            documents: List of SearchDocument objects to upload

        Returns:
            Dictionary with upload results
        """
        search_client = SearchClient(
            endpoint=self.search_endpoint,
            index_name=index_name,
            credential=self.credential
        )
        
        # Convert SearchDocument objects to dictionaries
        docs_to_upload = []
        for doc in documents:
            doc_dict = {
                'id': doc.id,
                'title': doc.title,
                'title_vector': doc.title_vector,
                'content': doc.content,
                'content_vector': doc.content_vector,
                'metadata': str(doc.metadata)  # Convert dict to string for storage
            }
            docs_to_upload.append(doc_dict)
        
        # Upload in batches
        batch_size = 100
        total_uploaded = 0
        
        for i in range(0, len(docs_to_upload), batch_size):
            batch = docs_to_upload[i:i + batch_size]
            result = search_client.upload_documents(documents=batch)
            total_uploaded += len(batch)
            print(f"Uploaded {total_uploaded}/{len(docs_to_upload)} documents")
        
        return {
            'total_documents': len(documents),
            'uploaded': total_uploaded,
            'status': 'success'
        }

    def vector_search(
        self,
        index_name: str,
        query_text: str,
        top_k: int = 5,
        search_field: str = 'content_vector',
        select_fields: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Perform vector search on the index using semantic similarity.

        Args:
            index_name: Name of the search index to query
            query_text: The text query to search for
            top_k: Number of top results to return (default: 5)
            search_field: Vector field to search against ('content_vector' or 'title_vector')
            select_fields: List of fields to return in results (default: all fields)

        Returns:
            List of search results with scores
        """

        # Generate embedding for the query text
        query_vector = self.generate_embeddings([query_text])[0]

        # Create search client
        search_client = SearchClient(
            endpoint=self.search_endpoint,
            index_name=index_name,
            credential=self.credential
        )

        # Create vector query
        vector_query = VectorizedQuery(
            vector=query_vector,
            k_nearest_neighbors=top_k,
            fields=search_field
        )

        # Perform search
        results = search_client.search(
            search_text=None,  # Pure vector search
            vector_queries=[vector_query],
            select=select_fields if select_fields else ['id', 'title', 'content', 'metadata'],
            top=top_k
        )

        # Format results
        search_results = []
        for result in results:
            result_dict = {
                'score': result['@search.score'],
                'id': result.get('id'),
                'title': result.get('title'),
                'content': result.get('content'),
                'metadata': result.get('metadata')
            }
            # Add any additional selected fields
            if select_fields:
                for field in select_fields:
                    if field not in result_dict and field in result:
                        result_dict[field] = result[field]
            search_results.append(result_dict)

        return search_results

    def rag_query(
        self,
        index_name: str,
        query: str,
        top_k: int = 3,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1000
    ) -> Dict[str, Any]:
        """
        Perform RAG (Retrieval-Augmented Generation) query by searching the index 
        and using retrieved context to generate an LLM response.

        Args:
            index_name: Name of the search index to query
            query: User's question or query
            top_k: Number of top search results to use as context (default: 3)
            system_prompt: Custom system prompt (default: generic RAG prompt)
            temperature: LLM temperature for response generation (default: 0.7)
            max_tokens: Maximum tokens in LLM response (default: 1000)

        Returns:
            Dictionary containing:
                - query: Original user query
                - answer: LLM-generated response
                - sources: List of retrieved documents used as context
                - metadata: Additional information about the retrieval and generation
        """
        if not self.openai_client:
            raise ValueError("OpenAI client not initialized. Ensure azure_openai_endpoint is configured.")
        
        # Step 1: Vector search to retrieve relevant documents
        search_results = self.vector_search(
            index_name=index_name,
            query_text=query,
            top_k=top_k,
            search_field='content_vector'
        )
        
        if not search_results:
            return {
                'query': query,
                'answer': "I couldn't find any relevant information to answer your question.",
                'sources': [],
                'metadata': {'retrieved_docs': 0}
            }
        
        # Step 2: Build context from retrieved documents
        context_parts = []
        for i, result in enumerate(search_results, 1):
            doc_text = f"[Document {i}]\nTitle: {result.get('title', 'N/A')}\nContent: {result.get('content', 'N/A')}\n"
            context_parts.append(doc_text)
        
        context = "\n".join(context_parts)
        
        # Step 3: Build system prompt
        if not system_prompt:
            system_prompt = """
            You are a helpful AI assistant. Answer the user's question based on the provided context documents.
            If the context doesn't contain enough information to answer the question, say so clearly.
            Cite the document numbers when referencing specific information.
            
            """
        
        # Step 4: Build user message with context
        user_message = f"""
        Context from knowledge base:
        {context}

        User Question: {query}

        Please provide a comprehensive answer based on the context above.
        """

        # Step 5: Call LLM to generate response
        try:
            response = self.openai_client.chat.completions.create(
                model=self.gpt_deployment,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ],
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            answer = response.choices[0].message.content
            
            # Step 6: Format sources for transparency
            sources = [
                {
                    'id': result.get('id'),
                    'title': result.get('title'),
                    'score': result.get('score'),
                    'content_preview': result.get('content', '')[:200] + '...' if len(result.get('content', '')) > 200 else result.get('content', '')
                }
                for result in search_results
            ]
            
            return {
                'query': query,
                'answer': answer,
                'sources': sources,
                'metadata': {
                    'retrieved_docs': len(search_results),
                    'model': self.gpt_deployment,
                    'temperature': temperature,
                    'max_tokens': max_tokens
                }
            }
        
        except Exception as e:
            return {
                'query': query,
                'answer': f"Error generating response: {str(e)}",
                'sources': sources if 'sources' in locals() else [],
                'metadata': {'error': str(e)}
            }

    def process_and_index_pdf(
        self,
        pdf_path: str,
        index_name: str,
        title: str,
        chunking_strategy: Literal['page', 'section'] = 'page',
        section_headers: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        create_index: bool = True,
        embedding_dimensions: int = 1536
    ) -> Dict[str, Any]:
        """
        Complete workflow: Extract PDF, chunk, generate embeddings, and index.

        Args:
            pdf_path: Path to the PDF file
            index_name: Name of the search index
            title: Title of the document
            chunking_strategy: 'page' or 'section'
            section_headers: List of section headers (required for 'section' strategy)
            metadata: Additional metadata to include
            create_index: Whether to create the index if it doesn't exist
            embedding_dimensions: Dimension of embedding vectors

        Returns:
            Dictionary with processing results
        """
        print(f"Processing PDF: {pdf_path}")
        
        # Step 1: Chunk the document
        if chunking_strategy == 'page':
            chunks = self.chunk_by_page(pdf_path, title, metadata)
        elif chunking_strategy == 'section':
            if not section_headers:
                section_headers = ['Chapter', 'Section', 'Introduction', 'Conclusion']
            chunks = self.chunk_by_section(pdf_path, title, section_headers, metadata)
        else:
            raise ValueError("chunking_strategy must be 'page' or 'section'")
        
        print(f"Created {len(chunks)} chunks")
        
        # Step 2: Generate embeddings and prepare documents
        search_docs = self.prepare_search_documents(chunks)
        
        # Step 3: Create index if needed
        if create_index:
            self.create_index(index_name, embedding_dimensions)
        
        # Step 4: Upload documents
        result = self.upload_documents(index_name, search_docs)
        
        result['chunks_created'] = len(chunks)
        result['chunking_strategy'] = chunking_strategy
        
        return result

    def search_docs(
        self,
        index_name: str,
        query: str,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Search documents in the specified Azure AI Search index.

        Args:
            index_name: Name of the search index
            query: Search query string
            top_k: Number of top results to return

        Returns:
            List of search result documents
        """
        search_client = SearchClient(
            endpoint=self.search_endpoint,
            index_name=index_name,
            credential=self.credential
        )

        results = search_client.search(
            search_text=query,
            top=top_k
        )

        search_results = []
        for result in results:
            search_results.append(result)

        return search_results
