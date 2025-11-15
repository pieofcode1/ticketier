# Azure AI Search Helper for PDF Documents

A comprehensive Python helper class for processing PDF documents, generating embeddings, and indexing them in Azure AI Search with vector search capabilities.

## Features

- **PDF Processing**: Extract text from PDF documents
- **Flexible Chunking**: 
  - Chunk by page for simple document splitting
  - Chunk by section for intelligent content organization
- **Embedding Generation**: Automatic embedding generation using Azure OpenAI
- **Vector Search**: Full support for Azure AI Search vector search capabilities
- **Semantic Search**: Built-in semantic search configuration
- **Batch Processing**: Efficient batch upload of documents
- **Flexible Authentication**: Support for both API keys and Azure DefaultAzureCredential

## Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Or update your `pyproject.toml` with:

```toml
dependencies = [
    "azure-identity>=1.25.1",
    "azure-search-documents>=11.6.0",
    "openai>=1.0.0",
    "PyPDF2>=3.0.0",
]
```

## Quick Start

### Basic Usage

```python
from core.azure_search_helper import AzureSearchHelper

# Initialize the helper (reads from environment variables)
helper = AzureSearchHelper()

# Process and index a PDF (all-in-one)
result = helper.process_and_index_pdf(
    pdf_path="document.pdf",
    index_name="documents-index",
    title="My Document",
    chunking_strategy='page'
)
```

Or with explicit parameters:

```python
from core.azure_search_helper import AzureSearchHelper

# Initialize the helper with explicit parameters
helper = AzureSearchHelper(
    search_endpoint="https://your-search-service.search.windows.net",
    search_key="your-search-key",
    azure_openai_endpoint="https://your-openai.openai.azure.com",
    azure_openai_key="your-openai-key",
    embedding_deployment="text-embedding-3-small"
)

# Process and index a PDF (all-in-one)
result = helper.process_and_index_pdf(
    pdf_path="document.pdf",
    index_name="documents-index",
    title="My Document",
    chunking_strategy='page'
)
```

### Using Azure Managed Identity

```python
# Set environment variable
# USE_AZURE_CREDENTIAL=true

helper = AzureSearchHelper(
    use_azure_credential=True
)

# Or set via environment variable and use defaults
helper = AzureSearchHelper()
```

## Chunking Strategies

### 1. Chunk by Page

Splits the PDF into chunks based on pages. Simple and effective for most documents.

```python
chunks = helper.chunk_by_page(
    pdf_path="document.pdf",
    title="Technical Manual",
    metadata={
        'category': 'technical',
        'author': 'John Doe'
    }
)
```

### 2. Chunk by Section

Splits the PDF based on section headers. Ideal for structured documents like research papers.

```python
chunks = helper.chunk_by_section(
    pdf_path="research.pdf",
    title="Research Paper",
    section_headers=[
        'Abstract',
        'Introduction',
        'Methodology',
        'Results',
        'Conclusion'
    ],
    metadata={'type': 'research'},
    max_chunk_size=2000
)
```

## Index Schema

The helper creates indexes with the following fields:

| Field | Type | Description |
|-------|------|-------------|
| `id` | String | Unique identifier (auto-generated UUID) |
| `title` | String | Document/chunk title (searchable, filterable) |
| `title_vector` | Collection(Single) | Embedding vector for the title |
| `content` | String | Text content (searchable with analyzer) |
| `content_vector` | Collection(Single) | Embedding vector for the content |
| `metadata` | String | JSON string with custom metadata (filterable) |

### Vector Search Configuration

- **Algorithm**: HNSW (Hierarchical Navigable Small World)
- **Metric**: Cosine similarity
- **Parameters**:
  - `m`: 4 (number of bi-directional links)
  - `efConstruction`: 400 (size of dynamic candidate list for construction)
  - `efSearch`: 500 (size of dynamic candidate list for search)

## Complete Workflow Example

```python
from core.azure_search_helper import AzureSearchHelper

# Initialize using environment variables
helper = AzureSearchHelper()

# Step 1: Create chunks
chunks = helper.chunk_by_page(
    pdf_path="technical_doc.pdf",
    title="Technical Documentation",
    metadata={'version': '1.0', 'department': 'engineering'}
)
print(f"Created {len(chunks)} chunks")

# Step 2: Generate embeddings
search_docs = helper.prepare_search_documents(chunks)

# Step 3: Create index
helper.create_index(
    index_name="tech-docs",
    embedding_dimensions=1536,
    recreate_if_exists=False
)

# Step 4: Upload documents
result = helper.upload_documents(
    index_name="tech-docs",
    documents=search_docs
)
print(f"Uploaded {result['uploaded']} documents")
```

## Batch Processing Multiple PDFs

```python
from core.azure_search_helper import AzureSearchHelper

# All configuration from environment variables
helper = AzureSearchHelper()

# Create index once
helper.create_index("all-documents", embedding_dimensions=1536)

# Process multiple PDFs
pdf_files = [
    ("manual1.pdf", "User Manual"),
    ("guide2.pdf", "Installation Guide"),
    ("faq3.pdf", "FAQ Document"),
]

for pdf_path, title in pdf_files:
    helper.process_and_index_pdf(
        pdf_path=pdf_path,
        index_name="all-documents",
        title=title,
        chunking_strategy='page',
        create_index=False  # Already created
    )
```

## Environment Variables

The helper automatically reads configuration from environment variables. This is the recommended approach for production use:

```bash
export AZURE_SEARCH_SERVICE_ENDPOINT="https://your-search.search.windows.net"
export AZURE_SEARCH_API_KEY="your-search-key"
export AI_FOUNDRY_OPENAI_ENDPOINT="https://your-openai.openai.azure.com/"
export AI_FOUNDRY_API_KEY="your-openai-key"
export AZURE_AI_API_VERSION="2025-04-01-preview"  # Optional, defaults to "2025-04-01-preview"
export EMBEDDING_DEPLOYMENT_NAME="text-embedding-ada-002"  # Optional, defaults to "text-embedding-ada-002"
export USE_AZURE_CREDENTIAL="false"  # Optional, defaults to "false"
```

### Supported Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `AZURE_SEARCH_SERVICE_ENDPOINT` | Yes | - | Azure AI Search service endpoint |
| `AZURE_SEARCH_API_KEY` | Conditional* | - | Azure AI Search admin key |
| `AI_FOUNDRY_OPENAI_ENDPOINT` | No** | - | Azure OpenAI endpoint for embeddings |
| `AI_FOUNDRY_API_KEY` | Conditional*** | - | Azure OpenAI API key |
| `AZURE_AI_API_VERSION` | No | `2025-04-01-preview` | Azure OpenAI API version |
| `EMBEDDING_DEPLOYMENT_NAME` | No | `text-embedding-ada-002` | Embedding deployment name |
| `USE_AZURE_CREDENTIAL` | No | `false` | Use DefaultAzureCredential for auth |

\* Required if `USE_AZURE_CREDENTIAL=false`  
\** Required only if you want to generate embeddings  
\*** Required if `USE_AZURE_CREDENTIAL=false` and using embeddings

Then in your code:

```python
from core.azure_search_helper import AzureSearchHelper

# All configuration from environment variables
helper = AzureSearchHelper()
```

## API Reference

### AzureSearchHelper

#### Constructor

```python
AzureSearchHelper(
    search_endpoint: Optional[str] = None,
    search_key: Optional[str] = None,
    azure_openai_endpoint: Optional[str] = None,
    azure_openai_key: Optional[str] = None,
    azure_openai_api_version: Optional[str] = None,
    embedding_deployment: Optional[str] = None,
    use_azure_credential: Optional[bool] = None
)
```

All parameters are optional and will be read from environment variables if not provided.
See [Environment Variables](#environment-variables) section for details.

#### Methods

##### `extract_text_from_pdf(pdf_path: str) -> List[Dict[str, Any]]`
Extract text from PDF, organized by pages.

##### `chunk_by_page(pdf_path: str, title: str, metadata: Optional[Dict] = None) -> List[DocumentChunk]`
Chunk PDF by pages.

##### `chunk_by_section(pdf_path: str, title: str, section_headers: List[str], metadata: Optional[Dict] = None, max_chunk_size: int = 2000) -> List[DocumentChunk]`
Chunk PDF by sections.

##### `generate_embeddings(texts: List[str]) -> List[List[float]]`
Generate embeddings for texts.

##### `prepare_search_documents(chunks: List[DocumentChunk]) -> List[SearchDocument]`
Convert chunks to search documents with embeddings.

##### `create_index(index_name: str, embedding_dimensions: int = 1536, recreate_if_exists: bool = False) -> SearchIndex`
Create an Azure AI Search index.

##### `upload_documents(index_name: str, documents: List[SearchDocument]) -> Dict[str, Any]`
Upload documents to an index.

##### `process_and_index_pdf(...) -> Dict[str, Any]`
Complete workflow: extract, chunk, embed, and index a PDF.

## Best Practices

1. **Chunking Strategy**: 
   - Use `page` for simpler documents or when page boundaries are meaningful
   - Use `section` for structured documents like papers, manuals, or books

2. **Embedding Dimensions**: 
   - `text-embedding-ada-002`: 1536 dimensions (Azure OpenAI legacy)
   - `text-embedding-3-small`: 1536 dimensions (cost-effective)
   - `text-embedding-3-large`: 3072 dimensions (higher quality)

3. **Metadata**: Include relevant metadata for filtering:
   ```python
   metadata = {
       'source': 'technical_docs',
       'version': '2.0',
       'department': 'engineering',
       'date': '2024-11-13',
       'classification': 'internal'
   }
   ```

4. **Batch Size**: The helper automatically batches uploads (100 docs) and embeddings (16 texts) to optimize performance.

5. **Error Handling**: Always wrap operations in try-except blocks for production use.

## Troubleshooting

### Common Issues

1. **Authentication Errors**: Ensure your keys are valid or Azure credentials are properly configured
2. **Embedding Dimension Mismatch**: Ensure the embedding dimensions match your model
3. **PDF Extraction Issues**: Some PDFs may have encoding issues; consider preprocessing
4. **Rate Limits**: The helper batches requests, but you may need to add retry logic for large datasets

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues.
