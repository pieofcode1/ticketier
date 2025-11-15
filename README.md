# Ticketier

Agentic AI Application leveraging Azure AI Foundry for intelligent ticket analysis and knowledge base management.

## Features

- **Knowledge Base Management**: Upload and index PDF documents with vector embeddings
- **AI-Powered Search**: Vector search using Azure AI Search with semantic capabilities
- **RAG (Retrieval-Augmented Generation)**: Query documents and get AI-generated answers with source citations
- **Multi-Agent Architecture**: Built on Azure AI Foundry with agentic framework
- **Interactive UI**: Streamlit-based web interface for easy interaction

## Prerequisites

- Python 3.12 or higher
- [uv](https://github.com/astral-sh/uv) package manager
- Azure subscription with:
  - Azure AI Foundry project
  - Azure OpenAI Service (with GPT and embedding models deployed)
  - Azure AI Search service
  - PostgreSQL database (optional)

## Setup

### 1. Clone the Repository

```bash
git clone <repository-url>
cd ticketier
```

### 2. Install Dependencies

Install uv package manager if not already installed:

```bash
# Windows (PowerShell)
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Install project dependencies:

```bash
uv sync
```

### 3. Configure Environment Variables

Copy the sample environment file and fill in your Azure credentials:

```bash
cp .env.sample .env
```

Edit `.env` file with your Azure resource details:

```env
# Azure AI Foundry
AI_FOUNDRY_PROJECT_ENDPOINT=https://<your-project>.services.ai.azure.com/api/projects/<project-name>
AI_FOUNDRY_OPENAI_ENDPOINT=https://<your-resource>.openai.azure.com/
AI_FOUNDRY_API_KEY=<your-api-key>

# Azure OpenAI
AZURE_AI_API_VERSION=2025-04-01-preview
GPT_MODEL_DEPLOYMENT_NAME=gpt-4.1
EMBEDDING_DEPLOYMENT_NAME=text-embedding-ada-002

# Azure AI Search
AI_SEARCH_SERVICE_ENDPOINT=https://<your-search-service>.search.windows.net
AI_SEARCH_API_KEY=<your-search-api-key>
AI_SEARCH_INDEX_NAME=<your-index-name>

# PostgreSQL (optional)
PG_HOST=<your-postgres-host>
PG_DATABASE=postgres
PG_USERNAME=<username>
PG_PASSWORD=<password>
```

### 4. Activate Virtual Environment

```bash
# Windows PowerShell
.\.venv\Scripts\Activate.ps1

# macOS/Linux
source .venv/bin/activate
```

## Running the Application

### Start the Streamlit App

```bash
streamlit run src/app/main.py
```

Or using Python module:

```bash
python -m streamlit run src/app/main.py
```

The application will open in your default browser at `http://localhost:8501`.

### Application Pages

1. **Home** (`main.py`): Overview and navigation
2. **Build Knowledge Base** (`pages/0_📘_Build_Knowledge_base.py`): Upload and index PDF documents
3. **AI Search RAG** (`pages/1_🔎_AISearch_RAG.py`): Query the knowledge base with AI-powered answers

## Project Structure

```
ticketier/
├── src/
│   ├── app/
│   │   ├── main.py              # Main Streamlit app entry point
│   │   └── pages/               # Streamlit pages
│   │       ├── 0_📘_Build_Knowledge_base.py
│   │       └── 1_🔎_AISearch_RAG.py
│   ├── core/
│   │   ├── azure_search_helper.py  # Azure AI Search helper class
│   │   └── __init__.py
│   ├── scripts/
│   │   └── azure_search_example.py  # Example usage scripts
│   └── notebooks/               # Jupyter notebooks
├── infra/                       # Infrastructure as Code
├── .env                         # Environment variables (not in git)
├── .env.sample                  # Environment template
├── pyproject.toml               # Project dependencies
└── README.md
```

## Usage Examples

### Programmatic Usage

```python
from core.azure_search_helper import AzureSearchHelper

# Initialize helper (reads from environment variables)
helper = AzureSearchHelper()

# Process and index a PDF
result = helper.process_and_index_pdf(
    pdf_path="documents/sample.pdf",
    index_name="my-knowledge-base",
    title="Sample Document",
    chunking_strategy="page"
)

# Perform vector search
search_results = helper.vector_search(
    index_name="my-knowledge-base",
    query_text="What is machine learning?",
    top_k=5
)

# RAG query with AI-generated answer
rag_response = helper.rag_query(
    index_name="my-knowledge-base",
    query="Explain the key concepts from the document",
    top_k=3
)

print(rag_response['answer'])
print("Sources:", rag_response['sources'])
```

## Key Features

### PDF Processing
- Extract text from PDF documents
- Chunk by page or by section/headers
- Generate embeddings using Azure OpenAI

### Vector Search
- HNSW algorithm for efficient similarity search
- Configurable vector dimensions (default: 1536)
- Semantic search capabilities

### RAG Pipeline
- Retrieves relevant context from knowledge base
- Generates answers using GPT models
- Provides source citations for transparency

## Development

### Install Development Dependencies

```bash
uv sync --all-extras
```

### Run Tests

```bash
pytest
```

### Code Formatting

```bash
black src/
ruff check src/
```

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure `PYTHONPATH=.` is set in your `.env` file
2. **Authentication Errors**: Verify your API keys and endpoints in `.env`
3. **Model Not Found**: Check that your model deployment names match those in Azure OpenAI
4. **Index Errors**: Ensure the search index exists or set `create_index=True`

### Logs

Enable verbose logging by setting:

```env
AZURE_TRACING_GEN_AI_CONTENT_RECORDING_ENABLED=true
OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=true
```

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

[Specify your license here]

## Support

For issues and questions:
- Create an issue in the GitHub repository
- Check Azure AI documentation: https://learn.microsoft.com/azure/ai-services/