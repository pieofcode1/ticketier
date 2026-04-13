# Project Guidelines — Ticketier

## Overview

Ticketier is an agentic AI application built on **Azure AI Foundry** for intelligent ticket analysis and RAG-based knowledge retrieval. It combines a Streamlit multi-page UI, Azure OpenAI (Responses API), Azure AI Search, PostgreSQL + pgvector, and the Microsoft Agent Framework.

## Code Style

- **Python 3.12+**, managed with **`uv`** (no `requirements.txt`; use `pyproject.toml` + `uv sync`)
- Type hints everywhere: use `Optional`, `List`, `Dict`, `Annotated`, `Literal` from `typing`
- Dataclasses (`@dataclass`) for data containers; Enums for type safety
- Google-style docstrings with Args/Returns/Raises on all public classes and methods
- See [src/core/azure_search_helper.py](src/core/azure_search_helper.py) and [src/core/openai_manager.py](src/core/openai_manager.py) for canonical patterns

## Architecture

```
src/
├── app/              # Streamlit UI (main.py + pages/)
├── assistants/       # Agent definitions (Microsoft Agent Framework)
├── core/             # Reusable service helpers (Azure Search, OpenAI, PostgreSQL, chat history)
├── data/             # CSV ticket datasets loaded into PostgreSQL
├── history/          # JSON chat history files (auto-generated)
├── mcp/              # Model Context Protocol servers (placeholder)
├── notebooks/        # Jupyter exploration/prototyping notebooks
└── scripts/          # Standalone example scripts
```

- **Core layer** (`src/core/`): Stateless helper classes wrapping Azure services — `AzureSearchHelper`, `OpenAIManager`, `PGHelper`, `ChatHistoryHelper`. Each accepts explicit constructor params or falls back to environment variables.
- **UI layer** (`src/app/pages/`): Streamlit pages use `st.session_state` for persistence and `@st.cache_resource` for expensive initializations.
- **Agent layer** (`src/assistants/`): Agents use `agent_framework.azure.AzureAIClient` with function tools. Tool functions use `Annotated[str, Field(description=...)]` for parameter descriptions.

### Key data flows

1. **Knowledge Base**: PDF upload → PyPDF2 extraction → chunking → Azure OpenAI embeddings → Azure AI Search index
2. **RAG Query**: User query → vector search (HNSW cosine) → context assembly → GPT completion → answer with sources
3. **Ticket Search**: Query → `pg_helper.search_similar_issues()` → pgvector cosine similarity → results
4. **Agent**: `AzureAIClient.create_agent()` with function tools → `agent.run()` or `agent.run_stream()`

## Build and Test

```bash
uv sync                          # Install all dependencies
streamlit run src/app/main.py    # Run the Streamlit app
pytest                           # Run tests (framework in place, tests TBD)
```

- Copy `.env.sample` to `.env` and fill in Azure credentials before running
- **`PYTHONPATH=.`** must be set in `.env` for `src.*` imports to resolve

## Project Conventions

- **Environment-first config**: All Azure endpoints/keys come from environment variables via `os.getenv()`, never hardcoded. Every helper class supports both API key and `DefaultAzureCredential` (toggle via `USE_AZURE_CREDENTIAL` env var).
- **OpenAI Responses API**: `OpenAIManager` uses the newer Responses API (not Chat Completions). See [src/core/openai_manager.py](src/core/openai_manager.py) for the custom `base_url` construction and `api_version="preview"` usage.
- **Agent function tools**: Expose callable functions via a `user_functions` set. Parameters must use `Annotated[type, Field(description="...")]`. See [src/core/pg_helper.py](src/core/pg_helper.py) for the pattern.
- **Streamlit pages**: Prefix filenames with number + emoji (`0_📘_`, `1_🔎_`, `2_🖼️_`). Always call `load_dotenv()` at the top. Use `st.session_state` for chat messages and UI state.
- **Chat history**: Persisted as timestamped JSON files in `src/history/` via `ChatHistoryHelper`.

## Integration Points

| Service | SDK | Config Env Vars |
|---------|-----|-----------------|
| Azure OpenAI | `openai` | `AI_FOUNDRY_OPENAI_ENDPOINT`, `AI_FOUNDRY_API_KEY`, `GPT_MODEL_DEPLOYMENT_NAME`, `EMBEDDING_DEPLOYMENT_NAME` |
| Azure AI Search | `azure-search-documents` | `AI_SEARCH_SERVICE_ENDPOINT`, `AI_SEARCH_API_KEY`, `AI_SEARCH_INDEX_NAME` |
| Azure AI Foundry | `azure-ai-projects` | `AI_FOUNDRY_PROJECT_ENDPOINT` |
| PostgreSQL + pgvector | `psycopg2-binary` | `PG_HOST`, `PG_DATABASE`, `PG_USERNAME`, `PG_PASSWORD` |
| Azure Identity | `azure-identity` | `USE_AZURE_CREDENTIAL` (toggles DefaultAzureCredential vs API key) |

## Security

- **Never hardcode credentials** — all secrets must come from environment variables or Azure Identity
- `.env` is gitignored; `.env.sample` is the committed template
- Dual auth support: API key for local dev, `DefaultAzureCredential` for deployed environments
- The Eddie agent architecture enforces **human-in-the-loop** for money movement and cancellations — AI must never auto-execute financial actions (see [docs/eddie-use-case.md](docs/eddie-use-case.md))
