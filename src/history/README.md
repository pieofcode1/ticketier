# Chat History Storage

This folder stores saved chat histories from the Ticketier application.

Files are saved in JSON format with the following naming convention:
- `vision_rag_YYYYMMDD_HHMMSS.json` - Vision RAG chat histories
- `aisearch_rag_YYYYMMDD_HHMMSS.json` - AI Search RAG chat histories

Each file contains:
- `page_type`: The type of page that generated the history
- `timestamp`: When the history was saved
- `metadata`: Additional context (image name, index name, etc.)
- `messages`: The chat message history
