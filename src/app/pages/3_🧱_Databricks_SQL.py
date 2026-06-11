"""
Databricks Integration Page - SQL & Vector Search

This page allows users to:
1. Ask questions in natural language → translated to SQL → executed on Databricks
2. Perform semantic/vector search against Databricks Vector Search indexes
"""

import os
import streamlit as st
import pandas as pd
from dotenv import load_dotenv
from openai import AzureOpenAI
from databricks import sql as databricks_sql
from databricks.ai_search.client import VectorSearchClient
from databricks.sdk.core import Config

load_dotenv()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SQL_SYSTEM_PROMPT = """You are a SQL expert. Given the user's natural language question and the table schema below, generate a valid Databricks SQL query that answers the question.

Rules:
- Return ONLY the SQL query, no explanation, no markdown fences.
- Use standard Databricks SQL syntax.
- Do not use INSERT, UPDATE, DELETE, DROP, or any DDL/DML that modifies data.
- Do NOT query tables ending with '_index' — those are search indexes, not queryable tables.
- If the question cannot be answered from the schema, reply with: -- CANNOT_ANSWER

Schema:
{schema}
"""


@st.cache_resource
def get_openai_client():
    """Initialize and cache the Azure OpenAI client."""
    return AzureOpenAI(
        azure_endpoint=os.getenv("AI_FOUNDRY_OPENAI_ENDPOINT"),
        api_key=os.getenv("AI_FOUNDRY_API_KEY"),
        api_version=os.getenv("AZURE_AI_API_VERSION", "2025-04-01-preview"),
    )


def _databricks_sp_creds() -> tuple[str, str, str]:
    """Return (client_id, client_secret, tenant_id) for an Azure Entra Service Principal.

    Raises if any required value is missing.
    """
    client_id = os.getenv("DATABRICKS_CLIENT_ID")
    client_secret = os.getenv("DATABRICKS_CLIENT_SECRET")
    tenant_id = os.getenv("DATABRICKS_TENANT_ID")
    if not client_id or not client_secret or not tenant_id:
        raise RuntimeError(
            "Azure Entra Service Principal credentials are required. "
            "Set DATABRICKS_CLIENT_ID, DATABRICKS_CLIENT_SECRET, and "
            "DATABRICKS_TENANT_ID in .env."
        )
    return client_id, client_secret, tenant_id


def get_databricks_connection():
    """Create a Databricks SQL connection using an Azure Entra Service Principal."""
    server_hostname = os.getenv("DATABRICKS_SERVER_HOSTNAME")
    http_path = os.getenv("DATABRICKS_HTTP_PATH")
    client_id, client_secret, tenant_id = _databricks_sp_creds()

    def credential_provider():
        cfg = Config(
            host=f"https://{server_hostname}",
            azure_client_id=client_id,
            azure_client_secret=client_secret,
            azure_tenant_id=tenant_id,
            auth_type="azure-client-secret",
        )
        return cfg.authenticate

    return databricks_sql.connect(
        server_hostname=server_hostname,
        http_path=http_path,
        credentials_provider=credential_provider,
    )


def get_vector_search_client():
    """Create a Databricks Vector Search client using an Azure Entra Service Principal."""
    workspace_url = f"https://{os.getenv('DATABRICKS_SERVER_HOSTNAME')}"
    client_id, client_secret, tenant_id = _databricks_sp_creds()

    return VectorSearchClient(
        workspace_url=workspace_url,
        service_principal_client_id=client_id,
        service_principal_client_secret=client_secret,
        azure_tenant_id=tenant_id,
        disable_notice=True,
    )


def fetch_schema(catalog: str, schema: str) -> str:
    """Fetch table and column metadata from Databricks information_schema."""
    query = f"""
        SELECT table_name, column_name, data_type
        FROM {catalog}.information_schema.columns
        WHERE table_schema = '{schema}'
          AND table_name NOT LIKE '%\\_index' ESCAPE '\\\\'
        ORDER BY table_name, ordinal_position
    """
    with get_databricks_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(query)
            rows = cursor.fetchall()

    if not rows:
        return "No tables found."

    lines = []
    current_table = None
    for table_name, column_name, data_type in rows:
        if table_name != current_table:
            current_table = table_name
            lines.append(f"\nTable: {catalog}.{schema}.{table_name}")
            lines.append("  Columns:")
        lines.append(f"    - {column_name} ({data_type})")

    return "\n".join(lines)


def fetch_vector_indexes(endpoint_name: str) -> list[dict]:
    """Fetch vector search indexes for the given endpoint."""
    vs_client = get_vector_search_client()
    indexes = vs_client.list_indexes(name=endpoint_name)
    return indexes.get("vector_indexes", [])


def _infer_field_type(value: dict) -> str:
    """Infer column type from a Protobuf-style scan field value dict."""
    if not isinstance(value, dict):
        return type(value).__name__
    if "string_value" in value:
        return "string"
    if "number_value" in value:
        return "number"
    if "bool_value" in value:
        return "bool"
    if "list_value" in value:
        return "vector"
    if "null_value" in value:
        return "string"  # nullable; assume string until a non-null row reveals otherwise
    return "unknown"


def _discover_index_columns(index) -> tuple[list[dict], str | None]:
    """Discover the actual columns present in a vector index by scanning 1 row.

    The databricks-ai-search SDK returns scan() results in Protobuf JSON form:
        {"data": [{"fields": [{"key": "<col>", "value": {<type>_value: ...}}, ...]}], ...}

    Also handles two legacy shapes for forward-compat. Returns (columns, error).
    """
    try:
        scan_result = index.scan(num_results=1)
    except Exception as e:
        return [], f"scan() failed: {e}"

    if not isinstance(scan_result, dict):
        return [], f"scan() returned {type(scan_result).__name__}"

    data = scan_result.get("data", [])

    # Format A (current SDK): data[0].fields is a list of {key, value: {<type>_value: ...}}
    if data and isinstance(data[0], dict) and isinstance(data[0].get("fields"), list):
        cols: list[dict] = []
        for f in data[0]["fields"]:
            name = f.get("key")
            if not name:
                continue
            cols.append({"name": name, "type": _infer_field_type(f.get("value", {}))})
        if cols:
            return cols, None

    # Format B (legacy): data[0] is a flat {col: val} dict
    if data and isinstance(data[0], dict) and data[0]:
        return ([{"name": k, "type": type(v).__name__} for k, v in data[0].items()], None)

    # Format C (similarity_search-style): {"manifest": {"columns": [...]}}
    manifest_cols = scan_result.get("manifest", {}).get("columns", [])
    if manifest_cols:
        return (
            [{"name": c.get("name", ""), "type": c.get("type", "?")} for c in manifest_cols if c.get("name")],
            None,
        )

    return [], f"scan() returned unknown format; keys={list(scan_result.keys())}"


def fetch_index_details(endpoint_name: str, index_name: str) -> dict:
    """Fetch detailed metadata (schema, status) for a single vector index."""
    try:
        vs_client = get_vector_search_client()
        index = vs_client.get_index(endpoint_name=endpoint_name, index_name=index_name)
        desc = index.describe()
        # Try multiple known locations for the indexed-column list before scanning.
        delta_spec = desc.get("delta_sync_index_spec", {}) or {}
        direct_spec = desc.get("direct_access_index_spec", {}) or {}
        schema_cols = (
            delta_spec.get("columns_to_sync")
            or desc.get("schema")
            or desc.get("columns")
            or direct_spec.get("schema", {}).get("columns")
            or []
        )
        if not schema_cols:
            actual_cols, scan_err = _discover_index_columns(index)
            if actual_cols:
                desc.setdefault("delta_sync_index_spec", {})["columns_to_sync"] = actual_cols
            elif scan_err:
                desc["_column_discovery_error"] = scan_err
        return desc
    except Exception as e:
        return {"name": index_name, "error": str(e)}


def nl_to_sql(question: str, schema_text: str) -> str:
    """Translate natural language to SQL using Azure OpenAI."""
    client = get_openai_client()
    model = os.getenv("GPT_MODEL_DEPLOYMENT_NAME", "gpt-5-mini")

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SQL_SYSTEM_PROMPT.format(schema=schema_text)},
            {"role": "user", "content": question},
        ],
        max_completion_tokens=1024,
    )
    raw = response.choices[0].message.content.strip()
    # Strip markdown code fences if present
    if raw.startswith("```"):
        lines = raw.splitlines()
        lines = [l for l in lines if not l.strip().startswith("```")]
        raw = "\n".join(lines).strip()
    return raw


def execute_sql(sql: str) -> pd.DataFrame:
    """Execute SQL on Databricks and return a DataFrame."""
    with get_databricks_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(sql)
            columns = [desc[0] for desc in cursor.description]
            rows = cursor.fetchall()
    return pd.DataFrame(rows, columns=columns)


def vector_search(index_name: str, endpoint_name: str, query_text: str,
                  num_results: int = 10, columns: list[str] | None = None) -> pd.DataFrame:
    """Perform similarity search on a Databricks Vector Search index."""
    vs_client = get_vector_search_client()
    index = vs_client.get_index(
        endpoint_name=endpoint_name,
        index_name=index_name,
    )
    # columns is required by the API; if not provided, derive from index schema
    if not columns:
        desc = index.describe()
        delta_spec = desc.get("delta_sync_index_spec", {})
        direct_spec = desc.get("direct_access_index_spec", {})
        schema_cols = delta_spec.get("columns_to_sync", [])
        if not schema_cols:
            schema_cols = direct_spec.get("schema", {}).get("columns", [])
        if schema_cols:
            if isinstance(schema_cols, list) and schema_cols and isinstance(schema_cols[0], dict):
                columns = [c.get("name", "") for c in schema_cols if c.get("name")]
            else:
                columns = [str(c) for c in schema_cols]
        # Still empty? Discover from a 1-row scan of the index
        if not columns:
            actual_cols, _ = _discover_index_columns(index)
            columns = [c["name"] for c in actual_cols if c.get("name")]
        # Filter out the embedding vector column and internal CDC columns —
        # similarity_search rejects these as requested return columns.
        columns = [
            c for c in (columns or [])
            if not c.startswith("_") and not c.endswith("_vector")
        ]
    if not columns:
        raise ValueError("Could not determine columns for vector search. Please specify columns explicitly.")

    # Embed the query locally with Azure OpenAI and pass query_vector instead of
    # query_text. This avoids a flaky internal token-exchange path in the
    # Vector Search SDK that intermittently fails under Streamlit's threaded
    # rerun model with "Token exchange failed ... HTTP client is closing".
    embedding_deployment = os.getenv("EMBEDDING_DEPLOYMENT_NAME", "text-embedding-ada-002")
    embed_resp = get_openai_client().embeddings.create(
        model=embedding_deployment,
        input=query_text,
    )
    query_vector = embed_resp.data[0].embedding

    results = index.similarity_search(
        query_vector=query_vector,
        columns=columns,
        num_results=num_results,
    )
    # results is a dict with 'manifest' and 'result' keys
    col_names = [c["name"] for c in results.get("manifest", {}).get("columns", [])]
    data_rows = results.get("result", {}).get("data_array", [])
    return pd.DataFrame(data_rows, columns=col_names)


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------

def main():
    st.set_page_config(page_title="Databricks Integration", page_icon="🧱", layout="wide")

    st.header("Databricks Integration 🧱", divider="orange")
    st.caption("Query your Databricks workspace with natural language SQL or semantic vector search.")

    # Session state
    if "dbx_chat_history" not in st.session_state:
        st.session_state.dbx_chat_history = []
    if "dbx_schema_text" not in st.session_state:
        st.session_state.dbx_schema_text = None
    if "dbx_vector_indexes" not in st.session_state:
        st.session_state.dbx_vector_indexes = []
    if "dbx_index_details" not in st.session_state:
        st.session_state.dbx_index_details = {}

    # Sidebar: connection settings
    with st.sidebar:
        st.subheader("🔗 Databricks Connection")

        catalog = st.text_input("Catalog", value=os.getenv("DATABRICKS_CATALOG", "main"))
        schema = st.text_input("Schema", value=os.getenv("DATABRICKS_SCHEMA", "default"))
        vs_endpoint = st.text_input(
            "Vector Search Endpoint",
            value=os.getenv("DATABRICKS_VS_ENDPOINT", "vs_endpoint"),
        )

        if st.button("🔄 Load Schema & Indexes", use_container_width=True):
            with st.spinner("Fetching schema and vector indexes from Databricks..."):
                # Load table schema
                try:
                    st.session_state.dbx_schema_text = fetch_schema(catalog, schema)
                except Exception as e:
                    st.error(f"Failed to fetch table schema: {e}")

                # Load vector indexes from endpoint
                try:
                    st.session_state.dbx_vector_indexes = fetch_vector_indexes(vs_endpoint)
                    # Fetch details (schema) for each discovered index
                    details = {}
                    for idx in st.session_state.dbx_vector_indexes:
                        idx_name = idx.get("name", "") if isinstance(idx, dict) else str(idx)
                        if idx_name:
                            details[idx_name] = fetch_index_details(vs_endpoint, idx_name)
                    st.session_state.dbx_index_details = details
                except Exception as e:
                    st.warning(f"Could not list vector indexes from endpoint '{vs_endpoint}': {e}")
                    st.session_state.dbx_vector_indexes = []
                    st.session_state.dbx_index_details = {}

                st.success("Loaded!")

        if st.session_state.dbx_schema_text:
            with st.expander("📋 Table Schema", expanded=False):
                st.code(st.session_state.dbx_schema_text, language="text")

        if st.session_state.dbx_index_details:
            with st.expander("📋 Index Schema", expanded=True):
                schema_lines = []
                for idx_name, detail in st.session_state.dbx_index_details.items():
                    if "error" in detail:
                        schema_lines.append(f"Index: {idx_name}")
                        schema_lines.append(f"  ERROR: {detail['error']}")
                        schema_lines.append("")
                        continue
                    index_type = detail.get("index_type", "N/A")
                    status = detail.get("status", {}).get("ready", "unknown")
                    pk = detail.get("primary_key", "")
                    schema_lines.append(f"Index: {idx_name}")
                    schema_lines.append(f"  Type: {index_type} | Ready: {status} | PK: {pk}")
                    schema_lines.append("  Columns:")
                    # Extract columns from the index spec
                    delta_spec = detail.get("delta_sync_index_spec", {})
                    direct_spec = detail.get("direct_access_index_spec", {})
                    schema_cols = delta_spec.get("columns_to_sync", [])
                    if not schema_cols:
                        schema_cols = direct_spec.get("schema", {}).get("columns", [])
                    if schema_cols:
                        if isinstance(schema_cols, list) and schema_cols and isinstance(schema_cols[0], dict):
                            for c in schema_cols:
                                schema_lines.append(f"    - {c.get('name', '?')} ({c.get('type', '?')})")
                        else:
                            for c in schema_cols:
                                schema_lines.append(f"    - {c}")
                    else:
                        err = detail.get("_column_discovery_error")
                        if err:
                            schema_lines.append(f"    (column discovery failed: {err})")
                        else:
                            schema_lines.append("    (no column list available)")
                    # Show embedding source/vector info
                    emb_src = delta_spec.get("embedding_source_columns", [])
                    emb_vec = direct_spec.get("embedding_vector_columns", [])
                    if emb_src:
                        for e in emb_src:
                            if isinstance(e, dict):
                                schema_lines.append(f"    [embedding source] {e.get('name', '?')} → model: {e.get('embedding_model_endpoint_name', '?')}")
                            else:
                                schema_lines.append(f"    [embedding source] {e}")
                    if emb_vec:
                        for e in emb_vec:
                            if isinstance(e, dict):
                                schema_lines.append(f"    [embedding vector] {e.get('name', '?')} (dim: {e.get('embedding_dimension', '?')})")
                            else:
                                schema_lines.append(f"    [embedding vector] {e}")
                    schema_lines.append("")
                st.code("\n".join(schema_lines), language="text")

            with st.expander("🐛 Raw describe() output (debug)", expanded=False):
                st.json(st.session_state.dbx_index_details)

        st.markdown("---")
        if st.session_state.dbx_chat_history:
            if st.button("🗑️ Clear History", use_container_width=True):
                st.session_state.dbx_chat_history = []
                st.rerun()

    # Main area — mode selector
    mode = st.radio(
        "Query Mode",
        ["💬 SQL (Natural Language → SQL)", "🔍 Vector Search (Semantic)"],
        horizontal=True,
    )

    if mode.startswith("💬"):
        _render_sql_mode()
    else:
        _render_vector_mode(vs_endpoint)

    # Display history (latest first)
    if st.session_state.dbx_chat_history:
        st.divider()
        for entry in reversed(st.session_state.dbx_chat_history):
            with st.chat_message("user"):
                st.write(entry["question"])
            with st.chat_message("assistant"):
                if entry.get("sql"):
                    st.code(entry["sql"], language="sql")
                if entry.get("mode") == "vector":
                    st.caption(f"🔍 Vector search on `{entry.get('index_name')}`")
                if entry.get("error"):
                    st.error(f"Query error: {entry['error']}")
                elif entry.get("result") is not None:
                    st.dataframe(entry["result"], use_container_width=True)


def _render_sql_mode():
    """Render the SQL natural-language query mode."""
    if not st.session_state.dbx_schema_text:
        st.info("👈 Load the schema from the sidebar first.")
        return

    user_question = st.chat_input("Ask a question about your data (SQL mode)...")

    if user_question:
        with st.spinner("Generating SQL..."):
            generated_sql = nl_to_sql(user_question, st.session_state.dbx_schema_text)

        if "-- CANNOT_ANSWER" in generated_sql or not generated_sql:
            st.warning("The model could not generate a query for this question based on the available schema.")
            return

        result_df = None
        error = None
        with st.spinner("Running query on Databricks..."):
            try:
                result_df = execute_sql(generated_sql)
            except Exception as e:
                error = str(e)

        st.session_state.dbx_chat_history.append({
            "question": user_question,
            "sql": generated_sql,
            "result": result_df,
            "error": error,
            "mode": "sql",
        })
        st.rerun()


def _render_vector_mode(vs_endpoint: str):
    """Render the vector search mode."""
    # Build index options from loaded details
    loaded_indexes = list(st.session_state.get("dbx_index_details", {}).keys())
    default_index = os.getenv("DATABRICKS_VS_INDEX", "")

    if loaded_indexes:
        options = loaded_indexes
        default_idx = options.index(default_index) if default_index in options else 0
        index_name = st.selectbox("Vector Index", options=options, index=default_idx)
    else:
        index_name = st.text_input(
            "Index Name",
            value=default_index,
            placeholder="catalog.schema.index_name",
            help="Full three-level name of the vector index. Load index schema from sidebar first.",
        )

    # Show schema for selected index inline
    if index_name and index_name in st.session_state.get("dbx_index_details", {}):
        detail = st.session_state.dbx_index_details[index_name]
        if "error" not in detail:
            delta_spec = detail.get("delta_sync_index_spec", {})
            direct_spec = detail.get("direct_access_index_spec", {})
            schema_cols = delta_spec.get("columns_to_sync", [])
            if not schema_cols:
                schema_cols = direct_spec.get("schema", {}).get("columns", [])
            if schema_cols:
                if isinstance(schema_cols, list) and schema_cols and isinstance(schema_cols[0], dict):
                    col_names = [c.get("name", "") for c in schema_cols if c.get("name")]
                else:
                    col_names = [str(c) for c in schema_cols]
                with st.expander(f"📋 Columns in `{index_name}`", expanded=False):
                    st.code("\n".join(f"- {c}" for c in col_names), language="text")

    # Pre-fill columns from index schema
    suggested_cols = ""
    if index_name and index_name in st.session_state.get("dbx_index_details", {}):
        detail = st.session_state.dbx_index_details[index_name]
        delta_spec = detail.get("delta_sync_index_spec", {})
        direct_spec = detail.get("direct_access_index_spec", {})
        schema_cols = delta_spec.get("columns_to_sync", [])
        if not schema_cols:
            schema_cols = direct_spec.get("schema", {}).get("columns", [])
        if schema_cols:
            if isinstance(schema_cols, list) and schema_cols and isinstance(schema_cols[0], dict):
                suggested_cols = ", ".join(c.get("name", "") for c in schema_cols if c.get("name"))
            else:
                suggested_cols = ", ".join(str(c) for c in schema_cols)

    col_input = st.text_input(
        "Columns to return (comma-separated, blank = all from schema)",
        value=suggested_cols,
        help="Columns to include in results. Leave blank to auto-detect from index schema.",
    )
    num_results = st.slider("Number of results", min_value=1, max_value=50, value=10)

    user_question = st.chat_input("Search for similar documents (Vector mode)...")

    if user_question:
        if not index_name:
            st.warning("Please specify a vector index name.")
            return

        columns = [c.strip() for c in col_input.split(",") if c.strip()] if col_input else None

        result_df = None
        error = None
        with st.spinner("Searching vector index..."):
            try:
                result_df = vector_search(
                    index_name=index_name,
                    endpoint_name=vs_endpoint,
                    query_text=user_question,
                    num_results=num_results,
                    columns=columns,
                )
            except Exception as e:
                error = str(e)

        st.session_state.dbx_chat_history.append({
            "question": user_question,
            "sql": None,
            "result": result_df,
            "error": error,
            "mode": "vector",
            "index_name": index_name,
        })
        st.rerun()


if __name__ == "__main__":
    main()
