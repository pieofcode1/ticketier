"""
Databricks SQL Page - Natural Language to Databricks SQL

This page allows users to ask questions in natural language,
translates them to SQL using Azure OpenAI, executes on a
Databricks SQL Warehouse, and displays the results.
"""

import os
import streamlit as st
import pandas as pd
from dotenv import load_dotenv
from openai import AzureOpenAI
from databricks import sql as databricks_sql

load_dotenv()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a SQL expert. Given the user's natural language question and the table schema below, generate a valid Databricks SQL query that answers the question.

Rules:
- Return ONLY the SQL query, no explanation, no markdown fences.
- Use standard Databricks SQL syntax.
- Do not use INSERT, UPDATE, DELETE, DROP, or any DDL/DML that modifies data.
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


def get_databricks_connection():
    """Create a Databricks SQL connection."""
    return databricks_sql.connect(
        server_hostname=os.getenv("DATABRICKS_SERVER_HOSTNAME"),
        http_path=os.getenv("DATABRICKS_HTTP_PATH"),
        access_token=os.getenv("DATABRICKS_ACCESS_TOKEN"),
    )


def fetch_schema(catalog: str, schema: str) -> str:
    """Fetch table and column metadata from Databricks information_schema."""
    query = f"""
        SELECT table_name, column_name, data_type
        FROM {catalog}.information_schema.columns
        WHERE table_schema = '{schema}'
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


def nl_to_sql(question: str, schema_text: str) -> str:
    """Translate natural language to SQL using Azure OpenAI."""
    client = get_openai_client()
    model = os.getenv("GPT_MODEL_DEPLOYMENT_NAME", "gpt-5-mini")

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT.format(schema=schema_text)},
            {"role": "user", "content": question},
        ],
        max_completion_tokens=1024,
    )
    return response.choices[0].message.content.strip()


def execute_sql(sql: str) -> pd.DataFrame:
    """Execute SQL on Databricks and return a DataFrame."""
    with get_databricks_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(sql)
            columns = [desc[0] for desc in cursor.description]
            rows = cursor.fetchall()
    return pd.DataFrame(rows, columns=columns)


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------

def main():
    st.set_page_config(page_title="Databricks SQL", page_icon="🧱", layout="wide")

    st.header("Databricks SQL 🧱", divider="orange")
    st.caption("Ask questions in plain English — they'll be translated to SQL and run on your Databricks workspace.")

    # Session state
    if "dbx_chat_history" not in st.session_state:
        st.session_state.dbx_chat_history = []
    if "dbx_schema_text" not in st.session_state:
        st.session_state.dbx_schema_text = None

    # Sidebar: connection settings
    with st.sidebar:
        st.subheader("🔗 Databricks Connection")

        catalog = st.text_input("Catalog", value=os.getenv("DATABRICKS_CATALOG", "main"))
        schema = st.text_input("Schema", value=os.getenv("DATABRICKS_SCHEMA", "default"))

        if st.button("🔄 Load Schema", use_container_width=True):
            with st.spinner("Fetching schema from Databricks..."):
                try:
                    st.session_state.dbx_schema_text = fetch_schema(catalog, schema)
                    st.success("Schema loaded!")
                except Exception as e:
                    st.error(f"Failed to fetch schema: {e}")

        if st.session_state.dbx_schema_text:
            with st.expander("📋 Table Schema", expanded=False):
                st.code(st.session_state.dbx_schema_text, language="text")

        st.divider()
        if st.session_state.dbx_chat_history:
            if st.button("🗑️ Clear History", use_container_width=True):
                st.session_state.dbx_chat_history = []
                st.rerun()

    # Main area
    if not st.session_state.dbx_schema_text:
        st.info("👈 Configure your Databricks catalog/schema in the sidebar and click **Load Schema** to get started.")
        return

    # Chat input
    user_question = st.chat_input("Ask a question about your data...")

    if user_question:
        with st.spinner("Generating SQL..."):
            generated_sql = nl_to_sql(user_question, st.session_state.dbx_schema_text)

        if "-- CANNOT_ANSWER" in generated_sql:
            st.warning("The model could not generate a query for this question based on the available schema.")
            return

        # Execute
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
        })

    # Display history (latest first)
    for entry in reversed(st.session_state.dbx_chat_history):
        with st.chat_message("user"):
            st.write(entry["question"])
        with st.chat_message("assistant"):
            st.code(entry["sql"], language="sql")
            if entry.get("error"):
                st.error(f"Query error: {entry['error']}")
            elif entry.get("result") is not None:
                st.dataframe(entry["result"], use_container_width=True)


if __name__ == "__main__":
    main()
