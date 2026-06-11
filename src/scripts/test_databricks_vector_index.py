"""
Probe a Databricks Vector Search index to discover what columns it actually contains.

Uses env vars from .env:
  DATABRICKS_SERVER_HOSTNAME
  DATABRICKS_CLIENT_ID
  DATABRICKS_CLIENT_SECRET
  DATABRICKS_TENANT_ID       (optional; only for Azure Entra SP)
  DATABRICKS_VS_ENDPOINT
  DATABRICKS_VS_INDEX        (optional; if absent, lists all indexes on the endpoint)

Run:
  python -m src.scripts.test_databricks_vector_index
  python -m src.scripts.test_databricks_vector_index manufacturing.bronze.warranty_claims_index
"""

import json
import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from databricks.ai_search.client import VectorSearchClient

load_dotenv()

# Force UTF-8 stdout on Windows so non-ASCII chars don't crash printing
try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except Exception:
    pass

OUT_DIR = Path("src/scripts/_vector_probe_out")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _safe(name: str) -> str:
    return name.replace("/", "_").replace(".", "_")


def _write_json(filename: str, obj) -> Path:
    path = OUT_DIR / filename
    path.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")
    return path


def _dump(label: str, obj) -> None:
    print(f"\n--- {label} ---")
    try:
        print(json.dumps(obj, indent=2, default=str)[:2000])
        if isinstance(obj, (dict, list)) and len(json.dumps(obj, default=str)) > 2000:
            print("  ... (truncated; full output written to file)")
    except Exception:
        print(repr(obj))


def get_client() -> VectorSearchClient:
    host = os.getenv("DATABRICKS_SERVER_HOSTNAME")
    if not host:
        raise SystemExit("DATABRICKS_SERVER_HOSTNAME must be set in .env")

    client_id = os.getenv("DATABRICKS_CLIENT_ID")
    client_secret = os.getenv("DATABRICKS_CLIENT_SECRET")
    tenant_id = os.getenv("DATABRICKS_TENANT_ID")

    if not client_id or not client_secret or not tenant_id:
        raise SystemExit(
            "DATABRICKS_CLIENT_ID, DATABRICKS_CLIENT_SECRET, and DATABRICKS_TENANT_ID "
            "must be set in .env (Azure Entra Service Principal auth is required)."
        )

    print(f"Auth: Azure Entra SP (client_id={client_id[:8]}..., tenant={tenant_id})")
    return VectorSearchClient(
        workspace_url=f"https://{host}",
        service_principal_client_id=client_id,
        service_principal_client_secret=client_secret,
        azure_tenant_id=tenant_id,
        disable_notice=True,
    )


def list_indexes(client: VectorSearchClient, endpoint: str) -> list[str]:
    raw = client.list_indexes(name=endpoint)
    _dump(f"list_indexes('{endpoint}') raw", raw)
    items = raw.get("vector_indexes", []) if isinstance(raw, dict) else []
    names = [i.get("name") for i in items if isinstance(i, dict) and i.get("name")]
    print(f"\nFound {len(names)} index(es) on endpoint '{endpoint}':")
    for n in names:
        print(f"  - {n}")
    return names


def probe_index(client: VectorSearchClient, endpoint: str, index_name: str) -> None:
    print(f"\n{'=' * 70}\nProbing index: {index_name}\n{'=' * 70}")
    index = client.get_index(endpoint_name=endpoint, index_name=index_name)

    # 1. describe()
    try:
        desc = index.describe()
        _dump("describe()", desc)
        _write_json(f"{_safe(index_name)}__describe.json", desc)
    except Exception as e:
        print(f"\ndescribe() FAILED: {e}")
        desc = {}

    # 2. Hunt for column lists in the description
    print("\n--- column-list candidates inside describe() ---")
    candidates = {
        "delta_sync_index_spec.columns_to_sync": desc.get("delta_sync_index_spec", {}).get("columns_to_sync"),
        "delta_sync_index_spec.source_table_columns": desc.get("delta_sync_index_spec", {}).get("source_table_columns"),
        "delta_sync_index_spec.embedding_source_columns": desc.get("delta_sync_index_spec", {}).get("embedding_source_columns"),
        "direct_access_index_spec.schema.columns": desc.get("direct_access_index_spec", {}).get("schema", {}).get("columns"),
        "direct_access_index_spec.embedding_vector_columns": desc.get("direct_access_index_spec", {}).get("embedding_vector_columns"),
        "top-level: schema": desc.get("schema"),
        "top-level: columns": desc.get("columns"),
    }
    for label, val in candidates.items():
        marker = "[Y]" if val else "[ ]"
        print(f"  {marker} {label}: {val!r}")

    # 3. scan(num_results=1) — most reliable way to see what is really stored
    try:
        scan_result = index.scan(num_results=1)
        _dump("scan(num_results=1)", scan_result)
        _write_json(f"{_safe(index_name)}__scan.json", scan_result)
        if isinstance(scan_result, dict):
            manifest_cols = scan_result.get("manifest", {}).get("columns")
            data = scan_result.get("data")
            if manifest_cols:
                print("\nIndex columns (from scan manifest):")
                for c in manifest_cols:
                    print(f"  - {c.get('name')} ({c.get('type')})")
            elif data and isinstance(data, list) and isinstance(data[0], dict):
                print("\nIndex columns (from scan data[0] keys):")
                for k, v in data[0].items():
                    print(f"  - {k} ({type(v).__name__})")
    except Exception as e:
        print(f"\nscan() FAILED: {e}")

    # 4. similarity_search with a dummy term — fallback to learn column names
    try:
        pk = desc.get("primary_key") or "id"
        sim = index.similarity_search(query_text="test", columns=[pk], num_results=1)
        _dump("similarity_search(columns=[pk], num_results=1)", sim)
        _write_json(f"{_safe(index_name)}__similarity_search.json", sim)
    except Exception as e:
        print(f"\nsimilarity_search FAILED: {e}")


def main() -> None:
    endpoint = os.getenv("DATABRICKS_VS_ENDPOINT")
    if not endpoint:
        raise SystemExit("DATABRICKS_VS_ENDPOINT must be set in .env")

    client = get_client()

    cli_index = sys.argv[1] if len(sys.argv) > 1 else None
    target_index = cli_index or os.getenv("DATABRICKS_VS_INDEX")

    if target_index:
        probe_index(client, endpoint, target_index)
        return

    names = list_indexes(client, endpoint)
    if not names:
        print("No indexes to probe.")
        return
    for n in names:
        probe_index(client, endpoint, n)


if __name__ == "__main__":
    main()
