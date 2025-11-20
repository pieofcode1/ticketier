"""PostgreSQL tools for ticket support system."""
import os
import psycopg2
from psycopg2.extras import RealDictCursor
from typing import List, Dict, Any, Optional, Set, Callable
from dotenv import load_dotenv
import numpy as np
import json


load_dotenv()

# Database connection parameters
DB_CONFIG = {
    'host': os.getenv('PG_HOST', 'localhost'),
    'port': os.getenv('DB_PORT', '5432'),
    'database': os.getenv('PG_DATABASE', 'your_database'),
    'user': os.getenv('PG_USERNAME', 'postgres'),
    'password': os.getenv('PG_PASSWORD', '')
}


def get_db_connection():
    """Get a database connection."""
    return psycopg2.connect(**DB_CONFIG)


def get_query_embedding(conn, text: str) -> List[float]:

    """Generate embedding for the given text using the database's embedding function."""
    cursor = conn.cursor()
    try:
        query = "SELECT azure_openai.create_embeddings('text-embedding-ada-002', %s);"
        cursor.execute(query, (text,))
        result = cursor.fetchone()
        if result:
            return result[0]  # Assuming the function returns an array of floats
        else:
            return []

    except Exception as e:
        print(f"Error generating embedding: {e}")
        return []
    finally:
        cursor.close()

def search_similar_issues(
    query: str,
    top_k: int = 5,
    similarity_threshold: float = 0.7
) -> List[Dict[str, Any]]:
    """
    Search for similar issues using vector similarity.
    
    Args:
        query: search query text
        top_k: Number of top similar results to return (default: 5)
        similarity_threshold: Minimum cosine similarity threshold (default: 0.7)
    
    Returns:
        List of dictionaries containing ticket_id, created, top_contact_reason, 
        customer_description, and similarity_score
    """
    conn = get_db_connection()
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    
    try:

        query_embedding = get_query_embedding(conn, query)
        # Convert embedding list to PostgreSQL array format
        embedding_str = '[' + ','.join(map(str, query_embedding)) + ']'
        
        # SQL query using cosine similarity with pgvector
        # Assumes you're using pgvector extension: CREATE EXTENSION vector;
        query = """
            SELECT 
                ticket_id,
                CAST(created as TEXT),
                status,
                issue_summary,
                priority,
                affected_service,
                customer_impact,
                assigned_team,
                1 - (issue_summary_vector <=> %s::vector) AS similarity_score
            FROM servicenow_tickets_with_category
            WHERE issue_summary_vector IS NOT NULL
                AND 1 - (issue_summary_vector <=> %s::vector) >= %s
            ORDER BY issue_summary_vector <=> %s::vector
            LIMIT %s;
        """
        
        cursor.execute(query, (embedding_str, embedding_str, similarity_threshold, embedding_str, top_k))
        results = cursor.fetchall()
        
        # Convert to list of dicts
        return [dict(row) for row in results]
        
    except Exception as e:
        print(f"Error searching similar issues: {e}")
        return []
    finally:
        cursor.close()
        conn.close()


def get_tickets_by_category(
    category: str,
    limit: int = 10
) -> List[Dict[str, Any]]:
    """
    Get tickets filtered by category.
    
    Args:
        category: The category to filter by
        limit: Maximum number of results to return
    
    Returns:
        List of ticket summaries
    """
    conn = get_db_connection()
    cursor = conn.cursor(cursor_factory=RealDictCursor)

    try:
        query = """
            SELECT 
                ticket_id,
                created,
                issue_summary,
                priority,
                status,
                customer_description
            FROM support_tickets
            WHERE category = %s
            ORDER BY created DESC
            LIMIT %s;
        """

        cursor.execute(query, (category, limit))
        results = cursor.fetchall()

        return [dict(row) for row in results]

    except Exception as e:
        print(f"Error retrieving tickets by category: {e}")
        return []
    finally:
        cursor.close()
        conn.close()


# Statically defined user functions for fast reference
user_functions: Set[Callable[..., Any]] = {
    search_similar_issues,
    # get_tickets_by_category
}
