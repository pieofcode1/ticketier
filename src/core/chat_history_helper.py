"""
Chat History Helper - Save and load chat histories for Streamlit apps.

This module provides utilities for persisting chat histories to JSON files.
"""

import os
import json
from datetime import datetime
from typing import List, Dict, Any, Optional
from pathlib import Path


def get_history_dir() -> Path:
    """Get the history directory path, creating it if necessary."""
    # Get the src directory (parent of this file's parent)
    src_dir = Path(__file__).parent.parent
    history_dir = src_dir / "history"
    history_dir.mkdir(parents=True, exist_ok=True)
    return history_dir


def generate_filename(prefix: str, extension: str = "json") -> str:
    """Generate a unique filename with timestamp."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{timestamp}.{extension}"


def save_chat_history(
    chat_history: List[Dict[str, Any]],
    page_type: str,
    metadata: Optional[Dict[str, Any]] = None
) -> str:
    """
    Save chat history to a JSON file.
    
    Args:
        chat_history: List of chat message dictionaries
        page_type: Type of page (e.g., "vision_rag", "aisearch_rag")
        metadata: Optional metadata to include (e.g., image name, index name)
        
    Returns:
        Path to the saved file
    """
    if not chat_history:
        return ""
    
    history_dir = get_history_dir()
    filename = generate_filename(page_type)
    filepath = history_dir / filename
    
    # Build the save data
    save_data = {
        "page_type": page_type,
        "timestamp": datetime.now().isoformat(),
        "metadata": metadata or {},
        "messages": chat_history
    }
    
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False)
    
    return str(filepath)


def load_chat_history(filepath: str) -> Dict[str, Any]:
    """
    Load chat history from a JSON file.
    
    Args:
        filepath: Path to the history file
        
    Returns:
        Dictionary containing the loaded history data
    """
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def list_saved_histories(page_type: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    List all saved chat histories.
    
    Args:
        page_type: Optional filter by page type
        
    Returns:
        List of dictionaries with file info
    """
    history_dir = get_history_dir()
    histories = []
    
    for filepath in history_dir.glob("*.json"):
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
                
            # Filter by page type if specified
            if page_type and data.get("page_type") != page_type:
                continue
                
            histories.append({
                "filepath": str(filepath),
                "filename": filepath.name,
                "page_type": data.get("page_type", "unknown"),
                "timestamp": data.get("timestamp", ""),
                "message_count": len(data.get("messages", [])),
                "metadata": data.get("metadata", {})
            })
        except (json.JSONDecodeError, KeyError):
            continue
    
    # Sort by timestamp descending (newest first)
    histories.sort(key=lambda x: x["timestamp"], reverse=True)
    return histories


def delete_history(filepath: str) -> bool:
    """
    Delete a saved history file.
    
    Args:
        filepath: Path to the history file
        
    Returns:
        True if deleted successfully
    """
    try:
        os.remove(filepath)
        return True
    except OSError:
        return False
