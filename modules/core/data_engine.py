import os
import pandas as pd
import streamlit as st
from typing import List, Dict, Any

DATA_DIR = "data/uploads"


def _get_file_mtime(active_file: str) -> float:
    """Get file modification time for cache invalidation."""
    file_path = os.path.join(DATA_DIR, active_file)
    try:
        return os.path.getmtime(file_path)
    except OSError:
        return 0.0

@st.cache_data
def load_and_standardize(active_file: str, _file_mtime: float = 0.0) -> pd.DataFrame:
    """
    Reads a CSV file and standardizes column names.
    CACHED: Cache is invalidated when file modification time changes.
    
    Args:
        active_file (str): The absolute path to the CSV file.
        _file_mtime (float): File modification timestamp (cache-busting key).
        
    Returns:
        pd.DataFrame: The standardized DataFrame.
    """
    try:
        file_path = os.path.join(DATA_DIR, active_file)
        if not os.path.exists(file_path):
             from modules.ui.components import styled_alert
             styled_alert(f"File not found: {file_path}", "error")
             return pd.DataFrame()
        df = pd.read_csv(file_path)
        # Vectorized string operation for column cleaning
        df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
        return df
    except Exception as e:
        from modules.ui.components import styled_alert
        styled_alert(f"Error loading file: {e}", "error")
        return pd.DataFrame()

def process_inventory(library: List[Dict], search_query: str = "") -> pd.DataFrame:
    """
    Converts the library list to a DataFrame, filters by search query, and sorts by date.
    
    Args:
        library (List[Dict]): List of file metadata dictionaries.
        search_query (str): Optional search query to filter file names.
        
    Returns:
        pd.DataFrame: A processed DataFrame ready for display.
    """
    df = pd.DataFrame(library)
    
    if df.empty:
        return df
        
    # Vectorized Search Filter
    if search_query:
        # Case-insensitive containment check
        mask = df['name'].str.contains(search_query, case=False, na=False)
        df = df[mask]
        
    # Sort by Date (descending) if data exists
    if not df.empty and 'date' in df.columns:
        df = df.sort_values(by="date", ascending=False)
        
    return df

def compute_dataset_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Computes detailed metrics for a dataset preview using efficient pandas operations.
    
    Args:
        df (pd.DataFrame): The DataFrame to analyze.
        
    Returns:
        Dict: Contains rows, cols, memory usage, duplicates, and missing percentage.
    """
    rows, cols = df.shape
    memory_usage = df.memory_usage(deep=True).sum() / 1024 / 1024 # MB
    duplicates = df.duplicated().sum()
    total_cells = rows * cols
    missing_cells = df.isnull().sum().sum()
    missing_pct = (missing_cells / total_cells * 100) if total_cells > 0 else 0
    
    return {
        "rows": rows,
        "cols": cols,
        "memory_mb": memory_usage,
        "duplicates": duplicates,
        "missing_pct": missing_pct
    }

