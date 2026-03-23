import os
import numpy as np
import pandas as pd
import streamlit as st
from typing import List, Dict, Any

DATA_DIR = "data/uploads"

# Columns to drop immediately upon load (no analytical value)
USELESS_COLUMNS_TO_DROP = ["fnlwgt"]


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
        
        # Drop meaningless columns if they exist
        drop_cols = [c for c in USELESS_COLUMNS_TO_DROP if c in df.columns]
        if drop_cols:
            df = df.drop(columns=drop_cols)
            
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


# ─────────────────────────────────────────────────────────────────────────────
# CORRELATION ENCODING — domain-knowledge ordinal mapping
# ─────────────────────────────────────────────────────────────────────────────

#: Static ordinal mapping for computing Pearson correlation on cleaned data.
#: Order reflects **domain-knowledge hierarchy** (e.g. education level,
#: income-correlated occupation ranking).
#: Source: user-defined reference table.
_CORRELATION_ENCODING_MAP: Dict[str, Dict[str, int]] = {
    "relationship": {
        "Unmarried": 0,
        "Not-in-family": 1,
        "Other-relative": 2,
        "Own-child": 3,
        "Wife": 4,
        "Husband": 5,
    },
    "marital_status": {
        "Never-married": 0,
        "Separated": 1,
        "Divorced": 2,
        "Widowed": 3,
        "Married-spouse-absent": 4,
        "Married-AF-spouse": 5,
        "Married-civ-spouse": 6,
    },
    "sex": {
        "Female": 0,
        "Male": 1,
    },
    "education": {
        "Preschool": 1,
        "1st-4th": 2,
        "5th-6th": 3,
        "7th-8th": 4,
        "9th": 5,
        "10th": 6,
        "11th": 7,
        "12th": 8,
        "HS-grad": 9,
        "Some-college": 10,
        "Assoc-acdm": 11,
        "Assoc-voc": 12,
        "Bachelors": 13,
        "Masters": 14,
        "Prof-school": 15,
        "Doctorate": 16,
    },
    "workclass": {
        "Never-worked": 0,
        "Without-pay": 1,
        "Self-emp-not-inc": 2,
        "Self-emp-inc": 3,
        "Private": 4,
        "Local-gov": 5,
        "State-gov": 6,
        "Federal-gov": 7,
    },
    "occupation": {
        "Other-service": 1,
        "Priv-house-serv": 2,
        "Handlers-cleaners": 3,
        "Craft-repair": 4,
        "Farming-fishing": 5,
        "Transport-moving": 6,
        "Machine-op-inspct": 7,
        "Protective-serv": 8,
        "Tech-support": 9,
        "Sales": 10,
        "Adm-clerical": 11,
        "Armed-Forces": 12,
        "Prof-specialty": 13,
        "Exec-managerial": 14,
    },
    "income": {
        "<=50K": 0,
        ">50K": 1,
    },
}


def encode_for_correlation(df: pd.DataFrame) -> pd.DataFrame:
    """Create a fully numeric copy of *df* for Pearson correlation analysis.

    Applies domain-knowledge ordinal mapping from ``_CORRELATION_ENCODING_MAP``
    to known categorical columns.  Any remaining non-numeric columns that are
    **not** in the map are encoded via alphabetical ``OrdinalEncoder`` as a
    fallback.

    **The original DataFrame is NEVER mutated.**

    Args:
        df: Input DataFrame (typically the ``_cleaned`` snapshot).

    Returns:
        A new DataFrame where every column is numeric (suitable for ``.corr()``).
    """
    from sklearn.preprocessing import OrdinalEncoder

    df_enc = df.copy()

    # ── Phase 0: Drop redundant features (same rule as Step 8) ────────────
    from modules.core.preprocessing_engine import PreprocessingEngine
    cols_lower = {c.lower() for c in df_enc.columns}
    drop_cols = [
        col for col in df_enc.columns
        if col.lower() in PreprocessingEngine._REDUNDANT_PAIRS
        and PreprocessingEngine._REDUNDANT_PAIRS[col.lower()] in cols_lower
    ]
    if drop_cols:
        df_enc = df_enc.drop(columns=drop_cols)

    # ── Phase 1: Apply domain-knowledge mapping ──────────────────────────
    for col in df_enc.columns:
        col_lower = col.lower()
        mapping = _CORRELATION_ENCODING_MAP.get(col_lower)
        if mapping is None:
            continue
        # Convert Categorical dtype → str to avoid "Cannot setitem on a
        # Categorical with a new category" when .map() returns int codes.
        if df_enc[col].dtype.name == "category":
            df_enc[col] = df_enc[col].astype(str)
        # Map known values; unmapped values become NaN (coerced safely)
        df_enc[col] = df_enc[col].map(mapping)
        df_enc[col] = pd.to_numeric(df_enc[col], errors="coerce")

    # ── Phase 2: Fallback for remaining non-numeric columns ──────────────
    remaining_cat = df_enc.select_dtypes(
        include=["object", "category"],
    ).columns.tolist()

    for col in remaining_cat:
        if df_enc[col].dtype.name == "category":
            df_enc[col] = df_enc[col].astype(str)
        non_null = df_enc[col].notna()
        if not non_null.any():
            continue
        encoder = OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1,
        )
        values = df_enc.loc[non_null, col].astype(str).values.reshape(-1, 1)
        df_enc.loc[non_null, col] = (
            encoder.fit_transform(values).ravel().astype(int)
        )
        df_enc[col] = pd.to_numeric(df_enc[col], errors="coerce")

    return df_enc
