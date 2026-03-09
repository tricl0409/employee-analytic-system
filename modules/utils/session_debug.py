"""
session_debug.py — Session State Debug Utility

Wraps st.session_state writes so that every new or updated key is
automatically persisted to /data/temp/<key>.(json|csv) for easy inspection.

Usage:
    from modules.utils.session_debug import set_state, get_state

    set_state("my_key", my_value)   # writes to session_state + /data/temp/
    val = get_state("my_key", default=None)
"""

import json
import logging
import streamlit as st
import pandas as pd
from pathlib import Path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Resolve temp directory relative to this file's project root
# (modules/utils/ → ../../data/temp)
# ---------------------------------------------------------------------------
_TEMP_DIR = Path(__file__).parents[2] / "data" / "temp"


def _ensure_temp_dir() -> Path:
    """Create /data/temp if it doesn't exist, return the Path."""
    _TEMP_DIR.mkdir(parents=True, exist_ok=True)
    return _TEMP_DIR


def _dump_to_temp(key: str, value) -> None:
    """
    Dumps `value` to /data/temp/<key>.csv (DataFrames) or .json (everything else).
    Always overwrites the file if it already exists.
    """
    try:
        temp_dir = _ensure_temp_dir()
        # Sanitize key: remove characters invalid in filenames
        safe_key = "".join(c if c.isalnum() or c in ("_", "-") else "_" for c in key)

        if isinstance(value, pd.DataFrame):
            out_path = temp_dir / f"{safe_key}.csv"
            value.to_csv(out_path, index=False, encoding="utf-8-sig")
        else:
            out_path = temp_dir / f"{safe_key}.json"
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(value, f, indent=2, default=str, ensure_ascii=False)
    except Exception as exc:
        # Never let debug I/O crash the main app
        logger.warning("session_debug: failed to dump key '%s': %s", key, exc)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def set_state(key: str, value) -> None:
    """
    Write `value` to st.session_state[key] and dump a debug snapshot to
    /data/temp/<key>.json|csv (overwriting any previous file for that key).
    """
    st.session_state[key] = value
    _dump_to_temp(key, value)


def get_state(key: str, default=None):
    """
    Thin wrapper around st.session_state.get().
    No file I/O on reads — use this for symmetry / explicitness.
    """
    return st.session_state.get(key, default)
