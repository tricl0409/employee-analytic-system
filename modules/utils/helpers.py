import os
import streamlit as st
from modules.utils.localization import get_text


def _get_current_lang():
    return st.session_state.get('lang', 'en')


def _ensure_workspace_active():
    """
    Ensures that a workspace is active before proceeding with page rendering.
    Stops execution and displays an informational message if no file is selected.
    """
    lang = _get_current_lang()
    active_file = st.session_state.get('active_file')
    no_data_loaded = get_text('no_data_loaded', lang)
    
    if not active_file or active_file == no_data_loaded:
        st.info(get_text('select_dataset_first', lang, page=get_text('overview', lang)))
        st.stop()


# ── Temp file helper ───────────────────────────────────────────────────────────────

def save_temp_csv(df, prefix: str = "temp") -> None:
    """
    Save a DataFrame to data/temp/<prefix>.csv.
    Delegates to session_debug._dump_to_temp() — same mechanism as audit page.
    Always overwrites (latest snapshot wins).
    """
    from modules.utils.session_debug import _dump_to_temp
    _dump_to_temp(prefix, df)
