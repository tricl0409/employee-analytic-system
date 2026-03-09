"""
overview.py — Overview Dashboard Page
"""

import streamlit as st
from modules.ui import overview_header, workspace_status, file_inventory, preview_panel, feature_navigation
from modules.utils.localization import get_text


def main():
    """
    Main function for the Overview Dashboard.
    Refactored for performance (Lazy Loading) and Modularity.
    """
    lang = st.session_state.get('lang', 'en')

    # --- PAGE HEADER ---
    overview_header(lang)

    # --- WORKSPACE STATUS ---
    active_file = st.session_state.get('active_file', get_text('no_data_loaded', lang))
    workspace_status(active_file)

    # --- FILE INVENTORY (METADATA ONLY) ---
    file_inventory(active_file)

    # --- PREVIEW PANEL (LAZY LOADED) ---
    preview_panel()

    # --- FEATURE NAVIGATION (CONDITIONAL) ---
    if active_file is not None and active_file != get_text('no_data_loaded', lang):
        from modules.core.data_engine import load_and_standardize, _get_file_mtime
        try:
            df = load_and_standardize(active_file, _file_mtime=_get_file_mtime(active_file))
            num_records = f"{len(df):,}"
        except Exception:
            num_records = "0"

        feature_navigation(num_records, lang)
    else:
        st.markdown(
            f"<br><br><div style='text-align:center; color: var(--text-muted); font-style: italic;'>{get_text('empty_state_msg', lang)}</div>",
            unsafe_allow_html=True,
        )

    # --- AUTO-SCROLL LOGIC ---
    if st.session_state.get('scroll_to_modules', False):
        import streamlit.components.v1 as components
        import time

        components.html(f"""
            <script>
                setTimeout(function() {{
                    const element = window.parent.document.getElementById('module-selection-anchor');
                    if (element) {{
                        element.scrollIntoView({{ behavior: 'smooth', block: 'start' }});
                    }}
                }}, 100);
            </script>
            <div style="display:none;">{time.time()}</div>
        """, height=0, width=0)
        st.session_state.scroll_to_modules = False


if __name__ == "__main__":
    main()