import os
import pandas as pd
import streamlit as st
from modules.core.file_manager import save_file, UPLOADS_DIR
from modules.core.audit_engine import validate_schema
from modules.utils.localization import get_text
from modules.ui.icons import get_icon


def _alert_html(message: str, kind: str = "error") -> str:
    """Render a styled alert box with an SVG icon from the registry."""
    if kind == "error":
        icon_svg = get_icon("x_circle", size=16, color="#FF5B5C")
        bg, border, text_color = "rgba(255,91,92,0.12)", "#FF5B5C", "#FF5B5C"
    elif kind == "warning":
        icon_svg = get_icon("alert_triangle", size=16, color="#FF9F43")
        bg, border, text_color = "rgba(255,159,67,0.12)", "#FF9F43", "#FF9F43"
    else:
        icon_svg = get_icon("check_circle", size=16, color="#A6CE39")
        bg, border, text_color = "rgba(166,206,57,0.12)", "#A6CE39", "#A6CE39"
    return f"""
    <div style="display:flex;align-items:flex-start;gap:10px;background:{bg};border:1px solid {border};
                border-radius:10px;padding:12px 16px;margin:8px 0;font-size:0.82rem;color:{text_color};font-weight:600;">
        <span style="flex-shrink:0;margin-top:1px;">{icon_svg}</span>
        <span style="line-height:1.5;">{message}</span>
    </div>"""


@st.dialog("Upload New Data")
def upload_dialog():
    """Modal dialog for uploading new CSV files."""
    lang = st.session_state.get('lang', 'en')

    st.write(get_text("upload_instruction", lang, default="Select a CSV file to upload to the system (Max 200MB)."))

    up_file = st.file_uploader("CSV", type="csv", label_visibility="collapsed")

    if up_file:
        if st.button(f":material/upload: {get_text('upload_file', lang)}", use_container_width=True, type="primary"):
            # 1. Validate Duplicate Filename
            file_path = os.path.join(UPLOADS_DIR, up_file.name)
            if os.path.exists(file_path):
                st.markdown(
                    _alert_html(f"File '<b>{up_file.name}</b>' already exists. Please rename or delete the existing file.", kind="warning"),
                    unsafe_allow_html=True,
                )
                return

            # 2. Schema Validation
            try:
                up_file.seek(0)
                df = pd.read_csv(up_file)
                df.columns = df.columns.str.strip()
                sv = validate_schema(df, lang)

                if sv.get("status") not in ("no_rule", "pass"):
                    missing        = sv.get("missing_columns", [])
                    extra          = sv.get("extra_columns", [])
                    type_mismatches = sv.get("type_mismatches", [])

                    # ── Column name mismatch → BLOCK upload ──────────────────
                    if missing or extra:
                        parts = []
                        if missing:
                            parts.append(f"<b>{get_text('schema_missing_cols', lang)}:</b> {', '.join(missing)}")
                        if extra:
                            parts.append(f"<b>{get_text('schema_extra_cols', lang)}:</b> {', '.join(extra)}")
                        st.markdown(_alert_html("<br>".join(parts), kind="error"), unsafe_allow_html=True)
                        return

                    # ── Type mismatch only → WARN, still allow upload ─────────
                    if type_mismatches:
                        cols = [m["column"] for m in type_mismatches]
                        st.toast(f"⚠️ Type warnings: {', '.join(cols)}")

            except Exception:
                pass  # No schema rule defined → allow upload


            # 3. Save File (only reached if schema is valid)
            try:
                up_file.seek(0)
                save_file(up_file)
            except Exception as e:
                st.markdown(_alert_html(get_text("error_upload", lang, error=str(e)), kind="error"), unsafe_allow_html=True)
                return

            st.toast(get_text("success_upload", lang), icon="✅")
            st.rerun()



@st.dialog("Profile Settings")
def profile_dialog():
    """Modal dialog for password change and profile update."""
    from modules.core.auth_engine import AuthEngine

    lang = st.session_state.get('lang', 'en')
    username = st.session_state.get('username', '')

    tab_profile, tab_password = st.tabs([get_text('edit_profile', lang), get_text('change_password', lang)])

    with tab_profile:
        display_name = st.text_input(
            get_text('display_name', lang),
            value=st.session_state.get('display_name', ''),
            key="dialog_display_name"
        )

        if st.button(f":material/save: {get_text('edit_profile', lang)}", key="btn_save_profile", type="primary", use_container_width=True):
            AuthEngine.update_profile(username, display_name=display_name)
            st.success(get_text('profile_updated', lang))
            st.rerun()

    with tab_password:
        old_pw = st.text_input(get_text('old_password', lang), type="password", key="dialog_old_pw")
        new_pw = st.text_input(get_text('new_password', lang), type="password", key="dialog_new_pw")
        confirm_pw = st.text_input(get_text('confirm_password', lang), type="password", key="dialog_confirm_pw")

        if st.button(f":material/lock: {get_text('change_password', lang)}", key="btn_change_pw", type="primary", use_container_width=True):
            if new_pw != confirm_pw:
                st.error(get_text('password_mismatch', lang))
            elif AuthEngine.change_password(username, old_pw, new_pw):
                st.success(get_text('password_changed', lang))
            else:
                st.error(get_text('password_wrong', lang))

