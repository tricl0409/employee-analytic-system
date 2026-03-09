"""
user_management.py — Admin-only User Management Page

Changes vs. previous version:
  C-3  created_at formatted as "DD Mon YYYY" instead of raw ISO
  U-1  last_login column shown in table
  S-3  passes current_username to update_user_role
  C-1  surfaces "password_too_short" error key
  S-2  surfaces "cannot_delete_last_admin" error key
"""

import streamlit as st
from datetime import datetime
from modules.core.auth_engine import AuthEngine
from modules.ui import page_header
from modules.utils.localization import get_text


# ==============================================================================
# HELPERS
# ==============================================================================

def _fmt_date(iso_str: str | None) -> str:
    """C-3: Convert ISO timestamp to human-readable 'DD Mon YYYY', or '—' if None."""
    if not iso_str:
        return "—"
    try:
        return datetime.fromisoformat(iso_str).strftime("%d %b %Y")
    except ValueError:
        return iso_str[:10]  # fallback: just the date part


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    """Admin-only user management dashboard."""
    lang = st.session_state.get("lang", "en")
    current_user = st.session_state.get("username", "")

    # --- RBAC GUARD ---
    if st.session_state.get("user_role") != "admin":
        st.error(get_text("access_denied", lang))
        st.stop()

    # --- PAGE HEADER ---
    page_header(
        title=get_text("user_management_title", lang),
        subtitle=get_text("user_management_subtitle", lang)
    )

    # --- CONTROL BAR ---
    col_search, col_add = st.columns([3, 1], gap="small")
    with col_search:
        search_query = st.text_input(
            "Search",
            placeholder=get_text("search_users", lang),
            label_visibility="collapsed",
            key="search_users_input"
        )
    with col_add:
        add_clicked = st.button(
            f":material/person_add: {get_text('add_user', lang)}",
            type="primary",
            use_container_width=True,
            key="btn_add_user"
        )

    # --- ADD USER FORM ---
    if add_clicked:
        st.session_state["show_add_user"] = True

    if st.session_state.get("show_add_user", False):
        with st.expander(get_text("add_user", lang), expanded=True):
            new_username = st.text_input(get_text("username", lang), key="new_user_username")
            new_display  = st.text_input(get_text("display_name", lang), key="new_user_display")
            new_password = st.text_input(get_text("password", lang), type="password", key="new_user_pw")
            new_role     = st.selectbox(get_text("select_role", lang), ["user", "admin"], key="new_user_role")

            c1, c2 = st.columns(2)
            with c1:
                if st.button(
                    f":material/check: {get_text('add_user', lang)}",
                    key="btn_confirm_add",
                    type="primary",
                    use_container_width=True
                ):
                    if new_username and new_password:
                        success, err = AuthEngine.create_user(new_username, new_password, new_display, new_role)
                        if success:
                            st.success(get_text("user_created", lang, username=new_username))
                            st.session_state["show_add_user"] = False
                            st.rerun()
                        else:
                            st.error(get_text(err, lang, username=new_username))
                    else:
                        st.error(get_text("login_failed", lang))
            with c2:
                if st.button(
                    f":material/close: {get_text('cancel', lang)}",
                    key="btn_cancel_add",
                    use_container_width=True
                ):
                    st.session_state["show_add_user"] = False
                    st.rerun()

    # --- USER LIST ---
    users = AuthEngine.list_users(search_query)
    total = len(users)

    st.markdown(f"""
        <div class="status-bar" style="background: linear-gradient(90deg, rgba(59, 130, 246, 0.15) 0%, rgba(59, 130, 246, 0.05) 100%); justify-content: space-between; border: 1px solid rgba(59, 130, 246, 0.2); margin-bottom: 20px;">
            <div style="display:flex; align-items:center;">
                <span class="status-label" style="color: var(--accent-blue);">{get_text('total_users', lang).upper()}:</span>
            </div>
            <span style="color: var(--text-muted); font-size: 0.75rem;"><b style="color:var(--accent-blue)">{get_text('users_count', lang, count=total)}</b></span>
        </div>
    """, unsafe_allow_html=True)

    # Table Header — now 6 cols including Last Login (U-1)
    st.markdown("<div style='margin-top: 20px;'></div>", unsafe_allow_html=True)
    h1, h2, h3, h4, h5, h6, h7 = st.columns([2.0, 2.0, 1.0, 1.2, 1.2, 0.75, 0.75])
    h1.markdown(f"<div class='user-table-header'>{get_text('username', lang)}</div>",     unsafe_allow_html=True)
    h2.markdown(f"<div class='user-table-header'>{get_text('display_name', lang)}</div>", unsafe_allow_html=True)
    h3.markdown(f"<div class='user-table-header'>{get_text('role', lang)}</div>",         unsafe_allow_html=True)
    h4.markdown(f"<div class='user-table-header'>{get_text('created_at', lang)}</div>",   unsafe_allow_html=True)
    h5.markdown("<div class='user-table-header'>Last Login</div>",                         unsafe_allow_html=True)
    h6.markdown(f"<div class='user-table-header'>{get_text('actions', lang)}</div>",      unsafe_allow_html=True)
    h7.markdown("",                                                                         unsafe_allow_html=True)
    st.markdown("<hr style='margin: 6px 0 12px 0; border:0; border-top:1px solid rgba(255,255,255,0.05);'>", unsafe_allow_html=True)

    if not users:
        st.info(get_text("no_users_found", lang))
        return

    for user in users:
        c1, c2, c3, c4, c5, c6, c7 = st.columns([2.0, 2.0, 1.0, 1.2, 1.2, 0.75, 0.75], gap="small")

        # Username (highlight self)
        is_current = (user["username"] == current_user)
        name_html  = (
            f"<span style='color:var(--accent-blue); font-weight:700'>● &nbsp;{user['username']}</span>"
            if is_current else user["username"]
        )
        c1.markdown(f"<div class='user-table-cell'>{name_html}</div>", unsafe_allow_html=True)

        # Display Name
        c2.markdown(f"<div class='user-table-cell'>{user['display_name']}</div>", unsafe_allow_html=True)

        # Role Badge
        badge_class = f"user-role-badge-{user['role']}"
        role_label  = get_text("role_admin", lang) if user["role"] == "admin" else get_text("role_user", lang)
        c3.markdown(f"<div class='user-table-cell'><span class='{badge_class}'>{role_label}</span></div>", unsafe_allow_html=True)

        # Created At — C-3: formatted date
        c4.markdown(
            f"<div class='user-table-cell' style='color:var(--text-muted);'>{_fmt_date(user['created_at'])}</div>",
            unsafe_allow_html=True
        )

        # Last Login — U-1
        c5.markdown(
            f"<div class='user-table-cell' style='color:var(--text-muted);'>{_fmt_date(user.get('last_login'))}</div>",
            unsafe_allow_html=True
        )

        # Edit Button
        if c6.button(":material/edit:", key=f"btn_edit_{user['username']}", use_container_width=True, help=get_text("edit_user", lang)):
            st.session_state["editing_user"] = user["username"]
            st.rerun()

        # Delete Button (cannot delete self)
        if not is_current:
            if c7.button(":material/delete:", key=f"btn_del_{user['username']}", use_container_width=True, help=get_text("delete", lang)):
                success, err = AuthEngine.delete_user(user["username"], current_user)
                if success:
                    st.success(get_text("user_deleted", lang, username=user["username"]))
                    st.rerun()
                else:
                    st.error(get_text(err, lang))
        else:
            c7.markdown("<div class='user-table-cell' style='color:var(--text-muted); font-size:0.7rem;'>—</div>", unsafe_allow_html=True)

    # --- EDIT USER PANEL ---
    editing_username = st.session_state.get("editing_user")
    if editing_username:
        edit_user = AuthEngine.get_user(editing_username)
        if edit_user:
            st.markdown("---")
            st.markdown(f"### {get_text('edit_user', lang)}: **{editing_username}**")

            col_edit1, col_edit2 = st.columns(2)
            with col_edit1:
                edit_display = st.text_input(
                    get_text("display_name", lang),
                    value=edit_user["display_name"],
                    key="edit_display_name"
                )
            with col_edit2:
                edit_role = st.selectbox(
                    get_text("select_role", lang),
                    ["user", "admin"],
                    index=0 if edit_user["role"] == "user" else 1,
                    key="edit_role"
                )

            new_pw = st.text_input(
                get_text("new_password", lang) + " (leave blank to keep)",
                type="password",
                key="edit_new_pw"
            )

            col_save, col_cancel = st.columns(2)
            with col_save:
                if st.button(
                    f":material/save: {get_text('edit_profile', lang)}",
                    key="btn_save_edit",
                    type="primary",
                    use_container_width=True
                ):
                    errors = []
                    AuthEngine.update_profile(editing_username, display_name=edit_display)

                    # S-3: pass current_username so self-demotion is blocked
                    ok_role, err_role = AuthEngine.update_user_role(
                        editing_username, edit_role, current_username=current_user
                    )
                    if not ok_role:
                        errors.append(get_text(err_role, lang))

                    if new_pw:
                        ok_pw, err_pw = AuthEngine.reset_password(editing_username, new_pw)
                        if not ok_pw:
                            errors.append(get_text(err_pw, lang))

                    if errors:
                        for e in errors:
                            st.error(e)
                    else:
                        st.session_state["editing_user"] = None
                        st.success(get_text("profile_updated", lang))
                        st.rerun()

            with col_cancel:
                if st.button(
                    f":material/close: {get_text('cancel', lang)}",
                    key="btn_cancel_edit",
                    use_container_width=True
                ):
                    st.session_state["editing_user"] = None
                    st.rerun()


if __name__ == "__main__":
    main()
