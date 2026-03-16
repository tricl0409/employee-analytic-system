import streamlit as st
from modules.ui import sidebar_branding, footer, sidebar_user_info, sidebar_ai_chat
from modules.ui.styles import apply_style
from modules.ui.dialogs import profile_dialog
from modules.utils.localization import get_text
from modules.core.auth_engine import AuthEngine
from modules.utils.db_config_manager import seed_default_rules, load_rules_into_session

# ==============================================================================
# CONFIG & SETUP
# ==============================================================================

st.set_page_config(page_title="Employee Analytics System", layout="wide", page_icon="💎")
apply_style()

AuthEngine.init_db()
seed_default_rules()
load_rules_into_session()

if 'lang' not in st.session_state:
    st.session_state.lang = 'en'
if 'cleaned_data' not in st.session_state:
    st.session_state['cleaned_data'] = None
if 'active_file' not in st.session_state:
    st.session_state['active_file'] = None
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False

lang = st.session_state.lang

# ==============================================================================
# NAVIGATION
# ==============================================================================

if st.session_state.authenticated:
    home_p = st.Page("pages/overview.py", title=get_text("overview", lang), url_path="overview", default=True)
    quality_p = st.Page("pages/data_audit.py", title=get_text("data_audit", lang), url_path="data_audit")
    prep_p = st.Page("pages/preprocessing.py", title=get_text("preprocessing", lang), url_path="preprocessing")
    eda_p = st.Page("pages/eda.py", title=get_text("eda", lang), url_path="eda")
    conclusion_p   = st.Page("pages/conclusion.py", title="Conclusion & Recommendation", url_path="conclusion")

    nav_dict = {
        get_text("nav_system", lang): [home_p],
        get_text("nav_analytics", lang): [quality_p, prep_p, eda_p, conclusion_p],
    }

    if st.session_state.get('user_role') == 'admin':
        user_mgmt_p = st.Page("pages/management/user_management.py", title=get_text("user_management", lang))
        admin_settings_p = st.Page("pages/management/analytic_rule_settings.py", title=get_text("admin_settings", lang))
        nav_dict[get_text("nav_management", lang)] = [user_mgmt_p, admin_settings_p]

    pg = st.navigation(nav_dict)
else:
    # Login state — dedicated login page, sidebar nav hidden
    login_page = st.Page("pages/login.py", title="Login", default=True)
    pg = st.navigation([login_page], position="hidden")

# ==============================================================================
# LOGIN GUARD — stop here if not authenticated
# ==============================================================================

if not st.session_state.authenticated:
    # Hide sidebar completely on login screen
    st.markdown("""
    <style>
        [data-testid="stSidebar"] { display: none !important; }
        [data-testid="stSidebarCollapsedControl"] { display: none !important; }
    </style>
    """, unsafe_allow_html=True)
    pg.run()
    st.stop()

# ==============================================================================
# AUTHENTICATED — SIDEBAR + PAGE
# ==============================================================================

lang_clicked, profile_clicked, logout_clicked = sidebar_user_info()
sidebar_ai_chat()

if logout_clicked:
    saved_lang = st.session_state.lang
    st.session_state.clear()
    st.session_state.lang = saved_lang
    st.session_state.authenticated = False
    st.rerun()

if lang_clicked:
    st.session_state.lang = 'vi' if st.session_state.lang == 'en' else 'en'
    st.rerun()

if profile_clicked:
    profile_dialog()

sidebar_branding()

pg.run()

footer()