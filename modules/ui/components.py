import time
import pandas as pd
import streamlit as st

from modules.ui.icons import ICONS, get_icon
from modules.core.file_manager import get_data_library, delete_data
from modules.core.data_engine import load_and_standardize, _get_file_mtime, process_inventory, compute_dataset_metrics
from modules.core import audit_engine
from modules.core.audit_engine import generate_column_report
from modules.ui import visualizer
from modules.ui.dialogs import upload_dialog
from modules.utils.localization import get_text
from modules.utils.helpers import _get_current_lang
from modules.utils.theme_manager import STATUS_COLORS

# ─────────────────────────────────────────────────────────────────────────────
# MODULE-LEVEL PRIVATE HELPERS (shared by the 3 preprocessing tab renderers)
# Defined once here instead of being re-declared inside each @staticmethod.
# ─────────────────────────────────────────────────────────────────────────────

def _pp_hex(h: str) -> str:
    """Convert `#RRGGBB` to `'R,G,B'` string for CSS rgba()."""
    h = h.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"{r},{g},{b}"


def _pp_key(t: str) -> str:
    """Wrap text in a white bold <strong> tag."""
    return f"<strong style='color:white;'>{t}</strong>"


def _pp_card(content: str, color: str = None) -> str:
    """Return an info-card HTML block with optional accent colour."""
    bg = f"rgba({_pp_hex(color)},0.06)" if color else "rgba(255,255,255,0.03)"
    br = _pp_hex(color) if color else "255,255,255"
    return (
        f"<div class='pp-info-card' style='background:{bg};"
        f"border:1px solid rgba({br},0.18);border-radius:10px;"
        f"padding:14px 18px;margin-bottom:12px;"
        f"font-size:0.87rem;color:var(--text-secondary);line-height:1.75;'>"
        f"{content}</div>"
    )


def _pp_step_hdr(num: int, title: str, color: str, icon_key: str = None) -> str:
    """Return HTML for a numbered step header with optional icon."""
    from modules.ui.icons import get_icon as _gi
    ico = (
        f"<span style='margin-right:8px;display:inline-flex;align-items:center;'>"
        f"{_gi(icon_key, 18, 'white')}</span>"
        if icon_key else ""
    )
    return (
        f"<div style='display:flex;align-items:center;gap:12px;margin-bottom:16px;'>"
        f"<div class='pp-step-circle' style='width:34px;height:34px;border-radius:50%;"
        f"background:linear-gradient(135deg,{color},{color}bb);color:white;"
        f"display:flex;align-items:center;justify-content:center;"
        f"font-weight:800;font-size:0.85rem;flex-shrink:0;"
        f"box-shadow:0 0 14px {color}66;'>{num}</div>"
        f"<div style='font-size:1rem;font-weight:700;color:white;"
        f"display:flex;align-items:center;letter-spacing:-0.2px;'>{ico}{title}</div>"
        f"</div>"
    )


class UiComponents:
    """
    A collection of UI components for the Employee Analytic System.
    """

    @staticmethod
    def section_divider():
        """
        Renders a consistent section divider.
        """
        st.markdown("""
            <div style="height:1px; background:linear-gradient(90deg,
                transparent 0%, rgba(255,255,255,0.06) 20%, rgba(255,255,255,0.06) 80%, transparent 100%);
                margin:28px 0 24px 0;"></div>
        """, unsafe_allow_html=True)    

    @staticmethod
    def active_file_scan_progress_bar(key):
        """
        Renders a progress bar for the scan animation.
        """
        active_file = st.session_state.get("active_file")
        audit_key = f"{key}_{active_file}"
        if audit_key not in st.session_state:
            scan_placeholder = st.empty()
            with scan_placeholder.container():
                UiComponents.scan_animation()
            time.sleep(1.2)
            scan_placeholder.empty()
            st.session_state[audit_key] = True

    @staticmethod
    def page_header(title, subtitle):
        """
        Renders a consistent Hero Section for all pages.
        """
        st.markdown(f"""
        <div class="page-header-container">
            <h1 style='color: white; font-size: 3.8rem; font-weight: 800; letter-spacing: -1.5px; margin-bottom: 10px; text-shadow: 0 0 30px rgba(255,255,255,0.1);'>
                {title}
            </h1>
            <p style='color: var(--text-secondary); font-size: 1.2rem; max-width: 900px; margin: 0 auto; line-height: 1.6;'>
                {subtitle}
            </p>
        </div>
        """, unsafe_allow_html=True)

    @staticmethod
    def overview_header(lang):
        """
        Renders the Overview page header with interactive boxes.
        Styles are in styles.py (OVERVIEW_FEATURE_STYLES).
        """
        st.markdown(f"""
        <div style="text-align: center; margin-bottom: 40px; margin-top: 20px;">
            <h1 style='color: white; font-size: 3.8rem; font-weight: 800; letter-spacing: -1.5px; margin-bottom: 10px; text-shadow: 0 0 30px rgba(255,255,255,0.1);'>
                {get_text('home_title', lang)}
            </h1>
            <p style='color: var(--accent-orange); font-size: 1.2rem; font-style: italic; max-width: 900px; margin: 0 auto; line-height: 1.6;'>
                {get_text('home_subtitle', lang)}
            </p>
        </div>
        
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 50px; max-width: 1100px; margin-left: auto; margin-right: auto; align-items: stretch;">
            <div class="overview-box" style="height: 100%;">
                <div class="overview-box-icon">✦</div>
                <div style="color: var(--text-secondary); font-size: 1rem; line-height: 1.5; padding-top: 4px;">
                    {get_text('home_box1', lang)}
                </div>
            </div>
            <div class="overview-box" style="height: 100%;">
                <div class="overview-box-icon">✦</div>
                <div style="color: var(--text-secondary); font-size: 1rem; line-height: 1.5; padding-top: 4px;">
                    {get_text('home_box2', lang)}
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    @staticmethod
    def sidebar_branding():
        lang = _get_current_lang()
        orbit_icon = get_icon("orbit", size=40, color="#7FB135")
        brand_html = f"""
        <div class="sidebar-branding-bottom">
            <hr>
            <div style='text-align: center; padding: 4px 0 0 0;'>
            <div style='display: flex; justify-content: center;'>{orbit_icon}</div>
            <div style='color: var(--text-main); font-weight: 800; font-size: 1.1rem; margin-top: 6px;'>{get_text('brand_name', lang)}</div>
            <div style='color: var(--text-secondary); font-size: 0.6rem; letter-spacing: 2.5px; text-transform: uppercase; margin-top: 2px;'>{get_text('brand_sub', lang)}</div>
        </div>
        </div>
        """
        st.sidebar.markdown(brand_html, unsafe_allow_html=True)

    @staticmethod
    def footer():
        lang = _get_current_lang()
        st.markdown(f"""
        <div class="page-footer">
            {get_text('footer_text', lang)}
        </div>
        """, unsafe_allow_html=True)

    @staticmethod
    def workspace_status(active_file):
        """Renders the Workspace Status and Data Inventory bars."""
        st.markdown(f"""
            <div class="status-bar" style="background: linear-gradient(90deg, rgba(242, 112, 36, 0.15) 0%, rgba(242, 112, 36, 0.05) 100%); border: 1px solid rgba(242, 112, 36, 0.2);">
            <span class="status-label" style="color: var(--accent-orange);">Current Workspace:</span>
            <span class="status-value" style="color: var(--text-main); font-weight: 700;">{active_file}</span>
        </div>
        """, unsafe_allow_html=True)

    @staticmethod
    def file_inventory(active_file):
        """
        Renders the File Inventory with search, actions, and clean state management.
        """
        lang = _get_current_lang()
        library = get_data_library()

        # === CONTROL BAR ===
        total_assets = len(library)
        st.markdown("<div style='margin-top: 12px;'></div>", unsafe_allow_html=True)
        
        # 1. Inventory Status (Header)
        st.markdown(f"""
            <div class="status-bar" style="background: linear-gradient(90deg, rgba(59, 130, 246, 0.15) 0%, rgba(59, 130, 246, 0.05) 100%); justify-content: space-between; border: 1px solid rgba(59, 130, 246, 0.2); margin-bottom: 20px;">
                <div style="display:flex; align-items:center;">
                    <span class="status-label" style="color: var(--accent-blue);">{get_text('total_assets', lang).upper()}:</span>
                </div>
                <span style="color: var(--text-muted); font-size: 0.75rem;"><b style="color:var(--accent-blue)">{get_text('files_count', lang, count=total_assets)}</b></span>
            </div>
        """, unsafe_allow_html=True)

        col_search, col_upload = st.columns([3, 1], gap="small")
        
        with col_search:
            search_query = st.text_input(
                "Search", 
                placeholder=get_text('search_placeholder', lang), 
                label_visibility="collapsed",
                key="search_inventory"
            )
        with col_upload:
            if st.button(f":material/upload: {get_text('upload_file', lang)}", type="primary", use_container_width=True, key="btn_upload"):
                upload_dialog()

        # === DATA PROCESSING ===
        lib_df = process_inventory(library, search_query)

        # === TABLE HEADER ===
        st.markdown("<div style='margin-top: 20px;'></div>", unsafe_allow_html=True)
        h1, h2, h3, h_actions = st.columns([3.0, 0.9, 1.2, 2.0])
        h1.markdown(f"<div class='inventory-header'>{get_text('file_name', lang)}</div>", unsafe_allow_html=True)
        h2.markdown(f"<div class='inventory-header'>{get_text('size', lang)}</div>", unsafe_allow_html=True)
        h3.markdown(f"<div class='inventory-header'>{get_text('date', lang)}</div>", unsafe_allow_html=True)
        h_actions.markdown(f"<div class='inventory-header'>{get_text('actions', lang)}</div>", unsafe_allow_html=True)
        st.markdown("<hr style='margin: 6px 0 12px 0; border:0; border-top:1px solid rgba(255,255,255,0.05);'></hr>", unsafe_allow_html=True)

        # === TABLE ROWS ===
        if lib_df.empty:
            st.info(get_text('no_files_match', lang))
            return

        # Pagination Logic
        ITEMS_PER_PAGE = 5
        if "inventory_page" not in st.session_state:
            st.session_state.inventory_page = 1
            
        total_files = len(lib_df)
        total_pages = max(1, (total_files - 1) // ITEMS_PER_PAGE + 1)
        
        # Ensure active page is valid
        if st.session_state.inventory_page > total_pages:
            st.session_state.inventory_page = total_pages
        
        current_page = st.session_state.inventory_page
        start_idx = (current_page - 1) * ITEMS_PER_PAGE
        end_idx = start_idx + ITEMS_PER_PAGE
        
        # Slice Data
        paginated_df = lib_df.iloc[start_idx:end_idx]

        for _, row in paginated_df.iterrows():
            is_active = (row['name'] == active_file)
            
            # Active State Styling
            if is_active:
                name_display = f"<span style='color:var(--accent-blue); font-weight:700'>● &nbsp;{row['name']}</span>"
            else:
                name_display = row['name']
            
            # Column Layout (4 action buttons: Preview, Activate, Download, Delete)
            c1, c2, c3, c4, c5, c6, c7 = st.columns([3.0, 0.9, 1.2, 0.5, 0.5, 0.5, 0.5], gap="small")
            
            # Data Columns
            c1.markdown(f"<div class='inventory-cell'>{name_display}</div>", unsafe_allow_html=True)
            c2.markdown(f"<div class='inventory-cell'>{row['size']}</div>", unsafe_allow_html=True)
            c3.markdown(f"<div class='inventory-cell inventory-cell-muted'>{row['date']}</div>", unsafe_allow_html=True)
            
            # Action Buttons
            if c4.button(":material/visibility:", key=f"btn_prev_{row['name']}", use_container_width=True, help=get_text('preview', lang)):
                try:
                    with st.spinner("Loading..."):
                        st.session_state.preview_df = load_and_standardize(row['name'], _file_mtime=_get_file_mtime(row['name']))
                        st.session_state.preview_name = row['name']
                        st.rerun()
                except Exception as e:
                    st.error(f"Error: {e}")
            
            if c5.button(":material/check_circle:", key=f"btn_act_{row['name']}", use_container_width=True, help=get_text('active', lang)):
                st.session_state.active_file = row['name']
                st.session_state.scroll_to_modules = True
                st.rerun()

            # Download button — reads the file bytes on-demand
            try:
                from modules.core.file_manager import UPLOADS_DIR
                import os
                file_path = os.path.join(UPLOADS_DIR, row['name'])
                with open(file_path, "rb") as fp:
                    file_bytes = fp.read()
                c6.download_button(
                    label=":material/download:",
                    data=file_bytes,
                    file_name=row['name'],
                    mime="text/csv",
                    key=f"btn_dl_{row['name']}",
                    use_container_width=True,
                    help="Download"
                )
            except Exception:
                c6.markdown("<div style='height:40px;'></div>", unsafe_allow_html=True)

            # Prevent deleting the currently active file
            if is_active:
                c7.markdown("<div style='height:40px;'></div>", unsafe_allow_html=True)
            else:
                if c7.button(":material/delete:", key=f"btn_del_{row['name']}", use_container_width=True, help=get_text('delete', lang)):
                    delete_data(row['name'])
                    if st.session_state.get('active_file') == row['name']:
                        st.session_state.active_file = get_text('no_data_loaded', lang)
                    st.rerun()

        # Pagination Controls
        if total_pages > 1:
            st.markdown("<div style='margin-top: 15px;'></div>", unsafe_allow_html=True)
            p1, p2, p3 = st.columns([1, 2, 1])
            
            with p1:
                if current_page > 1:
                    if st.button(f":material/chevron_left: {get_text('previous', lang)}", key="nav_prev", use_container_width=True):
                        st.session_state.inventory_page -= 1
                        st.rerun()
                        
            with p2:
                st.markdown(f"<div style='text-align:center; padding-top:8px;' class='pagination-info'>{get_text('page_info', lang, current=current_page, total=total_pages)}</div>", unsafe_allow_html=True)
                
            with p3:
                if current_page < total_pages:
                    if st.button(f":material/chevron_right: {get_text('next', lang)}", key="nav_next", use_container_width=True):
                        st.session_state.inventory_page += 1
                        st.rerun()

    @staticmethod
    def metric_card(label, value, delta="", glow="blue"):
        """
        Canonical metric card — glassmorphism, animated glow border.
        Single source of truth used by all pages.
        glow: 'blue' | 'green' | 'orange' | 'red'
        """
        color_map = {
            "blue":   "var(--accent-blue)",
            "green":  "var(--accent-green)",
            "orange": "var(--accent-orange)",
            "red":    "var(--accent-red)",
        }
        val_color = color_map.get(glow, "var(--text-main)")
        st.markdown(f"""
            <div class="audit-metric glow-{glow}">
                <div class="metric-label">{label}</div>
                <div class="metric-value" style="color: {val_color};">{value}</div>
                <div class="metric-delta" style="color: var(--text-secondary);">{delta}</div>
            </div>
        """, unsafe_allow_html=True)

    @staticmethod
    def preview_panel():
        """Renders the Data Preview Panel (Lazy Loaded)"""
        lang = _get_current_lang()
        if "preview_df" in st.session_state and "preview_name" in st.session_state:
            df = st.session_state.preview_df
            st.markdown("<div style='margin-top: 20px;'></div>", unsafe_allow_html=True)
            
            # Calculate Metrics (DELEGATED TO DATA ENGINE)
            metrics = compute_dataset_metrics(df)
            
            # Container with Glassmorphism
            with st.container():
                # Header
                st.markdown(f"""
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px;">
                    <div style="display:flex; align-items:center; gap:12px;">
                        <div style="background:rgba(59, 130, 246, 0.2); width:42px; height:42px; border-radius:10px; display:flex; align-items:center; justify-content:center;">
                            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="var(--accent-blue)" stroke-width="2"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path><polyline points="14 2 14 8 20 8"></polyline><line x1="16" y1="13" x2="8" y2="13"></line><line x1="16" y1="17" x2="8" y2="17"></line><polyline points="10 9 9 9 8 9"></polyline></svg>
                        </div>
                        <div>
                            <div style="color: var(--text-secondary); font-size: 0.7rem; text-transform:uppercase; letter-spacing:1px; font-weight:700;">{get_text('preview_dataset', lang)}</div>
                            <div style="color: var(--text-main); font-weight: 700; font-size: 1.1rem; letter-spacing: -0.5px;">{st.session_state.preview_name}</div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # Metrics Grid (Styled)
                m1, m2, m3, m4, m5 = st.columns(5, gap="small")

                with m1: UiComponents.metric_card(get_text('rows', lang),         f"{metrics['rows']:,}",          glow="blue")
                with m2: UiComponents.metric_card(get_text('columns', lang),      f"{metrics['cols']}",            glow="blue")
                with m3: UiComponents.metric_card(get_text('memory', lang),       f"{metrics['memory_mb']:.1f} MB", glow="blue")
                with m4: UiComponents.metric_card(get_text('duplicates', lang),   f"{metrics['duplicates']}",      glow="red"   if metrics['duplicates'] > 0  else "green")
                with m5: UiComponents.metric_card(get_text('missing_data', lang), f"{metrics['missing_pct']:.1f}%", glow="red"  if metrics['missing_pct'] > 0 else "green")
                
                st.markdown("<div style='margin-bottom: 24px;'></div>", unsafe_allow_html=True)

                # Tabs
                tab1, tab2 = st.tabs(["Data Preview", "Column Analysis"])
                
                with tab1:
                    st.dataframe(df.head(10), use_container_width=True, height=350)
                
                with tab2:
                    report_df = generate_column_report(df)
                    st.dataframe(report_df, use_container_width=True, height=350)
                
                st.markdown("<div style='margin-top: 15px;'></div>", unsafe_allow_html=True)
                
                # Footer Action
                _, c_close = st.columns([4, 1])
                with c_close:
                    if st.button(f":material/close: {get_text('close_preview', lang)}", key="close_preview", use_container_width=True):
                        del st.session_state.preview_df
                        del st.session_state.preview_name
                        st.rerun()

                st.markdown("</div>", unsafe_allow_html=True)

    @staticmethod
    def feature_navigation(num_records, lang):
        """
        Renders the complete Feature Navigation section:
        - 3-Step Analytical Journey cards
        - Data Anatomy column
        - Research Objectives column
        All text sourced from localization, styles from styles.py.
        """
        t = lambda key: get_text(key, lang)

        # -- Tag helpers --
        quant_tags = "".join(f'<span class="anatomy-tag">{tag.strip()}</span>' for tag in t('overview_anatomy_quant_tags').split(","))
        cat_prof_tags = "".join(f'<span class="anatomy-tag">{tag.strip()}</span>' for tag in t('overview_anatomy_cat_prof_tags').split(","))
        cat_demo_tags = "".join(f'<span class="anatomy-tag">{tag.strip()}</span>' for tag in t('overview_anatomy_cat_demo_tags').split(","))
        cat_fam_tags = "".join(f'<span class="anatomy-tag">{tag.strip()}</span>' for tag in t('overview_anatomy_cat_fam_tags').split(","))
        
        fin_tags = "".join(f'<span class="anatomy-tag">{tag.strip()}</span>' for tag in t('overview_anatomy_fin_tags').split(","))
        target_tags = "".join(f'<span class="anatomy-tag anatomy-tag-red">{val.strip()}</span>' for val in t('overview_anatomy_target_value').split(","))

        # -- Objective icons from centralized registry --
        icon_bar_chart = get_icon("bar_chart", size=22)
        icon_clock = get_icon("clock", size=22)
        icon_zap = get_icon("zap", size=22)
        icon_heart_pulse = get_icon("heart_pulse", size=22)
        icon_briefcase = get_icon("briefcase", size=22)
        icon_users = get_icon("users", size=22)
        icon_heart = get_icon("heart", size=22)
        icon_home = get_icon("home", size=22)

        html = f"""
<div id='module-selection-anchor'></div>
<div class="section-title">{t('overview_section_journey')}</div>
<div class="section-divider"></div>

<div class="cards-grid">
    <a target="_self" class="journey-card journey-card-1">
        <div class="journey-step step-1">1</div>
        <div class="journey-title">{t('overview_journey_audit_title')}</div>
        <div class="journey-desc">{t('overview_journey_audit_desc')}</div>
    </a>
    <a target="_self" class="journey-card journey-card-2">
        <div class="journey-step step-2">2</div>
        <div class="journey-title">{t('overview_journey_preprocess_title')}</div>
        <div class="journey-desc">{t('overview_journey_preprocess_desc')}</div>
    </a>
    <a target="_self" class="journey-card journey-card-3">
        <div class="journey-step step-3">3</div>
        <div class="journey-title">{t('overview_journey_eda_title')}</div>
        <div class="journey-desc">{t('overview_journey_eda_desc')}</div>
    </a>
    <a target="_self" class="journey-card journey-card-4">
        <div class="journey-step step-4">4</div>
        <div class="journey-title">{t('overview_journey_feat_title')}</div>
        <div class="journey-desc">{t('overview_journey_feat_desc')}</div>
    </a>
</div>

<div class="two-columns">
    <!-- Left Column: Data Anatomy -->
    <div class="column-wrapper">
        <div class="header-row">
            <div class="col-title">{t('overview_col_anatomy')}</div>
            <div class="records-badge">{num_records} {t('overview_records_suffix')}</div>
        </div>

        <div class="column-boxes">
            <div class="anatomy-box">
                <div class="anatomy-title">{t('overview_anatomy_quant_title')}</div>
                <div class="anatomy-tags">{quant_tags}</div>
                <div class="anatomy-note">{t('overview_anatomy_quant_note')}</div>
            </div>

            <div class="anatomy-box anatomy-box-green">
                <div class="anatomy-title anatomy-title-green" style="margin-bottom: 12px;">{t('overview_anatomy_cat_title')}</div>
                
                <div style="margin-bottom: 12px;">
                    <div style="color: var(--text-secondary); font-size: 0.8rem; text-transform: uppercase; font-weight: 700; letter-spacing: 0.5px; margin-bottom: 6px; display: flex; align-items: center; gap: 6px;">
                        <span style="color: var(--accent-green);">-</span> {t('overview_anatomy_cat_prof_label')}
                    </div>
                    <div class="anatomy-tags" style="padding-left: 14px;">{cat_prof_tags}</div>
                </div>

                <div style="margin-bottom: 12px;">
                    <div style="color: var(--text-secondary); font-size: 0.8rem; text-transform: uppercase; font-weight: 700; letter-spacing: 0.5px; margin-bottom: 6px; display: flex; align-items: center; gap: 6px;">
                        <span style="color: var(--accent-green);">-</span> {t('overview_anatomy_cat_demo_label')}
                    </div>
                    <div class="anatomy-tags" style="padding-left: 14px;">{cat_demo_tags}</div>
                </div>

                <div>
                    <div style="color: var(--text-secondary); font-size: 0.8rem; text-transform: uppercase; font-weight: 700; letter-spacing: 0.5px; margin-bottom: 6px; display: flex; align-items: center; gap: 6px;">
                        <span style="color: var(--accent-green);">-</span> {t('overview_anatomy_cat_fam_label')}
                    </div>
                    <div class="anatomy-tags" style="padding-left: 14px;">{cat_fam_tags}</div>
                </div>
            </div>

            <div class="anatomy-box anatomy-box-blue">
                <div class="anatomy-title anatomy-title-blue">{t('overview_anatomy_fin_title')}</div>
                <div class="anatomy-tags">{fin_tags}</div>
                <div class="anatomy-note">{t('overview_anatomy_fin_note')}</div>
            </div>

            <div class="anatomy-box anatomy-box-red">
                <div class="anatomy-title anatomy-title-red">{t('overview_anatomy_target_title')}</div>
                <div style="display: flex; align-items: center; gap: 12px; margin-top: 8px;">
                    <span style="color: white; font-weight: bold; font-size: 0.95rem;">{t('overview_anatomy_target_label')}</span>
                    {target_tags}
                </div>
            </div>
        </div>
    </div>

    <!-- Right Column: Research Objectives -->
    <div class="column-wrapper">
        <div class="header-row">
            <div class="col-title">{t('overview_col_objectives')}</div>
        </div>

        <div class="column-boxes">
            <!-- GROUP 1 -->
            <div class="anatomy-title anatomy-title-blue" style="margin-bottom: 4px; margin-top: 8px;">{t('overview_obj_grp1_title')}</div>
            
            <div class="objective-card objective-card-blue">
                <div class="objective-icon icon-blue">{icon_bar_chart}</div>
                <div>
                    <div class="objective-title">{t('overview_obj_grp1_1_title')}</div>
                    <div class="objective-desc">{t('overview_obj_grp1_1_desc')}</div>
                </div>
            </div>

            <div class="objective-card objective-card-blue">
                <div class="objective-icon icon-blue">{icon_clock}</div>
                <div>
                    <div class="objective-title">{t('overview_obj_grp1_2_title')}</div>
                    <div class="objective-desc">{t('overview_obj_grp1_2_desc')}</div>
                </div>
            </div>

            <div class="objective-card objective-card-blue">
                <div class="objective-icon icon-blue">{icon_zap}</div>
                <div>
                    <div class="objective-title">{t('overview_obj_grp1_3_title')}</div>
                    <div class="objective-desc">{t('overview_obj_grp1_3_desc')}</div>
                </div>
            </div>

            <div class="objective-card objective-card-blue">
                <div class="objective-icon icon-blue">{icon_heart_pulse}</div>
                <div>
                    <div class="objective-title">{t('overview_obj_grp1_4_title')}</div>
                    <div class="objective-desc">{t('overview_obj_grp1_4_desc')}</div>
                </div>
            </div>

            <div class="objective-card objective-card-blue">
                <div class="objective-icon icon-blue">{icon_briefcase}</div>
                <div>
                    <div class="objective-title">{t('overview_obj_grp1_5_title')}</div>
                    <div class="objective-desc">{t('overview_obj_grp1_5_desc')}</div>
                </div>
            </div>

            <!-- GROUP 2 -->
            <div class="anatomy-title anatomy-title-orange" style="margin-bottom: 4px; margin-top: 16px;">{t('overview_obj_grp2_title')}</div>

            <div class="objective-card objective-card-orange">
                <div class="objective-icon icon-orange">{icon_users}</div>
                <div>
                    <div class="objective-title">{t('overview_obj_grp2_1_title')}</div>
                    <div class="objective-desc">{t('overview_obj_grp2_1_desc')}</div>
                </div>
            </div>

            <div class="objective-card objective-card-orange">
                <div class="objective-icon icon-orange">{icon_heart}</div>
                <div>
                    <div class="objective-title">{t('overview_obj_grp2_2_title')}</div>
                    <div class="objective-desc">{t('overview_obj_grp2_2_desc')}</div>
                </div>
            </div>

            <div class="objective-card objective-card-orange">
                <div class="objective-icon icon-orange">{icon_home}</div>
                <div>
                    <div class="objective-title">{t('overview_obj_grp2_3_title')}</div>
                    <div class="objective-desc">{t('overview_obj_grp2_3_desc')}</div>
                </div>
            </div>
        </div>
    </div>
</div>
"""
        # Strip leading whitespace to prevent Streamlit markdown code-block rendering
        html = "\n".join(line.strip() for line in html.split("\n"))
        st.markdown(html, unsafe_allow_html=True)

    # ==============================================================================
    # AUDIT PAGE COMPONENTS
    # ==============================================================================

    # Backward-compatible alias — kept so any lingering imports of audit_metric still work.
    audit_metric = metric_card

    @staticmethod
    def scan_animation(message=None):
        """Renders the scanning / pulse progress animation."""
        lang = _get_current_lang()
        if message is None:
            message = get_text('scanning', lang)
            
        st.markdown(f"""
            <div style="padding: 24px 0;">
                <div class="scan-label">{message}</div>
                <div class="scan-bar"></div>
            </div>
        """, unsafe_allow_html=True)

    @staticmethod
    def outlier_inspector(df: pd.DataFrame, lang: str, key_prefix: str = "audit"):
        """
        Reusable component for visually inspecting outliers with smart method recommendation.
        Renders the controls, the distribution plot with bell curve, and the flagged data table.
        """
        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
        if not numeric_cols:
            st.info(get_text('no_numeric_distribution', lang))
            return None

        # Calculate Smart Recommendation first
        selected_col = st.selectbox(
            get_text('select_column', lang), 
            numeric_cols,
            key=f"{key_prefix}_outlier_col",
            index=None,
            placeholder="Choose a numeric column to inspect..."
        )
        
        series = df[selected_col] if selected_col else None
        
        if series is not None:
            smart_eval = audit_engine.evaluate_outlier_method(series, lang)
            outlier_method = smart_eval["method"]
            method_hint_keys = {"IQR": "hint_iqr", "Z-Score": "hint_zscore", "Modified Z-Score": "hint_modified_zscore"}
            
            # Fetch Safe Zones for display
            safe_zones = audit_engine._get_safe_zones()
            sz_text = ""
            for zk, zv in safe_zones.items():
                if zk.lower().replace(" ", "_") == selected_col.lower().replace(" ", "_"):
                    lo, hi = zv.get("min"), zv.get("max")
                    if lo is not None and hi is not None:
                        sz_text = f" <b>(Safe Zone: {lo} ➔ {hi})</b>"
                    elif lo is not None:
                        sz_text = f" <b>(Safe Zone: ≥ {lo})</b>"
                    elif hi is not None:
                        sz_text = f" <b>(Safe Zone: ≤ {hi})</b>"
                    break
                    
            st.markdown(f"""
                <div style="background: rgba(91, 134, 229, 0.1); padding: 12px 16px; border-radius: 8px; border-left: 3px solid var(--accent-blue); margin-bottom: 16px; display: flex; flex-direction: column; gap: 4px;">
                    <div style="font-size: 0.95rem;"><strong>✨ {get_text('smart_recommendation', lang)}: {outlier_method}</strong></div>
                    <div style="font-size: 0.85rem; color: var(--text-secondary);">{smart_eval['reason']}</div>
                    <div style="font-size: 0.8rem; color: var(--text-muted); margin-top: 4px;">{get_text(method_hint_keys[outlier_method], lang)}</div>
                    <div style="font-size: 0.8rem; color: var(--status-warning); margin-top: 8px; border-top: 1px dashed rgba(255,255,255,0.1); padding-top: 8px; display: flex; align-items: flex-start; gap: 6px;">
                        <div>{get_text('safe_zone_outlier_note', lang)}{sz_text}</div>
                    </div>
                </div>
            """, unsafe_allow_html=True)

            method_map = {"IQR": "iqr", "Z-Score": "zscore", "Modified Z-Score": "modified_zscore"}
            risk_df, total_outliers = audit_engine.get_risk_records(df, selected_col, method=method_map[outlier_method])
            
            # Plot Distribution + Bell Curve Overlay + Boundaries
            fig = visualizer.plot_outlier_distribution(series, risk_df, method=method_map[outlier_method], lang=lang)
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

            if not risk_df.empty:
                st.markdown(f"""
                    <div style="display:flex; gap:10px; flex-wrap:wrap; margin-bottom:14px; margin-top:14px;">
                        <span class="status-badge badge-red">{get_text('flagged_rows', lang, count=total_outliers)}</span>
                        <span class="status-badge badge-blue">{get_text('column_label', lang, col=selected_col)}</span>
                        <span class="status-badge badge-green">{get_text('method_label', lang, method=outlier_method)}</span>
                    </div>
                """, unsafe_allow_html=True)
                
                if total_outliers > len(risk_df):
                    st.caption(f":material/info: Showing top {len(risk_df)} most extreme outliers of {total_outliers}.")
                    
                st.dataframe(risk_df, use_container_width=True, height=250, hide_index=True)
            else:
                st.success(get_text('no_outliers_detected', lang, col=selected_col, method=outlier_method))
        else:
            st.info("Select a column to evaluate its distribution and detect outliers.")
        
        return risk_df if 'risk_df' in locals() else pd.DataFrame()

    # ==============================================================================
    # PREPROCESSING COMPONENTS
    # ==============================================================================



    @staticmethod
    def pipeline_card():
        """Renders the full-width pipeline step-card grid with hover effects and tab linking."""
        WARN = STATUS_COLORS["warning"]["hex"]
        INFO = STATUS_COLORS["neutral"]["hex"]
        OK   = STATUS_COLORS["success"]["hex"]
        PURP = STATUS_COLORS["info"]["hex"]

        TAB_MAP = {"1": 0, "2": 0, "3": 1, "4": 1, "5": 2}

        step_defs = [
            ("1", "Garbage to NaN",    "Replace noise & invalid entries with NaN",           WARN, "trash"),
            ("2", "Scrub Text",         "Trim whitespace & normalize text casing",            INFO, "scissors"),
            ("3", "Fill Missing",       "Impute nulls: median (numeric) / mode (categorical)", OK,  "bandaid"),
            ("4", "Drop Duplicates",    "Remove exact duplicate rows from dataset",            WARN, "copy"),
            ("5", "Outlier Treatment",  "Auto-detect & clip outliers (IQR / Z-Score)",         PURP, "ruler"),
        ]

        hover_css = (
            '<style>'
            '.pp-step-card{'
            '  transition:all 0.25s cubic-bezier(0.4,0,0.2,1);cursor:pointer;'
            '}'
            '.pp-step-card:hover{'
            '  transform:translateY(-6px) scale(1.03);'
            '  box-shadow:0 12px 32px rgba(0,0,0,0.35),0 0 24px var(--glow-color);'
            '  border-color:var(--glow-color) !important;z-index:2;'
            '}'
            '.pp-step-card:active{'
            '  transform:translateY(-2px) scale(1.01);'
            '}'
            '</style>'
        )

        def _card(num, title, desc, color, icon_key):
            ico = get_icon(icon_key, 20, color)
            rgb = _pp_hex(color)
            tab_idx = str(TAB_MAP.get(num, 0))
            js = (
                "(function(){"
                "var tabs=window.parent.document.querySelectorAll('[data-baseweb=tab]');" 
                "if(tabs[" + tab_idx + "]){"
                "tabs[" + tab_idx + "].click();"
                "setTimeout(function(){"
                "var h=window.parent.document.querySelector('#pp-detail-anchor');" 
                "if(h)h.scrollIntoView({behavior:'smooth',block:'start'});" 
                "},300);}"
                "})()"
            )
            tpl = (
                '<div class="pp-step-card" style="--glow-color:rgba(__RGB__,0.5);'
                'flex:1;min-width:0;padding:20px 14px 18px;'
                'background:rgba(__RGB__,0.05);'
                'border:1px solid rgba(__RGB__,0.18);border-radius:14px;'
                'display:flex;flex-direction:column;align-items:center;'
                'text-align:center;gap:10px;position:relative;"'
                ' onclick="' + js + '">'
                '<div style="width:44px;height:44px;border-radius:50%;'
                'background:rgba(__RGB__,0.12);border:1.5px solid rgba(__RGB__,0.35);'
                'display:flex;align-items:center;justify-content:center;'
                'box-shadow:0 0 18px rgba(__RGB__,0.15);">'
                '__ICO__</div>'
                '<div style="font-size:0.58rem;font-weight:800;color:__COL__;'
                'text-transform:uppercase;letter-spacing:1.2px;'
                'background:rgba(__RGB__,0.15);border-radius:20px;'
                'padding:2px 8px;">STEP __NUM__</div>'
                '<div style="font-size:0.85rem;font-weight:700;color:white;'
                'line-height:1.3;">__TITLE__</div>'
                '<div style="font-size:0.72rem;color:rgba(255,255,255,0.4);'
                'line-height:1.5;padding:0 4px;">__DESC__</div>'
                '</div>'
            )
            return (tpl
                    .replace('__RGB__', rgb).replace('__ICO__', ico)
                    .replace('__COL__', color).replace('__NUM__', num)
                    .replace('__TITLE__', title).replace('__DESC__', desc))

        def _connector():
            return (
                '<div style="display:flex;align-items:center;'
                'flex-shrink:0;padding-top:22px;">'
                '<div style="height:1px;width:20px;background:'
                'linear-gradient(90deg,rgba(255,255,255,0.05),'
                'rgba(255,255,255,0.18),rgba(255,255,255,0.05));">'
                '</div>'
                '<span style="color:rgba(255,255,255,0.18);'
                'font-size:0.55rem;">&#9654;</span>'
                '<div style="height:1px;width:20px;background:'
                'linear-gradient(90deg,rgba(255,255,255,0.05),'
                'rgba(255,255,255,0.18),rgba(255,255,255,0.05));">'
                '</div>'
                '</div>'
            )

        parts = []
        for i, (num, title, desc, color, icon_key) in enumerate(step_defs):
            parts.append(_card(num, title, desc, color, icon_key))
            if i < len(step_defs) - 1:
                parts.append(_connector())

        zap_icon = get_icon('zap', 18, WARN)
        header_html = (
            '<div style="display:flex;align-items:center;gap:8px;margin-bottom:6px;">'
            + '<span style="color:' + WARN + ';display:inline-flex;align-items:center;">'
            + zap_icon + '</span>'
            + '<span style="font-size:1rem;font-weight:700;color:white;">Fixed Preprocessing Pipeline</span>'
            '<span style="background:rgba(255,255,255,0.08);border-radius:20px;'
            'font-size:0.65rem;padding:2px 9px;font-weight:600;'
            'letter-spacing:0.6px;">5 STEPS</span></div>'
        )
        subtitle_html = (
            '<div style="font-size:0.78rem;color:rgba(255,255,255,0.35);margin-bottom:18px;">'
            'Automated, sequential data cleaning &mdash; click any step to jump to its details.</div>'
        )
        steps_row = (
            '<div style="display:flex;align-items:stretch;gap:0;width:100%;margin-bottom:24px;">'
            + ''.join(parts) + '</div>'
        )
        st.markdown(
            hover_css + '<div style="margin-top:16px;">'
            + header_html + subtitle_html + steps_row + '</div>',
            unsafe_allow_html=True,
        )

    @staticmethod
    def pipeline_done_banner(res_file, rows_before, rows_after, dupes_dropped):
        """Renders the post-completion banner with metrics grid and save info."""
        OK   = STATUS_COLORS["success"]["hex"]
        WARN = STATUS_COLORS["warning"]["hex"]
        rgb_ok   = _pp_hex(OK)
        rgb_warn = _pp_hex(WARN)

        st.markdown(
            f"""
            <div class="pp-done-banner" style="
                background:linear-gradient(135deg,rgba(166,206,57,0.09) 0%,rgba(91,134,229,0.06) 100%);
                border:1px solid rgba({rgb_ok},0.35);border-radius:14px;
                padding:22px 26px;margin-bottom:20px;">
                <div style="color:{OK};font-size:1.05rem;font-weight:800;margin-bottom:16px;
                            display:flex;align-items:center;gap:8px;letter-spacing:-0.3px;">
                    <span style="display:inline-flex;align-items:center;">{get_icon('check_circle',22,OK)}</span>
                    Preprocessing Complete!
                </div>
                <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin-bottom:14px;">
                    <div class="pp-metric-card" style="background:rgba(255,255,255,0.04);
                         border:1px solid rgba(255,255,255,0.08);border-radius:10px;padding:14px;text-align:center;">
                        <div style="color:var(--text-muted);font-size:0.68rem;text-transform:uppercase;
                                    letter-spacing:1px;margin-bottom:4px;">Rows Before</div>
                        <div style="color:white;font-size:1.5rem;font-weight:800;">{rows_before:,}</div>
                    </div>
                    <div class="pp-metric-card" style="background:rgba(255,255,255,0.04);
                         border:1px solid rgba({rgb_ok},0.25);border-radius:10px;padding:14px;text-align:center;">
                        <div style="color:var(--text-muted);font-size:0.68rem;text-transform:uppercase;
                                    letter-spacing:1px;margin-bottom:4px;">Rows After</div>
                        <div style="color:{OK};font-size:1.5rem;font-weight:800;">{rows_after:,}</div>
                    </div>
                    <div class="pp-metric-card" style="background:rgba(255,255,255,0.04);
                         border:1px solid rgba({rgb_warn},0.2);border-radius:10px;padding:14px;text-align:center;">
                        <div style="color:var(--text-muted);font-size:0.68rem;text-transform:uppercase;
                                    letter-spacing:1px;margin-bottom:4px;">Duplicates Dropped</div>
                        <div style="color:{WARN};font-size:1.5rem;font-weight:800;">{dupes_dropped:,}</div>
                    </div>
                </div>
                <div style="color:var(--text-secondary);font-size:0.83rem;
                            border-top:1px solid rgba(255,255,255,0.07);padding-top:12px;
                            display:flex;align-items:center;gap:6px;">
                    {get_icon('download', 14, 'var(--text-muted)')}
                    Saved as <strong style="color:white;margin:0 2px;">{res_file}</strong>
                    &mdash; workspace switched automatically.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    @staticmethod
    def detail_analysis_header():
        """Renders the anchor and heading for the Detailed Preprocessing Analysis section."""
        st.markdown("<div id='pp-detail-anchor' style='height:8px'></div>", unsafe_allow_html=True)
        st.markdown(
            f"<h3 style='color:white;margin-bottom:16px;font-size:1.2rem;font-weight:700;"
            f"display:flex;align-items:center;gap:10px;letter-spacing:-0.3px;'>"
            f"{get_icon('eye', 24, 'rgba(255,255,255,0.8)')} "
            f"<span>Detailed Preprocessing Analysis</span>"
            f"<span style='background:rgba(255,255,255,0.08);border-radius:20px;font-size:0.75rem;"
            f"padding:3px 12px;font-weight:600;letter-spacing:0.5px;margin-left:4px;'>PREVIEW</span></h3>",
            unsafe_allow_html=True,
        )

    @staticmethod
    def comparison_charts(df_original, df_cleaned, column):
        """Renders side-by-side Before/After charts for a specific column."""
        from modules.ui.visualizer import plot_histogram, plot_box
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Before**")
            if column in df_original.columns:
                if pd.api.types.is_numeric_dtype(df_original[column]):
                    fig = plot_histogram(df_original, column, color=STATUS_COLORS["neutral"]["hex"])
                    st.plotly_chart(fig, use_container_width=True)
                    fig_box = plot_box(df_original, column, color=STATUS_COLORS["neutral"]["hex"])
                    st.plotly_chart(fig_box, use_container_width=True)
                else:
                    st.bar_chart(df_original[column].value_counts().head(10))

        with col2:
            st.markdown("**After**")
            if column in df_cleaned.columns:
                if pd.api.types.is_numeric_dtype(df_cleaned[column]):
                    fig = plot_histogram(df_cleaned, column, color=STATUS_COLORS["success"]["hex"])
                    st.plotly_chart(fig, use_container_width=True)
                    fig_box = plot_box(df_cleaned, column, color=STATUS_COLORS["success"]["hex"])
                    st.plotly_chart(fig_box, use_container_width=True)
                else:
                    st.bar_chart(df_cleaned[column].value_counts().head(10))

    @staticmethod
    def method_selectbox(label, method_keys, lang, key):
        """Selectbox with localized labels mapped to internal engine keys."""
        from modules.utils.localization import get_method_labels
        methods = get_method_labels(method_keys, lang)
        keys = list(methods.keys())
        labels = [methods[k][0] for k in keys]
        idx = st.selectbox(label, range(len(keys)), format_func=lambda i: labels[i], key=key)
        method_key = keys[idx]
        st.caption(methods[method_key][1])
        return method_key

    # --------------------------------------------------------------------------
    # PREPROCESSING — TAB RENDERERS
    # Called by pages/preprocessing.py to keep the page module thin.
    # --------------------------------------------------------------------------

    @staticmethod
    def render_scrubber_tab(df: pd.DataFrame) -> None:
        """
        Tab 1 renderer: Data Scrubber.

        Displays Step 1 (garbage value replacement) and Step 2 (text formatting).

        Args:
            df: Working DataFrame (pre-pipeline snapshot).
        """
        from modules.core.audit_engine import _compute_noise_mask, _get_cat_columns
        from modules.utils.theme_manager import STATUS_COLORS

        WARN = STATUS_COLORS["warning"]["hex"]
        INFO = STATUS_COLORS["neutral"]["hex"]
        OK   = STATUS_COLORS["success"]["hex"]

        # Use module-level shared helpers (OPT-9)
        _hex      = _pp_hex
        _key      = _pp_key
        _card     = _pp_card
        _step_hdr = _pp_step_hdr

        cat_cols = _get_cat_columns(df).tolist()

        # ── Step 1: Garbage values ───────────────────────────────────────
        st.markdown(_step_hdr(1, "Garbage Value Replacement", WARN, "trash"), unsafe_allow_html=True)
        noise_rows = []
        for col in cat_cols:
            series = df[col].dropna().astype(str)
            mask   = _compute_noise_mask(series)
            cnt    = int(mask.sum())
            if cnt > 0:
                noise_rows.append({
                    "Column": col, "Affected Cells": cnt,
                    "Examples": ", ".join(f"'{v}'" for v in series[mask].unique()[:4]),
                })

        if noise_rows:
            total = sum(r["Affected Cells"] for r in noise_rows)
            st.markdown(
                f"Detected {_key(f'{total:,} garbage cells')} across "
                f"{_key(str(len(noise_rows)))} columns — they will be replaced with "
                f"{_key('NaN')} so that they are handled as missing values in Step 4.",
                unsafe_allow_html=True,
            )
            st.dataframe(pd.DataFrame(noise_rows), use_container_width=True, hide_index=True,
                         column_config={"Affected Cells": st.column_config.NumberColumn(format="%d")})
        else:
            st.success(":material/check_circle: No garbage or placeholder values detected.")

        st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)

        # ── Step 2: Text formatting ──────────────────────────────────────
        st.markdown(_step_hdr(2, "Text Formatting (Trim + Normalize Casing)", INFO, "type"),
                    unsafe_allow_html=True)
        fmt_rows = []
        for col in cat_cols:
            series = df[col].dropna().astype(str)
            ws_count = int((series != series.str.strip()).sum())
            lm: dict = {}
            for v in series.unique():
                lm.setdefault(v.strip().lower(), []).append(v)
            casing_variants = sum(1 for vs in lm.values() if len(vs) > 1)
            if ws_count > 0 or casing_variants > 0:
                parts = []
                if ws_count:        parts.append(f"{ws_count} with whitespace")
                if casing_variants: parts.append(f"{casing_variants} casing variant groups")
                fmt_rows.append({"Column": col, "Issues": " | ".join(parts)})

        if fmt_rows:
            st.markdown(
                f"Found text quality issues in {_key(str(len(fmt_rows)))} columns. "
                f"Pipeline will {_key('trim whitespace')} then {_key('normalize casing')} "
                f"by choosing the most frequent variant as the canonical form.",
                unsafe_allow_html=True,
            )
            st.dataframe(pd.DataFrame(fmt_rows), use_container_width=True, hide_index=True)
        else:
            st.success(":material/check_circle: All text fields are clean — no formatting issues.")

        st.markdown(_card(
            f"<span style='display:inline-flex;align-items:center;vertical-align:top;margin-right:6px;'>"
            f"{get_icon('zap', 16, INFO)}</span> {_key('Action:')} Garbage values → replaced with NaN"
            f" &nbsp;·&nbsp; Whitespace → stripped &nbsp;·&nbsp;"
            f" Mixed casing → canonicalized to most-frequent form",
            INFO,
        ), unsafe_allow_html=True)

    @staticmethod
    def render_missing_and_dupes_tab(df: pd.DataFrame) -> None:
        """
        Tab 2 renderer: Missing Values and Duplicate Rows.

        Left column  — per-column missing count, skewness, fill strategy.
        Right column — total duplicate count and drop description.

        Args:
            df: Working DataFrame (pre-pipeline snapshot).
        """
        from modules.utils.theme_manager import STATUS_COLORS

        INFO = STATUS_COLORS["neutral"]["hex"]
        WARN = STATUS_COLORS["warning"]["hex"]
        OK   = STATUS_COLORS["success"]["hex"]

        # Use module-level shared helpers (OPT-9)
        _hex      = _pp_hex
        _key      = _pp_key
        _card     = _pp_card
        _step_hdr = _pp_step_hdr

        col_miss, col_dupe = st.columns(2, gap="large")

        with col_miss:
            st.markdown(_step_hdr(3, "Handle Missing Values", INFO, "bandaid"), unsafe_allow_html=True)
            missing_cols = df.columns[df.isnull().any()].tolist()

            if missing_cols:
                total_rows   = len(df)
                missing_data = []
                for mc in missing_cols:
                    cnt   = int(df[mc].isnull().sum())
                    dtype = "Numeric" if pd.api.types.is_numeric_dtype(df[mc]) else "Categorical"
                    skew_display = None
                    if dtype == "Numeric":
                        series   = df[mc].dropna()
                        skew_val = series.skew() if len(series) > 2 else float("nan")
                        if pd.notna(skew_val):
                            skew_display = float(skew_val)
                        strategy = ("Mean" if (len(series) > 2 and pd.notna(skew_val) and abs(skew_val) < 0.5)
                                    else "Median")
                    else:
                        strategy = "Mode"
                    missing_data.append({"Column": mc, "Type": dtype, "Missing": cnt,
                                         "% Missing": cnt / total_rows * 100,
                                         "Skewness": skew_display, "Strategy": strategy})
                st.dataframe(
                    pd.DataFrame(missing_data).sort_values("Missing", ascending=False),
                    use_container_width=True, hide_index=True,
                    column_config={
                        "Missing":   st.column_config.NumberColumn(format="%d"),
                        "% Missing": st.column_config.ProgressColumn(format="%.1f%%", min_value=0, max_value=100),
                        "Skewness":  st.column_config.NumberColumn("Skewness", format="%.3f"),
                        "Strategy":  st.column_config.TextColumn("Fill Strategy"),
                    },
                )
                st.markdown(_card(
                    f"<span style='display:inline-flex;align-items:center;vertical-align:top;margin-right:6px;'>"
                    f"{get_icon('ruler', 16, INFO)}</span> {_key('Numeric columns')} → filled with "
                    f"{_key('Mean / Median')} (auto-selected via skewness)<br>"
                    f"<span style='display:inline-flex;align-items:center;vertical-align:top;margin-right:6px;'>"
                    f"{get_icon('type', 16, INFO)}</span> {_key('Categorical columns')} → filled with "
                    f"{_key('Mode')} (most frequent value)",
                    INFO,
                ), unsafe_allow_html=True)
            else:
                st.success(":material/check_circle: No missing values in the dataset.")

        with col_dupe:
            st.markdown(_step_hdr(4, "Drop Duplicate Rows", WARN, "copy"), unsafe_allow_html=True)
            dupes = int(df.duplicated().sum())
            total = len(df)

            if dupes > 0:
                pct = dupes / total * 100
                st.markdown(
                    f"<div style='font-size:2.5rem;font-weight:800;color:{WARN};line-height:1;'>"
                    f"{dupes:,}</div>"
                    f"<div style='color:var(--text-secondary);font-size:0.85rem;margin-top:4px;'>"
                    f"duplicate rows detected ({pct:.1f}% of {total:,} total)</div>",
                    unsafe_allow_html=True,
                )
                st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
                st.markdown(_card(
                    f"<span style='display:inline-flex;align-items:center;vertical-align:top;margin-right:6px;'>"
                    f"{get_icon('trash', 16, WARN)}</span> All {_key(f'{dupes:,} duplicate rows')} will be dropped, "
                    f"retaining only the first occurrence of each unique record.",
                    WARN,
                ), unsafe_allow_html=True)
            else:
                st.success(":material/check_circle: No duplicate rows found.")
                st.markdown(_card(
                    f"<span style='display:inline-flex;align-items:center;vertical-align:middle;margin-right:6px;'>"
                    f"{get_icon('check_circle', 16, OK)}</span> "
                    f"This step will be skipped — dataset has no duplicate rows.",
                    OK,
                ), unsafe_allow_html=True)

    @staticmethod
    def render_outlier_tab(df: pd.DataFrame, compute_preview_row_fn) -> None:
        """
        Tab 3 renderer: Outlier Treatment preview.

        Uses ``compute_preview_row_fn`` (passed from preprocessing.py) so that
        this renderer stays decoupled from audit_engine internals.

        UI design:
          • Amber note callout explaining the priority logic (no threshold shown).
          • Per-column dataframe with auto-method, outlier count, and action.
          • Legend with an animated rainbow-border Safe Zone card.

        Args:
            df:                    Working DataFrame.
            compute_preview_row_fn: ``(df, col, safe_zones, threshold) → dict | None``.
        """
        from modules.core.audit_engine import _get_safe_zones
        from modules.utils.theme_manager import STATUS_COLORS

        INFO = STATUS_COLORS["neutral"]["hex"]
        WARN = STATUS_COLORS["warning"]["hex"]
        PURP = STATUS_COLORS["info"]["hex"]

        # Use module-level shared helpers (OPT-9)
        _hex      = _pp_hex
        _key      = _pp_key
        _card     = _pp_card
        _step_hdr = _pp_step_hdr


        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
        if not numeric_cols:
            st.info("No numeric columns available for outlier treatment.")
            return

        threshold   = 1.5
        safe_zones  = _get_safe_zones()

        st.markdown(_step_hdr(5, "Outlier Treatment (Auto-Method per Column)", PURP, "ruler"),
                    unsafe_allow_html=True)

        # ── Amber note callout — emphasises Safe Zone priority ────────────
        st.markdown("""
        <div class="pp-outlier-note">
            <div class="pp-outlier-note-icon">⚡</div>
            <div>
                <strong>How the pipeline handles outliers</strong><br>
                Skewness determines the auto-selected detection method per column.
                Values inside <strong>Admin Safe Zones</strong> are
                <strong>always protected</strong> — they are never treated regardless of
                the statistical result. Outliers outside a configured zone are
                <strong>clipped to its bounds</strong>; when no zone is defined,
                values are capped to the method's statistical fence.
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Build preview rows via the shared helper
        rows = []
        for col in numeric_cols:
            row = compute_preview_row_fn(df, col, safe_zones, threshold)
            if row is not None:
                rows.append({
                    "Column":            row["Column"],
                    "Skewness":          row["Skewness"],
                    "Auto Method":       row["Auto Method"],
                    "Outliers Detected": row["Outliers Detected"],
                    "Action":            row["Action"],
                })

        if rows:
            st.dataframe(
                pd.DataFrame(rows),
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Skewness":          st.column_config.NumberColumn(format="%.3f"),
                    "Outliers Detected": st.column_config.NumberColumn(format="%d"),
                },
            )

            # ── Method legend: Safe Zone card uses animated rainbow border ──
            st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
            leg0, leg1, leg2, leg3 = st.columns(4)

            with leg0:
                # Animated rainbow-gradient border card (defined in PREPROCESSING_STYLES)
                st.markdown(
                    "<div class='pp-legend-safezone'>"
                    "<div class='pp-legend-safezone-inner'>"
                    "<div style='font-size:0.72rem;font-weight:800;text-transform:uppercase;"
                    "letter-spacing:0.8px;background:linear-gradient(90deg,#F59E0B,#EC4899,#8B5CF6);"
                    "-webkit-background-clip:text;-webkit-text-fill-color:transparent;"
                    "margin-bottom:6px;'>🛡 Safe Zones</div>"
                    "<div style='color:#FDE68A;font-size:0.78rem;font-weight:700;margin-bottom:4px;'>"
                    "Priority Override</div>"
                    "<div style='color:var(--text-muted);font-size:0.75rem;line-height:1.5;'>"
                    "Protects valid data<br>Clips outliers to bounds</div>"
                    "</div></div>",
                    unsafe_allow_html=True,
                )
            with leg1:
                st.markdown(_card(
                    f"<span style='color:{INFO};font-size:1.2rem;display:inline-flex;margin-right:4px;'>●</span>"
                    f" {_key('Z-Score Cap')}<br>|skew| &lt; 0.5<br>Symmetric distributions",
                    INFO,
                ), unsafe_allow_html=True)
            with leg2:
                st.markdown(_card(
                    f"<span style='color:{WARN};font-size:1.2rem;display:inline-flex;margin-right:4px;'>●</span>"
                    f" {_key('IQR Cap')}<br>0.5 ≤ |skew| ≤ 1.0<br>Moderately skewed",
                    WARN,
                ), unsafe_allow_html=True)
            with leg3:
                st.markdown(_card(
                    f"<span style='color:{PURP};font-size:1.2rem;display:inline-flex;margin-right:4px;'>●</span>"
                    f" {_key('Modified Z-Score Cap')}<br>|skew| &gt; 1.0<br>Highly skewed / heavy-tailed",
                    PURP,
                ), unsafe_allow_html=True)
        else:
            st.info("No numeric columns had enough data for outlier analysis.")

    # ==============================================================================
    # AUTH COMPONENTS
    # ==============================================================================


    @staticmethod
    def login_form():
        """Renders a horizontal 2-column login: brand left, form right."""
        lang = _get_current_lang()
        orbit_icon = get_icon("orbit", size=40, color="#7FB135")

        # Background orbs
        st.markdown("""
        <div class="login-bg">
            <div class="login-orb login-orb-1"></div>
            <div class="login-orb login-orb-2"></div>
            <div class="login-orb login-orb-3"></div>
        </div>
        """, unsafe_allow_html=True)

        # 2-column layout: branding | form
        _, col_brand, col_form, _ = st.columns([0.5, 1.2, 1, 0.5])

        with col_brand:
            st.markdown(f"""
            <div class="login-card" style="height: 100%; display:flex; flex-direction:column; justify-content:center;">
                <div class="login-brand">
                    <div style="display:flex; justify-content:center;">{orbit_icon}</div>
                    <div class="login-brand-name">{get_text('brand_name', lang)}</div>
                    <div class="login-brand-sub">{get_text('brand_sub', lang)}</div>
                </div>
                <div class="login-title">{get_text('login_welcome', lang)}</div>
                <div class="login-subtitle">{get_text('login_instruction', lang)}</div>
            </div>
            """, unsafe_allow_html=True)

        with col_form:
            st.markdown("<div style='height: 60px'></div>", unsafe_allow_html=True)
            with st.form("login_form_ui", border=False):
                username = st.text_input(get_text('username', lang), key="login_username", placeholder=get_text('username', lang), label_visibility="collapsed")
                password = st.text_input(get_text('password', lang), type="password", key="login_password", placeholder=get_text('password', lang), label_visibility="collapsed")
                login_clicked = st.form_submit_button(f":material/login: {get_text('login_btn', lang)}", type="primary", use_container_width=True)

        return username, password, login_clicked

    @staticmethod
    def sidebar_user_info():
        """Renders user avatar, name, role badge, and action buttons in the sidebar."""
        lang = _get_current_lang()
        display_name = st.session_state.get('display_name', '')
        role = st.session_state.get('user_role', 'user')
        username = st.session_state.get('username', '')

        # Get initials for avatar
        initials = "".join([w[0].upper() for w in display_name.split()[:2]]) if display_name else username[0].upper() if username else "U"

        role_label = get_text('role_admin', lang) if role == 'admin' else get_text('role_user', lang)
        role_class = 'sidebar-role-admin' if role == 'admin' else 'sidebar-role-user'
        lang_label_short = "VI" if lang == 'en' else "EN"

        # User card
        st.sidebar.markdown(f"""
        <div class="sidebar-user-info">
            <div class="sidebar-avatar">{initials}</div>
            <div class="sidebar-username">{display_name or username}</div>
            <div><span class="sidebar-role-badge {role_class}">{role_label}</span></div>
        </div>
        """, unsafe_allow_html=True)

        # Action buttons — Material icons, 3 compact columns
        col_lang, col_profile, col_logout = st.sidebar.columns(3)
        with col_lang:
            lang_clicked = st.button(":material/language:", key="btn_sidebar_lang", use_container_width=True, help=f"{get_text('language', lang)} → {lang_label_short}")
        with col_profile:
            profile_clicked = st.button(":material/person:", key="btn_sidebar_profile", use_container_width=True, help=get_text('profile', lang))
        with col_logout:
            logout_clicked = st.button(":material/logout:", key="btn_sidebar_logout", use_container_width=True, help=get_text('logout', lang))

        return lang_clicked, profile_clicked, logout_clicked

    @staticmethod
    def sidebar_ai_chat():
        """Renders an AI Chat Assistant inside a popover in the sidebar."""
        lang = _get_current_lang()
        if 'chat_messages' not in st.session_state:
            st.session_state.chat_messages = []

        user = st.session_state.get('user', {})
        username = user.get('username')
        display_name = user.get('display_name')
        
        # Streamlit avatars must be 1-2 characters or a valid URL/emoji.
        if display_name:
            words = display_name.split()
            initials = words[0][0].upper()
            if len(words) > 1:
                initials += words[1][0].upper()
        elif username:
            initials = username[:2].upper()
        else:
            initials = "U"
            
        # For chat display, Streamlit sometimes fails with string initials due to font/unicode length.
        # Ensure we use an emoji to prevent StreamlitAPIException
        chat_avatar = "👤"

        with st.sidebar:
            with st.popover(f":material/smart_toy: {get_text('ai_assistant', lang)}", use_container_width=True):
                st.markdown(f"**✨ {get_text('ai_assistant', lang)}**")
                
                # Chat container
                chat_container = st.container(height=350)
                with chat_container:
                    if len(st.session_state.chat_messages) == 0:
                        st.caption(get_text('chat_greeting', lang))
                    for msg in st.session_state.chat_messages:
                        with st.chat_message(msg["role"], avatar=msg["role"]):
                            st.markdown(msg["content"])
                
                # Input
                if prompt := st.chat_input(get_text('chat_placeholder', lang), key="ai_chat_input"):
                    st.session_state.chat_messages.append({"role": "user", "content": prompt})
                    with chat_container:
                        with st.chat_message("user", avatar="user"):
                            st.markdown(prompt)
                        
                        # LLM response logic
                        from modules.core.llm_engine import stream_llm_response
                        from modules.core.data_engine import load_and_standardize, _get_file_mtime
                        
                        active_file = st.session_state.get("active_file")
                        df = None
                        if active_file and active_file != get_text('no_data_loaded', lang):
                            df = load_and_standardize(active_file, _file_mtime=_get_file_mtime(active_file))

                        with st.chat_message("assistant", avatar="assistant"):
                            # Thinking UI
                            with st.spinner(get_text('chat_thinking', lang)):
                                response_stream = stream_llm_response(prompt, st.session_state.chat_messages[:-1], df)
                                response = st.write_stream(response_stream)
                            
                        st.session_state.chat_messages.append({"role": "assistant", "content": response})
                        
                        st.rerun()
