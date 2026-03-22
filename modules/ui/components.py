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


def _styled_status(text: str, accent: str = "#7FB135") -> None:
    """Lightweight styled status message — replaces st.success/st.info globally."""
    h = accent.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    rgb = f"{r},{g},{b}"
    st.markdown(
        f'<div style="margin:4px 0 8px 12px; padding:10px 14px;'
        f' background:rgba({rgb},0.03);'
        f' border-left:2px solid rgba({rgb},0.4);'
        f' border-radius:0 8px 8px 0;'
        f' font-size:0.8rem; color:rgba(255,255,255,0.4);">'
        f'<span style="color:rgba({rgb},0.7); font-size:0.85rem;">\u2713</span>'
        f' {text}</div>',
        unsafe_allow_html=True,
    )


def styled_alert(text: str, kind: str = "info") -> None:
    """Styled notification — replaces st.success/error/warning/info globally.

    Args:
        text: Message content (supports HTML).
        kind: 'success' | 'info' | 'warning' | 'error'
    """
    _CFG = {
        "success": ("#7FB135", "check_circle"),
        "info":    ("#3B82F6", "zap"),
        "warning": ("#F59E0B", "alert_triangle"),
        "error":   ("#EF4444", "x_circle"),
    }
    color, icon_key = _CFG.get(kind, _CFG["info"])
    icon_svg = get_icon(icon_key, size=15, color=color)
    h = color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    rgb = f"{r},{g},{b}"
    st.markdown(
        f'<div style="margin:4px 0 8px 0; padding:10px 14px;'
        f' background:rgba({rgb},0.06);'
        f' border-left:2px solid rgba({rgb},0.5);'
        f' border-radius:0 8px 8px 0;'
        f' font-size:0.82rem; color:rgba({rgb},0.85);'
        f' display:flex; align-items:flex-start; gap:8px;">'
        f'<span style="flex-shrink:0;margin-top:1px;">{icon_svg}</span>'
        f'<span style="line-height:1.6;">{text}</span></div>',
        unsafe_allow_html=True,
    )


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
                margin:16px 0 14px 0;"></div>
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
        """Renders the Workspace Status bar."""
        lang = _get_current_lang()
        st.markdown(f"""
            <div class="status-bar" style="background: linear-gradient(90deg, rgba(242, 112, 36, 0.15) 0%, rgba(242, 112, 36, 0.05) 100%); border: 1px solid rgba(242, 112, 36, 0.2);">
            <span class="status-label" style="color: var(--accent-orange);">{get_text('current_workspace', lang)}:</span>
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
            _styled_status(get_text('no_files_match', lang), accent='#3B82F6')
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
                    styled_alert(f"Error: {e}", "error")
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
                # Header — icon from centralized registry
                file_icon = get_icon('file_text', 20, 'var(--accent-blue)')
                st.markdown(f"""
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px;">
                    <div style="display:flex; align-items:center; gap:12px;">
                        <div style="background:rgba(59, 130, 246, 0.2); width:42px; height:42px; border-radius:10px; display:flex; align-items:center; justify-content:center;">
                            {file_icon}
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
                # Tabs — localized labels
                tab1, tab2 = st.tabs([get_text('data_preview', lang), get_text('column_analysis', lang)])
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
        import base64, os
        t = lambda key: get_text(key, lang)

        # -- Load card background images as base64 data URIs --
        ASSETS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "assets")
        OBJ_BG_FILES = [
            "obj_income_drivers.png",
            "obj_demographic_profile.png",
            "obj_education_earnings.png",
            "obj_work_intensity.png",
            "obj_occupation_income.png",
        ]
        bg_data_uris = []
        for bg_file in OBJ_BG_FILES:
            bg_path = os.path.join(ASSETS_DIR, bg_file)
            try:
                with open(bg_path, "rb") as fp:
                    encoded = base64.b64encode(fp.read()).decode("ascii")
                bg_data_uris.append(f"data:image/png;base64,{encoded}")
            except Exception:
                bg_data_uris.append("")

        # -- Build per-card background CSS --
        obj_bg_css = ""
        for idx, data_uri in enumerate(bg_data_uris, start=1):
            if data_uri:
                obj_bg_css += f"""
                .obj-card-bg-{idx}::after {{
                    content: '';
                    position: absolute;
                    top: 0; right: 0; bottom: 0;
                    width: 55%;
                    background: url('{data_uri}') right center / cover no-repeat;
                    opacity: 0.15;
                    pointer-events: none;
                    border-radius: 12px;
                    mask-image: linear-gradient(to left, rgba(0,0,0,1) 10%, transparent 95%);
                    -webkit-mask-image: linear-gradient(to left, rgba(0,0,0,1) 10%, transparent 95%);
                    transition: opacity 0.3s ease;
                }}
                .obj-card-bg-{idx}:hover::after {{
                    opacity: 0.28;
                }}
                """
        if obj_bg_css:
            st.markdown(f"<style>{obj_bg_css}</style>", unsafe_allow_html=True)

        # -- Highlighted keyword sets (based on reference image) --
        QUANT_HL  = {"Age", "Hours_per_Week"}
        CAT_HL    = {"Occupation", "Education", "Marital_Status", "Sex"}
        FIN_HL    = {"Capital Gain"}
        def _tag(name, hl_set, hl_class):
            cls = f"anatomy-tag {hl_class}" if name.strip() in hl_set else "anatomy-tag"
            return f'<span class="{cls}">{name.strip()}</span>'
        # -- Tag helpers --
        quant_tags    = "".join(_tag(k, QUANT_HL, "anatomy-tag-orange") for k in t('overview_anatomy_quant_tags').split(","))
        cat_prof_tags = "".join(_tag(k, CAT_HL,   "anatomy-tag-orange") for k in t('overview_anatomy_cat_prof_tags').split(","))
        cat_demo_tags = "".join(_tag(k, CAT_HL,   "anatomy-tag-orange") for k in t('overview_anatomy_cat_demo_tags').split(","))
        cat_fam_tags  = "".join(_tag(k, CAT_HL,   "anatomy-tag-orange") for k in t('overview_anatomy_cat_fam_tags').split(","))
        fin_tags      = "".join(_tag(k, FIN_HL,   "anatomy-tag-orange") for k in t('overview_anatomy_fin_tags').split(","))
        target_tags   = "".join(f'<span class="anatomy-tag anatomy-tag-red">{v.strip()}</span>' for v in t('overview_anatomy_target_value').split(","))
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
<div class="journey-grid">
    <div class="journey-card journey-card-1">
        <div class="journey-accent accent-1"></div>
        <div class="journey-step step-1"><span>1</span></div>
        <div class="journey-label label-1">STEP 1</div>
        <div class="journey-title">{t('overview_journey_audit_title')}</div>
        <div class="journey-desc">{t('overview_journey_audit_desc')}</div>
    </div>
    <div class="journey-arrow">
        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="rgba(255,255,255,0.15)" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="9 18 15 12 9 6"></polyline></svg>
    </div>
    <div class="journey-card journey-card-2">
        <div class="journey-accent accent-2"></div>
        <div class="journey-step step-2"><span>2</span></div>
        <div class="journey-label label-2">STEP 2</div>
        <div class="journey-title">{t('overview_journey_preprocess_title')}</div>
        <div class="journey-desc">{t('overview_journey_preprocess_desc')}</div>
    </div>
    <div class="journey-arrow">
        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="rgba(255,255,255,0.15)" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="9 18 15 12 9 6"></polyline></svg>
    </div>
    <div class="journey-card journey-card-3">
        <div class="journey-accent accent-3"></div>
        <div class="journey-step step-3"><span>3</span></div>
        <div class="journey-label label-3">STEP 3</div>
        <div class="journey-title">{t('overview_journey_eda_title')}</div>
        <div class="journey-desc">{t('overview_journey_eda_desc')}</div>
    </div>
</div>
<div class="two-columns">
    <!-- Left Column: Research Objectives -->
    <div class="column-wrapper">
        <div class="header-row">
            <div class="col-title">{t('overview_col_objectives')}</div>
        </div>
        <div class="column-boxes">
            <div class="obj-card-premium obj-card-bg-1">
                <div class="obj-icon-wrap">{icon_bar_chart}</div>
                <div class="obj-content">
                    <div class="obj-title">{t('overview_obj_grp1_1_title')}</div>
                    <div class="obj-desc">{t('overview_obj_grp1_1_desc')}</div>
                </div>
            </div>
            <div class="obj-card-premium obj-card-bg-2">
                <div class="obj-icon-wrap">{icon_zap}</div>
                <div class="obj-content">
                    <div class="obj-title">{t('overview_obj_grp1_2_title')}</div>
                    <div class="obj-desc">{t('overview_obj_grp1_2_desc')}</div>
                </div>
            </div>
            <div class="obj-card-premium obj-card-bg-3">
                <div class="obj-icon-wrap">{icon_clock}</div>
                <div class="obj-content">
                    <div class="obj-title">{t('overview_obj_grp1_3_title')}</div>
                    <div class="obj-desc">{t('overview_obj_grp1_3_desc')}</div>
                </div>
            </div>
            <div class="obj-card-premium obj-card-bg-4">
                <div class="obj-icon-wrap">{icon_heart_pulse}</div>
                <div class="obj-content">
                    <div class="obj-title">{t('overview_obj_grp1_4_title')}</div>
                    <div class="obj-desc">{t('overview_obj_grp1_4_desc')}</div>
                </div>
            </div>
            <div class="obj-card-premium obj-card-bg-5">
                <div class="obj-icon-wrap">{icon_briefcase}</div>
                <div class="obj-content">
                    <div class="obj-title">{t('overview_obj_grp1_5_title')}</div>
                    <div class="obj-desc">{t('overview_obj_grp1_5_desc')}</div>
                </div>
            </div>
        </div>
    </div>
    <!-- Right Column: Data Anatomy -->
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
            <div class="anatomy-box anatomy-box-amber">
                <div class="anatomy-title anatomy-title-amber" style="margin-bottom: 12px;">{t('overview_anatomy_cat_title')}</div>
                <div style="margin-bottom: 12px;">
                    <div style="color: var(--text-secondary); font-size: 0.8rem; text-transform: uppercase; font-weight: 700; letter-spacing: 0.5px; margin-bottom: 6px; display: flex; align-items: center; gap: 6px;">
                        <span style="color: var(--accent-orange);">-</span> {t('overview_anatomy_cat_prof_label')}
                    </div>
                    <div class="anatomy-tags" style="padding-left: 14px;">{cat_prof_tags}</div>
                </div>
                <div style="margin-bottom: 12px;">
                    <div style="color: var(--text-secondary); font-size: 0.8rem; text-transform: uppercase; font-weight: 700; letter-spacing: 0.5px; margin-bottom: 6px; display: flex; align-items: center; gap: 6px;">
                        <span style="color: var(--accent-orange);">-</span> {t('overview_anatomy_cat_demo_label')}
                    </div>
                    <div class="anatomy-tags" style="padding-left: 14px;">{cat_demo_tags}</div>
                </div>
                <div>
                    <div style="color: var(--text-secondary); font-size: 0.8rem; text-transform: uppercase; font-weight: 700; letter-spacing: 0.5px; margin-bottom: 6px; display: flex; align-items: center; gap: 6px;">
                        <span style="color: var(--accent-orange);">-</span> {t('overview_anatomy_cat_fam_label')}
                    </div>
                    <div class="anatomy-tags" style="padding-left: 14px;">{cat_fam_tags}</div>
                </div>
            </div>
            <div class="anatomy-box anatomy-box-amber">
                <div class="anatomy-title anatomy-title-amber">{t('overview_anatomy_fin_title')}</div>
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
</div>
"""
        # Strip leading whitespace to prevent Streamlit markdown code-block rendering
        html = "\n".join(line.strip() for line in html.split("\n"))
        st.markdown(html, unsafe_allow_html=True)

    # ==============================================================================
    # AUDIT PAGE COMPONENTS
    # ==============================================================================

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
            _styled_status(get_text('no_numeric_distribution', lang), accent='#3B82F6')
            return None
        # Calculate Smart Recommendation first
        selected_col = st.selectbox(
            get_text('select_column', lang), 
            numeric_cols,
            key=f"{key_prefix}_outlier_col",
            index=0,
        )

        series = df[selected_col] if selected_col else None
        if series is not None:
            smart_eval = audit_engine.evaluate_outlier_method(series, lang)
            outlier_method = smart_eval["method"]
            _skew_val = smart_eval.get("skewness", 0.0)
            _is_zero_spread = smart_eval.get("zero_spread", False)

            # --- Smart Recommendation note ---
            method_hint_keys = {"IQR": "hint_iqr", "Z-Score": "hint_zscore", "Modified Z-Score": "hint_modified_zscore"}
            _note_html = (
                '<div style="margin:4px 0 12px 0; padding:12px 16px; background:rgba(59,130,246,0.12);'
                ' border-left:3px solid rgba(59,130,246,0.4); border-radius:0 8px 8px 0;'
                ' font-size:0.78rem; color:rgba(255,255,255,0.45); line-height:1.9;">'
                f'<b style="color:rgba(255,255,255,0.6);">'
                f'\u2728 {get_text("smart_recommendation", lang)}: '
                f'<span style="color:#F59E0B;">{outlier_method}</span></b><br>'
                f'{smart_eval["reason"]}'
                f' &nbsp;\u00b7&nbsp; <b style="color:#F59E0B;">Skewness: {_skew_val}</b><br>'
                f'<span style="color:rgba(255,255,255,0.35);">'
                f'{get_text(method_hint_keys[outlier_method], lang)}</span>'
                '</div>'
            )
            st.markdown(_note_html, unsafe_allow_html=True)

            method_map = {"IQR": "iqr", "Z-Score": "zscore", "Modified Z-Score": "modified_zscore"}
            method_key = method_map[outlier_method]
            clean_vals = series.dropna()

            if _is_zero_spread:
                # --- Zero-spread: statistical detection impossible ---
                dominant_val = clean_vals.mode().iloc[0] if not clean_vals.empty else 0
                dominant_count = int((clean_vals == dominant_val).sum())
                dominant_pct = round(dominant_count / len(clean_vals) * 100, 1) if len(clean_vals) > 0 else 0
                non_dom_count = len(clean_vals) - dominant_count
                unique_count = int(clean_vals.nunique())

                _warn_html = (
                    '<div style="margin:4px 0 12px 0; padding:12px 16px; background:rgba(245,158,11,0.10);'
                    ' border-left:3px solid rgba(245,158,11,0.5); border-radius:0 8px 8px 0;'
                    ' font-size:0.78rem; color:rgba(255,255,255,0.45); line-height:1.9;">'
                    f'<b style="color:rgba(255,255,255,0.6);">⚠ {get_text("zero_spread_title", lang)}</b><br>'
                    f'<span style="color:rgba(255,255,255,0.35);">'
                    f'{get_text("zero_spread_detail", lang)}</span><br>'
                    f'<b style="color:#F59E0B;">Dominant value:</b> '
                    f'<b style="color:rgba(255,255,255,0.7);">{dominant_val:,}</b>'
                    f' &nbsp;·&nbsp; <b style="color:#F59E0B;">{dominant_pct}%</b> of data'
                    f' ({dominant_count:,} / {len(clean_vals):,} rows)<br>'
                    f'<b style="color:#F59E0B;">Non-dominant values:</b> '
                    f'<b style="color:rgba(255,255,255,0.7);">{non_dom_count:,}</b> rows'
                    f' &nbsp;·&nbsp; <b style="color:#F59E0B;">{unique_count}</b> unique values<br>'
                    f'<span style="color:rgba(255,255,255,0.30);">'
                    f'{get_text("zero_spread_suggestion", lang)}</span>'
                    '</div>'
                )
                st.markdown(_warn_html, unsafe_allow_html=True)

                # Still render histogram for visual context (without fence lines)
                risk_df = pd.DataFrame()
                fig = visualizer.plot_outlier_distribution(
                    series, risk_df, method=method_key, lang=lang,
                    skip_fences=True,
                )
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
            else:
                # --- Normal flow: formula + detection + chart ---
                _threshold = audit_engine.default_outlier_threshold(method_key)

                if method_key == "iqr":
                    q1_val = round(float(clean_vals.quantile(0.25)), 2)
                    q3_val = round(float(clean_vals.quantile(0.75)), 2)
                    iqr_val = round(q3_val - q1_val, 2)
                    lower_val = round(q1_val - _threshold * iqr_val, 2)
                    upper_val = round(q3_val + _threshold * iqr_val, 2)
                    formula_html = (
                        f'<b style="color:rgba(255,255,255,0.6);">📐 Detection Formula — IQR Method</b><br>'
                        f'<span style="color:rgba(255,255,255,0.35);">'
                        f'Q1 = <b style="color:#F59E0B;">{q1_val:,}</b>'
                        f' &nbsp;·&nbsp; Q3 = <b style="color:#F59E0B;">{q3_val:,}</b>'
                        f' &nbsp;·&nbsp; IQR = Q3 − Q1 = <b style="color:#F59E0B;">{iqr_val:,}</b></span><br>'
                        f'<span style="color:rgba(255,255,255,0.35);">'
                        f'Lower = Q1 − {_threshold}×IQR = <b style="color:#F59E0B;">{lower_val:,}</b>'
                        f' &nbsp;·&nbsp; Upper = Q3 + {_threshold}×IQR = <b style="color:#F59E0B;">{upper_val:,}</b></span><br>'
                        f'<span style="color:rgba(255,255,255,0.30);">Outlier if value &lt; {lower_val:,} or value &gt; {upper_val:,}</span>'
                    )
                elif method_key == "zscore":
                    mean_val = round(float(clean_vals.mean()), 2)
                    std_val = round(float(clean_vals.std(ddof=1)), 2)
                    formula_html = (
                        f'<b style="color:rgba(255,255,255,0.6);">📐 Detection Formula — Z-Score Method</b><br>'
                        f'<span style="color:rgba(255,255,255,0.35);">'
                        f'Mean (μ) = <b style="color:#F59E0B;">{mean_val:,}</b>'
                        f' &nbsp;·&nbsp; Std (σ) = <b style="color:#F59E0B;">{std_val:,}</b>'
                        f' &nbsp;·&nbsp; Threshold = <b style="color:#F59E0B;">{_threshold}</b></span><br>'
                        f'<span style="color:rgba(255,255,255,0.35);">'
                        f'Z = |x − μ| / σ</span><br>'
                        f'<span style="color:rgba(255,255,255,0.30);">Outlier if |Z| &gt; {_threshold} '
                        f'(i.e. value &lt; {round(mean_val - _threshold * std_val, 2):,} '
                        f'or value &gt; {round(mean_val + _threshold * std_val, 2):,})</span>'
                    )
                else:  # modified_zscore
                    median_val = round(float(clean_vals.median()), 2)
                    mad_val = round(float((clean_vals - clean_vals.median()).abs().median()), 2)
                    formula_html = (
                        f'<b style="color:rgba(255,255,255,0.6);">📐 Detection Formula — Modified Z-Score Method</b><br>'
                        f'<span style="color:rgba(255,255,255,0.35);">'
                        f'Median = <b style="color:#F59E0B;">{median_val:,}</b>'
                        f' &nbsp;·&nbsp; MAD = <b style="color:#F59E0B;">{mad_val:,}</b>'
                        f' &nbsp;·&nbsp; Threshold = <b style="color:#F59E0B;">{_threshold}</b></span><br>'
                        f'<span style="color:rgba(255,255,255,0.35);">'
                        f'M = 0.6745 × |x − Median| / MAD</span><br>'
                        f'<span style="color:rgba(255,255,255,0.30);">Outlier if |M| &gt; {_threshold}</span>'
                    )

                st.markdown(
                    f'<div style="margin:4px 0 12px 0; padding:12px 16px; background:rgba(59,130,246,0.12);'
                    f' border-left:3px solid rgba(59,130,246,0.4); border-radius:0 8px 8px 0;'
                    f' font-size:0.78rem; color:rgba(255,255,255,0.45); line-height:1.9;">'
                    f'{formula_html}</div>',
                    unsafe_allow_html=True,
                )

                risk_df, total_outliers = audit_engine.get_risk_records(df, selected_col, method=method_key)
                # Plot Distribution + Bell Curve Overlay + Boundaries
                fig = visualizer.plot_outlier_distribution(series, risk_df, method=method_key, lang=lang)
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
                    _styled_status(get_text('no_outliers_detected', lang, col=selected_col, method=outlier_method), accent='#3B82F6')
        else:
            _styled_status("Select a column to evaluate its distribution and detect outliers.", accent='#3B82F6')

        return risk_df if 'risk_df' in locals() else pd.DataFrame()

    # ==============================================================================
    # PREPROCESSING COMPONENTS
    # ==============================================================================

    @staticmethod
    def pipeline_done_banner(res_file, rows_before, rows_after, dupes_dropped, stats=None):
        """Renders the post-completion banner with metrics, comparison table, and save info.

        The heatmaps are rendered separately by the caller via Plotly
        (``stats['corr_before']`` / ``stats['corr_after']``).
        """
        if stats is None:
            stats = {}
        OK   = STATUS_COLORS["success"]["hex"]
        WARN = STATUS_COLORS["warning"]["hex"]
        rgb_ok   = _pp_hex(OK)

        rows_removed = rows_before - rows_after

        # ── Comparison table rows ─────────────────────────────────────────
        comparison = stats.get("comparison", [])
        table_rows_html = ""
        for metric_name, before_val, after_val in comparison:
            # Color: green if resolved (after == 0), else amber
            if after_val == 0:
                after_color = OK
                after_display = f'<span style="color:{OK};font-weight:700;">✓ 0</span>'
            else:
                after_color = WARN
                after_display = f'<span style="color:{WARN};font-weight:700;">{after_val:,}</span>'

            table_rows_html += (
                '<tr style="border-bottom:1px solid rgba(255,255,255,0.05);">'
                '<td style="padding:10px 14px;font-size:0.82rem;color:rgba(255,255,255,0.7);'
                f'font-weight:600;">{metric_name}</td>'
                '<td style="padding:10px 14px;text-align:center;font-size:0.88rem;'
                f'font-weight:700;color:{WARN};">{before_val:,}</td>'
                f'<td style="padding:10px 14px;text-align:center;font-size:0.88rem;">'
                f'{after_display}</td>'
                '</tr>'
            )

        st.markdown(
            '<div class="pp-done-banner" style="'
            'background:linear-gradient(135deg,rgba(166,206,57,0.09) 0%,rgba(91,134,229,0.06) 100%);'
            f'border:1px solid rgba({rgb_ok},0.35);border-radius:14px;'
            'padding:22px 26px;margin-bottom:20px;">'

            # ── Header ────────────────────────────────────────────────────
            f'<div style="color:{OK};font-size:1.05rem;font-weight:800;margin-bottom:16px;'
            'display:flex;align-items:center;gap:8px;letter-spacing:-0.3px;">'
            f'<span style="display:inline-flex;align-items:center;">{get_icon("check_circle",22,OK)}</span>'
            'Preprocessing Complete!'
            f'<span style="background:rgba({rgb_ok},0.12);border-radius:12px;font-size:0.6rem;'
            f'padding:2px 10px;font-weight:600;color:{OK};letter-spacing:0.5px;'
            'margin-left:4px;">9 / 9 STEPS</span>'
            '</div>'

            # ── Metric cards row (3 cards) ────────────────────────────────
            '<div style="display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin-bottom:18px;">'

            # Card 1: Rows Before
            '<div style="background:rgba(255,255,255,0.04);'
            'border:1px solid rgba(255,255,255,0.08);border-radius:10px;padding:14px;text-align:center;">'
            '<div style="color:var(--text-muted);font-size:0.62rem;text-transform:uppercase;'
            'letter-spacing:1px;margin-bottom:4px;">Rows Before</div>'
            f'<div style="color:white;font-size:1.3rem;font-weight:800;">{rows_before:,}</div>'
            '</div>'

            # Card 2: Rows After
            '<div style="background:rgba(255,255,255,0.04);'
            f'border:1px solid rgba({rgb_ok},0.25);border-radius:10px;padding:14px;text-align:center;">'
            '<div style="color:var(--text-muted);font-size:0.62rem;text-transform:uppercase;'
            'letter-spacing:1px;margin-bottom:4px;">Rows After</div>'
            f'<div style="color:{OK};font-size:1.3rem;font-weight:800;">{rows_after:,}</div>'
            '</div>'

            # Card 3: Rows Cleaned
            '<div style="background:rgba(255,255,255,0.04);'
            f'border:1px solid rgba({_pp_hex(WARN)},0.2);border-radius:10px;padding:14px;text-align:center;">'
            '<div style="color:var(--text-muted);font-size:0.62rem;text-transform:uppercase;'
            'letter-spacing:1px;margin-bottom:4px;">Rows Cleaned</div>'
            f'<div style="color:{WARN};font-size:1.3rem;font-weight:800;">{rows_removed:,}</div>'
            '</div>'
            '</div>'

            # ── Comparison Table ──────────────────────────────────────────
            '<div style="border-top:1px solid rgba(255,255,255,0.06);padding-top:16px;margin-bottom:16px;">'
            '<div style="font-size:0.75rem;font-weight:700;color:rgba(255,255,255,0.5);'
            'text-transform:uppercase;letter-spacing:0.8px;margin-bottom:12px;">'
            'Data Quality — Before vs After</div>'
            '<table style="width:100%;border-collapse:collapse;">'
            '<thead><tr style="border-bottom:1px solid rgba(255,255,255,0.1);">'
            '<th style="padding:8px 14px;text-align:left;font-size:0.7rem;'
            'text-transform:uppercase;letter-spacing:0.8px;color:rgba(255,255,255,0.4);'
            'font-weight:600;">Metric</th>'
            '<th style="padding:8px 14px;text-align:center;font-size:0.7rem;'
            'text-transform:uppercase;letter-spacing:0.8px;color:rgba(255,255,255,0.4);'
            'font-weight:600;">Before</th>'
            '<th style="padding:8px 14px;text-align:center;font-size:0.7rem;'
            'text-transform:uppercase;letter-spacing:0.8px;color:rgba(255,255,255,0.4);'
            'font-weight:600;">After</th>'
            '</tr></thead>'
            f'<tbody>{table_rows_html}</tbody>'
            '</table></div>'

            # ── Save info footer ──────────────────────────────────────────
            '<div style="color:var(--text-secondary);font-size:0.83rem;'
            'border-top:1px solid rgba(255,255,255,0.07);padding-top:12px;'
            'line-height:1.8;">'
            f'{get_icon("download", 14, "var(--text-muted)")} '
            f'<strong style="color:white;">Data Cleaned</strong> → '
            f'<span style="color:rgba(255,255,255,0.7);">{res_file}</span><br>'
            f'{get_icon("download", 14, "var(--text-muted)")} '
            f'<strong style="color:white;">Feature Encoded</strong> → '
            f'<span style="color:rgba(255,255,255,0.7);">'
            f'{stats.get("encoded_filename", "")}</span><br>'
            '<span style="font-size:0.75rem;color:rgba(255,255,255,0.35);">'
            'Workspace switched to cleaned file automatically.</span>'
            '</div>'
            '</div>',
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
        Displays Step 1 (noise value replacement) and Step 2 (text formatting).
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
        # ── Step 1: Noise values ───────────────────────────────────────
        st.markdown(_step_hdr(1, "Noise Value Replacement", WARN, "trash"), unsafe_allow_html=True)
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
                f"Detected {_key(f'{total:,} noise cells')} across "
                f"{_key(str(len(noise_rows)))} columns — they will be replaced with "
                f"{_key('NaN')} so that they are handled as missing values in Step 4.",
                unsafe_allow_html=True,
            )
            st.dataframe(pd.DataFrame(noise_rows), use_container_width=True, hide_index=True,
                         column_config={"Affected Cells": st.column_config.NumberColumn(format="%d")})
        else:
            _styled_status("No noise or placeholder values detected.")
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
            _styled_status("All text fields are clean \u2014 no formatting issues.")
        st.markdown(_card(
            f"<span style='display:inline-flex;align-items:center;vertical-align:top;margin-right:6px;'>"
            f"{get_icon('zap', 16, INFO)}</span> {_key('Action:')} Noise values → replaced with NaN"
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
                        from modules.core.audit_engine import compute_skewness
                        skew_display = compute_skewness(df[mc])
                    from modules.core.audit_engine import recommend_fill_strategy
                    strategy = recommend_fill_strategy(df[mc]).capitalize()
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
                _styled_status("No missing values in the dataset.")
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
                _styled_status("No duplicate rows found.")
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
            compute_preview_row_fn: ``(df, col, safe_zones) → dict | None``.
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
            _styled_status("No numeric columns available for outlier treatment.", accent='#3B82F6')
            return
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
            row = compute_preview_row_fn(df, col, safe_zones)
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
            _styled_status("No numeric columns had enough data for outlier analysis.", accent='#3B82F6')

    # --------------------------------------------------------------------------
    # PREPROCESSING — PIPELINE SIDEBAR & DETAIL PANEL
    # --------------------------------------------------------------------------

    @staticmethod
    def render_pipeline_sidebar(active_step: int) -> int:
        """Render the vertical pipeline steps using ``st.radio`` + CSS styling.

        Imports ``PIPELINE_STEP_DEFS`` from ``PreprocessingEngine`` (Core layer)
        to keep pipeline configuration centralized.

        Returns:
            The 1-based step number currently selected.
        """
        from modules.core.preprocessing_engine import (
            PreprocessingEngine,
            BINNING_TYPE_NUMERIC, BINNING_TYPE_CATEGORY,
        )
        step_defs = PreprocessingEngine.PIPELINE_STEP_DEFS

        # --- Base CSS for radio options as pipeline step cards ---
        _radio_base_css = """
        <style>
        /* --- Hide radio group label --- */
        div[data-testid="stRadio"] > label { display:none !important; }
        div[data-testid="stRadio"],
        div[data-testid="stRadio"] > div {
            width: 100% !important;
        }
        div[data-testid="stRadio"] > div[role="radiogroup"] {
            gap: 0 !important;
            display: flex !important;
            flex-direction: column !important;
            width: 100% !important;
            align-items: stretch !important;
            margin-top: -8px !important;
        }

        /* --- Each radio option → full-width card --- */
        div[data-testid="stRadio"] > div[role="radiogroup"] > label {
            background: rgba(255,255,255,0.025) !important;
            border: 1px solid rgba(255,255,255,0.07) !important;
            border-radius: 10px !important;
            padding: 12px 16px !important;
            cursor: pointer !important;
            transition: all 0.3s cubic-bezier(0.4,0,0.2,1) !important;
            margin: 0 !important;
            width: 100% !important;
            min-width: 100% !important;
            box-sizing: border-box !important;
            display: flex !important;
            position: relative !important;
        }

        /* --- DATA PREPROCESSING section header above step 1 --- */
        div[data-testid="stRadio"] > div[role="radiogroup"] > label:nth-child(1) {
            margin-top: 20px !important;
        }
        div[data-testid="stRadio"] > div[role="radiogroup"] > label:nth-child(1)::before {
            content: 'DATA PREPROCESSING' !important;
            display: block !important;
            position: absolute !important;
            top: -22px !important;
            left: 0 !important;
            font-size: 0.55rem !important;
            font-weight: 800 !important;
            letter-spacing: 1.5px !important;
            color: rgba(59,130,246,0.6) !important;
            white-space: nowrap !important;
        }

        /* --- Hide native radio circle --- */
        div[data-testid="stRadio"] > div[role="radiogroup"] > label > div:first-child {
            display: none !important;
        }

        /* --- Label text --- */
        div[data-testid="stRadio"] > div[role="radiogroup"] > label p {
            font-size: 0.8rem !important;
            font-weight: 600 !important;
            letter-spacing: 0.3px !important;
            color: rgba(255,255,255,0.45) !important;
        }

        /* --- Down arrow connectors between radio options (skip between 5→6) --- */
        div[data-testid="stRadio"] > div[role="radiogroup"] > label:not(:last-child):not(:nth-child(5))::after {
            content: '▼' !important;
            display: block !important;
            position: absolute !important;
            bottom: -12px !important;
            left: 50% !important;
            transform: translateX(-50%) !important;
            font-size: 0.5rem !important;
            color: rgba(255,255,255,0.12) !important;
            z-index: 2 !important;
            line-height: 1 !important;
        }

        /* --- Ensure arrows also show between steps 6→7 and 7→8 within Feature Prep --- */
        div[data-testid="stRadio"] > div[role="radiogroup"] > label:not(:last-child) {
            margin-bottom: 16px !important;
        }

        /* --- Feature Preparation section divider before step 6 --- */
        div[data-testid="stRadio"] > div[role="radiogroup"] > label:nth-child(5) {
            margin-bottom: 0 !important;
        }
        div[data-testid="stRadio"] > div[role="radiogroup"] > label:nth-child(5)::after {
            display: none !important;
        }
        div[data-testid="stRadio"] > div[role="radiogroup"] > label:nth-child(6) {
            margin-top: 32px !important;
        }
        div[data-testid="stRadio"] > div[role="radiogroup"] > label:nth-child(6)::before {
            content: 'FEATURE PREPARATION' !important;
            display: block !important;
            position: absolute !important;
            top: -24px !important;
            left: 0 !important;
            font-size: 0.55rem !important;
            font-weight: 800 !important;
            letter-spacing: 1.5px !important;
            color: rgba(127,177,53,0.6) !important;
            white-space: nowrap !important;
        }

        </style>
        """
        st.markdown(_radio_base_css, unsafe_allow_html=True)

        # --- Per-step accent colors for hover & checked states ---
        _per_step_rules = []
        _nth = 'div[data-testid="stRadio"] > div[role="radiogroup"] > label'
        for idx, step_def in enumerate(step_defs, start=1):
            hex_color = step_def["color"]
            r, g, b = int(hex_color[1:3], 16), int(hex_color[3:5], 16), int(hex_color[5:7], 16)
            _per_step_rules.append(f"""
            {_nth}:nth-child({idx}):hover {{
                background: rgba({r},{g},{b},0.06) !important;
                border-color: rgba({r},{g},{b},0.25) !important;
                transform: translateX(3px) !important;
                box-shadow: 0 0 12px rgba({r},{g},{b},0.08) !important;
            }}
            {_nth}:nth-child({idx}):hover p {{
                color: rgba(255,255,255,0.85) !important;
            }}
            {_nth}:nth-child({idx}):has(input:checked) {{
                background: rgba({r},{g},{b},0.10) !important;
                border: 1.5px solid rgba({r},{g},{b},0.5) !important;
                box-shadow: 0 0 20px rgba({r},{g},{b},0.18),
                            inset 0 0 20px rgba({r},{g},{b},0.04) !important;
                transform: translateX(3px) !important;
            }}
            {_nth}:nth-child({idx}):has(input:checked) p {{
                color: rgba(255,255,255,0.95) !important;
                font-weight: 700 !important;
            }}
            """)
        st.markdown(f'<style>{" ".join(_per_step_rules)}</style>', unsafe_allow_html=True)

        # Build radio options — professional numbered format
        options = [
            f"0{s['num']}  —  {s['title']}" for s in step_defs
        ]

        selected = st.radio(
            "Pipeline Steps",
            options,
            index=active_step - 1,
            label_visibility="collapsed",
            key="pp_step_radio",
        )

        # Map selection back to step number
        return options.index(selected) + 1 if selected else active_step

    @staticmethod
    def render_detail_panel(
        step: int,
        df: pd.DataFrame,
        compute_fn,
    ) -> None:
        """Render the right-panel detail view for the selected pipeline step.

        Imports ``PIPELINE_STEP_DEFS`` from ``PreprocessingEngine`` (Core layer).
        Uses module-level UI helpers (``_pp_hex``, ``_pp_key``, ``_pp_card``).
        """
        from modules.core.audit_engine import (
            _compute_noise_mask, _get_cat_columns, _get_safe_zones,
            compute_skewness, recommend_fill_strategy,
            evaluate_outlier_method, default_outlier_threshold,
            _OUTLIER_METHODS,
        )
        from modules.core.preprocessing_engine import (
            PreprocessingEngine,
            ENC_LABEL, ENC_ONEHOT, ENC_DROP_REDUNDANT,
            BINNING_TYPE_NUMERIC, BINNING_TYPE_CATEGORY,
            LOG_METHOD_LOG1P, LOG_METHOD_YJ,
        )

        step_defs = PreprocessingEngine.PIPELINE_STEP_DEFS
        step_info = step_defs[step - 1]
        rgb = _pp_hex(step_info["color"])
        col_hex = step_info["color"]

        # ── Consistent section header ─────────────────────────────────────
        st.markdown(
            f'<div style="margin-bottom:24px; padding:20px 22px;'
            f' background:linear-gradient(135deg, rgba({rgb},0.08) 0%, rgba({rgb},0.02) 100%);'
            f' border:1px solid rgba({rgb},0.12); border-left:3px solid {col_hex};'
            f' border-radius:0 14px 14px 0;">'
            f'<div style="display:flex; align-items:center; gap:14px;">'
            f'<div style="width:40px; height:40px; border-radius:10px;'
            f' background:rgba({rgb},0.15); border:1px solid rgba({rgb},0.3);'
            f' display:flex; align-items:center; justify-content:center;'
            f' font-size:1.1rem; font-weight:800; color:{col_hex};'
            f' flex-shrink:0;">{step_info["num"]}</div>'
            f'<div>'
            f'<div style="font-size:0.58rem; font-weight:800; color:{col_hex};'
            f' text-transform:uppercase; letter-spacing:1.5px;'
            f' margin-bottom:2px; opacity:0.8;">Step {step_info["num"]}</div>'
            f'<div style="font-size:1.15rem; font-weight:700;'
            f' color:rgba(255,255,255,0.95); letter-spacing:-0.3px;'
            f' line-height:1.3;">{step_info["title"]}</div>'
            f'</div></div>'
            f'<p style="font-size:0.82rem; color:rgba(255,255,255,0.4);'
            f' margin:10px 0 0 54px; line-height:1.5;">{step_info["desc"]}</p>'
            f'</div>',
            unsafe_allow_html=True,
        )

        # ── Helper: info box ──────────────────────────────────────────────
        def _info_box(html_content: str, accent: str = col_hex) -> None:
            a_rgb = _pp_hex(accent)
            st.markdown(
                f'<div style="margin:4px 0 12px 0; padding:12px 16px;'
                f' background:rgba({a_rgb},0.08);'
                f' border-left:3px solid rgba({a_rgb},0.4);'
                f' border-radius:0 8px 8px 0;'
                f' font-size:0.78rem; color:rgba(255,255,255,0.45); line-height:1.9;">'
                f'{html_content}</div>',
                unsafe_allow_html=True,
            )

        # ── Helper: metric row ────────────────────────────────────────────
        def _metric_row(items: list) -> None:
            """Render a horizontal row of metric cards. items = [(label, value, color), ...]."""
            cards = ""
            for label, value, color in items:
                c_rgb = _pp_hex(color)
                cards += (
                    f'<div class="pp-metric-card" style="flex:1;'
                    f' background:rgba({c_rgb},0.04);'
                    f' border:1px solid rgba({c_rgb},0.12); border-radius:12px;'
                    f' padding:18px 16px; text-align:center; position:relative;'
                    f' overflow:hidden;">'
                    f'<div style="position:absolute; top:0; left:15%; right:15%;'
                    f' height:2px; background:linear-gradient(90deg,'
                    f' transparent, rgba({c_rgb},0.5), transparent);"></div>'
                    f'<div style="font-size:1.7rem; font-weight:800; color:{color};'
                    f' line-height:1; letter-spacing:-0.5px;">{value}</div>'
                    f'<div style="font-size:0.62rem; color:rgba(255,255,255,0.4);'
                    f' text-transform:uppercase; letter-spacing:1px;'
                    f' margin-top:8px; font-weight:600;">{label}</div></div>'
                )
            st.markdown(
                f'<div style="display:flex; gap:12px; margin-bottom:18px;">{cards}</div>',
                unsafe_allow_html=True,
            )

        # ── Helper: skip / all-clear card ─────────────────────────────
        def _skip_card(text: str) -> None:
            st.markdown(
                f'<div style="margin:4px 0 8px 12px; padding:10px 14px;'
                f' background:rgba({rgb},0.03);'
                f' border-left:2px solid rgba({rgb},0.4);'
                f' border-radius:0 8px 8px 0;'
                f' display:flex; align-items:center; gap:10px;'
                f' font-size:0.8rem; color:rgba(255,255,255,0.4);">'
                f'<span style="color:rgba({rgb},0.7); font-size:0.85rem;">✓</span>'
                f'<span>{text}</span></div>',
                unsafe_allow_html=True,
            )

        # ── Step 1: Standardize & Type Cast ────────────────────────────────
        if step == 1:
            cat_cols = _get_cat_columns(df).tolist()

            # ── Text issues (whitespace + casing) ─────────────────────────
            fmt_rows = []
            total_ws = 0
            total_casing = 0
            for col in cat_cols:
                series = df[col].dropna().astype(str)
                ws_count = int((series != series.str.strip()).sum())
                lm: dict = {}
                for unique_val in series.unique():
                    lm.setdefault(unique_val.strip().lower(), []).append(unique_val)
                casing_variants = sum(1 for vs in lm.values() if len(vs) > 1)
                total_ws += ws_count
                total_casing += casing_variants
                if ws_count > 0 or casing_variants > 0:
                    parts = []
                    if ws_count:
                        parts.append(f"{ws_count} whitespace")
                    if casing_variants:
                        parts.append(f"{casing_variants} casing groups")
                    fmt_rows.append({"Column": col, "Issues": " | ".join(parts)})

            # ── Dtype conversion candidates ───────────────────────────────
            type_cast_rows = PreprocessingEngine.get_type_cast_preview(df)
            n_type_casts = len(type_cast_rows)

            # ── Sub-header helper ─────────────────────────────────────────
            def _sub_header(label: str, accent: str = col_hex) -> None:
                st.markdown(
                    f'<div style="display:flex; align-items:center; gap:10px;'
                    f' margin:20px 0 12px 0;">'
                    f'<div style="width:4px; height:18px; border-radius:2px;'
                    f' background:{accent}; flex-shrink:0;"></div>'
                    f'<span style="font-size:1rem; font-weight:700;'
                    f' color:rgba(255,255,255,0.85); letter-spacing:-0.2px;">'
                    f'{label}</span></div>',
                    unsafe_allow_html=True,
                )

            # 1a. TEXT ISSUES
            _sub_header("Text Issues")
            if fmt_rows:
                _metric_row([
                    ("Whitespace Issues", f"{total_ws:,}", col_hex),
                    ("Casing Variants", f"{total_casing:,}", col_hex),
                    ("Affected Columns", str(len(fmt_rows)), col_hex),
                ])
                st.dataframe(pd.DataFrame(fmt_rows), use_container_width=True, hide_index=True)
                _info_box(
                    '<b style="color:rgba(255,255,255,0.6);">Action:</b> '
                    '<b style="color:#F59E0B;">Trim</b> leading/trailing whitespace, then '
                    '<b style="color:#F59E0B;">normalize casing</b> by choosing the most frequent '
                    'variant as the canonical form.'
                )
            else:
                _skip_card("No whitespace or casing issues — all text fields are clean.")

            # 1b. DTYPE ISSUES
            _sub_header("Dtype Issues")
            if type_cast_rows:
                _metric_row([
                    ("Type Conversions", str(n_type_casts), col_hex),
                    ("Total Convertible", f"{sum(r['Convertible'] for r in type_cast_rows):,}", col_hex),
                ])
                st.dataframe(
                    pd.DataFrame(type_cast_rows),
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Convertible": st.column_config.NumberColumn(format="%d"),
                        "% Convertible": st.column_config.ProgressColumn(
                            format="%.1f%%", min_value=0, max_value=100,
                        ),
                    },
                )
                _info_box(
                    '<b style="color:rgba(255,255,255,0.6);">Action:</b> '
                    'Strictly <b style="color:#F59E0B;">enforce data types</b> based on the '
                    '<b style="color:#F59E0B;">Admin Data Schema</b>, falling back to auto-conversion '
                    'only for undefined columns.'
                )
            else:
                _skip_card("All column dtypes are correct — no conversion needed.")

        # ── Step 2: Noise Cleaning ─────────────────────────────────────────
        elif step == 2:
            cat_cols = _get_cat_columns(df).tolist()
            noise_rows = []
            for col in cat_cols:
                series = df[col].dropna().astype(str)
                mask = _compute_noise_mask(series)
                cnt = int(mask.sum())
                if cnt > 0:
                    noise_rows.append({
                        "Column": col,
                        "Affected Cells": cnt,
                        "Examples": ", ".join(f"'{v}'" for v in series[mask].unique()[:4]),
                    })
            if noise_rows:
                total_cells = sum(r["Affected Cells"] for r in noise_rows)
                total_all_cells = len(df) * len(cat_cols)
                noise_pct = total_cells / total_all_cells * 100 if total_all_cells else 0
                _metric_row([
                    ("Noise Cells", f"{total_cells:,}", col_hex),
                    ("% Noise", f"{noise_pct:.2f}%", col_hex),
                    ("Affected Columns", str(len(noise_rows)), col_hex),
                ])
                st.dataframe(
                    pd.DataFrame(noise_rows),
                    use_container_width=True,
                    hide_index=True,
                    column_config={"Affected Cells": st.column_config.NumberColumn(format="%d")},
                )
                _info_box(
                    '<b style="color:rgba(255,255,255,0.6);">Action:</b> '
                    'All noise values (<b style="color:#F59E0B;">?, N/A, --, etc.</b>) '
                    'will be replaced with <b style="color:#F59E0B;">NaN</b> so they can be '
                    'properly handled in <b style="color:rgba(255,255,255,0.6);">Step 4 — Imputing Missing Values</b>.'
                )
            else:
                _skip_card("No noise or placeholder values detected — this step will be skipped.")

        # ── Step 3: Duplicate Removal ──────────────────────────────────────
        elif step == 3:
            dupes = int(df.duplicated().sum())
            total = len(df)
            if dupes > 0:
                pct = dupes / total * 100
                _metric_row([
                    ("Duplicate Rows", f"{dupes:,}", col_hex),
                    ("Percentage", f"{pct:.1f}%", col_hex),
                    ("Rows After Drop", f"{total - dupes:,}", col_hex),
                ])
                _info_box(
                    f'<b style="color:rgba(255,255,255,0.6);">Action:</b> '
                    f'All <b style="color:#F59E0B;">{dupes:,} duplicate rows</b> will be dropped, '
                    f'retaining only the <b style="color:#F59E0B;">first occurrence</b> '
                    f'of each unique record.'
                )
            else:
                _skip_card("No duplicate rows found — all records are unique. This step will be skipped.")

        # ── Step 4: Missing Value Handling ─────────────────────────────────
        elif step == 4:
            missing_cols = df.columns[df.isnull().any()].tolist()
            if missing_cols:
                total_rows = len(df)
                total_missing = int(df[missing_cols].isnull().sum().sum())
                missing_data = []
                for mc in missing_cols:
                    cnt = int(df[mc].isnull().sum())
                    dtype = "Numeric" if pd.api.types.is_numeric_dtype(df[mc]) else "Categorical"
                    skew_display = None
                    if dtype == "Numeric":
                        skew_display = compute_skewness(df[mc])
                    strategy = recommend_fill_strategy(df[mc]).capitalize()
                    missing_data.append({
                        "Column": mc, "Type": dtype, "Missing": cnt,
                        "% Missing": cnt / total_rows * 100,
                        "Skewness": skew_display, "Strategy": strategy,
                    })
                missing_pct = total_missing / (total_rows * len(df.columns)) * 100
                _metric_row([
                    ("Total Missing", f"{total_missing:,}", col_hex),
                    ("% Missing", f"{missing_pct:.2f}%", col_hex),
                    ("Affected Columns", str(len(missing_cols)), col_hex),
                ])
                st.dataframe(
                    pd.DataFrame(missing_data).sort_values("Missing", ascending=False),
                    use_container_width=True, hide_index=True,
                    column_config={
                        "Missing": st.column_config.NumberColumn(format="%d"),
                        "% Missing": st.column_config.ProgressColumn(format="%.1f%%", min_value=0, max_value=100),
                        "Skewness": st.column_config.NumberColumn("Skewness", format="%.3f"),
                        "Strategy": st.column_config.TextColumn("Fill Strategy"),
                    },
                )
                _info_box(
                    '<b style="color:rgba(255,255,255,0.6);">Strategy:</b> '
                    '<b style="color:#F59E0B;">Numeric</b> → Mean (symmetric) / Median (skewed) &nbsp;·&nbsp; '
                    '<b style="color:#F59E0B;">Categorical</b> → Mode (most frequent value)'
                )
            else:
                _skip_card("No missing values in the dataset — this step will be skipped.")

        # ── Step 5: Outlier Treatment ─────────────────────────────────────
        elif step == 5:
            numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
            if not numeric_cols:
                _skip_card("No numeric columns available for outlier treatment.")
            else:
                safe_zones = _get_safe_zones()
                method_info = PreprocessingEngine.METHOD_INFO

                rows = []
                zero_spread_cols = []
                for col in numeric_cols:
                    series = df[col].dropna()
                    if len(series) < 3:
                        continue
                    rec = evaluate_outlier_method(series)
                    skew_val = rec["skewness"]
                    method_name = rec["method"]
                    is_zero_spread = rec.get("zero_spread", False)
                    detect_key = method_info.get(
                        method_name, ("iqr", "iqr_capping", "IQR Cap")
                    )[0]
                    threshold = default_outlier_threshold(detect_key)
                    detect_fn = _OUTLIER_METHODS.get(detect_key, _OUTLIER_METHODS["iqr"])
                    stat_mask, _ = detect_fn(series.values, threshold)
                    raw_count = int(stat_mask.sum())

                    real_row = compute_fn(df, col, safe_zones)
                    action = real_row["Action"] if real_row else "No Outlier Treatment"
                    is_int = pd.api.types.is_integer_dtype(series) or (len(series) > 0 and (series % 1 == 0).all())

                    def _fmt(val):
                        if pd.isna(val) or val == "N/A":
                            return "N/A"
                        # Always show Ints as plain integers, but Floats with 2 decimals
                        if is_int and float(val).is_integer():
                            return f"{int(val):,}"
                        return f"{float(val):,.2f}"

                    col_min = _fmt(series.min()) if len(series) > 0 else "0"
                    col_max = _fmt(series.max()) if len(series) > 0 else "0"
                    total_rows = len(series)
                    pct_outlier = round(raw_count / total_rows * 100, 2) if total_rows > 0 else 0.0

                    # Compute Lower / Upper Limit
                    if is_zero_spread:
                        lower_limit = "N/A"
                        upper_limit = "N/A"
                        zero_spread_cols.append(col)
                    else:
                        if detect_key == "iqr":
                            q1 = float(series.quantile(0.25))
                            q3 = float(series.quantile(0.75))
                            iqr = q3 - q1
                            lower_limit = _fmt(q1 - threshold * iqr)
                            upper_limit = _fmt(q3 + threshold * iqr)
                        elif detect_key == "zscore":
                            mean_v = float(series.mean())
                            std_v = float(series.std(ddof=1))
                            lower_limit = _fmt(mean_v - threshold * std_v)
                            upper_limit = _fmt(mean_v + threshold * std_v)
                        else:  # modified_zscore
                            median_v = float(series.median())
                            mad_v = float((series - series.median()).abs().median())
                            if mad_v > 0:
                                lower_limit = _fmt(median_v - threshold * mad_v / 0.6745)
                                upper_limit = _fmt(median_v + threshold * mad_v / 0.6745)
                            else:
                                lower_limit = "N/A"
                                upper_limit = "N/A"

                    rows.append({
                        "Column": col,
                        "Min": col_min,
                        "Max": col_max,
                        "Skewness": skew_val,
                        "Method": method_name,
                        "Lower Limit": lower_limit,
                        "Upper Limit": upper_limit,
                        "Outliers Detected": raw_count,
                        "% Outlier": pct_outlier,
                        "Action": action,
                    })

                total_outliers = sum(r["Outliers Detected"] for r in rows)
                affected_cols = sum(1 for r in rows if r["Outliers Detected"] > 0)

                if total_outliers > 0:
                    _metric_row([
                        ("Total Outliers", f"{total_outliers:,}", col_hex),
                        ("Affected Columns", str(affected_cols), col_hex),
                        ("Numeric Columns", str(len(numeric_cols)), col_hex),
                    ])

                    if rows:
                        outlier_df = pd.DataFrame(rows)

                        # Highlight rows where detection was not possible (zero-spread)
                        def _highlight_zero_spread(row):
                            if row["Lower Limit"] == "N/A":
                                return [
                                    "background-color: rgba(245,158,11,0.18); "
                                    "color: rgba(245,158,11,0.7);"
                                ] * len(row)
                            return [""] * len(row)

                        styled = outlier_df.style.apply(_highlight_zero_spread, axis=1)
                        st.dataframe(
                            styled,
                            use_container_width=True,
                            hide_index=True,
                            column_config={
                                "Skewness": st.column_config.NumberColumn(format="%.3f"),
                                "Outliers Detected": st.column_config.NumberColumn(format="%d"),
                                "% Outlier": st.column_config.NumberColumn(format="%.2f%%"),
                            },
                        )

                    # Zero-spread note
                    if zero_spread_cols:
                        col_list = ", ".join(f'<b style="color:#F59E0B;">{c}</b>' for c in zero_spread_cols)
                        st.markdown(
                            '<div style="margin:4px 0 12px 0; padding:12px 16px; background:rgba(245,158,11,0.10);'
                            ' border-left:3px solid rgba(245,158,11,0.5); border-radius:0 8px 8px 0;'
                            ' font-size:0.78rem; color:rgba(255,255,255,0.45); line-height:1.9;">'
                            '<b style="color:rgba(255,255,255,0.6);">⚠ Zero-Spread Columns</b><br>'
                            f'{col_list} — '
                            '<span style="color:rgba(255,255,255,0.35);">'
                            'Both IQR and MAD equal zero because ≥50% of values are identical. '
                            'Statistical outlier detection cannot compute meaningful fences. '
                            'Consider using Admin Safe Zones for these columns.</span>'
                            '</div>',
                            unsafe_allow_html=True,
                        )

                    _info_box(
                        '<b style="color:rgba(255,255,255,0.6);">Methodology:</b> '
                        'Skewness determines detection method \u2014 '
                        '<b style="color:#F59E0B;">Z-Score</b> (|skew| < 0.5) \u00b7 '
                        '<b style="color:#F59E0B;">IQR</b> (0.5\u20131.0) \u00b7 '
                        '<b style="color:#F59E0B;">Modified Z-Score</b> (> 1.0).<br>'
                        'Detected outliers are capped to the method\u2019s statistical fence.'
                    )
                else:
                    _skip_card("No outliers detected across all numeric columns — this step will be skipped.")

        # ── Step 6: Log Transformation ────────────────────────────────────
        elif step == 6:
            candidates = PreprocessingEngine.get_log_transform_candidates(df)

            if candidates:
                n_log1p = sum(1 for c in candidates if c["Method"] == LOG_METHOD_LOG1P)
                n_yj = sum(1 for c in candidates if c["Method"] == LOG_METHOD_YJ)

                _metric_row([
                    ("Skewed Columns", str(len(candidates)), col_hex),
                    ("Log1p Applied", str(n_log1p), col_hex),
                    ("Yeo-Johnson Applied", str(n_yj), col_hex),
                ])

                st.dataframe(
                    pd.DataFrame(candidates),
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Skewness": st.column_config.NumberColumn(format="%.3f"),
                        "Min": st.column_config.NumberColumn(format="%.2f"),
                        "Max": st.column_config.NumberColumn(format="%.2f"),
                    },
                )

                _info_box(
                    '<b style="color:rgba(255,255,255,0.6);">Methodology:</b> '
                    'Columns with <b style="color:#F59E0B;">|skewness| > 1.0</b> are transformed.<br>'
                    '• <b style="color:#F59E0B;">log1p</b>: log(1+x) — used when min ≥ 0 (safe for zeros)<br>'
                    '• <b style="color:#F59E0B;">Yeo-Johnson</b>: power transform — used when min &lt; 0 (handles negatives)<br>'
                    '• <b style="color:#F59E0B;">Low-Variance columns</b> (≥ 80% identical values) are '
                    '<b>automatically excluded</b> — log transform is ineffective on zero-spike distributions '
                    'and would break downstream Binning boundaries.'
                )
            else:
                _skip_card("No highly-skewed columns detected — this step will be skipped.")

        # ── Step 7: Binning & Mapping ──────────────────────────────────────
        elif step == 7:
            from modules.utils.db_config_manager import get_rule
            binning_config = get_rule("binning_config") or {}
            preview = PreprocessingEngine.get_binning_preview(df, binning_config)

            if preview:
                n_bin = sum(1 for p in preview if p["Type"] == BINNING_TYPE_NUMERIC)
                n_map = sum(1 for p in preview if p["Type"] == BINNING_TYPE_CATEGORY)

                _metric_row([
                    ("Total Rules", str(len(preview)), col_hex),
                    ("Numeric Binning", str(n_bin), col_hex),
                    ("Category Mapping", str(n_map), col_hex),
                ])

                st.dataframe(
                    pd.DataFrame(preview),
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Unique Before": st.column_config.NumberColumn(format="%d"),
                    },
                )

                _info_box(
                    '<b style="color:rgba(255,255,255,0.6);">Methodology:</b><br>'
                    '• <b style="color:#F59E0B;">Numeric Binning</b>: discretize continuous values into labeled ranges (e.g. age → age groups)<br>'
                    '• <b style="color:#F59E0B;">Category Mapping</b>: group fine-grained categories into broader, analysis-ready groups'
                )
            else:
                _skip_card("No binning or mapping rules configured — this step will be skipped.")

        # ── Step 8: Feature Encoding ───────────────────────────────────────
        elif step == 8:
            from modules.utils.db_config_manager import get_rule as _get_rule_enc
            binning_cfg = _get_rule_enc("binning_config") or {}
            candidates = PreprocessingEngine.get_encoding_preview(
                df, binning_config=binning_cfg,
            )

            if candidates:
                n_label = sum(1 for c in candidates if c["Encoding"] == ENC_LABEL)
                n_onehot = sum(1 for c in candidates if c["Encoding"] == ENC_ONEHOT)
                n_drop = sum(1 for c in candidates if c["Encoding"] == ENC_DROP_REDUNDANT)

                _metric_row([
                    ("Total Columns", str(len(candidates)), col_hex),
                    (ENC_LABEL, str(n_label), col_hex),
                    (ENC_ONEHOT, str(n_onehot), col_hex),
                    (ENC_DROP_REDUNDANT, str(n_drop), col_hex),
                ])

                enc_df = pd.DataFrame(candidates)

                def _highlight_dropped(row):
                    if row["Encoding"] == ENC_DROP_REDUNDANT:
                        return [
                            "background-color: rgba(245,158,11,0.18); "
                            "color: rgba(245,158,11,0.7);"
                        ] * len(row)
                    return [""] * len(row)

                styled_enc = enc_df.style.apply(_highlight_dropped, axis=1)
                st.dataframe(
                    styled_enc,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Unique": st.column_config.NumberColumn(format="%d"),
                    },
                )

                if n_drop > 0:
                    drop_names = ", ".join(
                        f"<b style='color:#F59E0B;'>{c['Column']}</b>"
                        for c in candidates if c["Encoding"] == ENC_DROP_REDUNDANT
                    )
                    _info_box(
                        f'<b style="color:rgba(255,255,255,0.6);">Redundancy:</b> '
                        f'{drop_names} will be dropped — numeric equivalent already exists.'
                    )

                _info_box(
                    '<b style="color:rgba(255,255,255,0.6);">Methodology:</b><br>'
                    '• <b style="color:#F59E0B;">Label Encoding</b>: ordinal/binary columns → integer codes (preserves natural order)<br>'
                    '• <b style="color:#F59E0B;">One-Hot Encoding</b>: nominal columns → binary indicators (<code>drop_first=True</code> to avoid multicollinearity)<br>'
                    '• <b style="color:#F59E0B;">Drop (Redundant)</b>: columns with a numeric counterpart already in dataset'
                )
            else:
                _skip_card("No categorical columns found — this step will be skipped.")

        # ── Step 9: Feature Scaling ────────────────────────────────────────
        elif step == 9:
            _info_box(
                '<b style="color:rgba(255,255,255,0.6);">'
                '\u2139 About This Step</b><br>'
                'Scaler selection is computed on the '
                '<b style="color:#F59E0B;">post-encoding DataFrame</b> '
                '(after Steps 1\u20138) during pipeline execution. '
                'Column distributions change significantly through '
                'outlier treatment, log transformation, and encoding '
                '\u2014 so preview on raw or cleaned data would be inaccurate.'
            )

            _info_box(
                '<b style="color:rgba(255,255,255,0.6);">Methodology:</b><br>'
                '\u2022 <b style="color:#F59E0B;">StandardScaler</b>: '
                '(x \u2212 \u03bc) / \u03c3 \u2014 selected when |skewness| &lt; 0.5 '
                '(near-normal distribution)<br>'
                '\u2022 <b style="color:#F59E0B;">RobustScaler</b>: '
                '(x \u2212 median) / IQR \u2014 selected when |skewness| \u2265 0.5 '
                '(robust to skew &amp; remaining outliers from Safe Zones)<br>'
                '\u2022 <b style="color:rgba(255,255,255,0.5);">Binary columns</b> '
                '(\u2264 2 unique values) are automatically skipped.'
            )

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