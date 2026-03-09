"""
data_audit.py — Data Integrity Audit Page
Enterprise Dashboard with neon-themed charts and animated UI.
"""

import streamlit as st
import pandas as pd
from modules.core import data_engine, audit_engine
from modules.ui import page_header, workspace_status, metric_card, active_file_scan_progress_bar, section_divider
from modules.ui.components import UiComponents
from modules.ui.visualizer import plot_field_integrity, plot_quality_matrix, plot_correlation_matrix, plot_issue_composition
from modules.utils.localization import get_text
from modules.utils.helpers import _ensure_workspace_active, save_temp_csv
from modules.utils.theme_manager import STATUS_COLORS
from modules.utils.session_debug import set_state, get_state
from modules.core.report_engine import generate_audit_report
import json


def _render_executive_metrics(audit: dict, lang: str):
    """SECTION 1 — EXECUTIVE HEALTH METRICS"""
    st.markdown("<div class='audit-section'>", unsafe_allow_html=True)
    st.markdown(f"<div class='section-header'><h3>{get_text('health_score', lang)}</h3></div>", unsafe_allow_html=True)

    m1, m2, m3, m4, m5, m6 = st.columns(6)
    h = audit["health_score"]
    h_glow = "green" if h >= 85 else ("orange" if h >= 60 else "red")
    it = audit.get("inconsistency_total", 0)
    nt = audit.get("noise_total", 0)

    with m1: metric_card(get_text('health_score', lang), f"{h}%", get_text('weighted_index', lang), glow=h_glow)
    with m2: metric_card(get_text('total_records', lang), f"{audit['total_records']:,}", get_text('rows', lang), glow="blue")
    with m3: metric_card(get_text('duplicates', lang), f"{audit['duplicates']:,}", get_text('redundant_rows', lang), glow="red" if audit["duplicates"] > 0 else "green")
    with m4: metric_card(get_text('missing_cells', lang), f"{audit['missing_cells']:,}", get_text('null_values', lang), glow="orange" if audit["missing_cells"] > 0 else "green")
    with m5: metric_card(get_text('noise_values', lang), f"{nt:,}", get_text('noise_metric_subtitle', lang), glow="orange" if nt > 0 else "green")
    with m6: metric_card(get_text('audit_data_inconsistency', lang), f"{it:,}", get_text('audit_inconsistent_cells', lang), glow="orange" if it > 0 else "green")

    st.markdown("</div>", unsafe_allow_html=True)
    section_divider()


def _render_issue_composition(audit: dict, lang: str):
    """SECTION 2 — ISSUE COMPOSITION"""
    st.markdown("<div class='audit-section-delay'>", unsafe_allow_html=True)
    st.markdown(f"<div class='section-header'><h3>{get_text('issue_composition', lang)}</h3></div>", unsafe_allow_html=True)
    st.caption(get_text('issue_composition_caption', lang))

    # Build issue_composition from individual audit fields (no more duplication)
    # Duplicates are row-level → convert to cells (rows × cols) for fair comparison
    total_rows = audit.get("total_records", 0)
    total_cols = audit.get("attributes", 0)
    total_values = total_rows * total_cols
    duplicate_cells = audit.get("duplicates", 0) * total_cols

    issue_dict = {
        get_text("compose_missing", lang):        audit.get("missing_cells", 0),
        get_text("compose_duplicates", lang):      duplicate_cells,
        get_text("compose_inconsistencies", lang): audit.get("inconsistency_total", 0),
        get_text("compose_noise_values", lang):    audit.get("noise_total", 0),
    }

    if not issue_dict or total_values == 0:
        st.info(get_text('audit_no_issue_data', lang))
    else:
        # Display original row count for duplicates on the chart text
        dup_label = get_text("compose_duplicates", lang)
        display_overrides = {dup_label: audit.get("duplicates", 0)}

        # Summary row: total issues vs total cells
        total_issues = sum(issue_dict.values())
        clean_pct = round((total_values - total_issues) / total_values * 100, 1) if total_values > 0 else 100.0
        issue_pct = round(total_issues / total_values * 100, 2) if total_values > 0 else 0.0
        st.markdown(f"""
            <div style="display:flex; gap:10px; flex-wrap:wrap; margin-bottom:12px;">
                <span class="status-badge badge-blue">{get_text('audit_dataset_cells', lang, n=total_values)}</span>
                <span class="status-badge badge-red">{get_text('audit_affected_pct', lang, pct=issue_pct, n=total_issues)}</span>
                <span class="status-badge badge-green">{get_text('audit_clean_pct', lang, pct=clean_pct)}</span>
            </div>
        """, unsafe_allow_html=True)

        st.markdown("<div class='chart-glow'>", unsafe_allow_html=True)
        st.plotly_chart(
            plot_issue_composition(issue_dict, total_values, display_overrides=display_overrides),
            use_container_width=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)
    section_divider()


def _render_safe_zone_violations(audit: dict, lang: str):
    """SECTION 3 — SAFE ZONE VIOLATIONS"""
    sz_df = audit.get("safe_zone_violations", pd.DataFrame())
    sz_total = audit.get("safe_zone_total", 0)

    st.markdown(f"<div class='section-header'><h3>{get_text('safe_zone_title', lang)}</h3></div>", unsafe_allow_html=True)
    st.caption(get_text("safe_zone_caption", lang))

    if sz_total == 0:
        st.success(get_text("safe_zone_pass", lang))
    else:
        st.markdown(f"""
            <div style="display:flex; gap:10px; flex-wrap:wrap; margin-bottom:12px;">
                <span class="status-badge badge-red">{get_text('audit_violations', lang, n=sz_total)}</span>
                <span class="status-badge badge-blue">{get_text('audit_cols_affected', lang, n=len(sz_df))}</span>
            </div>
        """, unsafe_allow_html=True)
        st.dataframe(sz_df, use_container_width=True, hide_index=True)
    section_divider()


def _render_field_integrity(audit: dict, lang: str):
    """SECTION 4 — FIELD INTEGRITY + DISTRIBUTION"""
    st.markdown(f"<div class='section-header'><h3>{get_text('field_integrity', lang)}</h3></div>", unsafe_allow_html=True)
    st.caption(get_text('field_integrity_caption', lang))

    tab_integrity, tab_quality = st.tabs([
        f":material/assessment: {get_text('field_integrity', lang)}",
        f":material/grid_view: {get_text('audit_data_quality_matrix', lang)}",
    ])

    with tab_integrity:
        integrity_df = audit["field_integrity"]
        st.markdown("<div class='chart-glow'>", unsafe_allow_html=True)
        st.plotly_chart(plot_field_integrity(integrity_df), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        with st.expander(f":material/table_view: {get_text('field_integrity', lang)}"):
            st.dataframe(
                integrity_df.drop(columns=["Suggestion"], errors="ignore"),
                use_container_width=True, hide_index=True,
                column_config={
                    "Fill Rate (%)": st.column_config.ProgressColumn("Fill Rate (%)", min_value=0, max_value=100, format="%.1f%%"),
                    "Noise": st.column_config.NumberColumn("Noise", format="%d"),
                    "Missing": st.column_config.NumberColumn("Missing", format="%d"),
                    "Safe Zone Violations": st.column_config.NumberColumn("Safe Zone Violations", format="%d"),
                },
            )

            # Render Actionable Suggestions Below
            st.markdown("<div style='margin-top: 24px;'></div>", unsafe_allow_html=True)
            st.markdown(f"<h4 style='color: var(--text-muted); font-size: 0.8rem; text-transform: uppercase; letter-spacing: 1px; border-bottom: 1px solid rgba(255,255,255,0.05); padding-bottom: 8px; margin-bottom: 16px;'>{get_text('audit_actionable_suggestions', lang)}</h4>", unsafe_allow_html=True)

            suggestion_groups = {}
            if "Suggestion" in integrity_df.columns:
                # Vectorized extraction of suggestions mapping: {action: [columns]}
                # 1. Filter out empty, OK, and NaN suggestions
                valid_mask = integrity_df["Suggestion"].notna() & (integrity_df["Suggestion"] != "") & (~integrity_df["Suggestion"].str.contains("OK", na=False))
                valid_df = integrity_df.loc[valid_mask, ["Column", "Suggestion"]]
                
                # 2. Split suggestions by bullet and build dictionary
                for col, sug in zip(valid_df["Column"], valid_df["Suggestion"]):
                    for item in [s.strip() for s in str(sug).split("•") if s.strip()]:
                        suggestion_groups.setdefault(item, []).append(col)

            if suggestion_groups:
                for action, columns in suggestion_groups.items():
                    action_lower = action.lower()
                    if "missing" in action_lower or "drop" in action_lower or "impute" in action_lower:
                        badge_color, border_color = STATUS_COLORS["warning"]["hex"], STATUS_COLORS["warning"]["rgba"]
                    elif "safe zone" in action_lower or "clamp" in action_lower:
                        badge_color, border_color = STATUS_COLORS["critical"]["hex"], STATUS_COLORS["critical"]["rgba"]
                    elif "noise" in action_lower or "format" in action_lower:
                        badge_color, border_color = STATUS_COLORS["neutral"]["hex"], STATUS_COLORS["neutral"]["rgba"]
                    else:
                        badge_color, border_color = STATUS_COLORS["success"]["hex"], STATUS_COLORS["success"]["rgba"]
                    
                    cols_html = "".join([f"<span style='background:rgba(255,255,255,0.06); color:var(--text-secondary); padding:3px 10px; border-radius:6px; font-size:0.8rem; font-family:monospace;'>{c}</span>" for c in columns])
                    st.markdown(f"""
                        <div style="background: rgba(255,255,255,0.02); border-left: 3px solid {badge_color}; padding: 14px 18px; margin-bottom: 10px; border-radius: 0 10px 10px 0; transition: background 0.2s ease;" onmouseover="this.style.background='rgba(255,255,255,0.04)'" onmouseout="this.style.background='rgba(255,255,255,0.02)'">
                            <div style="margin-bottom: 8px;">
                                <span style='background:rgba(255,255,255,0.03); border:1px solid {border_color}; padding:4px 12px; border-radius:8px; font-size:0.75rem; font-weight:700; color:{badge_color}; display:inline-flex; align-items:center; gap:6px;'>
                                    {action}
                                </span>
                            </div>
                            <div style="display:flex; flex-wrap:wrap; gap:8px;">{cols_html}</div>
                        </div>
                    """, unsafe_allow_html=True)
            else:
                st.success(get_text('audit_all_fields_pass', lang))

    with tab_quality:
        quality_matrix = audit.get("quality_matrix", pd.DataFrame())
        if not quality_matrix.empty:
            st.info(get_text('quality_matrix_note', lang))
            st.markdown("<div class='chart-glow'>", unsafe_allow_html=True)
            st.plotly_chart(plot_quality_matrix(quality_matrix), use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.info(get_text('audit_not_enough_quality', lang))
    section_divider()


def _render_correlation_matrix(audit: dict, lang: str):
    """SECTION 5 — CORRELATION HEATMAP with method toggle and insight panel."""
    st.markdown(f"<div class='section-header'><h3>{get_text('correlation_matrix', lang)}</h3></div>", unsafe_allow_html=True)
    st.caption(get_text('correlation_caption', lang))

    base_corr = audit.get("correlation", pd.DataFrame())
    if base_corr.empty:
        st.info(get_text('no_numeric_correlation', lang))
        section_divider()
        return

    # --- Controls ---
    ctrl1, ctrl2 = st.columns([1, 2])
    with ctrl1:
        method = st.radio(
            get_text('corr_method_label', lang),
            ["Pearson", "Spearman"],
            horizontal=True,
            help="**Pearson** — linear relationships. **Spearman** — rank-based, robust to outliers.",
            key="corr_method",
        )
    with ctrl2:
        threshold = st.slider(
            get_text('corr_hide_weak', lang),
            min_value=0.0, max_value=0.9, value=0.0, step=0.05,
            format="%.2f",
            help="Pairs below this threshold are greyed out on the heatmap.",
            key="corr_threshold",
        )

    # Recompute only when Spearman is selected (Pearson result already cached)
    if method == "Spearman":
        # Retrieve the raw df from session to recompute with spearman
        active_file = st.session_state.get("active_file")
        from modules.core import data_engine
        raw_df = data_engine.load_and_standardize(active_file, _file_mtime=data_engine._get_file_mtime(active_file))
        corr_matrix = audit_engine.compute_correlation_matrix(raw_df, method="spearman")
    else:
        corr_matrix = base_corr

    if corr_matrix.empty:
        st.info(get_text('no_numeric_correlation', lang))
        section_divider()
        return

    # --- Heatmap ---
    st.markdown("<div class='chart-glow'>", unsafe_allow_html=True)
    st.plotly_chart(
        plot_correlation_matrix(corr_matrix, threshold=threshold),
        use_container_width=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

    # --- Top Correlations insight panel ---
    with st.expander(f":material/insights: {get_text('audit_top_correlations', lang)}", expanded=False):
        import numpy as np
        
        # Vectorized extraction of upper-triangle pairs (avoid duplicates and diagonal)
        # Keep only upper triangle (excluding diagonal) manually by masking lower+diag with NaN
        mask = np.triu(np.ones(corr_matrix.shape, dtype=bool), k=1)
        upper_corr = corr_matrix.where(mask)
        
        # Stack into a series, drop NaNs, reset index to unpivot
        pairs_df = (
            upper_corr.stack()
            .reset_index(name="r")
            .rename(columns={"level_0": "Column A", "level_1": "Column B"})
            .dropna(subset=["r"])
        )
        
        if not pairs_df.empty:
            pairs_df["r"] = pairs_df["r"].round(4)
            pairs_df = pairs_df.sort_values("r", key=abs, ascending=False).reset_index(drop=True)

            # Color-coded strength badges above the table
            strong  = pairs_df[pairs_df["r"].abs() >= 0.7]
            moderate = pairs_df[(pairs_df["r"].abs() >= 0.4) & (pairs_df["r"].abs() < 0.7)]
            weak    = pairs_df[pairs_df["r"].abs() < 0.4]

            st.markdown(f"""
                <div style="display:flex; gap:10px; flex-wrap:wrap; margin-bottom:12px;">
                    <span class="status-badge badge-red">{get_text('corr_strong', lang, n=len(strong))}</span>
                    <span class="status-badge badge-blue">{get_text('corr_moderate', lang, n=len(moderate))}</span>
                    <span class="status-badge badge-green">{get_text('corr_weak', lang, n=len(weak))}</span>
                </div>
            """, unsafe_allow_html=True)

            st.dataframe(
                pairs_df.head(20),
                use_container_width=True,
                hide_index=True,
                column_config={
                    "r": st.column_config.ProgressColumn(
                        "r (correlation)",
                        min_value=-1.0, max_value=1.0,
                        format="%.4f",
                    ),
                },
            )
        else:
            st.info(get_text('audit_no_correlation_pairs', lang))

    section_divider()



def _render_risk_inspector(df: pd.DataFrame, lang: str):
    """SECTION 6 — RISK RECORDS INSPECTOR"""
    st.markdown(f"<div class='section-header'><h3>{get_text('risk_records_inspector', lang)}</h3></div>", unsafe_allow_html=True)
    st.caption(get_text('risk_inspector_caption', lang))

    UiComponents.outlier_inspector(df, lang, key_prefix="audit")
    
    section_divider()


def _render_data_quality_details(audit: dict, lang: str):
    """SECTION 7 — DATA QUALITY DETAILS (Consistency + Noise)"""
    st.markdown(f"<div class='section-header'><h3>{get_text('categorical_consistency', lang)}</h3></div>", unsafe_allow_html=True)
    st.caption(get_text('data_quality_caption', lang))

    tab_consistency, tab_noise = st.tabs([
        f":material/check_circle: {get_text('categorical_consistency', lang)}",
        f":material/bug_report: {get_text('noise_values', lang)}",
    ])

    with tab_consistency:
        consistency_df = audit.get("consistency", pd.DataFrame())
        if not consistency_df.empty:
            n_issues = len(consistency_df)
            n_cols = consistency_df["Column"].nunique()
            total_affected = consistency_df["Count"].sum()
            st.markdown(f"""
                <div style="display:flex; gap:10px; flex-wrap:wrap; margin-bottom:12px;">
                    <span class="status-badge badge-red">{get_text('audit_issues_detected', lang, n=n_issues)}</span>
                    <span class="status-badge badge-blue">{get_text('audit_cols_affected', lang, n=n_cols)}</span>
                    <span class="status-badge badge-orange">{get_text('audit_total_cells_impacted', lang, n=total_affected)}</span>
                </div>
            """, unsafe_allow_html=True)
            
            st.dataframe(
                consistency_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Column": st.column_config.TextColumn("Column", width="medium"),
                    "Issue": st.column_config.TextColumn("Issue Type", width="medium"),
                    "Detail": st.column_config.TextColumn("Details / Examples", width="large"),
                    "Count": st.column_config.NumberColumn("Affected Rows", format="%d", width="small")
                }
            )
        else:
            st.success(get_text('all_consistent', lang))

    with tab_noise:
        noise_df = audit.get("noise_values", pd.DataFrame())
        if not noise_df.empty:
            noise_total = audit.get("noise_total", 0)
            n_noise_cols = len(noise_df)
            st.markdown(f"""
                <div style="display:flex; gap:10px; flex-wrap:wrap; margin-bottom:12px;">
                    <span class="status-badge badge-red">{get_text('audit_noise_values_count', lang, n=noise_total)}</span>
                    <span class="status-badge badge-blue">{get_text('audit_cols_affected', lang, n=n_noise_cols)}</span>
                </div>
            """, unsafe_allow_html=True)
            
            st.dataframe(
                noise_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Column": st.column_config.TextColumn("Column", width="medium"),
                    "Noise Count": st.column_config.NumberColumn("Noise Count", format="%d", width="small"),
                    "Examples": st.column_config.TextColumn("Examples found", width="large")
                }
            )
        else:
            st.success(get_text('no_noise_found', lang))
    section_divider()


def _render_download_report(df: pd.DataFrame, audit: dict, active_file: str, lang: str = "en"):
    """Generates the downloadable Professional PDF Report"""
    _, col_dl, _ = st.columns([1, 1, 1])
    with col_dl:
        try:
            field_df = audit.get("field_integrity", pd.DataFrame())
            missing_profile = pd.DataFrame()
            if not field_df.empty:
                m_df = field_df[field_df["Missing"] > 0].copy()
                if not m_df.empty:
                    m_df["Missing Count"] = m_df["Missing"]
                    total_rows = audit.get("total_records", 0)
                    m_df["Percentage"] = (m_df["Missing Count"] / total_rows * 100) if total_rows > 0 else 0
                    missing_profile = m_df[["Column", "Missing Count", "Percentage"]]

            total_cells = audit.get("total_records", 0) * audit.get("attributes", 0)
            missing_percentage = (audit.get("missing_cells", 0) / total_cells * 100) if total_cells > 0 else 0

            audit_payload = {
                "health_score": audit.get("health_score", 0),
                "total_rows": audit.get("total_records", 0),
                "total_columns": audit.get("attributes", 0),
                "duplicate_rows": audit.get("duplicates", 0),
                "missing_percentage": missing_percentage,
                "missing_df": missing_profile,
            }
            
            pdf_buffer = generate_audit_report(audit_payload, dataset_name=active_file)
            st.download_button(
                label=get_text('audit_download_pdf', lang),
                data=pdf_buffer,
                file_name=f"Data_Integrity_Audit_{active_file}.pdf",
                mime="application/pdf",
                use_container_width=True,
                type="primary"
            )
        except Exception as e:
            st.error(f"Failed to generate report: {e}")


def main():
    """Data Integrity Audit — Enterprise Dashboard."""
    lang = st.session_state.get('lang', 'en')

    # --- PAGE HEADER ---
    page_header(
        title=get_text('data_audit_title', lang),
        subtitle=get_text('overview_journey_audit_desc', lang)
    )

    # --- WORKSPACE STATUS ---
    _ensure_workspace_active()
    active_file = st.session_state.get("active_file")
    workspace_status(active_file)

    # --- LOAD DATA ---
    df = data_engine.load_and_standardize(active_file, _file_mtime=data_engine._get_file_mtime(active_file))

    # --- SAVE SNAPSHOT TO data/temp/ FOR DEBUG ---
    save_temp_csv(df, prefix="audit_snapshot")

    # --- SCANNING ANIMATION ---
    active_file_scan_progress_bar("_audit_done")

    # --- RUN AUDIT (smart cache — hash triggers recompute on data/rules/lang change) ---
    rules = get_state("analysis_rules", {})
    rules_hash = hash(json.dumps(rules, sort_keys=True, default=str))
    audit_cache_key = f"_audit_result_{lang}_{active_file}"
    audit_hash_key  = f"_audit_hash_{lang}_{active_file}"
    current_hash = (len(df), len(df.columns), int(df.isnull().sum().sum()), lang, rules_hash)

    needs_rerun = (
        audit_cache_key not in st.session_state
        or get_state(audit_hash_key) != current_hash
        or st.session_state.pop("_force_rerun_audit", False)
    )
    if needs_rerun:
        set_state(audit_cache_key, audit_engine.run_full_audit(df, lang=lang))
        set_state(audit_hash_key, current_hash)

    audit = get_state(audit_cache_key)

    # --- SAVE AUDIT RESULT TO data/temp/ FOR DEBUG ---
    save_temp_csv(audit, prefix="audit_full_result")


    # --- RENDER SECTIONS ---
    _render_executive_metrics(audit, lang)
    _render_issue_composition(audit, lang)
    _render_safe_zone_violations(audit, lang)
    _render_field_integrity(audit, lang)
    _render_correlation_matrix(audit, lang)
    _render_risk_inspector(df, lang)
    _render_data_quality_details(audit, lang)
    _render_download_report(df, audit, active_file, lang)


if __name__ == "__main__":
    main()