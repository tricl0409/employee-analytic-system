"""

data_audit.py — Data Integrity Audit Page
Enterprise Dashboard with neon-themed charts and animated UI.
"""

import streamlit as st
import pandas as pd
from modules.core import data_engine, audit_engine
from modules.ui import page_header, workspace_status, metric_card, active_file_scan_progress_bar, section_divider
from modules.ui.components import UiComponents
from modules.ui.visualizer import plot_issue_composition, plot_category_frequency
from modules.utils.localization import get_text
from modules.utils.helpers import _ensure_workspace_active, save_temp_csv
from modules.utils.theme_manager import STATUS_COLORS
from modules.utils.session_debug import set_state, get_state
import json


# Reusable HTML snippet: right-aligned "↑ Back to Overview" placed at section bottom
_BACK_TO_OVERVIEW = (
    "<div style='text-align:right; margin-top:8px; margin-bottom:4px;'>"
    "<a href='#section-issue-composition' "
    "style='color:rgba(255,255,255,0.2); text-decoration:none; "
    "font-size:0.75rem; font-weight:500; letter-spacing:0.5px; transition:color 0.2s;' "
    "onmouseover=\"this.style.color='rgba(255,255,255,0.5)'\" "
    "onmouseout=\"this.style.color='rgba(255,255,255,0.2)'\">↑ Back to Overview</a>"
    "</div>"
)


def _section_subtitle(text: str) -> None:
    """Render a muted subtitle below a section header — single source of truth for styling."""
    st.markdown(
        f"<p style='font-size:0.85rem; color:rgba(255,255,255,0.4); "
        f"margin-top:-8px; margin-bottom:12px;'>{text}</p>",
        unsafe_allow_html=True,
    )


def _badge_bar(badges: list) -> None:
    """

    Render a horizontal row of status badges.
    Args:
        badges: List of ``(label, css_class)`` tuples.
                *css_class* is one of ``'badge-blue'``, ``'badge-red'``,
                ``'badge-green'``, ``'badge-orange'``, or a raw ``style="..."``
                string for custom colours.
    """

    spans = []
    for label, css in badges:
        if css.startswith("style="):
            spans.append(f'<span class="status-badge" {css}>{label}</span>')
        else:
            spans.append(f'<span class="status-badge {css}">{label}</span>')
    st.markdown(
        f'<div style="display:flex; gap:10px; flex-wrap:wrap; margin-bottom:12px;">'
        f'{" ".join(spans)}</div>',
        unsafe_allow_html=True,
    )


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


def _render_key_issues_summary(audit: dict, df: pd.DataFrame) -> None:
    """Render a glassmorphism 'Key Issues Identified' card with severity dots."""
    # severity: "red" | "orange" | "yellow"
    issues: list = []
    # 1. Missing values
    missing_by_col = df.isnull().sum()
    missing_cols = missing_by_col[missing_by_col > 0]
    if not missing_cols.empty:
        col_names = ", ".join(f'<b style="color:#F59E0B">{c}</b>' for c in missing_cols.index)
        issues.append(("orange", f"Missing values detected in {col_names}"))
    # 2. Noise values
    noise_df = audit.get("noise_values", pd.DataFrame())
    if not noise_df.empty and "Column" in noise_df.columns:
        noise_cols = ", ".join(f'<b style="color:#F59E0B">{c}</b>' for c in noise_df["Column"].tolist())
        examples = noise_df["Examples"].tolist()[:2]
        example_str = ", ".join(f'\"{e}\"' for e in examples if e)
        if example_str:
            issues.append(("orange", f"Noise values identified in {noise_cols} (e.g. {example_str})"))
        else:
            issues.append(("orange", f"Noise values identified in {noise_cols}"))
    # 3. Extreme skewness
    skewed_cols = []
    for col in df.select_dtypes(include=["number"]).columns:
        clean = df[col].dropna()
        if len(clean) > 2 and abs(float(clean.skew())) > 1.0:
            skewed_cols.append(col)
    if skewed_cols:
        col_names = ", ".join(f'<b style="color:#F59E0B">{c}</b>' for c in skewed_cols)
        issues.append(("yellow", f'Extreme skewness (<b style="color:#F59E0B">|skew| > 1.0</b>) in {col_names}'))
    # 4. Duplicates
    dup_count = audit.get("duplicates", 0)
    if dup_count > 0:
        issues.append(("red", f'<b style="color:#F59E0B">{dup_count:,}</b> duplicate rows detected'))
    # 5. Outliers — centralised dispatch
    outlier_cols = []
    for col in df.select_dtypes(include=["number"]).columns:
        series = df[col].dropna()
        if len(series) > 2:
            rec = audit_engine.evaluate_outlier_method(series)
            method_key = {"IQR": "iqr", "Z-Score": "zscore", "Modified Z-Score": "modified_zscore"}[rec["method"]]
            threshold = 1.5 if method_key == "iqr" else 3.0
            compute_fn = audit_engine._OUTLIER_METHODS[method_key]
            mask, _ = compute_fn(series.values, threshold)
            if mask.sum() > 0:
                outlier_cols.append(col)
    if outlier_cols:
        col_names = ", ".join(f'<b style="color:#F59E0B">{c}</b>' for c in outlier_cols)
        issues.append(("orange", f"Outliers detected in {col_names}"))
    # 6. Inconsistencies
    incon = audit.get("inconsistency_total", 0)
    if incon > 0:
        issues.append(("red", f'<b style="color:#F59E0B">{incon:,}</b> data consistency violations found'))

    # --- Severity dot colors ---

    dot_colors = {"red": "#FF5B5C", "orange": "#FF9F43", "yellow": "#F59E0B"}

    # --- Section header ---

    st.markdown("<div class='section-header'><h3>Key Issues Identified</h3></div>", unsafe_allow_html=True)
    _section_subtitle(
        "Automated diagnostic summary based on the current dataset. "
        "Each finding is ranked by <b style='color:rgba(255,255,255,0.65)'>severity</b> to prioritize data cleaning."
    )

    # --- Render glassmorphism card ---

    if issues:
        rows_html = ""
        for severity, text in issues:
            dot = dot_colors.get(severity, dot_colors["orange"])
            rows_html += (
                f'<div style="display:flex; align-items:flex-start; gap:10px; padding:9px 0;'
                f' border-bottom:1px solid rgba(255,255,255,0.04);">'
                f'<span style="display:inline-block; min-width:8px; width:8px; height:8px;'
                f' border-radius:50%; background:{dot}; margin-top:6px;'
                f' box-shadow:0 0 6px {dot}40;"></span>'
                f'<span style="flex:1;">{text}</span>'
                f'</div>'
            )

        _card_html = (
            '<div style="margin:4px 0 8px 0; padding:16px 20px;'
            ' background:rgba(255,255,255,0.03);'
            ' backdrop-filter:blur(12px); -webkit-backdrop-filter:blur(12px);'
            ' border:1px solid rgba(255,255,255,0.06);'
            ' border-radius:16px;'
            ' font-size:0.82rem; color:rgba(255,255,255,0.55); line-height:1.7;">'
            '<div style="display:flex; justify-content:flex-end; margin-bottom:6px;">'
            f'<span style="background:rgba(255,95,92,0.15); color:#FF5B5C; padding:2px 10px;'
            f' border-radius:12px; font-size:0.72rem; font-weight:600;">'
            f'{len(issues)} issue{"s" if len(issues) != 1 else ""} found'
            f'</span></div>'
            f'{rows_html}'
            '</div>'
        )

        st.markdown(_card_html, unsafe_allow_html=True)
    else:
        _ok_html = (
            '<div style="margin:4px 0 8px 0; padding:16px 20px;'
            ' background:rgba(255,255,255,0.03);'
            ' backdrop-filter:blur(12px); -webkit-backdrop-filter:blur(12px);'
            ' border:1px solid rgba(255,255,255,0.06);'
            ' border-radius:16px;'
            ' font-size:0.82rem; color:rgba(255,255,255,0.55); line-height:1.7;">'
            '<div style="display:flex; align-items:center; gap:10px;">'
            '<span style="display:inline-block; width:8px; height:8px; border-radius:50%;'
            ' background:#7FB135; box-shadow:0 0 6px rgba(127,177,53,0.4);"></span>'
            '<b style="color:rgba(255,255,255,0.75);">No Issues Detected</b>'
            '</div>'
            '<span style="margin-left:18px; color:rgba(255,255,255,0.4);">'
            'All quality checks passed. The dataset appears clean and ready for analysis.'
            '</span></div>'
        )

        st.markdown(_ok_html, unsafe_allow_html=True)
    section_divider()


def _render_issue_composition(audit: dict, lang: str):
    """SECTION — ISSUE COMPOSITION CHART (overview bar chart)"""
    st.markdown(f"<div id='section-issue-composition' class='section-header'><h3>{get_text('issue_composition', lang)}</h3></div>", unsafe_allow_html=True)
    _section_subtitle("Proportional breakdown of all detected data quality issues — "
        "<b style='color:rgba(255,255,255,0.65)'>missing values</b>, "
        "<b style='color:rgba(255,255,255,0.65)'>duplicates</b>, "
        "<b style='color:rgba(255,255,255,0.65)'>inconsistencies</b>, and "
        "<b style='color:rgba(255,255,255,0.65)'>noise</b> — relative to total dataset cells.")
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
        dup_label = get_text("compose_duplicates", lang)
        display_overrides = {dup_label: audit.get("duplicates", 0)}
        total_issues = sum(issue_dict.values())
        clean_pct = round((total_values - total_issues) / total_values * 100, 1) if total_values > 0 else 100.0
        issue_pct = round(total_issues / total_values * 100, 2) if total_values > 0 else 0.0
        _badge_bar([
            (get_text('audit_dataset_cells', lang, n=total_values), "badge-blue"),
            (get_text('audit_affected_pct', lang, pct=issue_pct, n=total_issues), "badge-red"),
            (get_text('audit_clean_pct', lang, pct=clean_pct), "badge-green"),
        ])
        st.markdown("<div class='chart-glow'>", unsafe_allow_html=True)
        fig = plot_issue_composition(issue_dict, total_values, display_overrides=display_overrides)
        # Each bar click scrolls to the corresponding dedicated section below
        LABEL_TO_ANCHOR = {
            "missing":     "section-missing-values",
            "duplicat":    "section-duplicate-rows",
            "noise":       "section-noise-values",
            "inconsisten": "section-inconsistencies",
        }

        event = st.plotly_chart(
            fig,
            use_container_width=True,
            on_select="rerun",
            selection_mode=["points"],
            key="issue_chart",
        )

        if event and event.selection and event.selection.points:
            pt = event.selection.points[0]
            bar_label = str(pt.get("y", "")).lower()
            anchor = next(
                (anc for kw, anc in LABEL_TO_ANCHOR.items() if kw in bar_label), ""
            )

            if anchor:
                st.components.v1.html(
                    f"<script>"
                    f"var e=window.parent.document.getElementById('{anchor}');"
                    f"if(e)e.scrollIntoView({{behavior:'smooth',block:'start'}});"
                    f"</script>",
                    height=0,
                )

        st.markdown("</div>", unsafe_allow_html=True)
    section_divider()


def _render_missing_values(audit: dict, df: pd.DataFrame, lang: str):
    """SECTION — MISSING VALUES breakdown by column."""
    st.markdown("<div id='section-missing-values' class='section-header'><h3>Missing Values</h3></div>", unsafe_allow_html=True)
    _section_subtitle("Column-level breakdown of "
        "<b style='color:rgba(255,255,255,0.65)'>null</b> and "
        "<b style='color:rgba(255,255,255,0.65)'>empty cells</b> — shows which fields "
        "have gaps and how significant each gap is relative to "
        "<b style='color:rgba(255,255,255,0.65)'>total rows</b>.")
    total_rows = audit.get("total_records", 0)
    total_missing = audit.get("missing_cells", 0)
    if total_missing == 0:
        st.success("No missing values detected in the dataset.")
        section_divider()
        return
    total_cols = audit.get("attributes", 0)
    total_values = total_rows * total_cols
    missing_pct = round(total_missing / total_values * 100, 2) if total_values > 0 else 0.0
    # Build missing breakdown per column
    missing_series = df.isnull().sum()
    missing_df = pd.DataFrame({
        "Column": missing_series.index,
        "Missing": missing_series.values,
    })
    missing_df = missing_df[missing_df["Missing"] > 0].copy()
    missing_df["% Missing"] = (missing_df["Missing"] / total_rows * 100).round(2) if total_rows > 0 else 0.0
    missing_df = missing_df.sort_values("Missing", ascending=False).reset_index(drop=True)
    n_cols = len(missing_df)
    _badge_bar([
        (f"{total_missing:,} missing cells", "badge-red"),
        (f"{missing_pct}% of total values", 'style="background:rgba(245,158,11,0.12); color:#F59E0B;"'),
        (f"{n_cols} columns affected", "badge-blue"),
    ])
    st.dataframe(
        missing_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Column": st.column_config.TextColumn("Column", width=180),
            "Missing": st.column_config.NumberColumn("Missing Count", format="%d", width=120),
            "% Missing": st.column_config.ProgressColumn("% Missing", min_value=0, max_value=100, format="%.2f%%", width=150),
        },
    )

    st.markdown(_BACK_TO_OVERVIEW, unsafe_allow_html=True)
    section_divider()


def _render_noise_values(audit: dict, lang: str):
    """SECTION — NOISE VALUES — placeholder/garbage values detected."""
    st.markdown("<div id='section-noise-values' class='section-header'><h3>Noise Values</h3></div>", unsafe_allow_html=True)
    _section_subtitle("Detected "
        "<b style='color:rgba(255,255,255,0.65)'>placeholder</b> or "
        "<b style='color:rgba(255,255,255,0.65)'>corrupted values</b> such as "
        "<code style='color:#F59E0B; background:rgba(245,158,11,0.1); padding:1px 5px; border-radius:3px;'>?</code>, "
        "<code style='color:#F59E0B; background:rgba(245,158,11,0.1); padding:1px 5px; border-radius:3px;'>-</code>, "
        "<code style='color:#F59E0B; background:rgba(245,158,11,0.1); padding:1px 5px; border-radius:3px;'>N/A</code>, or "
        "<b style='color:rgba(255,255,255,0.65)'>whitespace-only</b> entries that should be cleaned before analysis.")
    noise_df = audit.get("noise_values", pd.DataFrame())
    noise_total = audit.get("noise_total", 0)
    if noise_total == 0 or noise_df.empty:
        st.success(get_text('no_noise_found', lang))
        section_divider()
        return
    n_noise_cols = len(noise_df)
    total_rows = audit.get("total_records", 0)
    noise_df = noise_df.copy()
    noise_df["% Noise"] = (noise_df["Noise Count"] / total_rows * 100).round(2) if total_rows > 0 else 0.0
    _badge_bar([
        (f"{noise_total:,} noise values", "badge-red"),
        (f"{n_noise_cols} columns affected", "badge-blue"),
    ])
    st.dataframe(
        noise_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Column": st.column_config.TextColumn("Column", width=180),
            "Noise Count": st.column_config.NumberColumn("Noise Count", format="%d", width=120),
            "% Noise": st.column_config.ProgressColumn("% Noise", min_value=0, max_value=100, format="%.2f%%", width=150),
            "Examples": st.column_config.TextColumn("Noise Values"),
        },
    )

    st.markdown(_BACK_TO_OVERVIEW, unsafe_allow_html=True)
    section_divider()


def _render_duplicate_rows(audit: dict, df: pd.DataFrame, lang: str):
    """SECTION — DUPLICATE ROWS with actual row content."""
    st.markdown("<div id='section-duplicate-rows' class='section-header'><h3>Duplicate Rows</h3></div>", unsafe_allow_html=True)
    _section_subtitle("<b style='color:rgba(255,255,255,0.65)'>Exact duplicate rows</b> "
        "detected in the dataset — rows that are "
        "<b style='color:rgba(255,255,255,0.65)'>identical across all columns</b> "
        "and should be deduplicated before modeling.")
    dup_count = audit.get("duplicates", 0)
    total_rows = audit.get("total_records", 0)
    if dup_count == 0:
        st.success("No duplicate rows detected in the dataset.")
        section_divider()
        return
    dup_pct = round(dup_count / total_rows * 100, 2) if total_rows > 0 else 0.0
    _badge_bar([
        (f"{dup_count:,} duplicate rows", "badge-red"),
        (f"{dup_pct}% of total rows", 'style="background:rgba(245,158,11,0.12); color:#F59E0B;"'),
        (f"{total_rows:,} total rows", "badge-blue"),
    ])

    # --- Build grouped duplicate table ---

    dup_mask = df.duplicated(keep=False)
    if dup_mask.any():
        dup_rows = df[dup_mask].copy()
        # Group identical rows and count occurrences
        group_cols = list(df.columns)
        dup_grouped = dup_rows.groupby(group_cols, dropna=False).size().reset_index(name="Occurrences")
        dup_grouped = dup_grouped.sort_values("Occurrences", ascending=False).reset_index(drop=True)
        # Move Occurrences to the first column
        cols = ["Occurrences"] + [c for c in dup_grouped.columns if c != "Occurrences"]
        dup_grouped = dup_grouped[cols]
        n_groups = len(dup_grouped)
        total_dup_rows = int(dup_grouped["Occurrences"].sum())
        redundant = total_dup_rows - n_groups  # rows that would be removed (keep one per group)
        st.markdown(f"""

            <p style="font-size:0.78rem; color:rgba(255,255,255,0.35); margin-bottom:10px;">
                <b style="color:rgba(255,255,255,0.55);">{n_groups:,}</b> unique row patterns found with
                <b style="color:rgba(255,255,255,0.55);">{total_dup_rows:,}</b> total instances
                — <b style="color:rgba(255,255,255,0.55);">{redundant:,}</b> redundant rows will be removed.
                Showing top {min(n_groups, 100)} groups.
            </p>
        """, unsafe_allow_html=True)

        max_occ = int(dup_grouped["Occurrences"].max())
        st.dataframe(
            dup_grouped.head(100),
            use_container_width=True,
            hide_index=True,
            column_config={
                "Occurrences": st.column_config.ProgressColumn("Occurrences", min_value=0, max_value=max_occ, format="%d", width=130),
            },
        )

    else:
        st.info(f"**{dup_count:,}** exact duplicate rows found out of **{total_rows:,}** total records.")
    st.markdown(_BACK_TO_OVERVIEW, unsafe_allow_html=True)
    section_divider()


def _render_inconsistencies(audit: dict, lang: str):
    """SECTION — INCONSISTENCIES — categorical data quality issues."""
    st.markdown("<div id='section-inconsistencies' class='section-header'><h3>Inconsistencies</h3></div>", unsafe_allow_html=True)
    _section_subtitle("Categorical data quality issues — "
        "<b style='color:rgba(255,255,255,0.65)'>mixed casing</b>, "
        "<b style='color:rgba(255,255,255,0.65)'>leading/trailing whitespace</b>, "
        "<b style='color:rgba(255,255,255,0.65)'>rare singletons</b>, "
        "and other formatting inconsistencies across text columns.")
    consistency_df = audit.get("consistency", pd.DataFrame())
    it = audit.get("inconsistency_total", 0)
    if it == 0 or consistency_df.empty:
        st.success(get_text('all_consistent', lang))
        st.markdown(_BACK_TO_OVERVIEW, unsafe_allow_html=True)
        section_divider()
        return
    n_issues = len(consistency_df)
    n_cols = consistency_df["Column"].nunique()
    total_affected = consistency_df["Count"].sum()
    _badge_bar([
        (get_text('audit_issues_detected', lang, n=n_issues), "badge-red"),
        (get_text('audit_cols_affected', lang, n=n_cols), "badge-blue"),
        (get_text('audit_total_cells_impacted', lang, n=total_affected), "badge-orange"),
    ])
    st.dataframe(
        consistency_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Column": st.column_config.TextColumn("Column", width="medium"),
            "Issue": st.column_config.TextColumn("Issue Type", width="medium"),
            "Detail": st.column_config.TextColumn("Details / Examples", width="large"),
            "Count": st.column_config.NumberColumn("Affected Rows", format="%d", width="small"),
        },
    )

    st.markdown(_BACK_TO_OVERVIEW, unsafe_allow_html=True)
    section_divider()


def _render_data_summary(df: pd.DataFrame, lang: str):
    """SECTION — STATISTICAL OVERVIEW (per-column descriptive statistics)."""
    st.markdown(f"<div class='section-header'><h3>{get_text('audit_data_summary_title', lang)}</h3></div>", unsafe_allow_html=True)
    _section_subtitle(get_text('audit_data_summary_caption', lang))
    summary_df = audit_engine.compute_data_summary(df)
    if summary_df.empty:
        st.info("No data available for summary.")
        section_divider()
        return
    n_total = len(summary_df)
    n_numeric = int(df.select_dtypes(include=['number']).shape[1])
    n_categorical = n_total - n_numeric
    _badge_bar([
        (get_text('audit_total_cols', lang, n=n_total), "badge-blue"),
        (get_text('audit_n_numeric', lang, n=n_numeric), "badge-green"),
        (get_text('audit_n_categorical', lang, n=n_categorical), "badge-orange"),
    ])

    # --- Explanation note for complex columns ---

    st.markdown("""

        <div style="margin:4px 0 12px 0; padding:12px 16px; background:rgba(59,130,246,0.12);
            border-left:3px solid rgba(59,130,246,0.4); border-radius:0 8px 8px 0;
            font-size:0.78rem; color:rgba(255,255,255,0.45); line-height:1.9;">
            <b style="color:rgba(255,255,255,0.6);">ℹ Methodology Notes</b><br>
            <b style="color:#F59E0B;">Distribution</b> —
            Classified by <b>skewness coefficient</b>:
            <b>|skew| &lt; 0.5</b> → <i>Approximately Symmetric</i>&nbsp;&nbsp;·&nbsp;&nbsp;
            <b>skew ≥ 0.5</b> → <i>Right Skewed</i>&nbsp;&nbsp;·&nbsp;&nbsp;
            <b>skew ≤ −0.5</b> → <i>Left Skewed</i><br>
            <b style="color:#F59E0B;">Central Tendency</b> —
            Categorical columns report the <b>Mode</b> (most frequent value).
            Numeric columns use <b>Mean</b> when no outliers are present,
            or <b>Median</b> when outliers are detected (more robust to extreme values).<br>
            <b style="color:#F59E0B;">Outliers</b> —
            Detection method is automatically selected based on the skewness of each column:
            <b>|skew| &lt; 0.5</b> → <b>Z-Score</b>&nbsp;&nbsp;·&nbsp;&nbsp;
            <b>0.5 ≤ |skew| ≤ 1.0</b> → <b>IQR</b>&nbsp;&nbsp;·&nbsp;&nbsp;
            <b>|skew| &gt; 1.0</b> → <b>Modified Z-Score</b>
        </div>
    """, unsafe_allow_html=True)

    st.dataframe(
        summary_df,
        use_container_width=True,
        hide_index=True,
        height=min(42 + len(summary_df) * 35, 600),
        column_config={
            "Column": st.column_config.TextColumn("Attribute", width=130),
            "Type": st.column_config.TextColumn("Dtype", width=75),
            "Records": st.column_config.NumberColumn("Non-Null", format="%d", width=80),
            "Missing %": st.column_config.ProgressColumn("Missing (%)", min_value=0, max_value=100, format="%.2f%%", width=105),
            "Unique Values": st.column_config.NumberColumn("Distinct", format="%d", width=75),
            "Noise": st.column_config.TextColumn("Noise", width=60),
            "Distribution": st.column_config.TextColumn("Distribution", width=220),
            "Central Value": st.column_config.TextColumn("Central Tendency", width=220),
            "Outliers": st.column_config.TextColumn("Outliers", width=230),
        },
    )

    section_divider()


def _render_category_frequency(df: pd.DataFrame, lang: str):
    """SECTION — CATEGORY FREQUENCY BAR CHART."""
    st.markdown("<div class='section-header'><h3>Category Frequency</h3></div>", unsafe_allow_html=True)
    _section_subtitle(
        "Visualize the <b style='color:rgba(255,255,255,0.65)'>value distribution</b> of categorical columns. "
        "<b style='color:rgba(255,255,255,0.65)'>Rare categories</b> (\u2264 1% frequency) are highlighted in <b style='color:#F59E0B'>amber</b>."
    )

    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    if not cat_cols:
        st.info("No categorical columns available.")
        section_divider()
        return
    selected_col = st.selectbox(
        "Select Column",
        cat_cols,
        key="cat_freq_col",
        index=0,
    )

    if selected_col:
        series = df[selected_col].dropna().astype(str)
        total = len(series)
        n_unique = series.nunique()
        rare_count = int((series.value_counts() / total * 100 <= 1.0).sum())
        _badge_bar([
            (f"Total: {total:,}", "badge-blue"),
            (f"Distinct: {n_unique}", "badge-green"),
            (f"Rare (\u2264 1%): {rare_count}", "badge-orange"),
        ])
        fig = plot_category_frequency(series, selected_col)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    section_divider()


def _render_risk_inspector(df: pd.DataFrame, lang: str):
    """SECTION — RISK RECORDS INSPECTOR"""
    st.markdown(f"<div class='section-header'><h3>{get_text('risk_records_inspector', lang)}</h3></div>", unsafe_allow_html=True)
    _section_subtitle(get_text('audit_risk_inspector_caption', lang))
    UiComponents.outlier_inspector(df, lang, key_prefix="audit")
    section_divider()


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

    # --- EXECUTIVE METRICS (always visible above tabs) ---
    _render_executive_metrics(audit, lang)

    # --- KEY ISSUES IDENTIFIED (standalone section) ---
    _render_key_issues_summary(audit, df)

    # --- 3-TAB LAYOUT ---
    tab1, tab2, tab3 = st.tabs([
        ":material/pie_chart: Quality Issues",
        ":material/table_chart: Statistical Profiling",
        ":material/bar_chart: Diagnostic Charts",
    ])
    with tab1:
        _render_issue_composition(audit, lang)
        _render_missing_values(audit, df, lang)
        _render_noise_values(audit, lang)
        _render_duplicate_rows(audit, df, lang)
        _render_inconsistencies(audit, lang)
    with tab2:
        _render_data_summary(df, lang)
    with tab3:
        _render_category_frequency(df, lang)
        _render_risk_inspector(df, lang)
if __name__ == "__main__":
    main()