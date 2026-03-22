"""

data_audit.py — Data Integrity Audit Page
Enterprise Dashboard with neon-themed charts and animated UI.
"""

import streamlit as st
import pandas as pd
from modules.core import data_engine, audit_engine
from modules.ui import page_header, workspace_status, active_file_scan_progress_bar, section_divider
from modules.ui.components import UiComponents
from modules.ui.visualizer import plot_issue_composition, plot_category_frequency
from modules.utils.localization import get_text
from modules.utils.helpers import _ensure_workspace_active, save_temp_csv
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


def _styled_header(
    title: str,
    anchor: str = "",
    accent: str = "#3B82F6",
) -> None:
    """Render a premium section header consistent with preprocessing detail panel."""
    h = accent.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    rgb = f"{r},{g},{b}"
    id_attr = f' id="{anchor}"' if anchor else ""
    st.markdown(
        f'<div{id_attr} style="margin-bottom:16px; margin-top:4px; padding:14px 18px;'
        f' background:linear-gradient(135deg, rgba({rgb},0.08) 0%, rgba({rgb},0.02) 100%);'
        f' border:1px solid rgba({rgb},0.12); border-left:3px solid {accent};'
        f' border-radius:0 12px 12px 0;">'
        f'<span style="font-size:1.05rem; font-weight:700;'
        f' color:rgba(255,255,255,0.92); letter-spacing:-0.2px;">{title}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )


def _skip_msg(text: str, accent: str = "#3B82F6") -> None:
    """Styled 'all clear' card — replaces st.success()."""
    h = accent.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    rgb = f"{r},{g},{b}"
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


def _glass_card(content_html: str) -> None:
    """Render content inside a reusable glassmorphism card container."""
    st.markdown(
        '<div style="margin:4px 0 8px 0; padding:16px 20px;'
        ' background:rgba(255,255,255,0.03);'
        ' backdrop-filter:blur(12px); -webkit-backdrop-filter:blur(12px);'
        ' border:1px solid rgba(255,255,255,0.06);'
        ' border-radius:16px;'
        f' font-size:0.82rem; color:rgba(255,255,255,0.55); line-height:1.7;">{content_html}</div>',
        unsafe_allow_html=True,
    )


def _render_dataset_introduction(lang: str) -> None:
    """Render the Dataset Introduction table — static schema metadata for Adult Census."""
    _styled_header(get_text('audit_dataset_intro_title', lang))
    _section_subtitle(
        "Reference table describing each "
        "<b style='color:rgba(255,255,255,0.65)'>attribute</b> in the dataset, "
        "grouped by "
        "<b style='color:rgba(255,255,255,0.65)'>domain category</b> "
        "with its data nature."
    )

    # ── colour tokens ────────────────────────────────────────────────────
    _GRP_COLORS = {
        "demo":    "#60A5FA",   # Blue
        "socio":   "#F59E0B",   # Amber
        "employ":  "rgba(255,255,255,0.72)",  # Bright white
        "finance": "#F87171",   # Red
        "meta":    "#A78BFA",   # Purple
        "target":  "#F97316",   # Orange
    }


    # ── table rows data ──────────────────────────────────────────────────
    # (group_label, group_color, rowspan, no, attr, meaning, nature_html)
    _ROWS = [
        ("Personal Demographics",          _GRP_COLORS["demo"],  4, 1, "Age",            "Age of the individual (in years)",                              "Quantitative Variables"),
        (None,                             None,                 0, 2, "Race",           "Ethnic or racial classification",                               "Categorical Variables"),
        (None,                             None,                 0, 3, "Sex",            "Biological sex of the individual",                              "Categorical Variables"),
        (None,                             None,                 0, 4, "Native_Country", "Country of origin or citizenship",                              "Categorical Variables"),

        ("Socioeconomic & Education",      _GRP_COLORS["socio"], 4, 5, "Education",      "Highest level of education attained",                           "Categorical Variables"),
        (None,                             None,                 0, 6, "Education_Num",  "Ordinal encoding of education level (1\u201316)",                    "Categorical Variables"),
        (None,                             None,                 0, 7, "Marital_Status", "Current marital status classification",                         "Categorical Variables"),
        (None,                             None,                 0, 8, "Relationship",   "Role within the household or family unit",                      "Categorical Variables"),

        ("Employment & Occupation",        _GRP_COLORS["employ"],3, 9, "Workclass",      "Employment sector (Private, Government, Self-employed, etc.)",  "Categorical Variables"),
        (None,                             None,                 0,10, "Occupation",     "Professional occupation or job classification",                 "Categorical Variables"),
        (None,                             None,                 0,11, "Hours_per_Week", "Average weekly working hours",                                  "Quantitative Variables"),

        ("Financial Indicators",           _GRP_COLORS["finance"],2,12,"Capital_Gain",   "Capital gains from investment or asset sales",                  "Financial Variables"),
        (None,                             None,                 0,13, "Capital_Loss",   "Capital losses from investment or asset sales",                 "Financial Variables"),

        ("Sampling & Technical Metadata",  _GRP_COLORS["meta"],1,14,"Fnlwgt",       "Final sampling weight assigned by the Census Bureau",           "Quantitative Variables"),

        ("Target Variable",                _GRP_COLORS["target"],1,15, "Income",         "Annual income bracket (\u226450K / >50K)",                           "Binary categorical label"),
    ]

    # ── build HTML rows ──────────────────────────────────────────────────
    rows_html = ""
    is_target_row = False
    for group_label, group_color, rowspan, no, attr, meaning, nature_html in _ROWS:
        group_td = ""
        if group_label is not None:
            is_target_row = group_label == "Target Variable"
            group_td = (
                f"<td rowspan='{rowspan}' style='"
                f"font-weight:700; font-size:0.82rem; color:{group_color};"
                f" padding:12px 14px; vertical-align:middle;"
                f" border-bottom:1px solid rgba(255,255,255,0.06);"
                f" border-right:1px solid rgba(255,255,255,0.06);"
                f" white-space:nowrap;"
                f"'>{group_label}</td>"
            )

        # Highlight target variable row with subtle accent background
        row_bg = " background:rgba(249,115,22,0.08);" if is_target_row else ""
        rows_html += (
            f"<tr style='border-bottom:1px solid rgba(255,255,255,0.06);{row_bg}'>"
            f"{group_td}"
            f"<td style='text-align:center; color:rgba(255,255,255,0.4); padding:10px 8px;"
            f" font-size:0.80rem; border-right:1px solid rgba(255,255,255,0.06);'>{no}</td>"
            f"<td style='font-weight:600; color:rgba(255,255,255,0.82); padding:10px 14px;"
            f" font-size:0.82rem; border-right:1px solid rgba(255,255,255,0.06);'>{attr}</td>"
            f"<td style='color:rgba(255,255,255,0.55); padding:10px 14px;"
            f" font-size:0.80rem; border-right:1px solid rgba(255,255,255,0.06);'>{meaning}</td>"
            f"<td style='text-align:left; padding:10px 14px;"
            f" font-size:0.80rem; color:rgba(255,255,255,0.55);'>{nature_html}</td>"
            f"</tr>"
        )

    # ── full table ───────────────────────────────────────────────────────
    table_html = (
        "<div style='"
        "background:rgba(255,255,255,0.02);"
        "border:1px solid rgba(255,255,255,0.06);"
        "border-radius:16px;"
        "overflow:hidden;"
        "backdrop-filter:blur(12px); -webkit-backdrop-filter:blur(12px);"
        "margin-bottom:16px;"
        "'>"
        "<table style='width:100%; border-collapse:collapse; table-layout:auto;'>"
        "<thead>"
        "<tr style='background:rgba(255,255,255,0.04); border-bottom:1px solid rgba(255,255,255,0.08);'>"
        "<th style='text-align:left; padding:12px 14px; font-size:0.75rem;"
        " font-weight:700; color:rgba(255,255,255,0.5);"
        " text-transform:uppercase; letter-spacing:1px; width:20%;"
        " border-right:1px solid rgba(255,255,255,0.06);'>Attribute Group</th>"
        "<th style='text-align:center; padding:12px 8px; font-size:0.75rem;"
        " font-weight:700; color:rgba(255,255,255,0.5);"
        " text-transform:uppercase; letter-spacing:1px; width:5%;"
        " border-right:1px solid rgba(255,255,255,0.06);'>No</th>"
        "<th style='text-align:left; padding:12px 14px; font-size:0.75rem;"
        " font-weight:700; color:rgba(255,255,255,0.5);"
        " text-transform:uppercase; letter-spacing:1px; width:16%;"
        " border-right:1px solid rgba(255,255,255,0.06);'>Attribute</th>"
        "<th style='text-align:left; padding:12px 14px; font-size:0.75rem;"
        " font-weight:700; color:rgba(255,255,255,0.5);"
        " text-transform:uppercase; letter-spacing:1px; width:36%;"
        " border-right:1px solid rgba(255,255,255,0.06);'>Meaning</th>"
        "<th style='text-align:left; padding:12px 14px; font-size:0.75rem;"
        " font-weight:700; color:rgba(255,255,255,0.5);"
        " text-transform:uppercase; letter-spacing:1px; width:23%;'>Nature Variables</th>"
        "</tr>"
        "</thead>"
        f"<tbody>{rows_html}</tbody>"
        "</table>"
        "</div>"
    )

    st.markdown(table_html, unsafe_allow_html=True)
    section_divider()



def _render_key_issues_summary(audit: dict, lang: str) -> None:
    """Render a glassmorphism 'Key Issues Identified' card with severity dots.

    Uses precomputed lists from `audit` dict — NO re-computation.
    """
    t = lambda key, **kw: get_text(key, lang, **kw)
    issues: list = []

    # 1. Missing values — from precomputed audit["missing_columns"]
    missing_cols = audit.get("missing_columns", [])
    if missing_cols:
        col_names = ", ".join(f'<b style="color:#F59E0B">{c}</b>' for c in missing_cols)
        issues.append(("orange", f"Missing values detected in {col_names}"))

    # 2. Noise values — from audit dict
    noise_df = audit.get("noise_values", pd.DataFrame())
    if not noise_df.empty and "Column" in noise_df.columns:
        noise_cols = ", ".join(f'<b style="color:#F59E0B">{c}</b>' for c in noise_df["Column"].tolist())
        examples = noise_df["Examples"].tolist()[:2]
        example_str = ", ".join(f'\"{e}\"' for e in examples if e)
        if example_str:
            issues.append(("orange", f"Noise values identified in {noise_cols} (e.g. {example_str})"))
        else:
            issues.append(("orange", f"Noise values identified in {noise_cols}"))

    # 3. Extreme skewness — from precomputed audit["skewed_columns"]
    skewed_cols = audit.get("skewed_columns", [])
    if skewed_cols:
        col_names = ", ".join(f'<b style="color:#F59E0B">{c}</b>' for c in skewed_cols)
        issues.append(("yellow", f'Extreme skewness (<b style="color:#F59E0B">|skew| > 1.0</b>) in {col_names}'))

    # 4. Duplicates — from audit dict
    dup_count = audit.get("duplicates", 0)
    if dup_count > 0:
        issues.append(("red", f'<b style="color:#FF5B5C">{dup_count:,}</b> duplicate rows detected'))

    # 5. Outliers — from precomputed audit["outlier_columns"]
    outlier_cols = audit.get("outlier_columns", [])
    if outlier_cols:
        col_names = ", ".join(f'<b style="color:#F59E0B">{c}</b>' for c in outlier_cols)
        issues.append(("orange", f"Outliers detected in {col_names}"))

    # 6. Inconsistencies — from audit dict
    incon = audit.get("inconsistency_total", 0)
    if incon > 0:
        issues.append(("red", f'<b style="color:#FF5B5C">{incon:,}</b> data consistency violations found'))

    # 7. Low-variance columns — from precomputed audit["low_variance_columns"]
    lv_cols = audit.get("low_variance_columns", [])
    if lv_cols:
        col_names = ", ".join(f'<b style="color:#F59E0B">{c}</b>' for c in lv_cols)
        issues.append(("yellow", f'Low-variance (≥ 80% identical values) in {col_names} — outlier detection may fail'))

    # --- Severity dot colors ---
    dot_colors = {"red": "#FF5B5C", "orange": "#FF9F43", "yellow": "#F59E0B"}

    # --- Section header ---
    _styled_header(t('audit_key_issues_title'))
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
        plural_s = "s" if len(issues) != 1 else ""
        badge_html = (
            f'<div style="display:flex; justify-content:flex-end; margin-bottom:6px;">'
            f'<span style="background:rgba(255,95,92,0.15); color:#FF5B5C; padding:2px 10px;'
            f' border-radius:12px; font-size:0.72rem; font-weight:600;">'
            f'{t("audit_n_issues_found", n=len(issues), s=plural_s)}'
            f'</span></div>'
        )
        _glass_card(f'{badge_html}{rows_html}')
    else:
        ok_content = (
            '<div style="display:flex; align-items:center; gap:10px;">'
            '<span style="display:inline-block; width:8px; height:8px; border-radius:50%;'
            ' background:#7FB135; box-shadow:0 0 6px rgba(127,177,53,0.4);"></span>'
            f'<b style="color:rgba(255,255,255,0.75);">{t("audit_no_issues")}</b>'
            '</div>'
            f'<span style="margin-left:18px; color:rgba(255,255,255,0.4);">{t("audit_all_clear")}</span>'
        )
        _glass_card(ok_content)
    section_divider()


def _render_issue_composition(audit: dict, lang: str):
    """SECTION — ISSUE COMPOSITION CHART (overview bar chart)"""
    _styled_header(get_text('issue_composition', lang), anchor='section-issue-composition')
    _section_subtitle("Proportional breakdown of all detected data quality issues — "
        "<b style='color:rgba(255,255,255,0.65)'>missing values</b>, "
        "<b style='color:rgba(255,255,255,0.65)'>duplicates</b>, "
        "<b style='color:rgba(255,255,255,0.65)'>inconsistencies</b>, "
        "<b style='color:rgba(255,255,255,0.65)'>noise</b>, and "
        "<b style='color:rgba(255,255,255,0.65)'>low-variance columns</b> — relative to total dataset cells.")
    total_rows = audit.get("total_records", 0)
    total_cols = audit.get("attributes", 0)
    total_values = total_rows * total_cols
    duplicate_cells = audit.get("duplicates", 0) * total_cols
    low_var_cols = audit.get("low_variance_total", 0)
    low_var_cells = low_var_cols * total_rows  # cell-based count for % calc
    issue_dict = {
        get_text("compose_missing", lang):        audit.get("missing_cells", 0),
        get_text("compose_duplicates", lang):      duplicate_cells,
        get_text("compose_inconsistencies", lang): audit.get("inconsistency_total", 0),
        get_text("compose_noise_values", lang):    audit.get("noise_total", 0),
        get_text("compose_low_variance", lang):    low_var_cells,
    }

    if not issue_dict or total_values == 0:
        _skip_msg(get_text('audit_no_issue_data', lang), accent='#3B82F6')
    else:
        dup_label = get_text("compose_duplicates", lang)
        lv_label = get_text("compose_low_variance", lang)
        display_overrides = {
            dup_label: audit.get("duplicates", 0),
            lv_label: low_var_cols,
        }
        total_issues = sum(issue_dict.values())
        clean_pct = round((total_values - total_issues) / total_values * 100, 1) if total_values > 0 else 100.0
        issue_pct = round(total_issues / total_values * 100, 2) if total_values > 0 else 0.0
        _badge_bar([
            (get_text('audit_dataset_cells', lang, n=total_values), "badge-blue"),
            (get_text('audit_affected_pct', lang, pct=issue_pct, n=total_issues), "badge-red"),
            (get_text('audit_clean_pct', lang, pct=clean_pct), "badge-green"),
        ])
        st.markdown("<div class='chart-glow'>", unsafe_allow_html=True)

        # Build rich annotation labels for each bar
        missing_label = get_text("compose_missing", lang)
        noise_label = get_text("compose_noise_values", lang)
        incon_label = get_text("compose_inconsistencies", lang)
        dup_count = audit.get("duplicates", 0)
        detail_labels = {
            missing_label: f"{audit.get('missing_cells', 0):,} values",
            noise_label: f"{audit.get('noise_total', 0):,} values",
            incon_label: f"{audit.get('inconsistency_total', 0):,} values",
            dup_label: f"{dup_count:,} rows · {duplicate_cells:,} values",
            lv_label: f"{low_var_cols:,} cols · {low_var_cells:,} values",
        }

        fig = plot_issue_composition(
            issue_dict, total_values,
            display_overrides=display_overrides,
            detail_labels=detail_labels,
        )
        # Each bar click scrolls to the corresponding dedicated section below
        LABEL_TO_ANCHOR = {
            "missing":     "section-missing-values",
            "duplicat":    "section-duplicate-rows",
            "noise":       "section-noise-values",
            "inconsisten": "section-inconsistencies",
            "low-variance": "section-low-variance",
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
    _styled_header(get_text('compose_missing', lang), anchor='section-missing-values')
    _section_subtitle("Column-level breakdown of "
        "<b style='color:rgba(255,255,255,0.65)'>null</b> and "
        "<b style='color:rgba(255,255,255,0.65)'>empty cells</b> — shows which fields "
        "have gaps and how significant each gap is relative to "
        "<b style='color:rgba(255,255,255,0.65)'>total rows</b>.")
    total_rows = audit.get("total_records", 0)
    total_missing = audit.get("missing_cells", 0)
    if total_missing == 0:
        _skip_msg(get_text('audit_no_missing', lang))
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
    """SECTION — NOISE VALUES — placeholder/noise values detected."""
    _styled_header(get_text('compose_noise_values', lang), anchor='section-noise-values')
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
        _skip_msg(get_text('no_noise_found', lang))
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
    _styled_header(get_text('compose_duplicates', lang), anchor='section-duplicate-rows')
    _section_subtitle("<b style='color:rgba(255,255,255,0.65)'>Exact duplicate rows</b> "
        "detected in the dataset — rows that are "
        "<b style='color:rgba(255,255,255,0.65)'>identical across all columns</b> "
        "and should be deduplicated before modeling.")
    dup_count = audit.get("duplicates", 0)
    total_rows = audit.get("total_records", 0)
    if dup_count == 0:
        _skip_msg(get_text('audit_no_duplicates', lang))
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
        _skip_msg(f"{dup_count:,} exact duplicate rows found out of {total_rows:,} total records.", accent='#3B82F6')
    st.markdown(_BACK_TO_OVERVIEW, unsafe_allow_html=True)
    section_divider()


def _render_inconsistencies(audit: dict, lang: str):
    """SECTION — INCONSISTENCIES — categorical data quality issues."""
    _styled_header(get_text('compose_inconsistencies', lang), anchor='section-inconsistencies')
    _section_subtitle("Categorical data quality issues — "
        "<b style='color:rgba(255,255,255,0.65)'>mixed casing</b>, "
        "<b style='color:rgba(255,255,255,0.65)'>leading/trailing whitespace</b>, "
        "<b style='color:rgba(255,255,255,0.65)'>rare singletons</b>, "
        "and other formatting inconsistencies across text columns.")
    consistency_df = audit.get("consistency", pd.DataFrame())
    it = audit.get("inconsistency_total", 0)
    if it == 0 or consistency_df.empty:
        _skip_msg(get_text('all_consistent', lang))
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


def _render_low_variance(audit: dict, lang: str):
    """SECTION — LOW-VARIANCE / ZERO-SPREAD COLUMNS."""
    _styled_header("Low-Variance Columns", anchor="section-low-variance", accent="#F59E0B")
    _section_subtitle(
        "Numeric columns where a <b style='color:rgba(255,255,255,0.65)'>single dominant value</b> "
        "occupies ≥ 80% of non-null records — causing "
        "<b style='color:rgba(255,255,255,0.65)'>IQR and MAD to collapse to zero</b>, "
        "which makes statistical outlier detection impossible."
    )

    low_var_df = audit.get("low_variance", pd.DataFrame())
    if low_var_df.empty:
        _skip_msg("No low-variance columns detected — all numeric features have sufficient spread.", accent="#F59E0B")
        section_divider()
        return

    n_cols = len(low_var_df)
    _badge_bar([
        (f"{n_cols} low-variance column{'s' if n_cols != 1 else ''}", "badge-orange"),
        ("≥ 80% single-value dominance", 'style="background:rgba(245,158,11,0.12); color:#F59E0B;"'),
    ])

    # Highlight rows with extreme dominance (≥ 95%)
    def _highlight_extreme(row):
        if row["Dominant %"] >= 95:
            return [
                "background-color: rgba(245,158,11,0.18); "
                "color: rgba(245,158,11,0.7);"
            ] * len(row)
        return [""] * len(row)

    styled = low_var_df.style.apply(_highlight_extreme, axis=1)
    st.dataframe(
        styled,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Column": st.column_config.TextColumn("Column", width=140),
            "Dominant Value": st.column_config.NumberColumn("Dominant Value", width=120),
            "Dominant %": st.column_config.ProgressColumn(
                "Dominant %", min_value=0, max_value=100, format="%.1f%%", width=130,
            ),
            "Distinct Values": st.column_config.NumberColumn("Distinct", format="%d", width=80),
        },
    )

    # Info box with explanation
    st.markdown(
        '<div style="margin:4px 0 12px 0; padding:12px 16px; background:rgba(245,158,11,0.10);'
        ' border-left:3px solid rgba(245,158,11,0.5); border-radius:0 8px 8px 0;'
        ' font-size:0.78rem; color:rgba(255,255,255,0.45); line-height:1.9;">'
        '<b style="color:rgba(255,255,255,0.6);">⚠ Impact on Preprocessing</b><br>'
        '<b style="color:#F59E0B;">Outlier Detection</b> — '
        '<span style="color:rgba(255,255,255,0.35);">'
        'IQR and MAD equal zero because ≥50% of values are identical. '
        'Statistical fences cannot be computed.</span><br>'
        '<b style="color:#F59E0B;">Recommended Action</b> — '
        '<span style="color:rgba(255,255,255,0.35);">'
        'Use <b>Binary Indicator Transform</b> (value vs. non-value) '
        'or <b>Binning</b> into meaningful groups via Step 7 configuration.</span>'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown(_BACK_TO_OVERVIEW, unsafe_allow_html=True)
    section_divider()


def _render_data_summary(df: pd.DataFrame, lang: str):
    """SECTION — STATISTICAL OVERVIEW (per-column descriptive statistics)."""
    _styled_header(get_text('audit_data_summary_title', lang))
    _section_subtitle(get_text('audit_data_summary_caption', lang))
    summary_df = audit_engine.compute_data_summary(df)
    if summary_df.empty:
        _skip_msg("No data available for summary.", accent='#3B82F6')
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
    _styled_header(get_text('audit_cat_frequency_title', lang), accent='#3B82F6')
    _section_subtitle(
        "Visualize the <b style='color:rgba(255,255,255,0.65)'>value distribution</b> of categorical columns. "
        "<b style='color:rgba(255,255,255,0.65)'>Rare categories</b> (\u2264 1% frequency) are highlighted in <b style='color:#F59E0B'>amber</b>."
    )

    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    if not cat_cols:
        _skip_msg(get_text('audit_no_cat_cols', lang), accent='#3B82F6')
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
    _styled_header(get_text('risk_records_inspector', lang))
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

    # Note: audit is a dict (not a DataFrame) — save_temp_csv is not applicable


    # ── Spacing (sync with other pages) ─────────────────────────────────
    st.markdown('<div style="height:20px;"></div>', unsafe_allow_html=True)

    # --- KEY ISSUES IDENTIFIED (standalone section) ---
    _render_key_issues_summary(audit, lang)

    # --- 4-TAB LAYOUT ---
    tab_intro, tab1, tab2, tab3 = st.tabs([
        get_text('audit_tab_intro', lang),
        get_text('audit_tab_quality', lang),
        get_text('audit_tab_profiling', lang),
        get_text('audit_tab_charts', lang),
    ])
    with tab_intro:
        _render_dataset_introduction(lang)
    with tab1:
        _render_issue_composition(audit, lang)
        _render_missing_values(audit, df, lang)
        _render_noise_values(audit, lang)
        _render_duplicate_rows(audit, df, lang)
        _render_inconsistencies(audit, lang)
        _render_low_variance(audit, lang)
    with tab2:
        _render_data_summary(df, lang)
    with tab3:
        _render_category_frequency(df, lang)
        _render_risk_inspector(df, lang)

if __name__ == "__main__":
    main()