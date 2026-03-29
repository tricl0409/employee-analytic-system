"""
eda.py — Employee Data Insight Page (EDA)

Tab-based analysis dashboard with 4 tabs:
  1. Dataset & Correlations — income distribution, feature correlation, demographic breakdown
  2. Intersecting Demographics — family role, age, education, gender cross-analysis
  3. Career & Occupations — education barriers, working hours, age, gender × occupation
  4. Capital Gain & Wealth — non-salary investment income segmentation by gender
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from modules.core import data_engine
from modules.core.preprocessing_engine import PreprocessingEngine, bin_label_sort_key
from modules.ui import (
    page_header, workspace_status,
    active_file_scan_progress_bar, section_divider, metric_card,
)
from modules.ui.components import styled_alert
from modules.ui.visualizer import (
    CHART_LAYOUT, MUTED_COLOR, BRIGHT_TEXT, GRID_COLOR,
    apply_global_theme, chart_target_correlation,
)
from modules.utils.localization import get_text
from modules.utils.helpers import _ensure_workspace_active, save_temp_csv, _high_mask
from modules.utils.theme_manager import STATUS_COLORS
from modules.ui.icons import get_icon


# ==============================================================================
# CONSTANTS & CONFIG
# ==============================================================================

_C_HIGH = STATUS_COLORS["warning"]["hex"]   # Amber  — high income (>50K)

_STRIP_KEYS = {"legend", "margin"}

# Section header accent (amber — consistent with insight boxes)
_ACCENT_AMBER   = "#FF9F43"
_ACCENT_TEAL    = "#2DD4BF"
_ACCENT_EMERALD = "#10B981"
_ACCENT_VIOLET  = "#8B5CF6"

# Shared amber colorscale for heatmaps (transparent → amber gradient)
_AMBER_SCALE = [
    [0.0, "rgba(255,255,255,0.03)"],
    [0.3, "rgba(255,159,67,0.20)"],
    [0.6, "rgba(255,159,67,0.45)"],
    [1.0, "rgba(255,159,67,0.80)"],
]

# Education color palette — amber family with strong contrast between levels
_EDU_COLORS = {
    "Basic":      "rgba(160,150,140,0.40)",     # Faded warm gray — lowest
    "HS-grad":    "rgba(217,175,120,0.55)",      # Tan / light khaki
    "Some/Assoc": "rgba(245,180,60,0.68)",       # Warm gold
    "Bachelors":  "rgba(255,140,30,0.82)",       # Rich orange
    "Advanced":   "rgba(240,90,20,0.95)",        # Deep burnt orange — highest
}

# Education sort order (descending: highest level first)
_EDU_ORDER = ["Advanced", "Bachelors", "Some/Assoc", "HS-grad", "Basic"]


def _base_layout() -> dict:
    """CHART_LAYOUT minus conflict-prone keys (legend, margin)."""
    return {k: v for k, v in CHART_LAYOUT.items() if k not in _STRIP_KEYS}


# ==============================================================================
# UTILITIES
# ==============================================================================

def _resolve_cols(df: pd.DataFrame) -> dict[str, str | None]:
    """Case-insensitive column resolver."""
    def _norm(s: str) -> str:
        return s.lower().replace("_", "").replace("-", "").replace(" ", "")

    lookup = {_norm(c): c for c in df.columns}
    _ALIASES = {
        "income":        ["income", "salary", "incomelabel"],
        "age":           ["age"],
        "occupation":    ["occupation", "job"],
        "hours":         ["hoursperweek", "workinghours", "hours"],
        "sex":           ["sex", "gender"],
        "education":     ["education"],
        "education_num": ["educationnum", "education_num"],
        "marital":       ["maritalstatus", "marital"],
        "relationship":  ["relationship"],
        "workclass":     ["workclass"],
        "capital_gain":  ["capitalgain", "capgain", "capitalgains", "capital_gain"],
        "capital_loss":  ["capitalloss", "capital_loss"],
        "race":          ["race", "ethnicity"],
    }
    return {
        field: next((lookup[a] for a in aliases if a in lookup), None)
        for field, aliases in _ALIASES.items()
    }


def _apply_binning_onthefly(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply binning & mapping from session analysis_rules on a COPY.

    This gives us readable binned labels (Age → '≤25', Education → 'Bachelors')
    without mutating the original DataFrame.
    """
    rules = st.session_state.get("analysis_rules", {})
    binning_config = rules.get("binning_config", {})
    if not binning_config:
        return df
    df_binned = df.copy()
    return PreprocessingEngine.apply_binning_mapping(df_binned, binning_config)


def _insight_box(html_text: str, accent: str = _ACCENT_AMBER) -> str:
    """
    Prominent insight callout with configurable accent color.
    Auto-converts <b>text</b> inside html_text to accent-colored bold spans.
    """
    hex_val = accent.lstrip("#")
    r_val, g_val, b_val = int(hex_val[0:2], 16), int(hex_val[2:4], 16), int(hex_val[4:6], 16)
    rgb = f"{r_val},{g_val},{b_val}"
    highlighted = html_text.replace(
        "<b>", f"<b style='color:{accent};font-weight:700;'>",
    )
    return (
        f"<div style='"
        f"background:rgba({rgb},0.07);"
        f"border:1px solid rgba({rgb},0.18);"
        f"border-left:3px solid rgba({rgb},0.85);"
        f"border-radius:8px;"
        f"padding:14px 16px;"
        f"margin-top:10px;"
        f"min-height:120px;"
        f"display:flex;flex-direction:column;justify-content:center;'"
        f">"
        f"<div style='"
        f"font-size:0.67rem;font-weight:700;"
        f"color:rgba({rgb},0.65);"
        f"text-transform:uppercase;letter-spacing:1.2px;"
        f"margin-bottom:8px;"
        f"'>" + get_icon('zap', size=13, color=f'rgba({rgb},0.65)') + " Insight</div>"
        f"<div style='"
        f"font-size:0.80rem;"
        f"color:rgba(255,255,255,0.72);"
        f"line-height:1.75;"
        f"'>{highlighted}</div>"
        f"</div>"
    )


def _insight_list_box(
    bullets: list[str],
    title: str = "Key Findings",
    icon: str = "bar_chart",
    accent: str = _ACCENT_AMBER,
    flex_wrap: bool = False,
) -> str:
    """Bulleted insight callout with configurable accent, icon, and title.

    Args:
        bullets:   List of HTML strings (one per bullet point).
        title:     Header label.
        icon:      Icon name from the SVG registry.
        accent:    Hex accent color.
        flex_wrap: If True, renders bullets in a flex-wrap row (compact).

    Returns:
        HTML string ready for ``st.markdown(…, unsafe_allow_html=True)``.
    """
    hex_val = accent.lstrip("#")
    r_v, g_v, b_v = int(hex_val[0:2], 16), int(hex_val[2:4], 16), int(hex_val[4:6], 16)
    rgb = f"{r_v},{g_v},{b_v}"
    # Auto-highlight <b> tags with accent color
    colored_bullets = [
        b.replace("<b>", f"<b style='color:{accent};font-weight:700;'>")
        for b in bullets
    ]
    bullet_html = "".join(
        f"<li style='margin-bottom:10px;line-height:1.75;'>{b}</li>"
        for b in colored_bullets
    )
    flex_style = "display:flex;flex-wrap:wrap;gap:0 40px;" if flex_wrap else ""
    icon_html = get_icon(icon, size=13, color=f'rgba({rgb},0.65)')
    return (
        f"<div style='"
        f"background:rgba({rgb},0.05);"
        f"border:1px solid rgba({rgb},0.15);"
        f"border-left:3px solid rgba({rgb},0.65);"
        f"border-radius:0 12px 12px 0;"
        f"padding:18px 22px;margin-top:10px;'>"
        f"<div style='font-size:0.67rem;font-weight:700;"
        f"color:rgba({rgb},0.65);"
        f"text-transform:uppercase;letter-spacing:1.2px;"
        f"margin-bottom:10px;'>"
        f"{icon_html} {title}</div>"
        f"<ul style='"
        f"font-size:0.82rem;color:rgba(255,255,255,0.72);"
        f"padding-left:18px;margin:0;list-style-type:disc;"
        f"{flex_style}'>{bullet_html}</ul></div>"
    )

def _section_header(
    title: str,
    subtitle: str = "",
    accent: str = _ACCENT_AMBER,
    icon_name: str = "bar_chart",
) -> None:
    """Render a prominent section header with icon, title, and optional subtitle."""
    hex_val = accent.lstrip("#")
    r_val, g_val, b_val = int(hex_val[0:2], 16), int(hex_val[2:4], 16), int(hex_val[4:6], 16)
    rgb = f"{r_val},{g_val},{b_val}"
    icon_html = get_icon(icon_name, size=18, color=accent)
    subtitle_html = (
        f'<div style="font-size:0.78rem;color:rgba(255,255,255,0.45);'
        f'margin-top:4px;line-height:1.5;">{subtitle}</div>'
        if subtitle else ""
    )
    st.markdown(
        f'<div style="margin-bottom:18px;margin-top:6px;padding:16px 20px;'
        f'background:linear-gradient(135deg,rgba({rgb},0.10) 0%,rgba({rgb},0.03) 100%);'
        f'border:1px solid rgba({rgb},0.15);border-left:4px solid {accent};'
        f'border-radius:0 14px 14px 0;">'
        f'<div style="display:flex;align-items:center;gap:10px;">'
        f'{icon_html}'
        f'<span style="font-size:1.10rem;font-weight:700;'
        f'color:rgba(255,255,255,0.95);letter-spacing:-0.3px;">{title}</span>'
        f'</div>'
        f'{subtitle_html}'
        f'</div>',
        unsafe_allow_html=True,
    )


def _row_spacer(height: int = 28) -> None:
    """Add vertical spacing between chart rows."""
    st.markdown(
        f'<div style="margin-top:{height}px;"></div>',
        unsafe_allow_html=True,
    )


def _tab_summary(lines: str) -> None:
    """Render a styled insight summary block at the top of a tab."""
    st.markdown(
        f"""<div style="margin:4px 0 20px 0; padding:12px 16px;
            background:rgba(59,130,246,0.08);
            border-left:3px solid rgba(59,130,246,0.35);
            border-radius:0 8px 8px 0;
            font-size:0.78rem; color:rgba(255,255,255,0.45); line-height:1.9;">
            {lines}
        </div>""",
        unsafe_allow_html=True,
    )



# ==============================================================================
# KPI METRIC CARDS
# ==============================================================================

def _render_kpis(df: pd.DataFrame, cols: dict[str, str | None]) -> None:
    """
    6-card KPI header — computed from raw data.

    Cards: Dataset Scale | High Income % | Median Age
           Gender Ratio  | Avg Hours/Wk | Age Gap (Hi vs Std)
    """
    total = len(df)
    n_cols = len(df.columns)
    size_mb = df.memory_usage(deep=True).sum() / (1024 ** 2)

    # High income stats
    if cols["income"]:
        high_mask = _high_mask(df[cols["income"]])
        n_high = int(high_mask.sum())
        n_std = total - n_high
        pct_high = round(n_high / total * 100, 1) if total else 0
        ratio_str = f"{n_high:,} vs {n_std:,}"
    else:
        n_high, pct_high, ratio_str = 0, 0.0, "—"
        high_mask = pd.Series(False, index=df.index)

    # Age stats
    if cols["age"]:
        age_series = pd.to_numeric(df[cols["age"]], errors="coerce").dropna()
        median_age = int(age_series.median())
        q1_age = int(age_series.quantile(0.25))
        q3_age = int(age_series.quantile(0.75))
        age_range = f"IQR {q1_age} – {q3_age} yrs"
    else:
        median_age, age_range = "—", "—"

    # Gender stats
    gender_col = cols.get("sex")
    if gender_col and gender_col in df.columns:
        vc = df[gender_col].astype(str).str.strip().str.lower().value_counts()
        male_n = vc.get("male", 0)
        female_n = vc.get("female", 0)
        total_gend = male_n + female_n if (male_n + female_n) > 0 else 1
        pct_male = round(male_n / total_gend * 100, 1)
        pct_female = round(female_n / total_gend * 100, 1)
        gender_val = f"{pct_male}% ♂"
        gender_sub = f"{pct_female}% ♀  ({male_n:,} / {female_n:,})"
        gender_glow = "blue"
    else:
        gender_val = "—"
        gender_sub = "No gender column found"
        gender_glow = "blue"

    # Hours stats
    if cols["hours"] and cols["hours"] in df.columns:
        hrs_series = pd.to_numeric(df[cols["hours"]], errors="coerce").dropna()
        avg_hrs = round(hrs_series.mean(), 1)
        pct_overtime = round((hrs_series > 40).sum() / len(hrs_series) * 100, 1)
        hrs_glow = "orange" if pct_overtime > 30 else "green"
        hrs_sub = f"{pct_overtime}% work OT (>40 h/w)"
    else:
        avg_hrs, hrs_sub, hrs_glow = "—", "—", "blue"

    # Age gap: High vs Std
    if cols["income"] and cols["age"]:
        age_numeric = pd.to_numeric(df[cols["age"]], errors="coerce")
        avg_hi = age_numeric[high_mask].mean()
        avg_lo = age_numeric[~high_mask].mean()
        if pd.notna(avg_hi) and pd.notna(avg_lo):
            gap = round(avg_hi - avg_lo, 1)
            corr_val = f"{gap:+.1f} yrs"
            corr_sub = f"Hi: {avg_hi:.0f} vs Std: {avg_lo:.0f}"
            corr_glow = "orange" if abs(gap) > 5 else "green"
        else:
            corr_val, corr_sub, corr_glow = "—", "—", "blue"
    else:
        corr_val, corr_sub, corr_glow = "—", "—", "blue"

    # Render 6 cards in 1 row
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    with c1:
        metric_card("Dataset Scale", f"{total:,}",
                    f"{n_cols} cols · {size_mb:.1f} MB", glow="blue")
    with c2:
        metric_card("High Income (>50K)", f"{pct_high}%",
                    ratio_str, glow="orange" if pct_high > 30 else "blue")
    with c3:
        metric_card("Median Age", f"{median_age} yrs",
                    age_range, glow="green")
    with c4:
        metric_card("Gender Ratio", gender_val,
                    gender_sub, glow=gender_glow)
    with c5:
        metric_card("Avg Hours / Week", f"{avg_hrs} h",
                    hrs_sub, glow=hrs_glow)
    with c6:
        metric_card("Age Gap (Hi vs Std)", corr_val,
                    corr_sub, glow=corr_glow)


# ==============================================================================
# SECTION 1 — Income Distribution Donut
# ==============================================================================

def _chart_donut(df: pd.DataFrame, income_col: str) -> go.Figure:
    """Donut: High Income (>50K) vs Standard Income (≤50K)."""
    hi_mask = _high_mask(df[income_col])
    n_high = int(hi_mask.sum())
    n_std = len(df) - n_high

    fig = go.Figure(go.Pie(
        labels=[f"Standard Income (≤50K)\n{n_std:,} Individuals",
                f"High Income (>50K)\n{n_high:,} Individuals"],
        values=[n_std, n_high],
        hole=0.6,
        marker=dict(
            colors=["rgba(255,255,255,0.12)", _C_HIGH],
            line=dict(color="rgba(0,0,0,0.4)", width=2),
        ),
        textinfo="percent",
        textfont=dict(size=13, color=BRIGHT_TEXT),
        textposition="outside",
        sort=False,
        direction="clockwise",
        rotation=200,
        hovertemplate="<b>%{label}</b><br>%{value:,} employees<br>%{percent}<extra></extra>",
    ))
    fig.update_layout(
        **_base_layout(),
        height=360,
        showlegend=True,
        legend=dict(
            orientation="h", y=-0.05, x=0.5, xanchor="center",
            font=dict(size=10, color=MUTED_COLOR),
        ),
        margin=dict(l=40, r=40, t=20, b=40),
        annotations=[dict(
            text=(f"<b style='font-size:26px;color:{_C_HIGH}'>50K</b><br>"
                  f"<span style='font-size:10px;color:{MUTED_COLOR}'>Threshold</span>"),
            x=0.5, y=0.5, showarrow=False, font=dict(size=16),
        )],
    )
    return fig


def _render_section1(df: pd.DataFrame, income_col: str) -> None:
    """Render Income Distribution Donut + Insight."""
    _section_header(
        "Income Distribution",
        subtitle="Overall proportion of Standard Income (≤50K) vs High Income (>50K)",
        icon_name="eye",
    )

    hi_mask = _high_mask(df[income_col])
    n_high = int(hi_mask.sum())
    n_std = len(df) - n_high
    total = len(df)
    pct_high = round(n_high / total * 100, 1)
    pct_std = round(100 - pct_high, 1)
    ratio = round(n_std / max(n_high, 1), 1)

    st.plotly_chart(
        _chart_donut(df, income_col),
        use_container_width=True, key="ch_donut",
    )

    # Dynamic insight
    return _insight_box(
        f"Out of <b>{total:,}</b> records, <b>{pct_std}%</b> "
        f"({n_std:,}) fall into Standard Income (≤50K) while only "
        f"<b>{pct_high}%</b> ({n_high:,}) reach High Income (>50K) "
        f"— a ratio of approximately <b>{ratio}:1</b>. "
        f"This class imbalance suggests that high income is driven "
        f"by specific structural factors worth exploring.",
    )


# Minimum |r| threshold to display in the heatmap
_CORR_MIN_THRESHOLD: float = 0.05
_CORR_DISPLAY_THRESHOLD: float = 0.20

# Display-name overrides (internal col → user-friendly label)
_FEATURE_DISPLAY_NAMES: dict[str, str] = {
    "education_num": "education",
}


def _compute_correlation_scores(
    df: pd.DataFrame,
    target_col: str,
    min_threshold: float = _CORR_MIN_THRESHOLD,
) -> pd.DataFrame:
    """Compute Pearson correlation of every feature with the target column.

    Uses ``encode_for_correlation`` from ``data_engine`` to convert all
    categorical columns to numeric (domain-knowledge ordinal mapping)
    on a **temporary copy** — the original *df* is never mutated.

    Args:
        df:             DataFrame (typically the cleaned/raw data).
        target_col:     Name of the encoded target column (e.g., 'is_high_income').
        min_threshold:  Minimum |r| to include (default 0.05).

    Returns:
        DataFrame with columns: ``attribute``, ``association``
        sorted by ``|association|`` descending, filtered by threshold.
    """
    df_encoded = data_engine.encode_for_correlation(df)

    # Ensure target exists as numeric
    if target_col not in df_encoded.columns:
        return pd.DataFrame(columns=["attribute", "association"])

    numeric_df = df_encoded.select_dtypes(include=["number"])
    if target_col not in numeric_df.columns:
        return pd.DataFrame(columns=["attribute", "association"])

    # Drop target features themselves from the correlation series
    cols_to_drop = [target_col, "is_high_income", "is_standard_income"]
    corr_series = numeric_df.corr()[target_col].drop(labels=cols_to_drop, errors="ignore")
    corr_series = corr_series.dropna()

    # Filter by threshold
    mask = corr_series.abs() >= min_threshold
    corr_series = corr_series[mask]

    result_df = pd.DataFrame({
        "attribute": corr_series.index,
        "association": corr_series.values.round(3),
    })
    if not result_df.empty:
        result_df["abs_assoc"] = result_df["association"].abs()
        result_df = result_df.sort_values(
            by=["abs_assoc", "attribute"],
            ascending=[False, True],
        ).drop(columns=["abs_assoc"]).reset_index(drop=True)
    return result_df


def _chart_correlation_bar(corr_df: pd.DataFrame, target_col: str) -> go.Figure | None:
    """Horizontal bar chart: Pearson r per feature, gradient-colored by |r|.

    Reuses ``chart_target_correlation`` from the visualizer module for
    consistent styling with the Preprocessing page.
    """
    # Build a mini correlation matrix from the corr_df rows
    features = corr_df["attribute"].tolist()
    scores = corr_df["association"].tolist()
    corr_dict = {f: s for f, s in zip(features, scores)}
    corr_dict[target_col] = 1.0  # dummy self-correlation
    corr_matrix = pd.DataFrame(
        {target_col: pd.Series(corr_dict)}
    )
    return chart_target_correlation(corr_matrix, target_col=target_col)


def _render_section2(
    df: pd.DataFrame,
    income_col: str,
) -> tuple[pd.DataFrame | None, str]:
    """Render Feature Correlation bar chart (encoded data) + Insight.

    Returns:
        tuple: (Filtered corr_df_hi or None, insight_html string)
    """
    _section_header(
        "Feature Correlation with Income Class",
        subtitle="Pearson r — features encoded via domain-knowledge ordinal mapping",
        icon_name="target",
        accent=_ACCENT_TEAL,
    )

    corr_df_hi = _compute_correlation_scores(df, "is_high_income")
    corr_df_std = _compute_correlation_scores(df, "is_standard_income")

    if corr_df_hi.empty and corr_df_std.empty:
        styled_alert("Insufficient data to compute correlations.", "info")
        return None, ""

    # Filter: only features with |r| >= 20%
    corr_df_hi = corr_df_hi[corr_df_hi["association"].abs() >= _CORR_DISPLAY_THRESHOLD].reset_index(drop=True)
    corr_df_std = corr_df_std[corr_df_std["association"].abs() >= _CORR_DISPLAY_THRESHOLD].reset_index(drop=True)
    
    if corr_df_hi.empty:
        styled_alert("No features have |r| ≥ 0.20 with Income.", "info")
        return None, ""

    # Rename education_num → education for display
    corr_df_hi["attribute"] = corr_df_hi["attribute"].replace(_FEATURE_DISPLAY_NAMES)
    corr_df_std["attribute"] = corr_df_std["attribute"].replace(_FEATURE_DISPLAY_NAMES)

    tab_hi, tab_std = st.tabs(["High Income (>50K)", "Standard Income (≤50K)"])
    
    with tab_hi:
        fig_hi = _chart_correlation_bar(corr_df_hi, "is_high_income")
        if fig_hi is not None:
            # Match donut chart height (~320px)
            fig_hi.update_layout(height=320, margin=dict(t=0, b=0))
            st.plotly_chart(fig_hi, use_container_width=True, key="ch_assoc_hi")
            
    with tab_std:
        fig_std = _chart_correlation_bar(corr_df_std, "is_standard_income")
        if fig_std is not None:
            fig_std.update_layout(height=320, margin=dict(t=0, b=0))
            st.plotly_chart(fig_std, use_container_width=True, key="ch_assoc_std")

    # Dynamic insight: top-3 and count (based on High Income)
    n_features = len(corr_df_hi)
    top3 = corr_df_hi.head(3)
    top3_names = top3["attribute"].tolist()
    top3_scores = top3["association"].tolist()

    top_parts = [f"<b>{n}</b> ({s:+.3f})" for n, s in zip(top3_names, top3_scores)]
    if len(top_parts) >= 3:
        top_text = f"{top_parts[0]}, {top_parts[1]}, and {top_parts[2]}"
    else:
        top_text = ", ".join(top_parts)

    insight_html = _insight_box(
        f"<b>{n_features}</b> features show strong correlation "
        f"(|r| ≥ {_CORR_DISPLAY_THRESHOLD:.0%}) with Income. "
        f"The strongest predictors for High Income are {top_text}. "
        f"These features implicitly have the exact inverse relationship with Standard Income.",
        accent=_ACCENT_TEAL,
    )

    return corr_df_hi, insight_html


# ==============================================================================
# SECTION 3 — Family Role & Gender: Cross-Tab Heatmaps
# ==============================================================================

def _chart_crosstab_heatmap(
    df: pd.DataFrame,
    income_col: str,
    row_col: str,
    col_col: str,
    title: str,
    colorscale: list | str = "RdBu",
    fmt_pct: bool = False,
) -> go.Figure:
    """
    Annotated heatmap: High Income Rate by cross-tabulation of two categorical columns.

    Args:
        df:         DataFrame (binned).
        income_col: Income column name.
        row_col:    Column for Y-axis (rows).
        col_col:    Column for X-axis (columns).
        title:      Chart title.
        colorscale: Plotly colorscale name.

    Returns:
        Plotly Figure with annotated heatmap.
    """
    hi_mask = _high_mask(df[income_col])

    # Build cross-tab of High Income Rate
    ct = hi_mask.groupby([df[row_col].astype(str), df[col_col].astype(str)]).mean()
    ct = ct.unstack(fill_value=0)

    row_labels = ct.index.tolist()
    col_labels = ct.columns.tolist()
    z_values = ct.values.round(2)

    # Annotation text for each cell
    annotations = []
    for row_idx, row_label in enumerate(row_labels):
        for col_idx, col_label in enumerate(col_labels):
            val = z_values[row_idx][col_idx]
            annotations.append(dict(
                x=col_label,
                y=row_label,
                text=f"{val:.0%}" if fmt_pct else f"{val:.2f}",
                font=dict(
                    size=11,
                    color="rgba(255,255,255,0.9)" if val > 0.25 else "rgba(255,255,255,0.7)",
                    weight=700 if val > 0.30 else 400,
                ),
                showarrow=False,
                xref="x",
                yref="y",
            ))

    fig = go.Figure(go.Heatmap(
        z=z_values,
        x=col_labels,
        y=row_labels,
        colorscale=colorscale,
        zmin=0,
        zmax=max(z_values.max().max(), 0.01),
        showscale=True,
        colorbar=dict(
            thickness=12,
            len=0.9,
            tickfont=dict(size=9, color=MUTED_COLOR),
            outlinewidth=0,
        ),
        hovertemplate=(
            "<b>%{y}</b> × <b>%{x}</b><br>"
            + ("High Income Rate: <b>%{z:.0%}</b><extra></extra>" if fmt_pct
               else "High Income Rate: <b>%{z:.2f}</b><extra></extra>")
        ),
        xgap=2,
        ygap=2,
    ))

    fig.update_layout(
        **_base_layout(),
        height=360,
        showlegend=False,
        margin=dict(l=140, r=60, t=35, b=50),
        title=dict(
            text=title,
            font=dict(size=11, color=MUTED_COLOR),
            x=0.5, xanchor="center",
        ),
        xaxis=dict(
            title=dict(text=col_col, font=dict(color=MUTED_COLOR, size=10)),
            tickfont=dict(color=MUTED_COLOR, size=10),
            side="bottom",
        ),
        yaxis=dict(
            title=dict(text=row_col, font=dict(color=MUTED_COLOR, size=10)),
            tickfont=dict(color=MUTED_COLOR, size=10),
            autorange="reversed",
        ),
        annotations=annotations,
    )
    return apply_global_theme(fig)


def _render_section3(
    df_binned: pd.DataFrame,
    cols: dict[str, str | None],
    income_col: str,
) -> None:
    """Render cross-tab heatmaps: Relationship×Sex and Marital Status×Sex."""
    _section_header(
        "Family Role & Gender: Impact on High Income",
        subtitle="Analyzing how relationship status and marital status interact with gender to shape High Income probability",
        icon_name="briefcase",
    )

    sex_col = cols.get("sex")
    rel_col = cols.get("relationship")
    marital_col = cols.get("marital")

    if not sex_col or sex_col not in df_binned.columns:
        styled_alert("Sex column not found in dataset.", "info")
        return

    hi_mask = _high_mask(df_binned[income_col])

    col_left, col_right = st.columns(2, gap="medium")

    # ── Left: Relationship × Sex ──────────────────────────────────────────
    with col_left:
        if rel_col and rel_col in df_binned.columns:
            rate_by_rel = hi_mask.groupby(df_binned[rel_col].astype(str)).mean()
            sorted_rel = rate_by_rel.sort_values(ascending=False).index.tolist()

            fig_rel = _chart_crosstab_heatmap(
                df_binned, income_col, rel_col, sex_col,
                title="High Income Rate: Relationship × Sex",
                colorscale=_AMBER_SCALE,
                fmt_pct=True,
            )
            # Inject LGBT Rainbow flag for Male - Wife
            rainbow_shapes = []
            if "Wife" in sorted_rel:
                try:
                    # 'Female', 'Male' are the unique categories in x
                    x_cats = sorted(df_binned[sex_col].dropna().unique().astype(str).tolist())
                    if "Male" in x_cats:
                        x_idx = x_cats.index("Male")
                        y_idx = sorted_rel.index("Wife")
                        # LGBT 6 Colors: Red, Orange, Yellow, Green, Blue, Violet
                        colors = ["#FF0018", "#FFA52C", "#FFFF41", "#008018", "#0000F9", "#86007D"]
                        h = 1.0 / len(colors)
                        # Plotly default for heatmap might place y=0 at top if category order is preserved
                        # so smaller y coordinates are higher up. Red first is typically correct.
                        # Wait, if y=0 is bottom, then smaller y is bottom.
                        # We use `y0=y_idx - 0.5 + ...` which aligns exactly with cell bounds.
                        
                        # Add reversed ordered check if needed, but standard is red at top
                        # We will order them from -0.5 (top if reversed/bottom if not) 
                        # We need red at top: If Wife is visually top, visually top is index 0. Smaller y is top. 
                        for i, color in enumerate(colors):
                            rainbow_shapes.append(dict(
                                type="rect",
                                xref="x", yref="y",
                                x0=x_idx - 0.5, x1=x_idx + 0.5,
                                y0=(y_idx - 0.5) + i * h,
                                y1=(y_idx - 0.5) + (i + 1) * h,
                                fillcolor=color,
                                opacity=0.25,  # Subtle enough to let text stand out
                                layer="above",
                                line_width=0,
                            ))
                except Exception:
                    pass

            fig_rel.update_layout(
                shapes=rainbow_shapes,
                xaxis=dict(title=dict(text="")),
                yaxis=dict(
                    title=dict(text=""),
                    categoryorder="array",
                    categoryarray=sorted_rel,
                ),
            )
            fig_rel.update_traces(showscale=False)
            st.plotly_chart(fig_rel, use_container_width=True, key="ch_ct_rel_sex")

    # ── Right: Marital Status × Sex ───────────────────────────────────────
    with col_right:
        if marital_col and marital_col in df_binned.columns:
            rate_by_mar = hi_mask.groupby(df_binned[marital_col].astype(str)).mean()
            sorted_mar = rate_by_mar.sort_values(ascending=False).index.tolist()

            fig_mar = _chart_crosstab_heatmap(
                df_binned, income_col, marital_col, sex_col,
                title="High Income Rate: Marital Status × Sex",
                colorscale=_AMBER_SCALE,
                fmt_pct=True,
            )
            fig_mar.update_layout(
                xaxis=dict(title=dict(text="")),
                yaxis=dict(
                    title=dict(text=""),
                    categoryorder="array",
                    categoryarray=sorted_mar,
                ),
            )
            st.plotly_chart(fig_mar, use_container_width=True, key="ch_ct_mar_sex")

    # ── Dynamic insight (with min sample filter) ──────────────────────────
    insight_parts = []
    min_samples = 30

    if rel_col and rel_col in df_binned.columns:
        ct_rel_rate = hi_mask.groupby(
            [df_binned[rel_col].astype(str), df_binned[sex_col].astype(str)]
        ).mean()
        ct_rel_count = df_binned.groupby(
            [df_binned[rel_col].astype(str), df_binned[sex_col].astype(str)]
        ).size()
        valid_rel = {idx: rate for idx, rate in ct_rel_rate.items()
                     if ct_rel_count.get(idx, 0) >= min_samples}
        if valid_rel:
            best = max(valid_rel, key=valid_rel.get)
            best_val = round(valid_rel[best] * 100, 1)
            worst = min(valid_rel, key=valid_rel.get)
            worst_val = round(valid_rel[worst] * 100, 1)
            insight_parts.append(
                f"Among groups with ≥{min_samples} employees, "
                f"<b>{best[0]} ({best[1]})</b> achieves the highest High Income Rate "
                f"at <b>{best_val}%</b>, while <b>{worst[0]} ({worst[1]})</b> "
                f"has the lowest at <b>{worst_val}%</b>."
            )

    if marital_col and marital_col in df_binned.columns:
        ct_mar_rate = hi_mask.groupby(
            [df_binned[marital_col].astype(str), df_binned[sex_col].astype(str)]
        ).mean()
        ct_mar_count = df_binned.groupby(
            [df_binned[marital_col].astype(str), df_binned[sex_col].astype(str)]
        ).size()
        valid_mar = {idx: rate for idx, rate in ct_mar_rate.items()
                     if ct_mar_count.get(idx, 0) >= min_samples}
        if valid_mar:
            best = max(valid_mar, key=valid_mar.get)
            best_val = round(valid_mar[best] * 100, 1)
            insight_parts.append(
                f"For Marital Status, <b>{best[0]} ({best[1]})</b> "
                f"leads at <b>{best_val}%</b>."
            )

    if insight_parts:
        st.markdown(
            _insight_box(" ".join(insight_parts)),
            unsafe_allow_html=True,
        )

# ==============================================================================
# SECTION 4 — Education & Age: Cross-Tab Heatmap
# ==============================================================================

def _render_section4_edu_age(
    df_binned: pd.DataFrame,
    cols: dict[str, str | None],
    income_col: str,
) -> None:
    """Render cross-tab heatmap: Age Group × Education → High Income Rate."""
    age_col = cols.get("age")
    edu_col = cols.get("education")

    if not age_col or age_col not in df_binned.columns:
        return
    if not edu_col or edu_col not in df_binned.columns:
        return

    _section_header(
        "Education & Age : Impact on High Income",
        subtitle="Education is the strongest single predictor — but its effect compounds significantly with age and experience",
        icon_name="bar_chart",
    )

    # Pre-compute sort orders
    hi_mask = _high_mask(df_binned[income_col])

    # Y-axis: age groups sorted descending (oldest at top)
    age_labels = df_binned[age_col].astype(str).unique().tolist()
    sorted_age = sorted(age_labels, key=bin_label_sort_key, reverse=True)

    # X-axis: education sorted by overall High Income Rate descending
    rate_by_edu = hi_mask.groupby(df_binned[edu_col].astype(str)).mean()
    sorted_edu = rate_by_edu.sort_values(ascending=False).index.tolist()

    fig = _chart_crosstab_heatmap(
        df_binned, income_col, age_col, edu_col,
        title="High Income Rate: Age Group × Education",
        colorscale=_AMBER_SCALE,
        fmt_pct=True,
    )
    fig.update_layout(
        height=400,
        margin=dict(l=80, r=60, t=35, b=90),
        xaxis=dict(
            tickangle=-45,
            categoryorder="array",
            categoryarray=sorted_edu,
        ),
        yaxis=dict(
            categoryorder="array",
            categoryarray=sorted_age,
        ),
    )
    st.plotly_chart(fig, use_container_width=True, key="ch_ct_age_edu")

    # Dynamic insight (reuse hi_mask from above)
    ct = hi_mask.groupby(
        [df_binned[age_col].astype(str), df_binned[edu_col].astype(str)]
    ).mean()

    if not ct.empty:
        best_idx = ct.idxmax()
        best_val = round(ct.max() * 100, 1)
        worst_idx = ct.idxmin()
        worst_val = round(ct.min() * 100, 1)

        st.markdown(
            _insight_box(
                f"Peak earners: <b>{best_idx[0]}</b> workers with "
                f"<b>{best_idx[1]}</b> education reach <b>{best_val}%</b> "
                f"High Income Rate. In contrast, <b>{worst_idx[0]}</b> "
                f"with <b>{worst_idx[1]}</b> education only reach "
                f"<b>{worst_val}%</b> — a gap of "
                f"<b>{round(best_val - worst_val, 1)} pp</b>. "
                f"This confirms that age and education have a compounding effect on income."
            ),
            unsafe_allow_html=True,
        )


# ==============================================================================
# SECTION 5 — Age & Gender (Grouped Bar)
# ==============================================================================

def _render_section5_age_gender(
    df_binned: pd.DataFrame,
    cols: dict[str, str | None],
    income_col: str,
) -> None:
    """Render grouped horizontal bar: Age Group × Gender → High Income Rate."""
    age_col = cols.get("age")
    sex_col = cols.get("sex")

    if not age_col or age_col not in df_binned.columns:
        return
    if not sex_col or sex_col not in df_binned.columns:
        return

    _section_header(
        "Age & Gender: Impact on High Income",
        subtitle="Tracking the High Income probability across age milestones between male and female populations",
        icon_name="trending_up",
    )

    hi_mask = _high_mask(df_binned[income_col])

    # Calculate High Income Rate: Group by Age and Sex
    ct_total = pd.crosstab(df_binned[age_col].astype(str), df_binned[sex_col].astype(str).str.strip().str.title())
    ct_hi = pd.crosstab(df_binned[age_col].astype(str), df_binned[sex_col].astype(str).str.strip().str.title(), values=hi_mask, aggfunc="sum")
    rate_df = (ct_hi / ct_total.replace(0, np.nan) * 100).fillna(0)

    # Sort Age groups ascending (youngest at bottom for horizontal bar)
    sorted_age = sorted(rate_df.index, key=bin_label_sort_key)
    rate_df = rate_df.reindex(sorted_age)

    sex_colors = {"Male": "rgba(59,130,246,0.75)", "Female": "rgba(236,72,153,0.75)"}

    col_chart, col_insight = st.columns([3, 2], gap="medium")

    with col_chart:
        fig = go.Figure()

        for gender_label in rate_df.columns:
            color = sex_colors.get(gender_label, "rgba(255,159,67,0.75)")
            vals = rate_df[gender_label].round(1).values

            fig.add_trace(go.Bar(
                y=rate_df.index.tolist(),
                x=vals,
                name=gender_label,
                orientation="h",
                marker=dict(color=color),
                text=[f"{v:.1f}%" for v in vals],
                textposition="outside",
                textfont=dict(size=9, color=MUTED_COLOR),
                cliponaxis=False,
                hovertemplate=f"<b>{gender_label}</b><br>Age: <b>%{{y}}</b><br>High Income Rate: <b>%{{x:.1f}}%</b><extra></extra>",
            ))

        fig.update_layout(
            **_base_layout(),
            height=max(300, len(rate_df) * 45),
            barmode="group",
            bargap=0.20,
            bargroupgap=0.08,
            showlegend=True,
            legend=dict(
                orientation="h", y=-0.18, x=0.5, xanchor="center",
                font=dict(size=10, color=MUTED_COLOR),
            ),
            margin=dict(l=90, r=60, t=20, b=50),
            xaxis=dict(
                title=dict(text="High Income Rate (%)", font=dict(color=MUTED_COLOR, size=10)),
                tickfont=dict(color=MUTED_COLOR, size=9),
                gridcolor=GRID_COLOR,
            ),
            yaxis=dict(tickfont=dict(color=MUTED_COLOR, size=10)),
        )
        st.plotly_chart(apply_global_theme(fig), use_container_width=True, key="ch_age_sex_bar")

    with col_insight:
        st.markdown("<div style='margin-top:20px;'></div>", unsafe_allow_html=True)
        insight_parts = []
        for gender_label in rate_df.columns:
            if rate_df[gender_label].sum() > 0:
                peak_age = rate_df[gender_label].idxmax()
                peak_rate = rate_df[gender_label].max()
                insight_parts.append(f"<b>{gender_label}s</b> peak at <b>{peak_age}</b> ({peak_rate:.1f}%).")

        # Compute gender gap at peak
        gap_series = rate_df.get("Male", pd.Series(dtype=float)) - rate_df.get("Female", pd.Series(dtype=float))
        max_gap_age = gap_series.idxmax() if not gap_series.empty else "N/A"
        max_gap_val = round(gap_series.max(), 1) if not gap_series.empty else 0

        st.markdown(
            _insight_box(
                " ".join(insight_parts) +
                f" The widest gender gap is at <b>{max_gap_age}</b> ({max_gap_val} pp), "
                "suggesting that career progression and earning trajectory diverge "
                "most significantly during this life stage."
            ),
            unsafe_allow_html=True,
        )


# ==============================================================================
# SECTION 6 — Education & Gender (Grouped Bar)
# ==============================================================================

def _render_section6_edu_gender(
    df_binned: pd.DataFrame,
    cols: dict[str, str | None],
    income_col: str,
) -> None:
    """Render grouped horizontal bar: Education × Gender → High Income Rate."""
    edu_col = cols.get("education")
    sex_col = cols.get("sex")

    if not edu_col or edu_col not in df_binned.columns:
        return
    if not sex_col or sex_col not in df_binned.columns:
        return

    _section_header(
        "Education & Gender: Impact on High Income",
        subtitle="Analyzing the High Income probability across education levels between male and female populations",
        icon_name="book_open",
    )

    hi_mask = _high_mask(df_binned[income_col])

    # Calculate High Income Rate: Group by Education and Sex
    ct_total = pd.crosstab(df_binned[edu_col].astype(str), df_binned[sex_col].astype(str).str.strip().str.title())
    ct_hi = pd.crosstab(df_binned[edu_col].astype(str), df_binned[sex_col].astype(str).str.strip().str.title(), values=hi_mask, aggfunc="sum")
    rate_df = (ct_hi / ct_total.replace(0, np.nan) * 100).fillna(0)

    # Sort Education — Basic at bottom, Advanced at top (for horizontal bar)
    sorted_edu = [e for e in reversed(_EDU_ORDER) if e in rate_df.index]
    remaining = [e for e in rate_df.index if e not in sorted_edu]
    sorted_edu = remaining + sorted_edu
    rate_df = rate_df.reindex(sorted_edu)

    sex_colors = {"Male": "rgba(59,130,246,0.75)", "Female": "rgba(236,72,153,0.75)"}

    col_chart, col_insight = st.columns([3, 2], gap="medium")

    with col_chart:
        fig = go.Figure()

        for gender_label in rate_df.columns:
            color = sex_colors.get(gender_label, "rgba(255,159,67,0.75)")
            vals = rate_df[gender_label].round(1).values

            fig.add_trace(go.Bar(
                y=rate_df.index.tolist(),
                x=vals,
                name=gender_label,
                orientation="h",
                marker=dict(color=color),
                text=[f"{v:.1f}%" for v in vals],
                textposition="outside",
                textfont=dict(size=9, color=MUTED_COLOR),
                cliponaxis=False,
                hovertemplate=f"<b>{gender_label}</b><br>Education: <b>%{{y}}</b><br>High Income Rate: <b>%{{x:.1f}}%</b><extra></extra>",
            ))

        fig.update_layout(
            **_base_layout(),
            height=max(300, len(rate_df) * 45),
            barmode="group",
            bargap=0.20,
            bargroupgap=0.08,
            showlegend=True,
            legend=dict(
                orientation="h", y=-0.18, x=0.5, xanchor="center",
                font=dict(size=10, color=MUTED_COLOR),
            ),
            margin=dict(l=120, r=60, t=20, b=50),
            xaxis=dict(
                title=dict(text="High Income Rate (%)", font=dict(color=MUTED_COLOR, size=10)),
                tickfont=dict(color=MUTED_COLOR, size=9),
                gridcolor=GRID_COLOR,
            ),
            yaxis=dict(tickfont=dict(color=MUTED_COLOR, size=10)),
        )
        st.plotly_chart(apply_global_theme(fig), use_container_width=True, key="ch_edu_sex_bar")

    with col_insight:
        st.markdown("<div style='margin-top:20px;'></div>", unsafe_allow_html=True)
        insight_parts = []
        for gender_label in rate_df.columns:
            if rate_df[gender_label].sum() > 0:
                peak_edu = rate_df[gender_label].idxmax()
                peak_rate = rate_df[gender_label].max()
                insight_parts.append(f"<b>{gender_label}s</b> peak at <b>{peak_edu}</b> ({peak_rate:.1f}%).")

        # Compute gender gap at peak education level
        gap_series = rate_df.get("Male", pd.Series(dtype=float)) - rate_df.get("Female", pd.Series(dtype=float))
        max_gap_edu = gap_series.idxmax() if not gap_series.empty else "N/A"
        max_gap_val = round(gap_series.max(), 1) if not gap_series.empty else 0

        st.markdown(
            _insight_box(
                " ".join(insight_parts) +
                f" The widest gender gap appears at <b>{max_gap_edu}</b> level ({max_gap_val} pp). "
                "Higher education increases earning potential for both groups, "
                "but the return on education still varies significantly by gender."
            ),
            unsafe_allow_html=True,
        )


# ==============================================================================
# SECTION 7 — Demographic Breakdown of High Income
# ==============================================================================

def _chart_hbar_rate(
    df: pd.DataFrame,
    income_col: str,
    group_col: str,
    title: str,
    top_n: int = 0,
    bar_accent: str = "255,159,67",
) -> go.Figure:
    """
    Horizontal bar: high-income rate by category.

    Args:
        top_n:       If > 0, show only top N categories by count. 0 = show all.
        bar_accent:  RGB string (e.g. '239,68,68') for bar gradient color.
    """
    hi_mask = _high_mask(df[income_col])
    rate = hi_mask.groupby(df[group_col].astype(str)).mean()

    if top_n > 0:
        top_cats = df[group_col].value_counts().head(top_n).index
        rate = rate[rate.index.isin(top_cats)]

    rate = rate.sort_values(ascending=True)
    rate_pct = (rate * 100).round(1)
    mx = max(rate_pct.max(), 0.01)
    colors = [f"rgba({bar_accent},{0.3 + 0.7 * v / mx:.2f})" for v in rate_pct.values]

    fig = go.Figure(go.Bar(
        y=rate_pct.index.tolist(),
        x=rate_pct.values,
        orientation="h",
        marker=dict(color=colors, line=dict(width=0)),
        text=[f"{v:.1f}%" for v in rate_pct.values],
        textposition="outside",
        textfont=dict(color=MUTED_COLOR, size=10),
        cliponaxis=False,
        hovertemplate="<b>%{y}</b><br>High Income Rate: <b>%{x:.1f}%</b><extra></extra>",
    ))

    fig.update_layout(
        **_base_layout(),
        height=280,
        showlegend=False,
        margin=dict(l=130, r=50, t=30, b=30),
        title=dict(
            text=title,
            font=dict(size=11, color=MUTED_COLOR),
            x=0.5, xanchor="center",
        ),
        xaxis=dict(
            title=dict(text="High Income Rate (%)", font=dict(color=MUTED_COLOR, size=10)),
            tickfont=dict(color=MUTED_COLOR, size=9),
            gridcolor=GRID_COLOR,
            range=[0, min(mx * 1.35, 100)],
        ),
        yaxis=dict(
            tickfont=dict(color=MUTED_COLOR, size=10),
        ),
    )
    return apply_global_theme(fig)


def _compute_feature_insight(
    df: pd.DataFrame,
    feature_col: str,
    income_col: str,
    assoc_score: float,
) -> str:
    """
    Compute a single dynamic insight bullet for one feature.

    Args:
        df:            DataFrame (binned).
        feature_col:   Column name of the feature.
        income_col:    Column name of income.
        assoc_score:   Cramér's V score from association computation.

    Returns:
        HTML string for one bullet point.
    """
    hi_mask = _high_mask(df[income_col])
    rate = hi_mask.groupby(df[feature_col].astype(str)).mean()

    top_cat = rate.idxmax()
    top_pct = round(rate.max() * 100, 1)
    bot_cat = rate.idxmin()
    bot_pct = round(rate.min() * 100, 1)
    gap_pp = round(top_pct - bot_pct, 1)

    return (
        f"<b>{feature_col}</b> (r = {assoc_score:.3f}): "
        f"<b>{top_cat}</b> has the highest High Income Rate at <b>{top_pct}%</b>, "
        f"vs <b>{bot_cat}</b> at <b>{bot_pct}%</b> "
        f"— a <b>{gap_pp} pp</b> gap."
    )


def _render_section7_breakdown(
    df_raw: pd.DataFrame,
    df_binned: pd.DataFrame,
    income_col: str,
    corr_df: pd.DataFrame | None = None,
) -> None:
    """
    Render Demographic Breakdown of High Income.

    Groups features into tiers by correlation strength (|r|):
      - Strong Impact   : |r| >= 0.40
      - Moderate Impact : 0.30 <= |r| < 0.40
      - Notable Impact  : 0.20 <= |r| < 0.30
    Each tier gets its own sub-header, chart grid, and insight box.
    """
    # Use pre-computed corr_df if available, otherwise compute fresh
    if corr_df is None or corr_df.empty:
        assoc_df = _compute_correlation_scores(df_raw, income_col)
        assoc_df = assoc_df[assoc_df["association"].abs() >= _CORR_DISPLAY_THRESHOLD].reset_index(drop=True)
    else:
        assoc_df = corr_df.copy()

    if len(assoc_df) < 1:
        styled_alert("Insufficient data for demographic breakdown.", "info")
        return

    # education_num fallback: use 'education' column for breakdown charts
    assoc_df["attribute"] = assoc_df["attribute"].apply(
        lambda f: "education" if f == "education_num" and "education" in df_binned.columns else f
    )

    # ── Main section header ───────────────────────────────────────────
    _section_header(
        "Demographic Breakdown of High Income",
        subtitle="High Income Rate across strongly associated features, grouped by correlation strength",
        icon_name="zap",
    )

    # ── Tier definitions ──────────────────────────────────────────────
    _TIERS = [
        {
            "label": "Strong Impact",
            "range": "r ≥ 0.40",
            "min": 0.40, "max": 1.01,
            "accent": "#EF4444",
            "icon": "alert_triangle",
        },
        {
            "label": "Moderate Impact",
            "range": "0.30 ≤ r < 0.40",
            "min": 0.30, "max": 0.40,
            "accent": "#F59E0B",
            "icon": "trending_up",
        },
        {
            "label": "Notable Impact",
            "range": "0.20 ≤ r < 0.30",
            "min": 0.20, "max": 0.30,
            "accent": "#3B82F6",
            "icon": "bar_chart",
        },
    ]

    chart_idx = 0  # global chart key counter

    for tier in _TIERS:
        tier_mask = (assoc_df["association"].abs() >= tier["min"]) & (assoc_df["association"].abs() < tier["max"])
        tier_df = assoc_df[tier_mask].reset_index(drop=True)

        if tier_df.empty:
            continue

        tier_features = tier_df["attribute"].tolist()
        tier_scores = tier_df["association"].tolist()

        feat_tags = ", ".join(
            f"<b style='color:{tier['accent']}'>{f}</b> ({s:+.3f})"
            for f, s in zip(tier_features, tier_scores)
        )

        # ── Tier sub-header ───────────────────────────────────────────
        hex_val = tier["accent"].lstrip("#")
        r_val, g_val, b_val = int(hex_val[0:2], 16), int(hex_val[2:4], 16), int(hex_val[4:6], 16)
        rgb = f"{r_val},{g_val},{b_val}"
        icon_html = get_icon(tier["icon"], size=14, color=tier["accent"])

        st.markdown(
            f'<div style="margin:20px 0 12px 0;padding:10px 16px;'
            f'background:rgba({rgb},0.06);'
            f'border-left:3px solid rgba({rgb},0.5);'
            f'border-radius:0 8px 8px 0;">'
            f'<div style="display:flex;align-items:center;gap:8px;">'
            f'{icon_html}'
            f'<span style="font-size:0.92rem;font-weight:700;'
            f'color:rgba(255,255,255,0.88);">{tier["label"]}</span>'
            f'<span style="font-size:0.72rem;color:rgba(255,255,255,0.35);'
            f'margin-left:4px;">({tier["range"]})</span>'
            f'</div>'
            f'<div style="font-size:0.76rem;color:rgba(255,255,255,0.40);'
            f'margin-top:4px;line-height:1.6;">{feat_tags}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        # ── Charts + Insight side by side ─────────────────────────────
        col_charts, col_insights = st.columns([3, 2], gap="medium")

        with col_charts:
            for i in range(0, len(tier_features), 2):
                c1, c2 = st.columns(2)
                for j, col_slot in enumerate([c1, c2]):
                    feat_idx = i + j
                    if feat_idx < len(tier_features):
                        feat = tier_features[feat_idx]
                        with col_slot:
                            if feat in df_binned.columns:
                                # Convert tier accent hex to RGB for bar coloring
                                th = tier["accent"].lstrip("#")
                                tier_rgb = f"{int(th[0:2],16)},{int(th[2:4],16)},{int(th[4:6],16)}"
                                fig = _chart_hbar_rate(
                                    df_binned, income_col, feat,
                                    title=f"High Income Rate by {feat}",
                                    bar_accent=tier_rgb,
                                )
                                st.plotly_chart(
                                    fig, use_container_width=True,
                                    key=f"ch_s4_{chart_idx}",
                                )
                        chart_idx += 1

        with col_insights:
            bullets = []
            for feat, score in zip(tier_features, tier_scores):
                if feat in df_binned.columns:
                    bullets.append(
                        _compute_feature_insight(
                            df_binned, feat, income_col, score,
                        )
                    )

            if bullets:
                st.markdown(
                    _insight_list_box(
                        bullets,
                        title=f"{tier['label']} — Key Findings",
                        icon=tier["icon"],
                    ),
                    unsafe_allow_html=True,
                )
# SECTION 8 — Education & Occupation (100% Stacked Bar)
# ==============================================================================

def _render_section8_edu_occ(
    df_binned: pd.DataFrame,
    cols: dict[str, str | None],
) -> None:
    """100% stacked horizontal bar: Education group composition per Occupation."""
    _section_header(
        "Education & Occupation: Impact on High Income",
        subtitle="Proportion of each education level within occupation categories — revealing the education barrier for different career paths",
        icon_name="briefcase",
    )

    occ_col = cols.get("occupation")
    edu_col = cols.get("education")
    if not occ_col or occ_col not in df_binned.columns:
        styled_alert("No occupation column found.", "info")
        return
    if not edu_col or edu_col not in df_binned.columns:
        styled_alert("No education column found.", "info")
        return

    ct = pd.crosstab(
        df_binned[occ_col].astype(str),
        df_binned[edu_col].astype(str),
        normalize="index",
    ) * 100

    # Sort education columns in descending order (highest level first)
    sorted_edu_cols = [e for e in _EDU_ORDER if e in ct.columns]
    remaining = [c for c in ct.columns if c not in sorted_edu_cols]
    sorted_edu_cols += remaining
    ct = ct[sorted_edu_cols]

    # Sort occupations (rows) by highest-level education % descending
    sort_keys = [col for col in sorted_edu_cols if col in ct.columns]
    ct = ct.sort_values(by=sort_keys, ascending=True)

    fig = go.Figure()
    for edu_group in ct.columns:
        color = _EDU_COLORS.get(edu_group, "rgba(148,163,184,0.5)")
        fig.add_trace(go.Bar(
            y=ct.index.tolist(),
            x=ct[edu_group].round(1).values,
            name=edu_group,
            orientation="h",
            marker=dict(color=color),
            text=[f"{v:.0f}%" if v >= 5 else "" for v in ct[edu_group].values],
            textposition="inside",
            textfont=dict(size=9, color=BRIGHT_TEXT),
            hovertemplate=f"<b>%{{y}}</b><br>{edu_group}: <b>%{{x:.1f}}%</b><extra></extra>",
        ))

    fig.update_layout(
        **_base_layout(),
        height=350,
        barmode="stack",
        margin=dict(l=150, r=30, t=20, b=70),
        legend=dict(
            orientation="h", y=-0.22, x=0.5, xanchor="center",
            font=dict(size=10, color=MUTED_COLOR),
        ),
        xaxis=dict(
            title=dict(text="% within Occupation", font=dict(color=MUTED_COLOR, size=10)),
            tickfont=dict(color=MUTED_COLOR, size=9),
            range=[0, 100],
        ),
        yaxis=dict(tickfont=dict(color=MUTED_COLOR, size=10)),
    )
    st.plotly_chart(apply_global_theme(fig), use_container_width=True, key="ch_occ_edu")

    # Dynamic insight: find occ with highest & lowest higher-education %
    higher_edu_cols = [c for c in ["Advanced", "Bachelors"] if c in ct.columns]
    if higher_edu_cols:
        higher_pct = ct[higher_edu_cols].sum(axis=1)
        top_occ = higher_pct.idxmax()
        top_val = round(higher_pct.max(), 1)
        low_occ = higher_pct.idxmin()
        low_val = round(higher_pct.min(), 1)

        st.markdown(
            _insight_box(
                f"<b>{top_occ}</b> leads with <b>{top_val}%</b> of workers "
                f"holding a Bachelor's degree or higher, while <b>{low_occ}</b> "
                f"has only <b>{low_val}%</b>. This <b>{round(top_val - low_val, 1)} pp</b> "
                f"gap highlights a significant education barrier between occupational tiers."
            ),
            unsafe_allow_html=True,
        )
# SECTION 9 — Occupation & Working Hours (Bubble Chart)
# ==============================================================================

def _render_section9_occ_hours(
    df_binned: pd.DataFrame,
    cols: dict[str, str | None],
    income_col: str,
) -> None:
    """Bubble chart: Occupation × Hours bins, size = count, color = High Income Rate."""
    _section_header(
        "Occupation & Working Hours: Impact on High Income",
        subtitle=(
            "Bubble size = employee count, color intensity = High Income Rate — "
            "revealing which occupation-hours combinations yield the highest earning potential"
        ),
        icon_name="clock",
    )

    occ_col = cols.get("occupation")
    hours_col = cols.get("hours")
    if not occ_col or occ_col not in df_binned.columns:
        styled_alert("No occupation column found.", "info")
        return
    if not hours_col or hours_col not in df_binned.columns:
        styled_alert("No hours column found.", "info")
        return

    occ = df_binned[occ_col].astype(str)
    hrs_binned = df_binned[hours_col].astype(str)
    hi_mask = _high_mask(df_binned[income_col])

    # Count & High Income Rate per (occ, hours) cell
    ct_count = pd.crosstab(occ, hrs_binned)
    ct_hi = pd.crosstab(occ, hrs_binned, values=hi_mask, aggfunc="sum")
    ct_rate = (ct_hi / ct_count.replace(0, np.nan)).fillna(0)

    # Sort X-axis (hours) in logical ascending order
    sorted_hours = sorted(ct_count.columns, key=bin_label_sort_key)
    ct_count = ct_count.reindex(columns=sorted_hours, fill_value=0)
    ct_rate = ct_rate.reindex(columns=sorted_hours, fill_value=0)

    # Sort occupations by total working hours descending (ascending for Plotly bottom-to-top)
    # Compute weighted total hours per occupation
    hrs_weights = {col: bin_label_sort_key(col) for col in sorted_hours}
    total_hrs = ct_count.apply(lambda row: sum(row[c] * hrs_weights.get(c, 0) for c in row.index), axis=1)
    ct_count = ct_count.loc[total_hrs.sort_values(ascending=True).index]
    ct_rate = ct_rate.reindex(ct_count.index)

    max_count = ct_count.values.max() if ct_count.values.max() > 0 else 1
    min_bubble_for_text = max_count * 0.03  # Hide text for bubbles < 3% of max

    # Flatten to arrays for single scatter trace
    x_vals, y_vals, sizes, texts, hovers, colors = [], [], [], [], [], []
    for occ_name in ct_count.index:
        for hrs_label in ct_count.columns:
            count = ct_count.loc[occ_name, hrs_label]
            if count == 0:
                continue
            rate = ct_rate.loc[occ_name, hrs_label]
            x_vals.append(hrs_label)
            y_vals.append(occ_name)
            sizes.append(max(8, count / max_count * 55))
            texts.append(f"{count:,}" if count >= min_bubble_for_text else "")
            colors.append(rate)
            hovers.append(
                f"<b>{occ_name}</b><br>"
                f"Hours: <b>{hrs_label}</b><br>"
                f"Count: <b>{count:,}</b><br>"
                f"High Income Rate: <b>{rate:.1%}</b>"
                "<extra></extra>"
            )

    fig = go.Figure(go.Scatter(
        x=x_vals,
        y=y_vals,
        mode="markers+text",
        marker=dict(
            size=sizes,
            color=colors,
            colorscale=[
                [0.0, "rgba(255,159,67,0.15)"],
                [0.3, "rgba(255,159,67,0.35)"],
                [0.6, "rgba(255,159,67,0.60)"],
                [1.0, "rgba(255,159,67,0.90)"],
            ],
            cmin=0,
            cmax=max(max(colors), 0.01) if colors else 1,
            colorbar=dict(
                title=dict(
                    text="High Income Rate",
                    font=dict(size=9, color=MUTED_COLOR),
                ),
                tickfont=dict(size=8, color=MUTED_COLOR),
                tickformat=".0%",
                thickness=10,
                len=0.6,
                outlinewidth=0,
            ),
            line=dict(color="rgba(255,159,67,0.3)", width=1),
            sizemode="diameter",
        ),
        text=texts,
        textposition="middle center",
        textfont=dict(size=8, color=BRIGHT_TEXT),
        hovertemplate=hovers,
        showlegend=False,
    ))

    fig.update_layout(
        **_base_layout(),
        height=450,
        margin=dict(l=150, r=60, t=20, b=60),
        xaxis=dict(
            title=dict(text="Working Hours Group", font=dict(color=MUTED_COLOR, size=10)),
            tickfont=dict(color=MUTED_COLOR, size=9),
            categoryorder="array",
            categoryarray=sorted_hours,
        ),
        yaxis=dict(tickfont=dict(color=MUTED_COLOR, size=10)),
    )
    st.plotly_chart(apply_global_theme(fig), use_container_width=True, key="ch_occ_hours_bubble")

    # ── Dynamic insight ───────────────────────────────────────────────────
    # Find occupation with highest median hours
    hrs_numeric = pd.to_numeric(
        df_binned[hours_col].astype(str).str.extract(r"(\d+)", expand=False),
        errors="coerce",
    )
    median_hrs_by_occ = hrs_numeric.groupby(occ).median()
    top_hrs_occ = median_hrs_by_occ.idxmax()
    top_hrs_val = median_hrs_by_occ.max()

    # Find (occ, hours) combo with highest High Income Rate (min 30 samples)
    best_rate, best_combo = 0.0, ("", "")
    for occ_name in ct_count.index:
        for hrs_label in ct_count.columns:
            cnt = ct_count.loc[occ_name, hrs_label]
            rate_val = ct_rate.loc[occ_name, hrs_label]
            if cnt >= 30 and rate_val > best_rate:
                best_rate = rate_val
                best_combo = (occ_name, hrs_label)

    insight_parts = []
    if top_hrs_occ:
        insight_parts.append(
            f"<b>{top_hrs_occ}</b> workers log the longest hours "
            f"(median ~<b>{top_hrs_val:.0f}</b>h/week)."
        )
    if best_rate > 0:
        insight_parts.append(
            f"The highest-earning combination is <b>{best_combo[0]}</b> "
            f"at <b>{best_combo[1]}</b> hours/week, achieving "
            f"<b>{best_rate:.1%}</b> High Income Rate."
        )

    if insight_parts:
        st.markdown(
            _insight_box(" ".join(insight_parts)),
            unsafe_allow_html=True,
        )


# ==============================================================================
# SECTION 10 — Occupation & Age (Heatmap)
# ==============================================================================

def _render_section10_occ_age(
    df_binned: pd.DataFrame,
    cols: dict[str, str | None],
    income_col: str,
) -> None:
    """Cross-tab heatmap: Age Group × Occupation → High Income Rate (from df_binned)."""
    _section_header(
        "Occupation & Age: Impact on High Income",
        subtitle="Two-dimensional view of how career experience (age) and occupational tier combine to determine High Income likelihood",
        icon_name="target",
    )

    age_col = cols.get("age")
    occ_col = cols.get("occupation")
    if not age_col or age_col not in df_binned.columns:
        styled_alert("No age column found.", "info")
        return
    if not occ_col or occ_col not in df_binned.columns:
        styled_alert("No occupation column found.", "info")
        return

    # Pre-compute sort orders
    hi_mask = _high_mask(df_binned[income_col])

    # Y-axis: age groups sorted descending (oldest at top → reversed for Plotly)
    age_labels = df_binned[age_col].astype(str).unique().tolist()

    sorted_age = sorted(age_labels, key=bin_label_sort_key, reverse=True)  # oldest at top

    # X-axis: occupations sorted by overall High Income Rate descending
    rate_by_occ = hi_mask.groupby(df_binned[occ_col].astype(str)).mean()
    sorted_occ = rate_by_occ.sort_values(ascending=False).index.tolist()

    fig = _chart_crosstab_heatmap(
        df_binned, income_col, age_col, occ_col,
        title="High Income Rate: Age Group × Occupation",
        colorscale=_AMBER_SCALE,
        fmt_pct=True,
    )
    fig.update_layout(
        height=350,
        margin=dict(l=80, r=60, t=35, b=90),
        xaxis=dict(
            tickangle=-35,
            categoryorder="array",
            categoryarray=sorted_occ,
        ),
        yaxis=dict(
            categoryorder="array",
            categoryarray=sorted_age,
        ),
    )
    st.plotly_chart(fig, use_container_width=True, key="ch_age_occ_heatmap")

    # Dynamic insight — filter cells with >= 30 samples to avoid noise
    ct_count = pd.crosstab(
        df_binned[age_col].astype(str), df_binned[occ_col].astype(str),
    )
    ct_rate = hi_mask.groupby(
        [df_binned[age_col].astype(str), df_binned[occ_col].astype(str)]
    ).mean()

    # Filter for statistically meaningful cells
    valid_cells = {idx: rate for idx, rate in ct_rate.items()
                   if ct_count.loc[idx[0], idx[1]] >= 30}

    if valid_cells:
        best_cell = max(valid_cells, key=valid_cells.get)
        best_val = round(valid_cells[best_cell] * 100, 1)
        worst_cell = min(valid_cells, key=valid_cells.get)
        worst_val = round(valid_cells[worst_cell] * 100, 1)

        st.markdown(
            _insight_box(
                f"Peak earners: <b>{best_cell[0]}</b> in <b>{best_cell[1]}</b> "
                f"roles reach <b>{best_val}%</b> High Income Rate (≥30 sample filter). "
                f"The lowest is <b>{worst_cell[0]}</b> in <b>{worst_cell[1]}</b> "
                f"at just <b>{worst_val}%</b> — a <b>{round(best_val - worst_val, 1)} pp</b> "
                f"gap demonstrating the compounding influence of career stage and occupational tier."
            ),
            unsafe_allow_html=True,
        )


# ==============================================================================
# SECTION 11 — Occupation & Sex (Grouped Bar)
# ==============================================================================

def _render_section11_occ_sex(
    df: pd.DataFrame,
    df_binned: pd.DataFrame,
    cols: dict[str, str | None],
    income_col: str,
) -> None:
    """Grouped horizontal bar: High Income rate by Occupation, colored by Sex."""
    _section_header(
        "Occupation & Sex: Impact on High Income",
        subtitle="Occupation-level decomposition of the gender income gap — identifying where disparity is largest and smallest",
        icon_name="users",
    )

    occ_col = cols.get("occupation")
    sex_col = cols.get("sex")
    if not occ_col or occ_col not in df_binned.columns:
        styled_alert("No occupation column found.", "info")
        return
    if not sex_col or sex_col not in df.columns:
        styled_alert("No sex/gender column found.", "info")
        return

    hi_mask = _high_mask(df[income_col])
    occ = df_binned[occ_col].astype(str)
    sex = df[sex_col].astype(str).str.strip().str.lower()

    rate_male = hi_mask[sex == "male"].groupby(occ[sex == "male"]).mean()
    rate_female = hi_mask[sex == "female"].groupby(occ[sex == "female"]).mean()

    all_occs = sorted(set(rate_male.index) | set(rate_female.index))
    rate_male = rate_male.reindex(all_occs, fill_value=0)
    rate_female = rate_female.reindex(all_occs, fill_value=0)

    # Sort by average of both genders
    sort_order = ((rate_male + rate_female) / 2).sort_values(ascending=True).index
    rate_male = rate_male[sort_order]
    rate_female = rate_female[sort_order]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=rate_male.index.tolist(),
        x=(rate_male.values * 100).round(1),
        name="Male",
        orientation="h",
        marker=dict(color="rgba(59,130,246,0.75)"),
        text=[f"{v:.1f}%" for v in rate_male.values * 100],
        textposition="outside",
        textfont=dict(size=9, color=MUTED_COLOR),
        cliponaxis=False,
        hovertemplate="<b>%{y}</b> (Male)<br>%{x:.1f}% High Income<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        y=rate_female.index.tolist(),
        x=(rate_female.values * 100).round(1),
        name="Female",
        orientation="h",
        marker=dict(color="rgba(236,72,153,0.75)"),
        text=[f"{v:.1f}%" for v in rate_female.values * 100],
        textposition="outside",
        textfont=dict(size=9, color=MUTED_COLOR),
        cliponaxis=False,
        hovertemplate="<b>%{y}</b> (Female)<br>%{x:.1f}% High Income<extra></extra>",
    ))

    fig.update_layout(
        **_base_layout(),
        height=max(360, len(all_occs) * 50),
        barmode="group",
        bargap=0.25,
        bargroupgap=0.08,
        margin=dict(l=150, r=70, t=20, b=70),
        legend=dict(
            orientation="h", y=-0.15, x=0.5, xanchor="center",
            font=dict(size=10, color=MUTED_COLOR),
        ),
        xaxis=dict(
            title=dict(text="High Income Rate (%)", font=dict(color=MUTED_COLOR, size=10)),
            tickfont=dict(color=MUTED_COLOR, size=9),
            gridcolor=GRID_COLOR,
        ),
        yaxis=dict(tickfont=dict(color=MUTED_COLOR, size=10)),
    )
    st.plotly_chart(apply_global_theme(fig), use_container_width=True, key="ch_sex_occ")

    # Gender gap insight
    avg_male = rate_male.mean() * 100
    avg_female = rate_female.mean() * 100
    gap = round(avg_male - avg_female, 1)
    # Identify occupation with largest & smallest gap
    occ_gap = (rate_male - rate_female) * 100
    max_gap_occ = occ_gap.idxmax()
    max_gap_val = round(occ_gap.max(), 1)
    min_gap_occ = occ_gap.idxmin()
    min_gap_val = round(occ_gap.min(), 1)
    st.markdown(
        _insight_box(
            f"The average High Income rate is <b>{avg_male:.1f}%</b> for men vs "
            f"<b>{avg_female:.1f}%</b> for women — a <b>{gap:.1f} pp</b> overall gap. "
            f"The widest disparity is in <b>{max_gap_occ}</b> ({max_gap_val} pp), "
            f"while <b>{min_gap_occ}</b> shows the narrowest gap ({min_gap_val} pp). "
            f"This occupation-level view reveals that gender inequality is not uniform but "
            f"varies significantly by career category."
        ),
        unsafe_allow_html=True,
    )


# ==============================================================================
# SEGMENT COLORS — shared by Section 12
# ==============================================================================

_SEGMENT_COLORS = {
    "High & Has CapGain": "rgba(250,175,30,0.90)",    # Gold — best segment
    "High & No CapGain": "rgba(255,140,40,0.72)",     # Strong amber
    "Std & Has CapGain": "rgba(255,190,120,0.50)",    # Light amber
    "Std & No CapGain": "rgba(140,150,165,0.45)",     # Muted slate — baseline
}


# ==============================================================================
# SECTION 12 — Income × Capital Gain Segmentation by Gender
# ==============================================================================

def _render_section12_capgain_gender(
    df: pd.DataFrame,
    cols: dict[str, str | None],
) -> None:
    """100% Stacked Bar: Income × CapGain combo grouped by Gender, using encoded data."""
    _section_header(
        "Income × Capital Gain Segmentation by Gender",
        subtitle="Comparing how non-salary wealth accumulation (Capital Gain) varies between Male and Female income cohorts",
        icon_name="users",
    )

    capgain_col = cols.get("capital_gain")
    sex_col = cols.get("sex")

    if not all(c and c in df.columns for c in [sex_col, capgain_col]):
        styled_alert("Requires sex and capital_gain columns.", "info")
        return

    # 1. Encode data
    df_encoded = data_engine.encode_for_correlation(df)

    if "is_high_income" not in df_encoded.columns or sex_col not in df_encoded.columns or capgain_col not in df_encoded.columns:
        styled_alert("Necessary encoded columns not found.", "info")
        return

    # Extract masks using encoded fields
    hi_mask = df_encoded["is_high_income"] == 1
    has_cg = df_encoded[capgain_col] > 0
    
    # sex mapping in encode_for_correlation: Female=0, Male=1
    sex_series = df_encoded[sex_col].map({1: "Male", 0: "Female"})

    # 2. Build 4 segments
    segs = pd.Series("", index=df_encoded.index)
    segs[(~hi_mask) & (~has_cg)] = "Std & No CapGain"
    segs[(~hi_mask) & (has_cg)]  = "Std & Has CapGain"
    segs[(hi_mask) & (~has_cg)]  = "High & No CapGain"
    segs[(hi_mask) & (has_cg)]   = "High & Has CapGain"

    ct = pd.crosstab(sex_series, segs, normalize="index") * 100

    for seg_name in _SEGMENT_COLORS:
        if seg_name not in ct.columns:
            ct[seg_name] = 0.0

    if "Female" in ct.index and "Male" in ct.index:
        ct = ct.loc[["Female", "Male"]] # Plotly bottom-up

    fig = go.Figure()
    segment_order = ["High & Has CapGain", "High & No CapGain", "Std & Has CapGain", "Std & No CapGain"]
    for seg_name in segment_order:
        vals = ct[seg_name].round(1).values
        fig.add_trace(go.Bar(
            y=ct.index.tolist(),
            x=vals,
            name=seg_name,
            orientation="h",
            marker=dict(color=_SEGMENT_COLORS.get(seg_name, "rgba(148,163,184,0.5)")),
            text=[f"{v:.1f}%" if v >= 3 else "" for v in vals],
            textposition="inside",
            textfont=dict(size=12, color=BRIGHT_TEXT),
            hovertemplate=f"<b>%{{y}}</b><br>{seg_name}: <b>%{{x:.1f}}%</b><extra></extra>",
            width=0.5,
        ))

    fig.update_layout(
        **_base_layout(),
        height=280,
        barmode="stack",
        margin=dict(l=80, r=40, t=30, b=60),
        legend=dict(
            orientation="h", y=-0.25, x=0.5, xanchor="center",
            font=dict(size=11, color=MUTED_COLOR),
        ),
        xaxis=dict(
            title=dict(text="Share of Employees (%)", font=dict(color=MUTED_COLOR, size=11)),
            tickfont=dict(color=MUTED_COLOR, size=10),
            range=[0, 100],
        ),
        yaxis=dict(tickfont=dict(color=BRIGHT_TEXT, size=13)),
    )
    st.plotly_chart(apply_global_theme(fig), use_container_width=True, key="ch_cg_sex_encoded")
    # Insight
    dual_m = ct.loc["Male", "High & Has CapGain"] if "Male" in ct.index else 0
    dual_f = ct.loc["Female", "High & Has CapGain"] if "Female" in ct.index else 0
    hi_m = ct.loc["Male", "High & Has CapGain"] + ct.loc["Male", "High & No CapGain"] if "Male" in ct.index else 0
    hi_f = ct.loc["Female", "High & Has CapGain"] + ct.loc["Female", "High & No CapGain"] if "Female" in ct.index else 0
    
    st.markdown(
        _insight_box(
            f"Using fully encoded data, we confirm that <b>{hi_m:.1f}%</b> of men vs <b>{hi_f:.1f}%</b> of women reach High Income "
            f"(a <b>{round(hi_m - hi_f, 1)} pp</b> gap). "
            f"When evaluating capital gain, <b>{dual_m:.1f}%</b> of men qualify as dual earners "
            f"(High Income + Capital Gain), compared to only <b>{dual_f:.1f}%</b> of women."
        ),
        unsafe_allow_html=True,
    )



# ==============================================================================
# MAIN
# ==============================================================================

def main() -> None:
    """Employee Data Insight — EDA Dashboard."""
    lang = st.session_state.get("lang", "en")

    page_header(
        title=get_text("eda_title", lang),
        subtitle=get_text("eda_subtitle", lang),
    )

    _ensure_workspace_active()
    active_file = st.session_state.get("active_file")
    workspace_status(active_file)

    # ── Load data ─────────────────────────────────────────────────────────
    df_raw = data_engine.load_and_standardize(
        active_file, _file_mtime=data_engine._get_file_mtime(active_file),
    )

    active_file_scan_progress_bar("_eda_done")

    if df_raw.empty:
        styled_alert("No data loaded. Please upload and activate a dataset first.", "warning")
        return

    save_temp_csv(df_raw, prefix="eda_snapshot")

    # ── Resolve columns ───────────────────────────────────────────────────
    cols = _resolve_cols(df_raw)
    income_col = cols.get("income")

    if not income_col:
        styled_alert(
            "No income/salary column detected. EDA requires an income column to analyze.",
            "warning",
        )
        return

    # ── Prepare binned data (on-the-fly, for charts needing binned Age/Edu)
    df_binned = _apply_binning_onthefly(df_raw)

    # ── KPI Metric Cards ──────────────────────────────────────────────────
    st.markdown("<div style='margin-top:16px;'></div>", unsafe_allow_html=True)
    _render_kpis(df_raw, cols)
    section_divider()

    # ── Tab Navigation ─────────────────────────────────────────────────────
    tab_labels = [
        ":material/monitoring: Dataset & Correlations",
        ":material/family_history: Intersecting Demographics",
        ":material/business_center: Career & Occupations",
        ":material/account_balance: Capital Gain & Wealth",
    ]
    tabs = st.tabs(tab_labels)


    # =================================================================
    # TAB 1 — Income Overview
    # =================================================================
    with tabs[0]:
        _tab_summary(
            "<b style='color:rgba(255,255,255,0.6);'>ℹ What this tab reveals</b><br>"
            "An executive summary of the dataset. Visualizes the "
            "<b style='color:#F59E0B;'>macro income distribution</b> and uses correlation analysis to isolate "
            "the <b style='color:#F59E0B;'>strongest single predictors</b> of High Income. "
            "Groups these demographic features into impact tiers for structured analysis."
        )

        # Income Donut | Correlation Heatmap
        col_t1a, col_t1b = st.columns(2, gap="medium")
        with col_t1a:
            insight_1 = _render_section1(df_raw, income_col)
        with col_t1b:
            result_sect2 = _render_section2(df_raw, income_col)
            if result_sect2[0] is None:
                corr_df = None
                insight_2 = ""
            else:
                corr_df, insight_2 = result_sect2

        # Draw vertically-aligned insight boxes
        col_t1a_insight, col_t1b_insight = st.columns(2, gap="medium")
        with col_t1a_insight:
            st.markdown(insight_1, unsafe_allow_html=True)
        with col_t1b_insight:
            if insight_2:
                st.markdown(insight_2, unsafe_allow_html=True)

        _row_spacer()

        # Demographic Breakdown — all correlated features
        _render_section7_breakdown(df_raw, df_binned, income_col, corr_df=corr_df)

    # =================================================================
    # TAB 2 — Career & Demographics
    # =================================================================
    with tabs[1]:
        _tab_summary(
            "<b style='color:rgba(255,255,255,0.6);'>ℹ What this tab reveals</b><br>"
            "How core demographic traits (<b style='color:#F59E0B;'>Gender, Age, Education, Family Role</b>) "
            "interact to compound or constrain earning potential. Identifies deep structural insights, "
            "such as the widening gender gap during peak career years and the synergy between experience and education."
        )

        # Family Role & Gender: Impact on High Income
        _render_section3(df_binned, cols, income_col)

        _row_spacer()

        # Age & Gender: Impact on High Income
        _render_section5_age_gender(df_binned, cols, income_col)

        _row_spacer()

        # Education & Age : Impact on High Income
        _render_section4_edu_age(df_binned, cols, income_col)

        _row_spacer()

        # Education & Gender: Impact on High Income
        _render_section6_edu_gender(df_binned, cols, income_col)



    # =================================================================
    # TAB 3 — Occupation & Working Patterns
    # =================================================================
    with tabs[2]:
        _tab_summary(
            "<b style='color:rgba(255,255,255,0.6);'>ℹ What this tab reveals</b><br>"
            "A structural analysis of <b style='color:#F59E0B;'>career pathways</b>. "
            "Evaluates how <b style='color:#F59E0B;'>education barriers, working hours, experience (age), "
            "and gender</b> shape the earning potential within different occupational tiers. "
            "Reveals which specific job profiles maximize the probability of reaching High Income."
        )

        # Education & Occupation: Impact on High Income
        _render_section8_edu_occ(df_binned, cols)

        _row_spacer()

        # Occupation & Age: Impact on High Income
        _render_section10_occ_age(df_binned, cols, income_col)

        _row_spacer()

        # Occupation & Working Hours: Impact on High Income
        _render_section9_occ_hours(df_binned, cols, income_col)

        _row_spacer()

        # Occupation & Sex: Impact on High Income
        _render_section11_occ_sex(df_raw, df_binned, cols, income_col)

    # =================================================================
    # TAB 4 — Gender Disparity
    # =================================================================
    with tabs[3]:
        _tab_summary(
            "<b style='color:rgba(255,255,255,0.6);'>ℹ What this tab reveals</b><br>"
            "Analyzes <b style='color:#F59E0B;'>capital gain prevalence</b> as a secondary wealth builder. "
            "Focuses on how non-salary investment income is distributed, investigating if capital gain "
            "serves to compound or narrow the structural income gaps between demographics."
        )

        # Income & CapGain by Sex (side-by-side stacked bars -> single stacked)
        _render_section12_capgain_gender(df_raw, cols)

    section_divider()


if __name__ == "__main__":
    main()
