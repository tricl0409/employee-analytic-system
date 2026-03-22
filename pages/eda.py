"""
eda.py — Employee Data Insight Page (EDA)

Slide 1 Layout (6 chart sections):
  ┌─────────────────┬──────────────────────┬──────────────────────┐
  │  SECTION 1       │  SECTION 2            │  SECTION 3         │
  │  Donut: Income   │  Heatmap: Association │  Age × Occ bars    │
  │  Split           │  Cramér's V + PB      │  + Insight         │
  ├─────────────────┴──────────────────────┴──────────────────────┤
  │  SECTION 4 (3/5):  Top Impacting to High Income               │
  │  4 sub-charts + dynamic bullet insights                       │
  │                      │  SECTION 5: Capital │  SECTION 6: Sex  │
  │                      │  Gain vs Income     │  vs Income Donut │
  └──────────────────────┴─────────────────────┴──────────────────┘
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import chi2_contingency

from modules.core import data_engine
from modules.core.preprocessing_engine import PreprocessingEngine
from modules.ui import (
    page_header, workspace_status,
    active_file_scan_progress_bar, section_divider, metric_card,
)
from modules.ui.components import styled_alert
from modules.ui.visualizer import (
    CHART_LAYOUT, MUTED_COLOR, BRIGHT_TEXT, GRID_COLOR,
    apply_global_theme,
)
from modules.utils.localization import get_text
from modules.utils.helpers import _ensure_workspace_active, save_temp_csv
from modules.utils.theme_manager import STATUS_COLORS
from modules.ui.icons import get_icon


# ==============================================================================
# CONSTANTS & CONFIG
# ==============================================================================

_C_HIGH = STATUS_COLORS["warning"]["hex"]   # Amber  — high income (>50K)
_C_STD  = STATUS_COLORS["neutral"]["hex"]   # Blue   — standard income (≤50K)

_STRIP_KEYS = {"legend", "margin"}

# Section header accent (amber — consistent with insight boxes)
_ACCENT_AMBER = "#FF9F43"

# Education color palette — gradient from low → high education
_EDU_COLORS = {
    "Basic":      "rgba(180,160,140,0.55)",     # Warm gray — lowest
    "HS-grad":    "rgba(255,190,120,0.45)",     # Light amber
    "SomeAssoc":  "rgba(255,159,67,0.55)",      # Mid amber
    "Some/Assoc": "rgba(255,159,67,0.55)",      # Mid amber (alias)
    "Bachelors":  "rgba(255,140,40,0.72)",      # Strong amber
    "Advanced":   "rgba(255,120,20,0.90)",      # Deep amber/orange
}

# Education sort order (descending: highest level first)
_EDU_ORDER = ["Advanced", "Bachelors", "Some/Assoc", "SomeAssoc", "HS-grad", "Basic"]


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


def _high_mask(series: pd.Series) -> pd.Series:
    """Boolean mask: True where income >50K."""
    return series.astype(str).str.strip().str.lower().str.contains(r">50k", regex=True, na=False)


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


def _insight_box(html_text: str) -> str:
    """
    Prominent insight callout — amber accent, icon header, bold highlights.
    Auto-converts <b>text</b> inside html_text to amber-colored bold spans.
    """
    highlighted = html_text.replace(
        "<b>", "<b style='color:#FF9F43;font-weight:700;'>",
    )
    return (
        "<div style='"
        "background:rgba(255,159,67,0.07);"
        "border:1px solid rgba(255,159,67,0.18);"
        "border-left:3px solid rgba(255,159,67,0.85);"
        "border-radius:8px;"
        "padding:14px 16px;"
        "margin-top:10px;'"
        ">"
        "<div style='"
        "font-size:0.67rem;font-weight:700;"
        "color:rgba(255,159,67,0.65);"
        "text-transform:uppercase;letter-spacing:1.2px;"
        "margin-bottom:8px;"
        "'>" + get_icon('zap', size=13, color='rgba(255,159,67,0.65)') + " Insight</div>"
        f"<div style='"
        f"font-size:0.80rem;"
        f"color:rgba(255,255,255,0.72);"
        f"line-height:1.75;"
        f"'>{highlighted}</div>"
        "</div>"
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


# ==============================================================================
# ASSOCIATION METRICS — Cramér's V + Point-Biserial
# ==============================================================================

def _cramers_v(col_a: pd.Series, col_b: pd.Series) -> float:
    """
    Compute Cramér's V association between two categorical variables.

    Returns a value in [0, 1] where 0 = independence, 1 = perfect association.
    """
    contingency = pd.crosstab(col_a, col_b)
    if contingency.shape[0] < 2 or contingency.shape[1] < 2:
        return 0.0
    chi2_stat = chi2_contingency(contingency)[0]
    n_obs = len(col_a)
    min_dim = min(contingency.shape) - 1
    if min_dim == 0 or n_obs == 0:
        return 0.0
    return float(np.sqrt(chi2_stat / (n_obs * min_dim)))


def _point_biserial(numeric_series: pd.Series, binary_series: pd.Series) -> float:
    """
    Compute Point-Biserial correlation between a numeric and a binary variable.

    Equivalent to Pearson correlation when one variable is binary {0, 1}.
    Returns value in [-1, 1].
    """
    valid_mask = numeric_series.notna() & binary_series.notna()
    if valid_mask.sum() < 10:
        return 0.0
    return float(np.corrcoef(
        numeric_series[valid_mask].values,
        binary_series[valid_mask].values,
    )[0, 1])


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
            colors=["rgba(255,255,255,0.08)", _C_HIGH],
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
    st.markdown(
        _insight_box(
            f"Out of <b>{total:,}</b> records, <b>{pct_std}%</b> "
            f"({n_std:,}) fall into Standard Income (≤50K) while only "
            f"<b>{pct_high}%</b> ({n_high:,}) reach High Income (>50K) "
            f"— a ratio of approximately <b>{ratio}:1</b>. "
            f"This class imbalance suggests that high income is driven "
            f"by specific structural factors worth exploring."
        ),
        unsafe_allow_html=True,
    )


# Minimum association threshold to display in the heatmap
_ASSOC_MIN_THRESHOLD: float = 0.20


def _compute_association_scores(
    df: pd.DataFrame,
    income_col: str,
    min_threshold: float = _ASSOC_MIN_THRESHOLD,
) -> pd.DataFrame:
    """
    Compute association strength of **every** non-income column with
    binary income, then keep only features with score ≥ *min_threshold*.

    When *df* is the binned copy, most/all columns are categorical,
    so **Cramér's V** is used. For any remaining numeric columns,
    Point-Biserial is used as fallback.

    Args:
        df:             DataFrame (ideally binned).
        income_col:     Name of the income column.
        min_threshold:  Minimum association score to include (default 0.20).

    Returns:
        DataFrame with columns: attribute, association, method
        sorted by association descending, filtered by threshold.
    """
    hi_binary = _high_mask(df[income_col]).astype(float)
    rows = []

    for col in df.columns:
        # Skip income itself
        if col == income_col:
            continue

        series = df[col].dropna()
        if len(series) < 10:
            continue

        # Choose method based on dtype
        if pd.api.types.is_numeric_dtype(series):
            numeric_vals = pd.to_numeric(df[col], errors="coerce")
            score = abs(_point_biserial(numeric_vals, hi_binary))
            method_label = "Point-Biserial"
        else:
            # Categorical (including binned columns) → Cramér's V
            valid_idx = df[col].notna() & df[income_col].notna()
            score = _cramers_v(
                df.loc[valid_idx, col].astype(str),
                hi_binary[valid_idx].astype(int).astype(str),
            )
            method_label = "Cramér's V"

        if score >= min_threshold:
            rows.append({
                "attribute": col,
                "association": round(score, 3),
                "method": method_label,
            })

    result_df = pd.DataFrame(rows)
    if not result_df.empty:
        result_df = result_df.sort_values("association", ascending=False).reset_index(drop=True)
    return result_df


def _chart_association_heatmap(assoc_df: pd.DataFrame) -> go.Figure:
    """Single-column heatmap: association strength [0, 1]."""
    attributes = assoc_df["attribute"].tolist()
    scores = assoc_df["association"].tolist()
    methods = assoc_df["method"].tolist()

    fig = go.Figure(go.Heatmap(
        z=[[s] for s in scores],
        x=["Association"],
        y=attributes,
        text=[[f"{s:.3f}"] for s in scores],
        texttemplate="%{text}",
        textfont=dict(size=12, color="rgba(255,255,255,0.9)"),
        colorscale=[
            [0.0, "rgba(255,255,255,0.03)"],
            [0.3, "rgba(255,159,67,0.20)"],
            [0.6, "rgba(255,159,67,0.45)"],
            [1.0, "rgba(255,159,67,0.80)"],
        ],
        zmin=0,
        zmax=max(scores) * 1.1 if scores else 1.0,
        customdata=methods,
        hovertemplate=(
            "<b>%{y}</b><br>"
            "Association: <b>%{z:.3f}</b><br>"
            "Method: %{customdata}"
            "<extra></extra>"
        ),
        showscale=True,
        colorbar=dict(
            title=dict(text="Strength", font=dict(size=10, color=MUTED_COLOR)),
            tickfont=dict(size=9, color=MUTED_COLOR),
            len=0.8,
            thickness=12,
        ),
        xgap=2,
        ygap=2,
    ))

    fig.update_layout(
        **_base_layout(),
        height=360,
        margin=dict(l=120, r=60, t=30, b=20),
        xaxis=dict(
            tickfont=dict(color=MUTED_COLOR, size=11),
            side="bottom",
        ),
        yaxis=dict(
            tickfont=dict(color=MUTED_COLOR, size=11),
            autorange="reversed",
        ),
    )
    return apply_global_theme(fig)


def _render_section2(
    df: pd.DataFrame,
    income_col: str,
) -> None:
    """Render Association Heatmap (Cramér's V on binned data) + Insight."""
    _section_header(
        "Feature Association with High Income",
        subtitle="Cramér's V score for each feature — only features with association ≥ 20% are shown",
        icon_name="target",
    )

    assoc_df = _compute_association_scores(df, income_col)

    if assoc_df.empty:
        styled_alert("Insufficient data to compute associations.", "info")
        return

    st.plotly_chart(
        _chart_association_heatmap(assoc_df),
        use_container_width=True, key="ch_assoc_heatmap",
    )

    # Dynamic insight: top-3 and count
    n_features = len(assoc_df)
    top3 = assoc_df.head(3)
    top3_names = top3["attribute"].tolist()
    top3_scores = top3["association"].tolist()

    top_parts = [f"<b>{n}</b> ({s:.3f})" for n, s in zip(top3_names, top3_scores)]
    if len(top_parts) >= 3:
        top_text = f"{top_parts[0]}, {top_parts[1]}, and {top_parts[2]}"
    else:
        top_text = ", ".join(top_parts)

    st.markdown(
        _insight_box(
            f"<b>{n_features}</b> out of {len(df.columns) - 1} features show "
            f"meaningful association (≥ 0.20) with High Income. "
            f"The strongest predictors are {top_text}. "
            f"These features should be prioritized in cross-feature analysis."
        ),
        unsafe_allow_html=True,
    )


# ==============================================================================
# SECTION 3 — Cross-Tab Heatmaps: Relationship/Marital × Sex → High Income%
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
            "High Income Rate: <b>%{z:.2f}</b><extra></extra>"
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
        "Family Role & Gender Impact on Income",
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
    _AMBER_SCALE = [
        [0.0, "rgba(255,255,255,0.03)"],
        [0.3, "rgba(255,159,67,0.20)"],
        [0.6, "rgba(255,159,67,0.45)"],
        [1.0, "rgba(255,159,67,0.80)"],
    ]

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
            fig_rel.update_layout(
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
# SECTION 3b — Cross-Tab Heatmap: Age Group × Education → High Income%
# ==============================================================================

def _render_section3b(
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
        "Age & Education: Combined Effect on Income",
        subtitle="Education is the strongest single predictor — but its effect compounds significantly with age and experience",
        icon_name="bar_chart",
    )

    # Pre-compute sort orders
    hi_mask_pre = _high_mask(df_binned[income_col])

    # Y-axis: age groups sorted descending (oldest at top)
    import re
    age_labels = df_binned[age_col].astype(str).unique().tolist()
    sorted_age = sorted(
        age_labels,
        key=lambda lbl: int(re.search(r"\d+", lbl).group()) if re.search(r"\d+", lbl) else 0,
        reverse=True,
    )

    # X-axis: education sorted by overall High Income Rate descending
    rate_by_edu = hi_mask_pre.groupby(df_binned[edu_col].astype(str)).mean()
    sorted_edu = rate_by_edu.sort_values(ascending=False).index.tolist()

    fig = _chart_crosstab_heatmap(
        df_binned, income_col, age_col, edu_col,
        title="High Income Rate: Age Group × Education",
        colorscale=[
            [0.0, "rgba(255,255,255,0.03)"],
            [0.3, "rgba(255,159,67,0.20)"],
            [0.6, "rgba(255,159,67,0.45)"],
            [1.0, "rgba(255,159,67,0.80)"],
        ],
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

    # Dynamic insight
    hi_mask = _high_mask(df_binned[income_col])
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
# SECTION 4 — Top Impacting to High Income (4 sub-charts + insights)
# ==============================================================================

def _chart_hbar_rate(
    df: pd.DataFrame,
    income_col: str,
    group_col: str,
    title: str,
    top_n: int = 0,
) -> go.Figure:
    """
    Horizontal bar: high-income rate by category.

    Args:
        top_n: If > 0, show only top N categories by count. 0 = show all.
    """
    hi_mask = _high_mask(df[income_col])
    rate = hi_mask.groupby(df[group_col].astype(str)).mean()

    if top_n > 0:
        top_cats = df[group_col].value_counts().head(top_n).index
        rate = rate[rate.index.isin(top_cats)]

    rate = rate.sort_values(ascending=True)
    mx = max(rate.max(), 0.01)
    colors = [f"rgba(255,159,67,{0.3 + 0.7 * v / mx:.2f})" for v in rate.values]

    fig = go.Figure(go.Bar(
        y=rate.index.tolist(),
        x=rate.round(2).values,
        orientation="h",
        marker=dict(color=colors, line=dict(width=0)),
        text=[f"{v:.2f}" for v in rate.values],
        textposition="outside",
        textfont=dict(color=MUTED_COLOR, size=10),
        cliponaxis=False,
        hovertemplate="<b>%{y}</b><br>High Income rate: <b>%{x:.2f}</b><extra></extra>",
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
            title=dict(text="High Income Rate", font=dict(color=MUTED_COLOR, size=10)),
            tickfont=dict(color=MUTED_COLOR, size=9),
            gridcolor=GRID_COLOR,
            range=[0, min(mx * 1.35, 1.0)],
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
        f"<b>{feature_col}</b> (V = {assoc_score:.3f}): "
        f"<b>{top_cat}</b> has the highest High Income Rate at <b>{top_pct}%</b>, "
        f"vs <b>{bot_cat}</b> at <b>{bot_pct}%</b> "
        f"— a <b>{gap_pp} pp</b> gap."
    )


def _render_section4(
    df_binned: pd.DataFrame,
    income_col: str,
) -> None:
    """
    Render Demographic Breakdown of High Income.

    Dynamically picks the top 4 features from association scores
    (same computation as Section 2) and renders hbar charts + insights.
    """
    # Compute association on binned data (reuse same logic as Section 2)
    assoc_df = _compute_association_scores(df_binned, income_col)

    if len(assoc_df) < 1:
        styled_alert("Insufficient data for demographic breakdown.", "info")
        return

    top4 = assoc_df.head(4)
    top4_features = top4["attribute"].tolist()
    top4_scores = top4["association"].tolist()

    # Build subtitle dynamically from top-4 feature names
    subtitle_features = ", ".join(top4_features[:3])
    if len(top4_features) >= 4:
        subtitle_features += f", and {top4_features[3]}"

    _section_header(
        "Demographic Breakdown of High Income",
        subtitle=f"High Income Rate across the top associated features: {subtitle_features}",
        icon_name="zap",
    )

    col_charts, col_insights = st.columns([3, 2], gap="medium")

    with col_charts:
        # Row 1: feature 1 + feature 2
        c1, c2 = st.columns(2)
        for idx, (col_slot, feat, score) in enumerate(
            zip([c1, c2], top4_features[:2], top4_scores[:2])
        ):
            with col_slot:
                if feat in df_binned.columns:
                    fig = _chart_hbar_rate(
                        df_binned, income_col, feat,
                        title=f"High Income Rate by {feat}",
                    )
                    st.plotly_chart(
                        fig, use_container_width=True,
                        key=f"ch_s4_{idx}",
                    )

        # Row 2: feature 3 + feature 4
        if len(top4_features) > 2:
            c3, c4 = st.columns(2)
            for idx, (col_slot, feat, score) in enumerate(
                zip([c3, c4], top4_features[2:4], top4_scores[2:4]),
                start=2,
            ):
                with col_slot:
                    if feat in df_binned.columns:
                        fig = _chart_hbar_rate(
                            df_binned, income_col, feat,
                            title=f"High Income Rate by {feat}",
                        )
                        st.plotly_chart(
                            fig, use_container_width=True,
                            key=f"ch_s4_{idx}",
                        )

    with col_insights:
        bullets = []
        for feat, score in zip(top4_features, top4_scores):
            if feat in df_binned.columns:
                bullets.append(
                    _compute_feature_insight(
                        df_binned, feat, income_col, score,
                    )
                )

        if bullets:
            bullet_html = "".join(
                f"<li style='margin-bottom:14px;line-height:1.75;'>{b}</li>"
                for b in bullets
            )
            st.markdown(
                f"<div style='"
                f"background:rgba(255,159,67,0.05);"
                f"border:1px solid rgba(255,159,67,0.15);"
                f"border-left:3px solid rgba(255,159,67,0.65);"
                f"border-radius:0 12px 12px 0;"
                f"padding:20px 22px;"
                f"margin-top:10px;"
                f"'>"
                f"<div style='font-size:0.67rem;font-weight:700;"
                f"color:rgba(255,159,67,0.65);"
                f"text-transform:uppercase;letter-spacing:1.2px;"
                f"margin-bottom:10px;'"
                f"'>" + get_icon('bar_chart', size=13, color='rgba(255,159,67,0.65)') + " Key Findings</div>"
                f"<ul style='"
                f"font-size:0.82rem;"
                f"color:rgba(255,255,255,0.72);"
                f"padding-left:18px;"
                f"margin:0;"
                f"list-style-type:disc;"
                f"'>{bullet_html}</ul>"
                f"</div>",
                unsafe_allow_html=True,
            )
        else:
            styled_alert("Insufficient data for insight analysis.", "info")


# ==============================================================================
# SECTION 5 — Capital Gain vs Income Level
# ==============================================================================

def _render_section5(
    df: pd.DataFrame,
    cols: dict[str, str | None],
    income_col: str,
) -> None:
    """Capital Gain distribution comparison by income group."""
    _section_header(
        "Capital Gain Distribution by Income Class",
        subtitle="Comparing non-salary investment income between standard and high earners",
        icon_name="bar_chart",
    )

    capgain_col = cols.get("capital_gain")
    if not capgain_col or capgain_col not in df.columns:
        styled_alert("No capital_gain column found.", "info")
        return

    hi_mask = _high_mask(df[income_col])
    capgain = pd.to_numeric(df[capgain_col], errors="coerce")

    # Compute metrics per group
    std_mean = capgain[~hi_mask].mean()
    high_mean = capgain[hi_mask].mean()
    std_pct_cg = round((capgain[~hi_mask] > 0).sum() / (~hi_mask).sum() * 100, 1) if (~hi_mask).sum() else 0
    high_pct_cg = round((capgain[hi_mask] > 0).sum() / hi_mask.sum() * 100, 1) if hi_mask.sum() else 0
    multiplier = round(high_mean / std_mean, 1) if std_mean > 0 else 0

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=["≤50K", ">50K"],
        y=[std_mean, high_mean],
        name="Avg Capital Gain",
        marker=dict(color=["rgba(255,159,67,0.35)", "rgba(255,159,67,0.75)"]),
        text=[f"${std_mean:,.0f}", f"${high_mean:,.0f}"],
        textposition="outside",
        textfont=dict(size=11, color=BRIGHT_TEXT),
        hovertemplate="<b>%{x}</b><br>Avg Capital Gain: <b>$%{y:,.0f}</b><extra></extra>",
    ))

    fig.update_layout(
        **_base_layout(),
        height=300,
        showlegend=False,
        margin=dict(l=40, r=20, t=30, b=40),
        title=dict(
            text="Average Capital Gain by Income Bracket",
            font=dict(size=11, color=MUTED_COLOR),
            x=0.5, xanchor="center",
        ),
        xaxis=dict(tickfont=dict(color=MUTED_COLOR, size=11)),
        yaxis=dict(
            title=dict(text="Avg Capital Gain ($)", font=dict(color=MUTED_COLOR, size=10)),
            tickfont=dict(color=MUTED_COLOR, size=9),
            gridcolor=GRID_COLOR,
        ),
    )
    st.plotly_chart(apply_global_theme(fig), use_container_width=True, key="ch_capgain")

    st.markdown(
        _insight_box(
            f"High Income earners generate <b>{multiplier}×</b> more capital gain on average "
            f"(${high_mean:,.0f} vs ${std_mean:,.0f}). "
            f"<b>{high_pct_cg}%</b> of High Income individuals have capital gain > 0, "
            f"compared to only <b>{std_pct_cg}%</b> of standard earners — "
            f"indicating that non-salary wealth accumulation is strongly concentrated "
            f"among higher earners."
        ),
        unsafe_allow_html=True,
    )


# ==============================================================================
# SECTION 6 — Sex vs Income Level (Dual Donut)
# ==============================================================================

def _render_section6(
    df: pd.DataFrame,
    cols: dict[str, str | None],
    income_col: str,
) -> None:
    """Dual donut: Male vs Female income split."""
    _section_header(
        "Sex vs. Income Level",
        subtitle="Income distribution split by gender",
        icon_name="users",
    )

    sex_col = cols.get("sex")
    if not sex_col or sex_col not in df.columns:
        styled_alert("No sex/gender column found.", "info")
        return

    hi_mask = _high_mask(df[income_col])
    sex_series = df[sex_col].astype(str).str.strip()

    # Get unique gender labels (case-insensitive matching)
    sex_lower = sex_series.str.lower()
    male_mask = sex_lower == "male"
    female_mask = sex_lower == "female"

    male_high = int((hi_mask & male_mask).sum())
    male_std = int((~hi_mask & male_mask).sum())
    female_high = int((hi_mask & female_mask).sum())
    female_std = int((~hi_mask & female_mask).sum())

    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{"type": "pie"}, {"type": "pie"}]],
        subplot_titles=["Male", "Female"],
    )

    # Male donut
    fig.add_trace(go.Pie(
        labels=["≤50K", ">50K"],
        values=[male_std, male_high],
        hole=0.55,
        marker=dict(colors=["rgba(255,255,255,0.08)", _C_HIGH]),
        textinfo="percent",
        textfont=dict(size=11, color=BRIGHT_TEXT),
        hovertemplate="Male<br><b>%{label}</b>: %{value:,}<br>%{percent}<extra></extra>",
        showlegend=False,
    ), row=1, col=1)

    # Female donut
    fig.add_trace(go.Pie(
        labels=["≤50K", ">50K"],
        values=[female_std, female_high],
        hole=0.55,
        marker=dict(colors=["rgba(255,255,255,0.08)", "rgba(59,130,246,0.7)"]),
        textinfo="percent",
        textfont=dict(size=11, color=BRIGHT_TEXT),
        hovertemplate="Female<br><b>%{label}</b>: %{value:,}<br>%{percent}<extra></extra>",
        showlegend=False,
    ), row=1, col=2)

    fig.update_layout(
        **_base_layout(),
        height=300,
        margin=dict(l=20, r=20, t=40, b=20),
    )
    # Subtitle font color
    for ann in fig.layout.annotations:
        ann.font.color = MUTED_COLOR
        ann.font.size = 12

    st.plotly_chart(apply_global_theme(fig), use_container_width=True, key="ch_sex_donut")

    # Compute percentages for insight
    male_total = male_high + male_std
    female_total = female_high + female_std
    male_pct = round(male_high / male_total * 100, 1) if male_total else 0
    female_pct = round(female_high / female_total * 100, 1) if female_total else 0
    gap = round(male_pct - female_pct, 1)

    st.markdown(
        _insight_box(
            f"<b>Gender Gap:</b> <b>{male_pct}%</b> of men earn >50K vs <b>{female_pct}%</b> of women "
            f"— a <b>{gap} percentage point</b> gap. "
            f"The percentage of men earning >50K is higher than that of women."
        ),
        unsafe_allow_html=True,
    )


# ==============================================================================
# SECTION 7 — Occupation vs Education (100% Stacked Bar)
# ==============================================================================

def _render_section7(
    df_binned: pd.DataFrame,
    cols: dict[str, str | None],
) -> None:
    """100% stacked horizontal bar: Education group composition per Occupation."""
    _section_header(
        "Education Composition by Occupation",
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


# ==============================================================================
# SECTION 8 — %Employees with Capital Gain >0 (Dual Heatmap)
# ==============================================================================

def _render_section8(
    df: pd.DataFrame,
    df_binned: pd.DataFrame,
    cols: dict[str, str | None],
    income_col: str,
) -> None:
    """Dual heatmap: % with CapGain>0 by Education×Occupation, split by income."""
    _section_header(
        "Capital Gain Prevalence: Education × Occupation",
        subtitle="Percentage of employees with non-zero capital gain, compared across income brackets",
        icon_name="eye",
    )

    occ_col = cols.get("occupation")
    edu_col = cols.get("education")
    capgain_col = cols.get("capital_gain")
    if not all(c and c in df.columns for c in [capgain_col]):
        styled_alert("Requires capital_gain column.", "info")
        return
    if not occ_col or occ_col not in df_binned.columns:
        styled_alert("No occupation column found.", "info")
        return
    if not edu_col or edu_col not in df_binned.columns:
        styled_alert("No education column found.", "info")
        return

    hi_mask = _high_mask(df[income_col])
    capgain = pd.to_numeric(df[capgain_col], errors="coerce")
    has_cg = capgain > 0
    edu_binned = df_binned[edu_col].astype(str)
    occ_binned = df_binned[occ_col].astype(str)

    heatmap_configs = [
        (~hi_mask, "Standard Income (≤50K)"),
        (hi_mask, "High Income (>50K)"),
    ]

    col_left, col_right = st.columns(2, gap="medium")

    best_cell_label = ""
    best_cell_val = 0.0

    for idx, (mask, label) in enumerate(heatmap_configs):
        sub_edu = edu_binned[mask]
        sub_occ = occ_binned[mask]
        sub_cg = has_cg[mask]

        ct_total = pd.crosstab(sub_edu, sub_occ)
        ct_cg = pd.crosstab(sub_edu, sub_occ, values=sub_cg, aggfunc="sum")
        pct = (ct_cg / ct_total.replace(0, np.nan) * 100).fillna(0).round(1)

        # Track peak cell in >50K group
        if idx == 1 and pct.values.max().max() > 0:
            max_idx = np.unravel_index(pct.values.argmax(), pct.values.shape)
            best_cell_label = f"{pct.index[max_idx[0]]} × {pct.columns[max_idx[1]]}"
            best_cell_val = pct.values[max_idx[0], max_idx[1]]

        fig = go.Figure(go.Heatmap(
            z=pct.values,
            x=pct.columns.tolist(),
            y=pct.index.tolist(),
            text=[[f"{v:.0f}%" for v in row] for row in pct.values],
            texttemplate="%{text}",
            textfont=dict(size=9, color="rgba(255,255,255,0.85)"),
            colorscale=[
                [0.0, "rgba(255,255,255,0.03)"],
                [0.5, "rgba(255,159,67,0.30)"],
                [1.0, "rgba(255,159,67,0.75)"],
            ],
            zmin=0,
            zmax=max(pct.values.max().max(), 1),
            showscale=(idx == 1),
            colorbar=dict(
                tickfont=dict(size=8, color=MUTED_COLOR),
                thickness=10, len=0.8, outlinewidth=0,
                ticksuffix="%",
            ) if idx == 1 else None,
            hovertemplate="<b>%{y}</b> × <b>%{x}</b><br>%{z:.1f}% have CapGain<extra></extra>",
            xgap=2,
            ygap=2,
        ))
        fig.update_layout(
            **_base_layout(),
            height=300,
            margin=dict(l=80, r=30 if idx == 0 else 50, t=30, b=80),
            title=dict(
                text=label,
                font=dict(size=11, color=MUTED_COLOR),
                x=0.5, xanchor="center",
            ),
            xaxis=dict(tickfont=dict(color=MUTED_COLOR, size=8), tickangle=-35),
            yaxis=dict(tickfont=dict(color=MUTED_COLOR, size=9), autorange="reversed"),
        )

        container = col_left if idx == 0 else col_right
        with container:
            st.plotly_chart(apply_global_theme(fig), use_container_width=True, key=f"ch_cg_{label}")

    # Dynamic insight
    insight_text = (
        f"In the High Income group, <b>{best_cell_label}</b> achieves the highest "
        f"capital gain prevalence at <b>{best_cell_val:.1f}%</b>. "
        f"Higher education and professional occupations are strongly associated with "
        f"non-salary wealth accumulation — suggesting these groups leverage investment income "
        f"as an additional wealth-building channel."
    ) if best_cell_val > 0 else (
        "Capital gain prevalence varies significantly across education and occupation segments."
    )

    st.markdown(
        _insight_box(insight_text),
        unsafe_allow_html=True,
    )


# ==============================================================================
# SECTION 9 — Occupation × Working Hours (Bubble)
# ==============================================================================

def _render_section9(
    df_binned: pd.DataFrame,
    cols: dict[str, str | None],
    income_col: str,
) -> None:
    """Bubble chart: Occupation × Hours bins, size = count, color = High Income Rate."""
    _section_header(
        "Working Hours Pattern by Occupation",
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
    def _hours_sort_key(label: str) -> int:
        """Extract a numeric sort key from a hours-group label."""
        digits = "".join(c for c in label if c.isdigit())
        return int(digits) if digits else 999

    sorted_hours = sorted(ct_count.columns, key=_hours_sort_key)
    ct_count = ct_count.reindex(columns=sorted_hours, fill_value=0)
    ct_rate = ct_rate.reindex(columns=sorted_hours, fill_value=0)

    # Sort occupations by total working hours descending (ascending for Plotly bottom-to-top)
    # Compute weighted total hours per occupation
    hrs_weights = {col: _hours_sort_key(col) for col in sorted_hours}
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
# SECTION 10 — %Employees >50K by Age × Occupation (Heatmap)
# ==============================================================================

def _render_section10(
    df_binned: pd.DataFrame,
    cols: dict[str, str | None],
    income_col: str,
) -> None:
    """Cross-tab heatmap: Age Group × Occupation → High Income Rate (from df_binned)."""
    _section_header(
        "Age & Occupation: Joint Income Probability",
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

    def _age_sort_key(label: str) -> int:
        """Extract the first number from an age-group label for sorting."""
        import re
        match = re.search(r"\d+", label)
        return int(match.group()) if match else 0

    sorted_age = sorted(age_labels, key=_age_sort_key, reverse=True)  # oldest at top

    # X-axis: occupations sorted by overall High Income Rate descending
    rate_by_occ = hi_mask.groupby(df_binned[occ_col].astype(str)).mean()
    sorted_occ = rate_by_occ.sort_values(ascending=False).index.tolist()

    fig = _chart_crosstab_heatmap(
        df_binned, income_col, age_col, occ_col,
        title="High Income Rate: Age Group × Occupation",
        colorscale=[
            [0.0, "rgba(255,255,255,0.03)"],
            [0.3, "rgba(255,159,67,0.20)"],
            [0.6, "rgba(255,159,67,0.45)"],
            [1.0, "rgba(255,159,67,0.80)"],
        ],
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
# SECTION 11 — High Income by Sex × Occupation (Grouped Bar)
# ==============================================================================

def _render_section11(
    df: pd.DataFrame,
    cols: dict[str, str | None],
    income_col: str,
) -> None:
    """Grouped horizontal bar: % >50K by Occupation, colored by Sex."""
    _section_header(
        "Employees Earned High Income by Sex and Occupation",
        subtitle="Gender gap analysis across occupation categories",
        icon_name="users",
    )

    occ_col = cols.get("occupation")
    sex_col = cols.get("sex")
    if not occ_col or occ_col not in df.columns:
        styled_alert("No occupation column found.", "info")
        return
    if not sex_col or sex_col not in df.columns:
        styled_alert("No sex/gender column found.", "info")
        return

    hi_mask = _high_mask(df[income_col])
    occ = df[occ_col].astype(str)
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
        marker=dict(color="rgba(59,130,246,0.7)"),
        text=[f"{v:.1f}%" for v in rate_male.values * 100],
        textposition="outside",
        textfont=dict(size=9, color=MUTED_COLOR),
        cliponaxis=False,
        hovertemplate="<b>%{y}</b> (Male)<br>%{x:.1f}% earn >50K<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        y=rate_female.index.tolist(),
        x=(rate_female.values * 100).round(1),
        name="Female",
        orientation="h",
        marker=dict(color="rgba(236,72,153,0.7)"),
        text=[f"{v:.1f}%" for v in rate_female.values * 100],
        textposition="outside",
        textfont=dict(size=9, color=MUTED_COLOR),
        cliponaxis=False,
        hovertemplate="<b>%{y}</b> (Female)<br>%{x:.1f}% earn >50K<extra></extra>",
    ))

    fig.update_layout(
        **_base_layout(),
        height=520,
        barmode="group",
        bargap=0.25,
        bargroupgap=0.08,
        margin=dict(l=150, r=70, t=20, b=70),
        legend=dict(
            orientation="h", y=-0.12, x=0.5, xanchor="center",
            font=dict(size=10, color=MUTED_COLOR),
        ),
        xaxis=dict(
            title=dict(text="% Earning >50K", font=dict(color=MUTED_COLOR, size=10)),
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
    st.markdown(
        _insight_box(
            f"<b>Gender Gap:</b> The chart shows that the percentage of men earning >50K "
            f"(<b>{avg_male:.1f}%</b> avg) is higher than that of women "
            f"(<b>{avg_female:.1f}%</b> avg) — a <b>{gap:.1f} pp</b> gap — "
            f"across virtually all occupation categories."
        ),
        unsafe_allow_html=True,
    )


# ==============================================================================
# SECTION 12 — Income & Capital Gain by Marital Status (100% Stacked)
# ==============================================================================

_SEGMENT_COLORS = {
    "≤50K & No CapGain": "rgba(100,116,139,0.60)",   # Slate — baseline
    "≤50K & Has CapGain": "rgba(56,189,248,0.70)",   # Cyan — CapGain lift
    ">50K & No CapGain": "rgba(251,146,60,0.75)",    # Orange — high income
    ">50K & Has CapGain": "rgba(250,204,21,0.90)",   # Gold — best segment
}


def _income_capgain_stacked(
    df: pd.DataFrame,
    group_col: str,
    income_col: str,
    capgain_col: str,
    chart_key: str,
    title: str,
) -> go.Figure:
    """Reusable 100% stacked horizontal bar: 4-segment Income×CapGain combo."""
    hi_mask = _high_mask(df[income_col])
    capgain = pd.to_numeric(df[capgain_col], errors="coerce")
    has_cg = capgain > 0

    # Create segment column
    segs = pd.Series("", index=df.index)
    segs[(~hi_mask) & (~has_cg)] = "≤50K & No CapGain"
    segs[(~hi_mask) & (has_cg)]  = "≤50K & Has CapGain"
    segs[(hi_mask) & (~has_cg)]  = ">50K & No CapGain"
    segs[(hi_mask) & (has_cg)]   = ">50K & Has CapGain"

    group = df[group_col].astype(str)
    ct = pd.crosstab(group, segs, normalize="index") * 100

    # Ensure all 4 segments exist
    for seg_name in _SEGMENT_COLORS:
        if seg_name not in ct.columns:
            ct[seg_name] = 0.0

    # Sort by >50K total descending
    ct["_sort"] = ct.get(">50K & No CapGain", 0) + ct.get(">50K & Has CapGain", 0)
    ct = ct.sort_values("_sort", ascending=True).drop(columns=["_sort"])

    fig = go.Figure()
    segment_order = ["≤50K & No CapGain", "≤50K & Has CapGain", ">50K & No CapGain", ">50K & Has CapGain"]
    for seg_name in segment_order:
        if seg_name not in ct.columns:
            continue
        vals = ct[seg_name].round(1).values
        fig.add_trace(go.Bar(
            y=ct.index.tolist(),
            x=vals,
            name=seg_name,
            orientation="h",
            marker=dict(color=_SEGMENT_COLORS.get(seg_name, "rgba(148,163,184,0.5)")),
            text=[f"{v:.0f}%" if v >= 5 else "" for v in vals],
            textposition="inside",
            textfont=dict(size=9, color=BRIGHT_TEXT),
            hovertemplate=f"<b>%{{y}}</b><br>{seg_name}: <b>%{{x:.1f}}%</b><extra></extra>",
        ))

    fig.update_layout(
        **_base_layout(),
        height=max(320, len(ct) * 32),
        barmode="stack",
        margin=dict(l=140, r=30, t=30, b=70),
        title=dict(
            text=title,
            font=dict(size=11, color=MUTED_COLOR),
            x=0.5, xanchor="center",
        ),
        legend=dict(
            orientation="h", y=-0.25, x=0.5, xanchor="center",
            font=dict(size=9, color=MUTED_COLOR),
        ),
        xaxis=dict(
            title=dict(text="Share of employees (%)", font=dict(color=MUTED_COLOR, size=10)),
            tickfont=dict(color=MUTED_COLOR, size=9),
            range=[0, 100],
        ),
        yaxis=dict(tickfont=dict(color=MUTED_COLOR, size=10)),
    )
    return apply_global_theme(fig)


def _render_section12(
    df: pd.DataFrame,
    cols: dict[str, str | None],
    income_col: str,
) -> None:
    """Income & Capital Gain by Marital Status — 100% stacked."""
    _section_header(
        "Marital Status × Income & Capital Gain",
        subtitle="4-segment composition showing how marital status relates to dual income streams (salary + investment)",
        icon_name="heart",
    )

    marital_col = cols.get("marital")
    capgain_col = cols.get("capital_gain")
    if not marital_col or marital_col not in df.columns:
        styled_alert("No marital_status column found.", "info")
        return
    if not capgain_col or capgain_col not in df.columns:
        styled_alert("No capital_gain column found.", "info")
        return

    fig = _income_capgain_stacked(df, marital_col, income_col, capgain_col,
                                   "ch_cg_marital", "By Marital Status (100% Stacked)")
    st.plotly_chart(fig, use_container_width=True, key="ch_cg_marital")

    # Dynamic insight
    hi_mask = _high_mask(df[income_col])
    capgain = pd.to_numeric(df[capgain_col], errors="coerce")
    has_cg = capgain > 0
    dual_earner = hi_mask & has_cg
    rate_by_marital = dual_earner.groupby(df[marital_col].astype(str)).mean() * 100
    if not rate_by_marital.empty:
        top_group = rate_by_marital.idxmax()
        top_val = round(rate_by_marital.max(), 1)
        low_group = rate_by_marital.idxmin()
        low_val = round(rate_by_marital.min(), 1)
        st.markdown(
            _insight_box(
                f"<b>{top_group}</b> individuals lead with <b>{top_val}%</b> "
                f"combining both High Income and capital gain (“dual earners”), "
                f"while <b>{low_group}</b> has only <b>{low_val}%</b>. "
                f"This suggests that family stability may correlate with "
                f"greater wealth accumulation through investment channels."
            ),
            unsafe_allow_html=True,
        )


# ==============================================================================
# SECTION 13 — Income & Capital Gain by Occupation (100% Stacked)
# ==============================================================================

def _render_section13(
    df: pd.DataFrame,
    cols: dict[str, str | None],
    income_col: str,
) -> None:
    """Income & Capital Gain by Occupation — 100% stacked."""
    _section_header(
        "Occupation × Income & Capital Gain",
        subtitle="Which occupations show the highest concentration of dual-income earners (salary + capital gain)?",
        icon_name="briefcase",
    )

    occ_col = cols.get("occupation")
    capgain_col = cols.get("capital_gain")
    if not occ_col or occ_col not in df.columns:
        styled_alert("No occupation column found.", "info")
        return
    if not capgain_col or capgain_col not in df.columns:
        styled_alert("No capital_gain column found.", "info")
        return

    fig = _income_capgain_stacked(df, occ_col, income_col, capgain_col,
                                   "ch_cg_occ", "By Occupation (100% Stacked)")
    st.plotly_chart(fig, use_container_width=True, key="ch_cg_occ")

    # Dynamic insight
    hi_mask = _high_mask(df[income_col])
    capgain = pd.to_numeric(df[capgain_col], errors="coerce")
    has_cg = capgain > 0
    dual_earner = hi_mask & has_cg
    rate_by_occ = dual_earner.groupby(df[occ_col].astype(str)).mean() * 100
    if not rate_by_occ.empty:
        top_occ = rate_by_occ.idxmax()
        top_val = round(rate_by_occ.max(), 1)
        low_occ = rate_by_occ.idxmin()
        low_val = round(rate_by_occ.min(), 1)
        st.markdown(
            _insight_box(
                f"<b>{top_occ}</b> has the highest dual-earner concentration at "
                f"<b>{top_val}%</b> (High Income + Capital Gain), "
                f"while <b>{low_occ}</b> has only <b>{low_val}%</b>. "
                f"This <b>{round(top_val - low_val, 1)} pp</b> gap suggests that "
                f"certain occupations provide significantly more opportunities for "
                f"non-salary wealth accumulation."
            ),
            unsafe_allow_html=True,
        )


# ==============================================================================
# SECTION 14 — Income & Capital Gain by Sex (Stacked, per Occupation)
# ==============================================================================

def _render_section14(
    df: pd.DataFrame,
    cols: dict[str, str | None],
    income_col: str,
) -> None:
    """2 stacked bar charts side-by-side: Male vs Female, grouped by Occupation."""
    _section_header(
        "Income & Capital Gain by Sex",
        subtitle="Male vs Female breakdown of income and capital gain by occupation",
        icon_name="users",
    )

    occ_col = cols.get("occupation")
    sex_col = cols.get("sex")
    capgain_col = cols.get("capital_gain")
    if not all(c and c in df.columns for c in [occ_col, sex_col, capgain_col]):
        styled_alert("Requires occupation, sex, and capital_gain columns.", "info")
        return

    sex_series = df[sex_col].astype(str).str.strip().str.lower()

    for idx, (gender, label) in enumerate([("male", "Male"), ("female", "Female")]):
        sub = df[sex_series == gender]
        if sub.empty:
            styled_alert(f"No {label} data found.", "info")
            continue
        fig = _income_capgain_stacked(
            sub, occ_col, income_col, capgain_col,
            f"ch_cg_sex_{gender}",
            f"{label} — Income & CapGain by Occupation",
        )
        st.plotly_chart(fig, use_container_width=True, key=f"ch_cg_sex_{gender}")

        if idx == 0:
            _row_spacer()

    # Insight
    hi_mask = _high_mask(df[income_col])
    male_hi_pct = round(hi_mask[sex_series == "male"].mean() * 100, 1) if (sex_series == "male").any() else 0
    female_hi_pct = round(hi_mask[sex_series == "female"].mean() * 100, 1) if (sex_series == "female").any() else 0
    st.markdown(
        _insight_box(
            f"<b>Gender Gap:</b> <b>{male_hi_pct}%</b> of men earn >50K vs <b>{female_hi_pct}%</b> "
            f"of women. Men in <b>Management/Professional</b> have the highest rate of earning "
            f">50K with capital gain."
        ),
        unsafe_allow_html=True,
    )


# ==============================================================================
# SECTION 15 — Typical High-Income Profile
# ==============================================================================

def _render_section15(
    df: pd.DataFrame,
    df_binned: pd.DataFrame,
    cols: dict[str, str | None],
    income_col: str,
) -> None:
    """Typical High-Income Profile: summary bullets + filtered stacked bar."""
    _section_header(
        "Typical High-Income Profile",
        subtitle="Characteristics of the typical >50K individual with capital gain",
        icon_name="zap",
    )

    occ_col = cols.get("occupation")
    edu_col = cols.get("education")
    sex_col = cols.get("sex")
    marital_col = cols.get("marital")
    capgain_col = cols.get("capital_gain")

    hi_mask = _high_mask(df[income_col])
    hi_df = df[hi_mask].copy()

    # Build profile bullets from data
    bullets = []

    # Most common sex
    if sex_col and sex_col in hi_df.columns:
        top_sex = hi_df[sex_col].astype(str).str.strip().mode()
        if not top_sex.empty:
            bullets.append(f"<b>{top_sex.iloc[0]}</b>")

    # Most common marital status
    if marital_col and marital_col in hi_df.columns:
        top_marital = hi_df[marital_col].astype(str).str.strip().mode()
        if not top_marital.empty:
            bullets.append(f"<b>{top_marital.iloc[0]}</b>")

    # Most common education (binned)
    if edu_col and edu_col in df_binned.columns:
        hi_edu = df_binned.loc[hi_mask, edu_col].astype(str).mode()
        if not hi_edu.empty:
            bullets.append(f"<b>{hi_edu.iloc[0]}</b> or higher education")

    # Most common occupation
    if occ_col and occ_col in hi_df.columns:
        top_occ = hi_df[occ_col].astype(str).str.strip().mode()
        if not top_occ.empty:
            bullets.append(f"Often holding <b>{top_occ.iloc[0]}</b> positions")

    # Capital gain
    if capgain_col and capgain_col in hi_df.columns:
        cg = pd.to_numeric(hi_df[capgain_col], errors="coerce")
        pct_cg = round((cg > 0).sum() / len(hi_df) * 100, 1) if len(hi_df) else 0
        if pct_cg > 0:
            bullets.append(f"<b>{pct_cg}%</b> have investment income (Capital Gain)")

    # Key Takeaway card — full-width
    if bullets:
        bullet_html = "".join(
            f"<li style='margin-bottom:10px;line-height:1.7;'>{b}</li>"
            for b in bullets
        )
        st.markdown(
            f"<div style='"
            f"background:rgba(255,159,67,0.05);"
            f"border:1px solid rgba(255,159,67,0.15);"
            f"border-left:3px solid rgba(255,159,67,0.65);"
            f"border-radius:0 12px 12px 0;"
            f"padding:18px 22px;"
            f"margin-bottom:8px;"
            f"'>"
            f"<div style='font-size:0.67rem;font-weight:700;"
            f"color:rgba(255,159,67,0.65);"
            f"text-transform:uppercase;letter-spacing:1.2px;"
            f"margin-bottom:10px;'"
            f"'>" + get_icon('zap', size=13, color='rgba(255,159,67,0.65)') + " Key Takeaway</div>"
            f"<ul style='"
            f"font-size:0.82rem;"
            f"color:rgba(255,255,255,0.72);"
            f"padding-left:18px;"
            f"margin:0;"
            f"list-style-type:disc;"
            f"display:flex;flex-wrap:wrap;gap:0 40px;"
            f"'>{bullet_html}</ul>"
            f"</div>",
            unsafe_allow_html=True,
        )

    _row_spacer()

    # Filtered stacked bar — full-width
    if occ_col and edu_col and sex_col and capgain_col:
        sex_lower = df[sex_col].astype(str).str.strip().str.lower()
        cg = pd.to_numeric(df[capgain_col], errors="coerce")
        filter_mask = hi_mask & (sex_lower == "male") & (cg > 0)
        filtered_df = df_binned[filter_mask]

        if not filtered_df.empty and occ_col in filtered_df.columns and edu_col in filtered_df.columns:
            ct = pd.crosstab(
                filtered_df[occ_col].astype(str),
                filtered_df[edu_col].astype(str),
                normalize="index",
            ) * 100

            fig = go.Figure()
            for edu_group in ct.columns:
                color = _EDU_COLORS.get(edu_group, "rgba(148,163,184,0.5)")
                vals = ct[edu_group].round(1).values
                fig.add_trace(go.Bar(
                    y=ct.index.tolist(),
                    x=vals,
                    name=edu_group,
                    orientation="h",
                    marker=dict(color=color),
                    text=[f"{v:.0f}%" if v >= 5 else "" for v in vals],
                    textposition="inside",
                    textfont=dict(size=9, color=BRIGHT_TEXT),
                    hovertemplate=(
                        f"<b>%{{y}}</b><br>{edu_group}: <b>%{{x:.1f}}%</b><extra></extra>"
                    ),
                ))

            fig.update_layout(
                **_base_layout(),
                height=350,
                barmode="stack",
                margin=dict(l=150, r=30, t=30, b=70),
                title=dict(
                    text="Education within Occ. (Male, >50K, CapGain>0)",
                    font=dict(size=11, color=MUTED_COLOR),
                    x=0.5, xanchor="center",
                ),
                legend=dict(
                    orientation="h", y=-0.20, x=0.5, xanchor="center",
                    font=dict(size=10, color=MUTED_COLOR),
                ),
                xaxis=dict(
                    range=[0, 100],
                    tickfont=dict(color=MUTED_COLOR, size=9),
                    title=dict(text="% within Occupation", font=dict(color=MUTED_COLOR, size=10)),
                ),
                yaxis=dict(tickfont=dict(color=MUTED_COLOR, size=10)),
            )
            st.plotly_chart(
                apply_global_theme(fig), use_container_width=True, key="ch_profile_stacked",
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
        ":material/monitoring: Income Overview",
        ":material/work: Career & Earning Factors",
        ":material/attach_money: Investment Income Analysis",
        ":material/wc: Gender Disparity",
        ":material/target: Archetype Profile",
    ]
    tabs = st.tabs(tab_labels)

    # -- helper: render tab insight summary --
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

    # =================================================================
    # TAB 1 — Income Overview
    # =================================================================
    with tabs[0]:
        _tab_summary(
            "<b style='color:rgba(255,255,255,0.6);'>ℹ What this tab reveals</b><br>"
            "Overall <b style='color:#F59E0B;'>income class distribution</b>, "
            "<b style='color:#F59E0B;'>feature-level association strength</b> with High Income, "
            "and a <b style='color:#F59E0B;'>demographic breakdown</b> of the top 4 most "
            "associated features — showing which categories within each feature "
            "have the highest and lowest High Income Rate."
        )

        # Income Donut | Correlation Heatmap
        col_t1a, col_t1b = st.columns(2, gap="medium")
        with col_t1a:
            _render_section1(df_raw, income_col)
        with col_t1b:
            _render_section2(df_binned, income_col)

        _row_spacer()

        # Demographic Breakdown — top 4 associated features (from binned data)
        _render_section4(df_binned, income_col)

    # =================================================================
    # TAB 2 — Career & Demographics
    # =================================================================
    with tabs[1]:
        _tab_summary(
            "<b style='color:rgba(255,255,255,0.6);'>ℹ What this tab reveals</b><br>"
            "A multi-dimensional analysis of how "
            "<b style='color:#F59E0B;'>family role & gender</b>, "
            "<b style='color:#F59E0B;'>age × education</b>, "
            "<b style='color:#F59E0B;'>education composition</b>, "
            "<b style='color:#F59E0B;'>working hours patterns</b>, and "
            "<b style='color:#F59E0B;'>age × occupation</b> jointly shape earning potential. "
            "Cross-tab heatmaps, stacked bars, and bubble charts reveal which "
            "career paths and demographic profiles yield the highest income probability."
        )

        # Cross-tab heatmaps: Relationship/Marital × Sex
        _render_section3(df_binned, cols, income_col)

        _row_spacer()

        # Cross-tab heatmap: Age Group × Education
        _render_section3b(df_binned, cols, income_col)

        _row_spacer()
        _render_section7(df_binned, cols)

        _row_spacer()

        # Occ×Hours Bubble (full-width)
        _render_section9(df_binned, cols, income_col)

        _row_spacer()

        # Age×Occ High Income Rate heatmap (full-width)
        _render_section10(df_binned, cols, income_col)

    # =================================================================
    # TAB 3 — Capital Gain Analysis
    # =================================================================
    with tabs[2]:
        _tab_summary(
            "<b style='color:rgba(255,255,255,0.6);'>ℹ What this tab reveals</b><br>"
            "How <b style='color:#F59E0B;'>capital gain</b> acts as a wealth multiplier beyond salary. "
            "Examines the <b style='color:#F59E0B;'>distribution gap</b> between income classes, "
            "<b style='color:#F59E0B;'>prevalence by education × occupation</b>, "
            "and <b style='color:#F59E0B;'>dual-earner concentration</b> across "
            "marital status and career paths."
        )

        # Capital Gain vs Income Level
        _render_section5(df_raw, cols, income_col)

        _row_spacer()

        # CapGain >0 × Education × Occupation (dual heatmap)
        _render_section8(df_raw, df_binned, cols, income_col)

        _row_spacer()

        # CapGain by Marital Status (full-width)
        _render_section12(df_raw, cols, income_col)

        _row_spacer()

        # CapGain by Occupation (full-width)
        _render_section13(df_raw, cols, income_col)

    # =================================================================
    # TAB 4 — Gender Disparity
    # =================================================================
    with tabs[3]:
        _tab_summary(
            "<b style='color:rgba(255,255,255,0.6);'>ℹ What this tab reveals</b><br>"
            "A multi-dimensional analysis of <b style='color:#F59E0B;'>gender-based income disparity</b>. "
            "Compares male vs. female earning patterns at the overall level, "
            "within individual <b style='color:#F59E0B;'>occupations</b>, and through the lens of "
            "<b style='color:#F59E0B;'>capital gain</b> — quantifying where gender gaps "
            "are widest and where they narrow."
        )

        # Sex vs Income Level (dual donut)
        _render_section6(df_raw, cols, income_col)

        _row_spacer()

        # Sex × Occupation (grouped bar)
        _render_section11(df_raw, cols, income_col)

        _row_spacer()

        # Income & CapGain by Sex (2 stacked bars Male/Female)
        _render_section14(df_raw, cols, income_col)

    # =================================================================
    # TAB 5 — Archetype Profile
    # =================================================================
    with tabs[4]:
        _tab_summary(
            "<b style='color:rgba(255,255,255,0.6);'>ℹ What this tab reveals</b><br>"
            "The <b style='color:#F59E0B;'>composite profile</b> of a typical high-income individual "
            "with capital gain — synthesizing insights from all previous tabs into a single "
            "<b style='color:#F59E0B;'>actionable archetype</b>. Use this as the concluding "
            "reference for demographic targeting and policy recommendations."
        )

        # Typical High-Income Profile (summary + chart)
        _render_section15(df_raw, df_binned, cols, income_col)

    section_divider()


if __name__ == "__main__":
    main()
