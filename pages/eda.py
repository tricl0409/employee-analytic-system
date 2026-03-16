"""
eda.py — Employee Data Insight Page (EDA)

Slide 1 Layout (6 chart sections):
  ┌─────────────────┬──────────────────────┬──────────────────────┐
  │  SECTION 1       │  SECTION 2            │  SECTION 3           │
  │  Donut: Income   │  Heatmap: Association │  Age × Occ bars      │
  │  Split           │  Cramér's V + PB      │  + Insight            │
  ├─────────────────┴──────────────────────┴──────────────────────┤
  │  SECTION 4 (3/5):  Top Impacting to High Income               │
  │  4 sub-charts + dynamic bullet insights                        │
  │                      │  SECTION 5: Capital │  SECTION 6: Sex     │
  │                      │  Gain vs Income     │  vs Income Donut    │
  └──────────────────────┴─────────────────────┴────────────────────┘
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

_C_HIGH = STATUS_COLORS["warning"]["hex"]   # Amber  — high earner (>50K)
_C_STD  = STATUS_COLORS["neutral"]["hex"]   # Blue   — standard earner (≤50K)

_STRIP_KEYS = {"legend", "margin"}

# Section header accent (amber — consistent with insight boxes)
_ACCENT_AMBER = "#FF9F43"

# Education color palette — gradient from low → high education
_EDU_COLORS = {
    "Basic":      "rgba(148,163,184,0.75)",   # Slate — lowest level
    "HS-grad":    "rgba(56,189,248,0.75)",     # Cyan — mid-low
    "SomeAssoc":  "rgba(99,102,241,0.75)",     # Indigo — mid
    "Some/Assoc": "rgba(99,102,241,0.75)",     # Indigo (alias)
    "Bachelors":  "rgba(52,211,153,0.80)",     # Emerald — mid-high
    "Advanced":   "rgba(251,191,36,0.85)",     # Amber — highest
}


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

    Cards: Dataset Scale | High Earner % | Median Age
           Gender Ratio  | Avg Hours/Wk | Age Gap (Hi vs Std)
    """
    total = len(df)
    n_cols = len(df.columns)
    size_mb = df.memory_usage(deep=True).sum() / (1024 ** 2)

    # High earner stats
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
        metric_card("High Earner (>50K)", f"{pct_high}%",
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
    """Donut: High Earner (>50K) vs Standard Earner (≤50K)."""
    hi_mask = _high_mask(df[income_col])
    n_high = int(hi_mask.sum())
    n_std = len(df) - n_high

    fig = go.Figure(go.Pie(
        labels=[f"Standard Earner (≤50K)\n{n_std:,} Individuals",
                f"High Earner (>50K)\n{n_high:,} Individuals"],
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
        "Income Distribution by Income Threshold",
        subtitle="Proportion of employees in each income bracket",
        icon_name="eye",
    )

    hi_mask = _high_mask(df[income_col])
    n_high = int(hi_mask.sum())
    n_std = len(df) - n_high
    pct_high = round(n_high / len(df) * 100, 1)
    pct_std = round(100 - pct_high, 1)

    st.plotly_chart(
        _chart_donut(df, income_col),
        use_container_width=True, key="ch_donut",
    )

    # Dynamic insight
    majority = "lower incomes (≤50K)" if pct_std > pct_high else "higher incomes (>50K)"
    st.markdown(
        _insight_box(
            f"The distribution is skewed toward <b>{majority}</b>. "
            f"<b>{pct_std}%</b> earn ≤50K ({n_std:,} individuals) vs "
            f"<b>{pct_high}%</b> earn >50K ({n_high:,} individuals). "
            f"This indicates income inequality and an imbalance between the two income groups."
        ),
        unsafe_allow_html=True,
    )


# ==============================================================================
# SECTION 2 — Association Heatmap (Cramér's V + Point-Biserial)
# ==============================================================================

_ASSOC_ATTRIBUTES = [
    "marital", "relationship", "education_num", "education",
    "hours", "age", "capital_gain", "sex", "occupation",
]

_ASSOC_DISPLAY = {
    "marital": "marital_status",
    "relationship": "relationship",
    "education_num": "education_num",
    "education": "education",
    "hours": "hours_per_week",
    "age": "age",
    "capital_gain": "capital_gain",
    "sex": "sex",
    "occupation": "occupation",
}


def _compute_association_scores(
    df: pd.DataFrame,
    cols: dict[str, str | None],
    income_col: str,
) -> pd.DataFrame:
    """
    Compute association strength of each attribute with binary income.

    Method selection:
      - Numeric features → Point-Biserial (= Pearson vs binary)
      - Categorical features → Cramér's V

    Returns DataFrame with columns: attribute, association, method
    sorted by association descending.
    """
    hi_binary = _high_mask(df[income_col]).astype(float)
    rows = []

    for attr_key in _ASSOC_ATTRIBUTES:
        actual_col = cols.get(attr_key)
        if actual_col is None or actual_col not in df.columns:
            continue

        series = df[actual_col].dropna()
        if len(series) < 10:
            continue

        # Choose method based on dtype
        if pd.api.types.is_numeric_dtype(series):
            numeric_vals = pd.to_numeric(df[actual_col], errors="coerce")
            score = abs(_point_biserial(numeric_vals, hi_binary))
            method_label = "Point-Biserial"
        else:
            # Align indices for Cramér's V
            valid_idx = df[actual_col].notna() & df[income_col].notna()
            score = _cramers_v(
                df.loc[valid_idx, actual_col].astype(str),
                hi_binary[valid_idx].astype(int).astype(str),
            )
            method_label = "Cramér's V"

        rows.append({
            "attribute": _ASSOC_DISPLAY.get(attr_key, attr_key),
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
    cols: dict[str, str | None],
    income_col: str,
) -> None:
    """Render Association Heatmap (Cramér's V + Point-Biserial) + Insight."""
    _section_header(
        "Top Attributes Impacting to High Income",
        subtitle="Association strength measured by Cramér's V and Point-Biserial",
        icon_name="target",
    )

    assoc_df = _compute_association_scores(df, cols, income_col)

    if assoc_df.empty:
        styled_alert("Insufficient data to compute associations.", "info")
        return

    # Subtitle
    st.markdown(
        "<div style='text-align:center;font-size:0.82rem;color:rgba(255,255,255,0.55);"
        "margin-bottom:8px;font-weight:600;'>"
        "Association strength with Income (Cramér's V for categorical, Point-Biserial for numeric)"
        "</div>",
        unsafe_allow_html=True,
    )

    st.plotly_chart(
        _chart_association_heatmap(assoc_df),
        use_container_width=True, key="ch_assoc_heatmap",
    )

    # Dynamic insight: pick top-3 strongest predictors
    top3 = assoc_df.head(3)["attribute"].tolist()
    if len(top3) >= 3:
        top3_bold = ", ".join(f"<b>{a}</b>" for a in top3[:2]) + f", and <b>{top3[2]}</b>"
    else:
        top3_bold = ", ".join(f"<b>{a}</b>" for a in top3)

    st.markdown(
        _insight_box(
            f"The findings show that high income is not driven by a single factor "
            f"but by a combination of structural advantages. "
            f"Among all variables, {top3_bold} emerge as the "
            f"strongest predictors of earning >50K."
        ),
        unsafe_allow_html=True,
    )


# ==============================================================================
# SECTION 3 — High Income by Age & Occupation
# ==============================================================================

def _render_section3(
    df: pd.DataFrame,
    df_binned: pd.DataFrame,
    cols: dict[str, str | None],
    income_col: str,
) -> None:
    """Render Age bins bar + Occupation bar side by side + Insight."""
    _section_header(
        "Employees Earned High Income by Age and Occupation",
        subtitle="% of employees earning >50K across age groups and occupations",
        icon_name="briefcase",
    )

    hi_mask = _high_mask(df[income_col])
    age_col = cols.get("age")
    occ_col = cols.get("occupation")

    # Pre-compute age rates (reused in chart + insight)
    rate_by_age = None
    if age_col and age_col in df_binned.columns:
        binned_age = df_binned[age_col].astype(str)
        rate_by_age = hi_mask.groupby(binned_age).mean().sort_index()

    col_age, col_occ = st.columns(2, gap="medium")

    # ── Age bins bar chart ────────────────────────────────────────────────
    with col_age:
        if rate_by_age is not None:

            fig_age = go.Figure(go.Bar(
                x=rate_by_age.index.tolist(),
                y=(rate_by_age.values * 100).round(1),
                marker=dict(
                    color=[f"rgba(255,159,67,{0.3 + 0.7 * v / max(rate_by_age.max(), 0.01):.2f})"
                           for v in rate_by_age.values],
                    line=dict(width=0),
                ),
                text=[f"{v:.1f}%" for v in rate_by_age.values * 100],
                textposition="outside",
                textfont=dict(color=MUTED_COLOR, size=10),
                hovertemplate="<b>%{x}</b><br>Earning >50K: <b>%{y:.1f}%</b><extra></extra>",
            ))
            fig_age.update_layout(
                **_base_layout(),
                height=300,
                showlegend=False,
                margin=dict(l=50, r=20, t=30, b=50),
                title=dict(
                    text="% Earning >50K by Age Group",
                    font=dict(size=11, color=MUTED_COLOR),
                    x=0.5, xanchor="center",
                ),
                xaxis=dict(
                    title=dict(text="Age Group", font=dict(color=MUTED_COLOR, size=10)),
                    tickfont=dict(color=MUTED_COLOR, size=9),
                ),
                yaxis=dict(
                    title=dict(text="% Earning >50K", font=dict(color=MUTED_COLOR, size=10)),
                    tickfont=dict(color=MUTED_COLOR, size=9),
                    gridcolor=GRID_COLOR,
                ),
            )
            st.plotly_chart(apply_global_theme(fig_age), use_container_width=True, key="ch_age_bar")

    # ── Occupation horizontal bar chart ───────────────────────────────────
    with col_occ:
        if occ_col and occ_col in df.columns:
            rate_by_occ = hi_mask.groupby(df[occ_col].astype(str)).mean()
            rate_by_occ = rate_by_occ.sort_values(ascending=True)
            mx_occ = max(rate_by_occ.max(), 0.01)

            fig_occ = go.Figure(go.Bar(
                y=rate_by_occ.index.tolist(),
                x=(rate_by_occ.values * 100).round(1),
                orientation="h",
                marker=dict(
                    color=[f"rgba(255,159,67,{0.3 + 0.7 * v / mx_occ:.2f})"
                           for v in rate_by_occ.values],
                    line=dict(width=0),
                ),
                text=[f"{v:.1f}%" for v in rate_by_occ.values * 100],
                textposition="outside",
                textfont=dict(color=MUTED_COLOR, size=10),
                cliponaxis=False,
                hovertemplate="<b>%{y}</b><br>Earning >50K: <b>%{x:.1f}%</b><extra></extra>",
            ))
            fig_occ.update_layout(
                **_base_layout(),
                height=420,
                showlegend=False,
                margin=dict(l=130, r=50, t=30, b=30),
                title=dict(
                    text="% Earning >50K by Occupation",
                    font=dict(size=11, color=MUTED_COLOR),
                    x=0.5, xanchor="center",
                ),
                xaxis=dict(
                    title=dict(text="% Earning >50K", font=dict(color=MUTED_COLOR, size=10)),
                    tickfont=dict(color=MUTED_COLOR, size=9),
                    gridcolor=GRID_COLOR,
                ),
                yaxis=dict(
                    tickfont=dict(color=MUTED_COLOR, size=10),
                ),
            )
            st.plotly_chart(apply_global_theme(fig_occ), use_container_width=True, key="ch_occ_bar")

    # ── Age×Occ insight ───────────────────────────────────────────────────
    if rate_by_age is not None:
        peak_age = rate_by_age.idxmax()
        peak_pct = round(rate_by_age.max() * 100, 1)

        insight_text = (
            f"Income >50K increases with age and experience. "
            f"It reaches its peak among the <b>{peak_age}</b> age group "
            f"(<b>{peak_pct}%</b>)."
        )
        if occ_col and occ_col in df.columns:
            rate_by_occ_vals = hi_mask.groupby(df[occ_col].astype(str)).mean()
            top_occ = rate_by_occ_vals.idxmax()
            insight_text += (
                f" Occupations like <b>{top_occ}</b> tend to have "
                f"a higher share of employees earning >50K."
            )
        st.markdown(_insight_box(insight_text), unsafe_allow_html=True)


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
            title=dict(text="High Income rate", font=dict(color=MUTED_COLOR, size=10)),
            tickfont=dict(color=MUTED_COLOR, size=9),
            gridcolor=GRID_COLOR,
            range=[0, min(mx * 1.35, 1.0)],
        ),
        yaxis=dict(
            tickfont=dict(color=MUTED_COLOR, size=10),
        ),
    )
    return apply_global_theme(fig)


def _chart_hours_trend(
    df: pd.DataFrame,
    income_col: str,
    hours_col: str,
) -> go.Figure:
    """Line chart: high-income rate by hours_per_week quantile."""
    df_temp = df[[hours_col, income_col]].dropna().copy()
    df_temp["hi"] = _high_mask(df_temp[income_col])
    df_temp["hours_num"] = pd.to_numeric(df_temp[hours_col], errors="coerce")
    df_temp = df_temp.dropna(subset=["hours_num"])

    # Create quantile bins
    df_temp["q_bin"] = pd.qcut(df_temp["hours_num"], q=4, duplicates="drop")
    rate = df_temp.groupby("q_bin", observed=False)["hi"].mean()
    rate = rate.sort_index()

    labels = [str(interval) for interval in rate.index]
    values = rate.values

    fig = go.Figure()

    # Line + markers
    fig.add_trace(go.Scatter(
        x=labels,
        y=values,
        mode="lines+markers+text",
        line=dict(color=_C_HIGH, width=2.5),
        marker=dict(size=8, color=_C_HIGH, line=dict(color="rgba(255,255,255,0.3)", width=1)),
        text=[f"{v:.2f}" for v in values],
        textposition="top center",
        textfont=dict(size=10, color=BRIGHT_TEXT),
        hovertemplate="<b>%{x}</b><br>High Income rate: <b>%{y:.2f}</b><extra></extra>",
    ))

    # Trend line
    x_num = np.arange(len(values))
    if len(x_num) >= 2:
        coeffs = np.polyfit(x_num, values, 1)
        trend_y = np.polyval(coeffs, x_num)
        fig.add_trace(go.Scatter(
            x=labels,
            y=trend_y,
            mode="lines",
            line=dict(color="rgba(239,68,68,0.5)", width=1.5, dash="dash"),
            hoverinfo="skip",
            showlegend=False,
        ))

    fig.update_layout(
        **_base_layout(),
        height=280,
        showlegend=False,
        margin=dict(l=50, r=30, t=30, b=60),
        title=dict(
            text="High Income Trend by Hours per Week (quantile)",
            font=dict(size=11, color=MUTED_COLOR),
            x=0.5, xanchor="center",
        ),
        xaxis=dict(
            title=dict(
                text="Working hours group (quantile)",
                font=dict(color=MUTED_COLOR, size=10),
            ),
            tickfont=dict(color=MUTED_COLOR, size=8),
            tickangle=-15,
        ),
        yaxis=dict(
            title=dict(text="High Income rate", font=dict(color=MUTED_COLOR, size=10)),
            tickfont=dict(color=MUTED_COLOR, size=9),
            gridcolor=GRID_COLOR,
        ),
    )
    return apply_global_theme(fig)


def _compute_dynamic_insights(
    df: pd.DataFrame,
    df_binned: pd.DataFrame,
    cols: dict[str, str | None],
    income_col: str,
) -> list[str]:
    """
    Compute 4 dynamic bullet insights for Section 4.

    Returns list of HTML strings (each is one bullet point).
    """
    hi_mask = _high_mask(df[income_col])
    bullets = []

    # 1. Marital Status insight
    marital_col = cols.get("marital")
    if marital_col and marital_col in df.columns:
        marital_rate = hi_mask.groupby(df[marital_col].astype(str)).mean()
        top_marital = marital_rate.idxmax()
        top_rate = round(marital_rate.max() * 100, 1)
        bullets.append(
            f"<b>{top_marital}</b> individuals have the highest likelihood of earning >50K "
            f"(<b>{top_rate}%</b>), suggesting a link between family stability and career focus."
        )

    # 2. Relationship insight
    rel_col = cols.get("relationship")
    if rel_col and rel_col in df.columns:
        rel_rate = hi_mask.groupby(df[rel_col].astype(str)).mean()
        top2_rel = rel_rate.nlargest(2)
        top_names = " and ".join(f"<b>{n}</b>" for n in top2_rel.index)
        bullets.append(
            f"High-income earners are mainly in core household roles ({top_names}), "
            f"reflecting stages of the economic life cycle."
        )

    # 3. Education insight (uses binned data for group labels)
    edu_col = cols.get("education")
    if edu_col and edu_col in df_binned.columns:
        edu_rate = hi_mask.groupby(df_binned[edu_col].astype(str)).mean()
        top_edu = edu_rate.idxmax()
        top_edu_rate = round(edu_rate.max() * 100, 1)
        above_threshold = edu_rate[edu_rate > 0.30].index.tolist()
        threshold_text = (
            f" with a clear threshold at <b>{', '.join(above_threshold[:2])}</b> or higher"
            if above_threshold else ""
        )
        bullets.append(
            f"<b>Education is a key structural driver of income</b>{threshold_text}. "
            f"<b>{top_edu}</b> achieves the highest rate at <b>{top_edu_rate}%</b>."
        )

    # 4. Hours insight
    hours_col = cols.get("hours")
    if hours_col and hours_col in df.columns:
        hours_numeric = pd.to_numeric(df[hours_col], errors="coerce")
        valid_mask = hours_numeric.notna()
        if valid_mask.sum() > 10:
            corr = round(hours_numeric[valid_mask].corr(hi_mask[valid_mask].astype(float)), 3)
            strength = "weak" if abs(corr) < 0.15 else ("moderate" if abs(corr) < 0.35 else "strong")
            bullets.append(
                f"Working longer hours helps, but work intensity alone does not guarantee "
                f"high income (r = <b>{corr}</b>, {strength} correlation)."
            )

    return bullets


def _render_section4(
    df: pd.DataFrame,
    df_binned: pd.DataFrame,
    cols: dict[str, str | None],
    income_col: str,
) -> None:
    """Render Top Impacting to High Income — 4 sub-charts + dynamic bullet insights."""
    _section_header(
        "Top Impacting to High Income",
        subtitle="High-income ratio breakdown by key demographic attributes",
        icon_name="zap",
    )

    # Resolve columns
    marital_col = cols.get("marital")
    rel_col = cols.get("relationship")
    edu_col = cols.get("education")
    hours_col = cols.get("hours")

    col_charts, col_insights = st.columns([3, 2], gap="medium")

    with col_charts:
        # Row 1: Marital Status + Relationship
        c1, c2 = st.columns(2)
        with c1:
            if marital_col and marital_col in df.columns:
                fig_marital = _chart_hbar_rate(
                    df, income_col, marital_col,
                    title="High Income Rate by Marital Status",
                )
                st.plotly_chart(fig_marital, use_container_width=True, key="ch_marital")
        with c2:
            if rel_col and rel_col in df.columns:
                fig_rel = _chart_hbar_rate(
                    df, income_col, rel_col,
                    title="High Income Rate by Relationship",
                )
                st.plotly_chart(fig_rel, use_container_width=True, key="ch_relationship")

        # Row 2: Education (binned) + Hours Trend
        c3, c4 = st.columns(2)
        with c3:
            if edu_col and edu_col in df_binned.columns:
                fig_edu = _chart_hbar_rate(
                    df_binned, income_col, edu_col,
                    title="High Income Rate by Education",
                )
                st.plotly_chart(fig_edu, use_container_width=True, key="ch_education")
        with c4:
            if hours_col and hours_col in df.columns:
                fig_hours = _chart_hours_trend(df, income_col, hours_col)
                st.plotly_chart(fig_hours, use_container_width=True, key="ch_hours_trend")

    with col_insights:
        bullets = _compute_dynamic_insights(df, df_binned, cols, income_col)

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
        "Capital Gain vs. Income Level",
        subtitle="Average capital gain comparison between income groups",
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
    std_has_cg = (capgain[~hi_mask] > 0).sum()
    high_has_cg = (capgain[hi_mask] > 0).sum()
    std_pct_cg = round(std_has_cg / (~hi_mask).sum() * 100, 1) if (~hi_mask).sum() else 0
    high_pct_cg = round(high_has_cg / hi_mask.sum() * 100, 1) if hi_mask.sum() else 0

    fig = go.Figure()

    # Stacked bars: base salary vs capital gain (conceptual)
    fig.add_trace(go.Bar(
        x=["≤50K", ">50K"],
        y=[std_mean, high_mean],
        name="Avg Capital Gain",
        marker=dict(color="rgba(255,159,67,0.7)"),
        text=[f"${std_mean:,.0f}", f"${high_mean:,.0f}"],
        textposition="inside",
        textfont=dict(size=11, color=BRIGHT_TEXT),
        hovertemplate="<b>%{x}</b><br>Avg Capital Gain: <b>$%{y:,.0f}</b><extra></extra>",
    ))

    fig.update_layout(
        **_base_layout(),
        height=300,
        showlegend=False,
        margin=dict(l=40, r=20, t=30, b=40),
        title=dict(
            text="Avg Capital Gain by Income Bracket",
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
            f"High Earners (>50K) have significantly higher capital gains. "
            f"<b>{high_pct_cg}%</b> of high earners have capital gain > 0 vs only "
            f"<b>{std_pct_cg}%</b> of standard earners. "
            f"They act as <b>Capital Masters</b> beyond base salary."
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
        "Occupation vs. Education",
        subtitle="Education group composition within each occupation category",
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

    fig = go.Figure()
    for edu_group in ct.columns:
        color = _EDU_COLORS.get(edu_group, "rgba(148,163,184,0.5)")
        fig.add_trace(go.Bar(
            y=ct.index.tolist(),
            x=ct[edu_group].round(1).values,
            name=edu_group,
            orientation="h",
            marker=dict(color=color),
            text=[f"{v:.0f}%" for v in ct[edu_group].values],
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

    st.markdown(
        _insight_box(
            "There is a clear positive relationship between education level and "
            "occupational hierarchy. <b>Higher education</b> leads to "
            "<b>Management / Professional</b> roles, while lower or mid-level "
            "education leads to Service, Blue-collar, or manual occupations."
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
        "Capital Gain > 0 vs. Education × Occupation",
        subtitle="% of employees with capital gain, split by income bracket",
        icon_name="eye",
    )

    occ_col = cols.get("occupation")
    edu_col = cols.get("education")
    capgain_col = cols.get("capital_gain")
    if not all(c and c in df.columns for c in [occ_col, edu_col, capgain_col]):
        styled_alert("Requires occupation, education, and capital_gain columns.", "info")
        return

    hi_mask = _high_mask(df[income_col])
    capgain = pd.to_numeric(df[capgain_col], errors="coerce")
    has_cg = capgain > 0
    edu_binned = df_binned[edu_col].astype(str)

    heatmap_configs = [
        (~hi_mask, "Income ≤50K"),
        (hi_mask, "Income >50K"),
    ]

    for idx, (mask, label) in enumerate(heatmap_configs):
        sub_df = df[mask].copy()
        sub_edu = edu_binned[mask]
        sub_cg = has_cg[mask]

        ct_total = pd.crosstab(sub_edu, sub_df[occ_col].astype(str))
        ct_cg = pd.crosstab(sub_edu, sub_df[occ_col].astype(str), values=sub_cg, aggfunc="sum")
        pct = (ct_cg / ct_total.replace(0, np.nan) * 100).fillna(0).round(1)

        fig = go.Figure(go.Heatmap(
            z=pct.values,
            x=pct.columns.tolist(),
            y=pct.index.tolist(),
            text=[[f"{v:.1f}%" for v in row] for row in pct.values],
            texttemplate="%{text}",
            textfont=dict(size=9, color="rgba(255,255,255,0.85)"),
            colorscale=[
                [0.0, "rgba(255,255,255,0.03)"],
                [0.5, "rgba(255,159,67,0.30)"],
                [1.0, "rgba(255,159,67,0.75)"],
            ],
            zmin=0,
            zmax=max(pct.values.max().max(), 1),
            showscale=False,
            hovertemplate="<b>%{y}</b> × <b>%{x}</b><br>%{z:.1f}% have CapGain<extra></extra>",
        ))
        fig.update_layout(
            **_base_layout(),
            height=280,
            margin=dict(l=80, r=30, t=30, b=80),
            title=dict(
                text=f"% CapGain (>0) | {label}",
                font=dict(size=11, color=MUTED_COLOR),
                x=0.5, xanchor="center",
            ),
            xaxis=dict(tickfont=dict(color=MUTED_COLOR, size=9), tickangle=-35),
            yaxis=dict(tickfont=dict(color=MUTED_COLOR, size=9), autorange="reversed"),
        )
        st.plotly_chart(apply_global_theme(fig), use_container_width=True, key=f"ch_cg_{label}")

        if idx == 0:
            _row_spacer()

    st.markdown(
        _insight_box(
            "High Earners (>50K) are often <b>Capital Masters</b>. "
            "They do not rely solely on salary but also generate significant capital gains, "
            "especially in <b>Management/Professional</b> and <b>Advanced education</b> groups."
        ),
        unsafe_allow_html=True,
    )


# ==============================================================================
# SECTION 9 — Occupation × Working Hours (Bubble)
# ==============================================================================

def _render_section9(
    df: pd.DataFrame,
    df_binned: pd.DataFrame,
    cols: dict[str, str | None],
) -> None:
    """Bubble chart: Occupation × Hours bins, size = count."""
    _section_header(
        "Correlation between Occupation & Working Hours",
        subtitle="Bubble size represents employee count in each group",
        icon_name="clock",
    )

    occ_col = cols.get("occupation")
    hours_col = cols.get("hours")
    if not occ_col or occ_col not in df.columns:
        styled_alert("No occupation column found.", "info")
        return
    if not hours_col or hours_col not in df_binned.columns:
        styled_alert("No hours column found.", "info")
        return

    # Use binned hours if available, else create quantile bins
    hrs_binned = df_binned[hours_col].astype(str)
    occ = df[occ_col].astype(str)

    ct = pd.crosstab(occ, hrs_binned)
    max_count = ct.values.max() if ct.values.max() > 0 else 1

    # Flatten crosstab into arrays for a single scatter trace (performance)
    x_vals, y_vals, sizes, texts, hovers = [], [], [], [], []
    for occ_name in ct.index:
        for hrs_label in ct.columns:
            count = ct.loc[occ_name, hrs_label]
            if count == 0:
                continue
            x_vals.append(hrs_label)
            y_vals.append(occ_name)
            sizes.append(max(8, count / max_count * 55))
            texts.append(f"{count:,}")
            hovers.append(
                f"<b>{occ_name}</b><br>"
                f"Hours: <b>{hrs_label}</b><br>"
                f"Count: <b>{count:,}</b>"
                "<extra></extra>"
            )

    fig = go.Figure(go.Scatter(
        x=x_vals,
        y=y_vals,
        mode="markers+text",
        marker=dict(
            size=sizes,
            color="rgba(255,159,67,0.6)",
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
        height=550,
        margin=dict(l=150, r=30, t=20, b=60),
        xaxis=dict(
            title=dict(text="Working Hours Group", font=dict(color=MUTED_COLOR, size=10)),
            tickfont=dict(color=MUTED_COLOR, size=9),
        ),
        yaxis=dict(tickfont=dict(color=MUTED_COLOR, size=10)),
    )
    st.plotly_chart(apply_global_theme(fig), use_container_width=True, key="ch_occ_hours_bubble")

    st.markdown(
        _insight_box(
            "Working more hours does not always guarantee higher income, "
            "but <b>high earners</b> often tend to work more. "
            "<b>Management/Professional</b> occupations concentrate in "
            "higher working hour brackets."
        ),
        unsafe_allow_html=True,
    )


# ==============================================================================
# SECTION 10 — %Employees >50K by Age × Occupation (Heatmap)
# ==============================================================================

def _render_section10(
    df: pd.DataFrame,
    df_binned: pd.DataFrame,
    cols: dict[str, str | None],
    income_col: str,
) -> None:
    """%>50K heatmap: Age bins (rows) × Occupation (cols)."""
    _section_header(
        "%Employees Earned >50K by Age × Occupation",
        subtitle="Percentage of employees earning >50K in each age-occupation cell",
        icon_name="target",
    )

    age_col = cols.get("age")
    occ_col = cols.get("occupation")
    if not age_col or age_col not in df_binned.columns:
        styled_alert("No age column found.", "info")
        return
    if not occ_col or occ_col not in df.columns:
        styled_alert("No occupation column found.", "info")
        return

    hi_mask = _high_mask(df[income_col])
    age_binned = df_binned[age_col].astype(str)
    occ = df[occ_col].astype(str)

    ct_total = pd.crosstab(age_binned, occ)
    ct_hi = pd.crosstab(age_binned, occ, values=hi_mask, aggfunc="sum")
    pct = (ct_hi / ct_total.replace(0, np.nan) * 100).fillna(0).round(1)

    fig = go.Figure(go.Heatmap(
        z=pct.values,
        x=pct.columns.tolist(),
        y=pct.index.tolist(),
        text=[[f"{v:.1f}%" for v in row] for row in pct.values],
        texttemplate="%{text}",
        textfont=dict(size=10, color="rgba(255,255,255,0.9)"),
        colorscale=[
            [0.0, "rgba(255,255,255,0.03)"],
            [0.3, "rgba(59,130,246,0.25)"],
            [0.6, "rgba(255,159,67,0.45)"],
            [1.0, "rgba(239,68,68,0.70)"],
        ],
        zmin=0,
        zmax=max(pct.values.max().max(), 1),
        showscale=True,
        colorbar=dict(
            title=dict(text="% >50K", font=dict(size=10, color=MUTED_COLOR)),
            tickfont=dict(size=9, color=MUTED_COLOR),
            len=0.8, thickness=12,
        ),
        hovertemplate="<b>Age: %{y}</b> × <b>%{x}</b><br>%{z:.1f}% earn >50K<extra></extra>",
    ))

    fig.update_layout(
        **_base_layout(),
        height=320,
        margin=dict(l=80, r=60, t=20, b=80),
        xaxis=dict(tickfont=dict(color=MUTED_COLOR, size=9), tickangle=-35),
        yaxis=dict(tickfont=dict(color=MUTED_COLOR, size=10), autorange="reversed"),
    )
    st.plotly_chart(apply_global_theme(fig), use_container_width=True, key="ch_age_occ_heatmap")

    # Find peak cell
    if pct.values.max().max() > 0:
        max_idx = np.unravel_index(pct.values.argmax(), pct.values.shape)
        peak_age = pct.index[max_idx[0]]
        peak_occ = pct.columns[max_idx[1]]
        peak_val = pct.values[max_idx[0], max_idx[1]]
        st.markdown(
            _insight_box(
                f"Both age (experience) and occupation type strongly influence income. "
                f"The peak is at <b>{peak_age}</b> × <b>{peak_occ}</b> "
                f"with <b>{peak_val:.1f}%</b> earning >50K. "
                f"Higher-skilled or managerial roles and middle-age workers have the highest proportions."
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
        "Income & Capital Gain by Marital Status",
        subtitle="4-segment breakdown: Income bracket × Capital Gain presence",
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

    st.markdown(
        _insight_box(
            "Individuals in the <b>Married</b> group tend to have higher and more stable "
            "incomes. This suggests a potential link between <b>family stability</b> and "
            "greater focus on long-term career development."
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
        "Income & Capital Gain by Occupation",
        subtitle="Which occupations combine high income with capital gain?",
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

    st.markdown(
        _insight_box(
            "<b>Management and Professional</b> occupations are becoming "
            "'promising fields' for those who want to improve their income status. "
            "They show the highest proportion of employees with both high income and capital gain."
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
        subtitle="Characteristics of the typical >50K earner with capital gain",
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
        ":material/work: Career & Demographics",
        ":material/attach_money: Capital Gain Analysis",
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
            "A bird's-eye view of the dataset's <b style='color:#F59E0B;'>income distribution</b> "
            "and the <b style='color:#F59E0B;'>statistical association</b> between each attribute "
            "and high-income status. Use this tab to identify which factors matter most before "
            "diving into cross-feature analysis."
        )

        # Income Donut | Correlation Heatmap
        col_t1a, col_t1b = st.columns(2, gap="medium")
        with col_t1a:
            _render_section1(df_raw, income_col)
        with col_t1b:
            _render_section2(df_raw, cols, income_col)

        _row_spacer()

        # Top Impacting to High Income (4 sub-charts + insights)
        _render_section4(df_raw, df_binned, cols, income_col)

    # =================================================================
    # TAB 2 — Career & Demographics
    # =================================================================
    with tabs[1]:
        _tab_summary(
            "<b style='color:rgba(255,255,255,0.6);'>ℹ What this tab reveals</b><br>"
            "How <b style='color:#F59E0B;'>age</b>, <b style='color:#F59E0B;'>occupation</b>, "
            "<b style='color:#F59E0B;'>education level</b>, and "
            "<b style='color:#F59E0B;'>working hours</b> interact to influence earning potential. "
            "Discover which career paths and demographic profiles are most strongly associated "
            "with high income."
        )

        # High Income by Age & Occupation (dual bar)
        _render_section3(df_raw, df_binned, cols, income_col)

        _row_spacer()

        # Occupation vs Education (100% stacked)
        _render_section7(df_binned, cols)

        _row_spacer()

        # Occ×Hours Bubble (full-width)
        _render_section9(df_raw, df_binned, cols)

        _row_spacer()

        # Age×Occ %Heatmap (full-width)
        _render_section10(df_raw, df_binned, cols, income_col)

    # =================================================================
    # TAB 3 — Capital Gain Analysis
    # =================================================================
    with tabs[2]:
        _tab_summary(
            "<b style='color:rgba(255,255,255,0.6);'>ℹ What this tab reveals</b><br>"
            "The role of <b style='color:#F59E0B;'>capital gain</b> as an income multiplier. "
            "Examines how non-salary investment income varies across "
            "<b style='color:#F59E0B;'>education</b>, <b style='color:#F59E0B;'>occupation</b>, "
            "and <b style='color:#F59E0B;'>marital status</b> — revealing whether capital gain "
            "is a universal wealth signal or concentrated in specific segments."
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
            "The <b style='color:#F59E0B;'>composite profile</b> of a typical high-income earner "
            "with capital gain — synthesizing insights from all previous tabs into a single "
            "<b style='color:#F59E0B;'>actionable archetype</b>. Use this as the concluding "
            "reference for demographic targeting and policy recommendations."
        )

        # Typical High-Income Profile (summary + chart)
        _render_section15(df_raw, df_binned, cols, income_col)

    section_divider()


if __name__ == "__main__":
    main()
