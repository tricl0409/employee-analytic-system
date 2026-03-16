"""
eda.py — Employee Data Insight Page (EDA)

Workflow:
  1. Load active CSV via data_engine
  2. Load binning_config from DB and apply non-destructively:
       - Numeric columns → pd.cut (binning)
       - Categorical columns → value map (grouping)
     Result saved to temp/ for debug — original file UNTOUCHED
  3. All charts & insights driven from transformed df

Layout:
  ┌──────────────────┬──────────────────────┐
  │  ROW 1           │                      │
  │  Donut:          │  Bar: %>50K by       │
  │  Income Split    │  Age Group (binned)  │
  ├──────────────────┼──────────────────────┤
  │  ROW 2           │                      │
  │  HBar: %>50K by  │  Heatmap:            │
  │  Occupation Map  │  Age × Occupation    │
  ├──────────────────┴──────────────────────┤
  │  ROW 3 (full-width)                     │
  │  Bubble: Occupation × Working Hours     │
  └─────────────────────────────────────────┘
"""

import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from modules.core import data_engine
from modules.ui import (
    page_header, workspace_status,
    active_file_scan_progress_bar, section_divider, metric_card,
)
from modules.ui.components import styled_alert
from modules.ui.visualizer import (
    CHART_LAYOUT, MUTED_COLOR, BRIGHT_TEXT, GRID_COLOR,
    ZERO_LINE_COLOR, BLUE, GREEN, ORANGE, RED,
    apply_global_theme,
)
from modules.utils.localization import get_text
from modules.utils.helpers import _ensure_workspace_active, save_temp_csv
from modules.utils.theme_manager import STATUS_COLORS
from modules.utils.db_config_manager import get_all_rules


# ==============================================================================
# CONSTANTS & CONFIG
# ==============================================================================

_C_HIGH = STATUS_COLORS["warning"]["hex"]   # Amber  — high earner (>50K)
_C_STD  = STATUS_COLORS["neutral"]["hex"]   # Blue   — standard earner (≤50K)
_C_OK   = STATUS_COLORS["success"]["hex"]   # Green  — positive signal

_STRIP_KEYS = {"legend", "margin"}


def _base_layout() -> dict:
    """CHART_LAYOUT minus conflict-prone keys (legend, margin)."""
    return {k: v for k, v in CHART_LAYOUT.items() if k not in _STRIP_KEYS}


# ==============================================================================
# BINNING & MAPPING ENGINE
# ==============================================================================

def _apply_binning_config(df: pd.DataFrame, binning_cfg: dict) -> pd.DataFrame:
    """
    Apply binning and mapping rules from DB config to a COPY of df.
    Original df is NEVER modified.

    For each rule:
      - type='bin': pd.cut on numeric column → replaces column values with label strings
      - type='map': build reverse lookup {raw_val → group_label} → map categorical column

    Returns transformed copy. Also saves to temp/ for debug.
    """
    df_t = df.copy()

    # Case-insensitive column lookup
    col_lookup = {c.lower().replace("_", "").replace("-", "").replace(" ", ""): c
                  for c in df_t.columns}

    def _find_col(rule_col: str) -> str | None:
        key = rule_col.lower().replace("_", "").replace("-", "").replace(" ", "")
        return col_lookup.get(key)

    for rule_col, rule in binning_cfg.items():
        actual_col = _find_col(rule_col)
        if actual_col is None:
            continue

        rtype = rule.get("type")

        # ── Numeric Binning ──────────────────────────────────────────────
        if rtype == "bin":
            bins   = rule.get("bins", [])
            labels = rule.get("labels", [])
            if len(bins) < 2 or len(labels) != len(bins) - 1:
                continue
            try:
                numeric_series = pd.to_numeric(df_t[actual_col], errors="coerce")
                df_t[actual_col] = pd.cut(
                    numeric_series,
                    bins=bins,
                    labels=labels,
                    right=True,
                    include_lowest=True,
                ).astype(str)
                # Mark 'nan' as the original label "Unknown"
                df_t[actual_col] = df_t[actual_col].replace("nan", "Unknown")
            except Exception:
                pass

        # ── Categorical Mapping ──────────────────────────────────────────
        elif rtype == "map":
            groups = rule.get("groups", {})
            # Build reverse map: raw_value → group_label
            reverse_map = {}
            for group_label, raw_values in groups.items():
                for rv in raw_values:
                    reverse_map[str(rv).strip()] = group_label

            if reverse_map:
                df_t[actual_col] = (
                    df_t[actual_col]
                    .astype(str)
                    .str.strip()
                    .map(reverse_map)
                    .fillna(df_t[actual_col].astype(str).str.strip())  # keep unmapped as-is
                )

    # ── Save binned data to data/temp/ for debug ─────────────────────────
    save_temp_csv(df_t, prefix="eda_transformed")

    return df_t


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
        "marital":       ["maritalstatus", "marital"],
        "workclass":     ["workclass"],
        "capital_gain":  ["capitalgain", "capgain", "capitalgains", "capital_gain"],
    }
    return {
        field: next((lookup[a] for a in aliases if a in lookup), None)
        for field, aliases in _ALIASES.items()
    }


def _high_mask(series: pd.Series) -> pd.Series:
    """Boolean mask: True where income >50K."""
    return series.astype(str).str.strip().str.lower().str.contains(r">50k", regex=True, na=False)


def _insight_box(html_text: str) -> str:
    """
    Prominent insight callout — amber accent, icon header, bold highlights in amber.
    Auto-converts <b>text</b> inside html_text to amber-colored bold spans.
    """
    highlighted = html_text.replace(
        "<b>", "<b style='color:#FF9F43;font-weight:700;'>"
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
        "'>💡 Insight</div>"
        f"<div style='"
        f"font-size:0.80rem;"
        f"color:rgba(255,255,255,0.72);"
        f"line-height:1.75;"
        f"'>{highlighted}</div>"
        "</div>"
    )


def _chart_caption(html_text: str) -> None:
    """
    Styled chart caption — replaces st.caption().
    Muted text with amber highlights on <b> tags.
    """
    highlighted = html_text.replace(
        "<b>", "<b style='color:#FF9F43;font-weight:700;'>"
    )
    st.markdown(
        f"<div style='"
        f"font-size:0.76rem;"
        f"color:rgba(255,255,255,0.45);"
        f"line-height:1.6;"
        f"margin-top:6px;"
        f"'>{highlighted}</div>",
        unsafe_allow_html=True,
    )


def _top_group(series: pd.Series, income: pd.Series, n: int = 1) -> list[str]:
    """Return top-n group labels by >50K income rate."""
    df2 = pd.DataFrame({"grp": series, "hi": _high_mask(income)})
    rate = df2.groupby("grp")["hi"].mean().sort_values(ascending=False)
    return rate.head(n).index.tolist()


# ==============================================================================
# CHART 1 — Income Split Donut
# ==============================================================================

def _chart_donut(df: pd.DataFrame, income_col: str) -> go.Figure:
    """Donut: High Earner (>50K) vs Standard Earner (≤50K)."""
    n_high = int(_high_mask(df[income_col]).sum())
    n_std  = len(df) - n_high

    fig = go.Figure(go.Pie(
        labels=["Standard Earner (≤50K)", "High Earner (>50K)"],
        values=[n_std, n_high],
        hole=0.6,
        marker=dict(
            colors=["rgba(255,255,255,0.08)", _C_HIGH],
            line=dict(color="rgba(0,0,0,0.4)", width=2),
        ),
        textinfo="percent+label",
        textfont=dict(size=11, color=BRIGHT_TEXT),
        textposition="outside",
        sort=False,
        direction="clockwise",
        rotation=200,
        hovertemplate="<b>%{label}</b><br>%{value:,} employees<br>%{percent}<extra></extra>",
    ))
    fig.update_layout(
        **_base_layout(),
        height=320,
        showlegend=False,
        margin=dict(l=40, r=40, t=10, b=10),
        annotations=[dict(
            text=(f"<b style='font-size:22px;color:{_C_HIGH}'>50K</b><br>"
                  f"<span style='font-size:10px;color:{MUTED_COLOR}'>Threshold</span>"),
            x=0.5, y=0.5, showarrow=False, font=dict(size=16),
        )],
    )
    return fig


# ==============================================================================
# CHART 2 — %>50K by Age Group (Vertical Bar)
# Reads pre-binned Age column — no re-binning here
# ==============================================================================

def _chart_age_bar(df: pd.DataFrame, income_col: str, age_col: str,
                   age_labels: list[str]) -> go.Figure:
    """
    Vertical bar: % earning >50K per age group.
    Uses already-binned age column values — order from age_labels.
    """
    df2 = df[[age_col, income_col]].dropna().copy()
    df2["hi"] = _high_mask(df2[income_col])

    # Use the actual unique values in the transformed column, ordered by age_labels if possible
    present = [lb for lb in age_labels if lb in df2[age_col].unique()]
    if not present:
        present = sorted(df2[age_col].unique())

    grp = df2.groupby(age_col, observed=False)["hi"].mean() * 100
    grp = grp.reindex(present).fillna(0)

    mx = max(grp.max(), 1)
    colors = [f"rgba(255,159,67,{0.3 + 0.7 * v / mx:.2f})" for v in grp.values]

    fig = go.Figure(go.Bar(
        x=grp.index.tolist(),
        y=grp.round(1).values,
        marker=dict(color=colors, line=dict(width=0)),
        text=[f"{v:.1f}%" for v in grp.values],
        textposition="auto",
        textfont=dict(color=BRIGHT_TEXT, size=11),
        cliponaxis=False,
        hovertemplate="<b>%{x}</b><br>>50K: <b>%{y:.1f}%</b><extra></extra>",
    ))
    fig.update_layout(
        **_base_layout(),
        height=340,
        bargap=0.25,
        margin=dict(l=50, r=20, t=30, b=50),
        xaxis=dict(
            title=dict(text="Age Group", font=dict(color=MUTED_COLOR, size=12)),
            tickfont=dict(color=MUTED_COLOR, size=12),
            gridcolor="rgba(0,0,0,0)",
        ),
        yaxis=dict(
            title=dict(text="% earning >50K", font=dict(color=MUTED_COLOR, size=12)),
            tickfont=dict(color=MUTED_COLOR, size=11),
            gridcolor=GRID_COLOR,
            range=[0, mx * 1.30],
        ),
    )
    return apply_global_theme(fig)


# ==============================================================================
# CHART 3 — %>50K by Occupation (Horizontal Bar)
# Reads pre-mapped Occupation column
# ==============================================================================

def _chart_occ_bar(df: pd.DataFrame, income_col: str, occ_col: str) -> go.Figure:
    """Horizontal bar: income rate per occupation group (already mapped), sorted asc."""
    df2 = df[[occ_col, income_col]].dropna().copy()
    df2["hi"] = _high_mask(df2[income_col])

    grp = df2.groupby(occ_col)["hi"].mean() * 100
    grp = grp.sort_values(ascending=True)

    mx = max(grp.max(), 1)
    colors = [f"rgba(255,159,67,{0.3 + 0.7 * v / mx:.2f})" for v in grp.values]

    fig = go.Figure(go.Bar(
        y=grp.index.tolist(),
        x=grp.round(1).values,
        orientation="h",
        marker=dict(color=colors, line=dict(width=0)),
        text=[f"{v:.1f}%" for v in grp.values],
        textposition="auto",
        textfont=dict(color=BRIGHT_TEXT, size=11),
        cliponaxis=False,
        hovertemplate="<b>%{y}</b><br>>50K: <b>%{x:.1f}%</b><extra></extra>",
    ))
    fig.update_layout(
        **_base_layout(),
        height=max(260, len(grp) * 48),
        bargap=0.22,
        margin=dict(l=165, r=75, t=20, b=40),
        xaxis=dict(
            title=dict(text="% >50K", font=dict(color=MUTED_COLOR, size=12)),
            tickfont=dict(color=MUTED_COLOR, size=11),
            gridcolor=GRID_COLOR,
            range=[0, mx * 1.50],
        ),
        yaxis=dict(
            tickfont=dict(color=MUTED_COLOR, size=11),
            gridcolor="rgba(0,0,0,0)",
        ),
    )
    return apply_global_theme(fig)


# ==============================================================================
# CHART 4 — Heatmap: Age Group × Occupation Group
# ==============================================================================

def _chart_heatmap(df: pd.DataFrame, income_col: str, age_col: str,
                   occ_col: str, age_labels: list[str]) -> go.Figure:
    """Heatmap: rows=age bands, columns=occupation groups. Cell = % >50K."""
    df2 = df[[age_col, occ_col, income_col]].dropna().copy()
    df2["hi"] = _high_mask(df2[income_col])

    # Use actual unique values, ordered by age_labels for age axis
    age_order = [lb for lb in age_labels if lb in df2[age_col].unique()] \
                or sorted(df2[age_col].unique())
    occ_order = sorted(df2[occ_col].unique())

    pct = df2.pivot_table(index=age_col, columns=occ_col, values="hi", aggfunc="mean") * 100
    cnt = df2.pivot_table(index=age_col, columns=occ_col, values="hi", aggfunc="count")
    pct = pct.reindex(index=[a for a in age_order if a in pct.index],
                      columns=[o for o in occ_order if o in pct.columns])
    cnt = cnt.reindex(index=pct.index, columns=pct.columns)
    pct[cnt < 10] = np.nan

    text_mat = pct.map(lambda v: f"{v:.1f}%" if pd.notna(v) else "").values
    colorscale = [
        [0.0,  "rgba(30,10,0,0.10)"],
        [0.15, "rgba(255,180,100,0.30)"],
        [0.40, "rgba(255,140,50,0.60)"],
        [0.70, "rgba(220,80,20,0.85)"],
        [1.0,  "rgba(180,20,0,1.00)"],
    ]

    fig = go.Figure(go.Heatmap(
        z=pct.values,
        x=pct.columns.tolist(),
        y=pct.index.tolist(),
        text=text_mat,
        texttemplate="%{text}",
        textfont=dict(size=11, color=BRIGHT_TEXT),
        colorscale=colorscale,
        zmin=0,
        zmax=max(50, pct.max().max() * 1.1) if pct.notna().any().any() else 50,
        xgap=4, ygap=4,
        hovertemplate="<b>%{y} × %{x}</b><br>>50K: <b>%{z:.1f}%</b><extra></extra>",
        colorbar=dict(
            title=dict(text="% >50K", font=dict(color=MUTED_COLOR, size=11)),
            tickfont=dict(color=MUTED_COLOR, size=10),
            thickness=12, len=0.7, outlinewidth=0, ticksuffix="%",
        ),
    ))
    fig.update_layout(
        **_base_layout(),
        height=360,
        margin=dict(l=60, r=80, t=10, b=90),
        xaxis=dict(tickangle=-30, tickfont=dict(color=MUTED_COLOR, size=11),
                   gridcolor="rgba(0,0,0,0)", side="bottom"),
        yaxis=dict(tickfont=dict(color=MUTED_COLOR, size=11),
                   gridcolor="rgba(0,0,0,0)", autorange="reversed"),
    )
    return fig


# ==============================================================================
# CHART 5 — Bubble: Occupation × Working Hours vs Income Probability
# ==============================================================================

def _chart_bubble(df: pd.DataFrame, income_col: str, occ_col: str,
                  hours_col: str) -> go.Figure:
    """
    Bubble: X=P(>50K), Y=Occupation group, Size=Avg hours/week.
    Uses already-mapped occupation column.
    """
    df2 = df[[occ_col, income_col, hours_col]].dropna().copy()
    df2["hi"]  = _high_mask(df2[income_col])

    # hours must be numeric (may have been binned → revert to original if type=bin)
    df2[hours_col] = pd.to_numeric(df2[hours_col], errors="coerce")
    df2 = df2.dropna(subset=[hours_col])

    grp = df2.groupby(occ_col).agg(
        pct  = ("hi",       "mean"),
        hrs  = (hours_col,  "mean"),
        n    = (hours_col,  "count"),
    ).reset_index()
    grp["pct"] = (grp["pct"] * 100).round(1)
    grp["hrs"] = grp["hrs"].round(1)
    grp = grp.sort_values("pct")

    sz_min, sz_max = grp["hrs"].min(), grp["hrs"].max()
    grp["sz"] = 14 + (grp["hrs"] - sz_min) / (sz_max - sz_min + 1e-9) * 31

    pct_max = max(grp["pct"].max(), 1)
    colors = [
        f"rgba(91,134,229,{0.35 + 0.6 * v / pct_max:.2f})"
        for v in grp["pct"].values
    ]

    fig = go.Figure(go.Scatter(
        x=grp["pct"],
        y=grp[occ_col],
        mode="markers",
        marker=dict(
            size=grp["sz"],
            color=colors,
            line=dict(color="rgba(255,255,255,0.12)", width=1),
        ),
        customdata=np.column_stack([grp["hrs"], grp["n"]]),
        hovertemplate=(
            "<b>%{y}</b><br>"
            "Probability >50K: <b>%{x:.1f}%</b><br>"
            "Avg hours/week: <b>%{customdata[0]:.1f}</b><br>"
            "Employees: %{customdata[1]:,}"
            "<extra></extra>"
        ),
    ))
    fig.update_layout(
        **_base_layout(),
        height=360,
        margin=dict(l=160, r=30, t=10, b=50),
        xaxis=dict(
            title=dict(text="Probability of >50K Income (%)", font=dict(color=MUTED_COLOR, size=12)),
            tickfont=dict(color=MUTED_COLOR, size=11),
            gridcolor=GRID_COLOR,
            range=[0, pct_max * 1.2],
        ),
        yaxis=dict(
            tickfont=dict(color=MUTED_COLOR, size=11),
            gridcolor="rgba(0,0,0,0)",
        ),
    )
    fig.add_annotation(
        x=0.97, y=0.03, xref="paper", yref="paper",
        text="<b>Bubble size</b> = Avg Hours/week",
        showarrow=False,
        font=dict(color=MUTED_COLOR, size=9),
        align="right",
        bgcolor="rgba(0,0,0,0.4)",
        bordercolor="rgba(255,255,255,0.06)",
        borderpad=5, borderwidth=1,
    )
    return apply_global_theme(fig)


# ==============================================================================
# CHART 6 — Grouped HBar: % >50K by Sex × Occupation
# ==============================================================================

def _chart_sex_occ_bar(
    df: pd.DataFrame, income_col: str, occ_col: str, sex_col: str
) -> go.Figure:
    """
    Grouped horizontal bar: Male vs Female % earning >50K per occupation group.
    Sorted by male income rate descending.
    """
    df2 = df[[occ_col, sex_col, income_col]].dropna().copy()
    df2["hi"]  = _high_mask(df2[income_col])
    df2["sex"] = df2[sex_col].astype(str).str.strip().str.title()

    # Keep only Male/Female
    df2 = df2[df2["sex"].isin(["Male", "Female"])]

    grp = (
        df2.groupby([occ_col, "sex"])["hi"].mean() * 100
    ).unstack("sex").fillna(0)
    grp = grp.sort_values("Male", ascending=True)

    occ_labels = grp.index.tolist()

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Male",
        y=occ_labels,
        x=grp.get("Male", pd.Series([0]*len(occ_labels))).values,
        orientation="h",
        marker=dict(color="rgba(91,134,229,0.75)", line=dict(width=0)),
        text=[f"{v:.1f}%" for v in grp.get("Male", pd.Series([0]*len(occ_labels))).values],
        textposition="auto",
        textfont=dict(color=BRIGHT_TEXT, size=10),
        cliponaxis=False,
        hovertemplate="<b>%{y}</b> · Male<br>>50K: <b>%{x:.1f}%</b><extra></extra>",
    ))
    fig.add_trace(go.Bar(
        name="Female",
        y=occ_labels,
        x=grp.get("Female", pd.Series([0]*len(occ_labels))).values,
        orientation="h",
        marker=dict(color="rgba(255,159,67,0.75)", line=dict(width=0)),
        text=[f"{v:.1f}%" for v in grp.get("Female", pd.Series([0]*len(occ_labels))).values],
        textposition="auto",
        textfont=dict(color=BRIGHT_TEXT, size=10),
        cliponaxis=False,
        hovertemplate="<b>%{y}</b> · Female<br>>50K: <b>%{x:.1f}%</b><extra></extra>",
    ))

    x_max = max(
        grp.get("Male",   pd.Series([0])).max(),
        grp.get("Female", pd.Series([0])).max(),
    )
    fig.update_layout(
        **_base_layout(),
        barmode="group",
        height=max(300, len(occ_labels) * 58),
        bargap=0.18,
        bargroupgap=0.06,
        margin=dict(l=175, r=90, t=10, b=65),
        legend=dict(
            orientation="h", x=0.5, xanchor="center", y=-0.20,
            font=dict(color=MUTED_COLOR, size=11),
        ),
        xaxis=dict(
            title=dict(text="% >50K", font=dict(color=MUTED_COLOR, size=12)),
            tickfont=dict(color=MUTED_COLOR, size=11),
            gridcolor=GRID_COLOR,
            range=[0, x_max * 1.55],
        ),
        yaxis=dict(tickfont=dict(color=MUTED_COLOR, size=11), gridcolor="rgba(0,0,0,0)"),
    )
    return apply_global_theme(fig)


# ==============================================================================
# CHART 7 — Dual Heatmap: CapGain>0 & Income>50K by Education × Occupation
# ==============================================================================

def _chart_capgain_dual_heatmap(
    df: pd.DataFrame, income_col: str, occ_col: str,
    edu_col: str, cg_col: str,
) -> tuple[go.Figure, go.Figure]:
    """
    Returns two heatmaps side-by-side:
      Left:  % with CapGain > 0 by Education × Occupation
      Right: % Income >50K (among those with CapGain>0) by Education × Occupation
    """
    df2 = df[[edu_col, occ_col, income_col, cg_col]].copy()
    df2[cg_col] = pd.to_numeric(df2[cg_col], errors="coerce").fillna(0)
    df2["has_cg"] = (df2[cg_col] > 0).astype(int)
    df2["hi"]     = _high_mask(df2[income_col]).astype(int)
    df2 = df2.dropna(subset=[edu_col, occ_col, income_col])

    edu_order = sorted(df2[edu_col].unique())
    occ_order = sorted(df2[occ_col].unique())

    _WARM = [
        [0.0,  "rgba(20,20,30,0.2)"],
        [0.25, "rgba(91,134,229,0.3)"],
        [0.5,  "rgba(255,159,67,0.5)"],
        [0.75, "rgba(220,80,20,0.8)"],
        [1.0,  "rgba(180,20,0,1.0)"],
    ]

    def _heatmap_trace(pivot, label, source_df):
        """Build heatmap trace with sample-count filter from *source_df*."""
        cnt = source_df.pivot_table(
            index=edu_col, columns=occ_col, values="hi", aggfunc="count",
        )
        pivot = pivot.reindex(index=[e for e in edu_order if e in pivot.index],
                              columns=[o for o in occ_order if o in pivot.columns])
        cnt   = cnt.reindex(index=pivot.index, columns=pivot.columns)
        pivot[cnt < 5] = np.nan
        text = pivot.map(lambda v: f"{v:.1f}%" if pd.notna(v) else "").values
        return go.Heatmap(
            z=pivot.values, x=pivot.columns.tolist(), y=pivot.index.tolist(),
            text=text, texttemplate="%{text}",
            textfont=dict(size=10, color=BRIGHT_TEXT),
            colorscale=_WARM, zmin=0,
            zmax=max(40, pivot.stack().max() * 1.1) if not pivot.stack().empty else 40,
            xgap=3, ygap=3,
            hovertemplate=f"<b>%{{y}} × %{{x}}</b><br>{label}: <b>%{{z:.1f}}%</b><extra></extra>",
            colorbar=dict(
                tickfont=dict(color=MUTED_COLOR, size=9),
                thickness=12, len=0.65, outlinewidth=0, ticksuffix="%",
                x=1.02, xpad=4,
            ),
        )

    # Left — % has cap gain
    pct_cg = df2.pivot_table(index=edu_col, columns=occ_col, values="has_cg", aggfunc="mean") * 100
    # Right — % >50K | cap gain > 0
    df_hi  = df2[df2["has_cg"] == 1]
    pct_hi = df_hi.pivot_table(index=edu_col, columns=occ_col, values="hi", aggfunc="mean") * 100

    _layout_base = dict(
        **_base_layout(),
        height=310,
        margin=dict(l=80, r=110, t=30, b=90),
        xaxis=dict(tickangle=-30, tickfont=dict(color=MUTED_COLOR, size=10),
                   gridcolor="rgba(0,0,0,0)"),
        yaxis=dict(tickfont=dict(color=MUTED_COLOR, size=10),
                   gridcolor="rgba(0,0,0,0)", autorange="reversed"),
    )

    fig_l = go.Figure(_heatmap_trace(pct_cg, "% CapGain>0", df2))
    fig_l.update_layout(**_layout_base)

    fig_r = go.Figure(_heatmap_trace(pct_hi, "% >50K | CapGain>0", df_hi))
    fig_r.update_layout(**_layout_base)

    return fig_l, fig_r


# ==============================================================================
# CHART 8 — Capital Gain Analysis: Bar + Boxplot by Education & Occupation
# ==============================================================================

def _chart_capgain_analysis(
    df: pd.DataFrame, income_col: str, occ_col: str,
    edu_col: str, cg_col: str,
) -> tuple[go.Figure, go.Figure, go.Figure, go.Figure]:
    """
    4 charts:
      fig_edu_bar  — % CapGain>0 by Education, split by income group (stacked)
      fig_edu_box  — log2(CapGain) distribution by Education (box)
      fig_occ_bar  — % CapGain>0 by Occupation, split by income group
      fig_occ_box  — log2(CapGain) distribution by Occupation (box)
    """
    df2 = df[[edu_col, occ_col, income_col, cg_col]].copy()
    df2[cg_col] = pd.to_numeric(df2[cg_col], errors="coerce").fillna(0)
    df2["has_cg"]  = df2[cg_col] > 0
    df2["hi"]      = _high_mask(df2[income_col])
    df2["income_g"] = df2["hi"].map({True: ">50K", False: "≤50K"})
    df2["log2cg"]  = np.log2(df2[cg_col].clip(lower=1))
    df2 = df2.dropna(subset=[edu_col, occ_col, income_col])

    _INCOME_COLOR = {">50K": "rgba(255,159,67,0.75)", "≤50K": "rgba(91,134,229,0.60)"}

    def _bar_by(group_col, order=None):
        """% has_cg by group, split income."""
        grp = (
            df2.groupby([group_col, "income_g"])["has_cg"].mean() * 100
        ).reset_index()
        grp.columns = [group_col, "income_g", "pct"]
        pivoted = grp.pivot(index=group_col, columns="income_g", values="pct").fillna(0)
        if order:
            pivoted = pivoted.reindex([o for o in order if o in pivoted.index])
        pivoted = pivoted.sort_values(">50K", ascending=True)
        fig = go.Figure()
        for grp_name in ["≤50K", ">50K"]:
            if grp_name in pivoted.columns:
                fig.add_trace(go.Bar(
                    name=grp_name,
                    y=pivoted.index.tolist(),
                    x=pivoted[grp_name].values,
                    orientation="h",
                    marker=dict(color=_INCOME_COLOR[grp_name], line=dict(width=0)),
                    text=[f"{v:.1f}%" for v in pivoted[grp_name].values],
                    textposition="auto",
                    textfont=dict(color=BRIGHT_TEXT, size=9),
                    cliponaxis=False,
                    hovertemplate="<b>%{y}</b><br>" + grp_name + ": <b>%{x:.1f}%</b><extra></extra>",
                ))
        x_max = pivoted.max().max() if not pivoted.empty else 30
        fig.update_layout(
            **_base_layout(),
            barmode="group",
            height=max(260, len(pivoted) * 48),
            bargap=0.2, bargroupgap=0.05,
            margin=dict(l=125, r=70, t=10, b=60),
            legend=dict(orientation="h", x=0.5, xanchor="center", y=-0.22,
                        font=dict(color=MUTED_COLOR, size=10)),
            xaxis=dict(title=dict(text="% CapGain>0", font=dict(color=MUTED_COLOR, size=11)),
                       tickfont=dict(color=MUTED_COLOR, size=10),
                       gridcolor=GRID_COLOR, range=[0, x_max * 1.55]),
            yaxis=dict(tickfont=dict(color=MUTED_COLOR, size=10), gridcolor="rgba(0,0,0,0)"),
        )
        return fig

    def _box_by(group_col):
        """Box: log2 CapGain distribution per group."""
        df3 = df2[df2["has_cg"]].copy()
        groups = sorted(df3[group_col].dropna().unique())
        fig = go.Figure()
        for i, g in enumerate(groups):
            vals = df3[df3[group_col] == g]["log2cg"].dropna().values
            fig.add_trace(go.Box(
                x=vals, name=str(g), orientation="h",
                marker=dict(color=f"rgba(255,159,67,{0.4 + 0.6 * i / max(len(groups)-1,1):.2f})"),
                line=dict(color="rgba(255,255,255,0.4)", width=1),
                fillcolor=f"rgba(255,159,67,{0.12 + 0.15 * i / max(len(groups)-1,1):.2f})",
                hovertemplate="<b>%{name}</b><br>log₂(CapGain): %{x:.2f}<extra></extra>",
            ))
        fig.update_layout(
            **_base_layout(),
            showlegend=False,
            height=max(260, len(groups) * 48),
            margin=dict(l=120, r=30, t=10, b=40),
            xaxis=dict(title=dict(text="log₂(Capital Gain)", font=dict(color=MUTED_COLOR, size=11)),
                       tickfont=dict(color=MUTED_COLOR, size=10), gridcolor=GRID_COLOR),
            yaxis=dict(tickfont=dict(color=MUTED_COLOR, size=10), gridcolor="rgba(0,0,0,0)",
                       autorange="reversed"),
        )
        return apply_global_theme(fig)

    return _bar_by(edu_col), _box_by(edu_col), _bar_by(occ_col), _box_by(occ_col)


# ==============================================================================
# KPI HEADER
# ==============================================================================

def _render_kpis(df_raw: pd.DataFrame, cols: dict) -> None:
    """6-card KPI header — computed from original (non-transformed) data."""
    total  = len(df_raw)
    n_cols = len(df_raw.columns)

    size_mb = df_raw.memory_usage(deep=True).sum() / (1024 ** 2)

    if cols["income"]:
        high_mask = _high_mask(df_raw[cols["income"]])
        n_high    = int(high_mask.sum())
        n_std     = total - n_high
        pct_high  = round(n_high / total * 100, 1) if total else 0
        ratio_str = f"{n_high:,} vs {n_std:,}"
    else:
        n_high, pct_high, ratio_str = 0, 0.0, "—"

    if cols["age"]:
        age_s      = pd.to_numeric(df_raw[cols["age"]], errors="coerce").dropna()
        median_age = int(age_s.median())
        q1_age     = int(age_s.quantile(0.25))
        q3_age     = int(age_s.quantile(0.75))
        age_range  = f"IQR  {q1_age} – {q3_age} yrs"
    else:
        median_age, age_range = "—", "—"

    gender_col = cols.get("sex")
    if gender_col:
        vc         = df_raw[gender_col].astype(str).str.strip().str.lower().value_counts()
        male_n     = vc.get("male", 0)
        female_n   = vc.get("female", 0)
        total_gend = male_n + female_n if (male_n + female_n) > 0 else 1
        pct_male   = round(male_n / total_gend * 100, 1)
        pct_female = round(female_n / total_gend * 100, 1)
        gender_val = f"{pct_male}% ♂"
        gender_sub = f"{pct_female}% ♀  ({male_n:,} / {female_n:,})"
        g_glow     = "blue"
    else:
        gender_val = "—"; gender_sub = "No gender column found"; g_glow = "blue"

    if cols["hours"]:
        hrs_s        = pd.to_numeric(df_raw[cols["hours"]], errors="coerce").dropna()
        avg_hrs      = round(hrs_s.mean(), 1)
        pct_overtime = round((hrs_s > 40).sum() / len(hrs_s) * 100, 1)
        hrs_glow     = "orange" if pct_overtime > 30 else "green"
        hrs_sub      = f"{pct_overtime}% work OT (>40 h/w)"
    else:
        avg_hrs, hrs_sub, hrs_glow = "—", "—", "blue"

    if cols["income"] and cols["age"]:
        _hi      = _high_mask(df_raw[cols["income"]])
        age_s2   = pd.to_numeric(df_raw[cols["age"]], errors="coerce")
        valid    = ~age_s2.isna()
        med_high = int(age_s2[valid & _hi].median())  if _hi.sum()   > 0 else 0
        med_std  = int(age_s2[valid & ~_hi].median()) if (~_hi).sum() > 0 else 0
        age_gap  = med_high - med_std
        corr_val = f"+{age_gap} yrs" if age_gap >= 0 else f"{age_gap} yrs"
        corr_sub = f"High Earner {med_high} | STD {med_std}"
        corr_glow = "orange" if age_gap > 5 else "green"
    else:
        corr_val, corr_sub, corr_glow = "—", "—", "blue"

    c1, c2, c3, c4, c5, c6 = st.columns(6, gap="small")
    with c1: metric_card("Dataset Scale",       f"{total:,}",      f"{n_cols} cols · {size_mb:.1f} MB", glow="blue")
    with c2: metric_card("High Earner (>50K)",  f"{pct_high}%",    ratio_str,                           glow="orange" if pct_high > 30 else "blue")
    with c3: metric_card("Median Age",          f"{median_age} yrs", age_range,                         glow="green")
    with c4: metric_card("Gender Ratio",        gender_val,        gender_sub,                          glow=g_glow)
    with c5: metric_card("Avg Hours / Week",    f"{avg_hrs} h",    hrs_sub,                             glow=hrs_glow)
    with c6: metric_card("Age Gap (Hi vs Std)", corr_val,          corr_sub,                            glow=corr_glow)


# ==============================================================================
# SECTION HEADER HELPER
# ==============================================================================

def _panel_header(title: str, subtitle: str = "") -> None:
    """Prominent chart panel heading — amber accent, gradient bg."""
    subtitle_html = (
        f"<div style='color:rgba(255,255,255,0.38);font-size:0.76rem;"
        f"font-weight:400;margin-top:4px;letter-spacing:0.1px;'>{subtitle}</div>"
        if subtitle else ""
    )
    st.markdown(
        f"<div style='padding:14px 18px;margin-bottom:12px;"
        f"background:linear-gradient(135deg, rgba(255,159,67,0.08) 0%, rgba(255,159,67,0.02) 100%);"
        f"border:1px solid rgba(255,159,67,0.12);"
        f"border-left:3px solid rgba(255,159,67,0.65);"
        f"border-radius:0 12px 12px 0;'>"
        f"<div style='font-size:1.05rem;font-weight:800;color:#FFFFFF;"
        f"letter-spacing:-0.3px;line-height:1.3;'>{title}</div>"
        f"{subtitle_html}"
        f"</div>",
        unsafe_allow_html=True,
    )


# ==============================================================================
# MAIN LAYOUT
# ==============================================================================

def main() -> None:
    """Entry point: load → apply binning/mapping → KPIs → 5 insight charts."""
    lang = st.session_state.get("lang", "en")

    page_header(
        title=get_text("eda_title", lang),
        subtitle=get_text("overview_journey_eda_desc", lang),
    )

    _ensure_workspace_active()
    active_file = st.session_state.get("active_file")
    workspace_status(active_file)

    # ── Load raw data ──────────────────────────────────────────────────────
    df_raw = data_engine.load_and_standardize(
        active_file, _file_mtime=data_engine._get_file_mtime(active_file)
    )
    active_file_scan_progress_bar("_eda_done")


    # ── Load binning_config from DB and apply ──────────────────────────────
    # OPT-4: cache the transformed DataFrame so pd.cut/map only runs when
    # the active file OR the binning config actually changes between reruns.
    rules          = get_all_rules()
    binning_cfg    = rules.get("binning_config", {})
    _cfg_key = ("_eda_transformed", active_file, str(sorted(binning_cfg.items())))
    if st.session_state.get(_cfg_key[0] + "_key") != _cfg_key:
        df = _apply_binning_config(df_raw, binning_cfg)
        st.session_state[_cfg_key[0]]        = df
        st.session_state[_cfg_key[0] + "_key"] = _cfg_key
    else:
        df = st.session_state[_cfg_key[0]]

    # ── Resolve column names ────────────────────────────────────────────────
    # Charts use transformed df; KPIs use raw df — resolve both independently.
    cols     = _resolve_cols(df)
    cols_raw = _resolve_cols(df_raw)

    if not cols["income"]:
        styled_alert(
            "No <b>income</b> column found. "
            "Ensure the dataset has a column named <code>income</code> or <code>salary</code>.",
            "warning",
        )
        return

    # ── Case-insensitive binning config key finder ─────────────────────────
    def _find_cfg_key(target: str) -> str | None:
        """Find the actual key in binning_cfg matching *target* case-insensitively."""
        norm = target.lower().replace("_", "")
        return next(
            (k for k in binning_cfg if k.lower().replace("_", "") == norm),
            None,
        )

    # ── Extract ordered labels for binned columns ──────────────────────────
    def _ordered_labels(col_rule_key: str | None, df_col: str | None) -> list[str]:
        if df_col is None or col_rule_key is None:
            return []
        rule = binning_cfg.get(col_rule_key) or {}
        if rule.get("type") == "bin" and rule.get("labels"):
            return rule["labels"]
        return sorted(df[df_col].dropna().unique().tolist())

    age_labels = _ordered_labels(_find_cfg_key("Age"), cols["age"])

    # ── KPI Cards — always from RAW (original numeric age, hours, etc.) ────
    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
    _render_kpis(df_raw, cols_raw)
    section_divider()

    # ── Compute insights dynamically ───────────────────────────────────────
    income_col = cols["income"]
    age_col    = cols["age"]
    occ_col    = cols["occupation"]
    hours_col  = cols["hours"]

    pct_high = round(_high_mask(df[income_col]).mean() * 100, 1) if income_col else 0

    # Peak age group
    if age_col and income_col:
        age_grp_rates = (
            df[[age_col, income_col]].dropna()
            .assign(hi=lambda x: _high_mask(x[income_col]))
            .groupby(age_col)["hi"].mean() * 100
        )
        peak_age_group = age_grp_rates.idxmax() if not age_grp_rates.empty else "—"
        peak_age_pct   = round(age_grp_rates.max(), 1) if not age_grp_rates.empty else 0
    else:
        peak_age_group, peak_age_pct = "—", 0

    # Top / Bottom occupation
    if occ_col and income_col:
        occ_rates = (
            df[[occ_col, income_col]].dropna()
            .assign(hi=lambda x: _high_mask(x[income_col]))
            .groupby(occ_col)["hi"].mean() * 100
        )
        top_occ  = occ_rates.idxmax() if not occ_rates.empty else "—"
        top_occ_pct  = round(occ_rates.max(), 1)  if not occ_rates.empty else 0
        bot_occ  = occ_rates.idxmin() if not occ_rates.empty else "—"
        bot_occ_pct  = round(occ_rates.min(), 1) if not occ_rates.empty else 0
    else:
        top_occ = bot_occ = "—"
        top_occ_pct = bot_occ_pct = 0

    # =====================================================================
    # ROW 1: Donut | Age Bar
    # =====================================================================
    c1, c2 = st.columns([4, 6])

    with c1:
        _panel_header(
            "Income Distribution — High vs Standard Earners",
            f"{pct_high}% earn >50K · {df[income_col].notna().sum():,} valid records",
        )
        if income_col:
            st.plotly_chart(_chart_donut(df, income_col), use_container_width=True, key="ch_donut")
            st.markdown(
                _insight_box(
                    f"<b>{pct_high}%</b> of employees earn &gt;50K. "
                    f"The distribution is skewed toward lower incomes, "
                    f"indicating structural income inequality in the dataset. "
                    f"({df[income_col].notna().sum():,} valid records)"
                ),
                unsafe_allow_html=True,
            )

    with c2:
        _panel_header(
            "High-Income Rate (>50K) by Age Group",
            f"Peak group: {peak_age_group} · {peak_age_pct}% earn >50K",
        )
        if income_col and age_col:
            st.plotly_chart(
                _chart_age_bar(df, income_col, age_col, age_labels),
                use_container_width=True, key="ch_age",
            )
            _chart_caption(
                f"The <b>{peak_age_group}</b> age group has the highest >50K rate "
                f"(<b>{peak_age_pct}%</b>). Income probability generally "
                f"rises with age and experience."
            )

    section_divider()

    # =====================================================================
    # ROW 2: Occupation Bar | Heatmap
    # =====================================================================
    c3, c4 = st.columns([4, 6])

    with c3:
        _panel_header(
            "High-Income Rate (>50K) by Occupation",
            f"Highest: {top_occ} ({top_occ_pct}%)  ·  Lowest: {bot_occ} ({bot_occ_pct}%)",
        )
        if income_col and occ_col:
            st.plotly_chart(_chart_occ_bar(df, income_col, occ_col), use_container_width=True, key="ch_occ")
            _chart_caption(
                f"<b>{top_occ}</b> leads with <b>{top_occ_pct}%</b> earning >50K, "
                f"while <b>{bot_occ}</b> has the lowest rate (<b>{bot_occ_pct}%</b>)."
            )

    with c4:
        _panel_header(
            "Income Rate Heatmap · Age Group × Occupation",
            "Joint effect of experience and role type on earning >50K",
        )
        if income_col and age_col and occ_col:
            st.plotly_chart(
                _chart_heatmap(df, income_col, age_col, occ_col, age_labels),
                use_container_width=True, key="ch_heat",
            )
            st.markdown(
                _insight_box(
                    f"<b>Age (experience)</b> and <b>occupation type</b> together "
                    f"strongly predict income. The highest concentrations of &gt;50K earners "
                    f"appear at the intersection of <b>{peak_age_group}</b> age and "
                    f"<b>{top_occ}</b> roles."
                ),
                unsafe_allow_html=True,
            )

    section_divider()

    # =====================================================================
    # ROW 3: Bubble — full width
    # =====================================================================
    _panel_header(
        "Income Rate vs Avg Hours Worked · by Occupation",
        "Bubble size = avg hours/week  ·  X-axis = probability of earning >50K",
    )
    if income_col and occ_col and hours_col:
        # Note: bubble chart needs numeric hours — use df_raw hours if binned
        hours_for_bubble = hours_col
        _hours_cfg_key = _find_cfg_key("Hours_per_Week")
        if _hours_cfg_key and binning_cfg[_hours_cfg_key].get("type") == "bin":
            # hours was binned → use original numeric column from df_raw
            df_bubble  = df.copy()
            if cols_raw["hours"]:
                df_bubble[hours_col] = df_raw[cols_raw["hours"]]
        else:
            df_bubble = df

        st.plotly_chart(
            _chart_bubble(df_bubble, income_col, occ_col, hours_col),
            use_container_width=True, key="ch_bubble",
        )
        _chart_caption(
            "Bubble size reflects <b>avg hours/week</b> per occupation group. "
            f"<b>{top_occ}</b> has the strongest earning rate "
            f"(<b>{top_occ_pct}%</b>)."
        )

    section_divider()

    # =====================================================================
    # ROW 4: Sex × Occupation — grouped bar
    # =====================================================================
    sex_col = cols.get("sex")
    edu_col = cols.get("education")
    cg_col  = cols.get("capital_gain")

    if income_col and occ_col and sex_col:
        df_sex = df[[occ_col, sex_col, income_col]].dropna().copy()
        df_sex["hi"]  = _high_mask(df_sex[income_col])
        df_sex["sex"] = df_sex[sex_col].astype(str).str.strip().str.title()
        male_rate   = round(df_sex[df_sex["sex"] == "Male"]["hi"].mean() * 100, 1)
        female_rate = round(df_sex[df_sex["sex"] == "Female"]["hi"].mean() * 100, 1)
        gap         = round(male_rate - female_rate, 1)

        _panel_header(
            "High-Income Rate (>50K) by Sex × Occupation",
            f"Male: {male_rate}%  ·  Female: {female_rate}%  ·  Gender gap: {gap:+.1f}pp",
        )
        st.plotly_chart(
            _chart_sex_occ_bar(df, income_col, occ_col, sex_col),
            use_container_width=True, key="ch_sex_occ",
        )
        st.markdown(
            _insight_box(
                f"A gender gap of <b>{gap:+.1f} percentage points</b> exists across all occupation groups. "
                f"Male employees reach <b>{male_rate}%</b> vs <b>{female_rate}%</b> for females — "
                f"even within the same occupation category."
            ),
            unsafe_allow_html=True,
        )
        section_divider()

    # =====================================================================
    # ROW 5: Capital Gain × Education × Occupation — Dual Heatmap
    # =====================================================================
    if income_col and occ_col and edu_col and cg_col:
        df_cg_raw  = pd.to_numeric(df[cg_col], errors="coerce").fillna(0)
        pct_has_cg = round((df_cg_raw > 0).mean() * 100, 1)
        med_cg     = int(df_cg_raw[df_cg_raw > 0].median()) if (df_cg_raw > 0).any() else 0

        _panel_header(
            "Capital Gain Penetration vs Income by Education × Occupation",
            f"{pct_has_cg}% of employees have Capital Gain >0  ·  Median gain among them: ${med_cg:,}",
        )
        fig_l, fig_r = _chart_capgain_dual_heatmap(df, income_col, occ_col, edu_col, cg_col)
        c_hl, c_hr = st.columns(2)
        with c_hl:
            st.markdown(
                "<div style='font-size:0.73rem;color:rgba(255,159,67,0.60);"
                "text-align:center;margin-bottom:4px;'>% Employees with Capital Gain > 0</div>",
                unsafe_allow_html=True,
            )
            st.plotly_chart(fig_l, use_container_width=True, key="ch_cg_hm_l")
        with c_hr:
            st.markdown(
                "<div style='font-size:0.73rem;color:rgba(255,159,67,0.60);"
                "text-align:center;margin-bottom:4px;'>% Earning >50K (among CapGain>0 only)</div>",
                unsafe_allow_html=True,
            )
            st.plotly_chart(fig_r, use_container_width=True, key="ch_cg_hm_r")
        st.markdown(
            _insight_box(
                f"Only <b>{pct_has_cg}%</b> of employees have capital gains — but among those who do, "
                f"the >50K rate is significantly elevated. High Earners (>50K) are often "
                f"<b>\"Capital Masters\"</b>: they do not rely solely on salary "
                f"but also actively generate capital gains."
            ),
            unsafe_allow_html=True,
        )
        section_divider()

    # =====================================================================
    # ROW 6: Capital Gain Analysis — 4-panel (Bar + Box by Edu & Occ)
    # =====================================================================
    if income_col and occ_col and edu_col and cg_col:
        _panel_header(
            "Capital Gain Distribution by Education & Occupation",
            "% with CapGain>0 split by income group  ·  log₂(CapGain) spread",
        )
        fe_bar, fe_box, fo_bar, fo_box = _chart_capgain_analysis(
            df, income_col, occ_col, edu_col, cg_col
        )

        st.markdown(
            "<div style='font-size:0.78rem;font-weight:700;color:rgba(255,255,255,0.45);"
            "text-transform:uppercase;letter-spacing:0.8px;margin:8px 0 6px;'>"
            "By Education Group</div>",
            unsafe_allow_html=True,
        )
        c1e, c2e = st.columns(2)
        with c1e:
            st.plotly_chart(fe_bar, use_container_width=True, key="ch_edu_bar")
        with c2e:
            st.plotly_chart(fe_box, use_container_width=True, key="ch_edu_box")
        _chart_caption(
            "Investing in education is the most sustainable form of <b>\"capital gain\"</b> — "
            "Advanced education groups show both higher frequency and larger magnitude of capital gains."
        )

        st.markdown(
            "<div style='font-size:0.78rem;font-weight:700;color:rgba(255,255,255,0.45);"
            "text-transform:uppercase;letter-spacing:0.8px;margin:18px 0 6px;'>"
            "By Occupation Group</div>",
            unsafe_allow_html=True,
        )
        c1o, c2o = st.columns(2)
        with c1o:
            st.plotly_chart(fo_bar, use_container_width=True, key="ch_occ_bar2")
        with c2o:
            st.plotly_chart(fo_box, use_container_width=True, key="ch_occ_box")
        # Dynamic: show top-2 occupations by >50K rate
        _top2 = occ_rates.nlargest(2).index.tolist() if 'occ_rates' in dir() else [top_occ]
        _top2_html = " and ".join(f"<b>{o}</b>" for o in _top2)
        st.markdown(
            _insight_box(
                f"{_top2_html} occupations are becoming "
                f"<b>\"promising fields\"</b> for income mobility — they combine high >50K rates "
                f"with notable capital gain activity among their top earners."
            ),
            unsafe_allow_html=True,
        )
        section_divider()


if __name__ == "__main__":
    main()
