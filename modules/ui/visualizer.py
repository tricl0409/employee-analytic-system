"""
visualizer.py — Neon-themed Chart Library
Plotly charts with consistent dark theme, glow effects, and smooth animations.
"""

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from typing import Dict, Any, List
import pandas as pd
import numpy as np

from modules.utils.theme_manager import CHART_COLORWAY, STATUS_COLORS


# =============================================================================
# THEME CONSTANTS
# =============================================================================

# Background
BG_PAPER  = "rgba(0,0,0,0)"
BG_PLOT   = "rgba(0,0,0,0)"

# Typography
CHART_FONT = dict(family='"Source Sans Pro", "Inter", sans-serif', color="#E0E0E0", size=12)
MUTED_COLOR = "#8892A0"
BRIGHT_TEXT = "#F8FAFC"

# Grid
GRID_COLOR = "rgba(255,255,255,0.06)"
ZERO_LINE_COLOR = "rgba(255,255,255,0.1)"

# Primary Palette (Theme Manager Palette)
BLUE   = CHART_COLORWAY[1]  # Sky Blue (#5B86E5)
GREEN  = CHART_COLORWAY[0]  # Lime (#A6CE39)
ORANGE = CHART_COLORWAY[2]  # Amber (#FF9F43)
RED    = CHART_COLORWAY[3]  # Coral Red (#FF5B5C)

# Neon accents
NEON_CYAN  = BLUE
NEON_GREEN = GREEN

# Gradient palette for multi-series charts
# We define a 5th color dynamically to avoid Zip loop truncations 
# or use a muted 5th color directly from standard status mappings.
GRADIENT_5 = CHART_COLORWAY + ["#64748B"] # Append Slate Gray for 5th
GRADIENT_8 = CHART_COLORWAY + ["#64748B", "#94A3B8", "#CBD5E1", "#F1F5F9"]


# =============================================================================
# SHARED LAYOUT
# =============================================================================

CHART_LAYOUT = dict(
    paper_bgcolor=BG_PAPER,
    plot_bgcolor=BG_PLOT,
    colorway=CHART_COLORWAY,
    font=CHART_FONT,
    margin=dict(l=40, r=20, t=30, b=40),
    legend=dict(
        bgcolor="rgba(0,0,0,0)",
        font=dict(family='"Inter", sans-serif', color=MUTED_COLOR, size=11),
    ),
    hoverlabel=dict(
        bgcolor=BG_PLOT,
        font_size=13,
        font_color=BRIGHT_TEXT,
        bordercolor=BRIGHT_TEXT,
    ),
    modebar=dict(bgcolor="rgba(0,0,0,0)"),
)

# Shared animation config for all charts
ANIMATION_CONFIG = dict(
    frame=dict(duration=600, redraw=True),
    transition=dict(duration=400, easing="cubic-in-out"),
)

def apply_global_theme(fig: go.Figure) -> go.Figure:
    """
    Applies the global overarching Metric Card theme constraints to any Plotly figure.
    - Enforces transparent background.
    - Assigns the categorical CHART_COLORWAY.
    - Adds subtle neon glow effects to lines/markers natively.
    """
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        colorway=CHART_COLORWAY,
        xaxis=dict(gridcolor="rgba(255,255,255,0.05)", zerolinecolor="rgba(255,255,255,0.1)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.05)", zerolinecolor="rgba(255,255,255,0.1)")
    )
    
    # Inject neon glow into lines (only for datasets with reasonable size to avoid WebGL lag)
    MAX_POINTS_FOR_GLOW = 500
    if hasattr(fig, 'data'):
        for i, trace in enumerate(fig.data):
            if isinstance(trace, go.Scatter) and getattr(trace, 'mode', '') in ['lines', 'lines+markers']:
                # Skip glow if dataset is too large
                if hasattr(trace, 'x') and trace.x is not None and len(trace.x) > MAX_POINTS_FOR_GLOW:
                    continue
                
                color = trace.line.color or CHART_COLORWAY[i % len(CHART_COLORWAY)]
                trace.line.width = getattr(trace.line, 'width', 2.5)
                # Ensure the primary line is bright
                
                # Add shadow layers
                shadow_trace1 = go.Scatter(
                    x=trace.x, y=trace.y,
                    mode='lines',
                    line=dict(color=color, width=trace.line.width + 4),
                    opacity=0.3,
                    hoverinfo='skip',
                    showlegend=False
                )
                shadow_trace2 = go.Scatter(
                    x=trace.x, y=trace.y,
                    mode='lines',
                    line=dict(color=color, width=trace.line.width + 8),
                    opacity=0.15,
                    hoverinfo='skip',
                    showlegend=False
                )
                fig.add_trace(shadow_trace1)
                fig.add_trace(shadow_trace2)
    return fig

# =============================================================================
# FIELD INTEGRITY — Vertical Bar Chart (sorted by missing count)
# =============================================================================

def plot_field_integrity(integrity_df: pd.DataFrame) -> go.Figure:
    """
    Vertical bar chart sorted by missing count descending.
    Color gradient: Light Green (low % null) → Teal → Deep Blue (high % null).
    """
    # Sort by missing descending
    df_sorted = integrity_df.sort_values("Missing", ascending=False).reset_index(drop=True)

    # Compute relative % for color (0 = no missing, 100 = max missing)
    max_missing = df_sorted["Missing"].max()
    pct_null = (df_sorted["Missing"] / max_missing * 100) if max_missing > 0 else df_sorted["Missing"] * 0

    # Customdata for hover tooltip
    hover_cols = ["Missing", "Noise", "Safe Zone Violations", "Real Fill Rate (%)"] if "Noise" in df_sorted.columns else ["Missing", "Safe Zone Violations", "Real Fill Rate (%)"]
    customdata = df_sorted[hover_cols].values

    if "Noise" in df_sorted.columns:
        hover_tpl = (
            "<b>%{x}</b><br>"
            "Missing: <b>%{y:,}</b><br>"
            "Noise: %{customdata[1]}<br>"
            "Safe Zone Violations: %{customdata[2]}<br>"
            "Real Fill Rate: %{customdata[3]}%"
            "<extra></extra>"
        )
    else:
        hover_tpl = (
            "<b>%{x}</b><br>"
            "Missing: <b>%{y:,}</b><br>"
            "Safe Zone Violations: %{customdata[1]}<br>"
            "Real Fill Rate: %{customdata[2]}%"
            "<extra></extra>"
        )

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=df_sorted["Column"],
        y=df_sorted["Missing"],
        marker=dict(
            color=pct_null,
            colorscale=[
                [0.0, "rgba(0,0,0,0)"],     # 0% Missing (Perfect) is transparent
                [0.001, "rgba(255,255,255,0.05)"], # Just above 0%
                [0.1, CHART_COLORWAY[1]],   # Sky Blue
                [0.6, CHART_COLORWAY[2]],   # Amber
                [1.0, CHART_COLORWAY[3]],   # Coral Red (High Missing)
            ],
            colorbar=dict(
                title=dict(text="% Missing", font=dict(color=MUTED_COLOR, size=11)),
                tickfont=dict(color=MUTED_COLOR, size=10),
                thickness=12,
                len=0.6,
                outlinewidth=0,
                x=1.02,
            ),
            line=dict(width=0),
        ),
        customdata=customdata,
        hovertemplate=hover_tpl,
        showlegend=False,
    ))

    fig.update_layout(
        **CHART_LAYOUT,
        height=440,
        bargap=0.15,
        xaxis=dict(
            gridcolor="rgba(0,0,0,0)",
            tickfont=dict(size=11, color=MUTED_COLOR),
            tickangle=-45,
        ),
        yaxis=dict(
            gridcolor=GRID_COLOR,
            zerolinecolor=ZERO_LINE_COLOR,
            zeroline=False,
            title=dict(text="Missing Values", font=dict(color=MUTED_COLOR, size=12)),
            tickfont=dict(size=11, color=MUTED_COLOR),
        ),
    )
    return apply_global_theme(fig)


# =============================================================================
# DATA QUALITY MATRIX — Heatmap
# =============================================================================

def plot_quality_matrix(quality_df: pd.DataFrame) -> go.Figure:
    """
    Heatmap chất lượng dữ liệu: Completeness / Accuracy / Consistency per column.
    Dark Slate → Teal → Emerald Green gradient (matching Fill Rate palette).
    """
    z = quality_df.values
    col_names = quality_df.index.tolist()
    dim_names = quality_df.columns.tolist()

    text_matrix = quality_df.round(1).astype(str).values

    # Sequential colorscale: Red (0) -> Amber (30) -> Sky Blue (70) -> Transparent (100)
    quality_colorscale = [
        [0.0, CHART_COLORWAY[3]],     # Error state (Coral Red)
        [0.3, CHART_COLORWAY[2]],     # Warning state (Amber)
        [0.7, CHART_COLORWAY[1]],     # Good state (Sky Blue)
        [0.999, "rgba(255, 255, 255, 0.05)"], # Just below 100%
        [1.0, "rgba(0, 0, 0, 0)"],    # 100% Perfect state is completely transparent
    ]

    fig = go.Figure(data=go.Heatmap(
        z=z,
        x=dim_names,
        y=col_names,
        colorscale=quality_colorscale,
        zmin=0, zmax=100,
        text=text_matrix,
        texttemplate="%{text}%",
        textfont=dict(size=13, color=BRIGHT_TEXT, family='"Inter", sans-serif'),
        xgap=3,
        ygap=3,
        hovertemplate=(
            "<b>%{y}</b><br>"
            "%{x}: <b>%{z:.1f}%</b>"
            "<extra></extra>"
        ),
        colorbar=dict(
            tickfont=dict(color=MUTED_COLOR, family='"Inter", sans-serif', size=10),
            title=dict(text="Score %", font=dict(color=MUTED_COLOR, size=11)),
            thickness=12,
            len=0.6,
            ticksuffix="%",
            outlinewidth=0,
        ),
    ))

    height = max(400, len(col_names) * 40)

    fig.update_layout(
        **CHART_LAYOUT,
        height=height,
        xaxis=dict(
            gridcolor="rgba(0,0,0,0)",
            side="top",
            tickfont=dict(size=13, color=BRIGHT_TEXT, family='"Inter", sans-serif'),
            tickangle=0,
        ),
        yaxis=dict(
            gridcolor="rgba(0,0,0,0)",
            autorange="reversed",
            tickfont=dict(size=12, color=MUTED_COLOR),
        ),
    )
    fig.layout.plot_bgcolor = "rgba(0,0,0,0)"
    fig.layout.paper_bgcolor = "rgba(0,0,0,0)"
    fig.layout.margin.update(l=20, r=20, t=50, b=20)
    return apply_global_theme(fig)


# =============================================================================
# CORRELATION — Heatmap
# =============================================================================

def plot_correlation_matrix(
    corr_matrix: pd.DataFrame,
    threshold: float = 0.0,
) -> go.Figure:
    """
    Heatmap tương quan với custom neon colorscale và annotation.

    Args:
        corr_matrix: Square symmetric correlation DataFrame (output of
                     compute_correlation_matrix).
        threshold:   Pairs with |r| < threshold are shown as transparent
                     (NaN) to reduce visual noise.  Default 0.0 (show all).
    """
    import numpy as _np

    # Work on a copy — mask diagonal (always 1.0) and weak pairs
    z = corr_matrix.values.astype(float).copy()

    # Mask diagonal → renders as transparent (no visual bias from self-correlation)
    _np.fill_diagonal(z, _np.nan)

    # Optional threshold masking for weak correlations
    if threshold > 0.0:
        weak = _np.abs(z) < threshold
        z[weak] = _np.nan

    # Text annotations: diagonal → "—", weak → "", others → r value
    n_cols = len(corr_matrix.columns)
    text_matrix = []
    raw = corr_matrix.values
    for i in range(n_cols):
        row_text = []
        for j in range(n_cols):
            if i == j:
                row_text.append("—")
            elif threshold > 0.0 and abs(raw[i, j]) < threshold:
                row_text.append("")
            else:
                row_text.append(f"{raw[i, j]:.2f}")
        text_matrix.append(row_text)

    # Diverging colorscale: Red (-1) → near-zero transparent → Sky Blue (+1)
    neon_colorscale = [
        [0.0,   RED],
        [0.25,  ORANGE],
        [0.45,  "rgba(255, 91, 92, 0.15)"],
        [0.50,  "rgba(255, 255, 255, 0.02)"],
        [0.55,  "rgba(91, 134, 229, 0.15)"],
        [1.0,   BLUE],
    ]

    n = len(corr_matrix.columns)
    height = max(420, n * 42)

    fig = go.Figure(data=go.Heatmap(
        z=z,
        x=corr_matrix.columns,
        y=corr_matrix.index,
        colorscale=neon_colorscale,
        zmin=-1, zmax=1,
        text=text_matrix,
        texttemplate="%{text}",
        textfont=dict(size=11, color=BRIGHT_TEXT),
        hovertemplate=(
            "<b>%{x}</b> vs <b>%{y}</b><br>"
            "Correlation: <b>%{z:.3f}</b>"
            "<extra></extra>"
        ),
        colorbar=dict(
            tickfont=dict(color=MUTED_COLOR, family='"Inter", sans-serif'),
            title=dict(text="r", font=dict(color=MUTED_COLOR)),
            thickness=12,
            len=0.8,
        ),
    ))

    fig.update_layout(
        **CHART_LAYOUT,
        height=height,
        xaxis=dict(gridcolor=GRID_COLOR, side="bottom"),
        yaxis=dict(gridcolor=GRID_COLOR, autorange="reversed"),
    )
    return apply_global_theme(fig)


# =============================================================================
# OUTLIER DISTRIBUTION — For Risk Inspector
# =============================================================================

def plot_outlier_distribution(
    s: pd.Series,
    risk_df: pd.DataFrame,
    method: str,
    lang: str = "en"
) -> go.Figure:
    """
    Plots the numeric distribution of a column with a Bell Curve overlay
    to indicate skewness. Also marks the boundaries of the selected outlier method.
    """
    import numpy as np
    from scipy.stats import norm
    
    clean_s = s.dropna()
    if len(clean_s) < 3 or clean_s.std() == 0:
        # Fallback to simple histogram if not enough data
        fig = px.histogram(clean_s, nbins=50, color_discrete_sequence=[BLUE])
        fig.update_layout(**CHART_LAYOUT, height=300)
        return apply_global_theme(fig)

    mean_val = clean_s.mean()
    std_val = clean_s.std()
    
    # 1. Histogram
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=clean_s,
        nbinsx=50,
        name="Data Range",
        marker_color=BLUE,
        opacity=0.7,
        histnorm='probability density'
    ))

    # 2. Bell Curve Overlay
    x_min, x_max = clean_s.min(), clean_s.max()
    x_range = np.linspace(x_min, x_max, 200)
    y_norm = norm.pdf(x_range, mean_val, std_val)
    
    fig.add_trace(go.Scatter(
        x=x_range,
        y=y_norm,
        mode="lines",
        name="Normal Curve",
        line=dict(color=ORANGE, width=2, dash="dash"),
        hovertemplate="Theoretical Normal<extra></extra>"
    ))

    # 3. Method Boundaries
    # Try to extract the min/max of the un-flagged region
    # If risk_df is empty, boundaries are outside the min/max
    outlier_vals = risk_df[s.name].dropna() if not risk_df.empty and s.name in risk_df.columns else []
    
    # Very rough calculation of fences for visual reference
    fence_lo, fence_hi = None, None
    if method == "iqr":
        Q1 = clean_s.quantile(0.25)
        Q3 = clean_s.quantile(0.75)
        IQR = Q3 - Q1
        fence_lo, fence_hi = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
    elif method == "zscore":
        fence_lo, fence_hi = mean_val - 3 * std_val, mean_val + 3 * std_val
    elif method == "modified_zscore":
        median = clean_s.median()
        mad = (clean_s - median).abs().median()
        if mad != 0:
            fence_lo = median - 3 * mad / 0.6745
            fence_hi = median + 3 * mad / 0.6745

    if fence_lo is not None and fence_hi is not None:
        # Draw threshold lines if they fall within the visible plot area
        if fence_lo > x_min - (x_max - x_min)*0.2:
            fig.add_vline(x=fence_lo, line_width=2, line_dash="solid", line_color=RED, 
                         annotation_text="Lower Limit", annotation_position="top left", annotation_font_color=RED)
        if fence_hi < x_max + (x_max - x_min)*0.2:
            fig.add_vline(x=fence_hi, line_width=2, line_dash="solid", line_color=RED, 
                         annotation_text="Upper Limit", annotation_position="top right", annotation_font_color=RED)

    # Copy base layout to avoid mutating the global default
    layout_cfg = CHART_LAYOUT.copy()
    layout_cfg["margin"] = dict(t=20, b=20, l=10, r=10)
    layout_cfg["legend"] = dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    layout_cfg["showlegend"] = True
    
    fig.update_layout(
        **layout_cfg,
        height=320,
        xaxis_title=s.name,
        yaxis_title="Probability Density",
        barmode='overlay',
        bargap=0.1  # Adds space between histogram bars
    )
    
    return apply_global_theme(fig)

# =============================================================================
# HISTOGRAM — For Preprocessing Comparison
# =============================================================================

def plot_histogram(df: pd.DataFrame, column: str, color: str = BLUE) -> go.Figure:
    """Histogram with neon-themed layout."""
    fig = px.histogram(df, x=column, nbins=30, color_discrete_sequence=[color])
    fig.update_layout(
        **CHART_LAYOUT,
        height=350,
        xaxis=dict(gridcolor=GRID_COLOR, zerolinecolor=ZERO_LINE_COLOR),
        yaxis=dict(gridcolor=GRID_COLOR, zerolinecolor=ZERO_LINE_COLOR),
    )
    return apply_global_theme(fig)

# =============================================================================
# BOXPLOT — For Outlier Detection PDF Exports
# =============================================================================

def plot_box(df: pd.DataFrame, column: str, color: str = ORANGE) -> go.Figure:
    """Horizontal boxplot revealing quartiles and outliers."""
    fig = px.box(df, x=column, color_discrete_sequence=[color], points="outliers")
    layout_args = CHART_LAYOUT.copy()
    layout_args.update(dict(
        height=250,
        xaxis=dict(gridcolor=GRID_COLOR, zerolinecolor=ZERO_LINE_COLOR),
        yaxis=dict(gridcolor=GRID_COLOR, zerolinecolor=ZERO_LINE_COLOR, showticklabels=False),
        showlegend=False,
        margin=dict(l=10, r=20, t=10, b=30),
    ))
    fig.update_layout(**layout_args)
    return apply_global_theme(fig)


# =============================================================================
# EDA — BAR DISTRIBUTION (Horizontal)
# =============================================================================

def plot_bar_distribution(df: pd.DataFrame, column: str, top_n: int = 15,
                          color: str = BLUE) -> go.Figure:
    """Horizontal bar chart for categorical column value counts."""
    counts = df[column].value_counts().head(top_n).sort_values()

    fig = go.Figure(go.Bar(
        x=counts.values,
        y=counts.index.astype(str),
        orientation="h",
        marker=dict(
            color=color,
            line=dict(width=0),
            opacity=0.85,
        ),
        hovertemplate="<b>%{y}</b><br>Count: %{x:,}<extra></extra>",
    ))

    fig.update_layout(
        **CHART_LAYOUT,
        height=max(350, len(counts) * 30),
        bargap=0.2,
        xaxis=dict(gridcolor=GRID_COLOR, zerolinecolor=ZERO_LINE_COLOR,
                    title=dict(text="Count", font=dict(color=MUTED_COLOR, size=12)),
                    tickfont=dict(color=MUTED_COLOR, size=11)),
        yaxis=dict(gridcolor="rgba(0,0,0,0)",
                    tickfont=dict(color=MUTED_COLOR, size=11)),
    )
    return apply_global_theme(fig)


# =============================================================================
# EDA — HISTOGRAM WITH INCOME OVERLAY
# =============================================================================

def plot_histogram_overlay(df: pd.DataFrame, column: str,
                            hue: str = None) -> go.Figure:
    """Histogram with optional income/category overlay using stacked bars."""
    if hue and hue in df.columns:
        groups = df[hue].dropna().unique()
        colors = GRADIENT_5[:len(groups)]
        fig = go.Figure()
        for g, clr in zip(sorted(groups), colors):
            subset = df[df[hue] == g][column].dropna()
            fig.add_trace(go.Histogram(
                x=subset, name=str(g), marker_color=clr,
                opacity=0.75, nbinsx=30,
                hovertemplate=f"<b>{g}</b><br>{{%{{x}}}}<br>Count: %{{y:,}}<extra></extra>",
            ))
        fig.update_layout(barmode="overlay")
    else:
        fig = go.Figure(go.Histogram(
            x=df[column].dropna(), nbinsx=30,
            marker_color=BLUE, opacity=0.85,
            hovertemplate="<b>%{x}</b><br>Count: %{y:,}<extra></extra>",
        ))

    fig.update_layout(
        **CHART_LAYOUT,
        height=400,
        xaxis=dict(gridcolor=GRID_COLOR, zerolinecolor=ZERO_LINE_COLOR,
                    title=dict(text=column, font=dict(color=MUTED_COLOR, size=12)),
                    tickfont=dict(color=MUTED_COLOR, size=11)),
        yaxis=dict(gridcolor=GRID_COLOR, zerolinecolor=ZERO_LINE_COLOR,
                    title=dict(text="Count", font=dict(color=MUTED_COLOR, size=12)),
                    tickfont=dict(color=MUTED_COLOR, size=11)),
    )
    fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    return apply_global_theme(fig)


# =============================================================================
# EDA — STACKED / GROUPED BAR
# =============================================================================

def plot_stacked_bar(df: pd.DataFrame, column: str, hue: str,
                      mode: str = "group", top_n: int = 15) -> go.Figure:
    """Stacked or grouped bar chart showing category × hue breakdown."""
    ct = pd.crosstab(df[column], df[hue])
    ct = ct.loc[ct.sum(axis=1).nlargest(top_n).index]
    ct = ct.sort_values(by=ct.columns.tolist(), ascending=True)

    colors = GRADIENT_5[:len(ct.columns)]
    fig = go.Figure()
    for i, col_name in enumerate(ct.columns):
        fig.add_trace(go.Bar(
            y=ct.index.astype(str),
            x=ct[col_name],
            name=str(col_name),
            orientation="h",
            marker_color=colors[i % len(colors)],
            hovertemplate=f"<b>%{{y}}</b><br>{col_name}: %{{x:,}}<extra></extra>",
        ))

    fig.update_layout(
        **CHART_LAYOUT,
        barmode=mode,
        height=max(400, len(ct) * 32),
        bargap=0.2,
        xaxis=dict(gridcolor=GRID_COLOR, zerolinecolor=ZERO_LINE_COLOR,
                    title=dict(text="Count", font=dict(color=MUTED_COLOR, size=12)),
                    tickfont=dict(color=MUTED_COLOR, size=11)),
        yaxis=dict(gridcolor="rgba(0,0,0,0)",
                    tickfont=dict(color=MUTED_COLOR, size=11)),
    )
    fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    return apply_global_theme(fig)


# =============================================================================
# EDA — DONUT CHART (General)
# =============================================================================

def plot_donut(df: pd.DataFrame, column: str, top_n: int = 10) -> go.Figure:
    """Donut chart for categorical distributions with neon styling."""
    counts = df[column].value_counts().head(top_n)
    labels = counts.index.astype(str).tolist()
    values = counts.values.tolist()
    total = sum(values)

    colors = GRADIENT_8[:len(labels)]

    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.55,
        marker=dict(colors=colors, line=dict(color=BG_PAPER, width=2)),
        textfont=dict(family='"Inter", sans-serif', size=11, color=BRIGHT_TEXT),
        textinfo="percent+label",
        textposition="outside",
        pull=[0.02] * len(labels),
        hovertemplate=(
            "<b>%{label}</b><br>"
            "Count: %{value:,}<br>"
            "Share: %{percent}"
            "<extra></extra>"
        ),
        rotation=90,
    )])

    fig.update_layout(
        **CHART_LAYOUT,
        showlegend=False,
        height=400,
        annotations=[dict(
            text=(
                f"<b style='font-size:22px;color:{BRIGHT_TEXT}'>{total:,}</b><br>"
                f"<span style='font-size:10px;color:{MUTED_COLOR}'>Total</span>"
            ),
            x=0.5, y=0.5,
            font=dict(size=20, color=BRIGHT_TEXT, family='"Inter", sans-serif'),
            showarrow=False,
        )],
    )
    return apply_global_theme(fig)


# =============================================================================
# EDA — VIOLIN PLOT
# =============================================================================

def plot_violin(df: pd.DataFrame, column: str, hue: str = None) -> go.Figure:
    """Violin plot for numeric distributions, optionally split by hue."""
    if hue and hue in df.columns:
        fig = px.violin(
            df, x=hue, y=column, color=hue, box=True, points=False,
            color_discrete_sequence=GRADIENT_5,
        )
    else:
        fig = px.violin(
            df, y=column, box=True, points=False,
            color_discrete_sequence=[BLUE],
        )

    fig.update_layout(
        **CHART_LAYOUT,
        height=400,
        showlegend=False,
        xaxis=dict(gridcolor="rgba(0,0,0,0)",
                    tickfont=dict(color=MUTED_COLOR, size=11)),
        yaxis=dict(gridcolor=GRID_COLOR, zerolinecolor=ZERO_LINE_COLOR,
                    title=dict(text=column, font=dict(color=MUTED_COLOR, size=12)),
                    tickfont=dict(color=MUTED_COLOR, size=11)),
    )
    return apply_global_theme(fig)


# =============================================================================
# EDA — TREEMAP
# =============================================================================

def plot_treemap(df: pd.DataFrame, column: str, top_n: int = 15) -> go.Figure:
    """Treemap showing top N categories by count."""
    counts = df[column].value_counts().head(top_n).reset_index()
    counts.columns = ["category", "count"]

    fig = px.treemap(
        counts, path=["category"], values="count",
        color="count",
        color_continuous_scale=[
            [0.0, "rgba(91,134,229,0.3)"],
            [0.5, BLUE],
            [1.0, "#93C5FD"],
        ],
    )

    fig.update_layout(
        **CHART_LAYOUT,
        height=450,
        coloraxis_showscale=False,
    )
    fig.update_layout(margin=dict(l=10, r=10, t=30, b=10))
    fig.update_traces(
        textfont=dict(family='"Inter", sans-serif', size=13, color=BRIGHT_TEXT),
        hovertemplate="<b>%{label}</b><br>Count: %{value:,}<extra></extra>",
        marker=dict(line=dict(width=1.5, color="rgba(0,0,0,0.3)")),
    )
    return fig


# =============================================================================
# EDA — SUNBURST
# =============================================================================

def plot_sunburst(df: pd.DataFrame, path_cols: list) -> go.Figure:
    """Sunburst chart for hierarchical categorical breakdown."""
    fig = px.sunburst(
        df, path=path_cols,
        color_discrete_sequence=GRADIENT_8,
    )

    fig.update_layout(
        **CHART_LAYOUT,
        height=500,
    )
    fig.update_layout(margin=dict(l=10, r=10, t=20, b=10))
    fig.update_traces(
        textfont=dict(family='"Inter", sans-serif', size=11, color=BRIGHT_TEXT),
        hovertemplate="<b>%{label}</b><br>Count: %{value:,}<br>Share: %{percentParent:.1%}<extra></extra>",
        insidetextorientation="radial",
        marker=dict(line=dict(width=1, color="rgba(0,0,0,0.3)")),
    )
    return fig


# =============================================================================
# EDA — SCATTER PLOT
# =============================================================================

def plot_scatter(df: pd.DataFrame, x_col: str, y_col: str,
                  color_col: str = None) -> go.Figure:
    """Scatter plot for exploring two numeric variables, optionally colored."""
    if color_col and color_col in df.columns:
        fig = px.scatter(
            df, x=x_col, y=y_col, color=color_col,
            color_discrete_sequence=GRADIENT_5,
            opacity=0.6,
        )
    else:
        fig = px.scatter(
            df, x=x_col, y=y_col,
            color_discrete_sequence=[BLUE],
            opacity=0.6,
        )

    fig.update_traces(marker=dict(size=5, line=dict(width=0)))

    fig.update_layout(
        **CHART_LAYOUT,
        height=450,
        xaxis=dict(gridcolor=GRID_COLOR, zerolinecolor=ZERO_LINE_COLOR,
                    title=dict(text=x_col, font=dict(color=MUTED_COLOR, size=12)),
                    tickfont=dict(color=MUTED_COLOR, size=11)),
        yaxis=dict(gridcolor=GRID_COLOR, zerolinecolor=ZERO_LINE_COLOR,
                    title=dict(text=y_col, font=dict(color=MUTED_COLOR, size=12)),
                    tickfont=dict(color=MUTED_COLOR, size=11)),
    )
    fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    return apply_global_theme(fig)

# =============================================================================
# ISSUE COMPOSITION — Horizontal Bar + Donut (Audit Page)
# =============================================================================

def plot_issue_composition(
    issue_dict: dict,
    total_values: int,
    display_overrides: dict = None,
) -> go.Figure:
    """
    Dual-panel chart for Issue Composition.
    Left (68%): Horizontal bar chart — bars colored by severity %.
    Right (32%): Donut showing Clean vs Affected cells.

    Args:
        issue_dict:        {label: count_in_cells} for % calculation.
        total_values:      Total cells in the dataset (rows x columns).
        display_overrides: {label: original_count} for bar text display.
    """
    from plotly.subplots import make_subplots

    # 1. Sort descending by count
    items = sorted(issue_dict.items(), key=lambda x: x[1], reverse=True)
    if display_overrides is None:
        display_overrides = {}
    labels = [it[0] for it in items]
    counts = [int(it[1]) for it in items]
    # Display counts: use original values (e.g. rows) where overridden
    display_counts = [display_overrides.get(lbl, cnt) for lbl, cnt in zip(labels, counts)]
    pcts   = [
        round(c / total_values * 100, 3) if total_values > 0 else 0.0
        for c in counts
    ]

    # 2. Severity color mapping
    _CRIT = STATUS_COLORS["critical"]["hex"]
    _WARN = STATUS_COLORS["warning"]["hex"]
    _INFO = CHART_COLORWAY[1]
    _OK   = STATUS_COLORS["success"]["hex"]

    def _color(pct):
        if pct >= 5:  return _CRIT
        if pct >= 2:  return _WARN
        if pct > 0:   return _INFO
        return _OK

    bar_colors = [_color(p) for p in pcts]

    # 3. Subplots: bar (left) + donut (right)
    fig = make_subplots(
        rows=1, cols=2,
        column_widths=[0.68, 0.32],
        specs=[[{"type": "xy"}, {"type": "domain"}]],
        horizontal_spacing=0.04,
    )

    # 4a. Horizontal bars
    hover_tpls = [
        f"<b>{lbl}</b><br>Count: <b>{dc:,}</b><br>Share: <b>{p:.3f}%</b><extra></extra>"
        for lbl, dc, p in zip(labels, display_counts, pcts)
    ]
    fig.add_trace(
        go.Bar(
            x=pcts,
            y=labels,
            orientation="h",
            marker=dict(color=bar_colors, opacity=0.88, line=dict(width=0)),
            text=[f"{p:.2f}%  ({dc:,})" for p, dc in zip(pcts, display_counts)],
            textposition="outside",
            textfont=dict(size=11, color=bar_colors, family='"Inter", sans-serif'),
            cliponaxis=False,
            hovertemplate=hover_tpls,
            showlegend=False,
        ),
        row=1, col=1,
    )

    # 4b. Donut — Clean vs Affected
    total_issues = sum(counts)
    clean_cells  = max(0, total_values - total_issues)
    clean_pct    = round(clean_cells / total_values * 100, 1) if total_values > 0 else 100.0

    fig.add_trace(
        go.Pie(
            labels=["Clean", "Affected"],
            values=[clean_cells, total_issues],
            hole=0.62,
            marker=dict(colors=[_OK, _CRIT], line=dict(color="rgba(0,0,0,0.3)", width=1.5)),
            textfont=dict(family='"Inter", sans-serif', size=11, color=BRIGHT_TEXT),
            textinfo="label+percent",
            textposition="outside",
            rotation=90,
            pull=[0, 0.04],
            domain=dict(x=[0, 1], y=[0.1, 0.9]),
            hovertemplate=(
                "<b>%{label}</b><br>Cells: %{value:,}<br>Share: %{percent}<extra></extra>"
            ),
            showlegend=False,
        ),
        row=1, col=2,
    )

    # 5. Centre annotation
    fig.add_annotation(
        text=f"<b>{clean_pct}%</b><br><span style='font-size:9px'>CLEAN</span>",
        xref="paper", yref="paper",
        x=0.875, y=0.50,
        showarrow=False,
        font=dict(size=15, color=BRIGHT_TEXT, family='"Inter", sans-serif'),
        align="center",
    )

    # 6. Layout & axes
    # Apply base theme first, then override margin/legend separately
    # (CHART_LAYOUT already contains margin & legend — cannot pass them twice)
    max_pct = max(pcts) if any(p > 0 for p in pcts) else 1.0
    x_end = max_pct * 1.55

    fig.update_layout(**CHART_LAYOUT)
    fig.update_layout(
        height=max(260, len(items) * 52 + 60),
        bargap=0.28,
        margin=dict(l=10, r=40, t=10, b=40),
    )
    fig.update_xaxes(
        range=[0, x_end], gridcolor=GRID_COLOR, zerolinecolor=ZERO_LINE_COLOR,
        ticksuffix="%", tickfont=dict(size=10, color=MUTED_COLOR),
        title=dict(text="% of Total Dataset Cells", font=dict(color=MUTED_COLOR, size=11)),
        row=1, col=1,
    )
    fig.update_yaxes(
        gridcolor="rgba(0,0,0,0)",
        tickfont=dict(size=12, color=BRIGHT_TEXT, family='"Inter", sans-serif'),
        autorange="reversed",
        row=1, col=1,
    )
    return fig
