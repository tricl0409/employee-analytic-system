"""
preprocessing.py — Data Preprocessing Page (slim orchestrator)

Responsibilities of this module:
  • Page layout, workspace guards, session-state management.
  • Defining `_compute_outlier_preview_row` — the single logic source used by
    BOTH the preview tab and the pipeline executor to ensure consistency.
  • Running the 6-step fixed pipeline (_run_pipeline).

All tab rendering is delegated to UiComponents:
  UiComponents.render_scrubber_tab(df)           → Tab 1
  UiComponents.render_missing_and_dupes_tab(df)  → Tab 2
  UiComponents.render_outlier_tab(df, fn)        → Tab 3

CSS is managed centrally in modules/ui/styles.py (PREPROCESSING_STYLES).
"""

import os
import time

import pandas as pd
import streamlit as st

from modules.core import file_manager, data_engine, preprocessing_engine
from modules.core.file_manager import UPLOADS_DIR, save_dataframe
from modules.core.audit_engine import (
    _get_safe_zones,
    _apply_safe_zone_mask,
    _OUTLIER_METHODS,
    compute_skewness,
    recommend_fill_strategy,
    evaluate_outlier_method,
    default_outlier_threshold,
)
from modules.ui import (
    page_header,
    section_divider,
    workspace_status,
    active_file_scan_progress_bar,
    pipeline_done_banner,
)

from modules.utils.localization import get_text
from modules.utils.helpers import _ensure_workspace_active


# ==============================================================================
# SHARED OUTLIER PREVIEW HELPER
# ==============================================================================

#: Maps human-readable method name → (detection_key, treatment_method, display_label)
_METHOD_INFO: dict[str, tuple[str, str, str]] = {
    "Z-Score":          ("zscore",          "zscore_capping",          "Z-Score Cap"),
    "IQR":              ("iqr",             "iqr_capping",             "IQR Cap"),
    "Modified Z-Score": ("modified_zscore", "modified_zscore_capping", "Modified Z-Score Cap"),
}


def _compute_outlier_preview_row(
    df: pd.DataFrame,
    col: str,
    safe_zones: dict,
) -> dict | None:
    """
    Compute a single outlier-preview row for one numeric column.

    **Single source of truth** used by both:
      - ``render_outlier_tab``  (Tab 3 preview via UiComponents)
      - ``_run_pipeline``       (live execution)

    Algorithm:
      1. Auto-select detection method by skewness (evaluate_outlier_method).
      2. Run the detection mask function to find statistical outliers.
      3. Apply ``_apply_safe_zone_mask`` to exclude values inside safe zones.
      4. Count true outliers (statistical AND outside safe zone).
      5. Determine the planned "Action" label.

    Args:
        df:         Working DataFrame.
        col:        Numeric column name.
        safe_zones: Admin-configured safe-zone bounds from ``_get_safe_zones()``.

    Returns:
        Dict with keys: Column, Skewness, Auto Method, detect_key,
        treatment_method, Outliers Detected, Action, has_safe_zone.
        Returns ``None`` if the column has fewer than 3 non-null values.
    """
    series = df[col].dropna()
    if len(series) < 3:
        return None

    rec         = evaluate_outlier_method(series)
    skew_val    = rec["skewness"]
    method_name = rec["method"]  # "Z-Score" | "IQR" | "Modified Z-Score"

    detect_key, treatment_method, base_action = _METHOD_INFO.get(
        method_name, ("iqr", "iqr_capping", "IQR Cap")
    )

    # Method-aware threshold — consistent with data_audit page
    threshold = default_outlier_threshold(detect_key)

    # Statistical outlier detection
    compute_fn = _OUTLIER_METHODS.get(detect_key, _OUTLIER_METHODS["iqr"])
    stat_mask, _ = compute_fn(series.values, threshold)

    # Safe-zone filtering — values inside the zone are protected
    safe_mask     = _apply_safe_zone_mask(series.values, col)
    target_key    = col.strip().lower().replace(" ", "_")
    has_safe_zone = any(
        zk.strip().lower().replace(" ", "_") == target_key
        for zk in safe_zones
    )

    final_mask      = stat_mask & safe_mask
    outlier_count   = int(final_mask.sum())
    raw_stat_count  = int(stat_mask.sum())
    total_rows      = len(series)
    pct_outlier     = round(outlier_count / total_rows * 100, 2) if total_rows > 0 else 0.0

    # 🛡 badge only when Safe Zone actually filtered out some raw outliers.
    # If raw stat count is already 0, Safe Zone did nothing → no badge.
    safe_zone_active = has_safe_zone and raw_stat_count > 0
    if outlier_count == 0:
        action = "No Outlier Treatment  🛡" if safe_zone_active else "No Outlier Treatment"
    else:
        action = f"{base_action}  🛡" if safe_zone_active else base_action

    return {
        "Column":            col,
        "Min":               round(float(series.min()), 2),
        "Max":               round(float(series.max()), 2),
        "Skewness":          skew_val,
        "Auto Method":       method_name,
        "detect_key":        detect_key,
        "treatment_method":  treatment_method,
        "Outliers Detected": outlier_count,
        "% Outlier":         pct_outlier,
        "Action":            action,
        "has_safe_zone":     has_safe_zone,
    }


# ==============================================================================
# PIPELINE EXECUTOR
# ==============================================================================

def _run_pipeline(
    df: pd.DataFrame,
    engine,
    active_file: str,
    rows_original: int,
    lang: str,
) -> pd.DataFrame:
    """
    Execute the full 8-step fixed preprocessing pipeline with live progress.

    Pipeline order:
      1. Standardize & Type Cast (trim whitespace, normalize casing, convert dtypes)
      2. Noise Cleaning          (clean_noise_values: replace noise tokens with NaN)
      3. Duplicate Removal       (drop_duplicates)
      4. Missing Value Handling  (handle_missing_smart: mean/median/mode)
      5. Outlier Treatment       (handle_outliers, per-column auto-method
                                  via _compute_outlier_preview_row)
      6. Log Transformation      (apply_log_transform: log1p / yeo-johnson)
      7. Binning & Mapping       (apply_binning_mapping: discretize + group)
      8. Feature Encoding        (apply_feature_encoding: label + one-hot)

    On completion, the cleaned DataFrame is saved as
    ``<original_stem>_cleaned.csv`` (auto-incremented if the file already
    exists), the workspace is switched to the new file, and
    ``st.session_state['preprocessing_result']`` is populated.

    Args:
        df:            Working DataFrame (copy of raw data).
        engine:        ``PreprocessingEngine`` class reference.
        active_file:   Basename of the currently active workspace file.
        rows_original: Row count of the raw DataFrame *before* any step,
                       captured by the caller so the completion banner always
                       shows the true pre-pipeline count.
        lang:          Active UI language code.

    Returns:
        Cleaned DataFrame after all 8 steps.
    """
    from modules.core.audit_engine import _get_cat_columns

    safe_zones  = _get_safe_zones()

    STEPS = [
        (":material/tune: Step 1/8 — Standardizing text & converting dtypes...",         "std"),
        (":material/delete: Step 2/8 — Cleaning noise & placeholder values...",          "noise"),
        (":material/content_copy: Step 3/8 — Removing duplicate rows...",                "dupes"),
        (":material/healing: Step 4/8 — Imputing missing values (Mean/Median/Mode)...",  "missing"),
        (":material/square_foot: Step 5/8 — Treating outliers (auto-method)...",         "outliers"),
        (":material/functions: Step 6/8 — Applying log transformations...",              "logtf"),
        (":material/category: Step 7/8 — Binning & mapping features...",                "binmap"),
        (":material/label: Step 8/8 — Encoding categorical features...",                "encode"),
    ]

    progress = st.progress(0, text=STEPS[0][0])

    # ── Capture initial quality metrics ───────────────────────────────────
    initial_missing = int(df.isna().sum().sum())
    initial_dupes   = int(df.duplicated().sum())

    # ── Step 1: Standardize & Type Cast ───────────────────────────────────
    progress.progress(0.05, text=STEPS[0][0])
    df = engine.standardize_and_type_cast(
        df, fix_whitespace=True, fix_casing=True, convert_dtypes=True,
    )
    time.sleep(0.1)

    # ── Step 2: Noise Cleaning ─────────────────────────────────────────
    progress.progress(0.15, text=STEPS[1][0])
    noise_before = df.isna().sum().sum()
    cat_cols = _get_cat_columns(df).tolist()
    df = engine.clean_noise_values(df, strategy="replace_nan", columns=cat_cols)
    noise_cleaned = int(df.isna().sum().sum() - noise_before)
    time.sleep(0.1)

    # ── Step 3: Duplicate Removal ──────────────────────────────────────
    progress.progress(0.27, text=STEPS[2][0])
    n_before_dedup = len(df)
    df = engine.drop_duplicates(df)
    dupes_dropped  = n_before_dedup - len(df)
    time.sleep(0.1)

    # ── Step 4: Missing Value Handling ─────────────────────────────────
    progress.progress(0.38, text=STEPS[3][0])
    missing_before_fill = int(df.isna().sum().sum())
    df = engine.handle_missing_smart(df)
    missing_after_fill = int(df.isna().sum().sum())
    time.sleep(0.1)

    # ── Step 5: Per-column outlier treatment ────────────────────────────────
    # Count total outliers across all columns BEFORE treatment.
    progress.progress(0.50, text=STEPS[4][0])
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    total_outliers_before = 0
    outlier_cols_treated = 0
    for col in numeric_cols:
        row = _compute_outlier_preview_row(df, col, safe_zones)
        if row is None or row["Outliers Detected"] == 0:
            continue
        total_outliers_before += row["Outliers Detected"]
        col_threshold = default_outlier_threshold(row["detect_key"])
        df = engine.handle_outliers(df, row["treatment_method"], [col], col_threshold)
        outlier_cols_treated += 1
    time.sleep(0.1)

    # ── Save cleaned file (after Step 5) ──────────────────────────────────
    progress.progress(0.56, text=":material/save: Saving cleaned dataset...")
    df_cleaned_snapshot = df.copy()          # freeze state before Steps 6-8 mutate
    basename  = active_file.replace("\\", "/").split("/")[-1]
    base_stem = basename.replace(".csv", "")
    cleaned_filename = f"{base_stem}_cleaned.csv"
    counter = 1
    while os.path.exists(os.path.join(UPLOADS_DIR, cleaned_filename)):
        cleaned_filename = f"{base_stem}_cleaned_{counter}.csv"
        counter += 1
    save_dataframe(df_cleaned_snapshot, cleaned_filename)
    df_cleaned_csv = df_cleaned_snapshot.to_csv(index=False).encode("utf-8")

    # ── Step 6: Log Transformation ─────────────────────────────────────
    progress.progress(0.62, text=STEPS[5][0])
    df = engine.apply_log_transform(df)
    time.sleep(0.1)

    # ── Step 7: Binning & Mapping ──────────────────────────────────────
    progress.progress(0.74, text=STEPS[6][0])
    df = engine.apply_binning_mapping(df)
    time.sleep(0.1)

    # ── Step 8: Feature Encoding ──────────────────────────────────────
    progress.progress(0.86, text=STEPS[7][0])
    candidates = engine.get_encoding_preview(df)
    n_label = sum(1 for c in candidates if c["Encoding"] == "Label Encoding")
    n_onehot = sum(1 for c in candidates if c["Encoding"] == "One-Hot (drop_first)")
    n_redundant = sum(1 for c in candidates if c["Encoding"] == "Drop (Redundant)")
    df = engine.apply_feature_encoding(df, candidates)
    time.sleep(0.1)

    # ── Save encoded file (after Step 8) ──────────────────────────────────
    progress.progress(0.95, text=":material/save: Saving encoded dataset...")
    encoded_filename = f"{base_stem}_encoded.csv"
    counter = 1
    while os.path.exists(os.path.join(UPLOADS_DIR, encoded_filename)):
        encoded_filename = f"{base_stem}_encoded_{counter}.csv"
        counter += 1
    save_dataframe(df, encoded_filename)
    df_encoded_csv = df.to_csv(index=False).encode("utf-8")

    progress.progress(1.0, text=":material/check_circle: Preprocessing complete!")
    time.sleep(0.4)
    progress.empty()

    # ── Compute "after" quality metrics from cleaned data (Step 5) ─────
    after_missing = int(df_cleaned_snapshot.isna().sum().sum())
    after_dupes   = int(df_cleaned_snapshot.duplicated().sum())

    # Re-compute noise after cleaning
    _cat_after = df_cleaned_snapshot.select_dtypes(include=["object"]).columns.tolist()
    after_noise = 0
    for _col_name in _cat_after:
        _noise_mask = df_cleaned_snapshot[_col_name].astype(str).str.strip().str.match(
            r"^[\?\-\.\!\#\*]+$|^(na|n/a|none|null|undefined|unknown|missing|\-)$",
            case=False, na=False,
        )
        after_noise += int(_noise_mask.sum())

    # Re-compute outliers after treatment
    after_outliers = 0
    for _col_name in df_cleaned_snapshot.select_dtypes(include=["number"]).columns:
        _row = _compute_outlier_preview_row(df_cleaned_snapshot, _col_name, safe_zones)
        if _row is not None:
            after_outliers += _row["Outliers Detected"]

    # ── Update session state ──────────────────────────────────────────────
    # Workspace switches to the _cleaned file (data quality focus)
    st.session_state["active_file"]          = cleaned_filename
    st.session_state["_preprocessing_file"]  = cleaned_filename
    st.session_state["cleaned_data"]         = None   # force reload on next visit
    st.session_state["preprocessing_done"]   = True
    st.session_state["preprocessing_result"] = {
        "cleaned_filename":  cleaned_filename,
        "encoded_filename":  encoded_filename,
        "rows_before":       rows_original,
        "rows_after":        len(df),
        "dupes_dropped":     dupes_dropped,
        "df_cleaned_csv":    df_cleaned_csv,
        "df_encoded_csv":    df_encoded_csv,
        # Comparison table — all "after" values computed from df_cleaned_snapshot
        "comparison": [
            ("Missing Values",  initial_missing + noise_cleaned, after_missing),
            ("Noise Values",    noise_cleaned, after_noise),
            ("Duplicate Rows",  initial_dupes, after_dupes),
            ("Outliers",        total_outliers_before, after_outliers),
        ],
        # Encoding summary
        "n_label_encoded":     n_label,
        "n_onehot_encoded":    n_onehot,
        "n_redundant_dropped": n_redundant,
        # Correlation matrix (post-encoding, for target correlation chart)
        "corr_after":  df.select_dtypes(include=["number"]).corr(),
    }

    return df


# ==============================================================================
# PIPELINE STEP DEFINITIONS
# ==============================================================================

_STEP_DEFS = [
    {
        "num": 1,
        "title": "Standardize & Type Cast",
        "desc": "Trim, normalize casing, convert dtypes",
        "icon": ":material/tune:",
        "color": "#3B82F6",
    },
    {
        "num": 2,
        "title": "Noise Cleaning",
        "desc": "Replace noise & placeholder values with NaN",
        "icon": ":material/delete:",
        "color": "#EF4444",
    },
    {
        "num": 3,
        "title": "Duplicate Removal",
        "desc": "Detect & remove exact duplicate rows",
        "icon": ":material/content_copy:",
        "color": "#F59E0B",
    },
    {
        "num": 4,
        "title": "Missing Value Imputation",
        "desc": "Smart imputation: mean / median / mode",
        "icon": ":material/healing:",
        "color": "#F27024",
    },
    {
        "num": 5,
        "title": "Outlier Treatment",
        "desc": "Auto-detect & cap statistical outliers",
        "icon": ":material/square_foot:",
        "color": "#8B5CF6",
    },
    {
        "num": 6,
        "title": "Log Transformation",
        "desc": "Reduce skewness via log1p / Yeo-Johnson",
        "icon": ":material/functions:",
        "color": "#7FB135",
    },
    {
        "num": 7,
        "title": "Binning & Mapping",
        "desc": "Discretize numerics & group categories",
        "icon": ":material/category:",
        "color": "#10B981",
    },
    {
        "num": 8,
        "title": "Feature Encoding",
        "desc": "Encode categoricals to numeric (Label + One-Hot)",
        "icon": ":material/label:",
        "color": "#EC4899",
    },
]


def _hex_to_rgb(hex_color: str) -> str:
    """Convert '#RRGGBB' → 'R,G,B' for CSS rgba()."""
    h = hex_color.lstrip("#")
    return f"{int(h[0:2], 16)},{int(h[2:4], 16)},{int(h[4:6], 16)}"


def _render_pipeline_sidebar(active_step: int) -> int:
    """Render the vertical pipeline steps using ``st.radio`` + CSS styling.

    Returns:
        The 1-based step number currently selected.
    """
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
    for idx, step_def in enumerate(_STEP_DEFS, start=1):
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
        f"0{s['num']}  —  {s['title']}" for s in _STEP_DEFS
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


def _render_detail_panel(
    step: int,
    df: pd.DataFrame,
    compute_fn,
) -> None:
    """Render the right-panel detail view for the selected pipeline step."""
    from modules.core.audit_engine import (
        _compute_noise_mask, _get_cat_columns, _get_safe_zones,
    )

    step_info = _STEP_DEFS[step - 1]
    rgb = _hex_to_rgb(step_info["color"])
    col_hex = step_info["color"]

    # ── Consistent section header ─────────────────────────────────────────
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

    # ── Helper: info box ──────────────────────────────────────────────────
    def _info_box(html_content: str, accent: str = col_hex) -> None:
        a_rgb = _hex_to_rgb(accent)
        st.markdown(
            f'<div style="margin:4px 0 12px 0; padding:12px 16px;'
            f' background:rgba({a_rgb},0.08);'
            f' border-left:3px solid rgba({a_rgb},0.4);'
            f' border-radius:0 8px 8px 0;'
            f' font-size:0.78rem; color:rgba(255,255,255,0.45); line-height:1.9;">'
            f'{html_content}</div>',
            unsafe_allow_html=True,
        )

    # ── Helper: metric row ────────────────────────────────────────────────
    def _metric_row(items: list[tuple[str, str, str]]) -> None:
        """Render a horizontal row of metric cards. items = [(label, value, color), ...]."""
        cards = ""
        for label, value, color in items:
            c_rgb = _hex_to_rgb(color)
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

    # ── Helper: skip / all-clear card ─────────────────────────────────
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

    # ── Step 1: Standardize & Type Cast ────────────────────────────────────
    if step == 1:
        from modules.core.audit_engine import _get_cat_columns as _get_cat_cols_std

        cat_cols = _get_cat_cols_std(df).tolist()

        # ── Text issues (whitespace + casing) ─────────────────────────────
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

        # ── Dtype conversion candidates ───────────────────────────────────
        type_cast_rows = preprocessing_engine.PreprocessingEngine.get_type_cast_preview(df)
        n_type_casts = len(type_cast_rows)

        # ── Sub-header helper ─────────────────────────────────────────────
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

        # ══════════════════════════════════════════════════════════════════
        # 1a. TEXT ISSUES — whitespace & casing
        # ══════════════════════════════════════════════════════════════════
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

        # ══════════════════════════════════════════════════════════════════
        # 1b. DTYPE ISSUES — type conversions
        # ══════════════════════════════════════════════════════════════════
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
                '<b style="color:#F59E0B;">Auto-convert</b> mistyped object columns to numeric '
                'when ≥ 90% of non-null values are valid numbers.'
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

    # ── Step 5: Outlier Treatment ─────────────────────────────────────────
    elif step == 5:
        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
        if not numeric_cols:
            st.markdown(
                '<div style="margin:4px 0 8px 12px; padding:10px 14px;'
                ' background:rgba(59,130,246,0.03);'
                ' border-left:2px solid rgba(59,130,246,0.4);'
                ' border-radius:0 8px 8px 0;'
                ' font-size:0.8rem; color:rgba(255,255,255,0.4);">'
                '✓ No numeric columns available for outlier treatment.</div>',
                unsafe_allow_html=True,
            )
        else:
            # Raw statistical outliers for display + safe-zone-aware action
            from modules.core.audit_engine import _get_safe_zones
            safe_zones = _get_safe_zones()

            rows = []
            for col in numeric_cols:
                series = df[col].dropna()
                if len(series) < 3:
                    continue
                rec = evaluate_outlier_method(series)
                skew_val = rec["skewness"]
                method_name = rec["method"]
                detect_key = _METHOD_INFO.get(
                    method_name, ("iqr", "iqr_capping", "IQR Cap")
                )[0]
                # Method-aware threshold — consistent with data_audit
                threshold = default_outlier_threshold(detect_key)
                detect_fn = _OUTLIER_METHODS.get(detect_key, _OUTLIER_METHODS["iqr"])
                stat_mask, _ = detect_fn(series.values, threshold)
                raw_count = int(stat_mask.sum())

                # Action label from the shared preview helper
                real_row = compute_fn(df, col, safe_zones)
                action = real_row["Action"] if real_row else "No Outlier Treatment"
                col_min = round(float(series.min()), 2) if len(series) > 0 else 0.0
                col_max = round(float(series.max()), 2) if len(series) > 0 else 0.0
                total_rows = len(series)
                pct_outlier = round(raw_count / total_rows * 100, 2) if total_rows > 0 else 0.0

                rows.append({
                    "Column": col,
                    "Min": col_min,
                    "Max": col_max,
                    "Skewness": skew_val,
                    "Auto Method": method_name,
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
                    st.dataframe(
                        pd.DataFrame(rows),
                        use_container_width=True,
                        hide_index=True,
                        column_config={
                            "Min": st.column_config.NumberColumn(format="%.2f"),
                            "Max": st.column_config.NumberColumn(format="%.2f"),
                            "Skewness": st.column_config.NumberColumn(format="%.3f"),
                            "Outliers Detected": st.column_config.NumberColumn(format="%d"),
                            "% Outlier": st.column_config.NumberColumn(format="%.2f%%"),
                        },
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
        candidates = preprocessing_engine.PreprocessingEngine.get_log_transform_candidates(df)

        if candidates:
            n_log1p = sum(1 for c in candidates if c["Method"] == "log1p")
            n_yj = sum(1 for c in candidates if c["Method"] == "yeo-johnson")

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
                '• <b style="color:#F59E0B;">Yeo-Johnson</b>: power transform — used when min &lt; 0 (handles negatives)'
            )
        else:
            _skip_card("No highly-skewed columns detected — this step will be skipped.")

    # ── Step 7: Binning & Mapping ──────────────────────────────────────
    elif step == 7:
        from modules.utils.db_config_manager import get_rule
        binning_config = get_rule("binning_config") or {}
        preview = preprocessing_engine.PreprocessingEngine.get_binning_preview(df, binning_config)

        if preview:
            n_bin = sum(1 for p in preview if p["Type"] == "Numeric Binning")
            n_map = sum(1 for p in preview if p["Type"] == "Category Mapping")

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
        candidates = preprocessing_engine.PreprocessingEngine.get_encoding_preview(
            df, binning_config=binning_cfg,
        )

        if candidates:
            n_label = sum(1 for c in candidates if c["Encoding"] == "Label Encoding")
            n_onehot = sum(1 for c in candidates if c["Encoding"] == "One-Hot (drop_first)")
            n_drop = sum(1 for c in candidates if c["Encoding"] == "Drop (Redundant)")

            _metric_row([
                ("Total Columns", str(len(candidates)), col_hex),
                ("Label Encoding", str(n_label), col_hex),
                ("One-Hot Encoding", str(n_onehot), col_hex),
                ("Redundant Drop", str(n_drop), col_hex),
            ])

            st.dataframe(
                pd.DataFrame(candidates),
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Unique": st.column_config.NumberColumn(format="%d"),
                },
            )

            if n_drop > 0:
                drop_names = ", ".join(
                    f"<b style='color:#F59E0B;'>{c['Column']}</b>"
                    for c in candidates if c["Encoding"] == "Drop (Redundant)"
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


# ==============================================================================
# MAIN LAYOUT
# ==============================================================================

def main():
    """Entry point for the Preprocessing page."""
    lang = st.session_state.get("lang", "en")

    # ── Page header ───────────────────────────────────────────────────────
    page_header(
        title=get_text("preprocessing_title", lang),
        subtitle=get_text("overview_journey_preprocess_desc", lang),
    )

    # ── Workspace guard ───────────────────────────────────────────────────
    _ensure_workspace_active()
    active_file = st.session_state.get("active_file")
    workspace_status(active_file)

    # ── Load data ─────────────────────────────────────────────────────────
    df_raw = data_engine.load_and_standardize(
        active_file, _file_mtime=data_engine._get_file_mtime(active_file)
    )

    # ── File-change guard ─────────────────────────────────────────────────
    if st.session_state.get("_preprocessing_file") != active_file:
        st.session_state["_preprocessing_file"]  = active_file
        st.session_state["cleaned_data"]         = None
        st.session_state["preprocessing_done"]   = False
        st.session_state["preprocessing_result"] = {}

    if st.session_state.get("cleaned_data") is None:
        st.session_state["cleaned_data"] = df_raw.copy()
    active_file_scan_progress_bar("_preprocessing_done")
    df_work = st.session_state["cleaned_data"]
    engine  = preprocessing_engine.PreprocessingEngine

    # ── Post-completion banner (full width) ────────────────────────────────
    done   = st.session_state.get("preprocessing_done", False)
    result = st.session_state.get("preprocessing_result", {})

    if done:
        cleaned_file  = result.get("cleaned_filename", "cleaned.csv")
        encoded_file  = result.get("encoded_filename", "encoded.csv")
        rows_bef      = result.get("rows_before", 0)
        rows_aft      = result.get("rows_after", 0)
        dupes         = result.get("dupes_dropped", 0)
        df_cleaned_csv = result.get("df_cleaned_csv", b"")
        df_encoded_csv = result.get("df_encoded_csv", b"")

        pipeline_done_banner(cleaned_file, rows_bef, rows_aft, dupes, stats=result)

        # ── Target Correlation: Income ────────────────────────────────────
        import plotly.graph_objects as go  # noqa: E402 (lazy import)

        corr_after = result.get("corr_after")

        if corr_after is not None:
            # Find Income column (case-insensitive)
            income_col = None
            for col_name in corr_after.columns:
                if col_name.lower() == "income":
                    income_col = col_name
                    break

            if income_col is not None:
                # Extract correlations with Income, excluding self
                target_corr = corr_after[income_col].drop(income_col, errors="ignore")
                target_corr = target_corr.dropna()

                # Sort by absolute value descending
                target_corr = target_corr.reindex(
                    target_corr.abs().sort_values(ascending=True).index
                )

                # Color: teal for positive, pink for negative
                colors = [
                    "#2DD4BF" if val >= 0 else "#F472B6"
                    for val in target_corr.values
                ]

                n_features = len(target_corr)
                chart_height = max(360, n_features * 26 + 80)

                fig = go.Figure(
                    data=go.Bar(
                        x=target_corr.values,
                        y=target_corr.index.tolist(),
                        orientation="h",
                        marker=dict(
                            color=colors,
                            line=dict(width=0),
                            opacity=0.85,
                        ),
                        text=[f"{v:+.3f}" for v in target_corr.values],
                        textposition="outside",
                        textfont=dict(size=10, color="rgba(255,255,255,0.6)"),
                        hovertemplate=(
                            "<b>%{y}</b><br>"
                            "r = %{x:+.4f}<extra></extra>"
                        ),
                    )
                )

                fig.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="rgba(255,255,255,0.6)", family="Inter"),
                    margin=dict(l=10, r=60, t=10, b=10),
                    height=chart_height,
                    xaxis=dict(
                        range=[-1.05, 1.05],
                        zeroline=True,
                        zerolinecolor="rgba(255,255,255,0.12)",
                        zerolinewidth=1,
                        showgrid=True,
                        gridcolor="rgba(255,255,255,0.04)",
                        tickfont=dict(size=9, color="rgba(255,255,255,0.4)"),
                        dtick=0.2,
                    ),
                    yaxis=dict(
                        tickfont=dict(size=10, color="rgba(255,255,255,0.6)"),
                        showgrid=False,
                    ),
                    bargap=0.25,
                )

                # Section title
                st.markdown(
                    '<div style="font-size:0.75rem;font-weight:700;'
                    'color:rgba(255,255,255,0.5);text-transform:uppercase;'
                    'letter-spacing:0.8px;margin:8px 0 4px 0;">'
                    'Target Correlation — Income</div>'
                    '<div style="font-size:0.72rem;color:rgba(255,255,255,0.3);'
                    'margin-bottom:12px;">Pearson correlation of each feature '
                    'with the target variable, sorted by |r|</div>',
                    unsafe_allow_html=True,
                )
                st.plotly_chart(fig, use_container_width=True, key="target_corr")

        # ── Download buttons ──────────────────────────────────────────────
        import io, zipfile  # noqa: E402

        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr(cleaned_file, df_cleaned_csv)
            zf.writestr(encoded_file, df_encoded_csv)
        zip_bytes = zip_buffer.getvalue()

        base_stem = cleaned_file.replace("_cleaned.csv", "")
        zip_name  = f"{base_stem}_processed.zip"

        c1, c2, c3 = st.columns([2, 2, 1])
        with c1:
            st.download_button(
                label=":material/download: Download Processed CSV Data",
                data=zip_bytes,
                file_name=zip_name,
                mime="application/zip",
                use_container_width=True,
                type="primary",
                key="dl_processed_zip",
            )
        with c2:
            if st.button(":material/refresh: Re-run Audit", use_container_width=True, key="btn_rerun_audit"):
                st.session_state["_force_rerun_audit"] = True
                st.switch_page("pages/data_audit.py")
        with c3:
            if st.button(":material/arrow_forward: Next to EDA", use_container_width=True, key="btn_next_eda"):
                st.switch_page("pages/eda.py")

        section_divider()

    # ── Spacing (sync with other pages) ─────────────────────────────────
    st.markdown('<div style="height:20px;"></div>', unsafe_allow_html=True)

    # ── 2-COLUMN LAYOUT with vertical divider ──────────────────────────────
    col_left, col_div, col_right = st.columns([25, 1, 74], gap="small")

    with col_left:
        # Pipeline title — prominent
        st.markdown(
            '<div style="margin-bottom:18px;">'
            '<div style="display:flex; align-items:center; gap:10px; margin-bottom:6px;">'
            '<div style="width:28px; height:28px; border-radius:8px;'
            ' background:rgba(59,130,246,0.12); border:1px solid rgba(59,130,246,0.2);'
            ' display:flex; align-items:center; justify-content:center;">'
            '<span style="color:var(--accent-blue); font-size:0.8rem;">⚡</span></div>'
            '<span style="font-size:1.1rem; font-weight:800; color:rgba(255,255,255,0.95);'
            ' letter-spacing:-0.3px;">'
            'Pipeline</span>'
            '<span style="background:rgba(255,255,255,0.06); border-radius:12px;'
            ' font-size:0.55rem; padding:2px 8px; font-weight:700;'
            ' color:rgba(255,255,255,0.4); letter-spacing:0.8px;">8 STEPS</span>'
            '</div>'
            '<div style="font-size:0.72rem; color:rgba(255,255,255,0.3);'
            ' padding-left:38px;">'
            'Select a step to preview its details</div>'
            '</div>',
            unsafe_allow_html=True,
        )

        active_step = _render_pipeline_sidebar(
            st.session_state.get("_pp_active_step", 1)
        )

    with col_div:
        st.markdown(
            '<div style="width:1px; min-height:400px;'
            ' background:linear-gradient(180deg,'
            ' rgba(255,255,255,0), rgba(255,255,255,0.08),'
            ' rgba(255,255,255,0.08), rgba(255,255,255,0));'
            ' margin:0 auto;"></div>',
            unsafe_allow_html=True,
        )

    with col_right:
        _render_detail_panel(
            active_step,
            df_work,
            _compute_outlier_preview_row,
        )

    # ── Run button (full-width, below both columns) ───────────────────────
    if not done:
        section_divider()
        btn_placeholder = st.empty()
        with btn_placeholder.container():
            _, col_ctr, _ = st.columns([1, 2, 1])
            with col_ctr:
                st.markdown(
                    '<div style="text-align:center; margin-bottom:12px;">'
                    '<div style="font-size:0.95rem; font-weight:700;'
                    ' color:rgba(255,255,255,0.7); margin-bottom:4px;">'
                    'Ready to clean your dataset?</div>'
                    '<div style="font-size:0.72rem; color:rgba(255,255,255,0.3);">'
                    'All 8 steps will run sequentially on the active workspace file.</div>'
                    '</div>',
                    unsafe_allow_html=True,
                )
                clicked = st.button(
                    "\u26a1 Run Preprocessing Pipeline",
                    type="primary",
                    use_container_width=True,
                    key="btn_run_pipeline",
                )
        if clicked:
            btn_placeholder.empty()
            rows_original = len(df_work)
            with st.spinner("Running automated preprocessing pipeline..."):
                _run_pipeline(df_work, engine, active_file, rows_original, lang)
            st.rerun()


if __name__ == "__main__":
    main()