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
    evaluate_outlier_method,
)
from modules.ui import (
    page_header,
    section_divider,
    workspace_status,
    active_file_scan_progress_bar,
    pipeline_card,
    pipeline_done_banner,
    detail_analysis_header,
    render_scrubber_tab,
    render_missing_and_dupes_tab,
    render_outlier_tab,
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
    threshold: float,
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
        threshold:  Detection threshold from admin config.

    Returns:
        Dict with keys: Column, Skewness, Auto Method, detect_key,
        treatment_method, Outliers Detected, Action, has_safe_zone.
        Returns ``None`` if the column has fewer than 3 non-null values.
    """
    series = df[col].dropna()
    if len(series) < 3:
        return None

    skew_val    = float(series.skew())
    rec         = evaluate_outlier_method(series)
    method_name = rec["method"]  # "Z-Score" | "IQR" | "Modified Z-Score"

    detect_key, treatment_method, base_action = _METHOD_INFO.get(
        method_name, ("iqr", "iqr_capping", "IQR Cap")
    )

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

    final_mask    = stat_mask & safe_mask
    outlier_count = int(final_mask.sum())

    if outlier_count == 0:
        action = "Skip Treatment"
    elif has_safe_zone:
        action = "Clip to Safe Zone Bounds"
    else:
        action = base_action

    return {
        "Column":            col,
        "Skewness":          round(skew_val, 3),
        "Auto Method":       method_name,
        "detect_key":        detect_key,
        "treatment_method":  treatment_method,
        "Outliers Detected": outlier_count,
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
    Execute the full 5-step fixed preprocessing pipeline with live progress.

    Steps:
      1. Garbage values → NaN   (clean_noise_values)
      2. Scrub text              (fix_text_formatting: trim whitespace + normalize casing)
      3. Fill missing values     (handle_missing_smart: mean/median/mode)
      4. Drop duplicates         (drop_duplicates)
      5. Treat outliers          (handle_outliers, per-column auto-method
                                  via _compute_outlier_preview_row)

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
        Cleaned DataFrame after all 5 steps.
    """
    from modules.core.audit_engine import _get_cat_columns

    threshold   = 1.5
    safe_zones  = _get_safe_zones()

    STEPS = [
        (":material/delete: Step 1/5 — Replacing garbage values with NaN...",          "noise"),
        (":material/content_cut: Step 2/5 — Scrubbing text (trim + normalize casing)...", "ws"),
        (":material/healing: Step 3/5 — Filling missing values (Mean/Median/Mode)...", "missing"),
        (":material/content_copy: Step 4/5 — Dropping duplicate rows...",              "dupes"),
        (":material/square_foot: Step 5/5 — Treating outliers (auto-method)...",       "outliers"),
    ]

    progress = st.progress(0, text=STEPS[0][0])

    # ── Step 1: Garbage → NaN ─────────────────────────────────────────────
    progress.progress(0.05, text=STEPS[0][0])
    cat_cols = _get_cat_columns(df).tolist()
    df = engine.clean_noise_values(df, strategy="replace_nan", columns=cat_cols)
    time.sleep(0.1)

    # ── Step 2: Trim whitespace AND normalize casing in one pass ──────────
    progress.progress(0.25, text=STEPS[1][0])
    df = engine.fix_text_formatting(df, fix_whitespace=True, fix_casing=True)
    time.sleep(0.1)

    # ── Step 3: Fill missing values ───────────────────────────────────────
    progress.progress(0.50, text=STEPS[2][0])
    df = engine.handle_missing_smart(df)
    time.sleep(0.1)

    # ── Step 4: Drop duplicates ─────────────────────────────────────────
    progress.progress(0.65, text=STEPS[3][0])
    n_before_dedup = len(df)
    df = engine.drop_duplicates(df)
    dupes_dropped  = n_before_dedup - len(df)
    time.sleep(0.1)

    # ── Step 5: Per-column outlier treatment ────────────────────────────────
    # Uses _compute_outlier_preview_row (same logic shown in Tab 3 preview).
    progress.progress(0.80, text=STEPS[4][0])
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    for col in numeric_cols:
        row = _compute_outlier_preview_row(df, col, safe_zones, threshold)
        if row is None or row["Outliers Detected"] == 0:
            continue
        df = engine.handle_outliers(df, row["treatment_method"], [col], threshold)
    time.sleep(0.1)

    # ── Save cleaned file ─────────────────────────────────────────────────
    progress.progress(0.95, text=":material/save: Saving cleaned dataset...")
    basename  = active_file.replace("\\", "/").split("/")[-1]
    base_stem = basename.replace(".csv", "") + "_cleaned"
    new_filename = f"{base_stem}.csv"
    counter = 1
    while os.path.exists(os.path.join(UPLOADS_DIR, new_filename)):
        new_filename = f"{base_stem}_{counter}.csv"
        counter += 1
    save_dataframe(df, new_filename)

    progress.progress(1.0, text=":material/check_circle: Preprocessing complete!")
    time.sleep(0.4)
    progress.empty()

    # ── Update session state ──────────────────────────────────────────────
    st.session_state["active_file"]          = new_filename
    st.session_state["_preprocessing_file"]  = new_filename   # keep guard in sync
    st.session_state["cleaned_data"]         = None   # force reload on next visit
    st.session_state["preprocessing_done"]   = True
    st.session_state["preprocessing_result"] = {
        "filename":      new_filename,
        "rows_before":   rows_original,   # true raw count before Step 1
        "rows_after":    len(df),
        "dupes_dropped": dupes_dropped,
        "df_csv":        df.to_csv(index=False).encode("utf-8"),
    }

    return df


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

    # ── File-change guard: reset preprocessing state when workspace changes ──
    if st.session_state.get("_preprocessing_file") != active_file:
        st.session_state["_preprocessing_file"]    = active_file
        st.session_state["cleaned_data"]           = None
        st.session_state["preprocessing_done"]     = False
        st.session_state["preprocessing_result"]   = {}

    if st.session_state.get("cleaned_data") is None:
        st.session_state["cleaned_data"] = df_raw.copy()
    active_file_scan_progress_bar("_preprocessing_done")
    df_work = st.session_state["cleaned_data"]
    engine  = preprocessing_engine.PreprocessingEngine


    # =========================================================================
    # PIPELINE CONTROL — pre-run card OR post-completion banner
    # =========================================================================
    done   = st.session_state.get("preprocessing_done", False)
    result = st.session_state.get("preprocessing_result", {})

    if not done:
        pipeline_card()

        _, col_run, _ = st.columns([1, 2, 1])
        with col_run:
            if st.button(
                f"\u26a1  Run Preprocessing + Save",
                type="primary",
                use_container_width=True,
                key="btn_run_pipeline",
            ):
                rows_original = len(df_work)
                with st.spinner("Running automated preprocessing pipeline..."):
                    _run_pipeline(df_work, engine, active_file, rows_original, lang)
                st.rerun()

        section_divider()
    else:
        res_file = result.get("filename", "cleaned_file.csv")
        rows_bef = result.get("rows_before", 0)
        rows_aft = result.get("rows_after", 0)
        dupes    = result.get("dupes_dropped", 0)
        df_csv   = result.get("df_csv", b"")

        pipeline_done_banner(res_file, rows_bef, rows_aft, dupes)

        c1, c2, c3 = st.columns([2, 2, 1])
        with c1:
            st.download_button(
                label=":material/download: Download Cleaned CSV",
                data=df_csv,
                file_name=res_file,
                mime="text/csv",
                use_container_width=True,
                type="primary",
                key="dl_cleaned_csv",
            )
        with c2:
            if st.button(":material/refresh: Re-run Audit", use_container_width=True, key="btn_rerun_audit"):
                st.session_state["_force_rerun_audit"] = True
                st.switch_page("pages/data_audit.py")
        with c3:
            if st.button(":material/arrow_forward: Next to EDA", use_container_width=True, key="btn_next_eda"):
                st.switch_page("pages/eda.py")

    # =========================================================================
    # DETAILED ANALYSIS TABS  (always visible)
    # =========================================================================
    detail_analysis_header()

    tabs = st.tabs([
        ":material/delete_sweep: Data Scrubber",
        ":material/healing: Missing & Duplicates",
        ":material/query_stats: Outlier Treatment",
    ])

    with tabs[0]:
        render_scrubber_tab(df_work)

    with tabs[1]:
        render_missing_and_dupes_tab(df_work)

    with tabs[2]:
        # Pass the local helper so the component doesn't import from this page
        render_outlier_tab(df_work, _compute_outlier_preview_row)


if __name__ == "__main__":
    main()
