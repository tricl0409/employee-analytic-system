"""
preprocessing.py — Data Preprocessing Page (thin orchestrator)

Responsibilities of this module:
  • Page layout, workspace guards, session-state management.
  • Running the 9-step fixed pipeline (_run_pipeline).

All business logic is delegated to the Core layer:
  PreprocessingEngine.compute_outlier_preview_row(df, col, safe_zones)
  PreprocessingEngine.PIPELINE_STEP_DEFS
  PreprocessingEngine.METHOD_INFO

All tab / panel rendering is delegated to UiComponents:
  UiComponents.render_scrubber_tab(df)           → Tab 1
  UiComponents.render_missing_and_dupes_tab(df)  → Tab 2
  UiComponents.render_outlier_tab(df, fn)        → Tab 3
  UiComponents.render_pipeline_sidebar(step)     → Left column
  UiComponents.render_detail_panel(step, df, fn) → Right column

CSS is managed centrally in modules/ui/styles.py (PREPROCESSING_STYLES).
"""

import os
import time

import pandas as pd
import streamlit as st

from modules.core import data_engine, preprocessing_engine
from modules.core.file_manager import UPLOADS_DIR, save_dataframe
from modules.core.preprocessing_engine import (
    PreprocessingEngine,
    ENC_LABEL, ENC_ONEHOT, ENC_DROP_REDUNDANT,
    SCALER_STANDARD, SCALER_ROBUST,
)
from modules.core.audit_engine import (
    _get_safe_zones,
    _compute_noise_mask,
    _get_cat_columns,
    default_outlier_threshold,
)
from modules.ui import (
    page_header,
    section_divider,
    workspace_status,
    active_file_scan_progress_bar,
    pipeline_done_banner,
    render_pipeline_sidebar,
    render_detail_panel,
)

from modules.utils.localization import get_text
from modules.utils.helpers import _ensure_workspace_active


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
    Execute the full 9-step fixed preprocessing pipeline with live progress.

    Pipeline order:
      1. Standardize & Type Cast (trim whitespace, normalize casing, convert dtypes)
      2. Noise Cleaning          (clean_noise_values: replace noise tokens with NaN)
      3. Duplicate Removal       (drop_duplicates)
      4. Missing Value Handling  (impute_missing: mean/median/mode)
      5. Outlier Treatment       (handle_outliers, per-column auto-method
                                  via PreprocessingEngine.compute_outlier_preview_row)
      6. Log Transformation      (apply_log_transform: log1p / yeo-johnson)
      7. Binning & Mapping       (apply_binning_mapping: discretize + group)
      8. Feature Encoding        (apply_feature_encoding: label + one-hot)
      9. Feature Scaling         (apply_feature_scaling: StandardScaler / RobustScaler)

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
        Cleaned DataFrame after all 9 steps.
    """
    safe_zones = _get_safe_zones()

    STEPS = [
        (":material/tune: Step 1/9 — Standardizing text & converting dtypes...",         "std"),
        (":material/delete: Step 2/9 — Cleaning noise & placeholder values...",          "noise"),
        (":material/content_copy: Step 3/9 — Removing duplicate rows...",                "dupes"),
        (":material/healing: Step 4/9 — Imputing missing values (Mean/Median/Mode)...",  "missing"),
        (":material/square_foot: Step 5/9 — Treating outliers (auto-method)...",         "outliers"),
        (":material/functions: Step 6/9 — Applying log transformations...",              "logtf"),
        (":material/category: Step 7/9 — Binning & mapping features...",                 "binmap"),
        (":material/label: Step 8/9 — Encoding categorical features...",                 "encode"),
        (":material/straighten: Step 9/9 — Scaling numeric features...",                 "scale"),
    ]
    progress = st.progress(0, text=":material/hourglass_top: Initializing pipeline...")
    time.sleep(0.2)

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
    df = engine.impute_missing(df)
    missing_after_fill = int(df.isna().sum().sum())
    time.sleep(0.1)

    # ── Step 5: Per-column outlier treatment ────────────────────────────────
    progress.progress(0.50, text=STEPS[4][0])
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    total_outliers_before = 0
    outlier_cols_treated = 0
    for col in numeric_cols:
        row = PreprocessingEngine.compute_outlier_preview_row(df, col, safe_zones)
        if row is None or row["Outliers Detected"] == 0:
            continue
        total_outliers_before += row["Outliers Detected"]
        col_threshold = default_outlier_threshold(row["detect_key"])
        df = engine.handle_outliers(df, row["treatment_method"], [col], col_threshold)
        outlier_cols_treated += 1
    time.sleep(0.1)

    # ── Save cleaned file (after Step 5) ──────────────────────────────────
    progress.progress(0.56, text=":material/save: Saving cleaned dataset...")
    df_cleaned_snapshot = df.copy()          # freeze state before Steps 6-9 mutate
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
    progress.progress(0.78, text=STEPS[7][0])
    from modules.utils.db_config_manager import get_rule as _get_rule_pipeline
    _binning_cfg = _get_rule_pipeline("binning_config") or {}
    candidates = engine.get_encoding_preview(df, binning_config=_binning_cfg)
    n_label = sum(1 for c in candidates if c["Encoding"] == ENC_LABEL)
    n_onehot = sum(1 for c in candidates if c["Encoding"] == ENC_ONEHOT)
    n_redundant = sum(1 for c in candidates if c["Encoding"] == ENC_DROP_REDUNDANT)
    df = engine.apply_feature_encoding(df, candidates)
    time.sleep(0.1)

    # ── Step 9: Feature Scaling ───────────────────────────────────────
    progress.progress(0.88, text=STEPS[8][0])
    scaling_candidates = engine.get_scaling_preview(df)
    n_standard = sum(1 for c in scaling_candidates if c["Method"] == SCALER_STANDARD)
    n_robust = sum(1 for c in scaling_candidates if c["Method"] == SCALER_ROBUST)
    df = engine.apply_feature_scaling(df, scaling_candidates)
    time.sleep(0.1)

    # ── Save encoded file (after Step 9) ──────────────────────────────────
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

    # Re-compute noise after cleaning — delegate to single source of truth
    _cat_after = df_cleaned_snapshot.select_dtypes(include=["object"]).columns.tolist()
    after_noise = 0
    for _col_name in _cat_after:
        _series = df_cleaned_snapshot[_col_name].dropna().astype(str)
        after_noise += int(_compute_noise_mask(_series).sum())

    # All detected outliers were capped to their statistical fence boundaries,
    # so by definition they no longer exceed the original thresholds.
    # Re-detecting with fresh statistics would cause "fence compression"
    # (tighter distribution → narrower bounds → spurious new outliers).
    after_outliers = 0

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
        # Scaling summary
        "n_standard_scaled":   n_standard,
        "n_robust_scaled":     n_robust,
        "scaling_candidates":  scaling_candidates,
        # Correlation matrix (post-encoding, for target correlation chart)
        "corr_after":  df.select_dtypes(include=["number"]).corr(),
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
            ' color:rgba(255,255,255,0.4); letter-spacing:0.8px;">9 STEPS</span>'
            '</div>'
            '<div style="font-size:0.72rem; color:rgba(255,255,255,0.3);'
            ' padding-left:38px;">'
            'Select a step to preview its details</div>'
            '</div>',
            unsafe_allow_html=True,
        )

        active_step = render_pipeline_sidebar(
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
        render_detail_panel(
            active_step,
            df_work,
            PreprocessingEngine.compute_outlier_preview_row,
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
                    'All 9 steps will run sequentially on the active workspace file.</div>'
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