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

import io
import time
import zipfile

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from modules.core import data_engine, preprocessing_engine
from modules.core.file_manager import save_with_auto_increment
from modules.core.preprocessing_engine import (
    PreprocessingEngine,
    ENC_LABEL, ENC_ONEHOT, ENC_DROP_REDUNDANT,
    SCALER_STANDARD, SCALER_ROBUST,
    compute_after_metrics,
)
from modules.core.audit_engine import (
    _get_safe_zones,
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
) -> pd.DataFrame:
    """Execute the full 9-step fixed preprocessing pipeline with live progress.

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

    On completion, three output files are saved via ``save_with_auto_increment``:
      - ``_cleaned.csv``        — after Step 5 (raw cleaned data)
      - ``_encoded.csv``        — after Step 9 (full pipeline output)
      - ``_numeric_trans.csv``  — cleaned + domain-knowledge ordinal encoding

    Args:
        df:            Working DataFrame (copy of raw data).
        engine:        ``PreprocessingEngine`` class reference.
        active_file:   Basename of the currently active workspace file.
        rows_original: Row count of the raw DataFrame *before* any step.

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
    df = engine.impute_missing(df)
    time.sleep(0.1)

    # ── Step 5: Per-column outlier treatment ────────────────────────────────
    progress.progress(0.50, text=STEPS[4][0])
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    total_outliers_before = 0
    for col in numeric_cols:
        row = PreprocessingEngine.compute_outlier_preview_row(df, col, safe_zones)
        if row is None or row["Outliers Detected"] == 0:
            continue
        total_outliers_before += row["Outliers Detected"]
        col_threshold = default_outlier_threshold(row["detect_key"])
        df = engine.handle_outliers(df, row["treatment_method"], [col], col_threshold)
    time.sleep(0.1)

    # ── Save cleaned file (after Step 5) ──────────────────────────────────
    progress.progress(0.56, text=":material/save: Saving cleaned dataset...")
    df_cleaned_snapshot = df.copy()
    basename  = active_file.replace("\\", "/").split("/")[-1]
    base_stem = basename.replace(".csv", "")
    cleaned_filename, df_cleaned_csv = save_with_auto_increment(
        df_cleaned_snapshot, base_stem, "_cleaned",
    )

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
    binning_cfg = _get_rule_pipeline("binning_config") or {}
    candidates = engine.get_encoding_preview(df, binning_config=binning_cfg)
    n_label     = sum(1 for c in candidates if c["Encoding"] == ENC_LABEL)
    n_onehot    = sum(1 for c in candidates if c["Encoding"] == ENC_ONEHOT)
    n_redundant = sum(1 for c in candidates if c["Encoding"] == ENC_DROP_REDUNDANT)
    df = engine.apply_feature_encoding(df, candidates, binning_config=binning_cfg)
    time.sleep(0.1)

    # ── Step 9: Feature Scaling ───────────────────────────────────────
    progress.progress(0.88, text=STEPS[8][0])
    scaling_candidates = engine.get_scaling_preview(df)
    n_standard = sum(1 for c in scaling_candidates if c["Method"] == SCALER_STANDARD)
    n_robust   = sum(1 for c in scaling_candidates if c["Method"] == SCALER_ROBUST)
    df = engine.apply_feature_scaling(df, scaling_candidates)
    time.sleep(0.1)

    # ── Save encoded file (after Step 9) ──────────────────────────────────
    progress.progress(0.95, text=":material/save: Saving encoded dataset...")
    encoded_filename, df_encoded_csv = save_with_auto_increment(
        df, base_stem, "_encoded",
    )

    # ── Save numeric-transformed file (cleaned + domain-knowledge encoding) ──
    progress.progress(0.97, text=":material/save: Saving numeric-transformed dataset...")
    df_numeric_trans = data_engine.encode_for_correlation(df_cleaned_snapshot)
    numeric_trans_filename, df_numeric_trans_csv = save_with_auto_increment(
        df_numeric_trans, base_stem, "_numeric_trans",
    )

    # ── Compute correlation matrix BEFORE dropping fnlwgt (for heatmap) ────
    corr_matrix = df_numeric_trans.select_dtypes(include=["number"]).corr()

    # ── Drop fnlwgt from all DataFrames & re-save CSVs ────────────────────
    _DROP_COL = "fnlwgt"
    for _df in (df_cleaned_snapshot, df, df_numeric_trans):
        if _DROP_COL in _df.columns:
            _df.drop(columns=[_DROP_COL], inplace=True)

    # Re-generate CSV bytes without fnlwgt
    cleaned_filename, df_cleaned_csv = save_with_auto_increment(
        df_cleaned_snapshot, base_stem, "_cleaned",
    )
    encoded_filename, df_encoded_csv = save_with_auto_increment(
        df, base_stem, "_encoded",
    )
    numeric_trans_filename, df_numeric_trans_csv = save_with_auto_increment(
        df_numeric_trans, base_stem, "_numeric_trans",
    )

    progress.progress(1.0, text=":material/check_circle: Preprocessing complete!")
    time.sleep(0.4)
    progress.empty()

    # ── Compute "after" quality metrics (delegated to core) ───────────────
    comparison = compute_after_metrics(
        df_cleaned_snapshot,
        initial_missing,
        noise_cleaned,
        initial_dupes,
        total_outliers_before,
    )

    # ── Update session state ──────────────────────────────────────────────
    st.session_state["active_file"]          = cleaned_filename
    st.session_state["_preprocessing_file"]  = cleaned_filename
    st.session_state["cleaned_data"]         = None
    st.session_state["preprocessing_done"]   = True
    st.session_state["preprocessing_result"] = {
        "cleaned_filename":      cleaned_filename,
        "encoded_filename":      encoded_filename,
        "numeric_trans_filename": numeric_trans_filename,
        "rows_before":           rows_original,
        "rows_after":            len(df),
        "dupes_dropped":         dupes_dropped,
        "df_cleaned_csv":        df_cleaned_csv,
        "df_encoded_csv":        df_encoded_csv,
        "df_numeric_trans_csv":  df_numeric_trans_csv,
        "comparison":            comparison,
        # Encoding summary
        "n_label_encoded":     n_label,
        "n_onehot_encoded":    n_onehot,
        "n_redundant_dropped": n_redundant,
        # Scaling summary
        "n_standard_scaled":   n_standard,
        "n_robust_scaled":     n_robust,
        "scaling_candidates":  scaling_candidates,
        # Correlation matrix (includes fnlwgt — for heatmap only)
        "corr_matrix":  corr_matrix,
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
        cleaned_file       = result.get("cleaned_filename", "cleaned.csv")
        encoded_file       = result.get("encoded_filename", "encoded.csv")
        numeric_trans_file = result.get("numeric_trans_filename", "numeric_trans.csv")
        rows_bef           = result.get("rows_before", 0)
        rows_aft           = result.get("rows_after", 0)
        dupes              = result.get("dupes_dropped", 0)
        df_cleaned_csv     = result.get("df_cleaned_csv", b"")
        df_encoded_csv     = result.get("df_encoded_csv", b"")
        df_numeric_trans_csv = result.get("df_numeric_trans_csv", b"")

        pipeline_done_banner(cleaned_file, rows_bef, rows_aft, dupes, stats=result)

        # ── Feature Correlation Heatmap ───────────────────────────────────
        corr_matrix = result.get("corr_matrix")
        if corr_matrix is not None and not corr_matrix.empty:
            from modules.ui.visualizer import CHART_LAYOUT, MUTED_COLOR, apply_global_theme

            st.markdown(
                '<div style="margin:20px 0 4px 0;padding:10px 16px;'
                'background:rgba(255,159,67,0.06);'
                'border-left:3px solid rgba(255,159,67,0.5);'
                'border-radius:0 8px 8px 0;">'
                '<div style="font-size:0.92rem;font-weight:700;'
                'color:rgba(255,255,255,0.88);">'
                'Feature Correlation Heatmap</div>'
                '<div style="font-size:0.76rem;color:rgba(255,255,255,0.40);'
                'margin-top:4px;line-height:1.6;">'
                'Pairwise Pearson correlation across all numeric features '
                'after domain-knowledge encoding</div>'
                '</div>',
                unsafe_allow_html=True,
            )

            cols = corr_matrix.columns.tolist()
            z_vals = np.round(corr_matrix.values, 2)

            # Annotation text — show value only if |r| >= 0.05
            annotations = []
            for row_idx, row_label in enumerate(cols):
                for col_idx, col_label in enumerate(cols):
                    val = z_vals[row_idx][col_idx]
                    text = f"{val:.2f}" if abs(val) >= 0.05 else ""
                    annotations.append(dict(
                        x=col_label, y=row_label,
                        text=text,
                        font=dict(
                            size=8,
                            color="rgba(255,255,255,0.85)" if abs(val) >= 0.3
                            else "rgba(255,255,255,0.45)",
                        ),
                        showarrow=False,
                    ))

            fig = go.Figure(data=go.Heatmap(
                z=z_vals,
                x=cols, y=cols,
                colorscale=[
                    [0.0,  "#0EA5E9"],
                    [0.2,  "rgba(14,165,233,0.30)"],
                    [0.5,  "rgba(30,35,60,0.90)"],
                    [0.8,  "rgba(255,159,67,0.55)"],
                    [1.0,  "#F59E0B"],
                ],
                zmin=-1, zmax=1,
                colorbar=dict(
                    title=dict(text="r", font=dict(color=MUTED_COLOR, size=10)),
                    tickfont=dict(color=MUTED_COLOR, size=9),
                    thickness=12, len=0.6,
                ),
                hovertemplate="<b>%{x}</b> vs <b>%{y}</b><br>r = %{z:.3f}<extra></extra>",
            ))

            _strip_keys = {"legend", "margin"}
            base_layout = {k: v for k, v in CHART_LAYOUT.items() if k not in _strip_keys}
            fig.update_layout(
                **base_layout,
                height=max(600, len(cols) * 45),
                margin=dict(l=130, r=50, t=20, b=120),
                xaxis=dict(
                    tickfont=dict(color=MUTED_COLOR, size=9),
                    tickangle=-45,
                ),
                yaxis=dict(
                    tickfont=dict(color=MUTED_COLOR, size=9),
                    autorange="reversed",
                ),
                annotations=annotations,
            )
            st.plotly_chart(
                apply_global_theme(fig),
                use_container_width=True,
                key="heatmap_corr_after",
            )

        # ── Download buttons ──────────────────────────────────────────────
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr(cleaned_file, df_cleaned_csv)
            zf.writestr(encoded_file, df_encoded_csv)
            zf.writestr(numeric_trans_file, df_numeric_trans_csv)
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
                _run_pipeline(df_work, engine, active_file, rows_original)
            st.rerun()


if __name__ == "__main__":
    main()