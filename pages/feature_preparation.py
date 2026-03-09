"""
feature_preparation.py — Feature Preparation Page

Workflow:
  1. Show pipeline preview card (what transformations will run)
  2. User clicks "Process Transformation & Standardization"
  3. Progress bar runs through encoding → scaling steps
  4. Results summary displayed (columns encoded/scaled, dtypes changed)
  5. Button switches to Download CSV (no save to workspace, no file switch)

No workspace switching or file saving — purely in-memory transformation for
downstream modelling preparation.
"""

import time
import pandas as pd
import streamlit as st

from modules.core import data_engine, preprocessing_engine
from modules.ui import (
    page_header,
    section_divider,
    workspace_status,
    active_file_scan_progress_bar,
    detail_analysis_header,
)
from modules.utils.localization import get_text
from modules.utils.helpers import _ensure_workspace_active


# ==============================================================================
# CONSTANTS
# ==============================================================================

_AMBER        = "#FF9F43"
_AMBER_DIM    = "rgba(255,159,67,0.12)"
_AMBER_BORDER = "rgba(255,159,67,0.30)"
_TEXT_DIM     = "rgba(255,255,255,0.55)"
_TEXT_MAIN    = "rgba(255,255,255,0.80)"

_ENC_OPTIONS = {
    "One-Hot Encoding":  "one_hot",
    "Ordinal Encoding":  "ordinal",
}
_SCALE_OPTIONS = {
    "Standard (Z-score)": "standard",
    "Min-Max [0, 1]":     "minmax",
}


# ==============================================================================
# UI HELPERS
# ==============================================================================

def _card(title: str, icon: str, body_html: str) -> None:
    st.markdown(
        f"""
        <div style="
            background:{_AMBER_DIM};
            border:1px solid {_AMBER_BORDER};
            border-left:3px solid {_AMBER};
            border-radius:10px;
            padding:16px 18px;
            margin-bottom:12px;
        ">
          <div style="
              font-size:0.78rem;font-weight:700;
              color:{_AMBER};
              text-transform:uppercase;letter-spacing:1px;
              margin-bottom:10px;
          ">{icon}&nbsp; {title}</div>
          <div style="font-size:0.82rem;color:{_TEXT_DIM};line-height:1.7;">
              {body_html}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _stat(label: str, value: str) -> str:
    return (
        f"<span style='color:{_TEXT_DIM};'>{label}:&nbsp;</span>"
        f"<b style='color:{_AMBER};font-weight:700;'>{value}</b>&emsp;"
    )


def _feature_pipeline_card(df: pd.DataFrame, enc_cols: list, enc_method: str,
                            scale_cols: list, scale_method: str) -> None:
    """Preview card — shows what the pipeline will do."""
    cat_total = len(df.select_dtypes(include=["object","category"]).columns)
    num_total = len(df.select_dtypes(include="number").columns)

    enc_label   = next((k for k,v in _ENC_OPTIONS.items() if v == enc_method), enc_method)
    scale_label = next((k for k,v in _SCALE_OPTIONS.items() if v == scale_method), scale_method)

    enc_preview   = ", ".join(enc_cols)   if enc_cols   else "—  (none selected)"
    scale_preview = ", ".join(scale_cols) if scale_cols else "—  (none selected)"

    st.markdown(
        f"""
        <div style="
            background:{_AMBER_DIM};
            border:1px solid {_AMBER_BORDER};
            border-radius:12px;
            padding:20px 22px;
            margin-bottom:18px;
        ">
          <div style="
              font-size:0.82rem;font-weight:700;color:{_AMBER};
              text-transform:uppercase;letter-spacing:1.1px;
              margin-bottom:14px;
          ">⚙️&nbsp; Transformation Pipeline Preview</div>

          <div style="display:flex;gap:24px;flex-wrap:wrap;margin-bottom:14px;">
            {_stat("Dataset", f"{len(df):,} rows × {len(df.columns)} cols")}
            {_stat("Categorical cols", str(cat_total))}
            {_stat("Numeric cols", str(num_total))}
          </div>

          <div style="
              border-top:1px solid {_AMBER_BORDER};
              padding-top:12px;display:grid;
              grid-template-columns:1fr 1fr;gap:14px;
          ">
            <div>
              <div style="font-size:0.72rem;color:{_AMBER};font-weight:700;
                  text-transform:uppercase;letter-spacing:0.8px;margin-bottom:6px;">
                Step 1 — Categorical Encoding
              </div>
              <div style="font-size:0.80rem;color:{_TEXT_DIM};">
                Method: <b style='color:{_TEXT_MAIN};'>{enc_label}</b><br/>
                Columns: <b style='color:{_TEXT_MAIN};'>{enc_preview}</b>
              </div>
            </div>
            <div>
              <div style="font-size:0.72rem;color:{_AMBER};font-weight:700;
                  text-transform:uppercase;letter-spacing:0.8px;margin-bottom:6px;">
                Step 2 — Numeric Standardization
              </div>
              <div style="font-size:0.80rem;color:{_TEXT_DIM};">
                Method: <b style='color:{_TEXT_MAIN};'>{scale_label}</b><br/>
                Columns: <b style='color:{_TEXT_MAIN};'>{scale_preview}</b>
              </div>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _done_banner(result: dict) -> None:
    """Post-processing summary banner."""
    enc_done   = result.get("enc_cols", [])
    scale_done = result.get("scale_cols", [])
    rows       = result.get("rows", 0)
    cols_after = result.get("cols_after", 0)

    st.markdown(
        f"""
        <div style="
            background:rgba(39,174,96,0.08);
            border:1px solid rgba(39,174,96,0.25);
            border-left:4px solid #27ae60;
            border-radius:12px;
            padding:20px 22px;
            margin-bottom:18px;
        ">
          <div style="font-size:0.85rem;font-weight:700;color:#27ae60;
              text-transform:uppercase;letter-spacing:1px;margin-bottom:12px;">
            ✅&nbsp; Transformation Complete
          </div>
          <div style="display:flex;gap:24px;flex-wrap:wrap;font-size:0.82rem;">
            {_stat("Rows", f"{rows:,}")}
            {_stat("Columns", str(cols_after))}
            {_stat("Encoded", str(len(enc_done)) + " col(s)")}
            {_stat("Scaled", str(len(scale_done)) + " col(s)")}
          </div>
          {"<div style='margin-top:10px;font-size:0.78rem;color:" + _TEXT_DIM + ";'>"
           + "Encoded: <b style='color:" + _TEXT_MAIN + ";'>" + (", ".join(enc_done) or "—") + "</b></div>"
           if enc_done else ""}
          {"<div style='font-size:0.78rem;color:" + _TEXT_DIM + ";'>"
           + "Scaled: <b style='color:" + _TEXT_MAIN + ";'>" + (", ".join(scale_done) or "—") + "</b></div>"
           if scale_done else ""}
        </div>
        """,
        unsafe_allow_html=True,
    )


# ==============================================================================
# PIPELINE EXECUTOR
# ==============================================================================

def _run_feature_pipeline(
    df: pd.DataFrame,
    engine,
    enc_method: str,
    enc_cols: list,
    scale_method: str,
    scale_cols: list,
) -> dict:
    """
    Run 2-step feature preparation pipeline with live progress.
    Returns result dict for the done banner + download button.
    Does NOT save to disk or switch workspace.
    """
    STEPS = [
        (":material/code: Step 1/2 — Encoding categorical columns...", "enc"),
        (":material/straighten: Step 2/2 — Scaling numeric columns...", "scale"),
    ]

    progress = st.progress(0, text=STEPS[0][0])
    enc_done   = []
    scale_done = []

    # ── Step 1: Encode ────────────────────────────────────────────────────
    progress.progress(0.10, text=STEPS[0][0])
    if enc_cols:
        df = engine.encode_features(df.copy(), enc_method, enc_cols)
        enc_done = enc_cols
    time.sleep(0.25)

    # ── Step 2: Scale ─────────────────────────────────────────────────────
    progress.progress(0.60, text=STEPS[1][0])
    if scale_cols:
        # Only keep cols that still exist after encoding (one-hot may have dropped originals)
        valid_scale = [
            c for c in scale_cols
            if c in df.columns and pd.api.types.is_numeric_dtype(df[c])
        ]
        if valid_scale:
            # Cast to float64 first — avoids TypeError when assigning float
            # result back to int64 columns (e.g. after pd.get_dummies)
            df[valid_scale] = df[valid_scale].astype(float)
            df = engine.scale_features(df.copy(), scale_method, valid_scale)
            scale_done = valid_scale

    time.sleep(0.25)

    progress.progress(1.0, text=":material/check_circle: Transformation complete!")
    time.sleep(0.3)
    progress.empty()

    csv_bytes = df.to_csv(index=False).encode("utf-8-sig")  # BOM for Excel

    return {
        "df":         df,
        "csv_bytes":  csv_bytes,
        "enc_cols":   enc_done,
        "scale_cols": scale_done,
        "rows":       len(df),
        "cols_after": len(df.columns),
    }


# ==============================================================================
# DETAIL TABS (always visible below the control section)
# ==============================================================================

def _render_detail_tabs(df: pd.DataFrame, enc_cols: list, scale_cols: list) -> None:
    """Show schema + column selectors info in the detail tab area."""
    detail_analysis_header()

    tabs = st.tabs([
        ":material/schema: Schema Overview",
        ":material/code: Encoding Preview",
        ":material/straighten: Scaling Preview",
    ])

    # ── Tab 1: Schema ─────────────────────────────────────────────────────
    with tabs[0]:
        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        num_cols = df.select_dtypes(include="number").columns.tolist()

        c1, c2 = st.columns(2)
        with c1:
            st.markdown(
                f"<div style='font-size:0.75rem;color:{_AMBER};font-weight:700;"
                f"text-transform:uppercase;letter-spacing:0.8px;margin-bottom:8px;'>"
                f"Categorical Columns ({len(cat_cols)})</div>",
                unsafe_allow_html=True,
            )
            for col in cat_cols:
                n_unique = df[col].nunique()
                st.markdown(
                    f"<div style='font-size:0.78rem;color:{_TEXT_DIM};"
                    f"padding:4px 0;border-bottom:1px solid rgba(255,255,255,0.05);'>"
                    f"<b style='color:{_TEXT_MAIN};'>{col}</b>"
                    f"&ensp;<span style='color:{_AMBER};font-size:0.70rem;'>{n_unique} unique</span></div>",
                    unsafe_allow_html=True,
                )
        with c2:
            st.markdown(
                f"<div style='font-size:0.75rem;color:{_AMBER};font-weight:700;"
                f"text-transform:uppercase;letter-spacing:0.8px;margin-bottom:8px;'>"
                f"Numeric Columns ({len(num_cols)})</div>",
                unsafe_allow_html=True,
            )
            for col in num_cols:
                col_min = round(float(df[col].min()), 2)
                col_max = round(float(df[col].max()), 2)
                st.markdown(
                    f"<div style='font-size:0.78rem;color:{_TEXT_DIM};"
                    f"padding:4px 0;border-bottom:1px solid rgba(255,255,255,0.05);'>"
                    f"<b style='color:{_TEXT_MAIN};'>{col}</b>"
                    f"&ensp;<span style='color:{_AMBER};font-size:0.70rem;'>[{col_min}, {col_max}]</span></div>",
                    unsafe_allow_html=True,
                )

    # ── Tab 2: Encoding Preview ──────────────────────────────────────────
    with tabs[1]:
        if not enc_cols:
            st.info("No columns selected for encoding.")
        else:
            st.markdown(
                f"<div style='font-size:0.80rem;color:{_TEXT_DIM};margin-bottom:10px;'>"
                f"Preview of selected encoding columns (first 50 rows):</div>",
                unsafe_allow_html=True,
            )
            preview_cols = [c for c in enc_cols if c in df.columns]
            if preview_cols:
                st.dataframe(
                    df[preview_cols].head(50),
                    use_container_width=True,
                    hide_index=True,
                )
            for col in enc_cols:
                if col in df.columns:
                    vc = df[col].value_counts().head(10)
                    st.markdown(
                        f"<div style='font-size:0.75rem;color:{_AMBER};font-weight:700;"
                        f"margin-top:12px;margin-bottom:4px;'>{col} — Top {len(vc)} values</div>",
                        unsafe_allow_html=True,
                    )
                    st.dataframe(
                        vc.rename("Count").to_frame(),
                        use_container_width=True,
                    )

    # ── Tab 3: Scaling Preview ───────────────────────────────────────────
    with tabs[2]:
        if not scale_cols:
            st.info("No columns selected for scaling.")
        else:
            preview_cols = [c for c in scale_cols if c in df.columns]
            st.markdown(
                f"<div style='font-size:0.80rem;color:{_TEXT_DIM};margin-bottom:10px;'>"
                f"Descriptive statistics for scaling targets:</div>",
                unsafe_allow_html=True,
            )
            if preview_cols:
                desc = df[preview_cols].describe().T.round(3)
                desc.index.name = "Column"
                st.dataframe(desc, use_container_width=True)


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    """Feature Preparation page — encode + scale, no workspace switching."""
    lang = st.session_state.get("lang", "en")

    # ── Page header ───────────────────────────────────────────────────────
    page_header(
        title=get_text("feature_prep_title", lang),
        subtitle=get_text("overview_journey_feat_desc", lang),
    )

    # ── Workspace guard ───────────────────────────────────────────────────
    _ensure_workspace_active()
    active_file = st.session_state.get("active_file")
    workspace_status(active_file)

    # ── Load data ─────────────────────────────────────────────────────────
    df_raw = data_engine.load_and_standardize(
        active_file, _file_mtime=data_engine._get_file_mtime(active_file)
    )

    # Reset state when workspace changes
    if st.session_state.get("_feat_prep_file") != active_file:
        st.session_state["_feat_prep_file"]    = active_file
        st.session_state["feat_prep_done"]     = False
        st.session_state["feat_prep_result"]   = {}

    active_file_scan_progress_bar("_feature_prep_done")

    engine = preprocessing_engine.PreprocessingEngine
    done   = st.session_state.get("feat_prep_done", False)
    result = st.session_state.get("feat_prep_result", {})

    # =========================================================================
    # CONFIGURATION SECTION  (always visible, used to build pipeline card)
    # =========================================================================
    st.markdown(
        f"<div style='font-size:0.78rem;font-weight:700;color:{_AMBER};"
        f"text-transform:uppercase;letter-spacing:1px;margin-bottom:12px;'>"
        f"⚙️&nbsp; Configure Transformations</div>",
        unsafe_allow_html=True,
    )

    cat_cols = df_raw.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = df_raw.select_dtypes(include="number").columns.tolist()

    cfg_col1, cfg_col2 = st.columns(2, gap="large")

    with cfg_col1:
        st.markdown(
            f"<div style='font-size:0.78rem;color:{_AMBER};font-weight:700;"
            f"margin-bottom:6px;'>Step 1 — Categorical Encoding</div>",
            unsafe_allow_html=True,
        )
        enc_method_label = st.selectbox(
            "Encoding method",
            options=list(_ENC_OPTIONS.keys()),
            key="sel_enc_method",
            label_visibility="collapsed",
        )
        enc_cols = st.multiselect(
            "Select categorical columns to encode",
            options=cat_cols,
            default=cat_cols,
            key="ms_enc_cols",
            placeholder="Choose columns…",
        )

    with cfg_col2:
        st.markdown(
            f"<div style='font-size:0.78rem;color:{_AMBER};font-weight:700;"
            f"margin-bottom:6px;'>Step 2 — Numeric Standardization</div>",
            unsafe_allow_html=True,
        )
        scale_method_label = st.selectbox(
            "Scaling method",
            options=list(_SCALE_OPTIONS.keys()),
            key="sel_scale_method",
            label_visibility="collapsed",
        )
        scale_cols = st.multiselect(
            "Select numeric columns to scale",
            options=num_cols,
            default=num_cols,
            key="ms_scale_cols",
            placeholder="Choose columns…",
        )

    enc_method   = _ENC_OPTIONS[enc_method_label]
    scale_method = _SCALE_OPTIONS[scale_method_label]

    section_divider()

    # =========================================================================
    # PIPELINE CARD + PROCESS/DOWNLOAD BUTTON
    # =========================================================================
    if not done:
        _feature_pipeline_card(df_raw, enc_cols, enc_method, scale_cols, scale_method)

        _, col_run, _ = st.columns([1, 2, 1])
        with col_run:
            if st.button(
                "⚡  Process Transformation & Standardization",
                type="primary",
                use_container_width=True,
                key="btn_run_feat",
            ):
                with st.spinner("Applying feature transformations..."):
                    r = _run_feature_pipeline(
                        df_raw.copy(), engine,
                        enc_method, enc_cols,
                        scale_method, scale_cols,
                    )
                st.session_state["feat_prep_done"]   = True
                st.session_state["feat_prep_result"] = r
                st.rerun()

    else:
        # ── Done state: show banner + download button ─────────────────────
        _done_banner(result)

        basename    = active_file.replace("\\", "/").split("/")[-1].replace(".csv", "")
        dl_filename = f"{basename}_featured.csv"

        _, col_dl, _ = st.columns([1, 2, 1])
        with col_dl:
            st.download_button(
                label=":material/download: Download Featured CSV",
                data=result.get("csv_bytes", b""),
                file_name=dl_filename,
                mime="text/csv",
                use_container_width=True,
                type="primary",
                key="dl_featured_csv",
            )

    section_divider()

    # =========================================================================
    # DETAIL ANALYSIS TABS  (always visible)
    # =========================================================================
    _render_detail_tabs(df_raw, enc_cols, scale_cols)


if __name__ == "__main__":
    main()
