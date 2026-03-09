"""
analytic_rule_settings.py — Admin-only Analytic Rule Settings
Professional 4-tab layout: Schema · Safe Zones · Noise Patterns · Binning & Grouping
Icons: synchronized with modules/ui/icons.py via get_icon()
"""

import streamlit as st
import pandas as pd

from modules.ui import page_header
from modules.ui.icons import get_icon
from modules.utils.localization import get_text
from modules.utils.db_config_manager import (
    get_all_rules, update_rule, load_rules_into_session,
    reset_all_rules, _DEFAULT_BINNING_CONFIG,
)


# ==============================================================================
# DESIGN TOKENS  — single amber accent
# ==============================================================================

_AMBER      = "rgba(255,159,67,0.80)"   # primary accent
_AMBER_DIM  = "rgba(255,159,67,0.40)"   # secondary / muted amber
_MUTED      = "rgba(255,255,255,0.32)"  # neutral dim text


# ==============================================================================
# UI HELPERS
# ==============================================================================

def _tab_header(icon_key: str, title: str, subtitle: str) -> None:
    """Section header with SVG icon from icons.py + amber left accent."""
    svg = get_icon(icon_key, size=20, color="#FF9F43")
    sub_html = (
        f"<div style='color:{_MUTED};font-size:0.76rem;margin-top:4px;'>{subtitle}</div>"
        if subtitle else ""
    )
    st.markdown(
        f"<div style='border-left:3px solid {_AMBER};"
        f"padding-left:14px;margin-bottom:18px;margin-top:6px;'>"
        f"<div style='display:flex;align-items:center;gap:9px;'>"
        f"{svg}"
        f"<span style='font-size:1.02rem;font-weight:800;color:#FFFFFF;"
        f"letter-spacing:-0.2px;'>{title}</span></div>"
        f"{sub_html}</div>",
        unsafe_allow_html=True,
    )


def _info_callout(text: str, kind: str = "info") -> None:
    """Callout box — all amber, differentiated by border opacity only.
    kind: 'info' (dim) | 'warning' (bright) | 'tip' (dim + check icon)
    """
    icon_map = {
        "info":    get_icon("zap",           size=14, color="#FF9F43"),
        "warning": get_icon("alert_triangle", size=14, color="#FF9F43"),
        "tip":     get_icon("check_circle",   size=14, color="#FF9F43"),
    }
    border = _AMBER if kind == "warning" else _AMBER_DIM
    svg    = icon_map.get(kind, icon_map["info"])
    st.markdown(
        f"<div style='background:rgba(255,159,67,0.05);"
        f"border-left:3px solid {border};"
        f"border-radius:6px;padding:10px 14px;font-size:0.78rem;"
        f"color:rgba(255,255,255,0.50);display:flex;gap:8px;"
        f"align-items:flex-start;margin-bottom:14px;'>"
        f"<span style='margin-top:1px;flex-shrink:0;'>{svg}</span>"
        f"<span style='line-height:1.6;'>{text}</span></div>",
        unsafe_allow_html=True,
    )


def _key_stat(label: str, value: str) -> str:
    """Inline metric chip — always amber accent."""
    return (
        f"<span style='background:rgba(255,159,67,0.06);"
        f"border:1px solid {_AMBER_DIM};border-radius:6px;"
        f"padding:3px 10px;font-size:0.76rem;color:{_AMBER};"
        f"font-weight:700;margin-right:6px;'>"
        f"{label}&nbsp;<span style='color:rgba(255,255,255,0.85);'>{value}</span></span>"
    )


def _divider() -> None:
    st.markdown(
        "<hr style='border:none;border-top:1px solid rgba(255,255,255,0.07);"
        "margin:18px 0;'>",
        unsafe_allow_html=True,
    )


def _save_and_refresh(rule_key: str, value, lang: str) -> None:
    """Persist rule → reload session → toast confirmation."""
    update_rule(rule_key, value)
    load_rules_into_session()
    st.toast(f"✅ {get_text('admin_save_success', lang)}", icon="✅")


# ==============================================================================
# TAB 1 — EMPLOYEE SCHEMA
# ==============================================================================

def _render_schema_editor(rules: dict, lang: str, v: int) -> None:
    _tab_header(
        "briefcase",
        get_text("admin_schema_title", lang),
        get_text("admin_schema_caption", lang),
    )

    schema    = rules.get("employee_schema", {"columns": []})
    cols_data = schema.get("columns", [])
    df        = (
        pd.DataFrame(cols_data)
        if cols_data
        else pd.DataFrame(columns=["name", "dtype", "category"])
    )

    n_numeric = sum(1 for c in cols_data if c.get("category") == "numeric")
    n_cat     = sum(1 for c in cols_data if c.get("category") == "categorical")

    # Key stats row
    st.markdown(
        _key_stat("Total columns", str(len(cols_data)))
        + _key_stat("Numeric",     str(n_numeric))
        + _key_stat("Categorical", str(n_cat)),
        unsafe_allow_html=True,
    )
    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

    _info_callout(
        "Schema defines the <b>expected structure</b> of uploaded CSV files. "
        "Columns missing from this definition will trigger a <b>validation error</b> on upload. "
        "The <code>category</code> field determines how each column is treated in analysis.",
        kind="info",
    )

    edited = st.data_editor(
        df,
        num_rows="dynamic",
        use_container_width=True,
        hide_index=True,
        key=f"editor_schema_{v}",
        column_config={
            "name": st.column_config.TextColumn(
                get_text("admin_col_name", lang), required=True,
                help="Exact column name (case-sensitive match against uploaded file)",
            ),
            "dtype": st.column_config.SelectboxColumn(
                get_text("admin_col_dtype", lang),
                options=["float64", "int64", "object", "bool", "datetime64"],
                required=True,
                help="Expected pandas dtype",
            ),
            "category": st.column_config.SelectboxColumn(
                get_text("admin_col_category", lang),
                options=["numeric", "categorical"],
                required=True,
                help="Determines how the column is used in audit & EDA",
            ),
        },
    )

    _divider()

    _, btn_col = st.columns([4, 1])
    with btn_col:
        if st.button(
            f":material/save: {get_text('admin_btn_save_schema', lang)}",
            key=f"btn_save_schema_{v}", type="primary", use_container_width=True,
        ):
            _save_and_refresh("employee_schema", {"columns": edited.to_dict(orient="records")}, lang)


# ==============================================================================
# TAB 2 — SAFE ZONES
# ==============================================================================

def _render_safe_zones(rules: dict, lang: str, v: int) -> None:
    _tab_header(
        "audit",
        get_text("admin_safe_zones_title", lang),
        get_text("admin_safe_zones_caption", lang),
    )

    safe = rules.get("safe_zones", {})
    rows = [{"Column": k, "Min": vv.get("min", 0), "Max": vv.get("max", 0)} for k, vv in safe.items()]
    df   = pd.DataFrame(rows) if rows else pd.DataFrame(columns=["Column", "Min", "Max"])

    st.markdown(
        _key_stat("Protected columns", str(len(safe))),
        unsafe_allow_html=True,
    )
    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

    _info_callout(
        "Safe Zones define the <b>acceptable value range</b> for numeric columns during outlier detection. "
        "Values <b>outside</b> the [Min, Max] range are flagged or clipped — even if they pass the IQR/Z-score test. "
        "Leave a column out of this list to rely solely on statistical methods.",
        kind="warning",
    )

    edited = st.data_editor(
        df,
        num_rows="dynamic",
        use_container_width=True,
        hide_index=True,
        key=f"editor_safe_zones_{v}",
        column_config={
            "Column": st.column_config.TextColumn(
                get_text("admin_col_name", lang), required=True,
                help="Column name must exactly match the schema definition",
            ),
            "Min": st.column_config.NumberColumn(
                get_text("admin_col_safe_min", lang), required=True,
                help="Values below this threshold are treated as outliers",
            ),
            "Max": st.column_config.NumberColumn(
                get_text("admin_col_safe_max", lang), required=True,
                help="Values above this threshold are treated as outliers",
            ),
        },
    )

    _divider()

    _, btn_col = st.columns([4, 1])
    with btn_col:
        if st.button(
            f":material/save: {get_text('admin_btn_save_safe', lang)}",
            key=f"btn_save_safe_{v}", type="primary", use_container_width=True,
        ):
            new_safe = {
                str(row["Column"]).strip(): {"min": float(row["Min"]), "max": float(row["Max"])}
                for _, row in edited.iterrows()
                if str(row["Column"]).strip()
            }
            _save_and_refresh("safe_zones", new_safe, lang)


# ==============================================================================
# TAB 3 — NOISE PATTERNS
# ==============================================================================

def _render_noise_patterns(rules: dict, lang: str, v: int) -> None:
    _tab_header(
        "scissors",
        get_text("admin_noise_title", lang),
        get_text("admin_noise_caption", lang),
    )

    patterns = rules.get("noise_patterns", [])

    st.markdown(
        _key_stat("Active patterns", str(len(patterns))),
        unsafe_allow_html=True,
    )
    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

    _info_callout(
        "Noise patterns are <b>string values recognized as missing data</b> during the scrubbing step. "
        "Matching cells (case-insensitive, trimmed) are replaced with <code>NaN</code> before analysis. "
        "Add one pattern per line. Common values: <code>?</code>, <code>n/a</code>, <code>unknown</code>.",
        kind="info",
    )

    c_editor, c_preview = st.columns([3, 2])

    with c_editor:
        text = st.text_area(
            get_text("admin_noise_label", lang),
            value="\n".join(patterns),
            height=280,
            key=f"ta_noise_patterns_{v}",
            help="One pattern per line. Matching is case-insensitive and whitespace-trimmed.",
        )

    with c_preview:
        parsed = [p.strip() for p in text.strip().split("\n") if p.strip()]
        st.markdown(
            f"<div style='color:{_MUTED};font-size:0.72rem;text-transform:uppercase;"
            f"letter-spacing:1px;margin-bottom:8px;'>Preview · {len(parsed)} patterns</div>",
            unsafe_allow_html=True,
        )
        chips = "".join(
            f"<span style='background:rgba(255,255,255,0.05);"
            f"border:1px solid rgba(255,159,67,0.3);color:rgba(255,255,255,0.55);"
            f"border-radius:4px;padding:2px 8px;font-size:0.72rem;"
            f"margin:3px;display:inline-block;font-family:monospace;'>{p}</span>"
            for p in parsed
        )
        st.markdown(
            f"<div style='line-height:2;padding:8px 0;'>{chips if chips else '<span style=\"color:rgba(255,255,255,0.2);\">—</span>'}</div>",
            unsafe_allow_html=True,
        )

    _divider()

    _, btn_col = st.columns([4, 1])
    with btn_col:
        if st.button(
            f":material/save: {get_text('admin_btn_save_noise', lang)}",
            key=f"btn_save_noise_{v}", type="primary", use_container_width=True,
        ):
            new_patterns = [p.strip() for p in text.strip().split("\n") if p.strip()]
            _save_and_refresh("noise_patterns", new_patterns, lang)


# ==============================================================================
# TAB 4 — BINNING & GROUPING
# ==============================================================================

_BIN_LABELS = {"bin": "Numeric Binning", "map": "Categorical Map"}
_BIN_COLORS = {"bin": _AMBER, "map": _AMBER_DIM}


def _render_binning_config(rules: dict, lang: str, v: int) -> None:
    _tab_header(
        "ruler",
        "Binning & Mapping Rules",
        "Define how raw column values are bucketed or merged for EDA and modelling.",
    )

    cfg: dict = rules.get("binning_config", _DEFAULT_BINNING_CONFIG)
    columns   = list(cfg.keys())

    # ── Column picker strip ───────────────────────────────────────────────
    n_bin = sum(1 for c in cfg.values() if c.get("type") == "bin")
    n_map = sum(1 for c in cfg.values() if c.get("type") == "map")
    st.markdown(
        _key_stat("Total rules", str(len(cfg)))
        + _key_stat("Numeric bins", str(n_bin))
        + _key_stat("Category maps", str(n_map)),
        unsafe_allow_html=True,
    )
    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

    _info_callout(
        "Binning & Grouping rules transform raw column values into <b>analysis-ready categories</b>. "
        "<b>Numeric Binning</b> uses edges (e.g. age → 5 groups). "
        "<b>Categorical Mapping</b> merges many raw labels into fewer macro-groups. "
        "These rules are applied <b>on-the-fly</b> in EDA — the original data is never modified.",
        kind="tip",
    )

    selected = st.selectbox(
        "Edit rule for column:",
        options=columns,
        key=f"bin_col_select_{v}",
        format_func=lambda c: f"{_BIN_LABELS.get(cfg[c]['type'], '?')}  ·  {c}",
    )

    if not selected:
        return

    rule   = cfg[selected]
    rtype  = rule.get("type", "map")
    glow   = _BIN_COLORS.get(rtype, _MUTED)
    svg_lbl = get_icon("bar_chart" if rtype == "bin" else "copy", size=13, color=glow)

    # ── Column metadata strip ─────────────────────────────────────────────
    st.markdown(
        f"<div style='display:flex;align-items:center;gap:10px;"
        f"padding:10px 14px;background:rgba(255,255,255,0.025);"
        f"border-radius:8px;margin-bottom:14px;border:1px solid rgba(255,255,255,0.06);'>"
        f"<span style='font-size:1.0rem;font-weight:800;color:#fff;'>{selected}</span>"
        f"<span style='background:rgba(255,255,255,0.05);border:1px solid {glow};"
        f"color:{glow};border-radius:20px;padding:2px 10px;"
        f"font-size:0.68rem;font-weight:700;display:flex;align-items:center;gap:5px;'>"
        f"{svg_lbl}&nbsp;{_BIN_LABELS[rtype]}</span>"
        f"<span style='color:{_MUTED};font-size:0.72rem;margin-left:auto;'>"
        f"rule key: <code style='color:{glow};'>binning_config.{selected}</code></span>"
        f"</div>",
        unsafe_allow_html=True,
    )

    # ── Numeric Binning editor ────────────────────────────────────────────
    if rtype == "bin":
        bins_raw   = rule.get("bins", [])
        labels_raw = rule.get("labels", [])

        _info_callout(
            f"Constraint: <b>len(labels) must equal len(edges) − 1</b>. "
            f"Currently: <b>{len(bins_raw)} edges → {len(bins_raw)-1} expected labels</b>.",
            kind="warning",
        )

        c_bins, c_labels = st.columns(2)
        with c_bins:
            st.markdown(
                f"<div style='color:{_MUTED};font-size:0.73rem;margin-bottom:4px;'>"
                f"📐 <b>Bin edges</b> — one numeric value per line</div>",
                unsafe_allow_html=True,
            )
            bins_text = st.text_area(
                "Edges", value="\n".join(str(b) for b in bins_raw),
                height=200, key=f"bin_edges_{selected}_{v}",
                label_visibility="collapsed",
            )
        with c_labels:
            st.markdown(
                f"<div style='color:{_MUTED};font-size:0.73rem;margin-bottom:4px;'>"
                f"🏷️ <b>Group labels</b> — one string per line (len = edges − 1)</div>",
                unsafe_allow_html=True,
            )
            labels_text = st.text_area(
                "Labels", value="\n".join(labels_raw),
                height=200, key=f"bin_labels_{selected}_{v}",
                label_visibility="collapsed",
            )

        _divider()
        _, btn_col = st.columns([4, 1])
        with btn_col:
            if st.button(
                ":material/save: Save",
                key=f"btn_save_bin_{selected}_{v}", type="primary", use_container_width=True,
            ):
                try:
                    new_bins   = [float(x.strip()) for x in bins_text.strip().split("\n") if x.strip()]
                    new_labels = [x.strip() for x in labels_text.strip().split("\n") if x.strip()]
                    if len(new_labels) != len(new_bins) - 1:
                        st.error(
                            f"❌ Expected {len(new_bins)-1} labels for {len(new_bins)} edges. "
                            f"Got {len(new_labels)}."
                        )
                    else:
                        cfg[selected] = {"type": "bin", "bins": new_bins, "labels": new_labels}
                        _save_and_refresh("binning_config", cfg, lang)
                except ValueError as e:
                    st.error(f"❌ Parse error: {e}")

    # ── Categorical Mapping editor ────────────────────────────────────────
    elif rtype == "map":
        groups: dict = rule.get("groups", {})

        _info_callout(
            f"<b>{len(groups)} groups</b> configured for <b>{selected}</b>. "
            "Expand each group below to rename or edit its raw values. "
            "Values <b>not listed</b> in any group will be left unchanged.",
            kind="info",
        )

        edited_groups: dict = {}
        for group_name, values in groups.items():
            n_vals = len(values)
            with st.expander(f"**{group_name}** · {n_vals} value{'s' if n_vals != 1 else ''}", expanded=False):
                ca, cb = st.columns([1, 2])
                with ca:
                    new_name = st.text_input(
                        "Group label", value=group_name,
                        key=f"grp_name_{selected}_{group_name}_{v}",
                        help="The new, clean label used in analysis output",
                    )
                with cb:
                    raw_vals = st.text_area(
                        "Raw values (one per line)", value="\n".join(values),
                        height=110, key=f"grp_vals_{selected}_{group_name}_{v}",
                        help="Original column values to map → this group label",
                    )
                edited_groups[new_name.strip() or group_name] = [
                    x.strip() for x in raw_vals.strip().split("\n") if x.strip()
                ]

        # Add new group
        _divider()
        with st.expander(":material/add_circle: Add new group", expanded=False):
            _info_callout("New group will be saved together with all existing groups when you click Save.", kind="tip")
            ca2, cb2 = st.columns([1, 2])
            with ca2:
                ng_name = st.text_input("Group label", key=f"new_grp_name_{selected}_{v}")
            with cb2:
                ng_vals = st.text_area(
                    "Raw values (one per line)", height=100,
                    key=f"new_grp_vals_{selected}_{v}",
                )
            if ng_name.strip():
                edited_groups[ng_name.strip()] = [
                    x.strip() for x in ng_vals.strip().split("\n") if x.strip()
                ]

        _divider()
        _, btn_col = st.columns([4, 1])
        with btn_col:
            if st.button(
                ":material/save: Save",
                key=f"btn_save_map_{selected}_{v}", type="primary", use_container_width=True,
            ):
                cfg[selected] = {"type": "map", "groups": edited_groups}
                _save_and_refresh("binning_config", cfg, lang)

    # ── Column chips (all configured rules) ──────────────────────────────
    _divider()
    st.markdown(
        f"<div style='color:{_MUTED};font-size:0.70rem;text-transform:uppercase;"
        f"letter-spacing:1px;margin-bottom:8px;'>All configured columns</div>",
        unsafe_allow_html=True,
    )
    chips = "".join(
        f"<span style='background:rgba(255,255,255,0.03);"
        f"border:1px solid {_BIN_COLORS.get(cfg[c]['type'], _MUTED)};"
        f"color:rgba(255,255,255,0.55);border-radius:20px;"
        f"padding:3px 12px;font-size:0.71rem;margin:3px;display:inline-block;'>"
        f"{'📐' if cfg[c]['type']=='bin' else '🏷️'}&nbsp;{c}</span>"
        for c in cfg
    )
    st.markdown(f"<div style='line-height:2.2;'>{chips}</div>", unsafe_allow_html=True)


# ==============================================================================
# MAIN
# ==============================================================================

def main() -> None:
    lang = st.session_state.get("lang", "en")

    # ── Reset handler — runs BEFORE get_all_rules() ───────────────────────
    if st.session_state.pop("_do_reset", False):
        reset_all_rules()
        load_rules_into_session()
        st.session_state["_rule_version"] = st.session_state.get("_rule_version", 0) + 1
        st.toast("✅ All rules have been reset to factory defaults.", icon="✅")

    v = st.session_state.get("_rule_version", 0)

    page_header(
        title=get_text("admin_settings_title", lang),
        subtitle=get_text("admin_settings_subtitle", lang),
    )

    # ── Access guard ──────────────────────────────────────────────────────
    if st.session_state.get("user_role") != "admin":
        st.warning(get_text("admin_access_denied", lang))
        st.stop()

    rules = get_all_rules()

    # ── Global toolbar ────────────────────────────────────────────────────
    _, col_reset = st.columns([5, 1])
    with col_reset:
        if st.button(
            f":material/restart_alt: {get_text('admin_btn_reset', lang)}",
            use_container_width=True,
            help="⚠️ This will overwrite ALL rules with factory defaults. Action cannot be undone.",
        ):
            st.session_state["_do_reset"] = True
            st.rerun()

    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

    # ── 4 Tabs ────────────────────────────────────────────────────────────
    tabs = st.tabs([
        f":material/table_chart: {get_text('admin_tab_schema', lang)}",
        f":material/shield: {get_text('admin_tab_safe_zones', lang)}",
        f":material/content_cut: {get_text('admin_tab_noise', lang)}",
        ":material/ruler: Binning & Mapping",
    ])

    with tabs[0]: _render_schema_editor(rules, lang, v)
    with tabs[1]: _render_safe_zones(rules, lang, v)
    with tabs[2]: _render_noise_patterns(rules, lang, v)
    with tabs[3]: _render_binning_config(rules, lang, v)


if __name__ == "__main__":
    main()
