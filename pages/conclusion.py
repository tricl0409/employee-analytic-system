"""
conclusion.py — Conclusion: High-Income Archetype Persona

Interactive persona wheel visualization showing the composite profile
of a typical high earner, computed from the active dataset.

Interaction: click the persona silhouette to reveal data-driven values
with staggered CSS animations.
"""

import math

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from modules.core import data_engine
from modules.core.preprocessing_engine import PreprocessingEngine
from modules.ui import page_header, workspace_status, active_file_scan_progress_bar
from modules.ui.components import styled_alert
from modules.ui.icons import get_icon
from modules.utils.helpers import _ensure_workspace_active


# ==============================================================================
# DESIGN TOKENS
# ==============================================================================

_AMBER = "#FF9F43"
_AMBER_DIM = "rgba(255,159,67,0.08)"
_AMBER_BORDER = "rgba(255,159,67,0.22)"


# ==============================================================================
# COLUMN RESOLVER & INCOME MASK
# ==============================================================================

def _norm(s: str) -> str:
    return s.lower().replace("_", "").replace("-", "").replace(" ", "")


def _resolve(df: pd.DataFrame) -> dict:
    lookup = {_norm(c): c for c in df.columns}
    ALIASES = {
        "income":       ["income", "salary", "incomelabel"],
        "age":          ["age"],
        "education":    ["education", "educationnum", "education_num"],
        "occupation":   ["occupation", "job"],
        "hours":        ["hoursperweek", "workinghours", "hours"],
        "sex":          ["sex", "gender"],
        "marital":      ["maritalstatus", "marital"],
        "capital_gain": ["capitalgain", "capgain", "capital_gain"],
    }
    return {k: next((lookup[a] for a in v if a in lookup), None) for k, v in ALIASES.items()}


def _high_mask(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().str.contains(r">50k", regex=True, na=False)


def _apply_binning(df: pd.DataFrame) -> pd.DataFrame:
    """Apply binning from session analysis_rules on a copy."""
    rules = st.session_state.get("analysis_rules", {})
    binning_config = rules.get("binning_config", {})
    if not binning_config:
        return df
    df_binned = df.copy()
    return PreprocessingEngine.apply_binning_mapping(df_binned, binning_config)


# ==============================================================================
# ARCHETYPE COMPUTATION
# ==============================================================================

def _compute_archetype(df: pd.DataFrame) -> dict:
    """Compute 7-trait High-Income Archetype profile."""
    cols = _resolve(df)
    inc = cols.get("income")
    if not inc:
        return {}

    hi_mask = _high_mask(df[inc])
    hi_df = df[hi_mask]
    hi_count = len(hi_df)

    if hi_count == 0:
        return {}

    df_binned = _apply_binning(df)
    hi_binned = df_binned[hi_mask]

    arch: dict = {
        "total": len(df),
        "high_count": hi_count,
        "high_pct": round(hi_mask.mean() * 100, 1),
    }

    # 1. Gender
    sex_col = cols.get("sex")
    if sex_col and sex_col in hi_df.columns:
        sex_vc = hi_df[sex_col].astype(str).str.strip().value_counts()
        if not sex_vc.empty:
            top_sex = sex_vc.index[0]
            top_sex_pct = round(sex_vc.iloc[0] / hi_count * 100, 1)
            is_male = "male" in top_sex.lower() and "fe" not in top_sex.lower()
            arch["gender"] = {"label": top_sex, "pct": top_sex_pct, "is_male": is_male}

    # 2. Age
    age_col = cols.get("age")
    if age_col and age_col in hi_df.columns:
        ages = pd.to_numeric(hi_df[age_col], errors="coerce").dropna()
        if not ages.empty:
            arch["age"] = {
                "median": int(ages.median()),
                "q1": int(ages.quantile(0.25)),
                "q3": int(ages.quantile(0.75)),
            }

    # 3. Marital (binned)
    marital_col = cols.get("marital")
    if marital_col and marital_col in hi_binned.columns:
        m_vc = hi_binned[marital_col].astype(str).value_counts()
        if not m_vc.empty:
            arch["marital"] = {"label": m_vc.index[0], "pct": round(m_vc.iloc[0] / hi_count * 100, 1)}

    # 4. Education (binned)
    edu_col = cols.get("education")
    if edu_col and edu_col in hi_binned.columns:
        e_vc = hi_binned[edu_col].astype(str).value_counts()
        if not e_vc.empty:
            arch["education"] = {"label": e_vc.index[0], "pct": round(e_vc.iloc[0] / hi_count * 100, 1)}

    # 5. Occupation (binned)
    occ_col = cols.get("occupation")
    if occ_col and occ_col in hi_binned.columns:
        o_vc = hi_binned[occ_col].astype(str).value_counts()
        if not o_vc.empty:
            arch["occupation"] = {"label": o_vc.index[0], "pct": round(o_vc.iloc[0] / hi_count * 100, 1)}

    # 6. Hours
    hrs_col = cols.get("hours")
    if hrs_col and hrs_col in hi_df.columns:
        hrs = pd.to_numeric(hi_df[hrs_col], errors="coerce").dropna()
        if not hrs.empty:
            arch["hours"] = {"avg": round(hrs.mean(), 1), "pct_overtime": round((hrs > 40).sum() / len(hrs) * 100, 1)}

    # 7. Capital Gain
    cg_col = cols.get("capital_gain")
    if cg_col and cg_col in hi_df.columns:
        cg = pd.to_numeric(hi_df[cg_col], errors="coerce")
        has_cg = (cg > 0).sum()
        arch["capital_gain"] = {"pct": round(has_cg / hi_count * 100, 1)}

    return arch


# ==============================================================================
# PERSONA WHEEL — HTML/CSS/JS COMPONENT
# ==============================================================================

def _persona_wheel_html(arch: dict) -> str:
    """Generate the full interactive persona wheel as a self-contained HTML page."""

    gender_info = arch.get("gender", {})
    is_male = gender_info.get("is_male", True)
    gender_label = gender_info.get("label", "N/A")
    gender_pct = gender_info.get("pct", 0)

    age_info = arch.get("age", {})
    marital = arch.get("marital", {})
    edu = arch.get("education", {})
    occ = arch.get("occupation", {})
    hours = arch.get("hours", {})
    cg = arch.get("capital_gain", {})

    high_count = arch.get("high_count", 0)
    high_pct = arch.get("high_pct", 0)

    accent = "rgba(59,130,246,0.85)" if is_male else "rgba(236,72,153,0.85)"
    accent_glow = "rgba(59,130,246,0.25)" if is_male else "rgba(236,72,153,0.25)"
    accent_dim = "rgba(59,130,246,0.12)" if is_male else "rgba(236,72,153,0.12)"

    # 7 segments positioned around the wheel
    segments = [
        {
            "label": "GENDER", "value": f"{gender_label}", "sub": f"{gender_pct}% of high earners",
            "color": "#FF9F43", "angle": -90,
        },
        {
            "label": "AGE", "value": f"Median {age_info.get('median', '—')}",
            "sub": f"IQR {age_info.get('q1', '—')}–{age_info.get('q3', '—')} yrs",
            "color": "#3B82F6", "angle": -90 + 360 / 7,
        },
        {
            "label": "EDUCATION", "value": f"{edu.get('label', '—')}",
            "sub": f"{edu.get('pct', 0)}%",
            "color": "#10B981", "angle": -90 + 2 * 360 / 7,
        },
        {
            "label": "OCCUPATION", "value": f"{occ.get('label', '—')}",
            "sub": f"{occ.get('pct', 0)}%",
            "color": "#F59E0B", "angle": -90 + 3 * 360 / 7,
        },
        {
            "label": "MARITAL", "value": f"{marital.get('label', '—')}",
            "sub": f"{marital.get('pct', 0)}%",
            "color": "#8B5CF6", "angle": -90 + 4 * 360 / 7,
        },
        {
            "label": "HOURS/WK", "value": f"Avg {hours.get('avg', '—')}h",
            "sub": f"{hours.get('pct_overtime', 0)}% overtime",
            "color": "#EC4899", "angle": -90 + 5 * 360 / 7,
        },
        {
            "label": "CAPITAL GAIN", "value": f"{cg.get('pct', 0)}%",
            "sub": "have investment income",
            "color": "#6366F1", "angle": -90 + 6 * 360 / 7,
        },
    ]

    # Compute positions
    label_items = []
    arc_items = []
    connector_items = []
    angle_step = 360 / len(segments)

    for idx, seg in enumerate(segments):
        angle_rad = math.radians(seg["angle"])
        # Label position (outer — percentage of container)
        lx = 50 + 40 * math.cos(angle_rad)
        ly = 50 + 40 * math.sin(angle_rad)
        # Dot position (wheel edge)
        dx = 50 + 26 * math.cos(angle_rad)
        dy = 50 + 26 * math.sin(angle_rad)

        text_align = "left" if lx > 55 else ("right" if lx < 45 else "center")

        label_items.append(f"""
        <div class="seg" id="seg{idx}" style="left:{lx:.1f}%;top:{ly:.1f}%;text-align:{text_align};">
            <div class="seg-label" style="color:{seg['color']}">{seg['label']}</div>
            <div class="seg-val">{seg['value']}</div>
            <div class="seg-sub">{seg['sub']}</div>
        </div>
        """)

        connector_items.append(f"""
        <line x1="{dx:.1f}%" y1="{dy:.1f}%" x2="{lx:.1f}%" y2="{ly:.1f}%"
              stroke="{seg['color']}" stroke-width="1" stroke-dasharray="3,4"
              opacity="0.25" class="conn" />
        <circle cx="{dx:.1f}%" cy="{dy:.1f}%" r="3" fill="{seg['color']}" opacity="0.5" class="cdot" />
        """)

        # Wheel arc segments
        start_a = -90 + idx * angle_step
        end_a = start_a + angle_step
        r = 26
        x1 = 50 + r * math.cos(math.radians(start_a))
        y1 = 50 + r * math.sin(math.radians(start_a))
        x2 = 50 + r * math.cos(math.radians(end_a))
        y2 = 50 + r * math.sin(math.radians(end_a))
        arc_items.append(
            f'<path d="M 50,50 L {x1:.2f},{y1:.2f} A {r},{r} 0 0,1 {x2:.2f},{y2:.2f} Z"'
            f' fill="{seg["color"]}" opacity="0.12" class="warc" />'
        )



    labels_html = "\n".join(label_items)
    conns_html = "\n".join(connector_items)
    arcs_html = "\n".join(arc_items)


    # Persona SVG
    if is_male:
        persona_svg = f"""
        <circle cx="50" cy="30" r="12" fill="{accent}"/>
        <rect x="38" y="44" width="24" height="22" rx="5" fill="{accent}" opacity="0.85"/>
        <rect x="34" y="66" width="14" height="20" rx="3" fill="{accent}" opacity="0.7"/>
        <rect x="52" y="66" width="14" height="20" rx="3" fill="{accent}" opacity="0.7"/>
        """
    else:
        persona_svg = f"""
        <circle cx="50" cy="28" r="12" fill="{accent}"/>
        <path d="M35,86 Q36,48 50,44 Q64,48 65,86 Z" fill="{accent}" opacity="0.85"/>
        <ellipse cx="50" cy="54" rx="15" ry="10" fill="{accent}" opacity="0.75"/>
        """

    return f"""<!DOCTYPE html><html><head><style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{background:transparent;font-family:'Inter',-apple-system,sans-serif;color:#fff;overflow:hidden}}

.wrap{{position:relative;width:100%;max-width:640px;margin:0 auto;aspect-ratio:1}}

/* ── Wheel SVG ───────────────────── */
.wsvg{{position:absolute;inset:0;width:100%;height:100%}}
.warc{{transition:opacity .6s ease}}
.conn{{transition:opacity .5s ease}}
.cdot{{transition:opacity .5s ease,r .3s ease}}

/* ── Center ──────────────────────── */
.center{{
    position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);
    width:26%;aspect-ratio:1;border-radius:50%;
    background:radial-gradient(circle,rgba(15,18,32,0.96),rgba(10,13,28,0.99));
    border:2px solid {accent};
    box-shadow:0 0 30px {accent_glow},0 0 60px {accent_glow},inset 0 0 30px rgba(0,0,0,0.5);
    cursor:pointer;z-index:10;
    display:flex;flex-direction:column;align-items:center;justify-content:center;
    transition:transform .35s cubic-bezier(.4,0,.2,1),box-shadow .4s ease;
}}
.center:hover{{transform:translate(-50%,-50%) scale(1.07);
    box-shadow:0 0 50px {accent_glow},0 0 100px {accent_glow},inset 0 0 30px rgba(0,0,0,0.5)}}
.center svg{{width:55%;height:55%}}
.hint{{font-size:8px;color:rgba(255,255,255,0.35);text-transform:uppercase;
    letter-spacing:2px;margin-top:2px;transition:opacity .3s ease}}

/* ── Segments ────────────────────── */
.seg{{position:absolute;transform:translate(-50%,-50%);width:120px;z-index:5;
    transition:all .4s cubic-bezier(.4,0,.2,1)}}
.seg-label{{font-size:9px;font-weight:800;text-transform:uppercase;letter-spacing:2px;
    transition:all .3s ease}}
.seg-val,.seg-sub{{opacity:0;max-height:0;overflow:hidden;
    transform:translateY(8px);transition:all .45s cubic-bezier(.4,0,.2,1)}}
.seg-val{{font-size:14px;font-weight:700;color:rgba(255,255,255,0.92);margin-top:3px}}
.seg-sub{{font-size:10px;color:rgba(255,255,255,0.4);margin-top:1px}}



/* ── Stats bar ───────────────────── */
.sbar{{text-align:center;font-size:11px;color:rgba(255,255,255,0.35);
    margin-top:12px;letter-spacing:.3px;transition:opacity .4s ease .5s;opacity:0}}
.sbar b{{color:#FF9F43;font-weight:700}}

/* ═══ Revealed state ═════════════ */
.R .seg-val,.R .seg-sub{{opacity:1;max-height:60px;transform:translateY(0)}}
.R .warc{{opacity:.3!important}}
.R .conn{{opacity:.6!important}}
.R .cdot{{opacity:.9!important}}
.R .hint{{opacity:0}}

.R .sbar{{opacity:1}}

/* Stagger animation delays */
.R #seg0 .seg-val,.R #seg0 .seg-sub{{transition-delay:.05s}}
.R #seg1 .seg-val,.R #seg1 .seg-sub{{transition-delay:.12s}}
.R #seg2 .seg-val,.R #seg2 .seg-sub{{transition-delay:.19s}}
.R #seg3 .seg-val,.R #seg3 .seg-sub{{transition-delay:.26s}}
.R #seg4 .seg-val,.R #seg4 .seg-sub{{transition-delay:.33s}}
.R #seg5 .seg-val,.R #seg5 .seg-sub{{transition-delay:.40s}}
.R #seg6 .seg-val,.R #seg6 .seg-sub{{transition-delay:.47s}}

/* Glow pulse on revealed */
@keyframes gp{{
    0%,100%{{box-shadow:0 0 30px {accent_glow},0 0 60px {accent_glow},inset 0 0 30px rgba(0,0,0,.5)}}
    50%{{box-shadow:0 0 50px {accent_glow},0 0 100px {accent_glow},inset 0 0 30px rgba(0,0,0,.5)}}
}}
.R .center{{animation:gp 3s ease-in-out infinite}}

/* Outer ring pulse */
@keyframes rp{{0%,100%{{opacity:.08}}50%{{opacity:.18}}}}
.R .oring{{animation:rp 3s ease-in-out infinite}}
</style></head><body>

<div class="wrap" id="W">
    <svg class="wsvg" viewBox="0 0 100 100">
        {arcs_html}
        {conns_html}
        <circle cx="50" cy="50" r="26" fill="none" stroke="rgba(255,255,255,0.08)"
                stroke-width=".4" class="oring"/>
    </svg>
    <div class="center" onclick="document.getElementById('W').classList.toggle('R')">
        <svg viewBox="0 0 100 100">{persona_svg}</svg>
        <div class="hint">Click to reveal</div>
    </div>
    {labels_html}
</div>

<div class="sbar">
    Based on <b>{high_count:,}</b> High Income earners &nbsp;·&nbsp; <b>{high_pct}%</b> of dataset
</div>
</body></html>"""


# ==============================================================================
# MAIN
# ==============================================================================

def main() -> None:
    lang = st.session_state.get("lang", "en")

    page_header(
        title="Conclusion & Archetype Profile",
        subtitle="Interactive portrait of the typical High-Income earner — synthesized from all analysis dimensions.",
    )

    _ensure_workspace_active()
    active_file = st.session_state.get("active_file")
    workspace_status(active_file)

    df_raw = data_engine.load_and_standardize(
        active_file, _file_mtime=data_engine._get_file_mtime(active_file)
    )
    active_file_scan_progress_bar("_conclusion_done")

    if df_raw.empty:
        styled_alert("No data loaded. Please upload and activate a dataset first.", "warning")
        return

    # ── Compute archetype ──────────────────────────────────────────────
    cache_key = f"_conclusion_arch_v4_{active_file}"
    size_key = f"_conclusion_size_v4_{active_file}"
    if (
        cache_key not in st.session_state
        or st.session_state.get(size_key) != len(df_raw)
    ):
        with st.spinner("Computing High-Income Archetype…"):
            st.session_state[cache_key] = _compute_archetype(df_raw)
            st.session_state[size_key] = len(df_raw)

    arch = st.session_state[cache_key]

    if not arch:
        styled_alert("Could not compute archetype — no income column found.", "warning")
        return

    # ── Section header ─────────────────────────────────────────────────
    gender_info = arch.get("gender", {})
    is_male = gender_info.get("is_male", True)

    icon_svg = get_icon("target", size=18, color=_AMBER)
    st.markdown(
        f"<div style='padding:14px 18px;margin-top:4px;margin-bottom:20px;"
        f"background:linear-gradient(135deg,{_AMBER_DIM} 0%,rgba(255,159,67,0.02) 100%);"
        f"border:1px solid {_AMBER_BORDER};border-left:3px solid rgba(255,159,67,0.70);"
        f"border-radius:0 12px 12px 0;'>"
        f"<div style='display:flex;align-items:center;gap:10px;'>"
        f"{icon_svg}"
        f"<span style='font-size:1.08rem;font-weight:800;color:rgba(255,255,255,0.92);"
        f"letter-spacing:-0.3px;'>High-Income Archetype</span>"
        f"</div>"
        f"<div style='font-size:0.76rem;color:rgba(255,255,255,0.38);margin-top:4px;'>"
        f"Click the persona at the center of the wheel to reveal key characteristics"
        f"</div>"
        f"</div>",
        unsafe_allow_html=True,
    )

    # ── Render persona wheel ───────────────────────────────────────────
    html_content = _persona_wheel_html(arch)
    components.html(html_content, height=660, scrolling=False)


if __name__ == "__main__":
    main()
