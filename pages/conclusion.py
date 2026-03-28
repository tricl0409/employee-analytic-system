"""
conclusion.py — Conclusion: High-Income Profile & Key Messages

Two-section layout:
1. Typical High-Income Profile — image + computed traits card
2. Key Message Delivery — three actionable insight cards
"""

import base64
import os

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from modules.core import data_engine
from modules.core.preprocessing_engine import PreprocessingEngine
from modules.ui import page_header, workspace_status, active_file_scan_progress_bar, section_divider
from modules.ui.components import styled_alert
from modules.ui.icons import get_icon
from modules.utils.helpers import _ensure_workspace_active, _high_mask
from modules.utils.localization import get_text


# ==============================================================================
# DESIGN TOKENS
# ==============================================================================

_AMBER = "#FF9F43"
_AMBER_DIM = "rgba(255,159,67,0.08)"
_AMBER_BORDER = "rgba(255,159,67,0.22)"

_BLUE = "#3B82F6"
_GREEN = "#10B981"

# Trait icon colors — consistent visual mapping
_TRAIT_COLORS = {
    "marital": ("#8B5CF6", "rgba(139,92,246,0.15)"),
    "education": ("#10B981", "rgba(16,185,129,0.15)"),
    "occupation": ("#F59E0B", "rgba(245,158,11,0.15)"),
    "capital": ("#6366F1", "rgba(99,102,241,0.15)"),
    "hours": ("#EC4899", "rgba(236,72,153,0.15)"),
    "age": ("#3B82F6", "rgba(59,130,246,0.15)"),
    "gender": ("#FF9F43", "rgba(255,159,67,0.15)"),
}

# Trait icon keys from the icons registry
_TRAIT_ICONS = {
    "marital": "heart",
    "education": "graduation_cap",
    "occupation": "briefcase",
    "capital": "trending_up",
    "hours": "clock",
    "age": "users",
    "gender": "user_icon",
}


# ==============================================================================
# COLUMN RESOLVER & INCOME MASK
# ==============================================================================

def _resolve(df: pd.DataFrame) -> dict:
    """Map expected archetype features to exact standardized column names."""
    return {
        "income": "income" if "income" in df.columns else None,
        "age": "age" if "age" in df.columns else None,
        "education": "education" if "education" in df.columns else "education_num" if "education_num" in df.columns else None,
        "occupation": "occupation" if "occupation" in df.columns else None,
        "hours": "hours_per_week" if "hours_per_week" in df.columns else None,
        "sex": "sex" if "sex" in df.columns else "gender" if "gender" in df.columns else None,
        "marital": "marital_status" if "marital_status" in df.columns else None,
        "relationship": "relationship" if "relationship" in df.columns else None,
        "capital_gain": "capital_gain" if "capital_gain" in df.columns else None,
    }


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
            arch["gender"] = {"label": top_sex, "pct": top_sex_pct}

    # 2. Age — compute Q1/Q3 + prime working-age share
    age_col = cols.get("age")
    if age_col and age_col in hi_df.columns:
        ages = pd.to_numeric(hi_df[age_col], errors="coerce").dropna()
        if not ages.empty:
            q1_val = int(ages.quantile(0.25))
            q3_val = int(ages.quantile(0.75))
            prime_mask = ages.between(36, 65)
            prime_pct = round(prime_mask.sum() / len(ages) * 100, 1)
            arch["age"] = {
                "median": int(ages.median()),
                "q1": q1_val,
                "q3": q3_val,
                "prime_pct": prime_pct,
            }

    # 3. Marital (binned)
    marital_col = cols.get("marital")
    if marital_col and marital_col in hi_binned.columns:
        m_vc = hi_binned[marital_col].astype(str).value_counts()
        if not m_vc.empty:
            arch["marital"] = {"label": m_vc.index[0], "pct": round(m_vc.iloc[0] / hi_count * 100, 1)}

    # 4. Education — group higher-education levels (use RAW data for matching)
    _HIGHER_ED = {"bachelors", "masters", "doctorate", "prof-school",
                  "bachelor's", "master's", "phd", "professional"}
    edu_col = cols.get("education")
    if edu_col and edu_col in hi_df.columns:
        e_series_raw = hi_df[edu_col].astype(str).str.strip()
        e_vc_raw = e_series_raw.value_counts()
        if not e_vc_raw.empty:
            # Group higher-education categories from raw values
            higher_mask = e_series_raw.str.lower().isin(_HIGHER_ED)
            higher_count = higher_mask.sum()
            higher_pct = round(higher_count / hi_count * 100, 1)
            top_label = e_vc_raw.index[0]
            top_pct = round(e_vc_raw.iloc[0] / hi_count * 100, 1)
            # Use grouped label if it covers significantly more than single top
            if higher_pct > top_pct + 5:
                arch["education"] = {
                    "label": "Bachelor's or higher",
                    "pct": higher_pct,
                    "detail": top_label,
                }
            else:
                arch["education"] = {"label": top_label, "pct": top_pct}

    # 5. Occupation (binned)
    occ_col = cols.get("occupation")
    if occ_col and occ_col in hi_binned.columns:
        o_vc = hi_binned[occ_col].astype(str).value_counts()
        if not o_vc.empty:
            arch["occupation"] = {"label": o_vc.index[0], "pct": round(o_vc.iloc[0] / hi_count * 100, 1)}

    # 6. Relationship (raw — core household roles)
    rel_col = cols.get("relationship")
    if rel_col and rel_col in hi_df.columns:
        r_vc = hi_df[rel_col].astype(str).str.strip().value_counts()
        if not r_vc.empty:
            arch["relationship"] = {"label": r_vc.index[0], "pct": round(r_vc.iloc[0] / hi_count * 100, 1)}

    # 7. Hours — full-time (>=40h) and average
    hrs_col = cols.get("hours")
    if hrs_col and hrs_col in hi_df.columns:
        hrs = pd.to_numeric(hi_df[hrs_col], errors="coerce").dropna()
        if not hrs.empty:
            arch["hours"] = {
                "avg": round(hrs.mean(), 1),
                "pct_fulltime": round((hrs >= 40).sum() / len(hrs) * 100, 1),
            }

    # 8. Capital Gain
    cg_col = cols.get("capital_gain")
    if cg_col and cg_col in hi_df.columns:
        cg = pd.to_numeric(hi_df[cg_col], errors="coerce")
        has_cg = (cg > 0).sum()
        arch["capital_gain"] = {"pct": round(has_cg / hi_count * 100, 1)}

    return arch


# ==============================================================================
# IMAGE LOADER — base64 encode the profile photo
# ==============================================================================

def _load_profile_image_b64() -> str:
    """Load high_income_profile.png as a base64 data URI."""
    assets_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets")
    img_path = os.path.join(assets_dir, "high_income_profile.png")
    try:
        with open(img_path, "rb") as fp:
            encoded = base64.b64encode(fp.read()).decode("ascii")
        return f"data:image/png;base64,{encoded}"
    except Exception:
        return ""


# ==============================================================================
# SECTION 1 — HIGH-INCOME PROFILE (premium HTML component)
# ==============================================================================

def _build_profile_html(arch: dict, img_b64: str, lang: str) -> str:
    """Build premium executive-style profile card as self-contained HTML.

    Args:
        arch: Archetype data computed from the dataset.
        img_b64: Base64-encoded profile image data URI.
        lang: Language code.

    Returns:
        Complete HTML string for the profile component.
    """
    t = lambda key, **kw: get_text(key, lang, **kw)

    gender = arch.get("gender", {})
    age = arch.get("age", {})
    marital = arch.get("marital", {})
    relationship = arch.get("relationship", {})
    edu = arch.get("education", {})
    occ = arch.get("occupation", {})
    hours = arch.get("hours", {})
    capital = arch.get("capital_gain", {})
    high_count = arch.get("high_count", 0)
    high_pct = arch.get("high_pct", 0)

    # Build trait items with stat bars
    traits = []
    if marital:
        traits.append({
            "label": t("conclusion_trait_marital"),
            "value": marital.get("label", "—"),
            "pct": marital.get("pct", 0),
            "color": "#8B5CF6",
        })
    if relationship:
        traits.append({
            "label": t("conclusion_trait_role"),
            "value": relationship.get("label", "—"),
            "pct": relationship.get("pct", 0),
            "color": "#14B8A6",
        })
    if edu:
        edu_value = edu.get("label", "—")
        edu_detail = edu.get("detail", "")
        if edu_detail:
            edu_value += f' (incl. {edu_detail})'
        traits.append({
            "label": t("conclusion_trait_education"),
            "value": edu_value,
            "pct": edu.get("pct", 0),
            "color": "#10B981",
        })
    if occ:
        traits.append({
            "label": t("conclusion_trait_occupation"),
            "value": occ.get("label", "—"),
            "pct": occ.get("pct", 0),
            "color": "#F59E0B",
        })
    if capital:
        traits.append({
            "label": t("conclusion_trait_capital"),
            "value": f'{capital.get("pct", 0)}% {t("conclusion_trait_invest_label")}',
            "pct": capital.get("pct", 0),
            "color": "#6366F1",
        })
    if hours:
        traits.append({
            "label": t("conclusion_trait_hours"),
            "value": f'{t("conclusion_trait_avg_label", val=hours.get("avg", "—"))}'
                     f' · {t("conclusion_trait_fulltime_label", pct=hours.get("pct_fulltime", 0))}',
            "pct": min(hours.get("pct_fulltime", 0), 100),
            "color": "#EC4899",
        })
    if age:
        prime_pct = age.get("prime_pct", 0)
        traits.append({
            "label": t("conclusion_trait_age"),
            "value": f'{t("conclusion_trait_age_range_label", q1=age.get("q1", "—"), q3=age.get("q3", "—"))}'
                     f' · {prime_pct}% in prime (36\u201365)',
            "pct": prime_pct,
            "color": "#3B82F6",
        })
    if gender:
        traits.append({
            "label": t("conclusion_trait_gender"),
            "value": gender.get("label", "—"),
            "pct": gender.get("pct", 0),
            "color": "#FF9F43",
        })

    # Build trait HTML rows with animated stat bars
    traits_html = ""
    for idx, tr in enumerate(traits):
        delay = 0.3 + idx * 0.12
        traits_html += f"""
        <div class="trait" style="animation-delay:{delay}s">
            <div class="trait-header">
                <span class="trait-label">{tr['label']}</span>
                <span class="trait-pct" style="color:{tr['color']}">{tr['pct']}%</span>
            </div>
            <div class="trait-value">{tr['value']}</div>
            <div class="stat-bar">
                <div class="stat-fill" style="width:{tr['pct']}%;background:{tr['color']};animation-delay:{delay + 0.2}s"></div>
            </div>
        </div>
        """

    # Footer line
    footer_text = t("conclusion_profile_based_on", count=f"{high_count:,}", pct=high_pct)

    return f"""<!DOCTYPE html><html><head>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap" rel="stylesheet">
<style>
* {{ margin:0; padding:0; box-sizing:border-box; }}
html, body {{ background:transparent; font-family:'Inter',sans-serif; color:#fff; overflow:hidden; }}

/* ── Container ── */
.profile-wrap {{
    display:flex;
    gap:0;
    border-radius:20px;
    overflow:hidden;
    border:1px solid rgba(255,159,67,0.20);
    box-shadow:0 12px 48px rgba(0,0,0,0.5), 0 0 40px rgba(255,159,67,0.06);
    background:rgba(15,18,35,0.85);
    backdrop-filter:blur(20px);
    animation:cardEntry 0.6s ease-out both;
    position:relative;
}}
.profile-wrap::before {{
    content:'';
    position:absolute;
    top:0;left:0;right:0;
    height:3px;
    background:linear-gradient(90deg,#FF9F43,#F27024,#FF9F43);
    background-size:200% 100%;
    animation:gradShift 4s ease infinite;
}}
@keyframes gradShift {{
    0%   {{ background-position:0% 50%; }}
    50%  {{ background-position:100% 50%; }}
    100% {{ background-position:0% 50%; }}
}}
@keyframes cardEntry {{
    from {{ opacity:0; transform:translateY(20px); }}
    to   {{ opacity:1; transform:translateY(0); }}
}}

/* ── Photo column ── */
.photo-col {{
    width:38%;
    position:relative;
    overflow:hidden;
    flex-shrink:0;
}}
.photo-col img {{
    width:100%;
    height:100%;
    object-fit:cover;
    display:block;
}}
.photo-overlay {{
    position:absolute;
    bottom:0;left:0;right:0;
    height:55%;
    background:linear-gradient(to top,rgba(15,18,35,0.95) 0%,rgba(15,18,35,0.4) 60%,transparent 100%);
    pointer-events:none;
}}
.photo-badge {{
    position:absolute;
    bottom:20px;left:20px;
    display:flex;
    flex-direction:column;
    gap:4px;
    z-index:2;
}}
.photo-title {{
    font-size:1.15rem;
    font-weight:800;
    color:#fff;
    letter-spacing:-0.5px;
    text-shadow:0 2px 12px rgba(0,0,0,0.6);
}}
.photo-sub {{
    font-size:0.7rem;
    font-weight:600;
    color:rgba(255,255,255,0.55);
    text-transform:uppercase;
    letter-spacing:1.5px;
}}
.photo-tag {{
    display:inline-flex;
    align-items:center;
    gap:5px;
    padding:4px 12px;
    background:rgba(255,159,67,0.18);
    border:1px solid rgba(255,159,67,0.35);
    border-radius:20px;
    font-size:0.65rem;
    font-weight:700;
    color:#FF9F43;
    letter-spacing:0.8px;
    text-transform:uppercase;
    margin-top:4px;
    width:fit-content;
}}
.photo-tag-dot {{
    width:6px;height:6px;border-radius:50%;
    background:#FF9F43;
    box-shadow:0 0 8px rgba(255,159,67,0.6);
    animation:dotPulse 2s ease-in-out infinite;
}}
@keyframes dotPulse {{
    0%,100% {{ opacity:1; }}
    50%     {{ opacity:0.4; }}
}}

/* ── Stats column ── */
.stats-col {{
    flex:1;
    padding:28px 28px 20px 28px;
    display:flex;
    flex-direction:column;
    gap:4px;
}}

/* ── Trait row ── */
.trait {{
    padding:8px 0;
    animation:traitIn 0.45s ease-out both;
}}
@keyframes traitIn {{
    from {{ opacity:0; transform:translateX(16px); }}
    to   {{ opacity:1; transform:translateX(0); }}
}}
.trait-header {{
    display:flex;
    justify-content:space-between;
    align-items:center;
    margin-bottom:3px;
}}
.trait-label {{
    font-size:0.68rem;
    font-weight:700;
    text-transform:uppercase;
    letter-spacing:1.2px;
    color:rgba(255,255,255,0.35);
}}
.trait-pct {{
    font-size:0.72rem;
    font-weight:800;
}}
.trait-value {{
    font-size:0.88rem;
    font-weight:700;
    color:rgba(255,255,255,0.88);
    margin-bottom:6px;
    letter-spacing:-0.2px;
}}

/* ── Stat bar ── */
.stat-bar {{
    width:100%;
    height:4px;
    border-radius:4px;
    background:rgba(255,255,255,0.06);
    overflow:hidden;
}}
.stat-fill {{
    height:100%;
    border-radius:4px;
    width:0%;
    animation:fillBar 1s cubic-bezier(0.4,0,0.2,1) forwards;
    box-shadow:0 0 8px currentColor;
}}
@keyframes fillBar {{
    from {{ width:0%; }}
}}

/* ── Footer ── */
.profile-footer {{
    text-align:center;
    font-size:0.7rem;
    color:rgba(255,255,255,0.25);
    padding-top:12px;
    margin-top:auto;
    border-top:1px solid rgba(255,255,255,0.05);
    letter-spacing:0.3px;
}}
.profile-footer b {{
    color:#FF9F43;
    font-weight:700;
}}
</style></head><body>

<div class="profile-wrap">
    <!-- Photo column -->
    <div class="photo-col">
        <img src="{img_b64}" alt="High Income Profile" />
        <div class="photo-overlay"></div>
        <div class="photo-badge">
            <div class="photo-title">High-Income<br>Archetype</div>
            <div class="photo-sub">Data-driven composite</div>
            <div class="photo-tag">
                <span class="photo-tag-dot"></span>
                Top {high_pct}% earners
            </div>
        </div>
    </div>

    <!-- Stats column -->
    <div class="stats-col">
        {traits_html}
        <div class="profile-footer">{footer_text}</div>
    </div>
</div>

</body></html>"""


def _render_profile_section(arch: dict, lang: str) -> None:
    """Render the Typical High-Income Profile as a premium HTML component."""
    t = lambda key, **kw: get_text(key, lang, **kw)

    # Section header
    icon_svg = get_icon("target", size=18, color=_AMBER)
    st.markdown(
        f"<div style='padding:14px 18px;margin-top:4px;margin-bottom:20px;"
        f"background:linear-gradient(135deg,{_AMBER_DIM} 0%,rgba(255,159,67,0.02) 100%);"
        f"border:1px solid {_AMBER_BORDER};border-left:3px solid rgba(255,159,67,0.70);"
        f"border-radius:0 12px 12px 0;'>"
        f"<div style='display:flex;align-items:center;gap:10px;'>"
        f"{icon_svg}"
        f"<span style='font-size:1.08rem;font-weight:800;color:rgba(255,255,255,0.92);"
        f"letter-spacing:-0.3px;'>{t('conclusion_profile_title')}</span>"
        f"</div>"
        f"<div style='font-size:0.76rem;color:rgba(255,255,255,0.38);margin-top:4px;'>"
        f"{t('conclusion_profile_hint')}"
        f"</div>"
        f"</div>",
        unsafe_allow_html=True,
    )

    # Load image and build HTML
    img_b64 = _load_profile_image_b64()
    if not img_b64:
        styled_alert("Profile image not found.", "warning")
        return

    html_content = _build_profile_html(arch, img_b64, lang)
    components.html(html_content, height=680, scrolling=False)


# ==============================================================================
# SECTION 2 — KEY MESSAGE DELIVERY
# ==============================================================================

def _render_messages_section(lang: str) -> None:
    """Render the 3 Key Message insight cards."""
    t = lambda key, **kw: get_text(key, lang, **kw)

    # Section header — amber accent (consistent with profile section)
    icon_svg = get_icon("zap", size=18, color=_AMBER)
    st.markdown(
        f"<div style='padding:14px 18px;margin-top:8px;margin-bottom:20px;"
        f"background:linear-gradient(135deg,{_AMBER_DIM} 0%,rgba(255,159,67,0.02) 100%);"
        f"border:1px solid {_AMBER_BORDER};border-left:3px solid rgba(255,159,67,0.70);"
        f"border-radius:0 12px 12px 0;'>"
        f"<div style='display:flex;align-items:center;gap:10px;'>"
        f"{icon_svg}"
        f"<span style='font-size:1.08rem;font-weight:800;color:rgba(255,255,255,0.92);"
        f"letter-spacing:-0.3px;'>{t('conclusion_msg_title')}</span>"
        f"</div>"
        f"<div style='font-size:0.76rem;color:rgba(255,255,255,0.38);margin-top:4px;'>"
        f"{t('conclusion_msg_hint')}"
        f"</div>"
        f"</div>",
        unsafe_allow_html=True,
    )

    # Message card data — all amber accent for page-level consistency
    messages = [
        {
            "title_key": "conclusion_msg1_title",
            "body_key": "conclusion_msg1_body",
            "icon_key": "graduation_cap",
            "color": _AMBER,
            "bg": _AMBER_DIM,
            "css_class": "conclusion-msg-amber",
        },
        {
            "title_key": "conclusion_msg2_title",
            "body_key": "conclusion_msg2_body",
            "icon_key": "lightbulb",
            "color": _AMBER,
            "bg": _AMBER_DIM,
            "css_class": "conclusion-msg-amber",
        },
        {
            "title_key": "conclusion_msg3_title",
            "body_key": "conclusion_msg3_body",
            "icon_key": "trending_up",
            "color": _AMBER,
            "bg": _AMBER_DIM,
            "css_class": "conclusion-msg-amber",
        },
    ]

    cols = st.columns(3, gap="medium")
    for col, msg in zip(cols, messages):
        icon_html = get_icon(msg["icon_key"], size=22, color=msg["color"])
        with col:
            st.markdown(
                f'<div class="conclusion-msg-card {msg["css_class"]}">'
                f'<div class="conclusion-msg-icon" style="background:{msg["bg"]};">'
                f'{icon_html}'
                f'</div>'
                f'<div class="conclusion-msg-title">{t(msg["title_key"])}</div>'
                f'<div class="conclusion-msg-body">{t(msg["body_key"])}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )


# ==============================================================================
# MAIN
# ==============================================================================

def main() -> None:
    lang = st.session_state.get("lang", "en")

    page_header(
        title=get_text('conclusion_title', lang),
        subtitle=get_text('conclusion_subtitle', lang),
    )

    _ensure_workspace_active()
    active_file = st.session_state.get("active_file")
    workspace_status(active_file)

    df_raw = data_engine.load_and_standardize(
        active_file, _file_mtime=data_engine._get_file_mtime(active_file)
    )
    active_file_scan_progress_bar("_conclusion_done")
    st.markdown('<div style="height:20px;"></div>', unsafe_allow_html=True)

    if df_raw.empty:
        styled_alert(get_text('empty_state_msg', lang), "warning")
        return

    # ── Compute archetype ──────────────────────────────────────────────
    cache_key = f"_conclusion_arch_v8_{active_file}"
    size_key = f"_conclusion_size_v5_{active_file}"
    if (
        cache_key not in st.session_state
        or st.session_state.get(size_key) != len(df_raw)
    ):
        with st.spinner(get_text('conclusion_computing', lang)):
            st.session_state[cache_key] = _compute_archetype(df_raw)
            st.session_state[size_key] = len(df_raw)

    arch = st.session_state[cache_key]

    if not arch:
        styled_alert(get_text('conclusion_no_income', lang), "warning")
        return

    # ── Section 1: Typical High-Income Profile ─────────────────────────
    _render_profile_section(arch, lang)

    # ── Divider ────────────────────────────────────────────────────────
    section_divider()

    # ── Section 2: Key Message Delivery ────────────────────────────────
    _render_messages_section(lang)


if __name__ == "__main__":
    main()
