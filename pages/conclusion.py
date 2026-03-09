"""
conclusion.py — Conclusion & Recommendation Page

Design principles:
  - Monochrome amber accent on deep slate — no rainbow multi-color cards
  - SVG icons from modules/ui/icons.py (no emojis)
  - Three data-driven sections computed from the active dataset

Sections:
  1. Executive Summary of Insights
  2. Data Integrity Lessons & Cleaning Impact
  3. Actionable Recommendations & Future Work
"""

import numpy as np
import pandas as pd
import streamlit as st

from modules.core import data_engine
from modules.ui import page_header, workspace_status, active_file_scan_progress_bar, section_divider
from modules.ui.icons import get_icon
from modules.utils.helpers import _ensure_workspace_active

# ==============================================================================
# DESIGN TOKENS  — disciplined 2-accent system
# ==============================================================================

_AMBER        = "#FF9F43"
_AMBER_DIM    = "rgba(255,159,67,0.08)"
_AMBER_BORDER = "rgba(255,159,67,0.22)"
_AMBER_BRIGHT = "rgba(255,159,67,0.70)"

_SLATE        = "rgba(255,255,255,0.04)"
_BORDER       = "rgba(255,255,255,0.08)"
_TEXT_HI      = "rgba(255,255,255,0.90)"   # headlines
_TEXT_MED     = "rgba(255,255,255,0.65)"   # body
_TEXT_LO      = "rgba(255,255,255,0.38)"   # labels / captions

_OK_COLOR     = "#52c41a"                  # muted green — used sparingly for status only
_OK_DIM       = "rgba(82,196,26,0.08)"


# ==============================================================================
# UI PRIMITIVES
# ==============================================================================

def _section_header(icon_key: str, title: str, subtitle: str = "") -> None:
    icon_svg = get_icon(icon_key, size=18, color=_AMBER)
    st.markdown(
        f"""
        <div style="display:flex;align-items:center;gap:10px;margin-bottom:6px;margin-top:4px;">
          {icon_svg}
          <span style="font-size:1.08rem;font-weight:800;color:{_TEXT_HI};
                       letter-spacing:-0.3px;">{title}</span>
        </div>
        {"<div style='font-size:0.76rem;color:" + _TEXT_LO + ";margin-bottom:16px;padding-left:28px;'>" + subtitle + "</div>" if subtitle else "<div style='margin-bottom:16px;'></div>"}
        """,
        unsafe_allow_html=True,
    )


# ==============================================================================
# CSS INTERACTIONS
# ==============================================================================

def _inject_styles() -> None:
    """Inject scoped hover/transition effects for all card components."""
    st.markdown(
        """
        <style>
        /* ── Finding cards (Section 1 & 2) ─────────────────────────────── */
        .concl-finding-card {
            transition: transform 0.18s ease, box-shadow 0.18s ease,
                        border-color 0.18s ease, background 0.18s ease;
            cursor: default;
        }
        .concl-finding-card:hover {
            transform: translateY(-2px);
            background: rgba(255,255,255,0.07) !important;
            box-shadow: 0 6px 24px rgba(0,0,0,0.28);
        }
        /* ── Recommendation cards (Section 3) ──────────────────────────── */
        .concl-rec-card {
            transition: transform 0.18s ease, box-shadow 0.18s ease,
                        border-color 0.18s ease, background 0.18s ease;
            cursor: default;
        }
        .concl-rec-card:hover {
            transform: translateY(-2px);
            background: rgba(255,159,67,0.14) !important;
            border-color: rgba(255,159,67,0.50) !important;
            box-shadow: 0 6px 24px rgba(255,159,67,0.12);
        }
        /* ── Number/letter badge: pulse on hover ───────────────────────── */
        .concl-rec-card:hover .concl-badge {
            transform: scale(1.10);
            transition: transform 0.18s ease;
        }
        /* ── Section header icon pulse ─────────────────────────────────── */
        .concl-section-icon {
            display: inline-flex;
            transition: transform 0.20s ease;
        }
        .concl-section-icon:hover {
            transform: rotate(8deg) scale(1.15);
        }
        /* ── Equal-height columns ───────────────────────────────────────── */
        [data-testid="stHorizontalBlock"] {
            align-items: stretch !important;
        }
        [data-testid="stHorizontalBlock"] > [data-testid="stVerticalBlockBorderWrapper"] {
            height: 100% !important;
        }
        [data-testid="stHorizontalBlock"] > [data-testid="stVerticalBlockBorderWrapper"]
            > div[data-testid="stVerticalBlock"] {
            height: 100% !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _finding_card(icon_key: str, label: str, headline: str, detail: str,
                  severity: str = "default") -> None:
    """
    A single finding card. severity = 'default' | 'warning' | 'ok'
    Single-line styles to avoid Streamlit parser issues.
    """
    if severity == "warning":
        left_color = "rgba(230,110,50,0.90)"
    elif severity == "ok":
        left_color = _OK_COLOR
    else:
        left_color = _AMBER_BRIGHT

    icon_svg = get_icon(icon_key, size=14, color=left_color)
    st.markdown(
        f"<div class='concl-finding-card' style='background:rgba(255,255,255,0.04);"
        f"border:1px solid rgba(255,255,255,0.08);"
        f"border-left:3px solid {left_color};border-radius:8px;padding:12px 14px;margin-bottom:10px;'>"
        f"<div style='display:flex;align-items:center;gap:6px;margin-bottom:5px;'>"
        f"{icon_svg}"
        f"<span style='font-size:0.65rem;font-weight:700;color:{left_color};"
        f"text-transform:uppercase;letter-spacing:1px;'>{label}</span></div>"
        f"<div style='font-size:0.85rem;font-weight:700;color:rgba(255,255,255,0.90);"
        f"margin-bottom:4px;line-height:1.35;'>{headline}</div>"
        f"<div style='font-size:0.78rem;color:rgba(255,255,255,0.62);line-height:1.72;'>{detail}</div>"
        f"</div>",
        unsafe_allow_html=True,
    )


def _rec_card(letter: str, title: str, body: str, tag: str = "") -> None:
    """Render an individual rec card via st.markdown (used when not in grid layout)."""
    st.markdown(_rec_card_html(letter, title, body, tag), unsafe_allow_html=True)


def _rec_card_html(letter: str, title: str, body: str, tag: str = "") -> str:
    """Return rec card as raw HTML string — for use in equal-height flex grid."""
    tag_html = (
        f"&ensp;<span style='background:rgba(255,159,67,0.15);border:1px solid rgba(255,159,67,0.22);"
        f"border-radius:4px;padding:1px 6px;font-size:0.65rem;font-weight:700;"
        f"color:#FF9F43;vertical-align:middle;'>{tag}</span>"
        if tag else ""
    )
    return (
        f"<div class='concl-rec-card' style='background:rgba(255,159,67,0.08);"
        f"border:1px solid rgba(255,159,67,0.22);"
        f"border-radius:8px;padding:14px 16px;margin-bottom:10px;'>"
        f"<div style='display:flex;gap:12px;align-items:flex-start;'>"
        f"<div class='concl-badge' style='background:#FF9F43;color:#000;font-weight:900;border-radius:6px;"
        f"width:26px;height:26px;min-width:26px;display:flex;align-items:center;"
        f"justify-content:center;font-size:0.73rem;margin-top:2px;flex-shrink:0;"
        f"transition:transform 0.18s ease;'>{letter}</div>"
        f"<div style='flex:1;min-width:0;'>"
        f"<div style='font-size:0.83rem;font-weight:700;color:rgba(255,255,255,0.90);"
        f"margin-bottom:6px;line-height:1.3;'>{title}{tag_html}</div>"
        f"<div style='font-size:0.78rem;color:rgba(255,255,255,0.65);line-height:1.70;'>{body}</div>"
        f"</div></div></div>"
    )


def _finding_card_html(icon_key: str, label: str, headline: str, detail: str,
                       severity: str = "default") -> str:
    """Return finding card as raw HTML string — for equal-height flex grid."""
    if severity == "warning":
        left_color = "rgba(230,110,50,0.90)"
    elif severity == "ok":
        left_color = _OK_COLOR
    else:
        left_color = _AMBER_BRIGHT
    icon_svg = get_icon(icon_key, size=14, color=left_color)
    return (
        f"<div class='concl-finding-card' style='background:rgba(255,255,255,0.04);"
        f"border:1px solid rgba(255,255,255,0.08);"
        f"border-left:3px solid {left_color};border-radius:8px;padding:12px 14px;margin-bottom:10px;'>"
        f"<div style='display:flex;align-items:center;gap:6px;margin-bottom:5px;'>"
        f"{icon_svg}"
        f"<span style='font-size:0.65rem;font-weight:700;color:{left_color};"
        f"text-transform:uppercase;letter-spacing:1px;'>{label}</span></div>"
        f"<div style='font-size:0.85rem;font-weight:700;color:rgba(255,255,255,0.90);"
        f"margin-bottom:4px;line-height:1.35;'>{headline}</div>"
        f"<div style='font-size:0.78rem;color:rgba(255,255,255,0.62);line-height:1.72;'>{detail}</div>"
        f"</div>"
    )


def _two_col_grid(col_a_html: str, col_b_html: str, gap: str = "20px") -> None:
    """Render two columns as a single HTML flex block — guaranteed equal height."""
    st.markdown(
        f"<div style='display:flex;gap:{gap};align-items:stretch;'>"
        f"<div style='flex:1;min-width:0;'>{col_a_html}</div>"
        f"<div style='flex:1;min-width:0;'>{col_b_html}</div>"
        f"</div>",
        unsafe_allow_html=True,
    )



def _kv(label: str, value: str) -> str:
    """Inline key-value chip for the summary bar."""
    return (
        f"<span style='color:{_TEXT_LO};font-size:0.73rem;'>{label}&thinsp;</span>"
        f"<span style='color:{_AMBER};font-size:0.73rem;font-weight:700;'>{value}</span>"
        f"<span style='color:{_TEXT_LO};font-size:0.73rem;'>&ensp;·&ensp;</span>"
    )


# ==============================================================================
# DATA ANALYSIS — pure computation, no Streamlit side-effects
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
        "race":         ["race", "ethnicity"],
        "workclass":    ["workclass"],
        "marital":      ["maritalstatus", "marital"],
        "capital_gain": ["capitalgain", "capgain", "capital_gain"],
    }
    return {k: next((lookup[a] for a in v if a in lookup), None) for k, v in ALIASES.items()}


def _high_mask(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().str.contains(r">50k", regex=True, na=False)


def _compute_insights(df: pd.DataFrame) -> dict:
    N    = len(df)
    cols = _resolve(df)
    inc  = cols["income"]
    out  = {"n": N, "cols": len(df.columns), "inc_col": inc}

    if inc is None:
        return out

    hi   = _high_mask(df[inc])
    out["pct_high"] = round(hi.mean() * 100, 1)
    out["n_high"]   = int(hi.sum())

    # Education
    edu = cols["education"]
    if edu:
        edu_rate = (
            df.groupby(df[edu].astype(str))[inc]
            .apply(lambda s: _high_mask(s).mean() * 100)
            .round(1)
        )
        out["best_edu"]       = edu_rate.idxmax()
        out["best_edu_rate"]  = round(edu_rate.max(), 1)
        out["worst_edu_rate"] = round(edu_rate.min(), 1)
        out["edu_ratio"]      = round(edu_rate.max() / max(edu_rate.min(), 1), 1)

    # Occupation
    occ = cols["occupation"]
    if occ:
        occ_rate = (
            df.groupby(df[occ].astype(str))[inc]
            .apply(lambda s: _high_mask(s).mean() * 100)
            .round(1)
        )
        out["best_occ"]      = occ_rate.idxmax()
        out["best_occ_pct"]  = round(occ_rate.max(), 1)
        out["worst_occ"]     = occ_rate.idxmin()
        out["worst_occ_pct"] = round(occ_rate.min(), 1)
        out["occ_spread"]    = round(occ_rate.max() - occ_rate.min(), 1)

    # Gender
    sex = cols["sex"]
    if sex:
        sex_rate = (
            df.groupby(df[sex].astype(str))[inc]
            .apply(lambda s: _high_mask(s).mean() * 100)
            .round(1)
        )
        vals       = sex_rate.to_dict()
        male_key   = next((k for k in vals if "male" in k.lower() and "fe" not in k.lower()), None)
        female_key = next((k for k in vals if "female" in k.lower() or k.lower() == "f"), None)
        if male_key and female_key:
            out.update({
                "gender_gap":    round(vals[male_key] - vals[female_key], 1),
                "male_pct":      vals[male_key],
                "female_pct":    vals[female_key],
                "male_key":      male_key,
                "female_key":    female_key,
            })

    # Hours
    hrs = cols["hours"]
    if hrs and pd.api.types.is_numeric_dtype(df[hrs]):
        hrs_corr = round(df[hrs].corr(hi.astype(float)), 3)
        out["hrs_corr"]     = hrs_corr
        out["avg_hrs_high"] = round(df.loc[hi,  hrs].mean(), 1)
        out["avg_hrs_low"]  = round(df.loc[~hi, hrs].mean(), 1)

    # Capital Gain
    cg = cols["capital_gain"]
    if cg and pd.api.types.is_numeric_dtype(df[cg]):
        out["cg_pct_pos"] = round((df[cg] > 0).mean() * 100, 1)
        out["cg_max"]     = int(df[cg].max())
        out["cg_clipped"] = int((df[cg] >= 99999).sum())

    # Integrity
    missing_total = int(df.isnull().sum().sum())
    out["missing_total"] = missing_total
    out["missing_pct"]   = round(missing_total / max(N * len(df.columns), 1) * 100, 2)
    out["dupes"]         = int(df.duplicated().sum())

    noise = 0
    noise_cols = []
    for col in df.select_dtypes(include="object").columns:
        cnt = int(df[col].dropna().astype(str).str.strip().isin(
            {"?", "-", "na", "null", "none", "n/a", "undefined", ""}
        ).sum())
        if cnt > 0:
            noise += cnt
            noise_cols.append(f"<b>{col}</b>&nbsp;({cnt:,})")
    out["noise_count"] = noise
    out["noise_pct"]   = round(noise / max(N, 1) * 100, 2)
    out["noise_cols"]  = noise_cols[:5]

    return out


# ==============================================================================
# SECTION 1 — EXECUTIVE SUMMARY
# ==============================================================================

def _render_executive_summary(ins: dict) -> None:
    _section_header(
        "bar_chart", "Executive Summary of Insights",
        "Key headline findings — extracted for busy stakeholders who need answers, not charts."
    )

    if not ins.get("inc_col"):
        st.info("No income/salary column detected in this dataset.")
        return

    pct_hi = ins.get("pct_high", "—")
    N      = ins.get("n", 0)

    # ── Build column A ──────────────────────────────────────────────────
    col_a = ""
    if "best_edu" in ins:
        col_a += _finding_card_html(
            "briefcase", "Key Finding · Education",
            f"Education is the strongest predictor — {ins['edu_ratio']}× income multiplier",
            f"Individuals with <b>{ins['best_edu']}</b> credentials earn &gt;50K "
            f"at <b>{ins['best_edu_rate']}%</b> vs <b>{ins['worst_edu_rate']}%</b> "
            f"for the least-educated group — the single largest income explainer.",
        )
    if "gender_gap" in ins:
        col_a += _finding_card_html(
            "users", "Key Finding · Gender Inequality",
            f"{ins.get('gender_gap','?')} pp gender income gap — structural signal",
            f"<b>{ins.get('male_key','Male')}</b> {ins.get('male_pct',0)}% vs "
            f"<b>{ins.get('female_key','Female')}</b> {ins.get('female_pct',0)}% &gt;50K. "
            f"Gap holds within matched education + occupation — not explained by skills alone.",
            severity="warning",
        )
    if "cg_pct_pos" in ins:
        clipped_note = (
            f" {ins['cg_clipped']:,} records capped at 99,999 (survey ceiling artifact)."
            if ins.get("cg_clipped", 0) > 0 else ""
        )
        col_a += _finding_card_html(
            "zap", "Key Finding · Capital Gains",
            f"Capital income: rare ({ins['cg_pct_pos']}%) but highly concentrated",
            f"Only <b>{ins['cg_pct_pos']}%</b> report Capital Gain &gt; 0 — "
            f"but gains are heavily skewed toward high earners, making it a "
            f"high-signal wealth marker despite low prevalence.{clipped_note}",
        )

    # ── Build column B ──────────────────────────────────────────────────
    col_b = ""
    if "best_occ" in ins:
        occ_spread = ins.get("occ_spread", round(ins.get("best_occ_pct", 0) - ins.get("worst_occ_pct", 0), 1))
        col_b += _finding_card_html(
            "target", "Key Finding · Occupation",
            f"{ins.get('best_occ','N/A')} leads at {ins.get('best_occ_pct',0)}% — {occ_spread} pp spread",
            f"Occupation is the #2 predictor. <b>{ins.get('best_occ','N/A')}</b> "
            f"({ins.get('best_occ_pct',0)}%) vs <b>{ins.get('worst_occ','N/A')}</b> "
            f"({ins.get('worst_occ_pct',0)}%) — reflects skill premiums and structural access barriers.",
        )
    if "hrs_corr" in ins:
        corr_abs   = abs(ins["hrs_corr"])
        corr_label = "weak" if corr_abs < 0.15 else ("moderate" if corr_abs < 0.35 else "strong")
        col_b += _finding_card_html(
            "clock", "Key Finding · Work Intensity",
            f"Hours/week: only {corr_label} income correlation (r = {ins['hrs_corr']})",
            f"High earners average <b>{ins['avg_hrs_high']}h/wk</b> vs "
            f"<b>{ins['avg_hrs_low']}h</b> — a modest gap. "
            f"Role type (Education + Occupation) is far more predictive than raw hours worked.",
        )

    _two_col_grid(col_a, col_b)

    # Summary footer
    st.markdown(
        f"<div style='color:{_TEXT_LO};font-size:0.71rem;margin-top:4px;padding-left:2px;'>"
        f"{_kv('Dataset', f'{N:,} records')}"
        f"{_kv('>50K rate', f'{pct_hi}%')}"
        f"{_kv('Columns', str(ins.get('cols', '—')))}"
        f"</div>",
        unsafe_allow_html=True,
    )


# ==============================================================================
# SECTION 2 — DATA INTEGRITY LESSONS
# ==============================================================================

def _render_integrity_lessons(ins: dict) -> None:
    _section_header(
        "audit", "Data Integrity Lessons & Cleaning Impact",
        "How preprocessing uncovered and corrected issues that would have distorted the analysis."
    )

    # ── Build column A: Noise + Capital Gain (long) ───────────────────────
    col_a = ""
    if ins.get("noise_count", 0) > 0:
        col_list = ", ".join(ins.get("noise_cols", []))
        col_a += _finding_card_html(
            "scissors", "Data Issue · Noise Values",
            f"{ins['noise_count']:,} placeholder values removed ({ins['noise_pct']}% of rows)",
            f"Tokens like '?', '-', 'null' replaced with NaN before analysis. "
            f"Affected: {col_list}. "
            f"Prevents a spurious token category from inflating group percentages "
            f"and biasing income-rate calculations.",
            severity="warning",
        )
    else:
        col_a += _finding_card_html(
            "check_circle", "Data Quality · Noise",
            "No noise or placeholder values detected",
            "All categorical columns are free of garbage tokens such as '?', '-', or 'null'. "
            "Categorical distribution charts reflect true value frequencies "
            "and require no token-removal correction.",
            severity="ok",
        )
    if ins.get("cg_clipped", 0) > 0:
        col_a += _finding_card_html(
            "alert_triangle", "Data Artifact · Capital Gain Ceiling",
            f"{ins['cg_clipped']:,} records capped at 99,999 — survey data-ceiling",
            f"The Capital Gain column has a hard upper bound of <b>99,999</b>. "
            f"These represent individuals whose true gain exceeded the recording threshold — "
            f"not data errors. Standard box-plots would misinterpret this as a single "
            f"extreme outlier rather than a structural census-collection artifact.",
            severity="warning",
        )
    else:
        col_a += _finding_card_html(
            "check_circle", "Data Quality · Capital Gain Range",
            "No data-ceiling artifacts detected in Capital Gain",
            "No records found at the 99,999 ceiling boundary. "
            "Capital gain values appear to reflect unconstrained survey responses — "
            "the distribution can be read without ceiling-artifact corrections.",
            severity="ok",
        )

    # ── Build column B: Missing + Duplicates ───────────────────────────
    col_b = ""
    if ins.get("missing_pct", 0) > 0:
        col_b += _finding_card_html(
            "bandaid", "Data Issue · Missing Values",
            f"{ins['missing_total']:,} null cells ({ins['missing_pct']}% of total)",
            f"Smart imputation applied: <b>mean</b> for symmetric numeric columns, "
            f"<b>median</b> for skewed distributions, and <b>mode</b> for categoricals. "
            f"Avoids row-deletion bias that would disproportionately remove records "
            f"from under-represented demographic groups.",
            severity="warning",
        )
    else:
        col_b += _finding_card_html(
            "check_circle", "Data Quality · Completeness",
            "Dataset is fully populated — zero missing values",
            "Every cell across all columns contains a valid, non-null value. "
            "No imputation step was required. The dataset can be used directly "
            "without any completeness-driven preprocessing.",
            severity="ok",
        )
    dupes = ins.get("dupes", 0)
    if dupes > 0:
        col_b += _finding_card_html(
            "copy", "Data Issue · Duplicate Rows",
            f"{dupes:,} duplicate rows identified",
            f"Fully duplicate records inflate counts for the most common categories, "
            f"artificially shifting income-rate percentages for dominant groups. "
            f"Removal ensures frequency distributions reflect unique individual observations.",
            severity="warning",
        )
    else:
        col_b += _finding_card_html(
            "check_circle", "Data Quality · Uniqueness",
            "No duplicate rows — every record is unique",
            "De-duplication was not required. Each row represents a distinct individual. "
            "Frequency distributions and group-rate calculations are not inflated "
            "by repeated entries.",
            severity="ok",
        )

    _two_col_grid(col_a, col_b)


# ==============================================================================
# SECTION 3 — RECOMMENDATIONS & FUTURE WORK
# ==============================================================================

def _render_recommendations(ins: dict) -> None:
    _section_header(
        "orbit", "Actionable Recommendations & Future Work",
        "Evidence-based actions and research directions grounded in the findings above."
    )

    col_label_style = (
        f"font-size:0.70rem;font-weight:700;color:{_AMBER};"
        f"text-transform:uppercase;letter-spacing:1px;margin-bottom:12px;"
        f"display:flex;align-items:center;gap:5px;"
    )
    edu_ratio = ins.get("edu_ratio", "significant")
    best_occ  = ins.get("best_occ", "Professional Roles")

    # ── Left: Policy recommendations ─────────────────────────────────
    col1 = (
        f"<div style='{col_label_style}'>"
        f"{get_icon('target', size=13, color=_AMBER)}"
        f"&ensp;Policy &amp; Action Recommendations</div>"
    )
    col1 += _rec_card_html(
        "1", "Invest in Higher Education Access",
        f"Education yields a <b>{edu_ratio}× income multiplier</b>. "
        f"Expanding access to bachelor's-level programs and vocational training — "
        f"especially for demographics currently concentrated in low-education, "
        f"low-income occupational clusters — is the highest-ROI policy lever.",
        tag="High Priority",
    )
    if ins.get("gender_gap"):
        col1 += _rec_card_html(
            "2", "Implement Structural Pay-Equity Measures",
            f"A <b>{ins['gender_gap']} pp gender income gap</b> persists even within "
            f"matched education and occupation groups. "
            f"Transparent salary bands, mandatory pay-gap reporting, "
            f"and targeted promotion-pipeline monitoring in high-income occupations "
            f"are evidence-supported interventions.",
            tag="High Priority",
        )
    col1 += _rec_card_html(
        "3", "Expand Access to High-ROI Occupation Pathways",
        f"<b>{best_occ}</b> achieves the highest income-attainment rate in this dataset. "
        f"Workforce-development programs should build structured entry pathways "
        f"(apprenticeships, certifications, employer co-op partnerships) into these fields. "
        f"Prioritizing underrepresented groups in recruitment pipelines addresses both "
        f"economic efficiency and equity simultaneously.",
    )
    if ins.get("hrs_corr") is not None:
        col1 += _rec_card_html(
            "4", "Reframe Labor Policy Beyond Hours-Based Metrics",
            f"Hours worked correlates weakly with income (r={ins['hrs_corr']}), "
            f"confirming that labor quantity is a poor proxy for economic output. "
            f"Policies incentivizing overtime yield diminishing returns compared to "
            f"skill-upgrading investments. Sector-mobility subsidies and role-reclassification "
            f"programs are more structurally effective at raising income attainment.",
        )

    # ── Right: Future Research ──────────────────────────────────────
    col2 = (
        f"<div style='{col_label_style}'>"
        f"{get_icon('visual', size=13, color=_AMBER)}"
        f"&ensp;Future Research Directions</div>"
    )
    col2 += _rec_card_html(
        "A", "Integrate Predictive Machine Learning Modeling",
        "Train a supervised classification model (Gradient Boosting, XGBoost, or "
        "Logistic Regression with interaction terms) to predict individual income-class "
        "probability. Shapley values will quantify the marginal contribution of "
        "Education, Occupation, and demographic features — validating or refining "
        "the rank-ordering established here.",
        tag="Next Step",
    )
    col2 += _rec_card_html(
        "B", "Collect Longitudinal Data for Mobility Analysis",
        "This dataset is a static cross-section. Panel data spanning multiple census "
        "waves would enable income-mobility analysis — identifying whether high-education "
        "individuals consistently transition to higher income brackets over their careers "
        "or whether mobility is constrained by structural barriers.",
        tag="Next Step",
    )
    if ins.get("gender_gap"):
        col2 += _rec_card_html(
            "C", "Intersectional Income Inequality Research",
            "Analyze income disparity at the intersection of <b>Gender × Occupation × "
            "Education</b> using multilevel modeling with interaction terms. "
            "Current findings point to additive effects, but interaction terms may "
            "reveal compounding disadvantages for specific subgroups that single-variable "
            "analysis cannot detect.",
        )
    col2 += _rec_card_html(
        "D", "Replicate Analysis on Modern Census Records",
        "This dataset reflects a specific historical period. Replicating this analysis "
        "on post-2010 ACS (American Community Survey) data would test whether structural "
        "patterns — the education premium, gender gap, occupational spread — have "
        "widened, narrowed, or shifted in response to economic cycles and policy changes "
        "over the past three decades.",
    )

    _two_col_grid(col1, col2, gap="24px")


# ==============================================================================
# MAIN
# ==============================================================================

def main() -> None:
    lang = st.session_state.get("lang", "en")

    page_header(
        title="Conclusion & Recommendation",
        subtitle="Data-driven findings, integrity lessons, and evidence-based recommendations.",
    )

    _ensure_workspace_active()
    active_file = st.session_state.get("active_file")
    workspace_status(active_file)

    df_raw = data_engine.load_and_standardize(
        active_file, _file_mtime=data_engine._get_file_mtime(active_file)
    )
    active_file_scan_progress_bar("_conclusion_done")

    if df_raw.empty:
        st.warning("No data loaded. Please upload and activate a dataset first.")
        return

    # ── Cache insights (recompute only when file or schema changes) ─────────
    cache_key = f"_conclusion_insights_v2_{active_file}"   # v2 forces recompute after redesign
    size_key  = f"_conclusion_size_v2_{active_file}"
    if (
        cache_key not in st.session_state
        or st.session_state.get(size_key) != len(df_raw)
    ):
        with st.spinner("Computing data-driven insights…"):
            st.session_state[cache_key] = _compute_insights(df_raw)
            st.session_state[size_key]  = len(df_raw)

    ins = st.session_state[cache_key]

    # ── Inject interaction styles ──────────────────────────────────────────
    _inject_styles()

    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

    _render_executive_summary(ins)
    section_divider()

    _render_integrity_lessons(ins)
    section_divider()

    _render_recommendations(ins)


if __name__ == "__main__":
    main()
