"""

audit_engine.py — Vectorized Data Audit Engine
Responsibilities:
    • Health score calculation via Data Quality Matrix
    • Outlier detection (IQR / Z-score / Modified Z-score)
    • Field-level integrity reporting
    • Noise value detection in categorical columns
    • Consistency checking (casing, whitespace, singletons, units, dates)
    • Safe-zone validation driven by admin DB rules
    • Full audit orchestration (single-pass, no redundant computation)
Design notes:
    - All public functions are pure (no Streamlit side-effects) except
      those that read `session_state` config (prefixed with config-getter helpers).
    - Heavy loops are vectorized with NumPy/pandas wherever possible.
    - `run_full_audit` is the single entry-point for the UI; results are
      returned as a dict and cached by the caller.
"""

import numpy as np
import pandas as pd
import streamlit as st
from typing import Dict, Any, List, Tuple
from modules.utils.localization import get_text

# =============================================================================
# CONSTANTS
# =============================================================================

# Placeholder / corrupted values treated as noise.
# Loaded from DB admin rules at runtime; this set is the hard-coded fallback.
_FALLBACK_NOISE_PATTERNS = frozenset({
    "?", "-", "--", "---", ".", "..", "...",
    "n/a", "na", "null", "none", "undefined", "#n/a",
    "missing", "unknown", "not available", "#value!",
    "#ref!", "#div/0!", "inf", "-inf", "nan",
})
# Consistency factor that converts MAD → equivalent Std for a normal distribution.
# Used by the Modified Z-score outlier detector.
_MAD_SCALE = 0.6745


def _get_noise_patterns() -> frozenset:
    """
    Return the active noise-pattern set.
    Prefers admin-configured patterns from ``session_state['analysis_rules']
    ['noise_patterns']``; falls back to ``_FALLBACK_NOISE_PATTERNS`` when
    no rules have been loaded yet (e.g. first run before DB sync).
    """
    rules = st.session_state.get("analysis_rules", {})
    custom = rules.get("noise_patterns")
    if custom and isinstance(custom, list):
        return frozenset(p.lower() for p in custom)
    return _FALLBACK_NOISE_PATTERNS


def _get_safe_zones() -> Dict[str, Dict[str, float]]:
    """
    Return safe-zone bounds dict from admin DB rules.
    Each key is a column name; each value is a dict with optional
    ``'min'`` and/or ``'max'`` float bounds.
    Returns an empty dict when no safe zones have been configured.
    """
    rules = st.session_state.get("analysis_rules", {})
    return rules.get("safe_zones", {})

# =============================================================================
# HELPERS (private)
# =============================================================================

def _is_categorical(s: pd.Series) -> bool:
    """Return True if *s* has object or category dtype (i.e. string-like column)."""
    return pd.api.types.is_object_dtype(s) or isinstance(s.dtype, pd.CategoricalDtype)


def _get_cat_columns(df: pd.DataFrame) -> pd.Index:
    """Return the Index of all object/category columns in *df*."""
    return df.select_dtypes(include=["object", "category"]).columns


def _get_num_columns(df: pd.DataFrame) -> pd.Index:
    """Return the Index of all numeric columns in *df*."""
    return df.select_dtypes(include=["number"]).columns


def _normalize_col_key(name: str) -> str:
    """Normalize a column name for safe-zone lookup: lowercase, underscored, trimmed."""
    return name.strip().lower().replace(" ", "_")


def _build_norm_zones() -> Dict[str, dict]:
    """
    Build normalised safe-zone lookup from session_state rules.
    Returns:
        Dict mapping ``{normalised_col_key: {min: ..., max: ...}}``.
    """
    return {
        _normalize_col_key(zk): zv
        for zk, zv in _get_safe_zones().items()
    }


def _check_safe_zone_violations(series: pd.Series, zone: dict) -> int:
    """
    Count values in a numeric *series* that fall outside the given *zone* bounds.
    Args:
        series: Numeric pandas Series (may contain NaNs, which are excluded).
        zone:   Dict with optional ``'min'`` and/or ``'max'`` keys.
                ``None`` means no safe zone configured.
    Returns:
        Number of out-of-range values.  Returns 0 when *zone* is None.
    """

    if zone is None:
        return 0
    lo = zone.get("min")
    hi = zone.get("max")
    lo_bound = lo if lo is not None else -np.inf
    hi_bound = hi if hi is not None else np.inf
    return int((~series.dropna().between(lo_bound, hi_bound)).sum())


def _compute_noise_mask(series: pd.Series) -> pd.Series:
    """
    Build a boolean mask that flags noisy/placeholder values in a string Series.
    A value is flagged if it matches ANY of:
      1. Exact match against the active noise-pattern set (case-insensitive).
      2. Empty string after stripping whitespace.
      3. Single non-alphanumeric character (e.g. '?', '.', '-').
    Args:
        series: A Series of strings (NaNs should already be dropped by the caller).
    Returns:
        Boolean Series aligned to *series*.
    """
    patterns = _get_noise_patterns()
    stripped = series.str.strip()
    return (
        stripped.str.lower().isin(patterns)          # known placeholder tokens
        | (stripped == "")                            # blank / whitespace-only
        | ((series.str.len() == 1) & (~series.str.isalnum()))  # lone special char
    )


def _auto_cast_category(df: pd.DataFrame) -> pd.DataFrame:
    """
    Memory-saving dtype downcast applied before any audit computation.
    Performs two passes on a *copy* of the DataFrame:
      1. Numeric downcast — int64 → smallest int type, float64 → float32.
         Cuts memory usage significantly for large integer columns.
      2. Object → category — columns whose cardinality is < 5 % of row count
         are converted to ``pd.CategoricalDtype``, reducing RAM and
         accelerating groupby / value_counts operations.
    Note:
        Always operates on a copy — the caller’s DataFrame is never mutated.
    """
    if df.empty:
        return df
    df = df.copy()  # never mutate the caller's DataFrame

    # --- Pass 1: downcast numeric columns ---
    for col in df.select_dtypes(include=["number"]).columns:
        if pd.api.types.is_integer_dtype(df[col]):
            df[col] = pd.to_numeric(df[col], downcast="integer")
        elif pd.api.types.is_float_dtype(df[col]):
            df[col] = pd.to_numeric(df[col], downcast="float")

    # --- Pass 2: convert low-cardinality object columns to category ---
    # Threshold: unique values < 5 % of total rows (heuristic for categorical data)
    threshold = 0.05
    n_rows = len(df)
    if n_rows > 0:
        for col in df.select_dtypes(include=["object"]).columns:
            if df[col].nunique() / n_rows < threshold:
                df[col] = df[col].astype("category")

    return df

# =============================================================================
# NOISE VALUE DETECTION
# =============================================================================

def detect_noise_values(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    """
    Scan all categorical columns for noisy / placeholder values.
    A value is considered noisy if it matches the active noise-pattern set
    (see ``_compute_noise_mask`` for the exact rules).
    Returns:
        Tuple of:
          - noise_df: DataFrame [Column, Noise Count, Examples] (one row per
            affected column). Empty DataFrame with correct schema when clean.
          - noise_total: sum of all noise cells across all columns.
    """
    records = []
    total = 0
    for col in _get_cat_columns(df):
        series = df[col].dropna().astype(str)
        mask = _compute_noise_mask(series)
        count = int(mask.sum())
        if count > 0:
            examples = series[mask].unique()[:5].tolist()
            records.append({
                "Column": col,
                "Noise Count": count,
                "Examples": ", ".join(examples),
            })
            total += count
    noise_df = pd.DataFrame(records) if records else pd.DataFrame(columns=["Column", "Noise Count", "Examples"])
    return noise_df, total

# =============================================================================
# RISK RECORDS (Outlier Inspector)
# =============================================================================

def _outlier_mask_iqr(arr: np.ndarray, threshold: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute outlier mask and severity score for the IQR method.
    Score = distance beyond the fence expressed in IQR units
    (0 for non-outliers, >0 for outliers).
    """
    q1, q3 = np.percentile(arr, [25, 75])
    iqr = q3 - q1
    if iqr == 0:
        return np.zeros(len(arr), dtype=bool), np.zeros(len(arr))
    lower, upper = q1 - threshold * iqr, q3 + threshold * iqr
    mask = (arr < lower) | (arr > upper)
    score = np.where(arr < lower, (lower - arr) / iqr, np.where(arr > upper, (arr - upper) / iqr, 0.0))
    return mask, np.round(score, 2)


def _outlier_mask_zscore(arr: np.ndarray, threshold: float) -> Tuple[np.ndarray, np.ndarray]:
    """Compute outlier mask and |z|-score for the classical Z-score method."""
    std = np.std(arr, ddof=1)
    if std == 0:
        return np.zeros(len(arr), dtype=bool), np.zeros(len(arr))
    z = np.abs((arr - np.mean(arr)) / std)
    return z > threshold, np.round(z, 2)


def _outlier_mask_modified_zscore(arr: np.ndarray, threshold: float) -> Tuple[np.ndarray, np.ndarray]:
    """Compute outlier mask and Modified Z-score (MAD-based) for each element."""
    median = np.median(arr)
    mad = np.median(np.abs(arr - median))
    if mad == 0:
        return np.zeros(len(arr), dtype=bool), np.zeros(len(arr))
    z = np.abs(_MAD_SCALE * (arr - median) / mad)
    return z > threshold, np.round(z, 2)

# ---------------------------------------------------------------------------
# Dispatch table for outlier-mask functions.
# To add a new method: implement _outlier_mask_<name> and add one line here.
# ---------------------------------------------------------------------------
_OUTLIER_METHODS = {
    "iqr":             _outlier_mask_iqr,
    "zscore":          _outlier_mask_zscore,
    "modified_zscore": _outlier_mask_modified_zscore,
}


def default_outlier_threshold(method_key: str) -> float:
    """Return the canonical default threshold for a given outlier detection method.

    **Single source of truth** — every module that needs a method-aware
    threshold MUST call this function instead of inlining the logic.

    Args:
        method_key: One of ``'iqr'``, ``'zscore'``, ``'modified_zscore'``.

    Returns:
        ``1.5`` for IQR (fence multiplier), ``3.0`` for Z-score variants
        (standard-deviation count).
    """
    return 1.5 if method_key == "iqr" else 3.0


def get_risk_records(
    df: pd.DataFrame,
    column: str,
    method: str = "iqr",
    threshold: float = None,
    max_rows: int = 200,
) -> Tuple[pd.DataFrame, int]:
    """
    Return the rows flagged as outliers for *column*, sorted by anomaly severity.
    Args:
        df:        Source DataFrame.
        column:    Name of the numeric column to inspect.
        method:    One of ``'iqr'``, ``'zscore'``, ``'modified_zscore'``.
        threshold: Detection threshold.  Defaults via ``default_outlier_threshold``;
                   1.5 for IQR and 3.0 for Z-score variants.
        max_rows:  Cap on returned rows to avoid overwhelming the UI.
    Returns:
        Tuple ``(flagged_df, total_outlier_count)``:
          - ``flagged_df``   — filtered, severity-sorted DataFrame (up to *max_rows*).
            Empty DataFrame when no outliers are found.
          - ``total_outlier_count`` — total number of true outliers (before the
            *max_rows* cap is applied).
    """
    _EMPTY = (pd.DataFrame(), 0)  # canonical empty return value
    if column not in df.columns:
        return _EMPTY
    vals = df[column].dropna()
    if len(vals) < 3:
        return _EMPTY
    if threshold is None:
        threshold = default_outlier_threshold(method)
    compute_fn = _OUTLIER_METHODS.get(method)
    if compute_fn is None:
        return _EMPTY

    # --- Step 1: Statistical outlier detection ---
    mask, scores = compute_fn(vals.values, threshold)
    if not mask.any():
        return _EMPTY
    total_outliers = int(mask.sum())

    # --- Step 2: Build result DataFrame, sorted by severity score (descending) ---
    flagged = df.loc[vals.index[mask]].copy()
    flagged["_score"] = scores[mask]
    flagged = flagged.sort_values("_score", ascending=False).drop(columns="_score")

    return flagged.head(max_rows), total_outliers


def compute_skewness(series: pd.Series) -> float | None:
    """Compute skewness of a numeric series — **single source of truth**.

    Guards:
      - Returns ``None`` when fewer than 3 non-null values exist.
      - Returns ``0.0`` when the standard deviation is zero (constant column).

    All other modules **MUST** call this instead of computing ``.skew()``
    inline to guarantee consistent precision (3 decimal places) and
    identical guard logic across the entire application.

    Args:
        series: Numeric pandas Series (may contain NaNs — they are dropped).

    Returns:
        Skewness rounded to 3 decimal places, or ``None``.
    """
    clean = series.dropna()
    if len(clean) < 3:
        return None
    if clean.std() == 0:
        return 0.0
    return round(float(clean.skew()), 3)


def recommend_fill_strategy(series: pd.Series) -> str:
    """Recommend the best missing-value fill strategy for a column.

    **Single source of truth** for the skewness-based threshold that
    determines whether to fill with ``mean`` or ``median``.

    Rules:
      - Non-numeric (object / category) → ``"mode"``.
      - Numeric with ``|skewness| < 0.5`` (near-symmetrical) → ``"mean"``.
      - Numeric otherwise (skewed or insufficient data) → ``"median"``.

    Args:
        series: A pandas Series (may contain NaNs).

    Returns:
        One of ``"mean"``, ``"median"``, ``"mode"``.
    """
    if not pd.api.types.is_numeric_dtype(series):
        return "mode"
    skew = compute_skewness(series)
    if skew is not None and abs(skew) < 0.5:
        return "mean"
    return "median"


def evaluate_outlier_method(s: pd.Series, lang: str = "en") -> Dict[str, str]:
    """
    Evaluate the skewness of a numeric column and recommend an outlier
    detection method.
    Returns a dict with:
      - 'method': "Z-Score" | "IQR" | "Modified Z-Score"
      - 'reason': Localized explanation string
      - 'skewness': float (rounded to 3 decimals)
    """
    skew = compute_skewness(s)
    if skew is None:
        return {"method": "IQR", "reason": get_text("rec_iqr", lang), "skewness": 0.0}
    abs_skew = abs(skew)
    if abs_skew < 0.5:
        return {"method": "Z-Score", "reason": get_text("rec_zscore", lang), "skewness": skew}
    elif abs_skew <= 1.0:
        return {"method": "IQR", "reason": get_text("rec_iqr", lang), "skewness": skew}
    else:
        return {"method": "Modified Z-Score", "reason": get_text("rec_mod_z", lang), "skewness": skew}

# =============================================================================
# COLUMN REPORT
# =============================================================================

def generate_column_report(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate a compact per-column summary: dtype, missing count/percentage,
    and unique value count.  Used in the Overview preview panel.
    """
    if df.empty:
        return pd.DataFrame()

    return pd.DataFrame({
        "Column Name": df.columns,
        "Type": df.dtypes.astype(str),
        "Missing": df.isnull().sum(),
        "Missing (%)": (df.isnull().mean() * 100).round(1).astype(str) + "%",
        "Unique Values": df.nunique(),
    }).reset_index(drop=True)

# =============================================================================
# DATA SUMMARY (Statistical Overview)
# =============================================================================

def compute_data_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a concise per-column statistical summary table.
    Columns produced:
        Column, Type, Records, Missing %, Unique Values,
        Distribution, Central Value, Outliers
    Rules:
        * **Unique Values** — ``nunique``; if noise tokens exist in a
          categorical column the label is suffixed with ``" (*)"``.
        * **Distribution** — skewness label for numeric columns:
          ``|skew| < 0.5`` → Approximately Symmetric,
          ``skew ≥ 0.5``  → Right Skewed,
          ``skew ≤ -0.5`` → Left Skewed.
          Categorical columns show ``"—"``.
        * **Central Value** — categorical: Mode.
          Numeric *without* outliers: ``"Mean: X  |  Std: Y"``.
          Numeric *with* outliers: ``"Median: X  |  Std: Y"``.
        * **Outliers** — IQR-based classification:
          ``None`` / ``Moderate Outliers`` / ``Extreme Outliers``.
    Returns:
        DataFrame sorted by Missing % descending, then column name.
    """
    if df.empty:

        return pd.DataFrame()
    total_rows = len(df)
    records: List[dict] = []
    for col in df.columns:
        series = df[col]
        non_null_count = int(series.notna().sum())
        missing_count = total_rows - non_null_count
        missing_pct = round(missing_count / total_rows * 100, 2) if total_rows > 0 else 0.0
        unique_count = int(series.nunique())

        # --- Noise count (categorical columns only) ---
        noise_count = 0
        if _is_categorical(series):
            noise_count = int(_compute_noise_mask(series.dropna().astype(str)).sum())

        # --- Distribution & Central Value & Outliers ---
        distribution = "—"
        central_value = "—"
        outlier_label = "—"
        if pd.api.types.is_numeric_dtype(series):
            clean_vals = series.dropna()
            # Distribution (skewness classification)
            skew_val = compute_skewness(clean_vals)
            if skew_val is not None:
                if abs(skew_val) < 0.5:
                    distribution = f"Approximately Symmetric ({skew_val})"
                elif skew_val > 0:
                    distribution = f"Right Skewed ({skew_val})"
                else:
                    distribution = f"Left Skewed ({skew_val})"
            # Outliers — use centralised _OUTLIER_METHODS dispatch (same as Inspector)
            has_outliers = False
            if len(clean_vals) > 2:
                # Auto-select method based on skewness
                rec = evaluate_outlier_method(series)
                method = rec["method"]
                # Map display name → dispatch key
                method_key = {"IQR": "iqr", "Z-Score": "zscore", "Modified Z-Score": "modified_zscore"}[method]
                # Method-aware default thresholds (single source of truth)
                default_threshold = default_outlier_threshold(method_key)
                compute_fn = _OUTLIER_METHODS[method_key]
                mask, _ = compute_fn(clean_vals.values, default_threshold)
                n_outliers = int(mask.sum())
                if n_outliers > 0:
                    has_outliers = True
                    # Classify severity via IQR fences regardless of detection method
                    q1 = float(clean_vals.quantile(0.25))
                    q3 = float(clean_vals.quantile(0.75))
                    iqr = q3 - q1
                    extreme_mask = (
                        (clean_vals < (q1 - 3.0 * iqr)) | (clean_vals > (q3 + 3.0 * iqr))
                    )

                    severity = "Extreme" if int(extreme_mask.sum()) > 0 else "Moderate"
                    outlier_label = f"{severity} ({method})  |  {n_outliers:,}"
                else:
                    outlier_label = f"None ({method})"
            # Central Value — mean vs median depends on outlier presence
            if len(clean_vals) > 0:
                if has_outliers:
                    median_val = round(float(clean_vals.median()), 2)
                    central_value = f"{median_val:,} (Median)"
                else:
                    mean_val = round(float(clean_vals.mean()), 2)
                    central_value = f"{mean_val:,} (Mean)"
        else:
            # Categorical — Central Value = Mode
            value_counts = series.dropna().astype(str).value_counts()
            if not value_counts.empty:
                central_value = str(value_counts.index[0])
        records.append({
            "Column": col,
            "Type": str(series.dtype),
            "Records": non_null_count,
            "Missing %": missing_pct,
            "Unique Values": unique_count,
            "Noise": noise_count if (_is_categorical(series) and noise_count > 0) else "—",
            "Distribution": distribution,
            "Central Value": central_value,
            "Outliers": outlier_label,
        })
    result_df = pd.DataFrame(records)

    return result_df.sort_values(
        ["Missing %", "Column"], ascending=[False, True]
    ).reset_index(drop=True)

# =============================================================================
# CONSISTENCY CHECK
# =============================================================================

def check_consistency(df: pd.DataFrame, lang: str = "en") -> pd.DataFrame:
    """
    Scan all categorical columns for common data-quality inconsistencies.
    Checks performed per column:
      1. Mixed Casing       — same token in different cases ('Male' vs 'male')
      2. Leading/Trailing Whitespace — values that differ only by surrounding spaces
      3. Rare Singletons    — values that appear exactly once (likely typos)
      4. Mixed Units        — numeric-with-unit strings using inconsistent units
      5. Date Format Issues — date-like column names with un-parseable values
    Note:
        Noise detection is intentionally excluded here (handled by
        ``detect_noise_values`` to avoid double-counting in issue_composition).
    Returns:
        DataFrame with columns [Column, Issue, Detail, Count].
        Empty DataFrame (with correct schema) when no issues are found.
    """

    issues = []
    for col in _get_cat_columns(df):
        vals = df[col].dropna()
        if vals.empty:
            continue
        str_vals = vals.astype(str)
        unique_raw = str_vals.unique()
        if len(unique_raw) == 0:
            continue
        # Compute value_counts ONCE per column — reused by all checks below
        vc: pd.Series = str_vals.value_counts()

        # ------------------------------------------------------------------
        # 1+2. Mixed Casing & Leading/Trailing Whitespace — single pass
        #    Build both maps in one loop over unique_raw (O(U) instead of 2×O(U)).
        # ------------------------------------------------------------------

        lower_map: Dict[str, list] = {}  # lower-stripped key → list of raw variants
        ws_map:    Dict[str, list] = {}  # stripped key        → list of raw variants
        for v in unique_raw:
            lower_map.setdefault(v.strip().lower(), []).append(v)
            ws_map.setdefault(v.strip(), []).append(v)
        # -- 1. Mixed Casing --
        for variants in lower_map.values():
            if len(variants) > 1:
                variant_counts = {v: int(vc.get(v, 0)) for v in variants}
                majority_variant = max(variant_counts, key=variant_counts.get)
                minority_variants = [v for v in variants if v != majority_variant]
                minority_cells = sum(variant_counts[v] for v in minority_variants)
                if minority_cells > 0:
                    issues.append({
                        "Column": col, "Issue": get_text("issue_mixed_casing", lang),
                        "Detail": " | ".join(sorted(minority_variants)), "Count": minority_cells,
                    })
        # -- 2. Leading / Trailing Whitespace --
        for variants in ws_map.values():
            if len(variants) > 1:
                variant_counts = {v: int(vc.get(v, 0)) for v in variants}
                majority_variant = max(variant_counts, key=variant_counts.get)
                minority_variants = [v for v in variants if v != majority_variant]
                minority_cells = sum(variant_counts[v] for v in minority_variants)
                if minority_cells > 0:
                    issues.append({
                        "Column": col, "Issue": get_text("issue_whitespace", lang),
                        "Detail": " | ".join(repr(e) for e in minority_variants[:5]), "Count": minority_cells,
                    })
        # (Check 3 — Noise — intentionally removed; handled by detect_noise_values)

        # ------------------------------------------------------------------
        # 4. Rare Singletons — values appearing exactly once (possible typo)
        #    Only applied when cardinality < 50 to avoid false positives on
        #    high-cardinality columns (IDs, free-text, etc.).
        # ------------------------------------------------------------------
        if len(unique_raw) < 50:
            singletons = vc[vc == 1]  # reuse pre-computed vc
            if 0 < len(singletons) < len(unique_raw) * 0.3 and len(singletons) <= 10:
                issues.append({
                    "Column": col, "Issue": get_text("issue_rare_singletons", lang),
                    "Detail": ", ".join(singletons.index[:5].tolist()), "Count": len(singletons),
                })

        # ------------------------------------------------------------------
        # 5. Mixed Measurement Units — '10 kg' mixed with '10 lbs'
        #    Vectorized: extract unit from unique values via str.extract,
        #    then look up occurrence counts via the pre-computed vc.
        # ------------------------------------------------------------------
        _UNIT_RE = r'^[0-9.,]+\s*([a-zA-Z/]+)$'
        unique_series = pd.Series(unique_raw)
        extracted_units = unique_series.str.strip().str.lower().str.extract(
            _UNIT_RE, expand=False
        )  # Series aligned to unique_series, NaN where no unit pattern matches
        if extracted_units.notna().sum() > 0 and extracted_units.nunique() > 1 and extracted_units.nunique() <= 5:
            # Build {normalized_unique_value: unit} lookup
            unit_of = dict(zip(
                unique_series.str.strip().str.lower(),
                extracted_units,
            ))
            # Aggregate row counts per unit using the pre-computed vc
            unit_counts: Dict[str, int] = {}
            for raw_val, cnt in vc.items():
                unit = unit_of.get(str(raw_val).strip().lower())
                if unit and not pd.isna(unit):
                    unit_counts[unit] = unit_counts.get(unit, 0) + int(cnt)
            if unit_counts:
                majority_unit = max(unit_counts, key=unit_counts.get)
                minority_count = sum(c for u, c in unit_counts.items() if u != majority_unit)
                # Only flag when the minority is clearly a minority (< 30 %)
                if minority_count > 0 and minority_count / len(str_vals) < 0.3:
                    minority_units = [u for u in unit_counts if u != majority_unit]
                    issues.append({
                        "Column": col, "Issue": get_text("issue_mixed_units", lang),
                        "Detail": " | ".join(minority_units), "Count": minority_count,
                    })


        # ------------------------------------------------------------------
        # 6. Date Format Issues (heuristic)
        #    Only inspects columns whose name suggests a date/time field.
        #    Un-parseable values in a partially-parseable column are flagged.
        # ------------------------------------------------------------------
        _DATE_KEYWORDS = frozenset(['date', 'time', 'dob', 'created', 'updated'])
        if any(kw in col.lower() for kw in _DATE_KEYWORDS):
            parsed_dates = pd.to_datetime(str_vals, errors='coerce')
            unparsed = parsed_dates.isna()
            unparsed_count = int(unparsed.sum())
            # Only flag when SOME values parse OK (partial failure = format mix)
            if 0 < unparsed_count < len(str_vals):
                bad_examples = str_vals[unparsed].unique()[:3]
                issues.append({
                    "Column": col, "Issue": get_text("issue_date_formats", lang),
                    "Detail": " | ".join(repr(e) for e in bad_examples), "Count": unparsed_count,
                })
    if not issues:
        return pd.DataFrame(columns=["Column", "Issue", "Detail", "Count"])
    return pd.DataFrame(issues)

# =============================================================================
# SCHEMA & SAFE-ZONE VALIDATION (DB-driven)
# =============================================================================

def validate_schema(df: pd.DataFrame, lang: str = "en") -> Dict[str, Any]:
    """
    Cross-reference the uploaded DataFrame's columns against the
    ``employee_schema`` admin rule stored in DB (via session state).
    Returns a dict with:
        status           — 'pass' | 'fail' | 'no_rule'
        expected_count   — number of columns defined in the schema
        actual_count     — number of columns in the uploaded file
        missing_columns  — columns expected but not present
        extra_columns    — columns present but not in schema
        type_mismatches  — list of {column, expected, actual} dicts
    """
    rules = st.session_state.get("analysis_rules", {})
    schema_rule = rules.get("employee_schema")
    if not schema_rule:

        return {"status": "no_rule", "details": []}
    expected_cols = {c["name"].strip().lower(): c for c in schema_rule.get("columns", [])}
    actual_cols = {c.strip().lower(): str(df[c].dtype) for c in df.columns}
    # Build normalised name → original name map for the DF
    actual_original = {c.strip().lower(): c for c in df.columns}
    missing = [expected_cols[k]["name"] for k in expected_cols if k not in actual_cols]
    extra   = [actual_original[k] for k in actual_cols if k not in expected_cols]
    type_mismatches = []
    for key, spec in expected_cols.items():
        if key in actual_cols:
            expected_cat = spec.get("category", "")
            actual_dtype = actual_cols[key]
            if expected_cat == "numeric" and "float" not in actual_dtype and "int" not in actual_dtype:
                type_mismatches.append({
                    "column": spec["name"],
                    "expected": spec["dtype"],
                    "actual": actual_dtype,
                })
            elif expected_cat == "categorical" and "object" not in actual_dtype and "category" not in actual_dtype:
                type_mismatches.append({
                    "column": spec["name"],
                    "expected": spec["dtype"],
                    "actual": actual_dtype,
                })
    status = "pass" if (not missing and not extra and not type_mismatches) else "fail"
    return {
        "status": status,
        "expected_count": len(expected_cols),
        "actual_count": len(actual_cols),
        "missing_columns": missing,
        "extra_columns": extra,
        "type_mismatches": type_mismatches,
    }


def validate_safe_zones(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    """
    Check all numeric columns against the admin-defined safe_zones rule.
    Values that fall outside [min, max] for their column are considered
    violations and are surfaced in the audit results.
    Returns:
        Tuple of:
          - violations_df: DataFrame [Column, Safe Min, Safe Max, Violations,
            Examples] (one row per affected column).
          - total_violations: total number of out-of-range cells.
    """
    rules = st.session_state.get("analysis_rules", {})
    safe_zones = rules.get("safe_zones", {})
    if not safe_zones:
        return pd.DataFrame(), 0
    # Build normalised column name map
    col_map = {_normalize_col_key(c): c for c in df.columns}
    records = []
    total = 0
    for col_key, bounds in safe_zones.items():
        norm_key = _normalize_col_key(col_key)
        actual_col = col_map.get(norm_key)
        if actual_col is None or not pd.api.types.is_numeric_dtype(df[actual_col]):
            continue
        lo, hi = bounds.get("min"), bounds.get("max")
        if lo is None and hi is None:
            continue
        series = df[actual_col].dropna()
        mask = pd.Series(False, index=series.index)
        if lo is not None:
            mask = mask | (series < lo)
        if hi is not None:
            mask = mask | (series > hi)
        count = int(mask.sum())
        if count > 0:
            records.append({
                "Column": actual_col,
                "Safe Min": lo,
                "Safe Max": hi,
                "Violations": count,
                "Examples": ", ".join(str(v) for v in series[mask].head(5).tolist()),
            })
            total += count
    vdf = pd.DataFrame(records) if records else pd.DataFrame(
        columns=["Column", "Safe Min", "Safe Max", "Violations", "Examples"]
    )
    return vdf, total

# =============================================================================
# SAFE ZONE MASK
# =============================================================================

def _apply_safe_zone_mask(values: np.ndarray, col_name: str) -> np.ndarray:
    """
    Return a boolean mask where True = value is OUTSIDE safe zone (real outlier).
    If no safe zone is configured for *col_name*, returns all-True (no filtering).
    """
    zones = _build_norm_zones()
    key = _normalize_col_key(col_name)
    zone = zones.get(key)
    if zone is None:
        return np.ones(len(values), dtype=bool)
    lo = zone.get("min")
    hi = zone.get("max")
    lo_bound = lo if lo is not None else -np.inf
    hi_bound = hi if hi is not None else np.inf
    return ~((values >= lo_bound) & (values <= hi_bound))


# =============================================================================
# 12. FULL AUDIT ORCHESTRATOR
# =============================================================================

def run_full_audit(df: pd.DataFrame, lang: str = "en") -> Dict[str, Any]:
    """
    Single-pass audit orchestrator — the primary entry-point for the UI.
    Runs all audit steps on the DataFrame and returns a flat result dict
    so that the UI can cache the dict and render individual panels without
    re-running expensive computations.
    Steps (in order):
      1. Memory optimisation  — downcast dtypes and convert to category.
      2. Missing / Noise      — null cells, duplicate rows, noise tokens.
      3. Consistency check    — computed FIRST so it can be reused by step 4.
      4. Quality Matrix       — Completeness / Validity / Consistency per column.
      5. Health Score         — mean of all Quality Matrix cells.
      6. Safe Zone validation — out-of-range values per admin rules.
      7. Heavy reports        — field integrity, correlation matrix.
    Note:
        Per-column outlier detection (``compute_field_integrity``) and the
        row-level inspector (``get_risk_records``) respect the admin-configured
        method (IQR / Z-score / Modified Z-score) and are called from the UI.
        This orchestrator always uses IQR for the Quality Matrix baseline.
    """
    # --- Step 1: Memory optimisation ---
    df = _auto_cast_category(df)

    # --- Step 2: Missing / Noise / Duplicates ---
    missing_cells         = int(df.isnull().sum().sum())
    duplicates            = int(df.duplicated().sum())
    noise_df, noise_total = detect_noise_values(df)

    # --- Step 3: Consistency issues ---
    consistency_df      = check_consistency(df, lang)
    inconsistency_total = int(consistency_df["Count"].sum()) if not consistency_df.empty else 0

    # --- Step 4: Health Score (direct computation) ---
    total_cells = len(df) * len(df.columns)
    if total_cells > 0:
        issue_cells = missing_cells + noise_total + inconsistency_total + (duplicates * len(df.columns))
        health = round(float(np.clip((1 - issue_cells / total_cells) * 100, 0, 100)), 2)
    else:
        health = 0.0

    # --- Step 5: Safe Zone validation ---
    safe_zone_df, safe_zone_total = validate_safe_zones(df)

    return {
        "health_score":       health,
        "total_records":      len(df),
        "attributes":         len(df.columns),
        "missing_cells":      missing_cells,
        "duplicates":         duplicates,
        "inconsistency_total": inconsistency_total,
        "consistency":        consistency_df,
        "noise_values":       noise_df,
        "noise_total":        noise_total,
        "safe_zone_violations": safe_zone_df,
        "safe_zone_total":      safe_zone_total,
    }