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

import re
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
# 1. HEALTH SCORE
# =============================================================================

def compute_health_score(quality_matrix: pd.DataFrame) -> float:
    """
    Compute the overall Health Score as the mean of the Data Quality Matrix.

    The Data Quality Matrix contains Completeness / Accuracy / Consistency
    scores (0–100) for every column.  Taking the unweighted mean of all
    cells gives a single representative percentage that is clipped to [0, 100].

    Returns 0.0 when the matrix is empty (no data loaded).
    """
    if quality_matrix is None or quality_matrix.empty:
        return 0.0

    score = quality_matrix.values.mean()  # unweighted mean across all cells
    return round(float(np.clip(score, 0, 100)), 2)


# =============================================================================
# 2. OUTLIER DETECTION (Vectorized)
# =============================================================================

def detect_outliers_zscore(df: pd.DataFrame, threshold: float = 3.0) -> Dict[str, int]:
    """
    Detect outliers using the classic Z-score method (mean & std).

    Best suited for data that is approximately normally distributed.
    Columns with fewer than 3 non-null values or zero variance are skipped.

    Args:
        df:        Input DataFrame.
        threshold: Number of standard deviations beyond which a value is
                   flagged as an outlier (default 3.0).

    Returns:
        Dict mapping column name → outlier count (only columns with ≥ 1 outlier).
    """
    outliers: Dict[str, int] = {}
    for col in _get_num_columns(df):
        vals = df[col].dropna().values
        if len(vals) < 3:
            continue
        std = np.std(vals, ddof=1)
        if std == 0:
            continue
        count = int(np.sum(np.abs((vals - np.mean(vals)) / std) > threshold))
        if count > 0:
            outliers[col] = count
    return outliers


def detect_outliers_modified_zscore(df: pd.DataFrame, threshold: float = 3.0) -> Dict[str, int]:
    """
    Detect outliers using the Modified Z-score (Median & MAD).

    More robust than the standard Z-score for skewed distributions and
    datasets that already contain extreme values, because it replaces
    mean/std with median/MAD.  Columns where MAD = 0 are skipped.

    Args:
        df:        Input DataFrame.
        threshold: Modified-Z threshold (default 3.0, same intuition as Z-score).

    Returns:
        Dict mapping column name → outlier count.
    """
    outliers: Dict[str, int] = {}
    for col in _get_num_columns(df):
        vals = df[col].dropna().values
        if len(vals) < 3:
            continue
        median = np.median(vals)
        mad = np.median(np.abs(vals - median))
        if mad == 0:
            continue
        count = int(np.sum(np.abs(_MAD_SCALE * (vals - median) / mad) > threshold))
        if count > 0:
            outliers[col] = count
    return outliers


def detect_outliers_iqr(df: pd.DataFrame, factor: float = 1.5) -> Dict[str, int]:
    """
    Detect outliers using the Inter-Quartile Range (IQR) fence method.

    A value is flagged when it lies outside [Q1 − factor×IQR, Q3 + factor×IQR].
    factor=1.5 catches mild outliers; factor=3.0 catches only extreme ones.
    Columns with IQR = 0 (constant or near-constant) are skipped.

    Args:
        df:     Input DataFrame.
        factor: Fence multiplier (default 1.5).

    Returns:
        Dict mapping column name → outlier count.
    """
    outliers: Dict[str, int] = {}
    for col in _get_num_columns(df):
        vals = df[col].dropna().values
        if len(vals) < 4:
            continue
        q1, q3 = np.percentile(vals, [25, 75])
        iqr = q3 - q1
        if iqr == 0:
            continue
        count = int(np.sum((vals < q1 - factor * iqr) | (vals > q3 + factor * iqr)))
        if count > 0:
            outliers[col] = count
    return outliers


# =============================================================================
# 3. FIELD-LEVEL INTEGRITY
# =============================================================================

def _suggest_action(
    col: str,
    s: pd.Series,
    missing: int,
    noise: int,
    safe_zone_violations: int,
    total: int,
    lang: str = "en",
) -> str:
    """
    Generate a human-readable, bullet-pointed action list for a single column.

    Priority order (each tier can contribute independently):
      1. Missing  — >30 % → drop/impute, >5 % → fill
      2. Noise    — >5 %  → clean, >0 %  → review
      3. Outliers — >10 % → cap/remove, >2 % → investigate
      4. Casing / whitespace (categorical only)
      5. Skewness             (numeric only, |skew| > 1)

    Returns a single '\u2022 ...' string or a multi-line bulleted list.
    Returns the “OK” token when no issues are detected.
    """
    suggestions: List[str] = []
    inv_total = 1.0 / total if total > 0 else 0.0  # avoid repeated division
    missing_pct = missing * inv_total * 100
    noise_pct   = noise   * inv_total * 100

    # --- 1. Missing values ---
    if missing_pct > 30:
        suggestions.append(get_text("sug_drop_impute", lang))
    elif missing_pct > 5:
        suggestions.append(get_text("sug_fill_missing", lang))

    # --- 2. Noise / placeholder tokens ---
    if noise_pct > 5:
        suggestions.append(get_text("sug_clean_noise", lang))
    elif noise_pct > 0:
        suggestions.append(get_text("sug_review_noise", lang))

    # --- 3. Safe Zone violations ---
    if safe_zone_violations > 0:
        suggestions.append(get_text("sug_safe_zone", lang))

    # --- 4. Casing / whitespace (categorical columns only) ---
    if _is_categorical(s):
        str_vals = s.dropna().astype(str)  # compute once, reuse below
        trimmed = str_vals.str.strip()
        if (str_vals != trimmed).any():
            suggestions.append(get_text("sug_fix_whitespace", lang))
        # Detect mixed-casing: group unique values by normalised lower-case key
        unique_raw = str_vals.unique()
        lower_groups: Dict[str, list] = {}
        for v in unique_raw:
            lower_groups.setdefault(v.strip().lower(), []).append(v)
        if any(len(vs) > 1 for vs in lower_groups.values()):
            suggestions.append(get_text("sug_fix_casing", lang))

    # --- 5. Skewness (numeric columns only) ---
    if pd.api.types.is_numeric_dtype(s):
        vals = s.dropna()
        if len(vals) > 10 and abs(float(vals.skew())) > 1:
            suggestions.append(get_text("sug_log_transform", lang))

    if not suggestions:
        return get_text("sug_ok", lang)

    # Format as a markdown bullet list for the dataframe renderer
    return "\n".join(f"\u2022 {item}" for item in suggestions)


def compute_field_integrity(
    df: pd.DataFrame,
    lang: str = "en",
    _noise_counts: Dict[str, int] = None,
    _norm_zones: Dict[str, dict] = None,
) -> pd.DataFrame:
    """
    Produce a per-column integrity report for the Field Level Integrity panel.

    Columns in the output DataFrame:
        Column                — column name
        Type                  — pandas dtype string
        Missing               — number of null cells
        Noise                 — number of noisy / placeholder cells (categorical only)
        Real Fill Rate (%)    — (total - missing - noise) / total × 100
        Unique                — number of distinct non-null values
        Safe Zone Violations  — count of numeric values outside the admin-defined
                                safe zone (0 when no safe zone is configured or
                                for categorical columns)
        Suggestion            — bullet-pointed recommended actions

    Args (optional for callers that already have these precomputed):
        _noise_counts: {col_name: noise_int} dict from detect_noise_values().
                       When provided, skips the per-column noise-mask recomputation.
        _norm_zones:   Normalised safe-zone lookup {norm_key: {min, max}} from
                       run_full_audit. When provided, skips the session_state read.

    Note: Per-column outlier detection is handled interactively in the Risk
    Records Inspector (user selects method + threshold there).
    """
    if df.empty:
        return pd.DataFrame()

    total = len(df)

    # Build a normalised safe-zone lookup: {normalised_col_key: {min, max}}
    # Use precomputed dict if provided (avoids repeated session_state reads).
    if _norm_zones is None:
        safe_zones = _get_safe_zones()
        _norm_zones = {
            zk.strip().lower().replace(" ", "_"): zv
            for zk, zv in safe_zones.items()
        }

    records = []
    for col in df.columns:
        s = df[col]
        missing = int(s.isna().sum())

        # Noise is only meaningful for text / categorical columns.
        # Use precomputed noise dict when available (avoids re-running noise mask).
        noise = 0
        if _is_categorical(s):
            if _noise_counts is not None:
                noise = _noise_counts.get(col, 0)
            else:
                noise = int(_compute_noise_mask(s.dropna().astype(str)).sum())

        # Real Fill Rate = (total - missing - noise) / total
        fill = round((total - missing - noise) / total * 100, 1) if total > 0 else 0.0

        # --- Safe Zone Violations (numeric columns only) ---
        sz_violations = 0
        if pd.api.types.is_numeric_dtype(s):
            col_key = col.strip().lower().replace(" ", "_")
            zone = _norm_zones.get(col_key)
            if zone is not None:
                lo = zone.get("min")
                hi = zone.get("max")
                lo_bound = lo if lo is not None else -np.inf
                hi_bound = hi if hi is not None else np.inf
                sz_violations = int((~s.dropna().between(lo_bound, hi_bound)).sum())

        suggestion = _suggest_action(col, s, missing, noise, sz_violations, total, lang)

        records.append({
            "Column": col,
            "Type": str(s.dtype),
            "Missing": missing,
            "Noise": noise,
            "Real Fill Rate (%)": fill,
            "Unique": s.nunique(),
            "Safe Zone Violations": sz_violations,
            "Suggestion": suggestion,
        })

    return pd.DataFrame(records).sort_values("Real Fill Rate (%)").reset_index(drop=True)


# =============================================================================
# 4. NOISE VALUE DETECTION
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
# 5. RISK RECORDS (Outlier Inspector)
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


def _apply_safe_zone_mask(vals: np.ndarray, col_name: str) -> np.ndarray:
    """
    Build a boolean *keep-as-outlier* mask that respects admin Safe Zones.

    For a given column and its non-null values array, this function looks up
    the admin-configured safe zone bounds and returns a boolean array where:
      • ``True``  → value is **outside** the safe zone (i.e. a real outlier).
      • ``False`` → value is **inside** the safe zone (protected; not an outlier).

    When no safe zone is configured for the column, all entries are ``True``
    (every statistically detected outlier is kept as-is).

    This helper is the **single source of truth** for safe-zone filtering and
    is shared by:
      - ``get_risk_records``         (audit display)
      - ``preprocessing_engine.handle_outliers`` (pipeline treatment)
      - ``preprocessing.py``         (outlier preview table)

    Args:
        vals:     1-D NumPy array of **non-null** numeric values for the column,
                  aligned to ``df[col].dropna().values``.
        col_name: The DataFrame column name (used to look up the safe zone).

    Returns:
        Boolean ``np.ndarray`` of shape ``(len(vals),)``.
        ``True`` = outside safe zone = treat as outlier.
        ``False`` = inside safe zone = protect from treatment.
    """
    safe_zones = _get_safe_zones()

    # Normalise the lookup key: lowercase + underscores (matches admin DB format)
    target_key = col_name.strip().lower().replace(" ", "_")

    for zk, zv in safe_zones.items():
        if zk.strip().lower().replace(" ", "_") == target_key:
            lo = zv.get("min")  # None means no lower bound
            hi = zv.get("max")  # None means no upper bound

            if lo is not None and hi is not None:
                # Value must be OUTSIDE [lo, hi] to remain flagged as an outlier
                return ~((vals >= lo) & (vals <= hi))
            elif lo is not None:
                return vals < lo   # only flag below the lower bound
            elif hi is not None:
                return vals > hi   # only flag above the upper bound
            # Safe zone exists but has no bounds → no filtering
            break

    # No safe zone configured → keep all statistically detected outliers
    return np.ones(len(vals), dtype=bool)


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
        threshold: Detection threshold.  Defaults to admin config; falls back to
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
        threshold = 1.5

    compute_fn = _OUTLIER_METHODS.get(method)
    if compute_fn is None:
        return _EMPTY

    # --- Step 1: Statistical outlier detection ---
    mask, scores = compute_fn(vals.values, threshold)
    if not mask.any():
        return _EMPTY

    # --- Step 2: Safe-zone filtering (values inside safe zone are NOT real outliers) ---
    # Delegates to the shared helper to avoid duplicating the zone-lookup logic.
    safe_mask = _apply_safe_zone_mask(vals.values, column)

    # Final outliers = statistically flagged AND outside safe zone
    final_mask = mask & safe_mask
    total_outliers = int(final_mask.sum())

    if total_outliers == 0:
        return _EMPTY

    # --- Step 3: Build result DataFrame, sorted by severity score (descending) ---
    flagged = df.loc[vals.index[final_mask]].copy()
    flagged["_score"] = scores[final_mask]
    flagged = flagged.sort_values("_score", ascending=False).drop(columns="_score")
    return flagged.head(max_rows), total_outliers


def evaluate_outlier_method(s: pd.Series, lang: str = "en") -> Dict[str, str]:
    """
    Evaluate the skewness of a numeric column and recommend an outlier
    detection method.
    
    Returns a dict with:
      - 'method': "Z-Score" | "IQR" | "Modified Z-Score"
      - 'reason': Localized explanation string
    """
    from modules.utils.localization import get_text
    
    clean_s = s.dropna()
    if len(clean_s) < 3 or clean_s.std() == 0:
        return {"method": "IQR", "reason": get_text("rec_iqr", lang)}

    skew = float(clean_s.skew())
    abs_skew = abs(skew)

    if abs_skew < 0.5:
        return {"method": "Z-Score", "reason": get_text("rec_zscore", lang)}
    elif abs_skew <= 1.0:
        return {"method": "IQR", "reason": get_text("rec_iqr", lang)}
    else:
        return {"method": "Modified Z-Score", "reason": get_text("rec_mod_z", lang)}



# =============================================================================
# 7. COLUMN REPORT
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
# 8. CONSISTENCY CHECK
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
# 9. DATA QUALITY MATRIX
# =============================================================================

def compute_quality_matrix(
    df: pd.DataFrame,
    lang: str = "en",
    consistency_df: pd.DataFrame = None,
) -> pd.DataFrame:
    """
    Build a 3-dimensional Data Quality Matrix (rows = columns, cols = dimensions).

    Dimensions:
        Completeness  — % of cells that are neither null nor noise.
        Validity      — % of numeric values within the admin-defined safe zone.
                        Columns without a configured safe zone score 100 %.
                        Categorical columns always score 100 %.
        Consistency   — % of rows NOT flagged by check_consistency (casing,
                        whitespace, singletons, mixed units, date formats).
                        Numeric columns always score 100 % (no consistency
                        checks are defined for them).

    The matrix values are in the range [0, 100] and serve as input to:
      • compute_health_score (overall score)
      • the heatmap visualisation on the Audit page

    Note: Consistency is now derived from check_consistency so the heatmap cell
    and the Consistency Issues panel always reflect the same underlying counts.
    """
    if df.empty:
        return pd.DataFrame()

    total = len(df)

    # Build normalised safe-zone lookup once for the whole DataFrame
    safe_zones = _get_safe_zones()
    norm_zones = {
        zk.strip().lower().replace(" ", "_"): zv
        for zk, zv in safe_zones.items()
    }

    # Run consistency check ONCE and aggregate flagged cell counts per column.
    # check_consistency returns [Column, Issue, Detail, Count] where Count = rows
    # affected by that issue in that column.  Summing Count per Column gives the
    # total number of inconsistent rows for each column.
    # If a precomputed result is provided (e.g. from run_full_audit) reuse it
    # to avoid running check_consistency twice.
    if consistency_df is None:
        consistency_df = check_consistency(df, lang)
    inconsistency_by_col: Dict[str, int] = (
        consistency_df.groupby("Column")["Count"].sum().to_dict()
        if not consistency_df.empty else {}
    )

    records = []
    label_comp = get_text("dim_completeness", lang)
    label_val  = get_text("dim_validity", lang)
    label_cons = get_text("dim_consistency", lang)

    for col in df.columns:
        s = df[col]
        missing = int(s.isna().sum())

        # Noise only applies to categorical columns
        noise = 0
        if _is_categorical(s):
            noise = int(_compute_noise_mask(s.dropna().astype(str)).sum())

        # --- Completeness: % of cells that are neither null nor noise ---
        completeness = round((total - missing - noise) / total * 100, 1) if total > 0 else 0.0

        # --- Validity: % of values within the configured safe zone ---
        # Numeric columns: look up the safe zone; if none configured, score = 100.
        # Categorical columns: validity is not applicable, score = 100.
        if pd.api.types.is_numeric_dtype(s):
            col_key = col.strip().lower().replace(" ", "_")
            zone = norm_zones.get(col_key)
            if zone is not None:
                lo = zone.get("min")
                hi = zone.get("max")
                lo_bound = lo if lo is not None else -np.inf
                hi_bound = hi if hi is not None else np.inf
                vals_non_null = s.dropna()
                violations = int((~vals_non_null.between(lo_bound, hi_bound)).sum())
                n_non_null = len(vals_non_null)
                validity = round((n_non_null - violations) / n_non_null * 100, 1) if n_non_null > 0 else 100.0
            else:
                validity = 100.0  # no constraint configured → no violation
        else:
            validity = 100.0  # categorical columns: N/A, treat as fully valid

        # --- Consistency: % of rows NOT flagged by check_consistency ---
        # Numeric columns: no consistency checks defined → always 100 %.
        # Cap inconsistent count at `total` to avoid negative scores when a
        # single row is counted across multiple overlapping issues.
        if _is_categorical(s):
            inconsistent = min(inconsistency_by_col.get(col, 0), total)
            consistency = round((total - inconsistent) / total * 100, 1) if total > 0 else 100.0
        else:
            consistency = 100.0

        records.append({
            "Column": col,
            label_comp: completeness,
            label_val:  validity,
            label_cons: consistency,
        })

    return pd.DataFrame(records).set_index("Column")


# =============================================================================
# 10. CORRELATION MATRIX
# =============================================================================

def compute_correlation_matrix(df: pd.DataFrame, method: str = "pearson") -> pd.DataFrame:
    """
    Compute a pairwise correlation matrix for all numeric columns.

    Args:
        df:     Input DataFrame.
        method: 'pearson' for linear correlation (default); 'spearman' for
                rank-based correlation that is robust to outliers and non-linear
                relationships.

    Constant columns (std = 0) are excluded as their correlation is undefined.
    Returns an empty DataFrame when fewer than 2 usable numeric columns exist.
    """
    numeric_df = df.select_dtypes(include=["number"])

    # Drop constant columns — their correlation is mathematically undefined
    numeric_df = numeric_df.loc[:, numeric_df.std() > 0]

    if numeric_df.shape[1] < 2:
        return pd.DataFrame()

    return numeric_df.corr(method=method, min_periods=10)


# =============================================================================
# 11. SCHEMA & SAFE-ZONE VALIDATION (DB-driven)
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
    col_map = {c.strip().lower(): c for c in df.columns}
    records = []
    total = 0

    for col_key, bounds in safe_zones.items():
        norm_key = col_key.strip().lower()
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
    # Downcast numeric types and convert low-cardinality strings to category.
    df = _auto_cast_category(df)

    # --- Step 2: Missing / noise / duplicates ---
    missing_cells         = int(df.isnull().sum().sum())
    duplicates            = int(df.duplicated().sum())
    noise_df, noise_total = detect_noise_values(df)

    # --- Step 5: Consistency issues (computed FIRST so it can be reused by quality matrix) ---
    consistency_df      = check_consistency(df, lang)
    inconsistency_total = int(consistency_df["Count"].sum()) if not consistency_df.empty else 0

    # --- Step 3-4: Quality Matrix → Health Score (reuses precomputed consistency_df) ---
    quality_matrix = compute_quality_matrix(df, lang, consistency_df=consistency_df)
    health         = compute_health_score(quality_matrix)

    # --- Step 6: Safe Zone validation (DB-driven) ---
    # Build normalised zone lookup ONCE and reuse in compute_field_integrity (OPT-3).
    safe_zones = _get_safe_zones()
    norm_zones = {
        zk.strip().lower().replace(" ", "_"): zv
        for zk, zv in safe_zones.items()
    }
    safe_zone_df, safe_zone_total = validate_safe_zones(df)

    # Build noise_counts dict from the already-finished noise_df (OPT-3).
    # noise_df has columns [Column, Noise Count, Examples].
    noise_counts: Dict[str, int] = (
        dict(zip(noise_df["Column"], noise_df["Noise Count"]))
        if not noise_df.empty else {}
    )

    return {
        "health_score":       health,
        "total_records":      len(df),
        "attributes":         len(df.columns),
        "missing_cells":      missing_cells,
        "duplicates":         duplicates,
        "inconsistency_total": inconsistency_total,

        # Pass precomputed noise_counts and norm_zones to avoid recomputation
        "field_integrity":    compute_field_integrity(df, lang,
                                  _noise_counts=noise_counts,
                                  _norm_zones=norm_zones),
        "consistency":        consistency_df,
        "quality_matrix":     quality_matrix,
        "correlation":        compute_correlation_matrix(df),
        "noise_values":       noise_df,
        "noise_total":        noise_total,

        # DB-driven validation results
        "safe_zone_violations": safe_zone_df,
        "safe_zone_total":      safe_zone_total,
    }