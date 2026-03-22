"""
preprocessing_engine.py — Vectorized Data Cleaning & Transformation Engine

Responsibilities:
    • Noise / placeholder-value removal (replace with NaN or mode)
    • Text formatting: whitespace trimming, canonical-casing normalization
    • Missing-value imputation (mean/median/mode based on skewness)
    • Duplicate row removal
    • Outlier treatment with Admin Safe Zone awareness:
        - IQR capping         (moderately skewed, |skew| 0.5–1.0)
        - Z-Score capping     (near-normal, |skew| < 0.5)
        - Modified Z-Score capping (highly skewed, |skew| > 1.0)
      Safe zones always take priority: outliers inside a configured safe zone
      are never treated. Outliers outside a safe zone are clipped to its bounds
      when a safe zone exists, or to the statistical fence otherwise.

Design notes:
    - All methods are ``@staticmethod`` (no instance state needed).
    - All heavy loops are vectorized with NumPy / pandas.
    - Safe-zone mask logic is **not** duplicated here — it is delegated to
      ``audit_engine._apply_safe_zone_mask``, which is the single source of truth.
    - ``scipy`` is **not** imported because all stats (IQR, z-score, MAD) are
      computed directly with NumPy for consistency with ``audit_engine.py``.
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# PRIVATE CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

# _MAD_SCALE imported from audit_engine — single source of truth (OPT-1)
from modules.core.audit_engine import _MAD_SCALE  # noqa: E402 (circular-safe: no UI imports)


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC CONSTANTS — single source of truth for labels used across the app
# ─────────────────────────────────────────────────────────────────────────────

# Encoding strategy labels (Step 8)
ENC_LABEL: str = "Label Encoding"
ENC_ONEHOT: str = "One-Hot (drop_first)"
ENC_DROP_REDUNDANT: str = "Drop (Redundant)"

# Scaler method labels (Step 9)
SCALER_STANDARD: str = "StandardScaler"
SCALER_ROBUST: str = "RobustScaler"

# Outlier action labels (Step 5)
ACTION_NO_TREATMENT: str = "No Outlier Treatment"

# Binning preview type labels (Step 7)
BINNING_TYPE_NUMERIC: str = "Numeric Binning"
BINNING_TYPE_CATEGORY: str = "Category Mapping"

# Log transform method labels (Step 6)
LOG_METHOD_LOG1P: str = "log1p"
LOG_METHOD_YJ: str = "yeo-johnson"

# Pipeline step colors — indexed by step number (1-based)
STEP_COLORS: Dict[int, str] = {
    1: "#3B82F6",  # Standardize & Type Cast
    2: "#EF4444",  # Noise Cleaning
    3: "#F59E0B",  # Duplicate Removal
    4: "#F27024",  # Missing Value Imputation
    5: "#8B5CF6",  # Outlier Treatment
    6: "#7FB135",  # Log Transformation
    7: "#10B981",  # Binning & Mapping
    8: "#EC4899",  # Feature Encoding
    9: "#06B6D4",  # Feature Scaling
}


# ─────────────────────────────────────────────────────────────────────────────
# PRIVATE HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _build_col_lower_map(df: pd.DataFrame) -> Dict[str, str]:
    """Build case-insensitive column name lookup: ``{lowercase → actual_name}``."""
    return {c.lower(): c for c in df.columns}


class PreprocessingEngine:
    """
    High-performance engine for data cleaning and transformation.

    All methods are pure ``@staticmethod``s — no instance state is maintained.
    Optimised for vectorized operations on datasets of 32 000+ records.

    Typical pipeline order
    ----------------------
    1. ``standardize_and_type_cast`` — trim, normalize casing, convert dtypes
    2. ``clean_noise_values``        — replace noise tokens with NaN
    3. ``drop_duplicates``           — keep first occurrence of each unique row
    4. ``impute_missing``            — impute missing values (mean/median/mode)
    5. ``handle_outliers``           — cap or remove per-column outliers
    6. ``apply_log_transform``       — log1p / Yeo-Johnson for skewed columns
    7. ``apply_binning_mapping``     — discretize numerics & group categories
    8. ``apply_feature_encoding``    — label + one-hot encoding for categoricals
    9. ``apply_feature_scaling``     — StandardScaler / RobustScaler per column
    """

    # ─────────────────────────────────────────────────────────────────────────
    # 1. UTILITIES
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove fully-duplicate rows, keeping the first occurrence.

        Returns:
            DataFrame with duplicate rows removed (index preserved, reset only
            if the caller needs a clean RangeIndex).
        """
        return df.drop_duplicates()

    # ─────────────────────────────────────────────────────────────────────────
    # 2. NOISE / PLACEHOLDER VALUES
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def clean_noise_values(
        df: pd.DataFrame,
        strategy: str = "replace_nan",
        columns: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Clean noise / placeholder values from categorical columns.

        Uses ``audit_engine._compute_noise_mask`` as the single source of truth
        for what constitutes a noisy value (e.g. "?", "-", "n/a", "null", …).

        Args:
            df:       Input DataFrame (mutated in-place for performance).
            strategy: How to handle detected noise cells:
                        - ``'replace_nan'``  — replace with ``NaN`` (default).
                          Noisy cells then flow into the imputation step.
                        - ``'replace_mode'`` — replace with the column's mode
                          of clean values.
                        - ``'drop'``         — drop entire rows containing noise.
            columns:  Column names to process.  ``None`` processes all
                      object / category columns.

        Returns:
            Cleaned DataFrame.
        """
        from modules.core.audit_engine import _compute_noise_mask, _get_cat_columns

        target_cols: List[str] = columns if columns else _get_cat_columns(df).tolist()
        if not target_cols:
            return df

        for col in target_cols:
            if col not in df.columns:
                continue

            # Operate only on non-null values to avoid str-casting NaN
            series = df[col].dropna().astype(str)
            noise_mask = _compute_noise_mask(series)

            if not noise_mask.any():
                continue  # column is clean — skip

            noise_idx = series[noise_mask].index

            if strategy == "drop":
                df = df.drop(index=noise_idx)

            elif strategy == "replace_nan":
                # NaN lets the imputation step (impute_missing) fill this
                df.loc[noise_idx, col] = np.nan

            elif strategy == "replace_mode":
                clean_vals = series[~noise_mask]
                if not clean_vals.empty:
                    mode_val = clean_vals.mode().iloc[0]
                    df.loc[noise_idx, col] = mode_val
                else:
                    # No clean values exist — fall back to NaN
                    df.loc[noise_idx, col] = np.nan

        return df

    # ─────────────────────────────────────────────────────────────────────────
    # 3. STANDARDIZE & TYPE CAST
    # ─────────────────────────────────────────────────────────────────────────

    #: Minimum ratio of successfully-coerced values required to auto-convert
    #: an ``object`` column to numeric dtype.
    _NUMERIC_COERCE_THRESHOLD: float = 0.90

    @staticmethod
    def standardize_and_type_cast(
        df: pd.DataFrame,
        fix_whitespace: bool = True,
        fix_casing: bool = True,
        convert_dtypes: bool = True,
        columns: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Trim whitespace, normalize casing, and auto-convert dtypes.

        Three phases executed in order:
          1. **Trim** leading/trailing whitespace from categorical columns.
          2. **Normalize casing** — pick the most-frequent surface form as
             the canonical variant (e.g. ``"Male"`` beats ``"male"``).
          3. **Auto dtype conversion** — detect ``object`` columns that
             actually contain numeric data and cast them.  A column is
             converted when ≥ 90 % of its non-null values successfully
             coerce to a number via ``pd.to_numeric``.

        Args:
            df:             Input DataFrame (modified in-place for performance).
            fix_whitespace: Strip leading/trailing whitespace when ``True``.
            fix_casing:     Normalize mixed-case variants to the most common
                            form when ``True``.
            convert_dtypes: Auto-detect and convert mistyped columns when
                            ``True``.
            columns:        Column names to process for text phases.
                            ``None`` processes all object/category columns.
                            Dtype conversion always scans all object columns.

        Returns:
            DataFrame with standardized text and corrected dtypes.
        """
        from modules.core.audit_engine import _get_cat_columns

        target_cols: List[str] = columns if columns else _get_cat_columns(df).tolist()

        # ── Phase 1 & 2: Text standardization (trim + casing) ────────────
        for col in target_cols:
            if col not in df.columns:
                continue

            # Phase 1: Trim leading/trailing whitespace
            if fix_whitespace:
                non_null = df[col].notna()
                if non_null.any():
                    df.loc[non_null, col] = df.loc[non_null, col].astype(str).str.strip()

            # Phase 2: Normalize mixed casing to most-frequent variant
            if fix_casing:
                non_null = df[col].notna()
                if not non_null.any():
                    continue

                str_vals = df.loc[non_null, col].astype(str)

                # Build frequency table once (O(N))
                val_counts = str_vals.value_counts()
                str_uniques = val_counts.index

                # Group unique values by their lowercased form
                lower_map: Dict[str, List[str]] = {}
                for unique_val in str_uniques:
                    lower_map.setdefault(unique_val.lower(), []).append(unique_val)

                # Build {minority_variant → canonical} replacement map
                replace_map: Dict[str, str] = {}
                for variants in lower_map.values():
                    if len(variants) > 1:
                        canonical = max(variants, key=lambda v: val_counts[v])
                        for variant in variants:
                            if variant != canonical:
                                replace_map[variant] = canonical

                if replace_map:
                    df.loc[non_null, col] = str_vals.replace(replace_map)

        # ── Phase 3: Schema-enforced dtype conversion ────────────
        if convert_dtypes:
            from modules.utils.db_config_manager import get_rule
            schema_rule = get_rule("employee_schema")
            
            target_dtypes = {}
            if schema_rule and "columns" in schema_rule:
                for col_info in schema_rule["columns"]:
                    target_dtypes[col_info["name"].strip().lower()] = col_info["dtype"].lower()

            for col in df.columns:
                norm_name = col.strip().lower()
                target_dtype = target_dtypes.get(norm_name)
                
                if target_dtype:
                    # Enforce schema type strictly
                    if target_dtype in ["int64", "int32", "int"]:
                        coerced = pd.to_numeric(df[col], errors="coerce")
                        try:
                            df[col] = coerced.astype("Int64")
                        except TypeError:
                            df[col] = coerced.round().astype("Int64")
                    elif target_dtype in ["float64", "float32", "float"]:
                        df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")
                    elif target_dtype in ["object", "category", "string"]:
                        if not pd.api.types.is_object_dtype(df[col]) and not isinstance(df[col].dtype, pd.CategoricalDtype):
                            df[col] = df[col].astype(str)
                else:
                    # Fallback to auto-conversion if not in schema and is object
                    if df[col].dtype == "object":
                        non_null_series = df[col].dropna()
                        if non_null_series.empty:
                            continue
                        coerced = pd.to_numeric(non_null_series, errors="coerce")
                        valid_count = int(coerced.notna().sum())
                        total_count = len(non_null_series)
                        
                        if total_count > 0 and valid_count / total_count >= PreprocessingEngine._NUMERIC_COERCE_THRESHOLD:
                            coerced_valid = coerced.dropna()
                            if not coerced_valid.empty and (coerced_valid == coerced_valid.astype(int)).all():
                                df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
                            else:
                                df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")

        return df


    @staticmethod
    def get_type_cast_preview(df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Preview which ``object`` columns would be auto-converted to numeric.

        Does NOT mutate the DataFrame — returns a list of dicts describing
        each candidate conversion for display in the UI panel.

        Returns:
            List of dicts with keys: ``Column``, ``Current Type``,
            ``Target Type``, ``Convertible``, ``% Convertible``.
        """
        results: List[Dict[str, Any]] = []
        from modules.utils.db_config_manager import get_rule
        schema_rule = get_rule("employee_schema")
        
        target_dtypes = {}
        if schema_rule and "columns" in schema_rule:
            for col_info in schema_rule["columns"]:
                target_dtypes[col_info["name"].strip().lower()] = col_info["dtype"].lower()

        for col in df.columns:
            norm_name = col.strip().lower()
            target_dtype = target_dtypes.get(norm_name)
            current_dtype = str(df[col].dtype)
            
            non_null_series = df[col].dropna()
            total_count = len(non_null_series)
            
            if target_dtype:
                # Check if it needs coercion
                if target_dtype in ["int64", "int32", "int"] and current_dtype not in ["int64", "Int64", "int32"]:
                    coerced = pd.to_numeric(non_null_series, errors="coerce")
                    valid_count = int(coerced.notna().sum())
                    pct = valid_count / total_count * 100 if total_count > 0 else 0.0
                    results.append({
                        "Column": col,
                        "Current Type": current_dtype,
                        "Target Type": "Int64 (Schema)",
                        "Convertible": valid_count,
                        "% Convertible": round(pct, 1),
                    })
                elif target_dtype in ["float64", "float32", "float"] and current_dtype not in ["float64", "float32"]:
                    coerced = pd.to_numeric(non_null_series, errors="coerce")
                    valid_count = int(coerced.notna().sum())
                    pct = valid_count / total_count * 100 if total_count > 0 else 0.0
                    results.append({
                        "Column": col,
                        "Current Type": current_dtype,
                        "Target Type": "float64 (Schema)",
                        "Convertible": valid_count,
                        "% Convertible": round(pct, 1),
                    })
            else:
                # Fallback to auto-preview for object columns not in schema
                if current_dtype == "object":
                    if non_null_series.empty:
                        continue
                    coerced = pd.to_numeric(non_null_series, errors="coerce")
                    valid_count = int(coerced.notna().sum())
                    pct = valid_count / total_count * 100 if total_count > 0 else 0.0
                    if pct >= PreprocessingEngine._NUMERIC_COERCE_THRESHOLD * 100:
                        coerced_valid = coerced.dropna()
                        if not coerced_valid.empty and (coerced_valid == coerced_valid.astype(int)).all():
                            t_type = "Int64 (Auto)"
                        else:
                            t_type = "float64 (Auto)"
                        results.append({
                            "Column": col,
                            "Current Type": current_dtype,
                            "Target Type": t_type,
                            "Convertible": valid_count,
                            "% Convertible": round(pct, 1),
                        })

        return results

    # ─────────────────────────────────────────────────────────────────────────
    # 4. MISSING VALUE IMPUTATION
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def impute_missing(df: pd.DataFrame) -> pd.DataFrame:
        """
        Auto-fill missing values based on column dtype and distribution shape.

        Rules:
          - **Numeric** — fills with ``mean`` when |skewness| < 0.5 (near-normal
            distribution), otherwise fills with ``median`` (robust to outliers /
            skew). Columns with fewer than 3 non-null values always use median.
          - **Categorical** — fills with ``mode`` (most frequent value).

        Designed for the fixed automated preprocessing pipeline where the user
        has no customization options.

        Returns:
            DataFrame with all missing values filled (no rows/columns removed).
        """
        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
        cat_cols     = df.select_dtypes(include=["object", "category"]).columns.tolist()

        # ── Numeric: mean vs. median based on skewness ──────────────────
        from modules.core.audit_engine import recommend_fill_strategy

        for col in numeric_cols:
            if not df[col].isnull().any():
                continue  # fast-path: no missing values in this column
            
            strategy = recommend_fill_strategy(df[col])
            non_null = df[col].dropna()
            val = float(non_null.mean()) if strategy == "mean" else float(non_null.median())
            
            if pd.api.types.is_integer_dtype(df[col]):
                val = int(round(val))
                
            df[col] = df[col].fillna(val)

        # ── Categorical: mode fill ───────────────────────────────────────
        for col in cat_cols:
            if df[col].isnull().any():
                mode_vals = df[col].mode()
                if not mode_vals.empty:
                    df[col] = df[col].fillna(mode_vals.iloc[0])

        return df

    # ─────────────────────────────────────────────────────────────────────────
    # 5. OUTLIER TREATMENT
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def handle_outliers(
        df: pd.DataFrame,
        method: str,
        columns: List[str],
        threshold: float = 1.5,
    ) -> pd.DataFrame:
        """
        Treat outliers in numeric columns, respecting Admin Safe Zones.

        Detection always uses ``audit_engine._OUTLIER_METHODS`` (the same
        dispatch table the audit page uses) so detection is 100% consistent
        between what the audit shows and what the pipeline cleans.

        Safe Zone priority (matches ``get_risk_records`` behavior):
          1. Values **inside** the safe zone are *never* treated — they are
             protected regardless of the statistical method.
          2. Values **outside** the safe zone that are also statistical outliers
             are clipped to the safe zone bounds.
          3. When no safe zone is configured, outliers are handled by *method*.

        Supported *method* values
        -------------------------
        ``'iqr_capping'``              — clip to IQR fence [Q1 - k·IQR, Q3 + k·IQR].
        ``'zscore_capping'``           — clip to Z-score fence [μ - k·σ, μ + k·σ].
        ``'modified_zscore_capping'``  — clip to MAD-based fence
                                         [median - k·MAD/c, median + k·MAD/c].
        ``'zscore_removal'``           — drop rows (used sparingly; safe zone
                                         overrides to capping when configured).

        Args:
            df:        Input DataFrame (rows may be dropped by ``zscore_removal``).
            method:    Treatment method key (see above).
            columns:   Column names to process (non-numeric columns are skipped).
            threshold: Fence multiplier / score threshold (default 1.5 for IQR,
                       typically 3.0 for Z-score variants).

        Returns:
            Treated DataFrame.
        """
        from modules.core.audit_engine import _OUTLIER_METHODS, _apply_safe_zone_mask

        # ── Map public method name → audit_engine detection key ───────────
        # Note: all "capping" variants use the same *detection* mask; only the
        # *treatment* (how bounds are computed) differs.
        _DETECT_KEY_MAP: Dict[str, str] = {
            "iqr_capping":             "iqr",
            "zscore_capping":          "zscore",
            "modified_zscore_capping": "modified_zscore",
            "zscore_removal":          "zscore",  # same detection, different action
        }

        detect_key = _DETECT_KEY_MAP.get(method, "iqr")
        compute_fn = _OUTLIER_METHODS.get(detect_key, _OUTLIER_METHODS["iqr"])

        numeric_cols = [c for c in columns if pd.api.types.is_numeric_dtype(df[c])]
        if not numeric_cols:
            return df

        safe_zones_raw = {}
        try:
            from modules.core.audit_engine import _get_safe_zones
            safe_zones_raw = _get_safe_zones()
        except Exception:
            pass  # session state not available (e.g. during unit tests)

        for col in numeric_cols:
            vals = df[col].dropna()
            if len(vals) < 3:
                continue  # too few values — skip outlier treatment

            # ── Step 1: Statistical detection (identifies mathematical outliers) ──
            mask, _ = compute_fn(vals.values, threshold)
            if not mask.any():
                continue  # no outliers detected for this column

            # ── Step 2: Safe-zone filtering via shared helper ─────────────
            # _apply_safe_zone_mask returns True = outside safe zone = real outlier
            safe_mask = _apply_safe_zone_mask(vals.values, col)

            # Combine: must be a statistical outlier AND outside the safe zone
            final_mask = mask & safe_mask
            if not final_mask.any():
                continue  # all outliers are within the safe zone — nothing to do

            outlier_indices = vals.index[final_mask]

            # ── Step 3: Resolve safe-zone bounds for capping (only when a safe zone exists) ──
            # has_safe_zone is True when _apply_safe_zone_mask returned at least one False value,
            # meaning some values were protected inside the safe zone.  We re-use
            # safe_zones_raw only here to fetch the bounds — no duplicate key normalisation.
            has_safe_zone = not safe_mask.all()  # all-True means NO safe zone is configured
            if has_safe_zone:
                target_key = col.strip().lower().replace(" ", "_")
                zone_cfg   = next(
                    (zv for zk, zv in safe_zones_raw.items()
                     if zk.strip().lower().replace(" ", "_") == target_key),
                    None,
                )
            else:
                zone_cfg = None
            safe_lo = zone_cfg.get("min") if zone_cfg else None
            safe_hi = zone_cfg.get("max") if zone_cfg else None

            # ── Step 4: Apply treatment ──────────────────────────────────────────────
            if has_safe_zone:
                # Safe zone present → always clip to its bounds regardless of method.
                # Missing bound uses the current data edge (no clipping on that side).
                c_min = safe_lo if safe_lo is not None else float(vals.min())
                c_max = safe_hi if safe_hi is not None else float(vals.max())
                df.loc[outlier_indices, col] = (
                    df.loc[outlier_indices, col].clip(lower=c_min, upper=c_max)
                )

            elif method == "iqr_capping":
                # IQR fence: [Q1 − k·IQR, Q3 + k·IQR]
                q1, q3   = np.percentile(vals.values, [25, 75])
                iqr      = q3 - q1
                lo_fence = q1 - threshold * iqr
                hi_fence = q3 + threshold * iqr
                df.loc[outlier_indices, col] = (
                    df.loc[outlier_indices, col].clip(lower=lo_fence, upper=hi_fence)
                )

            elif method == "zscore_capping":
                # Z-score fence: [μ − k·σ, μ + k·σ]
                mu       = float(vals.mean())
                sigma    = float(vals.std(ddof=1))
                lo_fence = mu - threshold * sigma
                hi_fence = mu + threshold * sigma
                df.loc[outlier_indices, col] = (
                    df.loc[outlier_indices, col].clip(lower=lo_fence, upper=hi_fence)
                )

            elif method == "modified_zscore_capping":
                # Modified Z-score (MAD-based) fence: [med − k·MAD/c, med + k·MAD/c]
                median   = float(np.median(vals.values))
                mad      = float(np.median(np.abs(vals.values - median)))
                if mad == 0:
                    # MAD = 0 means constant column — skip treatment
                    continue
                half_range = threshold * mad / _MAD_SCALE
                lo_fence   = median - half_range
                hi_fence   = median + half_range
                df.loc[outlier_indices, col] = (
                    df.loc[outlier_indices, col].clip(lower=lo_fence, upper=hi_fence)
                )

            elif method == "zscore_removal":
                # Drop rows — only used when explicitly requested and no safe zone
                df = df.drop(index=outlier_indices)

        return df

    # ─────────────────────────────────────────────────────────────────────────
    # 6. LOG TRANSFORMATION
    # ─────────────────────────────────────────────────────────────────────────

    #: Columns with ``|skewness| > SKEW_LOG_THRESHOLD`` are recommended for
    #: log transformation to reduce right-skewness.
    SKEW_LOG_THRESHOLD: float = 1.0

    @staticmethod
    def get_log_transform_candidates(
        df: pd.DataFrame,
    ) -> List[Dict[str, Any]]:
        """
        Identify numeric columns that would benefit from log transformation.

        A column is a candidate when:
          - ``|skewness| > SKEW_LOG_THRESHOLD`` (default 1.0)
          - Column has at least 3 non-null values
          - Column is **not** low-variance (dominant value ≥ 80%) — log
            transform is ineffective for zero-spike distributions and
            would break downstream Binning boundaries.

        For each candidate the recommended method is:
          - **log1p** when ``min >= 0`` (safe, preserves zeros)
          - **yeo-johnson** when ``min < 0`` (handles negative values)

        Args:
            df: Input DataFrame (not mutated).

        Returns:
            List of dicts with keys: ``Column``, ``Skewness``, ``Min``,
            ``Max``, ``Method``.
        """
        results: List[Dict[str, Any]] = []
        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()

        from modules.core.audit_engine import compute_skewness, detect_low_variance

        # Build set of low-variance columns to exclude
        low_var_df = detect_low_variance(df)
        low_var_cols = set(low_var_df["Column"].tolist()) if not low_var_df.empty else set()

        for col in numeric_cols:
            if col in low_var_cols:
                continue  # skip — log transform is ineffective for zero-spread cols

            series = df[col].dropna()
            if len(series) < 3:
                continue

            skew_val = compute_skewness(series)
            if skew_val is None:
                continue

            min_val = float(series.min())
            max_val = float(series.max())

            if abs(skew_val) > PreprocessingEngine.SKEW_LOG_THRESHOLD:
                method = LOG_METHOD_LOG1P if min_val >= 0 else LOG_METHOD_YJ
                results.append({
                    "Column": col,
                    "Skewness": skew_val,
                    "Min": round(min_val, 2),
                    "Max": round(max_val, 2),
                    "Method": method,
                })

        return results

    @staticmethod
    def apply_log_transform(
        df: pd.DataFrame,
        candidates: Optional[List[Dict[str, Any]]] = None,
    ) -> pd.DataFrame:
        """
        Apply log transformation to highly-skewed numeric columns.

        Strategy per column:
          - ``log1p``:       ``np.log1p(x)`` — used when all values ≥ 0.
          - ``yeo-johnson``: ``PowerTransformer(method='yeo-johnson')`` —
            used when column contains negative values.

        Args:
            df:         Input DataFrame (modified in-place for performance).
            candidates: Output of ``get_log_transform_candidates()``.
                        When ``None``, candidates are auto-detected.

        Returns:
            DataFrame with transformed columns.
        """
        if candidates is None:
            candidates = PreprocessingEngine.get_log_transform_candidates(df)

        if not candidates:
            return df

        for candidate in candidates:
            col = candidate["Column"]
            method = candidate["Method"]

            if col not in df.columns:
                continue

            # Ensure float dtype before transform (int64 → float64)
            if pd.api.types.is_integer_dtype(df[col]):
                df[col] = df[col].astype("float64")

            if method == LOG_METHOD_LOG1P:
                # Safe for values >= 0 (including zero)
                non_null = df[col].notna()
                df.loc[non_null, col] = np.log1p(df.loc[non_null, col])

            elif method == LOG_METHOD_YJ:
                from sklearn.preprocessing import PowerTransformer
                pt = PowerTransformer(method=LOG_METHOD_YJ, standardize=False)
                non_null = df[col].notna()
                values = df.loc[non_null, col].values.reshape(-1, 1)
                df.loc[non_null, col] = pt.fit_transform(values).ravel()

        return df

    # ─────────────────────────────────────────────────────────────────────────
    # 7. BINNING & MAPPING
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def get_binning_preview(
        df: pd.DataFrame,
        binning_config: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """
        Build a preview table showing which columns will be binned or mapped.

        Args:
            df:             Input DataFrame.
            binning_config: Dict from ``db_config_manager`` (rule key
                            ``'binning_config'``).

        Returns:
            List of dicts with keys: ``Column``, ``Type``, ``Action``,
            ``Details``, ``Unique Before``.
        """
        results: List[Dict[str, Any]] = []
        if not binning_config:
            return results

        col_lower_map = _build_col_lower_map(df)

        for col, cfg in binning_config.items():
            actual_col = col_lower_map.get(col.lower())
            if actual_col is None:
                continue

            rule_type = cfg.get("type", "")
            unique_before = int(df[actual_col].nunique())

            if rule_type == "bin":
                labels = cfg.get("labels", [])
                action = f"Numeric → {len(labels)} bins"
                details = " | ".join(labels)
                results.append({
                    "Column": actual_col,
                    "Type": BINNING_TYPE_NUMERIC,
                    "Action": action,
                    "Details": details,
                    "Unique Before": unique_before,
                })

            elif rule_type == "map":
                groups = cfg.get("groups", {})
                n_groups = len(groups)
                action = f"Category → {n_groups} groups"
                details = ", ".join(groups.keys())
                results.append({
                    "Column": actual_col,
                    "Type": BINNING_TYPE_CATEGORY,
                    "Action": action,
                    "Details": details,
                    "Unique Before": unique_before,
                })

        return results

    @staticmethod
    def apply_binning_mapping(
        df: pd.DataFrame,
        binning_config: Optional[Dict[str, Any]] = None,
    ) -> pd.DataFrame:
        """
        Apply numeric binning (``pd.cut``) and categorical mapping to reduce
        cardinality, using the admin-defined binning configuration.

        Args:
            df:             Input DataFrame (modified in-place for performance).
            binning_config: Dict from ``db_config_manager``. When ``None``,
                            loaded from ``st.session_state['analysis_rules']``.

        Returns:
            DataFrame with binned / mapped columns.
        """
        if binning_config is None:
            from modules.utils.db_config_manager import get_rule
            binning_config = get_rule("binning_config") or {}

        if not binning_config:
            return df

        col_lower_map = _build_col_lower_map(df)

        for col, cfg in binning_config.items():
            actual_col = col_lower_map.get(col.lower())
            if actual_col is None:
                continue

            rule_type = cfg.get("type", "")

            if rule_type == "bin":
                bins = cfg.get("bins", [])
                labels = cfg.get("labels", [])
                if len(bins) < 2 or len(labels) != len(bins) - 1:
                    continue
                # pd.cut assigns NaN for values outside bin edges
                df[actual_col] = pd.cut(
                    df[actual_col], bins=bins, labels=labels,
                    include_lowest=True, right=True,
                )

            elif rule_type == "map":
                groups = cfg.get("groups", {})
                if not groups:
                    continue
                # Build reverse map: original_value → group_name
                reverse_map: Dict[str, str] = {}
                for group_name, members in groups.items():
                    for member in members:
                        reverse_map[member] = group_name
                # Apply mapping — unmapped values stay as-is (vectorized)
                mapped = df[actual_col].map(reverse_map)
                df[actual_col] = mapped.fillna(df[actual_col])

        return df

    # ─────────────────────────────────────────────────────────────────────────
    # 8. FEATURE ENCODING
    # ─────────────────────────────────────────────────────────────────────────

    #: Columns whose binned values have an intrinsic order.
    #: After Step 7 (Binning & Mapping), these columns contain labels like
    #: "≤25", "26-35", … ("Age") or "Basic", "HS-grad", … ("Education").
    #: They must be Label-encoded (ordinal), **not** One-Hot-encoded.
    _ORDINAL_COLUMNS: set = {"age", "hours_per_week", "education"}

    #: Binary columns — only 2 unique values, Label Encoding is sufficient.
    _BINARY_COLUMNS: set = {"sex", "income"}

    #: When a categorical column has a numeric counterpart that carries the
    #: same information (e.g. ``Education`` vs ``Education_Num``), the
    #: categorical column is **dropped** to avoid redundancy.
    _REDUNDANT_PAIRS: Dict[str, str] = {
        "education": "education_num",  # drop "education" if "education_num" exists
    }

    @staticmethod
    def get_encoding_preview(
        df: pd.DataFrame,
        binning_config: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Build a preview of which categorical columns will be encoded.

        When *binning_config* is supplied, numeric columns that will be
        **binned** after Step 7 are included in the preview (they will
        become categorical), and mapped columns show their **post-mapping
        group names** as examples instead of raw values.

        Classification logic:
          1. Columns listed in ``_REDUNDANT_PAIRS`` whose numeric counterpart
             exists in the DataFrame are marked **Dropped (Redundant)**.
          2. Binary columns (≤ 2 unique) or columns in ``_BINARY_COLUMNS``
             are assigned **Label Encoding**.
          3. Ordinal columns in ``_ORDINAL_COLUMNS`` are assigned
             **Label Encoding**.
          4. All remaining categorical columns are assigned
             **One-Hot Encoding** (``drop_first=True``).

        Args:
            df:             Input DataFrame (not mutated).
            binning_config: Dict from ``db_config_manager`` (rule key
                            ``'binning_config'``).  When provided, the preview
                            simulates the post-Binning/Mapping state.

        Returns:
            List of dicts with keys: ``Column``, ``Unique``, ``Examples``,
            ``Encoding``, ``Reason``.
        """
        results: List[Dict[str, Any]] = []
        df_cols_lower = {c.lower() for c in df.columns}

        # ── Build lookup tables from binning_config ──────────────────────
        col_lower_map = _build_col_lower_map(df)
        bin_labels: Dict[str, List[str]] = {}   # col_lower → bin labels
        map_groups: Dict[str, List[str]] = {}   # col_lower → group names
        if binning_config:
            for cfg_key, cfg_val in binning_config.items():
                cfg_lower = cfg_key.lower()
                rule_type = cfg_val.get("type", "")
                if rule_type == "bin":
                    bin_labels[cfg_lower] = cfg_val.get("labels", [])
                elif rule_type == "map":
                    map_groups[cfg_lower] = list(cfg_val.get("groups", {}).keys())

        # ── Collect columns to preview ───────────────────────────────────
        # Start with current categorical columns
        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

        # Add numeric columns that WILL become categorical after binning
        binned_numeric_cols: List[str] = []
        for cfg_lower, labels in bin_labels.items():
            actual_col = col_lower_map.get(cfg_lower)
            if actual_col and actual_col not in cat_cols:
                binned_numeric_cols.append(actual_col)

        all_cols = cat_cols + binned_numeric_cols

        for col in all_cols:
            col_lower = col.lower()

            # Determine examples and unique count based on post-Step-7 state
            if col_lower in bin_labels:
                # Numeric column that will be binned → show bin labels
                labels = bin_labels[col_lower]
                n_unique = len(labels)
                examples = ", ".join(str(lbl) for lbl in labels)
            elif col_lower in map_groups:
                # Categorical column that will be mapped → show group names
                groups = map_groups[col_lower]
                n_unique = len(groups)
                examples = ", ".join(str(g) for g in groups)
            else:
                # Unaffected column → show current values
                unique_vals = df[col].dropna().unique()
                n_unique = len(unique_vals)
                examples = ", ".join(str(v) for v in unique_vals[:5])
                if n_unique > 5:
                    examples += ", …"

            # ── Check redundancy ─────────────────────────────────────────
            numeric_counterpart = PreprocessingEngine._REDUNDANT_PAIRS.get(col_lower)
            if numeric_counterpart and numeric_counterpart in df_cols_lower:
                results.append({
                    "Column": col,
                    "Unique": n_unique,
                    "Examples": examples,
                    "Encoding": ENC_DROP_REDUNDANT,
                    "Reason": f"Numeric equivalent '{numeric_counterpart}' exists",
                })
                continue

            # ── Binary (≤ 2 unique or known binary) ──────────────────────
            if n_unique <= 2 or col_lower in PreprocessingEngine._BINARY_COLUMNS:
                results.append({
                    "Column": col,
                    "Unique": n_unique,
                    "Examples": examples,
                    "Encoding": ENC_LABEL,
                    "Reason": "Binary column",
                })
                continue

            # ── Ordinal (known ordered columns) ──────────────────────────
            if col_lower in PreprocessingEngine._ORDINAL_COLUMNS:
                results.append({
                    "Column": col,
                    "Unique": n_unique,
                    "Examples": examples,
                    "Encoding": ENC_LABEL,
                    "Reason": "Ordinal — natural order",
                })
                continue

            # ── Binned numeric → ordinal (bins have inherent order) ───────
            if col in binned_numeric_cols:
                results.append({
                    "Column": col,
                    "Unique": n_unique,
                    "Examples": examples,
                    "Encoding": ENC_LABEL,
                    "Reason": "Binned numeric — ordinal",
                })
                continue

            # ── Nominal (everything else) → One-Hot ──────────────────────
            results.append({
                "Column": col,
                "Unique": n_unique,
                "Examples": examples,
                "Encoding": ENC_ONEHOT,
                "Reason": "Nominal — no natural order",
            })

        return results

    @staticmethod
    def apply_feature_encoding(
        df: pd.DataFrame,
        candidates: Optional[List[Dict[str, Any]]] = None,
    ) -> pd.DataFrame:
        """
        Encode categorical features using the hybrid strategy.

        Encoding rules:
          - **Drop (Redundant)**: Column is removed entirely (e.g.
            ``Education`` when ``Education_Num`` exists).
          - **Label Encoding**: Ordinal / binary columns are mapped to
            integer codes via ``sklearn.preprocessing.OrdinalEncoder``.
          - **One-Hot (drop_first)**: Nominal columns are expanded into
            binary indicator columns via ``pd.get_dummies`` with
            ``drop_first=True`` to avoid multicollinearity.

        Args:
            df:         Input DataFrame (modified in-place for performance).
            candidates: Output of ``get_encoding_preview()``.
                        When ``None``, candidates are auto-detected.

        Returns:
            DataFrame with all categorical columns encoded as numeric.
        """
        from sklearn.preprocessing import OrdinalEncoder

        if candidates is None:
            candidates = PreprocessingEngine.get_encoding_preview(df)

        if not candidates:
            return df

        label_cols: List[str] = []
        onehot_cols: List[str] = []
        drop_cols: List[str] = []

        for candidate in candidates:
            col = candidate["Column"]
            if col not in df.columns:
                continue
            encoding = candidate["Encoding"]

            if encoding == ENC_DROP_REDUNDANT:
                drop_cols.append(col)
            elif encoding == ENC_LABEL:
                label_cols.append(col)
            elif encoding == ENC_ONEHOT:
                onehot_cols.append(col)

        # ── Step 1: Drop redundant columns ───────────────────────────────
        if drop_cols:
            df = df.drop(columns=drop_cols)

        # ── Convert category dtype → object (safe for assignment) ─────────
        # After Step 7 (pd.cut / pd.Categorical), some columns are pandas
        # Categorical dtype.  OrdinalEncoder and get_dummies both need
        # object/str dtype to avoid "Cannot setitem on a Categorical" errors.
        all_encode_cols = label_cols + onehot_cols
        for col in all_encode_cols:
            if col in df.columns and df[col].dtype.name == "category":
                df[col] = df[col].astype(str)

        # ── Step 2: Label Encoding (ordinal / binary) ────────────────────
        for col in label_cols:
            if col not in df.columns:
                continue
            non_null = df[col].notna()
            if not non_null.any():
                continue
            encoder = OrdinalEncoder(
                handle_unknown="use_encoded_value",
                unknown_value=-1,
            )
            values = df.loc[non_null, col].astype(str).values.reshape(-1, 1)
            df.loc[non_null, col] = encoder.fit_transform(values).ravel().astype(int)
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # ── Step 3: One-Hot Encoding (nominal) ───────────────────────────
        if onehot_cols:
            existing_onehot = [c for c in onehot_cols if c in df.columns]
            if existing_onehot:
                df = pd.get_dummies(
                    df,
                    columns=existing_onehot,
                    drop_first=True,
                    dtype=int,
                )

        return df

    # ─────────────────────────────────────────────────────────────────────────
    # 9. FEATURE SCALING
    # ─────────────────────────────────────────────────────────────────────────

    #: Skewness threshold for auto-selecting the scaling method.
    #: |skew| < threshold → StandardScaler, |skew| ≥ threshold → RobustScaler.
    SKEW_SCALING_THRESHOLD: float = 0.5

    @staticmethod
    def get_scaling_preview(
        df: pd.DataFrame,
    ) -> List[Dict[str, Any]]:
        """
        Preview which scaler will be auto-selected for each numeric column.

        Classification logic:
          - Binary columns (≤ 2 unique non-null values) are **skipped** —
            scaling 0/1 indicators is not meaningful.
          - ``|skewness| < 0.5`` → **StandardScaler** (mean/std reliable).
          - ``|skewness| ≥ 0.5`` → **RobustScaler** (median/IQR robust
            against remaining skew and Safe-Zone-protected outliers).

        Args:
            df: Input DataFrame (not mutated). Should be post-encoding
                (all columns numeric).

        Returns:
            List of dicts with keys: ``Column``, ``Skewness``, ``Min``,
            ``Max``, ``Std``, ``Method``, ``Reason``.
        """
        from modules.core.audit_engine import compute_skewness

        results: List[Dict[str, Any]] = []
        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()

        for col in numeric_cols:
            series = df[col].dropna()
            if len(series) < 3:
                continue

            # Skip binary / one-hot indicator columns
            n_unique = int(series.nunique())
            if n_unique <= 2:
                continue

            skew_val = compute_skewness(series)
            if skew_val is None:
                continue

            min_val = round(float(series.min()), 2)
            max_val = round(float(series.max()), 2)
            std_val = round(float(series.std(ddof=1)), 2)

            if abs(skew_val) < PreprocessingEngine.SKEW_SCALING_THRESHOLD:
                method = SCALER_STANDARD
                reason = "Near-normal — mean/std reliable"
            else:
                method = SCALER_ROBUST
                reason = "Skewed — median/IQR robust"

            results.append({
                "Column": col,
                "Skewness": skew_val,
                "Min": min_val,
                "Max": max_val,
                "Std": std_val,
                "Method": method,
                "Reason": reason,
            })

        return results

    @staticmethod
    def apply_feature_scaling(
        df: pd.DataFrame,
        candidates: Optional[List[Dict[str, Any]]] = None,
    ) -> pd.DataFrame:
        """
        Scale numeric features using auto-selected method per column.

        Methods (pure NumPy — no sklearn dependency):
          - **StandardScaler**: ``(x - μ) / σ`` — centres on mean,
            scales by standard deviation.
          - **RobustScaler**: ``(x - median) / IQR`` — centres on median,
            scales by inter-quartile range. Robust to remaining outliers
            that were protected by Safe Zones in Step 5.

        Binary / one-hot columns (≤ 2 unique values) are automatically
        excluded from scaling.

        Args:
            df:         Input DataFrame (modified in-place for performance).
            candidates: Output of ``get_scaling_preview()``.
                        When ``None``, candidates are auto-detected.

        Returns:
            DataFrame with scaled numeric columns.
        """
        if candidates is None:
            candidates = PreprocessingEngine.get_scaling_preview(df)

        if not candidates:
            return df

        for candidate in candidates:
            col = candidate["Column"]
            method = candidate["Method"]

            if col not in df.columns:
                continue

            non_null = df[col].notna()
            if not non_null.any():
                continue

            # Ensure float dtype before scaling (int64 → float64)
            if pd.api.types.is_integer_dtype(df[col]):
                df[col] = df[col].astype("float64")

            values = df.loc[non_null, col].values.astype(float)

            if method == SCALER_STANDARD:
                mu = float(np.mean(values))
                sigma = float(np.std(values, ddof=1))
                if sigma == 0:
                    continue  # constant column — skip
                df.loc[non_null, col] = (values - mu) / sigma

            elif method == SCALER_ROBUST:
                median = float(np.median(values))
                q1 = float(np.percentile(values, 25))
                q3 = float(np.percentile(values, 75))
                iqr = q3 - q1
                if iqr == 0:
                    continue  # zero IQR — skip
                df.loc[non_null, col] = (values - median) / iqr

        return df

    # ─────────────────────────────────────────────────────────────────────────
    # 10. OUTLIER PREVIEW (shared between preview panel & pipeline executor)
    # ─────────────────────────────────────────────────────────────────────────

    #: Maps human-readable method name → (detection_key, treatment_method, display_label)
    METHOD_INFO: Dict[str, tuple] = {
        "Z-Score":          ("zscore",          "zscore_capping",          "Z-Score Cap"),
        "IQR":              ("iqr",             "iqr_capping",             "IQR Cap"),
        "Modified Z-Score": ("modified_zscore", "modified_zscore_capping", "Modified Z-Score Cap"),
    }

    @staticmethod
    def compute_outlier_preview_row(
        df: pd.DataFrame,
        col: str,
        safe_zones: dict,
    ) -> Dict[str, Any] | None:
        """
        Compute a single outlier-preview row for one numeric column.

        **Single source of truth** used by both:
          - ``render_detail_panel``  (Step 5 preview via UiComponents)
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

        Returns:
            Dict with keys: Column, Min, Max, Skewness, Auto Method, detect_key,
            treatment_method, Outliers Detected, % Outlier, Action, has_safe_zone.
            Returns ``None`` if the column has fewer than 3 non-null values.
        """
        from modules.core.audit_engine import (
            _apply_safe_zone_mask,
            _OUTLIER_METHODS,
            compute_skewness,
            evaluate_outlier_method,
            default_outlier_threshold,
        )

        series = df[col].dropna()
        if len(series) < 3:
            return None

        rec         = evaluate_outlier_method(series)
        skew_val    = rec["skewness"]
        method_name = rec["method"]  # "Z-Score" | "IQR" | "Modified Z-Score"

        detect_key, treatment_method, base_action = (
            PreprocessingEngine.METHOD_INFO.get(
                method_name, ("iqr", "iqr_capping", "IQR Cap"),
            )
        )

        # Method-aware threshold — consistent with data_audit page
        threshold = default_outlier_threshold(detect_key)

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

        final_mask      = stat_mask & safe_mask
        outlier_count   = int(final_mask.sum())
        raw_stat_count  = int(stat_mask.sum())
        total_rows      = len(series)
        pct_outlier     = round(outlier_count / total_rows * 100, 2) if total_rows > 0 else 0.0

        # 🛡 badge only when Safe Zone actually filtered out some raw outliers.
        safe_zone_active = has_safe_zone and raw_stat_count > 0
        if outlier_count == 0:
            action = f"{ACTION_NO_TREATMENT}  🛡" if safe_zone_active else ACTION_NO_TREATMENT
        else:
            action = f"{base_action}  🛡" if safe_zone_active else base_action

        return {
            "Column":            col,
            "Min":               round(float(series.min()), 2),
            "Max":               round(float(series.max()), 2),
            "Skewness":          skew_val,
            "Auto Method":       method_name,
            "detect_key":        detect_key,
            "treatment_method":  treatment_method,
            "Outliers Detected": outlier_count,
            "% Outlier":         pct_outlier,
            "Action":            action,
            "has_safe_zone":     has_safe_zone,
        }

    # ─────────────────────────────────────────────────────────────────────────
    # 11. PIPELINE STEP DEFINITIONS (shared config for sidebar & detail panel)
    # ─────────────────────────────────────────────────────────────────────────

    PIPELINE_STEP_DEFS: list = [
        {
            "num": 1,
            "title": "Standardize & Type Cast",
            "desc": "Trim, normalize casing, convert dtypes",
            "icon": ":material/tune:",
            "color": STEP_COLORS[1],
        },
        {
            "num": 2,
            "title": "Noise Cleaning",
            "desc": "Replace noise & placeholder values with NaN",
            "icon": ":material/delete:",
            "color": STEP_COLORS[2],
        },
        {
            "num": 3,
            "title": "Duplicate Removal",
            "desc": "Detect & remove exact duplicate rows",
            "icon": ":material/content_copy:",
            "color": STEP_COLORS[3],
        },
        {
            "num": 4,
            "title": "Missing Value Imputation",
            "desc": "Imputation: mean / median / mode",
            "icon": ":material/healing:",
            "color": STEP_COLORS[4],
        },
        {
            "num": 5,
            "title": "Outlier Treatment",
            "desc": "Auto-detect & cap statistical outliers",
            "icon": ":material/square_foot:",
            "color": STEP_COLORS[5],
        },
        {
            "num": 6,
            "title": "Log Transformation",
            "desc": "Reduce skewness via log1p / Yeo-Johnson",
            "icon": ":material/functions:",
            "color": STEP_COLORS[6],
        },
        {
            "num": 7,
            "title": "Binning & Mapping",
            "desc": "Discretize numerics & group categories",
            "icon": ":material/category:",
            "color": STEP_COLORS[7],
        },
        {
            "num": 8,
            "title": "Feature Encoding",
            "desc": "Encode categoricals to numeric (Label + One-Hot)",
            "icon": ":material/label:",
            "color": STEP_COLORS[8],
        },
        {
            "num": 9,
            "title": "Feature Scaling",
            "desc": "Normalize numeric features via auto-selected scaler",
            "icon": ":material/straighten:",
            "color": STEP_COLORS[9],
        },
    ]