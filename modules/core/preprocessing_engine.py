"""
preprocessing_engine.py — Vectorized Data Cleaning & Transformation Engine

Responsibilities:
    • Noise / garbage-value removal (replace with NaN or mode)
    • Text formatting: whitespace trimming, canonical-casing normalization
    • Smart missing-value imputation (mean/median/mode based on skewness)
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


class PreprocessingEngine:
    """
    High-performance engine for data cleaning and transformation.

    All methods are pure ``@staticmethod``s — no instance state is maintained.
    Optimised for vectorized operations on datasets of 32 000+ records.

    Typical pipeline order
    ----------------------
    1. ``clean_noise_values``   — replace garbage tokens with NaN
    2. ``fix_text_formatting``  — strip whitespace, canonicalize casing
    3. ``handle_missing_smart`` — impute missing values (mean/median/mode)
    4. ``drop_duplicates``      — keep first occurrence of each unique row
    5. ``handle_outliers``      — cap or remove per-column outliers
    """

    # ─────────────────────────────────────────────────────────────────────────
    # 1. UTILITIES
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def get_column_stats(df: pd.DataFrame) -> Dict[str, Any]:
        """
        Return a summary of basic data quality counts for the DataFrame.

        Returns:
            dict with keys:
              - ``missing_total``         — total null cells across all columns.
              - ``duplicates_total``      — number of fully duplicate rows.
              - ``columns_with_missing``  — list of column names that have ≥ 1 null.
        """
        return {
            "missing_total":        int(df.isnull().sum().sum()),
            "duplicates_total":     int(df.duplicated().sum()),
            "columns_with_missing": df.columns[df.isnull().any()].tolist(),
        }

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
    # 2. NOISE / GARBAGE VALUES
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def clean_noise_values(
        df: pd.DataFrame,
        strategy: str = "replace_nan",
        columns: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Clean noise / garbage placeholder values from categorical columns.

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
                # NaN lets the imputation step (handle_missing_smart) fill this
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
    # 3. TEXT FORMATTING
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def fix_text_formatting(
        df: pd.DataFrame,
        fix_whitespace: bool = True,
        fix_casing: bool = True,
        columns: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Fix leading / trailing whitespace and mixed casing in categorical columns.

        Casing normalization picks the **most-frequent** surface form as the
        canonical variant (e.g. "Male" beats "male" if "Male" appears more often).
        This is an O(N) operation: value_counts is computed once per column and
        reused for all variants.

        Args:
            df:             Input DataFrame.
            fix_whitespace: Strip leading/trailing whitespace when ``True``.
            fix_casing:     Normalize mixed-case variants to the most common
                            form when ``True``.
            columns:        Column names to process.  ``None`` processes all
                            object / category columns.

        Returns:
            DataFrame with corrected text values.
        """
        from modules.core.audit_engine import _get_cat_columns

        target_cols: List[str] = columns if columns else _get_cat_columns(df).tolist()
        if not target_cols:
            return df

        for col in target_cols:
            if col not in df.columns:
                continue

            # ── Phase 1: Trim leading/trailing whitespace ─────────────────
            if fix_whitespace:
                non_null = df[col].notna()
                if non_null.any():
                    df.loc[non_null, col] = df.loc[non_null, col].astype(str).str.strip()

            # ── Phase 2: Normalize mixed casing to most-frequent variant ──
            if fix_casing:
                non_null = df[col].notna()
                if not non_null.any():
                    continue

                str_vals = df.loc[non_null, col].astype(str)

                # Build frequency table once (O(N)) → used for canonical selection
                val_counts = str_vals.value_counts()
                str_uniques = val_counts.index  # already sorted by frequency

                # Group unique values by their lowercased form
                lower_map: Dict[str, List[str]] = {}
                for v in str_uniques:
                    lower_map.setdefault(v.lower(), []).append(v)

                # Build {minority_variant → canonical} replacement map
                replace_map: Dict[str, str] = {}
                for variants in lower_map.values():
                    if len(variants) > 1:
                        # Canonical = the variant with the highest frequency
                        canonical = max(variants, key=lambda v: val_counts[v])
                        for variant in variants:
                            if variant != canonical:
                                replace_map[variant] = canonical

                if replace_map:
                    # O(N) bulk replacement via pandas Series.replace
                    df.loc[non_null, col] = str_vals.replace(replace_map)

        return df

    # ─────────────────────────────────────────────────────────────────────────
    # 4. MISSING VALUE IMPUTATION
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def handle_missing_values(
        df: pd.DataFrame,
        strategy: str,
        columns: Optional[List[str]] = None,
        fill_value: Any = None,
    ) -> pd.DataFrame:
        """
        General-purpose missing-value handler with multiple strategies.

        Args:
            df:         Input DataFrame.
            strategy:   One of:
                          ``'drop_rows'``    — drop rows with any null in *columns*.
                          ``'drop_cols'``    — drop the specified columns entirely.
                          ``'fill_value'``   — fill with a fixed *fill_value*.
                          ``'fill_mean'``    — fill numeric columns with column mean.
                          ``'fill_median'``  — fill numeric columns with column median.
                          ``'fill_mode'``    — fill each column with its mode.
                          ``'ffill'``        — forward fill.
                          ``'bfill'``        — backward fill.
            columns:    Columns to process (``None`` = all columns).
            fill_value: Value used when ``strategy='fill_value'``.

        Returns:
            DataFrame with missing values handled per the chosen strategy.
        """
        target_cols = columns if columns else df.columns.tolist()

        if strategy == "drop_rows":
            return df.dropna(subset=columns) if columns else df.dropna()

        if strategy == "drop_cols":
            return df.drop(columns=target_cols)

        # ── Vectorized filling strategies ──────────────────────────────────
        if strategy == "fill_value" and fill_value is not None:
            df.loc[:, target_cols] = df[target_cols].fillna(fill_value)

        elif strategy == "fill_mean":
            num_cols = [c for c in target_cols if pd.api.types.is_numeric_dtype(df[c])]
            if num_cols:
                df.loc[:, num_cols] = df[num_cols].fillna(df[num_cols].mean())

        elif strategy == "fill_median":
            num_cols = [c for c in target_cols if pd.api.types.is_numeric_dtype(df[c])]
            if num_cols:
                df.loc[:, num_cols] = df[num_cols].fillna(df[num_cols].median())

        elif strategy == "fill_mode":
            for col in target_cols:
                mode = df[col].mode()
                if not mode.empty:
                    df[col] = df[col].fillna(mode.iloc[0])

        elif strategy == "ffill":
            df.loc[:, target_cols] = df[target_cols].ffill()

        elif strategy == "bfill":
            df.loc[:, target_cols] = df[target_cols].bfill()

        return df

    @staticmethod
    def handle_missing_smart(df: pd.DataFrame) -> pd.DataFrame:
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
        for col in numeric_cols:
            if not df[col].isnull().any():
                continue  # fast-path: no missing values in this column
            series = df[col].dropna()
            skew_val = series.skew() if len(series) > 2 else float("nan")
            if pd.notna(skew_val) and abs(skew_val) < 0.5:
                df[col] = df[col].fillna(float(series.mean()))
            else:
                df[col] = df[col].fillna(float(series.median()))

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
    # 6. FEATURE ENGINEERING (post-cleaning)
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def encode_features(
        df: pd.DataFrame,
        method: str,
        columns: List[str],
    ) -> pd.DataFrame:
        """
        Encode categorical features for downstream modelling.

        Args:
            df:      Input DataFrame.
            method:  ``'one_hot'``  — one-hot encode (get_dummies, integer output).
                     ``'ordinal'``  — ordinal (integer) encoding via factorize.
            columns: Categorical column names to encode.

        Returns:
            DataFrame with encoded columns (original columns replaced).
        """
        if method == "one_hot":
            return pd.get_dummies(df, columns=columns, prefix=columns, dtype=int)

        elif method == "ordinal":
            for col in columns:
                codes, _ = pd.factorize(df[col])
                df[col]  = codes

        return df

    @staticmethod
    def scale_features(
        df: pd.DataFrame,
        method: str,
        columns: List[str],
    ) -> pd.DataFrame:
        """
        Scale numeric features in-place.

        Args:
            df:      Input DataFrame.
            method:  ``'standard'`` — zero-mean, unit-variance standardisation
                                      (z = (x − μ) / σ).
                     ``'minmax'``   — scale to [0, 1]
                                      (x_scaled = (x − min) / (max − min)).
            columns: Numeric column names to scale.

        Returns:
            DataFrame with scaled columns.
        """
        numeric_cols = [c for c in columns if pd.api.types.is_numeric_dtype(df[c])]
        if not numeric_cols:
            return df

        if method == "standard":
            # OPT-2: cast to float64 first to avoid TypeError when columns
            # are int64 (e.g. after pd.get_dummies) and result is float.
            df[numeric_cols] = df[numeric_cols].astype(float)
            means = df[numeric_cols].mean()
            stds  = df[numeric_cols].std()
            # Avoid division by zero for constant columns
            stds[stds == 0] = 1
            df.loc[:, numeric_cols] = (df[numeric_cols] - means) / stds

        elif method == "minmax":
            df[numeric_cols] = df[numeric_cols].astype(float)  # OPT-2
            mins      = df[numeric_cols].min()
            maxs      = df[numeric_cols].max()
            range_    = maxs - mins
            # Avoid division by zero for constant columns
            range_[range_ == 0] = 1
            df.loc[:, numeric_cols] = (df[numeric_cols] - mins) / range_

        return df
