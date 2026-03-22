"""
db_config_manager.py — CRUD utility for analysis rules stored in SQLite.
All rule values are stored as JSON strings and parsed on read.
"""

import json
from datetime import datetime
from typing import Any, Dict, Optional

import streamlit as st

from modules.core.auth_engine import _get_db_connection


# ==============================================================================
# DEFAULTS
# ==============================================================================

_DEFAULT_EMPLOYEE_SCHEMA = {
    "columns": [
        {"name": "Age",              "dtype": "int64",   "category": "numeric"},
        {"name": "Workclass",        "dtype": "object",  "category": "categorical"},
        {"name": "Fnlwgt",           "dtype": "int64",   "category": "numeric"},
        {"name": "Education",        "dtype": "object",  "category": "categorical"},
        {"name": "Education_Num",    "dtype": "int64",   "category": "numeric"},
        {"name": "Marital_Status",   "dtype": "object",  "category": "categorical"},
        {"name": "Occupation",       "dtype": "object",  "category": "categorical"},
        {"name": "Relationship",     "dtype": "object",  "category": "categorical"},
        {"name": "Race",             "dtype": "object",  "category": "categorical"},
        {"name": "Sex",              "dtype": "object",  "category": "categorical"},
        {"name": "Capital_Gain",     "dtype": "int64",   "category": "numeric"},
        {"name": "Capital_Loss",     "dtype": "int64",   "category": "numeric"},
        {"name": "Hours_per_Week",   "dtype": "float64", "category": "numeric"},
        {"name": "Native_Country",   "dtype": "object",  "category": "categorical"},
        {"name": "Income",           "dtype": "object",  "category": "categorical"},
    ]
}


_DEFAULT_SAFE_ZONES = {
    "Age":            {"min": 10, "max": 90},
    "Fnlwgt":         {"min": 10000, "max": 1500000},
    "Education_Num":  {"min": 1,  "max": 16},
    "Capital_Gain":   {"min": 0,  "max": 99999},
    "Capital_Loss":   {"min": 0,  "max": 4356},
}

_DEFAULT_NOISE_PATTERNS = [
    "?", "-", "--", "---", ".", "..", "...",
    "n/a", "na", "null", "none", "undefined", "#n/a",
    "missing", "unknown", "not available", "#value!",
    "#ref!", "#div/0!", "inf", "-inf", "nan",
]

_DEFAULT_BINNING_CONFIG = {
    # ── Numeric Binning ───────────────────────────────────────────────────
    "Age": {
        "type":   "bin",
        "bins":   [0, 25, 35, 45, 55, 65, 120],
        "labels": ["≤25", "26-35", "36-45", "46-55", "56-65", ">65"],
    },
    "Hours_per_Week": {
        "type":   "bin",
        "bins":   [0, 20, 39, 40, 60, 168],
        "labels": ["≤20", "21-39", "40", "41-60", ">60"],
    },
    "Capital_Gain": {
        "type":   "bin",
        "bins":   [-1, 0, 5000, 99998, float("inf")],
        "labels": ["None", "Low", "High", "Extreme"],
    },
    "Capital_Loss": {
        "type":   "bin",
        "bins":   [-1, 0, 1900, float("inf")],
        "labels": ["None", "Low", "High"],
    },
    # ── Categorical Mapping ───────────────────────────────────────────────
    "Education": {
        "type": "map",
        "groups": {
            "Basic":        ["Preschool", "1st-4th", "5th-6th", "7th-8th", "9th", "10th", "11th", "12th"],
            "HS-grad":      ["HS-grad"],
            "Some/Assoc":   ["Some-college", "Assoc-acdm", "Assoc-voc"],
            "Bachelors":    ["Bachelors"],
            "Advanced":     ["Masters", "Doctorate", "Prof-school"],
        },
    },
    "Workclass": {
        "type": "map",
        "groups": {
            "Public":        ["Federal-gov", "Local-gov", "State-gov"],
            "Private":       ["Private"],
            "Self-employed": ["Self-emp-inc", "Self-emp-not-inc"],
            "Others":         ["Without-pay", "Never-worked", "?"],
        },
    },
    "Marital_Status": {
        "type": "map",
        "groups": {
            "Married":            ["Married-civ-spouse", "Married-AF-spouse"],
            "Never-married":      ["Never-married"],
            "Previously-married": ["Divorced", "Separated", "Widowed"],
            "Spouse-absent":      ["Married-spouse-absent"],
        },
    },
    "Occupation": {
        "type": "map",
        "groups": {
            "Management/Professional": ["Exec-managerial", "Prof-specialty", "Tech-support"],
            "Administrative/Sales":    ["Adm-clerical", "Sales"],
            "Blue-collar":             ["Craft-repair", "Machine-op-inspct", "Handlers-cleaners"],
            "Service":                 ["Other-service", "Protective-serv", "Priv-house-serv", "Armed-Forces"],
            "Transport":               ["Transport-moving"],
            "Farming/Fishing":         ["Farming-fishing"],
            "Others":                  ["?"],
        },
    },
    "Native_Country": {
        "type": "map",
        "groups": {
            "US":       ["United-States"],
            "Asia":     ["China", "India", "Japan", "Philippines", "Vietnam", "Taiwan", "Iran",
                         "Thailand", "Laos", "Cambodia", "Hong", "South", "Japan"],
            "Europe":   ["England", "Germany", "France", "Italy", "Poland", "Portugal",
                         "Greece", "Yugoslavia", "Ireland", "Hungary", "Scotland", "Holand-Netherlands"],
            "Americas": ["Mexico", "Cuba", "Jamaica", "Puerto-Rico", "El-Salvador",
                         "Dominican-Republic", "Guatemala", "Columbia", "Ecuador",
                         "Haiti", "Nicaragua", "Peru", "Honduras", "Trinadad&Tobago",
                         "Canada", "Outlying-US(Guam-USVI-etc)"],
        },
    },
}

_DEFAULTS: Dict[str, Any] = {
    "employee_schema": _DEFAULT_EMPLOYEE_SCHEMA,
    "safe_zones":      _DEFAULT_SAFE_ZONES,
    "noise_patterns":  _DEFAULT_NOISE_PATTERNS,
    "binning_config":  _DEFAULT_BINNING_CONFIG,
}


# ==============================================================================
# CRUD OPERATIONS
# ==============================================================================

@st.cache_data(ttl=300)
def get_rule(rule_key: str) -> Optional[Any]:
    """Fetch a single rule by key, returned as parsed JSON. Cached for 5 min."""
    conn = _get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT rule_value FROM analysis_rules WHERE rule_key = ?", (rule_key,))
    row = cursor.fetchone()
    if row:
        return json.loads(row["rule_value"])
    return None


@st.cache_data(ttl=300)
def get_all_rules() -> Dict[str, Any]:
    """Fetch all rules as a dict. Cached for 5 min."""
    conn = _get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT rule_key, rule_value FROM analysis_rules")
    return {row["rule_key"]: json.loads(row["rule_value"]) for row in cursor.fetchall()}


def update_rule(rule_key: str, rule_value: Any) -> None:
    """Upsert a rule (INSERT OR REPLACE), then clear caches."""
    conn = _get_db_connection()
    cursor = conn.cursor()
    now = datetime.now().isoformat()
    cursor.execute(
        "INSERT OR REPLACE INTO analysis_rules (rule_key, rule_value, updated_at) "
        "VALUES (?, ?, ?)",
        (rule_key, json.dumps(rule_value, ensure_ascii=False), now)
    )
    conn.commit()
    _clear_rule_caches()


def delete_rule(rule_key: str) -> None:
    """Delete a rule by key, then clear caches."""
    conn = _get_db_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM analysis_rules WHERE rule_key = ?", (rule_key,))
    conn.commit()
    _clear_rule_caches()


# ==============================================================================
# SEEDING & SESSION SYNC
# ==============================================================================

def seed_default_rules() -> None:
    """Idempotent: insert each default rule only if it doesn't already exist."""
    conn = _get_db_connection()
    cursor = conn.cursor()
    now = datetime.now().isoformat()

    # Single query to fetch all existing rule keys — avoids N round-trips
    cursor.execute("SELECT rule_key FROM analysis_rules")
    existing_keys = {row["rule_key"] for row in cursor.fetchall()}

    rows_to_insert = [
        (key, json.dumps(value, ensure_ascii=False), now)
        for key, value in _DEFAULTS.items()
        if key not in existing_keys
    ]
    if rows_to_insert:
        cursor.executemany(
            "INSERT INTO analysis_rules (rule_key, rule_value, updated_at) VALUES (?, ?, ?)",
            rows_to_insert,
        )
        conn.commit()
    cursor.close()


def reset_all_rules() -> None:
    """Force-overwrite ALL rules with factory defaults."""
    conn = _get_db_connection()
    cursor = conn.cursor()
    now = datetime.now().isoformat()
    for key, value in _DEFAULTS.items():
        cursor.execute(
            "INSERT OR REPLACE INTO analysis_rules (rule_key, rule_value, updated_at) VALUES (?, ?, ?)",
            (key, json.dumps(value, ensure_ascii=False), now)
        )
    conn.commit()
    _clear_rule_caches()


def load_rules_into_session() -> None:
    """Bulk-load all rules into st.session_state['analysis_rules'] for fast access."""
    st.session_state["analysis_rules"] = get_all_rules()


# ==============================================================================
# HELPERS
# ==============================================================================

def _clear_rule_caches() -> None:
    """Clear both get_rule and get_all_rules caches after a write."""
    get_rule.clear()
    get_all_rules.clear()