"""
llm_engine.py — Gemini AI integration for the Employee Analytics System.

Provides:
- Cached Gemini client initialization
- Rich dataset context builder (schema, stats, samples)
- Streaming LLM response with enhanced system prompt
"""

import os

import pandas as pd
import streamlit as st
from google import genai

from modules.core.data_engine import compute_dataset_metrics

# Gemini model to use for all AI responses
_GEMINI_MODEL = "gemini-2.5-flash"

# Maximum tokens for AI response
_MAX_OUTPUT_TOKENS = 2048


@st.cache_resource
def get_gemini_client():
    """Create and cache a Gemini API client.

    Returns:
        genai.Client or None if no API key is configured.
    """
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        try:
            api_key = st.secrets["GEMINI_API_KEY"]
        except (FileNotFoundError, KeyError):
            pass
    if not api_key:
        return None
    return genai.Client(api_key=api_key)


def get_dataset_context(df: pd.DataFrame = None, active_file_name: str = "") -> str:
    """Build rich dataset context string for the LLM system prompt.

    Includes:
    - Active file name (current workspace)
    - Basic metrics (rows, columns, memory, duplicates, missing)
    - Column schema (name, dtype)
    - Statistical summary for numeric columns
    - Top unique values for categorical columns
    - Sample rows for data shape understanding

    Args:
        df: The active DataFrame to describe.
        active_file_name: Name of the currently active file in the workspace.

    Returns:
        Formatted context string.
    """
    if df is None or df.empty:
        return "No dataset is currently loaded in the workspace."

    metrics = compute_dataset_metrics(df)

    # ── Basic metrics ─────────────────────────────────────────────────
    file_label = active_file_name if active_file_name else "Unknown"
    is_cleaned = "Yes (preprocessed)" if active_file_name == "__cleaned__" else "No (raw)"
    context_lines = [
        "=== CURRENT WORKSPACE ===",
        f"Active File: {file_label}",
        f"Preprocessed: {is_cleaned}",
        f"Total Rows: {metrics['rows']:,}",
        f"Total Columns: {metrics['cols']}",
        f"Memory Usage: {metrics['memory_mb']:.1f} MB",
        f"Duplicate Rows: {metrics['duplicates']:,}",
        f"Missing Data: {metrics['missing_pct']:.1f}%",
        "",
    ]

    # ── Column schema with dtypes ─────────────────────────────────────
    context_lines.append("=== COLUMN SCHEMA ===")
    for col_name in df.columns:
        dtype_str = str(df[col_name].dtype)
        null_count = df[col_name].isnull().sum()
        null_info = f" ({null_count:,} nulls)" if null_count > 0 else ""
        context_lines.append(f"- {col_name}: {dtype_str}{null_info}")
    context_lines.append("")

    # ── Numeric summary ───────────────────────────────────────────────
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    if numeric_cols:
        context_lines.append("=== NUMERIC SUMMARY ===")
        desc = df[numeric_cols].describe().round(2)
        for col_name in numeric_cols:
            stats = desc[col_name]
            context_lines.append(
                f"- {col_name}: "
                f"mean={stats['mean']}, std={stats['std']}, "
                f"min={stats['min']}, "
                f"25%={stats['25%']}, 50%={stats['50%']}, 75%={stats['75%']}, "
                f"max={stats['max']}"
            )
        context_lines.append("")

    # ── Categorical unique values ─────────────────────────────────────
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    if cat_cols:
        context_lines.append("=== CATEGORICAL COLUMNS ===")
        for col_name in cat_cols:
            unique_values = df[col_name].dropna().astype(str).value_counts()
            n_unique = len(unique_values)
            # Show top 8 values with counts
            top_values = unique_values.head(8)
            vals_str = ", ".join(
                f"{val}({cnt:,})" for val, cnt in top_values.items()
            )
            if n_unique > 8:
                vals_str += f", ... (+{n_unique - 8} more)"
            context_lines.append(f"- {col_name} [{n_unique} unique]: {vals_str}")
        context_lines.append("")

    # ── Sample rows ───────────────────────────────────────────────────
    context_lines.append("=== SAMPLE DATA (3 rows) ===")
    sample_df = df.head(3)
    context_lines.append(sample_df.to_string(index=False, max_colwidth=30))

    return "\n".join(context_lines)


def _build_system_prompt(context: str, page_context: str = "") -> str:
    """Build the full system prompt with role, context, and instructions.

    Args:
        context: Dataset context string from get_dataset_context().
        page_context: Name of the current page the user is viewing.

    Returns:
        Complete system prompt.
    """
    page_hint = ""
    if page_context:
        page_hint = (
            f"\nThe user is currently on the '{page_context}' page. "
            f"Tailor your answers to be relevant to this analysis stage."
        )

    return (
        "You are a Senior Data Science AI Assistant for 'The Transformers' "
        "Employee Analytics System. This system analyzes workforce census data "
        "to uncover socio-economic factors influencing income levels.\n\n"
        "## IMPORTANT: Current Workspace Focus\n"
        "You MUST always base your analysis, insights, and charts on the "
        "CURRENT WORKSPACE dataset provided below. Never fabricate data or "
        "use hypothetical values — only reference actual columns, values, "
        "and statistics from the loaded dataset.\n\n"
        "## Your Capabilities\n"
        "- Explain data quality issues, outlier patterns, distributions\n"
        "- Suggest preprocessing strategies (imputation, encoding, scaling)\n"
        "- Interpret EDA visualizations and statistical findings\n"
        "- Provide actionable insights on income drivers\n"
        "- **Generate interactive Plotly charts** on request\n\n"
        "## Response Guidelines\n"
        "- Use **Markdown** formatting: headers, bold, bullet points, tables\n"
        "- Include specific column names and values from the dataset\n"
        "- Be concise but thorough — prefer structured answers\n"
        "- Use professional, analytical tone\n\n"
        "## Chart Generation Rules\n"
        "When the user asks you to draw, plot, visualize, or chart something:\n"
        "1. Write a Python code block fenced with ```chart-python\n"
        "2. The variable `df` (pandas DataFrame) is pre-loaded — do NOT import or create it\n"
        "3. You MUST create a variable named `fig` as a Plotly Figure object\n"
        "4. Use `plotly.express` (imported as `px`) or `plotly.graph_objects` (imported as `go`)\n"
        "5. `pd` (pandas) and `np` (numpy) are also available\n"
        "6. Use a dark theme: `template='plotly_dark'`\n"
        "7. Set reasonable figure size: `fig.update_layout(height=450)`\n"
        "8. Do NOT call `fig.show()` or `st.plotly_chart()` — the app handles rendering\n"
        "9. You may add explanatory text BEFORE or AFTER the code block\n\n"
        "Example:\n"
        "```chart-python\n"
        "fig = px.histogram(df, x='age', color='income', barmode='overlay',\n"
        "                   template='plotly_dark', title='Age Distribution by Income')\n"
        "fig.update_layout(height=450)\n"
        "```\n"
        f"{page_hint}\n\n"
        f"{context}"
    )


def stream_llm_response(
    prompt: str,
    chat_history: list,
    df: pd.DataFrame = None,
    page_context: str = "",
):
    """Stream a response from the Gemini LLM.

    Args:
        prompt: The user's latest message.
        chat_history: Previous chat messages (excluding the latest user message).
        df: The active DataFrame for context.
        page_context: Current page name for contextual awareness.

    Yields:
        Text chunks of the AI response.
    """
    client = get_gemini_client()
    if not client:
        yield (
            "⚠️ Gemini API Key is missing. "
            "Please set the `GEMINI_API_KEY` environment variable "
            "or add it to `.streamlit/secrets.toml`."
        )
        return

    active_file = st.session_state.get("active_file", "")
    context = get_dataset_context(df, active_file_name=active_file)
    system_prompt = _build_system_prompt(context, page_context)

    # Format chat history for Gemini (role: user/model)
    contents = []
    for msg in chat_history:
        role = "model" if msg["role"] == "assistant" else "user"
        contents.append({"role": role, "parts": [{"text": msg["content"]}]})

    contents.append({"role": "user", "parts": [{"text": prompt}]})

    try:
        response = client.models.generate_content_stream(
            model=_GEMINI_MODEL,
            contents=contents,
            config=genai.types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=0.7,
                max_output_tokens=_MAX_OUTPUT_TOKENS,
            ),
        )

        for chunk in response:
            if chunk.text:
                yield chunk.text

    except Exception as exc:
        yield f"⚠️ API Error: {exc}"


# ==============================================================================
# CHART CODE PARSING & EXECUTION
# ==============================================================================

import re

_CHART_BLOCK_RE = re.compile(
    r"```chart-python\s*\n(.*?)```",
    re.DOTALL,
)


def parse_response_parts(response_text: str) -> list:
    """Split an AI response into text and chart-code parts.

    Args:
        response_text: Full AI response string.

    Returns:
        List of dicts: [{"type": "text", "content": "..."}, {"type": "chart", "code": "..."}]
    """
    parts = []
    last_end = 0

    for match in _CHART_BLOCK_RE.finditer(response_text):
        # Text before this code block
        text_before = response_text[last_end:match.start()].strip()
        if text_before:
            parts.append({"type": "text", "content": text_before})

        # The code block itself
        code = match.group(1).strip()
        if code:
            parts.append({"type": "chart", "code": code})

        last_end = match.end()

    # Remaining text after last code block
    text_after = response_text[last_end:].strip()
    if text_after:
        parts.append({"type": "text", "content": text_after})

    # If no chart blocks found, return the whole thing as text
    if not parts:
        parts.append({"type": "text", "content": response_text})

    return parts


def execute_chart_code(code: str, df: pd.DataFrame):
    """Execute AI-generated chart code in a sandboxed environment.

    Only plotly.express, plotly.graph_objects, pandas, and numpy are available.
    The code must produce a variable named `fig`.

    Args:
        code: Python code string generated by the AI.
        df: The active DataFrame to pass into the code.

    Returns:
        Plotly Figure object, or None if execution failed.

    Raises:
        ValueError: If the code produces no `fig` variable.
        RuntimeError: If code execution fails.
    """
    import numpy as np
    import plotly.express as px
    import plotly.graph_objects as go

    # Strip safe import lines — these modules are already provided in the sandbox
    _SAFE_IMPORT_RE = re.compile(
        r"^\s*(?:import\s+(?:plotly\.express|plotly\.graph_objects|plotly|pandas|numpy)"
        r"|from\s+(?:plotly\.express|plotly\.graph_objects|plotly|pandas|numpy)\s+import\s+.+)"
        r"(?:\s+as\s+\w+)?\s*$",
        re.MULTILINE,
    )
    code = _SAFE_IMPORT_RE.sub("", code).strip()

    # Reject obviously dangerous patterns
    _BLOCKED_PATTERNS = [
        "import os", "import sys", "import subprocess",
        "import shutil", "import socket", "import http",
        "__import__", "eval(", "exec(", "open(",
        "os.system", "os.popen", "subprocess.",
    ]
    code_lower = code.lower()
    for pattern in _BLOCKED_PATTERNS:
        if pattern.lower() in code_lower:
            raise RuntimeError(f"Blocked: '{pattern}' is not allowed in chart code.")

    # Sandboxed globals — only safe data science libraries
    safe_globals = {
        "pd": pd,
        "np": np,
        "px": px,
        "go": go,
        "df": df.copy() if df is not None else pd.DataFrame(),
        "__builtins__": {"range": range, "len": len, "str": str, "int": int,
                         "float": float, "list": list, "dict": dict,
                         "tuple": tuple, "round": round, "sorted": sorted,
                         "min": min, "max": max, "sum": sum, "abs": abs,
                         "enumerate": enumerate, "zip": zip, "map": map,
                         "filter": filter, "bool": bool, "set": set,
                         "True": True, "False": False, "None": None,
                         "print": lambda *a, **kw: None},
    }
    safe_locals = {}

    try:
        exec(code, safe_globals, safe_locals)
    except Exception as exc:
        raise RuntimeError(f"Chart code execution failed: {exc}") from exc

    fig = safe_locals.get("fig") or safe_globals.get("fig")
    if fig is None:
        raise ValueError(
            "Chart code did not produce a `fig` variable. "
            "Ensure the code creates a Plotly Figure named `fig`."
        )

    return fig

