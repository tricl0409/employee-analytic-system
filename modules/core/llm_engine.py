import os
import streamlit as st
from google import genai
from modules.core.data_engine import compute_dataset_metrics

# Gemini model to use for all AI responses
_GEMINI_MODEL = "gemini-2.5-flash"

def get_gemini_client():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        try:
            api_key = st.secrets["GEMINI_API_KEY"]
        except (FileNotFoundError, KeyError):
            pass
    if not api_key:
        return None
    return genai.Client(api_key=api_key)

def get_dataset_context(df=None):
    if df is None or df.empty:
        return "No dataset is currently loaded."
    
    metrics = compute_dataset_metrics(df)
    
    # Get problematic columns simply by finding missing values
    missing_by_col = df.isnull().sum()
    cols_with_missing = missing_by_col[missing_by_col > 0].to_dict()
    
    context = (
        f"Dataset Context:\n"
        f"- Total Rows: {metrics['rows']:,}\n"
        f"- Total Columns: {metrics['cols']}\n"
        f"- Memory Usage: {metrics['memory_mb']:.1f} MB\n"
        f"- Duplicate Rows: {metrics['duplicates']:,}\n"
        f"- Missing Data Percentage: {metrics['missing_pct']:.1f}%\n"
        f"- Columns with missing values: {cols_with_missing if cols_with_missing else 'None'}\n"
    )
    return context

def stream_llm_response(prompt: str, chat_history: list, df=None):
    client = get_gemini_client()
    if not client:
        yield "⚠️ Gemini API Key is missing. Please set the `GEMINI_API_KEY` environment variable or add it to `st.secrets`."
        return

    context = get_dataset_context(df)
    
    system_prompt = (
        "You are an expert Data Engineer AI Assistant for 'The Transformers' Employee Analytic System. "
        "You help users analyze their datasets and provide actionable advice. "
        "Respond concisely and professionally.\n\n"
        f"{context}"
    )
    
    # Format chat history for Gemini
    # Gemini expects contents to be a list of Content objects, but we can pass dicts
    contents = []
    for msg in chat_history:
        # Convert "assistant" to "model" for Gemini
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
                max_output_tokens=800,
            )
        )
        
        for chunk in response:
            if chunk.text:
                yield chunk.text

    except Exception as e:
        yield f"⚠️ API Error: An error occurred while communicating with the AI: {str(e)}"
