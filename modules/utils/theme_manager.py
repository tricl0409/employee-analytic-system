from typing import Dict

# --- METRIC CARD PALETTE ---
STATUS_COLORS = {
    "success":  {"hex": "#A6CE39", "rgba": "rgba(166, 206, 57, 0.2)"},
    "neutral":  {"hex": "#5B86E5", "rgba": "rgba(91, 134, 229, 0.2)"},
    "warning":  {"hex": "#F59E0B", "rgba": "rgba(245, 158, 11, 0.2)"},
    "critical": {"hex": "#FF5B5C", "rgba": "rgba(255, 91, 92, 0.2)"},
    "info":     {"hex": "#9C27B0", "rgba": "rgba(156, 39, 176, 0.2)"}
}

CHART_COLORWAY = [
    STATUS_COLORS["success"]["hex"],
    STATUS_COLORS["neutral"]["hex"],
    STATUS_COLORS["warning"]["hex"],
    STATUS_COLORS["critical"]["hex"],
    STATUS_COLORS["info"]["hex"]
]

# Define centralized color palettes
PALETTES = {
    "dark_glass": {
        "primary_bg": "#020617",
        "secondary_bg": "rgba(2, 6, 23, 0.4)",
        "card_bg": "rgba(255, 255, 255, 0.03)",
        "card_border": "rgba(255, 255, 255, 0.08)",
        "text_main": "#F8FAFC",
        "text_secondary": "#94A3B8",
        "text_muted": "#64748B",
        "accent_blue": STATUS_COLORS["neutral"]["hex"],
        "accent_green": STATUS_COLORS["success"]["hex"],
        "accent_orange": STATUS_COLORS["warning"]["hex"],
        "accent_red": STATUS_COLORS["critical"]["hex"],
        "accent_purple": STATUS_COLORS["info"]["hex"],
        
        "status_success": STATUS_COLORS["success"]["hex"],
        "status_success_bg": STATUS_COLORS["success"]["rgba"],
        "status_neutral": STATUS_COLORS["neutral"]["hex"],
        "status_neutral_bg": STATUS_COLORS["neutral"]["rgba"],
        "status_warning": STATUS_COLORS["warning"]["hex"],
        "status_warning_bg": STATUS_COLORS["warning"]["rgba"],
        "status_critical": STATUS_COLORS["critical"]["hex"],
        "status_critical_bg": STATUS_COLORS["critical"]["rgba"],
        "status_info": STATUS_COLORS["info"]["hex"],
        "status_info_bg": STATUS_COLORS["info"]["rgba"],
        "glass_blur": "15px",
        "font_family": "'Inter', sans-serif",
        "chart_bg": "#1A1E2E",
        "chart_paper": "#0E1117"
    }
}

def get_theme_css(theme_name: str = "dark_glass") -> str:
    """
    Generates CSS variables for the selected theme.

    Args:
        theme_name (str): The name of the theme ('dark_glass').

    Returns:
        str: A string containing the :root CSS block.
    """
    theme = PALETTES.get(theme_name, PALETTES["dark_glass"])
    
    css_vars = []
    for key, value in theme.items():
        var_name = f"--{key.replace('_', '-')}"
        css_vars.append(f"{var_name}: {value};")
        
    return f"""
    :root {{
        {chr(10).join(css_vars)}
    }}
    """
