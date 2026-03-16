import streamlit as st
from modules.utils.theme_manager import get_theme_css


# ==============================================================================
# 1. GLOBAL & FONTS
# ==============================================================================

def get_global_styles(theme="dark_glass"):
    css_vars = get_theme_css(theme)
    return f"""
    {css_vars}
    
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    /* GLOBAL RESET & BACKGROUND */
    html, body, .stApp, .stMarkdown, .stText {{
        font-family: var(--font-family) !important;
        font-size: 0.9rem;
    }}

    .stApp {{
        background-color: var(--primary-bg);
        background-image: 
            radial-gradient(at 0% 0%, rgba(59, 130, 246, 0.12) 0px, transparent 50%),
            radial-gradient(at 100% 0%, rgba(242, 112, 36, 0.12) 0px, transparent 50%);
        color: var(--text-main);
    }}

    /* Smooth page transition */
    @keyframes pageFadeIn {{
        from {{ opacity: 0; transform: translateY(12px); }}
        to   {{ opacity: 1; transform: translateY(0); }}
    }}
    .stMainBlockContainer {{
        animation: pageFadeIn 0.5s ease-out;
    }}

    /* Footer Position */
    .page-footer {{
        width: 100%;
        text-align: center;
        color: rgba(255, 255, 255, 0.4);
        font-size: 0.8rem;
        font-weight: 500;
        letter-spacing: 1px;
        margin-top: 4rem;
        padding-bottom: 2rem;
    }}

    /* Alerts — compact & refined */
    .stAlert {{
        padding: 0.5rem 1rem !important;
        border-radius: 10px !important;
    }}
    .stAlert > div {{
        display: flex !important;
        align-items: center !important;
        gap: 8px !important;
        padding: 0 !important;
    }}
    .stAlert [data-testid="stMarkdownContainer"] p {{
        font-size: 0.75rem !important;
        font-weight: 600 !important;
        letter-spacing: 0.3px !important;
        text-transform: uppercase !important;
        margin: 0 !important;
    }}
    """

# ==============================================================================
# 2. PAGE HEADER ANIMATIONS & CONTAINER
# ==============================================================================

HEADER_STYLES = """
    /* ANIMATIONS */
    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    @keyframes slideUpFade {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }

    @keyframes blink {
        0%, 100% { opacity: 1; }
        50% { opacity: 0; }
    }

    /* CONTAINER STYLE */
    .page-header-container {
        padding: 1.5rem 2rem;
        border-radius: 40px;
        margin-bottom: 1.5rem;
        text-align: center;
        border: 1px solid var(--card-border);
        
        background: linear-gradient(-45deg, 
            var(--status-neutral-bg), 
            var(--secondary-bg), 
            var(--status-warning-bg),
            var(--secondary-bg));
        background-size: 400% 400%;
        animation: 
            gradientShift 15s ease infinite, 
            slideUpFade 0.8s ease-out forwards;
            
        backdrop-filter: blur(var(--glass-blur));
        box-shadow: 0 10px 30px rgba(0,0,0,0.5);
    }

    .page-header-container:hover {
        border-color: rgba(91, 134, 229, 0.3);
        box-shadow: 0 0 40px var(--status-neutral-bg);
        transition: 0.5s ease;
    }
"""

# ==============================================================================
# 3. SIDEBAR BRANDING
# ==============================================================================

SIDEBAR_STYLES = """
    [data-testid="stSidebar"] {
        background-color: var(--secondary-bg) !important;
        backdrop-filter: blur(20px);
        border-right: 1px solid var(--card-border);
    }
    [data-testid="stSidebar"] button svg, 
    [data-testid="stSidebarCollapsedControl"] svg {
        font-family: inherit !important;
        fill: var(--accent-green) !important;
        color: var(--accent-green) !important;
    }
    [data-testid="stSidebarCollapsedControl"] {
        top: 10px !important;
        left: 10px !important;
    }

    /* ── Active Nav Item — Animated Focus Effect ── */
    @keyframes navShimmer {
        0%   { background-position: -200% center; }
        100% { background-position: 200% center; }
    }
    @keyframes navBorderPulse {
        0%, 100% { border-left-color: rgba(91, 134, 229, 0.8); box-shadow: inset 3px 0 10px rgba(91, 134, 229, 0.0); }
        50%       { border-left-color: #60A5FA;                  box-shadow: inset 3px 0 12px rgba(96, 165, 250, 0.25); }
    }

    [data-testid="stSidebarNavItems"] li div[data-testid="stSidebarNavLink"].st-emotion-cache-17lntkn,
    [data-testid="stSidebarNavItems"] li div[data-testid="stSidebarNavLink"][aria-current="page"],
    [data-testid="stSidebarNav"] a[aria-current="page"] {
        position: relative !important;
        background: linear-gradient(
            105deg,
            rgba(59, 130, 246, 0.18) 0%,
            rgba(91, 134, 229, 0.10) 40%,
            rgba(54, 209, 220, 0.12) 70%,
            rgba(59, 130, 246, 0.18) 100%
        ) !important;
        background-size: 200% auto !important;
        animation: navShimmer 3.5s linear infinite, navBorderPulse 2.5s ease-in-out infinite !important;
        border-left: 3px solid rgba(91, 134, 229, 0.8) !important;
        border-radius: 0 10px 10px 0 !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
    }

    [data-testid="stSidebarNavItems"] li div[data-testid="stSidebarNavLink"][aria-current="page"] span,
    [data-testid="stSidebarNav"] a[aria-current="page"] span {
        color: #E2E8FF !important;
        font-weight: 700 !important;
        text-shadow: 0 0 12px rgba(96, 165, 250, 0.6) !important;
        letter-spacing: 0.1px !important;
    }

    [data-testid="stSidebarNavItems"] li div[data-testid="stSidebarNavLink"][aria-current="page"] svg,
    [data-testid="stSidebarNav"] a[aria-current="page"] svg {
        fill: #60A5FA !important;
        color: #60A5FA !important;
        filter: drop-shadow(0 0 4px rgba(96, 165, 250, 0.7)) !important;
    }

    /* ── Inactive Nav Item — Hover Feedback ── */
    [data-testid="stSidebarNavItems"] li div[data-testid="stSidebarNavLink"],
    [data-testid="stSidebarNav"] a {
        transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
        border-left: 3px solid transparent !important;
        border-radius: 0 8px 8px 0 !important;
    }
    [data-testid="stSidebarNavItems"] li div[data-testid="stSidebarNavLink"]:not([aria-current="page"]):hover,
    [data-testid="stSidebarNav"] a:not([aria-current="page"]):hover {
        background: rgba(255, 255, 255, 0.04) !important;
        border-left: 3px solid rgba(255, 255, 255, 0.15) !important;
        transform: translateX(2px) !important;
    }
    [data-testid="stSidebarNavItems"] li div[data-testid="stSidebarNavLink"]:not([aria-current="page"]):hover span,
    [data-testid="stSidebarNav"] a:not([aria-current="page"]):hover span {
        color: rgba(255, 255, 255, 0.85) !important;
    }

    /* Push branding to bottom of sidebar using flex */
    [data-testid="stSidebar"] [data-testid="stSidebarContent"] {
        display: flex !important;
        flex-direction: column !important;
        min-height: 100vh !important;
    }
    .sidebar-branding-bottom {
        margin-top: auto;
        padding: 0 0 4px 0;
    }
    .sidebar-branding-bottom hr {
        border: 0 !important;
        border-top: 1px solid rgba(255,255,255,0.06) !important;
        margin: 0 0 10px 0 !important;
    }
"""

# ==============================================================================
# 4. STATUS BARS (WORKSPACE & INVENTORY)
# ==============================================================================

STATUS_STYLES = """
    .status-bar {
        background: var(--card-bg);
        border-radius: 8px;
        padding: 0 16px;
        margin-bottom: 8px;
        display: flex;
        align-items: center;
        height: 42px;
        border: 1px solid var(--card-border);
        box-sizing: border-box;
    }
    .status-label {
        color: var(--text-muted);
        font-size: 0.85rem;
        font-weight: 700;
        text-transform: uppercase;
        width: 140px;
        letter-spacing: 0.5px;
    }
    .status-value {
        color: var(--text-main);
        font-weight: 600;
        font-size: 0.85rem;
    }
"""

# ==============================================================================
# 5. DATA INVENTORY & TABLES
# ==============================================================================

INVENTORY_STYLES = """
    /* Text Input (Search) */
    input[type="text"], div[data-testid="stTextInput"] input {
        height: 38px !important;
        min-height: 38px !important;
        padding: 0 12px !important;
        border-radius: 8px !important;
        box-sizing: border-box !important;
    }

    /* Table Header */
    .inventory-header {
        color: var(--text-muted);
        font-weight: 800;
        font-size: 0.7rem;
        letter-spacing: 1.2px;
        text-transform: uppercase;
    }

    div[data-testid="stVerticalBlock"] > div[data-testid="stHorizontalBlock"] {
        margin-top: -10px; 
        padding: 6px 12px;
        border-radius: 8px;
        transition: background-color 0.25s ease;
    }
    
    div[data-testid="stVerticalBlock"] > div[data-testid="stHorizontalBlock"]:hover {
        background-color: rgba(255, 255, 255, 0.04);
    }

    /* Table Cells */
    .inventory-cell {
        display: flex;
        align-items: center;
        height: 32px;
        font-size: 0.85rem;
        color: var(--text-secondary);
        font-weight: 500;
        pointer-events: none; /* Let the row handle styling, don't interfere */
    }

    .inventory-cell-muted {
        color: var(--text-muted);
    }
"""

# ==============================================================================
# 6. BUTTONS & INTERACTIVITY
# ==============================================================================

BUTTON_STYLES = """
    /* All Buttons — unified glass style (like DROP DUPLICATES) */
    div[data-testid="stButton"] > button,
    div[data-testid="stDownloadButton"] > button {
        height: 36px !important;
        border-radius: 10px !important;
        font-size: 0.72rem !important;
        background: rgba(255,255,255,0.04) !important;
        border: 1px solid rgba(255,255,255,0.10) !important;
        color: #94A3B8 !important;
        transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
        font-weight: 600 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.6px !important;
        position: relative;
        z-index: 2;
        padding: 0 20px !important;
    }
    div[data-testid="stButton"] > button:hover,
    div[data-testid="stDownloadButton"] > button:hover {
        border-color: var(--accent-blue) !important;
        color: var(--text-main) !important;
        background: var(--status-neutral-bg) !important;
        box-shadow: 0 0 12px var(--status-neutral-bg) !important;
    }

    /* Primary variant — same base, just slightly stronger border */
    div[data-testid="stButton"] > button[kind="primary"],
    div[data-testid="stDownloadButton"] > button[kind="primary"] {
        background: rgba(255,255,255,0.04) !important;
        border: 1px solid rgba(59, 130, 246, 0.25) !important;
        color: #94A3B8 !important;
        font-weight: 700 !important;
        height: 36px !important;
        border-radius: 10px !important;
        box-shadow: none !important;
    }
    div[data-testid="stButton"] > button[kind="primary"]:hover,
    div[data-testid="stDownloadButton"] > button[kind="primary"]:hover {
        background: var(--status-neutral-bg) !important;
        border-color: var(--accent-blue) !important;
        color: var(--text-main) !important;
        box-shadow: 0 0 12px var(--status-neutral-bg) !important;
        transform: translateY(-1px);
    }
    div[data-testid="stButton"] > button[kind="primary"]:active,
    div[data-testid="stDownloadButton"] > button[kind="primary"]:active {
        transform: translateY(0px);
    }

    /* ---- Multiselect Tags ---- */
    span[data-baseweb="tag"] {
        background-color: var(--status-neutral-bg) !important;
        border: 1px solid rgba(91, 134, 229, 0.35) !important;
        border-radius: 6px !important;
        color: var(--status-neutral) !important;
    }
    span[data-baseweb="tag"] span[role="presentation"] {
        color: var(--status-neutral) !important;
    }
    span[data-baseweb="tag"]:hover {
        background-color: rgba(91, 134, 229, 0.3) !important;
    }
    span[data-baseweb="tag"] span[data-baseweb="tag-action"] {
        color: var(--status-neutral) !important;
    }
"""

# ==============================================================================
# 7. COMPONENTS (CARDS, ALERTS, PAGINATION)
# ==============================================================================

COMPONENT_STYLES = """
    /* HR ELEMENTS */
    hr {
        margin: 4px 0 12px 0 !important;
        border: 0 !important;
        border-top: 1px solid rgba(255,255,255,0.05) !important;
    }
    
    /* ALERTS */
    .stAlert, .stSuccess, .stInfo, .stError {
        font-size: 0.9rem !important;
    }

    /* FEATURE CARDS */
    .card-container {
        background: var(--card-bg);
        backdrop-filter: blur(var(--glass-blur));
        padding: 2.5rem 1.5rem;
        border-radius: 32px;
        border: 1px solid var(--card-border);
        text-align: center;
        height: 280px;
        width: 100%;
        display: flex;
        flex-direction: column;
        justify-content: center;
        transition: all 0.4s ease;
        cursor: pointer;
    }
    .card-blue:hover { 
        border-color: var(--accent-blue); 
        box-shadow: 0 0 25px rgba(59, 130, 246, 0.2); 
        transform: translateY(-5px); 
    }
    .card-green:hover { 
        border-color: var(--accent-green); 
        box-shadow: 0 0 25px rgba(127, 177, 53, 0.2); 
        transform: translateY(-5px); 
    }
    .card-orange:hover { 
        border-color: var(--accent-orange); 
        box-shadow: 0 0 25px rgba(242, 112, 36, 0.2); 
        transform: translateY(-5px); 
    }

    /* PAGINATION CONTROLS */
    .pagination-info {
        color: var(--text-muted);
        font-size: 0.8rem;
        font-weight: 600;
    }
"""

# ==============================================================================
# 8. AUDIT PAGE — GLASSMORPHISM, ANIMATION, CHART STYLES
# ==============================================================================

AUDIT_STYLES = """
    /* ---- Scanning / Pulse Animation ---- */
    @keyframes scanPulse {
        0%   { background-position: -200% 0; }
        100% { background-position: 200% 0; }
    }
    @keyframes glowPulse {
        0%, 100% { box-shadow: 0 0 8px var(--glow-color); }
        50%      { box-shadow: 0 0 22px var(--glow-color); }
    }

    .scan-bar {
        height: 4px;
        border-radius: 4px;
        background: linear-gradient(
            90deg,
            transparent 0%,
            var(--accent-blue) 30%,
            #60A5FA 50%,
            var(--accent-blue) 70%,
            transparent 100%
        );
        background-size: 200% 100%;
        animation: scanPulse 1.8s ease-in-out infinite;
    }

    .scan-label {
        color: var(--text-muted);
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin-bottom: 8px;
        text-align: center;
    }

    /* ---- Glassmorphism Metric Card ---- */
    .audit-metric {
        background: rgba(22, 27, 48, 0.8) !important;
        backdrop-filter: blur(12px) !important;
        -webkit-backdrop-filter: blur(12px) !important;
        border-radius: 16px;
        padding: 20px 18px;
        border: 1px solid var(--card-border);
        transition: all 0.35s ease;
        --glow-color: rgba(59, 130, 246, 0.25);
        animation: glowPulse 3s ease-in-out infinite;
    }
    .audit-metric:hover {
        transform: translateY(-3px);
        border-color: rgba(255, 255, 255, 0.15);
    }
    .audit-metric .metric-label {
        color: var(--text-muted);
        font-size: 0.7rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        margin-bottom: 8px;
    }
    .audit-metric .metric-value {
        font-size: 1.9rem;
        font-weight: 800;
        letter-spacing: -1px;
        line-height: 1.1;
    }
    .audit-metric .metric-delta {
        font-size: 0.75rem;
        font-weight: 600;
        margin-top: 6px;
    }

    /* ---- Glow Color Variants ---- */
    .audit-metric.glow-blue  { --glow-color: var(--status-neutral-bg); border-color: rgba(91, 134, 229, 0.25); }
    .audit-metric.glow-green { --glow-color: var(--status-success-bg); border-color: rgba(166, 206, 57, 0.25); }
    .audit-metric.glow-orange{ --glow-color: var(--status-warning-bg); border-color: rgba(255, 159, 67, 0.25); }
    .audit-metric.glow-red   { --glow-color: var(--status-critical-bg);  border-color: rgba(255, 91, 92, 0.25); }

    /* ---- Audit Chart Containers ---- */
    .audit-chart-title {
        color: #E0E0E0;
        font-size: 0.9rem;
        font-weight: 700;
        letter-spacing: -0.3px;
        margin-bottom: 4px;
    }
    .audit-chart-caption {
        color: #8892A0;
        font-size: 0.75rem;
        font-weight: 500;
        margin-bottom: 16px;
    }

    /* ---- Audit Data Tables ---- */
    .audit-section div[data-testid="stDataFrame"] {
        font-size: 0.85rem !important;
    }
    .audit-section div[data-testid="stDataFrame"] th {
        font-size: 0.8rem !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    /* ---- Tab Labels ---- */
    button[data-baseweb="tab"] {
        font-size: 0.9rem !important;
        font-weight: 600 !important;
    }
"""

# ==============================================================================
# 8. LOGIN PAGE STYLES
# ==============================================================================

LOGIN_STYLES = """
    /* — Floating orb background — */
    .login-bg {
        position: fixed;
        top: 0; left: 0; right: 0; bottom: 0;
        overflow: hidden;
        z-index: 0;
        pointer-events: none;
    }
    .login-orb {
        position: absolute;
        border-radius: 50%;
        filter: blur(80px);
        opacity: 0.35;
        animation-timing-function: ease-in-out;
        animation-iteration-count: infinite;
        animation-direction: alternate;
    }
    .login-orb-1 {
        width: 340px; height: 340px;
        background: radial-gradient(circle, #3b82f6 0%, transparent 70%);
        top: -5%; left: 15%;
        animation: orbFloat1 8s infinite alternate;
    }
    .login-orb-2 {
        width: 280px; height: 280px;
        background: radial-gradient(circle, #e87b35 0%, transparent 70%);
        bottom: 5%; right: 10%;
        animation: orbFloat2 10s infinite alternate;
    }
    .login-orb-3 {
        width: 200px; height: 200px;
        background: radial-gradient(circle, #7FB135 0%, transparent 70%);
        top: 50%; left: 60%;
        animation: orbFloat3 12s infinite alternate;
    }
    @keyframes orbFloat1 {
        0%   { transform: translate(0, 0) scale(1); }
        100% { transform: translate(60px, 80px) scale(1.15); }
    }
    @keyframes orbFloat2 {
        0%   { transform: translate(0, 0) scale(1); }
        100% { transform: translate(-50px, -60px) scale(1.1); }
    }
    @keyframes orbFloat3 {
        0%   { transform: translate(0, 0) scale(1); }
        100% { transform: translate(-40px, 50px) scale(1.2); }
    }

    /* — Login container — */
    .login-container {
        position: relative;
        z-index: 1;
        display: flex;
        justify-content: center;
        align-items: center;
        min-height: 40vh;
        margin-top: -1rem;
    }
    .login-card {
        background: rgba(15, 23, 42, 0.45);
        backdrop-filter: blur(40px);
        -webkit-backdrop-filter: blur(40px);
        border: 1px solid rgba(255, 255, 255, 0.12);
        border-radius: 28px;
        padding: 32px 36px 24px 36px;
        max-width: 400px;
        width: 100%;
        box-shadow: 0 20px 60px rgba(0, 0, 0, 0.5);
        animation: slideUpFade 0.6s ease-out forwards;
    }
    .login-card:hover {
        border-color: rgba(59, 130, 246, 0.3);
        box-shadow: 0 20px 80px rgba(59, 130, 246, 0.15);
        transition: 0.5s ease;
    }
    .login-title {
        color: var(--text-main);
        font-size: 1.8rem;
        font-weight: 800;
        text-align: center;
        margin-bottom: 4px;
        letter-spacing: -1px;
    }
    .login-subtitle {
        color: var(--text-secondary);
        font-size: 0.85rem;
        text-align: center;
        margin-bottom: 12px;
    }
    .login-brand {
        text-align: center;
        margin-bottom: 16px;
    }
    .login-brand-name {
        color: var(--text-main);
        font-weight: 800;
        font-size: 1.2rem;
        margin-top: 8px;
    }
    .login-brand-sub {
        color: var(--text-secondary);
        font-size: 0.6rem;
        letter-spacing: 2.5px;
        text-transform: uppercase;
        margin-top: 2px;
    }

    /* Password eye icon fix for dark theme */
    button[data-testid="stTextInputPasswordToggle"] {
        color: #94A3B8 !important;
    }
    button[data-testid="stTextInputPasswordToggle"]:hover {
        color: #CBD5E1 !important;
    }
    /* Hide browser native password reveal (Edge/Chrome duplicate eye) */
    input[type="password"]::-ms-reveal,
    input[type="password"]::-ms-clear,
    input[type="password"]::-webkit-credentials-auto-fill-button {
        display: none !important;
    }

    /* Checkbox spacing fix */
    .stCheckbox > div {
        margin-top: 4px;
        margin-bottom: 4px;
    }

    /* Sidebar user info */
    .sidebar-user-info {
        padding: 8px 16px;
        margin-bottom: 8px;
        text-align: center;
        border-bottom: 1px solid rgba(255,255,255,0.06);
    }
    .sidebar-avatar {
        width: 48px;
        height: 48px;
        border-radius: 50%;
        background: linear-gradient(135deg, var(--accent-blue), var(--accent-orange));
        display: flex;
        align-items: center;
        justify-content: center;
        margin: 0 auto 6px auto;
        font-size: 1.5rem;
        font-weight: 800;
        color: white;
        box-shadow: 0 4px 15px rgba(59, 130, 246, 0.3);
    }
    .sidebar-username {
        color: var(--text-main);
        font-weight: 700;
        font-size: 0.95rem;
    }
    .sidebar-role-badge {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 12px;
        font-size: 0.65rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-top: 4px;
    }
    .sidebar-role-admin {
        background: var(--status-warning-bg);
        color: var(--status-warning);
        border: 1px solid rgba(255, 159, 67, 0.2);
    }
    .sidebar-role-user {
        background: var(--status-neutral-bg);
        color: var(--status-neutral);
        border: 1px solid rgba(91, 134, 229, 0.2);
    }
"""

# ==============================================================================
# 9. USER MANAGEMENT STYLES
# ==============================================================================

USER_MGMT_STYLES = """
    .user-table-header {
        color: var(--text-muted);
        font-size: 0.65rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        padding: 8px 0;
    }
    .user-table-cell {
        color: var(--text-secondary);
        font-size: 0.85rem;
        padding: 10px 0;
        display: flex;
        align-items: center;
        min-height: 42px;
    }
    .user-role-badge-admin {
        background: var(--status-warning-bg);
        color: var(--status-warning);
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 0.7rem;
        font-weight: 700;
    }
    .user-role-badge-user {
        background: var(--status-neutral-bg);
        color: var(--status-neutral);
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 0.7rem;
        font-weight: 700;
    }
"""

# ==============================================================================
# SECTION - DATA AUDIT ANIMATIONS
# ==============================================================================

DATA_AUDIT_STYLES = """
    /* --- Fade-in keyframes --- */
    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(18px); }
        to   { opacity: 1; transform: translateY(0); }
    }
    @keyframes fadeIn {
        from { opacity: 0; }
        to   { opacity: 1; }
    }
    @keyframes pulseGlow {
        0%, 100% { box-shadow: 0 0 8px var(--status-neutral-bg); }
        50%      { box-shadow: 0 0 18px var(--accent-blue); }
    }
    @keyframes slideInLeft {
        from { opacity: 0; transform: translateX(-14px); }
        to   { opacity: 1; transform: translateX(0); }
    }

    /* --- Section wrapper with stagger --- */
    .audit-section { animation: fadeInUp 0.5s ease-out both; }
    .audit-section-delay { animation: fadeInUp 0.55s ease-out 0.08s both; }

    /* --- Status badge --- */
    .status-badge {
        display: inline-block;
        padding: 5px 14px;
        border-radius: 8px;
        font-size: 0.78rem;
        font-weight: 700;
        animation: fadeIn 0.4s ease-out both;
        transition: transform 0.2s ease;
    }
    .status-badge:hover { transform: scale(1.05); }
    .badge-red    { background: var(--status-critical-bg); color: var(--status-critical); }
    .badge-blue   { background: var(--status-neutral-bg); color: var(--status-neutral); }
    .badge-green  { background: var(--status-success-bg); color: var(--status-success); }
    .badge-orange { background: var(--status-warning-bg);  color: var(--status-warning); }

    /* --- Chart container with glow on hover --- */
    .chart-glow {
        border-radius: 12px;
        transition: box-shadow 0.3s ease;
    }
    .chart-glow:hover {
        box-shadow: 0 0 20px rgba(94,196,212,0.12);
    }

    /* --- Animated section header --- */
    .section-header {
        animation: fadeInUp 0.45s ease-out both;
        margin-bottom: 4px;
    }
"""

# ==============================================================================
# 10. ADMIN SETTINGS PAGE
# ==============================================================================

ADMIN_SETTINGS_STYLES = """
    /* Section header */
    .admin-section-header {
        display: flex;
        align-items: center;
        gap: 10px;
        margin-bottom: 12px;
        padding: 14px 18px;
        background: rgba(255,255,255,0.03);
        border-radius: 12px;
        border: 1px solid var(--card-border);
        animation: fadeInUp 0.45s ease-out both;
    }
    .admin-section-header span:last-child {
        color: var(--text-main);
        font-size: 1rem;
        font-weight: 700;
        letter-spacing: -0.3px;
    }
    .admin-icon {
        font-size: 1.2rem;
    }
"""

# ==============================================================================
# 11. EDA / VISUAL INTELLIGENCE STYLES
# ==============================================================================

EDA_STYLES = """
    /* --- EDA Section wrapper --- */
    .eda-section {
        animation: fadeInUp 0.5s ease-out both;
    }
    .eda-section-delay {
        animation: fadeInUp 0.55s ease-out 0.1s both;
    }

    /* --- EDA Insight Card --- */
    .eda-insight-card {
        background: rgba(255,255,255,0.025);
        border: 1px solid var(--card-border);
        border-radius: 14px;
        padding: 18px 20px;
        margin-bottom: 12px;
        transition: all 0.3s ease;
        animation: slideInLeft 0.45s ease-out both;
    }
    .eda-insight-card:hover {
        background: rgba(255,255,255,0.04);
        border-color: rgba(91, 134, 229, 0.2);
        transform: translateX(3px);
    }
    .eda-insight-title {
        color: var(--text-muted);
        font-size: 0.7rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        margin-bottom: 6px;
    }
    .eda-insight-value {
        color: var(--text-main);
        font-size: 1.3rem;
        font-weight: 800;
        letter-spacing: -0.5px;
    }
    .eda-insight-detail {
        color: var(--text-secondary);
        font-size: 0.8rem;
        font-weight: 500;
        margin-top: 4px;
    }

    /* --- EDA Filter Row --- */
    .eda-filter-row {
        display: flex;
        align-items: center;
        gap: 16px;
        padding: 10px 16px;
        background: rgba(255,255,255,0.02);
        border: 1px solid var(--card-border);
        border-radius: 12px;
        margin-bottom: 16px;
        animation: fadeIn 0.4s ease-out both;
    }
"""

# ==============================================================================
# 12. CHAT INTERFACE STYLES
# ==============================================================================

CHAT_STYLES = """
    /* Chat Popover Container */
    div[data-testid="stPopoverBody"] {
        background: rgba(15, 23, 42, 0.95) !important;
        backdrop-filter: blur(24px) !important;
        -webkit-backdrop-filter: blur(24px) !important;
        border: 1px solid rgba(59, 130, 246, 0.2) !important;
        border-radius: 16px !important;
        box-shadow: 0 20px 50px rgba(0, 0, 0, 0.6), inset 0 0 0 1px rgba(255, 255, 255, 0.05) !important;
        padding: 16px !important;
        width: 450px !important;
        max-width: 90vw !important;
    }

    /* Message Wrapper (The container) */
    div[data-testid="stChatMessage"] {
        background-color: transparent !important;
        padding: 0px;
        margin-bottom: 20px;
        border: none;
        transition: all 0.3s ease;
        display: flex;
        align-items: flex-start;
        gap: 12px;
    }

    /* Inner Bubble (User) */
    div[data-testid="stChatMessage"]:has(div[data-testid="stChatMessageAvatarUser"]) > div:nth-child(2) {
        background: rgba(59, 130, 246, 0.08) !important;
        border: 1px solid rgba(59, 130, 246, 0.15) !important;
        border-right: 3px solid var(--accent-blue) !important;
        padding: 10px 14px;
        border-radius: 12px 0 12px 12px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
        width: 100%;
        color: var(--text-main);
        font-size: 0.9rem;
    }

    /* Inner Bubble (Assistant) */
    div[data-testid="stChatMessage"]:has(div[data-testid="stChatMessageAvatarAssistant"]) > div:nth-child(2) {
        background: rgba(242, 112, 36, 0.05) !important;
        border: 1px solid rgba(242, 112, 36, 0.1) !important;
        border-left: 3px solid var(--accent-orange) !important;
        padding: 10px 14px;
        border-radius: 0 12px 12px 12px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
        width: 100%;
        color: var(--text-main);
        font-size: 0.9rem;
    }

    /* Avatars Setup */
    div[data-testid="stChatMessageAvatarUser"],
    div[data-testid="stChatMessageAvatarAssistant"] {
        background: transparent !important;
        color: transparent !important;
        width: 26px !important;
        height: 26px !important;
        min-width: 26px !important;
        border-radius: 8px !important;
        display: flex;
        align-items: center;
        justify-content: center;
        overflow: hidden;
    }
    
    /* Hide the default streamlit emojis */
    div[data-testid="stChatMessageAvatarUser"] * { display: none !important; }
    div[data-testid="stChatMessageAvatarAssistant"] * { display: none !important; }

    /* Custom SVG for User */
    div[data-testid="stChatMessageAvatarUser"] {
        background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='100%25' height='100%25' viewBox='0 0 24 24' fill='none' stroke='%233B82F6' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2'/%3E%3Ccircle cx='12' cy='7' r='4'/%3E%3C/svg%3E") !important;
        background-size: 80%;
        background-position: center;
        background-repeat: no-repeat;
        border: 1px solid rgba(59, 130, 246, 0.3) !important;
        background-color: rgba(59, 130, 246, 0.1) !important;
        box-shadow: 0 0 10px rgba(59, 130, 246, 0.2);
    }

    /* Custom SVG for Assistant (Transformers Orbit) */
    div[data-testid="stChatMessageAvatarAssistant"] {
        background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='100%25' height='100%25' viewBox='0 0 24 24' fill='none' stroke='%23F27024' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Ccircle cx='12' cy='12' r='3'/%3E%3Cpath d='M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z'/%3E%3C/svg%3E") !important;
        background-size: 80%;
        background-position: center;
        background-repeat: no-repeat;
        border: 1px solid rgba(242, 112, 36, 0.3) !important;
        background-color: rgba(242, 112, 36, 0.1) !important;
        box-shadow: 0 0 10px rgba(242, 112, 36, 0.2);
    }

    /* Chat Input */
    div[data-testid="stChatInput"] {
        background: rgba(15, 23, 42, 0.6) !important;
        border: 1px solid var(--card-border) !important;
        border-radius: 14px !important;
        transition: all 0.3s ease !important;
    }
    div[data-testid="stChatInput"]:focus-within {
        border-color: var(--accent-blue) !important;
        box-shadow: 0 0 12px rgba(59, 130, 246, 0.2) !important;
    }

    /* Thinking Animation */
    .ai-thinking {
        display: inline-flex;
        align-items: center;
        gap: 4px;
        color: var(--accent-orange);
        font-size: 0.85rem;
        font-weight: 600;
        margin-left: 10px;
        animation: pulseText 1.5s infinite;
    }
    .ai-thinking-dot {
        width: 6px;
        height: 6px;
        background-color: var(--accent-orange);
        border-radius: 50%;
        animation: blinkDot 1.4s infinite both;
    }
    .ai-thinking-dot:nth-child(2) { animation-delay: 0.2s; }
    .ai-thinking-dot:nth-child(3) { animation-delay: 0.4s; }

    @keyframes pulseText {
        0%, 100% { opacity: 0.6; }
        50% { opacity: 1; }
    }
    @keyframes blinkDot {
        0%, 80%, 100% { transform: scale(0); opacity: 0; }
        40% { transform: scale(1); opacity: 1; }
    }
        box-shadow: 0 0 12px rgba(59, 130, 246, 0.2) !important;
    }
    div[data-testid="stChatInputTextArea"] {
        color: var(--text-main) !important;
    }
    div[data-testid="stChatInputSubmitButton"] {
        color: var(--accent-blue) !important;
    }
    div[data-testid="stChatInputSubmitButton"]:hover {
        color: var(--primary-bg) !important;
        background: var(--accent-blue) !important;
        border-radius: 8px !important;
    }
"""

# ==============================================================================
# 13. OVERVIEW — FEATURE NAVIGATION
# ==============================================================================

OVERVIEW_FEATURE_STYLES = """
    /* Overview Header Boxes */
    .overview-box {
        flex: 1;
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 12px;
        padding: 24px;
        display: flex;
        align-items: flex-start;
        gap: 12px;
        transition: all 0.3s ease;
        cursor: default;
    }
    .overview-box:hover {
        background: rgba(255,255,255,0.08);
        border-color: var(--accent-orange);
        transform: translateY(-5px);
        box-shadow: 0 10px 20px rgba(0,0,0,0.2);
    }
    .overview-box-icon {
        color: var(--accent-orange);
        font-size: 1.5rem;
        transition: transform 0.3s ease;
    }
    .overview-box:hover .overview-box-icon {
        transform: scale(1.2);
    }

    /* Section Title & Divider */
    .section-title {
        text-align: center;
        color: white;
        font-size: 1.5rem;
        font-weight: 800;
        margin-top: 40px;
        margin-bottom: 5px;
    }
    .section-divider {
        width: 50px;
        height: 3px;
        background: var(--accent-orange);
        margin: 0 auto 30px auto;
        border-radius: 2px;
    }

    /* ── Journey Timeline Grid ── */
    @keyframes journey-fadeIn {
        from { opacity: 0; transform: translateY(16px); }
        to   { opacity: 1; transform: translateY(0); }
    }
    .journey-grid {
        display: flex;
        align-items: stretch;
        gap: 0;
        margin-bottom: 50px;
    }
    .journey-arrow {
        display: flex;
        align-items: center;
        justify-content: center;
        flex-shrink: 0;
        width: 36px;
        padding-bottom: 32px;
        opacity: 0.6;
    }

    /* Journey Cards */
    .journey-card {
        flex: 1;
        position: relative;
        background: rgba(255,255,255,0.025);
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 16px;
        padding: 32px 24px 28px;
        text-align: center;
        cursor: pointer;
        overflow: hidden;
        transition: all 0.35s cubic-bezier(0.4, 0, 0.2, 1);
        animation: journey-fadeIn 0.5s ease-out both;
    }
    .journey-card-1 { animation-delay: 0s; }
    .journey-card-2 { animation-delay: 0.12s; }
    .journey-card-3 { animation-delay: 0.24s; }

    /* Top accent bar */
    .journey-accent {
        position: absolute;
        top: 0;
        left: 20%;
        right: 20%;
        height: 3px;
        border-radius: 0 0 4px 4px;
        transition: all 0.35s ease;
    }
    .accent-1 { background: linear-gradient(90deg, transparent, var(--accent-blue), transparent); }
    .accent-2 { background: linear-gradient(90deg, transparent, var(--accent-green), transparent); }
    .accent-3 { background: linear-gradient(90deg, transparent, var(--accent-orange), transparent); }
    .journey-card:hover .journey-accent {
        left: 10%;
        right: 10%;
        height: 3px;
        filter: brightness(1.3);
    }

    /* Card hover states */
    .journey-card-1:hover {
        background: rgba(59,130,246,0.05);
        border-color: rgba(59,130,246,0.25);
        transform: translateY(-6px);
        box-shadow: 0 12px 32px rgba(59,130,246,0.1), 0 0 20px rgba(59,130,246,0.05);
    }
    .journey-card-2:hover {
        background: rgba(127,177,53,0.05);
        border-color: rgba(127,177,53,0.25);
        transform: translateY(-6px);
        box-shadow: 0 12px 32px rgba(127,177,53,0.1), 0 0 20px rgba(127,177,53,0.05);
    }
    .journey-card-3:hover {
        background: rgba(242,112,36,0.05);
        border-color: rgba(242,112,36,0.25);
        transform: translateY(-6px);
        box-shadow: 0 12px 32px rgba(242,112,36,0.1), 0 0 20px rgba(242,112,36,0.05);
    }

    /* Step circle */
    .journey-step {
        width: 54px;
        height: 54px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        margin: 0 auto 10px auto;
        font-size: 1.35rem;
        font-weight: 800;
        color: white;
        transition: transform 0.35s cubic-bezier(0.34, 1.56, 0.64, 1),
                    box-shadow 0.35s ease;
    }
    .step-1 {
        background: linear-gradient(135deg, var(--accent-blue), #2563EB);
        box-shadow: 0 0 20px rgba(59,130,246,0.3), 0 0 40px rgba(59,130,246,0.08);
    }
    .step-2 {
        background: linear-gradient(135deg, var(--accent-green), #5F9020);
        box-shadow: 0 0 20px rgba(127,177,53,0.3), 0 0 40px rgba(127,177,53,0.08);
    }
    .step-3 {
        background: linear-gradient(135deg, var(--accent-orange), #D95A15);
        box-shadow: 0 0 20px rgba(242,112,36,0.3), 0 0 40px rgba(242,112,36,0.08);
    }
    .journey-card:hover .journey-step {
        transform: scale(1.12);
    }
    .journey-card-1:hover .step-1 {
        box-shadow: 0 0 24px rgba(59,130,246,0.5), 0 0 48px rgba(59,130,246,0.15);
    }
    .journey-card-2:hover .step-2 {
        box-shadow: 0 0 24px rgba(127,177,53,0.5), 0 0 48px rgba(127,177,53,0.15);
    }
    .journey-card-3:hover .step-3 {
        box-shadow: 0 0 24px rgba(242,112,36,0.5), 0 0 48px rgba(242,112,36,0.15);
    }

    /* Step label */
    .journey-label {
        font-size: 0.6rem;
        font-weight: 800;
        letter-spacing: 2px;
        text-transform: uppercase;
        margin-bottom: 14px;
        opacity: 0.75;
    }
    .label-1 { color: var(--accent-blue); }
    .label-2 { color: var(--accent-green); }
    .label-3 { color: var(--accent-orange); }

    .journey-title {
        color: white;
        font-size: 1.15rem;
        font-weight: 700;
        margin-bottom: 10px;
        letter-spacing: -0.3px;
    }
    .journey-desc {
        color: var(--text-secondary);
        font-size: 0.85rem;
        line-height: 1.65;
        text-wrap: balance;
    }

    .two-columns {
        display: grid;
        grid-template-columns: 1fr 1.2fr;
        gap: 40px;
        margin-bottom: 50px;
        align-items: flex-start;
    }

    .column-wrapper {
        display: flex;
        flex-direction: column;
    }

    .column-boxes {
        display: flex;
        flex-direction: column;
        gap: 16px;
    }

    .header-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        height: 32px;
        margin-bottom: 24px;
    }
    .col-title {
        color: white;
        font-size: 1.5rem;
        font-weight: bold;
    }
    .records-badge {
        background: rgba(255,255,255,0.1);
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 800;
        color: white;
        letter-spacing: 1px;
        display: flex;
        align-items: center;
    }

    /* Anatomy Box */
    .anatomy-box {
        background: linear-gradient(90deg, var(--status-warning-bg) 0%, transparent 100%);
        border: 1px solid rgba(255,255,255,0.05);
        border-left: 4px solid var(--accent-orange);
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 0;
        transition: all 0.3s ease;
    }
    .anatomy-box:hover {
        background: linear-gradient(90deg, var(--status-warning-bg) 0%, rgba(255,255,255,0.02) 100%);
        transform: translateX(5px);
        border-color: var(--accent-orange);
        box-shadow: 0 0 15px var(--status-warning-bg);
    }
    .anatomy-box-green { 
        background: linear-gradient(90deg, var(--status-success-bg) 0%, transparent 100%);
        border-left-color: var(--accent-green); 
    }
    .anatomy-box-green:hover { 
        background: linear-gradient(90deg, var(--status-success-bg) 0%, rgba(255,255,255,0.02) 100%);
        border-color: var(--accent-green); box-shadow: 0 0 15px var(--status-success-bg); 
    }

    .anatomy-box-blue { 
        background: linear-gradient(90deg, var(--status-neutral-bg) 0%, transparent 100%);
        border-left-color: var(--accent-blue); 
    }
    .anatomy-box-blue:hover { 
        background: linear-gradient(90deg, var(--status-neutral-bg) 0%, rgba(255,255,255,0.02) 100%);
        border-color: var(--accent-blue); box-shadow: 0 0 15px var(--status-neutral-bg); 
    }

    .anatomy-box-amber { 
        background: linear-gradient(90deg, var(--status-warning-bg) 0%, transparent 100%);
        border-left-color: var(--accent-orange); 
    }
    .anatomy-box-amber:hover { 
        background: linear-gradient(90deg, var(--status-warning-bg) 0%, rgba(255,255,255,0.02) 100%);
        border-color: var(--accent-orange); box-shadow: 0 0 15px var(--status-warning-bg); 
    }

    .anatomy-box-red { 
        background: linear-gradient(90deg, var(--status-critical-bg) 0%, transparent 100%);
        border-left-color: var(--accent-red); 
    }
    .anatomy-box-red:hover { 
        background: linear-gradient(90deg, var(--status-critical-bg) 0%, rgba(255,255,255,0.02) 100%);
        border-color: var(--accent-red); box-shadow: 0 0 15px var(--status-critical-bg); 
    }

    .anatomy-title {
        color: var(--accent-orange);
        font-size: 0.75rem;
        font-weight: 800;
        letter-spacing: 1px;
        text-transform: uppercase;
        margin-bottom: 12px;
    }
    .anatomy-title-green { color: var(--accent-green); }
    .anatomy-title-blue { color: var(--accent-blue); }
    .anatomy-title-red { color: var(--accent-red); }
    .anatomy-title-amber { color: var(--accent-orange); }

    .anatomy-tags {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
    }
    .anatomy-tag {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 4px;
        padding: 6px 12px;
        font-size: 0.85rem;
        color: white;
    }
    .anatomy-tag-red {
        background: var(--status-critical-bg);
        border-color: rgba(255, 91, 92, 0.3);
        color: var(--status-critical);
        font-weight: bold;
    }
    .anatomy-tag-orange {
        background: var(--status-warning-bg);
        border-color: rgba(245, 158, 11, 0.35);
        color: var(--accent-orange);
        font-weight: 600;
    }
    .anatomy-tag-orange::before {
        content: '●';
        margin-right: 6px;
        font-size: 0.55rem;
        vertical-align: middle;
        color: var(--accent-orange);
    }
    .anatomy-tag-green {
        background: var(--status-success-bg);
        border-color: rgba(166, 206, 57, 0.35);
        color: var(--accent-green);
        font-weight: 600;
    }
    .anatomy-tag-green::before {
        content: '●';
        margin-right: 6px;
        font-size: 0.55rem;
        vertical-align: middle;
        color: var(--accent-green);
    }
    .anatomy-tag-blue {
        background: var(--status-neutral-bg);
        border-color: rgba(91, 134, 229, 0.35);
        color: var(--accent-blue);
        font-weight: 600;
    }
    .anatomy-tag-blue::before {
        content: '●';
        margin-right: 6px;
        font-size: 0.55rem;
        vertical-align: middle;
        color: var(--accent-blue);
    }
    .anatomy-note {
        font-size: 0.8rem;
        color: var(--text-muted);
        font-style: italic;
        margin-top: 10px;
    }

    /* Objective Card */
    .objective-card {
        background: rgba(255,255,255,0.02);
        border: 1px solid rgba(255,255,255,0.05);
        border-radius: 12px;
        padding: 16px;
        margin-bottom: 0;
        display: flex;
        align-items: flex-start;
        gap: 16px;
        transition: all 0.3s ease;
        cursor: pointer;
    }
    .objective-card-blue:hover {
        background: rgba(255,255,255,0.06);
        transform: translateY(-3px);
        border-color: var(--accent-blue);
        box-shadow: 0 4px 20px var(--status-neutral-bg);
    }
    .objective-card-green:hover {
        background: rgba(255,255,255,0.06);
        transform: translateY(-3px);
        border-color: var(--accent-green);
        box-shadow: 0 4px 20px var(--status-success-bg);
    }
    .objective-card-orange:hover {
        background: rgba(255,255,255,0.06);
        transform: translateY(-3px);
        border-color: var(--accent-orange);
        box-shadow: 0 4px 20px var(--status-warning-bg);
    }
    .objective-card-purple:hover {
        background: rgba(255,255,255,0.06);
        transform: translateY(-3px);
        border-color: #8B5CF6;
        box-shadow: 0 4px 20px rgba(139,92,246,0.15);
    }
    .objective-card-red:hover {
        background: rgba(255,255,255,0.06);
        transform: translateY(-3px);
        border-color: var(--accent-red);
        box-shadow: 0 4px 20px var(--status-critical-bg);
    }

    .objective-icon {
        width: 40px;
        height: 40px;
        border-radius: 8px;
        display: flex;
        align-items: center;
        justify-content: center;
        flex-shrink: 0;
        font-size: 1.2rem;
    }
    .icon-blue { background: var(--status-neutral-bg); color: var(--accent-blue); }
    .icon-green { background: var(--status-success-bg); color: var(--accent-green); }
    .icon-orange { background: var(--status-warning-bg); color: var(--accent-orange); }
    .icon-purple { background: rgba(139,92,246,0.1); color: #8B5CF6; }
    .icon-red { background: var(--status-critical-bg); color: var(--accent-red); }

    .objective-title {
        color: white;
        font-size: 1rem;
        font-weight: bold;
        margin-bottom: 4px;
    }
    .objective-desc {
        color: var(--text-secondary);
        font-size: 0.85rem;
        line-height: 1.5;
    }

    /* ── Premium Numbered Objective Cards ── */
    @keyframes obj-fadein {
        from { opacity: 0; transform: translateY(10px); }
        to   { opacity: 1; transform: translateY(0); }
    }
    .obj-card-premium {
        display: flex;
        align-items: flex-start;
        gap: 16px;
        padding: 18px 20px;
        background: linear-gradient(135deg, rgba(59,130,246,0.07) 0%, rgba(255,255,255,0.02) 100%);
        border: 1px solid rgba(59,130,246,0.15);
        border-left: 3px solid rgba(59,130,246,0.5);
        border-radius: 12px;
        cursor: default;
        transition: all 0.28s cubic-bezier(0.4,0,0.2,1);
        animation: obj-fadein 0.4s ease-out both;
        position: relative;
        overflow: hidden;
    }
    .obj-card-premium::before {
        content: '';
        position: absolute;
        inset: 0;
        background: linear-gradient(90deg, rgba(59,130,246,0.08), transparent);
        opacity: 0;
        transition: opacity 0.28s ease;
        border-radius: 12px;
        pointer-events: none;
    }
    .obj-card-premium:hover {
        background: linear-gradient(135deg, rgba(59,130,246,0.14) 0%, rgba(255,255,255,0.04) 100%);
        border-color: rgba(59,130,246,0.55);
        border-left-color: var(--accent-blue);
        transform: translateX(5px);
        box-shadow: 0 6px 24px rgba(59,130,246,0.18), -4px 0 16px rgba(59,130,246,0.12);
    }
    .obj-card-premium:hover::before {
        opacity: 1;
    }
    .obj-num {
        flex-shrink: 0;
        width: 36px;
        height: 36px;
        border-radius: 50%;
        background: linear-gradient(135deg, var(--accent-blue) 0%, rgba(91,134,229,0.6) 100%);
        color: white;
        font-size: 1rem;
        font-weight: 900;
        display: flex;
        align-items: center;
        justify-content: center;
        box-shadow: 0 0 14px rgba(59,130,246,0.4);
        transition: transform 0.28s ease, box-shadow 0.28s ease;
    }
    .obj-card-premium:hover .obj-num {
        transform: scale(1.12);
        box-shadow: 0 0 22px rgba(59,130,246,0.7);
    }
    .obj-icon-wrap {
        flex-shrink: 0;
        display: flex;
        align-items: center;
        justify-content: center;
        width: 36px;
        height: 36px;
        border-radius: 8px;
        background: rgba(59,130,246,0.1);
        border: 1px solid rgba(59,130,246,0.2);
        transition: background 0.28s ease;
    }
    .obj-card-premium:hover .obj-icon-wrap {
        background: rgba(59,130,246,0.2);
    }
    .obj-content { flex: 1; }
    .obj-title {
        color: white;
        font-size: 0.95rem;
        font-weight: 700;
        margin-bottom: 5px;
        letter-spacing: -0.2px;
    }
    .obj-desc {
        color: var(--text-secondary);
        font-size: 0.83rem;
        line-height: 1.6;
    }
    .obj-desc b { color: rgba(255,255,255,0.85); }
"""

# ==============================================================================
# PREPROCESSING PAGE STYLES
# ==============================================================================

PREPROCESSING_STYLES = """
    /* ── Badge hover ── */
    .pp-badge {
        transition: transform 0.18s ease, filter 0.18s ease, box-shadow 0.18s ease;
    }
    .pp-badge:hover {
        transform: translateY(-2px) scale(1.06);
        filter: brightness(1.25);
        box-shadow: 0 4px 14px rgba(0,0,0,0.25);
    }

    /* ── Info card hover ── */
    .pp-info-card {
        transition: transform 0.22s cubic-bezier(0.25,0.8,0.25,1),
                    box-shadow 0.22s ease,
                    background 0.22s ease;
    }
    .pp-info-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0,0,0,0.18);
        background: rgba(255,255,255,0.06) !important;
    }

    /* ── Step circle glow pulse ── */
    .pp-step-circle {
        transition: transform 0.35s cubic-bezier(0.34,1.56,0.64,1),
                    box-shadow 0.35s ease;
    }
    .pp-step-circle:hover {
        transform: rotate(12deg) scale(1.2);
    }

    /* ── Pipeline summary card ── */
    .pp-pipeline-card {
        transition: box-shadow 0.3s ease, border-color 0.3s ease;
    }
    .pp-pipeline-card:hover {
        box-shadow: 0 8px 32px rgba(0,0,0,0.22);
        border-color: rgba(255,255,255,0.18) !important;
    }

    /* ── Metric mini-cards shimmer on hover ── */
    .pp-metric-card {
        transition: transform 0.2s ease, box-shadow 0.2s ease, background 0.2s ease;
    }
    .pp-metric-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 6px 18px rgba(0,0,0,0.2);
        background: rgba(255,255,255,0.08) !important;
    }

    /* ── Run button pulse animation ── */
    @keyframes pp-pulse {
        0%   { box-shadow: 0 0 0 0 rgba(91,134,229,0.55); }
        70%  { box-shadow: 0 0 0 10px rgba(91,134,229,0); }
        100% { box-shadow: 0 0 0 0 rgba(91,134,229,0); }
    }
    [data-testid="stButton"] button[kind="primary"] {
        animation: pp-pulse 2.2s infinite;
        transition: transform 0.15s ease, filter 0.15s ease;
    }
    [data-testid="stButton"] button[kind="primary"]:hover {
        animation: none;
        transform: translateY(-2px);
        filter: brightness(1.12);
    }
    [data-testid="stButton"] button:not([kind="primary"]):hover {
        transform: translateY(-1px);
        filter: brightness(1.08);
    }

    /* ── Completion banner entry animation ── */
    @keyframes pp-slidein {
        from { opacity: 0; transform: translateY(12px); }
        to   { opacity: 1; transform: translateY(0); }
    }
    .pp-done-banner {
        animation: pp-slidein 0.4s ease forwards;
    }

    /* ── Outlier Treatment: Safe Zone legend card ── */
    /* Animated gradient border via pseudo-element shimmer */
    @keyframes pp-safezone-shimmer {
        0%   { background-position: 0% 50%; }
        50%  { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    .pp-legend-safezone {
        position: relative;
        border-radius: 12px;
        padding: 2px;  /* border thickness */
        background: linear-gradient(135deg, #F59E0B, #EF4444, #EC4899, #8B5CF6);
        background-size: 300% 300%;
        animation: pp-safezone-shimmer 4s ease infinite;
        box-shadow: 0 0 18px rgba(245,158,11,0.35), 0 0 36px rgba(139,92,246,0.18);
        transition: box-shadow 0.3s ease, transform 0.3s ease;
    }
    .pp-legend-safezone:hover {
        box-shadow: 0 0 28px rgba(245,158,11,0.55), 0 0 56px rgba(139,92,246,0.3);
        transform: translateY(-2px);
    }
    .pp-legend-safezone-inner {
        background: #0F172A;  /* same as primary-bg */
        border-radius: 10px;
        padding: 14px 18px;
        min-height: 100%;
    }

    /* ── Outlier Treatment: Pipeline Note callout ── */
    .pp-outlier-note {
        display: flex;
        align-items: flex-start;
        gap: 12px;
        background: linear-gradient(135deg, rgba(245,158,11,0.12) 0%, rgba(239,68,68,0.08) 100%);
        border: 1px solid rgba(245,158,11,0.35);
        border-left: 4px solid #F59E0B;
        border-radius: 10px;
        padding: 14px 18px;
        margin-bottom: 20px;
        font-size: 0.86rem;
        line-height: 1.65;
        color: var(--text-secondary);
    }
    .pp-outlier-note-icon {
        font-size: 1.2rem;
        line-height: 1;
        flex-shrink: 0;
        margin-top: 2px;
    }
    .pp-outlier-note strong {
        color: #FDE68A;
        font-weight: 700;
    }
"""


def apply_style(theme="dark_glass"):
    """
    Centralized UI Configuration.
    Combines all modular style definitions into a single stylesheet.
    """
    global_styles = get_global_styles(theme)
    st.markdown(f"""
        <style>
        {global_styles}
        {HEADER_STYLES}
        {SIDEBAR_STYLES}
        {STATUS_STYLES}
        {INVENTORY_STYLES}
        {BUTTON_STYLES}
        {COMPONENT_STYLES}
        {AUDIT_STYLES}
        {LOGIN_STYLES}
        {USER_MGMT_STYLES}
        {DATA_AUDIT_STYLES}
        {ADMIN_SETTINGS_STYLES}
        {EDA_STYLES}
        {CHAT_STYLES}
        {OVERVIEW_FEATURE_STYLES}
        {PREPROCESSING_STYLES}
        </style>
    """, unsafe_allow_html=True)

