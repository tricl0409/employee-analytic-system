"""
login.py — Dedicated Login Page
"""

import base64
import streamlit as st
from modules.ui import login_form, footer
from modules.utils.localization import get_text
from modules.core.auth_engine import AuthEngine


def set_video_background(video_path: str) -> None:
    """
    Injects a full-screen looping video background via base64-encoded MP4.
    A semi-transparent dark overlay keeps the login form legible.
    """
    try:
        with open(video_path, "rb") as f:
            video_b64 = base64.b64encode(f.read()).decode()
        st.markdown(
            f"""
            <style>
            /* Remove Streamlit's default page background */
            .stApp {{
                background: transparent !important;
            }}
            /* Full-screen video container */
            #login-video-bg {{
                position: fixed;
                top: 0; left: 0;
                width: 100vw; height: 100vh;
                z-index: -2;
                overflow: hidden;
                pointer-events: none;
            }}
            #login-video-bg video {{
                width: 100%; height: 100%;
                object-fit: cover;
            }}
            /* Dark overlay for readability */
            #login-video-overlay {{
                position: fixed;
                top: 0; left: 0;
                width: 100vw; height: 100vh;
                background: rgba(0, 0, 0, 0.55);
                z-index: -1;
                pointer-events: none;
            }}
            </style>
            <div id="login-video-bg">
                <video autoplay muted loop playsinline>
                    <source src="data:video/mp4;base64,{video_b64}" type="video/mp4">
                </video>
            </div>
            <div id="login-video-overlay"></div>
            """,
            unsafe_allow_html=True,
        )
    except FileNotFoundError:
        pass  # Silently fall back to no background


def main():
    """
    Login page — renders the glassmorphism login form and handles authentication.
    """
    lang = st.session_state.get('lang', 'en')
    
    set_video_background("assets/login_background.mp4")

    username, password, login_clicked = login_form()

    if login_clicked:
        if username and password:
            user = AuthEngine.login(username, password)
            if user:
                st.session_state.authenticated = True
                st.session_state.username = user['username']
                st.session_state.display_name = user['display_name']
                st.session_state.user_role = user['role']
                st.session_state.avatar_url = user.get('avatar_url', '')
                st.toast(get_text('login_success', lang), icon="✅")
                st.rerun()
            else:
                st.error(get_text('login_failed', lang))
        else:
            st.error(get_text('login_failed', lang))

    footer()


if __name__ == "__main__":
    main()
