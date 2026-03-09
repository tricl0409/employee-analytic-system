"""
auth_engine.py — User Authentication & Authorization Engine

Improvements vs. previous version:
  S-1  SHA-256 → PBKDF2-HMAC-SHA256 (310k iterations, built-in hashlib)
  S-2  delete_user blocks if target is the last admin
  S-3  update_user_role blocks admin self-demotion
  S-4  list_users / get_user never return password_hash / password_salt
  C-1  create_user / reset_password enforce min password length (≥ 6)
  C-2  update_user_role returns False when username not found
  U-1  last_login column: added via safe ALTER TABLE, updated on login()
"""

import sqlite3
import hashlib
import hmac
import secrets
import os
from datetime import datetime
from typing import Optional, Dict, List
import streamlit as st


# ==============================================================================
# CONFIG
# ==============================================================================
DB_PATH      = os.path.join("data", "system.db")
_OLD_DB_PATH = os.path.join("data", "users.db")

_PBKDF2_ITERS    = 310_000          # OWASP 2023 recommended minimum
_MIN_PASSWORD_LEN = 6


# ==============================================================================
# DATABASE CONNECTION
# ==============================================================================
@st.cache_resource
def _get_db_connection():
    """Returns a cached SQLite connection (singleton per Streamlit process)."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    # Auto-migrate: rename old users.db → system.db if needed
    if os.path.exists(_OLD_DB_PATH) and not os.path.exists(DB_PATH):
        os.rename(_OLD_DB_PATH, DB_PATH)
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


# ==============================================================================
# PASSWORD HASHING  (S-1: PBKDF2-HMAC-SHA256)
# ==============================================================================
def _hash_password(password: str, salt: str = None) -> tuple:
    """
    Hashes a password with PBKDF2-HMAC-SHA256 + random salt.
    Returns (hex_hash, hex_salt).
    """
    if salt is None:
        salt = secrets.token_hex(16)
    dk = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt.encode("utf-8"),
        _PBKDF2_ITERS,
    )
    return dk.hex(), salt


def _verify_password(password: str, stored_hash: str, salt: str) -> bool:
    """Constant-time comparison to prevent timing attacks."""
    candidate_hash, _ = _hash_password(password, salt)
    return hmac.compare_digest(candidate_hash, stored_hash)


# ==============================================================================
# HELPERS
# ==============================================================================
_USER_SAFE_COLS = (
    "id, username, display_name, role, avatar_url, created_at, updated_at, last_login"
)


def _row_to_dict(row) -> Dict:
    """Convert a sqlite3.Row to dict, excluding sensitive fields."""
    return {k: row[k] for k in row.keys()
            if k not in ("password_hash", "password_salt")}


# ==============================================================================
# AUTH ENGINE
# ==============================================================================
class AuthEngine:
    """Static class for all authentication and user management operations."""

    # ------------------------------------------------------------------
    # DB INIT
    # ------------------------------------------------------------------
    @staticmethod
    def init_db():
        """Creates / migrates the users table and seeds the default admin."""
        conn = _get_db_connection()
        cursor = conn.cursor()

        # Create table (includes last_login from the start for new installs)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                username      TEXT UNIQUE NOT NULL,
                display_name  TEXT NOT NULL DEFAULT '',
                role          TEXT NOT NULL DEFAULT 'user',
                password_hash TEXT NOT NULL,
                password_salt TEXT NOT NULL,
                avatar_url    TEXT DEFAULT '',
                created_at    TEXT NOT NULL,
                updated_at    TEXT NOT NULL,
                last_login    TEXT DEFAULT NULL
            )
        """)

        # U-1: Safe migration — add last_login to existing installs
        try:
            cursor.execute("ALTER TABLE users ADD COLUMN last_login TEXT DEFAULT NULL")
        except sqlite3.OperationalError:
            pass  # Column already exists

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS analysis_rules (
                rule_id    INTEGER PRIMARY KEY AUTOINCREMENT,
                rule_key   TEXT UNIQUE NOT NULL,
                rule_value TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """)
        conn.commit()

        # Seed default admin accounts (INSERT OR IGNORE — safe on existing DBs)
        now = datetime.now().isoformat()
        default_admins = [
            ("admin",     "Administrator", "admin123"),
            ("kina",      "Kina",          "admin123"),
            ("hongquan",  "Hong Quan",     "admin123"),
            ("ducthuan",  "Duc Thuan",     "admin123"),
        ]
        with conn:
            for uname, dname, pwd in default_admins:
                # Only insert if username doesn't already exist
                cursor.execute("SELECT id FROM users WHERE username = ?", (uname,))
                if cursor.fetchone() is None:
                    pw_hash, salt = _hash_password(pwd)
                    cursor.execute(
                        "INSERT INTO users "
                        "(username, display_name, role, password_hash, password_salt, avatar_url, created_at, updated_at) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                        (uname, dname, "admin", pw_hash, salt, "", now, now)
                    )

    # ------------------------------------------------------------------
    # LOGIN / LOGOUT
    # ------------------------------------------------------------------
    @staticmethod
    def login(username: str, password: str) -> Optional[Dict]:
        """
        Authenticates a user. Returns safe user-info dict on success, None on failure.
        Also records last_login timestamp (U-1).
        """
        conn = _get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users WHERE username = ?", (username,))
        row = cursor.fetchone()

        if row and _verify_password(password, row["password_hash"], row["password_salt"]):
            # U-1: record last login
            now = datetime.now().isoformat()
            with conn:
                cursor.execute(
                    "UPDATE users SET last_login = ? WHERE username = ?",
                    (now, username)
                )
            return {
                "id":           row["id"],
                "username":     row["username"],
                "display_name": row["display_name"],
                "role":         row["role"],
                "avatar_url":   row["avatar_url"],
            }
        return None

    @staticmethod
    def logout():
        """Clears authentication state from session."""
        st.session_state.authenticated = False
        for key in ["username", "user_role", "display_name", "avatar_url",
                    "editing_user", "show_add_user", "preview_df", "preview_name"]:
            st.session_state.pop(key, None)

    # ------------------------------------------------------------------
    # PASSWORD MANAGEMENT
    # ------------------------------------------------------------------
    @staticmethod
    def change_password(username: str, old_password: str, new_password: str) -> bool:
        """Changes a user's password after verifying old password. Returns True on success."""
        # C-1: enforce minimum length
        if len(new_password) < _MIN_PASSWORD_LEN:
            return False

        conn = _get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT password_hash, password_salt FROM users WHERE username = ?", (username,))
        row = cursor.fetchone()

        if not row or not _verify_password(old_password, row["password_hash"], row["password_salt"]):
            return False

        new_hash, new_salt = _hash_password(new_password)
        now = datetime.now().isoformat()
        with conn:
            cursor.execute(
                "UPDATE users SET password_hash = ?, password_salt = ?, updated_at = ? WHERE username = ?",
                (new_hash, new_salt, now, username)
            )
        return True

    @staticmethod
    def reset_password(username: str, new_password: str) -> tuple:
        """
        Admin-level password reset (no old password required).
        Returns (True, None) on success, (False, error_key) on failure.
        C-1: enforces minimum password length.
        """
        if len(new_password) < _MIN_PASSWORD_LEN:
            return False, "password_too_short"

        conn = _get_db_connection()
        cursor = conn.cursor()
        new_hash, new_salt = _hash_password(new_password)
        now = datetime.now().isoformat()
        with conn:
            cursor.execute(
                "UPDATE users SET password_hash = ?, password_salt = ?, updated_at = ? WHERE username = ?",
                (new_hash, new_salt, now, username)
            )
        return True, None

    # ------------------------------------------------------------------
    # PROFILE
    # ------------------------------------------------------------------
    @staticmethod
    def update_profile(username: str, display_name: str = None, avatar_url: str = None) -> bool:
        """Updates a user's display name and/or avatar. Returns True on success."""
        conn = _get_db_connection()
        cursor = conn.cursor()
        now = datetime.now().isoformat()

        updates, values = [], []
        if display_name is not None:
            updates.append("display_name = ?")
            values.append(display_name)
        if avatar_url is not None:
            updates.append("avatar_url = ?")
            values.append(avatar_url)

        if not updates:
            return False

        updates.append("updated_at = ?")
        values.extend([now, username])

        with conn:
            cursor.execute(
                f"UPDATE users SET {', '.join(updates)} WHERE username = ?",
                values
            )

        # Mirror changes into active session if editing own profile
        if st.session_state.get("username") == username:
            if display_name is not None:
                st.session_state.display_name = display_name
            if avatar_url is not None:
                st.session_state.avatar_url = avatar_url

        return True

    # ------------------------------------------------------------------
    # USER CRUD
    # ------------------------------------------------------------------
    @staticmethod
    def create_user(username: str, password: str, display_name: str = "", role: str = "user") -> tuple:
        """
        Creates a new user. Returns (True, None) on success, (False, error_key) on failure.
        C-1: password must be ≥ _MIN_PASSWORD_LEN characters.
        """
        # C-1: password strength
        if len(password) < _MIN_PASSWORD_LEN:
            return False, "password_too_short"

        conn = _get_db_connection()
        cursor = conn.cursor()

        # Duplicate username check
        cursor.execute("SELECT id FROM users WHERE username = ?", (username,))
        if cursor.fetchone() is not None:
            return False, "user_exists"

        pw_hash, salt = _hash_password(password)
        now = datetime.now().isoformat()
        with conn:
            cursor.execute(
                "INSERT INTO users "
                "(username, display_name, role, password_hash, password_salt, avatar_url, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (username, display_name or username, role, pw_hash, salt, "", now, now)
            )
        return True, None

    @staticmethod
    def delete_user(username: str, current_username: str) -> tuple:
        """
        Deletes a user. Returns (True, None) on success, (False, error_key) on failure.
        S-2: cannot delete the last admin.
        """
        if username == current_username:
            return False, "cannot_delete_self"

        conn = _get_db_connection()
        cursor = conn.cursor()

        # S-2: prevent deleting last admin
        cursor.execute("SELECT role FROM users WHERE username = ?", (username,))
        row = cursor.fetchone()
        if row and row["role"] == "admin":
            cursor.execute("SELECT COUNT(*) FROM users WHERE role = 'admin'")
            if cursor.fetchone()[0] <= 1:
                return False, "cannot_delete_last_admin"

        with conn:
            cursor.execute("DELETE FROM users WHERE username = ?", (username,))
        return True, None

    @staticmethod
    def update_user_role(username: str, new_role: str, current_username: str = "") -> tuple:
        """
        Updates a user's role. Returns (True, None) on success, (False, error_key) on failure.
        S-3: admin cannot demote themselves.
        C-2: returns False if username not found.
        """
        # S-3: block self-demotion
        if username == current_username and new_role != "admin":
            return False, "cannot_demote_self"

        conn = _get_db_connection()
        cursor = conn.cursor()
        now = datetime.now().isoformat()
        with conn:
            cursor.execute(
                "UPDATE users SET role = ?, updated_at = ? WHERE username = ?",
                (new_role, now, username)
            )
        # C-2: check actual DB impact
        return (True, None) if cursor.rowcount > 0 else (False, "user_not_found")

    @staticmethod
    def list_users(search_query: str = "") -> List[Dict]:
        """
        Returns all users, optionally filtered by search query.
        S-4: never includes password_hash or password_salt.
        """
        conn = _get_db_connection()
        cursor = conn.cursor()
        if search_query:
            cursor.execute(
                f"SELECT {_USER_SAFE_COLS} FROM users "
                "WHERE username LIKE ? OR display_name LIKE ? ORDER BY created_at DESC",
                (f"%{search_query}%", f"%{search_query}%"),
            )
        else:
            cursor.execute(f"SELECT {_USER_SAFE_COLS} FROM users ORDER BY created_at DESC")

        return [dict(r) for r in cursor.fetchall()]

    @staticmethod
    def get_user(username: str) -> Optional[Dict]:
        """
        Returns a single user by username.
        S-4: never includes password_hash or password_salt.
        """
        conn = _get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            f"SELECT {_USER_SAFE_COLS} FROM users WHERE username = ?",
            (username,)
        )
        row = cursor.fetchone()
        return dict(row) if row else None
