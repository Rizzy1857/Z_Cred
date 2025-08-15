import sqlite3
import hashlib
import secrets
from datetime import datetime, timedelta
import streamlit as st

# Authentication database
AUTH_DB = 'auth.db'

def init_auth_db():
    """Initialize the authentication database."""
    conn = sqlite3.connect(AUTH_DB)
    cursor = conn.cursor()
    
    # Users table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            user_id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            phone TEXT,
            password_hash TEXT NOT NULL,
            salt TEXT NOT NULL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            last_login TEXT,
            is_active INTEGER DEFAULT 1
        )
    """)
    
    # Sessions table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            session_id TEXT PRIMARY KEY,
            user_id INTEGER,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            expires_at TEXT,
            is_active INTEGER DEFAULT 1,
            FOREIGN KEY (user_id) REFERENCES users (user_id)
        )
    """)
    
    conn.commit()
    conn.close()

def hash_password(password, salt=None):
    """Hash password with salt."""
    if salt is None:
        salt = secrets.token_hex(32)
    
    pwd_hash = hashlib.pbkdf2_hmac('sha256', 
                                  password.encode('utf-8'), 
                                  salt.encode('utf-8'), 
                                  100000)
    return pwd_hash.hex(), salt

def register_user(name, email, phone, password):
    """Register a new user."""
    conn = sqlite3.connect(AUTH_DB)
    cursor = conn.cursor()
    
    # Check if email already exists
    cursor.execute("SELECT email FROM users WHERE email = ?", (email,))
    if cursor.fetchone():
        conn.close()
        return False, "Email already registered"
    
    # Hash password
    pwd_hash, salt = hash_password(password)
    
    try:
        cursor.execute("""
            INSERT INTO users (name, email, phone, password_hash, salt)
            VALUES (?, ?, ?, ?, ?)
        """, (name, email, phone, pwd_hash, salt))
        
        user_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        return True, user_id
    except Exception as e:
        conn.close()
        return False, str(e)

def authenticate_user(email, password):
    """Authenticate user login."""
    conn = sqlite3.connect(AUTH_DB)
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT user_id, name, password_hash, salt, is_active 
        FROM users WHERE email = ?
    """, (email,))
    
    user = cursor.fetchone()
    conn.close()
    
    if not user or not user[4]:  # Not found or inactive
        return False, None
    
    user_id, name, stored_hash, salt, is_active = user
    
    # Verify password
    pwd_hash, _ = hash_password(password, salt)
    
    if pwd_hash == stored_hash:
        # Update last login
        conn = sqlite3.connect(AUTH_DB)
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE users SET last_login = ? WHERE user_id = ?
        """, (datetime.now().isoformat(), user_id))
        conn.commit()
        conn.close()
        
        return True, {'user_id': user_id, 'name': name, 'email': email}
    
    return False, None

def create_session(user_id):
    """Create a new session for user."""
    session_id = secrets.token_urlsafe(32)
    expires_at = datetime.now() + timedelta(hours=24)
    
    conn = sqlite3.connect(AUTH_DB)
    cursor = conn.cursor()
    
    cursor.execute("""
        INSERT INTO sessions (session_id, user_id, expires_at)
        VALUES (?, ?, ?)
    """, (session_id, user_id, expires_at.isoformat()))
    
    conn.commit()
    conn.close()
    
    return session_id

def validate_session(session_id):
    """Validate if session is active and not expired."""
    if not session_id:
        return False, None
    
    conn = sqlite3.connect(AUTH_DB)
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT s.user_id, u.name, u.email, s.expires_at 
        FROM sessions s
        JOIN users u ON s.user_id = u.user_id
        WHERE s.session_id = ? AND s.is_active = 1
    """, (session_id,))
    
    session = cursor.fetchone()
    conn.close()
    
    if not session:
        return False, None
    
    user_id, name, email, expires_at = session
    
    # Check if session expired
    if datetime.fromisoformat(expires_at) < datetime.now():
        invalidate_session(session_id)
        return False, None
    
    return True, {'user_id': user_id, 'name': name, 'email': email}

def invalidate_session(session_id):
    """Invalidate a session (logout)."""
    conn = sqlite3.connect(AUTH_DB)
    cursor = conn.cursor()
    
    cursor.execute("""
        UPDATE sessions SET is_active = 0 WHERE session_id = ?
    """, (session_id,))
    
    conn.commit()
    conn.close()

def get_user_stats(user_id):
    """Get user-specific statistics."""
    from local_db import get_applicants
    
    # Get applicants for this user
    conn = sqlite3.connect('local_applicants.db')
    
    # First check if created_by column exists, if not add it
    cursor = conn.cursor()
    cursor.execute("PRAGMA table_info(applicants)")
    columns = [column[1] for column in cursor.fetchall()]
    
    if 'created_by' not in columns:
        cursor.execute("ALTER TABLE applicants ADD COLUMN created_by INTEGER")
        conn.commit()
    
    # Get user applicants
    cursor.execute("""
        SELECT COUNT(*) as total,
               AVG(total_trust_score) as avg_trust,
               AVG(CASE 
                   WHEN total_trust_score IS NOT NULL 
                   THEN 100 - total_trust_score 
                   ELSE NULL 
               END) as avg_obscurity,
               COUNT(CASE WHEN total_trust_score > 70 THEN 1 END) as graduated
        FROM applicants 
        WHERE created_by = ?
    """, (user_id,))
    
    stats = cursor.fetchone()
    conn.close()
    
    total, avg_trust, avg_obscurity, graduated = stats
    
    return {
        'total_applicants': total or 0,
        'avg_trust_score': round(avg_trust or 0, 1),
        'avg_obscurity': round(avg_obscurity or 0, 1),
        'graduated_percentage': round((graduated / total * 100) if total > 0 else 0, 1)
    }

# Streamlit integration functions
def init_auth():
    """Initialize authentication for Streamlit."""
    init_auth_db()
    
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
        st.session_state.user = None
        st.session_state.session_id = None

def require_auth():
    """Decorator/function to require authentication."""
    if not st.session_state.get('authenticated', False):
        return False
    
    # Validate session
    session_id = st.session_state.get('session_id')
    valid, user = validate_session(session_id)
    
    if not valid:
        st.session_state.authenticated = False
        st.session_state.user = None
        st.session_state.session_id = None
        return False
    
    # Update user info in session state
    st.session_state.user = user
    return True

def login_user(email, password):
    """Login user in Streamlit."""
    success, user = authenticate_user(email, password)
    
    if success:
        session_id = create_session(user['user_id'])
        st.session_state.authenticated = True
        st.session_state.user = user
        st.session_state.session_id = session_id
        return True
    
    return False

def logout_user():
    """Logout user in Streamlit."""
    if st.session_state.get('session_id'):
        invalidate_session(st.session_state.session_id)
    
    st.session_state.authenticated = False
    st.session_state.user = None
    st.session_state.session_id = None
