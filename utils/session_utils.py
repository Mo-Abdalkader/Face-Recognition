"""
FaceMatch Pro - Session State Utilities
Ensures session state is always initialized
"""

import streamlit as st
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))
from config import Config


def init_session_state():
    """
    Initialize all session state variables if they don't exist.
    MUST be called at the start of EVERY page.
    """
    # Language
    if 'language' not in st.session_state:
        st.session_state.language = Config.DEFAULT_LANGUAGE
    
    # Theme
    if 'theme' not in st.session_state:
        st.session_state.theme = 'light'
    
    # Threshold
    if 'threshold' not in st.session_state:
        st.session_state.threshold = Config.DEFAULT_THRESHOLD
    
    # Settings
    if 'settings_initialized' not in st.session_state:
        st.session_state.use_tta = Config.USE_TTA
        st.session_state.heatmap_alpha = Config.HEATMAP_ALPHA
        st.session_state.min_face_size = Config.MTCNN_MIN_FACE_SIZE
        st.session_state.grid_cols = Config.DEFAULT_GRID_COLS
        st.session_state.settings_initialized = True
    
    # Rating (for feedback)
    if 'rating' not in st.session_state:
        st.session_state.rating = 0


def ensure_language_set():
    """
    Ensure language is set in session state.
    Quick check for translation functions.
    """
    if 'language' not in st.session_state:
        st.session_state.language = Config.DEFAULT_LANGUAGE
    return st.session_state.language
