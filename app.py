"""
FaceMatch Pro - Main Application
Advanced Face Recognition System
"""

import streamlit as st
from streamlit_option_menu import option_menu
import plotly.graph_objects as go
from pathlib import Path

# Import configuration and translations
from config import Config
from translations import get_text, get_all_texts

# Import Pages
from pages import Face_Detection, Compare, Search, Batch, Demo, Settings, Feedback, About

# Set page config (MUST be first Streamlit command)
st.set_page_config(
    page_title=Config.APP_NAME,
    page_icon=Config.PAGE_ICON,
    layout=Config.LAYOUT,
    initial_sidebar_state=Config.INITIAL_SIDEBAR_STATE
)

# ==================== CRITICAL: Initialize session state FIRST ====================
if 'language' not in st.session_state:
    st.session_state.language = Config.DEFAULT_LANGUAGE
if 'theme' not in st.session_state:
    st.session_state.theme = 'light'
if 'threshold' not in st.session_state:
    st.session_state.threshold = Config.DEFAULT_THRESHOLD


def get_t(key):
    """Quick translation helper"""
    return get_text(key, st.session_state.language)


def apply_custom_css():
    """Apply custom CSS styling"""
    lang_dir = Config.LANGUAGES[st.session_state.language]['dir']

    st.markdown(f"""
    <style>
        [dir="{lang_dir}"] {{
            direction: {lang_dir};
        }}

        .main {{
            padding: 2rem;
        }}

        h1 {{
            color: {Config.COLORS['primary']};
            font-weight: 700;
            margin-bottom: 1rem;
        }}

        h2 {{
            color: {Config.COLORS['secondary']};
            font-weight: 600;
            margin-top: 2rem;
        }}

        h3 {{
            color: {Config.COLORS['dark']};
            font-weight: 500;
        }}

        .stButton>button {{
            background: linear-gradient(135deg, {Config.COLORS['primary']}, {Config.COLORS['secondary']});
            color: white;
            border-radius: 8px;
            padding: 0.5rem 2rem;
            font-weight: 600;
            border: none;
            transition: transform 0.2s;
        }}

        .stButton>button:hover {{
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(138, 43, 226, 0.3);
        }}

        .success-box {{
            background-color: {Config.get_color('success', 0.1)};
            border-left: 4px solid {Config.COLORS['success']};
            padding: 1rem;
            border-radius: 4px;
            margin: 1rem 0;
        }}

        .error-box {{
            background-color: {Config.get_color('danger', 0.1)};
            border-left: 4px solid {Config.COLORS['danger']};
            padding: 1rem;
            border-radius: 4px;
            margin: 1rem 0;
        }}

        .metric-card {{
            background: linear-gradient(135deg, {Config.get_color('primary', 0.1)}, {Config.get_color('secondary', 0.1)});
            padding: 1.5rem;
            border-radius: 10px;
            text-align: center;
            margin: 0.5rem;
        }}

        .metric-value {{
            font-size: 2.5rem;
            font-weight: 700;
            color: {Config.COLORS['primary']};
        }}

        .metric-label {{
            font-size: 1rem;
            color: {Config.COLORS['dark']};
            margin-top: 0.5rem;
        }}
        
        /* Hide plain Streamlit navigation, keep beautiful user content */
        section[data-testid="stSidebarNav"] {{
            display: none !important;
        }}
        
        /* Keep the beautiful user content navigation */
        section[data-testid="stSidebarUserContent"] {{
            display: block;
        }}
    </style>
    """, unsafe_allow_html=True)


def render_sidebar():
    """Render sidebar navigation and controls"""
    with st.sidebar:
        # Logo and title at top
        st.markdown(f"""
        <div style='text-align: center; padding: 1.5rem 0 1rem 0;'>
            <div style='font-size: 4rem; margin-bottom: 0.5rem;'>{Config.PAGE_ICON}</div>
            <h2 style='margin: 0; font-size: 1.5rem; color: {Config.COLORS['primary']};'>
                {get_t('app_name')}
            </h2>
            <p style='color: gray; font-size: 0.85rem; margin: 0.3rem 0 0 0;'>
                v{Config.VERSION}
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")

        # Language selector - compact
        col1, col2 = st.columns(2)
        with col1:
            if st.button("English", width="stretch",
                         type="primary" if st.session_state.language == 'en' else "secondary"):
                st.session_state.language = 'en'
                st.rerun()

        with col2:
            if st.button("عربي", width="stretch",
                         type="primary" if st.session_state.language == 'ar' else "secondary"):
                st.session_state.language = 'ar'
                st.rerun()

        st.markdown("---")

        # Navigation menu - CLEAN, NO DUPLICATES
        selected = option_menu(
            menu_title=None,
            options=[
                get_t('nav_home'),
                get_t('nav_face_detection'),
                get_t('nav_compare'),
                get_t('nav_search'),
                get_t('nav_batch'),
                get_t('nav_demo'),
                get_t('nav_settings'),
                get_t('nav_feedback'),
                get_t('nav_about')
            ],
            icons=['house', 'person-bounding-box', 'arrows-angle-contract', 'search',
                   'folder', 'joystick', 'gear', 'chat-dots', 'info-circle'],
            menu_icon="cast",
            default_index=0,
            styles={
                "container": {"padding": "0"},
                "icon": {"color": Config.COLORS['primary'], "font-size": "1.1rem"},
                "nav-link": {
                    "font-size": "0.95rem", 
                    "text-align": "left", 
                    "margin": "2px 0",
                    "padding": "0.6rem 1rem",
                    "border-radius": "5px"
                },
                "nav-link-selected": {
                    "background-color": Config.COLORS['primary'],
                    "font-weight": "600"
                },
            }
        )

        st.markdown("---")

        # Model status - compact
        st.markdown(f"""
        <div style='background: linear-gradient(135deg, {Config.get_color('info', 0.1)}, 
                    {Config.get_color('primary', 0.05)}); 
                    padding: 1rem; border-radius: 8px; border-left: 3px solid {Config.COLORS['info']};'>
            <h4 style='margin: 0 0 0.5rem 0; color: {Config.COLORS['info']}; font-size: 0.95rem;'>
                📊 Model Status
            </h4>
            <p style='margin: 0; font-size: 0.85rem; line-height: 1.6;'>
                ✅ <strong>Accuracy:</strong> {Config.MODEL_METRICS['accuracy'] * 100:.1f}%<br>
                🎯 <strong>Threshold:</strong> {st.session_state.threshold:.2f}
            </p>
        </div>
        """, unsafe_allow_html=True)

        return selected


def render_home_page():
    """Render home page - FIXED LAYOUT"""
    # Hero section at TOP
    st.markdown(f"""
    <div style='text-align: center; padding: 2rem 0 1rem 0;'>
        <h1 style='font-size: 3rem; margin-bottom: 0.5rem;'>{Config.PAGE_ICON}</h1>
        <h1 style='font-size: 2.5rem; margin-bottom: 0.5rem;'>{get_t('app_name')}</h1>
        <h2 style='font-size: 1.8rem; margin-bottom: 0.5rem;'>{get_t('home_title')}</h2>
        <p style='font-size: 1.2rem; color: gray;'>{get_t('home_subtitle')}</p>
    </div>
    """, unsafe_allow_html=True)

    # Quick start guide
    st.markdown(f"## {get_t('home_quick_start')}")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(f"""
        <div class='metric-card'>
            <div style='font-size: 3rem;'>📤</div>
            <p style='margin-top: 1rem;'>{get_t('home_step1')}</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class='metric-card'>
            <div style='font-size: 3rem;'>🔍</div>
            <p style='margin-top: 1rem;'>{get_t('home_step2')}</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class='metric-card'>
            <div style='font-size: 3rem;'>✅</div>
            <p style='margin-top: 1rem;'>{get_t('home_step3')}</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Features
    st.markdown(f"## {get_t('home_features')}")

    col1, col2, col3 = st.columns(3)

    features = [
        (get_t('home_feature1'), col1),
        (get_t('home_feature2'), col2),
        (get_t('home_feature3'), col3),
        (get_t('home_feature4'), col1),
        (get_t('home_feature5'), col2),
        (get_t('home_feature6'), col3),
    ]

    for feature, col in features:
        with col:
            st.markdown(f"""
            <div style='padding: 1rem; border-left: 3px solid {Config.COLORS['primary']}; margin-bottom: 1rem;'>
                {feature}
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Performance metrics
    st.markdown(f"## {get_t('home_performance')}")

    col1, col2, col3, col4 = st.columns(4)

    metrics = [
        (get_t('home_accuracy'), Config.MODEL_METRICS['accuracy'], col1),
        (get_t('home_precision'), Config.MODEL_METRICS['precision'], col2),
        (get_t('home_recall'), Config.MODEL_METRICS['recall'], col3),
        (get_t('home_f1'), Config.MODEL_METRICS['f1_score'], col4),
    ]

    for label, value, col in metrics:
        with col:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-value'>{value * 100:.1f}%</div>
                <div class='metric-label'>{label}</div>
            </div>
            """, unsafe_allow_html=True)

    # Gauge chart
    st.markdown("<br>", unsafe_allow_html=True)

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=Config.MODEL_METRICS['accuracy'] * 100,
        title={'text': get_t('home_accuracy')},
        gauge={
            'axis': {'range': [None, 100]},
            'bar': {'color': Config.COLORS['primary']},
            'steps': [
                {'range': [0, 70], 'color': Config.get_color('danger', 0.2)},
                {'range': [70, 85], 'color': Config.get_color('warning', 0.2)},
                {'range': [85, 100], 'color': Config.get_color('success', 0.2)}
            ],
            'threshold': {
                'line': {'color': Config.COLORS['success'], 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))

    fig.update_layout(height=300)
    st.plotly_chart(fig, use_container_width=True)


def main():
    """Main application"""
    apply_custom_css()

    # Render sidebar and get selected page
    selected_page = render_sidebar()

    # Route to pages - EXACT ORDER MATCH
    if selected_page == get_t('nav_home'):
        render_home_page()

    elif selected_page == get_t('nav_face_detection'):
        Face_Detection.main()

    elif selected_page == get_t('nav_compare'):
        Compare.main()

    elif selected_page == get_t('nav_search'):
        Search.main()

    elif selected_page == get_t('nav_batch'):
        Batch.main()

    elif selected_page == get_t('nav_demo'):
        Demo.main()

    elif selected_page == get_t('nav_settings'):
        Settings.main()

    elif selected_page == get_t('nav_feedback'):
        Feedback.main()

    elif selected_page == get_t('nav_about'):
        About.main()

    # Footer
    st.markdown("---")
    st.markdown(f"""
    <div style='text-align: center; color: gray; padding: 1rem 0;'>
        {get_t('app_name')} • {get_t('about_version')} {Config.VERSION} • {Config.VERSION_DATE}
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
