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

# Initialize session state
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
        /* RTL Support */
        [dir="{lang_dir}"] {{
            direction: {lang_dir};
        }}

        /* Main container */
        .main {{
            padding: 2rem;
        }}

        /* Headers */
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

        /* Cards */
        .stCard {{
            background: white;
            padding: 1.5rem;
            border-radius: 10px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            margin-bottom: 1rem;
        }}

        /* Buttons */
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

        /* Success/Error boxes */
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

        /* Metrics */
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

        /* File uploader */
        .uploadedFile {{
            border: 2px dashed {Config.COLORS['primary']};
            border-radius: 8px;
            padding: 1rem;
        }}

        /* Sidebar */
        .css-1d391kg {{
            background: linear-gradient(180deg, {Config.get_color('primary', 0.05)}, white);
        }}

        /* Progress bar */
        .stProgress > div > div > div {{
            background: linear-gradient(90deg, {Config.COLORS['primary']}, {Config.COLORS['secondary']});
        }}

        /* Match badge */
        .match-badge {{
            display: inline-block;
            padding: 0.5rem 1rem;
            border-radius: 20px;
            font-weight: 600;
            font-size: 1.1rem;
        }}

        .match-yes {{
            background-color: {Config.get_color('success', 0.2)};
            color: {Config.COLORS['success']};
            border: 2px solid {Config.COLORS['success']};
        }}

        .match-no {{
            background-color: {Config.get_color('danger', 0.2)};
            color: {Config.COLORS['danger']};
            border: 2px solid {Config.COLORS['danger']};
        }}
    </style>
    """, unsafe_allow_html=True)


def render_sidebar():
    """Render sidebar navigation and controls"""
    with st.sidebar:
        # Logo and title
        st.markdown(f"""
        <div style='text-align: center; padding: 1rem 0;'>
            <h1 style='font-size: 2.5rem; margin: 0;'>{Config.PAGE_ICON}</h1>
            <h2 style='margin: 0.5rem 0;'>{get_t('app_name')}</h2>
            <p style='color: gray; font-size: 0.9rem;'>v{Config.VERSION}</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")

        # Language selector
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🇬🇧 English", use_container_width=True,
                         type="primary" if st.session_state.language == 'en' else "secondary"):
                st.session_state.language = 'en'
                st.rerun()

        with col2:
            if st.button("عربي 🇸🇦", use_container_width=True,
                         type="primary" if st.session_state.language == 'ar' else "secondary"):
                st.session_state.language = 'ar'
                st.rerun()

        st.markdown("---")

        # Navigation menu
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
            icons=['house', 'person', 'arrows-angle-contract', 'search',
                   'folder', 'joystick', 'gear', 'chat', 'info-circle'],
            menu_icon="cast",
            default_index=0,
            styles={
                "container": {"padding": "0"},
                "icon": {"color": Config.COLORS['primary'], "font-size": "1.2rem"},
                "nav-link": {"font-size": "1rem", "text-align": "left", "margin": "0px"},
                "nav-link-selected": {"background-color": Config.COLORS['primary']},
            }
        )

        st.markdown("---")

        # Model status
        st.markdown(f"""
        <div style='background: {Config.get_color('info', 0.1)}; padding: 1rem; border-radius: 8px;'>
            <h4 style='margin: 0; color: {Config.COLORS['info']};'>📊 Model Status</h4>
            <p style='margin: 0.5rem 0 0 0; font-size: 0.9rem;'>
                ✅ Accuracy: {Config.MODEL_METRICS['accuracy'] * 100:.1f}%<br>
                🎯 Threshold: {st.session_state.threshold:.2f}
            </p>
        </div>
        """, unsafe_allow_html=True)

        return selected


def render_home_page():
    """Render home page"""
    # Hero section
    st.markdown(f"""
    <div style='text-align: center; padding: 3rem 0 2rem 0;'>
        <h1 style='font-size: 3rem; margin-bottom: 0.5rem;'>{get_t('home_title')}</h1>
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

    # Gauge chart for overall performance
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
    # Apply custom CSS
    apply_custom_css()

    # Render sidebar and get selected page
    selected_page = render_sidebar()

    # Route to pages
    if selected_page == get_t('nav_home'):
        render_home_page()

    elif selected_page == get_t('nav_face_detection'):
        st.title(get_t('fd_title'))
        st.info("🚧 This page will detect faces in your images using MTCNN")
        Face_Detection.main()

        # st.markdown("**Coming in the next files...**")

    elif selected_page == get_t('nav_compare'):
        st.title(get_t('comp_title'))
        st.info("🚧 This page will compare two faces with heatmap visualization")
        Compare.main()

        # st.markdown("**Coming in the next files...**")

    elif selected_page == get_t('nav_search'):
        st.title(get_t('search_title'))
        st.info("🚧 This page will search for matching faces in multiple images")
        Search.main()

        # st.markdown("**Coming in the next files...**")

    elif selected_page == get_t('nav_batch'):
        st.title(get_t('batch_title'))
        st.info("🚧 This page will process batches and find similar faces")
        Batch.main()

        # st.markdown("**Coming in the next files...**")

    elif selected_page == get_t('nav_demo'):
        st.title(get_t('demo_title'))
        st.info("🚧 This page will show demo scenarios")
        Demo.main()

        # st.markdown("**Coming in the next files...**")

    elif selected_page == get_t('nav_settings'):
        st.title(get_t('set_title'))
        st.info("🚧 This page will let you customize settings")
        Settings.main()

        # st.markdown("**Coming in the next files...**")

    elif selected_page == get_t('nav_feedback'):
        st.title(get_t('fb_title'))
        st.info("🚧 This page will send feedback to Telegram")
        Feedback.main()

        # st.markdown("**Coming in the next files...**")

    elif selected_page == get_t('nav_about'):
        st.title(get_t('about_title'))
        st.info("🚧 This page will show app information")
        About.main()

        # st.markdown("**Coming in the next files...**")

    # Footer
    st.markdown("---")
    st.markdown(f"""
    <div style='text-align: center; color: gray; padding: 1rem 0;'>
        {get_t('app_name')} • {get_t('about_version')} {Config.VERSION} • {Config.VERSION_DATE}
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
