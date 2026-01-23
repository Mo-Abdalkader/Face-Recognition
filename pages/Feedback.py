"""
FaceMatch Pro - Feedback Page
Anonymous feedback system with Telegram integration
"""

import streamlit as st
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from config import Config
from translations import get_text
from utils.telegram_utils import send_telegram_message, render_telegram_status


def get_t(key):
    """Translation helper"""
    return get_text(key, st.session_state.language)


def render_star_rating():
    """Render interactive star rating"""
    st.markdown("### " + get_t('fb_rating'))

    # Create star rating using columns
    cols = st.columns(5)

    # Initialize rating in session state
    if 'rating' not in st.session_state:
        st.session_state.rating = 0

    selected_rating = st.session_state.rating

    # Display stars
    for i, col in enumerate(cols):
        star_num = i + 1
        if star_num <= selected_rating:
            star = "⭐"
        else:
            star = "☆"

        if col.button(star, key=f"star_{star_num}", use_container_width=True):
            st.session_state.rating = star_num
            st.rerun()

    # Show selected rating
    if selected_rating > 0:
        st.markdown(f"""
        <div style='text-align: center; font-size: 1.5rem; margin-top: 0.5rem;'>
            {"⭐" * selected_rating} ({selected_rating}/5)
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style='text-align: center; color: gray; margin-top: 0.5rem;'>
            Click stars to rate
        </div>
        """, unsafe_allow_html=True)

    return selected_rating


def main():
    """Main feedback page"""
    st.title(f"💬 {get_t('fb_title')}")
    st.markdown(f"*{get_t('fb_subtitle')}*")

    st.markdown("---")

    # Check Telegram status
    with st.expander("🔧 Telegram Bot Status"):
        render_telegram_status()

    st.markdown("---")

    # Feedback form
    st.markdown(f"## 📝 Feedback Form")

    # Name (optional)
    name = st.text_input(
        get_t('fb_name'),
        placeholder=get_t('fb_name_placeholder'),
        help="Leave blank to send anonymously"
    )

    if not name or name.strip() == "":
        name = "Anonymous"

    # Category
    categories = Config.FEEDBACK_CATEGORIES[st.session_state.language]
    category = st.selectbox(
        get_t('fb_category'),
        categories
    )

    # Message
    message = st.text_area(
        get_t('fb_message'),
        placeholder=get_t('fb_message_placeholder'),
        height=150,
        help="Tell us what you think, report bugs, or suggest features"
    )

    # Star rating
    rating = render_star_rating()

    # Privacy notice
    st.info(f"🔒 {get_t('fb_privacy')}")

    # Buttons
    col1, col2 = st.columns(2)

    with col1:
        send_button = st.button(
            f"📤 {get_t('fb_send')}",
            use_container_width=True,
            type="primary"
        )

    with col2:
        clear_button = st.button(
            f"🗑️ {get_t('fb_clear')}",
            use_container_width=True
        )

    # Handle clear button
    if clear_button:
        st.session_state.rating = 0
        st.rerun()

    # Handle send button
    if send_button:
        if not message or message.strip() == "":
            st.error("⚠️ Please enter a message")
        else:
            # Send feedback
            with st.spinner("Sending your feedback..."):
                success = send_telegram_message(
                    message=message.strip(),
                    category=category,
                    name=name,
                    rating=rating if rating > 0 else None
                )

            if success:
                # Success message with animation
                st.balloons()

                st.markdown(f"""
                <div class='success-box' style='text-align: center; padding: 2rem;'>
                    <h2 style='color: {Config.COLORS["success"]}; margin: 0;'>
                        ✅ {get_t('fb_success')}
                    </h2>
                    <p style='font-size: 1.1rem; margin: 1rem 0 0 0;'>
                        {get_t('fb_success_desc')}
                    </p>
                </div>
                """, unsafe_allow_html=True)

                # Reset form
                st.session_state.rating = 0

                # Show thank you message
                st.markdown(f"""
                <div style='text-align: center; padding: 2rem; margin-top: 2rem; 
                            background: {Config.get_color('primary', 0.05)}; 
                            border-radius: 10px;'>
                    <h3>Thank You! 🙏</h3>
                    <p>Your feedback helps us improve FaceMatch Pro</p>
                </div>
                """, unsafe_allow_html=True)

            else:
                st.markdown(f"""
                <div class='error-box' style='text-align: center; padding: 2rem;'>
                    <h2 style='color: {Config.COLORS["danger"]}; margin: 0;'>
                        ❌ {get_t('fb_error')}
                    </h2>
                    <p style='font-size: 1.1rem; margin: 1rem 0 0 0;'>
                        {get_t('fb_error_desc')}
                    </p>
                </div>
                """, unsafe_allow_html=True)

    # Feedback guidelines
    st.markdown("---")

    st.markdown("### 💡 Feedback Guidelines")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        **🐛 Bug Report:**
        - Describe what happened
        - Steps to reproduce
        - Expected vs actual behavior
        - Browser/device info (optional)

        **💡 Feature Request:**
        - Describe the feature
        - Explain why it's useful
        - Provide examples if possible
        """)

    with col2:
        st.markdown("""
        **📝 Suggestion:**
        - UI/UX improvements
        - Performance enhancements
        - Workflow optimizations
        - Any other ideas

        **💬 General:**
        - Questions
        - Compliments
        - General comments
        - Anything else!
        """)

    # Examples
    with st.expander("📋 Example Feedback"):
        st.markdown("""
        **Good Bug Report:**
        > "When I upload images larger than 5MB, the page freezes. 
        > Using Chrome on Windows 10. Can you fix this?"

        **Good Feature Request:**
        > "It would be great to have a batch comparison feature where 
        > I can compare one person against multiple images at once."

        **Good Suggestion:**
        > "The similarity gauge is awesome! Could you add a history 
        > feature to see past comparisons?"
        """)

    # Statistics (if you want to show)
    st.markdown("---")

    with st.expander("📊 App Statistics"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Model Accuracy", f"{Config.MODEL_METRICS['accuracy'] * 100:.1f}%")

        with col2:
            st.metric("Version", Config.VERSION)

        with col3:
            st.metric("Last Updated", Config.VERSION_DATE)

        st.markdown(f"""
        <div style='margin-top: 1rem; padding: 1rem; 
                    background: {Config.get_color('info', 0.1)}; 
                    border-radius: 8px;'>
            <h4 style='margin: 0; color: {Config.COLORS["info"]};'>About FaceMatch Pro</h4>
            <p style='margin: 0.5rem 0 0 0; font-size: 0.9rem;'>
                FaceMatch Pro is an advanced face recognition system built with 
                state-of-the-art deep learning. Your feedback helps us make it better!
            </p>
        </div>
        """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()