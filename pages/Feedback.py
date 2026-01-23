"""
FaceMatch Pro - Feedback Page
Professional feedback system with Telegram integration
"""

import streamlit as st
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from config import Config
from translations import get_text
from utils.telegram_utils import send_telegram_message, get_telegram_credentials


def get_t(key):
    """Translation helper"""
    return get_text(key, st.session_state.language)


def render_star_rating():
    """Render professional star rating with clear labels"""
    st.markdown(f"### {get_t('fb_rating')}")
    
    # Initialize rating in session state
    if 'rating' not in st.session_state:
        st.session_state.rating = 0

    # Create horizontal layout for rating
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Star buttons in a single row
        star_cols = st.columns(5)
        
        for i, col in enumerate(star_cols):
            star_num = i + 1
            if star_num <= st.session_state.rating:
                star = "⭐"
            else:
                star = "☆"
            
            if col.button(star, key=f"star_{star_num}", use_container_width=True):
                st.session_state.rating = star_num
                st.rerun()
    
    with col2:
        # Show rating value and quality
        if st.session_state.rating > 0:
            quality_labels = {
                1: "Poor" if st.session_state.language == 'en' else "ضعيف",
                2: "Fair" if st.session_state.language == 'en' else "مقبول",
                3: "Good" if st.session_state.language == 'en' else "جيد",
                4: "Very Good" if st.session_state.language == 'en' else "جيد جداً",
                5: "Excellent" if st.session_state.language == 'en' else "ممتاز"
            }
            st.markdown(f"""
            <div style='text-align: center; padding: 0.5rem; 
                        background: {Config.get_color('primary', 0.1)}; 
                        border-radius: 8px;'>
                <div style='font-size: 1.5rem; font-weight: bold; 
                            color: {Config.COLORS['primary']};'>
                    {st.session_state.rating}/5
                </div>
                <div style='font-size: 0.9rem; color: gray;'>
                    {quality_labels[st.session_state.rating]}
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style='text-align: center; padding: 0.5rem; color: gray;'>
                <div style='font-size: 0.9rem;'>
                    Click stars<br>to rate
                </div>
            </div>
            """, unsafe_allow_html=True)

    return st.session_state.rating


def main():
    """Main feedback page"""
    st.title(f"💬 {get_t('fb_title')}")
    st.markdown(f"*{get_t('fb_subtitle')}*")

    st.markdown("---")

    # Professional header with info
    st.markdown(f"""
    <div style='background: linear-gradient(135deg, {Config.get_color('primary', 0.1)}, 
                {Config.get_color('secondary', 0.1)}); 
                padding: 1.5rem; border-radius: 10px; margin-bottom: 2rem;'>
        <h3 style='margin: 0; color: {Config.COLORS['primary']};'>
            📢 We Value Your Feedback
        </h3>
        <p style='margin: 0.5rem 0 0 0; font-size: 1rem;'>
            Help us improve FaceMatch Pro by sharing your thoughts, reporting bugs, 
            or suggesting new features. Your input makes a difference!
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Feedback form in a clean card
    st.markdown(f"""
    <div style='background: white; padding: 2rem; border-radius: 10px; 
                box-shadow: 0 2px 8px rgba(0,0,0,0.1);'>
    """, unsafe_allow_html=True)

    # Name (optional)
    col1, col2 = st.columns([2, 1])
    
    with col1:
        name = st.text_input(
            f"👤 {get_t('fb_name')}",
            placeholder=get_t('fb_name_placeholder'),
            help="Optional - Leave blank to send anonymously"
        )
    
    with col2:
        # Category with icon
        categories = Config.FEEDBACK_CATEGORIES[st.session_state.language]
        category_icons = {
            'Bug Report': '🐛', 'Feature Request': '💡', 
            'Suggestion': '📝', 'General': '💬', 'Other': '📌',
            'تقرير خطأ': '🐛', 'طلب ميزة': '💡',
            'اقتراح': '📝', 'عام': '💬', 'أخرى': '📌'
        }
        category = st.selectbox(
            f"📂 {get_t('fb_category')}",
            categories,
            format_func=lambda x: f"{category_icons.get(x, '📌')} {x}"
        )

    if not name or name.strip() == "":
        name = "Anonymous"

    st.markdown("<br>", unsafe_allow_html=True)

    # Star rating - compact and professional
    rating = render_star_rating()

    st.markdown("<br>", unsafe_allow_html=True)

    # Message with character counter
    message = st.text_area(
        f"💭 {get_t('fb_message')}",
        placeholder=get_t('fb_message_placeholder'),
        height=150,
        help="Tell us what you think, report bugs, or suggest features",
        max_chars=1000
    )
    
    # Character counter
    char_count = len(message) if message else 0
    st.caption(f"Characters: {char_count}/1000")

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Privacy notice - professional
    st.info(f"🔒 **Privacy:** {get_t('fb_privacy')}")

    # Action buttons - professional layout
    col1, col2, col3 = st.columns([2, 1, 1])

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
    
    with col3:
        # Connection status indicator (simplified)
        bot_token, chat_id = get_telegram_credentials()
        if bot_token and chat_id:
            st.markdown("""
            <div style='text-align: center; padding: 0.5rem; 
                        background: #28a745; color: white; 
                        border-radius: 8px; font-size: 0.9rem;'>
                ✓ Connected
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style='text-align: center; padding: 0.5rem; 
                        background: #ffc107; color: black; 
                        border-radius: 8px; font-size: 0.9rem;'>
                ⚠ Offline
            </div>
            """, unsafe_allow_html=True)

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
                # Success animation
                st.balloons()

                st.markdown(f"""
                <div style='background: {Config.get_color('success', 0.1)}; 
                            border-left: 4px solid {Config.COLORS['success']}; 
                            padding: 2rem; border-radius: 8px; text-align: center;'>
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

                # Thank you message
                st.markdown(f"""
                <div style='text-align: center; padding: 2rem; margin-top: 2rem; 
                            background: {Config.get_color('primary', 0.05)}; 
                            border-radius: 10px;'>
                    <h3 style='color: {Config.COLORS['primary']};'>Thank You! 🙏</h3>
                    <p>Your feedback helps us improve FaceMatch Pro</p>
                </div>
                """, unsafe_allow_html=True)

            else:
                st.markdown(f"""
                <div style='background: {Config.get_color('danger', 0.1)}; 
                            border-left: 4px solid {Config.COLORS['danger']}; 
                            padding: 2rem; border-radius: 8px; text-align: center;'>
                    <h2 style='color: {Config.COLORS["danger"]}; margin: 0;'>
                        ❌ {get_t('fb_error')}
                    </h2>
                    <p style='font-size: 1.1rem; margin: 1rem 0 0 0;'>
                        {get_t('fb_error_desc')}
                    </p>
                </div>
                """, unsafe_allow_html=True)

    # Feedback guidelines - professional cards
    st.markdown("---")
    st.markdown("### 💡 Feedback Guidelines")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"""
        <div style='background: {Config.get_color('primary', 0.05)}; 
                    padding: 1.5rem; border-radius: 8px; 
                    border-left: 4px solid {Config.COLORS['primary']};'>
            <h4 style='margin: 0 0 1rem 0; color: {Config.COLORS['primary']};'>
                🐛 Bug Report
            </h4>
            <ul style='margin: 0; padding-left: 1.5rem;'>
                <li>Describe what happened</li>
                <li>Steps to reproduce</li>
                <li>Expected vs actual behavior</li>
                <li>Browser/device info (optional)</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        st.markdown(f"""
        <div style='background: {Config.get_color('info', 0.05)}; 
                    padding: 1.5rem; border-radius: 8px; 
                    border-left: 4px solid {Config.COLORS['info']};'>
            <h4 style='margin: 0 0 1rem 0; color: {Config.COLORS['info']};'>
                📝 Suggestion
            </h4>
            <ul style='margin: 0; padding-left: 1.5rem;'>
                <li>UI/UX improvements</li>
                <li>Performance enhancements</li>
                <li>Workflow optimizations</li>
                <li>Any other ideas</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div style='background: {Config.get_color('warning', 0.05)}; 
                    padding: 1.5rem; border-radius: 8px; 
                    border-left: 4px solid {Config.COLORS['warning']};'>
            <h4 style='margin: 0 0 1rem 0; color: {Config.COLORS['warning']};'>
                💡 Feature Request
            </h4>
            <ul style='margin: 0; padding-left: 1.5rem;'>
                <li>Describe the feature</li>
                <li>Explain why it's useful</li>
                <li>Provide examples if possible</li>
                <li>Priority level (optional)</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        st.markdown(f"""
        <div style='background: {Config.get_color('success', 0.05)}; 
                    padding: 1.5rem; border-radius: 8px; 
                    border-left: 4px solid {Config.COLORS['success']};'>
            <h4 style='margin: 0 0 1rem 0; color: {Config.COLORS['success']};'>
                💬 General
            </h4>
            <ul style='margin: 0; padding-left: 1.5rem;'>
                <li>Questions</li>
                <li>Compliments</li>
                <li>General comments</li>
                <li>Anything else!</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    # Example feedback
    with st.expander("📋 Example Feedback"):
        st.markdown(f"""
        <div style='background: {Config.get_color('primary', 0.02)}; 
                    padding: 1rem; border-radius: 8px; margin-bottom: 1rem;'>
            <strong style='color: {Config.COLORS['primary']};'>
                ✓ Good Bug Report:
            </strong>
            <p style='margin: 0.5rem 0 0 0; font-style: italic;'>
                "When I upload images larger than 5MB, the page freezes. 
                Using Chrome on Windows 10. Can you fix this?"
            </p>
        </div>

        <div style='background: {Config.get_color('warning', 0.02)}; 
                    padding: 1rem; border-radius: 8px; margin-bottom: 1rem;'>
            <strong style='color: {Config.COLORS['warning']};'>
                ✓ Good Feature Request:
            </strong>
            <p style='margin: 0.5rem 0 0 0; font-style: italic;'>
                "It would be great to have a batch comparison feature where 
                I can compare one person against multiple images at once."
            </p>
        </div>

        <div style='background: {Config.get_color('info', 0.02)}; 
                    padding: 1rem; border-radius: 8px;'>
            <strong style='color: {Config.COLORS['info']};'>
                ✓ Good Suggestion:
            </strong>
            <p style='margin: 0.5rem 0 0 0; font-style: italic;'>
                "The similarity gauge is awesome! Could you add a history 
                feature to see past comparisons?"
            </p>
        </div>
        """, unsafe_allow_html=True)

    # App statistics
    st.markdown("---")

    with st.expander("📊 App Information"):
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Model Accuracy", f"{Config.MODEL_METRICS['accuracy'] * 100:.1f}%")

        with col2:
            st.metric("Version", Config.VERSION)

        with col3:
            st.metric("Last Updated", Config.VERSION_DATE)
        
        with col4:
            st.metric("Pages", "9")

        st.markdown(f"""
        <div style='margin-top: 1rem; padding: 1rem; 
                    background: {Config.get_color('info', 0.1)}; 
                    border-radius: 8px;'>
            <h4 style='margin: 0; color: {Config.COLORS["info"]};'>
                About FaceMatch Pro
            </h4>
            <p style='margin: 0.5rem 0 0 0; font-size: 0.9rem;'>
                FaceMatch Pro is an advanced face recognition system built with 
                state-of-the-art deep learning. Your feedback helps us make it better!
            </p>
        </div>
        """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
