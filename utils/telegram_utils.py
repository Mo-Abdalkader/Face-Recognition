"""
FaceMatch Pro - Telegram Utilities
Send anonymous feedback to Telegram bot
"""

import requests
import streamlit as st
from datetime import datetime
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))
from config import Config


def get_telegram_credentials():
    """
    Get Telegram bot credentials from secrets

    Returns:
        tuple: (bot_token, chat_id) or (None, None) if not configured
    """
    try:
        bot_token = st.secrets.get("TELEGRAM_BOT_TOKEN", None)
        chat_id = st.secrets.get("TELEGRAM_CHAT_ID", None)

        if bot_token and chat_id:
            return bot_token, chat_id
        else:
            return None, None

    except Exception:
        return None, None


def send_telegram_message(message, category="General", name="Anonymous", rating=None):
    """
    Send feedback message to Telegram

    Args:
        message: str - The feedback message
        category: str - Feedback category
        name: str - User's name (default: Anonymous)
        rating: int - Star rating (1-5), optional

    Returns:
        bool: True if sent successfully, False otherwise
    """
    bot_token, chat_id = get_telegram_credentials()

    if not bot_token or not chat_id:
        st.error("⚠️ Telegram bot not configured. Please add credentials to .streamlit/secrets.toml")
        return False

    try:
        # Format timestamp
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        # Build message
        stars = "⭐" * rating if rating else ""

        text = f"""
━━━━━━━━━━━━━━━━━━━━━━
🔔 NEW FEEDBACK
━━━━━━━━━━━━━━━━━━━━━━

👤 From: {name}
📂 Category: {category}
{f"⭐ Rating: {stars} ({rating}/5)" if rating else ""}

💬 Message:
{message}

📅 {timestamp}
━━━━━━━━━━━━━━━━━━━━━━
        """

        # Send via Telegram API
        url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        payload = {
            'chat_id': chat_id,
            'text': text,
            'parse_mode': 'HTML'
        }

        response = requests.post(url, json=payload, timeout=10)

        return response.status_code == 200

    except requests.exceptions.Timeout:
        st.error("⚠️ Request timed out. Please try again.")
        return False
    except requests.exceptions.RequestException as e:
        st.error(f"⚠️ Network error: {str(e)}")
        return False
    except Exception as e:
        st.error(f"⚠️ Error sending message: {str(e)}")
        return False


def test_telegram_connection():
    """
    Test Telegram bot connection

    Returns:
        bool: True if connection is successful
    """
    bot_token, chat_id = get_telegram_credentials()

    if not bot_token or not chat_id:
        return False

    try:
        # Test with getMe API
        url = f"https://api.telegram.org/bot{bot_token}/getMe"
        response = requests.get(url, timeout=5)

        if response.status_code == 200:
            bot_info = response.json()
            if bot_info.get('ok'):
                return True

        return False

    except Exception:
        return False


def send_test_message():
    """
    Send a test message to verify Telegram setup

    Returns:
        bool: True if test message sent successfully
    """
    return send_telegram_message(
        message="🎉 FaceMatch Pro is configured correctly!",
        category="System Test",
        name="System",
        rating=5
    )


def format_bug_report(error_message, page, user_action=""):
    """
    Format a bug report message

    Args:
        error_message: str - The error message
        page: str - Page where error occurred
        user_action: str - What user was doing

    Returns:
        str: Formatted bug report
    """
    return f"""
🐛 BUG REPORT

Page: {page}
User Action: {user_action}

Error Details:
{error_message}
"""


def format_feature_request(feature_description, priority="Medium"):
    """
    Format a feature request message

    Args:
        feature_description: str - Description of requested feature
        priority: str - Priority level

    Returns:
        str: Formatted feature request
    """
    return f"""
💡 FEATURE REQUEST

Priority: {priority}

Description:
{feature_description}
"""


def get_bot_info():
    """
    Get information about the configured Telegram bot

    Returns:
        dict: Bot information or None if not configured
    """
    bot_token, chat_id = get_telegram_credentials()

    if not bot_token:
        return None

    try:
        url = f"https://api.telegram.org/bot{bot_token}/getMe"
        response = requests.get(url, timeout=5)

        if response.status_code == 200:
            data = response.json()
            if data.get('ok'):
                bot = data['result']
                return {
                    'username': bot.get('username'),
                    'first_name': bot.get('first_name'),
                    'is_bot': bot.get('is_bot'),
                    'chat_id': chat_id
                }

        return None

    except Exception:
        return None


def render_telegram_status():
    """
    Render Telegram bot status in Streamlit
    Shows whether bot is configured and working
    """
    bot_info = get_bot_info()

    if bot_info:
        st.success(f"✅ Telegram bot connected: @{bot_info['username']}")

        with st.expander("Bot Details"):
            st.write(f"**Bot Name:** {bot_info['first_name']}")
            st.write(f"**Username:** @{bot_info['username']}")
            st.write(f"**Chat ID:** {bot_info['chat_id']}")

            if st.button("Send Test Message"):
                if send_test_message():
                    st.success("✅ Test message sent! Check your Telegram.")
                else:
                    st.error("❌ Failed to send test message.")
    else:
        st.warning("⚠️ Telegram bot not configured")

        with st.expander("How to Setup"):
            st.markdown("""
            ### Setup Telegram Bot

            1. **Create Bot:**
               - Open Telegram and search for `@BotFather`
               - Send `/newbot`
               - Follow instructions to create your bot
               - Copy the **TOKEN**

            2. **Get Chat ID:**
               - Start a chat with your bot
               - Send any message
               - Visit: `https://api.telegram.org/bot<YOUR_TOKEN>/getUpdates`
               - Copy the `chat.id` value

            3. **Add to secrets:**
               - Create `.streamlit/secrets.toml`
               - Add:
               ```toml
               TELEGRAM_BOT_TOKEN = "your_token_here"
               TELEGRAM_CHAT_ID = "your_chat_id_here"
               ```

            4. **Restart the app**
            """)


def send_analytics_update(stats_dict):
    """
    Send usage analytics to Telegram (optional)

    Args:
        stats_dict: dict with statistics (comparisons, searches, etc.)

    Returns:
        bool: True if sent successfully
    """
    bot_token, chat_id = get_telegram_credentials()

    if not bot_token or not chat_id:
        return False

    try:
        text = f"""
📊 DAILY ANALYTICS

Date: {datetime.now().strftime('%Y-%m-%d')}

Usage Statistics:
• Total Comparisons: {stats_dict.get('comparisons', 0)}
• Total Searches: {stats_dict.get('searches', 0)}
• Total Detections: {stats_dict.get('detections', 0)}
• Unique Users: {stats_dict.get('users', 0)}

Average Metrics:
• Avg Similarity: {stats_dict.get('avg_similarity', 0):.2f}
• Avg Processing Time: {stats_dict.get('avg_time', 0):.2f}s
        """

        url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        payload = {
            'chat_id': chat_id,
            'text': text
        }

        response = requests.post(url, json=payload, timeout=10)
        return response.status_code == 200

    except Exception:
        return False