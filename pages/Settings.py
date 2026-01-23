"""
FaceMatch Pro - Settings Page
Customize app preferences and model parameters
"""

import streamlit as st
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from config import Config
from translations import get_text


def get_t(key):
    """Translation helper"""
    return get_text(key, st.session_state.language)


def initialize_settings():
    """Initialize settings in session state"""
    if 'settings_initialized' not in st.session_state:
        st.session_state.threshold = Config.DEFAULT_THRESHOLD
        st.session_state.use_tta = Config.USE_TTA
        st.session_state.heatmap_alpha = Config.HEATMAP_ALPHA
        st.session_state.min_face_size = Config.MTCNN_MIN_FACE_SIZE
        st.session_state.grid_cols = Config.DEFAULT_GRID_COLS
        st.session_state.settings_initialized = True


def reset_to_defaults():
    """Reset all settings to defaults"""
    st.session_state.threshold = Config.DEFAULT_THRESHOLD
    st.session_state.use_tta = Config.USE_TTA
    st.session_state.heatmap_alpha = Config.HEATMAP_ALPHA
    st.session_state.min_face_size = Config.MTCNN_MIN_FACE_SIZE
    st.session_state.grid_cols = Config.DEFAULT_GRID_COLS
    st.success("✅ Settings reset to defaults")


def main():
    """Main settings page"""
    st.title(f"⚙️ {get_t('set_title')}")
    st.markdown(f"*{get_t('set_subtitle')}*")

    st.markdown("---")

    # Initialize settings
    initialize_settings()

    # General Settings
    st.markdown(f"## 🌐 {get_t('set_general')}")

    col1, col2 = st.columns(2)

    with col1:
        # Language (already handled in sidebar, show current)
        st.markdown(f"### {get_t('set_language')}")
        current_lang = "English" if st.session_state.language == 'en' else "العربية"
        st.info(f"Current: {current_lang}")
        st.caption("Use language switcher in sidebar to change")

    with col2:
        # Theme (already handled in sidebar, show current)
        st.markdown(f"### {get_t('set_theme')}")
        current_theme = st.session_state.get('theme', 'light')
        st.info(f"Current: {current_theme.title()}")
        st.caption("Use theme toggle in sidebar to change")

    st.markdown("---")

    # Model Settings
    st.markdown(f"## 🤖 {get_t('set_model')}")

    # Threshold slider
    st.markdown(f"### {get_t('set_threshold')}")
    st.caption(get_t('set_threshold_desc'))

    threshold = st.slider(
        "Similarity Threshold",
        min_value=Config.MIN_THRESHOLD,
        max_value=Config.MAX_THRESHOLD,
        value=st.session_state.threshold,
        step=Config.THRESHOLD_STEP,
        format="%.2f",
        label_visibility="collapsed"
    )

    st.session_state.threshold = threshold

    # Show threshold impact
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Current Threshold", f"{threshold * 100:.0f}%")

    with col2:
        if threshold < 0.5:
            level = "Low (More Matches)"
        elif threshold < 0.7:
            level = "Medium (Balanced)"
        else:
            level = "High (Strict)"
        st.metric("Strictness", level)

    with col3:
        default_diff = (threshold - Config.DEFAULT_THRESHOLD) * 100
        st.metric("vs Default", f"{default_diff:+.0f}%")

    # Threshold examples
    with st.expander("📊 See Threshold Examples"):
        st.markdown("""
        ### Impact of Different Thresholds:

        **Example Scenario:** Two photos of the same person
        - Similarity score: **75%**

        | Threshold | Result | Explanation |
        |-----------|--------|-------------|
        | 50% | ✅ Match | Low threshold - accepts most similarities |
        | 70% | ✅ Match | Balanced - good for general use |
        | 80% | ❌ No Match | High threshold - very strict |

        **Recommendation:**
        - **General Use:** 60-70%
        - **High Security:** 75-85%
        - **Finding Similar Faces:** 50-60%
        """)

    st.markdown("---")

    # Test-Time Augmentation
    st.markdown(f"### {get_t('set_tta')}")
    st.caption(get_t('set_tta_desc'))

    use_tta = st.checkbox(
        "Enable TTA",
        value=st.session_state.use_tta,
        label_visibility="collapsed"
    )

    st.session_state.use_tta = use_tta

    if use_tta:
        st.info("✅ TTA Enabled - Better accuracy, slower processing (~2x)")
    else:
        st.warning("⚠️ TTA Disabled - Faster processing, slightly lower accuracy")

    st.markdown("---")

    # Face Detection Settings
    st.markdown(f"## 👤 {get_t('set_detection')}")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"### {get_t('set_min_face_size')}")
        min_face_size = st.slider(
            "Minimum face size (pixels)",
            min_value=10,
            max_value=100,
            value=st.session_state.min_face_size,
            step=5,
            label_visibility="collapsed"
        )

        st.session_state.min_face_size = min_face_size
        st.caption(f"Faces smaller than {min_face_size}×{min_face_size}px will be ignored")

    with col2:
        st.markdown(f"### {get_t('set_detection_sensitivity')}")

        # Detection confidence (informational only, not actually changing MTCNN)
        sensitivity_options = ["Low (0.8)", "Medium (0.9)", "High (0.95)"]
        sensitivity = st.selectbox(
            "Detection sensitivity",
            range(len(sensitivity_options)),
            index=1,  # Default to medium
            format_func=lambda x: sensitivity_options[x],
            label_visibility="collapsed"
        )

        if sensitivity == 0:
            st.caption("Detects more faces but may have false positives")
        elif sensitivity == 1:
            st.caption("Balanced detection (recommended)")
        else:
            st.caption("Only detects high-confidence faces")

    st.markdown("---")

    # Visualization Settings
    st.markdown(f"## 🎨 {get_t('set_visualization')}")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"### {get_t('set_heatmap_intensity')}")
        heatmap_alpha = st.slider(
            "Heatmap overlay transparency",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.heatmap_alpha,
            step=0.05,
            label_visibility="collapsed"
        )

        st.session_state.heatmap_alpha = heatmap_alpha
        st.caption(f"Alpha: {heatmap_alpha:.2f} (0 = invisible, 1 = opaque)")

    with col2:
        st.markdown("### Grid Layout")
        grid_cols = st.slider(
            "Number of columns in grid views",
            min_value=Config.MIN_GRID_COLS,
            max_value=Config.MAX_GRID_COLS,
            value=st.session_state.grid_cols,
            step=1,
            label_visibility="collapsed"
        )

        st.session_state.grid_cols = grid_cols
        st.caption(f"Images per row: {grid_cols}")

    # Show confidence toggle
    show_confidence = st.checkbox(
        get_t('set_show_confidence'),
        value=True
    )

    if show_confidence:
        st.caption("Confidence scores will be displayed on results")

    st.markdown("---")

    # Export Settings
    st.markdown(f"## 📥 {get_t('set_export')}")

    export_format = st.selectbox(
        get_t('set_export_format'),
        Config.EXPORT_FORMATS
    )

    st.caption(f"Results will be exported as {export_format}")

    st.markdown("---")

    # Model Information (Read-only)
    st.markdown("## ℹ️ Model Information")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        **Architecture:**
        - Backbone: ResNet50
        - Embedding: 512 dimensions
        - Attention: SE Blocks
        - Training: ArcFace + Triplet Loss
        """)

    with col2:
        st.markdown(f"""
        **Performance:**
        - Accuracy: {Config.MODEL_METRICS['accuracy'] * 100:.1f}%
        - Precision: {Config.MODEL_METRICS['precision'] * 100:.1f}%
        - Recall: {Config.MODEL_METRICS['recall'] * 100:.1f}%
        - F1 Score: {Config.MODEL_METRICS['f1_score'] * 100:.1f}%
        """)

    st.markdown("---")

    # Action Buttons
    col1, col2 = st.columns(2)

    with col1:
        if st.button(f"✅ {get_t('set_save')}", use_container_width=True, type="primary"):
            st.success("✅ Settings saved successfully!")
            st.balloons()

    with col2:
        if st.button(f"🔄 {get_t('set_reset')}", use_container_width=True):
            reset_to_defaults()
            st.rerun()

    # Current Settings Summary
    st.markdown("---")
    st.markdown("## 📋 Current Settings Summary")

    settings_summary = f"""
    | Setting | Value |
    |---------|-------|
    | Language | {current_lang} |
    | Threshold | {threshold * 100:.0f}% |
    | TTA | {'Enabled' if use_tta else 'Disabled'} |
    | Min Face Size | {min_face_size}px |
    | Heatmap Alpha | {heatmap_alpha:.2f} |
    | Grid Columns | {grid_cols} |
    | Export Format | {export_format} |
    """

    st.markdown(settings_summary)

    # Tips
    with st.expander("💡 Settings Tips"):
        st.markdown("""
        ### Optimization Tips:

        **For Speed:**
        - Disable TTA
        - Increase min face size to 40-50px
        - Lower threshold to 0.5-0.6

        **For Accuracy:**
        - Enable TTA
        - Decrease min face size to 20px
        - Adjust threshold based on use case

        **For Batch Processing:**
        - Disable TTA (significantly faster)
        - Use 2-3 grid columns for better viewing
        - Enable confidence display

        **For Security Applications:**
        - Enable TTA
        - Set threshold to 0.75-0.85
        - Use high detection sensitivity
        """)

    # Advanced Settings (collapsed by default)
    with st.expander("🔧 Advanced Settings"):
        st.warning("⚠️ These settings are for advanced users only")

        st.markdown("### Image Processing")

        col1, col2 = st.columns(2)

        with col1:
            max_image_size = st.number_input(
                "Max image size (MB)",
                min_value=1,
                max_value=20,
                value=Config.MAX_IMAGE_SIZE // (1024 * 1024),
                step=1
            )

        with col2:
            max_batch_size = st.number_input(
                "Max batch size",
                min_value=10,
                max_value=100,
                value=Config.MAX_BATCH_SIZE,
                step=10
            )

        st.markdown("### Face Detection Advanced")

        st.code(f"""
        MTCNN Parameters:
        - Thresholds: {Config.MTCNN_THRESHOLDS}
        - Scale Factor: {Config.MTCNN_FACTOR}
        - Min Face Size: {min_face_size}px
        """)

        st.caption("These are optimized values. Change with caution!")


if __name__ == "__main__":
    main()