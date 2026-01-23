"""
FaceMatch Pro - About Page
Information about the app, model, and privacy
"""

import streamlit as st
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from config import Config
from translations import get_text

from utils.session_utils import init_session_state
init_session_state()


def get_t(key):
    """Translation helper"""
    return get_text(key, st.session_state.language)


def main():
    """Main about page"""
    st.title(f"ℹ️ {get_t('about_title')}")

    st.markdown("---")

    # Introduction
    st.markdown(f"## 👋 Welcome to {Config.APP_NAME}")

    st.markdown(get_t('about_description'))

    st.markdown(f"""
    <div style='background: {Config.get_color('primary', 0.05)}; 
                padding: 1.5rem; border-radius: 10px; margin: 1rem 0;'>
        <h3 style='margin: 0; color: {Config.COLORS['primary']};'>Version {Config.VERSION}</h3>
        <p style='margin: 0.5rem 0 0 0;'>Released: {Config.VERSION_DATE}</p>
    </div>
    """, unsafe_allow_html=True)

    # Model Architecture
    st.markdown("---")
    st.markdown(f"## 🏗️ {get_t('about_model')}")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"""
        ### Architecture Details

        **Backbone:**
        - ResNet50 (Pretrained on ImageNet)
        - 50 layers deep
        - 25.6M parameters

        **Enhancements:**
        - Squeeze-and-Excitation (SE) Blocks
        - Attention mechanism
        - Global Average Pooling

        **Embedding:**
        - 512-dimensional face vectors
        - L2 normalized
        - Metric learning optimized
        """)

    with col2:
        st.markdown(f"""
        ### Training Details

        **Loss Functions:**
        - ArcFace Loss (40%)
        - Online Hard Triplet Loss (60%)
        - Label Smoothing

        **Optimization:**
        - AdamW optimizer
        - Cosine annealing schedule
        - Stochastic Weight Averaging (SWA)

        **Data:**
        - VGGFace2 dataset
        - 8,631 identities
        - 3.3M images
        """)

    # Model diagram
    st.markdown("### 🔄 Model Pipeline")

    st.code("""
    Input Image (224×224)
        ↓
    ResNet50 Feature Extraction
        ↓ (2048-dim features)
    SE Attention Block
        ↓
    Global Average Pooling
        ↓
    Fully Connected (1024-dim)
        ↓ BatchNorm + PReLU + Dropout
    Fully Connected (512-dim)
        ↓ BatchNorm + L2 Normalization
    Face Embedding (512-dim)
    """, language="text")

    # Performance Metrics
    st.markdown("---")
    st.markdown(f"## 📊 {get_t('about_performance')}")

    col1, col2, col3, col4 = st.columns(4)

    metrics = [
        ("Accuracy", Config.MODEL_METRICS['accuracy'], col1),
        ("Precision", Config.MODEL_METRICS['precision'], col2),
        ("Recall", Config.MODEL_METRICS['recall'], col3),
        ("F1 Score", Config.MODEL_METRICS['f1_score'], col4),
    ]

    for label, value, col in metrics:
        with col:
            st.markdown(f"""
            <div style='text-align: center; padding: 1rem; 
                        background: {Config.get_color('success', 0.1)}; 
                        border-radius: 8px;'>
                <h2 style='margin: 0; color: {Config.COLORS['success']};'>
                    {value * 100:.1f}%
                </h2>
                <p style='margin: 0.5rem 0 0 0; font-size: 0.9rem;'>{label}</p>
            </div>
            """, unsafe_allow_html=True)

    # Additional metrics
    st.markdown("### Other Metrics")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("AUC (ROC)", "0.945")

    with col2:
        st.metric("Default Threshold", f"{Config.DEFAULT_THRESHOLD:.2f}")

    with col3:
        st.metric("Embedding Dimension", "512")

    # How it Works
    st.markdown("---")
    st.markdown(f"## 🔬 {get_t('about_how_it_works')}")

    steps = [
        ("1️⃣", get_t('about_step1'), "MTCNN detects faces with high accuracy"),
        ("2️⃣", get_t('about_step2'), "Neural network creates unique face fingerprints"),
        ("3️⃣", get_t('about_step3'), "Mathematical comparison of face vectors"),
        ("4️⃣", get_t('about_step4'), "Decision based on similarity threshold"),
    ]

    for emoji, title, desc in steps:
        st.markdown(f"""
        <div style='padding: 1rem; margin: 0.5rem 0; 
                    border-left: 4px solid {Config.COLORS['primary']};
                    background: {Config.get_color('primary', 0.02)};'>
            <h4 style='margin: 0;'>{emoji} {title}</h4>
            <p style='margin: 0.5rem 0 0 0; color: gray;'>{desc}</p>
        </div>
        """, unsafe_allow_html=True)

    # Privacy & Security
    st.markdown("---")
    st.markdown(f"## 🔒 {get_t('about_privacy')}")

    privacy_points = [
        get_t('about_privacy_1'),
        get_t('about_privacy_2'),
        get_t('about_privacy_3'),
        get_t('about_privacy_4'),
    ]

    for point in privacy_points:
        st.markdown(f"""
        <div style='padding: 0.75rem; margin: 0.5rem 0; 
                    background: {Config.get_color('success', 0.05)};
                    border-radius: 6px;'>
            {point}
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    ### Data Handling

    **What we DON'T do:**
    - ❌ Store your uploaded images
    - ❌ Save face embeddings
    - ❌ Share data with third parties
    - ❌ Track user behavior
    - ❌ Use images for training

    **What we DO:**
    - ✅ Process images in memory only
    - ✅ Delete images after processing
    - ✅ Encrypt data in transit
    - ✅ Keep everything local
    - ✅ Respect your privacy
    """)

    # Technologies Used
    st.markdown("---")
    st.markdown("## 🛠️ Technologies Used")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        ### Frontend
        - **Streamlit** - Interactive UI
        - **Plotly** - Data visualization
        - **Matplotlib** - Charts and graphs

        ### Face Detection
        - **MTCNN** - Multi-task CNN
        - **OpenCV** - Image processing

        ### Deep Learning
        - **PyTorch** - Neural networks
        - **TorchVision** - Computer vision models
        """)

    with col2:
        st.markdown("""
        ### Model Training
        - **ResNet50** - Backbone architecture
        - **ArcFace** - Angular margin loss
        - **Triplet Loss** - Metric learning

        ### Communication
        - **Telegram Bot API** - Feedback system
        - **Requests** - HTTP requests

        ### Analysis
        - **NumPy** - Numerical computing
        - **Pandas** - Data manipulation
        - **SciPy** - Scientific computing
        """)

    # Credits
    st.markdown("---")
    st.markdown(f"## 🎓 {get_t('about_credits')}")

    st.markdown("""
    ### Acknowledgments

    **Research Papers:**
    - Deng et al., "ArcFace: Additive Angular Margin Loss for Deep Face Recognition" (CVPR 2019)
    - Schroff et al., "FaceNet: A Unified Embedding for Face Recognition and Clustering" (CVPR 2015)
    - Zhang et al., "Joint Face Detection and Alignment using Multi-task Cascaded Convolutional Networks" (2016)

    **Datasets:**
    - VGGFace2: Large-scale face recognition dataset

    **Open Source:**
    - PyTorch team for the deep learning framework
    - Streamlit team for the amazing app framework
    - Open source community for various tools and libraries
    """)

    st.markdown(get_t('about_developer'))

    # Features
    st.markdown("---")
    st.markdown("## ✨ Features")

    features = {
        "Face Detection": "MTCNN-based detection with landmarks",
        "1-to-1 Comparison": "Compare two faces with confidence scores",
        "1-to-Many Search": "Find matches across multiple images",
        "Batch Processing": "Process and cluster multiple faces",
        "Demo Mode": "Interactive tutorials and examples",
        "Settings Panel": "Customize thresholds and preferences",
        "Quality Analysis": "Evaluate image and face quality",
        "Heatmap Visualization": "See what the model focuses on",
        "Bilingual Support": "English and Arabic interfaces",
        "Anonymous Feedback": "Send suggestions via Telegram",
    }

    cols_per_row = 2
    feature_items = list(features.items())

    for i in range(0, len(feature_items), cols_per_row):
        cols = st.columns(cols_per_row)

        for j, col in enumerate(cols):
            if i + j < len(feature_items):
                name, desc = feature_items[i + j]
                with col:
                    st.markdown(f"""
                    <div style='padding: 1rem; 
                                background: {Config.get_color('primary', 0.05)};
                                border-radius: 8px; margin: 0.5rem 0;
                                border-left: 3px solid {Config.COLORS['primary']};'>
                        <h4 style='margin: 0; color: {Config.COLORS['primary']};'>
                            {name}
                        </h4>
                        <p style='margin: 0.5rem 0 0 0; font-size: 0.9rem;'>
                            {desc}
                        </p>
                    </div>
                    """, unsafe_allow_html=True)

    # FAQ
    st.markdown("---")
    st.markdown("## ❓ Frequently Asked Questions")

    with st.expander("How accurate is the face recognition?"):
        st.markdown(f"""
        The model achieves **{Config.MODEL_METRICS['accuracy'] * 100:.1f}% accuracy** on the validation set.

        Accuracy depends on several factors:
        - Image quality (sharp, well-lit images work best)
        - Face angle (frontal views are most accurate)
        - Face size (larger faces are easier to recognize)
        - Lighting conditions

        For best results, use clear, high-quality images with visible faces.
        """)

    with st.expander("What makes a good face photo?"):
        st.markdown("""
        **Good photos have:**
        - ✅ Clear, sharp focus
        - ✅ Good lighting (not too dark or bright)
        - ✅ Frontal or near-frontal view
        - ✅ Minimal occlusion (no sunglasses, masks)
        - ✅ High resolution (at least 640×480)

        **Avoid:**
        - ❌ Blurry or out-of-focus images
        - ❌ Very dark or overexposed photos
        - ❌ Extreme angles (profile views)
        - ❌ Heavy makeup or filters
        - ❌ Very low resolution
        """)

    with st.expander("Is my data safe?"):
        st.markdown("""
        **Absolutely!** Your privacy is our top priority.

        - All processing happens locally in your browser session
        - Images are deleted immediately after processing
        - No data is stored on servers
        - No tracking or analytics
        - No third-party data sharing

        Even feedback sent via Telegram is anonymous unless you provide your name.
        """)

    with st.expander("How do I adjust the threshold?"):
        st.markdown(f"""
        Go to the **Settings** page to adjust the similarity threshold.

        **Current threshold:** {st.session_state.threshold * 100:.0f}%

        - **Lower threshold (50-60%)**: More matches, some false positives
        - **Medium threshold (60-75%)**: Balanced, recommended for general use
        - **Higher threshold (75-90%)**: Fewer matches, very strict

        Experiment to find what works best for your use case!
        """)

    with st.expander("Can I use this commercially?"):
        st.markdown("""
        This is a demonstration/educational project built for learning purposes.

        For commercial use:
        - Review licensing terms
        - Consider privacy regulations (GDPR, CCPA, etc.)
        - Implement proper security measures
        - Get necessary permissions

        Contact the developer for commercial licensing inquiries.
        """)

    # Footer
    st.markdown("---")

    st.markdown(f"""
    <div style='text-align: center; padding: 2rem; 
                background: {Config.get_color('primary', 0.03)};
                border-radius: 10px;'>
        <h3 style='color: {Config.COLORS['primary']};'>
            {Config.APP_NAME} • v{Config.VERSION}
        </h3>
        <p style='color: gray; margin: 0.5rem 0;'>
            Advanced Face Recognition • Powered by Deep Learning
        </p>
        <p style='color: gray; margin: 0.5rem 0;'>
            {Config.VERSION_DATE}
        </p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":

    main()
