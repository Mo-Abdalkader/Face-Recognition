"""
FaceMatch Pro - Demo Mode
Interactive demonstrations with sample images
"""

import streamlit as st
from pathlib import Path
import sys
import os

sys.path.append(str(Path(__file__).parent.parent))

from config import Config
from translations import get_text
from utils.model_utils import load_model, extract_embedding, compute_similarity
from utils.face_detection import detect_faces, crop_face, get_best_face
from utils.image_utils import load_image_from_path, create_side_by_side, add_match_badge
import plotly.graph_objects as go

from utils.session_utils import init_session_state
init_session_state()

def get_t(key):
    """Translation helper"""
    return get_text(key, st.session_state.language)


def create_similarity_gauge_simple(similarity, threshold):
    """Create simple similarity gauge"""
    is_match = similarity >= threshold
    color = Config.COLORS['match'] if is_match else Config.COLORS['no_match']

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=similarity * 100,
        title={'text': "Similarity"},
        number={'suffix': "%"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': color},
            'threshold': {
                'line': {'color': "red", 'width': 3},
                'thickness': 0.75,
                'value': threshold * 100
            }
        }
    ))

    fig.update_layout(height=250, margin=dict(l=20, r=20, t=50, b=20))

    return fig


def load_demo_images():
    """Load demo images from assets folder"""
    demo_dir = Config.DEMO_DIR

    # Check if demo directory exists
    if not demo_dir.exists():
        return None

    # Get all image files
    image_files = list(demo_dir.glob('*.jpg')) + list(demo_dir.glob('*.png'))

    if len(image_files) == 0:
        return None

    return {img.stem: img for img in image_files}


def run_demo_comparison(model, device, img1_path, img2_path, threshold):
    """Run a demo comparison"""
    # Load images
    img1 = load_image_from_path(img1_path)
    img2 = load_image_from_path(img2_path)

    if img1 is None or img2 is None:
        return None

    # Detect faces
    faces1 = detect_faces(img1)
    faces2 = detect_faces(img2)

    if len(faces1) == 0 or len(faces2) == 0:
        return {
            'error': 'No face detected in one or both images',
            'img1': img1,
            'img2': img2
        }

    # Get best faces
    face1 = get_best_face(faces1)
    face2 = get_best_face(faces2)

    # Crop faces
    face1_crop = crop_face(img1, face1)
    face2_crop = crop_face(img2, face2)

    # Extract embeddings
    emb1 = extract_embedding(model, face1_crop, device, use_tta=False)
    emb2 = extract_embedding(model, face2_crop, device, use_tta=False)

    # Compute similarity
    similarity = compute_similarity(emb1, emb2)
    match = similarity >= threshold

    return {
        'img1': img1,
        'img2': img2,
        'face1_crop': face1_crop,
        'face2_crop': face2_crop,
        'similarity': similarity,
        'match': match,
        'threshold': threshold
    }


def main():
    """Main demo page"""
    st.title(f"🎯 {get_t('demo_title')}")
    st.markdown(f"*{get_t('demo_subtitle')}*")

    st.markdown("---")

    # Load model
    with st.spinner(get_t('loading')):
        model, device = load_model()

    if model is None:
        st.error("❌ Failed to load model")
        return

    # Check for demo images
    demo_images = load_demo_images()

    if demo_images is None or len(demo_images) == 0:
        st.warning("⚠️ No demo images found")

        st.markdown("""
        ### 📸 Add Your Demo Images

        To use demo mode, add your images to:
        ```
        assets/demo_images/
        ```

        **Recommended images:**
        - `same_person_1.jpg` and `same_person_2.jpg` (same person, different angles/lighting)
        - `different_person_1.jpg` and `different_person_2.jpg` (different people)
        - `similar_1.jpg` and `similar_2.jpg` (similar but different people)

        Use your Photoshop-edited images as mentioned!
        """)

        # Show example using uploaded images instead
        st.markdown("---")
        st.markdown("### 📤 Or Upload Your Own Demo Images")

        col1, col2 = st.columns(2)

        with col1:
            demo_file1 = st.file_uploader("First Image", type=['jpg', 'png'], key='demo1')

        with col2:
            demo_file2 = st.file_uploader("Second Image", type=['jpg', 'png'], key='demo2')

        if demo_file1 and demo_file2:
            from utils.image_utils import load_image

            img1 = load_image(demo_file1)
            img2 = load_image(demo_file2)

            if img1 is not None and img2 is not None:
                # Save temporarily and run demo
                import tempfile

                with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp1:
                    import cv2
                    cv2.imwrite(tmp1.name, cv2.cvtColor(img1, cv2.COLOR_RGB2BGR))
                    tmp1_path = tmp1.name

                with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp2:
                    cv2.imwrite(tmp2.name, cv2.cvtColor(img2, cv2.COLOR_RGB2BGR))
                    tmp2_path = tmp2.name

                if st.button("🔍 Run Demo Comparison", type="primary"):
                    with st.spinner("Processing..."):
                        result = run_demo_comparison(
                            model, device,
                            tmp1_path, tmp2_path,
                            st.session_state.threshold
                        )

                    if result and 'error' not in result:
                        # Show results
                        st.markdown("---")

                        col1, col2 = st.columns(2)

                        with col1:
                            img1_badge = add_match_badge(
                                result['face1_crop'],
                                result['match'],
                                result['similarity']
                            )
                            st.image(img1_badge, use_column_width=True)

                        with col2:
                            img2_badge = add_match_badge(
                                result['face2_crop'],
                                result['match'],
                                result['similarity']
                            )
                            st.image(img2_badge, use_column_width=True)

                        # Gauge
                        fig = create_similarity_gauge_simple(
                            result['similarity'],
                            result['threshold']
                        )
                        st.plotly_chart(fig, use_container_width=True)

                        # Result
                        if result['match']:
                            st.success(f"✅ MATCH - Similarity: {result['similarity'] * 100:.1f}%")
                        else:
                            st.error(f"❌ NO MATCH - Similarity: {result['similarity'] * 100:.1f}%")

                    # Cleanup
                    try:
                        os.unlink(tmp1_path)
                        os.unlink(tmp2_path)
                    except:
                        pass

        return

    # Demo scenarios
    st.markdown("## 📋 Demo Scenarios")

    scenarios = Config.DEMO_SCENARIOS[st.session_state.language]

    selected_scenario = st.selectbox(
        get_t('demo_select'),
        range(len(scenarios)),
        format_func=lambda x: scenarios[x]
    )

    # Map scenarios to image pairs
    # This assumes you have specific naming in demo_images folder
    scenario_mappings = [
        ('same_person_1', 'same_person_2'),  # Same person - different angles
        ('same_light_1', 'same_light_2'),  # Same person - different lighting
        ('similar_1', 'similar_2'),  # Similar people
        ('diff_1', 'diff_2'),  # Different people
        ('high_quality', 'low_quality')  # Quality comparison
    ]

    if selected_scenario < len(scenario_mappings):
        img1_name, img2_name = scenario_mappings[selected_scenario]

        # Check if images exist
        img1_path = None
        img2_path = None

        for name, path in demo_images.items():
            if img1_name in name:
                img1_path = path
            if img2_name in name:
                img2_path = path

        if img1_path and img2_path:
            # Run demo button
            if st.button(f"▶️ {get_t('demo_run')}", type="primary", use_container_width=True):
                with st.spinner("Running demo..."):
                    result = run_demo_comparison(
                        model, device,
                        img1_path, img2_path,
                        st.session_state.threshold
                    )

                if result and 'error' not in result:
                    st.markdown("---")
                    st.markdown("### 📊 Demo Results")

                    # Show images
                    col1, col2 = st.columns(2)

                    with col1:
                        st.markdown("#### Image 1")
                        img1_badge = add_match_badge(
                            result['img1'],
                            result['match'],
                            result['similarity']
                        )
                        st.image(img1_badge, use_column_width=True)

                    with col2:
                        st.markdown("#### Image 2")
                        img2_badge = add_match_badge(
                            result['img2'],
                            result['match'],
                            result['similarity']
                        )
                        st.image(img2_badge, use_column_width=True)

                    # Similarity gauge
                    fig = create_similarity_gauge_simple(
                        result['similarity'],
                        result['threshold']
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    # Result message
                    if result['match']:
                        st.markdown(f"""
                        <div class='success-box' style='text-align: center;'>
                            <h2 style='color: {Config.COLORS["success"]}; margin: 0;'>
                                ✅ MATCH
                            </h2>
                            <p style='font-size: 1.2rem; margin: 0.5rem 0 0 0;'>
                                Similarity: {result['similarity'] * 100:.1f}%
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class='error-box' style='text-align: center;'>
                            <h2 style='color: {Config.COLORS["danger"]}; margin: 0;'>
                                ❌ NO MATCH
                            </h2>
                            <p style='font-size: 1.2rem; margin: 0.5rem 0 0 0;'>
                                Similarity: {result['similarity'] * 100:.1f}%
                            </p>
                        </div>
                        """, unsafe_allow_html=True)

                    # Explanation
                    st.markdown("---")
                    st.markdown("### 💡 Understanding the Results")

                    col1, col2 = st.columns(2)

                    with col1:
                        st.markdown(f"""
                        **Threshold:** {result['threshold'] * 100:.0f}%
                        - Similarity above this = Match
                        - Similarity below this = No Match
                        """)

                    with col2:
                        level = Config.get_similarity_level(
                            result['similarity'],
                            st.session_state.language
                        )
                        st.markdown(f"""
                        **Similarity Level:** {level}
                        - Score: {result['similarity'] * 100:.1f}%
                        - Match: {'Yes' if result['match'] else 'No'}
                        """)

                elif result and 'error' in result:
                    st.error(f"⚠️ {result['error']}")
        else:
            st.warning(f"⚠️ Demo images not found: `{img1_name}` and `{img2_name}`")

    # Interactive tutorial
    st.markdown("---")
    st.markdown(f"## 📚 {get_t('demo_tutorial')}")

    with st.expander("🎓 Understanding Face Recognition"):
        st.markdown("""
        ### How It Works:

        1. **Face Detection (MTCNN)**
           - Locates faces in images
           - Draws bounding boxes
           - Extracts facial landmarks

        2. **Embedding Extraction (ResNet50)**
           - Converts face to 512-dimensional vector
           - Each vector represents unique facial features
           - Similar faces have similar vectors

        3. **Similarity Computation**
           - Compares two face vectors
           - Uses cosine similarity (0-1 scale)
           - Higher score = more similar

        4. **Threshold Matching**
           - Compares similarity to threshold
           - Above threshold = Match
           - Below threshold = No Match
        """)

    with st.expander(f"🎯 {get_t('demo_understanding')}"):
        st.markdown(f"""
        ### What is a Threshold?

        The threshold is the minimum similarity score needed to consider two faces a "match".

        **Current Threshold:** {st.session_state.threshold * 100:.0f}%

        #### Example Scenarios:

        **Threshold = 50%** (Too Low)
        - Catches all matches ✅
        - But also many false matches ❌
        - Use for: Finding similar-looking people

        **Threshold = 70%** (Recommended)
        - Good balance ✅
        - Accurate matching ✅
        - Few false positives ✅

        **Threshold = 90%** (Too High)
        - Very strict ✅
        - May miss real matches ❌
        - Use for: High-security applications

        **Adjust in Settings page!**
        """)

    with st.expander(f"📸 {get_t('demo_quality')}"):
        st.markdown("""
        ### Image Quality Matters!

        **Good Quality Images:**
        - ✅ Clear, in-focus faces
        - ✅ Good lighting (not too dark/bright)
        - ✅ Frontal or near-frontal view
        - ✅ Minimal occlusion (no sunglasses/masks)
        - ✅ High resolution (min 640×480)

        **Poor Quality Images:**
        - ❌ Blurry or out-of-focus
        - ❌ Very dark or overexposed
        - ❌ Extreme angles (profile view)
        - ❌ Heavy occlusion
        - ❌ Very low resolution

        **Pro Tip:** Better quality images = better accuracy!
        """)


if __name__ == "__main__":

    main()
