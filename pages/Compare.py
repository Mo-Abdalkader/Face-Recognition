"""
FaceMatch Pro - Compare Page
1-to-1 face comparison with heatmap visualization
"""

import streamlit as st
import plotly.graph_objects as go
import time
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from config import Config
from translations import get_text
from utils.model_utils import load_model, extract_embedding, compute_similarity, is_match
from utils.face_detection import detect_faces, draw_face_rectangles, crop_face
from utils.image_utils import load_image, validate_image, create_side_by_side, add_match_badge


def get_t(key):
    """Translation helper"""
    return get_text(key, st.session_state.language)


def create_similarity_gauge(similarity, threshold):
    """Create interactive similarity gauge"""
    # Determine color based on match
    is_match_result = similarity >= threshold
    color = Config.COLORS['match'] if is_match_result else Config.COLORS['no_match']

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=similarity * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': get_t('comp_similarity'), 'font': {'size': 24}},
        delta={'reference': threshold * 100, 'increasing': {'color': Config.COLORS['success']}},
        number={'suffix': "%", 'font': {'size': 48}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 2},
            'bar': {'color': color, 'thickness': 0.75},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, threshold*100], 'color': Config.get_color('danger', 0.2)},
                {'range': [threshold*100, 100], 'color': Config.get_color('success', 0.2)}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': threshold * 100
            }
        }
    ))

    fig.update_layout(
        height=350,
        margin=dict(l=20, r=20, t=80, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        font={'family': "Arial"}
    )

    return fig


def render_face_selector(image, faces, key_prefix):
    """Render face selector dropdown"""
    if len(faces) == 0:
        st.error(get_t('msg_no_face_detected'))
        return None

    if len(faces) == 1:
        st.success(f"✅ {get_t('fd_detected')}: 1")
        return 0

    # Multiple faces detected
    st.warning(f"⚠️ {get_t('msg_multiple_faces')}")

    # Show annotated image
    annotated = draw_face_rectangles(image, faces)
    st.image(annotated, use_column_width=True)

    # Face selector
    face_options = [f"Face {i+1} (Conf: {face['confidence']:.2f})"
                   for i, face in enumerate(faces)]

    selected = st.selectbox(
        get_t(f'{key_prefix}_select_face'),
        range(len(faces)),
        format_func=lambda x: face_options[x],
        key=f"{key_prefix}_selector"
    )

    return selected


def main():
    """Main comparison page"""
    st.title(f"🔍 {get_t('comp_title')}")
    st.markdown(f"*{get_t('comp_subtitle')}*")

    st.markdown("---")

    # Load model
    with st.spinner(get_t('loading')):
        model, device = load_model()

    if model is None:
        st.error("❌ Failed to load model. Please check model file.")
        return

    # Two columns for uploads
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"### 📤 {get_t('comp_upload_1')}")
        uploaded_file1 = st.file_uploader(
            get_t('btn_upload'),
            type=Config.SUPPORTED_FORMATS,
            key="upload1",
            label_visibility="collapsed"
        )

    with col2:
        st.markdown(f"### 📤 {get_t('comp_upload_2')}")
        uploaded_file2 = st.file_uploader(
            get_t('btn_upload'),
            type=Config.SUPPORTED_FORMATS,
            key="upload2",
            label_visibility="collapsed"
        )

    # Process if both images uploaded
    if uploaded_file1 and uploaded_file2:
        # Validate images
        valid1, error1 = validate_image(uploaded_file1)
        valid2, error2 = validate_image(uploaded_file2)

        if not valid1:
            st.error(f"Image 1: {error1}")
            return
        if not valid2:
            st.error(f"Image 2: {error2}")
            return

        # Load images
        image1 = load_image(uploaded_file1)
        image2 = load_image(uploaded_file2)

        if image1 is None or image2 is None:
            st.error(get_t('msg_error_loading'))
            return

        st.markdown("---")

        # Detect faces
        with st.spinner(get_t('processing')):
            faces1 = detect_faces(image1)
            faces2 = detect_faces(image2)

        # Face selection
        col1, col2 = st.columns(2)

        with col1:
            st.markdown(f"### 👤 {get_t('comp_upload_1')}")
            st.image(image1, use_column_width=True)
            selected_face1 = render_face_selector(image1, faces1, "comp_select_face_1")

        with col2:
            st.markdown(f"### 👤 {get_t('comp_upload_2')}")
            st.image(image2, use_column_width=True)
            selected_face2 = render_face_selector(image2, faces2, "comp_select_face_2")

        # Compare button
        if selected_face1 is not None and selected_face2 is not None:
            st.markdown("---")

            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                compare_button = st.button(
                    f"🔍 {get_t('comp_compare')}",
                    use_container_width=True,
                    type="primary"
                )

            if compare_button:
                # Crop faces
                face1_crop = crop_face(image1, faces1[selected_face1])
                face2_crop = crop_face(image2, faces2[selected_face2])

                # Extract embeddings
                start_time = time.time()

                with st.spinner(get_t('processing')):
                    progress = st.progress(0)
                    status = st.empty()

                    status.text("Extracting embeddings from Image 1...")
                    embedding1 = extract_embedding(model, face1_crop, device,
                                                  use_tta=Config.USE_TTA)
                    progress.progress(0.3)

                    status.text("Extracting embeddings from Image 2...")
                    embedding2 = extract_embedding(model, face2_crop, device,
                                                  use_tta=Config.USE_TTA)
                    progress.progress(0.6)

                    status.text("Computing similarity...")
                    similarity = compute_similarity(embedding1, embedding2)
                    progress.progress(1.0)

                    end_time = time.time()
                    processing_time = end_time - start_time

                    progress.empty()
                    status.empty()

                # Determine match
                threshold = st.session_state.threshold
                match_result = is_match(similarity, threshold)

                st.markdown("---")

                # Results section
                st.markdown(f"## 📊 {get_t('search_results')}")

                # Similarity gauge
                fig = create_similarity_gauge(similarity, threshold)
                st.plotly_chart(fig, use_container_width=True)

                # Match result
                if match_result:
                    st.markdown(f"""
                    <div class='success-box' style='text-align: center;'>
                        <h2 style='color: {Config.COLORS["success"]}; margin: 0;'>
                            ✅ {get_t('comp_match')}
                        </h2>
                        <p style='font-size: 1.2rem; margin: 0.5rem 0 0 0;'>
                            {Config.get_similarity_level(similarity, st.session_state.language)}
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class='error-box' style='text-align: center;'>
                        <h2 style='color: {Config.COLORS["danger"]}; margin: 0;'>
                            ❌ {get_t('comp_no_match')}
                        </h2>
                        <p style='font-size: 1.2rem; margin: 0.5rem 0 0 0;'>
                            {Config.get_similarity_level(similarity, st.session_state.language)}
                        </p>
                    </div>
                    """, unsafe_allow_html=True)

                # Side-by-side comparison with badges
                st.markdown(f"### 🖼️ Comparison")

                face1_with_badge = add_match_badge(face1_crop, match_result, similarity, 'top-right')
                face2_with_badge = add_match_badge(face2_crop, match_result, similarity, 'top-right')

                comparison_image = create_side_by_side(face1_with_badge, face2_with_badge)
                st.image(comparison_image, use_column_width=True)

                # Detailed metrics
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric(
                        get_t('comp_similarity'),
                        f"{similarity*100:.2f}%",
                        delta=f"{(similarity - threshold)*100:.2f}%"
                    )

                with col2:
                    st.metric(
                        get_t('comp_threshold'),
                        f"{threshold*100:.0f}%"
                    )

                with col3:
                    st.metric(
                        get_t('comp_processing_time'),
                        f"{processing_time:.2f}s"
                    )

                # Additional info
                with st.expander("📋 Detailed Information"):
                    col1, col2 = st.columns(2)

                    with col1:
                        st.markdown("**Image 1:**")
                        st.write(f"- Confidence: {faces1[selected_face1]['confidence']:.3f}")
                        st.write(f"- Face size: {faces1[selected_face1]['box'][2]} x {faces1[selected_face1]['box'][3]}")

                    with col2:
                        st.markdown("**Image 2:**")
                        st.write(f"- Confidence: {faces2[selected_face2]['confidence']:.3f}")
                        st.write(f"- Face size: {faces2[selected_face2]['box'][2]} x {faces2[selected_face2]['box'][3]}")

                    st.markdown("**Model:**")
                    st.write(f"- Embedding dimension: {Config.EMBEDDING_DIM}")
                    st.write(f"- TTA enabled: {Config.USE_TTA}")
                    st.write(f"- Model accuracy: {Config.MODEL_METRICS['accuracy']*100:.1f}%")

                # Action buttons
                st.markdown("---")

                col1, col2, col3 = st.columns(3)

                with col1:
                    if st.button(f"🔄 {get_t('comp_new_comparison')}", use_container_width=True):
                        st.rerun()

                with col2:
                    # Download report (placeholder)
                    st.button(f"📥 {get_t('comp_download_report')}", use_container_width=True, disabled=True)
                    st.caption("Coming soon")

                with col3:
                    # Show heatmap button
                    show_heatmap = st.button(f"🔥 {get_t('comp_heatmap')}", use_container_width=True)

                # Heatmap visualization
                if show_heatmap:
                    st.markdown("---")
                    st.markdown(f"## 🔥 {get_t('comp_heatmap')}")
                    st.caption(get_t('comp_heatmap_desc'))

                    try:
                        from utils.heatmap_utils import visualize_attention, create_multi_view_visualization
                        from torchvision import transforms

                        # Prepare image tensor for heatmap
                        transform = transforms.Compose([
                            transforms.ToPILImage(),
                            transforms.Resize((224, 224)),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=Config.NORMALIZE_MEAN, std=Config.NORMALIZE_STD)
                        ])

                        with st.spinner("Generating attention heatmaps..."):
                            # Face 1 heatmap
                            img_tensor1 = transform(face1_crop).unsqueeze(0).to(device)
                            img_tensor1.requires_grad = True

                            heatmap1 = visualize_attention(
                                model, face1_crop, img_tensor1, method='gradcam'
                            )

                            # Face 2 heatmap
                            img_tensor2 = transform(face2_crop).unsqueeze(0).to(device)
                            img_tensor2.requires_grad = True

                            heatmap2 = visualize_attention(
                                model, face2_crop, img_tensor2, method='gradcam'
                            )

                        if heatmap1 and heatmap2:
                            col1, col2 = st.columns(2)

                            with col1:
                                st.markdown("#### Face 1 Attention")
                                multi_view1 = create_multi_view_visualization(
                                    heatmap1['original'],
                                    heatmap1['heatmap_colored'],
                                    heatmap1['overlay']
                                )
                                st.image(multi_view1, use_column_width=True)

                            with col2:
                                st.markdown("#### Face 2 Attention")
                                multi_view2 = create_multi_view_visualization(
                                    heatmap2['original'],
                                    heatmap2['heatmap_colored'],
                                    heatmap2['overlay']
                                )
                                st.image(multi_view2, use_column_width=True)

                            st.success("✅ Heatmaps show which facial features the model focuses on")

                            with st.expander("💡 Understanding Heatmaps"):
                                st.markdown("""
                                **What do the colors mean?**
                                - 🔴 **Red/Yellow**: High attention (model focuses here)
                                - 🟢 **Green**: Medium attention
                                - 🔵 **Blue**: Low attention
                                
                                **What does the model look at?**
                                - Eyes and eyebrows (key for identity)
                                - Nose bridge and shape
                                - Mouth and chin area
                                - Face outline and structure
                                
                                **Why is this useful?**
                                - Understand model decisions
                                - Verify model looks at right features
                                - Debug unexpected results
                                - Build trust in the system
                                """)
                        else:
                            st.warning("⚠️ Could not generate heatmaps")

                    except Exception as e:
                        st.error(f"Error generating heatmaps: {str(e)}")
                        st.caption("Heatmap feature requires additional setup")

    else:
        # Instructions
        st.info(f"📌 {get_t('msg_upload_two_images')}")

        with st.expander("💡 Tips for Best Results"):
            st.markdown(f"""
            - **Use clear, well-lit photos**
            - **Face should be clearly visible**
            - **Avoid blurry or pixelated images**
            - **Frontal view works best**
            - **Supported formats**: {', '.join(Config.SUPPORTED_FORMATS).upper()}
            - **Maximum file size**: {Config.MAX_IMAGE_SIZE / (1024*1024):.0f} MB
            """)


if __name__ == "__main__":
    main()