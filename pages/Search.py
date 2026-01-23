"""
FaceMatch Pro - Search Page
1-to-Many face search across multiple images
"""

import streamlit as st
import numpy as np
import pandas as pd
from pathlib import Path
import sys
import time

sys.path.append(str(Path(__file__).parent.parent))

from config import Config
from translations import get_text
from utils.model_utils import load_model, extract_embedding, compute_similarity
from utils.face_detection import detect_faces, draw_face_rectangles, crop_face, get_best_face
from utils.image_utils import load_image, validate_image, batch_load_images, add_match_badge


def get_t(key):
    """Translation helper"""
    return get_text(key, st.session_state.language)


def main():
    """Main search page"""
    st.title(f"📊 {get_t('search_title')}")
    st.markdown(f"*{get_t('search_subtitle')}*")

    st.markdown("---")

    # Load model
    with st.spinner(get_t('loading')):
        model, device = load_model()

    if model is None:
        st.error("❌ Failed to load model")
        return

    # Two-column layout for uploads
    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown(f"### 📤 {get_t('search_upload_ref')}")
        ref_file = st.file_uploader(
            get_t('btn_upload'),
            type=Config.SUPPORTED_FORMATS,
            key="ref_upload",
            label_visibility="collapsed"
        )

    with col2:
        st.markdown(f"### 📤 {get_t('search_upload_targets')}")
        target_files = st.file_uploader(
            get_t('btn_upload'),
            type=Config.SUPPORTED_FORMATS,
            accept_multiple_files=True,
            key="target_upload",
            label_visibility="collapsed"
        )

    # Process if uploads exist
    if ref_file and target_files:
        # Validate reference image
        valid, error = validate_image(ref_file)
        if not valid:
            st.error(f"Reference image: {error}")
            return

        # Check batch size
        if len(target_files) > Config.MAX_BATCH_SIZE:
            st.error(f"⚠️ Maximum {Config.MAX_BATCH_SIZE} target images allowed")
            return

        # Load reference image
        ref_image = load_image(ref_file)
        if ref_image is None:
            st.error(get_t('msg_error_loading'))
            return

        st.markdown("---")

        # Show reference image and detect face
        st.markdown("## 🎯 Reference Face Selection")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### Reference Image")
            st.image(ref_image, use_column_width=True)

        with col2:
            with st.spinner("Detecting reference face..."):
                ref_faces = detect_faces(ref_image)

            if len(ref_faces) == 0:
                st.error(get_t('msg_no_face_detected'))
                return

            # Show detected faces
            if len(ref_faces) > 1:
                st.warning(f"⚠️ {len(ref_faces)} faces detected. Please select one.")
                annotated = draw_face_rectangles(ref_image, ref_faces)
                st.image(annotated, use_column_width=True)

                # Face selector
                face_options = [f"Face {i + 1} (Conf: {face['confidence']:.2f})"
                                for i, face in enumerate(ref_faces)]
                selected_ref_idx = st.selectbox(
                    get_t('search_select_ref_face'),
                    range(len(ref_faces)),
                    format_func=lambda x: face_options[x]
                )
            else:
                st.success("✅ 1 face detected")
                st.image(ref_image, use_column_width=True)
                selected_ref_idx = 0

        # Extract reference embedding
        ref_face = ref_faces[selected_ref_idx]
        ref_crop = crop_face(ref_image, ref_face)

        # Search button
        st.markdown("---")

        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            search_button = st.button(
                f"🔍 {get_t('search_start')}",
                use_container_width=True,
                type="primary"
            )

        if search_button:
            # Extract reference embedding
            with st.spinner("Extracting reference embedding..."):
                ref_embedding = extract_embedding(model, ref_crop, device, use_tta=Config.USE_TTA)

            if ref_embedding is None:
                st.error("Failed to extract reference embedding")
                return

            # Load target images
            with st.spinner(f"Loading {len(target_files)} target images..."):
                target_images = batch_load_images(target_files, show_progress=True)

            if len(target_images) == 0:
                st.error("No valid target images loaded")
                return

            # Detect faces in all targets
            with st.spinner("Detecting faces in target images..."):
                progress_bar = st.progress(0)
                status_text = st.empty()

                target_faces_list = []
                for i, img in enumerate(target_images):
                    faces = detect_faces(img)
                    target_faces_list.append(faces)

                    progress = (i + 1) / len(target_images)
                    progress_bar.progress(progress)
                    status_text.text(f"Detecting: {i + 1}/{len(target_images)}")

                progress_bar.empty()
                status_text.empty()

            # Extract embeddings and compute similarities
            results = []

            with st.spinner("Computing similarities..."):
                progress_bar = st.progress(0)
                status_text = st.empty()

                for i, (img, faces) in enumerate(zip(target_images, target_faces_list)):
                    if len(faces) == 0:
                        # No face detected
                        results.append({
                            'image': img,
                            'filename': target_files[i].name,
                            'similarity': 0.0,
                            'match': False,
                            'face_detected': False,
                            'face_crop': None
                        })
                    else:
                        # Use best face if multiple
                        best_face = get_best_face(faces)
                        face_crop = crop_face(img, best_face)

                        # Extract embedding
                        embedding = extract_embedding(model, face_crop, device, use_tta=False)

                        if embedding is None:
                            similarity = 0.0
                        else:
                            similarity = compute_similarity(ref_embedding, embedding)

                        match = similarity >= st.session_state.threshold

                        results.append({
                            'image': img,
                            'filename': target_files[i].name,
                            'similarity': similarity,
                            'match': match,
                            'face_detected': True,
                            'face_crop': face_crop,
                            'face': best_face
                        })

                    progress = (i + 1) / len(target_images)
                    progress_bar.progress(progress)
                    status_text.text(f"Comparing: {i + 1}/{len(target_images)}")

                progress_bar.empty()
                status_text.empty()

            # Sort by similarity
            results.sort(key=lambda x: x['similarity'], reverse=True)

            # Count matches
            num_matches = sum(1 for r in results if r['match'])

            # Display results
            st.markdown("---")
            st.markdown(f"## 📊 {get_t('search_results')}")

            # Summary
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Total Images", len(results))

            with col2:
                st.metric("Faces Detected", sum(1 for r in results if r['face_detected']))

            with col3:
                st.metric("Matches Found", num_matches)

            with col4:
                match_rate = (num_matches / len(results) * 100) if len(results) > 0 else 0
                st.metric("Match Rate", f"{match_rate:.1f}%")

            # Filters and sorting
            st.markdown("---")

            col1, col2, col3 = st.columns(3)

            with col1:
                show_only_matches = st.checkbox(get_t('search_filter'), value=False)

            with col2:
                sort_option = st.selectbox(
                    get_t('search_sort'),
                    ['Similarity (High to Low)', 'Similarity (Low to High)', 'Filename']
                )

            with col3:
                grid_cols = st.slider(get_t('search_grid_cols'), 2, 5, 3)

            # Apply filters
            filtered_results = results

            if show_only_matches:
                filtered_results = [r for r in results if r['match']]

            # Apply sorting
            if sort_option == 'Similarity (Low to High)':
                filtered_results.sort(key=lambda x: x['similarity'])
            elif sort_option == 'Filename':
                filtered_results.sort(key=lambda x: x['filename'])

            # Display results grid
            if len(filtered_results) == 0:
                st.info(get_t('search_no_results'))
            else:
                st.markdown("---")
                st.markdown(f"### 🖼️ Results ({len(filtered_results)} images)")

                for i in range(0, len(filtered_results), grid_cols):
                    cols = st.columns(grid_cols)

                    for j, col in enumerate(cols):
                        if i + j < len(filtered_results):
                            result = filtered_results[i + j]

                            with col:
                                # Add badge to image
                                if result['face_detected']:
                                    img_with_badge = add_match_badge(
                                        result['image'],
                                        result['match'],
                                        result['similarity'],
                                        position='top-right'
                                    )
                                    st.image(img_with_badge, use_column_width=True)
                                else:
                                    st.image(result['image'], use_column_width=True)

                                # Info
                                if result['face_detected']:
                                    st.metric(
                                        "Similarity",
                                        f"{result['similarity'] * 100:.1f}%",
                                        delta="Match" if result['match'] else "No Match"
                                    )
                                else:
                                    st.caption("⚠️ No face detected")

                                st.caption(f"📄 {result['filename']}")

            # Export options
            st.markdown("---")
            st.markdown("### 📥 Export Options")

            col1, col2 = st.columns(2)

            with col1:
                # Export as CSV
                if st.button("📊 Export Results (CSV)", use_container_width=True):
                    df = pd.DataFrame([
                        {
                            'Filename': r['filename'],
                            'Similarity': f"{r['similarity']:.4f}",
                            'Match': r['match'],
                            'Face Detected': r['face_detected']
                        }
                        for r in results
                    ])

                    csv = df.to_csv(index=False)

                    st.download_button(
                        "Download CSV",
                        data=csv,
                        file_name="face_search_results.csv",
                        mime="text/csv",
                        use_container_width=True
                    )

            with col2:
                # Download matched pairs (placeholder)
                st.button(
                    f"📦 {get_t('search_download_matches')}",
                    use_container_width=True,
                    disabled=True
                )
                st.caption("Coming soon")

    else:
        # Instructions
        st.info("📌 Upload one reference image and multiple target images to search")

        with st.expander("💡 How Face Search Works"):
            st.markdown("""
            ### Search Process:

            1. **Upload Reference**: Select one image with the person you're looking for
            2. **Upload Targets**: Upload multiple images to search through
            3. **Face Detection**: System detects faces in all images
            4. **Embedding Extraction**: Converts each face to a 512-dimensional vector
            5. **Similarity Computation**: Compares reference face with all target faces
            6. **Results**: Shows matches sorted by similarity score

            ### Tips:
            - Reference image should have a clear, well-lit face
            - Works best with frontal or near-frontal views
            - Can handle up to {MAX_BATCH_SIZE} target images at once
            - Automatically selects best face if multiple detected

            ### Use Cases:
            - Find photos of a specific person in a collection
            - Identify duplicates or similar images
            - Search through event photos
            - Organize photo library by person
            """.format(MAX_BATCH_SIZE=Config.MAX_BATCH_SIZE))


if __name__ == "__main__":
    main()