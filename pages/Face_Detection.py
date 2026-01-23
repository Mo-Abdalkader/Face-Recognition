"""
FaceMatch Pro - Face Detection Page
Standalone face detection with MTCNN
"""

import streamlit as st
import cv2
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from config import Config
from translations import get_text
from utils.face_detection import (
    detect_faces,
    draw_face_rectangles,
    get_face_statistics,
    create_face_grid,
    batch_detect_faces
)
from utils.image_utils import load_image, validate_image, convert_to_bytes


def get_t(key):
    """Translation helper"""
    return get_text(key, st.session_state.language)


def render_face_info(face, idx):
    """Render information about a detected face"""
    x, y, w, h = face['box']
    confidence = face['confidence']

    st.markdown(f"""
    <div style='background: {Config.get_color('primary', 0.05)}; 
                padding: 1rem; border-radius: 8px; margin-bottom: 0.5rem;
                border-left: 4px solid {Config.COLORS['primary']}'>
        <h4 style='margin: 0;'>Face {idx + 1}</h4>
        <p style='margin: 0.5rem 0 0 0; font-size: 0.9rem;'>
            📍 Position: ({x}, {y})<br>
            📏 Size: {w} × {h} pixels<br>
            🎯 Confidence: {confidence:.3f} ({confidence * 100:.1f}%)
        </p>
    </div>
    """, unsafe_allow_html=True)


def main():
    """Main face detection page"""
    st.title(f"👤 {get_t('fd_title')}")
    st.markdown(f"*{get_t('fd_subtitle')}*")

    st.markdown("---")

    # Instructions
    st.info(f"📌 {get_t('fd_instructions')}")

    # Upload mode selector
    upload_mode = st.radio(
        "Upload Mode",
        ["Single Image", "Multiple Images"],
        horizontal=True
    )

    st.markdown("---")

    if upload_mode == "Single Image":
        # Single image mode
        uploaded_file = st.file_uploader(
            f"📤 {get_t('fd_upload')}",
            type=Config.SUPPORTED_FORMATS,
            key="single_upload"
        )

        if uploaded_file:
            # Validate
            valid, error = validate_image(uploaded_file)
            if not valid:
                st.error(error)
                return

            # Load image
            image = load_image(uploaded_file)
            if image is None:
                st.error(get_t('msg_error_loading'))
                return

            # Display original
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("### 📷 Original Image")
                st.image(image, use_column_width=True)

            # Detect faces
            with st.spinner(get_t('processing')):
                faces = detect_faces(image)

            if len(faces) == 0:
                with col2:
                    st.markdown("### 🔍 Detection Result")
                    st.warning(get_t('fd_no_faces'))
                return

            # Draw rectangles
            annotated = draw_face_rectangles(image, faces, labels=True)

            with col2:
                st.markdown(f"### 🎯 {get_t('fd_detected')}: {len(faces)}")
                st.image(annotated, use_column_width=True)

            # Statistics
            st.markdown("---")
            st.markdown(f"## 📊 Detection Statistics")

            stats = get_face_statistics(faces)

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric(get_t('fd_count'), stats['count'])

            with col2:
                st.metric("Avg Confidence", f"{stats['avg_confidence']:.3f}")

            with col3:
                st.metric("Min Confidence", f"{stats['min_confidence']:.3f}")

            with col4:
                st.metric("Max Confidence", f"{stats['max_confidence']:.3f}")

            # Individual face info
            st.markdown("---")
            st.markdown(f"## 👥 Individual Faces")

            cols_per_row = 3
            for i in range(0, len(faces), cols_per_row):
                cols = st.columns(cols_per_row)
                for j, col in enumerate(cols):
                    if i + j < len(faces):
                        with col:
                            render_face_info(faces[i + j], i + j)

            # Download button
            st.markdown("---")

            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                # Convert annotated image to bytes
                img_bytes = convert_to_bytes(annotated, format='PNG')

                st.download_button(
                    f"📥 {get_t('fd_download')}",
                    data=img_bytes,
                    file_name=f"detected_faces_{uploaded_file.name}",
                    mime="image/png",
                    use_container_width=True,
                    type="primary"
                )

    else:
        # Multiple images mode
        uploaded_files = st.file_uploader(
            f"📤 Upload Multiple Images",
            type=Config.SUPPORTED_FORMATS,
            accept_multiple_files=True,
            key="multi_upload"
        )

        if uploaded_files:
            if len(uploaded_files) > Config.MAX_BATCH_SIZE:
                st.error(f"⚠️ Maximum {Config.MAX_BATCH_SIZE} images allowed. You uploaded {len(uploaded_files)}.")
                return

            # Load all images
            with st.spinner("Loading images..."):
                images = []
                for file in uploaded_files:
                    valid, error = validate_image(file)
                    if valid:
                        img = load_image(file)
                        if img is not None:
                            images.append(img)

            if len(images) == 0:
                st.error("No valid images loaded")
                return

            st.success(f"✅ Loaded {len(images)} images")

            # Detect faces in all images
            with st.spinner("Detecting faces..."):
                all_faces = batch_detect_faces(images, show_progress=True)

            # Summary statistics
            total_faces = sum(len(faces) for faces in all_faces)
            images_with_faces = sum(1 for faces in all_faces if len(faces) > 0)

            st.markdown("---")
            st.markdown("## 📊 Batch Detection Summary")

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Total Images", len(images))

            with col2:
                st.metric("Images with Faces", images_with_faces)

            with col3:
                st.metric("Total Faces Found", total_faces)

            with col4:
                avg_faces = total_faces / len(images) if len(images) > 0 else 0
                st.metric("Avg Faces/Image", f"{avg_faces:.1f}")

            # Display results
            st.markdown("---")
            st.markdown("## 🖼️ Detection Results")

            # Grid display
            grid_cols = st.slider("Grid Columns", 2, 4, 3)

            for i in range(0, len(images), grid_cols):
                cols = st.columns(grid_cols)

                for j, col in enumerate(cols):
                    if i + j < len(images):
                        with col:
                            img_idx = i + j
                            image = images[img_idx]
                            faces = all_faces[img_idx]

                            # Draw rectangles if faces found
                            if len(faces) > 0:
                                annotated = draw_face_rectangles(image, faces, labels=True)
                                st.image(annotated, use_column_width=True)
                                st.caption(f"Image {img_idx + 1}: {len(faces)} face(s)")
                            else:
                                st.image(image, use_column_width=True)
                                st.caption(f"Image {img_idx + 1}: No faces")

            # Create face grid (all detected faces)
            if total_faces > 0:
                st.markdown("---")
                st.markdown("## 👥 All Detected Faces")

                with st.spinner("Creating face grid..."):
                    face_grid = create_face_grid(images, all_faces, grid_cols=6)

                if face_grid is not None:
                    st.image(face_grid, use_column_width=True)
                    st.caption(f"Grid of {total_faces} detected faces")

    # Tips
    st.markdown("---")

    with st.expander("💡 Tips for Best Detection Results"):
        st.markdown("""
        ### For Better Face Detection:

        **✅ Good Images:**
        - Clear, well-lit photos
        - Face is clearly visible
        - Minimal blur or motion
        - Frontal or slight angle view
        - High resolution (min 640×480)

        **❌ Avoid:**
        - Very dark or overexposed images
        - Extreme angles (profile view)
        - Heavy blur or low resolution
        - Faces smaller than 20×20 pixels
        - Multiple overlapping faces

        **MTCNN Settings:**
        - Minimum face size: 20 pixels
        - Confidence threshold: 0.9
        - Works best with faces > 50×50 pixels
        """)

    # Technical info
    with st.expander("🔧 Technical Information"):
        st.markdown(f"""
        ### Detection Algorithm

        **MTCNN (Multi-task Cascaded Convolutional Networks)**
        - 3-stage cascade architecture
        - Detects faces, bounding boxes, and facial landmarks
        - High accuracy on various face sizes and angles

        **Settings:**
        - Min face size: {Config.MTCNN_MIN_FACE_SIZE} pixels
        - Detection thresholds: {Config.MTCNN_THRESHOLDS}
        - Scale factor: {Config.MTCNN_FACTOR}

        **Output:**
        - Bounding box coordinates (x, y, width, height)
        - Confidence score (0-1)
        - Facial landmarks (eyes, nose, mouth)
        """)


if __name__ == "__main__":
    main()
