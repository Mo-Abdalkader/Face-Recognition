"""
FaceMatch Pro - Face Detection Page
Professional face detection with advanced controls
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
        <h4 style='margin: 0; color: {Config.COLORS['primary']};'>Face {idx + 1}</h4>
        <p style='margin: 0.5rem 0 0 0; font-size: 0.9rem;'>
            📍 Position: ({x}, {y})<br>
            📐 Size: {w} × {h} pixels<br>
            🎯 Confidence: {confidence:.3f} ({confidence * 100:.1f}%)
        </p>
    </div>
    """, unsafe_allow_html=True)


def main():
    """Main face detection page"""
    st.title(f"👤 {get_t('fd_title')}")
    st.markdown(f"*{get_t('fd_subtitle')}*")

    st.markdown("---")

    # Professional header with quick info
    st.markdown(f"""
    <div style='background: linear-gradient(135deg, {Config.get_color('primary', 0.1)}, 
                {Config.get_color('secondary', 0.1)}); 
                padding: 1.5rem; border-radius: 10px; margin-bottom: 2rem;'>
        <h3 style='margin: 0; color: {Config.COLORS['primary']};'>
            🎯 Advanced Face Detection
        </h3>
        <p style='margin: 0.5rem 0 0 0; font-size: 1rem;'>
            Upload images to automatically detect faces with high precision MTCNN algorithm
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Upload mode selector
    upload_mode = st.radio(
        "📁 Upload Mode",
        ["Single Image", "Multiple Images"],
        horizontal=True,
        help="Choose whether to process one image or multiple images at once"
    )

    st.markdown("---")

    if upload_mode == "Single Image":
        # ==================== SINGLE IMAGE MODE ====================
        
        # Upload section
        col1, col2 = st.columns([2, 1])
        
        with col1:
            uploaded_file = st.file_uploader(
                f"📤 {get_t('fd_upload')}",
                type=Config.SUPPORTED_FORMATS,
                key="single_upload",
                help=f"Supported formats: {', '.join(Config.SUPPORTED_FORMATS).upper()}"
            )
        
        with col2:
            if uploaded_file:
                st.success(f"✅ File: {uploaded_file.name}")
                st.caption(f"Size: {uploaded_file.size / 1024:.1f} KB")

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

            st.markdown("---")

            # ==================== DETECTION CONTROLS ====================
            st.markdown("### ⚙️ Detection Controls")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                confidence_threshold = st.slider(
                    "🎯 Confidence Threshold",
                    min_value=0.5,
                    max_value=1.0,
                    value=0.9,
                    step=0.05,
                    help="Minimum confidence for face detection (higher = stricter)"
                )
            
            with col2:
                border_thickness = st.slider(
                    "✏️ Border Thickness",
                    min_value=1,
                    max_value=10,
                    value=2,
                    step=1,
                    help="Thickness of face bounding boxes"
                )
            
            with col3:
                border_color_option = st.selectbox(
                    "🎨 Border Color",
                    ["Purple (Default)", "Red", "Green", "Blue", "Yellow", "Orange", "Cyan"],
                    help="Color of face bounding boxes"
                )
                
                # Map color names to BGR values
                color_map = {
                    "Purple (Default)": (138, 43, 226),
                    "Red": (0, 0, 255),
                    "Green": (0, 255, 0),
                    "Blue": (255, 0, 0),
                    "Yellow": (0, 255, 255),
                    "Orange": (0, 165, 255),
                    "Cyan": (255, 255, 0)
                }
                border_color = color_map[border_color_option]
            
            with col4:
                show_options = st.multiselect(
                    "👁️ Show Elements",
                    ["Labels", "Confidence", "Keypoints"],
                    default=["Labels", "Confidence"],
                    help="Choose what to display on detected faces"
                )

            st.markdown("---")

            # Detect faces
            with st.spinner(f"{get_t('processing')}"):
                faces = detect_faces(image, min_confidence=confidence_threshold)

            if len(faces) == 0:
                st.warning(f"⚠️ {get_t('fd_no_faces')}")
                st.info(f"""
                **💡 Tips for better detection:**
                - Lower the confidence threshold (try 0.7-0.8)
                - Ensure faces are clearly visible
                - Check if image is well-lit
                - Make sure faces are not too small
                """)
                
                # Still show original image
                st.markdown("### 📷 Original Image")
                st.image(image, use_column_width=True)
                return

            # ==================== RESULTS DISPLAY ====================
            
            # Draw rectangles with custom settings
            show_labels = "Labels" in show_options
            show_conf = "Confidence" in show_options
            show_keypoints = "Keypoints" in show_options
            
            annotated = draw_face_rectangles(
                image, faces, 
                labels=show_labels,
                color=border_color,
                thickness=border_thickness
            )

            # Display images side by side
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("### 📷 Original Image")
                st.image(image, use_column_width=True)

            with col2:
                st.markdown(f"### 🎯 Detected: {len(faces)} Face(s)")
                st.image(annotated, use_column_width=True)

            # ==================== STATISTICS ====================
            st.markdown("---")
            st.markdown("## 📊 Detection Statistics")

            stats = get_face_statistics(faces)

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.markdown(f"""
                <div style='background: {Config.get_color('primary', 0.1)}; 
                            padding: 1rem; border-radius: 8px; text-align: center;'>
                    <div style='font-size: 2.5rem; font-weight: bold; 
                                color: {Config.COLORS['primary']};'>
                        {stats['count']}
                    </div>
                    <div style='font-size: 0.9rem; color: gray;'>
                        {get_t('fd_count')}
                    </div>
                </div>
                """, unsafe_allow_html=True)

            with col2:
                st.markdown(f"""
                <div style='background: {Config.get_color('success', 0.1)}; 
                            padding: 1rem; border-radius: 8px; text-align: center;'>
                    <div style='font-size: 2.5rem; font-weight: bold; 
                                color: {Config.COLORS['success']};'>
                        {stats['avg_confidence']:.2f}
                    </div>
                    <div style='font-size: 0.9rem; color: gray;'>
                        Avg Confidence
                    </div>
                </div>
                """, unsafe_allow_html=True)

            with col3:
                st.markdown(f"""
                <div style='background: {Config.get_color('info', 0.1)}; 
                            padding: 1rem; border-radius: 8px; text-align: center;'>
                    <div style='font-size: 2.5rem; font-weight: bold; 
                                color: {Config.COLORS['info']};'>
                        {stats['min_confidence']:.2f}
                    </div>
                    <div style='font-size: 0.9rem; color: gray;'>
                        Min Confidence
                    </div>
                </div>
                """, unsafe_allow_html=True)

            with col4:
                st.markdown(f"""
                <div style='background: {Config.get_color('warning', 0.1)}; 
                            padding: 1rem; border-radius: 8px; text-align: center;'>
                    <div style='font-size: 2.5rem; font-weight: bold; 
                                color: {Config.COLORS['warning']};'>
                        {stats['max_confidence']:.2f}
                    </div>
                    <div style='font-size: 0.9rem; color: gray;'>
                        Max Confidence
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # ==================== INDIVIDUAL FACES ====================
            st.markdown("---")
            st.markdown("## 👥 Individual Face Details")

            cols_per_row = 3
            for i in range(0, len(faces), cols_per_row):
                cols = st.columns(cols_per_row)
                for j, col in enumerate(cols):
                    if i + j < len(faces):
                        with col:
                            render_face_info(faces[i + j], i + j)

            # ==================== DOWNLOAD SECTION ====================
            st.markdown("---")
            st.markdown("## 💾 Download Results")

            col1, col2, col3 = st.columns(3)
            
            with col1:
                # Download annotated image
                img_bytes = convert_to_bytes(annotated, format='PNG')
                st.download_button(
                    f"📥 Download Annotated (PNG)",
                    data=img_bytes,
                    file_name=f"detected_{uploaded_file.name.split('.')[0]}.png",
                    mime="image/png",
                    use_container_width=True,
                    type="primary"
                )
            
            with col2:
                # Download as JPEG (smaller size)
                img_bytes_jpg = convert_to_bytes(annotated, format='JPEG')
                st.download_button(
                    f"📥 Download Annotated (JPG)",
                    data=img_bytes_jpg,
                    file_name=f"detected_{uploaded_file.name.split('.')[0]}.jpg",
                    mime="image/jpeg",
                    use_container_width=True
                )
            
            with col3:
                # Export statistics as text
                stats_text = f"""Face Detection Results
{'='*50}
Image: {uploaded_file.name}
Detected Faces: {stats['count']}
Average Confidence: {stats['avg_confidence']:.3f}
Min Confidence: {stats['min_confidence']:.3f}
Max Confidence: {stats['max_confidence']:.3f}
{'='*50}

Individual Faces:
"""
                for i, face in enumerate(faces):
                    x, y, w, h = face['box']
                    stats_text += f"\nFace {i+1}:\n"
                    stats_text += f"  Position: ({x}, {y})\n"
                    stats_text += f"  Size: {w}x{h}\n"
                    stats_text += f"  Confidence: {face['confidence']:.3f}\n"
                
                st.download_button(
                    "📄 Export Statistics (TXT)",
                    data=stats_text,
                    file_name=f"stats_{uploaded_file.name.split('.')[0]}.txt",
                    mime="text/plain",
                    use_container_width=True
                )

    else:
        # ==================== MULTIPLE IMAGES MODE ====================
        
        uploaded_files = st.file_uploader(
            f"📤 Upload Multiple Images",
            type=Config.SUPPORTED_FORMATS,
            accept_multiple_files=True,
            key="multi_upload",
            help=f"Upload up to {Config.MAX_BATCH_SIZE} images"
        )

        if uploaded_files:
            if len(uploaded_files) > Config.MAX_BATCH_SIZE:
                st.error(f"⚠️ Maximum {Config.MAX_BATCH_SIZE} images allowed. You uploaded {len(uploaded_files)}.")
                return

            st.success(f"✅ Uploaded {len(uploaded_files)} images")

            # Detection controls for batch
            st.markdown("### ⚙️ Batch Detection Controls")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                batch_confidence = st.slider(
                    "🎯 Confidence Threshold",
                    min_value=0.5,
                    max_value=1.0,
                    value=0.9,
                    step=0.05
                )
            
            with col2:
                batch_thickness = st.slider(
                    "✏️ Border Thickness",
                    min_value=1,
                    max_value=10,
                    value=2,
                    step=1
                )
            
            with col3:
                grid_cols = st.slider(
                    "📐 Grid Columns",
                    min_value=2,
                    max_value=4,
                    value=3,
                    step=1
                )

            st.markdown("---")

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

            # Detect faces in all images
            with st.spinner("Detecting faces..."):
                all_faces = batch_detect_faces(images, min_confidence=batch_confidence, show_progress=True)

            # Summary statistics
            total_faces = sum(len(faces) for faces in all_faces)
            images_with_faces = sum(1 for faces in all_faces if len(faces) > 0)

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

            for i in range(0, len(images), grid_cols):
                cols = st.columns(grid_cols)

                for j, col in enumerate(cols):
                    if i + j < len(images):
                        with col:
                            img_idx = i + j
                            image = images[img_idx]
                            faces = all_faces[img_idx]

                            if len(faces) > 0:
                                annotated = draw_face_rectangles(
                                    image, faces, 
                                    labels=True,
                                    thickness=batch_thickness
                                )
                                st.image(annotated, use_column_width=True)
                                st.success(f"✅ {len(faces)} face(s)")
                            else:
                                st.image(image, use_column_width=True)
                                st.warning("⚠️ No faces")
                            
                            st.caption(f"📁 {uploaded_files[img_idx].name}")

            # All detected faces grid
            if total_faces > 0:
                st.markdown("---")
                st.markdown("## 👥 All Detected Faces")

                with st.spinner("Creating face grid..."):
                    face_grid = create_face_grid(images, all_faces, grid_cols=6)

                if face_grid is not None:
                    st.image(face_grid, use_column_width=True)
                    st.caption(f"Grid of {total_faces} detected faces")

    # ==================== TIPS AND TECHNICAL INFO ====================
    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        with st.expander("💡 Tips for Best Detection"):
            st.markdown(f"""
            ### For Better Face Detection:

            **✅ Good Images:**
            - Clear, well-lit photos
            - Face clearly visible
            - Minimal blur or motion
            - Frontal or slight angle view
            - High resolution (min 640×480)

            **❌ Avoid:**
            - Very dark or overexposed images
            - Extreme angles (profile view)
            - Heavy blur or low resolution
            - Faces smaller than 20×20 pixels
            - Multiple overlapping faces

            **⚙️ Recommended Settings:**
            - Confidence: 0.85-0.95 for accuracy
            - Border Thickness: 2-3 for visibility
            - Try different thresholds if faces missed
            """)

    with col2:
        with st.expander("🔧 Technical Information"):
            st.markdown(f"""
            ### Detection Algorithm

            **MTCNN (Multi-task Cascaded CNN)**
            - 3-stage cascade architecture
            - Detects faces + landmarks
            - High accuracy on various sizes/angles
            - PyTorch implementation

            **Current Settings:**
            - Confidence threshold: Adjustable
            - Border thickness: Adjustable  
            - Color: Customizable
            - Min face size: 20 pixels (default)

            **Output Information:**
            - Bounding box coordinates
            - Confidence scores (0-1)
            - Facial landmarks (optional)
            - Detection statistics

            **Performance:**
            - Single image: <1 second
            - Batch processing: ~0.5s per image
            - Supports up to {Config.MAX_BATCH_SIZE} images
            """)


if __name__ == "__main__":
    main()
