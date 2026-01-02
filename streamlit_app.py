"""
Face Recognition Streamlit Application
Full-featured face comparison and analysis tool
"""

import streamlit as st
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import io
import json
import pandas as pd
from pathlib import Path
import time
from datetime import datetime
# Use cv2 carefully - already imported via other dependencies
try:
    import cv2
except ImportError:
    import sys
    print("Warning: OpenCV not available, using PIL only", file=sys.stderr)
    cv2 = None

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import plotly.graph_objects as go
import plotly.express as px

# Set page config
st.set_page_config(
    page_title="Face Recognition System",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================================================================================
# CRITICAL: Fix for OpenMP library conflict
# ==================================================================================
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '0'  # Disable OpenEXR to reduce dependencies

# Import project modules (adjust paths as needed)
import sys
# Fix path to import from current directory, not parent
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import using absolute path to avoid cv2.config conflict
try:
    from config import Config
except ImportError:
    # If config.py is being shadowed by cv2.config, import directly
    import importlib.util
    config_path = project_root / "config.py"
    spec = importlib.util.spec_from_file_location("project_config", config_path)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)
    Config = config_module.Config

from inference.embedding_generator import FaceEmbeddingGenerator
from data.face_cropper import FaceCropper
from models.hybrid_encoder import HybridFaceEncoder
from data.transforms import DataTransforms


# ==================================================================================
# INITIALIZATION & CACHING
# ==================================================================================

@st.cache_resource
def load_model():
    """Load model once and cache it"""
    with st.spinner("🚀 Loading AI model..."):
        model_path = Path(Config.SAVED_FILES_DIR) / Config.BEST_MODEL_NAME
        
        if not model_path.exists():
            st.error(f"❌ Model not found: {model_path}")
            st.stop()
        
        model = HybridFaceEncoder(
            embedding_dim=Config.EMBEDDING_DIM,
            dropout=Config.DROPOUT_RATE
        )
        
        checkpoint = torch.load(
            model_path,
            map_location=Config.DEVICE,
            weights_only=False
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(Config.DEVICE)
        model.eval()
        
        torch.set_grad_enabled(False)
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
        
        return model


@st.cache_resource
def load_components():
    """Load all system components"""
    model = load_model()
    
    face_cropper = FaceCropper(
        mode=Config.FACE_CROP_MODE,
        margin=Config.FACE_CROP_MARGIN,
        min_confidence=Config.FACE_MIN_CONFIDENCE
    )
    
    transform = DataTransforms.get_val_transforms(Config)
    
    embedding_generator = FaceEmbeddingGenerator(
        model=model,
        transform=transform,
        face_cropper=face_cropper,
        device=Config.DEVICE
    )
    
    return embedding_generator, face_cropper


# ==================================================================================
# UTILITY FUNCTIONS
# ==================================================================================

def calculate_similarity(emb1, emb2):
    """Calculate cosine similarity between two embeddings"""
    emb1 = emb1 / np.linalg.norm(emb1)
    emb2 = emb2 / np.linalg.norm(emb2)
    similarity = np.dot(emb1, emb2)
    return float(similarity)


def draw_faces_on_image(image, faces_info):
    """Draw bounding boxes and labels on image"""
    img_draw = image.copy()
    draw = ImageDraw.Draw(img_draw)
    
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except:
        font = ImageFont.load_default()
    
    for idx, face_info in enumerate(faces_info):
        box = face_info['box']
        confidence = face_info['confidence']
        
        # Draw rectangle
        color = (0, 255, 0) if confidence > 0.95 else (255, 165, 0)
        draw.rectangle(box, outline=color, width=3)
        
        # Draw label
        label = f"Face {idx+1} ({confidence:.2%})"
        draw.text((box[0], box[1]-25), label, fill=color, font=font)
    
    return img_draw


def extract_face_crop(image, box):
    """Extract face crop from image"""
    x1, y1, x2, y2 = box
    return image.crop((x1, y1, x2, y2))


def create_comparison_image(img1, img2, faces1, faces2, similarity, threshold):
    """Create side-by-side comparison image"""
    # Draw faces on images
    img1_annotated = draw_faces_on_image(img1, faces1)
    img2_annotated = draw_faces_on_image(img2, faces2)
    
    # Resize to same height
    max_height = 600
    img1_annotated.thumbnail((1000, max_height), Image.Resampling.LANCZOS)
    img2_annotated.thumbnail((1000, max_height), Image.Resampling.LANCZOS)
    
    # Create combined image
    total_width = img1_annotated.width + img2_annotated.width + 60
    max_h = max(img1_annotated.height, img2_annotated.height)
    
    combined = Image.new('RGB', (total_width, max_h + 100), 'white')
    combined.paste(img1_annotated, (20, 50))
    combined.paste(img2_annotated, (img1_annotated.width + 40, 50))
    
    # Add similarity score
    draw = ImageDraw.Draw(combined)
    try:
        font = ImageFont.truetype("arial.ttf", 30)
    except:
        font = ImageFont.load_default()
    
    match_status = "✅ MATCH" if similarity >= threshold else "❌ NO MATCH"
    color = (0, 128, 0) if similarity >= threshold else (255, 0, 0)
    
    text = f"Similarity: {similarity:.2%} | Threshold: {threshold:.2%} | {match_status}"
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    draw.text(((total_width - text_width) // 2, 10), text, fill=color, font=font)
    
    return combined


def process_image_for_faces(image, face_cropper):
    """Detect all faces in an image"""
    # Convert PIL to numpy array
    img_array = np.array(image)
    
    # Detect faces
    try:
        faces = face_cropper.mtcnn.detect(img_array)
    except Exception as e:
        st.error(f"Face detection error: {e}")
        return []
    
    if faces[0] is None:
        return []
    
    boxes, probs, landmarks = faces
    
    faces_info = []
    for box, prob in zip(boxes, probs):
        faces_info.append({
            'box': [int(x) for x in box],
            'confidence': float(prob),
            'crop': extract_face_crop(image, box)
        })
    
    return faces_info


def create_embedding_visualization(embeddings, labels):
    """Create t-SNE visualization of embeddings"""
    if len(embeddings) < 2:
        return None
    
    embeddings_array = np.array(embeddings)
    
    # Use PCA if few samples, else t-SNE
    if len(embeddings) <= 3:
        reducer = PCA(n_components=2)
    else:
        reducer = TSNE(n_components=2, random_state=42, perplexity=min(5, len(embeddings)-1))
    
    reduced = reducer.fit_transform(embeddings_array)
    
    # Create plotly figure
    fig = go.Figure()
    
    unique_labels = list(set(labels))
    colors = px.colors.qualitative.Set1[:len(unique_labels)]
    
    for i, label in enumerate(unique_labels):
        mask = [l == label for l in labels]
        fig.add_trace(go.Scatter(
            x=reduced[mask, 0],
            y=reduced[mask, 1],
            mode='markers+text',
            name=label,
            text=[f"{label} {j+1}" for j in range(sum(mask))],
            textposition="top center",
            marker=dict(size=15, color=colors[i])
        ))
    
    fig.update_layout(
        title="Embedding Space Visualization",
        xaxis_title="Component 1",
        yaxis_title="Component 2",
        height=500,
        hovermode='closest'
    )
    
    return fig


def create_similarity_heatmap(similarities_matrix, labels):
    """Create similarity heatmap"""
    fig = go.Figure(data=go.Heatmap(
        z=similarities_matrix,
        x=labels,
        y=labels,
        colorscale='RdYlGn',
        zmin=0,
        zmax=1,
        text=np.round(similarities_matrix, 3),
        texttemplate='%{text}',
        textfont={"size": 10},
        colorbar=dict(title="Similarity")
    ))
    
    fig.update_layout(
        title="Face Similarity Heatmap",
        xaxis_title="Faces",
        yaxis_title="Faces",
        height=500
    )
    
    return fig


# ==================================================================================
# STREAMLIT APP
# ==================================================================================

def main():
    # Custom CSS
    st.markdown("""
        <style>
        .main-header {
            font-size: 3rem;
            font-weight: bold;
            text-align: center;
            margin-bottom: 1rem;
        }
        .sub-header {
            font-size: 1.2rem;
            text-align: center;
            color: #666;
            margin-bottom: 2rem;
        }
        .metric-card {
            background-color: #f0f2f6;
            padding: 1rem;
            border-radius: 0.5rem;
            text-align: center;
        }
        .success-box {
            background-color: #d4edda;
            border: 1px solid #c3e6cb;
            padding: 1rem;
            border-radius: 0.5rem;
            margin: 1rem 0;
        }
        .error-box {
            background-color: #f8d7da;
            border: 1px solid #f5c6cb;
            padding: 1rem;
            border-radius: 0.5rem;
            margin: 1rem 0;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown('<div class="main-header">🎯 Face Recognition System</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">AI-Powered Face Comparison & Analysis</div>', unsafe_allow_html=True)
    
    # Load components
    try:
        embedding_generator, face_cropper = load_components()
        st.success("✅ System initialized successfully!")
    except Exception as e:
        st.error(f"❌ Failed to initialize system: {e}")
        st.stop()
    
    # Sidebar
    st.sidebar.title("⚙️ Settings")
    
    # Mode selection
    mode = st.sidebar.radio(
        "Select Mode",
        ["🔍 Two-Image Comparison", 
         "📊 Multi-Face Analysis", 
         "🔄 Batch Comparison",
         "📸 Webcam Comparison"]
    )
    
    # Threshold slider
    threshold = st.sidebar.slider(
        "Similarity Threshold",
        min_value=0.0,
        max_value=1.0,
        value=Config.RECOGNITION_THRESHOLD,
        step=0.01,
        help="Faces with similarity above this threshold are considered matches"
    )
    
    st.sidebar.markdown("---")
    st.sidebar.info(
        "**How it works:**\n"
        "1. Upload face images\n"
        "2. AI detects and analyzes faces\n"
        "3. Compare similarity scores\n"
        "4. View results and visualizations"
    )
    
    # Main content based on mode
    if mode == "🔍 Two-Image Comparison":
        show_two_image_comparison(embedding_generator, face_cropper, threshold)
    
    elif mode == "📊 Multi-Face Analysis":
        show_multi_face_analysis(embedding_generator, face_cropper, threshold)
    
    elif mode == "🔄 Batch Comparison":
        show_batch_comparison(embedding_generator, face_cropper, threshold)
    
    elif mode == "📸 Webcam Comparison":
        show_webcam_comparison(embedding_generator, face_cropper, threshold)


def show_two_image_comparison(embedding_generator, face_cropper, threshold):
    """Two-image comparison mode"""
    st.header("🔍 Two-Image Face Comparison")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Image 1")
        img1_file = st.file_uploader("Upload first image", type=['jpg', 'jpeg', 'png'], key='img1')
    
    with col2:
        st.subheader("Image 2")
        img2_file = st.file_uploader("Upload second image", type=['jpg', 'jpeg', 'png'], key='img2')
    
    if img1_file and img2_file:
        img1 = Image.open(img1_file).convert('RGB')
        img2 = Image.open(img2_file).convert('RGB')
        
        col1, col2 = st.columns(2)
        with col1:
            st.image(img1, caption="Image 1", use_container_width=True)
        with col2:
            st.image(img2, caption="Image 2", use_container_width=True)
        
        if st.button("🚀 Compare Faces", type="primary", use_container_width=True):
            with st.spinner("🔍 Analyzing faces..."):
                try:
                    # Detect faces
                    faces1 = process_image_for_faces(img1, face_cropper)
                    faces2 = process_image_for_faces(img2, face_cropper)
                    
                    if not faces1 or not faces2:
                        st.error("❌ No faces detected in one or both images!")
                        return
                    
                    # Save temporary files
                    temp1 = "temp_img1.jpg"
                    temp2 = "temp_img2.jpg"
                    img1.save(temp1)
                    img2.save(temp2)
                    
                    # Generate embeddings
                    emb1 = embedding_generator.generate_embedding(temp1)
                    emb2 = embedding_generator.generate_embedding(temp2)
                    
                    # Calculate similarity
                    similarity = calculate_similarity(emb1, emb2)
                    
                    # Cleanup
                    Path(temp1).unlink(missing_ok=True)
                    Path(temp2).unlink(missing_ok=True)
                    
                    # Display results
                    st.markdown("---")
                    
                    # Metrics
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Similarity Score", f"{similarity:.2%}")
                    with col2:
                        st.metric("Threshold", f"{threshold:.2%}")
                    with col3:
                        match = "✅ Match" if similarity >= threshold else "❌ No Match"
                        st.metric("Result", match)
                    with col4:
                        confidence = abs(similarity - threshold) / threshold * 100
                        st.metric("Confidence", f"{min(confidence, 100):.1f}%")
                    
                    # Visual comparison
                    st.subheader("📊 Visual Comparison")
                    comparison_img = create_comparison_image(img1, img2, faces1, faces2, similarity, threshold)
                    st.image(comparison_img, use_container_width=True)
                    
                    # Download button
                    buf = io.BytesIO()
                    comparison_img.save(buf, format='PNG')
                    st.download_button(
                        "⬇️ Download Comparison",
                        data=buf.getvalue(),
                        file_name=f"comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                        mime="image/png"
                    )
                    
                    # Face details
                    with st.expander("🔍 Face Detection Details"):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write("**Image 1:**")
                            for i, face in enumerate(faces1):
                                st.write(f"- Face {i+1}: {face['confidence']:.2%} confidence")
                                st.image(face['crop'], width=150)
                        with col2:
                            st.write("**Image 2:**")
                            for i, face in enumerate(faces2):
                                st.write(f"- Face {i+1}: {face['confidence']:.2%} confidence")
                                st.image(face['crop'], width=150)
                    
                except Exception as e:
                    st.error(f"❌ Error: {e}")


def show_multi_face_analysis(embedding_generator, face_cropper, threshold):
    """Multi-face analysis mode"""
    st.header("📊 Multi-Face Detection & Analysis")
    
    uploaded_file = st.file_uploader(
        "Upload an image with multiple faces",
        type=['jpg', 'jpeg', 'png']
    )
    
    if uploaded_file:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption="Original Image", use_container_width=True)
        
        if st.button("🔍 Analyze Faces", type="primary", use_container_width=True):
            with st.spinner("🔍 Detecting faces..."):
                try:
                    faces = process_image_for_faces(image, face_cropper)
                    
                    if not faces:
                        st.warning("⚠️ No faces detected in the image!")
                        return
                    
                    # Display summary
                    st.success(f"✅ Detected {len(faces)} face(s)!")
                    
                    # Annotated image
                    st.subheader("📍 Detected Faces")
                    annotated = draw_faces_on_image(image, faces)
                    st.image(annotated, use_container_width=True)
                    
                    # Face grid
                    st.subheader("🖼️ Extracted Faces")
                    cols = st.columns(min(len(faces), 4))
                    for i, face in enumerate(faces):
                        with cols[i % 4]:
                            st.image(face['crop'], caption=f"Face {i+1} ({face['confidence']:.2%})")
                            
                            # Download button for each face
                            buf = io.BytesIO()
                            face['crop'].save(buf, format='PNG')
                            st.download_button(
                                f"⬇️ Download",
                                data=buf.getvalue(),
                                file_name=f"face_{i+1}.png",
                                mime="image/png",
                                key=f"download_{i}"
                            )
                    
                    # Generate embeddings for visualization
                    if len(faces) >= 2:
                        st.subheader("🎨 Embedding Visualization")
                        
                        with st.spinner("Generating embeddings..."):
                            embeddings = []
                            for i, face in enumerate(faces):
                                temp_path = f"temp_face_{i}.jpg"
                                face['crop'].save(temp_path)
                                emb = embedding_generator.generate_embedding(temp_path)
                                embeddings.append(emb)
                                Path(temp_path).unlink(missing_ok=True)
                            
                            # Create visualization
                            labels = [f"Face {i+1}" for i in range(len(faces))]
                            fig = create_embedding_visualization(embeddings, labels)
                            if fig:
                                st.plotly_chart(fig, use_container_width=True)
                            
                            # Similarity matrix
                            if len(faces) >= 2:
                                st.subheader("🔥 Similarity Heatmap")
                                similarities = np.zeros((len(embeddings), len(embeddings)))
                                for i in range(len(embeddings)):
                                    for j in range(len(embeddings)):
                                        similarities[i][j] = calculate_similarity(embeddings[i], embeddings[j])
                                
                                fig_heatmap = create_similarity_heatmap(similarities, labels)
                                st.plotly_chart(fig_heatmap, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"❌ Error: {e}")


def show_batch_comparison(embedding_generator, face_cropper, threshold):
    """Batch comparison mode"""
    st.header("🔄 Batch Face Comparison")
    
    st.info("💡 Upload one reference image and multiple comparison images")
    
    reference_file = st.file_uploader(
        "Upload Reference Image",
        type=['jpg', 'jpeg', 'png'],
        key='reference'
    )
    
    comparison_files = st.file_uploader(
        "Upload Comparison Images",
        type=['jpg', 'jpeg', 'png'],
        accept_multiple_files=True,
        key='batch'
    )
    
    if reference_file and comparison_files:
        st.write(f"📊 Reference: 1 image | Comparisons: {len(comparison_files)} images")
        
        if st.button("🚀 Start Batch Comparison", type="primary", use_container_width=True):
            with st.spinner("Processing batch..."):
                try:
                    # Process reference
                    ref_img = Image.open(reference_file).convert('RGB')
                    ref_img.save("temp_ref.jpg")
                    ref_emb = embedding_generator.generate_embedding("temp_ref.jpg")
                    Path("temp_ref.jpg").unlink(missing_ok=True)
                    
                    # Process comparisons
                    results = []
                    progress_bar = st.progress(0)
                    
                    for idx, comp_file in enumerate(comparison_files):
                        comp_img = Image.open(comp_file).convert('RGB')
                        comp_img.save("temp_comp.jpg")
                        comp_emb = embedding_generator.generate_embedding("temp_comp.jpg")
                        Path("temp_comp.jpg").unlink(missing_ok=True)
                        
                        similarity = calculate_similarity(ref_emb, comp_emb)
                        match = "✅ Match" if similarity >= threshold else "❌ No Match"
                        
                        results.append({
                            'Image': comp_file.name,
                            'Similarity': f"{similarity:.4f}",
                            'Percentage': f"{similarity:.2%}",
                            'Match': match,
                            'Similarity_Raw': similarity
                        })
                        
                        progress_bar.progress((idx + 1) / len(comparison_files))
                    
                    # Sort by similarity
                    results.sort(key=lambda x: x['Similarity_Raw'], reverse=True)
                    
                    # Display results
                    st.success("✅ Batch processing complete!")
                    
                    # Summary metrics
                    matches = sum(1 for r in results if "✅" in r['Match'])
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Total Images", len(results))
                    with col2:
                        st.metric("Matches", matches)
                    with col3:
                        st.metric("Non-Matches", len(results) - matches)
                    with col4:
                        avg_sim = np.mean([r['Similarity_Raw'] for r in results])
                        st.metric("Avg Similarity", f"{avg_sim:.2%}")
                    
                    # Results table
                    st.subheader("📊 Results Table")
                    df = pd.DataFrame(results).drop('Similarity_Raw', axis=1)
                    st.dataframe(df, use_container_width=True, hide_index=True)
                    
                    # Download results
                    csv = df.to_csv(index=False)
                    st.download_button(
                        "⬇️ Download Results (CSV)",
                        data=csv,
                        file_name=f"batch_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
                    
                    # Visualization
                    st.subheader("📈 Similarity Distribution")
                    fig = px.bar(
                        df,
                        x='Image',
                        y='Percentage',
                        color='Match',
                        title="Similarity Scores",
                        color_discrete_map={"✅ Match": "green", "❌ No Match": "red"}
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"❌ Error: {e}")


def show_webcam_comparison(embedding_generator, face_cropper, threshold):
    """Webcam comparison mode"""
    st.header("📸 Webcam Face Comparison")
    
    st.warning("⚠️ Webcam feature requires local deployment. Upload an image instead for demo.")
    
    reference_file = st.file_uploader(
        "Upload Reference Image",
        type=['jpg', 'jpeg', 'png'],
        key='webcam_ref'
    )
    
    camera_image = st.camera_input("Take a picture")
    
    if reference_file and camera_image:
        if st.button("Compare with Reference", type="primary"):
            with st.spinner("Comparing..."):
                try:
                    ref_img = Image.open(reference_file).convert('RGB')
                    cam_img = Image.open(camera_image).convert('RGB')
                    
                    ref_img.save("temp_ref.jpg")
                    cam_img.save("temp_cam.jpg")
                    
                    ref_emb = embedding_generator.generate_embedding("temp_ref.jpg")
                    cam_emb = embedding_generator.generate_embedding("temp_cam.jpg")
                    
                    Path("temp_ref.jpg").unlink(missing_ok=True)
                    Path("temp_cam.jpg").unlink(missing_ok=True)
                    
                    similarity = calculate_similarity(ref_emb, cam_emb)
                    match = similarity >= threshold
                    
                    if match:
                        st.success(f"✅ MATCH! Similarity: {similarity:.2%}")
                    else:
                        st.error(f"❌ NO MATCH. Similarity: {similarity:.2%}")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.image(ref_img, caption="Reference", use_container_width=True)
                    with col2:
                        st.image(cam_img, caption="Camera", use_container_width=True)
                    
                except Exception as e:
                    st.error(f"❌ Error: {e}")


if __name__ == "__main__":
    main()
