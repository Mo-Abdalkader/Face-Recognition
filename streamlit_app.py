"""
AI Face Recognition System - Production Streamlit App
Clean implementation with no dependency conflicts
"""

import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import io
from datetime import datetime
from pathlib import Path
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# ==================================================================================
# PAGE CONFIG
# ==================================================================================
st.set_page_config(
    page_title="AI Face Recognition",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================================================================================
# MODEL ARCHITECTURE (Embedded directly - no external dependencies)
# ==================================================================================

class HybridFaceEncoder(nn.Module):
    """Hybrid GoogleNet + ResNet18 Face Encoder"""
    
    def __init__(self, embedding_dim=512, dropout=0.3):
        super(HybridFaceEncoder, self).__init__()
        
        # GoogleNet features
        googlenet = models.googlenet(pretrained=False)
        self.googlenet_features = nn.Sequential(*list(googlenet.children())[:-1])
        
        # ResNet18 features
        resnet18 = models.resnet18(pretrained=False)
        self.resnet_features = nn.Sequential(*list(resnet18.children())[:-1])
        
        # Fusion layer: 1024 (GoogleNet) + 512 (ResNet) = 1536
        self.fusion = nn.Sequential(
            nn.Linear(1536, 1024),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(1024, embedding_dim)
        )
        
        self.embedding_dim = embedding_dim
    
    def forward(self, x):
        # GoogleNet features
        googlenet_out = self.googlenet_features(x)
        googlenet_out = googlenet_out.view(googlenet_out.size(0), -1)
        
        # ResNet features
        resnet_out = self.resnet_features(x)
        resnet_out = resnet_out.view(resnet_out.size(0), -1)
        
        # Concatenate and fuse
        combined = torch.cat([googlenet_out, resnet_out], dim=1)
        embedding = self.fusion(combined)
        
        # L2 normalize
        embedding = F.normalize(embedding, p=2, dim=1)
        
        return embedding


# ==================================================================================
# MTCNN FACE DETECTION (Simplified - using facenet_pytorch)
# ==================================================================================

@st.cache_resource
def load_face_detector():
    """Load MTCNN face detector"""
    try:
        from facenet_pytorch import MTCNN
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        mtcnn = MTCNN(
            keep_all=True,
            device=device,
            min_face_size=40,
            thresholds=[0.6, 0.7, 0.7]
        )
        return mtcnn
    except Exception as e:
        st.error(f"Failed to load face detector: {e}")
        return None


# ==================================================================================
# MODEL LOADING
# ==================================================================================

@st.cache_resource
def load_model():
    """Load the trained face recognition model"""
    model_path = Path("files/model.pth")
    
    if not model_path.exists():
        st.error(f"❌ Model not found at {model_path}")
        st.info("Please ensure 'files/model.pth' exists in your repository")
        st.stop()
    
    try:
        with st.spinner("🚀 Loading AI model..."):
            # Initialize model
            model = HybridFaceEncoder(embedding_dim=512, dropout=0.3)
            
            # Load checkpoint
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            checkpoint = torch.load(model_path, map_location=device)
            
            # Handle different checkpoint formats
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            
            model.to(device)
            model.eval()
            torch.set_grad_enabled(False)
            
            return model, device
    
    except Exception as e:
        st.error(f"❌ Failed to load model: {e}")
        st.stop()


@st.cache_resource
def get_image_transform():
    """Get image preprocessing transform"""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


# ==================================================================================
# FACE DETECTION & EMBEDDING GENERATION
# ==================================================================================

def detect_faces(image, mtcnn):
    """Detect all faces in image"""
    try:
        img_array = np.array(image)
        boxes, probs = mtcnn.detect(img_array)
        
        if boxes is None:
            return []
        
        faces_info = []
        for box, prob in zip(boxes, probs):
            if prob > 0.9:  # Confidence threshold
                x1, y1, x2, y2 = [int(coord) for coord in box]
                x1, y1 = max(0, x1), max(0, y1)
                x2 = min(image.width, x2)
                y2 = min(image.height, y2)
                
                face_crop = image.crop((x1, y1, x2, y2))
                
                faces_info.append({
                    'box': [x1, y1, x2, y2],
                    'confidence': float(prob),
                    'crop': face_crop
                })
        
        return faces_info
    
    except Exception as e:
        st.error(f"Face detection error: {e}")
        return []


def generate_embedding(face_image, model, transform, device):
    """Generate embedding for a face image"""
    try:
        # Preprocess
        img_tensor = transform(face_image).unsqueeze(0).to(device)
        
        # Generate embedding
        with torch.no_grad():
            embedding = model(img_tensor)
        
        return embedding.cpu().numpy().flatten()
    
    except Exception as e:
        st.error(f"Embedding generation error: {e}")
        return None


def calculate_similarity(emb1, emb2):
    """Calculate cosine similarity"""
    return float(np.dot(emb1, emb2))


# ==================================================================================
# VISUALIZATION FUNCTIONS
# ==================================================================================

def draw_faces_on_image(image, faces_info):
    """Draw bounding boxes on image"""
    img_draw = image.copy()
    draw = ImageDraw.Draw(img_draw)
    
    for idx, face_info in enumerate(faces_info):
        box = face_info['box']
        confidence = face_info['confidence']
        
        color = (0, 255, 0) if confidence > 0.95 else (255, 165, 0)
        draw.rectangle(box, outline=color, width=3)
        
        label = f"Face {idx+1} ({confidence:.1%})"
        draw.text((box[0], box[1]-20), label, fill=color)
    
    return img_draw


def create_comparison_image(img1, img2, faces1, faces2, similarity, threshold):
    """Create side-by-side comparison"""
    img1_ann = draw_faces_on_image(img1, faces1)
    img2_ann = draw_faces_on_image(img2, faces2)
    
    # Resize
    max_h = 600
    img1_ann.thumbnail((1000, max_h), Image.Resampling.LANCZOS)
    img2_ann.thumbnail((1000, max_h), Image.Resampling.LANCZOS)
    
    # Combine
    total_w = img1_ann.width + img2_ann.width + 60
    max_height = max(img1_ann.height, img2_ann.height)
    combined = Image.new('RGB', (total_w, max_height + 100), 'white')
    
    combined.paste(img1_ann, (20, 50))
    combined.paste(img2_ann, (img1_ann.width + 40, 50))
    
    # Add text
    draw = ImageDraw.Draw(combined)
    match_status = "✅ MATCH" if similarity >= threshold else "❌ NO MATCH"
    color = (0, 128, 0) if similarity >= threshold else (255, 0, 0)
    text = f"Similarity: {similarity:.1%} | Threshold: {threshold:.1%} | {match_status}"
    draw.text((50, 10), text, fill=color)
    
    return combined


def create_heatmap(similarities, labels):
    """Create similarity heatmap"""
    fig = go.Figure(data=go.Heatmap(
        z=similarities,
        x=labels,
        y=labels,
        colorscale='RdYlGn',
        zmin=0, zmax=1,
        text=np.round(similarities, 3),
        texttemplate='%{text}',
        colorbar=dict(title="Similarity")
    ))
    fig.update_layout(title="Face Similarity Heatmap", height=500)
    return fig


def create_embedding_plot(embeddings, labels):
    """Create embedding visualization"""
    if len(embeddings) < 2:
        return None
    
    embeddings_array = np.array(embeddings)
    
    if len(embeddings) <= 3:
        reducer = PCA(n_components=2)
    else:
        reducer = TSNE(n_components=2, random_state=42, perplexity=min(5, len(embeddings)-1))
    
    reduced = reducer.fit_transform(embeddings_array)
    
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
    
    fig.update_layout(title="Embedding Space", height=500)
    return fig


# ==================================================================================
# APP MODES
# ==================================================================================

def mode_two_image_comparison(model, mtcnn, transform, device, threshold):
    """Two-image comparison mode"""
    st.header("🔍 Two-Image Face Comparison")
    
    col1, col2 = st.columns(2)
    
    with col1:
        img1_file = st.file_uploader("Upload Image 1", type=['jpg', 'jpeg', 'png'], key='img1')
    with col2:
        img2_file = st.file_uploader("Upload Image 2", type=['jpg', 'jpeg', 'png'], key='img2')
    
    if img1_file and img2_file:
        img1 = Image.open(img1_file).convert('RGB')
        img2 = Image.open(img2_file).convert('RGB')
        
        col1, col2 = st.columns(2)
        with col1:
            st.image(img1, use_container_width=True)
        with col2:
            st.image(img2, use_container_width=True)
        
        if st.button("🚀 Compare Faces", type="primary", use_container_width=True):
            with st.spinner("Analyzing..."):
                # Detect faces
                faces1 = detect_faces(img1, mtcnn)
                faces2 = detect_faces(img2, mtcnn)
                
                if not faces1 or not faces2:
                    st.error("❌ No faces detected in one or both images!")
                    return
                
                # Generate embeddings
                emb1 = generate_embedding(faces1[0]['crop'], model, transform, device)
                emb2 = generate_embedding(faces2[0]['crop'], model, transform, device)
                
                if emb1 is None or emb2 is None:
                    st.error("❌ Failed to generate embeddings")
                    return
                
                # Calculate similarity
                similarity = calculate_similarity(emb1, emb2)
                
                # Display results
                st.markdown("---")
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Similarity", f"{similarity:.1%}")
                with col2:
                    st.metric("Threshold", f"{threshold:.1%}")
                with col3:
                    match = "✅ Match" if similarity >= threshold else "❌ No Match"
                    st.metric("Result", match)
                with col4:
                    confidence = abs(similarity - threshold) / threshold * 100
                    st.metric("Confidence", f"{min(confidence, 100):.0f}%")
                
                # Visual comparison
                st.subheader("📊 Visual Comparison")
                comparison = create_comparison_image(img1, img2, faces1, faces2, similarity, threshold)
                st.image(comparison, use_container_width=True)
                
                # Download
                buf = io.BytesIO()
                comparison.save(buf, format='PNG')
                st.download_button(
                    "⬇️ Download Comparison",
                    data=buf.getvalue(),
                    file_name=f"comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                    mime="image/png"
                )


def mode_multi_face_analysis(model, mtcnn, transform, device, threshold):
    """Multi-face analysis mode"""
    st.header("📊 Multi-Face Detection & Analysis")
    
    uploaded_file = st.file_uploader("Upload image with multiple faces", type=['jpg', 'jpeg', 'png'])
    
    if uploaded_file:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, use_container_width=True)
        
        if st.button("🔍 Analyze Faces", type="primary", use_container_width=True):
            with st.spinner("Detecting faces..."):
                faces = detect_faces(image, mtcnn)
                
                if not faces:
                    st.warning("⚠️ No faces detected!")
                    return
                
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
                        st.image(face['crop'], caption=f"Face {i+1} ({face['confidence']:.1%})")
                
                # Generate embeddings for visualization
                if len(faces) >= 2:
                    st.subheader("🎨 Embedding Visualization")
                    
                    embeddings = []
                    for face in faces:
                        emb = generate_embedding(face['crop'], model, transform, device)
                        if emb is not None:
                            embeddings.append(emb)
                    
                    if len(embeddings) >= 2:
                        labels = [f"Face {i+1}" for i in range(len(embeddings))]
                        
                        # t-SNE plot
                        fig = create_embedding_plot(embeddings, labels)
                        if fig:
                            st.plotly_chart(fig, use_container_width=True)
                        
                        # Heatmap
                        st.subheader("🔥 Similarity Heatmap")
                        similarities = np.zeros((len(embeddings), len(embeddings)))
                        for i in range(len(embeddings)):
                            for j in range(len(embeddings)):
                                similarities[i][j] = calculate_similarity(embeddings[i], embeddings[j])
                        
                        fig_heat = create_heatmap(similarities, labels)
                        st.plotly_chart(fig_heat, use_container_width=True)


def mode_batch_comparison(model, mtcnn, transform, device, threshold):
    """Batch comparison mode"""
    st.header("🔄 Batch Face Comparison")
    
    st.info("💡 Upload one reference image and multiple comparison images")
    
    ref_file = st.file_uploader("Reference Image", type=['jpg', 'jpeg', 'png'], key='ref')
    comp_files = st.file_uploader("Comparison Images", type=['jpg', 'jpeg', 'png'], 
                                   accept_multiple_files=True, key='batch')
    
    if ref_file and comp_files:
        st.write(f"📊 Reference: 1 image | Comparisons: {len(comp_files)} images")
        
        if st.button("🚀 Start Batch Processing", type="primary", use_container_width=True):
            with st.spinner("Processing..."):
                # Process reference
                ref_img = Image.open(ref_file).convert('RGB')
                ref_faces = detect_faces(ref_img, mtcnn)
                
                if not ref_faces:
                    st.error("❌ No face in reference image!")
                    return
                
                ref_emb = generate_embedding(ref_faces[0]['crop'], model, transform, device)
                
                # Process comparisons
                results = []
                progress = st.progress(0)
                
                for idx, comp_file in enumerate(comp_files):
                    comp_img = Image.open(comp_file).convert('RGB')
                    comp_faces = detect_faces(comp_img, mtcnn)
                    
                    if comp_faces:
                        comp_emb = generate_embedding(comp_faces[0]['crop'], model, transform, device)
                        if comp_emb is not None:
                            similarity = calculate_similarity(ref_emb, comp_emb)
                            match = "✅ Match" if similarity >= threshold else "❌ No Match"
                            
                            results.append({
                                'Image': comp_file.name,
                                'Similarity': f"{similarity:.4f}",
                                'Percentage': f"{similarity:.1%}",
                                'Match': match,
                                'Similarity_Raw': similarity
                            })
                    
                    progress.progress((idx + 1) / len(comp_files))
                
                # Display results
                if results:
                    results.sort(key=lambda x: x['Similarity_Raw'], reverse=True)
                    
                    st.success("✅ Batch processing complete!")
                    
                    matches = sum(1 for r in results if "✅" in r['Match'])
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Total", len(results))
                    with col2:
                        st.metric("Matches", matches)
                    with col3:
                        st.metric("Non-Matches", len(results) - matches)
                    with col4:
                        avg = np.mean([r['Similarity_Raw'] for r in results])
                        st.metric("Avg Similarity", f"{avg:.1%}")
                    
                    # Table
                    st.subheader("📊 Results Table")
                    df = pd.DataFrame(results).drop('Similarity_Raw', axis=1)
                    st.dataframe(df, use_container_width=True, hide_index=True)
                    
                    # Download
                    csv = df.to_csv(index=False)
                    st.download_button(
                        "⬇️ Download CSV",
                        data=csv,
                        file_name=f"batch_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
                    
                    # Chart
                    fig = px.bar(df, x='Image', y='Percentage', color='Match',
                                color_discrete_map={"✅ Match": "green", "❌ No Match": "red"})
                    st.plotly_chart(fig, use_container_width=True)


# ==================================================================================
# MAIN APP
# ==================================================================================

def main():
    # Custom CSS
    st.markdown("""
        <style>
        .main-header {font-size: 2.5rem; font-weight: bold; text-align: center; margin-bottom: 1rem;}
        .sub-header {font-size: 1.2rem; text-align: center; color: #666; margin-bottom: 2rem;}
        </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown('<div class="main-header">🎯 AI Face Recognition System</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Powered by Deep Learning</div>', unsafe_allow_html=True)
    
    # Load components
    model, device = load_model()
    mtcnn = load_face_detector()
    transform = get_image_transform()
    
    if mtcnn is None:
        st.error("❌ Failed to load face detector. Please check dependencies.")
        st.stop()
    
    st.success(f"✅ System Ready! Running on: {device}")
    
    # Sidebar
    st.sidebar.title("⚙️ Settings")
    
    mode = st.sidebar.radio(
        "Select Mode",
        ["🔍 Two-Image Comparison", 
         "📊 Multi-Face Analysis", 
         "🔄 Batch Comparison"]
    )
    
    threshold = st.sidebar.slider(
        "Similarity Threshold",
        0.0, 1.0, 0.6, 0.01,
        help="Faces with similarity above this are considered matches"
    )
    
    st.sidebar.markdown("---")
    st.sidebar.info(
        "**How it works:**\n"
        "1. Upload face images\n"
        "2. AI detects faces\n"
        "3. Generates 512-dim embeddings\n"
        "4. Compares similarity scores"
    )
    
    # Route to mode
    if mode == "🔍 Two-Image Comparison":
        mode_two_image_comparison(model, mtcnn, transform, device, threshold)
    elif mode == "📊 Multi-Face Analysis":
        mode_multi_face_analysis(model, mtcnn, transform, device, threshold)
    elif mode == "🔄 Batch Comparison":
        mode_batch_comparison(model, mtcnn, transform, device, threshold)


main()
