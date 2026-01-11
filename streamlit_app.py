"""
Face Recognition Streamlit Application
Lightweight deployment without database

FEATURES:
1. Compare Two Faces - Calculate similarity between two images
2. Compare Multiple Faces - Find most similar faces to reference
3. Face Verification - Batch verification with threshold
4. Live Similarity Matrix - Visual comparison grid

Author: Streamlit Deployment Version
"""

import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
import cv2
import numpy as np
from PIL import Image
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO
import base64

# Page config
st.set_page_config(
    page_title="Face Recognition System",
    page_icon="👤",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# MODEL ARCHITECTURE (Must match training)
# =============================================================================

class HybridFaceEncoder(nn.Module):
    """GoogleNet + ResNet-18 hybrid encoder"""
    
    def __init__(self, embedding_dim: int = 512, dropout: float = 0.5):
        super(HybridFaceEncoder, self).__init__()
        
        googlenet = models.googlenet(pretrained=False)
        self.googlenet_features = nn.Sequential(*list(googlenet.children())[:-1])
        
        resnet18 = models.resnet18(pretrained=False)
        self.resnet_features = nn.Sequential(*list(resnet18.children())[:-1])
        
        self.fusion = nn.Sequential(
            nn.Linear(1536, 1024),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(1024, embedding_dim)
        )
        
        self.embedding_dim = embedding_dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        googlenet_out = self.googlenet_features(x).view(x.size(0), -1)
        resnet_out = self.resnet_features(x).view(x.size(0), -1)
        combined = torch.cat([googlenet_out, resnet_out], dim=1)
        embedding = self.fusion(combined)
        return F.normalize(embedding, p=2, dim=1)

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_device():
    """Safe device detection"""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_image_transform():
    """Image preprocessing pipeline"""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

def load_and_preprocess_image(uploaded_file):
    """Load image from uploaded file"""
    try:
        image = Image.open(uploaded_file).convert('RGB')
        return image
    except Exception as e:
        st.error(f"Error loading image: {e}")
        return None

def generate_embedding(model, image, transform, device):
    """Generate embedding for single image"""
    try:
        img_tensor = transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            embedding = model(img_tensor)
        return embedding.cpu().numpy().flatten()
    except Exception as e:
        st.error(f"Error generating embedding: {e}")
        return None

def cosine_similarity(emb1, emb2):
    """Calculate cosine similarity (embeddings are already normalized)"""
    return float(np.dot(emb1, emb2))

def get_similarity_color(similarity):
    """Get color based on similarity score"""
    if similarity >= 0.7:
        return "green"
    elif similarity >= 0.5:
        return "orange"
    else:
        return "red"

def get_similarity_label(similarity):
    """Get label based on similarity score"""
    if similarity >= 0.7:
        return "✅ Very Similar (Match)"
    elif similarity >= 0.5:
        return "⚠️ Possibly Similar"
    else:
        return "❌ Different (No Match)"

# =============================================================================
# MODEL LOADING (CACHED)
# =============================================================================

@st.cache_resource
def load_model():
    """Load trained model (cached for performance)"""
    try:
        device = get_device()
        
        # Try loading inference model first
        try:
            checkpoint = torch.load(
                'inference_model.pth',
                map_location=device,
                weights_only=False
            )
            st.sidebar.success("✓ Loaded inference_model.pth")
        except FileNotFoundError:
            # Fallback to best_model.pth
            checkpoint = torch.load(
                'best_model.pth',
                map_location=device,
                weights_only=False
            )
            st.sidebar.success("✓ Loaded best_model.pth")
        
        # Extract config
        config = checkpoint.get('config', {})
        embedding_dim = config.get('embedding_dim', 512)
        dropout = config.get('dropout_rate', 0.5)
        
        # Initialize model
        model = HybridFaceEncoder(
            embedding_dim=embedding_dim,
            dropout=dropout
        )
        
        # Load weights
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        
        st.sidebar.info(f"🖥️ Device: {device}")
        st.sidebar.info(f"📊 Embedding Dim: {embedding_dim}")
        
        return model, device
        
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        st.error("Please ensure 'inference_model.pth' or 'best_model.pth' is in the app directory")
        st.stop()

# =============================================================================
# FEATURE 1: COMPARE TWO FACES
# =============================================================================

def feature_compare_two():
    st.header("👥 Compare Two Faces")
    st.markdown("Upload two face images to calculate their similarity score")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Image 1")
        img1_file = st.file_uploader("Upload first image", type=['jpg', 'jpeg', 'png'], key="img1")
        
    with col2:
        st.subheader("Image 2")
        img2_file = st.file_uploader("Upload second image", type=['jpg', 'jpeg', 'png'], key="img2")
    
    if img1_file and img2_file:
        # Load images
        img1 = load_and_preprocess_image(img1_file)
        img2 = load_and_preprocess_image(img2_file)
        
        if img1 and img2:
            # Display images
            col1, col2 = st.columns(2)
            with col1:
                st.image(img1, caption="Image 1", use_container_width=True)
            with col2:
                st.image(img2, caption="Image 2", use_container_width=True)
            
            # Generate embeddings
            with st.spinner("Analyzing faces..."):
                model, device = load_model()
                transform = get_image_transform()
                
                emb1 = generate_embedding(model, img1, transform, device)
                emb2 = generate_embedding(model, img2, transform, device)
                
                if emb1 is not None and emb2 is not None:
                    similarity = cosine_similarity(emb1, emb2)
                    
                    # Display results
                    st.markdown("---")
                    st.subheader("📊 Similarity Analysis")
                    
                    # Similarity score with color
                    color = get_similarity_color(similarity)
                    label = get_similarity_label(similarity)
                    
                    col1, col2, col3 = st.columns([1, 2, 1])
                    with col2:
                        st.metric(
                            label="Similarity Score",
                            value=f"{similarity:.4f}",
                            delta=label
                        )
                        
                        # Progress bar
                        st.progress(similarity)
                        
                        # Interpretation
                        st.markdown(f"**Interpretation:** :{color}[{label}]")
                        
                    # Additional stats
                    st.markdown("---")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Match Confidence", f"{similarity * 100:.1f}%")
                    with col2:
                        st.metric("Distance", f"{1 - similarity:.4f}")
                    with col3:
                        threshold = st.slider("Match Threshold", 0.0, 1.0, 0.6, 0.01)
                        match = "✅ MATCH" if similarity >= threshold else "❌ NO MATCH"
                        st.markdown(f"**Result:** {match}")

# =============================================================================
# FEATURE 2: COMPARE MULTIPLE FACES
# =============================================================================

def feature_compare_multiple():
    st.header("🔍 Compare Multiple Faces")
    st.markdown("Upload one reference image and multiple comparison images to find the most similar faces")
    
    # Reference image
    st.subheader("📸 Reference Image")
    ref_file = st.file_uploader("Upload reference face", type=['jpg', 'jpeg', 'png'], key="ref")
    
    if ref_file:
        ref_img = load_and_preprocess_image(ref_file)
        if ref_img:
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.image(ref_img, caption="Reference Face", use_container_width=True)
            
            # Multiple comparison images
            st.markdown("---")
            st.subheader("📂 Comparison Images")
            comp_files = st.file_uploader(
                "Upload faces to compare (multiple files)",
                type=['jpg', 'jpeg', 'png'],
                accept_multiple_files=True,
                key="comp"
            )
            
            if comp_files:
                st.info(f"Uploaded {len(comp_files)} images for comparison")
                
                if st.button("🚀 Analyze All", type="primary"):
                    model, device = load_model()
                    transform = get_image_transform()
                    
                    # Generate reference embedding
                    with st.spinner("Processing reference image..."):
                        ref_emb = generate_embedding(model, ref_img, transform, device)
                    
                    if ref_emb is not None:
                        results = []
                        
                        # Process each comparison image
                        progress_bar = st.progress(0)
                        for idx, comp_file in enumerate(comp_files):
                            comp_img = load_and_preprocess_image(comp_file)
                            if comp_img:
                                comp_emb = generate_embedding(model, comp_img, transform, device)
                                if comp_emb is not None:
                                    similarity = cosine_similarity(ref_emb, comp_emb)
                                    results.append({
                                        'file': comp_file,
                                        'image': comp_img,
                                        'similarity': similarity,
                                        'label': get_similarity_label(similarity)
                                    })
                            progress_bar.progress((idx + 1) / len(comp_files))
                        
                        # Sort by similarity
                        results.sort(key=lambda x: x['similarity'], reverse=True)
                        
                        # Display results
                        st.markdown("---")
                        st.subheader("📊 Ranked Results")
                        
                        # Summary statistics
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Total Compared", len(results))
                        with col2:
                            avg_sim = np.mean([r['similarity'] for r in results])
                            st.metric("Average Similarity", f"{avg_sim:.3f}")
                        with col3:
                            matches = sum(1 for r in results if r['similarity'] >= 0.6)
                            st.metric("Matches (>0.6)", matches)
                        
                        # Results table
                        st.markdown("### 🏆 Top Matches")
                        for rank, result in enumerate(results, 1):
                            with st.expander(f"#{rank} - {result['file'].name} | Similarity: {result['similarity']:.4f}"):
                                col1, col2 = st.columns([1, 2])
                                with col1:
                                    st.image(result['image'], use_container_width=True)
                                with col2:
                                    st.metric("Similarity Score", f"{result['similarity']:.4f}")
                                    st.progress(result['similarity'])
                                    st.markdown(f"**Status:** {result['label']}")
                        
                        # Visualization
                        st.markdown("---")
                        st.subheader("📈 Similarity Distribution")
                        
                        # Create bar chart
                        df = pd.DataFrame({
                            'Image': [f"Image {i+1}" for i in range(len(results))],
                            'Similarity': [r['similarity'] for r in results]
                        })
                        
                        fig = px.bar(
                            df,
                            x='Image',
                            y='Similarity',
                            title='Similarity Scores',
                            color='Similarity',
                            color_continuous_scale='RdYlGn'
                        )
                        fig.add_hline(y=0.6, line_dash="dash", line_color="red", 
                                      annotation_text="Match Threshold")
                        st.plotly_chart(fig, use_container_width=True)

# =============================================================================
# FEATURE 3: BATCH VERIFICATION
# =============================================================================

def feature_batch_verification():
    st.header("✅ Batch Face Verification")
    st.markdown("Upload multiple face pairs to verify if they match")
    
    st.info("💡 Upload pairs: Image1.jpg & Image1_compare.jpg, Image2.jpg & Image2_compare.jpg, etc.")
    
    files = st.file_uploader(
        "Upload all images (pairs should have similar names)",
        type=['jpg', 'jpeg', 'png'],
        accept_multiple_files=True,
        key="batch"
    )
    
    threshold = st.slider("Verification Threshold", 0.0, 1.0, 0.6, 0.01)
    
    if files and len(files) >= 2:
        if st.button("🔄 Verify All Pairs", type="primary"):
            model, device = load_model()
            transform = get_image_transform()
            
            # Generate all embeddings
            embeddings = {}
            progress_bar = st.progress(0)
            
            for idx, file in enumerate(files):
                img = load_and_preprocess_image(file)
                if img:
                    emb = generate_embedding(model, img, transform, device)
                    if emb is not None:
                        embeddings[file.name] = {'embedding': emb, 'image': img}
                progress_bar.progress((idx + 1) / len(files))
            
            # Create similarity matrix
            st.markdown("---")
            st.subheader("📊 Similarity Matrix")
            
            names = list(embeddings.keys())
            n = len(names)
            similarity_matrix = np.zeros((n, n))
            
            for i, name1 in enumerate(names):
                for j, name2 in enumerate(names):
                    if i <= j:
                        sim = cosine_similarity(
                            embeddings[name1]['embedding'],
                            embeddings[name2]['embedding']
                        )
                        similarity_matrix[i, j] = sim
                        similarity_matrix[j, i] = sim
            
            # Heatmap
            fig = go.Figure(data=go.Heatmap(
                z=similarity_matrix,
                x=names,
                y=names,
                colorscale='RdYlGn',
                zmid=0.6,
                text=similarity_matrix,
                texttemplate='%{text:.3f}',
                textfont={"size": 10}
            ))
            fig.update_layout(
                title="Face Similarity Heatmap",
                height=600
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Verification results
            st.markdown("---")
            st.subheader("✅ Verification Results")
            
            verified_pairs = []
            for i in range(n):
                for j in range(i+1, n):
                    sim = similarity_matrix[i, j]
                    if sim >= threshold:
                        verified_pairs.append({
                            'Pair': f"{names[i]} ↔ {names[j]}",
                            'Similarity': f"{sim:.4f}",
                            'Status': '✅ MATCH' if sim >= threshold else '❌ NO MATCH'
                        })
            
            if verified_pairs:
                st.success(f"Found {len(verified_pairs)} matching pairs!")
                st.dataframe(pd.DataFrame(verified_pairs), use_container_width=True)
            else:
                st.warning("No matching pairs found above threshold")

# =============================================================================
# FEATURE 4: EMBEDDING EXPLORER
# =============================================================================

def feature_embedding_explorer():
    st.header("🧠 Embedding Space Explorer")
    st.markdown("Visualize face embeddings in 2D space using dimensionality reduction")
    
    files = st.file_uploader(
        "Upload multiple face images to visualize",
        type=['jpg', 'jpeg', 'png'],
        accept_multiple_files=True,
        key="explorer"
    )
    
    if files and len(files) >= 3:
        if st.button("🎨 Generate Visualization", type="primary"):
            model, device = load_model()
            transform = get_image_transform()
            
            embeddings_list = []
            names = []
            
            progress_bar = st.progress(0)
            for idx, file in enumerate(files):
                img = load_and_preprocess_image(file)
                if img:
                    emb = generate_embedding(model, img, transform, device)
                    if emb is not None:
                        embeddings_list.append(emb)
                        names.append(file.name)
                progress_bar.progress((idx + 1) / len(files))
            
            if len(embeddings_list) >= 3:
                # Simple 2D projection using first 2 principal components
                from sklearn.decomposition import PCA
                
                embeddings_array = np.array(embeddings_list)
                pca = PCA(n_components=2)
                embeddings_2d = pca.fit_transform(embeddings_array)
                
                # Create scatter plot
                df = pd.DataFrame({
                    'x': embeddings_2d[:, 0],
                    'y': embeddings_2d[:, 1],
                    'name': names
                })
                
                fig = px.scatter(
                    df,
                    x='x',
                    y='y',
                    text='name',
                    title='Face Embeddings in 2D Space (PCA)',
                    labels={'x': 'Principal Component 1', 'y': 'Principal Component 2'}
                )
                fig.update_traces(textposition='top center', marker=dict(size=12))
                fig.update_layout(height=600)
                st.plotly_chart(fig, use_container_width=True)
                
                st.info(f"📊 Explained variance: PC1={pca.explained_variance_ratio_[0]:.1%}, PC2={pca.explained_variance_ratio_[1]:.1%}")

# =============================================================================
# MAIN APP
# =============================================================================

def main():
    # Sidebar
    st.sidebar.title("👤 Face Recognition")
    st.sidebar.markdown("---")
    
    # Feature selection
    feature = st.sidebar.radio(
        "Select Feature",
        [
            "👥 Compare Two Faces",
            "🔍 Compare Multiple Faces",
            "✅ Batch Verification",
            "🧠 Embedding Explorer"
        ]
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### ℹ️ About")
    st.sidebar.info(
        "This app uses a hybrid GoogleNet + ResNet-18 "
        "architecture trained with triplet loss for face recognition."
    )
    
    st.sidebar.markdown("### 📊 Model Info")
    try:
        model, device = load_model()
        # Info already displayed in load_model()
    except:
        pass
    
    # Main content
    st.title("🎭 Face Recognition System")
    st.markdown("*Powered by Deep Learning*")
    st.markdown("---")
    
    # Route to features
    if feature == "👥 Compare Two Faces":
        feature_compare_two()
    elif feature == "🔍 Compare Multiple Faces":
        feature_compare_multiple()
    elif feature == "✅ Batch Verification":
        feature_batch_verification()
    elif feature == "🧠 Embedding Explorer":
        feature_embedding_explorer()
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: gray;'>"
        "Face Recognition System | Deployment Version"
        "</div>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
