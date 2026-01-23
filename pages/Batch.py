"""
FaceMatch Pro - Batch Processing Page
Process multiple images and find similar faces
"""

import streamlit as st
import numpy as np
from pathlib import Path
import sys
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).parent.parent))

from config import Config
from translations import get_text
from utils.model_utils import load_model, extract_embedding, batch_extract_embeddings, compute_similarity
from utils.face_detection import detect_faces, batch_detect_faces, crop_face, get_best_face
from utils.image_utils import load_image, validate_image, batch_load_images
from utils.quality_check import batch_quality_analysis, get_quality_statistics

from utils.session_utils import init_session_state
init_session_state()

def get_t(key):
    """Translation helper"""
    return get_text(key, st.session_state.language)


def compute_similarity_matrix(embeddings):
    """
    Compute pairwise similarity matrix

    Args:
        embeddings: numpy array (N, 512)

    Returns:
        similarity_matrix: numpy array (N, N)
    """
    # Normalize embeddings
    embeddings_norm = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)

    # Compute cosine similarity
    similarity_matrix = np.dot(embeddings_norm, embeddings_norm.T)

    return similarity_matrix


def find_duplicates(similarity_matrix, threshold=0.95):
    """
    Find potential duplicate faces

    Args:
        similarity_matrix: numpy array (N, N)
        threshold: similarity threshold for duplicates

    Returns:
        list of duplicate pairs
    """
    duplicates = []
    n = similarity_matrix.shape[0]

    for i in range(n):
        for j in range(i + 1, n):
            if similarity_matrix[i, j] >= threshold:
                duplicates.append((i, j, similarity_matrix[i, j]))

    return duplicates


def cluster_faces(similarity_matrix, threshold=0.75):
    """
    Cluster similar faces using hierarchical clustering

    Args:
        similarity_matrix: numpy array (N, N)
        threshold: distance threshold for clustering

    Returns:
        list of cluster assignments
    """
    # Convert similarity to distance
    distance_matrix = 1 - similarity_matrix

    # Make it symmetric and zero diagonal
    np.fill_diagonal(distance_matrix, 0)

    # Convert to condensed distance matrix
    condensed_dist = squareform(distance_matrix)

    # Hierarchical clustering
    linkage_matrix = linkage(condensed_dist, method='average')

    return linkage_matrix


def plot_dendrogram(linkage_matrix, filenames=None):
    """
    Plot hierarchical clustering dendrogram

    Args:
        linkage_matrix: linkage matrix from clustering
        filenames: optional list of image filenames

    Returns:
        matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    dendrogram(
        linkage_matrix,
        ax=ax,
        labels=filenames,
        leaf_rotation=90,
        leaf_font_size=8
    )

    ax.set_xlabel('Image Index / Filename')
    ax.set_ylabel('Distance (1 - Similarity)')
    ax.set_title('Face Clustering Dendrogram')

    plt.tight_layout()

    return fig


def main():
    """Main batch processing page"""
    st.title(f"📁 {get_t('batch_title')}")
    st.markdown(f"*{get_t('batch_subtitle')}*")

    st.markdown("---")

    # Load model
    with st.spinner(get_t('loading')):
        model, device = load_model()

    if model is None:
        st.error("❌ Failed to load model")
        return

    # Upload files
    st.markdown(f"## 📤 {get_t('batch_upload')}")

    uploaded_files = st.file_uploader(
        "Upload multiple images",
        type=Config.SUPPORTED_FORMATS,
        accept_multiple_files=True,
        key="batch_upload",
        label_visibility="collapsed"
    )

    if uploaded_files:
        # Check batch size
        if len(uploaded_files) > Config.MAX_BATCH_SIZE:
            st.error(f"⚠️ Maximum {Config.MAX_BATCH_SIZE} images allowed. You uploaded {len(uploaded_files)}.")
            return

        st.success(f"✅ {len(uploaded_files)} images uploaded")

        # Process button
        if st.button(f"⚙️ {get_t('batch_process')}", type="primary", use_container_width=True):

            # Load images
            with st.spinner("Loading images..."):
                images = batch_load_images(uploaded_files, show_progress=True)

            if len(images) == 0:
                st.error("No valid images loaded")
                return

            st.success(f"✅ Loaded {len(images)} images")

            # Detect faces
            with st.spinner("Detecting faces..."):
                all_faces = batch_detect_faces(images, show_progress=True)

            # Extract faces
            face_crops = []
            face_indices = []
            filenames = []

            for i, (img, faces) in enumerate(zip(images, all_faces)):
                if len(faces) > 0:
                    best_face = get_best_face(faces)
                    face_crop = crop_face(img, best_face)
                    face_crops.append(face_crop)
                    face_indices.append(i)
                    filenames.append(uploaded_files[i].name)

            if len(face_crops) == 0:
                st.error("No faces detected in any image")
                return

            st.success(f"✅ Detected {len(face_crops)} faces")

            # Extract embeddings
            with st.spinner("Extracting embeddings..."):
                embeddings = batch_extract_embeddings(
                    model, face_crops, device,
                    use_tta=False, show_progress=True
                )

            # Compute similarity matrix
            with st.spinner("Computing similarities..."):
                similarity_matrix = compute_similarity_matrix(embeddings)

            # Display results
            st.markdown("---")
            st.markdown(f"## 📊 {get_t('search_results')}")

            # Summary statistics
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric(get_t('batch_total'), len(images))

            with col2:
                st.metric(get_t('batch_faces_found'), len(face_crops))

            with col3:
                # Find duplicates
                duplicates = find_duplicates(similarity_matrix, threshold=0.95)
                st.metric(get_t('batch_duplicates'), len(duplicates))

            with col4:
                # Average similarity
                avg_sim = np.mean(similarity_matrix[np.triu_indices_from(similarity_matrix, k=1)])
                st.metric("Avg Similarity", f"{avg_sim * 100:.1f}%")

            # Clustering
            st.markdown("---")
            st.markdown(f"## 🌳 {get_t('batch_cluster')}")

            # Clustering threshold
            cluster_threshold = st.slider(
                get_t('batch_threshold'),
                min_value=0.5,
                max_value=1.0,
                value=0.75,
                step=0.05
            )

            # Perform clustering
            linkage_matrix = cluster_faces(similarity_matrix, cluster_threshold)

            # Plot dendrogram
            fig = plot_dendrogram(linkage_matrix, filenames)
            st.pyplot(fig)

            # Show duplicates
            if len(duplicates) > 0:
                st.markdown("---")
                st.markdown(f"## 🔄 {get_t('batch_duplicates')}")

                st.info(f"Found {len(duplicates)} potential duplicate pairs (similarity ≥ 95%)")

                # Display duplicate pairs
                for i, (idx1, idx2, sim) in enumerate(duplicates[:10]):  # Show first 10
                    st.markdown(f"### Duplicate Pair {i + 1}")

                    col1, col2, col3 = st.columns([2, 2, 1])

                    with col1:
                        st.image(face_crops[idx1], caption=filenames[idx1], use_column_width=True)

                    with col2:
                        st.image(face_crops[idx2], caption=filenames[idx2], use_column_width=True)

                    with col3:
                        st.metric("Similarity", f"{sim * 100:.1f}%")

                if len(duplicates) > 10:
                    st.caption(f"... and {len(duplicates) - 10} more pairs")

            # Similarity matrix heatmap
            st.markdown("---")
            st.markdown("## 🔥 Similarity Heatmap")

            import plotly.graph_objects as go

            fig = go.Figure(data=go.Heatmap(
                z=similarity_matrix * 100,
                x=filenames,
                y=filenames,
                colorscale='RdYlGn',
                text=similarity_matrix * 100,
                texttemplate='%{text:.0f}%',
                textfont={"size": 8},
                colorbar=dict(title="Similarity (%)")
            ))

            fig.update_layout(
                title="Face Similarity Matrix",
                xaxis_title="Images",
                yaxis_title="Images",
                height=600
            )

            st.plotly_chart(fig, use_container_width=True)

            # Quality analysis
            st.markdown("---")
            st.markdown("## 📊 Image Quality Analysis")

            with st.spinner("Analyzing quality..."):
                quality_reports = batch_quality_analysis(
                    [images[i] for i in face_indices],
                    [all_faces[i] for i in face_indices]
                )

                quality_stats = get_quality_statistics(quality_reports)

            if quality_stats:
                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric("Avg Quality", f"{quality_stats['avg_quality']:.1f}/100")

                with col2:
                    st.metric("Excellent", quality_stats['excellent_count'])

                with col3:
                    st.metric("Good", quality_stats['good_count'])

                with col4:
                    st.metric("Fair/Poor", quality_stats['fair_count'] + quality_stats['poor_count'])

            # Face grid
            st.markdown("---")
            st.markdown("## 🖼️ All Detected Faces")

            grid_cols = st.slider("Grid Columns", 3, 8, 5)

            for i in range(0, len(face_crops), grid_cols):
                cols = st.columns(grid_cols)

                for j, col in enumerate(cols):
                    if i + j < len(face_crops):
                        with col:
                            st.image(face_crops[i + j], use_column_width=True)
                            st.caption(f"{filenames[i + j]}")

                            # Quality if available
                            if quality_reports[i + j]:
                                quality = quality_reports[i + j]['quality_score']
                                stars = '★' * Config.get_quality_stars(quality)
                                st.caption(f"Quality: {stars}")

            # Export options
            st.markdown("---")
            st.markdown(f"## 📥 {get_t('batch_export')}")

            col1, col2 = st.columns(2)

            with col1:
                # Export similarity matrix
                if st.button("📊 Export Similarity Matrix (CSV)", use_container_width=True):
                    import pandas as pd

                    df = pd.DataFrame(
                        similarity_matrix * 100,
                        columns=filenames,
                        index=filenames
                    )

                    csv = df.to_csv()

                    st.download_button(
                        "Download CSV",
                        data=csv,
                        file_name="similarity_matrix.csv",
                        mime="text/csv",
                        use_container_width=True
                    )

            with col2:
                # Export duplicates
                if len(duplicates) > 0:
                    if st.button("🔄 Export Duplicates (CSV)", use_container_width=True):
                        import pandas as pd

                        df = pd.DataFrame([
                            {
                                'Image 1': filenames[idx1],
                                'Image 2': filenames[idx2],
                                'Similarity': f"{sim * 100:.2f}%"
                            }
                            for idx1, idx2, sim in duplicates
                        ])

                        csv = df.to_csv(index=False)

                        st.download_button(
                            "Download CSV",
                            data=csv,
                            file_name="duplicates.csv",
                            mime="text/csv",
                            use_container_width=True
                        )

    else:
        # Instructions
        st.info("📌 Upload multiple images to process as a batch")

        with st.expander("💡 What is Batch Processing?"):
            st.markdown(f"""
            ### Batch Processing Features:

            **1. Face Detection**
            - Automatically detects faces in all images
            - Extracts best face from each image
            - Handles multiple faces per image

            **2. Similarity Analysis**
            - Compares all faces against each other
            - Creates similarity matrix
            - Visualizes relationships with heatmap

            **3. Duplicate Detection**
            - Finds highly similar faces (≥95% similarity)
            - Useful for finding duplicate photos
            - Lists all duplicate pairs

            **4. Clustering**
            - Groups similar faces together
            - Hierarchical clustering visualization
            - Adjustable similarity threshold

            **5. Quality Analysis**
            - Analyzes each face for quality
            - Provides quality scores
            - Identifies problematic images

            ### Use Cases:
            - **Photo Organization**: Group photos of same person
            - **Duplicate Removal**: Find and remove duplicate photos
            - **Quality Check**: Identify low-quality images
            - **Family Albums**: Organize by family members
            - **Event Photos**: Group attendees

            ### Tips:
            - Upload 10-50 images for best results
            - Images should be reasonably similar in size
            - Clear, well-lit faces work best
            - Processing time increases with image count

            **Maximum: {Config.MAX_BATCH_SIZE} images per batch**
            """)


if __name__ == "__main__":

    main()
