"""
FaceMatch Pro - Model Utilities
Model loading, inference, and embedding extraction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
import numpy as np
import streamlit as st
from pathlib import Path
import time

from huggingface_hub import hf_hub_download

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import Config


class SEBlock(nn.Module):
    """Squeeze-and-Excitation block"""

    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.squeeze(x).view(b, c)
        y = self.excitation(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class ImprovedEncoder(nn.Module):
    """Face Recognition Encoder Model"""

    def __init__(self, embedding_dim=512, dropout=0.3, use_attention=True):
        super(ImprovedEncoder, self).__init__()

        # Backbone: ResNet50
        resnet = models.resnet50(pretrained=False)
        self.features = nn.Sequential(*list(resnet.children())[:-2])

        feat_dim = 2048
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # Attention
        self.use_attention = use_attention
        if use_attention:
            self.attention = SEBlock(feat_dim)

        # Embedding head
        self.embedding = nn.Sequential(
            nn.Linear(feat_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.PReLU(),
            nn.Dropout(dropout),
            nn.Linear(1024, embedding_dim),
            nn.BatchNorm1d(embedding_dim)
        )

        self.embedding_dim = embedding_dim

    def forward(self, x, return_attention=False):
        # Extract features
        features = self.features(x)

        # Attention
        attention_map = None
        if self.use_attention:
            if return_attention:
                b, c, h, w = features.size()
                y = self.attention.squeeze(features).view(b, c)
                attention_weights = self.attention.excitation(y)
                attention_map = attention_weights.view(b, c, 1, 1).expand_as(features)
            features = self.attention(features)

        # Global pooling
        features = self.global_pool(features)
        features = features.view(features.size(0), -1)

        # Embedding
        embedding = self.embedding(features)
        embedding = F.normalize(embedding, p=2, dim=1)

        if return_attention:
            return embedding, attention_map
        return embedding


@st.cache_resource
def load_model(model_path=None):
    """
    Load trained model
    Cached to avoid reloading
    """
    if model_path is None:
        try:
            model_path = hf_hub_download(
                repo_id="Mo-Abdalkader/facematch-pro",
                filename="FaceSimilarity-v0.1.pth"
            )
        except Exception as e:
            st.error(f"Error downloading model from Hugging Face: {str(e)}")
            return None, None

    try:
        # Get device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load checkpoint - FIXED: Handle both old and new checkpoint formats
        checkpoint = torch.load(
            model_path,
            map_location=device,
            weights_only=False
        )

        # Create model with config values
        model = ImprovedEncoder(
            embedding_dim=Config.EMBEDDING_DIM,
            dropout=Config.DROPOUT_RATE,
            use_attention=Config.USE_ATTENTION
        )

        # Load weights - handle different checkpoint formats
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        elif 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            # Checkpoint IS the state dict
            model.load_state_dict(checkpoint)

        model.to(device)
        model.eval()

        st.success(f"✅ Model loaded successfully on {device}")
        return model, device

    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.error(f"Please ensure the model file exists at: {str(model_path)}")
        
        # Show helpful debug info
        with st.expander("🔧 Troubleshooting"):
            st.markdown(f"""
            **Error Details:** {str(e)}
            
            **Common Solutions:**
            1. Check if model file exists
            2. Ensure you have internet connection for Hugging Face download
            3. Try clearing Streamlit cache: `streamlit cache clear`
            4. Verify model architecture matches checkpoint
            
            **Model Configuration:**
            - Embedding Dim: {Config.EMBEDDING_DIM}
            - Dropout: {Config.DROPOUT_RATE}
            - Use Attention: {Config.USE_ATTENTION}
            """)
        
        return None, None


def get_transform(augment=False):
    """Get image transformation pipeline"""
    if augment:
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((Config.IMAGE_SIZE, Config.IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=Config.NORMALIZE_MEAN, std=Config.NORMALIZE_STD)
        ])
    else:
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((Config.IMAGE_SIZE, Config.IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=Config.NORMALIZE_MEAN, std=Config.NORMALIZE_STD)
        ])


def extract_embedding(model, image, device, use_tta=False, return_attention=False):
    """
    Extract face embedding from image

    Args:
        model: Face recognition model
        image: numpy array (H, W, 3) in RGB
        device: torch device
        use_tta: Use test-time augmentation
        return_attention: Return attention map for visualization

    Returns:
        embedding: 512-dim normalized vector
        attention_map (optional): Attention weights
    """
    transform = get_transform()

    try:
        with torch.no_grad():
            if use_tta and Config.USE_TTA:
                embeddings = []

                # Original
                img_tensor = transform(image).unsqueeze(0).to(device)
                embeddings.append(model(img_tensor))

                # Horizontal flip
                img_flipped = np.fliplr(image).copy()
                img_tensor = transform(img_flipped).unsqueeze(0).to(device)
                embeddings.append(model(img_tensor))

                # Average embeddings
                embedding = torch.stack(embeddings).mean(dim=0)
                embedding = F.normalize(embedding, p=2, dim=1)

                attention_map = None

            else:
                img_tensor = transform(image).unsqueeze(0).to(device)

                if return_attention:
                    embedding, attention_map = model(img_tensor, return_attention=True)
                else:
                    embedding = model(img_tensor)
                    attention_map = None

        if return_attention:
            return embedding.cpu().numpy(), attention_map
        return embedding.cpu().numpy()

    except Exception as e:
        st.error(f"Error extracting embedding: {str(e)}")
        return None


def compute_similarity(embedding1, embedding2):
    """
    Compute cosine similarity between two embeddings

    Args:
        embedding1: numpy array (1, 512)
        embedding2: numpy array (1, 512)

    Returns:
        similarity: float in [0, 1]
    """
    try:
        emb1 = embedding1 / (np.linalg.norm(embedding1) + 1e-8)
        emb2 = embedding2 / (np.linalg.norm(embedding2) + 1e-8)

        similarity = np.dot(emb1.flatten(), emb2.flatten())
        similarity = np.clip(similarity, 0, 1)

        return float(similarity)

    except Exception as e:
        st.error(f"Error computing similarity: {str(e)}")
        return 0.0


def is_match(similarity, threshold=None):
    """Determine if faces match based on threshold"""
    if threshold is None:
        threshold = Config.DEFAULT_THRESHOLD
    return similarity >= threshold


def batch_extract_embeddings(model, images, device, use_tta=False, show_progress=True):
    """Extract embeddings from multiple images"""
    embeddings = []

    if show_progress:
        progress_bar = st.progress(0)
        status_text = st.empty()

    for i, image in enumerate(images):
        embedding = extract_embedding(model, image, device, use_tta)
        if embedding is not None:
            embeddings.append(embedding)

        if show_progress:
            progress = (i + 1) / len(images)
            progress_bar.progress(progress)
            status_text.text(f"Processing: {i + 1}/{len(images)}")

    if show_progress:
        progress_bar.empty()
        status_text.empty()

    if len(embeddings) == 0:
        return None

    return np.vstack(embeddings)


def get_model_info():
    """Get model information for display"""
    return {
        'architecture': 'ResNet50 + Squeeze-and-Excitation',
        'embedding_dim': Config.EMBEDDING_DIM,
        'accuracy': f"{Config.MODEL_METRICS['accuracy'] * 100:.1f}%",
        'precision': f"{Config.MODEL_METRICS['precision'] * 100:.1f}%",
        'recall': f"{Config.MODEL_METRICS['recall'] * 100:.1f}%",
        'f1_score': f"{Config.MODEL_METRICS['f1_score'] * 100:.1f}%",
        'threshold': Config.DEFAULT_THRESHOLD,
        'version': Config.VERSION
    }


def measure_inference_time(model, image, device, num_runs=10):
    """Measure average inference time"""
    transform = get_transform()
    img_tensor = transform(image).unsqueeze(0).to(device)

    times = []

    with torch.no_grad():
        # Warmup
        for _ in range(3):
            _ = model(img_tensor)

        # Measure
        for _ in range(num_runs):
            start = time.time()
            _ = model(img_tensor)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            end = time.time()
            times.append(end - start)

    return np.mean(times)
