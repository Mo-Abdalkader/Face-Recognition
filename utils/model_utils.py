"""
FaceMatch Pro - Model Utilities
Model loading, inference, and embedding extraction
FIXED: Supports both training checkpoints and lightweight inference models
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


def _load_checkpoint_smart(checkpoint_path, device):
    """
    Smart checkpoint loader that handles multiple formats:
    1. Lightweight inference model (with metadata)
    2. Training checkpoint (with optimizer, etc.)
    3. Direct state_dict
    
    Returns:
        model_weights: state_dict
        metadata: dict with model config (or None)
    """
    try:
        # Load checkpoint
        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=False
        )
        
        # Format 1: Lightweight inference model (RECOMMENDED)
        # Structure: {'model_state_dict': ..., 'metadata': {...}}
        if isinstance(checkpoint, dict) and 'metadata' in checkpoint:
            return checkpoint['model_state_dict'], checkpoint['metadata']
        
        # Format 2: Training checkpoint
        # Structure: {'model_state_dict': ..., 'optimizer_state_dict': ..., 'epoch': ...}
        elif isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            # Extract metadata from checkpoint if available
            metadata = {
                'embedding_dim': checkpoint.get('config', Config).EMBEDDING_DIM if hasattr(checkpoint.get('config', Config), 'EMBEDDING_DIM') else Config.EMBEDDING_DIM,
                'dropout': checkpoint.get('config', Config).DROPOUT_RATE if hasattr(checkpoint.get('config', Config), 'DROPOUT_RATE') else Config.DROPOUT_RATE,
                'use_attention': checkpoint.get('config', Config).USE_ATTENTION if hasattr(checkpoint.get('config', Config), 'USE_ATTENTION') else Config.USE_ATTENTION,
                'optimal_threshold': checkpoint.get('val_metrics', {}).get('optimal_threshold', Config.DEFAULT_THRESHOLD),
                'val_accuracy': checkpoint.get('val_accuracy', -1),
            }
            return checkpoint['model_state_dict'], metadata
        
        # Format 3: Legacy format with 'state_dict' key
        elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            return checkpoint['state_dict'], None
        
        # Format 4: Direct state_dict (no wrapper)
        elif isinstance(checkpoint, dict):
            # Check if it looks like a state_dict (has layer keys)
            first_key = next(iter(checkpoint.keys()))
            if 'features' in first_key or 'embedding' in first_key or 'attention' in first_key:
                return checkpoint, None
        
        raise ValueError(f"Unrecognized checkpoint format. Keys: {checkpoint.keys() if isinstance(checkpoint, dict) else 'Not a dict'}")
    
    except Exception as e:
        raise RuntimeError(f"Failed to load checkpoint: {str(e)}")


@st.cache_resource
def load_model(model_path=None):
    """
    Load trained model - FIXED VERSION
    
    Supports multiple checkpoint formats:
    - Lightweight inference models (recommended)
    - Full training checkpoints
    - Legacy formats
    
    Cached to avoid reloading
    
    Returns:
        model: Loaded PyTorch model
        device: torch.device
    """
    # Download from Hugging Face if no path provided
    if model_path is None:
        try:
            with st.spinner("📥 Downloading model from Hugging Face..."):
                model_path = hf_hub_download(
                    repo_id="Mo-Abdalkader/facematch-pro",
                    filename="FaceSimilarity-v0.1.pth",
                    force_download=False,  # Use cache if available
                    cache_dir=None  # Use default cache
                )
            st.success(f"✅ Model downloaded successfully")
        except Exception as e:
            st.error(f"❌ Error downloading model from Hugging Face: {str(e)}")
            st.info("💡 **Troubleshooting Tips:**\n"
                   "1. Check your internet connection\n"
                   "2. Verify the repository exists: `Mo-Abdalkader/facematch-pro`\n"
                   "3. Ensure the file exists: `FaceSimilarity-v0.1.pth`\n"
                   "4. Try clearing cache: Delete `~/.cache/huggingface/`")
            return None, None
    
    try:
        # Get device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Smart checkpoint loading
        with st.spinner("🔄 Loading model weights..."):
            model_weights, metadata = _load_checkpoint_smart(model_path, device)
        
        # Extract model configuration
        if metadata is not None:
            # Use metadata from checkpoint (BEST - from lightweight inference model)
            embedding_dim = metadata.get('embedding_dim', Config.EMBEDDING_DIM)
            dropout = metadata.get('dropout', Config.DROPOUT_RATE)
            use_attention = metadata.get('use_attention', Config.USE_ATTENTION)
            optimal_threshold = metadata.get('optimal_threshold', Config.DEFAULT_THRESHOLD)
            val_accuracy = metadata.get('val_accuracy', -1)
            
            st.info(f"📊 **Model Info from Checkpoint:**\n"
                   f"- Embedding Dim: {embedding_dim}\n"
                   f"- Optimal Threshold: {optimal_threshold:.4f}\n"
                   + (f"- Validation Accuracy: {val_accuracy*100:.2f}%" if val_accuracy > 0 else ""))
        else:
            # Use Config defaults (fallback)
            embedding_dim = Config.EMBEDDING_DIM
            dropout = Config.DROPOUT_RATE
            use_attention = Config.USE_ATTENTION
            
            st.warning("⚠️ Using default config values (metadata not found in checkpoint)")
        
        # Create model with correct configuration
        model = ImprovedEncoder(
            embedding_dim=embedding_dim,
            dropout=dropout,
            use_attention=use_attention
        )
        
        # Load weights
        model.load_state_dict(model_weights)
        model.to(device)
        model.eval()
        
        # Verify model works with a dummy forward pass
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224).to(device)
            dummy_output = model(dummy_input)
            assert dummy_output.shape == (1, embedding_dim), f"Expected shape (1, {embedding_dim}), got {dummy_output.shape}"
        
        # Success message
        device_name = torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'
        st.success(f"✅ **Model loaded successfully!**\n"
                  f"- Device: {device.type.upper()} ({device_name})\n"
                  f"- Embedding Dim: {embedding_dim}\n"
                  f"- Parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        return model, device

    except FileNotFoundError:
        st.error(f"❌ Model file not found: {model_path}")
        st.info("💡 **Solutions:**\n"
               "1. Check if the file path is correct\n"
               "2. Ensure the model file exists\n"
               "3. Try re-downloading from Hugging Face")
        return None, None
    
    except RuntimeError as e:
        error_msg = str(e)
        st.error(f"❌ Runtime error loading model: {error_msg}")
        
        # Provide specific guidance based on error
        if "size mismatch" in error_msg.lower():
            st.error("🔍 **Size Mismatch Detected**\n\n"
                    "This usually means the checkpoint was trained with different settings.\n\n"
                    "**Possible causes:**\n"
                    "1. Wrong embedding dimension in Config\n"
                    "2. Different model architecture\n"
                    "3. Corrupted checkpoint file\n\n"
                    "**Solution:** Verify Config.EMBEDDING_DIM matches the trained model")
        elif "key" in error_msg.lower():
            st.error("🔍 **Missing Keys Detected**\n\n"
                    "The checkpoint is missing required model weights.\n\n"
                    "**Solution:** Re-export the model using the inference export script")
        
        with st.expander("🔧 Advanced Troubleshooting"):
            st.code(f"""
# Debug Information:
Error: {error_msg}

# Current Config:
- EMBEDDING_DIM: {Config.EMBEDDING_DIM}
- DROPOUT_RATE: {Config.DROPOUT_RATE}
- USE_ATTENTION: {Config.USE_ATTENTION}

# Try this:
1. Check if model was exported correctly
2. Verify checkpoint format
3. Re-export using: export_inference_model()
            """)
        
        return None, None
    
    except Exception as e:
        st.error(f"❌ Unexpected error loading model: {str(e)}")
        
        with st.expander("🔧 Full Error Details"):
            st.code(f"""
Error Type: {type(e).__name__}
Error Message: {str(e)}

Model Path: {model_path}

Config Settings:
- EMBEDDING_DIM: {Config.EMBEDDING_DIM}
- DROPOUT_RATE: {Config.DROPOUT_RATE}
- USE_ATTENTION: {Config.USE_ATTENTION}
- DEFAULT_THRESHOLD: {Config.DEFAULT_THRESHOLD}

Troubleshooting Steps:
1. Verify model file exists and is not corrupted
2. Check Config values match training settings
3. Try clearing Streamlit cache: 'streamlit cache clear'
4. Re-export model using export_inference_model()
5. Check PyTorch and CUDA compatibility
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
