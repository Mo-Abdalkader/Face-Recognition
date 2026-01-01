"""
Hybrid Face Encoder with Automatic Pretrained Model Download
GoogleNet + ResNet-18 fusion network with model caching
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from pathlib import Path
import os


class HybridFaceEncoder(nn.Module):
    """
    Hybrid feature extractor combining GoogleNet and ResNet-18
    Automatically downloads and caches pretrained models
    Output: L2-normalized embedding vector
    """

    def __init__(self, embedding_dim: int = 512, dropout: float = 0.3):
        super(HybridFaceEncoder, self).__init__()

        # Create models cache directory
        self.cache_dir = Path("files/pretrained_models")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        print("Loading pretrained models...")

        # GoogleNet (Inception v1) with auto-download
        print("  - Loading GoogleNet (Inception v1)...")
        googlenet = self._load_googlenet()
        self.googlenet_features = nn.Sequential(*list(googlenet.children())[:-1])
        print("    ✓ GoogleNet loaded (outputs 1024-dim)")

        # ResNet-18 with auto-download
        print("  - Loading ResNet-18...")
        resnet18 = self._load_resnet18()
        self.resnet_features = nn.Sequential(*list(resnet18.children())[:-1])
        print("    ✓ ResNet-18 loaded (outputs 512-dim)")

        # Fusion layer: 1024 (GoogleNet) + 512 (ResNet) = 1536
        self.fusion = nn.Sequential(
            nn.Linear(1536, 1024),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(1024, embedding_dim)
        )

        self.embedding_dim = embedding_dim
        print(f"✓ Hybrid encoder initialized (embedding_dim={embedding_dim})")

    def _load_googlenet(self):
        """Load GoogleNet with automatic download and caching"""
        model_path = self.cache_dir / "googlenet_pretrained.pth"

        try:
            if model_path.exists():
                # Load from cache
                print(f"    Loading cached model from {model_path}")
                googlenet = models.googlenet(pretrained=False)
                googlenet.load_state_dict(torch.load(model_path, map_location='cpu'))
            else:
                # Download and cache
                print("    Downloading GoogleNet pretrained weights...")
                googlenet = models.googlenet(pretrained=True)
                print(f"    Caching model to {model_path}")
                torch.save(googlenet.state_dict(), model_path)

            return googlenet

        except Exception as e:
            print(f"    ⚠ Error with cached model: {e}")
            print("    Downloading fresh GoogleNet weights...")
            googlenet = models.googlenet(pretrained=True)
            try:
                torch.save(googlenet.state_dict(), model_path)
            except:
                pass  # Cache save failed, but model loaded
            return googlenet

    def _load_resnet18(self):
        """Load ResNet-18 with automatic download and caching"""
        model_path = self.cache_dir / "resnet18_pretrained.pth"

        try:
            if model_path.exists():
                # Load from cache
                print(f"    Loading cached model from {model_path}")
                resnet18 = models.resnet18(pretrained=False)
                resnet18.load_state_dict(torch.load(model_path, map_location='cpu'))
            else:
                # Download and cache
                print("    Downloading ResNet-18 pretrained weights...")
                resnet18 = models.resnet18(pretrained=True)
                print(f"    Caching model to {model_path}")
                torch.save(resnet18.state_dict(), model_path)

            return resnet18

        except Exception as e:
            print(f"    ⚠ Error with cached model: {e}")
            print("    Downloading fresh ResNet-18 weights...")
            resnet18 = models.resnet18(pretrained=True)
            try:
                torch.save(resnet18.state_dict(), model_path)
            except:
                pass  # Cache save failed, but model loaded
            return resnet18

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass

        Args:
            x: Input images [B, 3, 224, 224]

        Returns:
            L2-normalized embeddings [B, embedding_dim]
        """
        # Extract GoogleNet features
        googlenet_out = self.googlenet_features(x)
        googlenet_out = googlenet_out.view(googlenet_out.size(0), -1)

        # Extract ResNet features
        resnet_out = self.resnet_features(x)
        resnet_out = resnet_out.view(resnet_out.size(0), -1)

        # Concatenate features
        combined = torch.cat([googlenet_out, resnet_out], dim=1)

        # Fusion to embedding
        embedding = self.fusion(combined)

        # L2 normalization (critical for face recognition)
        embedding = F.normalize(embedding, p=2, dim=1)

        return embedding