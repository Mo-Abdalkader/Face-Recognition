"""
Embedding Generator for Face Recognition
Uses trained model to generate face embeddings
"""

import torch
import numpy as np
from typing import List, Union, Optional
from pathlib import Path
import cv2


class FaceEmbeddingGenerator:
    """
    Generate embeddings for face images using trained model
    """

    def __init__(
            self,
            model: torch.nn.Module,
            transform: object,
            face_cropper: Optional[object] = None,
            device: Optional[torch.device] = None
    ):
        """
        Args:
            model: Trained face recognition model
            transform: Image transformation pipeline
            face_cropper: FaceCropper instance (optional)
            device: torch.device
        """
        self.model = model
        self.transform = transform
        self.face_cropper = face_cropper
        self.device = device if device is not None else torch.device('cpu')

        self.model.to(self.device)
        self.model.eval()

        print(f"✓ Embedding generator initialized on {self.device}")

    def generate_embedding(self, image_path: Union[str, Path]) -> Optional[np.ndarray]:
        """
        Generate embedding for single image

        Args:
            image_path: Path to face image

        Returns:
            Embedding array [512,] or None if error
        """
        # Load image
        img = cv2.imread(str(image_path))
        if img is None:
            print(f"Error: Cannot load image {image_path}")
            return None

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Apply face cropping if enabled
        if self.face_cropper is not None:
            cropped = self.face_cropper.detect_and_crop(img)
            if cropped is not None:
                img = cropped
            else:
                print(f"Warning: No face detected in {image_path}, using full image")

        # Transform and add batch dimension
        try:
            img_tensor = self.transform(img).unsqueeze(0).to(self.device)
        except Exception as e:
            print(f"Error transforming image {image_path}: {e}")
            return None

        # Generate embedding
        with torch.no_grad():
            embedding = self.model(img_tensor)

        # Convert to numpy
        embedding = embedding.cpu().numpy().flatten().astype(np.float32)

        return embedding

    def generate_embeddings_batch(
            self,
            image_paths: List[Union[str, Path]]
    ) -> List[np.ndarray]:
        """
        Generate embeddings for multiple images

        Args:
            image_paths: List of image paths

        Returns:
            List of embedding arrays
        """
        embeddings = []

        for img_path in image_paths:
            embedding = self.generate_embedding(img_path)
            if embedding is not None:
                embeddings.append(embedding)

        return embeddings