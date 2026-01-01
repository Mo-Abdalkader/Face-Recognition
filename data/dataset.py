"""
VGGFace2 Dataset Loader with Triplet Sampling
Includes face cropping pipeline
"""

import random
from pathlib import Path
from typing import Tuple, Optional, Callable
import cv2
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm


class VGGFace2TripletDataset(Dataset):
    """
    VGGFace2 dataset with triplet sampling
    Automatically crops faces using FaceCropper
    """

    def __init__(
            self,
            root_dir: str,
            transform: Optional[Callable] = None,
            samples_per_identity: int = 30,
            triplets_per_identity: int = 20,
            face_cropper: Optional[object] = None,
            cache_crops: bool = False
    ):
        """
        Args:
            root_dir: Path to VGGFace2 train/val directory
            transform: Image transformations
            samples_per_identity: Max images per person
            triplets_per_identity: Triplets generated per identity per epoch
            face_cropper: FaceCropper instance (optional)
            cache_crops: Cache cropped faces in memory (requires more RAM)
        """
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.samples_per_identity = samples_per_identity
        self.triplets_per_identity = triplets_per_identity
        self.face_cropper = face_cropper
        self.cache_crops = cache_crops

        self.identities = []
        self.identity_to_images = {}
        self.cropped_cache = {} if cache_crops else None

        self._load_dataset()
        self.dataset_size = len(self.identities) * self.triplets_per_identity

    def _load_dataset(self):
        """Build identity -> images mapping"""
        identity_folders = [f for f in self.root_dir.iterdir() if f.is_dir()]

        print(f"Loading dataset from: {self.root_dir}")
        print(f"Found {len(identity_folders)} identity folders")

        for identity_folder in tqdm(identity_folders, desc="Loading identities"):
            identity_id = identity_folder.name

            image_files = list(identity_folder.glob('*.jpg')) + \
                          list(identity_folder.glob('*.png'))

            if len(image_files) > self.samples_per_identity:
                image_files = random.sample(image_files, self.samples_per_identity)

            if len(image_files) >= 2:
                self.identities.append(identity_id)
                self.identity_to_images[identity_id] = image_files

        print(f"✓ Loaded {len(self.identities)} identities")
        print(f"✓ Dataset size: {len(self.identities) * self.triplets_per_identity} triplets per epoch")

    def __len__(self) -> int:
        return self.dataset_size

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate triplet: (anchor, positive, negative)
        """
        # Select anchor identity
        anchor_identity = random.choice(self.identities)
        anchor_images = self.identity_to_images[anchor_identity]

        # Sample anchor and positive
        if len(anchor_images) < 2:
            anchor_img_path = anchor_images[0]
            positive_img_path = anchor_images[0]
        else:
            anchor_img_path, positive_img_path = random.sample(anchor_images, 2)

        # Sample negative from different identity
        negative_identity = random.choice(
            [i for i in self.identities if i != anchor_identity]
        )
        negative_img_path = random.choice(
            self.identity_to_images[negative_identity]
        )

        # Load images
        anchor = self._load_image(anchor_img_path)
        positive = self._load_image(positive_img_path)
        negative = self._load_image(negative_img_path)

        return anchor, positive, negative

    def _load_image(self, img_path: Path) -> np.ndarray:
        """Load and preprocess image with optional face cropping"""
        # Check cache first
        if self.cache_crops and str(img_path) in self.cropped_cache:
            img = self.cropped_cache[str(img_path)]
        else:
            # Load image
            img = cv2.imread(str(img_path))
            if img is None:
                raise ValueError(f"Cannot load image: {img_path}")

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Apply face cropping if cropper provided
            if self.face_cropper is not None:
                cropped = self.face_cropper.detect_and_crop(img)
                if cropped is not None:
                    img = cropped
                # If face detection fails, use full image (no error)

            # Cache if enabled
            if self.cache_crops:
                self.cropped_cache[str(img_path)] = img.copy()

        # Apply transforms
        if self.transform:
            img = self.transform(img)

        return img

    def get_identity_count(self) -> int:
        """Return number of unique identities"""
        return len(self.identities)