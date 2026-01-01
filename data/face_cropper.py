"""
Face Detection and Cropping Module
Supports fast and accurate detection modes using MTCNN
"""

# # Fix for Kaggle protobuf/CUDA warnings
# import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import cv2
import numpy as np
from typing import Optional, Tuple, Literal

# Safe MTCNN import with error handling
try:
    from mtcnn import MTCNN

    MTCNN_AVAILABLE = True
except ImportError as e:
    print(f"⚠ MTCNN not available: {e}")
    print("  Face cropping will be disabled")
    MTCNN_AVAILABLE = False
    MTCNN = None


class FaceCropper:
    """
    Face detection and cropping with selectable modes

    Modes:
    - 'fast': Lower accuracy, higher speed (confidence threshold 0.8)
    - 'accurate': Higher accuracy, lower speed (confidence threshold 0.95)
    """

    def __init__(
            self,
            mode: Literal['fast', 'accurate'] = 'accurate',
            margin: float = 0.2,
            min_confidence: float = 0.9
    ):
        """
        Args:
            mode: Detection mode ('fast' or 'accurate')
            margin: Percentage margin around face (0.2 = 20%)
            min_confidence: Minimum detection confidence
        """
        self.mode = mode
        self.margin = margin
        self.min_confidence = min_confidence

        # Set confidence threshold based on mode
        if mode == 'fast':
            self.detection_threshold = 0.8
        else:  # accurate
            self.detection_threshold = 0.95

        # Initialize MTCNN detector if available
        if MTCNN_AVAILABLE:
            try:
                self.detector = MTCNN()
                print(f"✓ MTCNN face detector initialized ({mode} mode)")
            except Exception as e:
                print(f"⚠ MTCNN initialization failed: {e}")
                self.detector = None
        else:
            self.detector = None
            print("⚠ Face detection disabled - MTCNN not available")

    def detect_and_crop(
            self,
            image: np.ndarray,
            return_largest: bool = True
    ) -> Optional[np.ndarray]:
        """
        Detect face and return cropped region

        Args:
            image: Input image (BGR or RGB)
            return_largest: If multiple faces, return largest one

        Returns:
            Cropped face image or None if no face detected
        """
        # If detector not available, return original image
        if self.detector is None:
            return image

        # Ensure image is RGB (MTCNN expects RGB)
        if len(image.shape) == 2:  # Grayscale
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:  # RGBA
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

        # Detect faces
        try:
            faces = self.detector.detect_faces(image)
        except Exception as e:
            # On detection failure, return original image
            return image

        if not faces or len(faces) == 0:
            return None

        # Filter by confidence
        valid_faces = []
        for face_data in faces:
            if face_data['confidence'] >= self.detection_threshold:
                valid_faces.append(face_data)

        if len(valid_faces) == 0:
            return None

        # Select face (largest if multiple)
        if return_largest and len(valid_faces) > 1:
            # Calculate face area and select largest
            selected_face = max(
                valid_faces,
                key=lambda x: self._calculate_face_area(x['box'])
            )
        else:
            selected_face = valid_faces[0]

        # Extract face region with margin
        # MTCNN returns box as [x, y, width, height]
        x, y, width, height = selected_face['box']

        # Handle negative coordinates
        x = max(0, x)
        y = max(0, y)
        width = max(1, width)
        height = max(1, height)

        x1, y1 = x, y
        x2, y2 = x + width, y + height

        # Add margin
        margin_x = int(width * self.margin)
        margin_y = int(height * self.margin)

        # Calculate new coordinates with margin
        x1 = max(0, x1 - margin_x)
        y1 = max(0, y1 - margin_y)
        x2 = min(image.shape[1], x2 + margin_x)
        y2 = min(image.shape[0], y2 + margin_y)

        # Ensure valid crop region
        if x2 <= x1 or y2 <= y1:
            return None

        # Crop face
        face_crop = image[y1:y2, x1:x2]

        # Verify crop is not empty
        if face_crop.size == 0:
            return None

        return face_crop

    def _calculate_face_area(self, box: list) -> int:
        """
        Calculate area of detected face

        Args:
            box: MTCNN box format [x, y, width, height]

        Returns:
            Area in pixels
        """
        x, y, width, height = box
        return max(1, width) * max(1, height)

    def detect_and_crop_file(
            self,
            image_path: str,
            return_largest: bool = True
    ) -> Optional[np.ndarray]:
        """
        Load image from file and detect/crop face

        Args:
            image_path: Path to image file
            return_largest: If multiple faces, return largest one

        Returns:
            Cropped face image or None if no face detected
        """
        image = cv2.imread(image_path)
        if image is None:
            return None

        # Convert BGR to RGB (OpenCV loads as BGR, MTCNN expects RGB)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return self.detect_and_crop(image, return_largest)
