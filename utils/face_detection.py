"""
FaceMatch Pro - Face Detection Utilities
MTCNN-based face detection and processing
"""

import cv2
import numpy as np
from mtcnn import MTCNN
import streamlit as st
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))
from config import Config


@st.cache_resource
def get_face_detector():
    """
    Initialize MTCNN face detector
    Cached to avoid reloading
    """

    try:
        detector = MTCNN(
            min_face_size=Config.MTCNN_MIN_FACE_SIZE,
            thresholds=Config.MTCNN_THRESHOLDS,
            scale_factor=Config.MTCNN_FACTOR
        )
        return detector
    except Exception as e:
        st.error(f"Error initializing face detector: {str(e)}")
        return None


def detect_faces(image, min_confidence=0.9):
    """
    Detect all faces in an image using MTCNN

    Args:
        image: numpy array (H, W, 3) in RGB
        min_confidence: minimum confidence threshold for detection

    Returns:
        list of dicts with keys: 'box', 'confidence', 'keypoints'
        box format: [x, y, width, height]
    """
    detector = get_face_detector()

    if detector is None:
        return []

    try:
        # Detect faces
        faces = detector.detect_faces(image)

        # Filter by confidence
        filtered_faces = []
        for face in faces:
            if face['confidence'] >= min_confidence:
                filtered_faces.append(face)

        return filtered_faces

    except Exception as e:
        st.error(f"Error detecting faces: {str(e)}")
        return []


def draw_face_rectangles(image, faces, labels=True, color=None, thickness=None):
    """
    Draw rectangles around detected faces

    Args:
        image: numpy array (H, W, 3) in RGB
        faces: list of face dicts from detect_faces()
        labels: whether to draw face labels
        color: rectangle color (B, G, R) tuple, None for default
        thickness: line thickness, None for default

    Returns:
        annotated_image: numpy array with rectangles drawn
    """
    # Make a copy to avoid modifying original
    annotated = image.copy()

    # Default color and thickness
    if color is None:
        color = Config.FACE_RECT_COLOR
    if thickness is None:
        thickness = Config.FACE_RECT_THICKNESS

    for idx, face in enumerate(faces):
        # Get box coordinates
        x, y, w, h = face['box']
        confidence = face['confidence']

        # Draw rectangle
        cv2.rectangle(annotated, (x, y), (x + w, y + h), color, thickness)

        # Draw label if requested
        if labels:
            label = f"Face {idx + 1}"
            conf_text = f"{confidence:.2f}"

            # Label background
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(
                annotated,
                (x, y - label_size[1] - 10),
                (x + label_size[0] + 10, y),
                Config.FACE_LABEL_BG_COLOR,
                -1
            )

            # Label text
            cv2.putText(
                annotated,
                label,
                (x + 5, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                Config.FACE_LABEL_COLOR,
                2
            )

            # Confidence text
            cv2.putText(
                annotated,
                conf_text,
                (x, y + h + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1
            )

        # Draw keypoints (optional - eyes, nose, mouth corners)
        if 'keypoints' in face:
            keypoints = face['keypoints']
            for key, point in keypoints.items():
                cv2.circle(annotated, point, 2, (0, 255, 0), 2)

    return annotated


def crop_face(image, face, margin=0.2):
    """
    Crop face region from image with margin

    Args:
        image: numpy array (H, W, 3) in RGB
        face: face dict from detect_faces()
        margin: percentage margin to add around face (0.2 = 20%)

    Returns:
        cropped_face: numpy array of cropped face
    """
    x, y, w, h = face['box']

    # Add margin
    margin_x = int(w * margin)
    margin_y = int(h * margin)

    # Calculate crop coordinates with margin
    x1 = max(0, x - margin_x)
    y1 = max(0, y - margin_y)
    x2 = min(image.shape[1], x + w + margin_x)
    y2 = min(image.shape[0], y + h + margin_y)

    # Crop
    cropped = image[y1:y2, x1:x2]

    return cropped


def get_best_face(faces):
    """
    Get the best face from a list of detected faces
    Based on confidence and size

    Args:
        faces: list of face dicts

    Returns:
        best_face: face dict with highest score
    """
    if len(faces) == 0:
        return None

    if len(faces) == 1:
        return faces[0]

    # Score each face
    scores = []
    for face in faces:
        confidence = face['confidence']
        x, y, w, h = face['box']
        area = w * h

        # Weighted score: confidence (70%) + normalized area (30%)
        max_area = max([f['box'][2] * f['box'][3] for f in faces])
        normalized_area = area / max_area

        score = 0.7 * confidence + 0.3 * normalized_area
        scores.append(score)

    # Return face with highest score
    best_idx = np.argmax(scores)
    return faces[best_idx]


def align_face(image, face):
    """
    Align face based on eye positions

    Args:
        image: numpy array (H, W, 3) in RGB
        face: face dict with keypoints

    Returns:
        aligned_face: aligned face image
    """
    if 'keypoints' not in face:
        # No keypoints, return cropped face
        return crop_face(image, face)

    keypoints = face['keypoints']

    # Get eye positions
    left_eye = keypoints['left_eye']
    right_eye = keypoints['right_eye']

    # Calculate angle
    dx = right_eye[0] - left_eye[0]
    dy = right_eye[1] - left_eye[1]
    angle = np.degrees(np.arctan2(dy, dx))

    # Calculate center
    center = ((left_eye[0] + right_eye[0]) // 2, (left_eye[1] + right_eye[1]) // 2)

    # Rotation matrix
    M = cv2.getRotationMatrix2D(center, angle, 1.0)

    # Rotate image
    aligned = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))

    # Crop face from aligned image
    aligned_face = crop_face(aligned, face)

    return aligned_face


def batch_detect_faces(images, min_confidence=0.9, show_progress=True):
    """
    Detect faces in multiple images

    Args:
        images: list of numpy arrays
        min_confidence: minimum confidence threshold
        show_progress: show progress bar

    Returns:
        all_faces: list of lists (one list of faces per image)
    """
    all_faces = []

    iterator = enumerate(images)
    if show_progress:
        progress_bar = st.progress(0)
        status_text = st.empty()

    for i, image in iterator:
        faces = detect_faces(image, min_confidence)
        all_faces.append(faces)

        if show_progress:
            progress = (i + 1) / len(images)
            progress_bar.progress(progress)
            status_text.text(f"Detecting faces: {i + 1}/{len(images)}")

    if show_progress:
        progress_bar.empty()
        status_text.empty()

    return all_faces


def get_face_statistics(faces):
    """
    Get statistics about detected faces

    Args:
        faces: list of face dicts

    Returns:
        stats: dict with statistics
    """
    if len(faces) == 0:
        return {
            'count': 0,
            'avg_confidence': 0.0,
            'min_confidence': 0.0,
            'max_confidence': 0.0,
            'avg_size': 0.0
        }

    confidences = [f['confidence'] for f in faces]
    sizes = [f['box'][2] * f['box'][3] for f in faces]

    return {
        'count': len(faces),
        'avg_confidence': np.mean(confidences),
        'min_confidence': np.min(confidences),
        'max_confidence': np.max(confidences),
        'avg_size': np.mean(sizes)
    }


def visualize_face_detection(image, faces):
    """
    Create a comprehensive visualization of face detection

    Args:
        image: numpy array (H, W, 3) in RGB
        faces: list of face dicts

    Returns:
        tuple: (annotated_image, stats_dict)
    """
    # Draw rectangles
    annotated = draw_face_rectangles(image, faces, labels=True)

    # Get statistics
    stats = get_face_statistics(faces)

    return annotated, stats


def is_face_quality_good(face, image_shape):
    """
    Quick check if face quality is acceptable

    Args:
        face: face dict
        image_shape: tuple (H, W, C)

    Returns:
        bool: True if quality is good
    """
    x, y, w, h = face['box']
    confidence = face['confidence']

    # Check confidence
    if confidence < 0.95:
        return False

    # Check size (should be at least 5% of image area)
    face_area = w * h
    image_area = image_shape[0] * image_shape[1]
    if face_area < 0.05 * image_area:
        return False

    # Check if face is too close to edges
    margin = 10
    if x < margin or y < margin:
        return False
    if x + w > image_shape[1] - margin or y + h > image_shape[0] - margin:
        return False

    return True


def create_face_grid(images, faces_list, grid_cols=3):
    """
    Create a grid visualization of detected faces

    Args:
        images: list of original images
        faces_list: list of face lists (one per image)
        grid_cols: number of columns in grid

    Returns:
        grid_image: composite image with all faces
    """
    all_face_crops = []

    for image, faces in zip(images, faces_list):
        for face in faces:
            crop = crop_face(image, face, margin=0.1)
            # Resize to standard size
            crop_resized = cv2.resize(crop, (128, 128))
            all_face_crops.append(crop_resized)

    if len(all_face_crops) == 0:
        return None

    # Calculate grid dimensions
    num_faces = len(all_face_crops)
    grid_rows = (num_faces + grid_cols - 1) // grid_cols

    # Create grid
    grid = []
    for i in range(grid_rows):
        row = []
        for j in range(grid_cols):
            idx = i * grid_cols + j
            if idx < num_faces:
                row.append(all_face_crops[idx])
            else:
                # Pad with black image
                row.append(np.zeros((128, 128, 3), dtype=np.uint8))
        row_img = np.hstack(row)
        grid.append(row_img)

    grid_image = np.vstack(grid)

    return grid_image