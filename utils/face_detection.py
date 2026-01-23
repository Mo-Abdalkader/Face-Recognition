"""
FaceMatch Pro - Face Detection Utilities
facenet-pytorch MTCNN-based face detection and processing
"""

import cv2
import numpy as np
from facenet_pytorch import MTCNN
import streamlit as st
from PIL import Image
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))
from config import Config


@st.cache_resource
def get_face_detector():
    """
    Initialize MTCNN face detector (PyTorch version)
    Cached to avoid reloading
    """
    try:
        detector = MTCNN(
            keep_all=True,
            device='cpu',
            post_process=False,
            # min_face_size parameter removed - not supported in facenet-pytorch
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
        # Convert to PIL Image
        pil_image = Image.fromarray(image)
        
        # Detect faces - returns (boxes, probs, landmarks)
        boxes, probs, landmarks = detector.detect(pil_image, landmarks=True)

        # Handle no faces detected
        if boxes is None:
            return []

        # Convert to expected format
        filtered_faces = []
        for i, box in enumerate(boxes):
            confidence = float(probs[i])
            
            if confidence >= min_confidence:
                # Convert from [x1, y1, x2, y2] to [x, y, w, h]
                x1, y1, x2, y2 = map(int, box)
                x, y, w, h = x1, y1, x2 - x1, y2 - y1
                
                face_dict = {
                    'box': [x, y, w, h],
                    'confidence': confidence
                }
                
                # Add keypoints if available
                if landmarks is not None:
                    face_dict['keypoints'] = {
                        'left_eye': tuple(map(int, landmarks[i][0])),
                        'right_eye': tuple(map(int, landmarks[i][1])),
                        'nose': tuple(map(int, landmarks[i][2])),
                        'mouth_left': tuple(map(int, landmarks[i][3])),
                        'mouth_right': tuple(map(int, landmarks[i][4]))
                    }
                
                filtered_faces.append(face_dict)

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
    annotated = image.copy()

    if color is None:
        color = Config.FACE_RECT_COLOR
    if thickness is None:
        thickness = Config.FACE_RECT_THICKNESS

    for idx, face in enumerate(faces):
        x, y, w, h = face['box']
        confidence = face['confidence']

        cv2.rectangle(annotated, (x, y), (x + w, y + h), color, thickness)

        if labels:
            label = f"Face {idx + 1}"
            conf_text = f"{confidence:.2f}"

            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(
                annotated,
                (x, y - label_size[1] - 10),
                (x + label_size[0] + 10, y),
                Config.FACE_LABEL_BG_COLOR,
                -1
            )

            cv2.putText(
                annotated, label, (x + 5, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                Config.FACE_LABEL_COLOR, 2
            )

            cv2.putText(
                annotated, conf_text, (x, y + h + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1
            )

        if 'keypoints' in face:
            for point in face['keypoints'].values():
                cv2.circle(annotated, point, 2, (0, 255, 0), 2)

    return annotated


def crop_face(image, face, margin=0.2):
    """Crop face region from image with margin"""
    x, y, w, h = face['box']

    margin_x = int(w * margin)
    margin_y = int(h * margin)

    x1 = max(0, x - margin_x)
    y1 = max(0, y - margin_y)
    x2 = min(image.shape[1], x + w + margin_x)
    y2 = min(image.shape[0], y + h + margin_y)

    return image[y1:y2, x1:x2]


def get_best_face(faces):
    """Get the best face from a list based on confidence and size"""
    if len(faces) == 0:
        return None
    if len(faces) == 1:
        return faces[0]

    scores = []
    for face in faces:
        confidence = face['confidence']
        w, h = face['box'][2], face['box'][3]
        area = w * h
        max_area = max([f['box'][2] * f['box'][3] for f in faces])
        normalized_area = area / max_area
        score = 0.7 * confidence + 0.3 * normalized_area
        scores.append(score)

    return faces[np.argmax(scores)]


def align_face(image, face):
    """Align face based on eye positions"""
    if 'keypoints' not in face:
        return crop_face(image, face)

    keypoints = face['keypoints']
    left_eye = keypoints['left_eye']
    right_eye = keypoints['right_eye']

    dx = right_eye[0] - left_eye[0]
    dy = right_eye[1] - left_eye[1]
    angle = np.degrees(np.arctan2(dy, dx))

    center = ((left_eye[0] + right_eye[0]) // 2, (left_eye[1] + right_eye[1]) // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    aligned = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))

    return crop_face(aligned, face)


def batch_detect_faces(images, min_confidence=0.9, show_progress=True):
    """Detect faces in multiple images"""
    all_faces = []

    if show_progress:
        progress_bar = st.progress(0)
        status_text = st.empty()

    for i, image in enumerate(images):
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
    """Get statistics about detected faces"""
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
    """Create comprehensive visualization of face detection"""
    annotated = draw_face_rectangles(image, faces, labels=True)
    stats = get_face_statistics(faces)
    return annotated, stats


def is_face_quality_good(face, image_shape):
    """Quick check if face quality is acceptable"""
    x, y, w, h = face['box']
    confidence = face['confidence']

    if confidence < 0.95:
        return False

    face_area = w * h
    image_area = image_shape[0] * image_shape[1]
    if face_area < 0.05 * image_area:
        return False

    margin = 10
    if x < margin or y < margin:
        return False
    if x + w > image_shape[1] - margin or y + h > image_shape[0] - margin:
        return False

    return True


def create_face_grid(images, faces_list, grid_cols=3):
    """Create a grid visualization of detected faces"""
    all_face_crops = []

    for image, faces in zip(images, faces_list):
        for face in faces:
            crop = crop_face(image, face, margin=0.1)
            crop_resized = cv2.resize(crop, (128, 128))
            all_face_crops.append(crop_resized)

    if len(all_face_crops) == 0:
        return None

    num_faces = len(all_face_crops)
    grid_rows = (num_faces + grid_cols - 1) // grid_cols

    grid = []
    for i in range(grid_rows):
        row = []
        for j in range(grid_cols):
            idx = i * grid_cols + j
            if idx < num_faces:
                row.append(all_face_crops[idx])
            else:
                row.append(np.zeros((128, 128, 3), dtype=np.uint8))
        row_img = np.hstack(row)
        grid.append(row_img)

    return np.vstack(grid)
