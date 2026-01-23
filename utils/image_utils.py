"""
FaceMatch Pro - Image Utilities
Image loading, preprocessing, and manipulation
"""

import cv2
import numpy as np
from PIL import Image
import io
import streamlit as st
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))
from config import Config


def load_image(uploaded_file):
    """
    Load image from Streamlit uploaded file

    Args:
        uploaded_file: UploadedFile object from st.file_uploader

    Returns:
        numpy array (H, W, 3) in RGB format, or None if error
    """
    try:
        # Read bytes
        bytes_data = uploaded_file.read()

        # Convert to PIL Image
        pil_image = Image.open(io.BytesIO(bytes_data))

        # Convert to RGB if needed
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')

        # Convert to numpy array
        image = np.array(pil_image)

        return image

    except Exception as e:
        st.error(f"Error loading image: {str(e)}")
        return None


def load_image_from_path(image_path):
    """
    Load image from file path

    Args:
        image_path: str or Path object

    Returns:
        numpy array (H, W, 3) in RGB format
    """
    try:
        image = cv2.imread(str(image_path))
        if image is None:
            return None

        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        return image

    except Exception as e:
        st.error(f"Error loading image from path: {str(e)}")
        return None


def validate_image(uploaded_file):
    """
    Validate uploaded image

    Args:
        uploaded_file: UploadedFile object

    Returns:
        tuple: (is_valid, error_message)
    """
    # Check if file exists
    if uploaded_file is None:
        return False, "No file uploaded"

    # Check file size
    file_size = uploaded_file.size
    if file_size > Config.MAX_IMAGE_SIZE:
        max_mb = Config.MAX_IMAGE_SIZE / (1024 * 1024)
        return False, f"File too large. Maximum size: {max_mb:.1f} MB"

    # Check file format
    file_ext = uploaded_file.name.split('.')[-1].lower()
    if file_ext not in Config.SUPPORTED_FORMATS:
        return False, f"Unsupported format. Supported: {', '.join(Config.SUPPORTED_FORMATS)}"

    return True, None


def resize_image(image, target_size=None):
    """
    Resize image to target size while maintaining aspect ratio

    Args:
        image: numpy array
        target_size: int or tuple (width, height)

    Returns:
        resized_image: numpy array
    """
    if target_size is None:
        target_size = Config.IMAGE_SIZE

    if isinstance(target_size, int):
        target_size = (target_size, target_size)

    try:
        resized = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
        return resized

    except Exception as e:
        st.error(f"Error resizing image: {str(e)}")
        return image


def preprocess_image(image):
    """
    Preprocess image for display (consistent size and format)

    Args:
        image: numpy array

    Returns:
        preprocessed_image: numpy array
    """
    # Resize to reasonable display size
    max_size = 800
    h, w = image.shape[:2]

    if max(h, w) > max_size:
        scale = max_size / max(h, w)
        new_w = int(w * scale)
        new_h = int(h * scale)
        image = cv2.resize(image, (new_w, new_h))

    return image


def convert_to_pil(image):
    """
    Convert numpy array to PIL Image

    Args:
        image: numpy array in RGB

    Returns:
        PIL Image
    """
    return Image.fromarray(image.astype('uint8'))


def convert_to_bytes(image, format='JPEG'):
    """
    Convert numpy array to bytes for download

    Args:
        image: numpy array in RGB
        format: output format (JPEG, PNG)

    Returns:
        bytes object
    """
    pil_image = convert_to_pil(image)

    buffer = io.BytesIO()
    pil_image.save(buffer, format=format)
    buffer.seek(0)

    return buffer.getvalue()


def create_side_by_side(image1, image2, gap=20):
    """
    Create side-by-side comparison of two images

    Args:
        image1: numpy array
        image2: numpy array
        gap: pixels between images

    Returns:
        combined_image: numpy array
    """
    # Make images same height
    h1, w1 = image1.shape[:2]
    h2, w2 = image2.shape[:2]

    target_height = max(h1, h2)

    if h1 != target_height:
        scale = target_height / h1
        image1 = cv2.resize(image1, (int(w1 * scale), target_height))

    if h2 != target_height:
        scale = target_height / h2
        image2 = cv2.resize(image2, (int(w2 * scale), target_height))

    # Create gap
    gap_array = np.ones((target_height, gap, 3), dtype=np.uint8) * 255

    # Concatenate
    combined = np.hstack([image1, gap_array, image2])

    return combined


def add_text_overlay(image, text, position='top', bg_color=(0, 0, 0),
                     text_color=(255, 255, 255), alpha=0.7):
    """
    Add text overlay to image

    Args:
        image: numpy array
        text: str or list of str
        position: 'top', 'bottom', 'center'
        bg_color: background color (B, G, R)
        text_color: text color (B, G, R)
        alpha: background transparency

    Returns:
        image with text overlay
    """
    img = image.copy()
    h, w = img.shape[:2]

    if isinstance(text, str):
        text = [text]

    # Calculate text size
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    thickness = 2

    text_sizes = [cv2.getTextSize(t, font, font_scale, thickness)[0] for t in text]
    max_text_width = max([ts[0] for ts in text_sizes])
    total_text_height = sum([ts[1] for ts in text_sizes]) + len(text) * 10

    # Determine position
    if position == 'top':
        y_start = 10
    elif position == 'bottom':
        y_start = h - total_text_height - 10
    else:  # center
        y_start = (h - total_text_height) // 2

    # Draw background rectangle
    overlay = img.copy()
    cv2.rectangle(
        overlay,
        (5, y_start - 5),
        (max_text_width + 15, y_start + total_text_height + 5),
        bg_color,
        -1
    )

    # Blend overlay
    img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

    # Draw text
    y = y_start + 20
    for t in text:
        cv2.putText(img, t, (10, y), font, font_scale, text_color, thickness)
        y += text_sizes[0][1] + 10

    return img


def add_match_badge(image, is_match, similarity, position='top-right'):
    """
    Add match/no-match badge to image

    Args:
        image: numpy array
        is_match: bool
        similarity: float (0-1)
        position: 'top-left', 'top-right', 'bottom-left', 'bottom-right'

    Returns:
        image with badge
    """
    img = image.copy()
    h, w = img.shape[:2]

    # Badge properties
    badge_text = "✓ MATCH" if is_match else "✗ NO MATCH"
    score_text = f"{similarity * 100:.1f}%"

    color = Config.COLORS['match'] if is_match else Config.COLORS['no_match']
    # Convert hex to BGR
    r, g, b = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)
    bgr_color = (b, g, r)

    font = cv2.FONT_HERSHEY_SIMPLEX

    # Calculate positions
    badge_size = cv2.getTextSize(badge_text, font, 0.8, 2)[0]
    score_size = cv2.getTextSize(score_text, font, 0.6, 2)[0]

    padding = 15
    badge_width = max(badge_size[0], score_size[0]) + 2 * padding
    badge_height = badge_size[1] + score_size[1] + 3 * padding

    if 'right' in position:
        x = w - badge_width - 10
    else:
        x = 10

    if 'bottom' in position:
        y = h - badge_height - 10
    else:
        y = 10

    # Draw badge background
    overlay = img.copy()
    cv2.rectangle(overlay, (x, y), (x + badge_width, y + badge_height), bgr_color, -1)
    img = cv2.addWeighted(overlay, 0.8, img, 0.2, 0)

    # Draw border
    cv2.rectangle(img, (x, y), (x + badge_width, y + badge_height), bgr_color, 3)

    # Draw text
    text_x = x + padding
    text_y = y + padding + badge_size[1]

    cv2.putText(img, badge_text, (text_x, text_y), font, 0.8, (255, 255, 255), 2)
    cv2.putText(img, score_text, (text_x, text_y + score_size[1] + padding),
                font, 0.6, (255, 255, 255), 2)

    return img


def apply_blur(image, kernel_size=15):
    """Apply Gaussian blur to image"""
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)


def adjust_brightness(image, factor=1.2):
    """
    Adjust image brightness

    Args:
        image: numpy array
        factor: brightness factor (>1 brighter, <1 darker)

    Returns:
        adjusted image
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    hsv = hsv.astype(np.float32)
    hsv[:, :, 2] = hsv[:, :, 2] * factor
    hsv[:, :, 2] = np.clip(hsv[:, :, 2], 0, 255)
    hsv = hsv.astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


def adjust_contrast(image, factor=1.2):
    """
    Adjust image contrast

    Args:
        image: numpy array
        factor: contrast factor

    Returns:
        adjusted image
    """
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    lab = lab.astype(np.float32)
    lab[:, :, 0] = lab[:, :, 0] * factor
    lab[:, :, 0] = np.clip(lab[:, :, 0], 0, 255)
    lab = lab.astype(np.uint8)
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def create_thumbnail(image, size=128):
    """Create thumbnail of image"""
    return cv2.resize(image, (size, size))


def batch_load_images(uploaded_files, show_progress=True):
    """
    Load multiple images from uploaded files

    Args:
        uploaded_files: list of UploadedFile objects
        show_progress: show progress bar

    Returns:
        list of numpy arrays
    """
    images = []

    iterator = enumerate(uploaded_files)
    if show_progress:
        progress_bar = st.progress(0)
        status_text = st.empty()

    for i, file in iterator:
        image = load_image(file)
        if image is not None:
            images.append(image)

        if show_progress:
            progress = (i + 1) / len(uploaded_files)
            progress_bar.progress(progress)
            status_text.text(f"Loading: {i + 1}/{len(uploaded_files)}")

    if show_progress:
        progress_bar.empty()
        status_text.empty()

    return images