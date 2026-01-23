"""
FaceMatch Pro - Quality Check Utilities
Image and face quality analysis
"""

import cv2
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))
from config import Config


def calculate_laplacian_variance(image):
    """
    Calculate blur score using Laplacian variance
    Higher value = sharper image

    Args:
        image: numpy array (H, W, 3) in RGB

    Returns:
        float: blur score
    """
    # Convert to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image

    # Calculate Laplacian variance
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    variance = laplacian.var()

    return variance


def calculate_brightness(image):
    """
    Calculate average brightness of image

    Args:
        image: numpy array (H, W, 3) in RGB

    Returns:
        float: brightness value (0-255)
    """
    # Convert to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image

    return np.mean(gray)


def calculate_contrast(image):
    """
    Calculate image contrast (standard deviation of pixel values)

    Args:
        image: numpy array

    Returns:
        float: contrast value
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image

    return np.std(gray)


def check_exposure(brightness):
    """
    Check if image is properly exposed

    Args:
        brightness: average brightness value

    Returns:
        tuple: (status, message)
    """
    if brightness < 50:
        return 'underexposed', 'Image is too dark'
    elif brightness > 200:
        return 'overexposed', 'Image is too bright'
    else:
        return 'good', 'Exposure is good'


def check_blur(blur_score):
    """
    Check if image is blurry

    Args:
        blur_score: Laplacian variance

    Returns:
        tuple: (status, message)
    """
    threshold = Config.QUALITY_SCORES['blur_threshold']

    if blur_score < threshold * 0.5:
        return 'very_blurry', 'Image is very blurry'
    elif blur_score < threshold:
        return 'slightly_blurry', 'Image is slightly blurry'
    else:
        return 'sharp', 'Image is sharp'


def analyze_face_quality(face_image, face_box=None):
    """
    Comprehensive face quality analysis

    Args:
        face_image: numpy array of face crop
        face_box: optional face bounding box (x, y, w, h)

    Returns:
        dict with quality metrics
    """
    # Basic metrics
    blur_score = calculate_laplacian_variance(face_image)
    brightness = calculate_brightness(face_image)
    contrast = calculate_contrast(face_image)

    # Check exposure
    exposure_status, exposure_msg = check_exposure(brightness)

    # Check blur
    blur_status, blur_msg = check_blur(blur_score)

    # Face size (if box provided)
    face_size = 'unknown'
    face_size_pixels = 0
    if face_box is not None:
        x, y, w, h = face_box
        face_size_pixels = w * h
        if face_size_pixels < 2500:  # 50x50
            face_size = 'too_small'
        elif face_size_pixels < 10000:  # 100x100
            face_size = 'small'
        elif face_size_pixels < 40000:  # 200x200
            face_size = 'medium'
        else:
            face_size = 'large'

    # Overall quality score (0-100)
    quality_score = calculate_overall_quality(
        blur_score, brightness, contrast, face_size_pixels
    )

    return {
        'blur_score': blur_score,
        'blur_status': blur_status,
        'blur_message': blur_msg,
        'brightness': brightness,
        'exposure_status': exposure_status,
        'exposure_message': exposure_msg,
        'contrast': contrast,
        'face_size': face_size,
        'face_size_pixels': face_size_pixels,
        'quality_score': quality_score,
        'quality_level': get_quality_level(quality_score)
    }


def calculate_overall_quality(blur_score, brightness, contrast, face_size):
    """
    Calculate overall quality score (0-100)

    Args:
        blur_score: Laplacian variance
        brightness: average brightness
        contrast: standard deviation
        face_size: face area in pixels

    Returns:
        float: quality score (0-100)
    """
    # Normalize blur score (0-100)
    blur_normalized = min(100, (blur_score / Config.QUALITY_SCORES['blur_threshold']) * 100)

    # Brightness score (optimal at 100-150)
    if 100 <= brightness <= 150:
        brightness_score = 100
    elif brightness < 100:
        brightness_score = (brightness / 100) * 100
    else:
        brightness_score = max(0, 100 - (brightness - 150) / 2)

    # Contrast score (higher is better, up to a point)
    contrast_score = min(100, (contrast / 50) * 100)

    # Size score
    if face_size < 2500:
        size_score = 0
    elif face_size < 10000:
        size_score = 50
    else:
        size_score = 100

    # Weighted average
    quality = (
            blur_normalized * 0.4 +
            brightness_score * 0.2 +
            contrast_score * 0.2 +
            size_score * 0.2
    )

    return min(100, max(0, quality))


def get_quality_level(quality_score):
    """
    Get quality level from score

    Args:
        quality_score: float (0-100)

    Returns:
        str: quality level
    """
    if quality_score >= 80:
        return 'excellent'
    elif quality_score >= 60:
        return 'good'
    elif quality_score >= 40:
        return 'fair'
    else:
        return 'poor'


def get_quality_stars(quality_score):
    """
    Convert quality score to star rating (1-5)

    Args:
        quality_score: float (0-100)

    Returns:
        int: number of stars (1-5)
    """
    return Config.get_quality_stars(quality_score)


def check_face_angle(keypoints):
    """
    Estimate face angle from keypoints

    Args:
        keypoints: dict with facial landmarks

    Returns:
        dict with angle information
    """
    if 'left_eye' not in keypoints or 'right_eye' not in keypoints:
        return {'status': 'unknown', 'angle': 0, 'message': 'Cannot estimate angle'}

    left_eye = np.array(keypoints['left_eye'])
    right_eye = np.array(keypoints['right_eye'])

    # Calculate eye distance and angle
    dx = right_eye[0] - left_eye[0]
    dy = right_eye[1] - left_eye[1]

    angle = np.degrees(np.arctan2(dy, dx))

    # Determine if angle is acceptable
    if abs(angle) < 5:
        status = 'frontal'
        message = 'Good frontal view'
    elif abs(angle) < 15:
        status = 'slight_angle'
        message = 'Slight angle, acceptable'
    else:
        status = 'angled'
        message = 'Significant angle, may affect accuracy'

    return {
        'status': status,
        'angle': abs(angle),
        'message': message
    }


def generate_quality_report(face_image, face_box=None, keypoints=None):
    """
    Generate comprehensive quality report

    Args:
        face_image: numpy array
        face_box: optional bounding box
        keypoints: optional facial landmarks

    Returns:
        dict with complete quality report
    """
    # Basic quality analysis
    quality = analyze_face_quality(face_image, face_box)

    # Add angle analysis if keypoints available
    if keypoints:
        angle_info = check_face_angle(keypoints)
        quality['angle_status'] = angle_info['status']
        quality['angle_degrees'] = angle_info['angle']
        quality['angle_message'] = angle_info['message']

    # Generate recommendations
    recommendations = []

    if quality['blur_status'] in ['very_blurry', 'slightly_blurry']:
        recommendations.append("Use a sharper image or enable image stabilization")

    if quality['exposure_status'] == 'underexposed':
        recommendations.append("Increase lighting or adjust camera exposure")
    elif quality['exposure_status'] == 'overexposed':
        recommendations.append("Reduce lighting or decrease camera exposure")

    if quality['face_size'] == 'too_small':
        recommendations.append("Move closer to camera or use higher resolution")
    elif quality['face_size'] == 'small':
        recommendations.append("Consider using a closer shot for better accuracy")

    if quality['contrast'] < 20:
        recommendations.append("Improve lighting contrast")

    if keypoints and quality.get('angle_status') == 'angled':
        recommendations.append("Face camera more directly for better results")

    quality['recommendations'] = recommendations
    quality['is_acceptable'] = quality['quality_score'] >= 40

    return quality


def create_quality_visualization(image, quality_report):
    """
    Create visual representation of quality metrics

    Args:
        image: original image
        quality_report: quality report dict

    Returns:
        annotated image with quality info
    """
    img = image.copy()
    h, w = img.shape[:2]

    # Create overlay
    overlay = img.copy()

    # Background for text
    cv2.rectangle(overlay, (10, 10), (w - 10, 150), (0, 0, 0), -1)
    img = cv2.addWeighted(overlay, 0.7, img, 0.3, 0)

    # Add quality info
    font = cv2.FONT_HERSHEY_SIMPLEX
    y = 40

    # Quality score
    score_color = (0, 255, 0) if quality_report['quality_score'] >= 60 else (0, 165, 255)
    cv2.putText(img, f"Quality: {quality_report['quality_score']:.0f}/100",
                (20, y), font, 0.7, score_color, 2)

    # Stars
    stars = get_quality_stars(quality_report['quality_score'])
    cv2.putText(img, f"{'★' * stars}{'☆' * (5 - stars)}",
                (20, y + 30), font, 0.6, (255, 255, 0), 2)

    # Status
    cv2.putText(img, f"Blur: {quality_report['blur_status']}",
                (20, y + 60), font, 0.5, (255, 255, 255), 1)
    cv2.putText(img, f"Exposure: {quality_report['exposure_status']}",
                (20, y + 85), font, 0.5, (255, 255, 255), 1)
    cv2.putText(img, f"Size: {quality_report['face_size']}",
                (20, y + 110), font, 0.5, (255, 255, 255), 1)

    return img


def batch_quality_analysis(images, faces_list):
    """
    Analyze quality of multiple face images

    Args:
        images: list of images
        faces_list: list of face detections per image

    Returns:
        list of quality reports
    """
    reports = []

    for image, faces in zip(images, faces_list):
        if len(faces) > 0:
            # Use first/best face
            face = faces[0]

            # Crop face
            x, y, w, h = face['box']
            face_crop = image[y:y + h, x:x + w]

            # Analyze
            report = generate_quality_report(
                face_crop,
                face['box'],
                face.get('keypoints')
            )

            reports.append(report)
        else:
            reports.append(None)

    return reports


def get_quality_statistics(reports):
    """
    Get statistics from multiple quality reports

    Args:
        reports: list of quality reports

    Returns:
        dict with statistics
    """
    valid_reports = [r for r in reports if r is not None]

    if len(valid_reports) == 0:
        return None

    quality_scores = [r['quality_score'] for r in valid_reports]

    return {
        'total_images': len(reports),
        'valid_faces': len(valid_reports),
        'avg_quality': np.mean(quality_scores),
        'min_quality': np.min(quality_scores),
        'max_quality': np.max(quality_scores),
        'excellent_count': sum(1 for r in valid_reports if r['quality_level'] == 'excellent'),
        'good_count': sum(1 for r in valid_reports if r['quality_level'] == 'good'),
        'fair_count': sum(1 for r in valid_reports if r['quality_level'] == 'fair'),
        'poor_count': sum(1 for r in valid_reports if r['quality_level'] == 'poor'),
    }