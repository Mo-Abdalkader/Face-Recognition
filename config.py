"""
FaceMatch Pro - Configuration File
All settings and constants for the application
"""

import os
from pathlib import Path


class Config:
    """Main configuration class for FaceMatch Pro"""

    # ==================== APP SETTINGS ====================
    APP_NAME = "FaceMatch Pro"
    APP_NAME_AR = "فيس ماتش برو"
    PAGE_ICON = "🎭"
    LAYOUT = "wide"
    INITIAL_SIDEBAR_STATE = "expanded"

    # Version
    VERSION = "1.0.0"
    VERSION_DATE = "2026-01-22"

    # ==================== PATHS ====================
    BASE_DIR = Path(__file__).parent
    ASSETS_DIR = BASE_DIR / "assets"
    DEMO_DIR = ASSETS_DIR / "demo_images"
    MODELS_DIR = BASE_DIR / "models"
    TEMP_DIR = BASE_DIR / "temp"

    # Model path
    MODEL_PATH = MODELS_DIR / "best_model.pth"

    # ==================== MODEL SETTINGS ====================
    EMBEDDING_DIM = 512
    DROPOUT_RATE = 0.3
    USE_ATTENTION = True

    # Default threshold (from your evaluation)
    DEFAULT_THRESHOLD = 0.65
    MIN_THRESHOLD = 0.0
    MAX_THRESHOLD = 1.0
    THRESHOLD_STEP = 0.05

    # ==================== IMAGE PROCESSING ====================
    # Image size
    IMAGE_SIZE = 224
    NORMALIZE_MEAN = [0.485, 0.456, 0.406]
    NORMALIZE_STD = [0.229, 0.224, 0.225]

    # Upload limits
    MAX_IMAGE_SIZE = 10 * 1024 * 1024  # 10MB
    SUPPORTED_FORMATS = ['jpg', 'jpeg', 'png', 'bmp']
    MAX_BATCH_SIZE = 50

    # ==================== FACE DETECTION (MTCNN) ====================
    MTCNN_MIN_FACE_SIZE = 20
    MTCNN_THRESHOLDS = [0.6, 0.7, 0.7]
    MTCNN_FACTOR = 0.709
    FACE_DETECTION_DEVICE = 'cuda'  # Will auto-fallback to 'cpu'

    # Rectangle drawing
    FACE_RECT_COLOR = (138, 43, 226)  # Blue-Purple
    FACE_RECT_THICKNESS = 2
    FACE_LABEL_COLOR = (255, 255, 255)
    FACE_LABEL_BG_COLOR = (138, 43, 226)

    # ==================== TTA (Test-Time Augmentation) ====================
    USE_TTA = True
    TTA_NUM_AUGMENTS = 4

    # ==================== HEATMAP SETTINGS ====================
    HEATMAP_ALPHA = 0.5  # Overlay transparency
    HEATMAP_COLORMAP = 'jet'  # OpenCV colormap
    HEATMAP_BLUR_SIZE = 15  # Gaussian blur kernel

    # ==================== UI/UX SETTINGS ====================
    # Theme colors
    COLORS = {
        'primary': '#8B2BE2',  # Blue-Violet
        'secondary': '#4169E1',  # Royal Blue
        'success': '#28A745',  # Green
        'danger': '#DC3545',  # Red
        'warning': '#FFC107',  # Amber
        'info': '#17A2B8',  # Cyan
        'light': '#F8F9FA',  # Light Gray
        'dark': '#343A40',  # Dark Gray
        'match': '#28A745',  # Green for match
        'no_match': '#DC3545',  # Red for no match
    }

    # Grid layout
    DEFAULT_GRID_COLS = 3
    MIN_GRID_COLS = 2
    MAX_GRID_COLS = 5

    # ==================== LANGUAGE SETTINGS ====================
    LANGUAGES = {
        'en': {'name': 'English', 'flag': '🇬🇧', 'dir': 'ltr'},
        'ar': {'name': 'العربية', 'flag': '🇸🇦', 'dir': 'rtl'}
    }
    DEFAULT_LANGUAGE = 'en'

    # ==================== TELEGRAM FEEDBACK ====================
    # NOTE: Set these in .streamlit/secrets.toml for security
    # Format:
    # TELEGRAM_BOT_TOKEN = "your_bot_token_here"
    # TELEGRAM_CHAT_ID = "your_chat_id_here"

    FEEDBACK_CATEGORIES = {
        'en': ['Bug Report', 'Feature Request', 'Suggestion', 'General', 'Other'],
        'ar': ['تقرير خطأ', 'طلب ميزة', 'اقتراح', 'عام', 'أخرى']
    }

    # ==================== EXPORT SETTINGS ====================
    EXPORT_FORMATS = ['PDF', 'CSV', 'JSON']
    PDF_DPI = 300

    # ==================== QUALITY THRESHOLDS ====================
    QUALITY_SCORES = {
        'blur_threshold': 100,  # Laplacian variance
        'brightness_min': 50,  # Min brightness
        'brightness_max': 200,  # Max brightness
        'min_face_size': 50,  # Min face dimension
    }

    # Star rating
    STAR_RATINGS = [1, 2, 3, 4, 5]

    # ==================== DEMO MODE ====================
    DEMO_SCENARIOS = {
        'en': [
            'Same Person - Different Angles',
            'Same Person - Different Lighting',
            'Similar People - Not Same',
            'Different People',
            'Low Quality vs High Quality'
        ],
        'ar': [
            'نفس الشخص - زوايا مختلفة',
            'نفس الشخص - إضاءة مختلفة',
            'أشخاص متشابهون - ليسوا نفس الشخص',
            'أشخاص مختلفون',
            'جودة منخفضة مقابل جودة عالية'
        ]
    }

    # ==================== SIMILARITY DISPLAY ====================
    SIMILARITY_RANGES = {
        'very_high': (0.85, 1.0, 'Very High Match', 'تطابق عالي جداً'),
        'high': (0.7, 0.85, 'High Match', 'تطابق عالي'),
        'medium': (0.5, 0.7, 'Medium Match', 'تطابق متوسط'),
        'low': (0.3, 0.5, 'Low Match', 'تطابق منخفض'),
        'very_low': (0.0, 0.3, 'No Match', 'لا يوجد تطابق')
    }

    # ==================== PERFORMANCE METRICS ====================
    # From your model evaluation
    MODEL_METRICS = {
        'accuracy': 0.905,  # 90.5%
        'precision': 0.89,
        'recall': 0.93,
        'f1_score': 0.91,
        'auc': 0.945
    }

    # ==================== CLUSTERING SETTINGS ====================
    CLUSTERING_METHOD = 'hierarchical'  # or 'kmeans'
    MIN_CLUSTERS = 2
    MAX_CLUSTERS = 10
    SIMILARITY_THRESHOLD_CLUSTER = 0.75

    # ==================== CACHING ====================
    CACHE_TTL = 3600  # 1 hour in seconds

    # ==================== PRIVACY ====================
    AUTO_DELETE_UPLOADS = True
    UPLOAD_RETENTION_HOURS = 1

    # ==================== MISC ====================
    SHOW_PROCESSING_TIME = True
    ANIMATION_DURATION = 0.3  # seconds

    @classmethod
    def create_directories(cls):
        """Create necessary directories if they don't exist"""
        for directory in [cls.MODELS_DIR, cls.ASSETS_DIR, cls.DEMO_DIR, cls.TEMP_DIR]:
            directory.mkdir(parents=True, exist_ok=True)

    @classmethod
    def get_color(cls, key, alpha=None):
        """Get color with optional alpha channel"""
        color = cls.COLORS.get(key, cls.COLORS['primary'])
        if alpha is not None:
            # Convert hex to rgba
            r = int(color[1:3], 16)
            g = int(color[3:5], 16)
            b = int(color[5:7], 16)
            return f'rgba({r}, {g}, {b}, {alpha})'
        return color

    @classmethod
    def get_similarity_level(cls, similarity, lang='en'):
        """Get similarity level description based on score"""
        for level, (min_val, max_val, desc_en, desc_ar) in cls.SIMILARITY_RANGES.items():
            if min_val <= similarity < max_val:
                return desc_ar if lang == 'ar' else desc_en
        return 'Unknown' if lang == 'en' else 'غير معروف'

    @classmethod
    def get_quality_stars(cls, quality_score):
        """Convert quality score (0-100) to star rating (1-5)"""
        if quality_score >= 90:
            return 5
        elif quality_score >= 75:
            return 4
        elif quality_score >= 60:
            return 3
        elif quality_score >= 40:
            return 2
        else:
            return 1


# Create directories on import
Config.create_directories()
