"""
Central Configuration Module
All hyperparameters, paths, and settings in one place
"""

import torch
from pathlib import Path


class Config:
    """Centralized configuration class"""

    # ==================== PATHS ====================
    # Dataset paths (Kaggle)
    TRAIN_DIR = "/kaggle/input/vggface2/train"
    VAL_DIR = "/kaggle/input/vggface2/val"

    # Output paths
    SAVED_FILES_DIR = "files"
    DATABASE_PATH = "face_recognition.db"
    CHECKPOINT_PREFIX = "checkpoint_epoch"
    BEST_MODEL_NAME = "model.pth"
    FINAL_MODEL_NAME = "model.pth"

    # ==================== MODEL ARCHITECTURE ====================
    EMBEDDING_DIM = 512
    DROPOUT_RATE = 0.3

    # ==================== DATASET CONFIGURATION ====================
    # Number of images to load per identity
    SAMPLES_PER_IDENTITY_TRAIN = 30  # 20 # 30
    SAMPLES_PER_IDENTITY_VAL = 15  # 10 # 10

    # Number of triplets to generate per identity per epoch
    TRIPLETS_PER_IDENTITY = 30  # 10 # 20

    # Face cropping configuration
    FACE_CROP_MODE = "accurate"  # "accurate"  # Options: "fast", "accurate"
    FACE_CROP_MARGIN = 0.2  # Add 20% margin around detected face
    FACE_MIN_CONFIDENCE = 0.9  # Minimum detection confidence

    # ==================== DATA AUGMENTATION ====================
    # Image preprocessing
    IMAGE_SIZE = 224

    # Training augmentation parameters
    AUGMENTATION = {
        'horizontal_flip_prob': 0.5,
        'rotation_degrees': 10,
        'brightness': 0.2,
        'contrast': 0.2,
        'saturation': 0.2,
        'hue': 0.1
    }

    # Normalization (ImageNet stats)
    NORMALIZE_MEAN = [0.485, 0.456, 0.406]
    NORMALIZE_STD = [0.229, 0.224, 0.225]

    # ==================== TRAINING HYPERPARAMETERS ====================
    BATCH_SIZE = 64  # 32
    NUM_EPOCHS = 50
    LEARNING_RATE = 0.0001
    WEIGHT_DECAY = 1e-5

    # Loss function
    TRIPLET_MARGIN = 0.5

    # Learning rate scheduler
    LR_SCHEDULER_FACTOR = 0.5
    LR_SCHEDULER_PATIENCE = 3
    LR_SCHEDULER_MIN_LR = 1e-7

    # Gradient clipping
    GRADIENT_CLIP_VALUE = 1.0

    # Mixed precision training
    USE_MIXED_PRECISION = True

    # Early stopping
    EARLY_STOPPING_PATIENCE = 7
    EARLY_STOPPING_MIN_DELTA = 1e-4

    # ==================== DATA LOADING ====================
    NUM_WORKERS = 0  # Set to 0 for Kaggle to avoid worker crashes
    PIN_MEMORY = True
    PREFETCH_FACTOR = None  # Only used when NUM_WORKERS > 0

    # ==================== CHECKPOINTING ====================
    SAVE_INTERVAL = 5  # Save checkpoint every N epochs
    SAVE_TOP_K = 3  # Keep only top K best models

    # ==================== INFERENCE ====================
    RECOGNITION_THRESHOLD = 0.6  # Cosine similarity threshold

    # ==================== GUI CONFIGURATION ====================
    # Preview sizes
    PREVIEW_IMAGE_SIZE = (200, 200)  # Thumbnail size for image previews

    # Duplicate detection
    DUPLICATE_THRESHOLD = 0.85  # Similarity threshold for duplicate detection
    MAX_DUPLICATES_SHOWN = 3  # Maximum number of similar faces to show

    # UI Colors
    COLOR_SUCCESS = "green"
    COLOR_SUCCESS_HOVER = "darkgreen"
    COLOR_ERROR = "red"
    COLOR_WARNING = "orange"
    COLOR_INFO = "gray"
    COLOR_INFO_HOVER = "darkgray"
    COLOR_SUCCESS_TEXT = "green"
    COLOR_ERROR_TEXT = "red"
    COLOR_WARNING_TEXT = "orange"
    COLOR_INFO_TEXT = "gray"

    # UI Spacing
    PADDING_LARGE = 20
    PADDING_MEDIUM = 15
    PADDING_SMALL = 10
    PADDING_TINY = 5

    # UI Element Sizes
    BUTTON_HEIGHT_LARGE = 50
    BUTTON_HEIGHT_MEDIUM = 40
    BUTTON_WIDTH_FORM = 400
    BUTTON_WIDTH_ACTION = 250
    BUTTON_WIDTH_SECONDARY = 150

    TEXTBOX_HEIGHT_NOTES = 80

    # Font Sizes
    FONT_SIZE_TITLE = 28
    FONT_SIZE_LARGE = 20
    FONT_SIZE_SECTION = 18
    FONT_SIZE_LABEL = 14
    FONT_SIZE_BUTTON = 16
    FONT_SIZE_SMALL = 12
    FONT_SIZE_TINY = 10

    # Toast notification settings
    TOAST_DURATION_SUCCESS = 3000  # ms
    TOAST_DURATION_ERROR = 5000  # ms
    TOAST_DURATION_WARNING = 4000  # ms
    TOAST_DURATION_INFO = 3000  # ms

    # History Panel Configuration
    HISTORY_OLD_RECORDS_DAYS = 90  # Days threshold for "old" records
    HISTORY_CONFIDENCE_HIGH = 0.7  # High confidence threshold (green)
    HISTORY_CONFIDENCE_MEDIUM = 0.5  # Medium confidence threshold (orange)

    # Row colors for history display
    HISTORY_ROW_RECOGNIZED_LIGHT = "gray85"
    HISTORY_ROW_RECOGNIZED_DARK = "gray20"
    HISTORY_ROW_UNKNOWN_LIGHT = "gray90"
    HISTORY_ROW_UNKNOWN_DARK = "gray15"

    # ==================== DEVICE ====================
    # Safe device detection with fallback
    if torch.cuda.is_available():
        try:
            # Test CUDA is actually working
            torch.cuda.init()
            test_tensor = torch.zeros(1).cuda()
            del test_tensor
            torch.cuda.empty_cache()
            DEVICE = torch.device('cuda')
            print(f"✅ CUDA device available: {torch.cuda.get_device_name(0)}")
        except Exception as e:
            print(f"⚠ CUDA available but failed to initialize: {e}")
            print("  Falling back to CPU")
            DEVICE = torch.device('cpu')
    else:
        DEVICE = torch.device('cpu')
        print("ℹ No CUDA device available, using CPU")

    # ==================== RANDOM SEEDS ====================
    RANDOM_SEED = 42

    # ==================== VALIDATION METRICS ====================
    VALIDATION_PAIRS = 1000  # Number of pairs to evaluate

    @classmethod
    def print_config(cls):
        """Print current configuration"""
        print("=" * 80)
        print("CONFIGURATION SUMMARY")
        print("=" * 80)
        print(f"Device: {cls.DEVICE}")
        print(f"Batch Size: {cls.BATCH_SIZE}")
        print(f"Learning Rate: {cls.LEARNING_RATE}")
        print(f"Epochs: {cls.NUM_EPOCHS}")
        print(f"Embedding Dim: {cls.EMBEDDING_DIM}")
        print(f"Triplet Margin: {cls.TRIPLET_MARGIN}")
        print(f"Mixed Precision: {cls.USE_MIXED_PRECISION}")
        print(f"Early Stopping Patience: {cls.EARLY_STOPPING_PATIENCE}")
        print(f"Face Crop Mode: {cls.FACE_CROP_MODE}")
        print(f"Samples per Identity (Train): {cls.SAMPLES_PER_IDENTITY_TRAIN}")
        print(f"Triplets per Identity: {cls.TRIPLETS_PER_IDENTITY}")
        print("=" * 80)