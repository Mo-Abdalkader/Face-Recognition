"""
Inference Script
Performs face recognition using trained model
"""

import torch
from pathlib import Path

print("\n" + "=" * 80)
print("FACE RECOGNITION INFERENCE")
print("=" * 80)

# Create global config instance
from config import Config

CFG = Config()

# Paths
MODEL_PATH = f"{CFG.SAVED_FILES_DIR}/{CFG.BEST_MODEL_NAME}"
DB_PATH = f"{CFG.SAVED_FILES_DIR}/{CFG.DATABASE_PATH}"

# Check if model exists
if not Path(MODEL_PATH).exists():
    print(f"Error: Model not found at {MODEL_PATH}")
    print("Please train the model first!")
else:
    print(f"✓ Model found: {MODEL_PATH}")

    # Load model
    print("\nLoading model...")
    from models.hybrid_encoder import HybridFaceEncoder  # commented for Kaggle

    model = HybridFaceEncoder(
        embedding_dim=CFG.EMBEDDING_DIM,
        dropout=CFG.DROPOUT_RATE
    )

    checkpoint = torch.load(MODEL_PATH, map_location=CFG.DEVICE, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(CFG.DEVICE)
    model.eval()

    print(f"✓ Model loaded successfully")
    print(f"  Epoch: {checkpoint['epoch'] + 1}")
    print(f"  Embedding dim: {checkpoint['config']['embedding_dim']}")

    # Initialize face cropper
    from data.face_cropper import FaceCropper  # commented for Kaggle

    face_cropper = FaceCropper(
        mode=CFG.FACE_CROP_MODE,
        margin=CFG.FACE_CROP_MARGIN
    )

    # Create transform
    from data.transforms import DataTransforms  # commented for Kaggle

    transform = DataTransforms.get_val_transforms(CFG)

    # Initialize embedding generator
    from inference.embedding_generator import FaceEmbeddingGenerator  # commented for Kaggle

    embedding_generator = FaceEmbeddingGenerator(
        model=model,
        transform=transform,
        face_cropper=face_cropper,
        device=CFG.DEVICE
    )

    # Initialize database (if doesn't exist, create it)
    from database.face_database import FaceDatabase  # commented for Kaggle

    if not Path(DB_PATH).exists():
        print(f"\nCreating new database: {DB_PATH}")
        db = FaceDatabase(DB_PATH)
        db.close()

    # Initialize face recognizer
    from inference.face_recognizer import FaceRecognizer  # commented for Kaggle

    recognizer = FaceRecognizer(
        db_path=DB_PATH,
        embedding_generator=embedding_generator,
        threshold=CFG.RECOGNITION_THRESHOLD
    )

    print("\n" + "=" * 80)
    print("INFERENCE SYSTEM READY")
    print("=" * 80)
    print("\nYou can now:")
    print("1. Add faces to database:")
    print("   db = FaceDatabase(DB_PATH)")
    print("   db.add_person(face_id='EMP001', name='John Doe')")
    print("   embedding = embedding_generator.generate_embedding('john.jpg')")
    print("   db.add_embedding('EMP001', embedding, 'john.jpg')")
    print("   recognizer.reload_database()")
    print()
    print("2. Recognize faces:")
    print("   result = recognizer.recognize_face('test_image.jpg')")
    print("   if result['recognized']:")
    print("       print(f\"Identified: {result['person_info']['name']}\")")
    print("       print(f\"Confidence: {result['confidence']:.2%}\")")
    print("=" * 80)
