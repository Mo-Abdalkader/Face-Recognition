"""
Training Script
Trains face recognition model on VGGFace2 dataset
"""

import random
import numpy as np
import torch
from torch.utils.data import DataLoader

# Create global config instance
from config import Config

CFG = Config()

# Set random seeds
random.seed(CFG.RANDOM_SEED)
np.random.seed(CFG.RANDOM_SEED)
torch.manual_seed(CFG.RANDOM_SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(CFG.RANDOM_SEED)

# Print configuration
CFG.print_config()

# Initialize face cropper
# from data.face_cropper import FaceCropper  # commented for Kaggle
# face_cropper = FaceCropper(
#     mode=CFG.FACE_CROP_MODE,
#     margin=CFG.FACE_CROP_MARGIN,
#     min_confidence=CFG.FACE_MIN_CONFIDENCE
# )
face_cropper = None

# Create transforms
from data.transforms import DataTransforms  # commented for Kaggle
train_transform = DataTransforms.get_train_transforms(CFG)
val_transform = DataTransforms.get_val_transforms(CFG)

# Create datasets
print("\n" + "=" * 80)
print("LOADING DATASETS")
print("=" * 80)

from data.dataset import VGGFace2TripletDataset  # commented for Kaggle
train_dataset = VGGFace2TripletDataset(
    root_dir=CFG.TRAIN_DIR,
    transform=train_transform,
    samples_per_identity=CFG.SAMPLES_PER_IDENTITY_TRAIN,
    triplets_per_identity=CFG.TRIPLETS_PER_IDENTITY,
    face_cropper=face_cropper,
    cache_crops=True  # Set True if enough RAM
)

val_dataset = VGGFace2TripletDataset(
    root_dir=CFG.VAL_DIR,
    transform=val_transform,
    samples_per_identity=CFG.SAMPLES_PER_IDENTITY_VAL,
    triplets_per_identity=CFG.TRIPLETS_PER_IDENTITY // 2,
    face_cropper=face_cropper,
    cache_crops=True
)

# Create data loaders with Kaggle-safe settings
train_loader = DataLoader(
    train_dataset,
    batch_size=CFG.BATCH_SIZE,
    shuffle=True,
    num_workers=0,  # CRITICAL: Must be 0 in Kaggle to prevent worker crashes
    pin_memory=CFG.PIN_MEMORY if torch.cuda.is_available() else False,
    drop_last=True  # Avoid incomplete batches
)

val_loader = DataLoader(
    val_dataset,
    batch_size=CFG.BATCH_SIZE,
    shuffle=False,
    num_workers=0,  # CRITICAL: Must be 0 in Kaggle
    pin_memory=CFG.PIN_MEMORY if torch.cuda.is_available() else False,
    drop_last=False
)

print(f"✓ Train batches: {len(train_loader)}")
print(f"✓ Val batches: {len(val_loader)}")

# Initialize model
print("\n" + "=" * 80)
print("INITIALIZING MODEL")
print("=" * 80)

from models.hybrid_encoder import HybridFaceEncoder  # commented for Kaggle
model = HybridFaceEncoder(
    embedding_dim=CFG.EMBEDDING_DIM,
    dropout=CFG.DROPOUT_RATE
).to(CFG.DEVICE)

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")

# Initialize loss function
from training.losses import TripletLoss  # commented for Kaggle
criterion = TripletLoss(margin=CFG.TRIPLET_MARGIN)

# Initialize early stopping
from training.early_stopping import EarlyStopping  # commented for Kaggle
early_stopping = EarlyStopping(
    patience=CFG.EARLY_STOPPING_PATIENCE,
    min_delta=CFG.EARLY_STOPPING_MIN_DELTA,
    verbose=True
)

# Initialize trainer
from training.trainer import FaceRecognitionTrainer  # commented for Kaggle

trainer = FaceRecognitionTrainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    criterion=criterion,
    config=CFG,
    early_stopping=early_stopping,
    # use_multi_gpu = True # If TPU False
)

# Start training
trainer.train(save_dir=CFG.SAVED_FILES_DIR)

print("\n✓ Training completed successfully!")
print(f"✓ Best model saved: {CFG.SAVED_FILES_DIR}/{CFG.BEST_MODEL_NAME}")
