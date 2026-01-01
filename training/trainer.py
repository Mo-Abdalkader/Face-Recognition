"""
Face Recognition Training Pipeline
Includes: Mixed precision, gradient clipping, early stopping, comprehensive metrics
"""

import os
import time
from pathlib import Path
from typing import Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import matplotlib.pyplot as plt


class FaceRecognitionTrainer:
    """
    Complete training pipeline with production features
    """

    def __init__(
            self,
            model: nn.Module,
            train_loader: DataLoader,
            val_loader: DataLoader,
            criterion: nn.Module,
            config,
            early_stopping: Optional[object] = None
    ):
        """
        Args:
            model: Neural network model
            train_loader: Training data loader
            val_loader: Validation data loader
            criterion: Loss function
            config: Configuration object
            early_stopping: EarlyStopping instance (optional)
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.config = config
        self.early_stopping = early_stopping

        # Optimizer
        self.optimizer = Adam(
            model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY
        )

        # Learning rate scheduler
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=config.LR_SCHEDULER_FACTOR,
            patience=config.LR_SCHEDULER_PATIENCE,
            min_lr=config.LR_SCHEDULER_MIN_LR,
            # verbose=True # HIIII
        )

        # Mixed precision scaler
        self.scaler = GradScaler() if config.USE_MIXED_PRECISION else None

        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_accuracy': [],
            'val_precision': [],
            'val_recall': [],
            'learning_rates': [],
            'epoch_times': []
        }

        self.best_val_loss = float('inf')
        self.best_epoch = 0

    def train_epoch(self, epoch: int) -> float:
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch + 1} [TRAIN]")

        for batch_idx, (anchor, positive, negative) in enumerate(pbar):
            # Move to device
            anchor = anchor.to(self.config.DEVICE)
            positive = positive.to(self.config.DEVICE)
            negative = negative.to(self.config.DEVICE)

            # Zero gradients
            self.optimizer.zero_grad()

            # Mixed precision forward pass
            if self.config.USE_MIXED_PRECISION:
                with autocast():
                    anchor_embed = self.model(anchor)
                    positive_embed = self.model(positive)
                    negative_embed = self.model(negative)
                    loss = self.criterion(anchor_embed, positive_embed, negative_embed)

                # Backward pass with scaling
                self.scaler.scale(loss).backward()

                # Gradient clipping
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.GRADIENT_CLIP_VALUE
                )

                # Optimizer step
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # Standard training
                anchor_embed = self.model(anchor)
                positive_embed = self.model(positive)
                negative_embed = self.model(negative)
                loss = self.criterion(anchor_embed, positive_embed, negative_embed)

                loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.GRADIENT_CLIP_VALUE
                )

                self.optimizer.step()

            # Update metrics
            running_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        epoch_loss = running_loss / len(self.train_loader)
        return epoch_loss

    def validate_epoch(self, epoch: int) -> tuple:
        """Validate for one epoch with metrics"""
        self.model.eval()
        running_loss = 0.0

        # Collect embeddings for metrics
        all_anchors = []
        all_positives = []
        all_negatives = []

        pbar = tqdm(self.val_loader, desc=f"Epoch {epoch + 1} [VAL]  ")

        with torch.no_grad():
            for anchor, positive, negative in pbar:
                anchor = anchor.to(self.config.DEVICE)
                positive = positive.to(self.config.DEVICE)
                negative = negative.to(self.config.DEVICE)

                # Forward pass
                anchor_embed = self.model(anchor)
                positive_embed = self.model(positive)
                negative_embed = self.model(negative)

                # Calculate loss
                loss = self.criterion(anchor_embed, positive_embed, negative_embed)
                running_loss += loss.item()

                # Store embeddings for metrics
                all_anchors.append(anchor_embed)
                all_positives.append(positive_embed)
                all_negatives.append(negative_embed)

                pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        epoch_loss = running_loss / len(self.val_loader)

        # Calculate metrics
        all_anchors = torch.cat(all_anchors, dim=0)
        all_positives = torch.cat(all_positives, dim=0)
        all_negatives = torch.cat(all_negatives, dim=0)

        from training.metrics import FaceRecognitionMetrics  # commented for Kaggle
        metrics = FaceRecognitionMetrics.calculate_metrics(
            all_anchors,
            all_positives,
            all_negatives,
            threshold=self.config.RECOGNITION_THRESHOLD
        )

        return epoch_loss, metrics

    def train(self, save_dir: str = 'face_models'):
        """
        Complete training loop

        Args:
            save_dir: Directory to save checkpoints
        """
        print("\n" + "=" * 80)
        print("STARTING TRAINING")
        print("=" * 80)
        self.config.print_config()

        # Create save directory
        os.makedirs(save_dir, exist_ok=True)

        for epoch in range(self.config.NUM_EPOCHS):
            epoch_start_time = time.time()

            # Train
            train_loss = self.train_epoch(epoch)

            # Validate
            val_loss, metrics = self.validate_epoch(epoch)

            # Update learning rate
            self.scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]['lr']

            # Calculate epoch time
            epoch_time = time.time() - epoch_start_time

            # Save history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['val_accuracy'].append(metrics['accuracy'])
            self.history['val_precision'].append(metrics['precision'])
            self.history['val_recall'].append(metrics['recall'])
            self.history['learning_rates'].append(current_lr)
            self.history['epoch_times'].append(epoch_time)

            # Print epoch summary
            print(f"\nEpoch {epoch + 1}/{self.config.NUM_EPOCHS} [{epoch_time:.1f}s]")
            print(f"  Train Loss:  {train_loss:.4f}")
            print(f"  Val Loss:    {val_loss:.4f}")
            print(f"  Accuracy:    {metrics['accuracy']:.4f}")
            print(f"  Precision:   {metrics['precision']:.4f}")
            print(f"  Recall:      {metrics['recall']:.4f}")
            print(f"  Pos Sim:     {metrics['avg_positive_similarity']:.4f}")
            print(f"  Neg Sim:     {metrics['avg_negative_similarity']:.4f}")
            print(f"  LR:          {current_lr:.6f}")

            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_epoch = epoch
                self.save_checkpoint(
                    os.path.join(save_dir, self.config.BEST_MODEL_NAME),
                    epoch,
                    is_best=True
                )
                print(f"  ✓ Best model saved! (Val Loss: {val_loss:.4f})")

            # Save checkpoint
            if (epoch + 1) % self.config.SAVE_INTERVAL == 0:
                checkpoint_path = os.path.join(
                    save_dir,
                    f"{self.config.CHECKPOINT_PREFIX}_{epoch + 1}.pth"
                )
                self.save_checkpoint(checkpoint_path, epoch)
                print(f"  ✓ Checkpoint saved: {checkpoint_path}")

            # Early stopping check
            if self.early_stopping is not None:
                if self.early_stopping(val_loss, epoch):
                    print(f"\nTraining stopped early at epoch {epoch + 1}")
                    break

            print("-" * 80)

        # Save final model
        final_path = os.path.join(save_dir, self.config.FINAL_MODEL_NAME)
        self.save_checkpoint(final_path, epoch, is_final=True)

        print("\n" + "=" * 80)
        print("TRAINING COMPLETED!")
        print("=" * 80)
        print(f"Best validation loss: {self.best_val_loss:.4f} at epoch {self.best_epoch + 1}")
        print(f"Models saved in: {save_dir}/")
        print("=" * 80)

        # Plot history
        self.plot_history(save_dir)

    def save_checkpoint(
            self,
            path: str,
            epoch: int,
            is_best: bool = False,
            is_final: bool = False
    ):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'history': self.history,
            'config': {
                'embedding_dim': self.config.EMBEDDING_DIM,
                'triplet_margin': self.config.TRIPLET_MARGIN,
                'image_size': self.config.IMAGE_SIZE
            },
            'is_best': is_best,
            'is_final': is_final
        }

        if self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()

        torch.save(checkpoint, path)

    def plot_history(self, save_dir: str):
        """Plot training history"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        epochs = range(1, len(self.history['train_loss']) + 1)

        # Loss curves
        axes[0, 0].plot(epochs, self.history['train_loss'], 'b-', label='Train Loss', linewidth=2)
        axes[0, 0].plot(epochs, self.history['val_loss'], 'r-', label='Val Loss', linewidth=2)
        axes[0, 0].set_xlabel('Epoch', fontsize=12)
        axes[0, 0].set_ylabel('Triplet Loss', fontsize=12)
        axes[0, 0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
        axes[0, 0].legend()
        axes[0, 0].grid(alpha=0.3)

        # Metrics
        axes[0, 1].plot(epochs, self.history['val_accuracy'], 'g-', label='Accuracy', linewidth=2)
        axes[0, 1].plot(epochs, self.history['val_precision'], 'm-', label='Precision', linewidth=2)
        axes[0, 1].plot(epochs, self.history['val_recall'], 'c-', label='Recall', linewidth=2)
        axes[0, 1].set_xlabel('Epoch', fontsize=12)
        axes[0, 1].set_ylabel('Score', fontsize=12)
        axes[0, 1].set_title('Validation Metrics', fontsize=14, fontweight='bold')
        axes[0, 1].legend()
        axes[0, 1].grid(alpha=0.3)

        # Learning rate
        axes[1, 0].plot(epochs, self.history['learning_rates'], 'orange', linewidth=2)
        axes[1, 0].set_xlabel('Epoch', fontsize=12)
        axes[1, 0].set_ylabel('Learning Rate', fontsize=12)
        axes[1, 0].set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
        axes[1, 0].set_yscale('log')
        axes[1, 0].grid(alpha=0.3)

        # Epoch times
        axes[1, 1].plot(epochs, self.history['epoch_times'], 'purple', linewidth=2)
        axes[1, 1].set_xlabel('Epoch', fontsize=12)
        axes[1, 1].set_ylabel('Time (seconds)', fontsize=12)
        axes[1, 1].set_title('Epoch Training Time', fontsize=14, fontweight='bold')
        axes[1, 1].grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'training_history.png'), dpi=300, bbox_inches='tight')
        plt.show()

        print(f"✓ Training history plot saved: {save_dir}/training_history.png")