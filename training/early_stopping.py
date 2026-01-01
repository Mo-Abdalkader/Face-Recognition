"""
Early Stopping Callback
Stops training when validation loss stops improving
"""

import numpy as np


class EarlyStopping:
    """
    Early stopping to prevent overfitting
    Stops training when validation loss doesn't improve for patience epochs
    """

    def __init__(self, patience: int = 7, min_delta: float = 1e-4, verbose: bool = True):
        """
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
            verbose: Print messages
        """
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose

        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_epoch = 0

    def __call__(self, val_loss: float, epoch: int) -> bool:
        """
        Check if training should stop

        Args:
            val_loss: Current validation loss
            epoch: Current epoch number

        Returns:
            True if training should stop, False otherwise
        """
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_epoch = epoch
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter}/{self.patience}")

            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print(f"\nEarly stopping triggered!")
                    print(f"Best validation loss: {self.best_loss:.6f} at epoch {self.best_epoch}")
                return True
        else:
            self.best_loss = val_loss
            self.best_epoch = epoch
            self.counter = 0

        return False