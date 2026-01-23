"""
FaceMatch Pro - Heatmap Utilities
Attention visualization and GradCAM for face recognition
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import streamlit as st
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))
from config import Config


class AttentionExtractor:
    """Extract attention maps from the model"""

    def __init__(self, model):
        self.model = model
        self.activations = []
        self.gradients = []
        self.hooks = []

    def register_hooks(self):
        """Register forward and backward hooks"""

        def forward_hook(module, input, output):
            self.activations.append(output.detach())

        def backward_hook(module, grad_input, grad_output):
            self.gradients.append(grad_output[0].detach())

        # Hook into the last convolutional layer
        # For ResNet50, this is layer4
        if hasattr(self.model, 'features'):
            # Get the last conv layer in features
            target_layer = None
            for name, module in self.model.features.named_modules():
                if isinstance(module, torch.nn.Conv2d):
                    target_layer = module

            if target_layer is not None:
                self.hooks.append(target_layer.register_forward_hook(forward_hook))
                self.hooks.append(target_layer.register_full_backward_hook(backward_hook))

    def remove_hooks(self):
        """Remove all hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def __del__(self):
        """Cleanup"""
        self.remove_hooks()


def generate_gradcam_heatmap(model, image_tensor, target_class=None):
    """
    Generate GradCAM heatmap for face recognition

    Args:
        model: Face recognition model
        image_tensor: torch tensor (1, 3, H, W)
        target_class: target class index (None for embedding-based)

    Returns:
        heatmap: numpy array (H, W) with values in [0, 1]
    """
    model.eval()

    # Create attention extractor
    extractor = AttentionExtractor(model)
    extractor.register_hooks()

    # Requires gradient
    image_tensor.requires_grad = True

    try:
        # Forward pass
        output = model(image_tensor)

        # For face recognition, we don't have classes
        # Use the norm of the embedding as target
        if target_class is None:
            target = output.norm()
        else:
            target = output[0, target_class]

        # Backward pass
        model.zero_grad()
        target.backward()

        # Get activations and gradients
        if len(extractor.activations) == 0 or len(extractor.gradients) == 0:
            return None

        activations = extractor.activations[-1]
        gradients = extractor.gradients[-1]

        # Global average pooling of gradients
        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)

        # Weighted combination of activation maps
        cam = torch.sum(weights * activations, dim=1, keepdim=True)

        # ReLU
        cam = F.relu(cam)

        # Normalize
        cam = cam.squeeze().cpu().numpy()
        cam = cam - np.min(cam)
        if np.max(cam) != 0:
            cam = cam / np.max(cam)

        return cam

    finally:
        extractor.remove_hooks()


def generate_attention_heatmap(model, image_tensor):
    """
    Generate attention heatmap from SE blocks

    Args:
        model: Face recognition model with attention
        image_tensor: torch tensor (1, 3, H, W)

    Returns:
        heatmap: numpy array (H, W) with values in [0, 1]
    """
    if not hasattr(model, 'attention'):
        return None

    model.eval()

    attention_weights = []

    def hook_fn(module, input, output):
        # SE block outputs channel attention weights
        attention_weights.append(output.detach())

    # Register hook on SE block
    hook = model.attention.register_forward_hook(hook_fn)

    try:
        with torch.no_grad():
            _ = model(image_tensor)

        if len(attention_weights) == 0:
            return None

        # Get attention map
        attention = attention_weights[0]

        # Average across channels to get spatial attention
        if len(attention.shape) == 4:
            # (B, C, H, W)
            spatial_attention = torch.mean(attention, dim=1, keepdim=True)
        else:
            # Need to reshape
            b, c = attention.shape[:2]
            # For SE block, output is usually (B, C, 1, 1)
            # We need to expand it to spatial dimensions
            spatial_attention = attention.view(b, c, 1, 1)
            spatial_attention = torch.mean(spatial_attention, dim=1, keepdim=True)

        # Convert to numpy
        heatmap = spatial_attention.squeeze().cpu().numpy()

        # Normalize
        heatmap = heatmap - np.min(heatmap)
        if np.max(heatmap) != 0:
            heatmap = heatmap / np.max(heatmap)

        return heatmap

    finally:
        hook.remove()


def create_heatmap_overlay(image, heatmap, alpha=0.5, colormap=cv2.COLORMAP_JET):
    """
    Create heatmap overlay on image

    Args:
        image: numpy array (H, W, 3) in RGB
        heatmap: numpy array (H_h, W_h) with values in [0, 1]
        alpha: overlay transparency
        colormap: OpenCV colormap

    Returns:
        overlay: RGB image with heatmap overlay
    """
    # Resize heatmap to match image size
    if heatmap.shape != image.shape[:2]:
        heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))

    # Apply Gaussian blur for smoothness
    heatmap = cv2.GaussianBlur(heatmap, (Config.HEATMAP_BLUR_SIZE, Config.HEATMAP_BLUR_SIZE), 0)

    # Convert heatmap to uint8
    heatmap_uint8 = (heatmap * 255).astype(np.uint8)

    # Apply colormap
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, colormap)

    # Convert BGR to RGB
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    # Blend with original image
    overlay = cv2.addWeighted(image, 1 - alpha, heatmap_colored, alpha, 0)

    return overlay


def visualize_attention(model, image, image_tensor, method='gradcam'):
    """
    Complete attention visualization pipeline

    Args:
        model: Face recognition model
        image: numpy array (H, W, 3) in RGB (original image)
        image_tensor: torch tensor (1, 3, H, W) (preprocessed)
        method: 'gradcam' or 'attention'

    Returns:
        dict with visualization results
    """
    # Generate heatmap
    if method == 'gradcam':
        heatmap = generate_gradcam_heatmap(model, image_tensor)
    elif method == 'attention':
        heatmap = generate_attention_heatmap(model, image_tensor)
    else:
        return None

    if heatmap is None:
        return None

    # Create overlay
    overlay = create_heatmap_overlay(image, heatmap, alpha=Config.HEATMAP_ALPHA)

    # Create heatmap visualization (colored heatmap without image)
    heatmap_colored = cv2.applyColorMap(
        (heatmap * 255).astype(np.uint8),
        cv2.COLORMAP_JET
    )
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    return {
        'heatmap': heatmap,
        'overlay': overlay,
        'heatmap_colored': heatmap_colored,
        'original': image
    }


def create_multi_view_visualization(original, heatmap_colored, overlay):
    """
    Create 3-panel visualization: Original | Heatmap | Overlay

    Args:
        original: original image
        heatmap_colored: colored heatmap
        overlay: overlay image

    Returns:
        combined: 3-panel visualization
    """
    # Make all images same size
    h, w = original.shape[:2]

    if heatmap_colored.shape[:2] != (h, w):
        heatmap_colored = cv2.resize(heatmap_colored, (w, h))

    if overlay.shape[:2] != (h, w):
        overlay = cv2.resize(overlay, (w, h))

    # Add labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    label_height = 40

    # Create label backgrounds
    original_labeled = np.vstack([
        np.ones((label_height, w, 3), dtype=np.uint8) * 255,
        original
    ])
    heatmap_labeled = np.vstack([
        np.ones((label_height, w, 3), dtype=np.uint8) * 255,
        heatmap_colored
    ])
    overlay_labeled = np.vstack([
        np.ones((label_height, w, 3), dtype=np.uint8) * 255,
        overlay
    ])

    # Add text
    cv2.putText(original_labeled, "Original", (10, 25), font, 0.7, (0, 0, 0), 2)
    cv2.putText(heatmap_labeled, "Attention Map", (10, 25), font, 0.7, (0, 0, 0), 2)
    cv2.putText(overlay_labeled, "Overlay", (10, 25), font, 0.7, (0, 0, 0), 2)

    # Add gaps
    gap = np.ones((h + label_height, 20, 3), dtype=np.uint8) * 255

    # Concatenate
    combined = np.hstack([original_labeled, gap, heatmap_labeled, gap, overlay_labeled])

    return combined


def analyze_attention_regions(heatmap, image_shape, top_k=5):
    """
    Analyze which regions have highest attention

    Args:
        heatmap: numpy array (H, W)
        image_shape: tuple (H, W, C)
        top_k: number of top regions to identify

    Returns:
        list of dicts with region info
    """
    # Resize heatmap if needed
    if heatmap.shape != image_shape[:2]:
        heatmap = cv2.resize(heatmap, (image_shape[1], image_shape[0]))

    # Threshold to get high-attention regions
    threshold = np.percentile(heatmap, 80)
    mask = (heatmap >= threshold).astype(np.uint8)

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Analyze each contour
    regions = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 100:  # Skip tiny regions
            continue

        x, y, w, h = cv2.boundingRect(contour)
        center_x, center_y = x + w // 2, y + h // 2

        # Get average attention in this region
        region_mask = np.zeros_like(heatmap)
        cv2.drawContours(region_mask, [contour], 0, 1, -1)
        avg_attention = np.mean(heatmap[region_mask == 1])

        regions.append({
            'bbox': (x, y, w, h),
            'center': (center_x, center_y),
            'area': area,
            'avg_attention': avg_attention
        })

    # Sort by attention
    regions.sort(key=lambda x: x['avg_attention'], reverse=True)

    return regions[:top_k]


def get_attention_statistics(heatmap):
    """
    Get statistical analysis of attention map

    Args:
        heatmap: numpy array (H, W)

    Returns:
        dict with statistics
    """
    return {
        'mean': float(np.mean(heatmap)),
        'std': float(np.std(heatmap)),
        'min': float(np.min(heatmap)),
        'max': float(np.max(heatmap)),
        'median': float(np.median(heatmap)),
        'q25': float(np.percentile(heatmap, 25)),
        'q75': float(np.percentile(heatmap, 75)),
        'concentration': float(np.std(heatmap))  # Higher = more concentrated
    }


def create_attention_legend():
    """
    Create a legend for attention heatmap

    Returns:
        legend: numpy array with color legend
    """
    # Create gradient bar
    gradient = np.linspace(0, 255, 256).astype(np.uint8)
    gradient = np.tile(gradient, (40, 1))

    # Apply colormap
    colored_gradient = cv2.applyColorMap(gradient, cv2.COLORMAP_JET)
    colored_gradient = cv2.cvtColor(colored_gradient, cv2.COLOR_BGR2RGB)

    # Add labels
    legend = np.ones((80, 256, 3), dtype=np.uint8) * 255
    legend[:40, :] = colored_gradient

    # Add text
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(legend, "Low", (5, 65), font, 0.5, (0, 0, 0), 1)
    cv2.putText(legend, "High", (220, 65), font, 0.5, (0, 0, 0), 1)
    cv2.putText(legend, "Attention Level", (70, 65), font, 0.5, (0, 0, 0), 1)

    return legend


@st.cache_data
def precompute_heatmap(image, _model, image_tensor, method='gradcam'):
    """
    Cached heatmap computation
    Note: _model has underscore to exclude from cache hash

    Args:
        image: numpy array
        _model: model (excluded from cache)
        image_tensor: input tensor
        method: heatmap method

    Returns:
        visualization results
    """
    return visualize_attention(_model, image, image_tensor, method)