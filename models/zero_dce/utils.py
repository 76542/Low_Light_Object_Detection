import torch
import torch.nn as nn

# -------------------------------
# 1. Spatial Consistency Loss
# -------------------------------
# Keeps the enhanced image spatially similar to the original in terms of structure/edges
def spatial_consistency_loss(enhanced, original):
    def gradient(x):
        h_gradient = x[:, :, 1:, :] - x[:, :, :-1, :]  # Horizontal gradient
        w_gradient = x[:, :, :, 1:] - x[:, :, :, :-1]  # Vertical gradient
        return h_gradient, w_gradient

    enhanced_h, enhanced_w = gradient(enhanced)
    original_h, original_w = gradient(original)

    loss_h = torch.mean(torch.abs(enhanced_h - original_h))
    loss_w = torch.mean(torch.abs(enhanced_w - original_w))

    return loss_h + loss_w

# -------------------------------
# 2. Exposure Control Loss
# -------------------------------
# Encourages brightness of the image to be close to a target mean (e.g., 0.6)
def exposure_loss(enhanced, patch_size=16, mean_val=0.6):
    pool = nn.AvgPool2d(patch_size)  # Downsample into patches
    mean = pool(enhanced)           # Average brightness of each patch
    loss = torch.mean((mean - mean_val) ** 2)  # MSE with target brightness
    return loss

# -------------------------------
# 3. Color Constancy Loss
# -------------------------------
# Encourages R ≈ G ≈ B to avoid unnatural color tones
def color_constancy_loss(image):
    r, g, b = image[:, 0, :, :], image[:, 1, :, :], image[:, 2, :, :]
    loss = torch.mean((r - g) ** 2) + torch.mean((r - b) ** 2) + torch.mean((g - b) ** 2)
    return loss

# -------------------------------
# 4. Illumination Smoothness Loss
# -------------------------------
# Penalizes sudden changes in the enhancement curve maps to make them smooth
def illumination_smoothness_loss(curves):
    loss = 0
    for r in curves:
        h_grad = torch.abs(r[:, :, 1:, :] - r[:, :, :-1, :])  # Horizontal
        w_grad = torch.abs(r[:, :, :, 1:] - r[:, :, :, :-1])  # Vertical
        loss += torch.mean(h_grad) + torch.mean(w_grad)
    return loss
